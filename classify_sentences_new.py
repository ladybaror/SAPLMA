# classify_sentences_new.py
# ------------------------------------------------------------
# Train a small MLP on hidden-state embeddings and evaluate with
# leave-one-dataset-out. Produce a table (columns = held-out datasets,
# values = accuracy@optimal-threshold). Save the best classifier bundle.
# ------------------------------------------------------------

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__name__), "..")))

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, accuracy_score

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# ---- Config (from instruct_saplma.config) ----
from instruct_saplma.config import (
    # paths / naming
    DATASET_FOLDER,
    MODEL_NAME,
    LIST_OF_DATASETS,
    LAYERS_TO_USE,
    REMOVE_PERIOD,
    # training switches
    REPEAT_EACH,
    CHECK_UNCOMMON,
    CHECK_GENERATED,
    KEEP_PROBABILITIES,
    # extra knobs you added
    SEED,
    EPOCHS,
    BATCH_SIZE,
    DEV_SPLIT,
    USE_EARLY_STOPPING,
    EARLY_STOP_PATIENCE,
    TEST_VAL_SPLIT,
    ALLOW_SINGLE_DATASET_FALLBACK,
    OUTPUT_DIR,
    SAVE_GLOBAL_BEST,
    SUMMARY_TABLE_PATH,
    # filenames/templates
    KERAS_MODEL_FILENAME,
    THRESHOLD_FILENAME,
    META_FILENAME,
)

# Optional template; fall back to default pattern if missing
try:
    from instruct_saplma.config import EMBEDDINGS_FILE_TEMPLATE  # noqa: F401
    HAS_TEMPLATE = True
except Exception:
    HAS_TEMPLATE = False

# -----------------------------
# Utilities
# -----------------------------

def set_seeds(seed: int = 42):
    np.random.seed(seed)
    tf.random.set_seed(seed)

def correct_str(str_arr: str) -> str:
    """
    Normalize the serialized 'embeddings' string in CSVs to a comma
    separated numeric list so np.fromstring(..., sep=',') can parse it.
    """
    return (
        str_arr
        .replace("[array(", "")
        .replace("dtype=float32)]", "")
        .replace("\n", "")
        .replace(" ", "")
        .replace("],", "]")
        .replace("[", "")
        .replace("]", "")
    )

def parse_embeddings_column(series: pd.Series) -> np.ndarray:
    """Convert a Series[str] of serialized vectors into a 2D float32 array."""
    return np.array([np.fromstring(correct_str(e), sep=',') for e in series.tolist()], dtype=np.float32)

def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute ROC-AUC; return NaN if y_true has a single class."""
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return np.nan
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return auc(fpr, tpr)

def choose_optimal_threshold(y_val: np.ndarray, y_val_prob: np.ndarray) -> float:
    """
    Choose the probability threshold maximizing accuracy on a validation split.
    Fall back to 0.5 if y_val is single-class or thresholds are empty.
    """
    y_val = np.asarray(y_val)
    if len(np.unique(y_val)) < 2:
        return 0.5
    thresholds = roc_curve(y_val, y_val_prob)[2]
    if thresholds.size == 0:
        return 0.5
    idx = np.argmax([accuracy_score(y_val, (y_val_prob > thr).astype(int)) for thr in thresholds])
    return float(thresholds[idx])

def save_bundle(model: tf.keras.Model, threshold: float, meta: dict, out_dir: Path):
    """
    Save a classifier bundle:
      - model (single-file .keras)
      - threshold.txt
      - meta.json
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(out_dir / KERAS_MODEL_FILENAME)
    (out_dir / THRESHOLD_FILENAME).write_text(str(threshold))
    (out_dir / META_FILENAME).write_text(json.dumps(meta, indent=2))

def build_csv_path(dataset_name: str, layer_from_end: int) -> str:
    """
    Build the embeddings CSV path for a given dataset name and layer index.
    Supports optional EMBEDDINGS_FILE_TEMPLATE; otherwise uses default pattern.
    Default pattern:
      embeddings_with_labels_{name}{MODEL_NAME}_{abs(layer)}_rmv_period.csv
    """
    layer_abs = abs(layer_from_end)
    if HAS_TEMPLATE:
        from instruct_saplma.config import EMBEDDINGS_FILE_TEMPLATE
        filename = EMBEDDINGS_FILE_TEMPLATE.format(name=dataset_name, layer=layer_abs)
    else:
        suffix = "_rmv_period" if REMOVE_PERIOD else ""
        filename = f"embeddings_with_labels_{dataset_name}{MODEL_NAME}_{layer_abs}{suffix}.csv"
    return os.path.join(DATASET_FOLDER, filename)

# -----------------------------
# Main
# -----------------------------

def main():
    set_seeds(SEED)

    dataset_names_order = LIST_OF_DATASETS[:]  # stable column order
    table_rows: List[Dict[str, float]] = []    # rows for the final table

    # Track best classifier across ALL layers/datasets/repeats
    best_global = {
        "acc": -1.0,
        "dir": None,
        "meta": None,
    }

    for layer_num_from_end in LAYERS_TO_USE:
        # ---------- Load CSVs for this layer ----------
        datasets: List[pd.DataFrame] = []
        for name in LIST_OF_DATASETS:
            path = build_csv_path(name, layer_num_from_end)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing CSV for dataset '{name}': {path}")
            df = pd.read_csv(path)
            if df.empty:
                raise ValueError(f"Loaded dataset '{name}' is empty: {path}")
            if "embeddings" not in df.columns or "label" not in df.columns:
                raise ValueError(f"CSV '{path}' must contain 'embeddings' and 'label' columns.")
            datasets.append(df)

        # How many held-out runs to do
        dataset_loop_length = 1 if (CHECK_UNCOMMON or CHECK_GENERATED) else len(datasets)

        # For this layer, keep per-held-out-dataset accuracy@threshold
        layer_table = {name: np.nan for name in LIST_OF_DATASETS}

        # For logging detailed repeats
        results: List[Tuple[str, int, float, float, float, float]] = []  # (test_name, rep, keras_acc, AUC, thr, acc@thr)

        # ---------- Leave-one-dataset-out loop ----------
        for ds in range(dataset_loop_length):
            # Decide test dataset index
            test_idx = 0 if (CHECK_UNCOMMON or CHECK_GENERATED) else ds
            test_df = datasets[test_idx]

            # Build training pool from all OTHER datasets
            train_parts = [df for i, df in enumerate(datasets) if i != test_idx]

            if len(train_parts) == 0:
                if not ALLOW_SINGLE_DATASET_FALLBACK:
                    raise ValueError(
                        "Only one dataset provided and fallback is disabled. "
                        "Provide ≥2 datasets or set ALLOW_SINGLE_DATASET_FALLBACK=True."
                    )
                print("[WARN] Only one dataset provided. Falling back to stratified row split.")
                tr_df, te_df = train_test_split(
                    test_df,
                    test_size=0.3,
                    random_state=SEED,
                    stratify=test_df["label"] if "label" in test_df.columns else None,
                )
                train_df = tr_df.reset_index(drop=True)
                test_df = te_df.reset_index(drop=True)
            else:
                train_df = pd.concat(train_parts, ignore_index=True)

            # Optionally accumulate averaged probabilities on the final test set
            all_probs = np.zeros((len(test_df), 1), dtype=np.float32)

            # ---------- Repeat training ----------
            for i in range(REPEAT_EACH):
                # Parse embeddings
                train_embeddings = parse_embeddings_column(train_df["embeddings"])
                test_embeddings  = parse_embeddings_column(test_df["embeddings"])
                train_labels = train_df["label"].to_numpy()
                test_labels  = test_df["label"].to_numpy()

                # Dev split from training pool for early stopping/ckpt
                X_tr, X_dev, y_tr, y_dev = train_test_split(
                    train_embeddings, train_labels,
                    test_size=DEV_SPLIT,
                    stratify=train_labels,
                    random_state=SEED
                )

                # Define a simple MLP classifier
                model = Sequential([
                    Dense(256, activation='relu', input_dim=train_embeddings.shape[1]),
                    Dense(128, activation='relu'),
                    Dense(64, activation='relu'),
                    Dense(1, activation='sigmoid'),
                ])
                model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

                # Per-run checkpoint path
                ckpt_dir  = OUTPUT_DIR / f"tmp_ckpt_layer{abs(layer_num_from_end)}_heldout_{LIST_OF_DATASETS[test_idx]}_rep{i}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                ckpt_path = ckpt_dir / "weights.keras"

                callbacks = []
                if USE_EARLY_STOPPING:
                    callbacks = [
                        ModelCheckpoint(str(ckpt_path), monitor="val_loss", save_best_only=True),
                        EarlyStopping(monitor="val_loss", patience=EARLY_STOP_PATIENCE, restore_best_weights=True),
                    ]

                # Train
                model.fit(
                    X_tr, y_tr,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    validation_data=(X_dev, y_dev),
                    callbacks=callbacks,
                    verbose=1
                )

                # Ensure best weights are loaded if we saved a checkpoint
                if ckpt_path.exists():
                    model.load_weights(str(ckpt_path))

                # Keras accuracy (threshold=0.5) on full test set
                _, keras_acc = model.evaluate(test_embeddings, test_labels, verbose=0)

                # Probabilities on test set
                test_pred_prob = model.predict(test_embeddings, verbose=0)
                if KEEP_PROBABILITIES:
                    all_probs += test_pred_prob

                # AUC (safe)
                roc_auc = safe_auc(test_labels, test_pred_prob)

                # Pick threshold using a validation split FROM the test set
                X_val, X_test, y_val, y_test = train_test_split(
                    test_embeddings, test_labels,
                    test_size=TEST_VAL_SPLIT, random_state=SEED
                )
                y_val_pred_prob  = model.predict(X_val, verbose=0).ravel()
                y_test_pred_prob = model.predict(X_test, verbose=0).ravel()

                optimal_threshold = choose_optimal_threshold(y_val, y_val_pred_prob)

                y_test_pred = (y_test_pred_prob > optimal_threshold).astype(int)
                test_accuracy = accuracy_score(y_test, y_test_pred)

                test_name = LIST_OF_DATASETS[test_idx]
                results.append((test_name, i, float(keras_acc), float(roc_auc) if not np.isnan(roc_auc) else np.nan,
                                float(optimal_threshold), float(test_accuracy)))

                print(
                    f"[INFO] layer={layer_num_from_end} test={test_name} "
                    f"rep={i} AUC={roc_auc if not np.isnan(roc_auc) else float('nan'):.4f} "
                    f"opt_thr={optimal_threshold:.4f} acc@thr={test_accuracy:.4f}"
                )

                # ---- Save the best global bundle across all runs ----
                if SAVE_GLOBAL_BEST and test_accuracy > best_global["acc"]:
                    bundle_dir = OUTPUT_DIR / f"BEST_layer{abs(layer_num_from_end)}__heldout_{test_name}"
                    meta = {
                        "model_name": MODEL_NAME,
                        "layer_from_end": int(layer_num_from_end),
                        "heldout_dataset": test_name,
                        "train_datasets": [n for n in LIST_OF_DATASETS if n != test_name],
                        "embedding_dim": int(train_embeddings.shape[1]),
                        "repeat_index": int(i),
                        "metric": "accuracy@optimal_threshold",
                        "metric_value": float(test_accuracy),
                        "optimal_threshold": float(optimal_threshold),
                    }
                    save_bundle(model, float(optimal_threshold), meta, bundle_dir)
                    best_global.update(acc=float(test_accuracy), dir=str(bundle_dir), meta=meta)

            if KEEP_PROBABILITIES:
                all_probs /= REPEAT_EACH
                print("-- Averaged probabilities (first 10) --")
                print(all_probs[:10].ravel())

            # Aggregate across repeats for THIS held-out dataset
            start = ds * REPEAT_EACH
            end   = (ds + 1) * REPEAT_EACH
            accs         = [t[2] for t in results[start:end]]
            aucs         = [t[3] for t in results[start:end]]
            opt_threshes = [t[4] for t in results[start:end]]
            acc_thr      = [t[5] for t in results[start:end]]

            test_name = LIST_OF_DATASETS[test_idx]
            mean_acc_thr = float(np.nanmean(acc_thr))

            print(
                f"dataset: {test_name} "
                f"layer: {layer_num_from_end} "
                f"Avg_acc: {np.nanmean(accs):.4f} "
                f"Avg_AUC: {np.nanmean(aucs):.4f} "
                f"Avg_threshold: {np.nanmean(opt_threshes):.4f} "
                f"Avg_thrs_acc: {mean_acc_thr:.4f}"
            )

            # Store for the table (column = held-out dataset)
            layer_table[test_name] = mean_acc_thr

        # ---------- Build a row for this layer ----------
        layer_label = "last-layer" if abs(layer_num_from_end) == 1 else f"last-{abs(layer_num_from_end)}"
        row = {"Model": layer_label}
        for name in dataset_names_order:
            row[name] = layer_table.get(name, np.nan)
        row["Average"] = np.nanmean([row[name] for name in dataset_names_order])
        table_rows.append(row)

    # ---------- Final table & overall averages ----------
    table_df = pd.DataFrame(table_rows, columns=["Model"] + dataset_names_order + ["Average"])

    print("\n================= Leave-one-dataset-out accuracy@optimal-threshold =================")
    fmt = {col: (lambda x: f"{x:.4f}") for col in dataset_names_order + ["Average"]}
    print(table_df.to_string(index=False, justify="center", col_space=9, formatters=fmt))

    # Per-dataset averages across layers (column means), and global overall average
    per_dataset_means = table_df[dataset_names_order].mean(skipna=True)
    overall_avg_all = float(per_dataset_means.mean())

    print("\nPer-dataset average across layers:")
    for name, val in per_dataset_means.items():
        print(f"{name:>12}: {val:.4f}")

    print(f"\nOverall average across all datasets & layers: {overall_avg_all:.4f}")

    # Save summary table
    SUMMARY_TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    table_df.to_csv(SUMMARY_TABLE_PATH, index=False)
    print(f"\nSaved summary table to: {SUMMARY_TABLE_PATH}")

    # Report best bundle
    if SAVE_GLOBAL_BEST and best_global["dir"]:
        print(f"\n[BEST] Saved best SAPLMA classifier bundle to: {best_global['dir']}")
        print(f"[BEST] Acc@thr = {best_global['acc']:.4f}")
        print(f"[BEST] Meta    = {json.dumps(best_global['meta'], indent=2)}")

    # Optional: LaTeX one-liner
    # print(table_df.to_latex(index=False, float_format='%.4f', column_format='l' + 'c'*(len(dataset_names_order)+1)))

if __name__ == "__main__":
    main()





# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__name__), "..")))

# # classify_sentences_new.py
# """
# Train a simple feed-forward classifier on token-level hidden-state embeddings
# (exported earlier) to predict binary labels per row.

# Pipeline (for each layer in LAYERS_TO_USE):
# 1) Load CSVs of embeddings + labels for each dataset name in LIST_OF_DATASETS.
# 2) For each test dataset (leave-one-dataset-out):
#    - Train on the concatenation of the remaining datasets.
#    - Repeat REPEAT_EACH times:
#        * Fit a small MLP (Keras Sequential) for a few epochs.
#        * Evaluate ROC AUC and accuracy.
#        * Derive an optimal probability threshold using a validation split
#          from the test set (X_val / X_test), then report test accuracy
#          at that threshold.
#    - Optionally average predicted probabilities across repeats (KEEP_PROBABILITIES).
# 3) Print per-dataset averages for accuracy, AUC, threshold, and accuracy@threshold.

# Notes:
# - Embeddings are read from CSVs created by the previous step (e.g., llama_llmRunMultiLayers.py).
# - The 'embeddings' column is stored as a string; correct_str() normalizes it so we can parse via np.fromstring.
# - CHECK_UNCOMMON / CHECK_GENERATED let you fix the test set (dataset[0]) and train on the rest.
# """

# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import roc_curve, auc, accuracy_score
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense

# # Configuration:
# # MODEL_NAME: suffix used in filenames
# # LAYERS_TO_USE: list of layer indices from the end (e.g., [-1, -2, ...])
# # DATASET_FOLDER: path where CSVs live
# # REPEAT_EACH: number of repeated training runs per dataset
# # CHECK_UNCOMMON / CHECK_GENERATED: if True, always test on datasets[0] and train on the rest
# # KEEP_PROBABILITIES: if True, average probabilities across REPEAT_EACH runs for the (fixed) test set

# # from instruct_saplma.config import (
# from instruct_saplma.config import (
#     MODEL_NAME,
#     LAYERS_TO_USE,
#     DATASET_FOLDER,
#     REPEAT_EACH,
#     CHECK_UNCOMMON,
#     CHECK_GENERATED,
#     KEEP_PROBABILITIES,
#     LIST_OF_DATASETS
# )

# def correct_str(str_arr: str) -> str:
#     """
#     Normalize a serialized array string found in the 'embeddings' CSV column.

#     The CSV may store embeddings like:
#         "[array([0.1, 0.2, ...], dtype=float32)]"
#     or with extra spaces/newlines. We strip wrappers and whitespace so that
#     numpy.fromstring can parse it with sep=','.

#     Parameters
#     ----------
#     str_arr : str
#         Raw string from the 'embeddings' column.

#     Returns
#     -------
#     str
#         Comma-separated numeric string suitable for np.fromstring(..., sep=',').
#     """
#     val_to_ret = (
#         str_arr
#         .replace("[array(", "")
#         .replace("dtype=float32)]", "")
#         .replace("\n", "")
#         .replace(" ", "")
#         .replace("],", "]")
#         .replace("[", "")
#         .replace("]", "")
#     )
#     return val_to_ret


# # -----------------------------
# # Main loop over layers
# # -----------------------------
# for layer_num_from_end in LAYERS_TO_USE:
#     # Which datasets to use this run
#     dataset_names = LIST_OF_DATASETS

#     # Load all dataset CSVs for this layer. Each CSV must contain:
#     # - 'embeddings' column: serialized vector for each row
#     # - 'label' column: binary target {0,1}
#     datasets = [
#         pd.read_csv(
#             DATASET_FOLDER
#             + f"/embeddings_with_labels_{name}{MODEL_NAME}_{abs(layer_num_from_end)}_rmv_period.csv"
#         )
#         for name in dataset_names
#     ]

#     results = []
#     # If CHECK_UNCOMMON / CHECK_GENERATED is True, always test on datasets[0]; otherwise iterate all datasets
#     dataset_loop_length = 1 if CHECK_UNCOMMON or CHECK_GENERATED else len(dataset_names)

#     # -----------------------------
#     # Leave-one-dataset-out loop
#     # -----------------------------
#     for ds in range(dataset_loop_length):
#         # Select test and train splits at the dataset level
#         test_df = datasets[0] if (CHECK_UNCOMMON or CHECK_GENERATED) else datasets[ds]
#         train_df = (
#             pd.concat(datasets[1:], ignore_index=True)
#             if (CHECK_UNCOMMON or CHECK_GENERATED)
#             else pd.concat(datasets[:ds] + datasets[ds + 1:], ignore_index=True)
#         )

#         # Will hold (optional) averaged probs across REPEAT_EACH runs
#         all_probs = np.zeros((len(test_df), 1))

#         # -----------------------------
#         # Repeat training REPEAT_EACH times
#         # -----------------------------
#         for i in range(REPEAT_EACH):
#             # --- Parse embeddings/labels ---
#             # Convert serialized embedding strings to numeric numpy arrays
#             train_embeddings = np.array(
#                 [np.fromstring(correct_str(e), sep=',') for e in train_df['embeddings'].tolist()]
#             )
#             test_embeddings = np.array(
#                 [np.fromstring(correct_str(e), sep=',') for e in test_df['embeddings'].tolist()]
#             )

#             train_labels = np.array(train_df['label'])
#             test_labels = np.array(test_df['label'])

#             # --- Define a simple MLP classifier ---
#             model = Sequential()
#             model.add(Dense(256, activation='relu', input_dim=train_embeddings.shape[1]))
#             model.add(Dense(128, activation='relu'))
#             model.add(Dense(64, activation='relu'))
#             model.add(Dense(1, activation='sigmoid'))  # binary classification

#             model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#             # --- Train (few epochs for speed) ---
#             model.fit(
#                 train_embeddings, train_labels,
#                 epochs=5, batch_size=32,
#                 validation_data=(test_embeddings, test_labels),
#                 verbose=1
#             )

#             # --- Evaluate on full test set (Keras accuracy) ---
#             loss, accuracy = model.evaluate(test_embeddings, test_labels, verbose=0)

#             # --- Get predicted probabilities on test set ---
#             test_pred_prob = model.predict(test_embeddings, verbose=0)

#             # Optionally accumulate for averaging across repeats
#             if KEEP_PROBABILITIES:
#                 all_probs += test_pred_prob

#             # --- ROC & AUC ---
#             fpr, tpr, _ = roc_curve(test_labels, test_pred_prob)
#             roc_auc = auc(fpr, tpr)
#             print("AUC of the classifier:", roc_auc)

#             # --- Determine optimal threshold on a validation split from the test set ---
#             # We split the test set once more to pick a threshold that maximizes accuracy on 'val'
#             X_val, X_test, y_val, y_test = train_test_split(
#                 test_embeddings, test_labels, test_size=0.7, random_state=42
#             )

#             y_val_pred_prob = model.predict(X_val, verbose=0)
#             thresholds_val = roc_curve(y_val, y_val_pred_prob)[2]

#             # Choose the threshold that maximizes accuracy on the validation portion
#             optimal_threshold = thresholds_val[
#                 np.argmax([accuracy_score(y_val, (y_val_pred_prob > thr).astype(int)) for thr in thresholds_val])
#             ]

#             # Evaluate accuracy on the held-out portion of the test split using the chosen threshold
#             y_test_pred_prob = model.predict(X_test, verbose=0)
#             y_test_pred = (y_test_pred_prob > optimal_threshold).astype(int)
#             test_accuracy = accuracy_score(y_test, y_test_pred)

#             print("Optimal threshold:", optimal_threshold)
#             print("Test accuracy:", test_accuracy)

#             # Keep run-level metrics: (dataset_name, repeat_idx, keras_acc, AUC, opt_thr, acc@thr)
#             results.append((dataset_names[ds], i, accuracy, roc_auc, optimal_threshold, test_accuracy))

#         # If requested, average the probabilities across repeats and print them
#         if KEEP_PROBABILITIES:
#             all_probs /= REPEAT_EACH
#             print("-- Averaged probabilities --")
#             print(all_probs)

#         # -----------------------------
#         # Aggregate metrics across repeats for this dataset
#         # -----------------------------
#         accs = [t[2] for t in results[ds * REPEAT_EACH:(ds + 1) * REPEAT_EACH]]
#         aucs = [t[3] for t in results[ds * REPEAT_EACH:(ds + 1) * REPEAT_EACH]]
#         opt_thresh = [t[4] for t in results[ds * REPEAT_EACH:(ds + 1) * REPEAT_EACH]]
#         acc_thr_test = [t[5] for t in results[ds * REPEAT_EACH:(ds + 1) * REPEAT_EACH]]

#         print(
#             f"dataset: {dataset_names[ds]} "
#             f"layer: {layer_num_from_end} "
#             f"Avg_acc: {np.mean(accs):.4f} "
#             f"Avg_AUC: {np.mean(aucs):.4f} "
#             f"Avg_threshold: {np.mean(opt_thresh):.4f} "
#             f"Avg_thrs_acc: {np.mean(acc_thr_test):.4f}"
#         )

