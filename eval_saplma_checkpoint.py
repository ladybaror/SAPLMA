# pack_checkpoint_as_bundle.py
# Turn tmp_ckpt .../animals_rep0/weights.keras into a loadable SAPLMA bundle.

import json
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# --- helpers copied from your training/eval utilities ---
def parse_embeddings_column(series: pd.Series) -> np.ndarray:
    def norm(s: str) -> str:
        return (s.replace("[array(", "")
                 .replace("dtype=float32)]", "")
                 .replace("\n", "")
                 .replace(" ", "")
                 .replace("],", "]")
                 .replace("[", "")
                 .replace("]", ""))
    return np.array([np.fromstring(norm(x), sep=',') for x in series.tolist()], dtype=np.float32)

def choose_optimal_threshold(y_val: np.ndarray, y_val_prob: np.ndarray) -> float:
    from sklearn.metrics import roc_curve, accuracy_score
    y_val = np.asarray(y_val)
    if len(np.unique(y_val)) < 2:
        return 0.5
    thresholds = roc_curve(y_val, y_val_prob)[2]
    if thresholds.size == 0:
        return 0.5
    idx = np.argmax([accuracy_score(y_val, (y_val_prob > thr).astype(int)) for thr in thresholds])
    return float(thresholds[idx])

def build_mlp(embedding_dim: int):
    return Sequential([
        Dense(256, activation='relu', input_dim=embedding_dim),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid'),
    ])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help=".../tmp_ckpt_layer12_heldout_data/animals_rep0/weights.keras")
    ap.add_argument("--ref-bundle", required=True, help="Path to any existing bundle (to copy meta like embedding_dim, model_name, layer)")
    ap.add_argument("--out-bundle", required=True, help="Where to write the new bundle dir")
    ap.add_argument("--calib-csv", default=None, help="Optional embeddings CSV to calibrate threshold (has 'embeddings' and optional 'label')")
    ap.add_argument("--label-col", default="label")
    args = ap.parse_args()

    ref = Path(args.ref_bundle)
    out = Path(args.out_bundle)
    out.mkdir(parents=True, exist_ok=True)

    # 1) read reference meta to get embedding_dim, model_name, layer, etc.
    meta_ref = json.loads((ref / "meta.json").read_text())
    embedding_dim = int(meta_ref["embedding_dim"])
    layer_from_end = int(meta_ref["layer_from_end"])
    model_name = meta_ref.get("model_name", "unknown")

    # 2) build the same MLP and load weights
    model = build_mlp(embedding_dim)
    model.load_weights(args.weights)

    # 3) pick a threshold
    thr = 0.5
    if args.calib_csv:
        df = pd.read_csv(args.calib_csv)
        if "embeddings" not in df.columns:
            raise SystemExit("calib-csv must contain an 'embeddings' column.")
        X = parse_embeddings_column(df["embeddings"])
        if X.shape[1] != embedding_dim:
            raise SystemExit(f"Embedding dim {X.shape[1]} != expected {embedding_dim}")
        # predict probabilities with Keras
        prob = model.predict(X, verbose=0).ravel()
        if args.label_col in df.columns:
            y = df[args.label_col].to_numpy()
            # ensure 0/1 ints
            if df[args.label_col].dtype.kind not in "biu":
                y = np.array([1 if str(v).strip().lower() in {"true","truth","correct","yes","y","1"} else 0 for v in df[args.label_col]])
            thr = choose_optimal_threshold(y, prob)
        else:
            print("[WARN] calib-csv has no labels; keeping threshold=0.5")

    # 4) write bundle files
    model.save(out / "model.keras")
    (out / "threshold.txt").write_text(str(thr))
    meta = {
        "model_name": model_name,
        "layer_from_end": layer_from_end,
        "heldout_dataset": "animals",
        "train_datasets": [n for n in meta_ref.get("train_datasets", []) if n != "animals"],
        "embedding_dim": embedding_dim,
        "repeat_index": 0,
        "metric": "accuracy@optimal_threshold",
        "metric_value": None,  # unknown here
        "optimal_threshold": float(thr),
        "source_weights": str(Path(args.weights).resolve()),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[OK] Bundle written to: {out}")

if __name__ == "__main__":
    main()


"""
[animals, capitals, companies, elements, facts, inventions]


python eval_saplma_checkpoint.py \
  --weights pretrained_saplma/instruct/format_4/saplma_checkpoints_LLAMA7/tmp_ckpt_layer12_heldout_data/companies_rep9/weights.keras \
  --ref-bundle pretrained_saplma/instruct/format_4/saplma_checkpoints_LLAMA7/BEST_layer12__heldout_data/capitals \
  --out-bundle pretrained_saplma/instruct/format_4/saplma_checkpoints_LLAMA7/companies_rep9_bundle


# python eval_saplma_checkpoint.py \
#   --weights pretrained_saplma/instruct/format_2/saplma_checkpoints_LLAMA7/tmp_ckpt_layer12_heldout_data/animals_rep0/weights.keras \
#   --ref-bundle pretrained_saplma/instruct/format_2/saplma_checkpoints_LLAMA7/BEST_layer12__heldout_data/capitals \
#   --out-bundle saplma_tests_results/no_user_model/checkpoints/format_2_test_dataset_results/animals_rep0_bundle


"""