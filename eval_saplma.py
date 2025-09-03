# eval_saplma.py
# ------------------------------------------------------------
# Evaluate a pretrained SAPLMA bundle on a dataset CSV created by your
# llama_llmRunMultiLayers.py script (i.e., CSV has an 'embeddings' column).
# - Auto-reads the bundle's layer_from_end and embedding_dim
# - Parses your 'embeddings' serialization (list or "array(..., dtype=float32)")
# - Computes accuracy@bundle-threshold, accuracy@0.5, ROC-AUC, confusion matrices
# - Saves predictions.csv + metrics.json into <bundle>/eval_<name>/
# ------------------------------------------------------------

import argparse
import json
from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

from guarded_infer.saplma_api import load_best_bundle, predict_embeddings

# ---------------- helpers ----------------

def parse_embeddings_column(series: pd.Series) -> np.ndarray:
    """
    Handles both:
      "[array(0.1, 0.2, ..., dtype=float32)]"  and  "[[0.1, 0.2, ...]]"
    emitted by various pipelines.
    """
    def norm(s: str) -> str:
        return (
            s.replace("[array(", "")
             .replace("dtype=float32)]", "")
             .replace("\n", "")
             .replace(" ", "")
             .replace("],", "]")
             .replace("[", "")
             .replace("]", "")
        )
    return np.array([np.fromstring(norm(x), sep=',') for x in series.tolist()], dtype=np.float32)

def coerce_labels(series: Optional[pd.Series]) -> Optional[np.ndarray]:
    if series is None or series.name is None:
        return None
    s = series
    if s is None or s.isna().all():
        return None
    if s.dtype.kind in "biuf":
        uniq = set(pd.unique(s.dropna()))
        if not uniq.issubset({0, 1, 0.0, 1.0}):
            raise ValueError(f"Numeric labels must be 0/1. Got {sorted(uniq)}")
        return s.astype(int).to_numpy()
    mapping = {
        "true": 1, "truth": 1, "correct": 1, "yes": 1, "y": 1, "1": 1,
        "false": 0, "lie": 0, "incorrect": 0, "no": 0, "n": 0, "0": 0,
    }
    def map_one(v):
        if pd.isna(v): return np.nan
        k = str(v).strip().lower()
        if k in mapping: return mapping[k]
        raise ValueError(f"Unrecognized label '{v}'. Use 0/1 or true/false/lie/yes/no.")
    out = s.map(map_one)
    if out.isna().any():
        raise ValueError("Found NaN after label mapping.")
    return out.astype(int).to_numpy()

def build_csv_path_from_config(dataset_name: str, layer_from_end: int) -> Path:
    """
    Reconstruct your file name:
      DATASET_FOLDER + f"/embeddings_with_labels_{dataset}{MODEL_NAME}_{abs(layer)}_rmv_period.csv"
    """
    from instruct_saplma.config import DATASET_FOLDER, MODEL_NAME, REMOVE_PERIOD
    suffix = "_rmv_period" if REMOVE_PERIOD else ""
    fname = f"embeddings_with_labels_{dataset_name}{MODEL_NAME}_{abs(layer_from_end)}{suffix}.csv"
    return Path(DATASET_FOLDER) / fname

# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser(description="Evaluate a pretrained SAPLMA on your created embeddings CSV.")
    ap.add_argument("--bundle", required=True, help="Path to bundle dir (contains .keras, threshold.txt, meta.json)")
    ap.add_argument("--csv", default=None, help="Path to embeddings CSV. If omitted, use --dataset-name with config pattern.")
    ap.add_argument("--dataset-name", default=None, help="Base dataset name used when creating the CSV (e.g., 'capitals').")
    ap.add_argument("--label-col", default="label", help="Label column name if present.")
    ap.add_argument("--out-dir", default=None, help="Where to save outputs (default: <bundle>/eval_<name>/)")
    args = ap.parse_args()

    # Load bundle
    bundle_dir = Path(args.bundle)
    model, thr, meta, _ = load_best_bundle(bundle=bundle_dir)
    layer_from_end = int(meta["layer_from_end"])
    expected_dim   = int(meta["embedding_dim"])

    # Resolve CSV path
    if args.csv:
        csv_path = Path(args.csv)
        dataset_name_for_out = csv_path.stem
    else:
        if not args.dataset_name:
            raise SystemExit("Provide either --csv path OR --dataset-name to auto-resolve via config.py")
        csv_path = build_csv_path_from_config(args.dataset_name, layer_from_end)
        dataset_name_for_out = args.dataset_name
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    # Load CSV
    df = pd.read_csv(csv_path)
    if "embeddings" not in df.columns:
        raise SystemExit("CSV must contain an 'embeddings' column produced by your creation script.")
    X = parse_embeddings_column(df["embeddings"])
    if X.shape[1] != expected_dim:
        raise SystemExit(
            f"Embedding dim {X.shape[1]} != bundle expected_dim {expected_dim}. "
            "Make sure you used the same base model + same layer when creating the CSV."
        )

    # Predict with bundle
    pred = predict_embeddings(model, thr, X)
    prob_true = np.asarray(pred["prob_true"]).ravel()
    pred_best = np.asarray(pred["pred_best"], dtype=str)  # 'True' | 'Lie'
    pred_0p5  = np.asarray(pred["pred_0p5"],  dtype=str)

    # Metrics (if labels present)
    metrics = {}
    y = df[args.label_col] if args.label_col in df.columns else None
    y_true = coerce_labels(y) if y is not None else None
    if y_true is not None:
        map_pred = {"true": 1, "lie": 0}
        yhat_best = np.array([map_pred[str(v).lower()] for v in pred_best])
        yhat_05   = np.array([map_pred[str(v).lower()] for v in pred_0p5])

        metrics["accuracy_at_bundle_thr"] = float(accuracy_score(y_true, yhat_best))
        metrics["accuracy_at_0p5"]        = float(accuracy_score(y_true, yhat_05))

        uniq = set(np.unique(y_true))
        if uniq.issuperset({0,1}):
            try:
                metrics["roc_auc"] = float(roc_auc_score(y_true, prob_true))
            except Exception:
                metrics["roc_auc"] = float("nan")
        else:
            metrics["roc_auc"] = float("nan")

        cm_best = confusion_matrix(y_true, yhat_best, labels=[0,1]).tolist()
        cm_05   = confusion_matrix(y_true, yhat_05,   labels=[0,1]).tolist()
        metrics["confusion_matrix_at_bundle_thr"] = cm_best
        metrics["confusion_matrix_at_0p5"]        = cm_05
    else:
        metrics["note"] = "No labels column found; wrote predictions only."

    # Outputs
    out_dir = Path(args.out_dir) if args.out_dir else (bundle_dir / f"eval_{dataset_name_for_out}")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_pred = df.copy()
    out_pred["prob_true"] = prob_true
    out_pred["pred_best"] = pred_best
    out_pred["pred_0p5"]  = pred_0p5
    out_pred.to_csv(out_dir / "predictions.csv", index=False)

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[OK] predictions -> {out_dir/'predictions.csv'}")
    print(f"[OK] metrics     -> {out_dir/'metrics.json'}")
    if "accuracy_at_bundle_thr" in metrics:
        print(f"Acc@bundle_thr: {metrics['accuracy_at_bundle_thr']:.4f}")
    if "accuracy_at_0p5" in metrics:
        print(f"Acc@0.5:        {metrics['accuracy_at_0p5']:.4f}")
    if "roc_auc" in metrics:
        print(f"AUC:            {metrics['roc_auc']}")
        
if __name__ == "__main__":
    main()



    """
    python eval_saplma.py \
    --bundle pretrained_saplma/instruct/format_4/saplma_checkpoints_LLAMA7/BEST_layer12__heldout_data/capitals \
    --csv /home/ddn1/Documents/GitHub/SAPLMA/data/capital_true_false_instruct/format4/embeddings_with_labels_data/capitalsLLAMA7_12_rmv_period.csv \
    --label-col label \
    --out-dir saplma_tests_results/format_4_model/format_4_test_dataset_results
    
    """