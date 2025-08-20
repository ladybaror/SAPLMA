# saplma_run.py
# ------------------------------------------------------------
# Swiss-army knife for SAPLMA (truthfulness; last-token embeddings only)
# Runtime-configurable: pass --config paper.config (or set SAPLMA_CONFIG).
# ------------------------------------------------------------

import os
import json
import ast
import importlib
from types import SimpleNamespace
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from tensorflow.keras.models import load_model

# ---- defaults; will be overridden at runtime from the selected config ----
OUTPUT_DIR = Path("./saplma_checkpoints_DEFAULT")
BASE_MODEL_PATH = "models/Llama-2-7b-chat-hf"
KERAS_MODEL_FILENAME = "model.keras"
THRESHOLD_FILENAME   = "threshold.txt"
META_FILENAME        = "meta.json"

# truth/lie labels
POS_LABEL_NAME = "True"
NEG_LABEL_NAME = "Lie"
PROB_NAME      = "prob_true"


# ------------------ runtime config loader ------------------

def _load_cfg(config_module: Optional[str]) -> SimpleNamespace:
    module_path = config_module or os.getenv("SAPLMA_CONFIG", "instruct_saplma.config")
    cfg = importlib.import_module(module_path)

    base_model_path = getattr(cfg, "BASE_MODEL_PATH", BASE_MODEL_PATH)
    keras_fname     = getattr(cfg, "KERAS_MODEL_FILENAME", KERAS_MODEL_FILENAME)
    thr_fname       = getattr(cfg, "THRESHOLD_FILENAME", THRESHOLD_FILENAME)
    meta_fname      = getattr(cfg, "META_FILENAME", META_FILENAME)

    output_dir = getattr(cfg, "OUTPUT_DIR", None)
    if output_dir is None:
        data_root = Path(getattr(cfg, "DATASET_FOLDER", "."))
        model_name = getattr(cfg, "MODEL_NAME", "MODEL")
        output_dir = data_root / f"saplma_checkpoints_{model_name}"
    return SimpleNamespace(
        OUTPUT_DIR=Path(output_dir),
        BASE_MODEL_PATH=base_model_path,
        KERAS_MODEL_FILENAME=keras_fname,
        THRESHOLD_FILENAME=thr_fname,
        META_FILENAME=meta_fname,
    )

# ------------------ bundle discovery / loading ------------------

def _bundle_ok(bundle_dir: Path) -> bool:
    return (
        (bundle_dir / KERAS_MODEL_FILENAME).exists() and
        (bundle_dir / THRESHOLD_FILENAME).exists() and
        (bundle_dir / META_FILENAME).exists()
    )

def _has_best_ancestor(path: Path, root: Path) -> bool:
    cur = path
    while cur != root and cur != cur.parent:
        if cur.name.startswith("BEST_"):
            return True
        cur = cur.parent
    return False

def discover_bundles(output_dir: Path) -> List[Path]:
    output_dir = Path(output_dir)
    bundles: List[Path] = []
    for meta_path in output_dir.rglob(META_FILENAME):
        if not _has_best_ancestor(meta_path.parent, output_dir):
            continue
        bd = meta_path.parent
        if _bundle_ok(bd):
            bundles.append(bd)
    return bundles

def find_best_bundle(output_dir: Path, prefer_substring: Optional[str]=None) -> Path:
    cands = discover_bundles(output_dir)
    scored = []
    for d in cands:
        try:
            meta = json.loads((d / META_FILENAME).read_text())
            mv = float(meta.get("metric_value", float("nan")))
            if not np.isnan(mv):
                scored.append((mv, d))
        except Exception:
            pass
    if not scored:
        raise FileNotFoundError(f"No BEST bundles with {META_FILENAME} found under {output_dir}")
    if prefer_substring:
        pref = [t for t in scored if prefer_substring in str(t[1])]
        if pref:
            scored = pref
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]

def load_bundle(bundle_dir: Path):
    bundle_dir = Path(bundle_dir)
    model = load_model(bundle_dir / KERAS_MODEL_FILENAME)
    thr   = float((bundle_dir / THRESHOLD_FILENAME).read_text())
    meta  = json.loads((bundle_dir / META_FILENAME).read_text())
    return model, thr, meta


# ------------------ embedding (LAST TOKEN ONLY) ------------------

def get_tokenizer_and_model(model_path: str, device: str = "auto"):
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    dtype = torch.float16 if torch.cuda.is_available() and device != "cpu" else torch.float32
    if device == "cpu":
        mdl = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, device_map=None)
        mdl.to("cpu")
    else:
        mdl = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, device_map="auto")
    mdl.eval()
    return tok, mdl

@torch.no_grad()
def embed_texts_last_token(
    texts: List[str],
    model_path: str,
    layer_from_end: int,
    device: str = "auto",
    max_length: Optional[int] = None,
) -> np.ndarray:
    """
    texts -> embeddings matrix (N, D) from the requested layer.
    Uses the last NON-PAD token (attention_mask-aware).
    """
    tok, mdl = get_tokenizer_and_model(model_path, device=device)
    dev = next(mdl.parameters()).device
    out_vecs: List[np.ndarray] = []

    for text in texts:
        enc = tok(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length if max_length else None,
        )
        enc = {k: v.to(dev) for k, v in enc.items()}
        out = mdl(**enc, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states  # tuple(len=num_layers+1)
        layer_states = hs[layer_from_end][0]  # [seq, hidden]

        attn = enc.get("attention_mask", torch.ones_like(enc["input_ids"]))[0]  # [seq]
        valid_tokens = int(attn.sum().item())
        last_idx = max(valid_tokens - 1, 0)

        emb = layer_states[last_idx]  # [hidden]
        out_vecs.append(emb.float().cpu().numpy())

    return np.stack(out_vecs, axis=0)  # [N, hidden]


# ------------------ misc helpers ------------------

def correct_str(raw: str) -> str:
    return (raw.replace("[array(", "")
               .replace("dtype=float32)]", "")
               .replace("\n", "")
               .replace(" ", "")
               .replace("],", "]")
               .replace("[", "")
               .replace("]", ""))

def parse_embeddings_column(series: pd.Series) -> np.ndarray:
    return np.array([np.fromstring(correct_str(s), sep=',') for s in series.tolist()], dtype=np.float32)

def predict_from_embeddings(clf, threshold_bundle: float, X: np.ndarray, extra_threshold: Optional[float]=None):
    probs = clf.predict(X, verbose=0).ravel()  # -> prob(True)
    pred_best  = (probs > threshold_bundle).astype(int)
    pred_05    = (probs > 0.5).astype(int)
    pred_custom = None
    if extra_threshold is not None:
        pred_custom = (probs > float(extra_threshold)).astype(int)
    return probs, pred_best, pred_05, pred_custom


# ------------------ CLI ------------------

def main():
    import argparse

    ap = argparse.ArgumentParser(description="Run SAPLMA (truthfulness; last-token embeddings) with runtime config.")
    ap.add_argument("--config", type=str, default=None,
                    help="Config module path, e.g. 'paper.config' or 'instruct_saplma.config'. "
                         "Falls back to env SAPLMA_CONFIG or 'instruct_saplma.config'.")

    # bundle selection
    ap.add_argument("--bundle", type=str, default=None, help="Path to a specific BEST bundle dir.")
    ap.add_argument("--prefer", type=str, default=None, help="Prefer bundle paths containing this substring.")
    ap.add_argument("--list-bundles", action="store_true", help="List discovered bundles and exit.")

    # inputs (mutually exclusive)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--text", type=str, help="Single sentence.")
    g.add_argument("--text-file", type=str, help="File with one sentence per line.")
    g.add_argument("--csv", type=str, help="CSV with either 'embeddings' or 'text' column.")
    g.add_argument("--npy", type=str, help="Path to .npy array of shape (N, D).")
    g.add_argument("--vec", type=str, help='Raw Python list/tuple string of one vector, e.g. "[0.1, 0.2, ...]".')

    # embedding options (only used when we need to embed text)
    ap.add_argument("--model-path", type=str, default=None, help="Override base model path for embeddings.")
    ap.add_argument("--layer", type=int, default=None, help="Layer from end for embeddings (default: bundle meta).")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto", help="Device for embedding model.")
    ap.add_argument("--max-length", type=int, default=None, help="Max tokens when embedding (truncate).")

    # thresholds / output
    ap.add_argument("--threshold", type=str, default="best",
                    help="'best' to use bundle optimal, '0.5', or a float e.g. 0.62.")
    ap.add_argument("--out", type=str, default=None, help="Save predictions to this CSV/JSON (by extension).")
    ap.add_argument("--print-meta", action="store_true", help="Print bundle meta before results.")
    args = ap.parse_args()

    # Load runtime config and override globals
    cfg = _load_cfg(args.config)
    global OUTPUT_DIR, BASE_MODEL_PATH, KERAS_MODEL_FILENAME, THRESHOLD_FILENAME, META_FILENAME
    OUTPUT_DIR = cfg.OUTPUT_DIR
    BASE_MODEL_PATH = cfg.BASE_MODEL_PATH if args.model_path is None else args.model_path
    KERAS_MODEL_FILENAME = cfg.KERAS_MODEL_FILENAME
    THRESHOLD_FILENAME   = cfg.THRESHOLD_FILENAME
    META_FILENAME        = cfg.META_FILENAME

    output_dir = Path(OUTPUT_DIR)

    # list only
    if args.list_bundles:
        found = discover_bundles(output_dir)
        if not found:
            print("No bundles found.")
            return
        for d in sorted(found):
            try:
                mv = json.loads((d / META_FILENAME).read_text()).get("metric_value")
            except Exception:
                mv = "?"
            print(f"{d}   metric_value={mv}")
        return

    # resolve bundle
    if args.bundle:
        bundle = Path(args.bundle)
        if not _bundle_ok(bundle):
            raise FileNotFoundError(f"--bundle is missing required files in {bundle}")
    else:
        bundle = find_best_bundle(output_dir, prefer_substring=args.prefer)

    clf, thr_opt, meta = load_bundle(bundle)
    if args.print_meta:
        print(json.dumps(meta, indent=2))

    # threshold selection
    thr_mode = args.threshold.strip().lower()
    custom_thr: Optional[float] = None
    if thr_mode == "best":
        thr_use = float(thr_opt)
    elif thr_mode == "0.5":
        thr_use = 0.5
    else:
        try:
            thr_use = float(thr_mode)
            custom_thr = thr_use
        except ValueError:
            raise ValueError("--threshold must be 'best', '0.5', or a float")

    # layer choice
    layer_from_end = int(args.layer) if args.layer is not None else int(meta["layer_from_end"])

    # ------------- build embeddings (LAST TOKEN) -------------
    X: Optional[np.ndarray] = None
    rows_payload: List[dict] = []  # for writing outputs nicely

    if args.text:
        emb = embed_texts_last_token([args.text], BASE_MODEL_PATH, layer_from_end,
                                     device=args.device, max_length=args.max_length)
        X = emb
        rows_payload = [{"text": args.text}]

    elif args.text_file:
        lines = [ln.strip() for ln in Path(args.text_file).read_text().splitlines() if ln.strip()]
        emb = embed_texts_last_token(lines, BASE_MODEL_PATH, layer_from_end,
                                     device=args.device, max_length=args.max_length)
        X = emb
        rows_payload = [{"text": t} for t in lines]

    elif args.vec:
        vec = np.array(ast.literal_eval(args.vec), dtype=np.float32).reshape(1, -1)
        X = vec
        rows_payload = [{"source": "vec"}]

    elif args.npy:
        X = np.load(args.npy).astype("float32")
        rows_payload = [{"row": i} for i in range(X.shape[0])]

    elif args.csv:
        df = pd.read_csv(args.csv)
        if "embeddings" in df.columns:
            X = parse_embeddings_column(df["embeddings"])
            rows_payload = df.to_dict("records")
        elif "text" in df.columns:
            texts = df["text"].astype(str).tolist()
            emb = embed_texts_last_token(texts, BASE_MODEL_PATH, layer_from_end,
                                         device=args.device, max_length=args.max_length)
            X = emb
            rows_payload = df.to_dict("records")
        else:
            raise ValueError("CSV must contain either 'embeddings' column or 'text' column.")
    else:
        raise RuntimeError("No input provided (this should be unreachable due to argparse).")

    # dim check if available
    d_model = int(meta.get("embedding_dim", X.shape[1]))
    if X.shape[1] != d_model:
        raise ValueError(f"Embedding dim mismatch: got {X.shape[1]} but classifier expects {d_model}")

    # ------------- run classifier -------------
    probs, pred_best, pred_05, pred_custom = predict_from_embeddings(clf, thr_opt, X, extra_threshold=custom_thr)

    # print concise results (truthfulness)
    print(f"\nConfig module: {args.config or os.getenv('SAPLMA_CONFIG', 'instruct_saplma.config')}")
    print(f"Bundle: {bundle}")
    print(f"Layer from end: {layer_from_end}")
    print(f"Using threshold: {thr_use:.4f} ({'best' if thr_mode=='best' else 'custom' if custom_thr else '0.5'})")
    hdr = "\nIdx | prob(True) | pred@best | pred@0.5"
    if custom_thr is not None:
        hdr += f" | pred@{custom_thr:.2f}"
    print(hdr)
    for i, p in enumerate(probs):
        col_best = POS_LABEL_NAME if pred_best[i] else NEG_LABEL_NAME
        col_05   = POS_LABEL_NAME if pred_05[i] else NEG_LABEL_NAME
        row = [f"{i:>3}", f"{p:>10.4f}", f"{col_best:>9}", f"{col_05:>9}"]
        if custom_thr is not None:
            col_c = POS_LABEL_NAME if pred_custom[i] else NEG_LABEL_NAME
            row.append(f"{col_c:>10}")
        print(" ".join(row))

    # ------------- save outputs if requested -------------
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        base_records = []
        for i, r in enumerate(rows_payload):
            rec = dict(r)
            rec.update({
                "index": i,
                PROB_NAME: float(probs[i]),  # prob(True)
                "pred_best_thr": POS_LABEL_NAME if pred_best[i] else NEG_LABEL_NAME,
                "pred_0p5": POS_LABEL_NAME if pred_05[i] else NEG_LABEL_NAME,
                "config_module": args.config or os.getenv("SAPLMA_CONFIG", "instruct_saplma.config"),
            })
            if custom_thr is not None:
                rec[f"pred_{custom_thr:.3f}"] = POS_LABEL_NAME if pred_custom[i] else NEG_LABEL_NAME
            base_records.append(rec)

        if out_path.suffix.lower() == ".json":
            out_path.write_text(json.dumps(base_records, indent=2))
        else:
            pd.DataFrame(base_records).to_csv(out_path, index=False)
        print(f"\n[OK] Wrote predictions → {out_path}")


if __name__ == "__main__":
    main()
