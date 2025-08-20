# quick_check_saplma.py
# ------------------------------------------------------------
# Single example: sentence -> LAST-TOKEN embedding -> SAPLMA (truthfulness)
# Uses your saved BEST bundle (auto-discovered) unless --bundle is given.
# Prints Prob(True) and decisions True/Lie at 0.5 and best threshold.
# ------------------------------------------------------------

import os
import json
from pathlib import Path
import argparse
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tensorflow.keras.models import load_model

# ---- from your config ----
from paper.config import (
    BASE_MODEL_PATH,
    OUTPUT_DIR,
)

# ---- files ----
KERAS_MODEL_FILENAME = "model.keras"
THRESHOLD_FILENAME   = "threshold.txt"
META_FILENAME        = "meta.json"

# ---- labels ----
POS_LABEL_NAME = "True"
NEG_LABEL_NAME = "Lie"


def _bundle_ok(p: Path) -> bool:
    return (p / KERAS_MODEL_FILENAME).exists() and (p / THRESHOLD_FILENAME).exists() and (p / META_FILENAME).exists()

def _has_best_ancestor(p: Path, root: Path) -> bool:
    cur = p
    while cur != root and cur != cur.parent:
        if cur.name.startswith("BEST_"):
            return True
        cur = cur.parent
    return False

def find_best_bundle(output_dir: Path) -> Path:
    output_dir = Path(output_dir)
    cands = []
    for meta_path in output_dir.rglob(META_FILENAME):
        if not _has_best_ancestor(meta_path.parent, output_dir):
            continue
        bundle = meta_path.parent
        if not _bundle_ok(bundle):
            continue
        try:
            meta = json.loads(meta_path.read_text())
            mv = float(meta.get("metric_value", float("nan")))
            if not np.isnan(mv):
                cands.append((mv, bundle))
        except Exception:
            pass
    if not cands:
        raise FileNotFoundError(f"No BEST bundles found under {output_dir}")
    cands.sort(key=lambda x: x[0], reverse=True)
    return cands[0][1]

def load_bundle(bundle_dir: Path):
    bundle_dir = Path(bundle_dir)
    model = load_model(bundle_dir / KERAS_MODEL_FILENAME)
    thr = float((bundle_dir / THRESHOLD_FILENAME).read_text())
    meta = json.loads((bundle_dir / META_FILENAME).read_text())
    return model, thr, meta

# ---- LAST-TOKEN embedding ----
@torch.no_grad()
def embed_text_last_token(
    text: str,
    model_path: str,
    layer_from_end: int,
    device: str = "auto",
    max_length: int | None = None,
) -> np.ndarray:
    """
    Return 1D numpy vector from requested layer (last NON-PAD token).
    """
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

    dev = next(mdl.parameters()).device
    enc = tok(text, return_tensors="pt", truncation=True, max_length=max_length)
    enc = {k: v.to(dev) for k, v in enc.items()}

    out = mdl(**enc, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states
    layer_states = hs[layer_from_end][0]  # [seq, hidden]

    attn = enc.get("attention_mask", torch.ones_like(enc["input_ids"]))[0]  # [seq]
    valid_tokens = int(attn.sum().item())
    last_idx = max(valid_tokens - 1, 0)

    emb = layer_states[last_idx]  # [hidden]
    return emb.float().cpu().numpy()

def run_single(text: str, bundle: Path | None, device: str = "auto", max_length: int | None = None):
    # Locate bundle
    if bundle is None:
        bundle = find_best_bundle(Path(OUTPUT_DIR))
    else:
        bundle = Path(bundle)
        if not _bundle_ok(bundle):
            raise FileNotFoundError(f"--bundle missing required files in {bundle}")

    clf, thr_opt, meta = load_bundle(bundle)

    # Use the layer the classifier was trained on
    layer_from_end = int(meta["layer_from_end"])

    emb = embed_text_last_token(text, BASE_MODEL_PATH, layer_from_end, device=device, max_length=max_length)

    # dim check
    d_model = int(meta.get("embedding_dim", emb.shape[0]))
    if emb.shape[0] != d_model:
        raise ValueError(f"Embedding dim mismatch: got {emb.shape[0]} but model expects {d_model}")

    x = emb.reshape(1, -1)
    prob_true = float(clf.predict(x, verbose=0).ravel()[0])

    pred_05   = POS_LABEL_NAME if (prob_true > 0.5) else NEG_LABEL_NAME
    pred_best = POS_LABEL_NAME if (prob_true > thr_opt) else NEG_LABEL_NAME

    print("\n--- SAPLMA single example (last-token; truthfulness) ---")
    print(f"Bundle: {bundle}")
    print(f"Text: {text}")
    print(f"Layer from end: {layer_from_end}")
    print(f"Prob(True): {prob_true:.4f}")
    print(f"Decision @ 0.50: {pred_05}")
    print(f"Decision @ best thr ({thr_opt:.4f}): {pred_best}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Single example → LAST-TOKEN embedding → SAPLMA (truthfulness) → result")
    ap.add_argument("--text", required=True, help="Sentence to check")
    ap.add_argument("--bundle", default=None, help="Path to a specific BEST bundle (folder with model.keras etc.)")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto", help="Device for embedding model.")
    ap.add_argument("--max-length", type=int, default=None, help="Max tokens when embedding (truncate).")
    args = ap.parse_args()
    run_single(args.text, args.bundle, device=args.device, max_length=args.max_length)



"""
usage example:

*** Single sentence:

python quick_check_saplma.py --text "Paris is the capital of France."


*** CSV with text column → embed + predict, save results:

python saplma_run.py --csv data/my_texts.csv --out preds.csv

*** Pick a specific saved bundle and custom threshold:

python saplma_run.py \
  --bundle pretrained_saplma/completion/saplma_checkpoints_LLAMA7/BEST_layer12__heldout_data/elements \
  --text "Mercury is the hottest planet in the Solar System." \
  --threshold 0.6

"""
