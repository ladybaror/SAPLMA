# saplma_api.py
# ------------------------------------------------------------
# Minimal API for SAPLMA (truthfulness; last-token embeddings).
# Defaults to paper.config (can override via config_module or SAPLMA_CONFIG).
#
# Exposed functions (completion style, unchanged):
#   load_best_bundle(...): -> (keras_model, threshold, meta, bundle_path)
#   embed_texts_last_token(texts, model_path, layer_from_end, device, max_length) -> np.ndarray
#   embed_text_last_token_with_loaded_model(tok, mdl, text, layer_from_end, max_length) -> np.ndarray
#   predict_embeddings(model, thr_bundle, X, custom_threshold=None) -> dict of arrays
#   predict_texts(texts, ...): high-level helper -> pd.DataFrame
#
# NEW (chat-templated / instruct style; added without altering existing APIs):
#   build_chat_formatted_text(...)
#   embed_texts_last_token_chat(...)
#   embed_text_last_token_with_loaded_model_chat(...)
#   predict_texts_chat(...)
# ------------------------------------------------------------

from __future__ import annotations
import json
import os
import importlib
from types import SimpleNamespace
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tensorflow.keras.models import load_model

# Label semantics (1 = True, 0 = Lie)
POS_LABEL_NAME = "True"
NEG_LABEL_NAME = "Lie"

# Default config to match your training script
_DEFAULT_CONFIG = "paper.config"


# ------------------ runtime config loader ------------------

def _load_cfg(config_module: Optional[str] = None) -> SimpleNamespace:
    """
    Load a config module dynamically. Priority:
      1) explicit config_module arg
      2) env SAPLMA_CONFIG
      3) default 'paper.config'
    Expected config (with safe defaults if missing):
      - OUTPUT_DIR (Path | str)
      - BASE_MODEL_PATH (str)
      - KERAS_MODEL_FILENAME (str)  default: model.keras
      - THRESHOLD_FILENAME (str)    default: threshold.txt
      - META_FILENAME (str)         default: meta.json
      - (optional) DATASET_FOLDER, MODEL_NAME (to derive OUTPUT_DIR if missing)
    """
    module_path = config_module or os.getenv("SAPLMA_CONFIG", _DEFAULT_CONFIG)
    cfg = importlib.import_module(module_path)

    base_model_path = getattr(cfg, "BASE_MODEL_PATH", "models/Llama-2-7b-chat-hf")
    keras_fname     = getattr(cfg, "KERAS_MODEL_FILENAME", "model.keras")
    thr_fname       = getattr(cfg, "THRESHOLD_FILENAME", "threshold.txt")
    meta_fname      = getattr(cfg, "META_FILENAME", "meta.json")

    output_dir = getattr(cfg, "OUTPUT_DIR", None)
    if output_dir is None:
        data_root = Path(getattr(cfg, "DATASET_FOLDER", "."))
        model_name = getattr(cfg, "MODEL_NAME", "MODEL")
        output_dir = data_root / f"saplma_checkpoints_{model_name}"
    output_dir = Path(output_dir)

    return SimpleNamespace(
        OUTPUT_DIR=output_dir,
        BASE_MODEL_PATH=base_model_path,
        KERAS_MODEL_FILENAME=keras_fname,
        THRESHOLD_FILENAME=thr_fname,
        META_FILENAME=meta_fname,
    )


# ------------------ bundle discovery / loading ------------------

def _bundle_ok(bundle_dir: Path, KERAS_MODEL_FILENAME: str, THRESHOLD_FILENAME: str, META_FILENAME: str) -> bool:
    return (
        (bundle_dir / KERAS_MODEL_FILENAME).exists()
        and (bundle_dir / THRESHOLD_FILENAME).exists()
        and (bundle_dir / META_FILENAME).exists()
    )

def _has_best_ancestor(path: Path, root: Path) -> bool:
    cur = path
    while cur != root and cur != cur.parent:
        if cur.name.startswith("BEST_"):
            return True
        cur = cur.parent
    return False

def discover_bundles(
    output_dir: Path | str,
    META_FILENAME: str,
    KERAS_MODEL_FILENAME: str,
    THRESHOLD_FILENAME: str,
) -> List[Path]:
    """Recursively find bundle dirs that live under a folder named BEST_*."""
    output_dir = Path(output_dir)
    bundles: List[Path] = []
    for meta_path in output_dir.rglob(META_FILENAME):
        if not _has_best_ancestor(meta_path.parent, output_dir):
            continue
        bd = meta_path.parent
        if _bundle_ok(bd, KERAS_MODEL_FILENAME, THRESHOLD_FILENAME, META_FILENAME):
            bundles.append(bd)
    return bundles

def find_best_bundle(
    output_dir: Path | str,
    META_FILENAME: str,
    KERAS_MODEL_FILENAME: str,
    THRESHOLD_FILENAME: str,
    prefer_substring: Optional[str] = None,
) -> Path:
    """Pick bundle with highest metric_value in meta.json. Optionally filter by substring in path."""
    cands = discover_bundles(output_dir, META_FILENAME, KERAS_MODEL_FILENAME, THRESHOLD_FILENAME)
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
        raise FileNotFoundError(f"No BEST bundles found under {output_dir}")
    if prefer_substring:
        preferred = [t for t in scored if prefer_substring in str(t[1])]
        if preferred:
            scored = preferred
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]

def load_best_bundle(
    bundle: Optional[str | Path] = None,
    prefer: Optional[str] = None,
    config_module: Optional[str] = None,
    output_dir: Optional[str | Path] = None,
):
    """
    Return (keras_model, threshold, meta_dict, bundle_path).

    Selection logic:
      - If 'bundle' is provided: use it (validate files).
      - Else: choose best under (output_dir or cfg.OUTPUT_DIR), optionally filtered by 'prefer'.
    """
    cfg = _load_cfg(config_module)
    out_dir = Path(output_dir) if output_dir is not None else cfg.OUTPUT_DIR

    if bundle is None:
        bundle_path = find_best_bundle(
            out_dir,
            cfg.META_FILENAME,
            cfg.KERAS_MODEL_FILENAME,
            cfg.THRESHOLD_FILENAME,
            prefer_substring=prefer,
        )
    else:
        bundle_path = Path(bundle)
        if not _bundle_ok(bundle_path, cfg.KERAS_MODEL_FILENAME, cfg.THRESHOLD_FILENAME, cfg.META_FILENAME):
            raise FileNotFoundError(f"Provided bundle missing required files: {bundle_path}")

    model = load_model(bundle_path / cfg.KERAS_MODEL_FILENAME)
    thr   = float((bundle_path / cfg.THRESHOLD_FILENAME).read_text())
    meta  = json.loads((bundle_path / cfg.META_FILENAME).read_text())
    return model, thr, meta, bundle_path


# ------------------ embedding (LAST NON-PAD TOKEN) ------------------

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
    texts -> embeddings (N, D) from requested layer; uses last NON-PAD token per sequence.
    """
    tok, mdl = get_tokenizer_and_model(model_path, device=device)
    dev = next(mdl.parameters()).device
    out = []

    for text in texts:
        enc = tok(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length if max_length else None,
        )
        enc = {k: v.to(dev) for k, v in enc.items()}
        res = mdl(**enc, output_hidden_states=True, use_cache=False)
        hs = res.hidden_states
        layer_states = hs[layer_from_end][0]  # [seq, hidden]
        attn = enc.get("attention_mask", torch.ones_like(enc["input_ids"]))[0]
        last_idx = max(int(attn.sum().item()) - 1, 0)
        out.append(layer_states[last_idx].float().cpu().numpy())

    return np.stack(out, axis=0)

@torch.no_grad()
def embed_text_last_token_with_loaded_model(tok, mdl, text: str, layer_from_end: int, max_length: Optional[int] = None):
    dev = next(mdl.parameters()).device
    enc = tok(text, return_tensors="pt", truncation=True, max_length=max_length)
    enc = {k: v.to(dev) for k, v in enc.items()}
    out = mdl(**enc, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states
    layer_states = hs[layer_from_end][0]
    attn = enc.get("attention_mask", torch.ones_like(enc["input_ids"]))[0]
    last_idx = max(int(attn.sum().item()) - 1, 0)
    return layer_states[last_idx].float().cpu()


# ------------------ NEW: chat-templated helpers (added) ------------------

def build_chat_formatted_text(
    tok: AutoTokenizer,
    content: str,
    role: str = "assistant",
    user_turn: Optional[str] = None,
    system: Optional[str] = None,
    add_generation_prompt: bool = False,
) -> str:
    """
    Serialize a message `content` into the model's chat template if available.
    Fallback to a simple tagged string if the tokenizer lacks a template.

    Parameters
    ----------
    tok : AutoTokenizer
    content : str
        The text span you want to classify (e.g., a sentence or cumulative text).
    role : {"assistant","user","system"}
        Role assigned to `content`.
    user_turn : Optional[str]
        An optional preceding user message to include in the chat template.
    system : Optional[str]
        Optional system prompt to include.
    add_generation_prompt : bool
        Passed to tokenizer.apply_chat_template.

    Returns
    -------
    str
        A serialized string ready for tokenization by the same tokenizer.
    """
    messages: List[Dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    if user_turn is not None:
        messages.append({"role": "user", "content": user_turn})
    messages.append({"role": role, "content": content})

    if hasattr(tok, "apply_chat_template"):
        try:
            return tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_generation_prompt
            )
        except Exception:
            # Fall through to a very simple fallback below
            pass

    # Fallback: a minimal tagged format that is stable across tokenizers
    parts = []
    if system:
        parts.append(f"<|system|>\n{system}\n")
    if user_turn is not None:
        parts.append(f"<|user|>\n{user_turn}\n")
    parts.append(f"<|{role}|>\n{content}")
    return "\n".join(parts)


@torch.no_grad()
def embed_texts_last_token_chat(
    texts: List[str],
    model_path: str,
    layer_from_end: int,
    device: str = "auto",
    max_length: Optional[int] = None,
    *,
    role: str = "assistant",
    user_turn: Optional[str] = None,
    system: Optional[str] = None,
    add_generation_prompt: bool = False,
) -> np.ndarray:
    """
    Chat-templated variant of embed_texts_last_token (does not modify the original).
    Each `text` is serialized via the tokenizer's chat template before embedding.
    """
    tok, mdl = get_tokenizer_and_model(model_path, device=device)
    dev = next(mdl.parameters()).device
    out = []

    for text in texts:
        chat_text = build_chat_formatted_text(
            tok, text, role=role, user_turn=user_turn, system=system,
            add_generation_prompt=add_generation_prompt
        )
        enc = tok(chat_text, return_tensors="pt", truncation=True, max_length=max_length)
        enc = {k: v.to(dev) for k, v in enc.items()}
        res = mdl(**enc, output_hidden_states=True, use_cache=False)
        hs = res.hidden_states
        layer_states = hs[layer_from_end][0]
        attn = enc.get("attention_mask", torch.ones_like(enc["input_ids"]))[0]
        last_idx = max(int(attn.sum().item()) - 1, 0)
        out.append(layer_states[last_idx].float().cpu().numpy())

    return np.stack(out, axis=0)


@torch.no_grad()
def embed_text_last_token_with_loaded_model_chat(
    tok,
    mdl,
    text: str,
    layer_from_end: int,
    max_length: Optional[int] = None,
    *,
    role: str = "assistant",
    user_turn: Optional[str] = None,
    system: Optional[str] = None,
    add_generation_prompt: bool = False,
):
    """
    Chat-templated variant of embed_text_last_token_with_loaded_model (unchanged original kept).
    Serializes `text` with the tokenizer's chat template before embedding.
    """
    chat_text = build_chat_formatted_text(
        tok, text, role=role, user_turn=user_turn, system=system,
        add_generation_prompt=add_generation_prompt
    )
    dev = next(mdl.parameters()).device
    enc = tok(chat_text, return_tensors="pt", truncation=True, max_length=max_length)
    enc = {k: v.to(dev) for k, v in enc.items()}
    out = mdl(**enc, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states
    layer_states = hs[layer_from_end][0]
    attn = enc.get("attention_mask", torch.ones_like(enc["input_ids"]))[0]
    last_idx = max(int(attn.sum().item()) - 1, 0)
    return layer_states[last_idx].float().cpu()


# ------------------ prediction ------------------

def predict_embeddings(
    model,
    threshold_bundle: float,
    X: np.ndarray,
    custom_threshold: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    """
    X: (N, D) embeddings. Returns dict with:
      - prob_true: (N,)
      - pred_best: (N,) 'True'/'Lie' via bundle threshold
      - pred_0p5:  (N,) 'True'/'Lie' via 0.5
      - pred_custom: (N,) (only if custom_threshold given)
    """
    probs = model.predict(X, verbose=0).ravel()  # prob(True)
    pred_best = np.where(probs > threshold_bundle, POS_LABEL_NAME, NEG_LABEL_NAME)
    pred_05   = np.where(probs > 0.5,             POS_LABEL_NAME, NEG_LABEL_NAME)
    out = {
        "prob_true": probs,
        "pred_best": pred_best,
        "pred_0p5":  pred_05,
    }
    if custom_threshold is not None:
        pred_c = np.where(probs > float(custom_threshold), POS_LABEL_NAME, NEG_LABEL_NAME)
        out["pred_custom"] = pred_c
    return out


def predict_texts(
    texts: List[str],
    bundle: Optional[str | Path] = None,
    prefer: Optional[str] = None,
    model_path: Optional[str] = None,
    layer_from_end: Optional[int] = None,
    device: str = "auto",
    max_length: Optional[int] = None,
    threshold: str | float = "best",
    config_module: Optional[str] = None,
    output_dir: Optional[str | Path] = None,
) -> pd.DataFrame:
    """
    High-level helper (completion style):
      texts -> embed (last-token) -> run classifier -> DataFrame
    """
    cfg = _load_cfg(config_module)

    clf, thr_opt, meta, bundle_path = load_best_bundle(
        bundle=bundle, prefer=prefer, config_module=config_module, output_dir=output_dir
    )

    layer = int(layer_from_end) if layer_from_end is not None else int(meta["layer_from_end"])
    base_model_path = model_path or cfg.BASE_MODEL_PATH

    X = embed_texts_last_token(texts, base_model_path, layer, device=device, max_length=max_length)

    # sanity check on dim
    dim_expected = int(meta.get("embedding_dim", X.shape[1]))
    if X.shape[1] != dim_expected:
        raise ValueError(f"Embedding dim mismatch: got {X.shape[1]} vs model expects {dim_expected}")

    # threshold selection
    if isinstance(threshold, str):
        thr_s = threshold.strip().lower()
        if thr_s == "best":
            custom = None
        elif thr_s == "0.5":
            custom = None
        else:
            raise ValueError("threshold must be 'best', '0.5', or a float")
    else:
        custom = float(threshold)

    res = predict_embeddings(clf, thr_opt, X, custom_threshold=custom)

    # Build dataframe
    df = pd.DataFrame({
        "text": texts,
        "prob_true": res["prob_true"],
        "pred_best_thr": res["pred_best"],
        "pred_0p5": res["pred_0p5"],
    })
    if "pred_custom" in res:
        df["pred_custom"] = res["pred_custom"]

    # Attach metadata for traceability
    df.attrs["bundle"] = str(bundle_path)
    df.attrs["layer_from_end"] = layer
    df.attrs["config_module"] = config_module or os.getenv("SAPLMA_CONFIG", _DEFAULT_CONFIG)
    df.attrs["output_dir"] = str(output_dir) if output_dir is not None else str(cfg.OUTPUT_DIR)
    df.attrs["model_path"] = base_model_path

    return df


# ------------------ NEW: chat-templated prediction (added) ------------------

def predict_texts_chat(
    texts: List[str],
    bundle: Optional[str | Path] = None,
    prefer: Optional[str] = None,
    model_path: Optional[str] = None,
    layer_from_end: Optional[int] = None,
    device: str = "auto",
    max_length: Optional[int] = None,
    threshold: str | float = "best",
    config_module: Optional[str] = None,
    output_dir: Optional[str | Path] = None,
    *,
    role: str = "assistant",
    user_turn: Optional[str] = None,
    system: Optional[str] = None,
    add_generation_prompt: bool = False,
) -> pd.DataFrame:
    """
    High-level helper (chat-templated):
      texts -> serialize with chat template -> embed (last-token) -> classify
    """
    cfg = _load_cfg(config_module)

    clf, thr_opt, meta, bundle_path = load_best_bundle(
        bundle=bundle, prefer=prefer, config_module=config_module, output_dir=output_dir
    )

    layer = int(layer_from_end) if layer_from_end is not None else int(meta["layer_from_end"])
    base_model_path = model_path or cfg.BASE_MODEL_PATH

    X = embed_texts_last_token_chat(
        texts=texts,
        model_path=base_model_path,
        layer_from_end=layer,
        device=device,
        max_length=max_length,
        role=role,
        user_turn=user_turn,
        system=system,
        add_generation_prompt=add_generation_prompt,
    )

    # sanity check on dim
    dim_expected = int(meta.get("embedding_dim", X.shape[1]))
    if X.shape[1] != dim_expected:
        raise ValueError(f"Embedding dim mismatch: got {X.shape[1]} vs model expects {dim_expected}")

    # threshold selection
    if isinstance(threshold, str):
        thr_s = threshold.strip().lower()
        if thr_s == "best":
            custom = None
        elif thr_s == "0.5":
            custom = None
        else:
            raise ValueError("threshold must be 'best', '0.5', or a float")
    else:
        custom = float(threshold)

    res = predict_embeddings(clf, thr_opt, X, custom_threshold=custom)

    # Build dataframe
    df = pd.DataFrame({
        "text": texts,
        "prob_true": res["prob_true"],
        "pred_best_thr": res["pred_best"],
        "pred_0p5": res["pred_0p5"],
    })
    if "pred_custom" in res:
        df["pred_custom"] = res["pred_custom"]

    # Attach metadata for traceability
    df.attrs["bundle"] = str(bundle_path)
    df.attrs["layer_from_end"] = layer
    df.attrs["config_module"] = config_module or os.getenv("SAPLMA_CONFIG", _DEFAULT_CONFIG)
    df.attrs["output_dir"] = str(output_dir) if output_dir is not None else str(cfg.OUTPUT_DIR)
    df.attrs["model_path"] = base_model_path
    df.attrs["chat_role"] = role
    df.attrs["chat_has_user_turn"] = user_turn is not None
    df.attrs["chat_has_system"] = system is not None
    df.attrs["chat_add_generation_prompt"] = add_generation_prompt

    return df





# # saplma_api.py
# # ------------------------------------------------------------
# # Minimal API for SAPLMA (truthfulness; last-token embeddings).
# # Defaults to paper.config (can override via config_module or SAPLMA_CONFIG).
# #
# # Exposed functions:
# #   load_best_bundle(...): -> (keras_model, threshold, meta, bundle_path)
# #   embed_texts_last_token(texts, model_path, layer_from_end, device, max_length) -> np.ndarray
# #   predict_embeddings(model, thr_bundle, X, custom_threshold=None) -> dict of arrays
# #   predict_texts(texts, ...): high-level helper -> pd.DataFrame
# # ------------------------------------------------------------

# from __future__ import annotations
# import json
# import os
# import importlib
# from types import SimpleNamespace
# from pathlib import Path
# from typing import List, Optional, Dict

# import numpy as np
# import pandas as pd
# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from tensorflow.keras.models import load_model

# # Label semantics (1 = True, 0 = Lie)
# POS_LABEL_NAME = "True"
# NEG_LABEL_NAME = "Lie"

# # Default config to match your training script
# _DEFAULT_CONFIG = "paper.config"


# # ------------------ runtime config loader ------------------

# def _load_cfg(config_module: Optional[str] = None) -> SimpleNamespace:
#     """
#     Load a config module dynamically. Priority:
#       1) explicit config_module arg
#       2) env SAPLMA_CONFIG
#       3) default 'paper.config'
#     Expected config (with safe defaults if missing):
#       - OUTPUT_DIR (Path | str)
#       - BASE_MODEL_PATH (str)
#       - KERAS_MODEL_FILENAME (str)  default: model.keras
#       - THRESHOLD_FILENAME (str)    default: threshold.txt
#       - META_FILENAME (str)         default: meta.json
#       - (optional) DATASET_FOLDER, MODEL_NAME (to derive OUTPUT_DIR if missing)
#     """
#     module_path = config_module or os.getenv("SAPLMA_CONFIG", _DEFAULT_CONFIG)
#     cfg = importlib.import_module(module_path)

#     base_model_path = getattr(cfg, "BASE_MODEL_PATH", "models/Llama-2-7b-chat-hf")
#     keras_fname     = getattr(cfg, "KERAS_MODEL_FILENAME", "model.keras")
#     thr_fname       = getattr(cfg, "THRESHOLD_FILENAME", "threshold.txt")
#     meta_fname      = getattr(cfg, "META_FILENAME", "meta.json")

#     output_dir = getattr(cfg, "OUTPUT_DIR", None)
#     if output_dir is None:
#         data_root = Path(getattr(cfg, "DATASET_FOLDER", "."))
#         model_name = getattr(cfg, "MODEL_NAME", "MODEL")
#         output_dir = data_root / f"saplma_checkpoints_{model_name}"
#     output_dir = Path(output_dir)

#     return SimpleNamespace(
#         OUTPUT_DIR=output_dir,
#         BASE_MODEL_PATH=base_model_path,
#         KERAS_MODEL_FILENAME=keras_fname,
#         THRESHOLD_FILENAME=thr_fname,
#         META_FILENAME=meta_fname,
#     )


# # ------------------ bundle discovery / loading ------------------

# def _bundle_ok(bundle_dir: Path, KERAS_MODEL_FILENAME: str, THRESHOLD_FILENAME: str, META_FILENAME: str) -> bool:
#     return (
#         (bundle_dir / KERAS_MODEL_FILENAME).exists()
#         and (bundle_dir / THRESHOLD_FILENAME).exists()
#         and (bundle_dir / META_FILENAME).exists()
#     )

# def _has_best_ancestor(path: Path, root: Path) -> bool:
#     cur = path
#     while cur != root and cur != cur.parent:
#         if cur.name.startswith("BEST_"):
#             return True
#         cur = cur.parent
#     return False

# def discover_bundles(
#     output_dir: Path | str,
#     META_FILENAME: str,
#     KERAS_MODEL_FILENAME: str,
#     THRESHOLD_FILENAME: str,
# ) -> List[Path]:
#     """Recursively find bundle dirs that live under a folder named BEST_*."""
#     output_dir = Path(output_dir)
#     bundles: List[Path] = []
#     for meta_path in output_dir.rglob(META_FILENAME):
#         if not _has_best_ancestor(meta_path.parent, output_dir):
#             continue
#         bd = meta_path.parent
#         if _bundle_ok(bd, KERAS_MODEL_FILENAME, THRESHOLD_FILENAME, META_FILENAME):
#             bundles.append(bd)
#     return bundles

# def find_best_bundle(
#     output_dir: Path | str,
#     META_FILENAME: str,
#     KERAS_MODEL_FILENAME: str,
#     THRESHOLD_FILENAME: str,
#     prefer_substring: Optional[str] = None,
# ) -> Path:
#     """Pick bundle with highest metric_value in meta.json. Optionally filter by substring in path."""
#     cands = discover_bundles(output_dir, META_FILENAME, KERAS_MODEL_FILENAME, THRESHOLD_FILENAME)
#     scored = []
#     for d in cands:
#         try:
#             meta = json.loads((d / META_FILENAME).read_text())
#             mv = float(meta.get("metric_value", float("nan")))
#             if not np.isnan(mv):
#                 scored.append((mv, d))
#         except Exception:
#             pass
#     if not scored:
#         raise FileNotFoundError(f"No BEST bundles found under {output_dir}")
#     if prefer_substring:
#         preferred = [t for t in scored if prefer_substring in str(t[1])]
#         if preferred:
#             scored = preferred
#     scored.sort(key=lambda x: x[0], reverse=True)
#     return scored[0][1]

# def load_best_bundle(
#     bundle: Optional[str | Path] = None,
#     prefer: Optional[str] = None,
#     config_module: Optional[str] = None,
#     output_dir: Optional[str | Path] = None,
# ):
#     """
#     Return (keras_model, threshold, meta_dict, bundle_path).

#     Selection logic:
#       - If 'bundle' is provided: use it (validate files).
#       - Else: choose best under (output_dir or cfg.OUTPUT_DIR), optionally filtered by 'prefer'.
#     """
#     cfg = _load_cfg(config_module)
#     out_dir = Path(output_dir) if output_dir is not None else cfg.OUTPUT_DIR

#     if bundle is None:
#         bundle_path = find_best_bundle(
#             out_dir,
#             cfg.META_FILENAME,
#             cfg.KERAS_MODEL_FILENAME,
#             cfg.THRESHOLD_FILENAME,
#             prefer_substring=prefer,
#         )
#     else:
#         bundle_path = Path(bundle)
#         if not _bundle_ok(bundle_path, cfg.KERAS_MODEL_FILENAME, cfg.THRESHOLD_FILENAME, cfg.META_FILENAME):
#             raise FileNotFoundError(f"Provided bundle missing required files: {bundle_path}")

#     model = load_model(bundle_path / cfg.KERAS_MODEL_FILENAME)
#     thr   = float((bundle_path / cfg.THRESHOLD_FILENAME).read_text())
#     meta  = json.loads((bundle_path / cfg.META_FILENAME).read_text())
#     return model, thr, meta, bundle_path


# # ------------------ embedding (LAST NON-PAD TOKEN) ------------------

# def get_tokenizer_and_model(model_path: str, device: str = "auto"):
#     tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
#     if tok.pad_token is None:
#         tok.pad_token = tok.eos_token
#         tok.pad_token_id = tok.eos_token_id
#     dtype = torch.float16 if torch.cuda.is_available() and device != "cpu" else torch.float32
#     if device == "cpu":
#         mdl = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, device_map=None)
#         mdl.to("cpu")
#     else:
#         mdl = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, device_map="auto")
#     mdl.eval()
#     return tok, mdl

# @torch.no_grad()
# def embed_texts_last_token(
#     texts: List[str],
#     model_path: str,
#     layer_from_end: int,
#     device: str = "auto",
#     max_length: Optional[int] = None,
# ) -> np.ndarray:
#     """
#     texts -> embeddings (N, D) from requested layer; uses last NON-PAD token per sequence.
#     """
#     tok, mdl = get_tokenizer_and_model(model_path, device=device)
#     dev = next(mdl.parameters()).device
#     out = []

#     for text in texts:
#         enc = tok(
#             text,
#             return_tensors="pt",
#             truncation=True,
#             max_length=max_length if max_length else None,
#         )
#         enc = {k: v.to(dev) for k, v in enc.items()}
#         res = mdl(**enc, output_hidden_states=True, use_cache=False)
#         hs = res.hidden_states
#         layer_states = hs[layer_from_end][0]  # [seq, hidden]
#         attn = enc.get("attention_mask", torch.ones_like(enc["input_ids"]))[0]
#         last_idx = max(int(attn.sum().item()) - 1, 0)
#         out.append(layer_states[last_idx].float().cpu().numpy())

#     return np.stack(out, axis=0)

# @torch.no_grad()
# def embed_text_last_token_with_loaded_model(tok, mdl, text: str, layer_from_end: int, max_length: Optional[int] = None):
#     dev = next(mdl.parameters()).device
#     enc = tok(text, return_tensors="pt", truncation=True, max_length=max_length)
#     enc = {k: v.to(dev) for k, v in enc.items()}
#     out = mdl(**enc, output_hidden_states=True, use_cache=False)
#     hs = out.hidden_states
#     layer_states = hs[layer_from_end][0]
#     attn = enc.get("attention_mask", torch.ones_like(enc["input_ids"]))[0]
#     last_idx = max(int(attn.sum().item()) - 1, 0)
#     return layer_states[last_idx].float().cpu()

# # ------------------ prediction ------------------

# def predict_embeddings(
#     model,
#     threshold_bundle: float,
#     X: np.ndarray,
#     custom_threshold: Optional[float] = None,
# ) -> Dict[str, np.ndarray]:
#     """
#     X: (N, D) embeddings. Returns dict with:
#       - prob_true: (N,)
#       - pred_best: (N,) 'True'/'Lie' via bundle threshold
#       - pred_0p5:  (N,) 'True'/'Lie' via 0.5
#       - pred_custom: (N,) (only if custom_threshold given)
#     """
#     probs = model.predict(X, verbose=0).ravel()  # prob(True)
#     pred_best = np.where(probs > threshold_bundle, POS_LABEL_NAME, NEG_LABEL_NAME)
#     pred_05   = np.where(probs > 0.5,             POS_LABEL_NAME, NEG_LABEL_NAME)
#     out = {
#         "prob_true": probs,
#         "pred_best": pred_best,
#         "pred_0p5":  pred_05,
#     }
#     if custom_threshold is not None:
#         pred_c = np.where(probs > float(custom_threshold), POS_LABEL_NAME, NEG_LABEL_NAME)
#         out["pred_custom"] = pred_c
#     return out


# def predict_texts(
#     texts: List[str],
#     bundle: Optional[str | Path] = None,
#     prefer: Optional[str] = None,
#     model_path: Optional[str] = None,
#     layer_from_end: Optional[int] = None,
#     device: str = "auto",
#     max_length: Optional[int] = None,
#     threshold: str | float = "best",
#     config_module: Optional[str] = None,
#     output_dir: Optional[str | Path] = None,
# ) -> pd.DataFrame:
#     """
#     High-level helper:
#       texts -> embed (last-token) -> run classifier -> DataFrame

#     threshold:
#       'best'  → bundle's saved optimal threshold
#       '0.5'   → fixed 0.5
#       float   → custom threshold
#     """
#     cfg = _load_cfg(config_module)

#     clf, thr_opt, meta, bundle_path = load_best_bundle(
#         bundle=bundle, prefer=prefer, config_module=config_module, output_dir=output_dir
#     )

#     layer = int(layer_from_end) if layer_from_end is not None else int(meta["layer_from_end"])
#     base_model_path = model_path or cfg.BASE_MODEL_PATH

#     X = embed_texts_last_token(texts, base_model_path, layer, device=device, max_length=max_length)

#     # sanity check on dim
#     dim_expected = int(meta.get("embedding_dim", X.shape[1]))
#     if X.shape[1] != dim_expected:
#         raise ValueError(f"Embedding dim mismatch: got {X.shape[1]} vs model expects {dim_expected}")

#     # threshold selection
#     if isinstance(threshold, str):
#         thr_s = threshold.strip().lower()
#         if thr_s == "best":
#             custom = None
#         elif thr_s == "0.5":
#             custom = None
#         else:
#             raise ValueError("threshold must be 'best', '0.5', or a float")
#     else:
#         custom = float(threshold)

#     res = predict_embeddings(clf, thr_opt, X, custom_threshold=custom)

#     # Build dataframe
#     df = pd.DataFrame({
#         "text": texts,
#         "prob_true": res["prob_true"],
#         "pred_best_thr": res["pred_best"],
#         "pred_0p5": res["pred_0p5"],
#     })
#     if "pred_custom" in res:
#         df["pred_custom"] = res["pred_custom"]

#     # Attach metadata for traceability
#     df.attrs["bundle"] = str(bundle_path)
#     df.attrs["layer_from_end"] = layer
#     df.attrs["config_module"] = config_module or os.getenv("SAPLMA_CONFIG", _DEFAULT_CONFIG)
#     df.attrs["output_dir"] = str(output_dir) if output_dir is not None else str(cfg.OUTPUT_DIR)
#     df.attrs["model_path"] = base_model_path

#     return df
