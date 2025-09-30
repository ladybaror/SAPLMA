# saplma_guarded_generate_completion_auto.py
# ---------------------------------------------------------------------------
# Guardrailed generation with SAPLMA (Simple Accuracy Prediction via Last-token
# Model Activation) for sentence-by-sentence filtering, template-friendly.
#
# This revision:
# - Uses a sentence tokenizer (spaCy -> NLTK -> regex fallback).
# - Caches HF model/tokenizer, SAPLMA bundle, and the sentence tokenizer.
# - No prompt echo in assistant output.
# - Strong first-token hygiene; bans newline as first token.
# - No auto-capitalization; model controls casing.
# - Avoid leading space on the very first append to final_text.
# - Strips EOS/whitespace-piece tokens from accepted_ids after each re-render.
# - Numeric rule: a bare "3." is NOT EOS; "3.14." (second dot) is EOS.
# - ACCEPTED sentences have rich diagnostics, mirroring REJECTED.
# - Keeps objects for `accepted_sentences` / `rejected` (attribute access OK).
# - Adds GuardedGenerationResult.to_dict() for JSON-safe serialization.
# ---------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Set, List, Tuple, Dict, Any
import copy
import logging
import time
import re
import os
import sys
from inspect import signature

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# =========================
# Module-level singletons
# =========================
_TOK = None
_MDL = None
_MODEL_PATH = None

_SAPLMA = None  # (saplma_model, embed_fn, layer_from_end, thr_opt_base)
_BUNDLE_PATH = None

_SENT_TOK = None  # sentence tokenizer singleton


# ---------------------------------------------------------------------------
# Dataclasses (internal)
# ---------------------------------------------------------------------------
@dataclass
class SAPLMARejected:
    type: str                 # "filter" | "saplma" | "runon"
    attempt: int
    mode: str                 # "GREEDY" | "SAMPLE"
    sentence_out: str
    clean_sentence: str
    classify_text: str
    prob_true: float | None
    threshold: float | None
    reason: str | None        # e.g., "prob<thr", "too short", etc.

@dataclass
class SAPLMAAccepted:
    type: str                 # "saplma"
    attempt: int
    mode: str                 # "GREEDY" | "SAMPLE"
    sentence_out: str
    clean_sentence: str
    classify_text: str
    prob_true: float
    threshold: float
    reason: str               # e.g., "prob>=thr"

@dataclass
class GuardedGenerationResult:
    final_text: str
    accepted_sentences: List[SAPLMAAccepted]   # keep objects (attribute access OK)
    rejected: List[SAPLMARejected]             # keep objects (attribute access OK)
    summary: Dict[str, int | float | bool | str]
    accepted_texts: List[str]                  # convenience strings

    # JSON-safe projection
    def to_dict(self) -> Dict[str, Any]:
        return {
            "final_text": self.final_text,
            "accepted_sentences": [asdict(e) for e in self.accepted_sentences],
            "rejected": [asdict(e) for e in self.rejected],
            "summary": self.summary,
            "accepted_texts": self.accepted_texts,
        }


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def _setup_logger(level: str = "INFO"):
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s | %(levelname)-5s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("saplma_guard")


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
def _nucleus_sample(
    next_logits: torch.Tensor,
    top_p: float = 0.8,
    temperature: float = 0.8,
    min_tokens_to_keep: int = 5
) -> int:
    if temperature and temperature != 1.0:
        next_logits = next_logits / temperature
    probs = F.softmax(next_logits, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cum = torch.cumsum(sorted_probs, dim=-1)

    mask = cum <= top_p
    if min_tokens_to_keep > 0:
        mask[:min(min_tokens_to_keep, mask.numel())] = True

    keep_probs = sorted_probs[mask]
    keep_idx = sorted_idx[mask]
    keep_probs = keep_probs / keep_probs.sum()
    pick_rel = torch.multinomial(keep_probs, 1).item()
    return int(keep_idx[pick_rel].item())


# ---------------------------------------------------------------------------
# Content filters & cleaning
# ---------------------------------------------------------------------------
ENUM_PREFIX_RE = re.compile(r"^\s*(?:Answer:\s*)?(?:\(?[A-Za-z]\)|[A-Za-z]\)|\d+\)|\d+\.)\s*", re.UNICODE)
DATE_RE   = re.compile(r"^\s*\d{4}[-/]\d{2}[-/]\d{2}\s*\.?\s*$")
TIME_RE   = re.compile(r"^\s*\d{1,2}:\d{2}(?::\d{2})?\s*(AM|PM|am|pm)?\s*\.?\s*$")
UTC_RE    = re.compile(r"^\s*UTC\.?\s*$", re.IGNORECASE)
PUNC_RE   = re.compile(r"^\s*[\.\!\?]\s*$")
NUM_RE    = re.compile(r"^\s*[\d\W_]+\s*$")
WIKI_RE   = re.compile(r"Asked by Wiki User|Trivia Questions", re.IGNORECASE)

def _leading_fragment_reason(s: str) -> Optional[str]:
    """
    Detect obvious leading fragments like “'s ...” or one-letter splits.
    """
    s_stripped = s.lstrip()
    if not s_stripped:
        return None
    if re.match(r"^[’']s\b", s_stripped):  # "'s" or "’s"
        return "leading fragment ('s)"
    first_token = s_stripped.split(None, 1)[0]
    if len(first_token) == 1 and first_token.isalpha() and first_token.upper() not in ("I", "A"):
        if re.match(r"^\S+\s+\w", s_stripped):
            return "leading fragment (single-letter split)"
    return None

def _reject_reason(
    s: str,
    require_space: bool,
    min_chars: int,
    min_alpha_chars: int,
    require_keywords: Optional[List[str]],
    allow_short_no_space_len: int = 8,  # allow "Sure.", "Indeed." etc.
) -> Optional[str]:
    s_stripped = s.strip()
    if PUNC_RE.match(s_stripped):                  return "only punctuation"
    if DATE_RE.match(s_stripped):                  return "looks like a date"
    if TIME_RE.match(s_stripped):                  return "looks like a time"
    if UTC_RE.match(s_stripped):                   return "UTC token"
    if WIKI_RE.search(s_stripped):                 return "wiki/trivia footer"
    if NUM_RE.match(s_stripped):                   return "numeric or non-alphabetic"

    if len(s_stripped) < min_chars:                return f"too short (<{min_chars})"
    alpha = sum(ch.isalpha() for ch in s_stripped)
    if alpha < min_alpha_chars:                    return f"too few letters (<{min_alpha_chars})"

    if require_space and (" " not in s_stripped):
        if len(s_stripped) <= allow_short_no_space_len and s_stripped.endswith(('.', '!', '?')):
            pass
        else:
            return "no space"

    if require_keywords:
        s_low = s_stripped.lower()
        if not any(kw.lower() in s_low for kw in require_keywords):
            return f"missing required keywords {require_keywords}"
    return None

_STRIP_DOT_RE = re.compile(r"\.\s*$")
def _strip_trailing_period(s: str) -> str:
    return _STRIP_DOT_RE.sub("", s)


# ---------------------------------------------------------------------------
# HF forward-compat and EOS/space trimming
# ---------------------------------------------------------------------------
def _strip_trailing_eos(ids: List[int], eos_id: int) -> List[int]:
    j = len(ids)
    while j > 0 and ids[j-1] == eos_id:
        j -= 1
    return ids[:j]

def _strip_trailing_space_tokens(ids: List[int], tok: AutoTokenizer) -> List[int]:
    """
    Remove trailing tokens that decode to only whitespace (spaces/newlines/tabs).
    """
    j = len(ids)
    while j > 0:
        piece = tok.decode([ids[j-1]], skip_special_tokens=True)
        if piece == "" or piece.isspace():
            j -= 1
            continue
        break
    return ids[:j]

def _model_forward_compat(mdl, **kwargs):
    sig = signature(mdl.forward)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters and v is not None}
    return mdl(**filtered)

def _newline_token_ids(tok: AutoTokenizer) -> Set[int]:
    s: Set[int] = set()
    for seq in ("\n", "\r\n", "\n\n"):
        s.update(tok.encode(seq, add_special_tokens=False))
    return s


# ---------------------------------------------------------------------------
# Sentence tokenizer (spaCy -> NLTK -> regex) with numeric '.' rule
# ---------------------------------------------------------------------------
class SentenceTokenizer:
    _ABBREV = {
        "mr","mrs","ms","dr","prof","sr","jr","st","vs","etc",
        "e.g","i.e","u.s","u.k","no","fig","dept","inc","ltd",
        "jan","feb","mar","apr","jun","jul","aug","sep","sept",
        "oct","nov","dec",
    }

    _ENUM_LINE_RE = re.compile(r"^\s*\d{1,4}(?:[.)])\s*$")
    _COLON_ENUM_BLOCK_RE = re.compile(
        r"^(.*?:)\s*(?:\r?\n){1,}\s*\d{1,4}(?:[.)])\s*$",
        re.DOTALL,
    )

    def __init__(self, lang: str = "en"):
        self.backend = "regex"
        self.lang = lang
        self._nlp = None
        self._nltk_sent_tokenize = None
        try:
            import spacy
            import nltk
            try:
                nlp = spacy.load(f"{lang}_core_web_sm",
                                 disable=["ner","lemmatizer","textcat","tok2vec"])
                if "parser" not in nlp.pipe_names and "senter" not in nlp.pipe_names:
                    if "sentencizer" not in nlp.pipe_names:
                        nlp.add_pipe("sentencizer")
            except Exception:
                nlp = spacy.blank(lang)
                if "sentencizer" not in nlp.pipe_names:
                    nlp.add_pipe("sentencizer")
            self._nlp = nlp
            self.backend = "spacy"
        except Exception:
            try:
                from nltk.tokenize import sent_tokenize
                self._nltk_sent_tokenize = sent_tokenize
                self.backend = "nltk"
            except Exception:
                self.backend = "regex"

    def _is_abbrev_before(self, text: str, dot_idx: int) -> bool:
        j = dot_idx - 1
        while j >= 0 and text[j].isspace(): j -= 1
        k = j
        while k >= 0 and (text[k].isalpha() or text[k] == "."): k -= 1
        token = text[k+1:j+1]
        if not token: return False
        token_norm = token.strip().lower().rstrip(".")
        if not token_norm: return False
        if len(token_norm) == 1 and token.endswith("."):  # e.g., "A."
            return True
        return token_norm in self._ABBREV

    def _postfix_enumeration_rules(self, candidate: str) -> str | None:
        s = candidate.strip()
        if self._ENUM_LINE_RE.match(s):
            return None
        m = self._COLON_ENUM_BLOCK_RE.match(s)
        if m:
            return m.group(1).strip()
        return s  # unchanged

    # --- Numeric helpers ---
    def _is_second_dot_in_number(self, text: str, dot_idx: int) -> bool:
        j = dot_idx - 1
        while j >= 0 and text[j].isspace():
            j -= 1
        if j < 0 or not text[j].isdigit():
            return False
        k = j
        seen_prev_dot = False
        while k >= 0 and (text[k].isdigit() or text[k] == "."):
            if text[k] == ".":
                seen_prev_dot = True
            k -= 1
        return seen_prev_dot

    def _ends_with_single_number_dot(self, s: str) -> bool:
        if not s:
            return False
        t = s.rstrip()
        while t and t[-1] in "\"')]}":
            t = t[:-1].rstrip()
        if not t or t[-1] != ".":
            return False
        if len(t) < 2 or not t[-2].isdigit():
            return False
        k = len(t) - 2
        seen_prev_dot = False
        while k >= 0 and (t[k].isdigit() or t[k] == "."):
            if t[k] == ".":
                seen_prev_dot = True
            k -= 1
        return not seen_prev_dot

    # --- core detection (regex backend) ---
    def _regex_first_complete(self, text: str) -> Optional[str]:
        if not text:
            return None

        # Allow colon+newline boundary
        m_colon = re.search(r":\s*(?:\r?\n)", text)
        if m_colon:
            cand = text[:m_colon.start()+1].strip()
            fixed = self._postfix_enumeration_rules(cand)
            return fixed if fixed else None

        n = len(text)
        for i, ch in enumerate(text):
            if ch not in (".", "!", "?"):
                continue
            # Abbreviation like "Dr.", "e.g."
            if ch == "." and self._is_abbrev_before(text, i):
                continue
            # Numeric rule: '.' after a digit is EOS only if it's the second '.' in the same number
            if ch == ".":
                j = i - 1
                while j >= 0 and text[j].isspace():
                    j -= 1
                if j >= 0 and text[j].isdigit():
                    if not self._is_second_dot_in_number(text, i):
                        continue

            # Allow closing quotes/brackets to trail punctuation
            end = i + 1
            while end < n and text[end] in "\"')]}":
                end += 1
            if end >= n or text[end].isspace():
                cand = text[:end].strip()
                fixed = self._postfix_enumeration_rules(cand)
                if fixed and not self._ends_with_single_number_dot(fixed):
                    return fixed

        # If whole text ends with terminal punctuation, apply same post-check
        if re.search(r"[.!?][\"')\]]*\s*$", text):
            cand = text.strip()
            fixed = self._postfix_enumeration_rules(cand)
            if fixed and not self._ends_with_single_number_dot(fixed):
                return fixed
        return None

    def first_complete(self, text: str) -> Optional[str]:
        if not text:
            return None

        if self.backend == "spacy":
            doc = self._nlp(text)
            sents = list(doc.sents)
            if not sents:
                return None
            if len(sents) > 1:
                cand = sents[0].text.strip()
                fixed = self._postfix_enumeration_rules(cand)
                if fixed and not self._ends_with_single_number_dot(fixed):
                    return fixed
                return None
            first = sents[0].text
            if re.search(r"[.!?][\"')\]]*\s*$", first) or re.search(r":\s*(?:\r?\n)", first):
                cand = first.strip()
                fixed = self._postfix_enumeration_rules(cand)
                if fixed and not self._ends_with_single_number_dot(fixed):
                    return fixed
            return None

        if self.backend == "nltk":
            sents = self._nltk_sent_tokenize(text)
            if not sents:
                return None
            if len(sents) > 1:
                cand = sents[0].strip()
                fixed = self._postfix_enumeration_rules(cand)
                if fixed and not self._ends_with_single_number_dot(fixed):
                    return fixed
                return None
            first = sents[0]
            if re.search(r"[.!?][\"')\]]*\s*$", first) or re.search(r":\s*(?:\r?\n)", first):
                cand = first.strip()
                fixed = self._postfix_enumeration_rules(cand)
                if fixed and not self._ends_with_single_number_dot(fixed):
                    return fixed
            return None

        # regex fallback
        return self._regex_first_complete(text)


# ---------------------------------------------------------------------------
# Chat template helpers
# ---------------------------------------------------------------------------
def _render_for_generation(tok: AutoTokenizer, messages: List[Dict]) -> List[int]:
    return tok.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    )[0].tolist()

def _render_for_classification(tok: AutoTokenizer, messages_prefix: List[Dict], assistant_text: str) -> str:
    msgs = copy.deepcopy(messages_prefix)
    if not msgs or msgs[-1].get("role") != "assistant":
        msgs.append({"role": "assistant", "content": assistant_text})
    else:
        msgs[-1]["content"] = assistant_text
    rendered = tok.apply_chat_template(
        msgs,
        add_generation_prompt=False,
        tokenize=False,
    )
    return rendered


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
def _get_model_and_tokenizer(
    model_path: str,
    device: str,
    torch_dtype,
    max_memory: Optional[Dict[str, str]],
    log: logging.Logger,
):
    global _TOK, _MDL, _MODEL_PATH
    if _MDL is not None and _MODEL_PATH == model_path:
        return _TOK, _MDL

    hf_kwargs = dict(
        torch_dtype=torch_dtype,
        device_map=None if device == "cpu" else "auto",
        low_cpu_mem_usage=True,
    )
    if max_memory is not None:
        hf_kwargs["max_memory"] = max_memory

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id

    mdl = AutoModelForCausalLM.from_pretrained(model_path, **hf_kwargs)
    if device == "cpu":
        mdl.to("cpu")
    mdl.eval()

    _TOK, _MDL, _MODEL_PATH = tok, mdl, model_path
    if log is not None:
        log.info("Loaded HF model/tokenizer and cached for reuse.")
    return tok, mdl

def _get_saplma(bundle_path: str, log: logging.Logger):
    global _SAPLMA, _BUNDLE_PATH
    if _SAPLMA is not None and _BUNDLE_PATH == bundle_path:
        return _SAPLMA

    import tensorflow as tf
    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass
    from saplma_api import load_best_bundle, embed_text_last_token_with_loaded_model

    saplma_model, thr_opt, meta, _ = load_best_bundle(bundle=bundle_path)
    layer_from_end = int(meta["layer_from_end"])
    thr_opt_base = float(thr_opt)

    _SAPLMA = (saplma_model, embed_text_last_token_with_loaded_model, layer_from_end, thr_opt_base)
    _BUNDLE_PATH = bundle_path
    log.info("Loaded SAPLMA bundle and cached for reuse.")
    return _SAPLMA


# ---------------------------------------------------------------------------
# First-token hygiene helpers
# ---------------------------------------------------------------------------
def _bad_first_piece(piece: str) -> bool:
    s = piece.lstrip()
    if not s:
        return True
    if s[:1] in ("\n", "\r"):
        return True
    if re.match(r"^[’']s\b", s):
        return True
    if re.match(r"^[,:;)\]]$", s):
        return True
    if len(s) == 1 and s.isalpha() and s.upper() not in ("I", "A"):
        return True
    if s and not s[0].isascii() and s[0].isalpha():
        return True
    if re.match(r"^[\.\!\?\-\–\—\'\"]+$", s):
        return True
    return False


# ---------------------------------------------------------------------------
# Core generation loop
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_with_saplma_guardrail(
    prompt: str,
    bundle_path: str,
    model_path: str,
    mdl: AutoModelForCausalLM | None,
    tok: AutoTokenizer | None,

    # Decoding policy
    decode_mode: str = "hybrid",     # {"greedy","hybrid"}
    temperature: float = 1.1,
    top_p: float = 0.8,
    min_tokens_to_keep: int = 5,

    # Classification scope
    classification_mode: str = "sentence",  # {"sentence","cumulative"}
    cumulative_exclude_prompt: bool = False,  # compat; no prompt echo now
    strip_enumeration_for_class: bool = True,

    # SAPLMA input normalization
    classify_strip_trailing_period: bool = True,

    # Budgets
    max_sentences: int = 20,
    max_new_tokens_total: int = 512,
    max_tokens_per_sentence: int = 64,
    retries_per_sentence: int = 5,

    # Content filters (on *new* sentence only)
    min_sentence_chars: int = 20,
    min_alpha_chars: int = 8,
    require_space_in_sentence: bool = True,
    require_keywords: Optional[List[str]] = None,

    # Last-retry relaxation
    relax_filters_on_last_retry: bool = True,

    # Acceptance strictness
    reject_leading_fragments: bool = True,     # drop "'s ..." and single-letter split starts

    # SAPLMA threshold controls
    threshold_offset: float = 0.0,             # additive offset to bundle's optimal threshold
    saplma_threshold: Optional[float] = None,  # if provided, overrides (clamped to [0,1])

    # System
    device: str = "auto",
    log_level: str = "INFO",

    # NEW: rich results
    return_details: bool = False,

    # Optional memory cap during HF load (e.g., {"cuda:0": "70%"})
    max_memory: Optional[Dict[str, str]] = None,

    # First-token resampling cap
    first_token_max_resamples: int = 5,

    # Optional debug: log token traces at DEBUG level
    debug_token_trace: bool = False,
) -> str | GuardedGenerationResult:
    if decode_mode not in ("greedy", "hybrid"):
        raise ValueError("decode_mode must be 'greedy' or 'hybrid'")
    if classification_mode not in ("sentence", "cumulative"):
        raise ValueError("classification_mode must be 'sentence' or 'cumulative'")

    log = _setup_logger(log_level)
    t0 = time.time()

    # -- Load / reuse tokenizer & model
    torch_dtype = torch.float16 if torch.cuda.is_available() and device != "cpu" else torch.float32
    if model_path is not None:
        tok, mdl = _get_model_and_tokenizer(model_path, device, torch_dtype, max_memory, log)

    cfg_max_ctx = getattr(mdl.config, "max_position_embeddings", 4096)
    eos_id = tok.eos_token_id
    NL_IDS = _newline_token_ids(tok)

    # -- Load / reuse SAPLMA bundle
    saplma_model, embed_text_last_token_with_loaded_model, layer_from_end, thr_opt_base = _get_saplma(bundle_path, log)

    # -- Threshold selection (override wins; else base+offset), clamped conservatively
    if saplma_threshold is not None:
        thr_use = float(min(max(float(saplma_threshold), 0.0), 0.7))
        log.info("SAPLMA threshold  : %.4f (override provided; base=%.4f, offset=%+.4f ignored)",
                 thr_use, thr_opt_base, float(threshold_offset))
    else:
        thr_use = float(min(max(thr_opt_base + float(threshold_offset), 0.0), 0.7))
        log.info("SAPLMA threshold  : %.4f (base=%.4f, offset=%+.4f)",
                 thr_use, thr_opt_base, float(threshold_offset))

    # -- Build chat messages (no prompt echo)
    accepted_text = ""  # assistant-only running content
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": accepted_text},
    ]
    messages_prefix = copy.deepcopy(messages)

    # -- Sentence tokenizer (reuse across calls)
    global _SENT_TOK
    if _SENT_TOK is None:
        _SENT_TOK = SentenceTokenizer(lang="en")
    sent_tok = _SENT_TOK
    log.info("Sentence tokenizer backend: %s", sent_tok.backend)

    # -- Diagnostics
    log.info("======== SAPLMA Guarded Generation ========")
    log.info("Prompt            : %s", prompt)
    log.info("Base model path   : %s", model_path)
    log.info("Bundle            : %s", bundle_path)
    log.info("Layer from end    : %d", layer_from_end)
    log.info("Context window    : %d", cfg_max_ctx)
    log.info("Decoding          : %s", "GREEDY" if decode_mode=="greedy" else "HYBRID")
    log.info("Limits            : %d sentences | %d total tokens | %d tokens/sentence | %d retries/sentence",
             max_sentences, max_new_tokens_total, max_tokens_per_sentence, retries_per_sentence)

    # -- State
    accepted_tokens_total = 0
    sentences_accepted = 0
    sentences_rejected = 0
    trivial_skips = 0
    eos_early = False
    stopped_by_budget = False

    accepted_ids = _render_for_generation(tok, messages)
    accepted_ids = _strip_trailing_eos(accepted_ids, eos_id)            # ensure no EOS in starting context
    accepted_ids = _strip_trailing_space_tokens(accepted_ids, tok)      # trim trailing '▁'/whitespace

    banned_starts: Set[int] = set()
    banned_pairs: Set[Tuple[int, int]] = set()

    rejected_events: List[SAPLMARejected] = []
    accepted_texts: List[str] = []
    accepted_events: List[SAPLMAAccepted] = []

    TRIVIAL_REASONS = (
        "only punctuation",
        "too short",
        "no space",
        "numeric or non-alphabetic",
        "no terminal punctuation",
    )

    # -- Main loop
    for s_idx in range(max_sentences):
        accepted_this_sentence = False

        for attempt in range(retries_per_sentence):
            cur_ids: List[int] = []
            cur_tokens = 0
            first_token_id: Optional[int] = None
            first_bigram: Optional[Tuple[int, int]] = None
            last_attempt_full_sample = (decode_mode == "hybrid" and attempt == retries_per_sentence - 1)
            first_piece_resamples = 0

            while True:
                if accepted_tokens_total + cur_tokens >= max_new_tokens_total:
                    stopped_by_budget = True
                    log.warning("[STOP] Reached max_new_tokens_total=%d", max_new_tokens_total)
                    return _finish(
                        accepted_text, log, t0,
                        sentences_accepted, sentences_rejected, trivial_skips,
                        eos_early, stopped_by_budget,
                        return_details=return_details,
                        accepted_texts=accepted_texts,
                        rejected_events=rejected_events,
                        accepted_events=accepted_events,
                    )

                # Build context (accepted_ids + in-progress sentence)
                ctx_ids = accepted_ids + cur_ids

                # Debug token traces (optional)
                if debug_token_trace and cur_tokens == 0:
                    log.debug("CTX(tokens): %s | INPROG: %s",
                              tok.convert_ids_to_tokens(accepted_ids),
                              tok.convert_ids_to_tokens(cur_ids))
                if len(ctx_ids) > (cfg_max_ctx - 1):
                    ctx_ids = ctx_ids[-(cfg_max_ctx - 1):]

                input_ids = torch.tensor([ctx_ids], device=mdl.device)
                attn_mask = torch.ones_like(input_ids)

                out = _model_forward_compat(
                    mdl,
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    use_cache=True
                )
                next_logits = out.logits[0, -1]

                # Apply bans
                if cur_tokens == 0 and banned_starts:
                    next_logits[list(banned_starts)] = float("-inf")
                elif cur_tokens == 1 and first_token_id is not None and banned_pairs:
                    for t1, t2 in banned_pairs:
                        if t1 == first_token_id:
                            next_logits[t2] = float("-inf")

                # NEW: don't start a sentence with a bare newline token
                if cur_tokens == 0 and NL_IDS:
                    for _id in NL_IDS:
                        next_logits[_id] = float("-inf")

                # Choose next token:
                if cur_tokens == 0:
                    mode = "SAMPLE"
                    candidate_id = _nucleus_sample(
                        next_logits, top_p=top_p, temperature=temperature,
                        min_tokens_to_keep=min_tokens_to_keep
                    )
                    while first_piece_resamples < first_token_max_resamples:
                        piece = tok.decode([candidate_id], skip_special_tokens=True)
                        if not _bad_first_piece(piece):
                            break
                        candidate_id = _nucleus_sample(
                            next_logits, top_p=top_p, temperature=temperature,
                            min_tokens_to_keep=min_tokens_to_keep
                        )
                        first_piece_resamples += 1
                    next_id = candidate_id
                else:
                    if decode_mode == "greedy":
                        mode = "GREEDY"
                        next_id = int(torch.argmax(next_logits))
                    else:
                        if last_attempt_full_sample:
                            mode = "SAMPLE"
                            next_id = _nucleus_sample(
                                next_logits, top_p=top_p, temperature=temperature,
                                min_tokens_to_keep=min_tokens_to_keep
                            )
                        else:
                            mode = "GREEDY"
                            next_id = int(torch.argmax(next_logits))

                if cur_tokens == 0:
                    first_token_id = next_id

                # EOS -> finish
                if next_id == eos_id:
                    eos_early = True
                    log.info("[EOS] encountered - returning final answer.]")
                    return _finish(
                        accepted_text, log, t0,
                        sentences_accepted, sentences_rejected, trivial_skips,
                        eos_early, stopped_by_budget,
                        return_details=return_details,
                        accepted_texts=accepted_texts,
                        rejected_events=rejected_events,
                        accepted_events=accepted_events,
                    )

                cur_ids.append(next_id)
                cur_tokens += 1
                accepted_tokens_total += 1

                if cur_tokens == 2 and first_token_id is not None:
                    first_bigram = (first_token_id, next_id)

                # ---- Sentence boundary detection via tokenizer ----
                cur_text = tok.decode(cur_ids, skip_special_tokens=True)
                sentence_out = sent_tok.first_complete(cur_text)

                if sentence_out is not None:
                    clean_sentence = " ".join(sentence_out.replace("\n", " ").split()).strip()

                    # Build classification content
                    if classification_mode == "sentence":
                        cls_content = clean_sentence
                    else:
                        cls_content = (accepted_text + ("" if accepted_text.endswith((" ", "\n")) or clean_sentence.startswith(" ")
                                                        else " ") + clean_sentence)

                    if strip_enumeration_for_class:
                        cls_content = ENUM_PREFIX_RE.sub("", cls_content).strip()
                    if classify_strip_trailing_period:
                        cls_content = _strip_trailing_period(cls_content)

                    classify_text_str = _render_for_classification(tok, messages_prefix, cls_content)

                    # Content filters on NEW sentence
                    to_filter = ENUM_PREFIX_RE.sub("", clean_sentence).strip() if strip_enumeration_for_class else clean_sentence

                    if reject_leading_fragments:
                        frag_reason = _leading_fragment_reason(to_filter)
                        if frag_reason:
                            rejected_events.append(SAPLMARejected(
                                type="filter",
                                attempt=attempt,
                                mode=mode,
                                sentence_out=sentence_out,
                                clean_sentence=to_filter,
                                classify_text=classify_text_str,
                                prob_true=None,
                                threshold=None,
                                reason=frag_reason
                            ))
                            trivial_skips += 1
                            if first_bigram is not None:
                                banned_pairs.add(first_bigram)
                            break

                    reason = _reject_reason(
                        to_filter,
                        require_space=require_space_in_sentence,
                        min_chars=min_sentence_chars,
                        min_alpha_chars=min_alpha_chars,
                        require_keywords=require_keywords,
                    )

                    # Relax some filters on last retry
                    if reason and relax_filters_on_last_retry and last_attempt_full_sample:
                        if reason.startswith("too short") or reason == "no space":
                            reason = None

                    if reason:
                        rejected_events.append(SAPLMARejected(
                            type="filter",
                            attempt=attempt,
                            mode=mode,
                            sentence_out=sentence_out,
                            clean_sentence=to_filter,
                            classify_text=classify_text_str,
                            prob_true=None,
                            threshold=None,
                            reason=reason
                        ))
                        trivial_skips += 1
                        trivial = reason.startswith(TRIVIAL_REASONS)
                        if first_token_id is not None and not trivial:
                            banned_starts.add(first_token_id)
                        if first_bigram is not None and not trivial:
                            banned_pairs.add(first_bigram)
                        break

                    # ---- SAPLMA classification ----
                    emb = embed_text_last_token_with_loaded_model(tok, mdl, classify_text_str,
                                                                  layer_from_end, max_length=cfg_max_ctx)
                    X = emb.unsqueeze(0).cpu().numpy().astype("float32")
                    prob_true = float(saplma_model.predict(X, verbose=0).ravel()[0])

                    is_true = prob_true > thr_use

                    # Console + log for visibility
                    print(f"[SENTENCE] scope={classification_mode} | mode={mode} | prob_true={prob_true:.4f} thr={thr_use:.4f} -> {'ACCEPT' if is_true else 'REJECT'}")
                    log.info("[SENTENCE] scope=%s | mode=%s | prob_true=%.4f thr=%.4f -> %s",
                             classification_mode, mode, prob_true, thr_use,
                             "ACCEPT" if is_true else "REJECT")

                    if is_true:
                        # Append to final text without adding a leading space at the start.
                        out_to_add = sentence_out
                        if accepted_text:
                            if not out_to_add.startswith((" ", "\n")):
                                accepted_text += " " + out_to_add
                            else:
                                accepted_text += out_to_add
                        else:
                            accepted_text += out_to_add.lstrip()

                        messages[-1]["content"] = accepted_text
                        messages_prefix[-1]["content"] = ""

                        accepted_ids = _render_for_generation(tok, messages)
                        accepted_ids = _strip_trailing_eos(accepted_ids, eos_id)       # keep EOS out
                        accepted_ids = _strip_trailing_space_tokens(accepted_ids, tok) # trim trailing '▁'/whitespace

                        # Record rich details for accepted sentence
                        accepted_events.append(SAPLMAAccepted(
                            type="saplma",
                            attempt=attempt,
                            mode=mode,
                            sentence_out=sentence_out,
                            clean_sentence=clean_sentence,
                            classify_text=classify_text_str,
                            prob_true=prob_true,
                            threshold=thr_use,
                            reason="prob>=thr"
                        ))

                        sentences_accepted += 1
                        accepted_this_sentence = True
                        accepted_texts.append(out_to_add.strip())
                    else:
                        sentences_rejected += 1
                        rejected_events.append(SAPLMARejected(
                            type="saplma",
                            attempt=attempt,
                            mode=mode,
                            sentence_out=sentence_out,
                            clean_sentence=clean_sentence,
                            classify_text=classify_text_str,
                            prob_true=prob_true,
                            threshold=thr_use,
                            reason="prob<thr"
                        ))
                        if first_token_id is not None:
                            banned_starts.add(first_token_id)
                        if first_bigram is not None:
                            banned_pairs.add(first_bigram)
                    break  # handled a sentence (accepted or rejected)

                # Guard against run-ons
                if cur_tokens >= max_tokens_per_sentence:
                    sentences_rejected += 1
                    log.info("[RETRY] sentence exceeded max_tokens_per_sentence=%d -> reject & retry | mode=%s",
                             max_tokens_per_sentence, mode)
                    txt = tok.decode(cur_ids, skip_special_tokens=True)
                    rejected_events.append(SAPLMARejected(
                        type="runon",
                        attempt=attempt,
                        mode=mode,
                        sentence_out=txt,
                        clean_sentence=txt,
                        classify_text="",
                        prob_true=None,
                        threshold=None,
                        reason="exceeded max_tokens_per_sentence"
                    ))
                    if first_token_id is not None:
                        banned_starts.add(first_token_id)
                    if first_bigram is not None:
                        banned_pairs.add(first_bigram)
                    break  # retry with new opening

            if accepted_this_sentence:
                break  # next sentence

        if not accepted_this_sentence:
            log.warning("[GIVEUP] Could not produce an acceptable sentence after %d attempts.", retries_per_sentence)
            return _finish(
                accepted_text, log, t0,
                sentences_accepted, sentences_rejected, trivial_skips,
                eos_early, stopped_by_budget,
                return_details=return_details,
                accepted_texts=accepted_texts,
                rejected_events=rejected_events,
                accepted_events=accepted_events,
            )

    log.info("[DONE] Max sentences reached.]")
    return _finish(
        accepted_text, log, t0,
        sentences_accepted, sentences_rejected, trivial_skips,
        eos_early, stopped_by_budget,
        return_details=return_details,
        accepted_texts=accepted_texts,
        rejected_events=rejected_events,
        accepted_events=accepted_events,
    )


def _finish(text: str, log: logging.Logger, t0: float,
            sentences_accepted: int, sentences_rejected: int,
            trivial_skips: int, eos_early: bool, stopped_by_budget: bool,
            return_details: bool = False,
            accepted_texts: List[str] | None = None,
            rejected_events: List[SAPLMARejected] | None = None,
            accepted_events: List[SAPLMAAccepted] | None = None
            ) -> str | GuardedGenerationResult:
    elapsed = time.time() - t0
    log.info("======== SUMMARY ========")
    log.info("Accepted sentences : %d", sentences_accepted)
    log.info("Rejected sentences : %d", sentences_rejected)
    log.info("Trivial skips      : %d", trivial_skips)
    log.info("Stopped by EOS     : %s", eos_early)
    log.info("Stopped by budget  : %s", stopped_by_budget)
    log.info("Elapsed            : %.2fs", elapsed)
    log.info("Final answer       : %s", text)

    if not return_details:
        return text

    return GuardedGenerationResult(
        final_text=text,
        accepted_sentences=accepted_events or [],
        rejected=rejected_events or [],
        summary={
            "accepted": sentences_accepted,
            "rejected": sentences_rejected,
            "skipped": trivial_skips,
            "eos_early": eos_early,
            "stopped_by_budget": stopped_by_budget,
            "elapsed_sec": round(elapsed, 3),
        },
        accepted_texts=accepted_texts or [],
    )


# Example manual run
if __name__ == "__main__":
    PROMPT = "Tell me some true facts about France."
    MODEL  = "../models/Llama-2-7b-chat-hf"
    BUNDLE = "../pretrained_saplma/instruct/format_3/saplma_checkpoints_LLAMA7/BEST_layer12__heldout_data/capitals"

    out = generate_with_saplma_guardrail(
        prompt=PROMPT,
        bundle_path=BUNDLE,
        model_path=MODEL,
        mdl=None, tok=None,  # loaders handle caching
        decode_mode="hybrid",
        classification_mode="cumulative",
        max_sentences=3,
        retries_per_sentence=8,
        max_new_tokens_total=256,
        max_tokens_per_sentence=64,
        min_sentence_chars=5,
        min_alpha_chars=3,
        require_space_in_sentence=True,
        top_p=0.8,
        temperature=0.8,
        min_tokens_to_keep=5,
        relax_filters_on_last_retry=True,
        threshold_offset=0.0,
        # saplma_threshold=0.85,
        device="auto",
        log_level="INFO",
        return_details=True,
        first_token_max_resamples=5,
        debug_token_trace=False,
    )
    print("\n=== FINAL (GUARDED) ===")
    if isinstance(out, GuardedGenerationResult):
        import json
        # Attribute access works (objects):
        print("Rejected count:", len(out.rejected))
        if out.rejected:
            r0 = out.rejected[0]
            print("First rejected.type:", r0.type)

        # JSON-safe dump:
        print(json.dumps(out.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(out)
