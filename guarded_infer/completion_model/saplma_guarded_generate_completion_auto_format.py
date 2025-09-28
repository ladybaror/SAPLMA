# saplma_guarded_generate_completion_auto.py
# ---------------------------------------------------------------------------
# Guardrailed generation with SAPLMA (Simple Accuracy Prediction via Last-token
# Model Activation) for sentence-by-sentence filtering, template-friendly.
# Loads the HF model first (Torch on GPU), then imports Keras (TF) on CPU only.
# ---------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Set, List, Tuple, Dict
import copy
import logging, time, re, os, sys
from inspect import signature

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

# Ensure TF is quiet if it ends up imported anywhere before we pin it to CPU.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# If your guarded_infer package is relative to this file, keep this:
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ---------------------------------------------------------------------------
# Dataclasses for detailed outputs
# ---------------------------------------------------------------------------
@dataclass
class SAPLMARejected:
    type: str                     # "filter" | "saplma" | "runon"
    attempt: int                  # which attempt (0-based)
    mode: str                     # "GREEDY" | "SAMPLE"
    sentence_out: str             # exact punctuated sentence model produced (maybe partial on runon)
    clean_sentence: str           # whitespace-normalized sentence used for filters
    classify_text: str            # full rendered chat-template string fed to SAPLMA
    prob_true: float | None       # SAPLMA probability (None for filter/runon)
    threshold: float | None       # threshold used (None for filter/runon)
    reason: str | None            # e.g., "too short", "prob<thr", "exceeded max_tokens_per_sentence"

@dataclass
class GuardedGenerationResult:
    final_text: str
    accepted_sentences: List[str]
    rejected: List[SAPLMARejected]
    summary: Dict[str, int | float | bool | str]


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

def _reject_reason(
    s: str,
    require_space: bool,
    min_chars: int,
    min_alpha_chars: int,
    require_keywords: Optional[List[str]],
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
    if require_space and (" " not in s_stripped):  return "no space"

    if require_keywords:
        s_low = s_stripped.lower()
        if not any(kw.lower() in s_low for kw in require_keywords):
            return f"missing required keywords {require_keywords}"
    return None

_STRIP_DOT_RE = re.compile(r"\.\s*$")
def _strip_trailing_period(s: str) -> str:
    return _STRIP_DOT_RE.sub("", s)


# ---------------------------------------------------------------------------
# HF forward-compat and EOS trim
# ---------------------------------------------------------------------------
def _strip_trailing_eos(ids: List[int], eos_id: int) -> List[int]:
    """Remove 1+ trailing EOS tokens from a token id list."""
    j = len(ids)
    while j > 0 and ids[j-1] == eos_id:
        j -= 1
    return ids[:j]

def _model_forward_compat(mdl, **kwargs):
    """
    Call model.forward with only supported kwargs.
    Works across HF variants (including ones with new cache APIs).
    """
    sig = signature(mdl.forward)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters and v is not None}
    return mdl(**filtered)


# ---------------------------------------------------------------------------
# Helpers for chat template usage
# ---------------------------------------------------------------------------
def _render_for_generation(tok: AutoTokenizer, messages: List[Dict]) -> List[int]:
    """
    Render chat messages to input_ids ready for generation.
    add_generation_prompt=True appends the template's 'assistant:' cue so the model
    continues the assistant turn.
    """
    return tok.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    )[0].tolist()  # shape [T]

def _render_for_classification(tok: AutoTokenizer, messages_prefix: List[Dict], assistant_text: str) -> str:
    """
    Render a *complete* chat sample (no generation prompt) whose last token should be
    the end of the assistant_text. We return a STRING so the SAPLMA embedder can
    tokenize it the same way as the base model.
    """
    msgs = copy.deepcopy(messages_prefix)
    # Ensure there is an assistant turn
    if not msgs or msgs[-1].get("role") != "assistant":
        msgs.append({"role": "assistant", "content": assistant_text})
    else:
        msgs[-1]["content"] = assistant_text

    rendered = tok.apply_chat_template(
        msgs,
        add_generation_prompt=False,  # crucial: full completed assistant text
        tokenize=False,               # return a str (not ids); embedder will tokenize
    )
    return rendered


# ---------------------------------------------------------------------------
# Core generation loop
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_with_saplma_guardrail(
    prompt: str,
    bundle_path: str,
    model_path: str,

    # Decoding policy
    decode_mode: str = "hybrid",     # {"greedy","hybrid"}
    temperature: float = 0.8,
    top_p: float = 0.8,
    min_tokens_to_keep: int = 5,

    # Classification scope
    classification_mode: str = "sentence",  # {"sentence","cumulative"}
    cumulative_exclude_prompt: bool = False,
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

    # SAPLMA threshold tweak
    threshold_offset: float = 0.0,

    # System
    device: str = "auto",
    log_level: str = "INFO",
    format: int = 3,

    # NEW: rich results
    return_details: bool = False,

    # Optional memory cap during HF load (e.g., {"cuda:0": "70%"})
    max_memory: Optional[Dict[str, str]] = None,
) -> str | GuardedGenerationResult:
    if decode_mode not in ("greedy", "hybrid"):
        raise ValueError("decode_mode must be 'greedy' or 'hybrid'")
    if classification_mode not in ("sentence", "cumulative"):
        raise ValueError("classification_mode must be 'sentence' or 'cumulative'")

    log = _setup_logger(log_level)
    t0 = time.time()

    # ---------------- Load tokenizer / model (Torch grabs GPU first) ----------------
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id

    torch_dtype = torch.float16 if torch.cuda.is_available() and device != "cpu" else torch.float32

    hf_kwargs = dict(
        torch_dtype=torch_dtype,
        device_map=None if device == "cpu" else "auto",
        low_cpu_mem_usage=True,
    )
    if max_memory is not None:
        hf_kwargs["max_memory"] = max_memory

    mdl = AutoModelForCausalLM.from_pretrained(
        model_path,
        **hf_kwargs
    )
    if device == "cpu":
        mdl.to("cpu")
    mdl.eval()

    cfg_max_ctx = getattr(mdl.config, "max_position_embeddings", 4096)
    eos_id = tok.eos_token_id

    # ---------------- NOW import SAPLMA bits and force TF to CPU ----------------
    # Do this *after* the HF model has claimed the GPU.
    def _load_saplma_bundle_cpu_only(bundle_path: str):
        # Pin TensorFlow to CPU (no visible GPUs)
        import tensorflow as tf
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass
        from guarded_infer.saplma_api import load_best_bundle, embed_text_last_token_with_loaded_model
        return load_best_bundle, embed_text_last_token_with_loaded_model

    load_best_bundle, embed_text_last_token_with_loaded_model = _load_saplma_bundle_cpu_only(bundle_path)

    # ---------------- Load SAPLMA bundle ----------------
    saplma_model, thr_opt, meta, _ = load_best_bundle(bundle=bundle_path)
    layer_from_end = int(meta["layer_from_end"])
    thr_use = float(min(max(float(thr_opt) + float(threshold_offset), 0.0), 1.0))

    # ---------------- Build chat messages ----------------
    accepted_text = prompt if prompt.endswith((" ", "\n")) else prompt + " "
    initial_accepted_text = accepted_text

    if format == 1:
        messages = [{"role": "assistant", "content": accepted_text}]
    elif format == 2:
        messages = [{"role": "user", "content": ""}, 
                    {"role": "assistant", "content": accepted_text}]
    elif format == 3:
        messages = [{"role": "user", "content": "Tell me a true fact"}, 
                    {"role": "assistant", "content": accepted_text}]
    else:
        messages = [{"role": "user", "content": "Tell me a false fact"}, 
                    {"role": "assistant", "content": accepted_text}]

    # For classification rendering we need a prefix copy we’ll mutate each time
    messages_prefix = copy.deepcopy(messages)

    # ---------------- Diagnostics header ----------------
    log.info("======== SAPLMA Guarded Generation ========")
    log.info("Prompt            : %s", prompt)
    log.info("Base model path   : %s", model_path)
    log.info("Bundle            : %s", bundle_path)
    log.info("Layer from end    : %d", layer_from_end)
    log.info("SAPLMA threshold  : %.4f (best=%.4f, offset=%+.4f)", thr_use, float(thr_opt), float(threshold_offset))
    log.info("Context window    : %d", cfg_max_ctx)
    log.info("Decoding          : %s", "GREEDY" if decode_mode=="greedy" else "HYBRID (sample first token; last retry sampled)")
    log.info("Classify scope    : %s%s", classification_mode.upper(),
             " (exclude_prompt)" if (classification_mode=="cumulative" and cumulative_exclude_prompt) else "")
    log.info("Device            : %s | dtype=%s", str(next(mdl.parameters()).device), str(torch_dtype))
    log.info("Limits            : %d sentences | %d total tokens | %d tokens/sentence | %d retries/sentence",
             max_sentences, max_new_tokens_total, max_tokens_per_sentence, retries_per_sentence)

    # ---------------- State ----------------
    accepted_tokens_total = 0
    sentences_accepted = 0
    sentences_rejected = 0
    trivial_skips = 0
    eos_early = False
    stopped_by_budget = False

    accepted_ids = _render_for_generation(tok, messages)
    accepted_ids = _strip_trailing_eos(accepted_ids, eos_id)

    SENT_END_CHARS = (".", "!", "?")
    banned_starts: Set[int] = set()
    banned_pairs: Set[Tuple[int, int]] = set()

    rejected_events: List[SAPLMARejected] = []
    accepted_sentences_list: List[str] = []

    # ---------------- Main loop over sentences ----------------
    for s_idx in range(max_sentences):
        accepted_this_sentence = False
        # Optional prints for debugging:
        # print(("Starting guarded generation for sentence %d", s_idx))

        for attempt in range(retries_per_sentence):
            # print(("Attempt %d", attempt))

            cur_ids: List[int] = []
            cur_tokens = 0
            first_token_id: Optional[int] = None
            first_bigram: Optional[Tuple[int, int]] = None
            last_attempt_full_sample = (decode_mode == "hybrid" and attempt == retries_per_sentence - 1)

            while True:
                # print("looping")

                if accepted_tokens_total + cur_tokens >= max_new_tokens_total:
                    stopped_by_budget = True
                    log.warning(f"[STOP] Reached max_new_tokens_total={max_new_tokens_total}")
                    return _finish(
                        accepted_text, log, t0,
                        sentences_accepted, sentences_rejected, trivial_skips,
                        eos_early, stopped_by_budget,
                        return_details=return_details,
                        accepted_sentences_list=accepted_sentences_list,
                        rejected_events=rejected_events
                    )

                # Build context (accepted_ids + in-progress sentence)
                ctx_ids = accepted_ids + cur_ids
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

                # Choose next token (hybrid)
                if decode_mode == "greedy":
                    mode = "GREEDY"
                    next_id = int(torch.argmax(next_logits))
                else:
                    if last_attempt_full_sample or cur_tokens == 0:
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

                # EOS -> finish immediately
                if next_id == eos_id:
                    eos_early = True
                    log.info("[EOS] encountered - returning final answer.")
                    return _finish(
                        accepted_text, log, t0,
                        sentences_accepted, sentences_rejected, trivial_skips,
                        eos_early, stopped_by_budget,
                        return_details=return_details,
                        accepted_sentences_list=accepted_sentences_list,
                        rejected_events=rejected_events
                    )

                cur_ids.append(next_id)
                cur_tokens += 1
                accepted_tokens_total += 1

                if (cur_tokens % 16) == 0:
                    piece = tok.decode([next_id], skip_special_tokens=True)
                    log.debug(f"[t={cur_tokens}] last_token={piece} ...")

                if cur_tokens == 2 and first_token_id is not None:
                    first_bigram = (first_token_id, next_id)

                # Check for sentence boundary on the *newly generated* text
                cur_text = tok.decode(cur_ids, skip_special_tokens=True)
                end_found = any(ch in cur_text for ch in SENT_END_CHARS) or cur_text.rstrip().endswith("\n")
                if end_found:
                    # Trim to earliest boundary
                    cut_pos = len(cur_text)
                    for ch in SENT_END_CHARS:
                        p = cur_text.find(ch)
                        if p != -1:
                            cut_pos = min(cut_pos, p + 1)
                    npos = cur_text.find("\n")
                    if npos != -1:
                        cut_pos = min(cut_pos, npos)

                    sentence_out = cur_text[:cut_pos]
                    clean_sentence = " ".join(sentence_out.replace("\n", " ").split()).strip()

                    # Build assistant content to classify
                    if classification_mode == "sentence":
                        cls_content = clean_sentence
                    else:
                        candidate_full = (accepted_text + ("" if accepted_text.endswith((" ", "\n")) or clean_sentence.startswith(" ")
                                                           else " ") + clean_sentence)
                        if cumulative_exclude_prompt:
                            start_idx = len(initial_accepted_text)
                            candidate_full = candidate_full[start_idx:].lstrip()
                        cls_content = candidate_full

                    # Optional normalization on assistant content before rendering
                    if strip_enumeration_for_class:
                        cls_content = ENUM_PREFIX_RE.sub("", cls_content).strip()
                    if classify_strip_trailing_period:
                        cls_content = _strip_trailing_period(cls_content)

                    # Render full chat string for SAPLMA (NO generation prompt)
                    classify_text_str = _render_for_classification(tok, messages_prefix, cls_content)

                    # ---------------- Content filters on NEW sentence only ----------------
                    to_filter = ENUM_PREFIX_RE.sub("", clean_sentence).strip() if strip_enumeration_for_class else clean_sentence
                    reason = _reject_reason(
                        to_filter,
                        require_space=require_space_in_sentence,
                        min_chars=min_sentence_chars,
                        min_alpha_chars=min_alpha_chars,
                        require_keywords=require_keywords,
                    )

                    # Optionally relax some filters on the last retry
                    if reason and relax_filters_on_last_retry and (decode_mode == "hybrid") and last_attempt_full_sample:
                        if reason.startswith("too short") or reason == "no space":
                            log.debug("[RELAX] Overriding filter '%s' on last retry.", reason)
                            reason = None

                    if reason:
                        # Record filter rejection
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
                        log.info("[SKIP] %r (%s) | mode=%s", to_filter, reason, mode)
                        if first_token_id is not None:
                            banned_starts.add(first_token_id)
                        if first_bigram is not None:
                            banned_pairs.add(first_bigram)
                        break  # retry the sentence

                    # ---------------- SAPLMA classification ----------------
                    emb = embed_text_last_token_with_loaded_model(tok, mdl, classify_text_str,
                                                                  layer_from_end, max_length=cfg_max_ctx)
                    X = emb.unsqueeze(0).cpu().numpy().astype("float32")

                    # Keras/TensorFlow is on CPU; predict -> probability scalar
                    prob_true = float(saplma_model.predict(X, verbose=0).ravel()[0])
                    is_true = prob_true > thr_use

                    log.info("[SENTENCE] scope=%s | mode=%s | prob_true=%.4f thr=%.4f -> %s",
                             classification_mode, mode, prob_true, thr_use,
                             "ACCEPT" if is_true else "REJECT")

                    if is_true:
                        # Append the ORIGINAL (punctuated) sentence to assistant content
                        to_add = sentence_out if accepted_text.endswith((" ", "\n")) or sentence_out.startswith(" ") else " " + sentence_out
                        accepted_text += to_add
                        # Update chat messages and re-render base context for next sentence
                        messages[-1]["content"] = accepted_text
                        messages_prefix[-1]["content"] = ""  # for classification we always overwrite with cls_content
                        accepted_ids = _render_for_generation(tok, messages)
                        sentences_accepted += 1
                        accepted_this_sentence = True
                        accepted_sentences_list.append(sentence_out)
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
                    break  # accepted or rejected

                # Guard against excessively long run-ons
                if cur_tokens >= max_tokens_per_sentence:
                    sentences_rejected += 1
                    log.info(f"[RETRY] sentence exceeded max_tokens_per_sentence={max_tokens_per_sentence} -> reject & retry | mode={mode}")
                    rejected_events.append(SAPLMARejected(
                        type="runon",
                        attempt=attempt,
                        mode=mode,
                        sentence_out=tok.decode(cur_ids, skip_special_tokens=True),
                        clean_sentence=tok.decode(cur_ids, skip_special_tokens=True),
                        classify_text="",
                        prob_true=None,
                        threshold=None,
                        reason="exceeded max_tokens_per_sentence"
                    ))
                    if first_token_id is not None:
                        banned_starts.add(first_token_id)
                    if first_bigram is not None:
                        banned_pairs.add(first_bigram)
                    break  # retry with a new opening

            if accepted_this_sentence:
                break  # proceed to next sentence

        if not accepted_this_sentence:
            log.warning(f"[GIVEUP] Could not produce a truthful sentence after {retries_per_sentence} attempts.")
            return _finish(
                accepted_text, log, t0,
                sentences_accepted, sentences_rejected, trivial_skips,
                eos_early, stopped_by_budget,
                return_details=return_details,
                accepted_sentences_list=accepted_sentences_list,
                rejected_events=rejected_events
            )

    log.info("[DONE] Max sentences reached.")
    return _finish(
        accepted_text, log, t0,
        sentences_accepted, sentences_rejected, trivial_skips,
        eos_early, stopped_by_budget,
        return_details=return_details,
        accepted_sentences_list=accepted_sentences_list,
        rejected_events=rejected_events
    )


def _finish(text: str, log: logging.Logger, t0: float,
            sentences_accepted: int, sentences_rejected: int,
            trivial_skips: int, eos_early: bool, stopped_by_budget: bool,
            return_details: bool = False,
            accepted_sentences_list: List[str] | None = None,
            rejected_events: List[SAPLMARejected] | None = None
            ) -> str | GuardedGenerationResult:
    elapsed = time.time() - t0
    log.info("======== SUMMARY ========")
    log.info(f"Accepted sentences : {sentences_accepted}")
    log.info(f"Rejected sentences : {sentences_rejected}")
    log.info(f"Trivial skips      : {trivial_skips}")
    log.info(f"Stopped by EOS     : {eos_early}")
    log.info(f"Stopped by budget  : {stopped_by_budget}")
    log.info(f"Elapsed            : {elapsed}")
    log.info(f"Final answer       : {text}")

    if not return_details:
        return text

    return GuardedGenerationResult(
        final_text=text,
        accepted_sentences=accepted_sentences_list or [],
        rejected=rejected_events or [],
        summary={
            "accepted": sentences_accepted,
            "rejected": sentences_rejected,
            "skipped": trivial_skips,
            "eos_early": eos_early,
            "stopped_by_budget": stopped_by_budget,
            "elapsed_sec": round(elapsed, 3),
        }
    )


# ---------------------------------------------------------------------------
# Example (manual)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    PROMPT = "Dogs are loyal and also"
    MODEL  = "../models/Llama-2-7B-Chat-fp16"
    BUNDLE = "../pretrained_saplma/completion/saplma_checkpoints_LLAMA7/BEST_layer12__heldout_data/capitals"

    out = generate_with_saplma_guardrail(
        prompt=PROMPT,
        bundle_path=BUNDLE,
        model_path=MODEL,
        decode_mode="hybrid",
        classification_mode="cumulative",
        cumulative_exclude_prompt=False,
        max_sentences=3,
        retries_per_sentence=10,
        max_new_tokens_total=512,
        max_tokens_per_sentence=64,
        min_sentence_chars=1,
        min_alpha_chars=1,
        require_space_in_sentence=True,
        require_keywords=None,
        strip_enumeration_for_class=True,
        classify_strip_trailing_period=True,
        top_p=0.8,
        temperature=0.8,
        min_tokens_to_keep=5,
        relax_filters_on_last_retry=True,
        threshold_offset=0.0,
        device="auto",
        log_level="INFO",
        return_details=True,
        # Optional: cap VRAM during load, e.g. 70% to avoid OOM on load
        # max_memory={"cuda:0": "70%"},
    )
    print("\n=== FINAL (GUARDED) ===")
    if isinstance(out, GuardedGenerationResult):
        print(out.final_text)
        print("\nRejected count:", len(out.rejected))
    else:
        print(out)
