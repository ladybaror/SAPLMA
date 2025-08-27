# guarded_instruct.py
# ------------------------------------------------------------
# Guardrailed generation with SAPLMA, but the *classifier input*
# is wrapped using the tokenizer's chat_template (instruct style).
#
# Generation itself remains token-by-token (completion style),
# while the last-token embedding for SAPLMA is taken from the
# chat-serialized string:
#   [system?] + [user: <prompt>] + [assistant: <candidate>]
#
# Choose whether to include the original prompt as the "user turn"
# and optionally provide a system prompt.
# ------------------------------------------------------------

from __future__ import annotations
from typing import Optional, Set, List, Tuple
import logging, time, re

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__name__), "..")))

# Use the NEW chat-templated embedding helper; do not change saplma_api’s existing functions.
from guarded_infer.saplma_api import (
    load_best_bundle,
    embed_text_last_token_with_loaded_model_chat,  # <— instruct/chat variant
)

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
    return logging.getLogger("saplma_guard_instruct")


# ---------------------------------------------------------------------------
# Robust nucleus (top-p) sampling
# ---------------------------------------------------------------------------
def _nucleus_sample(
    next_logits: torch.Tensor,
    top_p: float = 0.95,
    temperature: float = 1.1,
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

DATE_RE   = re.compile(r"^\s*\d{4}[-/]\d{2}[-/]\d{2}\s*\.?\s*$")                     # 2019-02-19
TIME_RE   = re.compile(r"^\s*\d{1,2}:\d{2}(?::\d{2})?\s*(AM|PM|am|pm)?\s*\.?\s*$")   # 17:30:00 / 01:00 AM
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

# Remove exactly one trailing '.', if present (stabilizes SAPLMA for some bundles)
_STRIP_DOT_RE = re.compile(r"\.\s*$")
def _strip_trailing_period(s: str) -> str:
    return _STRIP_DOT_RE.sub("", s)


# ---------------------------------------------------------------------------
# Core (instruct/chat-templated SAPLMA embedding)
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_with_saplma_guardrail_instruct(
    prompt: str,
    bundle_path: str,
    model_path: str,

    # Decoding policy (same knobs as completion version)
    decode_mode: str = "hybrid",     # {"greedy","hybrid"}
    temperature: float = 1.1,
    top_p: float = 0.95,
    min_tokens_to_keep: int = 5,

    # Classification scope
    classification_mode: str = "sentence",  # {"sentence","cumulative"}
    cumulative_exclude_prompt: bool = False,
    strip_enumeration_for_class: bool = True,

    # SAPLMA input normalization
    classify_strip_trailing_period: bool = True,

    # Chat template knobs (ONLY affect the text fed into SAPLMA)
    chat_role: str = "assistant",
    chat_user_turn_from_prompt: bool = True,  # if True, put `prompt` as the user message
    chat_user_turn: Optional[str] = None,     # override user content; ignored if the above is True
    chat_system: Optional[str] = None,
    chat_add_generation_prompt: bool = False,

    # Budgets
    max_sentences: int = 20,
    max_new_tokens_total: int = 512,
    max_tokens_per_sentence: int = 128,
    retries_per_sentence: int = 5,

    # Content filters (applied to the *new* sentence only)
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
) -> str:
    """
    Same generation loop as guarded_completion, but classification embeddings
    are taken from a chat-templated serialization via
    embed_text_last_token_with_loaded_model_chat(...).
    """
    if decode_mode not in ("greedy", "hybrid"):
        raise ValueError("decode_mode must be 'greedy' or 'hybrid'")
    if classification_mode not in ("sentence", "cumulative"):
        raise ValueError("classification_mode must be 'sentence' or 'cumulative'")

    log = _setup_logger(log_level)
    t0 = time.time()

    # --- tokenizer/model for generation (and for embeddings) ---
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id

    torch_dtype = torch.float16 if torch.cuda.is_available() and device != "cpu" else torch.float32
    mdl = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=None if device == "cpu" else "auto"
    )
    if device == "cpu":
        mdl.to("cpu")
    mdl.eval()

    cfg_max_ctx = getattr(mdl.config, "max_position_embeddings", 4096)
    eos_id = tok.eos_token_id

    # --- SAPLMA bundle ---
    saplma_model, thr_opt, meta, _ = load_best_bundle(bundle=bundle_path)
    layer_from_end = int(meta["layer_from_end"])
    thr_use = float(min(max(float(thr_opt) + float(threshold_offset), 0.0), 1.0))

    # --- header ---
    log.info("======== SAPLMA Guarded Generation (INSTRUCT) ========")
    log.info("Prompt            : %s", prompt)
    log.info("Base model path   : %s", model_path)
    log.info("Bundle            : %s", bundle_path)
    log.info("Layer from end    : %d", layer_from_end)
    log.info("SAPLMA threshold  : %.4f (best=%.4f, offset=%+.4f)", thr_use, float(thr_opt), float(threshold_offset))
    log.info("Context window    : %d", cfg_max_ctx)
    if decode_mode == "greedy":
        log.info("Decoding          : GREEDY (argmax)")
    else:
        log.info("Decoding          : HYBRID (top-p=%.2f, temp=%.2f, keep>=%d; sample-all on last retry)",
                 top_p, temperature, min_tokens_to_keep)
    log.info("Classify scope    : %s%s", classification_mode.upper(),
             " (exclude_prompt)" if (classification_mode=="cumulative" and cumulative_exclude_prompt) else "")
    log.info("Chat template     : role=%s | user_from_prompt=%s | add_gen_prompt=%s | system=%s",
             chat_role, str(chat_user_turn_from_prompt), str(chat_add_generation_prompt),
             "yes" if chat_system else "no")
    log.info("Device            : %s | dtype=%s", str(next(mdl.parameters()).device), str(torch_dtype))
    log.info("Limits            : %d sentences | %d total tokens | %d tokens/sentence | %d retries/sentence",
             max_sentences, max_new_tokens_total, max_tokens_per_sentence, retries_per_sentence)
    log.info("Filters           : min_len=%d | min_alpha=%d | require_space=%s | require_keywords=%s | strip_enum_for_class=%s | relax_last=%s",
             min_sentence_chars, min_alpha_chars, str(require_space_in_sentence),
             str(require_keywords), str(strip_enumeration_for_class), str(relax_filters_on_last_retry))
    log.info("[START]")

    # --- state ---
    accepted_text = prompt if prompt.endswith((" ", "\n")) else prompt + " "
    initial_accepted_text = accepted_text
    accepted_tokens_total = 0
    sentences_accepted = 0
    sentences_rejected = 0
    trivial_skips = 0
    eos_early = False
    stopped_by_budget = False

    accepted_ids = tok(accepted_text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
    SENT_END_CHARS = (".", "!", "?")

    banned_starts: Set[int] = set()
    banned_pairs: Set[Tuple[int, int]] = set()

    # --- helper to decide chat user turn text ---
    def _chat_user_text() -> Optional[str]:
        if chat_user_turn_from_prompt:
            return prompt
        return chat_user_turn

    # --- main loop ---
    for s_idx in range(max_sentences):
        accepted_this_sentence = False

        for attempt in range(retries_per_sentence):
            cur_ids: List[int] = []
            cur_tokens = 0
            first_token_id: Optional[int] = None
            first_bigram: Optional[Tuple[int, int]] = None

            while True:
                if accepted_tokens_total + cur_tokens >= max_new_tokens_total:
                    stopped_by_budget = True
                    log.warning("[STOP] Reached max_new_tokens_total=%d", max_new_tokens_total)
                    return _finish(accepted_text, log, t0, sentences_accepted, sentences_rejected,
                                   trivial_skips, eos_early, stopped_by_budget)

                ctx_ids = accepted_ids + cur_ids
                if len(ctx_ids) > (cfg_max_ctx - 1):
                    ctx_ids = ctx_ids[-(cfg_max_ctx - 1):]
                input_ids = torch.tensor([ctx_ids], device=mdl.device)

                out = mdl(input_ids=input_ids, use_cache=False)
                next_logits = out.logits[0, -1]

                if cur_tokens == 0 and banned_starts:
                    next_logits[list(banned_starts)] = float("-inf")
                elif cur_tokens == 1 and first_token_id is not None and banned_pairs:
                    for t1, t2 in banned_pairs:
                        if t1 == first_token_id:
                            next_logits[t2] = float("-inf")

                if decode_mode == "greedy":
                    mode = "GREEDY"
                    next_id = int(torch.argmax(next_logits))
                else:
                    mode = "SAMPLE-ALL"
                    next_id = _nucleus_sample(
                        next_logits, top_p=top_p, temperature=temperature,
                        min_tokens_to_keep=min_tokens_to_keep
                    )

                if cur_tokens == 0:
                    first_token_id = next_id

                if next_id == eos_id:
                    eos_early = True
                    log.info("[EOS] encountered - returning final answer.")
                    return _finish(accepted_text, log, t0, sentences_accepted, sentences_rejected,
                                   trivial_skips, eos_early, stopped_by_budget)

                cur_ids.append(next_id)
                cur_tokens += 1
                accepted_tokens_total += 1

                if cur_tokens == 2 and first_token_id is not None:
                    first_bigram = (first_token_id, next_id)

                cur_text = tok.decode(cur_ids, skip_special_tokens=True)

                end_found = any(ch in cur_text for ch in SENT_END_CHARS) or cur_text.rstrip().endswith("\n")
                if end_found:
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

                    # -------- Build classification texts (plain & chat parts) --------
                    # Plain (string) pathway mirrors completion version:
                    if classification_mode == "sentence":
                        classify_text = clean_sentence
                    else:
                        candidate_full = (accepted_text
                                          + ("" if accepted_text.endswith((" ", "\n")) or clean_sentence.startswith(" ") else " ")
                                          + clean_sentence)
                        if cumulative_exclude_prompt:
                            start_idx = len(initial_accepted_text)
                            candidate_full = candidate_full[start_idx:].lstrip()
                        classify_text = candidate_full

                    if strip_enumeration_for_class:
                        classify_text_stripped = ENUM_PREFIX_RE.sub("", classify_text).strip()
                    else:
                        classify_text_stripped = classify_text

                    if classify_strip_trailing_period:
                        classify_text_stripped = _strip_trailing_period(classify_text_stripped)

                    # Chat-templated parts:
                    #   - We typically want [user: prompt], [assistant: <assistant_part>]
                    #   - For sentence mode: assistant_part = sentence
                    #   - For cumulative (exclude_prompt=False): assistant_part = (accepted_text + sentence) minus the prompt
                    #   - For cumulative (exclude_prompt=True): assistant_part = classify_text_stripped (already sans prompt)
                    chat_user_content = _chat_user_text()

                    if classification_mode == "sentence":
                        chat_assistant_part = classify_text_stripped
                    else:
                        if cumulative_exclude_prompt:
                            chat_assistant_part = classify_text_stripped
                        else:
                            # Remove the prompt+space prefix so assistant_part is only the model's output
                            # initial_accepted_text == prompt + " " (we ensured a trailing space at init)
                            if classify_text.startswith(initial_accepted_text):
                                assistant_only = classify_text[len(initial_accepted_text):].lstrip()
                            else:
                                assistant_only = classify_text
                            # Apply the same normalization choices:
                            if strip_enumeration_for_class:
                                assistant_only = ENUM_PREFIX_RE.sub("", assistant_only).strip()
                            if classify_strip_trailing_period:
                                assistant_only = _strip_trailing_period(assistant_only)
                            chat_assistant_part = assistant_only

                    # ---------------- Content filters on NEW sentence only ----------------
                    to_filter = ENUM_PREFIX_RE.sub("", clean_sentence).strip() if strip_enumeration_for_class else clean_sentence
                    reason = _reject_reason(
                        to_filter,
                        require_space=require_space_in_sentence,
                        min_chars=min_sentence_chars,
                        min_alpha_chars=min_alpha_chars,
                        require_keywords=require_keywords,
                    )
                    if reason and relax_filters_on_last_retry and (decode_mode == "hybrid") and (attempt == retries_per_sentence - 1):
                        if reason.startswith("too short") or reason == "no space":
                            reason = None

                    if reason:
                        trivial_skips += 1
                        log.info("[SKIP] %r (%s) | mode=%s", to_filter, reason, mode)
                        if first_token_id is not None:
                            banned_starts.add(first_token_id)
                        if first_bigram is not None:
                            banned_pairs.add(first_bigram)
                        break  # retry

                    # ---------------- SAPLMA classification (chat-templated) ----------------
                    emb = embed_text_last_token_with_loaded_model_chat(
                        tok, mdl, chat_assistant_part, layer_from_end, max_length=cfg_max_ctx,
                        role=chat_role,
                        user_turn=chat_user_content,
                        system=chat_system,
                        add_generation_prompt=chat_add_generation_prompt,
                    )
                    X = emb.unsqueeze(0).cpu().numpy().astype("float32")
                    prob_true = float(saplma_model.predict(X, verbose=0).ravel()[0])
                    is_true = prob_true > thr_use

                    log.info("[SENTENCE] scope=%s | mode=%s | prob_true=%.4f thr=%.4f -> %s",
                             classification_mode, mode, prob_true, thr_use,
                             "ACCEPT" if is_true else "REJECT")

                    if is_true:
                        to_add = sentence_out if accepted_text.endswith((" ", "\n")) or sentence_out.startswith(" ") else " " + sentence_out
                        accepted_text += to_add
                        accepted_ids = tok(accepted_text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
                        sentences_accepted += 1
                        accepted_this_sentence = True
                    else:
                        sentences_rejected += 1
                        if first_token_id is not None:
                            banned_starts.add(first_token_id)
                        if first_bigram is not None:
                            banned_pairs.add(first_bigram)
                    break  # finish this attempt

                if cur_tokens >= max_tokens_per_sentence:
                    sentences_rejected += 1
                    log.info("[RETRY] sentence exceeded max_tokens_per_sentence=%d -> reject & retry | mode=%s",
                             max_tokens_per_sentence, mode)
                    if first_token_id is not None:
                        banned_starts.add(first_token_id)
                    if first_bigram is not None:
                        banned_pairs.add(first_bigram)
                    break  # retry

            if accepted_this_sentence:
                break  # next sentence

        if not accepted_this_sentence:
            log.warning("[GIVEUP] Could not produce a truthful sentence after %d attempts.", retries_per_sentence)
            return _finish(accepted_text, log, t0, sentences_accepted, sentences_rejected,
                           trivial_skips, eos_early, stopped_by_budget)

    log.info("[DONE] Max sentences reached.")
    return _finish(accepted_text, log, t0, sentences_accepted, sentences_rejected,
                   trivial_skips, eos_early, stopped_by_budget)


def _finish(text: str, log: logging.Logger, t0: float,
            sentences_accepted: int, sentences_rejected: int,
            trivial_skips: int, eos_early: bool, stopped_by_budget: bool) -> str:
    elapsed = time.time() - t0
    log.info("======== SUMMARY ========")
    log.info("Accepted sentences : %d", sentences_accepted)
    log.info("Rejected sentences : %d", sentences_rejected)
    log.info("Trivial skips      : %d", trivial_skips)
    log.info("Stopped by EOS     : %s", eos_early)
    log.info("Stopped by budget  : %s", stopped_by_budget)
    log.info("Elapsed            : %.2fs", elapsed)
    log.info("Final answer       : %s", text)
    return text


# ---------------------------------------------------------------------------
# Example (manual test)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    PROMPT = "Dogs are loyal and also"
    MODEL  = "../models/Llama-2-7B-Chat-fp16"
    BUNDLE = "../pretrained_saplma/completion/saplma_checkpoints_LLAMA7/BEST_layer12__heldout_data/capitals"

    out = generate_with_saplma_guardrail_instruct(
        prompt=PROMPT,
        bundle_path=BUNDLE,
        model_path=MODEL,
        decode_mode="hybrid",
        classification_mode="cumulative",
        cumulative_exclude_prompt=False,
        # filters
        min_sentence_chars=1,
        min_alpha_chars=1,
        require_space_in_sentence=True,
        strip_enumeration_for_class=True,
        classify_strip_trailing_period=True,
        # chat wrapping for SAPLMA
        chat_role="assistant",
        chat_user_turn_from_prompt=True,   # supply the original PROMPT as the user turn
        chat_user_turn=None,               # (ignored because of ^)
        chat_system=None,                  # or a system string
        chat_add_generation_prompt=False,
        # budgets
        max_sentences=6,
        retries_per_sentence=10,
        max_new_tokens_total=512,
        max_tokens_per_sentence=128,
        # sampling
        top_p=0.8,
        temperature=0.8,
        min_tokens_to_keep=5,
        relax_filters_on_last_retry=True,
        # saplma thr tweak
        threshold_offset=0.0,
        device="auto",
        log_level="INFO",
    )
    print("\n=== FINAL ANSWER (INSTRUCT / CUMULATIVE) ===")
    print(out)
