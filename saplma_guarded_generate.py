# saplma_guarded_generate.py
# ------------------------------------------------------------
# Guardrailed generation with SAPLMA:
#   1) Start from user prompt
#   2) Generate tokens until sentence end (., !, ?, or newline)
#   3) Classify either:
#        - only the NEW sentence (classification_mode="sentence"), or
#        - the CUMULATIVE response-so-far incl. the new sentence (classification_mode="cumulative")
#   4) If True  -> append to output and continue (accepted text is fed back)
#      If False -> discard & retry (ban the rejected start token)
#
# decode_mode:
#   - "greedy"  -> pure argmax at every step (old behavior)
#   - "hybrid"  -> greedy, but last retry per sentence uses top-p sampling
# ------------------------------------------------------------

from __future__ import annotations
from typing import Optional, Set, List
import logging, time, re

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

# expects these in your saplma_api.py
from saplma_api import load_best_bundle, embed_text_last_token_with_loaded_model


# ---------------- logging ----------------
def _setup_logger(level: str = "INFO"):
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s | %(levelname)-5s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("saplma_guard")


# ---------------- nucleus sampling (hybrid last retry) ----------------
def _nucleus_sample(next_logits: torch.Tensor, top_p: float = 0.95, temperature: float = 0.8) -> int:
    if temperature > 0:
        next_logits = next_logits / temperature
    probs = F.softmax(next_logits, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cum = torch.cumsum(sorted_probs, dim=-1)
    keep_mask = cum <= top_p
    if not torch.any(keep_mask):
        keep_mask = torch.zeros_like(sorted_probs, dtype=torch.bool)
        keep_mask[0] = True
    k = int(torch.nonzero(keep_mask)[-1].item())
    keep_probs = sorted_probs[:k + 1]
    keep_probs = keep_probs / keep_probs.sum()
    pick_rel = torch.multinomial(keep_probs, 1).item()
    return int(sorted_idx[pick_rel].item())


# ---------------- content filters & cleaning ----------------
ENUM_PREFIX_RE = re.compile(r"^\s*(?:Answer:\s*)?(?:\(?[A-Za-z]\)|[A-Za-z]\)|\d+\)|\d+\.)\s*", re.UNICODE)

DATE_RE   = re.compile(r"^\s*\d{4}[-/]\d{2}[-/]\d{2}\s*\.?\s*$")                     # 2019-02-19
TIME_RE   = re.compile(r"^\s*\d{1,2}:\d{2}(?::\d{2})?\s*(AM|PM|am|pm)?\s*\.?\s*$")   # 17:30:00 / 01:00 AM
UTC_RE    = re.compile(r"^\s*UTC\.?\s*$", re.IGNORECASE)
PUNC_RE   = re.compile(r"^\s*[\.\!\?]\s*$")
NUM_RE    = re.compile(r"^\s*[\d\W_]+\s*$")  # only digits/punct/space/underscore
WIKI_RE   = re.compile(r"Asked by Wiki User|Trivia Questions", re.IGNORECASE)

def _reject_reason(
    s: str,
    require_space: bool,
    min_chars: int,
    min_alpha_chars: int,
    require_keywords: Optional[List[str]],
) -> Optional[str]:
    """Return a reason to reject based on heuristics; None means pass."""
    s_stripped = s.strip()
    # hard bans
    if PUNC_RE.match(s_stripped):                  return "only punctuation"
    if DATE_RE.match(s_stripped):                  return "looks like a date"
    if TIME_RE.match(s_stripped):                  return "looks like a time"
    if UTC_RE.match(s_stripped):                   return "UTC token"
    if WIKI_RE.search(s_stripped):                 return "wiki/trivia footer"
    if NUM_RE.match(s_stripped):                   return "numeric or non-alphabetic"
    # length/letters/spaces
    if len(s_stripped) < min_chars:                return f"too short (<{min_chars})"
    alpha = sum(ch.isalpha() for ch in s_stripped)
    if alpha < min_alpha_chars:                    return f"too few letters (<{min_alpha_chars})"
    if require_space and (" " not in s_stripped):  return "no space"
    # topic keywords (contains, case-insensitive)
    if require_keywords:
        s_low = s_stripped.lower()
        if not any(kw.lower() in s_low for kw in require_keywords):
            return f"missing required keywords {require_keywords}"
    return None


# ---------------- core ----------------
@torch.no_grad()
def generate_with_saplma_guardrail(
    prompt: str,
    bundle_path: str,
    model_path: str,

    # decoding
    decode_mode: str = "hybrid",     # "greedy" or "hybrid"
    temperature: float = 0.8,        # used only on sampling fallback (hybrid)
    top_p: float = 0.95,             # used only on sampling fallback (hybrid)

    # classification scope
    classification_mode: str = "sentence",  # "sentence" or "cumulative"
    cumulative_exclude_prompt: bool = False,  # when cumulative, exclude the original prompt?
    strip_enumeration_for_class: bool = True, # strip "1.", "A)", "Answer:" for classification only

    # budgets
    max_sentences: int = 20,
    max_new_tokens_total: int = 512,
    max_tokens_per_sentence: int = 128,
    retries_per_sentence: int = 5,

    # content filters (applied to the *new* sentence text)
    min_sentence_chars: int = 20,
    min_alpha_chars: int = 8,
    require_space_in_sentence: bool = True,
    require_keywords: Optional[List[str]] = None,  # e.g., ["Venus"]

    # SAPLMA threshold tweak
    threshold_offset: float = 0.0,

    # system
    device: str = "auto",
    log_level: str = "INFO",
) -> str:
    """
    Returns final accepted text (prompt + accepted sentences).

    classification_mode:
      - "sentence"   -> classify ONLY the newly generated sentence (default)
      - "cumulative" -> classify the FULL response-so-far (prompt + accepted + new sentence);
                        set cumulative_exclude_prompt=True to drop the original prompt from the classified text.
    """
    if decode_mode not in ("greedy", "hybrid"):
        raise ValueError("decode_mode must be 'greedy' or 'hybrid'")
    if classification_mode not in ("sentence", "cumulative"):
        raise ValueError("classification_mode must be 'sentence' or 'cumulative'")

    log = _setup_logger(log_level)
    t0 = time.time()

    # tokenizer/model
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

    # SAPLMA
    saplma_model, thr_opt, meta, _ = load_best_bundle(bundle=bundle_path)
    layer_from_end = int(meta["layer_from_end"])
    thr_use = float(min(max(float(thr_opt) + float(threshold_offset), 0.0), 1.0))

    # header
    log.info("======== SAPLMA Guarded Generation ========")
    log.info("Prompt            : %s", prompt)
    log.info("Base model path   : %s", model_path)
    log.info("Bundle            : %s", bundle_path)
    log.info("Layer from end    : %d", layer_from_end)
    log.info("SAPLMA threshold  : %.4f (best=%.4f, offset=%+.4f)", thr_use, float(thr_opt), float(threshold_offset))
    log.info("Context window    : %d", cfg_max_ctx)
    if decode_mode == "greedy":
        log.info("Decoding          : GREEDY (argmax at every step)")
    else:
        log.info("Decoding          : HYBRID (greedy; last retry uses top-p=%.2f, temperature=%.2f)", top_p, temperature)
    log.info("Classify scope    : %s%s", classification_mode.upper(),
             " (exclude_prompt)" if (classification_mode=="cumulative" and cumulative_exclude_prompt) else "")
    log.info("Device            : %s | dtype=%s", str(next(mdl.parameters()).device), str(torch_dtype))
    log.info("Limits            : %d sentences | %d total tokens | %d tokens/sentence | %d retries/sentence",
             max_sentences, max_new_tokens_total, max_tokens_per_sentence, retries_per_sentence)
    log.info("Filters           : min_len=%d | min_alpha=%d | require_space=%s | require_keywords=%s | strip_enum_for_class=%s",
             min_sentence_chars, min_alpha_chars, str(require_space_in_sentence),
             str(require_keywords), str(strip_enumeration_for_class))
    log.info("[START]")

    # state
    accepted_text = prompt if prompt.endswith((" ", "\n")) else prompt + " "
    initial_accepted_text = accepted_text  # for cumulative_exclude_prompt
    accepted_tokens_total = 0
    sentences_accepted = 0
    sentences_rejected = 0
    trivial_skips = 0
    eos_early = False
    stopped_by_budget = False

    accepted_ids = tok(accepted_text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
    SENT_END_CHARS = (".", "!", "?")

    for s_idx in range(max_sentences):
        accepted_this_sentence = False
        banned_starts: Set[int] = set()

        for attempt in range(retries_per_sentence):
            cur_ids: List[int] = []
            cur_tokens = 0
            first_token_id: Optional[int] = None

            while True:
                if accepted_tokens_total + cur_tokens >= max_new_tokens_total:
                    stopped_by_budget = True
                    log.warning("[STOP] Reached max_new_tokens_total=%d", max_new_tokens_total)
                    return _finish(accepted_text, log, t0, sentences_accepted, sentences_rejected,
                                   trivial_skips, eos_early, stopped_by_budget)

                # context = accepted_ids + cur_ids (truncate to leave room for one new token)
                ctx_ids = accepted_ids + cur_ids
                if len(ctx_ids) > (cfg_max_ctx - 1):
                    ctx_ids = ctx_ids[-(cfg_max_ctx - 1):]
                input_ids = torch.tensor([ctx_ids], device=mdl.device)

                out = mdl(input_ids=input_ids, use_cache=False)  # forward() ignores temperature
                next_logits = out.logits[0, -1]

                # ban starts that led to false sentences
                if cur_tokens == 0 and banned_starts:
                    next_logits[list(banned_starts)] = float("-inf")

                # pick next token
                if decode_mode == "hybrid" and attempt == retries_per_sentence - 1:
                    next_id = _nucleus_sample(next_logits, top_p=top_p, temperature=temperature)
                else:
                    next_id = int(torch.argmax(next_logits))

                # first token bookkeeping
                if cur_tokens == 0:
                    first_token_id = next_id

                # EOS?
                if next_id == eos_id:
                    eos_early = True
                    log.info("[EOS] encountered – returning final answer.")
                    return _finish(accepted_text, log, t0, sentences_accepted, sentences_rejected,
                                   trivial_skips, eos_early, stopped_by_budget)

                # append token, decode attempt
                cur_ids.append(next_id)
                cur_tokens += 1
                accepted_tokens_total += 1

                cur_text = tok.decode(cur_ids, skip_special_tokens=True)

                # sentence boundary?
                end_found = any(ch in cur_text for ch in SENT_END_CHARS) or cur_text.rstrip().endswith("\n")
                if end_found:
                    # cut at earliest boundary
                    cut_pos = len(cur_text)
                    for ch in SENT_END_CHARS:
                        p = cur_text.find(ch)
                        if p != -1:
                            cut_pos = min(cut_pos, p + 1)
                    npos = cur_text.find("\n")
                    if npos != -1:
                        cut_pos = min(cut_pos, npos)

                    sentence_out = cur_text[:cut_pos]  # what we'll append if accepted (keep enumeration)
                    clean_sentence = " ".join(sentence_out.replace("\n", " ").split()).strip()

                    # ---- Build the text we will classify
                    if classification_mode == "sentence":
                        classify_text = clean_sentence
                    else:
                        # cumulative
                        candidate_full = (accepted_text + ("" if accepted_text.endswith((" ", "\n")) or clean_sentence.startswith(" ")
                                                           else " ") + clean_sentence)
                        if cumulative_exclude_prompt:
                            # drop the prompt portion
                            start_idx = len(initial_accepted_text)
                            candidate_full = candidate_full[start_idx:].lstrip()
                        classify_text = candidate_full

                    # strip enumeration only for classification
                    if strip_enumeration_for_class:
                        classify_text_stripped = ENUM_PREFIX_RE.sub("", classify_text).strip()
                    else:
                        classify_text_stripped = classify_text

                    # ---- Content filters are evaluated on the NEW sentence only
                    # (so we don't accidentally pass dates/times/etc.)
                    to_filter = ENUM_PREFIX_RE.sub("", clean_sentence).strip() if strip_enumeration_for_class else clean_sentence
                    reason = _reject_reason(
                        to_filter,
                        require_space=require_space_in_sentence,
                        min_chars=min_sentence_chars,
                        min_alpha_chars=min_alpha_chars,
                        require_keywords=require_keywords,
                    )
                    if reason:
                        trivial_skips += 1
                        logging.info("[SKIP] %r (%s)", to_filter, reason)
                        if first_token_id is not None:
                            banned_starts.add(first_token_id)
                        break  # retry

                    # ---- SAPLMA classify (on chosen scope)
                    emb = embed_text_last_token_with_loaded_model(tok, mdl, classify_text_stripped,
                                                                  layer_from_end, max_length=cfg_max_ctx)
                    X = emb.unsqueeze(0).cpu().numpy().astype("float32")
                    prob_true = float(saplma_model.predict(X, verbose=0).ravel()[0])
                    is_true = prob_true > thr_use

                    log.info("[SENTENCE] scope=%s | text=%r | prob_true=%.4f thr=%.4f -> %s",
                             classification_mode, classify_text_stripped, prob_true, thr_use,
                             "ACCEPT" if is_true else "REJECT")

                    if is_true:
                        # append ORIGINAL sentence and retokenize accepted_ids
                        to_add = sentence_out if accepted_text.endswith((" ", "\n")) or sentence_out.startswith(" ") else " " + sentence_out
                        accepted_text += to_add
                        accepted_ids = tok(accepted_text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
                        sentences_accepted += 1
                        accepted_this_sentence = True
                    else:
                        sentences_rejected += 1
                        if first_token_id is not None:
                            banned_starts.add(first_token_id)
                    break  # finish attempt

                # guard against run-ons
                if cur_tokens >= max_tokens_per_sentence:
                    sentences_rejected += 1
                    log.info("[RETRY] sentence exceeded max_tokens_per_sentence=%d -> reject & retry",
                             max_tokens_per_sentence)
                    if first_token_id is not None:
                        banned_starts.add(first_token_id)
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


# ---------------- example ----------------
if __name__ == "__main__":
    PROMPT = "Capital of israel is"
    MODEL  = "models/Llama-2-7b-chat-hf"
    # BUNDLE = "pretrained_saplma/completion/saplma_checkpoints_LLAMA7/BEST_layer12__heldout_data/elements"
    BUNDLE = "pretrained_saplma/completion/saplma_checkpoints_LLAMA7/BEST_layer12__heldout_data/capitals"

    # 1) Classify only the new sentence (default behavior)
    out_sentence = generate_with_saplma_guardrail(
        prompt=PROMPT,
        bundle_path=BUNDLE,
        model_path=MODEL,
        decode_mode="hybrid",                # or "greedy"
        classification_mode="sentence",      # <— only new sentence
        # budgets
        max_sentences=6,
        retries_per_sentence=10,
        max_new_tokens_total=512,
        max_tokens_per_sentence=128,
        # filters
        min_sentence_chars=1,
        min_alpha_chars=1,
        require_space_in_sentence=True,
        require_keywords=[],
        strip_enumeration_for_class=True,
        # threshold tweak
        threshold_offset=0.0,
        # system
        device="auto",
        log_level="INFO",
    )
    print("\n=== FINAL ANSWER (SENTENCE SCOPE) ===")
    print(out_sentence)

    # 2) Classify the cumulative response-so-far (exclude the prompt from judged text)
    out_cumulative = generate_with_saplma_guardrail(
        prompt=PROMPT,
        bundle_path=BUNDLE,
        model_path=MODEL,
        decode_mode="hybrid",
        classification_mode="cumulative",    # <— cumulative scope
        cumulative_exclude_prompt=True,      # don't let the prompt influence truth score
        # budgets
        max_sentences=6,
        retries_per_sentence=10,
        max_new_tokens_total=512,
        max_tokens_per_sentence=128,
        # filters (still applied to the *new* sentence)
        min_sentence_chars=1,
        min_alpha_chars=1,
        require_space_in_sentence=True,
        require_keywords=[],
        strip_enumeration_for_class=True,
        threshold_offset=0.0,
        device="auto",
        log_level="INFO",
    )
    print("\n=== FINAL ANSWER (CUMULATIVE SCOPE) ===")
    print(out_cumulative)
