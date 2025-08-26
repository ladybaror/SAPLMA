# saplma_guarded_generate.py
"""
Guardrailed generation with SAPLMA (Simple Accuracy Prediction via Last-token
Model Activation) for sentence-by-sentence filtering.

TL;DR
------
This module wraps a base causal LM (e.g., LLaMA) with a *truthfulness* gate.
At each sentence boundary, it classifies the candidate sentence (or the full
response-so-far) using a lightweight SAPLMA classifier that reads the *last-token
hidden state* from your base model. If the probability of "True" exceeds a
threshold, the sentence is accepted and appended to the running answer; otherwise
the sentence is rejected and we retry decoding with a different opening.

Key Ideas
---------
1) **Sentence-by-sentence generation**:
   - We generate tokens until a sentence terminator ('.', '!', '?', or newline).
   - That sentence is *proposed* for classification/filters.

2) **Classification modes**:
   - "sentence": classify the new sentence only.
   - "cumulative": classify the (prompt + accepted_text + new_sentence).
     Use `cumulative_exclude_prompt` to drop the original prompt from the text
     being classified (keeps the judge focused on the model’s output).

3) **Decoding modes**:
   - "greedy": deterministic argmax at every step.
   - "hybrid": diversity without chaos.
       • For attempts 0..(retries-2): sample the *first token*, then go greedy.
       • On the last retry: sample *all tokens* in the sentence.
     This yields varied openings but keeps sentences coherent.

4) **Ban lists to avoid loops**:
   - If a candidate sentence is rejected or skipped by filters, we ban the
     first token and the first bigram (first two tokens). This avoids
     re-entering the same (near-)deterministic groove.

5) **Content filters**:
   - Quick heuristics to avoid wasting retries on junk: reject empty strings,
     pure punctuation, dates/times, all-nonalpha, etc. Filters are applied
     only to the *new sentence text*. On the last retry we can optionally
     relax some filters (too-short / no-space) so sampling has a chance.

6) **Why last-token embedding?**
   - SAPLMA reads the final hidden state (layer_from_end = -k) of the *text
     being classified*. That representation is very sensitive to the context
     (prompt + accepted_text + new_sentence). Decide whether you want to
     classify *only the new sentence* vs. *cumulative context* depending on
     your research goals.

Typical Use
-----------
from saplma_guarded_generate import generate_with_saplma_guardrail

text = generate_with_saplma_guardrail(
    prompt="Dogs are loyal",
    bundle_path="pretrained_saplma/.../capitals",
    model_path="models/Llama-2-7B-Chat-fp16",
    decode_mode="hybrid",
    classification_mode="cumulative",
    cumulative_exclude_prompt=False,   # make loop match CLI behavior
)

Design Tradeoffs & Practical Tips
---------------------------------
• Diversity vs. Stability: sampling the *first token* provides large trajectory
  diversity with minimal risk of rambling. The last-retry "sample all tokens"
  is a pressure-release valve when the greedy path keeps failing.

• If you want your loop’s probabilities to match an external CLI that classifies
  *full text*, set `classification_mode="cumulative"` and `cumulative_exclude_prompt=False`.

• If you see *identical* rejections each run:
  - You might be effectively greedy (top_p too low, temperature too low).
  - Your nucleus sampler might degenerate to top-1; ensure `min_tokens_to_keep >= 5`.
  - Seed control elsewhere (torch/np/random) can make sampling repeatable.

• Throughput: we disable `use_cache` because we repeatedly truncate context to
  respect max length and want fresh logits. You can experiment with enabling
  cache + careful slicing if you need speed.

• Safety: this is about *veracity* gating only. You should add other guardrails
  (toxicity/safety) if needed upstream or in parallel.

"""


from __future__ import annotations
from typing import Optional, Set, List, Tuple
import logging, time, re

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__name__), "..")))

# Expects these helpers to be provided by your project:
# - load_best_bundle(...) -> (keras_model, best_threshold, metadata, extra)
# - embed_text_last_token_with_loaded_model(tokenizer, model, text, layer_from_end, max_length)
from guarded_infer.saplma_api import load_best_bundle, embed_text_last_token_with_loaded_model


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def _setup_logger(level: str = "INFO"):
    """
    Create a module-scoped logger with consistent formatting.

    Parameters
    ----------
    level : {"DEBUG","INFO","WARNING","ERROR","CRITICAL"}
        Logging verbosity. Use "DEBUG" for detailed inspection of sampling
        policy, bans, filters, and classifier decisions.

    Returns
    -------
    logging.Logger
        Configured logger named "saplma_guard".
    """
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s | %(levelname)-5s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("saplma_guard")


# ---------------------------------------------------------------------------
# Robust nucleus (top-p) sampling
# ---------------------------------------------------------------------------
def _nucleus_sample(
    next_logits: torch.Tensor,
    top_p: float = 0.95,
    temperature: float = 1.1,
    min_tokens_to_keep: int = 5
) -> int:
    """
    Pick a token via nucleus sampling, with safeguards against degenerate
    (top-1) candidate sets.

    Why `min_tokens_to_keep`?
        In practice, chat LMs can be extremely peaked at the next step.
        If the top token alone already accounts for > top_p mass, a naive
        implementation would behave exactly like greedy. We force a minimum
        candidate set to preserve some stochasticity.

    Parameters
    ----------
    next_logits : torch.Tensor
        Logits for the next token (shape: [vocab]).
    top_p : float, default 0.95
        Cumulative probability mass to retain.
    temperature : float, default 1.1
        >1.0 flattens the distribution (more diversity), <1.0 sharpens it.
    min_tokens_to_keep : int, default 5
        Ensures at least this many top tokens are eligible (unless vocab is
        smaller), preventing collapse to top-1.

    Returns
    -------
    int
        The sampled token ID (int(index) in the model’s vocabulary).
    """
    if temperature and temperature != 1.0:
        next_logits = next_logits / temperature
    probs = F.softmax(next_logits, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cum = torch.cumsum(sorted_probs, dim=-1)

    mask = cum <= top_p
    # Guarantee a non-degenerate candidate set:
    if min_tokens_to_keep > 0:
        mask[:min(min_tokens_to_keep, mask.numel())] = True

    keep_probs = sorted_probs[mask]
    keep_idx = sorted_idx[mask]
    # Normalize for numerical safety
    keep_probs = keep_probs / keep_probs.sum()
    pick_rel = torch.multinomial(keep_probs, 1).item()
    return int(keep_idx[pick_rel].item())


# ---------------------------------------------------------------------------
# Content filters & cleaning
# ---------------------------------------------------------------------------
# Strips enumeration-like prefixes, e.g., "A)", "1.", "Answer:", etc.
ENUM_PREFIX_RE = re.compile(r"^\s*(?:Answer:\s*)?(?:\(?[A-Za-z]\)|[A-Za-z]\)|\d+\)|\d+\.)\s*", re.UNICODE)

# Lightweight heuristics for junk/formatting-only lines:
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
    """
    Decide whether a candidate *new sentence* should be skipped outright
    (before classification) based on quick syntactic/format heuristics.

    Rationale
    ---------
    These checks prevent wasting classifier calls and retries on input that is
    almost certainly not a usable sentence (pure punctuation, emoji-only, etc.).

    Parameters
    ----------
    s : str
        Candidate sentence (cleaned) to check.
    require_space : bool
        If True, enforce at least one space (discourages single-token nonsense).
    min_chars : int
        Minimum total characters (after strip()).
    min_alpha_chars : int
        Minimum number of alphabetic characters (letters only).
    require_keywords : list[str] or None
        If given, require the sentence to contain *at least one* of these
        case-insensitive substrings.

    Returns
    -------
    str or None
        A human-readable reason string if rejected; otherwise None (passes).
    """
    s_stripped = s.strip()
    # Hard bans
    if PUNC_RE.match(s_stripped):                  return "only punctuation"
    if DATE_RE.match(s_stripped):                  return "looks like a date"
    if TIME_RE.match(s_stripped):                  return "looks like a time"
    if UTC_RE.match(s_stripped):                   return "UTC token"
    if WIKI_RE.search(s_stripped):                 return "wiki/trivia footer"
    if NUM_RE.match(s_stripped):                   return "numeric or non-alphabetic"

    # Length / letters / spaces
    if len(s_stripped) < min_chars:                return f"too short (<{min_chars})"
    alpha = sum(ch.isalpha() for ch in s_stripped)
    if alpha < min_alpha_chars:                    return f"too few letters (<{min_alpha_chars})"
    if require_space and (" " not in s_stripped):  return "no space"

    # Topic keywords
    if require_keywords:
        s_low = s_stripped.lower()
        if not any(kw.lower() in s_low for kw in require_keywords):
            return f"missing required keywords {require_keywords}"
    return None


# Strip a trailing '.' only for the text passed into SAPLMA (optional)
_STRIP_DOT_RE = re.compile(r"\.\s*$")
def _strip_trailing_period(s: str) -> str:
    """
    Remove exactly one trailing '.' (and trailing spaces) from the end of a string.

    Why?
        SAPLMA reads the *last token embedding*. A terminal '.' might be the
        last token and may slightly perturb the embedding. Stripping it can
        stabilize classification if your training data predominantly lacked
        terminal punctuation.
    """
    return _STRIP_DOT_RE.sub("", s)


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
    temperature: float = 1.1,        # (hybrid) sampling temperature
    top_p: float = 0.95,             # (hybrid) nucleus mass
    min_tokens_to_keep: int = 5,     # (hybrid) avoid degenerate top-1

    # Classification scope
    classification_mode: str = "sentence",  # {"sentence","cumulative"}
    cumulative_exclude_prompt: bool = False,
    strip_enumeration_for_class: bool = True,

    # SAPLMA input normalization
    classify_strip_trailing_period: bool = True,

    # Budgets
    max_sentences: int = 20,
    max_new_tokens_total: int = 512,
    max_tokens_per_sentence: int = 128,
    retries_per_sentence: int = 5,

    # Content filters (applied to the *new* sentence text only)
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
    Generate a response from a base causal LM while gating each sentence with
    a SAPLMA veracity classifier.

    Algorithm (high level)
    ----------------------
    1) Initialize accepted_text = prompt + " ".
    2) Repeat up to `max_sentences`:
       a) For each attempt up to `retries_per_sentence`:
          i)   Decode tokens until a boundary (., !, ?, newline) or limit.
               • In "hybrid": sample the *first token* of attempts 0..(n-2),
                 then greedy; on last attempt, sample *all tokens*.
               • Apply ban lists to steer away from previously rejected starts.
          ii)  Clean the candidate sentence (for filters).
          iii) If filters pass, format the classification text according to
               `classification_mode` (+/- exclude prompt), then optionally
               strip trailing '.' before embedding.
          iv)  Embed the classification text (last-token at layer_from_end),
               call SAPLMA, compare against threshold.
          v)   If accepted: append the *original sentence* to accepted_text.
               If rejected: add bans and retry.
       b) If no attempt succeeded: return what we have (give up).

    Parameters (highlights)
    -----------------------
    prompt : str
        User prompt / seed text. A trailing space is ensured so the next token
        attaches cleanly.
    bundle_path : str
        Path to the *best* SAPLMA bundle (weights, metadata, threshold).
    model_path : str
        Path / repo ID of the base causal LM used both for generation and for
        extracting last-token embeddings (must match the SAPLMA training family).

    decode_mode : {"greedy","hybrid"}
        - greedy: deterministic argmax (for ablations / baselines).
        - hybrid: diversity with control (see module docstring).

    classification_mode : {"sentence","cumulative"}
        Choose whether to classify the new sentence alone or the concatenation
        of prompt + accepted_text + new_sentence. Use
        `cumulative_exclude_prompt=True` to drop the original prompt from the
        judged text (but keep accepted_text + new_sentence).

    Returns
    -------
    str
        The final text: original prompt + accepted sentences (with punctuation).

    Notes
    -----
    • To match a standalone CLI that classifies full strings, set:
        classification_mode="cumulative", cumulative_exclude_prompt=False
    • If you need reproducibility, manage seeds *outside* this function
      (random, numpy, torch, etc.). The hybrid sampler is nondeterministic.
    """
    # Validate enums early to fail fast on typos
    if decode_mode not in ("greedy", "hybrid"):
        raise ValueError("decode_mode must be 'greedy' or 'hybrid'")
    if classification_mode not in ("sentence", "cumulative"):
        raise ValueError("classification_mode must be 'sentence' or 'cumulative'")

    log = _setup_logger(log_level)
    t0 = time.time()

    # ---------------- Load tokenizer / model ----------------
    # Important: tokenizer & model must be compatible with how SAPLMA was trained.
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        # Ensure padding is safe for causal LMs; align with eos if absent.
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

    # ---------------- Load SAPLMA bundle ----------------
    # Meta is expected to include "layer_from_end" (e.g., -12)
    saplma_model, thr_opt, meta, _ = load_best_bundle(bundle=bundle_path)
    layer_from_end = int(meta["layer_from_end"])
    thr_use = float(min(max(float(thr_opt) + float(threshold_offset), 0.0), 1.0))

    # ---------------- Diagnostics header ----------------
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
        log.info("Decoding          : HYBRID (sample first token each attempt; last retry samples all tokens; top-p=%.2f, temp=%.2f, keep>=%d)",
                 top_p, temperature, min_tokens_to_keep)
    log.info("Classify scope    : %s%s", classification_mode.upper(),
             " (exclude_prompt)" if (classification_mode=="cumulative" and cumulative_exclude_prompt) else "")
    log.info("Device            : %s | dtype=%s", str(next(mdl.parameters()).device), str(torch_dtype))
    log.info("Limits            : %d sentences | %d total tokens | %d tokens/sentence | %d retries/sentence",
             max_sentences, max_new_tokens_total, max_tokens_per_sentence, retries_per_sentence)
    log.info("Filters           : min_len=%d | min_alpha=%d | require_space=%s | require_keywords=%s | strip_enum_for_class=%s | relax_last=%s",
             min_sentence_chars, min_alpha_chars, str(require_space_in_sentence),
             str(require_keywords), str(strip_enumeration_for_class), str(relax_filters_on_last_retry))
    log.info("[START]")

    # ---------------- State ----------------
    # Ensure a trailing space so the model can start the first new token cleanly.
    accepted_text = prompt if prompt.endswith((" ", "\n")) else prompt + " "
    initial_accepted_text = accepted_text  # kept for (optional) prompt exclusion
    accepted_tokens_total = 0
    sentences_accepted = 0
    sentences_rejected = 0
    trivial_skips = 0
    eos_early = False
    stopped_by_budget = False

    # Keep an up-to-date tokenized version of accepted_text for fast concatenation.
    accepted_ids = tok(accepted_text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
    SENT_END_CHARS = (".", "!", "?")

    # Ban sets to avoid repeating the same doomed opening
    banned_starts: Set[int] = set()           # first token id
    banned_pairs: Set[Tuple[int, int]] = set()# (first_token_id, second_token_id)

    # ---------------- Main loop over sentences ----------------
    for s_idx in range(max_sentences):
        accepted_this_sentence = False

        # Multiple attempts to produce a verifiable sentence
        for attempt in range(retries_per_sentence):
            cur_ids: List[int] = []
            cur_tokens = 0
            first_token_id: Optional[int] = None
            first_bigram: Optional[Tuple[int, int]] = None

            # On the last attempt we sample *all* tokens (more exploration):
            last_attempt_full_sample = (decode_mode == "hybrid" and attempt == retries_per_sentence - 1)

            while True:
                # Global budget check
                if accepted_tokens_total + cur_tokens >= max_new_tokens_total:
                    stopped_by_budget = True
                    log.warning("[STOP] Reached max_new_tokens_total=%d", max_new_tokens_total)
                    return _finish(accepted_text, log, t0, sentences_accepted, sentences_rejected,
                                   trivial_skips, eos_early, stopped_by_budget)

                # Build context window: accepted_ids + current in-progress sentence
                # Truncate from the left to keep room for the next token
                ctx_ids = accepted_ids + cur_ids
                if len(ctx_ids) > (cfg_max_ctx - 1):
                    ctx_ids = ctx_ids[-(cfg_max_ctx - 1):]
                input_ids = torch.tensor([ctx_ids], device=mdl.device)

                # Forward pass (no cache; see "Throughput" note in module docstring)
                out = mdl(input_ids=input_ids, use_cache=False)
                next_logits = out.logits[0, -1]

                # Apply ban lists
                if cur_tokens == 0 and banned_starts:
                    next_logits[list(banned_starts)] = float("-inf")
                elif cur_tokens == 1 and first_token_id is not None and banned_pairs:
                    # Ban second token if (first, second) is a known-bad bigram
                    for t1, t2 in banned_pairs:
                        if t1 == first_token_id:
                            next_logits[t2] = float("-inf")

                # Choose the next token according to decode policy
                if decode_mode == "greedy":
                    mode = "GREEDY"
                    next_id = int(torch.argmax(next_logits))

                # HYBRID policy:
                else:
                    # Sample *all* tokens on the last retry
                    mode = "SAMPLE-ALL"
                    next_id = _nucleus_sample(
                        next_logits, top_p=top_p, temperature=temperature,
                        min_tokens_to_keep=min_tokens_to_keep
                    )

                # Bookkeeping for bans
                if cur_tokens == 0:
                    first_token_id = next_id

                # If the model emits EOS, end immediately (return current answer)
                if next_id == eos_id:
                    eos_early = True
                    log.info("[EOS] encountered - returning final answer.")
                    return _finish(accepted_text, log, t0, sentences_accepted, sentences_rejected,
                                   trivial_skips, eos_early, stopped_by_budget)

                # Append the token and keep going
                cur_ids.append(next_id)
                cur_tokens += 1
                accepted_tokens_total += 1

                # Capture the first bigram (for ban_pairs)
                if cur_tokens == 2 and first_token_id is not None:
                    first_bigram = (first_token_id, next_id)

                cur_text = tok.decode(cur_ids, skip_special_tokens=True)

                # Did we hit a sentence boundary?
                end_found = any(ch in cur_text for ch in SENT_END_CHARS) or cur_text.rstrip().endswith("\n")
                if end_found:
                    # Cut at the earliest boundary mark
                    cut_pos = len(cur_text)
                    for ch in SENT_END_CHARS:
                        p = cur_text.find(ch)
                        if p != -1:
                            cut_pos = min(cut_pos, p + 1)
                    npos = cur_text.find("\n")
                    if npos != -1:
                        cut_pos = min(cut_pos, npos)

                    # sentence_out: exactly what we'd append if accepted (keep punctuation)
                    sentence_out = cur_text[:cut_pos]
                    # clean_sentence: whitespace-normalized for filters/classification
                    clean_sentence = " ".join(sentence_out.replace("\n", " ").split()).strip()

                    # ---------------- Build classification text ----------------
                    if classification_mode == "sentence": # TODO - Problem - Get only the generated text (half a sentence)
                        classify_text = clean_sentence
                    else:
                        # cumulative: accepted_text (+ optional space) + clean_sentence
                        candidate_full = (accepted_text + ("" if accepted_text.endswith((" ", "\n")) or clean_sentence.startswith(" ")
                                                           else " ") + clean_sentence)
                        if cumulative_exclude_prompt:
                            # Drop the original prompt portion, keep accepted output so far
                            start_idx = len(initial_accepted_text)
                            candidate_full = candidate_full[start_idx:].lstrip()
                        classify_text = candidate_full

                    # Optional: strip enumeration prefixes for classification input only
                    if strip_enumeration_for_class:
                        classify_text_stripped = ENUM_PREFIX_RE.sub("", classify_text).strip()
                    else:
                        classify_text_stripped = classify_text

                    # Optional: strip trailing '.' for classification input
                    if classify_strip_trailing_period:
                        classify_text_stripped = _strip_trailing_period(classify_text_stripped)

                    # ---------------- Content filters on NEW sentence only ----------------
                    to_filter = ENUM_PREFIX_RE.sub("", clean_sentence).strip() if strip_enumeration_for_class else clean_sentence
                    reason = _reject_reason(
                        to_filter,
                        require_space=require_space_in_sentence,
                        min_chars=min_sentence_chars,
                        min_alpha_chars=min_alpha_chars,
                        require_keywords=require_keywords,
                    )

                    # Optionally relax some filters on the last retry to let sampling breathe
                    if reason and relax_filters_on_last_retry and last_attempt_full_sample:
                        if reason.startswith("too short") or reason == "no space":
                            log.debug("[RELAX] Overriding filter '%s' on last retry.", reason)
                            reason = None

                    if reason:
                        # Skip this candidate without calling SAPLMA
                        trivial_skips += 1
                        log.info("[SKIP] %r (%s) | mode=%s", to_filter, reason, mode)
                        # Add bans so we don't regenerate the same doomed start
                        if first_token_id is not None:
                            banned_starts.add(first_token_id)
                        if first_bigram is not None:
                            banned_pairs.add(first_bigram)
                        break  # retry the sentence

                    # ---------------- SAPLMA classification ----------------
                    print("Curr_sentence to check: ", classify_text_stripped)
                    # Embed the classification text at the chosen layer (last-token)
                    emb = embed_text_last_token_with_loaded_model(tok, mdl, classify_text_stripped,
                                                                  layer_from_end, max_length=cfg_max_ctx)
                    X = emb.unsqueeze(0).cpu().numpy().astype("float32")
                    prob_true = float(saplma_model.predict(X, verbose=0).ravel()[0])
                    is_true = prob_true > thr_use

                    log.info("[SENTENCE] scope=%s | mode=%s | text=%r | prob_true=%.4f thr=%.4f -> %s",
                             classification_mode, mode, classify_text_stripped, prob_true, thr_use,
                             "ACCEPT" if is_true else "REJECT")

                    if is_true:
                        # Append the ORIGINAL (punctuated) sentence to the output
                        to_add = sentence_out if accepted_text.endswith((" ", "\n")) or sentence_out.startswith(" ") else " " + sentence_out
                        accepted_text += to_add
                        # Refresh accepted_ids for the next sentence context
                        accepted_ids = tok(accepted_text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
                        sentences_accepted += 1
                        accepted_this_sentence = True
                    else:
                        sentences_rejected += 1
                        # Add bans to avoid repeating the same opening
                        if first_token_id is not None:
                            banned_starts.add(first_token_id)
                        if first_bigram is not None:
                            banned_pairs.add(first_bigram)
                    break  # finish this attempt; either accepted or rejected

                # Guard against excessively long run-ons
                if cur_tokens >= max_tokens_per_sentence:
                    sentences_rejected += 1
                    log.info("[RETRY] sentence exceeded max_tokens_per_sentence=%d -> reject & retry | mode=%s",
                             max_tokens_per_sentence, mode)
                    if first_token_id is not None:
                        banned_starts.add(first_token_id)
                    if first_bigram is not None:
                        banned_pairs.add(first_bigram)
                    break  # retry with a new opening

            if accepted_this_sentence:
                break  # proceed to next sentence

        # If all attempts failed for this sentence, return what we have
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
    """
    Centralized epilogue: emit a summary and return the built text.

    Parameters
    ----------
    text : str
        Final text (prompt + accepted sentences).
    log : logging.Logger
        Logger used for this run.
    t0 : float
        Start time (time.time()) for wall-clock measurement.
    sentences_accepted, sentences_rejected, trivial_skips : int
        Counters for analytics.
    eos_early : bool
        Whether generation terminated due to EOS.
    stopped_by_budget : bool
        Whether generation halted due to token budget.

    Returns
    -------
    str
        The completed answer.
    """
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
    # Minimal sanity test. For parity with a CLI that classifies the *full*
    # string, we keep cumulative_exclude_prompt=False here. If you flip it to
    # True, expect different probabilities because SAPLMA will then see
    # (accepted_text + sentence) without the original prompt.
    PROMPT = "Dogs are loyal and also"
    MODEL  = "../models/Llama-2-7B-Chat-fp16"
    BUNDLE = "../pretrained_saplma/completion/saplma_checkpoints_LLAMA7/BEST_layer12__heldout_data/capitals"
    
    out_cumulative = generate_with_saplma_guardrail(
        prompt=PROMPT,
        bundle_path=BUNDLE,
        model_path=MODEL,
        decode_mode="hybrid",                 # hybrid is truly non-deterministic here
        classification_mode="cumulative",     # Note - If we use "sentence" it has a problem we will get only the generated text (half a sentence).
        cumulative_exclude_prompt=False,
        max_sentences=6,
        retries_per_sentence=10,
        max_new_tokens_total=512,
        max_tokens_per_sentence=128,
        # Filters: relaxed for quick testing; tighten in production
        min_sentence_chars=1,
        min_alpha_chars=1,
        require_space_in_sentence=True,
        require_keywords=[],
        strip_enumeration_for_class=True,
        classify_strip_trailing_period=True,
        # Hybrid sampling knobs
        top_p=0.95,
        temperature=1.1,
        min_tokens_to_keep=5,
        relax_filters_on_last_retry=True,
        # Threshold tweak
        threshold_offset=0.0,
        device="auto",
        log_level="INFO",
    )
    print("\n=== FINAL ANSWER (CUMULATIVE SCOPE) ===")
    print(out_cumulative)
