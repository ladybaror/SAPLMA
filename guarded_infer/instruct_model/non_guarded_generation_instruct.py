# non_guarded_generate_instruct.py
# ------------------------------------------------------------
# Non-guarded text generation for *instruct/chat* models.
# - Uses tokenizer.apply_chat_template(add_generation_prompt=True)
# - Preserves your nucleus sampling, sentence gating, budgets, logging style
# - Works with Llama-2-7B-Chat, Llama-3.x-Instruct, etc.
# ------------------------------------------------------------

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__name__), "..")))

import time
import re
import logging
from typing import List, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

# Accepts answers like "A) foo", "1. bar", etc.
ENUM_PREFIX_RE = re.compile(r"^\s*(?:Answer:\s*)?(?:\(?[A-Za-z]\)|[A-Za-z]\)|\d+\)|\d+\.)\s*", re.UNICODE)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def _setup_logger(level: str = "INFO"):
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s | %(levelname)-5s | %(message)s",
        datefmt="%d-%m-%Y__%H:%M:%S",
    )
    return logging.getLogger("no_guard_instruct")


# ---------------------------------------------------------------------------
# Sampling utils
# ---------------------------------------------------------------------------
def _nucleus_sample(
    next_logits: torch.Tensor,
    top_p: float = 0.8,
    temperature: float = 0.8,
    min_tokens_to_keep: int = 5,
) -> int:
    """Nucleus sampling with a minimum candidate set size."""
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

    pick_rel = torch.multinomial(keep_probs, num_samples=1)
    return int(keep_idx[pick_rel].item())


def _finish(text: str, log: logging.Logger, t0: float, eos_early: bool, stopped_by_budget: bool) -> str:
    elapsed = time.time() - t0
    log.info("======= SUMMARY =======")
    log.info("stopped by EOS    :  %s", eos_early)
    log.info("Stopped by budget : %s", stopped_by_budget)
    log.info("Elapsed           : %.2fs", elapsed)
    log.info("Final answer      : %s", text)
    return text


# ---------------------------------------------------------------------------
# Chat templating helpers
# ---------------------------------------------------------------------------
def _serialize_chat(
    tokenizer,
    messages,
    add_generation_prompt: bool = True,
) -> list[int]:
    """
    Serialize chat to token IDs. Use tokenizer's chat template if available,
    else fall back to a simple role-tagged format and append an assistant cue.
    """
    can_use_template = (
        hasattr(tokenizer, "apply_chat_template")
        and getattr(tokenizer, "chat_template", None)  # <-- key change
    )

    if can_use_template:
        input = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=False,
            return_tensors=None,
        )
    else:
        # Minimal, safe fallback
        input = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                input.append({"role": "system", "content": content})
            elif role == "user":
                input.append({"role": "user", "content": content})
            else:
                input.append({"role": role.lower(), "content": content})
        if add_generation_prompt:
            input.append({"role": "assistant", "content": content})

    out = tokenizer(input, return_tensors="pt", add_special_tokens=True)
    return out["input_ids"][0].tolist()


# ---------------------------------------------------------------------------
# Instruct generator (sentence-by-sentence with budgets)
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_instruct(
    # Chat input
    messages: List[Dict[str, str]],
    model_path: str,

    # Sampling
    temperature: float = 0.8,
    top_p: float = 0.8,
    min_tokens_to_keep: int = 5,

    # Sentence gating
    min_sentence_chars: int = 20,
    sentence_end_chars: Tuple[str, ...] = (".", "!", "?", "\n"),

    # Budgets
    max_sentences: int = 5,
    max_new_tokens_total: int = 512,
    max_tokens_per_sentence: int = 128,   # kept for parity; we enforce by budget checks

    # System
    device: str = "auto",
    log_level: str = "INFO",
):
    """
    Generate assistant text for an instruct/chat model with a custom incremental loop.

    messages: HF chat format, e.g.:
        [
          {"role": "system", "content": "You are helpful."},
          {"role": "user", "content": "Explain transformers simply."}
        ]
    """
    logger = _setup_logger(log_level)
    logger.info("======== Non Guarded Generation (Instruct) ========")

    t0 = time.time()

    # Tokenizer
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    eos_id = tok.eos_token_id
    eos_token_ids = set([tok.eos_token_id]) if tok.eos_token_id is not None else set()

    # Model
    torch_dtype = torch.float16 if torch.cuda.is_available() and device != "cpu" else torch.float32
    mdl = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map="cpu" if device == "cpu" else "auto",
        trust_remote_code=True,
    )
    mdl.eval()

    # Context window
    cfg_max_ctx = getattr(mdl.config, "max_position_embeddings", 4096)

    # Serialize initial conversation with *generation prompt*:
    # i.e., ends where the assistant is expected to start talking.
    accepted_ids = _serialize_chat(tok, messages, add_generation_prompt=True)

    # We keep an accumulating decoded string for the final answer (assistant side only).
    # For clean UX, we’ll only append what we generate now (not the entire conversation).
    generated_text = ""
    total_token_count = 0
    sentence_count = 0
    loops = 0

    while loops < max_sentences:
        cur_ids: List[int] = []
        cur_text = ""
        cur_token_count = 0

        while True:
            # Budgets
            if total_token_count >= max_new_tokens_total:
                logger.warning("[STOP] Reached max_new_tokens_total=%d", max_new_tokens_total)
                return _finish(generated_text, logger, t0, eos_early=False, stopped_by_budget=True)
            if cur_token_count >= max_tokens_per_sentence:
                logger.info("[SENTENCE] Reached max_tokens_per_sentence=%d", max_tokens_per_sentence)
                break

            # Build sliding window
            ctx_ids = accepted_ids + cur_ids
            if len(ctx_ids) > (cfg_max_ctx - 1):
                ctx_ids = ctx_ids[-(cfg_max_ctx - 1):]

            input_ids = torch.tensor([ctx_ids], device=mdl.device)
            out = mdl(input_ids=input_ids, use_cache=False)
            next_logits = out.logits[0, -1]

            next_id = _nucleus_sample(
                next_logits=next_logits,
                top_p=top_p,
                temperature=temperature,
                min_tokens_to_keep=min_tokens_to_keep,
            )

            # EOS?
            if next_id in eos_token_ids:
                logger.info("[EOS] encountered - returning final answer.")
                return _finish(generated_text.strip(), logger, t0, eos_early=True, stopped_by_budget=False)

            cur_ids.append(next_id)
            cur_token_count += 1
            total_token_count += 1

            # Decode current partial sentence
            cur_text = tok.decode(cur_ids, skip_special_tokens=True)

            # Sentence boundary?
            end_found = any(c in cur_text for c in sentence_end_chars)
            if end_found:
                # Find earliest boundary occurrence
                cut_pos = len(cur_text)
                for c in sentence_end_chars:
                    p = cur_text.find(c)
                    if p != -1:
                        cut_pos = min(cut_pos, p + (0 if c == "\n" else 1))
                sentence_out = cur_text[:cut_pos]
                clean_sentence = " ".join(sentence_out.replace("\n", " ").split()).strip()
                clean_sentence_stripped = ENUM_PREFIX_RE.sub("", clean_sentence).strip()

                # Filter too-short sentences
                if len(clean_sentence_stripped) < min_sentence_chars:
                    logger.info("[SKIP] Too short sentence: '%s' (%d chars)", clean_sentence, len(clean_sentence_stripped))
                    # reset this sentence and try again (continue decoding more tokens)
                    # slight tweak: allow continuing to grow this sentence
                    continue

                # Accept sentence:
                piece = sentence_out if sentence_out.startswith(" ") else " " + sentence_out
                generated_text += piece
                # Update accepted_ids to include what we just generated (to keep continuity)
                accepted_ids = _serialize_chat(
                    tok,
                    messages + [{"role": "assistant", "content": generated_text}],
                    add_generation_prompt=True,
                )
                sentence_count += 1
                logger.info("[Add] sentence: %s", clean_sentence)
                break  # proceed to next sentence

        loops += 1

    logger.info("[DONE] Max sentences reached.")
    return _finish(generated_text.strip(), logger, t0, eos_early=False, stopped_by_budget=True)


# ---------------------------------------------------------------------------
# Quick CLI demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Example conversation (Llama-2/3 chat styles supported)
    MESSAGES = [
        # {"role": "system", "content": "You are a helpful, concise assistant."},
        {"role": "user", "content": "Humans are social"},
    ]

    # MODEL = "../models/Llama-3.2-1B-Instruct"
    MODEL = "../models/Llama-2-7B-Chat-fp16"

    out = generate_instruct(
        messages=MESSAGES,
        model_path=MODEL,
        temperature=0.8,
        top_p=0.9,
        max_sentences=3,
        max_new_tokens_total=128,
        device="auto",
        log_level="INFO",
    )

    print("\n=== FINAL ANSWER (ASSISTANT) ===")
    print(out)
