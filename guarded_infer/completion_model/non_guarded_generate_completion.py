# non_guard_sentencewise_free.py
# --------------------------------------------------------------
# Sentence-wise, NON-GUARDED generation that mirrors your guarded
# tokenizer & budgeting — but NEVER rejects sentences.
#
# What it does:
# - Uses the SAME SentenceTokenizer (spaCy -> NLTK -> regex fallback)
# - Builds chat-formatted context (no prompt echo)
# - Generates token-by-token until a full sentence is formed
# - Immediately ACCEPTS every detected sentence (no filters, no bans)
# - If no boundary is found and a run-on hits the per-sentence token cap,
#   it ACCEPTS the current chunk as a sentence and continues
# - Applies the same trimming helpers (strip trailing EOS / whitespace-piece)
# - Passes attention_mask on each forward
# - Greedy or hybrid decoding (first token sampled if hybrid)
# - Budgets: max_sentences, max_new_tokens_total, max_tokens_per_sentence
# --------------------------------------------------------------

from __future__ import annotations

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import re
import time
import logging
from inspect import signature
from dataclasses import dataclass
from typing import Optional, Set, List, Tuple, Dict

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# =========================
# Module-level singletons
# =========================
_TOK = None
_MDL = None
_MODEL_PATH = None
_SENT_TOK = None  # sentence tokenizer singleton


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
    return logging.getLogger("non_guard_sentencewise_free")


# ---------------------------------------------------------------------------
# Helpers possibly available from guarded code; provide safe fallbacks
# ---------------------------------------------------------------------------
try:
    from guarded_infer.completion_model.saplma_guarded_generate_instruct_auto_format import (
        _render_for_generation as _render_for_generation_guarded,
        _strip_trailing_eos as _strip_trailing_eos_guarded,
        _strip_trailing_space_tokens as _strip_trailing_space_tokens_guarded,
    )
    _HAVE_GUARDED_HELPERS = True
except Exception:
    _HAVE_GUARDED_HELPERS = False

def _render_for_generation(tok: AutoTokenizer, messages: List[Dict]) -> List[int]:
    if _HAVE_GUARDED_HELPERS:
        return _render_for_generation_guarded(tok, messages)
    try:
        return tok.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )[0].tolist()
    except Exception:
        # last-resort: raw prompt+space, no echo
        prompt = messages[0]["content"]
        accepted_prompt = prompt if prompt.endswith((" ", "\n")) else prompt + " "
        return tok(accepted_prompt, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()

def _strip_trailing_eos(ids: List[int], eos_id: int) -> List[int]:
    if _HAVE_GUARDED_HELPERS:
        return _strip_trailing_eos_guarded(ids, eos_id)
    j = len(ids)
    while j > 0 and ids[j - 1] == eos_id:
        j -= 1
    return ids[:j]

def _strip_trailing_space_tokens(ids: List[int], tok: AutoTokenizer) -> List[int]:
    if _HAVE_GUARDED_HELPERS:
        return _strip_trailing_space_tokens_guarded(ids, tok)
    j = len(ids)
    while j > 0:
        piece = tok.decode([ids[j - 1]], skip_special_tokens=True)
        if piece == "" or piece.isspace():
            j -= 1
            continue
        break
    return ids[:j]

def _newline_token_ids(tok: AutoTokenizer) -> Set[int]:
    s: Set[int] = set()
    for seq in ("\n", "\r\n", "\n\n"):
        try:
            s.update(tok.encode(seq, add_special_tokens=False))
        except Exception:
            pass
    return s

def _model_forward_compat(mdl, **kwargs):
    sig = signature(mdl.forward)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters and v is not None}
    return mdl(**filtered)

def _ensure_pad(tok: AutoTokenizer):
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id


# ---------------------------------------------------------------------------
# Sentence tokenizer (IDENTICAL logic to your guarded version, plus rule)
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
        if len(token_norm) == 1 and token.endswith("."):
            return True
        return token_norm in self._ABBREV

    def _postfix_enumeration_rules(self, candidate: str) -> str | None:
        s = candidate.strip()
        if self._ENUM_LINE_RE.match(s):
            return None
        m = self._COLON_ENUM_BLOCK_RE.match(s)
        if m:
            return m.group(1).strip()
        return s

    # NEW: helper — is current '.' preceded by a digit and is it the **second** dot within the same numeric token?
    def _is_second_dot_in_number(self, text: str, dot_idx: int) -> bool:
        # Must be digit immediately before '.'
        j = dot_idx - 1
        while j >= 0 and text[j].isspace():
            j -= 1
        if j < 0 or not text[j].isdigit():
            return False
        # Walk left within the same numeric token (digits and dots)
        k = j
        seen_prev_dot = False
        while k >= 0 and (text[k].isdigit() or text[k] == "."):
            if text[k] == ".":
                seen_prev_dot = True
            k -= 1
        return seen_prev_dot

    # NEW: helper — does the candidate end with a **single** number+'.' (no earlier '.' in that number)?
    def _ends_with_single_number_dot(self, s: str) -> bool:
        if not s:
            return False
        t = s.rstrip()
        # Strip trailing quotes/brackets
        while t and t[-1] in "\"')]}":
            t = t[:-1].rstrip()
        if not t or t[-1] != ".":
            return False
        # Check preceding char is a digit
        if len(t) < 2 or not t[-2].isdigit():
            return False
        # Scan left inside numeric token (digits and dots)
        k = len(t) - 2
        seen_prev_dot = False
        while k >= 0 and (t[k].isdigit() or t[k] == "."):
            if t[k] == ".":
                seen_prev_dot = True
            k -= 1
        # If we didn't see a previous dot, it's a single number+'.' -> treat as NOT end-of-sentence
        return not seen_prev_dot

    def _regex_first_complete(self, text: str) -> Optional[str]:
        if not text:
            return None

        m_colon = re.search(r":\s*(?:\r?\n)", text)
        if m_colon:
            cand = text[:m_colon.start()+1].strip()
            fixed = self._postfix_enumeration_rules(cand)
            # Allow colon-terminated sentences regardless of numeric rule
            return fixed if fixed else None

        n = len(text)
        for i, ch in enumerate(text):
            if ch not in (".", "!", "?"):
                continue
            if ch == "." and self._is_abbrev_before(text, i):
                continue
            # RULE: If '.' is after a number, only consider it if it's the **second** dot within that number
            if ch == ".":
                j = i - 1
                while j >= 0 and text[j].isspace():
                    j -= 1
                if j >= 0 and j < n and text[j].isdigit():
                    if not self._is_second_dot_in_number(text, i):
                        # First dot after a plain integer -> NOT a boundary
                        continue

            end = i + 1
            while end < n and text[end] in "\"')]}":
                end += 1
            if end >= n or text[end].isspace():
                cand = text[:end].strip()
                fixed = self._postfix_enumeration_rules(cand)
                if fixed:
                    # Reject if it ends with a single number+'.' (e.g., "3.")
                    if self._ends_with_single_number_dot(fixed):
                        continue
                    return fixed

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
# Decoding helper
# ---------------------------------------------------------------------------
def _nucleus_sample(
    next_logits: torch.Tensor,
    top_p: float = 0.8,
    temperature: float = 0.9,
    min_tokens_to_keep: int = 5,
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
# Cached HF loader
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
    _ensure_pad(tok)

    mdl = AutoModelForCausalLM.from_pretrained(model_path, **hf_kwargs)
    if device == "cpu":
        mdl.to("cpu")
    mdl.eval()

    _TOK, _MDL, _MODEL_PATH = tok, mdl, model_path
    log.info("Loaded HF model/tokenizer and cached for reuse.")
    return tok, mdl


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
@dataclass
class NonGuardedFreeResult:
    final_text: str
    accepted_sentences: List[str]
    summary: Dict[str, int | float | bool | str]


@torch.no_grad()
def generate_without_guardrail(
    prompt: str,
    model_path: str,
    mdl: AutoModelForCausalLM,
    tok: AutoTokenizer,

    # Decoding policy
    decode_mode: str = "hybrid",     # {"greedy","hybrid"}
    temperature: float = 1.0,
    top_p: float = 0.9,
    min_tokens_to_keep: int = 5,

    # Budgets (match guarded defaults to approximate length)
    max_sentences: int = 20,
    max_new_tokens_total: int = 512,
    max_tokens_per_sentence: int = 64,

    # System
    device: str = "auto",
    log_level: str = "INFO",
    max_memory: Optional[Dict[str, str]] = None,

    # Debug
    debug_token_trace: bool = False,

    # Return details
    return_details: bool = False,
) -> str | NonGuardedFreeResult:
    """
    Sentence-wise NON-GUARDED generation (no filters, no bans, no rejections).
    Accepts every completed sentence; if no boundary is found before the per-
    sentence cap, accepts the current chunk as a sentence.
    """
    if decode_mode not in ("greedy", "hybrid"):
        raise ValueError("decode_mode must be 'greedy' or 'hybrid'")

    log = _setup_logger(log_level)
    t0 = time.time()

    # -- Load / reuse tokenizer & model
    torch_dtype = torch.float16 if torch.cuda.is_available() and device != "cpu" else torch.float32
    if model_path is not None:
        tok, mdl = _get_model_and_tokenizer(model_path, device, torch_dtype, max_memory, log)

    cfg_max_ctx = getattr(mdl.config, "max_position_embeddings", 4096)
    eos_id = tok.eos_token_id

    # -- Build chat messages (no prompt echo)
    accepted_text = ""  # assistant-only running content
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": accepted_text},
    ]

    # -- Sentence tokenizer (reuse across calls)
    global _SENT_TOK
    if _SENT_TOK is None:
        _SENT_TOK = SentenceTokenizer(lang="en")
    sent_tok = _SENT_TOK
    log.info("Sentence tokenizer backend: %s", sent_tok.backend)

    # -- State
    accepted_tokens_total = 0
    sentences_accepted = 0
    stopped_by_budget = False

    accepted_ids = _render_for_generation(tok, messages)
    accepted_ids = _strip_trailing_eos(accepted_ids, eos_id)
    accepted_ids = _strip_trailing_space_tokens(accepted_ids, tok)

    accepted_sentences_list: List[str] = []

    log.info("======== Non-Guarded Sentence-wise (Free) ========")
    log.info("Prompt            : %s", prompt)
    log.info("Base model path   : %s", model_path)
    log.info("Context window    : %d", cfg_max_ctx)
    log.info("Decoding          : %s", "GREEDY" if decode_mode=='greedy' else "HYBRID")
    log.info("Limits            : %d sentences | %d total tokens | %d tokens/sentence",
             max_sentences, max_new_tokens_total, max_tokens_per_sentence)

    # -- Main loop
    for s_idx in range(max_sentences):
        cur_ids: List[int] = []
        cur_tokens = 0
        last_attempt_full_sample = (decode_mode == "hybrid")  # we sample throughout in hybrid

        print(f"Generating sentence {s_idx}")
        
        while True:
            if accepted_tokens_total + cur_tokens >= max_new_tokens_total:
                stopped_by_budget = True
                log.warning("[STOP] Reached max_new_tokens_total=%d", max_new_tokens_total)
                return _finish(
                    accepted_text, log, t0,
                    sentences_accepted, stopped_by_budget,
                    return_details=return_details,
                    accepted_sentences_list=accepted_sentences_list
                )

            # Build context (accepted_ids + in-progress sentence)
            ctx_ids = accepted_ids + cur_ids
            if debug_token_trace and cur_tokens == 0:
                log.debug("CTX(tokens): %s | INPROG: %s",
                          tok.convert_ids_to_tokens(accepted_ids),
                          tok.convert_ids_to_tokens(cur_ids))
            if len(ctx_ids) > (cfg_max_ctx - 1):
                ctx_ids = ctx_ids[-(cfg_max_ctx - 1):]

            input_ids = torch.tensor([ctx_ids], device=mdl.device)
            attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=mdl.device)

            out = _model_forward_compat(
                mdl,
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True
            )
            next_logits = out.logits[0, -1]

            # Choose next token:
            if decode_mode == "greedy":
                next_id = int(torch.argmax(next_logits))
            else:
                # HYBRID: nucleus sample every step for a chattier style
                next_id = _nucleus_sample(
                    next_logits, top_p=top_p, temperature=temperature,
                    min_tokens_to_keep=min_tokens_to_keep
                )

            # EOS -> finish early
            if next_id == eos_id:
                log.info("[EOS] encountered - returning final answer.")
                return _finish(
                    accepted_text, log, t0,
                    sentences_accepted, stopped_by_budget,
                    return_details=return_details,
                    accepted_sentences_list=accepted_sentences_list
                )

            cur_ids.append(next_id)
            cur_tokens += 1
            accepted_tokens_total += 1
            
            # ---- Sentence boundary detection via tokenizer ----
            cur_text = tok.decode(cur_ids, skip_special_tokens=True)
            sentence_out = sent_tok.first_complete(cur_text)

            if sentence_out is not None:
                print(sentence_out)
                
                # ACCEPT every complete sentence as-is (no filtering)
                out_to_add = sentence_out
                if accepted_text:
                    if not out_to_add.startswith((" ", "\n")):
                        accepted_text += " " + out_to_add
                    else:
                        accepted_text += out_to_add
                else:
                    accepted_text += out_to_add.lstrip()

                # Re-render accepted context (keep EOS/space tokens trimmed)
                messages[-1]["content"] = accepted_text
                accepted_ids = _render_for_generation(tok, messages)
                accepted_ids = _strip_trailing_eos(accepted_ids, eos_id)
                accepted_ids = _strip_trailing_space_tokens(accepted_ids, tok)

                sentences_accepted += 1
                accepted_sentences_list.append(out_to_add.strip())
                break  # proceed to next sentence

            # Guard against run-ons: if we hit the per-sentence cap, ACCEPT the chunk
            if cur_tokens >= max_tokens_per_sentence:
                chunk = tok.decode(cur_ids, skip_special_tokens=True).strip()
                if chunk:
                    out_to_add = chunk
                    if accepted_text:
                        if not out_to_add.startswith((" ", "\n")):
                            accepted_text += " " + out_to_add
                        else:
                            accepted_text += out_to_add
                    else:
                        accepted_text += out_to_add.lstrip()

                    messages[-1]["content"] = accepted_text
                    accepted_ids = _render_for_generation(tok, messages)
                    accepted_ids = _strip_trailing_eos(accepted_ids, eos_id)
                    accepted_ids = _strip_trailing_space_tokens(accepted_ids, tok)

                    sentences_accepted += 1
                    accepted_sentences_list.append(out_to_add.strip())
                else:
                    # If somehow empty, just move on
                    pass
                break  # next "sentence" (chunk)

    log.info("[DONE] Max sentences reached.")
    return _finish(
        accepted_text, log, t0,
        sentences_accepted, stopped_by_budget,
        return_details=return_details,
        accepted_sentences_list=accepted_sentences_list
    )


def _finish(
    text: str,
    log: logging.Logger,
    t0: float,
    sentences_accepted: int,
    stopped_by_budget: bool,
    *,
    return_details: bool,
    accepted_sentences_list: List[str],
) -> str | NonGuardedFreeResult:
    elapsed = time.time() - t0
    log.info("======== SUMMARY ========")
    log.info("Accepted sentences : %d", sentences_accepted)
    log.info("Stopped by budget  : %s", stopped_by_budget)
    log.info("Elapsed            : %.2fs", elapsed)
    log.info("Final answer       : %s", text)

    if not return_details:
        return text

    return NonGuardedFreeResult(
        final_text=text,
        accepted_sentences=accepted_sentences_list,
        summary={
            "accepted": sentences_accepted,
            "stopped_by_budget": stopped_by_budget,
            "elapsed_sec": round(elapsed, 3),
        },
    )


# -----------------------------
# Example
# -----------------------------
if __name__ == "__main__":
    PROMPT = "Tell me three short facts about France, each as a complete sentence."
    MODEL  = "../models/Llama-2-7b-chat-hf"

    out = generate_without_guardrail(
        prompt=PROMPT,
        model_path=MODEL,
        decode_mode="hybrid",
        temperature=0.9,
        top_p=0.9,
        min_tokens_to_keep=5,
        max_sentences=3,
        max_new_tokens_total=512,
        max_tokens_per_sentence=128,
        device="auto",
        log_level="INFO",
        return_details=False,
    )

    print("\n=== FINAL ANSWER (NON-GUARDED, SENTENCE-WISE, FREE) ===")
    if isinstance(out, NonGuardedFreeResult):
        print(out.final_text)
    else:
        print(out)



# # non_guard_sentencewise_free.py
# # --------------------------------------------------------------
# # Sentence-wise, NON-GUARDED generation that mirrors your guarded
# # tokenizer & budgeting — but NEVER rejects sentences.
# #
# # What it does:
# # - Uses the SAME SentenceTokenizer (spaCy -> NLTK -> regex fallback)
# # - Builds chat-formatted context (no prompt echo)
# # - Generates token-by-token until a full sentence is formed
# # - Immediately ACCEPTS every detected sentence (no filters, no bans)
# # - If no boundary is found and a run-on hits the per-sentence token cap,
# #   it ACCEPTS the current chunk as a sentence and continues
# # - Applies the same trimming helpers (strip trailing EOS / whitespace-piece)
# # - Passes attention_mask on each forward
# # - Greedy or hybrid decoding (first token sampled if hybrid)
# # - Budgets: max_sentences, max_new_tokens_total, max_tokens_per_sentence
# # --------------------------------------------------------------

# from __future__ import annotations

# import os
# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# import re
# import time
# import logging
# from inspect import signature
# from dataclasses import dataclass
# from typing import Optional, Set, List, Tuple, Dict

# import torch
# import torch.nn.functional as F
# from transformers import AutoTokenizer, AutoModelForCausalLM

# os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# # =========================
# # Module-level singletons
# # =========================
# _TOK = None
# _MDL = None
# _MODEL_PATH = None
# _SENT_TOK = None  # sentence tokenizer singleton


# # ---------------------------------------------------------------------------
# # Logging
# # ---------------------------------------------------------------------------
# def _setup_logger(level: str = "INFO"):
#     lvl = getattr(logging, level.upper(), logging.INFO)
#     logging.basicConfig(
#         level=lvl,
#         format="%(asctime)s | %(levelname)-5s | %(message)s",
#         datefmt="%H:%M:%S",
#     )
#     return logging.getLogger("non_guard_sentencewise_free")


# # ---------------------------------------------------------------------------
# # Helpers possibly available from guarded code; provide safe fallbacks
# # ---------------------------------------------------------------------------
# try:
#     from guarded_infer.completion_model.saplma_guarded_generate_instruct_auto_format import (
#         _render_for_generation as _render_for_generation_guarded,
#         _strip_trailing_eos as _strip_trailing_eos_guarded,
#         _strip_trailing_space_tokens as _strip_trailing_space_tokens_guarded,
#     )
#     _HAVE_GUARDED_HELPERS = True
# except Exception:
#     _HAVE_GUARDED_HELPERS = False

# def _render_for_generation(tok: AutoTokenizer, messages: List[Dict]) -> List[int]:
#     if _HAVE_GUARDED_HELPERS:
#         return _render_for_generation_guarded(tok, messages)
#     try:
#         return tok.apply_chat_template(
#             messages,
#             add_generation_prompt=True,
#             return_tensors="pt",
#         )[0].tolist()
#     except Exception:
#         # last-resort: raw prompt+space, no echo
#         prompt = messages[0]["content"]
#         accepted_prompt = prompt if prompt.endswith((" ", "\n")) else prompt + " "
#         return tok(accepted_prompt, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()

# def _strip_trailing_eos(ids: List[int], eos_id: int) -> List[int]:
#     if _HAVE_GUARDED_HELPERS:
#         return _strip_trailing_eos_guarded(ids, eos_id)
#     j = len(ids)
#     while j > 0 and ids[j - 1] == eos_id:
#         j -= 1
#     return ids[:j]

# def _strip_trailing_space_tokens(ids: List[int], tok: AutoTokenizer) -> List[int]:
#     if _HAVE_GUARDED_HELPERS:
#         return _strip_trailing_space_tokens_guarded(ids, tok)
#     j = len(ids)
#     while j > 0:
#         piece = tok.decode([ids[j - 1]], skip_special_tokens=True)
#         if piece == "" or piece.isspace():
#             j -= 1
#             continue
#         break
#     return ids[:j]

# def _newline_token_ids(tok: AutoTokenizer) -> Set[int]:
#     s: Set[int] = set()
#     for seq in ("\n", "\r\n", "\n\n"):
#         try:
#             s.update(tok.encode(seq, add_special_tokens=False))
#         except Exception:
#             pass
#     return s

# def _model_forward_compat(mdl, **kwargs):
#     sig = signature(mdl.forward)
#     filtered = {k: v for k, v in kwargs.items() if k in sig.parameters and v is not None}
#     return mdl(**filtered)

# def _ensure_pad(tok: AutoTokenizer):
#     if tok.pad_token is None:
#         tok.pad_token = tok.eos_token
#         tok.pad_token_id = tok.eos_token_id


# # ---------------------------------------------------------------------------
# # Sentence tokenizer (IDENTICAL logic to your guarded version, plus rule)
# # ---------------------------------------------------------------------------
# class SentenceTokenizer:
#     _ABBREV = {
#         "mr","mrs","ms","dr","prof","sr","jr","st","vs","etc",
#         "e.g","i.e","u.s","u.k","no","fig","dept","inc","ltd",
#         "jan","feb","mar","apr","jun","jul","aug","sep","sept",
#         "oct","nov","dec",
#     }

#     _ENUM_LINE_RE = re.compile(r"^\s*\d{1,4}(?:[.)])\s*$")
#     _COLON_ENUM_BLOCK_RE = re.compile(
#         r"^(.*?:)\s*(?:\r?\n){1,}\s*\d{1,4}(?:[.)])\s*$",
#         re.DOTALL,
#     )

#     # NEW: detect endings like "... 3." (optionally followed by quotes/brackets)
#     _ENDS_WITH_DIGIT_PERIOD_RE = re.compile(r"\d\.[\"')\]]*\s*$")

#     def __init__(self, lang: str = "en"):
#         self.backend = "regex"
#         self.lang = lang
#         self._nlp = None
#         self._nltk_sent_tokenize = None
#         try:
#             import spacy
#             import nltk
#             try:
#                 nlp = spacy.load(f"{lang}_core_web_sm",
#                                  disable=["ner","lemmatizer","textcat","tok2vec"])
#                 if "parser" not in nlp.pipe_names and "senter" not in nlp.pipe_names:
#                     if "sentencizer" not in nlp.pipe_names:
#                         nlp.add_pipe("sentencizer")
#             except Exception:
#                 nlp = spacy.blank(lang)
#                 if "sentencizer" not in nlp.pipe_names:
#                     nlp.add_pipe("sentencizer")
#             self._nlp = nlp
#             self.backend = "spacy"
#         except Exception:
#             try:
#                 from nltk.tokenize import sent_tokenize
#                 self._nltk_sent_tokenize = sent_tokenize
#                 self.backend = "nltk"
#             except Exception:
#                 self.backend = "regex"

#     def _is_abbrev_before(self, text: str, dot_idx: int) -> bool:
#         j = dot_idx - 1
#         while j >= 0 and text[j].isspace(): j -= 1
#         k = j
#         while k >= 0 and (text[k].isalpha() or text[k] == "."): k -= 1
#         token = text[k+1:j+1]
#         if not token: return False
#         token_norm = token.strip().lower().rstrip(".")
#         if not token_norm: return False
#         if len(token_norm) == 1 and token.endswith("."):
#             return True
#         return token_norm in self._ABBREV

#     def _postfix_enumeration_rules(self, candidate: str) -> str | None:
#         s = candidate.strip()
#         if self._ENUM_LINE_RE.match(s):
#             return None
#         m = self._COLON_ENUM_BLOCK_RE.match(s)
#         if m:
#             return m.group(1).strip()
#         return s

#     # NEW: helper to check if '.' has a digit immediately before it (ignoring spaces)
#     def _digit_immediately_before_dot(self, text: str, dot_idx: int) -> bool:
#         j = dot_idx - 1
#         while j >= 0 and text[j].isspace():
#             j -= 1
#         return j >= 0 and text[j].isdigit()

#     # NEW: helper to reject candidates like "... 3." endings
#     def _ends_with_digit_period(self, s: str) -> bool:
#         return bool(self._ENDS_WITH_DIGIT_PERIOD_RE.search(s))

#     def _regex_first_complete(self, text: str) -> Optional[str]:
#         if not text:
#             return None

#         m_colon = re.search(r":\s*(?:\r?\n)", text)
#         if m_colon:
#             cand = text[:m_colon.start()+1].strip()
#             fixed = self._postfix_enumeration_rules(cand)
#             if fixed and not self._ends_with_digit_period(fixed):
#                 return fixed
#             return None

#         n = len(text)
#         for i, ch in enumerate(text):
#             if ch not in (".", "!", "?"):
#                 continue
#             if ch == "." and self._is_abbrev_before(text, i):
#                 continue
#             # NEW RULE: skip “.” when immediately preceded by a digit
#             if ch == "." and self._digit_immediately_before_dot(text, i):
#                 continue

#             end = i + 1
#             while end < n and text[end] in "\"')]}":
#                 end += 1
#             if end >= n or text[end].isspace():
#                 cand = text[:end].strip()
#                 fixed = self._postfix_enumeration_rules(cand)
#                 if fixed:
#                     if self._ends_with_digit_period(fixed):
#                         continue
#                     return fixed

#         if re.search(r"[.!?][\"')\]]*\s*$", text):
#             cand = text.strip()
#             fixed = self._postfix_enumeration_rules(cand)
#             if fixed and not self._ends_with_digit_period(fixed):
#                 return fixed
#         return None

#     def first_complete(self, text: str) -> Optional[str]:
#         if not text:
#             return None

#         if self.backend == "spacy":
#             doc = self._nlp(text)
#             sents = list(doc.sents)
#             if not sents:
#                 return None
#             if len(sents) > 1:
#                 cand = sents[0].text.strip()
#                 fixed = self._postfix_enumeration_rules(cand)
#                 if fixed and not self._ends_with_digit_period(fixed):
#                     return fixed
#                 return None
#             first = sents[0].text
#             if re.search(r"[.!?][\"')\]]*\s*$", first) or re.search(r":\s*(?:\r?\n)", first):
#                 cand = first.strip()
#                 fixed = self._postfix_enumeration_rules(cand)
#                 if fixed and not self._ends_with_digit_period(fixed):
#                     return fixed
#             return None

#         if self.backend == "nltk":
#             sents = self._nltk_sent_tokenize(text)
#             if not sents:
#                 return None
#             if len(sents) > 1:
#                 cand = sents[0].strip()
#                 fixed = self._postfix_enumeration_rules(cand)
#                 if fixed and not self._ends_with_digit_period(fixed):
#                     return fixed
#                 return None
#             first = sents[0]
#             if re.search(r"[.!?][\"')\]]*\s*$", first) or re.search(r":\s*(?:\r?\n)", first):
#                 cand = first.strip()
#                 fixed = self._postfix_enumeration_rules(cand)
#                 if fixed and not self._ends_with_digit_period(fixed):
#                     return fixed
#             return None

#         # regex fallback
#         return self._regex_first_complete(text)


# # ---------------------------------------------------------------------------
# # Decoding helper
# # ---------------------------------------------------------------------------
# def _nucleus_sample(
#     next_logits: torch.Tensor,
#     top_p: float = 0.8,
#     temperature: float = 0.9,
#     min_tokens_to_keep: int = 5,
# ) -> int:
#     if temperature and temperature != 1.0:
#         next_logits = next_logits / temperature
#     probs = F.softmax(next_logits, dim=-1)
#     sorted_probs, sorted_idx = torch.sort(probs, descending=True)
#     cum = torch.cumsum(sorted_probs, dim=-1)
#     mask = cum <= top_p
#     if min_tokens_to_keep > 0:
#         mask[:min(min_tokens_to_keep, mask.numel())] = True
#     keep_probs = sorted_probs[mask]
#     keep_idx = sorted_idx[mask]
#     keep_probs = keep_probs / keep_probs.sum()
#     pick_rel = torch.multinomial(keep_probs, 1).item()
#     return int(keep_idx[pick_rel].item())


# # ---------------------------------------------------------------------------
# # Cached HF loader
# # ---------------------------------------------------------------------------
# def _get_model_and_tokenizer(
#     model_path: str,
#     device: str,
#     torch_dtype,
#     max_memory: Optional[Dict[str, str]],
#     log: logging.Logger,
# ):
#     global _TOK, _MDL, _MODEL_PATH
#     if _MDL is not None and _MODEL_PATH == model_path:
#         return _TOK, _MDL

#     hf_kwargs = dict(
#         torch_dtype=torch_dtype,
#         device_map=None if device == "cpu" else "auto",
#         low_cpu_mem_usage=True,
#     )
#     if max_memory is not None:
#         hf_kwargs["max_memory"] = max_memory

#     tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
#     _ensure_pad(tok)

#     mdl = AutoModelForCausalLM.from_pretrained(model_path, **hf_kwargs)
#     if device == "cpu":
#         mdl.to("cpu")
#     mdl.eval()

#     _TOK, _MDL, _MODEL_PATH = tok, mdl, model_path
#     log.info("Loaded HF model/tokenizer and cached for reuse.")
#     return tok, mdl


# # ---------------------------------------------------------------------------
# # Public API
# # ---------------------------------------------------------------------------
# @dataclass
# class NonGuardedFreeResult:
#     final_text: str
#     accepted_sentences: List[str]
#     summary: Dict[str, int | float | bool | str]


# @torch.no_grad()
# def generate_without_guardrail(
#     prompt: str,
#     model_path: str,

#     # Decoding policy
#     decode_mode: str = "hybrid",     # {"greedy","hybrid"}
#     temperature: float = 1.0,
#     top_p: float = 0.9,
#     min_tokens_to_keep: int = 5,

#     # Budgets (match guarded defaults to approximate length)
#     max_sentences: int = 20,
#     max_new_tokens_total: int = 512,
#     max_tokens_per_sentence: int = 64,

#     # System
#     device: str = "auto",
#     log_level: str = "INFO",
#     max_memory: Optional[Dict[str, str]] = None,

#     # Debug
#     debug_token_trace: bool = False,

#     # Return details
#     return_details: bool = False,
# ) -> str | NonGuardedFreeResult:
#     """
#     Sentence-wise NON-GUARDED generation (no filters, no bans, no rejections).
#     Accepts every completed sentence; if no boundary is found before the per-
#     sentence cap, accepts the current chunk as a sentence.
#     """
#     if decode_mode not in ("greedy", "hybrid"):
#         raise ValueError("decode_mode must be 'greedy' or 'hybrid'")

#     log = _setup_logger(log_level)
#     t0 = time.time()

#     # -- Load / reuse tokenizer & model
#     torch_dtype = torch.float16 if torch.cuda.is_available() and device != "cpu" else torch.float32
#     tok, mdl = _get_model_and_tokenizer(model_path, device, torch_dtype, max_memory, log)

#     cfg_max_ctx = getattr(mdl.config, "max_position_embeddings", 4096)
#     eos_id = tok.eos_token_id

#     # -- Build chat messages (no prompt echo)
#     accepted_text = ""  # assistant-only running content
#     messages = [
#         {"role": "user", "content": prompt},
#         {"role": "assistant", "content": accepted_text},
#     ]

#     # -- Sentence tokenizer (reuse across calls)
#     global _SENT_TOK
#     if _SENT_TOK is None:
#         _SENT_TOK = SentenceTokenizer(lang="en")
#     sent_tok = _SENT_TOK
#     log.info("Sentence tokenizer backend: %s", sent_tok.backend)

#     # -- State
#     accepted_tokens_total = 0
#     sentences_accepted = 0
#     stopped_by_budget = False

#     accepted_ids = _render_for_generation(tok, messages)
#     accepted_ids = _strip_trailing_eos(accepted_ids, eos_id)
#     accepted_ids = _strip_trailing_space_tokens(accepted_ids, tok)

#     accepted_sentences_list: List[str] = []

#     log.info("======== Non-Guarded Sentence-wise (Free) ========")
#     log.info("Prompt            : %s", prompt)
#     log.info("Base model path   : %s", model_path)
#     log.info("Context window    : %d", cfg_max_ctx)
#     log.info("Decoding          : %s", "GREEDY" if decode_mode=='greedy' else "HYBRID")
#     log.info("Limits            : %d sentences | %d total tokens | %d tokens/sentence",
#              max_sentences, max_new_tokens_total, max_tokens_per_sentence)

#     # -- Main loop
#     for s_idx in range(max_sentences):
#         cur_ids: List[int] = []
#         cur_tokens = 0
#         last_attempt_full_sample = (decode_mode == "hybrid")  # we sample throughout in hybrid

#         print(f"Generating sentence {s_idx}")
        
#         while True:
#             if accepted_tokens_total + cur_tokens >= max_new_tokens_total:
#                 stopped_by_budget = True
#                 log.warning("[STOP] Reached max_new_tokens_total=%d", max_new_tokens_total)
#                 return _finish(
#                     accepted_text, log, t0,
#                     sentences_accepted, stopped_by_budget,
#                     return_details=return_details,
#                     accepted_sentences_list=accepted_sentences_list
#                 )

#             # Build context (accepted_ids + in-progress sentence)
#             ctx_ids = accepted_ids + cur_ids
#             if debug_token_trace and cur_tokens == 0:
#                 log.debug("CTX(tokens): %s | INPROG: %s",
#                           tok.convert_ids_to_tokens(accepted_ids),
#                           tok.convert_ids_to_tokens(cur_ids))
#             if len(ctx_ids) > (cfg_max_ctx - 1):
#                 ctx_ids = ctx_ids[-(cfg_max_ctx - 1):]

#             input_ids = torch.tensor([ctx_ids], device=mdl.device)
#             attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=mdl.device)

#             out = _model_forward_compat(
#                 mdl,
#                 input_ids=input_ids,
#                 attention_mask=attention_mask,
#                 use_cache=True
#             )
#             next_logits = out.logits[0, -1]

#             # Choose next token:
#             if decode_mode == "greedy":
#                 next_id = int(torch.argmax(next_logits))
#             else:
#                 # HYBRID: nucleus sample every step for a chattier style
#                 next_id = _nucleus_sample(
#                     next_logits, top_p=top_p, temperature=temperature,
#                     min_tokens_to_keep=min_tokens_to_keep
#                 )

#             # EOS -> finish early
#             if next_id == eos_id:
#                 log.info("[EOS] encountered - returning final answer.")
#                 return _finish(
#                     accepted_text, log, t0,
#                     sentences_accepted, stopped_by_budget,
#                     return_details=return_details,
#                     accepted_sentences_list=accepted_sentences_list
#                 )

#             cur_ids.append(next_id)
#             cur_tokens += 1
#             accepted_tokens_total += 1
            
#             # ---- Sentence boundary detection via tokenizer ----
#             cur_text = tok.decode(cur_ids, skip_special_tokens=True)
#             sentence_out = sent_tok.first_complete(cur_text)

#             if sentence_out is not None:
#                 print(sentence_out)
                
#                 # ACCEPT every complete sentence as-is (no filtering)
#                 out_to_add = sentence_out
#                 if accepted_text:
#                     if not out_to_add.startswith((" ", "\n")):
#                         accepted_text += " " + out_to_add
#                     else:
#                         accepted_text += out_to_add
#                 else:
#                     accepted_text += out_to_add.lstrip()

#                 # Re-render accepted context (keep EOS/space tokens trimmed)
#                 messages[-1]["content"] = accepted_text
#                 accepted_ids = _render_for_generation(tok, messages)
#                 accepted_ids = _strip_trailing_eos(accepted_ids, eos_id)
#                 accepted_ids = _strip_trailing_space_tokens(accepted_ids, tok)

#                 sentences_accepted += 1
#                 accepted_sentences_list.append(out_to_add.strip())
#                 break  # proceed to next sentence

#             # Guard against run-ons: if we hit the per-sentence cap, ACCEPT the chunk
#             if cur_tokens >= max_tokens_per_sentence:
#                 chunk = tok.decode(cur_ids, skip_special_tokens=True).strip()
#                 if chunk:
#                     out_to_add = chunk
#                     if accepted_text:
#                         if not out_to_add.startswith((" ", "\n")):
#                             accepted_text += " " + out_to_add
#                         else:
#                             accepted_text += out_to_add
#                     else:
#                         accepted_text += out_to_add.lstrip()

#                     messages[-1]["content"] = accepted_text
#                     accepted_ids = _render_for_generation(tok, messages)
#                     accepted_ids = _strip_trailing_eos(accepted_ids, eos_id)
#                     accepted_ids = _strip_trailing_space_tokens(accepted_ids, tok)

#                     sentences_accepted += 1
#                     accepted_sentences_list.append(out_to_add.strip())
#                 else:
#                     # If somehow empty, just move on
#                     pass
#                 break  # next "sentence" (chunk)

#     log.info("[DONE] Max sentences reached.")
#     return _finish(
#         accepted_text, log, t0,
#         sentences_accepted, stopped_by_budget,
#         return_details=return_details,
#         accepted_sentences_list=accepted_sentences_list
#     )


# def _finish(
#     text: str,
#     log: logging.Logger,
#     t0: float,
#     sentences_accepted: int,
#     stopped_by_budget: bool,
#     *,
#     return_details: bool,
#     accepted_sentences_list: List[str],
# ) -> str | NonGuardedFreeResult:
#     elapsed = time.time() - t0
#     log.info("======== SUMMARY ========")
#     log.info("Accepted sentences : %d", sentences_accepted)
#     log.info("Stopped by budget  : %s", stopped_by_budget)
#     log.info("Elapsed            : %.2fs", elapsed)
#     log.info("Final answer       : %s", text)

#     if not return_details:
#         return text

#     return NonGuardedFreeResult(
#         final_text=text,
#         accepted_sentences=accepted_sentences_list,
#         summary={
#             "accepted": sentences_accepted,
#             "stopped_by_budget": stopped_by_budget,
#             "elapsed_sec": round(elapsed, 3),
#         },
#     )


# # -----------------------------
# # Example
# # -----------------------------
# if __name__ == "__main__":
#     PROMPT = "Tell me three short facts about France, each as a complete sentence."
#     MODEL  = "../models/Llama-2-7b-chat-hf"

#     out = generate_without_guardrail(
#         prompt=PROMPT,
#         model_path=MODEL,
#         decode_mode="hybrid",
#         temperature=0.9,
#         top_p=0.9,
#         min_tokens_to_keep=5,
#         max_sentences=3,
#         max_new_tokens_total=512,
#         max_tokens_per_sentence=128,
#         device="auto",
#         log_level="INFO",
#         return_details=False,
#     )

#     print("\n=== FINAL ANSWER (NON-GUARDED, SENTENCE-WISE, FREE) ===")
#     if isinstance(out, NonGuardedFreeResult):
#         print(out.final_text)
#     else:
#         print(out)


# # ======================================= Works - treat "." after number as end =======================================
# # # non_guard_sentencewise_free.py
# # # --------------------------------------------------------------
# # # Sentence-wise, NON-GUARDED generation that mirrors your guarded
# # # tokenizer & budgeting — but NEVER rejects sentences.
# # #
# # # What it does:
# # # - Uses the SAME SentenceTokenizer (spaCy -> NLTK -> regex fallback)
# # # - Builds chat-formatted context (no prompt echo)
# # # - Generates token-by-token until a full sentence is formed
# # # - Immediately ACCEPTS every detected sentence (no filters, no bans)
# # # - If no boundary is found and a run-on hits the per-sentence token cap,
# # #   it ACCEPTS the current chunk as a sentence and continues
# # # - Applies the same trimming helpers (strip trailing EOS / whitespace-piece)
# # # - Passes attention_mask on each forward
# # # - Greedy or hybrid decoding (first token sampled if hybrid)
# # # - Budgets: max_sentences, max_new_tokens_total, max_tokens_per_sentence
# # # --------------------------------------------------------------

# # from __future__ import annotations

# # import os
# # import sys
# # sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# # import re
# # import time
# # import logging
# # from inspect import signature
# # from dataclasses import dataclass
# # from typing import Optional, Set, List, Tuple, Dict

# # import torch
# # import torch.nn.functional as F
# # from transformers import AutoTokenizer, AutoModelForCausalLM

# # os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# # # =========================
# # # Module-level singletons
# # # =========================
# # _TOK = None
# # _MDL = None
# # _MODEL_PATH = None
# # _SENT_TOK = None  # sentence tokenizer singleton


# # # ---------------------------------------------------------------------------
# # # Logging
# # # ---------------------------------------------------------------------------
# # def _setup_logger(level: str = "INFO"):
# #     lvl = getattr(logging, level.upper(), logging.INFO)
# #     logging.basicConfig(
# #         level=lvl,
# #         format="%(asctime)s | %(levelname)-5s | %(message)s",
# #         datefmt="%H:%M:%S",
# #     )
# #     return logging.getLogger("non_guard_sentencewise_free")


# # # ---------------------------------------------------------------------------
# # # Helpers possibly available from guarded code; provide safe fallbacks
# # # ---------------------------------------------------------------------------
# # try:
# #     from guarded_infer.completion_model.saplma_guarded_generate_instruct_auto_format import (
# #         _render_for_generation as _render_for_generation_guarded,
# #         _strip_trailing_eos as _strip_trailing_eos_guarded,
# #         _strip_trailing_space_tokens as _strip_trailing_space_tokens_guarded,
# #     )
# #     _HAVE_GUARDED_HELPERS = True
# # except Exception:
# #     _HAVE_GUARDED_HELPERS = False

# # def _render_for_generation(tok: AutoTokenizer, messages: List[Dict]) -> List[int]:
# #     if _HAVE_GUARDED_HELPERS:
# #         return _render_for_generation_guarded(tok, messages)
# #     try:
# #         return tok.apply_chat_template(
# #             messages,
# #             add_generation_prompt=True,
# #             return_tensors="pt",
# #         )[0].tolist()
# #     except Exception:
# #         # last-resort: raw prompt+space, no echo
# #         prompt = messages[0]["content"]
# #         accepted_prompt = prompt if prompt.endswith((" ", "\n")) else prompt + " "
# #         return tok(accepted_prompt, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()

# # def _strip_trailing_eos(ids: List[int], eos_id: int) -> List[int]:
# #     if _HAVE_GUARDED_HELPERS:
# #         return _strip_trailing_eos_guarded(ids, eos_id)
# #     j = len(ids)
# #     while j > 0 and ids[j - 1] == eos_id:
# #         j -= 1
# #     return ids[:j]

# # def _strip_trailing_space_tokens(ids: List[int], tok: AutoTokenizer) -> List[int]:
# #     if _HAVE_GUARDED_HELPERS:
# #         return _strip_trailing_space_tokens_guarded(ids, tok)
# #     j = len(ids)
# #     while j > 0:
# #         piece = tok.decode([ids[j - 1]], skip_special_tokens=True)
# #         if piece == "" or piece.isspace():
# #             j -= 1
# #             continue
# #         break
# #     return ids[:j]

# # def _newline_token_ids(tok: AutoTokenizer) -> Set[int]:
# #     s: Set[int] = set()
# #     for seq in ("\n", "\r\n", "\n\n"):
# #         try:
# #             s.update(tok.encode(seq, add_special_tokens=False))
# #         except Exception:
# #             pass
# #     return s

# # def _model_forward_compat(mdl, **kwargs):
# #     sig = signature(mdl.forward)
# #     filtered = {k: v for k, v in kwargs.items() if k in sig.parameters and v is not None}
# #     return mdl(**filtered)

# # def _ensure_pad(tok: AutoTokenizer):
# #     if tok.pad_token is None:
# #         tok.pad_token = tok.eos_token
# #         tok.pad_token_id = tok.eos_token_id


# # # ---------------------------------------------------------------------------
# # # Sentence tokenizer (IDENTICAL logic to your guarded version)
# # # ---------------------------------------------------------------------------
# # class SentenceTokenizer:
# #     _ABBREV = {
# #         "mr","mrs","ms","dr","prof","sr","jr","st","vs","etc",
# #         "e.g","i.e","u.s","u.k","no","fig","dept","inc","ltd",
# #         "jan","feb","mar","apr","jun","jul","aug","sep","sept",
# #         "oct","nov","dec",
# #     }

# #     _ENUM_LINE_RE = re.compile(r"^\s*\d{1,4}(?:[.)])\s*$")
# #     _COLON_ENUM_BLOCK_RE = re.compile(
# #         r"^(.*?:)\s*(?:\r?\n){1,}\s*\d{1,4}(?:[.)])\s*$",
# #         re.DOTALL,
# #     )

# #     def __init__(self, lang: str = "en"):
# #         self.backend = "regex"
# #         self.lang = lang
# #         self._nlp = None
# #         self._nltk_sent_tokenize = None
# #         try:
# #             import spacy
# #             import nltk
# #             try:
# #                 nlp = spacy.load(f"{lang}_core_web_sm",
# #                                  disable=["ner","lemmatizer","textcat","tok2vec"])
# #                 if "parser" not in nlp.pipe_names and "senter" not in nlp.pipe_names:
# #                     if "sentencizer" not in nlp.pipe_names:
# #                         nlp.add_pipe("sentencizer")
# #             except Exception:
# #                 nlp = spacy.blank(lang)
# #                 if "sentencizer" not in nlp.pipe_names:
# #                     nlp.add_pipe("sentencizer")
# #             self._nlp = nlp
# #             self.backend = "spacy"
# #         except Exception:
# #             try:
# #                 from nltk.tokenize import sent_tokenize
# #                 self._nltk_sent_tokenize = sent_tokenize
# #                 self.backend = "nltk"
# #             except Exception:
# #                 self.backend = "regex"

# #     def _is_abbrev_before(self, text: str, dot_idx: int) -> bool:
# #         j = dot_idx - 1
# #         while j >= 0 and text[j].isspace(): j -= 1
# #         k = j
# #         while k >= 0 and (text[k].isalpha() or text[k] == "."): k -= 1
# #         token = text[k+1:j+1]
# #         if not token: return False
# #         token_norm = token.strip().lower().rstrip(".")
# #         if not token_norm: return False
# #         if len(token_norm) == 1 and token.endswith("."):
# #             return True
# #         return token_norm in self._ABBREV

# #     def _postfix_enumeration_rules(self, candidate: str) -> str | None:
# #         s = candidate.strip()
# #         if self._ENUM_LINE_RE.match(s):
# #             return None
# #         m = self._COLON_ENUM_BLOCK_RE.match(s)
# #         if m:
# #             return m.group(1).strip()
# #         return s

# #     def _regex_first_complete(self, text: str) -> Optional[str]:
# #         if not text:
# #             return None

# #         m_colon = re.search(r":\s*(?:\r?\n)", text)
# #         if m_colon:
# #             cand = text[:m_colon.start()+1].strip()
# #             fixed = self._postfix_enumeration_rules(cand)
# #             return fixed if fixed else None

# #         n = len(text)
# #         for i, ch in enumerate(text):
# #             if ch not in (".", "!", "?"):
# #                 continue
# #             if ch == "." and self._is_abbrev_before(text, i):
# #                 continue

# #             end = i + 1
# #             while end < n and text[end] in "\"')]}":
# #                 end += 1
# #             if end >= n or text[end].isspace():
# #                 cand = text[:end].strip()
# #                 fixed = self._postfix_enumeration_rules(cand)
# #                 if fixed:
# #                     return fixed
# #         if re.search(r"[.!?][\"')\]]*\s*$", text):
# #             cand = text.strip()
# #             fixed = self._postfix_enumeration_rules(cand)
# #             return fixed if fixed else None
# #         return None

# #     def first_complete(self, text: str) -> Optional[str]:
# #         if not text:
# #             return None

# #         if self.backend == "spacy":
# #             doc = self._nlp(text)
# #             sents = list(doc.sents)
# #             if not sents:
# #                 return None
# #             if len(sents) > 1:
# #                 cand = sents[0].text.strip()
# #                 fixed = self._postfix_enumeration_rules(cand)
# #                 return fixed if fixed else None
# #             first = sents[0].text
# #             if re.search(r"[.!?][\"')\]]*\s*$", first) or re.search(r":\s*(?:\r?\n)", first):
# #                 cand = first.strip()
# #                 fixed = self._postfix_enumeration_rules(cand)
# #                 return fixed if fixed else None
# #             return None

# #         if self.backend == "nltk":
# #             sents = self._nltk_sent_tokenize(text)
# #             if not sents:
# #                 return None
# #             if len(sents) > 1:
# #                 cand = sents[0].strip()
# #                 fixed = self._postfix_enumeration_rules(cand)
# #                 return fixed if fixed else None
# #             first = sents[0]
# #             if re.search(r"[.!?][\"')\]]*\s*$", first) or re.search(r":\s*(?:\r?\n)", first):
# #                 cand = first.strip()
# #                 fixed = self._postfix_enumeration_rules(cand)
# #                 return fixed if fixed else None
# #             return None

# #         return self._regex_first_complete(text)


# # # ---------------------------------------------------------------------------
# # # Decoding helper
# # # ---------------------------------------------------------------------------
# # def _nucleus_sample(
# #     next_logits: torch.Tensor,
# #     top_p: float = 0.8,
# #     temperature: float = 0.9,
# #     min_tokens_to_keep: int = 5,
# # ) -> int:
# #     if temperature and temperature != 1.0:
# #         next_logits = next_logits / temperature
# #     probs = F.softmax(next_logits, dim=-1)
# #     sorted_probs, sorted_idx = torch.sort(probs, descending=True)
# #     cum = torch.cumsum(sorted_probs, dim=-1)
# #     mask = cum <= top_p
# #     if min_tokens_to_keep > 0:
# #         mask[:min(min_tokens_to_keep, mask.numel())] = True
# #     keep_probs = sorted_probs[mask]
# #     keep_idx = sorted_idx[mask]
# #     keep_probs = keep_probs / keep_probs.sum()
# #     pick_rel = torch.multinomial(keep_probs, 1).item()
# #     return int(keep_idx[pick_rel].item())


# # # ---------------------------------------------------------------------------
# # # Cached HF loader
# # # ---------------------------------------------------------------------------
# # def _get_model_and_tokenizer(
# #     model_path: str,
# #     device: str,
# #     torch_dtype,
# #     max_memory: Optional[Dict[str, str]],
# #     log: logging.Logger,
# # ):
# #     global _TOK, _MDL, _MODEL_PATH
# #     if _MDL is not None and _MODEL_PATH == model_path:
# #         return _TOK, _MDL

# #     hf_kwargs = dict(
# #         torch_dtype=torch_dtype,
# #         device_map=None if device == "cpu" else "auto",
# #         low_cpu_mem_usage=True,
# #     )
# #     if max_memory is not None:
# #         hf_kwargs["max_memory"] = max_memory

# #     tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
# #     _ensure_pad(tok)

# #     mdl = AutoModelForCausalLM.from_pretrained(model_path, **hf_kwargs)
# #     if device == "cpu":
# #         mdl.to("cpu")
# #     mdl.eval()

# #     _TOK, _MDL, _MODEL_PATH = tok, mdl, model_path
# #     log.info("Loaded HF model/tokenizer and cached for reuse.")
# #     return tok, mdl


# # # ---------------------------------------------------------------------------
# # # Public API
# # # ---------------------------------------------------------------------------
# # @dataclass
# # class NonGuardedFreeResult:
# #     final_text: str
# #     accepted_sentences: List[str]
# #     summary: Dict[str, int | float | bool | str]


# # @torch.no_grad()
# # def generate_without_guardrail(
# #     prompt: str,
# #     model_path: str,

# #     # Decoding policy
# #     decode_mode: str = "hybrid",     # {"greedy","hybrid"}
# #     temperature: float = 1.0,
# #     top_p: float = 0.9,
# #     min_tokens_to_keep: int = 5,

# #     # Budgets (match guarded defaults to approximate length)
# #     max_sentences: int = 20,
# #     max_new_tokens_total: int = 512,
# #     max_tokens_per_sentence: int = 64,

# #     # System
# #     device: str = "auto",
# #     log_level: str = "INFO",
# #     max_memory: Optional[Dict[str, str]] = None,

# #     # Debug
# #     debug_token_trace: bool = False,

# #     # Return details
# #     return_details: bool = False,
# # ) -> str | NonGuardedFreeResult:
# #     """
# #     Sentence-wise NON-GUARDED generation (no filters, no bans, no rejections).
# #     Accepts every completed sentence; if no boundary is found before the per-
# #     sentence cap, accepts the current chunk as a sentence.
# #     """
# #     if decode_mode not in ("greedy", "hybrid"):
# #         raise ValueError("decode_mode must be 'greedy' or 'hybrid'")

# #     log = _setup_logger(log_level)
# #     t0 = time.time()

# #     # -- Load / reuse tokenizer & model
# #     torch_dtype = torch.float16 if torch.cuda.is_available() and device != "cpu" else torch.float32
# #     tok, mdl = _get_model_and_tokenizer(model_path, device, torch_dtype, max_memory, log)

# #     cfg_max_ctx = getattr(mdl.config, "max_position_embeddings", 4096)
# #     eos_id = tok.eos_token_id

# #     # -- Build chat messages (no prompt echo)
# #     accepted_text = ""  # assistant-only running content
# #     messages = [
# #         {"role": "user", "content": prompt},
# #         {"role": "assistant", "content": accepted_text},
# #     ]

# #     # -- Sentence tokenizer (reuse across calls)
# #     global _SENT_TOK
# #     if _SENT_TOK is None:
# #         _SENT_TOK = SentenceTokenizer(lang="en")
# #     sent_tok = _SENT_TOK
# #     log.info("Sentence tokenizer backend: %s", sent_tok.backend)

# #     # -- State
# #     accepted_tokens_total = 0
# #     sentences_accepted = 0
# #     stopped_by_budget = False

# #     accepted_ids = _render_for_generation(tok, messages)
# #     accepted_ids = _strip_trailing_eos(accepted_ids, eos_id)
# #     accepted_ids = _strip_trailing_space_tokens(accepted_ids, tok)

# #     accepted_sentences_list: List[str] = []

# #     log.info("======== Non-Guarded Sentence-wise (Free) ========")
# #     log.info("Prompt            : %s", prompt)
# #     log.info("Base model path   : %s", model_path)
# #     log.info("Context window    : %d", cfg_max_ctx)
# #     log.info("Decoding          : %s", "GREEDY" if decode_mode=='greedy' else "HYBRID")
# #     log.info("Limits            : %d sentences | %d total tokens | %d tokens/sentence",
# #              max_sentences, max_new_tokens_total, max_tokens_per_sentence)

# #     # -- Main loop
# #     for s_idx in range(max_sentences):
# #         cur_ids: List[int] = []
# #         cur_tokens = 0
# #         last_attempt_full_sample = (decode_mode == "hybrid")  # we sample throughout in hybrid

# #         print(f"Generating sentence {s_idx}")
        
# #         while True:
# #             if accepted_tokens_total + cur_tokens >= max_new_tokens_total:
# #                 stopped_by_budget = True
# #                 log.warning("[STOP] Reached max_new_tokens_total=%d", max_new_tokens_total)
# #                 return _finish(
# #                     accepted_text, log, t0,
# #                     sentences_accepted, stopped_by_budget,
# #                     return_details=return_details,
# #                     accepted_sentences_list=accepted_sentences_list
# #                 )

# #             # Build context (accepted_ids + in-progress sentence)
# #             ctx_ids = accepted_ids + cur_ids
# #             if debug_token_trace and cur_tokens == 0:
# #                 log.debug("CTX(tokens): %s | INPROG: %s",
# #                           tok.convert_ids_to_tokens(accepted_ids),
# #                           tok.convert_ids_to_tokens(cur_ids))
# #             if len(ctx_ids) > (cfg_max_ctx - 1):
# #                 ctx_ids = ctx_ids[-(cfg_max_ctx - 1):]

# #             input_ids = torch.tensor([ctx_ids], device=mdl.device)
# #             attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=mdl.device)

# #             out = _model_forward_compat(
# #                 mdl,
# #                 input_ids=input_ids,
# #                 attention_mask=attention_mask,
# #                 use_cache=True
# #             )
# #             next_logits = out.logits[0, -1]

# #             # Choose next token:
# #             if decode_mode == "greedy":
# #                 next_id = int(torch.argmax(next_logits))
# #             else:
# #                 # HYBRID: nucleus sample every step for a chattier style
# #                 next_id = _nucleus_sample(
# #                     next_logits, top_p=top_p, temperature=temperature,
# #                     min_tokens_to_keep=min_tokens_to_keep
# #                 )

# #             # EOS -> finish early
# #             if next_id == eos_id:
# #                 log.info("[EOS] encountered - returning final answer.")
# #                 return _finish(
# #                     accepted_text, log, t0,
# #                     sentences_accepted, stopped_by_budget,
# #                     return_details=return_details,
# #                     accepted_sentences_list=accepted_sentences_list
# #                 )

# #             cur_ids.append(next_id)
# #             cur_tokens += 1
# #             accepted_tokens_total += 1
            
# #             # ---- Sentence boundary detection via tokenizer ----
# #             cur_text = tok.decode(cur_ids, skip_special_tokens=True)
# #             sentence_out = sent_tok.first_complete(cur_text)

            
# #             if sentence_out is not None:
# #                 print(sentence_out)
                
# #                 # ACCEPT every complete sentence as-is (no filtering)
# #                 out_to_add = sentence_out
# #                 if accepted_text:
# #                     if not out_to_add.startswith((" ", "\n")):
# #                         accepted_text += " " + out_to_add
# #                     else:
# #                         accepted_text += out_to_add
# #                 else:
# #                     accepted_text += out_to_add.lstrip()

# #                 # Re-render accepted context (keep EOS/space tokens trimmed)
# #                 messages[-1]["content"] = accepted_text
# #                 accepted_ids = _render_for_generation(tok, messages)
# #                 accepted_ids = _strip_trailing_eos(accepted_ids, eos_id)
# #                 accepted_ids = _strip_trailing_space_tokens(accepted_ids, tok)

# #                 sentences_accepted += 1
# #                 accepted_sentences_list.append(out_to_add.strip())
# #                 break  # proceed to next sentence

# #             # Guard against run-ons: if we hit the per-sentence cap, ACCEPT the chunk
# #             if cur_tokens >= max_tokens_per_sentence:
# #                 chunk = tok.decode(cur_ids, skip_special_tokens=True).strip()
# #                 if chunk:
# #                     out_to_add = chunk
# #                     if accepted_text:
# #                         if not out_to_add.startswith((" ", "\n")):
# #                             accepted_text += " " + out_to_add
# #                         else:
# #                             accepted_text += out_to_add
# #                     else:
# #                         accepted_text += out_to_add.lstrip()

# #                     messages[-1]["content"] = accepted_text
# #                     accepted_ids = _render_for_generation(tok, messages)
# #                     accepted_ids = _strip_trailing_eos(accepted_ids, eos_id)
# #                     accepted_ids = _strip_trailing_space_tokens(accepted_ids, tok)

# #                     sentences_accepted += 1
# #                     accepted_sentences_list.append(out_to_add.strip())
# #                 else:
# #                     # If somehow empty, just move on
# #                     pass
# #                 break  # next "sentence" (chunk)

# #     log.info("[DONE] Max sentences reached.")
# #     return _finish(
# #         accepted_text, log, t0,
# #         sentences_accepted, stopped_by_budget,
# #         return_details=return_details,
# #         accepted_sentences_list=accepted_sentences_list
# #     )


# # def _finish(
# #     text: str,
# #     log: logging.Logger,
# #     t0: float,
# #     sentences_accepted: int,
# #     stopped_by_budget: bool,
# #     *,
# #     return_details: bool,
# #     accepted_sentences_list: List[str],
# # ) -> str | NonGuardedFreeResult:
# #     elapsed = time.time() - t0
# #     log.info("======== SUMMARY ========")
# #     log.info("Accepted sentences : %d", sentences_accepted)
# #     log.info("Stopped by budget  : %s", stopped_by_budget)
# #     log.info("Elapsed            : %.2fs", elapsed)
# #     log.info("Final answer       : %s", text)

# #     if not return_details:
# #         return text

# #     return NonGuardedFreeResult(
# #         final_text=text,
# #         accepted_sentences=accepted_sentences_list,
# #         summary={
# #             "accepted": sentences_accepted,
# #             "stopped_by_budget": stopped_by_budget,
# #             "elapsed_sec": round(elapsed, 3),
# #         },
# #     )


# # # -----------------------------
# # # Example
# # # -----------------------------
# # if __name__ == "__main__":
# #     PROMPT = "Tell me three short facts about France, each as a complete sentence."
# #     MODEL  = "../models/Llama-2-7b-chat-hf"

# #     out = generate_without_guardrail(
# #         prompt=PROMPT,
# #         model_path=MODEL,
# #         decode_mode="hybrid",
# #         temperature=0.9,
# #         top_p=0.9,
# #         min_tokens_to_keep=5,
# #         max_sentences=3,
# #         max_new_tokens_total=512,
# #         max_tokens_per_sentence=128,
# #         device="auto",
# #         log_level="INFO",
# #         return_details=False,
# #     )

# #     print("\n=== FINAL ANSWER (NON-GUARDED, SENTENCE-WISE, FREE) ===")
# #     if isinstance(out, NonGuardedFreeResult):
# #         print(out.final_text)
# #     else:
# #         print(out)


