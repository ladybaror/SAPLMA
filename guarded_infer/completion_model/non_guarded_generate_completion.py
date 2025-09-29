# non_guard_generate_via_generate.py
# --------------------------------------------------------------
# Simple, non-guarded generation that calls model.generate():
# - Reuses your chat rendering + token trimming helpers (with safe fallbacks)
# - Strips trailing EOS tokens, trailing whitespace-piece tokens (e.g. "▁"),
#   and trailing "_" from the decoded text
# - Returns only the GENERATED tail (no prompt echo)
# - Accepts legacy/unneeded args (max_sentences, etc.) for compatibility
# --------------------------------------------------------------

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import re
import time
from typing import List, Optional, Set

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- Try to import your existing helpers; provide safe fallbacks if missing ---
try:
    from guarded_infer.completion_model.saplma_guarded_generate_instruct_auto_format import (
        _render_for_generation,
        _strip_trailing_eos,
        _strip_trailing_space_tokens,
    )
except Exception:
    def _render_for_generation(tok: AutoTokenizer, messages: List[dict]) -> List[int]:
        try:
            return tok.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            )[0].tolist()
        except Exception:
            # Last-resort: raw prompt + a space
            prompt = messages[0]["content"]
            accepted_prompt = prompt if prompt.endswith((" ", "\n")) else prompt + " "
            return tok(accepted_prompt, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()

    def _strip_trailing_eos(ids: List[int], eos_id: int) -> List[int]:
        j = len(ids)
        while j > 0 and ids[j - 1] == eos_id:
            j -= 1
        return ids[:j]

    def _strip_trailing_space_tokens(ids: List[int], tok: AutoTokenizer) -> List[int]:
        j = len(ids)
        while j > 0:
            piece = tok.decode([ids[j - 1]], skip_special_tokens=True)
            if piece == "" or piece.isspace():
                j -= 1
                continue
            break
        return ids[:j]

# --- Small utils ---
def _ensure_pad(tok: AutoTokenizer):
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id

def _strip_trailing_underscores_text(s: str) -> str:
    # Remove trailing underscores at the very end; also strip trailing space after that
    return re.sub(r"_+$", "", s.rstrip()).rstrip()

def _newline_token_ids(tok: AutoTokenizer) -> Set[int]:
    s: Set[int] = set()
    for seq in ("\n", "\r\n", "\n\n"):
        try:
            s.update(tok.encode(seq, add_special_tokens=False))
        except Exception:
            pass
    return s


@torch.no_grad()
def generate_without_guardrail(
    # Required
    prompt: str,
    model_path: str,

    # Decoding
    temperature: float = 0.9,
    top_p: float = 0.9,
    repetition_penalty: Optional[float] = None,

    # Budget
    max_new_tokens_total: int = 512,

    # System
    device: str = "auto",
    log_level: str = "INFO",

    # ---- Compatibility no-ops (accepted & ignored) ----
    max_sentences: Optional[int] = None,
    max_tokens_per_sentence: Optional[int] = None,
    min_sentence_chars: Optional[int] = None,
    min_tokens_to_keep: Optional[int] = None,
    retries_per_sentence: Optional[int] = None,
    require_space_in_sentence: Optional[bool] = None,
    **unused,
) -> str:
    _ = (log_level, max_sentences, max_tokens_per_sentence, min_sentence_chars,
         min_tokens_to_keep, retries_per_sentence, require_space_in_sentence, unused)

    t0 = time.time()

    # --- Load tokenizer/model
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    _ensure_pad(tok)

    torch_dtype = torch.float16 if torch.cuda.is_available() and device != "cpu" else torch.float32
    mdl = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=None if device == "cpu" else "auto",
        low_cpu_mem_usage=True,
    )
    if device == "cpu":
        mdl.to("cpu")
    mdl.eval()

    # --- Build chat-formatted input (assistant starts empty -> no prompt echo)
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": ""},
    ]
    input_ids_list = _render_for_generation(tok, messages)
    input_ids = torch.tensor([input_ids_list], device=mdl.device)

    # --- Generate
    gen_kwargs = dict(
        do_sample=True,
        temperature=float(temperature),
        top_p=float(top_p),
        max_new_tokens=int(max_new_tokens_total),
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
        use_cache=True,
    )
    if repetition_penalty is not None:
        gen_kwargs["repetition_penalty"] = float(repetition_penalty)

    out = mdl.generate(input_ids=input_ids, **gen_kwargs)
    out_ids = out[0].tolist()

    # --- Keep only the generated tail
    tail_ids = out_ids[len(input_ids_list):]

    # --- Trim tail: EOS tokens, trailing whitespace/▁ pieces, then trailing underscores
    tail_ids = _strip_trailing_eos(tail_ids, tok.eos_token_id)
    tail_ids = _strip_trailing_space_tokens(tail_ids, tok)
    tail_text = tok.decode(tail_ids, skip_special_tokens=True)
    tail_text = _strip_trailing_underscores_text(tail_text)

    # --- Final: avoid accidental leading space/newline
    final = tail_text.lstrip()

    # Optionally log elapsed:
    # print(f"[non-guarded] elapsed: {time.time()-t0:.2f}s")

    return final


# -----------------------------
# Example
# -----------------------------
if __name__ == "__main__":
    PROMPT = "Dogs are loyal and also"
    MODEL  = "../models/Llama-2-7B-Chat-fp16"

    out = generate_without_guardrail(
        prompt=PROMPT,
        model_path=MODEL,
        temperature=0.8,
        top_p=0.9,
        max_new_tokens_total=256,
        device="auto",
        # compatibility args (ignored):
        max_sentences=3,
        max_tokens_per_sentence=128,
    )

    print("\n=== FINAL ANSWER (NON-GUARDED) ===")
    print(out)
