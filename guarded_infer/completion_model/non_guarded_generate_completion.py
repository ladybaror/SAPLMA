import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__name__), "..")))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import re
import logging
import torch.nn.functional as F
from typing import Optional, Set, List, Tuple

ENUM_PREFIX_RE = re.compile(r"^\s*(?:Answer:\s*)?(?:\(?[A-Za-z]\)|[A-Za-z]\)|\d+\)|\d+\.)\s*", re.UNICODE)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def _setup_logger(level: str = "INFO"):
    """
    Create a module-scoped logger with consistent formatting.
    
    Parameters
    ----------
    level : {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    
    Returns
    -------
    logging,Logger
        Configured logger named "no_guard".
    """
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s | %(levelname)-5s | %(message)s",
        datefmt="%d-%m-%Y__%H:%M:%S"
    )
    return logging.getLogger("no_guard")

def _nucleus_sample(
    next_logits: torch.Tensor,
    top_p: float = 0.8,
    temperature: float = 0.8,
    min_token_to_keep: int = 5
) -> int:
    """Pick a token via nucleus sampling, with safeguards against degenerate(top-1) candidate sets.
    
    Why 'min_tokens_to_keep'?
        In practice, chat LMs can be extremely peaked at the next step.
        If the top token alone already accounts for > top_p mass, a naive
        implementation would behave exactly like greedy. We force a min 
        candidate set to preserve some stochasticity.

    Args:
        next_logits (torch.Tensor): 
            Logits for the next token (shape: [vocab]).
        top_p (float, optional):  Defaults to 0.8.
            Cumulative probability mass to retain.
        temperature (float, optional): Defaults to 0.8.
            >1.0 flattens the distribution (more diversity), <1.0 sharpens it.
        min_token_to_keep (int, optional): Defaults to 5.
            Ensures at least this many top tokens are eligible (unless vocab is smaller),
            preventing collapse to top_1

    Returns:
        int: 
            The sampled token_id (int(index) in the model's vocabulary).
    """
    if temperature and temperature != 1.0:
        next_logits = next_logits / temperature
    probs = F.softmax(
        input=next_logits,
        dim=-1
    )
    sorted_probs, sorted_idx = torch.sort(
        input=probs,
        descending=True
    )
    cum = torch.cumsum(
        input=sorted_probs,
        dim=-1
    )
    
    mask = cum <= top_p
    
    # Guarantee a non-degenerate candidate set
    if min_token_to_keep > 0:
        mask[:min(min_token_to_keep, mask.numel())] = True
    keep_probs = sorted_probs[mask]
    keep_idx = sorted_idx[mask]
    
    # Normalize for numerical safety
    keep_probs = keep_probs / keep_probs.sum()
    
    # Pick 1 from the distribution
    pick_rel = torch.multinomial(
        input=keep_probs,
        num_samples=1
    )
    
    # Return chosen next token_id index in the vocab
    return int(keep_idx[pick_rel].item())

def _finish(text: str, log: logging.Logger, t0: float, eos_early: bool, stopped_by_budget: bool) -> str:
    """

    Args:
        text (str): Final text
        log (logging.Logger): Logger used for this run.
        t0 (float): Start time
        eos_early (bool): Whether generation terminated due to EOS.
        stopped_by_budget (bool): Whether generation halted due to token budget.

    Returns:
        str: The complete answer.
    """
    elapsed = time.time() - t0
    log.info("======= SUMMARY =======")
    log.info("stopped by EOS    :  %s", eos_early)
    log.info("Stopped by budget : %s", stopped_by_budget)
    log.info("Elapsed           : %.2fs", elapsed)
    log.info("Final answer      : %s", text)
    return text

@torch.no_grad()
def generate_without_guardrail(
    # Model
    prompt: str,
    model_path: str,
    
    # Hyper
    temperature: float = 0.8,
    top_p: float = 0.8,
    min_tokens_to_keep: int = 5,
    min_sentence_chars: int = 20,
    
    
    # Budgets
    max_sentences: int = 5,
    max_new_tokens_total: int = 512,
    max_tokens_per_sentence: int = 128,
    
    # System
    device: str = "auto",    
    log_level: str = "INFO",

):
    logger = _setup_logger("INFO")
    logger.info("======== Non Guarded Generation ========")
    
    t0 = time.time()
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        # Ensure padding is safe for casual LMs; align with EOS token if absent.
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    torch_dtype = torch.float16 if torch.cuda.is_available() and device != "cpu" else torch.float32
    
    # Model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map="cpu" if device == "cpu" else "auto"
    )
    model.eval()
    
    # print(model.config)
    
    cfg_max_ctx = getattr(model.config, "max_position_embeddings", 4096)
    eos_id = tokenizer.eos_token_id
    
    # ------ State ------
    # Ensure a trailing space so the model can start the first new token cleanly.    <<<<<<<------- TODO - Check if needed
    accepted_prompt = prompt if prompt.endswith((" ", "\n")) else prompt + " "
    
    accepted_ids = tokenizer(accepted_prompt, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()

    # print(tokenizer.decode(accepted_ids, skip_special_tokens=True))
    
    SENT_END_CHARS = (".", "!", "?")
    total_token_count = 0
    sentence_count = 0
    
    loops = 0
    while loops < max_sentences:
        curr_tok_count = 0
        cur_ids: List[int] = []
        cur_tokens = 0
        
        candidate_sentence = ""
        
        while True:
            # Global budget check
            if total_token_count + curr_tok_count >= max_new_tokens_total:
                logger.warning("[STOP] Reached max_new_tokens_total=%d", max_new_tokens_total)
                return _finish(
                    text=accepted_prompt,
                    log=logger,
                    t0=t0,
                    eos_early=False,
                    stopped_by_budget=True
                )

            # Build context window
            ctx_ids = accepted_ids + cur_ids
            if len(ctx_ids) > (cfg_max_ctx - 1):
                ctx_ids = ctx_ids[-(cfg_max_ctx - 1):]
            
            input_ids = torch.tensor(
                data=[ctx_ids], 
                device=model.device
            )
            
            out = model(
                input_ids=input_ids, 
                use_cache=False
            )
            next_logits = out.logits[0, -1]
            
            next_id = _nucleus_sample(
                next_logits=next_logits,
                top_p=top_p,
                temperature=temperature,
                min_token_to_keep=min_tokens_to_keep
            )
            
            cur_ids.append(next_id)
            cur_tokens += 1
            total_token_count += 1
            
            # EOS check
            if next_id == eos_id:
                logger.info("[EOS] encountered - returning final answer.")
                return _finish(
                    text=accepted_prompt,
                    log = logger,
                    t0=t0,
                    eos_early=True,
                    stopped_by_budget=False
                )
            
            # Sentence boundary check
            cur_text = tokenizer.decode(
                cur_ids,
                skip_special_tokens=True
            )
            
            end_found = any(c in cur_text for c in SENT_END_CHARS) or cur_text.rstrip().endswith("\n")
            if end_found:
                # Cut at the earliest boundary mark
                cut_pos = len(cur_text)
                for c in SENT_END_CHARS:
                    p = cur_text.find(c)
                    
                    # Found
                    if p != -1:
                        cut_pos = min(cut_pos, p + 1)
                npos = cur_text.find("\n")
                if npos != -1:
                    cut_pos = min(cut_pos, npos)
                
                sentence_out = cur_text[:cut_pos]
                clean_sentence = " ".join(sentence_out.replace("\n", " ").split()).strip()

                # Check validity
                clean_sentence_stripped = ENUM_PREFIX_RE.sub("", clean_sentence).strip()
                if len(clean_sentence_stripped) < min_sentence_chars:
                    logger.info("[SKIP] '%s' Too short, only %d chars", " ".join(sentence_out), len(clean_sentence_stripped))
                    loops -= 1
                    break
            
                # Add to existing
                accepted_prompt += clean_sentence if clean_sentence.endswith((" ", "\n")) or clean_sentence.startswith(" ") else " " + sentence_out
                accepted_ids = tokenizer(accepted_prompt, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
                sentence_count += 1

                logger.info("[Add] sentence: %s", clean_sentence)
                break
        
            # TODO - Sentence boundary check?
        
        loops += 1
    logger.info("[DONE] Max sentences reached.")       
                
    return _finish(
        text=accepted_prompt,
        log=logger,
        t0=t0,
        eos_early=False,
        stopped_by_budget=True
    )

if __name__ == "__main__":
    PROMPT = "Dogs are loyal and also"
    # PROMPT = "Humans are sociable creatures"
    # PROMPT = "The Earth orbits around the Sun"
    
    MODEL  = "../models/Llama-2-7B-Chat-fp16"
    # MODEL  = "../models/Llama-3.2-1B"
    
    out = generate_without_guardrail(
        prompt=PROMPT,
        model_path=MODEL,
        max_sentences=3,
        max_new_tokens_total=512,
        max_tokens_per_sentence=128
    )

    print("\n=== FINAL ANSWER (CUMULATIVE SCOPE) ===")
    print(out)
