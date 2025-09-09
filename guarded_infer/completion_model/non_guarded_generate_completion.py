import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__name__), "..")))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import re
import logging


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

def _finish(text: str, log: logging.Logger, t0: float,
            sentences_accepted: int, eos_early: bool, stopped_by_budget: bool) -> str:
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
    
    # ------ State ------
    # Ensure a trailing space so the model can start the first new token cleanly.    <<<<<<<------- TODO - Check if needed
    accepted_prompt = prompt if prompt.endswith((" ", "\n")) else prompt + " "
    
    accepted_ids = tokenizer(accepted_prompt, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()

    # print(tokenizer.decode(accepted_ids, skip_special_tokens=True))
    
    SENT_END_CHARS = (".", "!", "?")
    total_token_count = 0
    
    for s_idx in range(max_sentences):
        curr_tok_count = 0
        while True:
            # Global budget check
            if total_token_count + curr_tok_count >= max_new_tokens_total:
                pass
    return "FINISHED"

if __name__ == "__main__":
    PROMPT = "Humans are sociable creatures"
    
    MODEL  = "../models/Llama-2-7B-Chat-fp16"
    
    out = generate_without_guardrail(
        prompt=PROMPT,
        model_path=MODEL
    )

    print("\n=== FINAL ANSWER (CUMULATIVE SCOPE) ===")
    print(out)
