# llm_continue.py
from __future__ import annotations
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

PROMPT = "Dogs are loyal and also"
# MODEL  = "models/Llama-2-7B-Chat-fp16"
MODEL  = "models/Llama-2-7b-chat-hf"

# Generation knobs
MAX_NEW_TOKENS = 80
TEMPERATURE = 1.1
TOP_P = 0.95
REPETITION_PENALTY = 1.05
SEED = 1234  # for reproducible sampling; set None to disable

# If you want to format as chat (since it's a *chat* model), set this True.
# Otherwise we'll just do plain completion on your sentence.
USE_CHAT_TEMPLATE = False

def build_prompt_text(tok: AutoTokenizer, user_prompt: str, use_chat_template: bool) -> str:
    if use_chat_template and hasattr(tok, "apply_chat_template"):
        messages = [{"role": "user", "content": user_prompt}]
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # plain completion; adding a trailing space usually yields nicer flow
    return user_prompt if user_prompt.endswith((" ", "\n")) else user_prompt + " "

def first_sentence(text: str) -> str:
    # Return substring up to the earliest '. ! ?' or newline (inclusive)
    cut = len(text)
    for ch in (".", "!", "?"):
        p = text.find(ch)
        if p != -1:
            cut = min(cut, p + 1)
    npos = text.find("\n")
    if npos != -1:
        cut = min(cut, npos)
    return text[:cut] if cut < len(text) else text

def generate_once(
    mdl: AutoModelForCausalLM,
    tok: AutoTokenizer,
    prompt_text: str,
    *,
    do_sample: bool,
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_new_tokens: int = 80,
):
    inputs = tok(prompt_text, return_tensors="pt")
    inputs = {k: v.to(mdl.device) for k, v in inputs.items()}

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )
    if do_sample:
        gen_kwargs.update(temperature=temperature, top_p=top_p)

    with torch.no_grad():
        out = mdl.generate(**inputs, **gen_kwargs)

    # Strip the prompt part and decode only the newly generated tokens
    gen_only = out[0, inputs["input_ids"].shape[1]:]
    text = tok.decode(gen_only, skip_special_tokens=True)
    return text

def main():
    if SEED is not None:
        set_seed(SEED)
        torch.manual_seed(SEED)

    tok: AutoTokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    
    # Print all special tokens
    print("Special tokens:", tok.special_tokens_map)

    # Print the actual token strings
    print("All special tokens:", tok.all_special_tokens)

    # Print their IDs
    print("All special token IDs:", tok.all_special_ids)

    # Example: Access individual special tokens
    print("BOS token:", tok.bos_token, "->", tok.bos_token_id)
    print("EOS token:", tok.eos_token, "->", tok.eos_token_id)
    print("PAD token:", tok.pad_token, "->", tok.pad_token_id)

    # if tok.pad_token is None:
    #     tok.pad_token = tok.eos_token
    #     tok.pad_token_id = tok.eos_token_id

    # torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    # mdl = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch_dtype, device_map="auto")
    # mdl.eval()

    # prompt_text = build_prompt_text(tok, PROMPT, USE_CHAT_TEMPLATE)

    # # 1) Greedy continuation (deterministic)
    # greedy_text = generate_once(
    #     mdl, tok, prompt_text,
    #     do_sample=False,
    #     max_new_tokens=MAX_NEW_TOKENS,
    # )

    # # 2) Nucleus sampling continuation (stochastic)
    # sampled_text = generate_once(
    #     mdl, tok, prompt_text,
    #     do_sample=True,
    #     temperature=TEMPERATURE,
    #     top_p=TOP_P,
    #     max_new_tokens=MAX_NEW_TOKENS,
    # )

    # print("=== PROMPT ===")
    # print(repr(PROMPT))
    # print("\n=== GREEDY (full) ===")
    # print(greedy_text)
    # print("\n--- GREEDY (first sentence) ---")
    # print(first_sentence(greedy_text))

    # print("\n=== SAMPLED (full) ===")
    # print(sampled_text)
    # print("\n--- SAMPLED (first sentence) ---")
    # print(first_sentence(sampled_text))

if __name__ == "__main__":
    main()
