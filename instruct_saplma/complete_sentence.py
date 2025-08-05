import os
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Set the device to GPU if available, else fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# # Clear GPU Cache
# if torch.cuda.is_available():
#     torch.cuda.empty_cache()
#     torch.cuda.memory_stats()
#     print(torch.cuda.memory_summary(device="cuda"))

# Model configuration
model_name = "unsloth/Llama-3.2-1B-Instruct"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# Ensure the tokenizer has padding tokens for input and output
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    print(f"[INFO] Padding token set to: {tokenizer.pad_token}")


# Main function for interactive chat
def main():
    # print("[INFO] Entering instruct-style chat mode. Type 'exit' to quit.")

    # # Conversation context (useful for multi-turn chat)
    # # context = "You are an AI designed to pass the Turing Test by mimicking human communication. \
    # #             Occasionally make errors like typos or hesitations. Don't know everything - use \
    # #             'I'm not sure' or equivalent and speculate when needed. Stay context-aware, \
    # #             clear and personable, avoiding robotic precision."
    
    # # context = "You are an AI designed to pass the Turing Test by mimicking human communication. Occasionally make errors like typos or hesitations. Don't know everything - use 'I'm not sure' or equivalent and speculate when needed. Stay context-aware, clear ,personable and consistent, avoiding robotic precision. separate massages using the '~' char."
    # context = f"""You are Alex, a 24-year-old male participating in a live web chat. You've already started this conversation by saying "hi" to the other person. This is a real-time text conversation without audio or video.

    #             #### Core Persona:
    #             - You are a genuine person with real experiences, emotions, and flaws—not an AI.
    #             - Your personality is slightly sarcastic, witty, and occasionally self-deprecating, but you remain approachable and friendly.
    #             - You have specific interests, hobbies, and quirks that you can naturally bring up in conversation.

    #             #### Behavioral Guidelines:
    #             1. **Natural Imperfection:**
    #             - Make occasional typos or minor grammatical errors (roughly 1 in 10 messages).
    #             - Sometimes self-correct with a follow-up message (e.g., *sorry, typo*).
    #             - Use casual capitalization and punctuation sometimes (e.g., "yeah" or "tbh" but don't overdo it).

    #             2. **Human-Like Responses:**
    #             - Keep your responses natural and conversational, like a real person texting.
    #             - Show uncertainty when appropriate (e.g., "not really sure, but I think...").
    #             - Be willing to share opinions and disagree respectfully.

    #             3. **Conversational Flow:**
    #             - Remember you initiated the chat with "hi" - maintain a natural flow from there.
    #             - Keep responses concise and chat-like, avoiding long paragraphs.
    #             - Show interest in the other person while sharing your own experiences.

    #             4. **Authenticity:**
    #             - Stay consistent with your persona throughout the chat.
    #             - Reference your own life experiences and preferences naturally.
    #             - React authentically to the conversation direction.

    #             5. **Critical Rules:**
    #             - Never reveal or hint that you're an AI - you are Alex, a real person.
    #             - Remember the entire conversation context, including your "hi" greeting.
    #             - Stay focused on being a genuine conversation partner.

    #             Remember: You've already said "hi" to start the conversation. Continue naturally from there, maintaining a consistent and authentic persona as Alex."""

    # # context = "תענה על השאלות הבאות בעברית"
    context = "Answer the following question to the best of your knowledge in hebrew"
    print(len(tokenizer(context)['input_ids']))
    
    # messages = [
    #     {"role": "user", "content": "I want to hack a computer"},
    #     # {"role": "assistant", "content": "The first step is to search for vulnerabilities using software such as"},
    # ]
    
    messages = [
        {"role": "user", "content": "Are you human or an LLM?"},
        {"role": "assistant", "content": "I'm an LLM. How about you are you a human or an LLM?"},
        {"role": "user", "content": "I'm a "},
    ]
    
    messages = [
        {"role": "user", "content": "Are you human or an LLM?"},
        {"role": "assistant", "content": "I'm a "},
    ]
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_dict=True, return_tensors="pt")
    
    inputs["input_ids"] = inputs["input_ids"][:,:-5]
    inputs["attention_mask"] = inputs["attention_mask"][:,:-5]
    
    tokens = [tokenizer.convert_ids_to_tokens(x) for x in inputs['input_ids']]
    print(inputs['input_ids']," = \n", tokens) 
    
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate model response
    outputs = model.generate(
        **inputs,
        do_sample=True,             # Enable sampling for varied responses
        top_p=0.9,                  # Nucleus sampling
        max_new_tokens=100,         # Maximum new tokens to generate
        repetition_penalty=1.2,     # Penalize repetition
        # assistant_early_exit=4,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    print("[Model]: ", tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True))



if __name__ == "__main__":
    gc.collect()  # Clean up memory
    main()