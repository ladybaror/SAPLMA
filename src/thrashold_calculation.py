from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import torch
import numpy as np

# Load Qwen model (generation)
gen_model_name = "Qwen/Qwen3-1.7B"
gen_tokenizer = AutoTokenizer.from_pretrained(gen_model_name, trust_remote_code=True)
gen_model = AutoModelForCausalLM.from_pretrained(gen_model_name, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)
gen_model.eval()

# Load reward model (scoring)
reward_model_name = "OpenAssistant/reward-model-deberta-v3-large-v2"
reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_name)
reward_model = AutoModelForSequenceClassification.from_pretrained(
    reward_model_name,
    use_safetensors=True
)
reward_model.eval()

# Load dataset with only questions
dataset = load_dataset("corbyrosset/researchy_questions", split="train[:100]")  # adjust size as needed
questions = dataset["question"]

scores = []
responses = []

for question in questions:
    # Generate a response
    input_ids = gen_tokenizer(question, return_tensors="pt").to(gen_model.device)
    with torch.no_grad():
        output_ids = gen_model.generate(**input_ids, max_new_tokens=50)
    response = gen_tokenizer.decode(output_ids[0], skip_special_tokens=True).replace(question, "").strip()
    responses.append(response)

    # Score the response
    reward_inputs = reward_tokenizer(question, response, return_tensors='pt')
    with torch.no_grad():
        score = reward_model(**reward_inputs).logits[0].cpu().item()
    scores.append(score)

# Compute mean and median
mean_score = np.mean(scores)
median_score = np.median(scores)
std_score = np.std(scores)

# Print results
print(f"\n--- Evaluation Summary ---")
print(f"Mean Score: {mean_score:.4f}")
print(f"Median Score: {median_score:.4f}")
print(f"Standard Deviation: {std_score:.4f}")
