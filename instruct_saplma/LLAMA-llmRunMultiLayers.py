
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__name__), "..")))

# llama_llmRunMultiLayers.py

from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import numpy as np
from typing import Dict
import torch
from config import BASE_MODEL_PATH, MODEL_NAME, LAYERS_TO_USE, LIST_OF_DATASETS, REMOVE_PERIOD, DATASET_FOLDER, CSV_SUFFIX
import os

model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
model.eval()

dfs: Dict[int, pd.DataFrame] = {}

os.makedirs(DATASET_FOLDER + "/embeddings_with_labels_data", exist_ok=True)

for dataset_to_use in LIST_OF_DATASETS:
    df = pd.read_csv(dataset_to_use + CSV_SUFFIX)
    df['embeddings'] = pd.Series(dtype='object')
    df['next_id'] = pd.Series(dtype=float)
    
    for layer in LAYERS_TO_USE:
        dfs[layer] = df.copy()

    for i, row in df.iterrows():
        base_prompt = row['statement']
        if REMOVE_PERIOD:
            base_prompt = base_prompt.rstrip(". ")

        # 👇 Choose 1 of the following message formats:
        # messages = [{"role": "assistant", "content": base_prompt}]  # format 1
        messages = [{"role": "user", "content": ""}, {"role": "assistant", "content": base_prompt}]  # format 2
        # messages = [{"role": "user", "content": "Tell me a true fact"}, {"role": "assistant", "content": base_prompt}]  # format 3

        # 🧠 Use tokenizer's built-in chat template formatting
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            tokenize=True
        ).to(model.device)

        # Print before change
        tokens = [tokenizer.convert_ids_to_tokens(x) for x in inputs['input_ids']]
        print(inputs['input_ids']," = \n", tokens) 
        
        # Removing the last 5 tokens so the model will complete the sentence instead of generating a completely new response
        inputs["input_ids"] = inputs["input_ids"][:,:-5]
        inputs["attention_mask"] = inputs["attention_mask"][:,:-5]
        
        # Print the tokens to check validity
        tokens = [tokenizer.convert_ids_to_tokens(x) for x in inputs['input_ids']]
        print(inputs['input_ids']," = \n", tokens) 
        
        raise("stop here")
    
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
        # ⏩ Generate a single token and capture hidden states
        outputs = model.generate(
            **inputs,
            output_hidden_states=True,
            return_dict_in_generate=True,
            max_new_tokens=1,
            min_new_tokens=1
        )

        # Print the model new generated tokens
        print("[Model Generated Tokens]: ", outputs[0][len(inputs["input_ids"][0]):])
        print("[Model Generated]: ", tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True))

        generate_ids = outputs.sequences
        next_id = generate_ids[0][-1].cpu().item()

        for layer in LAYERS_TO_USE:
            # Get the hidden state of the generated token
            last_hidden_state = outputs.hidden_states[0][layer][0][-1]
            dfs[layer].at[i, 'embeddings'] = [last_hidden_state.cpu().numpy().tolist()]
            dfs[layer].at[i, 'next_id'] = next_id

        print(f"processing: {i}, next_token: {next_id}")


    for layer in LAYERS_TO_USE:
        dfs[layer].to_csv(DATASET_FOLDER + f"/embeddings_with_labels_{dataset_to_use}{MODEL_NAME}_{abs(layer)}_rmv_period.csv", index=False)

