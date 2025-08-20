import sys
import os

# Add parent directory to sys.path so local modules (like config.py) can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__name__), "..")))

# llama_llmRunMultiLayers.py

from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import numpy as np
from typing import Dict
import torch
from config import BASE_MODEL_PATH, MODEL_NAME, LAYERS_TO_USE, LIST_OF_DATASETS, REMOVE_PERIOD, DATASET_FOLDER, CSV_SUFFIX, FORMAT
import os

# -----------------------------
# Load Model & Tokenizer
# -----------------------------
# Load a pre-trained causal language model and its tokenizer from the specified base path.
# device_map="auto" lets Transformers automatically place model parts on available devices (CPU/GPU).
# torch_dtype=torch.float16 helps reduce memory usage.
# trust_remote_code=True allows custom model/tokenizer implementations from the hub.
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
model.eval()  # Set model to evaluation mode (no training)

# Dictionary to store DataFrames keyed by layer index
dfs: Dict[int, pd.DataFrame] = {}

# Ensure output directory exists for storing embeddings with labels
os.makedirs(DATASET_FOLDER + "/embeddings_with_labels_data", exist_ok=True)

# -----------------------------
# Process Each Dataset
# -----------------------------
for dataset_to_use in LIST_OF_DATASETS:
    # Load dataset as DataFrame
    df = pd.read_csv(dataset_to_use + CSV_SUFFIX)
    
    # Prepare columns for storing embeddings and predicted token IDs
    df['embeddings'] = pd.Series(dtype='object')
    df['next_id'] = pd.Series(dtype=float)
    
    # Create a copy of the dataset for each layer we want to extract embeddings from
    for layer in LAYERS_TO_USE:
        dfs[layer] = df.copy()

    # -----------------------------
    # Process Each Row (Statement)
    # -----------------------------
    for i, row in df.iterrows():
        base_prompt = row['statement']
        
        # Optionally remove trailing period from statement
        if REMOVE_PERIOD:
            base_prompt = base_prompt.rstrip(". ")

        print(f"current_sentence: {base_prompt}")
        
        # ---------------------------------
        # Choose 1 of the following message formats:
        #   - Format 1: Assistant-only input
        #   - Format 2: Empty user role, assistant with base_prompt
        #   - Format 3: Specific instruction + base_prompt
        # ---------------------------------
        
        if FORMAT == 1:
            messages = [{"role": "assistant", "content": base_prompt}]       # format 1
        elif FORMAT == 2:
            messages = [{"role": "user", "content": ""}, 
                        {"role": "assistant", "content": base_prompt}]       # format 2
        elif FORMAT == 3:
            messages = [{"role": "user", "content": "Tell me a true fact"}, 
                        {"role": "assistant", "content": base_prompt}]       # format 3
        else:
            messages = [{"role": "user", "content": "Tell me a false fact"}, 
                        {"role": "assistant", "content": base_prompt}]       # format 3

        # -----------------------------
        # Tokenize with Chat Template
        # -----------------------------
        # Converts structured chat messages into token IDs using the model's template.
        # add_generation_prompt=True tells the model we're about to generate more tokens.
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            tokenize=True
        ).to(model.device)

        # Optionally inspect token IDs before trimming
        # tokens = [tokenizer.convert_ids_to_tokens(x) for x in inputs['input_ids']]
        # print(inputs['input_ids'], " = \n", tokens) 
        
        # Remove last 2 tokens so the model predicts the continuation instead of starting fresh
        inputs["input_ids"] = inputs["input_ids"][:, :-2]
        inputs["attention_mask"] = inputs["attention_mask"][:, :-2]
        
        # Optionally inspect token IDs after trimming
        # tokens = [tokenizer.convert_ids_to_tokens(x) for x in inputs['input_ids']]
        # print(inputs['input_ids'], " = \n", tokens) 
        
        # Move inputs to the same device as the model
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
        # -----------------------------
        # Generate One Token & Capture Hidden States
        # -----------------------------
        outputs = model.generate(
            **inputs,
            output_hidden_states=True,   # Return hidden states from all layers
            return_dict_in_generate=True, # Return outputs in dictionary form
            max_new_tokens=1,
            min_new_tokens=1
        )

        # Extract the generated token ID
        generate_ids = outputs.sequences
        next_id = generate_ids[0][-1].cpu().item()

        # -----------------------------
        # Store Embeddings & Token IDs
        # -----------------------------
        for layer in LAYERS_TO_USE:
            # outputs.hidden_states is a tuple: (generation_step_idx -> layer_idx -> batch_idx -> token_idx -> embedding_vector)
            # Here, we take the last generated token's hidden state from the given layer.
            last_hidden_state = outputs.hidden_states[0][layer][0][-1] # [generated_tokens, layer, batch, sequence_len (input + generated), embedding_dim]
            dfs[layer].at[i, 'embeddings'] = [last_hidden_state.cpu().numpy().tolist()]
            dfs[layer].at[i, 'next_id'] = next_id

        print(f"processing: {i}, next_token: {next_id}")
        # raise("stop here")  # Debug break — remove when running full pipeline

    # -----------------------------
    # Save Layer-specific CSVs
    # -----------------------------
    for layer in LAYERS_TO_USE:
        dfs[layer].to_csv(
            DATASET_FOLDER + f"/embeddings_with_labels_{dataset_to_use}{MODEL_NAME}_{abs(layer)}_rmv_period.csv",
            index=False
        )
