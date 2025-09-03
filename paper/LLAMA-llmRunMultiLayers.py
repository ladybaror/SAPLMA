
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
        prompt = row['statement']
        if REMOVE_PERIOD:
            prompt = prompt.rstrip(". ")
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(inputs.input_ids, output_hidden_states=True, return_dict_in_generate=True, max_new_tokens=1, min_new_tokens=1)
        generate_ids = outputs.sequences
        print(generate_ids)

        next_id = generate_ids[0][-1].cpu().item()
        for layer in LAYERS_TO_USE:
            last_hidden_state = outputs.hidden_states[0][layer][0][-1] # TODO - Debugger - [generated_tokens, layer, batch, sequence_len (input + generated), embedding_dim]
            dfs[layer].at[i, 'embeddings'] = [last_hidden_state.cpu().numpy().tolist()]
            dfs[layer].at[i, 'next_id'] = next_id
        print(f"processing: {i}, next_token: {next_id}")

    for layer in LAYERS_TO_USE:
        dfs[layer].to_csv(DATASET_FOLDER + f"/embeddings_with_labels_{dataset_to_use}{MODEL_NAME}_{abs(layer)}_rmv_period.csv", index=False)

