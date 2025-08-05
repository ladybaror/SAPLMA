# config.py

from pathlib import Path

# ----- Paths -----
BASE_MODEL_PATH = "models/Llama-2-7B-Chat-fp16"
DATASET_FOLDER = "data/capital_true_false"
CSV_SUFFIX = "_true_false.csv"

# ----- Model -----
MODEL_NAME = "LLAMA7"  # e.g., "LLAMA7", "1.3b", etc.
LAYERS_TO_USE = [-12]  # e.g., [-1, -4, -8, -12, -16]

# ----- Datasets -----
LIST_OF_DATASETS = [
    "data/capitals",
    "data/inventions", 
    "data/elements", 
    # "data/animals", 
    "data/companies", 
    "data/facts", 
    # "data/movies", 
    # "data/olympics"
]

REMOVE_PERIOD = True

# ----- Classifier Training -----
REPEAT_EACH = 10
CHECK_UNCOMMON = False
CHECK_GENERATED = False
KEEP_PROBABILITIES = CHECK_UNCOMMON or CHECK_GENERATED
