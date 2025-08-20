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
    "data/animals", 
    "data/companies", 
    "data/facts", 
]

REMOVE_PERIOD = True

# ----- Classifier Training -----
REPEAT_EACH = 10
CHECK_UNCOMMON = False
CHECK_GENERATED = False
KEEP_PROBABILITIES = CHECK_UNCOMMON or CHECK_GENERATED


# Reproducibility
SEED = 42

# Keras training knobs
EPOCHS = 5
BATCH_SIZE = 32
DEV_SPLIT = 0.10               # taken from the training pool (for early stopping & checkpoints)
USE_EARLY_STOPPING = True
EARLY_STOP_PATIENCE = 2

# Threshold selection on the test set (we split test into X_val/X_test)
TEST_VAL_SPLIT = 0.70          # portion that becomes X_test; remaining is X_val to pick threshold
THRESHOLD_METHOD = "val-acc-max"  # (fixed in code) choose threshold that maximizes accuracy on X_val

# Single-dataset safety
ALLOW_SINGLE_DATASET_FALLBACK = True  # if only one CSV is present, do a stratified row-level split

# ----- Saving / Outputs -----
# Where the best classifier bundle (model + threshold + meta) will be saved
OUTPUT_DIR = Path("pretrained_saplma") / "completion" / f"saplma_checkpoints_{MODEL_NAME}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Save a global BEST bundle (highest accuracy@optimal-threshold across all runs)
SAVE_GLOBAL_BEST = True

# If you later want one best per held-out dataset, set this True and adapt the script
SAVE_PER_DATASET_BEST = False

# Summary table path
SUMMARY_TABLE_PATH = Path("pretrained_saplma") / "completion" / f"summary_table_{MODEL_NAME}_acc_thr.csv"

# Keras save format (single-file .keras)
KERAS_MODEL_FILENAME = "model.keras"
THRESHOLD_FILENAME = "threshold.txt"
META_FILENAME = "meta.json"