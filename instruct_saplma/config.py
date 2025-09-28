# config.py

from pathlib import Path

# ----- Paths -----
# MODEL = "Llama-2-7B-Chat-fp16"
MODEL = "Llama-2-7b-chat-hf"

# MODEL = "Llama-3.2-1B-Instruct"
BASE_MODEL_PATH = f"models/{MODEL}"
FORMAT = 4
DATASET_FOLDER = f"data/try/{MODEL}/capital_true_false_instruct/format{FORMAT}"
CSV_SUFFIX = "_true_false.csv"

# If you prefer a custom CSV naming scheme, uncomment and adapt:
# EMBEDDINGS_FILE_TEMPLATE = "embeddings_with_labels_{name}{MODEL}_{layer}_rmv_period.csv"

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
EPOCHS = 10
BATCH_SIZE = 32
DEV_SPLIT = 0.10               # taken from the training pool (for early stopping & threshold)
USE_EARLY_STOPPING = True
EARLY_STOP_PATIENCE = 10

# Threshold selection on the test set (used only in LOO mode)
TEST_VAL_SPLIT = 0.70          # portion that becomes X_test; remaining is X_val to pick threshold

# Single-dataset safety
ALLOW_SINGLE_DATASET_FALLBACK = True  # if only one CSV is present, do a stratified row-level split

# ----- Train-on-all mode (NEW) -----
# If True, we train a single model on the union of all datasets per layer.
TRAIN_ON_ALL = False
# Optionally carve out a final holdout set (never used for threshold selection) for an unbiased metric:
EVAL_ON_HOLDOUT = False
FINAL_HOLDOUT_FRACTION = 0.20   # e.g., 20% of all rows as final holdout

# ----- Saving / Outputs -----
OUTPUT_DIR = Path("pretrained_saplma") / "instruct" / MODEL /f"format_{FORMAT}" / f"saplma_checkpoints_{MODEL_NAME}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAVE_GLOBAL_BEST = True  # used only in LOO mode
SAVE_PER_DATASET_BEST = False

SUMMARY_TABLE_PATH = Path("pretrained_saplma") / "instruct" / MODEL / f"format_{FORMAT}" / f"summary_table_{MODEL_NAME}_acc_thr.csv"

# Keras save format (single-file .keras)
KERAS_MODEL_FILENAME = "model.keras"
THRESHOLD_FILENAME = "threshold.txt"
META_FILENAME = "meta.json"




