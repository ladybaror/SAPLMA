import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__name__), "..")))

from huggingface_hub import snapshot_download

from private_keys.keys import Keys

model_name = "Llama-2-7b-chat-hf"
model_path = f"meta-llama/{model_name}"


snapshot_download(
    repo_id=model_path,
    local_dir=f"./{model_name}",
    local_dir_use_symlinks=False,  # ensures actual files are saved, not symlinks
    token=Keys().API_KEYS["hugging_token"]
)
