import os

# Replace with your actual intended download directory
local_dir = "/data/not_backed_up/yxu209/models/qwen"

# Make sure the parent folders exist
os.makedirs(local_dir, exist_ok=True)

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen3.8-27B",
    local_dir=local_dir,
)
