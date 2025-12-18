import os
from huggingface_hub import snapshot_download

def fetch_data():
    HF_DATASET = "kvasirseg"

    # Download dataset từ HF
    hf_path = snapshot_download(
        repo_id=HF_DATASET,
        repo_type="dataset"
    )

    TARGET_DATASET_PATH = "/workspace/coco_kvasirseg"

    os.makedirs(os.path.dirname(TARGET_DATASET_PATH), exist_ok=True)

    if not os.path.exists(TARGET_DATASET_PATH):
        os.symlink(hf_path, TARGET_DATASET_PATH)

    print("✅ Dataset ready at:", TARGET_DATASET_PATH)