# from huggingface_hub import hf_hub_download

# REPO_ID = "ShapeNet/ShapeNetCore"
# FILENAME = "03001627.zip"

# print(hf_hub_download(repo_id=REPO_ID, repo_type="dataset"))

from datasets import load_dataset

# Load the ShapeNet/ShapeNetCore dataset
dataset = load_dataset("ShapeNet/ShapeNetCore")

print(dataset.cache_files)
