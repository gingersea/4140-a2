import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
from huggingface_hub import snapshot_download
p = snapshot_download("facebook/esm2_t30_150M_UR50D", local_dir="models/esm2_t30_150M_UR50D")
print("downloaded to", p)
