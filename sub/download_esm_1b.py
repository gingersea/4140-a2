import os
# 直连 huggingface.co(代理环境已在 shell 中)
os.environ.pop("HF_ENDPOINT", None)
from huggingface_hub import snapshot_download
p = snapshot_download("facebook/esm2_t36_3B_UR50D", local_dir="models/esm2_t36_3B_UR50D")
print("downloaded to", p)
