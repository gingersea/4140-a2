"""Extract ESM-2 3B embeddings (batch on single GPU, fp16 + gradient checkpointing off)."""
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
import sys
import numpy as np
import torch
from transformers import EsmModel, EsmTokenizer

MODEL_DIR = "models/esm2_t36_3B_UR50D"
MAXLEN = 1022

def read_fasta(path):
    seqs, ids = [], []
    cur_h, cur_s = None, []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if cur_h is not None:
                    seqs.append("".join(cur_s)); ids.append(cur_h)
                cur_h = line; cur_s = []
            elif line.strip():
                cur_s.append(line.strip())
    if cur_h is not None:
        seqs.append("".join(cur_s)); ids.append(cur_h)
    return ids, seqs

def extract(paths, tag):
    tok = EsmTokenizer.from_pretrained(MODEL_DIR)
    model = EsmModel.from_pretrained(MODEL_DIR, torch_dtype=torch.float16)
    model = model.cuda().eval()
    ids_all, seqs_all = [], []
    for p in paths:
        i, s = read_fasta(p)
        ids_all += i; seqs_all += s
    bs = 2
    embeds = np.zeros((len(seqs_all), model.config.hidden_size), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(seqs_all), bs):
            batch = seqs_all[start:start+bs]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=MAXLEN)
            enc = {k: v.cuda() for k, v in enc.items()}
            out = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            mean = (out * mask).sum(1) / mask.sum(1)
            embeds[start:start+bs] = mean.cpu().numpy()
            if (start//bs) % 200 == 0:
                print(f"  {tag}: {start}/{len(seqs_all)}", flush=True)
    np.save(f"data/emb/emb3b_{tag}.npy", embeds)
    with open(f"data/emb/ids3b_{tag}.txt", "w") as f:
        f.write("\n".join(i[1:] for i in ids_all))
    print(f"saved {tag}: {embeds.shape}")

if __name__ == "__main__":
    extract(["data/train.fasta"], "train")
    extract(["data/val.fasta"], "val")
    extract(["data/test.fasta"], "test")
