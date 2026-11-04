"""Extract ESM-2 (650M) protein embeddings for train/val/test, then train a
linear probe for ARG classification.

Usage:
  python3 esm_embed.py embed   # extract embeddings -> data/emb/*.npy (GPU)
  python3 esm_embed.py probe   # train probe on train, evaluate val, predict test
"""
import os
import sys

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
import torch
from transformers import EsmModel, EsmTokenizer

MODEL_DIR = "models/esm2_t33_650M_UR50D"
EMB_DIR = "data/emb"
MAXLEN = 1022  # ESM2 650M max positions - 2


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
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    ids_all, seqs_all = [], []
    for p in paths:
        i, s = read_fasta(p)
        ids_all += i; seqs_all += s

    bs = 16
    embeds = np.zeros((len(seqs_all), model.module.config.hidden_size
                       if isinstance(model, torch.nn.DataParallel)
                       else model.config.hidden_size), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(seqs_all), bs):
            batch = seqs_all[start:start + bs]
            enc = tok(batch, return_tensors="pt", padding=True,
                      truncation=True, max_length=MAXLEN)
            enc = {k: v.cuda() for k, v in enc.items()}
            out = model(**enc).last_hidden_state  # (B, L, D)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            mean = (out * mask).sum(1) / mask.sum(1)
            embeds[start:start + bs] = mean.cpu().numpy()
            if (start // bs) % 50 == 0:
                print(f"  {tag}: {start}/{len(seqs_all)}", flush=True)

    np.save(os.path.join(EMB_DIR, f"emb_{tag}.npy"), embeds)
    with open(os.path.join(EMB_DIR, f"ids_{tag}.txt"), "w") as f:
        f.write("\n".join(i[1:] for i in ids_all))
    print(f"saved {tag}: {embeds.shape}")


def probe():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score
    import pandas as pd

    X_tr = np.load(f"{EMB_DIR}/emb_train.npy")
    X_va = np.load(f"{EMB_DIR}/emb_val.npy")
    X_te = np.load(f"{EMB_DIR}/emb_test.npy")
    tr_ids = open(f"{EMB_DIR}/ids_train.txt").read().split("\n")
    va_ids = open(f"{EMB_DIR}/ids_val.txt").read().split("\n")
    te_ids = open(f"{EMB_DIR}/ids_test.txt").read().split("\n")

    y_tr = np.array([1 if "|FEATURES|" in i else 0 for i in tr_ids])
    y_va = np.array([1 if "|FEATURES|" in i else 0 for i in va_ids])

    # normalize embeddings
    mu, sd = X_tr.mean(0), X_tr.std(0) + 1e-6
    X_tr = (X_tr - mu) / sd
    X_va = (X_va - mu) / sd
    X_te = (X_te - mu) / sd

    for C in [1e-3, 1e-2, 0.1, 1.0]:
        clf = LogisticRegression(C=C, max_iter=2000, solver="liblinear")
        clf.fit(X_tr, y_tr)
        p = clf.predict(X_va)
        acc = accuracy_score(y_va, p)
        auc = roc_auc_score(y_va, clf.decision_function(X_va))
        print(f"  C={C}: val acc={acc:.4f} auc={auc:.4f}")

    clf = LogisticRegression(C=0.1, max_iter=2000, solver="liblinear")
    clf.fit(X_tr, y_tr)
    p = clf.predict(X_va)
    print(f"final: val acc={accuracy_score(y_va, p):.4f} "
          f"auc={roc_auc_score(y_va, clf.decision_function(X_va)):.4f}")
    p_te = clf.predict(X_te)
    pd.DataFrame({"id": te_ids, "label": p_te}).to_csv(
        "submission_esm.csv", index=False)
    print(f"saved submission_esm.csv, pos rate={p_te.mean():.4f}")


if __name__ == "__main__":
    os.makedirs(EMB_DIR, exist_ok=True)
    cmd = sys.argv[1]
    if cmd == "embed":
        extract(["data/train.fasta"], "train")
        extract(["data/val.fasta"], "val")
        extract(["data/test.fasta"], "test")
    elif cmd == "probe":
        probe()
