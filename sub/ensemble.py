"""Three-model ensemble for ARG 15-class classification.

Members:
  1. ESM-2 650M mean-pool embeddings + logistic regression probe (LR)
  2. ESM-2 650M LoRA fine-tune (FT)
  3. 3-mer TF-IDF + logistic regression (kmer)

Ensemble: weighted average of predicted probabilities (default 2:2:1).
"""
import importlib.util
import os

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

M = {'aminoglycoside': 0, 'macrolide-lincosamide-streptogramin': 1,
     'polymyxin': 2, 'fosfomycin': 3, 'trimethoprim': 4,
     'bacitracin': 5, 'quinolone': 6, 'multidrug': 7,
     'chloramphenicol': 8, 'tetracycline': 9, 'rifampin': 10,
     'beta_lactam': 11, 'sulfonamide': 12, 'glycopeptide': 13}
NONARG = 14


def read_fasta(path):
    seqs, hs = [], []
    cur_h, cur_s = None, []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if cur_h is not None:
                    seqs.append("".join(cur_s)); hs.append(cur_h)
                cur_h = line; cur_s = []
            elif line.strip():
                cur_s.append(line.strip())
    if cur_h is not None:
        seqs.append("".join(cur_s)); hs.append(cur_h)
    return hs, seqs


def labels_from_ids(ids):
    return np.array([M[h.split("|")[3]] if "|FEATURES|" in h else NONARG
                     for h in ids])


class KmerModel:
    def __init__(self):
        self.vec = CountVectorizer(analyzer="char", ngram_range=(3, 3),
                                   max_features=200000, lowercase=False)
        self.clf = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")

    def fit(self, seqs, y):
        X = self.vec.fit_transform(seqs)
        self.clf.fit(X, y)

    def predict_proba(self, seqs):
        return self.clf.predict_proba(self.vec.transform(seqs))


class EmbedProbeModel:
    """ESM-2 embeddings (pre-extracted) + LR."""
    def __init__(self):
        self.clf = LogisticRegression(C=0.1, max_iter=3000, solver="lbfgs")
        self.mu = self.sd = None

    def fit(self, X, y):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        self.clf.fit((X - self.mu) / self.sd, y)

    def predict_proba(self, X):
        return self.clf.predict_proba((X - self.mu) / self.sd)


class FinetuneModel:
    def __init__(self):
        import torch
        import torch.nn as nn
        from transformers import EsmTokenizer
        spec = importlib.util.spec_from_file_location(
            "esmf", "esm_finetune.py")
        esmf = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(esmf)
        self.tok = EsmTokenizer.from_pretrained(esmf.MODEL_DIR)
        model, head = esmf.build_model()
        ck = torch.load(esmf.CKPT, map_location="cpu")
        model.load_state_dict(ck["model"]); head.load_state_dict(ck["head"])
        self.model = model.cuda().eval(); self.head = head.eval()
        self.batch = esmf.BATCH
        self.maxlen = esmf.MAXLEN

    def predict_proba(self, seqs):
        import torch
        probs = []
        with torch.no_grad():
            for s in range(0, len(seqs), self.batch):
                enc = self.tok(seqs[s:s + self.batch], return_tensors="pt",
                               padding=True, truncation=True,
                               max_length=self.maxlen)
                enc = {k: v.cuda() for k, v in enc.items()}
                out = self.model(**enc).last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1).float()
                mean = (out * mask).sum(1) / mask.sum(1)
                probs.append(torch.softmax(self.head(mean.float()), 1)
                             .cpu().numpy())
        return np.concatenate(probs)


def main(weights=(2, 2, 1)):
    tr_h, tr_s = read_fasta("data/train.fasta")
    va_h, va_s = read_fasta("data/val.fasta")
    te_h, te_s = read_fasta("data/test.fasta")
    y_tr = labels_from_ids(tr_h)
    y_va = labels_from_ids(va_h)

    # 1) kmer
    print("training kmer model...")
    km = KmerModel()
    km.fit(tr_s, y_tr)

    # 2) embed probe
    print("training embed probe...")
    X_tr = np.load("data/emb/emb_train.npy")
    X_va = np.load("data/emb/emb_val.npy")
    X_te = np.load("data/emb/emb_test.npy")
    ep = EmbedProbeModel()
    ep.fit(X_tr, y_tr)

    # 3) finetune
    print("loading finetune model...")
    ft = FinetuneModel()

    w_ep, w_ft, w_km = weights
    p_ep = ep.predict_proba(X_va)
    p_ft = ft.predict_proba(va_s)
    p_km = km.predict_proba(va_s)
    ens = w_ep * p_ep + w_ft * p_ft + w_km * p_km
    print(f"val ensemble acc={accuracy_score(y_va, ens.argmax(1)):.4f} "
          f"macroF1={f1_score(y_va, ens.argmax(1), average='macro'):.4f}")

    # test
    p_ep_t = ep.predict_proba(X_te)
    p_ft_t = ft.predict_proba(te_s)
    p_km_t = km.predict_proba(te_s)
    ens_t = w_ep * p_ep_t + w_ft * p_ft_t + w_km * p_km_t
    pred = ens_t.argmax(1)
    pd.DataFrame({"id": [h[1:] for h in te_h], "label": pred}).to_csv(
        "submission_ensemble.csv", index=False)
    print("saved submission_ensemble.csv")
    print("test dist:", dict(zip(*np.unique(pred, return_counts=True))))


if __name__ == "__main__":
    main()
