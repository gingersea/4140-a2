"""Baseline: 3-mer + TF-IDF + linear model (LR) for ARG classification.

Fast, reproducible baseline. Produces submission_baseline.csv.
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score


def read_fasta(path):
    seqs, ids, labels = [], [], []
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


def main():
    np.random.seed(42)

    tr_h, tr_s = read_fasta("data/train.fasta")
    va_h, va_s = read_fasta("data/val.fasta")
    te_h, te_s = read_fasta("data/test.fasta")

    # labels: FEATURES headers carry explicit 0/1; non-FEATURES = 0
    y_tr = np.array([1 if "|FEATURES|" in h else 0 for h in tr_h])
    y_va = np.array([1 if "|FEATURES|" in h else 0 for h in va_h])

    # 3-mer features on spaced representation
    vec = CountVectorizer(analyzer="char", ngram_range=(3, 3), max_features=200000,
                          lowercase=False)
    X_tr = vec.fit_transform(tr_s)
    X_va = vec.transform(va_s)
    X_te = vec.transform(te_s)
    print("vocab size:", len(vec.vocabulary_))

    clf = LogisticRegression(C=1.0, max_iter=1000, n_jobs=-1, solver="liblinear")
    clf.fit(X_tr, y_tr)

    p_va = clf.predict(X_va)
    acc = accuracy_score(y_va, p_va)
    auc = roc_auc_score(y_va, clf.decision_function(X_va))
    print(f"val accuracy: {acc:.4f}  AUC: {auc:.4f}")

    p_te = clf.predict(X_te)
    sub = pd.DataFrame({"id": [h[1:] for h in te_h], "label": p_te})
    sub.to_csv("submission_baseline.csv", index=False)
    print("saved submission_baseline.csv, pos rate:",
          f"{p_te.mean():.4f}")


if __name__ == "__main__":
    main()
