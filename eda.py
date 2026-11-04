"""EDA for AIST4010 Fall 2026 A2 — Antibiotic Resistance Gene (ARG) classification.

Task: binary classification of protein sequences (ARG vs non-ARG).
"""
import re
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_fasta(path):
    seqs, headers = [], []
    cur_h, cur_s = None, []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if cur_h is not None:
                    seqs.append("".join(cur_s))
                    headers.append(cur_h)
                cur_h = line
                cur_s = []
            elif line.strip():
                cur_s.append(line.strip())
    if cur_h is not None:
        seqs.append("".join(cur_s))
        headers.append(cur_h)
    return headers, seqs


def parse_feat_header(h):
    """'>id|FEATURES|src|drug|gene|mechanism|label'"""
    p = h[1:].split("|")
    return {"id": p[0], "src": p[2], "drug": p[3], "gene": p[4],
            "mech": p[5], "label": int(p[6])}


def main():
    for fn in ["data/train.fasta", "data/val.fasta", "data/test.fasta"]:
        headers, seqs = read_fasta(fn)
        print(f"== {fn}: {len(seqs)} sequences")
        lens = np.array([len(s) for s in seqs])
        print(f"   length: min={lens.min()} max={lens.max()} "
              f"mean={lens.mean():.1f} p50={np.median(lens):.1f}")
        print(f"   length >1022 (ESM-2 650M max): {(lens > 1022).sum()}")

    headers, seqs = read_fasta("data/train.fasta")
    feats = [parse_feat_header(h) for h in headers if "|FEATURES|" in h]
    print(f"\ntrain: ARG={len(feats)}, non-ARG={len(seqs) - len(feats)}")

    # per-drug class counts
    by_drug = Counter(f["drug"] for f in feats)
    print("\ndrug classes:")
    for k, v in sorted(by_drug.items(), key=lambda x: -x[1]):
        print(f"   {k}: {v}")

    by_mech = Counter(f["mech"] for f in feats)
    print("\nmechanisms:", dict(by_mech))

    # length histograms: ARG vs non-ARG
    arg_lens = []
    nonarg_lens = []
    cur_h = None
    cur_s = []
    is_arg = {}
    for h in headers:
        if "|FEATURES|" in h:
            is_arg[h[1:].split("|")[0]] = True
    h_idx = 0
    for i, h in enumerate(headers):
        pass
    # simpler: iterate records again
    arg_ids = set(f["id"] for f in feats)
    for h, s in zip(headers, seqs):
        (arg_lens if h[1:].split("|")[0] in arg_ids else nonarg_lens).append(len(s))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist(arg_lens, bins=50, alpha=0.6, label="ARG")
    axes[0].hist(nonarg_lens, bins=50, alpha=0.6, label="non-ARG")
    axes[0].set_title("sequence length: ARG vs non-ARG (train)")
    axes[0].set_xlabel("length (aa)")
    axes[0].legend()

    axes[1].hist(np.log10(arg_lens), bins=50, alpha=0.6, label="ARG")
    axes[1].hist(np.log10(nonarg_lens), bins=50, alpha=0.6, label="non-ARG")
    axes[1].set_title("log10 length")
    axes[1].set_xlabel("log10 length")
    axes[1].legend()

    # amino acid composition difference
    aa = "ACDEFGHIKLMNPQRSTVWY"
    comp_arg = np.zeros(20)
    comp_non = np.zeros(20)
    n_arg = n_non = 0
    for h, s in zip(headers, seqs):
        c = Counter(s)
        tot = len(s)
        v = np.array([c[a] / tot for a in aa])
        if h[1:].split("|")[0] in arg_ids:
            comp_arg += v; n_arg += 1
        else:
            comp_non += v; n_non += 1
    comp_arg /= n_arg; comp_non /= n_non
    x = np.arange(20)
    axes[2].bar(x - 0.2, comp_arg, 0.4, label="ARG")
    axes[2].bar(x + 0.2, comp_non, 0.4, label="non-ARG")
    axes[2].set_xticks(x); axes[2].set_xticklabels(list(aa))
    axes[2].set_title("avg amino-acid composition")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig("eda.png", dpi=120)
    print("\nsaved eda.png")


if __name__ == "__main__":
    main()
