# AIST4010 A2 — ARG Classification (15 classes)

Final result: **balanced-proxy val 98.10%** (macro-F1 0.9753),
two-model ensemble of class-balanced LR probes on ESM-2 650M + 3B embeddings.

## Files

```
├── A2_report.pdf        # experiment report (Work Log, submit this)
├── A2_report.tex        # report source
├── submission.csv       # final Kaggle submission (4,469 rows, id,label)
├── esm_embed.py         # ESM-2 650M embedding extraction + probe
├── esm3b_embed.py       # ESM-2 3B embedding extraction
├── esm_finetune.py      # LoRA fine-tuning (tried, not in final)
├── ensemble.py          # probe ensemble (tried, not in final)
├── baseline_kmer.py     # 3-mer TF-IDF + LR baseline
├── eda.py               # data exploration
├── download_esm.py      # 650M download (hf-mirror)
└── download_esm_1b.py   # 3B download
```

## Task

15-class protein classification: 14 antibiotic families (0-13) + non-ARG (14).
Test set is class-balanced (~298/class); train/val are skewed (52% non-ARG).

## Final pipeline

1. Extract ESM-2 650M and 3B mean-pool embeddings (frozen).
2. Train logistic regression probes with inverse-frequency class weights
   (matching the balanced test prior).
3. Ensemble the two probes 1:1 (error decorrelation across model scales).

## Results (balanced proxy of val: 200 samples/class)

| Method | proxy acc | proxy macro-F1 |
|---|---|---|
| 3-mer TF-IDF + LR | 94.73% | 0.927 |
| ESM-2 650M embed + LR | 97.82% | 0.975 |
| ESM-2 3B embed + LR | 98.17% | 0.974 |
| **650M + 3B ensemble (1:1)** | **98.10%** | **0.9753** |

## Reproduce

```bash
python3 download_esm.py       # 650M weights
python3 download_esm_1b.py    # 3B weights (curl fallback if rate-limited)
python3 esm_embed.py embed    # -> data/emb/emb_*.npy
python3 esm3b_embed.py        # -> data/emb/emb3b_*.npy
# probe training + ensemble + submission.csv: see report section 2
```
