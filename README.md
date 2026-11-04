# AIST4010 Fall 2026 A2 — ARG Classification (Kaggle)

**Task**: binary protein-sequence classification — antibiotic resistance
gene (ARG) vs non-ARG, from amino-acid FASTA.
Metric: accuracy.

- train: 21,209 seqs (10,067 ARG / 11,142 non-ARG)
- val: 4,387 seqs — local validation
- test: 4,469 seqs — Kaggle submission (`id,label`)

## Pipeline

1. `eda.py` — dataset statistics + figures (`eda.png`)
2. `baseline_kmer.py` — 3-mer TF-IDF + logistic regression (val 95.9%)
3. `esm_embed.py` — ESM-2 650M mean-pool embeddings + linear probe
   (val 98.61%, AUC 0.9984)
4. `esm_finetune.py` — LoRA fine-tune of ESM-2 650M + linear head
   (val **99.13%**, AUC 0.9964) → `submission.csv`

## Results

| Method | val acc | val AUC |
|---|---|---|
| 3-mer + LR | 95.90% | 0.9900 |
| ESM-2 embed + LR probe | 98.61% | 0.9984 |
| ESM-2 LoRA fine-tune | **99.13%** | 0.9964 |

## Reproduce

```bash
# download model (hf-mirror)
python3 download_esm.py

# embeddings + probe
python3 esm_embed.py embed   # GPU, ~10 min
python3 esm_embed.py probe

# fine-tune + predict
python3 esm_finetune.py train
python3 esm_finetune.py predict   # -> submission.csv
```

Note: model weights under `models/` are not committed (large files).
