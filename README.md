# AIST4010 Fall 2026 A2 — ARG Classification (Kaggle)

**Task**: 15-class protein-sequence classification from amino-acid FASTA.
Predict whether a protein is an antibiotic resistance gene (ARG) and, if so,
which of 14 antibiotic families it confers resistance to.

| label | meaning |
|---|---|
| 0–13 | 14 antibiotic families (aminoglycoside, MLS, polymyxin, fosfomycin, trimethoprim, bacitracin, quinolone, multidrug, chloramphenicol, tetracycline, rifampin, beta_lactam, sulfonamide, glycopeptide) |
| 14 | non-ARG |

Metric: accuracy. Test set is class-balanced (~300/class).

- train: 21,209 seqs (10,067 ARG with drug annotation / 11,142 non-ARG)
- val: 4,387 seqs — local validation
- test: 4,469 seqs — Kaggle submission (`id,label`)

## Label mapping (kept in sync with the Kaggle page)

```python
{"aminoglycoside": 0, "macrolide-lincosamide-streptogramin": 1,
 "polymyxin": 2, "fosfomycin": 3, "trimethoprim": 4,
 "bacitracin": 5, "quinolone": 6, "multidrug": 7,
 "chloramphenicol": 8, "tetracycline": 9, "rifampin": 10,
 "beta_lactam": 11, "sulfonamide": 12, "glycopeptide": 13}
# non-ARG -> 14
```

## Pipeline

1. `eda.py` — dataset statistics + figures (`eda.png`)
2. `baseline_kmer.py` — 3-mer TF-IDF + logistic regression (binary only, kept for reference)
3. `esm_embed.py` — ESM-2 650M mean-pool embeddings + linear probe
4. `esm_finetune.py` — LoRA fine-tune of ESM-2 650M + linear head,
   class-balanced CE loss, early stop on val macro-F1 → `submission.csv`

## Results (15-class)

| Method | val acc | val macro-F1 |
|---|---|---|
| ESM-2 embed + LR probe (15-cls) | 98.84% | 0.9693 |
| ESM-2 LoRA fine-tune | 98.43% | **0.9538** |

Per-class F1 of the fine-tuned model (val): rifampin 0.82 (8 samples) is the
weakest class; beta_lactam 0.996, bacitracin 0.997, nonarg 0.989.

## Reproduce

```bash
python3 download_esm.py          # hf-mirror

python3 esm_embed.py embed       # GPU, ~10 min
python3 esm_embed.py probe

python3 esm_finetune.py train    # LoRA fine-tune (8 epochs)
python3 esm_finetune.py predict  # -> submission.csv
```

Note: model weights under `models/` are not committed (large files).
