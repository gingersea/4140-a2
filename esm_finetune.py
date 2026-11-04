"""Fine-tune ESM-2 650M for ARG 15-class classification (LoRA + head).

Task: 14 antibiotic families (0-13) + nonarg (14).
Usage:
  python3 esm_finetune.py train   # LoRA fine-tune, early stop on val macro-F1
  python3 esm_finetune.py predict # predict test + val with best checkpoint
"""
import os
import sys

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import EsmModel, EsmTokenizer
from peft import LoraConfig, get_peft_model
from sklearn.metrics import accuracy_score, f1_score

MODEL_DIR = "models/esm2_t33_650M_UR50D"
CKPT = "checkpoints/esm_lora_15cls_best.pt"
MAXLEN = 256
BATCH = 4
ACCUM = 8
EPOCHS = 8
LR = 1e-4
SEED = 42

LABEL_MAP = {
    "aminoglycoside": 0,
    "macrolide-lincosamide-streptogramin": 1,
    "polymyxin": 2,
    "fosfomycin": 3,
    "trimethoprim": 4,
    "bacitracin": 5,
    "quinolone": 6,
    "multidrug": 7,
    "chloramphenicol": 8,
    "tetracycline": 9,
    "rifampin": 10,
    "beta_lactam": 11,
    "sulfonamide": 12,
    "glycopeptide": 13,
}
NONARG = 14
NUM_LABELS = 15


def read_fasta(path):
    seqs, ids, labels = [], [], []
    cur_h, cur_s = None, []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if cur_h is not None:
                    seqs.append("".join(cur_s))
                    ids.append(cur_h)
                    labels.append(LABEL_MAP[cur_h.split("|")[3]]
                                 if "|FEATURES|" in cur_h else NONARG)
                cur_h = line; cur_s = []
            elif line.strip():
                cur_s.append(line.strip())
    if cur_h is not None:
        seqs.append("".join(cur_s))
        ids.append(cur_h)
        labels.append(LABEL_MAP[cur_h.split("|")[3]]
                     if "|FEATURES|" in cur_h else NONARG)
    return ids, seqs, labels


class SeqDS(Dataset):
    def __init__(self, seqs, labels, tok):
        self.enc = tok(seqs, return_tensors="pt", padding=True,
                       truncation=True, max_length=MAXLEN)
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return (self.enc["input_ids"][i], self.enc["attention_mask"][i],
                self.labels[i])


def build_model():
    model = EsmModel.from_pretrained(MODEL_DIR, torch_dtype=torch.float16)
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        target_modules=["query", "key", "value"]))
    head = nn.Linear(model.config.hidden_size, NUM_LABELS)
    head = head.cuda().to(torch.float32)
    return model, head


def train():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    tok = EsmTokenizer.from_pretrained(MODEL_DIR)

    _, tr_s, tr_y = read_fasta("data/train.fasta")
    _, va_s, va_y = read_fasta("data/val.fasta")

    ds_tr = SeqDS(tr_s, tr_y, tok)
    ds_va = SeqDS(va_s, va_y, tok)
    dl_tr = DataLoader(ds_tr, batch_size=BATCH, shuffle=True,
                       num_workers=4, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=BATCH, shuffle=False,
                       num_workers=4, pin_memory=True)

    model, head = build_model()
    model = model.cuda().train()

    # class-balanced weights (test set is uniform over 15 classes)
    counts = np.bincount(tr_y, minlength=NUM_LABELS).astype(np.float64)
    w = counts.sum() / (NUM_LABELS * counts)
    w = torch.tensor(w, dtype=torch.float32).cuda()
    lossf = nn.CrossEntropyLoss(weight=w)

    opt = torch.optim.AdamW(
        list(model.parameters()) + list(head.parameters()),
        lr=LR, weight_decay=0.01)
    steps_per_epoch = len(dl_tr)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR, epochs=EPOCHS, steps_per_epoch=steps_per_epoch,
        pct_start=0.1)

    best_f1 = 0.0
    for ep in range(EPOCHS):
        tot = 0.0; n = 0
        for i, (ii, am, yy) in enumerate(dl_tr):
            ii, am, yy = ii.cuda(), am.cuda(), yy.cuda()
            out = model(ii, attention_mask=am).last_hidden_state
            mask = am.unsqueeze(-1).float()
            mean = (out * mask).sum(1) / mask.sum(1)
            logits = head(mean.float())
            loss = lossf(logits, yy) / ACCUM
            loss.backward()
            if (i + 1) % ACCUM == 0:
                opt.step(); sched.step(); opt.zero_grad()
            tot += loss.item() * len(yy) * ACCUM; n += len(yy)
            if i % 500 == 0:
                print(f"  ep{ep} step{i}/{steps_per_epoch} "
                      f"loss={loss.item():.4f}", flush=True)
        tr_loss = tot / n

        model.eval()
        preds, truths = [], []
        with torch.no_grad():
            for ii, am, yy in dl_va:
                ii, am, yy = ii.cuda(), am.cuda(), yy.cuda()
                out = model(ii, attention_mask=am).last_hidden_state
                mask = am.unsqueeze(-1).float()
                mean = (out * mask).sum(1) / mask.sum(1)
                preds.append(head(mean.float()).argmax(1).cpu())
                truths.append(yy.cpu())
        preds = torch.cat(preds).numpy()
        truths = torch.cat(truths).numpy()
        acc = accuracy_score(truths, preds)
        f1 = f1_score(truths, preds, average="macro")
        print(f"epoch {ep}: train_loss={tr_loss:.4f} val_acc={acc:.4f} "
              f"val_macroF1={f1:.4f}", flush=True)
        model.train()
        if f1 > best_f1:
            best_f1 = f1
            os.makedirs("checkpoints", exist_ok=True)
            torch.save({"model": model.state_dict(), "head": head.state_dict()},
                       CKPT)
            print("  saved best", flush=True)
    print("best val macro-F1:", best_f1)


def predict():
    tok = EsmTokenizer.from_pretrained(MODEL_DIR)
    model, head = build_model()
    ck = torch.load(CKPT, map_location="cpu")
    model.load_state_dict(ck["model"]); head.load_state_dict(ck["head"])
    model = model.cuda().eval(); head.eval()

    import pandas as pd

    def run(seqs):
        probs = []
        with torch.no_grad():
            for s in range(0, len(seqs), BATCH):
                enc = tok(seqs[s:s + BATCH], return_tensors="pt", padding=True,
                          truncation=True, max_length=MAXLEN)
                enc = {k: v.cuda() for k, v in enc.items()}
                out = model(**enc).last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1).float()
                mean = (out * mask).sum(1) / mask.sum(1)
                probs.append(torch.softmax(head(mean.float()), 1).cpu().numpy())
        return np.concatenate(probs)

    va_h, va_s, va_y = read_fasta("data/val.fasta")
    pv = run(va_s)
    pred = pv.argmax(1)
    print(f"val acc={accuracy_score(va_y, pred):.4f} "
          f"macro-F1={f1_score(va_y, pred, average='macro'):.4f}")

    te_h, te_s, _ = read_fasta("data/test.fasta")
    pt = run(te_s)
    pd.DataFrame({"id": [h[1:] for h in te_h],
                  "label": pt.argmax(1)}).to_csv(
        "submission_esm_ft.csv", index=False)
    print("saved submission_esm_ft.csv",
          "dist:", dict(zip(*np.unique(pt.argmax(1), return_counts=True))))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "train":
        train()
    elif cmd == "predict":
        predict()
