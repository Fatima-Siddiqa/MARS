"""STEP 2. Find likely-wrong LLM labels with cross-fitting (no model ever predicts a headline it trained on).

For each fold: train FinBERT+LoRA on the other folds' LLM labels, predict this fold. A headline is
FLAGGED when the out-of-fold model disagrees with the LLM label AND is confident (p >= --conf).
Gold (human test) headlines and their duplicates are excluded, so nothing here touches the test set.

Run from the backend folder:
    python -m app.agents.sentiment.newsagent.training.scripts.crossfit_filter ^
        --labels app\\agents\\sentiment\\newsagent\\training\\data\\labelling_prelabelled.csv ^
        --gold app\\agents\\sentiment\\newsagent\\training\\data\\gold.csv
Outputs in <out>/: oof.csv (every headline with out-of-fold probabilities), flagged.csv (to inspect),
                   filtered_labels.csv (LLM labels minus flagged; feed to finetune_finbert_labelled.py)
Test the plumbing without a GPU/model:  add --dry-run  (random numbers, NOT valid results).
"""
import argparse
import csv
import random
from pathlib import Path

import numpy as np

from app.agents.sentiment.newsagent.training.scripts.finetune_finbert_labelled import (
    CLASSES,
    norm,
    read_labels,
)
from app.agents.sentiment.newsagent.training.scripts.ft_lib import train_and_predict

TRAINING_DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--out", default=str(TRAINING_DATA_DIR / "cf"))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--conf", type=float, default=0.80, help="confidence needed to flag a disagreement")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    gold = read_labels(a.gold)
    gold_ids = {r["article_id"] for r in gold}
    gold_keys = {norm(r["headline"]) for r in gold}
    pool = [r for r in read_labels(a.labels)
            if r["article_id"] not in gold_ids and norm(r["headline"]) not in gold_keys]
    print(f"training pool: {len(pool)} headlines (gold excluded: {len(gold)})")

    groups = {}
    for r in pool:
        groups.setdefault(norm(r["headline"]), []).append(r)
    keys = sorted(groups)
    random.Random(a.seed).shuffle(keys)
    fold_of = {}
    for i, k in enumerate(keys):
        for r in groups[k]:
            fold_of[r["article_id"]] = i % a.folds

    probs = {}
    for f in range(a.folds):
        tr = [r for r in pool if fold_of[r["article_id"]] != f]
        te = [r for r in pool if fold_of[r["article_id"]] == f]
        print(f"fold {f + 1}/{a.folds}: train {len(tr)}, predict {len(te)}")
        p = train_and_predict(tr, [r["headline"] for r in te], seed=a.seed + f,
                              epochs=a.epochs, dry_run=a.dry_run)
        for r, row in zip(te, p):
            probs[r["article_id"]] = row

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, flagged = [], []
    for r in pool:
        p = probs[r["article_id"]]
        pred = CLASSES[int(np.argmax(p))]
        ent = float(-(p * np.log(np.clip(p, 1e-12, 1))).sum())
        row = {"article_id": r["article_id"], "headline": r["headline"], "kind": r["kind"],
               "date": r["date"], "llm_label": r["label"], "oof_pred": pred,
               "p_positive": round(float(p[0]), 4), "p_negative": round(float(p[1]), 4),
               "p_neutral": round(float(p[2]), 4), "p_max": round(float(p.max()), 4),
               "entropy": round(ent, 4)}
        row["flagged"] = int(pred != r["label"] and p.max() >= a.conf)
        rows.append(row)
        if row["flagged"]:
            flagged.append(row)
    cols = list(rows[0].keys())
    for name, data in (("oof.csv", rows), ("flagged.csv", flagged)):
        with open(out / name, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader(); w.writerows(data)
    keep = [r for r in rows if not r["flagged"]]
    with open(out / "filtered_labels.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["article_id", "kind", "date", "headline", "label"])
        for r in keep:
            w.writerow([r["article_id"], r["kind"], r["date"], r["headline"], r["llm_label"]])

    agree = np.mean([r["oof_pred"] == r["llm_label"] for r in rows])
    print(f"\nout-of-fold agreement with LLM labels: {agree:.1%}")
    print(f"flagged (confident disagreement, p>={a.conf}): {len(flagged)} of {len(rows)}")
    for lab in CLASSES:
        sub = [r for r in flagged if r["llm_label"] == lab]
        if sub:
            print(f"  LLM said {lab}: {len(sub)}  ->  " + ", ".join(
                f"{c} {sum(1 for r in sub if r['oof_pred'] == c)}" for c in CLASSES if c != lab))
    print(f"\nwrote {out}/oof.csv, flagged.csv, filtered_labels.csv ({len(keep)} rows kept)")
    if a.dry_run:
        print("DRY RUN: probabilities are random; do not use these files.")


if __name__ == "__main__":
    main()
