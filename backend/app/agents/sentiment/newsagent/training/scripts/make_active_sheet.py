"""STEP 3. Build a BLIND sheet of the headlines most worth a human label (active labelling).

Picks, from the out-of-fold results of crossfit_filter.py:
  * half: headlines where the model disagrees with the LLM label (largest confidence first)
  * half: the remaining headlines with the highest uncertainty (entropy)
Gold headlines are excluded. The sheet shows ONLY ticker and headline, with an empty label column,
so neither the LLM label nor the model's guess anchors you. Label with the same guide as before.

    python -m app.agents.sentiment.newsagent.training.scripts.make_active_sheet ^
        --oof app\\agents\\sentiment\\newsagent\\training\\data\\cf\\oof.csv ^
        --labels app\\agents\\sentiment\\newsagent\\training\\data\\labelling_prelabelled.csv ^
        --gold app\\agents\\sentiment\\newsagent\\training\\data\\gold.csv --n 120
Writes training/data/active_sheet.csv (give to the annotator) and
training/data/active_key.csv (metadata; do not open until done).
"""
import argparse
import csv
from pathlib import Path

from app.agents.sentiment.newsagent.training.scripts.finetune_finbert_labelled import (
    norm,
    read_labels,
)

TRAINING_DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", required=True)
    ap.add_argument("--labels", required=True, help="the LLM label file (for the ticker column)")
    ap.add_argument("--gold", required=True)
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--out", default=str(TRAINING_DATA_DIR / "active_sheet.csv"))
    ap.add_argument("--key", default=str(TRAINING_DATA_DIR / "active_key.csv"))
    a = ap.parse_args()

    gold = read_labels(a.gold)
    gold_ids = {r["article_id"] for r in gold}
    gold_keys = {norm(r["headline"]) for r in gold}
    ticker = {}
    with open(a.labels, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            ticker[(r.get("article_id") or "").strip()] = r.get("ticker", "")
    with open(a.oof, newline="", encoding="utf-8-sig") as fh:
        rows = [r for r in csv.DictReader(fh)
                if r["article_id"] not in gold_ids and norm(r["headline"]) not in gold_keys]

    half = a.n // 2
    dis = sorted((r for r in rows if r["oof_pred"] != r["llm_label"]), key=lambda r: -float(r["p_max"]))
    picked, seen = [], set()
    for r in dis:
        k = norm(r["headline"])
        if k not in seen and len(picked) < half:
            picked.append(("disagreement", r)); seen.add(k)
    for r in sorted(rows, key=lambda r: -float(r["entropy"])):
        k = norm(r["headline"])
        if k not in seen and len(picked) < a.n:
            picked.append(("uncertain", r)); seen.add(k)
    import random
    random.Random(5).shuffle(picked)
    with open(a.out, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["article_id", "ticker", "headline", "label"])
        for _, r in picked:
            w.writerow([r["article_id"], ticker.get(r["article_id"], ""), r["headline"], ""])
    with open(a.key, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["article_id", "why_picked", "llm_label", "oof_pred", "p_max"])
        for why, r in picked:
            w.writerow([r["article_id"], why, r["llm_label"], r["oof_pred"], r["p_max"]])
    print(f"wrote {a.out} ({len(picked)} headlines, blind) and {a.key}")


if __name__ == "__main__":
    main()
