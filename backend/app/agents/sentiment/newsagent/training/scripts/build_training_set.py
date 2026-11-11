"""STEP 4. Build the final training file: human labels first, then LLM labels, minus flagged errors.

    python -m app.agents.sentiment.newsagent.training.scripts.build_training_set ^
        --llm app\\agents\\sentiment\\newsagent\\training\\data\\labelling_prelabelled.csv ^
        --flagged app\\agents\\sentiment\\newsagent\\training\\data\\cf\\flagged.csv ^
        --human app\\agents\\sentiment\\newsagent\\training\\data\\active_sheet.csv ^
        --gold app\\agents\\sentiment\\newsagent\\training\\data\\gold.csv
Rules: a human label always replaces the LLM label for the same article; flagged LLM labels that
no human re-labelled are dropped; gold (test) headlines are never included.
Then:  python -m app.agents.sentiment.newsagent.training.scripts.finetune_finbert_labelled ^
    --labels app\\agents\\sentiment\\newsagent\\training\\data\\train_mix.csv ^
    --gold app\\agents\\sentiment\\newsagent\\training\\data\\gold.csv ^
    --out app\\agents\\sentiment\\newsagent\\training\\runs\\mix1
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
    ap.add_argument("--llm", required=True)
    ap.add_argument("--flagged")
    ap.add_argument("--human", help="the active sheet you labelled")
    ap.add_argument("--gold", required=True)
    ap.add_argument("--out", default=str(TRAINING_DATA_DIR / "train_mix.csv"))
    a = ap.parse_args()

    gold = read_labels(a.gold)
    gold_ids = {r["article_id"] for r in gold}
    gold_keys = {norm(r["headline"]) for r in gold}
    human = [r for r in read_labels(a.human)] if a.human else []
    human_ids = {r["article_id"] for r in human}
    flagged = set()
    if a.flagged:
        with open(a.flagged, newline="", encoding="utf-8-sig") as fh:
            flagged = {r["article_id"] for r in csv.DictReader(fh)}

    out, used = [], set()
    for src, rows in (("human", human), ("llm", read_labels(a.llm))):
        for r in rows:
            if r["article_id"] in used or r["article_id"] in gold_ids or norm(r["headline"]) in gold_keys:
                continue
            if src == "llm" and r["article_id"] in flagged and r["article_id"] not in human_ids:
                continue
            used.add(r["article_id"])
            out.append({**r, "source": src})
    with open(a.out, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["article_id", "kind", "date", "headline", "label", "source"])
        for r in out:
            w.writerow([r["article_id"], r["kind"], r["date"], r["headline"], r["label"], r["source"]])
    n_h = sum(1 for r in out if r["source"] == "human")
    print(f"wrote {a.out}: {len(out)} rows ({n_h} human, {len(out) - n_h} LLM); "
          f"{len(flagged - human_ids)} flagged LLM labels dropped")


if __name__ == "__main__":
    main()
