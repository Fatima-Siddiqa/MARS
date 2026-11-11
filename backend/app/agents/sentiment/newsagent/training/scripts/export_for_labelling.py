"""Export PSX headlines for hand-labelling (read-only on the database).

Writes two CSVs:
  labelling.csv    - the headlines to label (column `label` is blank)
  relabel_100.csv  - a random 100-row subset, shuffled, blank labels. Label it
                     at least a week AFTER finishing labelling.csv: it measures how
                     consistent you are with yourself (single-annotator quality check).

The files do NOT contain FinBERT's predictions or any price information, so your
labels are not anchored by the model and are blind to what the stock did afterwards.

LABELLING GUIDE (write `positive`, `negative` or `neutral` in the label column)
  Judge each headline as a shareholder of the named company would, for the
  next few weeks, using only what the headline says.
    positive : good for the business or the share price (profit up, dividend or
               bonus declared, big contract, upgrade, approval, debt cleared)
    negative : bad (loss, profit down, dividend cut, default, penalty, downgrade,
               shutdown, circular-debt build-up, regulatory action)
    neutral  : no clear direction (routine notices, meeting schedules, results
               announced without figures, market-wide commentary, unclear)
  When torn between a direction and neutral, choose neutral. Put doubts in `notes`.

Usage (backend folder, venv active):
    python -m app.agents.sentiment.newsagent.training.scripts.export_for_labelling ^
        --db app\\agents\\sentiment\\newsagent\\data\\snapshots\\news_after_aliases.db
"""
import argparse
import csv
import random
import re
import sqlite3
from pathlib import Path


def norm(s):
    s = re.sub(r"[^a-z0-9 ]+", " ", (s or "").lower())
    return re.sub(r"\s+", " ", re.sub(r"\d+", " ", s)).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--announcements", type=int, default=150,
                    help="how many distinct announcement headlines to include")
    ap.add_argument("--seed", type=int, default=11)
    training_data = Path(__file__).resolve().parents[1] / "data"
    ap.add_argument("--out", default=str(training_data / "labelling.csv"))
    ap.add_argument("--relabel-out", default=str(training_data / "relabel_100.csv"))
    ap.add_argument("--relabel-n", type=int, default=100)
    a = ap.parse_args()

    conn = sqlite3.connect(a.db)
    conn.execute("PRAGMA query_only = ON")
    rows = conn.execute(
        """SELECT a.id, a.kind, a.outlet, a.effective_at, a.headline, a.cluster_id,
                  GROUP_CONCAT(DISTINCT e.entity_id) AS tickers
           FROM articles a
           JOIN article_entities e ON e.article_id = a.id
                AND e.entity_type = 'ticker' AND e.relevance >= 1.0
           GROUP BY a.id ORDER BY a.effective_at, a.id""").fetchall()
    conn.close()

    news, ann = [], []
    seen_cluster, seen_head = set(), set()
    for aid, kind, outlet, eff, head, cid, tickers in rows:
        if not head:
            continue
        rec = {"article_id": aid, "kind": kind, "ticker": tickers or "", "outlet": outlet or "",
               "date": (eff or "")[:10], "headline": head, "label": "", "notes": ""}
        if kind == "news":
            key = cid if cid is not None else aid
            if key in seen_cluster:
                continue
            seen_cluster.add(key)
            news.append(rec)
        else:
            h = norm(head)
            if h in seen_head:
                continue
            seen_head.add(h)
            ann.append(rec)

    rng = random.Random(a.seed)
    rng.shuffle(ann)
    chosen = news + ann[: a.announcements]
    rng.shuffle(chosen)

    cols = ["article_id", "kind", "ticker", "outlet", "date", "headline", "label", "notes"]
    with open(a.out, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(chosen)

    sub = rng.sample(chosen, min(a.relabel_n, len(chosen)))
    with open(a.relabel_out, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(sub)

    print(f"news headlines (one per cluster): {len(news)}")
    print(f"distinct announcement headlines available: {len(ann)} (using {min(a.announcements, len(ann))})")
    print(f"wrote {len(chosen)} rows to {a.out} and {len(sub)} rows to {a.relabel_out}")
    print("Open the CSVs in Excel/LibreOffice, fill the `label` column, save as CSV (UTF-8).")


if __name__ == "__main__":
    main()