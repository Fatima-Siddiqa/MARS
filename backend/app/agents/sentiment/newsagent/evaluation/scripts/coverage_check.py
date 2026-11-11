"""How often is a ticker's rolling window informed by actual NEWS (not announcements)?

Read-only. Mirrors the engine's rules: items usable only after effective_at, decision
time 11:00 UTC, window of N days, only the first item of each cluster counts per ticker,
headline-level ticker matches only (relevance >= 1.0).

Usage (from the backend folder, venv active):
    python -m app.agents.sentiment.newsagent.evaluation.scripts.coverage_check
    python -m app.agents.sentiment.newsagent.evaluation.scripts.coverage_check ^
        --db app\\agents\\sentiment\\newsagent\\data\\snapshots\\news_backup.db
"""
import argparse
import sqlite3
from collections import defaultdict

import numpy as np

from app.agents.sentiment.newsagent.config import DB_PATH


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(DB_PATH))
    ap.add_argument("--window", type=int, default=14)
    ap.add_argument("--decision-hour", type=int, default=11)
    ap.add_argument("--start", default="2018-01-01")
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    conn.execute("PRAGMA query_only = ON")
    rows = conn.execute(
        """SELECT e.entity_id, a.kind, a.effective_at, a.cluster_id, a.id
           FROM articles a JOIN article_entities e ON e.article_id = a.id
           WHERE e.entity_type = 'ticker' AND e.relevance >= 1.0
           ORDER BY a.effective_at, a.id"""
    ).fetchall()
    conn.close()

    seen, times = set(), defaultdict(lambda: {"news": [], "announcement": []})
    for ticker, kind, eff, cid, aid in rows:
        key = (ticker, cid if cid is not None else aid)
        if key in seen:
            continue          # not the first member of its cluster
        seen.add(key)
        times[ticker][kind].append(np.datetime64(eff[:19]))

    last_ts = max(t for d in times.values() for lst in d.values() for t in lst)
    days = np.arange(np.datetime64(args.start), last_ts.astype("datetime64[D]") + np.timedelta64(1, "D"))
    cut = days.astype("datetime64[s]") + np.timedelta64(args.decision_hour, "h")
    lo = cut - np.timedelta64(args.window, "D")

    per_year = defaultdict(lambda: [0, 0, 0, 0])      # days, days_with_news, sum_news, sum_ann
    per_ticker_test = defaultdict(lambda: [0, 0])     # 2022+: days, days_with_news
    years = days.astype("datetime64[Y]").astype(int) + 1970

    for ticker, d in sorted(times.items()):
        counts = {}
        for kind in ("news", "announcement"):
            arr = np.sort(np.array(d[kind], dtype="datetime64[s]"))
            counts[kind] = (np.searchsorted(arr, cut, side="right")
                            - np.searchsorted(arr, lo, side="right"))
        has_news = counts["news"] > 0
        for y in np.unique(years):
            m = years == y
            r = per_year[int(y)]
            r[0] += int(m.sum())
            r[1] += int(has_news[m].sum())
            r[2] += int(counts["news"][m].sum())
            r[3] += int(counts["announcement"][m].sum())
        t = years >= 2022
        per_ticker_test[ticker][0] += int(t.sum())
        per_ticker_test[ticker][1] += int(has_news[t].sum())

    print(f"\nWindow = {args.window} days, decision {args.decision_hour}:00 UTC, db = {args.db}\n")
    print(f"{'year':<6}{'ticker-days':>12}{'with news':>11}{'mean news':>11}{'mean annc':>11}")
    for y in sorted(per_year):
        n, w, sn, sa = per_year[y]
        print(f"{y:<6}{n:>12}{w / n:>11.1%}{sn / n:>11.2f}{sa / n:>11.2f}")
    print("\n2022+ (test window) share of days with news, by ticker:")
    for t, (n, w) in sorted(per_ticker_test.items()):
        print(f"  {t:<8}{w / n:>7.1%}")


if __name__ == "__main__":
    main()