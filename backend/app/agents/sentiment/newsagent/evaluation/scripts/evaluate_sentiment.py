"""Evaluate the MARS sentiment score against forward excess returns (read-only).

For every trading day t in the evaluation window it takes the agent's score for
ticker i (as computed by Engine.score_table from a COPY of the database), and
correlates it (Spearman rank IC) with the H-trading-day forward excess return:

    fwd_excess = AdjClose[t+H]/AdjClose[t] - 1  -  KSE100[t+H]/KSE100[t] - 1

Two variants are evaluated from the same database:
    combined   - the agent exactly as it is (news + PSX announcements)
    news_only  - announcements removed from the copy, so tone comes from news only

Confidence intervals come from a block bootstrap over calendar blocks of H trading
days, because H-day forward returns on consecutive days overlap heavily.
The bootstrap resamples blocks of pre-computed ranks (a standard approximation to
re-ranking each replicate).

Run from the backend folder with the venv active:
    pip install openpyxl            (once)
    python -m app.agents.sentiment.newsagent.evaluation.scripts.evaluate_sentiment ^
        --db app\\agents\\sentiment\\newsagent\\data\\snapshots\\news_backup.db
    python -m app.agents.sentiment.newsagent.evaluation.scripts.evaluate_sentiment ^
        --db app\\agents\\sentiment\\newsagent\\data\\snapshots\\news_after_aliases.db

Nothing is written to your databases: the script works on temporary copies.
"""
import argparse
import csv
import hashlib
import shutil
import sqlite3
import sys
import tempfile
from datetime import date, datetime
from pathlib import Path

import numpy as np

# ticker -> price file name inside --prices-dir. CHECK SERL.xlsx really is Searle.
FILES = {
    "ENGROH": "ENGROH.xlsx",
    "HBL": "HBL.xlsx",
    "HUBC": "HUBC.xlsx",
    "LUCKY": "Lucky Cement.xlsx",
    "OGDCL": "OGDC.xlsx",
    "PPL": "PPL.xlsx",
    "PTCL": "PTCL.xlsx",
    "SEARL": "SERL.xlsx",
    "SYSTEMS": "Systems Limited.xlsx",
    "UBL": "UBL.xlsx",
}


# ------------------------------------------------------------------ prices
def load_stock(path):
    """Return (dates ndarray datetime64[D], adj_close ndarray) sorted ascending."""
    try:
        from openpyxl import load_workbook
    except ImportError:
        sys.exit("openpyxl is required to read the .xlsx price files: pip install openpyxl")
    wb = load_workbook(path, read_only=True, data_only=True)
    rows = wb.worksheets[0].iter_rows(values_only=True)
    header = [str(h).strip() if h is not None else "" for h in next(rows)]
    di = header.index("Date")
    ci = header.index("Adj. Close") if "Adj. Close" in header else header.index("Close")
    out = {}
    for r in rows:
        d, p = r[di], r[ci]
        if isinstance(d, datetime):
            d = d.date()
        if not isinstance(d, date) or p in (None, ""):
            continue
        try:
            out[d] = float(p)
        except (TypeError, ValueError):
            continue
    wb.close()
    ds = sorted(out)
    return (np.array(ds, dtype="datetime64[D]"), np.array([out[d] for d in ds], dtype=float))


def load_index(path):
    out = {}
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            try:
                d = datetime.strptime(row["Date"].strip(), "%d-%b-%Y").date()
                v = float(row["KSE 100 Index"])
            except (ValueError, KeyError, TypeError):
                continue          # skips the stray 'Daily' header row
            out[d] = v
    ds = sorted(out)
    return (np.array(ds, dtype="datetime64[D]"), np.array([out[d] for d in ds], dtype=float))


def forward_excess(dates, px, idx_dates, idx_vals, h):
    """Per stock trading day i: H-day forward stock return minus index return. NaN if unavailable."""
    n = len(dates)
    pos = np.searchsorted(idx_dates, dates, side="right") - 1      # as-of index level
    ok_idx = pos >= 0
    lvl = np.where(ok_idx, idx_vals[np.clip(pos, 0, None)], np.nan)
    ex = np.full(n, np.nan)
    if n > h:
        ex[: n - h] = (px[h:] / px[: n - h] - 1.0) - (lvl[h:] / lvl[: n - h] - 1.0)
    return ex


# ----------------------------------------------------------- score sources
def make_conn(db, tmpdir, name, news_only):
    dst = Path(tmpdir) / f"{name}.db"
    shutil.copyfile(db, dst)
    conn = sqlite3.connect(dst)
    conn.row_factory = sqlite3.Row
    if news_only:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("DELETE FROM articles WHERE kind = 'announcement'")
        conn.commit()
    return conn


def get_scores(conn, start, end, tickers):
    from app.agents.sentiment.newsagent.engine import Engine
    rows = Engine(conn).score_table(start, end, tickers)
    out = {}
    for r in rows:
        nt = r.get("net_tone")
        out[(r["ticker"], date.fromisoformat(r["as_of"]))] = (
            float(r["score"]), float("nan") if nt is None else float(nt))
    return out


def news_counts(conn, ticker, dates, window_days, hour):
    """Primary news items (headline-level ticker match) inside the window before each decision."""
    rows = conn.execute(
        """SELECT a.effective_at, a.cluster_id, a.id FROM articles a
           JOIN article_entities e ON e.article_id = a.id
           WHERE e.entity_type='ticker' AND e.entity_id=? AND e.relevance>=1.0 AND a.kind='news'
           ORDER BY a.effective_at, a.id""", (ticker,)).fetchall()
    seen, ts = set(), []
    for r in rows:
        key = r["cluster_id"] if r["cluster_id"] is not None else r["id"]
        if key in seen:
            continue
        seen.add(key)
        ts.append(np.datetime64(r["effective_at"][:19]))
    arr = np.sort(np.array(ts, dtype="datetime64[s]"))
    cut = np.array([np.datetime64(f"{d.isoformat()}T{hour:02d}:00:00") for d in dates],
                   dtype="datetime64[s]")
    lo = cut - np.timedelta64(window_days, "D")
    return np.searchsorted(arr, cut, side="right") - np.searchsorted(arr, lo, side="left")


# ------------------------------------------------------------------- stats
def rank(a):
    u, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    avg = np.cumsum(cnt) - (cnt - 1) / 2.0
    return avg[inv]


def corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else float("nan")


def ic_with_ci(dates_ord, x, y, block, reps, rng):
    """Spearman IC and a time-block bootstrap 95% CI. dates_ord: integer day ordinals."""
    order = np.argsort(dates_ord, kind="stable")
    d, x, y = dates_ord[order], x[order], y[order]
    ic = corr(rank(x), rank(y))
    rx, ry = rank(x), rank(y)
    _, first = np.unique(d, return_index=True)
    bounds = list(first) + [len(d)]
    blocks = [(bounds[i], bounds[min(i + block, len(first))])
              for i in range(0, len(first), block)]
    nb = len(blocks)
    boots = []
    for _ in range(reps):
        pick = rng.integers(0, nb, nb)
        idx = np.concatenate([np.arange(*blocks[k]) for k in pick])
        v = corr(rx[idx], ry[idx])
        if not np.isnan(v):
            boots.append(v)
    lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots else (float("nan"),) * 2)
    return ic, lo, hi, nb


def fmt(label, n, ic, lo, hi, nb):
    flag = "" if np.isnan(lo) else ("   <- CI excludes 0" if (lo > 0 or hi < 0) else "")
    return f"  {label:<34}{n:>7}{ic:>8.3f}   [{lo:>6.3f}, {hi:>6.3f}]  blocks={nb}{flag}"


# -------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--prices-dir", default=r"..\data\raw\stocks")
    ap.add_argument("--index", default=r"..\data\raw\KSE_100_index.csv")
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--end", default=None, help="default: last date with prices")
    ap.add_argument("--horizon", type=int, default=20, help="forward horizon in trading days")
    ap.add_argument("--tickers", nargs="*")
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--dump", help="write observation-level rows to this CSV")
    a = ap.parse_args()

    from app.agents.sentiment.newsagent.config import DEFAULT_PARAMS
    window_days, hour = DEFAULT_PARAMS.window_days, DEFAULT_PARAMS.decision_hour_utc
    start = date.fromisoformat(a.start)
    tickers = [t for t in FILES if not a.tickers or t in a.tickers]

    idx_dates, idx_vals = load_index(a.index)
    prices, skipped = {}, []
    for t in tickers:
        p = Path(a.prices_dir) / FILES[t]
        if not p.is_file():
            skipped.append(t)
            continue
        prices[t] = load_stock(p)
        d, v = prices[t]
        print(f"prices {t:<8} {d[0]} -> {d[-1]}  n={len(d)}  last adj close={v[-1]:.2f}  "
              f"max |daily move|={np.nanmax(np.abs(np.diff(v) / v[:-1])):.1%}")
    if skipped:
        print("WARNING no price file for:", ", ".join(skipped))
    tickers = [t for t in tickers if t in prices]
    last_price = max(prices[t][0][-1] for t in tickers).astype(object)
    end = date.fromisoformat(a.end) if a.end else last_price

    sha = hashlib.sha1(Path(a.db).read_bytes()).hexdigest()[:12]
    print(f"\ndb={a.db} (sha1 {sha})  params_key={DEFAULT_PARAMS.key()}")
    print(f"window {start} -> {end}  horizon={a.horizon} trading days  benchmark=KSE-100  "
          f"reps={a.reps} seed={a.seed}")

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        conns = {"combined": make_conn(a.db, tmp, "combined", False),
                 "news_only": make_conn(a.db, tmp, "news_only", True)}
        scores = {k: get_scores(c, start, end, tickers) for k, c in conns.items()}

        obs = {"ticker": [], "date": [], "excess": [], "news_n": [],
               "combined_score": [], "combined_tone": [], "news_only_score": [], "news_only_tone": []}
        for t in tickers:
            dts, px = prices[t]
            ex = forward_excess(dts, px, idx_dates, idx_vals, a.horizon)
            keep = [i for i, d in enumerate(dts.astype(object))
                    if start <= d <= end and not np.isnan(ex[i])
                    and (t, d) in scores["combined"] and (t, d) in scores["news_only"]]
            if not keep:
                continue
            ds = [dts[i].astype(object) for i in keep]
            nn = news_counts(conns["combined"], t, ds, window_days, hour)
            for j, i in enumerate(keep):
                d = ds[j]
                obs["ticker"].append(t); obs["date"].append(d.toordinal())
                obs["excess"].append(ex[i]); obs["news_n"].append(int(nn[j]))
                for v in ("combined", "news_only"):
                    s, tone = scores[v][(t, d)]
                    obs[f"{v}_score"].append(s); obs[f"{v}_tone"].append(tone)
        for c in conns.values():
            c.close()

    tk = np.array(obs["ticker"])
    dord = np.array(obs["date"])
    y = np.array(obs["excess"])
    news_n = np.array(obs["news_n"])
    n_all = len(y)
    if n_all == 0:
        sys.exit("No overlapping observations. Check dates, tickers and price files.")
    print(f"\nobservations: {n_all}  ticker-days with >=1 news in window: "
          f"{(news_n > 0).sum()} ({(news_n > 0).mean():.1%})")
    print(f"last usable decision date: {date.fromordinal(int(dord.max()))} "
          f"(prices end {last_price}, need {a.horizon} days of forward return)")

    rng = np.random.default_rng(a.seed)
    print(f"\n  {'spec':<34}{'n':>7}{'IC':>8}   {'95% block-bootstrap CI':<22}")
    for variant in ("combined", "news_only"):
        for sig, key in (("score", f"{variant}_score"), ("raw tone", f"{variant}_tone")):
            x = np.array(obs[key], dtype=float)
            for subset, mask in (("all days", np.ones(n_all, bool)), ("news days", news_n > 0)):
                m = mask & np.isfinite(x) & np.isfinite(y)
                if m.sum() < 30:
                    print(f"  {variant + ' ' + sig + ', ' + subset:<34}{int(m.sum()):>7}   too few")
                    continue
                ic, lo, hi, nb = ic_with_ci(dord[m], x[m], y[m], a.horizon, a.reps, rng)
                print(fmt(f"{variant} {sig}, {subset}", int(m.sum()), ic, lo, hi, nb))
        print()

    print("per ticker, score vs forward excess return (no CI; descriptive only):")
    print(f"  {'ticker':<9}{'n':>6}{'combined':>10}{'news_only':>11}{'news-day share':>16}")
    for t in tickers:
        m = tk == t
        if m.sum() < 30:
            continue
        c1 = corr(rank(np.array(obs['combined_score'])[m]), rank(y[m]))
        c2 = corr(rank(np.array(obs['news_only_score'])[m]), rank(y[m]))
        print(f"  {t:<9}{int(m.sum()):>6}{c1:>10.3f}{c2:>11.3f}{(news_n[m] > 0).mean():>15.1%}")

    if a.dump:
        with open(a.dump, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(list(obs))
            for row in zip(*obs.values()):
                w.writerow(row)
        print(f"\nobservation-level rows written to {a.dump}")


if __name__ == "__main__":
    main()