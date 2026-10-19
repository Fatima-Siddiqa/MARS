"""Walk-forward evaluation + optional model adoption (methodology sec 6).

Run from backend/:  python -m scripts.technical_eval
Splits: hyper-params fit on 2017-07..2020-12, chosen on 2021; each test year 2022.. is
predicted by a model refit on all earlier data with purging (label window must end before the
test block) and a 20-day embargo. Adopt only if the shipped blend beats Level 1 in >=3 test
years AND the block bootstrap of the daily rank-IC difference is positive in >= BOOT_P of resamples.
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

from app.agents.technical import model as M
from app.agents.technical.features import (
    FEATURE_COLS,
    compute_raw,
    family_scores,
    feature_scores,
    rule_score,
)
from app.ingestion.prices import UNIVERSE, load_benchmark, load_stock

HORIZON, EMBARGO = 20, 20
TRAIN_START, VAL_END = "2017-07-01", "2021-12-31"
TEST_YEARS = [2022, 2023, 2024, 2025, 2026]
BOOT_P, N_BOOT, BLOCK = 0.90, 2000, 20
C_GRID = [0.01, 0.1, 1.0]


def level1_score(raw: pd.DataFrame) -> pd.Series:
    """Level 1 baseline: 4 fixed signals (price/SMA50, SMA20/SMA50, MACD hist, RSI zone)."""
    c = raw["close"]
    a = np.tanh((c / raw["sma50"] - 1) / 0.05)
    b = np.tanh((raw["sma20"] / raw["sma50"] - 1) / 0.03)
    m = np.tanh((raw["macd_hist"] / c) / 0.005)
    r = raw["rsi14"]
    d = np.where(r < 30, np.minimum(1, (30 - r) / 15), np.where(r > 70, -np.minimum(1, (r - 70) / 15), 0.0))  # noqa: E501
    return pd.concat([a, b, m, pd.Series(d, index=c.index).where(r.notna())], axis=1).mean(axis=1, skipna=False)  # noqa: E501


def build_panel(tickers: list[str] | None = None) -> pd.DataFrame:
    bench = load_benchmark()
    if bench is None:
        raise SystemExit("KSE-100 file missing (data/raw/KSE_100_index.csv)")
    frames = []
    for t in tickers or UNIVERSE:
        df = load_stock(t)
        if df is None:
            print(f"skip {t}: no file")
            continue
        raw = compute_raw(df, bench)
        sc = feature_scores(raw)
        b = bench.reindex(df.index.union(bench.index)).ffill().reindex(df.index)
        p = sc.copy()
        p["rule"] = rule_score(family_scores(sc))
        p["level1"] = level1_score(raw)
        fs, fb = df["Close"].shift(-HORIZON) / df["Close"] - 1, b.shift(-HORIZON) / b - 1
        p["fwd_excess"] = fs - fb
        p["label_end"] = pd.Series(df.index, index=df.index).shift(-HORIZON)
        p["ticker"] = t
        frames.append(p[p.index >= TRAIN_START].dropna(subset=["fwd_excess", "label_end"]))
    panel = pd.concat(frames).rename_axis("date").reset_index()
    panel["y"] = (panel["fwd_excess"] > 0).astype(int)
    return panel


def _xy(d: pd.DataFrame):
    return d[FEATURE_COLS].fillna(0.0).to_numpy(float), d["y"].to_numpy()


def _train_mask(panel, all_dates, test_start):
    emb = all_dates[all_dates < test_start][-EMBARGO]
    return (panel["date"] < emb) & (panel["label_end"] < test_start)


def _fit(kind, hp, X, y):
    if kind == "logit":
        return LogisticRegression(C=hp, max_iter=2000).fit(X, y)
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=hp,
                                          min_samples_leaf=200, l2_regularization=1.0,
                                          early_stopping=False, random_state=0).fit(X, y)


def select_hyperparams(panel, all_dates):
    tr = panel[_train_mask(panel, all_dates, pd.Timestamp("2021-01-01"))]
    va = panel[(panel["date"] >= "2021-01-01") & (panel["label_end"] <= VAL_END)]
    Xt, yt = _xy(tr)
    Xv, yv = _xy(va)
    best = {}
    for C in C_GRID:
        m = _fit("logit", C, Xt, yt)
        auc = roc_auc_score(yv, m.predict_proba(Xv)[:, 1])
        if "logit" not in best or auc > best["logit"]["auc"]:
            best["logit"] = {"hp": C, "auc": auc}
    g = _fit("gbt", 300, Xt, yt)
    ll = [log_loss(yv, p[:, 1]) for p in g.staged_predict_proba(Xv)]
    n = int(np.argmin(ll)) + 1
    gm = _fit("gbt", n, Xt, yt)
    best["gbt"] = {"hp": n, "auc": roc_auc_score(yv, gm.predict_proba(Xv)[:, 1])}
    kind = max(best, key=lambda k: best[k]["auc"])
    return kind, best


def daily_ic(d: pd.DataFrame, col: str) -> pd.Series:
    def f(g):
        return g[col].rank().corr(g["fwd_excess"].rank()) if len(g) >= 5 and g[col].nunique() > 1 else np.nan  # noqa: E501
    return d.groupby("date").apply(f, include_groups=False).dropna()


def hit_and_spread(d: pd.DataFrame, col: str) -> dict:
    bull, bear = d[d[col] > 0.2], d[d[col] < -0.2]
    hits = list(bull["fwd_excess"] > 0) + list(bear["fwd_excess"] < 0)

    def spread(g):
        if len(g) < 6:
            return np.nan
        r = g[col].rank(method="first")
        k = len(g) // 3
        return g.loc[r.nlargest(k).index, "fwd_excess"].mean() - g.loc[r.nsmallest(k).index, "fwd_excess"].mean()  # noqa: E501
    return {"hit_rate": float(np.mean(hits)) if hits else None, "n_calls": len(hits),
            "top_minus_bottom": float(d.groupby("date").apply(spread, include_groups=False).mean())}


def block_bootstrap_p(diff: pd.Series, seed=0) -> float:
    x = diff.to_numpy()
    rng = np.random.default_rng(seed)
    nb = max(1, len(x) // BLOCK)
    starts = np.arange(0, len(x) - BLOCK + 1)
    means = [np.concatenate([x[s:s + BLOCK] for s in rng.choice(starts, nb)]).mean() for _ in range(N_BOOT)]  # noqa: E501
    return float(np.mean(np.array(means) > 0))


def run(save: bool = True) -> dict:
    panel = build_panel()
    all_dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
    kind, hp_info = select_hyperparams(panel, all_dates)
    hp = hp_info[kind]["hp"]
    rng = np.random.default_rng(0)
    preds = []
    for yr in TEST_YEARS:
        ts, te = pd.Timestamp(f"{yr}-01-01"), pd.Timestamp(f"{yr}-12-31")
        te_d = panel[(panel["date"] >= ts) & (panel["date"] <= te)].copy()
        tr = panel[_train_mask(panel, all_dates, ts)]
        if te_d.empty or tr.empty:
            continue
        m = _fit(kind, hp, *_xy(tr))
        te_d["p"] = m.predict_proba(_xy(te_d)[0])[:, 1]
        te_d["model"] = 2 * te_d["p"] - 1
        w = M.BLEND_WEIGHT
        te_d["blend"] = w * te_d["model"] + (1 - w) * te_d["rule"].fillna(0)
        te_d["random"] = rng.uniform(-1, 1, len(te_d))
        te_d["always_bull"] = 1.0
        preds.append(te_d)
    test = pd.concat(preds)
    cols = ["level1", "rule", "model", "blend", "random"]
    ics = {c: daily_ic(test, c) for c in cols}
    per_year = {}
    for yr in TEST_YEARS:
        per_year[yr] = {c: round(float(ics[c][ics[c].index.year == yr].mean()), 4) for c in cols}
    wins = sum(per_year[y]["blend"] > per_year[y]["level1"] for y in per_year)
    diff = (ics["blend"] - ics["level1"]).dropna()
    p_pos = block_bootstrap_p(diff)
    adopted = bool(wins >= 3 and p_pos >= BOOT_P)
    report = {
        "adopted": adopted, "model_kind": kind, "hyperparams": hp_info, "test_years": TEST_YEARS,
        "mean_rank_ic": {c: round(float(ics[c].mean()), 4) for c in cols},
        "rank_ic_by_year": per_year, "years_blend_beats_level1": wins,
        "bootstrap_p_blend_gt_level1": round(p_pos, 3),
        "hit_rate_and_spread": {c: hit_and_spread(test, c) for c in ["level1", "rule", "blend"]},
        "always_bullish_hit_rate": float((test["fwd_excess"] > 0).mean()),
        "n_test_rows": len(test), "n_train_rows_final": int(len(panel)),
        "variants_tried": len(C_GRID) + 1, "feature_cols": FEATURE_COLS,
    }
    final = None
    if save:
        M.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        if adopted:
            tr = panel[panel["label_end"] <= panel["date"].max()]
            final = _fit(kind, hp, *_xy(tr))
            joblib.dump(final, M.MODEL_PATH)
            if kind == "logit":
                report["logit_coefficients"] = dict(zip(FEATURE_COLS, map(float, final.coef_[0]), strict=True))  # noqa: E501
        elif M.MODEL_PATH.exists():
            M.MODEL_PATH.unlink()
        M.REPORT_PATH.write_text(json.dumps(report, indent=2, default=float))
    return report
