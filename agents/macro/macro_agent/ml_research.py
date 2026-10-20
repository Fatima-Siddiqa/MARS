"""Research only (not used by the live agent): do ML challengers beat the rule-based macro score?

Monthly design. At each month-end t the as-of inputs come from the same no-look-ahead pipeline the agent uses
(rule composite, pillar scores, oriented feature scores). Targets are KSE-100 log returns:
  * skip  (primary): log K[t+2] / K[t+1]. Clean even when K is a monthly-AVERAGE index (World Bank GEM): it shares
                     no prices with anything known at t, so averaging cannot leak predictability into the label.
  * next  (secondary): log K[t+1] / K[t]. With a monthly-average index this overlaps month t and is mechanically
                     autocorrelated (Working 1960), so it inflates momentum/AR results. Reported for comparison only.

Walk-forward: expanding window, refit once per test block, purge = label horizon (a training row is used only
when its label was fully observed before the test block starts). Rows whose label reaches into the lock-box period
are excluded, so this never touches the agent's final hold-out.

Models: historical mean, AR(1)/AR(3) on market returns (time-series baselines), the rules (raw and linearly
calibrated), ridge on pillars / features, XGBoost (monotone-constrained and free). Ablations: leave-one-pillar-out,
feature groups, monotone constraint, label choice, sub-period. Metrics: Spearman IC (+ block-bootstrap CI),
directional hit rate, out-of-sample R^2 vs the historical mean (Campbell & Thompson 2008), long/flat Sharpe.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .pipeline import MacroPipeline

HORIZON_END = {"skip": 2, "next": 1}   # months after t at which the label is fully observed


@dataclass
class MLResult:
    report: dict
    predictions: pd.DataFrame
    path_json: Path | None = None
    path_md: Path | None = None


# ------------------------------------------------------------------------------------------- data
def month_end_rows(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last calendar date of each month present in a daily index (the as-of date of the monthly decision)."""
    s = pd.Series(index, index=index)
    return pd.DatetimeIndex(s.groupby(index.to_period("M")).max().to_numpy())


def market_monthly(kse: pd.Series) -> pd.Series:
    """KSE level per month (month-end stamp). Daily input -> last value of the month; monthly input unchanged."""
    k = kse.dropna().sort_index()
    return k.groupby(k.index.to_period("M")).last().pipe(lambda s: s.set_axis(s.index.to_timestamp("M")))


def make_targets(km: pd.Series) -> pd.DataFrame:
    lk = np.log(km)
    return pd.DataFrame({"next": lk.shift(-1) - lk, "skip": lk.shift(-2) - lk.shift(-1),
                         "mom1": lk - lk.shift(1), "mom2": lk.shift(1) - lk.shift(2), "mom3": lk.shift(2) - lk.shift(3)})


def build_dataset(pipe: MacroPipeline) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    df, sf, _, _ = pipe.score_series(None)
    rows = month_end_rows(df.index)
    active = list(pipe.cfg.active_features())
    feats = sf.adj_score.reindex(columns=active).loc[rows]
    pillars = sf.pillars.loc[rows].add_prefix("p_")
    X = pd.concat([df.loc[rows, ["score"]].rename(columns={"score": "rule_score"}), pillars,
                   feats.add_prefix("f_")], axis=1)
    X.index = X.index.to_period("M").to_timestamp("M")
    km = market_monthly(pipe.kse100())
    if km.empty:
        raise RuntimeError("kse100 series is required (run `macro-agent fetch-open` then `macro-agent refresh`)")
    data = X.join(make_targets(km), how="inner")
    data = data[data["rule_score"].notna()]
    groups = {"pillars": list(pillars.columns), "features": [f"f_{c}" for c in active],
              "momentum": ["mom1", "mom2", "mom3"]}
    pill_of = {f"f_{k}": f"p_{v.pillar}" for k, v in pipe.cfg.active_features().items()}
    groups["pillar_of"] = pill_of  # type: ignore[assignment]
    return data, groups


# ------------------------------------------------------------------------------------------- models
def _ridge(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, alpha: float) -> np.ndarray:
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd == 0] = 1.0
    A, B = (Xtr - mu) / sd, (Xte - mu) / sd
    w = np.linalg.solve(A.T @ A + alpha * np.eye(A.shape[1]), A.T @ (ytr - ytr.mean()))
    return ytr.mean() + B @ w


def ridge_model(cols: list[str], alpha: float) -> Callable:
    def fit_predict(tr: pd.DataFrame, te: pd.DataFrame, y: str) -> np.ndarray:
        return _ridge(tr[cols].fillna(0).to_numpy(), tr[y].to_numpy(), te[cols].fillna(0).to_numpy(), alpha)
    return fit_predict


XGB_PARAMS = {"max_depth": 2, "eta": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5,
              "lambda": 5.0, "objective": "reg:squarederror", "nthread": 1, "verbosity": 0}
XGB_ROUNDS = 150   # small, shallow, heavily regularised: ~200 monthly rows cannot support more


def xgb_fit(X: pd.DataFrame, y: pd.Series, mono: tuple[int, ...], seed: int = 7):
    import xgboost as xgb  # native API: no scikit-learn dependency
    params = {**XGB_PARAMS, "seed": seed, "monotone_constraints": "(" + ",".join(map(str, mono)) + ")"}
    return xgb.train(params, xgb.DMatrix(X, label=y), num_boost_round=XGB_ROUNDS)


def xgb_model(cols: list[str], monotone: bool, monotone_cols: set[str] | None = None, seed: int = 7) -> Callable:
    def fit_predict(tr: pd.DataFrame, te: pd.DataFrame, y: str) -> np.ndarray:
        import xgboost as xgb
        mono = tuple(1 if (monotone and c in (monotone_cols or set())) else 0 for c in cols)
        return xgb_fit(tr[cols], tr[y], mono, seed).predict(xgb.DMatrix(te[cols]))
    return fit_predict


def hist_mean(tr: pd.DataFrame, te: pd.DataFrame, y: str) -> np.ndarray:
    return np.full(len(te), tr[y].mean())


def ar_model(lags: int) -> Callable:
    cols = [f"mom{i}" for i in range(1, lags + 1)]

    def fit_predict(tr: pd.DataFrame, te: pd.DataFrame, y: str) -> np.ndarray:
        t = tr.dropna(subset=cols)
        Xtr = np.column_stack([np.ones(len(t)), t[cols].to_numpy()])
        b, *_ = np.linalg.lstsq(Xtr, t[y].to_numpy(), rcond=None)
        Xte = np.column_stack([np.ones(len(te)), te[cols].fillna(0).to_numpy()])
        return Xte @ b
    return fit_predict


def rules_linear(tr: pd.DataFrame, te: pd.DataFrame, y: str) -> np.ndarray:
    b, a = np.polyfit(tr["rule_score"].to_numpy(), tr[y].to_numpy(), 1)
    return a + b * te["rule_score"].to_numpy()


def rules_raw(tr: pd.DataFrame, te: pd.DataFrame, y: str) -> np.ndarray:
    return te["rule_score"].to_numpy()   # unfitted; ranked against returns (IC/hit only, no R^2)


# ------------------------------------------------------------------------------------------- walk-forward
def folds(index: pd.DatetimeIndex, initial: int, test: int, horizon_end: int):
    """Expanding window over a monthly index. Train rows s satisfy s + horizon_end <= first test month - 1,
    i.e. their label was fully observed before the test block's first decision date."""
    n = len(index)
    start = initial
    while start < n:
        stop = min(start + test, n)
        train_end = start - horizon_end   # exclusive
        if train_end > 0:
            yield np.arange(0, train_end), np.arange(start, stop)
        start = stop


def walk_forward(data: pd.DataFrame, model: Callable, y: str, initial: int, test: int) -> pd.Series:
    preds = []
    for tr, te in folds(data.index, initial, test, HORIZON_END[y]):
        p = model(data.iloc[tr], data.iloc[te], y)
        preds.append(pd.Series(p, index=data.index[te]))
    return pd.concat(preds) if preds else pd.Series(dtype=float)


# ------------------------------------------------------------------------------------------- metrics
def block_bootstrap_ic(p: np.ndarray, y: np.ndarray, block: int = 6, reps: int = 500, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(p)
    if n < 2 * block:
        return (np.nan, np.nan)
    out = []
    for _ in range(reps):
        starts = rng.integers(0, n - block + 1, size=int(np.ceil(n / block)))
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        out.append(stats.spearmanr(p[idx], y[idx]).statistic)
    return float(np.nanpercentile(out, 5)), float(np.nanpercentile(out, 95))


def evaluate(pred: pd.Series, y: pd.Series, bench: pd.Series | None, fitted: bool = True, rank: bool = True) -> dict:
    """rank=False for forecasts that are constant within a test block (historical mean): their IC is meaningless."""
    d = pd.concat([pred.rename("p"), y.rename("y")], axis=1, sort=True).dropna()
    if len(d) < 12:
        return {"n": len(d)}
    p, yy = d["p"].to_numpy(), d["y"].to_numpy()
    ic = stats.spearmanr(p, yy) if rank else None
    lo, hi = block_bootstrap_ic(p, yy) if rank else (None, None)
    pos = (p > 0).astype(float)
    strat = pos * yy
    if not rank:
        ic, lo, hi = None, None, None
    out = {"n": len(d), "ic": None if ic is None else float(ic.statistic),
           "ic_p": None if ic is None else float(ic.pvalue), "ic_ci90": [lo, hi],
           "hit": float(np.mean(np.sign(p) == np.sign(yy))),
           "sharpe_long_flat": float(strat.mean() / strat.std() * np.sqrt(12)) if strat.std() > 0 else None,
           "pct_months_long": float(pos.mean())}
    if fitted and bench is not None:
        b = bench.reindex(d.index).to_numpy()
        out["r2_oos"] = float(1 - np.sum((yy - p) ** 2) / np.sum((yy - b) ** 2))
    return out


# ------------------------------------------------------------------------------------------- experiment
def run_ml_research(pipe: MacroPipeline, out_dir: str | Path | None = None, initial: int = 60, test: int = 12,
                    write: bool = True) -> MLResult:
    cfg = pipe.cfg
    ev = cfg.evaluation
    data, g = build_dataset(pipe)
    lock = pd.Timestamp(ev["lockbox_start"])
    alpha = float(ev.get("ridge_alpha", 10.0))
    feats, pills, mom = g["features"], g["pillars"], g["momentum"]
    mono_cols = set(feats) | set(pills) | {"rule_score"}   # oriented scores: higher = more bullish by construction
    pillar_of: dict[str, str] = g["pillar_of"]  # type: ignore[assignment]

    models: dict[str, tuple[Callable, bool]] = {
        "hist_mean": (hist_mean, True),
        "ar1": (ar_model(1), True),
        "ar3": (ar_model(3), True),
        "rules_raw": (rules_raw, False),
        "rules_linear": (rules_linear, True),
        "ridge_pillars": (ridge_model(pills, alpha), True),
        "ridge_features": (ridge_model(feats, alpha), True),
        "xgb_mono_features": (xgb_model(feats, True, mono_cols), True),
        "xgb_free_features": (xgb_model(feats, False), True),
        "xgb_mono_pillars": (xgb_model(pills, True, mono_cols), True),
        "xgb_mono_features+momentum": (xgb_model(feats + mom, True, mono_cols), True),
        "xgb_mono_features+rules": (xgb_model(feats + ["rule_score"], True, mono_cols), True),
    }
    # leave-one-pillar-out ablations on the two main learners
    for p in pills:
        keep = [f for f in feats if pillar_of.get(f) != p]
        if keep and len(keep) < len(feats):
            models[f"ridge_features-{p[2:]}"] = (ridge_model(keep, alpha), True)
            models[f"xgb_mono_features-{p[2:]}"] = (xgb_model(keep, True, mono_cols), True)

    report: dict = {"generated_at": datetime.now(timezone.utc).isoformat(), "model_version": cfg.model_version,
                    "design": {"frequency": "monthly", "initial_train_months": initial, "test_block_months": test,
                               "lockbox_start": str(lock.date()), "ridge_alpha": alpha, "features": feats},
                    "targets": {}}
    all_preds = {}
    for y, h in HORIZON_END.items():
        # exclude rows whose label is observed on/after the lock-box start
        dev = data[(data.index + pd.offsets.MonthEnd(h)) < lock].dropna(subset=[y])
        res, preds = {}, {}
        for name, (m, fitted) in models.items():
            preds[name] = walk_forward(dev, m, y, initial, test)
        bench = preds["hist_mean"]
        for name, (_, fitted) in models.items():
            res[name] = evaluate(preds[name], dev[y], bench, fitted, rank=name != "hist_mean")
        oos = preds["hist_mean"].index
        split = pd.Timestamp("2018-12-31")   # monetary pillar (policy rate since 2015 + 36 months) is live after this
        sub = {}
        for name in ("rules_raw", "ridge_features", "xgb_mono_features", "ar1"):
            sub[name] = {"pre_2019": evaluate(preds[name][preds[name].index <= split], dev[y], bench, models[name][1]),
                         "2019_on": evaluate(preds[name][preds[name].index > split], dev[y], bench, models[name][1])}
        report["targets"][y] = {"rows": len(dev), "oos_period": [str(oos.min().date()), str(oos.max().date())]
                                if len(oos) else None, "results": res, "subperiods": sub,
                                "buy_and_hold_sharpe": _bh(dev[y].reindex(oos))}
        all_preds[y] = pd.DataFrame(preds).join(dev[y].rename("actual"))
    report["xgb_contributions"] = _contributions(data, feats, mono_cols, lock)
    report["verdict"] = _verdict(report)
    predictions = pd.concat(all_preds, axis=1)
    out = MLResult(report, predictions)
    if write:
        od = Path(out_dir) if out_dir else cfg.root / "reports" / "ml"
        od.mkdir(parents=True, exist_ok=True)
        out.path_json = od / "ml_research.json"
        out.path_md = od / "ml_research.md"
        out.path_json.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        out.path_md.write_text(render_md(report), encoding="utf-8")
        predictions.to_csv(od / "ml_predictions.csv")
    return out


def _bh(y: pd.Series) -> float | None:
    y = y.dropna()
    return float(y.mean() / y.std() * np.sqrt(12)) if len(y) > 12 and y.std() > 0 else None


def _contributions(data: pd.DataFrame, feats: list[str], mono_cols: set[str], lock: pd.Timestamp) -> dict:
    """Mean |SHAP| per feature (XGBoost native pred_contribs) of the monotone model fit on all dev rows."""
    from xgboost import DMatrix
    dev = data[(data.index + pd.offsets.MonthEnd(2)) < lock].dropna(subset=["skip"])
    m = xgb_fit(dev[feats], dev["skip"], tuple(1 for _ in feats))
    c = m.predict(DMatrix(dev[feats]), pred_contribs=True)[:, :-1]
    s = pd.Series(np.abs(c).mean(0), index=[f[2:] for f in feats]).sort_values(ascending=False)
    return {k: float(v) for k, v in s.items()}


def _verdict(r: dict) -> str:
    res = r["targets"]["skip"]["results"]
    rules = res.get("rules_raw", {}).get("ic")
    best = max(((k, v.get("ic")) for k, v in res.items() if v.get("ic") is not None and k != "hist_mean"),
               key=lambda kv: kv[1], default=(None, None))
    r2 = {k: v.get("r2_oos") for k, v in res.items() if v.get("r2_oos") is not None}
    pos_r2 = [k for k, v in r2.items() if v > 0]
    sig = [k for k, v in res.items() if v.get("ic_ci90") and v["ic_ci90"][0] is not None and v["ic_ci90"][0] > 0]
    n_models = sum(1 for v in res.values() if v.get("ic") is not None)
    return (f"Clean (skip-month) target: rules IC={_f(rules)}; best IC {best[0]}={_f(best[1])}. "
            f"Models with positive out-of-sample R^2 vs historical mean: {pos_r2 or 'none'}. "
            f"Models whose 90% block-bootstrap IC interval excludes zero: {sig or 'none'}. "
            f"{n_models} variants were tested, so about {n_models * 0.05:.0f} would clear a 90% one-sided interval by chance; "
            "with ~160 out-of-sample months, IC differences below ~0.1 are within noise.")


def _f(x, fmt="{:+.3f}") -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else fmt.format(x)


def render_md(r: dict) -> str:
    L = ["# Macro agent: ML challengers vs rules (research)", "", f"Generated {r['generated_at']} - model {r['model_version']}", "",
         (f"Monthly walk-forward: {r['design']['initial_train_months']} initial months, refit every "
         f"{r['design']['test_block_months']} months, purge = label horizon, lock-box from {r['design']['lockbox_start']} excluded."), "",
         f"**Verdict:** {r['verdict']}", ""]
    for y, t in r["targets"].items():
        title = "skip-month (primary, clean)" if y == "skip" else "next-month (inflated by monthly averaging; comparison only)"
        L += [f"## Target: {title}", "", f"Rows {t['rows']}, out-of-sample {t['oos_period']}, buy-and-hold Sharpe {_f(t['buy_and_hold_sharpe'], '{:.2f}')}", "",
              "| model | n | IC | IC 90% CI | p | hit | R2 oos | Sharpe long/flat | % long |", "|---|---|---|---|---|---|---|---|---|"]
        for k, v in t["results"].items():
            if "hit" not in v:
                L.append(f"| {k} | {v.get('n')} | n/a | | | | | | |")
                continue
            ci = v["ic_ci90"]
            L.append(f"| {k} | {v['n']} | {_f(v['ic'])} | [{_f(ci[0])}, {_f(ci[1])}] | {_f(v['ic_p'], '{:.2f}')} | {v['hit']:.2f} | "
                     f"{_f(v.get('r2_oos'))} | {_f(v['sharpe_long_flat'], '{:.2f}')} | {v['pct_months_long']:.0%} |")
        L += ["", "Sub-periods (IC):", "", "| model | to 2018 | 2019 on |", "|---|---|---|"]
        for k, v in t["subperiods"].items():
            L.append(f"| {k} | {_f(v['pre_2019'].get('ic'))} (n={v['pre_2019'].get('n')}) | {_f(v['2019_on'].get('ic'))} (n={v['2019_on'].get('n')}) |")
        L.append("")
    L += ["## XGBoost (monotone) mean |SHAP| contribution, skip-month target", "", "| feature | mean abs contribution |", "|---|---|"]
    L += [f"| {k} | {v:.5f} |" for k, v in r["xgb_contributions"].items()]
    return "\n".join(L) + "\n"
