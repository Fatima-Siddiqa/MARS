"""Offline research and evaluation pipeline.
Label -> walk-forward split (expanding window, purge) -> compare baselines -> evaluate -> select -> lock-box test.
Macro predictors are often weak out-of-sample (Welch & Goyal; Goyal, Welch & Zafirov), so a small or null return
signal is reported as it is, and risk/context usefulness is reported separately."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import nnls

from .pipeline import MacroPipeline
from .scoring import ScoreFrame, score_features, state_of

STATES = ["Bullish", "Neutral", "Bearish"]


# ------------------------------------------------------------------------------------------------ 1. labels
def make_labels(close: pd.Series, horizon: int = 20, ticker_close: pd.Series | None = None) -> pd.DataFrame:
    """Forward `horizon`-trading-day return (and excess return for a ticker), forward realised vol and max drawdown.
    Row t describes days t+1..t+h; it is only known at t+h, which is why the splitter purges."""
    c = close.dropna().sort_index()
    fwd = c.shift(-horizon) / c - 1
    lr = np.log(c).diff()
    vals, mdd = c.to_numpy(), np.full(len(c), np.nan)
    for i in range(len(c) - horizon):
        w = vals[i: i + horizon + 1]
        mdd[i] = (w / np.maximum.accumulate(w) - 1).min()
    sd = lr.rolling(horizon).std().shift(-horizon) * np.sqrt(252)
    out = pd.DataFrame({"fwd_ret": fwd, "fwd_vol": sd, "fwd_mdd": mdd}, index=c.index)
    if ticker_close is not None:
        t = ticker_close.dropna().sort_index().reindex(c.index).ffill()
        out["fwd_excess"] = (t.shift(-horizon) / t - 1) - fwd
    return out.dropna(subset=["fwd_ret"])


# ------------------------------------------------------------------------------------------------ 2. splits
def walk_forward_folds(index: pd.DatetimeIndex, initial: int, test: int, purge: int):
    """Expanding window. Train = everything up to `purge` days before the test block, so no training label overlaps it."""
    folds, start = [], initial
    while start < len(index):
        stop = min(start + test, len(index))
        tr_end = start - purge
        if tr_end > 0:
            folds.append((index[:tr_end], index[start:stop]))
        start = stop
    return folds


# ------------------------------------------------------------------------------------------------ 3. baselines
def baseline_scores(sf: ScoreFrame, cfg, comp) -> dict[str, pd.Series]:
    s, w = sf.adj_score, sf.weight
    def wmean(cols):
        cols = [c for c in cols if c in s.columns]
        ws = w[cols].where(s[cols].notna(), 0.0).fillna(0.0)
        return ((s[cols].fillna(0.0) * ws).sum(axis=1) / ws.sum(axis=1).where(ws.sum(axis=1) > 0))
    return {
        "full_macro_agent": sf.composite,
        "no_macro": pd.Series(0.0, index=s.index),
        "policy_rate_only": wmean(cfg.quality["critical_groups"]["policy_rate"]),
        "pkr_only": wmean(cfg.quality["critical_groups"]["pkr_usd"]),
        "flat_features_no_pillars": wmean(list(s.columns)),
    }


def regime_labels(raw: pd.DataFrame, thr: dict) -> pd.Series:
    """Evaluation-only regime tags (the agent itself never labels regimes). Precedence: rate_hike > pkr_shock > oil_shock."""
    r = pd.Series("calm", index=raw.index, dtype=object)
    r[raw["brent_3m"].abs() >= thr["oil_shock_pct"]] = "oil_shock"
    r[raw["pkr_dep_30d"] >= thr["pkr_shock_pct"]] = "pkr_shock"
    r[raw["rate_change_6m"] >= thr["rate_hike_pp"]] = "rate_hike"
    return r


# ------------------------------------------------------------------------------------------------ 4. metrics
def _safe_spearman(a: pd.Series, b: pd.Series) -> tuple[float | None, float | None]:
    d = pd.concat([a, b], axis=1).dropna()
    if len(d) < 10 or d.iloc[:, 0].nunique() < 2:
        return None, None
    rho, p = stats.spearmanr(d.iloc[:, 0], d.iloc[:, 1])
    return float(rho), float(p)


def metrics(score: pd.Series, state: pd.Series, conf: pd.Series | None, lab: pd.DataFrame, horizon: int,
            regimes: pd.Series | None = None, bins: int = 5, target: str = "fwd_ret") -> dict:
    d = pd.concat([score.rename("score"), state.rename("state"), lab], axis=1, join="inner").dropna(subset=["score", target])
    if conf is not None:
        d["conf"] = conf.reindex(d.index)
    n = len(d)
    if n < 30:
        return {"n": n, "note": "too few observations"}
    rho, p_all = _safe_spearman(d["score"], d[target])
    sub = d.iloc[::horizon]                                         # non-overlapping windows -> honest significance
    rho_no, p_no = _safe_spearman(sub["score"], sub[target])
    nz = d[d["state"] != "Neutral"]
    hit = None if nz.empty else float(((nz["state"] == "Bullish") == (nz[target] > 0)).mean())
    dirn = d[d["score"] != 0]
    out = {
        "n": n, "n_non_overlapping": len(sub), "rank_corr": rho, "rank_corr_pvalue_overlapping_unreliable": p_all,
        "rank_corr_non_overlapping": rho_no, "rank_corr_pvalue_non_overlapping": p_no,
        "hit_rate_bull_bear": hit, "share_non_neutral": float(len(nz) / n),
        "directional_hit_rate": None if dirn.empty else float(((dirn["score"] > 0) == (dirn[target] > 0)).mean()),
        "base_rate_up": float((d[target] > 0).mean()),
        "by_state": {},
    }
    for st in STATES:
        g = d[d["state"] == st]
        if len(g):
            out["by_state"][st] = {"n": len(g), "mean_fwd_ret": float(g["fwd_ret"].mean()), "mean_fwd_vol": float(g["fwd_vol"].mean()),
                                   "mean_fwd_max_drawdown": float(g["fwd_mdd"].mean())}
    if conf is not None and d["conf"].nunique() > 1:
        try:
            d["bin"] = pd.qcut(d["conf"], bins, duplicates="drop")
            cal = d.groupby("bin", observed=True).apply(lambda g: pd.Series({
                "n": len(g), "mean_confidence": g["conf"].mean(),
                "directional_hit": ((g["score"] > 0) == (g[target] > 0)).mean()}), include_groups=False)
            out["confidence_calibration"] = [{"bin": str(i), **{k: float(v) for k, v in r.items()}} for i, r in cal.iterrows()]
        except ValueError:
            pass
    if regimes is not None:
        rg = {}
        for name, g in d.join(regimes.rename("regime"), how="left").groupby("regime"):
            r, _ = _safe_spearman(g["score"], g[target])
            rg[name] = {"n": len(g), "rank_corr": r, "directional_hit_rate": float(((g["score"] > 0) == (g[target] > 0)).mean())}
        out["by_regime"] = rg
    return out


# ------------------------------------------------------------------------------------------------ 5. model selection
def fit_pillar_weights(P: pd.DataFrame, y: pd.Series, alpha: float) -> np.ndarray:
    """Non-negative ridge of forward return on pillar scores -> pillar weights (mean 1). Equal weights if nothing fits."""
    d = pd.concat([P, y.rename("y")], axis=1).dropna(subset=["y"])
    X = d[P.columns].fillna(0.0).to_numpy()
    yy = d["y"].to_numpy()
    if len(d) < 100:
        return np.ones(P.shape[1])
    Xc, yc = X - X.mean(0), yy - yy.mean()
    A = np.vstack([Xc, np.sqrt(alpha) * np.eye(X.shape[1])])
    b = np.concatenate([yc, np.zeros(X.shape[1])])
    w, _ = nnls(A, b)
    return np.ones(P.shape[1]) if w.sum() <= 0 else w / w.mean()


def select_weights(sf: ScoreFrame, cfg, lab: pd.DataFrame, dev_index: pd.DatetimeIndex, ev: dict) -> dict:
    """Fitted pillar weights are used only if they beat equal weights out-of-sample (walk-forward) by a margin."""
    P = sf.pillars.reindex(dev_index)
    y = lab["fwd_ret"].reindex(dev_index)
    folds = walk_forward_folds(dev_index, int(ev["initial_train_days"]), int(ev["test_window_days"]), int(ev["purge_days"]))
    oos_fit, oos_eq = [], []
    for tr, te in folds:
        w = fit_pillar_weights(P.loc[tr], y.loc[tr], float(ev["ridge_alpha"]))
        Pt = P.loc[te]
        avail = Pt.notna().to_numpy() * w
        oos_fit.append(pd.Series((Pt.fillna(0).to_numpy() * avail).sum(1) / np.where(avail.sum(1) > 0, avail.sum(1), np.nan), te))
        oos_eq.append(sf.composite.reindex(te))
    if not folds:
        return {"selected": "equal_weights", "reason": "not enough history for walk-forward", "folds": 0}
    fit, eq = pd.concat(oos_fit), pd.concat(oos_eq)
    r_fit, _ = _safe_spearman(fit, y.reindex(fit.index))
    r_eq, _ = _safe_spearman(eq, y.reindex(eq.index))
    margin = float(ev["select_margin"])
    use_fit = r_fit is not None and r_fit > 0 and (r_eq is None or r_fit > r_eq + margin)   # must also show a positive OOS signal
    w_final = fit_pillar_weights(P, y.where(y.index <= dev_index[-1 - int(ev["purge_days"])]), float(ev["ridge_alpha"]))
    return {"selected": "fitted_weights" if use_fit else "equal_weights", "folds": len(folds), "oos_rank_corr_fitted": r_fit,
            "oos_rank_corr_equal": r_eq, "required_margin": margin,
            "candidate_weights": dict(zip(P.columns, map(float, w_final))),
            "reason": "fitted beat equal weights out-of-sample by the required margin" if use_fit else
                      "fitted weights did not beat equal weights out-of-sample with a positive signal; rules kept"}


# ------------------------------------------------------------------------------------------------ 6. runner
@dataclass
class EvalResult:
    report: dict
    path_json: Path | None = None
    path_md: Path | None = None


def run_evaluation(pipe: MacroPipeline, lockbox: bool = False, force: bool = False, out_dir: str | Path | None = None,
                   ticker: str | None = None) -> EvalResult:
    cfg, ev = pipe.cfg, pipe.cfg.evaluation
    h = int(ev["label_horizon"])
    kse = pipe.kse100()
    if kse.empty:
        raise RuntimeError("KSE-100 series is required for evaluation")
    lab = make_labels(kse, h)
    df, sf, qf, comp = pipe.score_series(ticker)
    idx = lab.index.intersection(df.index)
    lock_start = pd.Timestamp(ev["lockbox_start"])
    purge = int(ev["purge_days"])
    dev_idx = idx[idx < lock_start]
    dev_idx = dev_idx[: max(0, len(dev_idx) - purge)]                 # purge: dev labels must not reach into the lock-box
    lock_idx = idx[idx >= lock_start]
    regimes = regime_labels(comp.ff.raw, ev["regime_thresholds"])
    strategies = baseline_scores(sf, cfg, comp)
    folds = walk_forward_folds(dev_idx, int(ev["initial_train_days"]), int(ev["test_window_days"]), purge)
    oos_idx = dev_idx[int(ev["initial_train_days"]):] if folds else dev_idx

    def run_block(index: pd.DatetimeIndex) -> dict:
        res = {}
        for name, sc in strategies.items():
            st = qf.state if name == "full_macro_agent" else state_of(sc, cfg)
            cf = qf.confidence if name == "full_macro_agent" else None
            res[name] = metrics(sc.reindex(index), st.reindex(index), None if cf is None else cf.reindex(index),
                                lab.reindex(index), h, regimes, int(ev["confidence_bins"]))
        return res

    report: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(), "model_version": cfg.model_version, "horizon_days": h,
        "ticker": ticker, "data_warning": None,
        "design": {"walk_forward_folds": len(folds), "initial_train_days": ev["initial_train_days"], "test_window_days": ev["test_window_days"],
                   "purge_days": purge, "lockbox_start": str(lock_start.date()), "dev_period": [str(dev_idx.min().date()), str(dev_idx.max().date())] if len(dev_idx) else None,
                   "oos_period": [str(oos_idx.min().date()), str(oos_idx.max().date())] if len(oos_idx) else None},
        "walk_forward_oos": run_block(oos_idx),
        "weight_selection": select_weights(sf, cfg, lab, dev_idx, ev) if len(dev_idx) else {},
    }
    for k, fr in comp.frames.items():
        if fr["status"].astype(str).str.contains("synthetic").any():
            report["data_warning"] = "SYNTHETIC DEMO DATA - results are meaningless and only show the machinery works."
            break
    fm = report["walk_forward_oos"].get("full_macro_agent", {})
    report["honest_summary"] = honest_summary(fm, report["walk_forward_oos"])

    root = Path(out_dir) if out_dir else cfg.root / "reports"
    root.mkdir(parents=True, exist_ok=True)
    marker = cfg.root / "data" / "lockbox_run.json"
    if lockbox:
        if marker.exists() and not force:
            raise RuntimeError(f"The lock-box was already run ({marker}). It is run once and reported as it is; use force=True only to re-report.")
        report["lockbox"] = run_block(lock_idx) if len(lock_idx) else {"note": "no data after lockbox_start"}
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({"run_at": report["generated_at"], "model_version": cfg.model_version}))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pj, pm = root / f"macro_eval_{stamp}.json", root / f"macro_eval_{stamp}.md"
    pj.write_text(json.dumps(report, indent=2, default=str))
    pm.write_text(render_md(report))
    _log_mlflow(report)
    return EvalResult(report, pj, pm)


def honest_summary(full: dict, allres: dict) -> str:
    if not full or full.get("rank_corr_non_overlapping") is None:
        return "Not enough out-of-sample data to assess the return signal."
    rho, p = full["rank_corr_non_overlapping"], full["rank_corr_pvalue_non_overlapping"]
    sig = p is not None and p < 0.05
    base = allres.get("policy_rate_only", {}).get("rank_corr_non_overlapping")
    msg = (f"Out-of-sample rank correlation of the macro score with the forward return is {rho:+.3f} on non-overlapping windows "
           f"(p={p:.2f}). " + ("This is statistically distinguishable from zero at 5%." if sig else
                                "This is not statistically distinguishable from zero; a weak or null return signal is a reportable result."))
    if base is not None:
        msg += f" Policy-rate-only baseline: {base:+.3f}."
    msg += " Value as a risk/context signal is judged separately from the state-conditional drawdown and volatility tables."
    return msg


def render_md(r: dict) -> str:
    L = [f"# Macro agent evaluation ({r['model_version']})", ""]
    if r.get("data_warning"):
        L += [f"> **{r['data_warning']}**", ""]
    L += [f"Horizon: {r['horizon_days']} trading days. Walk-forward folds: {r['design']['walk_forward_folds']}, "
          f"OOS period: {r['design']['oos_period']}, purge: {r['design']['purge_days']} days.", "", "## Summary", r["honest_summary"], "",
          "## Walk-forward out-of-sample", "", "| strategy | n | rank corr (non-overlap) | p | hit rate bull/bear | share non-neutral |", "|---|---|---|---|---|---|"]
    def fmt(x, f="{:+.3f}"): return "n/a" if x is None else f.format(x)
    for k, m in r["walk_forward_oos"].items():
        L.append(f"| {k} | {m.get('n')} | {fmt(m.get('rank_corr_non_overlapping'))} | {fmt(m.get('rank_corr_pvalue_non_overlapping'), '{:.2f}')} | "
                 f"{fmt(m.get('hit_rate_bull_bear'), '{:.2f}')} | {fmt(m.get('share_non_neutral'), '{:.2f}')} |")
    full = r["walk_forward_oos"].get("full_macro_agent", {})
    if full.get("by_state"):
        L += ["", "## Risk usefulness: forward outcomes by state (full agent)", "", "| state | n | mean fwd return | mean fwd vol (ann.) | mean fwd max drawdown |", "|---|---|---|---|---|"]
        for s, m in full["by_state"].items():
            L.append(f"| {s} | {m['n']} | {m['mean_fwd_ret']:+.3%} | {m['mean_fwd_vol']:.1%} | {m['mean_fwd_max_drawdown']:.1%} |")
    if full.get("by_regime"):
        L += ["", "## Regime-wise (evaluation-only labels)", "", "| regime | n | rank corr | directional hit |", "|---|---|---|---|"]
        for s, m in full["by_regime"].items():
            L.append(f"| {s} | {m['n']} | {fmt(m['rank_corr'])} | {m['directional_hit_rate']:.2f} |")
    ws = r.get("weight_selection") or {}
    if ws:
        L += ["", "## Weight selection", f"Selected: **{ws.get('selected')}** - {ws.get('reason')}",
              f"OOS rank corr fitted={fmt(ws.get('oos_rank_corr_fitted'))}, equal={fmt(ws.get('oos_rank_corr_equal'))}, required margin {ws.get('required_margin')}."]
    if "lockbox" in r:
        L += ["", "## Lock-box (run once, reported as is)", "```json", json.dumps(r["lockbox"].get("full_macro_agent", r["lockbox"]), indent=2, default=str), "```"]
    return "\n".join(L) + "\n"


def _log_mlflow(report: dict) -> None:
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        return
    try:
        import mlflow
        with mlflow.start_run(run_name=report["model_version"]):
            mlflow.log_param("model_version", report["model_version"])
            for k, m in report["walk_forward_oos"].items():
                for mk in ("rank_corr_non_overlapping", "hit_rate_bull_bear", "directional_hit_rate"):
                    if m.get(mk) is not None:
                        mlflow.log_metric(f"{k}.{mk}", m[mk])
    except Exception:                                      # noqa: BLE001 - MLflow is optional
        pass
