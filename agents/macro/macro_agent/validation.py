"""Layer 1.3 - validate and clean. Every check writes to the data-quality log. Outliers and jumps are flagged,
never deleted. Rows are excluded from the *clean* frame only by documented rules (implausible value, or before the
configured use_from date after a definition change); macro_raw always keeps the original."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SeriesCfg


def _issue(key, check, severity, detail, obs_date=None) -> dict:
    d = None if obs_date is None else pd.Timestamp(obs_date).strftime("%Y-%m-%d")
    return dict(series_key=key, obs_date=d, check=check, severity=severity, detail=detail)


def validate_series(df: pd.DataFrame, cfg: SeriesCfg, jump_k: float = 10.0) -> tuple[pd.DataFrame, list[dict]]:
    """df: obs_date, value, status. Returns (clean, issues). clean has an extra bool column break_flag."""
    key, issues = cfg.key, []
    df = df.sort_values("obs_date").reset_index(drop=True)

    dup = df["obs_date"].duplicated(keep=False)
    if dup.any():
        for d in df.loc[dup, "obs_date"].unique():
            issues.append(_issue(key, "duplicate", "warning", "duplicate observation date; last kept", d))
        df = df.drop_duplicates("obs_date", keep="last").reset_index(drop=True)

    lo, hi = cfg.plausible
    bad = ~df["value"].between(lo, hi)
    for r in df[bad].itertuples():
        issues.append(_issue(key, "range", "error", f"value {r.value} outside plausible [{lo}, {hi}] {cfg.unit}; excluded", r.obs_date))
    df = df[~bad].reset_index(drop=True)

    if cfg.frequency == "daily":
        wk = df["obs_date"].dt.dayofweek >= 5
        for r in df[wk].itertuples():
            issues.append(_issue(key, "non_trading_day", "info", "observation on a weekend", r.obs_date))

    if cfg.frequency != "decision" and len(df) > 2:
        gap_days = df["obs_date"].diff().dt.days
        limit = cfg.expected_interval_days + cfg.tolerance_days
        for i in np.flatnonzero(gap_days.to_numpy() > limit):
            issues.append(_issue(key, "gap", "warning",
                                 f"{int(gap_days.iloc[i])} days since previous observation (limit {limit})", df["obs_date"].iloc[i]))

    if cfg.frequency not in ("decision", "annual") and len(df) > 30:
        d = df["value"].diff().dropna()
        med = d.median()
        mad = 1.4826 * (d - med).abs().median()
        if mad > 0:
            for idx in d.index[((d - med).abs() > jump_k * mad).to_numpy()]:
                issues.append(_issue(key, "jump", "warning",
                                     f"change {d[idx]:.4g} is > {jump_k:g} robust SD; flagged, not removed", df["obs_date"][idx]))

    if cfg.use_from:
        cut = pd.Timestamp(cfg.use_from)
        n = int((df["obs_date"] < cut).sum())
        if n:
            issues.append(_issue(key, "structural_break", "info",
                                 f"{n} rows before {cfg.use_from} excluded (series definition changed); original kept in macro_raw"))
        df = df[df["obs_date"] >= cut].reset_index(drop=True)

    df["break_flag"] = False
    for b in cfg.breaks:
        bt = pd.Timestamp(b)
        after = df.index[df["obs_date"] >= bt]
        if len(after) and df["obs_date"].iloc[0] < bt:
            df.loc[after[0], "break_flag"] = True
            issues.append(_issue(key, "structural_break", "info", f"definition/base-year break at {b}; flagged", df["obs_date"].iloc[after[0]]))

    if df.empty:
        issues.append(_issue(key, "empty", "error", "no usable observations after cleaning"))
    return df, issues
