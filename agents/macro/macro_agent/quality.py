"""Layer 5 - quality, confidence and gates.
Confidence = Coverage x Freshness x Agreement. Gates: fewer than 3 pillars, or a missing critical group
(policy rate, PKR/USD, CPI) -> state forced Neutral with low confidence."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Config
from .features import FeatureFrame
from .scoring import ScoreFrame
from .standardise import Standardised


@dataclass
class QualityFrame:
    coverage: pd.Series
    freshness: pd.Series
    agreement: pd.Series
    confidence: pd.Series        # after gates
    raw_confidence: pd.Series    # before gates
    gate_pillars: pd.Series      # bool: fewer than min_pillars
    gate_critical: pd.DataFrame  # bool per critical group: missing
    gated: pd.Series             # any gate fired
    state: pd.Series             # after gates
    series_factor: pd.DataFrame  # per-series freshness factor (cal x series)


def series_limit(cfg: Config, key: str) -> float:
    s = cfg.series[key]
    return float(s.expected_interval_days + s.tolerance_days)


def freshness_factor(age: pd.Series, limit: float, levels: dict[str, float]) -> pd.Series:
    f = pd.Series(np.nan, index=age.index)
    f[age <= limit] = levels["fresh"]
    f[(age > limit) & (age <= 2 * limit)] = levels["aging"]
    f[age > 2 * limit] = levels["stale"]
    return f


def assess(ff: FeatureFrame, std: Standardised, sf: ScoreFrame, cfg: Config,
           multipliers: dict[str, float] | None = None) -> QualityFrame:
    q, mult = cfg.quality, multipliers or {}
    feats = {k: f for k, f in cfg.active_features().items() if mult.get(k, 1.0) != 0}
    idx = sf.composite.index
    nominal = sum(f.weight for f in feats.values())
    used = sf.adj_score[list(feats)].notna()
    wsum = sf.weight[list(feats)].where(used, 0.0).fillna(0.0).sum(axis=1)
    coverage = (wsum / nominal).clip(0, 1)

    levels = q["freshness_levels"]
    sfac = pd.DataFrame({k: freshness_factor(p["age_days"], series_limit(cfg, k), levels)
                         for k, p in ff.panel.items() if k in cfg.series})
    fac = pd.DataFrame(index=idx, columns=list(feats), dtype=float)
    for name, f in feats.items():
        cols = [s for s in f.series if s in sfac.columns]
        fac[name] = sfac[cols].min(axis=1, skipna=False) if len(cols) == len(f.series) else np.nan
    freshness = fac.where(used).mean(axis=1).fillna(0.0)

    P = sf.pillars
    neg = float(q["agreement_negligible"])
    big = P.abs() >= neg
    sign_s = np.sign(sf.composite)
    match = (np.sign(P).mul(sign_s, axis=0) > 0) & big
    n_big = big.sum(axis=1)
    agreement = (match.sum(axis=1) / n_big.where(n_big > 0)).fillna(1.0)

    raw_conf = (coverage * freshness * agreement).clip(0, 1)
    gate_p = sf.n_pillars < int(q["min_pillars"])
    crit = pd.DataFrame({g: std.score[[c for c in cols if c in std.score.columns]].isna().all(axis=1)
                         for g, cols in q["critical_groups"].items()})
    gated = gate_p | crit.any(axis=1)
    conf = raw_conf.where(~gated, raw_conf.clip(upper=float(q["low_confidence_cap"])))
    state = sf.state.where(~gated, "Neutral")
    return QualityFrame(coverage, freshness, agreement, conf, raw_conf, gate_p, crit, gated, state, sfac)
