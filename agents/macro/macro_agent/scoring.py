"""Layer 4 - pillar scores, market composite, state, sector sensitivity. Vectorised over dates so the live
path and the backtest share the same code (Pipeline Runner)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Config
from .standardise import Standardised


@dataclass
class ScoreFrame:
    adj_score: pd.DataFrame      # sector-adjusted oriented feature scores (NaN = not used)
    weight: pd.DataFrame         # effective weights where used
    pillars: pd.DataFrame        # cal x pillars
    composite: pd.Series
    state: pd.Series
    contrib: pd.DataFrame        # contribution of each feature to the composite (sums to composite)
    n_pillars: pd.Series


def apply_sector(score: pd.DataFrame, weight: pd.DataFrame, multipliers: dict[str, float]):
    """m[sector, feature] in {+1,-1,0} or a scale in between; multiplies the feature score before pillars are averaged.
    A multiplier of 0 removes the feature from the average (and from coverage)."""
    s, w = score.copy(), weight.copy()
    for f, m in multipliers.items():
        if f in s.columns:
            if m == 0:
                s[f], w[f] = np.nan, np.nan
            else:
                s[f] = s[f] * m
    return s, w


def state_of(score: pd.Series, cfg: Config) -> pd.Series:
    hi, lo = cfg.state["bullish"], cfg.state["bearish"]
    st = pd.Series("Neutral", index=score.index, dtype=object)
    st[score >= hi] = "Bullish"
    st[score <= lo] = "Bearish"
    st[score.isna()] = "Neutral"
    return st


def pillar_weights(cfg: Config) -> dict[str, float]:
    return {p: float(cfg.pillar_weights.get(p, 1.0)) for p in cfg.pillars}


def score_features(std: Standardised, cfg: Config, multipliers: dict[str, float] | None = None,
                   pw: dict[str, float] | None = None) -> ScoreFrame:
    s, w = apply_sector(std.score, std.weight, multipliers or {})
    feats = cfg.active_features()
    pw = pw or pillar_weights(cfg)
    pillars, den = {}, {}
    for p in cfg.pillars:
        cols = [f for f, c in feats.items() if c.pillar == p and f in s.columns]
        if not cols:
            pillars[p] = pd.Series(np.nan, index=s.index)
            den[p] = pd.Series(0.0, index=s.index)
            continue
        ws = w[cols].where(s[cols].notna(), 0.0).fillna(0.0)
        d = ws.sum(axis=1)
        num = (s[cols].fillna(0.0) * ws).sum(axis=1)
        pillars[p] = (num / d).where(d > 0)
        den[p] = d
    P = pd.DataFrame(pillars)
    avail = P.notna()
    pwv = pd.DataFrame({p: avail[p] * pw[p] for p in cfg.pillars})
    pw_sum = pwv.sum(axis=1)
    comp = (P.fillna(0.0) * pwv).sum(axis=1) / pw_sum.where(pw_sum > 0)
    contrib = pd.DataFrame(index=s.index, columns=list(s.columns), dtype=float)
    for f in s.columns:
        c = feats[f]
        share = (pwv[c.pillar] / pw_sum.where(pw_sum > 0)) * (w[f].where(s[f].notna(), 0.0).fillna(0.0) / den[c.pillar].where(den[c.pillar] > 0))
        contrib[f] = (share * s[f]).where(s[f].notna())
    return ScoreFrame(adj_score=s, weight=w, pillars=P, composite=comp, state=state_of(comp, cfg),
                      contrib=contrib, n_pillars=avail.sum(axis=1))
