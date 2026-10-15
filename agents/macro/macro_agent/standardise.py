"""Layer 3 - standardise and orient.
Expanding robust z-score (median, 1.4826 x MAD) using only data up to t, >= 36 month-end observations;
clip to [-3, 3], divide by 3 -> [-1, +1]; orient so + means bullish. Short or annual series fall back to a rank
percentile s = 2*pct - 1 (or a fixed rule) and are marked low-weight.
Statistics are refreshed at each calendar month end (history = month-end values <= t), which keeps carried-forward
step series from being over-counted and makes the 36-month rule literal."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Config, FeatureCfg


@dataclass
class Standardised:
    score: pd.DataFrame      # oriented, [-1, 1], NaN if unavailable
    weight: pd.DataFrame     # effective weight, NaN if unavailable
    method: pd.DataFrame     # 'robust_z' | 'percentile' | 'fixed' | '' (object)
    z: pd.DataFrame          # clipped robust z (unoriented), NaN if method != robust_z


def _samples(x: pd.Series, method: str) -> pd.Series:
    x = x.dropna()
    if method == "percentile":                       # distinct releases (annual series)
        return x[x != x.shift()]
    return x[x.index.is_month_end]                   # month-end history


def _expanding_stats(vals: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(vals)
    med, mad, mean_ad = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    for i in range(n):
        h = vals[: i + 1]
        m = np.median(h)
        med[i] = m
        mad[i] = scale * np.median(np.abs(h - m))
        mean_ad[i] = 1.2533 * np.mean(np.abs(h - m))   # fallback scale when MAD == 0
    return med, mad, mean_ad


def _standardise_one(x: pd.Series, f: FeatureCfg, cal: pd.DatetimeIndex, sc: dict) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    clip, scale = sc["clip"], sc["mad_scale"]
    min_m, fb_min, pct_min = int(sc["min_months"]), int(sc["fallback_min_obs"]), int(sc["percentile_min_obs"])
    s = _samples(x, f.method)
    n = len(cal)
    score, method, zc = np.full(n, np.nan), np.full(n, "", dtype=object), np.full(n, np.nan)
    if s.empty:
        idx = cal
        return (pd.Series(score, idx), pd.Series(method, idx), pd.Series(zc, idx), pd.Series(np.nan, idx))
    sv = s.to_numpy(dtype=float)
    med, mad, mad2 = _expanding_stats(sv, scale)
    ptr = np.searchsorted(s.index.to_numpy(), cal.to_numpy(), side="right")   # samples <= t
    xv = x.to_numpy(dtype=float)
    sorted_cache: dict[int, np.ndarray] = {}
    for d in range(n):
        p, xt = ptr[d], xv[d]
        if p == 0 or np.isnan(xt):
            continue
        k = p - 1
        use_z = f.method == "robust_z" and p >= min_m
        if use_z:
            denom = mad[k] if mad[k] > 0 else mad2[k]
            if denom <= 0:                       # history is constant: any change is extreme, no change is zero
                z = 0.0 if xt == med[k] else float(np.sign(xt - med[k]) * clip)
            else:
                z = float(np.clip((xt - med[k]) / denom, -clip, clip))
            score[d], method[d], zc[d] = z / clip, "robust_z", z
            continue
        need = fb_min if f.method == "robust_z" else pct_min
        if p >= need:
            h = sorted_cache.get(p)
            if h is None:
                h = np.sort(sv[:p])
                sorted_cache = {p: h}
            pct = (np.searchsorted(h, xt, "left") + 0.5 * (np.searchsorted(h, xt, "right") - np.searchsorted(h, xt, "left"))) / p
            score[d], method[d] = 2 * pct - 1, "percentile"
        elif f.fixed_rule:
            score[d] = float(np.clip((xt - f.fixed_rule["neutral"]) / f.fixed_rule["scale"], -1, 1))
            method[d] = "fixed"
    idx = cal
    return pd.Series(score, idx), pd.Series(method, idx), pd.Series(zc, idx), pd.Series(np.nan, idx)


def standardise_features(raw: pd.DataFrame, cfg: Config) -> Standardised:
    sc, cal = cfg.standardise, raw.index
    S, W, M, Z = {}, {}, {}, {}
    low = sc["low_weight"]
    for name, f in cfg.active_features().items():
        s, m, z, _ = _standardise_one(raw[name], f, cal, sc)
        S[name] = s * f.orient
        M[name] = m
        Z[name] = z
        w = np.where(m == "robust_z", f.weight, np.minimum(f.weight, low))
        W[name] = pd.Series(np.where(s.notna(), w, np.nan), cal)
    return Standardised(pd.DataFrame(S), pd.DataFrame(W), pd.DataFrame(M), pd.DataFrame(Z))
