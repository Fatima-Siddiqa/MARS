"""Layer 2 - feature engineering. Everything is computed from the as-of panel or from release-ordered observations,
so a feature at date t only uses information released on or before t."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .alignment import asof_join
from .config import Config


@dataclass
class FeatureFrame:
    raw: pd.DataFrame        # cal x feature : raw feature value
    age: pd.DataFrame        # cal x feature : age in days of the oldest input
    panel: dict[str, pd.DataFrame]   # series key -> as-of panel (value, obs_date, release_date, age_days)


def _monthly_complete(df: pd.DataFrame) -> pd.DataFrame:
    """Reindex a month-end series to every month so rolling windows never straddle a missing month."""
    s = df.set_index("obs_date")["value"]
    rel = df.set_index("obs_date")["release_date"]
    idx = pd.date_range(s.index.min(), s.index.max(), freq="ME")
    return pd.DataFrame({"value": s.reindex(idx), "release_date": rel.reindex(idx)})


def _derived(obs: pd.DataFrame, values: pd.Series, cal: pd.DatetimeIndex) -> pd.DataFrame:
    """As-of join a derived observation-level series; carries the underlying release date and age."""
    d = pd.DataFrame({"obs_date": obs["obs_date"].to_numpy(), "release_date": obs["release_date"].to_numpy(),
                      "value": values.to_numpy()}).dropna(subset=["value", "release_date"])
    return asof_join(d, cal)


def _pct_rank_last(w: np.ndarray) -> float:
    last, hist = w[-1], w[:-1]
    return float(((hist < last).sum() + 0.5 * (hist == last).sum()) / len(hist))


def build_features(frames: dict[str, pd.DataFrame], cal: pd.DatetimeIndex, cfg: Config) -> FeatureFrame:
    """frames: series key -> clean obs with release_date. Missing series simply yield NaN features."""
    P = {k: asof_join(df, cal) for k, df in frames.items()}
    nan = pd.Series(np.nan, index=cal)
    age_nan = pd.Series(np.nan, index=cal)

    def v(k):
        return P[k]["value"] if k in P else nan

    def a(k):
        return P[k]["age_days"] if k in P else age_nan

    raw, age = {}, {}
    fp = cfg.feature_params

    # --- monetary
    pr = v("policy_rate")
    raw["rate_change_6m"] = pr - pr.shift(182)
    age["rate_change_6m"] = a("policy_rate")
    raw["rate_vs_12m_avg"] = pr - pr.rolling(365, min_periods=365).mean()
    age["rate_vs_12m_avg"] = a("policy_rate")
    raw["yield_spread"] = v("tbill_3m") - pr
    age["yield_spread"] = pd.concat([a("tbill_3m"), a("policy_rate")], axis=1).max(axis=1, skipna=False)

    # --- inflation
    raw["cpi_level"], age["cpi_level"] = v("cpi_yoy"), a("cpi_yoy")
    if "cpi_yoy" in frames and len(frames["cpi_yoy"]) >= 7:
        m = _monthly_complete(frames["cpi_yoy"])
        m3 = m["value"].rolling(3, min_periods=3).mean()
        trend = m3 - m3.shift(3)
        j = _derived(m.reset_index(names="obs_date").dropna(subset=["release_date"]),
                     trend.loc[m["release_date"].notna()], cal)
        raw["cpi_trend"], age["cpi_trend"] = j["value"], j["age_days"]
    else:
        raw["cpi_trend"], age["cpi_trend"] = nan, age_nan

    # --- external
    p = v("pkr_usd")
    raw["pkr_dep_30d"] = (p / p.shift(30) - 1) * 100
    raw["pkr_dep_90d"] = (p / p.shift(90) - 1) * 100
    for f in ("pkr_dep_30d", "pkr_dep_90d"):
        age[f] = a("pkr_usd")
    if "pkr_usd" in frames and len(frames["pkr_usd"]) > 40:
        o = frames["pkr_usd"].sort_values("obs_date").reset_index(drop=True)
        ret = o["value"].pct_change() * 100
        vol = ret.rolling(30, min_periods=20).std()
        j = _derived(o, vol, cal)
        raw["pkr_vol_30d"], age["pkr_vol_30d"] = j["value"], j["age_days"]
    else:
        raw["pkr_vol_30d"], age["pkr_vol_30d"] = nan, age_nan

    r = v("fx_reserves")
    raw["reserves_chg_3m"] = (r / r.shift(91) - 1) * 100
    age["reserves_chg_3m"] = a("fx_reserves")

    if "current_account" in frames and len(frames["current_account"]) >= 3:
        m = _monthly_complete(frames["current_account"])
        s3 = m["value"].rolling(3, min_periods=3).sum()
        ok = m["release_date"].notna()
        j = _derived(m.reset_index(names="obs_date")[ok.to_numpy()], s3[ok], cal)
        raw["ca_3m_sum"], age["ca_3m_sum"] = j["value"], j["age_days"]
    else:
        raw["ca_3m_sum"], age["ca_3m_sum"] = nan, age_nan

    if "remittances" in frames and len(frames["remittances"]) >= 13:
        m = _monthly_complete(frames["remittances"])
        yoy = (m["value"] / m["value"].shift(12) - 1) * 100
        ok = m["release_date"].notna()
        j = _derived(m.reset_index(names="obs_date")[ok.to_numpy()], yoy[ok], cal)
        raw["remittances_yoy"], age["remittances_yoy"] = j["value"], j["age_days"]
    else:
        raw["remittances_yoy"], age["remittances_yoy"] = nan, age_nan

    # --- commodity
    b = v("brent").where(lambda s: s > 0)
    raw["brent_3m"] = (b / b.shift(91) - 1) * 100
    age["brent_3m"] = a("brent")
    win = int(fp.get("brent_percentile_window", 252))
    if "brent" in frames and len(frames["brent"]) > win // 2:
        o = frames["brent"].sort_values("obs_date").reset_index(drop=True)
        pct = o["value"].rolling(win, min_periods=win // 2).apply(_pct_rank_last, raw=True)
        j = _derived(o, pct, cal)
        raw["brent_pct_252"], age["brent_pct_252"] = j["value"], j["age_days"]
    else:
        raw["brent_pct_252"], age["brent_pct_252"] = nan, age_nan
    g = v("gold").where(lambda s: s > 0)
    raw["gold_3m"] = (g / g.shift(91) - 1) * 100
    age["gold_3m"] = a("gold")

    # --- growth
    raw["gdp_growth"], age["gdp_growth"] = v("gdp_growth"), a("gdp_growth")

    names = list(cfg.features)
    return FeatureFrame(raw=pd.DataFrame(raw)[names], age=pd.DataFrame(age)[names], panel=P)
