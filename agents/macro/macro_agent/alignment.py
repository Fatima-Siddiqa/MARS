"""Layer 1.4/1.5 - release-date alignment (as-of join) and as-of snapshot.
x_s(t) = value of the latest observation with release_date <= t. Each carried value records its age."""
from __future__ import annotations

import pandas as pd

from .config import SeriesCfg


def with_release_dates(df: pd.DataFrame, cfg: SeriesCfg) -> pd.DataFrame:
    out = df.copy()
    out["release_date"] = out["obs_date"] + pd.Timedelta(days=cfg.release_lag_days)
    return out


def truncate_as_of(df: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Drop everything not yet released on `as_of` (the hard no-look-ahead guard)."""
    return df[df["release_date"] <= as_of].copy()


def build_calendar(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.date_range(pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize(), freq="D").astype("datetime64[ns]")


def asof_join(obs: pd.DataFrame, cal: pd.DatetimeIndex) -> pd.DataFrame:
    """obs needs release_date, obs_date, value. Returns frame indexed by cal with value, obs_date, release_date, age_days.
    NaN before the first release."""
    left = pd.DataFrame({"t": pd.DatetimeIndex(cal).astype("datetime64[ns]")})
    if obs.empty:
        return pd.DataFrame({"value": float("nan"), "obs_date": pd.NaT, "release_date": pd.NaT, "age_days": float("nan")},
                            index=cal)
    right = (obs.sort_values(["release_date", "obs_date"]).drop_duplicates("release_date", keep="last")
             [["release_date", "obs_date", "value"]].copy())
    right["release_date"] = right["release_date"].astype("datetime64[ns]")
    right["obs_date"] = right["obs_date"].astype("datetime64[ns]")
    m = pd.merge_asof(left, right, left_on="t", right_on="release_date", direction="backward")
    m["age_days"] = (m["t"] - m["release_date"]).dt.days.astype(float)
    return m.set_index(pd.DatetimeIndex(cal))[["value", "obs_date", "release_date", "age_days"]]


def snapshot(frames: dict[str, pd.DataFrame], as_of: pd.Timestamp) -> dict[str, pd.Series | None]:
    """Latest released observation per series on a given date (used for evidence)."""
    out: dict[str, pd.Series | None] = {}
    for k, df in frames.items():
        d = df[df["release_date"] <= as_of]
        out[k] = None if d.empty else d.sort_values(["release_date", "obs_date"]).iloc[-1]
    return out
