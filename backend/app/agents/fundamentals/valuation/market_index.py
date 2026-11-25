"""Equal-weighted PSX market index built from the official daily files.

Each day: the median log price change of listed companies that traded on both days
(volume > 0, |change| < 50% to drop corporate-action jumps). Cumulated from the first stored day
(= 1.0). Used to remove the market level from valuation targets, so the model learns how
fundamentals map to value relative to the market, not which year it is.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from app.ingestion import psx_eod


@lru_cache(maxsize=1)
def _wide(n_files: int) -> pd.DataFrame:
    closes = {}
    for sym in psx_eod.symbols():
        df = psx_eod.load_symbol(sym)
        if df is None or len(df) < 20:
            continue
        df = df[df["volume"] > 0]
        closes[sym] = df["close"]
    if not closes:
        return pd.DataFrame(index=pd.DatetimeIndex([]))
    return pd.DataFrame(closes).sort_index()


def wide_closes() -> pd.DataFrame:
    """Traded closes of every listed company (dates x symbols)."""
    return _wide(len(list(psx_eod.EOD_DIR.glob("????-??-??.Z"))))


def market_return(t0, t1) -> float | None:
    """Median log return of all listed companies that traded around both dates."""
    w = wide_closes()
    if w.empty:
        return None
    w = w.ffill(limit=5)
    a, b = w[w.index <= pd.Timestamp(t0)], w[w.index <= pd.Timestamp(t1)]
    if a.empty or b.empty:
        return None
    r = np.log(b.iloc[-1] / a.iloc[-1]).dropna()
    return float(r.median()) if len(r) >= 50 else None


@lru_cache(maxsize=1)
def _index(n_files: int) -> pd.Series:
    wide = _wide(n_files)
    rets = np.log(wide).diff()
    rets = rets.where(rets.abs() < np.log(1.5))
    daily = rets.median(axis=1, skipna=True).fillna(0.0)
    return np.exp(daily.cumsum()).rename("psx_ew_index")


def market_index() -> pd.Series:
    """Index level by trading date (cached until the number of daily files changes)."""
    return _index(len(list(psx_eod.EOD_DIR.glob("????-??-??.Z"))))


def level_on(d) -> float | None:
    idx = market_index()
    idx = idx[idx.index <= pd.Timestamp(d)]
    return None if idx.empty else float(idx.iloc[-1])
