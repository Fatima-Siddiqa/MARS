"""Market valuation level on a date: the median trailing P/E of listed PSX companies.

For date d and every company with an annual profit published on PSX in the 15 months before d
(positive profit, share count = profit / reported EPS of that year) and a traded price around d:
    log P/E = log(price on d x shares / profit)
The level is the median over companies. Only information public on d is used.

The valuation model predicts log(market cap) MINUS this level (= the company's valuation relative
to the market's typical multiple on that date). Normalising by the market's own multiple removes
year-to-year re-rating (PSX large caps re-rated sharply in 2024-2026), which a price index cannot.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from app.agents.fundamentals.data.psx_official import STATEMENTS_CSV
from app.agents.fundamentals.valuation.market_index import wide_closes

MIN_COMPANIES = 50
LOOKBACK_DAYS = 456  # ~15 months: the latest annual result should be published within this


@lru_cache(maxsize=1)
def _annual(mtime: float) -> pd.DataFrame:
    s = pd.read_csv(STATEMENTS_CSV)
    s = (
        s[s["period_kind"] == "annual"]
        .sort_values("version")
        .drop_duplicates(["ticker", "line_item", "period_end"], keep="last")
    )
    w = s.pivot_table(
        index=["ticker", "period_end"], columns="line_item", values="value", aggfunc="last"
    ).reset_index()
    pub = s.groupby(["ticker", "period_end"])["published_at"].max().reset_index()
    w = w.merge(pub, on=["ticker", "period_end"])
    w = w[(w["net_income"] > 0) & (w["eps_reported"].abs() >= 0.05)]
    w["shares"] = w["net_income"] * 1000 / w["eps_reported"]
    w = w[w["shares"] > 0]
    w["profit_m"] = w["net_income"] / 1000
    w["published_at"] = pd.to_datetime(w["published_at"])
    return w[["ticker", "published_at", "shares", "profit_m"]]


def annual_profits() -> pd.DataFrame:
    return _annual(STATEMENTS_CSV.stat().st_mtime if STATEMENTS_CSV.exists() else 0.0)


def level_on(d) -> float | None:
    """Median log P/E of listed companies on date d (None if too few companies)."""
    d = pd.Timestamp(d)
    a = annual_profits()
    a = a[(a["published_at"] <= d) & (a["published_at"] > d - pd.Timedelta(days=LOOKBACK_DAYS))]
    a = a.sort_values("published_at").drop_duplicates("ticker", keep="last").set_index("ticker")
    w = wide_closes()
    if w.empty:
        return None
    w = w[(w.index <= d) & (w.index > d - pd.Timedelta(days=10))]
    if w.empty:
        return None
    px = w.ffill().iloc[-1]
    common = a.index.intersection(px.dropna().index)
    if len(common) < MIN_COMPANIES:
        return None
    pe = np.log(px[common] * a.loc[common, "shares"] / 1e6 / a.loc[common, "profit_m"])
    return float(pe.median())


@lru_cache(maxsize=4096)
def _cached(d: str, key: float) -> float | None:
    return level_on(d)


def level_cached(d) -> float | None:
    key = STATEMENTS_CSV.stat().st_mtime if STATEMENTS_CSV.exists() else 0.0
    return _cached(pd.Timestamp(d).date().isoformat(), key)
