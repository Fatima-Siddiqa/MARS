"""PSX-wide training set for the valuation model (official PSX data only).

One sample per company per published fiscal year:
- features: that year's official figures from the PSX company page (revenue / total income,
  profit after tax), growth vs the previous year, margin, bank flag, PSX sector;
- observation date: the first trading day AFTER the figures were published on PSX;
- target: log(market cap on that day) - market median log P/E on that day
  (valuation relative to the market's typical multiple; see market_level.py).

Market cap = official PSX close x shares, with shares = profit after tax / reported EPS for that
year (PSX shows only today's share count; EPS was reported on the shares of that year).
No price-derived feature is used, so the model cannot copy the answer.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from app.agents.fundamentals.data.psx_official import (
    REFERENCE_CSV,
    SHARES_CSV,
    STATEMENTS_CSV,
    UNIVERSE_CSV,
)
from app.agents.fundamentals.valuation.market_level import level_cached
from app.core.config import REPO_ROOT
from app.ingestion import psx_eod

log = logging.getLogger(__name__)
PROCESSED = REPO_ROOT / "data" / "processed" / "fundamentals"
TRAINING_CSV = PROCESSED / "training_set.csv"

NUMERIC_FEATURES = [
    "log_revenue",
    "log_profit",
    "log_loss",
    "is_loss",
    "net_margin",
    "revenue_growth",
    "profit_growth",
    "is_bank",
]
CATEGORICAL_FEATURES = ["sector_code"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "log_rel_mcap"
MIN_ABS_EPS = 0.05  # smaller EPS values are too rounded to back out a share count


def company_table() -> pd.DataFrame:
    """ticker -> company_type, sector_code for the reference set and the whole-market universe."""
    ref = pd.read_csv(REFERENCE_CSV, dtype={"psx_sector_code": str})
    rows = [ref[["ticker", "company_type", "psx_sector_code"]]]
    if UNIVERSE_CSV.exists():
        uni = pd.read_csv(UNIVERSE_CSV)
        sectors = {}
        for t in uni["ticker"]:
            df = psx_eod.load_symbol(t)
            if df is not None and "sector_code" in df:
                sectors[t] = str(int(df["sector_code"].iloc[-1])).zfill(4)
        uni["psx_sector_code"] = uni["ticker"].map(sectors)
        rows.append(uni[["ticker", "company_type", "psx_sector_code"]])
    return pd.concat(rows).dropna().drop_duplicates("ticker").set_index("ticker")


def _close_after(ticker: str, published) -> tuple[pd.Timestamp, float] | None:
    df = psx_eod.load_symbol(ticker)
    if df is None:
        return None
    after = df[(df.index > pd.Timestamp(published)) & (df["volume"] > 0)]
    if after.empty or (after.index[0] - pd.Timestamp(published)).days > 10:
        return None
    return after.index[0], float(after["close"].iloc[0])


def features_from(revenue, profit, prev_revenue, prev_profit, is_bank, sector_code) -> dict:
    """Feature row from 12-month figures in PKR millions (shared by training and inference)."""
    rev = revenue if revenue is not None and revenue > 0 else None
    margin = None if rev is None or profit is None else float(np.clip(profit / rev, -1, 1))
    rg = None
    if rev is not None and prev_revenue and prev_revenue > 0:
        rg = float(np.clip(rev / prev_revenue - 1, -1, 3))
    pg = None
    if profit is not None and prev_profit:
        pg = float(np.clip((profit - prev_profit) / abs(prev_profit), -3, 3))
    return {
        "log_revenue": np.log(rev) if rev else np.nan,
        "log_profit": np.log1p(max(profit, 0)) if profit is not None else np.nan,
        "log_loss": np.log1p(max(-profit, 0)) if profit is not None else np.nan,
        "is_loss": float(profit < 0) if profit is not None else np.nan,
        "net_margin": margin if margin is not None else np.nan,
        "revenue_growth": rg if rg is not None else np.nan,
        "profit_growth": pg if pg is not None else np.nan,
        "is_bank": float(is_bank),
        "sector_code": sector_code,
    }


def build_training_set(save: bool = True) -> pd.DataFrame:
    st = pd.read_csv(STATEMENTS_CSV)
    st = (
        st[st["period_kind"] == "annual"]
        .sort_values("version")
        .drop_duplicates(["ticker", "line_item", "period_end"], keep="last")
    )
    comp = company_table()
    listed = (
        pd.read_csv(SHARES_CSV).sort_values("as_of").groupby("ticker")["shares_outstanding"].last()
    )
    rows, dropped = [], {}

    def drop(reason):
        dropped[reason] = dropped.get(reason, 0) + 1

    for ticker, g in st.groupby("ticker"):
        if ticker not in comp.index:
            drop("no sector / type")
            continue
        ctype, sector = comp.loc[ticker, "company_type"], comp.loc[ticker, "psx_sector_code"]
        is_bank = ctype == "bank"
        wide = g.pivot_table(
            index="period_end", columns="line_item", values="value", aggfunc="last"
        )
        pub = g.groupby("period_end")["published_at"].max()
        ends = sorted(wide.index)
        for i, end in enumerate(ends):
            r = wide.loc[end]
            rev_item = "total_income" if is_bank else "revenue"
            rev, ni, eps = r.get(rev_item), r.get("net_income"), r.get("eps_reported")
            if pd.isna(ni) or pd.isna(eps) or abs(eps) < MIN_ABS_EPS or ni == 0:
                drop("missing profit or EPS")
                continue
            shares = ni * 1000 / eps  # profit in Rs '000
            if shares <= 0:
                drop("profit and EPS of opposite sign")
                continue
            if ticker in listed.index and not 0.1 < shares / listed[ticker] < 10:
                drop("implied shares implausible vs PSX count")
                continue
            obs = _close_after(ticker, pub[end])
            if obs is None:
                drop("no traded price within 10 days after publication")
                continue
            day, close = obs
            level = level_cached(day)
            if level is None:
                drop("market P/E level not available on that date")
                continue
            prev = wide.loc[ends[i - 1]] if i > 0 else None
            if prev is not None and (pd.Timestamp(end) - pd.Timestamp(ends[i - 1])).days > 400:
                prev = None
            to_m = 1e-3  # Rs '000 -> PKR millions
            feats = features_from(
                None if pd.isna(rev) else rev * to_m,
                ni * to_m,
                None if prev is None or pd.isna(prev.get(rev_item)) else prev.get(rev_item) * to_m,
                None
                if prev is None or pd.isna(prev.get("net_income"))
                else prev.get("net_income") * to_m,
                is_bank,
                sector,
            )
            mcap = close * shares / 1e6
            rows.append(
                {
                    "ticker": ticker,
                    "period_end": end,
                    "published_at": pub[end],
                    "obs_date": day.date().isoformat(),
                    "close": close,
                    "shares": shares,
                    "market_cap": mcap,
                    "market_log_pe": level,
                    **feats,
                    TARGET: np.log(mcap) - level,
                }
            )
    ds = pd.DataFrame(rows)
    if not ds.empty:
        ds["obs_year"] = pd.to_datetime(ds["obs_date"]).dt.year
    if save:
        PROCESSED.mkdir(parents=True, exist_ok=True)
        ds.to_csv(TRAINING_CSV, index=False)
    log.info(
        "training set: %d rows, %d companies; dropped %s",
        len(ds),
        ds["ticker"].nunique() if len(ds) else 0,
        dropped,
    )
    ds.attrs["dropped"] = dropped
    return ds
