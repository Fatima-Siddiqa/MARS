"""Point-in-time data layer: what was publicly known about a company on a given date.

Combines the official sources (see docs/fundamentals_data.md):
- keyed filing figures (keyed_statements.csv): balance sheets + year-to-date income lines,
- PSX company-page figures (psx_statements.csv): annual + discrete quarterly income lines,
- official PSX daily closes (app.ingestion.psx_eod), unadjusted, as needed for market cap,
- shares implied by the filed share capital (or PSX listed shares as a fallback).

Rules:
- Only facts with `published_at <= as_of` exist; of several versions, the latest published by
  then is used (restatements never leak backwards).
- For the same fact, a filed report (keyed) beats the PSX company page.
- All money is converted to PKR millions; the original unit stays in the source data.
- Anything that cannot be computed is None with a reason in `warnings`; nothing defaults to 0.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache

import pandas as pd

from app.agents.fundamentals.config import (
    FLOW_ITEMS,
    MAX_PRICE_AGE_DAYS,
    STALE_STATEMENT_DAYS,
    STOCK_ITEMS,
    UNIT_TO_MILLIONS,
)
from app.agents.fundamentals.data.psx_official import (
    OFFICIAL_DIR,
    REFERENCE_CSV,
    SHARES_CSV,
    STATEMENTS_CSV,
)
from app.ingestion import psx_eod

KEYED_CSV = OFFICIAL_DIR / "keyed_statements.csv"
DOC_URL = "https://dps.psx.com.pk/download/document/{doc_id}.pdf#page={page}"


@dataclass
class Fact:
    value: float  # PKR millions (EPS: PKR per share)
    period_start: date | None
    period_end: date
    published_at: date
    source: str
    source_ref: str


@dataclass
class Snapshot:
    ticker: str
    as_of: date
    company_type: str
    sector: str
    price: float | None = None
    price_date: date | None = None
    shares: float | None = None
    shares_source: str | None = None
    market_cap: float | None = None  # PKR millions
    balance: dict[str, Fact] = field(default_factory=dict)  # latest balance sheet items
    balance_date: date | None = None
    ttm: dict[str, Fact] = field(default_factory=dict)  # trailing-twelve-month flows
    ttm_method: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


# --- loading ------------------------------------------------------------------------------------


def companies() -> pd.DataFrame:
    return pd.read_csv(REFERENCE_CSV).set_index("ticker")


def _d(s) -> date | None:
    return None if pd.isna(s) or s == "" else pd.Timestamp(s).date()


@lru_cache(maxsize=1)
def _facts_cached(keyed_mtime: float, page_mtime: float) -> pd.DataFrame:
    frames = []
    if KEYED_CSV.exists():
        k = pd.read_csv(KEYED_CSV, dtype={"doc_id": str})
        frames.append(pd.DataFrame({
            "ticker": k["ticker"], "line_item": k["line_item"],
            "value": k["value"] * k["unit"].map(UNIT_TO_MILLIONS),
            "kind": k["period_kind"], "period_start": k["period_start"],
            "period_end": k["period_end"], "published_at": k["published_at"], "version": 1,
            "priority": 0, "source": "PSX filing (" + k["published_via"] + ")",
            "source_ref": [DOC_URL.format(doc_id=d, page=p)
                           for d, p in zip(k["doc_id"], k["doc_page"])],
        }))
    if STATEMENTS_CSV.exists():
        s = pd.read_csv(STATEMENTS_CSV)
        scale = s["unit"].map({**UNIT_TO_MILLIONS, "PKR_per_share": 1.0})
        frames.append(pd.DataFrame({
            "ticker": s["ticker"], "line_item": s["line_item"], "value": s["value"] * scale,
            "kind": s["period_kind"], "period_start": s["period_start"],
            "period_end": s["period_end"], "published_at": s["published_at"],
            "version": s["version"], "priority": 1, "source": s["source"],
            "source_ref": s["source_ref"],
        }))
    if not frames:
        return pd.DataFrame(columns=["ticker", "line_item", "value", "kind", "period_start",
                                     "period_end", "published_at", "version", "priority",
                                     "source", "source_ref"])
    f = pd.concat(frames, ignore_index=True)
    for c in ("period_start", "period_end", "published_at"):
        f[c] = pd.to_datetime(f[c], errors="coerce").dt.date
    return f


def load_facts() -> pd.DataFrame:
    mt = lambda p: p.stat().st_mtime if p.exists() else 0.0  # noqa: E731
    return _facts_cached(mt(KEYED_CSV), mt(STATEMENTS_CSV))


def facts_as_of(facts: pd.DataFrame, ticker: str, as_of: date) -> pd.DataFrame:
    """Facts public on `as_of`: latest version per fact, filed report preferred over page."""
    f = facts[(facts["ticker"] == ticker) & (facts["published_at"] <= as_of)]
    if f.empty:
        return f
    f = f.sort_values(["priority", "version"], ascending=[True, False])
    key = ["line_item", "period_start", "period_end"]
    return f.drop_duplicates(key, keep="first")


# --- flows: year-to-date and trailing twelve months ---------------------------------------------


def _fact(row) -> Fact:
    return Fact(row.value, row.period_start, row.period_end, row.published_at, row.source,
                row.source_ref)


def _period(f: pd.DataFrame, item: str, start: date, end: date) -> list[Fact] | None:
    """Amount for exactly [start, end]: a direct fact, else the sum of its discrete quarters."""
    rows = f[(f["line_item"] == item) & (f["period_start"] == start) & (f["period_end"] == end)]
    if len(rows):
        return [_fact(rows.iloc[0])]
    parts, s = [], pd.Timestamp(start)
    while s.date() <= end:
        q_end = (s + pd.DateOffset(months=3) - pd.Timedelta(days=1)).date()
        if q_end > end:
            return None
        q = f[(f["line_item"] == item) & (f["period_start"] == s.date())
              & (f["period_end"] == q_end)]
        if q.empty:
            return None
        parts.append(_fact(q.iloc[0]))
        s = pd.Timestamp(q_end) + pd.Timedelta(days=1)
    return parts or None


def _fy_start(end: date, fy_end_month: int) -> date:
    """Start of the fiscal year containing `end`."""
    y = end.year if end.month > fy_end_month else end.year - 1
    return (pd.Timestamp(year=y, month=fy_end_month, day=1) + pd.offsets.MonthEnd(0)
            + pd.Timedelta(days=1)).date()


def ttm(f: pd.DataFrame, item: str, end: date, fy_end_month: int) -> tuple[Fact, str] | None:
    """Trailing twelve months ending `end` (PKR millions), with the method used."""
    fy_start = _fy_start(end, fy_end_month)
    fy_end = (pd.Timestamp(fy_start) + pd.DateOffset(years=1) - pd.Timedelta(days=1)).date()
    if end == fy_end:
        parts = _period(f, item, fy_start, end)
        method = "annual"
    else:
        prev_start = (pd.Timestamp(fy_start) - pd.DateOffset(years=1)).date()
        prev_end = (pd.Timestamp(fy_start) - pd.Timedelta(days=1)).date()
        prior_ytd_end = (pd.Timestamp(end) - pd.DateOffset(years=1)).date()
        prior_ytd_end = (pd.Timestamp(prior_ytd_end) + pd.offsets.MonthEnd(0)).date()
        cur = _period(f, item, fy_start, end)
        annual = _period(f, item, prev_start, prev_end)
        prior = _period(f, item, prev_start, prior_ytd_end)
        if not (cur and annual and prior):
            return None
        value = (sum(p.value for p in annual) + sum(p.value for p in cur)
                 - sum(p.value for p in prior))
        sources = cur + annual + prior
        latest = max(sources, key=lambda p: p.published_at)
        return Fact(value, None, end, latest.published_at, "; ".join(sorted({p.source for p in
                    sources})), " | ".join(dict.fromkeys(p.source_ref for p in sources))), \
            "prior annual + year-to-date - prior year-to-date"
    if not parts:
        return None
    latest = max(parts, key=lambda p: p.published_at)
    return Fact(sum(p.value for p in parts), fy_start, end, latest.published_at,
                "; ".join(sorted({p.source for p in parts})),
                " | ".join(dict.fromkeys(p.source_ref for p in parts))), method


# --- prices and shares --------------------------------------------------------------------------


def price_on(ticker: str, as_of: date) -> tuple[float, date] | None:
    """Official unadjusted PSX close on or before `as_of` (None if no row)."""
    df = psx_eod.load_symbol(ticker)
    if df is None:
        return None
    df = df[df.index <= pd.Timestamp(as_of)]
    if df.empty:
        return None
    return float(df["close"].iloc[-1]), df.index[-1].date()


def _face_value(ticker: str) -> float | None:
    if not SHARES_CSV.exists():
        return None
    sh = pd.read_csv(SHARES_CSV)
    sh = sh[sh["ticker"] == ticker].sort_values("as_of")
    if sh.empty or pd.isna(sh["face_value"].iloc[-1]):
        return None
    return float(sh["face_value"].iloc[-1])


def shares_on(f: pd.DataFrame, ticker: str, as_of: date) -> tuple[float, str] | None:
    """Shares outstanding known on `as_of`: filed share capital / face value, else PSX count."""
    cap = f[f["line_item"] == "share_capital"].sort_values("period_end")
    face = _face_value(ticker)
    if len(cap) and face:
        row = cap.iloc[-1]
        return row.value * 1e6 / face, f"share capital {row.period_end} / face value Rs {face:g}"
    if SHARES_CSV.exists():
        sh = pd.read_csv(SHARES_CSV)
        sh = sh[(sh["ticker"] == ticker) & (pd.to_datetime(sh["as_of"]).dt.date <= as_of)]
        if len(sh):
            r = sh.sort_values("as_of").iloc[-1]
            return float(r["shares_outstanding"]), f"PSX listed shares {r['as_of']}"
    return None


# --- snapshot -----------------------------------------------------------------------------------


def build_snapshot(ticker: str, as_of: date | None = None) -> Snapshot:
    ticker = ticker.upper()
    as_of = as_of or date.today()
    meta = companies()
    if ticker not in meta.index:
        raise KeyError(f"{ticker} is not in the fundamentals universe")
    m = meta.loc[ticker]
    snap = Snapshot(ticker, as_of, m["company_type"], m["sector"])
    f = facts_as_of(load_facts(), ticker, as_of)
    if f.empty:
        snap.warnings.append(f"No financial statements published by {as_of}")
        return snap

    # Balance sheet: the latest instant published by as_of.
    inst = f[f["kind"] == "instant"]
    if len(inst):
        bs_date = inst["period_end"].max()
        snap.balance_date = bs_date
        for r in inst[inst["period_end"] == bs_date].itertuples():
            if r.line_item in STOCK_ITEMS:
                snap.balance[r.line_item] = _fact(r)
        if (as_of - bs_date).days > STALE_STATEMENT_DAYS:
            snap.warnings.append(f"Latest balance sheet is from {bs_date} (stale)")
    else:
        snap.warnings.append("No balance sheet published by this date")

    # Trailing twelve months for every flow item, ending at the latest reported period end.
    flows = f[f["line_item"].isin(FLOW_ITEMS)]
    fy_month = int(m["fiscal_year_end_month"])
    for item in sorted(flows["line_item"].unique()):
        ends = sorted(flows[flows["line_item"] == item]["period_end"].unique(), reverse=True)
        for end in ends[:3]:  # latest first; fall back if the latest cannot form a TTM
            res = ttm(flows, item, end, fy_month)
            if res:
                snap.ttm[item], snap.ttm_method[item] = res
                break
        else:
            snap.warnings.append(f"Trailing twelve-month {item} cannot be formed")

    px = price_on(ticker, as_of)
    if px is None:
        snap.warnings.append("No official PSX price on or before this date")
    else:
        snap.price, snap.price_date = px
        age = (as_of - snap.price_date).days
        if age > MAX_PRICE_AGE_DAYS:
            snap.warnings.append(f"Latest price is {age} days old ({snap.price_date}); not used")
            snap.price = None
    sh = shares_on(f, ticker, as_of)
    if sh is None:
        snap.warnings.append("Shares outstanding unknown")
    else:
        snap.shares, snap.shares_source = sh
    if snap.price is not None and snap.shares:
        snap.market_cap = snap.price * snap.shares / 1e6
    return snap
