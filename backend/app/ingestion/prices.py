"""PSX price ingestion (OHLCV) + cleaning. Owner: Fatima. Feeds technical (+ fx) agents.

Raw files are NOT committed. Copy them to `data/raw/stocks/` (xlsx) and the KSE-100 csv to
`data/raw/KSE_100_index.csv`. Override with env vars MARS_PRICE_DIR / MARS_BENCHMARK_PATH.
Prices after the xlsx files end come from official PSX daily files: refresh them with
`python -m app.ingestion.psx_eod` (see app/ingestion/psx_eod.py).
"""
import asyncio
import os
from pathlib import Path

import pandas as pd

from app.core.config import REPO_ROOT

PRICE_DIR = Path(os.environ.get("MARS_PRICE_DIR", REPO_ROOT / "data" / "raw" / "stocks"))
BENCHMARK_PATH = Path(
    os.environ.get("MARS_BENCHMARK_PATH", REPO_ROOT / "data" / "raw" / "KSE_100_index.csv")
)

# symbol -> file stem (files shipped in the FYP dataset have irregular names)
FILE_ALIASES = {
    "LUCK": "Lucky Cement",
    "SYS": "Systems Limited",
    "SEARL": "SERL",
    "ENGRO": "ENGROH",
    # symbols used by the frontend ticker list (frontend/data/tickers.ts)
    "LUCKY": "Lucky Cement",
    "SYSTEMS": "Systems Limited",
    "OGDCL": "OGDC",
}
UNIVERSE = ["HBL", "HUBC", "OGDC", "UBL", "PPL", "PTCL", "SEARL", "LUCK", "SYS", "ENGROH"]
PRICE_SOURCE = (
    "PSX historical prices (FYP dataset, Adj. Close) + PSX Data Portal daily market summary"
)


def price_path(symbol: str) -> Path | None:
    stem = FILE_ALIASES.get(symbol.upper(), symbol.upper())
    for ext in (".xlsx", ".csv"):
        p = PRICE_DIR / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def clean_prices(raw: pd.DataFrame) -> pd.DataFrame:
    """Cleaning rules from the methodology (sec 3.2). Returns ascending, adjusted OHLCV.

    - '-' in Volume/Change -> NaN; Volume NaN => no_trade=True, Volume=0
    - Open/High/Low scaled by Adj.Close/Close; Close := Adj.Close
    - Change column dropped (returns are computed from Adj.Close)
    """
    d = raw.copy()
    d["Date"] = pd.to_datetime(d["Date"])
    d = d.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)
    for c in ["Open", "High", "Low", "Close", "Adj. Close", "Volume"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["Close", "Adj. Close"]).reset_index(drop=True)
    d["no_trade"] = d["Volume"].isna() | (d["Volume"] == 0)
    d["Volume"] = d["Volume"].fillna(0.0)
    f = d["Adj. Close"] / d["Close"]
    for c in ["Open", "High", "Low"]:
        d[c] = d[c] * f
    d["Close"] = d["Adj. Close"]
    d = d.drop(columns=["Adj. Close", "Change"], errors="ignore")
    return d.set_index("Date")[["Open", "High", "Low", "Close", "Volume", "no_trade"]]


def _official_rows(symbol: str, after: pd.Timestamp | None) -> pd.DataFrame | None:
    """Official PSX EOD rows (app.ingestion.psx_eod) after `after`, in the raw-file layout.

    PSX closes are unadjusted; Adj. Close = Close for these rows. Where the xlsx history ends the
    adjustment factor is 1 (its last Adj. Close equals its Close), so the series stays continuous.
    """
    from app.ingestion import psx_eod

    canonical = {"LUCKY": "LUCK", "SYSTEMS": "SYS", "OGDCL": "OGDC"}.get(symbol.upper(), symbol)
    e = psx_eod.load_symbol(canonical)
    if e is None:
        return None
    if after is not None:
        e = e[e.index > after]
    if e.empty:
        return None
    return pd.DataFrame({
        "Date": e.index, "Open": e["open"].values, "High": e["high"].values,
        "Low": e["low"].values, "Close": e["close"].values, "Adj. Close": e["close"].values,
        "Volume": e["volume"].values,
    })


def load_stock(symbol: str) -> pd.DataFrame | None:
    """FYP xlsx history, extended with official PSX daily prices after the file ends."""
    p = price_path(symbol)
    raw = None
    if p is not None:
        raw = pd.read_excel(p) if p.suffix == ".xlsx" else pd.read_csv(p)
    after = pd.to_datetime(raw["Date"]).max() if raw is not None else None
    extra = _official_rows(symbol, after)
    if raw is None and extra is None:
        return None
    parts = [x for x in (raw, extra) if x is not None]
    return clean_prices(pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0])


def load_benchmark() -> pd.Series | None:
    """KSE-100 daily close, ascending. Tolerates the junk 'Daily,' second row."""
    if not BENCHMARK_PATH.exists():
        return None
    k = pd.read_csv(BENCHMARK_PATH)
    k.columns = ["Date", "Close"]
    k["Date"] = pd.to_datetime(k["Date"], format="%d-%b-%Y", errors="coerce")
    k["Close"] = pd.to_numeric(k["Close"], errors="coerce")
    k = k.dropna().drop_duplicates("Date").sort_values("Date")
    return k.set_index("Date")["Close"]


async def fetch_prices(symbol: str, period: str = "max") -> pd.DataFrame:
    """Async wrapper (file IO off the event loop). `period` is kept for API compatibility."""
    df = await asyncio.to_thread(load_stock, symbol)
    if df is None:
        raise FileNotFoundError(f"No price file for {symbol} in {PRICE_DIR}")
    return df


def sanity_warnings(df: pd.DataFrame) -> list[str]:
    """Flag suspicious data (daily move > 20% needs investigation, per methodology)."""
    r = df["Close"].pct_change().abs()
    big = r[r > 0.20]
    return [f"Daily move >20% on {d.date()} ({v:.0%}); verify corporate action" for d, v in big.items()][:3] if len(big) else []  # noqa: E501


__all__ = ["fetch_prices", "load_stock", "load_benchmark", "clean_prices", "UNIVERSE"]
