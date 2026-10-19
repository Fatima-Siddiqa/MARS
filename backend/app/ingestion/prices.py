"""PSX price ingestion (OHLCV) + cleaning. Owner: Fatima. Feeds technical (+ fx) agents.

Raw files are NOT committed. Copy them to `data/raw/stocks/` (xlsx) and the KSE-100 csv to
`data/raw/KSE_100_index.csv`. Override with env vars MARS_PRICE_DIR / MARS_BENCHMARK_PATH.
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
}
UNIVERSE = ["HBL", "HUBC", "OGDC", "UBL", "PPL", "PTCL", "SEARL", "LUCK", "SYS", "ENGROH"]
PRICE_SOURCE = "PSX historical prices (FYP dataset, Adj. Close)"


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


def load_stock(symbol: str) -> pd.DataFrame | None:
    p = price_path(symbol)
    if p is None:
        return None
    raw = pd.read_excel(p) if p.suffix == ".xlsx" else pd.read_csv(p)
    return clean_prices(raw)


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
