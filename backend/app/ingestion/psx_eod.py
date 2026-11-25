"""Official PSX end-of-day prices from the daily market-summary files.

Source: Pakistan Stock Exchange Data Portal, Downloads -> "Market Summary"
(`https://dps.psx.com.pk/download/mkt_summary/YYYY-MM-DD.Z`). One file per trading day with every
listed symbol (non-trading days return 404). The archive is a zip (older days: gzip) holding a
pipe-delimited `closing*.lis`:

    24NOV2026|HBL|0807|Habib Bank|306.0|308.74|304.65|305.7|1046958|303.07|||
    date      sym  sector name       open  high   low    close volume  ldcp

Prices are the exchange's official closes, NOT adjusted for dividends/bonus issues.

Raw daily files are cached in `data/raw/psx_eod/` (so a refresh only downloads missing days and
every row can be traced to its file); parsed rows are kept per symbol in
`data/raw/psx_eod/symbols/{SYMBOL}.csv`.

    python -m app.ingestion.psx_eod                  # refresh: last stored day -> today
    python -m app.ingestion.psx_eod --since 2026-07-15
    python -m app.ingestion.psx_eod --status         # last date per tracked symbol
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import logging
import os
import time
import zipfile
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

from app.core.config import REPO_ROOT

log = logging.getLogger(__name__)

BASE_URL = "https://dps.psx.com.pk/download/mkt_summary/{d}.Z"
SOURCE_NAME = "PSX Data Portal, daily market summary"
EOD_DIR = Path(os.environ.get("MARS_PSX_EOD_DIR", REPO_ROOT / "data" / "raw" / "psx_eod"))
DEFAULT_SINCE = date(2026, 7, 15)  # day after the FYP xlsx price files end
REQUEST_DELAY_S = 0.5  # be polite to the exchange's server
USER_AGENT = "MARS-FYP-research/0.1 (academic project; daily PSX market summary)"

# MARS symbol -> PSX trading symbol, where they differ.
PSX_SYMBOL = {"PTCL": "PTC"}
MARS_SYMBOL = {v: k for k, v in PSX_SYMBOL.items()}

COLUMNS = [
    "date", "symbol", "sector_code", "name", "open", "high", "low", "close", "volume", "ldcp",
]


def day_path(d: date) -> Path:
    return EOD_DIR / f"{d.isoformat()}.Z"


def download_day(d: date, session: requests.Session | None = None) -> Path | None:
    """Download one day's file into the cache. Returns None on a non-trading day (404)."""
    p = day_path(d)
    if p.exists():
        return p
    s = session or requests.Session()
    url = BASE_URL.format(d=d.isoformat())
    for attempt in range(3):
        try:
            r = s.get(url, headers={"User-Agent": USER_AGENT}, timeout=60)
        except requests.RequestException as exc:
            log.warning("PSX %s attempt %d failed: %s", d, attempt + 1, exc)
            time.sleep(2 * (attempt + 1))
            continue
        if r.status_code == 404:
            return None
        r.raise_for_status()
        if not (r.content[:2] == b"PK" or r.content[:2] == b"\x1f\x8b"):
            raise ValueError(f"Unexpected (non-archive) response for {url}")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(r.content)
        return p
    raise ConnectionError(f"Could not download {url}")


def _read_lis(raw: bytes) -> str:
    if raw[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            name = next(n for n in z.namelist() if n.lower().endswith(".lis"))
            return z.read(name).decode("latin-1")
    return gzip.decompress(raw).decode("latin-1")


def parse_day(path: Path) -> pd.DataFrame:
    """Parse one cached daily file into rows (MARS symbols, numeric prices)."""
    rows = []
    for line in _read_lis(path.read_bytes()).splitlines():
        parts = line.strip().split("|")
        if len(parts) < 10 or not parts[1]:
            continue
        rows.append(parts[:10])
    df = pd.DataFrame(rows, columns=COLUMNS)
    df["date"] = pd.to_datetime(df["date"], format="%d%b%Y")
    for c in ["open", "high", "low", "close", "volume", "ldcp"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # Listed companies only: equity sector codes are 08xx; futures (40) and other
    # instruments are not needed by any agent.
    df = df[df["sector_code"].str.strip().str.startswith("08")].copy()
    df["symbol"] = df["symbol"].str.strip().replace(MARS_SYMBOL)
    df["name"] = df["name"].str.strip()
    df["source_url"] = BASE_URL.format(d=path.stem)
    return df


def symbol_path(symbol: str) -> Path:
    safe = "".join(ch for ch in symbol.upper() if ch.isalnum() or ch in "-_")
    return EOD_DIR / "symbols" / f"{safe}.csv"


def _status() -> dict:
    p = EOD_DIR / "status.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _write(frames: list[pd.DataFrame]) -> None:
    """Merge newly parsed days into the per-symbol files (one small CSV per symbol)."""
    if not frames:
        return
    new = pd.concat(frames, ignore_index=True)
    for sym, rows in new.groupby("symbol"):
        p = symbol_path(sym)
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists():
            rows = pd.concat([pd.read_csv(p, parse_dates=["date"]), rows], ignore_index=True)
        rows = rows.drop_duplicates("date", keep="last").sort_values("date")
        rows.to_csv(p, index=False)
    last = new["date"].max().date().isoformat()
    st = _status()
    if last > st.get("last_date", ""):
        st["last_date"] = last
    (EOD_DIR / "status.json").write_text(json.dumps(st), encoding="utf-8")


def refresh(since: date | None = None, until: date | None = None, flush_every: int = 40) -> dict:
    """Download every missing trading day in [since, until] into the per-symbol files.

    Without `since`, continues from the last stored day (or DEFAULT_SINCE for an empty store).
    """
    if since is None:
        last = _status().get("last_date")
        since = date.fromisoformat(last) + timedelta(days=1) if last else DEFAULT_SINCE
    until = until or date.today()
    session = requests.Session()
    pending, added, closed = [], 0, 0
    d = since
    while d <= until:
        if d.weekday() < 5:  # PSX trades Mon-Fri
            cached = day_path(d).exists()
            p = download_day(d, session)
            if p is None:
                closed += 1
            else:
                pending.append(parse_day(p).assign(retrieved_at=datetime.now(UTC).isoformat()))
                added += 1
            if not cached:
                time.sleep(REQUEST_DELAY_S)
            if len(pending) >= flush_every:
                _write(pending)
                pending = []
        d += timedelta(days=1)
    _write(pending)
    return {
        "range": [since.isoformat(), until.isoformat()],
        "trading_days_added": added,
        "non_trading_days": closed,
        "last_date": _status().get("last_date"),
    }


def rebuild() -> dict:
    """Recreate every per-symbol file from the cached daily files (no downloads)."""
    days = sorted(EOD_DIR.glob("????-??-??.Z"))
    frames = [parse_day(p).assign(retrieved_at="cached") for p in days]
    for f in (EOD_DIR / "symbols").glob("*.csv"):
        f.unlink()
    (EOD_DIR / "status.json").unlink(missing_ok=True)
    for i in range(0, len(frames), 100):
        _write(frames[i:i + 100])
    return {"days": len(days), "last_date": _status().get("last_date")}


def load_symbol(symbol: str) -> pd.DataFrame | None:
    """Official EOD rows for one MARS symbol, indexed by date (ascending), or None."""
    p = symbol_path(symbol)
    if not p.exists():
        return None
    df = pd.read_csv(p, parse_dates=["date"])
    return None if df.empty else df.sort_values("date").set_index("date")


def symbols() -> list[str]:
    return sorted(p.stem for p in (EOD_DIR / "symbols").glob("*.csv"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Refresh official PSX end-of-day prices")
    ap.add_argument("--since", type=date.fromisoformat)
    ap.add_argument("--until", type=date.fromisoformat)
    ap.add_argument("--status", action="store_true", help="show last stored date per symbol")
    ap.add_argument("--rebuild", action="store_true", help="rebuild files from cached days")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    if a.rebuild:
        print(rebuild())
        return
    if a.status:
        from app.core.config import get_settings

        tracked = pd.read_csv(get_settings().ticker_map_path)["symbol"]
        for sym in sorted(set(tracked)):
            rows = load_symbol(sym)
            last = rows.index.max().date() if rows is not None else "-"
            print(f"{sym:8s} {last}  ({0 if rows is None else len(rows)} rows)")
        return
    print(refresh(a.since, a.until))


if __name__ == "__main__":
    main()
