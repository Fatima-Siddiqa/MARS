"""Official PSX daily market-summary parsing + merge into the price loader (offline, no network)."""
import gzip
import io
import zipfile
from datetime import date

import pandas as pd
import pytest

from app.ingestion import prices, psx_eod

LIS = (
    "24NOV2026|HBL|0807|Habib Bank|306.0|308.74|304.65|305.7|1046958|303.07|||\n"
    "24NOV2026|PTC|0828|P.T.C.L.|44.01|45.15|43.65|43.89|4179973|43.7|||\n"
)


def _zip(text: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("closing11.lis", text)
    return buf.getvalue()


@pytest.fixture
def eod_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(psx_eod, "EOD_DIR", tmp_path)
    return tmp_path


def test_parse_zip_and_gzip_and_symbol_mapping(eod_dir):
    (eod_dir / "2026-11-24.Z").write_bytes(_zip(LIS))
    old = LIS.replace("24NOV2026", "14JUL2016").encode()
    (eod_dir / "2016-07-14.Z").write_bytes(gzip.compress(old))
    z = psx_eod.parse_day(eod_dir / "2026-11-24.Z")
    g = psx_eod.parse_day(eod_dir / "2016-07-14.Z")
    assert list(z["symbol"]) == ["HBL", "PTCL"]  # PSX trades PTCL as PTC
    assert z.loc[0, "close"] == 305.7 and z.loc[0, "date"] == pd.Timestamp("2026-11-24")
    assert g.loc[0, "date"] == pd.Timestamp("2016-07-14")
    assert z.loc[0, "source_url"].endswith("/mkt_summary/2026-11-24.Z")


def test_refresh_skips_weekends_and_non_trading_days(eod_dir, monkeypatch):
    calls = []

    def fake_download(d, session=None):
        calls.append(d)
        if d == date(2026, 11, 24):
            p = eod_dir / "2026-11-24.Z"
            p.write_bytes(_zip(LIS))
            return p
        return None  # holiday -> 404

    monkeypatch.setattr(psx_eod, "download_day", fake_download)
    monkeypatch.setattr(psx_eod, "REQUEST_DELAY_S", 0)
    out = psx_eod.refresh(date(2026, 11, 21), date(2026, 11, 25))  # Sat..Wed
    assert date(2026, 11, 22) not in calls and date(2026, 11, 21) not in calls
    assert out["trading_days_added"] == 1 and out["last_date"] == "2026-11-24"
    assert psx_eod.load_symbol("HBL")["close"].iloc[-1] == 305.7


def test_load_stock_appends_official_rows_after_xlsx_end(eod_dir, tmp_path, monkeypatch):
    hist = pd.DataFrame({
        "Date": pd.to_datetime(["2026-11-20", "2026-11-23"]), "Open": [300, 301],
        "High": [302, 303], "Low": [299, 300], "Close": [300.0, 303.07],
        "Adj. Close": [300.0, 303.07],
        "Change": [0, 0], "Volume": [10, 20],
    })
    price_dir = tmp_path / "stocks"
    price_dir.mkdir()
    hist.to_csv(price_dir / "HBL.csv", index=False)
    monkeypatch.setattr(prices, "PRICE_DIR", price_dir)
    (eod_dir / "2026-11-24.Z").write_bytes(_zip(LIS))
    psx_eod._write([psx_eod.parse_day(eod_dir / "2026-11-24.Z")])

    df = prices.load_stock("HBL")
    assert list(df.index.date) == [date(2026, 11, 20), date(2026, 11, 23), date(2026, 11, 24)]
    assert df["Close"].iloc[-1] == 305.7
    # A symbol with no xlsx file is served from official rows alone.
    assert prices.load_stock("PTCL")["Close"].iloc[-1] == 43.89
