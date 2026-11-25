"""Point-in-time data layer for the fundamentals agent (offline fixtures, no network)."""
from datetime import date

import pandas as pd
import pytest

from app.agents.fundamentals.data import loader
from app.ingestion import psx_eod


def _page(item, value, start, end, kind, published, version=1):
    return {"ticker": "TST", "line_item": item, "value": value, "unit": "PKR_thousands",
            "period_start": start, "period_end": end, "period_kind": kind, "fiscal_label": "",
            "published_at": published, "published_via": "x", "source": "PSX page",
            "source_ref": "page", "retrieved_at": "", "is_restated": version > 1,
            "version": version}


def _keyed(item, value, kind, start, end, published, unit="PKR_thousands"):
    return {"ticker": "TST", "line_item": item, "value": value, "unit": unit, "period_kind": kind,
            "period_start": start, "period_end": end, "doc_id": "999", "doc_page": 3,
            "published_at": published, "published_via": "report", "extraction": "pdf_text",
            "note": ""}


@pytest.fixture
def data(tmp_path, monkeypatch):
    ref = tmp_path / "companies.csv"
    pd.DataFrame([{"ticker": "TST", "psx_symbol": "TST", "company_name": "Test",
                   "company_type": "non_bank", "psx_sector_code": "0804", "sector": "Cement",
                   "fiscal_year_end_month": 12,
                   "statement_basis": "unconsolidated"}]).to_csv(ref, index=False)
    page = pd.DataFrame([
        _page("net_income", 400_000, "2025-01-01", "2025-12-31", "annual", "2026-03-01"),
        _page("net_income", 100_000, "2026-01-01", "2026-03-31", "quarter", "2026-04-25"),
        _page("net_income", 110_000, "2026-04-01", "2026-06-30", "quarter", "2026-08-20"),
        _page("net_income", 999_999, "2026-07-01", "2026-09-30", "quarter", "2026-10-25"),
        # restated in December: only visible from then on
        _page("net_income", 130_000, "2026-04-01", "2026-06-30", "quarter", "2026-12-01", 2),
    ])
    keyed = pd.DataFrame([
        _keyed("net_income", 330_000, "ytd", "2026-01-01", "2026-09-30", "2026-10-30"),
        _keyed("net_income", 300_000, "ytd", "2025-01-01", "2025-09-30", "2026-10-30"),
        _keyed("total_equity", 2_000_000, "instant", "", "2026-09-30", "2026-10-30"),
        _keyed("share_capital", 1_000_000, "instant", "", "2026-09-30", "2026-10-30"),
    ])
    shares = pd.DataFrame([{"ticker": "TST", "as_of": "2026-11-26", "shares_outstanding": 9e7,
                            "face_value": 10.0, "unit": "shares", "source": "s",
                            "source_ref": "r", "retrieved_at": ""}])
    for name, df in [("st.csv", page), ("keyed.csv", keyed), ("shares.csv", shares)]:
        df.to_csv(tmp_path / name, index=False)
    monkeypatch.setattr(loader, "REFERENCE_CSV", ref)
    monkeypatch.setattr(loader, "STATEMENTS_CSV", tmp_path / "st.csv")
    monkeypatch.setattr(loader, "KEYED_CSV", tmp_path / "keyed.csv")
    monkeypatch.setattr(loader, "SHARES_CSV", tmp_path / "shares.csv")
    monkeypatch.setattr(psx_eod, "EOD_DIR", tmp_path / "eod")
    (tmp_path / "eod" / "symbols").mkdir(parents=True)
    pd.DataFrame({"date": ["2026-11-20", "2026-11-25"], "symbol": "TST", "close": [50.0, 52.0]}
                 ).to_csv(tmp_path / "eod" / "symbols" / "TST.csv", index=False)
    loader._facts_cached.cache_clear()
    yield
    loader._facts_cached.cache_clear()


def test_only_published_facts_and_latest_version_by_then(data):
    f = loader.load_facts()
    before = loader.facts_as_of(f, "TST", date(2026, 9, 1))
    q2 = before[(before["period_end"] == date(2026, 6, 30))]
    assert list(q2["value"]) == [110.0]  # PKR millions; restatement not yet published
    after = loader.facts_as_of(f, "TST", date(2026, 12, 5))
    q2 = after[(after["period_end"] == date(2026, 6, 30))]
    assert list(q2["value"]) == [130.0]
    assert loader.facts_as_of(f, "TST", date(2026, 2, 1)).empty


def test_ttm_prefers_filed_ytd_over_page_quarters(data):
    s = loader.build_snapshot("TST", date(2026, 11, 26))
    # annual 2025 (400) + filed 9M 2026 (330) - filed 9M 2025 (300); page Q3 (999.999) ignored
    assert s.ttm["net_income"].value == pytest.approx(430.0)
    assert "prior annual" in s.ttm_method["net_income"]


def test_snapshot_price_shares_and_market_cap(data):
    s = loader.build_snapshot("TST", date(2026, 11, 26))
    assert (s.price, s.price_date) == (52.0, date(2026, 11, 25))
    assert s.shares == pytest.approx(100_000_000)  # Rs 1,000m share capital / Rs 10 face value
    assert s.market_cap == pytest.approx(5_200.0)  # PKR millions
    assert s.balance["total_equity"].value == pytest.approx(2_000.0)
    assert s.balance_date == date(2026, 9, 30)


def test_stale_price_is_not_used(data):
    s = loader.build_snapshot("TST", date(2026, 12, 20))
    assert s.price is None and s.market_cap is None
    assert any("days old" in w for w in s.warnings)


def test_before_any_filing_nothing_is_invented(data):
    s = loader.build_snapshot("TST", date(2026, 2, 1))
    assert s.ttm == {} and s.balance == {} and s.market_cap is None
    assert s.warnings == ["No financial statements published by 2026-02-01"]
