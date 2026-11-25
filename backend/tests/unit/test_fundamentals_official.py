"""Official PSX fundamentals: parsing, fiscal periods, publication dates, versions (offline)."""
from datetime import date

import pandas as pd

from app.agents.fundamentals.data import psx_official as po
from app.agents.fundamentals.data import validate_official as vo

PAGE = """
<div id="financials">
 <table><tr><th></th><th>2026</th><th>2025</th></tr>
  <tr><td>Sales</td><td>1,000</td><td>900</td></tr>
  <tr><td>Profit after Taxation</td><td>(50)</td><td>-</td></tr></table>
 <table><tr><th></th><th>Q1 2027</th><th>Q3 2026</th></tr>
  <tr><td>Sales</td><td>300</td><td>250</td></tr></table>
</div>
<div id="equity">Equity Profile Market Cap 1 Shares 1,465,000,000 Face Value 2.00 Free Float</div>
<table>
 <tr><td>Oct 29, 2026</td><td>Financial Results for the Quarter Ended September 30, 2026</td>
     <td><a href="/download/document/1.pdf">PDF</a></td></tr>
 <tr><td>Oct 30, 2026</td>
     <td>Transmission of Quarterly Report for the Period Ended September 30, 2026</td>
     <td><a href="/download/document/2.pdf">PDF</a></td></tr>
 <tr><td>Oct 22, 2026</td><td>Financial Results for the Quarter ended September 30 3036 REVOKED</td>
     <td><a href="/download/document/3.pdf">PDF</a></td></tr>
 <tr><td>Oct 13, 2026</td><td>Board Meeting for Announcement of Financial Results</td>
     <td><a href="/download/document/4.pdf">PDF</a></td></tr>
</table>
"""


def test_fiscal_periods_for_june_and_december_year_ends():
    assert po.fiscal_period("Q1 2027", 6) == (date(2026, 7, 1), date(2026, 9, 30), "quarter")
    assert po.fiscal_period("2026", 6) == (date(2025, 7, 1), date(2026, 6, 30), "annual")
    assert po.fiscal_period("Q3 2026", 12) == (date(2026, 7, 1), date(2026, 9, 30), "quarter")
    assert po.fiscal_period("2025", 12) == (date(2025, 1, 1), date(2025, 12, 31), "annual")


def test_parse_page_financials_shares_and_announcements():
    rows = po.parse_financials(PAGE)
    assert ("2026", "revenue", 1000.0) in rows
    assert ("2026", "net_income", -50.0) in rows  # brackets are negatives
    assert not any(r[0] == "2025" and r[1] == "net_income" for r in rows)  # "-" is missing, not 0
    assert ("Q1 2027", "revenue", 300.0) in rows
    assert po.parse_shares(PAGE) == {"shares_outstanding": 1_465_000_000.0, "face_value": 2.0}
    anns = po.parse_announcements(PAGE)
    assert [a["kind"] for a in anns] == ["results", "report"]  # revoked + board meeting dropped
    assert all(a["period_end"] == date(2026, 9, 30) for a in anns)


def test_publication_prefers_results_then_report_then_portal():
    anns = po.parse_announcements(PAGE)
    pe = date(2026, 9, 30)
    pub = po.publication(pe, "quarter", date(2027, 6, 30), anns, [])
    assert pub[0] == date(2026, 10, 29) and "results" in pub[1]
    portal = [{"kind": "quarterly", "period_ended": "2026-03-31", "posting_date": "2026-04-28",
               "url": "u"},
              {"kind": "quarterly", "period_ended": "2025-12-31", "posting_date": "2025-11-01",
               "url": "bad"}]  # posted before the period ended: impossible, ignored
    assert po.publication(date(2026, 3, 31), "quarter", date(2026, 6, 30), [], portal)[0] == \
        date(2026, 4, 28)
    assert po.publication(date(2025, 12, 31), "quarter", date(2026, 6, 30), [], portal) is None


def _row(value, retrieved="2026-11-26T10:00:00+00:00"):
    return {"ticker": "LUCK", "line_item": "net_income", "value": value, "unit": "PKR_thousands",
            "period_start": "2026-07-01", "period_end": "2026-09-30", "period_kind": "quarter",
            "fiscal_label": "Q1 2027", "published_at": "2026-10-29", "published_via": "x",
            "source": "s", "source_ref": "r", "retrieved_at": retrieved, "is_restated": False,
            "version": 1}


def test_restatement_is_a_new_version_never_an_overwrite():
    old = pd.DataFrame([_row(100.0)])
    same = po.merge_versions(old, pd.DataFrame([_row(100.0)]))
    assert len(same) == 1
    changed = po.merge_versions(old, pd.DataFrame([_row(120.0, "2026-12-05T09:00:00+00:00")]))
    assert list(changed["value"]) == [100.0, 120.0]
    new = changed.iloc[1]
    assert new["version"] == 2 and bool(new["is_restated"])
    assert new["published_at"] == "2026-12-05"  # knowable only from when it was observed


def test_validator_balance_identity_and_printed_format():
    k = pd.DataFrame([
        {"ticker": "X", "period_kind": "instant", "period_end": "2026-09-30",
         "line_item": li, "value": v}
        for li, v in [("total_assets", 100), ("total_liabilities", 60), ("total_equity", 39)]
    ])
    assert vo.check_balance_identity(k) == ["X 2026-09-30: assets - liabilities - equity = 1"]
    assert vo._printed(-1217192) == "(1,217,192)" and vo._printed(2959476) == "2,959,476"
