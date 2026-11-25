"""Automatic statement reading: parsing, subtotal detection and the arithmetic gates (offline)."""

from app.agents.fundamentals.data import auto_extract as ax

BANK_BS = """
Condensed Interim Unconsolidated Statement of Financial Position
ASSETS
Cash and balances with treasury banks 5 500 400
Investments 8 1,000 900
Advances 9 2,000 1,800
Deferred tax assets - -
Other assets 13 500 400
4,000 3,500
LIABILITIES
Bills payable 14 100 90
Deposits and other accounts 16 3,400 3,000
Other liabilities 20 200 160
3,700 3,250
NET ASSETS 300 250
Share capital 100 100
Reserves 50 40
Unappropriated profit 150 110
300 250
"""

NON_BANK_BS = """
STATEMENT OF FINANCIAL POSITION
Share capital 4 200 200
Capital reserves 300 300
General reserves 100 100
Unappropriated profit 400 350
500 450
1,000 950
Long term borrowings - secured 5 600 650
Trade and other payables 6 400 300
TOTAL LIABILITIES 1,000 950
TOTAL EQUITY AND LIABILITIES 2,000 1,900
"""

PL = """
STATEMENT OF PROFIT OR LOSS
Three month period ended Nine month period ended
Turnover - net 13 400 350 1,100 1,000
PROFIT FOR THE PERIOD 50 40 140 120
"""


def test_bank_subtotals_found_and_identity_confirmed():
    got, notes = ax.balance_sheet(BANK_BS)
    assert got["total_assets"] == 4000 and got["total_liabilities"] == 3700
    assert got["total_equity"] == 300 and got["deposits"] == 3400 and got["advances"] == 2000
    assert "equity components add up" in notes
    assert "identity holds on independently read totals" in notes


def test_non_bank_equity_from_components_skips_reserve_subtotal():
    got, notes = ax.balance_sheet(NON_BANK_BS)
    assert got["total_equity"] == 1000  # 500 is the revenue-reserves subtotal, not equity
    assert got["total_assets"] == 2000 and got["total_liabilities"] == 1000
    assert not any(n.startswith("FAILED") for n in notes)


def test_inconsistent_statement_is_flagged():
    bad = NON_BANK_BS.replace("TOTAL LIABILITIES 1,000 950", "TOTAL LIABILITIES 1,100 950")
    _, notes = ax.balance_sheet(bad)
    assert any(n.startswith("FAILED identity") for n in notes)


def test_ytd_column_chosen_by_psx_quarters():
    got, how = ax.ytd_lines(PL, quarters_sum=140.0, unit_scale=1.0)
    assert got["net_income"] == (140, 120) and got["revenue"] == (1100, 1000)
    assert "column 3" in how
    none, why = ax.ytd_lines(PL, quarters_sum=999.0, unit_scale=1.0)
    assert none == {} and "matches" in why
