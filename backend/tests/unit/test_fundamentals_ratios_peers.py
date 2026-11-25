"""Ratios and peer comparison on hand-built snapshots (no data files)."""

from datetime import date

import pytest

from app.agents.fundamentals.data.loader import Fact, Snapshot
from app.agents.fundamentals.features import compute_ratios, coverage
from app.agents.fundamentals.peers import baseline, compare

D = date(2026, 9, 30)


def F(v, start=None, end=D, ref="ref"):
    return Fact(v, start, end, date(2026, 10, 30), "PSX filing", ref)


def snap(
    t,
    ctype="non_bank",
    sector="0804",
    price=100.0,
    shares=1e8,
    ni=500.0,
    eq=4000.0,
    rev=2000.0,
    **bal,
):
    s = Snapshot(
        t,
        date(2026, 11, 26),
        ctype,
        "Sector",
        sector,
        price=price,
        price_date=date(2026, 11, 25),
        shares=shares,
        shares_source="test",
    )
    s.market_cap = None if price is None else price * shares / 1e6
    s.balance = {
        "total_equity": F(eq),
        "total_assets": F(10000.0),
        "total_liabilities": F(10000.0 - eq),
        **{k: F(v) for k, v in bal.items()},
    }
    s.ttm = {"net_income": F(ni)} | ({"revenue": F(rev)} if rev is not None else {})
    s.annual = {"net_income": [F(400.0, end=date(2024, 12, 31)), F(ni, end=date(2025, 12, 31))]}
    return s


def by_name(ratios):
    return {r.name: r for r in ratios}


def test_non_bank_ratios_values_and_sources():
    r = by_name(compute_ratios(snap("A", total_borrowings=1000.0, cash_and_equivalents=200.0)))
    assert r["pe"].value == pytest.approx(10_000 / 500)  # market cap 10,000m / NI 500m
    assert r["pb"].value == pytest.approx(2.5)
    assert r["roe"].value == pytest.approx(0.125)
    assert r["net_margin"].value == pytest.approx(0.25)
    assert r["net_debt_to_equity"].value == pytest.approx(0.2)
    assert r["earnings_growth"].value == pytest.approx(0.25)
    assert "ref" in r["roe"].sources


def test_missing_and_negative_inputs_give_reasons_not_zero():
    r = by_name(compute_ratios(snap("A", ni=-50.0, eq=-10.0, rev=None, price=None)))
    assert r["pe"].value is None and r["pe"].reason.startswith("market price")
    assert r["roe"].value is None and r["roe"].reason == "denominator is negative"
    assert r["net_margin"].value is None and r["net_margin"].reason == "input not available"
    assert r["debt_to_equity"].value is None
    assert 0 < coverage(list(r.values())) < 1


def test_bank_ratios_use_year_to_date_lines():
    s = snap("B", ctype="bank", advances=3000.0, deposits=6000.0, investments=4000.0)
    s.ytd = {
        "operating_expenses": F(300.0, start=date(2026, 1, 1)),
        "total_income": F(600.0, start=date(2026, 1, 1)),
        "net_interest_income": F(450.0, start=date(2026, 1, 1)),
    }
    r = by_name(compute_ratios(s))
    assert r["advances_to_deposits"].value == pytest.approx(0.5)
    assert r["cost_to_income"].value == pytest.approx(0.5)
    assert r["nii_to_assets"].value == pytest.approx(450 * 12 / 9 / 10000)  # 9 months annualised
    assert "ps" not in r


def _uni(*snaps):
    return {s.ticker: (s, compute_ratios(s)) for s in snaps}


def test_peers_same_sector_and_type_target_excluded():
    a, b, c = snap("A"), snap("B", ni=1000.0), snap("C", ni=250.0)
    other_sector, bank = snap("X", sector="0809"), snap("K", ctype="bank")
    unpriced = snap("U", price=None)
    res = compare(a, _uni(a, b, c, other_sector, bank, unpriced))
    assert res.peers == ["B", "C"] and res.strength == "weak"
    assert res.stats["pe"].peer_median == pytest.approx((10 + 40) / 2)


def test_peer_baseline_value_and_gap():
    a, b, c, d = snap("A"), snap("B", ni=1000.0), snap("C", ni=250.0), snap("D", ni=500.0)
    uni = _uni(a, b, c, d)
    res = compare(a, uni)
    assert res.strength == "adequate"
    bl = baseline(a, uni["A"][1], res)
    # peer P/E median = 20 (10, 20, 40); EPS = 500m / 100m shares = 5 -> implied 100 = price
    assert bl.method == "peer_pe" and bl.implied_value_per_share == pytest.approx(100.0)
    assert bl.gap == pytest.approx(0.0)
    loss_maker = snap("L", ni=-5.0)
    bl2 = baseline(
        loss_maker, compute_ratios(loss_maker), compare(loss_maker, uni | _uni(loss_maker))
    )
    assert bl2.method == "peer_pb"  # P/E is meaningless with negative earnings
