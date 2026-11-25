"""Financial ratios from a point-in-time Snapshot (banks and non-banks separately).

Every ratio is a `Ratio`: a value with its formula and the source references of its inputs, or
None with the reason (missing input, zero/negative denominator, ...). Nothing defaults to 0.
Percentages are returned as fractions (0.15 = 15%).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.agents.fundamentals.data.loader import Fact, Snapshot


@dataclass
class Ratio:
    name: str
    label: str
    value: float | None
    formula: str
    family: str  # valuation | profitability | growth | leverage | liquidity | efficiency
    unit: str = "x"  # x (multiple) | pct (fraction)
    reason: str | None = None
    sources: list[str] = field(default_factory=list)


def _refs(*facts: Fact | None) -> list[str]:
    return list(dict.fromkeys(f.source_ref for f in facts if f is not None))


def _div(
    name,
    label,
    num,
    den,
    formula,
    family,
    unit="x",
    positive_den=True,
    num_fact=None,
    den_fact=None,
    missing=None,
) -> Ratio:
    if num is None or den is None:
        return Ratio(
            name, label, None, formula, family, unit, reason=missing or "input not available"
        )
    if den == 0 or (positive_den and den < 0):
        why = "denominator is zero" if den == 0 else "denominator is negative"
        return Ratio(
            name, label, None, formula, family, unit, reason=why, sources=_refs(num_fact, den_fact)
        )
    return Ratio(name, label, num / den, formula, family, unit, sources=_refs(num_fact, den_fact))


def _v(f: Fact | None) -> float | None:
    return None if f is None else f.value


def _growth(s: Snapshot, item: str, name: str, label: str) -> Ratio:
    years = s.annual.get(item, [])
    formula = f"latest fiscal-year {item} / previous fiscal-year {item} - 1"
    if len(years) < 2:
        return Ratio(
            name,
            label,
            None,
            formula,
            "growth",
            "pct",
            reason="fewer than two published fiscal years",
        )
    prev, last = years
    if prev.value <= 0:
        return Ratio(
            name,
            label,
            None,
            formula,
            "growth",
            "pct",
            reason="previous year is zero or negative",
            sources=_refs(prev, last),
        )
    return Ratio(
        name,
        label,
        last.value / prev.value - 1,
        formula,
        "growth",
        "pct",
        sources=_refs(prev, last),
    )


def _annualised(f: Fact | None) -> float | None:
    """Year-to-date amount scaled to twelve months."""
    if f is None or f.period_start is None:
        return None
    months = round((f.period_end - f.period_start).days / 30.44)
    return None if months <= 0 else f.value * 12 / months


def _common(s: Snapshot) -> list[Ratio]:
    ni, eq, ta = (
        s.ttm.get("net_income"),
        s.balance.get("total_equity"),
        s.balance.get("total_assets"),
    )
    tl = s.balance.get("total_liabilities")
    mcap = s.market_cap
    no_mcap = "market price not available (see warnings)" if mcap is None else None
    out = [
        _div(
            "pe",
            "Price / earnings (TTM)",
            mcap,
            _v(ni),
            "market cap / TTM net income",
            "valuation",
            den_fact=ni,
            missing=no_mcap,
        ),
        _div(
            "pb",
            "Price / book",
            mcap,
            _v(eq),
            "market cap / total equity",
            "valuation",
            den_fact=eq,
            missing=no_mcap,
        ),
        _div(
            "earnings_yield",
            "Earnings yield",
            _v(ni),
            mcap,
            "TTM net income / market cap",
            "valuation",
            "pct",
            num_fact=ni,
            missing=no_mcap,
        ),
        _div(
            "roe",
            "Return on equity",
            _v(ni),
            _v(eq),
            "TTM net income / total equity",
            "profitability",
            "pct",
            num_fact=ni,
            den_fact=eq,
        ),
        _div(
            "roa",
            "Return on assets",
            _v(ni),
            _v(ta),
            "TTM net income / total assets",
            "profitability",
            "pct",
            num_fact=ni,
            den_fact=ta,
        ),
        _div(
            "liabilities_to_assets",
            "Liabilities / assets",
            _v(tl),
            _v(ta),
            "total liabilities / total assets",
            "leverage",
            "pct",
            num_fact=tl,
            den_fact=ta,
        ),
        _growth(s, "net_income", "earnings_growth", "Earnings growth (last fiscal year)"),
    ]
    if mcap is not None and s.price and s.shares:
        for r in out:
            if r.family == "valuation" and r.value is not None:
                r.sources = [*r.sources, f"PSX close {s.price_date} x {s.shares_source}"]
    return out


def bank_ratios(s: Snapshot) -> list[Ratio]:
    b, y = s.balance, s.ytd
    adv, dep, eq, ta, inv = (
        b.get(k) for k in ("advances", "deposits", "total_equity", "total_assets", "investments")
    )
    opex, tinc, nii = (
        y.get("operating_expenses"),
        y.get("total_income"),
        y.get("net_interest_income"),
    )
    return [
        *_common(s),
        _div(
            "advances_to_deposits",
            "Advances / deposits",
            _v(adv),
            _v(dep),
            "advances / deposits",
            "liquidity",
            "pct",
            num_fact=adv,
            den_fact=dep,
        ),
        _div(
            "investments_to_deposits",
            "Investments / deposits",
            _v(inv),
            _v(dep),
            "investments / deposits",
            "liquidity",
            "pct",
            num_fact=inv,
            den_fact=dep,
        ),
        _div(
            "equity_to_assets",
            "Equity / assets",
            _v(eq),
            _v(ta),
            "total equity / total assets",
            "leverage",
            "pct",
            num_fact=eq,
            den_fact=ta,
        ),
        _div(
            "cost_to_income",
            "Cost / income (year to date)",
            _v(opex),
            _v(tinc),
            "operating expenses / total income (fiscal year to date)",
            "efficiency",
            "pct",
            num_fact=opex,
            den_fact=tinc,
        ),
        _div(
            "nii_to_assets",
            "Net interest income / assets (annualised)",
            _annualised(nii),
            _v(ta),
            "annualised YTD net interest income / total assets (NIM proxy: "
            "earning assets are not reported)",
            "profitability",
            "pct",
            num_fact=nii,
            den_fact=ta,
        ),
        _growth(s, "markup_earned", "markup_growth", "Mark-up earned growth (last fiscal year)"),
    ]


def non_bank_ratios(s: Snapshot) -> list[Ratio]:
    b, t = s.balance, s.ttm
    rev, ni = t.get("revenue"), t.get("net_income")
    eq, debt, cash = b.get("total_equity"), b.get("total_borrowings"), b.get("cash_and_equivalents")
    net_debt = None if debt is None or cash is None else debt.value - cash.value
    mcap = s.market_cap
    return [
        *_common(s),
        _div(
            "ps",
            "Price / sales (TTM)",
            mcap,
            _v(rev),
            "market cap / TTM revenue",
            "valuation",
            den_fact=rev,
            missing=None if mcap is not None else "market price not available",
        ),
        _div(
            "net_margin",
            "Net margin (TTM)",
            _v(ni),
            _v(rev),
            "TTM net income / TTM revenue",
            "profitability",
            "pct",
            num_fact=ni,
            den_fact=rev,
        ),
        _div(
            "debt_to_equity",
            "Debt / equity",
            _v(debt),
            _v(eq),
            "total borrowings / total equity",
            "leverage",
            num_fact=debt,
            den_fact=eq,
        ),
        _div(
            "net_debt_to_equity",
            "Net debt / equity",
            net_debt,
            _v(eq),
            "(borrowings - cash) / total equity",
            "leverage",
            num_fact=debt,
            den_fact=eq,
        ),
        _div(
            "cash_to_assets",
            "Cash / assets",
            _v(cash),
            _v(b.get("total_assets")),
            "cash and bank balances / total assets",
            "liquidity",
            "pct",
            num_fact=cash,
            den_fact=b.get("total_assets"),
        ),
        _growth(s, "revenue", "revenue_growth", "Revenue growth (last fiscal year)"),
    ]


def compute_ratios(s: Snapshot) -> list[Ratio]:
    return bank_ratios(s) if s.company_type == "bank" else non_bank_ratios(s)


def coverage(ratios: list[Ratio]) -> float:
    """Share of ratios that could be computed."""
    return sum(r.value is not None for r in ratios) / len(ratios) if ratios else 0.0
