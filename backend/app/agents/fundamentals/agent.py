"""Agent 5: Sector & Fundamental Comparison. Owner: Aneeqa.

Pipeline (official PSX data only, no LLM):
    point-in-time snapshot -> ratios -> PSX sector peers -> valuation model (trained on all
    PSX companies, see valuation/) -> calibrated BUY / HOLD / AVOID call -> AgentOutput.

The call compares the model's fair value with the market price:
    gap >= T  -> BUY (bullish)     gap <= -T (log-symmetric) -> AVOID (bearish)     else HOLD
T comes from valuation/calibrate.py (tested on out-of-sample predictions vs later returns).
If no model is available the transparent peer baseline is used instead (and confidence drops).

The call lives in `metrics["call"]`, and the shared contract's stance/score/confidence carry the
same decision, so synthesis can weigh it. It is decision support, never a trade instruction.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import asdict, is_dataclass
from datetime import date

from app.agents.base import BaseAgent
from app.agents.fundamentals.data.loader import Snapshot, build_snapshot, companies
from app.agents.fundamentals.features import Ratio, compute_ratios, coverage
from app.agents.fundamentals.peers import PeerResult, baseline, compare, peer_universe
from app.agents.fundamentals.valuation.calibrate import load_calibration
from app.agents.fundamentals.valuation.predict import Valuation, value
from app.ingestion.psx_eod import BASE_URL as PSX_DAILY_URL
from app.schemas.agent_output import AgentName, AgentOutput, AgentStatus, Evidence, Stance

CALLS = {"BUY": Stance.BULLISH, "HOLD": Stance.NEUTRAL, "AVOID": Stance.BEARISH}
PEER_FACTOR = {"adequate": 1.0, "weak": 0.8, "none": 0.6}
BASELINE_FIT = 0.3  # confidence factor when only the peer baseline is available


def decide(gap: float | None, threshold: float) -> tuple[str, float]:
    """(call, score in [-1, 1]) from a valuation gap (implied / price - 1)."""
    if gap is None or gap <= -1:
        return "HOLD", 0.0
    g, t = math.log1p(gap), math.log1p(threshold)
    call = "BUY" if g >= t else "AVOID" if g <= -t else "HOLD"
    return call, max(-1.0, min(1.0, g / (2 * t)))


def confidence(
    call: str,
    gap: float | None,
    threshold: float,
    ratio_cov: float,
    fit: float,
    supported: bool,
    peer_strength: str,
) -> float:
    """ratio coverage x model fit x evidence x peers x margin (see docs/fundamentals_agent.md).

    margin: BUY/AVOID 0 at the threshold -> 1 at twice the threshold; HOLD 1 at a zero gap ->
    0 at the threshold.
    """
    if gap is None or gap <= -1:
        return 0.0
    m = abs(math.log1p(gap)) / math.log1p(threshold)
    margin = min(1.0, max(0.0, m - 1.0)) if call != "HOLD" else max(0.0, 1.0 - m)
    evidence = 1.0 if supported else 0.5
    c = ratio_cov * fit * evidence * PEER_FACTOR[peer_strength] * (0.5 + 0.5 * margin)
    return round(max(0.0, min(0.95, c)), 3)


def _plain(x):
    if is_dataclass(x):
        return _plain(asdict(x))
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, date):
        return x.isoformat()
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return x


def run_analysis(ticker: str, as_of: date | None = None) -> dict:
    """Full analysis as plain data (used by the API page and by the AgentOutput)."""
    s: Snapshot = build_snapshot(ticker, as_of)
    ratios: list[Ratio] = compute_ratios(s)
    uni = peer_universe(s)
    uni[s.ticker] = (s, ratios)
    peers: PeerResult = compare(s, uni)
    base = baseline(s, ratios, peers)
    val: Valuation = value(s)
    cal = load_calibration()
    t = float(cal.get("threshold", 0.25))
    if val.gap is not None:
        gap, source = val.gap, f"model ({val.method})"
        fit = math.exp(-val.test_metrics.get("mae_log", 1.0))
    elif base.gap is not None:
        gap, source, fit = base.gap, f"peer baseline ({base.method})", BASELINE_FIT
    else:
        gap, source, fit = None, "none", 0.0
    call, score = decide(gap, t)
    conf = confidence(
        call, gap, t, coverage(ratios), fit, bool(cal.get("supported")), peers.strength
    )
    meta = companies().loc[s.ticker]
    return _plain(
        {
            "ticker": s.ticker,
            "as_of": s.as_of,
            "call": call,
            "score": score,
            "confidence": conf,
            "gap": gap,
            "gap_source": source,
            "threshold": t,
            "calibration": {
                k: cal.get(k)
                for k in (
                    "supported",
                    "threshold",
                    "reason",
                    "spearman_ic",
                    "ic_p_value",
                    "samples",
                )
            },
            "company": {
                "type": s.company_type,
                "sector": s.sector,
                "sector_code": s.sector_code,
                "coverage": meta.get("coverage"),
            },
            "market": {
                "price": s.price,
                "price_date": s.price_date,
                "shares": s.shares,
                "shares_source": s.shares_source,
                "market_cap": s.market_cap,
            },
            "statements": {
                "balance_date": s.balance_date,
                "balance": {k: f.value for k, f in s.balance.items()},
                "ttm": {
                    k: {"value": f.value, "period_end": f.period_end, "method": s.ttm_method.get(k)}
                    for k, f in s.ttm.items()
                },
            },
            "ratios": ratios,
            "ratio_coverage": coverage(ratios),
            "peers": peers,
            "peer_baseline": base,
            "valuation": val,
            "warnings": s.warnings,
            "_snapshot": s,
        }
    )


def _evidence(a: dict, s: dict) -> list[Evidence]:
    ev = []
    m = a["market"]
    if m["price"] is not None:
        ev.append(
            Evidence(
                claim=f"Price Rs {m['price']:,.2f} on {m['price_date']}; market cap "
                f"PKR {m['market_cap']:,.0f}m ({m['shares_source']}).",
                source_name="PSX Data Portal daily market summary",
                source_url=PSX_DAILY_URL.format(d=m["price_date"]),
                source_type="price_data",
            )
        )
    bal = s["balance"]
    if "total_equity" in bal:
        f = bal["total_equity"]
        ev.append(
            Evidence(
                claim=f"Total equity PKR {f['value']:,.0f}m at {f['period_end']}.",
                source_name=f["source"],
                source_url=f["source_ref"].split(" | ")[0],
                source_type="filing",
            )
        )
    if "net_income" in s["ttm"]:
        f = s["ttm"]["net_income"]
        ev.append(
            Evidence(
                claim=f"Trailing twelve-month profit after tax PKR {f['value']:,.0f}m "
                f"(to {f['period_end']}).",
                source_name=f["source"],
                source_url=f["source_ref"].split(" | ")[0],
                source_type="filing",
                excerpt=s["ttm_method"].get("net_income"),
            )
        )
    v = a["valuation"]
    if v["implied_value_per_share"] is not None:
        tm = v["test_metrics"]
        ev.append(
            Evidence(
                claim=f"Model fair value Rs {v['implied_value_per_share']:,.2f} per share "
                f"({v['method']}, {v['model_version']}).",
                source_name="MARS fundamentals valuation model (trained on official PSX data)",
                source_type="model",
                excerpt=f"Walk-forward test: median absolute error "
                f"{tm.get('median_abs_pct_error', float('nan')):.0%}, R² "
                f"{tm.get('r2', float('nan')):.2f} on {tm.get('n')} unseen company-years.",
            )
        )
    p = a["peers"]
    if p["peers"]:
        ev.append(
            Evidence(
                claim=f"Compared with {len(p['peers'])} PSX peers in {a['company']['sector']} "
                f"({p['strength']} peer group).",
                source_name="PSX Data Portal company pages",
                source_url=f"https://dps.psx.com.pk/company/{p['peers'][0]}",
                source_type="filing",
                excerpt=", ".join(p["peers"][:12]),
            )
        )
    return ev


def to_agent_output(a: dict) -> AgentOutput:
    s = a.pop("_snapshot")
    rmap = {r["name"]: r for r in a["ratios"]}
    v, call = a["valuation"], a["call"]
    if a["gap"] is None:
        status = AgentStatus.DEGRADED if a["ratio_coverage"] > 0 else AgentStatus.UNAVAILABLE
        summary = f"{a['ticker']}: HOLD (no valuation possible). " + "; ".join(
            a["warnings"] or [v.get("reason") or "insufficient data"]
        )
    else:
        status = AgentStatus.OK if not a["warnings"] else AgentStatus.DEGRADED
        fair = v["implied_value_per_share"] or a["peer_baseline"]["implied_value_per_share"]
        summary = (
            f"{a['ticker']}: {call}. Fair value Rs {fair:,.2f} per share vs price "
            f"Rs {a['market']['price']:,.2f} ({a['gap']:+.0%}, {a['gap_source']}); "
            f"threshold ±{a['threshold']:.0%}."
        )
    findings = []
    for name in (
        "pe",
        "pb",
        "roe",
        "net_margin",
        "cost_to_income",
        "advances_to_deposits",
        "debt_to_equity",
        "earnings_growth",
    ):
        r = rmap.get(name)
        if r and r["value"] is not None:
            val = f"{r['value']:.1%}" if r["unit"] == "pct" else f"{r['value']:.2f}x"
            st = a["peers"]["stats"].get(name)
            peer = (
                f" (peer median {st['peer_median']:.1%})"
                if st and st["peer_median"] is not None and r["unit"] == "pct"
                else f" (peer median {st['peer_median']:.2f}x)"
                if st and st["peer_median"] is not None
                else ""
            )
            findings.append(f"{r['label']}: {val}{peer}")
    for c in v["contributions"][:3]:
        findings.append(f"Model driver: {c['label']} {c['effect_pct']:+.0%} on fair value")
    risks = list(a["warnings"])
    if not a["calibration"].get("supported"):
        risks.append(
            "BUY/HOLD/AVOID threshold not confirmed by historical returns (confidence reduced)."
        )
    if a["peers"]["strength"] != "adequate":
        risks.append(f"Peer group is {a['peers']['strength']} ({len(a['peers']['peers'])} peers).")
    if a["company"]["coverage"] and "no filed" in a["company"]["coverage"]:
        risks.append("No filed balance sheet keyed for this company: balance-sheet ratios missing.")
    de = rmap.get("debt_to_equity")
    if de and de["value"] is not None and de["value"] > 2:
        risks.append(f"High leverage: debt / equity {de['value']:.1f}x.")
    metrics = {
        k: a[k]
        for k in (
            "call",
            "gap",
            "gap_source",
            "threshold",
            "calibration",
            "company",
            "market",
            "statements",
            "ratio_coverage",
        )
    }
    metrics.update(
        valuation=v,
        peer_baseline=a["peer_baseline"],
        peers={
            "strength": a["peers"]["strength"],
            "peers": a["peers"]["peers"],
            "stats": a["peers"]["stats"],
        },
        ratios=a["ratios"],
    )
    return AgentOutput(
        agent=AgentName.FUNDAMENTALS,
        ticker=a["ticker"],
        status=status,
        stance=CALLS[call],
        score=round(a["score"], 4),
        confidence=a["confidence"],
        summary=summary,
        key_findings=findings,
        risks=risks,
        evidence=_evidence(a, s),
        metrics=metrics,
        warnings=a["warnings"],
    )


class FundamentalsAgent(BaseAgent):
    name = AgentName.FUNDAMENTALS

    async def run(self, ticker: str, as_of: date | None = None) -> AgentOutput:  # type: ignore[override]
        if as_of is None:
            return await super().run(ticker)
        try:
            return await self.analyze(ticker, as_of)
        except Exception as exc:  # noqa: BLE001
            return AgentOutput(
                agent=self.name,
                ticker=ticker,
                status=AgentStatus.ERROR,
                warnings=[f"{type(exc).__name__}: {exc}"],
            )

    async def analyze(self, ticker: str, as_of: date | None = None) -> AgentOutput:
        ticker = ticker.upper()
        if ticker not in companies().index:
            return AgentOutput(
                agent=self.name,
                ticker=ticker,
                status=AgentStatus.UNAVAILABLE,
                warnings=[f"{ticker} has no official PSX fundamentals collected"],
            )
        a = await asyncio.to_thread(run_analysis, ticker, as_of)
        return to_agent_output(a)
