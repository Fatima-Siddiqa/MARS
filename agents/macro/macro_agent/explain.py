"""Layer 6 - explanation and evidence. Deterministic templates built from a facts dict. The optional LLM (llm.py)
may only reword; it never produces numbers."""
from __future__ import annotations

import math

from .config import Config

# value formatting by unit ---------------------------------------------------------------------------------------------
def display_value(value: float, unit: str) -> float:
    """The number a reader sees (percentile is shown 0-100)."""
    return round(value * 100, 0) if unit == "percentile" else round(value, 1 if abs(value) < 1000 else 0)


def fmt_value(value: float, unit: str) -> str:
    d = display_value(value, unit)
    if unit == "percent":
        return f"{d:.1f}%"
    if unit == "pp":
        return f"{d:+.1f} percentage points"
    if unit == "percentile":
        return f"the {d:.0f}th percentile"
    if unit == "USD million":
        return f"{d:,.0f} million US dollars"
    return f"{d} {unit}"


# layman templates: {core} describes the thing; the pos/neg tail is added from the sign of the contribution ---------------
LAYMAN = {
    "rate_change_6m": "The policy rate is the interest rate the State Bank sets; when it rises, loans cost more and companies usually earn less. It has changed by {v} over six months.",
    "rate_vs_12m_avg": "The policy rate stands {v} relative to its 12-month average; rates above their recent norm make borrowing more expensive.",
    "yield_spread": "Short-term government borrowing costs are {v} compared with the policy rate, a hint of where markets think rates are heading.",
    "cpi_level": "Inflation is how fast prices rise. At {v} a year, it eats into household budgets and company margins when it is high.",
    "cpi_trend": "Inflation has moved by {v} compared with the previous three months, showing whether price pressure is building or easing.",
    "pkr_dep_30d": "The rupee has lost {v} against the US dollar over 30 days; a weaker rupee makes imports, fuel and foreign debt dearer.",
    "pkr_dep_90d": "Over 90 days the rupee has lost {v} against the dollar, which raises the cost of imports and foreign debt.",
    "pkr_vol_30d": "The rupee has been swinging by {v} a day recently; unstable currencies make planning harder for businesses.",
    "reserves_chg_3m": "The State Bank's foreign-currency reserves changed by {v} over three months; larger reserves help the country pay for imports and debt.",
    "ca_3m_sum": "Over three months the current account (trade and transfers with the world) came to {v}; surpluses ease pressure on the rupee.",
    "remittances_yoy": "Money sent home by overseas Pakistanis changed by {v} from a year ago; it is a key source of dollars.",
    "brent_3m": "Oil prices have changed by {v} over three months; Pakistan imports oil, so higher prices raise costs for most firms.",
    "brent_pct_252": "Oil prices are at {v} of the past year's range; high prices mean a bigger import bill.",
    "gold_3m": "Gold has changed by {v} over three months; it often rises when investors are nervous.",
    "gdp_growth": "The economy grew {v} in the latest year; faster growth generally helps company earnings.",
}
TAIL = {"bullish": " This is currently a positive for shares.", "bearish": " This is currently a negative for shares."}
STATE_WORD = {"Bullish": "bullish", "Neutral": "neutral", "Bearish": "bearish"}


def _series_text(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def build_facts(*, state: str, score: float, confidence: float, as_of: str, drivers: list[dict],
                pillars: dict, warnings: list[str], ticker: str | None, sector: str | None) -> dict:
    return dict(state=state, score=round(score, 2), confidence=round(confidence, 2), as_of=as_of, ticker=ticker,
                sector=sector, drivers=drivers, pillars={k: (None if v is None else round(v, 2)) for k, v in pillars.items()},
                warnings=warnings)


def template_sentence(facts: dict) -> str:
    top = facts["drivers"][:3]
    names = [d["label"] for d in top]
    state = STATE_WORD[facts["state"]]
    if not names:
        return f"Macro conditions look {state}, but there is not enough data to name the main drivers."
    verb = "had" if len(names) > 1 else "had"
    return f"Macro conditions look {state}: {_series_text(names)} {verb} the largest effect."


def plain_summary(facts: dict) -> str:
    parts = [template_sentence(facts)]
    for d in facts["drivers"][:3]:
        tpl = LAYMAN.get(d["feature"])
        if tpl:
            parts.append(tpl.format(v=fmt_value(d["value"], d["unit"])) + TAIL[d["direction"]])
    if facts["ticker"] and facts.get("sector"):
        parts.append(f"For {facts['ticker']} ({facts['sector'].replace('_', ' ')}) some effects are adjusted for how this sector reacts.")
    if facts["warnings"]:
        parts.append("Please note: " + " ".join(w.rstrip(".") + "." for w in facts["warnings"][:3]))
    return " ".join(parts)


def pick_drivers(cfg: Config, contrib: dict[str, float], raw: dict[str, float], k: int = 3) -> list[dict]:
    """Top-k features by absolute contribution to the composite."""
    items = [(f, c) for f, c in contrib.items() if c is not None and not math.isnan(c)]
    items.sort(key=lambda t: abs(t[1]), reverse=True)
    out = []
    for f, c in items[:k]:
        fc = cfg.features[f]
        out.append(dict(feature=f, label=fc.label, contribution=round(c, 4), value=float(raw[f]), unit=fc.unit,
                        direction="bullish" if c > 0 else "bearish"))
    return out
