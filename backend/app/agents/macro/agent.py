"""MacroAgent: adapter between the standalone macro_agent service and the shared AgentOutput contract.

Owner: Mariam. The scoring logic lives in agents/macro (run with `macro-agent serve --port 8010`).
This file only calls that service and maps its JSON to `AgentOutput`; it does not change the contract.
Set MACRO_AGENT_URL if the service is not on http://localhost:8010.
"""
import os
from datetime import UTC, datetime
from urllib.parse import urlparse

import httpx

from app.agents.base import BaseAgent
from app.schemas.agent_output import AgentName, AgentOutput, AgentStatus, Evidence, Stance

MACRO_URL = os.getenv("MACRO_AGENT_URL", "http://localhost:8010")
TIMEOUT_S = float(os.getenv("MACRO_AGENT_TIMEOUT", "20"))

_SOURCE_NAMES = {
    "easydata.sbp.org.pk": "SBP EasyData",
    "www.sbp.org.pk": "State Bank of Pakistan",
    "www.pbs.gov.pk": "Pakistan Bureau of Statistics",
    "fred.stlouisfed.org": "FRED (St. Louis Fed)",
    "api.worldbank.org": "World Bank",
    "data.worldbank.org": "World Bank",
    "www.worldbank.org": "World Bank",
    "dps.psx.com.pk": "Pakistan Stock Exchange",
}


def _source_name(url: str | None) -> str:
    host = urlparse(url or "").netloc.lower()
    return _SOURCE_NAMES.get(host, host or "macro data source")


def _pretty(key: str) -> str:
    return key.replace("_", " ")


def _parse_date(text: str | None) -> datetime:
    try:
        return datetime.fromisoformat(str(text)).replace(tzinfo=UTC)
    except (TypeError, ValueError):
        return datetime.now(UTC)


def map_macro_output(p: dict, ticker: str) -> AgentOutput:
    """Convert the macro service JSON (methodology section 9) into the shared AgentOutput."""
    drivers = p.get("drivers", [])
    warnings = list(p.get("warnings", []))
    indicators = []
    for d in drivers:
        label = d.get("label") or _pretty(d.get("feature", ""))
        if d.get("value") is not None:
            unit = d.get("unit") or ""
            indicators.append({"label": label, "value": f"{d['value']:.2f} {unit}".strip()
                               + (f" ({d['direction']})" if d.get("direction") else "")})
    evidence = [
        Evidence(
            claim=f"{_pretty(e['series_key'])}: {e['value']:.4g} {e.get('unit', '')} as of {e['date']}".strip(),
            source_name=_source_name(e.get("source_url")),
            source_url=e.get("source_url"),
            source_type="macro_feed",
            excerpt=f"series key: {e['series_key']}",
        )
        for e in p.get("evidence", [])
    ]
    return AgentOutput(
        agent=AgentName.MACRO,
        ticker=ticker,
        as_of=_parse_date(p.get("as_of")),
        status=AgentStatus.DEGRADED if warnings else AgentStatus.OK,
        stance=Stance(str(p.get("state", "Neutral")).lower()),
        score=max(-1.0, min(1.0, float(p.get("score", 0.0)))),
        confidence=max(0.0, min(1.0, float(p.get("confidence", 0.0)))),
        summary=p.get("plain_summary", ""),
        key_findings=[
            f"{d.get('label') or _pretty(d['feature'])}: {d.get('direction', 'neutral')} "
            f"(contribution {d['contribution']:+.2f})"
            for d in drivers[:3]
        ],
        risks=[f"{d.get('label') or _pretty(d['feature'])} is a headwind"
               for d in drivers if d.get("direction") == "bearish"],
        evidence=evidence,
        metrics={
            "pillars": p.get("pillars", {}),
            "indicators": indicators,  # label/value pairs, ready for the UI's indicator tiles
            "horizon_days": p.get("horizon_days"),
            "model_version": p.get("model_version"),
        },
        warnings=warnings,
    )


class MacroAgent(BaseAgent):
    name = AgentName.MACRO

    async def analyze(self, ticker: str) -> AgentOutput:
        try:
            async with httpx.AsyncClient(timeout=TIMEOUT_S) as client:
                r = await client.get(f"{MACRO_URL}/agents/macro", params={"ticker": ticker})
        except httpx.HTTPError as exc:
            return AgentOutput(agent=self.name, ticker=ticker, status=AgentStatus.UNAVAILABLE,
                               warnings=[f"Macro service not reachable at {MACRO_URL}: {type(exc).__name__}"])
        if r.status_code != 200:
            return AgentOutput(agent=self.name, ticker=ticker, status=AgentStatus.UNAVAILABLE,
                               warnings=[f"Macro service returned {r.status_code}: {r.text[:200]}"])
        return map_macro_output(r.json(), ticker)