"""SHARED CONTRACT: every specialist agent returns an `AgentOutput`.

Changing this file affects all agents + synthesis + frontend.
Open a PR, tag all three members, and update docs/agent_contract.md.
"""
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

SCHEMA_VERSION = "1.0"


def utcnow() -> datetime:
    return datetime.now(UTC)


class AgentName(str, Enum):
    TECHNICAL = "technical"        # Historical Price & Technical
    SENTIMENT = "sentiment"        # Sentiment & News
    MACRO = "macro"                # Macroeconomic Indicators (SBP, inflation)
    FX = "fx"                      # FX Impact (PKR)
    FUNDAMENTALS = "fundamentals"  # Sector & Fundamental Comparison
    FORENSICS = "forensics"        # Financial Forensics


class Stance(str, Enum):
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"


class AgentStatus(str, Enum):
    OK = "ok"
    DEGRADED = "degraded"        # partial data / fallback used
    UNAVAILABLE = "unavailable"  # agent not implemented or data feed down
    ERROR = "error"


class Evidence(BaseModel):
    """A citable source backing one claim (every claim needs one)."""

    claim: str
    source_name: str
    source_url: str | None = None
    source_type: str | None = None  # e.g. "price_data", "news", "filing", "macro_feed"
    excerpt: str | None = None
    retrieved_at: datetime = Field(default_factory=utcnow)


class AgentOutput(BaseModel):
    schema_version: str = SCHEMA_VERSION
    agent: AgentName
    ticker: str
    as_of: datetime = Field(default_factory=utcnow)
    status: AgentStatus = AgentStatus.OK

    stance: Stance = Stance.NEUTRAL
    score: float = Field(0.0, ge=-1.0, le=1.0, description="-1 very bearish ... +1 very bullish")
    confidence: float = Field(0.0, ge=0.0, le=1.0)

    summary: str = ""
    key_findings: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)

    metrics: dict[str, Any] = Field(default_factory=dict, description="Agent-specific numbers")
    warnings: list[str] = Field(default_factory=list)
    latency_ms: int | None = None
