from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

from app.schemas.agent_output import AgentOutput, Evidence, utcnow

DISCLAIMER = (
    "MARS is a decision-support tool, not financial advice. "
    "It never places trades. Investing carries risk."
)


class Call(str, Enum):
    BUY = "BUY"
    HOLD = "HOLD"
    AVOID = "AVOID"


class Regime(str, Enum):
    BULL = "BULL"
    BEAR = "BEAR"
    NEUTRAL = "NEUTRAL"


class Recommendation(BaseModel):
    ticker: str
    call: Call
    confidence: float = Field(ge=0.0, le=1.0)
    thesis: str = ""
    key_risks: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    agent_outputs: list[AgentOutput] = Field(default_factory=list)
    regime: Regime | None = None  # filled once regime detection exists
    method: str = "baseline_average"  # later: "regime_aware"
    generated_at: datetime = Field(default_factory=utcnow)
    disclaimer: str = DISCLAIMER
