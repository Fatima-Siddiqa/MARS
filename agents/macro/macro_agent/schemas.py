"""Output contract (methodology section 9). Extra keys inside drivers/evidence are additive."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Driver(BaseModel):
    feature: str
    contribution: float
    label: str | None = None
    value: float | None = None
    unit: str | None = None
    score: float | None = None
    direction: Literal["bullish", "bearish"] | None = None


class Evidence(BaseModel):
    series_key: str
    value: float
    date: str
    unit: str
    source_url: str
    used_by: list[str] = Field(default_factory=list)


class MacroOutput(BaseModel):
    agent: str = "macro"
    ticker: str | None = None
    score: float
    state: Literal["Bullish", "Neutral", "Bearish"]
    confidence: float
    pillars: dict[str, float | None]
    drivers: list[Driver]
    evidence: list[Evidence]
    warnings: list[str]
    plain_summary: str
    as_of: str
    horizon_days: int = 20
    model_version: str
