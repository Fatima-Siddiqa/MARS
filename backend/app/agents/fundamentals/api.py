"""HTTP routes for the Sector & Fundamentals agent."""

from __future__ import annotations

import asyncio
import json
from datetime import date

from fastapi import APIRouter, HTTPException

from app.agents.fundamentals.agent import FundamentalsAgent, run_analysis
from app.agents.fundamentals.data.loader import companies
from app.agents.fundamentals.valuation.calibrate import load_calibration
from app.agents.fundamentals.valuation.train import REPORT_JSON
from app.schemas.agent_output import AgentOutput

router = APIRouter(prefix="/agents/fundamentals", tags=["agents"])
_agent = FundamentalsAgent()


@router.get("/model")
def model_report() -> dict:
    """Training/test report of the valuation model and the BUY/HOLD/AVOID calibration."""
    report = json.loads(REPORT_JSON.read_text()) if REPORT_JSON.exists() else None
    return {"model": report, "calibration": load_calibration()}


@router.get("/{ticker}", response_model=AgentOutput)
async def run_fundamentals(ticker: str, as_of: date | None = None) -> AgentOutput:
    return await _agent.run(ticker.upper(), as_of)


@router.get("/{ticker}/analysis")
async def full_analysis(ticker: str, as_of: date | None = None) -> dict:
    t = ticker.upper()
    if t not in companies().index:
        raise HTTPException(404, f"{t} has no official PSX fundamentals collected")
    a = await asyncio.to_thread(run_analysis, t, as_of)
    a.pop("_snapshot", None)
    return a
