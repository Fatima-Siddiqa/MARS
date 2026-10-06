import asyncio

from fastapi import APIRouter

from app.agents.registry import AGENTS
from app.schemas.recommendation import Recommendation
from app.synthesis.baseline_average import synthesize

router = APIRouter(prefix="/recommendation", tags=["recommendation"])


@router.get("/{ticker}", response_model=Recommendation)
async def get_recommendation(ticker: str) -> Recommendation:
    ticker = ticker.upper()
    outputs = await asyncio.gather(*(agent.run(ticker) for agent in AGENTS.values()))
    return synthesize(ticker, list(outputs))
