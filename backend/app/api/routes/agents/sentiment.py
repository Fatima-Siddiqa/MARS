from fastapi import APIRouter

from app.agents.registry import AGENTS
from app.schemas.agent_output import AgentName, AgentOutput

router = APIRouter(prefix="/agents/sentiment", tags=["agents"])


@router.get("/{ticker}", response_model=AgentOutput)
async def run_sentiment(ticker: str) -> AgentOutput:
    return await AGENTS[AgentName.SENTIMENT].run(ticker.upper())
