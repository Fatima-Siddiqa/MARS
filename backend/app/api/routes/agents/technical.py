from fastapi import APIRouter

from app.agents.registry import AGENTS
from app.schemas.agent_output import AgentName, AgentOutput

router = APIRouter(prefix="/agents/technical", tags=["agents"])


@router.get("/{ticker}", response_model=AgentOutput)
async def run_technical(ticker: str) -> AgentOutput:
    return await AGENTS[AgentName.TECHNICAL].run(ticker.upper())
