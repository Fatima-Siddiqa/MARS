from fastapi import APIRouter

from app.agents.registry import AGENTS
from app.schemas.agent_output import AgentName, AgentOutput

router = APIRouter(prefix="/agents/macro", tags=["agents"])


@router.get("/{ticker}", response_model=AgentOutput)
async def run_macro(ticker: str) -> AgentOutput:
    return await AGENTS[AgentName.MACRO].run(ticker.upper())
