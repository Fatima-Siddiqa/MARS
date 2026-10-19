from fastapi import APIRouter

from app.agents.registry import AGENTS
from app.schemas.agent_output import AgentName, AgentOutput

router = APIRouter(prefix="/agents/technical", tags=["agents"])


@router.get("/{ticker}", response_model=AgentOutput)
async def run_technical(ticker: str, as_of: str | None = None) -> AgentOutput:
    """`as_of` (YYYY-MM-DD) analyses the stock as of a past date; default is today."""
    return await AGENTS[AgentName.TECHNICAL].run(ticker.upper(), as_of)  # type: ignore[call-arg]
