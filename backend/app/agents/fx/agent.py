"""FxAgent agent. Owner: Fatima. Implement analyze(); keep the AgentOutput contract."""
from app.agents.base import BaseAgent
from app.schemas.agent_output import AgentName, AgentOutput


class FxAgent(BaseAgent):
    name = AgentName.FX

    async def analyze(self, ticker: str) -> AgentOutput:
        raise NotImplementedError
