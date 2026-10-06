"""TechnicalAgent agent. Owner: Fatima. Implement analyze(); keep the AgentOutput contract."""
from app.agents.base import BaseAgent
from app.schemas.agent_output import AgentName, AgentOutput


class TechnicalAgent(BaseAgent):
    name = AgentName.TECHNICAL

    async def analyze(self, ticker: str) -> AgentOutput:
        raise NotImplementedError
