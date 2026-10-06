"""FundamentalsAgent agent. Owner: Aneeqa. Implement analyze(); keep the AgentOutput contract."""
from app.agents.base import BaseAgent
from app.schemas.agent_output import AgentName, AgentOutput


class FundamentalsAgent(BaseAgent):
    name = AgentName.FUNDAMENTALS

    async def analyze(self, ticker: str) -> AgentOutput:
        raise NotImplementedError
