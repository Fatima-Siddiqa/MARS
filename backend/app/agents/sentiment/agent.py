"""SentimentAgent agent. Owner: Aneeqa. Implement analyze(); keep the AgentOutput contract."""
from app.agents.base import BaseAgent
from app.schemas.agent_output import AgentName, AgentOutput


class SentimentAgent(BaseAgent):
    name = AgentName.SENTIMENT

    async def analyze(self, ticker: str) -> AgentOutput:
        raise NotImplementedError
