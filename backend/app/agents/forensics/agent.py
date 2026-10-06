"""ForensicsAgent agent. Owner: Mariam. Implement analyze(); keep the AgentOutput contract."""
from app.agents.base import BaseAgent
from app.schemas.agent_output import AgentName, AgentOutput


class ForensicsAgent(BaseAgent):
    name = AgentName.FORENSICS

    async def analyze(self, ticker: str) -> AgentOutput:
        raise NotImplementedError
