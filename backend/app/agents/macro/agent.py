"""MacroAgent agent. Owner: Mariam. Implement analyze(); keep the AgentOutput contract."""
from app.agents.base import BaseAgent
from app.schemas.agent_output import AgentName, AgentOutput


class MacroAgent(BaseAgent):
    name = AgentName.MACRO

    async def analyze(self, ticker: str) -> AgentOutput:
        raise NotImplementedError
