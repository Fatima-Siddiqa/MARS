from app.agents.base import BaseAgent
from app.agents.forensics.agent import ForensicsAgent
from app.agents.fundamentals.agent import FundamentalsAgent
from app.agents.fx.agent import FxAgent
from app.agents.macro.agent import MacroAgent
from app.agents.sentiment.agent import SentimentAgent
from app.agents.technical.agent import TechnicalAgent
from app.schemas.agent_output import AgentName

AGENTS: dict[AgentName, BaseAgent] = {
    a.name: a
    for a in (
        TechnicalAgent(),
        SentimentAgent(),
        MacroAgent(),
        FxAgent(),
        FundamentalsAgent(),
        ForensicsAgent(),
    )
}
