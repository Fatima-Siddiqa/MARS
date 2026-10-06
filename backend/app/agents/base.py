import logging
import time
from abc import ABC, abstractmethod

from app.schemas.agent_output import AgentName, AgentOutput, AgentStatus

log = logging.getLogger(__name__)


class BaseAgent(ABC):
    """All six specialist agents subclass this and implement `analyze`.

    Use `run()` (not `analyze()`) from routes/orchestration: it times the call and
    degrades gracefully, so one failing agent never crashes a full analysis.
    """

    name: AgentName

    @abstractmethod
    async def analyze(self, ticker: str) -> AgentOutput:
        """Return an AgentOutput for `ticker`. Raise NotImplementedError until built."""

    async def run(self, ticker: str) -> AgentOutput:
        start = time.perf_counter()
        try:
            out = await self.analyze(ticker)
        except NotImplementedError:
            out = AgentOutput(
                agent=self.name,
                ticker=ticker,
                status=AgentStatus.UNAVAILABLE,
                warnings=["Agent not implemented yet"],
            )
        except Exception as exc:  # noqa: BLE001
            log.exception("Agent %s failed for %s", self.name.value, ticker)
            out = AgentOutput(
                agent=self.name,
                ticker=ticker,
                status=AgentStatus.ERROR,
                warnings=[f"{type(exc).__name__}: {exc}"],
            )
        out.latency_ms = int((time.perf_counter() - start) * 1000)
        return out
