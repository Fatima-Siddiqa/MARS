"""Starter tables. Extend via Alembic migrations (alembic revision --autogenerate)."""
from datetime import UTC, datetime

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base


def _now() -> datetime:
    return datetime.now(UTC)


class AnalysisRun(Base):
    """One user request: ticker in -> recommendation out."""

    __tablename__ = "analysis_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    recommendation: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    agent_results: Mapped[list["AgentResult"]] = relationship(back_populates="run")


class AgentResult(Base):
    """Stored AgentOutput (see schemas/agent_output.py) for one agent in one run."""

    __tablename__ = "agent_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("analysis_runs.id"), index=True)
    agent: Mapped[str] = mapped_column(String(32), index=True)
    output: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    run: Mapped[AnalysisRun] = relationship(back_populates="agent_results")
