"""MARS Sentiment & News adapter for the shared FYP AgentOutput contract."""
import asyncio
import sqlite3
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

from app.agents.base import BaseAgent
from app.agents.sentiment.newsagent import config
from app.agents.sentiment.newsagent.engine import Engine
from app.agents.sentiment.newsagent.entities import load_companies
from app.schemas.agent_output import AgentName, AgentOutput, AgentStatus, Evidence, Stance

# Explicit aliases for the same listed company. ENGRO is deliberately not mapped to
# ENGROH: Engro Corporation and Engro Holdings should not be conflated.
TICKER_ALIASES = {"LUCK": "LUCKY", "OGDC": "OGDCL", "SYS": "SYSTEMS", "PTC": "PTCL"}


def _decision_date() -> date:
    """Use the latest decision date whose configured UTC cutoff has passed."""
    now = datetime.now(timezone.utc)
    result = now.date()
    if now.hour < config.DEFAULT_PARAMS.decision_hour_utc:
        result -= timedelta(days=1)
    return result


def _compute_raw(ticker: str) -> dict:
    db_path = Path(config.DB_PATH)
    if not db_path.is_file():
        raise FileNotFoundError(f"Sentiment database not found: {db_path}")
    conn = sqlite3.connect(str(db_path), timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        # The database already contains FinBERT inference scores. The API request
        # computes the aggregate and does not load/download FinBERT itself.
        return Engine(conn).output(ticker, _decision_date())
    finally:
        conn.close()


def _map_output(raw: dict, requested_ticker: str) -> AgentOutput:
    state = str(raw.get("state", "Neutral")).casefold()
    stance = {"bullish": Stance.BULLISH, "bearish": Stance.BEARISH}.get(state, Stance.NEUTRAL)
    quality = float(raw.get("quality") or 0.0)
    context = raw.get("context") or {}
    flags = list(raw.get("flags") or [])
    scorers = list(raw.get("scorer") or [])
    warnings = [f"Sentiment flag: {flag}" for flag in flags]
    if any(s != "finbert" for s in scorers):
        warnings.append(f"Non-FinBERT scorer reported: {', '.join(scorers)}")
    if quality < 0.30:
        warnings.append("Low sentiment data quality; interpret the score cautiously.")
    n_articles = int(context.get("articles_in_window") or 0)
    window_days = int(context.get("window_days") or config.DEFAULT_PARAMS.window_days)
    score = float(raw.get("score") or 0.0)
    state_label = str(raw.get("state") or "Neutral")
    summary = (
        f"{requested_ticker} sentiment is {state_label} (score {score:+.2f}) based on "
        f"{n_articles} distinct articles in the last {window_days} days."
    )
    if "no_recent_news" in flags or n_articles == 0:
        summary = (f"No recent company-specific articles were found for {requested_ticker}; "
                   "the score may rely on sector/market sentiment or neutral fallback.")

    driver_labels = {
        "net_tone": "Overall news tone",
        "tone_change": "Change in news tone",
        "negative_share": "Share of negative news",
    }
    findings = []
    for d in raw.get("drivers") or []:
        label = driver_labels.get(d.get("name"), str(d.get("name", "Sentiment driver")).replace("_", " ").title())
        findings.append(
            f"{label}: {d.get('direction', 'neutral')} (value {d.get('value')}, z-score {d.get('z')})."
        )
    if not findings:
        findings.append(f"Sentiment state is {state_label}; quality score is {quality:.2f}.")

    evidence = []
    for item in raw.get("evidence") or []:
        title = str(item.get("title") or "Untitled article")
        outlet = str(item.get("outlet") or "Unknown source")
        url = item.get("url")
        tags = item.get("event_tags") or []
        tone = item.get("tone")
        excerpt_parts = []
        if tone is not None:
            excerpt_parts.append(f"Adjusted article tone: {tone}")
        if tags:
            excerpt_parts.append("Event tags: " + ", ".join(str(x) for x in tags))
        evidence.append(Evidence(
            claim=title,
            source_name=outlet,
            source_url=url,
            source_type=str(item.get("kind") or "news"),
            excerpt="; ".join(excerpt_parts) or None,
        ))

    degraded_flags = {"no_recent_news", "low_coverage", "insufficient_history"}
    status = AgentStatus.DEGRADED if any(f in degraded_flags for f in flags) or quality < 0.30 else AgentStatus.OK
    metrics = {
        "quality": quality,
        "horizon_days": raw.get("horizon_days"),
        "articles_in_window": n_articles,
        "effective_articles": context.get("effective_articles", 0.0),
        "window_days": window_days,
        "agreement": context.get("agreement"),
        "attention_z": context.get("attention_z"),
        "parent": context.get("parent"),
        "parent_score": context.get("parent_score"),
        "model_version": raw.get("model_version"),
        "scorer": scorers,
        "drivers": raw.get("drivers") or [],
        "flags": flags,
    }
    as_of_text = str(raw.get("as_of") or _decision_date().isoformat())
    as_of = datetime.combine(date.fromisoformat(as_of_text), time.min, tzinfo=timezone.utc)
    return AgentOutput(
        agent=AgentName.SENTIMENT,
        ticker=requested_ticker,
        as_of=as_of,
        status=status,
        stance=stance,
        score=max(-1.0, min(1.0, score)),
        # The shared UI has one reliability field. For sentiment this is the
        # uncalibrated MARS quality index, shown in the UI as "Data quality".
        confidence=max(0.0, min(1.0, quality)),
        summary=summary,
        key_findings=findings,
        risks=[w for w in warnings if "Low sentiment data quality" in w or "Non-FinBERT" in w],
        evidence=evidence,
        metrics=metrics,
        warnings=warnings,
    )


class SentimentAgent(BaseAgent):
    name = AgentName.SENTIMENT

    async def analyze(self, ticker: str) -> AgentOutput:
        requested = ticker.strip().upper()
        actual = TICKER_ALIASES.get(requested, requested)
        supported = {row["ticker"] for row in load_companies()}
        if actual not in supported:
            return AgentOutput(
                agent=AgentName.SENTIMENT,
                ticker=requested,
                status=AgentStatus.UNAVAILABLE,
                warnings=[f"Ticker {requested} is not supported by the sentiment agent."],
            )
        raw = await asyncio.to_thread(_compute_raw, actual)
        out = _map_output(raw, requested)
        if actual != requested:
            out.warnings.append(f"Mapped FYP ticker alias {requested} to sentiment ticker {actual}.")
        return out