from app.agents.sentiment.agent import _map_output
from app.schemas.agent_output import AgentStatus, Stance


def test_sentiment_output_maps_to_shared_contract():
    out = _map_output({
        "ticker": "UBL", "as_of": "2026-10-15", "score": -0.55,
        "state": "Bearish", "quality": 0.42, "horizon_days": 20,
        "drivers": [{"name": "net_tone", "value": -0.2, "z": -2.0, "direction": "bearish"}],
        "evidence": [{"title": "UBL reports results", "outlet": "Dawn", "url": "https://example.com", "kind": "news", "tone": -0.3}],
        "context": {"articles_in_window": 4, "effective_articles": 3.5, "window_days": 14},
        "flags": [], "scorer": ["finbert"], "model_version": "sentiment-news-1.0",
    }, "UBL")
    assert out.agent.value == "sentiment"
    assert out.stance == Stance.BEARISH
    assert out.score == -0.55
    assert out.confidence == 0.42
    assert out.status == AgentStatus.OK
    assert out.evidence[0].source_name == "Dawn"
    assert out.evidence[0].source_url == "https://example.com"
    assert out.metrics["quality"] == 0.42


def test_low_quality_marks_output_degraded():
    out = _map_output({
        "ticker": "HUBC", "as_of": "2026-10-15", "score": 0.0,
        "state": "Neutral", "quality": 0.1, "context": {"articles_in_window": 1, "window_days": 14},
        "flags": ["low_coverage"], "scorer": ["finbert"], "evidence": [], "drivers": [],
    }, "HUBC")
    assert out.status == AgentStatus.DEGRADED
    assert any("Low sentiment data quality" in warning for warning in out.warnings)
