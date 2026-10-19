from app.agents.macro.agent import MacroAgent, map_macro_output
from app.schemas.agent_output import AgentName, AgentOutput, AgentStatus, Stance

SAMPLE = {
    "agent": "macro", "ticker": "OGDC", "score": 0.0252, "state": "Neutral", "confidence": 0.6,
    "pillars": {"monetary": 0.162, "inflation": -0.2613, "external": 0.3032, "commodity": -0.5394, "growth": 0.4615},
    "drivers": [
        {"feature": "gdp_growth", "contribution": 0.0923, "label": "economic growth", "value": 4.6149,
         "unit": "percent", "score": 0.4615, "direction": "bullish"},
        {"feature": "brent_3m", "contribution": -0.1, "label": "oil price move", "value": 9.0,
         "unit": "percent", "score": -0.5, "direction": "bearish"},
    ],
    "evidence": [{"series_key": "policy_rate", "value": 11.5, "date": "2026-04-28", "unit": "percent",
                  "source_url": "https://easydata.sbp.org.pk/", "used_by": ["rate_change_6m"]}],
    "warnings": ["DEMO DATA: synthetic series"], "plain_summary": "Macro conditions look neutral.",
    "as_of": "2026-10-15", "horizon_days": 20, "model_version": "test",
}


def test_mapping_matches_contract():
    out = map_macro_output(SAMPLE, "OGDC")
    assert isinstance(out, AgentOutput) and out.agent == AgentName.MACRO
    assert out.stance == Stance.NEUTRAL and -1 <= out.score <= 1 and 0 <= out.confidence <= 1
    assert out.status == AgentStatus.DEGRADED          # warnings present
    assert out.evidence[0].source_name == "SBP EasyData" and out.evidence[0].source_url
    assert out.risks == ["oil price move is a headwind"]
    assert out.metrics["indicators"][0] == {"label": "economic growth", "value": "4.61 percent (bullish)"}


def test_no_warnings_is_ok():
    assert map_macro_output({**SAMPLE, "warnings": []}, "OGDC").status == AgentStatus.OK


async def test_unreachable_service_is_unavailable(monkeypatch):
    import app.agents.macro.agent as m
    monkeypatch.setattr(m, "MACRO_URL", "http://127.0.0.1:9")
    out = await MacroAgent().run("OGDC")
    assert out.status == AgentStatus.UNAVAILABLE and out.warnings
