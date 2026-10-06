from app.schemas.agent_output import AgentName, AgentOutput, AgentStatus
from app.schemas.recommendation import Call
from app.synthesis.baseline_average import synthesize


def _out(agent, score, conf=0.8, status=AgentStatus.OK):
    return AgentOutput(agent=agent, ticker="MEBL", score=score, confidence=conf, status=status)


def test_bullish_gives_buy():
    outs = [_out(AgentName.TECHNICAL, 0.8), _out(AgentName.SENTIMENT, 0.6)]
    assert synthesize("MEBL", outs).call == Call.BUY


def test_bearish_gives_avoid():
    outs = [_out(AgentName.TECHNICAL, -0.7), _out(AgentName.MACRO, -0.5)]
    assert synthesize("MEBL", outs).call == Call.AVOID


def test_no_usable_agents_gives_hold_zero_confidence():
    outs = [_out(AgentName.TECHNICAL, 0.0, 0.0, AgentStatus.UNAVAILABLE)]
    rec = synthesize("MEBL", outs)
    assert rec.call == Call.HOLD and rec.confidence == 0.0
