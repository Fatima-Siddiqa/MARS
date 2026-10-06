"""Baseline synthesis: confidence-weighted average of agent scores.

Kept permanently as (a) the Iteration-1/2 synthesis and (b) the fallback + evaluation
baseline that regime-aware conflict resolution is compared against.
"""
from app.schemas.agent_output import AgentOutput, AgentStatus
from app.schemas.recommendation import Call, Recommendation

BUY_THRESHOLD = 0.25
AVOID_THRESHOLD = -0.25
_USABLE = {AgentStatus.OK, AgentStatus.DEGRADED}


def synthesize(ticker: str, outputs: list[AgentOutput]) -> Recommendation:
    usable = [o for o in outputs if o.status in _USABLE]
    if not usable:
        return Recommendation(
            ticker=ticker,
            call=Call.HOLD,
            confidence=0.0,
            thesis="No agent produced a usable result.",
            agent_outputs=outputs,
        )

    weights = [o.confidence for o in usable]
    total = sum(weights)
    if total > 0:
        score = sum(o.score * w for o, w in zip(usable, weights)) / total
    else:
        score = sum(o.score for o in usable) / len(usable)

    if score >= BUY_THRESHOLD:
        call = Call.BUY
    elif score <= AVOID_THRESHOLD:
        call = Call.AVOID
    else:
        call = Call.HOLD

    return Recommendation(
        ticker=ticker,
        call=call,
        confidence=round(total / len(usable), 3),
        thesis=f"Baseline weighted average score {score:+.2f} from {len(usable)} agent(s).",
        key_risks=[r for o in usable for r in o.risks],
        evidence=[e for o in usable for e in o.evidence],
        agent_outputs=outputs,
    )
