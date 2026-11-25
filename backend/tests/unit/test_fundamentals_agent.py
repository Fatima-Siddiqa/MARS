"""BUY / HOLD / AVOID decision and confidence rules of the fundamentals agent."""

import math

import pytest

from app.agents.fundamentals.agent import confidence, decide


@pytest.mark.parametrize(
    "gap,call",
    [
        (0.30, "BUY"),
        (0.25, "BUY"),
        (0.10, "HOLD"),
        (-0.10, "HOLD"),
        (-0.20, "AVOID"),
        (-0.50, "AVOID"),
        (None, "HOLD"),
    ],
)
def test_calls_are_log_symmetric_around_the_threshold(gap, call):
    # AVOID boundary for T=25% is 1/1.25 - 1 = -20% (same distance in log terms)
    assert decide(gap, 0.25)[0] == call


def test_score_is_bounded_and_signed():
    assert decide(0.25, 0.25)[1] == pytest.approx(0.5)
    assert decide(3.0, 0.25)[1] == 1.0 and decide(-0.9, 0.25)[1] == -1.0
    assert decide(0.0, 0.25)[1] == 0.0


def test_confidence_rises_with_evidence_peers_and_margin():
    base = dict(threshold=0.25, ratio_cov=1.0, fit=math.exp(-0.4))
    weak = confidence("BUY", 0.30, supported=False, peer_strength="weak", **base)
    strong = confidence("BUY", 0.30, supported=True, peer_strength="adequate", **base)
    far = confidence("BUY", 0.80, supported=True, peer_strength="adequate", **base)
    assert 0 < weak < strong < far <= 0.95
    # HOLD is most confident when the gap is near zero
    assert confidence("HOLD", 0.0, supported=True, peer_strength="adequate", **base) > confidence(
        "HOLD", 0.2, supported=True, peer_strength="adequate", **base
    )
    assert confidence("HOLD", None, supported=True, peer_strength="adequate", **base) == 0.0
