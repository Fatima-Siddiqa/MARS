"""Stance, confidence (agreement x quality) and optional model blending."""
import numpy as np
import pandas as pd

from app.schemas.agent_output import Stance

THRESH = 0.20


def stance_of(score: float) -> Stance:
    if score > THRESH:
        return Stance.BULLISH
    return Stance.BEARISH if score < -THRESH else Stance.NEUTRAL


def agreement(family_vals: list[float]) -> float:
    nz = [x for x in family_vals if x is not None and not np.isnan(x) and abs(x) > 1e-9]
    if not nz:
        return 0.0
    avg = float(np.mean(nz))
    return sum(np.sign(x) == np.sign(avg) for x in nz) / len(nz)


def freshness(last_date: pd.Timestamp, as_of: pd.Timestamp) -> float:
    gap = max(0, int(np.busday_count(last_date.date(), as_of.date())))  # weekdays ~ trading days
    return 1.0 if gap <= 3 else 0.5 if gap <= 10 else 0.2


def quality(n_days: int, no_trade_20: float, fresh: float) -> dict[str, float]:
    hist = min(1.0, n_days / 252)
    act = 1.0 - float(no_trade_20)
    return {"history": hist, "activity": act, "freshness": fresh, "quality": hist * act * fresh}
