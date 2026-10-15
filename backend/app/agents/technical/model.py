"""Optional supervised model (Level 2). Used only if evaluation adopted it (see evaluation.py)."""
import json
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from app.agents.technical.features import FEATURE_COLS
from app.core.config import REPO_ROOT

ARTIFACT_DIR = REPO_ROOT / "data" / "processed" / "technical"
MODEL_PATH = ARTIFACT_DIR / "model.joblib"
REPORT_PATH = ARTIFACT_DIR / "evaluation_report.json"
BLEND_WEIGHT = 0.5  # final = 0.5*rule + 0.5*(2p-1)


@lru_cache
def load_adopted_model():
    """Return the fitted model only if the evaluation report says it was adopted."""
    if not (MODEL_PATH.exists() and REPORT_PATH.exists()):
        return None
    try:
        if not json.loads(REPORT_PATH.read_text()).get("adopted"):
            return None
        return joblib.load(MODEL_PATH)
    except Exception:  # noqa: BLE001  corrupt artifact -> fall back to rules
        return None


def predict_prob(model, scores_row: pd.DataFrame) -> float:
    x = scores_row[FEATURE_COLS].fillna(0.0).to_numpy(float)
    return float(model.predict_proba(x)[0, 1])


def blend(rule: float, p: float) -> float:
    return float(np.clip((1 - BLEND_WEIGHT) * rule + BLEND_WEIGHT * (2 * p - 1), -1, 1))
