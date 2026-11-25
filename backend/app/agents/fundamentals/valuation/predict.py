"""Model-implied value for one company, with a range and SHAP explanations.

Features use the same definitions as training (dataset.features_from): size and margin from the
latest trailing-twelve-month figures, growth from the last two published fiscal years. The
prediction is relative to the PSX equal-weighted index, so it is scaled back by the index level
on the valuation date.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from app.agents.fundamentals.data.loader import Snapshot
from app.agents.fundamentals.valuation.dataset import FEATURES, NUMERIC_FEATURES, features_from
from app.agents.fundamentals.valuation.market_index import level_on
from app.agents.fundamentals.valuation.train import MODEL_PATH

LABELS = {
    "log_revenue": "Revenue (size)",
    "log_profit": "Profit (size)",
    "log_loss": "Loss (size)",
    "is_loss": "Loss-making",
    "net_margin": "Net margin",
    "revenue_growth": "Revenue growth",
    "profit_growth": "Profit growth",
    "is_bank": "Bank",
    "sector_code": "Sector",
}


@dataclass
class Valuation:
    method: str  # sector_pe_baseline | ridge | gbm | unavailable
    model_version: str | None
    implied_market_cap: float | None = None  # PKR millions
    implied_value_per_share: float | None = None
    range_per_share: tuple[float, float] | None = None  # 80% range from test errors
    price: float | None = None
    gap: float | None = None  # implied / price - 1
    features: dict = field(default_factory=dict)
    contributions: list[dict] = field(default_factory=list)  # SHAP, largest first
    test_metrics: dict = field(default_factory=dict)
    reason: str | None = None


@lru_cache(maxsize=1)
def _load(mtime: float) -> dict:
    return joblib.load(MODEL_PATH)


def load_model() -> dict | None:
    return _load(MODEL_PATH.stat().st_mtime) if MODEL_PATH.exists() else None


def inference_features(s: Snapshot) -> dict | None:
    bank = s.company_type == "bank"
    item = "total_income" if bank else "revenue"
    rev, ni = s.ttm.get(item), s.ttm.get("net_income")
    if ni is None:
        return None
    feats = features_from(
        None if rev is None else rev.value, ni.value, None, None, bank, s.sector_code
    )
    a_rev, a_ni = s.annual.get(item, []), s.annual.get("net_income", [])
    g = features_from(
        a_rev[-1].value if a_rev else None,
        a_ni[-1].value if a_ni else None,
        a_rev[0].value if len(a_rev) == 2 else None,
        a_ni[0].value if len(a_ni) == 2 else None,
        bank,
        s.sector_code,
    )
    feats["revenue_growth"], feats["profit_growth"] = g["revenue_growth"], g["profit_growth"]
    return feats


def _shap(art: dict, X: pd.DataFrame) -> list[dict]:
    name, model = art["name"], art["model"]
    if name == "sector_pe_baseline":
        return []
    pre, est = model.named_steps["pre"], model.named_steps["model"]
    Xt = pre.transform(X)
    names = list(pre.get_feature_names_out())
    if name == "gbm":
        import shap

        values = shap.TreeExplainer(est).shap_values(np.asarray(Xt, dtype=float))[0]
    else:  # linear: exact contribution = coef x (x - background mean)
        bg = pre.transform(art["background"])
        bg = bg.toarray() if hasattr(bg, "toarray") else bg
        xt = Xt.toarray() if hasattr(Xt, "toarray") else Xt
        values = est.coef_ * (xt[0] - bg.mean(axis=0))
    agg: dict[str, float] = {}
    for n, v in zip(names, values):
        base = n.split("__", 1)[1]
        base = base.replace("missingindicator_", "")
        base = "sector_code" if base.startswith("sector_code") else base
        agg[base] = agg.get(base, 0.0) + float(v)
    out = [
        {
            "feature": f,
            "label": LABELS.get(f, f),
            "value": X.iloc[0][f],
            "contribution_log": c,
            "effect_pct": float(np.expm1(c)),
        }
        for f, c in agg.items()
    ]
    return sorted(out, key=lambda d: abs(d["contribution_log"]), reverse=True)


def value(s: Snapshot) -> Valuation:
    art = load_model()
    if art is None:
        return Valuation(
            "unavailable",
            None,
            price=s.price,
            reason="no trained model: run python -m app.agents.fundamentals.valuation.train",
        )
    rep = art["report"]
    v = Valuation(
        art["name"],
        rep["model_version"],
        price=s.price,
        test_metrics=rep["pooled_test"].get(art["name"], {}),
    )
    feats = inference_features(s)
    level = level_on(s.as_of)
    if feats is None:
        v.reason = "trailing twelve-month profit not available"
        return v
    if s.market_cap is None or not s.shares or level is None:
        v.reason = "no current price, share count or market index on this date"
        return v
    X = pd.DataFrame([feats])[FEATURES]
    X[NUMERIC_FEATURES] = X[NUMERIC_FEATURES].astype(float)
    pred = float(art["model"].predict(X)[0])
    v.features = {
        k: (None if isinstance(x, float) and np.isnan(x) else x) for k, x in feats.items()
    }
    v.implied_market_cap = float(np.exp(pred) * level)
    v.implied_value_per_share = v.implied_market_cap * 1e6 / s.shares
    v.gap = v.implied_market_cap / s.market_cap - 1
    iv = rep.get("residual_interval_80pct_log")
    if iv:
        v.range_per_share = (
            v.implied_value_per_share * float(np.exp(iv[0])),
            v.implied_value_per_share * float(np.exp(iv[1])),
        )
    v.contributions = _shap(art, X)
    return v
