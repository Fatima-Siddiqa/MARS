"""Valuation model pieces on synthetic data (offline): features, baseline, selection, prediction."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from app.agents.fundamentals.data.loader import Fact, Snapshot
from app.agents.fundamentals.valuation import predict, train
from app.agents.fundamentals.valuation.dataset import FEATURES, TARGET, features_from


def test_features_from_handles_losses_and_missing():
    f = features_from(1000.0, 100.0, 800.0, 50.0, False, "0804")
    assert f["net_margin"] == pytest.approx(0.1)
    assert f["revenue_growth"] == pytest.approx(0.25) and f["profit_growth"] == pytest.approx(1.0)
    assert f["is_loss"] == 0 and f["log_loss"] == 0
    loss = features_from(None, -30.0, None, None, True, "0807")
    assert loss["is_loss"] == 1 and np.isnan(loss["log_revenue"]) and np.isnan(loss["net_margin"])
    assert loss["log_profit"] == 0 and loss["log_loss"] == pytest.approx(np.log1p(30))


def _synthetic(n_per_year=120, years=(2023, 2024, 2025, 2026), seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for y in years:
        for i in range(n_per_year):
            rev = float(rng.lognormal(8, 1))
            margin = float(rng.uniform(-0.1, 0.3))
            prev = rev / (1 + rng.uniform(-0.2, 0.5))
            sector = rng.choice(["0804", "0807", "0828"])
            f = features_from(
                rev, rev * margin, prev, prev * margin * 0.9, sector == "0807", sector
            )
            # value depends on size, margin and growth (non-linear in margin) + noise
            tgt = (
                0.9 * f["log_revenue"]
                + 3 * max(margin, 0) ** 0.5
                + 0.5 * f["revenue_growth"]
                + {"0804": 0.2, "0807": -0.3, "0828": 0.6}[sector]
                + rng.normal(0, 0.15)
            )
            rows.append({"ticker": f"T{i}", "obs_year": y, **f, TARGET: tgt})
    return pd.DataFrame(rows)


def test_sector_baseline_predicts_from_sector_multiple():
    X = pd.DataFrame(
        [
            features_from(100.0, 10.0, None, None, False, "0804"),
            features_from(100.0, 20.0, None, None, False, "0804"),
        ]
    )
    y = pd.Series([np.log1p(10.0) + 2.0, np.log1p(20.0) + 2.0])
    b = train.SectorMultipleBaseline().fit(X, y)
    new = pd.DataFrame([features_from(100.0, 40.0, None, None, False, "0804")])
    assert b.predict(new)[0] == pytest.approx(np.log1p(40.0) + 2.0)


def test_walk_forward_uses_only_earlier_years_and_selects_a_fitted_model():
    ds = _synthetic()
    ev = train.walk_forward(ds)
    assert [f["test_year"] for f in ev["folds"]] == [2025, 2026]  # 2024 has only 120 train rows
    assert all(f["n_train"] == len(ds[ds["obs_year"] < f["test_year"]]) for f in ev["folds"])
    chosen, why = train.choose(ev)
    assert chosen in {"ridge", "gbm"}  # signal is learnable: a fitted model beats the baseline
    assert ev["pooled"][chosen]["mae_log"] < ev["pooled"]["sector_pe_baseline"]["mae_log"]


def test_choose_keeps_baseline_when_models_do_not_beat_it():
    ev = {
        "pooled": {
            "sector_pe_baseline": {"mae_log": 0.5},
            "ridge": {"mae_log": 0.6},
            "gbm": {"mae_log": 0.55},
        },
        "folds": [],
    }
    assert train.choose(ev)[0] == "sector_pe_baseline"


def test_value_scales_by_index_and_explains(tmp_path, monkeypatch):
    ds = _synthetic()
    monkeypatch.setattr(train, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(train, "MODEL_PATH", tmp_path / "m.joblib")
    monkeypatch.setattr(train, "REPORT_JSON", tmp_path / "r.json")
    monkeypatch.setattr(train, "REPORT_MD", tmp_path / "r.md")
    monkeypatch.setattr(train, "OOS_CSV", tmp_path / "oos.csv")
    monkeypatch.setattr(train, "build_training_set", lambda: ds)
    rep = train.train()
    assert rep["chosen"] in {"ridge", "gbm"} and (tmp_path / "r.md").exists()
    monkeypatch.setattr(predict, "MODEL_PATH", tmp_path / "m.joblib")
    monkeypatch.setattr(predict, "level_on", lambda d: 2.0)
    predict._load.cache_clear()
    d = date(2026, 9, 30)
    s = Snapshot(
        "X",
        date(2026, 11, 26),
        "non_bank",
        "Cement",
        "0804",
        price=50.0,
        shares=1e8,
        market_cap=5000.0,
    )
    s.ttm = {
        "revenue": Fact(3000.0, None, d, d, "s", "r"),
        "net_income": Fact(450.0, None, d, d, "s", "r"),
    }
    v = predict.value(s)
    expected = np.exp(rep_pred(tmp_path, s) + 2.0)  # prediction is relative to the market log P/E
    assert v.implied_market_cap == pytest.approx(expected)
    assert v.gap == pytest.approx(expected / 5000.0 - 1)
    assert v.range_per_share[0] < v.implied_value_per_share < v.range_per_share[1]
    assert v.contributions and {"feature", "effect_pct"} <= set(v.contributions[0])


def rep_pred(path, s):
    import joblib

    art = joblib.load(path / "m.joblib")
    X = pd.DataFrame([predict.inference_features(s)])[FEATURES]
    return float(art["model"].predict(X)[0])
