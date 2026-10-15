import numpy as np
import pandas as pd
import pytest

from conftest import D, make_pipe
from macro_agent.llm import rewrite, validate_numbers

SKIP_FOR_FEW = ["tbill_3m", "pkr_usd", "fx_reserves", "current_account", "remittances", "brent", "gold", "gdp_growth"]


def test_contributions_sum_to_composite(pipe):
    df, sf, qf, comp = pipe.score_series(None, end=pd.Timestamp("2024-06-30"))
    assert sf.contrib.iloc[-1].sum() == pytest.approx(sf.composite.iloc[-1], abs=1e-9)
    # equal-weight mean of available pillars
    assert sf.pillars.iloc[-1].mean() == pytest.approx(sf.composite.iloc[-1], abs=1e-9)


def test_output_contract(pipe):
    out = pipe.score("2024-06-30")
    assert set(out) == {"agent", "ticker", "score", "state", "confidence", "pillars", "drivers", "evidence", "warnings",
                        "plain_summary", "as_of", "horizon_days", "model_version"}
    assert out["agent"] == "macro" and out["horizon_days"] == 20 and -1 <= out["score"] <= 1
    assert out["state"] in ("Bullish", "Neutral", "Bearish") and 0 <= out["confidence"] <= 1
    assert set(out["pillars"]) == {"monetary", "inflation", "external", "commodity", "growth"}
    assert len(out["drivers"]) <= 3
    for e in out["evidence"]:
        assert {"series_key", "value", "date", "unit", "source_url"} <= set(e)
    assert any("DEMO DATA" in w for w in out["warnings"])                # synthetic data is always labelled


def test_state_thresholds(pipe):
    df, *_ = pipe.score_series(None, end=pd.Timestamp("2025-12-31"))
    ok = df[~df["gated"] & df["score"].notna()]
    assert (ok.loc[ok["score"] >= 0.20, "state"] == "Bullish").all()
    assert (ok.loc[ok["score"] <= -0.20, "state"] == "Bearish").all()
    assert (ok.loc[ok["score"].abs() < 0.20, "state"] == "Neutral").all()


def test_gate_fewer_than_three_pillars_forces_neutral(tmp_path, data):
    p = make_pipe(tmp_path, data, skip=SKIP_FOR_FEW)                     # only policy_rate + cpi_yoy -> 2 pillars
    out = p.score("2025-12-31")
    assert out["state"] == "Neutral" and out["confidence"] <= 0.20
    assert any("pillars have data" in w for w in out["warnings"])


@pytest.mark.parametrize("drop,label", [("cpi_yoy", "cpi"), ("pkr_usd", "pkr usd"), ("policy_rate", "policy rate")])
def test_gate_missing_critical_input_forces_neutral(tmp_path, data, drop, label):
    p = make_pipe(tmp_path, data, skip=[drop])
    out = p.score("2025-12-31")
    assert out["state"] == "Neutral" and out["confidence"] <= 0.20
    assert any("Critical input missing" in w and label in w.lower() for w in out["warnings"])


def test_stale_series_is_named_and_lowers_confidence(tmp_path, data, pipe):
    cut = pd.Timestamp("2025-06-30")
    d2 = dict(data)
    d2["cpi_yoy"] = data["cpi_yoy"][data["cpi_yoy"]["obs_date"] <= cut]   # CPI stops 15 months before the as-of date
    p = make_pipe(tmp_path, d2)
    out, ref = p.score("2026-10-10"), pipe.score("2026-10-10")
    assert any(w.startswith("cpi_yoy is stale") for w in out["warnings"])
    assert out["confidence"] < ref["confidence"]


def test_sector_sensitivity_flips_oil_effect(pipe):
    mkt, og = pipe.score("2025-12-31"), pipe.score("2025-12-31", "OGDC")
    assert mkt["pillars"]["commodity"] is not None
    assert og["pillars"]["commodity"] == pytest.approx(-mkt["pillars"]["commodity"])
    assert og["pillars"]["monetary"] == pytest.approx(mkt["pillars"]["monetary"])


def test_unknown_ticker_returns_market_score_with_warning(pipe):
    out = pipe.score("2025-12-31", "ZZZZ")
    assert out["score"] == pytest.approx(pipe.score("2025-12-31")["score"])
    assert any("No sector mapping" in w for w in out["warnings"])


def test_regime_inputs_have_flags_not_labels(pipe):
    r = pipe.regime_inputs("2025-12-31")
    assert set(r["flags"]) == {"rate_cycle", "inflation_shock", "rupee_stress", "oil_shock"}
    assert "regime" not in r


# ---- LLM number check ----------------------------------------------------------------------------------------------
FACTS = {"state": "Neutral", "score": 0.05, "drivers": [{"label": "the inflation rate", "value": 24.87, "unit": "percent", "display_value": 24.9}],
         "as_of": "2026-10-10", "warnings": ["cpi_yoy is stale: last released 197 days ago"], "pillars": {"growth": 0.46}}


def test_number_check_accepts_rounding_and_rejects_new_numbers():
    assert validate_numbers("Inflation is 24.9% (or about 25%) as of 10 October 2026.", FACTS)[0]
    assert validate_numbers("Growth pillar scored 0.46, data is 197 days old.", FACTS)[0]
    ok, bad = validate_numbers("Inflation is 26.1% right now.", FACTS)
    assert not ok and bad == [26.1]
    assert not validate_numbers("Expect a 5% rise next month.", FACTS)[0]


def test_llm_rewrite_falls_back_on_mismatch_and_errors(monkeypatch):
    import requests
    tpl = "Inflation is 24.9%."
    cfg = {"enabled": True, "url": "http://x", "model": "m", "timeout_s": 1}

    class R:
        def __init__(self, t): self.t = t
        def raise_for_status(self): pass
        def json(self): return {"response": self.t}

    monkeypatch.setattr(requests, "post", lambda *a, **k: R("Prices are rising 31.5% a year."))
    assert rewrite(tpl, FACTS, cfg) == (tpl, False)
    monkeypatch.setattr(requests, "post", lambda *a, **k: R("Prices are rising about 24.9% a year."))
    assert rewrite(tpl, FACTS, cfg) == ("Prices are rising about 24.9% a year.", True)

    def boom(*a, **k): raise requests.ConnectionError("down")
    monkeypatch.setattr(requests, "post", boom)
    assert rewrite(tpl, FACTS, cfg) == (tpl, False)
    assert rewrite(tpl, FACTS, {**cfg, "enabled": False}) == (tpl, False)
