import numpy as np
import pandas as pd
import pytest

from conftest import D, make_pipe
from macro_agent.alignment import asof_join
from macro_agent.config import FeatureCfg, load_config
from macro_agent.loaders.base import parse_dates
from macro_agent.quality import freshness_factor
from macro_agent.standardise import _standardise_one, standardise_features
from macro_agent.storage import Store
from macro_agent.validation import validate_series


# ---- as-of join / no look-ahead ------------------------------------------------------------------------------------
def test_asof_uses_only_released_values():
    obs = pd.DataFrame({"obs_date": pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-31"]),
                        "value": [1.0, 2.0, 3.0]})
    obs["release_date"] = obs["obs_date"] + pd.Timedelta(days=5)
    cal = pd.date_range("2024-02-01", "2024-03-20", freq="D")
    j = asof_join(obs, cal)
    assert np.isnan(j.loc["2024-02-01":"2024-02-04", "value"]).all()   # Jan value is released on 5 Feb: not usable before
    assert j.loc["2024-02-05", "value"] == 1.0
    assert j.loc["2024-03-04", "value"] == 1.0          # Feb value released 5 Mar
    assert j.loc["2024-03-05", "value"] == 2.0
    assert j.loc["2024-03-20", "value"] == 2.0          # March value (released 5 Apr) never leaks
    assert j.loc["2024-03-20", "age_days"] == 15


def test_asof_before_first_release_is_nan():
    obs = pd.DataFrame({"obs_date": pd.to_datetime(["2024-01-31"]), "value": [1.0]})
    obs["release_date"] = obs["obs_date"] + pd.Timedelta(days=10)
    j = asof_join(obs, pd.date_range("2024-01-31", "2024-02-15"))
    assert j.loc["2024-02-09", "value"] != j.loc["2024-02-09", "value"]
    assert j.loc["2024-02-10", "value"] == 1.0


def test_policy_rate_is_step_function(pipe):
    comp = pipe.compute(pd.Timestamp("2020-01-01"))
    v = comp.ff.panel["policy_rate"]["value"].dropna()
    assert (v.diff().dropna() != 0).sum() < len(v) / 20          # changes only on decision dates


def test_truncation_invariance(tmp_path, data, pipe):
    """A score for date D must be identical whether or not later data exists (no look-ahead anywhere)."""
    full = pipe.score(D, persist=False)
    trunc = make_pipe(tmp_path, data, max_release=D).score(D, persist=False)
    assert full["score"] == pytest.approx(trunc["score"], abs=1e-9)
    assert full["confidence"] == pytest.approx(trunc["confidence"], abs=1e-9)
    assert full["state"] == trunc["state"]
    assert full["pillars"] == pytest.approx(trunc["pillars"], abs=1e-9)
    assert {d["feature"] for d in full["drivers"]} == {d["feature"] for d in trunc["drivers"]}


# ---- dates ---------------------------------------------------------------------------------------------------------
def test_iso_dates_are_not_read_day_first():
    s = parse_dates(pd.Series(["2013-07-01", "2015-12-31"]))
    assert s.iloc[0] == pd.Timestamp("2013-07-01") and s.iloc[1] == pd.Timestamp("2015-12-31")


def test_sbp_style_dates_parse():
    s = parse_dates(pd.Series(["25-May-2015", "02-Jul-2013"]))
    assert s.iloc[0] == pd.Timestamp("2015-05-25") and s.iloc[1] == pd.Timestamp("2013-07-02")


# ---- raw storage is versioned, never overwritten -------------------------------------------------------------------
def test_raw_is_versioned_not_overwritten(tmp_path):
    st = Store(f"sqlite:///{tmp_path}/v.db")
    df = pd.DataFrame({"obs_date": pd.to_datetime(["2024-01-31", "2024-02-29"]), "value": [10.0, 11.0], "status": "final"})
    assert st.append_raw("x", df, "u", "url") == 2
    assert st.append_raw("x", df, "u", "url") == 0                      # unchanged -> no new version
    df2 = df.copy(); df2.loc[1, "value"] = 12.0
    assert st.append_raw("x", df2, "u", "url") == 1                     # revision -> new version
    assert st.raw_version_count("x") == 3                               # old row still there
    latest = st.read_raw_latest("x").set_index("obs_date")["value"]
    assert latest[pd.Timestamp("2024-02-29")] == 12.0


# ---- validation ----------------------------------------------------------------------------------------------------
def test_validator_flags_jumps_but_does_not_delete_and_excludes_implausible():
    cfg = load_config().series["pkr_usd"]
    d = pd.bdate_range("2020-01-01", periods=120)
    v = 150 + np.random.default_rng(0).normal(0, 0.2, 120).cumsum() * 0.1
    v[60] = v[59] * 1.5                                   # jump, plausible -> flagged, kept
    v[90] = -5                                            # implausible -> excluded from clean
    df = pd.DataFrame({"obs_date": d, "value": v, "status": "final"})
    clean, issues = validate_series(df, cfg)
    checks = {(i["check"]) for i in issues}
    assert "jump" in checks and "range" in checks
    assert len(clean) == 119 and (clean["value"] > 0).all()
    assert pd.Timestamp(d[60]) in set(clean["obs_date"])


def test_policy_rate_use_from_rule_keeps_original_but_drops_pre_break_rows():
    cfg = load_config().series["policy_rate"]
    df = pd.DataFrame({"obs_date": pd.to_datetime(["2014-01-01", "2015-05-25", "2016-01-01"]), "value": [9.0, 6.5, 6.0], "status": "final"})
    clean, issues = validate_series(df, cfg)
    assert list(clean["value"]) == [6.5, 6.0]
    assert any(i["check"] == "structural_break" for i in issues)


# ---- standardisation -----------------------------------------------------------------------------------------------
def _monthly_step(vals, start="2015-01-31"):
    me = pd.date_range(start, periods=len(vals), freq="ME")
    cal = pd.date_range(me[0], me[-1], freq="D")
    s = pd.Series(np.nan, index=cal)
    s.loc[me] = vals
    return s.ffill(), cal


def test_robust_z_known_value_and_clip():
    sc = load_config().standardise
    f = FeatureCfg(name="t", pillar="inflation", orient=1, series=["cpi_yoy"], unit="percent", label="t")
    x, cal = _monthly_step(list(range(1, 41)))
    score, method, z, _ = _standardise_one(x, f, cal, sc)
    assert method.iloc[-1] == "robust_z"
    assert z.iloc[-1] == pytest.approx((40 - 20.5) / (1.4826 * 10), abs=1e-6)
    assert score.iloc[-1] == pytest.approx(z.iloc[-1] / 3)
    x2, cal2 = _monthly_step(list(range(1, 40)) + [10000])
    score2, *_ = _standardise_one(x2, f, cal2, sc)
    assert score2.iloc[-1] == pytest.approx(1.0)                         # clipped at +3 -> +1


def test_needs_36_months_else_percentile_fallback_then_nan():
    sc = load_config().standardise
    f = FeatureCfg(name="t", pillar="inflation", orient=1, series=["cpi_yoy"], unit="percent", label="t")
    x, cal = _monthly_step(list(range(1, 21)))                           # 20 months -> percentile fallback
    score, method, *_ = _standardise_one(x, f, cal, sc)
    assert method.iloc[-1] == "percentile"
    assert score.iloc[-1] == pytest.approx(2 * ((19 + 0.5) / 20) - 1)
    x3, cal3 = _monthly_step(list(range(1, 9)))                          # 8 months -> unavailable
    assert np.isnan(_standardise_one(x3, f, cal3, sc)[0].iloc[-1])


def test_orientation_signs():
    cfg = load_config()
    n = 60
    cpi, cal = _monthly_step(list(np.linspace(5, 25, n)))                # inflation rising  -> bearish
    res, _ = _monthly_step(list(np.linspace(1, 10, n)))                  # reserves rising   -> bullish
    raw = pd.DataFrame(np.nan, index=cal, columns=list(cfg.features))
    raw["cpi_level"], raw["reserves_chg_3m"] = cpi, res
    std = standardise_features(raw, cfg)
    assert std.score["cpi_level"].iloc[-1] < -0.3
    assert std.score["reserves_chg_3m"].iloc[-1] > 0.3


def test_freshness_levels():
    lv = {"fresh": 1.0, "aging": 0.5, "stale": 0.2}
    f = freshness_factor(pd.Series([10.0, 40.0, 70.0]), 31, lv)
    assert list(f) == [1.0, 0.5, 0.2]
