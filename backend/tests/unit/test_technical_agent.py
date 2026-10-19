import numpy as np
import pandas as pd
import pytest

from app.agents.technical import scoring
from app.agents.technical.agent import TechnicalAgent
from app.agents.technical.features import compute_raw, expanding_robust_z, feature_scores
from app.ingestion import prices
from app.ingestion.prices import clean_prices
from app.schemas.agent_output import AgentStatus, Stance


def _raw(n=600, seed=0, dash_rows=()):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.015, n)))
    d = pd.DataFrame({
        "Date": pd.bdate_range("2020-01-01", periods=n), "Open": close, "High": close * 1.01,
        "Low": close * 0.99, "Close": close * 2, "Adj. Close": close, 
        "Change": pd.Series(0.0, index=range(n)).astype(object),
        "Volume": rng.integers(1e5, 1e6, n).astype(object)})
    for i in dash_rows:
        d.loc[i, ["Volume", "Change"]] = "-"
    return d.sort_values("Date", ascending=False)  # raw files are newest-first


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    _raw(dash_rows=(300,)).to_csv(tmp_path / "TEST.csv", index=False)
    _raw(n=30).to_csv(tmp_path / "SHORT.csv", index=False)
    bench = _raw(seed=1)[["Date", "Close"]].copy()
    bench["Date"] = bench["Date"].dt.strftime("%d-%b-%Y")
    bench.to_csv(tmp_path / "kse.csv", index=False)
    monkeypatch.setattr(prices, "PRICE_DIR", tmp_path)
    monkeypatch.setattr(prices, "BENCHMARK_PATH", tmp_path / "kse.csv")
    return tmp_path


def test_frontend_symbols_resolve_to_files(data_dir, monkeypatch):
    monkeypatch.setattr(prices, "FILE_ALIASES", {**prices.FILE_ALIASES, "LUCKY": "TEST"})
    assert prices.price_path("LUCKY") == data_dir / "TEST.csv"
    assert {"LUCKY", "SYSTEMS", "OGDCL"} <= set(prices.FILE_ALIASES)


def test_clean_handles_dashes_and_adjusts():
    c = clean_prices(_raw(dash_rows=(300,)))
    assert c.index.is_monotonic_increasing
    assert c["no_trade"].sum() == 1 and (c["Volume"] == 0).sum() == 1
    assert c["Close"].iloc[0] < 150  # Adj. Close used, raw Close was 2x
    assert abs(c["High"].iloc[10] / c["Close"].iloc[10] - 0.505) < 1e-9  # H scaled by Adj/Close


def test_no_lookahead_in_features():
    c = clean_prices(_raw())
    full = feature_scores(compute_raw(c))
    cut = feature_scores(compute_raw(c.iloc[:450]))
    pd.testing.assert_frame_equal(full.iloc[:450], cut, check_exact=False, atol=1e-9)


def test_z_bounded_and_last_only_matches():
    s = pd.Series(np.random.default_rng(2).normal(size=400))
    z = expanding_robust_z(s)
    assert z.dropna().abs().max() <= 1 and z.iloc[:251].isna().all()
    assert expanding_robust_z(s, last_only=True).iloc[-1] == pytest.approx(z.iloc[-1])


def test_freshness_and_agreement():
    assert scoring.freshness(pd.Timestamp("2026-07-14"), pd.Timestamp("2026-07-16")) == 1.0
    assert scoring.freshness(pd.Timestamp("2026-07-14"), pd.Timestamp("2026-10-09")) == 0.2
    assert scoring.agreement([0.5, 0.3, -0.2, 0.0]) == pytest.approx(2 / 3)


async def test_normal_run_contract(data_dir):
    out = await TechnicalAgent().run("TEST", "2022-05-01")
    assert out.status in (AgentStatus.OK, AgentStatus.DEGRADED)
    assert -1 <= out.score <= 1 and 0 <= out.confidence <= 1
    assert out.evidence and out.summary and "not a forecast" in out.summary
    assert len(out.metrics["chart"]) == 250


async def test_short_history_neutral_low_conf(data_dir):
    out = await TechnicalAgent().run("SHORT", "2030-01-01")
    assert out.stance == Stance.NEUTRAL and out.confidence < 0.2


async def test_stale_data_low_freshness(data_dir):
    out = await TechnicalAgent().run("TEST", "2030-01-01")
    assert out.metrics["quality"]["freshness"] == 0.2
    assert any("Stale" in w for w in out.warnings)


async def test_unknown_ticker_unavailable(data_dir):
    out = await TechnicalAgent().run("NOPE", "2022-05-01")
    assert out.status == AgentStatus.UNAVAILABLE


async def test_missing_benchmark_degrades_gracefully(data_dir, monkeypatch):
    monkeypatch.setattr(prices, "BENCHMARK_PATH", data_dir / "none.csv")
    out = await TechnicalAgent().run("TEST", "2022-05-01")
    assert "relative_strength" not in out.metrics["families_used"]
    assert any("benchmark" in w for w in out.warnings)
