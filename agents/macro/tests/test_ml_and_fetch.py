import numpy as np
import pandas as pd
import pytest

from macro_agent.fetch_open import parse_bis_xru, parse_gem, parse_imf_monthly, parse_pink_sheet
from macro_agent.ml_research import HORIZON_END, folds, make_targets, market_monthly, run_ml_research


# ---------------------------------------------------------------------------------------- open-data parsers
def test_parse_gem_monthly_to_month_end_and_drops_nulls():
    payload = [{"page": 1}, [{"date": "2024M02", "value": 23.1}, {"date": "2024M01", "value": None},
                             {"date": "2023", "value": 30.0}, {"date": "2023M12", "value": 29.7}]]
    df = parse_gem(payload)
    assert df["obs_date"].dt.strftime("%Y-%m-%d").tolist() == ["2023-12-31", "2024-02-29"]
    assert df["value"].tolist() == [29.7, 23.1]


def test_parse_bis_monthly_to_month_end():
    text = "FREQ,TIME_PERIOD,OBS_VALUE\nM,2026-07,280.1\nM,2026-08,277.56\n"
    df = parse_bis_xru(text)
    assert df["obs_date"].dt.strftime("%Y-%m-%d").tolist() == ["2026-07-31", "2026-08-31"]


def test_parse_imf_keeps_monthly_only():
    text = "COUNTRY,FREQUENCY,TIME_PERIOD,OBS_VALUE\nPAK,A,2021,9.9\nPAK,M,2021-M11,10.5\nPAK,M,2021-M12,11.3\n"
    df = parse_imf_monthly(text)
    assert len(df) == 2 and df["obs_date"].iloc[-1] == pd.Timestamp("2021-12-31")


def test_parse_pink_sheet_finds_gold_column(tmp_path):
    rows = [["World Bank Commodity Price Data", None, None], [None, "Crude oil, Brent", "Gold"],
            [None, "($/bbl)", "($/troy oz)"], ["2024M01", 80.1, 2034.0], ["2024M02", 83.5, 2023.5]]
    f = tmp_path / "pink.xlsx"
    pytest.importorskip("openpyxl")
    pd.DataFrame(rows).to_excel(f, sheet_name="Monthly Prices", header=False, index=False)
    df = parse_pink_sheet(f)
    assert df["value"].tolist() == [2034.0, 2023.5]
    assert df["obs_date"].iloc[0] == pd.Timestamp("2024-01-31")


# ---------------------------------------------------------------------------------------- ML research design
def test_skip_label_shares_no_month_with_information_at_t():
    km = pd.Series([100.0, 110.0, 121.0, 133.1], index=pd.date_range("2020-01-31", periods=4, freq="ME"))
    t = make_targets(km)
    # row t=Jan: next = Feb/Jan, skip = Mar/Feb (starts after the decision month ends)
    assert t["next"].iloc[0] == pytest.approx(np.log(110 / 100))
    assert t["skip"].iloc[0] == pytest.approx(np.log(121 / 110))
    assert t["mom1"].iloc[1] == pytest.approx(np.log(110 / 100))


def test_market_monthly_from_daily_takes_month_end_value():
    d = pd.Series([1.0, 2.0, 3.0], index=pd.to_datetime(["2024-01-02", "2024-01-31", "2024-02-15"]))
    m = market_monthly(d)
    assert m.tolist() == [2.0, 3.0] and m.index[0] == pd.Timestamp("2024-01-31")


@pytest.mark.parametrize("target", list(HORIZON_END))
def test_folds_purge_so_training_labels_end_before_test(target):
    idx = pd.date_range("2010-01-31", periods=100, freq="ME")
    h = HORIZON_END[target]
    seen = []
    for tr, te in folds(idx, initial=40, test=12, horizon_end=h):
        assert tr.max() + h < te.min()          # label of the last training row is observed before testing starts
        seen.extend(te)
    assert seen == list(range(40, 100))         # every later month is tested exactly once


def test_ml_research_runs_end_to_end_on_demo_data(pipe, tmp_path):
    pytest.importorskip("xgboost")
    res = run_ml_research(pipe, out_dir=tmp_path, initial=36, test=12)
    r = res.report["targets"]["skip"]["results"]
    assert {"rules_raw", "ridge_features", "xgb_mono_features", "ar1"} <= set(r)
    assert r["hist_mean"]["ic"] is None and r["hist_mean"]["r2_oos"] == pytest.approx(0.0)
    assert (tmp_path / "ml_research.md").exists()
    # no forecast is made for a month whose label would reach the lock-box
    lock = pd.Timestamp(pipe.cfg.evaluation["lockbox_start"])
    assert (res.predictions.index + pd.offsets.MonthEnd(2) < lock)[res.predictions[("skip", "actual")].notna()].all()
