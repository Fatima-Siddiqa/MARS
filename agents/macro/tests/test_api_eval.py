import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from macro_agent.evaluation import make_labels, run_evaluation, walk_forward_folds
from macro_agent.orchestrator import create_app


@pytest.fixture(scope="module")
def client(pipe):
    return TestClient(create_app(pipe, scheduler=False))


def test_api_returns_contract_and_persists(client, pipe):
    r = client.get("/agents/macro", params={"ticker": "ogdc", "as_of": "2025-12-31"})
    assert r.status_code == 200
    j = r.json()
    assert j["ticker"] == "OGDC" and j["as_of"] == "2025-12-31" and j["model_version"] == pipe.cfg.model_version
    stored = pipe.store.load_output("OGDC", "2025-12-31", pipe.cfg.model_version)
    assert stored["score"] == j["score"]                                  # audit row per ticker/date


def test_api_validation(client):
    assert client.get("/agents/macro", params={"ticker": "BAD TICKER!"}).status_code == 400
    assert client.get("/agents/macro", params={"as_of": "2999-01-01"}).status_code == 400
    assert client.post("/admin/refresh").status_code == 403


def test_api_health_and_sources(client):
    assert client.get("/health").json()["status"] == "ok"
    assert {r["series"] for r in client.get("/sources").json()} >= {"policy_rate", "pkr_usd", "cpi_yoy"}


def test_walk_forward_purges_and_never_overlaps():
    idx = pd.bdate_range("2010-01-01", periods=3000)
    folds = walk_forward_folds(idx, initial=1000, test=250, purge=20)
    assert folds
    for tr, te in folds:
        pos_te0 = idx.get_loc(te[0])
        assert idx.get_loc(tr[-1]) + 20 <= pos_te0                        # train labels end before the test block starts
        assert tr[-1] < te[0]
    tests = [d for _, te in folds for d in te]
    assert len(tests) == len(set(tests))


def test_labels_are_forward_looking():
    c = pd.Series([100.0 + i for i in range(60)], index=pd.bdate_range("2020-01-01", periods=60))
    lab = make_labels(c, 20)
    assert lab["fwd_ret"].iloc[0] == pytest.approx(c.iloc[20] / c.iloc[0] - 1)
    assert len(lab) == 40


def test_evaluation_runs_and_lockbox_runs_once(pipe, tmp_path):
    pipe.cfg.root = tmp_path
    res = run_evaluation(pipe, out_dir=tmp_path)
    rep = res.report
    assert rep["data_warning"] and "full_macro_agent" in rep["walk_forward_oos"]
    assert rep["weight_selection"]["selected"] in ("equal_weights", "fitted_weights")
    assert "lockbox" not in rep
    run_evaluation(pipe, lockbox=True, out_dir=tmp_path)
    with pytest.raises(RuntimeError, match="already run"):
        run_evaluation(pipe, lockbox=True, out_dir=tmp_path)
    json.loads(res.path_json.read_text())
