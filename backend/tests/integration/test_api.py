from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_tickers_lookup():
    assert client.get("/tickers/mebl").json()["symbol"] == "MEBL"
    assert client.get("/tickers/NOPE").status_code == 404


def test_recommendation_degrades_gracefully_with_stub_agents():
    body = client.get("/recommendation/MEBL").json()
    assert body["call"] == "HOLD"
    assert len(body["agent_outputs"]) == 6
    assert all(a["status"] == "unavailable" for a in body["agent_outputs"])
