"""API contract tests with FastAPI's TestClient against the committed model."""

import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import CARD, app

client = TestClient(app)
SAMPLE = json.loads((Path(__file__).parent / "sample_request.json").read_text())


def test_health_reports_registered_version():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["version"] == CARD["registry_version"]


def test_predict_returns_one_non_negative_value_per_record_and_handles_missing_lags():
    r = client.post("/predict", json=SAMPLE)
    assert r.status_code == 200
    p = r.json()["predictions"]
    assert len(p) == 2 and all(v >= 0 for v in p)
    assert p[0] > p[1]  # weekday 8 am rush hour in summer vs 3 am wet winter Sunday


def test_invalid_input_is_rejected_with_422():
    bad = {"records": [dict(SAMPLE["records"][0], hr=27)]}
    assert client.post("/predict", json=bad).status_code == 422
    assert client.post("/predict", json={"records": []}).status_code == 422


def test_metrics_and_drift_endpoints():
    m = client.get("/metrics").text
    assert "predict_requests_total" in m and "model_info" in m
    d = client.get("/drift", params={"min_records": 1}).json()
    assert d["status"] == "ok" and "temp" in d["features"]
