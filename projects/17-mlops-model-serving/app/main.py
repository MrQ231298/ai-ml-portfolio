"""FastAPI service for the hourly bike-demand model.

Endpoints
  GET  /health        liveness + model version
  GET  /model         model card (version, training window, validation metrics)
  POST /predict       1-500 records -> predicted rides per hour
  GET  /drift         PSI of recent request features vs the training reference
  GET  /metrics       Prometheus-style counters (plain text)
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from drift import level, psi  # noqa: E402

MODEL_DIR = Path(os.environ.get("MODEL_DIR", ROOT / "models"))
MODEL = joblib.load(MODEL_DIR / "model.joblib")
CARD = json.loads((MODEL_DIR / "model_card.json").read_text())
FEATURES = CARD["features"]
_ref = json.loads((MODEL_DIR / "reference_profile.json").read_text())
REFERENCE = {k: np.array(v, dtype=float) for k, v in _ref["features"].items()}
CATEGORICAL = {"season", "hr", "holiday", "weekday", "workingday", "weathersit"}
RECENT = deque(maxlen=int(os.environ.get("DRIFT_WINDOW", 2000)))
STATS = {"requests": 0, "records": 0, "errors": 0, "latency_sum": 0.0}

app = FastAPI(title="Bike demand model", version=str(CARD["registry_version"]))


class Record(BaseModel):
    season: Literal[1, 2, 3, 4]
    hr: int = Field(ge=0, le=23)
    holiday: Literal[0, 1]
    weekday: int = Field(ge=0, le=6)
    workingday: Literal[0, 1]
    weathersit: Literal[1, 2, 3, 4]
    temp: float = Field(ge=0, le=1, description="normalised temperature (t / 41 C)")
    atemp: float = Field(ge=0, le=1)
    hum: float = Field(ge=0, le=1)
    windspeed: float = Field(ge=0, le=1)
    cnt_lag_1h: float | None = Field(default=None, ge=0, description="rides in the previous hour")
    cnt_lag_24h: float | None = Field(default=None, ge=0)
    cnt_lag_168h: float | None = Field(default=None, ge=0)
    cnt_mean_prev_168h: float | None = Field(default=None, ge=0)


class PredictRequest(BaseModel):
    records: list[Record] = Field(min_length=1, max_length=500)


@app.get("/health")
def health():
    return {"status": "ok", "model": CARD["name"], "version": CARD["registry_version"]}


@app.get("/model")
def model_card():
    return CARD


@app.post("/predict")
def predict(req: PredictRequest):
    t0 = time.perf_counter()
    STATS["requests"] += 1
    try:
        X = pd.DataFrame([r.model_dump() for r in req.records])[FEATURES].astype(float)
        y = np.clip(MODEL.predict(X), 0, None)
    except Exception as e:  # pragma: no cover - defensive
        STATS["errors"] += 1
        raise HTTPException(500, f"prediction failed: {e}") from None
    RECENT.extend(X.to_dict("records"))
    STATS["records"] += len(X)
    STATS["latency_sum"] += time.perf_counter() - t0
    return {"model_version": CARD["registry_version"], "predictions": [round(float(v), 1) for v in y]}


@app.get("/drift")
def drift(min_records: int = 200):
    if len(RECENT) < min_records:
        return {"status": "insufficient data", "records": len(RECENT), "min_records": min_records}
    cur = pd.DataFrame(list(RECENT))
    out = {}
    for f in FEATURES:
        c = cur[f].dropna()
        r = REFERENCE[f][~np.isnan(REFERENCE[f])]
        if len(c) == 0:
            continue
        v = psi(r, c, categorical=f in CATEGORICAL)
        out[f] = {"psi": round(v, 4), "level": level(v)}
    return {"status": "ok", "records": len(cur), "features": out,
            "alerts": [f for f, d in out.items() if d["level"] == "alert" and f != "season"]}


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    return "\n".join([
        "# TYPE predict_requests_total counter", f"predict_requests_total {STATS['requests']}",
        "# TYPE predict_records_total counter", f"predict_records_total {STATS['records']}",
        "# TYPE predict_errors_total counter", f"predict_errors_total {STATS['errors']}",
        "# TYPE predict_latency_seconds_sum counter", f"predict_latency_seconds_sum {STATS['latency_sum']:.6f}",
        f'model_info{{name="{CARD["name"]}",version="{CARD["registry_version"]}"}} 1', ""])
