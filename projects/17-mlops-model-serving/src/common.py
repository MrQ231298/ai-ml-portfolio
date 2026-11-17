"""Shared paths, data loading and feature definitions."""

from __future__ import annotations

import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DOWNLOADS = ROOT / "downloads"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
MLRUNS = ROOT / "mlruns"
TRACKING_URI = f"sqlite:///{MLRUNS / 'mlflow.db'}"
MODEL_NAME = "bike-demand"

URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
SHA256 = "b70182d0d0508e9abbb79306ce5c0cec34869000f8220175ac83d11dbe845401"
LICENCE = "CC BY 4.0"
CITATION = "Fanaee-T, H. (2013). Bike Sharing [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5W894"

# Month is deliberately NOT a feature: in 2011 (the first year of the scheme) it mostly encodes the
# ramp-up of demand, not seasonality. Demand level comes from lag features instead.
CATEGORICAL = ["season", "hr", "holiday", "weekday", "workingday", "weathersit"]
NUMERIC = ["temp", "atemp", "hum", "windspeed"]
LAGS = ["cnt_lag_1h", "cnt_lag_24h", "cnt_lag_168h", "cnt_mean_prev_168h"]
FEATURES = CATEGORICAL + NUMERIC + LAGS
TARGET = "cnt"


def load_hourly() -> pd.DataFrame:
    """Download (once), verify SHA-256 and return the hourly table with a timestamp column."""
    DOWNLOADS.mkdir(exist_ok=True)
    z = DOWNLOADS / "bike_sharing.zip"
    if not z.exists():
        urllib.request.urlretrieve(URL, z)
    digest = hashlib.sha256(z.read_bytes()).hexdigest()
    if digest != SHA256:
        raise RuntimeError(f"checksum mismatch: {digest}")
    with zipfile.ZipFile(z) as zf:
        df = pd.read_csv(io.BytesIO(zf.read("hour.csv")))
    df["ts"] = pd.to_datetime(df["dteday"]) + pd.to_timedelta(df["hr"], unit="h")
    df = df.sort_values("ts").reset_index(drop=True)
    return add_lags(df)


def add_lags(df: pd.DataFrame) -> pd.DataFrame:
    """Lag features on a complete hourly grid (the source has some missing hours; those lags stay NaN)."""
    s = df.set_index("ts")["cnt"].asfreq("h")
    lags = pd.DataFrame({"cnt_lag_1h": s.shift(1), "cnt_lag_24h": s.shift(24), "cnt_lag_168h": s.shift(168),
                         "cnt_mean_prev_168h": s.shift(1).rolling(168, min_periods=120).mean()})
    return df.join(lags, on="ts")


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))
