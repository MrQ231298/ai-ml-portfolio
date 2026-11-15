"""Smoke tests (run `pytest -q` from the project folder after the back-test)."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from forecast_lib import load_wide, mase_scale, seasonal_naive, smape  # noqa: E402


def test_data_shape():
    w = load_wide()
    assert w.shape[1] == 13 and w.isna().sum().sum() == 0
    assert w.index.freqstr == "MS"


def test_seasonal_naive_repeats_last_year():
    y = pd.Series(np.arange(24, dtype=float), index=pd.date_range("2020-01-01", periods=24, freq="MS"))
    assert list(seasonal_naive(y, 3)) == [12.0, 13.0, 14.0]


def test_metrics_basic():
    assert smape([100, 100], [100, 100]) == 0
    y = pd.Series(np.r_[np.ones(36), np.ones(12) * 2], index=pd.date_range("2020-01-01", periods=48, freq="MS"))
    assert mase_scale(y) > 0


def test_reports_exist_and_beat_naive():
    m = json.loads((ROOT / "reports" / "metrics.json").read_text())
    assert m["overall"][m["best_model"]]["MASE"] < m["overall"]["naive"]["MASE"]
    fut = pd.read_csv(ROOT / "reports" / "future_forecast.csv")
    assert len(fut) == 13 * 6 and (fut.lo80 <= fut.forecast).all() and (fut.forecast <= fut.hi80).all()
