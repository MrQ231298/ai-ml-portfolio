"""Lag features, the baseline and the committed model's recorded quality."""

import json

import numpy as np
import pandas as pd

from common import MODELS, REPORTS, add_lags
from model_lib import SeasonalNaive


def test_lags_use_only_the_past_and_respect_missing_hours():
    ts = pd.date_range("2011-01-01", periods=400, freq="h")
    df = pd.DataFrame({"ts": ts, "cnt": np.arange(400)}).drop(index=[200])  # one missing hour
    out = add_lags(df).set_index("ts")
    assert out.loc[ts[230], "cnt_lag_24h"] == 206
    assert np.isnan(out.loc[ts[224], "cnt_lag_24h"])  # 24 h earlier is the missing hour
    assert out.loc[ts[300], "cnt_lag_168h"] == 132
    assert np.isnan(out.loc[ts[10], "cnt_mean_prev_168h"])


def test_seasonal_naive_falls_back_to_yesterday():
    X = pd.DataFrame({"cnt_lag_168h": [10.0, np.nan], "cnt_lag_24h": [5.0, 7.0]})
    assert SeasonalNaive().fit(X, [1, 2]).predict(X).tolist() == [10.0, 7.0]


def test_selected_model_beats_the_baseline_on_validation():
    t = json.loads((REPORTS / "training_metrics.json").read_text())
    card = json.loads((MODELS / "model_card.json").read_text())
    assert t["selected"] == card["candidate"]
    assert t["candidates"][t["selected"]]["valid"]["mae"] < t["candidates"]["seasonal_naive_baseline"]["valid"]["mae"]
