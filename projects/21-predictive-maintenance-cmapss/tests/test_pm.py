import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import alerting as al
from anomaly import HealthModel, first_alarm
from data import split_units
from features import build, informative_sensors
from rul import FleetLifeBaseline, capped, nasa_score, rmse

ROOT = Path(__file__).resolve().parents[1]


def toy_fleet(n_units=6, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(1, n_units + 1):
        life = int(rng.integers(120, 200))
        for c in range(1, life + 1):
            wear = max(0, c - life + 60) / 60
            rows.append({"unit": u, "cycle": c, "s1": 5.0, "s2": 100 + rng.normal(0, 1) + 10 * wear ** 2,
                         "s3": 50 + rng.normal(0, 1) - 5 * wear ** 2, "rul": life - c})
    return pd.DataFrame(rows)


def test_engine_split_is_disjoint_and_complete():
    s = split_units(range(1, 101))
    assert sorted(s["fit"] + s["val"] + s["test"]) == list(range(1, 101))
    assert len(s["fit"]) == 60 and not set(s["fit"]) & set(s["test"])


def test_features_are_causal():
    df = toy_fleet()
    sens = informative_sensors(df)
    assert sens == ["s2", "s3"]                       # constant s1 dropped
    full = build(df, sens)
    cut = df[df.cycle <= 80]
    part = build(cut, sens)
    pd.testing.assert_frame_equal(full.loc[part.index], part)  # future cycles never change past features


def test_health_model_scores_rise_with_wear():
    df = toy_fleet()
    F = build(df, ["s2", "s3"])
    hm = HealthModel(["s2", "s3"]).fit(F, (df.cycle <= 30).to_numpy())
    for m in ("limit_check", "mahalanobis", "isolation_forest"):
        s = hm.score(F, m)
        assert s[(df.rul <= 5).to_numpy()].mean() > s[(df.cycle <= 30).to_numpy()].mean()


def test_first_alarm_needs_consecutive_hits():
    assert first_alarm(np.array([0, 5, 0, 5, 5, 5, 0]), 1, k=3) == 5
    assert first_alarm(np.array([5, 0, 5]), 1, k=3) is None


def test_cost_model():
    c = al.Costs(c_plan=1, c_fail=10, c_waste=0.01, lead_time=10)
    assert al.unit_cost(None, c) == (10, True, 0.0)
    assert al.unit_cost(5, c) == (10, True, 0.0)              # alert too late to schedule the job
    assert al.unit_cost(30, c) == (1 + 0.2, False, 20.0)


def test_threshold_choice_prefers_cheapest():
    rul = np.arange(100, -1, -1.0)
    units = {1: (rul.copy(), rul), 2: (rul.copy(), rul)}      # perfect predictor
    th, curve = al.choose_threshold(units, range(0, 50), below=True, c=al.Costs())
    assert th == 10                                            # alert exactly at the lead time


def test_fixed_interval_alerts():
    c = al.Costs(lead_time=10)
    assert al.fixed_interval_alerts({1: 200, 2: 100}, 150, c) == [60.0, None]


def test_metrics_helpers():
    assert rmse([0, 0], [3, 4]) == pytest.approx(np.sqrt(12.5))
    assert nasa_score([10], [10]) == 0 and nasa_score([10], [20]) > nasa_score([20], [10])
    assert capped(np.array([200]))[0] == 125
    assert FleetLifeBaseline(200).predict(pd.DataFrame({"cycle": [10, 150, 250]})).tolist() == [125, 50, 0]


@pytest.mark.skipif(not (ROOT / "reports" / "metrics.json").exists(), reason="run src/run.py")
def test_recorded_metrics_are_consistent():
    m = json.loads((ROOT / "reports" / "metrics.json").read_text())["FD001"]
    assert m["engines"] == {"fit": 60, "val": 20, "test": 20}
    p = m["policies"]
    assert p["oracle"]["mean_cost"] <= p["predicted_rul_alert"]["mean_cost"] <= p["run_to_failure"]["mean_cost"]
    off = pd.read_csv(ROOT / "reports" / "official_test_predictions_FD001.csv")
    assert len(off) == 100
    best = m["rul"]["selected"]["hgb"]
    assert rmse(off.true_rul, off.pred_rul) == pytest.approx(m["rul"]["models"][best]["official_test_rmse_vs_true_rul"], abs=0.01)


@pytest.mark.skipif(not (ROOT / "reports" / "engine_traces_FD001.csv").exists(), reason="run src/run.py")
def test_app_renders():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    assert not at.exception and at.metric[0].label == "Predicted RUL"
