"""Smoke tests: run with `pytest -q` from the project folder after training."""

import json
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from churn_lib import AUDIT_COLS, FEATURE_COLS, load_clean, net_benefit  # noqa: E402


def test_clean_data():
    df = load_clean()
    assert len(df) == 7043
    assert df["TotalCharges"].isna().sum() == 0
    assert set(df["Churn"].unique()) == {0, 1}
    assert not set(AUDIT_COLS) & set(FEATURE_COLS)


def test_net_benefit_simple():
    cfg = {"offer_cost": 10, "save_rate": 0.5, "value_months": 2}
    # one churner (100/month) contacted, one non-churner contacted
    assert net_benefit([1, 0], [True, True], [100, 50], cfg) == 0.5 * 2 * 100 - 20


def test_bundle_predicts_probabilities():
    b = joblib.load(ROOT / "models" / "churn_model.joblib")
    df = load_clean().head(50)
    p = b["model"].predict_proba(df[b["feature_cols"]])[:, 1]
    assert np.all((p >= 0) & (p <= 1))
    assert 0 < b["threshold"] < 1


def test_metrics_beat_dummy():
    m = json.loads((ROOT / "reports" / "metrics.json").read_text())
    assert m["test"]["hgb_isotonic"]["roc_auc"] > m["test"]["dummy_prior"]["roc_auc"] + 0.2
