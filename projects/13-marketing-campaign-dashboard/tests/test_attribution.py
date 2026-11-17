"""Tests for attribution models and the analysis outputs. Run after simulate.py and analyse.py."""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import attribution as A  # noqa: E402

T0 = pd.Timestamp("2026-01-01")


@pytest.fixture
def toy():
    tp = pd.DataFrame({"user_id": [1, 1, 1, 2, 3, 3],
                       "ts": [T0, T0 + pd.Timedelta(days=5), T0 + pd.Timedelta(days=9), T0, T0, T0 + pd.Timedelta(days=1)],
                       "channel": ["Social", "Search", "Email", "Search", "Social", "Email"]})
    conv = pd.DataFrame({"user_id": [1, 2], "ts": [T0 + pd.Timedelta(days=10), T0 + pd.Timedelta(days=1)], "revenue": [100.0, 50.0]})
    return A.journeys(tp, conv)


def test_rule_models_on_toy(toy):
    nz = lambda s: {k: v for k, v in s.to_dict().items() if v}  # noqa: E731
    assert nz(A.last_click(toy)["conversions"]) == {"Email": 1.0, "Search": 1.0}
    assert nz(A.first_click(toy)["conversions"]) == {"Search": 1.0, "Social": 1.0}
    lin = A.linear(toy)["conversions"]
    assert lin["Search"] == pytest.approx(1 + 1 / 3) and lin["Social"] == pytest.approx(1 / 3)
    pb = A.position_based(toy)["conversions"]
    assert pb["Search"] == pytest.approx(1.2) and pb["Social"] == pytest.approx(0.4)


def test_time_decay_prefers_recent(toy):
    td = A.time_decay(toy, half_life_days=7)["conversions"]
    assert td["Email"] > td["Social"]


def test_models_conserve_conversions(toy):
    for name, fn in A.MODELS.items():
        assert fn(toy)["conversions"].sum() == pytest.approx(2.0), name
    assert A.shapley(toy, base_rate=0.0)["conversions"].sum() == pytest.approx(2.0)


def test_markov_removal_effect_hand_checked():
    # A->conv, B->null, A->B->conv : removing A kills both conversions, removing B kills one of the paths
    tp = pd.DataFrame({"user_id": [1, 2, 3, 3], "ts": [T0, T0, T0, T0 + pd.Timedelta(hours=1)], "channel": ["A", "B", "A", "B"]})
    conv = pd.DataFrame({"user_id": [1, 3], "ts": [T0 + pd.Timedelta(days=1)] * 2, "revenue": [10.0, 10.0]})
    m = A.markov(A.journeys(tp, conv))["conversions"]
    assert m["A"] > m["B"]


def test_outputs_and_truth_consistency():
    m = json.loads((ROOT / "reports" / "metrics.json").read_text())
    t = json.loads((ROOT / "data" / "truth_incrementality.json").read_text())
    assert m["totals"]["orders"] == t["_total"]["conversions"]
    assert m["platform_over_claim"]["ratio"] > 1
    lo, hi = m["holdout_test_paid_social_prospecting"]["estimated_incremental_ci95"]
    assert lo > 0  # the randomised test detects a positive effect
    daily = pd.read_csv(ROOT / "data" / "SYNTHETIC_campaign_daily.csv")
    assert (daily["data_flag"] == "SYNTHETIC_EXAMPLE_DATA").all()
    assert (daily["clicks"] <= daily["impressions"]).all()
