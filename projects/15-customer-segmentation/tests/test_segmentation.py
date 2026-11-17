"""Tests for feature building, preprocessing, naming rules and the saved model."""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fetch_and_features import clean, features  # noqa: E402
from seg_lib import FEATURES, log_and_clip, preprocess  # noqa: E402
from segment import name_segments, rfm_scores  # noqa: E402


def _raw():
    t = pd.Timestamp
    rows = [  # InvoiceNo, StockCode, Quantity, InvoiceDate, UnitPrice, CustomerID, Country
        ("1001", "85123A", 2, t("2011-01-10 10:00"), 5.0, 1.0, "UK"),
        ("1001", "22423", 1, t("2011-01-10 10:00"), 10.0, 1.0, "UK"),
        ("1001", "22423", 1, t("2011-01-10 10:00"), 10.0, 1.0, "UK"),   # exact duplicate -> dropped
        ("1002", "85123A", 4, t("2011-03-01 09:00"), 5.0, 1.0, "UK"),
        ("C1003", "85123A", -2, t("2011-03-05 09:00"), 5.0, 1.0, "UK"),  # cancellation worth -10
        ("1004", "POST", 1, t("2011-03-01 09:00"), 18.0, 1.0, "UK"),  # postage -> dropped
        ("1005", "84879", 1, t("2011-02-01 12:00"), 50.0, 2.0, "France"),
        ("1006", "84879", 1, t("2011-02-01 12:00"), 50.0, np.nan, "UK"),  # no customer -> dropped
        ("1007", "21212", 1, t("2011-02-02 12:00"), 0.0, 2.0, "France"),  # zero price -> dropped
    ]
    return pd.DataFrame(rows, columns=["InvoiceNo", "StockCode", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID", "Country"])


def test_clean_drops_duplicates_missing_ids_non_products_and_zero_prices():
    d, log = clean(_raw())
    assert log == {"raw_rows": 9, "exact_duplicates_removed": 1, "rows_with_customer_id": 7,
                   "rows_after_non_product_and_price_filter": 5}
    assert d["is_cancel"].sum() == 1


def test_features_hand_computed():
    d, _ = clean(_raw())
    f = features(d, pd.Timestamp("2011-01-01"), pd.Timestamp("2011-04-01"))
    c1 = f.loc["1"]
    assert c1["frequency"] == 2
    assert c1["monetary"] == pytest.approx(20 + 20 - 10)       # two baskets of 20, minus a 10 cancellation
    assert c1["avg_basket_value"] == pytest.approx(20)
    assert c1["recency_days"] == 30                              # 1 March -> 1 April (exclusive end)
    assert c1["tenure_days"] == 80
    assert c1["cancel_rate"] == pytest.approx(10 / 40)
    assert c1["distinct_products"] == 2
    assert f.loc["2", "country"] == "France" and f.loc["2", "cancel_rate"] == 0


def test_log_and_clip_and_scaler():
    X = np.array([[0, 1, 10, 5, 2, 3, 1.7], [9, 3, 100, 50, 20, 30, -0.1]], dtype=float)
    out = log_and_clip(X)
    assert np.allclose(out[:, :6], np.log1p(X[:, :6]))
    assert out[0, 6] == 1 and out[1, 6] == 0
    Z = preprocess().fit_transform(X)
    assert np.allclose(Z.mean(axis=0), 0)


def test_rfm_scores_order():
    df = pd.DataFrame({"recency_days": np.arange(10), "frequency": np.arange(10, 0, -1), "monetary": np.arange(10, 0, -1) * 100.0})
    r = rfm_scores(df)
    assert r.loc[0, "R"] == 5 and r.loc[9, "R"] == 1          # most recent -> highest R
    assert r.loc[0, "rfm_segment"] == "Champions" and r.loc[9, "rfm_segment"] == "Lost / hibernating"


def test_name_segments_rules():
    prof = pd.DataFrame({"recency_days": [140, 20, 120, 20, 30], "frequency": [1, 8, 2, 1, 4], "monetary": [300, 5000, 300, 320, 900],
                         "tenure_days": [200, 250, 200, 30, 250], "cancel_rate": [0.0, 0.0, 0.4, 0.0, 0.0]})
    assert name_segments(prof) == {0: "Lapsed one-off", 1: "VIP / wholesale-scale", 2: "High-cancellation",
                                   3: "New this quarter", 4: "Active regulars"}


artefacts = pytest.mark.skipif(not (ROOT / "models" / "segmenter.joblib").exists(), reason="run src/segment.py first")


@artefacts
def test_saved_model_reproduces_assignments_and_metrics_follow_rule():
    bundle = joblib.load(ROOT / "models" / "segmenter.joblib")
    cust = pd.read_csv(ROOT / "reports" / "customer_segments.csv")
    pred = bundle["kmeans"].predict(bundle["preprocess"].transform(cust[FEATURES].values))
    assert (pred == cust["segment"].values).all()
    m = json.loads((ROOT / "reports" / "metrics.json").read_text())
    grid = pd.DataFrame(m["k_grid"])
    ok = grid[grid.k.between(3, 8) & (grid.bootstrap_ari_mean >= 0.75)]
    assert m["k_selected"] == int(ok.sort_values("silhouette").iloc[-1]["k"])
    assert len(cust) == m["data"]["customers"]
