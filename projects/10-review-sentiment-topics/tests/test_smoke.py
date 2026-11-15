"""Smoke tests (run `pytest -q` from the project folder after training)."""

import json
import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from review_lib import load_reviews, rating_to_label  # noqa: E402
from topics import npmi_coherence  # noqa: E402


def test_labels():
    assert [rating_to_label(r) for r in (1, 2, 3, 4, 5)] == ["negative", "negative", "neutral", "positive", "positive"]


def test_load():
    df = load_reviews()
    assert len(df) == 22641 and df.text.str.len().min() > 0


def test_npmi_perfect_cooccurrence():
    import scipy.sparse as sp
    X = sp.csc_matrix([[1, 1], [1, 1], [0, 0], [0, 0]], dtype=float)
    assert abs(npmi_coherence([0, 1], X) - 1.0) < 1e-9


def test_models_predict():
    s = joblib.load(ROOT / "models" / "sentiment_tfidf.joblib")
    assert s["model"].predict(["terrible quality, fell apart, returning it"])[0] == "negative"
    assert s["model"].predict(["absolutely love it, perfect fit and gorgeous"])[0] == "positive"


def test_metrics_beat_dummy():
    m = json.loads((ROOT / "reports" / "sentiment_metrics.json").read_text())
    assert m["test"][m["best_model"]]["macro_f1"] > m["test"]["dummy"]["macro_f1"] + 0.2
