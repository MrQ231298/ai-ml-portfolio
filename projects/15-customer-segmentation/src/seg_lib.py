"""Shared preprocessing so the saved model can be reloaded by the app and tests."""

from __future__ import annotations

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

LOG_COLS = ["recency_days", "frequency", "monetary", "tenure_days", "avg_basket_value", "distinct_products"]
FEATURES = LOG_COLS + ["cancel_rate"]


def log_and_clip(X):
    """log1p for the six skewed features (all >= 0); cancel_rate clipped to [0, 1]."""
    X = np.asarray(X, dtype=float)
    return np.column_stack([np.log1p(np.maximum(X[:, :6], 0)), np.clip(X[:, 6:], 0, 1)])


def preprocess():
    return make_pipeline(FunctionTransformer(log_and_clip, validate=True), StandardScaler())
