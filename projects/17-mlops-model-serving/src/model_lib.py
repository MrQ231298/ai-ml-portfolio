"""Model definitions shared by training, monitoring, serving and tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from common import CATEGORICAL, FEATURES, LAGS, TARGET

CANDIDATES = {
    "ridge_onehot": {"kind": "ridge", "alpha": 1.0},
    "hgb_small": {"kind": "hgb", "max_iter": 200, "learning_rate": 0.1, "max_leaf_nodes": 15},
    "hgb_large": {"kind": "hgb", "max_iter": 500, "learning_rate": 0.05, "max_leaf_nodes": 31},
}


class PreviousHourNaive:
    """'Same as the previous hour': the other standard no-ML baseline for a 1-hour-ahead forecast."""

    def fit(self, X, y):
        self.fallback_ = float(np.mean(y))
        return self

    def predict(self, X):
        p = X["cnt_lag_1h"].to_numpy(dtype=float)
        return np.where(np.isnan(p), self.fallback_, p)


class SeasonalNaive:
    """'Same hour last week' (falls back to same hour yesterday): the standard no-ML baseline."""

    def fit(self, X, y):
        self.fallback_ = float(np.mean(y))
        return self

    def predict(self, X):
        p = X["cnt_lag_168h"].to_numpy(dtype=float)
        p = np.where(np.isnan(p), X["cnt_lag_24h"].to_numpy(dtype=float), p)
        return np.where(np.isnan(p), self.fallback_, p)


def build(params: dict):
    if params["kind"] == "baseline":
        return SeasonalNaive()  # not wrapped: predicts counts directly
    if params["kind"] == "baseline_prev_hour":
        return PreviousHourNaive()
    if params["kind"] == "ridge":
        pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
                                 ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                                                   ("log", FunctionTransformer(np.log1p))]), LAGS)],
                                remainder="passthrough")
        reg = Pipeline([("pre", pre), ("ridge", Ridge(alpha=params["alpha"]))])
    else:
        reg = HistGradientBoostingRegressor(
            max_iter=params["max_iter"], learning_rate=params["learning_rate"],
            max_leaf_nodes=params["max_leaf_nodes"], random_state=17,
            categorical_features=[FEATURES.index(c) for c in ("season", "weathersit", "weekday")])
    return TransformedTargetRegressor(regressor=reg, func=np.log1p, inverse_func=np.expm1)


def fit(df: pd.DataFrame, params: dict):
    m = build(params)
    if params["kind"].startswith("baseline"):
        return m.fit(df[FEATURES], df[TARGET])
    m.fit(df[FEATURES], df[TARGET])
    return m


def scores(y_true, y_pred) -> dict:
    y_pred = np.clip(y_pred, 0, None)
    return {"mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "r2": float(r2_score(y_true, y_pred))}
