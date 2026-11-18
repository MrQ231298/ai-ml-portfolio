"""Remaining-useful-life regression: two naive baselines, ridge and gradient boosting."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RUL_CAP = 125  # piecewise-linear target: early life is treated as 'healthy', RUL capped (Heimes, 2008)


def capped(rul):
    return np.minimum(rul, RUL_CAP)


class MeanBaseline:
    def fit(self, X, y):
        self.m = float(np.mean(y)); return self

    def predict(self, X):
        return np.full(len(X), self.m)


class FleetLifeBaseline:
    """What a time-based schedule assumes: every engine lives the fleet median; RUL = median life - cycle."""

    def __init__(self, median_life: float):
        self.life = median_life

    def fit(self, X, y):
        return self

    def predict(self, X):
        return capped(np.clip(self.life - X["cycle"].to_numpy(), 0, None))


def ridge(alpha=10.0):
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))


def hgb(lr=0.05, leaves=15, seed=21):
    return HistGradientBoostingRegressor(learning_rate=lr, max_leaf_nodes=leaves, max_iter=300,
                                         min_samples_leaf=40, l2_regularization=1.0, random_state=seed)


HGB_GRID = [{"lr": lr, "leaves": lv} for lr in (0.03, 0.1) for lv in (15, 31)]
RIDGE_GRID = [{"alpha": a} for a in (1.0, 10.0, 100.0)]


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def nasa_score(y_true, y_pred):
    """PHM08 asymmetric score: late predictions (d > 0) are penalised more than early ones."""
    d = np.asarray(y_pred) - np.asarray(y_true)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))
