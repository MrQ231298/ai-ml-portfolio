"""Unsupervised health monitoring: learn 'normal' from the start of each engine's life only.

C-MAPSS engines are described as operating normally at the start of every
series, so the first HEALTHY_CYCLES cycles of the fit engines are the only
training data. No failure information is used to fit or threshold these models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

HEALTHY_CYCLES = 30
CONSECUTIVE = 3


class HealthModel:
    """Scores are computed on per-engine EWMA-smoothed sensors (the *_ewm feature columns)."""

    methods = ("limit_check", "mahalanobis", "pca_spe", "isolation_forest")

    def __init__(self, sensors: list[str], seed: int = 21):
        self.cols = [f"{s}_ewm" for s in sensors]
        self.seed = seed

    def _z(self, X: pd.DataFrame) -> np.ndarray:
        return (X[self.cols].to_numpy() - self.mu) / self.sd

    def fit(self, X: pd.DataFrame, healthy_mask: np.ndarray) -> "HealthModel":
        H = X.loc[healthy_mask, self.cols].to_numpy()
        self.mu, self.sd = H.mean(0), H.std(0) + 1e-9
        Z = (H - self.mu) / self.sd
        self.cov = LedoitWolf().fit(Z)
        self.pca = PCA(n_components=0.90, random_state=self.seed).fit(Z)
        self.iso = IsolationForest(n_estimators=200, random_state=self.seed).fit(Z)
        return self

    def score(self, X: pd.DataFrame, method: str) -> np.ndarray:
        Z = self._z(X)
        if method == "limit_check":          # classic per-sensor limit: worst |z| across sensors
            return np.abs(Z).max(1)
        if method == "mahalanobis":
            return self.cov.mahalanobis(Z)
        if method == "pca_spe":              # squared prediction error outside the healthy subspace
            R = Z - self.pca.inverse_transform(self.pca.transform(Z))
            return (R ** 2).sum(1)
        if method == "isolation_forest":
            return -self.iso.score_samples(Z)
        raise ValueError(method)


def first_alarm(scores: np.ndarray, threshold: float, k: int = CONSECUTIVE) -> int | None:
    """Index of the first cycle that completes a run of k consecutive scores >= threshold."""
    run = 0
    for i, s in enumerate(scores):
        run = run + 1 if s >= threshold else 0
        if run >= k:
            return i
    return None
