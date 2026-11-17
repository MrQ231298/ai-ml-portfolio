"""Data drift statistics: Population Stability Index and two-sample KS test."""

from __future__ import annotations

import numpy as np
from scipy import stats

PSI_WARN, PSI_ALERT = 0.10, 0.25


def psi(reference, current, bins: int = 10, categorical: bool = False, eps: float = 1e-4) -> float:
    """PSI = sum((c - r) * ln(c / r)) over bins. Numeric bins are reference quantiles."""
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    if categorical:
        cats = np.union1d(np.unique(ref), np.unique(cur))
        r = np.array([(ref == c).mean() for c in cats])
        c = np.array([(cur == k).mean() for k in cats])
    else:
        edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        r = np.histogram(ref, edges)[0] / len(ref)
        c = np.histogram(cur, edges)[0] / len(cur)
    r = np.clip(r, eps, None)
    c = np.clip(c, eps, None)
    return float(np.sum((c - r) * np.log(c / r)))


def ks(reference, current) -> tuple[float, float]:
    res = stats.ks_2samp(reference, current)
    return float(res.statistic), float(res.pvalue)


def level(value: float) -> str:
    return "alert" if value >= PSI_ALERT else "warn" if value >= PSI_WARN else "ok"
