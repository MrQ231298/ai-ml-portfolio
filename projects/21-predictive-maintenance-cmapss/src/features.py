"""Causal per-engine features: every value at cycle t uses cycles <= t only."""

from __future__ import annotations

import numpy as np
import pandas as pd

ROLL, SLOPE, EWM_ALPHA = 10, 20, 0.1


def informative_sensors(train: pd.DataFrame, tol: float = 1e-4) -> list[str]:
    """Sensors that actually vary in the fit data (several C-MAPSS channels are constant)."""
    s = [c for c in train.columns if c.startswith("s") and c[1:].isdigit()]
    return [c for c in s if train[c].std() / (abs(train[c].mean()) + 1e-9) > tol]


def _slope(g: pd.Series, w: int) -> pd.Series:
    t = pd.Series(np.arange(len(g), dtype=float), index=g.index)
    mt, mx = t.rolling(w, min_periods=3).mean(), g.rolling(w, min_periods=3).mean()
    mtx, mtt = (t * g).rolling(w, min_periods=3).mean(), (t * t).rolling(w, min_periods=3).mean()
    return ((mtx - mt * mx) / (mtt - mt * mt)).fillna(0.0)


def build(df: pd.DataFrame, sensors: list[str]) -> pd.DataFrame:
    df = df.sort_values(["unit", "cycle"])
    out = {"cycle": df.cycle.to_numpy(dtype=float)}
    g = df.groupby("unit", sort=False)
    for s in sensors:
        out[s] = df[s].to_numpy()
        out[f"{s}_mean{ROLL}"] = g[s].transform(lambda x: x.rolling(ROLL, min_periods=1).mean()).to_numpy()
        out[f"{s}_std{ROLL}"] = g[s].transform(lambda x: x.rolling(ROLL, min_periods=2).std()).fillna(0).to_numpy()
        out[f"{s}_slope{SLOPE}"] = g[s].transform(lambda x: _slope(x, SLOPE)).to_numpy()
        out[f"{s}_ewm"] = g[s].transform(lambda x: x.ewm(alpha=EWM_ALPHA, adjust=False).mean()).to_numpy()
    return pd.DataFrame(out, index=df.index)
