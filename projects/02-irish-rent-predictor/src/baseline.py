"""Persistence baseline for official average rents.

Kept in its own module so the saved joblib bundle can be loaded from the
Streamlit app as well as from train.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "avg_monthly_rent_eur"


class PersistenceBaseline:
    """
    Naive but strong baseline for official averages that move slowly:

    1. most recent training-year value for the same location / type / beds
    2. else the training mean for that group
    3. else the training mean for the county / type / beds
    4. else the global training mean
    """

    def fit(self, frame: pd.DataFrame):
        self.global_mean_ = float(frame[TARGET].mean())
        last = (
            frame.sort_values("year")
            .groupby(["location", "property_type", "n_bedrooms"], as_index=False)
            .tail(1)
        )
        self.last_lookup_ = {
            (r.location, r.property_type, int(r.n_bedrooms)): float(
                r.avg_monthly_rent_eur
            )
            for r in last.itertuples()
        }
        self.group_mean_ = (
            frame.groupby(["location", "property_type", "n_bedrooms"])[TARGET]
            .mean()
            .to_dict()
        )
        self.county_mean_ = (
            frame.groupby(["county", "property_type", "n_bedrooms"])[TARGET]
            .mean()
            .to_dict()
        )
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        out = []
        for r in frame.itertuples():
            key = (r.location, r.property_type, int(r.n_bedrooms))
            if key in self.last_lookup_:
                out.append(self.last_lookup_[key])
                continue
            if key in self.group_mean_:
                out.append(self.group_mean_[key])
                continue
            ckey = (r.county, r.property_type, int(r.n_bedrooms))
            if ckey in self.county_mean_:
                out.append(self.county_mean_[ckey])
                continue
            out.append(self.global_mean_)
        return np.asarray(out, dtype=float)
