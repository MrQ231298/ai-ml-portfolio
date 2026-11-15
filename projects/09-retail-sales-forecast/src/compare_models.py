"""
Is the best model's advantage over seasonal naive real, or noise?

Origin-block bootstrap: resample the 25 forecast origins with replacement
(keeping all 13 series x 6 horizons of an origin together, because those
errors are correlated), recompute MASE for each model, and report a 95%
interval for the MASE ratio model / seasonal_naive.

    python src/compare_models.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
B = 2000


def main() -> None:
    bt = pd.read_csv(ROOT / "reports" / "backtest_forecasts.csv", parse_dates=["origin"])
    per_origin = bt.groupby(["origin", "model"]).abs_scaled_err.agg(["sum", "count"]).unstack("model")
    origins = per_origin.index.to_numpy()
    rng = np.random.default_rng(0)
    out = {}
    for m in ["ensemble", "global_gbm_yoy", "seasonal_naive_growth", "global_gbm_level", "ets"]:
        ratios = []
        for _ in range(B):
            idx = rng.integers(0, len(origins), len(origins))
            s = per_origin.iloc[idx]
            mase_m = s[("sum", m)].sum() / s[("count", m)].sum()
            mase_s = s[("sum", "seasonal_naive")].sum() / s[("count", "seasonal_naive")].sum()
            ratios.append(mase_m / mase_s)
        point = (per_origin[("sum", m)].sum() / per_origin[("count", m)].sum()) / \
                (per_origin[("sum", "seasonal_naive")].sum() / per_origin[("count", "seasonal_naive")].sum())
        lo, hi = np.percentile(ratios, [2.5, 97.5])
        out[m] = {"mase_ratio_vs_seasonal_naive": round(float(point), 4),
                  "ci95": [round(float(lo), 4), round(float(hi), 4)],
                  "share_of_bootstrap_better": round(float(np.mean(np.array(ratios) < 1)), 4)}
        print(m, out[m])
    metrics_path = ROOT / "reports" / "metrics.json"
    m = json.loads(metrics_path.read_text())
    m["bootstrap_vs_seasonal_naive"] = {"method": f"origin-block bootstrap, B={B}, seed 0", "results": out}
    metrics_path.write_text(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
