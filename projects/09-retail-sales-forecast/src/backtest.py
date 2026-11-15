"""
Rolling-origin back-test of monthly retail-sales forecasts for 13 Irish
retail sectors (CSO Retail Sales Index, chain-linked 2015-2026).

Models compared at every origin (forecast 1-6 months ahead):
  naive, seasonal_naive, seasonal_naive_growth (seasonal naive x trailing
  YoY growth), ets (Holt-Winters on logs, per series), global_gbm_level
  (round-1 global gradient boosting on level-scaled lags), global_gbm_yoy
  (round-2 global gradient boosting on year-on-year ratios, lockdown
  rows excluded) and ensemble (mean of seasonal_naive_growth and
  global_gbm_yoy).

Round 1 of this project only had naive, seasonal_naive, ets, the level
GBM and an ets+GBM ensemble; seasonal naive won. The YoY models were
added in round 2 *after* seeing that result, so the back-test was used
twice. That is reported in the README as a limitation.

Metrics: MASE (scaled by in-sample seasonal-naive MAE on post-2021
months up to the origin) and sMAPE, overall, per series and per horizon. Empirical 80% prediction
intervals are built from log forecast errors of *earlier* origins only,
and their coverage is checked on later origins.

    python src/backtest.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from forecast_lib import (  # noqa: E402
    H, ROOT, ets, global_gbm_all, global_gbm_yoy_all, load_wide, mase_scale, naive,
    seasonal_naive, seasonal_naive_growth, smape,
)

REPORTS = ROOT / "reports"
MODELS = ROOT / "models"
N_ORIGINS = 25
# Round-1 overall results (recorded 15 Nov 2026 before the YoY models were added;
# MASE there used the full-history scale, so only the ranking is comparable).
ROUND1 = json.loads((ROOT / "reports" / "round1_metrics.json").read_text())["overall"] \
    if (ROOT / "reports" / "round1_metrics.json").exists() else None  # monthly origins; the last one still has 6 observed months after it


def main() -> None:
    REPORTS.mkdir(exist_ok=True); MODELS.mkdir(exist_ok=True)
    wide = load_wide()
    last = wide.index[-1]
    origins = pd.date_range(end=last - pd.DateOffset(months=H), periods=N_ORIGINS, freq="MS")
    print(f"series={wide.shape[1]} months={wide.shape[0]} ({wide.index[0]:%Y-%m}..{last:%Y-%m})")
    print(f"origins {origins[0]:%Y-%m}..{origins[-1]:%Y-%m}")

    rows = []
    for o in origins:
        gbm = global_gbm_all(wide, o)
        gbm_yoy = global_gbm_yoy_all(wide, o)
        for col in wide.columns:
            y_tr = wide[col].loc[:o]
            actual = wide[col].loc[o + pd.DateOffset(months=1): o + pd.DateOffset(months=H)].to_numpy()
            scale = mase_scale(y_tr)
            f = {
                "naive": naive(y_tr), "seasonal_naive": seasonal_naive(y_tr),
                "seasonal_naive_growth": seasonal_naive_growth(y_tr),
                "ets": ets(y_tr), "global_gbm_level": gbm[col], "global_gbm_yoy": gbm_yoy[col],
            }
            f["ensemble"] = (f["seasonal_naive_growth"] + f["global_gbm_yoy"]) / 2
            for m, pred in f.items():
                for h in range(H):
                    rows.append({
                        "origin": o, "series": col, "model": m, "h": h + 1,
                        "target_date": o + pd.DateOffset(months=h + 1),
                        "actual": actual[h], "forecast": float(pred[h]),
                        "abs_scaled_err": abs(actual[h] - pred[h]) / scale,
                        "log_err": float(np.log(actual[h] / pred[h])),
                    })
        print("origin", f"{o:%Y-%m}", "done", flush=True)
    bt = pd.DataFrame(rows)
    bt.to_csv(REPORTS / "backtest_forecasts.csv", index=False)

    def agg(g):
        return pd.Series({"MASE": g.abs_scaled_err.mean(), "sMAPE": smape(g.actual, g.forecast)})

    overall = bt.groupby("model").apply(agg, include_groups=False).sort_values("MASE")
    by_h = bt.groupby(["model", "h"]).apply(agg, include_groups=False)["MASE"].unstack("h")
    by_series = bt.groupby(["series", "model"]).apply(agg, include_groups=False)["MASE"].unstack("model")
    wins = by_series.idxmin(axis=1).value_counts().to_dict()
    print(overall.round(4)); print(by_h.round(3))

    # ---- empirical 80% intervals: quantiles from the first half of origins ----
    best = overall.index[0]
    half = origins[N_ORIGINS // 2]
    cal = bt[(bt.model == best) & (bt.target_date <= half)]
    q = cal.groupby("h").log_err.quantile([0.1, 0.9]).unstack()
    ev = bt[(bt.model == best) & (bt.origin > half)].copy()
    ev["lo"] = ev.forecast * np.exp(ev.h.map(q[0.1]))
    ev["hi"] = ev.forecast * np.exp(ev.h.map(q[0.9]))
    coverage = float(((ev.actual >= ev.lo) & (ev.actual <= ev.hi)).mean())
    print(f"80% interval coverage on later origins ({best}): {coverage:.3f}")

    # ---- final fit from the last month, with intervals from all origins ----
    q_all = bt[bt.model == best].groupby("h").log_err.quantile([0.1, 0.9]).unstack()
    fut_idx = pd.date_range(last, periods=H + 1, freq="MS")[1:]
    gbm_f, gbm_yoy_f = global_gbm_all(wide, last), global_gbm_yoy_all(wide, last)
    future = {}
    for col in wide.columns:
        y = wide[col]
        sng = seasonal_naive_growth(y)
        pred = {"ets": lambda: ets(y), "global_gbm_level": lambda: gbm_f[col],
                "global_gbm_yoy": lambda: gbm_yoy_f[col], "ensemble": lambda: (sng + gbm_yoy_f[col]) / 2,
                "seasonal_naive_growth": lambda: sng, "seasonal_naive": lambda: seasonal_naive(y),
                "naive": lambda: naive(y)}[best]()
        future[col] = pd.DataFrame({
            "date": fut_idx, "forecast": pred,
            "lo80": pred * np.exp(q_all[0.1].to_numpy()), "hi80": pred * np.exp(q_all[0.9].to_numpy()),
        })
    fut_df = pd.concat([d.assign(series=s) for s, d in future.items()])
    fut_df.to_csv(REPORTS / "future_forecast.csv", index=False)

    # ---- plots ----
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for m in by_h.index:
        ax.plot(by_h.columns, by_h.loc[m], marker="o", label=m, lw=2.5 if m == best else 1.2)
    ax.axhline(1.0, color="grey", ls=":", lw=1)
    ax.set_xlabel("months ahead"); ax.set_ylabel("MASE (lower is better)")
    ax.set_title(f"Back-test accuracy by horizon ({N_ORIGINS} origins x 13 sectors)")
    ax.legend(); fig.tight_layout(); fig.savefig(REPORTS / "mase_by_horizon.png", dpi=130); plt.close(fig)

    show = ["Supermarkets / non-specialised food", "Clothing, footwear & textiles", "Bars", "Electrical goods"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    for ax, col in zip(axes.ravel(), show):
        hist = wide[col].loc["2022-01":]
        ax.plot(hist.index, hist.values, color="black", lw=1.2, label="actual (CSO)")
        f = future[col]
        ax.plot(f.date, f.forecast, color="#2b6cb0", label=f"{best} forecast")
        ax.fill_between(f.date, f.lo80, f.hi80, color="#2b6cb0", alpha=0.2, label="80% interval")
        one = bt[(bt.model == best) & (bt.series == col) & (bt.h == 1)]
        ax.plot(one.target_date, one.forecast, color="#dd6b20", lw=1, ls="--", label="1-month-ahead back-test")
        ax.set_title(col, fontsize=10)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Irish Retail Sales Index (value, unadjusted): back-test and 6-month forecast")
    fig.tight_layout(); fig.savefig(REPORTS / "forecast_examples.png", dpi=120); plt.close(fig)

    metrics = {
        "run_at_europe_dublin": datetime.now(ZoneInfo("Europe/Dublin")).isoformat(timespec="seconds"),
        "data": {"series": int(wide.shape[1]), "months": int(wide.shape[0]),
                 "start": f"{wide.index[0]:%Y-%m}", "end": f"{last:%Y-%m}"},
        "protocol": {"origins": N_ORIGINS, "first_origin": f"{origins[0]:%Y-%m}",
                     "last_origin": f"{origins[-1]:%Y-%m}", "horizon_months": H,
                     "forecasts_per_model": int(len(bt) / bt.model.nunique())},
        "overall": overall.round(4).to_dict(orient="index"),
        "mase_by_horizon": {m: {int(h): round(float(v), 4) for h, v in r.items()} for m, r in by_h.iterrows()},
        "mase_by_series": by_series.round(4).to_dict(orient="index"),
        "series_wins_by_mase": {k: int(v) for k, v in wins.items()},
        "best_model": best,
        "round1_results_before_yoy_models": ROUND1,
        "interval_80_coverage_later_half": round(coverage, 4),
        "interval_eval_points": int(len(ev)),
    }
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    joblib.dump({"best_model": best, "interval_quantiles": q_all.to_dict()}, MODELS / "forecast_config.joblib")
    print("wrote", REPORTS / "metrics.json")


if __name__ == "__main__":
    main()
