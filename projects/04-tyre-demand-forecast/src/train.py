"""
Back-test weekly tyre-sales forecasts on SYNTHETIC data.

Models (all evaluated on the same time-based origins):
  1. Seasonal naive — same week last year (ŷ_t = y_{t-52}).
  2. ETS (Holt–Winters) — additive trend and additive weekly seasonality
     of period 52, via statsmodels ExponentialSmoothing.
  3. Gradient boosting with lag features — recursive multi-step forecasts.

The main evaluation is an *expanding-window back-test*, not a shuffled
split: at each origin we forecast the next H weeks using only data up
to that origin, then step the origin forward.

A single 2025 hold-out is also scored and plotted so the README can
show one clear picture.

Usage
-----
    python src/train.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing

sys.path.insert(0, str(Path(__file__).resolve().parent))

RANDOM_STATE = 42
HORIZON = 8          # weeks ahead at each back-test origin
STEP = 4             # weeks between origins
MIN_TRAIN_WEEKS = 52 * 3
HOLDOUT_START = pd.Timestamp("2025-01-06")  # first Monday of 2025 in this series
LAGS = (1, 2, 4, 13, 26, 52)
ROLLS = (4, 13)


def project_paths() -> dict[str, Path]:
    root = Path(__file__).resolve().parents[1]
    return {
        "root": root,
        "data": root / "data" / "SYNTHETIC_weekly_tyre_sales.csv",
        "models": root / "models",
        "reports": root / "reports",
    }


def metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 2),
        "mape": round(float(mean_absolute_percentage_error(y_true, y_pred) * 100), 2),
        "n": int(len(y_true)),
    }


# ---------------------------------------------------------------------------
# Feature helpers for the lag model
# ---------------------------------------------------------------------------

def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for lag in LAGS:
        out[f"lag_{lag}"] = out["units_sold"].shift(lag)
    for w in ROLLS:
        out[f"roll_mean_{w}"] = out["units_sold"].shift(1).rolling(w).mean()
    out["week_sin"] = np.sin(2 * np.pi * out["week_of_year"] / 52.0)
    out["week_cos"] = np.cos(2 * np.pi * out["week_of_year"] / 52.0)
    out["t"] = np.arange(len(out), dtype=float)
    return out


FEATURE_COLS = [f"lag_{lag}" for lag in LAGS] + [f"roll_mean_{w}" for w in ROLLS] + [
    "week_of_year",
    "week_sin",
    "week_cos",
    "t",
]


def fit_boosting(train: pd.DataFrame) -> HistGradientBoostingRegressor:
    feat = add_features(train)
    usable = feat.dropna(subset=FEATURE_COLS)
    model = HistGradientBoostingRegressor(
        max_depth=6,
        max_iter=200,
        learning_rate=0.06,
        min_samples_leaf=8,
        random_state=RANDOM_STATE,
    )
    model.fit(usable[FEATURE_COLS], usable["units_sold"])
    return model


def recursive_boosting_forecast(
    model: HistGradientBoostingRegressor, history: pd.DataFrame, horizon: int
) -> np.ndarray:
    """
    Walk forward one week at a time, appending each prediction so later
    lags can use it. Only history up to the origin is used as actuals.
    """
    work = history[["week_start", "year", "week_of_year", "units_sold"]].copy()
    preds = []
    for step in range(horizon):
        last = work["week_start"].iloc[-1]
        nxt = last + pd.Timedelta(days=7)
        iso = nxt.isocalendar()
        work = pd.concat(
            [
                work,
                pd.DataFrame(
                    {
                        "week_start": [nxt],
                        "year": [int(iso.year)],
                        "week_of_year": [int(iso.week)],
                        "units_sold": [np.nan],
                    }
                ),
            ],
            ignore_index=True,
        )
        feat = add_features(work)
        row = feat.iloc[[-1]][FEATURE_COLS]
        if row.isna().any(axis=None):
            # If a lag is not yet available (should not happen with 3+ years
            # of history), fall back to last actual.
            pred = float(history["units_sold"].iloc[-1])
        else:
            pred = float(model.predict(row)[0])
        work.loc[work.index[-1], "units_sold"] = pred
        preds.append(pred)
    return np.asarray(preds, dtype=float)


def seasonal_naive_forecast(history: pd.DataFrame, horizon: int) -> np.ndarray:
    y = history["units_sold"].to_numpy(dtype=float)
    out = []
    for h in range(1, horizon + 1):
        idx = len(y) - 52 + h - 1
        if 0 <= idx < len(y):
            out.append(y[idx])
        else:
            out.append(y[-52] if len(y) >= 52 else y[-1])
    return np.asarray(out, dtype=float)


def ets_forecast(history: pd.DataFrame, horizon: int) -> np.ndarray:
    y = history["units_sold"].to_numpy(dtype=float)
    model = ExponentialSmoothing(
        y,
        trend="add",
        seasonal="add",
        seasonal_periods=52,
        initialization_method="estimated",
    )
    fitted = model.fit(optimized=True)
    return np.asarray(fitted.forecast(horizon), dtype=float)


def backtest(df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """
    Expanding-window back-test. Origins run from MIN_TRAIN_WEEKS until
    there are HORIZON actuals left at the end of the series.
    """
    records = []
    n = len(df)
    last_origin = n - HORIZON
    for origin in range(MIN_TRAIN_WEEKS, last_origin + 1, STEP):
        history = df.iloc[:origin].copy()
        actual = df.iloc[origin : origin + HORIZON]["units_sold"].to_numpy(dtype=float)
        # Three forecasts, all from the same origin.
        forecasts = {
            "seasonal_naive": seasonal_naive_forecast(history, HORIZON),
            "ets": ets_forecast(history, HORIZON),
            "gradient_boosting": recursive_boosting_forecast(
                fit_boosting(history), history, HORIZON
            ),
        }
        origin_week = history["week_start"].iloc[-1]
        for name, pred in forecasts.items():
            records.append(
                {
                    "origin": str(origin_week.date()),
                    "model": name,
                    **metrics(actual, pred),
                }
            )
    rec_df = pd.DataFrame.from_records(records)
    summary = {}
    for name, part in rec_df.groupby("model"):
        summary[name] = {
            "mae": round(float(part["mae"].mean()), 2),
            "mape": round(float(part["mape"].mean()), 2),
            "n_origins": int(len(part)),
            "horizon_weeks": HORIZON,
            "step_weeks": STEP,
        }
    return summary, rec_df


def holdout_forecasts(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    train = df[df["week_start"] < HOLDOUT_START].copy()
    test = df[df["week_start"] >= HOLDOUT_START].copy()
    horizon = len(test)
    forecasts = {
        "seasonal_naive": seasonal_naive_forecast(train, horizon),
        "ets": ets_forecast(train, horizon),
        "gradient_boosting": recursive_boosting_forecast(fit_boosting(train), train, horizon),
    }
    actual = test["units_sold"].to_numpy(dtype=float)
    scores = {name: metrics(actual, pred) for name, pred in forecasts.items()}
    out = test[["week_start", "units_sold"]].rename(columns={"units_sold": "actual"}).copy()
    for name, pred in forecasts.items():
        out[name] = pred
    return out, scores


def plot_holdout(df: pd.DataFrame, holdout: pd.DataFrame, out: Path) -> None:
    context = df[df["week_start"] >= "2023-01-01"]
    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    ax.plot(context["week_start"], context["units_sold"], color="#333", lw=1.2, label="actual")
    colours = {
        "seasonal_naive": "#1f77b4",
        "ets": "#ff7f0e",
        "gradient_boosting": "#2ca02c",
    }
    labels = {
        "seasonal_naive": "seasonal naive",
        "ets": "ETS (Holt–Winters)",
        "gradient_boosting": "gradient boosting (lags)",
    }
    for name in colours:
        ax.plot(holdout["week_start"], holdout[name], color=colours[name], lw=1.6, label=labels[name])
    ax.axvline(HOLDOUT_START, color="grey", ls="--", lw=0.9, label="hold-out starts (2025)")
    ax.set_ylabel("Weekly units sold")
    ax.set_xlabel("Week starting")
    ax.set_title("SYNTHETIC weekly tyre sales — 2025 hold-out forecasts")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_backtest(rec_df: pd.DataFrame, out: Path) -> None:
    summary = rec_df.groupby("model")[["mae", "mape"]].mean().reindex(
        ["seasonal_naive", "ets", "gradient_boosting"]
    )
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
    labels = ["seasonal naive", "ETS", "gradient boosting"]
    axes[0].bar(labels, summary["mae"], color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    axes[0].set_ylabel("MAE (units)")
    axes[0].set_title(f"Back-test MAE (mean over origins, h={HORIZON})")
    axes[1].bar(labels, summary["mape"], color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    axes[1].set_ylabel("MAPE (%)")
    axes[1].set_title(f"Back-test MAPE (mean over origins, h={HORIZON})")
    for ax in axes:
        ax.tick_params(axis="x", labelrotation=15)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    paths = project_paths()
    paths["models"].mkdir(parents=True, exist_ok=True)
    paths["reports"].mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(paths["data"], parse_dates=["week_start"])
    if not df["data_source"].eq("SYNTHETIC_EXAMPLE_DATA").all():
        raise ValueError("Expected every row to be tagged SYNTHETIC_EXAMPLE_DATA.")
    df = df.sort_values("week_start").reset_index(drop=True)
    print(f"Loaded {len(df)} SYNTHETIC weeks "
          f"({df['week_start'].min().date()} to {df['week_start'].max().date()})")

    print(f"\nRunning expanding-window back-test (h={HORIZON}, step={STEP})…")
    backtest_summary, rec_df = backtest(df)
    rec_df.to_csv(paths["reports"] / "backtest_origins.csv", index=False)
    print("Back-test (mean over origins):")
    for name, scores in backtest_summary.items():
        print(f"  {name:22s}  MAE={scores['mae']:.2f}  MAPE={scores['mape']:.2f}%  "
              f"origins={scores['n_origins']}")

    holdout, holdout_scores = holdout_forecasts(df)
    holdout.to_csv(paths["reports"] / "holdout_2025_forecasts.csv", index=False)
    print("\n2025 hold-out (single origin at end of 2024):")
    for name, scores in holdout_scores.items():
        print(f"  {name:22s}  MAE={scores['mae']:.2f}  MAPE={scores['mape']:.2f}%  n={scores['n']}")

    forecast_png = paths["reports"] / "forecast_plot.png"
    backtest_png = paths["reports"] / "backtest_mae_mape.png"
    plot_holdout(df, holdout, forecast_png)
    plot_backtest(rec_df, backtest_png)

    # Persist the models fitted on everything up to the hold-out, for the app.
    train = df[df["week_start"] < HOLDOUT_START].copy()
    ets_fit = ExponentialSmoothing(
        train["units_sold"].to_numpy(dtype=float),
        trend="add",
        seasonal="add",
        seasonal_periods=52,
        initialization_method="estimated",
    ).fit(optimized=True)
    boosting = fit_boosting(train)
    bundle = {
        "train_end": str(train["week_start"].max().date()),
        "holdout_start": str(HOLDOUT_START.date()),
        "history": train,
        "full_series": df,
        "ets_params": {"trend": "add", "seasonal": "add", "seasonal_periods": 52},
        "boosting": boosting,
        "metrics": {
            "backtest": backtest_summary,
            "holdout_2025": holdout_scores,
            "backtest_design": {
                "horizon_weeks": HORIZON,
                "step_weeks": STEP,
                "min_train_weeks": MIN_TRAIN_WEEKS,
            },
        },
        "data_source": "SYNTHETIC_EXAMPLE_DATA",
    }
    model_path = paths["models"] / "tyre_demand_forecast.joblib"
    joblib.dump(bundle, model_path, compress=3)

    all_metrics = {
        "data_source": "SYNTHETIC_EXAMPLE_DATA",
        "n_weeks": int(len(df)),
        "start": str(df["week_start"].min().date()),
        "end": str(df["week_start"].max().date()),
        "backtest": backtest_summary,
        "holdout_2025": holdout_scores,
        "backtest_design": bundle["metrics"]["backtest_design"],
        "artefacts": {
            "forecast_plot_png": "reports/forecast_plot.png",
            "backtest_plot_png": "reports/backtest_mae_mape.png",
            "model_bundle": "models/tyre_demand_forecast.joblib",
        },
        "generated_at_dublin": datetime.now(ZoneInfo("Europe/Dublin")).strftime(
            "%Y-%m-%d %H:%M %Z"
        ),
    }
    # Pick the back-test winner by MAE for the app headline.
    winner = min(backtest_summary, key=lambda n: backtest_summary[n]["mae"])
    all_metrics["primary_model_backtest_mae"] = winner
    (paths["reports"] / "metrics.json").write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")

    lines = [
        "Tyre demand forecast — SYNTHETIC weekly series",
        f"{df['week_start'].min().date()} to {df['week_start'].max().date()}, n={len(df)}",
        f"Back-test: expanding window, horizon={HORIZON} weeks, step={STEP} weeks",
        "",
        "Back-test (mean over origins):",
    ]
    for name, scores in backtest_summary.items():
        lines.append(f"  {name:22s}  MAE={scores['mae']:.2f}  MAPE={scores['mape']:.2f}%")
    lines += ["", "2025 hold-out:"]
    for name, scores in holdout_scores.items():
        lines.append(f"  {name:22s}  MAE={scores['mae']:.2f}  MAPE={scores['mape']:.2f}%")
    lines.append(f"\nLowest back-test MAE: {winner}")
    (paths["reports"] / "metrics.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nLowest back-test MAE: {winner}")
    print(f"Saved {forecast_png}")
    print(f"Saved {model_path}")


if __name__ == "__main__":
    main()
