"""
Forecasting models and metrics for the Irish retail sales project.

All models produce a 1..H month-ahead forecast from a single origin,
using only data up to and including that origin.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "irish_retail_sales_index_monthly.csv"
H = 6           # forecast horizon in months
SEASON = 12
N_LAGS = 13


def load_wide(path: Path = DATA_PATH) -> pd.DataFrame:
    """Return a wide frame: index = month start, columns = sector series."""
    df = pd.read_csv(path, parse_dates=["date"])
    return df.pivot(index="date", columns="series", values="value").sort_index().asfreq("MS")


# ---------------------------------------------------------------- baselines
def naive(y: pd.Series, h: int = H) -> np.ndarray:
    return np.repeat(y.iloc[-1], h)


def seasonal_naive(y: pd.Series, h: int = H) -> np.ndarray:
    return np.array([y.iloc[-SEASON + (i % SEASON)] for i in range(h)])


def seasonal_naive_growth(y: pd.Series, h: int = H) -> np.ndarray:
    """Seasonal naive scaled by trailing year-on-year growth (last 12 vs prior 12 months)."""
    g = y.iloc[-12:].sum() / y.iloc[-24:-12].sum()
    return seasonal_naive(y, h) * g


def ets(y: pd.Series, h: int = H) -> np.ndarray:
    """Holt-Winters on log values: damped additive trend, additive seasonality."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = ExponentialSmoothing(
            np.log(y.to_numpy()), trend="add", damped_trend=True,
            seasonal="add", seasonal_periods=SEASON, initialization_method="estimated",
        ).fit(optimized=True)
    return np.exp(fit.forecast(h))


# ------------------------------------------------- global gradient boosting
def _features(window: np.ndarray, month: int, series_code: int) -> list[float]:
    """Features from the last N_LAGS values, scaled by the last-12-month mean."""
    level = window[-SEASON:].mean()
    lags = list(window[::-1][:N_LAGS] / level)          # lag1..lag13
    return lags + [month, series_code, np.log(level)]


def build_training(wide: pd.DataFrame, end: pd.Timestamp, h: int):
    """Rows for a direct h-step model using targets observed up to `end`."""
    X, y = [], []
    dates = wide.loc[:end].index
    for code, col in enumerate(wide.columns):
        s = wide[col].loc[:end].to_numpy()
        for t in range(N_LAGS - 1, len(s) - h):
            win = s[: t + 1]
            level = win[-SEASON:].mean()
            X.append(_features(win, dates[t + h].month, code))
            y.append(np.log(s[t + h] / level))
    return np.array(X), np.array(y)


def global_gbm_all(wide: pd.DataFrame, origin: pd.Timestamp, h: int = H) -> dict[str, np.ndarray]:
    """One direct model per horizon, trained across *all* series (global model)."""
    preds = {c: np.zeros(h) for c in wide.columns}
    future_months = pd.date_range(origin, periods=h + 1, freq="MS")[1:]
    for step in range(1, h + 1):
        X, y = build_training(wide, origin, step)
        model = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20,
            categorical_features=[N_LAGS + 1], random_state=0,
        ).fit(X, y)
        for code, col in enumerate(wide.columns):
            win = wide[col].loc[:origin].to_numpy()
            x = np.array([_features(win, future_months[step - 1].month, code)])
            preds[col][step - 1] = np.exp(model.predict(x)[0]) * win[-SEASON:].mean()
    return preds


# ------------------------------------ global gradient boosting, YoY target
COVID_START, COVID_END = pd.Timestamp("2020-03-01"), pd.Timestamp("2021-12-01")


def _yoy_features(s: np.ndarray, t: int, target_month: int, code: int) -> list[float]:
    """Year-on-year log ratios of the last 6 months, recent month-on-month
    changes, the target calendar month and the series code."""
    yoy = [np.log(s[t - k] / s[t - k - 12]) for k in range(6)]
    mom = [np.log(s[t - k] / s[t - k - 1]) for k in range(3)]
    return yoy + mom + [np.mean(yoy[:3]), target_month, code]


def global_gbm_yoy_all(wide: pd.DataFrame, origin: pd.Timestamp, h: int = H) -> dict[str, np.ndarray]:
    """Direct multi-horizon global model whose target is the year-on-year
    log ratio log(y[t+h] / y[t+h-12]); the forecast is then the value
    12 months earlier times exp(prediction). Training rows that touch the
    COVID-19 lockdown period (Mar 2020 - Dec 2021) in either the features
    or the target are dropped, because that regime is not expected to recur."""
    dates = wide.loc[:origin].index
    future_months = pd.date_range(origin, periods=h + 1, freq="MS")[1:]
    preds = {c: np.zeros(h) for c in wide.columns}
    cat_idx = 10  # position of the series code in the feature list
    for step in range(1, h + 1):
        X, y = [], []
        for code, col in enumerate(wide.columns):
            s = wide[col].loc[:origin].to_numpy()
            for t in range(18, len(s) - step):
                first, last = dates[t - 18], dates[t + step]
                if last >= COVID_START and first <= COVID_END:
                    continue
                X.append(_yoy_features(s, t, dates[t + step].month, code))
                y.append(np.log(s[t + step] / s[t + step - 12]))
        model = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20,
            categorical_features=[cat_idx], random_state=0,
        ).fit(np.array(X), np.array(y))
        for code, col in enumerate(wide.columns):
            s = wide[col].loc[:origin].to_numpy()
            t = len(s) - 1
            x = np.array([_yoy_features(s, t, future_months[step - 1].month, code)])
            preds[col][step - 1] = s[t + step - 12] * np.exp(model.predict(x)[0])
    return preds


# ----------------------------------------------------------------- metrics
def smape(a, f) -> float:
    a, f = np.asarray(a, float), np.asarray(f, float)
    return float(np.mean(2 * np.abs(f - a) / (np.abs(a) + np.abs(f))) * 100)


def mase_scale(y_train: pd.Series) -> float:
    """In-sample MAE of the seasonal naive forecast (Hyndman & Koehler, 2006),
    computed on post-pandemic months only (targets from Jan 2022), so the
    huge 2020-21 lockdown errors do not inflate the scale."""
    err = (y_train - y_train.shift(SEASON)).abs()
    return float(err.loc["2022-01-01":].mean())
