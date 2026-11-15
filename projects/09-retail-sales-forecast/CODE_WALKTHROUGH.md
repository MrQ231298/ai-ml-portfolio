# Code walkthrough — Irish retail sales forecasting

## The pipeline at a glance

```
1. python src/fetch_cso_retail.py
        ↓ data/RSM05_raw.csv, data/RSM08_raw.csv (raw CSO downloads)
        ↓ data/irish_retail_sales_index_monthly.csv (13 chain-linked series)

2. OMP_NUM_THREADS=2 python src/backtest.py
        ↓ 25 origins × 13 series × 6 horizons × 7 models
        ↓ reports/backtest_forecasts.csv, future_forecast.csv, metrics.json, *.png

3. python src/compare_models.py   → adds a bootstrap check to metrics.json
4. pytest -q
5. streamlit run app.py
```

## What each file is for

| File | Purpose |
| --- | --- |
| [`src/fetch_cso_retail.py`](src/fetch_cso_retail.py) | Downloads both CSO tables, keeps 13 sectors, chain-links with month-specific ratios, saves diagnostics. |
| [`src/forecast_lib.py`](src/forecast_lib.py) | All models (naive, seasonal naive, growth-adjusted, ETS, two global GBMs) and the MASE / sMAPE metrics. |
| [`src/backtest.py`](src/backtest.py) | Rolling-origin back-test, summaries, empirical intervals, final 6-month forecast, plots. |
| [`src/compare_models.py`](src/compare_models.py) | Origin-block bootstrap of MASE ratios vs seasonal naive. |
| [`app.py`](app.py) | Streamlit + Altair dashboard. |
| [`tests/test_smoke.py`](tests/test_smoke.py) | Data shape, baseline logic, metrics, outputs present. |

## Key ideas in the code

**No peeking.** At each origin `o`, every model sees `wide.loc[:o]` only.
The global models rebuild their training table from that slice.

**A global model.** `global_gbm_yoy_all` stacks rows from all 13 sectors
into one table (sector code as a categorical feature), so the model
learns patterns shared across sectors from roughly 1,000 rows instead of under 100
per sector.

**Direct multi-horizon.** One model per horizon h = 1…6 predicts
`y[t+h]` straight from information at `t`, which avoids feeding a model
its own errors (the "recursive" approach used in project 4).

**Year-on-year target.** The YoY model predicts `log(y[t+h] / y[t+h-12])`
and multiplies back by last year's value. That builds seasonality in for
free and lets the model focus on growth.

**Dropping the lockdown.** Training rows whose features or target fall in
March 2020 – December 2021 are skipped (`COVID_START`, `COVID_END`).

**MASE scale.** `mase_scale` uses the seasonal-naive error from January
2022 onward, so a model is not flattered by comparison with the chaotic
lockdown months.

**Empirical intervals.** The 10th / 90th percentiles of past
`log(actual / forecast)` per horizon, applied multiplicatively.
