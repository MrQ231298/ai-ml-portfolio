# Code walkthrough

Read in this order. Paths are relative to this project folder.

1. **`src/data.py`**: `fetch()` downloads NASA's archive once, checks its
   SHA-256 and extracts the inner `CMAPSSData.zip`. `load(fd)` reads the
   26 space-separated columns and adds `rul = last cycle − cycle` for
   training engines. It returns the truncated official test engines and
   their true RUL. `split_units()` shuffles **engine ids** (seeded) into
   60 / 20 / 20 fit / validation / test.

2. **`src/features.py`**: `informative_sensors()` drops channels with no
   variation. `build()` makes per-engine, past-only features: raw value,
   rolling mean/std (10), rolling least-squares slope (20, from rolling
   moments) and EWMA (α = 0.1), plus the cycle count.

3. **`src/anomaly.py`**: `HealthModel` standardises the EWMA sensors on
   the healthy cycles (first 30 of each fit engine) and fits Ledoit-Wolf
   covariance, PCA (90% variance) and an Isolation Forest. `score()`
   gives the limit check (max |z|), Mahalanobis distance, PCA squared
   residual or negated Isolation Forest score. `first_alarm()` needs k
   consecutive exceedances.

4. **`src/rul.py`**: the capped target (125), the two baselines (mean, and
   fleet median life − cycle), ridge and HGB factories with their small
   grids, RMSE and the PHM08 asymmetric score.

5. **`src/alerting.py`**: the cost model.
   - `Costs` holds the assumptions.
   - `unit_cost()` scores one engine from the true RUL at the alert.
   - `alert_from_threshold()` finds the first crossing.
   - `choose_threshold()` sweeps a grid on validation engines and keeps
     the cheapest.
   - `fixed_interval_alerts()` models a time-based schedule.

6. **`src/run.py`**: the recorded experiment, per sub-dataset.
   - It fits the health model and scores the four methods; the method
     with the best validation AUROC is used for the health-score policy.
   - It fits every RUL candidate on fit engines and selects on validation
     RMSE. It reports held-out and official-test metrics (refit on
     fit + validation for the official test).
   - It chooses each policy's parameter on validation by cost, reports
     on held-out engines, and repeats for four failure costs and five
     engine splits.
   - It writes `reports/metrics.json`, cost curves, per-cycle traces for
     validation/test engines, and official-test predictions.

7. **`src/charts.py`**: the two README figures per sub-dataset.

8. **`app.py`**: Streamlit replay of a held-out engine. It shows
   predicted RUL, health score, each policy's state and the sensors up to
   a chosen cycle, from the saved traces.

9. **`tests/test_pm.py`**: unit tests on a synthetic fleet (causality,
   scores, cost model, thresholds), checks of the recorded metrics, and
   an app render.
