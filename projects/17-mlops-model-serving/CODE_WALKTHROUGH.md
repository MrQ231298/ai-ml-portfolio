# Code walkthrough

Read in this order. Paths are relative to this project folder.

1. **`src/common.py`**: the paths, the MLflow tracking URI (a local SQLite
   file under `mlruns/`), the data URL and checksum. `load_hourly()`
   downloads and verifies the zip and builds timestamps. `add_lags()`
   rebuilds a complete hourly grid with `asfreq("h")`, so lags never jump
   over missing hours.

2. **`src/model_lib.py`**: the two naive baselines, `build()` for ridge
   (one-hot plus imputed, log-scaled lags) and gradient boosting (native
   NaN handling), both wrapped in `TransformedTargetRegressor` with
   log1p/expm1. `scores()` gives MAE, RMSE and R².

3. **`src/train.py`**
   - It fits every candidate on 2011 and scores Q1 2012. Each candidate is
     one `mlflow.start_run` with params, metrics, tags (data checksum,
     licence) and a logged model with a signature.
   - MLflow 3 saves scikit-learn models with *skops*. `SKOPS_TRUSTED`
     lists the types we reviewed rather than trusting everything.
   - It registers the best model as `bike-demand`, sets the `champion`
     alias, and writes the serving artefacts to `models/`: the joblib
     model, a model card, and a reference profile of training features
     for drift checks.

4. **`src/drift.py`**: `psi()` uses reference-quantile bins for numbers,
   category shares for categories, and an epsilon to avoid log(0).
   `ks()` wraps SciPy. The thresholds are 0.10 (warn) and 0.25 (alert).

5. **`src/monitor.py`**: the production replay. The rules are constants at
   the top of the file. For each month it computes drift, MAE and the
   "never retrain" counterfactual. On a performance alert it trains a
   challenger, scores both models on the month's last 14 days and, if the
   challenger wins, refits it, registers a new version, moves the alias
   and exports it to `models/`.

6. **`app/main.py`**: FastAPI.
   - `Record` sets the input bounds with pydantic, so a bad hour or
     humidity gets a 422.
   - `/predict` keeps the last 2,000 records in a deque for `/drift`.
   - `/metrics` returns Prometheus text.

7. **`Dockerfile`, `ci/`**: a slim, non-root image with a health check.
   The GitHub Actions workflow runs the tests, builds the image and
   smoke-tests it. `dockerless_smoke.sh` repeats the image steps without
   Docker.

8. **`src/load_test.py`**: starts uvicorn and times 300 single-record and
   300 hundred-record requests, plus 400 from 8 threads, with httpx.

9. **`tests/`**: drift statistics on known distributions, lag
   correctness with a missing hour, the baseline fallback, and the API
   contract.
