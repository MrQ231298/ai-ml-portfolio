# Code walkthrough — tyre demand forecast

This note is for someone who can run a command and open a file, but is
still new to forecasting. It explains the files, the run order, and
the few lines that matter.

The two metric tables and the critical evaluation live in
[README.md](README.md). The short client write-up is in
[CASE_STUDY.md](CASE_STUDY.md).

**Data reminder:** the weekly series is invented and tagged
`SYNTHETIC_EXAMPLE_DATA`.

---

## The idea in one minute

We want an eight-week-ahead forecast of weekly tyre units. Because
demand repeats every year, "the same week last year" is already a
serious guess. We compare that seasonal naive rule with a classical
ETS model and a lag-based gradient-boosting model, and we judge them
by rolling the forecast origin forward — not by shuffling weeks.

---

## Run order

```text
1. python src/generate_synthetic_series.py
        ↓ writes data/SYNTHETIC_weekly_tyre_sales.csv

2. python src/train.py
        ↓ back-tests all three models
        ↓ writes reports/forecast_plot.png and metrics.json
        ↓ writes models/tyre_demand_forecast.joblib

3. streamlit run app.py
```

`train.py` may take a minute: it refits ETS and the lag model at
every back-test origin.

---

## What each file is for

| File | Purpose |
| --- | --- |
| [`src/generate_synthetic_series.py`](src/generate_synthetic_series.py) | Builds the invented weekly series (trend, winter/spring/summer bumps, a 2020 dip, noise). |
| [`data/SYNTHETIC_weekly_tyre_sales.csv`](data/SYNTHETIC_weekly_tyre_sales.csv) | 365 Monday-start weeks, 2019–2025. |
| [`src/train.py`](src/train.py) | Back-test, 2025 hold-out, plots, saved bundle. |
| [`models/tyre_demand_forecast.joblib`](models/tyre_demand_forecast.joblib) | Models and history for the app. |
| [`reports/`](reports/) | Real metrics, the forecast plot, and origin-by-origin back-test scores. |
| [`app.py`](app.py) | Streamlit demo. |

---

## How to run it locally

From this folder (`projects/04-tyre-demand-forecast`):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python src/generate_synthetic_series.py
python src/train.py
streamlit run app.py
```

---

## Key code, with plain-English notes

### 1. Seasonal naive is one line of history

```python
# forecast for the next h weeks: the value 52 weeks earlier
ŷ_{t+h} = y_{t+h-52}
```

If last November was busy, this year's November starts from that
number. It will also repeat one-off shocks (the invented 2020 dip
would be copied into 2021).

### 2. ETS (Holt–Winters) is a classical seasonal smoother

```python
ExponentialSmoothing(
    y, trend="add", seasonal="add", seasonal_periods=52,
    initialization_method="estimated",
).fit(optimized=True).forecast(horizon)
```

It estimates a level, a trend and a 52-week seasonal pattern, then
projects them forward. It cannot see a lag of "what happened last
week" except through that estimated state.

### 3. The lag model predicts one week, then feeds itself

Features at time t include `y_{t-1}`, `y_{t-52}`, rolling means, and
the week of the year. For an 8-week forecast we cannot peek at future
actuals, so each prediction is written back as if it were an actual
before the next step is predicted. That is why it is called
*recursive*.

### 4. The back-test is the fair comparison

```python
for origin in range(MIN_TRAIN_WEEKS, n - HORIZON + 1, STEP):
    history = df.iloc[:origin]          # only data up to the origin
    actual = df.iloc[origin:origin+8]  # the next eight weeks
    # forecast with each model from this history
```

The reported back-test MAE is the *mean* of those origin-level MAEs.
A single 2025 hold-out is also computed, because it is easier to
plot, but it is not allowed to overrule the back-test on its own.

---

## If something goes wrong

- **`statsmodels` is not installed** — use the project's
  `requirements.txt`.
- **`ConvergenceWarning` from ETS** — the run still produces a
  forecast; the README discusses this as a limitation.
- **Saved model not found** — run `python src/train.py` from this
  folder.

---

## What to read next

1. Look at the 2025 plot in the app, then at the back-test table.
   They do not pick the same winner; the README explains why the
   back-test is the headline.
2. Read the **critical evaluation** in [README.md](README.md).
3. Read [CASE_STUDY.md](CASE_STUDY.md) for the buying-meeting version
   of the same argument.
