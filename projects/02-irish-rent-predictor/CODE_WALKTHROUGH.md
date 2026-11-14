# Code walkthrough — Irish rent predictor

This note is for someone who can run a command and open a file, but is
still new to machine learning. It explains what each file is for, the
order to run things, and the few lines that do the real work.

The results, data licence and critical evaluation live in
[README.md](README.md). The short client write-up is in
[CASE_STUDY.md](CASE_STUDY.md).

**Data reminder:** the rent figures are real official statistics from
the RTB / CSO (table RIA02, CC BY 4.0). They are published *averages*,
not rents for named dwellings.

---

## The idea in one minute

We want to estimate the official average monthly rent for a place in
Ireland, given the year, the location, the property type and the
number of bedrooms. We compare a simple "use the last published
figure" rule with two machine-learning models, and we test them on
*later years* (2024–2025) rather than on a random shuffle of rows.

In this run, the simple rule won. The walkthrough is written so you can
see how that comparison was made, not just how to fit a model.

---

## Run order

```text
1. python src/fetch_cso_ria02.py
        ↓ downloads the official cube
        ↓ writes data/ria02_average_monthly_rent.csv and source_metadata.json

2. python src/train.py
        ↓ writes models/rent_predictor.joblib
        ↓ writes reports/metrics.json and the PNG plots

3. streamlit run app.py
        ↓ loads the saved models and opens a local web app
```

The tidy CSV is already in the repository, so you can start at step 2
if you cannot reach the CSO API. Please keep the attribution in
`data/README.md` if you reuse the file.

---

## What each file is for

| File | Purpose |
| --- | --- |
| [`src/fetch_cso_ria02.py`](src/fetch_cso_ria02.py) | Downloads RIA02, drops aggregate categories and county totals, writes the tidy CSV and the licence/source metadata. |
| [`data/ria02_average_monthly_rent.csv`](data/ria02_average_monthly_rent.csv) | The modelling table: one published average per row. |
| [`data/ria02_raw.json`](data/ria02_raw.json) | Cached copy of the official JSON-stat cube. |
| [`data/source_metadata.json`](data/source_metadata.json) | URL, fetch time, licence, SHA-256 of the raw file. |
| [`data/README.md`](data/README.md) | Attribution and a description of the filters. |
| [`src/baseline.py`](src/baseline.py) | The persistence baseline, in its own module so the app can unpickle it. |
| [`src/train.py`](src/train.py) | The temporal split, the three models, metrics, plots and the saved bundle. |
| [`models/rent_predictor.joblib`](models/rent_predictor.joblib) | Saved models plus location lists for the app. |
| [`reports/`](reports/) | Real metrics and pictures from the training run. |
| [`app.py`](app.py) | Streamlit demo. |
| [`README.md`](README.md) | Problem, method, **actual** metrics, critical evaluation. |
| [`CASE_STUDY.md`](CASE_STUDY.md) | The same project in client language. |

---

## How to run it locally

From this folder (`projects/02-irish-rent-predictor`):

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python src/fetch_cso_ria02.py      # optional if data/ already exists
python src/train.py
streamlit run app.py
```

Open the local address Streamlit prints (usually
`http://localhost:8501`). Pick a county and location, a property type
and bedroom count, and a year. The headline euro figure is whichever
model had the lowest error on 2024–2025 (in this run, the persistence
baseline).

---

## Key code, with plain-English notes

### 1. A temporal split, not a random one

We train on earlier years and test on later ones. If we shuffled rows,
the model could "see" 2025 rents for some areas while being tested on
2024 for others, which is easier than the real problem.

```python
train = df[df["year"] <= 2023]
test = df[df["year"].isin([2024, 2025])]
```

### 2. The persistence baseline

For each test row, use the most recent training-year rent for the same
location, property type and bedroom count. If that cell was never
published in training, fall back to a county mean, then to the global
mean.

```python
# inside PersistenceBaseline.predict
key = (r.location, r.property_type, int(r.n_bedrooms))
if key in self.last_lookup_:
    out.append(self.last_lookup_[key])
```

This is a "naive" forecast, and it was the most accurate of the three
models on the hold-out. That is why it is in a small module of its own
and why the app treats it as a first-class model, not a footnote.

### 3. Log-linear regression

Rents are skewed (a few very high averages). The linear model learns
on `log(rent)` and we convert predictions back to Euro before scoring,
so all models are judged in the same units.

```python
TransformedTargetRegressor(
    regressor=inner, func=np.log, inverse_func=np.exp, check_inverse=False
)
```

### 4. Gradient boosting, and a cardinality limit

The tree model is sklearn's `HistGradientBoostingRegressor`. A large
random forest reached the same kind of accuracy but produced a 200 MB
file, which is too big for this repository. Histogram boosting stays
under 1 MB.

sklearn will not accept a categorical feature with more than 255
levels. Location has 378, so only county, location level and property
type are marked categorical. Location is passed as an integer code.
That is a compromise, and it is one reason county outranks location in
the importance plot.

### 5. Metrics are always in Euro

```python
mae = mean_absolute_error(y_true, y_pred)
rmse = sqrt(mean_squared_error(y_true, y_pred))
bias = mean(y_pred - y_true)   # negative means the model is too low
```

Mean bias is reported because, in this run, almost all of the
persistence error was a systematic under-prediction (rents kept rising)
rather than random scatter.

### 6. The app loads the bundle and scores all three models

```python
for name, model in bundle["models"].items():
    pred = model.predict(row)[0]
```

The headline model is `bundle["model_name"]`, which `train.py` sets to
whichever model had the lowest test MAE. The script does not hard-code
"always use the fanciest model."

---

## If something goes wrong

- **CSO API error** when fetching — use the CSV already in `data/` and
  skip the fetch step.
- **"Saved model not found"** in the app — run `python src/train.py`
  from this folder, then reload.
- **`Can't get attribute PersistenceBaseline`** — the app must be
  started from this folder so it can import `src/baseline.py`. Use
  `streamlit run app.py` as shown above.
- Metrics change after a new CSO release — that is expected. Retrain
  and update the README from `reports/metrics.json`.

---

## What to read next

1. Try the same location in 2023 and in 2025 in the app, and notice
   that the persistence headline will not "know about" increases after
   the last training year.
2. Read the **results** and **critical evaluation** in
   [README.md](README.md). The most important finding is that the
   baseline won, and why.
3. Read [CASE_STUDY.md](CASE_STUDY.md) for the same project in
   business language, including the CC BY credit to the RTB and CSO.
