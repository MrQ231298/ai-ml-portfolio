# Code walkthrough — churn prediction

A plain-English tour of the files. For results, see [README.md](README.md).

## The pipeline at a glance

```
1. python src/fetch_data.py   (optional)
        ↓ writes data/Telco-Customer-Churn.csv and data/source_metadata.json

2. python src/train.py
        ↓ cross-validation, calibration, threshold, SHAP, fairness slices
        ↓ writes models/churn_model.joblib and reports/*.png, reports/metrics.json

3. pytest -q          → 4 smoke tests
4. streamlit run app.py
```

## What each file is for

| File | Purpose |
| --- | --- |
| [`src/fetch_data.py`](src/fetch_data.py) | Downloads the IBM CSV and records URL, licence, time and SHA-256. |
| [`src/churn_lib.py`](src/churn_lib.py) | One place for loading/cleaning, the feature lists, and the `net_benefit` formula, so training, app and tests agree. |
| [`src/train.py`](src/train.py) | The experiment: CV comparison, HGB grid, calibration, cost threshold, SHAP, permutation importance, fairness slices, plots. |
| [`app.py`](app.py) | Streamlit demo: profile → calibrated probability, decision, SHAP reasons. |
| [`tests/test_smoke.py`](tests/test_smoke.py) | Checks cleaning, the cost formula, the saved model and that it beats the dummy. |
| [`reports/`](reports/) | Real outputs of the last run. |

## Key ideas in the code

**Cleaning with an assertion.** `load_clean()` converts `TotalCharges`
to numbers and fills the 11 blanks with 0 — but only after asserting
that every blank belongs to a customer with `tenure == 0`.

**Protected attributes are kept out of `FEATURE_COLS`.** `AUDIT_COLS`
(`gender`, `SeniorCitizen`) travel alongside the data only so the
fairness table can be computed at the end.

**Selection never looks at the test set.** The HGB grid is scored with
`cross_validate` on the training rows; the calibration method is picked
by out-of-fold Brier (`cross_val_predict`); the threshold is picked on
out-of-fold probabilities. The test set is only used to report.

**The threshold search** is a simple loop over 0.02–0.95: for each
value, `net_benefit()` adds `save_rate × value_months × monthly charge`
for every contacted churner and subtracts `offer_cost` for every
contact.

**SHAP on the tree model.** The pipeline one-hot encodes first, so
`TreeExplainer` explains the booster on the encoded columns. In the app,
only the one-hot columns that are "on" for that customer are shown.
