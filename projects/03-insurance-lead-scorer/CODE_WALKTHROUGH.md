# Code walkthrough — insurance lead scorer

This note is for someone who can run a command and open a file, but is
still new to machine learning. It explains the files, the run order,
and the few lines that matter.

The results, calibration discussion and ethics note live in
[README.md](README.md). The short client write-up is in
[CASE_STUDY.md](CASE_STUDY.md).

**Data reminder:** every lead is invented example data, tagged
`SYNTHETIC_EXAMPLE_DATA`. This is not a real prospects file.

---

## The idea in one minute

We want to rank commercial insurance leads by how likely they are to
convert. Because only about one in four to five leads in this file
converts, "accuracy" would be a poor headline: a model that always
says "no" would look good and be useless. We therefore look at
ranking metrics (ROC-AUC, PR-AUC) and at precision@k: *of the k
leads we would call first, how many actually convert?*

---

## Run order

```text
1. python src/generate_synthetic_leads.py
        ↓ writes data/SYNTHETIC_commercial_insurance_leads.csv

2. python src/train.py
        ↓ writes models/lead_scorer.joblib
        ↓ writes reports/metrics.json and the PNG plots

3. streamlit run app.py
        ↓ loads the saved models and opens a local web app
```

---

## What each file is for

| File | Purpose |
| --- | --- |
| [`src/generate_synthetic_leads.py`](src/generate_synthetic_leads.py) | Invents 3,200 labelled leads, with a documented noisy process for the label. |
| [`data/SYNTHETIC_commercial_insurance_leads.csv`](data/SYNTHETIC_commercial_insurance_leads.csv) | The modelling table. |
| [`data/README.md`](data/README.md) | Warning that the file is synthetic. |
| [`src/train.py`](src/train.py) | Split, models, metrics, SHAP / permutation plots, slice analysis. |
| [`models/lead_scorer.joblib`](models/lead_scorer.joblib) | Saved models for the app. |
| [`reports/`](reports/) | Real metrics and pictures from the training run. |
| [`app.py`](app.py) | Streamlit demo. |
| [`README.md`](README.md) | Problem, method, **actual** metrics, ethics. |
| [`CASE_STUDY.md`](CASE_STUDY.md) | The same project in client language. |

---

## How to run it locally

From this folder (`projects/03-insurance-lead-scorer`):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python src/generate_synthetic_leads.py
python src/train.py
streamlit run app.py
```

If `shap` fails to install, remove it from `requirements.txt` and
re-run the trainer: it will fall back to permutation importance.

---

## Key code, with plain-English notes

### 1. A known process created the labels

The generator does not "observe" conversions. It builds a log-odds
score from source, follow-up speed, size and cover type, adds noise,
and tosses a biased coin. That process is written down in
`conversion_logit()` so we cannot pretend the labels came from a
real market.

### 2. A stratified split keeps the 23% conversion rate in both sides

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
```

### 3. Class weights, not a rewritten data set

```python
LogisticRegression(class_weight="balanced", ...)
HistGradientBoostingClassifier(class_weight="balanced", ...)
```

This tells both models to care more about each converting lead. It
usually improves ranking and usually *inflates* the predicted
percentages. That is why `train.py` also draws a calibration plot.

### 4. Precision@k is "the next k calls"

```python
idx = np.argsort(-proba)[:k]
return float(np.mean(y_true[idx]))
```

If a desk can work 50 leads properly, precision@50 is the metric that
matches the job. ROC-AUC still matters, but it averages over
thresholds the desk will never use.

### 5. SHAP, with a safety net

`train.py` tries to import `shap` and explain the boosting model. If
that fails, it records the failure and keeps the permutation-importance
plot so the report still has an explainability artefact.

### 6. The app must not sound like an underwriter

The warning at the top of `app.py` is part of the design. A conversion
score is a sales aid. It is not a decision about whether a business
should be offered cover.

---

## If something goes wrong

- **"Saved model not found"** — run `python src/train.py` from this
  folder, then reload the app.
- **`shap` import errors** — it is optional. The trainer will say so
  and write `reports/permutation_importance.png` instead.
- Metrics change after you edit the generator — retraining is
  expected. Update the README from `reports/metrics.json`.

---

## What to read next

1. Move the "hours until first follow-up" slider in the app and watch
   the score. Then read the **calibration** section in the README:
   the number you see is a ranking aid, not a precise probability.
2. Read the **fairness and ethics** section in [README.md](README.md).
3. Read [CASE_STUDY.md](CASE_STUDY.md) for the same project in
   business language.
