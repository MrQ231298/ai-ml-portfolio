# Code walkthrough — enquiry classifier

This note is for someone who is comfortable opening a file and running a
command, but is still new to machine learning. It explains *what each
file is for*, *the order to run things*, and *the few bits of code that
do the real work* — in plain English.

The technical results, limitations, and GDPR discussion live in
[README.md](README.md). The short client write-up is in
[CASE_STUDY.md](CASE_STUDY.md). You do not need those first, but they
are the next thing to read after you have seen the demo.

**Data reminder:** every training message is invented example text,
tagged `SYNTHETIC_EXAMPLE_DATA`. This project never trains on a real
customer inbox.

---

## The idea in one minute

A trade sales desk gets mixed messages: quotes, stock checks, "where's
my order?", complaints, invoice queries, and the odd question about
opening hours. This project teaches a simple model to *suggest a folder*
for a new message.

The model does not "understand" English the way a person does. It
learns which *words and short phrases* tend to appear in each folder
(for example "please quote" vs "in stock" vs "credit note"), then
applies that to a new message. The Streamlit app is just a small
website on your machine where you can paste text and see the suggestion
plus a confidence score.

---

## Run order

Always this sequence. Later steps need the files the earlier ones
create.

```text
1. python src/generate_synthetic_data.py
        ↓ writes data/SYNTHETIC_tyre_desk_enquiries.csv

2. python src/train.py
        ↓ writes models/*.joblib
        ↓ writes reports/metrics.json, classification reports, confusion-matrix PNGs

3. streamlit run app.py
        ↓ loads the saved model and opens a local web app
```

If you skip step 1, training has nothing to load. If you skip step 2,
the app will show an error asking you to train first. The repository
already contains the CSV, the saved model, and the reports from the
run on 14 November 2026, so you can start at step 3 if you only want
to try the demo. Re-run 1 and 2 when you change the generator or the
training script.

---

## What each file is for

| File | Purpose |
| --- | --- |
| [`src/generate_synthetic_data.py`](src/generate_synthetic_data.py) | Invents ~540 labelled example messages and writes the CSV. Nothing here is real customer mail. |
| [`data/SYNTHETIC_tyre_desk_enquiries.csv`](data/SYNTHETIC_tyre_desk_enquiries.csv) | The dataset: one row per message, with `text`, `label`, and a `data_source` flag. |
| [`data/README.md`](data/README.md) | A short warning that the file is synthetic, so nobody mistakes it for a live export. |
| [`src/train.py`](src/train.py) | Splits the data, builds TF-IDF features, trains three models, prints metrics, saves plots and the model file. |
| [`models/tfidf_logistic_regression.joblib`](models/tfidf_logistic_regression.joblib) | The saved vectorizer + Logistic Regression model the app loads. |
| [`models/tfidf_logistic_regression_pipeline.joblib`](models/tfidf_logistic_regression_pipeline.joblib) | The same model wrapped as a scikit-learn `Pipeline` (handy in a notebook). |
| [`reports/`](reports/) | Numbers and pictures from the real training run — not estimates. |
| [`app.py`](app.py) | The Streamlit demo: paste a message, see the predicted class and confidence. |
| [`requirements.txt`](requirements.txt) | The Python packages (and versions) needed to rerun the project. |
| [`README.md`](README.md) | Problem, method, **actual** metrics, and a critical evaluation. |
| [`CASE_STUDY.md`](CASE_STUDY.md) | The same project, written for a sales manager or potential client. |
| [`CODE_WALKTHROUGH.md`](CODE_WALKTHROUGH.md) | This file. |

---

## How to run it locally

You need Python 3.11 or newer (the recorded run used 3.13.5) and a
terminal. These commands assume you are in this project folder
(`projects/01-enquiry-classifier`).

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python src/generate_synthetic_data.py
python src/train.py
streamlit run app.py
```

Streamlit will print a local address, usually `http://localhost:8501`.
Open it in a browser. Pick one of the built-in synthetic examples, or
paste your own message, and click **Predict class**. You should see a
folder name, a confidence percentage, and a bar chart of all six
classes.

To stop the app, go back to the terminal and press `Ctrl+C`.

The generator and the trainer are seeded (`RANDOM_SEED=42` and
`random_state=42`), so a clean re-run should reproduce the same split
and the same metrics as [README.md](README.md).

---

## Key code, with plain-English notes

You do not need to memorise this. The point is to be able to point at
the line that does each job.

### 1. The data is tagged as synthetic, on every row

In `generate_synthetic_data.py`, each invented message is stored with
a fixed flag. The filename already says `SYNTHETIC_`; this column is a
second, machine-readable reminder.

```python
rows.append(
    {
        "message_id": f"SYN-{len(rows) + 1:04d}",
        "text": candidate,
        "label": label,
        "data_source": "SYNTHETIC_EXAMPLE_DATA",
    }
)
```

`train.py` refuses to start if that flag is missing or mixed. That is
a small safety rail so a live mailbox export cannot be trained on by
accident.

```python
if "data_source" not in df.columns or not df["data_source"].eq(
    "SYNTHETIC_EXAMPLE_DATA"
).all():
    raise ValueError(
        "Expected every row to be tagged SYNTHETIC_EXAMPLE_DATA. "
        "Refusing to train on an unlabelled or mixed file."
    )
```

### 2. A stratified train/test split (no peeking)

We hold back 20% of the messages to test the model. *Stratified* means
the test set keeps the same mix of classes as the full file (so we do
not end up with, say, almost no complaints in the test set by bad luck).
`random_state=42` makes the split the same every time you run the
script.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)
```

### 3. Text becomes numbers with TF-IDF

Computers cannot train a logistic regression model on raw sentences.
TF-IDF turns each message into a list of weights: words and two-word
phrases that are typical of a message score higher; very common words
score lower. The important rule is that the vocabulary is **learned
from the training text only**. The test messages are then *transformed
with that already-learned vocabulary*. If we fitted on train and test
together, the test score would be slightly cheated.

```python
vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),   # single words and two-word phrases, e.g. "credit note"
    min_df=2,             # ignore terms that appear only once
    max_df=0.95,
    stop_words="english",
)
X_train_vec = vectorizer.fit_transform(X_train)  # learn + transform
X_test_vec = vectorizer.transform(X_test)        # transform only
```

### 4. Three models, same features, same split

A **dummy** model always predicts the most common class (`quote_request`).
It is a sanity check: if Logistic Regression cannot beat this, it is
not learning anything useful.

**Logistic Regression** is the model the app uses, because it can
output a *probability* for each class (the confidence you see in the
demo).

**LinearSVC** is a second, well-known linear classifier for text. On
this run it scored slightly higher, but it is not served in the app
because the default estimator does not provide probabilities.

```python
DummyClassifier(strategy="most_frequent", random_state=42)

LogisticRegression(
    max_iter=1000,
    class_weight="balanced",  # slightly counters the uneven class counts
    random_state=42,
)

LinearSVC(
    class_weight="balanced",
    random_state=42,
    max_iter=2000,
)
```

Each model is fitted on `X_train_vec` and scored on `X_test_vec`. The
script then writes a classification report, confusion-matrix images,
and `reports/metrics.json`. Those files are the source of the numbers
in the README — we do not type the scores in by hand.

### 5. The app only does "transform, then predict"

`app.py` does not retrain anything. It loads the joblib bundle, applies
the *already fitted* vectorizer, and asks the classifier for class
probabilities.

```python
X = bundle["vectorizer"].transform([text])
proba = bundle["model"].predict_proba(X)[0]
```

The class with the largest probability is the suggested folder.
"Confidence" is that probability, shown as a percentage. High
confidence is not a guarantee of being right — the README explains
why, especially on tidy synthetic text.

---

## If something goes wrong

- **"Saved model not found"** in the app — run `python src/train.py`
  from this folder, then reload the browser tab.
- **Dataset not found** when training — run
  `python src/generate_synthetic_data.py` first.
- **`streamlit` or `sklearn` is not recognised** — activate the
  virtual environment, then `pip install -r requirements.txt`.
- Metrics look different after you edit the generator — that is
  expected. Retrain and update the README from `reports/metrics.json`
  rather than leaving the old numbers in place.

---

## What to read next

1. Try two or three examples in the app, including one that is vague
   ("any update?") and see how the confidence changes.
2. Read [README.md](README.md) — especially the results table and the
   **critical evaluation**. The high accuracy on this file is real,
   and also optimistic, for reasons that section spells out.
3. Read [CASE_STUDY.md](CASE_STUDY.md) if you want the same project in
   business language.
