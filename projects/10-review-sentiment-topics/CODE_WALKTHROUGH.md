# Code walkthrough — review sentiment & topics

## The pipeline at a glance

```
1. python src/fetch_data.py       → data/womens_clothing_reviews.csv (+ metadata)
2. python src/train_sentiment.py  → models/sentiment_tfidf.joblib, reports/sentiment_metrics.json,
                                    confusion_matrix.png, misclassified_examples.csv
3. python src/topics.py           → models/topics_nmf.joblib, reports/topic_metrics.json,
                                    topic_summary.csv, insights_report.md, *.png
4. pytest -q
5. streamlit run app.py
```

## What each file is for

| File | Purpose |
| --- | --- |
| [`src/fetch_data.py`](src/fetch_data.py) | Downloads the CSV from a public mirror and records its SHA-256. |
| [`src/review_lib.py`](src/review_lib.py) | Loading, title + body text, star → label rule. |
| [`src/train_sentiment.py`](src/train_sentiment.py) | Dummy, TF-IDF + LR (CV-tuned) and MiniLM + LR; metrics, confusion matrix, error examples, top words. |
| [`src/topics.py`](src/topics.py) | NMF sweep, NPMI coherence, seed stability, topic table, plots and the Markdown insights report. |
| [`app.py`](app.py) | Paste a review → sentiment probabilities + closest topics; topic table. |
| [`tests/test_smoke.py`](tests/test_smoke.py) | Label rule, loading, NPMI sanity check, model predictions, metrics. |

## Key ideas in the code

**Word + character features.** `FeatureUnion` joins word 1–2-grams
("too small", "going back") with character 2–5-grams, which cope with
typos ("compaint", "hte") common in real reviews.

**Balanced classes.** Only ~10% of reviews are negative, so
`class_weight="balanced"` stops the model ignoring them; macro-F1 is
the headline because accuracy rewards always saying "positive" (the
dummy gets 77% accuracy).

**NPMI from scratch.** `npmi_coherence` counts how often each pair of
top words appears in the same review, compared with chance, normalised
to −1…1. Higher means the topic's words really go together.

**Stability check.** Re-fitting NMF with random initialisation and other
seeds, then matching topics by Jaccard overlap, shows how much the
topics depend on luck.

**Report generation.** The final block of `topics.py` writes
`insights_report.md` using f-strings over the computed tables, so the
report always matches the data.
