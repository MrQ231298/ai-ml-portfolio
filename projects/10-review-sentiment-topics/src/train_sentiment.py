"""
Three-class review sentiment (negative / neutral / positive) on 22,641 real
clothing reviews, labels derived from the star rating.

Models (same stratified 80/20 split, seed 42; test set used once):
  - dummy (most frequent class)
  - TF-IDF (word 1-2 grams + char 2-5 grams) + logistic regression,
    class_weight="balanced"; C chosen by 5-fold CV macro-F1 on train
  - MiniLM sentence embeddings + logistic regression (balanced)

Also reports a binary view (negative vs positive, neutral excluded) and
saves misclassified examples for error analysis.

    python src/train_sentiment.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             confusion_matrix, f1_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
from review_lib import EMBED_MODEL, LABELS, ROOT, load_reviews  # noqa: E402

REPORTS, MODELS = ROOT / "reports", ROOT / "models"


def tfidf_pipeline() -> Pipeline:
    feats = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True, max_features=200000)),
    ])
    return Pipeline([("tfidf", feats), ("clf", LogisticRegression(max_iter=3000, class_weight="balanced"))])


def summarise(y, pred) -> dict:
    return {
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "macro_f1": round(float(f1_score(y, pred, average="macro")), 4),
        "per_class_f1": {k: round(float(v), 4) for k, v in zip(LABELS, f1_score(y, pred, average=None, labels=LABELS))},
    }


def main() -> None:
    REPORTS.mkdir(exist_ok=True); MODELS.mkdir(exist_ok=True)
    df = load_reviews()
    tr, te = train_test_split(df, test_size=0.2, stratify=df.label, random_state=42)
    print(f"reviews={len(df)} train={len(tr)} test={len(te)}", df.label.value_counts(normalize=True).round(4).to_dict())
    results, preds = {}, {}

    dummy = DummyClassifier(strategy="most_frequent").fit(tr.text, tr.label)
    preds["dummy"] = dummy.predict(te.text)

    t = time.time()
    gs = GridSearchCV(tfidf_pipeline(), {"clf__C": [0.5, 2.0, 8.0]}, scoring="f1_macro",
                      cv=StratifiedKFold(5, shuffle=True, random_state=42), n_jobs=3)
    gs.fit(tr.text, tr.label)
    tfidf_model = gs.best_estimator_
    preds["tfidf_logreg"] = tfidf_model.predict(te.text)
    proba_tfidf = tfidf_model.predict_proba(te.text)
    print("tfidf done", round(time.time() - t, 1), "s; best C", gs.best_params_, "cv macro-F1", round(gs.best_score_, 4))

    t = time.time()
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(EMBED_MODEL)
    E_tr = enc.encode(tr.text.tolist(), batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    E_te = enc.encode(te.text.tolist(), batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    emb_clf = LogisticRegression(max_iter=3000, class_weight="balanced", C=2.0).fit(E_tr, tr.label)
    preds["minilm_logreg"] = emb_clf.predict(E_te)
    print("embeddings done", round(time.time() - t, 1), "s")

    for k, p in preds.items():
        results[k] = summarise(te.label, p)
        print(k, results[k])

    # binary view on the best model: negative vs positive only
    best = max(["tfidf_logreg", "minilm_logreg"], key=lambda k: results[k]["macro_f1"])
    mask = te.label.isin(["negative", "positive"]).to_numpy()
    classes = list(tfidf_model.classes_)
    if best == "tfidf_logreg":
        p_pos = proba_tfidf[:, classes.index("positive")]
    else:
        pr = emb_clf.predict_proba(E_te); p_pos = pr[:, list(emb_clf.classes_).index("positive")]
    binary_auc = roc_auc_score((te.label[mask] == "positive").astype(int), p_pos[mask])

    cm = confusion_matrix(te.label, preds[best], labels=LABELS)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay(cm, display_labels=LABELS).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{best}: test confusion matrix"); fig.tight_layout()
    fig.savefig(REPORTS / "confusion_matrix.png", dpi=130); plt.close(fig)

    # error analysis sample
    err = te.assign(pred=preds[best])
    err = err[err.label != err.pred]
    err.groupby(["label", "pred"]).head(5)[["rating", "label", "pred", "text"]].to_csv(REPORTS / "misclassified_examples.csv", index=False)
    pair_counts = err.groupby(["label", "pred"]).size().rename("n").reset_index().to_dict(orient="records")

    # top words for the TF-IDF model (word features only)
    word_vec = tfidf_model.named_steps["tfidf"].transformer_list[0][1]
    n_word = len(word_vec.vocabulary_)
    coefs = tfidf_model.named_steps["clf"].coef_[:, :n_word]
    vocab = np.array(word_vec.get_feature_names_out())
    top_words = {c: vocab[np.argsort(-coefs[i])[:15]].tolist() for i, c in enumerate(classes)}

    joblib.dump({"model": tfidf_model, "labels": LABELS}, MODELS / "sentiment_tfidf.joblib")
    metrics = {
        "run_at_europe_dublin": datetime.now(ZoneInfo("Europe/Dublin")).isoformat(timespec="seconds"),
        "data": {"reviews": int(len(df)), "train": int(len(tr)), "test": int(len(te)),
                 "label_share": df.label.value_counts(normalize=True).round(4).to_dict(),
                 "label_rule": "1-2 stars negative, 3 neutral, 4-5 positive"},
        "tfidf_cv": {"best_C": gs.best_params_["clf__C"], "cv_macro_f1": round(float(gs.best_score_), 4)},
        "test": results, "best_model": best,
        "binary_neg_vs_pos_roc_auc_best": round(float(binary_auc), 4),
        "confusion_matrix_best": {"labels": LABELS, "matrix": cm.tolist()},
        "error_pairs_best": pair_counts,
        "tfidf_top_words": top_words,
        "classification_report_best": classification_report(te.label, preds[best], digits=4),
    }
    (REPORTS / "sentiment_metrics.json").write_text(json.dumps(metrics, indent=2))
    print("best", best, "binary AUC", round(binary_auc, 4))


if __name__ == "__main__":
    main()
