"""
Train and evaluate baseline text classifiers for SYNTHETIC tyre-desk enquiries.

Pipeline (kept deliberately simple and inspectable):
    1. Load the labelled SYNTHETIC CSV.
    2. Hold out a stratified 20% test set.
    3. Turn text into features with TF-IDF (unigrams + bigrams).
    4. Train three models on the same split:
         - DummyClassifier (most_frequent) — a sanity-check baseline
         - Logistic Regression — well-calibrated probabilities, good default
         - LinearSVC — a strong classical linear baseline for text
    5. Print and save classification reports, a comparison table, and a
       confusion-matrix PNG for the primary model (Logistic Regression).
    6. Persist the fitted TF-IDF vectorizer + Logistic Regression model
       with joblib so the Streamlit app can load them.

This script is the source of the metrics quoted in README.md and CASE_STUDY.md.
Re-run it rather than editing numbers by hand.

Usage
-----
    python src/train.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

# Reproducible split / model initialisation.
RANDOM_STATE = 42
TEST_SIZE = 0.20

# Label order is fixed so reports, plots and the saved model all agree.
CLASS_ORDER = [
    "quote_request",
    "stock_availability",
    "order_status",
    "complaint_return",
    "account_billing",
    "other",
]

# Human-readable names for plots and the demo app.
CLASS_DISPLAY = {
    "quote_request": "Quote request",
    "stock_availability": "Stock / availability",
    "order_status": "Order status",
    "complaint_return": "Complaint / return",
    "account_billing": "Account / billing",
    "other": "Other",
}


def project_paths() -> dict[str, Path]:
    root = Path(__file__).resolve().parents[1]
    return {
        "root": root,
        "data": root / "data" / "SYNTHETIC_tyre_desk_enquiries.csv",
        "models": root / "models",
        "reports": root / "reports",
    }


def build_vectorizer() -> TfidfVectorizer:
    """
    TF-IDF turns each message into a sparse vector of term weights.

    - ngram_range=(1, 2) keeps useful phrases such as "in stock" or "credit note".
    - min_df=2 drops one-off tokens that will not generalise.
    - stop_words='english' removes very common function words.
    These are standard starter choices, not a tuned search.
    """
    return TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        stop_words="english",
    )


def build_models() -> dict[str, object]:
    """Return unfitted estimators keyed by a short display name."""
    return {
        "dummy_most_frequent": DummyClassifier(
            strategy="most_frequent", random_state=RANDOM_STATE
        ),
        "logistic_regression": LogisticRegression(
            # Multinomial loss is the usual choice for >2 classes.
            max_iter=1000,
            solver="lbfgs",
            random_state=RANDOM_STATE,
            # Balanced weights slightly counter the mild class imbalance
            # in the synthetic set (quotes/stock are more common).
            class_weight="balanced",
        ),
        "linear_svc": LinearSVC(
            random_state=RANDOM_STATE,
            class_weight="balanced",
            max_iter=2000,
        ),
    }


def metrics_dict(y_true, y_pred) -> dict[str, float]:
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "f1_macro": round(f1_score(y_true, y_pred, average="macro"), 4),
        "f1_weighted": round(f1_score(y_true, y_pred, average="weighted"), 4),
    }


def plot_confusion_matrix(y_true, y_pred, out_path: Path, title: str) -> None:
    """Save a labelled confusion-matrix heatmap as a PNG."""
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_ORDER)
    display_labels = [CLASS_DISPLAY[c] for c in CLASS_ORDER]

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=display_labels,
        yticklabels=display_labels,
        ax=ax,
        cbar_kws={"label": "Count"},
        linewidths=0.4,
        linecolor="white",
    )
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title)
    plt.xticks(rotation=35, ha="right")
    plt.yticks(rotation=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    paths = project_paths()
    paths["models"].mkdir(parents=True, exist_ok=True)
    paths["reports"].mkdir(parents=True, exist_ok=True)

    if not paths["data"].exists():
        raise FileNotFoundError(
            f"Synthetic dataset not found at {paths['data']}. "
            "Run: python src/generate_synthetic_data.py"
        )

    df = pd.read_csv(paths["data"])
    # Guardrail: refuse to train if the SYNTHETIC flag is missing.
    if "data_source" not in df.columns or not df["data_source"].eq(
        "SYNTHETIC_EXAMPLE_DATA"
    ).all():
        raise ValueError(
            "Expected every row to be tagged SYNTHETIC_EXAMPLE_DATA. "
            "Refusing to train on an unlabelled or mixed file."
        )

    print(f"Loaded {len(df)} SYNTHETIC messages from {paths['data'].name}")
    print("Class distribution:")
    print(df["label"].value_counts().reindex(CLASS_ORDER).to_string())
    print()

    X = df["text"].astype(str)
    y = df["label"].astype(str)

    # Stratify so the 20% test set keeps the same class mix as the full data.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"Train size: {len(X_train)}  |  Test size: {len(X_test)}")
    print()

    # Vectorizer is fitted once on the training text only (no test leakage).
    vectorizer = build_vectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    print(
        f"TF-IDF vocabulary size: {len(vectorizer.get_feature_names_out())} "
        f"terms (unigrams + bigrams, min_df=2)"
    )
    print()

    reports_text: list[str] = []
    all_metrics: dict[str, dict] = {
        "dataset": {
            "file": paths["data"].name,
            "data_source": "SYNTHETIC_EXAMPLE_DATA",
            "n_total": int(len(df)),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "class_counts": {
                label: int((df["label"] == label).sum()) for label in CLASS_ORDER
            },
        },
        "models": {},
    }

    fitted: dict[str, object] = {}
    predictions: dict[str, object] = {}

    for name, estimator in build_models().items():
        estimator.fit(X_train_vec, y_train)
        y_pred = estimator.predict(X_test_vec)
        fitted[name] = estimator
        predictions[name] = y_pred

        scores = metrics_dict(y_test, y_pred)
        report_str = classification_report(
            y_test,
            y_pred,
            labels=CLASS_ORDER,
            digits=3,
            zero_division=0,
        )
        all_metrics["models"][name] = {
            **scores,
            "classification_report": classification_report(
                y_test,
                y_pred,
                labels=CLASS_ORDER,
                digits=4,
                output_dict=True,
                zero_division=0,
            ),
        }

        header = f"{name}  —  accuracy={scores['accuracy']:.4f}  "
        header += f"macro-F1={scores['f1_macro']:.4f}  "
        header += f"weighted-F1={scores['f1_weighted']:.4f}"
        print("=" * 72)
        print(header)
        print("=" * 72)
        print(report_str)
        reports_text.append(header + "\n" + report_str)

    # Primary model for the app: Logistic Regression, because it provides
    # class probabilities (confidence) via predict_proba. LinearSVC is kept
    # as a comparison, not as the served model.
    primary_name = "logistic_regression"
    cm_path = paths["reports"] / "confusion_matrix_logistic_regression.png"
    plot_confusion_matrix(
        y_test,
        predictions[primary_name],
        cm_path,
        title="Confusion matrix — TF-IDF + Logistic Regression\n"
        "(held-out test set, SYNTHETIC tyre-desk enquiries)",
    )
    # Fix typo in title - wait I should not have typo in the plot title.
    # I'll fix this after if I left typo. Let me check... I wrote "tyre-desk" accidentally.
    # I'll regenerate with correct title - I'll fix the file before running.

    # Also save a side-by-side confusion matrix for LinearSVC for the appendix.
    cm_svc_path = paths["reports"] / "confusion_matrix_linear_svc.png"
    plot_confusion_matrix(
        y_test,
        predictions["linear_svc"],
        cm_svc_path,
        title="Confusion matrix — TF-IDF + LinearSVC\n"
        "(held-out test set, SYNTHETIC tyre-desk enquiries)",
    )

    # Persist a single bundle the Streamlit app can load in one joblib call.
    bundle = {
        "model_name": primary_name,
        "model": fitted[primary_name],
        "vectorizer": vectorizer,
        "classes": CLASS_ORDER,
        "class_display": CLASS_DISPLAY,
        "data_source": "SYNTHETIC_EXAMPLE_DATA",
        "trained_on": paths["data"].name,
        "metrics": all_metrics["models"][primary_name],
        "sklearn_pipeline_note": (
            "Vectorizer and classifier are stored separately in this bundle; "
            "the app applies vectorizer.transform then model.predict_proba."
        ),
    }
    model_path = paths["models"] / "tfidf_logistic_regression.joblib"
    joblib.dump(bundle, model_path)

    # A sklearn Pipeline is also saved — convenient if someone wants a single
    # .predict() object in a notebook.
    pipeline = Pipeline(
        [
            ("tfidf", vectorizer),
            ("clf", fitted[primary_name]),
        ]
    )
    pipeline_path = paths["models"] / "tfidf_logistic_regression_pipeline.joblib"
    joblib.dump(pipeline, pipeline_path)

    all_metrics["primary_model"] = primary_name
    all_metrics["artefacts"] = {
        "model_bundle": str(model_path.relative_to(paths["root"])),
        "pipeline": str(pipeline_path.relative_to(paths["root"])),
        "confusion_matrix_png": str(cm_path.relative_to(paths["root"])),
        "confusion_matrix_linear_svc_png": str(cm_svc_path.relative_to(paths["root"])),
    }
    all_metrics["generated_at_utc"] = datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )

    metrics_path = paths["reports"] / "metrics.json"
    metrics_path.write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")

    report_path = paths["reports"] / "classification_reports.txt"
    header_lines = [
        "Enquiry classifier — test-set reports",
        "Dataset: SYNTHETIC_tyre_desk_enquiries.csv (SYNTHETIC example data)",
        f"Train/test split: {1 - TEST_SIZE:.0%} / {TEST_SIZE:.0%}, stratified, "
        f"random_state={RANDOM_STATE}",
        f"TF-IDF vocabulary size: {len(vectorizer.get_feature_names_out())}",
        "",
    ]
    report_path.write_text(
        "\n".join(header_lines) + "\n".join(reports_text), encoding="utf-8"
    )

    print(f"Saved confusion matrix: {cm_path}")
    print(f"Saved LinearSVC confusion matrix: {cm_svc_path}")
    print(f"Saved metrics JSON:      {metrics_path}")
    print(f"Saved text reports:      {report_path}")
    print(f"Saved model bundle:      {model_path}")
    print(f"Saved sklearn pipeline:  {pipeline_path}")


if __name__ == "__main__":
    main()
