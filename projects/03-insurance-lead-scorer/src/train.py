"""
Train and evaluate lead-conversion classifiers on SYNTHETIC insurance leads.

The positive class is rare (about one in eight leads in this file), so
accuracy would be a misleading headline. We report:

- ROC-AUC (ranking quality across thresholds)
- PR-AUC (more informative under imbalance)
- precision@k (what a sales team would see if they called the top k)
- a calibration plot (does 30% mean 30%?)

Class imbalance is handled with `class_weight="balanced"` on both
logistic regression and histogram gradient boosting. A dummy prior
classifier is included so the PR curve has a no-skill baseline.

Usage
-----
    python src/train.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

RANDOM_STATE = 42
TEST_SIZE = 0.20
# Precision at these absolute ranks, plus the top 10% of the test set.
K_VALUES = (25, 50, 100)

CAT_COLS = ["industry", "revenue_band", "coverage_type", "source", "region"]
NUM_COLS = [
    "n_employees",
    "years_trading",
    "n_locations",
    "response_time_hours",
    "has_claims_last_3y",
]
FEATURE_COLS = CAT_COLS + NUM_COLS
TARGET = "converted"


def project_paths() -> dict[str, Path]:
    root = Path(__file__).resolve().parents[1]
    return {
        "root": root,
        "data": root / "data" / "SYNTHETIC_commercial_insurance_leads.csv",
        "models": root / "models",
        "reports": root / "reports",
    }


def precision_at_k(y_true: np.ndarray, proba: np.ndarray, k: int) -> float:
    """Fraction of the top-k highest-scoring leads that actually converted."""
    k = min(int(k), len(y_true))
    if k == 0:
        return float("nan")
    idx = np.argsort(-proba)[:k]
    return float(np.mean(y_true[idx]))


def metric_block(y_true, proba, ks) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    proba = np.asarray(proba, dtype=float)
    out = {
        "roc_auc": round(float(roc_auc_score(y_true, proba)), 4),
        "pr_auc": round(float(average_precision_score(y_true, proba)), 4),
        "prevalence": round(float(y_true.mean()), 4),
        "n": int(len(y_true)),
        "n_positives": int(y_true.sum()),
        "precision_at_k": {},
    }
    for k in ks:
        out["precision_at_k"][str(k)] = round(precision_at_k(y_true, proba, k), 4)
    top10 = max(1, int(round(0.10 * len(y_true))))
    out["precision_at_top_10pct"] = round(precision_at_k(y_true, proba, top10), 4)
    out["k_top_10pct"] = top10
    return out


def build_dummy() -> DummyClassifier:
    # Predicts with the training prevalence — the no-skill PR baseline.
    return DummyClassifier(strategy="prior", random_state=RANDOM_STATE)


def build_logistic() -> Pipeline:
    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
            ("num", StandardScaler(), NUM_COLS),
        ]
    )
    return Pipeline(
        [
            ("pre", pre),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    solver="lbfgs",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def build_boosting() -> Pipeline:
    pre = ColumnTransformer(
        [
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                CAT_COLS,
            ),
            ("num", "passthrough", NUM_COLS),
        ]
    )
    return Pipeline(
        [
            ("pre", pre),
            (
                "model",
                HistGradientBoostingClassifier(
                    max_depth=6,
                    max_iter=200,
                    learning_rate=0.08,
                    min_samples_leaf=20,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    categorical_features=[0, 1, 2, 3, 4],
                ),
            ),
        ]
    )


def plot_roc(y_true, probas: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    for name, proba in probas.items():
        if name == "dummy_prior":
            continue  # a constant score traces the diagonal; already drawn as no-skill
        fpr, tpr, _ = roc_curve(y_true, proba)
        auc = roc_auc_score(y_true, proba)
        ax.plot(fpr, tpr, label=f"{name.replace('_', ' ')}  (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=0.9, label="no-skill (AUC 0.50)")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve — SYNTHETIC insurance leads (test set)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_pr(y_true, probas: dict, out: Path) -> None:
    prevalence = float(np.mean(y_true))
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    for name, proba in probas.items():
        if name == "dummy_prior":
            continue  # a constant score is the prevalence line, not a triangle to precision=1
        prec, rec, _ = precision_recall_curve(y_true, proba)
        ap = average_precision_score(y_true, proba)
        ax.plot(rec, prec, label=f"{name.replace('_', ' ')}  (AP {ap:.3f})")
    ax.axhline(prevalence, ls="--", color="grey", lw=0.9, label=f"no-skill / prevalence ({prevalence:.1%})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_ylim(0, 1.05)
    ax.set_title("Precision–recall curve — SYNTHETIC insurance leads (test set)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_calibration(y_true, probas: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=0.9, label="perfectly calibrated")
    for name, proba in probas.items():
        if name == "dummy_prior":
            continue
        frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=8, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", label=name.replace("_", " "))
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed conversion rate")
    ax.set_title("Calibration — SYNTHETIC insurance leads (test set)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_precision_at_k(y_true, probas: dict, ks, out: Path) -> None:
    rows = []
    prevalence = float(np.mean(y_true))
    for name, proba in probas.items():
        for k in ks:
            rows.append(
                {
                    "model": name.replace("_", " "),
                    "k": str(k),
                    "precision": precision_at_k(y_true, proba, k),
                }
            )
    plot_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    sns.barplot(data=plot_df, x="k", y="precision", hue="model", ax=ax)
    ax.axhline(prevalence, ls="--", color="grey", lw=0.9, label="prevalence")
    ax.set_xlabel("k (highest-scoring leads in the test set)")
    ax.set_ylabel("Precision@k")
    ax.set_ylim(0, 1)
    ax.set_title("Precision@k — would these be good leads to call first?")
    ax.legend(title="")
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_importance(imp_df: pd.DataFrame, title: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ordered = imp_df.sort_values("importance_mean")
    ax.barh(ordered["feature"], ordered["importance_mean"], xerr=ordered["importance_std"], color="#2a6f97")
    ax.set_xlabel("Importance")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def try_shap(pipeline: Pipeline, X_sample: pd.DataFrame, out: Path) -> dict | None:
    """
    Attempt a SHAP summary for the gradient-boosting model. Returns a
    small metadata dict on success, or None if SHAP is missing / fails.
    """
    try:
        import shap  # type: ignore
    except Exception as exc:
        print(f"SHAP not available ({exc}). Falling back to permutation importance.")
        return None
    try:
        pre = pipeline.named_steps["pre"]
        model = pipeline.named_steps["model"]
        X_trans = pre.transform(X_sample)
        # Recover transformed feature names for the plot.
        feature_names = [n.split('__', 1)[-1] for n in pre.get_feature_names_out()]
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_trans)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        mean_abs = np.mean(np.abs(shap_values), axis=0)
        # Save a compact bar plot (avoid depending on shap's matplotlib theme).
        imp = (
            pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
            .sort_values("mean_abs_shap", ascending=False)
            .head(12)
        )
        fig, ax = plt.subplots(figsize=(7.4, 5.0))
        ax.barh(imp["feature"][::-1], imp["mean_abs_shap"][::-1], color="#2a6f97")
        ax.set_xlabel("Mean |SHAP value| (test sample)")
        ax.set_title("SHAP feature impact — gradient boosting")
        fig.tight_layout()
        fig.savefig(out, dpi=160, bbox_inches="tight")
        plt.close(fig)
        return {
            "library_version": getattr(shap, "__version__", "unknown"),
            "n_rows_explained": int(len(X_sample)),
            "top_features": imp.to_dict(orient="records"),
            "plot": out.name,
        }
    except Exception as exc:
        print(f"SHAP ran but failed ({type(exc).__name__}: {exc}). Using permutation importance.")
        return None


def slice_metrics(frame: pd.DataFrame, y_true, proba, col: str) -> dict:
    """ROC-AUC and conversion rate per group, where the group has both classes."""
    out = {}
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    for value, idx in frame.groupby(col).groups.items():
        pos = frame.index.get_indexer(idx)
        yt, pr = y_true[pos], proba[pos]
        rec = {
            "n": int(len(yt)),
            "n_positives": int(yt.sum()),
            "conversion_rate": round(float(yt.mean()), 4),
            "mean_predicted": round(float(pr.mean()), 4),
        }
        if yt.sum() > 0 and yt.sum() < len(yt) and len(yt) >= 20:
            rec["roc_auc"] = round(float(roc_auc_score(yt, pr)), 4)
        else:
            rec["roc_auc"] = None
        out[str(value)] = rec
    return out


def main() -> None:
    paths = project_paths()
    paths["models"].mkdir(parents=True, exist_ok=True)
    paths["reports"].mkdir(parents=True, exist_ok=True)

    if not paths["data"].exists():
        raise FileNotFoundError(
            f"{paths['data']} not found. Run: python src/generate_synthetic_leads.py"
        )

    df = pd.read_csv(paths["data"])
    if "data_source" not in df.columns or not df["data_source"].eq("SYNTHETIC_EXAMPLE_DATA").all():
        raise ValueError("Expected every row to be tagged SYNTHETIC_EXAMPLE_DATA.")

    df = df.copy()
    df["has_claims_last_3y"] = df["has_claims_last_3y"].astype(int)

    print(f"Loaded {len(df)} SYNTHETIC leads from {paths['data'].name}")
    print(f"  conversion rate: {df[TARGET].mean():.1%}  ({int(df[TARGET].sum())} positives)")

    X = df[FEATURE_COLS]
    y = df[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    # Keep a frame for group-wise fairness slices (same order as y_test).
    test_frame = X_test.copy()
    print(f"Train: {len(X_train)}  |  Test: {len(X_test)}  "
          f"(test prevalence {y_test.mean():.1%})")
    print()

    models = {
        "dummy_prior": build_dummy(),
        "logistic_regression": build_logistic(),
        "gradient_boosting": build_boosting(),
    }
    probas: dict[str, np.ndarray] = {}
    fitted = {}
    all_metrics: dict = {
        "dataset": {
            "file": paths["data"].name,
            "data_source": "SYNTHETIC_EXAMPLE_DATA",
            "n_total": int(len(df)),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "overall_conversion_rate": round(float(y.mean()), 4),
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "split": "stratified random 80/20",
            "imbalance_handling": "class_weight='balanced' on LR and HGB",
        },
        "models": {},
    }

    for name, est in models.items():
        est.fit(X_train, y_train)
        # Dummy and the others all implement predict_proba.
        proba = est.predict_proba(X_test)[:, 1]
        fitted[name] = est
        probas[name] = proba
        scores = metric_block(y_test, proba, K_VALUES)
        all_metrics["models"][name] = scores
        print("=" * 68)
        print(
            f"{name:22s}  ROC-AUC={scores['roc_auc']:.4f}  "
            f"PR-AUC={scores['pr_auc']:.4f}  "
            f"P@50={scores['precision_at_k']['50']:.3f}  "
            f"P@top10%={scores['precision_at_top_10pct']:.3f}"
        )
    print("=" * 68)

    primary = max(
        (n for n in all_metrics["models"] if n != "dummy_prior"),
        key=lambda n: all_metrics["models"][n]["pr_auc"],
    )
    all_metrics["primary_model"] = primary
    print(f"Primary model (highest PR-AUC, excluding dummy): {primary}")

    # Explainability: prefer SHAP on a sample of the test set; otherwise
    # permutation importance (ROC-AUC drop) on the primary model.
    shap_plot = paths["reports"] / "shap_importance.png"
    shap_info = try_shap(fitted[primary], X_test, shap_plot)
    explainer_used = "shap" if shap_info else "permutation_importance"
    all_metrics["explainability"] = {"method": explainer_used, "shap": shap_info}

    print("\nComputing permutation importance…")
    perm = permutation_importance(
        fitted[primary],
        X_test,
        y_test,
        n_repeats=15,
        random_state=RANDOM_STATE,
        scoring="roc_auc",
        n_jobs=-1,
    )
    imp_df = pd.DataFrame(
        {
            "feature": FEATURE_COLS,
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    print(imp_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    all_metrics["permutation_importance_roc_auc"] = imp_df.to_dict(orient="records")
    perm_plot = paths["reports"] / "permutation_importance.png"
    plot_importance(
        imp_df,
        f"Permutation importance (ROC-AUC drop) — {primary.replace('_', ' ')}",
        perm_plot,
    )
    if not shap_info:
        # Copy the permutation plot into the "main" explainability filename
        # so the README can always point at one path.
        import shutil

        shutil.copyfile(perm_plot, paths["reports"] / "explainability.png")
    else:
        import shutil

        shutil.copyfile(shap_plot, paths["reports"] / "explainability.png")

    # Fairness / slice analysis: does ranking quality differ by region
    # or industry? These are not protected characteristics, but they
    # are the groups a commercial desk might quietly deprioritise.
    all_metrics["slices"] = {
        "region": slice_metrics(test_frame, y_test, probas[primary], "region"),
        "industry": slice_metrics(test_frame, y_test, probas[primary], "industry"),
        "source": slice_metrics(test_frame, y_test, probas[primary], "source"),
    }

    y_np = y_test.to_numpy()
    roc_p = paths["reports"] / "roc_curve.png"
    pr_p = paths["reports"] / "pr_curve.png"
    cal_p = paths["reports"] / "calibration.png"
    pk_p = paths["reports"] / "precision_at_k.png"
    plot_roc(y_np, probas, roc_p)
    plot_pr(y_np, probas, pr_p)
    plot_calibration(y_np, probas, cal_p)
    plot_precision_at_k(y_np, {k: v for k, v in probas.items()}, K_VALUES, pk_p)

    bundle = {
        "model_name": primary,
        "models": fitted,
        "feature_cols": FEATURE_COLS,
        "cat_cols": CAT_COLS,
        "num_cols": NUM_COLS,
        "target": TARGET,
        "choices": {col: sorted(df[col].unique().tolist()) for col in CAT_COLS},
        "numeric_ranges": {
            col: {"min": int(df[col].min()), "max": int(df[col].max())}
            if col != "response_time_hours"
            else {"min": float(df[col].min()), "max": float(df[col].max())}
            for col in NUM_COLS
            if col != "has_claims_last_3y"
        },
        "metrics": all_metrics,
        "data_source": "SYNTHETIC_EXAMPLE_DATA",
    }
    model_path = paths["models"] / "lead_scorer.joblib"
    joblib.dump(bundle, model_path, compress=3)

    all_metrics["artefacts"] = {
        "model_bundle": str(model_path.relative_to(paths["root"])),
        "roc_curve_png": str(roc_p.relative_to(paths["root"])),
        "pr_curve_png": str(pr_p.relative_to(paths["root"])),
        "calibration_png": str(cal_p.relative_to(paths["root"])),
        "precision_at_k_png": str(pk_p.relative_to(paths["root"])),
        "explainability_png": "reports/explainability.png",
        "permutation_importance_png": str(perm_plot.relative_to(paths["root"])),
    }
    all_metrics["generated_at_dublin"] = datetime.now(
        ZoneInfo("Europe/Dublin")
    ).strftime("%Y-%m-%d %H:%M %Z")
    metrics_path = paths["reports"] / "metrics.json"
    metrics_path.write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")

    lines = [
        "Insurance lead scorer — SYNTHETIC data, stratified 80/20 test set",
        f"n_train={len(X_train)}  n_test={len(X_test)}  "
        f"test prevalence={y_test.mean():.1%}",
        f"Imbalance handling: class_weight='balanced'. Primary model: {primary}",
        "",
    ]
    for name, scores in all_metrics["models"].items():
        lines.append(
            f"{name:22s}  ROC-AUC={scores['roc_auc']:.4f}  "
            f"PR-AUC={scores['pr_auc']:.4f}  "
            f"P@25={scores['precision_at_k']['25']:.3f}  "
            f"P@50={scores['precision_at_k']['50']:.3f}  "
            f"P@100={scores['precision_at_k']['100']:.3f}  "
            f"P@top10%={scores['precision_at_top_10pct']:.3f}"
        )
    lines.append("")
    lines.append(f"Explainability method: {explainer_used}")
    (paths["reports"] / "metrics.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSaved metrics and plots under {paths['reports']}")
    print(f"Saved {model_path}")


if __name__ == "__main__":
    main()
