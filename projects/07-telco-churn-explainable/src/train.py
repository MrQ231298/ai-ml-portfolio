"""
Train, calibrate, explain and cost-optimise a customer-churn model on the
public IBM Telco Customer Churn dataset (7,043 customers).

Protocol
--------
1. Stratified 80/20 split -> a held-out test set touched only at the end.
2. On the training set: 5-fold stratified cross-validation comparing a
   dummy prior, a logistic-regression baseline and a small grid of
   histogram gradient boosting models (ROC-AUC, PR-AUC, Brier, log loss).
3. Calibration: isotonic and sigmoid (Platt) calibration of the chosen
   gradient boosting model with internal 5-fold CV; compared on Brier
   score and expected calibration error (ECE) on the test set.
4. Business threshold: out-of-fold calibrated probabilities on the
   training set are used to pick the contact threshold that maximises an
   illustrative net benefit (offer cost vs value of a saved customer).
   That threshold is then applied, unchanged, to the test set.
5. Explainability: SHAP (TreeExplainer) on the gradient boosting model
   plus permutation importance on raw columns.
6. Fairness audit: gender and SeniorCitizen are excluded from the model
   inputs, then used to slice test metrics and contact rates.

    python src/train.py
"""

from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from churn_lib import (  # noqa: E402
    AUDIT_COLS, BUNDLE_PATH, CAT_COLS, FEATURE_COLS, NUM_COLS, ROOT, TARGET,
    load_clean, net_benefit,
)

RANDOM_STATE = 42
REPORTS = ROOT / "reports"

# Illustrative retention-campaign economics (dataset currency, assumed USD).
COST_CFG = {"offer_cost": 40.0, "save_rate": 0.30, "value_months": 12}

HGB_GRID = [
    {"learning_rate": lr, "max_leaf_nodes": leaves, "min_samples_leaf": msl}
    for lr in (0.03, 0.1)
    for leaves in (8, 31)
    for msl in (20, 80)
]


def preprocessor(scale: bool) -> ColumnTransformer:
    num = StandardScaler() if scale else "passthrough"
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
            ("num", num, NUM_COLS),
        ],
        verbose_feature_names_out=False,
    )


def make_lr() -> Pipeline:
    return Pipeline([("prep", preprocessor(True)), ("clf", LogisticRegression(max_iter=2000, C=1.0))])


def make_hgb(**params) -> Pipeline:
    clf = HistGradientBoostingClassifier(
        max_iter=300, early_stopping=True, validation_fraction=0.15,
        random_state=RANDOM_STATE, **params,
    )
    return Pipeline([("prep", preprocessor(False)), ("clf", clf)])


def ece(y, p, n_bins: int = 10) -> float:
    """Expected calibration error with equal-width bins."""
    y, p = np.asarray(y), np.asarray(p)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(total)


def scores(y, p) -> dict:
    return {
        "roc_auc": round(float(roc_auc_score(y, p)), 4),
        "pr_auc": round(float(average_precision_score(y, p)), 4),
        "brier": round(float(brier_score_loss(y, p)), 4),
        "log_loss": round(float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))), 4),
        "ece": round(ece(y, p), 4),
    }


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    REPORTS.mkdir(exist_ok=True)
    BUNDLE_PATH.parent.mkdir(exist_ok=True)

    df = load_clean()
    X, y = df[FEATURE_COLS + AUDIT_COLS + ["customerID"]], df[TARGET].to_numpy()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    Xf_tr, Xf_te = X_tr[FEATURE_COLS], X_te[FEATURE_COLS]
    print(f"rows={len(df)} churn_rate={y.mean():.4f} train={len(y_tr)} test={len(y_te)}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {"roc_auc": "roc_auc", "pr_auc": "average_precision",
               "brier": "neg_brier_score", "log_loss": "neg_log_loss"}

    def cv_summary(model) -> dict:
        r = cross_validate(model, Xf_tr, y_tr, cv=cv, scoring=scoring, n_jobs=-1)
        out = {}
        for k in scoring:
            vals = r[f"test_{k}"] * (-1 if k in ("brier", "log_loss") else 1)
            out[k] = {"mean": round(float(vals.mean()), 4), "std": round(float(vals.std()), 4)}
        return out

    # ---- 1. Cross-validated model comparison on the training set ----
    cv_results = {
        "dummy_prior": cv_summary(Pipeline([("prep", preprocessor(False)), ("clf", DummyClassifier(strategy="prior"))])),
        "logistic_regression": cv_summary(make_lr()),
    }
    grid_results = []
    for params in HGB_GRID:
        res = cv_summary(make_hgb(**params))
        grid_results.append({"params": params, **res})
        print("HGB", params, res["roc_auc"])
    best = max(grid_results, key=lambda r: r["roc_auc"]["mean"])
    best_params = best["params"]
    cv_results["hist_gradient_boosting_best"] = {k: best[k] for k in scoring}
    print("best HGB params", best_params)

    # ---- 2. Fit final models on the full training set ----
    lr = make_lr().fit(Xf_tr, y_tr)
    hgb = make_hgb(**best_params).fit(Xf_tr, y_tr)
    cal = {}
    for method in ("isotonic", "sigmoid"):
        cal[method] = CalibratedClassifierCV(make_hgb(**best_params), method=method, cv=5).fit(Xf_tr, y_tr)

    p_te = {
        "dummy_prior": np.full(len(y_te), y_tr.mean()),
        "logistic_regression": lr.predict_proba(Xf_te)[:, 1],
        "hgb_uncalibrated": hgb.predict_proba(Xf_te)[:, 1],
        "hgb_isotonic": cal["isotonic"].predict_proba(Xf_te)[:, 1],
        "hgb_sigmoid": cal["sigmoid"].predict_proba(Xf_te)[:, 1],
    }
    test_scores = {k: scores(y_te, v) for k, v in p_te.items()}
    for k, v in test_scores.items():
        print(k, v)

    # Choose the deployed model: lowest test-agnostic choice = lowest CV Brier
    # among calibrated variants is not available, so we pick by *training*
    # out-of-fold Brier below (never by test score).
    oof = {}
    for method in ("isotonic", "sigmoid"):
        oof[method] = cross_val_predict(
            CalibratedClassifierCV(make_hgb(**best_params), method=method, cv=5),
            Xf_tr, y_tr, cv=cv, method="predict_proba", n_jobs=-1,
        )[:, 1]
    oof_brier = {m: round(float(brier_score_loss(y_tr, p)), 4) for m, p in oof.items()}
    chosen = min(oof_brier, key=oof_brier.get)
    print("OOF Brier", oof_brier, "-> chosen", chosen)
    model = cal[chosen]
    p_model_te = p_te[f"hgb_{chosen}"]

    # ---- 3. Business-cost threshold on out-of-fold training predictions ----
    thresholds = np.round(np.arange(0.02, 0.96, 0.01), 2)
    mc_tr = Xf_tr["MonthlyCharges"].to_numpy()
    nb_tr = [net_benefit(y_tr, oof[chosen] >= t, mc_tr, COST_CFG) for t in thresholds]
    t_star = float(thresholds[int(np.argmax(nb_tr))])

    mc_te = Xf_te["MonthlyCharges"].to_numpy()

    def policy(mask) -> dict:
        mask = np.asarray(mask, dtype=bool)
        return {
            "contacted": int(mask.sum()),
            "contact_rate": round(float(mask.mean()), 4),
            "precision": round(float(precision_score(y_te, mask, zero_division=0)), 4),
            "recall": round(float(recall_score(y_te, mask, zero_division=0)), 4),
            "net_benefit": round(net_benefit(y_te, mask, mc_te, COST_CFG), 2),
        }

    policies = {
        "contact_nobody": policy(np.zeros(len(y_te), bool)),
        "contact_everybody": policy(np.ones(len(y_te), bool)),
        "threshold_0.50": policy(p_model_te >= 0.5),
        f"cost_optimal_threshold_{t_star:.2f}": policy(p_model_te >= t_star),
    }
    for k, v in policies.items():
        print(k, v)

    # ---- 4. Explainability ----
    prep = hgb.named_steps["prep"]
    Xt_te = pd.DataFrame(prep.transform(Xf_te), columns=prep.get_feature_names_out())
    shap_info = {"method": None}
    try:
        import shap

        explainer = shap.TreeExplainer(hgb.named_steps["clf"])
        sv = explainer.shap_values(Xt_te)
        sv = np.asarray(sv)
        if sv.ndim == 3:
            sv = sv[..., 1]
        mean_abs = pd.Series(np.abs(sv).mean(axis=0), index=Xt_te.columns).sort_values(ascending=False)
        shap_info = {"method": f"shap.TreeExplainer {shap.__version__}",
                     "top10_mean_abs_shap": {k: round(float(v), 4) for k, v in mean_abs.head(10).items()}}
        fig, ax = plt.subplots(figsize=(7, 5))
        mean_abs.head(15)[::-1].plot.barh(ax=ax, color="#2b6cb0")
        ax.set_xlabel("mean |SHAP value| (log-odds)")
        ax.set_title("Top 15 drivers of churn risk (test set, SHAP)")
        fig.tight_layout(); fig.savefig(REPORTS / "shap_importance.png", dpi=130); plt.close(fig)
    except Exception as exc:  # pragma: no cover - documented fallback
        shap_info = {"method": "unavailable", "error": repr(exc)}
        print("SHAP failed:", exc)

    perm = permutation_importance(hgb, Xf_te, y_te, scoring="roc_auc", n_repeats=10,
                                  random_state=RANDOM_STATE, n_jobs=-1)
    perm_s = pd.Series(perm.importances_mean, index=FEATURE_COLS).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 5))
    perm_s[::-1].plot.barh(ax=ax, color="#dd6b20")
    ax.set_xlabel("drop in ROC-AUC when column is shuffled")
    ax.set_title("Permutation importance (test set, raw columns)")
    fig.tight_layout(); fig.savefig(REPORTS / "permutation_importance.png", dpi=130); plt.close(fig)

    # ---- 5. Fairness slice audit (attributes NOT used as inputs) ----
    slices = {}
    contact = p_model_te >= t_star
    for col in AUDIT_COLS:
        for val in sorted(X_te[col].unique()):
            m = (X_te[col] == val).to_numpy()
            slices[f"{col}={val}"] = {
                "n": int(m.sum()),
                "churn_rate": round(float(y_te[m].mean()), 4),
                "mean_predicted": round(float(p_model_te[m].mean()), 4),
                "roc_auc": round(float(roc_auc_score(y_te[m], p_model_te[m])), 4),
                "contact_rate": round(float(contact[m].mean()), 4),
                "recall": round(float(recall_score(y_te[m], contact[m], zero_division=0)), 4),
            }

    # ---- 6. Plots ----
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    for name in ("logistic_regression", "hgb_uncalibrated", f"hgb_{chosen}"):
        fr, mp = calibration_curve(y_te, p_te[name], n_bins=10, strategy="quantile")
        ax.plot(mp, fr, marker="o", label=f"{name} (ECE {test_scores[name]['ece']:.3f})")
    ax.set_xlabel("mean predicted churn probability"); ax.set_ylabel("observed churn rate")
    ax.set_title("Reliability diagram (test set)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(REPORTS / "calibration.png", dpi=130); plt.close(fig)

    nb_te = [net_benefit(y_te, p_model_te >= t, mc_te, COST_CFG) for t in thresholds]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(thresholds, np.array(nb_tr) / len(y_tr), label="train (out-of-fold), per customer")
    ax.plot(thresholds, np.array(nb_te) / len(y_te), label="test, per customer")
    ax.axvline(t_star, color="k", ls="--", lw=1, label=f"chosen threshold {t_star:.2f}")
    ax.axvline(0.5, color="grey", ls=":", lw=1, label="default 0.50")
    ax.set_xlabel("contact if calibrated churn probability >= threshold")
    ax.set_ylabel("net benefit per customer")
    ax.set_title("Illustrative retention-campaign value by threshold"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(REPORTS / "cost_curve.png", dpi=130); plt.close(fig)

    # ---- 7. Persist ----
    joblib.dump({
        "model": model, "hgb_uncalibrated": hgb, "threshold": t_star, "cost_cfg": COST_CFG,
        "feature_cols": FEATURE_COLS, "cat_choices": {c: sorted(df[c].unique().tolist()) for c in CAT_COLS},
        "num_ranges": {c: [float(df[c].min()), float(df[c].max()), float(df[c].median())] for c in NUM_COLS},
        "calibration": chosen,
    }, BUNDLE_PATH)

    metrics = {
        "run_at_europe_dublin": datetime.now(ZoneInfo("Europe/Dublin")).isoformat(timespec="seconds"),
        "dataset": {"rows": int(len(df)), "churn_rate": round(float(y.mean()), 4),
                    "train": int(len(y_tr)), "test": int(len(y_te)), "test_churners": int(y_te.sum())},
        "features_used": FEATURE_COLS, "excluded_for_fairness_audit": AUDIT_COLS,
        "cv_5fold_train": cv_results, "hgb_grid": grid_results, "hgb_best_params": best_params,
        "test": test_scores, "oof_brier_calibration": oof_brier, "deployed_calibration": chosen,
        "cost_assumptions": COST_CFG, "cost_optimal_threshold": t_star, "policies_test": policies,
        "explainability": shap_info,
        "permutation_importance_top10": {k: round(float(v), 4) for k, v in perm_s.head(10).items()},
        "fairness_slices_test": slices,
    }
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("wrote", REPORTS / "metrics.json")


if __name__ == "__main__":
    main()
