"""
Train and evaluate models that predict published RTB average monthly rent.

Target
------
`avg_monthly_rent_eur` — the official average, in Euro, for one
year × location × property type × bedroom-count cell in table RIA02.

Split
-----
A *temporal* hold-out, not a random row split: train on 2008–2023,
test on 2024–2025. That matches the real use of the model (estimate
a recent or current average from the history) and avoids leaking
future rent levels into training.

Models (same features, same split)
----------------------------------
1. Persistence baseline — last observed training-year rent for the
   same location / type / bedrooms, with fallbacks.
2. Linear regression on a log(rent) scale (back-transformed to Euro).
3. Gradient boosting (HistGradientBoosting, ordinal-encoded categoricals, rent on the Euro scale).

Metrics are always reported in Euro: MAE, RMSE, R². Feature importance
is permutation importance on the held-out years.

Usage
-----
    python src/train.py
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

# Allow `from baseline import ...` whether we are run as src/train.py or as a module.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from baseline import PersistenceBaseline

TRAIN_YEARS_MAX = 2023
TEST_YEARS = (2024, 2025)
RANDOM_STATE = 42

CAT_COLS = ["location", "county", "location_level", "property_type"]
NUM_COLS = ["year", "n_bedrooms"]
FEATURE_COLS = CAT_COLS + NUM_COLS
TARGET = "avg_monthly_rent_eur"


def project_paths() -> dict[str, Path]:
    root = Path(__file__).resolve().parents[1]
    return {
        "root": root,
        "data": root / "data" / "ria02_average_monthly_rent.csv",
        "meta": root / "data" / "source_metadata.json",
        "models": root / "models",
        "reports": root / "reports",
    }


def metrics_dict(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    bias = float(np.mean(y_pred - y_true))  # negative => under-prediction
    return {
        "mae": round(float(mae), 2),
        "rmse": round(rmse, 2),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
        "mean_bias": round(bias, 2),
        "n": int(len(y_true)),
    }


def build_linear_pipeline() -> TransformedTargetRegressor:
    """Linear regression on log(rent), predictions back in Euro."""
    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
            ("num", StandardScaler(), NUM_COLS),
        ]
    )
    inner = Pipeline(
        [
            ("pre", pre),
            ("model", LinearRegression()),
        ]
    )
    return TransformedTargetRegressor(
        regressor=inner, func=np.log, inverse_func=np.exp, check_inverse=False
    )


def build_boosting_pipeline() -> Pipeline:
    """
    Histogram gradient boosting is the tree model used here. It keeps a
    much smaller artefact than a 300-tree random forest (important for
    GitHub) and accepts a mix of categorical and numeric columns.
    After the ColumnTransformer, columns 0–3 are the encoded categoricals
    and columns 4–5 are year and bedroom count.
    """
    pre = ColumnTransformer(
        [
            (
                "cat",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value", unknown_value=-1
                ),
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
                HistGradientBoostingRegressor(
                    max_depth=8,
                    max_iter=300,
                    learning_rate=0.08,
                    min_samples_leaf=10,
                    random_state=RANDOM_STATE,
                    # sklearn caps categorical cardinality at 255; location
                    # has more unique values, so only county / level / type
                    # are declared categorical (indices 1–3 after encoding).
                    # Location (index 0) is passed as an ordinal integer.
                    categorical_features=[1, 2, 3],
                ),
            ),
        ]
    )


def plot_pred_vs_actual(y_true, predictions: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharex=True, sharey=True)
    low = min(y_true.min(), min(p.min() for p in predictions.values()))
    high = max(y_true.max(), max(p.max() for p in predictions.values()))
    pad = 50
    for ax, (name, pred) in zip(axes, predictions.items()):
        ax.scatter(y_true, pred, alpha=0.35, s=16, edgecolors="none")
        ax.plot([low - pad, high + pad], [low - pad, high + pad], color="black", lw=0.8)
        ax.set_title(name.replace("_", " "))
        ax.set_xlabel("Actual average rent (€)")
        ax.set_ylabel("Predicted (€)")
        ax.set_xlim(low - pad, high + pad)
        ax.set_ylim(low - pad, high + pad)
    fig.suptitle(
        "Predicted vs actual RTB average monthly rent — test years 2024–2025",
        y=1.03,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_importance(imp_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    sns.barplot(
        data=imp_df.sort_values("importance_mean"),
        y="feature",
        x="importance_mean",
        color="#2a6f97",
        xerr=imp_df.sort_values("importance_mean")["importance_std"],
        ax=ax,
    )
    ax.set_xlabel("Permutation importance (increase in MAE, Euro)")
    ax.set_ylabel("")
    ax.set_title("Gradient boosting — permutation importance on 2024–2025")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_mae_by_year(frame: pd.DataFrame, predictions: dict, out_path: Path) -> None:
    rows = []
    for name, pred in predictions.items():
        tmp = frame.assign(pred=pred, model=name)
        for year, part in tmp.groupby("year"):
            rows.append(
                {
                    "year": int(year),
                    "model": name.replace("_", " "),
                    "mae": mean_absolute_error(part[TARGET], part["pred"]),
                }
            )
    plot_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    sns.barplot(data=plot_df, x="year", y="mae", hue="model", ax=ax)
    ax.set_ylabel("MAE (€)")
    ax.set_xlabel("Test year")
    ax.set_title("MAE by test year")
    ax.legend(title="")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    paths = project_paths()
    paths["models"].mkdir(parents=True, exist_ok=True)
    paths["reports"].mkdir(parents=True, exist_ok=True)

    if not paths["data"].exists():
        raise FileNotFoundError(
            f"{paths['data']} not found. Run: python src/fetch_cso_ria02.py"
        )

    df = pd.read_csv(paths["data"])
    source_meta = {}
    if paths["meta"].exists():
        source_meta = json.loads(paths["meta"].read_text(encoding="utf-8"))

    train = df[df["year"] <= TRAIN_YEARS_MAX].copy()
    test = df[df["year"].isin(TEST_YEARS)].copy()
    if train.empty or test.empty:
        raise RuntimeError(
            f"Unexpected year coverage: train {len(train)} test {len(test)}"
        )

    print(
        f"Loaded {len(df):,} published average-rent cells "
        f"({df['location'].nunique()} locations, "
        f"{int(df['year'].min())}–{int(df['year'].max())})"
    )
    print(f"Train 2008–{TRAIN_YEARS_MAX}: {len(train):,}  |  "
          f"Test {TEST_YEARS[0]}–{TEST_YEARS[1]}: {len(test):,}")
    print()

    y_train = train[TARGET]
    y_test = test[TARGET]
    X_train = train[FEATURE_COLS]
    X_test = test[FEATURE_COLS]

    fitted: dict = {}
    predictions: dict[str, np.ndarray] = {}
    all_metrics: dict = {
        "data": {
            "source_table": "RIA02",
            "licence": source_meta.get("licence"),
            "api_url": source_meta.get("api_url"),
            "n_rows": int(len(df)),
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "train_years": f"2008-{TRAIN_YEARS_MAX}",
            "test_years": list(TEST_YEARS),
            "n_locations": int(df["location"].nunique()),
            "split": "temporal hold-out (not a random split)",
        },
        "models": {},
    }

    # 1. Persistence baseline
    baseline = PersistenceBaseline().fit(train)
    pred_base = baseline.predict(test)
    fitted["persistence_baseline"] = baseline
    predictions["persistence_baseline"] = pred_base
    all_metrics["models"]["persistence_baseline"] = metrics_dict(y_test, pred_base)

    # 2. Log-linear regression
    linear = build_linear_pipeline()
    linear.fit(X_train, y_train)
    pred_lr = linear.predict(X_test)
    fitted["linear_regression"] = linear
    predictions["linear_regression"] = pred_lr
    all_metrics["models"]["linear_regression"] = metrics_dict(y_test, pred_lr)

    # 3. Gradient boosting
    boosting = build_boosting_pipeline()
    boosting.fit(X_train, y_train)
    pred_gb = boosting.predict(X_test)
    fitted["gradient_boosting"] = boosting
    predictions["gradient_boosting"] = pred_gb
    all_metrics["models"]["gradient_boosting"] = metrics_dict(y_test, pred_gb)

    for name, scores in all_metrics["models"].items():
        print("=" * 64)
        print(
            f"{name:24s}  MAE={scores['mae']:.2f}  "
            f"RMSE={scores['rmse']:.2f}  R2={scores['r2']:.4f}  "
            f"bias={scores['mean_bias']:.2f}  n={scores['n']}"
        )
    print("=" * 64)

    # The served "headline" model is whichever has the lowest test MAE.
    # On official averages this is often the persistence baseline — do not
    # prefer an ML model just because it is more complex.
    primary = min(all_metrics["models"], key=lambda n: all_metrics["models"][n]["mae"])
    all_metrics["primary_model"] = primary
    print(f"Lowest test MAE: {primary}")

    # Per-year breakdown for the report
    by_year = {}
    for year in TEST_YEARS:
        mask = test["year"].eq(year).to_numpy()
        by_year[str(year)] = {
            name: metrics_dict(y_test[mask], pred[mask])
            for name, pred in predictions.items()
        }
    all_metrics["by_test_year"] = by_year

    # Permutation importance on the original columns, scored by MAE.
    print("\nComputing permutation importance (this can take a minute)…")
    perm = permutation_importance(
        boosting,
        X_test,
        y_test,
        n_repeats=15,
        random_state=RANDOM_STATE,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )
    # With scoring="neg_mean_absolute_error", importances_mean is already
    # the increase in MAE (Euro) when that column is shuffled — positive
    # means the feature was useful. Do not flip the sign.
    imp_rows = []
    for i, col in enumerate(FEATURE_COLS):
        values = perm.importances[i]
        imp_rows.append(
            {
                "feature": col,
                "importance_mean": float(values.mean()),
                "importance_std": float(values.std(ddof=1)),
            }
        )
    imp_df = pd.DataFrame(imp_rows).sort_values(
        "importance_mean", ascending=False
    )
    print(imp_df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    all_metrics["permutation_importance_mae"] = imp_df.to_dict(orient="records")

    # Plots
    pred_plot = paths["reports"] / "pred_vs_actual.png"
    imp_plot = paths["reports"] / "feature_importance.png"
    year_plot = paths["reports"] / "mae_by_year.png"
    plot_pred_vs_actual(y_test.to_numpy(), predictions, pred_plot)
    plot_importance(imp_df, imp_plot)
    plot_mae_by_year(test, predictions, year_plot)

    # Save all three models. The headline model is the one with lowest test MAE.
    bundle = {
        "model_name": all_metrics["primary_model"],
        "models": {
            "persistence_baseline": baseline,
            "linear_regression": linear,
            "gradient_boosting": boosting,
        },
        "feature_cols": FEATURE_COLS,
        "cat_cols": CAT_COLS,
        "num_cols": NUM_COLS,
        "target": TARGET,
        "locations": sorted(df["location"].unique().tolist()),
        "counties": sorted(df["county"].unique().tolist()),
        "location_to_county": (
            df.drop_duplicates("location")
            .set_index("location")["county"]
            .to_dict()
        ),
        "location_to_level": (
            df.drop_duplicates("location")
            .set_index("location")["location_level"]
            .to_dict()
        ),
        "property_types": sorted(df["property_type"].unique().tolist()),
        "bedroom_map": (
            df.drop_duplicates("bedrooms")
            .set_index("bedrooms")["n_bedrooms"]
            .sort_values()
            .to_dict()
        ),
        "year_min": int(df["year"].min()),
        "year_max": int(df["year"].max()),
        "metrics": all_metrics,
        "source": {
            "table_id": "RIA02",
            "attribution": source_meta.get("attribution"),
            "licence": source_meta.get("licence"),
            "api_url": source_meta.get("api_url"),
        },
    }
    model_path = paths["models"] / "rent_predictor.joblib"
    joblib.dump(bundle, model_path, compress=3)

    all_metrics["artefacts"] = {
        "model_bundle": str(model_path.relative_to(paths["root"])),
        "pred_vs_actual_png": str(pred_plot.relative_to(paths["root"])),
        "feature_importance_png": str(imp_plot.relative_to(paths["root"])),
        "mae_by_year_png": str(year_plot.relative_to(paths["root"])),
    }
    all_metrics["generated_at_dublin"] = datetime.now(
        ZoneInfo("Europe/Dublin")
    ).strftime("%Y-%m-%d %H:%M %Z")

    metrics_path = paths["reports"] / "metrics.json"
    metrics_path.write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")

    lines = [
        "Irish rent predictor — temporal test years 2024–2025",
        "Source: RTB / CSO table RIA02 (CC BY 4.0)",
        f"Train: 2008–{TRAIN_YEARS_MAX} ({len(train)} rows)  "
        f"Test: 2024–2025 ({len(test)} rows)",
        "",
    ]
    for name, scores in all_metrics["models"].items():
        lines.append(
            f"{name:24s}  MAE={scores['mae']:.2f}  "
            f"RMSE={scores['rmse']:.2f}  R2={scores['r2']:.4f}  "
            f"bias={scores['mean_bias']:.2f}  n={scores['n']}"
        )
    lines.append("")
    lines.append("Permutation importance (MAE increase, Euro) — gradient boosting:")
    for rec in all_metrics["permutation_importance_mae"]:
        lines.append(
            f"  {rec['feature']:18s}  {rec['importance_mean']:.2f} "
            f"± {rec['importance_std']:.2f}"
        )
    report_path = paths["reports"] / "metrics.txt"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nSaved {pred_plot}")
    print(f"Saved {imp_plot}")
    print(f"Saved {year_plot}")
    print(f"Saved {metrics_path}")
    print(f"Saved {model_path}")


if __name__ == "__main__":
    main()
