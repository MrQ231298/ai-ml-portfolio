"""Train candidates on 2011, select on Q1 2012, track everything in MLflow and
register the winner as `bike-demand` with the alias `champion`."""

from __future__ import annotations

import json
import time

import joblib
import mlflow
import numpy as np
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from common import CITATION, FEATURES, LICENCE, MLRUNS, MODEL_NAME, MODELS, REPORTS, SHA256, TARGET, TRACKING_URI, URL, load_hourly, write_json
from model_lib import CANDIDATES, fit, scores

# Types reviewed and trusted for MLflow's skops serialisation (we created these files ourselves).
SKOPS_TRUSTED = ["functools.partial", "numpy.log1p", "sklearn.preprocessing._function_transformer.FunctionTransformer", "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
                 "sklearn.utils.validation.check_array", "numpy.log1p", "numpy.expm1", "numpy.dtype"]

TRAIN_END = "2012-01-01"
VALID_END = "2012-04-01"


def reference_profile(df, model) -> dict:
    """Feature samples and predictions kept with the model so the monitor has a reference."""
    pred = np.clip(model.predict(df[FEATURES]), 0, None)
    return {"n": int(len(df)), "window": [str(df.ts.min()), str(df.ts.max())],
            "features": {c: df[c].tolist() for c in FEATURES}, "prediction": pred.round(3).tolist()}


def main():
    MLRUNS.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("bike-demand-hourly")
    df = load_hourly()
    tr, va = df[df.ts < TRAIN_END], df[(df.ts >= TRAIN_END) & (df.ts < VALID_END)]
    results = {}
    for name, params in {"seasonal_naive_baseline": {"kind": "baseline"},
                         "previous_hour_baseline": {"kind": "baseline_prev_hour"}, **CANDIDATES}.items():
        with mlflow.start_run(run_name=name) as run:
            t0 = time.time()
            m = fit(tr, params)
            fit_s = time.time() - t0
            s = scores(va[TARGET], m.predict(va[FEATURES]))
            mlflow.log_params({**params, "train_window": f"2011-01-01..2011-12-31", "valid_window": "2012-01-01..2012-03-31",
                               "n_features": len(FEATURES)})
            mlflow.log_metrics({f"valid_{k}": v for k, v in s.items()} | {"fit_seconds": fit_s})
            mlflow.set_tags({"data_sha256": SHA256, "licence": LICENCE})
            if not params["kind"].startswith("baseline"):
                sig = infer_signature(va[FEATURES].head(50), m.predict(va[FEATURES].head(50)))
                mlflow.sklearn.log_model(m, name="model", signature=sig, input_example=va[FEATURES].head(3),
                                         skops_trusted_types=SKOPS_TRUSTED)
            results[name] = {"run_id": run.info.run_id, "params": params, "valid": s, "fit_seconds": round(fit_s, 3), "model": m}
            print(name, {k: round(v, 3) for k, v in s.items()})
    best = min((k for k in results if not k.endswith("_baseline")), key=lambda k: results[k]["valid"]["mae"])
    best_baseline = min((k for k in results if k.endswith("_baseline")), key=lambda k: results[k]["valid"]["mae"])
    client = MlflowClient()
    mv = mlflow.register_model(f"runs:/{results[best]['run_id']}/model", MODEL_NAME)
    client.set_registered_model_alias(MODEL_NAME, "champion", mv.version)
    client.set_model_version_tag(MODEL_NAME, mv.version, "selected_by", "lowest valid MAE, Q1 2012")
    # Serving artefact (the API loads this file; no MLflow server needed in the container)
    MODELS.mkdir(exist_ok=True)
    model = results[best]["model"]
    joblib.dump(model, MODELS / "model.joblib", compress=3)
    card = {"name": MODEL_NAME, "registry_version": int(mv.version), "alias": "champion", "candidate": best,
            "params": results[best]["params"], "features": FEATURES, "target": TARGET,
            "train_window": ["2011-01-01", "2011-12-31"], "valid_window": ["2012-01-01", "2012-03-31"],
            "valid_metrics": results[best]["valid"], "data": {"url": URL, "sha256": SHA256, "licence": LICENCE, "citation": CITATION},
            "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    write_json(MODELS / "model_card.json", card)
    write_json(MODELS / "reference_profile.json", reference_profile(tr, model))
    write_json(REPORTS / "training_metrics.json", {
        "selected": best, "registry_version": int(mv.version), "best_baseline": best_baseline,
        "improvement_vs_best_baseline_mae_pct": round(100 * (1 - results[best]["valid"]["mae"] / results[best_baseline]["valid"]["mae"]), 2),
        "candidates": {k: {"valid": v["valid"], "fit_seconds": v["fit_seconds"], "params": v["params"], "mlflow_run_id": v["run_id"]}
                       for k, v in results.items()},
        "rows": {"train": int(len(tr)), "valid": int(len(va)), "total": int(len(df))}})
    print("registered", MODEL_NAME, "v", mv.version, "=", best)


if __name__ == "__main__":
    main()
