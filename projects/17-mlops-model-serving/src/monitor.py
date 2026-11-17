"""Replay April-December 2012 month by month as if the model were in production.

Each month: feature and prediction drift against the champion's training data
(PSI, KS), and accuracy once the month's actual counts are known. Rules are
fixed here, before the replay:

  drift alert  any monitored (non-calendar) feature or the prediction has PSI >= 0.25
  perf alert   monthly MAE > 1.3 x the champion's reference MAE
  retrain      only on a perf alert (drift alone means "investigate", because
               drift without an accuracy loss is common and retraining has a cost)
  challenger   same model config, trained on the trailing 365 days ending 14 days
               before month end; champion and challenger are scored on those last
               14 days; promote if challenger MAE <= 0.95 x champion MAE; a promoted
               challenger is then refitted on the full trailing 365 days.

Promoted models are registered in MLflow as new versions of `bike-demand` and
take the `champion` alias. A counterfactual "never retrain" run is kept.
"""

from __future__ import annotations

import json

import matplotlib
import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from common import CATEGORICAL, FEATURES, MODEL_NAME, MODELS, REPORTS, TARGET, TRACKING_URI, load_hourly, write_json
from drift import PSI_ALERT, PSI_WARN, ks, level, psi
from model_lib import fit, scores
import joblib

from train import SKOPS_TRUSTED, reference_profile

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CALENDAR = {"season"}  # drifts by construction when one month is compared with a full year
MONITORED = [f for f in FEATURES if f not in CALENDAR]
PERF_FACTOR, PROMOTE_FACTOR, HOLDOUT_DAYS = 1.3, 0.95, 14
MONTHS = pd.period_range("2012-04", "2012-12", freq="M")


def drift_report(ref: pd.DataFrame, ref_pred, cur: pd.DataFrame, cur_pred) -> dict:
    out = {}
    for f in FEATURES:
        r, c = ref[f].dropna(), cur[f].dropna()
        cat = f in CATEGORICAL
        d = {"psi": round(psi(r, c, categorical=cat), 4)}
        if not cat:
            d["ks"], d["ks_p"] = (round(x, 4) for x in ks(r, c))
        d["level"] = level(d["psi"])
        out[f] = d
    out["prediction"] = {"psi": round(psi(ref_pred, cur_pred), 4), "level": level(psi(ref_pred, cur_pred))}
    return out


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()
    card = json.loads((MODELS / "model_card.json").read_text())
    params = card["params"]
    df = load_hourly()
    champion = fit(df[(df.ts >= "2011-01-01") & (df.ts < "2012-01-01")], params)
    static = champion
    ref = df[df.ts < "2012-01-01"]
    ref_pred = champion.predict(ref[FEATURES])
    ref_mae = card["valid_metrics"]["mae"]
    version = card["registry_version"]
    rows, events = [], []
    mlflow.set_experiment("bike-demand-monitoring")
    with mlflow.start_run(run_name="replay-2012-04-to-12"):
        mlflow.log_params({"perf_factor": PERF_FACTOR, "promote_factor": PROMOTE_FACTOR, "psi_alert": PSI_ALERT,
                           "psi_warn": PSI_WARN, "holdout_days": HOLDOUT_DAYS, "monitored_features": ",".join(MONITORED)})
        for i, m in enumerate(MONTHS):
            cur = df[(df.ts >= m.start_time) & (df.ts <= m.end_time)]
            pred = np.clip(champion.predict(cur[FEATURES]), 0, None)
            s = scores(cur[TARGET], pred)
            s_static = scores(cur[TARGET], static.predict(cur[FEATURES]))
            dr = drift_report(ref, ref_pred, cur, pred)
            drifted = [f for f in MONITORED if dr[f]["level"] == "alert"]
            drift_alert = bool(drifted) or dr["prediction"]["level"] == "alert"
            perf_alert = s["mae"] > PERF_FACTOR * ref_mae
            row = {"month": str(m), "champion_version": version, "mae": round(s["mae"], 2),
                   "mae_never_retrain": round(s_static["mae"], 2), "r2": round(s["r2"], 3),
                   "reference_mae": round(ref_mae, 2), "mae_ratio": round(s["mae"] / ref_mae, 3),
                   "perf_alert": perf_alert, "drift_alert": drift_alert, "drifted_features": ",".join(drifted),
                   "prediction_psi": dr["prediction"]["psi"], "season_psi_ignored": dr["season"]["psi"],
                   **{f"psi_{f}": dr[f]["psi"] for f in MONITORED}, "action": "none"}
            mlflow.log_metrics({"mae": s["mae"], "mae_never_retrain": s_static["mae"], "prediction_psi": dr["prediction"]["psi"],
                                "n_drifted_features": len(drifted)}, step=i)
            if perf_alert:
                cut = m.end_time.floor("D") + pd.Timedelta(days=1) - pd.Timedelta(days=HOLDOUT_DAYS)
                start = cut - pd.Timedelta(days=365)
                chall = fit(df[(df.ts >= start) & (df.ts < cut)], params)
                hold = df[(df.ts >= cut) & (df.ts <= m.end_time)]
                mc = scores(hold[TARGET], champion.predict(hold[FEATURES]))["mae"]
                mch = scores(hold[TARGET], chall.predict(hold[FEATURES]))["mae"]
                promote = mch <= PROMOTE_FACTOR * mc
                ev = {"month": str(m), "champion_holdout_mae": round(mc, 2), "challenger_holdout_mae": round(mch, 2),
                      "promoted": promote, "challenger_train_window": [str(start.date()), str(cut.date())]}
                if promote:
                    end = m.end_time.floor("D") + pd.Timedelta(days=1)
                    win = df[(df.ts >= end - pd.Timedelta(days=365)) & (df.ts < end)]
                    champion = fit(win, params)
                    ref, ref_pred, ref_mae = win, champion.predict(win[FEATURES]), mch
                    with mlflow.start_run(run_name=f"retrain-{m}", nested=True) as run:
                        mlflow.log_params({**params, "train_window": f"{(end - pd.Timedelta(days=365)).date()}..{end.date()}"})
                        mlflow.log_metrics({"challenger_holdout_mae": mch, "champion_holdout_mae": mc})
                        sig = infer_signature(win[FEATURES].head(50), champion.predict(win[FEATURES].head(50)))
                        mlflow.sklearn.log_model(champion, name="model", signature=sig, skops_trusted_types=SKOPS_TRUSTED)
                    mv = mlflow.register_model(f"runs:/{run.info.run_id}/model", MODEL_NAME)
                    client.set_model_version_tag(MODEL_NAME, str(version), "status", f"replaced {m}")
                    client.set_registered_model_alias(MODEL_NAME, "champion", mv.version)
                    version = int(mv.version)
                    ev["new_version"] = version
                    ev["refit_window"] = [str((end - pd.Timedelta(days=365)).date()), str(end.date())]
                row["action"] = "retrained+promoted" if promote else "challenger rejected"
                events.append(ev)
            rows.append(row)
            print(row["month"], row["mae"], row["mae_never_retrain"], row["perf_alert"], row["drifted_features"], row["action"])
    mon = pd.DataFrame(rows)
    mon.to_csv(REPORTS / "monitoring_2012.csv", index=False)
    summary = {
        "rules": {"psi_warn": PSI_WARN, "psi_alert": PSI_ALERT, "perf_factor": PERF_FACTOR, "promote_factor": PROMOTE_FACTOR,
                  "holdout_days": HOLDOUT_DAYS, "monitored_features": MONITORED, "ignored_calendar_features": sorted(CALENDAR)},
        "months": rows, "retrain_events": events,
        "mae_apr_dec_with_policy": round(float(np.average(mon.mae, weights=[len(df[(df.ts >= m.start_time) & (df.ts <= m.end_time)]) for m in MONTHS])), 2),
        "mae_apr_dec_never_retrain": round(float(np.average(mon.mae_never_retrain, weights=[len(df[(df.ts >= m.start_time) & (df.ts <= m.end_time)]) for m in MONTHS])), 2),
        "final_champion_version": version,
    }
    write_json(REPORTS / "monitoring_metrics.json", summary)
    if version != card["registry_version"]:  # deploy the promoted champion to the serving folder
        last = [e for e in events if e.get("promoted")][-1]
        joblib.dump(champion, MODELS / "model.joblib", compress=3)
        card.update({"registry_version": version, "train_window": last["refit_window"],
                     "valid_window": None, "valid_metrics": {"mae": last["challenger_holdout_mae"],
                                                             "note": "14-day hold-out MAE of the challenger before refit"},
                     "promoted_after": last["month"], "previous_version": card["registry_version"]})
        write_json(MODELS / "model_card.json", card)
        write_json(MODELS / "reference_profile.json", reference_profile(ref, champion))
    reg = []
    for mv in client.search_model_versions(f"name='{MODEL_NAME}'"):
        reg.append({"version": int(mv.version), "run_id": mv.run_id, "tags": mv.tags,
                    "aliases": list(mv.aliases), "created": mv.creation_timestamp})
    champ = client.get_model_version_by_alias(MODEL_NAME, "champion")
    for r in reg:
        r["aliases"] = ["champion"] if r["version"] == int(champ.version) else []
        r["created"] = pd.Timestamp(r["created"], unit="ms", tz="UTC").isoformat()
    write_json(REPORTS / "mlflow_registry.json", sorted(reg, key=lambda r: r["version"]))
    fig, ax = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
    ax[0].plot(mon.month, mon.mae_never_retrain, "o--", label="never retrain (v1)")
    ax[0].plot(mon.month, mon.mae, "o-", label="with retraining policy")
    ax[0].plot(mon.month, PERF_FACTOR * mon.reference_mae, ":", color="grey", label="perf alert line")
    for e in events:
        if e["promoted"]:
            ax[0].axvline(e["month"], color="green", alpha=.3)
    ax[0].set_ylabel("MAE (rides / hour)"); ax[0].legend(); ax[0].set_title("Monthly error, April-December 2012")
    for f in ["cnt_lag_168h", "cnt_mean_prev_168h", "temp", "hum", "windspeed"]:
        ax[1].plot(mon.month, mon[f"psi_{f}"], marker=".", label=f)
    ax[1].plot(mon.month, mon.prediction_psi, "k-", lw=2, label="prediction")
    ax[1].axhline(PSI_ALERT, color="red", ls=":"); ax[1].set_yscale("log"); ax[1].set_ylabel("PSI vs reference (log)"); ax[1].legend(fontsize=7, ncol=3)
    plt.xticks(rotation=45); plt.tight_layout(); plt.savefig(REPORTS / "monitoring_2012.png", dpi=120)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("months",)}, indent=1))


if __name__ == "__main__":
    main()
