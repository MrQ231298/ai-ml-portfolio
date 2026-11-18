"""Full recorded experiment for FD001 (headline) and FD003 (two fault modes): writes reports/metrics.json."""

from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import alerting as al
import rul as R
from anomaly import CONSECUTIVE, HEALTHY_CYCLES, HealthModel, first_alarm
from data import ROOT, SEED, load, split_units
from features import build, informative_sensors

REPORTS = ROOT / "reports"
BASE = al.Costs()
REPEAT_SEEDS = [21, 22, 23, 24, 25]  # the headline split (21) plus 4 more random engine splits


def anomaly_block(hm, F, df, splits):
    out = {}
    healthy = (df.cycle <= HEALTHY_CYCLES).to_numpy()
    degraded = (df.rul <= 30).to_numpy()
    for m in hm.methods:
        s = hm.score(F, m)
        res = {}
        for name in ("val", "test"):
            u = df.unit.isin(splits[name]).to_numpy()
            mask = u & (healthy | degraded)
            res[f"{name}_auroc_healthy_vs_last30"] = round(roc_auc_score(degraded[mask], s[mask]), 4)
        thr = float(np.quantile(s[df.unit.isin(splits["val"]).to_numpy() & healthy], 0.995))
        lead, early, missed = [], 0, 0
        for unit in splits["test"]:
            idx = np.where(df.unit.to_numpy() == unit)[0]
            i = first_alarm(s[idx], thr)
            if i is None:
                missed += 1
                continue
            if df.cycle.iloc[idx[i]] <= HEALTHY_CYCLES:
                early += 1
            lead.append(int(df.rul.iloc[idx[i]]))
        res.update({"threshold_val_healthy_q995": round(thr, 4), "consecutive": CONSECUTIVE,
                    "test_units_alarmed": len(lead), "test_units_never_alarmed": missed,
                    "test_false_alarms_in_first_30_cycles": early,
                    "test_lead_time_cycles": {"median": float(np.median(lead)) if lead else None,
                                              "min": int(min(lead)) if lead else None, "max": int(max(lead)) if lead else None}})
        out[m] = res
    return out


def per_unit(df, signal, units):
    return {u: (signal[(df.unit == u).to_numpy()], df.rul[df.unit == u].to_numpy()) for u in units}


def policies(df, pred, health, splits, c):
    """Choose each policy's parameter on validation engines by cost, report on held-out test engines."""
    val_rul, test_rul = per_unit(df, pred, splits["val"]), per_unit(df, pred, splits["test"])
    val_h, test_h = per_unit(df, health, splits["val"]), per_unit(df, health, splits["test"])
    lives = df.groupby("unit").cycle.max()
    out, curves = {}, {}
    out["run_to_failure"] = al.evaluate([None] * len(splits["test"]), c)
    Ts = range(100, 400, 2)
    vcost = [al.evaluate(al.fixed_interval_alerts(lives[splits["val"]].to_dict(), T, c), c)["mean_cost"] for T in Ts]
    T = list(Ts)[int(np.argmin(vcost))]
    out["fixed_interval"] = {"maintenance_cycle": T, **al.evaluate(al.fixed_interval_alerts(lives[splits["test"]].to_dict(), T, c), c)}
    hq = np.quantile(np.concatenate([s for s, _ in val_h.values()]), np.linspace(0.5, 0.999, 120))
    th, curves["health_score"] = al.choose_threshold(val_h, hq, below=False, c=c, k=CONSECUTIVE)
    out["health_score_alert"] = {"threshold": round(th, 4), **al.evaluate(
        [al.alert_from_threshold(s, r, th, False, CONSECUTIVE) for s, r in test_h.values()], c)}
    th, curves["predicted_rul"] = al.choose_threshold(val_rul, range(0, 126), below=True, c=c)
    out["predicted_rul_alert"] = {"alert_when_predicted_rul_le": th, **al.evaluate(
        [al.alert_from_threshold(s, r, th, True) for s, r in test_rul.values()], c)}
    out["oracle"] = al.evaluate([float(c.lead_time)] * len(splits["test"]), c)
    return out, curves


def run(fd: str, seed: int = SEED) -> tuple[dict, pd.DataFrame]:
    t0 = time.time()
    d = load(fd)
    df = d["train"].sort_values(["unit", "cycle"]).reset_index(drop=True)
    splits = split_units(df.unit.unique(), seed)
    fit_mask = df.unit.isin(splits["fit"]).to_numpy()
    sensors = informative_sensors(df[fit_mask])
    F = build(df, sensors)
    y = R.capped(df.rul.to_numpy())
    res = {"engines": {k: len(v) for k, v in splits.items()}, "rows": int(len(df)), "sensors_used": sensors,
           "n_features": int(F.shape[1]), "split_units": splits,
           "engine_life_cycles": {k: round(float(v), 1) for k, v in df.groupby("unit").cycle.max().describe()[["mean", "min", "50%", "max"]].items()}}

    hm = HealthModel(sensors).fit(F, fit_mask & (df.cycle <= HEALTHY_CYCLES).to_numpy())
    res["anomaly"] = anomaly_block(hm, F, df, splits)
    best_anom = max(hm.methods, key=lambda m: res["anomaly"][m]["val_auroc_healthy_vs_last30"])
    res["anomaly_selected_on_val"] = best_anom

    # ---- RUL models: select on validation engines, score on held-out engines and the official test set
    m = {u: df.unit.isin(splits[u]).to_numpy() for u in ("fit", "val", "test")}
    median_life = float(df[m["fit"]].groupby("unit").cycle.max().median())
    cands = {"baseline_mean": lambda: R.MeanBaseline(), "baseline_fleet_life": lambda: R.FleetLifeBaseline(median_life)}
    cands |= {f"ridge_a{g['alpha']}": (lambda g=g: R.ridge(**g)) for g in R.RIDGE_GRID}
    cands |= {f"hgb_lr{g['lr']}_l{g['leaves']}": (lambda g=g: R.hgb(**g)) for g in R.HGB_GRID}
    fitted, val_rmse = {}, {}
    for name, mk in cands.items():
        fitted[name] = mk().fit(F[m["fit"]], y[m["fit"]])
        val_rmse[name] = round(R.rmse(y[m["val"]], fitted[name].predict(F[m["val"]])), 3)
    best = {"ridge": min((k for k in val_rmse if k.startswith("ridge")), key=val_rmse.get),
            "hgb": min((k for k in val_rmse if k.startswith("hgb")), key=val_rmse.get)}
    keep = ["baseline_mean", "baseline_fleet_life", best["ridge"], best["hgb"]]
    late = m["test"] & (df.rul <= 50).to_numpy()
    rul_res = {"val_rmse_all_candidates": val_rmse, "selected": best, "models": {}}
    test_df = d["test"].sort_values(["unit", "cycle"]).reset_index(drop=True)
    Ft = build(test_df, sensors)
    last = test_df.groupby("unit").cycle.idxmax().sort_index().to_numpy()
    fv = m["fit"] | m["val"]
    med_fv = float(df[fv].groupby("unit").cycle.max().median())
    for name in keep:
        p_test = fitted[name].predict(F[m["test"]])
        refit = (R.FleetLifeBaseline(med_fv) if name == "baseline_fleet_life" else cands[name]()).fit(F[fv], y[fv])
        p_off = np.clip(refit.predict(Ft.iloc[last]), 0, None)
        if name == best["hgb"] and seed == SEED:
            pd.DataFrame({"unit": test_df.unit.iloc[last].to_numpy(), "true_rul": d["test_rul"],
                          "pred_rul": p_off.round(1)}).to_csv(REPORTS / f"official_test_predictions_{fd}.csv", index=False)
        rul_res["models"][name] = {
            "val_rmse": val_rmse[name],
            "heldout_rmse_all_cycles_capped": round(R.rmse(y[m["test"]], p_test), 3),
            "heldout_rmse_last50_cycles": round(R.rmse(df.rul[late], fitted[name].predict(F[late])), 3),
            "official_test_rmse_vs_true_rul": round(R.rmse(d["test_rul"], p_off), 3),
            "official_test_rmse_vs_capped_rul": round(R.rmse(R.capped(d["test_rul"]), p_off), 3),
            "official_test_nasa_score": round(R.nasa_score(d["test_rul"], p_off), 1)}
    res["rul"] = rul_res
    res["official_test_engines"] = int(len(last))

    # ---- maintenance policies, base costs + sensitivity to the failure cost
    pred = np.clip(fitted[best["hgb"]].predict(F), 0, None)   # val/test engines are out-of-sample for this model
    health = hm.score(F, best_anom)
    res["costs_assumed"] = BASE.__dict__
    res["policies"], curves = policies(df, pred, health, splits, BASE)
    sens = {}
    for cf in (5.0, 10.0, 20.0, 50.0):
        c = al.Costs(c_fail=cf)
        p, _ = policies(df, pred, health, splits, c)
        sens[f"c_fail_{int(cf)}"] = {k: {kk: v[kk] for kk in v if kk not in ("engines", "mean_wasted_cycles")}
                                     for k, v in p.items()}
    res["policy_sensitivity_test"] = sens
    res["seconds"] = round(time.time() - t0, 1)
    trace = df[["unit", "cycle", "rul"]].assign(pred_rul=pred.round(1), health_score=health.round(4),
                                                split=np.select([m["fit"], m["val"]], ["fit", "val"], "test"))
    trace = pd.concat([trace, df[sensors[:4]]], axis=1)
    res["_curves"] = curves
    return res, trace


def main():
    fds = sys.argv[1:] or ["FD001", "FD003"]
    out = {"data": "NASA C-MAPSS turbofan run-to-failure simulation (Saxena et al., 2008), public domain",
           "seed": SEED}
    for fd in fds:
        res, trace = run(fd)
        curves = res.pop("_curves")
        (REPORTS / f"cost_curves_{fd}.json").write_text(json.dumps(curves))
        trace[trace.split != "fit"].to_csv(REPORTS / f"engine_traces_{fd}.csv", index=False)
        out[fd] = res
        reps = {"seeds": [], "hgb_heldout_rmse": [], "policy_mean_cost": {}, "policy_failures": {}}
        for sd in REPEAT_SEEDS:
            r, _ = run(fd, sd)
            reps["seeds"].append(sd)
            reps["hgb_heldout_rmse"].append(r["rul"]["models"][r["rul"]["selected"]["hgb"]]["heldout_rmse_all_cycles_capped"])
            for k, v in r["policies"].items():
                reps["policy_mean_cost"].setdefault(k, []).append(v["mean_cost"])
                reps["policy_failures"].setdefault(k, []).append(v["failures"])
        reps["summary_mean_cost"] = {k: {"mean": round(float(np.mean(v)), 3), "sd": round(float(np.std(v, ddof=1)), 3)}
                                     for k, v in reps["policy_mean_cost"].items()}
        out[fd]["repeated_engine_splits"] = reps
        print(fd, "repeats", json.dumps(reps["summary_mean_cost"]), reps["hgb_heldout_rmse"], flush=True)
        print(fd, json.dumps({k: res[k] for k in ("anomaly_selected_on_val", "policies")}, indent=1), flush=True)
        print(json.dumps(res["rul"]["models"], indent=1), flush=True)
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "metrics.json").write_text(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
