"""
Compute the dashboard KPIs, cross-check them against an independent SQL
implementation, and test whether simple, generic monitoring rules find
the problems planted in the SYNTHETIC data (and how many false alarms
they raise).

    python src/analyse.py      -> reports/metrics.json, reports/*.png, reports/*.csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kpi_lib as K  # noqa: E402
from kpi_sql import kpi_query  # noqa: E402

ROOT = K.ROOT
REPORTS = ROOT / "reports"
BASELINE_END = 20260831       # Phase I (limits) = Jul-Aug; Phase II (monitor) = Sep-Oct
ALPHA = 0.01
RNG = np.random.default_rng(7)


def cross_check(p: pd.DataFrame) -> dict:
    py_all = K.kpis(p)
    sql_all = kpi_query().iloc[0]
    diffs = [abs(py_all[m] - sql_all[m]) for m in ["availability", "performance", "quality", "output_vs_target"]]
    diffs.append(abs(py_all["oee_direct"] - sql_all["oee"]))
    py_line = K.kpis_by(p, "line").set_index("line")
    sql_line = kpi_query("line").set_index("line")
    for m in ["availability", "performance", "quality", "output_vs_target"]:
        diffs += list((py_line[m] - sql_line[m]).abs())
    diffs += list((py_line["oee_direct"] - sql_line["oee"]).abs())
    return {"values_compared": len(diffs), "max_abs_difference": float(max(diffs))}


def aggregation_pitfall(p: pd.DataFrame) -> dict:
    per_shift = p.assign(oee=p["run_minutes"] / p["planned_production_minutes"]
                         * p["ideal_minutes_total"] / p["run_minutes"]
                         * p["first_pass_good_count"] / p["total_count"])
    return {"weighted_oee": K.kpis(p)["oee"], "mean_of_shift_oee": float(per_shift["oee"].mean()),
            "a_x_p_x_q_vs_direct_gap": K.kpis(p)["oee"] - K.kpis(p)["oee_direct"]}


def shift_performance_test(p: pd.DataFrame) -> dict:
    """Night vs day shifts performance gap, bootstrap over production days."""
    days = p["date_key"].unique()
    def gap(sub):
        night = sub[sub["shift"] == "Night"]; day = sub[sub["shift"] != "Night"]
        return K.kpis(night)["performance"] - K.kpis(day)["performance"]
    point = gap(p)
    by_day = {d: g for d, g in p.groupby("date_key")}
    boots = []
    for _ in range(1000):
        pick = RNG.choice(days, size=len(days), replace=True)
        boots.append(gap(pd.concat([by_day[d] for d in pick])))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"night_minus_day_performance": float(point), "ci95": [float(lo), float(hi)], "bootstrap_reps": 1000}


def weekly_pchart(t: dict, p: pd.DataFrame) -> pd.DataFrame:
    """Weekly defect proportion per line x defect; limits from Jul-Aug only."""
    weeks = t["dim_date"][["date_key", "iso_week"]]
    tot = p.groupby(["line", "iso_week"], as_index=False)["total_count"].sum()
    d = (t["fact_defects"].merge(weeks, on="date_key").merge(t["dim_line"][["line_key", "line"]], on="line_key")
         .merge(t["dim_defect"][["defect_key", "defect"]], on="defect_key")
         .groupby(["line", "defect", "iso_week"], as_index=False)["defect_count"].sum())
    combos = d[["line", "defect"]].drop_duplicates()
    grid = combos.merge(tot, on="line").merge(d, on=["line", "defect", "iso_week"], how="left").fillna({"defect_count": 0})
    first_monitor_week = int(t["dim_date"].loc[t["dim_date"]["date_key"] > BASELINE_END, "iso_week"].min())
    rows = []
    for (line, defect), g in grid.groupby(["line", "defect"]):
        base = g[g["iso_week"] < first_monitor_week]
        pbar = base["defect_count"].sum() / base["total_count"].sum()
        for _, r in g.iterrows():
            n = r["total_count"]; sigma = np.sqrt(pbar * (1 - pbar) / n)
            ucl = pbar + 3 * sigma
            rows.append({"line": line, "defect": defect, "iso_week": int(r["iso_week"]),
                         "phase": "monitor" if r["iso_week"] >= first_monitor_week else "baseline",
                         "rate": r["defect_count"] / n, "centre": pbar, "ucl": ucl,
                         "signal": bool(r["defect_count"] / n > ucl)})
    return pd.DataFrame(rows)


def weekly_downtime_chart(t: dict) -> pd.DataFrame:
    """Weekly downtime-event counts per line x reason vs a Poisson upper limit
    (0.99865 quantile, the one-sided 3-sigma equivalent) from Jul-Aug, scaled by shifts worked."""
    weeks = t["dim_date"][["date_key", "iso_week"]]
    shifts = (t["fact_production"].merge(weeks, on="date_key")
              .groupby(["line_key", "iso_week"], as_index=False).size().rename(columns={"size": "shifts"}))
    ev = (t["fact_downtime"].merge(weeks, on="date_key")
          .groupby(["line_key", "reason_key", "iso_week"], as_index=False).size().rename(columns={"size": "events"}))
    combos = ev[["line_key", "reason_key"]].drop_duplicates()
    grid = combos.merge(shifts, on="line_key").merge(ev, on=["line_key", "reason_key", "iso_week"], how="left").fillna({"events": 0})
    grid = grid.merge(t["dim_line"][["line_key", "line"]], on="line_key").merge(t["dim_downtime_reason"][["reason_key", "reason"]], on="reason_key")
    first_monitor_week = int(t["dim_date"].loc[t["dim_date"]["date_key"] > BASELINE_END, "iso_week"].min())
    rows = []
    for (line, reason), g in grid.groupby(["line", "reason"]):
        base = g[g["iso_week"] < first_monitor_week]
        rate = base["events"].sum() / base["shifts"].sum()
        if rate * g["shifts"].median() < 0.5:
            continue  # too rare to chart weekly
        for _, r in g.iterrows():
            ucl = stats.poisson.ppf(0.99865, rate * r["shifts"])
            rows.append({"line": line, "reason": reason, "iso_week": int(r["iso_week"]),
                         "phase": "monitor" if r["iso_week"] >= first_monitor_week else "baseline",
                         "events": int(r["events"]), "expected": rate * r["shifts"], "ucl": float(ucl),
                         "signal": bool(r["events"] > ucl)})
    return pd.DataFrame(rows)


def line_reason_scan(t: dict) -> pd.DataFrame:
    """Is any line's event rate for a reason out of line with the other lines
    that share the failure mode? Conditional binomial test, Bonferroni-corrected."""
    shifts = t["fact_production"].groupby("line_key").size()
    ev = t["fact_downtime"].groupby(["reason_key", "line_key"]).size().unstack(fill_value=0)
    rows = []
    for rk, r in ev.iterrows():
        lines = [lk for lk in r.index if r[lk] > 0]
        for lk in lines:
            others = [o for o in lines if o != lk]
            if not others:
                continue
            a, b = int(r[lk]), int(r[others].sum())
            ea, eb = shifts[lk], shifts[others].sum()
            pval = stats.binomtest(a, a + b, ea / (ea + eb), alternative="greater").pvalue
            rows.append({"reason_key": rk, "line_key": lk, "events": a, "rate_per_shift": a / ea,
                         "others_rate_per_shift": b / eb, "rate_ratio": (a / ea) / (b / eb), "p_value": pval})
    df = pd.DataFrame(rows)
    df["p_bonferroni"] = np.minimum(df["p_value"] * len(df), 1.0)
    df["signal"] = df["p_bonferroni"] < ALPHA
    return (df.merge(t["dim_line"][["line_key", "line"]], on="line_key")
            .merge(t["dim_downtime_reason"][["reason_key", "reason"]], on="reason_key")
            .drop(columns=["line_key", "reason_key"]).sort_values("p_value", ignore_index=True))


def plots(t, p, pareto, pch, dch):
    REPORTS.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(pareto["reason"], pareto["minutes"] / 60, color="#3b6ea5")
    ax2 = ax.twinx(); ax2.plot(pareto["reason"], pareto["cumulative_share"] * 100, color="#d1495b", marker="o")
    ax2.axhline(80, ls="--", color="grey", lw=0.8); ax2.set_ylim(0, 105); ax2.set_ylabel("cumulative %")
    ax.set_ylabel("unplanned downtime (hours)"); ax.set_title("Downtime Pareto, all lines, Jul-Oct 2026 (SYNTHETIC)")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right"); fig.tight_layout(); fig.savefig(REPORTS / "downtime_pareto.png", dpi=120); plt.close(fig)

    wk = K.kpis_by(p, ["iso_week", "line"])
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for line, g in wk.groupby("line"):
        ax.plot(g["iso_week"], g["oee"] * 100, marker=".", label=line)
    ax.axhline(85, ls="--", color="grey", lw=0.8, label="85% reference")
    ax.set_xlabel("ISO week (2026)"); ax.set_ylabel("OEE %"); ax.set_title("Weekly OEE by line (SYNTHETIC)"); ax.legend(ncol=5, fontsize=8)
    fig.tight_layout(); fig.savefig(REPORTS / "oee_weekly_by_line.png", dpi=120); plt.close(fig)

    loss = K.oee_losses_minutes(p)
    labels = ["Availability loss", "Performance loss", "Quality loss", "Fully productive"]
    vals = [loss["availability_loss"], loss["performance_loss"], loss["quality_loss"], loss["fully_productive"]]
    fig, ax = plt.subplots(figsize=(8, 4))
    left = 0
    for lab, v, c in zip(labels, vals, ["#d1495b", "#edae49", "#8e6c8a", "#00798c"]):
        ax.barh([0], [v / 60], left=left / 60, color=c, label=f"{lab} ({v / loss['planned_production'] * 100:.1f}%)"); left += v
    ax.set_yticks([]); ax.set_xlabel("hours of planned production time"); ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2)
    ax.set_title("Where planned production time went (SYNTHETIC)"); fig.tight_layout(); fig.savefig(REPORTS / "oee_loss_breakdown.png", dpi=120); plt.close(fig)

    g = pch[(pch["line"] == "SMT-1") & (pch["defect"] == "Solder bridge")]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(g["iso_week"], g["rate"] * 100, marker="o", label="weekly rate")
    ax.plot(g["iso_week"], g["ucl"] * 100, ls="--", color="#d1495b", label="UCL (3σ, Jul-Aug baseline)")
    ax.axhline(g["centre"].iloc[0] * 100, color="grey", lw=0.8, label="baseline centre")
    sig = g[g["signal"] & (g["phase"] == "monitor")]
    ax.scatter(sig["iso_week"], sig["rate"] * 100, color="#d1495b", s=80, zorder=3, label="signal")
    ax.set_xlabel("ISO week (2026)"); ax.set_ylabel("% of units"); ax.set_title("p-chart: solder bridges on SMT-1 (SYNTHETIC)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(REPORTS / "pchart_smt1_solder_bridge.png", dpi=120); plt.close(fig)


def main() -> None:
    t = K.load_star()
    p = K.production_view(t)
    planted = json.loads((K.STAR / "planted_issues.json").read_text())["issues"]
    head = K.kpis(p)
    pareto = K.downtime_pareto(t)
    pch = weekly_pchart(t, p)
    dch = weekly_downtime_chart(t)
    scan = line_reason_scan(t)
    REPORTS.mkdir(exist_ok=True)
    pch.to_csv(REPORTS / "pchart_weekly.csv", index=False)
    dch.to_csv(REPORTS / "downtime_chart_weekly.csv", index=False)
    scan.to_csv(REPORTS / "line_reason_scan.csv", index=False)
    pareto.to_csv(REPORTS / "downtime_pareto.csv", index=False)

    # --- score the monitoring rules against the planted issues (monitor phase only)
    wk = t["dim_date"][["date_key", "iso_week"]]
    p1_weeks = set(wk[(wk["date_key"] >= 20260907) & (wk["date_key"] <= 20260920)]["iso_week"])
    p3_weeks = set(wk[(wk["date_key"] >= 20261001)]["iso_week"])
    pm = pch[pch["phase"] == "monitor"]; dm = dch[dch["phase"] == "monitor"]
    p_hits = pm[pm["signal"]]
    p1_true = p_hits[(p_hits["line"] == "SMT-1") & (p_hits["defect"] == "Solder bridge") & p_hits["iso_week"].isin(p1_weeks)]
    d_hits = dm[dm["signal"]]
    p3_true = d_hits[(d_hits["line"] == "EOL-1") & (d_hits["reason"] == "Tester / fixture fault") & d_hits["iso_week"].isin(p3_weeks)]
    p3_cells = dm[(dm["line"] == "EOL-1") & (dm["reason"] == "Tester / fixture fault") & dm["iso_week"].isin(p3_weeks)]
    scan_sig = scan[scan["signal"]]
    detection = {
        "P1_solder_spike": {"planted_weeks": sorted(int(w) for w in p1_weeks), "weeks_flagged": sorted(int(w) for w in p1_true["iso_week"]),
                            "recall_weeks": len(p1_true) / len(p1_weeks)},
        "P3_fixture_faults": {"planted_weeks_monitored": int(len(p3_cells)), "weeks_flagged": sorted(int(w) for w in p3_true["iso_week"]),
                              "recall_weeks": len(p3_true) / max(len(p3_cells), 1)},
        "pchart_false_alarms": {"signals": int(len(p_hits) - len(p1_true)), "cells_monitored": int(len(pm) - len(p1_weeks))},
        "downtime_chart_false_alarms": {"signals": int(len(d_hits) - len(p3_true)), "cells_monitored": int(len(dm) - len(p3_cells))},
        "P2_line_reason_scan": {"signals": scan_sig[["line", "reason", "rate_ratio", "p_bonferroni"]].to_dict("records"),
                                "found_planted": bool(((scan_sig["line"] == "SMT-2") & (scan_sig["reason"] == "Changeover overrun")).any())},
        "P4_night_shift": shift_performance_test(p),
    }
    metrics = {
        "data_flag": "SYNTHETIC_EXAMPLE_DATA",
        "period": "2026-07-01 to 2026-10-31",
        "rows": {n: int(len(t[n])) for n in K.FACTS + K.DIMS},
        "headline": {k: (round(v, 6) if isinstance(v, float) else v) for k, v in head.items()},
        "rty": K.rty(p),
        "by_line": K.kpis_by(p, "line")[["line", "availability", "performance", "quality", "oee", "output_vs_target"]].round(6).to_dict("records"),
        "by_shift": K.kpis_by(p, "shift")[["shift", "availability", "performance", "quality", "oee"]].round(6).to_dict("records"),
        "downtime_pareto_top3": pareto.head(3)[["reason", "minutes", "share", "cumulative_share"]].round(4).to_dict("records"),
        "oee_losses_minutes": K.oee_losses_minutes(p),
        "aggregation_pitfall": aggregation_pitfall(p),
        "python_vs_sql_cross_check": cross_check(p),
        "detection": detection,
        "planted_issues": planted,
    }
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    plots(t, p, pareto, pch, dch)
    print(json.dumps({k: metrics[k] for k in ["headline", "rty", "aggregation_pitfall", "python_vs_sql_cross_check", "detection"]}, indent=1, default=float))


if __name__ == "__main__":
    main()
