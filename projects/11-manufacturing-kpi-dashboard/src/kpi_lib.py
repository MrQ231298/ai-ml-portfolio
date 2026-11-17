"""
KPI definitions for the manufacturing dashboard, written once in pandas.

Each function mirrors a DAX measure in `powerbi/measures.dax` so the
Streamlit dashboard and a Power BI report built from the guide show the
same numbers. `src/kpi_sql.py` re-implements the same measures in SQL
(DuckDB) as an independent cross-check.

Definitions (standard OEE, as used by most plants):
    Planned production time = shift time - planned stops
    Availability = run time / planned production time
    Performance  = (ideal cycle time x total count) / run time
    Quality      = first-pass good count / total count
    OEE          = Availability x Performance x Quality
                 = (ideal cycle time x first-pass good) / planned production time
OEE over several shifts or lines is computed from *summed* minutes and
counts (a weighted ratio), never as an average of per-shift OEE values.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STAR = ROOT / "data" / "star_schema"
FACTS = ["fact_production", "fact_downtime", "fact_defects", "fact_target"]
DIMS = ["dim_date", "dim_line", "dim_product", "dim_shift", "dim_downtime_reason", "dim_defect"]


def load_star(folder: Path = STAR) -> dict[str, pd.DataFrame]:
    t = {n: pd.read_csv(folder / f"{n}.csv") for n in FACTS + DIMS}
    t["dim_date"]["date"] = pd.to_datetime(t["dim_date"]["date"])
    for f in FACTS:
        assert (t[f]["data_flag"] == "SYNTHETIC_EXAMPLE_DATA").all(), f"{f}: untagged rows"
    return t


def production_view(t: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Production fact joined to its dimensions plus the target (same grain)."""
    p = (t["fact_production"]
         .merge(t["fact_target"].drop(columns="data_flag"), on=["date_key", "shift_key", "line_key"], how="left")
         .merge(t["dim_date"][["date_key", "date", "iso_week", "month"]], on="date_key")
         .merge(t["dim_line"], on="line_key").merge(t["dim_shift"][["shift_key", "shift"]], on="shift_key")
         .merge(t["dim_product"][["product_key", "product_code"]], on="product_key"))
    p["ideal_minutes_total"] = p["ideal_cycle_time_s"] * p["total_count"] / 60
    p["ideal_minutes_fpg"] = p["ideal_cycle_time_s"] * p["first_pass_good_count"] / 60
    return p


def kpis(p: pd.DataFrame) -> dict[str, float]:
    """All headline measures for any filtered slice of `production_view`."""
    planned = p["planned_production_minutes"].sum()
    run = p["run_minutes"].sum()
    total = p["total_count"].sum()
    fpg = p["first_pass_good_count"].sum()
    good = p["good_count"].sum()
    out = {
        "planned_production_minutes": float(planned),
        "run_minutes": float(run),
        "unplanned_downtime_minutes": float(p["unplanned_downtime_minutes"].sum()),
        "total_count": int(total), "good_count": int(good), "first_pass_good_count": int(fpg),
        "scrap_count": int(p["scrap_count"].sum()), "rework_count": int(p["rework_count"].sum()),
        "target_units": int(p["target_units"].sum()),
        "availability": run / planned if planned else float("nan"),
        "performance": p["ideal_minutes_total"].sum() / run if run else float("nan"),
        "quality": fpg / total if total else float("nan"),
        "fpy": fpg / total if total else float("nan"),
        "yield_after_rework": good / total if total else float("nan"),
        "scrap_rate": p["scrap_count"].sum() / total if total else float("nan"),
        "output_vs_target": good / p["target_units"].sum() if p["target_units"].sum() else float("nan"),
    }
    out["oee"] = out["availability"] * out["performance"] * out["quality"]
    out["oee_direct"] = p["ideal_minutes_fpg"].sum() / planned if planned else float("nan")
    return out


def kpis_by(p: pd.DataFrame, by: list[str] | str) -> pd.DataFrame:
    rows = []
    for key, g in p.groupby(by, sort=True):
        r = kpis(g)
        key = key if isinstance(key, tuple) else (key,)
        r.update(dict(zip([by] if isinstance(by, str) else by, key)))
        rows.append(r)
    return pd.DataFrame(rows)


def rty(p: pd.DataFrame) -> float:
    """Rolled throughput yield along the route: product of each step's FPY.
    The two SMT lines are parallel, so step 1 is their combined FPY."""
    steps = p.groupby("route_step")[["first_pass_good_count", "total_count"]].sum()
    out = 1.0
    for _, r in steps.iterrows():
        out *= r["first_pass_good_count"] / r["total_count"]
    return float(out)


def downtime_pareto(t: dict[str, pd.DataFrame], line_keys=None, date_keys=None, shift_keys=None) -> pd.DataFrame:
    d = t["fact_downtime"]
    if line_keys is not None:
        d = d[d["line_key"].isin(line_keys)]
    if date_keys is not None:
        d = d[d["date_key"].isin(date_keys)]
    if shift_keys is not None:
        d = d[d["shift_key"].isin(shift_keys)]
    g = (d.merge(t["dim_downtime_reason"], on="reason_key")
         .groupby(["reason", "reason_category"], as_index=False)
         .agg(minutes=("duration_minutes", "sum"), events=("event_id", "count"))
         .sort_values("minutes", ascending=False, ignore_index=True))
    g["share"] = g["minutes"] / g["minutes"].sum()
    g["cumulative_share"] = g["share"].cumsum()
    return g


def oee_losses_minutes(p: pd.DataFrame) -> dict[str, float]:
    """Planned production time split into OEE time and the three losses (minutes)."""
    planned = p["planned_production_minutes"].sum()
    run = p["run_minutes"].sum()
    ideal_total = p["ideal_minutes_total"].sum()
    ideal_fpg = p["ideal_minutes_fpg"].sum()
    return {"planned_production": float(planned),
            "availability_loss": float(planned - run),
            "performance_loss": float(run - ideal_total),
            "quality_loss": float(ideal_total - ideal_fpg),
            "fully_productive": float(ideal_fpg)}
