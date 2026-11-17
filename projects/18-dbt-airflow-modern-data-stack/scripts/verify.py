"""Checks run after the Airflow backfill, written to reports/metrics.json.

1. Row accounting per month (raw -> valid / rejected by rule) and mart sizes.
2. Idempotency: the warehouse fingerprint before and after re-running January.
3. Incremental == full refresh: a copy of the warehouse rebuilt with
   `dbt build --full-refresh` must give identical fct_trips and DQ tables.
4. Quality gate: a copy with one planted bad row (unknown zone 999, negative
   fare) must make `dbt build` fail, naming the tests that caught it.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
DBT_DIR = ROOT / "nyc_taxi"
WH = ROOT / "warehouse" / "nyc_taxi.duckdb"
DBT = os.environ.get("DBT_BIN", "dbt")
AIRFLOW = os.environ.get("AIRFLOW_BIN", "airflow")

FINGERPRINT_SQL = {
    "fct_trips": "select count(*), md5(string_agg(trip_key || '|' || total_amount::varchar, ',' order by trip_key)) from marts.fct_trips",
    "dq_rejections_by_month": "select count(*), md5(string_agg(source_month || outcome || row_count, ',' order by source_month, outcome)) from marts.dq_rejections_by_month",
    "agg_daily_borough": "select count(*), md5(string_agg(trip_date::varchar || coalesce(pickup_borough,'') || trips || revenue, ',' order by trip_date, pickup_borough)) from marts.agg_daily_borough",
}


def fingerprint(db: Path) -> dict:
    con = duckdb.connect(str(db), read_only=True)
    out = {k: list(con.execute(q).fetchone()) for k, q in FINGERPRINT_SQL.items()}
    con.close()
    return out


def dbt(args: list[str], db: Path) -> tuple[int, dict, float]:
    t0 = time.time()
    p = subprocess.run([DBT, *args, "--profiles-dir", ".", "--project-dir", ".", "--target-path", f"target_verify"],
                       cwd=DBT_DIR, env={**os.environ, "DBT_DUCKDB_PATH": str(db)}, capture_output=True, text=True)
    rr = json.loads((DBT_DIR / "target_verify" / "run_results.json").read_text())
    return p.returncode, rr, time.time() - t0


def accounting(db: Path) -> dict:
    con = duckdb.connect(str(db), read_only=True)
    raw = dict(con.execute("select source_month, count(*) from raw.yellow_trips group by 1 order by 1").fetchall())
    dq = con.execute("select source_month, outcome, row_count, pct_of_month from marts.dq_rejections_by_month order by 1, 3 desc").fetchall()
    sizes = {t: con.execute(f"select count(*) from marts.{t}").fetchone()[0]
             for t in ("fct_trips", "dim_zones", "agg_daily_borough", "agg_hour_of_week", "dq_rejections_by_month")}
    top = con.execute("""select pickup_borough, sum(trips) trips, round(sum(revenue), 2) revenue,
                         round(sum(revenue) / sum(trips), 2) rev_per_trip from marts.agg_daily_borough
                         group by 1 order by 2 desc limit 6""").fetchall()
    peak = con.execute("select iso_weekday, hour_of_day, avg_trips from marts.agg_hour_of_week order by avg_trips desc limit 3").fetchall()
    con.close()
    out = {"raw_rows": raw, "outcomes": {}, "mart_rows": sizes,
           "borough_totals": [dict(zip(["borough", "trips", "revenue", "revenue_per_trip"], r)) for r in top],
           "peak_hours": [dict(zip(["iso_weekday", "hour", "avg_trips"], r)) for r in peak]}
    for m, o, n, pct in dq:
        out["outcomes"].setdefault(m, {})[o] = {"rows": n, "pct": float(pct)}
    return out


def main():
    res = {"accounting": accounting(WH), "fingerprint_after_backfill": fingerprint(WH)}
    gates = sorted((ROOT / "reports" / "airflow_runs").glob("gate_*.json"))
    res["airflow_quality_gates"] = [json.loads(g.read_text()) for g in gates]

    # 2. idempotency: re-run January through Airflow
    env = {**os.environ}
    t0 = time.time()
    p = subprocess.run([AIRFLOW, "dags", "test", "nyc_taxi_monthly", "2024-01-01"], env=env, capture_output=True, text=True)
    res["rerun_january"] = {"airflow_exit_code": p.returncode, "seconds": round(time.time() - t0, 1),
                            "fingerprint": fingerprint(WH)}
    res["rerun_january"]["identical"] = res["rerun_january"]["fingerprint"] == res["fingerprint_after_backfill"]
    rl = [l for l in (p.stdout + p.stderr).splitlines() if '"rows_replaced"' in l]
    res["rerun_january"]["load_log"] = rl[-1][rl[-1].find("{"):rl[-1].find("}") + 1] if rl else None

    tmp = Path(tempfile.mkdtemp())
    try:
        # 3. incremental vs full refresh
        full = tmp / "full.duckdb"
        shutil.copy(WH, full)
        rc, rr, secs = dbt(["build", "--full-refresh"], full)
        res["full_refresh"] = {"exit_code": rc, "seconds": round(secs, 1), "fingerprint": fingerprint(full),
                               "nodes": len(rr["results"])}
        res["full_refresh"]["identical_to_incremental"] = res["full_refresh"]["fingerprint"] == res["rerun_january"]["fingerprint"]
        # 4. planted bad row must fail the build
        bad = tmp / "bad.duckdb"
        shutil.copy(WH, bad)
        con = duckdb.connect(str(bad))
        con.execute("""insert into raw.yellow_trips by name
                       select * replace (999 as PULocationID, -12.5 as fare_amount, -3.0 as total_amount,
                                         '2024-02' as source_month, now()::timestamp as _loaded_at)
                       from raw.yellow_trips where source_month = '2024-02' limit 1""")
        con.execute("""insert into raw.yellow_trips by name
                       select * replace (998 as PULocationID, 40.0 as fare_amount, 52.0 as total_amount,
                                         timestamp '2024-02-10 10:00:00' as tpep_pickup_datetime,
                                         timestamp '2024-02-10 10:30:00' as tpep_dropoff_datetime,
                                         3.1 as trip_distance, 1 as passenger_count,
                                         '2024-02' as source_month, now()::timestamp as _loaded_at)
                       from raw.yellow_trips where source_month = '2024-02' limit 1""")
        con.close()
        rc, rr, secs = dbt(["build", "--vars", '{"month": "2024-02"}', "--exclude", "resource_type:seed"], bad)
        res["planted_bad_rows"] = {
            "rows": ["zone 999 with negative fare (should be rejected by the negative_amount rule)",
                     "zone 998 with a valid fare (passes DQ rules, must be caught by the relationships test)"],
            "dbt_exit_code": rc,
            "failed_tests": [r["unique_id"] for r in rr["results"] if r["status"] in ("fail", "error")],
            "skipped_downstream": [r["unique_id"] for r in rr["results"] if r["status"] == "skipped"]}
    finally:
        shutil.rmtree(tmp)
        shutil.rmtree(DBT_DIR / "target_verify", ignore_errors=True)
    (ROOT / "reports" / "metrics.json").write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps({k: res[k] for k in ("rerun_january", "full_refresh", "planted_bad_rows")}, indent=1, default=str))


if __name__ == "__main__":
    main()
