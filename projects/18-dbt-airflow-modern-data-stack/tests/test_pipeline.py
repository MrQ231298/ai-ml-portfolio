"""Loader idempotency and dbt data-quality rules on a tiny hand-made warehouse.

Needs the dbt virtual environment (DBT_BIN, default `dbt`). No download needed.
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ingest"))
from load_month import load  # noqa: E402

DBT = os.environ.get("DBT_BIN", str(Path(sys.executable).parent / "dbt"))


def trip(pickup, dropoff, dist=2.0, fare=12.0, total=16.0, pu=161, do=236, pax=1, pay=1):
    return {"VendorID": 2, "tpep_pickup_datetime": pd.Timestamp(pickup), "tpep_dropoff_datetime": pd.Timestamp(dropoff),
            "passenger_count": float(pax), "trip_distance": dist, "RatecodeID": 1.0, "store_and_fwd_flag": "N",
            "PULocationID": pu, "DOLocationID": do, "payment_type": pay, "fare_amount": fare, "extra": 1.0,
            "mta_tax": 0.5, "tip_amount": 2.0, "tolls_amount": 0.0, "improvement_surcharge": 1.0,
            "total_amount": total, "congestion_surcharge": 2.5, "Airport_fee": 0.0}


ROWS = [
    trip("2024-03-05 08:00", "2024-03-05 08:20"),                       # valid
    trip("2024-03-05 08:00", "2024-03-05 08:20"),                       # exact duplicate
    trip("2024-02-28 23:50", "2024-03-01 00:10"),                       # pickup outside file month
    trip("2024-03-05 09:00", "2024-03-05 08:59"),                       # negative duration
    trip("2024-03-05 10:00", "2024-03-05 10:15", dist=0.0),             # zero distance
    trip("2024-03-05 11:00", "2024-03-05 11:15", fare=-5, total=-2),    # negative amount
    trip("2024-03-05 12:00", "2024-03-05 12:15", pax=0),                # zero passengers
    trip("2024-03-06 18:00", "2024-03-06 18:40", fare=40, total=55),    # valid
]


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    d = tmp_path_factory.mktemp("wh")
    pq = d / "yellow_tripdata_2024-03.parquet"
    pd.DataFrame(ROWS).to_parquet(pq, index=False)
    db = d / "test.duckdb"
    first = load("2024-03", pq, db)
    second = load("2024-03", pq, db)
    env = {**os.environ, "DBT_DUCKDB_PATH": str(db)}
    p = subprocess.run([DBT, "build", "--profiles-dir", ".", "--project-dir", ".", "--target-path", str(d / "target"),
                        "--vars", json.dumps({"month": "2024-03"})], cwd=ROOT / "nyc_taxi", env=env,
                       capture_output=True, text=True)
    return {"db": db, "first": first, "second": second, "dbt": p}


def test_loader_is_idempotent(warehouse):
    assert warehouse["first"]["rows_loaded"] == 8 and warehouse["first"]["rows_replaced"] == 0
    assert warehouse["second"]["rows_loaded"] == 8 and warehouse["second"]["rows_replaced"] == 8


def test_dbt_build_passes_on_clean_fixture(warehouse):
    assert warehouse["dbt"].returncode == 0, warehouse["dbt"].stdout[-2000:]


def test_each_quality_rule_rejects_exactly_its_row(warehouse):
    con = duckdb.connect(str(warehouse["db"]), read_only=True)
    out = dict(con.execute("select outcome, row_count from marts.dq_rejections_by_month").fetchall())
    assert out == {"valid": 2, "duplicate_record": 1, "pickup_outside_file_month": 1, "duration_out_of_range": 1,
                   "distance_out_of_range": 1, "negative_amount": 1, "zero_passengers": 1}
    assert con.execute("select count(*), sum(total_amount) from marts.fct_trips").fetchone() == (2, 71)


def test_marts_reconcile(warehouse):
    con = duckdb.connect(str(warehouse["db"]), read_only=True)
    rows = con.execute("select pickup_borough, trips, revenue from marts.agg_daily_borough order by trip_date").fetchall()
    assert [r[1] for r in rows] == [1, 1] and float(sum(r[2] for r in rows)) == 71.0
