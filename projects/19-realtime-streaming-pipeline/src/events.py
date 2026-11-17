"""Turn real NYC taxi trips into an out-of-order event stream with known defects.

Event time = pickup time. Arrival (send) order is event time plus a random
network/device delay, so the stream is out of order. Two kinds of defect are
injected at fixed, seeded rates so the pipeline can be scored against truth:

  late events   1.5% of events are delayed 15-60 minutes (past the allowed lateness)
  duplicates    1.0% of events are re-sent (as a client retry would) a few seconds later
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
P18_DOWNLOAD = ROOT.parent / "18-dbt-airflow-modern-data-stack" / "downloads" / "yellow_tripdata_2024-01.parquet"
ZONES = ROOT.parent / "18-dbt-airflow-modern-data-stack" / "nyc_taxi" / "seeds" / "taxi_zones.csv"
EVENTS = ROOT / "data" / "events.parquet"
TLC_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet"
LOCAL_DOWNLOAD = ROOT / "downloads" / "yellow_tripdata_2024-01.parquet"


def source_file() -> Path:
    """Re-use project 18's download if present, otherwise fetch the public file once."""
    if P18_DOWNLOAD.exists():
        return P18_DOWNLOAD
    if not LOCAL_DOWNLOAD.exists():
        import urllib.request
        LOCAL_DOWNLOAD.parent.mkdir(exist_ok=True)
        urllib.request.urlretrieve(TLC_URL, LOCAL_DOWNLOAD)
    return LOCAL_DOWNLOAD
SEED = 19
LATE_RATE, DUP_RATE = 0.015, 0.010


def build(day_from: str = "2024-01-15", day_to: str = "2024-01-17", path: Path | None = None) -> pd.DataFrame:
    path = path or source_file()
    con = duckdb.connect()
    df = con.execute(f"""
        select tpep_pickup_datetime as event_time, z.Borough as borough, cast(PULocationID as int) as zone_id,
               cast(total_amount as double) as amount, cast(trip_distance as double) as distance
        from read_parquet('{path}') t
        left join read_csv_auto('{ZONES}') z on t.PULocationID = z.LocationID
        where tpep_pickup_datetime >= '{day_from}' and tpep_pickup_datetime < '{day_to}'
          and total_amount between 0 and 1000
        order by event_time
    """).df()
    rng = np.random.default_rng(SEED)
    n = len(df)
    df.insert(0, "event_id", [f"e{i:07d}" for i in range(n)])
    delay = rng.exponential(20.0, n)  # seconds: normal jitter
    late = rng.random(n) < LATE_RATE
    delay[late] = rng.uniform(15 * 60, 60 * 60, late.sum())
    df["arrival_time"] = df.event_time + pd.to_timedelta(delay, unit="s")
    df["injected_late"] = late
    df["is_duplicate"] = False
    dups = df[rng.random(n) < DUP_RATE].copy()
    dups["arrival_time"] = dups.arrival_time + pd.to_timedelta(rng.uniform(1, 30, len(dups)), unit="s")
    dups["is_duplicate"] = True
    out = pd.concat([df, dups]).sort_values("arrival_time", kind="stable").reset_index(drop=True)
    out["borough"] = out["borough"].fillna("Unknown")
    return out


def to_message(row) -> bytes:
    return json.dumps({"event_id": row.event_id, "event_time": row.event_time.isoformat(),
                       "borough": row.borough, "zone_id": int(row.zone_id), "amount": round(float(row.amount), 2),
                       "distance": round(float(row.distance), 2)}).encode()


if __name__ == "__main__":
    ev = build()
    EVENTS.parent.mkdir(exist_ok=True)
    ev.to_parquet(EVENTS, index=False)
    print(len(ev), "messages;", int(ev.is_duplicate.sum()), "duplicates;", int(ev.injected_late.sum()), "late-delayed")
