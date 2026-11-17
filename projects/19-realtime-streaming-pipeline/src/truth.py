"""Batch reference: read the whole topic back and recompute the expected
results with an INDEPENDENT, vectorised implementation of the same rules
(pandas, no shared code with windowing.py), then compare with the sink."""

from __future__ import annotations

import json
import sqlite3

import numpy as np
import pandas as pd
from confluent_kafka import Consumer, TopicPartition
from confluent_kafka.admin import AdminClient

from config import BOOTSTRAP, PARTITIONS, TOPIC


def read_topic() -> pd.DataFrame:
    c = Consumer({"bootstrap.servers": BOOTSTRAP, "group.id": "truth-reader", "enable.auto.commit": False})
    rows = []
    for p in range(PARTITIONS):
        lo, hi = c.get_watermark_offsets(TopicPartition(TOPIC, p), timeout=10)
        c.assign([TopicPartition(TOPIC, p, lo)])
        last = lo - 1
        while last < hi - 1:
            for m in c.consume(1000, timeout=5):
                if not m.error():
                    rows.append({**json.loads(m.value()), "partition": p, "offset": m.offset()})
                    last = m.offset()
    c.close()
    return pd.DataFrame(rows)


def expected(df: pd.DataFrame) -> tuple[pd.DataFrame, set, int]:
    df = df.sort_values(["partition", "offset"]).copy()
    df["et"] = pd.to_datetime(df.event_time)
    first = ~df.duplicated(["partition", "event_id"])
    n_dup = int((~first).sum())
    d = df[first].copy()
    d["ws"] = d.et.dt.floor("5min")
    prev_max = d.groupby("partition").et.transform(lambda s: s.cummax().shift(1))
    closed_up_to = (prev_max - pd.Timedelta(minutes=10)).dt.floor("5min")
    d["late"] = (d.ws + pd.Timedelta(minutes=5)) <= closed_up_to
    ok = d[~d.late]
    w = ok.groupby(["borough", "ws"]).agg(trips=("event_id", "size"), revenue=("amount", "sum")).reset_index()
    w["window_start"] = w.ws.map(lambda t: t.isoformat())
    return w[["borough", "window_start", "trips", "revenue"]], set(d[d.late].event_id), n_dup


def compare(sink_path, table="window_results") -> dict:
    df = read_topic()
    exp, late_ids, n_dup = expected(df)
    con = sqlite3.connect(sink_path)
    if table == "naive_window_results":
        got = pd.read_sql(f"select borough, window_start, sum(trips) trips, sum(revenue) revenue from {table} group by 1, 2", con)
        got_late = set()
    else:
        got = pd.read_sql(f"select borough, window_start, trips, revenue from {table}", con)
        got_late = set(pd.read_sql("select event_id from late_events", con).event_id)
    m = exp.merge(got, on=["borough", "window_start"], how="outer", suffixes=("_exp", "_got"), indicator=True)
    both = m[m._merge == "both"]
    exact = (both.trips_exp == both.trips_got) & (np.abs(both.revenue_exp - both.revenue_got) < 0.005)
    return {"topic_messages": int(len(df)), "expected_windows": int(len(exp)), "sink_windows": int(len(got)),
            "windows_exact_match": int(exact.sum()), "windows_mismatched": int((~exact).sum()),
            "windows_missing_in_sink": int((m._merge == "left_only").sum()),
            "windows_extra_in_sink": int((m._merge == "right_only").sum()),
            "expected_trips": int(exp.trips.sum()), "sink_trips": int(got.trips.sum()),
            "trip_count_error": int(got.trips.sum() - exp.trips.sum()),
            "expected_revenue": round(float(exp.revenue.sum()), 2), "sink_revenue": round(float(got.revenue.sum()), 2),
            "expected_late_events": len(late_ids), "sink_late_events": len(got_late),
            "late_sets_equal": late_ids == got_late if table == "window_results" else None,
            "duplicates_in_topic": n_dup}


if __name__ == "__main__":
    print(json.dumps(compare("data/sink.db"), indent=1))
