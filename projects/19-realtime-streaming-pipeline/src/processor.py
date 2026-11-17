"""Stream processor: Kafka -> event-time windows -> SQLite sink, effectively once.

Exactly-once results come from a transactional sink: window results, late
events, the processor state (open windows, de-dup ids, watermark) AND the next
Kafka offset for each partition are written in ONE SQLite transaction. Kafka
offsets are never auto-committed; on start-up the consumer seeks to the offsets
stored in the sink. A crash between checkpoints therefore replays exactly the
messages whose effects were not yet committed.

--mode naive is the common anti-pattern for comparison: auto-committed offsets,
in-memory state, no de-duplication, append-only writes.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import statistics
import time

from confluent_kafka import OFFSET_BEGINNING, Consumer, TopicPartition

from config import BOOTSTRAP, REPORTS, SINK, TOPIC
from windowing import PartitionState, flush, process

SCHEMA = """
CREATE TABLE IF NOT EXISTS window_results (borough TEXT, window_start TEXT, trips INTEGER, revenue REAL,
    partition INTEGER, emitted_at REAL, PRIMARY KEY (borough, window_start));
CREATE TABLE IF NOT EXISTS late_events (event_id TEXT PRIMARY KEY, borough TEXT, event_time TEXT, amount REAL,
    partition INTEGER, kafka_offset INTEGER, watermark TEXT);
CREATE TABLE IF NOT EXISTS checkpoints (partition INTEGER PRIMARY KEY, next_offset INTEGER, state TEXT,
    messages INTEGER, duplicates INTEGER, late INTEGER, updated_at REAL);
CREATE TABLE IF NOT EXISTS naive_window_results (borough TEXT, window_start TEXT, trips INTEGER, revenue REAL,
    partition INTEGER, emitted_at REAL);
CREATE TABLE IF NOT EXISTS throughput (ts REAL, partition INTEGER, next_offset INTEGER, messages INTEGER);
"""


def connect():
    SINK.parent.mkdir(exist_ok=True)
    con = sqlite3.connect(SINK, isolation_level=None, timeout=30)
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript(SCHEMA)
    return con


def run(mode: str, crash_after: int, checkpoint_every: int, idle_exit: float, group: str, tag: str):
    con = connect()
    states: dict[int, PartitionState] = {}
    counters: dict[int, dict] = {}
    next_off: dict[int, int] = {}
    exactly = mode == "exactly_once"

    def on_assign(consumer, parts):
        for tp in parts:
            row = con.execute("SELECT next_offset, state, messages, duplicates, late FROM checkpoints WHERE partition=?",
                              (tp.partition,)).fetchone() if exactly else None
            if row:
                tp.offset = row[0]
                states[tp.partition] = PartitionState.from_json(json.loads(row[1]))
                counters[tp.partition] = {"messages": row[2], "duplicates": row[3], "late": row[4]}
            else:
                if exactly:
                    tp.offset = OFFSET_BEGINNING
                states[tp.partition] = PartitionState()
                counters[tp.partition] = {"messages": 0, "duplicates": 0, "late": 0}
        consumer.assign(parts)

    cfg = {"bootstrap.servers": BOOTSTRAP, "group.id": group, "auto.offset.reset": "earliest",
           "enable.auto.commit": not exactly, "auto.commit.interval.ms": 1000, "enable.partition.eof": True}
    c = Consumer(cfg)
    c.subscribe([TOPIC], on_assign=on_assign)
    pending_w, pending_late, pending_send = [], [], []
    processed = 0
    lat_consume, lat_durable = [], []
    last_ckpt, last_msg, eof = time.time(), time.time(), set()
    t_start = None

    def checkpoint():
        nonlocal pending_w, pending_late, pending_send, last_ckpt
        now = time.time()
        con.execute("BEGIN IMMEDIATE")
        if exactly:
            con.executemany("INSERT OR REPLACE INTO window_results VALUES (?,?,?,?,?,?)",
                            [(w["borough"], w["window_start"], w["trips"], w["revenue"], w["partition"], now) for w in pending_w])
            con.executemany("INSERT OR REPLACE INTO late_events VALUES (?,?,?,?,?,?,?)", pending_late)
            for p, st in states.items():
                if p in next_off:
                    k = counters[p]
                    con.execute("INSERT OR REPLACE INTO checkpoints VALUES (?,?,?,?,?,?,?)",
                                (p, next_off[p], json.dumps(st.to_json()), k["messages"], k["duplicates"], k["late"], now))
                    con.execute("INSERT INTO throughput VALUES (?,?,?,?)", (now, p, next_off[p], k["messages"]))
        else:
            con.executemany("INSERT INTO naive_window_results VALUES (?,?,?,?,?,?)",
                            [(w["borough"], w["window_start"], w["trips"], w["revenue"], w["partition"], now) for w in pending_w])
        con.execute("COMMIT")
        lat_durable.extend(now - s for s in pending_send[::10])
        pending_w, pending_late, pending_send = [], [], []
        last_ckpt = now

    try:
        while True:
            msgs = c.consume(num_messages=1000, timeout=0.1)
            if not msgs and processed and states and len(eof) >= len(states) and time.time() - last_msg > idle_exit:
                break
            for m in msgs:
                if m.error():
                    if m.error().code() == -191:  # partition EOF
                        eof.add(m.partition())
                    continue
                last_msg = time.time()
                t_start = t_start or last_msg
                eof.discard(m.partition())
                ev = json.loads(m.value())
                p = m.partition()
                st = states[p]
                if not exactly:
                    st.seen.clear()  # naive: no de-duplication
                r = process(st, ev)
                counters[p]["messages"] += 1
                if r["status"] == "duplicate":
                    counters[p]["duplicates"] += 1
                elif r["status"] == "late":
                    counters[p]["late"] += 1
                    pending_late.append((ev["event_id"], ev["borough"], ev["event_time"], ev["amount"], p, m.offset(),
                                         st.watermark.isoformat()))
                pending_w.extend(dict(w, partition=p) for w in r["closed"])
                next_off[p] = m.offset() + 1
                now = time.time()
                if processed % 10 == 0:
                    lat_consume.append(now - ev["send_ts"])
                pending_send.append(ev["send_ts"])
                processed += 1
                if crash_after and processed >= crash_after:
                    print(f"simulated crash after {processed} messages (uncommitted: {len(pending_send)})", flush=True)
                    os._exit(3)
            if processed and (len(pending_send) >= checkpoint_every or time.time() - last_ckpt > 1.0):
                checkpoint()
        for p, st in states.items():  # end of stream: close remaining windows
            pending_w.extend(dict(w, partition=p) for w in flush(st))
        checkpoint()
    finally:
        c.close()
    wall = (last_msg - t_start) if t_start else 0
    q = lambda xs, p: sorted(xs)[int(p * (len(xs) - 1))] * 1000 if xs else None  # noqa: E731
    stats = {"mode": mode, "tag": tag, "messages_this_run": processed, "active_seconds": round(wall, 2),
             "msgs_per_s": round(processed / wall, 1) if wall else None,
             "consume_latency_ms": {"p50": q(lat_consume, .5), "p95": q(lat_consume, .95), "p99": q(lat_consume, .99)},
             "durable_latency_ms": {"p50": q(lat_durable, .5), "p95": q(lat_durable, .95), "p99": q(lat_durable, .99)},
             "counters": counters, "checkpoint_every": checkpoint_every}
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / f"processor_{tag}.json").write_text(json.dumps(stats, indent=2, default=str))
    print(json.dumps({k: v for k, v in stats.items() if k != "counters"}, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["exactly_once", "naive"], default="exactly_once")
    ap.add_argument("--crash-after", type=int, default=0)
    ap.add_argument("--checkpoint-every", type=int, default=5000)
    ap.add_argument("--idle-exit", type=float, default=5.0)
    ap.add_argument("--group", default="trip-windows")
    ap.add_argument("--tag", default="run")
    a = ap.parse_args()
    run(a.mode, a.crash_after, a.checkpoint_every, a.idle_exit, a.group, a.tag)
