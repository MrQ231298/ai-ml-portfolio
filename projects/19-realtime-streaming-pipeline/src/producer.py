"""Publish the event stream to Kafka with an idempotent producer.

--rate 0 sends as fast as possible (throughput test); --rate N paces to N msg/s.
Each message carries its send time so the processor can measure latency.
Delivery reports are counted per partition.
"""

from __future__ import annotations

import argparse
import json
import time

import pandas as pd
from confluent_kafka import Producer
from confluent_kafka.admin import AdminClient, NewTopic

from config import BOOTSTRAP, PARTITIONS, REPORTS, TOPIC
from events import EVENTS


def ensure_topic(recreate: bool):
    admin = AdminClient({"bootstrap.servers": BOOTSTRAP})
    if recreate and TOPIC in admin.list_topics(timeout=10).topics:
        admin.delete_topics([TOPIC])[TOPIC].result()
        time.sleep(2)
    if TOPIC not in admin.list_topics(timeout=10).topics:
        admin.create_topics([NewTopic(TOPIC, PARTITIONS, 1)])[TOPIC].result()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", type=float, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--recreate", action="store_true")
    ap.add_argument("--create-only", action="store_true")
    a = ap.parse_args()
    ensure_topic(a.recreate)
    if a.create_only:
        return
    ev = pd.read_parquet(EVENTS)
    if a.limit:
        ev = ev.head(a.limit)
    p = Producer({"bootstrap.servers": BOOTSTRAP, "enable.idempotence": True, "acks": "all",
                  "linger.ms": 5, "compression.type": "lz4", "batch.num.messages": 10000})
    delivered = []

    def cb(err, msg):
        if err is None:
            delivered.append((msg.key().decode(), msg.partition(), msg.offset()))

    t0 = time.perf_counter()
    for i, row in enumerate(ev.itertuples(index=False)):
        if a.rate:
            target = t0 + i / a.rate
            delay = target - time.perf_counter()
            if delay > 0:
                p.poll(0)
                time.sleep(delay)
        payload = {"event_id": row.event_id, "event_time": row.event_time.isoformat(), "borough": row.borough,
                   "zone_id": int(row.zone_id), "amount": round(float(row.amount), 2), "send_ts": time.time()}
        while True:
            try:
                p.produce(TOPIC, key=row.borough.encode(), value=json.dumps(payload).encode(), on_delivery=cb)
                break
            except BufferError:
                p.poll(0.05)
        if i % 1000 == 0:
            p.poll(0)
    p.flush(60)
    wall = time.perf_counter() - t0
    dd = pd.DataFrame(delivered, columns=["borough", "partition", "offset"])
    stats = {"messages": len(ev), "delivered": len(dd), "seconds": round(wall, 2),
             "msgs_per_s": round(len(ev) / wall, 1), "rate_target": a.rate}
    stats["per_partition"] = dd.groupby("partition").size().to_dict()
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / f"producer_rate{int(a.rate)}.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
