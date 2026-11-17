"""Run the full recorded experiment against a live Kafka broker.

1. throughput: publish all events as fast as possible (idempotent producer)
2. exactly-once: process with a simulated crash after 60,000 messages, restart, finish
3. naive: same crash point with auto-commit / in-memory state / no de-dup
4. both sinks compared with the batch reference (truth.py)
5. latency: a second topic paced at 2,000 msg/s, processed live
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
SRC = ROOT / "src"


def run(args, env=None):
    t0 = time.time()
    p = subprocess.run([PY, *args], cwd=SRC, env={**os.environ, **(env or {})}, capture_output=True, text=True)
    print(" ".join(args), "->", p.returncode, round(time.time() - t0, 1), "s", flush=True)
    if p.stderr.strip():
        print(p.stderr[-800:])
    return p


def paced(res, data):
    env = {"STREAM_TOPIC": "taxi-trips-paced", "STREAM_SINK": str(data / "sink_paced.db")}
    for f in ("sink_paced.db", "sink_paced.db-wal", "sink_paced.db-shm"):
        (data / f).unlink(missing_ok=True)
    run(["producer.py", "--recreate", "--create-only"], env)
    proc = subprocess.Popen([PY, "processor.py", "--tag", "paced_2000", "--idle-exit", "8", "--group", "paced"], cwd=SRC,
                            env={**os.environ, **env})
    time.sleep(8)  # let the consumer join and get its partitions
    run(["producer.py", "--rate", "2000", "--limit", "60000"], env)
    proc.wait(300)
    res["processor_paced_2000"] = json.loads((ROOT / "reports" / "processor_paced_2000.json").read_text())
    res["producer_paced"] = json.loads((ROOT / "reports" / "producer_rate2000.json").read_text())


def main():
    if "--only-paced" in sys.argv:
        res = json.loads((ROOT / "reports" / "metrics.json").read_text())
        paced(res, ROOT / "data")
        (ROOT / "reports" / "metrics.json").write_text(json.dumps(res, indent=2, default=str))
        print(json.dumps(res["processor_paced_2000"], indent=1))
        return
    data = ROOT / "data"
    for f in ("sink.db", "sink.db-wal", "sink.db-shm", "sink_paced.db", "sink_paced.db-wal", "sink_paced.db-shm"):
        (data / f).unlink(missing_ok=True)
    res = {}
    run(["producer.py", "--recreate"])
    res["producer_max_rate"] = json.loads((ROOT / "reports" / "producer_rate0.json").read_text())
    p1 = run(["processor.py", "--crash-after", "60000", "--tag", "eo_part1_crash"])
    res["eo_crash_exit_code"] = p1.returncode
    res["eo_crash_stdout"] = p1.stdout.strip().splitlines()[-1]
    run(["processor.py", "--tag", "eo_part2_resume"])
    run(["processor.py", "--mode", "naive", "--group", "naive", "--crash-after", "60000", "--tag", "naive_part1_crash"])
    run(["processor.py", "--mode", "naive", "--group", "naive", "--tag", "naive_part2_resume"])
    sys.path.insert(0, str(SRC))
    from truth import compare
    res["exactly_once_vs_truth"] = compare(data / "sink.db")
    res["naive_vs_truth"] = compare(data / "sink.db", table="naive_window_results")
    paced(res, data)
    for tag in ("eo_part1_crash", "eo_part2_resume", "naive_part1_crash", "naive_part2_resume"):
        f = ROOT / "reports" / f"processor_{tag}.json"
        if f.exists():
            res[f"processor_{tag}"] = json.loads(f.read_text())
    res["broker"] = "Apache Kafka 4.4.0 (KRaft, single node, 4 partitions, replication 1), Temurin JRE 21, heap 768 MB"
    (ROOT / "reports" / "metrics.json").write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps({k: res[k] for k in ("exactly_once_vs_truth", "naive_vs_truth")}, indent=1))


if __name__ == "__main__":
    main()
