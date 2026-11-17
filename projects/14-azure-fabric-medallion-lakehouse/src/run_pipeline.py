"""
Orchestrator - the local stand-in for an ADF / Fabric Data Pipeline:
ingest (bronze) -> transform (silver) -> model (gold) -> data quality.

    python src/run_pipeline.py                     # everything in the landing zone
    python src/run_pipeline.py --until 2010-11     # only files up to a month (incremental demo)
    python src/run_pipeline.py --reset             # delete the local lake first
Exit code 1 if an error-level DQ check fails.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

import bronze
import dq
import gold
import silver
from common import ROOT, path


def run(base: Path = ROOT, until: str | None = None, landing: Path | None = None, quiet: bool = False) -> dict:
    landing = landing or path("landing")
    files = sorted(p for p in Path(landing).glob("*.csv") if until is None or p.stem <= until)
    run_id = pd.Timestamp.now(tz="Europe/Dublin").strftime("%Y%m%dT%H%M%S%f")
    t = {}
    t0 = time.perf_counter(); b = bronze.ingest(base, files, run_id); t["bronze_s"] = time.perf_counter() - t0
    months = [Path(f).stem for f in b["changed_files"]]
    t0 = time.perf_counter(); s = silver.build(base, months); t["silver_s"] = time.perf_counter() - t0
    t0 = time.perf_counter(); g = gold.build(base); t["gold_s"] = time.perf_counter() - t0
    t0 = time.perf_counter(); q = dq.run(base, run_id); t["dq_s"] = time.perf_counter() - t0
    failed = [c for c in q if not c["passed"] and c["severity"] == "error"]
    out = {"run_id": run_id, "bronze": b, "silver": s, "gold": g, "dq": q, "timings": t, "failed": len(failed)}
    if not quiet:
        print(f"bronze: {b['files_ingested']} files ingested ({b['files_replaced']} replaced), {b['files_skipped_unchanged']} unchanged, {b['rows']:,} rows")
        print(f"silver: {s}")
        print(f"gold:   {g}")
        for c in q:
            print(f"  [{'PASS' if c['passed'] else 'FAIL'}] {c['severity']:5s} {c['check']:30s} observed={c['observed']:.4g} {c['rule']}")
        print({k: round(v, 2) for k, v in t.items()})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--until", help="last landing month to include, e.g. 2010-11")
    ap.add_argument("--reset", action="store_true")
    a = ap.parse_args()
    if a.reset:
        shutil.rmtree(ROOT / "lake", ignore_errors=True)
    return 1 if run(until=a.until)["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
