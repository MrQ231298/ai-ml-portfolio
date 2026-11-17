"""Extract + load one month of NYC TLC yellow-taxi trips into DuckDB (raw layer).

Idempotent: the month's rows are deleted and re-inserted in one transaction,
so re-running a month (or an Airflow retry) never duplicates data. The file's
SHA-256 is recorded in data_manifest.json on first download and verified after.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
DOWNLOADS = ROOT / "downloads"
WAREHOUSE = ROOT / "warehouse" / "nyc_taxi.duckdb"
MANIFEST = ROOT / "data_manifest.json"
URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{month}.parquet"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fetch(month: str) -> Path:
    DOWNLOADS.mkdir(exist_ok=True)
    path = DOWNLOADS / f"yellow_tripdata_{month}.parquet"
    if not path.exists():
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(URL.format(month=month), tmp)
        tmp.rename(path)
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    digest = sha256(path)
    if month in manifest and manifest[month]["sha256"] != digest:
        raise RuntimeError(f"{month}: checksum changed ({digest}); TLC republished the file? Investigate before loading.")
    manifest.setdefault(month, {"url": URL.format(month=month), "sha256": digest, "bytes": path.stat().st_size,
                                "first_downloaded": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return path


def load(month: str, path: Path, db: Path = WAREHOUSE) -> dict:
    db.parent.mkdir(exist_ok=True)
    con = duckdb.connect(str(db))
    con.execute("SET memory_limit = '2GB'; SET threads = 4;")
    con.execute("CREATE SCHEMA IF NOT EXISTS raw")
    con.execute(f"""CREATE TABLE IF NOT EXISTS raw.yellow_trips AS
                    SELECT *, ''::VARCHAR AS source_month, now()::TIMESTAMP AS _loaded_at, ''::VARCHAR AS _source_file
                    FROM read_parquet('{path}') LIMIT 0""")
    t0 = time.time()
    con.execute("BEGIN TRANSACTION")
    deleted = con.execute("DELETE FROM raw.yellow_trips WHERE source_month = ?", [month]).fetchone()[0]
    con.execute(f"""INSERT INTO raw.yellow_trips BY NAME
                    SELECT *, '{month}' AS source_month, now()::TIMESTAMP AS _loaded_at, '{path.name}' AS _source_file
                    FROM read_parquet('{path}')""")
    con.execute("COMMIT")
    n = con.execute("SELECT COUNT(*) FROM raw.yellow_trips WHERE source_month = ?", [month]).fetchone()[0]
    con.close()
    return {"month": month, "rows_loaded": int(n), "rows_replaced": int(deleted), "seconds": round(time.time() - t0, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", required=True, help="YYYY-MM")
    a = ap.parse_args()
    print(json.dumps(load(a.month, fetch(a.month))))


if __name__ == "__main__":
    main()
