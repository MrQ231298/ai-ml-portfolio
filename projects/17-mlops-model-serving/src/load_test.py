"""Start the API with uvicorn and measure latency and throughput with real HTTP calls."""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from common import FEATURES, REPORTS, load_hourly, write_json  # noqa: E402

PORT = 8765


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


def main():
    df = load_hourly()
    recs = json.loads(df[df.ts >= "2012-06-01"][FEATURES].head(2000).to_json(orient="records"))
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(PORT), "--workers", "1",
                             "--log-level", "warning"], cwd=ROOT)
    try:
        url = f"http://127.0.0.1:{PORT}"
        for _ in range(60):
            try:
                if httpx.get(f"{url}/health").status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.5)
        out = {}
        with httpx.Client(base_url=url) as c:
            for _ in range(20):  # warm-up
                c.post("/predict", json={"records": recs[:1]})
            for batch in (1, 100):
                lat = []
                t0 = time.perf_counter()
                for i in range(300):
                    s = time.perf_counter()
                    r = c.post("/predict", json={"records": recs[(i * batch) % 1900:(i * batch) % 1900 + batch]})
                    r.raise_for_status()
                    lat.append((time.perf_counter() - s) * 1000)
                wall = time.perf_counter() - t0
                out[f"batch_{batch}"] = {"requests": 300, "p50_ms": round(statistics.median(lat), 2),
                                         "p95_ms": round(pct(lat, 95), 2), "p99_ms": round(pct(lat, 99), 2),
                                         "requests_per_s": round(300 / wall, 1), "records_per_s": round(300 * batch / wall, 1)}

        def one(i):
            with httpx.Client(base_url=url) as c2:
                s = time.perf_counter()
                c2.post("/predict", json={"records": recs[i % 1900:i % 1900 + 1]}).raise_for_status()
                return (time.perf_counter() - s) * 1000
        t0 = time.perf_counter()
        with ThreadPoolExecutor(8) as ex:
            lat = list(ex.map(one, range(400)))
        wall = time.perf_counter() - t0
        out["concurrent_8_clients_batch_1"] = {"requests": 400, "p50_ms": round(statistics.median(lat), 2),
                                                "p95_ms": round(pct(lat, 95), 2), "requests_per_s": round(400 / wall, 1)}
        drift = httpx.get(f"{url}/drift").json()
        out["drift_endpoint_after_test"] = {"records": drift.get("records"), "alerts": drift.get("alerts")}
        out["setup"] = "uvicorn, 1 worker, same machine as client (8 vCPU shared), HTTP/1.1, no container"
        write_json(REPORTS / "serving_benchmark.json", out)
        print(json.dumps(out, indent=1))
    finally:
        proc.terminate()
        proc.wait(10)


if __name__ == "__main__":
    main()
