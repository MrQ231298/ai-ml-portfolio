"""
Download the IBM Telco Customer Churn sample dataset.

Source: IBM's public GitHub repository `telco-customer-churn-on-icp4d`
(Apache License 2.0). A copy of the CSV is already committed in
`data/`, so this script is only needed to re-fetch it.

    python src/fetch_data.py
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "Telco-Customer-Churn.csv"


def main() -> None:
    resp = requests.get(URL, timeout=60)
    resp.raise_for_status()
    OUT.write_bytes(resp.content)
    meta = {
        "source_url": URL,
        "licence": "Apache License 2.0 (IBM/telco-customer-churn-on-icp4d)",
        "fetched_europe_dublin": datetime.now(ZoneInfo("Europe/Dublin")).isoformat(timespec="seconds"),
        "sha256": hashlib.sha256(resp.content).hexdigest(),
        "bytes": len(resp.content),
    }
    (ROOT / "data" / "source_metadata.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
