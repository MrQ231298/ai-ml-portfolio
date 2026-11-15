"""
Download the Women's E-Commerce Clothing Reviews dataset (23,486 real,
anonymised customer reviews; CC0 Public Domain on Kaggle, published by
Nick Brooks). Kaggle requires a login, so this script uses a public
GitHub mirror of the same CSV and records its SHA-256 for verification.

    python src/fetch_data.py
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

URL = ("https://raw.githubusercontent.com/hanzhang0420/Women-Clothing-E-commerce/master/"
       "Womens%20Clothing%20E-Commerce%20Reviews.csv")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "womens_clothing_reviews.csv"


def main() -> None:
    r = requests.get(URL, timeout=120)
    r.raise_for_status()
    OUT.write_bytes(r.content)
    meta = {
        "dataset": "Women's E-Commerce Clothing Reviews",
        "original": "https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews",
        "licence": "CC0: Public Domain (as stated on the Kaggle dataset page)",
        "mirror_used": URL,
        "fetched_europe_dublin": datetime.now(ZoneInfo("Europe/Dublin")).isoformat(timespec="seconds"),
        "sha256": hashlib.sha256(r.content).hexdigest(),
        "bytes": len(r.content),
        "note": "Retailer name already replaced with 'retailer' in the source; reviewer identities are not included.",
    }
    (ROOT / "data" / "source_metadata.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
