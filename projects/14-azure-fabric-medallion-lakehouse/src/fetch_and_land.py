"""
Download UCI Online Retail II (CC BY 4.0), verify its checksum, and drop
it into the landing zone as one CSV per calendar month - the way a
source system would deliver monthly extracts to ADLS / OneLake Files.
Values are written exactly as read (as text); no cleaning happens here.

    python src/fetch_and_land.py
"""

from __future__ import annotations

import io
import json
import urllib.request
import zipfile

import pandas as pd

from common import CONF, ROOT, path, sha256

DL = ROOT / "downloads"


def main() -> None:
    DL.mkdir(exist_ok=True)
    z = DL / "online_retail_ii.zip"
    if not z.exists():
        urllib.request.urlretrieve(CONF["source"]["url"], z)
    got = sha256(z)
    if got != CONF["source"]["sha256"]:
        raise SystemExit(f"checksum mismatch: {got}")
    with zipfile.ZipFile(z) as zf:
        xlsx = zf.read("online_retail_II.xlsx")
    sheets = pd.read_excel(io.BytesIO(xlsx), sheet_name=None, dtype=str)
    raw = pd.concat(sheets.values(), ignore_index=True)
    month = raw["InvoiceDate"].str.slice(0, 7)
    land = path("landing")
    land.mkdir(parents=True, exist_ok=True)
    counts = {}
    for m, g in raw.groupby(month, sort=True):
        g.to_csv(land / f"{m}.csv", index=False)
        counts[m] = len(g)
    meta = {"source": CONF["source"], "rows": int(len(raw)), "sheets": {k: int(len(v)) for k, v in sheets.items()},
            "files": counts, "landed_at": pd.Timestamp.now(tz="Europe/Dublin").isoformat(timespec="seconds")}
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "landing_manifest.json").write_text(json.dumps(meta, indent=1))
    print(f"{len(raw):,} rows in {len(counts)} monthly files -> {land}")


if __name__ == "__main__":
    main()
