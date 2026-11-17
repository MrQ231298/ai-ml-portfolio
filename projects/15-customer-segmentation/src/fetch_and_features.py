"""
Download UCI Online Retail (CC BY 4.0; Chen, Sain & Guo 2012), verify
its checksum, clean the transactions, and build one row of RFM-style
features per customer for a CALIBRATION period, plus their behaviour in
the following 90-day HOLD-OUT period (used only to validate segments).

    python src/fetch_and_features.py
"""

from __future__ import annotations

import io
import json
import urllib.request
import zipfile
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DL, DATA = ROOT / "downloads", ROOT / "data"
URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
SHA = "f5385cbb54bbebf7196389109c6b0621faab0c304e3702548165e71c84aede8b"
CAL_END = pd.Timestamp("2011-09-10")      # calibration: 2010-12-01 .. 2011-09-09
HOLD_END = pd.Timestamp("2011-12-10")     # hold-out: 2011-09-10 .. 2011-12-09 (91 days)
NON_PRODUCT = {"POST", "DOT", "M", "D", "C2", "BANK CHARGES", "AMAZONFEE", "CRUK", "PADS", "S", "B"}


def load_raw() -> pd.DataFrame:
    DL.mkdir(exist_ok=True)
    z = DL / "online_retail.zip"
    if not z.exists():
        urllib.request.urlretrieve(URL, z)
    if sha256(z.read_bytes()).hexdigest() != SHA:
        raise SystemExit("checksum mismatch")
    with zipfile.ZipFile(z) as zf:
        x = zf.read("Online Retail.xlsx")
    return pd.read_excel(io.BytesIO(x), dtype={"InvoiceNo": str, "StockCode": str, "Description": str, "Country": str})


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    log = {"raw_rows": int(len(raw))}
    d = raw.drop_duplicates()
    log["exact_duplicates_removed"] = int(len(raw) - len(d))
    d = d[d["CustomerID"].notna()]
    log["rows_with_customer_id"] = int(len(d))
    d = d[~d["StockCode"].str.upper().isin(NON_PRODUCT) & (d["UnitPrice"] > 0)]
    log["rows_after_non_product_and_price_filter"] = int(len(d))
    d = d.assign(customer_id=d["CustomerID"].astype(int).astype(str), amount=d["Quantity"] * d["UnitPrice"],
                 is_cancel=d["InvoiceNo"].str.startswith("C"))
    return d, log


def features(d: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    w = d[(d["InvoiceDate"] >= start) & (d["InvoiceDate"] < end)]
    sales, canc = w[~w.is_cancel], w[w.is_cancel]
    inv = sales.groupby(["customer_id", "InvoiceNo"]).agg(ts=("InvoiceDate", "min"), value=("amount", "sum")).reset_index()
    g = inv.groupby("customer_id")
    f = pd.DataFrame({
        "recency_days": (end - g["ts"].max()).dt.days,
        "frequency": g["InvoiceNo"].nunique(),
        "monetary": g["value"].sum() + canc.groupby("customer_id")["amount"].sum().reindex(g.size().index).fillna(0),
        "tenure_days": (end - g["ts"].min()).dt.days,
        "avg_basket_value": g["value"].mean(),
        "distinct_products": sales.groupby("customer_id")["StockCode"].nunique(),
        "cancel_rate": (-canc.groupby("customer_id")["amount"].sum()).reindex(g.size().index).fillna(0) / g["value"].sum(),
    })
    gaps = inv.sort_values("ts").groupby("customer_id")["ts"].apply(lambda s: s.diff().dt.days.mean())
    f["mean_days_between_orders"] = gaps.reindex(f.index)
    f["country"] = sales.groupby("customer_id")["Country"].agg(lambda s: s.mode().iat[0])
    return f


def main() -> None:
    raw = load_raw()
    d, log = clean(raw)
    start = d["InvoiceDate"].min().normalize()
    cal = features(d, start, CAL_END)
    cal = cal[cal["monetary"] > 0]                               # net buyers only
    hold = d[(d["InvoiceDate"] >= CAL_END) & (d["InvoiceDate"] < HOLD_END)]
    fut = hold.groupby("customer_id")["amount"].sum()
    cal["future_90d_revenue"] = fut.reindex(cal.index).fillna(0.0).round(2)
    cal["future_90d_purchased"] = hold[~hold.is_cancel].groupby("customer_id")["InvoiceNo"].nunique().reindex(cal.index).fillna(0).gt(0)
    cal.index.name = "customer_id"
    DATA.mkdir(exist_ok=True)
    cal.round(4).to_csv(DATA / "customer_features_calibration.csv")
    meta = {"source": URL, "sha256": SHA, "licence": "CC BY 4.0",
            "citation": "Chen, D. (2012). Online Retail [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5BW33",
            "calibration_period": [str(start.date()), str((CAL_END - pd.Timedelta(days=1)).date())],
            "holdout_period": [str(CAL_END.date()), str((HOLD_END - pd.Timedelta(days=1)).date())],
            "cleaning": log, "customers": int(len(cal)), "holdout_buyer_share": float(cal["future_90d_purchased"].mean()),
            "built_at": pd.Timestamp.now(tz="Europe/Dublin").isoformat(timespec="seconds")}
    (DATA / "source_metadata.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))
    print(cal.describe().T.round(2))


if __name__ == "__main__":
    main()
