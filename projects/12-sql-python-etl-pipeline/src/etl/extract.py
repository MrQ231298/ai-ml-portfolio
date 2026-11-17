"""Extract: read one batch folder exactly as delivered, with file hashes."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

FLAG = "SYNTHETIC_EXAMPLE_DATA"


@dataclass
class Batch:
    batch_id: str
    extract_date: pd.Timestamp
    folder: Path
    hashes: dict[str, str]
    customers: pd.DataFrame
    products: pd.DataFrame
    fx: pd.DataFrame
    orders: pd.DataFrame
    lines: pd.DataFrame


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_batch(folder: Path) -> Batch:
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest.get("data_flag") != FLAG:
        raise ValueError(f"{folder}: manifest is not tagged {FLAG}; refusing to load")
    hashes = {f: sha256(folder / f) for f in manifest["files"]}
    # everything is read as text first; typing is the cleaner's job
    rd = lambda f: pd.read_csv(folder / f, dtype=str, keep_default_na=False, na_values=[""])  # noqa: E731
    prod_json = json.loads((folder / "products.json").read_text())
    products = pd.DataFrame(prod_json["products"]).astype("string")
    b = Batch(manifest["batch_id"], pd.Timestamp(manifest["extract_date"]), folder, hashes,
              rd("customers.csv"), products, rd("fx_rates.csv"), rd("orders.csv"), rd("order_lines.csv"))
    for name in ["customers", "fx", "orders", "lines"]:
        df = getattr(b, name)
        if "data_flag" not in df or (df["data_flag"] != FLAG).any():
            raise ValueError(f"{folder}/{name}: rows not tagged {FLAG}")
        setattr(b, name, df.drop(columns="data_flag"))
    return b
