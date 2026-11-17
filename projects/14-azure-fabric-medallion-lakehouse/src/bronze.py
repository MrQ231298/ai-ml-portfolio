"""
BRONZE - land raw files as-is into a Delta table, plus lineage columns.
Equivalent to an ADF / Fabric Data Pipeline Copy activity into a
Lakehouse table. Incremental: a control table records each file's
SHA-256; unchanged files are skipped, changed files replace their own
partition (so a corrected re-delivery does not duplicate rows).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa

from common import exists, path, read, sha256, write

COLS = ["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "Customer ID", "Country"]


def control_table(base: Path) -> pd.DataFrame:
    p = path("control", base)
    return read(p).to_pandas() if exists(p) else pd.DataFrame(columns=["file_name", "sha256", "batch_id", "rows"])


def ingest(base: Path, files: list[Path], batch_id: str) -> dict:
    ctl = control_table(base)
    known = dict(zip(ctl.file_name, ctl.sha256))
    stats = {"files_seen": len(files), "files_ingested": 0, "files_skipped_unchanged": 0, "files_replaced": 0, "rows": 0, "changed_files": []}
    new_ctl = []
    for f in files:
        h = sha256(f)
        if known.get(f.name) == h:
            stats["files_skipped_unchanged"] += 1
            continue
        df = pd.read_csv(f, dtype=str, keep_default_na=False, na_values=[""])
        assert list(df.columns) == COLS, f"{f.name}: unexpected columns {list(df.columns)}"
        df.columns = ["invoice", "stock_code", "description", "quantity", "invoice_date", "price", "customer_id", "country"]
        df["_source_file"] = f.name
        df["_source_month"] = f.stem
        df["_source_sha256"] = h
        df["_batch_id"] = batch_id
        df["_row_number"] = range(1, len(df) + 1)
        tbl = pa.Table.from_pandas(df, preserve_index=False)
        if f.name in known:   # corrected re-delivery: overwrite only that file's partition
            write(path("bronze", base), tbl, mode="overwrite", partition_by=["_source_month"], predicate=f"_source_month = '{f.stem}'")
            stats["files_replaced"] += 1
        else:
            write(path("bronze", base), tbl, mode="append", partition_by=["_source_month"])
        stats["files_ingested"] += 1
        stats["rows"] += len(df)
        stats["changed_files"].append(f.name)
        new_ctl.append({"file_name": f.name, "sha256": h, "batch_id": batch_id, "rows": len(df)})
    if new_ctl:
        ctl = pd.concat([ctl[~ctl.file_name.isin([c["file_name"] for c in new_ctl])], pd.DataFrame(new_ctl)], ignore_index=True)
        ctl["rows"] = ctl["rows"].astype("int64")
        write(path("control", base), pa.Table.from_pandas(ctl, preserve_index=False), mode="overwrite")
    return stats
