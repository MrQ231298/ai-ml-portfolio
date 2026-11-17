"""
Load: idempotent upserts into DuckDB core tables, referential checks
against what is already loaded, a quarantine for rejected rows, and a
retry of quarantined rows whose parent has since arrived.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd

from . import clean as C
from .extract import Batch

SQL_DIR = Path(__file__).resolve().parents[2] / "sql"
RETRYABLE = ("unknown_customer", "unknown_product", "unknown_order")


def connect(db_path: Path | str) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(str(db_path))
    con.execute((SQL_DIR / "00_schema.sql").read_text())
    return con


def already_loaded(con, batch: Batch) -> bool:
    rows = con.execute("SELECT file_name, sha256 FROM etl.batch_log WHERE batch_id = ?", [batch.batch_id]).fetchall()
    return bool(rows) and {f: h for f, h in rows} == batch.hashes


def _upsert(con, table: str, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    con.register("_incoming", df)
    cols = ", ".join(df.columns)
    con.execute(f"INSERT OR REPLACE INTO {table} ({cols}) SELECT {cols} FROM _incoming")
    con.unregister("_incoming")
    return len(df)


def _quarantine(con, rej: pd.DataFrame, batch_id: str) -> int:
    if rej.empty:
        return 0
    q = rej.assign(batch_id=batch_id, resolved_batch=None).drop_duplicates(["table_name", "record_key", "reason"])
    con.register("_rej", q)
    con.execute("""INSERT OR REPLACE INTO etl.quarantine
                   SELECT table_name, record_key, reason, payload, batch_id, CAST(resolved_batch AS VARCHAR) FROM _rej""")
    con.unregister("_rej")
    return len(q)


def _existing(con, table: str, col: str) -> set[str]:
    return {r[0] for r in con.execute(f"SELECT {col} FROM {table}").fetchall()}


def _split_orphans(df, key_col, known, table, record_key, reason):
    bad = ~df[key_col].isin(known)
    rej = pd.DataFrame({"table_name": table, "record_key": record_key[bad].values, "reason": reason,
                        "payload": df[bad].to_json(orient="records", lines=True, date_format="iso").splitlines() if bad.any() else []})
    return df[~bad], rej


def _retry(con, batch_id: str, stats: dict) -> None:
    """Re-check quarantined orphans against the parents loaded so far."""
    for table, parent_tbl, parent_col, reason in [
        ("orders", "core.customer", "customer_id", "unknown_customer"),
        ("order_lines", "core.sales_order", "order_id", "unknown_order"),
        ("order_lines", "core.product", "product_id", "unknown_product"),
    ]:
        q = con.execute("""SELECT record_key, payload FROM etl.quarantine
                           WHERE table_name = ? AND reason = ? AND resolved_batch IS NULL""", [table, reason]).fetchall()
        if not q:
            continue
        df = pd.DataFrame([json.loads(p) for _, p in q])
        df["_key"] = [k for k, _ in q]
        ok = df[df[parent_col].isin(_existing(con, parent_tbl, parent_col))]
        if table == "order_lines" and not ok.empty:   # a line needs every parent
            ok = ok[ok["order_id"].isin(_existing(con, "core.sales_order", "order_id"))
                    & ok["product_id"].isin(_existing(con, "core.product", "product_id"))]
        if ok.empty:
            continue
        rows = ok.drop(columns="_key").assign(last_batch_id=batch_id)
        if table == "orders":
            rows["order_ts"] = pd.to_datetime(rows["order_ts"])
            _upsert(con, "core.sales_order", rows)
        else:
            _upsert(con, "core.order_line", rows)
        con.executemany("UPDATE etl.quarantine SET resolved_batch = ? WHERE table_name = ? AND record_key = ? AND reason = ?",
                        [[batch_id, table, k, reason] for k in ok["_key"]])
        stats[(table, f"retried_ok_{reason}")] = len(ok)


def load_batch(con, batch: Batch, run_id: int) -> dict:
    stats: dict[tuple[str, str], int] = {}
    # 1. dimensions first
    cust, rej_c, s = C.clean_customers(batch.customers)
    stats.update({("customers", k): v for k, v in s.items()})
    prod, rej_p, s = C.clean_products(batch.products)
    stats.update({("products", k): v for k, v in s.items()})
    fx = C.clean_fx(batch.fx)
    stats[("customers", "rows_in")] = len(batch.customers)
    stats[("products", "rows_in")] = len(batch.products)
    stats[("customers", "upserted")] = _upsert(con, "core.customer", cust.drop(columns="email_valid").assign(last_batch_id=batch.batch_id))
    stats[("products", "upserted")] = _upsert(con, "core.product", prod.assign(last_batch_id=batch.batch_id))
    _upsert(con, "core.fx_rate", fx)
    # 2. retry earlier orphans now that new parents may exist
    _retry(con, batch.batch_id, stats)
    # 3. orders, checked against customers
    orders, rej_o, s = C.clean_orders(batch.orders, batch.extract_date)
    stats.update({("orders", k): v for k, v in s.items()})
    stats[("orders", "rows_in")] = len(batch.orders)
    orders, rej_o2 = _split_orphans(orders, "customer_id", _existing(con, "core.customer", "customer_id"),
                                    "orders", orders["order_id"], "unknown_customer")
    stats[("orders", "upserted")] = _upsert(con, "core.sales_order", orders.assign(last_batch_id=batch.batch_id))
    # 4. lines, checked against orders and products
    lines, rej_l, s = C.clean_lines(batch.lines)
    stats.update({("order_lines", k): v for k, v in s.items()})
    stats[("order_lines", "rows_in")] = len(batch.lines)
    key = lambda d: d["order_id"] + "#" + d["line_no"].astype(str)  # noqa: E731
    lines, rej_l2 = _split_orphans(lines, "product_id", _existing(con, "core.product", "product_id"),
                                   "order_lines", key(lines), "unknown_product")
    lines, rej_l3 = _split_orphans(lines, "order_id", _existing(con, "core.sales_order", "order_id"),
                                   "order_lines", key(lines), "unknown_order")
    stats[("order_lines", "upserted")] = _upsert(con, "core.order_line", lines.assign(last_batch_id=batch.batch_id))
    # 5. quarantine everything rejected in this batch
    all_rej = pd.concat([rej_c, rej_p, rej_o, rej_o2, rej_l, rej_l2, rej_l3], ignore_index=True)
    stats[("all", "quarantined_this_batch")] = _quarantine(con, all_rej, batch.batch_id)
    for (t, r), n in all_rej.groupby(["table_name", "reason"]).size().items():
        stats[(t, f"rejected_{r}")] = int(n)
    # 6. batch log
    now = pd.Timestamp.now(tz="Europe/Dublin").tz_localize(None)
    con.executemany("INSERT INTO etl.batch_log VALUES (?, ?, ?, ?, ?)",
                    [[batch.batch_id, f, h, run_id, now] for f, h in batch.hashes.items()])
    con.executemany("INSERT OR REPLACE INTO etl.run_stats VALUES (?, ?, ?, ?, ?)",
                    [[run_id, batch.batch_id, t, m, int(v)] for (t, m), v in stats.items()])
    return stats


def build_marts(con, extract_date: str) -> None:
    for f in sorted(SQL_DIR.glob("[1-9]*.sql")):
        con.execute(f.read_text().replace("{extract_date}", extract_date))
