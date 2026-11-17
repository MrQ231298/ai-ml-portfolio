"""Shared paths, config and small Delta helpers for the local lakehouse."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import duckdb
import pyarrow as pa
from deltalake import DeltaTable, write_deltalake

ROOT = Path(__file__).resolve().parents[1]
CONF = json.loads((ROOT / "conf" / "lakehouse.json").read_text())


def path(key: str, base: Path | None = None) -> Path:
    return (base or ROOT) / CONF["paths"][key]


def sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def exists(p: Path) -> bool:
    return (Path(p) / "_delta_log").exists()


def read(p: Path, version: int | None = None) -> pa.Table:
    return DeltaTable(str(p), version=version).to_pyarrow_table()


def write(p: Path, data, mode: str = "append", partition_by=None, predicate: str | None = None) -> None:
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    kw = {"partition_by": partition_by} if partition_by and not exists(p) else {}
    if predicate:
        kw["predicate"] = predicate
    write_deltalake(str(p), data, mode=mode, **kw)


def duck() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("SET threads TO 4")
    return con
