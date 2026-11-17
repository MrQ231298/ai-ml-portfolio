"""
Evaluate the local lakehouse: full vs incremental equivalence, replay
idempotency, a corrected-file re-delivery, Delta time travel, and a
business summary of the gold layer.

    python src/evaluate.py   -> reports/metrics.json, reports/*.png, reports/gold_samples/*.csv
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from deltalake import DeltaTable

import run_pipeline
from common import ROOT, duck, path, read
from gold import GOLD

REPORTS = ROOT / "reports"
WORK = ROOT / "lake" / "_eval"


def checksum(base: Path) -> dict[str, str]:
    con = duck()
    out = {}
    tables = {"silver": path("silver", base)} | {f"gold.{g}": path("gold", base) / g for g in GOLD}
    for name, p in tables.items():
        t = read(p)
        cols = [c for c in t.column_names if not c.startswith("_")]
        con.register("t", t.select(cols))
        out[name] = con.sql(f"SELECT md5(string_agg(CAST(ROW({', '.join(cols)}) AS VARCHAR), '|' ORDER BY {', '.join(cols)})) FROM t").fetchall()[0][0]
        con.unregister("t")
    return out


def main() -> None:
    shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True)
    # 1. full run
    full = WORK / "full"
    t0 = time.perf_counter(); r_full = run_pipeline.run(full, quiet=True); full_s = time.perf_counter() - t0
    ck_full = checksum(full)
    # 2. incremental in three loads
    inc = WORK / "incremental"
    steps = []
    for until in ["2010-05", "2010-12", None]:
        r = run_pipeline.run(inc, until=until, quiet=True)
        steps.append({"until": until or "2011-12", "files_ingested": r["bronze"]["files_ingested"],
                      "months_rebuilt": r["silver"]["months_rebuilt"], "failed_checks": r["failed"]})
    ck_inc = checksum(inc)
    # 3. replay the same landing zone
    r_replay = run_pipeline.run(full, quiet=True)
    ck_replay = checksum(full)
    # 4. corrected re-delivery of one month
    land2 = WORK / "landing_corrected"
    shutil.copytree(path("landing"), land2)
    f = land2 / "2011-03.csv"
    df = pd.read_csv(f, dtype=str, keep_default_na=False, na_values=[""])
    sales = df[~df["Invoice"].str.startswith(("C", "A"))].index
    df.loc[sales[0], "Quantity"] = str(int(df.loc[sales[0], "Quantity"]) + 100)   # one quantity corrected
    df = df.drop(index=sales[1:11])                                                   # ten lines withdrawn
    df.to_csv(f, index=False)
    silver_v_before = DeltaTable(str(path("silver", inc))).version()
    before = duck().sql("SELECT year_month, net_sales_gbp FROM t").df() if False else None  # placeholder for readability
    con = duck(); con.register("a", read(path("gold", inc) / "agg_monthly_country"))
    month_before = con.sql("SELECT year_month, SUM(net_sales_gbp) s FROM a GROUP BY 1").df().set_index("year_month")["s"]
    r_corr = run_pipeline.run(inc, landing=land2, quiet=True)
    con.register("b", read(path("gold", inc) / "agg_monthly_country"))
    month_after = con.sql("SELECT year_month, SUM(net_sales_gbp) s FROM b GROUP BY 1").df().set_index("year_month")["s"]
    changed_months = sorted(m for m in month_before.index if abs(float(month_before[m]) - float(month_after[m])) > 0.001)
    silver_now = DeltaTable(str(path("silver", inc)))
    rows_now = silver_now.to_pyarrow_table().num_rows
    rows_then = DeltaTable(str(path("silver", inc)), version=silver_v_before).to_pyarrow_table().num_rows
    # 5. business summary from the full gold layer
    g = {n: read(path("gold", full) / n).to_pandas() for n in ["agg_monthly_country", "fact_sales", "dim_product", "dim_customer"]}
    s = read(path("silver", full)).to_pandas()
    monthly = g["agg_monthly_country"].groupby("year_month")[["gross_sales_gbp", "cancellations_gbp", "net_sales_gbp"]].sum().astype(float)
    country = g["agg_monthly_country"].groupby("country")["net_sales_gbp"].sum().astype(float).sort_values(ascending=False)
    line_types = s["line_type"].value_counts().to_dict()
    metrics = {
        "source": json.loads((ROOT / "reports" / "landing_manifest.json").read_text())["source"],
        "run_locally_not_on_azure": True,
        "rows": {"bronze": r_full["bronze"]["rows"], "silver": r_full["silver"]["rows_out"],
                 "exact_duplicates_removed": r_full["silver"]["exact_duplicates_removed"],
                 "untypeable_rows_dropped": r_full["silver"]["untypeable_rows_dropped"], "gold": r_full["gold"]},
        "silver_line_types": {k: int(v) for k, v in line_types.items()},
        "guest_line_share_silver": float(s["is_guest"].mean()),
        "timings_full_run_s": {k: round(v, 2) for k, v in r_full["timings"].items()} | {"total_s": round(full_s, 2)},
        "throughput_rows_per_s_end_to_end": round(r_full["bronze"]["rows"] / full_s),
        "dq_full_run": r_full["dq"],
        "incremental": {"steps": steps, "identical_to_full_run": ck_inc == ck_full,
                        "tables_compared": len(ck_full), "differing": [k for k in ck_full if ck_full[k] != ck_inc[k]]},
        "replay": {"files_ingested": r_replay["bronze"]["files_ingested"], "files_skipped": r_replay["bronze"]["files_skipped_unchanged"],
                   "identical_after_replay": ck_replay == ck_full},
        "correction": {"files_replaced": r_corr["bronze"]["files_replaced"], "months_rebuilt": r_corr["silver"]["months_rebuilt"],
                       "silver_rows_before": rows_then, "silver_rows_after": rows_now, "silver_row_change": rows_now - rows_then,
                       "gold_months_changed": changed_months, "silver_version_before": silver_v_before, "silver_version_after": silver_now.version(),
                       "time_travel_read_ok": rows_then != rows_now},
        "business": {"net_sales_gbp_total": round(float(monthly["net_sales_gbp"].sum()), 2),
                     "gross_sales_gbp_total": round(float(monthly["gross_sales_gbp"].sum()), 2),
                     "cancellations_gbp_total": round(float(monthly["cancellations_gbp"].sum()), 2),
                     "uk_share_of_net_sales": float(country.get("United Kingdom", 0) / country.sum()),
                     "top_countries_net_sales_gbp": {k: round(v, 2) for k, v in country.head(6).items()},
                     "products": int(len(g["dim_product"])), "products_with_multiple_descriptions": int((g["dim_product"]["description_variants"] > 1).sum()),
                     "customers": int(len(g["dim_customer"])), "customers_in_more_than_one_country": int((g["dim_customer"]["countries_seen"] > 1).sum())},
    }
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    fig, ax = plt.subplots(figsize=(10, 4))
    (monthly[["gross_sales_gbp", "net_sales_gbp"]] / 1000).plot(ax=ax, marker=".")
    ax.set_ylabel("£ thousand"); ax.set_xlabel(""); ax.set_title("Gold layer: monthly sales, UCI Online Retail II (Dec 2011 is a part month)")
    ax.legend(["Gross sales", "Net of cancellations"]); fig.tight_layout(); fig.savefig(REPORTS / "gold_monthly_sales.png", dpi=120); plt.close(fig)
    samples = REPORTS / "gold_samples"; samples.mkdir(exist_ok=True)
    g["agg_monthly_country"].to_csv(samples / "agg_monthly_country.csv", index=False)
    g["dim_product"].head(200).to_csv(samples / "dim_product_first200.csv", index=False)
    print(json.dumps({k: metrics[k] for k in ["rows", "silver_line_types", "timings_full_run_s", "incremental", "replay", "correction", "business"]}, indent=1, default=str))
    print([(c["check"], c["passed"]) for c in r_full["dq"] if not c["passed"]])


if __name__ == "__main__":
    main()
