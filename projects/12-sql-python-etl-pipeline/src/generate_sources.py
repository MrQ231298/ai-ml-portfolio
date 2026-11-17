"""
Generate SYNTHETIC, deliberately messy source extracts for a fictional
Irish online retailer: a CRM customer export, a product catalogue (JSON),
order headers and order lines (CSV) and a monthly FX table, in two
batches (Jan-Jun and Jul-Oct 2026).

The generator first builds a *clean truth*, then corrupts a copy. Every
injected defect is logged in data/truth/injected_defects.csv and the
revenue a correct pipeline should report is saved in
data/truth/expected_totals.json. The pipeline never reads data/truth/;
only the evaluation does.

    python src/generate_sources.py
"""

from __future__ import annotations

import json
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
TRUTH = ROOT / "data" / "truth"
SEED = 1211
EXTRACT_DATE = pd.Timestamp("2026-11-01")
FLAG = "SYNTHETIC_EXAMPLE_DATA"
FUTURE_IDS: set[str] = set()

FIRST = ["Aoife", "Ciara", "Sean", "Conor", "Niamh", "Liam", "Siobhan", "Darragh", "Orla", "Cian", "Emma",
         "Jack", "Sarah", "James", "Roisin", "Patrick", "Grace", "Eoin", "Aisling", "Michael"]
LAST = ["Murphy", "Kelly", "Byrne", "Ryan", "O'Brien", "Walsh", "O'Sullivan", "Doyle", "McCarthy", "Gallagher",
        "Doherty", "Kennedy", "Lynch", "Quinn", "Brennan", "Nolan", "Duffy", "Boyle", "McLaughlin", "Daly"]
COUNTIES = ["Dublin", "Cork", "Galway", "Limerick", "Donegal", "Kerry", "Mayo", "Wexford", "Louth", "Kildare"]
CATS = ["Home & Garden", "Kitchen", "Outdoor", "Toys", "Stationery", "Gifts"]
COUNTRY_VARIANTS = {"IE": ["IE", "Ireland", "ireland ", "Rep. of Ireland", "IRL"], "GB": ["GB", "UK", "United Kingdom", "u.k."],
                    "NI": ["Northern Ireland", "N. Ireland", "GB-NIR"]}


def truth_tables(rng):
    n_c, n_p = 1200, 150
    cust = pd.DataFrame({
        "customer_id": [f"C{100000 + i}" for i in range(n_c)],
        "first_name": rng.choice(FIRST, n_c), "last_name": rng.choice(LAST, n_c),
        "country": rng.choice(["IE", "GB", "NI"], n_c, p=[0.75, 0.15, 0.10]),
        "county": rng.choice(COUNTIES, n_c),
        "signup_date": pd.to_datetime("2024-01-01") + pd.to_timedelta(rng.integers(0, 900, n_c), unit="D"),
        "phone": [f"+353 8{rng.integers(3, 10)} {rng.integers(100, 999)} {rng.integers(1000, 9999)}" for _ in range(n_c)],
    })
    cust["email"] = [f"{f.lower()}.{l.lower().replace(chr(39), '')}{i}@example.ie" for i, (f, l) in enumerate(zip(cust.first_name, cust.last_name))]
    prod = pd.DataFrame({
        "product_id": [f"P{5000 + i}" for i in range(n_p)],
        "product_name": [f"{rng.choice(['Classic', 'Premium', 'Mini', 'Eco', 'Deluxe'])} {rng.choice(['Mug', 'Lamp', 'Planter', 'Notebook', 'Puzzle', 'Blanket', 'Candle', 'Tray', 'Kite', 'Frame'])} {i}" for i in range(n_p)],
        "category": rng.choice(CATS, n_p),
        "list_price_eur": np.round(rng.lognormal(np.log(18), 0.6, n_p), 2),
        "unit_cost_eur": 0.0,
    })
    prod["unit_cost_eur"] = np.round(prod["list_price_eur"] * rng.uniform(0.35, 0.65, n_p), 2)
    months = pd.period_range("2026-01", "2026-10", freq="M")
    fx = pd.DataFrame({"month": months.astype(str), "currency": "GBP",
                       "eur_per_unit": np.round(1.17 + np.cumsum(rng.normal(0, 0.006, len(months))), 4)})
    n_o = 7000
    odate = pd.to_datetime("2026-01-01") + pd.to_timedelta(rng.integers(0, 304 * 24 * 60, n_o), unit="min")
    orders = pd.DataFrame({
        "order_id": [f"SO-{200000 + i}" for i in range(n_o)],
        "customer_id": rng.choice(cust.customer_id, n_o),
        "order_ts": odate.sort_values().values,
        "channel": rng.choice(["web", "app", "marketplace"], n_o, p=[0.6, 0.3, 0.1]),
        "status": rng.choice(["completed", "completed", "completed", "completed", "cancelled"], n_o),
    })
    orders = orders.merge(cust[["customer_id", "country"]], on="customer_id")
    orders["currency"] = np.where(orders["country"] == "GB", "GBP", "EUR")
    orders = orders.drop(columns="country")
    lines = []
    for oid, cur, ts in zip(orders.order_id, orders.currency, orders.order_ts):
        k = rng.integers(1, 5)
        ps = rng.choice(len(prod), k, replace=False)
        m = pd.Timestamp(ts).strftime("%Y-%m")
        rate = fx.loc[fx.month == m, "eur_per_unit"].iloc[0]
        for ln, pi in enumerate(ps, 1):
            price_eur = prod.list_price_eur.iloc[pi]
            price = round(price_eur / rate, 2) if cur == "GBP" else price_eur
            lines.append((oid, ln, prod.product_id.iloc[pi], int(rng.integers(1, 6)), price,
                          float(rng.choice([0, 0, 0, 0.1, 0.2]))))
    lines = pd.DataFrame(lines, columns=["order_id", "line_no", "product_id", "quantity", "unit_price", "discount_pct"])
    # a few genuine returns (negative quantity), which are valid business records
    ret = rng.choice(len(lines), 60, replace=False)
    lines.loc[ret, "quantity"] = -lines.loc[ret, "quantity"].abs()
    return cust, prod, orders, lines, fx


def fmt_date(ts, style):
    ts = pd.Timestamp(ts)
    return {0: ts.strftime("%Y-%m-%d"), 1: ts.strftime("%d/%m/%Y"), 2: ts.strftime("%b %d %Y")}[style]


def fmt_ts(ts, style):
    ts = pd.Timestamp(ts)
    return {0: ts.strftime("%Y-%m-%dT%H:%M:%S"), 1: ts.strftime("%d/%m/%Y %H:%M"), 2: ts.strftime("%Y-%m-%d %H:%M:%S")}[style]


def money_str(x, rng):
    s = rng.integers(0, 4)
    return [f"{x:.2f}", f"€{x:.2f}", f"{x:.2f}".replace(".", ","), f"EUR {x:.2f}"][s]


def corrupt(cust, prod, orders, lines, fx, rng):
    defects = []
    log = lambda table, key, kind: defects.append((table, key, kind))  # noqa: E731
    # ---- customers
    c = cust.copy()
    c["signup_date"] = [fmt_date(d, rng.integers(0, 3)) for d in c.signup_date]
    c["country"] = [rng.choice(COUNTRY_VARIANTS[x]) for x in c.country]
    c["phone"] = [p if rng.random() < 0.5 else p.replace("+353 ", "0").replace(" ", "") for p in c.phone]
    bad_email = rng.choice(len(c), 24, replace=False)
    for i in bad_email:
        c.loc[i, "email"] = c.loc[i, "email"].replace("@", " at ") if rng.random() < 0.5 else c.loc[i, "email"].split("@")[0]
        log("customers", c.loc[i, "customer_id"], "invalid_email")
    case_mess = rng.choice(len(c), 120, replace=False)
    c.loc[case_mess, "first_name"] = c.loc[case_mess, "first_name"].str.upper() + "  "
    dups = c.sample(36, random_state=SEED)
    for k in dups.customer_id:
        log("customers", k, "exact_duplicate")
    c = pd.concat([c, dups], ignore_index=True)
    # ---- products
    p = prod.drop(columns="unit_cost_eur").copy()
    p["category"] = [x.upper() if rng.random() < 0.15 else (x.lower() if rng.random() < 0.15 else x) for x in p.category]
    p["list_price"] = [money_str(x, rng) for x in p.list_price_eur]
    miss = rng.choice(len(p), 3, replace=False)
    p.loc[miss, "list_price"] = None
    for i in miss:
        log("products", p.loc[i, "product_id"], "missing_price")
    p["unit_cost_eur"] = prod["unit_cost_eur"]
    p = p.drop(columns="list_price_eur")
    # ---- orders
    o = orders.copy()
    o["order_ts"] = [fmt_ts(t, rng.integers(0, 3)) for t in o.order_ts]
    o["status"] = [s.upper() if rng.random() < 0.1 else s for s in o.status]
    orph = rng.choice(len(o), 35, replace=False)
    for i in orph:
        o.loc[i, "customer_id"] = f"C{900000 + i}"
        log("orders", o.loc[i, "order_id"], "orphan_customer")
    fut = rng.choice(np.setdiff1d(np.arange(len(o)), orph), 12, replace=False)
    for i in fut:
        o.loc[i, "order_ts"] = fmt_ts(EXTRACT_DATE + pd.Timedelta(days=int(rng.integers(30, 400))), 0)
        log("orders", o.loc[i, "order_id"], "future_date")
    # ---- lines
    li = lines.copy()
    li["quantity"] = li["quantity"].astype("object")
    bad_q = rng.choice(len(li), 40, replace=False)
    for i in bad_q:
        li.loc[i, "quantity"] = None if rng.random() < 0.5 else 0
        log("order_lines", f"{li.loc[i, 'order_id']}#{li.loc[i, 'line_no']}", "null_or_zero_quantity")
    orph_p = rng.choice(np.setdiff1d(np.arange(len(li)), bad_q), 30, replace=False)
    for i in orph_p:
        li.loc[i, "product_id"] = f"P{9900 + int(i) % 90}"
        log("order_lines", f"{li.loc[i, 'order_id']}#{li.loc[i, 'line_no']}", "orphan_product")
    dup_l = li.sample(250, random_state=SEED)
    for _, r in dup_l.iterrows():
        log("order_lines", f"{r.order_id}#{r.line_no}", "exact_duplicate")
    li = pd.concat([li, dup_l], ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)
    return c, p, o, li, pd.DataFrame(defects, columns=["table", "record_key", "defect_type"])


def expected_totals(prod, orders, lines, fx, defects):
    """What a correct pipeline should report, computed on the CLEAN truth."""
    bad_orders = set(defects.loc[defects.defect_type.isin(["orphan_customer", "future_date"]), "record_key"])
    bad_lines = set(defects.loc[defects.defect_type.isin(["null_or_zero_quantity", "orphan_product"]), "record_key"])
    bad_prod = set(defects.loc[defects.defect_type == "missing_price", "record_key"])
    li = lines.merge(orders[["order_id", "order_ts", "currency", "status"]], on="order_id")
    li["key"] = li.order_id + "#" + li.line_no.astype(str)
    keep = (~li.order_id.isin(bad_orders)) & (~li.key.isin(bad_lines)) & (~li.product_id.isin(bad_prod)) & (li.status == "completed")
    li = li[keep].copy()
    li["month"] = pd.to_datetime(li.order_ts).dt.strftime("%Y-%m")
    li = li.merge(fx[["month", "eur_per_unit"]], on="month", how="left")
    li["rate"] = np.where(li.currency == "GBP", li.eur_per_unit, 1.0)
    D = lambda x: Decimal(str(x))  # noqa: E731  exact decimal arithmetic, half-up to the cent
    li["net_eur"] = [float((D(q) * D(p) * (1 - D(d)) * D(r)).quantize(Decimal("0.01"), ROUND_HALF_UP))
                     for q, p, d, r in zip(li.quantity, li.unit_price, li.discount_pct, li.rate)]
    by_month = li.groupby("month")["net_eur"].sum().round(2)
    return {"net_revenue_eur": round(float(li.net_eur.sum()), 2), "completed_order_lines": int(len(li)),
            "completed_orders": int(li.order_id.nunique()),
            "net_revenue_by_month": {k: float(v) for k, v in by_month.items()}}


def write_batches(c, p, o, li, fx, true_ts):
    # split on the TRUE timestamp (the corrupted strings mix ISO and day-first formats;
    # pandas' format="mixed", dayfirst=True silently swaps day and month in ISO dates)
    o_ts = pd.Series(pd.to_datetime(true_ts.values), index=o.index)
    o_ts[o.order_id.isin(FUTURE_IDS)] = pd.Timestamp("2099-01-01")   # future-dated rows arrive in the latest extract
    cut = pd.Timestamp("2026-07-01")
    b1_orders = o[o_ts < cut]
    b2_orders = o[o_ts >= cut]
    # batch 2 re-sends the last 150 orders of batch 1 (overlapping extract window)
    b2_orders = pd.concat([b1_orders.tail(150), b2_orders])
    for name, ords in [("batch_2026_06", b1_orders), ("batch_2026_10", b2_orders)]:
        d = RAW / name
        d.mkdir(parents=True, exist_ok=True)
        ords.assign(data_flag=FLAG).to_csv(d / "orders.csv", index=False)
        li[li.order_id.isin(ords.order_id)].assign(data_flag=FLAG).to_csv(d / "order_lines.csv", index=False)
        cust = c if name == "batch_2026_10" else c.sample(frac=0.8, random_state=1)
        if name == "batch_2026_10":   # 40 customers moved county: should overwrite (SCD type 1)
            cust = cust.copy()
            idx = cust.sample(40, random_state=2).index
            cust.loc[idx, "county"] = "Sligo"
        cust.assign(data_flag=FLAG).to_csv(d / "customers.csv", index=False)
        (d / "products.json").write_text(json.dumps({"data_flag": FLAG, "products": p.to_dict("records")}, indent=1, default=str))
        fx.assign(data_flag=FLAG).to_csv(d / "fx_rates.csv", index=False)
        extract = "2026-07-01" if name == "batch_2026_06" else str(EXTRACT_DATE.date())
        (d / "manifest.json").write_text(json.dumps({"batch_id": name, "extract_date": extract, "data_flag": FLAG,
                                                     "files": ["customers.csv", "products.json", "fx_rates.csv", "orders.csv", "order_lines.csv"]}, indent=1))
    # orders whose customer exists only in batch 2's customer file are NOT a problem:
    # the pipeline loads customers before orders in every batch.


def main() -> None:
    rng = np.random.default_rng(SEED)
    cust, prod, orders, lines, fx = truth_tables(rng)
    c, p, o, li, defects = corrupt(cust, prod, orders, lines, fx, rng)
    FUTURE_IDS.update(defects.loc[defects.defect_type == "future_date", "record_key"])
    write_batches(c, p, o, li, fx, orders.set_index("order_id").loc[o.order_id, "order_ts"])
    TRUTH.mkdir(parents=True, exist_ok=True)
    defects.to_csv(TRUTH / "injected_defects.csv", index=False)
    exp = expected_totals(prod, orders, lines, fx, defects)
    exp["unique_customers"] = int(len(cust)); exp["unique_products_with_price"] = int(len(prod) - 3)
    exp["final_county_sligo"] = None  # filled by evaluation from batch file (SCD1 check)
    (TRUTH / "expected_totals.json").write_text(json.dumps(exp, indent=1))
    print(defects.defect_type.value_counts().to_string())
    print({k: v for k, v in exp.items() if k != "net_revenue_by_month"})


if __name__ == "__main__":
    main()
