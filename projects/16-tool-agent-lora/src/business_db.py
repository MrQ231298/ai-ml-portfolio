"""Build the SYNTHETIC business database and policy documents used by the agent.

Fictional company: "Northlight Components Ltd", an Irish distributor of
electronic components. Every row is generated from a fixed seed; no real
customer or employer data is used.
"""

from __future__ import annotations

import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "SYNTHETIC_northlight.db"
DOCS_DIR = ROOT / "data" / "docs"
SEED = 16

COUNTIES = ["Dublin", "Cork", "Galway", "Limerick", "Donegal", "Sligo",
            "Mayo", "Kerry", "Waterford", "Louth", "Kildare", "Wexford"]
SEGMENTS = ["trade", "retail", "education", "public sector"]
CATEGORIES = {
    "connectors": [("CN-100", "USB-C panel connector", 2.40), ("CN-220", "M12 circular connector", 7.90),
                   ("CN-310", "Automotive wiring harness plug", 4.15)],
    "sensors": [("SN-010", "Temperature sensor NTC", 1.85), ("SN-045", "Hall effect sensor", 3.60),
                ("SN-120", "Pressure sensor module", 18.50)],
    "microcontrollers": [("MC-032", "32-bit MCU board", 12.90), ("MC-064", "Dual-core MCU board", 21.40)],
    "displays": [("DS-020", "2.4 inch TFT display", 9.75), ("DS-070", "7 inch touch display", 46.00)],
    "power": [("PW-005", "5 V buck converter", 3.20), ("PW-120", "12 V DC power supply", 14.80),
              ("PW-480", "48 V industrial PSU", 62.00)],
    "cables": [("CB-001", "1 m ribbon cable", 1.10), ("CB-050", "CAN bus cable 5 m", 8.40)],
}
MONTHS_2026 = list(range(1, 11))  # January to October 2026

DOCS = {
    "returns_policy": """# Returns policy

## Return window
Customers may return unused items within 30 days of delivery for a full refund.
Trade account customers have an extended return window of 45 days.

## Condition of returns
Items must be unused and in the original anti-static packaging. Opened reels of
surface-mount components cannot be returned.

## Restocking fee
A restocking fee of 15% applies to returns of custom-cut cables.

## Faulty goods
Faulty goods are replaced free of charge. Report a fault within 14 days of
delivery to get a prepaid return label.
""",
    "shipping_policy": """# Shipping and delivery

## Delivery times
Orders placed before 2 pm on a working day are dispatched the same day.
Standard delivery within Ireland takes 1 to 2 working days.

## Delivery charges
Standard delivery costs EUR 6.50 per order. Delivery is free for orders over EUR 150
before VAT.

## Northern Ireland and Great Britain
Deliveries to Northern Ireland take 2 to 3 working days. Deliveries to Great
Britain take 3 to 5 working days and may need customs paperwork.

## Collection
Customers can collect orders from the Letterkenny depot between 9 am and 5 pm.
""",
    "payment_terms": """# Payment terms

## Trade accounts
Trade accounts are invoiced monthly with payment due within 30 days of the
invoice date. The credit limit for a new trade account is EUR 5,000.

## Late payment
Late payments are charged interest at 8% above the ECB reference rate, in line
with the European Communities (Late Payment in Commercial Transactions) Regulations.

## Card and bank transfer
Retail customers pay by card at checkout. Bank transfers must clear before dispatch.
""",
    "vat_and_pricing": """# VAT and pricing

## VAT rate
All list prices exclude VAT. The standard Irish VAT rate of 23% is added at checkout.

## Education discount
Schools and colleges receive a 10% discount on list prices.

## Volume discount
Orders of 500 units or more of a single product receive a 5% volume discount.

## Price changes
List prices are reviewed every quarter and published on the first working day
of January, April, July and October.
""",
    "warranty": """# Warranty

## Standard warranty
Power supplies and displays carry a 2-year warranty. Other products carry a
1-year warranty.

## Making a claim
Warranty claims need the order number and a description of the fault. Claims
are answered within 5 working days.

## Exclusions
The warranty does not cover damage from incorrect wiring, over-voltage or
electrostatic discharge.
""",
    "quality_and_traceability": """# Quality and traceability

## Certification
Northlight Components is certified to ISO 9001:2015.

## Batch traceability
Every reel and tray carries a batch code. Batch records are kept for 7 years.

## Counterfeit prevention
All semiconductors are bought from franchised distributors or the original
manufacturer only.

## Inspection
Incoming goods are sampled under AQL 1.0 general inspection level II.
""",
}


def build(db_path: Path = DB_PATH, docs_dir: Path = DOCS_DIR, seed: int = SEED) -> dict:
    """Create the SQLite database and Markdown documents. Returns row counts."""
    rng = random.Random(seed)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.executescript("""
        CREATE TABLE customers (customer_id INTEGER PRIMARY KEY, name TEXT, county TEXT, segment TEXT);
        CREATE TABLE products (sku TEXT PRIMARY KEY, name TEXT, category TEXT, unit_price REAL);
        CREATE TABLE orders (order_id INTEGER PRIMARY KEY, customer_id INTEGER, order_date TEXT, status TEXT);
        CREATE TABLE order_lines (order_id INTEGER, sku TEXT, quantity INTEGER, unit_price REAL);
    """)
    customers = []
    for cid in range(1, 241):
        county = rng.choice(COUNTIES)
        customers.append((cid, f"Customer {cid:03d}", county, rng.choices(SEGMENTS, [5, 3, 1, 1])[0]))
    cur.executemany("INSERT INTO customers VALUES (?,?,?,?)", customers)
    products = [(sku, name, cat, price) for cat, items in CATEGORIES.items() for sku, name, price in items]
    cur.executemany("INSERT INTO products VALUES (?,?,?,?)", products)
    oid = 10000
    lines = []
    orders = []
    start = date(2026, 1, 1)
    for day in range((date(2026, 10, 31) - start).days + 1):
        d = start + timedelta(days=day)
        if d.weekday() >= 5:
            continue
        for _ in range(rng.randint(2, 7)):
            oid += 1
            status = rng.choices(["delivered", "cancelled", "returned"], [90, 6, 4])[0]
            orders.append((oid, rng.randint(1, 240), d.isoformat(), status))
            for sku, _, _, price in rng.sample(products, rng.randint(1, 3)):
                lines.append((oid, sku, rng.choice([1, 2, 5, 10, 20, 50, 100]), price))
    cur.executemany("INSERT INTO orders VALUES (?,?,?,?)", orders)
    cur.executemany("INSERT INTO order_lines VALUES (?,?,?,?)", lines)
    con.commit()
    con.close()
    docs_dir.mkdir(parents=True, exist_ok=True)
    for name, text in DOCS.items():
        (docs_dir / f"{name}.md").write_text(text, encoding="utf-8")
    return {"customers": len(customers), "products": len(products), "orders": len(orders),
            "order_lines": len(lines), "documents": len(DOCS)}


SCHEMA_TEXT = """customers(customer_id, name, county, segment)
products(sku, name, category, unit_price)
orders(order_id, customer_id, order_date 'YYYY-MM-DD', status in delivered|cancelled|returned)
order_lines(order_id, sku, quantity, unit_price)"""


if __name__ == "__main__":
    print(build())
