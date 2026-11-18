"""Generate the agent task sets with gold tool plans.

Every task has a question, a gold plan (list of tool calls) and a gold answer
obtained by EXECUTING the gold plan, so the answer key cannot drift from the data.

Splits
- train:    phrasings 0-3 of each family, training entities
- test_id:  phrasings 0-3, training entities, but question strings never seen in train
- test_ood: phrasings 4-5 (never trained on) and held-out entities
            (counties Kildare / Wexford, four held-out products)
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from business_db import CATEGORIES, COUNTIES, SEGMENTS
from executor import run_plan

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data"
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October"]
HELD_COUNTIES = ["Kildare", "Wexford"]
TRAIN_COUNTIES = [c for c in COUNTIES if c not in HELD_COUNTIES]
PRODUCTS = [(sku, name, cat) for cat, items in CATEGORIES.items() for sku, name, _ in items]
HELD_SKUS = {"SN-045", "DS-070", "PW-005", "CB-050"}
# families with few distinct questions: cap training so test_id still gets unseen questions
TRAIN_CAP = {"top_product": 28, "price_lookup": 30}


def mm(month: str) -> str:
    return f"{MONTHS.index(month) + 1:02d}"


def _q_count(county, month):
    return ("SELECT COUNT(*) FROM orders o JOIN customers c ON o.customer_id = c.customer_id "
            f"WHERE c.county = '{county}' AND strftime('%m', o.order_date) = '{mm(month)}'")


def _q_rev(cat, month):
    return ("SELECT ROUND(SUM(l.quantity * l.unit_price), 2) FROM order_lines l "
            "JOIN orders o ON l.order_id = o.order_id JOIN products p ON l.sku = p.sku "
            f"WHERE p.category = '{cat}' AND o.status = 'delivered' AND strftime('%m', o.order_date) = '{mm(month)}'")


def _q_top(month):
    return ("SELECT p.name, SUM(l.quantity) AS units FROM order_lines l "
            "JOIN orders o ON l.order_id = o.order_id JOIN products p ON l.sku = p.sku "
            f"WHERE o.status = 'delivered' AND strftime('%m', o.order_date) = '{mm(month)}' "
            "GROUP BY p.sku ORDER BY units DESC LIMIT 1")


def _q_price(name):
    return f"SELECT unit_price FROM products WHERE name = '{name}'"


def _q_seg(seg, county):
    return f"SELECT COUNT(*) FROM customers WHERE segment = '{seg}' AND county = '{county}'"


def sql(q):
    return {"tool": "sql", "args": {"query": q}}


def calc(e):
    return {"tool": "calculator", "args": {"expression": e}}


def docs(q):
    return {"tool": "search_docs", "args": {"query": q}}


# Each family: phrasings (0-3 train, 4-5 held-out), a sampler of slot values and a plan builder.
FAMILIES = {
    "count_orders": {
        "phrasings": ["How many orders did customers in {county} place in {month}?",
                      "Number of orders from {county} customers in {month}?",
                      "Count the orders placed by {county} customers during {month}.",
                      "In {month}, how many orders came from customers based in {county}?",
                      "What was the order count for {county} in {month}?",
                      "Tell me how many {county} orders we had in {month}."],
        "plan": lambda s: [sql(_q_count(s["county"], s["month"]))],
        "slots": ["county", "month"],
    },
    "revenue_category": {
        "phrasings": ["What was the revenue from {category} in {month}?",
                      "How much revenue did {category} bring in during {month}?",
                      "Total {category} sales value for {month}?",
                      "In {month}, what were our {category} revenues?",
                      "Give me {month} revenue for the {category} category.",
                      "How much did we sell in {category} in {month}, in euro?"],
        "plan": lambda s: [sql(_q_rev(s["category"], s["month"]))],
        "slots": ["category", "month"],
    },
    "top_product": {
        "phrasings": ["Which product sold the most units in {month}?",
                      "What was our best-selling product by units in {month}?",
                      "Top product by quantity sold in {month}?",
                      "In {month}, which item had the highest unit sales?",
                      "Name the product with the most units shipped in {month}.",
                      "What sold best in {month} by volume?"],
        "plan": lambda s: [sql(_q_top(s["month"]))],
        "slots": ["month"],
    },
    "price_lookup": {
        "phrasings": ["What is the list price of the {product}?",
                      "How much does the {product} cost?",
                      "Price of the {product}?",
                      "What do we charge for one {product}?",
                      "Look up the unit price for the {product}.",
                      "How much is a single {product} before VAT?"],
        "plan": lambda s: [sql(_q_price(s["product"]))],
        "slots": ["product"],
    },
    "segment_count": {
        "phrasings": ["How many {segment} customers are in {county}?",
                      "Number of {segment} customers based in {county}?",
                      "Count our {segment} customers in {county}.",
                      "In {county}, how many customers are in the {segment} segment?",
                      "How many customers in {county} belong to the {segment} segment?",
                      "What is the {segment} customer count for {county}?"],
        "plan": lambda s: [sql(_q_seg(s["segment"], s["county"]))],
        "slots": ["segment", "county"],
    },
    "vat_calc": {
        "phrasings": ["What is EUR {amount} including 23% VAT?",
                      "Add VAT at 23% to EUR {amount}.",
                      "EUR {amount} plus 23% VAT comes to how much?",
                      "Gross price for a net amount of EUR {amount} at 23% VAT?",
                      "If the net price is EUR {amount}, what is it with 23% VAT on top?",
                      "Work out EUR {amount} with VAT of 23% added."],
        "plan": lambda s: [calc(f"{s['amount']} * 1.23")],
        "slots": ["amount"],
    },
    "discount_calc": {
        "phrasings": ["What is EUR {amount} after a {pct}% discount?",
                      "Take {pct}% off EUR {amount}.",
                      "Apply a {pct}% discount to EUR {amount}.",
                      "EUR {amount} less {pct}% is how much?",
                      "If we discount EUR {amount} by {pct}%, what is the new price?",
                      "Reduce EUR {amount} by {pct} percent."],
        "plan": lambda s: [calc(f"{s['amount']} * (1 - {s['pct']} / 100)")],
        "slots": ["amount", "pct"],
    },
    "revenue_incl_vat": {
        "phrasings": ["What was {category} revenue in {month} including 23% VAT?",
                      "{month} {category} revenue with 23% VAT added?",
                      "How much did {category} bring in during {month}, gross of 23% VAT?",
                      "Give me {category} sales for {month} plus 23% VAT.",
                      "Including VAT at 23%, what was the {month} revenue from {category}?",
                      "Total {category} revenue in {month} after adding 23% VAT?"],
        "plan": lambda s: [sql(_q_rev(s["category"], s["month"])), calc("{0} * 1.23")],
        "slots": ["category", "month"],
    },
    "order_total": {
        "phrasings": ["What would {qty} units of the {product} cost including 23% VAT?",
                      "Price for {qty} x {product} with VAT at 23%?",
                      "Quote {qty} of the {product}, VAT included at 23%.",
                      "How much are {qty} {product} units after 23% VAT?",
                      "If a customer buys {qty} of the {product}, what is the total with 23% VAT?",
                      "Total cost, with 23% VAT, of {qty} pieces of the {product}?"],
        "plan": lambda s: [sql(_q_price(s["product"])), calc(f"{{0}} * {s['qty']} * 1.23")],
        "slots": ["product", "qty"],
    },
}

# Document facts: (doc, section, key phrase in the passage, six phrasings, search query)
DOC_FACTS = [
    ("returns_policy", "Return window", "30 days", "return window for unused items",
     ["How long do customers have to return unused items?", "What is the return period for unused goods?",
      "How many days can a customer take to send back an unused item?", "What's the standard returns window?",
      "Until when can unused products be returned for a refund?", "Is there a deadline for returning unused stock?"]),
    ("returns_policy", "Restocking fee", "15%", "restocking fee",
     ["Is there a restocking fee?", "What fee applies to returned custom-cut cables?",
      "Do we charge for restocking returns?", "How much is the restocking charge?",
      "What percentage is deducted when custom cables are returned?", "Any charge for sending back cut-to-length cable?"]),
    ("returns_policy", "Faulty goods", "14 days", "report faulty goods",
     ["How soon must a fault be reported?", "What is the deadline to report faulty goods?",
      "Within how many days should customers report a faulty item?", "When do faults need to be reported for a free label?",
      "If something arrives broken, how long do I have to tell you?", "Time limit for flagging a defective delivery?"]),
    ("shipping_policy", "Delivery charges", "EUR 150", "free delivery threshold",
     ["When is delivery free?", "What order value gets free shipping?", "What is the free delivery threshold?",
      "How much do I need to spend for free delivery?", "Above what amount do you waive the delivery charge?",
      "Is shipping ever free, and from what spend?"]),
    ("shipping_policy", "Delivery times", "2 pm", "same day dispatch cut-off",
     ["What is the cut-off time for same-day dispatch?", "By what time must I order to ship today?",
      "When is the daily dispatch cut-off?", "What time do orders need to be in for same-day dispatch?",
      "How late can I order and still have it sent out the same day?", "Latest order time for dispatch today?"]),
    ("shipping_policy", "Northern Ireland and Great Britain", "3 to 5 working days", "delivery to Great Britain",
     ["How long does delivery to Great Britain take?", "What are delivery times to GB?",
      "How many days to ship to Great Britain?", "Delivery lead time for customers in Britain?",
      "If I'm in England, when will my order arrive?", "How quickly do you deliver to the UK mainland?"]),
    ("payment_terms", "Trade accounts", "30 days", "trade account payment terms",
     ["What are the payment terms for trade accounts?", "When are trade invoices due?",
      "How long do trade customers have to pay?", "What credit terms do trade accounts get?",
      "How many days after invoice must a trade account settle?", "Trade customers: when is payment expected?"]),
    ("payment_terms", "Trade accounts", "EUR 5,000", "credit limit new trade account",
     ["What is the credit limit for a new trade account?", "How much credit does a new trade account get?",
      "Starting credit limit for trade customers?", "What's the initial trade credit limit?",
      "When we open a trade account, how much credit is allowed?", "Maximum credit on a brand-new trade account?"]),
    ("payment_terms", "Late payment", "8%", "late payment interest",
     ["What interest is charged on late payments?", "What is the late payment interest rate?",
      "How much interest do overdue invoices attract?", "What penalty rate applies to late payment?",
      "If a customer pays late, what interest do we add?", "Overdue account interest: what rate?"]),
    ("vat_and_pricing", "Education discount", "10%", "education discount schools",
     ["What discount do schools get?", "Is there an education discount?", "How much off do colleges receive?",
      "What's the discount for educational customers?", "Do universities or schools get cheaper prices, and by how much?",
      "Percentage reduction for education buyers?"]),
    ("warranty", "Standard warranty", "2-year", "warranty power supplies displays",
     ["How long is the warranty on power supplies?", "What warranty do displays carry?",
      "Warranty period for PSUs and displays?", "How many years of warranty on a display?",
      "If I buy a power supply, how long is it covered?", "What guarantee comes with a screen or PSU?"]),
    ("quality_and_traceability", "Batch traceability", "7 years", "batch records kept",
     ["How long are batch records kept?", "What is the retention period for batch records?",
      "For how many years do you keep traceability records?", "How long is batch traceability data stored?",
      "If I need a batch record from years ago, how far back do you keep them?", "Batch history retention time?"]),
]


def _sample(slot, rng, held):
    if slot == "county":
        return rng.choice(HELD_COUNTIES if held else TRAIN_COUNTIES)
    if slot == "month":
        return rng.choice(MONTHS)
    if slot == "category":
        return rng.choice(list(CATEGORIES))
    if slot == "segment":
        return rng.choice(SEGMENTS)
    if slot == "product":
        pool = [n for s, n, _ in PRODUCTS if (s in HELD_SKUS) == held]
        return rng.choice(pool)
    if slot == "amount":
        return rng.choice([rng.randint(20, 5000), round(rng.uniform(10, 2000), 2)])
    if slot == "pct":
        return rng.choice([5, 10, 12, 15, 20, 25, 30])
    if slot == "qty":
        return rng.choice([3, 5, 10, 12, 25, 40, 50, 100, 250])
    raise KeyError(slot)


def make_task(family, phr_idx, slots, tid):
    f = FAMILIES[family]
    q = f["phrasings"][phr_idx].format(**slots)
    q = q[0].upper() + q[1:]
    plan = f["plan"](slots)
    return {"id": tid, "family": family, "phrasing": phr_idx, "question": q, "plan": plan,
            "gold": run_plan(plan)["final"]}


def make_doc_task(i, phr_idx, tid):
    doc, sec, key, query, phr = DOC_FACTS[i]
    plan = [docs(query)]
    return {"id": tid, "family": "doc_lookup", "phrasing": phr_idx, "question": phr[phr_idx], "plan": plan,
            "gold": {"doc": doc, "section": sec, "key": key}}


def generate(seed: int = 16, n_train_per_family: int = 90, n_test_per_family: int = 14):
    rng = random.Random(seed)
    splits = {"train": [], "test_id": [], "test_ood": []}
    seen = set()
    for fam, f in FAMILIES.items():
        n_train = TRAIN_CAP.get(fam, n_train_per_family)
        for split, n, phr_range, held in (("train", n_train, range(4), False),
                                          ("test_id", n_test_per_family, range(4), False),
                                          ("test_ood", n_test_per_family, range(4, 6), True)):
            made, tries = 0, 0
            while made < n and tries < 5000:
                tries += 1
                slots = {s: _sample(s, rng, held and s in ("county", "product")) for s in f["slots"]}
                phr = rng.choice(list(phr_range))
                q = f["phrasings"][phr].format(**slots)
                if q in seen:
                    continue
                seen.add(q)
                splits[split].append(make_task(fam, phr, slots, f"{split}-{fam}-{made}"))
                made += 1
    # documents: phrasings 0-3 for train and test_id (test_id uses phrasings unseen *for that fact*
    # is impossible with 4, so test_id re-asks with a different trained phrasing), 4-5 for ood
    for i in range(len(DOC_FACTS)):
        order = rng.sample(range(4), 4)
        for p in order[:3]:
            for rep in range(3):  # repeat so documents carry similar weight in training
                splits["train"].append(make_doc_task(i, p, f"train-doc-{i}-{p}-{rep}"))
        splits["test_id"].append(make_doc_task(i, order[3], f"test_id-doc-{i}"))
        for p in (4, 5):
            splits["test_ood"].append(make_doc_task(i, p, f"test_ood-doc-{i}-{p}"))
    for v in splits.values():
        rng.shuffle(v)
    return splits


def main():
    splits = generate()
    for name, rows in splits.items():
        with open(OUT / f"tasks_{name}.jsonl", "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        print(name, len(rows))


if __name__ == "__main__":
    main()
