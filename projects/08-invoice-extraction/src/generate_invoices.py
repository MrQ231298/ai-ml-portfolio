"""
Generate SYNTHETIC supplier invoices as PDFs, with a ground-truth CSV.

Every company, VAT number, address and amount is invented. Each PDF
carries a visible "SYNTHETIC EXAMPLE" footer.

Four layouts imitate four different suppliers:
  A, B, C  - "development" layouts; the rule-based extractor and the ML
             model are built and trained on these only.
  D        - held-out layout with different labels, date style and
             positions, used to test generalisation to an unseen supplier.

    python src/generate_invoices.py
"""

from __future__ import annotations

import random
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "invoices_pdf"
TRUTH = ROOT / "data" / "SYNTHETIC_invoice_ground_truth.csv"
SEED = 7
PER_TEMPLATE = 60

PREFIX = ["Atlantic", "Shannon", "Corrib", "Liffey", "Burren", "Errigal", "Slaney", "Boyne",
          "Moher", "Barrow", "Suir", "Nephin", "Galtee", "Mourne", "Foyle", "Wicklow", "Kerry Way"]
SUFFIX = ["Supplies", "Logistics", "Office Solutions", "Catering", "Engineering", "Print Works",
          "Facilities", "IT Services", "Hardware", "Packaging", "Cleaning Co.", "Fleet Parts"]
FORM = ["Ltd", "Limited", "Teoranta", "DAC"]
TOWNS = ["Galway", "Limerick", "Sligo", "Athlone", "Letterkenny", "Waterford", "Tralee", "Navan", "Carlow"]
ITEMS = ["A4 paper (box)", "Toner cartridge", "Delivery charge", "Consultancy (hours)", "Pallet wrap",
         "Hi-vis vests", "Network switch", "Cleaning service (visit)", "Sandwich platter", "Brochure print run",
         "Brake pads (set)", "Labour (hours)", "Shelving unit", "Laptop stand", "USB-C dock"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]


def vat_number(rng: random.Random) -> str:
    return f"IE{rng.randint(1000000, 9999999)}{rng.choice('ABCDEFGHJKLMNPRSTW')}{rng.choice(['', 'H', 'A'])}"


def money(x: float) -> str:
    return f"{x:,.2f}"


def fmt_date(d: date, style: str) -> str:
    return {
        "A": d.strftime("%d/%m/%Y"),
        "B": d.strftime("%d %b %Y"),
        "C": d.isoformat(),
        "D": f"{MONTHS[d.month - 1]} {d.day}, {d.year}",
    }[style]


def make_record(rng: random.Random, template: str, i: int) -> dict:
    vendor = f"{rng.choice(PREFIX)} {rng.choice(SUFFIX)} {rng.choice(FORM)}"
    issued = date(2025, 1, 1) + timedelta(days=rng.randint(0, 680))
    due = issued + timedelta(days=rng.choice([14, 21, 30, 45, 60]))
    n_items = rng.randint(1, 6)
    lines = []
    for _ in range(n_items):
        qty = rng.randint(1, 20)
        unit = round(rng.uniform(3, 450), 2)
        lines.append((rng.choice(ITEMS), qty, unit, round(qty * unit, 2)))
    net = round(sum(l[3] for l in lines), 2)
    rate = rng.choice([0.23, 0.23, 0.23, 0.135])
    vat = round(net * rate, 2)
    total = round(net + vat, 2)
    inv_no = {
        "A": f"INV-{issued.year}-{rng.randint(1, 9999):04d}",
        "B": f"{rng.randint(1000, 999999):06d}",
        "C": f"{rng.choice('ABCDEFGHKMNPRS')}{rng.choice('ABCDEFGHKMNPRS')}-{rng.randint(10000, 99999)}",
        "D": f"{issued.year}/{rng.randint(1, 99999):05d}",
    }[template]
    return {
        "file": f"{template}_{i:03d}.pdf", "template": template, "vendor_name": vendor,
        "vendor_vat_number": vat_number(rng), "invoice_number": inv_no,
        "invoice_date": issued.isoformat(), "due_date": due.isoformat(),
        "net_amount": net, "vat_amount": vat, "total_amount": total, "vat_rate": rate,
        "_lines": lines, "_phone": f"0{rng.choice([1, 21, 61, 71, 74, 91])} {rng.randint(100000, 9999999)}",
        "_other_ref": {"A": f"CR-{rng.randint(10000, 99999)}", "B": f"45{rng.randint(10000, 99999)}",
                       "C": f"AC{rng.randint(100000, 999999)}", "D": f"ORD-{rng.randint(10000, 99999)}"}[template], "_town": rng.choice(TOWNS), "_cust": f"{rng.choice(PREFIX)} {rng.choice(['Hotel', 'Garage', 'Clinic', 'Stores', 'GAA Club'])}",
    }


def draw_lines(c, rec, x, y, headers):
    c.setFont("Helvetica-Bold", 9)
    cols = [x, x + 230, x + 290, x + 370]
    for cx, h in zip(cols, headers):
        c.drawString(cx, y, h)
    c.setFont("Helvetica", 9)
    for desc, qty, unit, amt in rec["_lines"]:
        y -= 14
        c.drawString(cols[0], y, desc); c.drawString(cols[1], y, str(qty))
        c.drawString(cols[2], y, money(unit)); c.drawString(cols[3], y, money(amt))
    return y


def render(rec: dict, path: Path) -> None:
    t, c = rec["template"], canvas.Canvas(str(path), pagesize=A4)
    W, Hh = A4
    d_iss = fmt_date(date.fromisoformat(rec["invoice_date"]), t)
    d_due = fmt_date(date.fromisoformat(rec["due_date"]), t)
    pct = f"{rec['vat_rate'] * 100:g}%"
    if t == "A":
        c.setFont("Helvetica-Bold", 14); c.drawString(50, Hh - 60, rec["vendor_name"])
        c.setFont("Helvetica", 9); c.drawString(50, Hh - 75, f"Unit 4, Business Park, {rec['_town']}")
        c.drawString(50, Hh - 88, f"VAT No: {rec['vendor_vat_number']}")
        c.setFont("Helvetica-Bold", 20); c.drawString(400, Hh - 60, "INVOICE")
        c.setFont("Helvetica", 10)
        c.drawString(380, Hh - 90, f"Invoice No: {rec['invoice_number']}")
        c.drawString(380, Hh - 104, f"Invoice Date: {d_iss}")
        c.drawString(380, Hh - 118, f"Due Date: {d_due}")
        c.drawString(50, Hh - 130, f"Bill to: {rec['_cust']}")
        c.drawString(50, Hh - 144, f"Customer Ref: {rec['_other_ref']}   Tel: {rec['_phone']}")
        y = draw_lines(c, rec, 50, Hh - 170, ["Description", "Qty", "Unit", "Amount"])
        c.drawString(340, y - 30, "Subtotal:"); c.drawRightString(500, y - 30, money(rec["net_amount"]))
        c.drawString(340, y - 44, f"VAT @ {pct}:"); c.drawRightString(500, y - 44, money(rec["vat_amount"]))
        c.setFont("Helvetica-Bold", 11)
        c.drawString(340, y - 62, "Total Due:"); c.drawRightString(500, y - 62, "€" + money(rec["total_amount"]))
    elif t == "B":
        c.setFont("Helvetica-Bold", 16); c.drawCentredString(W / 2, Hh - 50, rec["vendor_name"])
        c.setFont("Helvetica", 9)
        c.drawCentredString(W / 2, Hh - 64, f"Main Street, {rec['_town']} | VAT Reg: {rec['vendor_vat_number']}")
        c.setFont("Helvetica-Bold", 12); c.drawString(50, Hh - 100, "Tax Invoice")
        c.setFont("Helvetica", 10)
        c.drawString(50, Hh - 118, f"Invoice #: {rec['invoice_number']}")
        c.drawString(250, Hh - 118, f"Date: {d_iss}")
        c.drawString(50, Hh - 132, f"Customer: {rec['_cust']}")
        c.drawString(250, Hh - 132, f"Payment due: {d_due}")
        c.drawString(50, Hh - 146, f"PO number: {rec['_other_ref']}")
        c.drawString(250, Hh - 146, f"Phone: {rec['_phone']}")
        y = draw_lines(c, rec, 50, Hh - 170, ["Item", "Qty", "Price", "Line total"])
        c.line(50, y - 10, 520, y - 10)
        c.drawString(50, y - 28, f"Net amount  {money(rec['net_amount'])}")
        c.drawString(200, y - 28, f"VAT ({pct})  {money(rec['vat_amount'])}")
        c.setFont("Helvetica-Bold", 11)
        c.drawString(350, y - 28, f"Amount payable  EUR {money(rec['total_amount'])}")
    elif t == "C":
        c.setFont("Helvetica-Bold", 13); c.drawString(50, Hh - 55, rec["vendor_name"].upper())
        c.setFont("Helvetica", 9)
        c.drawString(50, Hh - 70, f"{rec['_town']}, Ireland"); c.drawString(50, Hh - 83, f"Tax ID: {rec['vendor_vat_number']}")
        c.drawString(50, Hh - 96, f"Tel {rec['_phone']}")
        c.drawString(50, Hh - 140, f"Account: {rec['_other_ref']}")
        c.rect(360, Hh - 130, 180, 75)
        c.setFont("Helvetica", 10)
        c.drawString(370, Hh - 72, f"Ref: {rec['invoice_number']}")
        c.drawString(370, Hh - 90, f"Issued: {d_iss}")
        c.drawString(370, Hh - 108, f"Due: {d_due}")
        y = draw_lines(c, rec, 50, Hh - 180, ["Description", "Units", "Rate", "Total"])
        c.drawString(360, y - 30, "Goods total"); c.drawRightString(530, y - 30, money(rec["net_amount"]))
        c.drawString(360, y - 44, f"Tax {pct}"); c.drawRightString(530, y - 44, money(rec["vat_amount"]))
        c.setFont("Helvetica-Bold", 11)
        c.drawString(360, y - 60, "Grand Total"); c.drawRightString(530, y - 60, money(rec["total_amount"]))
    else:  # D: held-out layout
        c.setFont("Helvetica", 10)
        c.drawString(50, Hh - 50, f"Document number: {rec['invoice_number']}")
        c.drawString(50, Hh - 64, f"Dated: {d_iss}")
        c.drawString(50, Hh - 78, f"Pay by: {d_due}")
        c.setFont("Helvetica-Bold", 12); c.drawRightString(540, Hh - 50, rec["vendor_name"])
        c.setFont("Helvetica", 9)
        c.drawRightString(540, Hh - 64, f"{rec['_town']} | VAT number {rec['vendor_vat_number']}")
        c.drawString(50, Hh - 110, f"Supplied to: {rec['_cust']}   Order ref: {rec['_other_ref']}")
        c.drawRightString(540, Hh - 78, f"Tel. {rec['_phone']}")
        y = draw_lines(c, rec, 50, Hh - 150, ["Goods / services", "No.", "Each", "Value"])
        c.drawString(300, y - 30, "Total excl. VAT"); c.drawRightString(540, y - 30, money(rec["net_amount"]))
        c.drawString(300, y - 44, "VAT amount"); c.drawRightString(540, y - 44, money(rec["vat_amount"]))
        c.setFont("Helvetica-Bold", 11)
        c.drawString(300, y - 62, "Balance to pay"); c.drawRightString(540, y - 62, "EUR " + money(rec["total_amount"]))
    c.setFont("Helvetica-Oblique", 7)
    c.drawString(50, 30, "SYNTHETIC EXAMPLE INVOICE - invented for an ML portfolio; not a real company or transaction.")
    c.save()


def main() -> None:
    rng = random.Random(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    recs = []
    for t in "ABCD":
        for i in range(PER_TEMPLATE):
            r = make_record(rng, t, i)
            render(r, OUT_DIR / r["file"])
            recs.append({k: v for k, v in r.items() if not k.startswith("_")})
    df = pd.DataFrame(recs)
    df["source_flag"] = "SYNTHETIC_EXAMPLE_DATA"
    df.to_csv(TRUTH, index=False)
    print(f"wrote {len(df)} invoices to {OUT_DIR} and {TRUTH}")


if __name__ == "__main__":
    main()
