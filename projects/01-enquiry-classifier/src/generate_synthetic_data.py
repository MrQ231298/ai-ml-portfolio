"""
Generate a SYNTHETIC customer-enquiry dataset for a fictional B2B tyre /
automotive-parts sales desk.

IMPORTANT
---------
Every row in the output file is invented example text. It is not taken from
any real inbox, CRM, or customer record. Filenames, the CSV itself, and the
project docs are all labelled SYNTHETIC so this can never be mistaken for
live personal data.

Why generate data at all?
    A portfolio / RPL artefact must not contain real customer messages
    (GDPR / confidentiality). Synthetic text lets us practise the full
    supervised-learning workflow and still be honest about the limits of
    the data (see the project README critical evaluation).

Usage
-----
    python src/generate_synthetic_data.py
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

# Fixed seed so the dataset is reproducible when the script is re-run.
RANDOM_SEED = 42

# Six business classes a sales administrator would typically route.
CLASSES = (
    "quote_request",
    "stock_availability",
    "order_status",
    "complaint_return",
    "account_billing",
    "other",
)

# Mildly imbalanced counts, roughly how a real tyre desk inbox might look
# (quotes and stock checks dominate; "other" is a smaller leftover bucket).
# Total = 540 messages, inside the requested 400-600 range.
TARGET_COUNTS = {
    "quote_request": 120,
    "stock_availability": 110,
    "order_status": 85,
    "complaint_return": 80,
    "account_billing": 80,
    "other": 65,
}

# ---------------------------------------------------------------------------
# Shared vocabulary (all fictional)
# ---------------------------------------------------------------------------

# Fictional trade-account names — not real customers.
ACCOUNT_NAMES = [
    "Ashfield Motors",
    "Ballymore Tyre Bay",
    "Cedar Hill Garage",
    "Drogheda Fleet Care",
    "East Pier Autocare",
    "Fairgreen Commercials",
    "Glenbrook Service Station",
    "Harbour Road Motors",
    "Inishowen Van Centre",
    "Kilbride Truck Services",
    "Liffey Side Autos",
    "Millfield Recovery",
    "North Quay Recoveries",
    "Oakleaf Body Shop",
    "Portside Plant Hire",
    "Quay Street MOT",
    "Riverside Couriers",
    "Stonebridge Haulage",
    "Tullow Farm Machinery",
    "Westgate Taxis",
    "Yellowline Logistics",
    "Ardmore Panel Works",
    "Blackwater Buses",
    "Cairn Hill Plant",
    "Dunlewey Garage",
]

FIRST_NAMES = [
    "Aoife", "Brian", "Ciara", "Declan", "Eimear", "Fergal", "Grainne",
    "Hugh", "Iseult", "James", "Kate", "Liam", "Maeve", "Niall", "Orla",
    "Padraig", "Roisin", "Sean", "Tara", "Una", "Conor", "Deirdre",
    "Eoin", "Fiona", "Gerard",
]

TYRE_SIZES = [
    "165/70R14", "175/65R14", "185/65R15", "195/65R15", "205/55R16",
    "205/60R16", "215/55R17", "225/45R17", "225/50R17", "235/45R18",
    "235/55R18", "245/40R18", "255/35R19", "255/40R19", "265/65R17",
    "275/40R20", "195/70R15C", "205/65R16C", "215/65R16C", "235/65R16C",
    "245/70R17.5", "265/70R19.5", "295/80R22.5", "315/80R22.5",
    "385/65R22.5", "425/65R22.5",
]

# Widely sold public product names, used only as catalogue examples.
BRANDS = [
    "Michelin", "Continental", "Pirelli", "Goodyear", "Bridgestone",
    "Dunlop", "Hankook", "Toyo", "Falken", "Kumho", "Maxxis",
    "Triangle", "Linglong", "Sailun", "Yokohama",
]

SEASONS = ["summer", "winter", "all-season", "all-weather", "mud-terrain", "highway"]
QTY = [2, 4, 6, 8, 10, 12, 16, 20, 24, 40, 50]
AXLES = ["steer", "drive", "trailer", "all-round"]

# Fictional order / invoice / account references (not real documents).
def order_ref() -> str:
    style = random.choice(["SO", "ORD", "WEB", "PO"])
    return f"{style}-{random.randint(10000, 99999)}"


def invoice_ref() -> str:
    return f"INV-{random.randint(2024, 2026)}-{random.randint(1000, 9999)}"


def account_ref() -> str:
    return f"ACC-{random.randint(1000, 8999)}"


def person() -> str:
    return random.choice(FIRST_NAMES)


def account() -> str:
    return random.choice(ACCOUNT_NAMES)


def size() -> str:
    return random.choice(TYRE_SIZES)


def brand() -> str:
    return random.choice(BRANDS)


def season() -> str:
    return random.choice(SEASONS)


def qty() -> int:
    return random.choice(QTY)


def axle() -> str:
    return random.choice(AXLES)


GREETINGS = [
    "Hi team,",
    "Hello,",
    "Hi there,",
    "Good morning,",
    "Good afternoon,",
    "Hi,",
    "Afternoon,",
    "",
]

SIGN_OFFS = [
    "Thanks",
    "Thanks,",
    "Many thanks",
    "Cheers",
    "Kind regards",
    "Regards",
    "Thanks in advance",
    "Appreciate it",
    "",
]


def wrap(body: str) -> str:
    """Optionally add a greeting, sign-off and name so messages feel like email."""
    parts = []
    greet = random.choice(GREETINGS)
    if greet:
        parts.append(greet)
    parts.append(body.strip())
    sign = random.choice(SIGN_OFFS)
    if sign:
        if random.random() < 0.6:
            parts.append(f"{sign},\n{person()}")
        else:
            parts.append(sign)
    text = "\n".join(parts).strip()
    # Light realistic noise: collapse some line breaks, extra spaces, case tweaks.
    if random.random() < 0.12:
        text = text.replace("\n", " ")
    if random.random() < 0.08:
        text = text.replace("  ", " ")
    if random.random() < 0.05:
        text = text.upper()
    return " ".join(text.split()) if random.random() < 0.15 else text


# ---------------------------------------------------------------------------
# Per-class template banks
# Each function returns a list of message strings for that class.
# ---------------------------------------------------------------------------

def templates_quote_request() -> list[str]:
    s, b, n, ax, a = size(), brand(), qty(), axle(), account()
    return [
        f"Can you price {n} x {s} {b} {season()} tyres, delivered to {a} please?",
        f"Looking for a quote on {n} {s} {season()} tyres. Budget fitment, not looking for premium unless the difference is small.",
        f"Please quote {n} {b} {s} for our {ax} axle. Need them this week if possible.",
        f"We need pricing for a set of {s} — {n} tyres. Can you include fitting and disposal?",
        f"Quote please: {n} pcs {s} {b}. Trade rate, account {account_ref()}.",
        f"Could you send a trade price for {n} {s} tyres, {b} or equivalent? Delivery to {a}.",
        f"Hi, can I get a price on {n} {season()} tyres in {s}? Prefer {b} but open to alternatives.",
        f"Need a written quote for a fleet changeover: {n} x {s} {ax} tyres. Can you also price a mid-range option?",
        f"What's your best price on {b} {s}? We'd take {n} if the rate is sharp.",
        f"Please quote supply only for {n} {s} {season()} tyres and a separate line for delivery to {a}.",
        f"Can you price a mixed order: {n} x {s} {b} and {qty()} x {size()} {brand()}?",
        f"Looking for commercial pricing on {s} {ax} tyres, quantity {n}. When could you deliver?",
        f"We have a customer waiting — please quote {n} {s} {b} {season()} and an economy alternative.",
        f"Requesting a quote for {n} steer tyres in {s}, preferably {b}. Account name {a}.",
        f"Any chance of a price on {n} {s}? Also tell me the lead time and whether you have a matching {ax} option.",
        f"Can you do a package price for {n} tyres size {s} plus {qty()} tubes if needed?",
        f"Trade quote wanted: {b} {s} x {n}. Please show net and VAT.",
        f"How much for {n} {s} all-round? We're {a} — existing account.",
        f"Please price {n} x {s} {season()} and confirm if a volume break applies above {n}.",
        f"Could I have a same-day quote for {n} {b} {s} tyres? Collection from your yard is fine.",
        f"Need costs for replacing a full set, size {s}, brand {b} or similar, qty {n}.",
        f"Quote for {n} {s} winter tyres before the weather turns. Delivery to {a} midweek if you can.",
        f"What would {n} {s} {b} come to on our trade account, supply only?",
        f"Can you send pricing and availability together for {n} x {s} {season()} {b}?",
        f"We're tendering a local authority van fleet — please quote {n} {s} {season()} tyres, 12-month hold if possible.",
        f"Price check: {s} {b}, quantity {n}. If you can beat our current supplier we'll move the account.",
        f"Hi, would you quote {n} drive-axle {s} and {qty()} trailer {size()} for {a}?",
        f"Please cost up {n} {s} tyres. We need a formal quote we can put on a PO.",
        f"Can I get a ballpark on {n} {s} {season()} tyres and a firm figure tomorrow?",
        f"Looking for your trade rate on {b} {s}. We'll need {n} initially and maybe a top-up next month.",
    ]


def templates_stock_availability() -> list[str]:
    s, b, n, a = size(), brand(), qty(), account()
    return [
        f"Have you got {n} x {s} {b} in stock today?",
        f"Do you hold {s} {season()} tyres at the moment? Need {n} if you have them.",
        f"Any {b} {s} on the shelf, or is it a factory order?",
        f"Can you check stock on {s} for me? Looking for {n} {axle()} tyres.",
        f"Are {s} {season()} tyres available for collection this afternoon?",
        f"Quick stock check: {n} {s} {b}. If not, what's the soonest ETA?",
        f"Have ye any {s} in at the minute? Not fussy on brand if you can do {n} today.",
        f"Is the {b} {s} in your warehouse or only at the other depot?",
        f"Do you have matching pairs in {s}? We need {n} and they have to be the same DOT date if possible.",
        f"Stock enquiry for {a}: {n} x {s} {season()}. Can you reserve them if they're there?",
        f"Just checking whether {b} {s} is available before I send a PO for {n}.",
        f"Have you {n} {s} commercial tyres in stock for a morning collection?",
        f"What's current availability like on {s}? We might need {n} later this week.",
        f"Can you confirm if {b} {s} {season()} is in stock and how many you can release today?",
        f"Any chance you have {s} in a budget brand? Need {n} ASAP.",
        f"Do you stock {s} TPMS-compatible {season()} tyres? Quantity {n}.",
        f"Are you out of {b} {s} again or have they come in since last week?",
        f"Please check your system for {n} x {s} {axle()} tyres — customer is in the yard.",
        f"Have you got run-flats in {s}? Need a pair if not a full set of {n}.",
        f"Is {s} {b} a stocked line for you, or special order only?",
        f"Can you hold {n} {s} {season()} until tomorrow if they're in stock?",
        f"We're {a} — any {s} heavy-duty van tyres on hand? Need {n}.",
        f"Quick one: stock on {s} and {size()}? Might take both if you have them.",
        f"Do you have {n} {s} in black letters, not white? {b} preferred.",
        f"Availability please for {b} {s}, qty {n}, and an equivalent if you're short.",
        f"Have you received the container of {s} yet, or are we still waiting?",
        f"Need to know if {n} x {s} {season()} can go out on today's van to {a}.",
        f"Are winter {s} still in stock or have they sold through?",
        f"Can someone check the rack for {s}? Screen says {n} but I was told otherwise yesterday.",
        f"Looking to see if you can supply {n} {s} {b} from stock before Friday.",
    ]


def templates_order_status() -> list[str]:
    o, a, s, n = order_ref(), account(), size(), qty()
    return [
        f"Any update on {o}? We were expecting delivery yesterday.",
        f"Can you check the status of order {o} for {a} please?",
        f"Has {o} been dispatched yet? We need the {s} typres for a job in the morning.",
        f"Where is our delivery for {o}? Tracking hasn't moved all day.",
        f"Hi, just chasing order {o}. Your email said it would ship on Monday.",
        f"Please confirm whether {o} is still on this afternoon's run.",
        f"Order {o} — {n} x {s} — has it left the depot?",
        f"We placed {o} two days ago and haven't had a confirmation. Can you look it up?",
        f"What's the ETA on {o}? The vehicle is up on the ramp.",
        f"Has the back order on {o} come in yet?",
        f"Can you tell me if {o} will arrive before lunch, or should we rebook the customer?",
        f"Chasing delivery of {o} to {a}. Driver hasn't called.",
        f"Order status please: {o}. Also, was the {s} substituted for anything?",
        f"We got part of {o} but the remaining {n} tyres weren't on the pallet. When is the balance due?",
        f"Is {o} ready for collection or are you sending it out?",
        f"Could you resend the dispatch note for {o}? I can't see it in my inbox.",
        f"Our PO against {o} still shows as processing. Has it been picked?",
        f"Just checking {o} didn't get held at the gate — we're {a}.",
        f"When will {o} be delivered? The original date was today.",
        f"Any tracking for {o}? Need to plan workshop time around the {s} fitment.",
        f"Has anyone looked at {o}? I rang earlier and was told someone would call back.",
        f"Please update us on {o}. If it's delayed we'll need to source elsewhere this once.",
        f"Was order {o} cancelled or is it still coming? Nothing on the portal.",
        f"Can you confirm the delivery address on {o} is still {a}?",
        f"We're still waiting on {o} from last week. This is the second chase.",
        f"Did {o} go on the Belfast run or the Dublin run today?",
        f"Order {o} shows packed but not shipped. Is there a problem?",
        f"Could you give me a time window for {o} tomorrow?",
        f"Has the missing line on {o} been added for the next delivery?",
        f"Quick status on {o} please — {n} {s} for {a}.",
    ]


def templates_complaint_return() -> list[str]:
    o, a, s, b, n = order_ref(), account(), size(), brand(), qty()
    return [
        f"The {s} tyres on {o} arrived damaged. Two have scuffed sidewalls. We need a collection and replacements.",
        f"I'm not happy with the last delivery. Wrong size sent — we ordered {s} and got {size()}. Please arrange a return.",
        f"We've had an early wear complaint on {b} {s} fitted six weeks ago. What's the warranty process?",
        f"Please raise a return for {n} unused {s} from {o}. Customer changed their mind, tyres still wrapped.",
        f"One tyre in {o} has a manufacturing bulge. Photos attached in a separate mail. We want a credit or replacement.",
        f"The load index on the {s} you sent doesn't match the quote. We can't fit them. Collection please.",
        f"Complaint: yesterday's delivery to {a} was four hours late and the driver wouldn't help unload.",
        f"These {b} {s} are noisier than expected and the customer wants them off. Can we return them under your policy?",
        f"We were short-delivered on {o} — invoice says {n} but only {max(2, n - 2)} arrived. Please sort this.",
        f"There's a puncture in a brand-new {s} taken off the pallet. Definitely not us. Need a decision today.",
        f"Requesting a warranty inspection for a {b} {s}, DOT on the sidewall, fitted in March. Premature wear on the inner shoulder.",
        f"Please collect the incorrect {s} tyres and credit {o}. We've already bought the right size elsewhere to keep the job moving.",
        f"The rims you supplied with {o} had paint chips out of the box. Not acceptable — returning the lot.",
        f"I want to formally complain about being sold a part-worn as new. The {s} had old labels under the new sticker.",
        f"Can we return {n} {s} {season()} tyres? They're the wrong season for what the customer booked.",
        f"Fitment issue: {s} {b} is rubbing on the customer's arch. You recommended this size. What's the remedy?",
        f"Delivery to {a} was left in the rain with no wrapping. Casings are wet and dirty. We expect replacements.",
        f"Still waiting on the credit for the damaged {s} we reported last week against {o}. This is a reminder.",
        f"The customer is furious — second {b} {s} failure in a month. We need a technical view, not just a thanks-for-your-email.",
        f"Please log a complaint: order {o} came with the wrong brand and nobody told us about a substitution.",
        f"Returning {n} tubes that don't fit the {s} tyres they were sold with. Please advise collection.",
        f"We've been charged for disposal on tyres we never received. Also the {s} that did arrive have flat spots.",
        f"This is the third time a {s} order has arrived incomplete. We need a better process or we'll move suppliers.",
        f"Can someone call me about a sidewall crack on a new {b} {s}? I don't want to keep emailing.",
        f"Request to return surplus stock: {n} x {s} unopened, purchased on {o}. Within your 14-day window.",
        f"The balancing on the {s} set is off and we're getting vibration complaints. What will you do?",
        f"I asked for OEM-quality {s} and received a budget casing. That's not what we agreed. Collect and replace.",
        f"Damaged pallet on {o}: at least one {s} has a crushed bead. Photos ready.",
        f"Please escalate — we still have no resolution on the faulty {b} tyres raised a fortnight ago.",
        f"Not happy: invoice paid, goods not as described, {s} instead of {size()}. Arrange a return today.",
    ]


def templates_account_billing() -> list[str]:
    inv, acc, a, o = invoice_ref(), account_ref(), account(), order_ref()
    return [
        f"Can you resend {inv}? We didn't get the PDF and our accounts team is looking for it.",
        f"There's a VAT query on {inv}. It looks like it's been charged at 23% on a line that should be zero-rated — can you check?",
        f"Please update the billing address on {acc} to our new accounts office. I'll confirm the Eircode on the call.",
        f"We think we've been double charged for {o}. {inv} and another invoice both seem to cover the same tyres.",
        f"Can you extend the credit terms on {acc}? We're a long-standing account ({a}) and cash flow is tight this month.",
        f"The payment for {inv} went out on Friday. Can you confirm it's been allocated and lift the stop on the account?",
        f"Our statement doesn't match. We're missing a credit note that was promised against {inv}.",
        f"Please put {inv} on hold — the PO number is wrong. It should reference {o}.",
        f"Who do I send a remittance advice to for {acc}? Paying {inv} today.",
        f"Can you break out the delivery charge on {inv}? Our bookkeeper needs it as a separate line.",
        f"We've been put on stop according to the sales desk. Account {acc}, {a}. There's a query on {inv} we raised last week.",
        f"Please issue a credit note for the returned tyres on {o} and show it against {inv}.",
        f"Is {inv} payable yet? The date looks earlier than our agreed 30 days.",
        f"Can someone in accounts call me about {acc}? There's a balance we don't recognise.",
        f"We need a copy of all invoices for October on {acc} for our audit file.",
        f"The IBAN on your statement is different to the one we have on file. Please confirm bank details before we pay {inv}.",
        f"Please add our new PO prefix to {acc} or invoices will be rejected by our system.",
        f"Invoice {inv} is missing our VAT number. Can you reissue it?",
        f"We're querying labour/fitting on {inv} — we agreed supply only.",
        f"Can you apply the early-settlement discount to {inv} if we pay this week?",
        f"Account {acc} is still showing an old trading name. We now trade as {a}. Please update invoices.",
        f"A direct debit failed for {inv}. What's the outstanding total so I can pay by transfer?",
        f"Please put a copy of {inv} and the delivery docket in one email for our records.",
        f"There's a price discrepancy on {inv} versus the quote we accepted for {o}.",
        f"Can you raise a credit for the overcharge on {inv} and rebill the correct amount?",
        f"We're waiting on a year-to-date statement for {acc} before we close our books.",
        f"Who is the credit controller for {a}? I need to discuss a payment plan for {inv}.",
        f"Please confirm whether {inv} includes the promotional rebate we were offered.",
        f"Our finance team rejected {inv} because the account number {acc} is wrong by one digit.",
        f"Can you email a copy of the original credit application for {acc}? We're updating authorised signatories.",
    ]


def templates_other() -> list[str]:
    a = account()
    return [
        "What are your opening hours on Saturday?",
        "Are you open over the bank holiday weekend, or should we order on Tuesday?",
        "Can you send a current trade catalogue and price list to this email?",
        "Do you have a public price list or is everything on application?",
        "We're looking for a new tyre supplier for our workshop. Can someone call to talk through account setup?",
        f"Is there a minimum order value for delivery to {a}?",
        "Can I book a sales rep visit for next week? Prefer Tuesday afternoon.",
        "Do you offer a mobile fitting service or is it supply only from the depot?",
        "Where is the trade counter, and is there parking for a rigid?",
        "Could you add me to the weekly stock-clearance mailing list?",
        "What documents do you need to open a trade account?",
        "Do you have an apprenticeship or counter staff vacancy at the moment? I was told to email this address.",
        "Happy Christmas — could you confirm your shutdown dates between Christmas and New Year?",
        "Can we arrange a site visit? We'd like to see the warehouse before we move our account.",
        "Are you certified to dispose of waste tyres, and can we drop casings to you?",
        "Do you run any training sessions on tyre labelling or load-index regs?",
        "Is your online portal down? I can't reset my password.",
        "Can you tell me the Eircode for the depot? The sat nav is sending our driver to the wrong estate.",
        "We're a local GAA club looking for a sponsorship for junior tyres on the minibus — is that something you do?",
        "Would you be interested in advertising in our garage association newsletter?",
        "Do you supply agricultural tyres at all, or only passenger and commercial?",
        "Can I get a copy of your terms and conditions and credit policy?",
        "What's the best phone number for the trade counter? The one on the website rings out.",
        "Are dogs allowed in the trade counter? I'll be calling in with the van.",
        f"Please note we'll be closed for two weeks from the 1st — no change to the {a} account, just a heads-up.",
        "Do you have a recommended tyre pressure chart you share with trade accounts?",
        "Can you confirm whether you can deliver to a site with no forklift?",
        "I'd like to complain about the website search — this isn't an order issue, I just can't find part numbers.",
        "Is there a loyalty or volume rebate scheme if we consolidate our spend with you?",
        "We're organising a charity run and wondered if you'd donate a raffle prize.",
        "What time does the last collection leave the depot on a Friday?",
        "Do you have an environmental policy I can pass to our procurement team?",
        "Can someone from marketing send high-res logos for a workshop display?",
        "Are you the same company as the retail bay on the main road, or a different group?",
        "Just saying thanks for the help last week — no action needed.",
        "Hi, I think I was put through to the wrong department. I was looking for jobs in accounts, not an order.",
        "Could you point me to your data protection / privacy notice? Our DPO asked for it.",
        "Do you accept cash at the trade counter or is it account and card only?",
        "Is there EV-specific tyre advice you offer to independent garages?",
        "Can I bring a work experience student for a morning at the counter? School request.",
    ]


TEMPLATE_FNS = {
    "quote_request": templates_quote_request,
    "stock_availability": templates_stock_availability,
    "order_status": templates_order_status,
    "complaint_return": templates_complaint_return,
    "account_billing": templates_account_billing,
    "other": templates_other,
}


def generate_messages(target_counts: dict[str, int], rng_seed: int = RANDOM_SEED) -> list[dict[str, str]]:
    """Build a labelled list of unique synthetic messages."""
    random.seed(rng_seed)
    rows: list[dict[str, str]] = []
    seen: set[str] = set()

    for label, target in target_counts.items():
        produced = 0
        # Safety cap so we cannot loop forever if uniqueness gets hard.
        attempts = 0
        max_attempts = target * 40
        while produced < target and attempts < max_attempts:
            attempts += 1
            # Fresh random slots on every call of the template bank.
            candidate = wrap(random.choice(TEMPLATE_FNS[label]()))
            key = candidate.strip().lower()
            if not candidate.strip() or key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "message_id": f"SYN-{len(rows) + 1:04d}",
                    "text": candidate,
                    "label": label,
                    "data_source": "SYNTHETIC_EXAMPLE_DATA",
                }
            )
            produced += 1
        if produced < target:
            raise RuntimeError(
                f"Could only generate {produced}/{target} unique messages for {label}."
            )

    random.shuffle(rows)
    # Re-number after shuffle so IDs follow file order.
    for i, row in enumerate(rows, start=1):
        row["message_id"] = f"SYN-{i:04d}"
    return rows


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    out_path = data_dir / "SYNTHETIC_tyre_desk_enquiries.csv"

    rows = generate_messages(TARGET_COUNTS)
    fieldnames = ["message_id", "text", "label", "data_source"]
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Small sidecar so anyone opening the folder sees the warning first.
    readme = data_dir / "README.md"
    readme.write_text(
        """# Data folder — SYNTHETIC example data only

The file `SYNTHETIC_tyre_desk_enquiries.csv` is **entirely invented**.

- It is not extracted from any real inbox, CRM, phone log, or customer file.
- Names, account numbers, order numbers and invoices are fictional.
- The `data_source` column is set to `SYNTHETIC_EXAMPLE_DATA` on every row.
- Do not mix this file with live customer data.

It exists so the enquiry-classifier project can be trained, evaluated and
demonstrated without processing personal data.
""",
        encoding="utf-8",
    )

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["label"]] = counts.get(row["label"], 0) + 1
    print(f"Wrote {len(rows)} SYNTHETIC messages to {out_path}")
    for label in CLASSES:
        print(f"  {label:22s} {counts.get(label, 0)}")


if __name__ == "__main__":
    main()
