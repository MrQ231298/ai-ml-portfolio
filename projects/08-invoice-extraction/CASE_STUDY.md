# Case study: turning supplier invoices into a spreadsheet automatically

**Audience:** a finance manager or bookkeeper — not a data-science
reviewer. Method and full results are in [README.md](README.md).

**Status:** portfolio demonstration on 240 *invented* invoices in four
different supplier layouts, as clean PDFs and as simulated scans.

---

## The business problem

Someone in accounts types the same eight things off every invoice:
supplier, VAT number, invoice number, issue date, due date, net, VAT
and total. It is slow, it is error-prone, and off-the-shelf templates
break every time a new supplier sends a different layout.

## What this demo shows

- Drop in a PDF (or a scanned copy) and get the eight fields back,
  checked for **net + VAT = total**, and a CSV to download.
- On the three supplier layouts it was built on, it read **99.3% of
  fields correctly even from scanned copies**.
- On a **fourth supplier it had never seen**, it still got most fields
  right (dates and amounts), but it **mixed up the invoice number with
  the order reference every time**. We show that openly, because that is
  what happens in real life when a new supplier appears.
- After a person corrected just **five** invoices from the new supplier,
  the system got **every field right** on that supplier's other 55
  digital invoices.

## What this could be worth

- Hours a week back for the accounts team.
- Fewer keying errors and late-payment surprises (due dates captured
  automatically).
- A simple rule: anything that fails the arithmetic check, or comes from
  a new supplier, goes to a person first.

## What we would do next with a client

1. Take a sample of 100–200 of your real invoices (kept on your systems).
2. Measure accuracy per field and per supplier before automating anything.
3. Add line-item extraction if you need it, and a check against your
   supplier list.
4. Export to your accounting package's import format.
