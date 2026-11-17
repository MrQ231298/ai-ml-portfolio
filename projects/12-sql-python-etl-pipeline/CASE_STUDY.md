# Case study: monthly reporting you can trust, without the spreadsheet fixes

**Audience:** a finance, operations or e-commerce manager — not a data
engineer. Method and full results are in [README.md](README.md).

**Status:** portfolio demonstration on **synthetic** exports for a
fictional Irish online retailer (1,200 customers, 7,000 orders,
January–October 2026). No real customer data was used.

---

## The business problem

Your shop, CRM and finance systems all export CSV files. Every month
someone combines them in Excel and fixes the same problems by hand:
duplicate rows, dates in different formats, prices typed as text,
orders from customers "not in the system yet", sterling mixed with
euro. It takes days, and two people never get the same total.

## What this demo shows

- An automated pipeline that takes the raw exports and produces clean
  tables and ready-made reports: **monthly revenue by channel, a
  customer summary and a product league table**.
- **It matched the correct revenue to the cent** (€858,984.52) after
  dealing with hundreds of deliberately planted problems.
- **Nothing disappears silently.** Every row it could not use is kept
  in a "quarantine" list with the reason — for example 3 products with
  no price, which were holding back 385 sales lines. The pipeline
  raised a warning about exactly that.
- **Late information sorts itself out.** 809 orders arrived before their
  customer record; they were held and loaded automatically when the
  customer appeared in the next export.
- **Safe to run again.** Running it twice, or re-sending the same files,
  changes nothing.
- It **stops and says why** when something critical is missing, such as
  an exchange rate.

## What this could be worth

Days of month-end spreadsheet work become a command that runs in
seconds, and the numbers in your dashboards, board pack and accounts
come from one source.

## What we would do next with you

1. Collect a few months of your real exports.
2. Agree the business rules (what counts as revenue, how to treat
   returns, cancellations and currency).
3. Run the pipeline on history and reconcile with your accounts.
4. Schedule it and connect your BI tool to the reporting tables.

## Limits to keep in mind

The problems in this demo were planted, so catching all of them is
expected; real data always has surprises. Files must be loaded in date
order (or the pipeline extended to check record dates). Near-duplicate
customers need a matching step not included here.
