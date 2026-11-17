# Case study: a trustworthy data platform on Azure / Fabric, designed before you spend

**Audience:** an IT, finance or operations lead considering Microsoft
Fabric or Azure for reporting — not a data engineer. Method and full
results are in [README.md](README.md).

**Status:** portfolio demonstration on **real public data** (1.07 million
transactions of an online gift retailer, 2009–2011, CC BY 4.0), built
and tested **on a laptop**. Not deployed to Azure.

---

## The business problem

"Lakehouse" and "medallion architecture" are the standard advice for
Fabric and Azure. The risk is building three layers of folders without
the guarantees that make the numbers trustworthy: what happens when a
file is sent twice, or corrected, or arrives late?

## What this demo shows

- Raw files kept exactly as delivered (**bronze**), cleaned and
  de-duplicated records (**silver**), and report-ready tables (**gold**).
- **34,335 duplicate rows** (3.2%) found and removed in the real data
  before they could inflate sales.
- **Loading month by month gives exactly the same result as loading
  everything at once**, checked across every table.
- **Sending the same files again changes nothing**, and **a corrected
  month replaces only that month**, with the previous version still
  readable for audit.
- A complete run of a million rows takes about **5 seconds**.
- The design is mapped piece by piece to Azure Data Factory, ADLS and
  Fabric, with infrastructure code that compiles.

## What this could be worth

You can settle design questions and test your own data on a laptop
before paying for cloud capacity, and then move to Fabric with the same
tables and logic.

## What we would do next with you

1. Run this pattern on a sample of your exports.
2. Agree the rules for duplicates, corrections and late data.
3. Deploy to a Fabric trial capacity and re-run the same checks there.
4. Connect Power BI to the gold tables.

## Limits to keep in mind

Nothing was run in Azure or Fabric for this demo; cloud-specific
performance and security settings still have to be tested there. The
data is from 2009–2011 and describes one UK retailer.
