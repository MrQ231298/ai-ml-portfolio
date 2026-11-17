# Case study: a monthly data pipeline that checks itself

**Audience:** an operations, finance or BI manager, not an engineer.
Method and full results are in [README.md](README.md).

**Status:** portfolio demonstration on **real, public** data: 6 million
New York yellow-taxi trips from January and February 2024 (NYC TLC open
data).

---

## The business problem

Many companies build their monthly reports from files that arrive from a
system, a supplier or a regulator. The usual pain points:

- Someone loads a file twice and the totals double.
- A bad record breaks a dashboard, or quietly skews it.
- Nobody can say why the report differs from the source system.

## What this demo shows

- **An automated monthly run**. Each month the pipeline fetches the file,
  loads it, cleans it, builds the reporting tables, and runs 23 automatic
  checks before anything is published.
- **Safe to rerun.** Loading January a second time left every reporting
  table exactly the same, and rebuilding everything from scratch did too.
- **It stops on bad data.** A record with an unknown location was planted
  in a test copy. The checks caught it and the run stopped before any
  report was refreshed.
- **Every rejected row is explained.** Between 4% and 5% of records were removed
  each month, mostly trips with impossible distances, negative fares or
  zero passengers. The count for each reason is a report of its own.
- **Useful numbers come out.** Manhattan has 90% of trips. Trips that start in
  Queens, mostly at the two airports, are worth three times as much as
  Manhattan trips. Thursday at 6 pm is
  the busiest hour of the week.

## What it means in practice

The same set-up works for sales, stock, production or claims files. You
get reports that you can trust and rerun, and an audit trail for every
excluded record. It uses the tools data teams already hire for (dbt and
Airflow), so it can move to your cloud warehouse later.

## Limits

This demo ran on one machine, not on a cloud platform or a permanent
scheduler server. The cleaning rules are reasonable choices, not official
TLC rules. Only two months of data were processed.
