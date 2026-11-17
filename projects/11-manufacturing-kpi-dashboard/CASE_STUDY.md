# Case study: one trusted OEE number and early warnings for a plant

**Audience:** a plant, operations or quality manager — not a data
reviewer. Method and full results are in [README.md](README.md).

**Status:** portfolio demonstration on **synthetic** data for a
fictional automotive electronics plant (four lines, three shifts,
July–October 2026). No real plant data was used.

---

## The business problem

Every plant tracks output, stops and scrap, but often in separate
spreadsheets that disagree. OEE gets averaged the wrong way, the top
downtime reasons are argued about rather than measured, and quality
problems are spotted after weeks of rework.

## What this demo shows

- A dashboard with **OEE, availability, performance, quality, first-pass
  yield and output vs target**, filterable by date, line, shift and
  product.
- **Where the time went:** of every 100 planned minutes, about 79 made
  good product first time, 11 were lost to stops, 9 to running slow and
  1 to defects.
- A **downtime Pareto**: changeover overruns alone were 31% of
  unplanned downtime, concentrated on one line (SMT-2), which ran at
  74% OEE against 79–83% for the others.
- **Early warnings that were tested.** Four problems were hidden in the
  data on purpose. Simple control-chart rules found the September
  solder-defect spike in both affected weeks, the changeover problem,
  the slower night shift and 3 of 5 weeks of a failing test fixture —
  and raised 3 false alarms in 88 checks.
- The same measures delivered as a **Power BI** build kit, so your team
  can own the report.

## What this could be worth

On this example, SMT-2's availability gap to the other SMT line is
about 5.6 percentage points of planned time, mostly changeovers.
Closing half of it is about 3,600 more running minutes, or roughly
9,900 extra good units over the four months at that line's own output
rate — before touching speed losses. (Computed from the synthetic
data; a real figure depends on your mix and whether downstream lines
can absorb it.)

## What we would do next with you

1. Map your shift report, MES stop codes and test data to this model.
2. Agree the OEE definitions in writing (planned stops, ideal rates).
3. Build the Power BI report on your data and check it against SQL.
4. Tune the alerts on a few months of history before switching them on.

## Limits to keep in mind

Synthetic data and planted problems make the alerts look better than
they will on day one. Targets here are a fixed share of capacity. The
Power BI file still has to be built on Windows from the guide.
