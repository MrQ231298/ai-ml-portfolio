# Case study: is a forecast better than "same month last year"?

**Audience:** a retail, wholesale or supply-chain planner — not a
data-science reviewer. Method and full results are in [README.md](README.md).

**Status:** portfolio demonstration on **real** Irish official
statistics: the CSO Retail Sales Index for 13 retail sectors,
January 2015 – September 2026.

---

## The business problem

Most planning starts from "what did we sell in the same month last
year?" It is simple and often surprisingly good. Before paying for a
forecasting system, a business should know: **does it beat last year,
by how much, and is that just luck?**

## What this demo shows

- A **dashboard** with each sector's history, a six-month forecast and
  a likely range.
- A fair test: **1,950 forecasts per method**, each made using only the
  data available at the time, over two years (March 2024 – March 2026).
- The best method (a blend of a machine-learning model and last year's
  value adjusted for growth) was about **5% more accurate** than "same
  month last year" overall, and clearly better for sectors with a
  steady trend such as pharmacy/cosmetics and supermarkets.
- We also report the uncomfortable parts: in 6 of 13 sectors last
  year's value was still best, the 5% advantage is **not yet
  statistically certain**, and the "80%" ranges were right about 77% of
  the time.

## What this could be worth

For a business with many products, branches or categories:

- A forecast you can trust *because* it has been tested against the
  simple rule your team already uses.
- Attention focused on the categories where a model really helps, and
  "last year" kept where it is just as good — cheaper to run and easier
  to explain.
- Ranges that support safety-stock and staffing decisions.

## What we would do next with a client

1. Load your weekly or monthly sales by product / branch.
2. Run the same back-test against your current method.
3. Add the drivers you know matter (promotions, holidays, weather,
   prices).
4. Hand over the dashboard and re-check accuracy every month.
