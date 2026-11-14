# Data folder — SYNTHETIC example data only

`SYNTHETIC_weekly_tyre_sales.csv` is **entirely invented**.

- It is not extracted from any tyre distributor's sales system.
- It is not official NCT, CSO or supplier data.
- The `data_source` column is `SYNTHETIC_EXAMPLE_DATA` on every row.

365 weekly observations, 2019-01-07 to
2025-12-29 (weeks starting Monday). The series is
built from a trend, a winter-tyre bump, a spring NCT-inspired bump,
a summer lull, a short 2020 dip, and noise — see
`src/generate_synthetic_series.py`.
