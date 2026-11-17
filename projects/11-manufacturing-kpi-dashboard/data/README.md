# Data folder — SYNTHETIC plant data (fictional)

**Everything here is invented.** It was produced by
[`../src/generate_data.py`](../src/generate_data.py) with seed
`20261117`. No real plant, employer or supplier data was used; the
product codes, lines and numbers are fictional.

| Item | Detail |
| --- | --- |
| Plant | Fictional automotive electronics plant: SMT-1, SMT-2 (parallel) → ASSY-1 → EOL-1 |
| Period | 1 July – 31 October 2026; Mon–Fri 3 × 8 h shifts, Sat early shift, Sun closed |
| Planned stops | 30 min per shift (breaks / handover), excluded from planned production time |
| Flag | Every fact row: `data_flag = SYNTHETIC_EXAMPLE_DATA` |
| Format | CSV star schema in `star_schema/`, ready for Power BI, DuckDB or pandas |

## Tables

| File | Grain / content |
| --- | --- |
| `fact_production.csv` | line × shift × day: planned, run and downtime minutes, ideal cycle time, total / first-pass good / good / scrap / rework counts |
| `fact_target.csv` | target units for the same grain (85% of theoretical capacity) |
| `fact_downtime.csv` | one row per unplanned stop: reason, duration (0.1 min) |
| `fact_defects.csv` | defects by type: count, reworked, scrapped |
| `dim_*.csv` | date, line, shift, product, downtime reason, defect lookups |
| `planted_issues.json` | the four problems deliberately added (ground truth for the alert tests) |

Counts obey `first_pass_good ≤ good ≤ total`, `total − good = scrap`,
and `run + unplanned downtime = planned production` (tested).
