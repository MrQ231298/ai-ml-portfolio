# Data folder — SYNTHETIC retail extracts (fictional)

**Everything here is invented** by [`../src/generate_sources.py`](../src/generate_sources.py)
(seed `1211`). The retailer, customers (names from a list of common
Irish first and last names, emails at `example.ie`), products and orders
are fictional. Every row carries `data_flag = SYNTHETIC_EXAMPLE_DATA`
and the pipeline refuses a batch whose manifest is not tagged that way.

| Folder | Content |
| --- | --- |
| `raw/batch_2026_06/` | Extract taken 1 July 2026: orders Jan–Jun, 80% of customers |
| `raw/batch_2026_10/` | Extract taken 1 November 2026: orders Jul–Oct plus 150 re-sent June orders, all customers (40 with a new county), future-dated rows |
| `truth/injected_defects.csv` | Every injected defect: table, record key, defect type |
| `truth/expected_totals.json` | Revenue, lines and orders a correct pipeline should report, computed from the clean truth with exact decimal arithmetic |

Files per batch: `manifest.json` (batch id, extract date, file list),
`customers.csv`, `products.json`, `fx_rates.csv` (EUR per GBP by month,
invented), `orders.csv`, `order_lines.csv`.

`truth/` is used only by `src/evaluate.py` and the tests.
