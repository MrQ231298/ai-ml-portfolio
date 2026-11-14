# Data folder — SYNTHETIC example data only

The file `SYNTHETIC_commercial_insurance_leads.csv` is **entirely invented**.

- It is not extracted from any broker, insurer, or business CRM.
- Names, lead IDs and outcomes are fictional.
- The `data_source` column is `SYNTHETIC_EXAMPLE_DATA` on every row.
- Do not mix this file with live prospect data.

3,200 rows. Conversion rate in this draw: 22.6%.
The labels come from a noisy process documented in
`src/generate_synthetic_leads.py` (source, response time, size, cover
type and a few interactions). Region was given only a small effect so
we can later ask whether a fitted model over-uses it.
