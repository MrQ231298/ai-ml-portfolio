# Data

**REAL public data.** UCI *Online Retail* (Chen, D., 2012), UCI Machine
Learning Repository, [doi:10.24432/C5BW33](https://doi.org/10.24432/C5BW33),
licensed **CC BY 4.0**. It holds every transaction of a UK-based online
gift retailer (many customers are wholesalers) from 1 December 2010 to
9 December 2011.

- `src/fetch_and_features.py` downloads the zip into `downloads/`
  (git-ignored) and checks its SHA-256 (`f5385cbb…`) before use. The raw
  file is **not committed**.
- `customer_features_calibration.csv` is **derived** by this project: one
  row per customer, with features from the calibration window
  (2010-12-01 to 2011-09-09) and two hold-out outcome columns
  (`future_90d_purchased`, `future_90d_revenue`, 2011-09-10 to
  2011-12-09). Committed so the app and tests run without the download.
- `source_metadata.json` gives the source URL, checksum, licence,
  citation, cleaning row counts and build time.

**Personal data:** `customer_id` is the dataset's own pseudonymous
5-digit number. There are no names, emails or addresses. The country is
the only location field.

Cleaning (from `source_metadata.json`): 541,909 raw rows → 5,268 exact
duplicates removed → 401,604 rows with a customer ID → 399,656 rows after
dropping non-product codes (postage, fees, manual adjustments, bad debt)
and zero or negative prices. Customers whose net calibration spend was not
positive are excluded, which leaves 3,360 customers.
