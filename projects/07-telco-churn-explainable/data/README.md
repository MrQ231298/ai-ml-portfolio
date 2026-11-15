# Data folder — IBM Telco Customer Churn (public sample)

**Real-format public sample data, not a client extract.**

| Item | Detail |
| --- | --- |
| File | `Telco-Customer-Churn.csv` (7,043 rows, 21 columns) |
| Source | IBM, GitHub repository `IBM/telco-customer-churn-on-icp4d`, file `data/Telco-Customer-Churn.csv` |
| URL | https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv |
| Licence | Apache License 2.0 (the repository's licence) |
| Fetched | 15 November 2026, 16:19 (Europe/Dublin) by `src/fetch_data.py`; SHA-256 in `source_metadata.json` |

IBM publishes this as a *sample* dataset for analytics tutorials. It
describes a fictional telecom provider's customers, so it is realistic
in structure but should not be treated as a live operator's churn rate.
There are no names, phone numbers or addresses; `customerID` is an
opaque code and is never used as a model input.

## Known data-quality issue (handled in `src/churn_lib.py`)

`TotalCharges` is stored as text and is blank for 11 customers. All 11
have `tenure = 0` (signed up but not yet billed), so the blanks are set
to 0.0. The loader asserts this, so a different file with unexplained
blanks will fail loudly instead of being silently imputed.

## Columns used and not used

- **Model inputs (17):** contract, tenure, charges, payment method,
  paperless billing, the internet / phone services and add-ons, partner
  and dependants.
- **Excluded from the model, used only for the fairness audit:**
  `gender`, `SeniorCitizen`.
- **Target:** `Churn` (Yes = left within the last month).
