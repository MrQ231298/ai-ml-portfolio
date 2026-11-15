# Data folder — Women's E-Commerce Clothing Reviews (real, public)

**Real customer reviews, anonymised by the publisher.**

| Item | Detail |
| --- | --- |
| Dataset | Women's E-Commerce Clothing Reviews (Nick Brooks) |
| Original | https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews |
| Licence | **CC0: Public Domain** (as stated on the Kaggle page) |
| File used | `womens_clothing_reviews.csv` — 23,486 rows, 8.5 MB |
| Mirror | Kaggle needs a login, so `src/fetch_data.py` downloads the identical CSV from a public GitHub mirror (URL and SHA-256 in `source_metadata.json`) |
| Fetched | 15 November 2026, 16:40 (Europe/Dublin) |

Columns: clothing ID, reviewer age, title, review text, rating (1–5),
recommended (0/1), positive-feedback count, division, department, class.

## Privacy

The publisher replaced the retailer's name with "retailer" and the file
contains no reviewer names or contact details. Reviewer **age** is
present; this project does not use it as a model input or report
results by age. Short verbatim quotes appear in `reports/` for error
analysis and insight illustration, which CC0 permits.

## Preparation (in `src/review_lib.py`)

- 845 rows with no review text are dropped → **22,641 reviews**.
- Model text = title + ". " + review body.
- Labels from stars: 1–2 negative (10.5%), 3 neutral (12.5%), 4–5
  positive (77.1%). Stars are the customer's own rating, so labels are
  not hand-annotated sentiment; see the README critical evaluation.
