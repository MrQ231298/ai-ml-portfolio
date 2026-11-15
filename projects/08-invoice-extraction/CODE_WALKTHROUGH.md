# Code walkthrough — invoice extraction

## The pipeline at a glance

```
1. python src/generate_invoices.py
        ↓ data/invoices_pdf/*.pdf (240) + data/SYNTHETIC_invoice_ground_truth.csv

2. python src/evaluate.py [--re-ocr]
        ↓ digital text (pdftotext) for all; simulated scan + OCR for test invoices
        ↓ trains the ML ranker on layouts A–C (train split) → models/candidate_ranker.joblib
        ↓ scores rules / ML / hybrid on 4 test conditions + 5-shot adaptation
        ↓ reports/metrics.json, extracted_fields.csv, sample output, plots

3. pytest -q
4. streamlit run app.py
```

## What each file is for

| File | Purpose |
| --- | --- |
| [`src/generate_invoices.py`](src/generate_invoices.py) | Invents invoices and draws them in four layouts with reportlab. |
| [`src/extract_lib.py`](src/extract_lib.py) | Text extraction (pdftotext / RapidOCR), parsing helpers, `RuleExtractor`, candidate generation and features, `CandidateModel`, `arithmetic_ok`. |
| [`src/evaluate.py`](src/evaluate.py) | Splits, OCR cache, training, scoring, adaptation experiment, plots. |
| [`app.py`](app.py) | Upload or pick a PDF, optional simulated scan, see fields + check + CSV. |
| [`tests/test_smoke.py`](tests/test_smoke.py) | Parsers, a rule extraction, model loading, recorded metrics. |

## Key ideas in the code

**Candidates, not templates.** `candidates(text)` returns every money
amount, date, ID-like token and top-of-page segment, each with features.
The model never learns "the total is at x = 400"; it learns things like
"the total is usually the largest amount and equals two other amounts
added together".

**Arithmetic features.** For each amount, `is_sum` (equals the sum of
two others), `is_part` (is one of two amounts that add to a third),
`vat_like` (≈ 23% / 13.5% / 9% of another amount) and `has_vat_of_it`.
These survived the change to an unseen layout; label words did not.

**One logistic regression per field.** `DictVectorizer` turns the
feature dictionaries (including `ctx=word` and `last=word` for the words
to the left) into a sparse matrix. At prediction time the highest
probability candidate is chosen.

**OCR line rebuilding.** RapidOCR returns boxes. `boxes_to_layout_text`
groups boxes whose vertical centres are close into one line and joins
them with three spaces, imitating `pdftotext -layout` so both
extractors see similar text.

**Normalised comparison.** `norm_value` makes scoring fair: amounts to
two decimals, ISO dates, vendor names without spaces or case (OCR
drops spaces).
