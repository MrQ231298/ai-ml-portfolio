# AI / ML portfolio — Owen Quigley

Beginner-friendly machine learning work that is meant to do two jobs at once:

1. **Client-ready demos and case studies** — small, working prototypes a
   non-specialist can try, with a plain-English write-up of the business
   problem and the limits of the result.
2. **Documented evidence for a Recognition of Prior Learning (RPL)
   application** toward a **QQI Level 9 (Master's)** programme in the
   Irish NFQ, most likely in computing, data analytics, or a related
   field. The exact course is not yet confirmed, so the learning-outcome
   map uses the generic Level 9 NFQ descriptors and can be relabelled
   against a specific module list later.

Background: sales administrator at a tyre distributor
until November 2026, with further interests in Irish landlord / rental
work and commercial insurance lead generation. The first project draws
on that sales-desk context. It does **not** use real customer or company
mail; all training text is synthetic.

This repository is a starting point, not a claim that a single baseline
model already satisfies a Master's award. Later projects should deepen
the research, evaluation and professional practice that Level 9 asks
for. The [critical evaluation](projects/01-enquiry-classifier/README.md#critical-evaluation)
in project 1 is there to show that distinction clearly.

---

## Project index

| # | Project | Domain | What you can try | Status |
| --- | --- | --- | --- | --- |
| 01 | [Enquiry classifier](projects/01-enquiry-classifier/) | B2B tyre / automotive-parts sales desk | [Streamlit app](projects/01-enquiry-classifier/app.py); [code walkthrough](projects/01-enquiry-classifier/CODE_WALKTHROUGH.md) | Complete baseline (trained 14 Nov 2026). SYNTHETIC data. |
| — | *Planned: landlord / rental or commercial insurance lead-gen case study* | Irish rental / insurance | — | Not started. Reserved so the portfolio can show a second domain. |

Each finished project should contain:

- a beginner-friendly `README.md` (problem, data, method, **actual**
  metrics, limitations)
- a `CODE_WALKTHROUGH.md` that explains the files and key code in plain English
- a short `CASE_STUDY.md` written for a potential client
- runnable code, a requirements file, and saved artefacts from a real run

---

## How to run project 1

You need Python 3.11 or newer (the recorded run used 3.13.5).

A step-by-step tour of the files and the important lines of code is in
[`projects/01-enquiry-classifier/CODE_WALKTHROUGH.md`](projects/01-enquiry-classifier/CODE_WALKTHROUGH.md).

```bash
cd projects/01-enquiry-classifier
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python src/generate_synthetic_data.py
python src/train.py
streamlit run app.py
```

On the recorded run (14 November 2026, Europe/Dublin):

| Model | Accuracy | Macro-F1 | Weighted-F1 |
| --- | ---: | ---: | ---: |
| Dummy (most frequent class) | 0.2222 | 0.0606 | 0.0808 |
| TF-IDF + Logistic Regression (used in the app) | 0.9630 | 0.9607 | 0.9626 |
| TF-IDF + LinearSVC | 0.9815 | 0.9798 | 0.9814 |

These scores were produced by the training script on **synthetic**
messages. They are intentionally discussed as optimistic in the project
README.

The Streamlit demo after predicting a synthetic quote-request message
(captured 14 November 2026, Europe/Dublin):

![Streamlit demo predicting Quote request for a synthetic tyre-desk enquiry](projects/01-enquiry-classifier/reports/app_screenshot.png)

Logistic Regression confusion matrix on the held-out synthetic test set:

![Confusion matrix for TF-IDF + Logistic Regression on the held-out synthetic test set](projects/01-enquiry-classifier/reports/confusion_matrix_logistic_regression.png)

---

## RPL evidence

| File | Purpose |
| --- | --- |
| [`rpl/LEARNING_OUTCOMES_MAP.md`](rpl/LEARNING_OUTCOMES_MAP.md) | Editable table: NFQ Level 9 descriptor → technical outcome → project → evidence file |
| [`rpl/RPL_EVIDENCE_LOG.md`](rpl/RPL_EVIDENCE_LOG.md) | Dated activity log, with first entries for 14 November 2026 |

If a chosen programme publishes its own module outcomes, copy a new
column into the map rather than rewriting the projects.

---

## Data protection

Do not put live customer, tenant or claimant data in this repository.
Project 1 uses a file explicitly named and tagged as
`SYNTHETIC` example data. That is a GDPR-aware design choice: a
portfolio can demonstrate a full machine-learning workflow without
processing personal data. A real deployment would need a lawful basis,
minimisation, and usually a DPIA — see the project critical evaluation.

---

## Repository layout

```
ai-ml-portfolio/
  README.md                          this file
  LICENSE                            MIT, Copyright (c) 2026 Owen Quigley
  rpl/LEARNING_OUTCOMES_MAP.md
  rpl/RPL_EVIDENCE_LOG.md
  projects/01-enquiry-classifier/
```

This repository is released under the [MIT License](LICENSE).
Copyright (c) 2026 Owen Quigley.
