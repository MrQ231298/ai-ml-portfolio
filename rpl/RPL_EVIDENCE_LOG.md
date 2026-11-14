# RPL evidence log

Dated record of work that can be cited in a Recognition of Prior Learning
application, aimed at a **QQI Level 9 (Master's)** programme on the Irish
NFQ. The exact award is not yet confirmed; outcome codes refer to
[`LEARNING_OUTCOMES_MAP.md`](LEARNING_OUTCOMES_MAP.md) and can be
relabelled later.

Times below are in **Europe/Dublin**. Hours are recorded only when they
were actually kept; otherwise they are left as "not recorded" rather than
guessed.

---

## How to add an entry (template)

Copy the block and fill it in on the day the work is done. Point at
files that exist. Do not back-fill metrics from memory.

```
### YYYY-MM-DD — short title

- **Time / duration:** e.g. 19:30–21:00 (Europe/Dublin), or "not recorded"
- **Activity:** what was actually done
- **Artefacts:** paths in this repository (and any external URL)
- **Level 9 strands:** e.g. L9-CE, L9-ET (see the outcomes map)
- **Technical outcomes:** e.g. T-DP, T-ML, T-EV
- **Reflection:** what was learned, what did not work, what is still weak
- **Assessor notes:** *(leave blank)*
```

---

## Log

### 2026-11-14 — Portfolio scaffold and Level 9 RPL mapping

- **Time / duration:** evening session, 14 November 2026 (Europe/Dublin); hours not recorded
- **Activity:** Created the `ai-ml-portfolio` repository structure (top-level README, `rpl/` and `projects/`). Drafted a generic NFQ Level 9 learning-outcomes map (advanced knowledge, critical evaluation, research-informed practice, autonomy, ethics/GDPR, communication to specialist and non-specialist audiences) plus a technical-outcome table that can be retargeted to a specific programme. Set the tone of the file as *emerging* evidence toward Level 9, not a claim that the first baseline already meets a Master's award.
- **Artefacts:**
  - `README.md`
  - `rpl/LEARNING_OUTCOMES_MAP.md`
  - `rpl/RPL_EVIDENCE_LOG.md` (this file)
  - `.gitignore`
- **Level 9 strands:** L9-AU, L9-CO
- **Technical outcomes:** T-DC
- **Reflection:** Writing the map first, before dressing up a project as "Level 9", forced a useful distinction between a complete beginner baseline and the critical / ethical / communication work that starts to support RPL at Master's level. The map will need a programme-specific column once a course is chosen.
- **Assessor notes:**

### 2026-11-14 — Synthetic enquiry dataset (no live personal data)

- **Time / duration:** evening session, 14 November 2026 (Europe/Dublin); hours not recorded
- **Activity:** Designed a six-class scheme for a B2B tyre / automotive-parts sales desk (`quote_request`, `stock_availability`, `order_status`, `complaint_return`, `account_billing`, `other`). Wrote a seeded generator that produced **540** unique invented messages with a mild, realistic class imbalance. Every row is tagged `SYNTHETIC_EXAMPLE_DATA`; the filename and data README state the same. Fictional account names and document references only — no real customer or company correspondence was used.
- **Artefacts:**
  - `projects/01-enquiry-classifier/src/generate_synthetic_data.py`
  - `projects/01-enquiry-classifier/data/SYNTHETIC_tyre_desk_enquiries.csv`
  - `projects/01-enquiry-classifier/data/README.md`
- **Level 9 strands:** L9-RI, L9-ET, L9-AU
- **Technical outcomes:** T-DP, T-PY, T-ET
- **Reflection:** Using synthetic data is the right GDPR choice for a public portfolio, but it creates a serious validity problem: templates are tidier than a real inbox. That trade-off is now an explicit part of the project evaluation rather than a footnote. A future lawful pilot on live mail would need a DPIA and must not be committed to this repo.
- **Assessor notes:**

### 2026-11-14 — Train, evaluate and persist baseline models

- **Time / duration:** evening session, 14 November 2026 (Europe/Dublin); hours not recorded
- **Activity:** Built a leak-free pipeline: stratified 80/20 split (432 train / 108 test, `random_state=42`), TF-IDF unigrams+bigrams (vocabulary size **1,439**), and three models on the same features — DummyClassifier (most frequent), Logistic Regression, and LinearSVC. Trained in a project virtualenv on Python 3.13.5. Saved classification reports, `metrics.json`, two confusion-matrix PNGs, and a joblib model bundle for the demo app. Metrics below are copied from that run, not estimated.
- **Artefacts:**
  - `projects/01-enquiry-classifier/src/train.py`
  - `projects/01-enquiry-classifier/requirements.txt`
  - `projects/01-enquiry-classifier/reports/metrics.json`
  - `projects/01-enquiry-classifier/reports/classification_reports.txt`
  - `projects/01-enquiry-classifier/reports/confusion_matrix_logistic_regression.png`
  - `projects/01-enquiry-classifier/reports/confusion_matrix_linear_svc.png`
  - `projects/01-enquiry-classifier/models/tfidf_logistic_regression.joblib`
  - `projects/01-enquiry-classifier/models/tfidf_logistic_regression_pipeline.joblib`
- **Level 9 strands:** L9-K, L9-CE, L9-RI
- **Technical outcomes:** T-ML, T-EV, T-PY
- **Reflection:** Recorded test results:

  | Model | Accuracy | Macro-F1 | Weighted-F1 |
  | --- | ---: | ---: | ---: |
  | Dummy (most frequent) | 0.2222 | 0.0606 | 0.0808 |
  | Logistic Regression | 0.9630 | 0.9607 | 0.9626 |
  | LinearSVC | 0.9815 | 0.9798 | 0.9814 |

  Logistic Regression misclassified 4/108 messages, all at low confidence (approximately 0.21–0.29); LinearSVC misclassified 2/108. The high headline scores were expected once the errors were inspected: the synthetic classes have fairly distinct vocabulary. Serving Logistic Regression rather than the slightly stronger LinearSVC is a deliberate trade-off so the app can show a probability / confidence score. A single split of 108 test rows is not a strong enough validation for a Level 9 claim — that is recorded as a gap, not hidden.
- **Assessor notes:**

### 2026-11-14 — Streamlit demo, two-audience documentation, critical evaluation

- **Time / duration:** evening session, 14 November 2026 (Europe/Dublin); hours not recorded
- **Activity:** Wrote a Streamlit app so a non-specialist can paste a message and see the predicted class, confidence, and the full probability breakdown, with a warning that training data is synthetic and that personal data should not be pasted. Wrote a beginner-friendly technical README that includes a Level 9-oriented **critical evaluation** (method limits, bias/optimism in synthetic data, and what a real GDPR-aware deployment would need). Wrote a short client-facing `CASE_STUDY.md` in plain English. Smoke-tested that the Streamlit app starts, then stopped it.
- **Artefacts:**
  - `projects/01-enquiry-classifier/app.py`
  - `projects/01-enquiry-classifier/README.md`
  - `projects/01-enquiry-classifier/CASE_STUDY.md`
- **Level 9 strands:** L9-CE, L9-ET, L9-CO, L9-AU
- **Technical outcomes:** T-DC, T-ET, T-PY
- **Reflection:** The two-audience split (technical README + client case study + clickable demo) is the communication evidence; the critical evaluation is the part that stops the high accuracy number from being misleading. Next improvements that would actually move the RPL file toward Level 9 are methodological (cross-validation, a non-template test set, a rules baseline), not a more glamorous model.
- **Assessor notes:**

### 2026-11-14 — Code walkthrough, demo screenshot, MIT licence

- **Time / duration:** evening session, 14 November 2026 (Europe/Dublin); hours not recorded
- **Activity:** Added a beginner-friendly `CODE_WALKTHROUGH.md` for project 1 (each file's purpose, the generate → train → app run order, annotated snippets, and local run instructions) and linked it from the project README and the top-level README. Captured a real screenshot of the running Streamlit app showing a **Quote request** prediction, saved at `projects/01-enquiry-classifier/reports/app_screenshot.png`, and embedded it with *relative* image paths so it renders on GitHub. Converted the confusion-matrix links in both READMEs from file links to relative Markdown images for the same reason. Added an MIT `LICENSE` (Copyright (c) 2026 Owen Quigley).
- **Artefacts:**
  - `projects/01-enquiry-classifier/CODE_WALKTHROUGH.md`
  - `projects/01-enquiry-classifier/reports/app_screenshot.png`
  - `projects/01-enquiry-classifier/README.md`
  - `README.md`
  - `LICENSE`
- **Level 9 strands:** L9-CO, L9-AU
- **Technical outcomes:** T-DC, T-PY
- **Reflection:** The walkthrough is the specialist-to-beginner communication piece that the technical README was not trying to be. The screenshot is evidence that the demo actually runs, not just that an `app.py` file exists. Relative paths matter once the repo is on GitHub; absolute `/workspace/...` paths would break there. The screenshot was taken with headless Chrome via Playwright against a local Streamlit server and then the server was stopped. Playwright was not added to `requirements.txt` because it is only a documentation tool, not a project dependency.
- **Assessor notes:**

### 2026-11-14 — Project 2: Irish rent predictor on official RTB / CSO data

- **Time / duration:** evening session, 14 November 2026 (Europe/Dublin); hours not recorded
- **Activity:** Built project 2, a regression model of *published* average monthly rents. Downloaded the real CSO PxStat cube **RIA02** (RTB Average Monthly Rent Report, CC BY 4.0), recorded the API URL, fetch time, cube update stamp and SHA-256, and filtered out aggregate bedroom/property-type bands and county-wide totals. Trained on 2008–2023 (20,285 rows) and tested on a **temporal** hold-out of 2024–2025 (1,065 rows). Compared a persistence baseline, a log-linear regression, and histogram gradient boosting. Reported MAE, RMSE, R², mean bias, and permutation importance. The persistence baseline had the lowest test MAE; that result was kept and used as the app headline rather than discarded. Wrote the technical README (including a Level 9 critical evaluation), a client case study, a code walkthrough, and a Streamlit demo with a real screenshot. Attribution to the RTB and CSO is in the data README, the app, and the case study.
- **Artefacts:**
  - `projects/02-irish-rent-predictor/` (full project)
  - `projects/02-irish-rent-predictor/data/ria02_average_monthly_rent.csv`
  - `projects/02-irish-rent-predictor/data/source_metadata.json`
  - `projects/02-irish-rent-predictor/reports/metrics.json`
  - `projects/02-irish-rent-predictor/reports/pred_vs_actual.png`
  - `projects/02-irish-rent-predictor/reports/feature_importance.png`
  - `projects/02-irish-rent-predictor/reports/app_screenshot.png`
- **Level 9 strands:** L9-K, L9-CE, L9-RI, L9-AU, L9-ET, L9-CO
- **Technical outcomes:** T-DP, T-ML, T-EV, T-PY, T-DC, T-ET
- **Reflection:** Using real official statistics, with a CC BY credit, is a different kind of evidence from project 1's synthetic text. The more important Level 9 moment was that the naive persistence baseline (MAE €160.71, R² 0.8580) beat gradient boosting (€172.40) and log-linear regression (€198.42) on later years, and that almost all of that error was a downward bias as rents kept rising. An informal post-hoc check — growing the last figure by the median 2020–2023 year-on-year change of 4.63% — reduced MAE to €84.21, but that check was designed after seeing the residuals and is labelled as such. A first 300-tree random forest produced a 200 MB+ artefact, which is not usable on GitHub; it was replaced with compact histogram gradient boosting (496 KB) before the commit. Year had permutation importance of zero on 2024–2025 because trees cannot extrapolate past the last training year — a finding, not a plotting bug.
- **Assessor notes:**

---

*End of entries for 14 November 2026.*
