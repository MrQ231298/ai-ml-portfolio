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

### 2026-11-14 — Project 3: commercial insurance lead scorer (SYNTHETIC)

- **Time / duration:** evening session, 14 November 2026 (Europe/Dublin); hours not recorded
- **Activity:** Built project 3, a binary conversion classifier on 3,200 invented commercial insurance leads (22.6% converted; no real CRM data). Used a stratified 80/20 split, `class_weight="balanced"` on logistic regression and histogram gradient boosting, and a dummy-prior baseline. Reported ROC-AUC, PR-AUC, precision@k (25/50/100 and top 10%) and a calibration plot. SHAP 0.54.1 installed cleanly and was used for a mean-|SHAP| plot, alongside permutation importance and slice metrics by region, industry and source. Wrote the technical README (including a fairness/ethics section: not underwriting, no protected characteristics, GDPR warning, feedback loops, Dublin over-prediction), a client case study, a code walkthrough, and a Streamlit demo with a real screenshot.
- **Artefacts:**
  - `projects/03-insurance-lead-scorer/` (full project)
  - `projects/03-insurance-lead-scorer/reports/metrics.json`
  - `projects/03-insurance-lead-scorer/reports/pr_curve.png`
  - `projects/03-insurance-lead-scorer/reports/calibration.png`
  - `projects/03-insurance-lead-scorer/reports/explainability.png`
  - `projects/03-insurance-lead-scorer/reports/app_screenshot.png`
- **Level 9 strands:** L9-K, L9-CE, L9-RI, L9-AU, L9-ET, L9-CO
- **Technical outcomes:** T-DP, T-ML, T-EV, T-PY, T-DC, T-ET
- **Reflection:** The headline lift is real but modest (gradient boosting ROC-AUC 0.6433, PR-AUC 0.3190 against a 0.2266 no-skill PR baseline; precision@50 = 0.38). Logistic regression was close, and actually better at precision@100. Class weights helped ranking and visibly *worsened* calibration — that is reported, not hidden. A first draft of the PR plot included the dummy classifier as a triangle up to precision=1, which would have been misleading; it was redrawn so the no-skill line is the prevalence. Slice analysis showed Dublin leads over-scored (mean predicted 0.37 vs actual 0.22, ROC-AUC 0.56), which is the fairness finding I would want an assessor to see.
- **Assessor notes:**

### 2026-11-14 — Project 4: weekly tyre demand forecast (SYNTHETIC)

- **Time / duration:** evening session, 14 November 2026 (Europe/Dublin); hours not recorded
- **Activity:** Built project 4, a weekly sales forecast on 365 invented weeks (2019–2025) with a planted trend, winter-tyre seasonality, a spring NCT-inspired bump, a summer lull, a short 2020 dip, and noise. Compared seasonal naive, ETS (Holt–Winters, period 52) and lag-based gradient boosting under an expanding-window back-test (8-week horizon, 4-week step, 51 origins) plus a single 2025 hold-out. The two protocols disagreed on the winner (ETS on the back-test, gradient boosting on the one-year hold-out); both are reported and the back-test is treated as the headline. Wrote the technical README, case study, walkthrough, and a Streamlit demo with a real screenshot.
- **Artefacts:**
  - `projects/04-tyre-demand-forecast/` (full project)
  - `projects/04-tyre-demand-forecast/reports/metrics.json`
  - `projects/04-tyre-demand-forecast/reports/forecast_plot.png`
  - `projects/04-tyre-demand-forecast/reports/backtest_mae_mape.png`
  - `projects/04-tyre-demand-forecast/reports/app_screenshot.png`
- **Level 9 strands:** L9-K, L9-CE, L9-RI, L9-AU, L9-CO
- **Technical outcomes:** T-DP, T-ML, T-EV, T-PY, T-DC
- **Reflection:** The important result is not a spectacular MAPE (all three models sit around 5% on this tidy invented series) but the protocol effect: a single 2025 hold-out would have declared gradient boosting the winner (MAE 47.57 vs ETS 53.95), while the rolling back-test prefers ETS (48.28 vs 53.01 vs 54.47). Seasonal naive was close enough to be a serious baseline. ETS also raised convergence warnings on some origins, which is recorded as a limitation. The series is synthetic, so the 5% error is not a live-depot claim.
- **Assessor notes:**


### 2026-11-14 — Project 5: tenancy Q&A over public CI / RTB guidance

- **Time / duration:** evening session, 14 November 2026 (Europe/Dublin); hours not recorded
- **Activity:** Built project 5, an extractive retrieval Q&A over a snapshot of 10 public Irish tenancy guidance pages (4 Citizens Information, 6 RTB), covering landlord obligations, registration, deposits, notice periods and the 14 September 2026 law changes. Each page was saved with its source URL and fetch time. Citizens Information returned HTTP 403 to a custom educational User-Agent and served the same public pages to a standard browser User-Agent; that is documented, and the committed snapshot uses the successful fetch. Pages were chunked into 148 passages and embedded with sentence-transformers `all-MiniLM-L6-v2` (installed cleanly; no paid API). A TF-IDF cosine baseline was built for comparison. Evaluated on 16 hand-written questions with expected source pages. Wrote the technical README (including a Level 9 critical evaluation and a prominent not-legal-advice / copyright note), a client case study, a code walkthrough, and a Streamlit demo with a real screenshot.
- **Artefacts:**
  - `projects/05-tenancy-qa-assistant/` (full project)
  - `projects/05-tenancy-qa-assistant/data/manifest.json`
  - `projects/05-tenancy-qa-assistant/eval/questions.json`
  - `projects/05-tenancy-qa-assistant/reports/metrics.json`
  - `projects/05-tenancy-qa-assistant/reports/retrieval_metrics.png`
  - `projects/05-tenancy-qa-assistant/reports/app_screenshot.png`
- **Level 9 strands:** L9-K, L9-CE, L9-RI, L9-AU, L9-ET, L9-CO
- **Technical outcomes:** T-DP, T-ML, T-EV, T-PY, T-DC, T-ET
- **Reflection:** MiniLM recorded hit@1 / hit@3 / hit@5 / MRR of 1.0000 on the 16-question set; TF-IDF scored 0.9375 (it missed a rent-review notice question and retrieved a notice of termination page). Those perfect MiniLM figures were kept and then *qualified*: the questions were written after reading the pages, the corpus is only ten pages, gold labels are page-level, and a hit does not mean the passage settles a legal case. Rank-2 distractors (a deposit question also surfacing a registration-fee passage) and mid-sentence chunks are recorded as limits, not hidden. The ethics boundary — not legal advice, publishers' copyright, no case papers in the demo — is the other Level 9 piece of this project.
- **Assessor notes:**

---

*End of entries for 14 November 2026.*

---

### 2026-11-15 — Project 6: document Q&A with hybrid retrieval, re-ranking and abstention (SYNTHETIC documents)

- **Time / duration:** afternoon session, 15 November 2026 (Europe/Dublin); hours not recorded
- **Activity:** Built a retrieval-augmented question-answering system over a SYNTHETIC eight-document pack for a fictional facilities company (Markdown, PDF and text). Implemented ingestion with heading detection (including a heuristic for PDFs), overlapping chunking, BM25 from scratch, TF-IDF, dense MiniLM retrieval, reciprocal-rank fusion and cross-encoder re-ranking; an extractive answer with citations; an abstention rule; and an optional OpenAI-compatible LLM hook (not evaluated). Wrote 44 paraphrased evaluation questions (36 answerable, 8 not in the documents) split into dev and test. A first version leaked test scores into the threshold candidate grid; this was caught and fixed so that the threshold is chosen from dev scores only.
- **Artefacts:**
  - `projects/06-document-qa-rag/` (full project)
  - `projects/06-document-qa-rag/eval/questions.json`
  - `projects/06-document-qa-rag/reports/metrics.json`, `answer_eval.csv`, `retrieval_ranks.csv`
  - `projects/06-document-qa-rag/reports/retrieval_metrics.png`, `abstention_threshold.png`, `app_screenshot.png`
- **Level 9 strands:** L9-K, L9-CE, L9-RI, L9-AU, L9-ET, L9-CO
- **Technical outcomes:** T-DP, T-ML, T-EV, T-PY, T-DC, T-ET
- **Reflection:** Section-level hit@1 rose from 0.5556 (BM25) and 0.8611 (dense) to 0.9444 with cross-encoder re-ranking. Hybrid fusion on its own *reduced* hit@1 to 0.7778, contradicting common advice for this corpus and question style. On the test split, answer accuracy was 0.8889 but only 1 of 4 unanswerable questions was declined: a single score threshold tuned on four negative examples is not a reliable "I don't know" mechanism. The next step is answer verification and a larger, client-written question set.
- **Assessor notes:**

### 2026-11-15 — Project 7: calibrated, explainable churn model with a cost-based threshold (public IBM sample)

- **Time / duration:** afternoon session, 15 November 2026 (Europe/Dublin); hours not recorded
- **Activity:** Downloaded the IBM Telco Customer Churn sample (Apache 2.0) with provenance and checksum, fixed the blank `TotalCharges` issue with an assertion-backed rule, and excluded gender and senior-citizen status from model inputs. Ran 5-fold stratified CV on the training set to compare a dummy, logistic regression and an 8-point gradient boosting grid; calibrated the chosen model (isotonic vs sigmoid, selected by out-of-fold Brier); chose a contact threshold that maximises an illustrative retention net benefit on out-of-fold predictions; explained the model with SHAP and permutation importance; audited test metrics by gender and senior-citizen status. Built a Streamlit app with per-customer SHAP reasons.
- **Artefacts:**
  - `projects/07-telco-churn-explainable/` (full project)
  - `projects/07-telco-churn-explainable/reports/metrics.json`
  - `projects/07-telco-churn-explainable/reports/calibration.png`, `cost_curve.png`, `shap_importance.png`, `permutation_importance.png`, `app_screenshot.png`
- **Level 9 strands:** L9-K, L9-CE, L9-RI, L9-AU, L9-ET, L9-CO
- **Technical outcomes:** T-DP, T-ML, T-EV, T-PY, T-DC, T-ET
- **Reflection:** Test ROC-AUC 0.8466 for the calibrated booster vs 0.8408 for logistic regression — a difference smaller than the CV standard deviation, so the more complex model is not clearly justified. Calibration made no measurable improvement because the booster was already well calibrated. The real value was the decision layer: the cost-optimal threshold (0.17) produced 60,051 vs 43,754 illustrative net benefit at 0.5. The fairness audit showed weaker ranking for senior citizens (ROC-AUC 0.79 vs 0.85) even though the attribute was excluded — a concrete example of proxy effects.
- **Assessor notes:**

### 2026-11-15 — Project 8: invoice field extraction, unseen-layout and OCR evaluation (SYNTHETIC invoices)

- **Time / duration:** afternoon session, 15 November 2026 (Europe/Dublin); hours not recorded
- **Activity:** Generated 240 SYNTHETIC invoices in four layouts with ground truth (layout D held out entirely). Extracted text with `pdftotext -layout`, and from simulated scans (rotation, blur, noise) with RapidOCR after Tesseract could not be installed (apt mirror unreachable — documented). Wrote a rule-based extractor from layouts A–C only, a candidate-ranking ML extractor with layout-independent features (amount arithmetic, VAT ratios, date order, left-context words), and a hybrid. Evaluated per field on known and unseen layouts, digital and OCR, plus a 5-example adaptation experiment and an arithmetic self-check.
- **Artefacts:**
  - `projects/08-invoice-extraction/` (full project)
  - `projects/08-invoice-extraction/data/SYNTHETIC_invoice_ground_truth.csv`
  - `projects/08-invoice-extraction/reports/metrics.json`, `extracted_fields.csv`, `sample_output_unseen_D_hybrid.csv`
  - `projects/08-invoice-extraction/reports/field_accuracy_hybrid.png`, `extractor_comparison.png`, `app_screenshot.png`
- **Level 9 strands:** L9-K, L9-CE, L9-RI, L9-AU, L9-ET, L9-CO
- **Technical outcomes:** T-DP, T-ML, T-EV, T-PY, T-DC, T-ET
- **Reflection:** Known layouts were easy (hybrid 0.9931 mean field accuracy on OCR). The held-out layout was the honest test: rules fell to 0.3750, the ML ranker reached 0.8750, but no invoice had all eight fields right because the invoice number was always confused with an order reference, and the arithmetic check could not detect it. Five labelled examples raised the ranker to 1.0000 on that layout's digital PDFs. Two generic fixes (a rule gap in layout A and a VAT-number regex for OCR) were prompted by test-set errors; that is disclosed in the README rather than hidden.
- **Assessor notes:**

### 2026-11-15 — Project 9: multi-series retail forecasting with a global model and bootstrap test (CSO data)

- **Time / duration:** afternoon session, 15 November 2026 (Europe/Dublin); hours not recorded
- **Activity:** Fetched CSO Retail Sales Index tables RSM05 and RSM08 (CC BY 4.0). Discovered that a single splice ratio between the two base years distorted seasonality (ratio CV up to 36.8%) and replaced it with calendar-month-specific ratios, with diagnostics saved. Back-tested seven methods over 25 rolling origins × 13 sectors × 6 horizons (1,950 forecasts each): naive, seasonal naive, growth-adjusted seasonal naive, Holt–Winters, a level-based global gradient boosting model, a year-on-year global gradient boosting model excluding lockdown months, and an ensemble. Added empirical 80% intervals with out-of-sample coverage, an origin-block bootstrap against seasonal naive, and a Streamlit/Altair dashboard.
- **Artefacts:**
  - `projects/09-retail-sales-forecast/` (full project)
  - `projects/09-retail-sales-forecast/data/source_metadata.json` (chain-link diagnostics)
  - `projects/09-retail-sales-forecast/reports/metrics.json`, `round1_metrics.json`, `backtest_forecasts.csv`, `future_forecast.csv`
  - `projects/09-retail-sales-forecast/reports/mase_by_horizon.png`, `forecast_examples.png`, `app_screenshot.png`
- **Level 9 strands:** L9-K, L9-CE, L9-RI, L9-AU, L9-CO
- **Technical outcomes:** T-DP, T-ML, T-EV, T-PY, T-DC
- **Reflection:** In round 1, seasonal naive beat every model. The round-2 year-on-year ensemble reached MASE 0.5142 vs 0.5432 for seasonal naive (about 5% better), but the bootstrap 95% interval for the ratio (0.891–1.012) includes 1, and round 2 was designed after seeing round 1. I report the result as suggestive, not proven, and name a fresh-period re-test as the clean validation. Holt–Winters (MASE 0.9625) showed how badly a structural break (2020–21 lockdowns) can damage a classical model.
- **Assessor notes:**

### 2026-11-15 — Project 10: review sentiment classification and topic insights (public CC0 reviews)

- **Time / duration:** afternoon session, 15 November 2026 (Europe/Dublin); hours not recorded
- **Activity:** Downloaded 23,486 public, anonymised clothing reviews (CC0) with checksum; derived three sentiment classes from stars (22,641 reviews with text). Compared a dummy, a CV-tuned TF-IDF (word + character n-gram) logistic regression and MiniLM embeddings + logistic regression on a held-out test set; produced a confusion matrix and misclassified examples. Built NMF topic models for k = 6–15, chose k by NPMI coherence implemented from scratch, measured seed stability, and generated an insights report linking topics to negativity, recommendations and departments. Built a Streamlit app.
- **Artefacts:**
  - `projects/10-review-sentiment-topics/` (full project)
  - `projects/10-review-sentiment-topics/reports/sentiment_metrics.json`, `topic_metrics.json`, `topic_summary.csv`, `insights_report.md`
  - `projects/10-review-sentiment-topics/reports/confusion_matrix.png`, `coherence_sweep.png`, `topics_negativity.png`, `app_screenshot.png`
- **Level 9 strands:** L9-K, L9-CE, L9-RI, L9-AU, L9-ET, L9-CO
- **Technical outcomes:** T-DP, T-ML, T-EV, T-PY, T-DC, T-ET
- **Reflection:** TF-IDF + LR reached macro-F1 0.6679 and beat frozen MiniLM embeddings (0.6134): phrase-level cues matter more for sentiment than general sentence meaning. Negative vs positive is easy (ROC-AUC 0.9725); the neutral class (F1 0.4704) is limited partly by noisy star labels. Topic results are useful but fragile — coherence was still rising at the largest k tried and seed stability was 0.63 — so the report frames topics as a reading aid and uses association language, not causal claims.
- **Assessor notes:**

---

*End of entries for 15 November 2026.*
