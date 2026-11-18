# Learning outcomes map (QQI / NFQ Level 9)

**Purpose.** A working table that can be pasted into, or relabelled for,
an RPL application. The target is a **QQI Level 9 (Master's)** programme.
The exact award and module list are not yet confirmed, so this map uses
the *generic* NFQ Level 9 indicators (and the usual Master's expectation
that you can brief both specialists and non-specialists). When a course
handbook is chosen, add a column for that programme's module outcomes
rather than rewriting the projects.

**How to read the status column.**

- `evidenced (emerging)` — this portfolio already contains a concrete
  artefact, but a Level 9 assessor would reasonably want it deepened
  (more literature, stronger validation, or a more authentic data
  setting).
- `evidenced` — used from project 6 onward where the artefact rests on a
  documented protocol (tuning separated from testing, baselines,
  disclosed limitations). Still subject to a specific programme's review.
- `partial` — some evidence exists; a clear gap is noted.
- `planned` — reserved for a later project or a revision of this one.

**Honesty note.** Project 1 is a classical TF-IDF baseline, and project 2
is a standard tabular regression, not work at the forefront of machine
learning. Level 9 alignment here is mainly in *how the work is
evaluated, documented, and ethically framed* — including, in project 2,
reporting that a persistence baseline beat the machine-learning models
on a temporal hold-out.

NFQ wording below is paraphrased from the public Grid of Level
Indicators for Level 9 (knowledge; know-how and skill; competence) so
the table stays editable and is not a substitute for the official grid
or a specific programme document.

---

## 1. NFQ Level 9 descriptors → evidence

| ID | NFQ Level 9 strand (generic) | What this means in this portfolio | Project | Evidence file(s) | Status | Gap / how to strengthen |
| --- | --- | --- | --- | --- | --- | --- |
| L9-K | **Advanced knowledge** — systematic understanding of a field, informed by current practice (Knowledge: breadth / kind) | Explains and applies a standard supervised text-classification pipeline (problem framing, TF-IDF features, linear models, a dummy baseline, train/test hygiene) | 01-enquiry-classifier | `projects/01-enquiry-classifier/README.md` (Problem, Approach); `projects/01-enquiry-classifier/src/train.py` | evidenced (emerging) | Add a short literature / current-practice note (classical baselines vs rules vs small transformers) and why this baseline was chosen first |
| L9-CE | **Critical evaluation** — critical awareness of current problems, limits of method, and the quality of evidence (Knowledge: kind; Learning to learn) | Reports only run-derived metrics; compares models to a dummy baseline; analyses the four misclassifications; explains why synthetic-data scores are optimistic; states what would falsify the result | 01-enquiry-classifier | `projects/01-enquiry-classifier/README.md` § Critical evaluation; `projects/01-enquiry-classifier/reports/metrics.json`; `projects/01-enquiry-classifier/reports/confusion_matrix_logistic_regression.png`; `projects/01-enquiry-classifier/reports/classification_reports.txt` | evidenced (emerging) | Replace the single split with stratified k-fold; add a non-template held-out set; discuss cost-sensitive errors |
| L9-RI | **Research-informed practice** — uses standard and specialised tools and techniques of enquiry; selects methods on purpose (Know-how & skill: range / selectivity) | Seeded, leak-free split; stratified sampling because of mild imbalance; macro-F1 as well as accuracy; LinearSVC as a second model; joblib artefacts so the experiment is inspectable | 01-enquiry-classifier | `projects/01-enquiry-classifier/src/train.py`; `projects/01-enquiry-classifier/src/generate_synthetic_data.py`; `projects/01-enquiry-classifier/reports/` | evidenced (emerging) | Document a light method comparison protocol (rules baseline, CV, pre-registered metrics) before any more complex model |
| L9-AU | **Autonomy** — act in an ill-defined professional context; take responsibility for planning, quality and next steps (Competence: context / role / learning to learn) | Scoped a useful desk problem, built a reproducible repo, chose synthetic data rather than taking a GDPR shortcut, and recorded what is *not* claimed | 01-enquiry-classifier; this repository | `README.md`; `projects/01-enquiry-classifier/README.md` § Next steps; `rpl/RPL_EVIDENCE_LOG.md` | evidenced (emerging) | A later project should show an independent brief (for example landlord / insurance lead-gen) taken from an ambiguous prompt to a defended design |
| L9-ET | **Ethics and data protection (incl. GDPR)** — scrutinise professional and social implications and act on them (Competence: insight) | No real customer data; `SYNTHETIC` in file names, CSV flag and docs; training script refuses untagged files; deployment section covers lawful basis, DPIA, minimisation, human review | 01-enquiry-classifier | `projects/01-enquiry-classifier/data/README.md`; `projects/01-enquiry-classifier/data/SYNTHETIC_tyre_desk_enquiries.csv`; `projects/01-enquiry-classifier/src/generate_synthetic_data.py`; `projects/01-enquiry-classifier/src/train.py` (source-flag guard); `projects/01-enquiry-classifier/README.md` § What a real deployment would need; `projects/01-enquiry-classifier/app.py` (warning not to paste personal data) | evidenced (emerging) | Evidence of a real DPIA / data-sharing protocol if a lawful live-data pilot ever runs; keep that work *out* of a public repo |
| L9-CO | **Communication to specialist and non-specialist audiences** | Specialist: commented code, metrics, confusion matrices, critical evaluation. Non-specialist: client case study and a Streamlit demo with confidence scores in plain language | 01-enquiry-classifier | `projects/01-enquiry-classifier/README.md`; `projects/01-enquiry-classifier/CODE_WALKTHROUGH.md`; `projects/01-enquiry-classifier/CASE_STUDY.md`; `projects/01-enquiry-classifier/app.py`; `projects/01-enquiry-classifier/reports/app_screenshot.png`; `projects/01-enquiry-classifier/src/*.py` | evidenced (emerging) | Optional short slide or walk-through recording for an RPL interview; keep the two-audience split on every future project |

---

## 2. Technical learning outcomes (relabel these to a module handbook)

These are the practical computing / data-analytics outcomes that many
Level 6–8 *and* Level 9 programmes will still want to see. They sit
underneath the Level 9 strands above, so an assessor can map either way.

| ID | Technical outcome | Linked Level 9 strand(s) | Project | Evidence file(s) | Status |
| --- | --- | --- | --- | --- | --- |
| T-DP | **Data preparation** — define a class scheme, generate or acquire a labelled set, document provenance, handle class imbalance in the split | L9-RI, L9-ET | 01-enquiry-classifier | `src/generate_synthetic_data.py`; `data/SYNTHETIC_tyre_desk_enquiries.csv`; `data/README.md`; `src/train.py` (stratified split) | evidenced (emerging) |
| T-ML | **Supervised machine learning** — frame a classification task; train more than one learner on shared features; persist a model | L9-K, L9-RI | 01-enquiry-classifier | `src/train.py`; `models/tfidf_logistic_regression.joblib`; `models/tfidf_logistic_regression_pipeline.joblib` | evidenced (emerging) |
| T-EV | **Model evaluation** — dummy baseline, accuracy, macro- and weighted-F1, classification report, confusion matrix, error review | L9-CE, L9-RI | 01-enquiry-classifier | `reports/metrics.json`; `reports/classification_reports.txt`; `reports/confusion_matrix_logistic_regression.png`; `reports/confusion_matrix_linear_svc.png`; README § Results and § Critical evaluation | evidenced (emerging) |
| T-PY | **Python programming** — readable, commented scripts and a small app; virtualenv and pinned requirements; reproducible seeds | L9-AU | 01-enquiry-classifier | `src/generate_synthetic_data.py`; `src/train.py`; `app.py`; `requirements.txt` | evidenced (emerging) |
| T-DC | **Documentation and communication** — technical README, client case study, RPL log | L9-CO | 01-enquiry-classifier; rpl | `projects/01-enquiry-classifier/README.md`; `CODE_WALKTHROUGH.md`; `CASE_STUDY.md`; `reports/app_screenshot.png`; `rpl/RPL_EVIDENCE_LOG.md`; this file | evidenced (emerging) |
| T-ET | **Ethics / GDPR** — synthetic data, no live PII, deployment obligations named | L9-ET | 01-enquiry-classifier | `data/README.md`; README § Critical evaluation; `app.py` notice | evidenced (emerging) |
| T-SQL | **SQL and relational modelling** — star schemas, keyed upserts, analytical SQL, cross-checks against a second implementation (added 17 Nov 2026) | L9-RI | 11, 12, 14 | `projects/11-manufacturing-kpi-dashboard/src/kpi_sql.py`; `projects/12-sql-python-etl-pipeline/sql/`; `projects/14-azure-fabric-medallion-lakehouse/src/gold.py` | evidenced |
| T-DE | **Data engineering** — incremental and idempotent pipelines, quarantine and retry, data-quality gates, lineage, lakehouse layers (added 17 Nov 2026) | L9-RI, L9-CE | 12, 14 | `projects/12-sql-python-etl-pipeline/src/etl/`; `projects/14-azure-fabric-medallion-lakehouse/src/`; both `reports/metrics.json` | evidenced |
| T-BI | **Business intelligence and KPI design** — KPI definitions, dashboards, Power BI data model and DAX (added 17 Nov 2026) | L9-CO, L9-K | 11, 13 | `projects/11-manufacturing-kpi-dashboard/powerbi/`; both `app.py` | evidenced (emerging): DAX not executed |
| T-CL | **Cloud data platform design** — Azure / Fabric service mapping, infrastructure as code (added 17 Nov 2026) | L9-K, L9-AU | 14 | `projects/14-azure-fabric-medallion-lakehouse/docs/AZURE_FABRIC_MAPPING.md`; `infra/main.bicep` | partial: designed and compiled, never deployed |
| T-LLM | **LLM applications** — tool-using agents, structured output, parameter-efficient fine-tuning (LoRA), evaluation on held-out phrasings (added 18 Nov 2026) | L9-K, L9-RI, L9-CE | 16 | `projects/16-tool-agent-lora/src/agent.py`; `src/finetune_lora.py`; `reports/metrics.json` | evidenced (80-task subset) |
| T-MLOPS | **MLOps** — experiment tracking, model registry, model serving, drift monitoring, retraining policy, CI and containers (added 17 Nov 2026) | L9-RI, L9-AU | 17 | `projects/17-mlops-model-serving/src/`; `app/main.py`; `Dockerfile`; `ci/` | evidenced (Docker and CI written, not run) |
| T-STREAM | **Stream processing** — Kafka producers and consumers, event-time windows, watermarks, late data, exactly-once results (added 17 Nov 2026) | L9-K, L9-RI | 19 | `projects/19-realtime-streaming-pipeline/src/`; `reports/metrics.json` | evidenced (single broker) |
| T-CV | **Computer vision** — CNN training and transfer learning, anomaly detection on images, Grad-CAM explainability (added 17 Nov 2026) | L9-K, L9-CE | 20 | `projects/20-pcb-defect-vision/src/`; `reports/metrics.json`; `reports/gradcam_grid.png` | evidenced |
| T-PM | **Predictive maintenance / time series** — causal features, unsupervised health scoring, remaining-useful-life regression, cost-based alerting (added 18 Nov 2026) | L9-K, L9-RI, L9-CE | 21 | `projects/21-predictive-maintenance-cmapss/src/`; `reports/metrics.json` | evidenced |

---

## 3. How to adjust this map later

1. When a programme is confirmed, add a column **Programme module /
   learning outcome code** and map each `L9-*` and `T-*` row to it.
   Some rows will map to more than one module; that is normal.
2. If an outcome is not yet evidenced, leave the project cell as
   `planned` and add a line to the evidence log when the work is done.
3. Do not retrofit metrics or invent artefacts to fill a cell. Point at
   a file that exists, or mark the gap.
4. Further domain projects should add a section or rows, not overwrite
   earlier evidence. Project 2 (official Irish rents) is summarised
   below.

---

## 4. Project 2 additions (02-irish-rent-predictor)

Real public data: RTB Average Monthly Rent Report, CSO PxStat table
RIA02, CC BY 4.0. Temporal hold-out 2008–2023 / 2024–2025.

| ID | How project 2 adds evidence | Evidence file(s) | Status |
| --- | --- | --- | --- |
| L9-K | Tabular regression on official averages; log-linear and gradient-boosting pipelines; target clearly defined as a *published average*, not a unit rent | `projects/02-irish-rent-predictor/README.md`; `src/train.py` | evidenced (emerging) |
| L9-CE | Temporal hold-out; baseline included and *allowed to win* (persistence MAE €160.71 vs boosting €172.40 vs log-linear €198.42); bias analysis; year-importance of zero explained; post-hoc growth check labelled as informal | `README.md` § Results and § Critical evaluation; `reports/metrics.json`; `reports/pred_vs_actual.png`; `reports/mae_by_year.png` | evidenced (emerging) |
| L9-RI | Public API fetch with licence, fetch stamp and SHA-256; filters documented; permutation importance; compact model chosen after a 200 MB random-forest artefact proved unusable | `src/fetch_cso_ria02.py`; `data/source_metadata.json`; `data/README.md`; `src/train.py` | evidenced (emerging) |
| L9-AU | Independent landlord-domain brief; headline model chosen by test MAE, not by novelty | `README.md` § Next steps; `app.py` | evidenced (emerging) |
| L9-ET | CC BY 4.0 attribution to RTB / CSO; no tenancy microdata; app and case study state this is not a valuation or advice | `data/README.md`; `CASE_STUDY.md`; `app.py` disclaimer | evidenced (emerging) |
| L9-CO | Technical README, beginner walkthrough, client case study, working demo screenshot | `README.md`; `CODE_WALKTHROUGH.md`; `CASE_STUDY.md`; `reports/app_screenshot.png` | evidenced (emerging) |
| T-DP | Download, tidy and filter a JSON-stat cube; derive county / location level; drop nested aggregates | `src/fetch_cso_ria02.py`; `data/ria02_average_monthly_rent.csv` | evidenced (emerging) |
| T-ML | Persistence, log-linear regression, histogram gradient boosting; saved joblib bundle | `src/baseline.py`; `src/train.py`; `models/rent_predictor.joblib` | evidenced (emerging) |
| T-EV | MAE, RMSE, R², mean bias, permutation importance, predicted-vs-actual and MAE-by-year plots | `reports/` | evidenced (emerging) |
| T-PY / T-DC | Commented scripts, pinned requirements, two-audience docs | `requirements.txt`; `app.py`; docs as above | evidenced (emerging) |

---

## 5. Project 3 additions (03-insurance-lead-scorer)

SYNTHETIC commercial insurance leads, 3,200 rows, 22.6% conversion.
Imbalance-aware evaluation; SHAP + permutation importance; ethics note.

| ID | How project 3 adds evidence | Evidence file(s) | Status |
| --- | --- | --- | --- |
| L9-K | Binary classification under class imbalance; ROC/PR, precision@k, calibration; sales-score vs underwriting distinction | `projects/03-insurance-lead-scorer/README.md`; `src/train.py` | evidenced (emerging) |
| L9-CE | Modest lift reported honestly; calibration failure of class-weighted probabilities shown; P@k compared across models; synthetic-label limit stated | `README.md` § Results and § Critical evaluation; `reports/calibration.png`; `reports/metrics.json` | evidenced (emerging) |
| L9-RI | Documented generative process; SHAP 0.54.1 used when it installed cleanly, with permutation importance as a companion; slice analysis by region / source | `src/generate_synthetic_leads.py`; `reports/explainability.png`; `reports/permutation_importance.png` | evidenced (emerging) |
| L9-ET | No protected characteristics; explicit "not underwriting" boundary; GDPR warning for live prospect data; Dublin over-prediction discussed as an operational fairness issue | `README.md` § Fairness and ethics; `CASE_STUDY.md`; `app.py` disclaimer | evidenced (emerging) |
| L9-CO | Technical README, walkthrough, client case study, demo screenshot | `README.md`; `CODE_WALKTHROUGH.md`; `CASE_STUDY.md`; `reports/app_screenshot.png` | evidenced (emerging) |
| T-DP / T-ML / T-EV / T-PY / T-DC | Synthetic table with a known process; LR + HGB + dummy; ROC/PR/P@k/calibration; commented scripts and two-audience docs | `src/`; `reports/`; `app.py`; `requirements.txt` | evidenced (emerging) |

---

## 6. Project 4 additions (04-tyre-demand-forecast)

SYNTHETIC weekly tyre sales, 365 weeks (2019–2025). Expanding-window
back-test plus a 2025 hold-out.

| ID | How project 4 adds evidence | Evidence file(s) | Status |
| --- | --- | --- | --- |
| L9-K | Classical ETS vs lag-based gradient boosting vs seasonal naive; weekly seasonality of period 52; recursive multi-step forecasts | `projects/04-tyre-demand-forecast/README.md`; `src/train.py` | evidenced (emerging) |
| L9-CE | Headline protocol is the back-test, not the prettier single hold-out; the two disagree and that is reported; seasonal naive kept as a serious baseline; ETS convergence warnings noted | `README.md` § Results and § Critical evaluation; `reports/metrics.json`; `reports/forecast_plot.png`; `reports/backtest_mae_mape.png` | evidenced (emerging) |
| L9-RI | Expanding window, fixed horizon and step, 51 origins; MAE and MAPE; planted seasonal design documented in the generator | `src/generate_synthetic_series.py`; `reports/backtest_origins.csv` | evidenced (emerging) |
| L9-CO | Technical README, walkthrough, client case study, demo screenshot | `README.md`; `CODE_WALKTHROUGH.md`; `CASE_STUDY.md`; `reports/app_screenshot.png` | evidenced (emerging) |
| T-ML / T-EV / T-PY / T-DC | statsmodels ETS, sklearn HGB with lags, time-based evaluation, commented scripts, two-audience docs | `src/`; `reports/`; `app.py`; `requirements.txt` | evidenced (emerging) |


## 7. Project 5 additions (05-tenancy-qa-assistant)

REAL public web snapshot: 10 Citizens Information / RTB guidance
pages, fetched 14 November 2026, 148 passages. Extractive retrieval
with citations; MiniLM vs TF-IDF; not legal advice.

| ID | How project 5 adds evidence | Evidence file(s) | Status |
| --- | --- | --- | --- |
| L9-K | Local embedding-based retrieval (all-MiniLM-L6-v2) with a TF-IDF baseline; extractive passages and provenance rather than a generative answer | `projects/05-tenancy-qa-assistant/README.md`; `src/chunk_and_index.py`; `src/retrieve.py` | evidenced (emerging) |
| L9-CE | Perfect MiniLM hit@1 / MRR of 1.0000 reported *and* qualified (author-written questions, small closed corpus, page-level gold, distractor passages); TF-IDF miss on a rent-review vs termination confusion kept in the record | `README.md` § Results and § Critical evaluation; `reports/metrics.json`; `reports/eval_results.csv`; `reports/retrieval_metrics.png` | evidenced (emerging) |
| L9-RI | Hand-picked official sources with URL + fetch stamp; documented 403/User-Agent issue; 16-question evaluation with pre-specified gold pages; no paid API | `src/fetch_pages.py`; `data/manifest.json`; `data/README.md`; `eval/questions.json`; `src/evaluate.py` | evidenced (emerging) |
| L9-ET | Prominent not-legal-advice disclaimer; publishers' copyright respected; no tenant case files; warning not to paste personal documents | `README.md` § Ethics; `CASE_STUDY.md`; `app.py` disclaimer; `data/README.md` | evidenced (emerging) |
| L9-CO | Technical README, walkthrough, client case study, demo screenshot with visible disclaimer and citations | `README.md`; `CODE_WALKTHROUGH.md`; `CASE_STUDY.md`; `reports/app_screenshot.png` | evidenced (emerging) |
| T-DP / T-ML / T-EV / T-PY / T-DC / T-ET | Public-page snapshot and chunking; MiniLM + TF-IDF; hit@k and MRR; commented scripts; two-audience docs; ethics/copyright framing | `src/`; `data/`; `reports/`; `app.py`; `requirements.txt` | evidenced (emerging) |


---

## 8. Advanced set (projects 6–10, 15 November 2026)

Projects 6–10 were built to move the evidence from *emerging* toward
Master's-level practice: more than one method compared under a
pre-defined protocol, explicit separation of tuning and test data,
statistical or adaptation experiments, and disclosure where the protocol
was imperfect. Status stays `evidenced (emerging)` where the data are
synthetic or the evaluation set is small.

### 8.1 Project 6 additions (06-document-qa-rag)

SYNTHETIC eight-document pack, 47 chunks, 44 questions (dev/test).

| ID | How project 6 adds evidence | Evidence file(s) | Status |
| --- | --- | --- | --- |
| L9-K | Retrieval-augmented generation architecture: lexical (BM25 implemented from the formula), dense bi-encoder, reciprocal-rank fusion, cross-encoder re-ranking, extractive answering, LLM hook | `projects/06-document-qa-rag/src/rag.py`; `README.md` § Approach | evidenced |
| L9-CE | Hybrid fusion shown to *hurt* without re-ranking; abstention shown to be weak (1 of 4 unanswerable declined on test); threshold-protocol revision disclosed | `README.md` § Critical evaluation; `reports/metrics.json`; `reports/answer_eval.csv` | evidenced |
| L9-RI | Pre-written question set with gold sections and key facts; dev/test split; threshold chosen on dev only; five retrievers on identical chunks | `eval/questions.json`; `src/evaluate.py`; `reports/retrieval_ranks.csv` | evidenced (emerging) |
| L9-AU | Self-directed design choices (local models, no paid API, mixed formats) and self-detected protocol leak fixed | `rpl/RPL_EVIDENCE_LOG.md` 2026-11-15; `README.md` | evidenced (emerging) |
| L9-ET | Synthetic company documents; local-only default; DPIA / processor note for hosted LLMs | `data/README.md`; `README.md` § Critical evaluation | evidenced (emerging) |
| L9-CO | README, client case study, walkthrough, demo screenshot with citations | `README.md`; `CASE_STUDY.md`; `CODE_WALKTHROUGH.md`; `reports/app_screenshot.png` | evidenced |

### 8.2 Project 7 additions (07-telco-churn-explainable)

Public IBM Telco sample, 7,043 customers.

| ID | How project 7 adds evidence | Evidence file(s) | Status |
| --- | --- | --- | --- |
| L9-K | Probability calibration (isotonic / Platt, Brier, ECE), cost-sensitive decision thresholds, SHAP vs permutation importance | `projects/07-telco-churn-explainable/src/train.py`; `README.md` | evidenced |
| L9-CE | Boosting gain within CV noise; calibration gave no improvement; economics flagged as assumptions needing a randomised test; uplift modelling named as the correct target | `README.md` § Critical evaluation; `reports/metrics.json` | evidenced |
| L9-RI | All selection (grid, calibration method, threshold) on training-set CV / out-of-fold predictions; test used once | `src/train.py`; `reports/metrics.json` (`cv_5fold_train`, `oof_brier_calibration`) | evidenced |
| L9-ET | Protected attributes excluded from inputs and used for a slice audit; proxy effect on senior citizens reported; GDPR Art. 22 / Equal Status Acts noted | `README.md` § Fairness audit; `src/churn_lib.py` (`AUDIT_COLS`) | evidenced |
| L9-CO | Client case study framed around contact policy and money; app with per-customer reasons | `CASE_STUDY.md`; `app.py`; `reports/app_screenshot.png` | evidenced |

### 8.3 Project 8 additions (08-invoice-extraction)

240 SYNTHETIC invoices, four layouts (one held out), digital + OCR.

| ID | How project 8 adds evidence | Evidence file(s) | Status |
| --- | --- | --- | --- |
| L9-K | Document AI pipeline: PDF text layout, OCR, rule-based vs learned candidate ranking with arithmetic / ratio features, human-in-the-loop check | `projects/08-invoice-extraction/src/extract_lib.py`; `README.md` | evidenced (emerging) |
| L9-CE | Held-out layout result (0% fully-correct invoices) reported as the headline limitation; arithmetic check shown to miss ID errors; two test-prompted fixes disclosed | `README.md` § Results and § Critical evaluation; `reports/metrics.json` | evidenced |
| L9-RI | Out-of-distribution test design (rules and training restricted to layouts A–C); few-shot adaptation experiment; per-field exact-match scoring with documented normalisation | `src/evaluate.py`; `data/README.md`; `reports/extracted_fields.csv` | evidenced |
| L9-AU | Adapted when Tesseract could not be installed (RapidOCR) and documented the substitution | `requirements.txt`; `README.md` § Approach | evidenced (emerging) |
| L9-ET | Synthetic suppliers and VAT numbers; GDPR note on sole-trader invoices; upload warning in app | `data/README.md`; `app.py`; `README.md` | evidenced (emerging) |
| L9-CO | Case study for a finance audience; demo screenshot deliberately showing a failure | `CASE_STUDY.md`; `reports/app_screenshot.png` | evidenced |

### 8.4 Project 9 additions (09-retail-sales-forecast)

CSO Retail Sales Index, 13 sectors, Jan 2015 – Sep 2026.

| ID | How project 9 adds evidence | Evidence file(s) | Status |
| --- | --- | --- | --- |
| L9-K | Global vs local forecasting models, direct multi-horizon strategy, YoY targets, structural-break handling, MASE, empirical intervals | `projects/09-retail-sales-forecast/src/forecast_lib.py`; `README.md` | evidenced |
| L9-CE | Seasonal naive won round 1; round-2 gain (~5%) not significant (bootstrap CI includes 1) and designed after seeing round 1 — both disclosed; ETS failure explained | `README.md` § Critical evaluation; `reports/metrics.json` (`bootstrap_vs_seasonal_naive`, `round1_results_before_yoy_models`) | evidenced |
| L9-RI | Rolling-origin back-test (1,950 forecasts per model); origin-block bootstrap; interval coverage checked on later origins; chain-link diagnostics | `src/backtest.py`; `src/compare_models.py`; `data/source_metadata.json` | evidenced |
| L9-AU | Found and corrected a data-preparation problem (single splice ratio) without being told | `src/fetch_cso_retail.py` docstring; `data/README.md` | evidenced |
| L9-CO | Interactive dashboard; client case study framed as "does it beat last year?" | `app.py`; `CASE_STUDY.md`; `reports/app_screenshot.png` | evidenced |

### 8.5 Project 10 additions (10-review-sentiment-topics)

22,641 public CC0 clothing reviews.

| ID | How project 10 adds evidence | Evidence file(s) | Status |
| --- | --- | --- | --- |
| L9-K | Sparse vs dense text representations; NMF topic modelling; NPMI coherence and stability metrics | `projects/10-review-sentiment-topics/src/*.py`; `README.md` | evidenced |
| L9-CE | Simpler TF-IDF model beat embeddings, explained; label noise in the neutral class; topic fragility (edge-of-grid k, 0.63 stability); association not causation | `README.md` § Critical evaluation; `reports/sentiment_metrics.json`; `reports/topic_metrics.json` | evidenced |
| L9-RI | CV-tuned regularisation; held-out test; coherence-based model selection; seed-stability check; misclassification review | `src/train_sentiment.py`; `src/topics.py`; `reports/misclassified_examples.csv` | evidenced |
| L9-ET | CC0 licence verified; reviewer age not used; personal-data caution for client text | `data/README.md`; `README.md` | evidenced (emerging) |
| L9-CO | Auto-generated insights report for non-specialists; app; case study | `reports/insights_report.md`; `CASE_STUDY.md`; `reports/app_screenshot.png` | evidenced |

| Technical outcome | Projects 6–10 evidence |
| --- | --- |
| T-DP | Mixed-format ingestion (6); data-quality assertion (7); synthetic generator with held-out layout (8); chain-linking two official tables (9); label derivation from ratings (10) |
| T-ML | Retrieval + re-ranking (6); calibrated gradient boosting (7); candidate-ranking classifiers (8); global gradient boosting forecasters (9); text classifiers and NMF (10) |
| T-EV | hit@k / MRR and abstention metrics (6); Brier / ECE / net benefit / fairness slices (7); per-field exact match on OOD and OCR (8); MASE / bootstrap / interval coverage (9); macro-F1 / AUC / NPMI / stability (10) |
| T-PY | Five projects with shared library modules, pinned requirements and `pytest` smoke tests |
| T-DC | README with critical evaluation, `CASE_STUDY.md`, `CODE_WALKTHROUGH.md` and screenshot for each |
| T-ET | Fairness audit (7); synthetic data and upload warnings (6, 8); licence checks (7, 9, 10) |

## 9. Analytics and data-engineering set (projects 11–15, 17 November 2026)

Projects 11–15 widen the evidence from model building to the work
analytics teams do day to day: KPI design and BI, SQL and ETL, data
quality, measurement of marketing effect, lakehouse engineering and
segmentation. The same protocol applies: metrics come only from real
runs, results are checked against a known answer or a hold-out where
possible, and what was *not* done is stated. Projects 11–13 use
SYNTHETIC data with a known answer planted by a seeded generator;
projects 14–15 use REAL public data (UCI, CC BY 4.0). Project 14 ran
locally; nothing was deployed to Azure or Fabric.

### 9.1 Project 11 additions (11-manufacturing-kpi-dashboard)

SYNTHETIC four-line electronics assembly plant, July–October 2026, four planted issues.

| ID | How project 11 adds evidence | Evidence file(s) | Status |
| --- | --- | --- | --- |
| L9-K | OEE decomposition, rolled throughput yield, statistical process control (p- and c-charts), multiple-testing correction | `projects/11-manufacturing-kpi-dashboard/src/kpi_lib.py`; `README.md` § Approach | evidenced |
| L9-CE | Detection scored against planted truth including false alarms (3 / 88 p-chart); a by-design line flagged by the rate scan, discussed; DAX not executed, stated | `README.md` § Critical evaluation; `reports/metrics.json` | evidenced |
| L9-RI | Issues planted and recorded before analysis; two independent KPI implementations agree to 1 × 10⁻¹⁵; bootstrap CI for the shift effect | `data/star_schema/planted_issues.json`; `src/analyse.py`; `src/kpi_sql.py` | evidenced |
| L9-CO | Power BI build guide for a non-developer; Streamlit dashboard; case study | `powerbi/POWERBI_BUILD_GUIDE.md`; `app.py`; `CASE_STUDY.md`; `reports/app_screenshot.png` | evidenced |
| T-BI / T-SQL / T-EV / T-PY | DAX measures and theme; DuckDB SQL; SPC and detection metrics; 6 tests | `powerbi/`; `src/`; `tests/` | evidenced (emerging) for T-BI |

### 9.2 Project 12 additions (12-sql-python-etl-pipeline)

SYNTHETIC two-batch retail extracts with six injected defect types.

| ID | How project 12 adds evidence | Evidence file(s) | Status |
| --- | --- | --- | --- |
| L9-K | Layered warehouse design, SCD type 1, late-arriving dimension handling, idempotency by file hash | `projects/12-sql-python-etl-pipeline/README.md` § Approach; `src/etl/load.py` | evidenced |
| L9-CE | Ordering limit (newest-first load leaves 29 stale counties) found and documented; own generator bug found and fixed; injected defects test my rules, not unknown errors | `README.md` § Critical evaluation; `reports/metrics.json` | evidenced |
| L9-RI | Reconciliation to an independent clean truth (difference 0.00); recall and precision per defect type; MD5 comparison across four load scenarios | `src/evaluate.py`; `reports/metrics.json` | evidenced |
| L9-ET | Invalid emails quarantined; synthetic personal fields only | `data/README.md`; `reports/dq_report.md` | evidenced (emerging) |
| T-DE / T-SQL / T-PY | Modular ETL package, CLI, 15 DQ checks, SQL marts, 18 unit and integration tests | `src/etl/`; `sql/`; `tests/` | evidenced |

### 9.3 Project 13 additions (13-marketing-campaign-dashboard)

SYNTHETIC simulation of 160,000 people, six channels, known causal effect per channel.

| ID | How project 13 adds evidence | Evidence file(s) | Status |
| --- | --- | --- | --- |
| L9-K | Seven attribution models including Markov removal effect and Shapley values; incrementality and randomised hold-out tests | `projects/13-marketing-campaign-dashboard/src/attribution.py`; `README.md` | evidenced |
| L9-CE | Every model scored against causal truth; first-click "win" identified as a simulation artefact; data-driven models no better than linear | `README.md` § Results and § Critical evaluation; `reports/metrics.json` | evidenced |
| L9-RI | Simulation with ground truth chosen deliberately because real data has no known answer; hold-out CI checked against truth | `src/simulate.py`; `data/truth_incrementality.json` | evidenced |
| L9-CO | Dashboard and plain-language insights for a marketing manager | `app.py`; `reports/insights.md`; `CASE_STUDY.md` | evidenced |
| T-BI / T-EV / T-PY | Campaign KPIs; misallocation and rank metrics; 5 tests | `src/analyse.py`; `tests/` | evidenced |

### 9.4 Project 14 additions (14-azure-fabric-medallion-lakehouse)

REAL UCI Online Retail II, 1,067,371 rows. **Run locally with DuckDB and delta-rs; not deployed to Azure or Fabric.**

| ID | How project 14 adds evidence | Evidence file(s) | Status |
| --- | --- | --- | --- |
| L9-K | Medallion architecture, Delta Lake transaction log and time travel, partition overwrite, Azure / Fabric service equivalents | `projects/14-azure-fabric-medallion-lakehouse/docs/AZURE_FABRIC_MAPPING.md`; `src/` | evidenced |
| L9-CE | Three self-found bugs fixed (double count, non-deterministic tie, missing customers); cloud artefacts explicitly not deployed or executed | `README.md` § Critical evaluation; `reports/metrics.json` | evidenced |
| L9-RI | Full vs incremental equivalence by MD5 across 7 tables; replay, correction and time-travel tests; 10 DQ checks | `src/evaluate.py`; `src/dq.py`; `reports/metrics.json` | evidenced |
| L9-AU | Self-directed translation of a cloud design into a locally testable pipeline without a subscription | `README.md`; this log 2026-11-17 | evidenced (emerging) |
| L9-ET | CC BY 4.0 attribution; pseudonymous IDs; raw data not committed | `README.md` § Data status | evidenced |
| T-DE / T-SQL / T-CL | Delta pipeline; DuckDB SQL; Bicep (0 warnings, not deployed), ADF JSON and Fabric notebooks (not executed) | `src/`; `infra/`; `adf/`; `fabric/` | evidenced (T-DE, T-SQL); partial (T-CL) |

### 9.5 Project 15 additions (15-customer-segmentation)

REAL UCI Online Retail, 3,360 customers, calibration and 91-day hold-out windows.

| ID | How project 15 adds evidence | Evidence file(s) | Status |
| --- | --- | --- | --- |
| L9-K | k-means and Gaussian mixtures; internal validity indices; bootstrap ARI stability; RFM as an industry baseline | `projects/15-customer-segmentation/src/segment.py`; `README.md` § Approach | evidenced |
| L9-CE | RFM beat k-means on the hold-out (AUC 0.739 vs 0.661), reported as the finding; weak structure and unstable k = 4 stated; post-hoc naming rule disclosed | `README.md` § Results and § Critical evaluation; `reports/metrics.json` | evidenced |
| L9-RI | Time-split features vs outcomes; k-selection rule fixed in advance; scores built only from calibration data | `src/fetch_and_features.py`; `src/segment.py`; `reports/k_selection.csv` | evidenced |
| L9-ET | CC BY 4.0 attribution; pseudonymous IDs only; derived features committed, raw data not | `data/README.md` | evidenced |
| L9-CO | Plain-language segments, case study, adviser-segmentation transfer note, app | `CASE_STUDY.md`; `README.md` § Client value; `app.py` | evidenced |
| T-ML / T-EV / T-PY | Unsupervised learning, AUC and η² validation, 6 tests | `src/`; `tests/` | evidenced |

### 9.6 Summary for projects 11–15

| Technical outcome | Projects 11–15 evidence |
| --- | --- |
| T-DP | Seeded generators with planted or injected truth (11, 12, 13); checksum-verified public downloads and documented cleaning (14, 15) |
| T-ML | Clustering with stability and hold-out validation (15) |
| T-EV | Detection vs planted issues (11); reconciliation and defect recall / precision (12); attribution vs causal truth and hold-out CI (13); load-equivalence tests (14); hold-out AUC vs baseline (15) |
| T-SQL | DuckDB KPI SQL (11); warehouse and marts (12); gold star schema (14) |
| T-DE | Quarantine, retry, idempotency (12); Delta medallion layers, watermark, correction handling (14) |
| T-BI | Power BI kit (11); KPI dashboards (11, 13, 15) |
| T-CL | Azure / Fabric mapping and Bicep, not deployed (14) |
| T-PY | Five projects, pinned requirements, 40 `pytest` tests in total |
| T-DC | README with critical evaluation, `CASE_STUDY.md`, `CODE_WALKTHROUGH.md` for each |
| T-ET | Synthetic data where no real data could be used (11–13); licence attribution and pseudonymous IDs (14, 15) |

## 10. Production ML and engineering set (projects 16–21, 17–18 November 2026)

Projects 16–21 move from building models to running them: an LLM agent
that acts through tools, a model lifecycle with monitoring, orchestrated
and streaming data pipelines, computer vision for inspection, and
maintenance decisions driven by cost. The protocol is unchanged: metrics
come only from real runs, every result is checked against a baseline or
an independent answer, and what was *not* run is stated (Docker and CI in
17, a multi-broker Kafka cluster in 19, whole-board detection in 20, real
machine data in 21). Projects 20 and 21 draw on experience as a Test
Engineer in automotive electronics manufacturing; no employer data is
used.

### 10.1 Project 16 additions (16-tool-agent-lora)

SYNTHETIC distributor database; SmolLM2-360M-Instruct with LoRA on CPU; 80-task evaluation subset.

| ID | How project 16 adds evidence | Evidence file(s) | Status |
| --- | --- | --- | --- |
| L9-K | Tool-using agents, structured output, parameter-efficient fine-tuning (LoRA), loss masking | `projects/16-tool-agent-lora/src/agent.py`; `src/finetune_lora.py`; `README.md` § Approach | evidenced |
| L9-CE | Failure analysis of OOD errors; template overfitting discussed; subset evaluation and missing hosted-model comparison stated | `README.md` § Results and § Critical evaluation; `reports/eval_rows.csv` | evidenced |
| L9-RI | Gold answers by execution; held-out phrasings and entities; four baselines and a harness check; bootstrap CIs | `src/tasks.py`; `src/evaluate.py`; `reports/metrics.json` | evidenced |
| L9-ET | Read-only SQL and no-`eval` calculator; synthetic data; local model keeps data in-house | `src/tools.py`; `tests/test_tools.py` | evidenced |
| T-LLM / T-SQL / T-PY | Agent, LoRA fine-tune, SQL tool, 10 tests | `src/`; `tests/` | evidenced |

### 10.2 Project 17 additions (17-mlops-model-serving)

REAL UCI Bike Sharing, hourly, 2011–2012.

| ID | How project 17 adds evidence | Evidence file(s) | Status |
| --- | --- | --- | --- |
| L9-K | Model registry and aliases, PSI/KS drift, performance-triggered retraining, champion/challenger promotion | `projects/17-mlops-model-serving/src/monitor.py`; `src/drift.py`; `README.md` § Approach | evidenced |
| L9-CE | Drift alerts fired 9 of 9 months and were useless alone, reported; first feature set lost to the baseline, disclosed; Docker and CI not run, stated | `README.md` § Results and § Critical evaluation; `reports/metrics.json` | evidenced |
| L9-RI | Time-based split; retraining rule fixed before the replay; replay vs never-retrain counterfactual (MAE 37.02 vs 49.65) | `src/train.py`; `src/monitor.py` | evidenced |
| L9-AU | Production concerns (validation, metrics endpoint, load test, trusted-type review for model deserialisation) chosen without a brief | `app/main.py`; `src/load_test.py`; `src/train.py` | evidenced |
| T-MLOPS / T-ML / T-PY | MLflow tracking and registry, FastAPI, load test, 11 tests | `src/`; `app/`; `tests/` | evidenced |

### 10.3 Project 18 additions (18-dbt-airflow-modern-data-stack)

REAL NYC TLC yellow-taxi trips, January–February 2024 (5,972,150 rows).

| ID | How project 18 adds evidence | Evidence file(s) | Status |
| --- | --- | --- | --- |
| L9-K | Layered dbt modelling, incremental strategy, data tests, DAG orchestration with a quality gate | `projects/18-dbt-airflow-modern-data-stack/nyc_taxi/`; `airflow/dags/nyc_taxi_monthly.py` | evidenced |
| L9-CE | Every rejected row accounted for by reason; limits of DuckDB vs a cloud warehouse stated | `README.md` § Critical evaluation; `reports/metrics.json` | evidenced |
| L9-RI | Idempotency and full-refresh equivalence by MD5; a planted bad row proves the gate stops the run | `scripts/verify.py`; `reports/metrics.json` | evidenced |
| T-DE / T-SQL / T-PY | dbt SQL models and 23 tests, Airflow DAG, 5 `pytest` tests | `nyc_taxi/`; `airflow/`; `tests/` | evidenced |

### 10.4 Project 19 additions (19-realtime-streaming-pipeline)

REAL TLC trips replayed through Kafka, with injected delays, late events and duplicates.

| ID | How project 19 adds evidence | Evidence file(s) | Status |
| --- | --- | --- | --- |
| L9-K | Event time vs processing time, watermarks, late-data side outputs, delivery semantics and transactional sinks | `projects/19-realtime-streaming-pipeline/src/windowing.py`; `src/processor.py`; `README.md` § Approach | evidenced |
| L9-CE | Naive design measured (+33.6% trips after one crash); partition skew in my own design found and reported; exactly-once results distinguished from Kafka transactions | `README.md` § Results and § Critical evaluation | evidenced |
| L9-RI | Independent vectorised batch truth; window-by-window comparison after a simulated crash; streaming-vs-batch equivalence test | `src/truth.py`; `tests/test_windowing.py`; `reports/metrics.json` | evidenced |
| L9-AU | Kafka installed and run without Docker or a managed service | `scripts/start_kafka.sh`; `README.md` § How to run | evidenced |
| T-STREAM / T-DE / T-PY | Producer, processor, sink, live dashboard, 6 tests | `src/`; `app.py`; `tests/` | evidenced (single broker) |

### 10.5 Project 20 additions (20-pcb-defect-vision)

REAL DeepPCB board images (MIT licence) with artificially added defects; 4,640 test patches.

| ID | How project 20 adds evidence | Evidence file(s) | Status |
| --- | --- | --- | --- |
| L9-K | Transfer learning, class-weighted training, golden-template inspection, PatchCore-style anomaly detection, Grad-CAM | `projects/20-pcb-defect-vision/src/`; `README.md` § Approach | evidenced |
| L9-CE | Missed defects and false alarms reported per method; template baseline made fair (setting chosen on train, not test); synthetic defects, bare boards and patch-level scoring stated | `README.md` § Results and § Critical evaluation; `reports/metrics.json` | evidenced |
| L9-RI | Official test split scored once; thresholds fixed on validation; seen vs unseen image groups compared | `src/evaluate.py`; `reports/metrics.json` | evidenced |
| L9-ET | Licence and citation recorded; no employer images used | `data/patch_metadata.json`; `README.md` § Data status | evidenced |
| T-CV / T-ML / T-PY | Two CNNs, anomaly detector, board scan app, 9 tests | `src/`; `app.py`; `tests/` | evidenced |

### 10.6 Project 21 additions (21-predictive-maintenance-cmapss)

Public SIMULATED NASA C-MAPSS turbofan data (FD001 headline, FD003 check).

| ID | How project 21 adds evidence | Evidence file(s) | Status |
| --- | --- | --- | --- |
| L9-K | RUL with a capped target, unsupervised health indices, PHM08 scoring, cost-based maintenance policies | `projects/21-predictive-maintenance-cmapss/src/`; `README.md` § Approach | evidenced |
| L9-CE | Failed anomaly scores reported (PCA residual AUROC 0.54); FD003 reverses the best policy; small test set and assumed costs stated; threshold close to the cost cliff discussed | `README.md` § Results and § Critical evaluation; `reports/metrics.json` | evidenced |
| L9-RI | Engine-level split; causality test on features; parameters chosen on validation; failure-cost sensitivity; 5 repeated splits | `src/run.py`; `tests/test_pm.py`; `reports/metrics.json` | evidenced |
| L9-CO | Plain-language cost table for a plant manager; engine replay app | `CASE_STUDY.md`; `app.py` | evidenced |
| T-PM / T-ML / T-PY | Feature, anomaly, RUL and alerting modules; 10 tests | `src/`; `tests/` | evidenced |

### 10.7 Summary for projects 16–21

| Technical outcome | Projects 16–21 evidence |
| --- | --- |
| T-LLM | Tool-using agent, LoRA fine-tune, held-out phrasing and entity evaluation (16) |
| T-MLOPS | MLflow registry, FastAPI serving, drift monitoring, retraining policy (17) |
| T-DE / T-SQL | dbt + Airflow (18); Kafka pipeline with a transactional sink (19); read-only SQL tool (16) |
| T-STREAM | Event-time windows, watermarks, exactly-once results checked against a batch truth (19) |
| T-CV | CNN classifiers, template baseline, anomaly detection, Grad-CAM (20) |
| T-PM | Health scores, RUL, cost-based alert policies (21) |
| T-EV | Baselines in every project; independent truth (19); validation-only thresholds (20, 21); repeated splits (21) |
| T-PY | Six projects, pinned requirements, 51 `pytest` tests in total |
| T-DC | README with critical evaluation, `CASE_STUDY.md`, `CODE_WALKTHROUGH.md` for each |
| T-ET | Licence attribution and no personal data in public datasets (17–21); synthetic company data (16); no employer data (20, 21) |

Last updated: 18 November 2026 (Europe/Dublin).
