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

Last updated: 14 November 2026 (Europe/Dublin).
