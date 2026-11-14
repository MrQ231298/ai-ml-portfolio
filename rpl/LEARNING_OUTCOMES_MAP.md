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

**Honesty note.** Project 1 is a classical TF-IDF baseline, not work at
the forefront of machine learning. Level 9 alignment here is mainly in
*how the work is evaluated, documented, and ethically framed*, not in
algorithmic novelty. That is deliberate and is called out in the
project's critical evaluation.

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
4. A second domain project (landlord / rental or commercial insurance
   lead generation) should add rows or a new project column, not
   overwrite project 1.

Last updated: 14 November 2026 (Europe/Dublin).
