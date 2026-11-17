# Code walkthrough

Read in this order. Everything runs from the project folder.

1. **`src/fetch_and_features.py`**
   - `load_raw()` downloads the UCI zip into `downloads/`, checks the
     SHA-256 and reads the Excel sheet.
   - `clean()` removes exact duplicates, rows with no customer, non-product
     codes (postage, fees, manual adjustments, bad debt) and non-positive
     prices. It returns row counts at each step, which go into
     `data/source_metadata.json`.
   - `features()` builds one row per customer from a time window: invoices
     become baskets; recency and tenure are measured to the window end;
     cancellations reduce `monetary` and form `cancel_rate`.
   - `main()` builds features from the calibration window only, then adds
     the hold-out outcomes (`future_90d_purchased`, `future_90d_revenue`)
     as separate columns that the clustering never uses.

2. **`src/seg_lib.py`**: the feature list and the preprocessing pipeline
   (`log_and_clip` then `StandardScaler`). It is a module-level function
   so the saved model can be unpickled by the app and the tests.

3. **`src/segment.py`**
   - `bootstrap_ari()` fits k-means on two overlapping 80% subsamples
     and compares their labels on the shared customers.
   - `main()` runs the k grid (k-means and GMM diagnostics), applies the
     fixed selection rule, refits with 50 starts, checks seed stability,
     builds profiles, names segments, and runs the hold-out comparison
     with the RFM baseline (`rfm_scores()`).
   - The AUC score for a group is its **calibration** purchase frequency,
     so no hold-out information leaks into the score.
   - Writes `reports/*.csv|png`, `reports/metrics.json` and
     `models/segmenter.joblib`.

4. **`app.py`**: Streamlit, reading only the files in `reports/` and
   `models/`. The last tab scores a typed-in customer with the saved
   model.

5. **`tests/test_segmentation.py`** (6 tests)
   - Cleaning and features on a 9-row hand-built fixture whose answers
     were worked out by hand (including a duplicate, a cancellation,
     postage, a missing customer and a zero price).
   - Preprocessing, the RFM ordering and the naming rules.
   - The saved model reproduces every stored assignment, and the stored k
     obeys the selection rule.

Design choices worth questioning: the 3–8 range and the 0.75 stability
bar; net-of-cancellation spend; excluding customers with non-positive
net spend; `log1p` scaling (it makes k-means less dominated by the
largest wholesale accounts).
