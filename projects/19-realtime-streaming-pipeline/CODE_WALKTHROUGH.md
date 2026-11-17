# Code walkthrough

Read in this order. Paths are relative to this project folder.

1. **`src/config.py`**: broker address, topic (`STREAM_TOPIC`), sink path
   (`STREAM_SINK`) and 4 partitions. The env vars let the paced latency
   run use its own topic and sink.

2. **`src/events.py`**: reads the TLC parquet with DuckDB, joins the zone
   lookup from project 18's seeds to get the borough, and keeps 15–16
   January 2024. It then adds a seeded arrival time (event time plus an
   exponential delay), delays 1.5% of events by 15–60 minutes, and re-sends
   1.0% a few seconds later. It sorts by arrival time, so the stream is
   out of order, and saves `data/events.parquet`. If project 18's download
   is missing, `source_file()` fetches the public file once.

3. **`src/windowing.py`**: the heart of the project, in pure Python so the
   tests can run it without Kafka.
   - `PartitionState` holds the watermark, open windows, seen event ids
     and the "closed up to" time, and serialises to JSON for checkpoints.
   - `process()` drops a seen `event_id` as a duplicate, sends an event
     whose window is already closed to the late output, otherwise adds it
     to its (borough, window) aggregate, then advances the watermark
     (max event time − 10 min).
   - `close_windows()` emits every window whose end is at or before the
     watermark, and prunes de-dup ids older than 2 hours of event time.
   - `flush()` closes everything at end of stream.

4. **`src/producer.py`**: an idempotent `confluent-kafka` producer
   (`acks=all`, lz4) keyed by borough. `--rate` paces sending, with
   `send_ts` in each message for latency. `--recreate` deletes and
   recreates the topic.

5. **`src/processor.py`**
   - **Exactly-once mode:** auto-commit is off. `on_assign` seeks each
     partition to the `next_offset` stored in the sink's `checkpoints`
     table and restores its state. Every 5,000 messages or 1 second,
     `checkpoint()` writes window results (upsert), late events, the state
     JSON and the next offsets in one `BEGIN IMMEDIATE … COMMIT`. Kafka's
     own committed offsets are never used.
   - **Naive mode:** auto-commit every second, de-dup disabled, state only
     in memory, windows appended to `naive_window_results`.
   - `--crash-after N` calls `os._exit(3)` after N messages: no clean-up,
     like a killed process.

6. **`src/truth.py`**: reads the whole topic back by partition and offset,
   then recomputes the expected windows with pandas: first occurrence per
   `event_id`, a running maximum event time per partition, and a vectorised
   late flag. It shares no code with `windowing.py`. `compare()` sums the
   sink's rows per window (so naive duplicates show up) and reports
   matches, trip and revenue totals, and whether the late-event sets are
   identical. One small difference: the batch de-dup has no TTL. That gives
   the same answer here, because no retry arrives 2 hours late.

7. **`src/run_experiment.py`**: the recorded run (see the README) that
   writes `reports/metrics.json`. `--only-paced` re-runs only the latency
   test.

8. **`src/charts.py`** and **`app.py`**: the two report charts and the
   Streamlit dashboard. It opens the sink read-only, shows windows per
   borough, late events and throughput, and refreshes every 2 s
   (`DASH_NO_REFRESH=1` turns that off for the test).

9. **`tests/`**: windowing rules, checkpoint round-trip equality, the
   streaming-vs-batch equivalence on random events, and a dashboard render
   test.

10. **`scripts/start_kafka.sh`**: formats and starts a single-node KRaft
    broker.
