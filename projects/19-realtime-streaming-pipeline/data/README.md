# data/

Generated locally, **not committed** (see `.gitignore`):

- `events.parquet`: the event stream built by `src/events.py` (about 4 MB).
- `sink.db`, `sink_paced.db`: the stream processor's SQLite sinks.

The source is the public NYC TLC Yellow Taxi file for January 2024. It is
re-used from project 18's `downloads/` folder if present; otherwise
`src/events.py` downloads it once into `downloads/` (also not committed).
