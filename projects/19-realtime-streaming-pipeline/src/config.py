from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP = "localhost:9092"
import os

TOPIC = os.environ.get("STREAM_TOPIC", "taxi-trips")
PARTITIONS = 4
SINK = Path(os.environ.get("STREAM_SINK", ROOT / "data" / "sink.db"))
PRODUCED = ROOT / "data" / "produced.parquet"
REPORTS = ROOT / "reports"
