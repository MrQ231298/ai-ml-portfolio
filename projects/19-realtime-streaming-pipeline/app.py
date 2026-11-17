"""Live dashboard over the stream processor's SQLite sink. Refreshes every 2 s.

Run the processor (src/processor.py) and the producer (src/producer.py --rate 2000)
in two terminals, then: streamlit run app.py
"""

import os
import sqlite3
import sys
import time
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from config import SINK  # noqa: E402

st.set_page_config(page_title="Streaming taxi windows", layout="wide")
st.title("Real-time taxi demand: 5-minute event-time windows from Kafka")
st.caption("REAL NYC TLC trips (Jan 2024) replayed as an out-of-order stream with injected late events and duplicates.")


def q(sql):
    con = sqlite3.connect(f"file:{SINK}?mode=ro", uri=True)
    try:
        return pd.read_sql(sql, con)
    finally:
        con.close()


if not SINK.exists():
    st.info(f"No sink yet at {SINK}. Start the processor.")
    st.stop()
ck = q("select partition, next_offset, messages, duplicates, late, updated_at from checkpoints order by partition")
w = q("select borough, window_start, trips, revenue, emitted_at from window_results")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Messages processed", f"{int(ck.messages.sum()):,}")
c2.metric("Duplicates dropped", f"{int(ck.duplicates.sum()):,}")
c3.metric("Late events (side output)", f"{int(ck.late.sum()):,}")
c4.metric("Closed windows", f"{len(w):,}")
if len(w):
    w["window_start"] = pd.to_datetime(w.window_start)
    top = w[w.borough.isin(["Manhattan", "Queens", "Brooklyn"])]
    st.altair_chart(alt.Chart(top).mark_line().encode(x="window_start:T", y="trips:Q", color="borough:N")
                    .properties(height=320, title="Trips per 5-minute window (closed windows only)"), use_container_width=True)
    tp = q("select ts, sum(messages) messages from throughput group by ts order by ts")
    if len(tp) > 1:
        tp["msgs_per_s"] = tp.messages.diff() / tp.ts.diff()
        tp["time"] = pd.to_datetime(tp.ts, unit="s")
        st.altair_chart(alt.Chart(tp.dropna()).mark_area(opacity=.5).encode(x="time:T", y="msgs_per_s:Q")
                        .properties(height=180, title="Processor throughput (msg/s, from checkpoints)"), use_container_width=True)
st.subheader("Checkpoints (offsets committed together with results)")
st.dataframe(ck, hide_index=True)
st.subheader("Latest late events")
st.dataframe(q("select * from late_events order by kafka_offset desc limit 20"), hide_index=True)
if not os.environ.get("DASH_NO_REFRESH"):
    time.sleep(2)
    st.rerun()
