"""Event-time tumbling windows with a watermark, late-event side output and
de-duplication. Pure Python, no Kafka: the same class runs inside the stream
processor and in the tests.

Rules (per Kafka partition, so results are deterministic for a given partition order)
  window           [start, start + WINDOW) by event time, keyed by borough
  watermark        max event time seen in the partition - OUT_OF_ORDERNESS
  window closes    when watermark >= window end; its result is emitted once
  late event       event whose window has already closed -> side output, not counted
  duplicate        event_id already seen (state kept for DEDUP_TTL of event time)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

WINDOW = timedelta(minutes=5)
OUT_OF_ORDERNESS = timedelta(minutes=10)
DEDUP_TTL = timedelta(hours=2)


def window_start(t: datetime) -> datetime:
    return t - timedelta(minutes=t.minute % 5, seconds=t.second, microseconds=t.microsecond)


@dataclass
class PartitionState:
    watermark: datetime | None = None
    max_event_time: datetime | None = None
    open: dict = field(default_factory=dict)      # (borough, window_start iso) -> [count, revenue]
    seen: dict = field(default_factory=dict)      # event_id -> event_time iso
    closed_up_to: datetime | None = None          # every window ending <= this is closed

    def to_json(self) -> dict:
        return {"watermark": self.watermark and self.watermark.isoformat(),
                "max_event_time": self.max_event_time and self.max_event_time.isoformat(),
                "open": [[b, w, c, r] for (b, w), (c, r) in self.open.items()],
                "seen": self.seen,
                "closed_up_to": self.closed_up_to and self.closed_up_to.isoformat()}

    @classmethod
    def from_json(cls, d: dict) -> "PartitionState":
        p = lambda s: datetime.fromisoformat(s) if s else None  # noqa: E731
        return cls(p(d["watermark"]), p(d["max_event_time"]), {(b, w): [c, r] for b, w, c, r in d["open"]},
                   dict(d["seen"]), p(d["closed_up_to"]))


def process(state: PartitionState, ev: dict) -> dict:
    """Apply one event. Returns {"status": "counted"|"duplicate"|"late", "closed": [window rows]}."""
    et = datetime.fromisoformat(ev["event_time"])
    if ev["event_id"] in state.seen:
        return {"status": "duplicate", "closed": []}
    state.seen[ev["event_id"]] = et.isoformat()
    ws = window_start(et)
    if state.closed_up_to is not None and ws + WINDOW <= state.closed_up_to:
        status = "late"
    else:
        key = (ev["borough"], ws.isoformat())
        agg = state.open.setdefault(key, [0, 0.0])
        agg[0] += 1
        agg[1] = round(agg[1] + ev["amount"], 2)
        status = "counted"
    if state.max_event_time is None or et > state.max_event_time:
        state.max_event_time = et
        state.watermark = et - OUT_OF_ORDERNESS
    closed = close_windows(state)
    return {"status": status, "closed": closed}


def close_windows(state: PartitionState) -> list[dict]:
    if state.watermark is None:
        return []
    out = []
    for (b, w), (c, r) in list(state.open.items()):
        if datetime.fromisoformat(w) + WINDOW <= state.watermark:
            out.append({"borough": b, "window_start": w, "trips": c, "revenue": r})
            del state.open[(b, w)]
    # everything ending at or before the watermark is closed from now on
    wm_floor = window_start(state.watermark)
    if state.closed_up_to is None or wm_floor > state.closed_up_to:
        state.closed_up_to = wm_floor
    cutoff = (state.watermark - DEDUP_TTL).isoformat()
    if len(state.seen) > 20000:
        state.seen = {k: v for k, v in state.seen.items() if v >= cutoff}
    return out


def flush(state: PartitionState) -> list[dict]:
    """End of stream: close every open window."""
    out = [{"borough": b, "window_start": w, "trips": c, "revenue": r} for (b, w), (c, r) in state.open.items()]
    state.open.clear()
    return out
