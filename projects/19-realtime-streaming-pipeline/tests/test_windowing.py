"""Windowing, watermark, late events, de-dup and checkpoint restore, without Kafka."""

import random
from datetime import datetime, timedelta

import pandas as pd

from truth import expected
from windowing import PartitionState, flush, process, window_start

T0 = datetime(2024, 1, 15, 8, 0)


def ev(i, minutes, borough="Manhattan", amount=10.0):
    return {"event_id": f"e{i}", "event_time": (T0 + timedelta(minutes=minutes)).isoformat(),
            "borough": borough, "amount": amount}


def test_window_start_floors_to_five_minutes():
    assert window_start(datetime(2024, 1, 1, 10, 7, 31)) == datetime(2024, 1, 1, 10, 5)


def test_window_closes_only_after_watermark_passes_its_end():
    s = PartitionState()
    assert process(s, ev(1, 1))["closed"] == []
    assert process(s, ev(2, 14))["closed"] == []          # watermark 08:04 < 08:05
    out = process(s, ev(3, 15.5))["closed"]               # watermark 08:05:30 >= 08:05
    assert out == [{"borough": "Manhattan", "window_start": T0.isoformat(), "trips": 1, "revenue": 10.0}]


def test_late_event_goes_to_side_output_and_duplicate_is_dropped():
    s = PartitionState()
    process(s, ev(1, 1)); process(s, ev(2, 30))
    assert process(s, ev(3, 2))["status"] == "late"       # 08:00 window closed when watermark reached 08:20
    assert process(s, ev(2, 30))["status"] == "duplicate"
    assert process(s, ev(4, 25))["status"] == "counted"   # out of order but within 10 min


def test_restore_from_checkpoint_gives_identical_results():
    rng = random.Random(0)
    events = [ev(i, i * 0.5 + rng.uniform(-8, 8), rng.choice(["Manhattan", "Queens"])) for i in range(400)]
    events += [dict(events[i]) for i in range(0, 400, 37)]  # duplicates
    rng.shuffle(events[350:])

    def run(evs, state=None):
        s, out = state or PartitionState(), []
        for e in evs:
            out += process(s, e)["closed"]
        return s, out

    _, straight = run(events)
    s_mid, first = run(events[:200])
    restored = PartitionState.from_json(__import__("json").loads(__import__("json").dumps(s_mid.to_json())))
    s_end, second = run(events[200:], restored)
    assert first + second == straight


def test_vectorised_reference_matches_stream_logic():
    rng = random.Random(1)
    evs = [ev(i, i * 0.2 + (rng.uniform(15, 40) if rng.random() < 0.05 else rng.uniform(0, 3)),
              rng.choice(["Manhattan", "Queens", "Bronx"]), round(rng.uniform(5, 60), 2)) for i in range(2000)]
    evs += [dict(evs[i]) for i in range(0, 2000, 53)]
    evs.sort(key=lambda e: e["event_time"] if rng.random() < 0.7 else "z")  # arbitrary arrival order
    s, rows, late = PartitionState(), [], set()
    for e in evs:
        r = process(s, e)
        rows += r["closed"]
        if r["status"] == "late":
            late.add(e["event_id"])
    rows += flush(s)
    df = pd.DataFrame([dict(e, partition=0, offset=i) for i, e in enumerate(evs)])
    exp, exp_late, n_dup = expected(df)
    got = pd.DataFrame(rows).sort_values(["borough", "window_start"]).reset_index(drop=True)
    exp = exp.sort_values(["borough", "window_start"]).reset_index(drop=True)
    assert exp_late == late and n_dup == 38
    assert (got.trips.values == exp.trips.values).all()
    assert abs(got.revenue - exp.revenue).max() < 0.005
