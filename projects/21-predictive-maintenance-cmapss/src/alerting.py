"""Turn a health signal into a maintenance decision, with the alert threshold chosen by COST.

Cost model (assumptions, stated in the README and varied in a sensitivity table):
  an alert at cycle t books maintenance LEAD_TIME cycles later;
  if the engine fails before then (true RUL at alert < LEAD_TIME) -> unplanned failure, cost C_FAIL;
  otherwise planned maintenance, cost C_PLAN + C_WASTE per cycle of useful life thrown away;
  no alert at all -> the engine runs to failure, cost C_FAIL.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Costs:
    c_plan: float = 1.0
    c_fail: float = 10.0
    c_waste: float = 1.0 / 200   # one planned job's worth per ~200 cycles of life lost (FD001 mean life ~206)
    lead_time: int = 10


def unit_cost(rul_at_alert: float | None, c: Costs) -> tuple[float, bool, float]:
    """-> (cost, failed, wasted cycles)."""
    if rul_at_alert is None or rul_at_alert < c.lead_time:
        return c.c_fail, True, 0.0
    wasted = float(rul_at_alert - c.lead_time)
    return c.c_plan + c.c_waste * wasted, False, wasted


def evaluate(alert_rul: list, c: Costs) -> dict:
    """alert_rul: true RUL at the alert cycle for each engine (None = never alerted)."""
    res = [unit_cost(r, c) for r in alert_rul]
    cost = np.array([r[0] for r in res]); failed = np.array([r[1] for r in res])
    wasted = np.array([r[2] for r in res])
    return {"engines": len(res), "mean_cost": round(float(cost.mean()), 4), "failures": int(failed.sum()),
            "mean_wasted_cycles": round(float(wasted[~failed].mean()), 1) if (~failed).any() else None}


def alert_from_threshold(signal: np.ndarray, rul: np.ndarray, threshold: float, below: bool, k: int = 1):
    """True RUL at the first cycle where the signal crosses the threshold for k consecutive cycles."""
    hit = signal <= threshold if below else signal >= threshold
    run = 0
    for i, h in enumerate(hit):
        run = run + 1 if h else 0
        if run >= k:
            return float(rul[i])
    return None


def choose_threshold(units: dict, grid, below: bool, c: Costs, k: int = 1):
    """units: {unit: (signal, true_rul)}. Pick the grid value with the lowest mean cost."""
    best = None
    curve = []
    for t in grid:
        m = evaluate([alert_from_threshold(s, r, t, below, k) for s, r in units.values()], c)["mean_cost"]
        curve.append((float(t), m))
        if best is None or m < best[1]:
            best = (float(t), m)
    return best[0], curve


def fixed_interval_alerts(lives: dict, T: int, c: Costs):
    """Time-based schedule: alert at cycle T - lead_time (maintenance at T) whatever the engine's state."""
    out = []
    for life in lives.values():
        alert_cycle = T - c.lead_time
        out.append(float(life - alert_cycle) if alert_cycle <= life else None)
    return out
