"""
Attribution models over user journeys (touch sequences ending in a
purchase or not). Each function returns credited conversions per channel
for the converting journeys; revenue credit uses the same weights.

Rule-based: last click, first click, linear, time decay, position-based.
Data-driven: first-order Markov chain removal effect; Shapley value on
conversion rates of channel sets.
"""

from __future__ import annotations

from itertools import combinations
from math import factorial

import numpy as np
import pandas as pd


def journeys(tp: pd.DataFrame, conv: pd.DataFrame) -> pd.DataFrame:
    """One row per touched user: ordered channel list, touch times, converted flag, revenue."""
    t = tp.sort_values(["user_id", "ts"])
    j = t.groupby("user_id").agg(path=("channel", list), times=("ts", list))
    c = conv.set_index("user_id")
    j["converted"] = j.index.isin(c.index)
    j["revenue"] = c["revenue"].reindex(j.index).fillna(0.0).values
    j["conv_ts"] = c["ts"].reindex(j.index).values
    return j


def _rule(j: pd.DataFrame, weight_fn) -> pd.DataFrame:
    rows = []
    for path, times, rev, cts in zip(j.path, j.times, j.revenue, j.conv_ts):
        w = np.asarray(weight_fn(path, times, cts), dtype=float)
        w = w / w.sum()
        for ch, wi in zip(path, w):
            rows.append((ch, wi, wi * rev))
    df = pd.DataFrame(rows, columns=["channel", "conversions", "revenue"])
    return df.groupby("channel")[["conversions", "revenue"]].sum()


def last_click(j):
    return _rule(j[j.converted], lambda p, t, c: [0] * (len(p) - 1) + [1])


def first_click(j):
    return _rule(j[j.converted], lambda p, t, c: [1] + [0] * (len(p) - 1))


def linear(j):
    return _rule(j[j.converted], lambda p, t, c: [1] * len(p))


def time_decay(j, half_life_days: float = 7.0):
    def w(p, t, c):
        age = np.array([(c - ti) / pd.Timedelta(days=1) for ti in t])
        return 0.5 ** (age / half_life_days)
    return _rule(j[j.converted], w)


def position_based(j, ends: float = 0.4):
    def w(p, t, c):
        n = len(p)
        if n == 1:
            return [1.0]
        if n == 2:
            return [0.5, 0.5]
        mid = (1 - 2 * ends) / (n - 2)
        return [ends] + [mid] * (n - 2) + [ends]
    return _rule(j[j.converted], w)


def _collapse(path):
    out = [path[0]]
    for p in path[1:]:
        if p != out[-1]:
            out.append(p)
    return out


def markov(j: pd.DataFrame) -> pd.DataFrame:
    """First-order Markov chain; removal effect = drop in P(conversion) when a channel is removed."""
    channels = sorted({c for p in j.path for c in p})
    states = ["START"] + channels + ["CONV", "NULL"]
    idx = {s: i for i, s in enumerate(states)}
    T = np.zeros((len(states), len(states)))
    for path, conv in zip(j.path, j.converted):
        seq = ["START"] + _collapse(path) + ["CONV" if conv else "NULL"]
        for a, b in zip(seq[:-1], seq[1:]):
            T[idx[a], idx[b]] += 1
    T[idx["CONV"], idx["CONV"]] = T[idx["NULL"], idx["NULL"]] = 1
    P = T / T.sum(axis=1, keepdims=True)

    def p_conv(P):
        transient = [idx["START"]] + [idx[c] for c in channels]
        Q = P[np.ix_(transient, transient)]
        R = P[np.ix_(transient, [idx["CONV"]])]
        N = np.linalg.solve(np.eye(len(transient)) - Q, R)
        return float(N[0, 0])

    base = p_conv(P)
    removal = {}
    for c in channels:
        P2 = P.copy()
        k = idx[c]
        P2[:, idx["NULL"]] += P2[:, k]   # anything that would go to c is lost
        P2[:, k] = 0
        P2[k, :] = 0
        P2[k, idx["NULL"]] = 1
        removal[c] = max(0.0, 1 - p_conv(P2) / base)
    tot = sum(removal.values())
    conv_total = j.converted.sum()
    rev_total = j.revenue.sum()
    return pd.DataFrame({"conversions": {c: removal[c] / tot * conv_total for c in channels},
                         "revenue": {c: removal[c] / tot * rev_total for c in channels}})


def shapley(j: pd.DataFrame, base_rate: float) -> pd.DataFrame:
    """Shapley values on v(S) = conversion rate of journeys whose channel set is exactly S
    (v(empty) = conversion rate of untouched people). Each converting journey's credit is split by
    the Shapley values of the game restricted to its own channels (negative values clipped to 0)."""
    sets = j.path.map(lambda p: frozenset(p))
    rate = j.groupby(sets)["converted"].mean().to_dict()
    rate[frozenset()] = base_rate

    def v(S):
        return rate.get(frozenset(S), None)

    cache = {}

    def shap(S):
        S = frozenset(S)
        if S in cache:
            return cache[S]
        n = len(S)
        phi = {}
        for c in S:
            others = [x for x in S if x != c]
            total = 0.0
            for k in range(len(others) + 1):
                for sub in combinations(others, k):
                    a, b = v(set(sub) | {c}), v(sub)
                    if a is None or b is None:
                        continue  # unobserved coalition: skip that marginal contribution
                    total += factorial(k) * factorial(n - k - 1) / factorial(n) * (a - b)
            phi[c] = max(total, 0.0)
        cache[S] = phi
        return phi

    rows = []
    for S, rev in zip(sets[j.converted], j.revenue[j.converted]):
        phi = shap(S)
        tot = sum(phi.values())
        w = {c: (phi[c] / tot if tot > 0 else 1 / len(S)) for c in S}
        for c, wi in w.items():
            rows.append((c, wi, wi * rev))
    df = pd.DataFrame(rows, columns=["channel", "conversions", "revenue"])
    return df.groupby("channel")[["conversions", "revenue"]].sum()


MODELS = {"Last click": last_click, "First click": first_click, "Linear": linear,
          "Time decay (7-day half-life)": time_decay, "Position-based 40/20/40": position_based,
          "Markov removal effect": markov}
