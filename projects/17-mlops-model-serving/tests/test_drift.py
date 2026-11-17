"""Drift statistics behave as expected on known distributions."""

import numpy as np

from drift import PSI_ALERT, ks, level, psi


def test_psi_is_near_zero_for_same_distribution():
    rng = np.random.default_rng(0)
    assert psi(rng.normal(size=5000), rng.normal(size=5000)) < 0.02


def test_psi_flags_a_shifted_distribution():
    rng = np.random.default_rng(1)
    v = psi(rng.normal(size=5000), rng.normal(1.0, 1, size=5000))
    assert v > PSI_ALERT and level(v) == "alert"


def test_categorical_psi_handles_new_categories():
    ref = np.array([1] * 90 + [2] * 10)
    cur = np.array([1] * 50 + [2] * 10 + [3] * 40)
    assert psi(ref, cur, categorical=True) > PSI_ALERT


def test_ks_detects_shift_and_not_noise():
    rng = np.random.default_rng(2)
    assert ks(rng.normal(size=2000), rng.normal(size=2000))[1] > 0.01
    assert ks(rng.normal(size=2000), rng.normal(0.5, 1, size=2000))[1] < 1e-6
