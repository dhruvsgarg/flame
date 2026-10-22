# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Stall streak leaks instead of resetting: one noisy eval must not zero it."""

from examples.fwdllm.expts.saturation_stop import SaturationDetector


def _det(**kw):
    # window 1, horizon 1: g = (acc_t - acc_{t-1}) / acc_t, so each eval is its own stall test
    return SaturationDetector(0, window=1, slope_horizon=1, stall_frac=0.003, decay=False, **kw)


def _feed(det, accs):
    for c, a in enumerate(accs, 1):
        if det.update(c, a):
            return c
    return None


def _plateau_with_spikes(n=200, every=10):
    # flat at 0.80; every `every`-th eval one noisy +0.01 uptick (g > frac, then back)
    return [0.81 if i % every == every - 1 else 0.80 for i in range(n)]


def test_reset_never_fires_on_noisy_plateau():
    assert _feed(_det(stall_leak=None), _plateau_with_spikes()) is None


def test_leak_fires_on_noisy_plateau():
    fired = _feed(_det(), _plateau_with_spikes())
    assert fired is not None and fired < 40


def test_leak_matches_reset_on_clean_plateau():
    flat = [0.80] * 100
    assert _feed(_det(), flat) == _feed(_det(stall_leak=None), flat) == 21


def test_leak_never_fires_while_climbing_with_noisy_dips():
    # +1% on two evals, -0.2% on the third: a slow climb with one stall-looking eval in three
    accs, a = [], 0.50
    for i in range(300):
        a *= 0.998 if i % 3 == 2 else 1.01
        accs.append(a)
    assert _feed(_det(), accs) is None


def test_leak_never_goes_negative():
    d = _det()
    _feed(d, [0.5 + 0.01 * i for i in range(50)])
    assert d._stalls == 0
