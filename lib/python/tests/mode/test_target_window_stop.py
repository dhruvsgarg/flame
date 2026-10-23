# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""FX-D229: `targetWindow` stops on the mean of the last w evals; unset keeps the consecutive rule."""
from types import SimpleNamespace

from flame.mode.horizontal.syncfl.top_aggregator import TopAggregator


def _agg(window, stable=20):
    return SimpleNamespace(_target_accuracy=0.7, _target_window=window, _target_evals=[], _stable_evals_above_target=stable,
                           _consecutive_above_target=0, _work_done=False)


def test_window_mean_stops_through_jitter():
    a = _agg(5)
    for acc in (0.72, 0.69, 0.71, 0.68):
        TopAggregator._check_target_stop(a, 1, acc)
    assert not a._work_done  # 4 evals: window not full
    TopAggregator._check_target_stop(a, 2, 0.71)
    assert a._work_done  # mean 0.702 >= 0.7 despite two dips


def test_window_mean_below_target_keeps_running():
    a = _agg(5)
    for acc in (0.75, 0.66, 0.66, 0.66, 0.66):
        TopAggregator._check_target_stop(a, 1, acc)
    assert not a._work_done


def test_unset_window_keeps_consecutive_rule():
    a = _agg(None, stable=2)
    for acc in (0.71, 0.69, 0.71):
        TopAggregator._check_target_stop(a, 1, acc)
    assert not a._work_done
    TopAggregator._check_target_stop(a, 1, 0.72)
    assert a._work_done
