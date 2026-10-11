# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""FX-D15: the sim wall ceiling times the simulated run from the join barrier, not process startup
(a 35s CPU warm-up + join made 60s smoke sims stop at vclock 6)."""

import time
from types import SimpleNamespace

from flame.mode.horizontal.syncfl.top_aggregator import TopAggregator


def _agg(started_ago_s, joined_ago_s=None):
    a = SimpleNamespace(simulated=True, _work_done=False, _round=1, _vclock=SimpleNamespace(now=5.0),
                        agg_start_time_ts=time.time() - started_ago_s,
                        config=SimpleNamespace(hyperparameters=SimpleNamespace(
                            max_experiment_runtime_s=60, sim_wall_ceiling_s=60, max_wall_runtime_s=None)))
    if joined_ago_s is not None:
        a._run_start_wall_ts = time.time() - joined_ago_s
    return a


def test_startup_before_the_join_barrier_is_not_charged():
    a = _agg(started_ago_s=90, joined_ago_s=30)
    TopAggregator._check_sim_wall_ceiling(a)
    assert a._work_done is False


def test_ceiling_fires_on_run_time_after_the_barrier():
    a = _agg(started_ago_s=200, joined_ago_s=70)
    TopAggregator._check_sim_wall_ceiling(a)
    assert a._work_done is True


def test_before_the_barrier_falls_back_to_aggregator_start():
    a = _agg(started_ago_s=70)
    TopAggregator._check_sim_wall_ceiling(a)
    assert a._work_done is True
