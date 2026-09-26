# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""asyncfl inner loop exits on a mid-cycle stop, and the sim wall ceiling fires without a round advance."""

import time
from types import SimpleNamespace

from flame.mode.horizontal.asyncfl.top_aggregator import TopAggregator
from flame.sim import VirtualClock


class _Agg(TopAggregator):
    def check_and_sleep(self): pass
    def evaluate(self): pass
    def initialize(self): pass
    def load_data(self): pass
    def train(self): pass


def _agg(simulated=True, wall_elapsed=0.0, **hp):
    a = _Agg.__new__(_Agg)
    a.simulated = simulated
    a._vclock = VirtualClock()
    a._round = 7
    a._agg_goal, a._agg_goal_cnt = 10, 3
    a._work_done = False
    a.agg_start_time_ts = time.time() - wall_elapsed
    a.config = SimpleNamespace(hyperparameters=SimpleNamespace(
        max_experiment_runtime_s=450, sim_wall_ceiling_s=None, max_wall_runtime_s=None, **hp))
    return a


def test_mid_cycle_stop_exits_inner_loop():
    a = _agg()
    assert not a._async_inner_loop_done()
    a._work_done = True  # e.g. [SIM_STARVATION] stopping run
    assert a._async_inner_loop_done()


def test_sim_wall_ceiling_fires_without_round_advance():
    a = _agg(wall_elapsed=500.0)
    assert a._async_inner_loop_done() and a._work_done


def test_real_mode_ignores_sim_ceiling():
    a = _agg(simulated=False, wall_elapsed=500.0)
    assert not a._async_inner_loop_done() and not a._work_done
