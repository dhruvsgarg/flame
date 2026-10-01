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


class _Ch:
    def ends(self, state):
        return []

    def has(self, e):
        return False


def _idle_agg(**hp):
    from flame.sim.virtual_clock import SimReorderBuffer
    from sortedcontainers import SortedDict
    a = _agg(**hp)
    a.config.hyperparameters.max_experiment_runtime_s = 1800
    a.cm = SimpleNamespace(get_by_tag=lambda tag: _Ch())
    a._sim_buffer = SimReorderBuffer()
    a.trainer_event_dict = {"t1": SortedDict({0.0: "UN_AVL", 900.0: "AVL_TRAIN"})}
    a._vclock.advance(600.0)
    a.pending_withheld = {"t2": 600.0}
    a._sim_withheld_payload = {"t2": (150.0, ({}, ("t2", None)))}
    a._sim_withheld_delivering = {}
    a.drained = []
    a._sim_recv_min = lambda ch, ends: (a.drained.append(len(a._sim_buffer)), (None, ("", None)))[1]
    return a


def test_idle_sim_commits_due_delivery_before_starving():
    """FX-N54: no recv end + a due withheld delivery -> drain it; don't jump the clock to the next AVL."""
    a = _idle_agg()
    a._sim_reinject_when_idle = True
    a._aggregate_weights("t")
    assert a.drained == [1] and a._vclock.now == 600.0


def test_idle_reinject_off_starves_past_due_delivery():
    a = _idle_agg()
    a._sim_reinject_when_idle = False
    a._aggregate_weights("t")
    assert a.drained == [] and a._vclock.now == 900.0


class _ChHeld(_Ch):
    def ends(self, state):
        return ["held"]

    def has(self, e):
        return True

    def get_end_property(self, e, k):
        return 550.0  # dispatched at vclock 550


def test_held_pick_is_not_a_recv_end_and_wakes_at_its_timeout():
    """FX-N56: a held pick's update is in the delivery ledger; sim wakes at its dispatch+90s, not spinning on recv."""
    a = _idle_agg()
    a.pending_withheld, a._sim_withheld_payload = {"held": 900.0}, {}
    a._withheld_slot_held_set = {"held"}
    a.cm = SimpleNamespace(get_by_tag=lambda tag: _ChHeld())
    a._aggregate_weights("t")
    assert a.drained == [] and abs(a._vclock.now - 640.0) < 1e-3
