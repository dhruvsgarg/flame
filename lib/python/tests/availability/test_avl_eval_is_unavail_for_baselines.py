# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""FX-D165: for every baseline but Felix (no eval tasks), a 3-state trace decides exactly as the same trace with AVL_EVAL
relabeled UN_AVL: train eligibility, delivery time, send-gating and the sim's starvation wake-up."""

import random
import types

import pytest
from sortedcontainers import SortedDict

from flame.availability.client_availability import ClientAvailability
from flame.mode.message import MessageType
from flame.sim.virtual_clock import SimReorderBuffer

STATES = ("AVL_TRAIN", "AVL_EVAL", "UN_AVL")


def _traces(seed, n=12, horizon=3000.0):
    rng, out = random.Random(seed), {}
    for i in range(n):
        t, s, ev = 0.0, rng.choice(STATES), SortedDict()
        while t < horizon:
            ev[t] = s
            t += rng.uniform(5, 120)
            s = rng.choice([x for x in STATES if x != s])
        out[f"t{i}"] = ev
    return out


def _collapse(tr):
    """AVL_EVAL -> UN_AVL, merging the repeats this creates."""
    out, last = SortedDict(), None
    for t, s in tr.items():
        s = "UN_AVL" if s == "AVL_EVAL" else s
        if s != last:
            out[t] = s
        last = s
    return out


class _Host(ClientAvailability):
    def __init__(self, traces, now, eval_goal=0.0):
        self.trainer_event_dict = traces
        self._trace_has_avl_eval = any("AVL_EVAL" in tr.values() for tr in traces.values())
        self.pending_withheld, self._task_ledger, self._now = {}, {}, now
        self.avail_select_filter = True
        self.config = types.SimpleNamespace(selector=types.SimpleNamespace(kwargs={"evalGoalFactor": eval_goal}))

    def _avail_now(self):
        return self._now


@pytest.mark.parametrize("seed", range(5))
def test_non_felix_sees_avl_eval_as_unavail(seed):
    three = _traces(seed)
    two = {k: _collapse(v) for k, v in three.items()}
    for now in [x * 7.3 for x in range(400)]:
        a, b = _Host(three, now), _Host(two, now)
        assert sorted(a.get_curr_task_ineligible_trainers("train")) == sorted(b.get_curr_task_ineligible_trainers("train"))
        assert a._next_avail_vclock() == b._next_avail_vclock()
        for end in three:
            assert a.compute_delivery_ts(end, now, "train") == b.compute_delivery_ts(end, now, "train")
            for h in (a, b):
                h._sim_buffer = SimReorderBuffer()
                h._sim_buffer.add(end, now, ({MessageType.WEIGHTS: 1}, None))
            assert a._sim_send_gated(end) == b._sim_send_gated(end)


def test_felix_wakes_on_avl_eval():
    tr = {"t0": SortedDict({0.0: "UN_AVL", 50.0: "AVL_EVAL", 90.0: "AVL_TRAIN"})}
    assert _Host(tr, 10.0, eval_goal=1.0)._next_avail_vclock() == 50.0
    assert _Host(tr, 10.0)._next_avail_vclock() == 90.0
