# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""FX-N97: AVL_EVAL is eval-only. A train result delivers only in AVL_TRAIN; an eval result in AVL_EVAL or AVL_TRAIN."""

import math

import pytest
from sortedcontainers import SortedDict

from flame.availability.client_availability import ClientAvailability
from flame.availability.trace import deliverable, next_avail_after
from flame.config import TrainerAvailState
from flame.mode.message import MessageType
from flame.sim.virtual_clock import SimReorderBuffer

# TRAIN [0,100) -> EVAL [100,200) -> UN_AVL [200,300) -> TRAIN [300,..)
_TRACE = SortedDict({0.0: "AVL_TRAIN", 100.0: "AVL_EVAL", 200.0: "UN_AVL", 300.0: "AVL_TRAIN"})


class _Host(ClientAvailability):
    def __init__(self, now=0.0):
        self.trainer_event_dict = {"t1": _TRACE}
        self.pending_withheld = {}
        self._task_ledger = {}
        self._now = now

    def _avail_now(self):
        return self._now


def test_deliverable_states():
    assert deliverable(TrainerAvailState.AVL_TRAIN, "train") and deliverable("AVL_TRAIN", "eval")
    assert not deliverable(TrainerAvailState.AVL_EVAL, "train") and deliverable("AVL_EVAL", "eval")
    assert not deliverable(TrainerAvailState.UN_AVL, "train") and not deliverable("UN_AVL", "eval")


def test_next_avail_after_is_task_aware():
    assert next_avail_after(SortedDict({0.0: "UN_AVL", 50.0: "AVL_EVAL", 90.0: "AVL_TRAIN"}), 10.0) == 50.0
    assert next_avail_after(SortedDict({0.0: "UN_AVL", 50.0: "AVL_EVAL", 90.0: "AVL_TRAIN"}), 10.0, "train") == 90.0
    assert next_avail_after(SortedDict({0.0: "UN_AVL", 50.0: "AVL_EVAL"}), 10.0, "train") == math.inf


@pytest.mark.parametrize("sct, task, want", [
    (50.0, "train", 50.0),     # AVL_TRAIN: immediate
    (150.0, "train", 300.0),   # AVL_EVAL: a train result waits for AVL_TRAIN
    (150.0, "eval", 150.0),    # AVL_EVAL: an eval result delivers
    (250.0, "eval", 300.0),    # UN_AVL: both wait
    (250.0, "train", 300.0),
])
def test_compute_delivery_ts(sct, task, want):
    assert _Host().compute_delivery_ts("t1", sct, task) == want


def test_compute_delivery_ts_defaults_to_train():
    assert _Host().compute_delivery_ts("t1", 150.0) == 300.0


def test_task_of_update():
    assert ClientAvailability._task_of_update(({MessageType.WEIGHTS: 1, MessageType.STAT_UTILITY: 2}, ("e", 1))) == "train"
    assert ClientAvailability._task_of_update(({MessageType.WEIGHTS_BYTES: b"", MessageType.STAT_UTILITY: 2}, None)) == "train"
    assert ClientAvailability._task_of_update(({MessageType.STAT_UTILITY: 2}, ("e", 1))) == "eval"
    assert ClientAvailability._task_of_update(None) == "train"


def test_inflight_task_is_latest_dispatch():
    h = _Host()
    assert h._inflight_task("t1") == "train"
    h._task_ledger = {("t1", "train"): [3, 10.0, 0], ("t1", "eval"): [3, 20.0, 0], ("t2", "train"): [3, 30.0, 0]}
    assert h._inflight_task("t1") == "eval"


def test_sim_send_gated_by_task():
    h = _Host()
    h._sim_buffer = SimReorderBuffer()
    h._sim_buffer.add("t1", 150.0, ({MessageType.WEIGHTS: 1}, None))
    assert h._sim_send_gated("t1")  # train result finished in AVL_EVAL
    h._sim_buffer.add("t1", 150.0, ({MessageType.STAT_UTILITY: 1}, None))
    assert not h._sim_send_gated("t1")  # eval result finished in AVL_EVAL


class _Sel:
    requester = "agg"

    def __init__(self):
        self.all_selected = {"t1": 1}
        self.selected_ends = {"agg": {"t1"}}


class _Chan:
    def __init__(self):
        self._selector, self._ends = _Sel(), {}

    def has(self, end):
        return False


@pytest.mark.parametrize("task, evicted", [("train", True), ("eval", False)])
def test_aware_eviction_frees_train_in_avl_eval(task, evicted):
    h = _Host(now=150.0)
    h.proactive_inflight_evict = True
    h._task_ledger = {("t1", task): [1, 100.0, 0]}
    ch = _Chan()
    h._sim_evict_unavail_inflight(ch)
    assert h.pending_withheld == ({"t1": 300.0} if evicted else {})
    assert ch._selector.selected_ends["agg"] == (set() if evicted else {"t1"})
