# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""`_sim_sync_recv_incremental` replaces the old one-shot
`_sync_sim_recv_first_k` call site in `sync_collect_and_accumulate_grads` so
sim can be called with `num_min_req=1` repeatedly, like real, without
stranding (discarding) whatever didn't make this call's cut.

The old `_sync_sim_recv_first_k` drained the WHOLE selected set into a local
buffer every call and dropped anything past `first_k` -- calling it with
`first_k=1` repeatedly would silently lose every candidate past the first.
`_sim_sync_recv_incremental` keeps a PERSISTENT, instance-level sct-ordered
buffer across calls instead: a candidate is added once and only ever popped,
never wholesale discarded.

These tests pin: never-strand, never-double-commit, sct-order preserved
across separate calls, plus backward-compat with the old single-call shape.
"""

from datetime import datetime, timedelta

from flame.mode.horizontal.syncfl.fwdllm_aggregator import TopAggregator
from flame.mode.message import MessageType
from flame.sim.virtual_clock import SimReorderBuffer, VirtualClock
from flame.selector.properties import (
    PROP_CLIENT_TASK_TRAIN_DURATION,
    PROP_SIM_SEND_TS,
)


class _FakeIncrementalChannel:
    """Like `_FakeBarrierChannel`, but distinguishes "selected/dispatched"
    (`has()`/`ends()`) from "message ready" (`recv_fifo` yields only for an
    end with a message already added) -- so a test can commit against a
    partial set of ready messages, then add more and drive a second call."""

    def __init__(self):
        self._selected = set()
        self._msgs = {}
        self._delivered = set()
        self._end_props = {}

    def select(self, end):
        self._selected.add(end)

    def add_msg(self, end, sct, dur=None):
        self.select(end)
        m = {MessageType.SIM_COMPLETION_TS: sct}
        if dur is not None:
            m[MessageType.SIM_CLIENT_TASK_TRAIN_DURATION_S] = dur
        self._msgs[end] = m

    def ends(self, state=None):
        return list(self._selected)

    def has(self, end):
        return end in self._selected

    def recv_fifo(self, end_ids, first_k=None, timeout=None):
        for e in end_ids:
            if e in self._delivered or e not in self._msgs:
                continue
            self._delivered.add(e)
            yield self._msgs[e], (e, datetime.now())

    def get_end_property(self, end, prop):
        return self._end_props.get((end, prop))

    def set_end_property(self, end, prop, value):
        self._end_props[(end, prop)] = value


class _FakeIncrementalAgg:
    """Binds the real incremental-collect method onto a minimal stand-in, same
    style as `test_fwdllm_sim_sync_barrier.py`'s `_FakeBarrierAgg`."""

    _sim_sync_recv_incremental = TopAggregator._sim_sync_recv_incremental
    _advance_sim_clock = TopAggregator._advance_sim_clock
    _sim_recv_timeout_s = TopAggregator._sim_recv_timeout_s
    _note_sim_known_delay = TopAggregator._note_sim_known_delay
    _sim_reinject_ready_withheld = TopAggregator._sim_reinject_ready_withheld
    _sim_withhold_if_unavail = TopAggregator._sim_withhold_if_unavail

    def __init__(self):
        self._round = 1
        self._vclock = VirtualClock()
        self._sim_buffer = SimReorderBuffer()
        self._sim_known_delay_s = {}


class TestIncrementalNeverStrandNeverDoubleCommit:
    def test_never_strands_a_candidate_past_num_min_req(self):
        """With num_min_req=1, candidates that don't win this call must
        survive for a later call, not be discarded."""
        agg = _FakeIncrementalAgg()
        ch = _FakeIncrementalChannel()
        ch.add_msg("A", sct=40.0)
        ch.add_msg("B", sct=10.0)
        ch.add_msg("C", sct=30.0)

        c1 = agg._sim_sync_recv_incremental(ch, ch.ends(), num_min_req=1)
        c2 = agg._sim_sync_recv_incremental(ch, ch.ends(), num_min_req=1)
        c3 = agg._sim_sync_recv_incremental(ch, ch.ends(), num_min_req=1)

        scts = [
            m[MessageType.SIM_COMPLETION_TS]
            for c in (c1, c2, c3)
            for m, _md in c
        ]
        # all three eventually commit, in ascending-sct order, across
        # separate num_min_req=1 calls -- nothing dropped.
        assert scts == [10.0, 30.0, 40.0]

    def test_never_double_commits_an_end_still_in_the_selected_set(self):
        agg = _FakeIncrementalAgg()
        ch = _FakeIncrementalChannel()
        ch.add_msg("A", sct=40.0)
        ch.add_msg("B", sct=10.0)

        c1 = agg._sim_sync_recv_incremental(ch, ch.ends(), num_min_req=1)
        assert [m[MessageType.SIM_COMPLETION_TS] for m, _ in c1] == [10.0]

        # B is still in ch.ends() (selector hasn't freed its slot yet) --
        # a second call must not re-add/re-commit it.
        c2 = agg._sim_sync_recv_incremental(ch, ch.ends(), num_min_req=1)
        assert [m[MessageType.SIM_COMPLETION_TS] for m, _ in c2] == [40.0]

        c3 = agg._sim_sync_recv_incremental(ch, ch.ends(), num_min_req=1)
        assert c3 == []

    def test_late_arriving_candidate_is_picked_up_on_a_later_call(self):
        agg = _FakeIncrementalAgg()
        ch = _FakeIncrementalChannel()
        ch.add_msg("A", sct=40.0)
        ch.select("B")  # dispatched, no message yet -- "still computing"

        c1 = agg._sim_sync_recv_incremental(ch, ch.ends(), num_min_req=1)
        assert [m[MessageType.SIM_COMPLETION_TS] for m, _ in c1] == [40.0]

        ch.add_msg("B", sct=5.0)  # B's grad lands after the first call
        c2 = agg._sim_sync_recv_incremental(ch, ch.ends(), num_min_req=1)
        assert [m[MessageType.SIM_COMPLETION_TS] for m, _ in c2] == [5.0]

    def test_vclock_advances_monotonically_across_incremental_calls(self):
        agg = _FakeIncrementalAgg()
        ch = _FakeIncrementalChannel()
        ch.add_msg("A", sct=40.0)
        ch.add_msg("B", sct=10.0)
        ch.add_msg("C", sct=30.0)

        agg._sim_sync_recv_incremental(ch, ch.ends(), num_min_req=1)
        assert agg._vclock.now == 10.0
        agg._sim_sync_recv_incremental(ch, ch.ends(), num_min_req=1)
        assert agg._vclock.now == 30.0
        agg._sim_sync_recv_incremental(ch, ch.ends(), num_min_req=1)
        assert agg._vclock.now == 40.0


class TestIncrementalBackwardCompat:
    def test_matches_batch_first_k_when_called_once(self):
        """A single call with num_min_req=k must still return the k smallest,
        ascending -- same observable result as the old one-shot
        `_sync_sim_recv_first_k` for the common (non-incremental) call shape."""
        agg = _FakeIncrementalAgg()
        ch = _FakeIncrementalChannel()
        for e, s in [("A", 40.0), ("B", 10.0), ("C", 30.0), ("D", 20.0)]:
            ch.add_msg(e, sct=s)

        committed = agg._sim_sync_recv_incremental(ch, ch.ends(), num_min_req=2)

        scts = [m[MessageType.SIM_COMPLETION_TS] for m, _md in committed]
        assert scts == [10.0, 20.0]
        assert agg._vclock.now == 20.0

    def test_stamps_client_task_train_duration_for_committed(self):
        agg = _FakeIncrementalAgg()
        ch = _FakeIncrementalChannel()
        ch.add_msg("B", sct=10.0, dur=10.0)
        ch.add_msg("A", sct=40.0, dur=40.0)

        agg._sim_sync_recv_incremental(ch, ch.ends(), num_min_req=1)

        assert ch.get_end_property("B", PROP_CLIENT_TASK_TRAIN_DURATION) == (
            timedelta(seconds=10.0)
        )
        # A not yet committed (still pending, not dropped) -> not stamped yet.
        assert ch.get_end_property("A", PROP_CLIENT_TASK_TRAIN_DURATION) is None

    def test_falls_back_to_sct_minus_send_when_no_duration_field(self):
        agg = _FakeIncrementalAgg()
        ch = _FakeIncrementalChannel()
        ch.add_msg("B", sct=10.0)  # no SIM_CLIENT_TASK_TRAIN_DURATION_S
        ch.set_end_property("B", PROP_SIM_SEND_TS, 3.0)

        agg._sim_sync_recv_incremental(ch, ch.ends(), num_min_req=1)

        assert ch.get_end_property("B", PROP_CLIENT_TASK_TRAIN_DURATION) == (
            timedelta(seconds=7.0)  # sct(10) - sim_send(3)
        )
