# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Batch 1 (simulate_fwdllm.md §I.8/§J.3): the SYNC-barrier sim drive.

fwdllm/fwdllm_plus are sync -- their primary commit path is the barrier
(sync_collect_and_accumulate_grads). In simulated mode it commits the k
trainers with the SMALLEST modeled sim_completion_ts (the k that would
physically finish first in real), advancing the virtual clock to the k-th
smallest -- immune to physical arrival jitter -- via _sync_sim_recv_first_k.
The per-update U6 visibility lag on a strict barrier (lag_i = max_completion -
completion_i) is computed by _barrier_anchored_lags. These tests drive both
directly with the availability gate OFF (byte-identical to no-avail).
"""

from datetime import datetime, timedelta

from flame.mode.horizontal.syncfl.top_aggregator import TopAggregator as _SyncBase
from flame.mode.message import MessageType
from flame.sim.virtual_clock import SimReorderBuffer, VirtualClock
from flame.selector.properties import (
    PROP_CLIENT_TASK_TRAIN_DURATION,
    PROP_SIM_SEND_TS,
)


class _FakeBarrierChannel:
    """Drains every selected end in one recv_fifo pass (as the barrier does),
    yielding each end's single grad message out of sct order to prove the
    commit set is chosen by sct, not arrival."""

    def __init__(self):
        self._msgs = {}          # end -> message dict
        self._delivered = set()
        self._end_props = {}     # (end, prop) -> value

    def add_msg(self, end, sct, dur=None):
        m = {MessageType.SIM_COMPLETION_TS: sct}
        if dur is not None:
            m[MessageType.SIM_CLIENT_TASK_TRAIN_DURATION_S] = dur
        self._msgs[end] = m

    def ends(self, state=None):
        return list(self._msgs)

    def has(self, end):
        return end in self._msgs

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


class _FakeBarrierAgg:
    """Binds the real sync-barrier sim methods onto a minimal stand-in with the
    availability gate OFF (pending_withheld / trainer_event_dict unset)."""

    _sync_sim_recv_first_k = _SyncBase._sync_sim_recv_first_k
    _advance_sim_clock = _SyncBase._advance_sim_clock
    _sim_recv_timeout_s = _SyncBase._sim_recv_timeout_s
    _note_sim_known_delay = _SyncBase._note_sim_known_delay
    _sim_reinject_ready_withheld = _SyncBase._sim_reinject_ready_withheld
    _sim_withhold_if_unavail = _SyncBase._sim_withhold_if_unavail
    _barrier_anchored_lags = staticmethod(_SyncBase._barrier_anchored_lags)

    def __init__(self):
        self._round = 1
        self._vclock = VirtualClock()
        self._sim_buffer = SimReorderBuffer()
        self._sim_known_delay_s = {}


class TestSyncFirstKSmallestSct:
    def test_commits_the_k_smallest_sct_regardless_of_arrival(self):
        agg = _FakeBarrierAgg()
        ch = _FakeBarrierChannel()
        # arrival order A,B,C,D but modeled completion 40,10,30,20.
        ch.add_msg("A", sct=40.0, dur=40.0)
        ch.add_msg("B", sct=10.0, dur=10.0)
        ch.add_msg("C", sct=30.0, dur=30.0)
        ch.add_msg("D", sct=20.0, dur=20.0)

        committed = agg._sync_sim_recv_first_k(ch, ch.ends(), first_k=2)

        scts = [m[MessageType.SIM_COMPLETION_TS] for m, _md in committed]
        assert scts == [10.0, 20.0]  # the two SMALLEST, ascending

    def test_vclock_advances_to_kth_smallest_no_past_dating(self):
        agg = _FakeBarrierAgg()
        ch = _FakeBarrierChannel()
        for e, s in [("A", 40.0), ("B", 10.0), ("C", 30.0), ("D", 20.0)]:
            ch.add_msg(e, sct=s, dur=s)

        agg._sync_sim_recv_first_k(ch, ch.ends(), first_k=2)

        # advanced to the k-th (2nd) smallest = 20.0, never past-dated below it.
        assert agg._vclock.now == 20.0

    def test_stamps_client_task_train_duration_for_committed(self):
        agg = _FakeBarrierAgg()
        ch = _FakeBarrierChannel()
        ch.add_msg("B", sct=10.0, dur=10.0)
        ch.add_msg("A", sct=40.0, dur=40.0)

        agg._sync_sim_recv_first_k(ch, ch.ends(), first_k=1)

        # B committed -> its intrinsic train duration is exposed to the selector.
        assert ch.get_end_property("B", PROP_CLIENT_TASK_TRAIN_DURATION) == (
            timedelta(seconds=10.0)
        )
        # A dropped (past the first_k quota) -> not stamped.
        assert ch.get_end_property("A", PROP_CLIENT_TASK_TRAIN_DURATION) is None

    def test_falls_back_to_sct_minus_send_when_no_duration_field(self):
        agg = _FakeBarrierAgg()
        ch = _FakeBarrierChannel()
        ch.add_msg("B", sct=10.0)  # no SIM_CLIENT_TASK_TRAIN_DURATION_S
        ch.set_end_property("B", PROP_SIM_SEND_TS, 3.0)

        agg._sync_sim_recv_first_k(ch, ch.ends(), first_k=1)

        assert ch.get_end_property("B", PROP_CLIENT_TASK_TRAIN_DURATION) == (
            timedelta(seconds=7.0)  # sct(10) - sim_send(3)
        )


class TestBarrierAnchoredLags:
    def test_lag_is_max_completion_minus_each(self):
        lags = _SyncBase._barrier_anchored_lags([2.0, 5.0, 3.0])
        assert lags == [3.0, 0.0, 2.0]  # barrier = 5.0

    def test_none_safe_for_missing_completions(self):
        lags = _SyncBase._barrier_anchored_lags([2.0, None, 5.0])
        assert lags == [3.0, None, 0.0]

    def test_all_none_yields_all_none(self):
        assert _SyncBase._barrier_anchored_lags([None, None]) == [None, None]

    def test_empty_is_empty(self):
        assert _SyncBase._barrier_anchored_lags([]) == []
