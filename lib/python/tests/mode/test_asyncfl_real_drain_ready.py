# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""FX-N18: asyncfl real ingest via streamer-free ``drain_ready`` (``real_drain_ready_ingest``).

recv_fifo's per-end streamer task left an arrived update in its rxq ("already has active
task") for seconds while others committed. ``_real_drain_recv`` sweeps the rxqs directly into
a persistent arrival-ordered buffer, and keeps a still-buffered end's in-flight slot (L4).
"""

from collections import deque

from flame.end import KEY_END_STATE, VAL_END_STATE_NONE, VAL_END_STATE_RECVD
from tests.mode.test_asyncfl_duplicate_contribution import _FakeChannel, _make_agg, _msg_for


class _End:
    def __init__(self):
        self.props = {}

    def set_property(self, k, v):
        self.props[k] = v


class _DrainChannel(_FakeChannel):
    """drain_ready returns and clears every delivered message of the asked ends, marking
    each RECVD as the real channel does. recv_fifo must never be called."""

    def __init__(self, ends):
        super().__init__([])
        self._ends = {e: _End() for e in ends}
        self._ready = deque()
        self.drain_calls = []

    def ends(self, state=None):
        return list(self._ends)

    def ends_with_pending_rx(self):
        return set()

    def has(self, end_id):
        return end_id in self._ends

    def deliver(self, end, ts, msg):
        self._ready.append((msg, (end, ts)))

    def drain_ready(self, end_ids, timeout=None):
        self.drain_calls.append((list(end_ids), timeout))
        out, keep = [], deque()
        for m, md in self._ready:
            (out if md[0] in end_ids else keep).append((m, md))
        self._ready = keep
        for _, (e, _) in out:
            self._ends[e].set_property(KEY_END_STATE, VAL_END_STATE_RECVD)
        return out

    def recv_fifo(self, *a, **k):
        raise AssertionError("recv_fifo called with real_drain_ready_ingest on")


def _agg(on=True):
    agg = _make_agg()
    agg._real_drain_ready_ingest = on
    agg._real_drain_pending = []
    agg._real_recv_seq = 0
    return agg


class TestRealDrainRecv:
    def test_ready_update_returns_on_nonblocking_sweep(self):
        agg, ch = _agg(), _DrainChannel(["a"])
        ch.deliver("a", 1.0, {"v": "a"})
        msg, md = agg._real_drain_recv(ch, ["a"])
        assert msg == {"v": "a"} and md == ("a", 1.0)
        assert ch.drain_calls == [(["a"], 0)]

    def test_arrival_order_across_calls_and_slot_kept(self):
        agg, ch = _agg(), _DrainChannel(["a", "b", "c"])
        ch.deliver("b", 2.0, {"v": "b"})
        ch.deliver("a", 1.0, {"v": "a"})
        ch.deliver("c", 3.0, {"v": "c"})
        assert agg._real_drain_recv(ch, ["a", "b", "c"])[0] == {"v": "a"}
        # b, c stay buffered and are reset to NONE so the selector keeps their slot.
        assert ch._ends["b"].props[KEY_END_STATE] == VAL_END_STATE_NONE
        assert ch._ends["c"].props[KEY_END_STATE] == VAL_END_STATE_NONE
        assert ch._ends["a"].props[KEY_END_STATE] == VAL_END_STATE_RECVD
        assert agg._real_drain_recv(ch, [])[0] == {"v": "b"}
        assert ch._ends["b"].props[KEY_END_STATE] == VAL_END_STATE_RECVD  # FX-D12: leaves RECV
        assert agg._real_drain_recv(ch, [])[0] == {"v": "c"}

    def test_same_end_two_versions_both_delivered(self):
        # The smoke case: one end's v123 and v124 both queued; neither waits on a streamer.
        agg, ch = _agg(), _DrainChannel(["a"])
        ch.deliver("a", 1.0, {"v": 123})
        ch.deliver("a", 2.0, {"v": 124})
        assert agg._real_drain_recv(ch, ["a"])[0] == {"v": 123}
        assert agg._real_drain_recv(ch, [])[0] == {"v": 124}

    def test_leave_notification_skipped_and_empty_returns_none(self):
        agg, ch = _agg(), _DrainChannel(["a"])
        ch.deliver("a", 1.0, None)
        msg, md = agg._real_drain_recv(ch, [])
        assert msg is None and md[0] == ""


    def test_idle_wait_ends_at_the_selector_reclaim(self):
        # FX-D50: a pick dispatched 89.5s ago is reclaimed at 90s; the idle wait ends then, not at the 30s poll.
        import time as _t
        agg, ch = _agg(), _DrainChannel(["a"])
        ch._selector = type("S", (), {"send_timeout_wait_s": 90.0, "ordered_updates_recv_ends": set(),
                                      "all_selected": {"a": _t.time() - 89.5}})()
        t0 = _t.time()
        assert agg._real_drain_recv(ch, ["a"])[0] is None
        assert _t.time() - t0 < 2.0


class TestAggregateWeightsRouting:
    def test_flag_on_routes_through_drain(self):
        agg, ch = _agg(), _DrainChannel(["t1"])
        agg.cm = type("CM", (), {"get_by_tag": lambda self, t: ch})()
        ch.deliver("t1", 1.0, _msg_for("t1")[0])
        agg._aggregate_weights("param-channel")
        assert agg._agg_goal_cnt == 1

    def test_buffered_update_committed_when_no_recv_ends(self):
        agg, ch = _agg(), _DrainChannel(["t1", "t2"])
        agg.cm = type("CM", (), {"get_by_tag": lambda self, t: ch})()
        ch.deliver("t1", 1.0, _msg_for("t1")[0])
        ch.deliver("t2", 2.0, _msg_for("t2")[0])
        agg._aggregate_weights("param-channel")
        ch.ends = lambda state=None: []  # nothing left in RECV; t2 sits in the buffer
        agg._aggregate_weights("param-channel")
        assert agg._agg_goal_cnt == 2

    def test_flag_absent_uses_recv_fifo(self):
        agg = _make_agg()  # bare init: no FX-N18 attributes ⇒ legacy path
        ch = _FakeChannel([_msg_for("t1")])
        agg.cm = type("CM", (), {"get_by_tag": lambda self, t: ch})()
        agg._aggregate_weights("param-channel")
        assert agg._agg_goal_cnt == 1

    def test_real_receipt_clears_withheld_ledger(self):
        # Run 4: real asyncfl never popped an evicted end, so its later dispatches counted as owed (real pool < sim).
        agg, ch = _agg(), _DrainChannel(["t1"])
        assert not agg.simulated
        agg.cm = type("CM", (), {"get_by_tag": lambda self, t: ch})()
        agg.pending_withheld = {"t1": 5.0}
        ch.deliver("t1", 1.0, _msg_for("t1")[0])
        agg._aggregate_weights("param-channel")
        assert agg.pending_withheld == {}
