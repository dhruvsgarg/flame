# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Real-side streamer-free collect (`_real_sync_recv_incremental`), simulate_fwdllm.md §H.

fwdllm's `sync_collect_and_accumulate_grads` collected via `channel.recv_fifo`,
whose fire-and-forget per-end streamer tasks outlive their caller under the
num_min_req=1 loop: a slow trainer's already-arrived grad strands until the
RECV_TIMEOUT_WAIT_S grace fires (~40% of real collect wall). `_real_sync_recv_
incremental` refills a persistent arrival-ordered buffer from streamer-free
`drain_ready` instead. These tests drive it directly with a fake channel;
they assert prompt commit, arrival-time ordering, and the per-call
num_min_req cap with cross-call buffering.
"""

from collections import deque

import flame.mode.horizontal.syncfl.fwdllm_aggregator as agg_mod
from flame.mode.horizontal.syncfl.fwdllm_aggregator import TopAggregator


class _FakeDrainChannel:
    """Mirrors channel.drain_ready's direct rxq sweep: each `deliver` puts a
    message into an end's queue; a `drain_ready` sweep returns and clears every
    ready message (arrival-ordered by delivery). Ignores `timeout` -- delivery
    is explicit, so readiness never depends on wall time."""

    def __init__(self, ends):
        self._ends = list(ends)
        self._ready = deque()

    def ends(self, state=None):
        return list(self._ends)

    def has(self, end):
        return end in self._ends

    def deliver(self, end, ts, msg):
        if end not in self._ends:
            self._ends.append(end)
        self._ready.append((msg, (end, ts)))

    def drain_ready(self, end_ids, timeout=None):
        live = {e for e in end_ids}
        out, keep = [], deque()
        for msg, md in self._ready:
            (out if md[0] in live else keep).append((msg, md))
        self._ready = keep
        return out


class _FakeRealAgg:
    """Binds the real streamer-free collect onto a minimal stand-in."""

    _real_sync_recv_incremental = TopAggregator._real_sync_recv_incremental

    def __init__(self):
        self._real_sync_pending = []
        self._real_recv_seq = 0


def _msg(tag):
    return {"grad": tag}


class TestRealDrainReadyCollect:
    def test_ready_grad_commits_without_timeout(self):
        # A grad already sitting in the rxq is returned on the first
        # non-blocking sweep -- no deadline poll, no 30s stall.
        agg = _FakeRealAgg()
        ch = _FakeDrainChannel(["A"])
        ch.deliver("A", ts=1.0, msg=_msg("A"))
        out = agg._real_sync_recv_incremental(ch, num_min_req=1)
        assert [m["grad"] for m, _ in out] == ["A"]
        assert agg._real_sync_pending == []

    def test_commits_in_arrival_time_order(self):
        # Delivered out of timestamp order; a single num_min_req=3 call returns
        # them earliest-arrival first (recv_fifo's commit order, preserved).
        agg = _FakeRealAgg()
        ch = _FakeDrainChannel(["A", "B", "C"])
        ch.deliver("A", ts=30.0, msg=_msg("A"))
        ch.deliver("B", ts=10.0, msg=_msg("B"))
        ch.deliver("C", ts=20.0, msg=_msg("C"))
        out = agg._real_sync_recv_incremental(ch, num_min_req=3)
        assert [m["grad"] for m, _ in out] == ["B", "C", "A"]

    def test_num_min_req_cap_buffers_remainder_across_calls(self):
        # Three ready, num_min_req=1: return only the earliest; the other two
        # stay buffered and are served (still in order) by later calls with no
        # new delivery -- the incremental per-call cadence recv_fifo had.
        agg = _FakeRealAgg()
        ch = _FakeDrainChannel(["A", "B", "C"])
        ch.deliver("A", ts=30.0, msg=_msg("A"))
        ch.deliver("B", ts=10.0, msg=_msg("B"))
        ch.deliver("C", ts=20.0, msg=_msg("C"))
        first = agg._real_sync_recv_incremental(ch, num_min_req=1)
        assert [m["grad"] for m, _ in first] == ["B"]
        assert len(agg._real_sync_pending) == 2
        second = agg._real_sync_recv_incremental(ch, num_min_req=1)
        third = agg._real_sync_recv_incremental(ch, num_min_req=1)
        assert [m["grad"] for m, _ in second] == ["C"]
        assert [m["grad"] for m, _ in third] == ["A"]
        assert agg._real_sync_pending == []

    def test_empty_returns_after_deadline(self, monkeypatch):
        # Nothing ready: return an empty commit list once the recv deadline
        # lapses (bounded busy-poll), never blocking forever. Short-circuit the
        # module-level RECV_TIMEOUT_WAIT_S so the test doesn't wait the real 30s.
        monkeypatch.setattr(agg_mod, "RECV_TIMEOUT_WAIT_S", 0.05)
        agg = _FakeRealAgg()
        ch = _FakeDrainChannel(["A"])
        out = agg._real_sync_recv_incremental(ch, num_min_req=1)
        assert out == []
        assert agg._real_sync_pending == []
