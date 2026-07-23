# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Real ASYNC streamer-free collect (`_real_async_recv_min_grad`), simulate_fwdllm.md §H.

fwdllm's async grad loop collected one grad per call via
`next(channel.recv_fifo(RECV, 1))`, whose fire-and-forget per-end streamer
tasks outlive their caller: the next call skips a still-active end and its
already-arrived grad strands until the grace lapses (~0.4s/cohort real never
needs, masking a ~9% throughput gap). `_real_async_recv_min_grad` refills a
persistent arrival-ordered buffer from streamer-free `drain_ready` instead, so
a grad commits at T+D like sim. These tests drive it directly with a fake
channel: prompt commit, arrival-time order, cross-call buffering, and the
bounded empty return.
"""

from collections import deque

import flame.mode.horizontal.syncfl.fwdllm_aggregator as agg_mod
from flame.mode.horizontal.syncfl.fwdllm_aggregator import TopAggregator


class _FakeDrainChannel:
    """Mirrors channel.drain_ready's direct rxq sweep: `deliver` queues a message
    for an end; a `drain_ready` sweep returns and clears every ready message.
    Ignores `timeout` -- delivery is explicit, never wall-time dependent."""

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


class _FakeAsyncAgg:
    """Binds the real async streamer-free collect onto a minimal stand-in."""

    _real_async_recv_min_grad = TopAggregator._real_async_recv_min_grad

    def __init__(self):
        self._real_async_pending = []
        self._real_recv_seq = 0


def _msg(tag):
    return {"grad": tag}


class TestRealAsyncDrainReadyCollect:
    def test_ready_grad_commits_without_timeout(self):
        # A grad already in the rxq is returned on the first non-blocking sweep --
        # no deadline poll, no streamer stall.
        agg = _FakeAsyncAgg()
        ch = _FakeDrainChannel(["A"])
        ch.deliver("A", ts=1.0, msg=_msg("A"))
        msg, md = agg._real_async_recv_min_grad(ch)
        assert msg["grad"] == "A" and md[0] == "A"
        assert agg._real_async_pending == []

    def test_pops_earliest_arrival_buffers_rest(self):
        # Three ready out of order: one call returns the earliest-arrival grad;
        # the other two stay buffered (recv_fifo's commit order, preserved).
        agg = _FakeAsyncAgg()
        ch = _FakeDrainChannel(["A", "B", "C"])
        ch.deliver("A", ts=30.0, msg=_msg("A"))
        ch.deliver("B", ts=10.0, msg=_msg("B"))
        ch.deliver("C", ts=20.0, msg=_msg("C"))
        msg, _ = agg._real_async_recv_min_grad(ch)
        assert msg["grad"] == "B"
        assert len(agg._real_async_pending) == 2

    def test_buffer_carries_across_calls_in_order(self):
        # The incremental one-grad-per-call cadence: later calls serve the
        # buffered remainder in arrival order with no new delivery.
        agg = _FakeAsyncAgg()
        ch = _FakeDrainChannel(["A", "B", "C"])
        ch.deliver("A", ts=30.0, msg=_msg("A"))
        ch.deliver("B", ts=10.0, msg=_msg("B"))
        ch.deliver("C", ts=20.0, msg=_msg("C"))
        got = [agg._real_async_recv_min_grad(ch)[0]["grad"] for _ in range(3)]
        assert got == ["B", "C", "A"]
        assert agg._real_async_pending == []

    def test_a_later_earlier_arrival_still_wins_the_sort(self):
        # A grad delivered on a LATER call but with an earlier timestamp than the
        # buffered one commits first -- the non-blocking pre-sweep + sort keeps
        # true arrival order across calls, not FIFO-of-discovery.
        agg = _FakeAsyncAgg()
        ch = _FakeDrainChannel(["A", "B"])
        ch.deliver("A", ts=20.0, msg=_msg("A"))
        # Buffer A, then deliver an earlier-ts B before the next pop.
        assert agg._real_async_recv_min_grad(ch)[0]["grad"] == "A"  # only A ready
        ch.deliver("A", ts=40.0, msg=_msg("A2"))
        ch.deliver("B", ts=15.0, msg=_msg("B"))
        assert agg._real_async_recv_min_grad(ch)[0]["grad"] == "B"

    def test_empty_returns_none_after_deadline(self, monkeypatch):
        # Nothing ready: return (None, ("", now)) once the recv deadline lapses
        # (bounded busy-poll), never blocking forever. Short-circuit the 30s wait.
        monkeypatch.setattr(agg_mod, "RECV_TIMEOUT_WAIT_S", 0.05)
        agg = _FakeAsyncAgg()
        ch = _FakeDrainChannel(["A"])
        msg, md = agg._real_async_recv_min_grad(ch)
        assert msg is None and md[0] == ""
        assert agg._real_async_pending == []
