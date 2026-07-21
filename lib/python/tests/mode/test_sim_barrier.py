# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Completion-barrier sim recv: single set-drain + logic invariance.

PARITY.md §6 fix: the sim aggregator must drain the whole in-flight/selected set
in ONE event-driven recv_fifo call (then commit by sim_completion_ts), instead of
probing each end with a fixed 0.5s timeout. These tests pin that property — they
fail loudly if per-end polling ever returns — while confirming the committed
order is still the sim_completion_ts order (logic unchanged, only wall pacing).
"""

from collections import defaultdict

import pytest

from flame.mode.horizontal.asyncfl.top_aggregator import (
    TopAggregator as AsyncAgg,
)
from flame.mode.horizontal.oort.top_aggregator import TopAggregator as OortAgg
from flame.mode.horizontal.syncfl.top_aggregator import TopAggregator as SyncAgg
from flame.mode.message import MessageType
from flame.sim import SimReorderBuffer, VirtualClock

SCTS = {"t1": 10.0, "t2": 5.0, "t3": 25.0, "t4": 15.0, "t5": 8.0}
SCRAMBLED = ["t3", "t1", "t5", "t4", "t2"]  # physical arrival ≠ sct order


class _End:
    def __init__(self):
        self._p = {}

    def get_property(self, k):
        return self._p.get(k)

    def set_property(self, k, v):
        self._p[k] = v


class RecordingChannel:
    """Models the real recv_fifo (drain ALL ready in one call, then (None,...))
    and records every recv_fifo invocation so a test can assert the barrier makes
    a single set-wide call rather than one call per end."""

    def __init__(self, scts, arrival_order):
        self._scts = dict(scts)
        self._queue = list(arrival_order)
        self._ends = {e: _End() for e in scts}
        self.recv_calls = []  # list of frozenset(end_ids) per call

    def has(self, e):
        return e in self._scts

    def ends(self, *a):
        return list(self._scts)

    def get_end_property(self, e, k):
        return self._ends[e].get_property(k)

    def set_end_property(self, e, k, v):
        self._ends[e].set_property(k, v)

    def recv_fifo(self, end_ids, first_k=0, timeout=None):
        self.recv_calls.append(frozenset(end_ids))
        ids = set(end_ids)
        i = 0
        while i < len(self._queue):
            e = self._queue[i]
            if e in ids:
                self._queue.pop(i)
                yield ({MessageType.WEIGHTS: f"w_{e}",
                        MessageType.SIM_COMPLETION_TS: self._scts[e]}, (e, None))
            else:
                i += 1
        yield (None, ("", None))


def _concrete(base):
    """Concrete subclass filling the abstract role methods so we can __new__ it."""
    return type("_C" + base.__name__, (base,), {
        "check_and_sleep": lambda self: None,
        "evaluate": lambda self: None,
        "initialize": lambda self: None,
        "load_data": lambda self: None,
        "train": lambda self: None,
    })


def _bare(cls):
    c = _concrete(cls)
    a = c.__new__(c)
    a._vclock = VirtualClock()
    a.simulated = True
    # Gate-off availability state (production sets these in __init__ /
    # _init_availability; __new__ bypasses both). The sim recv paths reference
    # _sim_buffer directly; trainer_event_dict=None keeps the mixin helpers no-op.
    a._sim_buffer = SimReorderBuffer()
    a.trainer_event_dict = None
    a.pending_withheld = {}
    a._sim_known_delay_s = {}
    return a


# ── single set-drain property (the speedup mechanism) ──────────────────────

def test_async_single_set_drain():
    agg = _bare(AsyncAgg)
    agg._sim_buffer = SimReorderBuffer()
    agg._sim_committed = set()
    agg._sim_pending_commit = set()
    ch = RecordingChannel(SCTS, SCRAMBLED)
    msg, (end, _) = agg._sim_recv_min(ch, ch.ends())
    # exactly ONE recv_fifo call, covering the full in-flight set (not per-end)
    assert len(ch.recv_calls) == 1, ch.recv_calls
    assert ch.recv_calls[0] == frozenset(SCTS)
    # logic preserved: the committed update is the global min sct (t2=5)
    assert end == "t2"
    assert agg._vclock.now == 5.0


def test_sync_single_set_drain():
    agg = _bare(SyncAgg)
    ch = RecordingChannel(SCTS, SCRAMBLED)
    out = agg._sync_sim_recv_first_k(ch, ch.ends(), first_k=3)
    assert len(ch.recv_calls) == 1, ch.recv_calls
    assert ch.recv_calls[0] == frozenset(SCTS)
    # logic preserved: the 3 smallest sct, ascending
    assert [md[0] for _m, md in out] == ["t2", "t5", "t1"]
    assert agg._vclock.now == 10.0  # k-th smallest


def test_oort_single_set_drain():
    agg = _bare(OortAgg)
    ch = RecordingChannel(SCTS, SCRAMBLED)
    committed = [md[0] for _m, md in agg._oort_sim_recv(ch, ch.ends())]
    assert len(ch.recv_calls) == 1, ch.recv_calls
    assert ch.recv_calls[0] == frozenset(SCTS)
    # logic preserved: yields all in ascending sct order
    assert committed == ["t2", "t5", "t1", "t4", "t3"]
    assert agg._vclock.now == 25.0


# ── §M shared per-trainer delay cache (replaces the grace EMA below) ───────

def test_note_sim_known_delay_populates_from_message():
    agg = _bare(SyncAgg)
    assert agg._sim_known_delay_s == {}
    agg._note_sim_known_delay("t1", {MessageType.MODELED_DELAY_S: 12.5})
    assert agg._sim_known_delay_s == {"t1": 12.5}
    # a later message for the same end is a harmless overwrite, not a decay/EMA
    agg._note_sim_known_delay("t1", {MessageType.MODELED_DELAY_S: 12.5})
    assert agg._sim_known_delay_s == {"t1": 12.5}


def test_note_sim_known_delay_ignores_none_and_missing():
    agg = _bare(SyncAgg)
    # None (training_delay_enabled=False) must stay distinguishable from
    # "not yet observed" -- do not cache it.
    agg._note_sim_known_delay("t2", {MessageType.MODELED_DELAY_S: None})
    assert "t2" not in agg._sim_known_delay_s
    # a message with no MODELED_DELAY_S key at all
    agg._note_sim_known_delay("t3", {MessageType.WEIGHTS: "w"})
    assert "t3" not in agg._sim_known_delay_s


def test_sim_recv_timeout_s_none_when_any_end_unknown():
    agg = _bare(SyncAgg)
    agg._sim_known_delay_s = {"t1": 10.0}
    # t2 has never been observed -> no bound for the whole cohort
    assert agg._sim_recv_timeout_s(["t1", "t2"]) is None
    assert agg._sim_recv_timeout_s([]) is None


def test_sim_recv_timeout_s_exact_bound_when_all_known():
    agg = _bare(SyncAgg)
    agg._sim_known_delay_s = {"t1": 10.0, "t2": 25.0}
    assert agg._sim_recv_timeout_s(["t1", "t2"]) == 25.0 + agg._SIM_RECV_MARGIN_S


def test_sync_barrier_zero_progress_when_all_delays_known_upfront():
    # Regression target: EMA-lock undershoot burned many zero-progress
    # barrier calls. All delays known -> one sufficient call.
    agg = _bare(SyncAgg)
    agg._sim_known_delay_s = {e: d for e, d in SCTS.items()}
    ch = RecordingChannel(SCTS, SCRAMBLED)
    out = agg._sync_sim_recv_first_k(ch, ch.ends(), first_k=len(SCTS))
    assert len(ch.recv_calls) == 1  # single barrier call
    assert len(out) == len(SCTS)  # drained the whole cohort, no shortfall
