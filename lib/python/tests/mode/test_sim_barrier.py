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


def test_oort_never_awaits_an_end_queued_for_cleanup():
    # Run 18 speech oort syn_50 sim: a stale-rejected straggler stayed in selected_ends, was re-probed, and with a
    # newly picked end of unknown delay the barrier blocked forever.
    ch = RecordingChannel(SCTS, SCRAMBLED)
    ch._selector = type("S", (), {"ordered_updates_recv_ends": ["t3"]})()
    assert OortAgg._awaited_ends(ch, ["t1", "t3", "t4"]) == ["t1", "t4"]


def test_redispatched_end_with_stale_return_is_still_awaited():
    # FX-D109: t3's v84 update landed in the cleanup queue after its v117 dispatch; it still owes v117.
    from flame.mode.horizontal.oort.top_aggregator import TopAggregator as OortAgg
    from types import SimpleNamespace
    ch = SimpleNamespace(_selector=SimpleNamespace(ordered_updates_recv_ends=["t3", "t5"]))
    sent, got = {"t3": {84: 0.0, 117: 1.0}, "t5": {116: 0.0}}, {"t3": 84, "t5": 116}
    assert OortAgg._awaited_ends(ch, ["t1", "t3", "t5"], sent, got) == ["t1", "t3"]


def test_stale_dropped_return_is_not_awaited_again():
    # FX-D112: a stale-dropped v1 return must count as returned, or FX-D109 awaits it forever (T3 speech oort sim).
    from types import SimpleNamespace
    agg = _bare(OortAgg)
    agg.simulated, agg._round = True, 3
    props = {}
    ch = SimpleNamespace(_selector=SimpleNamespace(ordered_updates_recv_ends=["t3"]),
                         set_end_property=lambda e, k, v: props.__setitem__((e, k), v),
                         get_end_property=lambda e, k: props.get((e, k)))
    agg._record_returned_trainer_props(ch, "t3", {MessageType.MODEL_VERSION: 1}, None)
    assert OortAgg._awaited_ends(ch, ["t1", "t3"], {"t3": {1: 0.0}}, agg._returned_version) == ["t1"]


def test_sync_over_quota_update_carries_to_next_barrier():
    # FX-D113: G0U speech feddance sim blocked forever awaiting a straggler whose consumed update had been dropped.
    agg = _bare(SyncAgg)
    ch = RecordingChannel(SCTS, SCRAMBLED)
    agg._sync_sim_recv_first_k(ch, ch.ends(), first_k=3)
    out = agg._sync_sim_recv_first_k(ch, ch.ends(), first_k=2)
    assert ch.recv_calls[1] == frozenset(SCTS) - {"t3", "t4"}  # carried ends aren't awaited again
    assert sorted(md[0] for _m, md in out) == ["t3", "t4"]


def test_sync_wait_k_caps_co_due_deliveries_at_k():
    # FX-D122: sim took 6 commits at K=5; real closed at the 5th.
    agg = _bare(SyncAgg)
    agg._sync_wait_k_on = lambda: True
    agg._sim_take_withheld_delivering = lambda e: None
    agg._sim_reinject_ready_withheld = lambda: None
    for e in ("t1", "t2", "t3"):
        agg._sim_buffer.add(e, 300.0, ({}, (e, None)))
    ch = RecordingChannel({}, [])
    out = agg._sync_sim_recv_first_k(ch, [], first_k=2)
    assert len(out) == 2 and len(agg._sim_sync_carry) == 1


def test_sync_withheld_delivery_stamps_speed_and_ready():
    # FX-D124: delivered updates had speed 0 and a hold-inflated lag.
    from flame.selector.properties import PROP_CLIENT_TASK_TRAIN_DURATION
    agg = _bare(SyncAgg)
    agg._sim_take_withheld_delivering = lambda e: None
    agg._sim_reinject_ready_withheld = lambda: None
    ch = RecordingChannel({"t1": 1.0}, [])
    agg._sim_buffer.add("t1", 300.0, ({MessageType.SIM_CLIENT_TASK_TRAIN_DURATION_S: 42.0}, ("t1", None)))
    agg._sync_sim_recv_first_k(ch, [], first_k=1)
    assert ch.get_end_property("t1", PROP_CLIENT_TASK_TRAIN_DURATION).total_seconds() == 42.0
    assert agg._sim_ready_ts == {"t1": 300.0}


@pytest.mark.parametrize("dispatch_round, probed", [(9, {"t1", "t2"}), (10, {"t1", "t2", "t3"})])
def test_sync_wait_k_skips_picks_that_owe_nothing(dispatch_round, probed):
    # FX-D127: t3 replied stale; probing it blocked the barrier max(D) of wall. A re-dispatch at this version still owes (FX-D109).
    from flame.selector.properties import PROP_ROUND_START_TIME
    agg = _bare(SyncAgg)
    agg._round = 10
    agg._sync_wait_k_on = lambda: True
    ch = RecordingChannel(SCTS, ["t1", "t2"])
    ch._selector = type("S", (), {"ordered_updates_recv_ends": ["t3"]})()
    ch.set_end_property("t3", PROP_ROUND_START_TIME, (dispatch_round, None))
    agg._note_returned_version("t3", 9)
    out = agg._sync_sim_recv_first_k(ch, ["t1", "t2", "t3"], first_k=2)
    assert ch.recv_calls[0] == frozenset(probed)
    assert [md[0] for _m, md in out] == ["t2", "t1"]  # ascending sct


def test_superseded_return_neither_replies_nor_frees_the_slot():
    # FX-D129 (PR28 C2 real oort 0379): its v3 return sat in the cleanup queue after a v4 re-dispatch; round-end cleanup
    # freed the slot (re-picked while gated, EV17) and "replied" hid the v4 task from the 90 s abandon.
    from flame.selector.properties import PROP_ROUND_START_TIME
    agg = _bare(SyncAgg)
    agg._round = 4
    ch = RecordingChannel(SCTS, [])
    ch._selector = type("S", (), {"ordered_updates_recv_ends": ["t1", "t3"]})()
    for e, (sent, returned) in {"t1": (4, 4), "t3": (4, 3)}.items():
        ch.set_end_property(e, PROP_ROUND_START_TIME, (sent, None))
        agg._note_returned_version(e, returned)
    assert agg._sync_replied(ch) == {"t1"}
    agg._drop_superseded_returns(ch)
    assert ch._selector.ordered_updates_recv_ends == ["t1"]


def test_carried_withheld_delivery_commits_as_a_withheld_delivery():
    # FX-D130 (PR28 G1 speech feddance EV16): a delivery carried past K (FX-D122) committed through the fresh path:
    # no withheld_delivery event, gate re-applied, slot freed at carry instead of at commit (FX-D90).
    agg = _bare(SyncAgg)
    agg._round = 4
    msg = {MessageType.WEIGHTS: "w", MessageType.SIM_COMPLETION_TS: 207.1}
    agg._sim_sync_carry = {"t3": (300.0, (msg, ("t3", None)))}
    agg._sim_sync_carry_withheld = {"t3"}
    agg._sim_withheld_delivering = {"t3": (207.1, 300.0)}
    agg._sim_withhold_if_unavail = lambda *a: pytest.fail("a carried delivery already passed its send-gate")
    seen = []
    agg._emit_withheld_delivery = lambda end, m, sct, dts: seen.append((end, sct, dts, agg._vclock.now))
    out = agg._sync_sim_recv_first_k(RecordingChannel(SCTS, []), [], first_k=1)
    assert [md[0] for _m, md in out] == ["t3"]
    assert seen == [("t3", 207.1, 300.0, 300.0)]
    assert agg._sim_ready_ts["t3"] == 300.0 and "t3" not in agg._sim_withheld_delivering


class _LateChannel(RecordingChannel):
    """`late` ends reach the rxq only after the first recv_fifo pass (compute overran its known-delay bound)."""

    def __init__(self, scts, arrival_order, late):
        super().__init__(scts, [e for e in arrival_order if e not in late])
        self._late = [e for e in arrival_order if e in late]

    def recv_fifo(self, end_ids, first_k=0, timeout=None):
        if self.recv_calls:
            self._queue += self._late
            self._late = []
        return super().recv_fifo(end_ids, first_k, timeout)


@pytest.mark.parametrize("on", [True, False])
def test_sync_barrier_awaits_overrun_pick(on):
    # FX-D138 (PR29 C1 feddance T3 syn_50): 0379's stub compute 3.6 s > D bound 3.3 s; the barrier dropped it, wait-K
    # jumped the vclock to the next flip, and its sct-130 update committed at 150.
    from flame.selector.properties import PROP_SIM_SEND_TS
    from types import SimpleNamespace
    agg = _bare(SyncAgg)
    agg.config = SimpleNamespace(hyperparameters=SimpleNamespace(sim_barrier_awaits_picks=on))
    agg._sim_known_delay_s = dict(SCTS)
    ch = _LateChannel(SCTS, SCRAMBLED, late={"t2"})
    for e in SCTS:
        ch.set_end_property(e, PROP_SIM_SEND_TS, 0.0)
    out = agg._sync_sim_recv_first_k(ch, ch.ends(), first_k=3)
    assert [md[0] for _m, md in out] == (["t2", "t5", "t1"] if on else ["t5", "t1", "t4"])
    assert ch.recv_calls == [frozenset(SCTS)] + ([frozenset({"t2"})] if on else [])


def test_barrier_never_awaits_an_answered_dispatch():
    # FX-D138: a pick whose reply to its latest dispatch already arrived owes nothing; no second pass.
    from flame.selector.properties import PROP_SIM_SEND_TS
    agg = _bare(OortAgg)
    ch = RecordingChannel(SCTS, SCRAMBLED)
    for e in SCTS:
        ch.set_end_property(e, PROP_SIM_SEND_TS, 0.0)
    agg._sim_answered_sst = {"t9": 0.0}
    list(agg._sim_barrier_recv(ch, list(SCTS)))
    assert len(ch.recv_calls) == 1
    ch._queue = []
    list(agg._sim_barrier_recv(ch, ["t1"]))              # t1 answered dispatch 0.0 above: still one pass
    assert len(ch.recv_calls) == 2
