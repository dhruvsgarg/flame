# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""The async grad-loop sim drive.

In simulated mode fwdllm_aggregator._aggregate_grads_async commits a grad by the
modeled sim_completion_ts via _sim_recv_min_grad -- an sct-ordered reorder buffer
+ a one-in-flight gate + a virtual-clock advance -- so ordering is immune to
physical arrival jitter. These tests drive _sim_recv_min_grad directly with
synthetic out-of-order grad messages (a fake channel), and cover the
fwdllm-specific rollback-safety of _release_sim_slots_at_agg_goal (a data_id
spans many agg-goal cycles; a variance-FAIL rolls back to the same data_id and
must NOT strand or double-commit a grad).
"""

import time
from collections import deque
from datetime import datetime

from flame.mode.horizontal.asyncfl.top_aggregator import TopAggregator as _AsyncBase
from flame.mode.horizontal.syncfl.fwdllm_aggregator import TopAggregator
from flame.mode.horizontal.syncfl.top_aggregator import TopAggregator as _SyncBase
from flame.mode.message import MessageType
from flame.sim.virtual_clock import SimReorderBuffer, VirtualClock


class _FakeGradChannel:
    """Serves at most one grad message per end (as fwdllm's one-message-per-call
    loop expects). A message with release_at=k is only yielded once recv_fifo
    has been probed more than k times -- this models a straggler whose modeled
    completion is earlier but whose physical arrival is later, which is exactly
    what the in-flight gate must wait for."""

    def __init__(self, ends):
        self._ends_set = set(ends)
        self._msgs = {}         # end -> message dict
        self._release_at = {}   # end -> probe index gating physical arrival
        self._delivered = set()
        self._probe = 0

    def add_msg(self, end, sct, budget=None, release_at=0):
        m = {MessageType.SIM_COMPLETION_TS: sct}
        if budget is not None:
            m[MessageType.MODELED_DELAY_S] = budget
        self._ends_set.add(end)
        self._msgs[end] = m
        self._release_at[end] = release_at

    def ends(self, state=None):
        return list(self._ends_set)

    def has(self, end):
        return end in self._ends_set

    def recv_fifo(self, end_ids, first_k=None, timeout=None):
        self._probe += 1
        for e in end_ids:
            if e in self._delivered or e not in self._msgs:
                continue
            if self._probe > self._release_at.get(e, 0):
                self._delivered.add(e)
                yield self._msgs[e], (e, datetime.now())

    def drain_ready(self, end_ids, timeout=None):
        """Non-blocking snapshot: return every arrived-but-undelivered message
        for the given ends (mirrors channel.drain_ready's direct rxq sweep)."""
        self._probe += 1
        out = []
        for e in end_ids:
            if e in self._delivered or e not in self._msgs:
                continue
            if self._probe > self._release_at.get(e, 0):
                self._delivered.add(e)
                out.append((self._msgs[e], (e, datetime.now())))
        return out


class _FakeGradAgg:
    """Binds the real sim grad-loop methods onto a minimal stand-in."""

    _sim_recv_min_grad = TopAggregator._sim_recv_min_grad
    _release_sim_slots_at_agg_goal = TopAggregator._release_sim_slots_at_agg_goal
    _advance_sim_clock = _SyncBase._advance_sim_clock
    # §M: shared per-trainer delay cache primitives (syncfl.TopAggregator).
    _sim_recv_timeout_s = _SyncBase._sim_recv_timeout_s
    _note_sim_known_delay = _SyncBase._note_sim_known_delay
    _SIM_RECV_MARGIN_S = _SyncBase._SIM_RECV_MARGIN_S
    # §6 Part 3 (simulate_fwdllm.md §G): shared safe-fast-path check.
    _sim_gate_is_safe = _SyncBase._sim_gate_is_safe
    _SIM_GATE_FAST_PROBE_TIMEOUT_S = _SyncBase._SIM_GATE_FAST_PROBE_TIMEOUT_S
    # §6 Part 4 (simulate_fwdllm.md §G): shared visibility-lag primitive.
    _update_visibility_lag = _SyncBase._update_visibility_lag
    # #13 step 2 ready-gating helper (inherited by the real fwdllm agg from asyncfl).
    _sim_end_has_ready_msg = staticmethod(TopAggregator._sim_end_has_ready_msg)
    # #13 step 4 freed-slot FIFO consumer (inherited from asyncfl).
    _pop_free_slot_ts = _AsyncBase._pop_free_slot_ts
    # fwdllm overrides the hold with the two-lifetime split.
    _sim_hold_busy_slots = TopAggregator._sim_hold_busy_slots
    # The return-path guard/slot release (defers to COMMIT in sim residence).
    _release_end_on_return = TopAggregator._release_end_on_return

    def __init__(self):
        self.simulated = True
        self._round = 1
        self._vclock = VirtualClock()
        self._sim_buffer = SimReorderBuffer()
        self._sim_committed = set()
        self._sim_inflight_expected = {}
        self._sim_known_delay_s = {}  # §M: shared per-trainer delay cache
        self._sim_dispatch_wall = {}  # #16: cold-start gate dispatch stamps
        self._sim_pending_commit = set()
        self._inflight_residence = False
        self._sim_staggered_redispatch = False   # #13 step 4 (default off)
        self._sim_free_slot_ts = deque(maxlen=128)
        self._trainer_state_dict = {}
        self._curr_agg_version = (1, 0)
        # data_id is the progress axis _sim_recv_min_grad stamps ingested
        # grads with; carried-surplus classification reads it back at pop time.
        self.data_id = 0
        self._sim_enqueue_data_id = {}

    def _drain(self, channel, recv_ends, n):
        """Commit n grads, returning the ordered list of committed scts."""
        scts = []
        for _ in range(n):
            msg, md = self._sim_recv_min_grad(channel, recv_ends)
            if not msg:
                break
            scts.append(msg[MessageType.SIM_COMPLETION_TS])
        return scts


class TestSctOrderedCommit:
    def test_commits_in_sct_order_not_arrival_order(self):
        agg = _FakeGradAgg()
        ch = _FakeGradChannel([])
        # Physical arrival order A,B,C but modeled completion order B<C<A.
        ch.add_msg("A", sct=30.0)
        ch.add_msg("B", sct=10.0)
        ch.add_msg("C", sct=20.0)

        scts = agg._drain(ch, ["A", "B", "C"], 3)

        assert scts == [10.0, 20.0, 30.0]

    def test_vclock_monotone_advances_to_each_commit(self):
        agg = _FakeGradAgg()
        ch = _FakeGradChannel([])
        ch.add_msg("A", sct=30.0)
        ch.add_msg("B", sct=10.0)
        ch.add_msg("C", sct=20.0)

        seen = []
        for _ in range(3):
            msg, _md = agg._sim_recv_min_grad(ch, ["A", "B", "C"])
            seen.append(agg._vclock.now)
        assert seen == [10.0, 20.0, 30.0]
        assert seen == sorted(seen)  # monotone non-decreasing

    def test_learns_modeled_budget_from_messages(self):
        agg = _FakeGradAgg()
        ch = _FakeGradChannel([])
        ch.add_msg("A", sct=30.0, budget=8.0)
        ch.add_msg("B", sct=10.0, budget=5.0)

        agg._drain(ch, ["A", "B"], 2)

        # §M: each trainer's own exact MODELED_DELAY_S, no cross-trainer min.
        assert agg._sim_known_delay_s == {"A": 8.0, "B": 5.0}


class TestVisibilityLagTelemetry:
    """_sim_recv_min_grad now also calls the shared _update_visibility_lag
    alongside the existing _commit_gap computation. Must not change the
    committed number, only add the standardized field triplet."""

    def test_visibility_lag_matches_commit_gap_bit_for_bit(self):
        agg = _FakeGradAgg()
        ch = _FakeGradChannel([])
        ch.add_msg("A", sct=30.0)

        agg._sim_recv_min_grad(ch, ["A"])

        assert agg._sim_last_update_visibility_lag_s == agg._sim_last_commit_gap_s
        assert agg._sim_last_update_ready_ts == 30.0
        assert agg._sim_last_update_committed_ts == agg._vclock.now


class TestInFlightGate:
    def test_holds_commit_for_an_earlier_expected_straggler(self):
        """A (sct=100) arrives first physically; B (sct=50) is EXPECTED to
        complete earlier (_sim_inflight_expected[B]=48) but arrives on the 2nd
        probe. The gate must hold A and commit B first."""
        agg = _FakeGradAgg()
        agg._sim_inflight_expected = {"A": 98.0, "B": 48.0}
        ch = _FakeGradChannel([])
        ch.add_msg("A", sct=100.0, release_at=0)   # arrives first
        ch.add_msg("B", sct=50.0, release_at=1)     # arrives second

        first_msg, _md = agg._sim_recv_min_grad(ch, ["A", "B"])
        assert first_msg[MessageType.SIM_COMPLETION_TS] == 50.0   # B, not A
        assert agg._vclock.now == 50.0

        second_msg, _md = agg._sim_recv_min_grad(ch, ["A", "B"])
        assert second_msg[MessageType.SIM_COMPLETION_TS] == 100.0  # then A
        assert agg._vclock.now == 100.0


class _RecordingChannel(_FakeGradChannel):
    """Records the end sets passed to recv_fifo so a test can assert WHICH
    in-flight ends the drain chose to probe (#13 step 2 ready-gating). Optionally
    reports a subset of ends as physically READY (non-empty rxq) via _ready."""

    def __init__(self, ends, ready=()):
        super().__init__(ends)
        self.probe_calls = []          # recv_fifo end-id lists
        self.drain_calls = []          # drain_ready end-id lists
        self.probe_timeouts = []       # recv_fifo timeout values, call-aligned
        self.drain_timeouts = []       # drain_ready timeout values, call-aligned
        self._ready = set(ready)

    def recv_fifo(self, end_ids, first_k=None, timeout=None):
        self.probe_calls.append(list(end_ids))
        self.probe_timeouts.append(timeout)
        yield from super().recv_fifo(end_ids, first_k=first_k, timeout=timeout)

    def drain_ready(self, end_ids, timeout=None):
        self.drain_calls.append(list(end_ids))
        self.drain_timeouts.append(timeout)
        return super().drain_ready(end_ids, timeout=timeout)

    # Mirrors channel._ends[e].is_rxq_empty() as read by _sim_end_has_ready_msg.
    class _E:
        def __init__(self, empty):
            self._empty = empty

        def is_rxq_empty(self):
            return self._empty

    @property
    def _ends(self):
        return {e: _RecordingChannel._E(e not in self._ready) for e in self._ends_set}


class TestProbeCeilingReadyGating:
    """#13 step 2: an in-flight end that is NOT a recv_end is probed ONLY if it is
    physically ready OR its modeled `exp` is at/before the buffered minimum
    (+slack) -- so the drain stops burning the full grace window on far-future /
    not-yet-arrived stragglers each pass."""

    def test_far_future_straggler_is_not_probed(self):
        agg = _FakeGradAgg()
        # FAR is expected far past the buffered min and has no ready message; the
        # buffered A (sct=10) is the true next completion.
        agg._sim_inflight_expected = {"FAR": 1000.0}
        ch = _RecordingChannel([])
        ch._ends_set.add("FAR")            # in-flight but no message queued
        agg._sim_buffer.add("A", 10.0, ({MessageType.SIM_COMPLETION_TS: 10.0}, ("A", datetime.now())))

        msg, _ = agg._sim_recv_min_grad(ch, [])   # A not a recv_end (already buffered)
        assert msg[MessageType.SIM_COMPLETION_TS] == 10.0
        # FAR was never handed to recv_fifo -> no grace burned waiting on it.
        assert all("FAR" not in call for call in ch.probe_calls)

    def test_near_expected_straggler_is_probed(self):
        agg = _FakeGradAgg()
        # NEAR's exp (11) is within bmin(10)+slack -> the gate may still wait for
        # it, so it MUST be probed.
        agg._sim_inflight_expected = {"NEAR": 11.0}
        ch = _RecordingChannel([])
        ch.add_msg("NEAR", sct=11.0, release_at=0)
        agg._sim_buffer.add("A", 10.0, ({MessageType.SIM_COMPLETION_TS: 10.0}, ("A", datetime.now())))

        agg._sim_recv_min_grad(ch, [])
        assert any("NEAR" in call for call in ch.probe_calls)

    def test_physically_ready_straggler_is_probed_regardless_of_exp(self):
        agg = _FakeGradAgg()
        # READY is expected far in the future BUT its grad has physically arrived
        # (ready rxq) -> drain it now so it buffers as a future rather than being
        # committed past-dated later.
        agg._sim_inflight_expected = {"READY": 1000.0}
        ch = _RecordingChannel([], ready={"READY"})
        ch.add_msg("READY", sct=1000.0, release_at=0)
        agg._sim_buffer.add("A", 10.0, ({MessageType.SIM_COMPLETION_TS: 10.0}, ("A", datetime.now())))

        agg._sim_recv_min_grad(ch, [])
        assert any("READY" in call for call in ch.probe_calls)


class TestSafeFastPathTiming:
    """When the gate is already provably safe from in-memory state alone
    (bmin known, no in-flight end's known delay beats bmin - slack), the
    probe must use the tiny `_SIM_GATE_FAST_PROBE_TIMEOUT_S` bound instead of
    the full `_sim_recv_timeout_s` bound -- the fix for multi-second blocking
    waits. Asserts the timeout VALUE, unlike TestProbeCeilingReadyGating
    above which covers eager-probe behavior."""

    def test_fast_path_uses_tiny_timeout_when_already_safe_and_known(self):
        agg = _FakeGradAgg()
        # READY's delay is cached and its exp (1000) is nowhere near bmin(10)
        # -> gate is provably safe. Still probed (readiness overrides exp),
        # but the call must use the tiny fast-path timeout.
        agg._sim_known_delay_s["READY"] = 3.0
        agg._sim_inflight_expected = {"READY": 1000.0}
        ch = _RecordingChannel([], ready={"READY"})
        ch.add_msg("READY", sct=1000.0, release_at=0)
        agg._sim_buffer.add(
            "A", 10.0, ({MessageType.SIM_COMPLETION_TS: 10.0}, ("A", datetime.now()))
        )

        agg._sim_recv_min_grad(ch, [])
        assert any("READY" in call for call in ch.probe_calls)
        assert ch.probe_timeouts[0] == agg._SIM_GATE_FAST_PROBE_TIMEOUT_S
        # sanity: the fast-path bound really is tiny relative to the full bound
        # this pass would otherwise have used (known delay 3.0 + margin 0.5).
        assert ch.probe_timeouts[0] < agg._sim_recv_timeout_s(["READY"])

    def test_unknown_delay_end_in_mix_forces_full_bound_not_fast_path(self):
        """Even with an otherwise-safe gate, any end whose delay isn't cached
        yet must keep the fully-conservative behavior (block) -- the fast
        path must never fire on an uncertain end."""
        agg = _FakeGradAgg()
        # UNKNOWN has never been observed before (no _sim_known_delay_s entry)
        # and is physically ready, so it's probed same as the test above --
        # but its delay is uncached, so the fast path must NOT engage.
        agg._sim_inflight_expected = {"UNKNOWN": 1000.0}
        ch = _RecordingChannel([], ready={"UNKNOWN"})
        ch.add_msg("UNKNOWN", sct=1000.0, release_at=0)
        agg._sim_buffer.add(
            "A", 10.0, ({MessageType.SIM_COMPLETION_TS: 10.0}, ("A", datetime.now()))
        )

        agg._sim_recv_min_grad(ch, [])
        assert any("UNKNOWN" in call for call in ch.probe_calls)
        assert ch.probe_timeouts[0] is None  # genuinely blocking, unchanged

    def test_earlier_stuck_end_forces_full_bound_not_fast_path(self):
        """A genuinely stuck straggler (known delay, expected BEFORE the
        buffered minimum) must keep using the full computed bound -- the gate
        is not safe, so the fast path must not engage."""
        agg = _FakeGradAgg()
        # STUCK's known delay puts its exp (5) well before bmin(10) - slack ->
        # earlier_stuck -> the gate is NOT safe, must wait the full bound.
        agg._sim_known_delay_s["STUCK"] = 1.0
        agg._sim_inflight_expected = {"STUCK": 5.0}
        ch = _RecordingChannel([])
        ch.add_msg("STUCK", sct=5.0, budget=1.0, release_at=1)  # arrives on 2nd probe
        agg._sim_buffer.add(
            "A", 10.0, ({MessageType.SIM_COMPLETION_TS: 10.0}, ("A", datetime.now()))
        )

        agg._sim_recv_min_grad(ch, [])
        assert any("STUCK" in call for call in ch.probe_calls)
        assert ch.probe_timeouts[0] == agg._sim_recv_timeout_s(["STUCK"])
        assert ch.probe_timeouts[0] != agg._SIM_GATE_FAST_PROBE_TIMEOUT_S


class TestSctOrderedDrain:
    """#13 step 3: with sim_sct_ordered_drain ON, the drain ingests via
    channel.drain_ready (direct rxq sweep, no per-end grace timeout) instead of
    the blocking recv_fifo streamer, while keeping sct-ordered commit."""

    def test_commits_via_drain_ready_not_recv_fifo(self):
        agg = _FakeGradAgg()
        agg._sim_sct_ordered_drain = True
        ch = _RecordingChannel([])
        ch.add_msg("A", sct=30.0)
        ch.add_msg("B", sct=10.0)

        scts = agg._drain(ch, ["A", "B"], 2)
        assert scts == [10.0, 30.0]        # still committed in sct order
        assert ch.drain_calls              # drain_ready was used...
        assert not ch.probe_calls          # ...and recv_fifo was NOT

    def test_recv_fifo_path_when_flag_off(self):
        agg = _FakeGradAgg()               # flag defaults off
        ch = _RecordingChannel([])
        ch.add_msg("A", sct=10.0)

        agg._drain(ch, ["A"], 1)
        assert ch.probe_calls              # recv_fifo used
        assert not ch.drain_calls          # drain_ready NOT used

    def test_drain_ready_still_honors_inflight_gate(self):
        """A (sct=100) arrives; B (sct=50) is expected earlier and arrives on the
        2nd sweep -- the gate must still hold A and commit B first."""
        agg = _FakeGradAgg()
        agg._sim_sct_ordered_drain = True
        agg._sim_inflight_expected = {"A": 98.0, "B": 48.0}
        ch = _RecordingChannel([])
        ch.add_msg("A", sct=100.0, release_at=0)
        ch.add_msg("B", sct=50.0, release_at=1)

        first, _ = agg._sim_recv_min_grad(ch, ["A", "B"])
        assert first[MessageType.SIM_COMPLETION_TS] == 50.0   # B, not A
        second, _ = agg._sim_recv_min_grad(ch, ["A", "B"])
        assert second[MessageType.SIM_COMPLETION_TS] == 100.0


class TestFreedSlotRefill:
    """#13 step 4: a grad commit stamps the freed-slot vclock into the FIFO; a
    later dispatch pops it (oldest-first, clamped) as the re-dispatched trainer's
    SEND vclock -- spreading expected completions instead of collapsing the cohort
    at one round frontier."""

    def test_commit_stamps_freed_slot_vclock_when_staggered(self):
        agg = _FakeGradAgg()
        agg._sim_staggered_redispatch = True
        agg._sim_inflight_expected = {"X": 10.0}
        ch = _FakeGradChannel([])
        ch.add_msg("X", sct=10.0)

        agg._sim_recv_min_grad(ch, ["X"])
        # vclock advanced to sct=10 on commit -> that freed-slot vclock is stamped.
        assert list(agg._sim_free_slot_ts) == [10.0]

    def test_commit_does_not_stamp_when_flag_off(self):
        agg = _FakeGradAgg()               # staggered off (default)
        agg._sim_inflight_expected = {"X": 10.0}
        ch = _FakeGradChannel([])
        ch.add_msg("X", sct=10.0)

        agg._sim_recv_min_grad(ch, ["X"])
        assert len(agg._sim_free_slot_ts) == 0

    def test_pop_free_slot_ts_is_fifo_and_clamped(self):
        agg = _FakeGradAgg()
        agg._sim_free_slot_ts.extend([5.0, 8.0])
        assert agg._pop_free_slot_ts(100.0) == 5.0    # oldest first
        assert agg._pop_free_slot_ts(100.0) == 8.0
        assert agg._pop_free_slot_ts(100.0) == 100.0  # empty -> live frontier
        agg._sim_free_slot_ts.append(50.0)
        assert agg._pop_free_slot_ts(30.0) == 30.0    # min(stamp, round_now)

    def test_staggered_expected_completions_spread_not_bunched(self):
        """End-to-end at the unit level: two commits free slots at vclock 10 and
        25; a subsequent 2-trainer refill pops those as SEND vclocks, so expected
        completions (sst + budget) SPREAD (14, 29) instead of bunching at one
        round frontier (which would give the same expected for both)."""
        agg = _FakeGradAgg()
        agg._sim_staggered_redispatch = True
        _budget = 4.0  # a hypothetical known per-trainer delay for this check
        for e, s in [("A", 10.0), ("B", 25.0)]:
            agg._sim_inflight_expected = {e: s}
            ch = _FakeGradChannel([])
            ch.add_msg(e, sct=s)
            agg._sim_recv_min_grad(ch, [e])
        assert list(agg._sim_free_slot_ts) == [10.0, 25.0]

        sst1 = agg._pop_free_slot_ts(100.0)
        sst2 = agg._pop_free_slot_ts(100.0)
        exp1 = sst1 + _budget
        exp2 = sst2 + _budget
        assert (exp1, exp2) == (14.0, 29.0)   # spread, not (round_now+budget)×2


class TestStuckEndEviction:
    """#13 step 1: a trainer EXPECTED to complete earlier than the buffered
    minimum but never physically arrives must be evicted from
    _sim_inflight_expected on the recv failsafe deadline -- else `earlier_stuck`
    re-fires the full 30s deadline every drain cycle and the composer freezes."""

    def _immediate_deadline(self, monkeypatch):
        # Fire the recv failsafe on the first pass (no real 30s wait).
        import flame.mode.horizontal.syncfl.fwdllm_aggregator as fa
        monkeypatch.setattr(fa, "RECV_TIMEOUT_WAIT_S", 0.0)

    def test_stuck_end_evicted_on_deadline_and_buffered_min_commits(self, monkeypatch):
        self._immediate_deadline(monkeypatch)
        agg = _FakeGradAgg()
        # STUCK is expected early (10) but has no message; A arrived at sct=100.
        agg._sim_inflight_expected = {"STUCK": 10.0, "A": 98.0}
        ch = _FakeGradChannel([])
        ch.add_msg("A", sct=100.0, release_at=0)

        msg, _md = agg._sim_recv_min_grad(ch, ["A", "STUCK"])

        # The buffered min commits despite the earlier-expected straggler...
        assert msg[MessageType.SIM_COMPLETION_TS] == 100.0
        # ...and STUCK is dropped so it can't block future cycles.
        assert "STUCK" not in agg._sim_inflight_expected
        assert agg._sim_gate_failsafe == 1

    def test_evicted_end_does_not_block_the_next_cycle(self, monkeypatch):
        self._immediate_deadline(monkeypatch)
        agg = _FakeGradAgg()
        agg._sim_inflight_expected = {"STUCK": 10.0}
        ch = _FakeGradChannel([])
        ch.add_msg("A", sct=100.0, release_at=0)
        ch.add_msg("B", sct=200.0, release_at=0)

        # Cycle 1: STUCK holds the gate, hits the failsafe, is evicted; A commits.
        m1, _ = agg._sim_recv_min_grad(ch, ["A", "B", "STUCK"])
        assert m1[MessageType.SIM_COMPLETION_TS] == 100.0
        assert "STUCK" not in agg._sim_inflight_expected

        # Cycle 2: with STUCK gone, B commits WITHOUT re-arming the failsafe.
        m2, _ = agg._sim_recv_min_grad(ch, ["A", "B", "STUCK"])
        assert m2[MessageType.SIM_COMPLETION_TS] == 200.0
        assert agg._sim_gate_failsafe == 1  # not re-incremented

    def test_no_spurious_eviction_when_buffered_min_is_the_true_next(self, monkeypatch):
        # STUCK expected LATER than the buffered min -> not `earlier_stuck` ->
        # commit proceeds without touching the failsafe or the expected set.
        self._immediate_deadline(monkeypatch)
        agg = _FakeGradAgg()
        agg._sim_inflight_expected = {"LATE": 500.0}
        ch = _FakeGradChannel([])
        ch.add_msg("A", sct=100.0, release_at=0)

        msg, _ = agg._sim_recv_min_grad(ch, ["A", "LATE"])
        assert msg[MessageType.SIM_COMPLETION_TS] == 100.0
        assert agg._sim_inflight_expected == {"LATE": 500.0}  # untouched
        assert getattr(agg, "_sim_gate_failsafe", 0) == 0


class TestRollbackSafety:
    def test_no_double_commit_within_a_cycle(self):
        agg = _FakeGradAgg()
        agg._sim_inflight_expected = {"X": 10.0}
        ch = _FakeGradChannel([])
        ch.add_msg("X", sct=10.0)

        msg1, _ = agg._sim_recv_min_grad(ch, ["X"])
        assert msg1[MessageType.SIM_COMPLETION_TS] == 10.0
        assert agg._sim_committed == {"X"}

        # Second call in the SAME cycle: X is already committed and no new
        # message exists -> nothing committable, no re-commit.
        msg2, _ = agg._sim_recv_min_grad(ch, ["X"])
        assert msg2 is None

    def test_boundary_clear_lets_same_end_recontribute_after_rollback(self):
        agg = _FakeGradAgg()
        # --- cycle 1: X commits on data_id d, iteration 0 ---
        agg._sim_inflight_expected = {"X": 10.0}
        ch1 = _FakeGradChannel([])
        ch1.add_msg("X", sct=10.0)
        m1, _ = agg._sim_recv_min_grad(ch1, ["X"])
        assert m1[MessageType.SIM_COMPLETION_TS] == 10.0
        assert agg._sim_committed == {"X"}

        # --- variance FAIL -> rollback: agg-goal boundary cleanup ---
        agg._release_sim_slots_at_agg_goal(ch1, is_async=False)
        assert agg._sim_committed == set()          # not stranded as committed
        assert len(agg._sim_buffer) == 0            # no stranded grad
        assert agg._sim_inflight_expected == {}

        # --- cycle 2: SAME data_id, iteration 1, X re-contributes ---
        agg._sim_inflight_expected = {"X": 25.0}    # re-armed at dispatch
        ch2 = _FakeGradChannel([])
        ch2.add_msg("X", sct=25.0)
        m2, _ = agg._sim_recv_min_grad(ch2, ["X"])
        # X is committed again (NOT skipped as a stale committed mark).
        assert m2[MessageType.SIM_COMPLETION_TS] == 25.0
        assert agg._sim_committed == {"X"}
        assert agg._vclock.now == 25.0


class _FakeEnd:
    def set_property(self, *_args, **_kwargs):
        pass


class _FakeSelector:
    def __init__(self, selected):
        self.requester = "agg"
        self.all_selected = {e: 0.0 for e in selected}
        self.selected_ends = {"agg": set(selected)}


class _FakeSelChannel(_FakeGradChannel):
    def __init__(self, ends):
        super().__init__(ends)
        self._selector = _FakeSelector(ends)
        self._ends = {e: _FakeEnd() for e in ends}

    def add_msg(self, end, sct, budget=None, release_at=0):
        # Keep the real channel invariant has(e) <=> e in _ends (the slot-hold
        # path does channel._ends[e] guarded only by channel.has(e)).
        super().add_msg(end, sct, budget, release_at)
        self._ends.setdefault(end, _FakeEnd())

    # Faithful mirror of channel.cleanup_provided_ends -> selector
    # _cleanup_provided_ends: drop the end from the re-pick guard + slot.
    def cleanup_provided_ends(self, end):
        self._selector.all_selected.pop(end, None)
        self._selector.selected_ends[self._selector.requester].discard(end)

    def cleanup_recvd_end(self, end):  # sync path (random selector)
        self.cleanup_provided_ends(end)


class TestAsyncBoundaryReleasesSlots:
    def test_async_boundary_frees_every_committed_slot(self):
        agg = _FakeGradAgg()
        agg._sim_pending_commit = set()
        agg._inflight_residence = False
        ch = _FakeSelChannel(["X", "Y"])
        agg._sim_committed = {"X", "Y"}

        agg._release_sim_slots_at_agg_goal(ch, is_async=True)

        # buffer empty -> nothing held -> both slots released for re-selection.
        assert ch._selector.all_selected == {}
        assert ch._selector.selected_ends["agg"] == set()
        assert agg._sim_committed == set()


class TestFlagOffNoOp:
    def test_release_is_noop_when_not_simulated(self):
        agg = _FakeGradAgg()
        agg.simulated = False
        agg._sim_committed = {"X"}
        agg._sim_inflight_expected = {"X": 1.0}
        agg._sim_buffer.add("X", 1.0, None)

        # Real mode: the boundary hook must not touch any sim state.
        agg._release_sim_slots_at_agg_goal(channel=None, is_async=False)

        assert agg._sim_committed == {"X"}
        assert agg._sim_inflight_expected == {"X": 1.0}
        assert len(agg._sim_buffer) == 1


class TestComputeTruthfulGate:
    """#15: the compute-truthful commit gate (`sim_compute_truthful_gate`,
    flag-gated, default off) must not block a ready commit on a trainer that is
    NOT actually computing -- one stamped 'expected' at dispatch but idle-in-recv
    behind the single-threaded drain. That phantom wait burns the grace floor /
    30s failsafe and, via hold-to-commit, starves re-dispatch. The guard stays
    SELECTIVE: a genuine in-window straggler is still waited for, so sct-commit
    order is preserved."""

    def test_idle_phantom_is_skipped_and_ready_grad_commits(self):
        agg = _FakeGradAgg()
        agg._sim_compute_truthful_gate = True
        agg._sim_gate_compute_cap_s = 10.0
        agg._sim_dispatch_wall = {}   # PHANTOM: never dispatched -> no wall stamp
        agg._sim_inflight_expected = {"PHANTOM": 10.0, "A": 98.0}
        ch = _FakeGradChannel([])
        ch.add_msg("A", sct=100.0, release_at=0)  # A arrived; PHANTOM never will

        # No monkeypatched deadline: flag OFF would spin to the 30s failsafe;
        # flag ON skips the phantom and commits A at once.
        msg, _md = agg._sim_recv_min_grad(ch, ["A", "PHANTOM"])

        assert msg[MessageType.SIM_COMPLETION_TS] == 100.0
        assert agg._sim_gate_phantom_skip >= 1
        assert getattr(agg, "_sim_gate_failsafe", 0) == 0  # never hit the failsafe

    def test_stale_dispatch_is_treated_as_phantom(self):
        agg = _FakeGradAgg()
        agg._sim_compute_truthful_gate = True
        agg._sim_gate_compute_cap_s = 10.0
        # dispatched 30s ago -> older than the 10s cap -> not genuinely computing.
        agg._sim_dispatch_wall = {"STALE": time.time() - 30.0}
        agg._sim_inflight_expected = {"STALE": 10.0, "A": 98.0}
        ch = _FakeGradChannel([])
        ch.add_msg("A", sct=100.0, release_at=0)

        msg, _md = agg._sim_recv_min_grad(ch, ["A", "STALE"])
        assert msg[MessageType.SIM_COMPLETION_TS] == 100.0
        assert agg._sim_gate_phantom_skip >= 1

    def test_in_window_straggler_is_still_held(self):
        """Selective guard: a trainer dispatched WITHIN the compute window is a
        genuine straggler and must still be waited for -- commit order preserved."""
        agg = _FakeGradAgg()
        agg._sim_compute_truthful_gate = True
        agg._sim_gate_compute_cap_s = 10.0
        agg._sim_dispatch_wall = {"B": time.time()}   # just dispatched -> computing
        agg._sim_inflight_expected = {"A": 98.0, "B": 48.0}
        ch = _FakeGradChannel([])
        ch.add_msg("A", sct=100.0, release_at=0)  # arrives first
        ch.add_msg("B", sct=50.0, release_at=1)   # arrives on the 2nd probe

        first_msg, _md = agg._sim_recv_min_grad(ch, ["A", "B"])
        assert first_msg[MessageType.SIM_COMPLETION_TS] == 50.0  # B still held-for
        assert agg._vclock.now == 50.0

    def test_flag_off_is_byte_identical(self, monkeypatch):
        """Flag OFF (default): a phantom still blocks to the failsafe (unchanged)
        and phantom_skip stays 0."""
        import flame.mode.horizontal.syncfl.fwdllm_aggregator as fa
        monkeypatch.setattr(fa, "RECV_TIMEOUT_WAIT_S", 0.0)
        agg = _FakeGradAgg()  # flag defaults off (never set)
        agg._sim_inflight_expected = {"PHANTOM": 10.0, "A": 98.0}
        ch = _FakeGradChannel([])
        ch.add_msg("A", sct=100.0, release_at=0)

        msg, _md = agg._sim_recv_min_grad(ch, ["A", "PHANTOM"])
        assert msg[MessageType.SIM_COMPLETION_TS] == 100.0
        assert getattr(agg, "_sim_gate_phantom_skip", 0) == 0
        assert agg._sim_gate_failsafe == 1


def _full_grad_msg(sct, model_version=0, iteration=0):
    """A buffered grad message as _sim_recv_min_grad returns it -- carries the
    sct for the drain AND the grad fields _process_single_trainer_message needs."""
    return {
        MessageType.SIM_COMPLETION_TS: sct,
        MessageType.MODEL_VERSION: model_version,
        MessageType.ITERATION_PER_DATA_ID: iteration,
        MessageType.GRADIENTS: {},
        MessageType.GRADIENTS_FOR_VAR_CHECK: None,
        MessageType.STAT_UTILITY: 1.0,
    }


class _LoopChannel(_FakeSelChannel):
    """_FakeSelChannel + the bits _process_single_trainer_message reads."""

    def __init__(self, ends):
        super().__init__(ends)
        self._selector.ordered_updates_recv_ends = []

    def set_end_property(self, *_a, **_k):
        pass

    def get_end_property(self, *_a, **_k):
        return None


class _LoopAgg(_FakeGradAgg):
    """_FakeGradAgg (drain + slot-hold) + the real
    _process_single_trainer_message, so a test can drive the exact
    _aggregate_grads_async seam: commit (discard) THEN process."""

    from flame.mode.horizontal.syncfl.fwdllm_aggregator import (
        _OrderedContributorList as _OCL,
    )
    _process = TopAggregator._process_single_trainer_message
    _round_cache_clock_now = TopAggregator._round_cache_clock_now

    def __init__(self):
        super().__init__()
        self.is_async = True
        self._model_version = 0
        self.iteration_per_data_id = 0
        self._agg_goal_cnt = 0
        self._updates_in_queue = 0
        self._updates_received = {}
        self._per_agg_trainer_list = _LoopAgg._OCL()
        self.grad_pool = []
        self._trainer_last_model_version = {}
        self._round_cache_activity_ts = {}
        self._commit_key_by_end = {}


class TestCommitThenProcessFreesTheSlot:
    """Regression (simulate_fwdllm.md §F.1-23). In sim, _aggregate_grads_async
    calls _sim_recv_min_grad (COMMIT: discards the end from _sim_pending_commit
    at its sct) and THEN _process_single_trainer_message on that same grad. The
    latter must NOT re-add to _sim_pending_commit -- doing so re-pins every
    committed trainer, `selected_ends` never shrinks, distribute finds no free
    slot, and re-dispatch across variance-retry iterations deadlocks. The unit
    test checks _process in isolation; this drives the full seam and asserts the
    slot actually frees."""

    def _dispatched(self, ends, scts):
        agg = _LoopAgg()
        agg._inflight_residence = True
        ch = _LoopChannel(ends)
        agg._sim_pending_commit = set(ends)            # dispatch pinned them
        agg._sim_inflight_expected = dict(zip(ends, scts))
        for e, s in zip(ends, scts):
            ch._msgs[e] = _full_grad_msg(sct=s)         # grads arrived, buffered
        return agg, ch

    def test_committed_trainer_freed_not_repinned(self):
        agg, ch = self._dispatched(["X", "Y"], [10.0, 20.0])

        msg, md = agg._sim_recv_min_grad(ch, ["X", "Y"])   # commit X (min sct)
        assert msg[MessageType.SIM_COMPLETION_TS] == 10.0
        agg._process(ch, msg, md[0], md[1])                 # process the SAME grad

        # X committed -> unpinned + slot freed; Y still in flight.
        assert agg._sim_pending_commit == {"Y"}
        assert ch._selector.selected_ends["agg"] == {"Y"}

    def test_all_commits_drain_the_pool(self):
        """The deadlock signature: once every dispatched grad commits+processes,
        `_sim_pending_commit` must be EMPTY so distribute can re-dispatch. The
        bug left all of them pinned -> `selected_ends` stuck full -> None."""
        agg, ch = self._dispatched(["X", "Y"], [10.0, 20.0])

        for _ in range(2):
            msg, md = agg._sim_recv_min_grad(ch, ["X", "Y"])
            agg._process(ch, msg, md[0], md[1])

        assert agg._sim_pending_commit == set()
        assert ch._selector.selected_ends["agg"] == set()


class TestChargeSimVclockOverhead:
    """#6: the agg commit-side wall (drain-tail + FedAvg) is charged to the
    vclock DYNAMICALLY (the measured span, never a pre-profiled constant),
    sim-only + gated, with an over-threshold warning."""

    @staticmethod
    def _cfg(flag=True, warn_s=5.0):
        import types
        return types.SimpleNamespace(hyperparameters=types.SimpleNamespace(
            sim_model_agg_compute_time=flag, sim_overhead_warn_s=warn_s))

    def test_charges_measured_span_when_flag_on(self):
        from flame.mode.horizontal.syncfl.fwdllm_aggregator import (
            charge_sim_vclock_overhead as chg)
        vc, cfg = VirtualClock(), self._cfg()
        assert chg(vc, True, cfg, 0.8, "fedavg") == 0.8 and vc.now == 0.8
        # cumulative: a second charge advances further (drain-tail then fedavg).
        assert chg(vc, True, cfg, 0.2, "drain_tail") == 0.2
        assert abs(vc.now - 1.0) < 1e-9

    def test_no_charge_when_flag_off(self):
        from flame.mode.horizontal.syncfl.fwdllm_aggregator import (
            charge_sim_vclock_overhead as chg)
        vc = VirtualClock()
        assert chg(vc, True, self._cfg(flag=False), 0.8, "fedavg") == 0.0
        assert vc.now == 0.0

    def test_no_charge_in_real_mode(self):
        from flame.mode.horizontal.syncfl.fwdllm_aggregator import (
            charge_sim_vclock_overhead as chg)
        vc = VirtualClock()
        assert chg(vc, False, self._cfg(), 0.8, "fedavg") == 0.0
        assert vc.now == 0.0

    def test_zero_or_none_span_is_noop(self):
        from flame.mode.horizontal.syncfl.fwdllm_aggregator import (
            charge_sim_vclock_overhead as chg)
        vc = VirtualClock()
        assert chg(vc, True, self._cfg(), 0.0, "x") == 0.0
        assert chg(vc, True, self._cfg(), None, "x") == 0.0
        assert vc.now == 0.0

    def test_warns_over_threshold(self, caplog):
        import logging
        from flame.mode.horizontal.syncfl.fwdllm_aggregator import (
            charge_sim_vclock_overhead as chg)
        with caplog.at_level(logging.WARNING):
            chg(VirtualClock(), True, self._cfg(warn_s=1.0), 2.5, "drain_tail")
        assert any("SIM_OVERHEAD" in r.getMessage() for r in caplog.records)

    @staticmethod
    def _events(tmp_path, event_name):
        import json
        path = tmp_path / "aggregator.jsonl"
        if not path.exists():
            return []
        lines = path.read_text().splitlines()
        return [e for e in (json.loads(l) for l in lines) if e["event"] == event_name]

    def test_emits_vclock_charge_ledger_event(self, tmp_path):
        """Every call emits `vclock_charge`, both modes -- one shared ledger
        for any label, no per-call-site plumbing (simulate_fwdllm.md §D-11)."""
        from flame import telemetry
        from flame.mode.horizontal.syncfl.fwdllm_aggregator import (
            charge_sim_vclock_overhead as chg)
        telemetry.configure(role="aggregator", run_dir=str(tmp_path))
        try:
            vc = VirtualClock()
            chg(vc, True, self._cfg(), 0.8, "fedavg")
            evs = self._events(tmp_path, "vclock_charge")
            assert len(evs) == 1
            assert evs[0]["label"] == "fedavg"
            assert evs[0]["span_s"] == 0.8
            assert evs[0]["charged_s"] == 0.8
            assert evs[0]["time_mode"] == "sim"
            assert evs[0]["vclock_now"] == 0.8
        finally:
            telemetry.shutdown()

    def test_measurement_only_mode_never_charges_but_still_emits(self, tmp_path):
        """`charge=False` never advances the vclock, even with the flag on,
        but still logs `span_s` for a not-yet-decided candidate category."""
        from flame import telemetry
        from flame.mode.horizontal.syncfl.fwdllm_aggregator import (
            charge_sim_vclock_overhead as chg)
        telemetry.configure(role="aggregator", run_dir=str(tmp_path))
        try:
            vc = VirtualClock()
            result = chg(vc, True, self._cfg(), 3.0, "redispatch_turnaround",
                         charge=False, payload_kind="var_bad")
            assert result == 0.0
            assert vc.now == 0.0

            evs = self._events(tmp_path, "vclock_charge")
            assert len(evs) == 1
            assert evs[0]["label"] == "redispatch_turnaround"
            assert evs[0]["span_s"] == 3.0
            assert evs[0]["charged_s"] == 0.0
            assert evs[0]["payload_kind"] == "var_bad"
        finally:
            telemetry.shutdown()

    def test_real_mode_ledger_has_no_vclock_now(self, tmp_path):
        from flame import telemetry
        from flame.mode.horizontal.syncfl.fwdllm_aggregator import (
            charge_sim_vclock_overhead as chg)
        telemetry.configure(role="aggregator", run_dir=str(tmp_path))
        try:
            chg(None, False, self._cfg(), 1.5, "drain_tail")
            evs = self._events(tmp_path, "vclock_charge")
            assert len(evs) == 1
            assert evs[0]["time_mode"] == "real"
            assert evs[0]["charged_s"] == 0.0
            assert evs[0]["vclock_now"] is None
        finally:
            telemetry.shutdown()


class TestColdStartUnknownDelayGate:
    """A trainer's first-ever contact has no _sim_known_delay_s entry
    (reactive cache, no fallback), so earlier_stuck was blind to it and a
    cold run committed whatever arrived first. Unconditional -- independent
    of sim_compute_truthful_gate -- reuses the same sim_gate_compute_cap_s
    bound."""

    def test_holds_for_a_still_unknown_faster_trainer(self):
        """Neither A nor B has a known delay. A arrives first physically but
        B's modeled completion is earlier -- gate must hold A, commit B first."""
        agg = _FakeGradAgg()
        agg._sim_gate_compute_cap_s = 10.0
        now = time.time()
        agg._sim_dispatch_wall = {"A": now, "B": now}
        ch = _FakeGradChannel([])
        ch.add_msg("A", sct=100.0, release_at=0)  # arrives first physically
        ch.add_msg("B", sct=50.0, release_at=1)    # arrives on the 2nd probe

        first_msg, _md = agg._sim_recv_min_grad(ch, ["A", "B"])
        assert first_msg[MessageType.SIM_COMPLETION_TS] == 50.0  # B, not A

    def test_releases_after_cap_elapses_for_a_never_arriving_unknown_trainer(self):
        """A trainer dispatched long enough ago that it's past the compute cap
        can't block a commit forever -- same "not genuinely computing anymore"
        reasoning as the #15 phantom-skip path, just for an unknown-delay end."""
        agg = _FakeGradAgg()
        agg._sim_gate_compute_cap_s = 0.05
        agg._sim_dispatch_wall = {"GHOST": time.time() - 1.0}  # cap long elapsed
        ch = _FakeGradChannel([])
        ch.add_msg("A", sct=100.0, release_at=0)
        # GHOST never gets a message -- would otherwise block forever.

        msg, _md = agg._sim_recv_min_grad(ch, ["A", "GHOST"])
        assert msg[MessageType.SIM_COMPLETION_TS] == 100.0
        assert getattr(agg, "_sim_gate_failsafe", 0) == 0  # cap resolved it

    def test_no_dispatch_wall_stamp_does_not_hold(self):
        """An end with no _sim_dispatch_wall entry (never dispatched via the
        real path) can't spuriously trigger the cold-start hold -- matches
        production, where dispatch always stamps it unconditionally."""
        agg = _FakeGradAgg()
        agg._sim_gate_compute_cap_s = 10.0
        # _sim_dispatch_wall stays {} (default from __init__).
        ch = _FakeGradChannel([])
        ch.add_msg("A", sct=100.0, release_at=0)
        ch.add_msg("B", sct=50.0, release_at=5)  # would arrive much later

        msg, _md = agg._sim_recv_min_grad(ch, ["A", "B"])
        assert msg[MessageType.SIM_COMPLETION_TS] == 100.0  # A commits, doesn't wait for B

    def test_flag_off_still_applies_the_cold_start_gate(self):
        """Unlike #15's phantom-skip, this gate does NOT depend on
        sim_compute_truthful_gate -- it must hold even with that flag at its
        default (off)."""
        agg = _FakeGradAgg()  # sim_compute_truthful_gate never set (off)
        agg._sim_gate_compute_cap_s = 10.0
        now = time.time()
        agg._sim_dispatch_wall = {"A": now, "B": now}
        ch = _FakeGradChannel([])
        ch.add_msg("A", sct=100.0, release_at=0)
        ch.add_msg("B", sct=50.0, release_at=1)

        first_msg, _md = agg._sim_recv_min_grad(ch, ["A", "B"])
        assert first_msg[MessageType.SIM_COMPLETION_TS] == 50.0
