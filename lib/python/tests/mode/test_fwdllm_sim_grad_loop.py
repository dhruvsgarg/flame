# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Batch 1 (simulate_fwdllm.md §I.8/§J.3): the async grad-loop sim drive.

fwdllm_aggregator._aggregate_grads_async committed a grad on WALL arrival; in
simulated mode it now commits by the modeled sim_completion_ts via
_sim_recv_min_grad -- an sct-ordered reorder buffer + a one-in-flight gate +
a virtual-clock advance -- so ordering is immune to physical arrival jitter.
These tests drive _sim_recv_min_grad directly with synthetic out-of-order grad
messages (a fake channel), and cover the fwdllm-specific rollback-safety of
_release_sim_slots_at_agg_goal (a data_id spans many agg-goal cycles; a
variance-FAIL rolls back to the same data_id and must NOT strand or double-
commit a grad).
"""

from datetime import datetime

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
            m[MessageType.TRAINING_BUDGET_S] = budget
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


class _FakeGradAgg:
    """Binds the real sim grad-loop methods onto a minimal stand-in."""

    _sim_recv_min_grad = TopAggregator._sim_recv_min_grad
    _release_sim_slots_at_agg_goal = TopAggregator._release_sim_slots_at_agg_goal
    _advance_sim_clock = _SyncBase._advance_sim_clock
    _sim_recv_grace_s = _SyncBase._sim_recv_grace_s
    # fwdllm overrides felix's hold with the Option-A two-lifetime split (K-D16).
    _sim_hold_busy_slots = TopAggregator._sim_hold_busy_slots
    # grace-window class knobs the base method reads off self
    SIM_RECV_GRACE_FLOOR_S = 0.0
    SIM_RECV_GRACE_FACTOR = 0.0

    def __init__(self):
        self.simulated = True
        self._round = 1
        self._vclock = VirtualClock()
        self._sim_buffer = SimReorderBuffer()
        self._sim_committed = set()
        self._sim_inflight_expected = {}
        self._sim_trainer_budget = {}
        self._sim_budget_min = 12.0
        self._sim_fill_ema = 0.0
        self._sim_pending_commit = set()
        self._sim_inflight_residence = False
        self._trainer_state_dict = {}

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

        assert agg._sim_trainer_budget == {"A": 8.0, "B": 5.0}
        assert agg._sim_budget_min == 5.0  # min(12.0, 8.0, 5.0)


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


class TestAsyncBoundaryReleasesSlots:
    def test_async_boundary_frees_every_committed_slot(self):
        agg = _FakeGradAgg()
        agg._sim_pending_commit = set()
        agg._sim_inflight_residence = False
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
