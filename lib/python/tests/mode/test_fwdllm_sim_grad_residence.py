# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Batch 2.5 (simulate_fwdllm.md §L.4 step 4): async grad-path residence +
commit-then-carry.

The 2026-07-03 fluxtune smoke ran 2x the forward passes of real: a residence
violation. `_release_sim_slots_at_agg_goal` cleared the reorder buffer + the
in-flight gate BEFORE `_sim_hold_busy_slots` read them, so it held nothing and
freed every trainer every agg-goal cycle -> re-dispatch -> recompute; and the
K-D6 buffer drop discarded the ~7 arrived-but-uncommitted grads (c=10 >> K=3).

The fix (K-D12): for the async path with `sim_inflight_residence` on, HOLD the
still-busy trainers (surplus buffered ∪ not-yet-arrived in-flight) in their
slots BEFORE clearing anything, release only the committed subset, and CARRY the
surplus buffer to the next cycle (never dropped). These tests drive the boundary
directly and assert (a) surplus carried, (b) busy trainers not re-selected,
(c) R1 one-in-flight residence holds, and (d) flag-off => byte-identical to the
Batch-1 drop behavior (sync baselines + async-without-residence unchanged).
"""

from flame.mode.horizontal.asyncfl.top_aggregator import (
    TopAggregator as _AsyncBase,
)
from flame.mode.horizontal.syncfl.fwdllm_aggregator import TopAggregator
from flame.mode.horizontal.syncfl.top_aggregator import TopAggregator as _SyncBase
from flame.mode.message import MessageType
from flame.sim.virtual_clock import SimReorderBuffer, VirtualClock

from tests.mode.test_fwdllm_sim_grad_loop import (
    _FakeGradAgg,
    _FakeSelChannel,
)


def _residence_agg(residence: bool) -> _FakeGradAgg:
    agg = _FakeGradAgg()
    agg._sim_pending_commit = set()
    agg._sim_inflight_residence = residence
    return agg


class TestCommitThenCarryResidenceOn:
    """async + sim_inflight_residence=True: hold-before-clear + carry surplus."""

    def test_surplus_carried_and_busy_held(self):
        agg = _residence_agg(residence=True)
        ch = _FakeSelChannel(["A", "B", "C", "D", "E"])
        # Committed this cycle: A, B (already popped from the buffer). Surplus
        # arrived: C, D (still buffered). Still in flight (not arrived): E.
        agg._sim_committed = {"A", "B"}
        agg._sim_buffer.add("C", 30.0, None)
        agg._sim_buffer.add("D", 40.0, None)
        agg._sim_inflight_expected = {"E": 50.0}

        agg._release_sim_slots_at_agg_goal(ch, is_async=True)

        # Surplus NOT dropped -- carried to the next cycle.
        assert set(agg._sim_buffer.pending_ends()) == {"C", "D"}
        # Not-yet-arrived trainer still in flight.
        assert agg._sim_inflight_expected == {"E": 50.0}
        # Per-cycle committed marks cleared so a re-contributor isn't skipped.
        assert agg._sim_committed == set()
        # Felix-aligned virtual-time in-flight (K-D17b): every OUTSTANDING trainer
        # (carried surplus C,D ∪ still-computing E) holds BOTH its re-pick guard
        # (all_selected) AND its compute slot (selected_ends) until it commits --
        # a returned-but-uncommitted trainer is still in flight in virtual time.
        # The two committed ones (A, B) are released.
        assert set(ch._selector.all_selected) == {"C", "D", "E"}
        assert ch._selector.selected_ends["agg"] == {"C", "D", "E"}

    def test_end_to_end_surplus_commits_next_cycle_no_refetch(self):
        agg = _residence_agg(residence=True)
        ends = ["A", "B", "C", "D", "E"]
        ch = _FakeSelChannel([])
        for e, sct in zip(ends, (10.0, 20.0, 30.0, 40.0, 50.0)):
            ch.add_msg(e, sct)
        agg._sim_inflight_expected = {e: sct + 0.1 for e, sct
                                      in zip(ends, (10, 20, 30, 40, 50))}

        # agg_goal = 2 commits this cycle (A, B by sct order); C, D, E arrive
        # into the buffer as surplus.
        scts = agg._drain(ch, ends, 2)
        assert scts == [10.0, 20.0]
        assert set(agg._sim_buffer.pending_ends()) == {"C", "D", "E"}

        agg._release_sim_slots_at_agg_goal(ch, is_async=True)
        # Carried, not dropped.
        assert set(agg._sim_buffer.pending_ends()) == {"C", "D", "E"}

        # Next cycle: the carried surplus commits from the buffer WITHOUT a new
        # dispatch/forward pass (no re-fetch), in sct order.
        scts2 = agg._drain(ch, ends, 3)
        assert scts2 == [30.0, 40.0, 50.0]

    def test_no_same_trainer_reselected_while_in_flight(self):
        """R1 residence: a held (still-busy) trainer must not reappear as a fresh
        selection slot -- that is exactly the re-dispatch-while-in-flight bug."""
        agg = _residence_agg(residence=True)
        ch = _FakeSelChannel(["A", "B", "C"])
        agg._sim_committed = {"A"}
        agg._sim_buffer.add("B", 20.0, None)      # surplus, arrived
        agg._sim_inflight_expected = {"C": 30.0}  # still computing

        agg._release_sim_slots_at_agg_goal(ch, is_async=True)

        held = set(ch._selector.all_selected)
        # The busy trainers stay held (not freed for a duplicate dispatch);
        # only the committed A is released.
        assert "B" in held and "C" in held and "A" not in held
        assert "B" in agg._sim_pending_commit and "C" in agg._sim_pending_commit
        # Felix-aligned (K-D17b): both the carried B and still-computing C keep
        # their compute slot (they're in flight in virtual time until commit);
        # neither can be re-picked (both in the all_selected guard).
        assert ch._selector.selected_ends["agg"] == {"B", "C"}


class TestVirtualInflightSlotHold:
    """K-D17b (felix-aligned, supersedes K-D16 Option-A): a returned-but-
    uncommitted trainer is still in flight in VIRTUAL time (its grad commits
    when the vclock reaches its sct), so it KEEPS its compute slot
    (selected_ends, drives `extra`) until COMMIT, not on physical return. Both
    ledgers track the same virtual-time in-flight set until commit."""

    def test_returned_trainer_keeps_slot_until_commit(self):
        agg = _residence_agg(residence=True)
        ch = _FakeSelChannel(["A", "B", "C", "D"])
        # A,B still computing; C,D returned (buffered surplus, not yet consumed).
        agg._sim_inflight_expected = {"A": 10.0, "B": 20.0, "C": 30.0, "D": 40.0}
        agg._sim_buffer.add("C", 30.0, None)
        agg._sim_buffer.add("D", 40.0, None)

        agg._release_sim_slots_at_agg_goal(ch, is_async=True)

        # Slot ledger (extra = c - len(selected_ends)): ALL four outstanding
        # trainers occupy a slot -- carried C,D are still in flight in virtual
        # time, so `in_flight` telemetry (= len(selected_ends)) counts them.
        assert ch._selector.selected_ends["agg"] == {"A", "B", "C", "D"}
        # Guard ledger: all four un-re-pickable until they commit.
        assert set(ch._selector.all_selected) == {"A", "B", "C", "D"}

    def test_selected_ends_tracks_inflight_across_a_commit(self):
        """The invariant that fixes the concurrency mis-measurement: after each
        commit, selected_ends == the still-outstanding (dispatched-not-committed)
        set. Committing one trainer releases exactly its slot; the rest stay."""
        agg = _residence_agg(residence=True)
        ch = _FakeSelChannel([])
        for e, sct in zip(["A", "B", "C"], (10.0, 20.0, 30.0)):
            ch.add_msg(e, sct)
        agg._sim_inflight_expected = {"A": 10.0, "B": 20.0, "C": 30.0}

        # Commit the smallest-sct grad (A). _sim_recv_min_grad reasserts the hold.
        scts = agg._drain(ch, ["A", "B", "C"], 1)
        assert scts == [10.0]
        # A committed -> slot released; B, C still in flight -> keep their slots.
        assert ch._selector.selected_ends["agg"] == {"B", "C"}
        assert "A" not in ch._selector.all_selected

    def test_triplet_guard_pruned_to_busy_on_commit(self):
        agg = _residence_agg(residence=True)
        ch = _FakeSelChannel(["A", "B", "C"])
        # A committed (popped) this cycle; B carried; C computing.
        agg._trainer_state_dict = {
            "A": (1, 0, 0), "B": (1, 0, 0), "C": (1, 0, 0),
        }
        agg._sim_committed = {"A"}
        agg._sim_buffer.add("B", 20.0, None)
        agg._sim_inflight_expected = {"C": 30.0}

        agg._release_sim_slots_at_agg_goal(ch, is_async=True)

        # The committed A is dropped from the triplet guard (re-pickable); the
        # still-outstanding B, C remain so async_oort's filter keeps skipping them.
        assert set(agg._trainer_state_dict) == {"B", "C"}


class TestFlagOffByteIdentical:
    """Residence OFF (default) => the Batch-1 legacy drop, unchanged. Sync
    baselines never take the carry path, so their boundary is untouched."""

    def test_async_residence_off_drops_and_releases_all(self):
        agg = _residence_agg(residence=False)
        ch = _FakeSelChannel(["X", "Y"])
        agg._sim_committed = {"X", "Y"}
        agg._sim_buffer.add("Z", 5.0, None)
        agg._sim_inflight_expected = {"W": 9.0}

        agg._release_sim_slots_at_agg_goal(ch, is_async=True)

        # Legacy drop: buffer + in-flight cleared, every slot released.
        assert len(agg._sim_buffer) == 0
        assert agg._sim_inflight_expected == {}
        assert agg._sim_committed == set()
        assert ch._selector.all_selected == {}

    def test_sync_barrier_always_drops_even_with_residence_flag(self):
        # The sync path (fwdllm/fwdllm_plus, c ~= agg_goal) has no surplus, so it
        # keeps the drop regardless of the residence flag.
        agg = _residence_agg(residence=True)
        agg._sim_committed = {"X"}
        agg._sim_buffer.add("X", 1.0, None)
        agg._sim_inflight_expected = {"X": 1.0}

        agg._release_sim_slots_at_agg_goal(channel=None, is_async=False)

        assert agg._sim_committed == set()
        assert len(agg._sim_buffer) == 0
        assert agg._sim_inflight_expected == {}

    def test_not_simulated_is_noop(self):
        agg = _residence_agg(residence=True)
        agg.simulated = False
        agg._sim_committed = {"X"}
        agg._sim_buffer.add("X", 1.0, None)
        agg._sim_inflight_expected = {"X": 1.0}

        agg._release_sim_slots_at_agg_goal(channel=None, is_async=True)

        assert agg._sim_committed == {"X"}
        assert len(agg._sim_buffer) == 1
        assert agg._sim_inflight_expected == {"X": 1.0}
