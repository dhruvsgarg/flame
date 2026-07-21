# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Async grad-path residence + commit-then-carry.

For the async path with `inflight_residence` on, `_release_sim_slots_at_agg_goal`
HOLDs the still-busy trainers (surplus buffered u not-yet-arrived in-flight) in
their slots BEFORE clearing anything, releases only the committed subset, and
CARRIEs the surplus buffer to the next cycle (never dropped) -- otherwise the
boundary re-dispatches busy trainers (2x forward passes) and drops arrived-but-
uncommitted grads. These tests drive the boundary directly and assert (a) surplus
carried, (b) busy trainers not re-selected, (c) R1 one-in-flight residence holds,
and (d) flag-off => byte-identical to the legacy drop behavior.
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
    agg._inflight_residence = residence
    return agg


class TestCommitThenCarryResidenceOn:
    """async + inflight_residence=True: hold-before-clear + carry surplus."""

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
        # Every OUTSTANDING trainer (carried surplus C,D u still-computing E) holds
        # BOTH its re-pick guard (all_selected) AND its compute slot (selected_ends)
        # until it commits; the two committed ones (A, B) are released.
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
        # Both the carried B and still-computing C keep their compute slot (in
        # flight in virtual time until commit); neither can be re-picked.
        assert ch._selector.selected_ends["agg"] == {"B", "C"}


class TestReturnPathGuardHeldToCommit:
    """The guard release on grad RETURN (`_release_end_on_return`, called from
    `_process_single_trainer_message`). With `_inflight_residence` on, RETURN
    must not release `all_selected` -- that tears a trainer out of the guard
    while its contribution hasn't committed -> re-selectable -> re-dispatch-
    while-in-flight. One check, no `is_async` branch: a no-op for sync
    (agg_goal == c, so every return already belongs to that cycle's commit)
    and for real (which needs the hold too, unlike sim -- §R 2026-07-11).
    """

    def test_guard_held_on_return_in_sim_residence(self):
        """async + sim + residence: return must NOT release the re-pick guard
        (held to COMMIT by _sim_hold_busy_slots). This is the regression guard."""
        agg = _residence_agg(residence=True)
        agg.is_async = True
        ch = _FakeSelChannel(["A", "B", "C"])   # all dispatched + in flight
        agg._release_end_on_return(ch, "A")     # A's grad returns (carried)
        # A stays in the guard -> cannot be re-picked while still outstanding.
        assert "A" in ch._selector.all_selected
        assert "A" in ch._selector.selected_ends["agg"]

    def test_guard_released_on_return_when_residence_off(self):
        """async WITHOUT residence: legacy behavior -- release immediately
        (return ~= commit), so the fix is byte-identical off the residence path."""
        agg = _residence_agg(residence=False)
        agg.is_async = True
        ch = _FakeSelChannel(["A", "B", "C"])
        agg._release_end_on_return(ch, "A")
        assert "A" not in ch._selector.all_selected

    def test_guard_held_on_return_in_sync_residence(self):
        """§R (2026-07-11): sync + residence on ALSO holds now -- one invariant,
        no is_async special case. Harmless in practice (agg_goal == c for sync,
        so this return already belongs to the cycle that's about to commit),
        but must not silently diverge from async's behavior."""
        agg = _residence_agg(residence=True)
        agg.is_async = False
        ch = _FakeSelChannel(["A", "B", "C"])
        agg._release_end_on_return(ch, "A")
        assert "A" in ch._selector.all_selected

    def test_sync_return_uses_recvd_cleanup_when_residence_off(self):
        """sync (random selector, is_async=False) WITHOUT residence: unchanged
        cleanup_recvd_end path -- releases on return, default/legacy."""
        agg = _residence_agg(residence=False)
        agg.is_async = False
        ch = _FakeSelChannel(["A", "B", "C"])
        agg._release_end_on_return(ch, "A")
        assert "A" not in ch._selector.all_selected

    def test_guard_held_on_return_in_real_residence(self):
        """§R (2026-07-11): async + REAL + residence on must ALSO hold to
        commit -- previously gated on `simulated`, so real always fell
        through to immediate release regardless of the flag."""
        agg = _residence_agg(residence=True)
        agg.is_async = True
        agg.simulated = False
        ch = _FakeSelChannel(["A", "B", "C"])
        agg._release_end_on_return(ch, "A")
        assert "A" in ch._selector.all_selected
        assert "A" in ch._selector.selected_ends["agg"]

    def test_guard_released_on_return_in_real_when_residence_off(self):
        """async + REAL + residence off: legacy immediate release preserved."""
        agg = _residence_agg(residence=False)
        agg.is_async = True
        agg.simulated = False
        ch = _FakeSelChannel(["A", "B", "C"])
        agg._release_end_on_return(ch, "A")
        assert "A" not in ch._selector.all_selected


class TestReturnPathBufferedReleasesImmediately:
    """07-21 fix (simulate_fwdllm.md FT cohort_sequence deep-dive): once a
    contribution is captured in `_pending_cohort_contribs` (P0-1 deferred-
    merge), re-dispatch can't lose/overwrite it -- `buffered=True` releases
    the slot immediately even with `_inflight_residence` on, instead of
    deferring to the whole cohort's commit. Restores true fedbuff continuous
    concurrency (flat, not a sawtooth) without reopening the R1 gap
    `_inflight_residence` was built to close for un-buffered returns."""

    def test_buffered_true_releases_immediately_despite_residence(self):
        agg = _residence_agg(residence=True)
        agg.is_async = True
        ch = _FakeSelChannel(["A", "B", "C"])
        agg._release_end_on_return(ch, "A", buffered=True)
        assert "A" not in ch._selector.all_selected
        assert "A" not in ch._selector.selected_ends["agg"]

    def test_buffered_false_still_holds_to_commit(self):
        """Explicit buffered=False (e.g. a non-gradient message) preserves
        the pre-fix hold-to-commit behavior -- same as the default."""
        agg = _residence_agg(residence=True)
        agg.is_async = True
        ch = _FakeSelChannel(["A", "B", "C"])
        agg._release_end_on_return(ch, "A", buffered=False)
        assert "A" in ch._selector.all_selected

    def test_buffered_true_in_real_releases_immediately(self):
        agg = _residence_agg(residence=True)
        agg.is_async = True
        agg.simulated = False
        ch = _FakeSelChannel(["A", "B", "C"])
        agg._release_end_on_return(ch, "A", buffered=True)
        assert "A" not in ch._selector.all_selected

    def test_buffered_true_with_residence_off_still_releases(self):
        """residence off already released immediately -- buffered=True must
        not change that (no double-release / no-op path)."""
        agg = _residence_agg(residence=False)
        agg.is_async = True
        ch = _FakeSelChannel(["A", "B", "C"])
        agg._release_end_on_return(ch, "A", buffered=True)
        assert "A" not in ch._selector.all_selected


class TestVirtualInflightSlotHold:
    """A returned-but-uncommitted trainer is still in flight in VIRTUAL time (its
    grad commits when the vclock reaches its sct), so it KEEPS its compute slot
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

    def test_triplet_guard_kept_until_tuple_advances_not_on_commit(self):
        # RC3 fix: a trainer's contributed version_key stamp survives its COMMIT
        # and is dropped only when the agg advances PAST that version_key.
        # Dropping it on commit (the old behavior) let a fast committer be
        # re-picked for the SAME version_key -> abort_training -> phantom
        # starvation.
        agg = _residence_agg(residence=True)
        agg._curr_agg_version = (1, 0)
        ch = _FakeSelChannel(["A", "B", "C", "D"])
        # A committed this cycle; B carried; C computing -- all at the CURRENT
        # version_key (1,0). D contributed to a PAST (1,0)-predecessor.
        agg._trainer_state_dict = {
            "A": (1, 0), "B": (1, 0), "C": (1, 0), "D": (0, 0),
        }
        agg._sim_committed = {"A"}
        agg._sim_buffer.add("B", 20.0, None)
        agg._sim_inflight_expected = {"C": 30.0}

        agg._release_sim_slots_at_agg_goal(ch, is_async=True)

        # A/B/C keep their (1,0) stamp (still the current version_key) -> the
        # committed A is NOT re-pickable for (1,0). Only D, stamped at a
        # superseded version_key, is dropped (re-pickable for the current one).
        assert set(agg._trainer_state_dict) == {"A", "B", "C"}

    def test_triplet_guard_all_dropped_once_tuple_advances(self):
        # When the agg has moved to a new version_key, every stale stamp drops
        # -> all trainers re-enter the pool for the new (model_version, iteration).
        agg = _residence_agg(residence=True)
        agg._curr_agg_version = (1, 1)   # iteration advanced past (1,0)
        ch = _FakeSelChannel(["A", "B"])
        agg._trainer_state_dict = {"A": (1, 0), "B": (1, 0)}
        agg._sim_committed = {"A"}
        agg._sim_inflight_expected = {"B": 30.0}

        agg._release_sim_slots_at_agg_goal(ch, is_async=True)

        assert agg._trainer_state_dict == {}


class TestFlagOffByteIdentical:
    """Residence OFF (default) => the legacy drop, unchanged. Sync baselines
    never take the carry path, so their boundary is untouched."""

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


class TestPendingCommitBridge:
    """The fwdllm aggregator maintains its VIRTUAL in-flight set
    (`_sim_pending_commit`) and BINDS it to the selector's
    `_agg_pending_commit_ref` so async_oort's eligibility filter excludes a
    returned-but-uncommitted trainer regardless of `all_selected` churn. The
    reconcile must SHRINK (a committed trainer becomes re-pickable); a
    `|= outstanding` accumulate would starve every committed trainer forever.
    """

    def test_hold_reconciles_pending_to_outstanding_and_binds_ref(self):
        agg = _residence_agg(residence=True)
        ch = _FakeSelChannel(["A", "B", "C"])
        agg._sim_committed = {"A"}                 # committed this cycle
        agg._sim_buffer.add("B", 20.0, None)       # returned, carried (uncommitted)
        agg._sim_inflight_expected = {"C": 30.0}   # still computing

        agg._sim_hold_busy_slots(ch)

        # pending == still-outstanding (carried B ∪ computing C); committed A dropped.
        assert agg._sim_pending_commit == {"B", "C"}
        # bound to the SAME live object the selector reads (never rebind).
        assert ch._selector._agg_pending_commit_ref is agg._sim_pending_commit

    def test_committed_trainer_drops_out_not_starved(self):
        agg = _residence_agg(residence=True)
        ch = _FakeSelChannel(["A", "B"])
        agg._sim_inflight_expected = {"A": 10.0, "B": 20.0}
        agg._sim_hold_busy_slots(ch)
        assert agg._sim_pending_commit == {"A", "B"}

        # A commits -> leaves in-flight, marked committed. `outstanding` now also
        # reads `_sim_pending_commit` (first-dispatch fix, see the `outstanding`
        # comment in _sim_hold_busy_slots), so a faithful simulation of "A
        # committed" must include the same discard the real commit path
        # (`_sim_recv_min_grad`) always performs before this function is next
        # reached -- not just mutating `_sim_inflight_expected`.
        agg._sim_inflight_expected = {"B": 20.0}
        agg._sim_committed = {"A"}
        agg._sim_pending_commit.discard("A")
        agg._sim_hold_busy_slots(ch)

        # A is re-pickable again (dropped from pending); `|=` would have kept it.
        assert agg._sim_pending_commit == {"B"}

    def test_unseen_delay_trainer_stays_held_until_commit(self):
        """R1 regression (07-15 fluxtune telemetry, trainer ...0449): a trainer's
        FIRST-EVER dispatch in a run has no `_sim_known_delay_s` entry yet (it's
        only learned from a trainer's OWN prior message), so `_sim_inflight_
        expected` never gets one either -- deliberately, see test_train_staggered_
        unseen_trainer_gets_no_gate_entry. Before this fix, `outstanding` read
        only `_sim_inflight_expected | buffered`, so such a trainer was invisible
        to it and got wiped from `all_selected` the moment ANY OTHER trainer's
        commit triggered this reconcile -- selectable again seconds before its
        own grad could possibly return (measured as `r1_inflight_overlap`'s
        19.4%). `_sim_pending_commit` (added unconditionally at dispatch,
        regardless of whether the delay is known) must keep it held instead.
        """
        agg = _residence_agg(residence=True)
        ch = _FakeSelChannel(["NEW", "B"])
        # NEW was just dispatched (the real dispatch path adds to all_selected +
        # selected_ends + _sim_pending_commit unconditionally) but has no known
        # delay yet, so _sim_inflight_expected has nothing for it.
        agg._sim_pending_commit = {"NEW"}
        agg._sim_inflight_expected = {"B": 20.0}   # B has a known delay, still computing

        agg._sim_hold_busy_slots(ch)

        assert "NEW" in agg._sim_pending_commit
        assert "NEW" in ch._selector.all_selected
        assert "NEW" in ch._selector.selected_ends["agg"]

    def test_commit_discards_from_pending(self):
        agg = _residence_agg(residence=True)
        ch = _FakeSelChannel([])
        for e, sct in zip(["A", "B"], (10.0, 20.0)):
            ch.add_msg(e, sct)
        agg._sim_inflight_expected = {"A": 10.0, "B": 20.0}
        agg._sim_pending_commit = {"A", "B"}

        agg._drain(ch, ["A", "B"], 1)   # commit the smallest-sct grad (A)

        assert "A" not in agg._sim_pending_commit   # discarded on COMMIT
        assert "B" in agg._sim_pending_commit       # still in flight

    def test_legacy_drop_path_clears_pending(self):
        # residence OFF -> the boundary drops all in-flight; pending must clear
        # too (else the sync-barrier set grows unbounded).
        agg = _residence_agg(residence=False)
        ch = _FakeSelChannel([])
        agg._sim_committed = {"X"}
        agg._sim_buffer.add("Y", 5.0, None)
        agg._sim_inflight_expected = {"Z": 9.0}
        agg._sim_pending_commit = {"X", "Y", "Z"}

        agg._release_sim_slots_at_agg_goal(ch, is_async=True)

        assert agg._sim_pending_commit == set()

    def test_recommitted_trainer_stays_pending_despite_stale_committed(self):
        """`_sim_committed` is a STALE cross-cycle marker (cleared only at the
        boundary). A trainer that committed then was re-picked + re-dispatched is
        back in `_sim_inflight_expected`; the reconcile must NOT drop it from
        pending just because it lingers in `_sim_committed` -- else it is
        re-pickable while its NEW dispatch is still in flight (R1 overlap).
        `outstanding` therefore keys on inflight/buffer membership only, never
        `- _sim_committed`."""
        agg = _residence_agg(residence=True)
        ch = _FakeSelChannel(["A", "B"])
        # A committed earlier this cycle (still in the stale marker) AND has been
        # re-dispatched -> back in flight. B is a first-time in-flight trainer.
        agg._sim_committed = {"A"}
        agg._sim_inflight_expected = {"A": 40.0, "B": 20.0}

        agg._sim_hold_busy_slots(ch)

        # A is genuinely in flight again -> stays pending + held (un-re-pickable).
        assert "A" in agg._sim_pending_commit
        assert "B" in agg._sim_pending_commit
        assert "A" in ch._selector.all_selected
        assert "A" in ch._selector.selected_ends["agg"]
