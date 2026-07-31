# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""§F-23 has two clauses and sim was implementing them with ONE set.

    "Commit frees the compute slot immediately, but a version_key re-pick guard
     keeps the trainer un-pickable for the SAME (model_version, iteration)."

Those are different roles over different domains:

  CAPACITY  -- which ends occupy one of the `c` dispatch slots. A global
               resource. Drives `extra = c - len(selected_ends)` and
               `_cap_dispatch_to_concurrency`.
  IDENTITY  -- which ends must not be re-picked right now. Per-trainer. Drives
               `_eligible_candidates` / `_exclude_pending_commit`.

`_sim_pending_commit` served both, so §D-15's fix -- hold a committed trainer's
re-pick guard to the agg-goal boundary so it isn't re-tasked mid-cycle -- also
held its SLOT, denying it to every other trainer. Measured on fluxtune: real sits
at 30/30 slots for 80.8% of the run, sim for 8.6%; mean in-flight 29.50 vs 24.66,
against an 11.9% throughput residual. `slot_starvation` was 0 in both modes, so
sim was never short of candidates -- it believed it had no free slots.

`_slot_holders()` is now the single authoritative CAPACITY answer; the identity
set is unchanged. Real is byte-identical (its `_PendingCommitUnion` drops an end
from both halves at commit, so the two roles already coincided).
"""

from types import SimpleNamespace

import pytest

from flame.mode.horizontal.syncfl.fwdllm_aggregator import TopAggregator


class _Agg:
    """Only the state `_slot_holders` / `_cap_dispatch_to_concurrency` read."""

    def __init__(self, simulated=True, pending=(), committed=(),
                 commit_frees_slot=True):
        self.simulated = simulated
        self._sim_pending_commit = set(pending)
        self._sim_committed = set(committed)
        self._sim_commit_frees_slot = commit_frees_slot
        self._real_pending_commit = set(pending)

    _sim_slot_holder_set = TopAggregator._sim_slot_holder_set
    _slot_holders = TopAggregator._slot_holders
    _outstanding_dispatch_count = TopAggregator._outstanding_dispatch_count


class _Sel:
    def __init__(self, pending=None, slots=None):
        if pending is not None:
            self._agg_pending_commit_ref = pending
        if slots is not None:
            self._agg_slot_holders_ref = slots


class _Chan:
    def __init__(self, selector, c=30, dynamic_c=None):
        self._selector = selector
        self.properties = {"dynamic_c": dynamic_c}
        self._c = c

    def get_c(self):
        return self._c


class TestSlotHoldersIsCapacityNotIdentity:
    def test_committed_end_does_not_hold_a_slot(self):
        agg = _Agg(pending={"a", "b", "c"}, committed={"c"})
        assert agg._slot_holders() == {"a", "b"}
        # ...but it is still in the identity set, so it can't be re-picked.
        assert "c" in agg._sim_pending_commit

    def test_dispatched_uncommitted_end_holds_a_slot(self):
        agg = _Agg(pending={"a", "b"}, committed=set())
        assert agg._slot_holders() == {"a", "b"}

    def test_outstanding_count_agrees_with_slot_holders(self):
        # The tripwire and the dispatch cap must never disagree on "in flight".
        agg = _Agg(pending={"a", "b", "c", "d"}, committed={"c", "d"})
        assert agg._outstanding_dispatch_count() == len(agg._slot_holders()) == 2

    def test_real_mode_is_unchanged(self):
        # Real's union already drops an end at commit -> the roles coincide.
        agg = _Agg(simulated=False, pending={"a", "b", "c"}, committed={"c"})
        assert agg._slot_holders() == {"a", "b", "c"}

    def test_kill_switch_restores_the_conflated_behaviour(self):
        agg = _Agg(pending={"a", "b", "c"}, committed={"c"},
                   commit_frees_slot=False)
        assert agg._slot_holders() == {"a", "b", "c"}

    def test_missing_state_is_not_an_error(self):
        agg = _Agg(pending={"a"})
        del agg._sim_pending_commit
        agg._sim_pending_commit = None
        assert agg._slot_holders() == set()


class TestDispatchCapUsesCapacityNotIdentity:
    """The regression that cost fluxtune ~5 of 30 slots: the cap counted
    committed-but-guarded ends, so a free slot was denied to everyone else."""

    def test_cap_ignores_committed_ends(self):
        # 30 slots, 28 genuinely in flight, 5 committed-but-guarded. 2 free.
        inflight = {f"t{i}" for i in range(28)}
        committed = {f"c{i}" for i in range(5)}
        chan = _Chan(_Sel(pending=inflight | committed, slots=inflight), c=30)
        out = TopAggregator._cap_dispatch_to_concurrency(chan, ["x", "y", "z"])
        assert out == ["x", "y"]            # 30 - 28, not 30 - 33 -> 0

    def test_without_the_split_the_cap_would_dispatch_nothing(self):
        # Same state, identity set only -- what the old code saw.
        inflight = {f"t{i}" for i in range(28)}
        committed = {f"c{i}" for i in range(5)}
        chan = _Chan(_Sel(pending=inflight | committed), c=30)
        assert TopAggregator._cap_dispatch_to_concurrency(chan, ["x", "y", "z"]) == []

    def test_falls_back_to_identity_ref_when_no_slot_ref(self):
        # Real / any baseline that doesn't publish a slot ref: byte-identical.
        chan = _Chan(_Sel(pending={"a", "b"}), c=3)
        assert TopAggregator._cap_dispatch_to_concurrency(chan, ["x", "y"]) == ["x"]

    def test_never_exceeds_c(self):
        inflight = {f"t{i}" for i in range(30)}
        chan = _Chan(_Sel(pending=inflight, slots=inflight), c=30)
        assert TopAggregator._cap_dispatch_to_concurrency(chan, ["x"]) == []

    def test_no_selector_is_a_noop(self):
        chan = _Chan(None, c=30)
        assert TopAggregator._cap_dispatch_to_concurrency(chan, ["x", "y"]) == ["x", "y"]

    def test_unknown_c_is_a_noop(self):
        chan = _Chan(_Sel(pending=set(), slots=set()), c=None)
        assert TopAggregator._cap_dispatch_to_concurrency(chan, ["x"]) == ["x"]

    def test_dynamic_c_wins_over_static_c(self):
        inflight = {"a"}
        chan = _Chan(_Sel(pending=inflight, slots=inflight), c=30, dynamic_c=2)
        assert TopAggregator._cap_dispatch_to_concurrency(chan, ["x", "y"]) == ["x"]


class TestIdentityGuardStillHolds:
    """The slot must free WITHOUT re-opening §D-15: a trainer that just
    committed still must not be re-picked before the version_key advances."""

    def test_committed_end_is_still_excluded_from_candidates(self):
        pending = {"a", "committed"}
        chan = _Chan(_Sel(pending=pending, slots={"a"}), c=30)
        kept = TopAggregator._exclude_pending_commit(chan, ["a", "committed", "fresh"])
        assert kept == ["fresh"]

    def test_the_two_refs_are_allowed_to_differ(self):
        # This is the whole point: identity ⊇ capacity, never the reverse.
        agg = _Agg(pending={"a", "b", "c"}, committed={"c"})
        assert agg._slot_holders() < set(agg._sim_pending_commit)
