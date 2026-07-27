# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Real's `_agg_pending_commit_ref` binding.

`_release_end_on_return`'s buffered=True path frees a trainer's channel slot
as soon as its grad is buffered, but real had no selector-level guard
matching sim's `_sim_pending_commit` -- the freed trainer was re-pickable
before its contribution committed, wasting dispatches. Fix binds
`_agg_pending_commit_ref` to `_per_agg_trainer_list` (real-only).

Companion fix: sim's `_sim_pending_commit` guard was only reconciled by
`_sim_hold_busy_slots` on other commit/boundary events, not this trainer's
own receipt -- `TestSimPendingCommitSyncOnReceipt` covers the synchronous
`.add(end)` fix that closes that gap.

Second companion fix (2026-07-27, simulate_fwdllm.md U3): `_per_agg_trainer_list`
alone only covers returned-but-uncommitted ends -- a still-training end that
never returned was invisible to it, so a round-cadence dispatch loop kept
re-dispatching to it every time the version_key advanced from OTHER ends'
progress, flooding its real channel queue with an unbounded backlog it could
never drain (unbounded real `staleness`; sim's `_sim_pending_commit` already
covered this half, capping sim's staleness at 1). `_agg_pending_commit_ref`
now binds to `_real_pending_commit`, the union of `_trainer_inflight_dispatch_
version` (dispatched, not yet returned) and `_per_agg_trainer_list` (returned,
not yet committed) -- `TestRealPendingCommitCoversInflightDispatch` covers it.
"""

from flame.mode.horizontal.syncfl.fwdllm_aggregator import (
    TopAggregator,
    _OrderedContributorList,
    _PendingCommitUnion,
)
from flame.mode.message import MessageType


class TestOrderedContributorList:
    """List order matters (canonical-commit reordering needs index positions,
    zipped 1:1 with `_pending_cohort_contribs`), but the selector's
    `_agg_pending_commit_ref` hook is duck-typed against a set (sim binds
    `_sim_pending_commit`) and calls `.discard()` on it."""

    def test_discard_removes_present_item(self):
        lst = _OrderedContributorList(["a", "b", "c"])
        lst.discard("b")
        assert list(lst) == ["a", "c"]

    def test_discard_missing_item_is_noop_not_an_error(self):
        lst = _OrderedContributorList(["a"])
        lst.discard("nonexistent")  # must not raise (unlike list.remove)
        assert list(lst) == ["a"]

    def test_still_behaves_like_a_plain_list(self):
        lst = _OrderedContributorList()
        lst.append("a")
        lst.append("b")
        assert lst == ["a", "b"]
        assert "a" in lst
        assert lst[0] == "a"


class _FakeSelector:
    def __init__(self):
        self.ordered_updates_recv_ends = []


class _FakeChannel:
    def __init__(self):
        self._selector = _FakeSelector()
        self._props = {}
        self.cleaned_up = []
        self.provided_cleaned_up = []

    def set_end_property(self, end, key, value):
        self._props[(end, key)] = value

    def get_end_property(self, end, key):
        return self._props.get((end, key))

    def cleanup_recvd_end(self, end):
        self.cleaned_up.append(end)

    def cleanup_provided_ends(self, end):
        self.provided_cleaned_up.append(end)


class _FakeAggregator:
    """Enough state for `_process_single_trainer_message` to run to
    completion on a single fresh GRADIENTS contribution."""

    process = TopAggregator._process_single_trainer_message
    _round_cache_clock_now = TopAggregator._round_cache_clock_now
    _release_end_on_return = TopAggregator._release_end_on_return

    def __init__(self, simulated: bool, is_async: bool = True):
        self.simulated = simulated
        self.is_async = is_async
        self._per_agg_trainer_list = _OrderedContributorList()
        self._trainer_inflight_dispatch_version = {}
        self._real_pending_commit = _PendingCommitUnion(
            self._trainer_inflight_dispatch_version, self._per_agg_trainer_list
        )
        self._round = 1
        self.data_id = 0
        self.iteration_per_data_id = 0
        self._model_version = 0
        self._agg_goal_cnt = 0
        self._round_cache_activity_ts = {}
        self._updates_in_queue = 0
        self._trainer_state_dict = {}
        self._updates_received = {}
        self.grad_pool = []
        self._trainer_last_model_version = {}
        self._inflight_residence = False
        self._sim_pending_commit = set()


def _grad_msg(model_version=0):
    return {
        MessageType.MODEL_VERSION: model_version,
        MessageType.GRADIENTS: {},
        MessageType.GRADIENTS_FOR_VAR_CHECK: None,
        MessageType.STAT_UTILITY: 1.0,
    }


class TestRealBindsPendingCommitRef:
    def test_real_mode_binds_ref_to_per_agg_trainer_list(self):
        agg = _FakeAggregator(simulated=False)
        channel = _FakeChannel()

        result = agg.process(channel, _grad_msg(), "t1", timestamp=0)

        assert result is True
        assert "t1" in agg._per_agg_trainer_list
        # Bound to the SAME live union object (never a copy) -- membership
        # changes after this call must still be visible to the selector.
        assert channel._selector._agg_pending_commit_ref is agg._real_pending_commit
        assert "t1" in channel._selector._agg_pending_commit_ref

    def test_sim_mode_does_not_touch_the_ref_here(self):
        """Sim's eligibility gate is bound separately (`_sim_hold_busy_slots`
        -> `_sim_pending_commit`, a superset also covering still-computing
        trainers). This path must leave it alone for sim, or it would clobber
        that richer set with the narrower `_per_agg_trainer_list`."""
        agg = _FakeAggregator(simulated=True)
        channel = _FakeChannel()

        agg.process(channel, _grad_msg(), "t1", timestamp=0)

        assert not hasattr(channel._selector, "_agg_pending_commit_ref")

    def test_second_real_trainer_extends_the_same_bound_list(self):
        agg = _FakeAggregator(simulated=False)
        channel = _FakeChannel()

        agg.process(channel, _grad_msg(), "t1", timestamp=0)
        agg.process(channel, _grad_msg(), "t2", timestamp=0)

        ref = channel._selector._agg_pending_commit_ref
        assert list(ref) == ["t1", "t2"]
        assert ref is agg._real_pending_commit


class TestBoundRefSurvivesInPlaceMutation:
    """The selector holds a live reference (never rebound) -- commit-boundary
    reset and canonical-order reorder must mutate `_per_agg_trainer_list` in
    place, not replace it with a new list object, or the union view's
    captured reference goes stale and silently stops excluding anyone."""

    def test_commit_reset_clears_in_place(self):
        agg = _FakeAggregator(simulated=False)
        channel = _FakeChannel()
        agg.process(channel, _grad_msg(), "t1", timestamp=0)
        bound_ref = channel._selector._agg_pending_commit_ref

        # Mirrors _process_aggregation_goal_met's commit-boundary reset.
        agg._per_agg_trainer_list.clear()

        assert bound_ref is agg._real_pending_commit
        assert list(bound_ref) == []
        assert "t1" not in bound_ref

    def test_canonicalize_reorder_mutates_in_place(self):
        agg = _FakeAggregator(simulated=False)
        channel = _FakeChannel()
        agg.process(channel, _grad_msg(), "t1", timestamp=0)
        agg.process(channel, _grad_msg(), "t2", timestamp=0)
        bound_ref = channel._selector._agg_pending_commit_ref

        agg._commit_key_by_end = {"t1": (5.0, "t1"), "t2": (1.0, "t2")}
        agg._pending_cohort_contribs = ["contrib_t1", "contrib_t2"]
        TopAggregator._canonicalize_cohort_commit_order(agg)

        # Reordered by (D, id): t2 (D=1.0) before t1 (D=5.0).
        assert list(agg._per_agg_trainer_list) == ["t2", "t1"]
        assert agg._pending_cohort_contribs == ["contrib_t2", "contrib_t1"]
        # Same object throughout -- the selector's reference is still valid.
        assert bound_ref is agg._real_pending_commit
        assert list(bound_ref) == ["t2", "t1"]


class TestSimCommitDoesNotRepinPendingCommit:
    """In sim, `_process_single_trainer_message` runs at COMMIT, not receipt:
    `_aggregate_grads_async` calls `_sim_recv_min_grad` (which already discarded
    the end from `_sim_pending_commit` once the vclock reached its sct) and THEN
    this. It must NOT re-add the end -- doing so re-pins every just-committed
    trainer forever, so `_sim_pending_commit` never drains, `selected_ends`
    stays full, and the sim deadlocks (simulate_fwdllm.md §F.1-23). The
    dispatch-time add is the pin; commit is the release."""

    def test_commit_does_not_readd_to_sim_pending_commit(self):
        agg = _FakeAggregator(simulated=True)
        channel = _FakeChannel()
        # State at commit: `_sim_recv_min_grad` has already discarded t1.
        agg._sim_pending_commit = set()

        agg.process(channel, _grad_msg(), "t1", timestamp=0)

        assert "t1" not in agg._sim_pending_commit

    def test_real_mode_does_not_touch_sim_pending_commit(self):
        agg = _FakeAggregator(simulated=False)
        channel = _FakeChannel()

        agg.process(channel, _grad_msg(), "t1", timestamp=0)

        assert agg._sim_pending_commit == set()


class TestPendingCommitUnion:
    """`_PendingCommitUnion` standalone -- the container itself, independent
    of the aggregator plumbing above."""

    def test_contains_checks_both_halves(self):
        inflight = {"t1": (0, 0)}
        returned = _OrderedContributorList(["t2"])
        union = _PendingCommitUnion(inflight, returned)

        assert "t1" in union
        assert "t2" in union
        assert "t3" not in union

    def test_falsy_when_both_empty(self):
        union = _PendingCommitUnion({}, _OrderedContributorList())
        assert not union
        assert (union or set()) == set()

    def test_truthy_when_either_half_populated(self):
        assert _PendingCommitUnion({"t1": (0, 0)}, _OrderedContributorList())
        assert _PendingCommitUnion({}, _OrderedContributorList(["t1"]))

    def test_iter_yields_both_halves_without_duplicates(self):
        inflight = {"t1": (0, 0), "t2": (0, 0)}
        returned = _OrderedContributorList(["t2", "t3"])
        union = _PendingCommitUnion(inflight, returned)

        # t2 present in both halves -- must appear once.
        assert sorted(union) == ["t1", "t2", "t3"]

    def test_discard_removes_from_both_halves(self):
        inflight = {"t1": (0, 0)}
        returned = _OrderedContributorList(["t1"])
        union = _PendingCommitUnion(inflight, returned)

        union.discard("t1")

        assert "t1" not in inflight
        assert "t1" not in returned
        assert "t1" not in union

    def test_discard_missing_item_is_noop(self):
        union = _PendingCommitUnion({}, _OrderedContributorList())
        union.discard("nonexistent")  # must not raise


class TestRealPendingCommitCoversInflightDispatch:
    """The gap this session's fix closes (simulate_fwdllm.md U3): a
    still-training end that hasn't returned anything yet was invisible to
    real's `_agg_pending_commit_ref` (bound only to `_per_agg_trainer_list`,
    which is return-populated) -- so a round-cadence dispatch loop kept
    re-dispatching to it every version_key advance, flooding its channel
    queue. `_trainer_inflight_dispatch_version` (dispatch-set, return-cleared)
    is now folded into the same ref via `_real_pending_commit`."""

    def test_dispatched_not_yet_returned_end_is_pending(self):
        agg = _FakeAggregator(simulated=False)
        # Mirrors the dispatch loop's bookkeeping (fwdllm_aggregator.py's
        # `_distribute_weights_async`/`_sync`) for an end that hasn't
        # returned anything this cycle.
        agg._trainer_inflight_dispatch_version["t1"] = (0, 0)

        assert "t1" in agg._real_pending_commit
        assert "t1" not in agg._per_agg_trainer_list  # confirms it's the NEW half

    def test_return_moves_end_from_inflight_to_returned_half(self):
        agg = _FakeAggregator(simulated=False)
        channel = _FakeChannel()
        agg._trainer_inflight_dispatch_version["t1"] = (0, 0)

        agg.process(channel, _grad_msg(), "t1", timestamp=0)

        # Cleared from the inflight half by _process_single_trainer_message...
        assert "t1" not in agg._trainer_inflight_dispatch_version
        # ...but still excluded overall -- now via the returned-pending half,
        # until commit clears _per_agg_trainer_list.
        assert "t1" in agg._per_agg_trainer_list
        assert "t1" in agg._real_pending_commit

    def test_never_returned_end_stays_excluded_across_multiple_dispatches(self):
        """The exact regression: repeated re-dispatch to a slow end (each one
        re-arming `_trainer_inflight_dispatch_version[end]` at a NEW
        version_key, as the round-cadence loop's `_already_served_current_
        instruction` guard re-serves on every version_key advance) must not
        make the end dispatchable again -- only its eventual RETURN, or
        commit, clears it."""
        agg = _FakeAggregator(simulated=False)

        # Three successive re-dispatches (global version_key advancing off
        # OTHER ends' progress) to a straggler that never responds.
        agg._trainer_inflight_dispatch_version["t1"] = (0, 0)
        assert "t1" in agg._real_pending_commit
        agg._trainer_inflight_dispatch_version["t1"] = (1, 0)
        assert "t1" in agg._real_pending_commit
        agg._trainer_inflight_dispatch_version["t1"] = (2, 0)
        assert "t1" in agg._real_pending_commit
