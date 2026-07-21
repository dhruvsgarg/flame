# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Real's `_agg_pending_commit_ref` binding (simulate_fwdllm.md §B, 07-21).

`_release_end_on_return`'s buffered=True path frees a trainer's channel slot
as soon as its grad is buffered, but real had no selector-level guard to
match sim's `_sim_pending_commit` -- the freed trainer was re-pickable before
its contribution committed, wasting dispatches. Fix binds
`_agg_pending_commit_ref` to `_per_agg_trainer_list` (real-only; sim's own
binding is untouched) -- reuses the existing dedup-guard list, no new
structure.

Companion fix (same session): sim's own `_sim_pending_commit` guard was only
reconciled by `_sim_hold_busy_slots` on OTHER commit/boundary events, not on
this trainer's own receipt -- `TestSimPendingCommitSyncOnReceipt` covers the
synchronous `.add(end)` closing that gap (cohort_sequence root cause).
"""

from flame.mode.horizontal.syncfl.fwdllm_aggregator import (
    TopAggregator,
    _OrderedContributorList,
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
    _release_end_on_return = TopAggregator._release_end_on_return

    def __init__(self, simulated: bool, is_async: bool = True):
        self.simulated = simulated
        self.is_async = is_async
        self._per_agg_trainer_list = _OrderedContributorList()
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
        # Bound to the SAME live object (never a copy) -- membership changes
        # after this call must still be visible to the selector.
        assert channel._selector._agg_pending_commit_ref is agg._per_agg_trainer_list

    def test_sim_mode_does_not_touch_the_ref_here(self):
        """Sim's eligibility gate is bound separately (`_sim_hold_busy_slots`
        -> `_sim_pending_commit`, a strict superset that also covers trainers
        still computing, not just buffered). This code path must leave it
        alone for sim, or it would clobber that richer set with the
        narrower `_per_agg_trainer_list`."""
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
        assert ref is agg._per_agg_trainer_list


class TestBoundRefSurvivesInPlaceMutation:
    """The selector holds a live reference (never rebound) -- commit-boundary
    reset and canonical-order reorder must mutate `_per_agg_trainer_list` in
    place, not replace it with a new list object, or the selector's captured
    reference goes stale and silently stops excluding anyone."""

    def test_commit_reset_clears_in_place(self):
        agg = _FakeAggregator(simulated=False)
        channel = _FakeChannel()
        agg.process(channel, _grad_msg(), "t1", timestamp=0)
        bound_ref = channel._selector._agg_pending_commit_ref

        # Mirrors _process_aggregation_goal_met's commit-boundary reset.
        agg._per_agg_trainer_list.clear()

        assert bound_ref is agg._per_agg_trainer_list
        assert list(bound_ref) == []

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
        assert bound_ref is agg._per_agg_trainer_list


class TestSimPendingCommitSyncOnReceipt:
    """Sim's `_sim_pending_commit` must exclude a trainer the INSTANT its
    grad is buffered, not only after `_sim_hold_busy_slots` next runs (which
    fires on OTHER commit/boundary events) -- else it stays wrongly
    re-pickable for however many calls until that next event, which is what
    grew cohort_sequence's real/sim pool divergence unbounded."""

    def test_end_added_to_sim_pending_commit_immediately(self):
        agg = _FakeAggregator(simulated=True)
        channel = _FakeChannel()

        agg.process(channel, _grad_msg(), "t1", timestamp=0)

        assert "t1" in agg._sim_pending_commit

    def test_real_mode_does_not_touch_sim_pending_commit(self):
        agg = _FakeAggregator(simulated=False)
        channel = _FakeChannel()

        agg.process(channel, _grad_msg(), "t1", timestamp=0)

        assert agg._sim_pending_commit == set()
