# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Tests for the `reselect_each_iteration` selection-granularity gate:
per-round (False) selects once and reuses the same trainer set for the
whole round; per-iteration (True, default) caches per version_key, re-invoking the
selector once per GENUINE (model_version, iteration)."""

import time

from flame.config import TrainerAvailState
from flame.mode.horizontal.syncfl.fwdllm_aggregator import (
    ROUND_CACHE_STUCK_TIMEOUT_S,
    TopAggregator,
)


class _FakeSelector:
    """Stand-in for RandomSelector exposing only `selected_ends`."""

    def __init__(self):
        self.selected_ends = set()


class _FakeAsyncSelector:
    """Stand-in for FedBuffSelector/AsyncOortSelector: `selected_ends` is a
    dict[requester, set], not a bare set (see `_FakeSelector` above) --
    real_async selectors' own convention, missed by the bare-set fake for a
    long time since `TestAsyncReselectGate` never drove a cache-hit against
    it (the bug: `_rearm_recv_eligibility` crashed `'set' object is not
    subscriptable` the first time a round-level async baseline actually ran,
    felix_round's 2026-07-23 smoke test)."""

    def __init__(self, requester="agg"):
        self.requester = requester
        self.selected_ends = {requester: set()}


class _FakeChannel:
    """Records each `ends()` call and returns the next canned selection.

    Also models end departure: `_removed` mimics a fully disconnected end
    (as `channel.remove()` would leave it -- `has()` returns False);
    `_unavail` mimics an end that explicitly reported `UN_AVL` but is
    still connected (as `channel.update_state()` would leave it -- `has()`
    still True, but its avl-state property reads `UN_AVL`).
    """

    def __init__(self, selections, c=None):
        self._selections = list(selections)
        self.calls = 0
        self._selector = _FakeSelector()
        self._removed = set()
        self._unavail = set()
        # Concurrency target: what sizes the pinned per-round cohort (NOT
        # agg_goal -- see _round_cohort_target). `properties` also carries
        # DynamicKCController's `dynamic_c` override, which takes precedence.
        self._c = c
        self.properties = {}

    def get_c(self):
        return self._c

    def ends(
        self,
        state,
        task_to_perform,
        agg_version_key=None,
        data_id=None,
        trainer_version_keys=None,
    ):
        self.calls += 1
        return self._selections[min(self.calls - 1, len(self._selections) - 1)]

    def has(self, end_id):
        return end_id not in self._removed

    def get_end_property(self, end_id, key):
        if end_id in self._unavail:
            return TrainerAvailState.UN_AVL
        return None


class _FakeAggregator:
    """Minimal stand-in exposing only the state
    `_select_ends_respecting_reselect_gate` touches."""

    def __init__(self, reselect_each_iteration=None, cadence=None, vclock_now=None):
        if cadence is None:
            cadence = "iteration" if reselect_each_iteration else "round"
        # None = real mode, mirroring TopAggregator.vclock_now's contract.
        self.vclock_now = vclock_now
        self._reselect_cadence = cadence
        self._reselect_each_iteration = cadence == "iteration"
        self._round_selected_ends = None
        self._pinned_cohort_key = None
        self._round_cache_activity_ts = {}
        self._round = 0
        self._model_version = 0
        self.iteration_per_data_id = 0
        self.data_id = 0
        self._reselect_true_cache_key = None
        self._reselect_true_cache_ends = None
        self._trainer_state_dict = {}
        self._curr_agg_version = None

    @property
    def version_key(self):
        return (self._model_version, self.iteration_per_data_id)

    select = TopAggregator._select_ends_respecting_reselect_gate
    select_async = TopAggregator._select_ends_for_async_respecting_reselect_gate
    _rearm_recv_eligibility = staticmethod(TopAggregator._rearm_recv_eligibility)
    _exclude_pending_commit = staticmethod(TopAggregator._exclude_pending_commit)
    _cap_dispatch_to_concurrency = staticmethod(
        TopAggregator._cap_dispatch_to_concurrency
    )
    _prune_departed_from_round_cache = (
        TopAggregator._prune_departed_from_round_cache
    )
    _round_cache_clock_now = TopAggregator._round_cache_clock_now
    _round_cohort_target = TopAggregator._round_cohort_target
    _trim_round_cohort = TopAggregator._trim_round_cohort
    _cohort_cache_key = TopAggregator._cohort_cache_key
    _invalidate_cohort_cache_if_stale = (
        TopAggregator._invalidate_cohort_cache_if_stale
    )


def _drive_two_databins_two_iterations(agg, channel):
    """2 databins x 2 iterations each, within one round -- each iteration
    bumps `iteration_per_data_id`, each databin bumps `_model_version`, so
    every call carries a distinct version_key."""
    for _databin in range(2):
        for _iteration in range(2):
            agg.select(channel, "train")
            agg.iteration_per_data_id += 1
        agg._model_version += 1
        agg.iteration_per_data_id = 0


class TestReselectGate:
    def test_per_round_selects_once_then_again_after_rollover(self):
        agg = _FakeAggregator(reselect_each_iteration=False)
        channel = _FakeChannel(selections=[["t1", "t2"]])

        _drive_two_databins_two_iterations(agg, channel)
        assert channel.calls == 1
        assert agg.select(channel, "train") == ["t1", "t2"]
        assert channel.calls == 1

        # Round rollover: advance self._round, as
        # _process_aggregation_goal_met does at the databin-wraparound point.
        agg._round += 1
        channel._selections = [["t3", "t4"]]
        channel.calls = 0
        ends = agg.select(channel, "train")
        assert ends == ["t3", "t4"]
        assert channel.calls == 1

        # Still cached for the rest of the new round.
        agg.select(channel, "train")
        assert channel.calls == 1

    def test_per_iteration_selects_every_call(self):
        agg = _FakeAggregator(reselect_each_iteration=True)
        channel = _FakeChannel(selections=[["t1"], ["t2"], ["t3"], ["t4"]])

        _drive_two_databins_two_iterations(agg, channel)
        assert channel.calls == 4  # 4 genuinely distinct version_keys

    def test_per_round_does_not_cache_empty_selection(self):
        """An empty/None selection (no trainers joined yet) must not be
        cached as the round's selection -- retry on the next call."""
        agg = _FakeAggregator(reselect_each_iteration=False)
        channel = _FakeChannel(selections=[None, None, ["t1"]])

        assert agg.select(channel, "train") is None
        assert agg.select(channel, "train") is None
        assert agg.select(channel, "train") == ["t1"]
        assert channel.calls == 3

        # Now cached -- a 4th call must not invoke the selector again.
        assert agg.select(channel, "train") == ["t1"]
        assert channel.calls == 3

    def test_per_round_accumulates_partial_selections_until_c(self):
        """Must keep merging in newly-selected trainers until the cache
        reaches `c`, not freeze on the first partial result."""
        agg = _FakeAggregator(reselect_each_iteration=False)
        channel = _FakeChannel(selections=[["t1"], ["t2"], ["t3"]], c=3)

        assert agg.select(channel, "train") == ["t1"]
        assert agg.select(channel, "train") == ["t1", "t2"]
        assert agg.select(channel, "train") == ["t1", "t2", "t3"]
        assert channel.calls == 3

        # Cohort has reached c -- further calls must not re-query.
        assert agg.select(channel, "train") == ["t1", "t2", "t3"]
        assert channel.calls == 3

    def test_cache_hit_rearms_selector_recv_eligibility(self):
        """Regression test (2026-06-28 live-run hang): a cache hit must
        re-arm selected_ends, or it drains to empty after one pass and
        the receive side permanently stalls."""
        agg = _FakeAggregator(reselect_each_iteration=False)
        channel = _FakeChannel(selections=[["t1", "t2"]], c=2)

        ends = agg.select(channel, "train")
        assert ends == ["t1", "t2"]
        assert channel._selector.selected_ends == {"t1", "t2"}

        # Simulate cleanup_recvd_end draining both after iteration 0.
        channel._selector.selected_ends.clear()
        assert channel._selector.selected_ends == set()

        ends = agg.select(channel, "train")
        assert ends == ["t1", "t2"]
        assert channel.calls == 1  # still cache-hit, no re-query
        assert channel._selector.selected_ends == {"t1", "t2"}

    def test_accumulate_path_also_rearms_selector_recv_eligibility(self):
        """The accumulate-until-c path re-arms too, so behavior
        doesn't depend on which branch is taken."""
        agg = _FakeAggregator(reselect_each_iteration=False)
        channel = _FakeChannel(selections=[["t1"], ["t2"]], c=2)

        agg.select(channel, "train")
        assert channel._selector.selected_ends == {"t1"}

        agg.select(channel, "train")
        assert channel._selector.selected_ends == {"t1", "t2"}


class TestRoundCohortSizedByConcurrency:
    """The pinned cohort is sized by `c` (what the system promises to keep
    training), never by `agg_goal` (only the aggregation trigger). Both
    directions were broken and BOTH modes were broken identically, so parity
    reported a pass on a 10-of-30-trainer run (fedbuff_round, felix_round)."""

    def test_cohort_fills_to_c_not_agg_goal(self):
        """Under-fill: fedbuff_round ran c=30/agg_goal=10 and froze at 10,
        leaving 20 dispatch slots idle for the whole round."""
        agg = _FakeAggregator(reselect_each_iteration=False)
        agg._agg_goal = 2  # must NOT cap the cohort
        channel = _FakeChannel(
            selections=[["t1", "t2"], ["t3", "t4"], ["t5"]], c=5
        )

        agg.select(channel, "train")
        agg.select(channel, "train")
        assert agg.select(channel, "train") == ["t1", "t2", "t3", "t4", "t5"]
        assert channel.calls == 3

        # At c now -- frozen for the rest of the round.
        assert agg.select(channel, "train") == ["t1", "t2", "t3", "t4", "t5"]
        assert channel.calls == 3

    def test_cohort_trimmed_to_exactly_c_on_overshoot(self):
        """Over-fill: the reuse check runs before the merge, so one batch can
        blow past `c` in a single step. Untrimmed, final size fell out of
        arrival-burst timing -- felix_round landed 30 real vs 40 sim."""
        agg = _FakeAggregator(reselect_each_iteration=False)
        channel = _FakeChannel(selections=[["t1", "t2", "t3", "t4", "t5"]], c=3)

        assert agg.select(channel, "train") == ["t1", "t2", "t3"]
        # Surplus must not linger in the stuck-timeout bookkeeping either.
        assert set(agg._round_cache_activity_ts) == {"t1", "t2", "t3"}

    def test_dynamic_c_overrides_static_c(self):
        """DynamicKCController pushes `dynamic_c` as a channel property; it
        takes precedence over the selector's static `c` (matches async_oort)."""
        agg = _FakeAggregator(reselect_each_iteration=False)
        channel = _FakeChannel(selections=[["t1", "t2", "t3", "t4"]], c=4)
        channel.properties["dynamic_c"] = 2

        assert agg.select(channel, "train") == ["t1", "t2"]

    def test_no_discoverable_c_keeps_legacy_reuse(self):
        """A selector without `c` -> no target -> reuse whenever non-empty,
        the pre-`c` fallback, rather than re-querying forever."""
        agg = _FakeAggregator(reselect_each_iteration=False)
        channel = _FakeChannel(selections=[["t1"], ["t2"]], c=None)

        assert agg.select(channel, "train") == ["t1"]
        assert agg.select(channel, "train") == ["t1"]
        assert channel.calls == 1

    def test_async_gate_cohort_sized_by_c(self):
        """Async twin of the same two defects."""
        agg = _FakeAggregator(reselect_each_iteration=False)
        agg._agg_goal = 2
        channel = _FakeChannel(selections=[["t1", "t2", "t3", "t4"]], c=3)
        channel._selector = _FakeAsyncSelector()

        assert agg.select_async(channel, "train") == ["t1", "t2", "t3"]
        assert agg.select_async(channel, "train") == ["t1", "t2", "t3"]
        assert channel.calls == 1


class TestReselectCadence:
    """Three cadences, coarsest first: round (a full total_data_bins lap),
    data_bin (one completed databin), iteration. The deprecated boolean
    `reselect_each_iteration` maps True->iteration, False->round."""

    def test_data_bin_cadence_repins_per_model_version(self):
        agg = _FakeAggregator(cadence="data_bin")
        channel = _FakeChannel(selections=[["t1"], ["t2"], ["t3"]], c=1)

        # Iterations within one databin reuse the pinned cohort.
        assert agg.select(channel, "train") == ["t1"]
        agg.iteration_per_data_id += 1
        assert agg.select(channel, "train") == ["t1"]
        assert channel.calls == 1

        # Databin advance -> re-pin.
        agg._model_version += 1
        agg.iteration_per_data_id = 0
        assert agg.select(channel, "train") == ["t2"]
        assert channel.calls == 2

    def test_round_cadence_survives_model_version_advance(self):
        """The discriminator against data_bin: a databin advance must NOT
        re-pin at round cadence -- only a lap does."""
        agg = _FakeAggregator(cadence="round")
        channel = _FakeChannel(selections=[["t1"], ["t2"]], c=1)

        assert agg.select(channel, "train") == ["t1"]
        agg._model_version += 1
        assert agg.select(channel, "train") == ["t1"]
        assert channel.calls == 1

        agg._round += 1
        assert agg.select(channel, "train") == ["t2"]
        assert channel.calls == 2

    def test_data_bin_key_does_not_collide_across_laps(self):
        """`_model_version` is monotone; `data_id` wraps at total_data_bins
        and would collide lap-to-lap (§F-2)."""
        agg = _FakeAggregator(cadence="data_bin")
        channel = _FakeChannel(selections=[["t1"], ["t2"]], c=1)

        agg.data_id = 0
        assert agg.select(channel, "train") == ["t1"]
        # New lap: data_id wraps back to 0, but model_version keeps climbing.
        agg._round += 1
        agg.data_id = 0
        agg._model_version += 1
        assert agg.select(channel, "train") == ["t2"]
        assert channel.calls == 2


class TestResolveReselectCadence:
    """`reselect_cadence` wins; the boolean alias is the fallback."""

    class _HP:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    class _Agg:
        _resolve_reselect_cadence = TopAggregator._resolve_reselect_cadence
        RESELECT_CADENCES = TopAggregator.RESELECT_CADENCES

        def __init__(self, hp):
            self.config = type("C", (), {"hyperparameters": hp})()

    def _resolve(self, **kw):
        return self._Agg(self._HP(**kw))._resolve_reselect_cadence()

    def test_defaults_to_iteration(self):
        assert self._resolve() == "iteration"

    def test_boolean_alias_maps_to_iteration_and_round(self):
        assert self._resolve(reselect_each_iteration=True) == "iteration"
        assert self._resolve(reselect_each_iteration=False) == "round"

    def test_explicit_cadence_wins_over_alias(self):
        assert self._resolve(
            reselect_cadence="data_bin", reselect_each_iteration=True
        ) == "data_bin"

    def test_unknown_cadence_raises(self):
        import pytest

        with pytest.raises(ValueError, match="reselect_cadence"):
            self._resolve(reselect_cadence="every_other_tuesday")


class TestPerIterationVersionKeyCache:
    """reselect_each_iteration=True caches per version_key -- repeated calls
    within the same (model_version, iteration) reuse one channel.ends()
    result instead of re-invoking the selector every tick."""

    def test_repeat_calls_within_same_version_key_are_cached(self):
        agg = _FakeAggregator(reselect_each_iteration=True)
        channel = _FakeChannel(selections=[["t1", "t2"]])

        for _ in range(5):
            ends = agg.select(channel, "train")
            assert ends == ["t1", "t2"]
        assert channel.calls == 1  # 5 calls, same version_key -> 1 fetch

    def test_version_key_change_invalidates_cache(self):
        agg = _FakeAggregator(reselect_each_iteration=True)
        channel = _FakeChannel(selections=[["t1"], ["t2"]])

        assert agg.select(channel, "train") == ["t1"]
        assert agg.select(channel, "train") == ["t1"]  # cached
        assert channel.calls == 1

        agg.iteration_per_data_id += 1  # new version_key
        assert agg.select(channel, "train") == ["t2"]
        assert channel.calls == 2

    def test_empty_selection_is_not_cached(self):
        agg = _FakeAggregator(reselect_each_iteration=True)
        channel = _FakeChannel(selections=[None, ["t1"]])

        assert agg.select(channel, "train") is None
        assert agg.select(channel, "train") == ["t1"]  # retried, not stuck at None
        assert channel.calls == 2

        # now cached (non-empty) -- a further call within the same
        # version_key must not re-query.
        assert agg.select(channel, "train") == ["t1"]
        assert channel.calls == 2

    def test_cache_hit_rearms_selector_recv_eligibility(self):
        agg = _FakeAggregator(reselect_each_iteration=True)
        channel = _FakeChannel(selections=[["t1", "t2"]])

        agg.select(channel, "train")
        assert channel._selector.selected_ends == {"t1", "t2"}

        channel._selector.selected_ends.clear()
        ends = agg.select(channel, "train")
        assert ends == ["t1", "t2"]
        assert channel.calls == 1  # still cache-hit
        assert channel._selector.selected_ends == {"t1", "t2"}


class TestStaleCachePruning:
    """Regression tests for the `_round_selected_ends` stale-cache gap:
    a cached-but-departed end must be pruned so the cache-size check stops
    reporting "full" and the round can backfill the freed slot, instead of
    stalling forever on a contribution that can never arrive."""

    def test_disconnected_end_pruned_and_backfilled(self):
        agg = _FakeAggregator(reselect_each_iteration=False)
        channel = _FakeChannel(selections=[["t1", "t2"], ["t3"]], c=2)

        assert agg.select(channel, "train") == ["t1", "t2"]
        assert channel.calls == 1

        # t1 fully disconnects (channel.remove() -- has() now False).
        channel._removed.add("t1")

        ends = agg.select(channel, "train")
        assert ends == ["t2", "t3"]
        assert channel.calls == 2  # re-queried the selector to backfill

        # Now cached again at c -- no further re-query.
        assert agg.select(channel, "train") == ["t2", "t3"]
        assert channel.calls == 2

    def test_un_avl_end_pruned_and_backfilled(self):
        agg = _FakeAggregator(reselect_each_iteration=False)
        channel = _FakeChannel(selections=[["t1", "t2"], ["t3"]], c=2)

        assert agg.select(channel, "train") == ["t1", "t2"]
        assert channel.calls == 1

        # t1 stays connected but reports UN_AVL (channel.update_state()).
        channel._unavail.add("t1")

        ends = agg.select(channel, "train")
        assert ends == ["t2", "t3"]
        assert channel.calls == 2

        assert agg.select(channel, "train") == ["t2", "t3"]
        assert channel.calls == 2

    def test_still_present_end_not_pruned(self):
        """Control case: no departure means no pruning and no re-query."""
        agg = _FakeAggregator(reselect_each_iteration=False)
        channel = _FakeChannel(selections=[["t1", "t2"]], c=2)

        assert agg.select(channel, "train") == ["t1", "t2"]
        assert agg.select(channel, "train") == ["t1", "t2"]
        assert channel.calls == 1


class TestStuckCachePruning:
    """Regression tests for the round-cache scale gap found on a real n=100
    run (see examples/MIGRATING_TO_LAUNCHER.md §9): a cached end that's
    still formally connected (not disconnected, not UN_AVL) but has gone
    ROUND_CACHE_STUCK_TIMEOUT_S without a real accepted contribution --
    e.g. one that never finished receiving its initial weights -- must
    also be pruned/backfilled, not just an explicitly departed one
    (TestStaleCachePruning above)."""

    def test_stuck_end_pruned_and_backfilled(self):
        agg = _FakeAggregator(reselect_each_iteration=False)
        channel = _FakeChannel(selections=[["t1", "t2"], ["t3"]], c=2)

        assert agg.select(channel, "train") == ["t1", "t2"]
        assert channel.calls == 1

        # t1 is still connected and AVL_TRAIN (neither departure check in
        # TestStaleCachePruning would catch it), but has gone well past the
        # stuck-timeout without a real accepted contribution.
        agg._round_cache_activity_ts["t1"] = time.time() - (ROUND_CACHE_STUCK_TIMEOUT_S + 10)

        ends = agg.select(channel, "train")
        assert ends == ["t2", "t3"]
        assert channel.calls == 2  # re-queried the selector to backfill
        assert "t1" not in agg._round_cache_activity_ts  # cleaned up on prune

        assert agg.select(channel, "train") == ["t2", "t3"]
        assert channel.calls == 2

    def test_recently_active_end_not_pruned(self):
        """An end within the timeout window (even if not the most recent to
        contribute) must not be pruned -- only genuinely stuck members
        should churn the cache."""
        agg = _FakeAggregator(reselect_each_iteration=False)
        channel = _FakeChannel(selections=[["t1", "t2"]], c=2)

        assert agg.select(channel, "train") == ["t1", "t2"]
        agg._round_cache_activity_ts["t1"] = time.time() - (ROUND_CACHE_STUCK_TIMEOUT_S - 30)

        assert agg.select(channel, "train") == ["t1", "t2"]
        assert channel.calls == 1  # no re-query -- nothing pruned

    def test_freshly_cached_end_not_immediately_pruned(self):
        """Control: an end that just entered the cache (activity_ts == now,
        set by the accumulate path itself) must not be immediately treated
        as stuck on the very next call."""
        agg = _FakeAggregator(reselect_each_iteration=False)
        channel = _FakeChannel(selections=[["t1", "t2"]], c=2)

        assert agg.select(channel, "train") == ["t1", "t2"]
        assert set(agg._round_cache_activity_ts.keys()) == {"t1", "t2"}

        assert agg.select(channel, "train") == ["t1", "t2"]
        assert channel.calls == 1


class TestRoundCacheClockSource:
    """R-C: the stuck timeout must be stamped and checked on the clock the run
    actually advances on -- virtual in sim, wall in real. Measuring virtual
    progress against wall meant a sim run never reached the timeout (0 stuck
    evictions where real had 6), so sim refilled one round more than real."""

    def test_real_mode_uses_wall_clock(self):
        agg = _FakeAggregator(reselect_each_iteration=False)
        assert abs(agg._round_cache_clock_now() - time.time()) < 5

    def test_sim_mode_uses_virtual_clock(self):
        agg = _FakeAggregator(reselect_each_iteration=False, vclock_now=1234.0)
        assert agg._round_cache_clock_now() == 1234.0

    def test_sim_stuck_end_pruned_on_virtual_advance(self):
        agg = _FakeAggregator(reselect_each_iteration=False, vclock_now=1000.0)
        channel = _FakeChannel(selections=[["t1", "t2"], ["t3"]], c=2)

        assert agg.select(channel, "train") == ["t1", "t2"]
        assert agg._round_cache_activity_ts["t1"] == 1000.0  # stamped in vtime

        # Virtual clock races past the timeout while wall barely moves -- the
        # case the wall-clock version could never evict.
        agg.vclock_now = 1000.0 + ROUND_CACHE_STUCK_TIMEOUT_S + 10
        agg._round_cache_activity_ts["t2"] = agg.vclock_now  # t2 just contributed

        assert agg.select(channel, "train") == ["t2", "t3"]
        assert channel.calls == 2  # re-queried to backfill
        assert "t1" not in agg._round_cache_activity_ts

    def test_sim_end_within_virtual_window_not_pruned(self):
        agg = _FakeAggregator(reselect_each_iteration=False, vclock_now=1000.0)
        channel = _FakeChannel(selections=[["t1", "t2"]], c=2)

        assert agg.select(channel, "train") == ["t1", "t2"]
        agg.vclock_now = 1000.0 + ROUND_CACHE_STUCK_TIMEOUT_S - 30

        assert agg.select(channel, "train") == ["t1", "t2"]
        assert channel.calls == 1  # no re-query -- nothing pruned


class TestAsyncReselectGate:
    """`_select_ends_for_async_respecting_reselect_gate` -- the async
    counterpart wired into `_distribute_weights_async` (round vs +IT
    baselines: fedbuff_round/felix_round vs fedbuff_it_*/felix_it).
    True (default) must stay a pure passthrough (today's only async
    behavior, unchanged); False reuses the same round-cache/prune-departed
    machinery as the sync gate."""

    def test_reselect_true_always_reinvokes_no_caching(self):
        """Default (unchanged) async behavior: every call re-queries the
        selector, even within the same version_key -- no caching at all,
        unlike the sync True-branch's per-version_key cache."""
        agg = _FakeAggregator(reselect_each_iteration=True)
        channel = _FakeChannel(selections=[["t1"], ["t2"], ["t1", "t3"]])

        assert agg.select_async(channel, "train") == ["t1"]
        assert agg.select_async(channel, "train") == ["t2"]
        assert agg.select_async(channel, "train") == ["t1", "t3"]
        assert channel.calls == 3

    def test_reselect_false_pins_cohort_until_round_advances(self):
        agg = _FakeAggregator(reselect_each_iteration=False)
        channel = _FakeChannel(selections=[["t1"], ["t2"]], c=2)

        assert agg.select_async(channel, "train") == ["t1"]
        assert agg.select_async(channel, "train") == ["t1", "t2"]
        assert channel.calls == 2

        # Cohort reached c -- further calls must not re-query.
        assert agg.select_async(channel, "train") == ["t1", "t2"]
        assert channel.calls == 2

        # Round rollover -- cohort re-forms.
        agg._round += 1
        channel._selections = [["t3", "t4"]]
        channel.calls = 0
        assert agg.select_async(channel, "train") == ["t3", "t4"]
        assert channel.calls == 1
        assert agg.select_async(channel, "train") == ["t3", "t4"]
        assert channel.calls == 1

    def test_reselect_false_does_not_cache_empty_selection(self):
        agg = _FakeAggregator(reselect_each_iteration=False)
        channel = _FakeChannel(selections=[None, None, ["t1"]], c=1)

        assert agg.select_async(channel, "train") == []
        assert agg.select_async(channel, "train") == []
        assert agg.select_async(channel, "train") == ["t1"]
        assert channel.calls == 3

        assert agg.select_async(channel, "train") == ["t1"]
        assert channel.calls == 3

    def test_reselect_false_prunes_departed_and_backfills(self):
        agg = _FakeAggregator(reselect_each_iteration=False)
        channel = _FakeChannel(selections=[["t1", "t2"], ["t3"]], c=2)

        assert agg.select_async(channel, "train") == ["t1", "t2"]
        assert channel.calls == 1

        # t1 departs mid-round -- a round-level baseline swaps it out, it
        # doesn't deadlock the cohort.
        channel._removed.add("t1")

        ends = agg.select_async(channel, "train")
        assert ends == ["t2", "t3"]
        assert channel.calls == 2

        assert agg.select_async(channel, "train") == ["t2", "t3"]
        assert channel.calls == 2

    def test_cache_hit_rearms_dict_based_async_selector_without_crashing(self):
        """Regression for the felix_round smoke-test crash: a cache-hit call
        (cohort already at c) must rearm recv-eligibility on a REAL
        async selector's dict[requester, set] `selected_ends`, not assume the
        sync selector's bare-set convention `_rearm_recv_eligibility` was
        originally written for."""
        agg = _FakeAggregator(reselect_each_iteration=False)
        channel = _FakeChannel(selections=[["t1", "t2"]], c=2)
        channel._selector = _FakeAsyncSelector()

        # Accumulate call: builds the round cache but (like the real
        # channel.ends() -> selector.select() path on a genuine miss) doesn't
        # touch selected_ends -- only a cache HIT rearms it.
        assert agg.select_async(channel, "train") == ["t1", "t2"]

        # Cache hit: cohort already at c -- must not raise, and must
        # keep selected_ends a dict (this call crashed with TypeError before
        # the fix: `_rearm_recv_eligibility` reassigned the whole attribute
        # to a bare set, so the NEXT selector.select() call -- reading
        # selected_ends[requester] -- blew up).
        assert agg.select_async(channel, "train") == ["t1", "t2"]
        assert isinstance(channel._selector.selected_ends, dict)
        assert channel._selector.selected_ends["agg"] == {"t1", "t2"}

    def test_cache_hit_excludes_pending_commit_ends(self):
        """Regression for the round-cadence staleness bug (2026-07-27): a
        cache-hit dispatch must drop ends still awaiting commit -- the guard
        this branch otherwise skips, since it never calls select()."""
        agg = _FakeAggregator(reselect_each_iteration=False)
        channel = _FakeChannel(selections=[["t1", "t2"]], c=2)
        channel._selector = _FakeAsyncSelector()

        assert agg.select_async(channel, "train") == ["t1", "t2"]

        # t1 hasn't committed yet -- must be withheld from dispatch.
        channel._selector._agg_pending_commit_ref = {"t1"}
        ends = agg.select_async(channel, "train")
        assert ends == ["t2"]
        assert channel.calls == 1  # still a cache hit, no re-query

        # RECV stays armed for both -- t1's eventual return still gets processed.
        assert channel._selector.selected_ends["agg"] == {"t1", "t2"}

        # Once t1 commits (leaves pending), it's dispatchable again.
        channel._selector._agg_pending_commit_ref = set()
        assert agg.select_async(channel, "train") == ["t1", "t2"]

    def test_accumulate_path_also_excludes_pending_commit_ends(self):
        """The ACCUMULATE branch never had the guard the cache-hit branch got in
        2026-07-27 -- so at a round boundary, where the cohort is re-drawn, it
        re-dispatched ends that already had a live dispatch (fedbuff_round sim:
        553 same-end concurrent dispatches, §D-20)."""
        agg = _FakeAggregator(reselect_each_iteration=False)
        channel = _FakeChannel(selections=[["t1"], ["t1", "t2"]], c=2)
        channel._selector = _FakeAsyncSelector()

        assert agg.select_async(channel, "train") == ["t1"]   # cohort 1/2
        # t1 is in flight; the accumulate call that fills the cohort to c must
        # not hand t1 back for a second concurrent dispatch.
        channel._selector._agg_pending_commit_ref = {"t1"}
        assert agg.select_async(channel, "train") == ["t2"]
        assert agg._round_selected_ends == ["t1", "t2"]        # roster still c


class TestConcurrencyBackfillAtRoundBoundary:
    """Operator call 07-30: a round boundary BACKFILLS -- the new round's roster
    is `c`, but only `c - still_in_flight` may be dispatched now, the rest as
    stragglers commit. Sizing the new cohort to `c` outright ran c+stragglers at
    once (fedbuff_round sim peaked at 35 vs c=30, §D-20). Real already behaves
    this way; this makes sim keep the same contract."""

    def test_new_cohort_is_capped_by_still_in_flight_ends(self):
        agg = _FakeAggregator(reselect_each_iteration=False)
        channel = _FakeChannel(selections=[["a", "b", "c", "d"]], c=4)
        channel._selector = _FakeAsyncSelector()
        # 3 stragglers from the previous round are still in flight against c=4,
        # so exactly one of the fresh cohort may go out now.
        channel._selector._agg_pending_commit_ref = {"x", "y", "z"}
        ends = agg.select_async(channel, "train")
        assert len(ends) == 1, ends
        assert agg._round_selected_ends == ["a", "b", "c", "d"]  # roster intact

    def test_remainder_dispatches_as_stragglers_commit(self):
        agg = _FakeAggregator(reselect_each_iteration=False)
        channel = _FakeChannel(selections=[["a", "b", "c", "d"]], c=4)
        channel._selector = _FakeAsyncSelector()
        channel._selector._agg_pending_commit_ref = {"x", "y", "z"}
        assert len(agg.select_async(channel, "train")) == 1
        # Two stragglers commit -> two more slots free -> backfill.
        channel._selector._agg_pending_commit_ref = {"z"}
        assert agg.select_async(channel, "train") == ["a", "b", "c"]
        channel._selector._agg_pending_commit_ref = set()
        assert agg.select_async(channel, "train") == ["a", "b", "c", "d"]

    def test_no_stragglers_means_no_cap(self):
        """The common case must be byte-identical: nothing in flight -> the whole
        cohort dispatches, exactly as before."""
        agg = _FakeAggregator(reselect_each_iteration=False)
        channel = _FakeChannel(selections=[["a", "b", "c", "d"]], c=4)
        channel._selector = _FakeAsyncSelector()
        assert agg.select_async(channel, "train") == ["a", "b", "c", "d"]

    def test_cap_is_a_noop_without_a_discoverable_c(self):
        agg = _FakeAggregator(reselect_each_iteration=False)
        channel = _FakeChannel(selections=[["a", "b"]], c=None)
        channel._selector = _FakeAsyncSelector()
        channel._selector._agg_pending_commit_ref = {"x", "y", "z"}
        assert agg.select_async(channel, "train") == ["a", "b"]
