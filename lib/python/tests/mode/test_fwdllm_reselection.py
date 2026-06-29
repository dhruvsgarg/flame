# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Tests for the `reselect_each_iteration` selection-granularity gate (P4
of the fwdllm launcher migration's Phase 7): per-round (False) selects
once and reuses the same trainer set for the whole round; per-iteration
(True, default) re-invokes the selector every call."""

from flame.mode.horizontal.syncfl.fwdllm_aggregator import TopAggregator


class _FakeSelector:
    """Stand-in for RandomSelector exposing only `selected_ends`."""

    def __init__(self):
        self.selected_ends = set()


class _FakeChannel:
    """Records each `ends()` call and returns the next canned selection."""

    def __init__(self, selections):
        self._selections = list(selections)
        self.calls = 0
        self._selector = _FakeSelector()

    def ends(self, state, task_to_perform):
        self.calls += 1
        return self._selections[min(self.calls - 1, len(self._selections) - 1)]


class _FakeAggregator:
    """Minimal stand-in exposing only the state
    `_select_ends_respecting_reselect_gate` touches."""

    def __init__(self, reselect_each_iteration, agg_goal=None):
        self._reselect_each_iteration = reselect_each_iteration
        self._round_selected_ends = None
        self._round_selected_ends_round = None
        self._round = 0
        if agg_goal is not None:
            self._agg_goal = agg_goal

    select = TopAggregator._select_ends_respecting_reselect_gate
    _rearm_recv_eligibility = staticmethod(TopAggregator._rearm_recv_eligibility)


def _drive_two_databins_two_iterations(agg, channel):
    """2 databins x 2 iterations each, within one round."""
    for _databin in range(2):
        for _iteration in range(2):
            agg.select(channel, "train")


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
        assert channel.calls == 4

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

    def test_per_round_accumulates_partial_selections_until_agg_goal(self):
        """Must keep merging in newly-selected trainers until the cache
        reaches `_agg_goal`, not freeze on the first partial result."""
        agg = _FakeAggregator(reselect_each_iteration=False, agg_goal=3)
        channel = _FakeChannel(selections=[["t1"], ["t2"], ["t3"]])

        assert agg.select(channel, "train") == ["t1"]
        assert agg.select(channel, "train") == ["t1", "t2"]
        assert agg.select(channel, "train") == ["t1", "t2", "t3"]
        assert channel.calls == 3

        # Cache has reached agg_goal -- further calls must not re-query.
        assert agg.select(channel, "train") == ["t1", "t2", "t3"]
        assert channel.calls == 3

    def test_cache_hit_rearms_selector_recv_eligibility(self):
        """Regression test (2026-06-28 live-run hang): a cache hit must
        re-arm selected_ends, or it drains to empty after one pass and
        the receive side permanently stalls."""
        agg = _FakeAggregator(reselect_each_iteration=False, agg_goal=2)
        channel = _FakeChannel(selections=[["t1", "t2"]])

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
        """The accumulate-until-agg_goal path re-arms too, so behavior
        doesn't depend on which branch is taken."""
        agg = _FakeAggregator(reselect_each_iteration=False, agg_goal=2)
        channel = _FakeChannel(selections=[["t1"], ["t2"]])

        agg.select(channel, "train")
        assert channel._selector.selected_ends == {"t1"}

        agg.select(channel, "train")
        assert channel._selector.selected_ends == {"t1", "t2"}
