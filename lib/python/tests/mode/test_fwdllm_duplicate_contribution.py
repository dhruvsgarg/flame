# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""An end must contribute at most once per (round, data_id,
iteration_per_data_id) collection cycle -- _process_single_trainer_message
must reject a repeat contribution from an end already in
_per_agg_trainer_list, rather than double-counting it."""

from flame.mode.horizontal.syncfl.fwdllm_aggregator import TopAggregator


class _FakeChannel:
    def __init__(self):
        self.cleaned_up = []

    def cleanup_recvd_end(self, end):
        self.cleaned_up.append(end)


class _FakeAggregator:
    """Minimal stand-in exposing only the state the duplicate-contribution
    guard touches."""

    def __init__(self, already_contributed):
        self._per_agg_trainer_list = list(already_contributed)
        self._agg_goal_cnt = len(already_contributed)
        self._round = 1
        self.data_id = 0
        self.iteration_per_data_id = 0

    process = TopAggregator._process_single_trainer_message


class TestDuplicateContributionGuard:
    def test_rejects_repeat_contribution_in_same_cycle(self):
        agg = _FakeAggregator(already_contributed=["t1"])
        channel = _FakeChannel()

        result = agg.process(channel, {}, "t1", timestamp=0)

        assert result is False
        assert channel.cleaned_up == ["t1"]
        # Must not be double-counted.
        assert agg._agg_goal_cnt == 1
        assert agg._per_agg_trainer_list == ["t1"]

    def test_does_not_touch_first_time_contributors(self):
        agg = _FakeAggregator(already_contributed=["t1"])
        channel = _FakeChannel()

        # t2 hasn't contributed yet -- the duplicate guard must not fire
        # (an empty msg fails later validation instead, via the unrelated
        # "Invalid message" branch -- confirms we got past the guard).
        assert "t2" not in agg._per_agg_trainer_list
        result = agg.process(channel, {}, "t2", timestamp=0)

        assert result is False
        assert channel.cleaned_up == []
