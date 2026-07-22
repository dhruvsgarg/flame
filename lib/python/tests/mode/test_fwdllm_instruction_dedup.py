# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""One-instruction-per-version_key dedup for the sync distribute loop (§H).

The sync `distribute -> collect(1)` loop re-runs ~agg_goal times per iteration,
and each `_distribute_weights_sync` pass used to send VAR=bad to the WHOLE cohort
-- ~10 messages/trainer/iteration. A busy straggler then drained that backlog
FIFO before it saw the live instruction, restarting its next compute up to ~10s
late (the whole residual real<->sim throughput gap). `_distribute_weights_sync`
now skips an end already dispatched the current `version_key`, collapsing the
cohort to one instruction each (matching the async path, already ~1x). These
tests pin the dedup predicate + mark, and the loop-level "exactly once per
version_key, re-served on advance" invariant.
"""

from flame.mode.horizontal.syncfl.fwdllm_aggregator import TopAggregator


class _DedupAgg:
    """Binds the real dedup helpers onto a minimal stand-in with a settable
    version_key (via model_version + iteration_per_data_id, as the property
    derives it)."""

    _already_served_current_instruction = (
        TopAggregator._already_served_current_instruction
    )
    _mark_instruction_served = TopAggregator._mark_instruction_served
    version_key = TopAggregator.version_key  # property over the two fields below

    def __init__(self, model_version=0, iteration=0):
        self._end_served_version_key = {}
        self._model_version = model_version
        self.iteration_per_data_id = iteration


class TestServedPredicate:
    def test_never_served_is_false(self):
        agg = _DedupAgg()
        assert agg._already_served_current_instruction("A") is False

    def test_marked_is_served_for_same_version_key(self):
        agg = _DedupAgg()
        agg._mark_instruction_served("A")
        assert agg._already_served_current_instruction("A") is True
        # A different end is unaffected.
        assert agg._already_served_current_instruction("B") is False

    def test_iteration_advance_reserves(self):
        agg = _DedupAgg(model_version=2, iteration=0)
        agg._mark_instruction_served("A")
        assert agg._already_served_current_instruction("A") is True
        agg.iteration_per_data_id = 1  # variance-FAIL -> next iteration
        assert agg._already_served_current_instruction("A") is False

    def test_model_version_advance_reserves(self):
        agg = _DedupAgg(model_version=2, iteration=3)
        agg._mark_instruction_served("A")
        assert agg._already_served_current_instruction("A") is True
        agg._model_version = 3  # data-bin advance
        agg.iteration_per_data_id = 0
        assert agg._already_served_current_instruction("A") is False


class TestLoopLevelDedup:
    """Simulate the distribute-per-collect loop body (skip-if-served else
    send+mark) and assert the send count collapses to one per version_key."""

    def _distribute_pass(self, agg, ends):
        sent = []
        for end in ends:
            if agg._already_served_current_instruction(end):
                continue
            sent.append(end)  # stands in for channel.send(end, payload)
            agg._mark_instruction_served(end)
        return sent

    def test_one_send_per_trainer_per_iteration_across_passes(self):
        agg = _DedupAgg(model_version=1, iteration=0)
        ends = ["A", "B", "C"]
        # First pass dispatches the whole cohort; the ~agg_goal later passes
        # (one per incremental collect) send nothing.
        first = self._distribute_pass(agg, ends)
        assert first == ends
        total = list(first)
        for _ in range(9):  # mimic 9 more distribute-per-collect passes
            total += self._distribute_pass(agg, ends)
        assert total == ends  # exactly one send each, not 10x

    def test_reserves_whole_cohort_on_version_key_advance(self):
        agg = _DedupAgg(model_version=1, iteration=0)
        ends = ["A", "B", "C"]
        assert self._distribute_pass(agg, ends) == ends
        assert self._distribute_pass(agg, ends) == []  # already served
        agg.iteration_per_data_id = 1  # variance-FAIL advances the iteration
        assert self._distribute_pass(agg, ends) == ends  # re-served once
        assert self._distribute_pass(agg, ends) == []
