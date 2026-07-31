# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""The `(round, data_id)` progress key must be MONOTONE across a lap boundary.

Two distinct source bugs made it non-monotone, both in
`_process_aggregation_goal_met`:

1. `agg_round` emitted `round_num=self._round` read LIVE, while
   `cycle_data_id`/`cycle_iteration`/`cycle_model_version` were snapshotted
   BEFORE the pass/fail branch. On the last bin of a lap that branch has already
   bumped `_round`, so the record paired a post-bump round with a pre-mutation
   data_id and a lap read `(1,148) -> (2,149) -> (2,0)`: `(2,149)` sorts LAST but
   happened FIRST. Anything that sorts, maxes or windows on the key is then
   wrong (it broke the parity checker's matched-budget primitive).

2. Advancing off the last bin left `self.data_id == total_data_bins` -- out of
   range -- until the wrap several statements later, and `version_bump_census`
   read it in that window, emitting data_id=150 with total_data_bins=150 once
   per lap.

The training state machine itself was always correct: bin 149 of round 1 does
complete before bin 0 of round 2, and staleness keys on `_model_version`, never
on `_round` (§F-2). These are record/observability defects, but a non-monotone
progress key is load-bearing for every downstream consumer, so both are fixed at
the source rather than normalized by readers.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from flame.mode.horizontal.syncfl.fwdllm_aggregator import TopAggregator


class _FakeChannel:
    def __init__(self):
        self.properties = {}

    def get_end_property(self, end, prop):
        return None

    def set_property(self, name, value):
        self.properties[name] = value

    def cleanup_recvd_ends(self):
        pass


class _Agg:
    """Minimal stand-in for the state `_process_aggregation_goal_met` touches."""

    def __init__(self, round_=1, data_id=149, total_data_bins=150,
                 var_good_enough=True):
        self.simulated = False
        self._agg_goal = 2
        self._agg_goal_cnt = 2
        self._per_agg_trainer_list = []
        self._cycle_grad_norms = []
        self._model_version_unique_trainers = set()
        self._model_version_trainer_stats = {"train_duration": [],
                                             "partial_stat_utility": []}
        self.grad_pool = []
        self.grad = []
        self.model = None
        self._max_iter_per_data_id = None
        self.iteration_per_data_id = 0
        self.var = 0.0
        self.var_threshold = 0.3
        self.var_good_enough = var_good_enough
        self.data_id = data_id
        self.total_data_bins = total_data_bins
        self._is_model_updated = False
        self._model_version = 7
        self._round = round_
        self._updates_in_queue = 2
        self._updates_received = {}
        self._n_aggs_completed = 0
        self._var_total_count = 0
        self._var_pass_count = 0
        self._dynamic_kc_controller = None
        self._weights_sent_this_cycle = set()
        self.config = SimpleNamespace(hyperparameters=SimpleNamespace(rounds=50))

    def add_local_trained_result(self, *a, **k):
        pass

    def aggregate(self, round_id):
        pass

    def eval_model(self, model=None):
        return {"eval_loss": 0.0}, None, None

    def _log_and_reset_model_version_stats(self):
        pass

    process = TopAggregator._process_aggregation_goal_met
    _replay_buffered_cohort_contribs = TopAggregator._replay_buffered_cohort_contribs
    _eval_snapshot_model = TopAggregator._eval_snapshot_model


@pytest.fixture
def emitted():
    """Capture every telemetry event the aggregation-goal path emits."""
    events = []

    def _emit(ev, **fields):
        events.append((ev, fields))

    with patch("flame.mode.horizontal.syncfl.fwdllm_aggregator.telemetry.is_enabled",
               return_value=True), \
         patch("flame.mode.horizontal.syncfl.fwdllm_aggregator.telemetry.emit",
               side_effect=_emit):
        yield events


@patch("flame.mode.horizontal.syncfl.fwdllm_aggregator.fc."
       "make_functional_with_buffers", return_value=(None, [], None))
class TestLapBoundaryProgressKey:
    def test_agg_round_round_is_the_cycle_it_worked_on(self, _ffb, emitted):
        # Committing the LAST bin of lap 1 must record (round=1, data_id=149) --
        # the cycle that actually ran -- not the post-bump (2, 149).
        agg = _Agg(round_=1, data_id=149)
        agg.process("tag", _FakeChannel())

        rounds = [f for ev, f in emitted if f.get("cycle_data_id") is not None]
        assert rounds, "no agg_round emitted"
        assert rounds[-1]["round"] == 1
        assert rounds[-1]["cycle_data_id"] == 149

    def test_state_advanced_even_though_the_record_says_round_1(self, _ffb, emitted):
        # The record is retrospective; the live state must still have rolled over.
        agg = _Agg(round_=1, data_id=149)
        agg.process("tag", _FakeChannel())
        assert (agg._round, agg.data_id) == (2, 0)

    def test_key_is_monotone_across_the_wrap(self, _ffb, emitted):
        # Walk 148 -> 149 -> 0 and assert the emitted keys strictly increase.
        keys = []
        agg = _Agg(round_=1, data_id=148)
        for _ in range(3):
            agg.process("tag", _FakeChannel())
            rec = [f for ev, f in emitted if f.get("cycle_data_id") is not None][-1]
            keys.append((rec["round"], rec["cycle_data_id"]))
            agg.var_good_enough = True
        assert keys == [(1, 148), (1, 149), (2, 0)]
        assert keys == sorted(keys), keys

    def test_mid_lap_round_is_unchanged(self, _ffb, emitted):
        agg = _Agg(round_=3, data_id=5)
        agg.process("tag", _FakeChannel())
        rec = [f for ev, f in emitted if f.get("cycle_data_id") is not None][-1]
        assert (rec["round"], rec["cycle_data_id"]) == (3, 5)
        assert (agg._round, agg.data_id) == (3, 6)

    def test_variance_retry_does_not_advance_the_key(self, _ffb, emitted):
        agg = _Agg(round_=1, data_id=149, var_good_enough=False)
        agg.process("tag", _FakeChannel())
        rec = [f for ev, f in emitted if f.get("cycle_data_id") is not None][-1]
        assert (rec["round"], rec["cycle_data_id"]) == (1, 149)
        assert (agg._round, agg.data_id) == (1, 149)


@patch("flame.mode.horizontal.syncfl.fwdllm_aggregator.fc."
       "make_functional_with_buffers", return_value=(None, [], None))
class TestDataIdNeverOutOfRange:
    def test_version_bump_census_sees_a_wrapped_data_id(self, _ffb, emitted):
        agg = _Agg(round_=1, data_id=149, total_data_bins=150)
        agg.process("tag", _FakeChannel())
        census = [f for ev, f in emitted if "new_model_version" in f]
        assert census, "no version_bump_census emitted"
        assert census[-1]["data_id"] == 0        # was 150 -- a bin that cannot exist

    def test_no_emitted_data_id_ever_reaches_total_data_bins(self, _ffb, emitted):
        agg = _Agg(round_=1, data_id=147, total_data_bins=150)
        for _ in range(5):
            agg.process("tag", _FakeChannel())
            agg.var_good_enough = True
        for _ev, f in emitted:
            for key in ("data_id", "cycle_data_id"):
                if f.get(key) is not None:
                    assert 0 <= f[key] < 150, (key, f[key])
