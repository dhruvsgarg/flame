# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""fwdllm's aggregator emitted zero telemetry (see
../../examples/MIGRATING_TO_LAUNCHER.md's telemetry gotchas) -- unlike
asyncfl/top_aggregator.py, it never called telemetry.emit() for agg_eval/
agg_round, so plots/performance/ and plots/insights/ were structurally empty
for every fwdllm-family baseline regardless of what the analyzer did.

This covers the fix: _process_aggregation_goal_met now emits agg_eval (right
after eval_model(), only on the variance-check-passed path) and agg_round
(once per completed aggregation cycle, on both the passed and failed paths)
when telemetry is enabled, and stays a true no-op (no emit call at all) when
it isn't.
"""

import time
from datetime import timedelta

import torch

from flame import telemetry
from flame.mode.horizontal.syncfl.fwdllm_aggregator import (
    PROP_ROUND_START_TIME,
    PROP_STAT_UTILITY,
    TopAggregator,
)
from flame.mode.message import MessageType
from flame.selector.properties import PROP_CLIENT_TASK_TRAIN_DURATION


class _FakeChannel:
    def __init__(self, durations=None, utilities=None):
        self._durations = durations or {}
        self._utilities = utilities or {}
        self.cleaned_up_rounds = []
        self.round_prop = None

    def get_end_property(self, end, key):
        if key == PROP_CLIENT_TASK_TRAIN_DURATION:
            return self._durations.get(end)
        if key == PROP_STAT_UTILITY:
            return self._utilities.get(end)
        return None

    def set_property(self, key, value):
        if key == "round":
            self.round_prop = value

    def cleanup_recvd_ends(self):
        self.cleaned_up_rounds.append("cleaned")


class _FakeHyperparameters:
    rounds = 1000


class _FakeConfig:
    hyperparameters = _FakeHyperparameters()


class _FakeAggregator:
    """Binds the real _process_aggregation_goal_met onto a minimal stand-in.

    Only stubs the heavy ML plumbing (aggregate/eval_model/model
    functionalization) that method touches but which is irrelevant to the
    telemetry-emission logic under test -- everything else (round/data_id
    bookkeeping, telemetry field construction) runs for real.
    """

    _process_aggregation_goal_met = TopAggregator._process_aggregation_goal_met

    def __init__(self, contributors, var_good_enough, staleness_map=None,
                 total_data_bins=150):
        # Real path skips the sim boundary hook, so telemetry is byte-identical.
        self.simulated = False
        self._per_agg_trainer_list = list(contributors)
        self._cycle_grad_norms = []  # G1: populated by the real caller per-contribution
        self._agg_goal_cnt = len(contributors)
        self._agg_goal = len(contributors) or 1
        self._model_version_unique_trainers = set()
        self._model_version_trainer_stats = {
            "train_duration": [], "partial_stat_utility": [],
        }
        self._trainer_last_model_version = staleness_map or {}
        self._model_version = 5
        self._round = 1
        self.data_id = 3
        self.iteration_per_data_id = 0
        self.total_data_bins = total_data_bins
        self._updates_in_queue = len(contributors)
        self._updates_received = {c: 1 for c in contributors}
        self._max_iter_per_data_id = None
        self._dynamic_kc_controller = None
        self._n_aggs_completed = 0
        self._var_pass_count = 0
        self._var_total_count = 0
        self.var = 0.1
        self.var_good_enough = var_good_enough
        self.var_threshold = 0.3
        self.grad_pool = []
        self.grad = [torch.zeros(1)]
        self.model = torch.nn.Linear(1, 1)
        self.config = _FakeConfig()
        self._eval_result = {"eval_loss": 0.42, "acc": 0.9, "mcc": 0.5}

        # Stubs for the heavy ML machinery this method calls but which this
        # test doesn't need to exercise for real.
        import functorch as fc
        self.fmodel, self.params, self.buffers = fc.make_functional_with_buffers(
            self.model
        )

    def aggregate(self, round_num):
        pass  # normally sets self.var/self.var_good_enough as a side effect

    def add_local_trained_result(self, *a, **k):
        pass

    def eval_model(self):
        return dict(self._eval_result), None, []

    def _log_and_reset_model_version_stats(self):
        pass


class TestAggEvalTelemetry:
    def test_emitted_on_variance_pass_with_expected_fields(self, tmp_path):
        telemetry.configure(role="aggregator", run_dir=str(tmp_path))
        try:
            agg = _FakeAggregator(contributors=["t1"], var_good_enough=True)
            channel = _FakeChannel(
                durations={"t1": timedelta(seconds=5)},
                utilities={"t1": 1.5},
            )

            agg._process_aggregation_goal_met(tag="aggregate", channel=channel)

            lines = (tmp_path / "aggregator.jsonl").read_text().splitlines()
            import json
            events = [json.loads(l) for l in lines]
            evals = [e for e in events if e["event"] == "agg_eval"]
            assert len(evals) == 1
            assert evals[0]["test-loss"] == 0.42
            assert evals[0]["test-accuracy"] == 0.9
            # data_id snapshot must be the pre-increment value (the data_id
            # that was actually evaluated), not the post-increment one.
            assert evals[0]["data_id"] == 3
            assert evals[0]["round"] == 1
        finally:
            telemetry.shutdown()

    def test_not_emitted_on_variance_fail(self, tmp_path):
        """eval_model() only runs on the variance-check-passed path, so no
        agg_eval event should appear when the check fails."""
        telemetry.configure(role="aggregator", run_dir=str(tmp_path))
        try:
            agg = _FakeAggregator(contributors=["t1"], var_good_enough=False)
            channel = _FakeChannel()

            agg._process_aggregation_goal_met(tag="aggregate", channel=channel)

            import json
            lines = (tmp_path / "aggregator.jsonl").read_text().splitlines()
            events = [json.loads(l) for l in lines]
            assert not [e for e in events if e["event"] == "agg_eval"]
        finally:
            telemetry.shutdown()

    def test_noop_when_telemetry_disabled(self, tmp_path):
        """No configure() call -- telemetry stays disabled; must not raise
        and must not write anything."""
        assert not telemetry.is_enabled()
        agg = _FakeAggregator(contributors=["t1"], var_good_enough=True)
        channel = _FakeChannel(durations={"t1": timedelta(seconds=5)})

        agg._process_aggregation_goal_met(tag="aggregate", channel=channel)

        assert not (tmp_path / "aggregator.jsonl").exists()


class TestAggRoundTelemetry:
    def test_emitted_every_cycle_regardless_of_variance_outcome(self, tmp_path):
        telemetry.configure(role="aggregator", run_dir=str(tmp_path))
        try:
            agg = _FakeAggregator(contributors=["t1", "t2"], var_good_enough=False)
            channel = _FakeChannel(
                durations={"t1": timedelta(seconds=5), "t2": timedelta(seconds=7)},
                utilities={"t1": 1.0, "t2": 2.0},
            )

            agg._process_aggregation_goal_met(tag="aggregate", channel=channel)

            import json
            lines = (tmp_path / "aggregator.jsonl").read_text().splitlines()
            events = [json.loads(l) for l in lines]
            rounds = [e for e in events if e["event"] == "agg_round"]
            assert len(rounds) == 1
            r = rounds[0]
            assert sorted(r["contributing_trainers"]) == ["t1", "t2"]
            assert sorted(r["trainer_speed_s"]) == [5.0, 7.0]
            assert sorted(r["stat_utility"]) == [1.0, 2.0]
        finally:
            telemetry.shutdown()

    def test_grad_norm_emitted_and_resets_next_cycle(self, tmp_path):
        """G1: per-cycle grad norms accumulated by aggregate_grads_from_trainers
        (simulated here directly, since this fake doesn't exercise the message-
        processing path) land on agg_round as `grad_norm`, and the accumulator
        is empty again for the next cycle."""
        telemetry.configure(role="aggregator", run_dir=str(tmp_path))
        try:
            agg = _FakeAggregator(contributors=["t1", "t2"], var_good_enough=False)
            agg._cycle_grad_norms = [1.5, 2.5]
            channel = _FakeChannel(
                durations={"t1": timedelta(seconds=5), "t2": timedelta(seconds=7)},
                utilities={"t1": 1.0, "t2": 2.0},
            )

            agg._process_aggregation_goal_met(tag="aggregate", channel=channel)

            import json
            events = [json.loads(l) for l in
                      (tmp_path / "aggregator.jsonl").read_text().splitlines()]
            r = [e for e in events if e["event"] == "agg_round"][0]
            assert r["grad_norm"] == [1.5, 2.5]
            assert agg._cycle_grad_norms == []  # reset for the next cycle
        finally:
            telemetry.shutdown()

    def test_speedup_fields_emitted(self, tmp_path):
        """#13: agg_round carries wall_elapsed_s in both modes and, in sim,
        sim_rate = vclock/wall so the slowdown is observable in telemetry."""
        telemetry.configure(role="aggregator", run_dir=str(tmp_path))
        try:
            agg = _FakeAggregator(contributors=["t1"], var_good_enough=False)
            # Put the aggregator on the sim path with a virtual clock ahead of
            # a known wall span.
            agg.simulated = True
            agg._vclock = type("V", (), {"now": 120.0})()
            agg.agg_start_time_ts = time.time() - 60.0  # ~60 wall-s elapsed
            # The sim boundary hook is exercised elsewhere; no-op it here so this
            # test isolates the speedup-telemetry emission.
            agg._release_sim_slots_at_agg_goal = lambda *a, **k: None
            channel = _FakeChannel(
                durations={"t1": timedelta(seconds=1)}, utilities={"t1": 0.1}
            )

            agg._process_aggregation_goal_met(tag="aggregate", channel=channel)

            import json
            events = [json.loads(l) for l in
                      (tmp_path / "aggregator.jsonl").read_text().splitlines()]
            r = [e for e in events if e["event"] == "agg_round"][0]
            assert r["wall_elapsed_s"] > 0
            # vclock 120 over ~60 wall-s -> sim_rate ~2 (a speedup); must be present
            assert r["sim_rate"] is not None and r["sim_rate"] > 1.0
        finally:
            telemetry.shutdown()

    def test_intrinsic_span_is_barrier_plus_eval_excludes_fedavg(self, tmp_path):
        """#6 anchor: intrinsic_span_s = the barrier (MAX committed-cohort
        duration) + eval_s (commit only), EXCLUDING the FedAvg merge -- it must
        mirror the sim vclock's composition (barrier sct + eval fold) so the
        clock-rate rungs anchor REAL like-for-like. A variance-FAIL cycle runs no
        eval, so intrinsic collapses to exactly the barrier -- which also proves
        max() (not min/sum) and that fedavg is not folded in."""
        telemetry.configure(role="aggregator", run_dir=str(tmp_path))
        try:
            agg = _FakeAggregator(contributors=["t1", "t2"],
                                  var_good_enough=False)
            channel = _FakeChannel(durations={"t1": timedelta(seconds=5),
                                              "t2": timedelta(seconds=3)})

            agg._process_aggregation_goal_met(tag="aggregate", channel=channel)

            import json
            events = [json.loads(l) for l in
                      (tmp_path / "aggregator.jsonl").read_text().splitlines()]
            r = [e for e in events if e["event"] == "agg_round"][0]
            # barrier = max(5, 3) = 5; no eval on the fail path; fedavg (stubbed
            # ~0) is excluded regardless -> intrinsic == the barrier.
            assert r["intrinsic_span_s"] is not None
            assert abs(r["intrinsic_span_s"] - 5.0) < 0.5, r
        finally:
            telemetry.shutdown()

    def test_wall_elapsed_emitted_in_real_mode_sim_rate_none(self, tmp_path):
        telemetry.configure(role="aggregator", run_dir=str(tmp_path))
        try:
            agg = _FakeAggregator(contributors=["t1"], var_good_enough=False)
            # default fake is real mode (simulated=False)
            agg.agg_start_time_ts = time.time() - 5.0
            channel = _FakeChannel(
                durations={"t1": timedelta(seconds=1)}, utilities={"t1": 0.1}
            )
            agg._process_aggregation_goal_met(tag="aggregate", channel=channel)
            import json
            events = [json.loads(l) for l in
                      (tmp_path / "aggregator.jsonl").read_text().splitlines()]
            r = [e for e in events if e["event"] == "agg_round"][0]
            assert r["wall_elapsed_s"] > 0
            assert r["sim_rate"] is None  # real mode has no virtual clock rate
        finally:
            telemetry.shutdown()

    def test_staleness_computed_against_pre_cycle_model_version(self, tmp_path):
        """staleness = the model version this cycle aggregated against, minus
        each contributor's last-known trained-on version -- captured BEFORE
        self._model_version potentially advances later in the same call."""
        telemetry.configure(role="aggregator", run_dir=str(tmp_path))
        try:
            agg = _FakeAggregator(
                contributors=["t1"], var_good_enough=True,
                staleness_map={"t1": 3},  # agg's _model_version starts at 5
            )
            channel = _FakeChannel(
                durations={"t1": timedelta(seconds=1)}, utilities={"t1": 0.1}
            )

            agg._process_aggregation_goal_met(tag="aggregate", channel=channel)

            import json
            lines = (tmp_path / "aggregator.jsonl").read_text().splitlines()
            events = [json.loads(l) for l in lines]
            rounds = [e for e in events if e["event"] == "agg_round"]
            assert rounds[0]["staleness"] == [2]  # 5 - 3, not 6 - 3
        finally:
            telemetry.shutdown()

    def test_cadence_fields_snapshot_pre_mutation(self, tmp_path):
        """Variance-cadence inputs: cycle_data_id/cycle_iteration identify the
        data_id this cycle WORKED on (pre-advance), and the pool sizes are captured
        at the variance gate. On a variance FAIL data_id does not advance, so
        cycle_data_id == the emitted (post) data_id == 3."""
        telemetry.configure(role="aggregator", run_dir=str(tmp_path))
        try:
            agg = _FakeAggregator(contributors=["t1"], var_good_enough=False)
            channel = _FakeChannel(durations={"t1": timedelta(seconds=2)})

            agg._process_aggregation_goal_met(tag="aggregate", channel=channel)

            import json
            events = [json.loads(l) for l in
                      (tmp_path / "aggregator.jsonl").read_text().splitlines()]
            r = [e for e in events if e["event"] == "agg_round"][0]
            assert r["cycle_data_id"] == 3 and r["cycle_iteration"] == 0
            # grad_pool got this cycle's grad appended before the snapshot; the
            # fake sets no cached_v, so cached_v_size defaults to 0.
            assert r["grad_pool_size"] == 1 and r["cached_v_size"] == 0
        finally:
            telemetry.shutdown()

    def test_cycle_data_id_is_pre_advance_on_commit(self, tmp_path):
        """On a variance PASS the emitted (post) data_id advances to 4, but
        cycle_data_id stays 3 -- the data_id this cycle committed. This is the
        off-by-one V1 relies on: bin cadence cycles by cycle_data_id, not the
        post-mutation data_id (which would attribute a commit to the next bin)."""
        telemetry.configure(role="aggregator", run_dir=str(tmp_path))
        try:
            agg = _FakeAggregator(contributors=["t1"], var_good_enough=True)
            channel = _FakeChannel(durations={"t1": timedelta(seconds=2)})

            agg._process_aggregation_goal_met(tag="aggregate", channel=channel)

            import json
            events = [json.loads(l) for l in
                      (tmp_path / "aggregator.jsonl").read_text().splitlines()]
            r = [e for e in events if e["event"] == "agg_round"][0]
            assert r["cycle_data_id"] == 3      # worked-on data_id
            assert r["data_id"] == 4            # post-commit advance
        finally:
            telemetry.shutdown()

    def test_cycle_model_version_is_pre_advance_on_commit(self, tmp_path):
        """Aggregation/model-version stage instrumentation: `cycle_model_version`
        is the SAME pre-mutation snapshot pattern as cycle_data_id -- the
        version this cycle worked on, not the post-commit bump, so a checker
        can correlate "this cycle committed" with "next cycle's
        cycle_model_version == this one + 1" directly off agg_round."""
        telemetry.configure(role="aggregator", run_dir=str(tmp_path))
        try:
            agg = _FakeAggregator(contributors=["t1"], var_good_enough=True)
            channel = _FakeChannel(durations={"t1": timedelta(seconds=2)})
            assert agg._model_version == 5

            agg._process_aggregation_goal_met(tag="aggregate", channel=channel)

            import json
            events = [json.loads(l) for l in
                      (tmp_path / "aggregator.jsonl").read_text().splitlines()]
            r = [e for e in events if e["event"] == "agg_round"][0]
            assert r["cycle_model_version"] == 5     # pre-bump
            assert agg._model_version == 6            # bumped by the commit

        finally:
            telemetry.shutdown()

    def test_cycle_model_version_unchanged_on_fail(self, tmp_path):
        telemetry.configure(role="aggregator", run_dir=str(tmp_path))
        try:
            agg = _FakeAggregator(contributors=["t1"], var_good_enough=False)
            channel = _FakeChannel(durations={"t1": timedelta(seconds=2)})

            agg._process_aggregation_goal_met(tag="aggregate", channel=channel)

            import json
            events = [json.loads(l) for l in
                      (tmp_path / "aggregator.jsonl").read_text().splitlines()]
            r = [e for e in events if e["event"] == "agg_round"][0]
            assert r["cycle_model_version"] == 5
            assert agg._model_version == 5
        finally:
            telemetry.shutdown()

    def test_contributor_list_captured_before_reset(self, tmp_path):
        """_per_agg_trainer_list is cleared at the end of this method --
        agg_round's contributing_trainers must reflect this cycle's
        contributors, not the post-reset empty list."""
        telemetry.configure(role="aggregator", run_dir=str(tmp_path))
        try:
            agg = _FakeAggregator(contributors=["a", "b", "c"], var_good_enough=False)
            channel = _FakeChannel()

            agg._process_aggregation_goal_met(tag="aggregate", channel=channel)

            assert agg._per_agg_trainer_list == []  # confirms the real reset ran
            import json
            lines = (tmp_path / "aggregator.jsonl").read_text().splitlines()
            events = [json.loads(l) for l in lines]
            rounds = [e for e in events if e["event"] == "agg_round"]
            assert sorted(rounds[0]["contributing_trainers"]) == ["a", "b", "c"]
        finally:
            telemetry.shutdown()

    def test_noop_when_telemetry_disabled(self, tmp_path):
        assert not telemetry.is_enabled()
        agg = _FakeAggregator(contributors=["t1"], var_good_enough=False)
        channel = _FakeChannel()

        agg._process_aggregation_goal_met(tag="aggregate", channel=channel)

        assert not (tmp_path / "aggregator.jsonl").exists()

    def test_agg_observed_s_keyed_by_end_id(self, tmp_path):
        """agg_observed_s reuses the same PROP_ROUND_DURATION values already
        read for trainer_speed_s, but as an {end_id: seconds} dict -- lets
        analyze_run.py's runtime_agg_vs_trainer/runtime_overhead_* plots work
        for fwdllm too (Part 6 follow-on to P5.2)."""
        telemetry.configure(role="aggregator", run_dir=str(tmp_path))
        try:
            agg = _FakeAggregator(contributors=["t1", "t2"], var_good_enough=False)
            channel = _FakeChannel(
                durations={"t1": timedelta(seconds=5), "t2": timedelta(seconds=7)},
            )

            agg._process_aggregation_goal_met(tag="aggregate", channel=channel)

            import json
            lines = (tmp_path / "aggregator.jsonl").read_text().splitlines()
            events = [json.loads(l) for l in lines]
            rounds = [e for e in events if e["event"] == "agg_round"]
            assert rounds[0]["agg_observed_s"] == {"t1": 5.0, "t2": 7.0}
        finally:
            telemetry.shutdown()


class TestContributorIntervalsEmission:
    """R1/W1 residence rungs read a per-contributor [dispatch, commit] interval
    list off each agg_round event. It must land once per contributor, carrying
    the ts captured in _process_single_trainer_message."""

    def test_intervals_emitted_per_contributor(self, tmp_path):
        telemetry.configure(role="aggregator", run_dir=str(tmp_path))
        try:
            agg = _FakeAggregator(contributors=["t1", "t2"], var_good_enough=False)
            # Simulate the per-contribution capture done in the message handler.
            agg._sim_contrib_intervals = {
                "t1": {"dispatch_ts": 1.0, "commit_ts": 6.0},
                "t2": {"dispatch_ts": 2.0, "commit_ts": 9.0},
            }
            channel = _FakeChannel(
                durations={"t1": timedelta(seconds=5), "t2": timedelta(seconds=7)},
            )

            agg._process_aggregation_goal_met(tag="aggregate", channel=channel)

            import json
            events = [json.loads(l) for l in
                      (tmp_path / "aggregator.jsonl").read_text().splitlines()]
            r = [e for e in events if e["event"] == "agg_round"][0]
            ci = {d["end"]: d for d in r["contributor_intervals"]}
            assert set(ci) == {"t1", "t2"}
            assert ci["t1"]["dispatch_ts"] == 1.0 and ci["t1"]["commit_ts"] == 6.0
            assert ci["t2"]["dispatch_ts"] == 2.0 and ci["t2"]["commit_ts"] == 9.0
        finally:
            telemetry.shutdown()

    def test_processing_wall_ts_flows_through_when_captured(self, tmp_path):
        """New commit-stage instrumentation: `processing_wall_ts` (the wall
        moment the aggregator's own drain loop accepted a grad, distinct from
        the trainer's dispatch/commit schedule) rides through to
        contributor_intervals whenever _process_single_trainer_message
        populated it -- lets a checker see a ready-but-unprocessed grad at
        per-contributor granularity (the #15 phantom-drain-gate bug class)."""
        telemetry.configure(role="aggregator", run_dir=str(tmp_path))
        try:
            agg = _FakeAggregator(contributors=["t1", "t2"], var_good_enough=False)
            agg._sim_contrib_intervals = {
                "t1": {"dispatch_ts": 1.0, "commit_ts": 6.0, "processing_wall_ts": 5.5},
                "t2": {"dispatch_ts": 2.0, "commit_ts": 9.0, "processing_wall_ts": 8.9},
            }
            channel = _FakeChannel(
                durations={"t1": timedelta(seconds=5), "t2": timedelta(seconds=7)},
            )

            agg._process_aggregation_goal_met(tag="aggregate", channel=channel)

            import json
            events = [json.loads(l) for l in
                      (tmp_path / "aggregator.jsonl").read_text().splitlines()]
            r = [e for e in events if e["event"] == "agg_round"][0]
            ci = {d["end"]: d for d in r["contributor_intervals"]}
            assert ci["t1"]["processing_wall_ts"] == 5.5
            assert ci["t2"]["processing_wall_ts"] == 8.9
        finally:
            telemetry.shutdown()

    def test_intervals_present_even_without_captured_ts(self, tmp_path):
        """When no interval was captured (e.g. a test double / real run with the
        dict unpopulated) the field is still emitted with null ts, so the rung
        SKIPs cleanly rather than the field being absent."""
        telemetry.configure(role="aggregator", run_dir=str(tmp_path))
        try:
            agg = _FakeAggregator(contributors=["t1"], var_good_enough=False)
            channel = _FakeChannel(durations={"t1": timedelta(seconds=5)})

            agg._process_aggregation_goal_met(tag="aggregate", channel=channel)

            import json
            events = [json.loads(l) for l in
                      (tmp_path / "aggregator.jsonl").read_text().splitlines()]
            r = [e for e in events if e["event"] == "agg_round"][0]
            assert r["contributor_intervals"] == [
                {"end": "t1", "dispatch_ts": None, "commit_ts": None,
                 "processing_wall_ts": None, "dispatch_model_version": None,
                 "agg_model_version_at_commit": 5,
                 "dispatch_version_key": None,
                 "agg_version_key_at_commit": [5, 0]}]
        finally:
            telemetry.shutdown()


class TestPerRoundWallDecomposition:
    """agg_round carries the per-round wall breakdown feeding #6 --
    aggregate_fedavg_s + eval_s always; barrier_wait_s/drain_tail_s when the
    dispatch/last-grad wall stamps were captured (else null, rung SKIPs)."""

    def test_fedavg_and_eval_present_on_pass(self, tmp_path):
        telemetry.configure(role="aggregator", run_dir=str(tmp_path))
        try:
            agg = _FakeAggregator(contributors=["t1"], var_good_enough=True)
            channel = _FakeChannel(durations={"t1": timedelta(seconds=2)})

            agg._process_aggregation_goal_met(tag="aggregate", channel=channel)

            import json
            events = [json.loads(l) for l in
                      (tmp_path / "aggregator.jsonl").read_text().splitlines()]
            r = [e for e in events if e["event"] == "agg_round"][0]
            # fedavg wall is measured around aggregate() and is non-negative
            assert r["aggregate_fedavg_s"] is not None
            assert r["aggregate_fedavg_s"] >= 0.0
            # eval ran (variance passed) -> eval_s measured, non-negative
            assert r["eval_s"] is not None and r["eval_s"] >= 0.0
        finally:
            telemetry.shutdown()

    def test_agg_compute_window_matches_fedavg_span(self, tmp_path):
        """§J step-1 telemetry: agg_compute_start_wall/end_wall bracket the
        aggregate() call exactly, so an overlap-measurement script can trust
        the window (end - start == aggregate_fedavg_s)."""
        telemetry.configure(role="aggregator", run_dir=str(tmp_path))
        try:
            agg = _FakeAggregator(contributors=["t1"], var_good_enough=True)
            channel = _FakeChannel(durations={"t1": timedelta(seconds=2)})

            agg._process_aggregation_goal_met(tag="aggregate", channel=channel)

            import json
            events = [json.loads(l) for l in
                      (tmp_path / "aggregator.jsonl").read_text().splitlines()]
            r = [e for e in events if e["event"] == "agg_round"][0]
            assert r["agg_compute_start_wall"] is not None
            assert r["agg_compute_end_wall"] is not None
            assert r["agg_compute_end_wall"] >= r["agg_compute_start_wall"]
            assert abs((r["agg_compute_end_wall"] - r["agg_compute_start_wall"])
                       - r["aggregate_fedavg_s"]) < 1e-6
        finally:
            telemetry.shutdown()

    def test_eval_s_null_on_variance_fail(self, tmp_path):
        """eval_model only runs on the pass path, so eval_s is null on a FAIL."""
        telemetry.configure(role="aggregator", run_dir=str(tmp_path))
        try:
            agg = _FakeAggregator(contributors=["t1"], var_good_enough=False)
            channel = _FakeChannel(durations={"t1": timedelta(seconds=2)})

            agg._process_aggregation_goal_met(tag="aggregate", channel=channel)

            import json
            events = [json.loads(l) for l in
                      (tmp_path / "aggregator.jsonl").read_text().splitlines()]
            r = [e for e in events if e["event"] == "agg_round"][0]
            assert r["eval_s"] is None
            assert r["aggregate_fedavg_s"] is not None  # aggregate always runs
        finally:
            telemetry.shutdown()

    def test_barrier_and_drain_from_wall_stamps(self, tmp_path):
        """When the dispatch + last-grad wall stamps exist, barrier_wait_s =
        last_grad - dispatch and drain_tail_s = commit - last_grad."""
        telemetry.configure(role="aggregator", run_dir=str(tmp_path))
        try:
            agg = _FakeAggregator(contributors=["t1"], var_good_enough=False)
            # Stamps the barrier collection would have set (dispatch then last
            # grad, both in the past relative to the commit inside the method).
            import time as _t
            now = _t.time()
            agg._round_dispatch_wall_ts = now - 5.0
            agg._last_grad_wall_ts = now - 2.0
            channel = _FakeChannel(durations={"t1": timedelta(seconds=2)})

            agg._process_aggregation_goal_met(tag="aggregate", channel=channel)

            import json
            events = [json.loads(l) for l in
                      (tmp_path / "aggregator.jsonl").read_text().splitlines()]
            r = [e for e in events if e["event"] == "agg_round"][0]
            assert abs(r["barrier_wait_s"] - 3.0) < 0.5  # (now-2) - (now-5)
            assert r["drain_tail_s"] >= 0.0  # commit is after last grad
        finally:
            telemetry.shutdown()

    def test_barrier_drain_null_without_stamps(self, tmp_path):
        """No wall stamps captured (test double / async path) -> null, not a
        crash; the rung SKIPs rather than the field being absent."""
        telemetry.configure(role="aggregator", run_dir=str(tmp_path))
        try:
            agg = _FakeAggregator(contributors=["t1"], var_good_enough=False)
            channel = _FakeChannel(durations={"t1": timedelta(seconds=2)})

            agg._process_aggregation_goal_met(tag="aggregate", channel=channel)

            import json
            events = [json.loads(l) for l in
                      (tmp_path / "aggregator.jsonl").read_text().splitlines()]
            r = [e for e in events if e["event"] == "agg_round"][0]
            assert r["barrier_wait_s"] is None
            assert r["drain_tail_s"] is None
        finally:
            telemetry.shutdown()


class _UtilityFakeChannel:
    """Generic fake for _process_single_trainer_message's channel calls --
    stores per-end properties in a dict, doesn't care about specific PROP_*
    identities beyond PROP_STAT_UTILITY/PROP_ROUND_START_TIME (both read by
    the code path under test)."""

    class _Selector:
        def __init__(self):
            self.ordered_updates_recv_ends = []

    def __init__(self, stat_utility=None):
        self._stat_utility = dict(stat_utility or {})
        self._selector = self._Selector()

    def get_end_property(self, end, key):
        if key == PROP_STAT_UTILITY:
            return self._stat_utility.get(end)
        if key == PROP_ROUND_START_TIME:
            return None  # skip PROP_ROUND_DURATION computation, irrelevant here
        return None

    def set_end_property(self, end, key, value):
        if key == PROP_STAT_UTILITY:
            self._stat_utility[end] = value

    def set_property(self, key, value):
        pass

    def cleanup_recvd_end(self, end):
        pass

    def cleanup_provided_ends(self, end):
        pass


class _UtilityFakeAggregator:
    """Minimal stand-in exposing only the state
    _process_single_trainer_message's STAT_UTILITY/utility_belief branch
    touches -- the GRADIENTS branch is stubbed out (aggregate_grads_from_
    trainers is a no-op) since it's irrelevant to the telemetry under test."""

    process = TopAggregator._process_single_trainer_message
    _release_end_on_return = TopAggregator._release_end_on_return

    def __init__(self, model_version=5, data_id=3, iteration_per_data_id=0,
                 is_async=False):
        self._per_agg_trainer_list = []
        self._trainer_last_model_version = {}
        self._updates_received = {}
        self._updates_in_queue = 0
        self._agg_goal_cnt = 0
        self._round_cache_activity_ts = {}
        self._model_version = model_version
        self.data_id = data_id
        self.iteration_per_data_id = iteration_per_data_id
        self.is_async = is_async
        self._round = 1
        self.grad_pool = []

    def aggregate_grads_from_trainers(self, *args, **kwargs):
        pass


def _msg(model_version=5, stat_utility=0.7):
    # GRADIENTS/GRADIENTS_FOR_VAR_CHECK go through _calculate_hash() (log-only,
    # unrelated to the telemetry under test) which calls .detach() on them --
    # must be real tensors, not plain lists.
    return {
        MessageType.MODEL_VERSION: model_version,
        MessageType.GRADIENTS: torch.zeros(1),
        MessageType.GRADIENTS_FOR_VAR_CHECK: torch.zeros(1),
        MessageType.STAT_UTILITY: stat_utility,
    }


class TestUtilityBeliefTelemetry:
    """fwdllm_aggregator.py never emitted utility_belief -- only
    asyncfl/top_aggregator.py did -- so selected_utility_believed_vs_actual*/
    selected_utility_belief_gap* were structurally impossible for fwdllm-
    family baselines regardless of selector (see
    ../../examples/MIGRATING_TO_LAUNCHER.md §9). Covers the fix in
    _process_single_trainer_message's STAT_UTILITY branch."""

    def test_emits_believed_and_actual(self, tmp_path):
        telemetry.configure(role="aggregator", run_dir=str(tmp_path))
        try:
            agg = _UtilityFakeAggregator()
            channel = _UtilityFakeChannel(stat_utility={"t1": 0.3})  # prior belief

            agg.process(channel, _msg(stat_utility=0.9), "t1", timestamp=0)

            import json
            events = [json.loads(l) for l in
                      (tmp_path / "aggregator.jsonl").read_text().splitlines()]
            ub = [e for e in events if e["event"] == "utility_belief"]
            assert len(ub) == 1
            assert ub[0]["believed"] == 0.3
            assert ub[0]["actual"] == 0.9
            assert ub[0]["end_id"] == "t1"
        finally:
            telemetry.shutdown()

    def test_believed_none_on_first_ever_return(self, tmp_path):
        """No prior PROP_STAT_UTILITY for this end -- believed must be None,
        not a crash or a fabricated 0."""
        telemetry.configure(role="aggregator", run_dir=str(tmp_path))
        try:
            agg = _UtilityFakeAggregator()
            channel = _UtilityFakeChannel()

            agg.process(channel, _msg(stat_utility=0.5), "t1", timestamp=0)

            import json
            events = [json.loads(l) for l in
                      (tmp_path / "aggregator.jsonl").read_text().splitlines()]
            ub = [e for e in events if e["event"] == "utility_belief"]
            assert ub[0]["believed"] is None
            assert ub[0]["actual"] == 0.5
        finally:
            telemetry.shutdown()

    def test_staleness_uses_model_version_not_round(self, tmp_path):
        """fwdllm's round can sit at 1 for an entire run -- staleness must be
        computed against self._model_version (the cycle-advancing quantity,
        matching agg_round's own staleness convention from P5.2), not round."""
        telemetry.configure(role="aggregator", run_dir=str(tmp_path))
        try:
            agg = _UtilityFakeAggregator(model_version=8)
            channel = _UtilityFakeChannel()

            agg.process(channel, _msg(model_version=5, stat_utility=0.5), "t1", timestamp=0)

            import json
            events = [json.loads(l) for l in
                      (tmp_path / "aggregator.jsonl").read_text().splitlines()]
            ub = [e for e in events if e["event"] == "utility_belief"]
            assert ub[0]["staleness"] == 3  # 8 - 5
        finally:
            telemetry.shutdown()

    def test_carries_data_id_and_iteration_for_progress_key(self, tmp_path):
        telemetry.configure(role="aggregator", run_dir=str(tmp_path))
        try:
            agg = _UtilityFakeAggregator(data_id=42, iteration_per_data_id=2)
            channel = _UtilityFakeChannel()

            agg.process(channel, _msg(stat_utility=0.5), "t1", timestamp=0)

            import json
            events = [json.loads(l) for l in
                      (tmp_path / "aggregator.jsonl").read_text().splitlines()]
            ub = [e for e in events if e["event"] == "utility_belief"]
            assert ub[0]["data_id"] == 42
            assert ub[0]["iteration_per_data_id"] == 2
        finally:
            telemetry.shutdown()

    def test_noop_when_telemetry_disabled(self, tmp_path):
        assert not telemetry.is_enabled()
        agg = _UtilityFakeAggregator()
        channel = _UtilityFakeChannel()

        agg.process(channel, _msg(stat_utility=0.5), "t1", timestamp=0)

        assert not (tmp_path / "aggregator.jsonl").exists()


class TestStalenessPolicy:
    """staleness_policy gate in _process_single_trainer_message.

    REJECT policies (round_data_id/exact) drop a stale grad; ACCEPT policies
    (none/fedbuff) consume it -- fedbuff is the async baseline default so its
    carried surplus grads (commit-then-carry) are accepted + down-weighted by
    (V'-V) in aggregate_grads_from_trainers, never silently dropped."""

    def _run(self, policy, msg_version, agg_version=5):
        agg = _UtilityFakeAggregator(model_version=agg_version, is_async=True)
        agg.staleness_policy = policy
        channel = _UtilityFakeChannel()
        agg.process(channel, _msg(model_version=msg_version, stat_utility=0.5),
                    "t1", timestamp=0)
        return agg

    def test_fedbuff_accepts_stale_grad(self):
        # msg trained on v3, agg now at v5 -> stale by 2, but fedbuff ACCEPTS it.
        agg = self._run("fedbuff", msg_version=3)
        assert agg._agg_goal_cnt == 1          # grad consumed
        assert agg._per_agg_trainer_list == ["t1"]

    def test_none_accepts_stale_grad(self):
        agg = self._run("none", msg_version=3)
        assert agg._agg_goal_cnt == 1

    def test_round_data_id_rejects_stale_grad(self):
        agg = self._run("round_data_id", msg_version=3)
        assert agg._agg_goal_cnt == 0          # dropped
        assert agg._per_agg_trainer_list == []

    def test_fedbuff_accepts_fresh_grad(self):
        agg = self._run("fedbuff", msg_version=5)  # not stale
        assert agg._agg_goal_cnt == 1


class TestRoundCacheActivityResetOnContribution:
    """A real accepted contribution must restart the round-cache
    stuck-timeout clock (see TestStuckCachePruning in
    test_fwdllm_reselection.py) -- otherwise a trainer that eventually does
    respond, just slowly, would still get evicted next time the cache is
    checked."""

    def test_accepted_contribution_updates_activity_ts(self):
        agg = _UtilityFakeAggregator()
        channel = _UtilityFakeChannel()
        agg._round_cache_activity_ts["t1"] = 0.0  # ancient/never-set

        agg.process(channel, _msg(stat_utility=0.5), "t1", timestamp=0)

        assert agg._round_cache_activity_ts["t1"] > 0.0
