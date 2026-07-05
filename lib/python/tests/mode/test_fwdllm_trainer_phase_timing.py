# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Stage A1 -- trainer per-phase wall timing.

The fwdllm trainer emitted no per-phase breakdown, so the 8 phase rungs
(mqtt_fetch_s / weights_to_ram_s / weights_to_gpu_s / pre_train_s /
gpu_compute_s / post_train_s / training_budget_s / trainer_phase) SKIPed. This
covers:
- the `_phase` context manager accumulates into `_phase_times` (fwdllm_trainer),
- `FedSgdTrainer.train_with_data_id` drains those + the pre/gpu/post/budget/
  phase terms into the emitted `trainer_round` event.
"""

import json
import os
import sys

import pytest

from flame import telemetry
from flame.mode.horizontal.syncfl.fwdllm_trainer import Trainer as FwdLLMTrainer

_EXAMPLE = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "examples", "fwdllm", "trainer", "forward_training",
)
sys.path.insert(0, os.path.abspath(_EXAMPLE))
import FedSgdTrainer as _fedsgd_mod  # noqa: E402

FedSgdTrainer = _fedsgd_mod.FedSGDTrainer

_PHASE_KEYS = [
    "mqtt_fetch_s",
    "weights_to_ram_s",
    "weights_to_gpu_s",
    "pre_train_s",
    "gpu_compute_s",
    "post_train_s",
    "training_budget_s",
    "trainer_phase",
]


class _PhaseHost:
    """Minimal host for the (abstract) fwdllm Trainer's _phase method."""

    _phase = FwdLLMTrainer._phase

    def __init__(self):
        self._phase_times = {}


class TestPhaseContextManager:
    def test_accumulates(self):
        t = _PhaseHost()
        with t._phase("gpu_compute_s"):
            pass
        with t._phase("gpu_compute_s"):
            pass
        assert "gpu_compute_s" in t._phase_times
        # two entries under one name accumulate, not overwrite
        assert t._phase_times["gpu_compute_s"] >= 0.0
        assert isinstance(t._phase_times["gpu_compute_s"], float)


class _AvlState:
    value = "AVL_TRAIN"


class _FakeFedSgd:
    """Binds the real train_with_data_id onto a stand-in that stubs only the
    heavy compute (perform_training / delay / availability); the phase-timing
    + telemetry-emit logic under test runs for real."""

    train_with_data_id = FedSgdTrainer.train_with_data_id
    # train_with_data_id now folds the B2 straggler into the sct (#6/Root B);
    # no config -> spread 0 -> offset 0 -> phase/duration values preserved.
    _sim_straggler_offset_s = FedSgdTrainer._sim_straggler_offset_s

    def __init__(self, phase_times=None, delay_s=1.5):
        self._round = 7
        self.data_id = 3
        self.iteration_per_data_id = 2
        self._model_version = 5
        self.trainer_id = "t1"
        self.simulated = False
        self.abort_training = False
        self.avl_state = _AvlState()
        self.dataset_size = 128
        self._stat_utility = 0.9
        self._sim_send_ts = None
        self.sim_completion_leg_s = 0.0
        self._sim_completion_ts = None
        self._sim_round_duration_s = None
        # Populated by _fetch_weights in a live run; pre-seeded here.
        self._phase_times = dict(phase_times or {})
        self._delay_s = delay_s

    def _check_availability(self):
        return True

    def _perform_training(self):
        pass

    def _emulate_training_delay(self, gpu_time_s=0.0):
        # K-D29 remainder-wait signature: (modeled_delay, remaining, overran).
        return self._delay_s, max(0.0, self._delay_s - gpu_time_s), False


class TestTrainWithDataIdEmitsPhases:
    def test_phase_fields_present(self, tmp_path):
        telemetry.configure(role="trainer", run_dir=str(tmp_path))
        try:
            t = _FakeFedSgd(
                phase_times={
                    "mqtt_fetch_s": 0.4,
                    "weights_to_ram_s": 0.2,
                    "weights_to_gpu_s": 0.1,
                }
            )
            t.train_with_data_id()

            lines = (tmp_path / "trainer.jsonl").read_text().splitlines()
            events = [json.loads(l) for l in lines]
            rounds = [e for e in events if e["event"] == "trainer_round"]
            assert len(rounds) == 1
            ev = rounds[0]
            for k in _PHASE_KEYS:
                assert k in ev, f"missing phase field {k}"
            # the three _fetch_weights phases flowed through verbatim
            assert ev["mqtt_fetch_s"] == 0.4
            assert ev["weights_to_ram_s"] == 0.2
            assert ev["weights_to_gpu_s"] == 0.1
            # training_budget_s is the modeled additive delay
            assert ev["training_budget_s"] == 1.5
            # post_train excludes the delay (stamped after it) -> pure post-proc.
            assert ev["post_train_s"] >= 0.0 and ev["post_train_s"] < 0.5
            # trainer_phase encodes round/data_id/iteration identity
            assert ev["trainer_phase"] == "7/3/2"
            # pre/post are real non-negative wall slivers
            assert ev["pre_train_s"] >= 0.0
            assert ev["post_train_s"] >= 0.0
        finally:
            telemetry.shutdown()

    def test_straggler_in_sct_not_in_training_budget(self, tmp_path):
        """#6/Root B: the B2 straggler spread is folded into the sct
        (sim_round_duration_s) but NOT into the emitted training_budget_s -- so
        training_budget stays a mode-invariant INPUT (T2 passes) while the sync
        barrier still gets its per-trainer dispersion."""
        from types import SimpleNamespace
        telemetry.configure(role="trainer", run_dir=str(tmp_path))
        try:
            t = _FakeFedSgd(delay_s=1.5)
            t.simulated = True
            t._sim_send_ts = 100.0
            t.config = SimpleNamespace(
                hyperparameters=SimpleNamespace(sim_straggler_spread_s=0.9))
            offset = t._sim_straggler_offset_s()
            assert offset > 0.0, "t1 should have a non-zero stable offset"
            t.train_with_data_id()

            ev = [json.loads(l) for l in
                  (tmp_path / "trainer.jsonl").read_text().splitlines()
                  if json.loads(l)["event"] == "trainer_round"][0]
            # training_budget = the BASE delay; straggler EXCLUDED.
            assert ev["training_budget_s"] == 1.5
            # sct carries the straggler: duration - budget == gpu(~0) + offset.
            assert (ev["sim_round_duration_s"] - ev["training_budget_s"]) == \
                pytest.approx(offset, abs=0.05)
        finally:
            telemetry.shutdown()

    def test_aborted_round_emits_nothing(self, tmp_path):
        telemetry.configure(role="trainer", run_dir=str(tmp_path))
        try:
            t = _FakeFedSgd()
            t.abort_training = True
            t.train_with_data_id()
            assert not (tmp_path / "trainer.jsonl").exists() or not (
                tmp_path / "trainer.jsonl"
            ).read_text().strip()
        finally:
            telemetry.shutdown()
