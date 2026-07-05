# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""fwdllm's trainer (FedSgdTrainer.py) reported only real_gpu_time_s, not
sim_round_duration_s -- unlike async_cifar10's trainer, which reports total
round wall time (gpu + modeled delay). fwdllm has no budget-vs-actual
contention model (its delay is a flat additive sleep, not a sleep-to-fill-
budget pattern), so only sim_round_duration_s is added here -- NOT
training_budget_s/overran/remaining_time_s, which would need a budget
concept fwdllm doesn't have (see ../../examples/MIGRATING_TO_LAUNCHER.md §9).

This covers _emulate_training_delay()'s return-value change: it now returns
the seconds actually slept (0.0 if delay emulation is disabled), which the
caller adds to real_gpu_time_s to report sim_round_duration_s.
"""

import os
import sys

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "examples", "fwdllm",
        "trainer", "forward_training",
    ),
)

import FedSgdTrainer as _fst_module  # noqa: E402
from FedSgdTrainer import FedSGDTrainer  # noqa: E402


class _FakeTrainer:
    """Minimal stand-in exposing only the state _emulate_training_delay
    touches; binds the real method under test."""

    _emulate_training_delay = FedSGDTrainer._emulate_training_delay
    _sim_straggler_offset_s = FedSGDTrainer._sim_straggler_offset_s

    def __init__(self, training_delay_enabled, training_delay_s=0.0,
                 training_delay_factor=1.0, speedup_factor=1.0, simulated=False):
        self.training_delay_enabled = training_delay_enabled
        self.training_delay_s = training_delay_s
        self.training_delay_factor = training_delay_factor
        self.speedup_factor = speedup_factor
        # Batch 1: real mode (simulated=False) sleeps D; sim mode skips the
        # sleep but still returns the same modeled D.
        self.simulated = simulated
        self.trainer_id = "t1"


class TestEmulateTrainingDelayReturnsSleptSeconds:
    def test_returns_zero_when_disabled(self):
        t = _FakeTrainer(training_delay_enabled="False", training_delay_s=10.0)
        assert t._emulate_training_delay() == 0.0

    def test_returns_computed_delay_when_enabled(self):
        t = _FakeTrainer(
            training_delay_enabled="True", training_delay_s=3.0,
            training_delay_factor=1.0, speedup_factor=1.0,
        )
        assert t._emulate_training_delay() == 3.0

    def test_speedup_factor_scales_the_returned_delay(self):
        """The returned value must match what was actually slept (eval_delay
        / speedup_factor), not the unscaled eval_delay -- otherwise
        sim_round_duration_s would overstate the real wall time under a
        speedup."""
        t = _FakeTrainer(
            training_delay_enabled="True", training_delay_s=10.0,
            training_delay_factor=2.0, speedup_factor=5.0,
        )
        # eval_delay = 10.0 / 2.0 = 5.0; slept = 5.0 / 5.0 = 1.0
        assert t._emulate_training_delay() == 1.0


class TestNoSleepOnSimPath:
    """Batch 1 core invariant: in simulated mode the trainer must NOT
    time.sleep to emulate the delay -- the aggregator advances a virtual clock
    instead -- yet must STILL return the same modeled D so the additive
    sim_round_duration_s = real_gpu + D is identical across modes."""

    def test_real_mode_sleeps_the_modeled_delay(self, monkeypatch):
        slept = []
        monkeypatch.setattr(_fst_module.time, "sleep", lambda s: slept.append(s))
        t = _FakeTrainer(
            training_delay_enabled="True", training_delay_s=4.0,
            training_delay_factor=2.0, speedup_factor=1.0, simulated=False,
        )
        d = t._emulate_training_delay()
        # eval_delay = 4.0/2.0 = 2.0; slept = 2.0/1.0 = 2.0
        assert d == 2.0
        assert slept == [2.0]

    def test_sim_mode_does_not_sleep_but_still_returns_delay(self, monkeypatch):
        slept = []
        monkeypatch.setattr(_fst_module.time, "sleep", lambda s: slept.append(s))
        t = _FakeTrainer(
            training_delay_enabled="True", training_delay_s=4.0,
            training_delay_factor=2.0, speedup_factor=1.0, simulated=True,
        )
        d = t._emulate_training_delay()
        assert d == 2.0          # same modeled D as real mode
        assert slept == []       # but NOTHING was slept on the sim path

    def test_sim_mode_disabled_returns_zero_no_sleep(self, monkeypatch):
        slept = []
        monkeypatch.setattr(_fst_module.time, "sleep", lambda s: slept.append(s))
        t = _FakeTrainer(training_delay_enabled="False", training_delay_s=9.0,
                         simulated=True)
        assert t._emulate_training_delay() == 0.0
        assert slept == []


class _FakeTime:
    """Scripted time source so the additive-stamp arithmetic is deterministic.
    Rebound only onto the FedSgdTrainer module's `time` name (not the shared
    time module), so timer_decorator's own runtime.time is untouched."""

    def __init__(self, ticks):
        self._ticks = list(ticks)
        self._i = 0

    def time(self):
        v = self._ticks[self._i]
        self._i = min(self._i + 1, len(self._ticks) - 1)
        return v

    def sleep(self, _s):  # must never be called on the sim path
        raise AssertionError("time.sleep called on the simulated path")


class _StampTrainer:
    """Binds the real train_with_data_id onto a minimal stand-in, stubbing the
    heavy compute so only the sim-stamp arithmetic is exercised."""

    train_with_data_id = FedSGDTrainer.train_with_data_id
    # train_with_data_id now folds the B2 straggler into the sct (#6/Root B);
    # no config -> spread 0 -> offset 0 -> additive duration preserved.
    _sim_straggler_offset_s = FedSGDTrainer._sim_straggler_offset_s

    def __init__(self, sim_send_ts, delay_d, leg_s=0.0):
        self.simulated = True
        self.abort_training = False
        self.trainer_id = "t1"
        self._round = 7
        self.data_id = 3
        self.iteration_per_data_id = 0
        self._sim_send_ts = sim_send_ts
        self.sim_completion_leg_s = leg_s
        self._sim_completion_ts = None
        self._sim_round_duration_s = None
        self._delay_d = delay_d

    def _check_availability(self):
        return True

    def _perform_training(self):
        pass  # no GPU work; wall time is scripted via _FakeTime

    def _emulate_training_delay(self):
        return self._delay_d  # modeled D (no sleep in sim mode)


class TestSimCompletionStampIsAdditive:
    """The sct the aggregator orders by must be ADDITIVE
    (sim_round_duration = real_gpu + D), matching fwdllm real mode's
    sleep-D-on-top-of-GPU semantics -- NOT cifar10's max(gpu, D). And
    _sim_completion_ts = _sim_send_ts + sim_round_duration + leg (K-D2)."""

    def test_additive_round_duration_and_completion_ts(self, monkeypatch):
        monkeypatch.setattr(_fst_module.telemetry, "is_enabled", lambda: False)
        # ticks: phase_entry=100.0, round_start=100.0, gpu-end=100.5 (+clamp)
        # -> real_gpu = 0.5s. (Stage A1 added the phase_entry tick.)
        monkeypatch.setattr(_fst_module, "time", _FakeTime([100.0, 100.0, 100.5]))
        t = _StampTrainer(sim_send_ts=10.0, delay_d=2.0, leg_s=0.0)

        t.train_with_data_id()

        # ADDITIVE: 0.5 (gpu) + 2.0 (D) = 2.5  (max(gpu,D) would be 2.0)
        assert t._sim_round_duration_s == 2.5
        # completion = send(10.0) + duration(2.5) + leg(0.0)
        assert t._sim_completion_ts == 12.5

    def test_completion_ts_includes_leg(self, monkeypatch):
        monkeypatch.setattr(_fst_module, "time", _FakeTime([100.0, 100.0, 100.5]))
        t = _StampTrainer(sim_send_ts=10.0, delay_d=2.0, leg_s=1.5)

        t.train_with_data_id()

        assert t._sim_round_duration_s == 2.5
        assert t._sim_completion_ts == 14.0  # 10.0 + 2.5 + 1.5
