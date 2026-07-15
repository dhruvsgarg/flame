# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""fwdllm's trainer sim-duration / delay model (K-D29 REMAINDER-WAIT, replacing
the earlier flat-additive model, K-D2).

The modeled mobile device takes ``_delay_s = training_delay_s/factor/speedup``.
On our GPU the forward pass takes ``gpu_time_s`` (SHOULD be << device time). So:
  - REAL mode sleeps only the remainder ``max(0, _delay_s - gpu)`` -> real wall
    ≈ _delay_s, GPU hidden inside it.
  - SIM mode skips the sleep; the sct round duration is ``max(gpu, _delay_s)``
    (NOT gpu + _delay_s). Per-trainer registry delays supply the completion
    SPREAD -> update order = delay order = deterministic + identical real↔sim.
  - OVERRUN: gpu > _delay_s => emulation unfaithful; flagged (remaining==0).

``_emulate_training_delay(gpu_time_s)`` returns
``(modeled_delay_s, remaining_s, overran)``.
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
from FedSgdTrainer import FedSGDTrainer, resolve_training_delay_s  # noqa: E402


class _FakeTrainer:
    """Minimal stand-in exposing only the state _emulate_training_delay
    touches; binds the real method under test."""

    _emulate_training_delay = FedSGDTrainer._emulate_training_delay

    def __init__(self, training_delay_enabled, training_delay_s=0.0,
                 training_delay_divisor=1.0, speedup_factor=1.0, simulated=False):
        self.training_delay_enabled = training_delay_enabled
        self.training_delay_s = training_delay_s
        self.training_delay_divisor = training_delay_divisor
        self.speedup_factor = speedup_factor
        self.simulated = simulated
        self.trainer_id = "t1"
        self.data_id = 3
        self.iteration_per_data_id = 0


class TestEmulateTrainingDelayRemainderWait:
    def test_returns_zero_tuple_when_disabled(self):
        t = _FakeTrainer(training_delay_enabled="False", training_delay_s=10.0)
        assert t._emulate_training_delay(0.2) == (0.0, 0.0, False)

    def test_modeled_delay_and_remainder_when_gpu_below_budget(self):
        # delay = 4.0/2.0/1.0 = 2.0; gpu = 0.5 -> remaining = 1.5, no overrun.
        t = _FakeTrainer(training_delay_enabled="True", training_delay_s=4.0,
                         training_delay_divisor=2.0, speedup_factor=1.0)
        modeled, remaining, overran = t._emulate_training_delay(0.5)
        assert modeled == 2.0 and remaining == 1.5 and overran is False

    def test_speedup_factor_scales_the_modeled_delay(self):
        # eval_delay = 10/2 = 5; modeled = 5/5 = 1.0; gpu 0.25 -> remaining 0.75.
        t = _FakeTrainer(training_delay_enabled="True", training_delay_s=10.0,
                         training_delay_divisor=2.0, speedup_factor=5.0)
        modeled, remaining, overran = t._emulate_training_delay(0.25)
        assert modeled == 1.0 and remaining == 0.75 and overran is False

    def test_overrun_when_gpu_exceeds_budget(self):
        # gpu 3.0 > budget 2.0 -> overran, remaining clamped to 0.
        t = _FakeTrainer(training_delay_enabled="True", training_delay_s=4.0,
                         training_delay_divisor=2.0, speedup_factor=1.0)
        modeled, remaining, overran = t._emulate_training_delay(3.0)
        assert modeled == 2.0 and remaining == 0.0 and overran is True


class TestSleepOnlyTheRemainderInRealMode:
    def test_real_mode_sleeps_the_remainder(self, monkeypatch):
        slept = []
        monkeypatch.setattr(_fst_module.time, "sleep", lambda s: slept.append(s))
        t = _FakeTrainer(training_delay_enabled="True", training_delay_s=4.0,
                         training_delay_divisor=2.0, speedup_factor=1.0,
                         simulated=False)
        modeled, remaining, _ = t._emulate_training_delay(0.5)
        assert modeled == 2.0 and remaining == 1.5
        assert slept == [1.5]          # ONLY the remainder, not the full delay

    def test_real_mode_overrun_sleeps_nothing(self, monkeypatch):
        slept = []
        monkeypatch.setattr(_fst_module.time, "sleep", lambda s: slept.append(s))
        t = _FakeTrainer(training_delay_enabled="True", training_delay_s=2.0,
                         training_delay_divisor=1.0, speedup_factor=1.0,
                         simulated=False)
        t._emulate_training_delay(5.0)  # gpu > budget
        assert slept == []             # nothing to sleep; overran

    def test_sim_mode_does_not_sleep(self, monkeypatch):
        slept = []
        monkeypatch.setattr(_fst_module.time, "sleep", lambda s: slept.append(s))
        t = _FakeTrainer(training_delay_enabled="True", training_delay_s=4.0,
                         training_delay_divisor=2.0, speedup_factor=1.0,
                         simulated=True)
        modeled, remaining, _ = t._emulate_training_delay(0.5)
        assert modeled == 2.0 and remaining == 1.5   # same modeled math as real
        assert slept == []                            # but NOTHING slept in sim

    def test_disabled_sim_no_sleep(self, monkeypatch):
        slept = []
        monkeypatch.setattr(_fst_module.time, "sleep", lambda s: slept.append(s))
        t = _FakeTrainer(training_delay_enabled="False", training_delay_s=9.0,
                         simulated=True)
        assert t._emulate_training_delay(0.1) == (0.0, 0.0, False)
        assert slept == []


class _FakeTime:
    """Scripted time source so the sct arithmetic is deterministic. Rebound only
    onto FedSgdTrainer's `time` name (not the shared module)."""

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
    heavy compute so only the sim-stamp arithmetic is exercised. No config ->
    straggler spread 0 -> offset 0."""

    train_with_data_id = FedSGDTrainer.train_with_data_id
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

    def _emulate_training_delay(self, gpu_time_s=0.0):
        # modeled D, remainder (irrelevant in sim), no overrun
        return self._delay_d, max(0.0, self._delay_d - gpu_time_s), False


class TestSimCompletionStampIsMaxGpuDelay:
    """K-D29: the sct the aggregator orders by is ``max(gpu, D)`` (the mobile
    device wall, GPU hidden inside), NOT the old additive gpu + D, and
    _sim_completion_ts = _sim_send_ts + max(gpu, D) + leg."""

    def test_max_round_duration_and_completion_ts(self, monkeypatch):
        monkeypatch.setattr(_fst_module.telemetry, "is_enabled", lambda: False)
        # real_gpu = 0.5s; D = 2.0 -> max(0.5, 2.0) = 2.0 (additive would be 2.5).
        monkeypatch.setattr(_fst_module, "time", _FakeTime([100.0, 100.0, 100.5]))
        t = _StampTrainer(sim_send_ts=10.0, delay_d=2.0, leg_s=0.0)
        t.train_with_data_id()
        assert t._sim_round_duration_s == 2.0
        assert t._sim_completion_ts == 12.0  # 10.0 + 2.0 + 0.0

    def test_gpu_dominates_when_over_budget(self, monkeypatch):
        monkeypatch.setattr(_fst_module.telemetry, "is_enabled", lambda: False)
        # real_gpu = 3.0s; D = 2.0 -> max = 3.0 (the overrun case).
        monkeypatch.setattr(_fst_module, "time", _FakeTime([100.0, 100.0, 103.0]))
        t = _StampTrainer(sim_send_ts=10.0, delay_d=2.0, leg_s=0.0)
        t.train_with_data_id()
        assert t._sim_round_duration_s == 3.0
        assert t._sim_completion_ts == 13.0

    def test_completion_ts_includes_leg(self, monkeypatch):
        monkeypatch.setattr(_fst_module.telemetry, "is_enabled", lambda: False)
        monkeypatch.setattr(_fst_module, "time", _FakeTime([100.0, 100.0, 100.5]))
        t = _StampTrainer(sim_send_ts=10.0, delay_d=2.0, leg_s=1.5)
        t.train_with_data_id()
        assert t._sim_round_duration_s == 2.0
        assert t._sim_completion_ts == 13.5  # 10.0 + 2.0 + 1.5


class TestResolveTrainingDelayS:
    """FWDLLM_DESIGN.md §O: floor the RAW registry delay before it's divided
    by training_delay_factor, so a trainer at/near the registry's class floor
    doesn't get a razor-thin (or negative-margin) budget once divided."""

    def test_no_floor_is_byte_identical(self):
        assert resolve_training_delay_s(2.0, 0.0) == 2.0
        assert resolve_training_delay_s(2.0, None) == 2.0

    def test_floor_raises_a_trainer_below_it(self):
        # fluxtune's floor (FWDLLM_DESIGN.md §O): a delay=2.0 (class floor)
        # trainer gets bumped to 7.0, not left at 2.0.
        assert resolve_training_delay_s(2.0, 7.0) == 7.0

    def test_floor_never_lowers_a_trainer_above_it(self):
        # a very_slow trainer (e.g. raw delay 20.0) is unaffected by a 7.0 floor.
        assert resolve_training_delay_s(20.0, 7.0) == 20.0

    def test_floor_at_exact_boundary_is_a_no_op(self):
        assert resolve_training_delay_s(7.0, 7.0) == 7.0
