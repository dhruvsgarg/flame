# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Phase 2 (speedup leak, root #13): the simulated clock must NOT pay
real-transport wall waits that have no fidelity value -- the sim's job is to
advance the virtual clock FASTER than physical wall (sim_rate >= 1), so every
skippable per-round sleep on the sim critical path is a slowdown.

Telemetry localization (banked run_20260704_134801 sim): the trainer's
inter-round `mqtt_fetch_s` is barrier-wait realized by the blocking recv in
_fetch_weights (which MUST stay -- it delivers the real weights the forward-grad
pass needs for grad mode-invariance; async_cifar10 keeps real MQTT recv in sim
too). The only additive, fidelity-free per-round wall the sim can skip is:

  (1) the trainer's `pause_execution` throttle -- formerly a `time.sleep(1)`
      chained at the tail of EVERY trainer loop iteration ("don't overwhelm
      mqtt", a real-transport artifact, principle #8). Now REMOVED entirely
      (§H): the blocking recv already paces the loop, and with the aggregator's
      one-instruction-per-version_key dedup there is no VAR=bad backlog to drain,
      so the sleep only added real-only latency (stacking one sleep per queued
      stale message for a busy straggler). pause_execution is now a no-op; and
  (2) the `_check_availability` avail-spin -- a `while UN_AVL: time.sleep(1)`
      that, in sim, would freeze the virtual clock (sim time can't advance while
      a trainer blocks on time.sleep); sim availability is enforced agg-side.

(2) is gated so REAL is byte-identical and SIM never wall-sleeps; (1) no longer
wall-sleeps in either mode.
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
from flame.config import TrainerAvailState  # noqa: E402
from flame.mode.horizontal.syncfl import fwdllm_trainer as _tr_module  # noqa: E402


class _PauseTrainer:
    """Minimal stand-in binding the base Trainer.pause_execution under test.
    timer_decorator reads its own runtime.time (untouched here) and only a
    best-effort self.fwd_llm_stage, so no other state is needed."""

    pause_execution = _tr_module.Trainer.pause_execution

    def __init__(self, simulated):
        self.simulated = simulated


class TestPauseExecutionIsNoOp:
    """pause_execution is a no-op in BOTH modes now (§H): the post-put 1s throttle
    was removed, so neither real nor sim wall-sleeps in it."""

    def test_real_mode_does_not_pause(self, monkeypatch):
        slept = []
        monkeypatch.setattr(_tr_module.time, "sleep", lambda s: slept.append(s))
        _PauseTrainer(simulated=False).pause_execution()
        assert slept == []  # throttle removed: no real-only per-round wall

    def test_sim_mode_does_not_pause(self, monkeypatch):
        slept = []
        monkeypatch.setattr(_tr_module.time, "sleep", lambda s: slept.append(s))
        _PauseTrainer(simulated=True).pause_execution()
        assert slept == []  # root #13: no per-round wall charged to the sim

    def test_missing_attr_still_no_pause(self, monkeypatch):
        """No-op regardless of the `simulated` attr -- the removed throttle can
        never re-appear on any code path."""
        slept = []
        monkeypatch.setattr(_tr_module.time, "sleep", lambda s: slept.append(s))
        t = _PauseTrainer(simulated=False)
        del t.simulated
        t.pause_execution()
        assert slept == []


class _AvailTrainer:
    """Binds the real _check_availability onto a minimal stand-in."""

    _check_availability = FedSGDTrainer._check_availability

    def __init__(self, simulated, avl_state, wait_until_next_avl=True):
        self.simulated = simulated
        self.avl_state = avl_state
        self.wait_until_next_avl = wait_until_next_avl
        self.trainer_id = "t1"


class TestCheckAvailabilityGatedInSim:
    def test_sim_never_spins_and_proceeds(self, monkeypatch):
        """UN_AVL trainer in sim: must NOT time.sleep (would freeze the vclock)
        and must return True so it proceeds -- the agg-side send-gate handles
        the withhold."""
        def _boom(_s):
            raise AssertionError("time.sleep called on the simulated avail path")
        monkeypatch.setattr(_fst_module.time, "sleep", _boom)
        t = _AvailTrainer(simulated=True, avl_state=TrainerAvailState.UN_AVL)
        assert t._check_availability() is True

    def test_real_mode_still_spins_until_available(self, monkeypatch):
        """Real mode is byte-identical: it wall-sleeps while UN_AVL and returns
        True once the trainer flips back to AVL_TRAIN."""
        t = _AvailTrainer(simulated=False, avl_state=TrainerAvailState.UN_AVL)
        slept = []

        def _sleep_then_avail(s):
            slept.append(s)
            t.avl_state = TrainerAvailState.AVL_TRAIN  # becomes available after 1 tick

        monkeypatch.setattr(_fst_module.time, "sleep", _sleep_then_avail)
        assert t._check_availability() is True
        assert slept == [1]

    def test_real_unavailable_no_wait_exits(self, monkeypatch):
        """wait_until_next_avl=False is unchanged in both modes: skip training."""
        monkeypatch.setattr(_fst_module.time, "sleep",
                            lambda s: (_ for _ in ()).throw(AssertionError("no sleep")))
        t = _AvailTrainer(simulated=False, avl_state=TrainerAvailState.UN_AVL,
                          wait_until_next_avl=False)
        assert t._check_availability() is False
