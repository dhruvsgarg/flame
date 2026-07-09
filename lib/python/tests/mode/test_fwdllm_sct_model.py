# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Stage B -- sct-model folds (K-D20 #6), all config-gated OFF => byte-identical.

B1  eval_s onto the vclock (aggregator): after a committed data_id's eval,
    advance the vclock by the measured eval wall when simModelEvalTime is set.
B2  per-trainer straggler spread (trainer): a stable offset in [0, spread) added
    to the modeled delay in SIM only.
B3  WAN transfer knob: documented, default 0 (verified inert here).

Only the MECHANISM + the flag-off byte-identical invariant is unit-tested;
whether the folds drive wall_disparity->~0 is an EMERGENT run quantity.
"""

import os
import sys
from types import SimpleNamespace

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "examples", "fwdllm",
        "trainer", "forward_training",
    ),
)
from FedSgdTrainer import FedSGDTrainer  # noqa: E402


class _Host:
    """Binds the straggler-offset helper onto a minimal stand-in."""

    _sim_straggler_offset_s = FedSGDTrainer._sim_straggler_offset_s

    def __init__(self, trainer_id="3", simulated=True, spread=0.0):
        self.trainer_id = trainer_id
        self.simulated = simulated
        self.config = SimpleNamespace(
            hyperparameters=SimpleNamespace(sim_straggler_spread_s=spread)
        )


class TestStragglerSpreadB2:
    def test_zero_spread_is_no_offset(self):
        assert _Host(spread=0.0)._sim_straggler_offset_s() == 0.0

    def test_real_mode_never_offsets(self):
        # real mode gets dispersion from GPU contention, not this knob
        assert _Host(simulated=False, spread=5.0)._sim_straggler_offset_s() == 0.0

    def test_offset_within_spread_and_stable(self):
        h = _Host(trainer_id="7", spread=2.3)
        o1 = h._sim_straggler_offset_s()
        o2 = h._sim_straggler_offset_s()
        assert 0.0 <= o1 < 2.3
        assert o1 == o2  # deterministic per trainer (reproducible)

    def test_offset_varies_across_trainers(self):
        offs = {
            _Host(trainer_id=str(i), spread=2.3)._sim_straggler_offset_s()
            for i in range(1, 11)
        }
        # crc32-derived fractions spread the cohort (not all identical)
        assert len(offs) > 1


class TestEvalOnVclockB1:
    """B1 mechanism: the aggregator advances the vclock by the measured eval
    wall only when simModelEvalTime is set (byte-identical off)."""

    def _run(self, flag):
        from flame import telemetry
        from tests.mode.test_fwdllm_agg_telemetry import (
            _FakeAggregator, _FakeChannel,
        )
        from datetime import timedelta

        agg = _FakeAggregator(contributors=["t1"], var_good_enough=True)
        agg.simulated = True
        # sim path calls the boundary slot-release hook; not under test here.
        agg._release_sim_slots_at_agg_goal = lambda *a, **k: None
        agg._vclock = SimpleNamespace(
            now=100.0,
            advance=lambda ts: setattr(agg._vclock, "now", max(agg._vclock.now, ts)),
        )
        agg.config.hyperparameters.sim_model_eval_time = flag
        # eval_model in the fake returns instantly, so measured eval_s ~ 0; force
        # a nonzero measured eval by making eval_model sleep a hair.
        import time as _t
        orig_eval = agg.eval_model

        def _slow_eval():
            _t.sleep(0.02)
            return orig_eval()

        agg.eval_model = _slow_eval
        channel = _FakeChannel(durations={"t1": timedelta(seconds=2)})
        agg._process_aggregation_goal_met(tag="aggregate", channel=channel)
        return agg._vclock.now

    def test_flag_off_does_not_advance_vclock(self):
        assert self._run(flag=False) == 100.0  # byte-identical: vclock untouched

    def test_flag_on_charges_eval_wall(self):
        assert self._run(flag=True) > 100.0  # vclock advanced by measured eval_s
