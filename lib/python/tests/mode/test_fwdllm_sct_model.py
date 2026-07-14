# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Stage B -- sct-model folds (K-D20 #6), all config-gated OFF => byte-identical.

B1  (REMOVED, §6 Part 6, simulate_fwdllm.md §G) used to advance the
    vclock by the measured eval wall when simModelEvalTime was set, after a
    committed data_id's synchronous eval. Removed once eval_model() was
    backgrounded on a daemon thread (mirroring async_cifar10's evaluate()) --
    the asymmetry the fold corrected for (real paid the synchronous eval wall,
    sim didn't) no longer exists once neither mode pays it on the critical
    path. TestEvalNoLongerFoldsVclock below is the regression guard: eval must
    never advance the vclock again, regardless of any legacy config.
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


class TestEvalNoLongerFoldsVclock:
    """§6 Part 6 (simulate_fwdllm.md §G) regression guard: eval_model()
    is now backgrounded on a daemon thread (mirroring async_cifar10's
    evaluate()), so it must NEVER advance the vclock, regardless of how slow the
    (backgrounded) eval actually is or what any legacy config says -- the
    sim_model_eval_time fold this class used to test was removed because the
    asymmetry it corrected for (real paid the synchronous eval wall, sim didn't)
    no longer exists once neither mode pays it on the critical path."""

    def _run(self, legacy_flag_value=None):
        from flame import telemetry
        from tests.mode.test_fwdllm_agg_telemetry import (
            _FakeAggregator, _FakeChannel, _wait_eval_done,
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
        # The field no longer exists in config.py; a stray/legacy value on the
        # hyperparameters object (e.g. from an un-migrated yaml) must still be
        # inert -- nothing in the aggregator reads this attribute anymore.
        if legacy_flag_value is not None:
            agg.config.hyperparameters.sim_model_eval_time = legacy_flag_value
        # eval_model in the fake returns instantly; force a nonzero measured
        # eval wall to prove a slow (backgrounded) eval still can't reach the
        # vclock -- if it ever did, this would have caught it immediately.
        import time as _t
        orig_eval = agg.eval_model

        def _slow_eval(model=None):
            _t.sleep(0.02)
            return orig_eval(model=model)

        agg.eval_model = _slow_eval
        channel = _FakeChannel(durations={"t1": timedelta(seconds=2)})
        agg._process_aggregation_goal_met(tag="aggregate", channel=channel)
        _wait_eval_done(agg)
        return agg._vclock.now

    def test_no_legacy_flag_vclock_untouched(self):
        assert self._run() == 100.0

    def test_stray_legacy_flag_true_still_inert(self):
        assert self._run(legacy_flag_value=True) == 100.0

    def test_stray_legacy_flag_false_still_inert(self):
        assert self._run(legacy_flag_value=False) == 100.0


class TestAggComputeOnVclock15:
    """#15 mechanism: the aggregator advances the vclock by the measured
    aggregate()-call wall only when simModelAggComputeTime is set (byte-
    identical off). Unlike B1 (eval_s), this fires on EVERY cycle -- pass or
    fail -- since aggregate() itself always runs."""

    def _run(self, flag, var_good_enough=True):
        from tests.mode.test_fwdllm_agg_telemetry import (
            _FakeAggregator, _FakeChannel,
        )
        from datetime import timedelta
        import time as _t

        agg = _FakeAggregator(contributors=["t1"], var_good_enough=var_good_enough)
        agg.simulated = True
        agg._release_sim_slots_at_agg_goal = lambda *a, **k: None
        agg._vclock = SimpleNamespace(
            now=100.0,
            advance=lambda ts: setattr(agg._vclock, "now", max(agg._vclock.now, ts)),
        )
        agg.config.hyperparameters.sim_model_agg_compute_time = flag
        agg.config.hyperparameters.sim_model_eval_time = False  # isolate #15 from B1
        orig_aggregate = agg.aggregate

        def _slow_aggregate(round_num):
            _t.sleep(0.02)
            return orig_aggregate(round_num)

        agg.aggregate = _slow_aggregate
        channel = _FakeChannel(durations={"t1": timedelta(seconds=2)})
        agg._process_aggregation_goal_met(tag="aggregate", channel=channel)
        return agg._vclock.now

    def test_flag_off_does_not_advance_vclock(self):
        assert self._run(flag=False) == 100.0  # byte-identical: vclock untouched

    def test_flag_on_charges_aggregate_wall(self):
        assert self._run(flag=True) > 100.0  # vclock advanced by measured aggregate_s

    def test_flag_on_charges_aggregate_wall_even_on_variance_fail(self):
        # aggregate() runs regardless of the variance-gate outcome, so the fold
        # must too -- this is what distinguishes #15 from the committed-only B1.
        assert self._run(flag=True, var_good_enough=False) > 100.0
