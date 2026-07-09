# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Real-mode sync-barrier liveness under availability scarcity.

`_await_dispatchable_under_scarcity` keeps the cohort == `agg_goal` and
sleep-to-next-avail (parity-faithful, matches the sim vclock-jump) instead of
hot-spin-dispatching when a trace keeps the eligible pool below `agg_goal`.

Invariants under test:
- no-op on the sim path and when availability tracking is off (byte-identical),
- no-op during startup join-lag (fewer than `agg_goal` trainers joined),
- returns immediately when an available un-contributed trainer exists,
- waits (bounded) while the pool is scarce, then proceeds once it recovers,
- self-terminates via `_check_early_stop_conditions` at the wall budget.
"""

from types import SimpleNamespace

import pytest

from flame.mode.horizontal.syncfl.fwdllm_aggregator import TopAggregator


class _FakeAggregator:
    """Minimal stand-in exposing only what the scarcity gate touches."""

    def __init__(
        self,
        *,
        simulated=False,
        trainer_event_dict=None,
        all_trainers=None,
        agg_goal=10,
        agg_goal_cnt=0,
        per_agg_trainer_list=None,
        unavail_sequence=None,
    ):
        self.simulated = simulated
        self.trainer_event_dict = trainer_event_dict
        self.all_trainers = set(all_trainers or [])
        self._agg_goal = agg_goal
        self._agg_goal_cnt = agg_goal_cnt
        self._per_agg_trainer_list = list(per_agg_trainer_list or [])
        self._round = 0
        self.data_id = 0
        self._work_done = False
        self.config = SimpleNamespace(hyperparameters=SimpleNamespace(scarcity_poll_s=0.0))
        # Each poll pops the next unavailable-set from this sequence (the last
        # one repeats), letting a test model availability recovering over time.
        self._unavail_sequence = list(unavail_sequence or [[]])
        self.slept = 0
        self.stop_checks = 0

    def get_curr_unavail_trainers(self):
        if len(self._unavail_sequence) > 1:
            return self._unavail_sequence.pop(0)
        return self._unavail_sequence[0]

    def _check_early_stop_conditions(self):
        self.stop_checks += 1

    _await_dispatchable_under_scarcity = (
        TopAggregator._await_dispatchable_under_scarcity
    )


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    import flame.mode.horizontal.syncfl.fwdllm_aggregator as mod

    def _count_sleep(_s):
        _count_sleep.calls += 1

    _count_sleep.calls = 0
    monkeypatch.setattr(mod.time, "sleep", _count_sleep)
    return _count_sleep


class TestScarcityWaitNoOp:
    def test_sim_path_never_waits(self, _no_real_sleep):
        agg = _FakeAggregator(
            simulated=True,
            trainer_event_dict={"1": []},
            all_trainers=[str(i) for i in range(10)],
            unavail_sequence=[[str(i) for i in range(10)]],  # all unavail
        )
        agg._await_dispatchable_under_scarcity("train")
        assert _no_real_sleep.calls == 0

    def test_tracking_off_never_waits(self, _no_real_sleep):
        agg = _FakeAggregator(
            trainer_event_dict=None,  # availability tracking disabled
            all_trainers=[str(i) for i in range(10)],
            unavail_sequence=[[str(i) for i in range(10)]],
        )
        agg._await_dispatchable_under_scarcity("train")
        assert _no_real_sleep.calls == 0

    def test_join_lag_does_not_wait(self, _no_real_sleep):
        # Only 3 of the 10 have joined -> startup, not scarcity: do not block.
        agg = _FakeAggregator(
            trainer_event_dict={"1": []},
            all_trainers=["0", "1", "2"],
            agg_goal=10,
            unavail_sequence=[["0", "1", "2"]],
        )
        agg._await_dispatchable_under_scarcity("train")
        assert _no_real_sleep.calls == 0

    def test_available_trainer_returns_without_wait(self, _no_real_sleep):
        agg = _FakeAggregator(
            trainer_event_dict={"1": []},
            all_trainers=[str(i) for i in range(10)],
            unavail_sequence=[["0", "1", "2"]],  # 7 available, none contributed
        )
        agg._await_dispatchable_under_scarcity("train")
        assert _no_real_sleep.calls == 0

    def test_barrier_already_met_returns(self, _no_real_sleep):
        agg = _FakeAggregator(
            trainer_event_dict={"1": []},
            all_trainers=[str(i) for i in range(10)],
            agg_goal=3,
            agg_goal_cnt=3,  # cohort complete
            unavail_sequence=[[str(i) for i in range(10)]],
        )
        agg._await_dispatchable_under_scarcity("train")
        assert _no_real_sleep.calls == 0


class TestScarcityWaitEngages:
    def test_waits_then_proceeds_when_availability_recovers(self, _no_real_sleep):
        allt = [str(i) for i in range(10)]
        # Scarce for two polls (all unavail), then trainer "5" comes back.
        agg = _FakeAggregator(
            trainer_event_dict={"1": []},
            all_trainers=allt,
            unavail_sequence=[list(allt), list(allt), [x for x in allt if x != "5"]],
        )
        agg._await_dispatchable_under_scarcity("train")
        assert _no_real_sleep.calls == 2  # slept exactly across the two scarce polls
        assert agg.stop_checks == 2  # re-checked early-stop each scarce poll

    def test_already_contributed_pool_still_scarce(self, _no_real_sleep):
        # The available trainers have ALL already contributed this cycle, and the
        # one still-uncontributed trainer ("5") is unavailable -> no dispatchable
        # end -> must wait, not spin-redispatch the already-used ones. On the next
        # poll "5" recovers -> dispatchable -> proceed.
        allt = [str(i) for i in range(10)]
        used = [x for x in allt if x != "5"]  # the 9 available already contributed
        agg = _FakeAggregator(
            trainer_event_dict={"1": []},
            all_trainers=allt,
            agg_goal=10,
            agg_goal_cnt=9,
            per_agg_trainer_list=used,
            unavail_sequence=[["5"], []],  # "5" unavail, then recovers
        )
        agg._await_dispatchable_under_scarcity("train")
        assert _no_real_sleep.calls == 1  # waited one poll, then "5" recovered

    def test_self_terminates_at_budget(self, _no_real_sleep):
        allt = [str(i) for i in range(10)]

        agg = _FakeAggregator(
            trainer_event_dict={"1": []},
            all_trainers=allt,
            unavail_sequence=[list(allt)],  # permanently scarce
        )

        # Model the wall budget firing after 3 polls.
        orig = agg._check_early_stop_conditions

        def _stop_after_3():
            orig()
            if agg.stop_checks >= 3:
                agg._work_done = True

        agg._check_early_stop_conditions = _stop_after_3
        agg._await_dispatchable_under_scarcity("train")
        assert agg._work_done is True
        assert _no_real_sleep.calls == 3  # bounded, then stopped
