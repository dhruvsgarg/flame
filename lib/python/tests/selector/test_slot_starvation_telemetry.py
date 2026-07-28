# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""D-10 (simulate_fwdllm.md, 2026-07-27): round cadence's pinned cohort can
run out of not-yet-contributed-to-this-version_key candidates before its
`agg_goal` batch closes, leaving a freed dispatch slot unfilled even though
`extra > 0`. `_handle_send_state` already computed the ingredients
(`extra`/`filtered_ends`/`feasible_extra`) but never surfaced when they
diverge. This covers the new `slot_starvation` telemetry event, emitted only
on a starved tick (`feasible_extra < extra`) to stay low-volume."""

from flame import telemetry
from flame.selector.async_oort import AsyncOortSelector


def _events(tmp_path, event_name):
    import json
    path = tmp_path / "aggregator.jsonl"
    if not path.exists():
        return []
    lines = path.read_text().splitlines()
    return [e for e in (json.loads(l) for l in lines) if e["event"] == event_name]


class TestSlotStarvationTelemetry:
    def test_starved_tick_emits_event(self, tmp_path, make_ends, channel_props):
        """5 open slots, only 2 eligible candidates -> starved by 3."""
        sel = AsyncOortSelector(
            c=30, aggGoal=10, evalGoalFactor=0.5,
            roundNudgeType="last_train", selectType="default",
        )
        sel.requester = "agg"
        sel.selected_ends = {"agg": set()}
        telemetry.configure(role="aggregator", run_dir=str(tmp_path))
        try:
            ends = make_ends(["t1", "t2"])
            sel._handle_send_state(
                ends=ends, concurrency=5, channel_props=channel_props,
                trainer_unavail_list=[], task_to_perform="train",
            )
            evs = _events(tmp_path, "slot_starvation")
            assert len(evs) == 1
            assert evs[0]["extra"] == 5
            assert evs[0]["n_filtered"] == 2
            assert evs[0]["feasible_extra"] == 2
            assert evs[0]["starved"] == 3
            assert evs[0]["concurrency"] == 5
        finally:
            telemetry.shutdown()

    def test_fully_filled_tick_emits_nothing(self, tmp_path, make_ends, channel_props):
        """2 open slots, 5 eligible candidates -> no starvation."""
        sel = AsyncOortSelector(
            c=30, aggGoal=10, evalGoalFactor=0.5,
            roundNudgeType="last_train", selectType="default",
        )
        sel.requester = "agg"
        sel.selected_ends = {"agg": set()}
        telemetry.configure(role="aggregator", run_dir=str(tmp_path))
        try:
            ends = make_ends(["t1", "t2", "t3", "t4", "t5"])
            sel._handle_send_state(
                ends=ends, concurrency=2, channel_props=channel_props,
                trainer_unavail_list=[], task_to_perform="train",
            )
            assert _events(tmp_path, "slot_starvation") == []
        finally:
            telemetry.shutdown()

    def test_noop_when_telemetry_disabled(self, make_ends, channel_props):
        """Must not raise when telemetry.configure() was never called."""
        assert not telemetry.is_enabled()
        sel = AsyncOortSelector(
            c=30, aggGoal=10, evalGoalFactor=0.5,
            roundNudgeType="last_train", selectType="default",
        )
        sel.requester = "agg"
        sel.selected_ends = {"agg": set()}
        ends = make_ends(["t1"])
        sel._handle_send_state(
            ends=ends, concurrency=5, channel_props=channel_props,
            trainer_unavail_list=[], task_to_perform="train",
        )  # no assertion needed -- just must not raise
