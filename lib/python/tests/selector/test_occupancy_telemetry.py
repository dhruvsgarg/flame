# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""FELIX_INSIGHTS §1 / I-F1: async selections report in-flight by task and train slots held by eval tasks."""

import time

from flame.selector.async_base import SelectContext
from flame.selector.async_oort import AsyncOortSelector


def _sel():
    sel = AsyncOortSelector(c=4, aggGoal=3, evalGoalFactor=1.0, roundNudgeType="last_train", selectType="default")
    sel.requester = "agg"
    sel.selected_ends = {"agg": set()}
    return sel


def _send(sel, ends, concurrency, task, channel_props):
    ctx = SelectContext(task_to_perform=task, channel_props=channel_props, trainer_unavail_list=[])
    sel._handle_send_state(ends, concurrency, ctx)
    return ctx


def test_dispatch_records_task_and_split(make_ends, channel_props):
    sel = _sel()
    ends = make_ends([f"t{i}" for i in range(8)])
    _send(sel, ends, 4, "train", channel_props)
    assert len(sel.selected_ends["agg"]) == 4 and set(sel._inflight_task.values()) == {"train"}
    occ = _send(sel, ends, 4, "train", channel_props).occupancy
    assert occ["in_flight_train"] == 4 and occ["in_flight_eval"] == 0 and occ["free_slots"] == 0
    assert occ["train_slots_held_by_eval"] == 0 and occ["train_blocked_by_eval"] is False


def test_eval_holding_train_slots_blocks_train(make_ends, channel_props):
    sel = _sel()
    ends = make_ends([f"t{i}" for i in range(8)])
    sel.selected_ends["agg"] = {"t0", "t1", "t2", "t3"}
    sel.all_selected = {e: time.time() for e in sel.selected_ends["agg"]}
    sel._inflight_task = {"t0": "train", "t1": "eval", "t2": "eval", "t3": "eval"}
    occ = _send(sel, ends, 4, "train", channel_props).occupancy
    assert occ == {"in_flight_train": 1, "in_flight_eval": 3, "cooling": 0, "free_slots": 0, "idle_eligible": 4,
                   "train_slots_held_by_eval": 3, "train_blocked_by_eval": True}


def test_eval_call_never_counts_train_blocking(make_ends, channel_props):
    sel = _sel()
    ends = make_ends([f"t{i}" for i in range(8)])
    sel.selected_ends["agg"] = {"t1"}
    sel._inflight_task = {"t1": "eval"}
    occ = _send(sel, ends, 7, "eval", channel_props).occupancy
    assert occ["train_slots_held_by_eval"] == 0 and occ["train_blocked_by_eval"] is False
