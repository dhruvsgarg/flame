# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""FX-D12: an MQTT UN_AVL report frees selector slots only when no availability substrate owns
in-flight release. With the substrate on, an in-flight trainer keeps its slot (T1), so it is not
re-dispatched while its update is outstanding (P2 syn_50 felix: EV10 9, EV5 lost v288)."""

import asyncio

from flame.channel import Channel
from flame.config import TrainerAvailState
from flame.end import PROP_END_AVL_STATE, End


class _Sel:
    def __init__(self):
        self.removed = []

    def remove_from_selected_ends(self, ends, end_id):
        self.removed.append(end_id)

    def _cleanup_removed_ends(self, end_id):
        self.removed.append(end_id)


class _Backend:
    def set_cleanup_ready(self, end_id):
        pass


def _channel():
    ch = Channel.__new__(Channel)
    ch._name = "t"
    ch._ends = {"e": End("e")}
    ch._selector = _Sel()
    ch._backend = _Backend()
    return ch


def _update(ch, state):
    asyncio.run(ch.update_state("e", state, "0"))


def test_unavail_frees_slot_by_default():
    ch = _channel()
    _update(ch, TrainerAvailState.UN_AVL)
    assert ch._selector.removed == ["e", "e"]


def test_substrate_owned_keeps_slot_and_records_state():
    ch = _channel()
    ch.release_slots_on_unavail = False
    _update(ch, TrainerAvailState.AVL_TRAIN)
    _update(ch, TrainerAvailState.UN_AVL)
    _update(ch, TrainerAvailState.AVL_TRAIN)
    _update(ch, TrainerAvailState.AVL_EVAL)
    assert ch._selector.removed == []
    assert ch._ends["e"].get_property(PROP_END_AVL_STATE) == TrainerAvailState.AVL_EVAL


def _claim(cls, trace=True):
    from types import SimpleNamespace
    agg = SimpleNamespace(_SUBSTRATE_OWNS_INFLIGHT_RELEASE=cls._SUBSTRATE_OWNS_INFLIGHT_RELEASE)
    agg.trainer_event_dict = {} if trace else None
    ch = SimpleNamespace()
    cls._claim_inflight_release(agg, [ch])
    return getattr(ch, "release_slots_on_unavail", True)


def test_asyncfl_claims_release_when_substrate_on():
    from flame.mode.horizontal.asyncfl.top_aggregator import TopAggregator
    assert _claim(TopAggregator) is False
    assert _claim(TopAggregator, trace=False) is True


def test_fwdllm_opts_out():
    from flame.mode.horizontal.syncfl.fwdllm_aggregator import TopAggregator
    assert _claim(TopAggregator) is True
