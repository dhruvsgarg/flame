# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""FX-D42: asyncfl pickles the dispatch payload once per (version, task, stamp)."""

from __future__ import annotations

import collections
import types

import pytest

from flame.mode.message import MessageType


class _Channel:
    def __init__(self, ends):
        self._ends = {e: None for e in ends}
        self._selector = types.SimpleNamespace(selected_ends=set())
        self.properties = {}
        self.dumps_calls = 0
        self.sent = []

    def await_join(self):
        pass

    def ends(self, state, task=None, **kw):
        return list(self._ends)

    def dumps(self, msg):
        self.dumps_calls += 1
        return dict(msg)

    def send_payload(self, end, payload):
        self.sent.append((end, payload))

    def set_end_property(self, end, key, value):
        pass

    def set_curr_unavailable_trainers(self, trainer_unavail_list=None):
        pass


def _agg(channel):
    from flame.mode.horizontal.asyncfl.top_aggregator import TopAggregator
    from flame.sim import VirtualClock

    class _Concrete(TopAggregator):
        def check_and_sleep(self): pass
        def evaluate(self): pass
        def initialize(self): pass
        def load_data(self): pass
        def train(self): pass

    agg = _Concrete.__new__(_Concrete)
    agg.simulated = False
    agg.agg_start_time_ts = 1_700_000_000.0
    agg._round = 3
    agg.trainer_event_dict = None
    agg.cm = types.SimpleNamespace(get_by_tag=lambda tag: channel)
    agg._await_min_trainers = lambda ch: None
    agg._update_weights = lambda: None
    agg._inject_oracle_utilities = lambda ch, task: None
    agg._avail_stamp_end_states = lambda ch: None
    agg._avail_now = lambda: 0.0
    agg._abandon_stalled = lambda ch: None
    agg._sim_evict_unavail_inflight = lambda ch: None
    agg.datasampler = types.SimpleNamespace(get_metadata=lambda r, e: {})
    agg.config = types.SimpleNamespace(
        selector=types.SimpleNamespace(kwargs={"aggr_num": 1}),
        hyperparameters=types.SimpleNamespace(inflight_residence=False),
    )
    agg._vclock = VirtualClock()
    agg._sim_staggered_redispatch = False
    agg._inflight_residence = False
    agg._sim_free_slot_ts = collections.deque(maxlen=128)
    agg._sim_last_commit_sct = {}
    agg._sim_inflight_expected = {}
    agg._sim_known_delay_s = {}
    agg._sim_redispatch_gap_s = 0.0
    agg._sim_cooldown_until = {}
    agg._real_distribute_settle_s = 0.0
    agg._track_trainer_version_duration_s = {}
    agg.weights = {"w": 1.0}
    return agg


@pytest.fixture(autouse=True)
def _identity_weights(monkeypatch):
    import flame.mode.horizontal.asyncfl.top_aggregator as mod
    monkeypatch.setattr(mod, "pack_weights", lambda w: dict(w))  # FX-N77: the down path packs WEIGHTS_BYTES


def test_same_version_reuses_payload_new_version_or_task_repickles():
    ch = _Channel(["e1"])
    agg = _agg(ch)
    agg._distribute_weights("tag", "train")
    agg._distribute_weights("tag", "train")
    assert ch.dumps_calls == 1
    assert ch.sent[0][1] == ch.sent[1][1]
    assert ch.sent[0][1][MessageType.WEIGHTS_BYTES] == {"w": 1.0}
    agg._distribute_weights("tag", "eval")
    assert ch.dumps_calls == 2
    agg._round = 4
    agg._distribute_weights("tag", "train")
    assert ch.dumps_calls == 3
    assert ch.sent[-1][1][MessageType.MODEL_VERSION] == 4


def test_commit_invalidates_cache():
    ch = _Channel(["e1"])
    agg = _agg(ch)
    agg._distribute_weights("tag", "train")
    agg._dispatch_payload = None  # what the commit path does after scale_add
    agg.weights = {"w": 2.0}
    agg._distribute_weights("tag", "train")
    assert ch.dumps_calls == 2
    assert ch.sent[-1][1][MessageType.WEIGHTS_BYTES] == {"w": 2.0}
