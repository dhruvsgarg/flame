# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Batch 4 finding 1: D.1 proactive in-flight eviction must run in real mode too.

`_sim_evict_unavail_inflight` (D.1, felix-only via `proactive_inflight_evict`)
has no vclock/sim dependency -- it reads `_avail_now()` (mode-dispatching
already) and the selector's own `selected_ends`, both equally valid in real
mode. It used to be called only inside `if self.simulated:` alongside
`_abandon_stalled` (sim-only on the async stack, whose selector has its own
wall-clock abandon, `SEND_TIMEOUT_WAIT_S`). That left
real-mode felix runs with no way to drop a stalled UN_AVL trainer out of
`selected_ends` (and therefore out of `channel.ends(VAL_CH_STATE_RECV)`,
since `async_oort`'s selector derives recv-state ends directly from
`selected_ends`) -- the aggregator looped on 30s `recv_fifo` timeouts forever
once enough trainers went quiet near a trace's tail, never reaching its own
`max_experiment_runtime_s` self-stop check (which only runs once recv_ends
goes empty). See UNAVAILABILITY_DESIGN.md, Batch 4 finding 1.

Mirrors the harness pattern in test_agg_start_ts_broadcast.py.
"""

from __future__ import annotations

import types

import pytest


class _DistChannel:
    """Records every dispatched payload/property, no real transport."""

    def __init__(self, ends):
        self._ends = {e: None for e in ends}
        self.properties = {}
        self.props = {}
        self.sent = {}
        self._selector = types.SimpleNamespace(selected_ends=set())

    def await_join(self):
        pass

    def ends(self, state, task=None, **kw):
        return list(self._ends.keys())

    def dumps(self, msg):
        return dict(msg)

    def send_payload(self, end, payload):
        self.sent[end] = payload

    def set_end_property(self, end, key, value):
        self.props.setdefault(end, {})[key] = value

    def set_curr_unavailable_trainers(self, trainer_unavail_list=None):
        self.unavail = list(trainer_unavail_list or [])


def _stub_common(agg, channel, *, simulated, evict_spy, abandon_spy):
    agg.simulated = simulated
    agg.agg_start_time_ts = 1_700_000_000.0
    agg._round = 3
    agg.trainer_event_dict = None  # gate off: skip the oracular unavailable-list path
    agg.cm = types.SimpleNamespace(get_by_tag=lambda tag: channel)
    agg._await_min_trainers = lambda ch: None
    agg._update_weights = lambda: None
    agg._inject_oracle_utilities = lambda ch, task: None
    agg._avail_stamp_end_states = lambda ch: None
    agg._avail_now = lambda: 0.0
    agg._abandon_stalled = abandon_spy
    agg._sim_evict_unavail_inflight = evict_spy
    agg.datasampler = types.SimpleNamespace(get_metadata=lambda r, e: {})
    agg.weights = {}


def _spies():
    calls = {"evict": 0, "abandon": 0}

    def evict_spy(ch):
        calls["evict"] += 1

    def abandon_spy(ch):
        calls["abandon"] += 1

    return calls, evict_spy, abandon_spy


@pytest.fixture(autouse=True)
def _identity_weights(monkeypatch):
    import flame.mode.horizontal.syncfl.top_aggregator as syncfl_mod
    import flame.mode.horizontal.oort.top_aggregator as oort_mod

    monkeypatch.setattr(syncfl_mod, "weights_to_device", lambda w, d: w)
    monkeypatch.setattr(oort_mod, "weights_to_device", lambda w, d: w)


def _make_syncfl_agg(channel, *, simulated, evict_spy, abandon_spy):
    from flame.mode.horizontal.syncfl.top_aggregator import TopAggregator

    class _Concrete(TopAggregator):
        def check_and_sleep(self):
            pass

        def evaluate(self):
            pass

        def initialize(self):
            pass

        def load_data(self):
            pass

        def train(self):
            pass

    agg = _Concrete.__new__(_Concrete)
    _stub_common(agg, channel, simulated=simulated, evict_spy=evict_spy, abandon_spy=abandon_spy)
    agg.config = types.SimpleNamespace(
        hyperparameters=types.SimpleNamespace(aggregation_goal=1)
    )
    return agg


def _make_oort_agg(channel, *, simulated, evict_spy, abandon_spy):
    from flame.mode.horizontal.oort.top_aggregator import TopAggregator

    class _Concrete(TopAggregator):
        def check_and_sleep(self):
            pass

        def evaluate(self):
            pass

        def initialize(self):
            pass

        def load_data(self):
            pass

        def train(self):
            pass

    agg = _Concrete.__new__(_Concrete)
    _stub_common(agg, channel, simulated=simulated, evict_spy=evict_spy, abandon_spy=abandon_spy)
    agg.config = types.SimpleNamespace(
        selector=types.SimpleNamespace(kwargs={"aggr_num": 1}),
        hyperparameters=types.SimpleNamespace(inflight_residence=False),
    )
    return agg


def _make_asyncfl_agg(channel, *, simulated, evict_spy, abandon_spy):
    from flame.mode.horizontal.asyncfl.top_aggregator import TopAggregator
    from flame.sim import VirtualClock

    class _Concrete(TopAggregator):
        def check_and_sleep(self):
            pass

        def evaluate(self):
            pass

        def initialize(self):
            pass

        def load_data(self):
            pass

        def train(self):
            pass

    agg = _Concrete.__new__(_Concrete)
    _stub_common(agg, channel, simulated=simulated, evict_spy=evict_spy, abandon_spy=abandon_spy)
    agg._vclock = VirtualClock()
    agg._sim_staggered_redispatch = False
    agg._inflight_residence = False
    agg._sim_free_slot_ts = __import__("collections").deque(maxlen=128)
    agg._sim_last_commit_sct = {}
    agg._sim_inflight_expected = {}
    agg._sim_known_delay_s = {}
    agg._sim_redispatch_gap_s = 0.0
    agg._sim_cooldown_until = {}
    agg._real_distribute_settle_s = 0.0
    agg._track_trainer_version_duration_s = {}
    agg.weights = {}
    return agg


class TestProactiveEvictBothModes:
    """D.1 eviction and the C.3 90s abandon both run in either mode on the sync stacks."""

    def test_syncfl_real_mode_still_evicts(self):
        ch = _DistChannel(["e1", "e2"])
        calls, evict_spy, abandon_spy = _spies()
        agg = _make_syncfl_agg(ch, simulated=False, evict_spy=evict_spy, abandon_spy=abandon_spy)
        agg._distribute_weights("tag", "train")
        assert calls["evict"] == 1
        assert calls["abandon"] == 1  # real sync selectors have no native abandon

    def test_syncfl_sim_mode_evicts_and_abandons(self):
        from flame.sim import VirtualClock

        ch = _DistChannel(["e1", "e2"])
        calls, evict_spy, abandon_spy = _spies()
        agg = _make_syncfl_agg(ch, simulated=True, evict_spy=evict_spy, abandon_spy=abandon_spy)
        agg._vclock = VirtualClock()
        agg._distribute_weights("tag", "train")
        assert calls["evict"] == 1
        assert calls["abandon"] == 1

    def test_oort_real_mode_still_evicts(self):
        ch = _DistChannel(["e1", "e2"])
        calls, evict_spy, abandon_spy = _spies()
        agg = _make_oort_agg(ch, simulated=False, evict_spy=evict_spy, abandon_spy=abandon_spy)
        agg._distribute_weights("tag", "train")
        assert calls["evict"] == 1
        assert calls["abandon"] == 1  # real sync selectors have no native abandon

    def test_oort_sim_mode_evicts_and_abandons(self):
        from flame.sim import VirtualClock

        ch = _DistChannel(["e1", "e2"])
        calls, evict_spy, abandon_spy = _spies()
        agg = _make_oort_agg(ch, simulated=True, evict_spy=evict_spy, abandon_spy=abandon_spy)
        agg._vclock = VirtualClock()
        agg._distribute_weights("tag", "train")
        assert calls["evict"] == 1
        assert calls["abandon"] == 1

    def test_asyncfl_real_mode_still_evicts(self):
        ch = _DistChannel(["e1", "e2"])
        calls, evict_spy, abandon_spy = _spies()
        agg = _make_asyncfl_agg(ch, simulated=False, evict_spy=evict_spy, abandon_spy=abandon_spy)
        agg._distribute_weights("tag", "train")
        assert calls["evict"] == 1
        assert calls["abandon"] == 0  # async selector abandons natively (SEND_TIMEOUT_WAIT_S)

    def test_asyncfl_sim_mode_evicts_and_abandons(self):
        ch = _DistChannel(["e1", "e2"])
        calls, evict_spy, abandon_spy = _spies()
        agg = _make_asyncfl_agg(ch, simulated=True, evict_spy=evict_spy, abandon_spy=abandon_spy)
        agg._distribute_weights("tag", "train")
        assert calls["evict"] == 1
        assert calls["abandon"] == 1


class TestOortResidenceStarvationWakeup:
    """FX-D6: at syn_0, residence-held (still computing) trainers are the only wake-up;
    the starvation branch must advance to the earliest held sct, not stop the run."""

    def _agg(self, residence):
        from sortedcontainers import SortedDict
        from flame.sim import SimReorderBuffer, VirtualClock

        ch = _DistChannel(["e1", "e2"])
        calls, evict_spy, abandon_spy = _spies()
        agg = _make_oort_agg(ch, simulated=True, evict_spy=evict_spy, abandon_spy=abandon_spy)
        agg.config.selector.kwargs["aggr_num"] = 2  # desired 2 > eligible 1
        agg.config.hyperparameters.inflight_residence = residence
        agg.config.hyperparameters.max_experiment_runtime_s = 300
        agg._vclock = VirtualClock()
        agg._vclock.advance(10.0)
        agg._avail_now = lambda: agg._vclock.now
        agg.trainer_event_dict = {e: SortedDict({0.0: "AVL_TRAIN"}) for e in ("e1", "e2")}
        agg._trace_has_avl_eval = False
        agg.pending_withheld = {}
        agg._work_done = False
        agg._sim_buffer = SimReorderBuffer()
        agg._sim_buffer.add("e1", 50.0)  # still computing at vclock 10
        return agg

    def test_advances_to_held_sct(self):
        agg = self._agg(residence=True)
        agg._distribute_weights("tag", "train")
        assert agg._vclock.now == 50.0 and agg._work_done is False

    def test_residence_off_keeps_legacy_stop(self):
        from sortedcontainers import SortedDict

        agg = self._agg(residence=False)
        agg.trainer_event_dict["e2"] = SortedDict({0.0: "UN_AVL"})  # starve, never recovers
        agg._distribute_weights("tag", "train")
        assert agg._vclock.now == 10.0 and agg._work_done is True
