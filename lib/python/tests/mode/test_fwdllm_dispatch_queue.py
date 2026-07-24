# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""#6 serial-dispatch queue: the aggregator sends a cohort's payloads one at a
time, so the k-th trainer's weights land after the first k-1 sends. When
`sim_model_dispatch_queue` is on, each trainer's sim_send_ts is offset by the
MEASURED cumulative send wall of the prior sends in its burst -- staggering
starts as real does, instead of stamping the whole burst at one frontier.
Default OFF -> every stamp equals the frontier (byte-identical)."""
import time
import types

from flame.channel import VAL_CH_STATE_RECV
from flame.mode.horizontal.syncfl.fwdllm_aggregator import TopAggregator
from flame.selector.properties import PROP_SIM_SEND_TS

_SEND_SLEEP = 0.02
_FRONTIER = 100.0


class _Chan:
    def __init__(self):
        self.properties = {}
        self.props = {}         # (end, key) -> value
        self.sent = []

    def await_join(self):
        return True

    def set_curr_unavailable_trainers(self, **_kw):
        pass

    def ends(self, *args, **kwargs):
        if args and args[0] == VAL_CH_STATE_RECV:
            return []                       # nothing in recv yet
        return ["A", "B", "C"]              # the dispatch burst

    def set_end_property(self, end, key, value):
        self.props[(end, key)] = value

    def get_end_property(self, end, key):
        return self.props.get((end, key))

    def send(self, end, payload):
        self.sent.append(end)
        time.sleep(_SEND_SLEEP)             # models per-send serialize+publish wall

    def get_c(self):
        return 30


class _Buf:
    def has(self, _e):
        return False


class _Agg:
    _distribute_weights_async = TopAggregator._distribute_weights_async
    _select_ends_for_async_respecting_reselect_gate = (
        TopAggregator._select_ends_for_async_respecting_reselect_gate
    )
    _should_send_full_weights = TopAggregator._should_send_full_weights
    _warn_if_redundant_weights_resend = TopAggregator._warn_if_redundant_weights_resend

    def __init__(self, dq):
        self.simulated = True
        self.trainer_event_dict = None
        self._chan = _Chan()
        self.cm = types.SimpleNamespace(get_by_tag=lambda _t: self._chan)
        self._round = 1
        self.data_id = 0
        self.iteration_per_data_id = 0
        self._model_version = 0
        self.var_good_enough = False
        self._weights_sent_this_cycle = set()
        self._redundant_weights_suppressed_total = 0
        self._trainer_last_model_version = {}
        self._trainer_state_dict = {}
        self._sim_inflight_expected = {}
        self._sim_known_delay_s = {"A": 5.0, "B": 5.0, "C": 5.0}
        self._sim_pending_commit = set()
        self._sim_dispatch_wall = {}
        self._sim_buffer = _Buf()
        self._sim_staggered_redispatch = False
        self._reselect_each_iteration = True
        self.weights = None
        self.config = types.SimpleNamespace(hyperparameters=types.SimpleNamespace(
            sim_model_dispatch_queue=dq, sim_overhead_warn_s=5.0))

    @property
    def version_key(self):
        return (1, 0)

    @property
    def vclock_now(self):
        return _FRONTIER

    def get_global_model_params(self):
        return {"w": 0}

    def _prepare_distribution_payload(self, _task, force_weights=False):
        return {"w": 0} if force_weights else {"v": 0}

    def _update_state_after_payload_prepared(self):
        pass


def _send_ts(agg):
    return [agg._chan.props[(e, PROP_SIM_SEND_TS)] for e in ("A", "B", "C")]


class TestSerialDispatchQueue:
    def test_flag_off_stamps_whole_burst_at_frontier(self):
        agg = _Agg(dq=False)
        agg._distribute_weights_async("t")
        assert _send_ts(agg) == [_FRONTIER, _FRONTIER, _FRONTIER]

    def test_flag_on_staggers_by_cumulative_send_wall(self):
        agg = _Agg(dq=True)
        agg._distribute_weights_async("t")
        a, b, c = _send_ts(agg)
        # Monotonic: each trainer waits behind the prior sends.
        assert a == _FRONTIER < b < c
        # B offset ~ one send, C ~ two sends (loose bounds for scheduler jitter).
        assert 0.5 * _SEND_SLEEP < (b - a) < 3 * _SEND_SLEEP
        assert 1.5 * _SEND_SLEEP < (c - a) < 5 * _SEND_SLEEP

    def test_flag_on_offsets_the_inflight_gate_too(self):
        # sct/gate entry = sim_send_ts + delay must inherit the stagger.
        agg = _Agg(dq=True)
        agg._distribute_weights_async("t")
        exp = agg._sim_inflight_expected
        assert exp["A"] < exp["B"] < exp["C"]
        assert exp["A"] == _FRONTIER + 5.0    # first trainer: frontier + its delay
