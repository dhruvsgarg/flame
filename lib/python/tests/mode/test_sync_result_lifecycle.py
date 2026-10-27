"""Real sync lifecycle with deterministic physical replies and modeled durations."""
from pathlib import Path
from types import SimpleNamespace
from collections import defaultdict, deque
from datetime import datetime
import time
import torch
import pytest
from sortedcontainers import SortedDict
from flame.common.util import MLFramework
from flame.channel import KEY_CH_STATE, VAL_CH_STATE_RECV
from flame.mode.message import MessageType as M
from flame.mode.horizontal.syncfl.top_aggregator import MemCache
from flame.optimizer.fedavg import FedAvg
from flame.selector.random import RandomSelector
from flame.selector.properties import PROP_SIM_SEND_TS
from tests.mode.test_sync_sim_ordering import _make_agg

class End:

    def __init__(self):
        self.props = {}

    def get_property(self, key):
        return self.props.get(key)

    def set_property(self, key, value):
        self.props[key] = value

class Channel:

    def __init__(self, durations, empty=False):
        self.durations = durations
        self.empty = empty
        self._ends = {e: End() for e in durations}
        self._selector = RandomSelector(c=len(durations))
        self.properties = {'round': 1}
        self.unavailable = []
        self.sent = []
        self.pending = deque()
        self._queue = self.pending
        self._scts = dict(durations)
        self.received = []
        self.timeouts = []

    def has(self, e):
        return e in self._ends

    def await_join(self):
        pass

    def ends(self, state=VAL_CH_STATE_RECV, task_to_perform='train', **kwargs):
        self.properties[KEY_CH_STATE] = state
        return list(self._selector.select(self._ends, self.properties, self.unavailable, task_to_perform, **kwargs))

    def set_curr_unavailable_trainers(self, trainer_unavail_list):
        self.unavailable = list(trainer_unavail_list)

    def set_property(self, k, v):
        self.properties[k] = v

    def get_end_property(self, end_id, key):
        return self._ends[end_id].get_property(key)

    def set_end_property(self, e, k, v):
        self._ends[e].set_property(k, v)

    def dumps(self, msg):
        return msg

    def send_payload(self, e, msg):
        self.sent.append(e)
        sct = msg[M.SIM_SEND_TS] + self.durations[e]
        self._scts[e] = sct
        result = {M.SIM_COMPLETION_TS: sct, M.MODELED_DELAY_S: self.durations[e], M.MODEL_VERSION: msg[M.MODEL_VERSION], M.DATASET_SIZE: 0 if self.empty else 1}
        if not self.empty:
            result[M.WEIGHTS] = {'weight': torch.ones(1, 1)}
        self.pending.append((e, result))

    def recv_fifo(self, ends, first_k=0, timeout=None):
        self.timeouts.append(timeout)
        self.received.append(set(ends))
        ready = []
        for _ in range(len(self.pending)):
            (e, msg) = self.pending.popleft()
            if e in ends:
                ready.append((msg, (e, datetime.now())))
            else:
                self.pending.append((e, msg))
        yield from ready
        yield (None, ('', datetime.now()))

class Optimizer(FedAvg):

    def do(self, base_weights, cache, **kw):
        self.contributors = set(cache)
        self.total = kw['total']
        return super().do(base_weights, cache, **kw)

def make_runtime(channel, goal, traces=None, budget=float('inf')):
    ch = channel
    a = _make_agg()
    a.config = SimpleNamespace(hyperparameters=SimpleNamespace(aggregation_goal=goal, max_experiment_runtime_s=budget, sync_wait_for_k=True))
    a._init_availability(a.config)
    a.trainer_event_dict = traces if traces is not None else {e: SortedDict({0.0: 'AVL_TRAIN'}) for e in ch._ends}
    (a._round, a._rounds, a._work_done) = (1, 6, False)
    a.agg_start_time_ts = time.time()
    a.framework = MLFramework.PYTORCH
    a.model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        a.model.weight.zero_()
    (a.cache, a.optimizer) = (MemCache(), Optimizer())
    a._agg_training_stats = {}
    a._round_update_stat_keys = ['staleness', 'stat_utility', 'trainer_speed', 'update_visibility_lag_s']
    a._round_update_values = {k: [] for k in a._round_update_stat_keys}
    a.cm = SimpleNamespace(get_by_tag=lambda tag: ch)
    a.dist_tag = 'tag'
    a._await_min_trainers = lambda ch: None
    a._inject_oracle_utilities = lambda ch, task: None
    a._sim_committed = set()
    a.link_runtime_uplink = a.link_runtime_downlink = None
    a.datasampler = SimpleNamespace(get_metadata=lambda r, ends: {})
    return a

def test_n200_goal175_retains_exact_quorum_across_commits():
    ch = Channel({f't{i:03}': (i + 1) / 10 for i in range(200)})
    agg = make_runtime(ch, 175)
    commits = 0
    for _ in range(20):
        agg._distribute_weights('tag')
        agg._aggregate_weights('tag')
        if agg._round_committed:
            commits += 1
            assert len(agg.optimizer.contributors) == 175
        agg.increment_round()
        if agg._work_done:
            break
    assert commits == 6 and agg._round == 7

def test_empty_selection_advances_to_recovery():
    ch = Channel({'a': 1.0, 'b': 2.0, 'c': 3.0})
    traces = {e: SortedDict({0.0: 'AVL_TRAIN'}) for e in ch._ends}
    traces['c'] = SortedDict({0.0: 'UN_AVL', 10.0: 'AVL_TRAIN'})
    agg = make_runtime(ch, 2, traces)
    agg._distribute_weights('tag')
    assert not ch.sent and agg._vclock.now == 10
    agg._distribute_weights('tag')
    assert set(ch.sent) == set(ch._ends)

def test_partial_dispatch_respects_available_pool():
    ch = Channel({'a': 1.0, 'b': 2.0, 'c': 3.0})
    agg = make_runtime(ch, 2, {'a': SortedDict({0.0: 'AVL_TRAIN'}), 'b': SortedDict({0.0: 'AVL_TRAIN'}), 'c': SortedDict({0.0: 'UN_AVL', 10.0: 'AVL_TRAIN'})})
    agg.config.hyperparameters.sync_partial_dispatch = True
    agg._distribute_weights('tag')
    assert set(ch.sent) == {'a', 'b'}
    agg._aggregate_weights('tag')
    assert agg._round_committed

def test_known_future_response_is_not_abandoned_or_received_again():
    ch = Channel({'a': 100.0, 'b': 200.0, 'c': 300.0})
    agg = make_runtime(ch, 2)
    agg._distribute_weights('tag')
    agg._aggregate_weights('tag')
    agg.increment_round()
    ch.sent.clear()
    ch._selector.time_sent['c'] = time.time() - 1000
    agg._distribute_weights('tag')
    assert not agg.pending_withheld
    assert set(ch.sent) == {'a', 'b'}
    agg._aggregate_weights('tag')
    assert 'c' not in ch.received[-1]

def test_pending_physical_work_is_received_before_clock_jump():
    ch = Channel({'a': 1.0, 'b': 2.0})
    agg = make_runtime(ch, 2)
    agg._distribute_weights('tag')
    agg._distribute_weights('tag')
    assert agg._vclock.now == 0 and (not agg._work_done)
