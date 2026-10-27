"""Exercise real sync dispatch, receive, selection and FedAvg across rounds."""

from types import SimpleNamespace
import time

import pytest
import torch
from sortedcontainers import SortedDict

from flame.channel import KEY_CH_STATE, VAL_CH_STATE_RECV
from flame.common.util import MLFramework
from flame.mode.horizontal.syncfl.top_aggregator import MemCache
from flame.mode.message import MessageType
from flame.optimizer.fedavg import FedAvg
from flame.selector.random import RandomSelector
from tests.mode.test_sync_sim_ordering import FakeSyncChannel, _make_agg


class Channel(FakeSyncChannel):
    def __init__(self, durations):
        super().__init__(durations, [])
        self.durations = durations
        self._selector = RandomSelector(c=len(durations))
        self.properties = {"round": 1}
        self.unavailable = []
        self.sent = []
        self.received = []

    def await_join(self):
        pass

    def ends(self, state=VAL_CH_STATE_RECV, task_to_perform="train"):
        self.properties[KEY_CH_STATE] = state
        return list(self._selector.select(self._ends, self.properties, self.unavailable, task_to_perform))

    def set_curr_unavailable_trainers(self, trainer_unavail_list):
        self.unavailable = list(trainer_unavail_list)

    def set_property(self, key, value):
        self.properties[key] = value

    def dumps(self, msg):
        return msg

    def send_payload(self, end, msg):
        self.sent.append(end)
        self._scts[end] = msg[MessageType.SIM_SEND_TS] + self.durations[end]
        self._queue.append(end)

    def recv_fifo(self, end_ids, first_k=0, timeout=None):
        self.received.append(set(end_ids))
        for msg, md in super().recv_fifo(end_ids, first_k, timeout):
            if msg:
                msg[MessageType.WEIGHTS] = {"weight": torch.ones(1, 1)}
                msg[MessageType.DATASET_SIZE] = 1
                msg[MessageType.MODEL_VERSION] = self.properties["round"]
            yield msg, md


class RecordingFedAvg(FedAvg):
    def do(self, base_weights, cache, **kwargs):
        self.contributors = set(cache)
        self.total = kwargs["total"]
        return super().do(base_weights, cache, **kwargs)


def make_runtime(channel, goal, traces=None, budget=float("inf")):
    agg = _make_agg()
    agg.config = SimpleNamespace(hyperparameters=SimpleNamespace(
        aggregation_goal=goal, max_experiment_runtime_s=budget
    ))
    agg._init_availability(agg.config)
    agg.trainer_event_dict = traces
    agg._round, agg._rounds, agg._work_done = 1, 6, False
    agg.agg_start_time_ts = time.time()
    agg.framework = MLFramework.PYTORCH
    agg.model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        agg.model.weight.zero_()
    agg.cache, agg.optimizer = MemCache(), RecordingFedAvg()
    agg._agg_training_stats = {}
    agg._round_update_stat_keys = ["staleness", "stat_utility", "trainer_speed", "update_visibility_lag_s"]
    agg._round_update_values = {k: [] for k in agg._round_update_stat_keys}
    agg.cm = SimpleNamespace(get_by_tag=lambda tag: channel)
    agg.dist_tag = "tag"
    agg._await_min_trainers = lambda ch: None
    agg._inject_oracle_utilities = lambda ch, task: None
    agg._sim_committed = set()
    agg.link_runtime_uplink = agg.link_runtime_downlink = None
    agg.datasampler = SimpleNamespace(get_metadata=lambda r, ends: {})
    return agg


@pytest.mark.parametrize("with_trace", [False, True])
def test_n200_goal175_completes_without_orphaned_results(with_trace):
    ch = Channel({f"t{i:03}": float(i+1) for i in range(200)})
    traces = {end: SortedDict({0.0: "AVL_TRAIN"}) for end in ch._ends} if with_trace else None
    agg = make_runtime(ch, 175, traces)
    for round_num in range(1, 7):
        assert agg._round == ch.properties["round"] == round_num
        ch.sent.clear()
        agg._distribute_weights("tag")
        assert not agg._work_done
        selected = set(ch.sent)
        assert len(selected) >= 175
        assert selected.isdisjoint(agg._sim_discarded_completions)
        agg._aggregate_weights("tag")
        assert ch.received[-1] == selected
        assert not ch._queue
        assert ch._selector.selected_ends == set(agg._sim_discarded_completions)
        assert set(ch._selector.all_selected) == set(agg._sim_discarded_completions)
        assert agg.optimizer.contributors == set(sorted(selected, key=lambda e: ch._scts[e])[:175])
        assert agg.optimizer.total == 175
        assert agg.model.weight.item() == pytest.approx(round_num, abs=1e-4)
        agg.increment_round()
    assert agg._work_done and agg._round == 7


def test_received_surplus_not_abandoned_or_rereceived():
    ch = Channel({"fast": 5.0, "slow": 250.0})
    agg = make_runtime(ch, 1, {e: SortedDict({0.0: "AVL_TRAIN"}) for e in ch._ends})
    agg._distribute_weights("tag")
    agg._aggregate_weights("tag")
    agg.increment_round()
    agg._vclock.advance(110)
    ch._selector.time_sent["slow"] = time.time() - 1000
    ch.sent.clear()
    agg._distribute_weights("tag")
    assert ch.sent == ["fast"]
    assert not agg.pending_withheld and not ch._selector.stale_ends
    agg._aggregate_weights("tag")
    assert ch.received[-1] == {"fast"}
    agg._vclock.advance(250)
    agg._sim_release_discarded(ch)
    assert "slow" not in ch._selector.selected_ends


def test_pending_completion_advances_without_empty_round():
    ch = Channel({"fast": 5.0, "slow": 25.0})
    agg = make_runtime(ch, 2)
    agg._sim_discarded_completions = {"slow": 25.0}
    agg._distribute_weights("tag")
    assert agg._vclock.now == 25.0 and not agg._work_done
    agg._aggregate_weights("tag")
    agg.increment_round()
    assert agg._round == 1
    agg._distribute_weights("tag")
    assert set(ch.sent) == {"fast", "slow"}


def test_inflight_work_is_not_trace_exhaustion():
    ch = Channel({"fast": 5.0, "slow": 25.0})
    agg = make_runtime(ch, 2, {e: SortedDict({0.0: "AVL_TRAIN"}) for e in ch._ends})
    agg._distribute_weights("tag")
    agg._distribute_weights("tag")
    assert not agg._work_done and agg._vclock.now == 0
    agg._aggregate_weights("tag")
    assert agg.model.weight.item() == 1


def test_empty_selection_advances_to_station_recovery():
    ch = Channel({"fast": 5.0, "slow": 25.0})
    agg = make_runtime(ch, 1, {"fast": SortedDict({0.0: "AVL_TRAIN"}),
                              "slow": SortedDict({0.0: "UN_AVL", 20.0: "AVL_TRAIN"})})
    agg._distribute_weights("tag")
    assert not ch.sent and not agg._work_done and agg._vclock.now == 20
    agg._aggregate_weights("tag")
    agg.increment_round()
    assert agg._round == 1
    agg._distribute_weights("tag")
    assert set(ch.sent) == {"fast", "slow"}


def test_trace_exhaustion_remains_terminal():
    ch = Channel({"fast": 5.0})
    agg = make_runtime(ch, 1, {"fast": SortedDict({0.0: "UN_AVL"})})
    agg._distribute_weights("tag")
    agg._aggregate_weights("tag")
    agg.increment_round()
    assert agg._work_done and agg._round == 1


def test_completion_cannot_advance_past_budget():
    ch = Channel({"fast": 5.0, "slow": 25.0})
    agg = make_runtime(ch, 2, budget=10)
    agg._sim_discarded_completions = {"slow": 25.0}
    agg._distribute_weights("tag")
    assert agg._vclock.now == 10
    agg._distribute_weights("tag")
    assert agg._work_done
