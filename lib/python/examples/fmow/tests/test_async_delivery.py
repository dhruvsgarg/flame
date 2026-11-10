"""FMoW workload with the real async engine, FedBuff selector and reducer.

Only the network and dataset/model size are synthetic. Satellite assignments,
virtual-clock scheduling, empty captures and partial/stale updates are real.
"""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from sortedcontainers import SortedDict

from flame.common.util import pack_weights
from flame.channel import KEY_CH_SELECT_REQUESTER
from flame.end import KEY_END_STATE, VAL_END_STATE_RECVD
from flame.mode.horizontal.syncfl.top_aggregator import TopAggregator as SyncAggregator
from flame.mode.message import MessageType as M
from flame.optimizer.fedbuff import FedBuff
from flame.selector.fedbuff import FedBuffSelector
from flame.selector.properties import PROP_LAST_EVAL_ROUND
from fmow.simulation.contacts import Contact, ContactSchedule
from fmow.simulation.links import Direction, LinkState
from tests.mode.test_sync_result_lifecycle import Channel, make_runtime


DELTA = pack_weights({"classifier.weight": torch.ones(1, 1)})


class AsyncChannel(Channel):
    def __init__(self, durations, goal, concurrency=None, capture_at=0):
        super().__init__(durations)
        self._selector = FedBuffSelector(c=concurrency or len(durations), aggGoal=goal, _seed=0)
        self.properties[KEY_CH_SELECT_REQUESTER] = "agg"
        self.capture_at = capture_at
        self.messages = []

    def send_payload(self, end, msg):
        self.messages.append((end, msg))
        header = msg[M.SATELLITE_ASSIGNMENTS][end]
        sct = header["model_arrival_s"] + self.durations[end] if header["complete"] else msg[M.SIM_SEND_TS]
        empty = not header["complete"] or header["model_arrival_s"] < self.capture_at
        result = {M.MODEL_VERSION: msg[M.MODEL_VERSION], M.TASK_TO_PERFORM: "train",
                  M.DATASET_SIZE: 0 if empty else 1, M.STAT_UTILITY: 0.,
                  M.MODELED_DELAY_S: self.durations[end], M.SIM_COMPLETION_TS: sct,
                  M.SATELLITE_ASSIGNMENT: {**header, "next_capture_s": self.capture_at}}
        if not empty:
            result[M.WEIGHTS_BYTES] = DELTA
        self.pending.append((end, result))
        self.sent.append(end)

    def recv_fifo(self, ends, first_k=0, timeout=None):
        for msg, md in super().recv_fifo(ends, first_k, timeout):
            if msg:
                self._ends[md[0]].set_property(KEY_END_STATE, VAL_END_STATE_RECVD)
            yield msg, md

    def drain_ready(self, ends, timeout=None):
        yield from ((msg, md) for msg, md in self.recv_fifo(ends, timeout=timeout) if msg)

    def ends_with_pending_rx(self):
        return {end for end, _ in self.pending}

    def cleanup_recvd_ends(self):
        self._selector._cleanup_recvd_ends(self._ends)

    def cleanup_provided_ends(self, end):
        self._selector._cleanup_provided_ends({end: self._ends[end]}, self._ends)


class Link:
    def __init__(self, upload_rates_bps=None, horizon=10000, propagation=0):
        self.upload_rates_bps = upload_rates_bps or {}
        self.horizon, self.propagation = horizon, propagation
        self.calls = []

    def sample(self, request):
        self.calls.append(request)
        # Return a fixed rate per satellite. Large relative to these toy deltas.
        rate = 1e12 if request.direction == Direction.MODEL_DOWNLOAD else self.upload_rates_bps.get(request.satellite_index, 1e6)
        return LinkState(rate, self.propagation, self.horizon)


def runtime(tmp_path, monkeypatch, durations, goal=1, *, concurrency=None, capture_at=0, trace=False, provider=None, horizon=10000):
    path = Path(__file__).parents[1] / "aggregator/pytorch/main_asyncfl_agg.py"
    monkeypatch.syspath_prepend(str(path.parent))
    spec = importlib.util.spec_from_file_location("fmow_async_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ch = AsyncChannel(durations, goal, concurrency, capture_at)
    base = make_runtime(ch, goal)
    base.config.hyperparameters.fmow_config_path = str(tmp_path / "fmow.yaml")
    (tmp_path / "fmow.yaml").write_text("{}")
    hp = base.config.hyperparameters
    hp.learning_rate, hp.batch_size, hp.reject_stale_updates = .0001, 16, False
    hp.sync_wait_for_k = False
    base.config.selector = SimpleNamespace(kwargs={"c": ch._selector.c})
    base._join_barrier_done = True
    base.optimizer = FedBuff(learning_rate=1.0, agg_rate_conf={"type": "old"})
    base.model = torch.nn.Module()
    base.model.features = torch.nn.Linear(1, 1, bias=False)
    base.model.classifier = torch.nn.Linear(1, 1, bias=False)
    base.model.features.weight.requires_grad_(False)
    with torch.no_grad():
        base.model.features.weight.fill_(7.)
        base.model.classifier.weight.zero_()
    if not trace:
        base.trainer_event_dict = None
    # Replace broker startup only; execute the actual async internal_init.
    monkeypatch.setattr(SyncAggregator, "internal_init", lambda self: self.__dict__.update(base.__dict__))
    agg = module.PyTorchFMoWAsyncAggregator(base.config)
    agg.internal_init()
    agg._reset_agg_goal_variables()
    agg._satellite_endpoints = {end: i for i, end in enumerate(durations)}
    agg._satellite_contacts = ContactSchedule(list(durations), ["gs"], 0, horizon,
        {i: [Contact("gs", 0, horizon)] for i in range(len(durations))})
    if provider:
        agg.configure_satellite_link_provider(provider)
    return agg, ch


@pytest.mark.parametrize("trace", [False, True])
@pytest.mark.parametrize("ordered_drain", [False, True])
def test_empty_captures_retry_same_version_and_complete_commits(tmp_path, monkeypatch, trace, ordered_drain):
    agg, ch = runtime(tmp_path, monkeypatch, {"a": 1, "b": 2}, goal=1, capture_at=20, trace=trace)
    agg._distribute_weights("tag")
    agg._sim_sct_ordered_drain = ordered_drain
    first_ids = {e: msg[M.SATELLITE_ASSIGNMENTS][e]["id"] for e, msg in ch.messages}
    agg._aggregate_weights("tag")
    assert agg._agg_goal_cnt == 0 and not agg._task_ledger
    assert not agg._sim_inflight_expected and not agg._sim_dispatch_wall
    assert all(ch.get_end_property(e, PROP_LAST_EVAL_ROUND) is None for e in ch._ends)
    agg._distribute_weights("tag")
    assert len(ch.messages) == 2  # capture still in the future
    agg._aggregate_weights("tag")
    assert agg._vclock.now == 20 and not agg._work_done
    agg._distribute_weights("tag")
    assert all(msg[M.SATELLITE_ASSIGNMENTS][e]["id"] != first_ids[e] for e, msg in ch.messages[2:])
    for _ in range(100):
        agg._distribute_weights("tag")
        agg._aggregate_weights("tag")
        if agg._agg_goal_cnt == agg._agg_goal:
            assert agg.model.features.weight.item() == 7.
            agg.increment_round()
            agg._reset_agg_goal_variables()
        if agg._work_done:
            break
    assert agg._round == 7 and agg._n_updates == 6
    assert any(s > 0 for history in agg._per_trainer_staleness_track.values() for s in history)
    assert not getattr(agg, "require_complete_sim_cohort", False)


def test_upload_arrival_orders_updates_and_stale_result_survives(tmp_path, monkeypatch):
    provider = Link({0: 100, 1: 1e6})
    agg, ch = runtime(tmp_path, monkeypatch, {"slow_upload": 1, "fast_upload": 2}, provider=provider)
    agg._distribute_weights("tag")
    agg._aggregate_weights("tag")
    assert agg._contributed_versions == {("fast_upload", 1)}
    slow_arrival = agg._satellite_uploads["slow_upload"].arrival_s
    assert agg._vclock.now == agg._satellite_uploads["fast_upload"].arrival_s < slow_arrival
    # Retain the old assignment in the reorder buffer across model versions.
    agg.increment_round()
    agg._reset_agg_goal_variables()
    result, (end, _) = agg._sim_recv_min(ch, ch.ends())
    assert end == "slow_upload" and result[M.MODEL_VERSION] == 1
    assert agg._vclock.now == slow_arrival
    assert sum(r.direction == Direction.UPDATE_UPLOAD for r in provider.calls) == 2


def test_async_dispatch_refreshes_headers_but_reuses_packed_model(tmp_path, monkeypatch):
    agg, ch = runtime(tmp_path, monkeypatch, {"a": 1}, capture_at=0)
    agg._distribute_weights("tag")
    first = ch.messages[-1][1]
    # A same-stamp retry must not replay the previously serialized assignment.
    agg._avail_free_slot_ledger(ch, "a")
    agg._task_ledger.clear()
    agg._distribute_weights("tag")
    second = ch.messages[-1][1]
    assert second[M.WEIGHTS_BYTES] is first[M.WEIGHTS_BYTES]
    assert second[M.SATELLITE_ASSIGNMENTS]["a"]["id"] != first[M.SATELLITE_ASSIGNMENTS]["a"]["id"]


def test_staggered_dispatch_plans_each_satellite_at_its_own_stamp(tmp_path, monkeypatch):
    agg, ch = runtime(tmp_path, monkeypatch, {"a": 1, "b": 1}, provider=Link())
    agg._vclock.advance(100)
    agg._sim_staggered_redispatch = True
    agg._sim_free_slot_ts.extend([91, 95])
    agg._distribute_weights("tag")
    assert sorted(msg[M.SIM_SEND_TS] for _, msg in ch.messages) == [91, 95]
    for end, msg in ch.messages:
        header = msg[M.SATELLITE_ASSIGNMENTS][end]
        assert header["download"]["start_s"] == msg[M.SIM_SEND_TS]


def test_exhausted_transfers_do_not_count_and_eventually_stop(tmp_path, monkeypatch):
    agg, ch = runtime(tmp_path, monkeypatch, {"a": 1}, provider=Link({0: 1}, horizon=5), horizon=5)
    agg._distribute_weights("tag")
    agg._aggregate_weights("tag")
    assert agg._agg_goal_cnt == 0 and agg.model.classifier.weight.item() == 0
    agg._distribute_weights("tag")
    agg._aggregate_weights("tag")
    agg._aggregate_weights("tag")
    assert agg._work_done and agg._async_inner_loop_done()
    agg.increment_round()
    assert agg._work_done and agg._round == 1 and not agg._round_committed


def test_long_buffered_upload_keeps_its_assignment_during_fast_commits(tmp_path, monkeypatch):
    agg, ch = runtime(tmp_path, monkeypatch, {"slow": 1, "fast": 10},
                      provider=Link({0: 10, 1: 1e6}))
    agg._rounds = 100
    agg._distribute_weights("tag")
    slow_id = agg._satellite_assignments["slow"]["id"]
    for _ in range(100):
        agg._aggregate_weights("tag")
        if ("slow", 1) in agg._contributed_versions:
            break
        if agg._agg_goal_cnt == agg._agg_goal:
            agg.increment_round()
            agg._reset_agg_goal_variables()
        agg._distribute_weights("tag")
        assert agg._satellite_assignments["slow"]["id"] == slow_id
    assert ("slow", 1) in agg._contributed_versions
    assert agg._vclock.now >= agg._satellite_uploads["slow"].arrival_s


def test_empty_train_is_not_misclassified_as_eval_in_real_mode(tmp_path, monkeypatch):
    agg, ch = runtime(tmp_path, monkeypatch, {"a": 1}, capture_at=20)
    agg._distribute_weights("tag")
    agg.simulated = False
    agg._real_drain_ready_ingest = False
    agg._aggregate_weights("tag")
    assert agg._agg_goal_cnt == 0 and not agg._task_ledger
    assert not ch._selector.all_selected
    assert ch.get_end_property("a", PROP_LAST_EVAL_ROUND) is None


def test_upload_resumes_across_contacts_without_second_availability_wait(tmp_path, monkeypatch):
    agg, ch = runtime(tmp_path, monkeypatch, {"a": .3}, trace=True, horizon=20)
    agg._satellite_contacts = ContactSchedule(["a"], ["gs"], 0, 20,
        {0: [Contact("gs", 0, 1), Contact("gs", 10, 11)]})
    agg.trainer_event_dict = {"a": SortedDict({0.: "AVL_TRAIN", 1.: "UN_AVL", 10.: "AVL_TRAIN", 11.: "UN_AVL"})}
    agg.configure_satellite_link_provider(Link({0: len(DELTA) * 8 / 1.55}, horizon=20, propagation=.1))
    agg._distribute_weights("tag")
    agg._aggregate_weights("tag")
    outcome = agg._satellite_uploads["a"]
    assert outcome.complete and outcome.arrival_s > 11
    assert outcome.transmitted_bytes == len(DELTA)
    assert outcome.wait_s == pytest.approx(9)
    assert outcome.transmit_s == pytest.approx(1.55)
    assert agg._vclock.now == outcome.arrival_s
    assert agg._contributed_versions == {("a", 1)} and not agg.pending_withheld


def test_smoke_export_resolves_existing_fedbuff_stack_without_launching():
    from flame.launch.baselines import load_baselines
    from flame.launch.experiment_config import load_experiment_config
    from flame.launch.runner import ExperimentRunner

    fmow = Path(__file__).parents[1]
    exp = load_experiment_config(fmow / "exports/fmow_fedbuff_smoke_n10.yaml").experiments[0]
    baseline = load_baselines(fmow.parent / "_metadata")[exp.baseline]
    runner = ExperimentRunner.__new__(ExperimentRunner)
    config, _ = runner._build_aggregator_config(exp, fmow / exp.aggregator.config_template, baseline)
    main = runner._resolve_aggregator_main(exp, baseline)
    runner._validate_stack(fmow / main, config)
    runner._validate_selector_label(exp, config)
    assert main == "aggregator/pytorch/main_asyncfl_agg.py"
    assert config["selector"]["kwargs"]["c"] == 10
    assert config["selector"]["kwargs"]["aggGoal"] == config["hyperparameters"]["aggGoal"] == 5
    assert config["optimizer"]["kwargs"]["learning_rate"] == 1.0
    assert config["optimizer"]["kwargs"]["dataset_name"] == "fmow"


def test_n200_async_completes_six_useful_commits_after_empty_captures(tmp_path, monkeypatch):
    agg, ch = runtime(tmp_path, monkeypatch,
        {f"t{i:03}": float(i + 1) for i in range(200)}, goal=175, capture_at=20)
    commits = 0
    for _ in range(1500):
        agg._distribute_weights("tag")
        agg._aggregate_weights("tag")
        if agg._agg_goal_cnt == agg._agg_goal:
            assert agg._round_committed and agg.model.features.weight.item() == 7.
            commits += 1
            agg.increment_round()
            agg._reset_agg_goal_variables()
        if agg._work_done:
            break
    assert commits == 6 and agg._round == 7 and agg._n_updates == 1050
    assert any(s > 0 for history in agg._per_trainer_staleness_track.values() for s in history)
    assert not agg._satellite_failed_ends
