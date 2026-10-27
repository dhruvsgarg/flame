"""Real role hooks, synthetic links, shared dispatch and FMoW capture cutoffs."""
from collections import deque
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from flame.common.util import MLFramework
from flame.mode.message import MessageType as M
from fmow.simulation import SatelliteSimulation
from fmow.simulation.contacts import Contact, ContactSchedule
from fmow.simulation.links import LinkState
from tests.mode.test_sync_result_lifecycle import Channel, make_runtime
from tests.mode.test_sync_sim_ordering import _ConcreteSyncAgg


class SatelliteAggregator(SatelliteSimulation, _ConcreteSyncAgg):
    pass


class DeliveryChannel(Channel):
    def __init__(self, durations):
        super().__init__(durations)
        self.messages = {}
        self.payload_objects = []

    def send_payload(self, end, msg):
        self.messages[end] = msg
        self.payload_objects.append(id(msg))
        super().send_payload(end, msg)

    def recv_fifo(self, *args, **kwargs):
        for msg, md in super().recv_fifo(*args, **kwargs):
            if msg:
                header = self.messages[md[0]][M.SATELLITE_ASSIGNMENTS][md[0]]
                msg[M.SATELLITE_ASSIGNMENT] = dict(header)
                if header["complete"]:
                    msg[M.SIM_COMPLETION_TS] = header["model_arrival_s"] + self.durations[md[0]]
                else:
                    msg.pop(M.WEIGHTS)
                    msg[M.DATASET_SIZE] = 0
            yield msg, md


class Link:
    def sample(self, request):
        return LinkState(1e9, .002, 100)


def make_satellite_aggregator(tmp_path, channel, goal=1, enabled=True):
    base = make_runtime(channel, goal)
    agg = SatelliteAggregator.__new__(SatelliteAggregator)
    agg.__dict__.update(base.__dict__)
    path = tmp_path / "fmow.yaml"
    path.write_text("{}")
    agg.config.hyperparameters.fmow_config_path = str(path)
    agg._init_satellite_simulation(agg.config, role="aggregator")
    agg._satellite_endpoints = {end: i for i, end in enumerate(channel._ends)}
    agg._satellite_contacts = ContactSchedule(list(channel._ends), ["gs"], 0, 100,
        {i: [Contact("gs", 10, 30)] for i in range(len(channel._ends))})
    if enabled:
        agg.configure_satellite_link_provider(Link())
    return agg


def make_trainer(tmp_path, monkeypatch, packet):
    path = Path(__file__).parents[1] / "trainer/pytorch/main.py"
    monkeypatch.syspath_prepend(str(path.parent))
    spec = importlib.util.spec_from_file_location("fmow_delivery_trainer_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cls = module.PyTorchFMoWTrainer
    trainer = cls.__new__(cls)
    config = SimpleNamespace(task_id="a", hyperparameters=SimpleNamespace(
        fmow_config_path=str(tmp_path / "fmow.yaml"), satellite_index=0))
    trainer._init_satellite_simulation(config, role="trainer")
    trainer._sim_send_ts = None
    trainer._agg_start_origin = None
    trainer.trainer_start_ts = 0
    trainer.simulated, trainer.time_mode = True, "simulated"
    trainer._work_done = False
    trainer._round, trainer._updates_returned_upto_round = 1, 0
    trainer.trainer_id, trainer.task_to_perform = "a", "train"
    trainer.model = torch.nn.Linear(1, 1, bias=False)
    trainer.framework = MLFramework.PYTORCH
    trainer.datasampler = SimpleNamespace(handle_metadata_from_aggregator=lambda msg: None)
    channel = SimpleNamespace(
        await_join=lambda: False, one_end=lambda state: "agg", recv=lambda end: (packet, None),
        cleanup_recvd_ends=Mock(), _selector=SimpleNamespace(ordered_updates_recv_ends=[]),
    )
    trainer.cm = SimpleNamespace(get_by_tag=lambda tag: channel)
    trainer.capture_events = np.array([[0, 0], [5, 1], [10, 2], [11, 3], [50, 4]])
    trainer.capture_idx, trainer.image_buffer = 0, deque()
    return trainer, channel


def test_model_arrival_controls_real_fmow_capture_cutoff(tmp_path, monkeypatch):
    ch = DeliveryChannel({"a": 1.0})
    agg = make_satellite_aggregator(tmp_path, ch)
    agg._distribute_weights("tag")
    packet = ch.messages["a"]
    header = packet[M.SATELLITE_ASSIGNMENTS]["a"]
    assert 10 < header["model_arrival_s"] < 11
    trainer, channel = make_trainer(tmp_path, monkeypatch, packet)
    trainer._fetch_weights("fetch")
    assert trainer.fetch_success
    trainer._admit_captured_images()
    assert list(trainer.image_buffer) == [0, 1, 2]
    # Host/GPU wait cannot advance this assignment's data cutoff.
    monkeypatch.setattr("time.time", lambda: 10000)
    trainer._admit_captured_images()
    assert list(trainer.image_buffer) == [0, 1, 2]
    assert trainer.satellite_training_start_s() == header["model_arrival_s"]


def test_fresh_assignment_can_reuse_model_version_but_duplicate_cannot(tmp_path, monkeypatch):
    ch = DeliveryChannel({"a": 1.0})
    agg = make_satellite_aggregator(tmp_path, ch, enabled=False)
    agg._distribute_weights("tag")
    first = ch.messages["a"]
    trainer, channel = make_trainer(tmp_path, monkeypatch, first)
    trainer._fetch_weights("fetch")
    trainer._updates_returned_upto_round = 1
    trainer._fetch_weights("fetch")
    assert not trainer.fetch_success
    # A completed no-data attempt can be retried without a fake model commit.
    agg._sim_release_discarded(ch)
    ch._selector.selected_ends.clear()
    ch._selector.all_selected.clear()
    agg._vclock.advance(20)
    agg._distribute_weights("tag")
    second = ch.messages["a"]
    channel.recv = lambda end: (second, None)
    trainer._fetch_weights("fetch")
    assert trainer.fetch_success and trainer._round == 1
    assert trainer.satellite_now_s() == 20


def test_shared_broadcast_and_disabled_provider_preserve_baseline_time(tmp_path):
    ch = DeliveryChannel({"a": 1.0, "b": 2.0})
    agg = make_satellite_aggregator(tmp_path, ch, enabled=False)
    agg._distribute_weights("tag")
    assert len(set(ch.payload_objects)) == 1
    assert all(h["model_arrival_s"] == 0 for h in ch.messages["a"][M.SATELLITE_ASSIGNMENTS].values())
    agg._aggregate_weights("tag")
    assert agg._vclock.now == 1


def test_undeliverable_model_skips_gpu_loading_and_cannot_form_quorum(tmp_path, monkeypatch):
    ch = DeliveryChannel({"a": 1.0})
    agg = make_satellite_aggregator(tmp_path, ch)
    agg._satellite_planner.contacts = ContactSchedule(["a"], ["gs"], 0, 100, {})
    agg._distribute_weights("tag")
    trainer, _ = make_trainer(tmp_path, monkeypatch, ch.messages["a"])
    trainer._update_model = Mock(side_effect=AssertionError("must not load undeliverable model"))
    trainer._fetch_weights("fetch")
    assert not trainer.satellite_assignment_delivered()
    trainer.train()
    assert trainer.dataset_size == 0
    with pytest.raises(RuntimeError, match="aggregation goal"):
        agg._aggregate_weights("tag")
    assert not ch._selector.selected_ends


def test_provider_factory_is_optional_and_loaded_once(tmp_path, monkeypatch):
    import sys
    from types import ModuleType
    ch = DeliveryChannel({"a": 1.0})
    agg = make_satellite_aggregator(tmp_path, ch, enabled=False)
    module = ModuleType("test_satellite_provider")
    factory = Mock(return_value=Link())
    module.build = factory
    monkeypatch.setitem(sys.modules, module.__name__, module)
    agg._satellite_provider_spec = {"factory": "test_satellite_provider:build", "kwargs": {"example": 1}}
    assert agg._get_satellite_planner() is agg._get_satellite_planner()
    factory.assert_called_once_with(example=1)


def test_provider_cannot_plan_past_experiment_budget(tmp_path):
    ch = DeliveryChannel({"a": 1.0})
    agg = make_satellite_aggregator(tmp_path, ch, enabled=False)
    agg.config.hyperparameters.max_experiment_runtime_s = 5
    agg.configure_satellite_link_provider(Link())
    agg._distribute_weights("tag")
    header = ch.messages["a"][M.SATELLITE_ASSIGNMENTS]["a"]
    assert not header["complete"]
    assert agg._satellite_planner.contacts.horizon_s == 5
