"""Logical upload delivery through the real sync barrier and FedAvg reducer."""
import cloudpickle
import pytest
import torch
from sortedcontainers import SortedDict

from flame.mode.message import MessageType as M
from fmow.simulation.contacts import Contact, ContactSchedule
from fmow.simulation.links import Direction, LinkState
from test_model_delivery import DeliveryChannel, make_satellite_aggregator, make_trainer


DELTA = cloudpickle.dumps({"weight": torch.ones(1, 1)})


class ByteChannel(DeliveryChannel):
    def recv_fifo(self, *args, **kwargs):
        for msg, metadata in super().recv_fifo(*args, **kwargs):
            if msg and M.WEIGHTS in msg:
                msg.pop(M.WEIGHTS)
                msg[M.WEIGHTS_BYTES] = DELTA
            yield msg, metadata


class Link:
    def __init__(self, upload_seconds=None, propagation=0, horizon=10000):
        self.upload_seconds = upload_seconds or {}
        self.propagation, self.horizon = propagation, horizon

    def sample(self, request):
        seconds = self.upload_seconds.get(request.satellite_index, 1)
        rate = 1e12 if request.direction == Direction.MODEL_DOWNLOAD else len(DELTA) * 8 / seconds
        return LinkState(rate, self.propagation, self.horizon)


def runtime(tmp_path, durations, goal=1, contacts=None, horizon=10000, provider=None):
    ch = ByteChannel(durations)
    agg = make_satellite_aggregator(tmp_path, ch, goal, enabled=False)
    agg._satellite_contacts = ContactSchedule(list(durations), ["gs"], 0, horizon,
        {i: contacts or [Contact("gs", 0, horizon)] for i in range(len(durations))})
    agg.configure_satellite_link_provider(provider or Link(horizon=horizon))
    return agg, ch


def test_first_k_uses_upload_arrival_instead_of_compute_or_host_order(tmp_path):
    agg, ch = runtime(tmp_path, {"fast_compute": 1, "fast_upload": 2}, provider=Link({0: 20, 1: 1}))
    agg._distribute_weights("tag")
    agg._aggregate_weights("tag")
    assert agg.optimizer.contributors == {"fast_upload"}
    assert agg._vclock.now == pytest.approx(3, abs=1e-6)
    assert agg.model.weight.item() == 1
    assert agg._satellite_uploads["fast_compute"].arrival_s == pytest.approx(21, abs=1e-6)
    assert ch._selector.selected_ends == {"fast_compute"}
    assert agg._sim_discarded_completions["fast_compute"] == agg._satellite_uploads["fast_compute"].arrival_s
    agg.increment_round()
    ch.sent.clear()
    agg._distribute_weights("tag")
    assert ch.sent == ["fast_upload"]
    agg._aggregate_weights("tag")
    assert ch.received[-1] == {"fast_upload"}
    agg._vclock.advance(22)
    agg._sim_release_discarded(ch)
    assert not ch._selector.selected_ends


def test_upload_resumes_next_contact_and_does_not_get_second_eligibility_wait(tmp_path):
    agg, ch = runtime(tmp_path, {"a": .3}, contacts=[Contact("gs", 0, 1), Contact("gs", 10, 11)],
        horizon=20, provider=Link({0: 1.55}, propagation=.1, horizon=20))
    agg.trainer_event_dict = {"a": SortedDict({0.: "AVL_TRAIN", 1.: "UN_AVL", 10.: "AVL_TRAIN", 11.: "UN_AVL"})}
    agg._distribute_weights("tag")
    agg._aggregate_weights("tag")
    result = agg._satellite_uploads["a"]
    assert result.complete and result.arrival_s > 11
    assert result.wait_s == pytest.approx(9)
    assert result.transmit_s == pytest.approx(1.55)
    assert agg._vclock.now == result.arrival_s
    assert agg.optimizer.contributors == {"a"}
    assert not agg.pending_withheld and not agg._sim_withheld_payload


def test_trace_exhaustion_fails_quorum_without_applying_incomplete_upload(tmp_path):
    agg, ch = runtime(tmp_path, {"a": 1}, contacts=[Contact("gs", 0, 2)], horizon=5,
        provider=Link({0: 10}, horizon=5))
    agg._distribute_weights("tag")
    with pytest.raises(RuntimeError, match="0/1 deliverable"):
        agg._aggregate_weights("tag")
    assert agg.model.weight.item() == 0
    assert not ch._selector.selected_ends
    result = agg._satellite_uploads["a"]
    assert not result.complete and result.arrival_s is None and result.remaining_bytes > 0
    assert agg._vclock.now == 0


def test_actual_delta_byte_count_is_used_without_materializing_surplus(tmp_path):
    agg, ch = runtime(tmp_path, {"a": 1, "b": 2})
    agg._distribute_weights("tag")
    agg._aggregate_weights("tag")
    for result in agg._satellite_uploads.values():
        assert result.transmitted_bytes == len(DELTA)
        assert result.transmit_s == 1


def test_n200_goal175_multiround_link_delivery(tmp_path):
    agg, ch = runtime(tmp_path, {f"t{i:03}": i + 1 for i in range(200)}, goal=175)
    for round_number in range(1, 7):
        ch.sent.clear()
        agg._distribute_weights("tag")
        selected = set(ch.sent)
        assert len(selected) >= 175
        agg._aggregate_weights("tag")
        assert agg.optimizer.contributors == set(sorted(selected, key=lambda e: ch.durations[e])[:175])
        assert ch._selector.selected_ends == set(agg._sim_discarded_completions)
        assert ch.received[-1] == selected
        assert agg.model.weight.item() == pytest.approx(round_number, abs=1e-4)
        agg.increment_round()
    assert agg._work_done


def test_current_assignment_required_and_compute_cannot_precede_model(tmp_path):
    agg, ch = runtime(tmp_path, {"a": 1})
    agg._distribute_weights("tag")
    header = ch.messages["a"][M.SATELLITE_ASSIGNMENTS]["a"]
    result = {M.MODEL_VERSION: 1, M.SATELLITE_ASSIGNMENT: {**header, "id": "stale"}, M.SIM_COMPLETION_TS: 0}
    with pytest.raises(ValueError, match="Unexpected satellite assignment"):
        agg._prepare_simulated_update(ch, "a", result, ("a", None))
    result[M.SATELLITE_ASSIGNMENT] = header
    with pytest.raises(ValueError, match="follow model arrival"):
        agg._prepare_simulated_update(ch, "a", result, ("a", None))


def test_end_of_training_clock_uses_final_dispatch_not_previous_model_arrival(tmp_path, monkeypatch):
    agg, ch = runtime(tmp_path, {"a": 1})
    agg._distribute_weights("tag")
    trainer, _ = make_trainer(tmp_path, monkeypatch, ch.messages["a"])
    trainer._fetch_weights("fetch")
    assert trainer.satellite_link_delivery_enabled()
    trainer._work_done, trainer._sim_send_ts = True, 50
    assert trainer.satellite_now_s() == 50


def test_two_link_timing_systems_cannot_silently_double_count(tmp_path):
    agg, _ = runtime(tmp_path, {"a": 1})
    agg.link_runtime_downlink = object()
    with pytest.raises(ValueError, match="one link timing system"):
        agg.configure_satellite_link_provider(Link())
