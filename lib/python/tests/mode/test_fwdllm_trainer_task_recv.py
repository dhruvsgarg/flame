# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Stage D1 (§H #8): the fwdllm trainer overrode _fetch_weights and dropped the
base trainer's task_recv emission, so field_coverage's sim_send_ts INV rung + K6
had no field to read (null/null both modes). Restored here -- task_recv carries
sim_send_ts (the aggregator's dispatch vclock), null in real mode.
"""

import json

from flame import telemetry
from flame.mode.horizontal.syncfl.fwdllm_trainer import Trainer
from flame.mode.message import MessageType


class _FakeSelector:
    def __init__(self):
        self.ordered_updates_recv_ends = []


class _FakeChannel:
    def __init__(self, msg):
        self._msg = msg
        self._selector = _FakeSelector()

    def await_join(self):
        pass

    def one_end(self, state):
        return "end_1"

    def recv(self, end_id):
        return self._msg, None

    def cleanup_recvd_ends(self):
        pass


class _FakeChannelManager:
    def __init__(self, channel):
        self._channel = channel

    def get_by_tag(self, tag):
        return self._channel


class _FakeTrainer:
    _fetch_weights = Trainer._fetch_weights

    def __init__(self, channel, time_mode="real"):
        self.cm = _FakeChannelManager(channel)
        self.trainer_id = "trainer_1"
        self.fetch_success = False
        self._work_done = False
        self.data_id = None
        self.iteration_per_data_id = None
        self._round = 1
        self._model_version = 0
        self.time_mode = time_mode


def _task_recv_events(tmp_path):
    lines = (tmp_path / "trainer.jsonl").read_text().splitlines()
    return [json.loads(l) for l in lines if json.loads(l)["event"] == "task_recv"]


class TestTaskRecvEmission:
    def test_sim_send_ts_emitted_in_sim(self, tmp_path):
        telemetry.configure(role="trainer", run_dir=str(tmp_path))
        try:
            channel = _FakeChannel(
                {MessageType.ROUND: 3, MessageType.SIM_SEND_TS: 42.5}
            )
            t = _FakeTrainer(channel, time_mode="simulated")
            t._fetch_weights("fetch")

            evs = _task_recv_events(tmp_path)
            assert len(evs) == 1
            assert evs[0]["sim_send_ts"] == 42.5
            assert evs[0]["time_mode"] == "simulated"
            assert evs[0]["trainer_id"] == "trainer_1"
        finally:
            telemetry.shutdown()

    def test_sim_send_ts_null_in_real(self, tmp_path):
        telemetry.configure(role="trainer", run_dir=str(tmp_path))
        try:
            # real mode: aggregator stamps no SIM_SEND_TS -> field is null, which
            # is what makes real~=sim task_recv directly comparable.
            channel = _FakeChannel({MessageType.ROUND: 3})
            t = _FakeTrainer(channel, time_mode="real")
            t._fetch_weights("fetch")

            evs = _task_recv_events(tmp_path)
            assert len(evs) == 1
            assert evs[0]["sim_send_ts"] is None
        finally:
            telemetry.shutdown()

    def test_noop_when_telemetry_disabled(self, tmp_path):
        assert not telemetry.is_enabled()
        channel = _FakeChannel({MessageType.ROUND: 3, MessageType.SIM_SEND_TS: 1.0})
        _FakeTrainer(channel)._fetch_weights("fetch")
        assert not (tmp_path / "trainer.jsonl").exists()
