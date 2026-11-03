# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Run-exit artifacts: whole round checkpoints (FX-N33) and the run_end event (FX-N31)."""

import os
import threading
import time
from types import SimpleNamespace

import torch

from flame.common.util import MLFramework
from flame.mode.horizontal.syncfl.top_aggregator import TopAggregator


class _Agg(TopAggregator):
    check_and_sleep = evaluate = initialize = load_data = train = lambda self, *a, **k: None


def _agg(tmp_path, monkeypatch):
    monkeypatch.setenv("FLAME_TELEMETRY_DIR", str(tmp_path / "telemetry"))
    agg = _Agg.__new__(_Agg)
    agg.config = SimpleNamespace(hyperparameters=SimpleNamespace(
        checkpoint={"enabled": "True", "every_n_rounds": 1}))
    agg.model = torch.nn.Linear(4, 2)
    agg.framework = MLFramework.PYTORCH
    agg._round, agg.simulated, agg.time_mode = 3, False, "real"
    agg.agg_start_time_ts = time.time()
    return agg


def test_checkpoint_writer_is_joined_at_exit_and_atomic(tmp_path, monkeypatch):
    started = []
    real_thread = threading.Thread

    def spy(*a, **kw):
        t = real_thread(*a, **kw)
        started.append(t)
        return t

    monkeypatch.setattr(threading, "Thread", spy)
    _agg(tmp_path, monkeypatch).save_round_checkpoint()
    assert started and not started[0].daemon
    started[0].join(10)
    ck = tmp_path / "checkpoints"
    assert sorted(os.listdir(ck)) == ["round_00003.pt"]
    assert torch.load(ck / "round_00003.pt", weights_only=False)["round"] == 3


def test_run_end_event_carries_final_vclock(tmp_path, monkeypatch):
    # FX-N31: EV12 reads a starved sim's last vclock jump from run_end.
    from flame import telemetry

    got = []
    monkeypatch.setattr(telemetry, "is_enabled", lambda: True)
    monkeypatch.setattr(telemetry, "emit", lambda ev, **f: got.append((ev, f)))
    agg = _agg(tmp_path, monkeypatch)
    agg.simulated, agg._work_done, agg.dist_tag = True, True, "distribute"
    agg._vclock = SimpleNamespace(now=124.75)
    agg._avail_now = lambda: 124.75
    sent = []
    agg.cm = SimpleNamespace(get_by_tag=lambda tag: SimpleNamespace(broadcast=sent.append))
    agg.inform_end_of_training()
    assert got == [("run_end", {"round": 3, "work_done": True, "vclock_now": 124.75})] and sent
