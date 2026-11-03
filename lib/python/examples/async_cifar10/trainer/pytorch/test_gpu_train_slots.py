# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""FX-D58: at most `gpu_train_slots` trainers train at once on one GPU (speech c=30 first wave OOMed an A40)."""

import types

import pytest
import torch

import main
from main import PyTorchCifar10Trainer


def _trainer(slots, gpu):
    t = PyTorchCifar10Trainer.__new__(PyTorchCifar10Trainer)
    t.device = torch.device("cuda", 0)
    t.config = types.SimpleNamespace(hyperparameters=types.SimpleNamespace(gpu_train_slots=slots))
    return t


def test_caps_holders_per_gpu(monkeypatch):
    gpu = f"t{id(object())}"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", gpu)
    a, b = _trainer(2, gpu)._acquire_gpu_slot(), _trainer(2, gpu)._acquire_gpu_slot()
    assert a is not None and b is not None and a.name != b.name

    def full(_s):
        raise TimeoutError
    monkeypatch.setattr(main.time, "sleep", full)
    with pytest.raises(TimeoutError):  # third holder waits
        _trainer(2, gpu)._acquire_gpu_slot()
    a.close()
    monkeypatch.undo()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", gpu)
    c = _trainer(2, gpu)._acquire_gpu_slot()  # a freed slot is reused
    assert c.name == a.name
    b.close(), c.close()


def test_uncapped_and_cpu_return_none():
    assert _trainer(0, "x")._acquire_gpu_slot() is None
    t = _trainer(2, "x")
    t.device = torch.device("cpu")
    assert t._acquire_gpu_slot() is None


def test_slot_released_after_cache_release():
    import inspect
    src = inspect.getsource(PyTorchCifar10Trainer.train)
    assert src.index("_acquire_gpu_slot(") < src.index("self._release_gpu_cache()") < src.index("_gpu_slot.close()")


def _sim_trainer(slots, gpu):
    t = _trainer(0, gpu)
    t.simulated = True
    t.config = types.SimpleNamespace(hyperparameters=types.SimpleNamespace(gpu_train_slots=7, sim_gpu_train_slots=slots))
    return t


def test_sim_slots_replace_real_cap_and_wake_earliest_deadline_first(monkeypatch):
    # FX-N98: one sim slot; of two waiters the earlier deadline gets it first, whichever polls first.
    import threading
    gpu = f"s{id(object())}"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", gpu)
    holder = _sim_trainer(1, gpu)._acquire_gpu_slot(5.0)
    got = []
    threads = [threading.Thread(target=lambda d=d: got.append((d, _sim_trainer(1, gpu)._acquire_gpu_slot(d))))
               for d in (90.0, 10.0)]
    for th in threads:
        th.start()
    import time
    time.sleep(0.3)  # both ticketed
    assert got == []
    holder.close()
    deadline = time.time() + 5
    while not got and time.time() < deadline:
        time.sleep(0.02)
    assert [d for d, _ in got] == [10.0]  # the later deadline still waits
    got[0][1].close()
    threads[0].join(5), threads[1].join(5)
    assert sorted(d for d, _ in got) == [10.0, 90.0]
    got[1][1].close()


def test_dead_waiter_ticket_is_dropped(tmp_path):
    import os
    d = tmp_path
    (d / "0000000000000001.000_999999999").write_text("")  # no such pid
    (d / f"0000000000000002.000_{os.getpid()}").write_text("")
    assert PyTorchCifar10Trainer._first_waiter(str(d), f"0000000000000002.000_{os.getpid()}")
    assert not (d / "0000000000000001.000_999999999").exists()


def test_slot_wait_not_charged_in_sim():
    import inspect
    src = inspect.getsource(PyTorchCifar10Trainer.train)
    assert src.index("_acquire_gpu_slot(") < src.index("_gpu_start = time.time()", src.index("_acquire_gpu_slot("))
