# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""FX-D7: the availability thread stops at EOT and never crashes on channel teardown."""

import threading

import pytest

import trainer.pytorch.main as main_mod
from trainer.pytorch.main import PyTorchCifar10Trainer


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(main_mod.time, "sleep", lambda s: None)


def _trainer(update):
    t = PyTorchCifar10Trainer.__new__(PyTorchCifar10Trainer)
    t._work_done = False
    t.check_and_update_state_avl = update
    return t


def test_exits_quietly_when_teardown_races_eot():
    t = _trainer(None)

    def _update():
        t._work_done = True  # EOT arrived; channel left mid-update
        raise RuntimeError("channel gone")
    t.check_and_update_state_avl = _update
    t.notify_trainer_avail()  # returns, no raise


def test_live_failure_still_raises():
    t = _trainer(lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError):
        t.notify_trainer_avail()


def test_stops_after_work_done():
    # The thread starts before run() sets _work_done, as in main().
    calls = []
    t = _trainer(None)
    del t._work_done

    def _update():
        calls.append(1)
        if len(calls) == 3:
            t._work_done = True
    t.check_and_update_state_avl = _update
    th = threading.Thread(target=t.notify_trainer_avail)
    th.start()
    th.join(timeout=2)
    assert not th.is_alive() and len(calls) == 3
