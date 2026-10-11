# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""FX-N13 ST1-ST3: stream_schedule.py linear/events/stagger and the trace clock."""
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import stream_schedule as ss  # noqa: E402

ON = {"enabled": "True", "full_data_available_after_s": 1000}


def test_off_and_zero_horizon_show_everything():
    assert ss.from_config({"enabled": "False"}, "t") is None
    assert ss.from_config({**ON, "full_data_available_after_s": 0}, "t", scale=1).visible(50, 10) == 50


def test_linear_legacy_matches_old_formula():
    s = ss.from_config(ON, "t", scale=1)
    for t in (0, 1, 250, 999.9, 1000, 5000):
        assert s.visible(1000, t) == min(1000, max(1, math.floor(min(1.0, t / 1000) * 1000)))


def test_linear_initial_frac():
    s = ss.from_config({**ON, "initial_frac": 0.1}, "t", scale=1)
    assert s.visible(100, 0) == 10 and s.visible(100, 500) == 55 and s.visible(100, 1000) == 100


def test_trace_clock_scales_and_run_clock_does_not():
    assert ss.from_config(ON, "t", scale=4).visible(100, 125) == 50
    assert ss.from_config({**ON, "clock": "run"}, "t", scale=4).visible(100, 125) == 12


def test_trace_clock_reads_env_by_default(monkeypatch):
    monkeypatch.setenv("FLAME_TRACE_TIME_SCALE", "4")
    assert ss.from_config(ON, "t").visible(100, 125) == 50


def test_events_steps_deterministic_and_full_at_horizon():
    cfg = {**ON, "mode": "events", "initial_frac": 0.1, "n_chunks": 9}
    a, b = ss.from_config(cfg, "trainer7", scale=1), ss.from_config(cfg, "trainer7", scale=1)
    assert a == b and len(a.times) == 9 and list(a.times) == sorted(a.times)
    assert all(0 <= x <= 1000 for x in a.times)
    seen = [a.visible(1000, t) for t in range(0, 1001, 5)]
    assert seen[0] >= 100 and seen[-1] == 1000 and seen == sorted(seen)
    assert set(seen) <= {100 + 100 * k for k in range(10)}  # only whole chunks
    assert a.times != ss.from_config(cfg, "trainer8", scale=1).times  # out of sync across trainers
    assert a.times != ss.from_config({**cfg, "seed": 1}, "trainer7", scale=1).times


def test_stagger_matches_previous_derivation():
    cfg = {**ON, "stagger": {"enabled": "True", "onset_max_s": 300, "rate_jitter": 0.5, "min_visible": 4}}
    s = ss.from_config(cfg, "abc", scale=1)
    onset, span = ss.stagger_params("abc", 300, 1000, 0.5)
    assert (s.onset_s, s.span_s, s.min_visible) == (onset, span, 4)
    assert s.visible(100, 0) == 4


@pytest.mark.parametrize("bad", [{"mode": "burst"}, {"initial_frac": 1.5}, {"clock": "wall"},
                                 {"mode": "events", "stagger": {"enabled": "True"}}])
def test_bad_config_raises(bad):
    with pytest.raises(ValueError):
        ss.from_config({**ON, **bad}, "t")


def test_trainer_reports_count_it_trained_on():
    import types
    from main import PyTorchCifar10Trainer
    t = PyTorchCifar10Trainer.__new__(PyTorchCifar10Trainer)
    t._stream_sched = ss.from_config({**ON, "initial_frac": 0.1}, "t", scale=1)
    t._stream_total, t._stream_gpu = 100, False
    t._stream_full_dataset, t._stream_order = list(range(100)), __import__("torch").arange(100)
    t._stream_train_kwargs = {"batch_size": 4}
    now = [500.0]
    t._sim_now = lambda: now[0]
    t._rebuild_stream_loader()
    now[0] = 1000.0  # clock moves during the task; the report must not
    assert (t._stream_visible, t._stream_clock_s, len(t.train_loader.dataset)) == (55, 500.0, 55)
    assert t._visible_sample_count() == 100
