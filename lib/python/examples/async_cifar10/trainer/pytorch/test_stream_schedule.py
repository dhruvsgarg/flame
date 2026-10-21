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


def test_usage_accrues_through_off_charger_sessions():
    """FX-N120: data accrues linearly while device_001 is off the charger (UN_AVL 0-753 s), and holds while it charges."""
    cfg = {**ON, "mode": "usage", "initial_frac": 0.0}
    tid = "505f9fc483cf4df68a2409257b5fad7d3c580370"  # trainer_001 -> device_001
    a = ss.from_config(cfg, tid, scale=1)
    assert a == ss.from_config(cfg, tid, scale=1) and a.times[0] == 0.0 and a.ends[0] == 753.0
    assert 0 < a.frac(376.5) < a.frac(753) and abs(a.frac(376.5) - a.frac(753) / 2) < 1e-9  # linear within the session
    assert a.frac(753) == a.frac(6000)  # charging 753-6862 s: no new data
    assert a.frac(7000) > a.frac(6862)  # next off-charger session accrues again
    seen = [a.visible(1000, t) for t in range(0, 6 * 86400 + 1, 600)]
    assert seen == sorted(seen) and seen[-1] == 1000
    assert a.times != ss.from_config(cfg, "505f9fc483cf4df68a2409257b5fad7d3c580371", scale=1).times
    assert ss.from_config(cfg, "trainer7", scale=1).times  # non-registry ids get a seeded device


def test_usage_reads_the_trace_offset(monkeypatch):
    """FX-N123: a leg that starts `offset` into the trace sees the data accrued by then at its t = 0."""
    cfg = {**ON, "mode": "usage", "initial_frac": 0.0}
    tid = "505f9fc483cf4df68a2409257b5fad7d3c580370"
    import dataclasses
    base = dataclasses.replace(ss.from_config(cfg, tid, scale=1), offset=0.0)
    monkeypatch.setenv("FLAME_TRACE_OFFSET_S", "86400")
    off = ss.from_config(cfg, tid, scale=1)
    assert off.frac(0) == base.frac(86400) and off.frac(3600) == base.frac(90000)


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


@pytest.mark.parametrize("on, clock", [(True, 500.4), (False, 500.0)])
def test_sim_stream_clock_at_train_start(on, clock):
    # FX-D160: real reads the stream after its download; sim read it at the dispatch stamp (speech: 13/67 tasks forked).
    from main import PyTorchCifar10Trainer
    t = PyTorchCifar10Trainer.__new__(PyTorchCifar10Trainer)
    t.simulated, t._sim_stream_at_start, t._sim_charge_lag, t.sim_download_leg_s = True, on, True, 0.2
    t._sim_send_ts, t._sim_wall_send_ts, t._wall_recv_ts = 500.0, 10.0, 10.4
    assert t._stream_now() == pytest.approx(clock)
