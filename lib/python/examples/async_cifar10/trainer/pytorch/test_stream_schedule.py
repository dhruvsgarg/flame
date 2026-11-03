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


def test_usage_follows_the_legs_own_trace():
    """Q-STREAM: `trace` = the leg's availability trace; data accrues only off AVL_TRAIN on it; syn_0 grows linearly."""
    from flame.availability.trace import load_trace
    tid = "505f9fc483cf4df68a2409257b5fad7d3c580370"  # trainer_001
    cfg = {**ON, "mode": "usage", "initial_frac": 0.0, "trace": "syn_50_3st"}
    a = ss.from_config(cfg, tid, scale=1, offset=0.0)
    assert a.times != ss.from_config({**cfg, "trace": ""}, tid, scale=1, offset=0.0).times  # not the MobiPerf device
    tr = load_trace("syn_50_3st", "trainer_001")
    ts = list(tr.keys())
    for t0, t1 in zip(ts, ts[1:40]):  # constant while AVL_TRAIN, growing otherwise
        grew = a.frac(t1) > a.frac(t0)
        assert grew == (tr[t0] != "AVL_TRAIN"), (t0, tr[t0])
    lin = ss.from_config({**cfg, "trace": "syn_0"}, tid, scale=1, offset=0.0)
    span = ss.USAGE_SPAN_S
    assert lin.frac(0) == 0 and abs(lin.frac(span / 2) - 0.5) < 1e-9 and lin.frac(span) == 1.0
    with pytest.raises(KeyError):
        ss.from_config({**cfg, "trace": "syn_99"}, tid, scale=1)


def test_usage_index_matches_the_trace_and_its_source():
    """FX-N126: the compact per-trace index (no 14 MB YAML per trainer) records its source checksum and yields the same sessions."""
    from flame.availability.trace import _DEFAULT_TRACE_DIR, _raw_synthetic
    idx = ss._usage_index("syn_50_3st")
    assert idx["sha256"] == ss._sha256(os.path.join(str(_DEFAULT_TRACE_DIR), "synthetic_traces.yaml"))
    raw = _raw_synthetic(str(_DEFAULT_TRACE_DIR))["syn_50_3st"]["per_trainer"]["n300"]["trainer_002"]
    tid = "505f9fc483cf4df68a2409257b5fad7d3c580371"  # trainer_002
    sessions = ss.usage_sessions(tid, trace="syn_50_3st")
    use = [(float(t), st != "AVL_TRAIN") for t, st in raw]
    starts = [t for (t, u), (_, pu) in zip(use, [(None, False)] + use[:-1]) if u and not pu and t < ss.USAGE_SPAN_S]
    assert list(sessions[0]) == starts


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


def test_validation_holdout_never_trains_and_grows_with_the_stream():
    # F42c: every 10th arrival is held out for every baseline; the train loader never sees it
    import torch
    from main import PyTorchCifar10Trainer
    t = PyTorchCifar10Trainer.__new__(PyTorchCifar10Trainer)
    t._stream_sched = ss.from_config({**ON, "initial_frac": 0.1}, "t", scale=1)
    t._stream_total, t._stream_gpu = 100, False
    t._stream_full_dataset, t._stream_order = list(range(100)), torch.arange(100)
    t._val_mask, t.validation_probe = torch.arange(100) % 10 == 9, True
    t._stream_train_kwargs = {"batch_size": 4}
    now = [500.0]
    t._sim_now = lambda: now[0]
    t._rebuild_stream_loader()
    train = set(t.train_loader.dataset.indices)
    assert t._stream_visible == 55 and len(train) == 50 and set(t._val_pos.tolist()) == {9, 19, 29, 39, 49}
    assert not train & set(t._val_pos.tolist())
    assert len(t._val_loader().dataset) == 5
    now[0] = 2000.0
    t._rebuild_stream_loader()
    assert len(t._val_pos) == 10 and t._train_visible == 90


def test_val_report_is_exact_eval_mode_and_paired_with_the_previous_report():
    # FX-N127 V2 / P1: the report = nll of the received model on the held-out samples (eval mode, buffers kept);
    # d_* pair this client's current loss with its previous report on the same (first n_paired) samples.
    import contextlib
    import torch
    import torch.nn.functional as F
    from main import PyTorchCifar10Trainer
    torch.manual_seed(0)
    t = PyTorchCifar10Trainer.__new__(PyTorchCifar10Trainer)
    X, Y = torch.randn(100, 4), torch.randint(0, 3, (100,))
    t._stream_sched = ss.from_config({**ON, "initial_frac": 0.1}, "t", scale=1)
    t._stream_total, t._stream_gpu = 100, False
    t._stream_full_dataset, t._stream_order = torch.utils.data.TensorDataset(X, Y), torch.arange(100)
    t._val_mask, t.validation_probe, t.validation_report = torch.arange(100) % 10 == 9, False, True
    t._stream_train_kwargs, t.device, t._amp, t._vr_prev = {"batch_size": 4}, "cpu", contextlib.nullcontext, None
    t.model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.BatchNorm1d(8), torch.nn.Linear(8, 3), torch.nn.LogSoftmax(1))
    t.model.train()
    now = [500.0]
    t._sim_now = lambda: now[0]
    t._rebuild_stream_loader()
    t._round = 7
    bufs = {k: v.clone() for k, v in t.model.named_buffers()}
    rep = t._val_report()
    pos = t._val_pos
    t.model.eval()
    with torch.no_grad():
        nll = F.nll_loss(t.model(X[pos]), Y[pos], reduction="none")
    t.model.train()
    assert t.model.training and all(torch.equal(v, bufs[k]) for k, v in t.model.named_buffers())
    assert rep[:3] == (pytest.approx(float(nll.sum()), rel=1e-6), rep[1], 5) and rep[3:] == (0.0, 0, 0, -1)
    with torch.no_grad():
        t.model[0].weight.mul_(2)  # a new model version arrives; the stream grows
    now[0], t._round = 2000.0, 9
    t._rebuild_stream_loader()
    rep2 = t._val_report()
    t.model.eval()
    with torch.no_grad():
        nll2 = F.nll_loss(t.model(X[t._val_pos]), Y[t._val_pos], reduction="none")
    assert rep2[2] == 10 and rep2[5] == 5 and rep2[6] == 7
    assert rep2[3] == pytest.approx(float(nll2[:5].sum() - nll.sum()), rel=1e-5)
