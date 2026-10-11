# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""FLAME_TRACE_TIME_SCALE compresses every trace consumer identically (harness only)."""

import pytest

from flame.availability import trace as tr


def test_unset_is_identity(monkeypatch):
    monkeypatch.delenv("FLAME_TRACE_TIME_SCALE", raising=False)
    ev = [[0, "AVL_TRAIN"], [600, "UN_AVL"]]
    assert tr.scale_events(ev) is ev


def test_load_trace_scaled(monkeypatch):
    monkeypatch.delenv("FLAME_TRACE_TIME_SCALE", raising=False)
    base = tr.load_trace("mobiperf_3st_50", "trainer_001")
    monkeypatch.setenv("FLAME_TRACE_TIME_SCALE", "4")
    scaled = tr.load_trace("mobiperf_3st_50", "trainer_001")
    assert list(scaled.keys()) == pytest.approx([t / 4 for t in base.keys()])
    assert list(scaled.values()) == list(base.values())


def test_state_and_next_avail_consistent_under_scale(monkeypatch):
    monkeypatch.setenv("FLAME_TRACE_TIME_SCALE", "4")
    t = tr.load_trace("syn_50", "trainer_001")
    base_ts = sorted(t.keys())
    for a, b in zip(base_ts, base_ts[1:]):
        mid = (a + b) / 2
        assert tr.state_at(t, mid).value == t[a]


def test_spawner_mobiperf_scaled(monkeypatch):
    from pathlib import Path
    from flame.launch.spawner import MetadataLoader
    md = Path(tr.__file__).resolve().parents[2] / "examples" / "_metadata"
    ml = MetadataLoader(md)
    monkeypatch.delenv("FLAME_TRACE_TIME_SCALE", raising=False)
    base = ml.get_mobiperf_trace(1, "3st_50")
    monkeypatch.setenv("FLAME_TRACE_TIME_SCALE", "2")
    assert [e[0] for e in ml.get_mobiperf_trace(1, "3st_50")] == pytest.approx([e[0] / 2 for e in base])


def test_rejects_nonpositive(monkeypatch):
    monkeypatch.setenv("FLAME_TRACE_TIME_SCALE", "0")
    with pytest.raises(ValueError):
        tr.trace_time_scale()


def test_bare_mobiperf_3st_aliases_to_50(monkeypatch):
    monkeypatch.delenv("FLAME_TRACE_TIME_SCALE", raising=False)
    assert dict(tr.load_trace("mobiperf_3st", "trainer_007")) == dict(tr.load_trace("mobiperf_3st_50", "trainer_007"))


def test_spawner_mobiperf_3st_mode(monkeypatch):
    from pathlib import Path
    from flame.launch.spawner import MetadataLoader
    md = Path(tr.__file__).resolve().parents[2] / "examples" / "_metadata"
    monkeypatch.delenv("FLAME_TRACE_TIME_SCALE", raising=False)
    ml = MetadataLoader(md)
    assert ml.get_mobiperf_trace(7, "3st_50")
