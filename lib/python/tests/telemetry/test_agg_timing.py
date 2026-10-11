# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""FX-N43: agg_timing splits a commit's wall into recv wait, ingest, commit and the rest."""

import time

from flame import telemetry
from flame.telemetry.agg_timing import AggTiming


def _slow(items, s):
    for x in items:
        time.sleep(s)
        yield x


def test_iterate_splits_recv_from_ingest_and_counts_a_break(monkeypatch):
    calls = []
    monkeypatch.setattr(telemetry, "is_enabled", lambda: True)
    monkeypatch.setattr(telemetry, "emit", lambda ev, **f: calls.append((ev, f)))
    t = AggTiming()
    t.begin()
    for i in t.iterate(_slow([1, 2, 3], 0.05)):
        time.sleep(0.02)
        if i == 2:
            break
    assert t.n == 2 and 0.09 <= t.recv_wait_s < 0.2 and 0.035 <= t.ingest_s < 0.1
    time.sleep(0.03)
    t.emit(7, commit_s=0.01, simulated=False)
    ev, f = calls[0]
    assert ev == "agg_timing" and f["round"] == 7 and f["n_updates"] == 2 and f["time_mode"] == "real"
    assert f["other_s"] >= 0.02 and abs(f["cycle_s"] - f["recv_wait_s"] - f["ingest_s"] - f["commit_s"] - f["other_s"]) < 1e-3
    assert t.t0 is None and t.n == 0  # reset for the next version


def test_begin_keeps_the_first_pass_of_a_version():
    t = AggTiming()
    t.begin()
    t0 = t.t0
    t.begin()  # a sync wait-K version spans several _aggregate_weights passes
    assert t.t0 == t0
