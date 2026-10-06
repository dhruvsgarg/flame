# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""FX-D45: a leg that stops committing rounds (S1) or stops logging (S2) trips; a slow but progressing one never does."""

import importlib.util
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "examples" / "scripts"
spec = importlib.util.spec_from_file_location("fail_fast", SCRIPTS / "fail_fast.py")
ff = importlib.util.module_from_spec(spec)
sys.modules["fail_fast"] = ff
spec.loader.exec_module(ff)


def _leg(tmp_path):
    run = tmp_path / "run_x"
    (run / "telemetry").mkdir(parents=True)
    return run, run / "telemetry" / "aggregator_x.jsonl", run / "x_aggregator.log"


def _append(f, text):
    with open(f, "a") as fh:
        fh.write(text)


def test_slow_but_progressing_never_trips(tmp_path):
    run, tel, log = _leg(tmp_path)
    w = ff.StallWatch(0.0, no_round_s=600, silent_s=300)
    for t in range(0, 6000, 500):  # a round every 500 s: slow, below both limits
        _append(tel, '{"event": "agg_round", "round": 1}\n')
        assert w.check([run], float(t)) == ""
    assert w.rounds == 12


def test_no_new_round_trips_s1_even_while_logging(tmp_path):
    run, tel, log = _leg(tmp_path)
    w = ff.StallWatch(0.0, no_round_s=600, silent_s=300)
    _append(tel, '{"event": "agg_round", "round": 1}\n')
    assert w.check([run], 10.0) == ""
    for t in range(100, 700, 100):
        _append(log, "still selecting\n")
        _append(tel, '{"event": "selection"}\n')
        w.check([run], float(t))
    assert w.check([run], 700.0).startswith("S1")


def test_silent_leg_trips_s2(tmp_path):
    run, tel, log = _leg(tmp_path)
    w = ff.StallWatch(0.0, no_round_s=0, silent_s=300)
    _append(log, "x\n")
    assert w.check([run], 10.0) == ""
    assert w.check([run], 400.0).startswith("S2")


def test_partial_line_waits(tmp_path):
    run, tel, log = _leg(tmp_path)
    w = ff.StallWatch(0.0)
    _append(tel, '{"event": "agg_round"')
    w.check([run], 1.0)
    assert w.rounds == 0
    _append(tel, ', "round": 1}\n')
    w.check([run], 2.0)
    assert w.rounds == 1


def test_ended_run_never_trips(tmp_path):
    # FX-D62: run 16 G1A legs were cut in post-run analysis, 10 min after the aggregator's last line.
    run, tel, log = _leg(tmp_path)
    w = ff.StallWatch(0.0, no_round_s=600, silent_s=300)
    _append(tel, '{"event": "agg_round", "round": 1}\n{"event": "run_end", "round": 1}\n')
    assert w.check([run], 10.0) == ""
    assert w.check([run], 5000.0) == ""
