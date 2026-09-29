# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""FX-N40: a fatal line in any leg's logs stops the batch; the FX-N33 exit abort does not."""

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

SCRIPTS = Path(__file__).resolve().parents[2] / "examples" / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ff = _load("fail_fast")
OOM = ("x | CRITICAL | custom_excepthook | Uncaught exception:\nTraceback (most recent call last):\n"
       '  File "main.py", line 454, in _warmup_device\ntorch.OutOfMemoryError: CUDA out of memory.\n')
FXN33 = "channel leave done for param-channel\nterminate called without an active exception\nFatal Python error: Aborted\n"


def _leg(tmp_path, name, **logs):
    run = tmp_path / f"run_{name}"
    run.mkdir()
    for fname, text in logs.items():
        (run / fname).write_text(text)
    out = tmp_path / name
    (out / "runs" / "syn_0_felix").mkdir(parents=True)
    (out / "runs" / "syn_0_felix" / "legs.txt").write_text(f"{run}\n")
    return out, run


def test_traceback_is_fatal_with_context(tmp_path):
    out, run = _leg(tmp_path, "a", **{"x_trainers.log": "ok\n" + OOM, "x_resources.log": OOM})
    found = ff.Scanner().scan(ff.leg_run_dirs(out))
    assert [(f.path.name, f.lineno) for f in found] == [("x_trainers.log", 3)]  # the node monitor is skipped
    assert "OutOfMemoryError" in found[0].context()


def test_fxn33_exit_abort_is_benign_but_a_bare_abort_is_not(tmp_path):
    out, _ = _leg(tmp_path, "a", **{"x_aggregator.log": FXN33})
    assert ff.Scanner().scan(ff.leg_run_dirs(out)) == []
    out, _ = _leg(tmp_path, "b", **{"x_aggregator.log": "Fatal Python error: Aborted\n"})
    assert len(ff.Scanner().scan(ff.leg_run_dirs(out))) == 1


def test_scan_is_incremental_and_waits_for_a_whole_line(tmp_path):
    _, run = _leg(tmp_path, "a", **{"x.log": "ok\nTraceback (most rec"})
    s = ff.Scanner()
    assert s.scan([run]) == []
    with open(run / "x.log", "a") as f:
        f.write("ent call last):\nValueError: x\n")
    assert [f.lineno for f in s.scan([run])] == [2]
    assert s.scan([run]) == []  # reported once


def test_cli_exit_code_and_abort_file(tmp_path):
    _, run = _leg(tmp_path, "a", **{"x.log": OOM})
    abort = tmp_path / "ABORT.txt"
    r = subprocess.run([sys.executable, str(SCRIPTS / "fail_fast.py"), str(run), "--abort-file", str(abort)],
                       capture_output=True, text=True)
    assert r.returncode == ff.EXIT_FATAL and "OutOfMemoryError" in abort.read_text()
    _, clean = _leg(tmp_path, "b", **{"x.log": "ok\n"})
    assert subprocess.run([sys.executable, str(SCRIPTS / "fail_fast.py"), str(clean)]).returncode == 0


def test_pool_aborts_on_a_fatal_leg(tmp_path):
    pool = _load("harness_pool")
    out, _ = _leg(tmp_path, "gs_G0_syn_20_felix_real", **{"x_trainers.log": OOM})
    p = pool.Pool(tmp_path, [], 1, 0, 0, 1, False)
    r = SimpleNamespace(out=out, job=SimpleNamespace(jid="gs_G0_syn_20_felix_real"))
    assert p._fatal(r) and p.aborted
    assert "gs_G0_syn_20_felix_real" in (tmp_path / "ABORT.txt").read_text()
    q = pool.Pool(tmp_path / "q", [], 1, 0, 0, 1, False, fail_fast=False)
    (tmp_path / "q").mkdir()
    assert not q._fatal(r) and not q.aborted


def test_broker_duplicate_client_id_is_fatal(tmp_path):
    # Run 4 gs_P11a: a neighbour pool's leg joined this leg's broker; fixed client ids kicked each other.
    log = tmp_path / "mosquitto.log"
    log.write_text("1790680237: New client connected from 127.0.0.1:1 as agg (p5, c1, k300).\n"
                   "1790680237: Client agg already connected, closing old connection.\n")
    found = ff.Scanner().scan_broker(log)
    assert len(found) == 1 and found[0].lineno == 2
    assert ff.Scanner().scan_broker(tmp_path / "absent.log") == []
