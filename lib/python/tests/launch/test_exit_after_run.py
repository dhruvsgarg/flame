# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""FX-D178: exit_after_run joins threads, runs atexit hooks and flushes before os._exit."""

import subprocess
import sys

_SCRIPT = """
import atexit, threading, time, logging, sys
logging.basicConfig(filename=sys.argv[1] + '.log', level=logging.INFO)
threading.Thread(target=lambda: (time.sleep(0.2), open(sys.argv[1] + '.thread', 'w').write('t'))).start()
atexit.register(lambda: open(sys.argv[1] + '.atexit', 'w').write('a'))
logging.info('done')
print('out', end='')
from flame.launch.cli import exit_after_run
exit_after_run(3)
raise SystemExit(9)
"""


def test_exit_after_run_runs_hooks_then_exits(tmp_path):
    base = str(tmp_path / "x")
    p = subprocess.run([sys.executable, "-c", _SCRIPT, base], capture_output=True, text=True, timeout=60)
    assert p.returncode == 3, p.stderr
    assert p.stdout == "out"
    for suffix, want in ((".thread", "t"), (".atexit", "a")):
        assert open(base + suffix).read() == want
    assert "done" in open(base + ".log").read()


def test_exit_after_run_off_returns(tmp_path):
    p = subprocess.run([sys.executable, "-c", _SCRIPT, str(tmp_path / "y")], capture_output=True, text=True, timeout=60,
                       env={**__import__("os").environ, "FLAME_HARD_EXIT": "0"})
    assert p.returncode == 9, p.stderr


def test_aggregator_mains_end_with_exit_after_run():
    # PL24: a main that skips it aborts in native teardown (fwdllm missed it until 10-24).
    import pathlib
    ex = pathlib.Path(__file__).resolve().parents[2] / "examples"
    mains = [*ex.glob("async_cifar10/aggregator/pytorch/main_*_agg.py"), ex / "fwdllm/aggregator/main_fedfwd_agg.py"]
    missing = [str(m) for m in mains if "exit_after_run()" not in m.read_text()]
    assert len(mains) >= 4 and not missing, missing
