# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""expt_assert_run's verdict: a SIGKILLed aggregator leaves no traceback, so the
runner's `aggregator FAILED (exit=...)` line is the only crash signal."""

import os
import subprocess
import time
from pathlib import Path

RUNNER = Path(__file__).resolve().parents[2] / "examples" / "scripts" / "expt_runner.sh"


def _verdict(tmp_path, out_text, agg_log="", agg_rounds=3):
    logdir = tmp_path / "logs"
    run = tmp_path / "ex" / "experiments" / "run_x" / "telemetry"
    logdir.mkdir()
    run.mkdir(parents=True)
    marker = logdir / ".marker_arm"
    marker.write_text("")
    old = time.time() - 60
    os.utime(marker, (old, old))
    (run / "aggregator_0.jsonl").write_text('{"event": "agg_round"}\n' * agg_rounds)
    (run.parent / "x_aggregator.log").write_text(agg_log)
    (logdir / "arm.out").write_text(out_text)
    script = (f'source "{RUNNER}" >/dev/null 2>&1; '
              f'expt_assert_run "{tmp_path / "ex"}" "{marker}" arm; '
              f'echo "rc=$? health=$EXPT_LAST_HEALTH"')
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True,
                       env={**os.environ, "FLAME_RUN_DIR_FILE": ""})
    return r.stdout.strip().splitlines()[-1]


def test_clean_run_is_completed(tmp_path):
    out = "  aggregator done (exit=0), waiting for trainers to exit...\n"
    assert _verdict(tmp_path, out) == "rc=0 health=COMPLETED"


def test_killed_aggregator_is_not_completed(tmp_path):
    out = "  aggregator FAILED (exit=-9 ⚠); terminating trainers now (skipping EOT grace).\n"
    assert _verdict(tmp_path, out) == "rc=1 health=AGG_KILLED"


def test_traceback_still_wins(tmp_path):
    out = "  aggregator FAILED (exit=1 ⚠); terminating trainers now (skipping EOT grace).\n"
    log = "Traceback (most recent call last):\n  boom\n"
    assert _verdict(tmp_path, out, agg_log=log) == "rc=1 health=CRASH"
