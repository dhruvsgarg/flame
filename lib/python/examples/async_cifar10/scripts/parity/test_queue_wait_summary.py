# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""FX-N18: `analyze_send_recv_lag.queue_wait_summary` reads LAG_DECOMP queue waits and recv_fifo skips."""

from __future__ import annotations

import os
import sys

_SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from analyze_send_recv_lag import _agg_log, queue_wait_summary  # noqa: E402

_LOG = """\
x [LAG_DECOMP] end=a version=1 wall_lag_s=5.0 agg_to_trainer_s=0.1 compute_s=4.0 post_wait_s=0.1 mqtt_lag_s=0.1 queue_wait_s=0.002 process_s=0.1
x [LAG_DECOMP] end=b version=1 wall_lag_s=9.0 agg_to_trainer_s=0.1 compute_s=4.0 post_wait_s=0.1 mqtt_lag_s=0.1 queue_wait_s=11.000 process_s=0.1
x [RECV_FIFO] Skipping end_id b - already has active task, queue_depth=2
x [SEND_RECV_LAG] end=a version=1 wall_lag_s=5.0
"""


def test_summary_counts_waits_and_skips(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "x_aggregator.log").write_text(_LOG)
    q = queue_wait_summary(_agg_log(str(run)))
    assert q["n"] == 2 and q["active_task_skips"] == 1
    assert q["p50"] == 11.0 and q["max"] == 11.0 and q["over_1s"] == 1


def test_empty_log_has_no_percentiles(tmp_path):
    f = tmp_path / "a_aggregator.log"
    f.write_text("nothing here\n")
    assert queue_wait_summary(str(f)) == {"n": 0, "active_task_skips": 0}
