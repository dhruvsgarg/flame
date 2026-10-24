# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""FX-N141: fl1_verdict judges time to target by eval tick, not by commit-lagged minutes."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples" / "scripts"))
import bench_report  # noqa: E402
import fl1_verdict  # noqa: E402


def _s(tta, final=0.6, eval_min=10.0):
    return {"final": final, "tta0.5": tta, "eval_min": eval_min}


def test_same_tick_ties_despite_commit_lag():
    # low-c arm's eval lands 2 min after the tick FedBuff's lands on
    assert fl1_verdict.judge(_s(142.0), _s(140.5), [0.5], 0.01)[0] == "PASS"


def test_next_tick_is_slower():
    assert fl1_verdict.judge(_s(151.0), _s(140.5), [0.5], 0.01)[0] == "FAIL"


def test_no_eval_period_falls_back_to_one_minute():
    assert fl1_verdict.judge(_s(142.0, eval_min=None), _s(140.5, eval_min=None), [0.5], 0.01)[0] == "FAIL"


def test_eval_period_from_config(tmp_path):
    (tmp_path / "aggregator_config.json").write_text(json.dumps({"hyperparameters": {"evalEverySeconds": 600}}))
    assert bench_report.eval_period_min(str(tmp_path)) == 10.0
    assert bench_report.eval_period_min(str(tmp_path / "missing")) is None
