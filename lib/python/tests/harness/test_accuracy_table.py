# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""FX-N74: accuracy windows count once the leg ran through them, not once an eval landed past them."""

import importlib.util
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "examples" / "scripts"
spec = importlib.util.spec_from_file_location("accuracy_table", SCRIPTS / "accuracy_table.py")
acc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(acc)


def test_window_reads_last_eval_when_leg_ran_through_it():
    c = [(60 * m, r, a) for m, r, a in ((30, 100, 0.2), (60, 200, 0.4), (84.6, 300, 0.507))]
    s = acc.summarize(c, 0.6, end=90.2 * 60)  # run 17 speech felix sim: last eval at 84.6 of 90.2 min
    assert s["acc@90m"] == 0.507 and s["acc@120m"] is None
    assert acc.summarize(c, 0.6)["acc@90m"] is None  # no leg end: the last eval bounds it
    assert acc.summarize(c, 0.4, end=90 * 60)["t_to_target_min"] == 60
