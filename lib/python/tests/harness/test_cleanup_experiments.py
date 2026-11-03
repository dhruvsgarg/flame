# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""cleanup_experiments: drops only old, unreferenced, uncited runs; keeps live, pooled, banked and cited ones."""

import importlib.util
import os
import sys
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "examples" / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("cleanup_experiments", SCRIPTS / "cleanup_experiments.py")
ce = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ce)


def _age(p: Path, days: float) -> None:
    t = time.time() - days * 86400
    for f in [*p.rglob("*"), p]:
        os.utime(f, (t, t))


def test_plan_keeps_referenced_cited_banked_recent(tmp_path, monkeypatch):
    pools, runs = tmp_path / "pools", tmp_path / "runs"
    for name in ("run_old", "run_pooled", "run_banked", "run_cited", "run_new"):
        (runs / name / "checkpoints").mkdir(parents=True)
        (runs / name / "checkpoints" / "round_00001.pt").write_bytes(b"x")
    (pools / "pool_new" / "P1" / "j" / "runs" / "1").mkdir(parents=True)
    (pools / "pool_new" / "P1" / "j" / "runs" / "1" / "legs.txt").write_text(f"{runs / 'run_pooled'}\n")
    (pools / "pool_old").mkdir()
    (pools / "_real_bank.tsv").write_text(f"ts\tkey\treal_dir\n1\tk\t{runs / 'run_banked'}\n")
    doc = tmp_path / "X.md"
    doc.write_text("evidence: run_cited")
    for name in ("run_old", "run_pooled", "run_banked", "run_cited"):
        _age(runs / name, 30)
    _age(pools / "pool_old", 30)
    _age(runs / "run_new", 3)
    monkeypatch.setattr(ce, "POOLS", pools)
    monkeypatch.setattr(ce, "RUNS", [runs])
    monkeypatch.setattr(ce, "tracked", lambda: {doc})
    drop_runs, drop_pools, ckpts = ce.plan(7, 2, False)
    assert [p.name for p in drop_runs] == ["run_old"]
    assert [p.name for p in drop_pools] == ["pool_old"]
    assert sorted(p.parent.parent.name for p in ckpts) == ["run_banked", "run_new", "run_pooled"]  # cited keeps its .pt
