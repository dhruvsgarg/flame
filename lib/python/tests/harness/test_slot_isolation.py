# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""FX-N22 P1: per-slot broker override and exact run-dir handoff (byte-identical when unset)."""

from types import SimpleNamespace

from flame.config import transform_brokers
from flame.launch.runner import ExperimentRunner

BROKERS = [{"host": "localhost", "sort": "mqtt"}, {"host": "localhost:10104", "sort": "p2p"}]


def test_broker_unchanged_without_env(monkeypatch):
    monkeypatch.delenv("FLAME_MQTT_BROKER", raising=False)
    assert transform_brokers(BROKERS).sort_to_host == {"mqtt": "localhost", "p2p": "localhost:10104"}


def test_slot_broker_overrides_mqtt_only(monkeypatch):
    monkeypatch.setenv("FLAME_MQTT_BROKER", "localhost:18831")
    assert transform_brokers(BROKERS).sort_to_host == {"mqtt": "localhost:18831", "p2p": "localhost:10104"}


def test_run_dir_file_records_each_leg(tmp_path, monkeypatch):
    rec = tmp_path / "legs.txt"
    monkeypatch.setenv("FLAME_RUN_DIR_FILE", str(rec))
    fake = SimpleNamespace(experiments_dir=tmp_path / "experiments")
    d1 = ExperimentRunner._create_experiment_directory(fake, SimpleNamespace(name="dbg_felix_real"))
    d2 = ExperimentRunner._create_experiment_directory(fake, SimpleNamespace(name="dbg_felix_sim"))
    assert rec.read_text().splitlines() == [str(d1), str(d2)]


def test_no_run_dir_file_without_env(tmp_path, monkeypatch):
    monkeypatch.delenv("FLAME_RUN_DIR_FILE", raising=False)
    fake = SimpleNamespace(experiments_dir=tmp_path / "experiments")
    ExperimentRunner._create_experiment_directory(fake, SimpleNamespace(name="x"))
    assert [p.name for p in tmp_path.iterdir()] == ["experiments"]


def test_same_second_slots_get_distinct_run_dirs(tmp_path, monkeypatch):
    """FX-D13: two slots launching the same leg in one second never share a run dir."""
    import datetime as dt

    import pytest

    import flame.launch.runner as runner_mod

    class _Frozen(dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 9, 26, 20, 0, 7)

    monkeypatch.setattr(runner_mod, "datetime", _Frozen)
    monkeypatch.delenv("FLAME_RUN_DIR_FILE", raising=False)
    fake = SimpleNamespace(experiments_dir=tmp_path)
    exp = SimpleNamespace(name="dbg_fedbuff_real")
    dirs = []
    for label in ("gs_P1", "gs_P10"):
        monkeypatch.setenv("FLAME_RUN_LABEL", label)
        dirs.append(ExperimentRunner._create_experiment_directory(fake, exp))
    assert dirs[0].name == "run_20260926_200007_gs_P1_dbg_fedbuff_real" and dirs[0] != dirs[1]
    with pytest.raises(FileExistsError):  # same label, same second: refuse to share
        ExperimentRunner._create_experiment_directory(fake, exp)


def test_slot_pids_sees_only_its_own_slot(monkeypatch):
    import os
    import subprocess
    import sys
    import time

    from flame.launch.runner import slot_pids

    marker = f"slotpidtest_{os.getpid()}"
    cmd = [sys.executable, "-c", f"import time; time.sleep(60)  # {marker}"]
    base = {k: v for k, v in os.environ.items() if k != "FLAME_RUN_TAG"}
    procs = {tag: subprocess.Popen(cmd, env={**base, **({"FLAME_RUN_TAG": tag} if tag else {})})
             for tag in ("a", "b", "")}
    try:
        time.sleep(0.5)
        for tag, p in procs.items():
            if tag:
                monkeypatch.setenv("FLAME_RUN_TAG", tag)
            else:
                monkeypatch.delenv("FLAME_RUN_TAG", raising=False)
            assert slot_pids(marker) == [p.pid], tag
    finally:
        for p in procs.values():
            p.kill()
