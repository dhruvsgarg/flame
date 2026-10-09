# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""FX-D102: FLAME_AGG_CMD_PREFIX wraps the aggregator command; unset leaves it unchanged."""

from flame.launch import aggregator_spawner as sp


def _cmd(monkeypatch, tmp_path, prefix):
    seen = {}

    class P:
        pid = 1

        def __init__(self, cmd, **kw):
            seen["cmd"] = cmd
    monkeypatch.setattr(sp.subprocess, "Popen", P)
    monkeypatch.setattr(sp.time, "sleep", lambda s: None)
    if prefix is None:
        monkeypatch.delenv("FLAME_AGG_CMD_PREFIX", raising=False)
    else:
        monkeypatch.setenv("FLAME_AGG_CMD_PREFIX", prefix)
    try:
        sp.AggregatorSpawner().spawn(tmp_path / "main.py", config_json="{}")
    except Exception:
        pass  # post-spawn checks on the fake process are irrelevant here
    return seen["cmd"]


def test_prefix_wraps_command(monkeypatch, tmp_path):
    plain = _cmd(monkeypatch, tmp_path, None)
    wrapped = _cmd(monkeypatch, tmp_path, "gdb -batch -ex 'thread apply all bt' --args")
    assert wrapped == ["gdb", "-batch", "-ex", "thread apply all bt", "--args"] + plain
