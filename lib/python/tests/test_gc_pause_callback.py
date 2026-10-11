# SPDX-License-Identifier: Apache-2.0
"""FX-D57: the GC pause callback survives interpreter exit (module global `time` cleared)."""
import flame.monitor.runtime as rt


def test_callback_survives_cleared_time_global(monkeypatch):
    monkeypatch.setattr(rt, "time", None)
    rt._gc_pause_callback("start", {})
    rt._gc_pause_callback("stop", {})
    assert rt._gc_pause_start is None
