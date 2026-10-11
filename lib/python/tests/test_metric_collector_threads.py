"""FX-N36: MetricCollector starts its NVML/psutil poll threads only when FLAME_STAT_THREADS=1."""

import threading

from flame.monitor.metric_collector import MetricCollector


def _started(monkeypatch):
    started = []
    monkeypatch.setattr(threading.Thread, "start", lambda self: started.append(self))
    return started


def test_no_poll_threads_by_default(monkeypatch):
    monkeypatch.delenv("FLAME_STAT_THREADS", raising=False)
    started = _started(monkeypatch)
    mc = MetricCollector()
    assert started == []
    mc.accumulate("bytes", "send", 3)
    assert mc.get() == {"send.bytes": 3}


def test_poll_threads_opt_in(monkeypatch):
    monkeypatch.setenv("FLAME_STAT_THREADS", "1")
    started = _started(monkeypatch)
    MetricCollector()
    assert len(started) == 2
