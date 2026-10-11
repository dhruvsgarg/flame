# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""FX-N77: opt-in wall-clock sampling profile of this process, every thread (`FLAME_PYSPY=<py-spy binary>`).

py-spy runs as this process's child attached by pid (PR_SET_PTRACER lets it under ptrace_scope 1) and writes
`<run_dir>/profile/<name>.<view>.txt` (collapsed stacks) when this process exits; views (`FLAME_PYSPY_VIEWS`):
`wall` = every sample incl. idle (where time goes), `cpu` = on-CPU samples only (what costs).
Report: `examples/scripts/profile_report.py <run_dir>`.
"""

from __future__ import annotations

import atexit
import ctypes
import logging
import os
import signal
import subprocess
from typing import List, Optional

logger = logging.getLogger(__name__)

ENV_PYSPY = "FLAME_PYSPY"
ENV_RATE = "FLAME_PYSPY_RATE"
ENV_VIEWS = "FLAME_PYSPY_VIEWS"
_VIEW_FLAGS = {"wall": ["--idle"], "cpu": [], "gil": ["--gil"]}
_PR_SET_PTRACER = 0x59616D61
_PR_SET_PTRACER_ANY = 2**64 - 1

_procs: List[subprocess.Popen] = []


def maybe_start(run_dir: str, name: str) -> Optional[str]:
    """Sample this process into `<run_dir>/profile/<name>.<view>.txt` if `FLAME_PYSPY` is set; returns the dir."""
    exe = os.environ.get(ENV_PYSPY)
    if not exe or _procs:
        return None
    out_dir = os.path.join(run_dir, "profile")
    try:
        os.makedirs(out_dir, exist_ok=True)
        ctypes.CDLL(None, use_errno=True).prctl(_PR_SET_PTRACER, ctypes.c_ulong(_PR_SET_PTRACER_ANY), 0, 0, 0)
        for view in os.environ.get(ENV_VIEWS, "wall,cpu").split(","):
            path = os.path.join(out_dir, f"{name}.{view}.txt")
            _procs.append(subprocess.Popen(
                [exe, "record", "--pid", str(os.getpid()), "--format", "raw", "--threads", "--nonblocking",
                 *_VIEW_FLAGS[view], "--rate", os.environ.get(ENV_RATE, "50"), "--output", path],
                stdout=subprocess.DEVNULL, stderr=open(path + ".log", "w")))
    except Exception as e:  # profiling must never break a run
        logger.warning(f"profiler not started: {e}")
        return None
    atexit.register(stop)
    logger.info(f"[PROFILER] py-spy x{len(_procs)} -> {out_dir}/{name}.*.txt")
    return out_dir


def stop(timeout_s: float = 20.0) -> None:
    """Flush the profiles: SIGINT makes py-spy write its output and exit."""
    procs = [p for p in _procs if p.poll() is None]
    _procs.clear()
    for p in procs:
        p.send_signal(signal.SIGINT)
    for p in procs:
        try:
            p.wait(timeout_s)
        except Exception:
            p.kill()
