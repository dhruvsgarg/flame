# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Structured, process-local telemetry for FLAME experiments.

Each process (aggregator, each trainer) writes typed JSONL event records to a
per-run directory so post-run analysis can compare selectors / aggregators /
trainers on an identical schema -- no log-regex scraping.

The writer is a process-global singleton configured once at startup. When no
run directory is configured (neither ``configure(run_dir=...)`` nor the
``FLAME_TELEMETRY_DIR`` env var), every :func:`emit` is a cheap no-op, so
existing runs and unit tests are unaffected unless telemetry is explicitly
enabled.

Generic by design: nothing here is example-specific. The async_cifar10 example
wires it first, but any example/aggregator/selector/trainer inherits the same
hooks.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import math
import os
import threading
import time
from typing import Any, Optional

from .events import (  # noqa: F401  (re-exported for convenience)
    EVENT_AGG_EVAL,
    EVENT_AGG_ROUND,
    EVENT_MODEL_HEALTH,
    EVENT_AVAIL_CHANGE,
    EVENT_RUN_META,
    EVENT_SELECTION,
    EVENT_TRAINER_ROUND,
    EVENT_UTIL_DISPARITY,
    KNOWN_EVENTS,
)

logger = logging.getLogger(__name__)

ENV_DIR = "FLAME_TELEMETRY_DIR"
ENV_LEAN = "FLAME_TELEMETRY_LEAN"  # FX-D227: "1" drops `selection.per_trainer` (99% of bytes at n 300)
ENV_FULL = "FLAME_TELEMETRY_FULL"  # FX-D240: "1" keeps every selection's full `per_trainer`
ENV_SNAPSHOT_S = "FLAME_TELEMETRY_SNAPSHOT_S"  # FX-D240: full `per_trainer` snapshot period (clock s), default 600

_writer: Optional["TelemetryWriter"] = None
_config_lock = threading.Lock()


def _json_default(obj: Any) -> Any:
    """Best-effort JSON encoder for the value types telemetry carries."""
    if isinstance(obj, (set, frozenset)):
        return sorted(obj, key=str)
    if isinstance(obj, _dt.timedelta):
        return obj.total_seconds()
    if isinstance(obj, _dt.datetime):
        return obj.isoformat()
    # numpy scalars / arrays without importing numpy
    if hasattr(obj, "item") and not hasattr(obj, "__len__"):
        try:
            return obj.item()
        except Exception:
            pass
    if hasattr(obj, "tolist"):
        try:
            return obj.tolist()
        except Exception:
            pass
    return str(obj)


class TelemetryWriter:
    """Append-only JSONL writer. Thread-safe (trainers emit from threads)."""

    def __init__(self, path: str, role: str, end_id: Optional[str] = None) -> None:
        self.path = path
        self.role = role
        self.end_id = end_id
        self._lock = threading.Lock()
        # line-buffered so a killed process still leaves complete records
        self._fh = open(path, "a", buffering=1)
        self._lean = os.environ.get(ENV_LEAN) == "1"
        self._full = os.environ.get(ENV_FULL) == "1"
        self._snap_s = float(os.environ.get(ENV_SNAPSHOT_S) or 600)
        self._next_snap: dict = {}  # task -> clock of its next full snapshot

    def emit(self, event: str, **fields: Any) -> None:
        rec: dict[str, Any] = {
            "ts": time.time(),
            "role": self.role,
            "event": event,
        }
        if self.end_id is not None:
            rec["end_id"] = self.end_id
        rec.update(fields)
        if self._lean and event == "selection":
            rec.pop("per_trainer", None)
        elif event == "selection" and not self._full and isinstance(rec.get("per_trainer"), dict):
            self._trim_selection(rec)
        line = json.dumps(rec, default=_json_default)
        with self._lock:
            self._fh.write(line + "\n")

    def _trim_selection(self, rec: dict) -> None:
        """FX-D240: `per_trainer` = the chosen ends only, plus the whole pool once per snapshot period per task
        (marked `per_trainer_full`): the pool at n 300 was 97% of telemetry bytes."""
        task = rec.get("task")
        clock = float(rec.get("vclock_now") if rec.get("vclock_now") is not None else rec["ts"])
        if clock >= self._next_snap.get(task, -math.inf):
            self._next_snap[task] = clock + self._snap_s
            rec["per_trainer_full"] = True
            return
        chosen = set(rec.get("chosen") or ())
        rec["per_trainer"] = {k: v for k, v in rec["per_trainer"].items() if k in chosen}
        rec["per_trainer_full"] = False  # pool readers skip it; older logs (no flag) are full

    def close(self) -> None:
        with self._lock:
            try:
                self._fh.flush()
                self._fh.close()
            except Exception:
                pass


def configure(
    role: str,
    end_id: Optional[str] = None,
    run_dir: Optional[str] = None,
    filename: Optional[str] = None,
) -> Optional[TelemetryWriter]:
    """Initialize the process-global telemetry writer.

    Parameters
    ----------
    role: short role label, e.g. "aggregator" or "trainer".
    end_id: per-process identity (e.g. trainer/task id); used in the filename.
    run_dir: output directory. Falls back to ``$FLAME_TELEMETRY_DIR``. If still
        unset, telemetry stays disabled and this returns ``None``.
    filename: override the default ``<role>[_<end_id>].jsonl`` filename.

    Returns the writer, or ``None`` if telemetry is disabled.
    """
    global _writer
    run_dir = run_dir or os.environ.get(ENV_DIR)
    if not run_dir:
        return None

    if filename is None:
        safe_id = str(end_id).replace("/", "_") if end_id is not None else None
        filename = f"{role}_{safe_id}.jsonl" if safe_id is not None else f"{role}.jsonl"

    try:
        os.makedirs(run_dir, exist_ok=True)
        path = os.path.join(run_dir, filename)
        with _config_lock:
            if _writer is not None:
                _writer.close()
            _writer = TelemetryWriter(path, role, end_id)
        logger.info(f"telemetry enabled: writing to {path}")
        from flame.telemetry import profiler
        profiler.maybe_start(run_dir, filename.rsplit(".", 1)[0])  # FX-N77
        return _writer
    except Exception as e:  # never let telemetry setup break a run
        logger.warning(f"telemetry disabled (configure failed): {e}")
        _writer = None
        return None


def emit(event: str, **fields: Any) -> None:
    """Emit one event. No-op when telemetry is disabled. Never raises."""
    w = _writer
    if w is None:
        return
    try:
        w.emit(event, **fields)
    except Exception as e:  # telemetry must never crash the experiment
        logger.debug(f"telemetry emit failed for {event}: {e}")


def is_enabled() -> bool:
    return _writer is not None


def get_run_dir() -> Optional[str]:
    w = _writer
    return os.path.dirname(w.path) if w is not None else None


def shutdown() -> None:
    global _writer
    with _config_lock:
        if _writer is not None:
            _writer.close()
            _writer = None
