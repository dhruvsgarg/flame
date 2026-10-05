# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""FX-N13 streaming schedule: how much of a trainer's local pool is visible at a given time.

One definition for every reader (trainer, oracle injection, offline replay, EV19), so real, sim and the
checker agree by construction. Config is `hyperparameters.data_streaming`:

  enabled                      "True" to stream
  mode                         linear (default) | events
  full_data_available_after_s  horizon T: trace seconds until 100% is visible
  initial_frac                 share visible at t=0 (ST1; default 0 = one sample)
  n_chunks                     events: equal chunks after the initial share (default 9)
  seed                         events: schedule seed (default 0)
  clock                        trace (default: stream clock x FLAME_TRACE_TIME_SCALE, ST3) | run
  stagger                      linear only: per-trainer onset/span ({enabled, onset_max_s, rate_jitter, min_visible})

linear: visible = initial + (1 - initial) * (t - onset) / span.
events: visible = initial + (1 - initial) * k / n_chunks, k = chunks whose seeded uniform time in [0, T] is <= t.
The stream clock is the trainers' own: vclock (sim), wall since the broadcast AGG_START_TS (real); FX-L30.
"""
from __future__ import annotations

import bisect
import hashlib
import math
import random
from dataclasses import dataclass, field
from typing import Optional

from flame.availability.trace import trace_time_scale

MODES = ("linear", "events")


def _true(v) -> bool:
    return str(v).lower() == "true"  # T13


def stagger_params(trainer_id, onset_max_s, base_span_s, rate_jitter):
    """Per-trainer (onset, span) from disjoint 32-bit slices of sha256(f"{trainer_id}:stagger")."""
    h = hashlib.sha256(f"{trainer_id}:stagger".encode()).hexdigest()
    u1 = int(h[0:8], 16) / 0xFFFFFFFF
    u2 = int(h[8:16], 16) / 0xFFFFFFFF
    span_s = base_span_s * (1.0 + rate_jitter * (2.0 * u2 - 1.0))
    return onset_max_s * u1, max(base_span_s / 4.0, span_s)


def chunk_times(trainer_id, horizon_s: float, n_chunks: int, seed: int = 0) -> list:
    """Sorted chunk arrival times (trace s), uniform in [0, horizon_s], deterministic in (trainer_id, seed)."""
    rng = random.Random(int(hashlib.sha256(f"{trainer_id}:{seed}:events".encode()).hexdigest()[:16], 16))
    return sorted(rng.uniform(0.0, horizon_s) for _ in range(n_chunks))


@dataclass(frozen=True)
class StreamSchedule:
    mode: str = "linear"
    horizon_s: float = 0.0
    initial_frac: float = 0.0
    onset_s: float = 0.0
    span_s: float = 0.0
    min_visible: int = 1
    trace_clock: bool = True
    times: tuple = field(default=())  # events only
    scale: Optional[float] = None  # None = this process's FLAME_TRACE_TIME_SCALE; a checker passes the run's (L29)

    def trace_t(self, stream_clock_s: float) -> float:
        if not self.trace_clock:
            return float(stream_clock_s)
        return float(stream_clock_s) * (trace_time_scale() if self.scale is None else self.scale)

    def frac(self, stream_clock_s: float) -> float:
        """Visible share of the pool at this stream-clock instant."""
        if self.horizon_s <= 0:
            return 1.0
        t = self.trace_t(stream_clock_s)
        if self.mode == "events":
            grown = bisect.bisect_right(self.times, t) / len(self.times) if self.times else 1.0
        else:
            span = self.span_s if self.span_s > 0 else self.horizon_s
            grown = min(1.0, max(0.0, (t - self.onset_s) / span))
        return min(1.0, self.initial_frac + (1.0 - self.initial_frac) * grown)

    def visible(self, total: int, stream_clock_s: float) -> int:
        """Visible sample count, >= min_visible so the loader is never empty."""
        if self.horizon_s <= 0:
            return total
        return min(total, max(self.min_visible, math.floor(self.frac(stream_clock_s) * total + 1e-9)))


def from_config(ds_cfg: Optional[dict], trainer_id, scale: Optional[float] = None) -> Optional[StreamSchedule]:
    """The trainer's schedule, or None when streaming is off."""
    ds = ds_cfg or {}
    if not _true(ds.get("enabled", "False")):
        return None
    mode = str(ds.get("mode", "linear") or "linear")
    if mode not in MODES:
        raise ValueError(f"data_streaming.mode={mode!r}: expected one of {MODES}")
    horizon = float(ds.get("full_data_available_after_s", 0) or 0)
    initial = float(ds.get("initial_frac", 0.0) or 0.0)
    if not 0.0 <= initial <= 1.0:
        raise ValueError(f"data_streaming.initial_frac={initial} not in [0, 1]")
    clock = str(ds.get("clock", "trace") or "trace")
    if clock not in ("trace", "run"):
        raise ValueError(f"data_streaming.clock={clock!r}: expected trace or run")
    stg = ds.get("stagger") or {}
    kw = dict(mode=mode, horizon_s=horizon, initial_frac=initial, trace_clock=clock == "trace", scale=scale)
    if mode == "events":
        if _true(stg.get("enabled", "False")):
            raise ValueError("data_streaming.stagger applies to mode=linear only")
        n = int(ds.get("n_chunks", 9) or 9)
        kw["times"] = tuple(chunk_times(trainer_id, horizon, n, int(ds.get("seed", 0) or 0)))
    elif _true(stg.get("enabled", "False")) and horizon > 0:
        kw["onset_s"], kw["span_s"] = stagger_params(trainer_id, float(stg.get("onset_max_s", 0.0) or 0.0),
                                                     horizon, float(stg.get("rate_jitter", 0.0) or 0.0))
        kw["min_visible"] = int(stg.get("min_visible", 1) or 1)
    return StreamSchedule(**kw)
