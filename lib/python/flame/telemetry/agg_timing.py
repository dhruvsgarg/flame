# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""FX-N43: per-commit wall split of an aggregation cycle (`agg_timing` event), both modes, every stack."""

from __future__ import annotations

import time
from typing import Iterable, Iterator, Optional

from flame import telemetry
from flame.telemetry.events import build_agg_timing


class AggTiming:
    """Accumulates one version's recv wait and ingest across `_aggregate_weights` passes until its commit."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.t0: Optional[float] = None
        self.recv_wait_s = self.ingest_s = 0.0
        self.n = 0

    def begin(self) -> None:
        if self.t0 is None:
            self.t0 = time.time()

    def add_recv(self, s: float) -> None:
        self.recv_wait_s += s

    def add_ingest(self, s: float) -> None:
        self.ingest_s += s
        self.n += 1

    def iterate(self, it: Iterable) -> Iterator:
        """Yield `it`: time inside next() is recv wait, the caller's loop body is ingest (a `break` still counts)."""
        it = iter(it)
        while True:
            t = time.time()
            try:
                x = next(it)
            except StopIteration:
                self.add_recv(time.time() - t)
                return
            self.add_recv(time.time() - t)
            t = time.time()
            try:
                yield x
            finally:
                self.add_ingest(time.time() - t)

    def emit(self, round_num: int, commit_s: float, simulated: bool, vclock_now: Optional[float] = None) -> None:
        now = time.time()
        if telemetry.is_enabled():
            ev, f = build_agg_timing(round_num=round_num, cycle_s=now - (self.t0 or now), recv_wait_s=self.recv_wait_s,
                                     ingest_s=self.ingest_s, commit_s=commit_s, n_updates=self.n,
                                     time_mode="sim" if simulated else "real", vclock_now=vclock_now)
            telemetry.emit(ev, **f)
        self.reset()


def agg_timing(obj) -> AggTiming:
    """The aggregator's AggTiming, created on first use (bare-init test aggregators have none)."""
    t = getattr(obj, "_agg_timing", None)
    if t is None:
        t = obj._agg_timing = AggTiming()
    return t
