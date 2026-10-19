# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Payload transfer time over a time-varying satellite link.

`estimate_transfer` walks a per-timestep rate (1 s per timestep, the same
resolution `LinkRuntime.result_for` indexes by): it waits while the rate is
0 (no path), transmits at each timestep's rate, pauses if the path drops
mid-transfer and resumes when it returns, then adds one-way propagation
delay at the moment transmission starts. The rate is a callable so costly
paths (ISL relays) are only evaluated for timesteps a transfer touches.

total_s = wait_s + transmit_s + propagation_s

Time past the end of the series wraps around (modulo its length), so runs
longer than the trajectory file keep seeing passes instead of a frozen
final position.
"""

import math
from dataclasses import dataclass, asdict
from typing import Callable, Optional


@dataclass
class TransferEstimate:
    direction: str
    routing: str
    nbytes: int
    start_s: float
    wait_s: float  # time with no path: before the first bit plus mid-transfer stalls
    transmit_s: float  # time actually sending bits
    propagation_s: float  # one-way path delay at transmission start
    total_s: float
    throughput_mbps: float  # effective rate over transmit_s
    hops: int  # 1 = direct ground station <-> satellite
    complete: bool  # False if the series never offers a path

    def as_telemetry(self, prefix: str) -> dict:
        return {f"{prefix}_{k}": v for k, v in asdict(self).items()}


def estimate_transfer(
    rate_bps_at: Callable[[int], float],
    n: int,
    start_s: float,
    nbytes: int,
    propagation_s_at: Callable[[int], float],
    hops_at: Callable[[int], int] = lambda k: 1,
    direction: str = "downlink",
    routing: str = "direct",
) -> TransferEstimate:
    """Transfer `nbytes` starting at `start_s` over `rate_bps_at(k)` (bits/s
    usable during timestep k of an `n`-step periodic series; 0 = no path).
    `propagation_s_at(k)` and `hops_at(k)` describe the path at the
    timestep transmission starts. Incomplete if a full cycle has no path."""
    start_s = max(float(start_s), 0.0)
    nbytes = max(int(nbytes), 0)
    unreachable = TransferEstimate(
        direction, routing, nbytes, start_s, math.inf, 0.0, 0.0, math.inf, 0.0, 0, False
    )
    if n <= 0:
        return unreachable

    bits_left = nbytes * 8.0
    t = start_s
    wait_s = 0.0
    transmit_s = 0.0
    idle_steps = 0  # consecutive steps without a path
    first_k: Optional[int] = None
    while True:
        k = int(math.floor(t)) % n
        dt = math.floor(t) + 1.0 - t
        r = float(rate_bps_at(k))
        if r <= 0:
            idle_steps += 1
            if idle_steps > n:
                return unreachable
            wait_s += dt
        else:
            idle_steps = 0
            if first_k is None:
                first_k = k
            need = bits_left / r
            if need <= dt:
                transmit_s += need
                break
            bits_left -= r * dt
            transmit_s += dt
        t += dt

    propagation_s = float(propagation_s_at(first_k))
    throughput_mbps = (nbytes * 8.0 / transmit_s / 1e6) if transmit_s > 0 else 0.0
    return TransferEstimate(
        direction=direction,
        routing=routing,
        nbytes=nbytes,
        start_s=start_s,
        wait_s=wait_s,
        transmit_s=transmit_s,
        propagation_s=propagation_s,
        total_s=wait_s + transmit_s + propagation_s,
        throughput_mbps=throughput_mbps,
        hops=int(hops_at(first_k)),
        complete=True,
    )
