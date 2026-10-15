# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""FX-N98: balanced GPU placement (default) spreads expected task rate (1/D) evenly; round_robin reverts."""

import random

from flame.launch.experiment_config import ExecutionConfig
from flame.launch.spawner import balanced_gpu_placement


def _rate_spread(delays, place, gpus):
    load = {g: sum(1 / delays[t] for t, gg in place.items() if gg == g) for g in gpus}
    return max(load.values()) / min(load.values())


def test_balanced_beats_round_robin_and_keeps_counts_even():
    rng = random.Random(0)
    delays = {t: rng.choice([10, 15, 20, 25, 35, 50, 65, 90, 235]) for t in range(1, 101)}
    gpus = [0, 2, 3, 4]
    place = balanced_gpu_placement(delays, gpus)
    rr = {t: gpus[(t - 1) % len(gpus)] for t in delays}
    assert sorted(place) == sorted(delays)
    counts = [list(place.values()).count(g) for g in gpus]
    assert max(counts) - min(counts) <= 1
    assert _rate_spread(delays, place, gpus) < 1.05 < _rate_spread(delays, rr, gpus)


def test_deterministic_and_default_on():
    delays = {1: 10.0, 2: 10.0, 3: 50.0, 4: 60.0, 5: 20.0}
    assert balanced_gpu_placement(delays, [0, 1]) == balanced_gpu_placement(dict(reversed(delays.items())), [0, 1])
    assert ExecutionConfig().gpu_placement == "balanced"
