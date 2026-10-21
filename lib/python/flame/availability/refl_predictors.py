# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""REFL IPS availability predictors (FELIX F42b, ROBUST R27): what REFL* and REFL-fc know about a client's next slots.

Both answer REFLAvailabilityTracker.is_available(client, now, window, slot) for the leg's clock (`now` = vclock / wall since start):
- TraceOracle (REFL*, the fork's simulation): the client's real future trace; labelled oracle.
- SeasonalForecaster (REFL-fc, on-device): the client's own past days only. A slot is predicted available when, on the earlier days
  at the same time of day, the client was AVL_TRAIN for most of that window. No past day yet → predicted available.
"""
import bisect
from typing import Dict, List, Optional

import yaml

from flame.availability.refl_tracker import REFLAvailabilityTracker
from flame.availability.trace import _METADATA_DIR, raw_events, trace_offset_s, trace_time_scale

DAY_S = 86400.0
AVL_TRAIN = "AVL_TRAIN"


def _avl_train_share(times: List[float], states: List[str], a: float, b: float) -> float:
    """Share of [a, b] the client spends in AVL_TRAIN (trace clock; before the first event = AVL_TRAIN)."""
    if b <= a:
        return 0.0
    i = bisect.bisect_right(times, a) - 1
    t, on = a, 0.0
    cur = states[i] if i >= 0 else AVL_TRAIN
    for j in range(i + 1, len(times)):
        if times[j] >= b:
            break
        on += (times[j] - t) * (cur == AVL_TRAIN)
        t, cur = times[j], states[j]
    on += (b - t) * (cur == AVL_TRAIN)
    return on / (b - a)


class _TracePredictor(REFLAvailabilityTracker):
    """Shared plumbing: per-client raw trace on the trace clock; the leg's clock maps to it by offset and scale."""

    def __init__(self, trace_name: str, registry_path: Optional[str] = None):
        super().__init__()
        reg = yaml.safe_load(open(registry_path or _METADATA_DIR / "trainer_registry.yaml"))["trainers"]
        self._ev: Dict[str, tuple] = {}
        for key, meta in reg.items():
            ev = raw_events(trace_name, key) if not trace_name.startswith("syn_0") else []
            self._ev[meta["task_id"]] = ([float(t) for t, _ in ev], [str(s).split(".")[-1] for _, s in ev])
        self._off, self._scale = trace_offset_s(), trace_time_scale()

    def _trace_t(self, now: float) -> float:
        return self._off + now * self._scale

    def is_available(self, client_id: str, cur_time: float, time_window: float, time_slots: int = 1) -> bool:
        ev = self._ev.get(client_id)
        if ev is None or not ev[0]:
            return True
        w = time_window * self._scale
        a = self._trace_t(cur_time) + (time_slots - 1) * w
        return self._predict(ev, a, a + w)

    def _predict(self, ev: tuple, a: float, b: float) -> bool:
        raise NotImplementedError


class TraceOracle(_TracePredictor):
    """REFL*: available iff the real trace keeps the client in AVL_TRAIN for the whole window."""

    def _predict(self, ev, a, b):
        return _avl_train_share(ev[0], ev[1], a, b) >= 1.0


class SeasonalForecaster(_TracePredictor):
    """REFL-fc: majority vote of the client's own earlier days at the same time of day (history strictly before `a`)."""

    def _predict(self, ev, a, b):
        shares = [_avl_train_share(ev[0], ev[1], a - k * DAY_S, b - k * DAY_S) for k in range(1, int(a // DAY_S) + 1)]
        return not shares or sum(shares) / len(shares) >= 0.5
