# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""REFL IPS availability predictors (FELIX F42b, ROBUST R27): what REFL* and REFL-fc know about a client's next slots.

Both answer REFLAvailabilityTracker.is_available(client, now, window, slot) for the leg's clock (`now` = vclock / wall since start):
- TraceOracle (REFL*, the fork's simulation): the client's real future trace; labelled oracle.
- SeasonalForecaster (REFL-fc, on-device): the client's own past days only. A slot is predicted available when, on the earlier days
  at the same time of day, the client was AVL_TRAIN for most of that window. No past day yet → predicted available.
- ProphetForecaster (REFL, F42-T3): the paper's per-device Prophet model; answers p(available in [μ, 2μ]) for Alg. 1.
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


class SeasonalForecaster(_TracePredictor):  # DEAD?(F42-T3): no arm since REFL runs Prophet
    """REFL-fc: majority vote of the client's own earlier days at the same time of day (history strictly before `a`)."""

    def _predict(self, ev, a, b):
        shares = [_avl_train_share(ev[0], ev[1], a - k * DAY_S, b - k * DAY_S) for k in range(1, int(a // DAY_S) + 1)]
        return not shares or sum(shares) / len(shares) >= 0.5


class ProphetForecaster(_TracePredictor):
    """REFL (paper §4.1, Alg. 1; F42-T3): a per-client Prophet model on the client's own AVL_TRAIN history, sampled every
    `step_s` with daily seasonality and refit once per trace day; p(a, b) = mean forecast over [a, b], clipped to [0, 1]. The device
    computes it from its own log (real: the trainer; sim: the aggregator replays it), so no history after `now` is ever read."""

    def __init__(self, trace_name: str, registry_path: Optional[str] = None, step_s: float = 300.0):
        super().__init__(trace_name, registry_path)
        self.step_s = float(step_s)
        self._fit: Dict[str, tuple] = {}  # client -> (fit day, grid start, yhat array)
        self.fits = 0

    def _model(self, cid: str, day: int):
        import logging as _lg
        import numpy as np
        import pandas as pd
        for name in ("cmdstanpy", "prophet"):
            _lg.getLogger(name).setLevel(_lg.ERROR)
        from prophet import Prophet
        times, states = self._ev[cid]
        end = day * DAY_S
        ts = np.arange(0.0, end, self.step_s)
        y = [_avl_train_share(times, states, t, t + self.step_s) for t in ts]
        m = Prophet(daily_seasonality=True, weekly_seasonality=False, yearly_seasonality=False)
        m.fit(pd.DataFrame({"ds": pd.to_datetime(ts, unit="s"), "y": y}))
        grid = np.arange(end, end + 2 * DAY_S, self.step_s)
        yhat = m.predict(pd.DataFrame({"ds": pd.to_datetime(grid, unit="s")}))["yhat"].to_numpy()
        self.fits += 1
        return day, end, np.clip(yhat, 0.0, 1.0)

    def p_trace(self, cid: str, now_t: float, a: float, b: float) -> Optional[float]:
        """P(AVL_TRAIN over [a, b]) on the trace clock, from history before the start of now_t's trace day; None = no full day yet."""
        ev = self._ev.get(cid)
        day = int(now_t // DAY_S)
        if ev is None or not ev[0] or day < 1:
            return None
        if self._fit.get(cid, (None,))[0] != day:
            self._fit[cid] = self._model(cid, day)
        _, g0, yhat = self._fit[cid]
        i, j = int((a - g0) // self.step_s), int((b - g0) // self.step_s) + 1
        seg = yhat[max(i, 0):max(min(j, len(yhat)), max(i, 0) + 1)]
        return float(seg.mean()) if len(seg) else None

    def p(self, cid: str, now: float, mu: float) -> Optional[float]:
        """REFL's query on the leg's clock: P(available in [now + mu, now + 2 mu])."""
        t = self._trace_t(now)
        w = mu * self._scale
        return self.p_trace(cid, t, t + w, t + 2 * w)

    def _predict(self, ev, a, b):  # REFLAvailabilityTracker API (unused by Alg. 1)
        raise NotImplementedError("ProphetForecaster answers p(), not is_available()")
