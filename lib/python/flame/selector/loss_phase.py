# SPDX-License-Identifier: Apache-2.0
"""F46 S4: training phase from the clients' per-sample loss of the model they received.

phi = mean of the last W losses / the highest such mean so far (W = one population sweep of updates). phi ~ 1 while
clients' data is still unlearned; it falls as their shards are fit and rises again when new data arrives (F45: 2.0 -> 0.1
on streams by hour 8, accuracy flat once it passes ~1). Scales the server step (Polyak: step ~ loss left) and / or c.
"""
from collections import deque


class LossPhase:
    def __init__(self, window: int):
        self._win = deque(maxlen=max(1, int(window)))
        self._sum, self._peak = 0.0, None

    def add(self, loss_per_sample: float) -> None:
        if len(self._win) == self._win.maxlen:
            self._sum -= self._win[0]
        self._win.append(float(loss_per_sample))
        self._sum += float(loss_per_sample)
        if len(self._win) == self._win.maxlen:
            m = self._sum / len(self._win)
            self._peak = m if self._peak is None else max(self._peak, m)

    @property
    def phi(self) -> float:
        """1.0 until the first full window; then in (0, 1]."""
        if self._peak is None or self._peak <= 0:
            return 1.0
        return min(1.0, max(0.0, self._sum / len(self._win) / self._peak))
