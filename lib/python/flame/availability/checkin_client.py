# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Trainer side of the F42-T1 check-in (real mode; sim replays it in flame.availability.checkin).

The device checks in (CHECK_IN notify, json {p}) when it turns AVL_TRAIN, when a task ends while AVL_TRAIN, and at the reconnect
time a rejection handed it if AVL_TRAIN then; otherwise that reconnect is a missed opportunity and it waits for its next flip.
It never reports leaving. p = REFL's on-device forecast for [now + mu, now + 2 mu] (mu from the server's last reply).
"""
import json
import threading
from typing import Callable, Optional


class CheckinClient:
    def __init__(self, send: Callable[[str], None], now: Callable[[], float], is_train: Callable[[], bool],
                 forecast: Optional[Callable[[float, float], Optional[float]]] = None, emit: Optional[Callable] = None):
        self._send, self._now, self._is_train, self._forecast, self._emit = send, now, is_train, forecast, emit
        self.hold: Optional[float] = None  # reconnect time handed out (trace clock)
        self.busy = False
        self.mu: Optional[float] = None
        self.sent = self.missed = 0
        self._lock = threading.RLock()

    def _checkin(self) -> None:
        t = self._now()
        p = self._forecast(t, self.mu) if self._forecast is not None and self.mu else None
        self.hold = None
        self.sent += 1
        self._send(json.dumps({"p": p}))
        if self._emit:
            self._emit("sent", t, {"p": p})

    def start(self) -> None:
        with self._lock:
            if self._is_train() and not self.busy:
                self._checkin()

    def on_flip(self, train: bool) -> None:
        with self._lock:
            if train and not self.busy and self.hold is None:
                self._checkin()

    def tick(self) -> None:
        """Poller: act on a due reconnect."""
        with self._lock:
            if self.busy or self.hold is None or self._now() < self.hold:
                return
            h = self.hold
            if self._is_train():
                self._checkin()
            else:
                self.hold, self.missed = None, self.missed + 1
                if self._emit:
                    self._emit("missed", h, {})

    def on_reconnect(self, at: float, mu: Optional[float] = None) -> None:
        with self._lock:
            if not self.busy:
                self.hold = float(at)
            self.mu = mu or self.mu

    def on_task_start(self) -> None:
        with self._lock:
            self.busy, self.hold = True, None

    def on_task_end(self) -> None:
        with self._lock:
            self.busy = False
            if self._is_train():
                self._checkin()
