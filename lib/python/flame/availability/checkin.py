# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Check-in with pace steering (Bonawitz et al. 2019; FELIX F42-T1, ROBUST R27): what a deployed FL server knows of availability.

A client checks in when it turns AVL_TRAIN, or at its reconnect time if AVL_TRAIN then. The server pools it while a selection is
open (sync: a window of W = window_frac x the round-time estimate, the selector picks at its close; async: while slots are free),
else rejects it with a reconnect time inside the next selection: sync = last round-start close + the training-phase estimate
+ U(0, W) (overdue: now + U(0, W), F42-T12 d; `overdue="round"` reverts to now + the estimate; within W before any estimate), async = now + m x gap x U(jitter) with m clients on hold, so
held clients return at the dispatch rate. The sync estimate excludes selection waits: a starved window cannot feed back into it.
Not AVL_TRAIN at its reconnect time = missed opportunity; it checks in again at its next flip. A departure is never reported:
the server learns it only by task failure / timeout. One ledger serves both modes: sim derives check-ins from the trace
(`advance`), real feeds the clients' CHECK_IN messages (`on_checkin`).
FX-N151 `replay="time"`: sim replays check-ins in time order (ties by end) and draws each client's jitter from its own stream,
so a run does not depend on end order or on how often `advance` runs; `"end"` = the old per-end replay on one shared stream.
"""
import heapq
import math
import random
from typing import Callable, Dict, Iterable, Optional, Tuple

TrainAt = Callable[[str, float], float]  # end, t -> first time >= t the client is AVL_TRAIN (inf = never)
IsTrain = Callable[[str, float], bool]

CHECKIN_MODES = ("sync", "async")


class CheckinLedger:
    def __init__(self, sync: bool, initial_gap_s: float, window_frac: float = 0.1, jitter: Tuple[float, float] = (0.5, 1.5),
                 alpha: float = 0.25, seed: int = 0, on_event: Optional[Callable] = None, overdue: str = "window",
                 replay: str = "time"):
        if initial_gap_s <= 0:
            raise ValueError(f"initial_gap_s={initial_gap_s}: the server's prior round / dispatch gap must be > 0")
        self.sync = sync
        self.window_frac, self.jitter, self.alpha = float(window_frac), tuple(jitter), float(alpha)
        self.overdue = overdue  # sync: a late round's early arrival polls within a window ("window") or waits a phase ("round")
        self.gap = float(initial_gap_s)  # EWMA (REFL α on history): sync round duration (round starts only), async dispatch gap
        if replay not in ("time", "end"):
            raise ValueError(f"replay={replay!r}: 'time' or 'end'")
        self.replay, self._seed, self._rngs = replay, seed, {}
        self._rng = random.Random(seed)
        self._last_sel: Optional[float] = None  # sync: last round start; async: last dispatch
        self._round = None
        self.train_d: Optional[float] = None  # sync: EWMA of round-start close -> next round start (the training phase)
        self._last_close: Optional[float] = None
        self._round_window = False
        self.pool: Dict[str, float] = {}  # checked in, waiting for a pick: end -> check-in time
        self.hold: Dict[str, float] = {}  # end -> reconnect time handed out
        self.free_since: Dict[str, float] = {}  # end -> earliest time it may check in (leg start, task end)
        self.window: Optional[Tuple[float, float]] = None  # sync: the open selection window
        self.open_since: Optional[float] = None if sync else 0.0  # async: slots free since (all free at the start)
        self.stats = dict(checkins=0, pooled=0, rejects=0, missed=0, picked=0)
        self.on_event = on_event  # (kind, end, t, fields) for telemetry

    # -- clock-free rules -------------------------------------------------
    def window_s(self) -> float:
        return self.window_frac * (self.train_d if self.train_d is not None else self.gap) if self.sync else 0.0

    def is_open(self, t: float) -> bool:
        if self.sync:  # open from its start until the selection closes it (a starved window keeps collecting)
            return self.window is not None and t >= self.window[0]
        return self.open_since is not None and t >= self.open_since

    def _uniform(self, end: Optional[str], a: float, b: float) -> float:
        if self.replay == "end" or end is None:
            return self._rng.uniform(a, b)
        rng = self._rngs.get(end)
        if rng is None:
            rng = self._rngs[end] = random.Random(f"{self._seed}:{end}")
        return rng.uniform(a, b)

    def reconnect_at(self, t: float, end: Optional[str] = None) -> float:
        if self.sync:
            if self.train_d is None or self._last_close is None:
                return t + self._uniform(end, 0.0, self.window_s())  # no estimate yet: come back within a window
            nxt = self._last_close + self.train_d
            late = self.train_d if self.overdue == "round" else 0.0
            return (nxt if nxt > t else t + late) + self._uniform(end, 0.0, self.window_s())
        return t + max(1, len(self.hold) + 1) * self.gap * self._uniform(end, *self.jitter)

    def _emit(self, kind: str, end: str, t: float, **f) -> None:
        if self.on_event is not None:
            self.on_event(kind, end, t, f)

    # -- one check-in (both modes) ----------------------------------------
    def on_checkin(self, end: str, t: float, p: Optional[float] = None) -> Optional[float]:
        """A client checked in at t: pooled (None) or rejected (its reconnect time)."""
        self.stats["checkins"] += 1
        self.free_since[end] = max(self.free_since.get(end, 0.0), t)
        self.hold.pop(end, None)
        if self.is_open(t):
            self.pool[end] = t
            self.stats["pooled"] += 1
            self._emit("pooled", end, t, p=p)
            return None
        return self._reject(end, t)

    def _reject(self, end: str, t: float) -> float:
        r = self.reconnect_at(t, end)
        self.hold[end] = r
        self.stats["rejects"] += 1
        self._emit("rejected", end, t, reconnect_at=r)
        return r

    def on_missed(self, end: str, t: float) -> None:
        self.stats["missed"] += 1
        self._emit("missed", end, t)

    # -- sim: derive check-ins from the trace -----------------------------
    def _next_checkin(self, end: str, train_at: TrainAt, is_train: IsTrain, now: float, commit: bool) -> Optional[float]:
        """This end's next check-in time; with commit, records a missed reconnect it passes on the way (<= now)."""
        base = self.free_since.get(end, 0.0)
        h = self.hold.get(end)
        if h is not None and h >= base:
            if is_train(end, h):
                return h
            if h > now:
                return h  # a wake-up: the missed opportunity is only known (and counted) once the clock reaches it
            if commit:
                del self.hold[end]
                self.free_since[end] = h
                self.on_missed(end, h)
            return train_at(end, h)
        return train_at(end, base)

    def _step(self, end: str, train_at: TrainAt, is_train: IsTrain) -> Tuple[float, bool]:
        """(time, missed) of this end's next step: a check-in, or a reconnect it is not AVL_TRAIN for (_next_checkin's rule)."""
        base = self.free_since.get(end, 0.0)
        h = self.hold.get(end)
        if h is not None and h >= base:
            return h, not is_train(end, h)
        return train_at(end, base), False

    def advance(self, now: float, idle: Iterable[str], train_at: TrainAt, is_train: IsTrain) -> None:
        """Sim: replay every check-in of the idle (not busy, not pooled) clients up to now under the current open state."""
        if self.replay == "time":
            steps = [(t, e, m) for e in idle if e not in self.pool for t, m in [self._step(e, train_at, is_train)] if t <= now]
            heapq.heapify(steps)
            for _ in range(100000 * max(1, len(steps))):
                if not steps:
                    break
                t, end, missed = heapq.heappop(steps)
                if missed:
                    del self.hold[end]
                    self.free_since[end] = t
                    self.on_missed(end, t)
                else:
                    self.on_checkin(end, t)
                    if end in self.pool:
                        continue
                t, missed = self._step(end, train_at, is_train)
                if t <= now:
                    heapq.heappush(steps, (t, end, missed))
            return
        for end in idle:
            for _ in range(100000):
                if end in self.pool:
                    break
                t = self._next_checkin(end, train_at, is_train, now, commit=True)
                if t is None or t > now or math.isinf(t):
                    break
                self.on_checkin(end, t)

    def next_checkin(self, now: float, idle: Iterable[str], train_at: TrainAt, is_train: IsTrain) -> Optional[float]:
        """Sim wake-up: the earliest future check-in (or reconnect) of an idle client."""
        ts = [t for e in idle if e not in self.pool
              for t in [self._next_checkin(e, train_at, is_train, now, commit=False)] if t is not None and t > now]
        return min(ts) if ts else None

    # -- selection boundaries ---------------------------------------------
    def open_window(self, now: float, round_id=None) -> float:
        """Sync: open a selection window at now (no-op if open); returns its close time. A new round_id is a round start: it
        updates the round-time estimate (mid-round top-up windows do not)."""
        if self.window is None:
            self._round_window = round_id is None or round_id != self._round
            if self._round_window:
                if self._last_close is not None and now > self._last_close:
                    d = now - self._last_close
                    self.train_d = d if self.train_d is None else (1 - self.alpha) * d + self.alpha * self.train_d
                    self.gap = self.train_d + self.window_s()  # the round estimate μ (REFL's slot, real `mu`)
                self._last_sel, self._round = now, round_id
            self.window = (now, now + self.window_s())
        return self.window[1]

    def after_select(self, now: float, picked: Iterable[str], slots_left: bool, keep_open: bool = False) -> list:
        """A selection at now picked `picked`; the rest of the pool is rejected. Returns [(end, reconnect_at)] to tell.
        Sync keep_open: the version still misses picks, so the window stays open and the pool keeps waiting."""
        picked = [e for e in picked if e in self.pool]
        for e in picked:
            del self.pool[e]
        self.stats["picked"] += len(picked)
        if not self.sync and picked:
            if self._last_sel is not None and now > self._last_sel:
                self.gap = (1 - self.alpha) * (now - self._last_sel) / len(picked) + self.alpha * self.gap
            self._last_sel = now
        if self.sync and keep_open:
            return []
        if self.sync:
            if self.window is not None and self._round_window:
                self._last_close = now
            self.window = None
        else:
            self.open_since = (self.open_since if self.open_since is not None else now) if slots_left else None
        out = []
        if not slots_left or self.sync:
            for e in (sorted(self.pool) if self.replay == "time" else list(self.pool)):
                del self.pool[e]
                out.append((e, self._reject(e, now)))
        return out

    def release(self, end: str, t: float) -> None:
        """A busy client's task ended (update, failure, timeout) at t: it may check in again from t.
        A check-in already pooled stays: it beat its own update's commit (F42-T9)."""
        self.hold.pop(end, None)
        self.free_since[end] = max(self.free_since.get(end, 0.0), t)
