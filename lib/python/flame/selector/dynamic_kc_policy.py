# Copyright 2023 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
#
# SPDX-License-Identifier: Apache-2.0
"""Dynamic K and C policies for FwdLLM federated learning."""

import logging
from abc import ABC, abstractmethod
from collections import deque
from typing import Optional

logger = logging.getLogger(__name__)


class DynamicKCPolicy(ABC):
    """Abstract base for dynamic K (aggregation goal) and C (concurrency) policies.

    Each method returns a new integer value, or None to leave the current value unchanged.
    All ``metrics`` keys are optional; implementations should use .get() with a None check.

    Common metrics keys:
        var_pass_rate, var_last, avg_staleness, p75_staleness, model_version,
        n_aggs_completed, var_threshold, n_eligible_train, n_eligible_eval
    """

    @abstractmethod
    def compute_new_k(self, current_k: int, metrics: dict) -> Optional[int]:
        """Return a new K value, or None to keep the current value."""

    @abstractmethod
    def compute_new_c(self, current_c: int, metrics: dict) -> Optional[int]:
        """Return a new C value, or None to keep the current value."""

    @abstractmethod
    def name(self) -> str:
        """Human-readable name used in log messages."""


class NoOpPolicy(DynamicKCPolicy):
    """Returns None for both K and C — equivalent to static K/C (baseline/testing)."""

    def compute_new_k(self, current_k: int, metrics: dict) -> Optional[int]:
        return None

    def compute_new_c(self, current_c: int, metrics: dict) -> Optional[int]:
        return None

    def name(self) -> str:
        return "noop"


class VarianceBasedPolicy(DynamicKCPolicy):
    """Adjusts K based on a rolling variance pass-rate window.

    pass_rate > high_threshold → decrease K (enough diversity, speed up).
    pass_rate < low_threshold  → increase K (too noisy, collect more grads).
    """

    def __init__(
        self,
        high_threshold: float = 0.8,
        low_threshold: float = 0.3,
        k_step: int = 2,
        window: int = 10,
    ):
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.k_step = k_step
        self._history: deque = deque(maxlen=window)

    def compute_new_k(self, current_k: int, metrics: dict) -> Optional[int]:
        rate = metrics.get("var_pass_rate")
        if rate is None:
            return None
        self._history.append(rate)
        if len(self._history) < self._history.maxlen:
            return None  # wait for a full window before acting
        avg = sum(self._history) / len(self._history)
        if avg > self.high_threshold:
            logger.debug(
                f"[VarianceBasedPolicy] avg_pass_rate={avg:.3f} > {self.high_threshold}: "
                f"decreasing K by {self.k_step}"
            )
            return current_k - self.k_step
        if avg < self.low_threshold:
            logger.debug(
                f"[VarianceBasedPolicy] avg_pass_rate={avg:.3f} < {self.low_threshold}: "
                f"increasing K by {self.k_step}"
            )
            return current_k + self.k_step
        return None

    def compute_new_c(self, current_c: int, metrics: dict) -> Optional[int]:
        return None

    def name(self) -> str:
        return "variance_based"


class StalenessBasedPolicy(DynamicKCPolicy):
    """Adjusts C based on rolling average staleness of received updates.

    avg_staleness > stale_threshold → decrease C (too many concurrent trainers).
    avg_staleness < fresh_threshold → increase C (room for more parallelism).
    """

    def __init__(
        self,
        stale_threshold: float = 3.0,
        fresh_threshold: float = 1.0,
        c_step: int = 5,
        window: int = 10,
    ):
        self.stale_threshold = stale_threshold
        self.fresh_threshold = fresh_threshold
        self.c_step = c_step
        self._history: deque = deque(maxlen=window)

    def compute_new_k(self, current_k: int, metrics: dict) -> Optional[int]:
        return None

    def compute_new_c(self, current_c: int, metrics: dict) -> Optional[int]:
        staleness = metrics.get("avg_staleness")
        if staleness is None:
            return None
        self._history.append(staleness)
        if len(self._history) < self._history.maxlen:
            return None
        avg = sum(self._history) / len(self._history)
        if avg > self.stale_threshold:
            logger.debug(
                f"[StalenessBasedPolicy] avg_staleness={avg:.3f} > {self.stale_threshold}: "
                f"decreasing C by {self.c_step}"
            )
            return current_c - self.c_step
        if avg < self.fresh_threshold:
            logger.debug(
                f"[StalenessBasedPolicy] avg_staleness={avg:.3f} < {self.fresh_threshold}: "
                f"increasing C by {self.c_step}"
            )
            return current_c + self.c_step
        return None

    def name(self) -> str:
        return "staleness_based"


class EligibleEndsBasedPolicy(DynamicKCPolicy):
    """Right-sizes C to track the actual eligible training pool.

    n_eligible > C * headroom_factor    → increase C (untapped parallelism).
    n_eligible < C * undercommit_factor → snap C down to pool size.
    """

    def __init__(
        self,
        headroom_factor: float = 1.5,
        undercommit_factor: float = 0.8,
        c_step: int = 5,
        window: int = 5,
    ):
        self.headroom_factor = headroom_factor
        self.undercommit_factor = undercommit_factor
        self.c_step = c_step
        self._history: deque = deque(maxlen=window)

    def compute_new_k(self, current_k: int, metrics: dict) -> Optional[int]:
        return None

    def compute_new_c(self, current_c: int, metrics: dict) -> Optional[int]:
        n_eligible = metrics.get("n_eligible_train")
        if n_eligible is None:
            return None
        self._history.append(n_eligible)
        if len(self._history) < self._history.maxlen:
            return None
        avg_eligible = sum(self._history) / len(self._history)
        if avg_eligible > current_c * self.headroom_factor:
            logger.debug(
                f"[EligibleEndsBasedPolicy] avg_eligible={avg_eligible:.1f} > "
                f"C*headroom={current_c * self.headroom_factor:.1f}: increasing C by {self.c_step}"
            )
            return current_c + self.c_step
        if avg_eligible < current_c * self.undercommit_factor:
            new_c = max(1, int(avg_eligible))
            logger.debug(
                f"[EligibleEndsBasedPolicy] avg_eligible={avg_eligible:.1f} < "
                f"C*undercommit={current_c * self.undercommit_factor:.1f}: "
                f"snapping C down to {new_c}"
            )
            return new_c
        return None

    def name(self) -> str:
        return "eligible_ends_based"


class StepSchedulePolicy(DynamicKCPolicy):
    """Deterministic K decay: decreases K by k_step every n_aggs_per_step aggregations."""

    def __init__(
        self,
        k_step: int = 2,
        n_aggs_per_step: int = 50,
        k_floor: int = 5,
    ):
        self.k_step = k_step
        self.n_aggs_per_step = n_aggs_per_step
        self.k_floor = k_floor

    def compute_new_k(self, current_k: int, metrics: dict) -> Optional[int]:
        n = metrics.get("n_aggs_completed", 0)
        if n > 0 and n % self.n_aggs_per_step == 0:
            new_k = max(self.k_floor, current_k - self.k_step)
            return new_k if new_k != current_k else None
        return None

    def compute_new_c(self, current_c: int, metrics: dict) -> Optional[int]:
        return None

    def name(self) -> str:
        return "step_schedule"


class AdaptiveKVarTrackingPolicy(DynamicKCPolicy):
    """Adapts K based on variance/threshold ratio and progress within a data_id.

    ratio    = var_last / var_threshold   (>1 = still failing)
    progress = (iter + 1) / target        (fraction of iteration budget used)
    trend    = mean(recent half) - mean(earlier half)   (<0 = improving)

    progress < 0.33          → no change
    0.33 <= progress < 0.75  → ratio > 1.5 and trend >= 0 → K += k_step
                               ratio < 0.5                 → K -= k_step
    progress >= 0.75         → ratio > 1.0                 → K += 2 * k_step
                               ratio < 0.4                 → K -= k_step

    The ratio window resets on each new data_id.
    """

    def __init__(
        self,
        target_iter_per_data_id: int = 15,
        k_step: int = 2,
        window: int = 6,
    ):
        self.target_iter_per_data_id = target_iter_per_data_id
        self.k_step = k_step
        self._ratio_history: deque = deque(maxlen=window)
        self._last_data_id: Optional[int] = None

    def _maybe_reset_for_new_data_id(self, data_id: Optional[int]) -> None:
        if data_id is None:
            return
        if self._last_data_id is None or data_id != self._last_data_id:
            self._ratio_history.clear()
            self._last_data_id = data_id

    def _compute_trend(self) -> float:
        """Return (mean of recent half) − (mean of earlier half). <0 = improving."""
        n = len(self._ratio_history)
        if n < 4:
            return 0.0
        half = n // 2
        values = list(self._ratio_history)
        earlier = values[:half]
        recent = values[half:]
        return (sum(recent) / len(recent)) - (sum(earlier) / len(earlier))

    def compute_new_k(self, current_k: int, metrics: dict) -> Optional[int]:
        var_last = metrics.get("var_last")
        var_threshold = metrics.get("var_threshold")
        iter_in_data_id = metrics.get("iteration_per_data_id")
        target = metrics.get("target_iter_per_data_id") or self.target_iter_per_data_id
        data_id = metrics.get("data_id")

        if var_last is None or var_threshold in (None, 0) or iter_in_data_id is None:
            return None
        if not target or target <= 0:
            return None

        self._maybe_reset_for_new_data_id(data_id)

        ratio = float(var_last) / float(var_threshold)
        self._ratio_history.append(ratio)
        progress = (int(iter_in_data_id) + 1) / float(target)
        trend = self._compute_trend()

        decision = None
        if 0.33 <= progress < 0.75:
            if ratio > 1.5 and trend >= 0:
                decision = current_k + self.k_step
            elif ratio < 0.5:
                decision = current_k - self.k_step
        elif progress >= 0.75:
            if ratio > 1.0:
                decision = current_k + 2 * self.k_step
            elif ratio < 0.4:
                decision = current_k - self.k_step

        if decision is not None and decision != current_k:
            logger.info(
                f"[AdaptiveK] data_id={data_id} iter={iter_in_data_id} "
                f"progress={progress:.2f} ratio={ratio:.3f} trend={trend:+.3f}: "
                f"K {current_k} → {decision}"
            )
        return decision

    def compute_new_c(self, current_c: int, metrics: dict) -> Optional[int]:
        return None

    def name(self) -> str:
        return "adaptive_k_var_tracking"


class CompositePolicy(DynamicKCPolicy):
    """Chains multiple policies; first non-None result wins for K and C independently."""

    def __init__(self, sub_policies: list):
        """sub_policies: list of dicts with "name" and optional "kwargs" keys."""
        self.policies = [
            build_policy(p["name"], p.get("kwargs", {})) for p in sub_policies
        ]

    def compute_new_k(self, current_k: int, metrics: dict) -> Optional[int]:
        for p in self.policies:
            result = p.compute_new_k(current_k, metrics)
            if result is not None:
                return result
        return None

    def compute_new_c(self, current_c: int, metrics: dict) -> Optional[int]:
        for p in self.policies:
            result = p.compute_new_c(current_c, metrics)
            if result is not None:
                return result
        return None

    def name(self) -> str:
        return f"composite[{','.join(p.name() for p in self.policies)}]"


def _interp(points: list, x: float) -> float:
    """Piecewise-linear [[x, y], ...] at x, held flat past either end."""
    pts = sorted(points)
    if x <= pts[0][0]:
        return pts[0][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return pts[-1][1]


class SchedulePolicy(DynamicKCPolicy):
    """FX-N109 I5 probe: K / c as piecewise-linear functions of aggregated updates (`n_updates`); tests phase dependence."""

    def __init__(self, k_points: Optional[list] = None, c_points: Optional[list] = None):
        self.k_points, self.c_points = k_points, c_points

    def compute_new_k(self, current_k: int, metrics: dict) -> Optional[int]:
        return round(_interp(self.k_points, metrics.get("n_updates", 0))) if self.k_points else None

    def compute_new_c(self, current_c: int, metrics: dict) -> Optional[int]:
        return round(_interp(self.c_points, metrics.get("n_updates", 0))) if self.c_points else None

    def name(self) -> str:
        return "schedule"


class AimdKPolicy(DynamicKCPolicy):
    """FX-N109 I5: find the step-size edge online, TCP-style (AIMD on K; the step per update is η w / K).

    Per window of one population sweep (n updates): if the clients' per-sample loss (I9: of the model they received) did not
    rise by more than 2 standard errors vs the previous window, K -= 1 (larger step); if it rose, K *= 2 (back off).
    memory (FX-N121): after a rise at K_e, K never steps below 2 K_e again, so it stops re-probing a known unstable edge.
    stall (Q-AIMD): paired per client (each client's loss vs its own previous one, so client hardness cancels): a significant
    fall steps K down, a rise doubles it, no measurable progress raises it by 1."""

    def __init__(self, memory: bool = False, stall: bool = False):
        self._win: list = []
        self._prev: Optional[tuple] = None  # (mean, standard error) of the previous window
        self.memory = str(memory).lower() == "true"
        self.stall = str(stall).lower() == "true"
        self.floor = 0
        self._last: dict = {}  # stall: client -> its last loss

    def _paired_k(self, current_k: int, metrics: dict) -> Optional[int]:
        for end, q in metrics.get("cycle_client_losses") or []:
            if end in self._last:
                self._win.append(q - self._last[end])
            self._last[end] = q
        if len(self._win) < max(4, (metrics.get("n_trainers") or 0) // 2):
            return None
        n = len(self._win)
        m = sum(self._win) / n
        se = (sum((x - m) ** 2 for x in self._win) / (n - 1) / n) ** 0.5
        self._win = []
        if m > 2 * se:
            if self.memory:
                self.floor = max(self.floor, 2 * current_k)
            return current_k * 2
        if m < -2 * se:
            return max(current_k - 1, self.floor)
        return current_k + 1

    def compute_new_k(self, current_k: int, metrics: dict) -> Optional[int]:
        if self.stall:
            return self._paired_k(current_k, metrics)
        self._win += list(metrics.get("cycle_losses") or [])
        if len(self._win) < max(2, metrics.get("n_trainers") or 0):
            return None
        n = len(self._win)
        m = sum(self._win) / n
        se = (sum((x - m) ** 2 for x in self._win) / (n - 1) / n) ** 0.5
        prev, self._prev, self._win = self._prev, (m, se), []
        if prev is None:
            return None
        rose = m - prev[0] > 2 * (se ** 2 + prev[1] ** 2) ** 0.5
        if rose and self.memory:
            self.floor = max(self.floor, 2 * current_k)
        return current_k * 2 if rose else max(current_k - 1, self.floor)

    def compute_new_c(self, current_c: int, metrics: dict) -> Optional[int]:
        return None

    def name(self) -> str:
        return "aimd_k"


class AvailabilityCapPolicy(DynamicKCPolicy):
    """FX-N109 I5: c = min(configured c, rho x train-eligible pool), so the selector keeps a choice when the pool shrinks."""

    def __init__(self, rho: float = 0.5):
        self.rho = rho

    def compute_new_k(self, current_k: int, metrics: dict) -> Optional[int]:
        return None

    def compute_new_c(self, current_c: int, metrics: dict) -> Optional[int]:
        n, c0 = metrics.get("n_eligible_train"), metrics.get("c_config")
        if n is None or c0 is None:
            return None
        return max(1, min(int(c0), int(self.rho * n)))

    def name(self) -> str:
        return "availability_cap"


class PaceCPolicy(DynamicKCPolicy):
    """F42d (Ledger C8): pace concurrency c by federated validation (Felix's eval probes on held-out client data).

    Probes `(t, nll_sum, correct, n)` are pooled per window of `window_s` (extended until `min_n` samples); the signal is the
    mean of the last two windows. Loss above (1 + rho) x its running min with accuracy no better than its best + delta
    → c x beta (floor c_floor x c_config); loss back at its min, or accuracy past best + delta → c x 2 (cap c_config)."""

    def __init__(self, window_s: float = 3600.0, min_n: int = 200, rho: float = 0.1, beta: float = 0.75,
                 c_floor: float = 0.25, delta: float = 0.01):
        self.window_s, self.min_n, self.rho = float(window_s), int(min_n), float(rho)
        self.beta, self.c_floor, self.delta = float(beta), float(c_floor), float(delta)
        self._t0, self._acc = None, [0.0, 0, 0]
        self._sig: list = []
        self.min_loss, self.best_acc, self.last = float("inf"), -1.0, None

    def compute_new_k(self, current_k: int, metrics: dict) -> Optional[int]:
        return None

    def compute_new_c(self, current_c: int, metrics: dict) -> Optional[int]:
        now = metrics.get("now")
        for _, nll, correct, n in metrics.get("val_probes") or []:
            self._acc[0] += nll; self._acc[1] += correct; self._acc[2] += n
        if now is None:
            return None
        if self._t0 is None:
            self._t0 = now
        if now - self._t0 < self.window_s or self._acc[2] < self.min_n:
            return None
        self._sig.append((self._acc[0] / self._acc[2], self._acc[1] / self._acc[2]))
        self._t0, self._acc = now, [0.0, 0, 0]
        loss = sum(x for x, _ in self._sig[-2:]) / len(self._sig[-2:])
        acc = sum(y for _, y in self._sig[-2:]) / len(self._sig[-2:])
        c_cfg = int(metrics.get("c_config") or current_c)
        self.last = dict(loss=loss, acc=acc, min_loss=self.min_loss, best_acc=self.best_acc)
        if loss <= self.min_loss or acc > self.best_acc + self.delta:
            new = min(c_cfg, 2 * current_c)
        elif loss > (1 + self.rho) * self.min_loss:
            new = max(max(1, round(self.c_floor * c_cfg)), round(self.beta * current_c))
        else:
            new = current_c
        self.min_loss, self.best_acc = min(self.min_loss, loss), max(self.best_acc, acc)
        return new

    def name(self) -> str:
        return "pace_c"


_POLICY_REGISTRY: dict = {
    "noop": NoOpPolicy,
    "variance_based": VarianceBasedPolicy,
    "staleness_based": StalenessBasedPolicy,
    "eligible_ends_based": EligibleEndsBasedPolicy,
    "step_schedule": StepSchedulePolicy,
    "adaptive_k_var_tracking": AdaptiveKVarTrackingPolicy,
    "composite": CompositePolicy,
    "schedule": SchedulePolicy,
    "aimd_k": AimdKPolicy,
    "availability_cap": AvailabilityCapPolicy,
    "pace_c": PaceCPolicy,
}


def build_policy(policy_name: str, policy_kwargs: dict) -> DynamicKCPolicy:
    """Instantiate a DynamicKCPolicy by name; raises ValueError if unknown."""
    cls = _POLICY_REGISTRY.get(policy_name)
    if cls is None:
        raise ValueError(
            f"Unknown dynamic_kc policy: '{policy_name}'. "
            f"Valid options: {sorted(_POLICY_REGISTRY)}"
        )
    return cls(**policy_kwargs)
