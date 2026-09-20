# Copyright 2022 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
"""selector abstract class."""

from abc import ABC, abstractmethod
from collections import deque
from typing import Optional, Tuple, Union
import hashlib
import logging
import time

import numpy as np
# Import classes directly: bare `import random` here resolves to the sibling
# flame/selector/random.py submodule, not stdlib.
from random import Random as _StdRandom
from numpy.random import RandomState as _NpRandomState

from .. import telemetry
from ..common.typing import Scalar
from ..end import End
from ..telemetry.events import build_selection
from .properties import (
    PROP_AVL_STATE,
    PROP_CLIENT_TASK_TRAIN_DURATION,
    PROP_LAST_EVAL_ROUND,
    PROP_STAT_UTILITY,
)

SelectorReturnType = dict[str, Union[None, Tuple[str, Scalar]]]

logger = logging.getLogger(__name__)


def _round_or_none(v, ndigits: int = 4):
    """Round for the decision fingerprint; pass through None / non-numerics."""
    try:
        return round(float(v), ndigits)
    except (TypeError, ValueError):
        return None


class AbstractSelector(ABC):
    """Abstract base class for selector implementation."""

    # Fallback seed when none is threaded in: every selector is deterministic
    # across real/sim by default, not a per-process PYTHONHASHSEED lottery.
    DEFAULT_SEED = 1234

    def __init__(self, **kwargs) -> None:
        # Reserved kwarg (consumed, not setattr'd). None -> DEFAULT_SEED.
        _seed = kwargs.pop("_seed", None)
        if _seed is None:
            _seed = self.DEFAULT_SEED
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.selected_ends: set = set()
        self.ordered_updates_recv_ends: list = []
        self._init_selector_stats()
        # Dedicated, seed-able RNGs insulated from the process-global np.random/
        # random. Selectors MUST draw from these (never bare np.random/random) so
        # selection is reproducible across real/sim. Always seeded (DEFAULT_SEED).
        self._seed = _seed
        self._rng = _NpRandomState(_seed)
        self._pyrng = _StdRandom(_seed)
        if _seed is not None:
            logger.info(
                f"[SELECTOR_SEED] {type(self).__name__} dedicated RNGs seeded "
                f"with seed={_seed} fingerprint={self.rng_fingerprint()}"
            )

    # --- sliding-window selection stats -------------------------------------
    # Purely observational: every selector kept a verbatim copy of the init
    # block + compute_trainer_stat_summary + the ingest loop, so a change had
    # to be made in four places. Lives here once; subclasses call
    # record_selection_stats() then maybe_log_stat_summary().
    STAT_WINDOWS = (50, 100, 200)
    STAT_METRICS = ("util", "speed", "round")
    STAT_LOG_EVERY = 5

    def _init_selector_stats(self) -> None:
        self._selector_stats = {
            task: {
                "data": {
                    f"{metric}_last_{window}": deque(maxlen=window)
                    for metric in self.STAT_METRICS
                    for window in self.STAT_WINDOWS
                },
                "summary": {},
            }
            for task in ("train", "eval")
        }
        self._select_run_counter = 0

    @staticmethod
    def _summarize(values) -> dict:
        vals = [v for v in (values or []) if v is not None]
        if not vals:
            return {k: None for k in ("min", "max", "p25", "p50", "p75")}
        arr = np.array(vals, dtype=float)
        return {
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "p25": float(np.percentile(arr, 25)),
            "p50": float(np.percentile(arr, 50)),
            "p75": float(np.percentile(arr, 75)),
        }

    def compute_trainer_stat_summary(self) -> None:
        for task, bucket in self._selector_stats.items():
            for metric, values in bucket["data"].items():
                # `util` keys carry a `stat_` prefix in the summary; kept for
                # log/plot compatibility.
                key = f"stat_{metric}" if "util" in metric else metric
                bucket["summary"][key] = self._summarize(values)

    def _reset_selector_stats(self) -> None:
        self._selector_stats = {}

    def record_selection_stats(
        self, ends: dict[str, End], chosen_ids, task: str = "train"
    ) -> None:
        """Append the chosen ends' utility/speed/last-round into the windows."""
        bucket = self._selector_stats.get(task)
        if bucket is None:
            return
        for end_id in chosen_ids:
            end = ends.get(end_id)
            if end is None:  # in-flight id no longer in the eligible pool
                continue
            speed = end.get_property(PROP_CLIENT_TASK_TRAIN_DURATION)
            for metric, value in (
                ("util", end.get_property(PROP_STAT_UTILITY)),
                (
                    "speed",
                    speed.total_seconds()
                    if hasattr(speed, "total_seconds")
                    else speed,
                ),
                ("round", end.get_property(PROP_LAST_EVAL_ROUND)),
            ):
                if value is None:
                    continue
                for window in self.STAT_WINDOWS:
                    bucket["data"][f"{metric}_last_{window}"].append(value)

    def maybe_log_stat_summary(self) -> None:
        """Recompute + log every STAT_LOG_EVERY selections; no-op otherwise."""
        self._select_run_counter += 1
        if self._select_run_counter % self.STAT_LOG_EVERY:
            return
        self.compute_trainer_stat_summary()
        for task in ("train", "eval"):
            logger.info(
                f"{task.capitalize()} selector stats summary: "
                f"{self._selector_stats[task]['summary']}"
            )
        self._select_run_counter = 0

    def rng_fingerprint(self) -> str:
        """Short hex digest of both dedicated RNGs' internal state.

        For determinism audits: two same-seed runs with differing
        fingerprints at the same call site prove extra draws happened
        between construction and that point.
        """
        py_state = repr(self._pyrng.getstate()).encode()
        np_state = repr(self._rng.get_state()).encode()
        return hashlib.sha256(py_state + np_state).hexdigest()[:12]

    def enforce_min_start(self, ends_count: int) -> bool:
        """Return True if selection should wait due to min-start threshold."""
        threshold = (
            int(self.minInitialTrainers)
            if hasattr(self, "minInitialTrainers")
            and self.minInitialTrainers is not None
            else -1
        )
        if ends_count < threshold:
            # LOUD, not silent, and rate-limited to once a minute. At 100 of 100
            # the quorum has zero tolerance: one dead trainer and this branch is
            # taken forever, which on 2026-08-21 returned `ends: []` on 12,937
            # consecutive distribute cycles while the run sat at 0 commits for
            # its whole grace, with only a DEBUG line to say why.
            now = time.time()
            self._min_start_waits = getattr(self, "_min_start_waits", 0) + 1
            if now - getattr(self, "_min_start_last_warn", 0.0) > 60.0:
                self._min_start_last_warn = now
                logger.warning(
                    f"selection BLOCKED on the join barrier: {ends_count} of "
                    f"{threshold} trainers registered "
                    f"({self._min_start_waits} waits so far). Trainers that "
                    f"died at startup never register -- check the trainers log."
                )
            logger.debug(
                f"Not enough ends to start selection, need at least {threshold}"
            )
            time.sleep(0.1)
            return True
        self._min_start_waits = 0
        return False

    @abstractmethod
    def select(
        self, ends: dict[str, End], channel_props: dict[str, Scalar]
    ) -> SelectorReturnType:
        """Abstract method to select ends.

        Parameters
        ----------
        ends: a dictionary whose key is end id and value is End object
        channel_props: properties set in channel

        Returns
        -------
        dictionary: key is end id and value is a property (as tuple)
                    used/created during selection process; value can be none
        """

    def emit_selection(
        self,
        round_num: int,
        task: str,
        ends: dict[str, End],
        eligible_ids,
        chosen_ids,
        per_trainer_extra: Optional[dict] = None,
        extra: Optional[dict] = None,
    ) -> None:
        """Emit a structured selector-decision event (no-op if telemetry off).

        Centralized here so every selector produces an identical schema, which
        is what makes cross-selector comparison possible. ``ends`` is the full
        candidate pool; availability composition and per-trainer utility/speed
        are derived from end properties.
        """
        if not telemetry.is_enabled():
            return
        try:
            chosen_set = set(chosen_ids)
            avail_composition: dict[str, int] = {}
            per_trainer: dict[str, dict] = {}
            for end_id, end in ends.items():
                state = end.get_property(PROP_AVL_STATE)
                state_name = getattr(state, "value", None) or (
                    str(state) if state is not None else "UNKNOWN"
                )
                avail_composition[state_name] = (
                    avail_composition.get(state_name, 0) + 1
                )
                util = end.get_property(PROP_STAT_UTILITY)
                speed = end.get_property(PROP_CLIENT_TASK_TRAIN_DURATION)
                entry = {
                    "utility": util,
                    "speed_s": speed.total_seconds()
                    if hasattr(speed, "total_seconds")
                    else speed,
                    "selected": end_id in chosen_set,
                    "avl_state": state_name,
                }
                if per_trainer_extra and end_id in per_trainer_extra:
                    entry.update(per_trainer_extra[end_id])
                per_trainer[end_id] = entry

            # in-flight count: selected_ends is a set/list for most selectors,
            # but a {requester: set(ends)} dict for fedbuff-style selectors.
            sel = self.selected_ends
            if isinstance(sel, dict):
                vals = list(sel.values())
                in_flight = (
                    sum(len(v) for v in vals)
                    if vals and all(isinstance(v, (set, list)) for v in vals)
                    else len(sel)
                )
            elif isinstance(sel, (set, list)):
                in_flight = len(sel)
            else:
                in_flight = 0

            # Determinism fingerprints: eligible = candidate set; decision = set +
            # per-candidate utility/speed + k. Same fingerprint but different
            # `chosen` => RNG desync; different fingerprint => input drift.
            elig = sorted(set(eligible_ids))
            elig_fp = hashlib.sha1(
                "|".join(elig).encode()
            ).hexdigest()[:12]
            dec_payload = ";".join(
                f"{e}:{_round_or_none(per_trainer.get(e, {}).get('utility'))}"
                f":{_round_or_none(per_trainer.get(e, {}).get('speed_s'))}"
                for e in elig
            ) + f"#k={len(chosen_set)}"
            dec_fp = hashlib.sha1(dec_payload.encode()).hexdigest()[:12]
            extra = dict(extra or {})
            extra.update({
                "seed": self._seed,
                "eligible_fingerprint": elig_fp,
                "decision_fingerprint": dec_fp,
            })

            ev, fields = build_selection(
                round_num=int(round_num),
                task=task,
                selector=type(self).__name__,
                num_candidates=len(ends),
                num_eligible=len(set(eligible_ids)),
                avail_composition=avail_composition,
                chosen=list(chosen_set),
                in_flight=in_flight,
                per_trainer=per_trainer,
                extra=extra,
            )
            telemetry.emit(ev, **fields)
        except Exception as e:  # telemetry must never break selection
            logger.debug(f"emit_selection failed: {e}")

    def on_update_received(
        self, end_id: str, msg: dict, round_num: int
    ) -> None:
        """Hook: aggregator calls this when a trainer update arrives.

        Default records the end_id for later cleanup. Subclasses override to
        extract per-update metrics (e.g. FedDance pulls LOCAL_ACCURACY).
        """
        if isinstance(self.selected_ends, set):
            self.ordered_updates_recv_ends.append(end_id)

    def on_round_completed(
        self, ends: dict[str, End], round_num: int
    ) -> None:
        """Hook: aggregator calls this after aggregation finishes.

        Default frees received-ends from the in-flight set. Subclasses with
        custom legacy cleanup (_cleanup_recvd_ends) get that called too.
        """
        if isinstance(self.selected_ends, set):
            for end_id in self.ordered_updates_recv_ends:
                self.selected_ends.discard(end_id)
            self.ordered_updates_recv_ends = []
        elif hasattr(self, "_cleanup_recvd_ends"):
            self._cleanup_recvd_ends(ends)
