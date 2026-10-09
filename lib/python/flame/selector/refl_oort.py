# Copyright 2023 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
"""REFL-enhanced OortSelector with priority-based selection and pacer."""

import logging
import random
from typing import Dict, List, Set, Optional
from collections import deque
import numpy as np

from flame.common.typing import Scalar
from flame.common.util import MLFramework, get_ml_framework_in_use
from flame.end import End
from flame.selector.oort import OortSelector
from flame.selector.properties import (
    PROP_DATASET_SIZE,
    PROP_END_ID,
    PROP_SELECTED_COUNT,
    PROP_STAT_UTILITY,
    PROP_UTILITY,
)
from flame.availability.refl_tracker import REFLAvailabilityTracker

logger = logging.getLogger(__name__)


class REFLOortSelector(OortSelector):
    """
    REFL-enhanced Oort selector with availability-aware priority selection.

    Extends base OortSelector with:
    - Priority-based client selection using availability predictions
    - Adaptive pacer mechanism for round threshold adjustment
    - Blacklisting to prevent over-selection of specific clients
    """

    def __init__(self, **kwargs):
        """Initialize REFL Oort selector."""
        super().__init__(**kwargs)

        ml_framework_in_use = get_ml_framework_in_use()
        if ml_framework_in_use != MLFramework.PYTORCH:
            raise NotImplementedError(
                "REFLOortSelector is currently only implemented in PyTorch"
            )

        # REFL-specific parameters
        self.avail_priority = kwargs.get(
            "avail_priority", 0
        )  # 0=none, 1=fill, 2=strict
        self.avail_probability = float(kwargs.get("avail_probability", 1.0))  # Accuracy 0-1
        # FX-N74: REFL fork `resampleClients` picks at random within the feasible set unless sample_mode == "oort"
        self.sample_mode = kwargs.get("sample_mode", "oort")

        # Blacklisting parameters
        self.blacklist_rounds = kwargs.get("blacklist_rounds", -1)  # -1 disables
        self.blacklist_max_len = kwargs.get("blacklist_max_len", 0.3)  # Max 30%

        # Pacer parameters
        self.pacer_step = kwargs.get("pacer_step", 20)
        self.pacer_delta = kwargs.get("pacer_delta", 5)
        self.sample_window = kwargs.get("sample_window", 5.0)  # REFL fork default

        # Availability tracker
        trace_file = kwargs.get("availability_trace_file", None)
        trainer_registry = kwargs.get("trainer_registry_file", None)

        if trace_file:
            self.avail_tracker = REFLAvailabilityTracker(trace_file, trainer_registry)
        else:
            self.avail_tracker = None
            if self.avail_priority > 0:
                logger.warning(
                    "avail_priority > 0 but no availability_trace_file provided. "
                    "Priority selection will be disabled."
                )
                self.avail_priority = 0

        self.exploitation_util_history = deque(maxlen=200)
        self.last_pacer_round = 0
        self.newly_selected_this_round: set = set()

        logger.info(
            f"REFLOortSelector initialized: "
            f"avail_priority={self.avail_priority}, "
            f"blacklist_rounds={self.blacklist_rounds}, "
            f"pacer_step={self.pacer_step}"
        )

    def select(
        self,
        ends: Dict[str, End],
        channel_props: Dict[str, Scalar],
        trainer_unavail_list: List,
        task_to_perform: str,
        **kwargs,
    ):
        """
        Select clients using REFL's priority-based approach for SyncFL.

        Args:
            ends: Dictionary of available ends
            channel_props: Channel properties including round number
            trainer_unavail_list: List of unavailable trainers
            task_to_perform: Task type (train/eval)
            **kwargs: Additional arguments (e.g., round_duration for priority calc)

        Returns:
            Dictionary of selected end_ids
        """
        logger.debug("Calling REFL Oort select")

        num_of_ends = min(len(ends), kwargs.get("num_to_select") or self.num_of_ends)  # FX-N37 top-up
        if num_of_ends == 0:
            logger.debug("ends is empty")
            return {}

        round_num = channel_props.get("round", 0)
        cur_time = channel_props.get("cur_time", 0)
        round_duration_hint = kwargs.get("round_duration", 100.0)

        logger.info(
            f"REFL Oort selecting {num_of_ends} ends for round {round_num}, "
            f"task: {task_to_perform}, avail_priority={self.avail_priority}"
        )

        agg_version_key = kwargs.get("agg_version_key")
        trainer_version_keys = kwargs.get("trainer_version_keys")
        # A dispatch (version keys passed) never re-sends the cache (see OortSelector.select).
        if (trainer_version_keys is None and round_num <= self._last_selection_round
                and len(self.newly_selected_this_round) != 0):
            return {key: None for key in self.newly_selected_this_round}

        if round_num != getattr(self, "_paced_round", None):  # once per round (FX-D10 repeats a round)
            self._paced_round = round_num
            self.pacer(round_num)

        self._draw_key = agg_version_key if agg_version_key is not None else round_num
        unavail_set = set(trainer_unavail_list) if trainer_unavail_list else set()

        eligible_ends = {
            end_id: end
            for end_id, end in ends.items()
            if end_id not in unavail_set and end_id not in self.selected_ends
            and not (trainer_version_keys is not None and agg_version_key is not None
                     and trainer_version_keys.get(end_id) == agg_version_key)  # FX-D9
        }

        if len(eligible_ends) == 0:
            logger.error(
                f"[REFL_SELECT] Round {round_num}: no eligible trainers "
                f"(total={len(ends)}, unavail={len(unavail_set)}, in_flight={len(self.selected_ends)})"
            )
            return {}

        num_to_select = min(num_of_ends, len(eligible_ends))
        if num_to_select < num_of_ends:
            logger.warning(
                f"[REFL_SELECT] Round {round_num}: only {num_to_select}/{num_of_ends} trainers eligible"
            )

        # Build blacklist if enabled
        blacklist = self.get_blacklist(eligible_ends) if self.blacklist_rounds > 0 else set()

        # Build priority lists using availability tracker
        priority_ends, remaining_ends = self.build_priority_lists(
            eligible_ends, cur_time, round_duration_hint, blacklist, trainer_unavail_list or []
        )

        logger.info(
            f"Priority split: {len(priority_ends)} priority, "
            f"{len(remaining_ends)} remaining, {len(blacklist)} blacklisted"
        )

        # Select based on priority mode
        if self.avail_priority == 0:
            # No priority: use standard Oort on all available ends
            all_candidates = set(priority_ends + remaining_ends)
            selected = self._pick(
                eligible_ends, all_candidates, num_to_select, round_num
            )

        elif self.avail_priority == 1:
            # Fill mode: prioritize high-priority, fill remaining from others
            selected = self._select_priority_fill(
                eligible_ends, priority_ends, remaining_ends, num_to_select, round_num
            )

        elif self.avail_priority == 2:
            # Strict mode: only select from high-priority clients
            # REFL fork: all priority clients when they fit, then the sampler over the rest.
            if len(priority_ends) <= num_to_select:
                selected = list(priority_ends) + self._pick(
                    eligible_ends, set(remaining_ends), num_to_select - len(priority_ends), round_num
                )
            else:
                selected = self._pick(
                    eligible_ends, set(priority_ends), num_to_select, round_num
                )

        else:
            logger.warning(
                f"Unknown avail_priority={self.avail_priority}, using mode 0"
            )
            all_candidates = set(priority_ends + remaining_ends)
            selected = self._pick(
                eligible_ends, all_candidates, num_to_select, round_num
            )

        newly_selected = set(selected)
        self.newly_selected_this_round = newly_selected
        self.selected_ends = self.selected_ends | newly_selected

        logger.info(
            f"[REFL_SELECT] round {round_num}: selected {len(newly_selected)} new, "
            f"in-flight total {len(self.selected_ends)}"
        )

        self._last_selection_round = round_num

        for end_id in selected:
            if end_id in ends:
                count = ends[end_id].get_property(PROP_SELECTED_COUNT) or 0
                ends[end_id].set_property(PROP_SELECTED_COUNT, count + 1)

        # Emit selector-decision telemetry (same schema as oort/feddance) so the
        # mis-selection oracle can score REFL too.
        self.emit_selection(
            round_num, task_to_perform, ends, eligible_ends.keys(), newly_selected,
            per_trainer_extra=getattr(self, "_audit_components", None),
            extra={
                "avail_priority": self.avail_priority,
                "num_priority": len(priority_ends),
                "num_blacklist": len(blacklist),
                "exploration_factor": self.exploration_factor,
                "explore_ids": list(getattr(self, "_last_explore", [])),
                "exploit_ids": list(getattr(self, "_last_exploit", [])),
                "num_unexplored": sum(
                    1 for e in eligible_ends.values() if e.get_property(PROP_STAT_UTILITY) is None
                ),
                "vclock_now": channel_props.get("vclock_now"),
            },
        )
        return {key: None for key in newly_selected}

    def build_priority_lists(
        self,
        ends: Dict[str, End],
        cur_time: float,
        round_duration: float,
        blacklist: Set[str],
        unavail_list: List[str],
    ):
        """
        Build priority and remaining client lists based on availability.

        Args:
            ends: All available ends
            cur_time: Current virtual time
            round_duration: Estimated duration of round
            blacklist: Set of blacklisted end IDs
            unavail_list: List of currently unavailable trainers

        Returns:
            Tuple of (priority_end_ids, remaining_end_ids)
        """
        if not self.avail_tracker or self.avail_priority == 0:
            # No availability tracking, all clients have equal priority
            available = [
                eid
                for eid in ends.keys()
                if eid not in blacklist and eid not in unavail_list
            ]
            return [], available

        # Get all available (online) clients
        available_end_ids = [
            eid
            for eid in ends.keys()
            if eid not in blacklist and eid not in unavail_list
        ]

        # Split by priority using availability tracker
        priority_ends, remaining_ends = self.avail_tracker.split_by_priority(
            available_end_ids,
            cur_time,
            round_duration,
            lookup_timeslots=2,
        )
        if 0 < self.avail_probability < 1:  # REFL fork: keep a seeded fraction of each list (predictor accuracy)
            keep = lambda xs: self._keyed(xs, int(len(xs) * self.avail_probability), "refl_acc")
            priority_ends, remaining_ends = keep(priority_ends), keep(remaining_ends)

        return priority_ends, remaining_ends

    def _select_priority_fill(
        self,
        ends: Dict[str, End],
        priority_ends: List[str],
        remaining_ends: List[str],
        num_to_select: int,
        round_num: int,
    ) -> List[str]:
        """avail_priority=1, REFL fork `resampleClients`: the feasible set is every priority
        client plus a RANDOM fill from the rest up to num_to_select; Oort then picks from it."""
        if self.sample_mode != "oort":  # shuffle(priority + random fill)[:k] = priority first, then a random fill
            if len(priority_ends) >= num_to_select:
                return self._keyed(priority_ends, num_to_select, "refl")
            return list(priority_ends) + self._keyed(remaining_ends, num_to_select - len(priority_ends), "refl")
        feasible = list(priority_ends)
        remain = num_to_select - len(feasible)
        if remain > 0 and remaining_ends:
            feasible += self._pyrng.sample(sorted(remaining_ends), min(remain, len(remaining_ends)))
        return self._pick(ends, set(feasible), num_to_select, round_num)

    def _pick(self, ends: Dict[str, End], candidate_end_ids: Set[str], num_to_select: int, round_num: int) -> List[str]:
        """REFL fork `resampleClients`: Oort UCB in sample_mode 'oort', else a uniform draw."""
        if self.sample_mode == "oort":
            return self._select_with_oort_ucb(ends, candidate_end_ids, num_to_select, round_num)
        return self._keyed(candidate_end_ids, num_to_select, "refl")

    def _keyed(self, ids, k: int, salt: str) -> List[str]:
        """Uniform k-subset from per-id draws keyed on (seed, salt, version, id): pool order never shifts a draw."""
        draw = lambda c: random.Random(f"{self._seed}|{salt}|{self._draw_key}|{c}").random()
        return sorted(ids, key=draw, reverse=True)[:k]

    def _select_with_oort_ucb(
        self,
        ends: Dict[str, End],
        candidate_end_ids: Set[str],
        num_to_select: int,
        round_num: int,
    ) -> List[str]:
        """REFL fork `getTopK` (thirdparty/oort/oort.py:261-406) over `candidate_end_ids`.

        Exploit at most len(explored)-1 by weighted draw above the cut-off; explore every
        remaining slot from the UNEXPLORED (no stat_utility yet), weighted by the arm's
        registration reward over the top `sample_window` x slots; pad at random.
        """
        if not candidate_end_ids or num_to_select == 0:
            return []

        # sorted(): candidate_end_ids is a set (hash-order); canonicalize so the
        # utility list + seeded draws are cross-mode-stable.
        candidate_ends = {eid: ends[eid] for eid in sorted(candidate_end_ids) if eid in ends}

        if len(candidate_ends) <= num_to_select:
            self._last_explore, self._last_exploit = [], []
            return list(candidate_ends.keys())

        # Reference decays exploration at the top of getTopK, before sizing the split.
        self.update_exploration_factor()

        utility_list = []
        unexplored = []
        for end_id, end in candidate_ends.items():
            stat_util = end.get_property(PROP_STAT_UTILITY)
            if stat_util is not None:
                utility_list.append({PROP_END_ID: end_id, PROP_UTILITY: stat_util})
            else:
                unexplored.append(end_id)

        exploit_clients = []
        if utility_list:
            # Calculate total utility with temporal uncertainty and system utility
            utility_list = self.calculate_total_utility(
                utility_list, candidate_ends, round_num
            )
            exploration_len = int(num_to_select * self.exploration_factor)
            num_exploit = min(num_to_select - exploration_len, len(utility_list) - 1)

            # Sort by utility (descending)
            utility_list = sorted(utility_list, key=lambda x: x[PROP_UTILITY], reverse=True)

            # Exploitation, faithful to the REFL fork (thirdparty/oort/oort.py:316-355):
            # threshold at cut_off_util * the exploitLen-th-highest score, augment the pool
            # down to that cutoff (or 10x exploitLen), then sample exploitLen WEIGHTED by utility.
            if num_exploit > 0:
                boundary = min(num_exploit, len(utility_list) - 1)
                cutoff = self.cut_off_util * utility_list[boundary][PROP_UTILITY]
                pool = []
                for item in utility_list:
                    if item[PROP_UTILITY] < cutoff and len(pool) > 10 * num_exploit:
                        break
                    pool.append(item)
                scores = np.array([max(p[PROP_UTILITY], 0.0) for p in pool], dtype=np.float64)
                ids = [p[PROP_END_ID] for p in pool]
                k = min(num_exploit, len(ids))
                # Stamp the final per-candidate draw probability + pool/cutoff membership
                # into the audit so the parity checker can compare the SELECTION weighting
                # (not just per-term KS) sim vs real — the divergence that drives refl K3b.
                _audit = getattr(self, "_audit_components", None)
                if _audit is not None:
                    _tot = float(scores.sum())
                    for _i, _eid in enumerate(ids):
                        if _eid in _audit:
                            _audit[_eid]["in_exploit_pool"] = True
                            _audit[_eid]["exploit_cutoff"] = cutoff
                            _audit[_eid]["selection_prob"] = (
                                float(scores[_i]) / _tot if _tot > 0 else 0.0
                            )
                if scores.sum() > 0:
                    exploit_clients = [
                        str(c) for c in self._rng.choice(ids, k, replace=False, p=scores / scores.sum())
                    ]
                else:
                    exploit_clients = ids[:k]

        explore_clients = []
        if unexplored:
            explore_len = min(len(unexplored), num_to_select - len(exploit_clients))
            if explore_len > 0:
                # Registration reward = dataset size when known (reference: min(size, local
                # steps x batch)); unknown before a first update, so equal. Seeded shuffle
                # first so equal rewards don't bias the window toward low ids.
                order = list(unexplored)
                self._pyrng.shuffle(order)
                reward = {e: float(candidate_ends[e].get_property(PROP_DATASET_SIZE) or 1.0) for e in order}
                order.sort(key=lambda e: reward[e], reverse=True)
                window = order[:max(explore_len, min(int(self.sample_window * explore_len), len(order)))]
                w = np.array([reward[e] for e in window], dtype=np.float64)
                explore_clients = [
                    str(c) for c in self._rng.choice(window, explore_len, replace=False, p=w / w.sum())
                ]

        selected = explore_clients + exploit_clients

        # Pad with random if needed (sorted() for cross-process-stable order)
        while len(selected) < num_to_select and len(candidate_ends) > len(selected):
            remaining = sorted(set(candidate_ends) - set(selected))
            selected.append(self._pyrng.choice(remaining))

        self._last_explore, self._last_exploit = explore_clients, exploit_clients
        return selected[:num_to_select]

    def get_blacklist(self, ends: Dict[str, End]) -> Set[str]:
        """
        Get set of blacklisted end IDs based on selection frequency.

        Args:
            ends: All available ends

        Returns:
            Set of blacklisted end IDs
        """
        if self.blacklist_rounds < 0:
            return set()

        blacklist = []

        # Sort by selection count (descending)
        end_counts = []
        for end_id, end in ends.items():
            count = end.get_property(PROP_SELECTED_COUNT)
            if count and count > self.blacklist_rounds:
                end_counts.append((end_id, count))

        # Sort by count descending
        end_counts.sort(key=lambda x: x[1], reverse=True)

        # Take top blacklist_max_len fraction
        max_blacklist = int(self.blacklist_max_len * len(ends))
        blacklist = [end_id for end_id, _ in end_counts[:max_blacklist]]

        if blacklist:
            logger.debug(f"Blacklisted {len(blacklist)} ends: {blacklist}")

        return set(blacklist)

    # pacer() inherited from OortSelector — the base is the faithful reference port
    # (flat→relax / sharp→tighten, current-round keyed), matching the REFL fork, so no
    # override is needed here (§S.pacer, third_party/Oort/oort/oort.py:184-199).
