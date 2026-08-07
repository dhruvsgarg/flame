# Copyright 2023 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License"); you
# may not use this file except in compliance with the License. You may
# obtain a copy of the License at
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
"""AsyncOortSelector class.

Re-based onto `AsyncSelectorBase` (simulate_fwdllm.md known gap): this file
used to carry its own copy of the send/recv concurrency mechanism, drifted
from fixes since landed only on the base (R1 pending-commit guard,
recv-bootstrap gate). Now holds Oort POLICY only -- utility scoring, pacer,
exploration/exploitation, the `select_type` strategies. Eval-task support
(`eval_goal_factor`/`curr_round_eval_slots_left`) is genuine extra state the
base doesn't have, threaded through the `_concurrency_for_task`/
`_task_extra_eligible`/`_choose`/`_cleanup_recvd_ends` hooks instead.
"""

import logging
import math
from datetime import timedelta

from flame.common.typing import Scalar
from flame.common.util import MLFramework, get_ml_framework_in_use
from flame.end import End
from flame.selector import scoring
from flame.selector.async_base import AsyncSelectorBase, SelectContext
from flame.selector.properties import (
    PROP_CLIENT_TASK_TRAIN_DURATION,
    PROP_DATASET_SIZE,
    PROP_END_ID,
    PROP_LAST_EVAL_ROUND,
    PROP_LAST_SELECTED_ROUND,
    PROP_SELECTED_COUNT,
    PROP_STAT_UTILITY,
    PROP_TOTAL_UNAVAIL_DURATION,
    PROP_UPDATE_COUNT,
    PROP_UTILITY,
)

logger = logging.getLogger(__name__)


class AsyncOortSelector(AsyncSelectorBase):
    """An AsyncFL selector class based on Oort."""

    CHOOSE_SALT = "async_oort"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        ml_framework_in_use = get_ml_framework_in_use()
        if ml_framework_in_use != MLFramework.PYTORCH:
            raise NotImplementedError(
                "FedBalancer is currently only implemented in PyTorch;"
            )

        # Last round pacer() actually ran for, so a same-round dispatch burst
        # can't re-fire it.
        self._last_pacer_round = None

        try:
            self.is_async = kwargs["is_async"]
        except KeyError:
            logger.info(
                "is_async param isn't specified in config. Defaulting to sync version"
            )
            self.is_async = False

        try:
            self.eval_goal_factor = kwargs["evalGoalFactor"]
        except KeyError:
            raise KeyError(
                "evalGoalFactor is not specified in config. It is the decimal "
                "multiplicative factor wrt agg goal for eval"
            )

        try:
            self.round_nudge_type = kwargs["roundNudgeType"]
        except KeyError:
            raise KeyError(
                "roundNudgeType is not specified in config. It is last_train or "
                "last_eval based on the selector nudging critera"
            )

        try:
            self.select_type = kwargs["selectType"]
        except KeyError:
            raise KeyError(
                "selectType is not specified in config. Can be default, "
                "fastest, or maxSamples"
            )

        # With Oort, we select 1.3 * k ends and wait until k ends to
        # complete at a round
        self.overcommitment = 1.3
        self.num_of_ends = int(self.agg_goal * self.overcommitment)

        # Algorithm hyperparameters default to the Oort paper
        # (scoring.OORT_PAPER_DEFAULTS), overridable via selector.kwargs.
        _d = scoring.OORT_PAPER_DEFAULTS
        self.exploration_factor = kwargs.get("exploration_factor", _d["exploration_factor"])
        self.exploration_factor_decay = kwargs.get("exploration_decay", _d["exploration_decay"])
        self.min_exploration_factor = kwargs.get("exploration_min", _d["exploration_min"])

        self.exploitation_util_history = []

        # Assuming a max round duration of 99999 seconds (~1.2 days)
        self.round_preferred_duration = timedelta(seconds=99999)
        self.round_threshold = kwargs.get("round_threshold", _d["round_threshold"])
        self.pacer_delta = kwargs.get("pacer_delta", _d["pacer_delta"])
        self.pacer_step = kwargs.get("pacer_step", _d["pacer_step"])

        self.blocklist_threshold = -1

        self.alpha = kwargs.get("round_penalty", _d["round_penalty"])  # system_util exponent

        # Normalize+clip the reward into ~[0,1] before adding temporal.
        self.normalize_reward = kwargs.get("normalize_reward", True)
        self.clip_bound = kwargs.get("clip_bound", _d["clip_bound"])
        self.cut_off_util = kwargs.get("cut_off_util", _d["cut_off_util"])  # breadth factor

        # Tracks eval updates received from trainers and makes them available
        # to select again. Populated externally (duck-typed hasattr check) by
        # top_aggregator.py; not a base-mechanism concept.
        self.trainer_eval_recv_ends = list()
        self.curr_round_eval_slots_left = int(self.eval_goal_factor * self.agg_goal)

    # ------------------------------------------------------------------ hooks

    def _concurrency_for_task(self, task_to_perform, num_ends, effective_c):
        if task_to_perform == "eval":
            if self.eval_goal_factor <= 0.0:
                return 0
            # Maximum possible concurrency for eval; narrowed later in
            # `_choose` based on eval tasks already sent/received this round.
            return min(num_ends, effective_c + self.curr_round_eval_slots_left)
        return min(num_ends, effective_c)

    def _pre_choose(self, ctx: SelectContext) -> None:
        """Run the pacer that controls `round_threshold`. TRAIN-ONLY: felix's
        eval hand leaves self.round/exploitation_util_history unchanged, so
        firing on eval would re-adjust off a stale round. ONCE-PER-ROUND:
        unlike Oort's reference (called once/round by construction), `select`
        fires many times per model_version (once per freed slot) -- without
        this guard every call re-fires the pacer, ratcheting round_threshold
        to 100 in one dispatch burst."""
        if ctx.task_to_perform == "train" and ctx.model_version != self._last_pacer_round:
            self.pacer(ctx.model_version)
            self._last_pacer_round = ctx.model_version

    def _task_extra_eligible(self, end_id, end, ctx: SelectContext) -> bool:
        """Eval-only staleness gate: only consider trainers that have trained
        before AND whose last eval is at least 35 model-versions old."""
        if ctx.task_to_perform != "eval":
            return True
        last_eval = end.get_property(PROP_LAST_EVAL_ROUND)
        return last_eval is not None and ctx.model_version - last_eval >= 35

    def _choose(self, candidates: dict[str, End], k: int, ctx: SelectContext) -> list:
        if ctx.task_to_perform == "eval":
            return self._choose_eval(candidates, k)
        return self._choose_train(candidates, k, ctx)

    def _choose_eval(self, candidates: dict[str, End], k: int) -> list:
        """Pick the `k` (capped by remaining eval headroom) candidates whose
        eval participation is stalest (ascending last_eval_round)."""
        feasible_extra = min(k, self.curr_round_eval_slots_left)
        end_to_last_eval = {
            end_id: end.get_property(PROP_LAST_EVAL_ROUND) or 0
            for end_id, end in candidates.items()
        }
        sorted_end_ids = sorted(end_to_last_eval, key=end_to_last_eval.get)
        chosen = sorted_end_ids[:feasible_extra]
        self.curr_round_eval_slots_left -= len(chosen)
        return chosen

    def _choose_train(self, candidates: dict[str, End], k: int, ctx: SelectContext) -> list:
        model_version = ctx.model_version
        agg_version_key = ctx.agg_version_key
        feasible_extra = k

        blocklist_end_ids = self.find_blocklists(candidates)
        trainer_unavail_list = ctx.trainer_unavail_list or []
        utility_list, unexplored_end_ids = self.fetch_statistical_utility(
            candidates, blocklist_end_ids, trainer_unavail_list
        )
        exploration_len, exploitation_len = self.calculate_num_of_exploration_exploitation(
            num_of_ends=feasible_extra, unexplored_end_ids=unexplored_end_ids
        )

        # First round: no end's utility has been measured yet -- random.
        if model_version == 0:
            self.round = model_version
            return self.select_random(
                candidates, num_of_ends=feasible_extra, agg_version_key=agg_version_key
            )

        # The percentile must see the FULL registered client population, not
        # this call's transient candidate pool -- reference Oort computes it
        # from ALL tracked arms, not just this dispatch's feasible set. Async's
        # candidate pool is almost always a singleton, so the percentile would
        # trivially return that one candidate's own duration and the speed
        # penalty could never bind. `connected_ends` is safe: only used for
        # lookups on ids already in utility_list, plus this percentile calc.
        _duration_pool = ctx.connected_ends if ctx.connected_ends is not None else candidates
        utility_list = self.calculate_total_utility(utility_list, _duration_pool, model_version)

        cutoff_utility = self.cutoff_util(utility_list, num_of_ends=feasible_extra)

        if len(utility_list) == 0:
            self.round = model_version
            return self.select_random(
                candidates, num_of_ends=feasible_extra, agg_version_key=agg_version_key
            )

        if self.select_type == "default":
            chosen, exploit_end_ids = self._select_candidates_using_default(
                cutoff_utility=cutoff_utility,
                utility_list=utility_list,
                exploitation_len=exploitation_len,
                exploration_len=exploration_len,
                unexplored_end_ids=unexplored_end_ids,
                agg_version_key=agg_version_key,
            )
        elif self.select_type == "fastest":
            chosen, exploit_end_ids = self._select_candidates_fastest(
                ends=candidates, num_of_ends=feasible_extra
            )
        elif self.select_type == "maxSamples":
            chosen, exploit_end_ids = self._select_candidates_maxSamples(
                ends=candidates, num_of_ends=feasible_extra
            )
        elif self.select_type == "prioritiseUnavail":
            chosen, exploit_end_ids = self._select_candidates_prioritiseUnavail(
                ends=candidates, num_of_ends=feasible_extra
            )
        elif self.select_type == "fairShare":
            chosen, exploit_end_ids = self._select_candidates_fairShare(
                ends=candidates, num_of_ends=feasible_extra
            )

        if self.select_type == "default":
            self.save_exploited_utility_history(candidates, exploit_end_ids)
            self.update_exploration_factor()

        self.increment_selected_count_on_selected_ends(
            candidates, {end_id: candidates[end_id] for end_id in chosen}
        )
        self.round = model_version
        return chosen

    def _selection_extra(self, ctx: SelectContext, results) -> dict:
        _pref = getattr(self, "round_preferred_duration", None)
        return {
            "exploration_factor": self.exploration_factor,
            "round_preferred_duration_s": (
                _pref.total_seconds() if hasattr(_pref, "total_seconds") else _pref
            ),
            "round_threshold": getattr(self, "round_threshold", None),
            "alpha": getattr(self, "alpha", None),
            **self._system_util_summary(results.keys()),
        }

    def _per_trainer_selection_extra(self, ends: dict[str, End]) -> dict:
        audit = getattr(self, "_audit_components", {}) or {}
        return {
            end_id: {
                "in_all_selected": end_id in self.all_selected,
                "in_pending_commit": end_id in getattr(self, "_agg_pending_commit_ref", set()),
                "last_eval_round": end.get_property(PROP_LAST_EVAL_ROUND),
                # last TRAIN selection round; with last_eval_round this shows
                # whether Felix's believed I_m was refreshed by eval vs train
                # (the freshness mechanism the staleness audit measures).
                "last_train_round": end.get_property(PROP_LAST_SELECTED_ROUND),
                # score components (believed_I, temporal, system_util) for the
                # offline counterfactual replay.
                **(audit.get(end_id, {})),
            }
            for end_id, end in ends.items()
        }

    # ---------------------------------------------------------- eval cleanup

    def _cleanup_recvd_ends(self, ends: dict[str, End]) -> None:
        """Fold eval-recv'd ends into the same drain as train commits (both
        free their concurrency slot) and reset the per-round eval-slot
        counter. Gated on `ordered_updates_recv_ends` being non-empty first --
        matching the original: an eval-only cycle with no train commit this
        tick does not reset eval slots or drain `trainer_eval_recv_ends`."""
        if not self.ordered_updates_recv_ends:
            return
        self.ordered_updates_recv_ends = (
            self.ordered_updates_recv_ends + self.trainer_eval_recv_ends
        )
        self.trainer_eval_recv_ends = []
        self.curr_round_eval_slots_left = int(self.eval_goal_factor * self.agg_goal)
        super()._cleanup_recvd_ends(ends)

    def _cleanup_removed_ends(self, end_id: str) -> None:
        """Release an end that left the channel -- and, unlike the base
        (which only touches `self.requester`'s in-flight set), sweep it as a
        ghost from EVERY requester's `selected_ends`. A departed end must not
        linger in any requester's in-flight set, else that aggregator waits on
        a gone trainer and wastes a concurrency slot."""
        if (end_id in self.all_selected) and (end_id not in self.ordered_updates_recv_ends):
            # Remove end from all_selected if we haven't got an update from it
            # yet. It would have flushed the agg-weights after initiating
            # channel.leave().
            selected_ends = self.selected_ends[self.requester]
            if end_id in selected_ends:
                selected_ends.remove(end_id)
                self.selected_ends[self.requester] = selected_ends

            self.track_selected_trainers_which_left[end_id] = (
                self.track_selected_trainers_which_left.get(end_id, 0) + 1
            )
            if end_id in self.all_selected.keys():
                del self.all_selected[end_id]
        elif (end_id in self.all_selected) and (end_id in self.ordered_updates_recv_ends):
            # Update was already received before it left -- participation is
            # complete, don't touch all_selected now.
            logger.debug(
                f"Update was already received from {end_id} before it left "
                f"the channel. Not deleting from all_ends now."
            )
        else:
            logger.warning(
                f"End_id {end_id} remove check from all_selected failed. "
                f"Need to check"
            )

        for _req, _ends in self.selected_ends.items():
            if end_id in _ends:
                _ends.discard(end_id)
                logger.debug(f"Removed ghost end_id {end_id} from selected_ends[{_req}]")

    # ------------------------------------------------------------ Oort policy

    def cutoff_util(
        self,
        sorted_utility_list: list[tuple[str, float]],
        num_of_ends: int,
    ) -> float:
        """Cutoff = cut_off_util * the (exploitLen-th HIGHEST) score (ref Oort
        oort.py:329). `sorted_utility_list` is ASCENDING -> index len-1-exploitLen."""
        if not sorted_utility_list:
            logger.debug("Got empty utility_list, returning 999999.0")
            return 999999.0

        exploit_len = int(num_of_ends * (1.0 - self.exploration_factor))
        index = len(sorted_utility_list) - 1 - exploit_len
        index = max(0, min(index, len(sorted_utility_list) - 1))

        return self.cut_off_util * sorted_utility_list[index][PROP_UTILITY]

    def sample_by_util(
        self,
        cutoff_utility: float,
        utility_list: list[dict[str, Scalar]],
        num_of_ends: int,
        agg_version_key=None,
    ) -> list[str]:
        """Sample num_of_ends clients by utility.

        `_keyed_weighted_topk`, not `np.random.choice(p=...)` -- the latter is
        pool-size/order-dependent, same anti-pattern `_keyed_topk` already
        fixed for the plain uniform draw (see its docstring).
        """
        over_cutoff_utility_end_ids = []
        over_cutoff_utility_probs = []
        over_cutoff_utility_sum = 0

        under_cutoff_utility_list = []

        # Divide ends on whether its utility exceeds cutoff_loss or not
        for utility_pair in utility_list:
            if utility_pair[PROP_UTILITY] >= cutoff_utility:
                over_cutoff_utility_end_ids.append(utility_pair[PROP_END_ID])
                over_cutoff_utility_probs.append(utility_pair[PROP_UTILITY])
                over_cutoff_utility_sum += utility_pair[PROP_UTILITY]
            else:
                under_cutoff_utility_list.append(utility_pair)

        # Select clients on the probability based on the utility divided by
        # the utility sum
        for prob_idx in range(len(over_cutoff_utility_probs)):
            over_cutoff_utility_probs[prob_idx] /= over_cutoff_utility_sum

        # Exclude zero-weight entries; A-ExpJ's key formula divides by weight.
        nz_pairs = [
            (e, p)
            for e, p in zip(over_cutoff_utility_end_ids, over_cutoff_utility_probs)
            if p > 0
        ]
        if not nz_pairs:
            return []

        return self._keyed_weighted_topk(
            nz_pairs, min(len(nz_pairs), num_of_ends), agg_version_key,
            "sample_by_util",
        )

    def _system_util_summary(self, selected_ids) -> dict:
        """Per-round speed-penalty summary over selected ends, for telemetry.

        When `pref` is non-binding the system_util penalty never fires and the
        selector ignores speed; logging this makes that visible without recompute.
        Mirrors oort.py's (sync) equivalent, keyed on this round's actually-
        selected ends (`results.keys()` at the call site) rather than
        `self.selected_ends`, which async accumulates per-requester across
        rounds, not per-round.
        """
        audit = getattr(self, "_audit_components", None) or {}
        sel = [
            audit[e]["system_util"]
            for e in selected_ids
            if e in audit and audit[e].get("system_util") is not None
        ]
        if not sel:
            return {"sys_util_mean": None, "frac_penalized": None, "pref_binds": None}
        penalized = sum(1 for su in sel if su < 1.0)
        return {
            "sys_util_mean": sum(sel) / len(sel),
            "frac_penalized": penalized / len(sel),
            "pref_binds": penalized > 0,
        }

    def sample_by_speed(
        self, unexplored_end_ids: list[str], num_of_ends: int, agg_version_key=None
    ) -> list[str]:
        """Sample num_of_ends clients by speed.

        Oort paper prioritizes unexplored ends with faster system speed; we
        implement uniform order-sampling here (_keyed_topk).
        """
        return self._keyed_topk(unexplored_end_ids, num_of_ends, agg_version_key,
                                "sample_by_speed")

    def _keyed_weighted_topk(
        self, candidate_weights: list[tuple], k: int, agg_version_key, salt: str
    ) -> list[str]:
        """Weighted counterpart of `_keyed_topk`, same `_keyed_draw` primitive:
        Efraimidis-Spirakis (A-ExpJ) keys `u_i ** (1/w_i)` turn the pool-
        independent uniform draw into weighted sampling without replacement.
        Replaces `np.random.choice(p=probs)`, whose output for every
        candidate shifts when the pool's size/order changes.
        """
        def _key(item: tuple) -> float:
            end_id, weight = item
            return self._keyed_draw(end_id, agg_version_key, salt) ** (1.0 / weight)

        ranked = sorted(candidate_weights, key=_key, reverse=True)
        return [end_id for end_id, _weight in ranked[:k]]

    def pacer(self, current_round: int) -> None:
        """Adapt `round_threshold` from the exploited-utility trend — faithful to
        the Oort pacer (third_party/Oort/oort/oort.py:184-199, byte-identical in
        the third_party/REFL fork). TWO symmetric moves on the reference's 0.1 / 5×
        bands over the last two `pacer_step` windows: a FLAT plateau
        (`|Δ| <= 0.1·last`) RELAXES (`round_threshold += pacer_delta`); a SHARP
        change (`|Δ| >= 5·last`) TIGHTENS (floored at `pacer_delta`). Keyed on
        `current_round` (= reference `training_round`); caller train-gates it
        and must only invoke this once per genuine round change -- takes an
        explicit param since `self.round` lags until the caller sets it after
        this returns.
        """
        if not (
            self.pacer_step > 0
            and current_round >= 2 * self.pacer_step
            and current_round % self.pacer_step == 0
            and len(self.exploitation_util_history) >= 2 * self.pacer_step
        ):
            return
        last_util = sum(
            self.exploitation_util_history[-2 * self.pacer_step : -self.pacer_step]
        )
        curr_util = sum(self.exploitation_util_history[-self.pacer_step :])
        delta = abs(curr_util - last_util)
        if delta <= last_util * 0.1:
            self.round_threshold = min(100.0, self.round_threshold + self.pacer_delta)
        elif delta >= last_util * 5.0:
            self.round_threshold = max(
                self.pacer_delta, self.round_threshold - self.pacer_delta
            )

    def find_blocklists(self, ends: dict[str, End]) -> list[str]:
        """Make a filter of blocklist ends."""
        blocklist_end_ids = []
        if self.blocklist_threshold != -1:
            for end_id in ends.keys():
                if (
                    ends[end_id].get_property(PROP_SELECTED_COUNT)
                    > self.blocklist_threshold
                ):
                    blocklist_end_ids.append(end_id)
        return blocklist_end_ids

    def calculate_num_of_exploration_exploitation(
        self, num_of_ends: int, unexplored_end_ids: list[str]
    ) -> tuple[int, int]:
        """Calculate number of ends to select for exploration and
        exploitation; Add 1 to exploration_len to avoid not exploring 0 ends
        while unexplored ends exist."""
        exploration_len = min(
            int(num_of_ends * self.exploration_factor) + 1,
            len(unexplored_end_ids),
        )
        exploitation_len = num_of_ends - exploration_len

        return exploration_len, exploitation_len

    def fetch_statistical_utility(
        self,
        ends: dict[str, End],
        blocklist_end_ids: list[str],
        trainer_unavail_list: list[str],
    ) -> tuple[list[tuple[str, float]], list[str]]:
        """Make a list of tuple (end_id, end_utility) as an utility_list. As
        unexplored ends that are not selected before do not have utility
        value, collect them separately with unexplored_end_ids list."""
        utility_list = []
        unexplored_end_ids = []

        # sorted(): ends' dict order = trainer JOIN order (differs real vs sim);
        # canonicalize before the seeded draws over unexplored_end_ids/utility_list
        # so a shared seed picks identical cohorts in both modes.
        for end_id in sorted(ends.keys()):
            if (end_id not in blocklist_end_ids) and (
                end_id not in trainer_unavail_list
            ):
                end_utility = ends[end_id].get_property(PROP_STAT_UTILITY)
                if end_utility is not None:
                    utility_list.append(
                        {PROP_END_ID: end_id, PROP_UTILITY: end_utility}
                    )
                else:
                    unexplored_end_ids.append(end_id)

        return utility_list, unexplored_end_ids

    def calculate_round_preferred_duration(self, ends: dict[str, End]) -> float:
        """Calculate round preferred duration based on round_threshold and
        end_round_duration of trainers. round_threshold is controlled by
        pacer."""
        if self.round_threshold < 100.0:
            sorted_round_duration = []
            for end_id in ends.keys():
                end_round_duration = ends[end_id].get_property(PROP_CLIENT_TASK_TRAIN_DURATION)
                if end_round_duration is not None:
                    sorted_round_duration.append(end_round_duration)
                elif end_round_duration is None:
                    # Comes here if the trainer participates in eval, so
                    # technically doesn't have a round duration. Set it to 60
                    # seconds since that is the max round duration for training.
                    sorted_round_duration.append(timedelta(seconds=60))
            # pref = round_threshold-th PERCENTILE -> sort first (ref Oort oort.py:272)
            sorted_round_duration.sort()
            round_preferred_duration = timedelta(
                seconds=sorted_round_duration[
                    min(
                        int(len(sorted_round_duration) * self.round_threshold / 100.0),
                        len(sorted_round_duration) - 1,
                    )
                ].total_seconds()
            )
        else:
            # Assuming a max round duration of 99999 seconds (~1.2 days)
            round_preferred_duration = timedelta(seconds=99999)

        return round_preferred_duration

    def calculate_temporal_uncertainty_of_trainer(
        self, ends: dict[str, End], end_id: str, model_version: int
    ) -> float:
        """Calculate temporal uncertainty term based on the end's last
        selected round."""
        if self.round_nudge_type == "last_train":
            end_last_selected_round = ends[end_id].get_property(
                PROP_LAST_SELECTED_ROUND
            )
        elif self.round_nudge_type == "last_eval":
            end_last_selected_round = ends[end_id].get_property(PROP_LAST_EVAL_ROUND)

        if (
            model_version == 0
            or model_version == end_last_selected_round
            or end_last_selected_round in (None, 0)
        ):
            return 0

        trainer_temporal_uncertainty = math.sqrt(
            0.1 * math.log(model_version) / end_last_selected_round
        )
        return trainer_temporal_uncertainty

    def calculate_global_system_utility_of_trainer(
        self, ends: dict[str, End], end_id: str
    ) -> float:
        """Calculate global system utility based on the end's round duration."""
        end_round_duration = ends[end_id].get_property(PROP_CLIENT_TASK_TRAIN_DURATION)

        # In normal training, the util of trainer is 1 if it is faster than
        # preferred round duration. This is a multiplier to the trainer
        # utility. Thus, if the trainer is slower than preferred round
        # duration, the multiplier is (0, 1) which means that the utility of
        # the trainer decreases.
        #
        # For eval-enabled training, it is possible that the trainer hasn't
        # trained yet but has only pushed an eval update. For these trainers,
        # we retain the multiplicative factor as 1 so as to incentivise them
        # to be picked whenever available to train.
        if end_round_duration is None:
            return 1

        if end_round_duration <= self.round_preferred_duration:
            return 1
        else:
            return math.pow(
                self.round_preferred_duration.total_seconds()
                / end_round_duration.total_seconds(),
                self.alpha,
            )

    def save_exploited_utility_history(
        self, ends: dict[str, End], exploit_end_ids: list[str]
    ) -> None:
        """Save the history of exploited utility at this round for pacer."""
        if len(exploit_end_ids) > 0:
            exploited_utility = 0
            for exploit_end_id in exploit_end_ids:
                exploited_utility += ends[exploit_end_id].get_property(
                    PROP_STAT_UTILITY
                )
            exploited_utility /= len(exploit_end_ids)
            self.exploitation_util_history.append(exploited_utility)

    def update_exploration_factor(self) -> None:
        """Update the exploration_factor."""
        self.exploration_factor = max(
            self.exploration_factor * self.exploration_factor_decay,
            self.min_exploration_factor,
        )

    def increment_selected_count_on_selected_ends(
        self, ends: dict[str, End], candidates: dict[str, End]
    ) -> None:
        """Increment the round selected count on selected ends."""
        for end_id in candidates:
            if ends[end_id].get_property(PROP_SELECTED_COUNT) is None:
                ends[end_id].set_property(PROP_SELECTED_COUNT, 1)
            else:
                ends[end_id].set_property(
                    PROP_SELECTED_COUNT,
                    ends[end_id].get_property(PROP_SELECTED_COUNT) + 1,
                )

    def select_random(self, ends: dict[str, End], num_of_ends: int,
                       agg_version_key=None) -> list[str]:
        """Select num_of_ends ends via _keyed_topk -- population-size-
        independent, unlike random.sample() (see _keyed_topk docstring)."""
        return self._keyed_topk(sorted(ends), num_of_ends, agg_version_key,
                                "select_random")

    def calculate_total_utility(
        self,
        utility_list: list[tuple[str, float]],
        ends: dict[str, End],
        model_version: int,
    ) -> list[tuple[str, float]]:
        """Calculate the total utility value of trainers with applying
        temporal uncertainty and global system utility, based on the Oort
        algorithm."""
        if utility_list == []:
            return []

        self.round_preferred_duration = self.calculate_round_preferred_duration(ends)

        # Normalize+clip the statistical reward across candidates (reference
        # Oort get_norm) so the temporal term is meaningful.
        if self.normalize_reward:
            _min, _range, _clip = scoring.oort_norm_stats(
                [u[PROP_UTILITY] for u in utility_list], self.clip_bound
            )
        else:
            _min, _range, _clip = None, None, None

        # Per-candidate score components stashed for the offline staleness audit.
        if getattr(self, "_audit_round", None) != model_version:
            self._audit_components = {}
            self._audit_round = model_version
        for utility_idx in range(len(utility_list)):
            curr_end_utility = utility_list[utility_idx][PROP_UTILITY]
            curr_end_id = utility_list[utility_idx][PROP_END_ID]

            stat_utility = curr_end_utility
            if self.normalize_reward and _range is not None:
                stat_utility = scoring.oort_normalize_reward(
                    stat_utility, _min, _range, _clip
                )

            temporal_uncertainty = self.calculate_temporal_uncertainty_of_trainer(
                ends, curr_end_id, model_version
            )
            global_system_utility = self.calculate_global_system_utility_of_trainer(
                ends, curr_end_id
            )

            self._audit_components[curr_end_id] = {
                "believed_I": stat_utility,
                "temporal": temporal_uncertainty,
                "system_util": global_system_utility,
            }
            # Score = (stat_util + temporal) * system_util via shared scorer.
            utility_list[utility_idx][PROP_UTILITY] = scoring.oort_combine_score(
                stat_utility, temporal_uncertainty, global_system_utility
            )

        # Explicit end_id tie-break: pre-sort order is already canonical
        # (sorted(ends.keys())), making the existing stable-sort tie-break
        # explicit instead of incidental.
        return sorted(utility_list, key=lambda x: (x[PROP_UTILITY], x[PROP_END_ID]))

    # Invoked when selection mode is oort's default (tradeoff between
    # exploitation/exploration and speed/stat_utility)
    def _select_candidates_using_default(
        self,
        cutoff_utility,
        utility_list,
        exploitation_len,
        exploration_len,
        unexplored_end_ids,
        agg_version_key=None,
    ):
        exploit_end_ids = self.sample_by_util(
            cutoff_utility, utility_list, exploitation_len,
            agg_version_key=agg_version_key,
        )

        # sample exploration_len of unexplored clients
        explore_end_ids = []
        if self.exploration_factor > 0.0 and len(unexplored_end_ids) > 0:
            explore_end_ids = self.sample_by_speed(
                unexplored_end_ids, exploration_len, agg_version_key=agg_version_key
            )

        candidates = [*explore_end_ids, *exploit_end_ids]
        return candidates, exploit_end_ids

    # Invoked when selection mode is maxSamples i.e. select clients with
    # largest local datasets
    def _select_candidates_maxSamples(
        self,
        ends: dict[str, End],
        num_of_ends: int,
    ) -> tuple[list[str], list[str]]:
        # get the end properties
        end_id_to_samples = {}
        for key, val in ends.items():
            # if the PROP_DATASET_SIZE is None, it means the trainer hasnt
            # trained even once till now. So we set it to 99999 to prioritize
            # it to get picked up atleast once.
            end_num_samples = val.get_property(PROP_DATASET_SIZE)
            if end_num_samples is None:
                end_num_samples = 99999
            end_id_to_samples[key] = end_num_samples

        # sort it in descending order of number of samples
        sorted_end_ids = list(
            dict(
                sorted(
                    end_id_to_samples.items(), key=lambda item: item[1], reverse=True
                )
            ).keys()
        )

        exploit_end_ids = []
        candidates = sorted_end_ids[:num_of_ends]
        return candidates, exploit_end_ids

    # Invoked when selection mode is fastest i.e. select fastest clients
    def _select_candidates_fastest(
        self,
        ends: dict[str, End],
        num_of_ends: int,
    ) -> tuple[list[str], list[str]]:
        # get the end properties
        end_id_to_round_durations = {}
        for key, val in ends.items():
            # if the PROP_CLIENT_TASK_TRAIN_DURATION is None, it means the
            # trainer hasnt trained even once till now. So we set it to
            # 00:00:00.000000 (upto microseconds) to prioritize it to get
            # picked up atleast once.
            round_duration = val.get_property(PROP_CLIENT_TASK_TRAIN_DURATION)
            if round_duration is None:
                round_duration = timedelta(
                    hours=0, minutes=0, seconds=0, microseconds=0
                )
            end_id_to_round_durations[key] = round_duration

        # sort it in ascending order of durations
        sorted_end_ids = list(
            dict(
                sorted(
                    end_id_to_round_durations.items(),
                    key=lambda item: item[1],
                    reverse=False,
                )
            ).keys()
        )

        exploit_end_ids = []
        candidates = sorted_end_ids[:num_of_ends]
        return candidates, exploit_end_ids

    # Invoked when selection mode is fair-share i.e. select clients such that
    # all clients participate almost equally
    def _select_candidates_fairShare(
        self,
        ends: dict[str, End],
        num_of_ends: int,
    ) -> tuple[list[str], list[str]]:
        # get the end properties
        end_id_to_update_count = {}
        for key, val in ends.items():
            # if the PROP_UPDATE_COUNT is None, it means the trainer hasnt
            # trained even once till now. So we set it to 0 to prioritize it
            # to get picked up atleast once.
            update_count = val.get_property(PROP_UPDATE_COUNT)
            if update_count is None:
                update_count = 0
            end_id_to_update_count[key] = update_count

        # sort it in ascending order of durations
        sorted_end_ids = list(
            dict(
                sorted(
                    end_id_to_update_count.items(),
                    key=lambda item: item[1],
                    reverse=False,
                )
            ).keys()
        )

        exploit_end_ids = []
        candidates = sorted_end_ids[:num_of_ends]
        return candidates, exploit_end_ids

    # Invoked when selection mode is prioritiseUnavail i.e. select and
    # prioritize clients that have been unavailable for long durations of the
    # training time
    def _select_candidates_prioritiseUnavail(
        self,
        ends: dict[str, End],
        num_of_ends: int,
    ) -> tuple[list[str], list[str]]:
        # get the end properties
        end_id_to_unavail_durations = {}
        for key, val in ends.items():
            # if the PROP_TOTAL_UNAVAIL_DURATION is None, it means the
            # trainer hasnt failed even once till now. So we set it to
            # 00:00:00.000000 (upto microseconds) to allow it to get picked
            # whenever there are no other unavailable clients to pick.
            unavail_duration = val.get_property(PROP_TOTAL_UNAVAIL_DURATION)
            if unavail_duration is None:
                unavail_duration = timedelta(
                    hours=0, minutes=0, seconds=0, microseconds=0
                )
            end_id_to_unavail_durations[key] = unavail_duration

        # sort it in ascending order of durations
        sorted_end_ids = list(
            dict(
                sorted(
                    end_id_to_unavail_durations.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
            ).keys()
        )

        exploit_end_ids = []
        candidates = sorted_end_ids[:num_of_ends]
        return candidates, exploit_end_ids
