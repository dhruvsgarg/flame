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
"""OortSelector class."""

import logging
import math
import random
import time
from datetime import timedelta

from flame.config import TrainerAvailState
import numpy as np
from flame.channel import (
    KEY_CH_SELECT_REQUESTER,
    KEY_CH_STATE,
    VAL_CH_STATE_RECV,
    VAL_CH_STATE_SEND,
)
from flame.common.typing import Scalar
from flame.common.util import MLFramework, get_ml_framework_in_use
from flame.end import KEY_END_STATE, VAL_END_STATE_NONE, VAL_END_STATE_RECVD, End
from flame.selector import AbstractSelector, SelectorReturnType

logger = logging.getLogger(__name__)

SEND_TIMEOUT_WAIT_S = 90      # 90 seconds timeout

PROP_UTILITY = "utility"
PROP_END_ID = "end_id"
PROP_SELECTED_COUNT = "selected_count"
PROP_ROUND_START_TIME = "round_start_time"
PROP_ROUND_DURATION = "round_duration"
PROP_STAT_UTILITY = "stat_utility"
PROP_DATASET_SIZE = "dataset_size"
PROP_UPDATE_COUNT = "update_count"
PROP_TOTAL_UNAVAIL_DURATION = "total_unavail_duration"
PROP_LAST_SELECTED_ROUND = "last_selected_round"
PROP_LAST_EVAL_ROUND = "last_eval_round"
PROP_AVL_STATE = "avl_state"

class AsyncOortSelector(AbstractSelector):
    """A AsyncFL selector class based on Oort."""

    def __init__(self, **kwargs):
        """Initailize instance."""
        super().__init__(**kwargs)

        ml_framework_in_use = get_ml_framework_in_use()
        if ml_framework_in_use != MLFramework.PYTORCH:
            raise NotImplementedError(
                "FedBalancer is currently only implemented in PyTorch;"
            )

        self.round = 0

        # CONFIG CHANGES FOR ASYNCFL WITH OORT
        try:
            self.c = kwargs["c"]
        except KeyError:
            raise KeyError("c (concurrency level) is not specified in config")

        try:
            self.agg_goal = kwargs["aggGoal"]
        except KeyError:
            raise KeyError("aggGoal is not specified in config")
        
        try:
            self.eval_goal_factor = kwargs["evalGoalFactor"]
        except KeyError:
            raise KeyError("evalGoalFactor is not specified in config. It is the decimal multiplicative factor wrt agg goal for eval")
        
        try:
            self.round_nudge_type = kwargs["roundNudgeType"]
        except KeyError:
            raise KeyError("roundNudgeType is not specified in config. It is last_train or last_eval based on the selector nudging critera")

        try:
            self.select_type = kwargs["selectType"]
        except KeyError:
            raise KeyError("selectType is not specified in config. Can be default, "
                           "fastest, or maxSamples")

        if self.agg_goal < 0:
            self.agg_goal = 1

        # With Oort, we select 1.3 * k ends and wait until k ends to
        # complete at a round
        self.overcommitment = 1.3
        self.num_of_ends = int(self.agg_goal * self.overcommitment)

        self.exploration_factor = 0.9
        self.exploration_factor_decay = 0.98
        self.min_exploration_factor = 0.2

        self.exploitation_util_history = []

        # Assuming a max round duration of 99999 seconds (~1.2 days)
        self.round_preferred_duration = timedelta(seconds=99999)
        self.round_threshold = 30
        self.pacer_delta = 5
        self.pacer_step = 20

        self.blocklist_threshold = -1

        self.alpha = 2

        # #### CHANGES BASED OFF FEDBUFF FOR ASYNCFL
        # Tracking selected ends to ensure selection correctness for
        # each round (a trainer can participate only once per round).
        self.all_selected = dict()
        self.selected_ends = dict()

        # Tracks weight updates received from trainers and makes them
        # available to select again
        self.ordered_updates_recv_ends = list()
        
        # Tracks eval updates received from trainers and makes them
        # available to select again
        self.trainer_eval_recv_ends = list()
        self.curr_round_eval_slots_left = int(self.eval_goal_factor * self.agg_goal)

        # Tracks timeouted trainers and number of times it happened to
        # a trainer
        self.track_trainer_timeouts = dict()

        # Tracks trainers that were selected but left training in
        # between
        self.track_selected_trainers_which_left = dict()
        self.check_three_state_avl= True

    def select(
        self,
        ends: dict[str, End],
        channel_props: dict[str, Scalar],
        trainer_unavail_list: list,
        task_to_perform: str,
    ) -> SelectorReturnType:
        """Return k number of ends from the given ends.

        NOTE: It incorporates the same send/recv mechanism from
        fedbuff. [From fedbuff selector]: Select ends from the given
        ends to meet concurrency level. This select method chooses
        ends differently depending on what state a channel is in. In
        'send' state, it chooses ends that are not in
        self.selected_ends. In 'recv' state, it chooses all ends from
        self.selected_ends. Essentially, if an end is in
        self.selected_ends, it means that we sent some message already
        to that end. For such an end, we exclude it from send and
        include it for recv in return.
        """
        logger.debug("calling async oort select")

        # TODO: (DG) Update later, currently setting eval concurrency
        # to be twice of training concurrency
        if task_to_perform == "train":
            concurrency = min(len(ends), self.c)
        elif task_to_perform == "eval":
            # Select ends for eval only if eval-goal is set in 3-state
            # availability tracking. Else, don't select any eval ends-
            # this would be for 2-state tracking.
            if self.eval_goal_factor > 0.0:
                # this is set to maximum possible concurrency for
                # eval. It will be adjusted later based on eval tasks
                # already sent and received for the round.
                concurrency = min(len(ends), self.c + self.curr_round_eval_slots_left)
            else:
                concurrency = 0
        logger.info(f"Task: {task_to_perform}, len(ends): {len(ends)}, c: {self.c}, chosen concurrency: {concurrency}")

        if concurrency == 0:
            logger.debug("ends is empty")
            return {}

        if KEY_CH_STATE not in channel_props:
            raise KeyError(f"channel property doesn't have {KEY_CH_STATE}")

        self.requester = channel_props[KEY_CH_SELECT_REQUESTER]
        if self.requester not in self.selected_ends:
            self.selected_ends[self.requester] = set()

        # TODO: (DG) Is explicit round tracking required here? round =
        # channel_props["round"] if "round" in channel_props else 0
        # logger.debug(f"let's select {num_of_ends} ends for new round
        # {round}")

        # default, availability unaware way of using ends
        eligible_ends = ends

        # Make a filter of unavailable ends, update eligible_ends
        # given trainer_unavail_list
        if trainer_unavail_list != [] and trainer_unavail_list is not None:
            # Updating passed ends and filtering out unavailable ones
            # before passing
            eligible_ends = {
                end_id: end
                for end_id, end in ends.items()
                if end_id not in trainer_unavail_list
                }
            logger.debug(f"Fedbuff select got non-empty trainer_unavail_list, "
                         f"populated eligible_ends: {eligible_ends}")

        results = {}
        if channel_props[KEY_CH_STATE] == VAL_CH_STATE_SEND:
            results = self._handle_send_state(
                eligible_ends,
                concurrency,
                channel_props,
                trainer_unavail_list,
                task_to_perform
                )

        elif channel_props[KEY_CH_STATE] == VAL_CH_STATE_RECV:
            # TODO: (DG) See if eligible_ends should be passed here
            # too in place of ends
            results = self._handle_recv_state(ends, concurrency)

        else:
            state = channel_props[KEY_CH_STATE]
            raise ValueError(f"unkown channel state: {state}")

        logger.debug(
            f"requester: {self.requester}, selected ends: {self.selected_ends}"
        )
        logger.debug(
            f"channel state: {channel_props[KEY_CH_STATE]}, results: {results}"
        )

        return results

    def cutoff_util(
            self,
            sorted_utility_list: list[tuple[str, float]],
            num_of_ends: int,
            ) -> float:
        """Return a cutoff utility based on Oort."""
        if not sorted_utility_list:
            logger.debug("Got empty utility_list, returning 999999.0")
            return 999999.0

        index = int(num_of_ends * (1 - self.exploration_factor)) - 1
        index = max(0, min(index, len(sorted_utility_list) - 1))

        return 0.95 * sorted_utility_list[index][PROP_UTILITY]

    def sample_by_util(
        self,
        cutoff_utility: float,
        utility_list: list[dict[str, Scalar]],
        num_of_ends: int,
    ) -> list[str]:
        """Sample num_of_ends clients by utility."""

        over_cutoff_utility_end_ids = []
        over_cutoff_utility_probs = []
        over_cutoff_utility_sum = 0

        under_cutoff_utility_list = []

        # Divide ends on whether its utility exceeds cutoff_loss or
        # not
        for utility_pair in utility_list:
            if utility_pair[PROP_UTILITY] >= cutoff_utility:
                over_cutoff_utility_end_ids.append(utility_pair[PROP_END_ID])
                over_cutoff_utility_probs.append(utility_pair[PROP_UTILITY])
                over_cutoff_utility_sum += utility_pair[PROP_UTILITY]
            else:
                under_cutoff_utility_list.append(utility_pair)

        # Select clients on the probability based on the utility
        # divided by the utility sum
        for prob_idx in range(len(over_cutoff_utility_probs)):
            over_cutoff_utility_probs[prob_idx] /= over_cutoff_utility_sum

        selected_ends = np.random.choice(
            over_cutoff_utility_end_ids,
            size=min(len(over_cutoff_utility_end_ids), num_of_ends),
            replace=False,
            p=over_cutoff_utility_probs,
        )

        return selected_ends

    def sample_by_speed(
        self, unexplored_end_ids: list[str], num_of_ends: int
    ) -> list[str]:
        """Sample num_of_ends clients by speed."""

        # Oort paper prioritizes unexplored ends with faster system
        # speed We initially implement to perform random here
        return np.random.choice(unexplored_end_ids, size=num_of_ends, replace=False)

    def pacer(self) -> None:
        """
        Controls round preferred duration based on the exploited
        statistical utility.
        """

        if (
            len(self.exploitation_util_history) >= 2 * self.pacer_step
            and self.round % self.pacer_step == 0
        ):
            last_pacer_step_util = sum(
                self.exploitation_util_history[-2 * self.pacer_step: -self.pacer_step]
            )
            curr_pacer_step_util = sum(
                self.exploitation_util_history[-self.pacer_step:]
            )

            # increases round threshold when recently exploited
            # statistical utility decreases
            if last_pacer_step_util > curr_pacer_step_util:
                self.round_threshold = min(
                    100.0, self.round_threshold + self.pacer_delta
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
        """
        Calculate number of ends to select for exploration and
        exploitation; Add 1 to exploration_len to avoid not exploring
        0 ends while unexplored ends exist.
        """

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
        """
        Make a list of tuple (end_id, end_utility) as an utility_list
        As unexplored ends that are not selected before do not have
        utility value, collect them separately with unexplored_end_ids
        list
        """

        utility_list = []
        unexplored_end_ids = []

        for end_id in ends.keys():
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
        """
        Calculate round preferred duration based on round_threshold
        and end_round_duration of trainers. round_threshold is
        controlled by pacer.
        """
        logger.debug(f"calculate_round_pref_duration ends.keys(): {ends.keys()}")
        if self.round_threshold < 100.0:
            sorted_round_duration = []
            for end_id in ends.keys():
                end_round_duration = ends[end_id].get_property(PROP_ROUND_DURATION)
                logger.debug(
                    f"end_id: {end_id}, end_round_duration: {end_round_duration}"
                )
                if end_round_duration is not None:
                    sorted_round_duration.append(end_round_duration)
                elif end_round_duration is None:
                    # (DG) HACK. Comes here if the trainer
                    # participates in eval, so technically doesnt have
                    # a round duration. Can set it to 60 seconds since
                    # that is the max round duration for training.
                    # TODO: Eval is half of round duration so can use
                    # that information to set correct round duration.
                    # But it might break other code, so leaving it for
                    # later.
                    sorted_round_duration.append(timedelta(seconds=60))
            logger.info(
                f"after for loop, sorted_round_duration: {sorted_round_duration}"
            )
            round_preferred_duration = timedelta(
                seconds=sorted_round_duration[
                    min(int(len(sorted_round_duration) * self.round_threshold / 100.0),
                        len(sorted_round_duration) - 1,
                        )
                        ].total_seconds()
                    )
        else:
            # Assuming a max round duration of 99999 seconds (~1.2 days)
            round_preferred_duration = timedelta(seconds=99999)
        
        logger.debug(
            f"returning round_preferred_duration: {round_preferred_duration}")
        return round_preferred_duration

    def calculate_temporal_uncertainty_of_trainer(
        self, ends: dict[str, End], end_id: str, round: int
    ) -> float:
        """
        Calculate temproal uncertainty term based on the end's last
        selected round.
        """
        
        # OPTION 1: nudge temporal rank of trainer based on last
        # trained round. This value might be stale based on when the
        # trainer was last selected for training. It represents the
        # original OORT style.
        
        # OPTION 2: nudge temporal rank of trainer based on last
        # evaluated round. This means that the value will be updated
        # for each train or eval task given to the trainer. It helps
        # track the utility to a fresher extent.
        
        if self.round_nudge_type == "last_train":
            end_last_selected_round = ends[end_id].get_property(PROP_LAST_SELECTED_ROUND)
        elif self.round_nudge_type == "last_eval":
            end_last_selected_round = ends[end_id].get_property(PROP_LAST_EVAL_ROUND)
        
        logger.debug(f"using round_nudge_type: {self.round_nudge_type} for end_id: {end_id}, end_last_selected_round: {end_last_selected_round}")
        
        # TODO: (DG) Enable a flag to use or not use temporal
        # uncertainty as 0 based on our solution. We might want to
        # disable it in the final selection process in FeLiX.
        if end_last_selected_round is None:
            trainer_temporal_uncertainty = 0
        else:
            trainer_temporal_uncertainty = math.sqrt(0.1 * math.log(round) / end_last_selected_round)
        return trainer_temporal_uncertainty

    def calculate_global_system_utility_of_trainer(
        self, ends: dict[str, End], end_id: str
    ) -> float:
        """
        Calculate global system utility based on the end's round
        duration.
        """

        end_round_duration = ends[end_id].get_property(PROP_ROUND_DURATION)
        
        # In normal training, the util of trainer is 1 if it is faster
        # than preferred round duration. This is a multiplier to the
        # trainer utility. Thus, if the trainer is slower than
        # preferred round duration, the multiplier is (0, 1) which
        # means that the utility of the trainer decreases.
        
        # For eval-enabled training, it is possible that the trainer
        # hasn't trained yet but has only pushed an eval update. For
        # these trainers, we retain the multiplicative factor as 1 so
        # as to incentivise them to be picked whenever available to
        # train.
        
        # TODO:(DG) Make it configurable via flag for Felix selection
        # policy.
        if end_round_duration is None:
            return 1

        if end_round_duration <= self.round_preferred_duration:
            return 1
        else:
            # Get both into datetime seconds before division
            return math.pow(
                self.round_preferred_duration.total_seconds() /
                end_round_duration.total_seconds(),
                self.alpha,
            )

    def save_exploited_utility_history(
        self, ends: dict[str, End], exploit_end_ids: list[str]
    ) -> None:
        """
        Save the history of exploited utility at this round for pacer.
        """

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
            self,
            ends: dict[str, End],
            candidates: list[str]
            ) -> None:
        """Increment the round selected count on selected ends."""

        # TODO: (DG): Using self.requester here since it is a
        # dict->list mapping now. Check
        for end_id in candidates:
            if ends[end_id].get_property(PROP_SELECTED_COUNT) is None:
                ends[end_id].set_property(PROP_SELECTED_COUNT, 1)
            else:
                ends[end_id].set_property(
                    PROP_SELECTED_COUNT,
                    ends[end_id].get_property(PROP_SELECTED_COUNT) + 1,
                )

    def select_random(self, ends: dict[str, End], num_of_ends: int) -> dict[str, None]:
        """Randomly select num_of_ends ends."""
        # TODO: (DG) Check. Changed from self.selected_ends to local
        # selected_ends.

        selected_random_ends = set(random.sample(list(ends), num_of_ends))
        logger.debug(f"selected_random_ends: {selected_random_ends}")

        return {key: None for key in selected_random_ends}

    def calculate_total_utility(
        self, utility_list: list[tuple[str, float]], ends: dict[str, End], round: int
    ) -> list[tuple[str, float]]:
        """
        Calculate the total utility value of trainers with applying
        temporal uncertainty and global system utility, based on the
        Oort algorithm.
        """
        if utility_list == []:
            logger.debug("Got empty utility_list in calculate_total_utility. "
                         "Returning empty")
            return []

        # Calculate preferred round duration TODO: (DG) Since we
        # return at the top, we don't have anything to do here?
        self.round_preferred_duration = self.calculate_round_preferred_duration(ends)

        # Sort the utility list by the utility value placed at the
        # index 1 of each tuple
        utility_list = sorted(utility_list, key=lambda x: x[PROP_UTILITY])

        # Calculate the clip value that caps utility value of a client
        # to no more than an upper bound (95% value in utility
        # distributions) NOTE: In cases of new clients added to the
        # system, there could be cases where the utility_list is
        # empty. In that case, we set clip_value to 100. TODO: (DG)
        # Verify that this would be okay.
        clip_value = utility_list[
            min(int(len(utility_list) * 0.95), len(utility_list) - 1)
        ][PROP_UTILITY]

        # Calculate the final utility value of a trainer by adding the
        # temporal uncertainty and multiplying the global system
        # utility
        for utility_idx in range(len(utility_list)):
            curr_end_utility = utility_list[utility_idx][PROP_UTILITY]
            curr_end_id = utility_list[utility_idx][PROP_END_ID]

            # Clip the utility value
            utility_list[utility_idx][PROP_UTILITY] = min(
                utility_list[utility_idx][PROP_UTILITY], clip_value
            )

            # Add temproal uncertainty term
            temporal_uncertainty = self.calculate_temporal_uncertainty_of_trainer(
                ends, curr_end_id, round
            )
            curr_end_utility += temporal_uncertainty
            logger.info(f"end_id: {curr_end_id}, adding temporal_uncertainty: {temporal_uncertainty} to get curr_end_utility: {curr_end_utility}")

            # Multiply global system utility
            global_system_utility = self.calculate_global_system_utility_of_trainer(
                ends, curr_end_id
            )
            curr_end_utility *= global_system_utility
            logger.info(f"end_id: {curr_end_id}, curr_end_utility: {curr_end_utility} after multiplying global_system_utility: {global_system_utility}")

            utility_list[utility_idx][PROP_UTILITY] = curr_end_utility

        # Sort the utility list again, with the updated utility value
        utility_list = sorted(utility_list, key=lambda x: x[PROP_UTILITY])

        return utility_list

    # #### CHANGES BASED OFF FEDBUFF FOR ASYNCFL
    def _cleanup_recvd_ends(self, ends: dict[str, End]):
        """Clean up ends whose a message was received, from selected
        ends.

        Note: It sets the end state to none which makes it eligible to
        be sampled again. This can cause problems if sampled in the
        same round. Thus, for aggregator, the _cleanup_recvd_ends
        should be triggered only after aggregation of weights succeeds
        on meeting agg_goal."""
        logger.debug(f"clean up recvd ends. selected_ends: {self.selected_ends}, ends: {ends.keys()}")

        selected_ends = self.selected_ends[self.requester]
        logger.debug(
            f"self.requester: {self.requester} and selected_ends: "
            f"{selected_ends} before processing"
        )

        num_ends_to_remove = min(
            len(self.ordered_updates_recv_ends), 
            self.agg_goal
            )
        if num_ends_to_remove != 0:
            ends_to_remove = self.ordered_updates_recv_ends[:num_ends_to_remove]
            logger.debug(f"Will remove these ends from "
                         f"ordered_updates_recv_ends: {ends_to_remove}"
                         f" and selected_ends and all_selected")

            # removing the first agg-goal number of ends to free them
            # to participate in the next round
            self.ordered_updates_recv_ends = (
                self.ordered_updates_recv_ends[num_ends_to_remove:]
            )
            logger.debug(f"self.ordered_updates_recv_ends after removing first "
                         f"num_ends_to_remove: {num_ends_to_remove} "
                         f"elements: {self.ordered_updates_recv_ends}")
            
            
            logger.info(f"Ends to remove based on trainer updates received: {ends_to_remove}")
            
            # Adding trainer_eval_recv_ends to accoount for trainers
            # that have finished eval updates. These trainers also
            # need to be freed up to participate in the next round.
            logger.info(f"Ends to remove based on eval updates received: {self.trainer_eval_recv_ends}")
            ends_to_remove = ends_to_remove + self.trainer_eval_recv_ends
            
            logger.info(f"All ends to remove (train + eval): {ends_to_remove}")
            
            self.trainer_eval_recv_ends = []
            logger.debug(f"Cleared trainer_eval_recv_ends: {self.trainer_eval_recv_ends}")
            
            self.curr_round_eval_slots_left = int(self.eval_goal_factor * self.agg_goal)
            logger.info(f"Reset curr_round_eval_slots_left: {self.curr_round_eval_slots_left}")

            for end_id in ends_to_remove:
                if end_id not in ends:
                    # something happened to end of end_id (e.g.,
                    # connection loss) let's remove it from
                    # selected_ends
                    logger.debug(f"no end id {end_id} in ends, removing "
                                 f"from selected_ends and all_selected")
                    # NOTE: it is not a guarantee that selected_ends
                    # will still contain the end_id. Thats because it
                    # might have got disconnected/ rejoined in the
                    # middle of a round
                    if end_id in selected_ends:
                        selected_ends.remove(end_id)
                        logger.debug(f"No end id {end_id} in ends, removed from "
                                     f"selected_ends: "
                                     f"{selected_ends}"
                                     )
                    if end_id in self.all_selected:
                        del self.all_selected[end_id]
                        logger.debug(f"No end id {end_id} in ends, removed from "
                                     f"self.all_selected: {self.all_selected}"
                                     )
                else:
                    state = ends[end_id].get_property(KEY_END_STATE)
                    logger.debug(
                        f"End_id {end_id} found in selected_ends in state: {state}, "
                        f"selected_ends: {selected_ends} and self.all_selected: "
                        f"{self.all_selected}"
                    )
                    if state == VAL_END_STATE_RECVD:
                        ends[end_id].set_property(KEY_END_STATE, VAL_END_STATE_NONE)
                        logger.debug(f"Setting {end_id} state to {VAL_END_STATE_NONE}, "
                                     f"and"
                                     f" removing from selected_ends and all_selected")
                        if end_id in selected_ends:
                            selected_ends.remove(end_id)
                            logger.debug(f"FOUND end id {end_id} in state: {state}.. "
                                         f"removed from "
                                         f"selected_ends: {selected_ends}"
                                         )
                        if end_id in self.all_selected:
                            del self.all_selected[end_id]
                            logger.debug(f"FOUND end id {end_id} in state: {state}.. "
                                         f"removed from "
                                         f"self.all_selected: "
                                         f"{self.all_selected}"
                                         )
                    elif state == VAL_END_STATE_NONE:
                        # TODO: (DG) Recheck if it needs to be deleted
                        # from here as well. Is the failure scenario
                        # being handled correctly if the trainer
                        # contributes, fails and then comes back
                        # within the same round.
                        logger.debug(f"Found end {end_id} in state {VAL_END_STATE_NONE}. Might have "
                                     f"left/rejoined. Need to remove it from "
                                     f"selected_ends and self.all_selected "
                                     f"if it was selected")
                        if end_id in selected_ends:
                            selected_ends.remove(end_id)
                            logger.debug(f"FOUND end id {end_id} in state: {state}.. "
                                         f"removed from "
                                         f"selected_ends: {selected_ends}"
                                         )
                        if end_id in self.all_selected:
                            del self.all_selected[end_id]
                            logger.debug(f"FOUND end id {end_id} in state: {state}.. "
                                         f"removed from "
                                         f"self.all_selected: "
                                         f"{self.all_selected} too"
                                         )
                    else:
                        logger.debug(f"FOUND end id {end_id} in state: {state}. "
                                     f"Not doing anything")
        else:
            logger.debug("No ends to remove so far")

    def _cleanup_removed_ends(self, end_id):
        logger.debug(f"Going to cleanup selector state for "
                     f"end_id {end_id} since it has left the channel")
        if (
            end_id in self.all_selected
            ) and (
                end_id not in self.ordered_updates_recv_ends
                ):
            # remove end from all_selected if we havent got an update
            # from it yet. It would have flushed the agg-weights after
            # initiating channel.leave().
            logger.debug(f"Removing end_id {end_id} from all_selected"
                         f" since no update received before it left the channel.")
            selected_ends = self.selected_ends[self.requester]
            if end_id in selected_ends:
                selected_ends.remove(end_id)
                logger.debug(f"Also removing end_id {end_id} from selected_ends")
                self.selected_ends[self.requester] = selected_ends

            # Track trainers that were sent weights but dropped off
            # before sending back an update
            if end_id in self.track_selected_trainers_which_left:
                self.track_selected_trainers_which_left[end_id] += 1
            else:
                self.track_selected_trainers_which_left[end_id] = 1

            total_trainers_dropped_off = 0
            for k, v in self.track_selected_trainers_which_left.items():
                total_trainers_dropped_off += v

            logger.info(f"Trainer: {end_id} with count "
                        f"{self.track_selected_trainers_which_left[end_id]}, left "
                        f"before returning update. "
                        f"total_trainers_dropped_off: {total_trainers_dropped_off} "
                        f"self.track_selected_trainers_which_left: "
                        f"{self.track_selected_trainers_which_left}")
            if end_id in self.all_selected.keys():
                del self.all_selected[end_id]
        elif (
            end_id in self.all_selected
         ) and (
             end_id in self.ordered_updates_recv_ends
             ):
            # Dont remove it if it was in all_selected and we have got
            # an update from it before it did channel.leave(). It has
            # completed its participation for this round.
            logger.debug(f"Update was alreacy received from {end_id} before it left "
                         f"the channel. Not deleting from all_ends now.")
        else:
            logger.warning(f"End_id {end_id} remove check from all_selected failed. "
                        f"Need to check")

    # Invoked when selection mode is default i.e. of oort which trades
    # off exploitation/exploration and speed/stat_utility
    def _select_candidates_using_default(
            self,
            cutoff_utility,
            utility_list,
            exploitation_len,
            exploration_len,
            unexplored_end_ids,
            ):
        logger.debug("Asyncoort selection using default (tradeoff)")
        logger.debug(f"Invoking sample_by_util() with cutoff_utility: {cutoff_utility}"
                     f", utility_list: {utility_list}, exploitation_len: "
                     f"{exploitation_len}")
        exploit_end_ids = self.sample_by_util(
            cutoff_utility, utility_list, exploitation_len
        )
        logger.debug(f"exploit_end_ids: {exploit_end_ids}")

        # sample exploration_len of unexplored clients
        explore_end_ids = []
        if self.exploration_factor > 0.0 and len(unexplored_end_ids) > 0:
            logger.debug(f"Invoking sample_by_speed(): with unexplored_end_ids: "
                         f"{unexplored_end_ids}, exploration_len: {exploration_len}")
            explore_end_ids = self.sample_by_speed(unexplored_end_ids, exploration_len)
        logger.debug(f"explore_end_ids: {explore_end_ids}")

        candidates = [*explore_end_ids, *exploit_end_ids]

        return candidates, exploit_end_ids

    # Invoked when selection mode is maxSamples i.e. select clients
    # with largest local datasets
    def _select_candidates_maxSamples(
            self,
            ends: dict[str, End],
            num_of_ends: int,
            ) -> tuple[list[str], list[str]]:
        logger.debug("Asyncoort selection using maxSamples")
        logger.debug(f"Will select num_ends: {num_of_ends} "
                     f"from ends of length: {len(ends)}")

        # get the end properties
        end_id_to_samples = {}
        for key, val in ends.items():
            # if the PROP_DATASET_SIZE is None, it means the trainer
            # hasnt trained even once till now. So we set it to 99999
            # to prioritize it to get picked up atleast once.
            end_num_samples = val.get_property(PROP_DATASET_SIZE)
            if end_num_samples is None:
                end_num_samples = 99999
                logger.debug(f"Num_samples for end_id: {key} was None, "
                             f"set to {end_num_samples} to incentivise getting picked")

            end_id_to_samples[key] = end_num_samples

        # sort it in descending order of number of samples
        sorted_end_ids = list(
            dict(
                sorted(
                    end_id_to_samples.items(),
                    key=lambda item: item[1],
                    reverse=True)
                ).keys()
            )

        # currently returning blank exploit_end_ids TODO: (DG) check
        # later about why it is needed
        exploit_end_ids = []

        # pick first k elements as candidates and return
        candidates = sorted_end_ids[:num_of_ends]
        logger.debug(f"Selected candidates being returned: {candidates}")

        return candidates, exploit_end_ids

    # Invoked when selection mode is fastest i.e. select fastest
    # clients
    def _select_candidates_fastest(
            self,
            ends: dict[str, End],
            num_of_ends: int,
            ) -> tuple[list[str], list[str]]:
        logger.debug("Asyncoort selection using fastestClients")
        logger.debug(f"Will select num_ends: {num_of_ends} "
                     f"from ends of length: {len(ends)}")

        # get the end properties
        end_id_to_round_durations = {}
        for key, val in ends.items():
            # if the PROP_ROUND_DURATION is None, it means the trainer
            # hasnt trained even once till now. So we set it to
            # 00:00:00.000000 (upto microseconds) to prioritize it to
            # get picked up atleast once.
            round_duration = val.get_property(PROP_ROUND_DURATION)
            if round_duration is None:
                round_duration = timedelta(
                    hours=0,
                    minutes=0,
                    seconds=0,
                    microseconds=0
                    )
                logger.debug(f"Round_duration for end_id: {key} was None, "
                             f"set to {round_duration} to incentivise getting picked")

            end_id_to_round_durations[key] = round_duration

        # sort it in ascending order of durations
        sorted_end_ids = list(
            dict(
                sorted(
                    end_id_to_round_durations.items(),
                    key=lambda item: item[1],
                    reverse=False)
                ).keys()
            )

        # currently returning blank exploit_end_ids TODO: (DG) check
        # later about why it is needed
        exploit_end_ids = []

        # pick first k elements as candidates and return
        candidates = sorted_end_ids[:num_of_ends]
        logger.debug(f"Selected candidates being returned: {candidates}")

        return candidates, exploit_end_ids
    
    # Invoked when selection mode is fair-share i.e. select clients
    # such that all clients participate almost equally
    def _select_candidates_fairShare(
            self,
            ends: dict[str, End],
            num_of_ends: int,
            ) -> tuple[list[str], list[str]]:
        logger.debug("Asyncoort selection using fairShare")
        logger.debug(f"Will select num_ends: {num_of_ends} "
                     f"from ends of length: {len(ends)}")

        # get the end properties
        end_id_to_update_count = {}
        for key, val in ends.items():
            # if the PROP_UPDATE_COUNT is None, it means the trainer
            # hasnt trained even once till now. So we set it to 0 to
            # prioritize it to get picked up atleast once.
            update_count = val.get_property(PROP_UPDATE_COUNT)
            if update_count is None:
                update_count = 0
                logger.debug(f"Update_count for end_id: {key} was None, "
                             f"set to {update_count} to incentivise getting picked")

            end_id_to_update_count[key] = update_count

        # sort it in ascending order of durations
        sorted_end_ids = list(
            dict(
                sorted(
                    end_id_to_update_count.items(),
                    key=lambda item: item[1],
                    reverse=False)
                ).keys()
            )

        # currently returning blank exploit_end_ids TODO: (DG) check
        # later about why it is needed
        exploit_end_ids = []

        # pick first k elements as candidates and return
        candidates = sorted_end_ids[:num_of_ends]
        logger.debug(f"Selected candidates being returned: {candidates}")

        return candidates, exploit_end_ids
    
    # Invoked when selection mode is prioritiseUnavail i.e. select and
    # prioritize clients that have been unavailable for long durations
    # of the training time
    def _select_candidates_prioritiseUnavail(
            self,
            ends: dict[str, End],
            num_of_ends: int,
            ) -> tuple[list[str], list[str]]:
        logger.debug("Asyncoort selection using prioritiseUnavail")
        logger.debug(f"Will select num_ends: {num_of_ends} "
                     f"from ends of length: {len(ends)}")

        # get the end properties
        end_id_to_unavail_durations = {}
        for key, val in ends.items():
            # if the PROP_TOTAL_UNAVAIL_DURATION is None, it means the
            # trainer hasnt failed even once till now. So we set it to
            # 00:00:00.000000 (upto microseconds) to allow it to get
            # picked whenever there are no other unavailable clients
            # to pick.
            unavail_duration = val.get_property(PROP_TOTAL_UNAVAIL_DURATION)
            if unavail_duration is None:
                unavail_duration = timedelta(
                    hours=0,
                    minutes=0,
                    seconds=0,
                    microseconds=0
                    )
                logger.debug(f"Unavail_duration for end_id: {key} was None, "
                             f"set to {unavail_duration} to allow  getting picked")
            else:
                logger.debug(f"Unavail_duration for end_id: {key} was set "
                             f"to {unavail_duration}")

            end_id_to_unavail_durations[key] = unavail_duration

        # sort it in ascending order of durations
        sorted_end_ids = list(
            dict(
                sorted(
                    end_id_to_unavail_durations.items(),
                    key=lambda item: item[1],
                    reverse=True)
                ).keys()
            )

        # currently returning blank exploit_end_ids TODO: (DG) check
        # later about why it is needed
        exploit_end_ids = []

        # pick first k elements as candidates and return
        candidates = sorted_end_ids[:num_of_ends]
        logger.debug(f"Selected candidates being returned: {candidates}")

        return candidates, exploit_end_ids

    def _handle_send_state(
        self,
        ends: dict[str, End],
        concurrency: int,
        channel_props: dict[str, Scalar],
        trainer_unavail_list: list = None,
        task_to_perform: str = "train",
    ) -> SelectorReturnType:
        selected_ends = self.selected_ends[self.requester]

        # Check for invalid selections and remove them
        for end_id in list(selected_ends):
            if end_id not in ends:
                # something happened to end of end_id (e.g.,
                # connection loss) let's remove it from selected_ends
                # so that you can fill that spot with another trainer
                logger.debug(f"Removing invalid prior selection! "
                             f"No end id {end_id} in ends, "
                             f"removing from selected_ends. "
                             f"NOT from all_selected right now "
                             f"cause aggregation for that "
                             f"round hasnt completed yet")
                selected_ends.remove(end_id)
                # NOTE: Not removing end_id from all_selected since it
                # might have already participated in the same round
                # (if it is still in all_ends)

        logger.debug(f"Current selected_ends: {selected_ends}")

        extra = max(0, concurrency - len(selected_ends))

        logger.debug(f"c: {concurrency}, "
                     f"len(selected_ends): {len(selected_ends)}, extra: {extra}, selected_ends: {selected_ends}," f"len(ends): {len(ends)}")
        candidates = []

        # ### From Oort selector
        # num_of_ends = min(len(ends), self.num_of_ends) if
        # num_of_ends == 0: logger.debug("ends is empty") return {}
        if extra == 0:
            logger.debug(f"extra: {extra}, nothing to select")
            return {}

        round = channel_props["round"] if "round" in channel_props else 0
        logger.debug(f"let's select {extra} ends for round {round}")
        
        if round % 100 == 0:
            # Log to info level the property of LAST_EVAL_ROUND for
            # all the ends
            for end_id, end in ends.items():
                logger.info(f"End ID: {end_id}, Last Eval Round: {end.get_property(PROP_LAST_EVAL_ROUND)}, Statistical Utility: {end.get_property(PROP_STAT_UTILITY)}")

        # NOTE: (DG) Assuming that shuffled_end_ids is not needed

        # Invalidate previous all_selected entry if you don't get an
        # update in UPDATE_TIMEOUT_WAIT_S. The client might have
        # dropped the message with transient unavailability.

        # TODO: (DG) Check if it is still needed after
        # cleanup_remove_end() method
        curr_all_selected_ends = list(self.all_selected.keys())
        for end in curr_all_selected_ends:
            current_time_s = time.time()
            if end in self.all_selected.keys():
                # Check again to avoid possible case of race condition
                # when all_selected has been updated from another
                # thread
                trainer_weight_send_timestamp_s = self.all_selected[end]
                if (trainer_weight_send_timestamp_s < (
                        current_time_s - SEND_TIMEOUT_WAIT_S)
                    ) and (
                        end not in self.ordered_updates_recv_ends):
                    # trainer hasn't returned with an update in
                    # SEND_TIMEOUT_WAIT_S delete it from
                    # self.all_selected so that it is eligible to be
                    # sampled again
                    logger.debug(f"Removing end {end} from self.all_selected "
                                 f"since havent "
                                 f"got its update in {SEND_TIMEOUT_WAIT_S}. "
                                 f"Last weight send timestamp was: {trainer_weight_send_timestamp_s}")

                    # Tracking timeouts and time spend waiting TODO:
                    # (DG) Check if it is okay to have it triggered
                    # for oracular too? TODO: (DG) pass
                    # timeout_duration as a flag from config, also
                    # pass enable/disable it?
                    if end in self.track_trainer_timeouts:
                        self.track_trainer_timeouts[end] += 1
                    else:
                        self.track_trainer_timeouts[end] = 1

                    # Capture total time spent in timeouts
                    num_of_timeouts_occured = 0
                    for k, v in self.track_trainer_timeouts.items():
                        num_of_timeouts_occured += v

                    total_time_spent_timeouts_s = (
                        num_of_timeouts_occured * SEND_TIMEOUT_WAIT_S
                    )

                    logger.info(f"Timeout for trainer: {end} with count "
                                f"{self.track_trainer_timeouts[end]}. "
                                f"num_of_timeouts_occured : "
                                f"{num_of_timeouts_occured}, "
                                f"total_time_spent_timeouts_s: "
                                f"{total_time_spent_timeouts_s}, "
                                f"Timeout frequency: {self.track_trainer_timeouts}")

                    # delete the end from self.all_selected
                    if end in self.all_selected.keys():
                        del self.all_selected[end]

        # Run pacer that controls round_threshold TODO: (DG) This
        # might need to be changed since round is not complete.
        # Another value can be set from the top_aggregator and be used
        # by OORT_ASYNC selector
        self.pacer()

        # TODO: (DG) Add code to allow only those ends (not in
        # all_selected) to be passed. filtered_ends consists of ends
        # that are not in all_selected and can be picked in this round
        # i.e. avoids repeating a trainer in the same round
        filtered_ends = dict()
        
        # track the ends that are eligible vs ineligible based on
        # their state
        count_avl_train = 0
        count_avl_eval = 0
        count_ineligible = 0
        
        # Check the eligible set first. Out of the ends, how many are not in
        # all_selected? Only those are eligible since the rest have weights
        # already sent to them for either train/eval task.
        count_eligible_set_to_check = [end for end in ends if end not in self.all_selected]
        logger.debug(f"Before creating filtered_ends. count_eligible_set_to_check: {len(count_eligible_set_to_check)} from total {len(ends)} ends.")
        
        for end_id in ends:
            if end_id not in self.all_selected.keys(): 
                logger.debug(f"Creating filtered ends. Checking end id {end_id}, avl_state = {ends[end_id].get_property(PROP_AVL_STATE)}")
                
                # If check_three_state_avl=False, no more checks,
                # directly add end to filtered_ends
                
                # If check_three_state_avl=True, filtered ends needs
                # to be populated based on the following conditions:
                # For task_to_perform=train, eligible ends are in
                # states {avl_train, None} For task_to_perform=eval,
                # eligible ends are in states {avl_train, avl_eval
                # None}
                    
                curr_end_id_avl_state = ends[end_id].get_property(PROP_AVL_STATE)
                # Even if client notify is not enabled, this logic
                # would work since curr_end_id_avl_state = None 
                if task_to_perform == "train" and (
                    curr_end_id_avl_state in (
                        TrainerAvailState.AVL_TRAIN.value,
                        None
                        )
                    ):
                    filtered_ends[end_id] = ends[end_id]
                    logger.debug(f"Adding end {end_id} to filtered ends. Three_state_avl_check is True, task_to_perform: {task_to_perform} in state: {ends[end_id].get_property(PROP_AVL_STATE)}")
                    count_avl_train += 1
                elif self.check_three_state_avl and task_to_perform == "eval" and (
                    curr_end_id_avl_state in (
                        TrainerAvailState.AVL_EVAL.value, 
                        TrainerAvailState.AVL_TRAIN.value,
                        None
                        )
                    ):
                    # NOTE: Picking from avl_train as well since it
                    # gave us better results. But need to set to low
                    # eval factor values because it might pick from
                    # AVL_TRAIN as well and slow down training.
                    filtered_ends[end_id] = ends[end_id]
                    logger.debug(f"Adding end {end_id} to filtered ends. Three_state_avl_check, task_to_perform: {task_to_perform} in state: {ends[end_id].get_property(PROP_AVL_STATE)}")
                    count_avl_eval += 1
                else:
                    logger.debug(f"Not adding end {end_id} to filtered ends since required for task{task_to_perform}, "
                                     f"but was in state {curr_end_id_avl_state}. Not eligible.")
                    count_ineligible += 1                    
                    
        logger.info(f"Filtered ends created. count_avl_train: {count_avl_train}, count_avl_eval: {count_avl_eval}, count_ineligible: {count_ineligible}")
        # extra informs about maximum possible available ends that can
        # be picked to meet the concurrency target. But it might count
        # infeasible ends too (ends that have already particpated in
        # the round). It is essentially a superset of feasible and
        # infeasible. Maximum feasible comes from filtered_ends. We
        # define and henceforth use feasible_extra to (i) use extra's
        # knowledge of how many to pick and (ii) use filtered_ends
        # knowledge of what is feasible to pick Eg scenarios:
        # (extra=1, filtered=3),  (extra=2, filtered=2), (extra=3,
        # filtered=1)
        feasible_extra = min(extra, len(filtered_ends))
        logger.info(f"desired extra: {extra}, len(filtered_ends): {len(filtered_ends)}, feasible_extra: {feasible_extra}")

        # Early exit if filtered_ends is none (can happen when all
        # ends available are less than concurrency requirement)
        if len(filtered_ends) == 0:
            logger.debug(f"len(filtered_ends): {len(filtered_ends)}, hence returning "
                         f"with empty candidates")
            return {}

        # TODO: (DG) Clean up this implementation later. Candidates
        # are being selected differently based on the train or eval
        # tasks.
        if task_to_perform == "train":
            # Make a filter of blocklist ends
            blocklist_end_ids = self.find_blocklists(filtered_ends)

            # TODO: (DG) Move trainer unavail list to inside select()
            # instead? Make a filter of unavailable ends
            if trainer_unavail_list != []:
                logger.info(
                    "### Oort select got non-empty trainer_unavail_list, will "
                    "remove unavail trainers from round"
                )

            # get the list of unavailable_ends and pass to
            # fetch_statistical_utility treat unavailable_ends like
            # blocklist_ends inside fetch_statistical_utility

            # Make a list of tuple (end_id, end_utility) as an
            # utility_list As unexplored ends that are not selected
            # before do not have utility value, collect them
            # separately with unexplored_end_ids list TODO: (DG) Check
            # inside fetch_statistical_utility to see if we can
            # directly pass eligible_ends or a subset of ends, that
            # take into account unavailable ends too.
            logger.debug(f"Invoking fetch_statistical_utility(): with filtered_ends: "
                        f"{filtered_ends}, blocklist_end_ids: {blocklist_end_ids}, "
                        f"trainer_unavail_list: {trainer_unavail_list}"
                        )
            utility_list, unexplored_end_ids = self.fetch_statistical_utility(
                filtered_ends, blocklist_end_ids, trainer_unavail_list
            )
            logger.debug(f"After fetch_statistical_utility(): utility_list: "
                        f"{utility_list}, unexplored_end_ids: {unexplored_end_ids}")

            # DG: Removed old check for first round This indicates the
            # first round, where no end's utility has been measured;
            # Then, perform random selection
            if round == 0:
                self.round = round

                logger.debug(f"Round: {self.round}, will sample feasible_extra: "
                            f"{feasible_extra} from len(filtered_ends): "
                            f"{len(filtered_ends)}")
                candidates_dict = self.select_random(
                    filtered_ends,
                    num_of_ends=feasible_extra
                    )
                # Invoke process_chosen_candidate_dict(). It will
                # appropriately add candidates to selected_ends and
                # all_selected
                self.process_chosen_candidate_dict(
                    candidates_dict=candidates_dict,
                    selected_ends=selected_ends
                    )

                logger.debug(f"handle_send_state returning "
                            f"candidates_dict: {candidates_dict}")

                return candidates_dict

            # Not the first round, performing Oort-based selection
            # Calculate number of ends to select for exploration and
            # exploitation
            logger.debug(f"Invoking calculate_num_of_exploration_exploitation() "
                        f"with num_of_ends: {feasible_extra}, "
                        f"unexplored_end_ids: {unexplored_end_ids}")
            (
                exploration_len,
                exploitation_len,
            ) = self.calculate_num_of_exploration_exploitation(
                num_of_ends=feasible_extra,
                unexplored_end_ids=unexplored_end_ids
            )
            logger.debug(f"After calculate_num_of_exploration_exploitation(), "
                        f"exploration_len: {exploration_len}, exploitation_len: "
                        f"{exploitation_len}")

            # Calculate the total utility value of trainers with
            # applying temporal uncertainty and global system utility
            logger.debug(f"Invoking calculate_total_utility() with utility_list: "
                        f"{utility_list}, filtered_ends: {filtered_ends}, round: {round}")
            utility_list = self.calculate_total_utility(utility_list, filtered_ends, round)

            logger.info(f"After calculate_total_utility, utility_list: {utility_list}")

            # cutOfUtil from Oort algorithm
            logger.debug(f"Invoking cutoff_util() with utility_list: {utility_list}, "
                        f"num_of_ends: {feasible_extra}")
            cutoff_utility = self.cutoff_util(
                utility_list,
                num_of_ends=feasible_extra
                )
            logger.info(f"After cutoff_util(), cutoff_utility: {cutoff_utility}")

            # perform random if cutoff_utility == 0 TODO: (DG) Check.
            # Removed "and len(self.selected_ends) == 0 from the if
            # condition"
            if len(utility_list) == 0:
                self.round = round
                logger.debug(f"len(utility_list) = {len(utility_list)}, will invoke "
                            f"select_random() with filtered_ends: {filtered_ends} and "
                            f"feasible_extra: {feasible_extra}")
                candidates_dict = self.select_random(
                    filtered_ends,
                    num_of_ends=feasible_extra
                    )
                # Invoke process_chosen_candidate_dict(). It will
                # appropriately add candidates to selected_ends and
                # all_selected
                self.process_chosen_candidate_dict(
                    candidates_dict=candidates_dict,
                    selected_ends=selected_ends
                    )

                logger.debug(f"handle_send_state returning "
                            f"candidates_dict: {candidates_dict}")

                return candidates_dict

            # TODO: (DG) Separate this out based on the async_oort
            # selection mode. Keep one for default, one for fastest
            # and one for maxSamples
            if self.select_type == "default":
                candidates, exploit_end_ids = self._select_candidates_using_default(
                    cutoff_utility=cutoff_utility,
                    utility_list=utility_list,
                    exploitation_len=exploitation_len,
                    exploration_len=exploration_len,
                    unexplored_end_ids=unexplored_end_ids
                )
            elif self.select_type == "fastest":
                candidates, exploit_end_ids = self._select_candidates_fastest(
                    ends=filtered_ends,
                    num_of_ends=feasible_extra
                )
            elif self.select_type == "maxSamples":
                candidates, exploit_end_ids = self._select_candidates_maxSamples(
                    ends=filtered_ends,
                    num_of_ends=feasible_extra
                )
            elif self.select_type == "prioritiseUnavail":
                candidates, exploit_end_ids = self._select_candidates_prioritiseUnavail(
                    ends=filtered_ends,
                    num_of_ends=feasible_extra
                )
            elif self.select_type == "fairShare":
                candidates, exploit_end_ids = self._select_candidates_fairShare(
                    ends=filtered_ends,
                    num_of_ends=feasible_extra
                )

            # Converting list of candidates to candidate_dict so that
            # it can be passed to a function to process it
            candidates_dict = {key: None for key in candidates}

            # Invoke process_chosen_candidate_dict(). It will
            # appropriately add candidates to selected_ends and
            # all_selected
            self.process_chosen_candidate_dict(
                candidates_dict=candidates_dict,
                selected_ends=selected_ends
                )

            # save the history of exploited utility at this round for
            # pacer TODO: (DG) check if ends needs to be passed or
            # filtered_ends
            if self.select_type == "default":
                logger.debug(f"Invoking save_exploited_utility_history() with ends: {ends},"
                            f" exploit_end_ids: {exploit_end_ids}")
                self.save_exploited_utility_history(ends, exploit_end_ids)

                # update the exploration_factor
                logger.debug("Invoking update_exploration_factor()")
                self.update_exploration_factor()

            # increment the round selected count on selected ends
            # TODO: (DG) simplify the code here
            candidate_ends = dict()
            for end_id in candidates:
                candidate_ends[end_id] = ends[end_id]

            logger.debug(f"Invoking increment_selected_count_on_selected_ends() "
                        f"with ends: {ends}, candidate_ends: {candidate_ends}")
            self.increment_selected_count_on_selected_ends(ends, candidate_ends)

            self.round = round
        
        elif task_to_perform == "eval":
            # Here we populate a mapping between all items in
            # filtered_ends and the last eval round they participated
            # in. We then sort this list in ascending order of the
            # eval rounds and pick the first feasible_extra items from
            # it. This ensures that the clients that have not been
            # picked for evaluation for the longest time are picked up
            # first.
            
            # feasible_extra is the number of ends that can be picked
            # in this round for eval.
            original_feasible_extra = feasible_extra
            feasible_extra = min(feasible_extra, self.curr_round_eval_slots_left)
            logger.info(f"feasible_extra: {feasible_extra} after min with original_feasible_extra: {original_feasible_extra} and curr_round_eval_slots_left: {self.curr_round_eval_slots_left}")
            
            end_id_to_last_eval_round = {
                end_id: ends[end_id].get_property(PROP_LAST_EVAL_ROUND) or 0
                for end_id in filtered_ends
            }

            sorted_end_ids = sorted(
                end_id_to_last_eval_round,
                key=end_id_to_last_eval_round.get
            )

            candidates = sorted_end_ids[:feasible_extra]
            
            # Adjust eval slots left based on candidate list chosen
            self.curr_round_eval_slots_left -= len(candidates)
            
            logger.info(f"Selected candidates with last_eval_rounds for eval: {[(end_id, end_id_to_last_eval_round[end_id]) for end_id in candidates]}, curr_round_eval_slots_left: {self.curr_round_eval_slots_left} for round {round}")
            
            candidates_dict = {key: None for key in candidates}

            # Invoke process_chosen_candidate_dict(). It will
            # appropriately add candidates to selected_ends and
            # all_selected
            self.process_chosen_candidate_dict(
                candidates_dict=candidates_dict,
                selected_ends=selected_ends
            )

        logger.info(f"handle_send_state returning candidates_dict: {candidates_dict} for task_to_perform: {task_to_perform}")

        return candidates_dict

    def _handle_recv_state(
        self, ends: dict[str, End], concurrency: int
    ) -> SelectorReturnType:
        selected_ends = self.selected_ends[self.requester]
        logger.debug(f"selected_ends: {selected_ends}")

        # from the selected ends, remove those that are in recv state
        # already This is done to avoid waiting on trainers that you
        # have already heard from. If selected ends is empty, get()
        # will proceed and wait on distribute_weights before running
        # again. Thus, it avoids stalling and ensures progress
        for end_id in list(selected_ends):
            # trainer might have become unavailable, check if it is
            # still available first
            if end_id in ends:
                curr_end_state = ends[end_id].get_property(KEY_END_STATE)
                if curr_end_state == VAL_END_STATE_RECVD:
                    selected_ends.remove(end_id)
                    logger.debug(f"Removed end_id {end_id} from selected ends since it "
                                 f"was already in {curr_end_state} state")
                # TODO: (DG) Remove ends from send state also here for
                # the trainer side? But how will it impact the
                # aggregator?
            else:
                # TODO: (DG) Should we not remove it from selected
                # ends here?
                logger.debug(f"Tried to check state of end {end_id} but it is no "
                             f"longer in self._ends")

        if len(selected_ends) == 0:
            logger.debug(f"len(selected_ends)=0, let's select {concurrency} ends")

            candidates = dict()
            for end_id, end in ends.items():
                curr_end_state = end.get_property(KEY_END_STATE)
                if end_id not in self.all_selected.keys():
                    if curr_end_state != VAL_END_STATE_NONE:
                        logging.debug(f"end_id {end_id} not in all_selected and in state: {curr_end_state}, adding "
                                  f"to candidates: key {end_id}, val: {end}")
                        candidates[end_id] = end
                    else:
                        logging.debug(f"end_id {end_id} not in all_selected but in state: {curr_end_state}, not adding "
                                  f"to candidates")

            cc = min(len(candidates), concurrency)
            logger.debug(
                f"Will pick cc: {cc} as min(candidates,concurrency) "
                f"from candidates: {candidates}"
            )
            selected_ends = set(random.sample(list(candidates), cc))

            self.selected_ends[self.requester] = selected_ends
            logger.debug(
                f"self.selected_ends[req]: {self.selected_ends[self.requester]}"
            )

            for selected_end in selected_ends:
                # Add to all_selected. {key: end, val: TS epoch (s)}
                self.all_selected[selected_end] = time.time()
            logging.debug(f"self.all_selected {self.all_selected} after combining with "
                          f"selected_ends {selected_ends}")

        logger.debug(f"handle_recv_state returning selected_ends: {selected_ends}")

        return {key: None for key in selected_ends}
    
    def reset_end_state_to_none(
            self, ends: dict[str, End], end_id: str) -> None:
        """Reset's the state of end_id from send/recv to none"""
        if end_id in ends.keys():
            curr_end_state = ends[end_id].get_property(KEY_END_STATE)
            ends[end_id].set_property(KEY_END_STATE, VAL_END_STATE_NONE)
            new_end_state = ends[end_id].get_property(KEY_END_STATE)
            logger.debug(f"Successfully reset state for end "
                         f"{end_id} from previous: {curr_end_state} to "
                         f"current: {new_end_state}")
        else:
            logger.debug(f"Attempted to reset end {end_id} state "
                         f"but it wasnt in ends")
        
    def remove_from_selected_ends(
            self, ends: dict[str, End], end_id: str) -> None:
        """Remove an end from selected ends"""
        selected_ends = self.selected_ends[self.requester]
        if end_id in ends.keys():
            if end_id in selected_ends:
                logger.debug(f"Going to remove end_id {end_id} from selected_ends "
                             f"{selected_ends}")
                selected_ends.remove(end_id)
                self.selected_ends[self.requester] = selected_ends
                logger.debug(f"self.selected_ends: {self.selected_ends} after "
                             f"removing end_id: {end_id}")
            else:
                logger.debug(f"Attempted to remove end {end_id} from "
                             f"self.selected_ends {self.selected_ends}, but it wasnt present")
        else:
            logger.debug(f"Attempted to remove end {end_id} from "
                         f"self.selected_ends {self.selected_ends}, but it wasnt in ends")

    def process_chosen_candidate_dict(
            self,
            candidates_dict: dict[str, None],
            selected_ends: set[str],
            ):
        candidates = list(candidates_dict.keys())
        logger.debug(f"Got candidates_dict as {candidates_dict} after "
                     f"select_random")
        logger.debug(f"candidates: {candidates}")

        # add candidates to selected ends
        selected_ends = selected_ends.union(candidates)
        self.selected_ends[self.requester] = selected_ends
        logger.debug(
            f"added candidates to selected_ends: {candidates}, selected_ends: "
            f"{selected_ends}, "
            f"self.selected_ends[req]: {self.selected_ends[self.requester]}"
        )

        for candidate_end in candidates:
            # Add to all_selected. {key: end, val: TS epoch (s)}
            self.all_selected[candidate_end] = time.time()
        logging.debug(f"self.all_selected {self.all_selected} after combining"
                      f" with candidates {candidates}")

        logger.debug("finished processing candidates_dict")
