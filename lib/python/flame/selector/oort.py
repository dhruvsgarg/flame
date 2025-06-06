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
from datetime import timedelta
from collections import deque
import numpy as np

import numpy as np
from flame.common.typing import Scalar
from flame.common.util import MLFramework, get_ml_framework_in_use
from flame.end import End
from flame.selector import AbstractSelector, SelectorReturnType

logger = logging.getLogger(__name__)

PROP_UTILITY = "utility"
PROP_END_ID = "end_id"
PROP_SELECTED_COUNT = "selected_count"
PROP_ROUND_START_TIME = "round_start_time"
PROP_ROUND_DURATION = "round_duration"
PROP_STAT_UTILITY = "stat_utility"
PROP_DATASET_SIZE = "dataset_size"
PROP_UPDATE_COUNT = "update_count"
PROP_LAST_SELECTED_ROUND = "last_selected_round"
PROP_LAST_EVAL_ROUND = "last_eval_round"


class OortSelector(AbstractSelector):
    """A selector class based on Oort."""

    def __init__(self, **kwargs):
        """Initailize instance."""
        super().__init__(**kwargs)

        ml_framework_in_use = get_ml_framework_in_use()
        if ml_framework_in_use != MLFramework.PYTORCH:
            raise NotImplementedError(
                "FedBalancer is currently only implemented in PyTorch;"
            )

        try:
            self.aggr_num = kwargs["aggr_num"]
        except KeyError:
            raise KeyError("aggr_num is not specified in config")

        if self.aggr_num < 0:
            self.aggr_num = 1
        self.round = 0

        # With Oort, we select 1.3 * k ends and wait until k ends to
        # complete at a round
        self.overcommitment = 1.3
        self.num_of_ends = int(self.aggr_num * self.overcommitment)

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

        # Tracks updates received from trainers and makes them
        # available to select again NOTE: Not used in sync but just
        # present there
        self.ordered_updates_recv_ends = list()
        
        # Track sliding window statistics for the selector
        self._selector_stats = {}
        for task in ["train", "eval"]:
            self._selector_stats[task] = {"data": {}, "summary": {}}
            for metric in ["util", "speed", "round"]:
                for window in [50, 100, 200]:
                    key = f"{metric}_last_{window}"
                    self._selector_stats[task]['data'][key] = deque(maxlen=window)
        
        self._select_run_counter = 0

    def compute_trainer_stat_summary(self):
        def compute_summary(values):
            # Filter out None values
            if values is None:
                return {
                    "min": None,
                    "max": None,
                    "p25": None,
                    "p50": None,
                    "p75": None,
                }
            values = [v for v in values if v is not None]
            if not values:
                return {
                    "min": None,
                    "max": None,
                    "p25": None,
                    "p50": None,
                    "p75": None,
                }

            values = np.array(values, dtype=float)
            return {
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "p25": float(np.percentile(values, 25)),
                "p50": float(np.percentile(values, 50)),
                "p75": float(np.percentile(values, 75)),
            }

        tasks = ["train", "eval"]
        metrics = [
            "util_last_50", "util_last_100", "util_last_200",
            "speed_last_50", "speed_last_100", "speed_last_200",
            "round_last_50", "round_last_100", "round_last_200"
        ]

        for task in tasks:
            for metric in metrics:
                values = self._selector_stats[task]['data'].get(metric, [])
                key = f"stat_{metric}" if "util" in metric else metric
                self._selector_stats[task]["summary"][key] = compute_summary(values)
        
    def _reset_selector_stats(self) -> None:
        self._selector_stats = {}
    
    def select(
        self,
        ends: dict[str, End],
        channel_props: dict[str, Scalar],
        trainer_unavail_list: list,
        task_to_perform: str,
    ) -> SelectorReturnType:
        """Return k number of ends from the given ends."""
        logger.debug("calling oort select")

        num_of_ends = min(len(ends), self.num_of_ends)
        if num_of_ends == 0:
            logger.debug("ends is empty")
            return {}

        round = channel_props["round"] if "round" in channel_props else 0
        logger.info(
            f"let's select {num_of_ends} ends for new round {round}, task: {task_to_perform}"
        )

        # Return existing selected end_ids if the round did not
        # proceed
        if round <= self.round and len(self.selected_ends) != 0:
            return {key: None for key in self.selected_ends}

        # Run pacer that controls round_threshold
        self.pacer()

        # Make a filter of blocklist ends
        blocklist_end_ids = self.find_blocklists(ends)

        # Make a filter of unavailable ends
        if trainer_unavail_list != []:
            logger.debug(
                "### Oort select got non-empty trainer_unavail_list, will "
                "remove unavail trainers from round"
            )

        # get the list of unavailable_ends and pass to
        # fetch_statistical_utility treat unavailable_ends like
        # blocklist_ends inside fetch_statistical_utility

        # Make a list of tuple (end_id, end_utility) as an
        # utility_list As unexplored ends that are not selected before
        # do not have utility value, collect them separately with
        # unexplored_end_ids list
        utility_list, unexplored_end_ids = self.fetch_statistical_utility(
            ends, blocklist_end_ids, trainer_unavail_list
        )

        # This indicates the first round, where no end's utility has
        # been measured; Then, perform random selection
        if len(utility_list) == 0 and len(self.selected_ends) == 0:
            self.round = round
            return self.select_random(ends, num_of_ends)

        # Not the first round, performing Oort-based selection
        # Calculate number of ends to select for exploration and
        # exploitation
        (
            exploration_len,
            exploitation_len,
        ) = self.calculate_num_of_exploration_exploitation(
            num_of_ends, unexplored_end_ids
        )

        # Calculate the total utility value of trainers with applying
        # temporal uncertainty and global system utility
        utility_list = self.calculate_total_utility(utility_list, ends, round)

        logger.debug(f"{utility_list=}")

        # cutOfUtil from Oort algorithm
        cutoff_utility = self.cutoff_util(utility_list, num_of_ends)

        # perform random if cutoff_utility == 0
        if len(utility_list) == 0 and len(self.selected_ends) == 0:
            self.round = round
            return self.select_random(ends, num_of_ends)

        # sample exploitation_len of clients by utility
        exploit_end_ids = self.sample_by_util(
            cutoff_utility, utility_list, exploitation_len
        )
        logger.debug(f"exploit-selected ends: {exploit_end_ids}")

        # sample exploration_len of unexplored clients
        explore_end_ids = []
        if self.exploration_factor > 0.0 and len(unexplored_end_ids) > 0:
            explore_end_ids = self.sample_by_speed(unexplored_end_ids, exploration_len)
        logger.debug(f"explore-selected ends: {explore_end_ids}")

        self.selected_ends = [*explore_end_ids, *exploit_end_ids]

        # save the history of exploited utility at this round for
        # pacer
        self.save_exploited_utility_history(ends, exploit_end_ids)

        # update the exploration_factor
        self.update_exploration_factor()

        # increment the round selected count on selected ends
        self.increment_selected_count_on_selected_ends(ends)

        logger.info(f"selected ends: {self.selected_ends}")
        self.round = round
        
        # Computations for selector statistics
        self._select_run_counter += 1
        
        for selected_end_id in self.selected_ends:
            end_stat_util = ends[selected_end_id].get_property(PROP_STAT_UTILITY)
            end_speed = ends[selected_end_id].get_property(PROP_ROUND_DURATION)
            end_last_round = ends[selected_end_id].get_property(PROP_LAST_EVAL_ROUND)
            # Insert to queues tracking stat_util, speed, round data
            for window in [50, 100, 200]:
                if end_stat_util is not None:
                    self._selector_stats[task_to_perform]['data'][f'util_last_{window}'].append(end_stat_util)
                if end_speed is not None:
                    self._selector_stats[task_to_perform]['data'][f'speed_last_{window}'].append(end_speed.total_seconds())
                if end_last_round is not None:
                    self._selector_stats[task_to_perform]['data'][f'round_last_{window}'].append(end_last_round)
            
        if self._select_run_counter % 5 == 0:
            self.compute_trainer_stat_summary()
            logger.info(f"Train selector stats summary: {self._selector_stats['train']['summary']}")
            logger.info(f"Eval selector stats summary: {self._selector_stats['eval']['summary']}")
            self._select_run_counter = 0
                
        return {key: None for key in self.selected_ends}

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
                self.exploitation_util_history[-2 * self.pacer_step : -self.pacer_step]
            )
            curr_pacer_step_util = sum(
                self.exploitation_util_history[-self.pacer_step :]
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
                    # TODO: (DG) Check if this is needed. Was put in
                    # as a hack for eval selector. Unsure if it will
                    # be used in sync OORT. Can set it to 60 seconds
                    # since that is the max round duration for
                    # training.
                    sorted_round_duration.append(timedelta(seconds=60))
            logger.debug(
                f"after for loop, sorted_round_duration: {sorted_round_duration}"
            )
            round_preferred_duration = timedelta(
                seconds=sorted_round_duration[
                    min(
                        int(len(sorted_round_duration) * self.round_threshold / 100.0),
                        len(sorted_round_duration) - 1,
                    )
                ].total_seconds()
            )
        else:
            # Assuming a max round duration of 99999 seconds (~1.2
            # days)
            round_preferred_duration = timedelta(seconds=99999)

        logger.debug(f"returning round_preferred_duration: {round_preferred_duration}")
        return round_preferred_duration

    def calculate_temporal_uncertainty_of_trainer(
        self, ends: dict[str, End], end_id: str, round: int
    ) -> float:
        """
        Calculate temproal uncertainty term based on the end's last
        selected round.
        """

        end_last_selected_round = ends[end_id].get_property(PROP_LAST_SELECTED_ROUND)
        return math.sqrt(0.1 * math.log(round) / end_last_selected_round)

    def calculate_global_system_utility_of_trainer(
        self, ends: dict[str, End], end_id: str
    ) -> float:
        """
        Calculate global system utility based on the end's round
        duration.
        """

        end_round_duration = ends[end_id].get_property(PROP_ROUND_DURATION)

        # TODO:(DG) Verify if this is needed for syncfl oort. Was put
        # in place to replicate async_oort.py.
        if end_round_duration is None:
            return 1

        if end_round_duration <= self.round_preferred_duration:
            return 1
        else:
            # Get both into datetime seconds before division
            return math.pow(
                self.round_preferred_duration.total_seconds()
                / end_round_duration.total_seconds(),
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

    def increment_selected_count_on_selected_ends(self, ends: dict[str, End]) -> None:
        """Increment the round selected count on selected ends."""

        for end_id in self.selected_ends:
            if ends[end_id].get_property(PROP_SELECTED_COUNT) == None:
                ends[end_id].set_property(PROP_SELECTED_COUNT, 1)
            else:
                ends[end_id].set_property(
                    PROP_SELECTED_COUNT,
                    ends[end_id].get_property(PROP_SELECTED_COUNT) + 1,
                )

    def select_random(self, ends: dict[str, End], num_of_ends: int) -> dict[str, None]:
        """Randomly select num_of_ends ends."""

        self.selected_ends = set(random.sample(list(ends), num_of_ends))
        logger.debug(f"selected ends: {self.selected_ends}")

        return {key: None for key in self.selected_ends}

    def calculate_total_utility(
        self, utility_list: list[tuple[str, float]], ends: dict[str, End], round: int
    ) -> list[tuple[str, float]]:
        """
        Calculate the total utility value of trainers with applying
        temporal uncertainty and global system utility, based on the
        Oort algorithm.
        """

        # Calculate preferred round duration
        self.round_preferred_duration = self.calculate_round_preferred_duration(ends)

        # Sort the utility list by the utility value placed at the
        # index 1 of each tuple
        utility_list = sorted(utility_list, key=lambda x: x[PROP_UTILITY])

        # Calculate the clip value that caps utility value of a client
        # to no more than an upper bound (95% value in utility
        # distributions)
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

            # Multiply global system utility
            global_system_utility = self.calculate_global_system_utility_of_trainer(
                ends, curr_end_id
            )
            curr_end_utility *= global_system_utility

            utility_list[utility_idx][PROP_UTILITY] = curr_end_utility

        # Sort the utility list again, with the updated utility value
        utility_list = sorted(utility_list, key=lambda x: x[PROP_UTILITY])

        return utility_list

    # TODO: (DG) Check why this is being invoked even if trainer
    # doesn't explicitly invoke remove() method
    def _cleanup_removed_ends(self, end_id):
        logger.debug(
            f"Going to cleanup selector state for "
            f"end_id {end_id} since it has left the channel"
        )
        
    def remove_from_selected_ends(self, ends: dict[str, End], end_id: str) -> None:
        """Remove an end from selected ends"""
        selected_ends = self.selected_ends[self.requester]
        if end_id in ends.keys():
            if end_id in selected_ends:
                logger.debug(
                    f"Going to remove end_id {end_id} from selected_ends "
                    f"{selected_ends}"
                )
                selected_ends.remove(end_id)
                self.selected_ends[self.requester] = selected_ends
                logger.debug(
                    f"self.selected_ends: {self.selected_ends} after "
                    f"removing end_id: {end_id}"
                )
            else:
                logger.debug(
                    f"Attempted to remove end {end_id} from "
                    f"self.selected_ends {self.selected_ends}, but it wasnt present"
                )
        else:
            logger.debug(
                f"Attempted to remove end {end_id} from "
                f"self.selected_ends {self.selected_ends}, but it wasnt in ends"
            )
