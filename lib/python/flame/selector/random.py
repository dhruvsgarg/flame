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
"""RandomSelector class."""

import logging
import random

from ..common.typing import Scalar
from ..end import End
from . import AbstractSelector, SelectorReturnType

logger = logging.getLogger(__name__)


class RandomSelector(AbstractSelector):
    """A random selector class."""

    def __init__(self, **kwargs):
        """Initailize instance."""
        super().__init__(**kwargs)

        try:
            self.k = kwargs["k"]
        except KeyError:
            raise KeyError("k is not specified in config")

        if self.k < 0:
            self.k = 1

        self.round = 0

        # Tracking selected ends to ensure selection correctness for
        # each round (a trainer can participate only once per round).
        self.all_selected = dict()
        self.selected_ends = dict()

        # Tracks updates received from trainers and makes them
        # available to select again
        self.ordered_updates_recv_ends = list()

        # Tracks timeouted trainers and number of times it happened to
        # a trainer
        self.track_trainer_timeouts = dict()

        # Tracks trainers that were selected but left training in
        # between
        self.track_selected_trainers_which_left = dict()
    def select(
        self,
        ends: dict[str, End],
        channel_props: dict[str, Scalar],
        task_to_perform: str = "train",
    ) -> SelectorReturnType:
        """Return k number of ends from the given ends."""
        logger.debug("calling random select")

        # default, availability unaware way of using ends
        eligible_ends = ends

        k = min(len(ends), self.k)
        if k == 0:
            logger.debug("ends is empty")
            return {}

        round = channel_props["round"] if "round" in channel_props else 0

        if len(self.selected_ends) == 0 or round > self.round:
            logger.debug(
                f"let's select {k} ends for new round {round}, task: {task_to_perform}"
            )

            # Return existing selected end_ids if the round did not proceed
            if round <= self.round and len(self.selected_ends) != 0:
                return {key: None for key in self.selected_ends}

            # Adopted from FwdLLM: For each variance comparison, we are
            # selecting the same clients each round
            # TODO (DG/NRL): Seeding needs to be put behind a config param
            # incase another selector doesnt need this?
            # TODO (DG/NRL): How will this work with Unavailability? Will be
            # severly disadvantaged with unavl. Naive solution for this?
            random.seed(round)
            logger.info(f"k is {k}")
            self.selected_ends = set(random.sample(list(ends), k))
            self.round = round

        logger.info(f"selected ends: {self.selected_ends}")

        return {key: None for key in self.selected_ends}
    def _cleanup_recvd_ends(self, ends: dict[str, End]):
        """Clean up ends whose a message was received, from selected
        ends.

        NOTE: It sets the end state to none which makes it eligible to
        be sampled again. This can cause problems if sampled in the
        same round. Thus, for aggregator, the _cleanup_recvd_ends
        should be triggered only after aggregation of weights succeeds
        on meeting agg_goal."""
        logger.debug("clean up recvd ends")
        logger.debug(f"ends: {ends.keys()}")
        logger.debug(f"selected ends: {self.selected_ends}")

        selected_ends = self.selected_ends

        num_ends_to_remove = min(len(self.ordered_updates_recv_ends), self.k)
        logger.debug(f"num_ends_to_remove: {num_ends_to_remove}")
        if num_ends_to_remove != 0:
            ends_to_remove = self.ordered_updates_recv_ends[:num_ends_to_remove]
            logger.debug(
                f"Will remove these ends from "
                f"ordered_updates_recv_ends: {ends_to_remove}"
                f" and selected_ends and all_selected"
            )

            # removing the first agg-goal number of ends to free them
            # to participate in the next round
            self.ordered_updates_recv_ends = self.ordered_updates_recv_ends[
                num_ends_to_remove:
            ]
            logger.debug(
                f"self.ordered_updates_recv_ends after removing first "
                f"num_ends_to_remove: {num_ends_to_remove} "
                f"elements: {self.ordered_updates_recv_ends}"
            )

            for end_id in ends_to_remove:
                if end_id not in ends:
                    # something happened to end of end_id (e.g.,
                    # connection loss) let's remove it from
                    # selected_ends
                    logger.debug(
                        f"no end id {end_id} in ends, removing "
                        f"from selected_ends and all_selected"
                    )
                    # NOTE: it is not a guarantee that selected_ends
                    # will still contain the end_id. Thats because it
                    # might have got disconnected/ rejoined in the
                    # middle of a round
                    if end_id in selected_ends:
                        selected_ends.remove(end_id)
                        logger.debug(
                            f"No end id {end_id} in ends, removed from "
                            f"selected_ends: "
                            f"{selected_ends}"
                        )
                    if end_id in self.all_selected:
                        del self.all_selected[end_id]
                        logger.debug(
                            f"No end id {end_id} in ends, removed from "
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
                        logger.debug(
                            f"Setting {end_id} state to {VAL_END_STATE_NONE}, "
                            f"and removing from selected_ends "
                            f"and all_selected"
                        )
                        if end_id in selected_ends:
                            selected_ends.remove(end_id)
                            logger.debug(
                                f"FOUND end id {end_id} in state: {state}.. "
                                f"removed from "
                                f"selected_ends: {selected_ends}"
                            )
                        if end_id in self.all_selected:
                            del self.all_selected[end_id]
                            logger.debug(
                                f"FOUND end id {end_id} in state: {state}.. "
                                f"removed from "
                                f"self.all_selected: "
                                f"{self.all_selected}"
                            )
                    elif state == VAL_END_STATE_NONE:
                        # TODO: (DG) Recheck if it needs to be deleted
                        # from here as well. Is the failure scenario
                        # being handled correctly if the trainer
                        # contributes, fails and then comes back
                        # within the same round. TODO: (DG) Need a
                        # diagram in the paper to explain this?
                        logger.debug(
                            f"Found end {end_id} in state {VAL_END_STATE_NONE}. Might have "
                            f"left/rejoined. Need to remove it from "
                            f"selected_ends and self.all_selected if it "
                            f"was selected"
                        )
                        if end_id in selected_ends:
                            selected_ends.remove(end_id)
                            logger.debug(
                                f"FOUND end id {end_id} in state: {state}.. "
                                f"removed from "
                                f"selected_ends: {selected_ends}"
                            )
                        if end_id in self.all_selected:
                            del self.all_selected[end_id]
                            logger.debug(
                                f"FOUND end id {end_id} in state: {state}.. "
                                f"removed from "
                                f"self.all_selected: "
                                f"{self.all_selected} too"
                            )
                    else:
                        logger.debug(
                            f"FOUND end id {end_id} in state: {state}. "
                            f"Not doing anything"
                        )
        else:
            logger.debug("No ends to remove so far")
