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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
"""FedBuffSelector class."""

import logging
import random

from flame.channel import (
    KEY_CH_SELECT_REQUESTER,
    KEY_CH_STATE,
    VAL_CH_STATE_RECV,
    VAL_CH_STATE_SEND,
)
from flame.common.typing import Scalar
from flame.end import KEY_END_STATE, VAL_END_STATE_NONE, VAL_END_STATE_RECVD, End
from flame.selector import AbstractSelector, SelectorReturnType

logger = logging.getLogger(__name__)


class FedBuffSelector(AbstractSelector):
    """A selector class for fedbuff-based asyncfl."""

    def __init__(self, **kwargs):
        """Initailize instance."""
        super().__init__(**kwargs)

        try:
            self.c = kwargs["c"]
        except KeyError:
            raise KeyError("c (concurrency level) is not specified in config")

        self.all_selected = set()
        self.selected_ends = dict()

    def select(
        self, ends: dict[str, End], channel_props: dict[str, Scalar]
    ) -> SelectorReturnType:
        """Select ends from the given ends to meet concurrency level.

        This select method chooses ends differently depending on what state
        a channel is in.
        In 'send' state, it chooses ends that are not in self.selected_ends.
        In 'recv' state, it chooses all ends from self.selected_ends.
        Essentially, if an end is in self.selected_ends, it means that we sent
        some message already to that end. For such an end, we exclude it from
        send and include it for recv in return.
        """
        # TODO (DG): The recv state should also only select from ends
        # IF an update doesnt already exist in the cache
        logger.debug("calling fedbuff select")

        concurrency = min(len(ends), self.c)
        logger.debug(f"len(ends): {len(ends)}, c: {self.c}, concurrency: {concurrency}")

        if concurrency == 0:
            logger.debug("ends is empty")
            return {}

        if KEY_CH_STATE not in channel_props:
            raise KeyError("channel property doesn't have {KEY_CH_STATE}")

        self.requester = channel_props[KEY_CH_SELECT_REQUESTER]
        if self.requester not in self.selected_ends:
            self.selected_ends[self.requester] = set()

        self._cleanup_recvd_ends(ends)
        results = {}
        if channel_props[KEY_CH_STATE] == VAL_CH_STATE_SEND:
            results = self._handle_send_state(ends, concurrency)

        elif channel_props[KEY_CH_STATE] == VAL_CH_STATE_RECV:
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

    def _cleanup_recvd_ends(self, ends: dict[str, End]):
        """Clean up ends whose a message was received, from selected ends."""
        logger.debug(
            f"In clean up recvd ends, ends.keys(): {ends.keys()} and "
            f"self.selected_ends: {self.selected_ends}"
        )

        selected_ends = self.selected_ends[self.requester]
        logger.debug(
            f"self.requester: {self.requester} and selected_ends: "
            f"{selected_ends} before processing"
        )
        for end_id in list(selected_ends):
            if end_id not in ends:
                # something happened to end of end_id
                # (e.g., connection loss)
                # let's remove it from selected_ends
                selected_ends.remove(end_id)
                self.all_selected.remove(end_id)
                logger.debug(
                    f"No end id {end_id} in ends, removed from selected_ends: "
                    f"{selected_ends} and self.all_selected: {self.all_selected}"
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
                    selected_ends.remove(end_id)
                    self.all_selected.remove(end_id)
                    logger.debug(
                        f"FOUND end id {end_id} in state: {state}.. removed from "
                        f"selected_ends: {selected_ends} and self.all_selected: "
                        f"{self.all_selected}"
                    )

    def _handle_send_state(
        self, ends: dict[str, End], concurrency: int
    ) -> SelectorReturnType:
        selected_ends = self.selected_ends[self.requester]

        extra = max(0, concurrency - len(selected_ends))
        logger.debug(
            f"concurrency: {concurrency}, ends: {ends.keys()}, selected_ends: "
            f"{selected_ends}, extra: {extra}, self.all_selected: {self.all_selected}"
        )

        # TODO (DG): Send back updates to all nodes that participated
        # extra = max(0, concurrency - len(selected_ends))
        # logger.debug(f"c: {concurrency}, ends: {ends.keys()}")

        candidates = []
        idx = 0
        # reservoir sampling
        for end_id in ends.keys():
            if end_id in self.all_selected:
                # skip if an end is already selected
                logger.debug(
                    f"end_id: {end_id} already in self.all_selected: "
                    f"{self.all_selected}, skipping"
                )
                continue

            idx += 1
            if len(candidates) < extra:
                candidates.append(end_id)
                logger.debug(f"Added end_id: {end_id} to candidates: {candidates}")
                continue

            i = random.randrange(idx)
            if i < extra:
                candidates[i] = end_id

        # add candidates to selected ends
        selected_ends = selected_ends.union(candidates)
        self.selected_ends[self.requester] = selected_ends
        logger.debug(
            f"candidates after shuffle: {candidates}, selected_ends: {selected_ends}, "
            f"self.selected_ends[req]: {self.selected_ends[self.requester]}"
        )

        self.all_selected = self.all_selected.union(candidates)

        logger.debug(
            f"self.all_selected after union with candidates: {self.all_selected}"
        )
        logger.debug(f"handle_send_state returning candidates: {candidates}")

        return {end_id: None for end_id in candidates}

    def _handle_recv_state(
        self, ends: dict[str, End], concurrency: int
    ) -> SelectorReturnType:
        selected_ends = self.selected_ends[self.requester]
        logger.debug(f"selected_ends: {selected_ends}")

        if len(selected_ends) == 0:
            logger.debug(f"len(selected_ends)=0, let's select {concurrency} ends")

            candidates = dict()
            for end_id, end in ends.items():
                if end_id not in self.all_selected:
                    candidates[end_id] = end

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

            self.all_selected = self.all_selected.union(selected_ends)
            logger.debug(
                f"self.all_selected after uninon with selected_ends: "
                f"{self.all_selected}"
            )

        logger.debug(f"handle_recv_state returning selected_ends: {selected_ends}")

        return {key: None for key in selected_ends}

    def reset_selected_ends(self, requester: str) -> None:
        """Reset mapping between requester and selected ends.

        This is needed when requester leaves channel due to e.g., failure.
        """
        logger.debug(f"trying to reset selected ends of {requester}")
        if requester not in self.selected_ends:
            return

        selected_ends = self.selected_ends[requester]
        self.all_selected = self.all_selected.difference(selected_ends)
        del self.selected_ends[requester]

        logger.debug(f"all selected: {self.all_selected}")
        logger.debug(f"selected_ends: {self.selected_ends}")
        logger.debug(f"done with resetting selected ends of {requester}")
