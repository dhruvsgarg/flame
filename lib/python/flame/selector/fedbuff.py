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
"""FedBuffSelector class."""

import logging
import random
import time

from flame.channel import (
    KEY_CH_SELECT_REQUESTER,
    KEY_CH_STATE,
    VAL_CH_STATE_HTBT_RECV,
    VAL_CH_STATE_HTBT_SEND,
    VAL_CH_STATE_RECV,
    VAL_CH_STATE_SEND,
)
from flame.common.typing import Scalar
from flame.end import KEY_END_STATE, VAL_END_STATE_NONE, VAL_END_STATE_RECVD, End
from flame.selector import AbstractSelector, SelectorReturnType

logger = logging.getLogger(__name__)

SEND_TIMEOUT_WAIT_S = 90      # 90 seconds timeout


class FedBuffSelector(AbstractSelector):
    """A selector class for fedbuff-based asyncfl."""

    def __init__(self, **kwargs):
        """Initailize instance."""
        super().__init__(**kwargs)

        try:
            self.c = kwargs["c"]
        except KeyError:
            raise KeyError("c (concurrency level) is not specified in config")
        
        try:
            self.agg_goal = kwargs["aggGoal"]
        except KeyError:
            raise KeyError("agg_goal (aggregation goal) is not specified in config")

        self.all_selected = dict()
        self.selected_ends = dict()

        # Tracks updates received from trainers and makes them
        # available to select again
        self.ordered_updates_recv_ends = list()

    def select(
        self, ends: dict[str, End], channel_props: dict[str, Scalar]
    ) -> SelectorReturnType:
        """Select ends from the given ends to meet concurrency level.

        This select method chooses ends differently depending on what
        state a channel is in. In 'send' state, it chooses ends that
        are not in self.selected_ends. In 'recv' state, it chooses all
        ends from self.selected_ends. Essentially, if an end is in
        self.selected_ends, it means that we sent some message already
        to that end. For such an end, we exclude it from send and
        include it for recv in return.
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
        
        results = {}
        if channel_props[KEY_CH_STATE] == VAL_CH_STATE_SEND:
            results = self._handle_send_state(ends, concurrency)

        elif channel_props[KEY_CH_STATE] == VAL_CH_STATE_RECV:
            results = self._handle_recv_state(ends, concurrency)
        
        elif channel_props[KEY_CH_STATE] == VAL_CH_STATE_HTBT_RECV:
            results = self._handle_htbt_recv_state(ends)

        elif channel_props[KEY_CH_STATE] == VAL_CH_STATE_HTBT_SEND:
            results = self._handle_htbt_send_state(ends)

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
        """Clean up ends whose a message was received, from selected
        ends.
        
        Note: It sets the end state to none which makes it eligible to
        be sampled again. This can cause problems if sampled in the
        same round. Thus, for aggregator, the _cleanup_recvd_ends
        should be triggered only after aggregation of weights succeeds
        on meeting agg_goal."""
        logger.debug("clean up recvd ends")
        logger.debug(f"ends: {ends.keys()}")
        logger.debug(f"selected ends: {self.selected_ends}")

        selected_ends = self.selected_ends[self.requester]
        logger.debug(
            f"self.requester: {self.requester} and selected_ends: "
            f"{selected_ends} before processing"
        )

        num_ends_to_remove = min(len(self.ordered_updates_recv_ends), self.agg_goal)
        logger.debug(f"num_ends_to_remove: {num_ends_to_remove}")
        if num_ends_to_remove != 0:
            ends_to_remove = self.ordered_updates_recv_ends[:num_ends_to_remove]
            logger.debug(f"Will remove these ends from "
                         f"ordered_updates_recv_ends: {ends_to_remove}"
                         f" and selected_ends and all_selected")
        
            # removing the first agg-goal number of ends to free them
            # to participate in the next round
            self.ordered_updates_recv_ends = self.ordered_updates_recv_ends[num_ends_to_remove:]
            logger.debug(f"self.ordered_updates_recv_ends after removing first "
                         f"num_ends_to_remove: {num_ends_to_remove} "
                         f"elements: {self.ordered_updates_recv_ends}")

            # TODO: (DG) check, replacing selected_ends with
            # ends_to_remove
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
                        logger.debug(f"No end id {end_id} in ends, removed from selected_ends: "
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
                        logger.debug(f"Setting {end_id} state to {VAL_END_STATE_NONE}, and"
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
                    elif state is None:
                        # TODO: (DG) Recheck if it needs to be deleted
                        # from here as well. Is the failure scenario
                        # being handled correctly if the trainer
                        # contributes, fails and then comes back
                        # within the same round.
                        # TODO: (DG) Need a diagram in the paper to
                        # explain this?
                        logger.debug(f"Found end {end_id} in state None. Might have left/rejoined. Need to remove it from selected_ends and self.all_selected if it was selected")
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
                        logger.debug(f"FOUND end id {end_id} in state: {state}. Not doing anything")
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
            del self.all_selected[end_id]
        elif (
            end_id in self.all_selected
         ) and (
             end_id in self.ordered_updates_recv_ends
             ):
            # Dont remove it if it was in all_selected and we have got
            # an update from it before it did channel.leave(). It has
            # completed its participation for this round.
            logger.debug(f"Update was alreacy received from {end_id} before it left the channel. Not deleting from all_ends now.")
        else:
            logger.warn(f"End_id {end_id} remove check from all_selected failed. Need to check")
  
    def _cleanup_send_ends(self):
        # TODO: (DG) Get a more principled solution here. Hacky right
        # now to fix the issue for trainer side when failures occur.
        logger.debug(f"Going to cleanup selector state after "
                     f"send.")
        selected_ends = self.selected_ends[self.requester]
        curr_list_selected_ends = list(selected_ends)
        for end_id in curr_list_selected_ends:
            logger.debug(f"Removing end_id {end_id} from selected_ends")
            selected_ends.remove(end_id)
            self.selected_ends[self.requester] = selected_ends
            if end_id in self.all_selected:
                logger.debug(f"Also removing end_id {end_id} from "
                             f"self.all_selected")
                del self.all_selected[end_id]

    def _handle_send_state(
        self, ends: dict[str, End], concurrency: int
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

        extra = max(0, concurrency - len(selected_ends))
        
        logger.debug(f"c: {concurrency}, ends: {ends.keys()},"
                     f"len(selected_ends): {len(selected_ends)}, extra: {extra}")
        candidates = []
        idx = 0

        # reservoir sampling
                
        # DG: Updated existing reservoir sampling to randomized
        # sampling. NOTE: Might revert back if needed
        random.seed(time.time())  # Seed with current system time
        shuffled_end_ids = list(ends.keys())    # get the keys
        logger.debug(f"Original shuffled_end_ids: {shuffled_end_ids}")
        random.shuffle(shuffled_end_ids)        # then shuffle
        logger.debug(f"Updated shuffled_end_ids: {shuffled_end_ids}")

        # Invalidate previous all_selected entry if you don't get an
        # update in UPDATE_TIMEOUT_WAIT_S. The client might have
        # dropped the message with transient unavailability.

        # TODO: (DG) Check if it affects trainer code
        # TODO: (DG) Check if it is still needed after cleanup_remove_end() method
        curr_all_selected_ends = list(self.all_selected.keys())
        for end in curr_all_selected_ends:
            current_time_s = time.time()
            trainer_weight_send_timestamp_s = self.all_selected[end]
            if (
                trainer_weight_send_timestamp_s < (current_time_s - SEND_TIMEOUT_WAIT_S)
                ) and (
                    end not in self.ordered_updates_recv_ends
                    ):
                # trainer hasn't returned with an update in
                # SEND_TIMEOUT_WAIT_S delete it from self.all_selected
                # so that it is eligible to be sampled again
                logger.debug(f"Removing end {end} from self.all_selected since havent "
                             f"got its update in {SEND_TIMEOUT_WAIT_S}")
                del self.all_selected[end]

        for end_id in shuffled_end_ids:
            if end_id in self.all_selected.keys():
                # skip if an end is already selected
                logger.debug(f"end_id {end_id} in all_selected, so skipping")
                continue

            idx += 1
            if len(candidates) < extra:
                candidates.append(end_id)
                logger.debug(f"Added end_id: {end_id} to candidates: {candidates}")
                continue

            i = random.randrange(idx)
            if i < extra:
                candidates[i] = end_id

        logger.debug(f"candidates: {candidates}")
 
        # add candidates to selected ends
        selected_ends = selected_ends.union(candidates)
        self.selected_ends[self.requester] = selected_ends
        logger.debug(
            f"added candidates to selected_ends: {candidates}, selected_ends: {selected_ends}, "
            f"self.selected_ends[req]: {self.selected_ends[self.requester]}"
        )

        for candidate_end in candidates:
            # Add to all_selected. {key: end, val: TS epoch (s)}
            self.all_selected[candidate_end] = time.time()
        logging.debug(f"self.all_selected {self.all_selected} after combining with "
                      f"candidates {candidates}")
        
        logger.debug(f"handle_send_state returning candidates: {candidates}")

        return {end_id: None for end_id in candidates}
    
    def _handle_htbt_send_state(
            self, ends: dict[str, End]
        ) -> SelectorReturnType:
        # TODO: Implement again Earlier (not fully functional)
        # implementation was using same code as handle_send_state()
        # and was commented out.

        logger.debug(f"ends: {ends}")
        return {end_id: None for end_id in ends}


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
                if ends[end_id].get_property(KEY_END_STATE) == VAL_END_STATE_RECVD:
                    selected_ends.remove(end_id)
                    logger.debug(f"Removed end_id {end_id} from selected ends since it "
                                 f"was already in recvd state")
                # TODO: (DG) Remove ends from send state also here for
                # the trainer side? But how will it impact the aggregator?
            else:
                # TODO: (DG) Should we not remove it from selected
                # ends here?
                logger.debug(f"Tried to check state of end {end_id} but it is no longer "
                             f"in self._ends")

        if len(selected_ends) == 0:
            logger.debug(f"len(selected_ends)=0, let's select {concurrency} ends")

            candidates = dict()
            for end_id, end in ends.items():
                if end_id not in self.all_selected.keys():
                    logging.debug(f"end_id {end_id} not in all_selected, adding "
                                  f"to candidates: key {end_id}, val: {end}")
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

            for selected_end in selected_ends:
                # Add to all_selected. {key: end, val: TS epoch (s)}
                self.all_selected[selected_end] = time.time()
            logging.debug(f"self.all_selected {self.all_selected} after combining with "
                          f"selected_ends {selected_ends}")

        logger.debug(f"handle_recv_state returning selected_ends: {selected_ends}")

        return {key: None for key in selected_ends}
    
    def _handle_htbt_recv_state(
        self, ends: dict[str, End]
    ) -> SelectorReturnType:
        # TODO: Implement again Earlier (not fully functional)
        # implementation was using same code as handle_recv_state()
        # and was commented out.

        logger.debug(f"ends: {ends}")
        return {key: None for key in ends}

    def reset_selected_ends(self, requester: str) -> None:
        """Reset mapping between requester and selected ends.

        This is needed when requester leaves channel due to e.g.,
        failure.
        """
        logger.debug(f"trying to reset selected ends of {requester}")
        if requester not in self.selected_ends:
            return

        selected_ends = self.selected_ends[requester]
        
        # Updated all_selected here when updating in other parts TODO:
        # (DG) Needs to be tested if it works
        all_selected_keys_set = set(self.all_selected.keys())
        diff_of_ends = all_selected_keys_set.difference(selected_ends)
        for end_id in diff_of_ends:
            # Add to all_selected. {key: end, val: TS epoch (s)}
            self.all_selected[end_id] = time.time()

        del self.selected_ends[requester]

        logger.debug(f"all selected: {self.all_selected}")
        logger.debug(f"selected_ends: {self.selected_ends}")
        logger.debug(f"done with resetting selected ends of {requester}")
