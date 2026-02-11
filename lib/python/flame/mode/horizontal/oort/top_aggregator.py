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
"""Oort horizontal FL top level aggregator."""

import logging
import time
from copy import deepcopy
from datetime import datetime
from typing import Any, Tuple

from flame.common.constants import DeviceType
from flame.common.util import weights_to_device, weights_to_model_device
from flame.mode.message import MessageType
from flame.optimizer.train_result import TrainResult
from flame.selector.oort import (
    PROP_LAST_SELECTED_ROUND,
    PROP_ROUND_DURATION,
    PROP_ROUND_START_TIME,
    PROP_STAT_UTILITY,
    PROP_LAST_EVAL_ROUND,
)

from ..top_aggregator import TopAggregator as BaseTopAggregator

logger = logging.getLogger(__name__)


class TopAggregator(BaseTopAggregator):
    """Oort Top level Aggregator implements an ML aggregation role."""

    def _aggregate_weights(self, tag: str) -> None:
        """
        Aggregate local model weights, accepting K trainers out of
        1.3K clients selected from selector. Moreover, trainers' round
        duration is measured, which determine the system utility of
        trainers for Oort algorithm.

        This method is overriden from one in horizontal top aggregator
        (..top_aggregator).
        """
        channel = self.cm.get_by_tag(tag)
        if not channel:
            return

        total = 0

        # receive local model parameters from trainers terminate
        # aggregating when received weights from k ends (k *
        # overcommitment is selected for training with Oort)

        end_ids = channel.ends()
        aggr_num = min(self.config.selector.kwargs["aggr_num"], len(end_ids))

        received_end_count = 0

        for msg, metadata in channel.recv_fifo(end_ids, aggr_num):
            end, _ = metadata

            if not msg:
                logger.info(f"[MSG_SKIP] No data from ...{end[-8:]}; skipping it")
                continue

            # Calculate staleness
            trainer_round = msg.get(MessageType.MODEL_VERSION, 0)
            staleness = self._round - trainer_round

            # Check if optimizer supports REFL staleness (has stale_update_max attribute)
            stale_update_max = getattr(self.optimizer, 'stale_update_max', None)
            
            # If staleness > 0, check if we should accept or reject based on REFL config
            if staleness > 0:
                if stale_update_max is not None:
                    # REFL mode: check against stale_update_max threshold
                    if stale_update_max >= 0 and staleness > stale_update_max:
                        logger.info(
                            f"[MSG_SKIP] Message from ...{end[-8:]} TOO STALE: "
                            f"staleness={staleness} > stale_update_max={stale_update_max}; skipping it"
                        )
                        continue
                    else:
                        logger.info(
                            f"[MSG_ACCEPT_STALE] Stale message from ...{end[-8:]}, "
                            f"staleness={staleness} <= stale_update_max={stale_update_max}, "
                            f"accepting with weight degradation"
                        )
                else:
                    # Non-REFL mode (standard Oort): reject all stale messages
                    logger.info(
                        f"[MSG_SKIP] Stale message from ...{end[-8:]}, "
                        f"expected_round={self._round}, got_round={trainer_round}; skipping it"
                    )
                    continue

            total = self._handle_weights_msg(msg, metadata, channel, total)

            if end not in self._updates_recevied.keys():
                self._updates_recevied[end] = 1
            else:
                self._updates_recevied[end] += 1
            
            # CRITICAL: Notify selector that this trainer has returned its update
            # This prevents the selector from re-selecting this trainer in the next round
            # before it has returned its update (key for SyncFL with overcommitment)
            channel._selector.ordered_updates_recv_ends.append(end)
            
            logger.info(f"[MSG_ACCEPTED] Message from ...{end[-8:]} accepted, received_end_count={received_end_count + 1}/{aggr_num}")

            # remove end_id if it sends a valid message with correct
            # round info break the for loop if k valid messages arrive
            received_end_count += 1
            # Only remove if end is in end_ids (stale messages from previous rounds won't be)
            if end in end_ids:
                end_ids.remove(end)
            if received_end_count == aggr_num:
                break

        # running the second loop to aggregate up to aggr_num updates
        # from trainers
        while received_end_count < aggr_num:
            for msg, metadata in channel.recv_fifo(end_ids, 1):
                end, _ = metadata

                if not msg:
                    logger.info(f"[MSG_SKIP] (loop2) No data from ...{end[-8:]}; skipping it")
                    continue

                # Calculate staleness
                trainer_round = msg.get(MessageType.MODEL_VERSION, 0)
                staleness = self._round - trainer_round

                # Check if optimizer supports REFL staleness (has stale_update_max attribute)
                stale_update_max = getattr(self.optimizer, 'stale_update_max', None)
                
                # If staleness > 0, check if we should accept or reject based on REFL config
                if staleness > 0:
                    if stale_update_max is not None:
                        # REFL mode: check against stale_update_max threshold
                        if stale_update_max >= 0 and staleness > stale_update_max:
                            logger.info(
                                f"[MSG_SKIP] (loop2) Message from ...{end[-8:]} TOO STALE: "
                                f"staleness={staleness} > stale_update_max={stale_update_max}; skipping it"
                            )
                            continue
                        else:
                            logger.info(
                                f"[MSG_ACCEPT_STALE] (loop2) Stale message from ...{end[-8:]}, "
                                f"staleness={staleness} <= stale_update_max={stale_update_max}, "
                                f"accepting with weight degradation"
                            )
                    else:
                        # Non-REFL mode (standard Oort): reject all stale messages
                        logger.info(
                            f"[MSG_SKIP] (loop2) Stale message from ...{end[-8:]}, "
                            f"expected_round={self._round}, got_round={trainer_round}; skipping it"
                        )
                        continue

                total = self._handle_weights_msg(msg, metadata, channel, total)
                
                # CRITICAL: Notify selector that this trainer has returned its update
                channel._selector.ordered_updates_recv_ends.append(end)
                
                logger.info(f"[MSG_ACCEPTED] (loop2) Message from ...{end[-8:]} accepted, received_end_count={received_end_count + 1}/{aggr_num}")

                # remove end_id if it sends a valid message with
                # correct round info break the for loop if k valid
                # messages arrive
                received_end_count += 1
                # Only remove if end is in end_ids (stale messages from previous rounds won't be)
                if end in end_ids:
                    end_ids.remove(end)
                if received_end_count == aggr_num:
                    break

        logger.debug(f"received {len(self.cache)} trainer updates in cache")

        # optimizer conducts optimization (in this case, aggregation)
        global_weights = self.optimizer.do(
            deepcopy(self.weights), self.cache, total=total
        )
        if global_weights is None:
            logger.debug("failed model aggregation")
            time.sleep(1)
            return

        self._compute_aggregator_stats()
        if self._round % 5 == 0:
            logger.info(f"_agg_training_stats: {self._agg_training_stats}")
        self._reset_aggregator_stats()

        # set global weights
        self.weights = global_weights

        # update model with global weights
        self._update_model()

        # CRITICAL: Clean up trainers who returned updates, freeing them from in-flight set
        # This must happen immediately after aggregation to prevent race condition where
        # trainers finishing after agg_goal but before next select() remain incorrectly
        # marked as busy, allowing them to be sent new work while still processing old work
        num_to_cleanup = len(channel._selector.ordered_updates_recv_ends)
        channel.cleanup_recvd_ends()
        logger.debug(
            f"[CLEANUP] Freed {num_to_cleanup} trainers from in-flight set after aggregation"
        )

        logger.info(
            f"====== aggregation finished for round {self._round}, "
            f"self._updates_recevied: "
            f"{self._updates_recevied}"
        )

    def _distribute_weights(self, tag: str, task_to_perform: str = "train") -> None:
        """
        Distribute local model weights to 1.3K clients, where K is the
        number of desired trainers to select. Moreover, measure the
        start time of a round to for round duration measurement on
        each trainers for measuring their system utility that is
        required for Oort algorithm.

        This method is overriden from one in horizontal top aggregator
        (..top_aggregator).
        """
        channel = self.cm.get_by_tag(tag)
        if not channel:
            logger.debug(f"channel not found for tag {tag}")
            return

        # this call waits for at least one peer to join this channel
        channel.await_join()

        # before distributing weights, update it from global model
        self._update_weights()

        # before invoking channel.ends() to select, set the
        # trainer_unavail if it isn't None
        if self.trainer_event_dict is not None:
            curr_unavail_trainer_list = self.get_curr_unavail_trainers()
            channel.set_curr_unavailable_trainers(
                trainer_unavail_list=curr_unavail_trainer_list
            )
        else:
            # Handling the case for oort's selector since it expects 3
            # arguments
            channel.set_curr_unavailable_trainers(trainer_unavail_list=[])

        logger.debug(
            f"Sending weights to trainers with task_to_perform = {task_to_perform}"
        )

        # send out global model parameters to trainers
        for end in channel.ends():
            logger.info(
                f"sending weights to {end} with model_version: {self._round} for task: {task_to_perform}"
            )
            logger.debug(
                f"Setting channel property {PROP_ROUND_START_TIME} for "
                f"end {end}. For round {self._round} at time: {datetime.now()}"
            )
            channel.set_end_property(
                end, PROP_ROUND_START_TIME, (self._round, datetime.now())
            )

            channel.send(
                end,
                {
                    MessageType.WEIGHTS: weights_to_device(
                        self.weights, DeviceType.CPU
                    ),
                    MessageType.ROUND: self._round,
                    MessageType.MODEL_VERSION: self._round,
                    MessageType.TASK_TO_PERFORM: task_to_perform,
                },
            )

    def _handle_weights_msg(
        self, msg: Any, metadata: Tuple[str, datetime], channel: Any, total: int
    ) -> int:
        end = metadata[0]
        timestamp = metadata[1]

        logger.info(f"[MSG_PROCESSING] Processing message from end ...{end[-8:]}, round={self._round}, msg_version={msg.get(MessageType.MODEL_VERSION, 'N/A')}")
        logger.debug(f"received data from {end}")

        # calculate round duration for this end, if the round number
        # information is identical with round_start_time
        round_start_time_tup = channel.get_end_property(end, PROP_ROUND_START_TIME)
        if round_start_time_tup[0] == msg[MessageType.MODEL_VERSION]:
            channel.set_end_property(
                end, PROP_ROUND_DURATION, timestamp - round_start_time_tup[1]
            )

        if MessageType.WEIGHTS in msg:
            weights = weights_to_model_device(msg[MessageType.WEIGHTS], self.model)

        if MessageType.DATASET_SIZE in msg:
            count = msg[MessageType.DATASET_SIZE]

        if MessageType.STAT_UTILITY in msg:
            channel.set_end_property(
                end, PROP_STAT_UTILITY, msg[MessageType.STAT_UTILITY]
            )
            logger.info(
                f"End {end} sent a message with utility {msg[MessageType.STAT_UTILITY]}"
            )

        trainer_model_version = 0  # default
        if MessageType.MODEL_VERSION in msg:
            channel.set_end_property(
                end, PROP_LAST_SELECTED_ROUND, msg[MessageType.MODEL_VERSION]
            )
            trainer_model_version = msg[MessageType.MODEL_VERSION]
            logger.info(
                f"End {end} sent a model update version {msg[MessageType.MODEL_VERSION]}, while current model version {self._round}"
            )

        # Set last eval round for the trainer since training also
        # means that eval was done for the same round.
        channel.set_end_property(end, PROP_LAST_EVAL_ROUND, trainer_model_version)

        stat_utility = 0  # default
        if MessageType.STAT_UTILITY in msg:
            channel.set_end_property(
                end, PROP_STAT_UTILITY, msg[MessageType.STAT_UTILITY]
            )
            stat_utility = msg[MessageType.STAT_UTILITY]

        logger.debug(f"{end}'s parameters trained with {count} samples")

        if weights is not None and count > 0:
            total += count
            
            # Calculate staleness BEFORE creating TrainResult
            update_staleness_val = self._round - trainer_model_version
            
            # Get round duration if available
            round_duration_obj = channel.get_end_property(end, PROP_ROUND_DURATION)
            round_duration_seconds = None
            if round_duration_obj:
                round_duration_seconds = round_duration_obj.total_seconds()
            
            # Create TrainResult with all REFL-required fields
            tres = TrainResult(
                weights=weights,
                count=count,
                version=trainer_model_version,
                stat_utility=stat_utility,
                staleness=update_staleness_val,
                round_duration=round_duration_seconds,
                end_id=end
            )
            
            # Save training result from trainer in a disk cache
            self.cache[end] = tres
            
            logger.debug(
                f"Created TrainResult for {end}: staleness={update_staleness_val}, "
                f"stat_utility={stat_utility}, round_duration={round_duration_seconds}"
            )

            # Populate round statistics vars
            self._round_update_values["staleness"].append(update_staleness_val)
            self._round_update_values["stat_utility"].append(stat_utility)
            # Only append trainer_speed if round_duration is available
            if round_duration_seconds is not None:
                self._round_update_values["trainer_speed"].append(round_duration_seconds)
            else:
                logger.debug(
                    f"Skipping trainer_speed for {end} - round_duration is None "
                    f"(stale update: msg_version={trainer_model_version}, current_round={self._round})"
                )

        return total
