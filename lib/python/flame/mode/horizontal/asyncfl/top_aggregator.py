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
"""Asynchronous horizontal FL top level aggregator."""

import logging
import time
from copy import deepcopy
import math
import numpy as np

from flame.channel import VAL_CH_STATE_RECV, VAL_CH_STATE_SEND
from flame.common.constants import DeviceType
from flame.common.util import weights_to_device, weights_to_model_device
from flame.mode.composer import CloneComposer
from flame.mode.horizontal.syncfl.top_aggregator import TAG_AGGREGATE, TAG_DISTRIBUTE
from flame.mode.horizontal.syncfl.top_aggregator import TopAggregator as SyncTopAgg
from flame.mode.message import MessageType
from flame.mode.tasklet import Loop, Tasklet
from flame.optimizer.train_result import TrainResult

logger = logging.getLogger(__name__)


class TopAggregator(SyncTopAgg):
    """Asynchronous top level Aggregator implements an ML aggregation role."""

    def internal_init(self) -> None:
        """Initialize internal state for role."""
        super().internal_init()

        self._agg_goal_cnt = 0
        self._agg_goal_weights = None
        self._agg_goal = self.config.hyperparameters.aggregation_goal or 1

        self._updates_recevied = {}
        self._trainer_participation_in_round_count = {}
        self._trainer_participation_in_round = {}
        self._per_round_update_list = []
        self._per_round_staleness_list = []
        self._aggregator_staleness_track_rounds = []
        self._aggregator_round_avg_staleness = []
        self._per_trainer_staleness_track = {}
        self.agg_start_time_ts = time.time()

    def _reset_agg_goal_variables(self):
        logger.info("##### reset agg goal variables")
        # reset agg goal count
        self._agg_goal_cnt = 0

        # reset agg goal weights
        self._agg_goal_weights = None
        logger.info(
            f"##### reset _agg_goal_cnt:{self._agg_goal_cnt}, _agg_goal_weights:{self._agg_goal_weights}"
        )

    def _aggregate_weights(self, tag: str) -> None:
        """Aggregate local model weights asynchronously.

        This method is overriden from one in synchronous top aggregator
        (..top_aggregator).
        """
        channel = self.cm.get_by_tag(tag)
        if not channel:
            return

        # receive local model parameters from a trainer who arrives first
        msg, metadata = next(channel.recv_fifo(channel.ends(VAL_CH_STATE_RECV), 1))
        end, _ = metadata
        if not msg:
            logger.debug(f"No data from {end}; skipping it")
            return

        logger.debug(f"received data from {end}")
        logger.info(f"*** received data from {end}")

        # capture telemetry on trainer participation in rounds
        self._per_round_update_list.append(end)

        if end not in self._updates_recevied.keys():
            self._updates_recevied[end] = 1
        else:
            self._updates_recevied[end] += 1

        if MessageType.WEIGHTS in msg:
            weights = weights_to_model_device(msg[MessageType.WEIGHTS], self.model)

        if MessageType.DATASET_SIZE in msg:
            count = msg[MessageType.DATASET_SIZE]

        if MessageType.MODEL_VERSION in msg:
            version = msg[MessageType.MODEL_VERSION]

        logger.debug(f"{end}'s parameters trained with {count} samples")

        if weights is not None and count > 0:
            tres = TrainResult(weights, count, version)
            # save training result from trainer in a disk cache
            self.cache[end] = tres
            logger.debug(f"received {len(self.cache)} trainer updates in cache")
            logger.debug(f"agg_version: {self._round}, trainer version: {tres.version}")
            update_staleness_val = self._round - tres.version
            logger.debug(f"update_staleness_val: {update_staleness_val}")
            self._per_round_staleness_list.append(update_staleness_val)

            # capture per trainer staleness
            if end in self._per_trainer_staleness_track.keys():
                logger.debug(f"found {end} in dict")
                self._per_trainer_staleness_track[end].append(update_staleness_val)
                logger.debug(
                    f"updated _per_trainer_staleness_track {self._per_trainer_staleness_track}"
                )
            else:
                logger.debug(f"NEW Entry {end} in dict")
                self._per_trainer_staleness_track[end] = []
                logger.debug(
                    f"created new list entry in dict _per_trainer_staleness_track {self._per_trainer_staleness_track}"
                )
                self._per_trainer_staleness_track[end].append(update_staleness_val)
                logger.debug(
                    f"updated _per_trainer_staleness_track {self._per_trainer_staleness_track}"
                )

            # staleness_alpha = 0.3
            # staleness_factor = staleness_alpha * (1 / (self._round - tres.version + 1))

            # DG-FIX: check trainer version, discard if stale
            # if (tres.version == (self._round - 1)) or ((tres.version == self._round)):

            # if tres.version == self._round:
            #     logger.debug("proceeding to agg weights")
            #     self._agg_goal_weights = self.optimizer.do(
            #         self._agg_goal_weights,
            #         self.cache,
            #         total=count,
            #         version=self._round,
            #         staleness_factor=staleness_factor,
            #     )
            #     # increment agg goal count
            #     self._agg_goal_cnt += 1
            # else:
            #     logger.debug("stale update from worker, discarding")
            #     return

            logger.debug("proceeding to agg weights")
            self._agg_goal_weights = self.optimizer.do(
                self._agg_goal_weights,
                self.cache,
                total=count,
                version=self._round,
                staleness_factor=0.0,
            )
            # increment agg goal count
            self._agg_goal_cnt += 1

        if self._agg_goal_cnt < self._agg_goal:
            # didn't reach the aggregation goal; return
            logger.debug("didn't reach agg goal")
            logger.debug(f" current: {self._agg_goal_cnt}; agg goal: {self._agg_goal}")
            return

        if self._agg_goal_weights is None:
            logger.debug("failed model aggregation")
            time.sleep(1)
            return

        # set global weights, by adding scaled aggregated weights with aggregation goal
        if self._agg_goal_cnt == self._agg_goal:
            logger.debug("reached agg goal")
            logger.debug(f" current: {self._agg_goal_cnt}; agg goal: {self._agg_goal}")

            # update per-trainer participation in round agg
            for trainer_update in self._per_round_update_list:
                if (
                    trainer_update
                    not in self._trainer_participation_in_round_count.keys()
                ):
                    self._trainer_participation_in_round_count[trainer_update] = 1
                    self._trainer_participation_in_round[trainer_update] = [
                        0
                    ] * 20000  # assuming max 20K rounds
                    self._trainer_participation_in_round[trainer_update][
                        self._round - 1
                    ] = 1
                else:
                    self._trainer_participation_in_round_count[trainer_update] += 1
                    self._trainer_participation_in_round[trainer_update][
                        self._round - 1
                    ] = 1

            # update staleness list for aggregator
            self._aggregator_staleness_track_rounds.append(
                self._per_round_staleness_list
            )

            self._aggregator_round_avg_staleness.append(
                np.mean(np.array(self._per_round_staleness_list))
            )

            self._per_round_update_list = []
            self._per_round_staleness_list = []

        # Computing rate
        rate = 1 / math.sqrt(1 + self._round - tres.version)
        logger.debug(f" rate at top_agg: {rate}")

        self.weights = self.optimizer.scale_add_agg_weights(
            self.weights, self._agg_goal_weights, self._agg_goal, rate
        )

        # update model with global weights
        self._update_model()

        logger.debug(f"aggregation finished for round {self._round}")
        logger.info(
            f"====== aggregation finished for round {self._round}, self._agg_goal_cnt: {self._agg_goal_cnt}, self._updates_recevied: {self._updates_recevied}, self._trainer_participation_in_round_count: {self._trainer_participation_in_round_count}"
        )
        if self._round % 100 == 0:
            logger.debug(
                f"top agg staleness list after round {self._round} is {self._aggregator_round_avg_staleness}"
            )
            logger.debug(
                f"top agg trainer participation in rounds, after round {self._round} is {self._trainer_participation_in_round}"
            )

        # print out data on staleness
        # for aggregator, per round
        # unroll the list of lists into a numpy array, get the avg
        agg_staleness_arr = np.hstack(self._aggregator_staleness_track_rounds)
        logger.info(f"==== aggregator avg staleness: {np.mean(agg_staleness_arr)}")

        # per trainer analytics
        for k, v in self._per_trainer_staleness_track.items():
            trainer_staleness_arr = np.array(v)
            logger.info(
                f"===== trainer {k} staleness info. Min {np.min(trainer_staleness_arr)}, Max {np.max(trainer_staleness_arr)}, Avg {np.mean(trainer_staleness_arr)}, P50 {np.median(trainer_staleness_arr)}, P90 {np.percentile(trainer_staleness_arr, 90)}, P99 {np.percentile(trainer_staleness_arr, 99)}"
            )

    def _distribute_weights(self, tag: str) -> None:
        """Distributed a global model in asynchronous FL fashion.

        This method is overriden from one in synchronous top aggregator
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

        # send out global model parameters to trainers
        for end in channel.ends(VAL_CH_STATE_SEND):
            # verify that end isn't being used for the second time in the same round
            # if trainer availability is being tracked, verify that the trainer is available
            picked_trainer_is_available = True
            if self.trainer_unavail_durations != None:
                logger.info(f"### Will check if trainer {end} is available")
                if end in self.trainer_unavail_durations.keys():
                    # get aggregator seconds from start
                    agg_time_since_start_s = time.time() - self.agg_start_time_ts
                    
                    curr_trainer_unavail_list = self.trainer_unavail_durations[end]

                    # iterate through unavailability list
                    # First, check if the current time is within any failure window

                    for start_time, duration in curr_trainer_unavail_list:
                        if start_time <= agg_time_since_start_s < start_time + duration:
                            print("### Trainer ", end, " attempted to be picked in failed state.")
                            picked_trainer_is_available = False
                            break
                    else:
                        print("### Trainer " , end, " is available.")
                        picked_trainer_is_available = True

                    # Remove entries that occurred in the past
                    updated_trainer_unavail_list = [(start_time, duration) for start_time, duration in curr_trainer_unavail_list if (start_time + duration) >= agg_time_since_start_s]

                    # Remove end from trainer_unavail_durations if list is empty
                    # TODO: Check if deletion is happening properly
                    if len(updated_trainer_unavail_list) == 0:
                        print("### Trainer ", end, " will no longer fail, removing from trainer_unavail_durations")
                        del self.trainer_unavail_durations[end]
                    else:
                        self.trainer_unavail_durations[end] = updated_trainer_unavail_list

            if (end not in self._trainers_used_in_curr_round) and picked_trainer_is_available:
                logger.info(f"sending weights to {end}")
                # we use _round to indicate a model version
                channel.send(
                    end,
                    {
                        MessageType.WEIGHTS: weights_to_device(
                            self.weights, DeviceType.CPU
                        ),
                        MessageType.ROUND: self._round,
                        MessageType.MODEL_VERSION: self._round,
                    },
                )
                # add trainer to list of trainers used in the current round
                self._trainers_used_in_curr_round.append(end)
            else:
                if not picked_trainer_is_available:
                    logger.info(f"Tried to send weights to trainer {end} in round {self._round}, unavailable")
                else:
                    logger.info(f"Tried to send weights again to trainer {end} in round {self._round}, not allowed")
                # remove the end from self.all_selected in fedbuff's select since it would have been added in it in 
                # _select_send_state and this will prevent the aggregator for sending the weights to the trainer once
                # the round increments, and the trainer is again eligible.
                channel._selector.all_selected.remove(end)
                channel._selector.selected_ends[channel._selector.requester].remove(end)
                logger.info(f"Removed {end} from channel._selector.all_selected {channel._selector.all_selected} and channel._selector.selected_ends[channel._selector.requester]: {channel._selector.selected_ends[channel._selector.requester]}")
                

    def compose(self) -> None:
        """Compose role with tasklets."""
        super().compose()

        with CloneComposer(self.composer) as _:
            task_internal_init = Tasklet("internal_init", self.internal_init)

            task_reset_agg_goal_vars = Tasklet(
                "reset_agg_goal_vars", self._reset_agg_goal_variables
            )

            task_put = Tasklet("distribute", self.put, TAG_DISTRIBUTE)

            task_get = Tasklet("aggregate", self.get, TAG_AGGREGATE)

        c = self.composer
        # unlink tasklets that are chained from the parent class
        # (i.e., super().compose()).
        #
        # unlink() internally calls tasklet.reset(), which in turn
        # initialize all loop related state, which includes cont_fn.
        # therefore, if cont_fn is needed for a tasklet, set_continue_fn()
        # in Tasklet class should be used.
        c.unlink()

        loop = Loop(loop_check_fn=lambda: self._work_done)

        # create a loop object for asyncfl to manage concurrency as well as
        # aggregation goal
        asyncfl_loop = Loop(loop_check_fn=lambda: self._agg_goal_cnt == self._agg_goal)

        # chain them again with new tasklets introduced in this class
        (
            task_internal_init
            >> c.tasklet("load_data")
            >> c.tasklet("initialize")
            >> loop(
                task_reset_agg_goal_vars
                >> asyncfl_loop(task_put >> task_get)
                >> c.tasklet("train")
                >> c.tasklet("evaluate")
                >> c.tasklet("analysis")
                >> c.tasklet("save_metrics")
                >> c.tasklet("inc_round")
            )
            >> c.tasklet("inform_end_of_training")
            >> c.tasklet("save_params")
            >> c.tasklet("save_model")
        )

    @classmethod
    def get_func_tags(cls) -> list[str]:
        """Return a list of function tags defined in the top level aggregator role."""
        return [TAG_DISTRIBUTE, TAG_AGGREGATE]
