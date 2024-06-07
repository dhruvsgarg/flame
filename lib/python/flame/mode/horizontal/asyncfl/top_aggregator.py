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
"""Asynchronous horizontal FL top level aggregator."""

import logging
import math
import threading
import time

import numpy as np
from flame.channel import VAL_CH_STATE_HTBT_RECV, VAL_CH_STATE_RECV, VAL_CH_STATE_SEND
from flame.common.constants import DeviceType
from flame.common.util import weights_to_device, weights_to_model_device
from flame.mode.composer import CloneComposer
from flame.mode.horizontal.syncfl.top_aggregator import (
    TAG_AGGREGATE,
    TAG_DISTRIBUTE,
    TAG_HEARTBEAT,
)
from flame.mode.horizontal.syncfl.top_aggregator import TopAggregator as SyncTopAgg
from flame.mode.message import MessageType
from flame.mode.tasklet import Loop, Tasklet
from flame.optimizer.train_result import TrainResult

logger = logging.getLogger(__name__)


class TopAggregator(SyncTopAgg):
    """Asynchronous top level Aggregator implements an ML aggregation
    role."""

    def internal_init(self) -> None:
        """Initialize internal state for role."""
        super().internal_init()

        self._agg_goal_cnt = 0
        self._agg_goal_weights = None
        self._agg_goal = self.config.hyperparameters.aggregation_goal or 1

        self._updates_in_queue = 0
        self._updates_recevied = {}
        self._trainer_participation_in_round_count = {}
        self._trainer_participation_in_round = {}
        self._per_round_update_list = []
        self._per_round_staleness_list = []
        self._aggregator_staleness_track_rounds = []
        self._aggregator_round_avg_staleness = []
        self._per_trainer_staleness_track = {}
        self._trainer_training_duration_s = {}

        # check if distribute_weights was successful
        self._prev_distribute_weights_success = False

        # variables related to checking trainer availability
        self._per_trainer_last_heartbeat_ts = {}
        if "heartbeat_freq_s" in self.config.hyperparameters.track_trainer_avail.keys():
            self._trainer_heartbeat_freq_s = (
                self.config.hyperparameters.track_trainer_avail["heartbeat_freq_s"]
            )
        else:
            self._trainer_heartbeat_freq_s = 99999
        
        if "max_allowed_miss_heartbeats" in self.config.hyperparameters.track_trainer_avail.keys():
            self._trainer_max_miss_heartbeats = (
                self.config.hyperparameters.track_trainer_avail["max_allowed_miss_heartbeats"]
            )
        else:
            self._trainer_max_miss_heartbeats = 99999

        # maintain a set of all trainers that have sent heartbeats
        # previously
        self.all_trainers = set()

    def _reset_agg_goal_variables(self):
        logger.info("##### reset agg goal variables")
        # reset agg goal count
        self._agg_goal_cnt = 0

        # reset agg goal weights
        self._agg_goal_weights = None
        logger.info(
            f"##### reset _agg_goal_cnt:{self._agg_goal_cnt}, _agg_goal_weights: "
            f"{self._agg_goal_weights}"
        )

    # TODO: (DG) Need to update or delete, not used right now
    def _read_heartbeat(self, tag: str) -> None:
        """Receive trainer heartbeat messaages asynchronously.

        This method is overriden from one in synchronous top
        aggregator (..top_aggregator).
        """
        channel = self.cm.get_by_tag(tag)
        if not channel:
            logger.info("No channel found")
            return
        
        logger.info(f"Channel {channel} found for tag {tag}")
        # receive heartbeat message from trainers
        msg, metadata = next(channel.recv_fifo(channel.ends(VAL_CH_STATE_HTBT_RECV), 1))
        end, _ = metadata
        if not msg:
            logger.debug(f"No data from {end}; skipping it")
            return

        logger.info(f"received heartbeat from {end}, will process further")
        self._process_trainer_heartbeat(msg=msg, end=end)
    
    def _process_trainer_heartbeat(self, msg, end) -> None:
        if MessageType.HEARTBEAT in msg:
            heartbeat_timestamp = msg[MessageType.HEARTBEAT]
            logger.info(f"received heartbeat from {end} "
                        f"with timestamp {heartbeat_timestamp} "
                        f"at current time: {time.time()}")
            
            # Add trainer to global_trainer set Used only to check
            # unavailable trainers later
            if end not in self.all_trainers:
                self.all_trainers.add(end)
                logger.info(f"Added end {end} to all_trainers set")
            
            # Add trainer to heartbeat dict if it isnt there Add only
            # most recent heartbeat timestamp as value Discard stale
            # heartbeats if received.
            if end not in self._per_trainer_last_heartbeat_ts.keys():
                self._per_trainer_last_heartbeat_ts[end] = heartbeat_timestamp
                logger.info(f"Added first timestamp for trainer {end} "
                            f"with timestamp {heartbeat_timestamp}")
            elif heartbeat_timestamp > self._per_trainer_last_heartbeat_ts[end]:
                logger.info(f"Will update timestamp for trainer {end} "
                            f" (current={self._per_trainer_last_heartbeat_ts[end]})"
                            f" with new timestamp {heartbeat_timestamp}")
                self._per_trainer_last_heartbeat_ts[end] = heartbeat_timestamp
            else:
                logger.info(f"the heartbeat for {end} with timestamp "
                            f"{heartbeat_timestamp} was stale")
        else:
            logger.warning(f"Got invalid {msg} while processing heartbeat")
    
    def _aggregate_weights(self, tag: str) -> None:
        """Aggregate local model weights asynchronously.

        This method is overriden from one in synchronous top
        aggregator (..top_aggregator).
        """
        channel = self.cm.get_by_tag(tag)
        if not channel:
            logger.info("No channel found")
            return
        
        logger.info(f"Channel {channel} found for tag {tag}")
        # receive local model parameters from a trainer who arrives
        # first NOTE: (DG) Right now, the leave notifications also
        # cause a message to be processed and yield (None,None) from
        # recv_fifo().
        msg, metadata = next(channel.recv_fifo(channel.ends(VAL_CH_STATE_RECV), 1))
        end, _ = metadata
        if not msg:
            logger.debug(f"No data from {end}; skipping it")
            return

        # If message contains model updates, handle it
        logger.debug(f"received data from {end}")
        if MessageType.MODEL_VERSION in msg:
            logger.info(f"received MODEL_VERSION message in agg_weights from {end}")
            channel._selector.ordered_updates_recv_ends.append(end)
            logger.debug(f"After appending {end} to ordered_updates_recv_ends: {channel._selector.ordered_updates_recv_ends}")
        else:
            logger.warn(f"received INCORRECT message {msg} in agg_weights from {end}")
            return

        if self.reject_stale_updates:
            logger.info("Check trainer model version, disallow stale updates")
            if MessageType.MODEL_VERSION in msg:
                version = msg[MessageType.MODEL_VERSION]

            if version != self._round:
                logger.info(f"Rejecting trainer update of version {version}, "
                            f"agg self._round: {self._round}")
                return

        self._updates_in_queue += 1

        # update _trainer_training_duration_s to capture training time
        if end not in self._trainer_training_duration_s.keys():
            logger.warning(f"{end} not in _trainer_training_duration_s at recv!")
        else:
            last_recv_wts_ts = time.time()
            self._trainer_training_duration_s[
                end]["last_recv_wts_ts"] = last_recv_wts_ts
            # recv_wts_ts should be strictly > send_wts_ts
            last_send_wts_ts = self._trainer_training_duration_s[
                end]["last_send_wts_ts"]
            if last_send_wts_ts > last_recv_wts_ts:
                logger.error(f"{end} last_recv_ts before last_send_ts!")
            else:
                curr_cumulative_training_s = self._trainer_training_duration_s[
                    end
                ]["total_training_time_s"]
                curr_round_time_s = last_recv_wts_ts - last_send_wts_ts
                new_cumulative_training_s = (
                    curr_cumulative_training_s + curr_round_time_s
                )
                self._trainer_training_duration_s[
                    end
                ]["total_training_time_s"] = new_cumulative_training_s
                logger.info(f"Updated training time record for {end}, details: "
                            f"{self._trainer_training_duration_s[end]}")

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
                    f"updated _per_trainer_staleness_track "
                    f"{self._per_trainer_staleness_track}"
                )
            else:
                logger.debug(f"NEW Entry {end} in dict")
                self._per_trainer_staleness_track[end] = []
                logger.debug(
                    f"created new list entry in dict _per_trainer_staleness_track "
                    f"{self._per_trainer_staleness_track}"
                )
                self._per_trainer_staleness_track[end].append(update_staleness_val)
                logger.debug(
                    f"updated _per_trainer_staleness_track "
                    f"{self._per_trainer_staleness_track}"
                )

            # staleness_alpha = 0.3 staleness_factor = staleness_alpha
            # * (1 / (self._round - tres.version + 1))

            # DG-FIX: check trainer version, discard if stale if
            # (tres.version == (self._round - 1)) or ((tres.version ==
            # self._round)):

            # if tres.version == self._round: logger.debug("proceeding
            #     to agg weights") self._agg_goal_weights =
            #     self.optimizer.do( self._agg_goal_weights,
            #         self.cache, total=count, version=self._round,
            #         staleness_factor=staleness_factor, ) # increment
            #         agg goal count self._agg_goal_cnt += 1 else:
            #         logger.debug("stale update from worker,
            #         discarding") return

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

        # set global weights, by adding scaled aggregated weights with
        # aggregation goal
        if self._agg_goal_cnt == self._agg_goal:
            logger.debug("reached agg goal")
            logger.debug(f" current: {self._agg_goal_cnt}; agg goal: {self._agg_goal}")
            logger.info(f"Reached agg_goal {self._agg_goal}, "
                        f"current _updates_in_queue: {self._updates_in_queue}, "
                        f"current round before agg: {self._round}"
                        )

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
            self.weights, self._agg_goal_weights, self._agg_goal
        )

        # update model with global weights
        self._update_model()

        # decrement counter since updates consumed from queue
        self._updates_in_queue -= self._agg_goal

        logger.debug(f"aggregation finished for round {self._round}")
        logger.info(
            f"====== aggregation finished for round {self._round}, "
            f"self._agg_goal_cnt: {self._agg_goal_cnt}, self._updates_recevied: "
            f"{self._updates_recevied}, self._trainer_participation_in_round_count: "
            f"{self._trainer_participation_in_round_count}"
        )
        logger.info(
            f"After round: {self._round}, remaining _updates_in_queue: "
            f"{self._updates_in_queue}"
        )
        
        if self._round % 100 == 0:
            logger.debug(
                f"top agg staleness list after round {self._round} is "
                f"{self._aggregator_round_avg_staleness}"
            )
            logger.debug(
                f"top agg trainer participation in rounds, after round "
                f"{self._round} is {self._trainer_participation_in_round}"
            )

        # print out data on staleness for aggregator, per round unroll
        # the list of lists into a numpy array, get the avg
        agg_staleness_arr = np.hstack(self._aggregator_staleness_track_rounds)
        logger.info(f"==== aggregator avg staleness: {np.mean(agg_staleness_arr)}")

        # per trainer analytics
        for k, v in self._per_trainer_staleness_track.items():
            trainer_staleness_arr = np.array(v)
            logger.info(
                f"Trainer {k} staleness info. Min {np.min(trainer_staleness_arr)}, "
                f"Max {np.max(trainer_staleness_arr)}, "
                f"Avg {np.mean(trainer_staleness_arr)}, "
                f"P50 {np.median(trainer_staleness_arr)}, "
                f"P90 {np.percentile(trainer_staleness_arr, 90)}, "
                f"P99 {np.percentile(trainer_staleness_arr, 99)}"
            )
        logger.debug("Agg goal reached, so resetting trainer end states in the channel")
        channel.cleanup_recvd_ends()

    def oracular_trainer_avail_check(self, end: str) -> bool:
        logger.debug("In oracular_trainer_avail_check")
        
        picked_trainer_is_available = True
        
        if end in self.trainer_unavail_durations.keys():
            # get aggregator seconds from start
            agg_time_since_start_s = time.time() - self.agg_start_time_ts

            curr_trainer_unavail_list = self.trainer_unavail_durations[end]

            # iterate through unavailability list First, check if the
            # current time is within any failure window

            for start_time, duration in curr_trainer_unavail_list:
                if start_time <= agg_time_since_start_s < start_time + duration:
                    print(
                        "### Trainer ",
                        end,
                        " attempted to be picked in failed state.",
                    )
                    picked_trainer_is_available = False
                    return picked_trainer_is_available
            else:
                print("### Trainer ", end, " is available.")
                picked_trainer_is_available = True

            # Remove entries that occurred in the past
            updated_trainer_unavail_list = [
                (start_time, duration)
                for start_time, duration in curr_trainer_unavail_list
                if (start_time + duration) >= agg_time_since_start_s
            ]

            # Remove end from trainer_unavail_durations if list is
            # empty TODO: Check if deletion is happening properly
            if len(updated_trainer_unavail_list) == 0:
                print(
                    "### Trainer ",
                    end,
                    " will no longer fail, removing from "
                    " trainer_unavail_durations",
                )
                del self.trainer_unavail_durations[end]
            else:
                self.trainer_unavail_durations[end] = (
                    updated_trainer_unavail_list
                )
        else:
            logger.info(f"No info on end {end} in self.trainer_unavail_durations"
                        f", returning TRUE (default)")
        return picked_trainer_is_available

    def hearbeat_trainer_avail_check(self, end: str) -> bool:
        picked_trainer_is_available = True
        last_acceptable_heartbeat_ts = time.time() - (
            self._trainer_max_miss_heartbeats * self._trainer_heartbeat_freq_s
            )
        
        # return True if: heartbeat was received from trainer and it
        # is within last_acceptable_heartbeat_ts

        # return False if: if end isnt in heartbeat dict, means that
        # the trainer hasn't given a heartbeat in a while and was
        # removed based on last_acceptable_heartbeat_ts

        # NOTE: During agg init, it might have registered a trainer,
        # but not received heartbeat in such a scenario, we return
        # True so that agg is able to send init_weights to trainer and
        # start the training process this is when trainer not in
        # all_trainers and not in dict

        if (end not in self._per_trainer_last_heartbeat_ts.keys()) and (end not in self.all_trainers):
            picked_trainer_is_available = True
            logger.info(f"Might be trainer init(), trainer {end} hasnt sent any"
                        f" heartbeats yet, but we return True")
        elif end not in self._per_trainer_last_heartbeat_ts.keys():
            picked_trainer_is_available = False
            logger.info(f"Trainer {end} was already marked unavailable")
        elif self._per_trainer_last_heartbeat_ts[end] < last_acceptable_heartbeat_ts:
            del self._per_trainer_last_heartbeat_ts[end]
            picked_trainer_is_available = False
            logger.info(f"Trainer {end} missed max_allowed_heartbeats, "
                        f"marked unavailable")
        elif self._per_trainer_last_heartbeat_ts[end] >= last_acceptable_heartbeat_ts:
            picked_trainer_is_available = True
            logger.info(f"Trainer {end} is available")
        else:
            logger.error(f"Availability check failed, trainer {end}, returning True")
        
        return picked_trainer_is_available
    
    def get_unavailable_trainers(self) -> list:
        # Works only for heartbeat based right now TODO: (DG) Extend
        # for other trainer_avail_checks too
        current_unavailable_trainers = [
            end for end in
            self.all_trainers if
            end not in
            self._per_trainer_last_heartbeat_ts.keys()]
        return current_unavailable_trainers
    
    def check_trainer_availability(self, end: str) -> bool:
        picked_trainer_is_available = True
        if self.track_trainer_avail["enabled"] == "False":
            return True
        elif self.track_trainer_avail["type"] == "ORACULAR":
            picked_trainer_is_available = self.oracular_trainer_avail_check(end)
        elif self.track_trainer_avail["type"] == "HEARTBEAT":
            picked_trainer_is_available = self.hearbeat_trainer_avail_check(end)

        return picked_trainer_is_available

    def _distribute_weights(self, tag: str) -> None:
        """Distribute a global model in asynchronous FL fashion.

        This method is overridden from one in synchronous top
        aggregator (..top_aggregator).
        """
        channel = self.cm.get_by_tag(tag)
        if not channel:
            logger.debug(f"channel not found for tag {tag}")
            return

        # this call waits for at least one peer to join this channel
        channel.await_join()

        # before distributing weights, update it from global model
        self._update_weights()

        # busy wait for 2 seconds before proceeding. This is to wait
        # on distribute_weights to let the system state get updated
        # before selector is invoked again
        logger.debug(f"Starting busy wait at time {time.time()}")
        time.sleep(2)
        logger.debug(f"Ended busy wait at time {time.time()}")

        # check if there are any ends to send weights to
        ends = channel.ends(VAL_CH_STATE_SEND)
        if not ends:
            # logger.debug(f"no trainers found for tag {tag}, retrying
            # in 1 second") time.sleep(1) continue

            logger.debug(f"No trainers found for tag {tag}, will "
                         f"move to get() in 0.1s for fetch weights from trainers")
            time.sleep(0.1)
            return

        # send out global model parameters to trainers
        for end in ends:
            # Send shouldn't be allowed if already sent to a trainer
            # in that same round
            logger.debug(f"sending weights to {end}")
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

    def compose(self) -> None:
        """Compose role with tasklets."""
        super().compose()

        with CloneComposer(self.composer) as _:
            task_internal_init = Tasklet("internal_init", self.internal_init)

            task_reset_agg_goal_vars = Tasklet(
                "reset_agg_goal_vars", self._reset_agg_goal_variables
            )

            task_put = Tasklet("distribute", self.put, TAG_DISTRIBUTE)

            task_get_weights = Tasklet("aggregate", self.get, TAG_AGGREGATE)

            # task_get_heartbeat = Tasklet("heartbeat", self.get,
            # TAG_HEARTBEAT)

        c = self.composer
        # unlink tasklets that are chained from the parent class
        # (i.e., super().compose()).
        #
        # unlink() internally calls tasklet.reset(), which in turn
        # initialize all loop related state, which includes cont_fn.
        # therefore, if cont_fn is needed for a tasklet,
        # set_continue_fn() in Tasklet class should be used.
        c.unlink()

        # # Reset the task_get_heartbeat to ensure it is in the correct
        # # state
        # task_get_heartbeat.reset()

        # # Start a separate thread for the heartbeat task
        # logger.debug("Going to start the thread for processing
        # heartbeats") heartbeat_thread = threading.Thread(
        #     target=self.heartbeat_task, args=(task_get_heartbeat,) )
        # heartbeat_thread.daemon = True heartbeat_thread.start()

        loop = Loop(loop_check_fn=lambda: self._work_done)
        # create a loop object for asyncfl to manage concurrency as
        # well as aggregation goal
        asyncfl_loop = Loop(loop_check_fn=lambda: self._agg_goal_cnt == self._agg_goal)

        # chain them again with new tasklets introduced in this class
        (
            task_internal_init
            >> c.tasklet("load_data")
            >> c.tasklet("initialize")
            >> loop(
                task_reset_agg_goal_vars
                # >> asyncfl_loop(task_put >> task_get_weights >>
                # >> task_get_heartbeat)
                >> asyncfl_loop(task_put >> task_get_weights)
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
        """Return a list of function tags defined in the top level
        aggregator role."""
        return [TAG_DISTRIBUTE, TAG_AGGREGATE, TAG_HEARTBEAT]
