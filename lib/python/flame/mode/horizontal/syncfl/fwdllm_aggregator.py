# Copyright 2023 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
#
# SPDX-License-Identifier: Apache-2.0
"""SyncFL horizontal FL top level aggregator for FwdLLM."""

import gc
import logging
import psutil
import time
from datetime import datetime
import sklearn
import numpy as np
from flame.channel import VAL_CH_STATE_HTBT_RECV, VAL_CH_STATE_RECV, VAL_CH_STATE_SEND
from flame.common.constants import DeviceType
from flame.common.util import weights_to_device, weights_to_model_device
from flame.mode.composer import CloneComposer
import pickle
from flame.mode.horizontal.syncfl.top_aggregator import (
    TAG_AGGREGATE,
    TAG_DISTRIBUTE,
    TAG_HEARTBEAT,
)
from sklearn.metrics import (
    confusion_matrix,
    matthews_corrcoef,
)
from flame.mode.horizontal.asyncfl.top_aggregator import TopAggregator as AsyncTopAgg
# from flame.mode.horizontal.syncfl.top_aggregator import TopAggregator as SyncTopAgg
from flame.mode.message import MessageType
from flame.mode.tasklet import Loop, Tasklet
from flame.optimizer.train_result import TrainResult
from flame.selector.oort import (
    PROP_DATASET_SIZE,
    PROP_LAST_SELECTED_ROUND,
    PROP_LAST_EVAL_ROUND,
    PROP_ROUND_DURATION,
    PROP_ROUND_START_TIME,
    PROP_STAT_UTILITY,
    PROP_UPDATE_COUNT,
)
import functorch as fc
import torch

from torch.nn import CrossEntropyLoss
import flame.monitor.runtime
from flame.monitor.runtime import FwdLLMStage, timer_decorator

logger = logging.getLogger(__name__)

PROP_ROUND_END_TIME = "round_end_time"

SEND_TIMEOUT_WAIT_S = 90  # 90 seconds timeout

@timer_decorator
def recv_fifo_wrapper(channel, ends):
    logger.info("[GJD] Entering recv_fifo_wrapper generator loop")
    for msg, metadata in channel.recv_fifo(ends):
        logger.info(f"[GJD] Yielding msg from {metadata}")
        yield msg, metadata
    logger.info("[GJD] Exiting recv_fifo_wrapper")

class TopAggregator(AsyncTopAgg):
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
        self._track_trainer_version_duration_s = {}

        # check if distribute_weights was successful
        self._prev_distribute_weights_success = False

        self.data_id = 0
        self.total_data_bins = 150
        self._is_model_updated = False
        self._model_version = 0
        self.grad_pool = []
        self.var = None
        self.ends_not_selected_yet = False
        self.iteration_per_data_id = 0
        # variables related to checking trainer availability
        self._per_trainer_last_heartbeat_ts = {}
        if "heartbeat_freq_s" in self.config.hyperparameters.track_trainer_avail.keys():
            self._trainer_heartbeat_freq_s = (
                self.config.hyperparameters.track_trainer_avail["heartbeat_freq_s"]
            )
        else:
            self._trainer_heartbeat_freq_s = 99999

        if (
            "max_allowed_miss_heartbeats"
            in self.config.hyperparameters.track_trainer_avail.keys()
        ):
            self._trainer_max_miss_heartbeats = (
                self.config.hyperparameters.track_trainer_avail[
                    "max_allowed_miss_heartbeats"
                ]
            )
        else:
            self._trainer_max_miss_heartbeats = 99999

        # maintain a set of all trainers that have sent heartbeats previously
        self.all_trainers = set()
        try:
            self.minInitialTrainers = self.config.selector.kwargs.get("minInitialTrainers")
            assert self.minInitialTrainers is not None
        except (KeyError, AssertionError):
            raise KeyError("minInitialTrainers must be specified in selector config & must not be None for determinism")
        logger.info("finished init for sync agg")

    def pause_execution(self):
        time.sleep(1)
        return

    def log_memory(self, tag, device):
        # GPU memory
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)

        # CPU memory
        process = psutil.Process()
        cpu_memory = process.memory_info().rss  # in bytes

        logging.info(
            f"[MEM:{tag}] "
            f"GPU Allocated: {allocated/1e6:.2f} MB | "
            f"GPU Reserved: {reserved/1e6:.2f} MB | "
            f"CPU Memory: {cpu_memory/1e6:.2f} MB | "
            f"Device: {device}, aggregator"
        )

    def print_trainable_params_stats(self, location=""):
        total_params = 0
        trainable_params = 0
        total_size = 0.0
        trainable_size = 0.0

        for param in self.model.parameters():
            numel = param.numel()
            size_MB = numel * param.element_size() / 1e6

            total_params += numel
            total_size += size_MB

            if param.requires_grad:
                trainable_params += numel
                trainable_size += size_MB

        fraction = trainable_params / total_params if total_params > 0 else 0
        loc_str = f"[{location}] " if location else ""

        print(
            f"{loc_str}Trainable params: {trainable_params:,} / {total_params:,} "
            f"({fraction:.2%}), Size: {trainable_size:.2f} MB / {total_size:.2f} MB"
        )

    def get_trainable_param_state_dict(self):
        return {
            name: param.detach().cpu()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

    def print_param_dict_stats(self, param_dict, location=""):
        total_params = 0
        total_size = 0.0

        for tensor in param_dict.values():
            numel = tensor.numel()
            size_MB = numel * tensor.element_size() / 1e6

            total_params += numel
            total_size += size_MB

        loc_str = f"[{location}] " if location else ""
        print(
            f"{loc_str}Param dict stats — Total params: {total_params:,}, Size: {total_size:.2f} MB"
        )

    def _reset_agg_goal_variables(self):
        logger.debug("##### reset agg goal variables")
        # reset agg goal count
        self._agg_goal_cnt = 0

        # reset agg goal weights
        self._agg_goal_weights = None
        logger.debug(
            f"##### reset _agg_goal_cnt:{self._agg_goal_cnt}, _agg_goal_weights: "
            f"{self._agg_goal_weights}"
        )

    # TODO: (DG) Need to update or delete, not used right now
    def _read_heartbeat(self, tag: str) -> None:
        """Receive trainer heartbeat messaages asynchronously.

        This method is overriden from one in synchronous top aggregator
        (..top_aggregator).
        """
        channel = self.cm.get_by_tag(tag)
        if not channel:
            logger.info("No channel found")
            return

        logger.debug(f"Channel {channel} found for tag {tag}")
        # receive heartbeat message from trainers
        msg, metadata = next(channel.recv_fifo(channel.ends(VAL_CH_STATE_HTBT_RECV), 1))
        end, _ = metadata
        if not msg:
            logger.debug(f"No data from {end}; skipping it")
            return

        logger.debug(f"received heartbeat from {end}, will process further")
        self._process_trainer_heartbeat(msg=msg, end=end)

    def _process_trainer_heartbeat(self, msg, end) -> None:
        if MessageType.HEARTBEAT in msg:
            heartbeat_timestamp = msg[MessageType.HEARTBEAT]
            logger.debug(
                f"received heartbeat from {end} "
                f"with timestamp {heartbeat_timestamp} "
                f"at current time: {time.time()}"
            )

            # Add trainer to global_trainer set Used only to check unavailable
            # trainers later
            if end not in self.all_trainers:
                self.all_trainers.add(end)
                logger.debug(f"Added end {end} to all_trainers set")

            # Add trainer to heartbeat dict if it isnt there Add only most
            # recent heartbeat timestamp as value Discard stale heartbeats if
            # received.
            if end not in self._per_trainer_last_heartbeat_ts.keys():
                self._per_trainer_last_heartbeat_ts[end] = heartbeat_timestamp
                logger.debug(
                    f"Added first timestamp for trainer {end} "
                    f"with timestamp {heartbeat_timestamp}"
                )
            elif heartbeat_timestamp > self._per_trainer_last_heartbeat_ts[end]:
                logger.debug(
                    f"Will update timestamp for trainer {end} "
                    f" (current={self._per_trainer_last_heartbeat_ts[end]})"
                    f" with new timestamp {heartbeat_timestamp}"
                )
                self._per_trainer_last_heartbeat_ts[end] = heartbeat_timestamp
            else:
                logger.info(
                    f"the heartbeat for {end} with timestamp "
                    f"{heartbeat_timestamp} was stale"
                )
        else:
            logger.warning(f"Got invalid {msg} while processing heartbeat")

    @timer_decorator
    def _aggregate_grads_async(self, tag: str) -> None:
        """
        Aggregate local model GRADIENTS asynchronously for FwdLLM.
        
        This method is overridden from AsyncTopAgg.
        It receives gradients, aggregates them until _agg_goal is met,
        then performs FwdLLM variance check and model update.
        """
        channel = self.cm.get_by_tag(tag)
        if not channel:
            logger.info("No channel found")
            return
        
        if(channel.ends(VAL_CH_STATE_RECV) is None):
            logger.info("no ends yet")
            return
        time.sleep(0.1)  # Slight delay to allow messages to arrive

        msg, metadata = next(channel.recv_fifo(channel.ends(VAL_CH_STATE_RECV), 1))
        end, _ = metadata
        if not msg:
            logger.debug(f"No data from {end}; skipping it")
            return

        if MessageType.GRADIENTS in msg and MessageType.GRADIENTS_FOR_VAR_CHECK in msg:
            logger.info(
                f"Received gradients from {end} "
                f"with model version {msg[MessageType.MODEL_VERSION]}"
            )
            channel.set_end_property(
                end, PROP_LAST_SELECTED_ROUND, msg[MessageType.MODEL_VERSION]
            )
            channel.set_end_property(
                end, PROP_LAST_EVAL_ROUND, msg[MessageType.MODEL_VERSION]
            )
        elif MessageType.STAT_UTILITY in msg:
            logger.info(
                f"Received eval-only message from {end}, "
                f"stat_utility {msg[MessageType.STAT_UTILITY]}"
            )
            channel.set_end_property(
                end, PROP_STAT_UTILITY, msg[MessageType.STAT_UTILITY]
            )
            channel.set_end_property(
                end, PROP_LAST_EVAL_ROUND, msg[MessageType.MODEL_VERSION]
            )
            channel._selector.trainer_eval_recv_ends.append(end)
            channel._selector.remove_from_selected_ends(channel._ends, end)
            channel._selector._cleanup_removed_ends(end)
            return
        else:
            logger.error(
                f"Invalid message received from {end} in aggregate_weights: {msg}"
            )
            return
        
        #TODO: Check if we want to discard after putting in the queue  
        if self.reject_stale_updates == True:
            logger.info("Check trainer model version, disallow stale updates")
            if MessageType.MODEL_VERSION in msg:
                version = msg[MessageType.MODEL_VERSION]
                logger.info(f"Model version aggregator: {self._model_version}, Model version trainer: {version}")

            if version != self._model_version:
                logger.info(
                    f"Rejecting trainer update of version {version}, "
                    f" self._model_version: {self._model_version}. Will return."
                )
                return

        channel._selector.ordered_updates_recv_ends.append(end)
        self._updates_in_queue += 1

        
        if MessageType.GRADIENTS in msg:
            trainer_gradients = msg[MessageType.GRADIENTS]
            self.aggregate_grads_from_trainers(trainer_gradients)
            # del trainer_gradients # Free memory
        
        if MessageType.GRADIENTS_FOR_VAR_CHECK in msg:
            self.grad_for_var_check_list.append(
                msg[MessageType.GRADIENTS_FOR_VAR_CHECK]
            )

        if MessageType.DATASET_SIZE in msg:
            count = msg[MessageType.DATASET_SIZE]
            channel.set_end_property(end, PROP_DATASET_SIZE, count)
        
        logger.info(f"Received and processed grads from {end}.")
        
        self._agg_goal_cnt += 1

        if self._agg_goal_cnt < self._agg_goal:
            logger.info(f"Agg goal not met. Have {self._agg_goal_cnt}/{self._agg_goal}")
            channel.set_end_property(
                end, PROP_UPDATE_COUNT, self._updates_recevied.get(end, 0) + 1
            )
            return

        if self._agg_goal_cnt == self._agg_goal:
            logger.info(f"Aggregation goal {self._agg_goal} reached. Performing FwdLLM aggregation.")
            
            self.grad_pool.append(self.grad)
            self.add_local_trained_result(0, self.grad, self._agg_goal_cnt) # Assuming 0 is ok
            
            self.fmodel, self.params, self.buffers = fc.make_functional_with_buffers(
                self.model
            )
            self.grad = [torch.zeros_like(p) for p in self.params]
            
            self.aggregate(self._round) # This sets self.var and self.var_good_enough
            
            if self.var_good_enough:

                logger.info(f"Variance check PASSED. Evaluating model and advancing data_id.")
                self.iteration_per_data_id += 1 # This is iter 1 for the new data_id. 
                result, _, _ = self.eval_model()
                logger.info(f"Round {self._round}, Data ID {self.data_id} Eval Loss: {result['eval_loss']}")
                self.data_id += 1
                self.iteration_per_data_id = 0 # Reset iteration count
                self._is_model_updated = True
                self._model_version +=1

                if self.data_id == self.total_data_bins:
                    logger.info(f"All data bins complete. Incrementing round to {self._round + 1}")
                    self._round += 1
                    self.data_id = 0
                    channel.set_property("round", self._round) # Update channel property
            
            else:
                logger.info(f"Variance check FAILED. Retrying on same data_id {self.data_id}.")
                self.iteration_per_data_id += 1
                self._is_model_updated = False



            self._updates_in_queue -= self._agg_goal
            self._agg_goal_cnt = 0 # Reset for the next batch
            self.fwd_llm_stage = FwdLLMStage(self._round, self.data_id, self.iteration_per_data_id)            
            # ASYNC: Clean up ends that just sent data
            logger.debug("Agg goal reached, so resetting trainer end states in the channel")
            channel.cleanup_recvd_ends()


    def aggregate_grads_from_trainers(self, trainer_grad):
        self.print_trainable_params_stats(
            location="[start,aggregate_grads_from_trainers()]"
        )
        all_zero = all(torch.allclose(g, torch.zeros_like(g)) for g in self.grad)
        logger.info(f"Are all grads zero initially? {all_zero}")

        self.log_memory("start aggregate_grads_from_trainers", self.device)

        # logger.info(f"len(self.model.named_parameters()):
        # {len(self.model.named_parameters())}, len(self.params):
        # {len(self.params)}") self.grad.to(DeviceType.CPU)
        # trainer_grad.to(DeviceType.CPU)
        np = self.model.named_parameters()

        for i, (name, param) in enumerate(np):  # Assuming self.params is a dict
            if param.requires_grad:
                if name in trainer_grad:
                    grad_device = self.grad[i].device
                    trainer_grad[name] = trainer_grad[name].to(grad_device)
                    # Ensure the layer name exists in trainer_grad

                    self.grad[i].add_(trainer_grad[name])
                else:
                    logger.warning(f"Gradient for {name} not found in trainer_grad.")

        self.log_memory("end aggregate_grads_from_trainers", self.device)
        self.print_trainable_params_stats(
            location="[end,aggregate_grads_from_trainers()]"
        )

    def aggregate_grad_pool(self, grad_list):
        self.print_trainable_params_stats(location="[start,aggregate_grad_pool()]")
        if len(grad_list) == 0:
            self.print_trainable_params_stats(location="[end,aggregate_grad_pool()]")
            return None
        if len(grad_list) == 1:
            self.print_trainable_params_stats(location="[end,aggregate_grad_pool()]")
            return grad_list[0]
        else:
            grad = grad_list[0]
            for id, k in enumerate(grad):
                for i in range(0, len(grad_list)):
                    if i == 0:
                        grad[id] = grad_list[i][id]
                    else:
                        grad[id] += grad_list[i][id]
            self.print_trainable_params_stats(location="[end,aggregate_grad_pool()]")
            return grad


    def eval_model(self, epoch=0, global_step=0, device=None):
        if not device:
            device = self.device

        logger.info(f"device inside eval_model() is set to: {device}")
        self.log_memory("start eval_model", self.device)

        results = {}

        eval_loss = 0.0
        nb_eval_steps = 0
        n_batches = len(self.test_global)
        test_sample_len = len(self.test_global.dataset)
        preds = np.empty((test_sample_len, self.num_labels))

        logger.info(
            f"Created n_batches: {n_batches}, test_sample_len: {test_sample_len} and preds.shape: {preds.shape}, location of model: {next(self.model.parameters()).device}"
        )

        out_label_ids = np.empty(test_sample_len)
        # Move model to device before performing the eval
        self.model.to(device)
        self.model.eval()
        self.fmodel, self.params, self.buffers = fc.make_functional_with_buffers(
            self.model
        )

        for i, batch in enumerate(self.test_global):
            with torch.no_grad():
                batch = tuple(t for t in batch)
                x = batch[1].to(device)
                labels = batch[4].to(device)

                output = self.model(x)
                logits = output[0]

                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                eval_loss += loss.item()

            nb_eval_steps += 1
            start_index = self.args.eval_batch_size * i

            end_index = (
                start_index + self.args.eval_batch_size
                if i != (n_batches - 1)
                else test_sample_len
            )
            preds[start_index:end_index] = logits.detach().cpu().numpy()
            out_label_ids[start_index:end_index] = labels.detach().cpu().numpy()

        eval_loss = eval_loss / nb_eval_steps

        model_outputs = preds
        preds = np.argmax(preds, axis=1)
        result, wrong = self.compute_metrics(
            preds, out_label_ids, self.test_global.examples
        )
        result["eval_loss"] = eval_loss
        results.update(result)

        # self.results.update(result)
        logging.info(f"results after eval are: {results}, len(wrong) is: {len(wrong)}, 'data_id_iterations': {self.iteration_per_data_id}")

        # TODO: Check if model needs to be moved back to cpu? Do we need to keep
        # moving the model between CPU and GPU repeatedly?
        del x, labels, output, logits, loss
        torch.cuda.empty_cache()
        gc.collect()

        self.log_memory("end eval_model", self.device)

        return result, model_outputs, wrong

    def compute_metrics(self, preds, labels, eval_examples=None):
        assert len(preds) == len(labels)
        self.log_memory("start compute_metrics", self.device)

        extra_metrics = {}
        extra_metrics["acc"] = sklearn.metrics.accuracy_score(labels, preds)
        mismatched = labels != preds

        if eval_examples:
            wrong = [i for (i, v) in zip(eval_examples, mismatched) if v.any()]
        else:
            wrong = ["NA"]

        mcc = matthews_corrcoef(labels, preds)

        tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()

        self.log_memory("end compute_metrics", self.device)

        return (
            {**{"mcc": mcc, "tp": tp, "tn": tn, "fp": fp, "fn": fn}, **extra_metrics},
            wrong,
        )

    def oracular_trainer_avail_check(self, end: str) -> bool:
        logger.debug("In oracular_trainer_avail_check")

        picked_trainer_is_available = True

        if end in self.trainer_unavail_durations.keys():
            # get aggregator seconds from start
            agg_time_since_start_s = time.time() - self.agg_start_time_ts

            curr_trainer_unavail_list = self.trainer_unavail_durations[end]

            # iterate through unavailability list First, check if the current
            # time is within any failure window

            for start_time, duration in curr_trainer_unavail_list:
                if start_time <= agg_time_since_start_s < start_time + duration:
                    logger.debug(
                        f"### Trainer {end} attempted to be picked in failed " f"state."
                    )
                    picked_trainer_is_available = False
                    return picked_trainer_is_available
                else:
                    logger.debug(f"### Trainer {end} is available.")
                    picked_trainer_is_available = True

            # Remove entries that occurred in the past
            updated_trainer_unavail_list = [
                (start_time, duration)
                for start_time, duration in curr_trainer_unavail_list
                if (start_time + duration) >= agg_time_since_start_s
            ]

            # Remove end from trainer_unavail_durations if list is empty TODO:
            # Check if deletion is happening properly
            if len(updated_trainer_unavail_list) == 0:
                logger.debug(
                    f"### Trainer {end} will no longer fail, removing from "
                    f"trainer_unavail_durations"
                )
                del self.trainer_unavail_durations[end]
            else:
                self.trainer_unavail_durations[end] = updated_trainer_unavail_list
        else:
            logger.info(
                f"No info on end {end} in self.trainer_unavail_durations"
                f", returning TRUE (default)"
            )
        return picked_trainer_is_available

    def hearbeat_trainer_avail_check(self, end: str) -> bool:
        picked_trainer_is_available = True
        last_acceptable_heartbeat_ts = time.time() - (
            self._trainer_max_miss_heartbeats * self._trainer_heartbeat_freq_s
        )

        # return True if: heartbeat was received from trainer and it is within
        # last_acceptable_heartbeat_ts

        # return False if: if end isnt in heartbeat dict, means that the trainer
        # hasn't given a heartbeat in a while and was removed based on
        # last_acceptable_heartbeat_ts

        # NOTE: During agg init, it might have registered a trainer, but not
        # received heartbeat in such a scenario, we return True so that agg is
        # able to send init_weights to trainer and start the training process
        # this is when trainer not in all_trainers and not in dict

        if (end not in self._per_trainer_last_heartbeat_ts.keys()) and (
            end not in self.all_trainers
        ):
            picked_trainer_is_available = True
            logger.info(
                f"Might be trainer init(), trainer {end} hasnt sent any"
                f" heartbeats yet, but we return True"
            )
        elif end not in self._per_trainer_last_heartbeat_ts.keys():
            picked_trainer_is_available = False
            logger.debug(f"Trainer {end} was already marked unavailable")
        elif self._per_trainer_last_heartbeat_ts[end] < last_acceptable_heartbeat_ts:
            del self._per_trainer_last_heartbeat_ts[end]
            picked_trainer_is_available = False
            logger.info(
                f"Trainer {end} missed max_allowed_heartbeats, " f"marked unavailable"
            )
        elif self._per_trainer_last_heartbeat_ts[end] >= last_acceptable_heartbeat_ts:
            picked_trainer_is_available = True
            logger.debug(f"Trainer {end} is available")
        else:
            logger.error(f"Availability check failed, trainer {end}, returning True")

        return picked_trainer_is_available

    def get_unavailable_trainers(self) -> list:
        # Works only for heartbeat based right now TODO: (DG) Extend for other
        # trainer_avail_checks too
        current_unavailable_trainers = [
            end
            for end in self.all_trainers
            if end not in self._per_trainer_last_heartbeat_ts.keys()
        ]
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

    def _distribute_weights(self, tag: str, task_to_perform: str = "train") -> None:
        """
        Distribute a global model in asynchronous FL fashion - for FwdLLM.
        This method actually sends either gradients or calc_more_var to
        trainers, not the actual model weights.

        This method is overridden from one in asynchronous top aggregator
        (..top_aggregator).
        """
        channel = self.cm.get_by_tag(tag)
        if not channel:
            logger.debug(f"channel not found for tag {tag}")
            return
        
        # this call waits for at least one peer to join this channel
        channel.await_join()
        global_model_params = self.get_global_model_params()
        self.weights = global_model_params
        logger.debug(f"Starting busy wait at time {time.time()}")
        time.sleep(0.1)
        logger.debug(f"Ended busy wait at time {time.time()}")
        if self.trainer_event_dict is not None:
            curr_unavail_trainer_list = self.get_curr_unavail_trainers()
            channel.set_curr_unavailable_trainers(
                trainer_unavail_list=curr_unavail_trainer_list
            )
            logger.info(
                f"Passed curr_unavail_trainer_list: "
                f"{curr_unavail_trainer_list} to channel"
            )
        else:
            # Handling the case for oort's selector since it expects 3
            # arguments
            channel.set_curr_unavailable_trainers(trainer_unavail_list=[])


        logger.debug(
            f"Sending weights to trainers with task_to_perform = {task_to_perform}"
        )
        
        # check if there are any ends to send weights to
        ends = channel.ends(VAL_CH_STATE_SEND, task_to_perform)
        logger.info(f"ends: {ends}")
        # TODO: check in agg_weights if ends is None
        if ends is None:
            self.ends_not_selected_yet = True
        else:
            self.ends_not_selected_yet = False
        if not ends:
            logger.debug(
                f"No trainers found for tag {tag}, will "
                f"move to get() for fetch weights from trainers"
            )
            return

        logger.info(f"Distributing tasks to {len(ends)} selected trainers...")

        if self.var:
            logger.info(
                f"self.var = {self.var}, self.var_threshold = {self.var_threshold}"
            )
        if self.var_good_enough == True:
            logger.info(
                "Will send new weights to ends since variance is less than threshold"
            )
        else:
            logger.info(
                "Sending variance = bad to trainers since variance is greater than threshold"
            )
        if self.var_good_enough:
            logger.info(
                "Variance is GOOD. Preparing and sending new model weights and grad_pool."
            )
            self.print_trainable_params_stats(location="[populate_params, _distr_weights]")
            trainable_params = self.get_trainable_param_state_dict()
            shared_weights = weights_to_device(trainable_params, DeviceType.CPU)
            
            shared_grad_pool = self.aggregate_grad_pool(self.grad_pool)

            shared_grad_pool_trainable = []
            if shared_grad_pool is None:
                shared_grad_pool_trainable = None
            else:
                idx = 0
                for param in self.model.parameters():
                    if param.requires_grad:
                        shared_grad_pool_trainable.append(shared_grad_pool[idx].clone())
                    idx += 1
            
            # Clear pools after model has been updated and we move to the next round!
            if self._is_model_updated:
                self.grad_pool = []  # update when model version is updated!
                self.grad_for_var_check_list = []  # Update once agg goal met
                self._is_model_updated = False
            
            payload = {
                MessageType.WEIGHTS: shared_weights,
                MessageType.GRAD_POOL: shared_grad_pool_trainable,
                MessageType.ROUND: self._round,
                MessageType.MODEL_VERSION: self._model_version,
                MessageType.TASK_TO_PERFORM: task_to_perform,
                MessageType.DATA_ID: self.data_id,
                MessageType.ITERATION_PER_DATA_ID: self.iteration_per_data_id,
            }
            # Clean up memory
            # del shared_weights
            # del shared_grad_pool
            # del shared_grad_pool_trainable
            
        else:
            logger.info(
                "Variance is BAD. Sending request for more variance checks."
            )
            payload = {
                MessageType.VAR: "bad",
                MessageType.ROUND: self._round,
                MessageType.MODEL_VERSION: self._model_version,
                MessageType.TASK_TO_PERFORM: task_to_perform,
                MessageType.DATA_ID: self.data_id,
                MessageType.ITERATION_PER_DATA_ID: self.iteration_per_data_id,
            }


        # ASYNC SEND LOOP (from AsyncTopAgg)
        for end in ends:
            logger.info(
                f"Sending payload to {end} with model_version: {self._model_version}, "
                f"data_id: {self.data_id}, iter: {self.iteration_per_data_id}"
            )
            
            # Set OORT property (from AsyncTopAgg)
            channel.set_end_property(
                end, PROP_ROUND_START_TIME, (self._round, datetime.now())
            )
            
            # Send the payload
            channel.send(end, payload)
            
            # Track send time (from AsyncTopAgg)
            if end not in self._track_trainer_version_duration_s.keys():
                self._track_trainer_version_duration_s[end] = {
                    "last_send_wts_ts": -1,
                    "sent_wts_version_ts": {},
                    "recv_wts_version_ts": {},
                    "total_training_time_s": 0, # Initialize to 0
                }
            self._track_trainer_version_duration_s[end]["sent_wts_version_ts"][
                self._model_version
            ] = datetime.now()

        # Clean up the large payload object
        del payload
        gc.collect()

    def compose(self) -> None:
        """Compose role with tasklets."""
        super().compose()
        with CloneComposer(self.composer) as _:
            task_internal_init = Tasklet("internal_init", self.internal_init)

            task_reset_agg_goal_vars = Tasklet(
                "reset_agg_goal_vars", self._reset_agg_goal_variables
            )

            # Created separate put tasklets for train and eval
            task_put_train = Tasklet("distribute", self.put, TAG_DISTRIBUTE, "train")

            task_get_weights = Tasklet("aggregate", self._aggregate_grads_async, TAG_AGGREGATE)

            # task_get_heartbeat = Tasklet("heartbeat", self.get,
            # TAG_HEARTBEAT)
            task_init = Tasklet("initialize", self.initialize)

        c = self.composer
        c.unlink()

        loop = Loop(loop_check_fn=lambda: self._work_done)
        # create a loop object for asyncfl to manage concurrency as
        # well as aggregation goal
        asyncfl_loop = Loop(loop_check_fn=lambda: self._agg_goal_cnt == self._agg_goal)

        # chain them again with new tasklets introduced in this class
        (
            task_internal_init
            >> task_init
            >> loop(
                task_reset_agg_goal_vars 
                # >> asyncfl_loop(task_put >> task_get_weights >>
                >> asyncfl_loop(task_put_train >> task_get_weights)
                >> c.tasklet("analysis")
                >> c.tasklet("save_metrics")
            )
            >> c.tasklet("inform_end_of_training")
        )



    @classmethod
    def get_func_tags(cls) -> list[str]:
        """Return a list of function tags defined in the top level
        aggregator role."""
        return [TAG_DISTRIBUTE, TAG_AGGREGATE, TAG_HEARTBEAT]
