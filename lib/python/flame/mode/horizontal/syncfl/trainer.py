# Copyright 2022 Cisco Systems, Inc. and its affiliates
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
"""horizontal FL trainer."""
import inspect
import logging
import math
import time

import torch
from flame.channel import VAL_CH_STATE_HTBT_SEND, VAL_CH_STATE_RECV, VAL_CH_STATE_SEND
from flame.channel_manager import ChannelManager
from flame.common.constants import DeviceType
from flame.common.custom_abcmeta import ABCMeta, abstract_attribute
from flame.common.util import (
    MLFramework,
    delta_weights_pytorch,
    delta_weights_tensorflow,
    get_ml_framework_in_use,
    valid_frameworks,
    weights_to_device,
    weights_to_model_device,
)
from flame.config import Config
from flame.datasamplers import datasampler_provider
from flame.mode.composer import Composer
from flame.mode.message import MessageType
from flame.mode.role import Role
from flame.mode.tasklet import Loop, Tasklet
from flame.optimizers import optimizer_provider
from flame.privacies import privacy_provider
from flame.registries import registry_provider

# TODO: (DG) torch is needed for asyncoort in oort_loss() function,
# but need to comment / uncomment based on the backend used. If it is
# commented, Flame can detect and use either of the backends. But if
# torch code is uncommented, it will be used and will not work for
# trainers wanting to use backends like tensorflow.


logger = logging.getLogger(__name__)

TAG_FETCH = "fetch"
TAG_UPLOAD = "upload"
TAG_HEARTBEAT = "heartbeat_send"


class Trainer(Role, metaclass=ABCMeta):
    """Trainer implements an ML training role."""

    @abstract_attribute
    def loss_fn(self):
        # Added for OORT
        """Abstract attribute for loss function."""
        
    def config(self) -> Config:
        """Abstract attribute for config object."""

    @abstract_attribute
    def model(self):
        """Abstract attribute for model object."""

    @abstract_attribute
    def dataset_size(self):
        """Abstract attribute for size of dataset used to train."""

    def internal_init(self) -> None:
        """Initialize internal state for role."""
        self.cm = ChannelManager()
        self.cm(self.config)
        self.cm.join_all()

        self.registry_client = registry_provider.get(self.config.registry.sort)
        # initialize registry client
        self.registry_client(self.config)

        self.registry_client.setup_run()
        self.metrics = dict()

        # needed for trainer-side optimization algorithms such as
        # fedprox
        temp_opt = optimizer_provider.get(
            self.config.optimizer.sort, **self.config.optimizer.kwargs
        )
        self.regularizer = temp_opt.regularizer

        self.datasampler = datasampler_provider.get(
            self.config.datasampler.sort, **self.config.datasampler.kwargs
        ).trainer_data_sampler

        self.privacy = privacy_provider.get(
            self.config.privacy.sort, **self.config.privacy.kwargs
        )

        self._round = 1
        self._work_done = False

        self.framework = get_ml_framework_in_use()
        if self.framework == MLFramework.UNKNOWN:
            raise NotImplementedError(
                "supported ml framework not found; "
                f"supported frameworks are: {valid_frameworks}"
            )

        if self.framework == MLFramework.PYTORCH:
            self._delta_weights_fn = delta_weights_pytorch

        elif self.framework == MLFramework.TENSORFLOW:
            self._delta_weights_fn = delta_weights_tensorflow

        self.fetch_success = False

        self.trainer_id = self.config.task_id

        # for tracking trainer round progress and checking before
        # sending updates
        self._updates_returned_upto_round = 0
        self._trainer_online_channel_status = True

    def get(self, tag: str) -> None:
        """Get data from remote role(s)."""
        if tag == TAG_FETCH:
            self._fetch_weights(tag)

    def _fetch_weights(self, tag: str) -> None:
        logger.debug(f"### FETCH WEIGHTS start for tag: {tag} "
                     f"and trainer_id {self.trainer_id}")

        self.fetch_success = False
        channel = self.cm.get_by_tag(tag)
        if not channel:
            logger.info(f"fetch weights, channel not found with tag {tag} "
                        f"for trainer_id {self.trainer_id}")
            # we don't want to keep calling this too fast so let's
            # sleep 1 second
            time.sleep(1)
            return

        # this call waits for at least one peer joins this channel
        logger.debug(f"_fetch_weights: waiting for someone to join channel: {channel} "
                     f"for trainer_id {self.trainer_id}")
        channel.await_join()

        # one aggregator is sufficient
        end = channel.one_end(VAL_CH_STATE_RECV)
        msg, _ = channel.recv(end)

        if not msg:
            logger.debug(f"NO msg received for trainer_id {self.trainer_id}")
            if self._work_done:
                # when the work is done, we cancel continue condition
                # (i.e., we set fetch_success to True)
                self.fetch_success = True
            # we don't want to keep calling this too fast so let's
            # sleep 1 second
            time.sleep(1)
            return

        logger.debug(f"New message received for trainer_id {self.trainer_id}")

        if MessageType.ROUND in msg:
            self._round = msg[MessageType.ROUND]

        if MessageType.WEIGHTS in msg:
            # Before proceeding, check if this model version is newer
            # than previously processed NOTE: The condition could have
            # been round <= updates_retuned. But there are scenarios
            # where the channel.leave() executes before the aggregator
            # processes the weight update. Hence, with <= condition,
            # the trainer would never make progress. We allow to
            # trainer to re-train for == round condition if the
            # message was dropped.
            if self._round <= self._updates_returned_upto_round:
                logger.info(f"Fetch weights aborted for given model version "
                            f"{self._round} while trainer has already sent updates "
                            f"upto round: {self._updates_returned_upto_round}")
                
                # Received old data but still allow aggregator cleanup
                # state to occur so as to receive the next update
                logger.info("Cleaning up recvd ends to allow fetch from aggregator "
                            "again and returning from function")
                channel._selector.ordered_updates_recv_ends.append(end)
                logger.debug(f"After appending {end} to ordered_updates_recv_ends: "
                             f"{channel._selector.ordered_updates_recv_ends}")
                channel.cleanup_recvd_ends()
                return

            self.weights = weights_to_model_device(msg[MessageType.WEIGHTS], self.model)
            self._update_model()

        if MessageType.EOT in msg:
            self._work_done = msg[MessageType.EOT]

        if MessageType.DATASAMPLER_METADATA in msg:
            self.datasampler.handle_metadata_from_aggregator(
                msg[MessageType.DATASAMPLER_METADATA]
            )

        self.fetch_success = True

        logger.info(f"### FETCH WEIGHTS complete for trainer_id {self.trainer_id}, "
                    f"round: {self._round} and work_done: {self._work_done} ###")

        logger.debug("Model weights received, so resetting aggregator end states in "
                     "the channel")
        
        channel._selector.ordered_updates_recv_ends.append(end)
        logger.debug(f"After appending {end} to ordered_updates_recv_ends: "
                     f"{channel._selector.ordered_updates_recv_ends}")
        
        channel.cleanup_recvd_ends()

    def put(self, tag: str) -> None:
        """Set data to remote role(s)."""
        if tag == TAG_UPLOAD:
            self._send_weights(tag)
        elif tag == TAG_HEARTBEAT:
            self._send_heartbeat_to_agg(tag)

    def _send_heartbeat_to_agg(self, tag: str) -> None:
        logger.debug(f"### SEND heartbeat for tag: {tag} "
                     f"and trainer_id: {self.trainer_id}")
        channel = self.cm.get_by_tag(tag)
        if not channel:
            logger.debug(f"[_send_heartbeat] channel not found with {tag}")
            return
        
        # this call waits for at least one peer to join this channel
        logger.debug(f"_send_heartbeat: waiting for someone to join channel: {channel} "
                     f"for trainer_id: {self.trainer_id}")
        channel.await_join()

        # one aggregator is sufficient
        end = channel.one_end(VAL_CH_STATE_HTBT_SEND)

        msg = {
            MessageType.HEARTBEAT: time.time(),
        }
        channel.send(end, msg)
        logger.info(f"sending heartbeat done for trainer_id: {self.trainer_id}")

        return

    def _send_weights(self, tag: str) -> None:
        logger.debug(f"### SEND WEIGHTS for tag: {tag} "
                     f"and trainer_id: {self.trainer_id}")
        channel = self.cm.get_by_tag(tag)
        if not channel:
            logger.debug(f"[_send_weights] channel not found with {tag}")
            return

        # this call waits for at least one peer to join this channel
        logger.debug(f"_send_weights: waiting for someone to join channel: {channel} "
                     f"for trainer_id: {self.trainer_id}")
        channel.await_join()

        # one aggregator is sufficient
        end = channel.one_end(VAL_CH_STATE_SEND)

        self._update_weights()

        delta_weights = self._delta_weights_fn(self.weights, self.prev_weights)

        delta_weights = self.privacy.apply_dp_fn(delta_weights)

        self.regularizer.update()

        # NOTE: Also sending stat_utility for OORT
        msg = {
            MessageType.WEIGHTS: weights_to_device(delta_weights, DeviceType.CPU),
            MessageType.DATASET_SIZE: self.dataset_size,
            MessageType.MODEL_VERSION: self._round,
            MessageType.DATASAMPLER_METADATA: self.datasampler.get_metadata(),
            MessageType.STAT_UTILITY: self._stat_utility,
        }

        # Do not proceed with sending of sending of weights if the
        # client-avail-notify thread has left the channel. Wait for it
        # to come back online before sending an update.
        send_start_time = time.time()
        while not self._trainer_online_channel_status:
            time.sleep(0.1)
            send_wait_time = math.ceil(time.time() - send_start_time)
            if send_wait_time % 5 == 0:
                logger.debug(f"Waiting for channel to join before send "
                             f"since: {send_wait_time} seconds")

        channel.send(end, msg)

        self._updates_returned_upto_round = self._round

        logger.info(f"sending weights done for trainer_id: {self.trainer_id} "
                    f"and _updates_returned_upto_round "
                    f"{self._updates_returned_upto_round}")

        channel._selector._cleanup_send_ends()

    def _perform_channel_leave(self, tag: str) -> None:
        logger.debug(f"In _perform_channel_leave for tag: {tag} "
                     f"and trainer_id: {self.trainer_id}")
        channel = self.cm.get_by_tag(tag)
        if not channel:
            logger.debug(f"[_perform_channel_leave] channel not found with {tag}")
            return

        # this call waits for at least one peer to join this channel
        logger.debug(f"_perform_channel_leave: waiting for someone to join channel: "
                     f"{channel} for trainer_id: {self.trainer_id}")
        channel.await_join()

        # Setting the channel status to False. Means that trainer
        # should not send updates during this time.
        self._trainer_online_channel_status = False

        channel.leave()
        logger.info(f"Sent channel leave message for channel: "
                    f"{channel._name} and trainer: {self.trainer_id}."
                    f" Set trainer_online_channel_status: "
                    f"{self._trainer_online_channel_status}")
        
    def _perform_channel_join(self, tag: str) -> None:
        logger.debug(f"In _perform_channel_join for tag: {tag} "
                     f"and trainer_id: {self.trainer_id}")
        channel = self.cm.get_by_tag(tag)
        if not channel:
            logger.debug(f"[_perform_channel_join] channel not found with {tag}")
            return

        # this call waits for at least one peer to join this channel
        logger.debug(f"_perform_channel_join: waiting for someone to join channel: "
                     f"{channel} for trainer_id: {self.trainer_id}")
        channel.await_join()

        channel.join()

        # Setting the channel status to True. Means that trainer can
        # now resume sending updates.
        self._trainer_online_channel_status = True

        logger.info(f"Sent channel join message for channel: "
                    f"{channel._name} and trainer: {self.trainer_id}."
                    f" Set trainer_online_channel_status: "
                    f"{self._trainer_online_channel_status}")

    def save_metrics(self):
        """Save metrics in a model registry."""
        # update self.metrics with metrics from MetricCollector
        # instance
        self.metrics = self.metrics | self.mc.get()
        self.mc.clear()
        logger.debug(f"saving metrics: {self.metrics}")
        if self.metrics:
            self.registry_client.save_metrics(self._round - 1, self.metrics)
            logger.debug("saving metrics done")
        self.metrics = dict()

    def update_metrics(self, metrics: dict[str, float]):
        """Update metrics."""
        self.metrics = self.metrics | metrics

    def _update_model(self):
        if self.framework == MLFramework.PYTORCH:
            self.model.load_state_dict(self.weights)
        elif self.framework == MLFramework.TENSORFLOW:
            self.model.set_weights(self.weights)

    def _update_weights(self):
        # save weights before updating it
        self.prev_weights = self.weights

        if self.framework == MLFramework.PYTORCH:
            self.weights = self.model.state_dict()
        elif self.framework == MLFramework.TENSORFLOW:
            self.weights = self.model.get_weights()
    
    def send_heartbeat_to_agg(self) -> None:
        logger.debug("Inside trainer.py will call self.put(heartbeat)")
        self.put(TAG_HEARTBEAT)

    # #### ADDED OORT RELATED FUNCTIONALITY
    def init_oort_variables(self) -> None:
        """Initialize Oort variables."""
        self._stat_utility = 0

        if "reduction" not in inspect.signature(self.loss_fn).parameters:
            msg = "Parameter 'reduction' not found in loss function "
            msg += f"'{self.loss_fn.__name__}', which is required for Oort"
            raise TypeError(msg)

    # TODO: Enable this in trainer code using a flag based on selector
    # used. Needs to also pass to trainer/main.py
    def oort_loss(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        epoch: int,
        batch_idx: int,
        **kwargs,
    ) -> torch.Tensor:
        """
        Measure the loss of a trainer during training. The trainer's
        statistical utility is measured at epoch 1.
        """
        if epoch == 1 and batch_idx == 0:
            if "reduction" in kwargs.keys():
                reduction = kwargs["reduction"]
            else:
                reduction = "mean"  # default reduction policy is mean
            kwargs_wo_reduction = {
                key: value for key, value in kwargs.items() if key != "reduction"
            }

            criterion = self.loss_fn(reduction="none", **kwargs_wo_reduction)
            loss_list = criterion(output, target)
            self._stat_utility += torch.square(loss_list).sum()

            if reduction == "mean":
                loss = loss_list.mean()
            elif reduction == "sum":
                loss = loss_list.sum()
        else:
            criterion = self.loss_fn(**kwargs)
            loss = criterion(output, target)

        return loss

    def normalize_stat_utility(self, epoch) -> None:
        """
        Normalize statistical utility of a trainer based on the size
        of the trainer's datset, at epoch 1.
        """
        if epoch == 1:
            self._stat_utility = len(self.train_loader.dataset) * math.sqrt(
                self._stat_utility / len(self.train_loader.dataset)
            )
        else:
            return

    def reset_stat_utility(self) -> None:
        """Reset the trainer's statistical utility to zero."""
        self._stat_utility = 0

    def compose(self) -> None:
        """Compose role with tasklets."""
        with Composer() as composer:
            self.composer = composer

            task_internal_init = Tasklet("internal_init", self.internal_init)

            task_init_oort_variables = Tasklet(
                "init_oort_variables",
                self.init_oort_variables
                )

            task_load_data = Tasklet("load_data", self.load_data)

            task_init = Tasklet("init", self.initialize)

            task_get = Tasklet("fetch", self.get, TAG_FETCH)
            task_get.set_continue_fn(cont_fn=lambda: not self.fetch_success)

            task_sleep_after_get = Tasklet("sleep_after_get", self.check_and_sleep)

            task_sleep_after_train = Tasklet("sleep_after_train", self.check_and_sleep)

            task_sleep_after_eval = Tasklet("sleep_after_eval", self.check_and_sleep)

            task_sleep_after_put_weight = Tasklet("sleep_after_put_weight",
                                                  self.check_and_sleep)

            task_sleep_after_save_metrics = Tasklet("sleep_after_save_metrics",
                                                    self.check_and_sleep)

            task_train = Tasklet("train", self.train)

            task_eval = Tasklet("evaluate", self.evaluate)

            task_put_weight = Tasklet("upload", self.put, TAG_UPLOAD)

            task_save_metrics = Tasklet("save_metrics", self.save_metrics)

            # create a loop object with loop exit condition function
            loop = Loop(loop_check_fn=lambda: self._work_done)

            # Now start the rest of the tasks
            (
                task_internal_init
                # >> task_init_oort_variables
                >> task_load_data
                >> task_init
                >> loop(
                    task_get >> task_sleep_after_get >> task_train >>
                    task_sleep_after_train >> task_eval >> task_sleep_after_eval >>
                    task_put_weight >> task_sleep_after_put_weight >>
                    task_save_metrics >> task_sleep_after_save_metrics
                )
            )

    def run(self) -> None:
        """Run role."""
        self.composer.run()

    @classmethod
    def get_func_tags(cls) -> list[str]:
        """Return a list of function tags defined in the trainer
        role."""
        return [TAG_FETCH, TAG_UPLOAD, TAG_HEARTBEAT]
