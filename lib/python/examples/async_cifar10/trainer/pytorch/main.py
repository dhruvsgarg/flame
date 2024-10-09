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
"""CIFAR-10 horizontal FL trainer for PyTorch.

The example below is implemented based on the following example from
pytorch:
https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html.
"""

import ast
import calendar
import gc
import logging
import threading
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
import torchvision.transforms as transforms
from flame.config import Config, TrainerAvailabilityStatus
from flame.mode.horizontal.trainer import Trainer
from torchvision.datasets import CIFAR10

logger = logging.getLogger(__name__)


class Net(nn.Module):
    """Net class."""

    def __init__(self):
        """Initialize."""
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.conv3 = nn.Conv2d(128, 256, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        """Forward."""
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


class PyTorchCifar10Trainer(Trainer):
    """PyTorch CIFAR-10 Trainer."""

    def __init__(self, config: Config) -> None:
        """Initialize a class instance."""
        self.config = config
        self.dataset_size = 0
        self.model = None
        # Oort requires its loss function to have 'reduction'
        # parameter
        self.loss_fn = torch.nn.CrossEntropyLoss

        self.device = None
        self.train_loader = None

        self.learning_rate = self.config.hyperparameters.learning_rate
        self.epochs = self.config.hyperparameters.epochs
        self.batch_size = self.config.hyperparameters.batch_size or 16
        self.trainer_id = self.config.task_id

        self.criterion = None

        self.task_to_perform = "train"

        # Enable/disable use of oort_loss fromt he config. Needed for
        # oort and asyncOORT.
        self.use_oort_loss_fn = self.config.hyperparameters.use_oort_loss_fn
        logger.info(f"Trainer: {self.trainer_id} has "
                    f"use_oort_loss_fn: {self.use_oort_loss_fn}")

        # TODO: (DG) Remove the hard requirement for config to include
        # trainer_indices_list and failure_durations_s Setting the
        # indices used by the trainer
        self.trainer_indices_list = self.config.hyperparameters.trainer_indices_list
        # Loading the failure durations for trainers
        self.trainer_start_ts = time.time()
        self.failure_durations_s = ast.literal_eval(
            self.config.hyperparameters.failure_durations_s
        )
        if len(self.failure_durations_s) > 0:
            self.timestamp_next_sleep_s = (
                self.trainer_start_ts + self.failure_durations_s[0][0]
            )
            print(f"# Trainer id: {self.trainer_id}, self.failure_durations_s: "
                  f"{self.failure_durations_s}")
        else:
            self.timestamp_next_sleep_s = calendar.timegm(
                time.strptime("Dec 31, 2030 @ 23:59:59 UTC", "%b %d, %Y @ %H:%M:%S UTC")
                )
            
        # TODO: (DG) Fix the hack later. Creating duplicate data
        # struct for dup_check_and_sleep() which is used by heartbeat
        # thread
        self.dup_failure_durations_s = ast.literal_eval(
            self.config.hyperparameters.failure_durations_s
        )
        if len(self.dup_failure_durations_s) > 0:
            self.dup_timestamp_next_sleep_s = (
                self.trainer_start_ts + self.dup_failure_durations_s[0][0]
            )
            print(f"# Trainer id: {self.trainer_id}, self.dup_failure_durations_s: "
                  f"{self.dup_failure_durations_s}")
        else:
            self.dup_timestamp_next_sleep_s = calendar.timegm(
                time.strptime("Dec 31, 2030 @ 23:59:59 UTC", "%b %d, %Y @ %H:%M:%S UTC")
                )

        # sending heartbeats to aggregator
        if "enabled" in self.config.hyperparameters.heartbeats.keys():
            self.heartbeats_enabled = self.config.hyperparameters.heartbeats["enabled"]
        else:
            self.heartbeats_enabled = False
        
        if "frequency_s" in self.config.hyperparameters.heartbeats.keys():
            self.heartbeats_second_freq = (
                self.config.hyperparameters.heartbeats["frequency_s"]
                )
        else:
            self.heartbeats_second_freq = 99999

        # TODO: (DG) self.timestamp_next_heartbeat_s might not be
        # getting used. Remove? if heartbeats are enabled, compute
        # first heartbeat time
        if self.heartbeats_enabled is True:
            self.timestamp_next_heartbeat_s = (
                self.trainer_start_ts + self.heartbeats_second_freq
            )
        else:
            self.timestamp_next_heartbeat_s = calendar.timegm(
                time.strptime("Dec 31, 2030 @ 23:59:59 UTC", "%b %d, %Y @ %H:%M:%S UTC")
                )
            
        # Check if client will notify aggregator of its availability
        self.client_avail_aware_notify = (
            self.config.hyperparameters.client_avail_aware_notify
        )

        # Check if client will emulate delays in training time
        self.training_delay_enabled = self.config.hyperparameters.training_delay_enabled
        self.training_delay_s = float(self.config.hyperparameters.training_delay_s)

        #NRL: added new code for availability_status_updates
        self.availability_status_updates = ast.literal_eval(
            self.config.hyperparameters.availability_status_updates
        )
        logger.info(f"NRL: availability_status_updates for trainer id {self.trainer_id} = {self.availability_status_updates}")

        self.availability_status = TrainerAvailabilityStatus.AVL_TRAIN

        #flag to flip between old logic (avl/unavl state) and new logic(avl_to_train/eval/unavl)
        self.check_availability_status = self.config.hyperparameters.check_availability_status

        #flag to decide whether the trainer upon unavailability will wait or exit
        self.wait_to_become_available = self.config.hyperparameters.wait_to_become_available
    
    def check_and_sleep(self):
        """Induce transient unavailability"""
        # Implement this if transient unavailability need to be
        # emulated in the trainer
        
        if (time.time() >= self.timestamp_next_sleep_s) and (
            len(self.failure_durations_s) > 0
        ):
            # pop leftmost element
            sleep_config_tuple = self.failure_durations_s.pop(0)

            # get the duration of sleep and set the params for next
            # sleep
            sleep_start_ts_from_trainer_init = (
                self.trainer_start_ts + sleep_config_tuple[0]
            )
            sleep_duration_s = sleep_config_tuple[1]

            # remaining sleep = trainer_start_ts + actual_sleep_start
            # + actual_sleep_duration - current_ts
            remaining_sleep_duration_s = (
                sleep_start_ts_from_trainer_init + sleep_duration_s - time.time()
            )
            logger.debug(f"Task_id: {self.trainer_id} given_sleep_duration_s: "
                         f"{sleep_duration_s} with remaining_sleep_duration_s: "
                         f"{remaining_sleep_duration_s} at timestamp: {time.time()}")
            
            if (remaining_sleep_duration_s <= 0):
                logger.info(f"Task_id: {self.trainer_id} got -ve remaining sleep "
                            f"at timestamp: {time.time()}")
                # Need to pop out failure intervals that occur in the
                # past
                time_elapsed_from_start = time.time() - self.trainer_start_ts
                while len(self.failure_durations_s) > 0 and (
                    time_elapsed_from_start > (
                        self.failure_durations_s[0][0] + self.failure_durations_s[0][1]
                        )
                ):
                    self.failure_durations_s.pop(0)
                    if len(self.failure_durations_s) == 0:
                        break
            else:
                # sleep for remaining time
                logger.info(f"Task_id: {self.trainer_id} going to sleep "
                            f"at timestamp: {time.time()}")
                time.sleep(remaining_sleep_duration_s)
                logger.info(f"Task_id: {self.trainer_id} woke up at timestamp: "
                            f"{time.time()}")

            # check if failure_list is now empty, if yes, reset
            # ts_next_sleep_s if not empty, set it to the next value
            if len(self.failure_durations_s) > 0:
                self.timestamp_next_sleep_s = max((
                    self.trainer_start_ts + self.failure_durations_s[0][0]),
                    time.time()+1
                    )
                if self.timestamp_next_sleep_s < time.time():
                    logger.error(f"Task_id: {self.trainer_id} ERROR - JUST SET NEXT "
                                 f"self.timestamp_next_sleep_s < time.time()")
            else:
                self.timestamp_next_sleep_s = calendar.timegm(
                    time.strptime(
                        "Dec 31, 2030 @ 23:59:59 UTC", "%b %d, %Y @ %H:%M:%S UTC"
                    )
                )
                logger.info(f"Task_id: {self.trainer_id} no more sleep for trainer")

        logger.debug(f"Task_id: {self.trainer_id} check_and_sleep completed at "
                     f"timestamp: {time.time()}")

    def dup_check_and_sleep(self):
        
        if (time.time() >= self.dup_timestamp_next_sleep_s) and (
            len(self.dup_failure_durations_s) > 0
        ):
            # pop leftmost element
            sleep_config_tuple = self.dup_failure_durations_s.pop(0)

            # get the duration of sleep and set the params for next
            # sleep
            sleep_start_ts_from_trainer_init = (
                self.trainer_start_ts + sleep_config_tuple[0]
            )
            sleep_duration_s = sleep_config_tuple[1]

            # remaining sleep = trainer_start_ts + actual_sleep_start
            # + actual_sleep_duration - current_ts 
            remaining_sleep_duration_s = (
                sleep_start_ts_from_trainer_init + sleep_duration_s - time.time()
            )
            logger.debug(f"Task_id: {self.trainer_id} given_sleep_duration_s: "
                         f"{sleep_duration_s} with remaining_sleep_duration_s: "
                         f"{remaining_sleep_duration_s} at timestamp: {time.time()}")
            
            if (remaining_sleep_duration_s <= 0):
                logger.info(f"Task_id: {self.trainer_id} got -ve remaining sleep at "
                            f"timestamp: {time.time()}")
                # Need to pop out failure intervals that occur in the
                # past
                time_elapsed_from_start = time.time() - self.trainer_start_ts
                while len(self.dup_failure_durations_s) > 0 and (
                    time_elapsed_from_start > (
                        self.dup_failure_durations_s[0][0] +
                        self.dup_failure_durations_s[0][1]
                        )
                ):
                    self.dup_failure_durations_s.pop(0)
                    if len(self.dup_failure_durations_s) == 0:
                        break
            else:
                # sleep for remaining time
                logger.info(f"Task_id: {self.trainer_id} going to sleep "
                            f"at timestamp: {time.time()}")
                time.sleep(remaining_sleep_duration_s)
                logger.info(f"Task_id: {self.trainer_id} woke up at timestamp: "
                            f"{time.time()}")

            # check if failure_list is now empty, if yes, reset
            # ts_next_sleep_s if not empty, set it to the next value
            if len(self.dup_failure_durations_s) > 0:
                self.dup_timestamp_next_sleep_s = max((
                    self.trainer_start_ts + self.dup_failure_durations_s[0][0]),
                    time.time()+1
                    )
                if (self.dup_timestamp_next_sleep_s < time.time()):
                    logger.error(f"Task_id: {self.trainer_id} ERROR - JUST SET NEXT "
                                 f"self.dup_timestamp_next_sleep_s < time.time()")
            else:
                self.dup_timestamp_next_sleep_s = calendar.timegm(
                    time.strptime(
                        "Dec 31, 2030 @ 23:59:59 UTC", "%b %d, %Y @ %H:%M:%S UTC"
                    )
                )
                logger.info(f"Task_id: {self.trainer_id} no more sleep for trainer")

        logger.debug(f"Task_id: {self.trainer_id} dup_check_and_sleep completed at "
                     f"timestamp: {time.time()}")
    
    def check_leave_sleep_join(self):
        """Indicate transient unavailability to aggregator"""
        # NOTE: Builds on top of dup_check_and_sleep, both cannot be
        # used together
        
        if (time.time() >= self.dup_timestamp_next_sleep_s) and (
            len(self.dup_failure_durations_s) > 0
        ):
            # pop leftmost element
            sleep_config_tuple = self.dup_failure_durations_s.pop(0)

            # get the duration of sleep and set the params for next
            # sleep
            sleep_start_ts_from_trainer_init = (
                self.trainer_start_ts + sleep_config_tuple[0]
                )
            sleep_duration_s = sleep_config_tuple[1]

            # remaining sleep = trainer_start_ts + actual_sleep_start
            # + actual_sleep_duration - current_ts
            remaining_sleep_duration_s = (
                sleep_start_ts_from_trainer_init + sleep_duration_s - time.time()
            )
            logger.debug(f"Task_id: {self.trainer_id} given_sleep_duration_s: "
                         f"{sleep_duration_s} with remaining_sleep_duration_s: "
                         f"{remaining_sleep_duration_s}"
                         f" at timestamp: {time.time()}"
                         )
            
            if remaining_sleep_duration_s < 0:
                logger.info(f"Task_id: {self.trainer_id} got -ve remaining sleep "
                            f"at timestamp: {time.time()}")
                # Need to pop out failure intervals that occur in the
                # past
                while len(self.dup_failure_durations_s) > 0 and (
                    (time.time() - self.trainer_start_ts) > (
                        self.dup_failure_durations_s[0][0] +
                        self.dup_failure_durations_s[0][1]
                        )
                ):
                    self.dup_failure_durations_s.pop(0)
                    if len(self.dup_failure_durations_s) == 0:
                        break
            else:
                # leave channel, if notify is enabled
                if self.client_avail_aware_notify == "True":
                    self._perform_channel_leave(tag="upload")

                # sleep for remaining time
                logger.info(f"Task_id: {self.trainer_id} going to sleep "
                            f"at timestamp: {time.time()}")
                time.sleep(remaining_sleep_duration_s)
                logger.info(f"Task_id: {self.trainer_id} woke up at timestamp: "
                            f"{time.time()}")

                # join channel, if notify is enabled
                if self.client_avail_aware_notify == "True":
                    self._perform_channel_join(tag="upload")

            # check if failure_list is now empty, if yes, reset
            # ts_next_sleep_s if not empty, set it to the next value
            if len(self.dup_failure_durations_s) > 0:
                self.dup_timestamp_next_sleep_s = max((
                    self.trainer_start_ts + self.dup_failure_durations_s[0][0]),
                    time.time()+1
                    )
                if (self.dup_timestamp_next_sleep_s < time.time()):
                    logger.error(f"Task_id: {self.trainer_id} ERROR - JUST SET NEXT "
                                 f"self.dup_timestamp_next_sleep_s < time.time()")
            else:
                self.dup_timestamp_next_sleep_s = calendar.timegm(
                    time.strptime(
                        "Dec 31, 2030 @ 23:59:59 UTC", "%b %d, %Y @ %H:%M:%S UTC"
                    )
                )
                logger.info(f"Task_id: {self.trainer_id} no more sleep for trainer")

        logger.debug(f"Task_id: {self.trainer_id} check_leave_sleep_join completed at "
                     f"timestamp: {time.time()}")

    def check_and_update_availability_status(self):
       if len(self.availability_status_updates) > 0 and time.time() >= self.trainer_start_ts + self.availability_status_updates[0][0]:
           status_to_set = self.availability_status_updates.pop(0)[1]
           old_status = self.availability_status.value
           try:
               self.availability_status = TrainerAvailabilityStatus(status_to_set)
           except ValueError:
               logger.error(f"NRL: Invalid status encountered: {status_to_set}. Retaining old status {old_status}.")
               return           
           new_status = self.availability_status.value
           logger.info(f"NRL: Changed the availability status of trainer {self.trainer_id} from {old_status} to {new_status}. Current list = {self.availability_status_updates}")
        #    self.send_availability_status("upload")
           self._perform_channel_state_update(tag="upload", state=self.availability_status, timestamp=str(time.time()))

    def initialize(self) -> None:
        """Initialize role."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = Net().to(self.device)
        logger.debug(f"Task_id: {self.trainer_id} initialize completed at timestamp: "
                     f"{time.time()}")

    def load_data(self) -> None:
        """Load data."""
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        dataset = CIFAR10(
            "./data", train=True, download=True, transform=transform_train
        )

        # create indices into a list and convert to tensor
        indices = torch.tensor(self.trainer_indices_list)

        dataset = data_utils.Subset(dataset, indices)
        train_kwargs = {
            "batch_size": self.batch_size,
            "drop_last": True,
            "shuffle": True,
            "num_workers": 2,
        }

        self.train_loader = torch.utils.data.DataLoader(dataset, **train_kwargs)

        # Release the memory of the full dataset
        del dataset
        gc.collect()

        logger.debug(f"Task_id: {self.trainer_id} load_data completed at timestamp: "
                     f"{time.time()}")

    def train(self) -> None:
        logger.info(f"NRL: Entered train method for {self.trainer_id}")
        if self.task_to_perform != "train":
            logger.info(f"Trainer {self.trainer_id} is not required to train")
            return
        if self.check_availability_status == "True" and self.availability_status != TrainerAvailabilityStatus.AVL_TRAIN:
            if self.wait_to_become_available == "True":
                logger.error(f"NRL: Trainer id {self.trainer_id} is not available to train. Waiting for it to be available")
                while self.availability_status != TrainerAvailabilityStatus.AVL_TRAIN:
                    time.sleep(0.1)
            else:
                logger.error(f"NRL: Trainer id {self.trainer_id} is not available to train. Exiting.")
                return
        
        logger.info(f"NRL: Trainer {self.trainer_id} available to train")
        
        """Train a model."""
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.learning_rate
        )

        # reset stat utility for OORT
        self.reset_stat_utility()

        for epoch in range(1, self.epochs + 1):
            self._train_epoch(epoch)

        # save dataset size so that the info can be shared with
        # aggregator
        self.dataset_size = len(self.train_loader.dataset)

        # emulate delays in training (due to compute resource and/or
        # dataset size and/or network latency) if enabled
        if self.training_delay_enabled == "True":
            time.sleep(self.training_delay_s)
            logger.debug(f"Delayed training time for trainer "
                         f"{self.trainer_id} by {self.training_delay_s}s")

    def _train_epoch(self, epoch):
        self.model.train()

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            
            if self.use_oort_loss_fn == "False":
                # Loss function to use with Fedbuff
                loss = F.nll_loss(output, target)
            elif self.use_oort_loss_fn == "True":
                # Calculate statistical utility of a trainer while
                # calculating loss
                loss = self.oort_loss(
                    output, target.squeeze(), epoch, batch_idx)

            loss.backward()
            self.optimizer.step()
            if batch_idx % 100 == 0:
                done = batch_idx * len(data)
                total = len(self.train_loader.dataset)
                percent = 100.0 * batch_idx / len(self.train_loader)
                logger.info(
                    f"epoch: {epoch} [{done}/{total} ({percent:.0f}%)]"
                    f"\tloss: {loss.item():.6f}"
                )
            
        # normalize statistical utility of a trainer based on the size
        # of the dataset
        self.normalize_stat_utility(epoch)

    def evaluate(self) -> None:
        """Evaluate a model."""
        # Implement only forward pass evaluate if the trainer is available to train or to evaluate
        #Evaluate after train is written in the train_epoch method itself 

        if self.wait_to_become_available == "True" and self.check_availability_status and self.availability_status == TrainerAvailabilityStatus.UNAVL:
            logger.warning(f"NRL: Trainer id {self.trainer_id} is not available to perform forward pass evaluate. Waiting for it to be available")
            while self.availability_status == TrainerAvailabilityStatus.UNAVAILABLE:
                time.sleep(0.1)

        if self.task_to_perform == "evaluate" and (self.check_availability_status == "True" and self.availability_status != TrainerAvailabilityStatus.UNAVL):
            for epoch in range(1, self.epochs + 1):
                for batch_idx, (data, target) in enumerate(self.train_loader):
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    
                    if self.use_oort_loss_fn == "False":
                        # Loss function to use with Fedbuff
                        loss = F.nll_loss(output, target)
                    elif self.use_oort_loss_fn == "True":
                        # Calculate statistical utility of a trainer while
                        # calculating loss
                        loss = self.oort_loss(
                            output, target.squeeze(), epoch, batch_idx)
                    if batch_idx % 100 == 0:
                        done = batch_idx * len(data)
                        total = len(self.train_loader.dataset)
                        percent = 100.0 * batch_idx / len(self.train_loader)
                        logger.info(
                            f"epoch: {epoch} [{done}/{total} ({percent:.0f}%)]"
                            f"\tloss: {loss.item():.6f}"
                        )
                    
                # normalize statistical utility of a trainer based on the size
                # of the dataset
                self.normalize_stat_utility(epoch)

        else:
            logger.warning(f"Evaluate (forward pass) will not be run for trainer id {self.trainer_id}. task_to_perform = {self.task_to_perform} and trainer availability_status = {self.availability_status.value} and wait_to_become_available = {self.wait_to_become_available}")


    def initiate_heartbeat(self) -> None:
        while True:
            # heartbeats are sent from a different thread. Ideally
            # heartbeats and sleep should have happened on the same
            # thread but in the current scenario, both threads need to
            # be put to sleep whenever the trainer is marked to be
            # unavailable.

            # issue: if i use check_and_sleep here as well, it will
            # modify existing data struct HACK: duplicate
            # check_and_sleep as dup_check_and_sleep and operate on a
            # duplicate data structure
            
            # TODO: DG Need to fix that arg isnt being used to
            # enable/disable this
            time.sleep(self.heartbeats_second_freq)
            self.dup_check_and_sleep()
            logger.debug("Initiating send heartbeat to aggregator")
            self.send_heartbeat_to_agg()

    def notify_trainer_avail(self) -> None:
        while True:
            # Adopted from initiate heartbeats
            
            time.sleep(0.1)             # Will check every 0.1 second
            if self.check_availability_status == "True":
                self.check_and_update_availability_status()
            else:
                self.check_leave_sleep_join()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("config", nargs="?", default="./config.json")

    args = parser.parse_args()
    config = Config(args.config)

    t = PyTorchCifar10Trainer(config)
    print(f"# Trainer id: {t.trainer_id}, has heartbeats_enabled: "
          f"{t.heartbeats_enabled}, has client_avail_aware_notify: "
          f"{t.client_avail_aware_notify}, has "
          f"training_delay_enabled: {t.training_delay_enabled}, "
          f"with training_delay_s: {t.training_delay_s}")

    if t.heartbeats_enabled == "True":
        logger.info(f"Will initiate thread to send heartbeats for "
                    f"trainer {t.trainer_id}")
        heartbeat_thread = threading.Thread(target=t.initiate_heartbeat)
        heartbeat_thread.daemon = True
        heartbeat_thread.start()
    elif t.client_avail_aware_notify == "True":
        logger.info(f"Will initiate thread to send avail notifications for "
                    f"trainer {t.trainer_id}")
        avail_notify_thread = threading.Thread(target=t.notify_trainer_avail)
        avail_notify_thread.daemon = True
        avail_notify_thread.start()

    t.compose()
    t.run()


if __name__ == "__main__":
    main()