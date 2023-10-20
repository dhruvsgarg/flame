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
"""CIFAR-10 horizontal FL trainer for PyTorch.

The example below is implemented based on the following example from pytorch:
https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html.
"""

import logging

from flame.config import Config
from flame.mode.horizontal.trainer import Trainer

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch import Tensor
import torch.optim as optim
import torch.utils.data as data_utils
from torchvision.datasets import CIFAR10
import time
import calendar

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

        self.device = None
        self.train_loader = None

        self.epochs = self.config.hyperparameters.epochs
        self.batch_size = self.config.hyperparameters.batch_size or 16

        self.criterion = None

        # Setting the indices used by the trainer
        self.trainer_indices_list = self.config.hyperparameters.trainer_indices_list
        # Loading the failure durations for trainers
        self.trainer_start_ts = time.time()
        self.failure_durations_s = self.config.hyperparameters.failure_durations_s
        self.timestamp_next_sleep_s = calendar.timegm(
            time.strptime("Dec 31, 2030 @ 23:59:59 UTC", "%b %d, %Y @ %H:%M:%S UTC")
        )
        if len(self.failure_durations_s) > 0:
            self.timestamp_next_sleep_s = (
                self.trainer_start_ts + self.failure_durations_s[0][0]
            )

    def check_and_sleep(self):
        curr_time = time.time()
        if (curr_time >= self.timestamp_next_sleep_s) and (
            len(self.timestamp_next_sleep_s) > 0
        ):
            # pop leftmost element
            sleep_config_tuple = self.failure_durations_s.pop(0)

            # get the duration of sleep and set the params for next sleep
            sleep_duration_s = sleep_config_tuple[1]
            print("Sleeping for time: ", sleep_duration_s, " at timestamp: ", curr_time)
            time.sleep(sleep_duration_s)
            print("Woke up at timestamp: ", curr_time)

            # check if failure_list is now empty, if yes, reset ts_next_sleep_s
            # if not empty, set it to the next value
            if len(self.failure_durations_s) > 0:
                self.timestamp_next_sleep_s = curr_time + self.failure_durations_s[0][0]
            else:
                self.timestamp_next_sleep_s = calendar.timegm(
                    time.strptime(
                        "Dec 31, 2030 @ 23:59:59 UTC", "%b %d, %Y @ %H:%M:%S UTC"
                    )
                )

        return

    def initialize(self) -> None:
        """Initialize role."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = Net().to(self.device)

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

        print("indices: ", indices)
        dataset = data_utils.Subset(dataset, indices)
        train_kwargs = {
            "batch_size": self.batch_size,
            "drop_last": True,
            "shuffle": True,
            "num_workers": 2,
        }

        self.train_loader = torch.utils.data.DataLoader(dataset, **train_kwargs)

    def train(self) -> None:
        self.check_and_sleep()
        """Train a model."""
        print("~~ inside train, calling train_epoch")
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4
        )

        for epoch in range(1, self.epochs + 1):
            self._train_epoch(epoch)

        # save dataset size so that the info can be shared with aggregator
        self.dataset_size = len(self.train_loader.dataset)

    def _train_epoch(self, epoch):
        self.check_and_sleep()
        print("==== started _train_epoch train()")
        self.model.train()

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.nll_loss(output, target)
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
        print("==== completed _train_epoch train()")

    def evaluate(self) -> None:
        """Evaluate a model."""
        # Implement this if testing is needed in trainer
        pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("config", nargs="?", default="./config.json")

    args = parser.parse_args()
    config = Config(args.config)

    t = PyTorchCifar10Trainer(config)
    t.compose()
    t.run()
