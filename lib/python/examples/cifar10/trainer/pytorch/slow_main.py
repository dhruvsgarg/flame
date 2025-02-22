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
from flame.mode.horizontal.slow_trainer import SlowTrainer

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch import Tensor
import torch.optim as optim
import torch.utils.data as data_utils
from torchvision.datasets import CIFAR10

logger = logging.getLogger(__name__)


class Net(nn.Module):
    """Net class."""

    def __init__(self):
        """Initialize."""
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        """Forward."""
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class PyTorchCifar10Trainer(SlowTrainer):
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
        self.trainer_indices_list = self.config.hyperparameters.trainer_indices_list

    def initialize(self) -> None:
        """Initialize role."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = Net().to(self.device)

    def load_data(self) -> None:
        """Load data."""
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        dataset = CIFAR10("./data", train=True, download=True, transform=transform)

        # indices = torch.arange(20000)

        # create indices into a list and convert to tensor
        # list_indices = list(range(0, 20000))
        indices = torch.tensor(self.trainer_indices_list)

        print("indices: ", indices)
        dataset = data_utils.Subset(dataset, indices)
        train_kwargs = {"batch_size": self.batch_size}

        self.train_loader = torch.utils.data.DataLoader(dataset, **train_kwargs)

    def train(self) -> None:
        """Train a model."""
        # self.optimizer = optim.Adadelta(self.model.parameters())
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

        for epoch in range(1, self.epochs + 1):
            self._train_epoch(epoch)

        # save dataset size so that the info can be shared with aggregator
        self.dataset_size = len(self.train_loader.dataset)

    # CODE FROM PYTORCH EXAMPLE - similar code is in _train_epoch
    # for epoch in range(2):  # loop over the dataset multiple times
    #     running_loss = 0.0
    #     for i, data in enumerate(trainloader, 0):
    #         # get the inputs; data is a list of [inputs, labels]
    #         inputs, labels = data
    #         # zero the parameter gradients
    #         optimizer.zero_grad()
    #         # forward + backward + optimize
    #         outputs = net(inputs)
    #         loss = criterion(outputs, labels)
    #         loss.backward()
    #         optimizer.step()
    # CODE FROM PYTORCH EXAMPLE - similar code is in _train_epoch

    def _train_epoch(self, epoch):
        self.model.train()

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % 10 == 0:
                done = batch_idx * len(data)
                total = len(self.train_loader.dataset)
                percent = 100.0 * batch_idx / len(self.train_loader)
                logger.info(
                    f"epoch: {epoch} [{done}/{total} ({percent:.0f}%)]"
                    f"\tloss: {loss.item():.6f}"
                )

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
