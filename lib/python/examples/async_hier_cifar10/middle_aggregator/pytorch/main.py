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
"""Hier_Cifar10 horizontal hierarchical FL middle level aggregator for Pytorch."""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from flame.config import Config
from flame.mode.horizontal.asyncfl.middle_aggregator import MiddleAggregator
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10

logger = logging.getLogger(__name__)


class PytorchCifar10MiddleAggregator(MiddleAggregator):
    """Pytorch Cifar10 Middle Level Aggregator."""

    def __init__(self, config: Config) -> None:
        """Initialize a class instance."""
        self.config = config

    def initialize(self):
        """Initialize role."""
        pass

    def load_data(self) -> None:
        """Load a test dataset."""
        pass

    def train(self) -> None:
        """Train a model."""
        pass

    def evaluate(self) -> None:
        """Evaluate (test) a model."""
        pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("config", nargs="?", default="./config.json")

    args = parser.parse_args()

    config = Config(args.config)

    a = PytorchCifar10MiddleAggregator(config)
    a.compose()
    a.run()
