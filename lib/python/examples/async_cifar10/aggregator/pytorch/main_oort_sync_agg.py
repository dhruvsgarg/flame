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
"""CIFAR-10 horizontal FL, OORT aggregator for PyTorch.

The example below is implemented based on the following example from
pytorch:
https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html.
"""

import logging
import os


# wandb setup
import wandb
from flame.config import Config
from flame.dataset import Dataset
from flame.mode.horizontal.oort.top_aggregator import TopAggregator
from flame import harness

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
import fl_data  # noqa: E402
from agg_common import ExampleAggregatorMixin  # noqa: E402
from oracle_utility import OracleInjectMixin  # noqa: E402


def initialize_wandb(run_name=None):
    wandb.init(
        # set the wandb project where this run will be logged
        project="ft-distr-ml",
        name=run_name,  # Set the run name
        # track hyperparameters and run metadata
        config={
            # REFL/Oort CIFAR-10 uses client LR = 0.01 (from REFL config)
            # REFL uses LR decay: decay_factor=0.98 every decay_epoch=10 rounds
            # Oort baselines don't use LR decay
            "client_learning_rate": 0.01,
            "lr_decay_enabled": True,  # For REFL experiments
            "lr_decay_factor": 0.98,
            "lr_decay_epoch": 10,
            "min_learning_rate": 0.0001,
            "architecture": "CNN",
            "dataset": "CIFAR-10",
            "fl-type": "sync, oort",
            "agg_rounds": 750,
            "trainer_epochs": 1,
            "config": "hetero",
            "alpha": 100,
            "failures": "No failure",
            "total clients N": 100,
            "client agg goal K": 10,
            "server_batch_size": 32,
            "client_batch_size": 32,
            "comments": "First oort no failure run",
        },
    )


logger = logging.getLogger(__name__)


Net = fl_data.CifarNet  # FX-N10: models live in fl_data (one per dataset)

class PyTorchCifar10Aggregator(ExampleAggregatorMixin, OracleInjectMixin, TopAggregator):
    """PyTorch CIFAR-10 Aggregator."""

    EVAL_ROUND_ONE = True

    def __init__(
        self, config: Config, log_to_wandb: bool, wandb_run_name: str = None
    ) -> None:
        """Initialize a class instance."""
        self.config = config
        # Before load_data, which runs ahead of initialize.
        self.harness_mode = harness.harness_mode(self.config.hyperparameters)
        self.model = None
        self.dataset: Dataset = None

        self.device = None
        self.test_loader = None

        self.learning_rate = self.config.hyperparameters.learning_rate
        self.batch_size = self.config.hyperparameters.batch_size or 16

        self.loss_list = []

        # Use wandb logging if enabled
        self.log_to_wandb = log_to_wandb
        if self.log_to_wandb:
            initialize_wandb(run_name=wandb_run_name)


if __name__ == "__main__":
    import argparse
    from flame.launch.cli import load_config_from_argv

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--log_to_wandb", action="store_true")
    parser.add_argument("--wandb_run_name", type=str)
    args, _ = parser.parse_known_args()

    config = load_config_from_argv()

    a = PyTorchCifar10Aggregator(config, args.log_to_wandb, args.wandb_run_name)

    from flame import telemetry

    telemetry.configure(role="aggregator", end_id=config.job.job_id)

    a.compose()
    a.run()
