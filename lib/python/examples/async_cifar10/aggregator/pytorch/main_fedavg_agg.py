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
"""CIFAR-10 horizontal FL, FedAvg  SyncFL aggregator for PyTorch.

The example below is implemented based on the following example from
pytorch:
https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html.
"""

import logging
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# wandb setup
import wandb
from flame.config import Config
from flame.dataset import Dataset
from flame.mode.horizontal.top_aggregator import TopAggregator
from flame import harness

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
import fl_data  # noqa: E402
from oracle_utility import OracleInjectMixin  # noqa: E402

def initialize_wandb():
    wandb.init(
        project="ft-distr-ml",
        config={
            "client_learning_rate": 0.25,
            "architecture": "CNN",
            "dataset": "CIFAR-10",
            "fl-type": "sync, fedavg",
            "agg_rounds": 750,
            "trainer_epochs": 1,
            "config": "hetero",
            "alpha": 100,
            "failures": "No failure",
            "total clients N": 100,
            "client agg goal K": 10,
            "server_batch_size": 32,
            "client_batch_size": 32,
            "comments": "Fedavg SyncFL no failure run",
        },
    )


logger = logging.getLogger(__name__)


Net = fl_data.CifarNet  # FX-N10: models live in fl_data (one per dataset)

class PyTorchCifar10Aggregator(OracleInjectMixin, TopAggregator):
    """PyTorch CIFAR-10 Aggregator."""

    def __init__(self, config: Config, log_to_wandb: bool = False) -> None:
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

        self.log_to_wandb = log_to_wandb
        if self.log_to_wandb:
            initialize_wandb()

    @property
    def data_spec(self) -> "fl_data.DatasetSpec":
        """FX-N10: this run's dataset (hyperparameters.dataset_name)."""
        return fl_data.spec_for(self.config.hyperparameters)

    def initialize(self):
        """Initialize role."""
        self.device = harness.device_for(self.harness_mode)

        self.model = self.data_spec.model().to(self.device)
        self._init_oracle_util(
            _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                          "..", "..", "data"))

    def load_data(self) -> None:
        """Load a test dataset."""
        n_test = harness.harness_test_samples(self.config.hyperparameters, self.harness_mode)
        if self.harness_mode == "stub":
            dataset = harness.synthetic_dataset(n_test, self.data_spec.stub_shape, self.data_spec.num_classes,
                                                seed_key="agg_test", label_skew=0.0)
        else:
            dataset = self.data_spec.test()
            if self.harness_mode == "tiny_cpu":
                dataset = torch.utils.data.Subset(dataset, list(range(n_test)))
            dataset = fl_data.in_memory(dataset)  # FX-D64: per-sample decode in the eval thread stalled real ingest

        test_kwargs = {
            "batch_size": self.batch_size,
            "shuffle": False,
            "num_workers": 0,  # Changed from 2 to 0 - reduces CPU RAM usage
            "pin_memory": False,  # FX-D64: pinning an in-memory test set cost 2 s per speech eval
        }

        self.test_loader = torch.utils.data.DataLoader(dataset, **test_kwargs)

        # store data into dataset for analysis (e.g., bias)
        self.dataset = Dataset(dataloader=self.test_loader)

    def train(self) -> None:
        """Train a model."""
        # Implement this if testing is needed in aggregator
        pass

    def evaluate(self) -> None:
        """Evaluate (test) a model."""
        # Gate eval cadence (evalEveryNRounds) instead of evaluating every round;
        # the full test pass dominates per-round cost at n300. Used by the
        # FedDance arm (this is its aggregator stack). Always eval round 1.
        eval_every = (
            getattr(self.config.hyperparameters, "eval_every_n_rounds", 10) or 10
        )
        if self._round != 1 and (self._round % eval_every != 0):
            return
        self._eval_every_n_commits = 1  # FX-D63: the round gate is the cadence; a commit stride on top halved it
        # Off the critical path: snapshot weights now, run the test-set forward
        # pass in a daemon thread so the aggregator keeps progressing. Backgrounding
        # is why this needs no sim_model_*_compute_time vclock fold — see
        # main_asyncfl_agg.py's evaluate().
        eval_model = self._eval_snapshot_model()
        if eval_model is None:
            return  # prior async eval still running
        round_num = self._round
        test_loader, device = self.test_loader, self.device

        def _job():
            try:
                _t0 = time.time()
                eval_model.eval()
                test_loss = 0
                correct = 0
                with torch.no_grad():
                    for data, target in test_loader:
                        data, target = data.to(device), target.to(device)
                        output = eval_model(data)
                        test_loss += F.nll_loss(output, target, reduction="sum").item()
                        pred = output.argmax(dim=1, keepdim=True)
                        correct += pred.eq(target.view_as(pred)).sum().item()
                total = len(test_loader.dataset)
                logger.info(f"[ASYNC_EVAL_TIMING] round={round_num} wall_s={time.time() - _t0:.2f} device={device}")  # FX-N70
                self._eval_emit(round_num, test_loss / total, correct / total)
            except Exception as e:  # eval must never break training
                logger.warning(f"[ASYNC_EVAL] failed (non-fatal): {e}")
                self._eval_inflight = False

        import threading
        threading.Thread(target=_job, daemon=True).start()

        # print to save to file
        logger.debug(f"loss list at cifar agg: {self.loss_list}")

    def check_and_sleep(self) -> None:
        """Induce transient unavailability"""
        # Implement this if transient unavailability need to be
        # emulated in aggregator
        pass


if __name__ == "__main__":
    import argparse

    from flame.launch.cli import load_config_from_argv

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--log_to_wandb", action="store_true")
    parser.add_argument("--wandb_run_name", type=str)
    args, _ = parser.parse_known_args()

    config = load_config_from_argv()

    a = PyTorchCifar10Aggregator(config, args.log_to_wandb)

    from flame import telemetry

    telemetry.configure(role="aggregator", end_id=config.job.job_id)

    a.compose()
    a.run()
