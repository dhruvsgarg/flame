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
"""CIFAR-10 horizontal FL aggregator for PyTorch.

The example below is implemented based on the following example from
pytorch:
https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html.
"""

import ast
import gc
import json
import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import wandb

# flame imports
from flame.config import Config
from flame.dataset import Dataset
from flame.mode.horizontal.asyncfl.top_aggregator import TopAggregator
from torch.utils.data import DataLoader
from torchaudio.datasets import SPEECHCOMMANDS


def initialize_wandb(run_name=None):
    wandb.init(
        # set the wandb project where this run will be logged
        project="ft-distr-ml",
        name=run_name,  # Set the run name
        # track hyperparameters and run metadata
        config={
            # fedbuff
            "server_learning_rate": 40.9,
            "client_learning_rate": 0.000195,
            
            # oort "client_learning_rate": 0.04,

            "architecture": "CNN",
            "dataset": "CIFAR-10",
            "fl-type": "async, fedbuff",
            "agg_rounds": 750,
            "trainer_epochs": 1,
            "config": "hetero",
            "alpha": 100,
            "failures": "No failure",
            "total clients N": 100,

            # fedbuff
            "client-concurrency C": 20,
            
            "client agg goal K": 10,
            "server_batch_size": 32,
            "client_batch_size": 32,
            "comments": "First oort no failure run",
        },
    )


LABEL_MAP = {
    'backward': 0,
    'bed': 1,
    'bird': 2,
    'cat': 3,
    'dog': 4,
    'down': 5,
    'eight': 6,
    'five': 7,
    'follow': 8,
    'forward': 9,
    'four': 10,
    'go': 11,
    'happy': 12,
    'house': 13,
    'learn': 14,
    'left': 15,
    'marvin': 16,
    'nine': 17,
    'no': 18,
    'off': 19,
    'on': 20,
    'one': 21,
    'right': 22,
    'seven': 23,
    'sheila': 24,
    'six': 25,
    'stop': 26,
    'three': 27,
    'tree': 28,
    'two': 29,
    'up': 30,
    'visual': 31,
    'wow': 32,
    'yes': 33,
    'zero': 34
}


logger = logging.getLogger(__name__)


class BasicBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet34_1D(nn.Module):
    def __init__(self, n_input=1, n_output=35):
        super(ResNet34_1D, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv1d(n_input, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 3)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, n_output)

    def _make_layer(self, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

        layers = []
        layers.append(BasicBlock1D(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(BasicBlock1D(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


class PyTorchSpeechCommandsAggregator(TopAggregator):
    """PyTorch Google Speech commands Aggregator."""

    def __init__(
            self,
            config: Config,
            log_to_wandb: bool,
            wandb_run_name: str = None) -> None:
        """Initialize a class instance."""
        self.config = config
        self.model = None
        self.dataset: Dataset = None

        self.device = None
        self.test_loader = None

        self.learning_rate = self.config.hyperparameters.learning_rate
        self.batch_size = self.config.hyperparameters.batch_size or 16

        self.track_trainer_avail = self.config.hyperparameters.track_trainer_avail or None
        self.reject_stale_updates = self.config.hyperparameters.reject_stale_updates or False
        self.trainer_unavail_durations = None
        if (
            self.track_trainer_avail["enabled"] and
            self.track_trainer_avail["type"] == "ORACULAR"
        ):
            self.trainer_unavail_durations = self.read_trainer_unavailability()
            print("self.trainer_unavail_durations: ", self.trainer_unavail_durations)

        self.loss_list = []

        # Use wandb logging if enabled
        self.log_to_wandb = log_to_wandb
        if self.log_to_wandb:
            initialize_wandb(run_name=wandb_run_name)
    
    def initialize(self):
        """Initialize role."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = ResNet34_1D().to(self.device)

    def read_trainer_unavailability(self) -> None:
        print("Came to read_trainer_unavailability")
        # maintain <trainer_id: [(unavail_start1, duration1), (start2,
        # duration2)..]>
        trainer_unavail_dict = {}

        # set path to read json files from TODO: Remove hardcoding
        # later
        files_path = "../../trainer/config_dir100_num100_traceFailure_1.5h"

        # set range of trainer ids to read from
        trainer_start_num = 1
        trainer_end_num = 100
        for i in range(trainer_start_num, trainer_end_num + 1):
            dirname = os.path.dirname(__file__)
            with open(os.path.join(dirname, files_path, "trainer_" + str(i) + ".json")
                      ) as f:
                trainer_json = json.load(f)
                curr_trainer_id = trainer_json["taskid"]
                curr_trainer_unavail_time = ast.literal_eval(trainer_json[
                    "hyperparameters"]["failure_durations_s"]
                    )
                trainer_unavail_dict[curr_trainer_id] = curr_trainer_unavail_time
                print("Completed file read for ", os.path.join(files_path, "trainer_"
                                                               + str(i) + ".json"))

        # selector - do a linear search in the selector based on
        # availability selector - delete those tuples whose sleep time
        # has passed
        print("Completed reading all trainer unavailability from files")
        return trainer_unavail_dict

    def load_data(self) -> None:
        """Load a test dataset."""
        data_dir = "./data"
        os.makedirs(data_dir, exist_ok=True)

        dataset = SPEECHCOMMANDS(
            data_dir, download=True, subset='testing'
        )
        
        def custom_collate_fn(batch):
            # Extract sequences and labels
            sequences = [item[0].squeeze(0) for item in batch]  # Remove the extra dimension
            labels = [item[2] for item in batch]  # Assuming item[2] is the label

            # Convert string labels to integers using the label map
            labels = [LABEL_MAP[label] for label in labels]

            # Determine the maximum length in this batch
            max_length = max(sequence.size(0) for sequence in sequences)

            # Pad sequences to the maximum length
            padded_sequences = torch.zeros((len(sequences), 1, max_length))  # Add channel dimension
            for i, sequence in enumerate(sequences):
                padded_sequences[i, 0, :sequence.size(0)] = sequence

            # Convert labels to tensor
            labels = torch.tensor(labels)

            return padded_sequences, labels

        test_kwargs = {
            "batch_size": self.batch_size,
            "shuffle": False,
            "num_workers": 2,
            "collate_fn": custom_collate_fn,
        }

        self.test_loader = DataLoader(dataset, **test_kwargs)

        # Store data into dataset for analysis (e.g., bias)
        self.dataset = Dataset(dataloader=self.test_loader)

        # Release the memory of the full dataset
        del dataset
        gc.collect()

    def train(self) -> None:
        """Train a model."""
        # Implement this if testing is needed in aggregator
        pass
    
    def evaluate(self) -> None:
        """Evaluate (test) a model."""

        # NOTE: TO increase speed of execution of the aggregator, not
        # going to evaluate for each round. Will do so only once in
        # every 100 rounds.
        if self._round % 100 != 0:
            logger.info(f"Skipping evaluate for round: {self._round}")
            return

        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                # Reshape output to [batch_size, n_output] from [batch_size, 1, n_output]
                output = output.squeeze(1)

                test_loss += F.nll_loss(
                    output, target, reduction="sum"
                ).item()  # sum up batch loss

                pred = output.argmax(
                    dim=1, keepdim=True
                )  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        total = len(self.test_loader.dataset)
        test_loss /= total
        test_accuracy = correct / total

        logger.info(f"Test loss: {test_loss}, test accuracy: "
                    f"{correct}/{total} ({test_accuracy})")

        # update metrics after each evaluation so that the metrics can
        # be logged in a model registry.
        self.update_metrics({"test-loss": test_loss, "test-accuracy": test_accuracy})

        # Send metrics to wandb
        if self.log_to_wandb:
            wandb.log({"test_acc": test_accuracy, "test_loss": test_loss})
        self.loss_list.append(test_loss)

        # print to save to file
        logger.debug(f"loss list at cifar agg: {self.loss_list}")

    def check_and_sleep(self) -> None:
        """Induce transient unavailability"""
        # Implement this if transient unavailability need to be
        # emulated in aggregator
        pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("config", nargs="?", default="./config.json")
    # Add the --log_to_wandb argument
    parser.add_argument(
        '--log_to_wandb',
        action='store_true',
        help='Flag to log to Weights and Biases'
    )
    parser.add_argument(
        '--wandb_run_name',
        type=str,
        help='Name of the Weights and Biases run'
    )

    args = parser.parse_args()

    config = Config(args.config)

    a = PyTorchSpeechCommandsAggregator(config, args.log_to_wandb, args.wandb_run_name)
    a.compose()
    a.run()
