# Copyright 2023 Cisco Systems, Inc. and its affiliates
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
"""MedMNIST Oort aggregator for PyTorch."""

import logging
import numpy as np
import torch
import torchvision

from flame.common.util import install_packages
from flame.config import Config
from flame.mode.horizontal.oort.top_aggregator import TopAggregator

install_packages(["scikit-learn"])

from sklearn.metrics import accuracy_score
from PIL import Image
from datetime import datetime

logger = logging.getLogger(__name__)


class PathMNISTDataset(torch.utils.data.Dataset):

    def __init__(self, transform=None, as_rgb=False):
        npz_file = np.load("pathmnist.npz")

        self.transform = transform
        self.as_rgb = as_rgb

        self.imgs = npz_file["val_images"]
        self.labels = npz_file["val_labels"]

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, index):
        img, target = self.imgs[index], self.labels[index].astype(int)
        img = Image.fromarray(img)

        if self.as_rgb:
            img = img.convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, target


class CNN(torch.nn.Module):
    """CNN Class"""

    def __init__(self, num_classes):
        """Initialize."""
        super(CNN, self).__init__()
        self.num_classes = num_classes
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 6, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(6),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(6, 16, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = torch.nn.Linear(16 * 7 * 7, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class PyTorchMedMNistAggregator(TopAggregator):
    """PyTorch MedMNist Aggregator"""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.model = None
        self.dataset = None

        self.batch_size = self.config.hyperparameters.batch_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.learning_rate = self.config.hyperparameters.learning_rate

        self.exp_start_time = datetime.now()

    def initialize(self):
        """Initialize."""
        self.model = CNN(num_classes=9)
        self.model.to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()

    def load_data(self) -> None:
        """Load a test dataset."""
        logger.info("in load_data")
        self._download()

        data_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                ),
            ]
        )

        dataset = PathMNISTDataset(transform=data_transform)

        self.loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )
        self.dataset_size = len(dataset)

    def _download(self) -> None:
        import requests

        r = requests.get(self.config.dataset, allow_redirects=True)
        open("pathmnist.npz", "wb").write(r.content)

    def train(self) -> None:
        """Train a model."""
        pass

    def evaluate(self) -> None:
        """Evaluate (test) a model."""
        self.model.eval()
        loss_lst = list()
        labels = torch.tensor([], device=self.device)
        labels_pred = torch.tensor([], device=self.device)
        with torch.no_grad():
            for data, label in self.loader:
                data, label = data.to(self.device), label.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, label.squeeze())
                loss_lst.append(loss.item())
                labels_pred = torch.cat([labels_pred, output.argmax(dim=1)], dim=0)
                labels = torch.cat([labels, label], dim=0)

        labels_pred = labels_pred.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()
        val_acc = accuracy_score(labels, labels_pred)
        val_loss = sum(loss_lst) / len(loss_lst)

        logger.info(f"Round: {self._round} Test loss: {val_loss}")
        logger.info(f"Round: {self._round} Test accuracy: {val_acc}")

        logger.info(f"Current time: {datetime.now() - self.exp_start_time}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("config", nargs="?", default="./config.json")

    args = parser.parse_args()

    config = Config(args.config)

    a = PyTorchMedMNistAggregator(config)
    a.compose()
    a.run()
