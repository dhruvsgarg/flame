# Copyright 2023 Cisco Systems, Inc. and its affiliates
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
"""MNIST horizontal FL trainer for Keras."""

import ast
import calendar
import logging
import time
from random import randrange
from statistics import mean

import numpy as np
from flame.config import Config
from flame.mode.horizontal.trainer import Trainer
from tensorflow import keras
from tensorflow.keras import layers

logger = logging.getLogger(__name__)


class KerasMnistTrainer(Trainer):
    """Keras Mnist Trainer."""

    def __init__(self, config: Config) -> None:
        """Initialize a class instance."""
        self.config = config
        self.dataset_size = 0

        self.num_classes = 10
        self.input_shape = (28, 28, 1)

        self.model = None
        self._x_train = None
        self._y_train = None
        self._x_test = None
        self._y_test = None

        self.epochs = self.config.hyperparameters.epochs
        print("==== Trainer epochs: ", self.epochs)
        self.batch_size = self.config.hyperparameters.batch_size or 128

        self.trainer_id = self.config.task_id

        # TODO: (DG) Remove the hard requirement for config to include
        # trainer_indices_list and failure_durations_s Setting the
        # indices used by the trainer Loading the failure durations
        # for trainers
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

    def initialize(self) -> None:
        """Initialize role."""
        model = keras.Sequential(
            [
                keras.Input(shape=self.input_shape),
                layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Flatten(),
                layers.Dropout(0.5),
                layers.Dense(self.num_classes, activation="softmax"),
            ]
        )

        model.compile(
            loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
        )

        self.model = model

    def load_data(self) -> None:
        """Load data."""
        # the data, split between train and test sets
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

        split_n = 10
        index = randrange(split_n)
        # reduce train sample size to reduce the runtime
        x_train = np.split(x_train, split_n)[index]
        y_train = np.split(y_train, split_n)[index]
        x_test = np.split(x_test, split_n)[index]
        y_test = np.split(y_test, split_n)[index]

        # Scale images to the [0, 1] range
        x_train = x_train.astype("float32") / 255
        x_test = x_test.astype("float32") / 255
        # Make sure images have shape (28, 28, 1)
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)

        # convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_test = keras.utils.to_categorical(y_test, self.num_classes)

        self._x_train = x_train
        self._y_train = y_train
        self._x_test = x_test
        self._y_test = y_test

    def train(self) -> None:
        """Train a model."""
        history = self.model.fit(
            self._x_train,
            self._y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=0.1,
        )

        i = 0
        while i < 1000000000:
            i += 1

        # save dataset size so that the info can be shared with
        # aggregator
        self.dataset_size = len(self._x_train)

        loss = mean(history.history["loss"])
        accuracy = mean(history.history["accuracy"])
        self.update_metrics({"loss": loss, "accuracy": accuracy})

    def evaluate(self) -> None:
        """Evaluate a model."""
        score = self.model.evaluate(self._x_test, self._y_test, verbose=0)

        logger.info(f"Test loss: {score[0]}")
        logger.info(f"Test accuracy: {score[1]}")

        # update metrics after each evaluation so that the metrics can
        # be logged in a model registry.
        self.update_metrics({"test-loss": score[0], "test-accuracy": score[1]})

    def check_and_sleep(self) -> None:
        """Induce transient unavailability"""
        # Implement this if transient unavailability need to be
        # emulated in the trainer
        curr_time = time.time()
        
        if (curr_time >= self.timestamp_next_sleep_s) and (
            len(self.failure_durations_s) > 0
        ):
            # pop leftmost element
            sleep_config_tuple = self.failure_durations_s.pop(0)

            # get the duration of sleep and set the params for next
            # sleep
            sleep_start_ts_from_trainer_init = self.trainer_start_ts + sleep_config_tuple[0]
            sleep_duration_s = sleep_config_tuple[1]

            # remaining sleep = trainer_start_ts + actual_sleep_start
            # + actual_sleep_duration - current_ts 
            remaining_sleep_duration_s = sleep_start_ts_from_trainer_init + sleep_duration_s - curr_time
            logger.debug(f"Task_id: {self.trainer_id} given_sleep_duration_s: {sleep_duration_s} with remaining_sleep_duration_s: {remaining_sleep_duration_s} at timestamp: {curr_time}")
            
            if (remaining_sleep_duration_s <= 0):
                logger.info(f"Task_id: {self.trainer_id} got -ve remaining sleep "
                            f"at timestamp: {curr_time}")
                # Need to pop out failure intervals that occur in the
                # past
                time_elapsed_from_start = curr_time - self.trainer_start_ts
                while time_elapsed_from_start > (self.failure_durations_s[0][0] + self.failure_durations_s[0][1]):
                    self.failure_durations_s.pop(0)
                    if len(self.failure_durations_s) == 0:
                        break
            else:
                # leave channel
                self._perform_channel_leave(tag="upload")

                # sleep for remaining time
                logger.info(f"Task_id: {self.trainer_id} going to sleep "
                            f"at timestamp: {time.time()}")
                time.sleep(remaining_sleep_duration_s)
                logger.info(f"Task_id: {self.trainer_id} woke up at timestamp: "
                            f"{time.time()}")

                # join channel
                self._perform_channel_join(tag="upload")

            # check if failure_list is now empty, if yes, reset
            # ts_next_sleep_s if not empty, set it to the next value
            if len(self.failure_durations_s) > 0:
                self.timestamp_next_sleep_s = self.trainer_start_ts + self.failure_durations_s[0][0]
                if(self.timestamp_next_sleep_s < time.time()):
                    logger.info(f"Task_id: {self.trainer_id} ERROR - JUST SET NEXT self.timestamp_next_sleep_s < curr_time")
            else:
                self.timestamp_next_sleep_s = calendar.timegm(
                    time.strptime(
                        "Dec 31, 2030 @ 23:59:59 UTC", "%b %d, %Y @ %H:%M:%S UTC"
                    )
                )
                logger.info(f"Task_id: {self.trainer_id} no more sleep for trainer")

        logger.debug(f"Task_id: {self.trainer_id} check_and_sleep completed at "
                     f"timestamp: {time.time()}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("config", nargs="?", default="./config.json")

    args = parser.parse_args()

    config = Config(args.config)

    t = KerasMnistTrainer(config)
    t.compose()
    t.run()
