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
"""Dummy Regularizer."""
import logging

from flame.common.constants import TrainState

logger = logging.getLogger(__name__)


class Regularizer:
    """Regularizer class."""

    def __init__(self):
        """Initialize Regularizer instance."""
        pass

    def get_term(self, **kwargs):
        """No regularizer term for dummy regularizer."""
        return 0.0

    def save_state(self, state: TrainState, **kwargs):
        """No states saved in dummy regularizer."""
        pass

    def update(self):
        """No need for update in dummy regularizer."""
        pass
