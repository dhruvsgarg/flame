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
"""optimizer abstract class."""

from abc import ABC, abstractmethod

from diskcache import Cache

from ..common.typing import ModelWeights


class AbstractOptimizer(ABC):
    """Abstract base class for optimizer implementation."""

    def __init__(self, **kwargs) -> None:
        """Initialize an instance with keyword-based arguments."""
        for key, value in kwargs.items():
            setattr(self, key, value)

    @abstractmethod
    def do(
        self,
        base_weights: ModelWeights,
        cache: Cache,
        *,
        total: int = 0,
        version: int = 0,
        **kwargs
    ) -> ModelWeights:
        """Abstract method to conduct optimization."""
