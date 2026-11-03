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
"""Model registry provider."""

from typing import TYPE_CHECKING, Union

from flame.config import RegistryType
from flame.object_factory import ObjectFactory
from flame.registry.dummy import DummyRegistryClient
from flame.registry.local import LocalRegistryClient

if TYPE_CHECKING:
    from flame.registry.mlflow import MLflowRegistryClient


class RegistryProvider(ObjectFactory):
    """Model registry provider."""

    def get(
        self, registry_name, **kwargs
    ) -> Union[DummyRegistryClient, "MLflowRegistryClient"]:
        """Return a registry client for a given registry name."""
        return self.create(registry_name, **kwargs)


def _mlflow_client(**kwargs) -> "MLflowRegistryClient":
    """Lazy: importing mlflow (+pandas) costs ~80 MB host RAM per process, paid by every trainer (FX-N84)."""
    from flame.registry.mlflow import MLflowRegistryClient

    return MLflowRegistryClient(**kwargs)


registry_provider = RegistryProvider()
registry_provider.register(RegistryType.DUMMY, DummyRegistryClient)
registry_provider.register(RegistryType.MLFLOW, _mlflow_client)
registry_provider.register(RegistryType.LOCAL, LocalRegistryClient)
