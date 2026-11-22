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
"""FMoW horizontal FL, FedAvg SyncFL aggregator for PyTorch."""

from flame.common.constants import DeviceType
from flame.common.util import materialize_weights, weights_to_device
from flame.mode.horizontal.top_aggregator import TopAggregator

from fmow_agg_common import FMoWAggregatorMixin, run_aggregator


class PyTorchFMoWAggregator(FMoWAggregatorMixin, TopAggregator):
    """Synchronous FMoW policy on the shared workload."""

    fl_type = "sync, fedavg"
    require_complete_sim_cohort = True

    def _sync_wait_k_on(self):
        """A FMoW model version requires K useful updates, including with no availability filter."""
        return True

    def _weights_for_aggregation(self, msg):
        """Buffer updates on CPU; FedAvg reduces them on the model's GPU."""
        weights = materialize_weights(msg)
        return weights_to_device(weights, DeviceType.CPU) if weights is not None else None


if __name__ == "__main__":
    run_aggregator(PyTorchFMoWAggregator)
