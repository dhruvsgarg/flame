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
"""honrizontal FL middle level aggregator."""

from flame.mode.horizontal.syncfl.slow_middle_aggregator import (
    TAG_AGGREGATE,
    TAG_DISTRIBUTE,
    TAG_FETCH,
    TAG_UPLOAD,
    WAIT_TIME_FOR_TRAINER,
    SlowMiddleAggregator,
)

# Redirect `flame.mode.horizontal.middle_aggregator` to
# `flame.mode.horizontal.syncfl.middle_aggregator`
# This is for backward compatibility
