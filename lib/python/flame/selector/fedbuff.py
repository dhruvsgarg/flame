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
"""FedBuffSelector class."""

import logging

from flame.end import End
from flame.selector.async_base import AsyncSelectorBase, SelectContext

logger = logging.getLogger(__name__)


class FedBuffSelector(AsyncSelectorBase):
    """Uniform-random async selection for fedbuff-based asyncfl.

    Policy is a uniform draw; the mechanism comes from `AsyncSelectorBase`.

    This class used to carry its own ~800-line copy of that mechanism, drifted
    from `async_oort.py`'s: no version_key re-pick guard, no virtual-clock
    timeout, no R1 pending-commit guard, no availability filter, an
    `agg_goal`-capped cleanup drain, and a resampling `_handle_recv_state`
    that raced dispatch. Re-basing IS the fix for all of them (R-A).
    """

    CHOOSE_SALT = "fedbuff"

    def _choose(self, candidates: dict[str, End], k: int, ctx: SelectContext) -> list:
        """Uniform draw over the eligible pool.

        `_keyed_topk`, not `random.sample`/reservoir sampling: an index-based
        draw makes every candidate's outcome depend on pool size and call
        order, so one trainer's incidental presence permanently desyncs real
        from sim. sorted() first -- `candidates` is keyed by join order, which
        differs between modes.
        """
        return self._keyed_topk(
            sorted(candidates), k, ctx.agg_version_key, self.CHOOSE_SALT
        )
