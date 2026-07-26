# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""AsyncRandomSelector class."""

import logging

from flame.end import End
from flame.selector.async_base import AsyncSelectorBase, SelectContext

logger = logging.getLogger(__name__)


class AsyncRandomSelector(AsyncSelectorBase):
    """Uniform-random async selection.

    Kept registered because `fluxtune_dynkc` names it -- the parked flag path
    for the pending dynamic-K/C contribution (BASELINES.md "park, do not
    delete"). Its previous 841-line body had no method of its own; every one
    was a drifted copy of FedBuff's or AsyncOort's. Only the RNG domain
    differs now, so it still draws its own order under a shared seed.
    """

    CHOOSE_SALT = "async_random"

    def _choose(self, candidates: dict[str, End], k: int, ctx: SelectContext) -> list:
        """Uniform draw over the eligible pool (see FedBuffSelector._choose)."""
        return self._keyed_topk(
            sorted(candidates), k, ctx.agg_version_key, self.CHOOSE_SALT
        )
