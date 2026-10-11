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
"""FedBuff optimizer.

The implementation is based on the following paper:
https://arxiv.org/pdf/2106.06639.pdf
https://arxiv.org/pdf/2111.04877.pdf

SecAgg algorithm is not the scope of this implementation.
"""
import logging
import math

import numpy as np
from diskcache import Cache

from ..common.typing import ModelWeights
from ..common.util import MLFramework, get_ml_framework_in_use, valid_frameworks
from .abstract import AbstractOptimizer
from .bn_buffers import clamp_running_var, is_bn_stat
from .regularizer.default import Regularizer

logger = logging.getLogger(__name__)


class FedBuff(AbstractOptimizer):
    """FedBuff class."""

    def __init__(self, **kwargs):
        """Initialize FedBuff instance."""
        super().__init__(**kwargs)

        self.agg_goal_weights = None
        self.clamp_running_var = str(kwargs.get("clamp_running_var", True)).lower() == "true"  # FX-N64
        # FX-D61: BN stats = mean of absolute stats (BN at the update's version + delta), outside rate and server lr.
        self.bn_absolute_mean = str(kwargs.get("bn_absolute_mean", True)).lower() == "true"
        self._bn_hist, self._bn_sum, self._bn_n, self._version = {}, {}, 0, None

        ml_framework_in_use = get_ml_framework_in_use()
        if ml_framework_in_use == MLFramework.PYTORCH:
            self.aggregate_fn = self._aggregate_pytorch
            self.scale_add_fn = self._scale_add_agg_weights_pytorch
        elif ml_framework_in_use == MLFramework.TENSORFLOW:
            self.aggregate_fn = self._aggregate_tensorflow
            self.scale_add_fn = self._scale_add_agg_weights_tensorflow
        else:
            raise NotImplementedError(
                "supported ml framework not found; "
                f"supported frameworks are: {valid_frameworks}"
            )

        self.regularizer = Regularizer()

        # FX-D97: server lr is config (baselines.yaml, datasets.yaml).
        try:
            self.learning_rate = float(kwargs["learning_rate"])
        except KeyError:
            raise KeyError("fedbuff optimizer needs an explicit learning_rate (S5)")

        # Set aggregation rate type between old (just staleness) and
        # new (tradeoff staleness and stat utility) Current options:
        # {"old", "new"}
        try:
            self.agg_rate_conf = kwargs["agg_rate_conf"]
        except KeyError:
            raise KeyError("Aggregation rate type not specified in the config")

    # #### FUNCTIONS TO TRADE-OFF STALENESS WITH STAT_UTILITY
    def alpha_polynomial(self, staleness, a_exp):
        return 1 / ((1 + staleness) ** a_exp)

    def alpha_exponential(self, staleness, a_exp):
        return np.exp(-a_exp * staleness)

    def beta_polynomial(self, loss, b_exp):
        return 1 - (1 / ((1 + loss) ** b_exp))

    def beta_polynomial_upshift(self, loss, b_exp):
        return 1 - (1 / ((1 + loss) ** b_exp)) + 0.5

    def beta_exponential(self, loss, b_exp):
        return 1 - np.exp(-b_exp * loss)

    def beta_exponential_custom(self, loss, b_exp):
        decay_constant = 500 / math.log(2)  # Adjusting the decay constant
        return math.exp(-loss / decay_constant)

    def weight_factor(
        self,
        scale,
        staleness,
        a_exp,
        loss,
        b_exp,
        alpha_type="polynomial",
        beta_type="polynomial",
    ):
        if alpha_type == "polynomial":
            alpha = self.alpha_polynomial(staleness, a_exp)
        elif alpha_type == "exponential":
            alpha = self.alpha_exponential(staleness, a_exp)
        else:
            raise ValueError("Invalid alpha type")

        if beta_type == "polynomial":
            beta = self.beta_polynomial(loss, b_exp)
        elif beta_type == "exponential":
            beta = self.beta_exponential(loss, b_exp)
        elif beta_type == "polynomial_upshift":
            beta = self.beta_polynomial_upshift(loss, b_exp)
        elif beta_type == "exponential_custom":
            beta = self.beta_exponential_custom(loss, b_exp)
        else:
            raise ValueError("Invalid beta type")

        # weight_factor range is [0, 1]
        return ((scale) * alpha) + ((1 - scale) * beta)

    def do(
        self,
        agg_goal_weights: ModelWeights,
        cache: Cache,
        *,
        total: int = 0,
        version: int = 0,
        staleness_factor: float = 0.0,
        **kwargs,
    ) -> ModelWeights:
        """Do aggregates models of trainers.

        Parameters
        ----------
        agg_goal_weights: delta weights aggregated until agg goal
        cache: a container that includes a list of weights for
        aggregation total: a number of data samples used to train
        weights in cache version: a version number of base weights

        Returns
        -------
        aggregated model: type is either list (tensorflow) or dict
        (pytorch)
        """
        logger.debug("calling fedbuff")

        self.agg_goal_weights = agg_goal_weights
        self.is_agg_weights_none = self.agg_goal_weights is None

        if len(cache) == 0 or total == 0:
            return None

        for k in list(cache.iterkeys()):
            # after popping, the item is removed from the cache hence,
            # explicit cache cleanup is not needed
            tres = cache.pop(k)

            # rate determined based on the staleness of local model
            if self.agg_rate_conf["type"] == "old":
                rate = 1 / math.sqrt(1 + version - tres.version)

            elif self.agg_rate_conf["type"] == "new":
                # New rate that trades off staleness and statistical
                # utility

                # agg_rate_conf will be a dict with keys: {type,
                # scale, a_exp, b_exp}
                scale_val = self.agg_rate_conf["scale"]
                a_exp_val = self.agg_rate_conf["a_exp"]
                b_exp_val = self.agg_rate_conf["b_exp"]

                rate = self.weight_factor(
                    scale=scale_val,
                    staleness=(version - tres.version),
                    a_exp=a_exp_val,
                    loss=tres.stat_utility,
                    b_exp=b_exp_val,
                    alpha_type="polynomial",
                    beta_type="polynomial_upshift",
                )

            logger.info(
                f"agg ver: {version}, trainer ver: {tres.version}, "
                f"trainer stat_utility: {tres.stat_utility}, rate: {rate}, "
                f"with agg_rate_type: {self.agg_rate_conf}"
            )
            self._version = version
            self._bn_base = self._bn_hist.get(tres.version)
            self.aggregate_fn(tres, rate)

        return self.agg_goal_weights

    def scale_add_agg_weights(
        self, base_weights: ModelWeights, agg_goal_weights: ModelWeights, agg_goal: int
    ) -> ModelWeights:
        """Scale aggregated weights and add it to the original
        weights, when aggregation goal is achieved.

        Parameters
        ----------
        base_weights: original weights of the aggregator
        agg_goal_weights: weights to be scaled and added agg_goal:
        aggregation goal of FedBuff algorithm.

        Returns
        -------
        updated weights
        """
        return self.scale_add_fn(base_weights, agg_goal_weights, agg_goal)

    def _scale_add_agg_weights_pytorch(
        self, base_weights: ModelWeights, agg_goal_weights: ModelWeights, agg_goal: int
    ) -> ModelWeights:
        logger.debug(f"base_weights.keys(): {base_weights.keys()}")

        learning_rate = self.learning_rate
        bn = self.bn_absolute_mean and self._version is not None
        if bn:
            self._bn_hist.setdefault(self._version, self._bn_copy(base_weights))
        for k in base_weights.keys():
            if bn and is_bn_stat(k):
                if self._bn_n:
                    base_weights[k] = (self._bn_sum[k] / self._bn_n).to(dtype=base_weights[k].dtype)
                continue
            base_weights[k] = (base_weights[k]) + (
                learning_rate * ((agg_goal_weights[k] / agg_goal))
            )
        if bn:  # result = global at version + 1
            self._bn_hist[self._version + 1] = self._bn_copy(base_weights)
            for v in [v for v in self._bn_hist if v < self._version - 63]:
                del self._bn_hist[v]
        self._bn_sum, self._bn_n = {}, 0
        if self.clamp_running_var and clamp_running_var(base_weights):
            logger.warning("[FEDBUFF_BN] negative running_var clamped to 0 (FX-N64)")
        if base_weights and not getattr(self, "_server_lr_logged", False):  # FX-N15: the server lr in force, once
            logger.info(f"[SERVER_LR] fedbuff server lr={learning_rate} from config learning_rate, agg_goal={agg_goal}")
            self._server_lr_logged = True
        return base_weights

    def _scale_add_agg_weights_tensorflow(
        self, base_weights: ModelWeights, agg_goal_weights: ModelWeights, agg_goal: int
    ) -> ModelWeights:
        learning_rate = self.learning_rate
        for idx in range(len(base_weights)):
            base_weights[idx] += learning_rate * (agg_goal_weights[idx] / agg_goal)
        return base_weights

    @staticmethod
    def _bn_copy(weights):
        return {k: v.detach().clone() for k, v in weights.items() if is_bn_stat(k)}

    def _aggregate_pytorch(self, tres, rate):
        logger.debug("calling _aggregate_pytorch")

        if self.is_agg_weights_none:
            self.agg_goal_weights = {}

        base = getattr(self, "_bn_base", None) if self.bn_absolute_mean else None
        if base is not None:  # a version older than the ring adds nothing to the BN mean
            self._bn_n += 1
        for k, v in tres.weights.items():
            if self.bn_absolute_mean and is_bn_stat(k):
                if base is not None:
                    a = base[k].double() + v.double()
                    self._bn_sum[k] = a if k not in self._bn_sum else self._bn_sum[k] + a
                v = v * 0  # keeps the key in agg_goal_weights; scale_add sets the mean
            tmp = v * rate
            # tmp.dtype is always float32 or double as rate is float
            # if v.dtype is integer (int32 or int64), there is type
            # mismatch this leads to the following error when
            #   self.agg_weights[k] += tmp: RuntimeError: result type
            #   Float can't be cast to the desired output type Long To
            # handle this issue, we typecast tmp to the original type
            # of v
            #
            # TODO: this may need to be revisited
            tmp = tmp.to(dtype=v.dtype) if tmp.dtype != v.dtype else tmp

            if self.is_agg_weights_none:
                self.agg_goal_weights[k] = tmp
            else:
                self.agg_goal_weights[k] += tmp

    def _aggregate_tensorflow(self, tres, rate):
        logger.debug("calling _aggregate_tensorflow")

        if self.is_agg_weights_none:
            self.agg_goal_weights = []

        for idx in range(len(tres.weights)):
            if self.is_agg_weights_none:
                self.agg_goal_weights.append(tres.weights[idx] * rate)
            else:
                self.agg_goal_weights[idx] += tres.weights[idx] * rate
