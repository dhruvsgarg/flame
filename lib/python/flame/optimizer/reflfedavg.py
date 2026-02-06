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
"""REFL-aware FedAvg optimizer with deadline filtering and staleness handling."""

import logging
import math
from typing import Dict, List, Optional
import numpy as np
import gc

from diskcache import Cache

from ..common.typing import ModelWeights
from ..common.util import MLFramework, get_ml_framework_in_use, valid_frameworks
from .abstract import AbstractOptimizer
from .regularizer.default import Regularizer
from .train_result import TrainResult

logger = logging.getLogger(__name__)


class REFLFedAvg(AbstractOptimizer):
    """
    REFL-aware FedAvg optimizer with deadline filtering and staleness handling.
    
    Implements REFL's key contributions:
    - Deadline-based filtering to identify slow trainers (stragglers)
    - Stale update caching and lifecycle management
    - Multiple staleness weighting strategies (Equal, AdaSGD, DynSGD, REFL)
    """

    def __init__(self, **kwargs):
        """Initialize REFL FedAvg optimizer."""
        super().__init__(**kwargs)
        
        self.agg_weights = None

        ml_framework_in_use = get_ml_framework_in_use()
        if ml_framework_in_use == MLFramework.PYTORCH:
            self.aggregate_fn = self._aggregate_pytorch
        elif ml_framework_in_use == MLFramework.TENSORFLOW:
            self.aggregate_fn = self._aggregate_tensorflow
        else:
            raise NotImplementedError(
                "supported ml framework not found; "
                f"supported frameworks are: {valid_frameworks}"
            )

        self.regularizer = Regularizer()
        
        # REFL deadline parameters
        self.deadline = kwargs.get('deadline', 0)  # 0 = use moving average
        self.mov_avg_deadline = kwargs.get('initial_deadline', 100.0)
        self.target_ratio = kwargs.get('target_ratio', 0.8)  # Target fraction to wait for
        
        # Stale update parameters
        self.stale_update_max = kwargs.get('stale_update', -1)  # -1 = no limit
        self.stale_factor = kwargs.get('stale_factor', 1)  # 1=equal, -2=AdaSGD, -3=DynSGD, -4=REFL
        self.stale_beta = kwargs.get('stale_beta', 0.9)  # REFL beta parameter
        self.scale_coff = kwargs.get('scale_coff', 10.0)  # REFL scaling coefficient
        
        # Stale update storage
        self.stale_weights: Dict[str, ModelWeights] = {}
        self.stale_remain_duration: Dict[str, float] = {}
        self.stale_rounds: Dict[str, int] = {}
        self.stale_stat_utility: Dict[str, float] = {}
        self.stale_count: Dict[str, int] = {}
        
        # Statistics
        self.round_deadline_history = []
        self.stale_applied_count = 0
        self.stale_discarded_count = 0
        
        logger.info(
            f"REFLFedAvg initialized: deadline={self.deadline}, "
            f"stale_update_max={self.stale_update_max}, stale_factor={self.stale_factor}"
        )

    def do(
        self,
        base_weights: ModelWeights,
        cache: Cache,
        *,
        total: int = 0,
        version: int = 0,
        **kwargs,
    ) -> ModelWeights:
        """
        Aggregate models with REFL's deadline-based filtering and staleness handling.

        Parameters
        ----------
        base_weights: Base weights for aggregation
        cache: Cache containing training results
        total: Total number of data samples
        version: Model version number
        **kwargs: Additional arguments including:
            - round_duration: Duration of the round (for deadline filtering)
            - cur_time: Current virtual time (for stale lifecycle)

        Returns
        -------
        Aggregated model weights
        """
        logger.debug("Calling REFL FedAvg")

        assert base_weights is not None

        # Reset global weights before aggregation
        self.agg_weights = base_weights

        if len(cache) == 0 or total == 0:
            return None
        
        # Get round metadata
        round_duration = kwargs.get('round_duration', 0)
        cur_time = kwargs.get('cur_time', 0)
        
        # Collect all training results from cache
        all_results = []
        for k in list(cache.iterkeys()):
            tres = cache.pop(k)
            all_results.append(tres)
        
        if not all_results:
            return None
        
        # Apply deadline filtering
        fast_results, slow_results = self.filter_by_deadline(all_results, round_duration)
        
        logger.info(
            f"Deadline filtering: {len(fast_results)} fast, {len(slow_results)} slow "
            f"(deadline={self.get_effective_deadline():.2f}s)"
        )
        
        # Cache stale updates from slow trainers
        self.cache_stale_updates(slow_results, round_duration)
        
        # Get applicable stale updates from previous rounds
        applicable_stale = self.get_applicable_stale_updates(round_duration)
        
        logger.info(
            f"Stale updates: {len(applicable_stale)} applicable, "
            f"{len(self.stale_weights)} still cached"
        )
        
        # Combine fast trainers + applicable stale updates
        all_trainers = fast_results + applicable_stale
        
        if not all_trainers:
            logger.warning("No trainers to aggregate (all filtered or cached)")
            return None
        
        # Compute importance weights for all trainers
        importance_weights = self.compute_importance_weights(all_trainers, total)
        
        # Aggregate with weighted averaging
        aggregated_count = 0
        for tres in all_trainers:
            rate = (tres.count / total) * importance_weights.get(tres.end_id, 1.0)
            self.aggregate_fn(tres, rate)
            aggregated_count += 1
        
        logger.info(
            f"Aggregated {aggregated_count} trainers "
            f"({len(fast_results)} fast + {len(applicable_stale)} stale)"
        )
        
        # Update moving average deadline if using adaptive deadline
        if self.deadline == 0 and fast_results:
            self.update_moving_avg_deadline(fast_results)
        
        return self.agg_weights
    
    def filter_by_deadline(
        self, 
        results: List[TrainResult],
        round_duration: float
    ) -> tuple[List[TrainResult], List[TrainResult]]:
        """
        Filter training results into fast and slow based on deadline.
        
        Args:
            results: List of all training results
            round_duration: Actual duration of the round
        
        Returns:
            Tuple of (fast_results, slow_results)
        """
        if self.deadline <= 0 and self.mov_avg_deadline <= 0:
            # No deadline filtering
            return results, []
        
        effective_deadline = self.get_effective_deadline()
        
        fast = []
        slow = []
        
        for tres in results:
            # Use completion_time if available, otherwise use round_duration as estimate
            trainer_duration = tres.round_duration if tres.round_duration else round_duration
            
            if trainer_duration <= effective_deadline:
                fast.append(tres)
            else:
                slow.append(tres)
        
        return fast, slow
    
    def get_effective_deadline(self) -> float:
        """Get the effective deadline (fixed or moving average)."""
        if self.deadline > 0:
            return self.deadline
        elif self.mov_avg_deadline > 0:
            return self.mov_avg_deadline
        else:
            return float('inf')
    
    def cache_stale_updates(
        self, 
        slow_results: List[TrainResult],
        round_duration: float
    ) -> None:
        """
        Cache stale updates from slow trainers for potential future use.
        
        Args:
            slow_results: Training results from slow trainers
            round_duration: Duration of the current round
        """
        for tres in slow_results:
            end_id = tres.end_id if tres.end_id else "unknown"
            
            # Store stale update
            self.stale_weights[end_id] = tres.weights
            
            # Calculate remaining duration based on how late trainer was
            trainer_duration = tres.round_duration if tres.round_duration else round_duration
            self.stale_remain_duration[end_id] = trainer_duration - self.get_effective_deadline()
            
            # Initialize staleness counter
            self.stale_rounds[end_id] = 0
            
            # Store statistical utility for REFL weighting
            self.stale_stat_utility[end_id] = tres.stat_utility if tres.stat_utility else 0.0
            
            # Store sample count
            self.stale_count[end_id] = tres.count
            
            logger.debug(
                f"Cached stale update from {end_id}: "
                f"remaining_duration={self.stale_remain_duration[end_id]:.2f}s, "
                f"stat_utility={self.stale_stat_utility[end_id]:.4f}"
            )
    
    def get_applicable_stale_updates(
        self, 
        round_duration: float
    ) -> List[TrainResult]:
        """
        Get stale updates that are now ready to apply.
        
        Updates their staleness and removes expired updates.
        
        Args:
            round_duration: Duration of current round
        
        Returns:
            List of TrainResult objects from stale cache
        """
        applicable = []
        expired_ids = []
        
        for end_id in list(self.stale_weights.keys()):
            # Decrement remaining duration
            self.stale_remain_duration[end_id] -= round_duration
            
            # Increment staleness counter
            self.stale_rounds[end_id] += 1
            
            # Check if ready to apply
            if self.stale_remain_duration[end_id] <= 0:
                # Check if not too stale
                if self.stale_update_max < 0 or self.stale_rounds[end_id] <= self.stale_update_max:
                    # Apply this stale update
                    tres = TrainResult(
                        weights=self.stale_weights[end_id],
                        count=self.stale_count[end_id],
                        stat_utility=self.stale_stat_utility[end_id],
                        staleness=self.stale_rounds[end_id],
                        end_id=end_id
                    )
                    applicable.append(tres)
                    self.stale_applied_count += 1
                    
                    logger.debug(
                        f"Applying stale update from {end_id}: "
                        f"staleness={self.stale_rounds[end_id]} rounds"
                    )
                else:
                    # Too stale, discard
                    self.stale_discarded_count += 1
                    logger.debug(
                        f"Discarding stale update from {end_id}: "
                        f"staleness={self.stale_rounds[end_id]} > {self.stale_update_max}"
                    )
                
                # Remove from cache
                expired_ids.append(end_id)
            else:
                logger.debug(
                    f"Stale update from {end_id} still cached: "
                    f"remaining={self.stale_remain_duration[end_id]:.2f}s, "
                    f"staleness={self.stale_rounds[end_id]} rounds"
                )
        
        # Clean up expired entries
        for end_id in expired_ids:
            del self.stale_weights[end_id]
            del self.stale_remain_duration[end_id]
            del self.stale_rounds[end_id]
            del self.stale_stat_utility[end_id]
            del self.stale_count[end_id]
        
        # Force garbage collection to free memory
        if expired_ids:
            gc.collect()
        
        return applicable
    
    def compute_importance_weights(
        self, 
        trainers: List[TrainResult],
        total: int
    ) -> Dict[str, float]:
        """
        Compute importance weights for trainers based on staleness strategy.
        
        Args:
            trainers: List of training results (fast + applicable stale)
            total: Total number of samples
        
        Returns:
            Dictionary mapping end_id to importance weight
        """
        weights = {}
        
        # Find max statistical utility for REFL method
        max_stat_utility = max(
            [tres.stat_utility for tres in trainers if tres.stat_utility], 
            default=1.0
        )
        
        for tres in trainers:
            end_id = tres.end_id if tres.end_id else "unknown"
            staleness = tres.staleness if tres.staleness else 0
            stat_utility = tres.stat_utility if tres.stat_utility else 0.0
            
            # Base weight (will be adjusted by staleness)
            base_weight = 1.0
            
            # Apply staleness weighting strategy
            if self.stale_factor > 1:
                # Divide by constant factor
                weight = base_weight / self.stale_factor
            
            elif self.stale_factor == 1:
                # Equal weight (standard FedAvg)
                weight = base_weight
            
            elif self.stale_factor == -1:
                # Average: divide by average staleness
                avg_staleness = np.mean([t.staleness for t in trainers if t.staleness])
                weight = base_weight / max(avg_staleness, 1.0)
            
            elif self.stale_factor == -2:
                # AdaSGD: divide by (staleness + 1)
                weight = base_weight / (staleness + 1)
            
            elif self.stale_factor == -3:
                # DynSGD: multiply by exp(-(staleness + 1))
                weight = base_weight * math.exp(-(staleness + 1))
            
            elif self.stale_factor == -4:
                # REFL: hybrid formula
                client_ratio = stat_utility / max(max_stat_utility, 1e-6)
                weight = (
                    (1 - self.stale_beta) / (staleness + 1) +
                    self.stale_beta * (1 - math.exp(-client_ratio / max_stat_utility) / self.scale_coff)
                )
            
            else:
                logger.warning(f"Unknown stale_factor={self.stale_factor}, using equal weight")
                weight = base_weight
            
            weights[end_id] = weight
        
        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        return weights
    
    def update_moving_avg_deadline(self, fast_results: List[TrainResult]) -> None:
        """
        Update moving average deadline based on fast trainers.
        
        Uses target_ratio percentile of completion times.
        
        Args:
            fast_results: Training results from fast trainers
        """
        if not fast_results:
            return
        
        # Get completion times
        durations = []
        for tres in fast_results:
            if tres.round_duration:
                durations.append(tres.round_duration)
        
        if not durations:
            return
        
        # Calculate target percentile
        durations.sort()
        target_idx = int(len(durations) * self.target_ratio)
        target_idx = min(target_idx, len(durations) - 1)
        
        new_deadline = durations[target_idx]
        
        # Update with exponential moving average (alpha = 0.3)
        alpha = 0.3
        self.mov_avg_deadline = alpha * new_deadline + (1 - alpha) * self.mov_avg_deadline
        
        self.round_deadline_history.append(self.mov_avg_deadline)
        
        logger.debug(
            f"Updated moving avg deadline: {self.mov_avg_deadline:.2f}s "
            f"(from {len(durations)} fast trainers, target_percentile={self.target_ratio})"
        )
    
    def _aggregate_pytorch(self, tres: TrainResult, rate: float):
        """Aggregate PyTorch weights."""
        logger.debug(f"Aggregating PyTorch weights with rate={rate:.4f}")

        for k, v in tres.weights.items():
            tmp = v * rate
            tmp = tmp.to(dtype=v.dtype) if tmp.dtype != v.dtype else tmp
            self.agg_weights[k] += tmp

    def _aggregate_tensorflow(self, tres: TrainResult, rate: float):
        """Aggregate TensorFlow weights."""
        logger.debug(f"Aggregating TensorFlow weights with rate={rate:.4f}")

        for idx in range(len(tres.weights)):
            self.agg_weights[idx] += tres.weights[idx] * rate
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about REFL aggregation.
        
        Returns:
            Dictionary with statistics
        """
        return {
            'stale_cached_count': len(self.stale_weights),
            'stale_applied_total': self.stale_applied_count,
            'stale_discarded_total': self.stale_discarded_count,
            'moving_avg_deadline': self.mov_avg_deadline,
            'deadline_history': self.round_deadline_history[-10:],  # Last 10 rounds
        }
