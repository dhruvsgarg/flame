# Copyright 2023 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
#
# SPDX-License-Identifier: Apache-2.0
"""Aysnc and SyncFL horizontal FL top level aggregator for FwdLLM."""

# TODO: Shift is_async param to hyperparameters
import cloudpickle
import gc
import logging
import psutil
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Union
import sklearn
import numpy as np
import yaml
from sortedcontainers import SortedDict
import torch.nn.functional as F
from flame.channel import VAL_CH_STATE_RECV, VAL_CH_STATE_SEND
from flame.common.constants import DeviceType
from flame.common.util import weights_to_device, weights_to_model_device
from flame.config import OptimizerType, TrainerAvailState
from flame.end import KEY_END_STATE, PROP_END_AVL_STATE, VAL_END_STATE_NONE
from flame.mode.composer import CloneComposer
import pickle
from flame.mode.horizontal.syncfl.top_aggregator import (
    TAG_AGGREGATE,
    TAG_DISTRIBUTE,
)
from sklearn.metrics import (
    confusion_matrix,
    matthews_corrcoef,
)
from flame.mode.horizontal.asyncfl.top_aggregator import (
    RECV_TIMEOUT_WAIT_S,
    TopAggregator as AsyncTopAgg,
    _SIM_GATE_MAX_PASSES,
    _SIM_GATE_POLL_TICK_S,
    _SIM_ORDER_SLACK_S,
)
from flame.mode.message import MessageType
from flame.mode.horizontal.client_duration import real_client_task_train_duration
from flame.mode.tasklet import Loop, Tasklet
from flame.sim.virtual_clock import SimReorderBuffer
from flame.optimizer.train_result import TrainResult
from flame.selector.oort import (
    PROP_DATASET_SIZE,
    PROP_LAST_SELECTED_ROUND,
    PROP_LAST_EVAL_ROUND,
    PROP_CLIENT_TASK_TRAIN_DURATION,
    PROP_ROUND_START_TIME,
    PROP_STAT_UTILITY,
    PROP_UPDATE_COUNT,
)
from flame.selector.properties import PROP_SIM_SEND_TS
import functorch as fc
import torch

from torch.nn import CrossEntropyLoss
import flame.monitor.runtime
from flame.monitor.runtime import FwdLLMStage, timer_decorator
import math

from flame import telemetry
from flame.telemetry.events import (
    build_agg_eval, build_agg_round, build_utility_belief, build_comm,
    build_version_bump_census,
)


logger = logging.getLogger(__name__)

PROP_ROUND_END_TIME = "round_end_time"

SEND_TIMEOUT_WAIT_S = 90  # 90 seconds timeout

# How long a trainer can sit in the per-round reselection cache
# (_round_selected_ends, reselect_each_iteration=False) without a real
# accepted contribution before it's treated as stuck and pruned/backfilled,
# same as an explicit departure (see _prune_departed_from_round_cache).
# Deliberately much longer than SEND_TIMEOUT_WAIT_S/RECV_TIMEOUT_WAIT_S
# (per-message waits) since this measures a full contribution cycle
# (dispatch -> real training -> accepted response), not one recv call.
ROUND_CACHE_STUCK_TIMEOUT_S = 300  # 5 minutes

# Default location of the shared examples/_metadata bundle, resolved relative
# to this library file (lib/python/flame/mode/horizontal/syncfl/ -> lib/python
# /examples/_metadata) rather than to any specific example's legacy directory
# -- this is what makes the oracular path example-agnostic, mirroring
# async_cifar10/aggregator/pytorch/main_oort_sync_agg.py's resolution.
_METADATA_DIR = Path(__file__).resolve().parents[4] / "examples" / "_metadata"
_TRACE_KEY_TO_MOBIPERF_SUB = {
    "mobiperf_2st": "states_2st",
    "mobiperf_3st_50": "states_3st_50",
    "mobiperf_3st_75": "states_3st_75",
}

import hashlib


def _calculate_hash(tensor):
    if tensor is None:
        return ""

    """Calculate a hash for a tensor for logging."""
    return hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest()


def log_error_distribution(probs, labels):
    """
    Analyzes error distribution across 4 classes and logs via logging.info.
    probs: ndarray [N, 4] - Softmax probabilities
    labels: ndarray [N] - Integer ground truth
    """
    # 1. Prediction and Error Masking
    if hasattr(probs, "detach"):
        probs = probs.detach().cpu().numpy()
    if hasattr(labels, "detach"):
        labels = labels.detach().cpu().numpy()

    # 2. Handle the float labels safely
    actual_labels = np.round(labels).astype(int)
    preds = np.argmax(probs, axis=1)

    wrong_mask = preds != actual_labels

    if not np.any(wrong_mask):
        logging.info("Accuracy is 100%.")
        return

    # 3. Filter for wrong predictions (Now safely NumPy)
    wrong_probs = probs[wrong_mask]
    logging.info(f"wrong probs len = {len(wrong_probs)}")

    # Now this will work perfectly
    confidences = np.max(wrong_probs, axis=1)

    # Margin calculation
    sorted_wrong = np.sort(wrong_probs, axis=1)
    margins = sorted_wrong[:, -1] - sorted_wrong[:, -2]

    # 3. Binning Logic (0.0 to 1.0)
    bins = np.linspace(0, 1.0, 11)
    margin_bins = np.digitize(margins, bins) - 1

    logging.debug("=== Error Distribution Analysis (Incorrect Predictions Only) ===")
    logging.debug(
        f"{'Margin Bin':<12} | {'Count':<8} | {'Avg Confidence':<15} | {'Max Confidence'}"
    )
    logging.debug("-" * 65)

    for i in range(len(bins) - 1):
        mask = margin_bins == i
        count = np.sum(mask)
        bin_label = f"{bins[i]:.1f}-{bins[i+1]:.1f}"

        if count > 0:
            avg_conf = np.mean(confidences[mask])
            max_conf = np.max(confidences[mask])
            logging.debug(
                f"{bin_label:<12} | {count:<8} | {avg_conf:<15.4f} | {max_conf:.4f}"
            )
        else:
            logging.debug(f"{bin_label:<12} | 0        | -               | -")

    # 4. Summary Statistics for "Confidently Wrong" samples
    high_margin_count = np.sum(margins > 0.5)
    logging.debug(
        f"Summary: {high_margin_count} errors have a margin > 0.5 (Confidently Wrong)."
    )


def log_margin_distribution(probs):
    # 2. Get the Top 2 values for every sample
    #    values shape: [N, 2], indices shape: [N, 2]
    top2_values, top2_indices = torch.topk(probs, k=2, dim=1)

    # 3. Calculate the Margin (Gap)
    #    Column 0 is the Winner, Column 1 is the Runner-up
    margins = top2_values[:, 0] - top2_values[:, 1]
    margins = margins.numpy()

    # 4. Define your "Indecision Zone"
    #    Samples where the gap between winner and loser is tiny (< 0.1)
    indecisive_count = np.sum(margins < 0.1)

    logging.debug(f"\n--- INDECISION REPORT ---")
    logging.debug(f"  Total Samples: {len(margins)}")
    logging.debug(
        f"  Samples with Margin < 0.1: {indecisive_count} ({(indecisive_count/len(margins))*100:.1f}%)"
    )
    logging.debug(f"  Avg Margin: {np.mean(margins):.4f}")

    # 5. (Optional) Histogram the margins to see the spread
    hist, bin_edges = np.histogram(margins, bins=10, range=(0.0, 1.0))
    logging.debug(f"  Margin Distribution: {hist}")
    logging.debug("------------------------------------------\n")


def compute_metrics_with_logging(probs, preds, out_label_ids, examples):

    logging.debug(f"'Hash' |  'Prob'  | 'Pred' | 'Actual'")

    for i, batch in enumerate(examples):
        batch = tuple(t.to("cpu") for t in batch)
        for j, example in enumerate(batch[1]):

            pred = preds[i * 8 + j]
            actual = out_label_ids[i * 8 + j]
            prob = probs[i * 8 + j]

            # 2. Print the row
            # We slice the hash to [:10] for better readability in the console
            logging.debug(
                f"{_calculate_hash(example)}... | {prob} | {pred} | {actual} "
            )

    return


@timer_decorator
def recv_fifo_wrapper(channel, ends):
    logger.debug("Entering recv_fifo_wrapper generator loop")
    for msg, metadata in channel.recv_fifo(ends):
        logger.debug(f"Yielding msg from {metadata}")
        yield msg, metadata
    logger.debug("Exiting recv_fifo_wrapper")


def charge_sim_vclock_overhead(vclock, simulated, config, span_s, label: str) -> float:
    """Fold a MEASURED aggregator-side wall span (`span_s`, seconds) into the
    vclock -- dynamically, using the live number, never a pre-profiled constant
    (#6). Sim-only and gated on `sim_model_agg_compute_time` (OFF -> no-op,
    byte-identical). Warns when the span exceeds `sim_overhead_warn_s` (excess
    sim-host overhead, not modeled deployment cost). Returns seconds charged."""
    if not span_s or span_s <= 0.0:
        return 0.0
    hp = getattr(config, "hyperparameters", None)
    _warn = getattr(hp, "sim_overhead_warn_s", None)
    if _warn and span_s > float(_warn):
        _now = f"{vclock.now:.1f}s" if vclock is not None else "n/a"
        logger.warning(
            f"[SIM_OVERHEAD] {label}={span_s:.3f}s > expected {float(_warn):.1f}s "
            f"(vclock={_now}) -- excess sim-host overhead"
        )
    if simulated and vclock is not None and getattr(
            hp, "sim_model_agg_compute_time", False):
        vclock.advance(vclock.now + span_s)
        return span_s
    return 0.0


class _OrderedContributorList(list):
    """Ordered list + `.discard()`, so `_per_agg_trainer_list` can bind straight
    to the selector's `_agg_pending_commit_ref` (duck-typed against sim's
    set-based `_sim_pending_commit`) without losing commit-order indexing."""

    def discard(self, item):
        try:
            self.remove(item)
        except ValueError:
            pass


class TopAggregator(AsyncTopAgg):
    """Top level Aggregator implements an ML aggregation
    role."""

    # The sim runs real forward-grad GPU + server eval, so its physical wall
    # legitimately exceeds the vclock budget (#6/#13); a 1x wall ceiling would
    # truncate it before vclock reached the budget. Decoupled to a generous
    # multiple so it is a runaway OUTER safety, not the primary stop (primary
    # stops: max_data_id_progress and vclock >= max_runtime_s). Override with an
    # explicit `sim_wall_ceiling_s` for a tighter bound.
    SIM_WALL_CEILING_FACTOR = 20.0

    def internal_init(self) -> None:
        """Initialize internal state for role."""
        super().internal_init()

        self._trainer_last_model_version = {}
        # Diagnostic: end -> full version_key (model_version,
        # iteration_per_data_id) of this end's last RETURNED contribution, not
        # a reduction to the bare model_version int. Kept alongside
        # _trainer_last_model_version (which staleness's scalar diff needs)
        # so comparisons can use the full key, not just its first component.
        self._trainer_last_version_key = {}
        # Diagnostic: end -> version_key this end was DISPATCHED at, for
        # every outstanding send. Set in the dispatch loop, cleared on
        # return. Lets a version-bump census count how many of the pool
        # still carry stale-version work when model_version advances.
        self._trainer_inflight_dispatch_version = {}

        self._agg_goal_cnt = 0
        self._agg_goal_weights = None
        self._agg_goal = self.config.hyperparameters.aggregation_goal or 1

        self._updates_in_queue = 0
        self._updates_received = {}
        self._per_agg_trainer_list = _OrderedContributorList()
        # Parallel to _per_agg_trainer_list: buffered per-contribution material,
        # merged into self.grad in canonical order, not raw arrival order.
        self._pending_cohort_contribs = []
        # Per-contributor raw (pre-rate-scaling) gradient L2 norm this cycle --
        # gradient values are mode-invariant given identical input+perturbation
        # seed, so this directly measures that instead of inferring it from
        # downstream cadence/variance symptoms.
        self._cycle_grad_norms = []
        # end -> canonical commit-order key (modeled_delay D, str(end)) for the
        # current cycle's cohort. Populated per contribution in aggregate_weights;
        # consumed by _canonicalize_cohort_commit_order to break equal-D ties by
        # trainer_id identically in real and sim.
        self._commit_key_by_end = {}
        self._model_version_unique_trainers = set()
        self._model_version_trainer_stats = {
            "train_duration": [],
            "partial_stat_utility": [],
        }

        self._per_round_staleness_list = []
        self._aggregator_staleness_track_rounds = []
        self._aggregator_round_avg_staleness = []
        self._per_trainer_staleness_track = {}
        self._track_trainer_version_duration_s = {}

        # Dictionary to store trainer state: Key = trainer_id, Value = version_key (model_version, iteration)
        self._trainer_state_dict = {}

        # check if distribute_weights was successful
        self._prev_distribute_weights_success = False

        self.data_id = 0
        self.total_data_bins = 150
        self._is_model_updated = False
        self._model_version = 0

        # end -> data_id it entered the sct reorder buffer under. Mirrors
        # asyncfl's `_sim_enqueue_round`, adapted to fwdllm's progress axis
        # (data_id, not round -- fwdllm's `_round` bumps once per full
        # 150-data_id lap). Lets `_sim_recv_min_grad` tell a genuine
        # carried-surplus commit (buffered under the PRIOR data_id) apart
        # from an actual pacing anomaly.
        self._sim_enqueue_data_id: dict = {}

        # Force-advance data_id after this many failed variance checks; None = disabled.
        self._max_iter_per_data_id = getattr(
            self.config.hyperparameters, "max_iterations_per_data_id", None
        )
        if self._max_iter_per_data_id is not None:
            logger.info(
                f"[MaxIterBypass] max_iterations_per_data_id={self._max_iter_per_data_id}"
            )

        # Opt-2 (charter §5c): variance-plateau stopping policy. At α=1 the
        # variance floor sits above the gate, so a data-bin crosses only on a
        # noise dip and grinds many iterations while the denoised estimate has
        # plateaued. This commits early once the variance-decay curve flattens.
        # fluxtune-only; absent/'off' => legacy max-iter cap governs.
        #   * fixed_cap: rely on max_iterations_per_data_id as the ceiling.
        #   * plateau:   ALSO commit when the relative var drop over the last N
        #                cycles < rel_delta while var is still above threshold.
        # Commit reason (natural/cap/plateau) is recorded for telemetry.
        self._var_stopping_policy = getattr(
            self.config.hyperparameters, "var_stopping_policy", None
        )
        self._var_plateau_patience = int(
            getattr(self.config.hyperparameters, "var_plateau_patience", 3)
        )
        self._var_plateau_rel_delta = float(
            getattr(self.config.hyperparameters, "var_plateau_rel_delta", 0.10)
        )
        self._force_commit_reason = None  # {natural, cap, plateau}; for agg_round
        self._grad_aware_gated_total = 0  # Opt-3: anti-aligned updates down-weighted
        if self._var_stopping_policy not in (None, "off", "fixed_cap", "plateau"):
            raise ValueError(
                f"unknown var_stopping_policy={self._var_stopping_policy!r}; "
                "expected one of off/fixed_cap/plateau"
            )
        if self._var_stopping_policy in ("fixed_cap", "plateau"):
            if (
                self._var_stopping_policy == "fixed_cap"
                and self._max_iter_per_data_id is None
            ):
                logger.warning(
                    "[VarStopPolicy] policy=fixed_cap but "
                    "max_iterations_per_data_id is unset -> no cap will fire."
                )
            logger.info(
                f"[VarStopPolicy] policy={self._var_stopping_policy} "
                f"cap(max_iter)={self._max_iter_per_data_id} "
                f"plateau_N={self._var_plateau_patience} "
                f"plateau_rel_delta={self._var_plateau_rel_delta}"
            )

        # Opt-1 (charter §5c): suppress byte-identical intra-databin weight
        # re-sends. Within a databin the WEIGHTS+GRAD_POOL payload is identical
        # across iterations, yet a trainer pulled in to refill concurrency reads
        # as "stale" (_trainer_last_model_version is written only on grad-RETURN)
        # and gets the identical payload again -- downgrade such a re-dispatch to
        # the tiny VAR=bad "keep training" message (the trainer caches its
        # weights). Unconditional invariant of the version-tracking logic, not
        # an opt-in: validated 0% redundant sends on a live pair, vs ~88-90%
        # before this landed as a flag.
        # Ends already sent the CURRENT model_version's full payload this data-bin
        # (cleared on every model_version advance). Separate from
        # _trainer_last_model_version so staleness accounting stays return-driven.
        self._weights_sent_this_cycle: set = set()
        self._redundant_weights_suppressed_total = 0
        # One-instruction-per-version_key (§H): end_id -> version_key last
        # dispatched to it. Without this the sync per-collect loop (~agg_goal
        # passes/iteration) re-floods VAR=bad to the whole cohort each pass;
        # skipping an already-served end matches async's ~1x. Re-serves
        # automatically on a version_key advance.
        self._end_served_version_key: dict = {}
        self.grad_pool = []
        self.cached_shared_grad_pool_trainable = None
        self.var = None
        # Cached scalar of self.var, set in _prepare_round_state. None
        # pre-first-aggregate.
        self._var_scalar = None
        self.ends_not_selected_yet = False
        self.iteration_per_data_id = 0

        # end_id -> {dispatch_ts, commit_ts} of its last accepted contribution,
        # emitted per-cycle as contributor_intervals for the R1/W1 rungs (§L.3).
        self._sim_contrib_intervals = {}

        # Real-side collect via streamer-free `drain_ready`, replacing recv_fifo
        # (§H, see _real_sync_recv_incremental). A/B flag, default OFF.
        self._real_drain_ready_ingest = bool(getattr(
            self.config.hyperparameters, "real_drain_ready_ingest", False))
        self._real_sync_pending: list = []
        # Async twin's buffer; kept separate so the two paths never alias.
        self._real_async_pending: list = []
        self._real_recv_seq = 0
        logger.info(f"real_drain_ready_ingest = {self._real_drain_ready_ingest}")

        # #15 compute-truthful commit gate (flag-gated; default off). The
        # earlier_stuck gate in _sim_recv_min_grad blocks on _sim_inflight_expected
        # entries stamped at DISPATCH. A trainer whose payload was sent long ago
        # but has not returned is idle-in-recv, not computing; blocking on it burns
        # the grace/failsafe and starves re-dispatch. When on, the gate only blocks
        # on a trainer still within its modeled compute window, so a stamped-but-
        # idle phantom no longer defers a ready commit. Hold-to-commit untouched.
        self._sim_compute_truthful_gate = bool(getattr(
            self.config.hyperparameters, "sim_compute_truthful_gate", False))
        # Wall seconds a dispatched grad may still plausibly be computing before it
        # is treated as idle/phantom. Only consulted when the gate flag is on. NOT
        # a universal constant -- real GPU compute time is roughly uniform across
        # trainers (unlike modeled delay D), so this should be set per-baseline
        # from that baseline's own observed real-compute p99/max + margin.
        # 10.0 is a generic fallback for configs that enable the flag without
        # setting their own derived value -- not correct for any specific baseline.
        _cap = getattr(self.config.hyperparameters, "sim_gate_compute_cap_s", 10.0)
        self._sim_gate_compute_cap_s = float(_cap) if _cap is not None else 10.0
        # end -> wall time its weights/VAR=bad payload was last sent (sim only).
        self._sim_dispatch_wall = {}

        # Selection granularity, shared by both paths (round vs +IT baselines,
        # BASELINES.md naming grammar). Cadence = how long a cohort stays
        # pinned: `round` (a full total_data_bins lap), `data_bin` (one
        # completed databin, unused so far -- wired for the ablation), or
        # `iteration` (re-select every SEND call, the default). Keyed on the
        # matching axis via _cohort_cache_key -- never data_id, which wraps
        # each lap (§F-2).
        self._reselect_cadence = self._resolve_reselect_cadence()
        # Derived bool: hot-path predicate, and back-compat for call sites.
        self._reselect_each_iteration = self._reselect_cadence == "iteration"
        self._round_selected_ends = None
        # Cadence-axis value the cohort was pinned under; a change invalidates.
        self._pinned_cohort_key = None
        # end_id -> _round_cache_clock_now() of its last real accepted
        # contribution (or of first entering the cache, if it hasn't
        # contributed yet) -- lets _prune_departed_from_round_cache also evict
        # a member that's stuck but not formally departed (see
        # ROUND_CACHE_STUCK_TIMEOUT_S).
        self._round_cache_activity_ts: dict = {}
        # reselect_each_iteration=True's selection cached per version_key
        # (mirrors the False branch's cache) -- fixes fwdllm_plus's
        # 11-13-calls-per-iteration real/sim call-count mismatch.
        self._reselect_true_cache_key = None
        self._reselect_true_cache_ends = None

        # Wire staleness_policy from config into an instance attr; without this
        # the message handler's getattr fell back to "none" for every run.
        self.staleness_policy = getattr(
            self.config.hyperparameters, "staleness_policy", None
        ) or "none"
        logger.info(f"staleness_policy = {self.staleness_policy}")

        self._optimizer_sort_value = self.config.optimizer.sort
        OPTIMIZERS_SUPPORTING_GRAD_AGGREGATION = (OptimizerType.FEDBUFF,)
        self._weighted_aggregation_enabled = (
            self._optimizer_sort_value in OPTIMIZERS_SUPPORTING_GRAD_AGGREGATION
        )
        if self.staleness_policy == "fedbuff" and not self._weighted_aggregation_enabled:
            # fedbuff accepts stale grads expecting the optimizer to down-weight
            # them by (V'-V); without a weighting optimizer the rate stays 1.0.
            logger.warning(
                f"staleness_policy=fedbuff but optimizer.sort={self._optimizer_sort_value} "
                "does not down-weight staleness (rate stays 1.0)."
            )
        if not self._weighted_aggregation_enabled:
            logger.info(
                f"Setting rate=1.0 for all updates because optimizer.sort is "
                f"{self._optimizer_sort_value}; weighted aggregation only supported by {OPTIMIZERS_SUPPORTING_GRAD_AGGREGATION}."
            )

        logger.info(f"Experiment set to run in is_async: {self.is_async}")

        self._n_aggs_completed = 0
        self._var_pass_count = 0
        self._var_total_count = 0
        self._dynamic_kc_controller = None
        _dynamic_kc_conf = self.config.selector.kwargs.get("dynamic_kc", {})
        if _dynamic_kc_conf.get("enabled", False):
            from flame.selector.dynamic_kc_controller import DynamicKCController
            from flame.selector.dynamic_kc_policy import build_policy

            _policy = build_policy(
                _dynamic_kc_conf.get("policy", "variance_based"),
                _dynamic_kc_conf.get("policy_kwargs", {}),
            )
            self._dynamic_kc_controller = DynamicKCController(
                policy=_policy,
                k_init=self._agg_goal,
                c_init=self.config.selector.kwargs.get("c", 1),
                k_min=_dynamic_kc_conf.get("k_min", 1),
                k_max=_dynamic_kc_conf.get("k_max", 100),
                c_min=_dynamic_kc_conf.get("c_min", 1),
                c_max=_dynamic_kc_conf.get("c_max", 100),
                update_every_n_aggs=_dynamic_kc_conf.get("update_every_n_aggs", 1),
            )
            logger.info(
                f"[DynamicKC] Controller initialized: policy={_policy.name()}, "
                f"k_init={self._agg_goal}, "
                f"c_init={self.config.selector.kwargs.get('c', 1)}"
            )

        # every trainer seen so far
        self.all_trainers = set()
        self.minInitialTrainers = self.config.selector.kwargs.get("minInitialTrainers")
        if self.is_async and self.minInitialTrainers is None:
            raise KeyError(
                "minInitialTrainers must be specified in selector config for async fwdllm"
            )
        self.trainer_unavail_durations = None
        self._cached_test_data = None
        logger.info("finished init for sync agg")

    @property
    def version_key(self) -> tuple[int, int]:
        """(model_version, iteration_per_data_id): fwdllm's step identity.
        data_id is NOT in the key -- model_version bumps once per completed
        data-bin, so it already identifies data_id uniquely; data_id stays a
        reporting/progress field only."""
        return (self._model_version, self.iteration_per_data_id)

    def pause_execution(self):
        time.sleep(1)
        return

    def log_memory(self, tag, device):
        """Diagnostic only -- was unconditional, paying a syscall + CUDA
        allocator query on every call regardless of log level."""
        if not logger.isEnabledFor(logging.DEBUG):
            return
        # GPU memory
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)

        # CPU memory
        process = psutil.Process()
        cpu_memory = process.memory_info().rss  # in bytes

        logger.debug(
            f"[MEM:{tag}] "
            f"GPU Allocated: {allocated/1e6:.2f} MB | "
            f"GPU Reserved: {reserved/1e6:.2f} MB | "
            f"CPU Memory: {cpu_memory/1e6:.2f} MB | "
            f"Device: {device}, aggregator"
        )

    def print_trainable_params_stats(self, location=""):
        """Diagnostic only -- was unconditional, iterating every model param
        on every call regardless of log level."""
        if not logger.isEnabledFor(logging.DEBUG):
            return
        total_params = 0
        trainable_params = 0
        total_size = 0.0
        trainable_size = 0.0

        for param in self.model.parameters():
            numel = param.numel()
            size_MB = numel * param.element_size() / 1e6

            total_params += numel
            total_size += size_MB

            if param.requires_grad:
                trainable_params += numel
                trainable_size += size_MB

        fraction = trainable_params / total_params if total_params > 0 else 0
        loc_str = f"[{location}] " if location else ""

        logger.debug(
            f"{loc_str}Trainable params: {trainable_params:,} / {total_params:,} "
            f"({fraction:.2%}), Size: {trainable_size:.2f} MB / {total_size:.2f} MB"
        )

    def get_trainable_param_state_dict(self):
        return {
            name: param.detach().cpu()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

    def print_param_dict_stats(self, param_dict, location=""):
        total_params = 0
        total_size = 0.0

        for tensor in param_dict.values():
            numel = tensor.numel()
            size_MB = numel * tensor.element_size() / 1e6

            total_params += numel
            total_size += size_MB

        loc_str = f"[{location}] " if location else ""
        print(
            f"{loc_str}Param dict stats — Total params: {total_params:,}, Size: {total_size:.2f} MB"
        )

    def _reset_agg_goal_variables(self):
        logger.debug("##### reset agg goal variables")
        # reset agg goal count
        self._agg_goal_cnt = 0

        # reset agg goal weights
        self._agg_goal_weights = None
        logger.debug(
            f"##### reset _agg_goal_cnt:{self._agg_goal_cnt}, _agg_goal_weights: "
            f"{self._agg_goal_weights}"
        )


    def read_trainer_unavailability(
        self, trace=None, base_dir: Optional[Union[str, Path]] = None
    ) -> dict:
        """Build task_id -> SortedDict(timestamp -> state) for `trace`.

        Reads from the shared examples/_metadata/ bundle (registry +
        traces), mirroring
        async_cifar10/aggregator/pytorch/main_oort_sync_agg.py's pattern --
        not from the legacy per-trainer json_scripts/trainer_*.json files.

        NOTE: the parameter MUST be named `base_dir` to match the caller in
        ClientAvailability._init_availability, which passes base_dir=...; this
        method shadows the mixin wrapper, so a mismatched name raises TypeError at
        aggregator init. Canonical impl now in
        flame.availability.trace.read_trainer_unavailability; this override is a
        deletion candidate once mobiperf/syn parity with load_trace is confirmed.
        """
        logger.info(f"Reading trainer unavailability for trace: {trace}")

        metadata_dir = Path(base_dir) if base_dir is not None else _METADATA_DIR

        registry_path = metadata_dir / "trainer_registry.yaml"
        with open(registry_path) as f:
            registry = yaml.safe_load(f)["trainers"]

        if trace in _TRACE_KEY_TO_MOBIPERF_SUB:
            sub = _TRACE_KEY_TO_MOBIPERF_SUB[trace]
            with open(metadata_dir / "availability_traces/mobiperf_traces.yaml") as f:
                traces = yaml.safe_load(f)["traces"]

            def lookup(tk: str, trainer_id: int) -> list:
                return traces[f"device_{trainer_id:03d}"][sub]

        elif trace and trace.startswith("syn_"):
            with open(metadata_dir / "availability_traces/synthetic_traces.yaml") as f:
                syn = yaml.safe_load(f)["traces"]
            if trace not in syn:
                logger.warning(f"trace {trace!r} not found in synthetic_traces.yaml")
                return None
            entry = syn[trace]
            per_trainer = entry.get("per_trainer", {}).get("n300", {})
            pattern = entry.get("pattern", [])

            def lookup(tk: str, trainer_id: int) -> list:
                return per_trainer.get(tk) or pattern

        else:
            logger.warning(f"unsupported trace name: {trace!r}")
            return None

        trainer_events_dict = {}
        for tk, meta in registry.items():
            trainer_id = meta["trainer_id"]
            task_id = meta["task_id"]
            events = lookup(tk, trainer_id)
            state_dict = SortedDict()
            for timestamp, state in events:
                state_dict[timestamp] = state
            trainer_events_dict[task_id] = state_dict

        logger.info(
            f"Loaded availability traces for {len(trainer_events_dict)} trainers "
            f"(trace={trace})"
        )
        return trainer_events_dict

    def aggregate_grads_from_trainers(
        self,
        trainer_grad,
        version_for_rate: int,
        stat_utility: float = 0.0,
        grad_for_var_check=None,
        jvp_for_snr_check=None,
    ):
        """Aggregate a single trainer's gradients into self.grad.

        All incoming tensors are scaled by `rate` before accumulation.
        If `grad_for_var_check` is provided (list of tensors), it is scaled by the
        same `rate` and appended to `self.grad_for_var_check_list` for variance checks.
        """
        # logger.info(f"trainer grad in {trainer_grad}")
        # Debug-gated: per-update GPU->CPU sha256, off the critical path unless DEBUG.
        if logger.isEnabledFor(logging.DEBUG):
            format_hash = lambda d: {k: _calculate_hash(v)[:8] for k, v in d.items()}
            logger.debug(f"Trainer grad received {format_hash(trainer_grad)}")
        self.print_trainable_params_stats(
            location="[start,aggregate_grads_from_trainers()]"
        )
        # Debug-gated: allclose forces a full-grad GPU sync per recv; numerically inert.
        if logger.isEnabledFor(logging.DEBUG):
            all_zero = all(torch.allclose(g, torch.zeros_like(g)) for g in self.grad)
            logger.debug(f"Are all grads zero initially? {all_zero}")

        self.log_memory("start aggregate_grads_from_trainers", self.device)

        # logger.info(f"len(self.model.named_parameters()):
        # {len(self.model.named_parameters())}, len(self.params):
        # {len(self.params)}") self.grad.to(DeviceType.CPU)
        # trainer_grad.to(DeviceType.CPU)
        np = self.model.named_parameters()

        # Raw (pre-rate-scaling) grad norm, trainable params only -- a fresh
        # named_parameters() call so it doesn't exhaust the `np` generator the
        # merge loop below still needs.
        self._cycle_grad_norms.append(
            self._flat_grad_norm(trainer_grad, self.model.named_parameters())
        )

        # rate = scale * alpha(staleness) + (1 - scale) * beta(stat_utility)
        # alpha: polynomial decay in staleness; beta: polynomial_upshift
        if not self._weighted_aggregation_enabled:
            logger.debug(f"no weighted aggregation, rate = 1.0")
            rate = 1.0
        else:
            staleness_val = self._model_version - version_for_rate

            if self.optimizer.agg_rate_conf["type"] == "old":
                rate = 1 / math.sqrt(1 + staleness_val)  # As per the Fedbuff paper

            elif self.optimizer.agg_rate_conf["type"] == "new":
                try:
                    scale_val = self.optimizer.agg_rate_conf["scale"]
                    a_exp_val = self.optimizer.agg_rate_conf["a_exp"]
                    b_exp_val = self.optimizer.agg_rate_conf["b_exp"]
                    rate = self.optimizer.weight_factor(
                        scale=scale_val,
                        staleness=staleness_val,
                        a_exp=a_exp_val,
                        loss=stat_utility,
                        b_exp=b_exp_val,
                        alpha_type="polynomial",
                        beta_type="polynomial_upshift",
                    )
                except Exception as e:
                    logger.warning(
                        f"Falling back to neutral rate due to error in weight_factor: {e}"
                    )
                    rate = 1.0

            elif self.optimizer.agg_rate_conf["type"] == "grad_aware":
                # Opt-3 (charter §5c): gradient-aware aggregation. The "new" rate
                # rescales magnitude by staleness×utility but can't refuse a wrong
                # direction, so anti-aligned JVP estimates still get averaged.
                # grad_aware weights by direction (align gate) and optionally
                # reliability (inverse-variance), bounded <= base so the effective
                # LR never inflates. OFF unless type set here.
                conf = self.optimizer.agg_rate_conf
                if conf.get("base", "new") == "new":
                    try:
                        base = self.optimizer.weight_factor(
                            scale=conf.get("scale", 0.4), staleness=staleness_val,
                            a_exp=conf.get("a_exp", 0.25), loss=stat_utility,
                            b_exp=conf.get("b_exp", 0.1), alpha_type="polynomial",
                            beta_type="polynomial_upshift",
                        )
                    except Exception:
                        base = 1.0
                else:
                    base = 1.0
                cos = self._cosine_flat(
                    trainer_grad, self.grad, list(self.model.named_parameters())
                )
                var_i = None
                if conf.get("inverse_var", False) and grad_for_var_check is not None:
                    try:
                        var_i = float(torch.stack(list(grad_for_var_check)).var())
                    except Exception:
                        var_i = None
                _align_floor = conf.get("align_floor", 0.0)
                rate = self._grad_aware_rate(
                    base, cos, var_i,
                    var_ref=getattr(self, "var_threshold", conf.get("var_ref", 0.3)),
                    align_gate=conf.get("align_gate", True),
                    inverse_var=conf.get("inverse_var", False),
                    align_floor=_align_floor, var_eps=conf.get("var_eps", 1e-8),
                )
                if cos is not None and cos < _align_floor:
                    self._grad_aware_gated_total = (
                        getattr(self, "_grad_aware_gated_total", 0) + 1
                    )
                    logger.info(
                        f"[GradAware] anti-aligned update gated: cos={cos:.3f} "
                        f"< floor={_align_floor} base={base:.3f} -> rate={rate:.4f}"
                    )

        if rate != 1.0:
            logger.info(
                f"Weighted received gradients by rate: {rate} with staleness: {staleness_val}, stat utility: {stat_utility}"
            )

        for i, (name, param) in enumerate(np):  # Assuming self.params is a dict
            if param.requires_grad:
                if name in trainer_grad:
                    grad_device = self.grad[i].device
                    trainer_grad[name] = trainer_grad[name].to(grad_device)
                    # Ensure the layer name exists in trainer_grad

                    # Apply scalar rate: g'_i = rate * g_i
                    self.grad[i].add_(trainer_grad[name] * rate)
                else:
                    logger.warning(f"Gradient for {name} not found in trainer_grad.")

        # Unscaled: the gate measures noise across comparable JVP samples, not
        # the model-update rate. Round-cadence's carried-surplus trainers are
        # genuinely stale (§F-17); scaling their sample toward zero shrank
        # `var` toward 0 and caused premature convergence. `rate` still
        # applies to `self.grad` above -- only this pool's input changes.
        if grad_for_var_check is not None:
            stacked = torch.stack(list(grad_for_var_check))
            self.grad_for_var_check_list.append(stacked)
            self.jvp_for_snr_check_list.append(jvp_for_snr_check)

        self.log_memory("end aggregate_grads_from_trainers", self.device)
        self.print_trainable_params_stats(
            location="[end,aggregate_grads_from_trainers()]"
        )

    def aggregate_grad_pool(self, grad_list):
        self.print_trainable_params_stats(location="[start,aggregate_grad_pool()]")
        if len(grad_list) == 0:
            self.print_trainable_params_stats(location="[end,aggregate_grad_pool()]")
            return None
        if len(grad_list) == 1:
            self.print_trainable_params_stats(location="[end,aggregate_grad_pool()]")
            return grad_list[0]
        else:
            grad = grad_list[0]
            for id, k in enumerate(grad):
                for i in range(0, len(grad_list)):
                    if i == 0:
                        grad[id] = grad_list[i][id]
                    else:
                        grad[id] += grad_list[i][id]
            self.print_trainable_params_stats(location="[end,aggregate_grad_pool()]")
            return grad

    def _sim_recv_min_grad(self, channel, recv_ends):
        """Grad-loop analog of asyncfl._sim_recv_min (§J.2).

        Ingest every ready grad into the sct-ordered reorder buffer (keyed by
        SIM_COMPLETION_TS), hold the commit while an in-flight trainer is expected
        to complete before the buffered minimum, then pop and commit the single
        smallest-sct message, advancing the virtual clock to it. Returns one
        (msg, metadata) matching next(channel.recv_fifo(...,1)), or (None, ("",
        now)) when nothing is committable.

        Not _sim_recv_min verbatim because fwdllm commits gradients one-per-call
        and releases slots on the agg-goal boundary (not per message), and must
        survive variance-FAIL rollbacks without stranding or double-committing a
        grad (§I.5). Slot/selector state is cleared at the boundary, not here.
        """
        live = [e for e in (recv_ends or []) if channel.has(e)]
        deadline = time.time() + RECV_TIMEOUT_WAIT_S
        for _pass in range(_SIM_GATE_MAX_PASSES):
            # Base probe: the live recv_ends (always drained), minus anything
            # already buffered or committed this cycle.
            _base = [
                e for e in live
                if not self._sim_buffer.has(e) and e not in self._sim_committed
            ]
            _seen = set(_base)
            # Derive gate safety from current in-memory state, before this
            # pass's ingest call. Deliberately skips the compute-truthful
            # phantom-skip filter (post-ingest) -- that could fast-path a
            # pass the real gate would still HOLD. Omitting it is only more
            # conservative, never less -- safe, just occasionally misses an
            # optimization.
            _pre_bmin = self._sim_buffer.peek_min_ts()
            _pre_inflight = [
                (e, exp) for e, exp in self._sim_inflight_expected.items()
                if not self._sim_buffer.has(e) and e not in self._sim_committed
            ]
            if getattr(self, "_sim_sct_ordered_drain", False):
                # #13: direct sct-ordered ingest. Drain each live in-flight end's
                # rx queue directly (no recv_fifo streamer) over the full in-flight
                # set. Two wins: (1) the buffer is a complete snapshot of every
                # arrived grad -- the streamer's background task/shared queue can
                # strand a delivered grad and let the clock lap it (past-dated
                # commit); (2) drain_ready sweeps ready rxqs non-blocking, so it
                # doesn't burn the full grace per not-ready end.
                to_probe = _base + [
                    e for e in self._sim_inflight_expected
                    if e not in _seen and channel.has(e)
                    and not self._sim_buffer.has(e) and e not in self._sim_committed
                ]
                if to_probe:
                    # Exact bound when known; else a poll tick (drain_ready
                    # can't block on timeout=None) -- the outer pass loop retries.
                    _timeout = self._sim_recv_timeout_s(to_probe)
                    _fast_safe = _timeout is not None and self._sim_gate_is_safe(
                        _pre_bmin, _pre_inflight
                    )
                    _probe_timeout = (
                        self._SIM_GATE_FAST_PROBE_TIMEOUT_S if _fast_safe
                        else (_timeout if _timeout is not None else _SIM_GATE_POLL_TICK_S)
                    )
                    for m, md in channel.drain_ready(
                        to_probe,
                        timeout=_probe_timeout,
                    ):
                        _e = md[0]
                        self._note_sim_known_delay(_e, m)
                        _s = m.get(MessageType.SIM_COMPLETION_TS)
                        _s = float(_s) if _s is not None else self._vclock.now
                        self._sim_buffer.add(_e, _s, (m, md))
                        if not hasattr(self, "_sim_enqueue_data_id"):
                            self._sim_enqueue_data_id = {}
                        self._sim_enqueue_data_id.setdefault(_e, self.data_id)
            else:
                # #13: probe-ceiling + ready-gating. Probe an in-flight-expected
                # end that is not already a recv_end only if it is physically ready
                # or its modeled completion `exp` is at/before the buffered minimum
                # (+slack); else the drain blocked the full grace window every pass
                # on far-future stragglers. Safe against the HOLD gate: any end that
                # could trigger `earlier_stuck` also satisfies exp <= bmin + slack,
                # so the gate's stuck end is always in this probe set.
                _bmin = self._sim_buffer.peek_min_ts()
                _probe_ceiling = (
                    _bmin + _SIM_ORDER_SLACK_S if _bmin is not None else float("inf")
                )
                to_probe = _base + [
                    e for e, exp in self._sim_inflight_expected.items()
                    if e not in _seen and channel.has(e)
                    and not self._sim_buffer.has(e) and e not in self._sim_committed
                    and (self._sim_end_has_ready_msg(channel, e) or exp <= _probe_ceiling)
                ]
                if to_probe:
                    # Exact bound when known; None to genuinely block.
                    _timeout = self._sim_recv_timeout_s(to_probe)
                    _fast_safe = _timeout is not None and self._sim_gate_is_safe(
                        _pre_bmin, _pre_inflight
                    )
                    _probe_timeout = (
                        self._SIM_GATE_FAST_PROBE_TIMEOUT_S if _fast_safe else _timeout
                    )
                    for m, md in channel.recv_fifo(
                        to_probe, first_k=len(to_probe), timeout=_probe_timeout
                    ):
                        if m is None:  # no more ready (bound expired or set drained)
                            break
                        _e = md[0]
                        self._note_sim_known_delay(_e, m)
                        _s = m.get(MessageType.SIM_COMPLETION_TS)
                        _s = float(_s) if _s is not None else self._vclock.now
                        self._sim_buffer.add(_e, _s, (m, md))
                        if not hasattr(self, "_sim_enqueue_data_id"):
                            self._sim_enqueue_data_id = {}
                        self._sim_enqueue_data_id.setdefault(_e, self.data_id)
            # Gate: earliest expected completion among un-buffered in-flight ends.
            bmin = self._sim_buffer.peek_min_ts()
            _stuck_end, min_stuck = None, None
            # #15 compute-truthful gate: only a trainer still within its modeled
            # compute window can be a live earlier-sct straggler. One whose last
            # dispatch is older than the compute cap (or never dispatched) is
            # idle-in-recv/phantom and won't produce a grad until the drain yields,
            # so blocking on it deadlocks to the failsafe. Skip it so the buffered
            # commit proceeds and the loop can re-dispatch.
            _truthful = getattr(self, "_sim_compute_truthful_gate", False)
            _now_wall = time.time()
            _cap = getattr(self, "_sim_gate_compute_cap_s", 10.0)
            for e, exp in self._sim_inflight_expected.items():
                if self._sim_buffer.has(e) or e in self._sim_committed:
                    continue
                if _truthful:
                    _dw = self._sim_dispatch_wall.get(e)
                    if _dw is None or (_now_wall - _dw) > _cap:
                        self._sim_gate_phantom_skip = getattr(
                            self, "_sim_gate_phantom_skip", 0) + 1
                        continue  # not genuinely computing -> can't block a commit
                if min_stuck is None or exp < min_stuck:
                    min_stuck, _stuck_end = exp, e
            # Cold-start gate: a trainer's FIRST contact has no
            # _sim_known_delay_s entry, so _sim_inflight_expected is never
            # armed for it and earlier_stuck above is blind to it -- round 1
            # used to commit whatever arrived first instead of the true
            # sct-minimum. Unconditional: hold while a probed end is still
            # unknown and within the compute cap of its own dispatch; clears
            # once it reports or ages out.
            unknown_stuck = any(
                e not in self._sim_known_delay_s
                and not self._sim_buffer.has(e) and e not in self._sim_committed
                and self._sim_dispatch_wall.get(e) is not None
                and (_now_wall - self._sim_dispatch_wall[e]) <= _cap
                for e in to_probe
            )
            earlier_stuck = (
                bmin is not None and min_stuck is not None
                and min_stuck + _SIM_ORDER_SLACK_S < bmin
            )
            if bmin is None and not to_probe:
                break  # nothing to commit and nothing in flight
            if not earlier_stuck and not unknown_stuck:
                break  # the buffered minimum is the true next completion
            if time.time() >= deadline:
                # #13 failsafe: the earliest-expected in-flight trainer never
                # arrived within the window. Without eviction it stays in
                # _sim_inflight_expected forever, so `earlier_stuck` re-fires the
                # full deadline every drain cycle -> pipeline starvation. Treat it
                # as lost: drop it from the expected set and commit the buffered
                # min now. Sim-only. (_stuck_end is None if unknown_stuck alone
                # triggered this -- guard the evict, that path is deadline-safe.)
                self._sim_gate_failsafe = getattr(self, "_sim_gate_failsafe", 0) + 1
                if _stuck_end is not None:
                    self._sim_inflight_expected.pop(_stuck_end, None)
                logger.info(
                    f"[SIM_GRAD_STUCK_EVICT] round={getattr(self, '_round', -1)} "
                    f"end={str(_stuck_end)[-4:]} exp={min_stuck} bmin={bmin} "
                    f"failsafe={self._sim_gate_failsafe}"
                )
                break  # failsafe: a stuck trainer never arrived; commit buffered
        popped = self._sim_buffer.pop_min()
        if popped is None:
            return None, ("", datetime.now())
        _end, sct, (m, md) = popped
        # Clock-jump clamp: a far-future straggler commit must not lap a fresh
        # in-flight cohort still expected to complete earlier.
        _now = self._vclock.now
        _min_future = None
        for e, exp in self._sim_inflight_expected.items():
            if e == _end or e in self._sim_committed:
                continue
            if exp > _now and (_min_future is None or exp < _min_future):
                _min_future = exp
        _advance_to = (
            sct if _min_future is None
            else max(_now, min(sct, _min_future + _SIM_ORDER_SLACK_S))
        )
        self._advance_sim_clock(_advance_to)
        # Past-dated-commit tracking, ported from asyncfl._sim_recv_min
        # (fwdllm's grad loop pops _sim_buffer directly, so it never had this
        # bookkeeping). A "past-dated" commit is one the clock already lapped
        # (sct < vclock by more than the gate slack). No withheld-delivery
        # case here (fwdllm's grad loop has no availability-withhold path).
        _was_recommit = _end in self._sim_committed
        self._sim_committed.add(_end)
        self._sim_inflight_expected.pop(_end, None)
        _commit_gap = self._vclock.now - sct
        # The shared visibility-lag primitive (syncfl/top_aggregator.py),
        # never previously called in fwdllm -- same value as _commit_gap
        # above, but emits the standardized update_ready_ts/
        # update_committed_ts/update_visibility_lag_s fields matching
        # felix's schema instead of a fwdllm-only ad hoc name.
        _vis_ready_ts, _vis_committed_ts, _vis_lag_s = self._update_visibility_lag(
            sct, md[1] if isinstance(md, tuple) and len(md) > 1 else None
        )
        # An item enqueued under a PRIOR data_id and only now popped is
        # carried surplus (deliberately preserved across the agg-goal
        # boundary by `_release_sim_slots_at_agg_goal`) -- expected
        # staleness, not a scheduling failure. Pruned here (not just read)
        # to keep the dict bounded to in-flight ends only.
        _enqueued_data_id = getattr(
            self, "_sim_enqueue_data_id", {}
        ).pop(_end, None)
        _is_carried_surplus = (
            _enqueued_data_id is not None and _enqueued_data_id != self.data_id
        )
        if _commit_gap > _SIM_ORDER_SLACK_S:
            _mv = m.get(MessageType.MODEL_VERSION) if isinstance(m, dict) else None
            _cur_round = getattr(self, "_round", -1)
            _round_lag = (_cur_round - int(_mv)) if _mv is not None else None
            if _is_carried_surplus:
                _src = "carried_surplus"
            elif _cur_round <= 1:
                _src = "round1"
            elif _was_recommit:
                _src = "redispatch"
            elif _round_lag is not None and _round_lag <= 1:
                _src = "fresh"
            else:
                _src = "straggler"
            if _src == "carried_surplus":
                # Expected steady-state of any c >> agg_goal fedbuff design --
                # tracked for visibility, excluded from the primary
                # (should-be-~0) pastdated counters below.
                self._sim_carried_surplus_commits = getattr(
                    self, "_sim_carried_surplus_commits", 0) + 1
                self._sim_carried_surplus_gap_max = max(
                    getattr(self, "_sim_carried_surplus_gap_max", 0.0), _commit_gap
                )
            else:
                self._sim_pastdated_commits = getattr(self, "_sim_pastdated_commits", 0) + 1
                self._sim_pastdated_gap_cum = getattr(self, "_sim_pastdated_gap_cum", 0.0) + _commit_gap
                self._sim_pastdated_gap_max = max(getattr(self, "_sim_pastdated_gap_max", 0.0), _commit_gap)
            if not hasattr(self, "_sim_pastdated_by_source"):
                self._sim_pastdated_by_source = {}
            _agg = self._sim_pastdated_by_source.setdefault(_src, [0, 0.0])
            _agg[0] += 1
            _agg[1] += _commit_gap
        # #13 freed-slot refill stamp: this commit frees a compute slot; record
        # the just-advanced vclock so the trainer refilling the slot rides THIS
        # vclock (not the round-start frontier). Spreads each cohort's expected
        # completions across the timeline (matching real's staggered returns)
        # instead of collapsing them at one frozen `_round_now`, so the drain gate
        # stops holding for a batch of same-expected stragglers. Sim-only + gated.
        if getattr(self, "_sim_staggered_redispatch", False):
            self._sim_free_slot_ts.append(self._vclock.now)
        # Grad committed -> trainer no longer in flight in virtual time -> drop it
        # from pending so it is re-pickable (_sim_hold_busy_slots reconciles too).
        self._sim_pending_commit.discard(_end)
        # MODELED_DELAY_S was already learned into _sim_known_delay_s
        # at ingest time above.
        # Reassert selected_ends == the virtual-time in-flight set after this
        # commit: recv_fifo just marked freshly-buffered ends RECVD (stripping
        # their slots), but they are still in flight until THEY commit; else the
        # in_flight telemetry undercounts.
        if getattr(self, "_inflight_residence", False):
            self._sim_hold_busy_slots(channel)
        # in_flight (virtual, dispatched-not-committed) vs physically-computing
        # selected_ends; kept for concurrency parity debugging.
        _sel = getattr(getattr(channel, "_selector", None), "selected_ends", None)
        if isinstance(_sel, dict):
            _sel_n = sum(len(v) for v in _sel.values())
        elif isinstance(_sel, (set, list)):
            _sel_n = len(_sel)
        else:
            _sel_n = -1
        logger.info(
            f"[SIM_GRAD_RECV] round={getattr(self, '_round', -1)} "
            f"end={str(_end)[-4:]} sct={sct:.1f} T_v={self._vclock.now:.1f} "
            f"buf_depth={len(self._sim_buffer)} "
            f"inflight_exp={len(self._sim_inflight_expected)} sel_ends={_sel_n} "
            f"phantom_skip={getattr(self, '_sim_gate_phantom_skip', 0)} "
            f"commit_gap_s={_commit_gap:.1f} "
            f"pastdated_n={getattr(self, '_sim_pastdated_commits', 0)} "
            f"pastdated_gap_max={getattr(self, '_sim_pastdated_gap_max', 0.0):.1f} "
            f"carried_surplus_n={getattr(self, '_sim_carried_surplus_commits', 0)}"
        )
        # Telemetry bridge: the fields above only ever reached this text log
        # line, never the structured agg_round JSONL event, so the analyzer's
        # buffer_health_over_rounds.pdf/commit_gap_cdf.pdf (built to read
        # exactly commit_gap_s/buf_depth off agg_round) silently rendered
        # nothing for fluxtune. Stash the per-commit snapshot here; the
        # agg_round emit site (end of _aggregate_grads_async's commit branch)
        # reads it back via getattr since this call's locals don't reach there.
        self._sim_last_commit_gap_s = _commit_gap
        self._sim_last_buf_depth = len(self._sim_buffer)
        self._sim_last_update_ready_ts = _vis_ready_ts
        self._sim_last_update_committed_ts = _vis_committed_ts
        self._sim_last_update_visibility_lag_s = _vis_lag_s
        return m, md

    def _release_sim_slots_at_agg_goal(self, channel, is_async):
        """Sim slot release at the agg-goal boundary. Two policies (§L):

        - async + inflight_residence (fluxtune, c >> agg_goal): commit-then-
          carry. Hold still-busy trainers before clearing and carry the surplus
          buffer to the next fedbuff step. `_sim_hold_busy_slots` holds every
          dispatched-but-not-committed trainer (computing ∪ carried) in both its
          compute slot and re-pick guard until its grad commits.
        - else (sync barriers c ≈ agg_goal, or residence off): legacy drop -- no
          surplus, so clearing is correct and flag-off is byte-identical.

        `_sim_committed` always clears so a variance-FAIL re-contributor on the
        rolled-back data_id isn't skipped.
        """
        if not self.simulated:
            return
        # Sync-only incremental-collect state (never touched by the async path
        # above) -- always safe to clear here, a no-op via getattr when unset.
        _sim_sync_pending = getattr(self, "_sim_sync_pending", None)
        if _sim_sync_pending is not None:
            _sim_sync_pending.clear()
        _sim_sync_committed = getattr(self, "_sim_sync_committed", None)
        if _sim_sync_committed is not None:
            _sim_sync_committed.clear()
        self._sim_sync_barrier_durs = []
        if is_async and getattr(self, "_inflight_residence", False):
            self._sim_hold_busy_slots(channel)   # reads buffer/in-flight -> hold before clear
            self._sim_committed.clear()
            return  # keep _sim_buffer / _sim_inflight_expected -> carry surplus
        self._sim_committed.clear()
        self._sim_buffer.clear()
        self._sim_inflight_expected.clear()
        # legacy-drop path abandons all in-flight -> nothing is pending. (async
        # path clears via _sim_hold_busy_slots' reconcile; sync barriers use the
        # random selector where the ref is inert, but keep the set bounded.)
        self._sim_pending_commit.clear()
        if is_async:
            self._sim_hold_busy_slots(channel)

    def _sim_hold_busy_slots(self, channel) -> None:
        """Assert `selected_ends` == the virtual-time in-flight set: every
        dispatched-but-not-committed trainer (`_sim_inflight_expected` ∪ buffered
        surplus ∪ `_sim_pending_commit`, the last covering first-ever dispatches
        whose delay isn't learned yet -- see the `outstanding` comment below),
        holding both its compute slot (`selected_ends`, drives
        `extra = c − len(selected_ends)`) and its re-pick guard (`all_selected`)
        until its grad commits.

        WHY: a returned-but-uncommitted trainer is still in flight in virtual time
        (its grad commits only when the vclock reaches its sct), so its slot is
        genuinely occupied. Freeing on physical RETURN collapsed `selected_ends`
        to the physically-computing few while many were truly in flight, so the
        in_flight telemetry undercounted and `extra` read false free capacity.
        Holding to COMMIT also keeps one-in-flight-per-trainer across the carry
        boundary + variance-FAIL rollbacks. Idempotent -- safe per-commit and at
        the boundary. fwdllm-class override only (adds the triplet prune).
        """
        sel = getattr(channel, "_selector", None)
        if sel is None:
            return
        requester = getattr(sel, "requester", None)
        all_selected = getattr(sel, "all_selected", None)
        selected_ends = getattr(sel, "selected_ends", None)

        buffered = set(self._sim_buffer.pending_ends())          # returned, grad carried
        # Outstanding = still in flight in virtual time. `_sim_inflight_expected`
        # only holds an entry once a trainer's delay has been LEARNED from a
        # prior message, so a trainer's FIRST-EVER dispatch is invisible to it.
        # Folding in `_sim_pending_commit` (added unconditionally at dispatch,
        # discarded only on commit) closes that gap: a first-time trainer now
        # stays held until it genuinely commits, instead of being wiped from
        # `all_selected` the instant any OTHER trainer's commit triggers this
        # reconcile. Safe to read here because `_sim_pending_commit.discard(_end)`
        # (commit path) always runs before this function for that same commit
        # event -- see `_sim_recv_min_grad`, which calls both in that order.
        #
        # Do NOT subtract `_sim_committed`: it is a stale cross-cycle marker
        # cleared only at the agg-goal boundary, so a trainer that committed then
        # got re-picked + re-dispatched (re-added to `_sim_inflight_expected`/
        # `_sim_pending_commit`) would be wrongly dropped -> re-pickable while its
        # new dispatch is in flight -> R1 violation. One that committed THIS cycle
        # is already absent from all three sets, so the subtraction was redundant.
        outstanding = set(self._sim_inflight_expected) | buffered | set(self._sim_pending_commit)
        # `_sim_pending_commit` is the authoritative virtual in-flight set;
        # reconcile it to `outstanding` in place (clear+update, never rebind -- the
        # selector holds a live reference) so a committed trainer drops out
        # (re-pickable) while a dispatched-but-uncommitted one stays. Bind the ref
        # so async_oort's eligibility filter (`_pending`) excludes it regardless of
        # all_selected churn. `|=` would never shrink -> committed trainer starved.
        self._sim_pending_commit.clear()
        self._sim_pending_commit.update(outstanding)
        sel._agg_pending_commit_ref = self._sim_pending_commit
        # Keep each trainer's contributed-tuple stamp until the aggregator
        # ADVANCES PAST that tuple (v != current), NOT until it commits. The
        # old `e in outstanding` filter dropped the stamp the instant a grad
        # committed, so the trainer was re-picked for the SAME version_key it
        # just answered -> abort_training -> phantom starvation. Now a
        # committed trainer stays excluded only while the agg is on that
        # tuple, re-entering the pool once the tuple advances.
        _cav = getattr(self, "_curr_agg_version", None)
        self._trainer_state_dict = {
            e: v for e, v in getattr(self, "_trainer_state_dict", {}).items()
            if v == _cav
        }
        # recv_fifo marked every delivered end RECVD, which the channel would
        # strip from selected_ends; reset the still-buffered (returned-but-
        # uncommitted) ends to NONE so they KEEP their slot until they commit.
        for eid in buffered:
            if channel is not None and channel.has(eid):
                channel._ends[eid].set_property(KEY_END_STATE, VAL_END_STATE_NONE)

        if isinstance(all_selected, dict):
            # Release slot + guard for every trainer no longer outstanding
            # (its grad committed this cycle) -> re-pickable next cycle.
            for eid in [e for e in list(all_selected.keys()) if e not in outstanding]:
                del all_selected[eid]
                if channel is not None and channel.has(eid):
                    channel._ends[eid].set_property(KEY_END_STATE, VAL_END_STATE_NONE)
                if isinstance(selected_ends, dict) and requester in selected_ends:
                    selected_ends[requester].discard(eid)
            # Hold every still-outstanding trainer in the re-pick guard.
            for eid in outstanding:
                if eid not in all_selected:
                    all_selected[eid] = time.time()

        # Compute slot: held by EVERY outstanding trainer (computing OR returned-
        # but-uncommitted) — both are in flight in virtual time until commit.
        if isinstance(selected_ends, dict) and requester in selected_ends:
            for eid in outstanding:
                selected_ends[requester].add(eid)

    def _aggregate_grads_async(self, tag: str) -> None:
        """
        Aggregate local model GRADIENTS asynchronously for FwdLLM.

        It receives gradients, aggregates them until _agg_goal is met,
        then performs FwdLLM variance check and model update.
        """
        channel = self.cm.get_by_tag(tag)
        if not channel:
            logger.info("No channel found")
            return

        recv_ends = channel.ends(VAL_CH_STATE_RECV)
        # Sim: the sct reorder buffer -- NOT the transient channel RECV state --
        # is the source of truth for what is committable. _sim_recv_min_grad
        # greedily drains ALL ready channel messages into _sim_buffer on its
        # first call (which empties RECV), so gating loop re-entry on RECV strands
        # the already-received grads in the buffer: 10 grads received, 1 popped,
        # then "no ends yet" forever -> agg_goal never met from a full cohort
        # (fluxtune deadlock, §K-D17). Keep draining while the buffer holds grads
        # or in-flight trainers are still expected. Real path unchanged
        # (principle #8: RECV gating is a real-transport artifact, no sim analog).
        sim_has_pending = self.simulated and (
            len(self._sim_buffer) > 0 or bool(self._sim_inflight_expected)
        )
        # Real drain_ready twin: a grad already pulled into the buffer keeps the
        # loop alive even when the channel shows no RECV-state ends this instant.
        real_has_pending = (
            not self.simulated
            and getattr(self, "_real_drain_ready_ingest", False)
            and bool(self._real_async_pending)
        )
        if recv_ends is None and not sim_has_pending and not real_has_pending:
            logger.info("no ends yet")
            return
        # time.sleep(0.1)  # Slight delay to allow messages to arrive

        # timeout=RECV_TIMEOUT_WAIT_S bounds the block on a quiet in-flight
        # trainer (default is block forever -- see channel.recv_fifo's
        # docstring). Without this, if a selected trainer never responds,
        # this call never returns, the composer loop never cycles back to
        # _distribute_weights, and _check_early_stop_conditions() (which
        # enforces max_runtime_s/max_data_id_progress) never gets a chance
        # to run -- the run hangs past its configured budget until manually
        # killed. Matches the same pattern asyncfl/top_aggregator.py's
        # _aggregate_weights already uses for this exact reason.
        #
        # Sim branch (Batch 1): instead of committing on wall arrival, order the
        # commit by the modeled completion sct via _sim_recv_min_grad (sct-ordered
        # reorder buffer + in-flight gate + virtual-clock advance). It returns ONE
        # (msg, metadata) matching next(recv_fifo(...,1)) -- one grad per call, as
        # this loop expects. Real branch byte-identical to before.
        if self.simulated:
            msg, metadata = self._sim_recv_min_grad(channel, recv_ends or [])
        elif getattr(self, "_real_drain_ready_ingest", False):
            # Streamer-free async collect: commit the earliest-arrival buffered
            # grad, minus recv_fifo's streamer stall (§H). Same one-grad-per-call
            # contract as the recv_fifo path below.
            msg, metadata = self._real_async_recv_min_grad(channel)
        else:
            msg, metadata = next(
                channel.recv_fifo(channel.ends(VAL_CH_STATE_RECV), 1,
                                  timeout=RECV_TIMEOUT_WAIT_S)
            )
        end, timestamp = metadata
        if not msg:
            logger.debug(f"No data from {end}; skipping it")
            return

        # Use new extracted helper
        if not self._process_single_trainer_message(channel, msg, end, timestamp):
            return

        logger.info(f"Received and processed grads from {end}.")

        if self._agg_goal_cnt < self._agg_goal:
            logger.info(f"Agg goal not met. Have {self._agg_goal_cnt}/{self._agg_goal}")
            channel.set_end_property(
                end, PROP_UPDATE_COUNT, self._updates_received.get(end, 0) + 1
            )
            return

        if self._agg_goal_cnt >= self._agg_goal:
            self._process_aggregation_goal_met(tag, channel, is_async=True)

    @timer_decorator
    def _release_end_on_return(self, channel, end, buffered: bool = False) -> None:
        """Release a returned trainer's slot + re-pick guard on RETURN --
        except when `_inflight_residence` is on AND this message wasn't just
        buffered, where commit-boundary release (`channel.cleanup_recvd_ends()`)
        owns the guard instead (R1, PARITY.md §3.resid: return != commit for
        fluxtune's pooled fedbuff, c >> agg_goal).

        `buffered=True`: a contribution captured in `_pending_cohort_contribs`
        can't be LOST by an early re-dispatch, so the R1 concern no longer
        requires holding the CHANNEL slot to the whole cohort's commit --
        release it now, restoring flat (not sawtoothing) concurrency to match
        true async fedbuff. `buffered=False` (default) keeps the old
        hold-to-commit behavior.

        This alone doesn't stop the freed trainer being RE-PICKED before it
        commits (wasted, not lost). `_process_single_trainer_message` closes
        that gap by binding the selector's `_agg_pending_commit_ref`.
        """
        if getattr(self, "_inflight_residence", False) and not buffered:
            return  # guard/slot held to COMMIT (see channel.cleanup_recvd_ends())
        if self.is_async:
            channel.cleanup_provided_ends(end)
        else:
            channel.cleanup_recvd_end(end)

    def _process_single_trainer_message(self, channel, msg, end, timestamp):
        # An end may only contribute once per (round, data_id,
        # iteration_per_data_id) collection cycle -- _per_agg_trainer_list is
        # reset only when that tuple advances (_process_aggregation_goal_met).
        # The trainer self-enforces this too, but resending WEIGHTS to the
        # round's frozen trainer set on every distribute loop (per-round
        # reselect) makes a duplicate/late message possible; guard here too.
        if end in self._per_agg_trainer_list:
            logger.info(
                f"Duplicate contribution from {end} for round={self._round}, "
                f"data_id={self.data_id}, iteration={self.iteration_per_data_id}; "
                f"ignoring."
            )
            if self.is_async:
                channel.cleanup_provided_ends(end)
            else:
                channel.cleanup_recvd_end(end)
            return False

        if MessageType.MODEL_VERSION in msg:
            version = msg[MessageType.MODEL_VERSION]
            if version != self._model_version:
                logger.info(
                    f"Received grad with staleness={self._model_version-version}."
                )
            # staleness_policy. REJECT: round_data_id, exact. ACCEPT: none (no
            # gate); fedbuff (consume + down-weight by V'-V) -- the async default,
            # since its carried surplus grads (§L) are stale by construction.
            # model_version alone identifies data_id (bumps once per data-bin),
            # so round_data_id needs no extra field; exact also checks
            # iteration_per_data_id.
            policy = getattr(self, "staleness_policy", "none")
            stale, stale_reason = False, None
            if policy == "round_data_id":
                if version != self._model_version:
                    stale, stale_reason = True, (
                        f"version={version} != agg model_version={self._model_version}"
                    )
            elif policy == "exact":
                msg_iter = msg.get(MessageType.ITERATION_PER_DATA_ID)
                if version != self._model_version:
                    stale, stale_reason = True, (
                        f"version={version} != agg model_version={self._model_version}"
                    )
                elif msg_iter != self.iteration_per_data_id:
                    stale, stale_reason = True, (
                        f"iteration_per_data_id={msg_iter} != "
                        f"agg iteration_per_data_id={self.iteration_per_data_id}"
                    )
            elif policy not in ("none", "fedbuff"):
                logger.warning(
                    f"Unrecognized staleness_policy={policy!r}; treating as 'none' "
                    f"(no staleness gate)."
                )

            if stale:
                logger.info(
                    f"Rejecting trainer update from {end} under "
                    f"staleness_policy={policy} ({stale_reason})."
                )
                if self.is_async:
                    channel.cleanup_provided_ends(end)
                else:
                    channel.cleanup_recvd_end(end)
                return False

        if MessageType.GRADIENTS in msg and MessageType.GRADIENTS_FOR_VAR_CHECK in msg:
            logger.info(
                f"received gradients from {end} "
                f"with model version {msg[MessageType.MODEL_VERSION]}"
            )
            self._agg_goal_cnt += 1
            # Wall of the most-recent accepted grad -> barrier_wait_s / drain_
            # tail_s in the per-round wall decomposition.
            self._last_grad_wall_ts = time.time()

            channel.set_end_property(
                end, PROP_LAST_SELECTED_ROUND, msg[MessageType.MODEL_VERSION]
            )
            channel.set_end_property(
                end, PROP_LAST_EVAL_ROUND, msg[MessageType.MODEL_VERSION]
            )
            logger.debug(
                f"Getting channel property {PROP_ROUND_START_TIME} for " f"end {end}"
            )
            round_start_time_tup = channel.get_end_property(end, PROP_ROUND_START_TIME)
            logger.debug(
                f"Returned round_start_time_tup: {round_start_time_tup} for "
                f"end {end} and timestamp {timestamp}"
            )
            if round_start_time_tup is not None:
                sent_ts = round_start_time_tup[1]
                # Selector speed signal (async_oort ranks on PROP_CLIENT_TASK_TRAIN_DURATION).
                # SIM doesn't sleep the budget, so WALL_SEND-WALL_RECV collapses to raw GPU
                # for every trainer, hiding modeled delay D -> selection flattens, cohort
                # diverges. Charge the modeled duration (= max(gpu, D)) in sim, matching
                # asyncfl/oort/syncfl; real anchors on the client stamps.
                if self.simulated:
                    _srd = msg.get(MessageType.SIM_CLIENT_TASK_TRAIN_DURATION_S)
                    round_duration = (
                        timedelta(seconds=float(_srd))
                        if _srd is not None
                        else timestamp - sent_ts
                    )
                else:
                    round_duration = real_client_task_train_duration(msg, sent_ts, timestamp)
                    if round_duration is None:
                        round_duration = timestamp - sent_ts
                channel.set_end_property(end, PROP_CLIENT_TASK_TRAIN_DURATION, round_duration)
                logger.info(
                    f"Set PROP_CLIENT_TASK_TRAIN_DURATION for {end}: {round_duration.total_seconds():.3f}s"
                )

            # Record this contribution's [dispatch, commit] interval for R1/W1
            # (§L.3): sim uses the vclock interval the trainer echoes back (exact
            # per contribution); real uses the wall interval (dispatch .. receipt).
            if getattr(self, "_sim_contrib_intervals", None) is not None:
                if self.simulated:
                    _disp = msg.get(MessageType.SIM_SEND_TS)
                    _comm = msg.get(MessageType.SIM_COMPLETION_TS)
                else:
                    _disp = (round_start_time_tup[1].timestamp()
                             if round_start_time_tup is not None else None)
                    _comm = timestamp.timestamp() if hasattr(timestamp, "timestamp") else None
                # REAL wall-clock (both modes) the drain loop accepted the
                # grad -- distinct from dispatch_ts/commit_ts (the trainer's
                # modeled schedule). Surfaces a ready-but-unprocessed grad at
                # per-contributor granularity.
                self._sim_contrib_intervals[end] = {
                    "dispatch_ts": float(_disp) if _disp is not None else None,
                    "commit_ts": float(_comm) if _comm is not None else None,
                    "processing_wall_ts": time.time(),
                }
        else:
            logger.error(
                f"Invalid message received from {end} in aggregate_weights: {msg}"
            )
            return False

        logger.debug(f"received data from {end}")
        channel.set_end_property(end, PROP_ROUND_END_TIME, (self._round, timestamp))
        # This end has just made real progress -- restart its round-cache
        # stuck-timeout clock (see ROUND_CACHE_STUCK_TIMEOUT_S /
        # _prune_departed_from_round_cache). No-op if reselect_each_iteration
        # is True (that path never populates this dict) or end isn't
        # currently cached (dict grows a harmless extra key either way).
        self._round_cache_activity_ts[end] = self._round_cache_clock_now()

        channel._selector.ordered_updates_recv_ends.append(end)
        self._updates_in_queue += 1
        self._per_agg_trainer_list.append(end)
        # Exclude this trainer from re-selection until its grad commits.
        # Real: live-rebind to `_per_agg_trainer_list`. Sim: NO add -- here runs
        # at COMMIT (`_sim_recv_min_grad` already discarded `end`), so re-adding
        # re-pins the committed trainer forever (§F.1-23). Dispatch add +
        # `_trainer_state_dict` version_key guard already cover it.
        if not getattr(self, "simulated", False):
            channel._selector._agg_pending_commit_ref = self._per_agg_trainer_list

        # Canonical commit-order key: the trainer's pure modeled delay D
        # (deterministic from the registry) + str(end) as tie-break. Lets
        # _canonicalize_cohort_commit_order reproduce real's D-ordered arrival and
        # break equal-D ties by trainer_id identically in both modes. None when
        # delays off / D unstamped -> that cycle falls back to arrival order.
        _md = msg.get(MessageType.MODELED_DELAY_S)
        self._commit_key_by_end = getattr(self, "_commit_key_by_end", {})
        self._commit_key_by_end[end] = (
            (float(_md), str(end)) if _md is not None else None
        )

        # Re-pick guard (async_oort triplet filter): record the exact version_key
        # (model_version, iteration) this trainer CONTRIBUTED to, so the selector
        # excludes it from re-selection for that key until the agg advances past
        # it. Correctness invariant: a trainer must never be picked twice for one
        # version_key, or it re-arrives, `abort_training` fires (no grad), and in
        # sim it strands in the in-flight ledger forever. Stamped on RETURN from
        # the MESSAGE's own key (NOT _curr_agg_version, which a staleness-accepted
        # late grad would mis-stamp; NOT at dispatch, which froze the pool). Both
        # modes; only for grad contributions.
        if MessageType.GRADIENTS in msg:
            _key = (
                msg.get(MessageType.MODEL_VERSION),
                msg.get(MessageType.ITERATION_PER_DATA_ID),
            )
            if None not in _key:
                self._trainer_state_dict[end] = _key

        if end not in self._updates_received.keys():
            self._updates_received[end] = 1
        else:
            self._updates_received[end] += 1

        if MessageType.GRADIENTS in msg:
            trainer_gradients = msg[MessageType.GRADIENTS]
            version_for_rate = msg[MessageType.MODEL_VERSION]
            grad_for_var_check = (
                msg[MessageType.GRADIENTS_FOR_VAR_CHECK]
                if MessageType.GRADIENTS_FOR_VAR_CHECK in msg
                else None
            )
            jvp_for_snr_check = (
                msg[MessageType.JVP_FOR_SNR_CHECK]
                if MessageType.JVP_FOR_SNR_CHECK in msg
                else None
            )
            logger.info(f"jvp_for_snr_check at aggregator: {jvp_for_snr_check}")
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"Buffering aggregate_grads_from_trainers call with grad_for_var_check: {_calculate_hash(grad_for_var_check)}"
                )
            # Buffer, don't merge yet -- self.grad's accumulation happens in
            # _process_aggregation_goal_met, in canonical order.
            # Use this message's stat_utility, not the channel property --
            # the property is set below, after this call, so reading it here
            # was always None on a trainer's first contribution (crashing
            # fedbuff's weight_factor() on `1 + None`).
            self._pending_cohort_contribs = getattr(self, "_pending_cohort_contribs", [])
            self._pending_cohort_contribs.append((
                trainer_gradients,
                version_for_rate,
                msg[MessageType.STAT_UTILITY],
                grad_for_var_check,
                jvp_for_snr_check,
            ))

            # del trainer_gradients # Free memory

        # This will add to the var check list twice, it is already added once
        # within self.aggregate_grads_from_trainers
        # if MessageType.GRADIENTS_FOR_VAR_CHECK in msg:
        #     self.grad_for_var_check_list.append(
        #         msg[MessageType.GRADIENTS_FOR_VAR_CHECK]
        #     )

        count = 0
        if MessageType.DATASET_SIZE in msg:
            count = msg[MessageType.DATASET_SIZE]
            channel.set_end_property(end, PROP_DATASET_SIZE, count)

        if MessageType.STAT_UTILITY in msg:
            # Believed (PROP_STAT_UTILITY before this overwrite, i.e. the
            # value from this end's PREVIOUS contribution) vs actual (this
            # message's fresh value) -- the staleness of whatever the
            # selector/aggregator last knew about this end's utility.
            # fwdllm_aggregator.py previously never emitted this (only
            # asyncfl/top_aggregator.py did), so selected_utility_believed_
            # vs_actual*/selected_utility_belief_gap* were structurally
            # impossible for fwdllm-family baselines regardless of selector.
            if telemetry.is_enabled():
                try:
                    _believed = channel.get_end_property(end, PROP_STAT_UTILITY)
                    _mv = msg.get(MessageType.MODEL_VERSION)
                    ev, f = build_utility_belief(
                        round_num=self._round,
                        end_id=end,
                        believed=float(_believed) if _believed is not None else None,
                        actual=float(msg[MessageType.STAT_UTILITY]),
                        # fwdllm's round stays coarse (can sit at 1 for an
                        # entire run); self._model_version advances every
                        # completed aggregation cycle (same quantity
                        # agg_round's own staleness list uses), so it's the
                        # meaningful staleness axis here, not round-based.
                        staleness=(self._model_version - _mv)
                        if isinstance(_mv, int) else None,
                        extra={
                            "data_id": self.data_id,
                            "iteration_per_data_id": self.iteration_per_data_id,
                        },
                    )
                    telemetry.emit(ev, **f)
                except Exception as e:  # telemetry must never break training
                    logger.debug(f"utility_belief telemetry emit failed: {e}")
            logger.info(
                f"received stat_utility from {end} "
                f"msg[MessageType.STAT_UTILITY] {msg[MessageType.STAT_UTILITY]}"
            )
            channel.set_end_property(
                end, PROP_STAT_UTILITY, msg[MessageType.STAT_UTILITY]
            )
            logger.info(f"grad_pool already has: {len(self.grad_pool)}")

        version = msg.get(MessageType.MODEL_VERSION, "unknown")
        self._trainer_last_model_version[end] = version
        # Diagnostic: full version_key of the RETURNED contribution, not the
        # reduced scalar -- lets a real/sim comparison verify iteration never
        # carries hidden staleness info the bare model_version diff would
        # mask. getattr-guarded: test doubles that bypass internal_init()
        # only stub the dicts they exercise.
        if getattr(self, "_trainer_last_version_key", None) is not None:
            self._trainer_last_version_key[end] = (
                version, msg.get(MessageType.ITERATION_PER_DATA_ID)
            )
        # Diagnostic: this end's dispatch is no longer outstanding.
        if getattr(self, "_trainer_inflight_dispatch_version", None) is not None:
            self._trainer_inflight_dispatch_version.pop(end, None)
        logger.info(
            f"Received grads from {end}. It was trained on model version {version}, with {count} samples"
        )
        # Release the slot/guard on return -- immediately if this message's
        # gradients were just buffered (safe to release, see
        # _release_end_on_return); otherwise the async residence path defers
        # to COMMIT. (cleanup_recvd_end() is sync-only, random selector;
        # async selectors only implement the batch _cleanup_provided_ends path.)
        self._release_end_on_return(channel, end, buffered=MessageType.GRADIENTS in msg)
        return True

    def _log_and_reset_model_version_stats(self):
        """Log a CSV summary of trainer participation for the completed model
        version window, then reset accumulators for the next window.

        Window = all aggregation cycles that ran between two consecutive
        variance-threshold breaches.
        """

        def compute_percentiles(values, reverse=False):
            if not values:
                return -1, -1, -1, -1, -1, -1, -1, -1
            arr = np.array(values)

            # This of this as the equivalent of sorting the array in reverse order where
            p1, p5, p20, p30, p50, p75, p90, p99 = (
                (99, 95, 80, 70, 50, 25, 10, 1) if reverse else (1, 5, 20, 30, 50, 75, 90, 99)
            )
            return (
                float(np.percentile(arr, p1, method='lower')),
                float(np.percentile(arr, p5, method='lower')),
                float(np.percentile(arr, p20, method='lower')),
                float(np.percentile(arr, p30, method='lower')),
                float(np.percentile(arr, p50, method='lower')),
                float(np.percentile(arr, p75, method='lower')),
                float(np.percentile(arr, p90, method='lower')),
                float(np.percentile(arr, p99, method='lower')),
            )

        n_unique = len(self._model_version_unique_trainers)
        rd_p1, rd_p5, rd_p20, rd_p30, rd_p50, rd_p75, rd_p90, rd_p99 = (
            compute_percentiles(self._model_version_trainer_stats["train_duration"])
        )
        su_p1, su_p5, su_p20, su_p30, su_p50, su_p75, su_p90, su_p99 = (
            compute_percentiles(
                self._model_version_trainer_stats["partial_stat_utility"], reverse=True
            )
        )

        logger.info(
            f"==== version_key advanced to {self._curr_agg_version} with updates from {n_unique} unique trainers. Stats of participating trainers: \n"
            f"p1, p5, p20, p30, p50, p75, p90, p99 of train duration \n{rd_p1:.3f}, {rd_p5:.3f}, {rd_p20:.3f}, {rd_p30:.3f}, {rd_p50:.3f}, {rd_p75:.3f}, {rd_p90:.3f}, {rd_p99:.3f} \n"
            f"p1, p5, p20, p30, p50, p75, p90, p99 of partial stat utilities \n{su_p1:.4f}, {su_p5:.4f}, {su_p20:.4f}, {su_p30:.4f}, {su_p50:.4f}, {su_p75:.4f}, {su_p90:.4f}, {su_p99:.4f}"
        )

        # Reset accumulators for the next model version window
        self._model_version_unique_trainers = set()
        self._model_version_trainer_stats = {
            "train_duration": [],
            "partial_stat_utility": [],
        }

    def _build_dynamic_kc_metrics(self, channel) -> dict:
        var_pass_rate = (
            self._var_pass_count / self._var_total_count
            if self._var_total_count > 0 else 0.0
        )
        staleness_list = self._per_round_staleness_list
        avg_staleness = (
            sum(staleness_list) / len(staleness_list) if staleness_list else 0.0
        )
        p75_staleness = (
            float(sorted(staleness_list)[int(0.75 * len(staleness_list))])
            if staleness_list else 0.0
        )

        _PROP_AVL_STATE = "avl_state"
        task_eligible_states = self.config.selector.kwargs.get(
            "task_eligible_states", {}
        )
        from flame.config import TrainerAvailState
        train_eligible = set(
            task_eligible_states.get("train", [TrainerAvailState.AVL_TRAIN.value])
        )
        eval_eligible = set(
            task_eligible_states.get(
                "eval",
                [TrainerAvailState.AVL_EVAL.value, TrainerAvailState.AVL_TRAIN.value],
            )
        )
        n_eligible_train = 0
        n_eligible_eval = 0
        for end_id in (channel.ends() or []):  # channel.ends() can transiently return None
            avl_state = channel.get_end_property(end_id, _PROP_AVL_STATE)
            if avl_state in train_eligible:
                n_eligible_train += 1
            if avl_state in eval_eligible:
                n_eligible_eval += 1

        _policy_kwargs = self.config.selector.kwargs.get("dynamic_kc", {}).get(
            "policy_kwargs", {}
        )
        target_iter = _policy_kwargs.get(
            "target_iter_per_data_id", self._max_iter_per_data_id
        )

        return {
            "var_pass_rate": var_pass_rate,
            "var_last": self.var if self.var is not None else 0.0,
            "avg_staleness": avg_staleness,
            "p75_staleness": p75_staleness,
            "model_version": self._model_version,
            "n_aggs_completed": self._n_aggs_completed,
            "var_threshold": getattr(self, "var_threshold", None),
            "n_eligible_train": n_eligible_train,
            "n_eligible_eval": n_eligible_eval,
            "iteration_per_data_id": self.iteration_per_data_id,
            "data_id": self.data_id,
            "max_iter_per_data_id": self._max_iter_per_data_id,
            "target_iter_per_data_id": target_iter,
        }

    def _canonicalize_cohort_commit_order(self):
        """Reorder this cycle's cohort commits to a canonical (D, trainer_id)
        order, identically in real and sim.

        Real receives updates in modeled-delay (D) order; sim commits in sct order
        (= D order). The sole residual divergence is the tie-break when two
        trainers share a D: real breaks it by physical arrival, sim by sct-sort.
        Sort by (D, str(end)) so equal-D ties break by trainer_id in both modes.

        Not cosmetic: `_pending_cohort_contribs` (buffered, not yet merged into
        self.grad) is permuted in lockstep, so self.grad's later summation
        replays in this canonical order rather than raw arrival order --
        float addition isn't associative, and grad_aware's rate reads the
        running self.grad, so arrival-order noise was a real divergence, not
        just float dust. Both lists share one append site
        (_process_single_trainer_message), so they stay 1:1.

        No-op unless every contributor stamped a modeled delay AND a tie actually
        changes the order.
        """
        ends = self._per_agg_trainer_list
        n = len(ends)
        if n < 2:
            return
        keys = [self._commit_key_by_end.get(e) for e in ends]
        if any(k is None for k in keys):
            return  # delays off / a contributor without a stamp → arrival order
        perm = sorted(range(n), key=lambda i: keys[i])
        if perm == list(range(n)):
            return  # already canonical (the common, non-tie path)
        # In-place, not rebind: `_agg_pending_commit_ref` holds a live reference.
        self._per_agg_trainer_list[:] = [ends[i] for i in perm]
        self._pending_cohort_contribs = [self._pending_cohort_contribs[i] for i in perm]
        logger.info(
            f"[COMMIT_CANON] equal-D tie → reordered {n}-cohort to (D,id) order "
            f"(perm={perm}); pending merge + receive order now real↔sim identical."
        )

    @timer_decorator
    def _replay_buffered_cohort_contribs(self):
        """Merge buffered per-trainer contributions into self.grad in canonical
        order. Split out and timed separately so this burst's own wall cost
        is visible in agg_step_timing_breakdown. getattr default: a caller
        that pre-populates self.grad (e.g. a test double) never buffers
        anything, so this is a no-op for it.
        """
        for (
            _pc_grad,
            _pc_version_for_rate,
            _pc_stat_utility,
            _pc_grad_for_var_check,
            _pc_jvp_for_snr_check,
        ) in getattr(self, "_pending_cohort_contribs", []):
            self.aggregate_grads_from_trainers(
                _pc_grad,
                version_for_rate=_pc_version_for_rate,
                stat_utility=_pc_stat_utility,
                grad_for_var_check=_pc_grad_for_var_check,
                jvp_for_snr_check=_pc_jvp_for_snr_check,
            )
        self._pending_cohort_contribs = []

    @timer_decorator
    def _process_aggregation_goal_met(self, tag, channel, is_async=False):
        logger.info(
            f"Aggregation goal {self._agg_goal} reached. Performing FwdLLM aggregation."
        )

        # Canonicalize this cohort's commit order to (D, trainer_id) before the
        # telemetry snapshot and aggregate() so the recorded receive order and the
        # split-half var use the same deterministic order in real and sim. Gated
        # on commit-key state -> skipped when delays are off (arrival order).
        # Timed inline to sub-phase drain_tail_s.
        _canon_wall_start = time.time()
        if getattr(self, "_commit_key_by_end", None):
            self._canonicalize_cohort_commit_order()
        _drain_tail_canonicalize_s = time.time() - _canon_wall_start

        # Merge this cohort's buffered contributions into self.grad now, in
        # the canonical order just established -- not raw arrival order. Must
        # run before telemetry below reads _cycle_grad_norms/
        # grad_for_var_check_list, both populated here.
        _replay_wall_start = time.time()
        self._replay_buffered_cohort_contribs()
        _drain_tail_replay_s = time.time() - _replay_wall_start

        # Snapshot for this cycle's agg_round telemetry (emitted further down,
        # after self._per_agg_trainer_list is cleared and self._model_version
        # may have advanced -- see build_agg_round call below).
        _cycle_contributors = list(self._per_agg_trainer_list)
        _cycle_grad_norm_list = list(self._cycle_grad_norms)
        _cycle_target_version = self._model_version
        _cycle_speed_s = []
        _cycle_stat_utility = []
        _cycle_staleness = []
        # end_id -> wall seconds between send (PROP_ROUND_START_TIME) and
        # this contribution being received/processed (set alongside
        # PROP_ROUND_DURATION in _process_single_trainer_message) -- the
        # same quantity asyncfl/top_aggregator.py reports as agg_observed_s,
        # letting analyze_run.py's aggregator-overhead sanity plots
        # (runtime_agg_vs_trainer/runtime_overhead_*) work for fwdllm too.
        _cycle_agg_observed_s = {}

        # Accumulate model-version-window stats; reset only when variance threshold is breached.
        for trainer_update in self._per_agg_trainer_list:
            self._model_version_unique_trainers.add(trainer_update)
            train_duration = channel.get_end_property(
                trainer_update, PROP_CLIENT_TASK_TRAIN_DURATION
            )
            if train_duration is not None:
                self._model_version_trainer_stats["train_duration"].append(
                    train_duration.total_seconds()
                )
                _cycle_speed_s.append(train_duration.total_seconds())
                _cycle_agg_observed_s[trainer_update] = train_duration.total_seconds()
            partial_stat_utility = channel.get_end_property(
                trainer_update, PROP_STAT_UTILITY
            )
            if partial_stat_utility is not None:
                self._model_version_trainer_stats["partial_stat_utility"].append(
                    partial_stat_utility
                )
                _cycle_stat_utility.append(partial_stat_utility)
            _trainer_version = self._trainer_last_model_version.get(trainer_update)
            if _trainer_version is not None:
                _cycle_staleness.append(_cycle_target_version - _trainer_version)

        self.grad_pool.append(self.grad)
        # Debug-gated: full-model GPU->CPU sha256 (~1s) per commit; free unless DEBUG.
        if logger.isEnabledFor(logging.DEBUG):
            format_hash = lambda d: [_calculate_hash(v) for v in d]
            logger.debug(
                f"self.grad when agg goal met - length : {len(self.grad)} - hash :  {format_hash(self.grad)}"
            )

        self.add_local_trained_result(0, self.grad, self._agg_goal_cnt)

        self.fmodel, self.params, self.buffers = fc.make_functional_with_buffers(
            self.model
        )
        self.grad = [torch.zeros_like(p) for p in self.params]

        # Set before aggregate() — that function rolls back weights on var failure and clears the flag.
        self._force_commit_this_cycle = (
            self._max_iter_per_data_id is not None
            and (self.iteration_per_data_id + 1) >= self._max_iter_per_data_id
        )
        _force_commit_planned = self._force_commit_this_cycle  # snapshot for logging
        if _force_commit_planned:
            logger.info(
                f"[MaxIterBypass] cap reached "
                f"(iter+1={self.iteration_per_data_id + 1} >= "
                f"cap={self._max_iter_per_data_id}); will force-commit weights "
                f"if variance check fails in aggregate()."
            )

        # Per-round wall decomposition (#6): the FedAvg merge, the dispatch->last-
        # grad barrier wait, and the last-grad->commit drain tail (a real-transport
        # artifact the sim's all-k barrier doesn't model). All wall (real); the
        # sim advances the vclock, so these read ~0 there UNLESS the fold flag
        # below is on -- _aggregate_fedavg_vclock_s makes that measurable
        # instead of asserted. eval_s/_eval_vclock_s (below) stay None
        # permanently: eval_model() is now backgrounded on a daemon thread
        # (like felix's evaluate()) rather than measured synchronously here,
        # so there is no synchronous eval duration on this critical path to
        # charge to the vclock in either mode -- the asymmetry
        # sim_model_eval_time corrected for no longer exists.
        _agg_vclock_start = getattr(self, "vclock_now", None)
        _disp = getattr(self, "_round_dispatch_wall_ts", None)
        _lastg = getattr(self, "_last_grad_wall_ts", None)
        _agg_start_wall = time.time()
        # drain-tail: last-grad-received -> aggregate() start (agg-side serial
        # wall: canonicalize + replay + residual). Computed here so it can be
        # charged alongside the FedAvg merge below.
        _barrier_wait_s = (_lastg - _disp) if (_disp and _lastg) else None
        _drain_tail_s = (_agg_start_wall - _lastg) if _lastg else None
        self.aggregate(self._round)
        _aggregate_fedavg_s = time.time() - _agg_start_wall
        # #6: charge the measured drain-tail + FedAvg merge wall to the vclock --
        # genuine server-step compute that runs every cycle but was never
        # credited. See `charge_sim_vclock_overhead`.
        _vc = getattr(self, "_vclock", None)
        charge_sim_vclock_overhead(
            _vc, self.simulated, self.config, _drain_tail_s, "drain_tail")
        charge_sim_vclock_overhead(
            _vc, self.simulated, self.config, _aggregate_fedavg_s, "fedavg")
        _agg_vclock_end = getattr(self, "vclock_now", None)
        _aggregate_fedavg_vclock_s = (
            _agg_vclock_end - _agg_vclock_start
            if _agg_vclock_start is not None and _agg_vclock_end is not None
            else None
        )
        # drain_tail_s sub-phases: canonicalize + replay are the two known
        # occupants; residual is everything else in the window.
        _drain_tail_residual_s = (
            _drain_tail_s - _drain_tail_canonicalize_s - _drain_tail_replay_s
            if _drain_tail_s is not None else None
        )
        _eval_s = None  # permanently None: eval is now backgrounded,
        _eval_vclock_s = None  # never measured synchronously on this path anymore.

        _var_thr = getattr(self, "var_threshold", None)
        # Cached scalar; getattr since this can run pre-first-aggregate.
        _var_val = getattr(self, "_var_scalar", None)
        _ratio = (
            _var_val / _var_thr
            if (_var_val is not None and _var_thr not in (None, 0))
            else None
        )
        logger.info(
            f"[IterProgress] data_id={self.data_id} "
            f"iter={self.iteration_per_data_id} "
            f"max_iter={self._max_iter_per_data_id} "
            f"var={_var_val} var_thr={_var_thr} ratio={_ratio} "
            f"var_good_enough={self.var_good_enough} "
            f"force_commit_planned={_force_commit_planned}"
        )

        # Snapshot the cycle identity BEFORE the pass/fail branch mutates
        # data_id/iteration_per_data_id. The emitted `data_id`/`iteration` fields
        # are post-mutation (a commit advances data_id and zeroes iteration) --
        # fine for the progress axis but ambiguous for the variance-cadence rungs.
        # `cycle_data_id`/`cycle_iteration` unambiguously identify the data_id this
        # cycle worked on and its 0-based attempt index.
        _cycle_data_id = self.data_id
        _cycle_iteration = self.iteration_per_data_id
        # Same pre-mutation snapshot for model_version (bumped below on a
        # commit): the version this cycle worked ON, readable off agg_round
        # directly instead of cross-referencing the DK-controller status dict.
        _cycle_model_version = self._model_version
        # Pool sizes at the variance gate (before a commit clears grad_pool):
        # grad_pool = realized contributions this data_id (G2); cached_v = carried
        # aggregated pool across variance-FAIL rollbacks (V3). getattr-guarded so
        # test doubles without the pools still emit.
        _grad_pool = getattr(self, "grad_pool", None)
        _grad_pool_size = len(_grad_pool) if _grad_pool is not None else None
        _cached_v = getattr(self, "cached_shared_grad_pool_trainable", None)
        _cached_v_size = len(_cached_v) if _cached_v is not None else 0
        # Per-contributor [dispatch, commit] intervals for R1/W1: one entry per
        # end that committed into this cycle. History preserved since each cycle
        # emits its own list. Explicit (agg_version_at_commit,
        # dispatch_version) pair per contributor -- direct evidence for staleness
        # without cross-referencing the separate `staleness` list by position.
        # version_key fields carry the full (model_version, iteration_per_data_id)
        # 2-tuple alongside the bare-int reduction the scalar `staleness` list
        # uses, so a real/sim diff can check whether iteration ever hides
        # information the bare model_version diff would miss.
        _contrib_map = getattr(self, "_sim_contrib_intervals", None) or {}
        _version_key_map = getattr(self, "_trainer_last_version_key", None) or {}
        _agg_version_key_at_commit = (_cycle_model_version, _cycle_iteration)
        _contributor_intervals = [
            {"end": str(_e), **_contrib_map.get(_e, {"dispatch_ts": None,
                                                     "commit_ts": None,
                                                     "processing_wall_ts": None}),
             "dispatch_model_version": self._trainer_last_model_version.get(_e),
             "agg_model_version_at_commit": _cycle_target_version,
             "dispatch_version_key": _version_key_map.get(_e),
             "agg_version_key_at_commit": _agg_version_key_at_commit}
            for _e in _cycle_contributors
        ]

        if self.var_good_enough:
            _pass_kind = (
                "FORCE-COMMITTED (max_iter bypass)"
                if _force_commit_planned and self.var > _var_thr
                else "PASSED"
            )
            logger.info(
                f"Variance check {_pass_kind}. Evaluating model and advancing data_id."
            )
            self.iteration_per_data_id += 1
            # eval_model() used to run synchronously, inline, here -- on the
            # critical path in BOTH real and sim mode (the reason
            # sim_model_eval_time's vclock fold existed: real paid this wall
            # cost every commit, sim didn't, without the fold). Mirrors
            # felix's own evaluate() pattern: snapshot now (cheap, main
            # thread), run the actual test-set pass in a daemon thread so the
            # aggregator's dispatch/commit loop never waits on it -- in
            # EITHER mode, symmetrically, so the asymmetry the fold corrected
            # for no longer exists. aggregate() itself is NOT backgrounded
            # here or anywhere -- it produces the model trainers need
            # immediately and must stay synchronous.
            #
            # Snapshot the cycle identity BEFORE launching: the thread may
            # still be running once self.data_id/self._round/
            # self.iteration_per_data_id have moved on to a LATER cycle, so
            # the emit closure must report what THIS cycle worked on. Kept as
            # separate locals from _cycle_data_id/_cycle_iteration since
            # _cycle_iteration is pre- the += 1 two lines up and the
            # telemetry contract below is post-, matching the old
            # synchronous numbers exactly.
            _eval_round_id = self._round
            _eval_data_id = self.data_id
            _eval_iteration = self.iteration_per_data_id
            _eval_model_snapshot = self._eval_snapshot_model()
            if _eval_model_snapshot is None:
                logger.debug(
                    "prior async fwdllm eval still running; skipping this cycle's eval"
                )
            else:
                def _fwdllm_eval_job(
                    _model=_eval_model_snapshot, _round_id=_eval_round_id,
                    _data_id=_eval_data_id, _iteration=_eval_iteration,
                ):
                    try:
                        _wall0 = time.time()
                        result, _, _ = self.eval_model(model=_model)
                        _wall_s = time.time() - _wall0  # diagnostic only, not folded
                        logger.info(
                            f"Round {_round_id}, Data ID {_data_id} "
                            f"Eval Loss: {result['eval_loss']}"
                        )
                        if telemetry.is_enabled():
                            ev, fields = build_agg_eval(
                                round_num=_round_id,
                                metrics={
                                    "test-loss": result.get("eval_loss"),
                                    "test-accuracy": result.get("acc"),
                                    "mcc": result.get("mcc"),
                                    # fwdllm's round is coarse (advances only once
                                    # all total_data_bins data_ids finish) --
                                    # data_id/iteration_per_data_id let the
                                    # analyzer's progress_key()
                                    # (scripts/analysis/analyze_run.py) place this
                                    # eval on a meaningful x-axis instead of
                                    # collapsing every eval in a round onto one
                                    # point.
                                    "data_id": _data_id,
                                    "iteration_per_data_id": _iteration,
                                    # Diagnostic-only wall duration of the
                                    # backgrounded eval -- NOT folded into the
                                    # vclock (that asymmetry no longer exists now
                                    # that both modes background it symmetrically).
                                    "eval_wall_s": _wall_s,
                                },
                            )
                            telemetry.emit(ev, **fields)
                    except Exception as e:  # eval must never break training
                        logger.warning(f"[FWDLLM_EVAL] failed (non-fatal): {e}")
                    finally:
                        self._eval_inflight = False

                threading.Thread(target=_fwdllm_eval_job, daemon=True).start()
            self.data_id += 1
            self.iteration_per_data_id = 0
            self._is_model_updated = True

            # model_version bumps once per completed data-bin, unconditionally
            # (was gated by the now-purged inc_model_version_per_data_id flag;
            # every baseline already ran with it True). version_key =
            # (model_version, iteration) then uniquely identifies a step
            # without needing data_id in the key.
            _old_mv = self._model_version
            self._model_version += 1

            # Diagnostic: snapshot the pool's in-flight staleness mix at the
            # instant of the bump -- how many trainers are still carrying a
            # dispatch from the version that just aged out, and by how much.
            if telemetry.is_enabled():
                try:
                    ev, fields = build_version_bump_census(
                        old_model_version=_old_mv,
                        new_model_version=self._model_version,
                        data_id=self.data_id,
                        inflight=dict(
                            getattr(self, "_trainer_inflight_dispatch_version", {})
                        ),
                        vclock_now=getattr(self, "vclock_now", None),
                    )
                    telemetry.emit(ev, **fields)
                except Exception as e:  # telemetry must never break training
                    logger.debug(f"version_bump_census telemetry emit failed: {e}")

            # Opt-1: the data-bin (and model_version) just advanced -> every
            # trainer is genuinely stale, so the "already sent this cycle" set must
            # start empty. getattr-guarded for test doubles that bypass __init__.
            if getattr(self, "_weights_sent_this_cycle", None) is not None:
                self._weights_sent_this_cycle.clear()

            self._log_and_reset_model_version_stats()

            if self.data_id == self.total_data_bins:
                logger.info(
                    f"All data bins complete. Incrementing round to {self._round + 1}"
                )
                self._round += 1
                self.data_id = 0
                channel.set_property("round", self._round)

                # fwdllm's TopAggregator extends the asyncfl base (not
                # syncfl's), which has no rounds-based stop condition of its
                # own -- self._work_done is otherwise never set here, so the
                # composer loop (Loop(loop_check_fn=lambda: self._work_done))
                # never exits and the aggregator process runs forever
                # regardless of hyperparameters.rounds.
                self._work_done = self._round > self.config.hyperparameters.rounds
                if self._work_done:
                    logger.info(
                        f"rounds={self.config.hyperparameters.rounds} reached "
                        f"at round {self._round}; stopping run."
                    )

        else:
            logger.info(
                f"Variance check FAILED. Retrying on same data_id {self.data_id}."
            )
            self.iteration_per_data_id += 1
            self._is_model_updated = False

        if telemetry.is_enabled():
            # Speedup instrumentation (#13): the sim must run virtual time faster
            # than wall (sim_rate >= 1). fwdllm emitted vclock_now but no paired
            # wall stamp, so a slowdown was invisible. Emit wall_elapsed_s in both
            # modes (real needs it for wall_speedup = real_wall/sim_wall) and
            # sim_rate = vclock/wall in sim only. agg_start_time_ts is re-anchored
            # past the join wait, so this excludes the initial join stall.
            _wall_elapsed_s = time.time() - getattr(
                self, "agg_start_time_ts", time.time()
            )
            _sim_rate = None
            if self.simulated and getattr(self, "_vclock", None) is not None:
                _sim_rate = (
                    float(self._vclock.now) / _wall_elapsed_s
                    if _wall_elapsed_s > 0
                    else None
                )
                # Live speedup log (throttled ~30s). The inherited base
                # [VCLOCK_PROGRESS] lives in increment_round, which fwdllm's
                # composer loop bypasses, so emit it here.
                _last = getattr(self, "_last_vclock_log_wall_ts", 0.0)
                if time.time() - _last >= 30.0 and _sim_rate is not None:
                    _slow = " SLOWDOWN" if _sim_rate < 1.0 else ""
                    logger.info(
                        f"[VCLOCK_PROGRESS] vclock={float(self._vclock.now):.1f}s "
                        f"wall={_wall_elapsed_s:.1f}s sim_rate={_sim_rate:.3f}{_slow} "
                        f"(virtual-s/wall-s) round={self._round} "
                        f"data_id={self.data_id}"
                    )
                    self._last_vclock_log_wall_ts = time.time()

            # Intrinsic algorithmic span of this cycle (#6 anchor): the genuine
            # per-cycle work the sim charges to the vclock -- the barrier (slowest
            # committed trainer's intrinsic compute+delay). Mirrors the sim vclock
            # composition exactly: the FedAvg merge is excluded because the sim
            # does not charge it to the vclock. `+ (_eval_s or 0.0)` is a no-op --
            # eval is now backgrounded off the critical path in both modes, so
            # _eval_s stays permanently None (kept rather than deleted so this
            # doesn't silently break if a future change reintroduces a
            # synchronous eval measurement here). Emitted in both modes so the
            # clock-rate rungs anchor real on intrinsic time instead of raw wall
            # Δts (real's wall bundles a ~constant inter-round transport artifact
            # the sim omits). The barrier uses the trainer intrinsic duration
            # (PROP_CLIENT_TASK_TRAIN_DURATION, _cycle_speed_s), not the agg-side
            # barrier_wait_s (which reads ~0 in real because trainers pipeline).
            # max() = the sync barrier.
            _barrier_span_s = max(_cycle_speed_s) if _cycle_speed_s else None
            _intrinsic_span_s = (
                _barrier_span_s + (_eval_s or 0.0)
                if _barrier_span_s is not None
                else None
            )
            try:
                ev, fields = build_agg_round(
                    round_num=self._round,
                    agg_goal=self._agg_goal,
                    agg_goal_count=self._agg_goal_cnt,
                    updates_in_queue=self._updates_in_queue,
                    staleness=_cycle_staleness,
                    stat_utility=_cycle_stat_utility,
                    trainer_speed_s=_cycle_speed_s,
                    contributing_trainers=_cycle_contributors,
                    agg_observed_s=_cycle_agg_observed_s,
                    extra={
                        # Virtual clock at commit (sim only). The parity engine
                        # gates its whole clock/throughput/convergence family on
                        # this; fwdllm never emitted it.
                        "vclock_now": getattr(self, "vclock_now", None),
                        "data_id": self.data_id,
                        "iteration_per_data_id": self.iteration_per_data_id,
                        "var": self.var,
                        "var_threshold": getattr(self, "var_threshold", None),
                        "var_good_enough": self.var_good_enough,
                        "force_commit_planned": _force_commit_planned,
                        # Opt-2: active stopping policy and why this cycle committed
                        # (natural gate / max-iter cap / plateau). None on non-commit
                        # cycles. Lets the reducer split commits by cause.
                        "stopping_policy": getattr(self, "_var_stopping_policy", None),
                        "commit_reason": getattr(self, "_force_commit_reason", None),
                        # Opt-3: active aggregation rate type + running count of
                        # anti-aligned updates the grad-aware gate down-weighted.
                        "agg_rate_type": (self.optimizer.agg_rate_conf.get("type")
                                          if getattr(self, "optimizer", None) and
                                          getattr(self.optimizer, "agg_rate_conf", None)
                                          else None),
                        "grad_aware_gated_total": getattr(
                            self, "_grad_aware_gated_total", 0),
                        "is_async": is_async,
                        # Variance-cadence rung inputs: cycle-relative identity for
                        # V1, pool sizes for V3/G2.
                        "cycle_data_id": _cycle_data_id,
                        "cycle_iteration": _cycle_iteration,
                        "cycle_model_version": _cycle_model_version,
                        "grad_pool_size": _grad_pool_size,
                        "cached_v_size": _cached_v_size,
                        # Per-contributor raw grad L2 norm this cycle (parity
                        # target: mode-invariant given identical input+seed).
                        "grad_norm": _cycle_grad_norm_list,
                        # R1/W1 residence rungs: per-contributor [dispatch_ts,
                        # commit_ts] intervals for this cycle.
                        "contributor_intervals": _contributor_intervals,
                        # Per-round wall decomposition (#6): barrier wait + drain
                        # tail (artifact) + fedavg + eval.
                        "barrier_wait_s": _barrier_wait_s,
                        "drain_tail_s": _drain_tail_s,
                        # drain_tail_s sub-phases, diagnostic only (not gated).
                        "drain_tail_canonicalize_s": _drain_tail_canonicalize_s,
                        "drain_tail_replay_s": _drain_tail_replay_s,
                        "drain_tail_residual_s": _drain_tail_residual_s,
                        "aggregate_fedavg_s": _aggregate_fedavg_s,
                        # Reorder-buffer health (sim only): last commit's
                        # vclock-vs-sct gap and post-pop buffer depth, plus the
                        # run-cumulative past-dating counters. A healthy sim
                        # commits at/near its sct (commit_gap_s ~ 0, buf_depth
                        # trending to 0 between arrival bursts) and never
                        # accumulates pastdated_commits. Feeds
                        # plots/aggregation/{buffer_health_over_rounds,
                        # commit_gap_cdf}.pdf and pastdated_commits_over_rounds.pdf.
                        "commit_gap_s": (
                            getattr(self, "_sim_last_commit_gap_s", None)
                            if self.simulated else None
                        ),
                        "buf_depth": (
                            getattr(self, "_sim_last_buf_depth", None)
                            if self.simulated else None
                        ),
                        # Standardized visibility-lag triplet, matching felix's
                        # own agg_round schema and the field name the shared
                        # analyzer already reads. Two distinct sources, gated
                        # on is_async since this dict is shared by both:
                        # - async (fluxtune): per-commit scalar stashed in
                        #   _sim_recv_min_grad (same number as commit_gap_s
                        #   above, both vclock.now - sct).
                        # - sync (fwdllm/fwdllm_plus): the barrier-anchored LIST
                        #   already computed in sync_collect_and_accumulate_grads
                        #   (_sync_barrier_lags_s). update_ready_ts/committed_ts
                        #   have no per-item meaning for a batched barrier commit,
                        #   so they stay None for sync.
                        # Real mode: async has no equivalent stash; sync's
                        # real-mode lag-anchor is an open question -- left None
                        # rather than guessed at.
                        "update_ready_ts": (
                            getattr(self, "_sim_last_update_ready_ts", None)
                            if self.simulated and is_async else None
                        ),
                        "update_committed_ts": (
                            getattr(self, "_sim_last_update_committed_ts", None)
                            if self.simulated and is_async else None
                        ),
                        "update_visibility_lag_s": (
                            (
                                getattr(self, "_sim_last_update_visibility_lag_s", None)
                                if is_async
                                else getattr(self, "_sync_barrier_lags_s", None)
                            )
                            if self.simulated else None
                        ),
                        "pastdated_commits": (
                            getattr(self, "_sim_pastdated_commits", 0)
                            if self.simulated else None
                        ),
                        "pastdated_gap_cum": (
                            getattr(self, "_sim_pastdated_gap_cum", 0.0)
                            if self.simulated else None
                        ),
                        "pastdated_gap_max": (
                            getattr(self, "_sim_pastdated_gap_max", 0.0)
                            if self.simulated else None
                        ),
                        # Carried-surplus commits, tracked separately from the
                        # primary pastdated_* counters above. Expected to step
                        # up once per data_id boundary for the life of any
                        # c >> agg_goal run -- non-alarming by design, NOT a
                        # correctness signal (unlike pastdated_commits, which
                        # should read ~0).
                        "carried_surplus_commits": (
                            getattr(self, "_sim_carried_surplus_commits", 0)
                            if self.simulated else None
                        ),
                        "carried_surplus_gap_max": (
                            getattr(self, "_sim_carried_surplus_gap_max", 0.0)
                            if self.simulated else None
                        ),
                        # Wall-clock span of the aggregate() variance-compute
                        # call, for measuring its overlap against other
                        # trainers' GPU passes (real-mode only -- sim's own
                        # concurrency is what the compute-truthful gate repairs).
                        "agg_compute_start_wall": _agg_start_wall,
                        "agg_compute_end_wall": _agg_start_wall + _aggregate_fedavg_s,
                        "eval_s": _eval_s,
                        # Sim-mode-only vclock delta for the two hand-timed terms
                        # above -- same nested-dict convention as the trainer's
                        # phase_vclock_s, so a real vs sim comparison of
                        # aggregate_fedavg_s/eval_s can check whether the vclock
                        # actually credited what the fold flags above claim to.
                        "phase_vclock_s": {
                            "aggregate_fedavg_s": _aggregate_fedavg_vclock_s,
                            "eval_s": _eval_vclock_s,
                        },
                        # #6 anchor: real's genuine per-cycle algorithmic time,
                        # the like-for-like counterpart to the sim's Δvclock; lets
                        # the clock-rate rungs exclude real's transport artifact.
                        "intrinsic_span_s": _intrinsic_span_s,
                        # Speedup metric (#13): wall in both modes; sim_rate =
                        # vclock/wall (sim only). sim_rate < 1 means slowdown.
                        "wall_elapsed_s": _wall_elapsed_s,
                        "sim_rate": _sim_rate,
                    },
                )
                telemetry.emit(ev, **fields)
            except Exception as e:  # telemetry must never break training
                logger.debug(f"agg_round telemetry emit failed: {e}")

        self._updates_in_queue -= self._agg_goal
        # In-place clear: `_agg_pending_commit_ref` holds a live reference.
        self._per_agg_trainer_list.clear()
        self._cycle_grad_norms = []
        self._commit_key_by_end = {}  # cohort-scoped
        self._pending_cohort_contribs = []  # already drained above; defensive

        logger.info(
            f"====== aggregation finished for round {self._round}, "
            f"self._agg_goal_cnt: {self._agg_goal_cnt}, self._updates_received: "
            f"{self._updates_received}"
        )

        self._agg_goal_cnt = 0

        self._n_aggs_completed += 1
        self._var_total_count += 1
        if self.var_good_enough:
            self._var_pass_count += 1

        if self._dynamic_kc_controller is not None:
            metrics = self._build_dynamic_kc_metrics(channel)
            new_k, new_c = self._dynamic_kc_controller.step(metrics)
            self._var_pass_count = 0
            self._var_total_count = 0
            if new_k != self._agg_goal:
                logger.info(f"[DynamicKC] _agg_goal updated: {self._agg_goal} → {new_k}")
                self._agg_goal = new_k
            channel.set_property("dynamic_c", new_c)
            logger.debug(
                f"[DynamicKC] step={self._n_aggs_completed}: "
                f"k={new_k}, c={new_c}, summary={self._dynamic_kc_controller.summary()}"
            )

        self.fwd_llm_stage = FwdLLMStage(
            self._round, self.data_id, self.iteration_per_data_id
        )

        if is_async:
            logger.debug(
                "Agg goal reached, so resetting trainer end states in the channel"
            )
        channel.cleanup_recvd_ends()

        # Sim rollback-safety: clear this cycle's committed marks + stranded
        # reorder-buffer entries + in-flight gate, and (async) release held slots,
        # so the next agg-goal cycle of a rolled-back data_id starts clean (§I.5).
        # Reached by both the variance-PASS and -FAIL branches. Inert in real.
        if self.simulated:
            self._release_sim_slots_at_agg_goal(channel, is_async)
        elif getattr(self, "_real_drain_ready_ingest", False) and self._real_sync_pending:
            # Real drain_ready twin of the sim clear: with agg_goal == cohort
            # this is normally empty, but a variance-FAIL rollback must never
            # strand a buffered-but-uncommitted grad into the next cycle.
            logger.info(
                f"Clearing {len(self._real_sync_pending)} uncommitted "
                f"drain_ready-buffered grads at agg-goal boundary."
            )
            self._real_sync_pending.clear()

        # Centralized cleanup
        # self._force_cuda_memory_cleanup()

    def _sim_sync_recv_incremental(self, channel, ends, num_min_req):
        """Sim analog of real's incremental `num_min_req` collect. fwdllm-scoped
        fork of `_sync_sim_recv_first_k` (shared `top_aggregator.py` stays
        untouched, other sync baselines unaffected) -- differs in exactly
        one way: `_sync_sim_recv_first_k`
        drains the WHOLE selected set every call into a LOCAL buffer and drops
        anything past `first_k`, so it can only ever be called once per cohort.
        This keeps a PERSISTENT sct-ordered buffer (`self._sim_sync_pending`)
        across calls -- populated once per dispatched end, only ever drained,
        never wholesale recomputed/discarded -- so a `num_min_req=1` call
        returns just the smallest-sct candidate without waiting for or
        discarding the rest of the cohort, matching real's per-commit refill
        cadence.

        Cleared at the agg-goal cycle boundary by
        `_release_sim_slots_at_agg_goal` (never strands a candidate across a
        rollback); an end already popped this cycle (`self._sim_sync_committed`)
        is never re-added or double-committed.
        """
        pending = getattr(self, "_sim_sync_pending", None)
        if pending is None:
            pending = self._sim_sync_pending = SimReorderBuffer()
        committed_this_cycle = getattr(self, "_sim_sync_committed", None)
        if committed_this_cycle is None:
            committed_this_cycle = self._sim_sync_committed = set()

        live = [e for e in ends if channel.has(e)]
        new_ends = [
            e for e in live if not pending.has(e) and e not in committed_this_cycle
        ]
        if new_ends:
            timeout = self._sim_recv_timeout_s(new_ends)
            for msg, md in channel.recv_fifo(new_ends, first_k=len(new_ends), timeout=timeout):
                if not msg:  # no more ready (bound expired or set drained)
                    break
                end = md[0]
                self._note_sim_known_delay(end, msg)
                sct = msg.get(MessageType.SIM_COMPLETION_TS)
                sct = float(sct) if sct is not None else self._vclock.now
                pending.add(end, sct, (msg, md))

        committed = []
        while len(committed) < num_min_req:
            popped = pending.pop_min()
            if popped is None:
                break
            end, sct, (msg, md) = popped
            if MessageType.WEIGHTS_BYTES in msg:
                msg[MessageType.WEIGHTS] = cloudpickle.loads(
                    msg.pop(MessageType.WEIGHTS_BYTES)
                )
            # E.1: send-gate — withhold if trainer is UN_AVL at completion.
            if self._sim_withhold_if_unavail(channel, end, sct, (msg, md)):
                continue
            self._advance_sim_clock(sct)
            committed_this_cycle.add(end)
            _sst = channel.get_end_property(end, PROP_SIM_SEND_TS)
            _srd = msg.get(MessageType.SIM_CLIENT_TASK_TRAIN_DURATION_S)
            if _srd is not None:
                channel.set_end_property(end, PROP_CLIENT_TASK_TRAIN_DURATION,
                                         timedelta(seconds=float(_srd)))
            elif _sst is not None:
                channel.set_end_property(end, PROP_CLIENT_TASK_TRAIN_DURATION,
                                         timedelta(seconds=max(0.0, sct - float(_sst))))
            logger.info(
                f"[SIM_SYNC_INCREMENTAL] committed {end[-4:]} sct={sct:.1f} "
                f"T_v={self._vclock.now:.1f} pending={len(pending)}"
            )
            committed.append((msg, md))

        # E.1 / E.2: withheld-delivery bonus -- unchanged mechanism, same
        # self._sim_buffer as _sync_sim_recv_first_k (separate lifecycle from
        # the pending-candidate buffer above).
        self._sim_reinject_ready_withheld()
        while True:
            wh = self._sim_buffer.pop_min()
            if wh is None:
                break
            wend, wdts, (wmsg, wmd) = wh
            if MessageType.WEIGHTS_BYTES in wmsg:
                wmsg[MessageType.WEIGHTS] = cloudpickle.loads(
                    wmsg.pop(MessageType.WEIGHTS_BYTES)
                )
            _wd = self._sim_take_withheld_delivering(wend)
            self._advance_sim_clock(wdts)
            if _wd is not None:
                self._emit_withheld_delivery(wend, wmsg, _wd[0], _wd[1])
            committed.append((wmsg, wmd))

        return committed

    def _real_sync_recv_incremental(self, channel, num_min_req):
        """Real twin of `_sim_sync_recv_incremental` via streamer-free
        `drain_ready` (§H): refill a persistent arrival-ordered buffer, pop the
        earliest `num_min_req` for the caller to commit.

        recv_fifo's fire-and-forget per-end tasks outlive their caller, so
        under num_min_req=1 a slow trainer's already-arrived grad strands for a
        full RECV_TIMEOUT_WAIT_S once per cohort (~40% of real collect wall).
        drain_ready pulls straight from each End rxq with no background task.
        Buffer is arrival-ordered (datetime + seq tiebreak) to preserve
        recv_fifo's commit order; stale/duplicate rejection stays in the
        caller's `_process_single_trainer_message`."""
        pending = self._real_sync_pending
        committed = []
        deadline = time.time() + RECV_TIMEOUT_WAIT_S

        def _buffer(drained):
            for msg, md in drained:
                if not msg:
                    continue
                self._real_recv_seq += 1
                pending.append((md[1], self._real_recv_seq, (msg, md)))

        while len(committed) < num_min_req:
            # Non-blocking sweep of everything already delivered to the End rxqs.
            _buffer(channel.drain_ready(channel.ends(), timeout=0))
            if not pending:
                remaining = deadline - time.time()
                if remaining <= 0:
                    logger.info(
                        f"No data within {RECV_TIMEOUT_WAIT_S}s (drain_ready); "
                        f"returning {len(committed)} of {num_min_req}"
                    )
                    break
                # Block-poll for the next arrival: drain_ready polls the End rxqs
                # at a few-ms tick and returns on the first message or `remaining`.
                _buffer(channel.drain_ready(channel.ends(), timeout=remaining))
                if not pending:
                    continue  # timed out empty -> next loop's remaining<=0 breaks
            pending.sort(key=lambda x: (x[0], x[1]))
            _ts, _seq, item = pending.pop(0)
            committed.append(item)
        return committed

    def _real_async_recv_min_grad(self, channel):
        """Real async twin of `next(channel.recv_fifo(RECV, 1))` via
        streamer-free `drain_ready` (§H): refill a persistent arrival-ordered
        buffer, pop the earliest `(msg, metadata)`.

        recv_fifo's fire-and-forget per-end tasks strand an already-arrived
        grad until the grace expires (~0.4s/cohort real never needs);
        drain_ready sweeps each End rxq directly so a grad commits at T+D like
        sim. Buffer persists across calls like recv_fifo's rxq (stale
        rejection stays in `_process_single_trainer_message`). Returns
        (None, ("", now)) on timeout, matching recv_fifo's contract."""
        pending = self._real_async_pending
        deadline = time.time() + RECV_TIMEOUT_WAIT_S

        def _buffer(drained):
            for msg, md in drained:
                if not msg:
                    continue
                self._real_recv_seq += 1
                pending.append((md[1], self._real_recv_seq, (msg, md)))

        # Non-blocking sweep first so a just-arrived earlier grad can win the sort.
        _buffer(channel.drain_ready(channel.ends(), timeout=0))
        while not pending:
            remaining = deadline - time.time()
            if remaining <= 0:
                return None, ("", datetime.now())
            _buffer(channel.drain_ready(channel.ends(), timeout=remaining))
        pending.sort(key=lambda x: (x[0], x[1]))
        _ts, _seq, (msg, md) = pending.pop(0)
        return msg, md

    @timer_decorator
    def sync_collect_and_accumulate_grads(self, tag, channel):
        """Aggregate trainer gradients synchronously, with timing and stage metadata."""
        self.fwd_llm_stage = FwdLLMStage(
            self._round, self.data_id, self.iteration_per_data_id, trainer_id=None
        )

        recv_ends = channel.ends()

        num_min_req = self._agg_goal  # change hardcoding, set it to aggGoal
        logger.info(f"Total ends: {len(recv_ends)}, required : {num_min_req}")
        num_min_req = min(num_min_req, len(recv_ends))
        # Sim now mirrors real's incremental collect exactly -- clamp to 1
        # whenever a dispatch pass selected >= agg_goal, in BOTH modes.
        # `_sim_sync_recv_incremental`'s persistent buffer (unlike the old
        # one-shot `_sync_sim_recv_first_k`) means a clamped call no longer
        # strands the rest of the cohort.
        if self.ends_not_selected_yet:
            logger.info(f"We are waiting to clear up queue")
            num_min_req = min(num_min_req, 1)

        # timeout=RECV_TIMEOUT_WAIT_S bounds the block on quiet in-flight
        # trainers (default is block forever -- see channel.recv_fifo's
        # docstring). Without this, if a selected trainer never responds,
        # this call never returns, the composer loop never gets to
        # re-check _check_early_stop_conditions() (max_runtime_s/
        # max_data_id_progress), and the run hangs past its configured
        # budget until manually killed. Same fix as _aggregate_grads_async.
        #
        # Sim barrier: instead of committing by physical arrival,
        # _sim_sync_recv_incremental commits the num_min_req trainers with the
        # smallest modeled sim_completion_ts (the k that would finish first in
        # real) out of its PERSISTENT pending pool, advancing the vclock to the
        # k-th smallest -- immune to arrival jitter, and safe to call
        # repeatedly with num_min_req=1 across a cohort like real does. Each
        # commit is fed through the same per-message path. Real unchanged.
        if self.simulated:
            committed = self._sim_sync_recv_incremental(
                channel, channel.ends(), num_min_req
            )
            _barrier_durs = getattr(self, "_sim_sync_barrier_durs", None)
            if _barrier_durs is None:
                _barrier_durs = self._sim_sync_barrier_durs = []
            for msg, metadata in committed:
                end, timestamp = metadata
                if not msg:
                    continue
                # dispatch-relative completion for the U6 barrier anchor = the
                # modeled round duration. Accumulated ACROSS calls (a cycle is
                # now assembled over many num_min_req=1 calls, not one
                # first_k=agg_goal call) so the barrier-anchored lag below
                # still covers the FULL cohort, not just this call's slice.
                _barrier_durs.append(
                    msg.get(MessageType.SIM_CLIENT_TASK_TRAIN_DURATION_S)
                )
                self._process_single_trainer_message(channel, msg, end, timestamp)
                if self._agg_goal_cnt >= self._agg_goal:
                    logger.info(
                        f"Reached agg_goal of {self._agg_goal} (sim barrier); "
                        f"proceeding to aggregate."
                    )
                    break
            # U6 visibility lag on a strict sync barrier: lag_i = max_completion -
            # completion_i over the committed cohort. Stashed for the U6 rung.
            self._sync_barrier_lags_s = self._barrier_anchored_lags(_barrier_durs)
            logger.info(
                f"[SYNC_SIM_BARRIER] round={self._round} committed={len(committed)} "
                f"T_v={self._vclock.now:.1f} barrier_lags_s={self._sync_barrier_lags_s}"
            )
        elif getattr(self, "_real_drain_ready_ingest", False):
            # Streamer-free ingest (§H): drain_ready avoids recv_fifo's
            # lingering-task stall. Same per-call num_min_req commit contract.
            for msg, metadata in self._real_sync_recv_incremental(channel, num_min_req):
                end, timestamp = metadata
                if not msg:
                    continue
                self._process_single_trainer_message(channel, msg, end, timestamp)
                if self._agg_goal_cnt >= self._agg_goal:
                    logger.info(
                        f"Reached agg_goal of {self._agg_goal} since agg_goal_count is {self._agg_goal_cnt}. Breaking from for loop, proceeding to aggregate."
                    )
                    break
        else:
            for msg, metadata in channel.recv_fifo(channel.ends(), num_min_req,
                                                   timeout=RECV_TIMEOUT_WAIT_S):
                end, timestamp = metadata
                if not msg:
                    logger.info(f"No data from {end}; skipping it")
                    continue

                self._process_single_trainer_message(channel, msg, end, timestamp)

                if self._agg_goal_cnt >= self._agg_goal:
                    logger.info(
                        f"Reached agg_goal of {self._agg_goal} since agg_goal_count is {self._agg_goal_cnt}. Breaking from for loop, proceeding to aggregate."
                    )
                    break

        # Second loop

    @timer_decorator
    def _aggregate_grads_sync(self, tag: str) -> None:
        """Aggregate trainer gradients synchronously."""
        logger.info("starting aggregate_grads_sync")
        self.log_memory("start _aggregate_grads_sync", self.device)
        self.print_trainable_params_stats(location="[start,_aggregate_grads_sync()]")

        channel = self.cm.get_by_tag(tag)
        if not channel:
            return

        logger.debug(f"Channel {channel} found for tag {tag}")
        if channel.ends(VAL_CH_STATE_RECV) is None:
            logger.info("no ends yet")
            return

        # receive local model parameters from trainers
        self.sync_collect_and_accumulate_grads(tag, channel)

        logger.debug(f"received {len(self.cache)} trainer updates in cache")

        if self._agg_goal_cnt < self._agg_goal:
            logger.info(f"did not reach agg goal, not aggregating")
            return

        self._process_aggregation_goal_met(tag, channel, is_async=False)
        self.log_memory("end _aggregate_grads_sync", self.device)

    @timer_decorator
    def _force_cuda_memory_cleanup(self):
        torch.cuda.empty_cache()
        gc.collect()

    @timer_decorator
    def invoke_gc(self):
        gc.collect()

    @timer_decorator
    def eval_model(self, epoch=0, global_step=0, device=None, model=None):
        """model: the model to evaluate (default self.model, so any other
        existing caller stays byte-identical). eval_model() is called from a
        background daemon thread against an isolated snapshot -- it must
        never touch self.model directly, or it races the main thread's
        concurrent aggregate()/next-cycle training on the SAME live model."""
        if not device:
            device = self.device
        if model is None:
            model = self.model

        logger.info(f"device inside eval_model() is set to: {device}")
        self.log_memory("start eval_model", self.device)

        results = {}

        eval_loss_total = torch.tensor(0.0, device=device)
        num_eval_steps = 0
        test_sample_len = len(self.test_global.dataset)

        # Move model to device before performing the eval. The
        # fc.make_functional_with_buffers(self.model) call this line used to
        # also do here was removed -- it deep-copies internally, so it never
        # mutated self.model, and its output was never read elsewhere in this
        # function; it only overwrote self.fmodel/self.params/self.buffers,
        # the SAME attributes the main training path assigns right before
        # self.aggregate() -- a race once this runs on a background thread.
        # Dead code + race source, not a needed side effect.
        model.to(device)
        model.eval()

        # One-time GPU data transfer for caching test data
        if not hasattr(self, "_cached_test_data") or self._cached_test_data is None:
            logger.info("One-time GPU data transfer for evaluation dataset")
            self._cached_test_data = [
                t.to(device) for t in self.test_global.dataset.tensors
            ]

        input_ids_all = self._cached_test_data[1]
        labels_all = self._cached_test_data[4]

        # Accumulate predictions on GPU
        preds_gpu = torch.empty((test_sample_len, self.num_labels), device=device)
        out_label_ids_gpu = torch.empty(
            test_sample_len, dtype=labels_all.dtype, device=device
        )

        batch_size = self.args.eval_batch_size
        loss_fct = CrossEntropyLoss()

        from torch.cuda.amp import autocast
        import contextlib

        autocast_cm = autocast() if self.args.fp16 else contextlib.nullcontext()
        if not self.args.fp16:
            logging.warning(f"Autocast is disabled: {self.args.fp16}")

        with torch.no_grad(), autocast_cm:
            for batch_start_idx in range(0, test_sample_len, batch_size):
                batch_end_idx = min(batch_start_idx + batch_size, test_sample_len)

                x = input_ids_all[batch_start_idx:batch_end_idx]
                labels = labels_all[batch_start_idx:batch_end_idx]

                output = model(x)
                if hasattr(output, "logits"):
                    logits = output.logits
                elif isinstance(output, (tuple, list)):
                    logits = output[0]
                else:
                    logits = output

                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                eval_loss_total += loss

                preds_gpu[batch_start_idx:batch_end_idx] = logits
                out_label_ids_gpu[batch_start_idx:batch_end_idx] = labels
                num_eval_steps += 1

        # Move to CPU only once at the end
        eval_loss = (eval_loss_total / num_eval_steps).item()
        preds = preds_gpu.cpu().numpy()
        out_label_ids = out_label_ids_gpu.cpu().numpy()

        logger.info(
            f"# of batches: {num_eval_steps} with (batch_size, seq_len): {input_ids_all.shape}. test_sample_len: {test_sample_len}, preds.shape: {preds.shape}, location of model: {next(model.parameters()).device}"
        )

        model_outputs = preds
        preds_argmax = np.argmax(preds, axis=1)
        result, wrong = self.compute_metrics(
            preds_argmax, out_label_ids, self.test_global.examples
        )

        # Uncomment the below to log the prediction stats
        probs = F.softmax(torch.tensor(preds), dim=1)  # Shape: [N, 4]
        log_margin_distribution(probs)
        compute_metrics_with_logging(probs, preds, out_label_ids, self.test_global)
        log_error_distribution(probs, out_label_ids)

        result["eval_loss"] = eval_loss
        results.update(result)

        # self.results.update(result)
        logging.info(
            f"results after eval are: {results}, len(wrong) is: {len(wrong)}, 'data_id_iterations': {self.iteration_per_data_id}"
        )

        # TODO: Check if model needs to be moved back to cpu? Do we need to keep
        # moving the model between CPU and GPU repeatedly?

        # No _force_cuda_memory_cleanup() here: it ran BEFORE this returns, so
        # the locals it claimed to free were still referenced, and empty_cache()
        # issues cudaFree (a device-wide sync) from the eval daemon thread,
        # stalling the main thread's aggregate(). Nothing to reclaim either --
        # memory is flat across the run. On OOM, tune PYTORCH_CUDA_ALLOC_CONF
        # rather than putting a device sync back on a hot path.
        self.log_memory("end eval_model", self.device)

        return result, model_outputs, wrong

    def compute_metrics(self, preds, labels, eval_examples=None):
        assert len(preds) == len(labels)
        self.log_memory("start compute_metrics", self.device)

        extra_metrics = {}
        extra_metrics["acc"] = sklearn.metrics.accuracy_score(labels, preds)
        mismatched = labels != preds

        if eval_examples:
            wrong = [i for (i, v) in zip(eval_examples, mismatched) if v.any()]
        else:
            wrong = ["NA"]

        mcc = matthews_corrcoef(labels, preds)

        tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()

        self.log_memory("end compute_metrics", self.device)

        return (
            {**{"mcc": mcc, "tp": tp, "tn": tn, "fp": fp, "fn": fn}, **extra_metrics},
            wrong,
        )

    def check_trainer_availability(self, end: str) -> bool:
        picked_trainer_is_available = True
        if self.track_trainer_avail["enabled"] == "False":
            return True
        elif self.track_trainer_avail["type"] == "ORACULAR":
            picked_trainer_is_available = self._trace_read_avail_check(end)

        return picked_trainer_is_available

    @timer_decorator
    def _prepare_distribution_payload(self, task_to_perform: str, force_weights: bool = False):
        """Build a WEIGHTS payload (always) or a VAR=bad payload (when var fails and force_weights=False)."""
        if not force_weights:
            # force_weights=False always means "build the tiny VAR=bad keep-
            # training message". Previously gated on `not var_good_enough`, which
            # returned WEIGHTS when var_good_enough=True, so Opt-1 could not
            # downgrade a commit-branch re-send. The message carries the current
            # model_version, so a trainer that cached it just keeps training.
            logger.info(  # self._var_scalar: cached float, no extra GPU sync
                f"[PreparePayload/VAR=bad] var={getattr(self, '_var_scalar', None)} vs "
                f"thr={self.var_threshold}; payload asks trainer to keep "
                f"training on current model_version={self._model_version}."
            )
            return {
                MessageType.VAR: "bad",
                MessageType.ROUND: self._round,
                MessageType.MODEL_VERSION: self._model_version,
                MessageType.TASK_TO_PERFORM: task_to_perform,
                MessageType.DATA_ID: self.data_id,
                MessageType.ITERATION_PER_DATA_ID: self.iteration_per_data_id,
            }

        _var_thr = getattr(self, "var_threshold", None)
        _var_val = getattr(self, "_var_scalar", None)  # cached float, no extra GPU sync
        _reason = (
            f"var_good_enough=True (var={_var_val} <= thr={_var_thr})"
            if self.var_good_enough
            else f"force_weights=True (var={_var_val}, thr={_var_thr})"
        )
        logger.info(
            f"[PreparePayload/WEIGHTS] {_reason}; building WEIGHTS payload "
            f"for model_version={self._model_version}, data_id={self.data_id}."
        )

        self.print_trainable_params_stats(location="[_prepare_distribution_payload]")
        trainable_params = self.get_trainable_param_state_dict()

        shared_weights = weights_to_device(
            trainable_params, DeviceType.CPU
        )  # Need to move to CPU for sending over MQTT

        if self._is_model_updated:
            shared_grad_pool = self.aggregate_grad_pool(self.grad_pool)
            shared_grad_pool_trainable = []
            if shared_grad_pool is None:
                shared_grad_pool_trainable = None
            else:
                idx = 0
                for param in self.model.parameters():
                    if param.requires_grad:
                        shared_grad_pool_trainable.append(shared_grad_pool[idx].clone())
                    idx += 1
            self.cached_shared_grad_pool_trainable = shared_grad_pool_trainable

        payload = {
            MessageType.WEIGHTS: shared_weights,
            MessageType.GRAD_POOL: self.cached_shared_grad_pool_trainable,
            MessageType.ROUND: self._round,
            MessageType.MODEL_VERSION: self._model_version,
            MessageType.TASK_TO_PERFORM: task_to_perform,
            MessageType.DATA_ID: self.data_id,
            MessageType.ITERATION_PER_DATA_ID: self.iteration_per_data_id,
        }

        return payload

    def _update_state_after_payload_prepared(self):
        """Update state after preparing payload.
        Reset grad pools if the model was updated.
        """
        if self._is_model_updated:
            self.grad_pool = []
            self.grad_for_var_check_list = []
            self.jvp_for_snr_check_list = []
            self._is_model_updated = False

    @staticmethod
    def _rearm_recv_eligibility(channel, ends):
        """Re-add `ends` to the selector's RECV-eligible set.

        Skipping the SEND-state call below (cache hit) also skips the only
        thing that normally repopulates `selected_ends`, which backs
        `channel.ends(VAL_CH_STATE_RECV)`. Each processed contribution
        removes its end from it via `cleanup_recvd_end(s)`; without this,
        it permanently empties out after one pass over the cached trainers.

        Two incompatible selector conventions share this helper (called from
        both the sync and async round-cache gates): sync selectors (random.py
        in its post-select state) keep `selected_ends` as a bare `set`; async
        selectors (fedbuff/async_random/async_oort) keep it as a
        dict[requester, set] -- reassigning the whole attribute to a bare set
        for those clobbers the dict and crashes the NEXT recv-state call with
        `TypeError: 'set' object is not subscriptable` (found via the
        felix_round smoke test -- async_oort.py:2146, first-ever exercise of
        a round-cache HIT on an async selector). Branch on the actual type
        instead of assuming one convention.
        """
        selector = getattr(channel, "_selector", None)
        if selector is None or not hasattr(selector, "selected_ends"):
            return
        if isinstance(selector.selected_ends, dict):
            requester = getattr(selector, "requester", None)
            if requester is None:
                return
            existing = selector.selected_ends.get(requester, set())
            selector.selected_ends[requester] = set(existing) | set(ends)
        else:
            selector.selected_ends = set(selector.selected_ends) | set(ends)

    @staticmethod
    def _exclude_pending_commit(channel, ends):
        """Drop ends still awaiting commit, mirroring `_eligible_candidates`'s
        pending exclusion (`async_base.py`) -- the guard the round-cache
        reuse branch above would otherwise skip, since it never calls
        `select()`. No-op for selectors without `_agg_pending_commit_ref`
        (e.g. sync's `RandomSelector`, whose busy-tracking lives in the
        aggregator instead)."""
        selector = getattr(channel, "_selector", None)
        pending = getattr(selector, "_agg_pending_commit_ref", None) if selector else None
        if not pending:
            return ends
        return [e for e in ends if e not in pending]

    def _round_cache_clock_now(self) -> float:
        """Clock for the round-cache stuck timeout: vclock in sim, wall in real
        (#1c, what `_abandon_clock_now` already fixed for the selector).
        Stamping and checking on wall measured a virtual-time run in wall
        seconds, so sim never reached ROUND_CACHE_STUCK_TIMEOUT_S: 0 stuck
        evictions vs real's 6, and sim refilled one round more (R-C).
        """
        vclock_now = getattr(self, "vclock_now", None)
        return vclock_now if vclock_now is not None else time.time()

    def _prune_departed_from_round_cache(self, channel):
        """Drop ends from `self._round_selected_ends` that have since
        departed (disconnected, or explicitly reported `UN_AVL`) -- or that
        have gone `ROUND_CACHE_STUCK_TIMEOUT_S` without a real accepted
        contribution despite still being formally connected (e.g. a trainer
        that never finished receiving its initial weights: still shows up
        as connected/AVL_TRAIN, so neither check below catches it, yet it
        never actually responds).

        The selector-level reclaim (`_cleanup_removed_ends`, invoked by
        `channel.remove`/`channel.update_state`) forgets a departed end at
        the selector's own bookkeeping level, but nothing else prunes it
        from this aggregator's own per-round cache. Left unpruned, the
        cache-size check below keeps reporting "full" forever with a
        member that can never respond, and the round stalls waiting on a
        contribution that can never arrive. Confirmed exactly this
        happening on a real n=100 run (see
        examples/MIGRATING_TO_LAUNCHER.md §9): the aggregator's working set
        stayed capped at ~30 of 100 trainers for 90 minutes, one cached
        member sat at `model_version=-1` (never initialized) the entire
        time, and progress fully stalled for the run's last 47 minutes --
        none of it caught by the departure checks below, since that
        trainer never disconnected or reported UN_AVL.
        """
        if not self._round_selected_ends:
            return
        now = self._round_cache_clock_now()
        still_present = []
        for end in self._round_selected_ends:
            if not channel.has(end):
                logger.info(
                    f"[ReselectGate] pruning departed (removed) end {end} "
                    f"from per-round cache for round={self._round}"
                )
                self._round_cache_activity_ts.pop(end, None)
                continue
            if channel.get_end_property(end, PROP_END_AVL_STATE) == TrainerAvailState.UN_AVL:
                logger.info(
                    f"[ReselectGate] pruning departed (UN_AVL) end {end} "
                    f"from per-round cache for round={self._round}"
                )
                self._round_cache_activity_ts.pop(end, None)
                continue
            last_active = self._round_cache_activity_ts.get(end, now)
            if now - last_active > ROUND_CACHE_STUCK_TIMEOUT_S:
                logger.info(
                    f"[ReselectGate] pruning stuck (no accepted contribution "
                    f"in {now - last_active:.0f}s > {ROUND_CACHE_STUCK_TIMEOUT_S}s) "
                    f"end {end} from per-round cache for round={self._round}"
                )
                self._round_cache_activity_ts.pop(end, None)
                continue
            still_present.append(end)
        if len(still_present) != len(self._round_selected_ends):
            self._round_selected_ends = still_present

    RESELECT_CADENCES = ("round", "data_bin", "iteration")

    def _resolve_reselect_cadence(self) -> str:
        """`reselect_cadence`, else the deprecated boolean alias
        (True->iteration, False->round), which every shipped baseline still
        sets -- so they stay byte-identical while `data_bin` becomes sayable.
        """
        hp = self.config.hyperparameters
        cadence = getattr(hp, "reselect_cadence", None)
        if cadence is not None:
            if cadence not in self.RESELECT_CADENCES:
                raise ValueError(
                    f"reselect_cadence={cadence!r} is not one of "
                    f"{self.RESELECT_CADENCES}"
                )
            return cadence
        return (
            "iteration"
            if bool(getattr(hp, "reselect_each_iteration", True))
            else "round"
        )

    def _cohort_cache_key(self):
        """Axis value the cohort is keyed on. `data_bin` uses the monotone
        `_model_version`, not `data_id`, which wraps each lap (§F-2)."""
        return (
            self._model_version
            if self._reselect_cadence == "data_bin"
            else self._round
        )

    def _invalidate_cohort_cache_if_stale(self) -> None:
        """Drop the pinned cohort when its cadence axis has advanced."""
        key = self._cohort_cache_key()
        if self._pinned_cohort_key != key:
            self._round_selected_ends = None
            self._pinned_cohort_key = key
            self._round_cache_activity_ts = {}

    def _round_cohort_target(self, channel) -> Optional[int]:
        """Cohort size = `c`, the promise to keep c trainers training;
        `agg_goal` is only the aggregation trigger. Sizing by agg_goal left
        c-agg_goal slots idle all round (fedbuff_round: c=30, agg_goal=10, 10
        of 100 committers) -- wrong in real and sim alike, so parity passed
        it. Honors `dynamic_c`; None -> legacy accumulate-forever path.
        """
        c = channel.properties.get("dynamic_c")
        if c is None:
            c = channel.get_c()
        try:
            c = int(c)
        except (TypeError, ValueError):
            return None
        return c if c > 0 else None

    def _trim_round_cohort(self, merged: list, target: Optional[int]) -> list:
        """Cap at exactly `target`, dropping the newest surplus. The reuse
        check runs before the merge, so one batch can overshoot and nothing
        trimmed it back -- size then fell out of arrival timing (felix_round:
        30 real vs 40 sim). Take-first is reproducible across modes.
        """
        if target is None or len(merged) <= target:
            return merged
        for end in merged[target:]:
            self._round_cache_activity_ts.pop(end, None)
        return merged[:target]

    def _select_ends_respecting_reselect_gate(self, channel, task_to_perform: str):
        """Return the SEND-state-selected ends, honoring `_reselect_cadence`.

        `iteration` (default): cached per `self.version_key` -- repeated calls
        within the SAME (model_version, iteration) reuse one channel.ends()
        result; a version_key change invalidates and re-fetches. `round` /
        `data_bin`: accumulate selections into a pinned cohort, re-invoking the
        selector each call until it reaches `c` (trainers join asynchronously,
        so one early call may only see a few); then reuse until the cadence
        axis advances (`_cohort_cache_key`).
        """
        self._invalidate_cohort_cache_if_stale()

        if self._reselect_each_iteration:
            if (self._reselect_true_cache_ends is not None
                    and self._reselect_true_cache_key == self.version_key):
                ends = self._reselect_true_cache_ends
                logger.info(
                    f"[ReselectGate] reselect_each_iteration=True; reusing "
                    f"cached selection ends={ends} for version_key="
                    f"{self.version_key}"
                )
                self._rearm_recv_eligibility(channel, ends)
                return ends

        if not self._reselect_each_iteration:
            self._prune_departed_from_round_cache(channel)

        target = self._round_cohort_target(channel)
        if not self._reselect_each_iteration and self._round_selected_ends is not None:
            # target None (no discoverable `c`) -> reuse whenever non-empty,
            # the pre-`c` fallback.
            if target is None or len(self._round_selected_ends) >= target:
                ends = list(self._round_selected_ends)
                logger.info(
                    f"[ReselectGate] cadence={self._reselect_cadence}; reusing "
                    f"pinned cohort ends={ends} at key={self._pinned_cohort_key}"
                )
                self._rearm_recv_eligibility(channel, ends)
                return ends

        # Thread version identity onto the SEND-side selection event so it's
        # placeable on the (data_id, iteration_per_data_id) axis -- was
        # missing on this call site (unlike the async distribute path below),
        # leaving no way to correlate a selection call back to its cycle
        # without cross-referencing wall timestamps.
        new_ends = channel.ends(
            VAL_CH_STATE_SEND, task_to_perform,
            agg_version_key=self.version_key, data_id=self.data_id,
        )
        if not self._reselect_each_iteration and new_ends:
            merged = list(self._round_selected_ends or [])
            for end in new_ends:
                if end not in merged:
                    merged.append(end)
                    # First time this end enters the cache -- starts its
                    # stuck-timeout clock (reset again on each real accepted
                    # contribution, see _process_single_trainer_message).
                    self._round_cache_activity_ts[end] = self._round_cache_clock_now()
            merged = self._trim_round_cohort(merged, target)
            self._round_selected_ends = merged
            logger.info(
                f"[ReselectGate] cadence={self._reselect_cadence}; accumulated "
                f"pinned cohort ends={merged} "
                f"({len(merged)}/{target}) at key={self._pinned_cohort_key}"
            )
            self._rearm_recv_eligibility(channel, merged)
            return merged
        if self._reselect_each_iteration and new_ends:
            # Cache for this version_key. Empty results are NOT cached --
            # a transient "no eligible trainers yet" must not stick.
            self._reselect_true_cache_key = self.version_key
            self._reselect_true_cache_ends = new_ends
            self._rearm_recv_eligibility(channel, new_ends)
        return new_ends

    def _select_ends_for_async_respecting_reselect_gate(
        self, channel, task_to_perform: str
    ):
        """Async-path counterpart of `_select_ends_respecting_reselect_gate`,
        honoring `_reselect_cadence` (round vs +IT baselines).

        `iteration` (default, preserves pre-existing behavior): re-invoke the
        selector's full utility/availability-based scoring on every dispatch
        call -- unlike the sync path, this one does NOT cache per version_key.
        `round` / `data_bin`: pin a cohort of `c` ends for the span of the
        cadence axis, reusing the same cache/prune-departed machinery as the
        sync path -- lets a round-level baseline swap out a departed client but
        not otherwise reselect on availability/utility churn.
        """
        if self._reselect_each_iteration:
            return channel.ends(
                state=VAL_CH_STATE_SEND,
                task_to_perform=task_to_perform,
                agg_version_key=self._curr_agg_version,
                trainer_version_keys=self._trainer_state_dict,
                data_id=self.data_id,
            )

        self._invalidate_cohort_cache_if_stale()
        self._prune_departed_from_round_cache(channel)

        target = self._round_cohort_target(channel)
        if self._round_selected_ends and (
            target is None or len(self._round_selected_ends) >= target
        ):
            ends = list(self._round_selected_ends)
            logger.info(
                f"[ReselectGate-async] cadence={self._reselect_cadence}; reusing "
                f"pinned cohort ends={ends} at key={self._pinned_cohort_key}"
            )
            # Re-arm RECV for the whole cohort, but only DISPATCH to ends not
            # still awaiting commit (§D-8) -- this branch skips select(), the
            # only other place `_agg_pending_commit_ref` gets checked.
            self._rearm_recv_eligibility(channel, ends)
            return self._exclude_pending_commit(channel, ends)

        new_ends = channel.ends(
            state=VAL_CH_STATE_SEND,
            task_to_perform=task_to_perform,
            agg_version_key=self._curr_agg_version,
            trainer_version_keys=self._trainer_state_dict,
            data_id=self.data_id,
        )
        if new_ends:
            merged = list(self._round_selected_ends or [])
            for end in new_ends:
                if end not in merged:
                    merged.append(end)
                    self._round_cache_activity_ts[end] = self._round_cache_clock_now()
            merged = self._trim_round_cohort(merged, target)
            self._round_selected_ends = merged
            logger.info(
                f"[ReselectGate-async] cadence={self._reselect_cadence}; "
                f"accumulated pinned cohort ends={merged} "
                f"({len(merged)}/{target}) at key={self._pinned_cohort_key}"
            )
            return merged
        return list(self._round_selected_ends or [])

    def _await_dispatchable_under_scarcity(self, task_to_perform: str) -> None:
        """Real-mode sync-barrier liveness under availability scarcity.

        When a trace keeps the eligible pool below `agg_goal`, the plain loop
        hot-re-dispatches every pass, burning the wall budget with no progress and
        ballooning the log. Parity-faithful fix: keep the cohort == `agg_goal` and
        wait for availability to recover (sleep-to-next-avail) instead of spinning
        -- the sim assembles the same full cohort by jumping its vclock past the
        unavailable window, so cohort size stays identical. Self-terminates via
        `_check_early_stop_conditions`. No-op when availability tracking is off or
        on the sim path (the vclock, not a wall sleep, models the wait).
        """
        if self.simulated or getattr(self, "trainer_event_dict", None) is None:
            return
        # Only a genuine post-join scarcity, not startup join-lag: wait only once
        # at least `agg_goal` trainers have joined.
        if len(getattr(self, "all_trainers", ())) < self._agg_goal:
            return
        poll_s = float(getattr(self.config.hyperparameters, "scarcity_poll_s", 2.0))
        warned = False
        while not self._work_done:
            unavail = set(self.get_curr_unavail_trainers())
            contributed = set(self._per_agg_trainer_list or [])
            # A trainer can still advance this cycle's barrier iff it is available
            # AND has not already contributed to it.
            dispatchable = [
                e for e in self.all_trainers
                if e not in unavail and e not in contributed
            ]
            if dispatchable or self._agg_goal_cnt >= self._agg_goal:
                return  # progress possible this pass (or barrier already met)
            if not warned:  # one line per stall, not one per spin
                logger.warning(
                    f"[SYNC_SCARCITY_WAIT] round={self._round} data_id={self.data_id} "
                    f"committed={self._agg_goal_cnt}/{self._agg_goal} "
                    f"unavail={len(unavail)}/{len(self.all_trainers)}; waiting for "
                    f"availability (poll={poll_s}s) instead of spin-dispatching."
                )
                warned = True
            time.sleep(poll_s)
            self._check_early_stop_conditions()  # self-terminate at max_runtime_s

    def _already_served_current_instruction(self, end) -> bool:
        """§H one-instruction-per-version_key: True iff `end` was already
        dispatched the CURRENT `version_key`, so re-sending this distribute pass
        would only queue a stale VAR=bad it will abort. Re-serves automatically
        on a version_key advance (the stored value stops matching), and a never-
        served end (not in the map) always returns False.

        getattr-guarded: test doubles built via __new__ skip __init__."""
        served = getattr(self, "_end_served_version_key", None)
        return served is not None and served.get(end) == self.version_key

    def _mark_instruction_served(self, end) -> None:
        """Record that the CURRENT version_key's instruction was dispatched to
        `end` (paired with `_already_served_current_instruction`).

        getattr-guarded: test doubles built via __new__ skip __init__."""
        served = getattr(self, "_end_served_version_key", None)
        if served is not None:
            served[end] = self.version_key

    def _should_send_full_weights(self, end, is_stale: bool) -> bool:
        """Decide WEIGHTS vs the tiny VAR=bad 'keep training' message for one end.

        Shared by both the sync and async distribute loops so the two paths stay
        identical.

        Opt-1 (charter §5c): within a data-bin the WEIGHTS+GRAD_POOL payload is
        byte-identical across iterations; an end already sent it this data-bin
        has it cached, so a re-send is pure redundancy -> downgrade to VAR=bad.
        The naive (round, is_stale)-only rule missed this because the redundancy
        is repeated distribute calls on the var_good_enough=True branch (which
        sends WEIGHTS unconditionally, so the stale check never ran). Fix: a
        send-time set (cleared on model_version advance) checked before both
        branches.
        """
        # Checked FIRST so it gates both the commit branch and the stale branch:
        # the dominant redundancy is repeated distribute calls within one data-bin,
        # each re-shipping the byte-identical model to trainers that already hold
        # it. Once an end got this model_version's payload this cycle -> VAR=bad.
        if end in self._weights_sent_this_cycle:
            return False
        if self.var_good_enough:
            # Commit / first distribution of this model_version: send it once.
            return True
        # A trainer the return-map still shows on an older version gets the
        # weights.
        return is_stale

    def _warn_if_redundant_weights_resend(self, end) -> None:
        """Regression tripwire: `_should_send_full_weights` already blocks this,
        so it should never fire -- catches a future call site that bypasses it
        and silently re-sends a redundant multi-MB payload."""
        if end in self._weights_sent_this_cycle:
            logger.warning(
                f"[SuppressRedundantWeights] INVARIANT VIOLATION: re-sending full "
                f"WEIGHTS to {end} for model_version={self._model_version}, "
                f"already sent this data-bin -- suppression should have caught "
                f"this; check for a new call site bypassing _should_send_full_weights."
            )

    def _should_force_commit_on_plateau(self) -> bool:
        """Opt-2 (charter §5c) variance-plateau rule -- pure decision (reads only
        instance attrs, unit-testable). Called from FedSGDAggregator.aggregate()
        once this cycle's var has been appended to `var_prev_iter_list`.

        Returns True iff the 'plateau' policy is active AND the per-bin variance
        curve has flattened (relative drop over the last N cycles < rel_delta)
        while var is still above the commit threshold -- the "more denoising buys
        nothing" signal. Policy off/absent, fewer than N+1 samples, or a
        non-positive baseline => False.
        """
        if getattr(self, "_var_stopping_policy", None) != "plateau":
            return False
        hist = self.var_prev_iter_list
        n = getattr(self, "_var_plateau_patience", 3)
        eps = getattr(self, "_var_plateau_rel_delta", 0.10)
        if len(hist) <= n or hist[-1 - n] <= 0:
            return False
        rel_drop = (hist[-1 - n] - hist[-1]) / hist[-1 - n]
        return rel_drop < eps and hist[-1] > self.var_threshold

    @staticmethod
    def _grad_aware_rate(base_rate, cos, var_i, var_ref, *, align_gate=True,
                         inverse_var=False, align_floor=0.0, var_eps=1e-8):
        """Opt-3 gradient-aware aggregation weight -- pure scalar math. Replaces
        the scalar staleness×utility rate with a direction/reliability-aware
        weight, bounded so it only ever down-weights (result <= base_rate), so it
        never inflates the effective server LR.
          * align_gate (primary): down-weight an update whose direction opposes the
            running aggregate. factor = 1 for cos >= align_floor, linearly -> 0 at
            cos = -1. Fixes averaging anti-aligned JVP estimates (mid-run instability).
          * inverse_var (optional): down-weight an update noisier than the reference
            var_ref: factor = min(1, var_ref / (var_i + eps)).
        cos / var_i == None => that factor is 1 (e.g. first update of a cycle).
        """
        rate = float(base_rate)
        if align_gate and cos is not None and cos < align_floor:
            denom = align_floor + 1.0
            rate *= (max(0.0, (cos + 1.0) / denom) if denom > 0 else 0.0)
        if inverse_var and var_i is not None:
            rate *= min(1.0, var_ref / (var_i + var_eps))
        return rate

    @staticmethod
    def _cosine_flat(grad_named, running_grad, named_params):
        """Cosine between a trainer's gradient (dict name→tensor) and the running
        aggregate `self.grad` (list indexed by `named_params` order), flattened over
        all trainable params. Returns None if either side has ~zero norm (e.g. the
        running aggregate before any update has landed this cycle)."""
        dot = nt = ng = 0.0
        for i, (name, _p) in enumerate(named_params):
            if name in grad_named:
                t = grad_named[name]
                g = running_grad[i].to(t.device)
                dot += float((t * g).sum())
                nt += float((t * t).sum())
                ng += float((g * g).sum())
        if nt <= 0.0 or ng <= 0.0:
            return None
        return dot / (math.sqrt(nt) * math.sqrt(ng))

    @staticmethod
    def _flat_grad_norm(grad_named, named_params):
        """L2 norm of a trainer's gradient (dict name→tensor), flattened over
        all trainable params. Mode-invariance target: given identical input +
        perturbation seed, this should match real vs sim to float-noise.

        Per-tensor squared sums stay on-device, reduced in ONE host sync (was a
        blocking `float(...)` sync per parameter, which stalled behind sim's JVP
        queue and fattened the drain-tail p90). float64 accumulation matches
        the prior python-float sum."""
        parts = [
            (grad_named[name] * grad_named[name]).sum()
            for name, _p in named_params
            if name in grad_named
        ]
        if not parts:
            return 0.0
        sq = float(torch.stack(parts).double().sum())
        return math.sqrt(sq) if sq > 0.0 else 0.0

    @timer_decorator
    def _distribute_weights_sync(
        self, tag: str, task_to_perform: str = "train"
    ) -> None:
        """Distribute a global model in synchronous FL fashion - for FwdLLM.
        This method actually sends either gradients or calc_more_var to
        trainers, not the actual model weights.

        This method is overridden from one in synchronous top aggregator
        """

        self.ends_not_selected_yet = False
        logger.info(f"Device for agg: {next(self.model.parameters()).device}")
        channel = self.cm.get_by_tag(tag)
        if not channel:
            logger.debug(f"channel not found for tag {tag}")
            return

        channel.await_join()
        # Sleep-to-next-avail under real-mode scarcity instead of hot-spinning
        # (Stage C); no-op on the sim path / when availability tracking is off.
        self._await_dispatchable_under_scarcity(task_to_perform)
        global_model_params = self.get_global_model_params()
        # Debug-gated (was INFO): full-model GPU->CPU sha256 per dispatch.
        if logger.isEnabledFor(logging.DEBUG):
            format_hash = lambda d: {k: _calculate_hash(v)[:8] for k, v in d.items()}
            logger.debug(
                f"Model distributed to clients (Hashed): {format_hash(global_model_params)}"
            )
        self.weights = global_model_params

        # Real-transport pad to let just-distributed messages settle before the
        # selection read (real-mode MQTT artifact, #8). No sim analog: the sim
        # orders by sct, not physical arrival, so skip it there. Real path gated
        # on real_distribute_settle_s (§H A/B: 0 = drop it).
        _settle = getattr(self, "_real_distribute_settle_s", 0.1)
        if not self.simulated and _settle > 0.0:
            time.sleep(_settle)

        if self.trainer_event_dict is not None:
            curr_unavail_trainer_list = self.get_curr_unavail_trainers()
            channel.set_curr_unavailable_trainers(
                trainer_unavail_list=curr_unavail_trainer_list
            )
            logger.debug(
                f"Passed curr_unavail_trainer_list: "
                f"{curr_unavail_trainer_list} to channel"
            )
        else:
            channel.set_curr_unavailable_trainers(trainer_unavail_list=[])

        self._curr_agg_version = self.version_key
        logger.debug(
            f"Aggregator version_key (model_version, iteration): {self._curr_agg_version}"
        )
        
        ends = self._select_ends_respecting_reselect_gate(channel, task_to_perform)
        logger.info(f"ends: {ends}")
        if ends is None or len(ends) >= self._agg_goal:
            self.ends_not_selected_yet = True
        else:
            self.ends_not_selected_yet = False

        if not ends:
            logger.debug(
                f"No trainers found for tag {tag}, will "
                f"move to get() for fetch weights from trainers"
            )
            return

        payload_with_weights = self._prepare_distribution_payload(task_to_perform, force_weights=True)
        # Opt-1: always need the VAR=bad variant too -- a commit-branch re-send
        # to an already-served end downgrades to VAR=bad regardless of var_good_enough.
        payload_without_weights = self._prepare_distribution_payload(task_to_perform, force_weights=False)

        self._update_state_after_payload_prepared()

        # Sim-clock dispatch stamp: in the sync barrier every selected end
        # dispatches at the same virtual instant, so one _round_now stamp is
        # injected into each payload variant (the trainer bases its modeled sct on
        # SIM_SEND_TS) and recorded as a per-end property. Inert in real mode
        # (SIM_SEND_TS absent -> arrival order).
        _round_now = getattr(self, "vclock_now", None)
        if self.simulated:
            for _p in (payload_with_weights, payload_without_weights):
                if _p is not None:
                    _p[MessageType.SIM_SEND_TS] = _round_now

        _n_weights_sent = 0
        _n_var_bad_sent = 0
        for end in ends:
            # §H: one instruction per version_key -- skip an end already served
            # this version_key so the distribute-per-collect loop stops flooding
            # VAR=bad the trainer would only abort. Re-serves on a version_key bump.
            if self._already_served_current_instruction(end):
                continue
            trainer_version = self._trainer_last_model_version.get(end, -1)
            is_stale = (trainer_version != self._model_version)

            # Opt-1: shared decision (identical in the async path) -- suppresses
            # a byte-identical intra-databin re-send to VAR=bad.
            send_weights = self._should_send_full_weights(end, is_stale)
            if send_weights:
                payload = payload_with_weights
                _n_weights_sent += 1
                self._warn_if_redundant_weights_resend(end)
                self._weights_sent_this_cycle.add(end)
                if not self.var_good_enough and is_stale:
                    logger.info(f"Trainer {end} hasn't received weights for model_version {self._model_version} (has {trainer_version}). Sending WEIGHTS payload instead of VAR=bad.")
            else:
                payload = payload_without_weights
                _n_var_bad_sent += 1
                if is_stale:
                    self._redundant_weights_suppressed_total += 1

            logger.debug(
                f"Setting channel property {PROP_ROUND_START_TIME} for "
                f"end {end}. For round {self._round} at time: {datetime.now()}"
            )
            channel.set_end_property(
                end, PROP_ROUND_START_TIME, (self._round, datetime.now())
            )
            if self.simulated:
                channel.set_end_property(end, PROP_SIM_SEND_TS, _round_now)

            # Label/size by the actual payload (send_weights), not var_good_enough:
            # keying off the latter mislabeled a var_good_enough=False is_stale
            # WEIGHTS send as var_bad, undercounting sync weight bytes.
            # _send_bytes feeds build_comm telemetry below -- always ONE
            # pickle.dumps(payload) call so the byte count stays level-independent.
            # The per-key breakdown was N extra redundant pickles unconditionally
            # -- now DEBUG-gated.
            if send_weights:
                logger.info(
                    f"sending weights to {end} with model_version: {self._model_version}, data_id: {self.data_id} for task: {task_to_perform}"
                )

                _send_bytes = len(pickle.dumps(payload))
                _payload_kind = "weights"

                if logger.isEnabledFor(logging.DEBUG):
                    sizes_mb = {
                        key.name if hasattr(key, "name") else str(key): len(
                            pickle.dumps(value)
                        )
                        / (1024 * 1024)
                        for key, value in payload.items()
                    }
                    logger.debug(
                        f"[DEBUG] Payload size breakdown for {end}: "
                        + ", ".join([f"{k}: {v:.2f} MB" for k, v in sizes_mb.items()])
                        + f", Total: {_send_bytes / (1024 * 1024):.2f} MB"
                    )
            else:
                logger.info(
                    f"sending var = bad to {end} with model_version: {self._model_version}, round: {self._round}, data_id: {self.data_id} for task: {task_to_perform}"
                )

                _send_bytes = len(pickle.dumps(payload))
                _payload_kind = "var_bad"
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f"[DEBUG] Payload size for {end}: {_send_bytes / (1024 * 1024):.2f} MB"
                    )

            # Network telemetry: one dispatch message onto the wire. Size reuses
            # the value already computed for the debug log above.
            try:
                ev, f = build_comm(
                    direction="agg_to_trainer", size_bytes=_send_bytes, peer_id=str(end),
                    round_num=int(self._round), data_id=self.data_id,
                    iteration=self.iteration_per_data_id, payload_kind=_payload_kind,
                    n_tensors=len(payload), model_version=self._model_version,
                )
                telemetry.emit(ev, **f)
            except Exception as e:
                logger.debug(f"comm telemetry emit failed (agg send): {e}")

            # Diagnostic: this end now carries an outstanding dispatch at
            # the CURRENT version_key until it returns (cleared in
            # _process_single_trainer_message on return).
            if getattr(self, "_trainer_inflight_dispatch_version", None) is not None:
                self._trainer_inflight_dispatch_version[end] = self.version_key

            channel.send(end, payload)
            self._mark_instruction_served(end)
            logger.info(f"Sent weights to {end}")
            # self.invoke_gc()

        # Sync path used to track this counter without logging it -- async
        # already had this line. Restores visibility.
        logger.info(
            f"[Distribute] Done. Sent {_n_weights_sent} WEIGHTS + "
            f"{_n_var_bad_sent} VAR=bad payloads to {len(ends)} trainers "
            f"(model_version={self._model_version}, data_id={self.data_id}, "
            f"iter={self.iteration_per_data_id}, "
            f"redundant_weights_suppressed_total={self._redundant_weights_suppressed_total})."
        )

        # Cohort-dispatch wall -> barrier_wait_s anchor. Anchor on the FIRST
        # pass that actually dispatched this version_key (later per-collect passes
        # skip the whole cohort under the version_key dedup), so the anchor stays
        # the true cohort-dispatch instant, not a no-op re-visit near the collect.
        if _n_weights_sent + _n_var_bad_sent > 0:
            self._round_dispatch_wall_ts = time.time()

    @timer_decorator
    def _distribute_weights_async(
        self, tag: str, task_to_perform: str = "train"
    ) -> None:
        """Distribute a global model in asynchronous FL fashion - for FwdLLM."""
        channel = self.cm.get_by_tag(tag)
        if not channel:
            logger.debug(f"channel not found for tag {tag}")
            return

        # this call waits for at least one peer to join this channel
        channel.await_join()
        global_model_params = self.get_global_model_params()
        self.weights = global_model_params
        # Real-transport pad to let just-distributed messages settle before the
        # selection read (real-mode MQTT artifact, #8). No sim analog: the sim
        # orders by sct, not physical arrival, so skip it there. Real path gated
        # on real_distribute_settle_s (§H A/B: 0 = drop it).
        _settle = getattr(self, "_real_distribute_settle_s", 0.1)
        if not self.simulated and _settle > 0.0:
            time.sleep(_settle)
        if self.trainer_event_dict is not None:
            curr_unavail_trainer_list = self.get_curr_unavail_trainers()
            channel.set_curr_unavailable_trainers(
                trainer_unavail_list=curr_unavail_trainer_list
            )
            logger.info(
                f"Passed curr_unavail_trainer_list: "
                f"{curr_unavail_trainer_list} to channel"
            )
        else:
            # Handling the case for oort's selector since it expects 3
            # arguments
            channel.set_curr_unavailable_trainers(trainer_unavail_list=[])

        logger.debug(
            f"Sending weights to trainers with task_to_perform = {task_to_perform}"
        )

        self._curr_agg_version = self.version_key
        logger.debug(
            f"Aggregator version_key (model_version, iteration): {self._curr_agg_version}"
        )
        # AsyncOortSelector reads channel_props["vclock_now"] to stamp its
        # in-flight abandon-timeout clock and to tag `selection_train`
        # telemetry -- fwdllm never set it here, unlike
        # asyncfl/top_aggregator.py's own dispatch, so it read None always in
        # sim: the abandon-timeout silently fell back to wall-clock
        # `time.time()` instead of virtual time, and every selection_train
        # event's `vclock_now` field was None. Only matters for a selector
        # that consumes it (AsyncOortSelector); harmless no-op for
        # RandomSelector (fwdllm/fwdllm_plus, sync path).
        channel.properties["vclock_now"] = self.vclock_now
        ends = self._select_ends_for_async_respecting_reselect_gate(
            channel, task_to_perform
        )
        logger.info(f"ends: {ends}")

        if not ends:
            logger.debug(
                f"No trainers found for tag {tag}, will "
                f"move to get() for fetch weights from trainers"
            )
            return

        _var_thr = getattr(self, "var_threshold", None)
        _var_val = getattr(self, "_var_scalar", None)  # cached float, no extra GPU sync
        logger.info(
            f"[Distribute] Starting distribute for model_version={self._model_version}, "
            f"data_id={self.data_id}, iter={self.iteration_per_data_id}, "
            f"task={task_to_perform}, |ends|={len(ends)}, "
            f"var={_var_val}, var_thr={_var_thr}, var_good_enough={self.var_good_enough}. "
            f"Stale trainers will get WEIGHTS; current trainers will get "
            f"{'WEIGHTS' if self.var_good_enough else 'VAR=bad'}."
        )

        payload_weights = self._prepare_distribution_payload(task_to_perform, force_weights=True)
        # Opt-1: always need VAR=bad too -- commit-branch re-sends to
        # already-served ends downgrade to VAR=bad regardless of var_good_enough.
        payload_var_bad = self._prepare_distribution_payload(task_to_perform, force_weights=False)

        self._update_state_after_payload_prepared()

        # Network telemetry: serialized size of each payload variant, computed once
        # per distribute (the async path has no per-end debug-size log), reused per
        # end below.
        try:
            _bytes_weights = len(pickle.dumps(payload_weights)) if payload_weights is not None else 0
            _bytes_var_bad = len(pickle.dumps(payload_var_bad)) if payload_var_bad is not None else 0
        except Exception as e:
            _bytes_weights = _bytes_var_bad = 0
            logger.debug(f"comm telemetry size calc failed (async): {e}")

        # Sim-clock dispatch stamp: the async path arms the in-flight gate
        # (_sim_inflight_expected[end] = send vclock + a lower-bound budget) so the
        # reorder-buffer drain can't lap a trainer still expected to complete.
        # Inert in real mode.
        #   #13 (staggered): when on, each end's SEND vclock is popped from the
        # freed-slot FIFO instead of the shared round frontier, spreading a cohort's
        # expected completions across the timeline. Off/real => one shared
        # _round_now injected into the two payload variants.
        _round_now = getattr(self, "vclock_now", None)
        _staggered = self.simulated and getattr(self, "_sim_staggered_redispatch", False)
        # #6 serial-dispatch queue: offset each trainer's sim_send_ts by the
        # MEASURED cumulative send wall of the prior sends in this burst, so the
        # k-th trainer starts after the agg finished sending to the first k-1 (a
        # real serial-server delay). Forces the per-trainer payload rebuild below.
        _dq = self.simulated and getattr(
            self.config.hyperparameters, "sim_model_dispatch_queue", False)
        _cum_dispatch_s = 0.0
        if self.simulated and not _staggered and not _dq:
            for _p in (payload_weights, payload_var_bad):
                if _p is not None:
                    _p[MessageType.SIM_SEND_TS] = _round_now

        _n_weights_sent = 0
        _n_var_bad_sent = 0
        for end in ends:
            # §F-25: skip an end already served this version_key -- the round-cache
            # re-invokes the whole cohort every tick; mirrors the sync path's guard
            # (~3654), which async lacked (was flooding VAR=bad, r1_inflight_overlap).
            if self._already_served_current_instruction(end):
                continue
            trainer_version = self._trainer_last_model_version.get(end, -1)
            is_stale = (trainer_version != self._model_version)
            # Opt-1: shared decision with the sync path (_should_send_full_weights).
            send_weights = self._should_send_full_weights(end, is_stale)
            if send_weights:
                payload = payload_weights
                _pk = "weights"          # kind set here (survives staggered rebuild below)
                _n_weights_sent += 1
                self._warn_if_redundant_weights_resend(end)
                self._weights_sent_this_cycle.add(end)
                if not self.var_good_enough and is_stale:
                    logger.debug(
                        f"[Distribute] Trainer {end} stale "
                        f"(has v{trainer_version}, need v{self._model_version}); "
                        f"sending WEIGHTS even though var_good_enough=False."
                    )
            else:
                payload = payload_var_bad
                _pk = "var_bad"
                _n_var_bad_sent += 1
                # Opt-1: this end read stale but already got this version's payload
                # this cycle -> a redundant full re-send we just avoided.
                if is_stale:
                    self._redundant_weights_suppressed_total += 1
                logger.debug(
                    f"[Distribute] Trainer {end} has current v{trainer_version}; "
                    f"sending VAR=bad (keep training)."
                )

            logger.debug(
                f"Setting channel property {PROP_ROUND_START_TIME} for "
                f"end {end}. For round {self._round} at time: {datetime.now()}"
            )
            channel.set_end_property(
                end, PROP_ROUND_START_TIME, (self._round, datetime.now())
            )
            if self.simulated:
                # R1 tripwire: dispatching a trainer already outstanding (in flight
                # or grad carried-but-uncommitted) IS the residence violation --
                # surface it at dispatch instead of reconstructing it offline.
                if end in self._sim_inflight_expected or self._sim_buffer.has(end):
                    logger.warning(
                        f"[SIM_R1_DISPATCH] end={end} re-dispatched while still "
                        f"outstanding (vclock={_round_now}) -- R1 residence violation"
                    )
                # #13: SEND vclock = the freed-slot stamp (staggered) else the
                # shared round frontier. _pop_free_slot_ts is FIFO + clamped <= now;
                # an empty queue falls back to _round_now.
                _sst = self._pop_free_slot_ts(_round_now) if _staggered else _round_now
                # Serial-dispatch queue: add the cumulative prior-send wall so
                # later trainers in the burst start later (measured, dynamic).
                if _dq and _sst is not None:
                    _sst = _sst + _cum_dispatch_s
                channel.set_end_property(end, PROP_SIM_SEND_TS, _sst)
                # Expected completion = SEND vclock + this end's own
                # MODELED_DELAY_S. No fallback: unseen -> no gate entry.
                _delay = self._sim_known_delay_s.get(end)
                if _delay is not None:
                    self._sim_inflight_expected[end] = _sst + _delay
                # Staggered/dispatch-queue: this end's payload must carry its OWN
                # SIM_SEND_TS, so rebuild a shallow copy (weights shared by ref;
                # small vs GPU cost).
                if (_staggered or _dq) and isinstance(payload, dict):
                    payload = dict(payload)
                    payload[MessageType.SIM_SEND_TS] = _sst
                # Once dispatched a trainer is in flight in virtual time -> add to
                # the pending-commit set so the selector's eligibility filter
                # excludes it until its grad COMMITS (discarded in _sim_recv_min_grad).
                self._sim_pending_commit.add(end)
                # NOTE: the async_oort re-pick triplet is stamped on grad RETURN,
                # not here at dispatch -- stamping the whole cohort at the current
                # _curr_agg_version froze the eligible pool before any commit could
                # advance the version (re-dispatch deadlock). An in-flight trainer
                # is already guarded by its compute slot.
            # One dispatch message onto the wire (async path).
            try:
                _sz = _bytes_weights if _pk == "weights" else _bytes_var_bad
                ev, f = build_comm(
                    direction="agg_to_trainer", size_bytes=_sz, peer_id=str(end),
                    round_num=int(self._round), data_id=self.data_id,
                    iteration=self.iteration_per_data_id, payload_kind=_pk,
                    n_tensors=len(payload) if isinstance(payload, dict) else None,
                    model_version=self._model_version,
                )
                telemetry.emit(ev, **f)
            except Exception as e:
                logger.debug(f"comm telemetry emit failed (agg async send): {e}")
            # Diagnostic: this end now carries an outstanding dispatch at the
            # CURRENT version_key until it returns (cleared on return in
            # _process_single_trainer_message).
            if getattr(self, "_trainer_inflight_dispatch_version", None) is not None:
                self._trainer_inflight_dispatch_version[end] = self.version_key
            _send_t0 = time.time()
            channel.send(end, payload)
            self._mark_instruction_served(end)
            _send_wall = time.time()
            # Serial-dispatch queue: accumulate this send's MEASURED wall (pickle +
            # publish) so the next trainer's sim_send_ts reflects waiting behind it.
            if _dq:
                _cum_dispatch_s += _send_wall - _send_t0
            # #15 compute-truthful gate: stamp the wall time this end was dispatched
            # so the drain can tell a live straggler from an idle-in-recv phantom.
            # Sim-only; inert unless the gate flag is on.
            if self.simulated:
                self._sim_dispatch_wall[end] = _send_wall
        _dq_warn = getattr(self.config.hyperparameters, "sim_overhead_warn_s", None)
        if _dq and _dq_warn and _cum_dispatch_s > float(_dq_warn):
            logger.warning(
                f"[SIM_OVERHEAD] dispatch_queue={_cum_dispatch_s:.3f}s across "
                f"{len(ends)} sends (vclock={_round_now}) -- excess serial-dispatch wall"
            )
        logger.info(
            f"[Distribute] Done. Sent {_n_weights_sent} WEIGHTS + "
            f"{_n_var_bad_sent} VAR=bad payloads to {len(ends)} trainers "
            f"(model_version={self._model_version}, data_id={self.data_id}, "
            f"iter={self.iteration_per_data_id}, "
            f"redundant_weights_suppressed_total={self._redundant_weights_suppressed_total})."
        )

        ends_in_recv_state = channel.ends(VAL_CH_STATE_RECV)
        logger.info(f"ends_in_recv_state: {ends_in_recv_state}")
        if ends_in_recv_state is None:
            self.ends_not_selected_yet = True
            logger.info(f"ends_in_recv is None")
        elif len(ends_in_recv_state) < channel.get_c():  
            self.ends_not_selected_yet = True
            logger.info(f"Selected only {len(ends)} in this round, total in flight {len(ends_in_recv_state)}, need {channel.get_c() - len(ends_in_recv_state)} to meet agg-goal.")
        else:
            self.ends_not_selected_yet = False
            logger.info("Distributed to c ends and can wait for k updates")

    def _check_early_stop_conditions(self) -> None:
        """Optional caps for short/smoke runs, checked independently of
        whether an aggregation goal is ever met: `max_data_id_progress`
        (stop once `self.data_id` reaches it) and `max_runtime_s` (stop once
        this much wall time has elapsed since the aggregator started).
        Whichever fires first wins. Both are unset (None) by default, so
        production runs are unaffected. Called from `_distribute_weights`,
        which runs on every composer tick on both the sync and async paths
        -- unlike the rounds-based stop in `_process_aggregation_goal_met`,
        this also fires when aggregation never completes.
        """
        if self._work_done:
            return

        max_data_id = getattr(
            self.config.hyperparameters, "max_data_id_progress", None
        )
        if max_data_id is not None and self.data_id >= max_data_id:
            logger.info(
                f"max_data_id_progress={max_data_id} reached "
                f"(data_id={self.data_id}); stopping run."
            )
            self._work_done = True
            return

        max_runtime_s = getattr(self.config.hyperparameters, "max_runtime_s", None)
        if max_runtime_s is not None:
            # Mode-dependent clock (mirrors base syncfl `max_experiment_runtime_s`):
            # real -> wall seconds; sim -> virtual seconds (`vclock.now`). One budget
            # then means the same amount of real work AND modeled work -- what the
            # matched-budget convergence parity needs. Until #6 (vclock models real
            # wall) lands the sim vclock under-counts, so the wall failsafe below
            # fires first -- itself the #6 signal.
            if self.simulated and hasattr(self, "_vclock"):
                elapsed = float(self._vclock.now)
                clock_label = "sim/vclock"
            else:
                elapsed = time.time() - self.agg_start_time_ts
                clock_label = "real/wall"
            if elapsed >= float(max_runtime_s):
                logger.info(
                    f"max_runtime_s={max_runtime_s}s reached "
                    f"({clock_label}_elapsed={elapsed:.0f}s); stopping run."
                )
                self._work_done = True
                return
            # Sim wall failsafe: the sim does real GPU + server eval, so its wall
            # legitimately exceeds the vclock budget -- a 1x ceiling truncated every
            # sim before its paired vclock stop. Decoupled to max_runtime_s ×
            # SIM_WALL_CEILING_FACTOR (a runaway OUTER safety), overridable by
            # `sim_wall_ceiling_s`. Primary stops stay max_data_id_progress /
            # vclock >= max_runtime_s.
            if self.simulated:
                _wall = time.time() - self.agg_start_time_ts
                _ceil = getattr(self.config.hyperparameters, "sim_wall_ceiling_s", None)
                _ceil = (float(_ceil) if _ceil is not None
                         else float(max_runtime_s) * self.SIM_WALL_CEILING_FACTOR)
                if _wall > _ceil:
                    logger.warning(
                        f"[SIM_WALL_CEILING] wall={_wall:.0f}s > ceiling={_ceil:.0f}s "
                        f"(={self.SIM_WALL_CEILING_FACTOR:g}× budget {max_runtime_s}s; "
                        f"vclock={self._vclock.now:.0f}s) -- runaway outer safety "
                        f"tripped (a healthy sim stops on max_data_id/vclock first); "
                        f"stopping run."
                    )
                    self._work_done = True

    def _async_inner_loop_done(self) -> bool:
        """Exit condition for the async/hybrid compose path's inner
        `asyncfl_loop` (see `compose()`).

        Must OR in `self._work_done`, not just check the agg-goal match: the
        outer `loop` only re-checks `_work_done` once this inner loop's
        ender (`task_get_weights`) completes, which can block for a long
        time if contributions trickle in slowly. Without this OR,
        `_check_early_stop_conditions()` reaching a cap (e.g.
        `max_data_id_progress`) mid-distribute sets `_work_done`, but the
        inner loop keeps spinning until an agg-goal happens to complete on
        its own -- silently swallowing the early-stop until the launcher's
        external watchdog force-kills the process instead.
        """
        return self._agg_goal_cnt == self._agg_goal or self._work_done

    def _distribute_weights(self, tag: str, task_to_perform: str = "train") -> None:
        self._check_early_stop_conditions()
        if self.is_async:
            logger.info("Inside distribute of async")
            self._distribute_weights_async(tag, task_to_perform)
        else:
            logger.info("Inside distribute of sync")
            self._distribute_weights_sync(tag, task_to_perform)

    def _aggregate_weights(self, tag: str) -> None:
        if self.is_async:
            logger.info("Inside async aggregator")
            self._aggregate_grads_async(tag)
        else:
            logger.info("Inside sync aggregator")
            self._aggregate_grads_sync(tag)

    # TODO: Cleanup compose loop
    def compose(self) -> None:
        """Compose role with tasklets."""
        logger.info(f"Fetch is_async value from config:")
        if self.config.selector.kwargs.get("is_async") is not None:
            self.is_async = self.config.selector.kwargs.get("is_async")
        else:
            self.is_async = False

        if self.is_async:
            super().compose()
            with CloneComposer(self.composer) as _:
                task_internal_init = Tasklet("internal_init", self.internal_init)

                task_reset_agg_goal_vars = Tasklet(
                    "reset_agg_goal_vars", self._reset_agg_goal_variables
                )

                # Created separate put tasklets for train and eval
                task_put_train = Tasklet(
                    "distribute", self.put, TAG_DISTRIBUTE, "train"
                )

                task_get_weights = Tasklet(
                    "aggregate", self._aggregate_weights, TAG_AGGREGATE
                )

                task_init = Tasklet("initialize", self.initialize)

            c = self.composer
            c.unlink()

            loop = Loop(loop_check_fn=lambda: self._work_done)
            # create a loop object for asyncfl to manage concurrency as
            # well as aggregation goal
            asyncfl_loop = Loop(loop_check_fn=self._async_inner_loop_done)
            logger.info("Hybrid compose")

            # chain them again with new tasklets introduced in this class
            (
                task_internal_init
                >> task_init
                >> loop(
                    task_reset_agg_goal_vars
                    # >> asyncfl_loop(task_put >> task_get_weights >>
                    >> asyncfl_loop(task_put_train >> task_get_weights)
                    >> c.tasklet("analysis")
                    >> c.tasklet("save_metrics")
                )
                >> c.tasklet("inform_end_of_training")
            )
        else:
            logger.info("Sync loop")
            super().compose()

            with CloneComposer(self.composer) as _:
                task_internal_init = Tasklet("internal_init", self.internal_init)
                task_pause_exec = Tasklet("pause_exec", self.pause_execution)

                task_reset_agg_goal_vars = Tasklet(
                    "reset_agg_goal_vars", self._reset_agg_goal_variables
                )

                # Created separate put tasklets for train and eval
                task_put_train = Tasklet(
                    "distribute", self.put, TAG_DISTRIBUTE, "train"
                )

                task_put_eval = Tasklet("distribute", self.put, TAG_DISTRIBUTE, "eval")

                # TODO: (DG) Update later, task_get_weights gets both weights from
                # train and eval tasks. Will create a cleaner separation later.
                task_get_weights = Tasklet("aggregate", self.get, TAG_AGGREGATE)

                task_init = Tasklet("initialize", self.initialize)

                task_aggregate_grads_sync = Tasklet(
                    "aggregate", self._aggregate_grads_sync, TAG_AGGREGATE
                )

            c = self.composer
            c.unlink()

            loop = Loop(loop_check_fn=lambda: self._work_done)
            # create a loop object for asyncfl to manage concurrency as well as
            # aggregation goal asyncfl_loop = Loop(loop_check_fn=lambda:
            # self._agg_goal_cnt == self._agg_goal)

            # chain them again with new tasklets introduced in this class
            (
                task_internal_init
                >> task_init
                >> loop(
                    # task_reset_agg_goal_vars
                    task_put_train
                    >> task_aggregate_grads_sync
                )
                >> c.tasklet("inform_end_of_training")
            )

    @classmethod
    def get_func_tags(cls) -> list[str]:
        """Return a list of function tags defined in the top level
        aggregator role."""
        return [TAG_DISTRIBUTE, TAG_AGGREGATE]
