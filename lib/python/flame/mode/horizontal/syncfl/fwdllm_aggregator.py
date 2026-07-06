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
import gc
import logging
import psutil
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Union
import sklearn
import numpy as np
import yaml
from sortedcontainers import SortedDict
import torch.nn.functional as F
from flame.channel import VAL_CH_STATE_HTBT_RECV, VAL_CH_STATE_RECV, VAL_CH_STATE_SEND
from flame.common.constants import DeviceType
from flame.common.util import weights_to_device, weights_to_model_device
from flame.config import OptimizerType, TrainerAvailState
from flame.end import KEY_END_STATE, PROP_END_AVL_STATE, VAL_END_STATE_NONE
from flame.mode.composer import CloneComposer
import pickle
from flame.mode.horizontal.syncfl.top_aggregator import (
    TAG_AGGREGATE,
    TAG_DISTRIBUTE,
    TAG_HEARTBEAT,
)
from sklearn.metrics import (
    confusion_matrix,
    matthews_corrcoef,
)
from flame.mode.horizontal.asyncfl.top_aggregator import (
    RECV_TIMEOUT_WAIT_S,
    TopAggregator as AsyncTopAgg,
    _SIM_GATE_MAX_PASSES,
    _SIM_ORDER_SLACK_S,
)
from flame.mode.message import MessageType
from flame.mode.horizontal.client_duration import real_client_task_train_duration
from flame.mode.tasklet import Loop, Tasklet
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
from flame.telemetry.events import build_agg_eval, build_agg_round, build_utility_belief, build_comm


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


class TopAggregator(AsyncTopAgg):
    """Top level Aggregator implements an ML aggregation
    role."""

    # Phase 4a (root S1): the sim runs REAL forward-grad GPU + server eval, so
    # its physical WALL legitimately exceeds the vclock budget by construction
    # (#6/#13). A wall ceiling = 1× max_runtime_s therefore ALWAYS truncated the
    # sim before its vclock reached the budget (the 2026-07-04 sign-off void).
    # The ceiling is decoupled to a GENEROUS multiple of the budget so it is a
    # true runaway OUTER safety, not the primary stop -- the primary stops are
    # max_data_id_progress (matched-data_id, 4b) and vclock >= max_runtime_s.
    # Override with an explicit `sim_wall_ceiling_s` for a tighter outer bound.
    SIM_WALL_CEILING_FACTOR = 20.0

    # #13 recv-grace floor (fwdllm-scoped override of SyncTopAgg's 2.0). This is
    # the per-pass recv_fifo window the drain waits for an in-flight grad to
    # physically arrive. async_cifar10's 2.0s suffices there (weights come back
    # fast); fwdllm's forward-grad "train" is a REAL GPU pass that can take ~4s,
    # so a 2s window can close before the grad reassembles and force needless
    # extra gate passes. 5s covers the slow GPU here.
    #   TUNABLE: keep this as LOW as possible without compromising correctness --
    #   too low and a genuinely-in-flight grad is missed within the pass; too high
    #   and every stuck-end failsafe pass pays the full window. Re-measure the
    #   real per-trainer GPU wall (fluxtune telemetry) and pull it down toward the
    #   observed max compute + a small slack once the drain is validated.
    SIM_RECV_GRACE_FLOOR_S = 5.0

    def internal_init(self) -> None:
        """Initialize internal state for role."""
        super().internal_init()

        self._trainer_last_model_version = {}

        self._agg_goal_cnt = 0
        self._agg_goal_weights = None
        self._agg_goal = self.config.hyperparameters.aggregation_goal or 1

        self._updates_in_queue = 0
        self._updates_received = {}
        self._per_agg_trainer_list = []
        # end -> canonical commit-order key (modeled_delay D, str(end)) for the
        # current cycle's cohort (K-D31/P2-7a). Populated per contribution in
        # aggregate_weights; consumed by _canonicalize_cohort_commit_order to
        # break equal-D ties by trainer_id IDENTICALLY in real and sim.
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

        # Dictionary to store trainer state: Key = trainer_id, Value = model_version, data_id, iteration_id
        self._trainer_state_dict = {}

        # check if distribute_weights was successful
        self._prev_distribute_weights_success = False

        self.data_id = 0
        self.total_data_bins = 150
        self._is_model_updated = False
        self._model_version = 0

        # Force-advance data_id after this many failed variance checks; None = disabled.
        self._max_iter_per_data_id = getattr(
            self.config.hyperparameters, "max_iterations_per_data_id", None
        )
        if self._max_iter_per_data_id is not None:
            logger.info(
                f"[MaxIterBypass] max_iterations_per_data_id={self._max_iter_per_data_id}"
            )
        self.grad_pool = []
        self.cached_shared_grad_pool_trainable = None
        self.var = None
        self.ends_not_selected_yet = False
        self.iteration_per_data_id = 0

        # end_id -> {dispatch_ts, commit_ts} of its last accepted contribution,
        # emitted per-cycle as contributor_intervals for the R1/W1 rungs (§L.3).
        self._sim_contrib_intervals = {}

        # #15 compute-truthful commit gate (flag-gated; default off = byte-identical).
        # `_sim_recv_min_grad`'s earlier_stuck gate blocks real wall on
        # `_sim_inflight_expected` entries stamped at DISPATCH. A trainer whose
        # weights/VAR=bad payload was sent long ago but has not returned is NOT
        # actually computing (it is idle-in-recv behind the single-threaded drain);
        # waiting for it burns the grace floor / 30s failsafe and throttles commits,
        # which (via hold-to-commit) starves re-dispatch -> concurrency collapses
        # (sim 1.65 vs real 7.69). When on, the gate only blocks on a trainer still
        # within its modeled compute window (dispatched recently enough to plausibly
        # still be computing) -- so a stamped-but-idle phantom no longer defers a
        # ready commit. Hold-to-commit (K-D17b) is untouched; this only unblocks the
        # commit path so held trainers are freed promptly (PARITY_LOGICAL_TASKS.md #15).
        self._sim_compute_truthful_gate = bool(getattr(
            self.config.hyperparameters, "sim_compute_truthful_gate", False))
        # Wall seconds a dispatched grad may still plausibly be computing before it
        # is treated as idle/phantom (comfortably above the ~3.6s mean / 5.6s max
        # observed JVP compute). Only consulted when the gate flag is on.
        _cap = getattr(self.config.hyperparameters, "sim_gate_compute_cap_s", 10.0)
        self._sim_gate_compute_cap_s = float(_cap) if _cap is not None else 10.0
        # end -> wall time its weights/VAR=bad payload was last sent (sim only).
        self._sim_dispatch_wall = {}

        # Selection granularity for the sync path (fwdllm/fwdllm_plus):
        # True (default, preserves pre-existing behavior) = re-select
        # trainers on every SEND-state call, i.e. every iteration of every
        # databin. False = select once per round and reuse that selection
        # across all databins/iterations until self._round advances. The
        # async path (fluxtune) is untouched by this flag.
        self._reselect_each_iteration = bool(
            getattr(
                self.config.hyperparameters, "reselect_each_iteration", True
            )
        )
        self._round_selected_ends = None
        self._round_selected_ends_round = None
        # end_id -> time.time() of its last real accepted contribution (or
        # of first entering the cache, if it hasn't contributed yet) -- lets
        # _prune_departed_from_round_cache also evict a member that's stuck
        # but not formally departed (see ROUND_CACHE_STUCK_TIMEOUT_S).
        self._round_cache_activity_ts: dict = {}

        # Wire staleness_policy from config into an instance attr; without this
        # the message handler's getattr fell back to "none" for every run (K-D15).
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
        # variables related to checking trainer availability
        self._per_trainer_last_heartbeat_ts = {}
        if "heartbeat_freq_s" in self.config.hyperparameters.track_trainer_avail.keys():
            self._trainer_heartbeat_freq_s = (
                self.config.hyperparameters.track_trainer_avail["heartbeat_freq_s"]
            )
        else:
            self._trainer_heartbeat_freq_s = 99999

        if (
            "max_allowed_miss_heartbeats"
            in self.config.hyperparameters.track_trainer_avail.keys()
        ):
            self._trainer_max_miss_heartbeats = (
                self.config.hyperparameters.track_trainer_avail[
                    "max_allowed_miss_heartbeats"
                ]
            )
        else:
            self._trainer_max_miss_heartbeats = 99999

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

        # maintain a set of all trainers that have sent heartbeats previously
        self.all_trainers = set()
        self.minInitialTrainers = self.config.selector.kwargs.get("minInitialTrainers")
        if self.is_async and self.minInitialTrainers is None:
            raise KeyError(
                "minInitialTrainers must be specified in selector config for async fwdllm"
            )
        self.trainer_unavail_durations = None
        self._cached_test_data = None
        logger.info("finished init for sync agg")

    def pause_execution(self):
        time.sleep(1)
        return

    def log_memory(self, tag, device):
        # GPU memory
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)

        # CPU memory
        process = psutil.Process()
        cpu_memory = process.memory_info().rss  # in bytes

        logging.info(
            f"[MEM:{tag}] "
            f"GPU Allocated: {allocated/1e6:.2f} MB | "
            f"GPU Reserved: {reserved/1e6:.2f} MB | "
            f"CPU Memory: {cpu_memory/1e6:.2f} MB | "
            f"Device: {device}, aggregator"
        )

    def print_trainable_params_stats(self, location=""):
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

        print(
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

    # TODO: (DG) Need to update or delete, not used right now
    def _read_heartbeat(self, tag: str) -> None:
        """Receive trainer heartbeat messaages asynchronously.

        This method is overriden from one in synchronous top aggregator
        (..top_aggregator).
        """
        channel = self.cm.get_by_tag(tag)
        if not channel:
            logger.info("No channel found")
            return

        logger.debug(f"Channel {channel} found for tag {tag}")
        # receive heartbeat message from trainers
        msg, metadata = next(channel.recv_fifo(channel.ends(VAL_CH_STATE_HTBT_RECV), 1))
        end, _ = metadata
        if not msg:
            logger.debug(f"No data from {end}; skipping it")
            return

        logger.debug(f"received heartbeat from {end}, will process further")
        self._process_trainer_heartbeat(msg=msg, end=end)

    def _process_trainer_heartbeat(self, msg, end) -> None:
        if MessageType.HEARTBEAT in msg:
            heartbeat_timestamp = msg[MessageType.HEARTBEAT]
            logger.debug(
                f"received heartbeat from {end} "
                f"with timestamp {heartbeat_timestamp} "
                f"at current time: {time.time()}"
            )

            # Add trainer to global_trainer set Used only to check unavailable
            # trainers later
            if end not in self.all_trainers:
                self.all_trainers.add(end)
                logger.debug(f"Added end {end} to all_trainers set")

            # Add trainer to heartbeat dict if it isnt there Add only most
            # recent heartbeat timestamp as value Discard stale heartbeats if
            # received.
            if end not in self._per_trainer_last_heartbeat_ts.keys():
                self._per_trainer_last_heartbeat_ts[end] = heartbeat_timestamp
                logger.debug(
                    f"Added first timestamp for trainer {end} "
                    f"with timestamp {heartbeat_timestamp}"
                )
            elif heartbeat_timestamp > self._per_trainer_last_heartbeat_ts[end]:
                logger.debug(
                    f"Will update timestamp for trainer {end} "
                    f" (current={self._per_trainer_last_heartbeat_ts[end]})"
                    f" with new timestamp {heartbeat_timestamp}"
                )
                self._per_trainer_last_heartbeat_ts[end] = heartbeat_timestamp
            else:
                logger.info(
                    f"the heartbeat for {end} with timestamp "
                    f"{heartbeat_timestamp} was stale"
                )
        else:
            logger.warning(f"Got invalid {msg} while processing heartbeat")

    def read_trainer_unavailability(
        self, trace=None, base_dir: Optional[Union[str, Path]] = None
    ) -> dict:
        """Build task_id -> SortedDict(timestamp -> state) for `trace`.

        Reads from the shared examples/_metadata/ bundle (registry +
        traces), mirroring
        async_cifar10/aggregator/pytorch/main_oort_sync_agg.py's pattern --
        not from the legacy per-trainer json_scripts/trainer_*.json files.

        NOTE: the parameter MUST be named `base_dir` to match the caller in
        ClientAvailability._init_availability (client_availability.py), which
        invokes self.read_trainer_unavailability(trace=..., base_dir=...). This
        method shadows the mixin's wrapper of the same name; a mismatched
        parameter name here raises TypeError at aggregator init (the fwdllm_plus
        ORACULAR-availability crash). The canonical implementation now lives in
        flame.availability.trace.read_trainer_unavailability -- this override is
        a candidate for deletion once mobiperf/syn parity with load_trace is
        confirmed.
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
        format_hash = lambda d: {k: _calculate_hash(v)[:8] for k, v in d.items()}
        logger.debug(f"Trainer grad received {format_hash(trainer_grad)}")
        self.print_trainable_params_stats(
            location="[start,aggregate_grads_from_trainers()]"
        )
        all_zero = all(torch.allclose(g, torch.zeros_like(g)) for g in self.grad)
        logger.info(f"Are all grads zero initially? {all_zero}")

        self.log_memory("start aggregate_grads_from_trainers", self.device)

        # logger.info(f"len(self.model.named_parameters()):
        # {len(self.model.named_parameters())}, len(self.params):
        # {len(self.params)}") self.grad.to(DeviceType.CPU)
        # trainer_grad.to(DeviceType.CPU)
        np = self.model.named_parameters()

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

        # Also accumulate var-check gradients with the same rate if provided.
        # Assumption: grad_for_var_check is an iterable of tensors.
        if grad_for_var_check is not None:
            stacked = torch.stack(list(grad_for_var_check))
            self.grad_for_var_check_list.append(stacked * rate)
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
        """Grad-loop analog of asyncfl._sim_recv_min (simulate_fwdllm.md §J.2).

        Ingest every ready grad message into the sct-ordered reorder buffer
        (keyed by SIM_COMPLETION_TS), HOLD the commit while an in-flight trainer
        is EXPECTED (_sim_inflight_expected) to complete before the buffered
        minimum, then pop and commit the single smallest-sct message, advancing
        the virtual clock to it. Returns one (msg, metadata) -- matching the
        shape of next(channel.recv_fifo(...,1)) -- or (None, ("", now)) when
        nothing is committable.

        Purpose-built rather than calling _sim_recv_min verbatim because fwdllm
        commits GRADIENTS one-per-call and releases concurrency slots on the
        agg-goal boundary (_release_sim_slots_at_agg_goal), NOT per message, and
        must survive variance-FAIL rollbacks without stranding or double-
        committing a grad (§I.5). Slot/selector state is deliberately untouched
        here -- all of it is cleared at the boundary.
        """
        live = [e for e in (recv_ends or []) if channel.has(e)]
        deadline = time.time() + RECV_TIMEOUT_WAIT_S
        for _pass in range(_SIM_GATE_MAX_PASSES):
            grace = self._sim_recv_grace_s()
            # Base probe: the live recv_ends (always drained), minus anything
            # already buffered or committed this cycle.
            _base = [
                e for e in live
                if not self._sim_buffer.has(e) and e not in self._sim_committed
            ]
            _seen = set(_base)
            if getattr(self, "_sim_sct_ordered_drain", False):
                # #13 step 3 -- direct sct-ordered ingest (felix _sim_recv_min:375-390).
                # Drain each live in-flight end's rx queue DIRECTLY (no recv_fifo
                # streamer), taking the FULL live in-flight set. Two wins: (1) the
                # buffer is a COMPLETE snapshot of every arrived grad -- the
                # streamer's background task + shared queue can strand a delivered
                # grad out of the buffer's view (is_rxq_empty true) and let the clock
                # lap it (past-dated commit); (2) drain_ready sweeps all ready rxqs
                # non-blocking and returns on the FIRST arrival, so it does NOT burn
                # the full grace PER not-ready end -- the recv_fifo per-end timeout
                # that dominated the post-step-2 wall (the 4-10s gaps, #13). No
                # ready-gating needed here (non-blocking sweep + poll-to-first).
                to_probe = _base + [
                    e for e in self._sim_inflight_expected
                    if e not in _seen and channel.has(e)
                    and not self._sim_buffer.has(e) and e not in self._sim_committed
                ]
                if to_probe:
                    for m, md in channel.drain_ready(to_probe, timeout=grace):
                        _e = md[0]
                        _s = m.get(MessageType.SIM_COMPLETION_TS)
                        _s = float(_s) if _s is not None else self._vclock.now
                        self._sim_buffer.add(_e, _s, (m, md))
            else:
                # #13 step 2 -- probe-ceiling + ready-gating (felix _sim_recv_min:399-411).
                # ALSO probe an in-flight-expected end that is NOT already a recv_end only
                # if it is physically READY (its grad arrived) OR its modeled completion
                # `exp` is at/before the buffered minimum (+slack). Without this the drain
                # blocked the FULL grace window every pass on far-future / not-yet-arrived
                # stragglers -- the dominant inter-burst wall waste (#13, the 4-10s gaps).
                # Safe against the HOLD gate below: any end that could trigger
                # `earlier_stuck` (exp < bmin - slack) also satisfies exp <= bmin + slack,
                # so the gate's stuck end is always in this probe set -- no new deadlock.
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
                    for m, md in channel.recv_fifo(
                        to_probe, first_k=len(to_probe), timeout=grace
                    ):
                        if m is None:  # no more ready (grace expired or set drained)
                            break
                        _e = md[0]
                        _s = m.get(MessageType.SIM_COMPLETION_TS)
                        _s = float(_s) if _s is not None else self._vclock.now
                        self._sim_buffer.add(_e, _s, (m, md))
            # Gate: earliest expected completion among un-buffered in-flight ends.
            bmin = self._sim_buffer.peek_min_ts()
            _stuck_end, min_stuck = None, None
            # #15 compute-truthful gate: only a trainer still within its modeled
            # compute window can be a live earlier-sct straggler. A trainer whose
            # last dispatch is older than the compute cap (or was never dispatched)
            # is idle-in-recv / phantom -- it will not produce a grad until the drain
            # yields and the aggregator re-dispatches it, so blocking on it deadlocks
            # to the grace/30s failsafe (PARITY_LOGICAL_TASKS.md #15). Skip it so the
            # already-buffered commit proceeds and the loop can re-dispatch.
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
            earlier_stuck = (
                bmin is not None and min_stuck is not None
                and min_stuck + _SIM_ORDER_SLACK_S < bmin
            )
            if bmin is None and not to_probe:
                break  # nothing to commit and nothing in flight
            if not earlier_stuck:
                break  # the buffered minimum is the true next completion
            if time.time() >= deadline:
                # #13 step 1 (felix _sim_recv_min:436-442): the earliest-expected
                # in-flight trainer never physically arrived within the failsafe
                # window. WITHOUT this eviction it stays in _sim_inflight_expected
                # forever, so `earlier_stuck` re-fires the FULL deadline on every
                # future drain cycle -> the composer freezes 30s/commit ->
                # pipeline starvation. Treat the straggler as lost: drop it from
                # the expected set so it can't block future commits, and commit
                # the buffered min now. Sim-only path (real never enters here).
                self._sim_gate_failsafe = getattr(self, "_sim_gate_failsafe", 0) + 1
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
        self._sim_committed.add(_end)
        self._sim_inflight_expected.pop(_end, None)
        # #13 step 4 -- freed-slot refill stamp (felix _sim_recv_min:513-520). This
        # grad commit frees a compute slot; record the just-advanced vclock so the
        # trainer that REFILLS the slot rides THIS vclock (popped FIFO in
        # _distribute_weights_async) instead of the round-start frontier. Spreads
        # each cohort's expected completions across the timeline (matching real's
        # staggered returns) instead of collapsing them at one frozen `_round_now`,
        # so the drain gate stops HOLDING for a batch of same-expected stragglers
        # (the residual 11-12s multi-pass holds after step 3). fwdllm's grad loop is
        # all-train (every commit frees a slot), so no train/eval discriminator is
        # needed. Sim-only + flag-gated (real never enters this method).
        if getattr(self, "_sim_staggered_redispatch", False):
            self._sim_free_slot_ts.append(self._vclock.now)
        # K-D27 (felix-parallel, asyncfl:618): the grad committed -> the trainer is
        # no longer in flight in virtual time -> drop it from pending so it is
        # re-pickable. (_sim_hold_busy_slots reconciles right after, but be explicit.)
        self._sim_pending_commit.discard(_end)
        # Learn this end's MODELED budget (contention-free lower bound) so the
        # gate fires on genuine stragglers for not-yet-observed trainers.
        _b = m.get(MessageType.TRAINING_BUDGET_S) if isinstance(m, dict) else None
        if _b is not None:
            self._sim_trainer_budget[_end] = float(_b)
            self._sim_budget_min = min(self._sim_budget_min, float(_b))
        # Reassert selected_ends == the virtual-time in-flight set after this
        # commit (K-D17b): recv_fifo just marked the freshly-buffered ends RECVD,
        # which strips their slots, but they are still in flight until THEY commit.
        # felix does this reset per-commit in _sim_recv_min; without it the
        # in_flight telemetry (sampled at the next distribute) undercounts ~3x.
        if getattr(self, "_sim_inflight_residence", False):
            self._sim_hold_busy_slots(channel)
        # in_flight (virtual, dispatched-not-committed) vs the physically-computing
        # selected_ends that feeds the in_flight telemetry — kept for concurrency
        # parity debugging.
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
            f"phantom_skip={getattr(self, '_sim_gate_phantom_skip', 0)}"
        )
        return m, md

    def _release_sim_slots_at_agg_goal(self, channel, is_async):
        """Sim slot release at the agg-goal boundary. Two policies (§L / K-D12/D16):

        - async + sim_inflight_residence (fluxtune, c >> agg_goal): COMMIT-THEN-
          CARRY (K-D12). Hold still-busy trainers BEFORE clearing and CARRY the
          surplus buffer to the next fedbuff step (never dropped). The overridden
          `_sim_hold_busy_slots` holds EVERY dispatched-but-not-committed trainer
          (computing ∪ carried) in BOTH its compute slot (selected_ends) and the
          re-pick guard (all_selected) until its grad commits — the virtual-time
          in-flight set, matching felix (K-D17b; supersedes the K-D16 Option-A
          split that wrongly freed the slot on physical RETURN).
        - else (sync barriers c ≈ agg_goal, or residence off): LEGACY DROP
          (K-D5/K-D6) -- no surplus, so clearing is correct and flag-off is
          byte-identical to Batch 1.

        `_sim_committed` always clears so a variance-FAIL re-contributor on the
        rolled-back data_id isn't skipped.
        """
        if not self.simulated:
            return
        if is_async and getattr(self, "_sim_inflight_residence", False):
            self._sim_hold_busy_slots(channel)   # reads buffer/in-flight -> hold before clear
            self._sim_committed.clear()
            return  # keep _sim_buffer / _sim_inflight_expected -> carry surplus
        self._sim_committed.clear()
        self._sim_buffer.clear()
        self._sim_inflight_expected.clear()
        # K-D27: legacy-drop path abandons all in-flight -> nothing is pending.
        # (async path clears via _sim_hold_busy_slots' reconcile; sync barriers use
        # the random selector where the ref is inert, but keep the set bounded.)
        self._sim_pending_commit.clear()
        if is_async:
            self._sim_hold_busy_slots(channel)

    def _sim_hold_busy_slots(self, channel) -> None:
        """Assert `selected_ends` == the VIRTUAL-time in-flight set, i.e. every
        dispatched-but-not-committed trainer (`_sim_inflight_expected` ∪ buffered
        surplus), holding BOTH its compute slot (`selected_ends`, drives
        `extra = c − len(selected_ends)`) AND its re-pick guard (`all_selected`)
        until its grad COMMITS. This realigns fwdllm with the felix reference
        (`asyncfl/top_aggregator.py::_sim_hold_busy_slots` / `_sim_recv_min`
        :606-627,1453) after the K-D16 Option-A deviation.

        WHY (K-D17b): a returned-but-uncommitted trainer is still in flight in
        VIRTUAL time — its grad commits only when the vclock reaches its sct — so
        its slot is genuinely occupied. Option A freed that slot on physical
        RETURN (a wall event with no virtual-time meaning): `selected_ends`
        collapsed to the ~3 physically-computing while ~8 were truly in flight,
        so the `in_flight` telemetry (`len(selected_ends)`) undercounted 3× and
        `extra` read false free capacity (measured `_sim_inflight_expected`≈7.8
        vs `selected_ends`≈2.7 vs real 9.75). Freeing on physical return is the
        exact "over-selection" hazard felix documents; fwdllm's added guard
        turned it into an under-count instead. Holding to COMMIT also keeps
        one-in-flight-per-trainer across the carry boundary + variance-FAIL
        rollbacks (principle #4). Idempotent — safe to call per-commit
        (`_sim_recv_min_grad`) and at the boundary. fwdllm-class override only
        (adds the triplet prune); async_cifar10 (asyncfl) untouched (principle #8).
        """
        sel = getattr(channel, "_selector", None)
        if sel is None:
            return
        requester = getattr(sel, "requester", None)
        all_selected = getattr(sel, "all_selected", None)
        selected_ends = getattr(sel, "selected_ends", None)

        buffered = set(self._sim_buffer.pending_ends())          # returned, grad carried
        # Outstanding = still in flight in virtual time. Membership in
        # `_sim_inflight_expected` (popped on commit @811) or `_sim_buffer` (popped
        # on commit @793) IS the "not yet committed" truth. Do NOT subtract
        # `_sim_committed` (K-D27 fix): it is a STALE, cross-cycle marker cleared
        # only at the agg-goal boundary, so a trainer that committed and was then
        # legitimately re-picked + re-dispatched (re-added to `_sim_inflight_expected`)
        # would be wrongly dropped from `outstanding` -> re-pickable while its NEW
        # dispatch is still in flight -> R1 residence violation. A trainer that
        # committed THIS cycle is already absent from both sets, so the subtraction
        # was redundant for it and harmful for the re-dispatch case.
        outstanding = set(self._sim_inflight_expected) | buffered
        # `_sim_pending_commit` is the aggregator's authoritative VIRTUAL in-flight
        # set; reconcile it to `outstanding` IN PLACE (clear+update, never rebind --
        # the selector holds a live reference) so a genuinely-committed trainer drops
        # out (re-pickable) while a dispatched-but-uncommitted one stays. Bind the
        # reference so async_oort's eligibility filter excludes it (see async_oort
        # `_pending`) regardless of all_selected churn. `|=` (the old accumulate)
        # would never shrink -> a committed trainer would be starved forever.
        self._sim_pending_commit.clear()
        self._sim_pending_commit.update(outstanding)
        sel._agg_pending_commit_ref = self._sim_pending_commit
        # Prune the (model_version, data_id, iteration) triplet guard to the
        # still-outstanding set (a trainer whose grad committed is re-pickable).
        self._trainer_state_dict = {
            e: v for e, v in getattr(self, "_trainer_state_dict", {}).items()
            if e in outstanding
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
        if recv_ends is None and not sim_has_pending:
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
    def _release_end_on_return(self, channel, end) -> None:
        """Release a returned trainer's compute slot + re-pick guard on grad
        RETURN -- EXCEPT on the async sim residence path, where the grad is
        CARRIED and commits later in virtual time. There the guard/slot
        lifetime is owned solely by `_sim_hold_busy_slots` (held to COMMIT,
        mirroring felix's agg-goal-boundary release, `asyncfl:1316`); releasing
        `all_selected` here on the physical RETURN would re-eligible a trainer
        whose carried grad has NOT yet committed -> re-dispatch-while-in-flight
        -> R1 residence violation (K-D19: 0% -> 44.7%). Real mode / non-residence:
        return ~= commit, so release immediately as before (byte-identical).
        """
        if self.is_async:
            if getattr(self, "simulated", False) and getattr(
                self, "_sim_inflight_residence", False
            ):
                return  # guard/slot held to COMMIT by _sim_hold_busy_slots
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
            # staleness_policy (flame/config.py). REJECT: round_data_id, exact.
            # ACCEPT: none (no gate); fedbuff (consume + down-weight by V'-V in
            # aggregate_grads_from_trainers) -- the async default, since its
            # carried surplus grads (§L) are stale by construction (K-D13).
            # model_version alone identifies (round, data_id) when
            # inc_model_version_per_data_id is set, so round_data_id needs no
            # extra field; exact also checks iteration_per_data_id.
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
            # tail_s in the per-round wall decomposition (Stage A2).
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
                # Client INTRINSIC duration (WALL_SEND - WALL_RECV, §S.dur); falls
                # back to recv - dispatch (timestamp - sent_ts) when the trainer did
                # not stamp the client times. Keeps the selector/telemetry duration
                # server-overhead-free, consistent with the other aggregators.
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
                self._sim_contrib_intervals[end] = {
                    "dispatch_ts": float(_disp) if _disp is not None else None,
                    "commit_ts": float(_comm) if _comm is not None else None,
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
        self._round_cache_activity_ts[end] = time.time()

        channel._selector.ordered_updates_recv_ends.append(end)
        self._updates_in_queue += 1
        self._per_agg_trainer_list.append(end)

        # Canonical commit-order key (K-D31/P2-7a): the trainer's pure modeled
        # delay D (deterministic from the registry) + str(end) as the tie-break.
        # Lets _canonicalize_cohort_commit_order reproduce real's D-ordered
        # arrival AND break equal-D ties by trainer_id identically in both modes.
        # None when delays are off / the trainer did not stamp D -> that cycle
        # falls back to arrival order (byte-identical legacy behavior).
        _md = msg.get(MessageType.MODELED_DELAY_S)
        self._commit_key_by_end = getattr(self, "_commit_key_by_end", {})
        self._commit_key_by_end[end] = (
            (float(_md), str(end)) if _md is not None else None
        )

        # Re-pick guard (async_oort triplet filter): record the (model_version,
        # data_id, iteration) this trainer just CONTRIBUTED at, so it is not
        # re-selected until the agg version advances (the commit boundary prunes
        # committed ends in _sim_hold_busy_slots). Stamped on RETURN, not
        # dispatch (§K-D17): at dispatch the whole cohort matches _curr_agg_version
        # and, since the version only advances at a commit, that froze the entire
        # eligible pool before any commit could happen -> re-dispatch deadlock.
        # The compute slot (selected_ends) already guards an in-flight trainer;
        # the triplet only needs to cover the return->commit carry window.
        # Residence-only so the sync baselines keep an empty map (inert).
        if getattr(self, "simulated", False) and getattr(
            self, "_sim_inflight_residence", False
        ):
            _agg_ver = getattr(self, "_curr_agg_version", None)
            if _agg_ver is not None:
                self._trainer_state_dict[end] = _agg_ver

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
            logger.debug(
                f"Calling aggregate_grads_for_trainers with grad_for_var_check: {_calculate_hash(grad_for_var_check)}"
            )
            # Use this message's stat_utility, not the channel property --
            # the property is set below, after this call, so reading it here
            # was always None on a trainer's first contribution (crashing
            # fedbuff's weight_factor() on `1 + None`).
            self.aggregate_grads_from_trainers(
                trainer_gradients,
                version_for_rate=version_for_rate,
                stat_utility=msg[MessageType.STAT_UTILITY],
                grad_for_var_check=grad_for_var_check,
                jvp_for_snr_check=jvp_for_snr_check,
            )

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
        logger.info(
            f"Received grads from {end}. It was trained on model version {version}, with {count} samples"
        )
        # Release the slot/guard on return -- but the async sim residence path
        # defers that release to COMMIT (K-D19); see _release_end_on_return.
        # (cleanup_recvd_end() is sync-only, random selector; the async batch
        # _cleanup_provided_ends path is the only one async selectors implement.)
        self._release_end_on_return(channel, end)
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
            f"==== Model version incremented to {self._curr_agg_version} with updates from {n_unique} unique trainers. Stats of participating trainers: \n"
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
        """Reorder THIS cycle's cohort commits to a canonical (D, trainer_id)
        order — IDENTICALLY in real and sim (K-D31/P2-7a).

        Real receives updates in strict modeled-delay (D) order (device wall =
        D, K-D29); sim commits in sct order (= D order). The sole residual
        real↔sim divergence is the tie-break when two trainers share a D (a
        realistic registry collision, e.g. trainers 3 & 9 both 13.0s → D=6.5):
        real breaks it by physical arrival, sim by sct-sort. Both tied members
        land in the SAME split-half so `var` is unchanged, but the EXACT-order
        `cohort_sequence` rung flags the swap. Here we canonicalize: sort the
        cohort by (D, str(end)) so equal-D ties break by trainer_id in BOTH
        modes → identical receive order; `var`/grads stay bit-identical (the
        aggregated grad is an order-independent sum; the split-half only ever
        reshuffles WITHIN a half on a tie).

        Scope = this cycle only. `grad_for_var_check_list` ACCUMULATES across a
        data_id's iterations (reset on commit) while `_per_agg_trainer_list` is
        per-cycle, so we reorder just the TRAILING len(cohort) slice of the
        grad/jvp lists — the sync barrier appends this cohort contiguously at
        the end, in `_per_agg_trainer_list` order.

        No-op unless every contributor stamped a modeled delay (delays on) AND
        a tie actually changes the order → delays-off / legacy runs and the
        common non-tie case keep arrival order (byte-identical).
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
        self._per_agg_trainer_list = [ends[i] for i in perm]
        # Reorder the trailing cohort slice of the accumulating var/jvp lists in
        # lockstep. Guard on length: only when the slice aligns 1:1 with this
        # cohort (it does under the sync barrier; a mismatch means a non-grad
        # message slipped in → leave the lists untouched rather than corrupt).
        for lst in (self.grad_for_var_check_list, self.jvp_for_snr_check_list):
            if len(lst) >= n:
                tail = lst[-n:]
                lst[-n:] = [tail[i] for i in perm]
        logger.info(
            f"[COMMIT_CANON] equal-D tie → reordered {n}-cohort to (D,id) order "
            f"(perm={perm}); var/grads unchanged, receive order now real↔sim identical."
        )

    @timer_decorator
    def _process_aggregation_goal_met(self, tag, channel, is_async=False):
        logger.info(
            f"Aggregation goal {self._agg_goal} reached. Performing FwdLLM aggregation."
        )

        # Canonicalize this cohort's commit order to (D, trainer_id) BEFORE the
        # telemetry snapshot and aggregate() so the recorded receive order and
        # the split-half var are computed on the SAME deterministic order in
        # real and sim (K-D31/P2-7a). Gated on the presence of commit-key state
        # (populated per contribution in aggregate_weights) → skipped entirely
        # when delays are off / no keys were stamped (arrival order, unchanged).
        if getattr(self, "_commit_key_by_end", None):
            self._canonicalize_cohort_commit_order()

        # Snapshot for this cycle's agg_round telemetry (emitted further down,
        # after self._per_agg_trainer_list is cleared and self._model_version
        # may have advanced -- see build_agg_round call below).
        _cycle_contributors = list(self._per_agg_trainer_list)
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

        # Per-round wall decomposition (Stage A2, grounds #6 / K-D20): the FedAvg
        # merge, the dispatch→last-grad barrier wait, and the last-grad→commit
        # drain tail (a real-transport ARTIFACT the sim's all-k barrier does not
        # model). eval_s is measured on the pass path below. All wall (real);
        # the sim advances the vclock instead, so these read ~0 there.
        _agg_start_wall = time.time()
        self.aggregate(self._round)
        _aggregate_fedavg_s = time.time() - _agg_start_wall
        _disp = getattr(self, "_round_dispatch_wall_ts", None)
        _lastg = getattr(self, "_last_grad_wall_ts", None)
        _barrier_wait_s = (_lastg - _disp) if (_disp and _lastg) else None
        _drain_tail_s = (_agg_start_wall - _lastg) if _lastg else None
        _eval_s = None  # set below only when the variance gate passes (eval runs)

        _var_thr = getattr(self, "var_threshold", None)
        _ratio = (
            float(self.var) / _var_thr
            if (self.var is not None and _var_thr not in (None, 0))
            else None
        )
        logger.info(
            f"[IterProgress] data_id={self.data_id} "
            f"iter={self.iteration_per_data_id} "
            f"max_iter={self._max_iter_per_data_id} "
            f"var={self.var} var_thr={_var_thr} ratio={_ratio} "
            f"var_good_enough={self.var_good_enough} "
            f"force_commit_planned={_force_commit_planned}"
        )

        # Snapshot the cycle identity BEFORE the pass/fail branch mutates
        # data_id/iteration_per_data_id below. The emitted `data_id`/
        # `iteration_per_data_id` fields are post-mutation (a commit advances
        # data_id and zeroes iteration, so a commit event carries the NEXT
        # data_id) -- fine for the analyzer's progress axis but ambiguous for
        # the variance-cadence rungs (V1). `cycle_data_id`/`cycle_iteration`
        # unambiguously identify the data_id this cycle worked on and its
        # 0-based attempt index, so V1 = count(cycles) grouped by cycle_data_id
        # is exact for both natural-pass and force-commit paths (§K-D9).
        _cycle_data_id = self.data_id
        _cycle_iteration = self.iteration_per_data_id
        # Pool sizes at the variance gate (before _update_state_after_payload_
        # prepared clears grad_pool on a commit): grad_pool = realized
        # contributions this data_id (G2); cached_v = carried aggregated pool
        # across variance-FAIL rollbacks (V3). See §K-D9. getattr-guarded like
        # var_threshold so test doubles without the pools still emit.
        _grad_pool = getattr(self, "grad_pool", None)
        _grad_pool_size = len(_grad_pool) if _grad_pool is not None else None
        _cached_v = getattr(self, "cached_shared_grad_pool_trainable", None)
        _cached_v_size = len(_cached_v) if _cached_v is not None else 0
        # Per-contributor [dispatch, commit] intervals for R1/W1 (§L.3): one
        # entry per end that committed into THIS cycle. History is preserved
        # because each cycle emits its own list (the per-end dict is overwritten
        # on a later contribution, but the emitted event already captured it).
        _contrib_map = getattr(self, "_sim_contrib_intervals", None) or {}
        _contributor_intervals = [
            {"end": str(_e), **_contrib_map.get(_e, {"dispatch_ts": None,
                                                     "commit_ts": None})}
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
            _eval_start_wall = time.time()
            result, _, _ = self.eval_model()
            _eval_s = time.time() - _eval_start_wall  # GENUINE server-eval term (B1)
            # B1 (K-D20 #6): the server eval is genuine algorithmic time between
            # committed data_ids that the sim's sct model omits, so the vclock
            # under-counts by ~eval_s/commit. When enabled, charge the MEASURED
            # eval wall (sim runs real eval compute, so it ≈ real's) to the
            # vclock -- consistent with the existing model already putting
            # real_gpu on the vclock via sct (principle #1: genuine compute, not
            # transport overhead). Config-gated OFF ⇒ byte-identical.
            if self.simulated and getattr(
                self.config.hyperparameters, "sim_model_eval_time", False
            ):
                self._vclock.advance(self._vclock.now + _eval_s)
            logger.info(
                f"Round {self._round}, Data ID {self.data_id} Eval Loss: {result['eval_loss']}"
            )
            if telemetry.is_enabled():
                try:
                    ev, fields = build_agg_eval(
                        round_num=self._round,
                        metrics={
                            "test-loss": result.get("eval_loss"),
                            "test-accuracy": result.get("acc"),
                            "mcc": result.get("mcc"),
                            # fwdllm's round is coarse (advances only once all
                            # total_data_bins data_ids finish) -- data_id/
                            # iteration_per_data_id let the analyzer's
                            # progress_key() (scripts/analysis/analyze_run.py)
                            # place this eval on a meaningful x-axis instead of
                            # collapsing every eval in a round onto one point.
                            "data_id": self.data_id,
                            "iteration_per_data_id": self.iteration_per_data_id,
                        },
                    )
                    telemetry.emit(ev, **fields)
                except Exception as e:  # telemetry must never break training
                    logger.debug(f"agg_eval telemetry emit failed: {e}")
            self.data_id += 1
            self.iteration_per_data_id = 0
            self._is_model_updated = True

            if self.config.hyperparameters.inc_model_version_per_data_id:
                self._model_version += 1
            else:
                self._model_version = self._round

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
            # Speedup instrumentation (§H issue #13 / principle #13): the sim
            # must run virtual time FASTER than physical wall (sim_rate >= 1).
            # fwdllm emitted vclock_now but no paired wall stamp, so the
            # slowdown (sim_rate~0.37x in the 2026-07-04 runs) was invisible.
            # Emit wall_elapsed_s in BOTH modes (real needs it for wall_speedup
            # = real_wall/sim_wall) and sim_rate = vclock/wall in sim only.
            # agg_start_time_ts is re-anchored past the join wait (syncfl base),
            # so this excludes the initial ~join stall.
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
                # composer loop bypasses, so it never fired — emit it here.
                _last = getattr(self, "_last_vclock_log_wall_ts", 0.0)
                if time.time() - _last >= 30.0 and _sim_rate is not None:
                    logger.info(
                        f"[VCLOCK_PROGRESS] vclock={float(self._vclock.now):.1f}s "
                        f"wall={_wall_elapsed_s:.1f}s sim_rate={_sim_rate:.3f} "
                        f"(virtual-s/wall-s; <1 == SLOWDOWN) round={self._round} "
                        f"data_id={self.data_id}"
                    )
                    self._last_vclock_log_wall_ts = time.time()

            # Intrinsic algorithmic span of this cycle (#6 anchor): the GENUINE
            # per-cycle work the sim charges to the vclock -- the barrier (slowest
            # committed trainer's intrinsic compute+delay) plus the server eval
            # (commit cycles only). Must mirror the sim vclock's COMPOSITION exactly
            # (barrier sct + eval fold): the FedAvg merge is deliberately EXCLUDED
            # because the sim does NOT charge it to the vclock -- including it made
            # real intrinsic overshoot sim by ~fedavg×cycles (wall_disparity 5.5->1.6
            # once dropped). If fedavg is ever folded into the sim vclock, add it
            # here too. Emitted in BOTH modes so the clock-rate rungs (K2/K3/K3b/K8/
            # U2 + wall_disparity) anchor REAL on its intrinsic time instead of raw
            # wall Δts: real's wall bundles a ~constant inter-round transport ARTIFACT
            # (mqtt re-fetch/redistribute/drain-tail/sleeps) the sim (correctly,
            # principle #1) omits, which spuriously fails #6. The barrier uses the
            # trainer INTRINSIC duration (PROP_CLIENT_TASK_TRAIN_DURATION = WALL_SEND-
            # WALL_RECV, _cycle_speed_s) not the agg-side barrier_wait_s, which reads
            # ~0 in real because real trainers pipeline (grads pre-queued at dispatch).
            # max() = the sync barrier (MAX-of-K sct, principle #12).
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
                        # this (K10); fwdllm never emitted it (§H finding #4).
                        "vclock_now": (self._vclock.now if self.simulated
                                       and getattr(self, "_vclock", None) else None),
                        "data_id": self.data_id,
                        "iteration_per_data_id": self.iteration_per_data_id,
                        "var": self.var,
                        "var_threshold": getattr(self, "var_threshold", None),
                        "var_good_enough": self.var_good_enough,
                        "force_commit_planned": _force_commit_planned,
                        "is_async": is_async,
                        # Variance-cadence rung inputs (Batch 2, §K-D9):
                        # cycle-relative identity for V1, pool sizes for V3/G2.
                        "cycle_data_id": _cycle_data_id,
                        "cycle_iteration": _cycle_iteration,
                        "grad_pool_size": _grad_pool_size,
                        "cached_v_size": _cached_v_size,
                        # R1/W1 residence rungs (§L.3): per-contributor
                        # [dispatch_ts, commit_ts] intervals for this cycle.
                        "contributor_intervals": _contributor_intervals,
                        # Per-round wall decomposition (Stage A2, #6/K-D20):
                        # barrier wait + drain tail (artifact) + fedavg + eval.
                        "barrier_wait_s": _barrier_wait_s,
                        "drain_tail_s": _drain_tail_s,
                        "aggregate_fedavg_s": _aggregate_fedavg_s,
                        "eval_s": _eval_s,
                        # #6 anchor: real's GENUINE per-cycle algorithmic time
                        # (barrier+fedavg+eval), the like-for-like counterpart to
                        # the sim's Δvclock. Lets the clock-rate rungs exclude
                        # real's inter-round transport artifact. See computation above.
                        "intrinsic_span_s": _intrinsic_span_s,
                        # Speedup metric (§H #13): wall in both modes; sim_rate
                        # = vclock/wall (sim only, None in real). sim_rate < 1
                        # means the sim is a SLOWDOWN (broken, principle #13).
                        "wall_elapsed_s": _wall_elapsed_s,
                        "sim_rate": _sim_rate,
                    },
                )
                telemetry.emit(ev, **fields)
            except Exception as e:  # telemetry must never break training
                logger.debug(f"agg_round telemetry emit failed: {e}")

        self._updates_in_queue -= self._agg_goal
        self._per_agg_trainer_list = []
        self._commit_key_by_end = {}  # cohort-scoped (K-D31/P2-7a)

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
        # reorder-buffer entries + in-flight gate, and (async) release held
        # concurrency slots -- so the next agg-goal cycle of a rolled-back
        # data_id starts clean (see §I.5/§J.2). Reached by BOTH the variance-
        # PASS and variance-FAIL branches. Inert in real mode.
        if self.simulated:
            self._release_sim_slots_at_agg_goal(channel, is_async)

        # Centralized cleanup
        # self._force_cuda_memory_cleanup()

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
        # Real-only: "commit 1 per pass" relies on uncommitted msgs persisting in
        # the queue. The sim barrier drains + drops past first_k, so clamping to 1
        # strands the cohort -> deadlock. See simulate_fwdllm.md §F #8.
        if self.ends_not_selected_yet and not self.simulated:
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
        # Sim barrier (Batch 1): instead of committing by physical arrival,
        # _sync_sim_recv_first_k commits the num_min_req trainers with the
        # SMALLEST modeled sim_completion_ts (the k that would physically finish
        # first in real), advancing the vclock to the k-th smallest -- immune to
        # arrival jitter. It returns an ascending-sct list of (msg, metadata) and
        # stamps PROP_CLIENT_TASK_TRAIN_DURATION per commit; we then feed each
        # through the SAME per-message path. Real branch byte-identical to before.
        if self.simulated:
            committed = self._sync_sim_recv_first_k(
                channel, channel.ends(), num_min_req
            )
            _barrier_durs = []
            for msg, metadata in committed:
                end, timestamp = metadata
                if not msg:
                    continue
                # dispatch-relative completion for the U6 barrier anchor (sim's
                # analog of WALL_SEND - dispatch) = the modeled round duration.
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
            # U6 visibility lag on a strict sync barrier: lag_i = max_completion
            # - completion_i over the committed cohort. Stashed for telemetry /
            # the U6 parity rung; full emission is deferred (§K-D7).
            self._sync_barrier_lags_s = self._barrier_anchored_lags(_barrier_durs)
            logger.info(
                f"[SYNC_SIM_BARRIER] round={self._round} committed={len(committed)} "
                f"T_v={self._vclock.now:.1f} barrier_lags_s={self._sync_barrier_lags_s}"
            )
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
    def eval_model(self, epoch=0, global_step=0, device=None):
        if not device:
            device = self.device

        logger.info(f"device inside eval_model() is set to: {device}")
        self.log_memory("start eval_model", self.device)

        results = {}

        eval_loss_total = torch.tensor(0.0, device=device)
        num_eval_steps = 0
        test_sample_len = len(self.test_global.dataset)

        # Move model to device before performing the eval
        self.model.to(device)
        self.model.eval()
        self.fmodel, self.params, self.buffers = fc.make_functional_with_buffers(
            self.model
        )

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

                output = self.model(x)
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
            f"# of batches: {num_eval_steps} with (batch_size, seq_len): {input_ids_all.shape}. test_sample_len: {test_sample_len}, preds.shape: {preds.shape}, location of model: {next(self.model.parameters()).device}"
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

        # Can delete x, labels, output, logits, loss in case we run into any memory issues
        self._force_cuda_memory_cleanup()

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

    def hearbeat_trainer_avail_check(self, end: str) -> bool:
        picked_trainer_is_available = True
        last_acceptable_heartbeat_ts = time.time() - (
            self._trainer_max_miss_heartbeats * self._trainer_heartbeat_freq_s
        )

        # return True if: heartbeat was received from trainer and it is within
        # last_acceptable_heartbeat_ts

        # return False if: if end isnt in heartbeat dict, means that the trainer
        # hasn't given a heartbeat in a while and was removed based on
        # last_acceptable_heartbeat_ts

        # NOTE: During agg init, it might have registered a trainer, but not
        # received heartbeat in such a scenario, we return True so that agg is
        # able to send init_weights to trainer and start the training process
        # this is when trainer not in all_trainers and not in dict

        if (end not in self._per_trainer_last_heartbeat_ts.keys()) and (
            end not in self.all_trainers
        ):
            picked_trainer_is_available = True
            logger.info(
                f"Might be trainer init(), trainer {end} hasnt sent any"
                f" heartbeats yet, but we return True"
            )
        elif end not in self._per_trainer_last_heartbeat_ts.keys():
            picked_trainer_is_available = False
            logger.debug(f"Trainer {end} was already marked unavailable")
        elif self._per_trainer_last_heartbeat_ts[end] < last_acceptable_heartbeat_ts:
            del self._per_trainer_last_heartbeat_ts[end]
            picked_trainer_is_available = False
            logger.info(
                f"Trainer {end} missed max_allowed_heartbeats, " f"marked unavailable"
            )
        elif self._per_trainer_last_heartbeat_ts[end] >= last_acceptable_heartbeat_ts:
            picked_trainer_is_available = True
            logger.debug(f"Trainer {end} is available")
        else:
            logger.error(f"Availability check failed, trainer {end}, returning True")

        return picked_trainer_is_available

    def get_unavailable_trainers(self) -> list:
        # Works only for heartbeat based right now TODO: (DG) Extend for other
        # trainer_avail_checks too
        current_unavailable_trainers = [
            end
            for end in self.all_trainers
            if end not in self._per_trainer_last_heartbeat_ts.keys()
        ]
        return current_unavailable_trainers

    def check_trainer_availability(self, end: str) -> bool:
        picked_trainer_is_available = True
        if self.track_trainer_avail["enabled"] == "False":
            return True
        elif self.track_trainer_avail["type"] == "ORACULAR":
            picked_trainer_is_available = self._trace_read_avail_check(end)
        elif self.track_trainer_avail["type"] == "HEARTBEAT":
            picked_trainer_is_available = self.hearbeat_trainer_avail_check(end)

        return picked_trainer_is_available

    @timer_decorator
    def _prepare_distribution_payload(self, task_to_perform: str, force_weights: bool = False):
        """Build a WEIGHTS payload (always) or a VAR=bad payload (when var fails and force_weights=False)."""
        if not self.var_good_enough and not force_weights:
            logger.info(
                f"[PreparePayload/VAR=bad] var={self.var} > "
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
        _reason = (
            f"var_good_enough=True (var={self.var} <= thr={_var_thr})"
            if self.var_good_enough
            else f"force_weights=True (var={self.var}, thr={_var_thr})"
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
        """
        selector = getattr(channel, "_selector", None)
        if selector is not None and hasattr(selector, "selected_ends"):
            selector.selected_ends = set(selector.selected_ends) | set(ends)

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
        now = time.time()
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

    def _select_ends_respecting_reselect_gate(self, channel, task_to_perform: str):
        """Return the SEND-state-selected ends, honoring
        `self._reselect_each_iteration`.

        True (default): re-invoke the selector every call. False: accumulate
        selections into a per-round cache, re-invoking the selector each
        call until the cache reaches `self._agg_goal` (trainers join the
        channel asynchronously, so one early call may only see a few of
        them); then reuse the cache until `self._round` advances.
        """
        if self._round_selected_ends_round != self._round:
            self._round_selected_ends = None
            self._round_selected_ends_round = self._round
            self._round_cache_activity_ts = {}

        if not self._reselect_each_iteration:
            self._prune_departed_from_round_cache(channel)

        if not self._reselect_each_iteration and self._round_selected_ends is not None:
            target = getattr(self, "_agg_goal", len(self._round_selected_ends))
            if len(self._round_selected_ends) >= target:
                ends = list(self._round_selected_ends)
                logger.info(
                    f"[ReselectGate] reselect_each_iteration=False; reusing "
                    f"cached per-round selection ends={ends} for round={self._round}"
                )
                self._rearm_recv_eligibility(channel, ends)
                return ends

        new_ends = channel.ends(VAL_CH_STATE_SEND, task_to_perform)
        if not self._reselect_each_iteration and new_ends:
            merged = list(self._round_selected_ends or [])
            for end in new_ends:
                if end not in merged:
                    merged.append(end)
                    # First time this end enters the cache -- starts its
                    # stuck-timeout clock (reset again on each real accepted
                    # contribution, see _process_single_trainer_message).
                    self._round_cache_activity_ts[end] = time.time()
            self._round_selected_ends = merged
            logger.info(
                f"[ReselectGate] reselect_each_iteration=False; accumulated "
                f"per-round selection ends={merged} "
                f"({len(merged)}/{getattr(self, '_agg_goal', '?')}) for round={self._round}"
            )
            self._rearm_recv_eligibility(channel, merged)
            return merged
        return new_ends

    def _await_dispatchable_under_scarcity(self, task_to_perform: str) -> None:
        """Real-mode sync-barrier liveness under availability scarcity (Stage C).

        With `agg_goal` clients required but a trace (e.g. mobiperf_2st) keeping
        the eligible pool below `agg_goal`, the plain loop hot-re-dispatches every
        pass -- burning the wall budget with no progress and ballooning the log
        (payload-size lines ×∞, the observed 292 MB). Parity-faithful fix: KEEP
        the cohort == `agg_goal` and WAIT for availability to recover
        (sleep-to-next-avail, mirroring the trainer's `wait_until_next_avl` loop)
        rather than spin -- the sim assembles the same full cohort by jumping its
        vclock past the unavailable window, so cohort size stays identical
        real↔sim. Self-terminates via `_check_early_stop_conditions` at
        `max_runtime_s`. Byte-identical when availability tracking is off
        (`trainer_event_dict is None` ⇒ nobody unavailable ⇒ never waits) and on
        the sim path (the vclock, not a wall sleep, models the wait).
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
        format_hash = lambda d: {k: _calculate_hash(v)[:8] for k, v in d.items()}
        logging.info(
            f"Model distributed to clients (Hashed): {format_hash(global_model_params)}"
        )
        self.weights = global_model_params

        # Real-transport pad to let just-distributed messages settle before the
        # selection read (real-mode MQTT artifact, principle #8). No sim analog:
        # the sim orders by sct, not physical arrival, so this is pure wall
        # overhead there (≥28.7 s/run, Stage E) -- skip it. Real unchanged.
        if not self.simulated:
            logger.debug(f"Starting busy wait at time {time.time()}")
            time.sleep(0.1)
            logger.debug(f"Ended busy wait at time {time.time()}")

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

        self._curr_agg_version = (
            self._model_version,
            self.data_id,
            self.iteration_per_data_id,
        )
        logger.debug(
            f"Aggregator version state (model_version, data_id, iteration_id): {self._curr_agg_version}"
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

        payload_without_weights = None
        payload_with_weights = self._prepare_distribution_payload(task_to_perform, force_weights=True)
        if not self.var_good_enough:
            payload_without_weights = self._prepare_distribution_payload(task_to_perform, force_weights=False)


        self._update_state_after_payload_prepared()

        # Sim-clock dispatch stamp (Batch 1): in the sync barrier every selected
        # end dispatches at the same virtual instant, so one _round_now stamp is
        # injected into each payload variant (the trainer bases its modeled sct
        # on SIM_SEND_TS) and recorded as a per-end property. Inert in real mode:
        # SIM_SEND_TS is absent -> the trainer stamps nothing -> the aggregator
        # falls back to arrival order (byte-identical to today).
        _round_now = self._vclock.now if self.simulated else None
        if self.simulated:
            for _p in (payload_with_weights, payload_without_weights):
                if _p is not None:
                    _p[MessageType.SIM_SEND_TS] = _round_now

        for end in ends:
            trainer_version = self._trainer_last_model_version.get(end, -1)
            is_stale = (trainer_version != self._model_version)

            if self.var_good_enough:
                payload = payload_with_weights
            else:
                if is_stale:
                    payload = payload_with_weights
                    logger.info(f"Trainer {end} hasn't received weights for model_version {self._model_version} (has {trainer_version}). Sending WEIGHTS payload instead of VAR=bad.")
                else:
                    payload = payload_without_weights

            logger.debug(
                f"Setting channel property {PROP_ROUND_START_TIME} for "
                f"end {end}. For round {self._round} at time: {datetime.now()}"
            )
            channel.set_end_property(
                end, PROP_ROUND_START_TIME, (self._round, datetime.now())
            )
            if self.simulated:
                channel.set_end_property(end, PROP_SIM_SEND_TS, _round_now)

            if self.var_good_enough:
                logger.info(
                    f"sending weights to {end} with model_version: {self._model_version}, data_id: {self.data_id} for task: {task_to_perform}"
                )

                sizes_mb = {
                    key.name if hasattr(key, "name") else str(key): len(
                        pickle.dumps(value)
                    )
                    / (1024 * 1024)
                    for key, value in payload.items()
                }
                total_size_mb = sum(sizes_mb.values())
                _send_bytes = int(round(total_size_mb * 1024 * 1024))
                _payload_kind = "weights"

                logger.info(
                    f"[DEBUG] Payload size breakdown for {end}: "
                    + ", ".join([f"{k}: {v:.2f} MB" for k, v in sizes_mb.items()])
                    + f", Total: {total_size_mb:.2f} MB"
                )
            else:
                logger.info(
                    f"sending var = bad to {end} with model_version: {self._model_version}, round: {self._round}, data_id: {self.data_id} for task: {task_to_perform}"
                )

                msg_bytes = pickle.dumps(payload)
                _send_bytes = len(msg_bytes)
                _payload_kind = "var_bad"
                logger.info(
                    f"[DEBUG] Payload size for {end}: {len(msg_bytes) / (1024 * 1024):.2f} MB"
                )

            # WS3-a network telemetry: one dispatch message onto the wire. Size is
            # the value already computed for the debug log above; no extra pickling.
            try:
                ev, f = build_comm(
                    direction="agg_to_trainer", size_bytes=_send_bytes, peer_id=str(end),
                    round_num=int(self._round), data_id=self.data_id,
                    iteration=self.iteration_per_data_id, payload_kind=_payload_kind,
                    n_tensors=len(payload),
                )
                telemetry.emit(ev, **f)
            except Exception as e:
                logger.debug(f"comm telemetry emit failed (agg send): {e}")

            channel.send(end, payload)
            logger.info(f"Sent weights to {end}")
            # self.invoke_gc()

        # Cohort-dispatch wall -> barrier_wait_s anchor (Stage A2). Sync barrier:
        # all `ends` dispatch in this one pass, so this marks the start of the
        # dispatch→last-grad window.
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
        # selection read (real-mode MQTT artifact, principle #8). No sim analog:
        # the sim orders by sct, not physical arrival, so this is pure wall
        # overhead there (≥28.7 s/run, Stage E) -- skip it. Real unchanged.
        if not self.simulated:
            logger.debug(f"Starting busy wait at time {time.time()}")
            time.sleep(0.1)
            logger.debug(f"Ended busy wait at time {time.time()}")
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

        self._curr_agg_version = (
            self._model_version,
            self.data_id,
            self.iteration_per_data_id,
        )
        logger.debug(
            f"Aggregator version state (model_version, data_id, iteration_id): {self._curr_agg_version}"
        )
        ends = channel.ends(
            state=VAL_CH_STATE_SEND,
            task_to_perform=task_to_perform,
            agg_version_state=self._curr_agg_version,
            trainer_version_states=self._trainer_state_dict,
        )
        logger.info(f"ends: {ends}")

        if not ends:
            logger.debug(
                f"No trainers found for tag {tag}, will "
                f"move to get() for fetch weights from trainers"
            )
            return

        _var_thr = getattr(self, "var_threshold", None)
        logger.info(
            f"[Distribute] Starting distribute for model_version={self._model_version}, "
            f"data_id={self.data_id}, iter={self.iteration_per_data_id}, "
            f"task={task_to_perform}, |ends|={len(ends)}, "
            f"var={self.var}, var_thr={_var_thr}, var_good_enough={self.var_good_enough}. "
            f"Stale trainers will get WEIGHTS; current trainers will get "
            f"{'WEIGHTS' if self.var_good_enough else 'VAR=bad'}."
        )

        payload_var_bad = None
        payload_weights = self._prepare_distribution_payload(task_to_perform, force_weights=True)
        if not self.var_good_enough:
            payload_var_bad = self._prepare_distribution_payload(task_to_perform, force_weights=False)

        self._update_state_after_payload_prepared()

        # WS3-a network telemetry (Experiment 4): serialized size of each payload
        # variant, computed ONCE per distribute (not per-end) — the async path has
        # no per-end debug-size log like the sync path, so we pickle the two shared
        # variants here and reuse per end below. Cheap (~trainable-param bytes, once).
        try:
            _bytes_weights = len(pickle.dumps(payload_weights)) if payload_weights is not None else 0
            _bytes_var_bad = len(pickle.dumps(payload_var_bad)) if payload_var_bad is not None else 0
        except Exception as e:
            _bytes_weights = _bytes_var_bad = 0
            logger.debug(f"comm telemetry size calc failed (async): {e}")

        # Sim-clock dispatch stamp (Batch 1): the async path arms the in-flight gate
        # (_sim_inflight_expected[end] = send vclock + a lower-bound budget) so the
        # reorder-buffer drain can't lap a trainer whose modeled completion is still
        # in the future. Inert in real mode.
        #   #13 step 4 (staggered): when on, each end's SEND vclock is popped from
        # the freed-slot FIFO (_pop_free_slot_ts) instead of the shared round
        # frontier, so a cohort's expected completions spread across the timeline
        # (see the commit-side stamp). Off/real ⇒ one shared _round_now injected
        # into the two payload variants (byte-identical to Batch 1).
        _round_now = self._vclock.now if self.simulated else None
        _staggered = self.simulated and getattr(self, "_sim_staggered_redispatch", False)
        if self.simulated and not _staggered:
            for _p in (payload_weights, payload_var_bad):
                if _p is not None:
                    _p[MessageType.SIM_SEND_TS] = _round_now

        _n_weights_sent = 0
        _n_var_bad_sent = 0
        for end in ends:
            trainer_version = self._trainer_last_model_version.get(end, -1)
            is_stale = (trainer_version != self._model_version)

            if self.var_good_enough or is_stale:
                payload = payload_weights
                _pk = "weights"          # WS3-a: kind set here (survives staggered rebuild below)
                _n_weights_sent += 1
                if not self.var_good_enough and is_stale:
                    logger.debug(
                        f"[Distribute] Trainer {end} stale "
                        f"(has v{trainer_version}, need v{self._model_version}); "
                        f"sending WEIGHTS even though var_good_enough=False."
                    )
            else:
                payload = payload_var_bad
                _pk = "var_bad"          # WS3-a
                _n_var_bad_sent += 1
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
                # R1 tripwire (K-D19): dispatching a trainer already outstanding
                # (in flight OR grad carried-but-uncommitted) IS the residence
                # violation -- surface it at the dispatch instant instead of only
                # reconstructing it offline from contributor_intervals.
                if end in self._sim_inflight_expected or self._sim_buffer.has(end):
                    logger.warning(
                        f"[SIM_R1_DISPATCH] end={end} re-dispatched while still "
                        f"outstanding (vclock={_round_now}) -- R1 residence violation"
                    )
                # #13 step 4: SEND vclock = the freed-slot stamp (staggered) else the
                # shared round frontier. _pop_free_slot_ts is FIFO + clamped <= now;
                # an empty queue (cold start / no held slot) falls back to _round_now.
                _sst = self._pop_free_slot_ts(_round_now) if _staggered else _round_now
                channel.set_end_property(end, PROP_SIM_SEND_TS, _sst)
                # Expected completion = SEND vclock + a lower-bound budget
                # (this end's own last-observed TRAINING_BUDGET_S, else the
                # running min) so the gate never laps a not-yet-committed trainer.
                _budget = self._sim_trainer_budget.get(end, self._sim_budget_min)
                self._sim_inflight_expected[end] = _sst + _budget
                # Staggered: this end's payload must carry its OWN SIM_SEND_TS, so
                # rebuild a shallow copy (weights shared by ref; small vs GPU cost).
                if _staggered and isinstance(payload, dict):
                    payload = dict(payload)
                    payload[MessageType.SIM_SEND_TS] = _sst
                # K-D27 (felix-parallel, asyncfl:1490): the instant a trainer is
                # dispatched it is in flight in virtual time -> add to the pending-
                # commit set so the selector's eligibility filter excludes it until
                # its grad COMMITS (discarded in _sim_recv_min_grad).
                self._sim_pending_commit.add(end)
                # NOTE: the async_oort re-pick triplet is stamped on grad RETURN
                # (in _process_single_trainer_message), NOT here at dispatch --
                # stamping the whole cohort at the current _curr_agg_version froze
                # the eligible pool before any commit could advance the version
                # (re-dispatch deadlock, §K-D17). An in-flight-but-not-returned
                # trainer is already guarded by its compute slot (selected_ends).
            # WS3-a: one dispatch message onto the wire (async path).
            try:
                _sz = _bytes_weights if _pk == "weights" else _bytes_var_bad
                ev, f = build_comm(
                    direction="agg_to_trainer", size_bytes=_sz, peer_id=str(end),
                    round_num=int(self._round), data_id=self.data_id,
                    iteration=self.iteration_per_data_id, payload_kind=_pk,
                    n_tensors=len(payload) if isinstance(payload, dict) else None,
                )
                telemetry.emit(ev, **f)
            except Exception as e:
                logger.debug(f"comm telemetry emit failed (agg async send): {e}")
            channel.send(end, payload)
            # #15 compute-truthful gate: stamp the wall time this end was actually
            # dispatched (weights OR VAR=bad), so the drain can tell a live straggler
            # from an idle-in-recv phantom. Sim-only; inert unless the gate flag is on.
            if self.simulated:
                self._sim_dispatch_wall[end] = time.time()
        logger.info(
            f"[Distribute] Done. Sent {_n_weights_sent} WEIGHTS + "
            f"{_n_var_bad_sent} VAR=bad payloads to {len(ends)} trainers "
            f"(model_version={self._model_version}, data_id={self.data_id})."
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
            # Mode-dependent clock (mirrors base syncfl `max_experiment_runtime_s`,
            # top_aggregator.py:1140): real -> WALL seconds; sim -> VIRTUAL seconds
            # (`vclock.now`). One budget then means "3600 wall-s of real work" AND
            # "3600 virtual-s of MODELED work" -- the matched-budget axis
            # convergence parity needs (real runs 3600s wall; sim runs until its
            # vclock reaches the SAME 3600s of modeled real-time). Until root #6
            # (vclock models real wall) lands, the sim vclock under-counts, so the
            # wall failsafe below will fire first -- which is itself the #6 signal.
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
            # Sim wall failsafe (Phase 4a, root S1): the sim does REAL GPU +
            # server eval, so its WALL legitimately exceeds the vclock budget --
            # a 1× ceiling truncated every sim before the vclock stop it was
            # paired with. Decoupled to `max_runtime_s × SIM_WALL_CEILING_FACTOR`
            # (a runaway OUTER safety), overridable by an explicit
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

                # task_get_heartbeat = Tasklet("heartbeat", self.get,
                # TAG_HEARTBEAT)
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

                task_get_heartbeat = Tasklet("heartbeat", self.get, TAG_HEARTBEAT)
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
                    # >> asyncfl_loop(task_put >> task_get_weights >>
                    # >> task_get_heartbeat
                    >> task_aggregate_grads_sync
                )
                >> c.tasklet("inform_end_of_training")
                # >> c.tasklet("load_data") c.tasklet("initialize")
                # >> task_get_heartbeat task_put_train c.tasklet("heartbeat") loop(
                # >> task_reset_agg_goal_vars # >> asyncfl_loop(task_put >>
                # >> task_get_weights >> c.tasklet("heartbeat") ) >>
                # >> asyncfl_loop(task_put_train >> task_put_eval >>
                # >> task_get_weights) >> c.tasklet("train") >>
                #     c.tasklet("evaluate") >> c.tasklet("analysis") >>
                #     c.tasklet("save_metrics") >> c.tasklet("inc_round") )
                # >> c.tasklet("inform_end_of_training") c.tasklet("save_params")
                # c.tasklet("save_model")
            )

    @classmethod
    def get_func_tags(cls) -> list[str]:
        """Return a list of function tags defined in the top level
        aggregator role."""
        return [TAG_DISTRIBUTE, TAG_AGGREGATE, TAG_HEARTBEAT]
