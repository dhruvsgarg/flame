# Copyright 2024 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""AvailabilityMixin — library-level oracular availability for all aggregators.

Mixed into flame/mode/horizontal/syncfl/top_aggregator.TopAggregator (the
common ancestor of the oort, asyncfl, syncfl, and fwdllm stacks). All four
aggregators inherit read_trainer_unavailability, get_curr_unavail_trainers,
_avail_now, _init_availability, and the dormant free_stalled_slot hook.

This consolidates the three duplicated read_trainer_unavailability copies that
previously lived in main_oort_sync_agg.py:173, main_asyncfl_agg.py:158, and
fwdllm_aggregator.py:481, and the duplicated get_curr_unavail_trainers in
main_oort_sync_agg.py:298 (which used wall-time in sim — now corrected to
use _vclock.now via _avail_now()).

Stage A: substrate only — no behavior change when sim_unavailability=False
(the default). _init_availability sets trainer_event_dict=None when the gate
is off, preserving byte-identical output on syn_0 runs.
Stage C will activate free_stalled_slot and wire the pending_withheld ledger.
"""

import logging
import time
from pathlib import Path
from typing import Optional

import yaml

from flame.availability.trace import load_trace, state_at
from flame.config import TrainerAvailState

logger = logging.getLogger(__name__)

_METADATA_DIR = Path(__file__).resolve().parents[2] / "examples/_metadata"


class AvailabilityMixin:
    """Oracular availability substrate for TopAggregator subclasses.

    Depends on attributes set by syncfl TopAggregator.internal_init():
        self.simulated       (bool)
        self._vclock         (VirtualClock)
        self.agg_start_time_ts  (float, epoch seconds)
        self.config          (Config)
    These are all present before initialize() runs, so _init_availability
    may be called from internal_init() or initialize().
    """

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _init_availability(self, config) -> None:
        """Populate trainer_event_dict from config; no-op when gate is off.

        Master gate: sim_unavailability (default False). Also accepts the
        legacy track_trainer_avail["enabled"]=True path so existing configs
        keep working without adding the new flag.

        Sets:
            self.trainer_event_dict  — dict[task_id → SortedDict] or None
            self._availability_aware — bool (Stage D proactive eviction)
            self.pending_withheld    — dict[end → delivery_ts] (Stage C)
        """
        hp = config.hyperparameters
        self.trainer_event_dict: Optional[dict] = None
        self._availability_aware: bool = bool(
            getattr(hp, "availability_aware", False)
        )
        self.pending_withheld: dict = {}

        sim_unavail = bool(getattr(hp, "sim_unavailability", False))
        track = getattr(hp, "track_trainer_avail", None) or {}
        legacy_enabled = str(track.get("enabled", "False")).strip().lower() == "true"

        if not sim_unavail and not legacy_enabled:
            return

        if sim_unavail:
            client_notify = getattr(hp, "client_notify", None) or {}
            trace_name = (
                client_notify.get("trace")
                or getattr(hp, "availability_trace", None)
                or track.get("trace")
            )
        else:
            # Legacy path: only activate for ORACULAR type
            if str(track.get("type", "")).upper() != "ORACULAR":
                return
            trace_name = track.get("trace")

        if not trace_name:
            logger.warning("[AVAIL] availability enabled but no trace name configured")
            return

        trace_dir = getattr(hp, "availability_trace_dir", None)
        self.trainer_event_dict = self.read_trainer_unavailability(
            trace=trace_name, base_dir=trace_dir
        )

    # ------------------------------------------------------------------
    # Time source — single call site for "what time is it on the trace timeline"
    # ------------------------------------------------------------------

    def _avail_now(self) -> float:
        """Current time on the availability timeline.

        sim:  self._vclock.now  (sim-seconds since experiment start)
        real: wall-elapsed since agg_start_time_ts

        Never uses a per-trainer frozen clock (_sim_send_ts) — see PARITY.md §S.dur
        and the REFL HIGH-1 frozen-clock root cause.
        """
        if getattr(self, "simulated", False):
            return float(self._vclock.now)
        return time.time() - self.agg_start_time_ts

    # ------------------------------------------------------------------
    # Trace loading (replaces three duplicated copies)
    # ------------------------------------------------------------------

    def read_trainer_unavailability(
        self,
        trace: Optional[str] = None,
        base_dir: Optional[str] = None,
    ) -> Optional[dict]:
        """Build task_id → SortedDict[ts_s → state_str] from the canonical store.

        Reads examples/_metadata/trainer_registry.yaml once; individual trace
        SortedDicts are built via load_trace() which caches the raw YAML.
        Returns None on fatal errors (caller treats None as gate-off).
        """
        if not trace:
            return None

        registry_path = _METADATA_DIR / "trainer_registry.yaml"
        try:
            with open(registry_path) as f:
                registry = yaml.safe_load(f)["trainers"]
        except FileNotFoundError:
            logger.error(f"[AVAIL] trainer registry not found: {registry_path}")
            return None

        trainer_events_dict: dict = {}
        errors = 0
        for tk, meta in registry.items():
            task_id = meta["task_id"]
            try:
                trainer_events_dict[task_id] = load_trace(
                    trace, tk, base_dir=base_dir
                )
            except (KeyError, FileNotFoundError) as exc:
                logger.warning(f"[AVAIL] skipping {tk}: {exc}")
                errors += 1

        if errors:
            logger.warning(
                f"[AVAIL] {errors}/{len(registry)} trainers had missing trace data"
            )
        logger.info(
            f"[AVAIL] loaded {len(trainer_events_dict)} trainer traces "
            f"(trace={trace!r})"
        )
        return trainer_events_dict or None

    # ------------------------------------------------------------------
    # Oracular selection gate
    # ------------------------------------------------------------------

    def get_curr_unavail_trainers(self) -> list:
        """Trainers in UN_AVL state per oracular trace read at _avail_now().

        Returns [] when trainer_event_dict is None (gate off) — byte-identical
        to today's behavior when sim_unavailability=False.

        Replaces the inlined bisect_right loop formerly duplicated in:
            syncfl/top_aggregator.py:1163
            main_oort_sync_agg.py:298  (wall-time bug now corrected)
        """
        if self.trainer_event_dict is None:
            return []

        now = self._avail_now()
        unavail = [
            tid
            for tid, trace in self.trainer_event_dict.items()
            if state_at(trace, now) == TrainerAvailState.UN_AVL
        ]
        logger.info(
            f"[ORACULAR] unavail={len(unavail)}/{len(self.trainer_event_dict)} "
            f"@ t={now:.1f}s"
        )
        return unavail

    # ------------------------------------------------------------------
    # Dormant eviction hook (Stage C wires this up)
    # ------------------------------------------------------------------

    def free_stalled_slot(self, channel, end: str, *, reason: str) -> None:
        """Free an in-flight slot and register its pending withheld delivery.

        Stage A: built but dormant. No-op until Stage C activates it for the
        90s vclock abandon path and Stage D activates it for aware proactive
        boundary eviction. Stage H will also trigger it via an avl_* message
        without changing the effect logic (the abstraction exists for exactly
        this swap).

        When active (Stage C+):
            1. Remove `end` from selected_ends / in-flight (slot ledger).
            2. Compute delivery_ts = max(sct, next_avail_after(trace, now)).
            3. Register pending_withheld[end] = delivery_ts (delivery ledger).
        The two ledgers are independent — Challenge 4 / invariant 1.
        """
        logger.debug(
            f"[AVAIL] free_stalled_slot({end!r}, reason={reason!r}) — dormant (Stage A)"
        )
