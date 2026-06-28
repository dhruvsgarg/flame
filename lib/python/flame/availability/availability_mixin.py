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
import math
import time
from pathlib import Path
from typing import Optional

import yaml

from flame import telemetry
from flame.availability.trace import load_trace, next_avail_after, state_at
from flame.config import TrainerAvailState
from flame.mode.message import MessageType
from flame.selector.properties import PROP_SIM_SEND_TS
from flame.telemetry.events import build_abandon_timeout, build_withheld_delivery

logger = logging.getLogger(__name__)

_METADATA_DIR = Path(__file__).resolve().parents[2] / "examples/_metadata"

_AVL_STATES = frozenset(
    {TrainerAvailState.AVL_TRAIN, TrainerAvailState.AVL_EVAL}
)

# Re-clock of the selector's wall-based abandon (SEND_TIMEOUT_WAIT_S) onto the
# vclock. Heuristic basis (PARITY.md / design §1): train takes ≤60s, so an
# in-flight trainer dispatched > 90 vclock-seconds ago is assumed offline.
_AVAIL_ABANDON_TIMEOUT_S = 90.0


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
        # C.2 send-time withhold ledgers (shared by asyncfl + oort commit loops):
        #   _sim_withheld_payload   end -> (orig_sct, (msg, metadata)) held update
        #   _sim_withheld_delivering end -> (orig_sct, delivery_ts) being re-injected
        # Initialized here (before the gate check) so the commit-loop helpers can
        # reference them unconditionally; they stay empty when the gate is off.
        if not hasattr(self, "_sim_withheld_payload"):
            self._sim_withheld_payload: dict = {}
        if not hasattr(self, "_sim_withheld_delivering"):
            self._sim_withheld_delivering: dict = {}

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
    # Delivery ledger (Stage C) — withheld update bookkeeping
    # ------------------------------------------------------------------
    #
    # Two ledgers, never one (Challenge 4 / invariant 1):
    #   * slot ledger    — in-flight count; freed here (free_stalled_slot).
    #   * delivery ledger — pending_withheld[end] = delivery_ts; the completed
    #                       update is HELD, not discarded, and committed later
    #                       (stale) by the live commit loop.
    # The same effect path serves all three triggers (90s vclock abandon, aware
    # boundary eviction, and the future avl_* message) — only the trigger differs.

    def compute_delivery_ts(self, end: str, sct: float) -> float:
        """When a withheld update from `end` becomes deliverable.

        The earliest time >= sct at which the trainer is AVL_* again: a
        completed-but-withheld update cannot deliver before it finished
        computing (`sct`) nor while the trainer is unreachable. If the trainer
        is already available at `sct` (it recovered before completing, or never
        went down) delivery is immediate (= sct); otherwise it waits for the
        next AVL_* window. Returns math.inf when the trace never recovers — the
        caller must guard (Challenge 10); the update is undeliverable in-window.
        """
        if self.trainer_event_dict is None:
            return float(sct)
        trace = self.trainer_event_dict.get(end)
        if not trace:
            return float(sct)
        ref = float(sct)
        if state_at(trace, ref) in _AVL_STATES:
            return ref
        nxt = next_avail_after(trace, ref)
        if nxt == math.inf:
            return math.inf
        return max(ref, float(nxt))

    def free_stalled_slot(
        self, channel, end: str, *, reason: str, sct: Optional[float] = None
    ) -> Optional[float]:
        """Free an in-flight slot and register its pending withheld delivery.

        Triggered by (a) the 90s vclock abandon for everyone (C.3) and (b) the
        aware boundary eviction for availability_aware baselines (Stage D); Stage H
        adds an avl_* message trigger without changing this effect logic — the
        abstraction exists for exactly that swap.

        1. Remove `end` from the selector slot ledger (selected_ends / all_selected).
        2. Compute delivery_ts = compute_delivery_ts(end, sct) (delivery ledger).
        3. Register pending_withheld[end] = delivery_ts.

        No-op (returns None) when the gate is off (trainer_event_dict is None),
        preserving byte-identity. Returns the registered delivery_ts otherwise.
        """
        if self.trainer_event_dict is None:
            logger.debug(
                f"[AVAIL] free_stalled_slot({end!r}) — gate off, no-op"
            )
            return None

        # 1. Slot ledger: release the concurrency slot so a replacement is
        #    selectable. Mirrors the abandon path's removal (random.py:215).
        #    Robust to BOTH selector shapes: asyncfl (fedbuff/async_oort/
        #    async_random) keep selected_ends as a dict keyed by requester plus an
        #    all_selected dict; oort/refl/feddance keep selected_ends as a flat set
        #    and have no all_selected. Drop `end` from whichever is present.
        self._avail_free_slot_ledger(channel, end)
        # Drop it from the asyncfl gate's in-flight tracker too (no-op on oort,
        # which tracks in-flight purely via selected_ends) — "free the slot" is one
        # effect across both stacks.
        self._avail_drop_inflight(end)

        # 2/3. Delivery ledger: hold the completed update until delivery_ts.
        if sct is None:
            sct = self._avail_now()
        delivery_ts = self.compute_delivery_ts(end, sct)
        self.pending_withheld[end] = delivery_ts
        logger.info(
            f"[AVAIL] free_stalled_slot({end!r}, reason={reason!r}) "
            f"sct={float(sct):.1f} delivery_ts={delivery_ts:.1f}"
        )
        return delivery_ts

    def withheld_held_ends(self, now: Optional[float] = None) -> set:
        """Ends whose withheld update has NOT yet reached its delivery_ts.

        These must stay out of the eligible pool (invariant 2: a still-down
        trainer is never re-selected) until vclock >= delivery_ts — the §4.5
        residence exclusion extended from `sct` to `delivery_ts`. Unioned into
        the unavailable list by the selection driver.
        """
        if not self.pending_withheld:
            return set()
        if now is None:
            now = self._avail_now()
        return {end for end, dts in self.pending_withheld.items() if dts > now}

    def ready_withheld(self, now: Optional[float] = None) -> list:
        """Withheld ends due for delivery (delivery_ts <= now), commit order.

        Ordered by (delivery_ts, end_id) so the late stale commits replay
        deterministically (Challenge 1/6 — past-dating is avoided because a
        withheld update commits at delivery_ts > sct, never at sct).
        """
        if not self.pending_withheld:
            return []
        if now is None:
            now = self._avail_now()
        due = [
            (end, dts) for end, dts in self.pending_withheld.items() if dts <= now
        ]
        due.sort(key=lambda kv: (kv[1], str(kv[0])))
        return due

    def commit_withheld(self, end: str) -> Optional[float]:
        """Pop `end` from the delivery ledger once its late update has committed."""
        return self.pending_withheld.pop(end, None)

    # ------------------------------------------------------------------
    # Slot-ledger helpers (robust to both selector shapes)
    # ------------------------------------------------------------------

    def _avail_free_slot_ledger(self, channel, end: str) -> None:
        """Drop `end` from the selector slot ledger + reset its end state.

        asyncfl selectors keep selected_ends as dict[requester -> set] + an
        all_selected dict; oort/refl/feddance keep selected_ends as a flat set
        and have no all_selected. Handle whichever is present.
        """
        sel = getattr(channel, "_selector", None)
        if sel is not None:
            all_selected = getattr(sel, "all_selected", None)
            if isinstance(all_selected, dict):
                all_selected.pop(end, None)
            selected_ends = getattr(sel, "selected_ends", None)
            if isinstance(selected_ends, dict):
                requester = getattr(sel, "requester", None)
                if requester in selected_ends:
                    selected_ends[requester].discard(end)
            elif isinstance(selected_ends, set):
                selected_ends.discard(end)
        if channel.has(end):
            from flame.end import KEY_END_STATE, VAL_END_STATE_NONE

            channel._ends[end].set_property(KEY_END_STATE, VAL_END_STATE_NONE)

    def _avail_drop_inflight(self, end: str) -> None:
        """Remove `end` from the asyncfl gate's in-flight tracker (no-op on oort)."""
        ie = getattr(self, "_sim_inflight_expected", None)
        if isinstance(ie, dict):
            ie.pop(end, None)

    def _avail_inflight_ends(self, channel) -> set:
        """In-flight (slot-ledger) ends, generic over both selector shapes."""
        sel = getattr(channel, "_selector", None)
        if sel is None:
            return set()
        se = getattr(sel, "selected_ends", None)
        if isinstance(se, dict):
            out: set = set()
            for v in se.values():
                out |= set(v)
            return out
        if isinstance(se, (set, list, tuple)):
            return set(se)
        return set()

    # ------------------------------------------------------------------
    # Shared commit-loop wiring (C.2 / C.3) — one logic for asyncfl + oort
    # ------------------------------------------------------------------
    #
    # Both stacks own a different sim commit loop (asyncfl _sim_recv_min's single
    # pop site; oort _sim_drain_buffer's generator pop loop), but the send-gate /
    # late-recommit / vclock-abandon EFFECT is identical, so it lives here and each
    # loop just calls in. Depends only on self._sim_buffer + the ledgers; never on
    # a stack-specific attribute (the gate tracker is reached via the guarded
    # _avail_drop_inflight hook). Off ⇒ pending_withheld stays empty ⇒ no-op.

    def _sim_reinject_ready_withheld(self) -> None:
        """C.2: re-inject withheld updates whose delivery_ts has arrived.

        Call before each pop. For every ledger entry due at the current vclock
        (ordered by (delivery_ts, end_id)), re-add the held payload to the reorder
        buffer keyed at delivery_ts so it commits stale through the normal path,
        then pop the ledger. A slot-only entry (the C.3 abandon registered a
        delivery_ts but the physical update never arrived) carries no payload —
        just drop the ledger entry. No-op when the ledger is empty.
        """
        if not getattr(self, "pending_withheld", None):
            return
        buf = getattr(self, "_sim_buffer", None)
        if buf is None:
            return
        for end, dts in self.ready_withheld(self._avail_now()):
            payload = self._sim_withheld_payload.pop(end, None)
            self.commit_withheld(end)
            if payload is None:
                continue  # slot-only registration; nothing to deliver
            orig_sct, msgmd = payload
            buf.add(end, float(dts), msgmd)
            self._sim_withheld_delivering[end] = (float(orig_sct), float(dts))
            logger.info(
                f"[WITHHELD_REINJECT] end={str(end)[-4:]} "
                f"sct={float(orig_sct):.1f} delivery_ts={float(dts):.1f}"
            )

    def _sim_withhold_if_unavail(self, channel, end, sct, msgmd) -> bool:
        """Per-update send-gate: True if this completed update is HELD, else False.

        The single-update primitive both commit loops share. When True the caller
        must skip the update — it has been removed from the buffer/slot accounting
        (held in the delivery ledger, delivered stale at delivery_ts). When False
        the update is committable now (trainer available at completion, or gate
        off). The caller applies any stack-specific gate (oort's still-computing
        carry-over) BEFORE this — a still-computing future-sct straggler has not
        reached its send-gate yet (Challenge 7), so carry-over wins.

        Returns False unchanged when sim_unavailability is off (compute_delivery_ts
        ⇒ sct), preserving byte-identity.
        """
        # Gate off (or a bare aggregator that never ran _init_availability): the
        # update is always committable and no ledger is touched. Keeps the shared
        # primitive safe to call from any partially-initialized commit loop.
        if getattr(self, "trainer_event_dict", None) is None:
            return False
        # invariant 1: never re-register / double-count an end whose slot was
        # already freed (C.3 abandon). Its arrived payload is stashed so the
        # reinject delivers it at the registered delivery_ts.
        if end in self.pending_withheld:
            self._sim_withheld_payload[end] = (float(sct), msgmd)
            self._avail_drop_inflight(end)
            return True
        dts = self.compute_delivery_ts(end, sct)
        if dts <= sct:
            return False  # available at completion (or gate off) — commit now
        # withhold: trainer is UN_AVL at sct; hold the completed update.
        if dts == math.inf:
            # trace never recovers in-window: the update is undeliverable.
            # free_stalled_slot still registers the ledger (end stays excluded);
            # drop the payload (acceptable v1 edge, Challenge 10).
            logger.info(
                f"[WITHHELD_LOST] end={str(end)[-4:]} sct={float(sct):.1f} "
                f"trace never recovers"
            )
        else:
            self._sim_withheld_payload[end] = (float(sct), msgmd)
        self.free_stalled_slot(
            channel, end, reason="send_gate_withhold", sct=float(sct)
        )
        return True

    def _sim_pop_committable(self, channel):
        """C.2 (asyncfl): pop the smallest buffered update that is committable now.

        Loops over _sim_withhold_if_unavail, skipping send-gated updates and
        popping the next-smallest committable one. Returns (end, sct, (msg,
        metadata)) or None (buffer drained). The vclock is NOT advanced for a
        withheld pop — only the committed update drives the clock.

        Off (or trainer available at sct) ⇒ a single pop identical to the prior
        pop_min(); reinject a no-op. Byte-identical when sim_unavailability is off.
        """
        buf = self._sim_buffer
        while True:
            popped = buf.pop_min()
            if popped is None:
                return None
            end, sct, msgmd = popped
            if self._sim_withhold_if_unavail(channel, end, sct, msgmd):
                continue
            return popped

    def _sim_take_withheld_delivering(self, end: str) -> Optional[tuple]:
        """Pop (orig_sct, delivery_ts) if `end`'s commit is a late withheld delivery.

        The commit body calls this to recognize a re-injected stale delivery (vs a
        fresh/straggler commit) so it can emit the withheld_delivery rung and tag
        the "withheld" past-dating bucket. Returns None for an ordinary commit.
        """
        d = getattr(self, "_sim_withheld_delivering", None)
        if not d:
            return None
        return d.pop(end, None)

    def _emit_withheld_delivery(self, end, msg, orig_sct, delivery_ts) -> None:
        """Emit the withheld_delivery rung for a late stale commit (best-effort)."""
        if not telemetry.is_enabled():
            return
        mv = msg.get(MessageType.MODEL_VERSION) if isinstance(msg, dict) else None
        ev, f = build_withheld_delivery(
            round_num=self._round, end_id=end,
            sct=float(orig_sct), delivery_ts=float(delivery_ts),
            staleness=(self._round - int(mv)) if mv is not None else None,
            accepted=True, time_mode="sim",
        )
        telemetry.emit(ev, **f)

    def _sim_abandon_stalled(self, channel) -> None:
        """C.3: free in-flight slots stalled past the 90s vclock deadline.

        Re-clocks the selector's wall-based abandon (inert in sim) onto the vclock.
        A trainer dispatched > 90 vclock-seconds ago whose update has neither
        buffered nor committed is assumed offline: free its slot (a replacement
        becomes selectable) and register the delivery ledger. If its update later
        physically arrives it is reconciled by _sim_pop_committable (payload stash)
        / _sim_reinject_ready_withheld.

        Slot ledger ⊥ delivery ledger (Challenge 4): the slot is freed here, the
        completed update is NOT discarded — it still commits (stale) and is
        accept/reject-gated by the baseline's existing staleness rule. No-op when
        the gate is off (trainer_event_dict is None) ⇒ byte-identical.
        """
        if getattr(self, "trainer_event_dict", None) is None:
            return
        inflight = self._avail_inflight_ends(channel)
        if not inflight:
            return
        now = self._avail_now()
        buf = getattr(self, "_sim_buffer", None)
        committed = getattr(self, "_sim_committed", set())
        for end in list(inflight):
            if buf is not None and buf.has(end):
                continue  # already arrived — not stalled
            if end in committed or end in self.pending_withheld:
                continue  # invariant 1: already committed / abandoned
            sst = channel.get_end_property(end, PROP_SIM_SEND_TS)
            if sst is None:
                continue
            if now - float(sst) <= _AVAIL_ABANDON_TIMEOUT_S:
                continue
            self.free_stalled_slot(
                channel, end, reason="abandon_90s_vclock", sct=now
            )
            logger.info(
                f"[ABANDON_90S] end={str(end)[-4:]} sim_send_ts={float(sst):.1f} "
                f"vclock={now:.1f} age={now - float(sst):.1f}s"
            )
            if telemetry.is_enabled():
                ev, f = build_abandon_timeout(
                    round_num=getattr(self, "_round", -1), end_id=end,
                    sim_send_ts=float(sst), vclock_now=now, time_mode="sim",
                    reason="abandon_90s_vclock",
                )
                telemetry.emit(ev, **f)

    def _sim_evict_unavail_inflight(self, channel) -> None:
        """D.1: Proactively free in-flight slots for trainers now showing UN_AVL.

        For availability_aware baselines the oracular trace read at the selection
        boundary is authoritative — no need to wait for the 90s vclock deadline
        (C.3). A trainer that transitioned to UN_AVL since it was dispatched has
        its slot freed immediately so a replacement is selectable this round.

        Same effect as free_stalled_slot (slot ledger freed + delivery ledger
        registered) — only the trigger differs from C.3. No-op when the gate is
        off (trainer_event_dict is None) or _availability_aware is False (unaware
        baselines stay on the C.3 90s path).
        """
        if not getattr(self, "_availability_aware", False):
            return
        if getattr(self, "trainer_event_dict", None) is None:
            return
        inflight = self._avail_inflight_ends(channel)
        if not inflight:
            return
        now = self._avail_now()
        buf = getattr(self, "_sim_buffer", None)
        committed = getattr(self, "_sim_committed", set())
        for end in list(inflight):
            if buf is not None and buf.has(end):
                continue  # update already arrived in buffer — not stalled
            if end in committed or end in self.pending_withheld:
                continue  # invariant 1: already committed / registered
            trace = self.trainer_event_dict.get(end)
            if not trace:
                continue
            if state_at(trace, now) != TrainerAvailState.UN_AVL:
                continue  # still available — leave the slot
            self.free_stalled_slot(
                channel, end, reason="aware_boundary_eviction", sct=now
            )
            logger.info(
                f"[AWARE_EVICT] end={str(end)[-4:]} vclock={now:.1f} "
                f"state=UN_AVL — proactive boundary eviction"
            )
            if telemetry.is_enabled():
                sst = channel.get_end_property(end, PROP_SIM_SEND_TS)
                ev, f = build_abandon_timeout(
                    round_num=getattr(self, "_round", -1), end_id=end,
                    sim_send_ts=float(sst) if sst is not None else now,
                    vclock_now=now, time_mode="sim",
                    reason="aware_boundary_eviction",
                )
                telemetry.emit(ev, **f)
