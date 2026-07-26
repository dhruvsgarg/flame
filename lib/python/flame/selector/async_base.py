# Copyright 2026 Cisco Systems, Inc. and its affiliates
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
"""AsyncSelectorBase: the send/recv dispatch mechanism shared by every
asyncfl selector (fedbuff, async_random, async_oort).

Extracted from `async_oort.py`, the only copy hardened against real<->sim
divergence. These ~600 lines existed three times over with silent drift:
`fedbuff.py` lacked the version_key re-pick guard, the virtual-clock timeout,
the R1 pending-commit guard and the availability filter -- which is why
`fedbuff_it_*` re-picked the same trainer on 34% of real commits vs 0.9% in sim.

BASE owns the mechanism (concurrency accounting, in-flight bookkeeping,
abandon-timeout reclaim, eligibility, the `_cleanup_*` family, telemetry).
SUBCLASS owns the policy: one method, `_choose(candidates, k, ctx)`.
"""

import logging
import time
from random import Random as _StdRandom

from flame.channel import (
    KEY_CH_SELECT_REQUESTER,
    KEY_CH_STATE,
    VAL_CH_STATE_RECV,
    VAL_CH_STATE_SEND,
)
from flame.common.typing import Scalar
from flame.config import TrainerAvailState
from flame.end import KEY_END_STATE, VAL_END_STATE_NONE, VAL_END_STATE_RECVD, End
from flame.selector import AbstractSelector, SelectorReturnType
from flame.selector.properties import PROP_AVL_STATE

logger = logging.getLogger(__name__)

SEND_TIMEOUT_WAIT_S = 90


class SelectContext:
    """Everything `_choose` may need beyond the candidate pool itself."""

    __slots__ = (
        "task_to_perform",
        "model_version",
        "agg_version_key",
        "trainer_version_keys",
        "channel_props",
        "connected_ends",
        "trainer_unavail_list",
        "concurrency",
    )

    def __init__(self, **kw):
        for slot in self.__slots__:
            setattr(self, slot, kw.get(slot))


class AsyncSelectorBase(AbstractSelector):
    """Send/recv concurrency mechanism for asyncfl selectors."""

    # Domain-separates `_keyed_topk` draws so two selectors with the same seed
    # don't produce identical orderings.
    CHOOSE_SALT = "async_base"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # #1c: abandon-timeout clock -- set per-select() from
        # channel_props["vclock_now"] (sim) or left None (real -> wall).
        self._sim_now_s = None
        self.round = 0

        try:
            self.c = kwargs["c"]
        except KeyError:
            raise KeyError("c (concurrency level) is not specified in config")

        try:
            self.agg_goal = kwargs["aggGoal"]
        except KeyError:
            raise KeyError("aggGoal is not specified in config")
        if self.agg_goal < 0:
            self.agg_goal = 1

        # Tracking selected ends so a trainer participates once per round.
        # `selected_ends` is {requester: set(end_id)} for async selectors --
        # not the bare set the sync ones use.
        self.all_selected = dict()
        self.selected_ends = dict()
        self.ordered_updates_recv_ends = list()

        self.track_trainer_timeouts = dict()
        self.track_selected_trainers_which_left = dict()

        # In-flight abandon timeout: a bare 90s evicted genuinely-busy (not
        # dead) fwdllm trainers, whose forward-grad rounds run longer than the
        # CNN/speech rounds this was tuned for. Now a workload knob
        # (hyperparameters.send_timeout_wait_s, threaded in by
        # channel_manager.py); defaults to the original constant.
        self.send_timeout_wait_s = kwargs.get(
            "send_timeout_wait_s", SEND_TIMEOUT_WAIT_S
        )

        self.check_three_state_avl = True  # back-compat; see _task_eligible_states
        self._task_eligible_states = self._validate_task_eligible_states(
            kwargs.get("task_eligible_states")
        )
        logger.info(
            f"[TaskEligibility] task_eligible_states = {self._task_eligible_states}"
        )

    @staticmethod
    def _validate_task_eligible_states(raw) -> dict:
        """Task -> eligible avl_states, defaulting to classic FL semantics."""
        if raw is None:
            raw = {
                "train": [TrainerAvailState.AVL_TRAIN.value],
                "eval": [
                    TrainerAvailState.AVL_EVAL.value,
                    TrainerAvailState.AVL_TRAIN.value,
                ],
            }
        valid = {v.value for v in TrainerAvailState}
        for task_name, states in raw.items():
            for s in states:
                if s not in valid:
                    raise ValueError(
                        f"task_eligible_states['{task_name}'] contains unknown "
                        f"state '{s}'. Valid states: {sorted(valid)}"
                    )
        return raw

    # ------------------------------------------------------------------ hooks

    def _choose(self, candidates: dict[str, End], k: int, ctx: SelectContext) -> list:
        """Pick up to `k` ends from `candidates`. Subclass policy goes here.

        `candidates` is already filtered for availability, in-flight status,
        the pending-commit guard and the version_key re-pick guard, and `k` is
        already clamped to the free concurrency slots -- so an implementation
        only has to rank and cut.
        """
        raise NotImplementedError

    def _concurrency_for_task(
        self, task_to_perform: str, num_ends: int, effective_c: int
    ) -> int:
        """Slots this task may occupy. Overridden where eval has its own pool."""
        return min(num_ends, effective_c)

    def _pre_choose(self, ctx: SelectContext) -> None:
        """Once-per-dispatch hook before candidate selection (e.g. Oort's pacer)."""

    def _selection_extra(self, ctx: SelectContext, results) -> dict:
        """Extra fields for the `selection` telemetry event."""
        return {}

    def _per_trainer_selection_extra(self, ends: dict[str, End]) -> dict:
        """Per-trainer fields for the `selection` telemetry event."""
        return None

    # ----------------------------------------------------------------- select

    def select(
        self,
        ends: dict[str, End],
        channel_props: dict[str, Scalar],
        trainer_unavail_list: list = None,
        task_to_perform: str = "train",
        **kwargs,
    ) -> SelectorReturnType:
        """Dispatch on channel state.

        SEND picks new trainers to send the model to; RECV reports the
        in-flight set to receive from. An end in `selected_ends` has already
        been sent to, so it is excluded from SEND and included in RECV.
        """
        ctx = SelectContext(
            task_to_perform=task_to_perform,
            agg_version_key=kwargs.get("agg_version_key"),
            trainer_version_keys=kwargs.get("trainer_version_keys"),
            channel_props=channel_props,
            connected_ends=ends,
            trainer_unavail_list=trainer_unavail_list,
        )

        if self.enforce_min_start(len(ends)):
            return {}

        # dynamic_c pushed by DynamicKCController, else the static config value.
        effective_c = int(channel_props.get("dynamic_c", self.c))
        concurrency = self._concurrency_for_task(
            task_to_perform, len(ends), effective_c
        )
        ctx.concurrency = concurrency
        logger.info(
            f"Task: {task_to_perform}, len(ends): {len(ends)}, c: {self.c}, "
            f"effective_c: {effective_c}, chosen concurrency: {concurrency}"
        )
        if concurrency == 0:
            logger.debug("no concurrency available")
            return {}

        if KEY_CH_STATE not in channel_props:
            raise KeyError(f"channel property doesn't have {KEY_CH_STATE}")

        self.requester = channel_props[KEY_CH_SELECT_REQUESTER]
        if self.requester not in self.selected_ends:
            self.selected_ends[self.requester] = set()

        # #1c: the abandon-timeout must run on the same clock the trainer
        # commits on -- virtual in sim, wall in real. Stashed so the dispatch
        # STAMP and the CHECK agree; None in real -> time.time().
        self._sim_now_s = channel_props.get("vclock_now")

        eligible_ends = ends
        if trainer_unavail_list:
            eligible_ends = {
                end_id: end
                for end_id, end in ends.items()
                if end_id not in trainer_unavail_list
            }

        state = channel_props[KEY_CH_STATE]
        if state == VAL_CH_STATE_SEND:
            results = self._handle_send_state(eligible_ends, concurrency, ctx)
            if results:
                self.record_selection_stats(ends, results.keys(), task_to_perform)
                self.maybe_log_stat_summary()
            self.emit_selection(
                channel_props.get("round", 0),
                task_to_perform,
                ends,
                eligible_ends.keys(),
                list(results.keys()),
                per_trainer_extra=self._per_trainer_selection_extra(ends),
                extra={
                    "concurrency": concurrency,
                    "effective_c": effective_c,
                    "requester": self.requester,
                    "vclock_now": channel_props.get("vclock_now"),
                    **self._selection_extra(ctx, results),
                },
            )
        elif state == VAL_CH_STATE_RECV:
            results = self._handle_recv_state(ends, concurrency)
        else:
            raise ValueError(f"unkown channel state: {state}")

        logger.debug(f"channel state: {state}, results: {results}")
        return results

    # ------------------------------------------------------------ send / recv

    def _abandon_clock_now(self) -> float:
        """Clock for the in-flight abandon-timeout: the virtual clock in sim
        (vclock_now, stashed per-select), physical wall in real. Keeping the
        STAMP (all_selected[end]) and the CHECK on the same clock makes the
        timeout mean virtual seconds in sim, so a slow sim no longer evicts a
        still-outstanding trainer from the re-pick guard (#1c)."""
        sim_now = getattr(self, "_sim_now_s", None)
        return sim_now if sim_now is not None else time.time()

    def _reclaim_timed_out_ends(self, selected_ends: set) -> None:
        """Free ends that never returned within `send_timeout_wait_s`.

        Must run BEFORE `extra` is computed and before any slot-exhaustion
        early return -- a reclaim gated behind the very exhaustion it relieves
        can never fire once concurrency saturates. Must free `selected_ends`
        too, not just `all_selected`: `extra` counts the former, so touching
        only the latter leaves the slot occupied forever.
        """
        # getattr-guarded: test doubles built via __new__ skip __init__.
        timeout_s = getattr(self, "send_timeout_wait_s", SEND_TIMEOUT_WAIT_S)
        for end in list(self.all_selected.keys()):
            now_s = self._abandon_clock_now()
            if end not in self.all_selected:
                # re-check: another thread may have cleared it
                continue
            sent_at = self.all_selected[end]
            if sent_at >= (now_s - timeout_s) or end in self.ordered_updates_recv_ends:
                continue

            self.track_trainer_timeouts[end] = (
                self.track_trainer_timeouts.get(end, 0) + 1
            )
            total = sum(self.track_trainer_timeouts.values())
            logger.info(
                f"Removing end {end} from all_selected; no update in {timeout_s} "
                f"(last send stamp {sent_at}). Timeout count for this end: "
                f"{self.track_trainer_timeouts[end]}, total timeouts: {total}, "
                f"total wait: {total * timeout_s}s"
            )
            del self.all_selected[end]
            selected_ends.discard(end)
            # R1: also drop it from the pending-commit set (sim's
            # `_sim_pending_commit` or real's `_per_agg_trainer_list`), or it
            # stays un-re-pickable forever despite the reclaim above.
            pending_ref = getattr(self, "_agg_pending_commit_ref", None)
            if pending_ref is not None:
                pending_ref.discard(end)

    def _drop_disconnected_selections(
        self, selected_ends: set, connected_ends: dict[str, End]
    ) -> None:
        """Free slots held by ends that left the channel.

        Challenge 13: membership is checked against the CONNECTED pool, not the
        availability-filtered one -- an in-flight trainer that merely went
        UN_AVL (or is the wrong task type) is absent from the filtered pool but
        still connected, and forgetting it makes the aggregator stop waiting.
        An empty eligible pool would otherwise wipe every shared selection.
        """
        for end_id in list(selected_ends):
            if end_id not in connected_ends:
                logger.info(
                    f"Removing invalid prior selection {end_id}: no longer "
                    f"connected. Left in all_selected -- it may already have "
                    f"participated in this round."
                )
                selected_ends.remove(end_id)

    def _eligible_candidates(
        self, ends: dict[str, End], ctx: SelectContext
    ) -> dict[str, End]:
        """Ends that may be dispatched to right now.

        Excludes, in order: already in-flight (`all_selected`), awaiting commit
        (`_agg_pending_commit_ref`, the R1 guard -- `all_selected` alone is not
        enough once a buffered return releases the channel slot early),
        availability-ineligible for this task, and finally any trainer that
        already contributed to this same `agg_version_key`.
        """
        pending = getattr(self, "_agg_pending_commit_ref", None) or set()
        eligible_states = self._task_eligible_states.get(ctx.task_to_perform, [])

        candidates = {}
        n_ineligible = 0
        for end_id, end in ends.items():
            if end_id in self.all_selected or end_id in pending:
                continue
            avl_state = end.get_property(PROP_AVL_STATE)
            # None avl_state == no heartbeat state set -> always eligible,
            # matching trainers without availability tracking.
            if avl_state is not None and avl_state not in eligible_states:
                n_ineligible += 1
                continue
            if not self._task_extra_eligible(end_id, end, ctx):
                n_ineligible += 1
                continue
            candidates[end_id] = end

        if ctx.agg_version_key is not None and ctx.trainer_version_keys is not None:
            # Drop trainers that already contributed to this version_key.
            candidates = {
                end_id: end
                for end_id, end in candidates.items()
                if ctx.trainer_version_keys.get(end_id) != ctx.agg_version_key
            }

        logger.info(
            f"Eligible candidates: {len(candidates)} of {len(ends)} "
            f"(ineligible by state: {n_ineligible})"
        )
        return candidates

    def _task_extra_eligible(
        self, end_id: str, end: End, ctx: SelectContext
    ) -> bool:
        """Extra per-task eligibility beyond avl_state (e.g. eval staleness)."""
        return True

    def _handle_send_state(
        self, ends: dict[str, End], concurrency: int, ctx: SelectContext
    ) -> SelectorReturnType:
        selected_ends = self.selected_ends[self.requester]

        self._reclaim_timed_out_ends(selected_ends)
        self._drop_disconnected_selections(
            selected_ends, ctx.connected_ends if ctx.connected_ends is not None else ends
        )

        # Cooling (committed, not-yet-redispatched) ends hold a slot so the
        # idle pool can't refill it -- else the redispatch gap is inert.
        cooling = int((ctx.channel_props or {}).get("sim_cooling_count", 0))
        extra = max(0, concurrency - len(selected_ends) - cooling)
        logger.debug(
            f"concurrency={concurrency}, in_flight={len(selected_ends)}, "
            f"cooling={cooling}, extra={extra}"
        )
        if extra == 0:
            return {}

        ctx.model_version = self._model_version_for(ctx)
        self._pre_choose(ctx)

        candidates = self._eligible_candidates(ends, ctx)
        if not candidates:
            logger.info("no eligible candidates; returning empty selection")
            return {}

        # `extra` counts free SLOTS; `candidates` bounds what is actually
        # pickable. Dispatch the smaller of the two.
        k = min(extra, len(candidates))
        chosen = self._choose(candidates, k, ctx)
        candidates_dict = {end_id: None for end_id in chosen}

        self.process_chosen_candidate_dict(candidates_dict, selected_ends)
        logger.info(f"handle_send_state dispatching: {list(candidates_dict)}")
        return candidates_dict

    def _model_version_for(self, ctx: SelectContext) -> int:
        """Step identity for the dispatch: version_key's model_version."""
        key = ctx.agg_version_key
        if key is not None and key[0] is not None:
            return key[0]
        props = ctx.channel_props or {}
        return props["round"] if "round" in props else self.round

    def _handle_recv_state(
        self, ends: dict[str, End], concurrency: int
    ) -> SelectorReturnType:
        """Read-only over `selected_ends`: reports who is outstanding, minus
        replies received. Never assigns new selections -- that's
        `_handle_send_state`'s job; a prior version that resampled here raced
        send-state dispatch and could deadlock. Returns {} if empty; the next
        send-state tick dispatches normally.

        Bootstrap exception: if `all_selected` is empty too (nothing ever
        dispatched, so nothing to race), pick directly. Needed by callers
        whose first-ever call is RECV, not SEND -- e.g. a trainer's own 1:1
        channel, which only reaches SEND on upload -- else RECV permanently
        returns {} and `channel.one_end` crashes on it.
        """
        selected_ends = self.selected_ends[self.requester]

        # Drop ends already heard from, so we don't wait on them again.
        for end_id in list(selected_ends):
            if end_id not in ends:
                logger.debug(f"end {end_id} no longer in ends; leaving in-flight")
                continue
            if ends[end_id].get_property(KEY_END_STATE) == VAL_END_STATE_RECVD:
                selected_ends.remove(end_id)
                logger.debug(f"Removed {end_id} from selected_ends: already RECVD")

        if not selected_ends and not self.all_selected and ends:
            bootstrap = sorted(ends)[:concurrency]
            selected_ends = set(bootstrap)
            self.selected_ends[self.requester] = selected_ends
            stamp = self._abandon_clock_now()
            for end_id in bootstrap:
                self.all_selected[end_id] = stamp
            logger.info(
                f"[RecvBootstrap] first-ever recv for requester={self.requester}, "
                f"nothing in flight yet; picked {bootstrap}"
            )

        # sorted(): process-stable order so real and sim agree.
        return {key: None for key in sorted(selected_ends)}

    def process_chosen_candidate_dict(
        self, candidates_dict: dict[str, None], selected_ends: set
    ) -> None:
        """Commit a dispatch decision into the in-flight bookkeeping."""
        candidates = list(candidates_dict.keys())
        self.selected_ends[self.requester] = selected_ends.union(candidates)
        for candidate_end in candidates:
            # {end: dispatch stamp}, on the abandon clock (#1c).
            self.all_selected[candidate_end] = self._abandon_clock_now()
        logger.debug(
            f"selected_ends now {self.selected_ends[self.requester]}, "
            f"all_selected now {sorted(self.all_selected)}"
        )

    def _keyed_topk(self, candidate_ids, k: int, agg_version_key, salt: str) -> list:
        """Order-sample top-k: each id's rank key depends only on its own
        (seed, salt, agg_version_key, id), never on pool membership/size/call
        order. Replaces index-based random.sample()/np.random.choice(), where
        one trainer's incidental presence/absence shifts every other
        candidate's draw and permanently desyncs later calls.

        Seed material is a str, not a raw tuple -- Random() hashes non-str/
        int/bytes seeds, and str hash() is PYTHONHASHSEED-randomized per
        process, which would silently break real/sim parity.
        """
        def _key(c: str) -> float:
            return _StdRandom(f"{self._seed}|{salt}|{agg_version_key}|{c}").random()

        return sorted(candidate_ids, key=_key, reverse=True)[:k]

    # --------------------------------------------------------------- cleanup

    def _free_end(self, end_id: str, selected_ends: set) -> None:
        """Drop one end from both in-flight structures."""
        selected_ends.discard(end_id)
        self.all_selected.pop(end_id, None)

    def _cleanup_recvd_ends(self, ends: dict[str, End]) -> None:
        """Free ends whose update was received, making them selectable again.

        Sets end state to NONE, so for the aggregator this must run only after
        aggregation succeeds on meeting agg_goal -- otherwise an end can be
        re-sampled inside the same round.

        Drains ALL received ends: capping at agg_goal permanently orphaned the
        excess each cycle and deadlocked once K changed dynamically.
        """
        if not self.ordered_updates_recv_ends:
            logger.debug("no recvd ends to clean up")
            return

        selected_ends = self.selected_ends[self.requester]
        ends_to_remove, self.ordered_updates_recv_ends = (
            self.ordered_updates_recv_ends,
            [],
        )
        logger.debug(f"cleaning up recvd ends: {ends_to_remove}")

        for end_id in ends_to_remove:
            if end_id not in ends:
                # Connection lost. Not guaranteed to still be in selected_ends
                # -- it may have disconnected/rejoined mid-round.
                logger.debug(f"end {end_id} gone from ends; freeing it")
                self._free_end(end_id, selected_ends)
                continue

            state = ends[end_id].get_property(KEY_END_STATE)
            if state == VAL_END_STATE_RECVD:
                ends[end_id].set_property(KEY_END_STATE, VAL_END_STATE_NONE)
                self._free_end(end_id, selected_ends)
            elif state == VAL_END_STATE_NONE:
                # May have left and rejoined; free it if it was still tracked.
                self._free_end(end_id, selected_ends)
            else:
                logger.debug(f"end {end_id} in state {state}; leaving alone")

    def _cleanup_provided_ends(
        self, ends_to_cleanup: dict[str, End], ends: dict[str, End]
    ) -> None:
        """Free specific ends so they can be sampled again -- used to reject
        stale updates in FwdLLM (async)."""
        selected_ends = self.selected_ends.get(self.requester, set())
        for end_id in ends_to_cleanup:
            state = ends[end_id].get_property(KEY_END_STATE)
            if state == VAL_END_STATE_RECVD:
                ends[end_id].set_property(KEY_END_STATE, VAL_END_STATE_NONE)
            self._free_end(end_id, selected_ends)
        self.selected_ends[self.requester] = selected_ends
        logger.info(
            f"Freed {len(ends_to_cleanup)} end(s) for resampling; "
            f"state set to {VAL_END_STATE_NONE}."
        )

    def _cleanup_recvd_end(self, end_id: str, end: End) -> None:
        """Per-end counterpart of `_cleanup_recvd_ends`.

        Only `RandomSelector` used to implement this, so `channel.cleanup_
        recvd_end()` raised AttributeError the first time an async selector
        reached it (fluxtune, production). The aggregator now routes async
        callers to `cleanup_provided_ends` instead, but implementing it here
        makes the interface total rather than relying on that routing.
        """
        self._cleanup_provided_ends({end_id: end}, {end_id: end})

    def _cleanup_removed_ends(self, end_id: str) -> None:
        """Release an end that left the channel."""
        if end_id not in self.all_selected:
            logger.warning(f"end {end_id} left but was not in all_selected")
            return

        if end_id in self.ordered_updates_recv_ends:
            # Its update already landed -- participation is complete, so the
            # bookkeeping stays until _cleanup_recvd_ends drains it.
            logger.debug(f"update already received from {end_id} before it left")
            return

        # Left without returning an update; it flushed agg-weights on leave.
        selected_ends = self.selected_ends[self.requester]
        self._free_end(end_id, selected_ends)
        self.selected_ends[self.requester] = selected_ends

        self.track_selected_trainers_which_left[end_id] = (
            self.track_selected_trainers_which_left.get(end_id, 0) + 1
        )
        logger.debug(
            f"Trainer {end_id} left before returning an update "
            f"(count {self.track_selected_trainers_which_left[end_id]}, "
            f"total drop-offs {sum(self.track_selected_trainers_which_left.values())})"
        )

    def _cleanup_send_ends(self) -> None:
        """Release every in-flight end for this requester.

        Trainer-side recovery hook (see syncfl trainer `_send_*` failure
        paths); a no-op for the aggregator, which frees per-end instead.
        """
        selected_ends = self.selected_ends[self.requester]
        for end_id in list(selected_ends):
            self._free_end(end_id, selected_ends)
        self.selected_ends[self.requester] = selected_ends

    # ------------------------------------------------------------ misc state

    def reset_end_state_to_none(self, ends: dict[str, End], end_id: str) -> None:
        """Reset an end's state from send/recv back to none."""
        if end_id not in ends:
            logger.debug(f"Attempted to reset end {end_id} but it wasn't in ends")
            return
        ends[end_id].set_property(KEY_END_STATE, VAL_END_STATE_NONE)

    def remove_from_selected_ends(self, ends: dict[str, End], end_id: str) -> None:
        """Remove an end from the in-flight set."""
        selected_ends = self.selected_ends[self.requester]
        if end_id in ends and end_id in selected_ends:
            selected_ends.remove(end_id)
            self.selected_ends[self.requester] = selected_ends
        else:
            logger.debug(f"end {end_id} not removable from selected_ends")

    def reset_selected_ends(self, requester: str) -> None:
        """Drop a requester's in-flight set (used when it leaves the channel)."""
        if requester not in self.selected_ends:
            return
        selected_ends = self.selected_ends[requester]
        # Ends selected by OTHER requesters keep their slot; re-stamp them so
        # the abandon timeout measures from now rather than firing instantly.
        for end_id in set(self.all_selected) - selected_ends:
            self.all_selected[end_id] = self._abandon_clock_now()
        del self.selected_ends[requester]
        logger.debug(f"reset selected ends of {requester}")
