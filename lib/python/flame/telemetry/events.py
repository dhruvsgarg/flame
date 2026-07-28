# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Telemetry event types and typed builders.

Centralizing the event names + field schemas here is what makes runs
comparable across selectors / aggregators / examples. Emitters should call the
``build_*`` helpers (or :func:`flame.telemetry.emit` with these constants) so
every run produces the same columns.
"""

from __future__ import annotations

from typing import Any, Optional

# ---- Event type constants -------------------------------------------------

EVENT_RUN_META = "run_meta"          # one-time run identity / config snapshot
EVENT_SELECTION = "selection"        # selector decision for a round
EVENT_AGG_EVAL = "agg_eval"          # aggregator test loss/accuracy
EVENT_AGG_ROUND = "agg_round"        # aggregation step: staleness/agg-goal/participation
EVENT_TRAINER_ROUND = "trainer_round"  # per-round trainer timing/availability
EVENT_UTIL_DISPARITY = "util_disparity"  # streamed-prefix vs full-pool utility
EVENT_AVAIL_CHANGE = "avail_change"  # trainer availability state transition
EVENT_TASK_RECV = "task_recv"        # trainer received a task from aggregator
EVENT_TASK_SEND = "task_send"        # trainer finished & sent the update back
EVENT_INFLIGHT_RESIDENCE = "inflight_residence"  # per-round in-flight drain accounting (oort sync)
EVENT_UTILITY_BELIEF = "utility_belief"  # believed (at selection) vs actual (at return) client utility
EVENT_DISPATCH = "dispatch"          # per-dispatch re-dispatch-stagger validation (felix)
EVENT_WITHHELD_DELIVERY = "withheld_delivery"  # late stale delivery of a send-gated update
EVENT_ABANDON_TIMEOUT = "abandon_timeout"      # 90s vclock slot-free of a stalled trainer
EVENT_AGG_BELIEF_CHANGE = "agg_belief_change"   # aggregator's belief about a trainer's avail state
EVENT_STEP_TIMING = "step_timing"    # per-function wall duration of a timed compute step
EVENT_COMM = "comm"                  # one message put on the wire (byte-size accounting)
EVENT_VERSION_BUMP_CENSUS = "version_bump_census"  # #S1: pool-wide in-flight state at a model_version bump
EVENT_VAR_CALC = "var_calc"          # fwdllm: grad-norm summary in/out of the variance gate (DEBUG-only audit)
EVENT_REDISPATCH_DECOMP = "redispatch_decomp"  # fwdllm round-cadence: commit->next-dispatch wall split
EVENT_SLOT_STARVATION = "slot_starvation"  # a freed dispatch slot had fewer eligible candidates than slots
EVENT_VCLOCK_CHARGE = "vclock_charge"  # every charge_sim_vclock_overhead() call: measured span vs actually-charged

KNOWN_EVENTS = frozenset(
    {
        EVENT_RUN_META,
        EVENT_SELECTION,
        EVENT_AGG_EVAL,
        EVENT_AGG_ROUND,
        EVENT_TRAINER_ROUND,
        EVENT_UTIL_DISPARITY,
        EVENT_AVAIL_CHANGE,
        EVENT_TASK_RECV,
        EVENT_TASK_SEND,
        EVENT_INFLIGHT_RESIDENCE,
        EVENT_UTILITY_BELIEF,
        EVENT_DISPATCH,
        EVENT_WITHHELD_DELIVERY,
        EVENT_ABANDON_TIMEOUT,
        EVENT_AGG_BELIEF_CHANGE,
        EVENT_STEP_TIMING,
        EVENT_COMM,
        EVENT_VERSION_BUMP_CENSUS,
        EVENT_VAR_CALC,
        EVENT_REDISPATCH_DECOMP,
        EVENT_SLOT_STARVATION,
        EVENT_VCLOCK_CHARGE,
    }
)


# ---- Typed builders -------------------------------------------------------
# Each returns (event_type, fields_dict). Callers do:
#   telemetry.emit(*build_selection(...))  -> emit(event, **fields)
# but emit() takes (event, **fields), so callers use:
#   ev, f = build_selection(...); telemetry.emit(ev, **f)


def build_selection(
    *,
    round_num: int,
    task: str,
    selector: str,
    num_candidates: int,
    num_eligible: int,
    avail_composition: dict[str, int],
    chosen: list[str],
    in_flight: int,
    per_trainer: Optional[dict[str, dict[str, Any]]] = None,
    extra: Optional[dict[str, Any]] = None,
) -> tuple[str, dict[str, Any]]:
    """Selector decision record.

    avail_composition: counts keyed by availability-state name
        (e.g. {"AVL_TRAIN": 30, "AVL_EVAL": 5, "UN_AVL": 65}).
    per_trainer: optional {end_id: {"utility": .., "speed_s": .., "selected": bool}}.
    extra: selector-specific fields (e.g. explore/exploit split, cutoff).
    """
    fields: dict[str, Any] = {
        "round": round_num,
        "task": task,
        "selector": selector,
        "num_candidates": num_candidates,
        "num_eligible": num_eligible,
        "avail_composition": avail_composition,
        "chosen": list(chosen),
        "num_chosen": len(chosen),
        "in_flight": in_flight,
    }
    if per_trainer is not None:
        fields["per_trainer"] = per_trainer
    if extra:
        fields.update(extra)
    return EVENT_SELECTION, fields


def build_agg_eval(
    *, round_num: int, metrics: dict[str, float]
) -> tuple[str, dict[str, Any]]:
    """Aggregator evaluation metrics (loss/accuracy/...)."""
    fields = {"round": round_num}
    fields.update(metrics)
    return EVENT_AGG_EVAL, fields


def build_agg_round(
    *,
    round_num: int,
    agg_goal: Optional[int] = None,
    agg_goal_count: Optional[int] = None,
    in_flight: Optional[int] = None,
    updates_in_queue: Optional[int] = None,
    staleness: Optional[list[float]] = None,
    stat_utility: Optional[list[float]] = None,
    trainer_speed_s: Optional[list[float]] = None,
    contributing_trainers: Optional[list[str]] = None,
    agg_observed_s: Optional[dict[str, float]] = None,
    extra: Optional[dict[str, Any]] = None,
) -> tuple[str, dict[str, Any]]:
    """Aggregation-step record (one per completed aggregation).

    agg_observed_s: {end_id -> wall seconds the aggregator observed between
        sending the model and receiving/processing that trainer's update}. Lets
        the analyzer compare aggregator-side turnaround to the trainer-reported
        time (overhead sanity check).
    """
    fields: dict[str, Any] = {"round": round_num}
    for k, v in (
        ("agg_goal", agg_goal),
        ("agg_goal_count", agg_goal_count),
        ("in_flight", in_flight),
        ("updates_in_queue", updates_in_queue),
        ("staleness", staleness),
        ("stat_utility", stat_utility),
        ("trainer_speed_s", trainer_speed_s),
        ("contributing_trainers", contributing_trainers),
        ("agg_observed_s", agg_observed_s),
    ):
        if v is not None:
            fields[k] = v
    if extra:
        fields.update(extra)
    return EVENT_AGG_ROUND, fields


def build_trainer_round(
    *,
    round_num: int,
    real_gpu_time_s: float,
    sim_round_duration_s: Optional[float] = None,
    wait_time_s: Optional[float] = None,
    avail_state: Optional[str] = None,
    visible_samples: Optional[int] = None,
    total_samples: Optional[int] = None,
    dataset_size: Optional[int] = None,
    stat_utility: Optional[float] = None,
    final_loss: Optional[float] = None,
    delta_weight_l2: Optional[float] = None,
    extra: Optional[dict[str, Any]] = None,
) -> tuple[str, dict[str, Any]]:
    """Per-round trainer timing/availability record.

    delta_weight_l2: L2 norm of the model update (||trained - received global||)
        the trainer uploads -- lets analysis relate update magnitude to the
        amount of unlocked data under streaming.
    """
    fields: dict[str, Any] = {
        "round": round_num,
        "real_gpu_time_s": real_gpu_time_s,
    }
    for k, v in (
        ("sim_round_duration_s", sim_round_duration_s),
        ("wait_time_s", wait_time_s),
        ("avail_state", avail_state),
        ("visible_samples", visible_samples),
        ("total_samples", total_samples),
        ("dataset_size", dataset_size),
        ("stat_utility", stat_utility),
        ("final_loss", final_loss),
        ("delta_weight_l2", delta_weight_l2),
    ):
        if v is not None:
            fields[k] = v
    if extra:
        fields.update(extra)
    return EVENT_TRAINER_ROUND, fields


def build_step_timing(
    *,
    func: str,
    duration_s: float,
    round_num: Optional[int] = None,
    data_id: Optional[int] = None,
    iteration: Optional[int] = None,
    trainer_id: Optional[str] = None,
    vclock_s: Optional[float] = None,
    vclock_now_s: Optional[float] = None,
    cpu_duration_s: Optional[float] = None,
    gc_pause_s: Optional[float] = None,
) -> tuple[str, dict[str, Any]]:
    """Per-function wall duration of one timed compute step (`timer_decorator`).

    Fine-grained companion to `trainer_round`'s coarse phase split: attributes
    wall time to individual forward-grad steps (functional-model setup,
    perturbation selection, per-batch JVP, delay emulation) for GPU-cost
    decomposition. Keyed by (data_id, iteration) to track cost across the cadence.

    `vclock_s`/`vclock_now_s` are sim-mode-only, absent (not 0.0) in real mode.
    `vclock_s` is this step's vclock delta (meaningful on the aggregator,
    where the clock ticks live; usually 0 on a trainer, which only has a
    last-known snapshot). `vclock_now_s` is the cumulative pointer as of this
    step's end, for cross-step alignment.

    `cpu_duration_s` is thread CPU time consumed, vs `duration_s`'s wall time.
    Wall time diverging real<->sim while CPU time doesn't means the function
    is waiting on contention, not doing more work.

    `gc_pause_s` is the portion of `duration_s` that overlapped a cyclic-GC
    collection, 0.0 if none ran. Isolates a GC pause from a genuine per-call
    compute regression.
    """
    fields: dict[str, Any] = {"func": func, "duration_s": duration_s}
    for k, v in (
        ("round", round_num),
        ("data_id", data_id),
        ("iteration_per_data_id", iteration),
        ("trainer_id", trainer_id),
        ("vclock_s", vclock_s),
        ("vclock_now_s", vclock_now_s),
        ("cpu_duration_s", cpu_duration_s),
        ("gc_pause_s", gc_pause_s),
    ):
        if v is not None:
            fields[k] = v
    return EVENT_STEP_TIMING, fields


def build_var_calc(
    *,
    round_num: int,
    data_id: int,
    iteration: int,
    input_grad_norms: list[float],
    output_var: float,
) -> tuple[str, dict[str, Any]]:
    """DEBUG-only audit: per-tensor L2 norm of the input `grad_for_var_check_
    list` feeding `calculate_var`, plus its scalar output, one record per
    `_compute_var` call. Diffable real vs sim to localize a `v2_var_trajectory`
    divergence to a specific input tensor vs the reduction itself. Gated at
    the call site -- the norm computation is a GPU->CPU sync.
    """
    return EVENT_VAR_CALC, {
        "round": round_num,
        "data_id": data_id,
        "iteration_per_data_id": iteration,
        "input_grad_norms": input_grad_norms,
        "output_var": output_var,
    }


def build_comm(
    *,
    direction: str,
    size_bytes: int,
    peer_id: Optional[str] = None,
    round_num: Optional[int] = None,
    data_id: Optional[int] = None,
    iteration: Optional[int] = None,
    payload_kind: Optional[str] = None,
    n_tensors: Optional[int] = None,
    trainer_id: Optional[str] = None,
    model_version: Optional[int] = None,
) -> tuple[str, dict[str, Any]]:
    """One message placed on the wire, for network-cost accounting (Experiment 4).

    Emitted by BOTH roles so total bytes / message counts / per-message size
    distributions are comparable across baselines (fluxtune sends perturbation
    seeds/scalars, not full gradients, so real wire size differs from the static
    model_param_count reconstruction).

    direction: "agg_to_trainer" (dispatch) | "trainer_to_agg" (update upload).
    peer_id: the other end (may be None trainer-side). payload_kind: "weights" /
    "var_bad" / "gradients", to split dispatch vs update and full-weight vs
    var-signal. size_bytes: serialized message size. model_version: the
    version this message carries (dispatch: what's being sent out; update:
    what the sender computed against), for staleness diagnostics.
    """
    fields: dict[str, Any] = {"direction": direction, "size_bytes": int(size_bytes)}
    for k, v in (
        ("peer_id", peer_id),
        ("round", round_num),
        ("data_id", data_id),
        ("iteration_per_data_id", iteration),
        ("payload_kind", payload_kind),
        ("n_tensors", n_tensors),
        ("trainer_id", trainer_id),
        ("model_version", model_version),
    ):
        if v is not None:
            fields[k] = v
    return EVENT_COMM, fields


def build_redispatch_decomp(
    *,
    end_id: str,
    round_num: int,
    data_id: int,
    iteration: int,
    redispatch_gap_wall_s: float,
    peer_wait_wall_s: float,
    post_close_overhead_wall_s: float,
    time_mode: str,
    payload_kind: str = "weights",
) -> tuple[str, dict[str, Any]]:
    """fwdllm round-cadence: split a trainer's commit->next-dispatch WALL gap
    into peer-wait vs post-close overhead.

    Round-cadence (`fedbuff_round`/`felix_round`) pins a fixed cohort and only
    re-dispatches a committed trainer once the WHOLE `agg_goal`-sized
    micro-batch's `version_key` advances (§D-8/F-25) -- so most of the gap is
    this trainer waiting on its round-mates, not idle server time. This event
    disambiguates the two, using ``self._last_round_close_wall_ts`` (wall
    ``aggregate()`` finished, real time in BOTH modes -- see §F-1) as the
    boundary:

    ``redispatch_gap_wall_s`` = now - this end's own last commit wall ts.
    ``peer_wait_wall_s``      = round-close wall ts - this end's own commit wall ts
                                 (0 if this end's own commit WAS the round-closer,
                                 or no round has closed since its commit).
    ``post_close_overhead_wall_s`` = now - round-close wall ts: genuine
                                 server-side redispatch turnaround, free of
                                 peer-wait -- the residual to actually calibrate
                                 `sim_redispatch_gap_s` against, if non-trivial.

    Emitted for both `send_weights` and `VAR=bad` dispatches (``payload_kind``
    distinguishes them) -- both share the same channel-send call, and VAR=bad
    retries are the majority of cycles (§D-11), so excluding them hid most of
    the signal. Wall-clock (`time.time()`) in BOTH modes: sim doesn't sleep to
    emulate the modeled training delay (§F-1), so a genuine sim/real gap here
    means the SIMULATOR's own wall-clock redispatch loop is faster, not that a
    cost is unmodeled on the vclock -- compare against `overhead_residual`/
    `per_round_advance` (vclock-based) before concluding a vclock gap exists.
    """
    return EVENT_REDISPATCH_DECOMP, {
        "end_id": end_id,
        "round": round_num,
        "data_id": data_id,
        "iteration_per_data_id": iteration,
        "redispatch_gap_wall_s": redispatch_gap_wall_s,
        "peer_wait_wall_s": peer_wait_wall_s,
        "post_close_overhead_wall_s": post_close_overhead_wall_s,
        "time_mode": time_mode,
        "payload_kind": payload_kind,
    }


def build_vclock_charge(
    *,
    label: str,
    span_s: float,
    charged_s: float,
    time_mode: str,
    vclock_now: Optional[float] = None,
    payload_kind: Optional[str] = None,
) -> tuple[str, dict[str, Any]]:
    """One record per `charge_sim_vclock_overhead()` call, both modes -- the
    shared ledger of what wall-time got charged onto the vclock, per baseline.

    ``span_s`` = measured wall duration passed in (both modes, comparable).
    ``charged_s`` = what actually landed on the vclock (0.0 in real always;
    0.0 in sim if the flag's off or `charge=False`, else `span_s`).
    ``vclock_now`` = sim's clock after this call (None in real).

    A real-vs-sim `span_s` gap that `charged_s` never reflects is the §F-1
    unmodeled-cost signature (simulate_fwdllm.md §D-11).
    """
    return EVENT_VCLOCK_CHARGE, {
        "label": label,
        "span_s": span_s,
        "charged_s": charged_s,
        "time_mode": time_mode,
        "vclock_now": vclock_now,
        "payload_kind": payload_kind,
    }


def build_slot_starvation(
    *,
    concurrency: int,
    extra: int,
    n_filtered: int,
    feasible_extra: int,
    model_version: Optional[int] = None,
) -> tuple[str, dict[str, Any]]:
    """A dispatch slot just freed up (`extra > 0`) but fewer eligible
    candidates existed than slots to fill (`feasible_extra < extra`,
    `async_oort.py::handle_send_state`) -- the candidate-POOL side of D-10
    (simulate_fwdllm.md): round cadence's pinned cohort can run out of
    not-yet-contributed-to-this-version_key members before its `agg_goal`
    batch closes, iteration cadence draws from the whole trainer pool
    instead. Emitted ONLY on a starved tick (`feasible_extra < extra`), not
    every `select()` call, to stay low-volume across a baseline sweep.
    """
    return EVENT_SLOT_STARVATION, {
        "concurrency": concurrency,
        "extra": extra,
        "n_filtered": n_filtered,
        "feasible_extra": feasible_extra,
        "starved": extra - feasible_extra,
        "model_version": model_version,
    }


def build_version_bump_census(
    *,
    old_model_version: int,
    new_model_version: int,
    data_id: int,
    inflight: dict[str, tuple[int, int]],
    vclock_now: Optional[float] = None,
) -> tuple[str, dict[str, Any]]:
    """Diagnostic: pool-wide snapshot at the instant model_version bumps.

    inflight: {end_id: dispatch_version_key} for every trainer with an
    outstanding (dispatched, not-yet-returned) send at this instant --
    version_key is the full (model_version, iteration_per_data_id) tuple, not
    just model_version, so a real/sim comparison isn't fooled by a matching
    model_version that hides a differing iteration. Compares how many
    trainers are stale at the bump, and at what version, between real and sim.
    """
    fields: dict[str, Any] = {
        "old_model_version": old_model_version,
        "new_model_version": new_model_version,
        "data_id": data_id,
        "n_inflight": len(inflight),
        "inflight_version_key": {e: list(k) for e, k in inflight.items()},
        "inflight_staleness": {
            e: new_model_version - k[0] for e, k in inflight.items()
        },
    }
    if vclock_now is not None:
        fields["vclock_now"] = vclock_now
    return EVENT_VERSION_BUMP_CENSUS, fields


def build_util_disparity(
    *,
    round_num: int,
    elapsed_s: float,
    visible_samples: int,
    total_samples: int,
    utility_streamed: float,
    utility_full: float,
    sample_size_used: Optional[int] = None,
) -> tuple[str, dict[str, Any]]:
    """Streamed-prefix vs full-dataset statistical-utility comparison."""
    visible_fraction = (
        visible_samples / total_samples if total_samples else None
    )
    ratio = (
        utility_streamed / utility_full
        if utility_full not in (0, None)
        else None
    )
    fields: dict[str, Any] = {
        "round": round_num,
        "elapsed_s": elapsed_s,
        "visible_samples": visible_samples,
        "total_samples": total_samples,
        "visible_fraction": visible_fraction,
        "utility_streamed": utility_streamed,
        "utility_full": utility_full,
        "utility_ratio": ratio,
    }
    if sample_size_used is not None:
        fields["sample_size_used"] = sample_size_used
    return EVENT_UTIL_DISPARITY, fields


def build_avail_change(
    *,
    round_num: Optional[int],
    old_state: str,
    new_state: str,
    sim_now: Optional[float] = None,
) -> tuple[str, dict[str, Any]]:
    """Trainer availability state transition.

    sim_now: the trainer's own trace-time-basis clock (``_sim_now()``) at the
    moment the transition was applied — sim: virtual-clock seconds; real:
    wall-elapsed since the shared AGG_START_TS origin (Batch 3 T3.0). Distinct
    from the record's own wall ``ts`` (always epoch time.time(), meaningless
    against a trace indexed in trace-seconds). Needed for A6
    (trainer_trace_fidelity, Batch 3 T3.2) to compare *when* a trainer applied
    a transition against ground truth, not just *that* it eventually did.
    Optional/back-compat: None for telemetry recorded before this field existed.
    """
    return EVENT_AVAIL_CHANGE, {
        "round": round_num,
        "old_state": old_state,
        "new_state": new_state,
        "sim_now": sim_now,
    }


def build_task_recv(
    *,
    round_num: int,
    trainer_id: str,
    time_mode: str,
    sim_send_ts: Optional[float] = None,
    avl_state: Optional[str] = None,
) -> tuple[str, dict[str, Any]]:
    """Trainer received a task (weights) from the aggregator.

    sim_send_ts: the virtual clock value stamped by the aggregator (sim mode only).
    Emitting None in real mode for both sim_send_ts and vclock makes the per-round
    trainer state directly comparable between real and sim telemetry.
    """
    fields: dict[str, Any] = {
        "round": round_num,
        "trainer_id": trainer_id,
        "time_mode": time_mode,
        "sim_send_ts": sim_send_ts,
        "avl_state": avl_state,
    }
    return EVENT_TASK_RECV, fields


def build_task_send(
    *,
    round_num: int,
    trainer_id: str,
    task_to_perform: Optional[str],
    wall_recv_ts: Optional[float],
    wall_send_ts: float,
    time_mode: str,
    send_gate_wait_s: Optional[float] = None,
    send_gate_sct: Optional[float] = None,
) -> tuple[str, dict[str, Any]]:
    """Trainer finished a task and sent the update back to the aggregator.

    Unlike trainer_round (emitted inside train(), BEFORE the real-mode budget
    sleep), this fires from _send_weights — AFTER the sleep and the upload — so
    ``[wall_recv_ts, wall_send_ts]`` brackets the trainer's true busy/in-flight
    window in real mode.  That interval is the sound basis for real concurrency
    in validate_real: trainer_round's own ts cannot bracket it.

    send_gate_wait_s / send_gate_sct (Batch 3 T3.4, real mode only): the
    [SEND_GATE] wait loop also runs strictly after trainer_round is emitted
    (train() -> put()/_send_weights, per the tasklet composition), so — like
    wall_send_ts above — task_send, not trainer_round, is the event that can
    actually carry it. send_gate_sct is the trainer's own _sim_now() sampled
    right before the gate check (same trace-time-basis clock T3.2's
    avail_change.sim_now uses); send_gate_wait_s is the wall-time actually
    spent blocked in the loop (0.0 when the gate never engaged). Always None
    in sim mode (the gate is a real-mode-only mechanism).
    """
    return EVENT_TASK_SEND, {
        "round": round_num,
        "trainer_id": trainer_id,
        "task_to_perform": task_to_perform,
        "wall_recv_ts": wall_recv_ts,
        "wall_send_ts": wall_send_ts,
        "time_mode": time_mode,
        "send_gate_wait_s": send_gate_wait_s,
        "send_gate_sct": send_gate_sct,
    }


def build_inflight_residence(
    *,
    round_num: int,
    time_mode: str,
    in_flight_before: int,
    in_flight_after: int,
    newly_selected: Optional[int] = None,
    committed_fresh: Optional[int] = None,
    cleaned: Optional[int] = None,
    stale_rejected: Optional[int] = None,
    residence_rounds: Optional[list[int]] = None,
    carried_over_ages: Optional[list[int]] = None,
    residence_staleness: Optional[list[int]] = None,
    residence_was_fresh: Optional[list[bool]] = None,
) -> tuple[str, dict[str, Any]]:
    """Per-round in-flight drain accounting for the oort sync aggregator.

    Localizes the in-flight RESIDENCE divergence (real holds ~15.6 in-flight, sim
    drains to the designed ~13): a straggler occupies ``selected_ends`` from selection
    until it is cleaned. ``residence_rounds`` = (current_round − entry_round) for each
    trainer cleaned this round; ``carried_over_ages`` = ages of those still in-flight
    AFTER cleanup. Comparing sim vs real residence distributions shows whether sim
    evicts stragglers a round too early (the eviction-timing fine-tune). ``time_mode``
    = "sim"|"real" so the two are directly comparable.

    ``residence_staleness`` / ``residence_was_fresh`` are PAIRED 1:1 with
    ``residence_rounds`` (same order, same cleaned ends): the commit staleness
    (``round − trained_version``) and the fresh-vs-stale-reject class of each cleaned
    end. They decompose the residence-distribution SHAPE gap (refl A2: real peaks at
    residence=3, sim flatter) by commit class — i.e. whether sim under-holds the
    fresh-committed body or the stale-carryover tail.
    """
    fields: dict[str, Any] = {
        "round": round_num,
        "time_mode": time_mode,
        "in_flight_before": in_flight_before,
        "in_flight_after": in_flight_after,
    }
    for k, v in (
        ("newly_selected", newly_selected),
        ("committed_fresh", committed_fresh),
        ("cleaned", cleaned),
        ("stale_rejected", stale_rejected),
        ("residence_rounds", residence_rounds),
        ("carried_over_ages", carried_over_ages),
        ("residence_staleness", residence_staleness),
        ("residence_was_fresh", residence_was_fresh),
    ):
        if v is not None:
            fields[k] = v
    return EVENT_INFLIGHT_RESIDENCE, fields


def build_dispatch(
    *,
    round_num: int,
    end_id: str,
    task: str,
    time_mode: str,
    sim_send_ts: Optional[float] = None,
    redispatch_stagger_s: Optional[float] = None,
    held_s: Optional[float] = None,
    staggered: Optional[bool] = None,
) -> tuple[str, dict[str, Any]]:
    """Per-dispatch re-dispatch-stagger validation (felix event-driven re-dispatch).

    ``redispatch_stagger_s`` = this end's ``sim_send_ts`` minus the cohort minimum
    in the same distribute call: 0 for the legacy round-boundary batch (all share
    one frozen vclock), spread across the round's advance once event-driven
    re-dispatch is on. ``held_s`` = vclock minus this end's PRIOR commit sct = how
    long (virtual seconds) it sat held since it last completed before being
    re-dispatched; the boundary backlog shows large held_s, continuous re-dispatch
    drives it toward 0. Sim-only fields; lets the run confirm the cohort next-sct
    spread recovers real's ~3.85s before reading K2/K3b/U3.
    """
    fields: dict[str, Any] = {
        "round": round_num,
        "end_id": end_id,
        "task": task,
        "time_mode": time_mode,
    }
    for k, v in (
        ("sim_send_ts", sim_send_ts),
        ("redispatch_stagger_s", redispatch_stagger_s),
        ("held_s", held_s),
        ("staggered", staggered),
    ):
        if v is not None:
            fields[k] = v
    return EVENT_DISPATCH, fields


def build_utility_belief(
    *,
    round_num: int,
    end_id: str,
    believed: Optional[float],
    actual: Optional[float],
    staleness: Optional[int] = None,
    time_mode: Optional[str] = None,
    extra: Optional[dict[str, Any]] = None,
) -> tuple[str, dict[str, Any]]:
    """Believed-vs-actual client statistical utility, per returning trainer.

    ``believed`` = the utility the selector held for this client when it was selected
    (``PROP_STAT_UTILITY`` *before* this return overwrites it — the value from the
    client's previous return, i.e. STALE by ``staleness`` rounds). ``actual`` = the
    fresh Oort statistical utility the client computed this round and reports on return
    (``MessageType.STAT_UTILITY``). Both are the SAME quantity (Oort stat-utility), so
    ``believed − actual`` is the pure staleness error in the selector's belief — the
    quantity the "believed vs actual utility" plot needs. Emitted for EVERY baseline
    (every client reports stat-utility on return, even non-utility selectors), so the
    plot compares felix/eval-refreshed beliefs against the stale-utility baselines.
    ``believed`` is None on a client's first-ever return (no prior belief)."""
    fields: dict[str, Any] = {
        "round": round_num,
        "end_id": end_id,
        "believed": believed,
        "actual": actual,
    }
    if staleness is not None:
        fields["staleness"] = staleness
    if time_mode is not None:
        fields["time_mode"] = time_mode
    if extra:
        fields.update(extra)
    return EVENT_UTILITY_BELIEF, fields


def build_withheld_delivery(
    *,
    round_num: int,
    end_id: str,
    sct: float,
    delivery_ts: float,
    staleness: Optional[int] = None,
    accepted: Optional[bool] = None,
    time_mode: str = "sim",
    actual_commit_ts: Optional[float] = None,
) -> tuple[str, dict[str, Any]]:
    """A send-gated update committing late (stale) at its ``delivery_ts``.

    ``delivery_ts − sct`` is the down-window delay the completed update waited
    while its trainer was ``UN_AVL`` (the compute-completes / gate-the-send /
    deliver-late model). ``staleness`` = the current round minus the update's
    ``MODEL_VERSION``; ``accepted`` records the staleness-gate outcome (async
    fedbuff always accepts; sync feddance may reject over tolerance — Stage E).
    Backs the ``withheld_delivery`` parity rung.

    actual_commit_ts (Batch 3 T3.5, K11): the aggregator's own clock
    (``_avail_now()`` — vclock in sim, wall-elapsed in real) sampled at the
    instant this update actually commits, i.e. AFTER the caller's
    ``_advance_sim_clock``/equivalent for this specific update, not before —
    the whole point is to catch reinjection-polling lag between ``delivery_ts``
    (the earliest legal commit time — already exactly what ``delivery_ts`` is,
    no separate ``earliest_legally_committable_time`` bookkeeping needed for
    this single-gate-type v1) and when the update actually lands.
    ``commit_slack_s = actual_commit_ts - delivery_ts`` is derived by the
    caller-facing K11 check, not stored here, to keep this builder a thin
    field-carrier like its siblings.
    """
    fields: dict[str, Any] = {
        "round": round_num,
        "end_id": end_id,
        "sct": sct,
        "delivery_ts": delivery_ts,
        "delay_s": float(delivery_ts) - float(sct),
        "time_mode": time_mode,
    }
    if staleness is not None:
        fields["staleness"] = staleness
    if accepted is not None:
        fields["accepted"] = accepted
    if actual_commit_ts is not None:
        fields["actual_commit_ts"] = float(actual_commit_ts)
    return EVENT_WITHHELD_DELIVERY, fields


def build_abandon_timeout(
    *,
    round_num: int,
    end_id: str,
    sim_send_ts: float,
    vclock_now: float,
    time_mode: str = "sim",
    reason: str = "abandon_90s_vclock",
) -> tuple[str, dict[str, Any]]:
    """A stalled in-flight trainer freed by a slot-free trigger.

    ``reason`` distinguishes C.3 90s-vclock abandons from D.1 aware boundary
    evictions. ``vclock_now − sim_send_ts`` is the in-flight age at trigger.
    Backs the ``abandon_timeout`` parity rung, which fails loudly if the
    deadline is measured on the wall instead of the vclock.
    """
    return EVENT_ABANDON_TIMEOUT, {
        "round": round_num,
        "end_id": end_id,
        "sim_send_ts": sim_send_ts,
        "vclock_now": vclock_now,
        "age_s": float(vclock_now) - float(sim_send_ts),
        "time_mode": time_mode,
        "reason": reason,
    }


def build_agg_belief_change(
    *,
    round_num: int,
    end_id: str,
    state: str,
    observed_at: float,
    checkpoint: str,
    source: str = "trace_read",
) -> tuple[str, dict[str, Any]]:
    """The aggregator's belief about `end_id`'s availability state (Batch 3 T3.3).

    ``checkpoint`` distinguishes WHERE the aggregator's view was read:
    ``"selection"`` (pre-round eligibility read) or ``"commit"`` (state at an
    update's completion time, whether or not the mechanism actually gates on
    it — real mode never gates here, sim's send-gate does; recording both
    lets A7 compare the aggregator's belief against ground truth regardless).
    ``source`` is the knowledge MECHANISM the belief came from — ``trace_read``
    (v1, live) today, ``client_notify``/``predictive`` later (Stage H) — kept
    mechanism-agnostic so a future populator just passes a different source
    without any telemetry/checker/plot change. ``observed_at`` is the
    trace-time-basis clock (vclock seconds / wall-elapsed since the shared
    origin) at which the belief was read, NOT the record's own wall ``ts``.
    """
    return EVENT_AGG_BELIEF_CHANGE, {
        "round": round_num,
        "end_id": end_id,
        "state": state,
        "observed_at": observed_at,
        "checkpoint": checkpoint,
        "source": source,
    }
