# Sim Unavailability — Design & Staged Plan

## Working agreement (standing instructions — read every session)

**Implement breadth-first across stages on ONE baseline with short runs; batch the long parity runs at
the end. Never block forward implementation on a long run.**

1. **Common first, one baseline first.** Land shared/library-level changes once, drive through a single
   reference baseline — oort/refl for async/unaware, felix only for aware-specific. Don't fan out until it
   behaves on the reference.
2. **Short runs to debug, long runs to confirm.** Gate forward work on unit tests + syn_0 byte-identity +
   shortest syn_20 smoke (~1800s vclock). Long runs confirm; never find first bugs.
3. **Across stages before across baselines, long runs last.** Stack mechanisms across stages on the reference
   baseline, then widen to other baselines, then batch longer parity runs. One long run confirms several stages.
4. **Keep this doc crisp.** Completed stages: mechanism + where it lives + exit met (2–3 lines max). Full
   detail only for active and next stages. Dead-ends in §9.

---

## Status (Jun 28 — syn_0 ✅, syn_50 n=10 starvation smoke in progress)

**A/B/C/C.6/D ✅ CONFIRMED syn_20. E ✅ CONFIRMED syn_20. F.2 ✅ CODE-COMPLETE.
syn_0 regression ✅ PASSED (no starvation fires, byte-identical model outputs).
n=10 syn_50 starvation smoke launched — results in morning. Batch 2 pending Stage F exit.**

- **A/B ✅** Substrate (`flame/availability/trace.py` + `AvailabilityMixin`) + A3 time-base CONTROL. Exit: A3 PASS oort/felix syn_20.
- **C ✅ CONFIRMED** Oracular gate, send-time withhold, vclock 90s abandon, `delivery_ts` ordering, `free_stalled_slot`. felix 49/49; oort 39/48 (3 pre-existing failures, see §7).
- **C.6 ✅ CONFIRMED** `_avail_stamp_end_states`, A4dur PASS (felix 0.0024, oort 0.0029), 5 availability plots.
- **D ✅ CONFIRMED** Proactive eviction (felix), task-aware eligibility with `_trace_has_avl_eval` guard, accept-stale withheld. Real send-gate confirmed (withheld n=7, accept_frac=1.0).
- **E ✅ CONFIRMED syn_20** Syncfl path (feddance+refl): abandon/evict/stamp + `_sync_sim_recv_first_k` withhold drain. feddance 46/47 (C2 emergent noise, not mechanism; A-rungs/U3/U6/K8/U2 PASS).
- **F.2 ✅ CODE-COMPLETE** Unified pre-selection return-early pattern in all three aggregators (see §5/Stage F). **syn_0 ✅**: Fst PASS (no starvation), C1/C2 diff=0.0, K1/K5 PASS. Oort: 43/50 (K3b pre-existing gates K2/K3); feddance: 46/48 (A2 shape artifact + gpu_compute short-run noise — K2/K3/P3 PASS). **n=10 syn_50 in progress.**
- **G.1 ✅** `starvation_advance` rung in `checks.py` + `report.py`; +4 starvation unit tests.

---

## ▶ Next actions

### Stage F exit — check syn_50 n=10 results (morning)

**syn_0 ✅ DONE.** syn_50 n=10 starvation smoke launched (both baselines, both modes). Check results:
```bash
cd lib/python/examples/async_cifar10
python -m scripts.parity.cli --batch --experiments-dir experiments --baselines oort --agg-goal 10
python -m scripts.parity.cli --batch --experiments-dir experiments --baselines feddance --agg-goal 10
```

**Stage F exit criteria:** `starvation_advance` rung populated (n_starvation_jumps > 0) for BOTH baselines; `[SIM_STARVATION]` log lines in sim trace; K1 monotone; no stalls.

If starvation didn't fire: check sim aggregator log for `[SIM_STARVATION]` lines and the `Fst` rung detail. The expected trigger point is t≈1200s vclock when min_avail≈3 (below both thresholds: desired_selection=13 for oort, agg_goal=10 for feddance).

### Batch 2 (after Stage F exit)

G.2 ramp: syn_50 → mobiperf, all baselines, 3h runs. The n=48 syn_50 runs already exist for feddance/oort — **don't re-run them**. One long run per baseline confirms E+F+G together.

### Open follow-ups (non-blocking)

- **oort K3b/A2/P3** (§9.1): K3b `overhead_residual` consistently ~0.116 at syn_20 (was PASS at 1.5h → run-length sensitive); A2 KS improving with run length (0.437→0.338, trend toward ≤0.2); P3 `trainer_speed` marginal at n=300. Investigate at Batch 2 long run.
- **feddance A4dur diagnostics gap**: syncfl stamps `avl_state` post-selection; A4dur expects pre-selection → SKIP at syn_50. A4 (duty-cycle fraction) PASS; mechanism unaffected. Fix = move `_avail_stamp_end_states` before selection in syncfl path. Defer to Batch 2.
- **C.3 abandon (90s vclock) SKIP at syn_20/syn_50 n=48**: train ≤60s rarely crosses 90s; felix D.1 fires first. Exercises naturally at oort/refl with longer runs.
- **`observation_lag` + `Aa` rungs (HELD)**: need syn_20 reference data to calibrate. Build once Batch 2 runs exist.
- **G.4 Terminology cleanup** (after Batch 2): rename `oracular_trainer_avail_check` → `_trace_read_avail_check`; update log messages (keep YAML field value `ORACULAR` as-is); consolidate legacy-gate + simUnavail-gate paths. No-op refactor — defer until all baselines confirmed.
- **[Stage H] Real notification lag**: measure on a real felix run before turning `client_notify` back on.

---

## v1 Scope

### Three axes — keep them separate

| Axis | Term | v1 value | Stage H |
|---|---|---|---|
| How the agg learns trainer state | **knowledge model** | **trace-read** (all baselines — agg binary-searches `trainer_event_dict`) | aware → **message-transport** (`avl_*` msgs) |
| When the agg frees a stalled slot | **slot-free timing** | **proactive** (felix: next selection boundary) OR **reactive-90s** (oort/refl/feddance: 90s vclock) | proactive → instant-on-message for aware |
| Which config knob activates | **config-gate** | **legacy-gate** (oort/refl: `trackTrainerAvail.type: ORACULAR`) OR **simUnavail-gate** (felix/feddance: `simUnavailability: True`) | no change |

**"ORACULAR" in code/YAML = only the legacy config field name.** All v1 baselines are trace-read. Use "trace-read" for the knowledge model.

v1 uses trace-read for every baseline; `client_notify` OFF. The proactive/reactive-90s distinction is slot-free timing only, not knowledge.

| | v1 (this spec) | Stage H (FUTURE) |
|---|---|---|
| Knowledge model | trace-read (all) | aware: message-transport; unaware: trace-read |
| When applied | selection boundaries only | continuous / event-scheduled |
| `client_notify` | OFF | ON for aware |

---

## 1. Core decisions (all resolved)

- **Knowledge model:** trace-read for all baselines. One shared trace + `state_at(trainer, vclock)` + one effect path in `AvailabilityMixin`. Per-baseline difference = slot-free timing only.
- **Mid-flight unavailability = compute-completes, gate the send, deliver-late (stale).** Trainer never stops computing. Upload gated at send-time (real) / agg-side buffer at `delivery_ts = max(sct, next_avail_ts)` (sim).
- **Two ledgers, never conflated:** slot ledger (90s vclock abandon, frees `selected_ends`) + delivery ledger (`pending_withheld[end] = delivery_ts`, commits stale through existing staleness gate).
- **Three invariants:** (1) no double-count — freed slot ≠ cancelled update; (2) withheld end stays out of pool until `delivery_ts`; (3) proactive vs reactive-90s = trigger only, identical downstream effect.
- **Busy ≠ unavailable ≠ withheld** — three distinct non-pool states, separate ledgers. Do NOT route busy→UN_AVL.
- **All availability time on the vclock in sim.** Never wall, never frozen per-trainer clock.
- **Config-gated, default OFF** → byte-identical. Legacy-gate (oort/refl) or simUnavail-gate (felix/feddance/fedbuff).

---

## 2. Concepts to keep crisp

- **availability state** (`AVL_TRAIN/AVL_EVAL/UN_AVL`) × **busy?** × **has in-flight update?** — orthogonal, never conflate.
- **send-time gate** (v1) vs **task-start gate** (old real behavior). Compute always completes.
- **slot ledger** (freed at boundary/90s) vs **delivery ledger** (commits at `delivery_ts`).
- **transition instant** vs **observation instant**; v1 lag = until next selection boundary.
- Return fates: on-time / straggler-hold / withheld-then-delivered (stale). No result cancellation.
- `syn_0/syn_20/syn_50` are **2-state** (AVL_TRAIN/UN_AVL only); `_trace_has_avl_eval` guard collapses D.2 for these traces.

---

## 3. Parity rungs (availability tier)

- **A1** `avail_composition`: per-state counts, binned. **A3** `trace_time_base_consistency` — hard gate, CONTROL (dep K3). **A4** `per_trainer_duty_cycle`. **A4dur** duration-weighted TVD (pass: `mean_err ≤ 0.05`, `frac_within_tol(τ=0.10) ≥ 0.95`).
- **withheld_delivery**: dist of `delivery_ts − sct` + staleness + accept/reject split.
- **abandon_timeout**: count + timing of 90s vclock abandons. Fails loud on wall-clock leak.
- **eligible_pool_reduction** (`Aa`): agg-observed fraction vs trace ground truth (HELD — needs run data).
- **observation_lag** (HELD): transition→effect boundary lag.
- **starvation_advance**: vclock jumps under scarcity, count + timing.
- **Ramp:** `syn_0` (regression) → `syn_20` (first validation) → `syn_50` → `mobiperf_*`.

---

## 4. Staged plan

### A ✅ — Substrate
`flame/availability/trace.py` + `AvailabilityMixin` (mixed into all four `TopAggregator`s). Default OFF → byte-identical. Exit: syn_0 clean.

### B ✅ — A3 time-base CONTROL
A3/A4 rungs, origin = `agg_start` both modes. Exit: A3 PASS oort syn_20.

### C ✅ CONFIRMED — Oracular driver + send-time gate + vclock abandon
`AvailabilityMixin` shared effect (not forked per stack): `compute_delivery_ts`, `free_stalled_slot`, `withheld_held_ends`, `_sim_withhold_if_unavail`, `_sim_pop_committable`, `_sim_reinject_ready_withheld`, `_sim_abandon_stalled`, `_emit_withheld_delivery`. felix 49/49 syn_20; oort 39/48 (pre-existing failures, §9.1).

### C.6 ✅ CONFIRMED — Aggregator tracking + plots
`_avail_stamp_end_states` writes `PROP_AVL_STATE` pre-selection (was all-UNKNOWN). A4dur PASS. Five availability plots in `analyze_run.py`.

### D ✅ CONFIRMED — Aware proactive eviction (felix)
`_sim_evict_unavail_inflight` (felix-gated, sim-only). Task-aware eligibility (`get_curr_task_ineligible_trainers`) with `_trace_has_avl_eval` 2-state guard. Real send-gate confirmed (withheld n=7, accept_frac=1.0).

### E ✅ CONFIRMED syn_20 — Sync baselines (feddance + refl)
Syncfl `_distribute_weights` (abandon/evict/stamp) + `_sync_sim_recv_first_k` (withhold + bonus drain). Accept-stale path (E.2: FedAvg has no staleness gate). feddance 46/47 syn_20 (C2 emergent noise; A-rungs/U3/U6/K8/U2 PASS; Challenge 9 ✅).

### F ✅ F.2 CODE-COMPLETE — Starvation / vclock-advance under scarcity

`_next_avail_vclock()` mixin helper returns `min(next_avail_transition_ts, next_pending_withheld_delivery_ts)`.

**F.2 unified pre-selection return-early pattern** (all three aggregators):

```python
_in_flight = getattr(channel._selector, 'selected_ends', set())
num_eligible = len(set(channel._ends.keys()) - set(curr_unavail_trainer_list) - _in_flight)

if num_eligible < threshold:  # oort: desired_selection=13; syncfl: agg_goal=10
    if self.simulated and self.trainer_event_dict is not None:
        _nxt = self._next_avail_vclock()
        if _nxt is not None and _nxt > self._vclock.now:
            self._vclock.advance(_nxt)
            self._sim_abandon_stalled(channel)
            # re-stamp at new vclock
            curr_unavail_trainer_list = self.get_curr_task_ineligible_trainers(task)
            _held = self.withheld_held_ends()
            if _held:
                curr_unavail_trainer_list = list(set(curr_unavail_trainer_list) | _held)
            channel.set_curr_unavailable_trainers(trainer_unavail_list=curr_unavail_trainer_list)
            self._avail_stamp_end_states(channel)
            channel.properties["vclock_now"] = self._vclock.now
        logger.info(f"[SIM_STARVATION] round={self._round} eligible={num_eligible} < {threshold}; vclock→{_nxt}")
    else:
        time.sleep(0.5)
    return  # outer run() loop retries non-blocking

selected_ends = channel.ends(VAL_CH_STATE_SEND, task_to_perform)
```

**What was wrong and what F.2 fixed:**
- Oort had `min_required=5` (50% of agg_goal) + `max_retries=5` + "proceed anyway" fallback + 2s blocking sleep. Fixed: threshold=`desired_selection=int(aggr_num×overcommitment)=13`, unbounded, non-blocking.
- SyncFL fired only at pool=0 post-selection (`if not selected_ends:`). FedDanceSelector returns partial selections (e.g., 3/10) which are non-empty → never caught. Fixed: `agg_goal=10` PRE-selection.
- AsyncFL had `time.sleep(0.5)` at no-recv-ends path in both modes. Fixed: sim path advances vclock.

**n=48 syn_50 smoke (Jun 28):** no stalls ✅, K1 cadence ✅, starvation not populated — correct (staggered n300 schedules: min_avail=24 at n=48 >> both thresholds). At n=10: min_avail=3 < both thresholds → both fire.

**Exit (pending n=10 syn_50 smoke):** `starvation_advance` rung populated for oort AND feddance; `[SIM_STARVATION]` log lines; K1 monotone.

### G.1 ✅ — Ladder integration
`starvation_advance` rung in `checks.py` + `report.py`; 67/67 parity tests (+4 starvation tests).

### G.2/G.3 — Ramp + sign-off (Batch 2, pending Stage F exit)
syn_50 → mobiperf, all baselines, 3h. Per-baseline sign-off.

### G.4 — Terminology cleanup (after Batch 2)
Rename `oracular_trainer_avail_check` → `_trace_read_avail_check`; update log/comment "ORACULAR" references (keep YAML field value); consolidate config-gate paths. No-op refactor — defer until all baselines confirmed passing.

### H (FUTURE) — Message-transport + continuous scheduling
Turn `client_notify` ON for aware baselines: swap trace-read for real `avl_*` trainer→agg messages, processed mid-round. Add event-scheduled vclock clamp. Effect logic unchanged (C.5/D.1 hook was built for this). Re-measure `observation_lag` (must be ≈0).

---

## 5. Challenges / land-mines

Resolved challenges are noted briefly; open ones have full detail.

1. ✅ **Ordering on `delivery_ts`, not `sct`** — withheld commits at `max(sct, next_avail) > sct`. Fixed; U6/U3 validated.
2. ✅ **A3 time-base drift** — hard CONTROL gate; 90s abandon re-clocked to vclock. Do not read A1/A2/A4 until A3 passes.
3. ⚠️ **A2 two-tolerance trap:** `eligible = candidates − in_flight − unavailable`; bimodal sim distribution (avail windows) vs smoother real → KS shape artifact. Means match (real=47.1, sim=47.3); not a mechanism bug. KS improving with run length (0.437→0.338). Expect ≤0.2 at Batch 2 3h run.
4. ✅ **Busy ≠ unavailable ≠ withheld** — three ledgers, never conflated.
5. ✅ **Real send-gate fidelity** — confirmed withheld-then-delivered (not drop); n=7 accept_frac=1.0.
6. ✅ **Determinism** — commit ordered by `(delivery_ts, end_id)`. [Stage H] full vclock tie-break.
7. ✅ **Compound states with carry-over** — oort §4.9 straggler + UN_AVL cross-product covered in unit tests.
8. ✅ **AVL_EVAL inert for oort** — oort dispatches 0 eval; `_trace_has_avl_eval` guard handles 2-state traces.
9. ✅ **Staleness on sync changes cohort** — K8/U2 movement expected; reuse existing threshold, no new scalar.
10. ✅ **Scarcity advance must not skip events** — `_next_avail_vclock()` = min(transitions, withheld deliveries). K1 guarded.
11. ✅ **Regression discipline** — syn_0 byte-identity on every stage before syn_20 validation.
12. ✅ **Library mixin spans examples** — `AvailabilityMixin` + `trace.py` in `flame/`; never re-add example-local copy.
13. ⚠️ **Empty per-task pool corrupts shared `selected_ends`** — `_handle_send_state` cleanup fed availability-filtered pool; guarded for 2-state traces by `_trace_has_avl_eval`. **Unguarded:** 3-state trace where every trainer is simultaneously AVL_EVAL (train pool empty). Root-cause fix (pass full connected pool to cleanup) out-of-scope — revisit when 3-state trace added.
14. ✅ **Scarcity threshold mismatch** — oort `min_required=5` + `max_retries` + proceed-anyway; syncfl pool=0 post-selection. Fixed by F.2 unified pattern.

---

## 6. Dead-ends (settled — do not retry)

- **busy → UN_AVL routing**: ramped in-flight to ~300; busy/unavailable/withheld are three distinct states.
- **Frozen per-trainer clock** (`_sim_now()` = last-dispatch `_sim_send_ts`): stuck UN_AVL forever. Availability reads `_vclock.now`.
- **Wall-clock in sim** for selection gate or 90s abandon: wall barely advances vs vclock → every deadline missed.
- **Per-tick MQTT broadcast**: comms storm + sub-optimal decisions. v1 = oracular pull (zero comms).
- **Ordering withheld commits by `sct`**: re-introduces past-dating. Fixed: order by `(delivery_ts, end_id)`.
- **Forking withhold/abandon per stack**: single shared `AvailabilityMixin` (Challenge 12).
- **A4 counting bare transition fraction**: brittle, blind in oracular mode. Replaced by `A4dur` + `Aa`.
- **Silent-OFF trace-name mismatch** (`avl_events_syn_20` → `syn_20`): fixed in `trace.py` name normalization.
- **D.2 excluding AVL_TRAIN from eval on 2-state traces**: made eval pool permanently empty → `_handle_send_state` cleanup wiped `selected_ends` across both tasks → run hangs exit-code 0. Fixed by `_trace_has_avl_eval` guard.

---

## 7. Known parity failures (non-blocking — investigate at Batch 2)

| Check | Baseline | Status | Verdict |
|---|---|---|---|
| K3b `overhead_residual` | oort | rel≈0.116 consistently | Was PASS at 1.5h → run-length sensitive. Root-cause unclear (P3 gates it at n=300). Investigate at Batch 2. |
| A2 `eligibility` KS | oort | 0.437 (1800s) → 0.338 (3600s) | Shape artifact: bimodal sim vs smoother real distribution. Means match. Improving with run length. |
| P3 `trainer_speed` | oort | ratio=1.153 at n=300 (tol 1.15) | Marginal tail divergence at full cohort. Possibly noise; gates K3b. Investigate at Batch 2. |
| C2 `loss` | feddance | avg_diff≈0.16 (2–3 eval pts) | Emergent early-training noise at α=0.1. K8/C1/utility PASS; not a mechanism gap. |
| U5 `inter-arrival` ρ | feddance | ρ=0.381 at syn_50 (non-enforced) | Worsened vs syn_20 (0.659). Watch at mobiperf. |
| A4dur | feddance | SKIP at syn_50 | syncfl stamps post-selection; diagnostics gap only. Fix deferred (A4 PASS). |
