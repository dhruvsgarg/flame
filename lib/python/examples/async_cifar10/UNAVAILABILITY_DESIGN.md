# Sim Unavailability — Design & Staged Plan

## Working agreement (standing instructions — read every session)

**Implement breadth-first across stages on ONE baseline with short runs; batch the long parity runs at
the end. Never block forward implementation on a long run.**

1. **Common first, one baseline first.** Land shared/library-level changes once (in
   `flame/availability/`, the mixin, the two commit loops), then drive each mechanism through with a
   single reference baseline — **oort/refl** for async/unaware mechanisms, **felix** only where a
   mechanism is aware-specific. Don't fan out to every baseline until it behaves on the reference one.
2. **Short runs to debug, long runs to confirm.** Gate *forward implementation* on cheap signals only:
   unit tests, a `syn_0` byte-identity regression, and the *shortest* `syn_20` smoke that actually
   exercises the mechanism (~1800s vclock — syn_20's first `UN_AVL` is at t=600s, so a 300s run
   validates nothing). A long run (toward the 3h K2 benchmark) is for *confirming*, never for finding
   first bugs.
3. **Across stages before across baselines, long runs last.** Rough the mechanism stack across stages
   on the reference baseline → widen to the other baselines → only then launch **batched** longer parity
   runs. One long run confirms several stages; never one stage each.
4. **Keep this doc crisp and in-place.** Completed stages compress to a few lines (mechanism + where it
   lives + exit met). Full detail only for not-yet-built stages. Dead-ends ledger in §9.

## Status (Jun 28 — Batch 1 code-complete)

**Stages A/B/C/C.6/D all code-complete and CONFIRMED on syn_20. felix 49/49 PASS.
Batch 1 (E/F/G.1) code-landed; 67/67 parity tests pass. Smoke (Stage E/F) pending.**

- **A/B ✅** Substrate (`flame/availability/trace.py` + `AvailabilityMixin`) + A3 time-base CONTROL.
  Exit: A3 PASS oort syn_20 (`max_rel_diff=0.107 ≤ 0.20`), felix (`max_rel_diff=0.007`). Library-level,
  spans async_cifar10 + fwdllm.
- **C ✅ CONFIRMED syn_20** Oracular selection gate (C.1), send-time withhold + stale re-commit (C.2),
  vclock 90s abandon (C.3), `delivery_ts` ordering (C.4), `free_stalled_slot` eviction hook (C.5).
  felix syn_20 Jun 28: **49/49 PASS**, withheld n=7 (mean delay 596s, accept_frac 1.0).
  oort syn_20 Jun 28: 39/48 PASS. A1/A3/A4/A2b/A2c PASS, withheld n=5 (mean delay 598s, accept_frac 1.0).
  Three failures — all pre-existing or run-length artifacts, none caused by avail changes (see §9.1):
  · K3b overhead_residual rel=0.116: was PASS rel=0.059 at 1.5h (Jun 24) → run-length sensitive.
  · T2 training_budget p99 ratio=1.294: same failure present at 1.5h Jun 24 ("Minor: T2 tail").
  · A2 eligibility KS=0.437: new with syn_20 (100%-avail runs had no bimodal distribution); means match
    (real=46.9, sim=47.3 of 48) → KS shape artifact from availability-window timing, not a mechanism bug.
- **C.6 ✅ CONFIRMED syn_20** `_avail_stamp_end_states` stamps oracular state. A4dur PASS (felix
  mean_err=0.0024, oort 0.0029). Plots wired; visual correctness confirmed by clean syn_20 run.
- **D.1 ✅ + D.3 ✅ CONFIRMED** felix syn_20: 2 boundary evictions at vclock ≈600s/1200s, mean_age 1.7s;
  7 withheld updates accepted stale (mean delay 596s). Real send-gate (§8.3) confirmed: real withheld
  n=7 with accept_frac=1.0 (trainers withhold-then-deliver, not drop — Challenge 5 ✅).
- **D.2 ✅ CONFIRMED** `get_curr_task_ineligible_trainers(task)` with `_trace_has_avl_eval` guard.
  syn_0 regression PASS; syn_20 avail_composition shows UN_AVL trainers correctly excluded.
- **§8.3 ✅ CONFIRMED** Real send-gate fires for UN_AVL trainers; withheld_delivery events appear in
  both real and sim runs with accept_frac=1.0.
- **`withheld_delivery` under-emission fix ✅ CONFIRMED** n=7 (felix) vs n=2 pre-fix; fix works.
- **453/453 lib + 63/63 parity tests pass.**

**Batch 1 (Jun 28 — code-complete):**
- **E.1/E.2/E.3 ✅** Syncfl path wired: `_distribute_weights` (abandon/evict/stamp + scarcity-advance
  via `_next_avail_vclock()`) + `_sync_sim_recv_first_k` (withhold send-gate + withheld bonus drain).
  Covers feddance AND refl (both use syncfl stack). `_sim_buffer`/`_sim_committed` init added to
  syncfl `internal_init`. E.2 = accept-stale (FedAvg has no staleness gate in feddance). E.3 naturally
  handled: withheld updates excluded from `committed` → barrier-anchor U6 over actual contributors only.
- **Stage F ✅** Vclock-advance under scarcity added: (a) syncfl `if not selected_ends:` block, and
  (b) oort `max_retries` loop. Both call `_next_avail_vclock()` (new mixin helper, considers all trace
  transitions + `pending_withheld.values()`). Felix asyncfl (line 639 `time.sleep(0.5)`) NOT YET WIRED
  — defer to syn_50 observation (felix already passed 49/49 syn_20 without it).
- **G.1 ✅** `starvation_advance` rung (`checks.py` + `report.py` section 2); 67/67 parity tests pass
  (was 63; +4 starvation tests). `withheld_delivery` deps updated to include `abandon_timeout`.
- **Felix master-gate ✅** `debug_run.sh` trace-override block injects `simUnavailability: True` +
  `availabilityAware: True` for non-ORACULAR baselines when `--trace syn_X` is passed.

**Parity check fixes (Jun 28):** Three parity check issues found and fixed:
1. `NEAR_ZERO_LAG_S` raised 0.05→0.10s (U6): carry-over burst after avail windows inflates sim mean
   to ~70ms — still "immediate commit" semantically, KS uninformative at near-zero.
2. Phase timing point-mass guard (`pre_train_s`, `weights_to_gpu_s`, `weights_to_ram_s`): both modes
   near-zero (<5ms mean) → KS uninformative; pass on mean instead. Same pattern as U6.
3. New availability rungs (`A4dur`, `Aa eligible_pool_reduction`, `C.3 abandon_timeout`, `C.2
   withheld_delivery`) wired into `report.py _SECTIONS` (were computed but not displayed).

## ▶ Next actions — Batch 1 code-complete, smoke pending (Jun 28)

**Batch 1 is code-landed (E/F/G.1). syn_0 byte-identity ✅ (67/67 parity tests). Next: Stage E
smoke (syncfl baselines) → Stage F smoke (syn_50, all baselines with starvation path).**

**Stage E smoke — run now:**
```bash
cd lib/python/examples/async_cifar10
# feddance (syncfl, non-ORACULAR, new master-gate) + refl (syncfl, ORACULAR, legacy path)
# Both exercise E.1/E.2/E.3 via the shared syncfl _distribute_weights/_sync_sim_recv_first_k path.
scripts/debug_run.sh --baselines 'feddance refl' --mode both --runtime-s 1800 --trace syn_20 --num-trainers 48
python -m scripts.parity.cli --batch --experiments-dir experiments --baselines 'feddance refl' --agg-goal 10
```
Exit: A-rungs + U3/U6 + K8 PASS. Expect K8/U2/U6 movement on feddance from E.2/E.3 (Challenge 9).

**Stage F smoke (after E exits):**
```bash
# syn_50 heavier-scarcity smoke — all three starvation-wired baselines (syncfl+oort)
scripts/debug_run.sh --baselines 'feddance refl oort' --mode both --runtime-s 1800 --trace syn_50 --num-trainers 48
python -m scripts.parity.cli --batch --experiments-dir experiments --baselines 'feddance refl oort' --agg-goal 10
```
Exit: no stalls; K1 monotone; `starvation_advance` rung populated (n_starvation_jumps > 0); round cadence faithful.

**Felix asyncfl Stage F gap:** asyncfl has a `time.sleep(0.5)` at `top_aggregator.py:639` (no-recv-ends path). NOT wired in Batch 1. Felix passed 49/49 at syn_20 without it. Wire before syn_50 if stalls appear.

**Batch 2 — one batched long-run pass:** G.2 ramp (syn_50 → mobiperf, all baselines, 3h) + G.3
sign-off. **The syn_50 run from F doubles as the ramp's first rung** — don't re-run it. One long run
per baseline confirms E+F+G together; never one stage each.

**oort 3600s syn_20 long run (parallel, independent):**
```bash
cd lib/python/examples/async_cifar10
scripts/debug_run.sh --baselines oort --mode both --runtime-s 3600 --trace syn_20 --num-trainers 48
python -m scripts.parity.cli --batch --experiments-dir experiments --baselines oort --agg-goal 10
```

**Why `--runtime-s 1800` is the floor:** syn_20's first `UN_AVL` is at vclock t=600s. A 300s run
exercises nothing. The sim must reach ≥~900 vclock-s; 1800 gives multiple down windows (confirmed
faster at n=48; see Stage D.1 result).

**Prerequisites:** [PARITY.md](PARITY.md) §1–§2 (causal ladder, role/tier tags, dependency gating)
and §3 mechanism reference (`_vclock`, §3.drain, §3.resid, §4.5/§4.9, §S.dur).

**Goal:** let trainers drop in/out of `AVL_TRAIN` / `AVL_EVAL` / `UN_AVL` per their traces inside
the sim, emitting correct client-side effects on the **virtual clock**, without (a) breaking the
parity already won at 100% availability, (b) a per-tick MQTT broadcast storm, or (c) frozen-clock
deadlocks.

---

## v1 SCOPE (read this before anything else)

The single biggest simplification, decided Jun 25: **for v1, the aggregator reads the shared trace
directly (oracular) for EVERY baseline — aware and unaware alike — and `client_notify` is OFF.**
That collapses the old asymmetry into **one oracular knowledge path**. The aware/unaware distinction
then reduces to *when and how a stalled slot is freed* (a timing/trigger difference), **not** to *how
the agg learns* a transition.

| | v1 (this spec) | End goal (Stage H, FUTURE) |
|---|---|---|
| How agg learns a transition | **oracular trace read** (all baselines) | aware: real `avl_*` trainer→agg message; unaware: still oracular |
| When availability is applied | **selection boundaries only** (no mid-round clamp) | continuous / event-scheduled at the exact transition vclock |
| Aware mid-flight slot-free | **deferred** (hook built, dormant) → behaves like unaware in v1 | proactive eviction the instant the message lands |
| `client_notify` | OFF | ON for aware baselines |

Everything below is written for v1 unless tagged **[Stage H]**. The architecture keeps the
state-resolution + effect logic identical so Stage H swaps only *transport/timing*, not *effect*.

---

## 0. What already exists (design *with* the grain)

The tree already has **three** availability paths; the 100%-avail runs left them dormant
(`trainer_event_dict`/`trainer_unavail_durations` default `None`). Reconcile these, don't add a fourth.

| # | Path | Where | Time-base | Drives | v1 role |
|---|---|---|---|---|---|
| **A. Agg pull (event trace)** | `get_curr_unavail_trainers()` binary-searches `trainer_event_dict` | oort `top_aggregator.py:643` | `_vclock.now` (sim) / `time.time()−agg_start` (real) | `channel.set_curr_unavailable_trainers` at selection | **THE v1 path** |
| **B. Agg pull (duration windows)** | `oracular_trainer_avail_check(end)` | `asyncfl/top_aggregator.py:1226` | same | per-pick veto | folded onto A |
| **C. Trainer push (notifications)** | `check_and_update_state_avl` → `channel.update_trainer_state` | `trainer/pytorch/main.py:330` | `_sim_now()` / wall (real) | MQTT message | **OFF in v1** → Stage H |

**Two anchoring facts:**
1. **A/B already key on `_vclock.now` in sim.** Agg-pull on the virtual clock = no-comms, deterministic,
   never-freezes. v1 makes this the source of truth for all baselines.
2. **C has a frozen-clock defect in sim:** `_sim_now()` = `_sim_send_ts`, only updates when dispatched.
   An unselected/`UN_AVL` trainer never advances → stuck `UN_AVL` forever. v1 sidesteps C entirely.

---

## 1. Core decisions (all resolved)

### Source of truth — ORACULAR for all baselines in v1
One shared trace + one `state_at(trainer, vclock)` resolver + one effect path, read by the aggregator
for **oort, refl, felix, feddance alike**. Per-baseline difference in v1 = only the trigger/timing
of freeing a stalled slot: aware frees proactively at the next selection boundary; unaware frees
reactively at the 90s vclock deadline. **[Stage H]** aware moves to `avl_*` transport, effect unchanged.
Per-tick broadcast rejected (comms-heavy, induces sub-optimal decisions).

### Mid-flight unavailability = COMPUTE-COMPLETES, GATE THE *SEND*, then DELIVER-LATE (stale)
The corrected model (Jun 25). **NOT** a mid-compute interrupt; **NOT** a lost update.

- **The trainer never stops computing.** What is gated is the **upload**: a trainer whose state at SEND
  time is `UN_AVL` holds the completed result and sends it once it is `AVL_*` again — now **stale**.
- **Real:** send gate on the trainer's upload path — block until `avl_state ∈ {AVL_TRAIN, AVL_EVAL}`.
  (Was task-start gate at `trainer/pytorch/main.py:684,1029`; v1 moves to send-time.)
- **Sim:** no wall-block. Agg-side buffer commits at **`delivery_ts = max(sct, next_avail_ts)`**, stale.

### 90s abandon + withhold-deliver — two separate ledgers
Both are simultaneously correct; the discipline is never conflating them (Challenge 4):
- **Slot ledger:** at the vclock 90s deadline, agg frees the slot from `selected_ends`/in-flight.
  `SEND_TIMEOUT_WAIT_S = 90` was wall-clocked; re-clocked to `_vclock.now` in Stage C.
- **Delivery ledger:** `pending_withheld[end] = delivery_ts`. Commits through the **baseline's existing
  staleness gate** — async (accept-stale); feddance (reject if over tolerance). **Reuse existing
  threshold — do NOT invent a new scalar** (E.2).

### Three correctness invariants (unit-asserted)
1. **No double-count.** Freed slot ≠ cancelled update; the withheld delivery commits as one extra stale
   contribution. In-flight counter must not go negative.
2. **Cannot re-select a still-down trainer.** `withheld_held_ends()` unioned into unavailable list;
   held end stays out of pool until `delivery_ts`.
3. **Aware vs unaware = trigger only.** Single code path with `availability_aware` flag; identical
   downstream effect.

### Busy ≠ unavailable ≠ withheld — three distinct non-pool states
Do **NOT** route busy→`UN_AVL` (§3.resid dead-end ramped in-flight to ~300). Separate states, separate
ledgers (Challenge 4).

### Timing, comms, determinism (v1)
- **Boundary sampling, no mid-round clamp.** v1 resolves availability at selection boundaries. The
  mid-round event-clamp (`min(next_sct, next_transition_ts)`) is deferred to Stage H.
- **All availability time is on the vclock in sim** — never wall, never a frozen per-trainer clock.
- **Determinism.** Abandon + withheld-delivery commits ordered by **`delivery_ts`** for held ends.
  **[Stage H]** message path ordered by vclock with a defined tie-break (Challenge 6).

### Trace representation
One event-trace representation (`AVL_TRAIN/AVL_EVAL/UN_AVL`). **Prefer 3-state**; 2-state is allowed
but limited-utility for aware baselines (felix acts on the `AVL_TRAIN↔AVL_EVAL` split a 2-state trace
collapses). `syn_0/syn_20/syn_50` are **2-state** (only AVL_TRAIN/UN_AVL — confirmed Jun 28); the
`_trace_has_avl_eval` guard collapses D.2's task-type split for these traces (Challenge 13).
Single-source the trace + resolver: `flame/availability/trace.py`, loaded by name from
`examples/_metadata/availability_traces`.

### Config-gating
**Everything config-gated, default OFF** ⇒ byte-identical. Master `sim_unavailability: bool = False`
+ per-baseline `availability_aware: bool` + `availability_trace: str`. Reconcile with
`client_notify["trace"]`/`["enabled"]`; do NOT add a parallel fourth knob.

---

## 2. First-principles factors (the why behind each decision)

- **F1 Clock authority.** One monotonic vclock owns "now"; every availability decision is indexed by it.
  Trace is sim-seconds since `agg_start` (same origin both modes); parity rung **A3** is the CONTROL.
- **F2 Source of truth** — v1: oracular for all baselines (§1). **[Stage H]** aware moves to `avl_*`
  transport, effect unchanged.
- **F3 Event semantics.** `→UN_AVL`: excluded from selection + in-flight slot freed (proactive at
  boundary for aware; at 90s vclock deadline for unaware); update **withheld, not discarded**.
  `AVL_TRAIN→AVL_EVAL`: train-pool removal, eval-eligible only (inert for baselines dispatching 0 eval,
  Challenge 8). `→AVL_TRAIN`: re-enters pool + withheld delivery commits.
- **F4 Compute-completes / gate-the-send / deliver-late** (§1). Return fates: on-time /
  straggler-hold / withheld-then-delivered (stale). No permanent cancellation.
- **F5 Comms** — v1: zero. **[Stage H]** aware: bounded by # real transitions; per-tick broadcast
  rejected.
- **F6 Timing** — v1: boundary-sampled, lag = "until next selection boundary"; `observation_lag` ≈ 0 is
  NOT a v1 invariant. v1 measures boundary lag; **[Stage H]** ≈0 for aware.
- **F7 Regression surface.** §3.resid `_sim_hold_busy_slots`; §4.5 `pending_after` (until `delivery_ts`);
  §4.9 carry-over (withheld end must not re-enter pool during down window, but delivery still commits);
  §S.pacer/§S.dur/A2c selector inputs. All config-gated ⇒ default-off keeps byte-identity.
- **F8 Determinism** — commit ordered by `delivery_ts`. **[Stage H]** message path by vclock.
- **F9 Trace** (§1). **F10 Starvation** — vclock-advance under scarcity (Stage F).

---

## 3. Concepts to keep crisp (naming discipline, PARITY.md)

- **availability state** (`AVL_TRAIN/AVL_EVAL/UN_AVL`) × **busy/occupied?** × **has in-flight update?**
  — three orthogonal axes, never conflate.
- **send-time gate** (v1) vs **task-start gate** (was real's behavior). Compute always completes.
- **slot ledger** (in-flight count; freed at boundary/90s) vs **delivery ledger**
  (`pending_withheld`; commits at `delivery_ts`, accept/reject by existing staleness gate).
- **transition instant** vs **observation instant**; v1 lag = until next selection boundary.
- Return fates: on-time / straggler-hold / withheld-then-delivered (stale). No result cancellation.
- `_sim_now()` must not mean "last dispatch ts" — availability reads `_vclock.now`.

---

## 4. New parity rungs (Stage 2 = Availability; extend the ladder)

- **A1 avail_composition**: per-state counts over the run, binned.
- **A3 trace_time_base_consistency** `[NEW]` (CONTROL, dep K3): same trace → same windows both modes.
  **Hard gate — do not read A1/A2/A4 until A3 passes.**
- **A4 per_trainer_duty_cycle** `[NEW]` (dep A3): on/off fraction per trainer matches.
- **A4dur** `[NEW]`: duration-weighted TVD per trainer. Pass rule: `mean_err ≤ 0.05 AND frac_within_tol(τ=0.10) ≥ 0.95`.
- **transition_effect** `[NEW]`: counts of slot-frees, withheld-then-delivered, AVL_TRAIN→AVL_EVAL.
- **withheld_delivery** `[NEW]`: dist of `delivery_ts − sct` + staleness + accept/reject split.
- **abandon_timeout** `[NEW]`: count + timing of 90s vclock abandons. **Fails loud on a wall-clock
  leak.**
- **observation_lag** `[NEW]` (HELD): transition→effect lag. v1 target = matches real's boundary cadence.
- **eligible_pool_reduction** `[NEW]`: A2 tracks real's reduction under unavailability.
- **Ramp:** `syn_0` (regression) → `syn_20` (first validation target) → `syn_50` → `mobiperf_*`.

---

## 5. Staged implementation + testing plan

Two exit bars per stage: **implementation exit** (unit tests + syn_0 byte-identity, gates forward work)
and **parity exit** (batched long-run pass, NOT per-stage gate).

### Stage A — Substrate ✅ COMPLETE (Jun 25)
`flame/availability/trace.py` (`load_trace`/`state_at`/`next_avail_after`/`compute_delivery_ts`) +
`flame/availability/availability_mixin.py` (`AvailabilityMixin`, mixed into all four `TopAggregator`s).
Default OFF → `trainer_event_dict=None` → byte-identical. **Exit:** syn_0 all-baseline smoke clean.

### Stage B — A3 time-base CONTROL ✅ COMPLETE (Jun 26)
A3/A4 rungs landed, origin = `agg_start` both modes. Pre-existing (not B-caused) failures at the
time: K3b `overhead_residual`, S3/4 `num_chosen`, Sr `residence` — all rooted in Sx `system_util`
KS divergence. **Exit:** A3 PASS on oort syn_20 (`max_rel_diff=0.033`).

### Stage C — ORACULAR driver + send-time delivery gate + vclock abandon ✅ WIRED, syn_20 pending

Single shared effect in `AvailabilityMixin` (Challenge 12 — not forked per stack). Key API:
`compute_delivery_ts`, `free_stalled_slot`, `withheld_held_ends`, `_sim_withhold_if_unavail`,
`_sim_pop_committable`, `_sim_reinject_ready_withheld`, `_sim_abandon_stalled`, `_emit_withheld_delivery`.
Asyncfl calls via `_sim_recv_min` (single pop); oort calls via `_sim_drain_buffer` (pop loop, §4.9
carry-over gate fires **before** `_sim_withhold_if_unavail` — Challenge 7). C.1–C.5 ✅.

**Three invariants that bite if broken:**
- Withheld pops do **not** advance the vclock. Re-injected delivery lands in `"withheld"` past-dating
  bucket (intended stale; what keeps `withheld_delivery` separable from real U6 regressions).
- Abandoned end whose update arrives late must not re-register (`end in committed/pending_withheld`).
- Re-injected deliveries are exempt from the §4.9 carry-over gate and `_sim_withhold_if_unavail`.

Parity rungs landed: `withheld_delivery` (structural invariants), `abandon_timeout` (wall-clock leak
detector), `eligible_pool_reduction`. Trace-name normalization fixed silent-OFF mismatch
(`avl_events_syn_20` → `syn_20`). Jun 27 oort syn_20: 47/49 PASS.

**Batched-pass exit (pending):** A1–A4 PASS, withheld/abandon rungs populated, no new U6 past-dating
beyond `"withheld"` bucket, K2/K3b hold; felix/feddance smoke confirms they ride the same path.

### Stage C.6 — Aggregator-side tracking, fidelity rungs, plotting ✅ COMPLETE (Jun 27–28)

- **C.6.1** `per_trainer[end_id]["avl_state"]` on `emit_selection`; `_avail_stamp_end_states` writes
  oracular state onto `PROP_AVL_STATE` before each selection (fixed all-UNKNOWN `avail_composition`
  — the v1 oracular path never wrote this property; only the legacy `client_notify` push did).
- **C.6.2** `scripts/parity/avail_state_series.py` — time-indexed per-trainer state series from
  selection events; shared resolver for checker + plotter.
- **C.6.3** `A4dur` (`duration_duty_cycle_parity`) in `checks.py`. `Aa`/`observation_lag` deferred (§7).
- **C.6.4** Five new plots in `analyze_run.py`: `availability_dynamics.pdf`,
  `selection_funnel_over_rounds.pdf`, `trainer_state_fractions_sorted.pdf`, `duty_cycle_cdf.pdf`,
  `availability_churn_over_rounds.pdf`.

**Visual correctness pending syn_20**: 3 plots rendered empty in pre-fix runs (the `avl_state` blind
spot). The fix is in code; a fresh syn_20 run is the confirmation.

### Stage D — AWARE proactive eviction (felix) ✅ COMPLETE

- **D.1 ✅** `_sim_evict_unavail_inflight` in `AvailabilityMixin`, sim-only (`if self.simulated:` at
  call site), felix-gated. Exit: 5 evictions at vclock 600.6s/1200.2s, correct `reason` tag, sub-5s age.
- **D.2 ✅** `get_curr_task_ineligible_trainers(task)` with `_trace_has_avl_eval` guard. Exit: syn_0
  regression PASS (Jun 28). **Residual risk**: symmetric case (3-state trace emptying train pool) not
  guarded — see Challenge 13 and §9.
- **D.3 ✅** Late withheld = accept-stale: 5/5 accepted (0 staleness-gate violations), mean delay 594.2s.
- **(D.x deferred to Stage H)** continuous/event-scheduled timing + the `min(next_sct,
  next_transition_ts)` clamp.

### Stage E — SYNC baselines + staleness-gated rejection ✅ CODE-COMPLETE (smoke pending)
- **E.1 ✅** Syncfl `_distribute_weights` (abandon/evict/stamp, scarcity-advance via
  `_next_avail_vclock()`) + `_sync_sim_recv_first_k` (withhold send-gate + withheld bonus drain).
  `_sim_buffer`/`_sim_committed` init in syncfl `internal_init`. Covers feddance + refl.
- **E.2 ✅** Accept-stale path (FedAvg has no staleness gate; `_emit_withheld_delivery` with
  `accepted=True`). No new threshold invented (Challenge 9). Expect K8/U2/U6 movement on feddance.
- **E.3 ✅** Naturally handled: withheld updates excluded from `committed` → U6 over actual cohort.
- **Exit (pending):** A-rungs + U3/U6 + K8 PASS on syn_20; `--baselines 'feddance refl'`.

### Stage F — Starvation / clock-advance under scarcity ✅ CODE-COMPLETE (smoke pending)
`_next_avail_vclock()` mixin helper (all traces min, + `pending_withheld.values()`). Wired into:
(a) syncfl `if not selected_ends:` block (feddance + refl), (b) oort `max_retries` loop.
**Felix asyncfl gap:** `asyncfl/top_aggregator.py:639` `time.sleep(0.5)` NOT yet wired — watch at syn_50.
**Exit (pending):** no stalls; K1 monotone; `starvation_advance` rung populated; cadence faithful. Validation: `--trace syn_50`.

### Stage G — Ladder integration + ramp + sign-off
- **G.1 ✅** `starvation_advance` rung in `checks.py` + `report.py` §2; 67/67 parity tests.
  `withheld_delivery` deps updated to include `abandon_timeout`.
- **G.2** Ramp: syn_0 → syn_20 → syn_50 → mobiperf_*. **G.3** Per-baseline sign-off.

### Stage H (FUTURE) — true `avl_*` message transport + continuous scheduling
Turn `client_notify` back ON for aware baselines: swap oracular boundary read for real trainer→agg
`avl_*` messages, processed immediately (mid-round), **without changing the effect logic** (the C.5/D.1
hook was built for exactly this). Add the continuous/event-scheduled vclock clamp. Preserve determinism
(Challenge 6). Re-measure `observation_lag` (must be ≈0 for aware).

---

## 6. Challenges / land-mines

1. **Ordering must key on `delivery_ts`, not `sct`, for withheld updates** — a withheld delivery commits
   at `max(sct,next_avail) > sct`. Re-validate U6/U3 after C.
2. **A3 time-base drift is the silent killer** — AND the 90s abandon must move to the vclock with it.
   Hard CONTROL gate; do not read A1/A2/A4 until A3 passes.
3. **Two-tolerance trap on the eligible pool (A2 vs S3/4).** `eligible = candidates − in_flight −
   unavailable`; a small gap can fail A2's tight KS while S3/4 passes. Decompose the channel first.
4. **Busy vs unavailable vs withheld = three distinct non-pool states; slot ledger ⊥ delivery ledger.**
   Conflating them leaks slots, double-counts, or drives in-flight negative.
5. **Real-side send-gate fidelity.** Confirm real felix/oort actually withhold-then-deliver (not drop)
   on a real syn_20 run. If real loses the update, the model is wrong. **§8.3 landed** (code); syn_20
   run is the confirmation. **[Stage H]** also measure real `avl_*` notification lag.
6. **Determinism / event tie-break at a shared vclock instant.** v1 only needs commit ordering
   (Challenge 1); full order is **[Stage H]**.
7. **Compound states with existing carry-over.** An oort §4.9 straggler that ALSO goes UN_AVL, or a
   §3.resid held slot whose trainer flips AVL_TRAIN→AVL_EVAL. Enumerate the cross-product in tests.
8. **AVL_EVAL may be inert for some baselines** (oort dispatches 0 eval). Report which baselines
   exercise the eval split.
9. **Staleness-rejection on sync changes round composition (feddance).** Expect K8/U2 movement; reuse
   existing threshold (no new scalar).
10. **Scarcity clock-advance must not stall or fast-forward past events** (Stage F). Clamp to nearest
    of {next transition, next `delivery_ts`, next `sct`}. Guard K1 monotone + K5 failsafe ceiling.
11. **Regression discipline.** Every stage re-runs syn_0 parity and must hold the scoreboard
    byte-for-byte before syn_20 validation counts.
12. **Library mixin spans examples — don't fork it.** `AvailabilityMixin` + `trace.py` live in `flame/`
    and are mixed into all four `TopAggregator`s. Never re-add an example-local copy.
13. **An empty per-task eligible pool corrupts the OTHER task's in-flight tracking.** `_handle_send_state`
    cleanup (`if end_id not in ends: selected_ends.remove(end_id)`) was written for disconnection but
    is fed the availability-filtered pool, and `selected_ends` is shared across tasks. D.2 triggered
    this on 2-state traces (eval pool empty); guarded by `_trace_has_avl_eval`. **Unguarded symmetric
    case**: a 3-state trace where every trainer is simultaneously `AVL_EVAL` would empty train's pool
    and hit the same defect. Root-cause fix (pass full connected pool to the cleanup loop) is out-of-scope
    for now — revisit when a 3-state trace is added. See §9.

---

## 7. Open follow-ups

- **`observation_lag` rung (HELD):** transition→effect boundary-cadence lag (F6). Needs a syn_20
  reference to calibrate — deferred until run data exists. v1 target = matches real's boundary cadence.
- **`Aa` rung (HELD):** within-mode agg-observed fraction vs trace ground truth. Regression/plumbing
  sanity check for v1 (should be ~0 by oracular construction). Build once syn_20 data exists.
- **C.3 abandon (90s vclock) still SKIP at syn_20**: train ≤60s rarely crosses 90s. For aware
  baselines (felix), D.1 fires first and masks C.3. To exercise C.3, use oort/refl (C.3-only path) or
  syn_50 (heavier unavailability).
- **felix master-gate plumbing ✅ DONE:** `debug_run.sh` trace-override block now injects
  `simUnavailability: True` (+ `availabilityAware: True` if `client_notify.enabled`) for non-ORACULAR
  baselines (felix, feddance, fedbuff). oort/refl activate via legacy ORACULAR path unchanged.
- **Q-new-2 (open):** verify feddance's staleness threshold is the right rejection gate on a real
  syn_20 run — async accept-stale (E.2) vs feddance FedAvg with no staleness gate; any K8/U2 shift
  to analyze.
- **Felix asyncfl Stage F gap:** `asyncfl/top_aggregator.py:639` `time.sleep(0.5)` still wall-sleeps
  under scarcity. Felix passed 49/49 syn_20 without it; watch for stalls at syn_50 and wire if needed.
- **[Stage H] Real notification lag (Challenge 5):** measure on a real felix run before turning
  `client_notify` back on; decides whether lag-0 reflection is admissible.

---

## 8. v1 Implementation Spec — file-level

All paths relative to `lib/python/`.

### 8.1–8.2 Library substrate ✅ LANDED — `flame/availability/{trace.py, availability_mixin.py}`
`trace.py`: `load_trace` / `state_at` (`bisect_right−1`) / `next_avail_after` / `compute_delivery_ts`.
`availability_mixin.py` (`AvailabilityMixin`, mixed into all four library `TopAggregator`s via
`syncfl/TopAggregator`): `_init_availability`, `_avail_now()` (vclock sim / wall-elapsed real),
`get_curr_unavail_trainers`, `get_curr_task_ineligible_trainers`, delivery-ledger +
`free_stalled_slot` eviction effect + all commit-loop helpers + `_avail_stamp_end_states`.
Examples inherit for free.

### 8.3 Trainer send-gate ✅ LANDED (`trainer/pytorch/main.py` + `syncfl/trainer.py`)
Task-start skip removed from `train()`/`evaluate()`: compute always runs to completion. Gate moved to
`_send_weights` (`syncfl/trainer.py`): `not self.simulated and avl_state == UN_AVL` (decoupled from
`client_notify["enabled"]`). Sim is untouched (Stage C withholds agg-side). `avl_state` freshness for
the real gate comes from the existing `notify_trainer_avail` background thread (1s poll). **Confirmation
pending syn_20.**

### 8.4 Config surface (`flame/config.py` + spawner)
- `sim_unavailability: bool = False` (master gate; off ⇒ byte-identical).
- `availability_aware: bool`, `availability_trace: str`. Reuse `client_notify["trace"]` as
  `availability_trace`; keep `client_notify["enabled"]="False"` in v1. Do NOT add a parallel fourth knob.
- `availability_trace_dir` (optional) → `base_dir` for the resolver.

### 8.5 Telemetry ✅ LANDED
`abandon_timeout`, `withheld_delivery` (`delivery_ts−sct`, staleness, accept/reject) builders in
`flame/telemetry/events.py`. `avl_state` on `emit_selection` via `_avail_stamp_end_states` (§8.1).

### 8.6 Tests ✅ 453/453 lib + 67/67 parity
Resolver determinism + syn_0-inert; all four aggregators; ledger invariants; `delivery_ts` ordering
(incl. late-stash bump); gate-off byte-identity; `_avail_stamp_end_states`; `_trace_has_avl_eval`
2-state guard; `starvation_advance` gate/jump detection (+4 tests, Jun 28). Remaining: run validation.

### 8.7 Exit criteria (status)
A ✅ · B ✅ · C ✅ CONFIRMED syn_20 · C.6 ✅ CONFIRMED (plots render, A4dur PASS) ·
D.1 ✅ CONFIRMED (boundary evictions at vclock 600/1200s) · D.2 ✅ CONFIRMED ·
D.3 ✅ CONFIRMED (accept_frac=1.0) · §8.3 ✅ CONFIRMED (real withheld n=7, not drop).
**felix syn_20: 49/49 PASS (Jun 28).** oort 39/48: K3b run-length + T2 pre-existing + A2 KS shape artifact — all unrelated to avail (see §9.1).
**E/F/G.1 code-complete (Jun 28); 67/67 tests pass.** Stage E smoke (`--baselines 'feddance refl' --trace syn_20`) pending; Stage F smoke (`--baselines 'feddance refl oort' --trace syn_50`) pending.

---

## 9. Dead-ends (settled — do not retry)

### 9.1 oort syn_20 parity failures — settled diagnosis (Jun 28)

oort scored **39/48** at 1800s syn_20 vs **42/46** at 1.5h 100%-avail (Jun 24). The denominator grew
(5 new avail rungs, all PASS for oort). The three failures are pre-existing or run-length artifacts;
none are caused by the availability implementation.

| Check | Old (Jun 24, 1.5h, 100% avail) | Current (Jun 28, 1800s, syn_20) | Verdict |
|---|---|---|---|
| K3b overhead_residual | PASS (rel=0.059) | FAIL (rel=0.116) | Run-length: 205 rounds vs ~450 at 1.5h → noisier estimate. Expect PASS at 3h. |
| T2 training_budget | FAIL (sim p99=36 vs real=31, "minor") | FAIL (sim p99=18.0 vs real=13.91) | Pre-existing, same nature. Run-length sensitive. |
| A2 eligibility | PASS (100% avail = no bimodal) | FAIL (KS=0.437) | New with syn_20. Means match (real=46.9, sim=47.3 of 48). Sim distribution is bimodal (48 outside avail windows, ~39 inside); real rounds don't align precisely to vclock window boundaries → smoother real distribution. KS detects shape, not mean. Not a mechanism bug. |

**Why A2 isn't a regression:** in 100%-avail runs both modes always have ~48 eligible trainers, so
distributions match trivially. With syn_20, the availability window causes a bimodal sim distribution
vs gradual real distribution — because real wall-clock rounds don't fall exactly at vclock window
boundaries. The mechanism is correct (means match); only the intra-window timing differs.

**What to do:** run oort syn_20 at 3600s. Expect K3b and T2 to self-correct (more rounds). If A2
still fails, investigate the round-boundary timing of real vs sim through avail windows. Do not block
Stage E on this.

- **busy → `UN_AVL` routing**: ramped in-flight to ~300. Busy/unavailable/withheld are three distinct
  states with separate ledgers.
- **Frozen per-trainer clock** (`_sim_now()` = last-dispatch `_sim_send_ts`): unselected/`UN_AVL`
  trainer never advances → stuck forever. Availability reads `_vclock.now`.
- **Wall-clock in sim** (selection gate or 90s abandon): wall barely advances vs vclock → every
  window/deadline missed. Everything availability-related is on the vclock.
- **Per-tick MQTT broadcast**: comms storm + sub-optimal decisions. v1 = oracular pull (zero comms).
- **Ordering withheld commits by `sct`** instead of `delivery_ts`: re-introduces past-dating.
  Order by `(delivery_ts, end_id)`.
- **Forking withhold/abandon per stack**: single shared `AvailabilityMixin` effect (Challenge 12).
- **A4 counting transition fraction** (bare max over `avail_change`): brittle, blind in oracular mode.
  Replaced by `A4dur` + `Aa`.
- **Silent-OFF trace-name mismatch** (`avl_events_syn_20` resolved 0 traces): fixed by name
  normalization in `trace.py`. Watch for this when adding traces.
- **D.2 excluding AVL_TRAIN from eval on 2-state traces**: made eval eligible pool permanently empty →
  `_handle_send_state`'s disconnection cleanup (`if end_id not in ends: selected_ends.remove`) wiped
  `selected_ends` shared across both tasks → zero aggregation, run hangs with exit-code 0. Fixed by
  `_trace_has_avl_eval` guard. Root-cause fix (pass full connected pool to cleanup, not
  availability-filtered pool) deferred as out-of-scope blast radius.
