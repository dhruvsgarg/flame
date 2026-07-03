# High-Fidelity Simulator for FwdLLM -- Design & Staged Build Plan

**Active build (branch `dg/fwdllm_sim_unavail`).** This document is the plan for building a
high-fidelity **simulated-clock** runner for the `fwdllm` example (FedFwd / forward-gradient FL) that
reaches **real<->sim parity** across the **fluxtune / fwdllm / fwdllm++** baselines, at
**100% availability (syn_0)** first, then under **unavailability (syn_20, syn_50, mobiperf)**, then
**beyond syn_0 traces**.

> **DESIGN PRINCIPLE -- concept-parity with async_cifar10, but deviate where the workload demands it.**
> We deliberately **reuse async_cifar10's concepts and machinery** (virtual clock, sct reorder buffer,
> in-flight gate, availability substrate, the parity ladder) wherever they transfer -- that shared
> substrate is the whole point and keeps the two examples auditable against each other. **But fwdllm is a
> fundamentally different workload** (aggregates GRADIENTS not weights, endogenous variance-gated
> dynamic-K commit cadence, `data_id` progress axis, one-message-per-call grad loop, rollback across
> agg-goal cycles), so a verbatim port is sometimes WRONG. When such a fork appears we **(a) evaluate the
> options carefully and make an explicit decision** (do not silently copy async_cifar10, and do not
> silently invent something new either), **and (b) APPEND that decision + its rationale to this document**
> -- so a later reader can tell an intentional fwdllm-specific divergence from an accidental discrepancy,
> and trace any real<->sim gap back to the choice that caused it. Deviation log lives in **§K** (running,
> newest-last); locked cross-cutting ones also surface in §F "Locked principles" / §J.4.

**Current state -- Phase 1 Batches 1-2 COMPLETE (checker + telemetry); Phase-1 syn_0 sign-off run is NEXT.**
Batch 1 landed the two structural clock ports; Batch 2 landed the variance-cadence rung layer + its
telemetry, pytest-gated. **Batch 1 done:** trainer sim path (`time_mode`/`simulated`, additive
`_sim_completion_ts`, no-sleep on the sim path), the async grad loop on the vclock (`_sim_recv_min_grad`
sct reorder buffer + in-flight gate + agg-goal-boundary rollback cleanup), the sync barrier
(`_sync_sim_recv_first_k` first-k-smallest-sct + U6 lags), the sim launchers, and the flag-off
byte-identical regression. **Batch 2 done:** the V1-V5 / DK1-DK3 / G1-G2 rungs in the shared parity engine
(`parity/checks.py` + `run_all_parity` + `CHECK_META`), the cycle-relative cadence telemetry they read
(`cycle_data_id`/`cycle_iteration`/`grad_pool_size`/`cached_v_size` added to the agg_round `extra`,
**§K-D9**), and their pytest gate (known-PASS + known-broken cadence fixtures + a driven emission test);
DK3/G1 emit deferred with a logged SKIP (**§K-D10**). fluxtune oort selection-fidelity is a **validation**
concern (the selection rungs already exist in the STAGE map, §I.4) folded into the sign-off run. Build
detail is history (code is the source of truth; deviation rationale in **§K**). **Still to build:** the
**Phase-1 syn_0 sign-off run** (smoke -> convergence per baseline -> `parity_checks` full battery, §I.6);
**(Batch 3)** the availability *effect* path + `EVENT_AVAIL_CHANGE` emission (Phase 2); **(Batch 4)** full
ladder under unavailability (Phase 3). See **§E** for the batch list and **§I** for the Batch-2 map.

**Prerequisites (read first):**
- [async_cifar10/PARITY.md](../async_cifar10/PARITY.md) -- the parity methodology (the ladder §1, roles/
  tiers/dependency-gating, workflow policy, run-length budget, landed sim mechanisms §3). **fwdllm's
  own rung catalog is PARITY.md §F** (modified rungs, the variance-cadence layer). This doc references
  §F for rung *definitions* and does not duplicate them; it states the fwdllm **build deltas**.
- [async_cifar10/UNAVAILABILITY_DESIGN.md](../async_cifar10/UNAVAILABILITY_DESIGN.md) -- the availability
  substrate this reuses wholesale (ClientAvailability mixin, trace-read effect path, the two flag axes,
  send-gate/deliver-late-stale, two-ledger discipline, starvation self-termination, the ground-truth
  fidelity rungs A6/A7/A8/K11). fwdllm inherits it via the aggregator class chain (§A).

**Starting reality (why this is real work, not a re-label):** FedFwd has **no simulated-clock support
today** -- every launcher yaml is `time_mode: real`, and the trainer emulates delay with a real
`time.sleep` (`trainer/forward_training/FedSgdTrainer.py:481` `_emulate_training_delay`, called at
`:523`; explicit note `trainer/main.py:57-78`). The aggregator machinery for a virtual clock is
*inherited but not driven* (§A). Adding the sim clock + wiring unavailability into the
variance-gated gradient loop is the build.

**Decisions locked (kickoff):**
- A trusted **real** wall-clock run is the reference per baseline (as in async_cifar10). Parity != tuning
  sim to real; real must pass `validate_real` admissibility first.
- The parity checker **extends the shared `async_cifar10/scripts/parity` engine** -- fwdllm rungs are
  *added* (PARITY.md §F is their catalog), the engine is reused.
- The fluxtune / fwdllm / fwdllm++ -> config mapping is **resolved** from the landed launcher configs
  (§C); it is no longer a blocking gap.

---

## §A  Already on the class (inherited) vs. NOT yet wired

The virtual clock is a **flame-core** capability. fwdllm's aggregator is
`examples/fwdllm/aggregator/FedSgdAggregator.py` -> `flame/mode/horizontal/syncfl/fwdllm_aggregator.py::
TopAggregator(AsyncTopAgg)` -> `asyncfl.TopAggregator` -> `syncfl.TopAggregator(ClientAvailability, Role)`.
So both the vclock machinery **and** the availability substrate are already **on the instance** -- the
work is **wiring them into fwdllm's gradient loop + trainer**, not re-implementing them.

### A.1  Aggregator spine -- clock machinery NOW DRIVEN (Batch 1)
| Mechanism (from `asyncfl/top_aggregator.py`) | Status for fwdllm |
|---|---|
| Virtual clock `_vclock`, `_advance_sim_clock` (`vclock = max(vclock, sct)`) | **DRIVEN** by the grad loop (Batch 1) |
| sct-ordered drain + in-flight gate | **DRIVEN** via purpose-built `_sim_recv_min_grad` (K-D4) / sync `_sync_sim_recv_first_k` |
| one-in-flight residence `_sim_hold_busy_slots` | **DRIVEN** at the agg-goal boundary (`_release_sim_slots_at_agg_goal`, K-D5) |
| §4.5/§4.9 sct-gated pool exclusion / carry-over (oort overlay) | available if a baseline selects via oort (fluxtune); **Batch-2 validation** |
| `ClientAvailability` substrate (trace-read gate, two ledgers, proactive evict, starvation advance) | inherited; **NOT yet wired** into the grad loop -- **Batch 3 (Phase 2)** |

### A.2  Selector + duration signal (already wired, baseline-agnostic)
- §S.dur duration helper `client_duration.py::real_client_task_train_duration` is already called in
  `_process_single_trainer_message` to set `PROP_CLIENT_TASK_TRAIN_DURATION`.
- Oort selector (`flame/selector/oort.py`, faithful pacer/UCB/D1-D8) is usable verbatim by fluxtune
  (`async_oort`) -- **Batch 2 validates its rungs fire**; inert for the two `random`-selector baselines.
- Dynamic-KC controller/policy is integrated (`_build_dynamic_kc_metrics`, `_dynamic_kc_controller.step()`);
  fluxtune leaves it **disabled** (fixed K/C). See [docs/dynamic_kc_design.md](docs/dynamic_kc_design.md).
- Analysis `scripts/analysis/analyze_run.py` is fwdllm-aware via `telemetry_manifest.yaml`.

### A.3  Remaining structural work (post-Batch-1)
1. **Endogenous commit cadence** -- variance-gated dynamic-K (PARITY.md §F.1): the emergent layer the
   async_cifar10 ladder does not model. The clock is now driven; **Batch 2 adds the V/DK/G rung layer** that
   verifies the *cadence* on top of it. Most of the per-cycle telemetry it needs already lands via
   `build_agg_round` (§I).
2. **Availability telemetry + effect path.** The fwdllm trainer emits **no `EVENT_AVAIL_CHANGE`**
   (`telemetry_manifest.yaml`: availability = *partial*), so A6/A7/A8 ground-truth rungs can't run yet, and
   the send-gate/delivery-buffer effect path is not wired. Builders exist in `flame/telemetry/events.py`.
   **Batch 3 (Phase 2).**

---

## §B  How FwdLLM differs structurally

Full detail (the crux that makes the ladder applicable, and the variance-cadence rung layer it forces) is
**PARITY.md §F.1**. In one line: fwdllm aggregates **gradients** (JVPs) not weights, its commit cadence is
**endogenous** (variance-gated dynamic-K), and its progress axis is **`data_id`** (committed variance
passes), not raw update count. Gradient *values* are mode-invariant given identical input+perturbation
seed, so fwdllm parity reduces to clock+selection+ordering parity **plus** a new variance-cadence layer.

Concrete anchors (`flame/mode/horizontal/syncfl/fwdllm_aggregator.py`): `aggregate()` var gate +
rollback/`cached_v` at each `_agg_goal` boundary; `total_data_bins=150` (`:271`); force-commit cap
`_max_iter_per_data_id` (`:276`, config key `max_iterations_per_data_id`); `_reselect_each_iteration`
(`:295`, per-iteration reselection for fwdllm++); sync path `_aggregate_grads_sync` (`:1354`).

---

## §C  Baseline matrix -- RESOLVED (from the landed launcher configs)

Read from `expt_scripts/{fluxtune,fwdllm,fwdllm_plus}_n10_smoke.yaml`. Canonical copy + taxonomy mapping
is **PARITY.md §F.2**; repeated here for the build plan.

| baseline | sync/async | selector | agg | tracking_mode / avail | reselection | K (agg_goal) | dynamic_kc |
|---|---|---|---|---|---|---|---|
| **fluxtune** | async | `async_oort` | fedbuff (+server LR, JVP) | `client_notify` (3-tier, mobiperf_3st_50) | -- | 3 | disabled (fixed K/C) |
| **fwdllm** | sync | `random` | fedavg | `default` (unaware) | per-round | 10 (=c; all selected required) | -- |
| **fwdllm_plus** | sync | `random` | fedavg | `oracular` (`_metadata`, mobiperf_2st) | per-iteration (`reselect_each_iteration=True`) | 2 | -- |

**Availability-taxonomy mapping** (UNAVAILABILITY_DESIGN.md two axes):
- **fwdllm** -- unaware at selection (`avail_select_filter=False`), reactive-90s in-flight
  (`proactive_inflight_evict=False`).
- **fwdllm_plus** -- aware at selection via `oracular` trace read (`avail_select_filter=True`),
  reactive-90s in-flight.
- **fluxtune** -- async, aware via `client_notify` (message-transport), reactive-90s in-flight.
  `client_notify` is async_cifar10's **deferred Stage-H** tracking model -> **decision D1**.

This decides: Stage 3 oort rungs run **only** for fluxtune; Stage 5 sync-barrier rungs run for
**fwdllm/fwdllm_plus**; V1/V4 distributions are shaped by each baseline's `max_iterations_per_data_id` /
`var_threshold`. Parity is staged **one baseline at a time** (PARITY.md workflow rule).

---

## §D  The parity ladder for fwdllm

**Rung definitions live in PARITY.md §F** (modified rungs §F.3; variance-cadence / dynamic-K/C /
forward-grad rungs §F.4). This section states only which rungs apply and how they gate.

- **Reused verbatim** (PARITY.md §2): Stage 0 TC1/K10; Stage 1 P3/K1/K7; Stage 2 A1/A2 (+ A3 time-base
  CONTROL, + A6/A7/A8 ground-truth once telemetry is ported); Stage 3 S3/4, A2c and the oort stack
  (Sx/Sd/S2) **only for fluxtune**; Stage 4 T2, per-phase timing, K6, T_mqtt; Stage 8 C1; Stage 9 budget/stop.
- **Modified** for variance-gated dynamic-K: K3a/K3b/K2/U3/K8/U2 -> PARITY.md §F.3 (all re-keyed to the
  **variance-pass boundary** / committed **data_id**).
- **New** -- the variance-cadence layer (the prize): V1-V5 (Stage 6'), DK1-DK3 (Stage 3'), G1-G2 (Stage 7')
  -> PARITY.md §F.4. Localize down, never fix an EMERGENT rung directly; `var_threshold` /
  `max_iterations_per_data_id` are config knobs, not parity levers.
- **Availability** rungs (A1-A5, A6/A7/A8/K11, withheld_delivery, abandon_timeout, starvation_advance,
  eligible_pool_reduction) are inherited from the async_cifar10 substrate and apply once §Stage-Avail
  (below) wires the effect path + telemetry.

---

## §E  Implementation plan -- 4 pytest-gated batches across 3 phases

**Batching principle (why this shape).** Each batch is a *large dev push* that lands a complete
mechanism cluster **plus its pytest coverage together**, ending in ONE "pause to test" that is
**pytest-only** -- no experiment run in the inner loop. Experiment runs (smoke -> convergence) happen
**once per phase, gated behind green pytests**, never as the dev loop. This is affordable because the
parity **rungs are pure functions over telemetry** (`parity_checks.run_all_parity`): we test them by
constructing synthetic real/sim telemetry pairs with known-correct and known-broken deltas and asserting
PASS/FAIL; and the mechanisms (sct reorder buffer, one-in-flight hold, delivery buffering, sync barrier)
are driven with synthetic message sequences. Every batch is **config-gated: flag-off => byte-identical**
to today; the flag-off regression is itself a pytest assertion.

**Phase order (locked at kickoff):** Phase 1 = **100% availability (syn_0) parity for all three
baselines**; Phase 2 = **unavailability (syn_20/50/mobiperf)**; Phase 3 = **beyond syn_0**. Do not start a
phase until the prior phase's sign-off run is banked in §H.

**Two staging changes from the original stage list (below):** (1) the **sync-path port**
(`_aggregate_grads_sync`, old "Stage 5") moves **into Phase 1 Batch 1** -- fwdllm/fwdllm_plus are *sync*,
so their primary commit path is the barrier; they cannot reach syn_0 parity without it. (2) old "Stage 0"
is a *validation gate*, not a batch -- its telemetry is already landed (see Current-state callout).

---

### PHASE 1 -- 100% availability (syn_0), all three baselines

#### Batch 1 -- both structural clock ports: trainer sim path + async grad loop + sync barrier -- ✅ DONE
Landed & pytest-gated (75 green). Trainer additive `_sim_completion_ts` + no-sleep; async
`_sim_recv_min_grad` sct buffer + in-flight gate + agg-goal-boundary rollback cleanup; sync
`_sync_sim_recv_first_k` first-k-smallest + U6 lags; sim launchers. **Build detail is history** -- see
the code (`ef4cb70e` +) and the deviation rationale **§K-D1..D8**. Cold-start map for the code lives in
**§I** (now repurposed as the Batch-2 map). U6 telemetry *emission* deferred (K-D7); the seeded mini-run
feeding `parity_checks` is folded into the Phase-1 sign-off run.

#### Batch 2 -- variance-cadence layer + fluxtune selection fidelity -- ✅ DONE (checker + telemetry, pytest-gated)
Landed: V1-V5 / DK1-DK3 / G1-G2 in `parity/checks.py` (registered in `run_all_parity` + `CHECK_META`),
the cycle-relative cadence telemetry they read (**§K-D9**), DK3/G1 emit deferred with a logged SKIP
(**§K-D10**). Pytest: `test_parity_checks.py` (per-rung known-PASS/known-broken + `run_all_parity`
fwdllm smoke + non-fwdllm-SKIP) and `test_fwdllm_agg_telemetry.py` (driven pre-mutation snapshot). The
fluxtune selection-fidelity rungs already exist in the STAGE map (§I.4) -> they fire in the sign-off run;
no new checker code. Original Batch-2 plan below kept for the rung-by-rung rationale.

- **Variance-cadence rungs** (PARITY.md §F.4): implement V1-V5 / DK1-DK3 / G1-G2 in the shared engine.
  Most dev is checker + the telemetry to feed it: per-cycle `var` trajectory (at each agg-goal),
  iterations-per-`data_id` (realized dynamic-K), force-commit (`max_iterations_per_data_id`) bypass rate,
  and a `cached_v` carry-over diagnostic (V3). Likely roots, confirm via the *lowest broken rung*:
  contributing-set/order divergence (U5/S2 upstream) -> V1 -> K2; grad-pool accumulation order
  (`cached_v` carry-over, `grad_pool.append` order) -> V2 with matched inputs = a true sim bug;
  force-commit rate (V4) = chronic variance divergence, not a separate bug. **DynamicKC: DK3 (policy
  *input*) before DK1/DK2** (inert unless a baseline enables dynamic_kc; fluxtune leaves it off).
- **Selection fidelity (fluxtune only, `async_oort`):** validate A2c/Sx/Sd/S2 + the §S.pacer/§S.temporal/
  §S.dur stack (already landed in `flame/selector/oort.py`) -- validation + any fwdllm-specific wiring, not
  a rebuild. Inert for the two `random`-selector baselines (S-rungs WARN/skip).

**Pause to test (pytest-only):** unit-test each new rung function against **constructed real/sim telemetry
pairs** with known cadence deltas (known-PASS fixtures + known-broken force-commit-rate / var-trajectory
fixtures). Then a seeded mini-run per baseline -> `parity_checks` -> assert applicable rungs PASS. **Exit
rungs:** V1/V2/V5 PASS; K2 (committed-data_ids/vsec) PASS; A2c PASS + Sd binding real~=sim + no pacer
ratchet (fluxtune); DK tracks if enabled.

#### Phase-1 sign-off run (the one time-consuming step of the phase)
Only after Batches 1-2 are green in pytest: one smoke (5 min) then one convergence run **per baseline** at
syn_0 -> `parity_checks` full battery + **C1/C2 at matched `data_id`**. Record in §H. Gate to Phase 2.

---

### PHASE 2 -- unavailability (syn_20/50/mobiperf)

#### Batch 3 (large dev) -- wire the ClientAvailability effect path into the grad loop
- **Effect path:** send-time gate (real) / `delivery_ts = max(sct, next_avail)` buffering (sim); the two
  ledgers; reactive-90s in-flight for all three baselines (no proactive-evict baseline here); starvation
  vclock-advance under scarcity; per-baseline `avail_select_filter` (fwdllm off; fwdllm_plus via the landed
  oracular read; **fluxtune via `trace_read` for v1 -- D1 resolved (b), `client_notify` deferred to
  Stage H**).
- **Availability telemetry (D2):** emit `EVENT_AVAIL_CHANGE` (trainer state machine) +
  `agg_belief_change`/`send_gate_wait` (aggregator belief hooks). All three builders exist in
  `flame/telemetry/events.py`; this unlocks A6/A7/A8/K11.
- **D3 interaction (over-instrument BEFORE trusting cadence):** a withheld/late grad meets the variance
  gate -- does it roll into `cached_v` on a rollback? Does a late grad against a stale `model_version`
  inflate the `var` signal (and thus dynamic-K)? Genuinely new vs async_cifar10 (which commits weights,
  not a variance-gated pool).

**Pause to test (pytest-only):** adapt async_cifar10's availability test patterns
(`scripts/parity/test_availability_rungs.py`, `test_delivery_ledger.py`, `test_starvation_termination.py`)
for fwdllm; unit-test delivery buffering, withheld-then-delivered (not dropped), starvation self-stop
(`"stopping run"`, no `SIM_WALL_CEILING`), and the **syn_0 gate-ON-vs-OFF byte-identical** invariant.
`test_eot_avail_catchup.py` already covers part of this. **Exit rungs:** A1-A5 + A6/A7/A8 PASS.

#### Phase-2 sign-off run: one run per (baseline x trace) -> A1-A5 + A6/A7/A8 PASS; self-stops; withheld
grads delivered not dropped. Record in §H. Gate to Phase 3.

---

### PHASE 3 -- beyond syn_0 (the added-complexity layer, last)

#### Batch 4 -- full ladder under syn_20/50/mobiperf + convergence sign-off
Mostly runs + checker, minimal new dev. V/DK rungs under scarcity (force-commit rate shifts), K8/U2
terminal-state @ matched `data_id`, C1/C2. **Bin V1/V2 by run-fraction** to separate a *constant* mix bias
from a *compounding* variance-feedback loop (the headline fwdllm risk, §G). **Min:** full (3-4 h+).
**Exit:** curves within tolerance at matched data_id; K8/U2 rel within bar; V1/V2 binned residual flat.

---

## §F  Key design decisions (open -- resolved at implementation)

- **D1 -- tracking-mode strategy. RESOLVED: option (b) -- map fluxtune onto `trace_read` for the v1 parity
  pass; `client_notify` is deferred to Stage H.** fluxtune's landed config uses `tracking_mode:
  client_notify`, which UNAVAILABILITY_DESIGN.md treats as the **deferred Stage-H** message-transport model
  (async_cifar10 v1 is all `trace_read`). We take the smaller Phase-2 scope: fluxtune runs an approximated
  `trace_read`-shaped availability model in v1 (aware-at-selection via trace read, reactive-90s in-flight),
  and first-class `client_notify` becomes a later stage once syn_0->unavailability parity is banked.
  fwdllm_plus's `oracular` and fwdllm's unaware paths are already `trace_read`-shaped, so this makes all
  three baselines share one substrate for v1. *(Original doc leaned (a); reversed at kickoff to keep
  Phase 2 tractable.)*
- **D2 -- availability telemetry port.** The fwdllm trainer emits no `EVENT_AVAIL_CHANGE`; the aggregator
  emits no `agg_belief_change`/`send_gate_wait`. All three builders exist in `flame/telemetry/events.py`
  -- port emission (trainer state machine + aggregator belief hooks) so A6/A7/A8/K11 light up. Prereq for
  Stage Avail's exit.
- **D3 -- variance-cadence x withheld/late grads.** Does a withheld grad roll into `cached_v` on rollback?
  Does a late grad against a stale `model_version` distort the `var` signal (and thus dynamic-K)? New to
  fwdllm; over-instrument in Stage Avail before Stage 3.
- **D4 -- eval-delay factor.** FedFwd eval is a forward pass (~= train cost), unlike async_cifar10's
  ~20x-faster eval. Confirm the factor per baseline before Stage 1's per-eval `sct` stamp.

### Locked principles (from async_cifar10, carried over)
1. **Sim does real forward-grad compute, charges modeled time.** GPU runs the real JVP; the agg stamps
   `sct = sim_send_ts + max(real_gpu_s, D) + leg`. Makes gradient values mode-invariant.
   **Never** put overhead on the virtual clock (`vclock = max(vclock, sct)` only).
2. **Progress axis is `data_id`** (committed variance passes). Update count per data_id is the dynamic-K
   random variable V1 validates -- an output to match, not an input to assume.
3. **Variance is an emergent gate; localize, never tune it.** V-rungs gate on U5/S2/V1. `var_threshold` is
   a baseline-defining config knob.
4. **Slot residence must survive variance rollbacks.** Release keys on the agg-goal boundary; the sct
   reorder buffer must not strand a grad across a rollback (the fwdllm-specific extension of §3.resid).
5. **DynamicKC: validate the input before the policy.** DK3 (CONTROL) before DK1/DK2 (MECHANISM). The
   controller is shared/baseline-agnostic -- do not fork it per baseline.
6. **Real is the reference only after admissibility.** A real<->sim gap has two fix directions; if a
   selection-mix residual appears, check whether the **real** input is the divergent side.
7. **syn_0 byte-identical gate OFF; unavailability config-gated.** Reuse the async_cifar10 flag axes and
   the `simUnavailability` gate; do not delete eligibility plumbing.

---

## §G  Open questions / risks / dead-ends

**Open (toward Stage 0):**
- D1-D4 above.
- Which real datasets/traces to use for the fwdllm reference runs (agnews H5 is the smoke default).

**Risks specific to fwdllm:**
- **Variance-cadence is feedback-compounding** -- a tiny per-cycle grad-pool order difference compounds into
  a different iterations-per-data_id (like the async clock residuals that surfaced only at 3 h). Bin V1/V2
  by run-fraction to separate a *constant* mix bias from a *compounding* feedback loop.
- **`cached_v` carry-over** is stateful across cycles; a divergence looks like a variance bug but is
  bookkeeping (V3 DIAG localizes it).
- **One-message-per-call grad loop** vs asyncfl's batch -- slot-hold/release wiring is genuinely different;
  do not copy async_cifar10's release points blindly.
- **Withheld grad x variance gate** (D3) -- a stale late grad may distort `var` and the dynamic-K decision.

**Pre-emptive dead-ends (from PARITY.md / UNAVAILABILITY_DESIGN.md):** overhead > 0 on the vclock;
prediction-only gates that never block; tuning a scalar redispatch/overhead knob instead of fixing the
mechanism; letting eval commits into the train/agg stream; expressing "busy" via the unavailable list.
**fwdllm-new:** do **not** tune `var_threshold`/`max_iterations_per_data_id` to force cadence parity --
those are baseline-defining knobs; a cadence gap is an upstream set/order/clock divergence.

---

## §H  Status

*(Run-log: record per-baseline run dirs, lowest broken rung, root hypothesis, score X/N, JSON path; keep
the run-length budget keyed. One section, updated in place. Batch 1 pytest gate is green; **first run-log
entry lands after the Phase-1 syn_0 sign-off run** -- which is gated behind Batch 2.)*

**Batch 1 landed (done):** all six subtasks complete, `test_fwdllm_*` = 75 green. Code is the source of
truth; the as-built decisions are logged in **§K-D1..D8**. Batch-1 pytest files:
`tests/mode/test_fwdllm_trainer_sim_duration.py` (trainer additive stamp + no-sleep),
`tests/mode/test_fwdllm_sim_grad_loop.py` (async sct buffer / in-flight gate / rollback),
`tests/mode/test_fwdllm_sim_sync_barrier.py` (first-k-smallest + U6 lags), plus the flag-off
`simulated=False` fixes across the four pre-existing fwdllm agg/trainer tests.

**Batch 2 landed (done):** variance-cadence rung layer + telemetry, pytest-gated. `test_fwdllm_*` = 81
green; `test_parity_checks.py` = 63 green (adds the V/DK/G suites); the parity sub-package = 115 green.
As-built decisions in **§K-D9/D10**. Files: `parity/checks.py` (V1-V5/DK1-DK3/G1-G2 + `run_all_parity` +
`CHECK_META`), `parity_checks.py` (re-export shim), `fwdllm_aggregator.py` (cadence `extra` snapshot),
`tests/mode/test_parity_checks.py` + `tests/mode/test_fwdllm_agg_telemetry.py`. **Next §H entry lands
after the Phase-1 syn_0 sign-off run** (§I.6).

---

## §I  Batch 2 implementation map (code-level -- cold-start; NEXT to build)

**Purpose:** durable engineering map so Batch 2 can start cold in a new context. Batch 2 = the
**variance-cadence rung layer** (V1-V5 / DK1-DK3 / G1-G2 in the shared parity engine) + **fluxtune oort
selection-fidelity validation**. It is a **checker + telemetry** batch, NOT a clock port -- the vclock is
already driven (Batch 1). Config-gated nothing new here (the rungs are pure functions over telemetry; they
only run when a `parity_checks` battery is invoked). **Re-grep line numbers before editing.**

### I.1  The prize + the localization discipline (read first)
fwdllm's headline risk (§G) is that the **variance-gated cadence is feedback-compounding**: a tiny
per-cycle grad-pool order difference -> a different `iterations_per_data_id` -> a different committed-
data_id throughput, which may only surface at 3 h. The V/DK/G rungs exist to **catch and localize** that.
**Never fix an EMERGENT rung directly** (PARITY.md §1 / §F.4): walk down to the lowest variance-cadence
rung whose *inputs* are matched. `var_threshold` / `max_iterations_per_data_id` are **baseline-defining
config knobs, not parity levers** -- a cadence gap is always an upstream set/order/clock divergence. Rung
definitions + deps live in **PARITY.md §F.3 (modified K3a/K3b/K2/U3/K8/U2) and §F.4 (new V/DK/G)**; this
section states only the fwdllm **build deltas** (where the checker + telemetry go).

### I.2  Where rungs plug in -- the shared parity engine (VERIFIED)
- Engine: `examples/async_cifar10/scripts/parity/checks.py`. Each rung is a **check function**
  `f(real_agg, sim_agg, ...) -> result_dict`, registered two places:
  1. **`run_all_parity(...)`** (`checks.py:~3431`): add `results["v1_iter_per_data_id"] = v1_...(real_agg,
     sim_agg)` etc. in a new **"Stage 6' variance cadence"** block (after Stage 5 updates, before/near the
     Stage 6 clock block -- deps are Stage-5 ordering + Stage-1 clock).
  2. **The STAGE map** (`checks.py:~3560`, the `{ "rung_name": {"stage":, "role":, "deps": ()} }` dict):
     add each new rung with its role/tier + deps from PARITY.md §F.4 (e.g. `v2_var_trajectory` role
     MECHANISM deps `("v1_iter_per_data_id",)`; `v5_pass_ratio` EMERGENT; `dk3_eligible_metric` CONTROL
     before `dk1_k_trajectory`/`dk2_c_trajectory`; `g1_grad_norm`/`g2_grad_pool_size` EMERGENT).
- Parsed telemetry shape (VERIFIED, `cli.py:74`): the loader hands each check `real_agg["agg_rounds"]` /
  `sim_agg["agg_rounds"]` = the list of **agg_round events**, plus `selection_train`, trainer dicts, etc.
  So the V/DK rungs read the per-cycle agg_round series.
- Tests: `tests/mode/test_parity_checks.py` (the existing 40-test fwdllm/shared battery). Add per-rung
  tests the **same way the batch was gated for Batch 1**: construct synthetic real/sim `agg_rounds` pairs
  with a **known-PASS** fixture and a **known-broken** fixture (e.g. a force-commit-rate skew, a var-
  trajectory divergence) and assert PASS/FAIL. **Pytest-only, no run.**

### I.3  Telemetry sources -- what the emitted agg_round ALREADY carries (VERIFIED)
The per-cycle agg_round event is emitted at `fwdllm_aggregator.py:~1372` (`build_agg_round`,
`flame/telemetry/events.py:108`). **VERIFIED** its `extra` dict carries exactly:
`data_id`, `iteration_per_data_id`, `var`, `var_threshold`, `var_good_enough`, `force_commit_planned`,
`is_async` (plus the generic `agg_goal`, `agg_goal_count`, `staleness`, `trainer_speed_s`,
`contributing_trainers`, `agg_observed_s`). NOTE: `_build_dynamic_kc_metrics` (`:~1190`) computes MORE
(`var_pass_rate`, `max_iter_per_data_id`, `n_eligible_train/eval`, `target_iter_per_data_id`) but that
dict feeds the **dynamic_kc controller**, and is **NOT** in the emitted event -- add fields to the `extra`
dict if a rung needs them. Mapping to rungs:
- **V1 (iterations-per-data_id dist / realized dynamic-K):** derive from the emitted
  `(data_id, iteration_per_data_id)` series -- max `iteration_per_data_id` reached before each `data_id`
  advances. **Present, no new emit.**
- **V2 (per-cycle `var` trajectory):** the emitted `var` per cycle. **Present.**
- **V4 (force-commit / `max_iterations_per_data_id` bypass rate):** the emitted `force_commit_planned`
  boolean (aggregator computes `_force_commit_this_cycle` at `:~1266`, `iter+1 >= max_iter`). **Present --
  V4 = mean(`force_commit_planned`) over cycles. No new emit** (correcting an earlier assumption).
- **V5 (variance-pass ratio per window):** derive from the emitted `var_good_enough` boolean series
  (pass ratio = fraction True per window). **Present** (or add `var_pass_rate` to `extra` if you want it
  precomputed).
- **V3 (`cached_v` pool size over time, DIAG) + G2 (grad_pool size at commit):** `grad_pool` (`:286`) and
  `cached_shared_grad_pool_trainable` (fwdllm `cached_v`, `:287`) sizes are **NOT emitted** -> **NEW emit**:
  add `grad_pool_size` + `cached_v_size` to the agg_round `extra` if V3/G2 are exercised.
- **DK1/DK2 (`_agg_goal` / `dynamic_c` trajectory):** inert for all three current baselines (dynamic_kc
  disabled) -> **DK rungs WARN/skip**. **DK3** (CONTROL, eligible-ends metric fed to the policy) lives in
  the controller metrics dict, not the event -> add `n_eligible_train`/`n_eligible_eval` to `extra` to
  wire DK3. Do NOT fork the shared controller.
- **G1 (per-update grad/JVP norm or SNR):** NOT emitted per update. Grad values are mode-invariant given
  identical input+perturbation seed, so G1 should be ~0 -- a FAIL means a perturbation seed/order leaked.
  **NEW per-update emit** (trainer-side grad norm, into `trainer_round` or a new event) IF G1 is exercised;
  else defer with a **logged skip** (§G "no silent caps"), do not silently drop.

### I.4  fluxtune oort selection-fidelity -- VALIDATION, not a rebuild
The oort stack (`flame/selector/oort.py`, faithful pacer/UCB/D1-D8) is landed and the selection rungs
already exist in the engine STAGE map (`selection_detail`, `residence`, `selection_bias`, `selector_score`,
`preferred_duration`, `participation`, `decision_determinism`, `selection` -- stage 3). Batch-2 work:
**confirm they fire for fluxtune** (`async_oort` selector) and are correctly **inert (WARN/skip) for the
two `random`-selector baselines** (fwdllm/fwdllm_plus). Any gap is fwdllm-specific *wiring* (e.g. a
selection event field the fwdllm aggregator does not stamp), NOT an oort rebuild. Exit: A2c PASS +
`preferred_duration` (Sd) binding real~=sim + no pacer ratchet (fluxtune); S-rungs WARN for `random`.

### I.5  Batch-2 pytest gate (pytest-only, no run) + exit rungs
1. Per new rung: known-PASS + known-broken synthetic `agg_rounds` fixtures in `test_parity_checks.py`
   (force-commit-rate skew, var-trajectory divergence, iter-per-data_id divergence).
2. Any new emitted field (V4 `force_commit`, V3 `grad_pool_size`/`cached_v_size`, G1 grad norm): a small
   emission test asserting the field lands in the agg_round/trainer_round event.
3. Full `test_fwdllm_*` + `test_parity_checks` green; flag-off byte-identical (adding fields must not
   change real-mode behavior).
**Exit rungs (PARITY.md §F.4):** V1/V2/V5 PASS; K2 (committed-data_ids/vsec) PASS; V4 bypass-rate ~= real;
A2c PASS + Sd binding + no pacer ratchet (fluxtune); DK tracks-or-skips as configured.

### I.6  After Batch 2 -- the Phase-1 sign-off run (first §H entry)
Only after Batches 1-2 are pytest-green: one smoke (~5 min) then one convergence run **per baseline** at
syn_0 (the `*_n10_smoke_sim.yaml` launchers + their real siblings) -> `parity_checks` full battery +
**C1/C2 at matched `data_id`**. **Enable modeled delays (D>0) in BOTH real and sim together** for the
convergence run (K-D8: the sim smokes keep D=0 for mechanics-only comparability; the parity run needs
D>0). Record in §H; gate to Phase 2 (Batch 3, availability).

---

## §K  DEVIATION LOG -- fwdllm-specific divergences from async_cifar10 (running, newest-last)

Per the DESIGN PRINCIPLE at the top: every place we chose NOT to copy async_cifar10 verbatim (or chose to
copy it despite a workload difference) is logged here with the options weighed + why. One entry per
decision. Keep appending; do not rewrite history (supersede with a new dated entry instead).

- **K-D1  time_mode default = `"real"` (trainer getattr fallback).** async_cifar10 defaults `time_mode` to
  `"simulated"` (`trainer/pytorch/main.py:174`). **Options:** (a) mirror "simulated" default; (b) default
  "real". **Chose (b):** fwdllm's entire existing config corpus is `time_mode: real` and FedFwd shipped
  with no sim path at all, so a "simulated" default risks silently flipping a manual/legacy config into a
  half-built sim path. Sim variants set `time_mode: simulated` explicitly. **Rationale/where:** J.1(1),
  `FedSgdTrainer.__init__` `getattr(..., "time_mode", "real")`. Revisit if a sim run without an explicit
  yaml value is ever wanted.
- **K-D2  Additive delay model `sim_round_duration = gpu + D`, NOT cifar's `max(gpu, D)`.** cifar "sleeps
  to fill a budget" (wall = max(gpu, D)); fwdllm's real mode sleeps D *on top of* GPU time (flat additive,
  `_emulate_training_delay` docstring + existing `sim_round_duration_s = real_gpu + delay`). **Chose
  additive** so sim sct matches what real wall-clock actually does for THIS workload -- copying cifar's
  max() would desync real<->sim. **Where:** §I CRITICAL CORRECTION, J.1(1) `train_with_data_id`. (This is a
  workload fact, not a preference -- logged so the sct formula difference is traceable.)
- **K-D3  Per-eval sct collapses to the same-iteration train sct (D4).** cifar stamps a DISTINCT eval sct
  = `send + max(gpu, D_eval)` with a ~20x eval speedup (`main.py:1086-1096`). **Options:** (a) port a
  distinct per-eval sct; (b) reuse the train sct. **Chose (b):** in fwdllm eval lives on the AGGREGATOR
  (`eval_model` on the global model); the trainer loop runs `train_with_data_id >> _send_grads` every
  iteration so `_sim_completion_ts` is always fresh, and the trainer's eval message is only a utility
  report, not a separately-clocked commit. No 20x factor (forward-grad "train" IS a forward pass). **Where:**
  D4, J.1(1). Revisit only if a trainer-side distinct eval task is added.
- **K-D4  Grad loop uses a purpose-built `_sim_recv_min_grad`, NOT asyncfl `_sim_recv_min` verbatim.**
  **Options:** (a) call `_sim_recv_min` directly (it already handles ingest/gate/advance); (b) write a
  grad-specific drain reusing the same primitives (`SimReorderBuffer`, `_advance_sim_clock`,
  `_sim_inflight_expected`). **Chose (b):** `_sim_recv_min` does per-commit slot release via
  `_sim_pending_commit`/`sel.all_selected` (asyncfl:615-627) and its withheld/staggered paths key on WEIGHTS
  semantics -- both wrong for fwdllm, which must release slots on the AGG-GOAL boundary and commits grads.
  Reused the primitives, forked the orchestration. **Where:** §J.2, J.1(3).
- **K-D5  Slot release + committed/buffer clear on the AGG-GOAL boundary, not per-commit.** async_cifar10
  releases a slot the moment its update commits (inside `_sim_recv_min`). **Chose boundary release**
  (`_release_sim_slots_at_agg_goal`) because fwdllm commits one grad per call and a `data_id` spans many
  agg-goal cycles with variance-FAIL rollbacks; per-commit release would strand/re-skip a re-contributing
  trainer across a rollback (the §I.6 rollback hole). **Where:** §I.5, §J.2, J.1(3). Locked principle #4.
- **K-D6  `_sim_buffer.clear()` at the boundary DROPS stranded arrived-but-uncommitted grads.** async path
  in cifar carries buffered futures across the barrier. **Chose drop** for rollback-cleanliness (a stranded
  grad was trained on a pre-rollback model_version -> stale next cycle anyway). Benign at syn_0
  (|selected| ~= agg_goal). **OPEN watch-point:** if a rung shows lost updates, switch to commit-then-carry.
  **Where:** J.4, J.2.
- **K-D7  U6 sync-barrier visibility-lag telemetry DEFERRED (compute+stash now, emit later).** The base
  syncfl `_aggregate_weights` finalizes `update_visibility_lag_s` from `_barrier_anchored_lags(_real_round_durs)`
  into a `_round_update_values` dict it then emits (`syncfl/top_aggregator.py:742-786`). fwdllm's
  `sync_collect_and_accumulate_grads` has NO such per-round telemetry struct. **Options:** (a) port the whole
  `_round_update_values` visibility-lag telemetry plumbing into fwdllm's sync path now; (b) compute the
  barrier lags now, stash on `self._sync_barrier_lags_s` + log, defer the telemetry-event emission.
  **Chose (b)** to keep Batch 1 the clock-drive port, not a telemetry port -- the value is COMPUTED and
  available (so the U6 exit-rung wiring in the pytest/parity step can read it), just not yet emitted as an
  event. **Where:** J.1(4). **TODO before the Batch-1 U6 exit-rung check:** either emit `_sync_barrier_lags_s`
  via the telemetry manifest or have `parity_checks` read the stashed value.
- **K-D8  sim smoke launchers keep `enable_training_delays: false` (D=0).** The sim variants
  (`*_n10_smoke_sim.yaml`) mirror the real smokes exactly except `time_mode: simulated` + `_sim` job id.
  **Options:** (a) enable modeled delays (D>0) in the sim smoke so the vclock advances by the modeled
  budget (the "interesting" sim behavior); (b) keep D=0 as in the real smoke. **Chose (b)** for step 5:
  with D=0 the sct = dispatch + real GPU time, which STILL drives the reorder buffer + vclock advance +
  sct ordering -- enough to validate the Batch-1 MECHANICS (does the sim path run/advance/not crash), and
  it keeps the sim smoke byte-for-byte comparable to its real sibling. Modeled-delay-on (D>0) belongs to
  the later convergence/parity runs and MUST be enabled in BOTH the real reference and the sim run together
  (else real sleeps D while sim charges D to the clock -> that's the point; mismatched enable would be a
  false divergence). **Where:** the three `*_n10_smoke_sim.yaml` headers.
- **K-D9  Cycle-relative cadence telemetry (`cycle_data_id`/`cycle_iteration`/`grad_pool_size`/
  `cached_v_size`) added to the agg_round `extra`, NOT reconstructed from the post-mutation series
  (Batch 2).** §I.3 assumed V1 needs "no new emit" -- derive iterations-per-data_id from the emitted
  `(data_id, iteration_per_data_id)`. But those fields are emitted **post-mutation**: on a variance PASS the
  branch advances `data_id += 1` and zeroes `iteration_per_data_id` BEFORE `build_agg_round`, so a commit
  event carries the **next** data_id (and iter 0). Binning cycles by the emitted `data_id` therefore
  attributes each commit to the following bin (off-by-one), and a first-try pass emits nothing under its own
  data_id. **Options:** (a) keep "no new emit" and reconstruct with off-by-one accounting (commit-maps-to-
  previous-data_id, force-commit vs natural-pass both counted) -- fragile, and fwdllm's headline risk (§G) is
  exactly this cadence being feedback-compounding and hard to debug; (b) snapshot the cycle identity BEFORE
  the pass/fail branch and emit it unambiguously. **Chose (b):** `cycle_data_id`/`cycle_iteration` name the
  data_id this cycle worked on + its 0-based attempt index, so **V1 = count(cycles) grouped by
  `cycle_data_id`** is exact for both natural-pass and force-commit paths. The existing post-mutation
  `data_id`/`iteration_per_data_id` are LEFT UNCHANGED (the analyzer's `progress_key` depends on them --
  adding fields, not changing them, keeps real-mode behavior byte-identical). `grad_pool_size` (G2) and
  `cached_v_size` (V3) are snapshotted at the same point (before `_update_state_after_payload_prepared`
  clears `grad_pool` on a commit). All four are `getattr`-guarded like `var_threshold` so test doubles
  without the pools still emit. **Where:** `fwdllm_aggregator.py::_process_aggregation_goal_met` (snapshot
  block before the `if self.var_good_enough:` branch + the `build_agg_round` `extra`);
  `parity/checks.py::_iters_per_data_id` / the V/G rung functions; `tests/mode/test_fwdllm_agg_telemetry.py`
  (`test_cadence_fields_snapshot_pre_mutation`, `test_cycle_data_id_is_pre_advance_on_commit`).
- **K-D10  DK3 (`n_eligible_train`/`n_eligible_eval`) and G1 (per-update grad/JVP norm) emit DEFERRED --
  checker reads-if-present, logged SKIP otherwise (Batch 2).** §I.3 flagged both as candidate NEW emits.
  DK3's eligible-ends metric feeds the DynamicKC policy, which is **disabled for all three current
  baselines** (fluxtune fixed K/C; fwdllm/fwdllm_plus have no controller), so the metric is never consumed;
  G1's per-update grad norm should be ~mode-invariant (grad values depend only on input+perturbation seed)
  and needs a **trainer-side per-update** emit, a larger surface than this checker+agg-telemetry batch.
  **Options:** (a) wire both emits now (add the eligible-count loop to every agg_round; add a per-update
  trainer emit) despite no active consumer; (b) implement the **checker** rungs to read the fields when
  present and **SKIP with a §K-D10-referencing note** when absent, deferring the emit. **Chose (b)** --
  matches the §G "no silent caps" discipline (a SKIP with a logged reason, not a dropped rung) and avoids
  adding inert per-cycle overhead / a trainer-telemetry port to a checker batch. The DK3/G1 check functions
  are still fully unit-tested (synthetic fixtures supply the fields for the PASS/FAIL paths + the absent-
  field SKIP path). DK1 (K trajectory) similarly SKIPs when K is constant across both modes (fixed-K
  baselines) rather than trivially passing. **Wire the emits when a DynamicKC baseline (DK3) or a G1
  exercise is added.** **Where:** `parity/checks.py::eligible_ends_metric_parity` / `grad_norm_parity` /
  `agg_goal_trajectory_parity`; `tests/mode/test_parity_checks.py` (`TestDK3EligibleEndsMetric`,
  `TestG1GradNorm`, `TestDK1AggGoalTrajectory`).
