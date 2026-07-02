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

**Current state (landed on this branch, PRs #63-#69) -- the doc's original "nothing built yet" is stale.**
Stage-0 scaffolding is largely done and is now a *validation gate*, not a build: trainer telemetry
(`build_trainer_round`: `real_gpu_time_s`, `sim_round_duration_s`, `avail_state`, `stat_utility`,
`FedSgdTrainer.py:533`), aggregator telemetry (14 tests), the shared parity-check engine + fwdllm checks
(`test_parity_checks.py`, 40 tests), the oracular availability *read* (`read_trainer_unavailability`,
`fwdllm_aggregator.py:524`), per-iteration reselection, and the opt-in e2e parity harness
(`test_real_sim_e2e_parity.py`). **Still to build (the real work):** (1) the trainer sim path
(`_emulate_training_delay:481` still `time.sleep`; no `time_mode`/`simulated`, no `_sim_completion_ts`);
(2) the aggregator grad loop on the vclock (`_aggregate_grads_async:693` still raw `recv_fifo`; no
`_sim_recv_min`/`_vclock`/`_sim_hold_busy_slots` anywhere); (3) the variance-cadence rung layer; (4) the
availability *effect* path + `EVENT_AVAIL_CHANGE` emission. The build is organized as **4 pytest-gated
batches across 3 phases** (§E).

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

### A.1  Aggregator spine (inherited, baseline-agnostic)
| Mechanism (from `asyncfl/top_aggregator.py`) | Status for fwdllm |
|---|---|
| Virtual clock `_vclock`, `_advance_sim_clock` (`vclock = max(vclock, sct)`) | inherited; **NOT yet driven** by the grad loop |
| §3.drain sct-ordered drain `_sim_recv_min` | inherited; **bypassed** (fwdllm's `_aggregate_grads_async` uses raw `recv_fifo`) |
| §3.resid one-in-flight residence `_sim_hold_busy_slots` | inherited; **bypassed** |
| §4.5/§4.9 sct-gated pool exclusion / carry-over (oort overlay) | available if a baseline selects via oort (fluxtune) |
| `ClientAvailability` substrate (trace-read gate, two ledgers, proactive evict, starvation advance) | inherited; **NOT yet wired** into the fwdllm grad loop |

### A.2  Selector + duration signal (already wired)
- §S.dur duration helper `client_duration.py::real_client_task_train_duration` (intrinsic
  `WALL_SEND_TS - WALL_RECV_TS`) is already called in `fwdllm_aggregator._process_single_trainer_message`
  (`:745`) to set `PROP_CLIENT_TASK_TRAIN_DURATION`. No port needed.
- Oort selector (faithful §S.pacer two-branch pacer, §S.temporal UCB, D1-D8 fixes) is in
  `flame/selector/oort.py`, usable verbatim by fluxtune (`async_oort`).
- Dynamic-KC controller/policy (`flame/selector/dynamic_kc_{controller,policy}.py`) is already integrated
  (`_build_dynamic_kc_metrics`, `_dynamic_kc_controller.step()`); fluxtune leaves it **disabled** (fixed
  K/C) per its config. See [docs/dynamic_kc_design.md](docs/dynamic_kc_design.md).
- Analysis: `scripts/analysis/analyze_run.py` is already fwdllm-aware via `telemetry_manifest.yaml`
  (data_id/iteration_per_data_id progress hierarchy; per-category populate/partial status documented).

### A.3  NOT reusable as-is -- the actual work
1. **Trainer sim path.** `FedSgdTrainer._emulate_training_delay` (`:481`) uses real `time.sleep`; it does
   NOT stamp a modeled completion (`SIM_COMPLETION_TS` / `SIM_CLIENT_TASK_TRAIN_DURATION_S`) or
   `WALL_SEND_TS`/`WALL_RECV_TS` the way async_cifar10's `trainer/pytorch/main.py` does. **Port required.**
2. **Aggregator grad loop is off the virtual clock.** `_aggregate_grads_async` (`fwdllm_aggregator.py:693`)
   calls `channel.recv_fifo(...,1)` directly and commits on wall arrival -- never consults `_sim_recv_min`,
   never advances `_vclock`, never holds in-flight slots. **The single biggest structural port.**
3. **Endogenous commit cadence** -- variance-gated dynamic-K, not fixed-K (PARITY.md §F.1): a new emergent
   layer the async_cifar10 ladder does not model.
4. **Availability telemetry.** The fwdllm trainer emits **no `EVENT_AVAIL_CHANGE`**
   (`telemetry_manifest.yaml`: availability = *partial*), so A6/A7/A8 ground-truth rungs can't run.
   The `avail_change` / `agg_belief_change` / `send_gate_wait` builders exist
   (`flame/telemetry/events.py:225/521`, `task_send` fields) and just need emitting from the fwdllm
   trainer + aggregator.

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

#### Batch 1 (large dev) -- both structural ports: trainer sim path + async grad loop + sync barrier
The single biggest structural batch. Lands three tightly-coupled ports (the trainer's stamped completion
is exactly what the aggregator's sct buffer consumes, so they are tested together). Reference port for the
trainer: async_cifar10 `trainer/pytorch/main.py` already has the full `time_mode`/`simulated`/
`_sim_completion_ts`/`sim_completion_leg_s` plumbing (lines 106/173-174/840-841/1086-1093/1151-1155) --
this is a port of a known-good shape, not a design problem.

- **Trainer** (`FedSgdTrainer.py`): add `time_mode`/`simulated`; replace `_emulate_training_delay`'s real
  `time.sleep` (`:481`) with stamping `_sim_completion_ts = sim_send_ts + max(real_gpu_s, modeled_budget_D)
  + sim_completion_leg_s`; send `SIM_COMPLETION_TS` + `SIM_CLIENT_TASK_TRAIN_DURATION_S` +
  `WALL_SEND_TS`/`WALL_RECV_TS`. Stamp a **per-eval** `sct` (dead-end: reusing the last train `sct`
  past-dates every eval); eval delay ~= train (forward pass), **not** async_cifar10's 20x speedup -- D4.
- **Async grad loop** (`_aggregate_grads_async:693`, fluxtune): replace raw `channel.recv_fifo(...,1)` with
  the inherited **`_sim_recv_min`** (sct-ordered reorder buffer + §3.drain) + **`_sim_hold_busy_slots`**
  (§3.resid, one-in-flight) + `_advance_sim_clock`. **fwdllm processes one message per call**, so key the
  slot release on the **agg-goal boundary** (where `_per_agg_trainer_list` clears), NOT per-message. Keep
  `simCommitOverheadSeconds=0` (overhead on the vclock is a hard dead-end). **Rollback risk:** a data_id
  spans **many** agg-goal cycles -- slot-hold + sct-buffer must survive rollbacks without leaking or
  double-committing a grad; over-instrument `inflight_residence` + per-cycle buffer occupancy.
- **Sync barrier** (`_aggregate_grads_sync:1354`, fwdllm/fwdllm_plus): barrier-anchored visibility-lag
  treatment (PARITY.md §6.u6); vclock advance at the barrier. *(Pulled forward from old Stage 5.)*
- **Launchers:** add `time_mode: simulated` variants of the three `expt_scripts/*_n10_smoke.yaml`.

**Pause to test (pytest-only):** unit tests for (a) trainer stamping formula + per-eval `sct` + a
"no `time.sleep` on the sim path" assertion; (b) the sct reorder buffer & one-in-flight hold over a
synthetic out-of-order message sequence **including a rollback** (no leaked/double-committed grad); (c)
sync-barrier vclock advance + no per-message past-dating. Then one tiny in-process 2-3-trainer **seeded**
run per path -> feed telemetry to `parity_checks`. Full existing `test_fwdllm_*` suite green (flag-off
byte-identical regression). **Exit rungs:** P3 matches; K6 advancing; T2 matched; K1 monotone; K3a/K3b
(per variance-pass) ~= 0; U5 inter-arrival; one-in-flight overlap ~= real; U6 barrier lag real~=sim;
commit_gap ~= 0.

#### Batch 2 (large dev) -- variance-cadence layer + fluxtune selection fidelity
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
the run-length budget keyed. One section, updated in place. No parity run yet -- first entry lands after
Batch 1's pytest gate.)*

---

## §I  Batch 1 implementation map (code-level -- captured during study, before edits)

**Purpose:** durable engineering map so Batch 1 can resume cold. All line numbers are as-of branch
`dg/fwdllm_sim_unavail` at capture time; re-grep before editing. Batch 1 = trainer sim path + async grad
loop vclock drive + sync barrier, all config-gated (flag-off => byte-identical), syn_0 only.

### I.1  Class chain + where the sim substrate already lives (INHERITED, reuse -- don't reimplement)
- Trainer: `examples/fwdllm/trainer/forward_training/FedSgdTrainer.py::FedSGDTrainer(Trainer)` ->
  `flame/mode/horizontal/syncfl/fwdllm_trainer.py::Trainer(Role)`. **The messaging (recv/send) lives in
  the base `fwdllm_trainer.Trainer`, not FedSgdTrainer.**
- Aggregator: `FedSgdAggregator` -> `syncfl/fwdllm_aggregator.py::TopAggregator(AsyncTopAgg)` ->
  `asyncfl/top_aggregator.py::TopAggregator` -> `syncfl/top_aggregator.py::TopAggregator`.
- **`self.simulated` / `self.time_mode` / `self._vclock` (VirtualClock) / `_advance_sim_clock(sct)` are
  defined in `syncfl/top_aggregator.py:199-203, 331`** (`vclock=max(vclock,sct)`; overhead default 0 --
  keep 0). Read `time_mode` from `config.hyperparameters.time_mode` (default "simulated").
- **`SimReorderBuffer` `self._sim_buffer` + `self._sim_committed`** inited in `syncfl/top_aggregator.py:228`
  AND `asyncfl/top_aggregator.py:112`. Async `__init__` also inits `_sim_inflight_expected`,
  `_sim_pending_commit`, `_sim_trainer_budget`, budget running-mean/min, withheld ledgers, cooldown, and
  the flag knobs `sim_sct_ordered_drain` / `sim_inflight_residence` / `sim_clock_jump_clamp` /
  `sim_staggered_redispatch` / `sim_redispatch_gap_s` (`asyncfl:112-195`). **TODO verify fwdllm agg's
  `__init__`/`internal_init` actually chains through the async base so these exist** (else lazy-init guard
  at `asyncfl:323` covers `_sim_inflight_expected` et al. but NOT `_sim_buffer`).
- **Sync-barrier machinery ALREADY EXISTS in the syncfl base and just needs calling:**
  `_sync_sim_recv_first_k(channel, ends, first_k)` (`syncfl/top_aggregator.py:360`) -- drains the selected
  set, commits the `first_k` smallest-sct, advances vclock to each, stamps
  `PROP_CLIENT_TASK_TRAIN_DURATION` from `SIM_CLIENT_TASK_TRAIN_DURATION_S`; and
  `_barrier_anchored_lags(durs)` (`:472`, the U6 helper: `lag_i = max_completion - completion_i`).
- **Async single-message commit machinery:** `_sim_recv_min(channel, recv_ends)` (`asyncfl:314`) returns
  ONE sct-ordered committed `(msg, md)` -- maps onto fwdllm's one-message-per-call loop -- but is coupled
  to WEIGHTS semantics (staggered-redispatch keys on WEIGHTS/WEIGHTS_BYTES; budget from
  TRAINING_BUDGET_S; withheld/eval paths). With gate OFF + defaults OFF these are inert. `_sim_hold_busy_
  slots(channel)` (`asyncfl:1453`) holds busy trainers' concurrency slots until commit (one-in-flight).
- **Reference port for the trainer half:** async_cifar10 `trainer/pytorch/main.py` has the full
  `time_mode`/`simulated`/`_sim_completion_ts`/`sim_completion_leg_s`/per-eval-sct plumbing
  (`:106,173-174,818-845 (train sct),1086-1096 (eval sct),1151-1158`). Use it for the *plumbing shape*,
  **NOT the timing formula** -- see the critical correction below.
- **!!! CRITICAL CORRECTION -- fwdllm uses an ADDITIVE delay model, NOT cifar10's `max(gpu, D)` !!!**
  cifar10's real mode "sleeps to fill a budget": wall = `max(gpu, D)`. **fwdllm's real mode sleeps D ON TOP
  of GPU time** (a flat additive emulated delay -- see `FedSgdTrainer._emulate_training_delay:481` docstring
  "flat additive sleep on top of GPU time, not a budget-minus-actual model" and the EXISTING telemetry
  `sim_round_duration_s = _real_gpu_time_s + _delay_s` at `FedSgdTrainer.py:545`). So for real<->sim parity
  the sim sct MUST be additive too:
  - `D = _emulate_training_delay()`'s modeled value = `training_delay_s / training_delay_factor`
    (÷`speedup_factor`, which is 1.0). NOTE the method is named "training" but models the **eval/forward
    cost** (`training_delay_factor` is the eval speedup); fwdllm's forward-grad "train" IS a forward pass.
  - `sim_round_duration = _real_gpu_time_s + D`  (ADDITIVE -- matches real's sleep-on-top)
  - `_sim_completion_ts = (_sim_send_ts) + sim_round_duration + (leg if simulated)`
  - **Port is minimal because the additive delay is already computed:** existing `train_with_data_id`
    already does `_delay_s = self._emulate_training_delay()` (`:523`) and `sim_round_duration_s =
    _real_gpu_time_s + _delay_s` (`:545`). The port just (a) makes `_emulate_training_delay` NOT sleep in
    sim mode but still RETURN D, and (b) stamps `_sim_completion_ts` from that same `sim_round_duration_s`.
  - **Latent bug to fix while here:** `_emulate_training_delay` gates on `training_delay_enabled == "True"`
    (STRING compare, `:488`) but the config schema types it as `bool` (`config.py:197`, default False). So
    with a real bool the delay NEVER fires. Normalize the check (`str(...)=="True" or ... is True`) or the
    modeled D is silently 0 and real==sim only trivially. Verify what the launcher actually passes.

### I.2  Trainer sim-path port -- exact edit sites
1. **`fwdllm_trainer.Trainer._fetch_weights` (`:168`, msg parse ~`:208-320`)** -- currently reads ROUND/
   WEIGHTS/DATA_ID but NOT sim stamps. ADD: `self._sim_send_ts = msg.get(MessageType.SIM_SEND_TS)` and
   `self._wall_recv_ts = time.time()` on model receipt. (Base syncfl trainer already does the analogous
   `_sim_send_ts = msg[SIM_SEND_TS]` at `syncfl/trainer.py:257` -- fwdllm's trainer doesn't extend it, so
   port the read.)
2. **`FedSgdTrainer.__init__` (`:115`)** -- config knobs at `:176-182` (`training_delay_enabled`,
   `training_delay_s`, `training_delay_factor`, `speedup_factor=1.0`). ADD:
   `self.time_mode = config.hyperparameters.time_mode` (default "simulated"),
   `self.simulated = self.time_mode == "simulated"`, `self.sim_completion_leg_s` (default 0.0),
   init `self._sim_send_ts=None`, `self._sim_completion_ts=None`, `self._sim_round_duration_s=None`,
   `self._wall_recv_ts=None`.
3. **`FedSgdTrainer._emulate_training_delay` (`:481`, called `:523`)** -- currently `time.sleep(_sleep_s)`.
   GATE: when `self.simulated`, DO NOT sleep; still RETURN the modeled delay D (so `_delay_s` stays correct
   for the additive `sim_round_duration_s`). Fix the `== "True"` string-gate bug (see CRITICAL CORRECTION).
4. **`FedSgdTrainer.train_with_data_id` (`:502`)** -- already computes `_delay_s` (`:523`) and
   `sim_round_duration_s = _real_gpu_time_s + _delay_s` (`:545`, ADDITIVE). ADD: store
   `self._sim_round_duration_s = _real_gpu_time_s + _delay_s` and
   `self._sim_completion_ts = (self._sim_send_ts or _fallback) + self._sim_round_duration_s + (leg if
   simulated)` for `_send_grads` to read. **Per-eval sct (D4):** fwdllm eval is a forward pass ~= train
   cost (NOT cifar's /20; `training_delay_factor` already encodes the modest eval speedup). If the trainer
   performs a distinct eval task, stamp eval's OWN sct = `_sim_send_ts + (real_eval_gpu + D_eval)` -- do NOT
   reuse the last train sct (past-dates every eval, poisons the reorder-buffer key, cifar `:1086-1096`).
   **VERIFY:** in fwdllm the *aggregator* runs `eval_model()` (`fwdllm_aggregator.py:1391`) on the global
   model after a variance pass; the trainer's `task_to_perform=="eval"` only sends MODEL_VERSION+STAT_UTILITY
   (`_send_grads:522-526`). So confirm whether a trainer-side eval sct is even needed for fwdllm, or whether
   D4 collapses to "train sct only" (likely the latter -- eval lives on the aggregator, off the grad clock).
5. **`fwdllm_trainer.Trainer._send_grads` (`:438`, msg built `:506-521`)** -- ADD to the train msg:
   `SIM_COMPLETION_TS=self._sim_completion_ts`, `SIM_CLIENT_TASK_TRAIN_DURATION_S=self._sim_round_duration_s`,
   `TRAINING_BUDGET_S=<modeled D>`, `WALL_SEND_TS=time.time()`, `WALL_RECV_TS=self._wall_recv_ts`. The eval
   msg (`:522-526`) also needs `SIM_COMPLETION_TS`/`WALL_*` for the per-eval sct.

### I.3  Aggregator dispatch stamping (both distribute paths) -- exact edit sites
- **`_distribute_weights_sync` (`:1777`, per-end send loop `:1849-1900`, `channel.send` `:1899`)** and
  **`_distribute_weights_async` (`:1904`, `channel.send` `:2008`)**. In each per-end loop, when
  `self.simulated`: set `_sst = self._vclock.now`; inject `payload[MessageType.SIM_SEND_TS] = _sst` (or a
  per-end copy -- payload is shared across ends, so stamp per-end or set the channel prop);
  `channel.set_end_property(end, PROP_SIM_SEND_TS, _sst)`. **Async only:** populate the gate
  `self._sim_inflight_expected[end] = _sst + <budget>` (budget = running-mean/min floor, see
  `asyncfl:1671` for the weights-path analog). `PROP_ROUND_START_TIME` already set at `:1866`.
  `PROP_SIM_SEND_TS`/`PROP_SIM_COMPLETION_TS` imported from `flame.selector.properties`.

### I.4  Aggregator grad-loop vclock drive -- exact edit sites + the fwdllm-specific wiring
- **Async: `_aggregate_grads_async` (`:693`)** -- currently `next(channel.recv_fifo(ends,1,timeout=...))`
  at `:719`. REPLACE (sim branch only) with an sct-ordered buffered pop. Design decision (see §G risk
  "one-message-per-call vs batch"): reuse the shared primitives (`SimReorderBuffer`, `_advance_sim_clock`,
  `_sim_inflight_expected` gate) but wire them for the grad loop rather than calling weights-coupled
  `_sim_recv_min` verbatim -- ingest all ready grad msgs into `_sim_buffer` keyed by
  `msg[SIM_COMPLETION_TS]`, pop min-sct committable, `_advance_sim_clock(sct)`, return one `(msg, md)`.
  Real branch unchanged.
- **Slot hold/release (one-in-flight):** call `_sim_hold_busy_slots(channel)` on dispatch; **release keys
  on the AGG-GOAL BOUNDARY**, i.e. where `_process_aggregation_goal_met` clears
  `self._per_agg_trainer_list = []` (`:1271`) and resets `_agg_goal_cnt=0` (`:1279`) -- NOT per message.
  (The doc's §3.resid release-point warning: do not copy asyncfl's per-batch release.)
- **Sync: `_aggregate_grads_sync` (`:1354`) -> `sync_collect_and_accumulate_grads` (`:1313`)** -- the drain
  loop `channel.recv_fifo(channel.ends(), num_min_req, timeout=...)` at `:1336`. REPLACE (sim branch) with
  `committed = self._sync_sim_recv_first_k(channel, channel.ends(), num_min_req)` then feed each committed
  `(msg, md)` through `_process_single_trainer_message`. Apply `_barrier_anchored_lags` for U6 telemetry.
- **Commit path already computes intrinsic duration:** `_process_single_trainer_message` (`:745`) at
  `:837` calls `real_client_task_train_duration(msg, sent_ts, timestamp)` (WALL_SEND-WALL_RECV) and sets
  `PROP_CLIENT_TASK_TRAIN_DURATION` -- keep; this is the selector/telemetry duration (server-overhead-free).

### I.5  ROLLBACK-SAFETY (the fwdllm-specific §3.resid risk -- D-note)
`_process_aggregation_goal_met` (`:1081`): `self.aggregate(self._round)` (`:1153`) runs the variance gate.
**PASS** (`var_good_enough`, `:1170`) -> advance `data_id`, `iteration_per_data_id=0`. **FAIL** (`:1237`) ->
retry SAME `data_id`, `iteration_per_data_id += 1` (the ROLLBACK path). Both branches clear
`_per_agg_trainer_list` (`:1271`) and reset `_agg_goal_cnt=0` (`:1279`). So a `data_id` can span MANY
agg-goal cycles. **Invariant to preserve:** the sct reorder buffer + slot-hold must not strand or
double-commit a grad across a rollback -- release slots and drain/clear the buffer's committed set at the
agg-goal boundary each cycle. `_sim_committed` is cleared by the async base `_reset_agg_goal_variables`
(`asyncfl:231-232`) but fwdllm's own reset (`:453`) does NOT -- **verify/port this clear into the fwdllm
agg-goal boundary** or the second cycle on a rolled-back data_id sees stale committed marks. Over-instrument
`inflight_residence` + per-cycle buffer occupancy.

### I.6  MessageType / API facts -- VERIFIED (capture time)
- **CONFIRMED** MessageType keys exist (`flame/mode/message.py:82-87`): `SIM_SEND_TS=34`,
  `SIM_COMPLETION_TS=35`, `SIM_CLIENT_TASK_TRAIN_DURATION_S=36`, `TRAINING_BUDGET_S=37`, `WALL_SEND_TS=38`,
  `WALL_RECV_TS=39`.
- **CONFIRMED** `SimReorderBuffer` (`flame/sim/virtual_clock.py:52`): `.add(end, sim_completion_ts,
  payload)`, `.has(end)`, `.pending_ends()->set`, `__len__`, `.peek_min_ts()->Optional[float]`,
  `.pending_after(ts)->set`, `.pop_min()->Optional[(end, ts, payload)]`, `.discard(end)`, `.clear()`.
  `VirtualClock`: `.now` (property), `.advance(ts)`, `.reset()`.
- **CONFIRMED** fwdllm agg chains the sim substrate: `fwdllm_aggregator.internal_init` (`:239`) calls
  `super().internal_init()` (`:241`) -> asyncfl `internal_init` (`asyncfl:83`, inits `_sim_buffer` at
  `:112`) -> syncfl base. So `_sim_buffer`/`_sim_inflight_expected`/`_vclock`/`simulated` all exist. (Role
  pattern uses `internal_init`, NOT `__init__`.)
- **ROLLBACK HOLE CONFIRMED (act on I.5):** fwdllm's `_reset_agg_goal_variables` (`:450`) does NOT clear
  `_sim_committed` (the async base's does, `asyncfl:231-232`). fwdllm's real agg-goal boundary is
  `_process_aggregation_goal_met` (clears `_per_agg_trainer_list`/`_agg_goal_cnt` directly at `:1271,1279`).
  **=> Port must explicitly `_sim_committed.clear()` + drain/clear any stranded `_sim_buffer` entries at that
  boundary each cycle**, so a rolled-back data_id's next cycle starts clean.

### I.7  Launcher configs
Add `time_mode: simulated` variants of `expt_scripts/{fluxtune,fwdllm,fwdllm_plus}_n10_smoke.yaml`
(current default is `time_mode: real`). Keep the real ones for the reference runs.

### I.8  Batch 1 pytest gate (pytest-only, no experiment run)
1. **Trainer stamping** (extend `tests/mode/test_fwdllm_trainer_sim_duration.py`): `_sim_completion_ts ==
   _sim_send_ts + max(gpu,D) + leg`; per-eval sct distinct from last train sct; NO `time.sleep` on the sim
   path (monkeypatch/assert). 2. **sct reorder buffer + one-in-flight** (new test): drive the grad loop
   with a synthetic out-of-order message sequence incl. a variance-FAIL rollback cycle -> assert commit
   order = sct order, vclock monotone, no leaked/double-committed grad across the rollback, slots released
   at agg-goal boundary. 3. **Sync barrier** (new test): `_sync_sim_recv_first_k` commits the k smallest
   sct; `_barrier_anchored_lags` correct; vclock advances to k-th; no per-message past-dating.
   4. **Flag-off byte-identical:** `time_mode: real` path unchanged. 5. Full existing `test_fwdllm_*` green.
**Exit rungs:** P3, K6, T2, K1 monotone, K3a/K3b~=0, U5, one-in-flight overlap~=real, U6 barrier lag, commit_gap~=0.

---

## §J  Batch 1 EXECUTION PROGRESS (resume point -- uncommitted working tree)

**Status:** Batch 1 in progress on branch `dg/fwdllm_sim_unavail`, **NOT committed**. Steps 1-2 of the
subtask list (trainer sim path, aggregator dispatch stamping) are **landed in the working tree and
`py_compile`-clean**. Step 3 (async grad loop drive) is **designed in full below but NOT yet written**.
Steps 4-6 (sync barrier, launchers, pytests) untouched. All line numbers below are as of the paused
working tree -- **re-grep before editing** (edits already shifted them from §I's capture).

**RESUME CHECKLIST (start a new session here -- point at this §J):**
- [x] **1. Trainer sim path** -- landed (J.1(1)).
- [x] **2. Aggregator dispatch stamping** -- landed (J.1(2)).
- [x] **3. Async grad loop drive** -- LANDED (working tree, imports clean in `dg_flame`). `_sim_recv_min_grad`
  + `_aggregate_grads_async` mode-branch + `_release_sim_slots_at_agg_goal` boundary cleanup. See §J.2
  (design) + J.1(3) below (as-landed sites).
- [x] **4. Sync barrier** -- LANDED (working tree, imports clean in `dg_flame`). `_sync_sim_recv_first_k`
  sim-drive + `_barrier_anchored_lags` stash in `sync_collect_and_accumulate_grads`. See J.1(4). **U6
  full telemetry emission DEFERRED -- §K-D7.**
- [x] **5. Launchers** -- LANDED. `{fwdllm,fluxtune,fwdllm_plus}_n10_smoke_sim.yaml` created (yaml-parse
  clean): identical to the real smokes except `time_mode: simulated` + `_sim` job id. `enable_training_delays`
  kept false for direct comparability -- see §K-D8.
- [ ] **6. Pytest gate** -- trainer stamping + no-sleep, sct reorder buffer incl. rollback, sync barrier;
  full `test_fwdllm_*` green (§I.8 + §J.3). Pytest-only (no broker in this env).

### J.1  LANDED (in working tree, uncommitted, compiles)

**(1) Trainer sim path -- DONE.**
- `examples/fwdllm/trainer/main.py:76` -- replaced the "no simulated-clock support" warning block with
  `config.hyperparameters.time_mode = _cli_args.time_mode` (Hyperparameters is `extra=allow`, verified
  attr-set works). So the trainer reads time_mode from config uniformly with the aggregator (runner injects
  the agg's at `launch/runner.py:128`). **Deviation from §I.2:** default is `"real"` via
  `getattr(..., "time_mode", "real")` (NOT "simulated") -- conservative for fwdllm's all-real existing
  configs; the launcher/CLI default is also "real". Sim variants set `time_mode: simulated` explicitly.
- `FedSgdTrainer.__init__:184-208` -- added `self.time_mode`/`self.simulated`/`self.sim_completion_leg_s`
  + inited `_sim_send_ts`/`_sim_completion_ts`/`_sim_round_duration_s`/`_wall_recv_ts = None`.
- `FedSgdTrainer._emulate_training_delay:500` (was :481) -- fixed the `== "True"` string-gate bug
  (`_enabled = self.training_delay_enabled in (True, "True", "true")` at :516); when `self.simulated` DO
  NOT `time.sleep` but STILL return modeled D. Real mode unchanged (still sleeps D).
- `FedSgdTrainer.train_with_data_id:560-572` -- ADDITIVE stamp (matches §I CRITICAL CORRECTION, NOT
  cifar's max(gpu,D)): `self._sim_round_duration_s = _real_gpu_time_s + _delay_s`; when simulated,
  `self._sim_completion_ts = (_sim_send_ts or time.time()) + _sim_round_duration_s + sim_completion_leg_s`.
- `fwdllm_trainer._fetch_weights:210-217` -- reads `self._sim_send_ts = msg.get(SIM_SEND_TS)` +
  `self._wall_recv_ts = time.time()` on model receipt.
- `fwdllm_trainer._send_grads:530-556` -- train msg now sends `SIM_COMPLETION_TS`,
  `SIM_CLIENT_TASK_TRAIN_DURATION_S`, `TRAINING_BUDGET_S` (all `= _sim_round_duration_s`; additive model
  has no separate budget), `WALL_SEND_TS=time.time()`, `WALL_RECV_TS=_wall_recv_ts`. Eval msg sends
  `SIM_COMPLETION_TS` + `WALL_*`. **D4 RESOLVED:** the trainer loop is
  `task_get >> train_with_data_id >> put(_send_grads)` every iteration
  (`fwdllm_trainer.py:817-828`), so `_sim_completion_ts` is ALWAYS fresh from the same iteration; fwdllm
  eval lives on the AGGREGATOR (`eval_model`), the trainer eval msg is only a utility report -> **eval
  reuses the same-iteration train sct, no distinct per-eval sct needed** (D4 collapses to "train sct only").

**(2) Aggregator dispatch stamping -- DONE.**
- `fwdllm_aggregator.py:64` -- `from flame.selector.properties import PROP_SIM_SEND_TS` (the oort import
  block does NOT export it; `PROP_SIM_SEND_TS="sim_send_ts"`, `PROP_SIM_COMPLETION_TS="sim_completion_ts"`).
- `_distribute_weights_sync:1850` -- before the per-end loop: `_round_now = self._vclock.now` (sim);
  inject `SIM_SEND_TS=_round_now` into both `payload_with_weights`/`payload_without_weights` (same instant
  for the whole sync barrier); inside the loop `channel.set_end_property(end, PROP_SIM_SEND_TS, _round_now)`.
  No gate (`_sim_inflight_expected`) on the sync path.
- `_distribute_weights_async:1993, 2039-2041` -- same stamp into `payload_weights`/`payload_var_bad`;
  inside the loop ALSO arms the gate: `_budget = self._sim_trainer_budget.get(end, self._sim_budget_min)`
  then `self._sim_inflight_expected[end] = _round_now + _budget`.
- All sim attrs (`_vclock`, `simulated`, `_sim_buffer`, `_sim_committed`, `_sim_inflight_expected`,
  `_sim_trainer_budget`, `_sim_budget_min=12.0`) confirmed inited via `internal_init` chain
  (`fwdllm_aggregator.internal_init:239` -> `super().internal_init()` -> asyncfl `internal_init:83`).

**(3) Async grad loop drive -- DONE** (working tree; imports clean in `dg_flame`, not yet pytest-run).
- `fwdllm_aggregator.py:50` -- import `_SIM_GATE_MAX_PASSES, _SIM_ORDER_SLACK_S` from the asyncfl module.
- `_sim_recv_min_grad:696` -- new method, body per §J.2 (ingest ready msgs into `_sim_buffer` keyed by
  SIM_COMPLETION_TS -> in-flight gate -> pop min-sct -> clock-jump-clamp `_advance_sim_clock` -> learn
  budget -> mark `_sim_committed`). Returns ONE `(msg, md)` or `(None, ("", now))`. Does NOT touch selector
  slots (all slot/buffer/committed clearing is boundary-only).
- `_release_sim_slots_at_agg_goal:788` -- new boundary helper: clears `_sim_committed` + `_sim_buffer` +
  `_sim_inflight_expected`; async-only `_sim_hold_busy_slots(channel)` (empty held set => releases slots).
- `_aggregate_grads_async:840-846` -- sim branch calls `_sim_recv_min_grad(channel, ends(VAL_CH_STATE_RECV))`;
  real branch unchanged (`next(recv_fifo(...,1))`).
- `_process_aggregation_goal_met:1441-1442` -- after `channel.cleanup_recvd_ends():1434`, when simulated,
  calls `_release_sim_slots_at_agg_goal(channel, is_async)` (reached by BOTH variance-PASS + FAIL branches).

**(4) Sync barrier drive -- DONE** (working tree; imports clean in `dg_flame`, not yet pytest-run).
- `sync_collect_and_accumulate_grads:1448` -- sim branch (`:1479`) calls
  `committed = self._sync_sim_recv_first_k(channel, channel.ends(), num_min_req)` (base at
  `syncfl/top_aggregator.py:360`: commits k-smallest-sct, advances vclock to k-th, stamps
  PROP_CLIENT_TASK_TRAIN_DURATION), then feeds each `(msg, md)` through `_process_single_trainer_message`,
  breaking at `_agg_goal_cnt >= _agg_goal`. Real branch moved into the `else:` -- byte-identical.
- `:1502` -- stashes `self._sync_barrier_lags_s = self._barrier_anchored_lags(_barrier_durs)` where
  `_barrier_durs` = each committed msg's `SIM_CLIENT_TASK_TRAIN_DURATION_S` (dispatch-relative completion,
  sim's analog of WALL_SEND - dispatch). **U6 emission deferred (§K-D7): fwdllm's sync path has no
  `_round_update_values`/visibility-lag telemetry struct like the base -- the lag is computed + logged
  (`[SYNC_SIM_BARRIER]`) and stashed on the instance, but not yet emitted as a telemetry event.**

### J.2  Async grad loop drive -- design (LANDED, see J.1(3) for as-landed sites)

Add a new method `_sim_recv_min_grad(self, channel, recv_ends)` on the fwdllm aggregator (place near
`_aggregate_grads_async:694`). It is a **purpose-built** grad analog of `asyncfl._sim_recv_min` (do NOT
call `_sim_recv_min` verbatim -- it does per-commit slot release via `_sim_pending_commit`/`sel.all_selected`
at asyncfl:615-627, but fwdllm must release on the AGG-GOAL boundary; and its withheld/staggered paths key
on WEIGHTS semantics). Needs module constants `_SIM_GATE_MAX_PASSES=64`, `_SIM_ORDER_SLACK_S=2.0` --
**import them from `flame.mode.horizontal.asyncfl.top_aggregator`** (they are module-level there; not yet
imported into fwdllm_aggregator). `RECV_TIMEOUT_WAIT_S` already imported (:48), `datetime` (:23), `time` (:22).

Method body (ingest -> gate -> pop -> advance -> budget -> mark committed; returns ONE `(msg, md)` matching
the shape of `next(channel.recv_fifo(...,1))`, or `(None, ("", datetime.now()))`):
```python
live = [e for e in (recv_ends or []) if channel.has(e)]
deadline = time.time() + RECV_TIMEOUT_WAIT_S
for _pass in range(_SIM_GATE_MAX_PASSES):
    grace = self._sim_recv_grace_s()
    to_probe = [e for e in set(live) | set(self._sim_inflight_expected)
                if channel.has(e) and not self._sim_buffer.has(e) and e not in self._sim_committed]
    if to_probe:
        for m, md in channel.recv_fifo(to_probe, first_k=len(to_probe), timeout=grace):
            if m is None: break
            _e = md[0]; _s = m.get(MessageType.SIM_COMPLETION_TS)
            self._sim_buffer.add(_e, float(_s) if _s is not None else self._vclock.now, (m, md))
    bmin = self._sim_buffer.peek_min_ts()
    min_stuck = None
    for e, exp in self._sim_inflight_expected.items():
        if self._sim_buffer.has(e) or e in self._sim_committed: continue
        if min_stuck is None or exp < min_stuck: min_stuck = exp
    earlier_stuck = (bmin is not None and min_stuck is not None and min_stuck + _SIM_ORDER_SLACK_S < bmin)
    if bmin is None and not to_probe: break
    if not earlier_stuck: break
    if time.time() >= deadline: break
popped = self._sim_buffer.pop_min()
if popped is None: return None, ("", datetime.now())
_end, sct, (m, md) = popped
# clock-jump clamp (don't lap a fresh in-flight cohort):
_now = self._vclock.now; _min_future = None
for e, exp in self._sim_inflight_expected.items():
    if e == _end or e in self._sim_committed: continue
    if exp > _now and (_min_future is None or exp < _min_future): _min_future = exp
_advance_to = sct if _min_future is None else max(_now, min(sct, _min_future + _SIM_ORDER_SLACK_S))
self._advance_sim_clock(_advance_to)
self._sim_committed.add(_end); self._sim_inflight_expected.pop(_end, None)
_b = m.get(MessageType.TRAINING_BUDGET_S) if isinstance(m, dict) else None
if _b is not None:
    self._sim_trainer_budget[_end] = float(_b); self._sim_budget_min = min(self._sim_budget_min, float(_b))
return m, md
```
Then in `_aggregate_grads_async` REPLACE the unconditional `next(channel.recv_fifo(...,1,...))` at **:720**
with a mode branch:
```python
if self.simulated:
    msg, metadata = self._sim_recv_min_grad(channel, channel.ends(VAL_CH_STATE_RECV))
else:
    msg, metadata = next(channel.recv_fifo(channel.ends(VAL_CH_STATE_RECV), 1, timeout=RECV_TIMEOUT_WAIT_S))
end, timestamp = metadata
```
(keep the existing `if not msg: return` guard right after).

**Agg-goal-boundary cleanup (rollback-safety, §I.5).** Add a helper and call it from
`_process_aggregation_goal_met` right AFTER `channel.cleanup_recvd_ends()` (**:1309**) -- this point is
reached by BOTH the variance-PASS and variance-FAIL(rollback) branches (both fall through past the
`_per_agg_trainer_list = []` clear at :1272):
```python
def _release_sim_slots_at_agg_goal(self, channel, is_async):
    if not self.simulated: return
    self._sim_committed.clear()      # else next cycle's gate skips a re-contributing end (rollback hole)
    self._sim_buffer.clear()         # drop stranded arrived-but-uncommitted grads so next cycle is clean
    self._sim_inflight_expected.clear()
    if is_async:
        self._sim_hold_busy_slots(channel)  # empty held set -> releases every committed end's slot
```
call site: `if self.simulated: self._release_sim_slots_at_agg_goal(channel, is_async)` after :1309.
(This is why fwdllm's own `_reset_agg_goal_variables:450` NOT clearing `_sim_committed` -- the §I.6 ROLLBACK
HOLE -- is handled: we clear it explicitly at the real boundary instead.) `_sim_hold_busy_slots`
(asyncfl:1453) needs `channel._selector` with `.requester/.all_selected/.selected_ends` -- provide in the
fake-channel test.

### J.3  REMAINING (untouched)

- **Sync barrier** (subtask 4): in `sync_collect_and_accumulate_grads:1315`, replace the drain loop
  `for msg, metadata in channel.recv_fifo(channel.ends(), num_min_req, timeout=...)` (**:1337**) with, in
  the sim branch, `committed = self._sync_sim_recv_first_k(channel, channel.ends(), num_min_req)` (base at
  `syncfl/top_aggregator.py:360`, returns ascending-sct list of `(msg, md)`, already stamps
  PROP_CLIENT_TASK_TRAIN_DURATION + advances vclock), then feed each through
  `_process_single_trainer_message`, breaking at `_agg_goal_cnt >= _agg_goal`. Apply
  `_barrier_anchored_lags(durs)` (`syncfl:472`) over the committed WALL_SEND completions for U6 telemetry.
  Real branch unchanged.
- **Launchers** (subtask 5): add `time_mode: simulated` sibling yamls of
  `expt_scripts/{fluxtune,fwdllm,fwdllm_plus}_n10_smoke.yaml` (current set `time_mode: real` at
  fluxtune:40 / fwdllm:35 / fwdllm_plus:39). Keep the real ones for reference runs.
- **Pytests** (subtask 6): see §I.8. No broker in this env -> pytest-only, drive `_sim_recv_min_grad` /
  `_sync_sim_recv_first_k` with a fake channel + synthetic out-of-order msgs incl. a variance-FAIL rollback
  cycle. Extend `tests/mode/test_fwdllm_trainer_sim_duration.py` for the trainer stamping + no-sleep assert.

### J.4  Open decisions taken during execution (record)
- time_mode default = `"real"` on the trainer getattr fallback (see J.1(1)); revisit if a sim run without an
  explicit yaml value is ever needed.
- `_sim_recv_min_grad` deliberately does NOT touch selector slots per-commit; ALL slot release + buffer +
  committed-mark clearing happens at the agg-goal boundary (`_release_sim_slots_at_agg_goal`). This is the
  fwdllm-specific divergence from asyncfl's per-commit release and the crux of rollback-safety.
- `_sim_buffer.clear()` at the boundary DROPS any stranded arrived-but-uncommitted grad. Benign at syn_0
  (|selected| ~= agg_goal, all commit). If a future rung shows lost updates, revisit (commit-then-carry
  instead of drop). Flagged as a fidelity watch-point.

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
