# High-Fidelity Simulator for FwdLLM -- Design & Staged Build Plan

**Active build (branch `dg/fwdllm_sim_unavail`).** This document is the plan for building a
high-fidelity **simulated-clock** runner for the `fwdllm` example (FedFwd / forward-gradient FL) that
reaches **real<->sim parity** across the **fluxtune / fwdllm / fwdllm++** baselines, at
**100% availability (syn_0)** first, then under **unavailability (syn_20, syn_50, mobiperf)**, then
**beyond syn_0 traces**.

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

*(empty -- first entry after Stage 0 smoke on the implementation branch. Record per-baseline run dirs,
lowest broken rung, root hypothesis, score X/N, JSON path; keep the run-length budget keyed. One section,
updated in place.)*
