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
> newest-last); the **curated at-a-glance delta table is §B.1** (kept current, unlike §K's append-only log);
> locked cross-cutting ones also surface in §F "Locked principles" / §J.4.

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

### §B.1  Curated real<->sim design deltas vs async_cifar10 -- **KEEP CURRENT**

Every place fwdllm's real+sim design intentionally diverges from async_cifar10, in one at-a-glance table
so a reviewer can separate an **intentional fwdllm-specific choice** from an accidental discrepancy without
reading the whole §K log. **This table is CURATED (rewritten in place to reflect the current design);
§K is the append-only rationale log.** *Maintenance rule (per the DESIGN PRINCIPLE at the top): when a
deviation lands or changes, append its rationale to §K **and** update the matching row here so this stays
the true current picture.* The `§K` column points at the full rationale.

| # | Axis | async_cifar10 | fwdllm | Why fwdllm differs | §K |
|---|---|---|---|---|---|
| 1 | Aggregated object | model **weights** | **gradients** (forward-grad JVPs) | grad values are mode-invariant given identical input+perturbation seed, so parity reduces to clock+order+selection parity **plus** the variance-cadence layer | §F.1 |
| 2 | Progress axis | update / round count | committed **`data_id`** (variance passes) | cadence (updates-per-data_id) is an **output to match**, not an input to assume | principle #2 |
| 3 | Commit cadence | fixed `agg_goal` | endogenous **variance-gated dynamic-K** | the emergent layer the cifar ladder doesn't model; V/DK/G rungs verify it | §F.1 |
| 4 | sct delay model | `sct = send + max(gpu, D)` (sleep-to-fill-budget) | `sct = send + gpu + D` (**additive**) | fwdllm's real mode sleeps D *on top of* GPU time; copying cifar's `max()` would desync real<->sim | **K-D2** |
| 5 | Per-eval sct | distinct eval sct, ~20x eval speedup | **collapses to the train sct** | eval lives on the aggregator; forward-grad "train" IS a forward pass (no 20x factor); trainer eval msg is a utility report, not a clocked commit | **K-D3** |
| 6 | Slot release | per-commit (inside `_sim_recv_min`), one ledger | **two-lifetime split at the boundary** — compute slot (`selected_ends`→`extra`) frees on grad RETURN; re-pick guard (`all_selected`→`filtered_ends`) frees on COMMIT | async_oort already tracks both ledgers; conflating them starved concurrency (D-e). Split: a returned trainer frees its slot (real frees on receipt) but stays un-re-pickable until commit, surviving carry + rollback | **K-D5**/**K-D16**, principle #4 |
| 7 | Buffered-but-uncommitted grad on rollback | carried across the barrier | **carried** (async/fluxtune, K-D12) — commit-then-carry + Option-A slot/guard split (K-D16); **dropped** stays correct for sync (c≈agg_goal, no surplus) | drop was benign only for `\|selected\|≈agg_goal` (sync); fluxtune (c=10≫agg_goal=3) dropped ~7 grads/cycle → 2× passes → reversed to carry; the carry then needed the slot-release timing fix (K-D16) | **K-D6**, **K-D12**, **K-D16**, §L |
| 8 | Async drain primitive | `_sim_recv_min` verbatim | purpose-built `_sim_recv_min_grad` / sync `_sync_sim_recv_first_k` (reuse the primitives, fork the orchestration) | `_sim_recv_min`'s per-commit slot release + withheld/staggered paths key on WEIGHTS semantics -- wrong for a grad pool released on the agg-goal boundary | **K-D4** |
| 9 | `time_mode` default | `"simulated"` | `"real"` (getattr fallback) | fwdllm's entire config corpus is `time_mode: real` and shipped with no sim path; a "simulated" default risks silently half-activating an unbuilt path | **K-D1** |
| 10 | Cadence telemetry | n/a | **pre-mutation** cycle snapshot (`cycle_data_id`/`cycle_iteration`/`grad_pool_size`/`cached_v_size`) | the post-mutation `data_id`/`iteration_per_data_id` advance BEFORE the event emits, so binning by them is off-by-one; snapshot before the pass/fail branch makes V1 exact | **K-D9** |
| 11 | Availability tracking (v1) | all `trace_read` | **mixed**: fwdllm unaware, fwdllm_plus `oracular`, fluxtune `client_notify` mapped onto approx `trace_read` | baselines carry different tracking models; first-class `client_notify` deferred to Stage H to keep Phase 2 tractable | **D1** |
| 12 | Launch tooling | single parity template (sim+real pairs, `baseline:` field) driven by `debug_run.sh` | per-baseline yamls + **separate `_sim` files** driven by `run_sequential.sh` | different config models; **both drivers now source the shared harness `examples/scripts/expt_runner.{sh,py}`** (conda activation, launch+ticker, log asserts, pre-flight + tiered hyperparam display) so only the config-discovery/patch adapter differs per example | this work |

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
- **D6 -- D-e fix: how to decouple compute-slot release from the re-pick guard. RESOLVED → Option A
  (K-D16).** *Decision gate answered:* async_oort's triplet filter (`async_oort.py:1660`) skips only when
  `trainer_version_states[end] == agg_version_state`, i.e. WITHIN the current cycle; since the triplet
  advances every agg-goal boundary it does NOT block re-pick of a *carried* trainer across the boundary, and
  `_trainer_state_dict` was never even populated — so **Option B is unsafe** and A is required. Landed: free
  `selected_ends` (compute slot) on RETURN, hold `all_selected` (re-pick guard) to COMMIT, carry the grad,
  survive the rollback boundary; triplet guard now populated at dispatch. See §H open-root #1 for the
  root-cause and K-D16 for the as-built. *Original options analysis kept below for the record.* The bug: the
  K-D12 commit-then-carry fix leaves a
  returned-but-uncommitted trainer in the selector's `all_selected`, which the dispatch top-up reads as an
  occupied compute slot (`extra = c − |all_selected|`, `count_avl_train` excludes `all_selected`,
  `async_oort.py:1580`) → sim refills the pipeline by ~1/cycle → concurrency collapses to ~1.5 vs real ~5.8.
  `all_selected` conflates two lifetimes that must split: **compute-slot occupancy** (must end at grad
  **RETURN** — a returned trainer is idle, its slot should refill to keep C computing, exactly what real's
  channel does on receipt) vs the **re-pick guard** (must hold until **COMMIT** — don't re-select the same
  trainer for the same `(model_version, data_id, iteration)` triplet while its grad is still carried).
  Two implementation strategies:
  - **Option A — two explicit sets (faithful, more code).** Add a distinct in-flight/computing set: a trainer
    enters at dispatch, leaves at grad RETURN; the top-up uses `extra = c − |in_flight_computing|` so the
    pipeline refills to C on return. A separate guard set holds the carried-but-uncommitted trainers
    un-re-pickable until commit, then clears. *Pros:* mirrors async_cifar10 exactly (§3.resid frees on
    receipt + §4.9 `simInflightCarryover` carries the grad); explicit, auditable, unit-testable in isolation
    (assert concurrency ≈ C and R1 overlap == 0%). *Cons:* new state to keep consistent with the carry buffer
    **across variance-FAIL rollbacks** at the agg-goal boundary (K-D5) — the main correctness surface to test.
  - **Option B — release the slot at return, reuse the existing triplet-version filter as the guard
    (minimal change).** Only fix WHEN the slot frees: remove a returned trainer from `all_selected` at grad
    receipt (so `count_avl_train`/`extra` see it free), and rely on the EXISTING `trainer_version_states`
    triplet filter (`async_oort.py:1660`, skips a trainer already holding the current triplet) to block
    re-pick until commit advances the version. *Pros:* smallest diff, no new bookkeeping, leans on selector
    machinery already present. *Cons:* the residence guarantee now depends on selector-internal filtering
    rather than an explicit set (harder to reason about / test alone); correctness hinges on the triplet
    filter covering the ENTIRE return→commit window **including across a variance rollback** (where
    model_version/data_id/iter can shift and the guard could leak → re-pick a still-carried trainer → R1
    violation). **Must prove R1 stays 0% under rollback before choosing B.**
  - **Leaning A** (explicit + testable + faithful) unless a scoping pass shows the triplet filter provably
    covers the return→commit window across rollbacks — then B is the cheaper equivalent. **Decision gate:**
    does `trainer_version_states` block re-pick for a returned-uncommitted trainer through a variance-FAIL
    rollback? Answer that first; it picks A vs B. Either way: `fwdllm_aggregator.py` + fluxtune yaml only,
    sync path untouched (K-D5/K-D11), shared weight path untouched (principle #8); log the chosen option in
    §K as the D-e resolution.

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
8. **Fix the concept, not the symptom -- and never regress a working real<->sim example.** Before patching
   a sim divergence, ask *what the mechanism is for*, not just how to get past the failure:
   - **Classify the mechanism: real-transport artifact vs. algorithmic property.** Devices that exist only
     to manage the MQTT/channel message queue (incremental "commit 1 per pass" draining, `recv_fifo`
     timeouts, "waiting to clear up queue", persistence of uncommitted messages across calls) are
     **real-transport artifacts with no sim analog** -- the sim commit path is the *barrier* (`sct`-ordered,
     single-pass, slot-release at the agg-goal boundary), not a trickling queue. Applying a real-transport
     discipline to the sim barrier is a category error. Gate such logic `and not self.simulated`; do **not**
     re-derive it in the sim path.
   - **Prefer the byte-identical-for-the-working-side fix.** When one side already works (usually real),
     the correct fix leaves that side provably unchanged. `and not self.simulated` on a real-only clamp is
     safe; globally reverting a flag that a prior stall-hotfix overloaded (e.g. `ends_not_selected_yet`'s
     `len(ends) >= agg_goal` clause from #42) is *not* -- it risks re-opening the bug that hotfix closed.
     A greedy fix in one direction is still greedy.
   - **Scope-check before editing shared code.** `grep` the symbol: if it lives only in
     `fwdllm_aggregator.py` the blast radius is fwdllm; if it's in `top_aggregator.py` / the shared
     `parity` engine / `_sim_recv_min`, a change can silently break **async_cifar10**, the previously-green
     baseline. Never trade a fwdllm fix for an async_cifar10 regression. (See principle #6: a real<->sim
     gap has two fix directions -- pick the one that doesn't move the already-admissible side.)
9. **Test cadence -- match the pytest scope to the blast radius; full-suite after a cross-stack change,
   not before every run.** The parity rungs + sim mechanisms are pure/deterministic over synthetic
   telemetry, so pytest is the fast gate -- but scope it to what you touched:
   - **fwdllm-only edit** (`fwdllm_aggregator.py`/`fwdllm_trainer.py`): `pytest tests/mode -k fwdllm`.
   - **shared parity engine / `parity_checks` shim** (also feeds async_cifar10): add
     `pytest examples/async_cifar10/scripts/parity` + `tests/mode -k parity`.
   - **shared stack** (`flame/launch/*`, `flame/telemetry/*`, `flame/config.py`, optimizer/selector, any
     `top_aggregator.py`): run the **full `pytest tests/`** -- these fan out across BOTH examples and every
     aggregator/trainer, so a subset can miss the regression they'd cause.
   You do **NOT** need full `pytest tests/` before every real<->sim run -- the per-scope subset is the
   inner-loop gate. Run the full suite **once after a major set of changes that spans the code stack /
   could affect other aggregators/trainers/examples** (this Batch 2.5 is exactly that: it touched
   `flame/launch/*` + the shared `parity` engine), and bank that green before launching. A cluster run is
   far more expensive than the full unit suite (minutes, no GPU/MQTT), so the insurance is cheap.
10. **Comment discipline -- comment the WHY, crisply; skip the trivial.** Write a comment when it carries
    knowledge a later reader (or a later you) would otherwise have to re-derive or re-learn the hard way:
    a conceptual/algorithmic choice, a baseline/architectural decision, a real<->sim divergence and its
    rationale, or a failure mode + why the fix takes the shape it does (so the mistake isn't repeated).
    Do **not** comment trivial mechanical edits (snake_case→camelCase, a rename, an obvious guard). When you
    do comment, be **crisp** -- one or two tight sentences, not a paragraph restating the code. The deeper
    rationale belongs in §K (append-only) with a one-line pointer from the code; the code comment states the
    decision, not its whole history.

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

*Discipline (from PARITY.md §Status / §Settled-roots / §Dead-ends): this section is the **consolidated
ground state**, rewritten in place — NOT an append-per-run log. The end goal is end-to-end real↔sim parity,
so what we keep is (1) the current per-baseline state table, (2) the **open roots** ranked lowest-rung-first,
(3) a **fixes-landed** ledger (what worked, so we don't redo it), and (4) a **dead-ends & corrections**
ledger (what we tried or believed that was wrong, so we don't retry it). Per-run detail lives in the JSON +
git history; the as-built rationale lives in §K.*

**Status (2026-07-03 → Option-A landed; the numbers below are the PRE-fix run, `max_data_id=3` — a SHORT
convergence run, agg_goal fwdllm/plus=10, fluxtune=3).** Batches 1–2 + §L Batch-2.5 + the **Option-A D-e fix
(K-D16)** + the **clock-family re-key (#2)** + the **field-coverage alias (#3)** are landed & pytest-green
(`tests/mode` 421 passed / 7 skipped incl. parity sub-package). Residence (R1) is exact on all three; grad
values are mode-invariant. **No open sim-mechanism root remains in code** — the three checker/mechanism roots
below (#1 D-e, #2 clock-family axis, #3 field-name) are FIXED and moved to *Fixes landed*; the only remaining
action is the **§L step-6 re-run** to bank the post-fix numbers (esp. fluxtune concurrency vs real ~5.8 and
wall) and a **longer run (max_data_id=10, the yaml default)** to clear the #4 short-run truncation. The table
below is the pre-fix baseline the re-run is measured against.

**Per-baseline ground state (PRE-fix; re-run pending):**
| baseline | wall real→sim | R1 resid | U3 staleness | V1/V2 cadence | status |
|---|---|---|---|---|---|
| **fwdllm** (sync) | 107→**96s** ✓ | PASS (0%) | PASS | PASS / PASS | clock-family axis re-keyed (#2 fixed); re-run to confirm |
| **fwdllm_plus** (sync) | 335→**184s** ✓ | PASS (0%) | PASS | FAIL(trunc) / FAIL | #2 fixed; V1 trunc needs the longer run (#4); selection_detail after re-run |
| **fluxtune** (async) | 125→**304s ✗ (2.4×)** | PASS (0%) | **FAIL** (sim 0.23<real 0.57) | FAIL / FAIL | **D-e FIXED (Option A, K-D16)**; re-run measures recovery of wall/U3/V1/selection |

### Open roots (fix lowest-rung-first)

**Roots #1 (D-e), #2 (clock-family axis), #3 (field_coverage) are FIXED in code (see *Fixes landed*); what
remains is operational — the re-run to bank post-fix numbers.**

1. **[FIXED — Option A / K-D16] D-e async concurrency starvation (fluxtune).** *Symptom (pre-fix):* sim kept
   ~1.5 trainers computing vs real's ~5.8, did fewer forward passes (89 vs 109) yet took **2.4× the wall**
   (304 vs 125s) — re-dispatching ~1 trainer/agg-goal-cycle. *Root:* K-D12 parked a returned-but-uncommitted
   trainer in async_oort's `selected_ends`, which drives `extra = c − len(selected_ends)`, so top-up refilled
   ~1/cycle. `selected_ends`/`all_selected` conflated compute-slot occupancy with the re-pick guard. *Fix:*
   Option A two-lifetime split — free `selected_ends` on RETURN (slot reopens → a DIFFERENT trainer refills
   C, mirroring real's channel freeing on receipt), hold `all_selected` (guard) to COMMIT so
   one-in-flight-per-trainer survives the carry + rollback. Details K-D16 / §F-D6. **Remaining: the §L step-6
   re-run must confirm the recovery** (concurrency, wall, U3, V1, selection) — the absolute in-flight number
   is emergent, measured not asserted.

2. **[FIXED — #2] Clock family re-keyed to the `data_id` progress axis.** `throughput_parity` /
   `total_commits_parity` / `terminal_state_parity` now count units on the axis the run advances (a unified
   `_progress_axis` / `_per_progress_last_event` helper): `round` for normal FL, committed `data_id` for
   fwdllm (was collapsing to `sim_rounds=1`). async_cifar10 auto-detects `round` → byte-identical. Checker
   fix, validates instantly on the banked dirs.

3. **[FIXED — #3] field_coverage accepts fwdllm field aliases.** The coverage spec rows for
   `gpu_compute_s` / `training_budget_s` now accept the fwdllm spellings `real_gpu_time_s` /
   `sim_round_duration_s` (a spec field may be a tuple of accepted names). No more false-FAIL / benign SKIP.

4. **[OPEN — operational] V1/V2/V5 short-run truncation (fwdllm_plus + fluxtune).** Per-data_id series match
   on every FULL data_id and diverge only on the last, truncated one; with 3 data_ids one boundary diff → KS
   0.33. **Fix = the longer run** (`max_data_id_progress=10`, already the yaml default — do NOT pass
   `--max-data-id 3`), not code. For fluxtune, partly downstream of the now-fixed D-e.

5. **[OPEN — re-validate after re-run] fluxtune selection_detail / preferred_duration (oort fidelity).**
   Likely downstream of D-e (a collapsed in-flight set changed the selection sequence). Re-validate after the
   re-run before treating as its own root.

### Fixes landed (what worked — do not redo)

- **Trainer sim clock** — additive `sct = send + gpu + D`, no-sleep on the sim path (Batch 1, K-D2/K-D3).
- **Async grad loop on the vclock** — purpose-built `_sim_recv_min_grad` sct reorder buffer + in-flight gate
  + agg-goal-boundary rollback cleanup (Batch 1, K-D4/K-D5).
- **Sync barrier** — `_sync_sim_recv_first_k` (first-k-smallest); the `ends_not_selected_yet` "commit-1-per-
  pass" clamp gated **real-only** so the sim barrier drains the full dynamic-K cohort in one pass (K-D11).
- **Variance-cadence rung layer** — V1-V5 / DK1-DK3 / G1-G2 + the pre-mutation `cycle_data_id` snapshot that
  makes V1 exact (Batch 2, K-D9/K-D10).
- **Residence: commit-then-carry + R1/W1 rungs** — killed the 2× recompute (228→~89 passes); R1 in-flight
  overlap exact 0% on all three (K-D12/K-D14). *(This fix also introduced D-e — the carry was right, the
  slot-release timing was not; now resolved by K-D16 below.)*
- **D-e: Option A two-lifetime split (K-D16)** — override fwdllm's `_sim_hold_busy_slots` so a
  returned/carried trainer frees its compute slot (`selected_ends` → `extra` reopens → concurrency refills to
  C with a different trainer) but stays un-re-pickable (`all_selected` guard) until COMMIT; carry + rollback
  survive. Also populate the `(model_version, data_id, iteration)` triplet at dispatch so async_oort's
  within-cycle filter has real state. Pytest-green (`TestOptionASlotGuardSplit`, residence suite);
  concurrency recovery measured by the §L step-6 re-run.
- **Clock-family re-key to `data_id` (#2)** — unified `_progress_axis`/`_per_progress_last_event` helper;
  `throughput`/`total_commits`/`terminal_state` now count on the axis the run advances (fwdllm `data_id`,
  normal FL `round`). async_cifar10 byte-identical (auto-detects `round`). `TestProgressAxisRekey`.
- **field_coverage aliases (#3)** — spec rows accept a tuple of field names; `gpu_compute_s`/
  `training_budget_s` also match fwdllm's `real_gpu_time_s`/`sim_round_duration_s`. `TestFieldCoverageAlias`.
- **W1 made asymmetric** — flags only a sim EXCESS over real (the 2× recompute it was built for); benign sim
  under-compute no longer fires.
- **staleness_policy wired from config** — was silently `none` for every run (K-D15); + fedbuff staleness-
  weighted accept for fluxtune (K-D13). Confirmed both sides now log `fedbuff`/`exact`/`round_data_id`.
- **camelCase key-collision** — fluxtune yaml carried both `stalenessPolicy:none` (base) and
  `staleness_policy:fedbuff` (override); base won at pydantic resolution → ran as `none`. Fixed to camelCase.
- **vclock_now emitted** on agg_round (mirrors asyncfl) — confirmed present (sim 6/6/26 agg_rounds).
- **fwdllm_plus crash** — `read_trainer_unavailability` kwarg rename (`base_dir`→`metadata_dir`); no longer
  crashes in `internal_init`.
- **Launcher fail-fast (D-d)** — detect early aggregator death / traceback; terminate trainers on non-zero
  agg exit instead of burning the 30s-per-trainer EOT grace.

### Dead ends & corrections — do NOT retry

- **K-D6 "drop stranded grads at the agg-goal boundary"** — REVERSED for async (K-D12). The
  "|selected| ≈ agg_goal" premise holds only for the sync baselines; for c ≫ agg_goal (fluxtune) it dropped
  ~7 grads/cycle → residence violation. Commit-then-carry replaced it. (Drop stays correct for sync.)
- **"sim wall ≈ real, comparable" (claimed after the first post-residence re-run).** WRONG — fluxtune sim is
  **2.4× real wall** (D-e). The earlier "82 vs 120 passes, comparable" read the pass *count* and missed the
  concurrency *collapse* behind it. Always check avg in-flight concurrency, not just total passes.
- **"D=0 smoke, so the clock-family fails are artifacts" (believed briefly this session).** WRONG — the runs
  are **D>0** (real trainers sleep ≈1.13s/pass; sim charges ≈1.12s to the vclock). The clock-family fails are
  real, and trace to the rung being keyed on `round` not `data_id` (open root #2) — not to D being off.
- **Tuning `var_threshold` / `max_iterations_per_data_id` to close a cadence gap** — pre-emptively rejected
  (§G): these are baseline-defining knobs, not parity levers. A cadence gap is an upstream set/order/clock
  divergence (here, D-e).

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

**How to launch (tooling landed post-Batch-2).** `expt_scripts/run_sequential.sh` drives the real<->sim
pairs via the shared harness `examples/scripts/expt_runner.{sh,py}` (§B.1 row 12): `--mode {sim|real|both}`
pairs each baseline and tags run dirs `_real`/`_sim` so `scripts.parity.cli` globs the pair; `--delays
{on|off}` sets `enable_training_delays` identically on both sides (K-D8); a pre-flight gate prints the
hyperparameters in three volatility tiers (① review-every-run, ② per-baseline, ③ config-baked) and blocks
infeasible configs (`agg_goal>c`, `num_gpus>visible`, `num_trainers<minInitialTrainers`). Always `--dry-run`
first (shows the table + checks, launches nothing). Smoke: `--mode both --delays off` (D=0, mechanics).
Convergence: `--mode both --delays on --max-runtime-s <budget> --max-data-id <cap>` (D>0 both sides).
Parity: `python expt_scripts/run_parity.py` (the repeatable post-fix checker) — it picks the LATEST
real/sim pair per baseline (parsing the baseline as an exact dir-name token, so `fwdllm` never captures
`fwdllm_plus`; the raw `scripts.parity.cli --batch` glob does), reads agg_goal from each run's config,
prints the pairs to confirm, then a compact cross-baseline rung summary. `--yes` skips the prompt;
`--validate` also reports live `staleness_policy` + `vclock_now` presence.

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
  **Where:** J.4, J.2. **⚠ SUPERSEDED by K-D12 (2026-07-03):** the watch-point TRIGGERED for fluxtune
  (async, c=10 ≫ agg_goal=3) — the "|selected| ≈ agg_goal" premise holds only for the sync baselines;
  async drops ~7 grads/cycle → residence violation → switch to commit-then-carry (see K-D12, §L).
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
- **K-D11  `ends_not_selected_yet` "commit 1 per pass" clamp gated real-only (`and not self.simulated`).**
  Surfaced by the first Phase-1 syn_0 sign-off smoke (`--mode both`, 2026-07-03): the **real** run aggregated
  and hit `max_data_id`; the **sim** run deadlocked to `max_runtime` with 0 aggregations. Root cause: in
  `sync_collect_and_accumulate_grads` (`:~1486`) the flag clamps `num_min_req` (the barrier `first_k`) to 1.
  That clamp is a **real-transport draining discipline** — it only works because uncommitted messages persist
  in the channel queue and are re-collected next pass (real accumulates 1→agg_goal over ~10 passes). The sim
  barrier `_sync_sim_recv_first_k` is single-pass: it drains the whole cohort, commits `first_k`, and **drops
  the rest** (slots release at the agg-goal boundary, K-D5). With `first_k=1` it committed 1 and discarded 9;
  trainers had already sent once and block in `_fetch_weights`, so the queue never refilled → goal never met.
  The `len(ends) >= agg_goal` clause that trips the flag on a *full* fresh cohort is itself an overload from a
  real-mode stall hotfix (#42); the flag's original meaning was just "no cohort selected yet". **Options:**
  (a) revert the #42 overload globally — risks re-opening the real stall; (b) gate the clamp real-only.
  **Chose (b):** real path byte-identical (`self.simulated=False`), async_cifar10 untouched (flag is
  fwdllm-only), sim keeps `num_min_req=min(agg_goal, len(recv_ends))` and commits the full dynamic-K cohort
  in one barrier pass. Instance of locked principle **#8** (real-transport artifact, no sim analog). **Where:**
  `fwdllm_aggregator.py::sync_collect_and_accumulate_grads`. **Verify on re-run:** sim `[SIM_BARRIER]` shows
  `first_k=10` + non-zero `buf_depth`, `committed` reaches agg_goal, `data_id` advances to `max_data_id`
  (stops on the cap, not `max_runtime`), eval metrics emit; then parity-compare vs the real sibling.
- **K-D12  fluxtune async grad path: commit-then-CARRY the surplus buffer + hold residence (REVERSES
  K-D6 for the async baseline).** Surfaced by the 2026-07-03 3-baseline smoke (§H, §L): fluxtune sim ran
  2× the forward passes of real. **Options:** (a) keep K-D6 drop (rollback-clean, but for c≫agg_goal it
  discards ~7 arrived grads/cycle → residence violation → 2× compute → cadence divergence); (b) carry the
  arrived-but-uncommitted grads across the agg-goal boundary and hold their trainers busy until commit —
  the fwdllm analog of felix `simInflightResidence` (§3.resid) + oort `simInflightCarryover` (§4.9).
  **Chose (b) for async (fluxtune); (a) stays correct for the sync baselines** (fwdllm/fwdllm_plus have
  c≈agg_goal, no surplus). K-D6's "benign at syn_0" premise held only for `\|selected\|≈agg_goal`; async
  breaks it. Full remediation design + staging is **§L**. **Where:** `_release_sim_slots_at_agg_goal`
  (clear-before-hold ordering bug) + `_sim_hold_busy_slots` + the `sim_inflight_residence` gate. Instance
  of locked principle #4 (slot residence must survive the boundary) and #8 (don't apply a sync-barrier
  discipline to the async path).
- **K-D14  R1/W1 residence telemetry sourced from an ECHOED per-contribution interval, not the aggregator's
  per-end dispatch stamp (Batch 2.5).** R1 (in-flight overlap) needs each grad's exact `[dispatch,
  completion]` interval. **Options:** (a) read dispatch from the aggregator's `PROP_SIM_SEND_TS` end
  property at commit time; (b) have the trainer ECHO `SIM_SEND_TS` back in the grad message (it already
  stores the received stamp) and capture `[SIM_SEND_TS, SIM_COMPLETION_TS]` in
  `_process_single_trainer_message`. **Chose (b):** the property (a) is overwritten on a re-dispatch, so if
  residence is broken (exactly the case R1 must catch) it would read the NEW dispatch and hide the overlap;
  the echoed pair is immutable per contribution. Emitted per cycle as `contributor_intervals` on the
  agg_round event; real mode falls back to the wall interval (`PROP_ROUND_START_TIME`..receipt). W1
  (compute-conservation) reads forward passes from `trainer_round` counts vs committed contributors — both
  already emitted. The rungs SKIP cleanly on the sync baselines (no re-selection ⇒ no ≥2 intervals/trainer)
  and non-fwdllm runs (field absent). R1 is wired UPSTREAM of V1 in `CHECK_META` so the dep chain proves a
  cadence divergence is downstream of a residence violation. **Where:** `fwdllm_trainer.py` (SIM_SEND_TS
  echo), `fwdllm_aggregator.py` (`_sim_contrib_intervals` capture + `contributor_intervals` emit),
  `parity/checks.py::inflight_overlap_parity`/`compute_conservation_parity`, `validate_real.py::
  check_grad_residence`, `tests/mode/test_fwdllm_sim_grad_residence.py` + R1/W1 fixtures.
- **K-D15  `self.staleness_policy` was never wired from config — the gate silently ran as `none` for every
  run (as-built fix, Batch 2.5).** Implementing D-c (fedbuff staleness-accept) surfaced that
  `_process_single_trainer_message` read `getattr(self, "staleness_policy", "none")` but NOTHING ever
  assigned `self.staleness_policy` — so `round_data_id`/`exact`/`fedbuff` configs were all inert (always
  "none"). **Fix:** set `self.staleness_policy` in `internal_init` from
  `config.hyperparameters.staleness_policy`, and recognize `fedbuff` as an explicit ACCEPT policy (no reject,
  no "unrecognized" warning) so its carried stale grads flow into the EXISTING fedbuff down-weighting
  (`aggregate_grads_from_trainers`, `rate = 1/sqrt(1+staleness)` when `_weighted_aggregation_enabled`). A
  guard warns if `staleness_policy=fedbuff` is set without a grad-aggregation-weighting optimizer (rate
  stays 1.0). The U3 staleness telemetry (`_cycle_staleness` on agg_round) already emitted, so no new emit.
  **Where:** `fwdllm_aggregator.py::internal_init` + `_process_single_trainer_message`; fluxtune yamls
  (`staleness_policy: fedbuff`, real+sim identical); `tests/mode/test_fwdllm_agg_telemetry.py::
  TestStalenessPolicy`.
- **K-D13  fluxtune staleness_policy = fedbuff staleness-weighted accept (resolves §L D5; provisional).**
  fluxtune's yaml leaves `staleness_policy` unset → defaults `none` (no staleness handling), wrong for an
  async fedbuff baseline whose carried grads (K-D12) are stale by construction. **Options:** (a) fedbuff
  staleness-weighted accept (down-weight by `V′−V`); (b) `round_data_id` reject+record (fwdllm_plus's sync
  policy); (c) keep `none`. **Chose (a)** — fluxtune IS async fedbuff, so accept-and-down-weight is the
  faithful treatment and feeds a non-trivial U3 staleness signal; **provisional** (may revisit the policy
  later). Set **identically in real and sim** (a baseline definition, not a parity lever — never tuned to
  close a gap; PARITY.md §F.4 / principle #3). **Obligation:** confirm the fedbuff optimizer weighting path
  is active and `staleness = V′−V` emits on the commit event (U3). **Where:** `fluxtune_n10_smoke*.yaml`
  `staleness_policy`; fedbuff optimizer weighting; commit-event `staleness` emit. Implement in §L step 5.
- **K-D16  D-e resolution — Option A two-lifetime split (free compute slot on RETURN, hold re-pick guard to
  COMMIT).** The K-D12 commit-then-carry fix parked a returned-but-uncommitted trainer in async_oort's
  `selected_ends`, which drives the dispatch top-up (`extra = c − len(selected_ends)`), so the pipeline
  refilled ~1/cycle and sim wall blew up 2.4× (§H open-root D-e). **Decision gate (D6) answered by reading
  the code:** async_oort's triplet filter (`async_oort.py:1660`, skip if `trainer_version_states[end] ==
  agg_version_state`) is a WITHIN-cycle guard only — the triplet advances every agg-goal cycle, so it does
  NOT block re-pick of a *carried* trainer across the boundary (its recorded triplet ≠ the new current one).
  So **Option B is unsafe** (its fallback guard doesn't cover the carry window), and `_trainer_state_dict`
  was in fact **never populated** (filter inert). **Chose Option A** (§F-D6): async_oort already tracks the
  two lifetimes in SEPARATE ledgers — `selected_ends` = compute-slot occupancy (drives `extra`),
  `all_selected` = re-pick guard (drives `filtered_ends`) — they were merely conflated. Override fwdllm's
  `_sim_hold_busy_slots` to hold a RETURNED/carried trainer in `all_selected` ONLY (frees its slot so a
  DIFFERENT trainer refills concurrency to C, mirroring real freeing on channel receipt) while
  still-computing trainers hold BOTH; the guard clears on COMMIT so one-in-flight-per-trainer survives the
  carry boundary + variance-FAIL rollback (principle #4). ALSO populate `_trainer_state_dict` at dispatch =
  `(model_version, data_id, iteration)` and prune to the busy set at the boundary, so the triplet filter has
  real state (the within-cycle guard the user asked to re-key) and the R1 rung can prove no per-triplet
  re-pick. **Options weighed:** (a) explicit two-set split [CHOSEN — the ledgers already exist; smallest
  faithful change]; (b) reuse the triplet filter as the guard [REJECTED — proven not to cover the carry
  window across rollbacks]. **Scope:** `fwdllm_aggregator.py` override + dispatch (fwdllm blast radius);
  async_oort/oort and the shared asyncfl `_sim_hold_busy_slots` UNTOUCHED (principle #8); sync path untouched
  (K-D5/K-D11). **Where:** `fwdllm_aggregator.py::_sim_hold_busy_slots` (override) +
  `_release_sim_slots_at_agg_goal` + `_distribute_weights_async` (triplet emit);
  `tests/mode/test_fwdllm_sim_grad_residence.py::TestOptionASlotGuardSplit`. **Concurrency is emergent** —
  the split restores dispatch top-up; the actual in-flight number (vs real ~5.8) is measured by the §L step-6
  re-run, not asserted analytically.

---

## §L  Phase-1 sign-off remediation — async grad-path residence & carry-over (Batch 2.5, staged)

**Status: IMPLEMENTED & pytest-gated (2026-07-03); the one re-run (step 6) is the only remaining step.**
All of §L.4 steps 1–5 + §L.5 landed and are green (`tests/mode` = 297 passed / 7 skipped; parity
sub-package = 115; the new suites: `test_fwdllm_sim_grad_residence.py`, R1/W1 fixtures in
`test_parity_checks.py`, contributor-interval + staleness-accept tests in `test_fwdllm_agg_telemetry.py`).
As-built decisions are logged in **§K-D14/D15**; the code is the source of truth. **Step 6 (the 3-baseline
syn_0 re-run → `parity_checks`) is the operator's next action** — it needs the cluster and cannot run in
pytest. The first real↔sim 3-baseline
syn_0 smoke (§H, 2026-07-03) surfaced three parity defects + one tooling defect. This section states the
**principled correctness model** (what BOTH real and sim must do), root-causes each defect against the
PARITY.md ladder, and gives the **staged, pytest-gated build plan** with exact code sites, new
telemetry/checks/tests, and exit criteria. This is a Batch inserted between Batch 2 and the Phase-1
convergence sign-off (the mechanics must be right before convergence/parity numbers mean anything).

### §L.1  The correctness model — what real AND sim must do (async fedbuff forward-grad, fluxtune)

The execution contract, mode-invariant (this is the "right thing" both sides must obey):

1. **Concurrency budget C.** The aggregator keeps up to C=`c` trainers in flight. A dispatched trainer is
   **BUSY** — held in the selector's `selected_ends` (a *slot*, not the unavailable list) — from dispatch
   until its update **commits**.
2. **One-in-flight-per-trainer (residence invariant, PARITY.md §3.resid).** A trainer is **re-pickable
   ONLY after its update returns AND is committed.** It is never re-dispatched while a prior update is
   outstanding, and never re-picked for the same `(round, data_id, iteration)` (fwdllm's tightening of the
   "not twice per round" FL rule). Real satisfies this by construction (the channel holds a dispatched end
   out of `VAL_CH_STATE_SEND` until aggregated → real overlap 0%). Sim must model it explicitly.
3. **Buffered accumulation (fedbuff).** Arriving grads accumulate in a buffer. When `agg_goal`=K grads are
   available, the aggregator runs one fedbuff step (staleness-weighted server-LR SGD) + the variance gate.
   The **surplus** in-flight trainers keep computing; a completed-but-not-yet-committed grad is **retained
   in the buffer and applied at a later step (carry-over)** — it is **never silently dropped**.
4. **Staleness is handled, not ignored.** A grad trained on `model_version` V that lands after the model
   advanced to V′>V is **stale by V′−V** and must be either (a) accepted with fedbuff staleness weighting,
   or (b) explicitly rejected AND recorded (speed/utility props still logged — PARITY.md "stale-property
   recording" lesson). "Silently drop" is neither and is wrong.
5. **Progress is `data_id`** (committed variance passes), not raw update count (§B, PARITY.md §F.1).

**Consequence to hold as an invariant/telemetry check:** over a run, **total forward passes ≈ total
committed grads + (grads still in flight at stop) + (grads explicitly stale-rejected)**. A large
unexplained gap = wasted recompute = a residence/carry violation. Real 109 fwd / 63 commit and sim 228 /
66 both show a gap; **sim's is far larger AND its participation is lock-step even** — the tell.

### §L.2  Root-caused defects (lowest-rung first)

- **D-a — Residence hold is nullified on the grad path (the root).** `_release_sim_slots_at_agg_goal`
  (`fwdllm_aggregator.py`) runs `self._sim_buffer.clear(); self._sim_inflight_expected.clear()` **before**
  `self._sim_hold_busy_slots(channel)`, which holds exactly `pending_in_buffer (∪ _sim_inflight_expected)`
  — both now empty → `held={}` → it **releases every trainer**. Plus `sim_inflight_residence=False` for
  fluxtune, so even the buffered set wouldn't be augmented by the in-flight set. Net: all C trainers freed
  every agg-goal cycle → re-dispatched → recompute. **Evidence:** even 22–23 rounds/trainer (sim) vs uneven
  7–14 (real); 228 vs 109 forward passes; identical per-pass GPU time. **Ladder:** this is the fwdllm async
  analog of felix's overlapping-re-dispatch root (PARITY.md §3.resid, "measure overlap from intervals not
  counters").
- **D-b — K-D6 buffer drop discards arrived grads (compounds D-a).** `_sim_buffer.clear()` at the boundary
  drops the ~7 arrived-but-uncommitted grads (c=10, K=3). Reverses to **commit-then-carry** (K-D12). For
  the sync baselines c≈K so there's no surplus — the drop stays correct there.
- **D-c — fluxtune `staleness_policy` unset → defaults `none`.** The yaml never sets it (§H config dump).
  For an async fedbuff baseline the surplus grads that carry over are stale-by-construction; `none` means
  no principled handling. **RESOLVED (D5/K-D13): fedbuff staleness-weighted accept**, set identically
  real+sim. NOT a parity lever to tune — a *definition* fixed once (may revisit the policy later).
- **D-d — launcher has no fail-fast (tooling).** fwdllm_plus crashed in `internal_init` at ~5s but the run
  burned 86s: `aggregator_spawner.wait_until_ready` is a fixed 5s "process alive" heuristic (misses a
  crash landing at ~5s), so trainers spawned anyway; then `runner.py` `wait_all(timeout_per_trainer=30.0)`
  waits 30s for trainers that will never get an EOT. Waste the operator could spend fixing the error.

### §L.3  What "correct" telemetry/checks look like — mirror the async_cifar10 lessons

The PARITY.md discipline (§1 ladder, "growth rule: every root leaves behind the finest check that would
have localized it"): D-a would have been caught instantly by a **residence/overlap invariant** rung. Add
these, reusing the shared engine (`async_cifar10/scripts/parity/checks.py` + STAGE map, §I.2):

| New/validate | Rung | Role/Tier | What it asserts | Telemetry it needs |
|---|---|---|---|---|
| **NEW** | **R1 in-flight overlap** | MECHANISM/**INV** | per-trainer dispatch→commit intervals do NOT overlap (one-in-flight); ≈0% both modes (felix metric, PARITY.md L295-298) | per-(trainer,cycle) `sim_send_ts`/dispatch ts + commit `vclock`/wall ts |
| **NEW** | **W1 compute conservation** | CONTROL/DIAG | `forward_passes ≈ commits + in_flight_at_stop + stale_rejected`, and the **real↔sim ratio ≈1**; localizes wasted recompute | count `trainer_round` events vs committed grads per mode (already emitted) |
| validate | **S2 participation** | EMERGENT/DIST | per-trainer chosen-count matches; the even-vs-uneven split is exactly this — must FAIL pre-fix, PASS post-fix (fluxtune only) | `selection_train` (exists) |
| validate | **V1 iters-per-data_id** | MECHANISM/DIST | data_id-2 80-vs-8 must flag; confirms V1 (Batch 2) fires and is **downstream** of R1 (dep chain proves it) | `cycle_data_id` (K-D9, exists) |
| validate | **U3 staleness** | MECHANISM/DIST | fedbuff staleness dist matched + non-trivial once D-c fixed | `staleness` in agg_round (exists) |

**Sanity gate (validate_real FIRST — HARD GATE, PARITY.md principle #6 / "verify real is correct before
tuning sim").** Real also shows fwd(109) > commit(63). Before making sim match real, **confirm real is
admissible:** (1) real R1 overlap is truly ~0% (channel residence), (2) the real fwd−commit gap is
explained by end-of-run in-flight + any stale-rejections, NOT by real *also* dropping carried grads. **If
real is itself wrong (overlap > 0, or real drops carried grads), FIX REAL FIRST** — make real correct,
re-bank it as the reference, and only then bring sim to parity (D-b becomes a two-sided fix, real side
first). Never tune sim toward a real that has not passed its own admissibility. Add a `validate_real`
assertion for R1==0% and the W1 decomposition; it **blocks** the §L.4 step-4 sim mechanism change.

### §L.4  Staged build (pytest-gated; flag-off ⇒ byte-identical; async-only, sync untouched)

Ordering follows the ladder: instrument → establish real-correctness → fix the mechanism → prove with
tests → re-run. Each step gated by pytest before the one run. **Steps 1–5 DONE & green (2026-07-03);
step 6 is the operator re-run.** As-built notes in K-D14/D15.

1. **✅ Telemetry (enables R1/W1).** Emit per-(trainer,cycle) dispatch ts + commit ts so overlap intervals
   are reconstructable (extend the agg_round `extra` and/or a `SIM_GRAD_RECV`-adjacent field;
   `sim_send_ts` + commit `vclock` already exist internally — surface them). Add nothing to the real hot
   path beyond a timestamp. *Exit:* a driven emission test asserts the fields land (mirror
   `test_fwdllm_agg_telemetry.py`).
2. **✅ Checks R1 + W1 in the shared engine** (`checks.py` `run_all_parity` + STAGE map; deps: R1 under
   Stage 3 residence, W1 CONTROL feeding V1/K2). *Exit:* known-PASS + known-broken synthetic fixtures
   (an overlapping-interval fixture FAILS R1; a 2×-recompute fixture FAILS W1) in `test_parity_checks.py`.
3. **✅ validate_real sanity** (R1==0% + W1 decomposition on the real dir) — run against the **already-banked
   2026-07-03 real run dirs** (checker-side, no cluster run; PARITY.md "checker fixes validate instantly").
   Decides whether D-b is one-sided or two-sided.
4. **✅ Mechanism fix (D-a + D-b), async grad path only.** In `_release_sim_slots_at_agg_goal`: (i) **hold
   before clear** — compute the held set from the pre-clear buffer/in-flight, release only the *committed*
   subset, keep the surplus buffered (commit-then-carry); (ii) turn on residence for fluxtune
   (`sim_inflight_residence=True` in `fluxtune_n10_smoke_sim.yaml`, and confirm `_sim_hold_busy_slots` holds
   `pending_in_buffer ∪ _sim_inflight_expected`). Reference implementations: felix `_sim_hold_busy_slots`
   (`asyncfl/top_aggregator.py:1453`, §3.resid) and oort `simInflightCarryover` (§4.9). **Scope-guard
   (principle #8):** the change is in `fwdllm_aggregator.py` (fwdllm blast radius) + a fluxtune yaml; do
   NOT touch the shared `_sim_recv_min`/`asyncfl` weight path (async_cifar10 is green). Sync path
   (`_sync_sim_recv_first_k`, K-D5/K-D11) unchanged — K-D6 drop stays correct there. *Exit:* a
   `test_fwdllm_sim_grad_residence.py` (mirror `test_async_inflight_residence.py`): drive c>K synthetic
   grads, assert surplus carried (not dropped), busy trainers not re-selected, R1==0%; + flag-off
   byte-identical.
5. **✅ Staleness definition (D-c) — D5 LOCKED: fedbuff-accept (K-D13/K-D15).** Wire fluxtune to fedbuff
   staleness-weighted accept: set `staleness_policy` in all fluxtune yamls (real+sim identical) so a
   carried stale grad is **accepted and down-weighted by `V′−V`**, and confirm the fedbuff optimizer's
   weighting path is active + emits `staleness` on the commit event (for U3). *Exit:* a unit test that a
   carried stale grad is accepted+weighted (never silently dropped) and that `staleness=V′−V` is emitted.
6. **✅ First re-run (2026-07-03).** Confirmed W1→PASS (sim under-computes, not over), R1==0% both modes,
   fwdllm_plus no longer crashes. **Surfaced D-e (concurrency starvation): the commit-then-carry fix freed
   the compute slot at COMMIT instead of at RETURN, collapsing sim in-flight concurrency to ~1.5 vs real
   ~5.8 → fluxtune sim wall 2.4× real, U3 low.**
7. **✅ D-e fixed — Option A two-lifetime split (K-D16).** Free `selected_ends` (compute slot) on RETURN,
   hold `all_selected` (re-pick guard) to COMMIT, carry + rollback survive; triplet populated at dispatch.
   Landed with the clock-family `data_id` re-key (#2) and field-coverage aliases (#3); pytest-green
   (`tests/mode` 421 passed / 7 skipped incl. parity). Current ground state in **§H**.
8. **▶ IN FLIGHT (operator launched the post-fix re-run, 2026-07-03).** One `--mode both` 3-baseline run at syn_0 with the yaml
   default `max_data_id_progress=10` (do NOT cap at 3 — clears the #4 truncation) → `run_parity.py`. Confirm:
   fluxtune concurrency recovers toward real ~5.8, wall gap closes, U3 non-trivial, clock-family rungs now
   read on the `data_id` axis, V1/V2 within tolerance on the longer run. Then the Phase-1 convergence
   sign-off (§I.6).

### §L.5  ✅ Launcher fail-fast (D-d) — parallel tooling task (no parity coupling) — DONE

Independent of the aggregation work; do in the harness/launcher. Two changes in `flame/launch/`:
- **Detect early aggregator death before committing to trainers.** `aggregator_spawner.wait_until_ready`
  (`:135`) should treat a process exit during the window as `False` immediately (it already checks
  `is_running()` each second — return `False` on death instead of the 5s "alive ⇒ ready" heuristic; better,
  watch the agg log for a readiness marker / a traceback). If not ready → abort the run, do NOT
  `spawn_all` trainers.
- **On aggregator non-zero exit, terminate trainers immediately.** `runner.py:324-327`: when
  `agg_rc not in (0, None)`, skip `wait_all(timeout_per_trainer=30.0)` and call the trainer spawner's
  terminate path now (the run has failed; the 30s EOT grace is only meaningful on a clean finish).
*Exit:* a crashed-aggregator run returns in ~seconds, not ~90s; the harness health line already reads
CRASH (done this session). Optional: surface the aggregator's traceback tail in the runner output so the
operator sees the cause without opening the log.

### §L.6  Locked decisions

- **D5 — fluxtune staleness treatment: RESOLVED → (a) fedbuff staleness-weighted accept (K-D13).** A
  carried grad trained on `model_version` V that commits at V′ is **accepted and down-weighted by `V′−V`**
  (fedbuff default, faithful to an async buffered aggregator). Rejected: (b) `round_data_id` reject+record
  (fwdllm_plus's sync policy — discards work an async baseline is designed to consume); (c) unset→`none`
  (no staleness signal for U3). **Provisional lock** — keep fedbuff-accept for the Phase-1 async parity
  pass; MAY revisit later. It is a baseline *definition*, not a parity lever (PARITY.md §F.4 / principle
  #3): **set identically real+sim, never tuned to close a gap.** Obligation: confirm the fedbuff optimizer
  weighting path is wired and `staleness = V′−V` emits on the commit event so U3 can score it.
- **Verify-real-first is a HARD GATE.** Real correctness is established BEFORE any sim mechanism change
  (§L.3 sanity gate blocks §L.4 step 4). If real is itself wrong (R1 overlap > 0, or real drops carried
  grads), we **fix real first**, re-bank it as the reference, then bring sim to parity — never tune sim to
  an inadmissible real (PARITY.md "verify real is correct first — a gap has two fix directions").
