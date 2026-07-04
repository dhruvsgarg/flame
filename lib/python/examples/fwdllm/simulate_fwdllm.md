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
| 6 | Slot release | per-commit (inside `_sim_recv_min`), hold slot to COMMIT | **same as felix — hold slot to COMMIT** (`selected_ends` AND `all_selected` held for every dispatched-but-not-committed trainer, released on commit) | a returned-but-uncommitted trainer is still in flight in VIRTUAL time (grad commits when vclock reaches sct), so its slot is occupied; K-D16 briefly freed it on physical RETURN → undercounted `in_flight` 3× (K-D17b reverted, realigned with felix) | **K-D5**/**K-D16**/**K-D17b**, principle #4 |
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
  survive the rollback boundary; triplet guard populated on grad RETURN (**corrected in K-D17** — the K-D16
  as-built stamped it at DISPATCH, which froze the pool before the first commit and deadlocked the re-run).
  See §H open-root #1/#1b for the root-cause and K-D16/K-D17 for the as-built. *Original options analysis
  kept below for the record.* The bug: the
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
11. **Telemetry-FIRST, then instrument, then (rarely) run — do not launch runs to guess-and-check a
    hypothesis.** A cluster run is the single most expensive and slowest step in this loop; treat it as the
    LAST resort, never the debugger. (Codifies async_cifar10/PARITY.md "Workflow policy" #1–#2 for fwdllm.)
    - **(a) Validate/refute a hypothesis from the telemetry ALREADY ON DISK before considering a run.** Every
      banked run dir carries per-commit/per-round/per-phase telemetry + the aggregator log; most roots
      (the R1 overlap in `[SIM_GRAD_RECV]`, the #6 rate gap in the per-phase timings, the #7 real-speed surplus
      in the `runtime.py` decorator lines) are already visible there. If you catch yourself saying "let's run it
      and see," stop — first name the exact stored field/line that would confirm or kill the hypothesis, and go
      read it. A run that only re-confirms what the logs already show is wasted.
    - **(b) When you implement anything new, ship its telemetry + plot + pytest IN THE SAME CHANGE.** Over-
      instrument pre-emptively (cheap to log, expensive to re-run for): add the per-event field that makes the
      new mechanism's correctness DIRECTLY observable in the checker (not inferred), a plot if it has a
      trajectory, and a pytest that drives it on synthetic telemetry and asserts the invariant. This is what
      lets (a) work next time: the R1 regression was catchable only because the diagnostic line existed. A fix
      with no new instrument/test is not done.
    - **(c) A run is justified only to observe an EMERGENT quantity that no stored telemetry can yield** (fresh
      convergence, a concurrency/wall number after a mechanism change, a cadence trajectory over more
      `data_id`s than any banked run reached) — and then run the SHORTEST length that exhibits it
      (PARITY.md run-length budget), smoke first, one mechanism per round.
    - **(d) Cross-pollinate fluxtune ↔ fwdllm/fwdllm_plus.** fluxtune is furthest from parity, so it surfaces
      roots first; when a fluxtune fix or learning could apply to the sync baselines (e.g. the #6/#1 shared
      time-base question, a selector-ignores-hold-set bug, a duration-input anchor), note it in §B.1/§K and
      check the others from THEIR banked telemetry before assuming it's async-only.
12. **Consult async_cifar10/PARITY.md's vclock principles BEFORE touching any sim/vclock behavior.** The virtual
    clock is a shared flame-core capability that async_cifar10 already exercises correctly across sync AND async;
    its rules are settled there. Before changing how the sim clock advances/increments/orders in any scenario,
    read the relevant PARITY.md principle and mirror it — deviate only with a logged §K rationale (DESIGN
    PRINCIPLE at the top). The settled rules (anchors verified 2026-07-04):
    - **Clock is a monotone `max`.** `VirtualClock.advance(ts)` moves forward only (`flame/sim/virtual_clock.py:32-37`);
      `_advance_sim_clock(sct)` = `vclock = max(vclock, sct)` (`syncfl/top_aggregator.py:331-343`). K1
      `sim_commit_order_monotone` asserts `vclock_now` non-decreasing.
    - **NEVER put overhead on the vclock.** `_sim_commit_overhead_s` defaults 0.0; PARITY.md dead-end forbids >0
      ("masks & drifts; clock must `= max(vclock, sct)`"). K3b `overhead_residual` catches an uncharged/overcharged term.
    - **sct = `sim_send_ts + max(gpu, D) + leg`** (`async_cifar10/.../main.py:831-845`); the `leg` goes on the vclock
      but NOT on `trainer_speed_s`/utility/gate inputs (keep those pure compute).
    - **sync charges MAX-of-K sct; async charges the K-th-fastest** (streaming) — the one real sync/async increment
      difference (PARITY.md K3a). Starvation = a clock JUMP to the next availability event, never a spin.
    - **The sim SKIPS real waits and reconstructs order from sct** (`SimReorderBuffer`, `virtual_clock.py:52-60`);
      `mqtt_fetch_s` (re-selection wait) and `realDistributeSettleSeconds` are deliberately OFF the vclock.
13. **The vclock is virtual wall-time; the sim MUST produce SPEEDUP (vclock ≥ physical-wall-elapsed).** (Operator's
    model, locked.) If a trainer's work is 10 real wall-seconds, the vclock advances 10s while the sim executes it in
    far less wall — same events, same order, same computation (fidelity preserved), but faster in time (the whole
    reason the sim exists). Therefore, **for a healthy sim, `vclock.now ≥ physical_wall_elapsed` at all times**, and
    `sim_rate = vclock/wall ≥ 1` and should GROW. A `sim_rate < 1` (as in the 2026-07-04 runs, 0.37×) means the sim is
    a slowdown = BROKEN: either it under-charges the vclock (small/zero D) or it fails to skip a real wait (the leak),
    or both. **fwdllm nuance:** the forward-grad "train" is a REAL GPU pass (~2s) that MUST run in sim for grad
    mode-invariance — that GPU wall is irreducible. So sim wall ≈ gpu + (skippable transport); speedup comes from
    skipping the transport/inter-round waits, NOT from skipping compute. `sim_rate` is the top-line health metric to
    emit and drive EVERY run — async_cifar10 already logs it (`[VCLOCK_PROGRESS]`, `syncfl/top_aggregator.py:1189-1203`;
    K7 `sim_rate_ok`); fwdllm must too (root S3 / issue #13).

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

**Status (2026-07-04 PM → the K-D21 Phase-1 sign-off run RAN but every pair is UNUSABLE — all 6 processes hit
the 600s WALL, none reached a natural stop, and no `parity_*.json` was generated.** `bash run_sequential.sh
--mode both --delays on --max-runtime-s 600` produced `run_20260704_133627..143445` (real+sim × 3 baselines).
Two upstream mis-configs made the run non-comparable (analysis-from-telemetry, principle #11a — NOT re-run):
> - **(root S1 — the "timeout") Every SIM died on `[SIM_WALL_CEILING]`, not the intended vclock budget.** The sim
>   does REAL forward-grad GPU compute for every dispatched trainer, so physical wall (566–588s) hit the 600s
>   ceiling while the vclock only reached **fwdllm 213 / fwdllm_plus 221 / fluxtune 85 s** (0.14–0.37× of the 600
>   budget). Piece A set `sim_wall_ceiling_s = max_runtime_s = 600`, so for a real-compute sim (wall ≫ vclock by
>   construction, #6/#9) the ceiling ALWAYS fires before the vclock-budget stop it was paired with — exactly what
>   piece A's own comment warned ("a vclock-bounded sim over-runs to the wall ceiling"). Real stopped on
>   `max_runtime_s` wall too. So the two sides stop at DIFFERENT progress and cannot be compared:
>   | baseline | real rounds·data_id·wall/rnd | sim rounds·data_id·vclock(wall)·wall/rnd | stop |
>   |---|---|---|---|
>   | fwdllm | 36 · **13** · 15.5s | 66 · **23** · 213s(566s) · 8.6s vc-3.2s | real max_runtime / sim CEILING |
>   | fwdllm_plus | 8 · **4** · 59.7s | 69 · **24** · 221s(588s) · 8.5s vc-3.2s | real max_runtime / sim CEILING |
>   | fluxtune | 133 · **15** · 3.6s | 103 · **9** · 85s(579s) · 5.6s vc-0.8s | real max_runtime / sim CEILING |
> - **(root S2 — CORRECTED after codepath trace) `--delays on` WAS honored by the trainer, but D was tiny (0.4–1.8s,
>   not the 4–18s registry delay) and the aggregator sim-folds were never wired.** The literal "delays didn't reach
>   the trainer" was WRONG: real trainers slept 0.4–1.8s (`FedSgdTrainer.py:530`), sim trainers modeled 1.0–1.3s and
>   charged it to the vclock (`:524`). Three real gaps: (1) `training_delay_factor=10` (hardcoded
>   `configs/trainer_base.yaml:96`) silently divides the registry 4–18s delay ÷10 → 0.4–1.8s; (2) the aggregator's
>   logged `training_delay_enabled=False` is an **orphaned field the fwdllm aggregator never reads** (the sim D flows
>   from the trainer-reported `sim_completion_ts`, not an agg hyperparam) — a misleading artifact; (3) the sim sct-model
>   folds (`sim_model_eval_time`, `sim_straggler_spread_s`) were never patched by the launcher (they live only in
>   `aggregator.config_overrides.hyperparameters`, which `--delays` never writes); (4) the banked
>   `execution_config.yaml` OMITS `enable_training_delays`, so post-hoc the run looked off. Reframed **open issue #12**.
>   NOTE: even a correct large D does NOT by itself give speedup — see root S3.
> - **(root S3 — THE headline: the sim delivers a SLOWDOWN, not speedup — it still pays real waits it must skip).**
>   The whole point of the sim is speedup: do the SAME events in the SAME logical order but advance the vclock
>   (virtual wall-time) FASTER than physical wall by not waiting. Measured, the sim runs at **sim_rate ≈ 0.37×**
>   (vclock 3.2 v-s/round vs physical wall 8.6 s/round) — SLOWER than wall. Root (codepath trace): the sim trainer
>   still physically blocks on `await_join()` + real MQTT `recv_wrapper` every round (`fwdllm_trainer.py:213,218`,
>   `mqtt_fetch_s`~1.8s steady / ~18s round 1), the aggregator still pays grace-bounded `recv_fifo` (≥2.0s floor,
>   `syncfl/top_aggregator.py:347`) + `await_join` on distribute (`fwdllm_aggregator.py:2280/2427`), and ungated
>   `time.sleep(1)` backoffs — **none gated `and not self.simulated`**. Real GPU compute (~2s) is irreducible in sim
>   (needed for grad mode-invariance), so speedup REQUIRES skipping the transport waits (async_cifar10's base does;
>   fwdllm's per-round sync trainer fetch does not). The invariant to restore: **vclock ≥ physical-wall-elapsed**
>   (operator's model). New **open issue #13**; tracked via a `sim_rate` metric fwdllm does not yet emit.
> - **fwdllm_plus real is pathologically slow even at syn_0 (no scarcity): 59.7 s/round (8 rounds, data_id 4)** vs
>   fwdllm real 15.5 s/round — ~4×. This is NOT the Stage-C scarcity liveness (#7, syn_0 has 100% avail); it is the
>   `reselect_each_iteration=True` per-iteration re-selection + oracular read overhead (29 selection events vs
>   fwdllm's 1). Confirms the "why is fwdllm_plus real slower/round" half of #7 as a distinct real-config cost, and
>   guarantees fwdllm_plus real can never reach the sim's data_id in a wall budget (#4 real-side truncation).
> **Principled fix (design, not yet implemented — awaiting go-ahead):** for a valid parity/sign-off run the
> stopping rule must be a **matched committed-`data_id` target** (finite `--max-data-id`, e.g. 10), NOT a wall or
> vclock budget — a time budget desyncs wall-bound real from ceiling-truncated sim (the overnight 10-`data_id`
> runs passed for exactly this reason). Couple that with: (1) fix the `--delays on` wiring so D>0 reaches BOTH the
> trainer sleep AND the sim sct model (#12); (2) decouple `sim_wall_ceiling_s` from the vclock budget (generous
> multiple / separate `--max-wall-s`) so the ceiling is an outer safety, not the primary stop; (3) resolve
> fwdllm_plus real slowness (#7) so its real side can reach the matched data_id. See open roots S1/#12 below.

**Status (2026-07-05 → overnight 10-`data_id` grounding runs BANKED; parity read below (K-D18). The fluxtune
R1 residence regression (K-D17b) is STILL the top blocker — do it first (fluxtune ran real-only overnight, so
its sim is unchanged from run `020254`, R1=44.7%).** New findings from the 10-`data_id` runs (bigger than the
prior 3-`data_id` runs — the operator intended 1 h but the `--max-data-id` default was 10, now raised to 9999,
K-D18):
> - **(a) fwdllm sync cadence is HEALTHY at 10 `data_id`s.** V1/V2/g2/R1/W1/U3/S2/conv/conv_loss all PASS —
>   the #4 short-run truncation cleared exactly as predicted (37 pass / 5 fail).
> - **(b) fwdllm's surviving fails are the clock-RATE family** (throughput/total_commits/terminal_state), NOT
>   the #2 axis re-key: at a matched virtual budget sim charges **~7.8 vclock-s/round** while real spends
>   **~34.9 wall-s/round** (both completed all 10 `data_id`s). New **open root #6** — must classify real's
>   ~27 s/round surplus as real-transport overhead (correct to exclude, principle #1/#8) vs a sim D under-charge,
>   FROM the existing per-phase timing telemetry, before touching the clock. **Shared-shaped with the fluxtune
>   2.4× wall gap** — the cross-baseline learning to watch.
> - **(c) fwdllm_plus real is pathologically slow.** It ran the FULL 1 h (`max_runtime_s=3600` cap) and reached
>   only **`data_id`≈3** (86 wall-s/round; sim did all 10 in 81 vclock-s / 274 wall-s). So its V1/V2/g2/
>   eligibility/selection fails are REAL-side **truncation** (comparing sim's 10 full `data_id`s to real's 3),
>   plus a genuine **real-speed root #7** (why is fwdllm_plus real ~2.5× slower/round than fwdllm real? — the
>   per-iteration reselection is the suspect; localize from telemetry, not a re-run).
> - **(d) field_coverage now fails on a real telemetry GAP** — `task_recv.sim_send_ts` is absent in both modes
>   (this is NOT the #3 gpu/budget alias, which passes). New **open root #8** (INV). `failsafe`/K5 also fires
>   falsely for fwdllm (sim WALL ≫ vclock because forward-grad compute is REAL) — a checker exemption, root #9.
>
> _Original 2026-07-04 EOD state: K-D17 + K-D17b landed; in_flight measurement FIXED (sim 9.49 vs real 9.75)
> but K-D17b introduced the R1 residence regression (0%→44.7%). Resume from the "NEXT SESSION" block below._
> **NEXT SESSION (2026-07-05) pickup — start here.**
> **UPDATE: #1c (R1) is now FIXED in code — K-D19.** The prime hypothesis below was **wrong on the mechanism**
> (async_oort *does* exclude `all_selected`; the NONE reset is a red herring). The real root was the guard
> being released on physical grad RETURN before the carried grad committed; fixed fwdllm-only via
> `_release_end_on_return` (defer release to COMMIT), + a `[SIM_R1_DISPATCH]` tripwire + a RETURN-path pytest,
> `tests/mode -k fwdllm` 110 green. **The only step left on #1c is the emergent smoke** (`bash
> run_sequential.sh --mode both --delays on --max-runtime-s 3600 --only fluxtune --yes`, or a shorter
> `--max-data-id 3` smoke first) → `run_parity.py --baselines fluxtune`, confirm R1≤2% + `in_flight`~9.5.
> _The original hypothesis + fix-options are kept below only as the record of what was refuted (K-D19)._
> **[REFUTED] R1 prime hypothesis:** `_sim_hold_busy_slots` resets buffered ends to `KEY_END_STATE=NONE`… but
> async_oort selects by avail/end-state and does NOT exclude `all_selected`, so a NONE-state buffered trainer
> is re-selectable → R1 overlap. **Refuted:** async_oort excludes `all_selected` (`async_oort.py:1586-87`) and
> the channel passes the full pool regardless of `KEY_END_STATE`; exclusion is by `all_selected` membership
> alone, and the trainer left the guard on RETURN. See K-D19.
> **Then:** re-check `selection_detail.chosen` (sim 0.83 vs real 1.76 — likely follows R1). The **longer run
> is already banked** (the overnight 10-`data_id` runs, K-D18) and cleared #4 for fwdllm; for fluxtune the
> fresh sim re-run is still owed (overnight was real-only). When re-running, the `--max-data-id` default is now
> **9999** (K-D18) so `--max-runtime-s` governs; pass `--max-data-id 10` only for a deliberately short run.
> Launch (after R1 is clean): `bash run_sequential.sh --mode both --delays on --max-runtime-s 3600 --only
> fluxtune --yes`. **First-principle before any launch: exhaust the existing telemetry** (principle #11) — the
> R1 root is already visible in run `020254`'s `[SIM_GRAD_RECV]` log; validate the fix there + by pytest, not
> by a launch.
> **Do NOT redo:** the in_flight measurement fix (K-D17b) is correct and validated — keep the hold-to-commit;
> only the state-reset that re-opens selection is at fault. The temporary `[SIM_GRAD_RECV] ... inflight_exp=/
> sel_ends=` diagnostic in `_sim_recv_min_grad` is useful — keep it until R1 is closed, then optionally trim.

**Status (2026-07-04 → K-D17 landed; the K-D16 re-run DEADLOCKED, root-caused to two bugs, now fixed —
awaiting a fresh fluxtune re-run).** Batches 1–2 + §L Batch-2.5 + Option-A (K-D16) + clock-family re-key (#2)
+ field alias (#3) + the **K-D17 drain-gate + triplet-at-return fixes** are landed & pytest-green (full
`tests/mode` 314 passed / 7 skipped). The **post-K-D16 fluxtune smoke (`run_20260703_230407`) did NOT
recover — it deadlocked** (1 cohort dispatched, only 2 grads committed, 0 `agg_round` events, in-flight
0.28, 1663 empty select-rounds). Root cause = **two independent bugs (K-D17)**: (A) the sim drain loop
gated on channel RECV state instead of the `_sim_buffer` — the greedy first drain emptied RECV and stranded
the other ~8 already-received grads (`"no ends yet"` ×1681), so agg_goal=3 was never met from a full cohort;
(B) K-D16 stamped the async_oort re-pick triplet at DISPATCH, freezing the whole eligible pool before any
commit could advance the version (bootstrap case the D6 note missed). Both fixed: drain now keys on the
buffer/in-flight set; the triplet is stamped on grad RETURN. Residence (R1) exact on all three; grad values
mode-invariant. **Remaining action: a fresh fluxtune re-run** to bank concurrency (vs real ~5.8) / wall / U3
/ V1 / selection, then the **longer run (`max_data_id_progress=10`)** to clear #4. The table below is the
PRE-K-D16 baseline the re-run is measured against (the K-D16 run is void — it deadlocked).

**Per-baseline ground state (2026-07-05, overnight 10-`data_id` runs; K-D18):**
| baseline | run (real→sim) | R1 | V1/V2/g2 cadence | S2/U3/conv | surviving fails | status |
|---|---|---|---|---|---|---|
| **fwdllm** (sync) | 10 `data_id`s both; wall 349→184s | PASS (0%) | **PASS / PASS / PASS** | PASS / PASS / PASS | throughput·commits·terminal (#6 clock-RATE), field_coverage (#8), failsafe (#9 false-pos) | **37 pass / 5 fail.** Cadence CLEAN at 10 `data_id`s (#4 cleared). Remaining fails all trace to the RATE gap #6 + two checker/telemetry gaps — no cadence bug |
| **fwdllm_plus** (sync) | sim 10 `data_id`s (274s wall); **real hit the 1 h cap at `data_id`≈3** | PASS (0%) | FAIL / FAIL / FAIL | PASS / PASS / PASS | +eligibility, selection_detail (real_chosen 4.88 vs sim 4.39) | **31 pass / 10 fail.** Cadence fails are **REAL-side truncation** (3 vs 10 `data_id`s) driven by real being ~2.5× slower/round (#7); NOT confirmed a sim bug until real completes 10 |
| **fluxtune** (async) | overnight **real-only** (10 `data_id`s); sim unchanged from `020254` | **FIXED in code (K-D19)**; was 44.7% | V2 PASS / — | — | (re-run owed) | **#1c FIXED + pytest-green.** Root = guard released on grad RETURN, not the doc's `all_selected` hypothesis; deferred to COMMIT via `_release_end_on_return`. in_flight measurement (9.49 vs 9.75) preserved. **Emergent-only step left:** one sim smoke to bank R1≤2% |

### Parity scoreboard — LATEST vs PENULTIMATE only (rewrite in place each run; do NOT append history)

*Discipline: keep exactly two columns per baseline — the latest parity JSON and the one before it — so drift is
visible without a growing log. `run_parity.py` regenerates both from the banked dirs.*

*NOTE: the 2026-07-04 PM sign-off run (`run_20260704_133627..143445`) produced NO parity JSON — every sim was
wall-ceiling-truncated and real/sim reached mismatched `data_id` (root S1 above), so a parity read would be
meaningless. The LATEST column below is still the last VALID banked pair (the overnight 10-`data_id` runs); it
does not advance until a matched-`data_id` sign-off run is banked.*

| baseline | penultimate (run · pass/fail/skip) | LATEST valid (run · pass/fail/skip) | Δ | failing rungs (latest) |
|---|---|---|---|---|
| **fwdllm** | `0703_224641` (3 `data_id`) · 37/5/32 | `0704_022610` (10 `data_id`) · **37/5/32** | =0 | field_coverage, throughput, total_commits, terminal_state, failsafe |
| **fwdllm_plus** | `0703_225614` (3 `data_id`) · 30/11/32 | `0704_033206` (10 `data_id`, real stalled) · **31/10/32** | +1 | above + eligibility, selection_detail, v1, v2, g2_grad_pool_size |
| **fluxtune** | `0704_020254` (sim, R1=44.7%) · — | overnight **real-only → no sim pair** · — | — | (R1 fixed in code K-D19; re-run owed) |

### SKIP audit — 32 skips, only ~16 are legitimate (rest are rigor gaps to close)

The user rule: skips are OK only for a true **example/config disparity** (the mechanism genuinely isn't in this
baseline); otherwise add the equivalent fwdllm test. Breakdown of the 32:
- **~16 LEGIT (Phase-1 syn_0 + `random` selector; correct to skip):** 7 availability ground-truth rungs
  (`*_trace_fidelity`, `agg_belief_fidelity_*`, `send_gate_wait_*`) + 4 delivery/withheld
  (`duty_cycle*`, `abandon_timeout`, `withheld_delivery`, `commit_promptness`, `state_timeline_agreement`) —
  the **availability effect path is Phase 2 (Batch 3)**, not built; + 3 DynamicKC (`dk1/dk2/dk3`) disabled by
  design; + 2 oort-only (`selector_score`, `preferred_duration`) — fwdllm uses `random`; + `residence` (async
  telemetry). No equivalent test exists because the mechanism isn't present. *(These un-skip as Phase 2 lands.)*
- **4 clock-advance rungs skip on the WRONG AXIS → RIGOR GAP (root #10):** `modeled_compute_advance` (K3a),
  `overhead_residual` (K3b), `per_round_advance` (K3), `overlap_factor` (K4) report "fewer than 2 sim rounds"
  even though the sim did 26 — they're still keyed on `round` (fwdllm stays at round=1, advancing `data_id`),
  the exact bug #2 fixed for throughput but never applied to these. **Re-key to the `data_id` axis** (extend the
  `_progress_axis` helper) → they run and directly validate root #6. *(shared parity engine → full-suite gate.)*
- **8 per-phase timing rungs skip on MISSING telemetry → RIGOR GAP = the root-#6 instrument (step 2):**
  `training_budget_s`, `gpu_compute_s`, `mqtt_fetch_s`, `weights_to_gpu_s`, `weights_to_ram_s`, `pre_train_s`,
  `post_train_s`, `trainer_phase`. Adding the per-phase wall telemetry un-skips them AND enables the #6
  decomposition. *(root #10 / step 2.)*

### Open issues — cross-baseline master index (OPEN top, RESOLVED moved down; 1-sentence issue + fix)

*Rewrite in place: when an issue resolves, move its row into the RESOLVED block with a crisp principled-fix
summary. Detailed rationale lives in the "Open roots" prose below + §K.*

| # | issue (1 sentence) | baseline(s) | principled fix / next step |
|---|---|---|---|
| **#13** ⭐⭐ | **sim delivers a SLOWDOWN not speedup (`sim_rate`≈0.37×).** Phase-2 skips **LANDED (K-D23)**: the doc's ranking was INVERTED by the `134801` telemetry + async_cifar10 map — the trainer `await_join`/`recv_wrapper` is barrier-wait that OVERLAPS irreducible agg eval+GPU (recv MUST stay), and the agg grace/`sleep(0.1)`/`await_join` were ALREADY sim-correct. The genuinely skippable per-round wall was the trainer `pause_execution` throttle + the `_check_availability` avail-spin. | fwdllm, fwdllm_plus (+ fluxtune) | **Phase-2 skips DONE (fwdllm-only, `and not self.simulated`, 146 green).** REMAINING for real-run `sim_rate>1`: **#6/Phase-4c** must fold `eval_s` (~13s/data_id, the syn_0 wall dominator) into the vclock, then the sign-off run confirms `vclock ≥ wall` live. Phase 2 exit (`sim_rate>1` on SYNTHETIC timings) is met. |
| **S1** ⭐ | sign-off run unusable: every sim died on `[SIM_WALL_CEILING]` (wall≫vclock for a real-compute sim), real died on `max_runtime_s` wall, so real/sim reached mismatched `data_id` — no comparison possible | all (2026-07-04 PM) | **FIXED (Phase 4a / K-D24):** `sim_wall_ceiling_s` decoupled to `max_runtime_s × 20` (was 1×), so the ceiling is a runaway OUTER safety, not the primary stop — a real-compute sim is no longer truncated before its vclock/data_id stop. Sign-off run should also **stop on a matched `--max-data-id`** (present: `max_data_id_progress`), not a wall/vclock budget. |
| **#12** | `--delays on` WAS honored by the trainer (real slept 0.4–1.8s; sim modeled it), but (a) `training_delay_factor=10` (`trainer_base.yaml:96`) shrank the 4–18s registry delay ÷10, (b) the agg-side sim-folds (`sim_model_eval_time`/`sim_straggler_spread_s`) were never wired from the launcher, (c) agg `training_delay_enabled` is an orphaned/misleading field, (d) banked `execution_config.yaml` omits the flag | all | **Config-flow fix (Phase 3 below).** Fan `--delays`/folds into `config_overrides.hyperparameters` for BOTH roles (mirror `agg_goal` single-source); expose `training_delay_factor` as a knob; bank the effective values + a post-launch assertion that banked config == requested flags; retire/wire the orphan. Extensible to async_cifar10 (same `runner.py`). |
| **#6** ⭐ | sim vclock under-models real wall — genuine unmodeled terms are `eval_s` (server eval, the syn_0 wall dominator) + barrier straggler under-spread. | fwdllm, fwdllm_plus (+ fluxtune 2.4× wall) | **FOLDS ENABLED (Phase 4c / K-D24):** B1 `sim_model_eval_time:true` charges the MEASURED eval_s to the vclock (self-calibrating, no constant); B2 `sim_straggler_spread_s:0.9` widens the barrier spread — both live in the 3 sim yamls. Do NOT add drain-tail/sleeps/localhost-MQTT (artifact, principle #1). `wall_disparity` DIAG rung (data_id axis) drives the residual → ~0; **final B2 value is calibrated FROM the sign-off run** (emergent). Ceiling decoupled (4a) so the vclock-budget stop is now reachable. |
| **#7** | fwdllm_plus real STALLED — 5 agg-rounds then dead-spun to the 1 h cap (log ballooned to 292 MB); **AND (new, syn_0) real is 4× slower/round even with NO scarcity** (59.7 vs fwdllm 15.5 s/round, 8 rounds/data_id 4 in 600s) | fwdllm_plus | **Scarcity half ROOT-CAUSED (K-D20) + FIXED (Stage C):** oracular `mobiperf_2st` avail 1/10 < `agg_goal=10`; real sync barrier spun. **Slowness half (NEW, syn_0):** `reselect_each_iteration=True` per-iteration re-select + oracular read (29 sel events vs fwdllm's 1) — a distinct real-config cost, profile from the banked per-phase log; it caps fwdllm_plus real's data_id below sim's (#4 real-side truncation). Not a sim bug. |
| **#8** | `field_coverage` INV fail: `task_recv.sim_send_ts` absent both modes | fwdllm, fwdllm_plus | Emit `task_recv.sim_send_ts` (over-instrument) or add it to the coverage alias tuple; unblocks K6. |
| **#9** | `failsafe`/K5 false-positive: sim WALL ≫ vclock (real forward-grad compute) | fwdllm, fwdllm_plus | K5 should compare sim wall vs the RUN wall budget, not the vclock, for a real-compute sim (checker fix). Partly reframed by piece A's `sim_wall_ceiling_s`. |
| **#10** | 12 rungs SKIP as rigor gaps (4 advance rungs wrong-axis, 8 phase rungs missing telemetry) | fwdllm, fwdllm_plus | Re-key K3a/K3b/K3/K4 to `data_id` (shared engine → full suite); add per-phase timing telemetry (= step 2, feeds #6). |
| **#11** | real-mode critical-path waste surfaced by profiling: fixed `time.sleep(0.1)` pads (≥28.7 s/run, `fwdllm_aggregator.py:2209/2338`) + one-grad-per-poll-tick drain tail (2.48 s/round) | fwdllm, fwdllm_plus (real) | Correctness-preserving optimization (profile-driven): verify each sleep's purpose, gate/remove off the critical path; drain-tail is the real 1-per-pass clamp (leave the sim all-k barrier). Watch GPU-shared trainer compute. Do NOT change grad values / cadence. |
| — RESOLVED ↓ — | | | |
| ~~#1c~~ | R1 in-flight overlap 0%→44.7% (fluxtune) | fluxtune | **FIXED K-D19** — root was guard released on grad RETURN, not `all_selected`; deferred release to COMMIT (`_release_end_on_return`), fwdllm-only. Emergent smoke owed. |
| ~~#1~~ | async concurrency starved to ~1.5 vs real ~5.8 (D-e) | fluxtune | **FIXED K-D16** — Option-A two-lifetime split (free slot on return, hold guard to commit). |
| ~~#1b~~ | fluxtune K-D16 re-run deadlocked | fluxtune | **FIXED K-D17** — drain keys on `_sim_buffer` not RECV; triplet stamped at RETURN not dispatch. |
| ~~#2~~ | clock family collapsed to `sim_rounds=1` | all | **FIXED** — re-key throughput/commits/terminal to the `data_id` axis (`_progress_axis`). Exposed #6. |
| ~~#3~~ | field_coverage false-fail on gpu/budget field names | all | **FIXED** — coverage spec accepts fwdllm aliases (tuple of names). |
| ~~#4~~ | V1/V2/g2 short-run truncation | fwdllm (cleared), fwdllm_plus | **CLEARED for fwdllm** at 10 `data_id`s; for fwdllm_plus it's real-side truncation → folds into #7. |
| ~~#5~~ | fluxtune `selection_detail` in_flight (2.7 vs 9.75) | fluxtune | **MOSTLY FIXED K-D17b** — hold the compute slot to COMMIT so `len(selected_ends)` = virtual in-flight. |

### ⭐ Next implementation plan (2026-07-04, post-sign-off) — SPEEDUP + CONFIG-FLOW + PARITY, phased

*This supersedes the "Pre-next-run implementation plan (K-D21)" below (that plan's sign-off run RAN and was
unusable — see the 2026-07-04 status block at the top of §H). The sign-off run surfaced THREE root causes that
must be fixed, pytest-gated, BEFORE any next launch (principle #11). Derived from three codepath traces
(async_cifar10 vclock principles; the bash→yaml→runtime config flow; the fwdllm sim vclock/wait machinery).
Phases 0→1 are prerequisites; Phases 2 (speedup) and 3 (config-flow) are INDEPENDENT parallel threads; Phase 4
depends on Phase 2. Every phase ships telemetry + pytest and ends pytest-green (no run in the inner loop).*

**PROGRESS (2026-07-04, update in place):** `✅ Phase 0` principles #12/#13 locked · `✅ Phase 1` speedup
instrumentation — `wall_elapsed_s`+`sim_rate` on agg_round, live `[VCLOCK_PROGRESS]` log, `sim_speedup` DIAG rung
(VALIDATED on the banked `134801` sim: sim_rate=0.377, wall_speedup=0.986, "SLOWDOWN"), rung+emit pytests green;
**Phase 1d PDF plot DEFERRED** (metric already in telemetry+log+rung; the analyze_run.py plotter add is low-value,
do with Phase 4) · `✅ Phase 3` config-flow (operator chose 3b framework fan + 3c factor knob) — runner.py fans
`enable_training_delays`(+factor) into the aggregator (single source, both roles agree; safe: `training_delay_enabled`
is read ONLY by the trainer, never the agg base, so async_cifar10 is behavior-neutral), honored-100% tripwire in
`_build_aggregator_config`, effective flags banked in `execution_config`, `--delay-factor` CLI, fan+bank pytests green
· `✅ Phase 2` **speedup leak (root #13) — DONE, pytest-green (K-D23).** 2a decision made FROM the banked
`134801` telemetry (principle #11a): the trainer inter-round `mqtt_fetch_s` is barrier-wait realized by the
**blocking recv** (~15s at each data_id boundary OVERLAPS the aggregator's irreducible server eval ~13s +
real GPU — NOT an independent skippable sleep; recv MUST stay, it delivers the real weights for grad
mode-invariance, exactly as async_cifar10 keeps real MQTT in sim). The only additive, fidelity-free per-round
wall the sim skips is the **trainer `pause_execution` throttle** (`time.sleep(1)` at the tail of EVERY loop
iteration, "don't overwhelm mqtt" — a real-transport artifact, principle #8) + the **`_check_availability`
avail-spin** (`while UN_AVL: sleep(1)`, which would freeze the vclock in sim; avail is agg-side enforced). Both
gated `and not self.simulated`; real byte-identical; `tests/mode -k fwdllm` 146 green (+6 new,
`test_fwdllm_sim_speedup_waits.py`). **Turned out fwdllm-ONLY (no shared-base edit)** — the aggregator was
already sim-correct (`_sync_sim_recv_first_k` 2s grace not 30/90s; `sleep(0.1)` pads gated real-only, Stage E;
`await_join` kept per async_cifar10; agg `pause_execution` not in the loop), so per principle #9 the scoped
suite is the gate, not full `tests/`. **REFRAME:** the leak inventory ranked "trainer await_join+recv" as
DOMINANT (#1) and "sleep(1) backoffs" LAST (#4); the telemetry + async_cifar10 map INVERT this. **Phase 2 alone
does NOT push the REAL-run `sim_rate>1`** — at syn_0 the sim wall is eval-dominated (#6/Phase-4c folds `eval_s`
into the vclock) — its exit is `sim_rate>1` on SYNTHETIC timings + pytest, which is met. · `✅ Phase 4`
**stopping rule + #6 sct model — DEV DONE, pytest-green (K-D24); emergent calibration is the sign-off run's
job.** 4a: `sim_wall_ceiling_s` DECOUPLED from the vclock budget (default `max_runtime_s × SIM_WALL_CEILING_
FACTOR=20`, explicit override honored) so a real-compute sim (wall ≫ vclock by construction) is no longer
truncated before its vclock/data_id stop — root S1's direct fix; `test_fwdllm_early_stop_conditions.py` 19
green. 4b: matched committed-`data_id` stop already present (`max_data_id_progress`; yamls 9999, `--max-data-id`
governs). 4c: the B1 eval-fold (`sim_model_eval_time`, self-calibrating on the MEASURED eval_s) + B2 straggler
spread (`sim_straggler_spread_s=0.9 ≈ real-compute-std·√12`) ENABLED in all 3 sim yamls (operator chose B1+B2);
`wall_disparity` DIAG rung (already landed, keyed on the fwdllm `data_id` axis) drives the residual → ~0 at the
run. **fwdllm-only edit (4a) → `tests/mode -k fwdllm` 148 green is the gate.** **NEXT: the Phase-1 sign-off
RUN** — matched `--max-data-id`, `--delays on`, `sim_rate>1` confirmed live.

**Scope decisions (which issues live / die):**
- **KEPT & PROMOTED — #13 (the sim is a slowdown, not speedup).** New headline root. Was implicit in #6; now
  first-class per the operator's vclock-is-virtual-wall-time model (principle #13). This is the prize.
- **REFRAMED — S1 & #6** are now understood as *downstream* of #13: the `[SIM_WALL_CEILING]` truncation and the
  "vclock ≪ wall/round" gap are the SYMPTOM of the speedup leak, not independent bugs. Fixing #13 makes the
  vclock-budget stop reachable and shrinks the #6 residual to the genuine unmodeled terms (eval_s, straggler).
- **REFRAMED — #12** is a config-flow bug (Phase 3), not a "flag dropped to trainer" bug (the trainer honored it).
- **My earlier S1 "matched-`data_id` stopping rule"** is DEMOTED to a Phase-4 convenience, not the primary fix —
  the real fix is #13 (make the sim fast so a time budget is honored on both sides). Kept, subordinated.

---

#### Phase 0 — Principles & invariant (DONE in this doc edit; no code)
Locked **principle #12** (consult async_cifar10/PARITY.md vclock rules before any sim-clock change) and
**principle #13** (vclock = virtual wall-time; sim must produce speedup, `vclock ≥ physical_wall_elapsed`,
`sim_rate ≥ 1`). These are the acceptance criteria the rest of the plan is measured against. *No pytest — doc only.*

#### Phase 1 — Speedup instrumentation (make the slowdown OBSERVABLE; prereq for 2 & 4)
*You cannot fix or verify #13 without a `sim_rate` metric — fwdllm emits `vclock_now` but no paired wall stamp
(principle #11b: ship the instrument with the mechanism). Mirror async_cifar10 which already has this.*
- **1a** Emit a per-round **`wall_elapsed_s`** (= `time.time() − agg_start_ts`) alongside `vclock_now` in the
  fwdllm `agg_round` `extra` (`fwdllm_aggregator.py` `build_agg_round`, ~:1627-1655), so `sim_rate` is joinable
  per round without external data.
- **1b** Add the **`[VCLOCK_PROGRESS] … sim_rate=…`** periodic log to the fwdllm sync/async loops (mirror
  `syncfl/top_aggregator.py:1189-1203`; confirm whether fwdllm inherits/drives it — if not, wire it).
- **1c** New parity DIAG rung **`sim_speedup`** (`sim_rate = final_vclock / sim_wall`, and `wall_speedup =
  real_wall / sim_wall` at matched `data_id`) in the shared engine; + re-use async_cifar10's **K7 `sim_rate_ok`**
  and **K1 monotonic-vclock** for fwdllm (they exist; ensure they run on the fwdllm axis). Shared engine →
  full-suite gate (principle #9).
- **1d** A speedup plot (sim_rate over `data_id`) in the run's `plots/`.
- **Pytest:** synthetic agg_round series → assert `sim_rate` computed correctly, vclock monotonic, rung PASS/FAIL
  on known-fast/known-slow fixtures. **Exit:** running Phase-1 on the BANKED 2026-07-04 sim dirs reproduces
  `sim_rate≈0.37×` from telemetry alone (validates the instrument before any code that changes behavior).

#### Phase 2 — Fix the speedup leak (#13; THE core work) — make the sim skip real waits
*Goal: sim wall/round → real GPU only (~2s); `vclock ≥ wall`; `sim_rate > 1` and growing. Each sub-step is
`and not self.simulated`-gated or vclock-driven so REAL mode is byte-identical (principle #8), and shared-base
edits (`syncfl/`/`asyncfl/top_aggregator.py`) carry async_cifar10-regression risk → FULL `tests/` (principle #9).*

> **COLD-START LEAK INVENTORY (from the 2026-07-04 codepath trace — don't re-derive).** The sim critical path
> pays these real waits (ranked by cost); each is the Phase-2 work. `agg` = `fwdllm_aggregator.py`,
> `fwd-tr` = `syncfl/fwdllm_trainer.py`, `FedSgd` = `trainer/forward_training/FedSgdTrainer.py`,
> `sync-base` = `syncfl/top_aggregator.py`, `async-base` = `asyncfl/top_aggregator.py`.
> 1. **Trainer inter-round fetch (dominant, ~1.8s steady / ~18s round-1 = `mqtt_fetch_s`):** `fwd-tr:213`
>    `await_join()` + `fwd-tr:217-218` `recv_wrapper` real blocking MQTT recv — UNGATED. (2a; needs the
>    barrier-vs-wire localization first.)
> 2. **Aggregator commit/barrier grace recv:** grad path `agg:756-758` `recv_fifo(..., timeout=grace)` looped to
>    `RECV_TIMEOUT_WAIT_S=30` (`async-base:66`); sync barrier `sync-base:386-398`; `grace` has a **2.0s floor ×4
>    EMA** (`sync-base:347-352`). (2b.)
> 3. **Aggregator `await_join` on distribute:** `agg:2280` (sync) / `agg:2427` (async) — UNGATED. (2b.)
> 4. **Ungated `time.sleep(1)` backoffs / avail spin:** `fwd-tr:205`, `fwd-tr:230`, `FedSgd:457`. (2c.)
>
> **Already correctly skipped in sim — do NOT touch (byte-identical real; re-gating would be a no-op or a bug):**
> post-distribute settle pads `agg:2295-2297`/`agg:2434-2436` (`if not self.simulated: time.sleep(0.1)`);
> scarcity poll `_await_dispatchable_under_scarcity` early-returns in sim (`agg:2232`). And the base async
> pattern to MIRROR: starvation = a vclock JUMP to the next avail event (`async-base:648-672`), the sim SKIPS the
> real delay-sleep while charging `sct` (that's the speedup), `SimReorderBuffer` reconstructs order from `sct`
> not wall (`virtual_clock.py:52-60`). fwdllm's deviation is that its SYNC per-round trainer fetch (leak #1) has
> no async_cifar10 analog and was never gated — that's the core of #13.
- **2a (dominant leak)** The sim trainer `_fetch_weights` blocks on `await_join()` + real MQTT `recv_wrapper`
  every round (`fwdllm_trainer.py:213,218`; `mqtt_fetch_s`~1.8s steady/~18s round-1). **DECISION REQUIRED**
  (design, not obvious): this inter-round fetch wait must become ~free in sim WITHOUT breaking real message
  transport (grads/weights are real). Options to evaluate against async_cifar10's pattern: (i) the wait is really
  *barrier* wait (trainer idle until the aggregator distributes) → it should overlap in virtual time, charged to
  the vclock not physical wall; (ii) on localhost the transport is ms — confirm the ~1.8s is barrier-wait not wire
  time, then the fix is to not physically block the trainer between rounds. Localize from the banked `mqtt_fetch_s`
  + phase telemetry FIRST (principle #11a).
- **2b** Aggregator `await_join` on distribute (`fwdllm_aggregator.py:2280/2427`) and the grace-bounded
  `recv_fifo` (≥2.0s floor, `syncfl/top_aggregator.py:347-352`; `agg:756`) in the sim commit/barrier loops →
  gate/shrink the grace to ~0 in sim (the sct reorder buffer, not a wall timeout, is the sim's ordering source —
  principle #8 "real-transport artifact vs algorithmic property").
- **2c** Ungated `time.sleep(1)` backoffs / avail spins (`fwdllm_trainer.py:205,230`; `FedSgdTrainer.py:457`) →
  gate `and not self.simulated`.
- **Pytest:** spy/mocked-channel test asserting the sim path issues NO blocking real-wait on the critical path
  (or a bounded ~0 grace); a driven test asserting `sim_rate > 1` on synthetic timings; **async_cifar10
  byte-identical** assertion for every shared-base edit. **Exit:** `sim_rate > 1` on a synthetic fwdllm round;
  full `tests/` green.

#### Phase 3 — Config-flow correctness (#12; INDEPENDENT thread) — flag → banked config → runtime, honored 100%
*Trace result: `--delays` reaches the trainer via a bespoke bridge (`runner.py:250`) but NOT the aggregator/sim
folds; there are ~10 layers that can set/override a delay value with disagreeing defaults; the effective value is
not banked. Design goal: one flag → one canonical sink → fanned to both roles, asserted post-launch, reusable by
async_cifar10 (same `runner.py`/harness).*
- **3a** Make the launcher patch write the aggregator-side knobs too: `--delays`/folds →
  `exp["aggregator"]["config_overrides"]["hyperparameters"]` (`sim_model_eval_time`, `sim_straggler_spread_s`,
  and `trainingDelayEnabled` if kept), symmetric to how `max_runtime_s` is already written there
  (`run_sequential.sh:260`). Smallest correct fix.
- **3b** Better/extensible: fan the resolved flag into BOTH roles' `config_overrides.hyperparameters` as the last
  merge layer in `runner.py` (mirror the `agg_goal` single-source pattern, `runner.py:531-543`), removing the
  bespoke trainer-only bridge as a special case. **DECISION: 3a (local, fast) vs 3b (framework, extensible) —
  choose per appetite; 3b is the principled one and helps async_cifar10.**
- **3c** Expose `training_delay_factor` (hardcoded `"10"` in `trainer_base.yaml:96`) as a launcher knob so
  "delays on" can mean the full registry delay, not a silent ÷10.
- **3d** Bank the effective values: include `enable_training_delays`/folds in `create_execution_config`'s output;
  add a **post-launch assertion** (in `expt_runner.render_and_gate` or a new `assert_banked_config`) that the
  banked `execution_config.yaml` + `aggregator_config.json` equal the requested flags — BLOCK the run otherwise.
- **3e** Retire or wire the orphaned aggregator `training_delay_enabled` (the fwdllm aggregator reads it nowhere)
  so the log stops misleading.
- **Pytest:** a launcher/render test that patches a config with `--delays on`, renders the yaml, loads it through
  `runner.py`, and asserts both roles' runtime `Hyperparameters` reflect the flag + folds; a negative test that
  the post-launch assertion BLOCKS on a mismatch. Extend the same test to an async_cifar10 config.

#### Phase 4 — Parity stopping rule + #6 sct model (depends on Phase 2)
*Only meaningful once the sim is fast (`sim_rate>1`). Then:*
- **4a** Decouple `sim_wall_ceiling_s` from the vclock budget (generous multiple / separate `--max-wall-s`) → an
  outer runaway safety, not the primary stop (which is vclock reaching `max_runtime_s`).
- **4b** For the parity comparison, stop both sides at a **matched committed-`data_id`** target (finite
  `--max-data-id`), so C1/C2/V1/V2 compare at identical progress (the overnight 10-`data_id` runs passed for
  exactly this reason). Demoted from "primary fix" to convenience.
- **4c** Calibrate B1/B2 folds (`eval_s` +3.34s, straggler-spread +2.3s) so `wall_disparity/data_id → ~0` (the
  genuine #6 residual once the artifact waits from Phase 2 are gone). B3 WAN knob stays doc-only.
- **Pytest:** ceiling-decouple + matched-data_id stop unit tests; `wall_disparity` rung on synthetic. **Exit:**
  gated by the sign-off run (below), not pytest alone.

#### THEN (not before): the Phase-1 sign-off RUN
Only after Phases 0–4 are pytest-green + full `tests/` + shared-parity suite green: smoke → convergence per
baseline at a matched `--max-data-id`, D>0 (full factor), **with `sim_rate>1` confirmed live**. It banks: the
12 un-skipped rungs, `sim_rate`/`wall_speedup`, `wall_disparity→~0`, fwdllm_plus completing the matched data_id,
and the fluxtune R1 emergent smoke. Record in the scoreboard.

### Pre-next-run implementation plan (staged — the "definition of done" before spending another run) — ⚠️ SUPERSEDED by the plan above (its sign-off run was unusable; see 2026-07-04 status)

*Goal: land ALL dev that lets the NEXT parity run extract maximum information — un-skip the 12 rigor-gap rungs,
make #6/#7 first-class in telemetry, and close the sct model — so the run measures a nearly-complete ladder,
not a half-instrumented one. Rung math: 74 total; today **42 active / 32 skip**; after Stage A → **~54 active /
20 skip** (the remaining 20 skips are the legit Phase-2/config ones). Each task ships telemetry + pytest
(principle #11). Anchors in **K-D20**.*

**PROGRESS (update markers in place as stages land):**
`✅ piece A` vclock-budget stop (landed, 17 tests) · `✅ K-D20` decomposition + anchors ·
`✅ A1` trainer phase timing · `✅ A2` agg wall-decomp · `✅ A3` re-key advance rungs · `✅ A4` wall_disparity rung ·
`✅ B1/B2` sct model (eval_s + straggler spread, config-gated) · `✅ B3` WAN knob (doc-only) · `✅ C` fwdllm_plus liveness ·
`✅ D1/D2` task_recv.sim_send_ts + K5 real-compute exemption · `✅ E` sleep(0.1)-pad gating (#11).
**ALL landed & pytest-green (K-D21). NEXT: the Phase-1 sign-off RUN** — smoke → convergence per baseline;
it (a) exercises the 12 un-skipped rungs, (b) drives `wall_disparity`→~0 and calibrates B1/B2's `simModelEvalTime`/
`simStragglerSpreadS`, (c) confirms fwdllm_plus completes 10 `data_id`s, (d) banks the fluxtune R1 emergent smoke.

- **Stage A — instrumentation (un-skip 12 rungs; make #6/#7 measurable).** Depends on nothing.
  - **A1** Trainer per-phase timing → un-skips the 8 phase rungs. Populate `_phase_times` in fwdllm's
    `_fetch_weights` override (`fwdllm_trainer.py:168`) + emit `mqtt_fetch_s`/`weights_to_ram_s`/
    `weights_to_gpu_s`/`pre_train_s`/`gpu_compute_s`/`post_train_s`/`training_budget_s`/`trainer_phase` in
    `FedSgdTrainer.train_with_data_id` extra (`:598-602`). Mirror `async_cifar10 main.py:784-908`. + manifest + pytest.
  - **A2** Aggregator per-round wall decomposition → `barrier_wait_s` (dispatch→last grad), `drain_tail_s`
    (last grad→commit, the artifact), `aggregate_fedavg_s`, `eval_s` (from the `eval_model` decorator) in
    `agg_round`. Makes phases (b)–(e) telemetry, not log-grep. + pytest.
  - **A3** Re-key the 4 advance rungs (K3a/K3b/K3/K4) to the `data_id` axis (extend `_progress_axis`) so they
    stop reporting "&lt;2 sim rounds." **Shared parity engine → FULL `tests/` + `examples/async_cifar10/scripts/
    parity` gate** (principle #9); assert async_cifar10 byte-identical (auto-detects `round`).
  - **A4** New DIAG rung `wall_disparity` = `|real_wall − sim_vclock|` per matched `data_id` — the recurring
    sanity metric to drive → ~0 (surfaced every run). + pytest.
- **Stage B — #6 `sct` model (piece B), reads A2's telemetry.** Fold the two GENUINE unmodeled terms into the
  vclock; leave artifacts off (principle #1).
  - **B1** `eval_s` per `data_id` onto the vclock (+3.34 s/round).
  - **B2** Widen the sim barrier straggler completion-spread to match real `trainer_speed_s` dispersion
    (2.66–4.05 s), so the k-th-smallest `sct` reflects real (+~2.3 s/round). + pytest.
  - **B3** WAN payload-transfer term (1.96 MB up / 3.45 MB down) — NOT measurable on localhost; add a documented
    config knob, **do not enable this run** (no ground truth). **Exit: `wall_disparity/data_id` → residual ≈ the
    4.7 s/round artifact, and throughput/K2/K3b within tolerance.**
- **Stage C — #7 fwdllm_plus real liveness** (so its 10-`data_id` run actually completes). Real-mode sync barrier
  under scarcity: relax the agg-goal to available-count OR honor `wait_until_next_avl` (sleep-to-next-avail)
  instead of spin-to-cap; bound the log. + pytest. Independent of A/B.
- **Stage D — cheap checker/telemetry gaps.** **D1** #8 emit `task_recv.sim_send_ts` (or alias). **D2** #9
  `failsafe`/K5 compares sim wall vs the RUN wall budget (not vclock) for a real-compute sim.
- **Stage E — optimization pass (#11; correctness-preserving, opportunistic).** Audit the `time.sleep(0.1)` pads
  (`fwdllm_aggregator.py:2209/2338`, ≥28.7 s/run) — verify each one's purpose, gate/remove off the critical
  path; profile GPU-shared trainer compute (A1's `gpu_compute_s` per trainer) for waste. **Re-run pytest to
  prove grad values + cadence unchanged** (never trade correctness for wall).

**Definition of done:** A + B + C + D pytest-green; **full `tests/` + shared parity suite green** (A3 spans the
stack); E opportunistic. THEN one run per baseline banks: 12 un-skipped rungs live, `wall_disparity`→~0,
fwdllm_plus completing 10 `data_id`s, and the fluxtune R1 emergent smoke.

### Open roots (fix lowest-rung-first)

**TOP: #1c (R1 residence regression) is now FIXED in code (K-D19) + pytest-green; the only thing owed on it is
ONE fluxtune sim smoke to bank the emergent R1/in_flight numbers — the LAST step, not the debugger (principle
#11c).** Roots #1 (D-e), #1b (K-D17 deadlock), #3 (field_coverage alias) FIXED; **#2 (clock-family AXIS) fixed
but exposed the RATE gap #6.** **Ranking of what's OPEN:** bank the #1c re-run → then the sync-baseline work
#6 (clock-RATE, shared with fluxtune #1) → #7 (fwdllm_plus real speed, gates #4) → #8/#9 (telemetry/checker
gaps). Attack #6 and #1's wall gap together — they are the same time-base question on the async vs sync path.

1c. **[FIXED in code — K-D19; emergent re-run pending] R1 in-flight overlap 0% → 44.7% (fluxtune).** Root was
   NOT the doc's hypothesis (async_oort *does* exclude `all_selected`) but the guard being released on physical
   grad **RETURN** (`_process_single_trainer_message` → `cleanup_provided_ends`) while the carried grad hadn't
   committed in virtual time. Fixed fwdllm-only via `_release_end_on_return` (defers the async+sim+residence
   release to COMMIT, felix-aligned); + a `[SIM_R1_DISPATCH]` tripwire + `TestReturnPathGuardHeldToCommit`
   (the RETURN-path blind spot). `tests/mode -k fwdllm` 110 green. Full rationale **K-D19**. **Remaining:** one
   fluxtune sim smoke to confirm R1≤2% + `in_flight`~9.5 (emergent — measured, not assertable from banked
   telemetry; the `020254` raw dir was cleaned).

1. **[FIXED — Option A / K-D16] D-e async concurrency starvation (fluxtune).** *Symptom (pre-fix):* sim kept
   ~1.5 trainers computing vs real's ~5.8, did fewer forward passes (89 vs 109) yet took **2.4× the wall**
   (304 vs 125s) — re-dispatching ~1 trainer/agg-goal-cycle. *Root:* K-D12 parked a returned-but-uncommitted
   trainer in async_oort's `selected_ends`, which drives `extra = c − len(selected_ends)`, so top-up refilled
   ~1/cycle. `selected_ends`/`all_selected` conflated compute-slot occupancy with the re-pick guard. *Fix:*
   Option A two-lifetime split — free `selected_ends` on RETURN (slot reopens → a DIFFERENT trainer refills
   C, mirroring real's channel freeing on receipt), hold `all_selected` (guard) to COMMIT so
   one-in-flight-per-trainer survives the carry + rollback. Details K-D16 / §F-D6. **NOTE:** the K-D16 re-run
   surfaced root #1b (K-D17) — the concurrency-recovery numbers are only measurable AFTER the K-D17 fixes.

1b. **[FIXED — K-D17] Fluxtune deadlock in the K-D16 re-run (drain-gate + triplet-at-dispatch).** The K-D16
   fluxtune smoke deadlocked: 2/10 grads committed, 0 `agg_round`, in-flight 0.28, 1663 empty select-rounds.
   Two bugs: **(A)** `_aggregate_grads_async` gated the sim drain on `channel.ends(VAL_CH_STATE_RECV)` (empty
   after the first greedy buffer-fill) → `"no ends yet"` ×1681 → the ~8 buffered grads never popped → agg_goal
   never met; **(B)** K-D16 stamped the async_oort triplet at DISPATCH → whole pool matched `agg_version_state`
   → `filtered_ends=0` → no re-dispatch, and the version never advanced to unfreeze it (bootstrap case D6
   missed). *Fix:* drain keys on `_sim_buffer`/`_sim_inflight_expected` (not RECV); triplet stamped on grad
   RETURN (not dispatch). Full `tests/mode` green (314/7). Details **K-D17**. **Remaining: the fresh fluxtune
   re-run** confirms recovery (concurrency, wall, U3, V1, selection) — emergent, measured not asserted.

2. **[AXIS FIXED — #2; but exposed the RATE gap #6] Clock family re-keyed to the `data_id` progress axis.**
   `throughput_parity` / `total_commits_parity` / `terminal_state_parity` now count units on the axis the run
   advances (a unified `_progress_axis` / `_per_progress_last_event` helper): `round` for normal FL, committed
   `data_id` for fwdllm (was collapsing to `sim_rounds=1`). async_cifar10 auto-detects `round` → byte-identical.
   The re-key was correct, but with the axis fixed the 10-`data_id` run shows these rungs **still FAIL on a
   genuine per-round TIME-BASE gap** (sim vclock-s/round ≪ real wall-s/round) — that residual is **new root #6**,
   not this checker fix.

3. **[FIXED — #3] field_coverage accepts fwdllm field aliases.** The coverage spec rows for
   `gpu_compute_s` / `training_budget_s` now accept the fwdllm spellings `real_gpu_time_s` /
   `sim_round_duration_s` (a spec field may be a tuple of accepted names). No more false-FAIL / benign SKIP.

4. **[CLEARED for fwdllm; now REAL-side for fwdllm_plus] V1/V2/V5/g2 short-run truncation.** Per-`data_id`
   series match on every FULL `data_id` and diverge only on the last, truncated one. **fwdllm: CLEARED** — at
   10 `data_id`s V1/V2/g2 all PASS (K-D18). **fwdllm_plus: now the truncation is on the REAL side** — sim did
   10 `data_id`s but real reached only ≈3 (1 h cap, root #7), so KS blows up comparing 10 vs 3. Fix is to make
   real reach 10 (i.e. resolve #7), NOT code and NOT a knob. The `--max-data-id` default is now 9999 (K-D18) so
   `--max-runtime-s` governs — but for fwdllm_plus even 1 h wasn't enough, so #7 must be understood first.

5. **[MOSTLY FIXED — K-D17b] fluxtune selection_detail (in_flight).** Was NOT an oort-fidelity gap: the
   `in_flight` metric = `len(selected_ends)`, and K-D16 freed `selected_ends` on physical RETURN, so it
   measured physically-COMPUTING (~2.7) not virtual-time in-flight (~7.8, measured — close to real 9.75).
   K-D17b holds the slot to COMMIT (felix-aligned) so `len(selected_ends)` = virtual in-flight. Residual
   ~7.8 vs 9.75 (~20%) + preferred_duration to re-validate on the fresh re-run.

6. **[OPEN — sync clock-RATE gap; new] throughput/commits/terminal: sim charges far less time/round than real.**
   At 10 `data_id`s (both sides complete), fwdllm sim advances **~7.8 vclock-s/round** vs real **~34.9
   wall-s/round** (rel 0.78); fwdllm_plus **~8.1** vs **~86** (but real truncated, #7 confounds). Do NOT read
   this as a cadence bug (V1/V2/g2 PASS — the WORK per `data_id` matches). It is a **time-base** question:
   real's per-round WALL bundles MQTT/dispatch/aggregation/eval overhead that the sim (correctly, principle #1)
   does NOT put on the vclock (`vclock = max(vclock, sct)`), and real emits **no `vclock_now`** so the checker
   uses real WALL as real's axis. **First move (telemetry, not a run):** decompose real's per-round wall into
   D + per-phase transport overhead using the existing `trainer_phase` / `mqtt_fetch_s` / per-phase splits, and
   compare against the sim `sct` model. Two outcomes: (i) the surplus is real-transport overhead → the rung is
   comparing WALL-vs-VCLOCK and should anchor real on an intrinsic per-round span (like async_cifar10's
   `WALL_SEND−WALL_RECV` duration lesson), a checker fix; (ii) part is modeled D the sim under-charges → a sim
   `sct` fix. **Shared-shaped with the fluxtune 2.4× wall gap (#1)** — resolve the classification once, apply to
   both. Dead-end guard: do NOT add a scalar overhead to the vclock (principle #1, async_cifar10 dead-ends).

7. **[FIXED in code — Stage C / K-D21; emergent re-run pending] fwdllm_plus real spun to the cap under
   `mobiperf_2st` scarcity (agg_goal=10 > avail).** ROOT was the real sync barrier hot-re-dispatching when the
   eligible pool < `agg_goal` (292 MB log, reached `data_id`≈3). Fixed: `_await_dispatchable_under_scarcity`
   sleep-to-next-avail (WAIT/accumulate — cohort stays == agg_goal, parity-faithful vs the sim vclock-jump;
   operator-chosen over relax-to-available). Self-terminates at `max_runtime_s`; one warn per stall. **Remaining:
   the sign-off run confirms fwdllm_plus now completes 10 `data_id`s** (emergent). The separate question of WHY
   fwdllm_plus real is slower/round (per-iteration reselection + oracular read) is a real-config profiling item,
   localizable from the banked per-phase log — now that the run can complete, it is judgeable.

8. **[FIXED — Stage D1 / K-D21] field_coverage INV `task_recv.sim_send_ts` absent.** `fwdllm_trainer._fetch_
   weights` re-emits `task_recv` carrying `sim_send_ts` (the override had dropped the base trainer's emission);
   null in real, non-null in sim. Un-SKIPs K6. `test_fwdllm_trainer_task_recv.py`.

9. **[FIXED — Stage D2 / K-D21] `failsafe`/K5 false-positive for fwdllm.** K5 flags sim WALL overshooting
   the virtual budget by >20% (184s wall vs 78s vclock). For fwdllm this is EXPECTED and benign: the sim does
   **real forward-grad GPU compute** (many passes), so sim wall ≫ vclock by construction — unlike async_cifar10
   where compute is cheap. FIXED: `failsafe_ok` now compares sim wall against the RUN wall budget (`max_runtime_s`)
   for a real-compute sim (auto-detected via the progress axis) and SKIPs when no run budget is available, instead
   of the wall-vs-vclock fallback. `TestFailsafeRealComputeSim`.

### Fixes landed (what worked — do not redo)

- **Phase 4 stopping rule + #6 folds (K-D24)** — 4a: `sim_wall_ceiling_s` decoupled to `max_runtime_s × 20`
  (class const `SIM_WALL_CEILING_FACTOR`, explicit override honored) so a real-compute sim isn't wall-truncated
  before its data_id/vclock stop (root S1). 4c: B1 `sim_model_eval_time:true` (self-calibrating eval_s fold) +
  B2 `sim_straggler_spread_s:0.9` ON in all 3 sim yamls. `test_fwdllm_early_stop_conditions.py` 19 green;
  `tests/mode -k fwdllm` 148. Do NOT re-derive the ceiling from 1× budget — it truncates real-compute sims.
- **Phase 2 speedup-leak skips (K-D23)** — gated the trainer per-round `pause_execution` `time.sleep(1)`
  throttle and the `_check_availability` avail-spin `and not self.simulated` (fwdllm-only; real byte-identical).
  The trainer `recv`/`await_join` and the aggregator grace/`await_join` are LEFT AS-IS (recv delivers the real
  weights; agg already uses the 2s sim grace + real-only `sleep(0.1)` pads). `test_fwdllm_sim_speedup_waits.py`
  (6), `tests/mode -k fwdllm` 146 green. Do NOT try to short-circuit the trainer recv — it's barrier-wait over
  irreducible agg eval+GPU, not a skippable sleep (K-D23). Real-run `sim_rate>1` still needs #6/Phase-4c.
- **Pre-next-run plan A+B+C+D+E (K-D21)** — Stage C scarcity-wait liveness; A1 trainer phase timing; A2 agg
  wall-decomp; A3 advance-rung re-key (shared engine, async byte-identical); A4 `wall_disparity` DIAG rung; B1/B2/
  B3 sct-model folds (config-gated OFF ⇒ byte-identical); D1 `task_recv.sim_send_ts`; D2 K5 real-compute exemption;
  E `sleep(0.1)`-pad gating. Full `tests/` 854 passed / 7 skipped. Emergent calibration (wall_disparity→~0, B1/B2
  values, fwdllm_plus 10-`data_id` completion, fluxtune R1) is the sign-off run's job.

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
- **D-e: Option A two-lifetime split (K-D16)** — *SUPERSEDED by K-D17b.* Freed the compute slot
  (`selected_ends`) on physical RETURN while holding the `all_selected` guard to COMMIT. The slot-on-return
  half was wrong for virtual time (undercounted `in_flight` 3×); reverted in K-D17b. The `all_selected`
  guard-to-commit half survives.
- **K-D17: drain-gate + triplet-at-return (unblocks the K-D16 deadlock)** — (A) `_aggregate_grads_async`
  now drains while `_sim_buffer`/`_sim_inflight_expected` is non-empty even when the channel has no RECV end
  (the sct buffer, not RECV state, is the sim source of truth); (B) the async_oort re-pick triplet is stamped
  on grad RETURN (in `_process_single_trainer_message`), not at dispatch, so the eligible pool isn't frozen
  before the first commit. `test_fwdllm_sim_drain_and_repick.py`.
- **K-D17b: hold the compute slot to COMMIT (felix-aligned; fixes the in_flight mis-measurement)** —
  `_sim_hold_busy_slots` now holds EVERY dispatched-but-not-committed trainer (computing ∪ carried) in BOTH
  `selected_ends` and `all_selected` until commit, called per-commit in `_sim_recv_min_grad`. `in_flight`
  (= `len(selected_ends)`) and `extra` now reflect virtual-time in-flight, matching felix
  (`asyncfl::_sim_hold_busy_slots`/`_sim_recv_min`). Reverts K-D16's slot-on-return; safe now that K-D17
  fixed the drain. Full `tests/mode` green (315/7); `TestVirtualInflightSlotHold`. Concurrency/downstream
  recovery measured by the fresh fluxtune re-run.
- **K-D19: R1 regression fixed — guard release deferred from RETURN to COMMIT (`_release_end_on_return`).**
  Root was NOT the recorded `all_selected` hypothesis (async_oort *does* exclude it) but
  `_process_single_trainer_message` freeing the re-pick guard on physical grad RETURN while the carried grad
  was still uncommitted in virtual time. Fixed fwdllm-only (no shared `async_oort`/`asyncfl` edit — principle
  #8); the async+sim+residence release now happens at COMMIT via `_sim_hold_busy_slots`, felix-aligned. Shipped
  WITH a `[SIM_R1_DISPATCH]` dispatch-time tripwire + `TestReturnPathGuardHeldToCommit` (closes the RETURN-path
  test blind spot). `tests/mode -k fwdllm` 110 green. Emergent R1/in_flight numbers pending one fluxtune smoke.
- **Piece A: vclock-budget stop (matched-budget convergence axis; step 1 toward #6).** `_check_early_stop_
  conditions` now interprets `max_runtime_s` as **wall in real / `vclock.now` in sim** (mirrors base
  `max_experiment_runtime_s`, `top_aggregator.py:1140`), so one budget = "3600 wall-s of real work" AND "3600
  virtual-s of modeled work". A `sim_wall_ceiling_s` failsafe (default = budget) stops a runaway when the vclock
  is under-modeled (root #6) — that ceiling firing IS the #6 signal until piece B lands. `tests/mode/
  test_fwdllm_early_stop_conditions.py` 17 green (4 new sim/vclock cases). **Does NOT alone fix #6** — the
  vclock must first model real wall (step 2) or a vclock-bounded sim over-runs to the wall ceiling.
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

- **"K-D16 fixed fluxtune; the re-run just banks numbers" (believed at K-D16 landing).** WRONG — the K-D16
  re-run DEADLOCKED (2/10 grads, 0 `agg_round`). Two bugs (K-D17): the sim drain was gated on channel RECV
  state (not the buffer), and K-D16 stamped the re-pick triplet at DISPATCH (freezing the pool pre-commit).
  *Lesson:* stamping a "already contributed this triplet" guard at DISPATCH conflates in-flight with
  contributed and, since the agg version only advances on commit, freezes the pool at bootstrap. The guard
  belongs at RETURN. And the sim commit path's readiness must key on its OWN reorder buffer, never on the
  real transport's RECV bookkeeping (principle #8).
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

- **K-D17  Two bugs that deadlocked the K-D16 fluxtune re-run (drain-gate + triplet-at-dispatch).** The
  post-K-D16 fluxtune syn_0 smoke (`run_20260703_230407`) did NOT recover — it DEADLOCKED: 1 cohort of 10
  dispatched, **only 2 grads ever committed** (agg_goal=3 never met once), **0 `agg_round` events** (so
  `vclock_telemetry`/`throughput`/`total_commits`/`terminal_state` all can't run), in-flight collapsed to
  **0.28** (worse than pre-fix ~1.5), and the agg spun 1663 empty select-rounds (`feasible_extra: 0` ×1617)
  until `max_runtime_s`. Two independent bugs, one masked the other:
  - **Bug A — sim drain gated on channel RECV state, not on `_sim_buffer`.** `_aggregate_grads_async`
    early-returned (`"no ends yet"`, **1681/1729** loop passes) whenever `channel.ends(VAL_CH_STATE_RECV) is
    None`. But `_sim_recv_min_grad` greedily drains ALL ready channel messages into `_sim_buffer` on its
    FIRST call (`recv_fifo(..., first_k=len(to_probe))`), which empties RECV — so the 8-9 already-received
    grads sat in the buffer and were never popped (`buf_depth=9` then stuck). The commit path's readiness
    must key on its OWN reorder buffer / in-flight set, not the real transport's RECV bookkeeping
    (principle #8: RECV gating is a real-transport artifact, no sim analog). *Fix:* fall through to
    `_sim_recv_min_grad` when `len(_sim_buffer) > 0 or _sim_inflight_expected` even with RECV empty; real
    path unchanged (guarded on `self.simulated`).
  - **Bug B — the K-D16 triplet was stamped at DISPATCH.** Stamping the whole cohort at the current
    `_curr_agg_version` made every dispatched trainer match `agg_version_state`, so async_oort's filter
    (`async_oort.py:1660`, skip if equal) excluded the ENTIRE pool → `filtered_ends=0` → no re-dispatch. The
    version only advances at a commit boundary, which (via Bug A) is never reached → permanent freeze. **This
    is the bootstrap case the D6 analysis missed:** D6 reasoned the triplet is safe because "it advances
    every agg-goal boundary" — true only once commits flow; BEFORE the first commit (or whenever the
    in-flight set alone can't reach agg_goal), it freezes the pool. Conceptually the triplet means "already
    CONTRIBUTED this triplet," a property of a RETURNED trainer; an in-flight-but-not-returned trainer is
    already guarded by its compute slot (`selected_ends`). *Fix:* stamp `_trainer_state_dict[end] =
    _curr_agg_version` on grad RETURN (in `_process_single_trainer_message`, residence-gated), NOT at
    dispatch. Preserves every K-D16 benefit (slot frees on return, guard holds return→commit, R1 intact) and
    unfreezes the top-up.
  Either fix alone breaks the deadlock (A lets the initial cohort drain to a commit which advances the
  version and unfreezes the triplet; B restores the stream of RECV transitions that masked A pre-K-D16) —
  both landed for correctness. **Where:** `fwdllm_aggregator.py::_aggregate_grads_async` (drain gate) +
  `_process_single_trainer_message` (triplet on return) + `_distribute_weights_async` (dispatch stamp
  removed); `tests/mode/test_fwdllm_sim_drain_and_repick.py` (`TestDrainGateNotBlockedByEmptyRecv` +
  `TestRepickTripletStampedOnReturn`). **Scope:** fwdllm class only; async_oort/oort/asyncfl + sync path
  untouched (empty triplet map ⇒ inert). Full `tests/mode` green (314 passed / 7 skipped). Concurrency/wall
  recovery is emergent — measured by the fluxtune re-run, not asserted.

- **K-D17b  The K-D17 fluxtune concurrency "gap" was mostly a MEASUREMENT artifact — revert the K-D16
  Option-A slot split, realign with felix (hold the slot to COMMIT).** The post-K-D17 fluxtune run was
  healthy (no deadlock) but `selection_detail` still FAILed: reported `in_flight` sim **3.38** vs real 9.75.
  A diagnostic (`inflight_exp` = dispatched-not-committed vs `sel_ends` = `len(selected_ends)`) settled it:
  the sim keeps **~7.8** trainers in flight in VIRTUAL time (close to real's 9.75), but the `in_flight`
  telemetry reads `len(selected_ends)` ≈ **2.73** — trainers physically COMPUTING right now. *Root:* the
  `in_flight` metric is `len(selected_ends)` (`selector/__init__.py:161`), and K-D16 Option A freed
  `selected_ends` on **physical RETURN** — a wall event with no virtual-time meaning. In virtual time a
  returned-but-uncommitted trainer is still in flight (its grad commits only when the vclock reaches its
  sct), so its slot is genuinely occupied. Freeing it undercounts `in_flight` 3× AND makes
  `extra = c − len(selected_ends)` read false free capacity (the selector churns `desired_extra` 7–10,
  blocked by the guard). *Confirmed against felix:* `asyncfl/top_aggregator.py::_sim_hold_busy_slots`
  (:1453) holds the FULL dispatched-but-not-committed set in `selected_ends` and `_sim_recv_min` (:606-627)
  releases only on COMMIT; its docstring names freeing-the-slot-on-return as the "over-selection bug." So
  fwdllm's Option-A split was a deviation from the proven reference; freeing the slot manifested as an
  UNDER-count only because fwdllm's extra `all_selected` guard blocked the over-selection felix warns of.
  *Fix (K-D17b):* rewrite `_sim_hold_busy_slots` to hold EVERY outstanding trainer (`_sim_inflight_expected`
  ∪ buffered surplus, minus committed) in BOTH `selected_ends` and `all_selected` until commit, and call it
  per-commit inside `_sim_recv_min_grad` (mirroring felix's per-commit reset, since recv_fifo marks
  freshly-buffered ends RECVD and the channel strips their slots). Now `len(selected_ends)` = virtual-time
  in-flight, so both `in_flight` and `extra` are correct. Safe to revert Option A *now* only because K-D17
  fixed the drain (the pre-K-D16 "concurrency 1.5" was the drain stall, not holding-to-commit). **Where:**
  `_sim_hold_busy_slots` + `_sim_recv_min_grad` (per-commit hold) + `_release_sim_slots_at_agg_goal`
  (docstring); `tests/mode/test_fwdllm_sim_grad_residence.py::TestVirtualInflightSlotHold` (+ updated
  `TestCommitThenCarryResidenceOn`). Full `tests/mode` green (315/7).

  **VALIDATION RESULT (run `020254`, 2026-07-04) — PARTIAL WIN + A NEW R1 REGRESSION (do not ship as-is):**
  - ✅ **in_flight measurement FIXED.** `sel_ends` 2.73 → **7.78**, matching `inflight_exp` (8.10). In the
    parity battery `selection_detail.in_flight` = sim **9.49** vs real 9.75 (**rel_diff 0.027**, was 0.65).
    The core goal is met — the telemetry now reports virtual-time in-flight.
  - ✅ **V2 var-trajectory now PASSES** (sim 0.894 vs real 1.008, KS 0.177<0.20); U3/S2/conv/W1 pass.
  - ❌ **R1 residence REGRESSED: 0.0% → 44.7% overlap** (real 1.8%, tol 2%). K-D17b re-dispatches a trainer
    while its prior grad is still in flight. **Root lead:** `_sim_hold_busy_slots` resets buffered-but-
    uncommitted ends to `KEY_END_STATE=NONE` (to stop the RECVD-strip of their slot, mirroring felix), but
    **async_oort's selection filter keys on avail-state/end-state, NOT on `all_selected`** (`async_oort.py`
    ~1643-1695 filters by `count_avl_train`+triplet; `all_selected` at :405 is only *logged*). So a NONE-state
    buffered trainer looks idle/available -> re-selected -> overlapping in-flight interval -> residence
    violation. fwdllm's async_oort does not exclude `all_selected` the way felix's path effectively does.
  - ❌ still failing: `selection_detail.chosen` (sim 0.83 vs real 1.76), throughput/total_commits/terminal
    (sim commits faster — 3 vs 1 at matched virtual budget 27.2s; partly short-run #4), field_coverage,
    avail_composition, V1/g2 (#4 truncation).

  **STOPPED before the overnight run** — R1 is a correctness regression; banking 6h of runs on it would be
  wrong. Resume steps are in **§H -> "NEXT SESSION (2026-07-05) pickup"**.

- **K-D18  Overnight 10-`data_id` grounding runs + the `--max-data-id` default raised to 9999.** The operator
  launched `run_sequential.sh --mode both --max-runtime-s 3600 --only fwdllm,fwdllm_plus` (then fluxtune
  real-only), intending 1 h runs, but `run_sequential.sh`'s `--max-data-id` **defaulted to 10** and its
  `patch()` always overrides the yaml, so every run stopped at `data_id=10` instead of at 1 h. This was a
  useful accident — 10 `data_id`s is >3× the prior banked runs and cleared the #4 truncation for fwdllm — but
  the trap is real, so the default is now **9999** (effectively unbounded; `--max-runtime-s` governs) in both
  `run_sequential.sh` and the six `expt_scripts/*_n10_smoke*.yaml`; pass `--max-data-id 10` (or 3) for a
  deliberately short run. **Parity read** (`run_parity.py --baselines fwdllm fwdllm_plus`, JSONs
  `parity_fwdllm_20260704_022610` / `parity_fwdllm_plus_20260704_033206`): *fwdllm* 37 pass / 5 fail — cadence
  (V1/V2/g2/R1/W1/U3/S2/conv) CLEAN, surviving fails = the clock-RATE family (open root #6), field_coverage
  (#8), failsafe (#9); *fwdllm_plus* 31 pass / 10 fail — but its **real run hit the 1 h cap at `data_id`≈3**
  (86 wall-s/round; sim did 10 in 81 vclock-s), so the cadence/eligibility/selection fails are REAL-side
  truncation (#4) driven by real-speed root #7, NOT a confirmed sim bug. *fluxtune* ran real-only (no sim), so
  no parity — its sim is unchanged from `020254` (R1=44.7%, #1c). **Discipline applied (principle #11):** all of
  the above was read from the ALREADY-banked telemetry + logs (stop-reason lines, per-round `runtime.py`
  timings, the parity JSONs); no new run was launched to reach these conclusions.

- **K-D19  The R1 residence regression (#1c) — root-caused to guard-release-on-RETURN, fixed fwdllm-only +
  pytest + tripwire. (Corrects the K-D17b/§H "NEXT SESSION" hypothesis.)** The hypothesis on record was
  "`_sim_hold_busy_slots` resets buffered ends to `KEY_END_STATE=NONE` and async_oort doesn't exclude
  `all_selected`, so they're re-selectable." **That was wrong on the mechanism:** async_oort *does* exclude
  `all_selected` (`async_oort.py:1586-1587`) and the channel passes the full pool regardless of `KEY_END_STATE`
  (`channel.py:221-240`) — so the NONE reset is a red herring for re-selection; exclusion depends solely on
  `all_selected` membership. **Real root:** on the async accept path, `_process_single_trainer_message` called
  `channel.cleanup_provided_ends(end)` on **physical grad RETURN, per-message, mid-cycle**
  (`fwdllm_aggregator.py:1259-60` pre-fix) → `async_oort._cleanup_provided_ends` (:889-916) deletes the trainer
  from `all_selected`. But in sim residence the grad is **carried** (its `sct` is in the future), so the trainer
  is torn out of the re-pick guard while still in flight in VIRTUAL time → any `_distribute_weights_async`
  before its commit re-dispatches it → R1 overlap 0%→44.7%. K-D17b's per-**commit** `_sim_hold_busy_slots`
  re-asserts the guard, but too late — it can't cover the RETURN→COMMIT window. felix (`asyncfl:1316`) releases
  the guard only at the agg-goal boundary, never per-return — the divergence. *This is the D6 "carry was right,
  release TIMING was wrong" pattern applied to the guard (`all_selected`), not just the compute slot.* **Fix
  (fwdllm-only, principle #8 clean — no shared `async_oort`/`asyncfl` edit):** extracted the release into
  `_release_end_on_return`, which in async+sim+residence **defers the release to COMMIT** (owned by
  `_sim_hold_busy_slots`); real mode / non-residence / sync unchanged (return≈commit → byte-identical).
  **Over-instrument (principle #11):** a dispatch-time `[SIM_R1_DISPATCH]` tripwire warns the instant a still-
  outstanding trainer is re-dispatched (violation visible live, not reconstructed offline). **Pytest:**
  `TestReturnPathGuardHeldToCommit` in `test_fwdllm_sim_grad_residence.py` drives the RETURN path directly —
  the blind spot that let R1=44.7% ship with `tests/mode` green (the prior residence tests only poked
  `_sim_hold_busy_slots`/`_release_sim_slots_at_agg_goal`, never `_release_end_on_return`). `tests/mode -k
  fwdllm` 110 green. **Remaining = emergent-only validation** (principle #11c): one fluxtune sim smoke to
  confirm R1≤2% with `in_flight` still ~9.5 — measured, not assertable from banked telemetry (the `020254` raw
  dir was cleaned).

- **K-D20  Real per-round wall decomposition — grounds #6 (`sct` model) and root-causes #7. (telemetry-only,
  no run.)** For fwdllm sync the real aggregator spends **13.4 s/agg-round**, the sim vclock **3.0 s** — but the
  gap is NOT all a modeling deficit. Barrier-anchored decomposition (dispatch → 1st grad → 10th grad → commit →
  eval, from `runtime.py` decorator lines + `RECV_FIFO` markers + `agg_round`/`agg_eval` telemetry): **(a)**
  trainer compute + round-trip 3.96 s (gpu 1.05 s; `sim_round_duration_s` 2.19 s already modeled), **(b)**
  straggler/sync-barrier wait 3.06 s (GENUINE), **(c)** aggregator drain tail 2.48 s (**ARTIFACT** — real clamps
  `num_min_req=min(agg_goal,1)` and acks one grad per poll-tick; the sim's all-k barrier is the correct model),
  **(d)** `eval_model` 8.68 s/`data_id` = 3.34 s/round amortized (GENUINE server eval, the largest unmodeled
  term), **(e)** FedAvg + redispatch ~0.6 s (mostly the `time.sleep(0.1)` pads, ARTIFACT). **Net: ~8.7 s/round
  genuine vs ~4.7 s/round artifact → the true sct gap is ~2.9×, not 4.5×.** So piece B folds ONLY `eval_s`
  (+3.34) + a widened straggler spread (+~2.3) into the vclock; the drain tail, the ≥28.7 s of `sleep(0.1)`
  pads, and localhost-MQTT/GIL stay OFF the clock (principle #1) — and become optimization targets (#11). **#7
  root cause:** fwdllm_plus real spun to the 1 h cap (log → 292 MB) because oracular `mobiperf_2st` drops avail
  to 1/10 < `agg_goal=10` and the real sync barrier can't assemble the cohort (`wait_until_next_avl=False`); the
  SIM completes the same trace (vclock jumps past the unavailability window), proving it's a **real-mode liveness
  bug**, not a sim/model bug. Full anchors (trainer `_phase_times` already exist at `syncfl/trainer.py:143-150`
  but fwdllm's `_fetch_weights` override drops them; per-field wrap points; the `async_cifar10 main.py:784-908`
  reference) are in the Stage-A/A1 plan above. Feeds the entire "Pre-next-run implementation plan."

- **K-D21  Pre-next-run implementation plan LANDED — Stages A + B + C + D + E (pytest-gated, config-gated;
  emergent calibration deferred to the sign-off run).** All dev that lets the next run measure a nearly-complete
  ladder is in code + pytest-green; nothing here needed a run (principle #11a/b — telemetry-first).
  - **Stage C (fwdllm_plus liveness, decision: WAIT/accumulate — parity-faithful, operator-chosen over relax):**
    `_await_dispatchable_under_scarcity` (real mode) sleep-to-next-avail instead of hot-re-dispatching when the
    eligible pool < `agg_goal` — KEEPS cohort == `agg_goal` (matches the sim's vclock-jump, so cohort size stays
    identical real↔sim; relaxing real's cohort would desync cadence). Self-terminates via
    `_check_early_stop_conditions` at `max_runtime_s`; one warn per stall (bounds the 292 MB log). Byte-identical
    when availability tracking is off / on the sim path. `test_fwdllm_sync_scarcity_wait.py` (8). *Note:* §C's
    fwdllm_plus `agg_goal` row read 2; the landed yaml is **10** (all clients) — the yaml is source of truth.
  - **Stage A1 (trainer per-phase timing):** `_phase`/`_phase_times` added to `fwdllm_trainer` (+ `mqtt_fetch_s`/
    `weights_to_{ram,gpu}_s` wrapped in `_fetch_weights`); `FedSGDTrainer.train_with_data_id` drains those +
    `pre_train_s`/`gpu_compute_s`/`post_train_s`/`training_budget_s`/`trainer_phase` into `trainer_round`. Un-skips
    the 8 phase rungs. Manifest note updated (the phase fields ARE emitted for the #6 decomposition; the DERIVED
    budget-slack/overrun plots stay not_populated — fwdllm's delay is a flat additive sleep). Reversed the Part-6
    "don't emit training_budget_s" note for the decomposition (not for budget-slack).
  - **Stage A2 (agg per-round wall decomposition):** `barrier_wait_s` (dispatch→last grad), `drain_tail_s` (last
    grad→commit, the K-D20(c) ARTIFACT), `aggregate_fedavg_s`, `eval_s` in `agg_round` — via a dispatch wall stamp
    (`_round_dispatch_wall_ts`) + a last-grad stamp (`_last_grad_wall_ts`) + timers around `aggregate()`/
    `eval_model()`. Null-safe (rung SKIPs, no crash). `TestPerRoundWallDecomposition` (4).
  - **Stage A3 (advance-rung re-key — SHARED engine):** `_per_round_advances` (feeds K3/K4/K3a/K3b) now keys on
    `_progress_axis` via `_per_progress_last_event` — fwdllm advances on `data_id` (was <2 `round`s → SKIP);
    async_cifar10 (round-advancing) falls through to `_per_round_last_event` → byte-identical. Full `tests/` +
    `examples/async_cifar10/scripts/parity` re-run green.
  - **Stage A4 (`wall_disparity` DIAG rung):** new `|real_wall − sim_vclock|` per matched progress unit
    (cumulative-from-first-matched so a non-zero-start run still aligns); registered in `run_all_parity`/
    `CHECK_META`/shim; never gates. The recurring #6 sanity metric → drive to ~0. `TestWallDisparity` (5).
  - **Stage B (sct-model folds — CONFIG-GATED, flag-off ⇒ byte-identical; emergent calibration = the run):**
    **B1** `simModelEvalTime` charges the MEASURED server-eval wall to the vclock after each committed data_id
    (consistent with the existing real-gpu-on-vclock model; the largest genuine unmodeled term). **B2**
    `simStragglerSpreadS` adds a STABLE per-trainer (crc32) offset in [0, spread) to the modeled delay in SIM
    only, widening the barrier's k-th sct to real dispersion. **B3** `simWanTransferS` — documented knob, left 0
    (no localhost ground truth; do NOT enable). All default OFF. `test_fwdllm_sct_model.py` (7) covers mechanism +
    flag-off byte-identical. **Whether the folds drive `wall_disparity`→~0 is the sign-off run's job**, not
    asserted here (principle #11c).
  - **Stage D (checker/telemetry gaps):** **D1** `fwdllm_trainer._fetch_weights` re-emits `task_recv` carrying
    `sim_send_ts` (the override had dropped the base trainer's emission → field_coverage INV + K6 had null/null);
    `test_fwdllm_trainer_task_recv.py` (3). **D2** `failsafe_ok`/K5 now compares sim wall against the RUN wall
    budget for a real-compute sim (auto-detected via the progress axis) and SKIPs when no run budget is available,
    instead of the wall-vs-vclock fallback that false-failed every fwdllm run (sim wall ≫ vclock BY construction —
    real forward-grad compute); `TestFailsafeRealComputeSim` (3).
  - **Stage E (optimization):** the two `_distribute_weights_{sync,async}` `time.sleep(0.1)` "busy wait" pads
    (real-transport artifacts, ≥28.7 s/run) gated `if not self.simulated` — real byte-identical, sim skips pure
    wall overhead. Full pytest suite re-run proves grad values + cadence unchanged.

- **K-D22  Availability params respected end-to-end — Phase-1 defaults to syn_0; the pre-flight table shows the
  RESOLVED config, not a hardcoded default.** Two consistency bugs in `run_sequential.sh`: (1) the "trace" row
  PRINTED `syn_0` whenever no `--avail-trace` was passed, but `patch()` only sets the mode when a trace is
  truthy — so with no flag the yaml's OWN `mode:` ran (fwdllm_plus `mobiperf_2st`, fluxtune `mobiperf_3st_50`)
  while the table claimed syn_0 (the K-D20 stall, mislabeled). (2) The yamls' source-of-truth `mode:` was a
  mobiperf trace, not syn_0. **Fixes:** the availability trace now DEFAULTS to `syn_0` (Phase-1) so `patch()`
  always sets trainer `availability.mode` + aggregator `trackTrainerAvail.trace` + `client_notify.trace`
  EXPLICITLY on every baseline; the table's trace row + a new per-baseline `avail` column are read BACK from the
  patched cfg (what actually launches), so print == run; the 4 source yamls flipped to `mode: syn_0` (switch to a
  mobiperf trace via `--avail-trace` only for Phase-2); and a new **feasibility gate BLOCKS** a full-participation
  sync barrier under a non-syn_0 trace (sync `agg_goal >= n_trainers` + unavailability = can't assemble → stall;
  `--force` to override). Verified: generated launch cfg carries `syn_0` in all three availability fields; the
  guard blocks `--avail-trace mobiperf_2st --only fwdllm_plus` at pre-flight (exit 2).
- **K-D23  Phase 2 (speedup leak, root #13) — the 2a "trainer inter-round fetch" hypothesis was REFUTED by
  the telemetry; the real skippable wall was two per-round sleeps, gated fwdllm-only.** The pre-plan leak
  inventory ranked the trainer's `await_join`+`recv_wrapper` fetch as the DOMINANT leak (#1) and "ungated
  `sleep(1)` backoffs" LAST (#4). Localizing from the banked `134801` sim (principle #11a) + a full
  async_cifar10 sim-path map INVERTED that ranking. **What the telemetry showed:** the trainer's
  `mqtt_fetch_s` splits by `trainer_phase` into ~1s steady (iter `/1`) and ~15–18s at each data_id boundary
  (iter `/0`); the ~15s OVERLAPS the aggregator's `agg_round` `eval_s` (~13s server eval) + real GPU — i.e.
  it is **barrier-wait for irreducible real work, not an independent trainer sleep.** **2a decision:** do
  NOT short-circuit the trainer `recv` — it delivers the real weights the forward-grad pass needs for grad
  mode-invariance, and async_cifar10 confirms the sim keeps real MQTT recv (the map: "sim trainer physically
  blocks on real MQTT for the fetch — intentional and cheap; do not expect a gate here to port"). The genuine,
  additive, fidelity-free per-round wall was: **(1)** the trainer `pause_execution` throttle (`time.sleep(1)`
  chained at the tail of EVERY loop iteration — `compose(): ... >> task_put_grad >> pause_exec`; comment
  ":494 don't overwhelm mqtt" = a real-transport artifact, principle #8); **(2)** the `_check_availability`
  avail-spin (`while UN_AVL: time.sleep(1)`, which in sim would FREEZE the vclock — sim time can't advance
  while a trainer blocks — and is redundant because sim availability is enforced agg-side, mirroring
  async_cifar10's gated trainer avail wait). Both gated `and not self.simulated`; real byte-identical.
  **Aggregator side needed NO change** — verified already sim-correct: `_sync_sim_recv_first_k` uses the 2s
  grace floor (not the 30/90s real timeout), the `sleep(0.1)` distribute pads are already gated real-only
  (Stage E), `await_join` is kept (cheap, round-1, matches async_cifar10), and the agg's own
  `pause_execution` is defined but NOT in its loop. **Blast radius:** fwdllm-only (`fwdllm_trainer.py`
  `pause_execution` + example `FedSgdTrainer._check_availability`), so per principle #9 `tests/mode -k fwdllm`
  (146 green, +6 in `test_fwdllm_sim_speedup_waits.py`) is the gate, NOT full `tests/` — the original plan
  assumed shared-base edits that turned out unnecessary. **Scope note:** Phase 2 alone does NOT make the
  REAL-run `sim_rate>1` — at syn_0 the sim wall is dominated by server `eval_s` (~13s/data_id), which is
  #6/Phase-4c's job to fold into the vclock; Phase 2's own exit (`sim_rate>1` on SYNTHETIC timings + pytest)
  is met. See §H issue #13 + the Phase-2 PROGRESS marker.
- **K-D24  Phase 4 (parity stopping rule + #6 sct model) — ceiling decoupled, eval/straggler folds enabled
  for the sign-off run; final calibration is the run's job (not pytest).** Three pieces: **(4a, root S1)**
  `sim_wall_ceiling_s` defaulted to `1× max_runtime_s`, which ALWAYS truncated a real-compute sim (wall ≫
  vclock by construction) before its paired vclock/data_id stop — the 2026-07-04 sign-off void. Decoupled to
  `max_runtime_s × SIM_WALL_CEILING_FACTOR` (=20, class const, overridable by explicit `sim_wall_ceiling_s`)
  so it is a runaway OUTER safety, not the primary stop; fwdllm-only (`_check_early_stop_conditions`), 19
  tests. **(4b)** the matched committed-`data_id` stop already existed (`max_data_id_progress`; the sign-off
  run passes `--max-data-id N` as the primary stop so real↔sim compare at identical progress, NOT a
  time budget — a wall/vclock budget desyncs wall-bound real from the sim). **(4c, #6)** the two GENUINE
  unmodeled `sct` terms (Stage B, landed config-gated in K-D21) turned ON in the 3 sim yamls per the
  operator: **B1** `sim_model_eval_time:true` charges the MEASURED server `eval_s` (~13s/data_id, the syn_0
  wall dominator) to the vclock — self-calibrating (uses the real eval wall, no magic constant), delay-
  independent, and the term that actually lifts `sim_rate` toward/over 1; **B2** `sim_straggler_spread_s:0.9`
  (≈ banked real-compute std 0.26s · √12, since the offset is uniform`[0,spread)`) widens the barrier's
  k-th-smallest sct to real's completion dispersion — active only with `--delays on`. The `wall_disparity`
  DIAG rung (A4, keyed on the fwdllm `data_id` axis) surfaces `|real_wall − sim_vclock|/data_id` every run;
  the **B2 value is calibrated to drive that → ~0 FROM the sign-off run** (emergent, per the doc — do NOT
  tune it blind, principle #3/#8). Blast radius: 4a is fwdllm-only → `tests/mode -k fwdllm` 148 green is the
  gate. This closes all PRE-RUN dev; the Phase-1 sign-off run is next.

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
