# FLUXTUNE vs FWDLLM / FWDLLM_PLUS — experiment design (living doc)

**Status:** Tooling **IMPLEMENTED & validated end-to-end at N=10** (2026-07-06). Signed-off
`main` condition (N=100, 82%, syn_0, α=0.1, delay_factor=2, agg_goal=10 matched) is loaded and
gated; **N=100 real convergence runs are the next action**.
**Owner:** dgarg39 · **Branch:** `dg/fwdllm_sim_unavail`

This is the human design doc. Its machine-readable twin is [`experiments.yaml`](experiments.yaml),
which the tooling **consumes** — `run_sequential.sh` launches the run-set from it and
`compare_baselines.py` reads it to know which runs feed which metric. Keep the two in sync;
same discipline as the pre-flight gate — *what is printed == what runs*. When they disagree,
`experiments.yaml` is the source of truth for **what ran**; this doc is the source of truth for
**what we intend and why**.

Related docs: [`simulate_fwdllm.md`](simulate_fwdllm.md) (principles), [`PARITY_LOGICAL_TASKS.md`](PARITY_LOGICAL_TASKS.md)
(real↔sim parity), [`fluxtune_contributions.md`](fluxtune_contributions.md).

---

## 0. Architecture: runs vs. analyses (the anti-redundancy backbone)

The single organizing rule, because it is what makes cross-baseline numbers reusable AND
trustworthy:

- **Run-set** — the expensive GPU work. Produced **once** per `(baseline × condition)`. Each run
  dir self-describes via `snapshot.yaml` + `telemetry/*.jsonl`. Governed by the **convergence stop**
  (§2) with `max_runtime_s` / `max_data_id_progress` as safety caps.
- **Analyses** — cheap, pure functions over `telemetry/`. **All five experiments are views over the
  same run-set.** Expt 1 *defines* the runs; Expts 2–5 add **zero** new GPU runs — they are reducers.
- **Mix-guard** — `compare_baselines.py` discovers the latest run per baseline (exact-token regex,
  so `fwdllm` never captures `fwdllm_plus`) and **warns if the baselines' shared axes
  (N/partition/trace) disagree** — the post-hoc twin of the launch-time `condition_fp`. Together they
  bracket a comparison: the fingerprint proves the runs launched identically, the mix-guard proves the
  dirs being compared still agree.

Consequence: adding "Experiment 6" later = writing a reducer, not launching hardware. Re-deriving a
metric after a telemetry fix = re-run the analysis, not the experiment.

```
experiments.yaml ──> run_sequential.sh ──> experiments/run_*/telemetry/*.jsonl
       │                                              │
       └────────────> compare_baselines.py <──────────┘   (reducers → one table/CSV + overlay plots)
```

---

## 1. Baselines (what distinguishes them)

Substance lives in `_metadata/baselines.yaml`; the run YAMLs only pick `baseline:` + a few overrides.

| Knob | **fwdllm** | **fwdllm_plus** | **fluxtune** |
|---|---|---|---|
| sync / async | sync | sync | **async** (`fedbuff`) |
| selector | `random` | `random` | `async_oort` |
| availability tracking | unaware | **ORACULAR** | client_notify (self-report) |
| reselect | per-round | per-iteration | continuous (async) |
| staleness | `exact` | `round_data_id` | `none` (down-weight) |
| JVP perturbation scoring | off | off | **on** (`jvp_perf_opt`) |
| native `agg_goal` | 10 | 10 | 3 |

**Signed-off experiment override:** the `main` condition matches **`agg_goal=10` across all three**
(fluxtune uses `agg_goal=10` with `C=30`, NOT its native fedbuff 3) — same aggregation batch for a
fair head-to-head. A pre-flight check enforces the match (warns if it ever diverges). See §7.0.

**Design risk (Phase 2, mobiperf):** fwdllm_plus's sync barrier (`agg_goal ≥ available trainers`)
**cannot assemble** under unavailability — the pre-flight gate already **blocks** this. A fair Expt-1
comparison under a `mobiperf_*` trace needs a deliberate answer for what fwdllm_plus does. Phase 1
(the signed-off `main` condition) uses `syn_0` (100% available) where all three complete; the mobiperf
policy is a Phase-2 decision.

---

## 2. Convergence stop (run termination) — runner-side watcher

**A run terminates when the aggregator has completed `W` consecutive data bins all remaining ≥ target
accuracy `τ`.** Defaults `W=20`, `τ` per-condition in `experiments.yaml`.

- **Signal:** `agg_eval` telemetry events. Verified cadence: **exactly one `agg_eval` per data bin**,
  emitted at bin completion (`data_id` k → the eval at its final `iteration_per_data_id`). If a bin ever
  emits multiple, the watcher takes the **last** as representative.
- **Mechanism:** a poller in the `expt_launch` ticker loop tails the aggregator telemetry, maintains the
  per-`data_id` representative accuracy, and fires when the last `W` distinct completed bins are all ≥ `τ`
  → clean process-group termination.
- **Health verdicts:** new `CONVERGED` (window satisfied) and `DID_NOT_CONVERGE` (hit a safety cap first),
  alongside existing `COMPLETED` / `CRASH` / `WALL_CEILING`.
- **Output:** per-run `converge.json` = time-to-converge in **wall + vclock (sim) + data_id + round**.
  *This is the Expt-1 metric captured at the source*, not reconstructed post-hoc.
- **Safety caps stay:** `max_runtime_s` / `max_data_id_progress` bound a non-converging run.

New flags on `run_sequential.sh`: `--target-acc τ`, `--converge-window W`.

---

## 3. Telemetry provenance

Most metrics are **already emitted** (see the map below). Two instrumentation additions were made,
both "route an already-known quantity to `telemetry.emit`", off when `FLAME_TELEMETRY_DIR` unset
(zero cost elsewhere). **Both are implemented and validated** (2026-07-06):

- **(WS3-a) Network bytes + message counts** — a dedicated **`comm` event**
  (`flame/telemetry/events.py:build_comm`), emitted **both directions** with `direction`
  (`agg_to_trainer`/`trainer_to_agg`), `size_bytes`, `payload_kind` (`weights`/`var_bad`/`gradients`),
  `n_tensors`, tagged with round/data_id. Sizes reuse the values already computed for the debug logs:
  - trainer upload at [`fwdllm_trainer.py`](../../flame/mode/horizontal/syncfl/fwdllm_trainer.py) `_send_weights`;
  - aggregator dispatch at **both** send sites — the sync `_distribute_weights` **and** the async
    `_distribute_weights_async` (the async site is the one fluxtune uses; instrumenting only the sync
    path silently produced zero agg→trainer events until fixed).
- **(WS3-b) Perturbations / forward-passes per client** — module-global counters in
  [`fwdgrad_utils.py`](trainer/forward_training/fwdgrad_utils.py) (`_FWD_PASSES` / `_JVP_EVALS`,
  incremented in `calculate_jvp*`; each trainer is its own process ⇒ counters are per-client), read
  into `trainer_round` as `forward_passes_iter/total` + `perturbations_iter/total`. Gives Expt 3 a
  **hardware-independent** compute denominator immune to the 8-GPU contention confound. (Validated:
  10 perturbations = 20 forward passes per iteration.)

**Aggregator "active GPU time" = aggregator compute wall-time** — measured from the NON-OVERLAPPING
`agg_round` phase fields (`aggregate_fedavg_s` + `eval_s`), **not** summed `step_timing` (which
double-counts because `timer_decorator` calls nest), and **not** CUDA-event-isolated GPU time.
CUDA-event timing only if a reviewer challenges the number.

---

## 4. The five experiments (analyses over the one run-set)

All reducers live in `expt_scripts/compare_baselines.py` (one function per experiment).
Legend — **provenance**: `EMIT` already in telemetry · `DERIVE` reducer over existing telemetry ·
`WS3` the instrumentation add (now emitted) · `WS2` from the convergence watcher.

### Experiment 1 — Time to target accuracy
> **Takeaway:** Fluxtune reaches target accuracy faster than FwdLLM and FwdLLM_Plus.
- **Config:** the `main` run-set. Baselines: all three. **Reuse:** none (this *defines* the runs).
- **Metrics reported:**
  - Time to reach `τ` (the convergence event) — wall, vclock, #rounds, #data_bins. *(WS2 `converge.json`)*
  - Maximum accuracy attained. *(DERIVE: max `agg_eval.test-accuracy`)*

### Experiment 2 — Resource utilization (wait-time reduction)
> **Takeaway:** Fluxtune improves utilization by cutting wait times at trainers (primary, thousands) and
> the aggregator (secondary, single).
- **Reuse:** **Expt-1 run-set** (no new runs).
- **Approach:** **derived busy/idle time-fraction** (operator call — no hardware sampler).
  - Trainer busy = Σ`gpu_compute_s` / (last−first ts); idle = `mqtt_fetch_s` + inter-round gaps.
  - Aggregator busy = Σ(`aggregate_fedavg_s`+`eval_s`) / `agg_round.wall_elapsed_s` (non-overlapping
    phases, NOT summed `step_timing`); wait fractions = Σ`barrier_wait_s` / Σ`drain_tail_s` over wall.
- **Metrics reported:** P50 / P90 / P99 busy-fraction **across trainers** over runtime; aggregator
  busy vs. barrier-wait vs. drain fraction. *(DERIVE)*

### Experiment 3 — Compute productivity (learning per unit compute)
> **Takeaway:** At resource-constrained clients, Fluxtune yields more learning per unit compute.
- **Reuse:** **Expt-1 run-set**.
- **Metrics reported:** `Δloss / cumulative compute`, reported against **two** compute denominators:
  - GPU-seconds: Σ(trainer `gpu_compute_s`) + aggregator compute wall-time. *(DERIVE)*
  - **Forward passes** (perturbations): Σ per-client perturbation count. *(WS3-b — hardware-independent)*
  - `Δloss` = first `agg_eval.test-loss` − final `agg_eval.test-loss`. *(EMIT)*

### Experiment 4 — Data transmitted over the network
> **Takeaway:** Fluxtune incurs lower total data overhead despite more messages per round/iteration.
- **Reuse:** **Expt-1 run-set** (requires WS3-a telemetry present at run time).
- **Metrics reported:** total messages sent (each side); total bytes transmitted (each side);
  per-message size distribution. *(WS3-a)*

### Experiment 5 — Client training-session durations & participation
> **Takeaway:** Fluxtune's client sessions are much shorter than FwdLLM's.
- **Reuse:** **Expt-1 run-set**.
- **Active-session definition (precise, per operator):**
  - **async (fluxtune):** selected → next reselection ≈ `dispatch_ts → commit_ts`
    (`agg_round.contributor_intervals`).
  - **sync (fwdllm / fwdllm_plus, round-based reselect):** one-round span from `trainer_round`/`agg_round`
    timestamps.
- **Metrics reported:**
  - P50 / P90 / P99 of session duration across clients + a histogram. *(DERIVE)*
  - **Per-client participation counts at three granularities across baselines** — #rounds, #data_bins,
    #iterations each client participated in. *(DERIVE: `trainer_round` + `agg_round.contributing_trainers`)*

---

## 5. Consolidated metric map (incl. the operator's second list)

| # | Metric | Provenance | Feeds |
|---|--------|-----------|-------|
| 1 | Experiment wall-time before exit | WS2 `converge.json` + `agg_round.wall_elapsed_s` | 1 |
| 2 | Aggregator active GPU time (compute wall-time) | `step_timing`/`agg_round.aggregate_fedavg_s`,`eval_s` | 2,3 |
| 3 | Each client active GPU time | `trainer_round.gpu_compute_s` | 2,3 |
| 4 | Loss at first iteration | first `agg_eval.test-loss` | 3 |
| 5 | Loss after training complete | last `agg_eval.test-loss` | 3 |
| 6 | Updates used per iteration | `agg_round.agg_goal_count`/`updates_in_queue` | — |
| 7 | Total updates over training | Σ `agg_round.agg_goal_count` | — |
| 8 | **Forward passes / perturbations per client** | **WS3-b** `trainer_round.forward_passes_total`/`perturbations_total` | 3 |
| 9 | Data sent from aggregator | **WS3-a** `comm{direction=agg_to_trainer}.size_bytes` | 4 |
| 10 | Data sent from clients | **WS3-a** `comm{direction=trainer_to_agg}.size_bytes` | 4 |
| 11 | Client active: selected→next reselection (async) | `agg_round.contributor_intervals` | 5 |
| 12 | Client active per round (sync reselect) | `trainer_round`/`agg_round` ts | 5 |
| 13 | **Participation counts: rounds / data_bins / iterations per client** | `trainer_round` + `contributing_trainers` | 5 |

---

## 6. Implementation workstreams & status

| WS | Item | State |
|----|------|-------|
| WS1 | `EXPERIMENTS.md` + `experiments.yaml` (backbone) | ✅ |
| WS2 | Convergence-stop watcher (`--target-acc`,`--converge-window`, `CONVERGED`/`DID_NOT_CONVERGE`, `converge.json`) | ✅ validated N=10 (fired at bin 0) |
| WS3-a | Network bytes + message-count telemetry (`comm` event, both directions) | ✅ validated (agg→trainer + trainer→agg) |
| WS3-b | Per-client perturbation / forward-pass counter (`fwdgrad_utils` counters → `trainer_round`) | ✅ validated (20 passes / 10 perturbations per iter) |
| WS4 | `compare_baselines.py` cross-baseline reducer → table/CSV + overlay plots | ✅ (mix-guard + degrades on missing telemetry) |
| — | Enhanced gate: `condition_fp` (incl. delay_factor), tier ② baseline internals, agg_goal-match check, `--run-set` registry launch | ✅ |
| — | Phase 2: fwdllm_plus-under-mobiperf policy; N=100/8-GPU utilization confound | ⬜ deferred |

**Validated end-to-end at N=10** (run `run_20260706_152025_fluxtune_n10_smoke_syn_0_real`):
`CONVERGED` fired, `comm` bytes (both directions) + `forward_passes`/`perturbations` landed in JSONL,
comparison table produced. **Sign-off applied** (2026-07-06): `main` is now N=100, delay_factor=2,
agg_goal=10 matched, target 0.82. **Next action: the N=100 real convergence runs** (§7).

---

## 7. Run workflow (multi-node, misconfig-proof)

The whole point of `experiments.yaml` + the enhanced gate is that a comparison run
is defined **once** and every node verifies it launched the same thing.

### 7.0 Signed-off condition (2026-07-06) — `run_set: main`
N=100 · K=10 · C=10 (sync) / 30 (fluxtune) · **agg_goal=10 matched across all three**
· partition alpha=0.1 · syn_0 · delays ON at **delay_factor=2** · target 0.82 / window 20.
A pre-flight check enforces the agg_goal match; the `condition_fp` (includes delay_factor)
must match across nodes. **Manual pre-run check:** verify the partition group
`niid_label_clients=100_alpha=0.1` exists in `agnews_partition.h5` (the gate only warns —
it can't read the H5 from the launch env) — a missing group crashes all 100 trainers.

### 7.1 Baseline defaults — REVIEW before the first real run
Resolved from `_metadata/baselines.yaml` (the gate's tier ② shows these live per run):

| knob | fwdllm | fwdllm_plus | fluxtune |
|---|---|---|---|
| mode | sync | sync | **async** |
| selector | random | random | **async_oort** |
| optimizer | fedavg | fedavg | **fedbuff** (lr 0.075) |
| native agg_goal | 10 | 10 | **3** |
| native k / c | 5 / 15 | 5 / 15 | 5 / 15 |
| availability tracking | unaware | **ORACULAR** | client_notify |

**agg_goal / k coupling (decided 2026-07-06):** `--c N` fans `agg_goal=N` + `minInit` to **every**
baseline and `--k N` sets k on all — so `--run-set main` (C.sync=10, K=10) drives fluxtune to
`agg_goal=10, k=10`. **This is intended:** the experiment matches `agg_goal=10` across all three, and
fluxtune's async concurrency is set by `C.async=30` (its `c` becomes 30, `agg_goal` stays 10). The
agg_goal-match pre-flight check enforces it. If a future experiment wants fluxtune's native
`agg_goal=3`, that needs a per-baseline agg_goal override (not currently wired).

### 7.2 Single-node run (all three baselines)
```bash
cd lib/python/examples/fwdllm/expt_scripts
bash run_sequential.sh --run-set main --mode real --yes
```

### 7.3 Two-node run (split baselines, ONE source of truth)
Both nodes read the SAME condition from `experiments.yaml` via `--run-set main`;
only `--only` differs. **The gate prints `condition_fp` — it MUST be identical on
both nodes.** If the two fingerprints differ, a knob was mistyped: stop and fix.

```bash
# node A
bash run_sequential.sh --run-set main --only fwdllm,fwdllm_plus --mode real --yes
# node B (shared filesystem: run dirs land in the same experiments/)
bash run_sequential.sh --run-set main --only fluxtune --mode real --yes
```
Pre-run checklist (the gate does most of this — eyeball, don't skip):
1. `condition_fp` identical across nodes.
2. tier ② `mode/selector/optim` match the baseline table above (right algorithm per baseline).
3. `target_acc`, `trace`, `part` are the intended values (🟢 = from flag/registry).
4. no ✗ pre-flight checks (a ⚠ on the niid partition group just says "verify it exists").

### 7.4 After the runs — compare
```bash
python compare_baselines.py --variant real --target-acc 0.82 --window 20 --plots
```
Reducers pull from the same run dirs; the **mix-guard** warns if the baselines'
shared axes (N/partition/trace) disagree — the post-hoc twin of `condition_fp`.

### 7.5 Convergence-stop knobs (recap)
`--target-acc 0.82 --converge-window 20` arm the watcher (WS2). A run ends
`CONVERGED` (window met, `converge.json` written) or `DID_NOT_CONVERGE` (hit a
safety cap). `--run-set main` sets these from the registry, so you rarely pass them.

---

## 8. Files & entry points

| File | Role |
|---|---|
| [`experiments.yaml`](experiments.yaml) | machine registry — run-sets, conditions, analyses (source of truth for *what runs*) |
| [`expt_scripts/run_sequential.sh`](expt_scripts/run_sequential.sh) | launcher — `--run-set`, condition_fp gate, tier ② internals, agg_goal-match check, convergence flags |
| [`../scripts/expt_runner.sh`](../scripts/expt_runner.sh) | shared harness — `expt_launch` (arms watcher), `expt_assert_run` (`CONVERGED`/`DID_NOT_CONVERGE`) |
| [`../scripts/converge_watch.py`](../scripts/converge_watch.py) | WS2 side-car — polls `agg_eval`, writes `converge.json`, kills the run on convergence |
| [`expt_scripts/compare_baselines.py`](expt_scripts/compare_baselines.py) | WS4 reducer — 5-experiment table/CSV + overlay plots + mix-guard |
| `flame/telemetry/events.py` | `build_comm` (WS3-a) |
| `flame/.../fwdllm_aggregator.py`, `fwdllm_trainer.py` | `comm` emit sites (WS3-a, both dispatch paths + upload) |
| `trainer/forward_training/fwdgrad_utils.py`, `FedSgdTrainer.py` | forward-pass counters (WS3-b) |

Session artifacts: `expt_scripts/smoke_logs/<ts>/` (`converge_<run>.json`, manifest, gate spec);
per-run `experiments/run_*/telemetry/*.jsonl`; comparison output `experiments/_compare/`.

---

## 9. Changelog
- **2026-07-06 — implementation landed & N=10-validated.** WS2 convergence watcher (`converge.json`,
  `CONVERGED`/`DID_NOT_CONVERGE`); WS3-a `comm` telemetry both directions (incl. the async-dispatch
  site); WS3-b forward-pass/perturbation counters; WS4 `compare_baselines.py`; enhanced gate
  (`condition_fp` incl. delay_factor, tier ② baseline internals, agg_goal-match check, `--run-set`
  registry launch). Sign-off applied to `main`: N=100, delay_factor=2, agg_goal=10 matched, target 0.82.
- _(init)_ Doc + registry scaffold. Backbone, convergence-stop spec, metric map incl. operator's
  second metric list (#8 forward passes, #13 participation granularities).
