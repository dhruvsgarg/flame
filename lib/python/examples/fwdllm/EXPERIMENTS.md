# FLUXTUNE vs FWDLLM / FWDLLM_PLUS — experiment design (living doc)

**Status:** Tooling **IMPLEMENTED & validated end-to-end at N=10** (2026-07-06). Signed-off
`main` condition (N=100, 84%, syn_0, α=1, delay_factor=2, agg_goal=10 matched) is loaded and
gated; **N=100 real convergence runs are the next action**.
**Owner:** dgarg39 · **Branch:** `dg/fwdllm_sim_unavail`

This is the human design doc. Its machine-readable twin is [`experiments.yaml`](experiments.yaml),
which the tooling **consumes** — `run_sequential.sh` launches the run-set from it and
`compare_baselines.py` reads it to know which runs feed which metric. Keep the two in sync;
same discipline as the pre-flight gate — *what is printed == what runs*. When they disagree,
`experiments.yaml` is the source of truth for **what ran**; this doc is the source of truth for
**what we intend and why**.

**Paper ⇄ code reconciliation:** conflicts-resolved + task tracking live in
[`EXPTS_CHARTER.md`](EXPTS_CHARTER.md) (the charter reconciling this doc with the paper draft
[`05-evaluation.tex`](05-evaluation.tex)). The **run ledger** — which log file on which node feeds
which result/sub-section — is §10 below.

Related docs: [`simulate_fwdllm.md`](simulate_fwdllm.md) (principles), [`PARITY_LOGICAL_TASKS.md`](PARITY_LOGICAL_TASKS.md)
(real↔sim parity), [`fluxtune_contributions.md`](fluxtune_contributions.md) (systems/ML contributions,
incl. the argued memory/inference-only-NPU thesis — a motivation/design claim, not an eval experiment).

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
- **Health verdicts:** `CONVERGED` (window satisfied), `STALLED` (early-terminated, not learning),
  `DID_NOT_CONVERGE` (hit the wall ceiling still learning), alongside `COMPLETED` / `CRASH` / `WALL_CEILING`.
- **Output:** per-run `converge.json` = time-to-converge in **wall + vclock (sim) + data_id + round**
  (or `stall.json` on a stall). *This is the Expt-1 metric captured at the source*, not reconstructed.
- **Wall ceiling = 48h.** A convergence run is accuracy-governed, so `max_runtime_s` defaults to
  **172800s (48h)** whenever `--target-acc` is set (a short default would kill a legitimately-learning
  run). `max_data_id_progress` also bounds it.
- **Stall guard (early-out).** Terminate BEFORE the 48h ceiling if the run is clearly not learning:
  **no PROGRESS within `stall_window_s` (default 7200s = 2h)** → verdict `STALLED`. Any progress
  resets the clock; convergence is checked first, so a just-converged run is never called stalled.
  `stall_window_s=0` disables it. **What counts as progress is set by `--stall-on` (default `either`):**
  - `acc` — best accuracy gained ≥ `stall_min_delta` (default 0.01 = **1% absolute**; accuracy ∈ [0,1]).
  - `loss` — best (running-min) test-loss dropped ≥ `loss_min_rel_delta` (default 0.01 = **1% relative**;
    loss is unbounded/scale-dependent, so the bar is *fractional vs the running-best*, not absolute).
  - `either` — reset the clock if **either** fired. Motivating case: a run can plateau in accuracy while
    test-loss keeps falling (still learning) — `either`/`loss` keeps it alive; `acc` would kill it.

  Both signals use the running-best (max acc / min loss), so a single noisy eval can neither reset the
  clock nor fake progress. *Validated end-to-end: flat acc + flat loss → `STALLED [either]`; flat acc +
  steadily-falling loss → clock resets on the loss signal, keeps running until loss too plateaus.*

New flags on `run_sequential.sh`: `--target-acc τ`, `--converge-window W`, `--stall-window-s S`
(or the hours alias `--stall-window-h H`, e.g. `6` ⇒ 21600s), `--stall-min-delta D`,
`--stall-on acc|loss|either`, `--loss-min-rel-delta R` (all also settable from the registry via
`--run-set`). **The CLI value overrides the registry** — so a run
that stalls too eagerly (flat accuracy but loss still falling) can be re-launched with a wider
idle window, e.g. `--stall-window-h 6`, without editing `experiments.yaml`. The stall window is
part of `condition_fp` (it changes the fingerprint), but it is a *termination-policy* knob, not a
scientific-condition one: the cross-baseline mix-guard compares only N/partition/trace, so widening
it for one baseline's re-run does not taint the Expt-1 comparison.

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

The metric logic lives in `expt_scripts/plotlib/reducers.py` (`load_run` → `RunResult`), consumed by
`compare_baselines.py` (the `expt1..5_*` table wrappers) and `plot_run.py`. Legend — **provenance**:
`EMIT` already in telemetry · `DERIVE` reducer over existing telemetry · `WS3` the instrumentation add
(now emitted) · `WS2` from the convergence watcher.

> ⚠ **Reducer-audit findings (2026-07-07, tracked in [`EXPTS_CHARTER.md`](EXPTS_CHARTER.md) §2b):**
> **N3** — E1 time-to-τ is **reconstructed** from `agg_eval` (streak-over-window scan, over the
> loss-truncated series), it does **not** read `converge.json`; it can silently diverge from the
> watcher's verdict. **N5** — E2 idle is only `1−busy_frac`; `mqtt_fetch_s` is emitted but unused, and
> `barrier_wait_s`/`drain_tail_s` read ≈0 in sim. **N6** — E5 sync sessions use `contributor_intervals`
> (dispatch→commit) for *all* baselines (the one-round-span method is unimplemented);
> `agg_round.contributing_trainers` is emitted but never consumed. Fix or re-scope before the claims land.

### Experiment 1 — Time to target accuracy
> **Takeaway:** Fluxtune reaches target accuracy faster than FwdLLM and FwdLLM_Plus.
- **Config:** the `main` run-set. Baselines: all three. **Reuse:** none (this *defines* the runs).
- **Metrics reported:**
  - Time to reach `τ` (the convergence event) — **wall, #rounds, #data_bins, #iterations**. *(WS2
    intent; currently DERIVE — see N3.)* **Virtual-clock is deferred** (real-mode runs emit no vclock;
    sim vclock is unvalidated, `sim_rate≈0.50`) — add later if a validated sim lands.
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
  - **Forward passes** (perturbations): Σ per-client perturbation count — **PRIMARY / clean**. *(WS3-b —
    hardware-independent: counts actual passes regardless of contention.)*
  - GPU-seconds: Σ(trainer `gpu_compute_s`) + aggregator compute wall-time — **SECONDARY, confounded**.
    *(DERIVE.)* ⚠ `gpu_compute_s` is **wall-time** GPU work; 100 trainers time-share **8 GPUs**, so
    contention inflates it (~8–10 ms/pass clean → ~0.21 s/pass under load, ≈20×), and Fluxtune's ~30
    concurrent clients contend differently than the sync baselines' K=10 bursts — so the GPU-second
    denominator measures scheduling contention, not algorithmic compute. Report it with this caveat; lean
    on the forward-pass denominator.
  - `Δloss` = first `agg_eval.test-loss` − final `agg_eval.test-loss`, **in time order** (never keyed by
    `data_id`, which cycles per round). *(EMIT)*

> ⚠ **Observed (N=100 smoke, 2026-07-07) — does NOT yet support the takeaway.** Learning-per-compute
> ranks **FwdLLM++ > Fluxtune > FwdLLM** on *both* denominators (Δloss/GPU-h 0.055 vs 0.016 vs −0.014;
> Δloss/M-fwd **5.62 vs 1.27** vs −1.26). Fluxtune attains the most *total* Δloss (0.82) and the fastest
> wall-clock convergence (E1) but spends **~5–6× the forward-pass compute** to get there, so per-unit it
> is *less* efficient. GPU-h is confounded by 8-GPU contention (§7.1); the hardware-independent
> forward-pass denominator is clean and still favors FwdLLM++.
> **Why:** Fluxtune's async high-concurrency design does more *unproductive* compute — (i) fedbuff
> **staleness**: concurrent clients train on stale models and their updates are down-weighted, so part of
> the forward-pass compute yields little Δloss; (ii) it keeps ~30 clients busy speculatively. FwdLLM++ is
> synchronous with **oracular** availability, so every forward pass feeds a fresh, fully-weighted update
> on a client that will contribute — no stale or dropped-out work. Concurrency buys wall-clock speed, not
> compute efficiency.
> **Optimize:** staleness-aware admission / adaptive concurrency + cutting JVP forward-pass overhead to
> close the per-compute gap while keeping Fluxtune's speed. (The per-resource-constrained-client framing
> the takeaway intends still needs a per-client-normalized metric — a metric decision, not a plot bug.)

### Experiment 4 — Data transmitted over the network
> **Takeaway:** Fluxtune incurs lower total data overhead despite more messages per round/iteration.
- **Reuse:** **Expt-1 run-set** (requires WS3-a telemetry present at run time).
- **Metrics reported:** total messages sent (each side); total bytes transmitted (each side);
  per-message size distribution. *(WS3-a)*

> ⚠ **Observed (N=100 smoke, 2026-07-07) — contradicts the takeaway.** Fluxtune transmits **~1.9× more
> total bytes** than FwdLLM++ (**146 vs 79 GB**) and ~1.6× more messages — it does *not* incur lower
> overhead here. Per-message size is identical (1.8 MB/upload), so the gap is message **count**, not
> payload.
> **Why:** (i) **Model distribution dominates** — FwdLLM++ (sync) sends the model once per round to the K
> selected clients (weights **9 GB**); Fluxtune (async) has ~30 clients *continuously re-pull* the latest
> global model as they finish and re-enlist → weights balloon to **88 GB**. (ii) **Uploads scale with
> iterations** — one gradient upload (1.8 MB) per trainer-iteration; Fluxtune runs ~1.6× more iterations
> (concurrency) → 58 vs 36 GB up. FwdLLM++'s synchronous rounds amortize both, and oracular selection
> avoids dispatching to non-contributors (it *does* pay a 33 GB method-specific `var_bad` payload, yet
> still totals less). **Same root cause as E3:** async concurrency does more total work.
> **Optimize:** delta/compressed model distribution on re-pull + staleness-aware throttling would cut
> Fluxtune's dominant weight-download term.

> 🔧 **These are single-contribution results — Fluxtune's efficiency levers are not yet delivering.**
> The three contributions (charter B2) are **C1** guided (JVP-magnitude) perturbation selection · **C2**
> dynamic K/C · **C3** intelligent (gradient-aware) aggregation. In the current `main` run-set:
> - **C1 is active** (the only lever exercised).
> - **C2 (dynamic K/C) is OFF** — `main` fixes `agg_goal=10` / `C=30` (static) for the agg_goal-matched
>   head-to-head (§7.0; controller exists but `dynamic_kc.enabled=false`; design in
>   [`docs/dynamic_kc_design.md`](docs/dynamic_kc_design.md)).
> - **C3 is a BORROWED PLACEHOLDER, not Fluxtune's intended aggregation.** ⚠ Correction to earlier
>   wording: the run does **not** use "plain fedbuff." Fluxtune runs the fedbuff **"new"** rate
>   `weight_factor = scale·α(staleness) + (1−scale)·β(stat_utility)` (scale 0.4, a_exp 0.25, b_exp 0.1;
>   `flame/optimizer/fedbuff.py:110`) **plus** a `var ≤ var_threshold=0.3` commit gate — a
>   staleness×utility **scalar rate** borrowed from weight-averaging async FL (async_cifar10 / REFL
>   lineage), applied as scalar multiplication of the update. **This is not gradient-aware and is not the
>   C3 we intend** (see §4-C3 investigation below). Default fedbuff ("old" rate `1/√(1+Δv)`) looked only
>   at round-based staleness; the "new" rate adds statistical utility, but still scalar-weights the update.
>
> These levers target *exactly* the inefficiencies E2–E4 surface: **dynamic C** throttles concurrency
> under high staleness → fewer wasted stale updates and fewer continuous model re-pulls (E3 compute + E4
> bytes); **gradient-aware aggregation** weights contributions by usefulness → more Δloss per forward pass
> (E3); **dynamic K** right-sizes the aggregation goal (E2). **So the E2/E3/E4 efficiency claims must be
> (re)made with C2 + a real C3 enabled** — with only C1, Fluxtune is *expected* to trade efficiency for
> speed (per the charter, we retain E3/E4 as hypotheses assuming C2/C3 deliver). This needs a **separate
> full-system Fluxtune run**: enabling dynamic K/C breaks the deliberate `agg_goal=10` match, so the
> agg_goal-matched condition isolates C1, while the efficiency claims need the full system.
> **E1 (speed + final accuracy) already holds on C1 alone.**

### Experiment 3-adjacent — C3 intelligent (gradient-aware) aggregation: design investigation
> **Status: NOT the intended contribution yet.** The active weighting (above) is a borrowed scalar rate.
> Fluxtune needs a **gradient-aware** aggregation rule. **Full design starter:**
> [`docs/aggregation_design.md`](docs/aggregation_design.md) (FedBuff→FeLiX→FluxTune regime, hypothesis,
> 5-axis design space, candidate schemes S0–S5). This section seeds that design (implement later,
> then move the feature doc to [`fluxtune_contributions.md`](fluxtune_contributions.md) and delete from here).
> **Dimensions to evaluate:**
> 1. **Staleness under iteration-based progression.** Staleness = `agg_model_version − trainer_version`;
>    `_model_version` advances **per data-bin completion**, and a data-bin completes only when the
>    **variance threshold is met** ([`fwdllm_aggregator.py:1422`](../../flame/mode/horizontal/syncfl/fwdllm_aggregator.py#L1422),
>    [`fedbuff.py:194`](../../flame/optimizer/fedbuff.py#L194)) → staleness accrues at the *variance-gated
>    data-bin rate, not wall-clock*. Hypothesis: staleness grows **slower** in Fluxtune than round-based
>    FL. Quantify the effective staleness distribution vs a round-based baseline.
> 2. **Scalar rate vs gradient-aware combination.** Scalar-multiplying a *gradient* update (forward-mode
>    JVP estimate) may be the wrong operator vs down-weighting *weights*. Explore direction-/variance-aware
>    combination (weight by JVP magnitude / SNR / agreement with the running aggregate), not just
>    staleness×loss.
> 3. **Interaction with C1 + the variance gate.** Updates already passed `var ≤ var_threshold`; does
>    re-weighting by loss (`stat_utility`) double-count what the gate filtered?
> 4. **Interaction with C2 (dynamic K/C).** Concurrency C sets how many stale/in-flight updates coexist;
>    aggregation rule and concurrency controller co-determine the wasted work E3/E4 measure.

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
· partition alpha=1 · syn_0 · delays ON at **delay_factor=2** · target 0.84 / window 20
· **wall ceiling 48h · stall-out after 2h with no ≥1% gain**.
A pre-flight check enforces the agg_goal match; the `condition_fp` (includes delay_factor)
must match across nodes. **Manual pre-run check:** verify the partition group
`niid_label_clients=100_alpha=1` exists in `agnews_partition.h5` (the gate only warns —
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

### 7.3 Multi-node run (split baselines, ONE source of truth)
All nodes read the SAME condition from `experiments.yaml` via `--run-set main`;
only `--only` differs. **The gate prints `condition_fp` — it MUST be identical on
every node.** If fingerprints differ, a knob was mistyped: stop and fix.

**Three-node run (one baseline per node) — pass `--clean` so each node auto-clears
any stray workers from a prior run before launching:**
```bash
# node A
bash run_sequential.sh --run-set main --only fwdllm       --mode real --clean --yes
# node B
bash run_sequential.sh --run-set main --only fwdllm_plus  --mode real --clean --yes
# node C  (shared filesystem: run dirs land in the same experiments/)
bash run_sequential.sh --run-set main --only fluxtune     --mode real --clean --yes
```
Pre-run checklist (the gate does most of this — eyeball, don't skip):
1. `condition_fp` identical across nodes (expect `04d64814` for the current `main`).
2. tier ② `mode/selector/optim` match the baseline table above (right algorithm per baseline).
3. `target_acc`, `trace`, `part` are the intended values (🟢 = from flag/registry); `part` = `niid_label_clients=100_alpha=1`.
4. no ✗ pre-flight checks (a ⚠ on the niid partition group just says "verify it exists").
5. `[<baseline>] clean slate verified` printed before launch (the clean-slate guard, §7.6).

### 7.4 After the runs — compare
```bash
python compare_baselines.py --variant real --target-acc 0.84 --window 20 --plots
```
Reducers pull from the same run dirs; the **mix-guard** warns if the baselines'
shared axes (N/partition/trace) disagree — the post-hoc twin of `condition_fp`.

### 7.5 Convergence-stop knobs (recap)
`--target-acc 0.84 --converge-window 20` arm the watcher (WS2). A run ends
`CONVERGED` (window met, `converge.json` written) or `DID_NOT_CONVERGE` (hit a
safety cap). `--run-set main` sets these from the registry, so you rarely pass them.

### 7.6 Stopping a run & the clean-slate guard
The run's trainers/aggregator run in their **own process group** (so the watcher
can signal the whole tree), which means a bare terminal **Ctrl+C would not reach
them**. `run_sequential.sh`/`expt_runner.sh` now install a **SIGINT/SIGTERM trap**:
one Ctrl+C tears down the run's process group + the convergence watcher
(`converge_watch.py`) + the progress ticker, escalates SIGTERM→SIGKILL after a
short grace (`EXPT_INT_GRACE_S`, default 5s), sweeps stragglers, and frees GPU/RAM.

**Manual teardown** (if a run was killed the wrong way and left orphans):
```bash
pkill -TERM -f 'flame.launch.run_experiment'; sleep 3
pkill -9 -f 'trainer/forward_training'; pkill -9 -f 'trainer/pytorch/main.py'
pkill -9 -f 'aggregator/pytorch/main_';  pkill -9 -f converge_watch.py
pkill -9 -f run_sequential.sh
# verify clean (want: nothing, GPU ~0 MiB)
pgrep -af -u "$USER" -f 'run_experiment|forward_training|trainer/pytorch/main.py|aggregator/pytorch/main_|converge_watch.py' || echo clean
nvidia-smi --query-gpu=index,memory.used --format=csv
```

**Clean-slate guard (`expt_assert_clean_slate`).** Before every launch the runner
checks for stray FL workers (own procs only) and residual GPU memory:
- **default:** if the node is dirty it **ABORTS** and prints the kill command (never
  nukes a process you didn't sign off on — safe on shared boxes);
- **`--clean`** (or `EXPT_AUTOCLEAN=1`): kills the stragglers, re-verifies, and only
  aborts if still dirty;
- `EXPT_GPU_FREE_MB` (default 500) warns on residual GPU memory; `EXPT_GPU_STRICT=1`
  turns that warning into an abort.

---

## 8. Files & entry points

| File | Role |
|---|---|
| [`experiments.yaml`](experiments.yaml) | machine registry — run-sets, conditions, analyses (source of truth for *what runs*) |
| [`expt_scripts/run_sequential.sh`](expt_scripts/run_sequential.sh) | launcher — `--run-set`, condition_fp gate, tier ② internals, agg_goal-match check, convergence flags |
| [`../scripts/expt_runner.sh`](../scripts/expt_runner.sh) | shared harness — `expt_launch` (arms watcher + SIGINT/SIGTERM teardown), `expt_assert_clean_slate` (pre-launch guard), `expt_assert_run` (`CONVERGED`/`DID_NOT_CONVERGE`) |
| [`../scripts/converge_watch.py`](../scripts/converge_watch.py) | WS2 side-car — polls `agg_eval`, writes `converge.json`, kills the run on convergence |
| [`expt_scripts/compare_baselines.py`](expt_scripts/compare_baselines.py) | WS4 reducer — 5-experiment table/CSV + overlay plots + mix-guard (cross-baseline) |
| [`expt_scripts/plot_run.py`](expt_scripts/plot_run.py) | per-run twin — streams ONE run's telemetry (handles the >1 GB agg JSONL) → full 5-experiment plot set + `summary.json` |
| `flame/telemetry/events.py` | `build_comm` (WS3-a) |
| `flame/.../fwdllm_aggregator.py`, `fwdllm_trainer.py` | `comm` emit sites (WS3-a, both dispatch paths + upload) |
| `trainer/forward_training/fwdgrad_utils.py`, `FedSgdTrainer.py` | forward-pass counters (WS3-b) |

Session artifacts: `expt_scripts/smoke_logs/<ts>/` (`converge_<run>.json`, manifest, gate spec);
per-run `experiments/run_*/telemetry/*.jsonl`; comparison output `experiments/_compare/`.

---

## 9. Changelog
- **2026-07-07 (g) — paper ⇄ code reconciliation ([`EXPTS_CHARTER.md`](EXPTS_CHARTER.md)).** Opened the
  charter reconciling this doc with `05-evaluation.tex`. Key corrections landed here: (i) **contribution
  taxonomy → C1 guided perturbations / C2 dynamic K/C / C3 intelligent aggregation** (async is structural,
  not a numbered contribution); (ii) **C3 is a borrowed placeholder, not the intended gradient-aware
  aggregation** — corrected the earlier "plain fedbuff / off" wording (fluxtune actually runs the fedbuff
  "new" staleness×utility scalar rate + variance gate; still not gradient-aware) and added the C3 design
  investigation (§4); (iii) **E1 units → wall/rounds/data-bins/iterations**, vclock deferred (real emits
  none, sim unvalidated); (iv) **E3 forward-pass denominator is primary**, GPU-seconds secondary with the
  8-GPU-contention caveat; (v) reducer-audit gaps N3/N5/N6 logged; (vi) added the run ledger (§10). Open
  item: α (config says 1, runs ran 0.1). Paper-side rewrites (DistilBERT/AG News substrate, FwdLLM_Plus
  definition, fidelity accuracy-parity, surcharge numbers) delivered in `05-evaluation.tex`.
- **2026-07-07 (f) — E3/E4 observed results (Fluxtune trades efficiency for speed).** N=100 smoke data
  contradicts the E3 and E4 takeaways: FwdLLM++ is **more compute-efficient** (Δloss/M-fwd 5.62 vs 1.27)
  **and more communication-efficient** (79 vs 146 GB total) than Fluxtune. Root cause (both): Fluxtune's
  async high-concurrency design does more total work — fedbuff staleness + speculative concurrency waste
  forward-pass compute, and continuous async model re-pulls balloon weight-download bytes (88 vs 9 GB),
  while more iterations mean more gradient uploads. FwdLLM++'s sync rounds + oracular availability
  amortize communication and spend every forward pass on a fresh, fully-weighted update. Fluxtune's win
  is E1 (wall-clock speed + final accuracy), not per-unit efficiency. Documented under §4 E3/E4 with
  optimization directions (staleness-aware admission, delta model distribution, JVP overhead reduction).
  **Caveat:** these runs exercise only Fluxtune's contribution 1 (guided perturbations); its efficiency
  mechanisms — dynamic K/C (2) and intelligent aggregation (3) — are OFF (static `agg_goal=10`/`C=30`,
  plain `fedbuff`), so the E2/E3/E4 efficiency claims must be re-made with a full-system Fluxtune run (a
  separate condition, since enabling dynamic K/C breaks the agg_goal match). E1 holds on contribution 1.
- **2026-07-07 (e) — target 0.82 → 0.84 + paper-figure pipeline.** Raised the `main` convergence
  target to **0.84** (fluxtune reaches it; the baselines don't — the E1 gap is the story) in
  `experiments.yaml` and the plot defaults. Added `expt_scripts/plotlib/` (single-source SOCC-2026
  style + baseline registry + streaming reducer + figure builders) and `make_paper_figs.py`
  (figs.yaml manifest → PDF-only, timestamped output). `plot_run.py`/`compare_baselines.py` migrated
  onto the shared reducer. Fixed a latent bug: eval reducers keyed accuracy/loss by `data_id`, which
  cycles per round, silently overwriting earlier rounds and corrupting Δloss — now a time-ordered
  series. Per-run cutoff at the **last significant test-loss improvement** (loss, not accuracy, is the
  grounded learning signal); visual EMA smoothing (`--smooth`).
- **2026-07-07 (c) — loss-aware stall guard (`--stall-on`).** The stall guard now resets its idle
  clock on progress in **accuracy, loss, or either** (default `either`), not accuracy alone. Loss
  progress is a **relative** drop vs the running-best (`--loss-min-rel-delta`, default 1%) because
  test-loss is unbounded; accuracy stays **absolute** 1%. Both use the running-best (max acc / min
  loss) so a noisy eval can't reset or fake progress. Fixes the failure mode where a run whose
  accuracy plateaus while test-loss keeps falling (still learning) was killed at the window. Wired
  through `converge_watch.py` (`--stall-on`, `--loss-min-rel-delta`; stall.json now records
  `stall_on`/`best_loss`/`milestone_loss`), `expt_runner.sh`, `run_sequential.sh` (flags + registry
  + gate row + `condition_fp`), and `experiments.yaml` defaults. Validated end-to-end: flat acc +
  flat loss → `STALLED [either]`; flat acc + falling loss → kept alive until loss also plateaus.
- **2026-07-07 (b) — CLI-configurable stall window (hours alias).** Added `--stall-window-h H`
  to `run_sequential.sh` (ergonomic hours alias for `--stall-window-s`); either overrides the
  registry's `stall_window_s`. Motivated by a fwdllm N=100 run the 2h guard killed at the very
  start (flat accuracy but loss still falling slowly). Re-launch with a 6h idle window:
  `run_sequential.sh --run-set main --only fwdllm --mode real --clean --yes --stall-window-h 6`.
- **2026-07-07 — per-run plots + ticker-orphan fix.** Added `plot_run.py`: the single-run
  twin of `compare_baselines.py` that streams one run's telemetry in a single pass (the
  aggregator JSONL runs >1 GB — never loaded whole) and renders the full 5-experiment plot
  set + `summary.json`. Used it on the first N=100 fluxtune convergence run
  (`run_20260706_185045…`, STALLED at **84.08%** > target, 139 bins). **Fixed a harness hang:**
  the progress ticker was assumed to be its own process group (`set -m`), but job control did
  not place the backgrounded subshell in a fresh group, so `kill -"$ticker_pid"` (a
  process-group signal) missed it and the following `wait "$ticker_pid"` blocked forever —
  `expt_launch` hung after a watcher/Ctrl+C stop and the orphaned ticker kept printing
  `… Ns elapsed …`. The ticker now **self-terminates** the instant the run process dies and is
  torn down **by PID** (+ its `sleep` child), never by group.
- **2026-07-06 (c) — teardown, clean-slate guard, alpha=1.** Ctrl+C/SIGTERM now cleanly tears
  down the whole run (own process group) + watcher + ticker and frees GPU/RAM (was: orphaned
  workers, forever-looping ticker). Added `expt_assert_clean_slate` pre-launch guard (`--clean` /
  `EXPT_AUTOCLEAN`) so a run never starts on top of a prior run's stragglers. Manual teardown +
  guard documented in §7.6. **`main` condition partition changed alpha=0.1 → alpha=1** (base-config
  default; group present in `agnews_partition.h5`); `condition_fp` is now `04d64814`.
- **2026-07-06 (b) — termination policy.** Convergence runs now use a **48h wall ceiling** (was 1h)
  when `--target-acc` is set, plus a **stall guard**: terminate early (`STALLED`) if best accuracy
  gains < `stall_min_delta` (1%) within `stall_window_s` (2h). Wired through the watcher, registry
  defaults, gate (`stall_guard` row + fingerprint), and `--run-set`. Stall logic unit + integration tested.
- **2026-07-06 — implementation landed & N=10-validated.** WS2 convergence watcher (`converge.json`,
  `CONVERGED`/`DID_NOT_CONVERGE`); WS3-a `comm` telemetry both directions (incl. the async-dispatch
  site); WS3-b forward-pass/perturbation counters; WS4 `compare_baselines.py`; enhanced gate
  (`condition_fp` incl. delay_factor, tier ② baseline internals, agg_goal-match check, `--run-set`
  registry launch). Sign-off applied to `main`: N=100, delay_factor=2, agg_goal=10 matched, target 0.82.
- _(init)_ Doc + registry scaffold. Backbone, convergence-stop spec, metric map incl. operator's
  second metric list (#8 forward passes, #13 participation granularities).

---

## 10. Run ledger (which log feeds which result)

Update as runs land — this is how we know exactly which log file on which node backs each figure/claim.
`Status`: SMOKE (validation, not for paper) · FINAL (paper number) · STALLED/CONVERGED/DNC (verdict).
⚠ The three N=100 runs below ran at **α=0.1** (log filenames say `alpha0p1`). **α=0.1 is now excluded** —
learning was too slow across all baselines to complete convergence runs — so the paper's primary condition
is **α=1** (`experiments.yaml main`), and these runs will be **re-run at α=1** for final numbers. Treat the
α=0.1 runs as smoke / evidence-that-0.1-is-too-slow, not paper numbers.

| Run dir | Baseline | Node | Condition | Verdict | Feeds | Notes |
|---|---|---|---|---|---|---|
| `run_20260707_015846_fwdllm_n100_smoke_syn_0_real` | fwdllm | shepherd | N=100, syn_0, α0.1, df=2, agg_goal=10 | SMOKE | E1–E5 (baseline) | log `07_07_26_01_59_random_n100_default_alpha0p1_syn0_*` |
| `run_20260706_185023_fwdllm_plus_n100_smoke_syn_0_real` | fwdllm_plus | kaylee | N=100, syn_0, α0.1, df=2, agg_goal=10 | SMOKE | E1–E5 (baseline) | log `06_07_26_18_50_random_n100_oracular_alpha0p1_syn0_*`; oracular **inert** at syn_0 |
| `run_20260706_185045_fluxtune_n100_smoke_syn_0_real` | fluxtune | shepherd | N=100, syn_0, α0.1, df=2, agg_goal=10, C=30 | STALLED @84.08% (139 bins) | E1–E5 (C1-only) | log `06_07_26_18_51_async_oort_n100_client_notify_alpha0p1_syn0_*`; C2 off, C3=placeholder |
| _pending_ | fluxtune (full-system) | — | C2 dynamic K/C **ON** + real C3 ON | — | E2/E3/E4 efficiency | breaks agg_goal match → separate run-set |
| _pending_ | all three | — | `mobiperf_*` (real-world availability) | — | E1 headline | needs fwdllm_plus-under-scarcity policy |
| _pending_ | fwdllm (or port) | — | fidelity: accuracy vs `xu2024fwdllm` | — | Setup (D3) | locate **old** run data; accuracy parity, not time |
| _pending_ | ablations | — | JVP-sens / K-C-sens / α∈{0.1,0.5} | — | Ablation §§ | tooling TBD (charter §2e) |
