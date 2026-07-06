# High-Fidelity Simulator for FwdLLM — Real↔Sim Parity

**Active build (branch `dg/fwdllm_sim_unavail`).** A simulated-clock runner for the `fwdllm` example
(FedFwd / forward-gradient FL) that reaches **real↔sim parity** across the **fluxtune / fwdllm / fwdllm++**
baselines — at **100% availability (syn_0)** first (Phase 1), then under **unavailability**
(syn_20/50/mobiperf, Phase 2), then **beyond syn_0** (Phase 3). fwdllm has no native sim clock; the build wires
flame-core's virtual clock + availability substrate into fwdllm's variance-gated gradient loop. We reuse
async_cifar10's virtual clock, sct reorder buffer, in-flight gate, availability substrate, and parity ladder
wherever they transfer, and deviate where the workload demands (fwdllm aggregates **gradients** not weights;
**variance-gated dynamic-K** commit cadence; **`data_id`** progress axis; one-message-per-call grad loop;
rollback across agg-goal cycles).

> ## PREAMBLE — how to maintain this doc (READ BEFORE EDITING)
> This is a **living status doc**, not a changelog. It must stay **rich but crisp**: enough to reconstruct *why*
> a decision was made, never a running history. Per-run history lives in git + the parity JSONs; the code is the
> source of truth for *what* the mechanism is. Every edit obeys:
> - **Current-truth only.** §A, the scoreboard, and the open-issues table describe the state **right now**.
>   Rewrite them **in place** — never stack dated "UPDATE" blocks. When something closes, **delete it from the
>   open list** and leave at most a one-line trace in §G (fixes) or §H (dead-ends).
> - **One line per landed item.** A fix that worked = one line in §G (**≤20 words problem + ≤20 words fix**). A
>   belief that was refuted = one line in §H. A deviation = one line in §K (anchor + rationale). No paragraphs.
> - **Keep only what teaches.** Retain a big root-cause or a conceptual correction that would otherwise be
>   re-litigated; **drop nitpicks** and mechanical scaffolding once landed (leave a pointer only if still
>   referenced). If an entry no longer changes a future decision, delete it.
> - **No contradictions.** An issue is in exactly one place: OPEN (open table) **xor** CLOSED (§G/§H one-liner).
>   Never both. Cross-link with §-anchors instead of restating.
> - **Decide forks explicitly.** When fwdllm diverges from async_cifar10, log the choice in §K and surface it in
>   the §B.1 delta table — don't silently copy or silently invent.

**Prerequisites:** [async_cifar10/PARITY.md](../async_cifar10/PARITY.md) — parity methodology (ladder,
roles/tiers/gating, run-length budget, landed sim mechanisms); **fwdllm's rung catalog is PARITY.md §F**.
[async_cifar10/UNAVAILABILITY_DESIGN.md](../async_cifar10/UNAVAILABILITY_DESIGN.md) — the availability substrate
fwdllm inherits via its aggregator class chain (ClientAvailability mixin, trace-read effect path, two-ledger
discipline, starvation self-termination, A6/A7/A8/K11 ground-truth rungs).

---

## §A  Current status

> **⏸ SIM DEBUG PAUSED (2026-07-06) — real-experiment telemetry/impl next on this branch.** #15 update: the phantom
> commit-path stall is **FIXED + P3-validated** (`sim_compute_truthful_gate`; `STUCK_EVICT=0`, `phantom_skip` firing,
> 30s failsafes gone). `sim_rate` is still <1 for a **NEW, non-drain-gate** reason — (a) the vclock omits `aggregate()`
> variance-compute wall (sim 32s / real 29s, symmetric, uncredited) + (b) GPU≈D no-headroom (#1d). So the "sim_rate=0.50
> — a COMMIT-PATH STALL / phantom drain gate" reading in the scoreboard + prose immediately below is now the **fixed**
> part; the live residual + resume plan live in [PARITY_LOGICAL_TASKS.md](PARITY_LOGICAL_TASKS.md) (PAUSED checkpoint at
> top + #15). Don't re-debug the sim until the real-experiment work lands.

**All three baselines run end-to-end (#14/#1c/#13 fixed). SYNC baselines are in good shape: with the full registry
delay (`--delay-factor 1`) `sim_rate` is now 2.9–3.0 (#12c RESOLVED for sync) and K-D31 makes bin-1 cohort order
BIT-EXACT (P2-7a validated). The two live fronts are now BOTH understood at the root: (1) a SYNC float-nondeterminism
wall at ~bin 7 (grads are not bit-reproducible → exact cadence parity is unattainable past bin ~6 → the parity
target beyond bin 1 must be DISTRIBUTIONAL); (2) fluxtune `sim_rate = 0.50` — a COMMIT-PATH STALL: hold-to-commit is
a CORRECTNESS check (a trainer is freed only when its update commits), so the commit RATE sets throughput; the sim's
drain blocks real wall on PHANTOM `_sim_inflight_expected` entries (30s failsafe) → correctly-held trainers idle far
longer than their GPU pass → 1.54× concurrency vs real's 3.37×. NOT a gate bug (felix's gate is INERT), NOT
over-restrictive hold-to-commit, NOT GPU under-provisioning (pinning clean — §H). Fix = fast/non-stalling commit path.
Grounding + fix in [PARITY_LOGICAL_TASKS.md](PARITY_LOGICAL_TASKS.md) (FELIX GROUNDING + #15).**

Latest FULL pairs (`run_sequential.sh --mode both --delays on --delay-factor 1 --max-runtime-s 2700`,
`run_20260705_1924 → 2046`):

| baseline | sim `sim_rate` | vclock / wall | verdict |
|---|---|---|---|
| **fwdllm** (sync) | **2.93** ✓ | 1139s / 389s | clean; sim ~3× faster than the virtual time it models |
| **fwdllm_plus** (sync) | **3.01** ✓ | 1137s / 378s | clean; sim ~3× faster |
| **fluxtune** (async) | **0.50** ⛔ | 1191s / 2425s | `sim_rate<1` = a commit-path stall (phantom drain gate, #15), not #12c; hold-to-commit is correct |

**Why fluxtune `sim_rate = 0.50` — objective real↔sim telemetry (supersedes the #12c-delay-headroom reading).**
Real and sim do the SAME GPU work (~3.7–4.0k trainer-s) with the same ~480s 8-way pipeline floor. Real packs it into
1210s wall by keeping **3.37 trainers on the GPUs at once (98% busy)**; the sim takes **2425s** at **1.54×
concurrency (85% busy)**, converting real's correctly-skipped device-delay waits (4775s) into **19807s of trainer
idle in `recv`** (vs real 380s). Per-commit: real reaches agg-goal in **4.30s wall**; the sim models **3.27s vclock**
but spends **6.60s wall** → sim_rate 0.50. **ROOT (corrected 2026-07-06):** hold-to-commit is a CORRECTNESS check — a
trainer is freed only when its update is committed (guards: no same-version re-dispatch, none while computing, none
while returned-but-uncommitted), so the commit RATE sets throughput. In felix/cifar commit is effectively instant
(compute ~0.4s) so held trainers barely idle; fluxtune's commit path STALLS — the drain's `earlier_stuck` gate blocks
real wall on a PHANTOM `_sim_inflight_expected` entry (a trainer stamped expected-at-dispatch that isn't computing,
e.g. still waiting for weights the gate-blocked aggregator can't send) → 30s failsafe → the correctly-held trainers
idle 30s instead of ~one GPU pass. Fix: make the commit path fast/non-stalling (gate only waits on a
genuinely-computing trainer); **hold-to-commit stays**. Grounding: [PARITY_LOGICAL_TASKS.md](PARITY_LOGICAL_TASKS.md)
FELIX GROUNDING F5/F6 + D1-D3.

### Parity scoreboard — REFERENCE baseline (checker run on the pairs above; `expt_scripts/run_parity.py --yes`)
*These are the numbers we hold against until the open issues resolve — it will be a while before a longer run.
The stale pre-rewrite counts (49/4/21 etc.) are retired; do NOT reference them. Numbers below are the ENFORCED
counts AFTER the logical-parity rungs landed (`cohort_sequence` EXACT + V2 mean-guard, [PARITY_LOGICAL_TASKS.md](PARITY_LOGICAL_TASKS.md)
P1) — these two now correctly FAIL the order/grad divergence that used to be invisible.*

| baseline | pass / fail / skip | JSON |
|---|---|---|
| **fwdllm/syn_0** | **41 / 13 / 22** | `experiments/_parity_reports/parity_fwdllm_syn_0_20260705_045724.json` |
| **fwdllm_plus/syn_0** | **36 / 17 / 22** | `parity_fwdllm_plus_syn_0_20260705_063350.json` |
| **fluxtune/syn_0** | **35 / 19 / 20** | `parity_fluxtune_syn_0_20260705_080931.json` |

*(skip +1 each vs the prior ref = the new `timing_overrun` DIAG rung, which SKIPs on these banked logs — they
predate its P2-6 `training_overran` telemetry; it populates on the databin1 run. Pass/fail unchanged: the K-D30
full-cohort gate flipped fwdllm's `selection`/`aggregation_sequence`/`utility` from a trivial GATED pass to GENUINE
enforcement without moving the counts — they pass truthfully; `utility` still fails on a pre-existing pooled KS.)*

**Fails, categorized by blast radius (fix the SHARED roots first — principle #14).**
- **SHARED — all 3 (top priority):** `throughput` / `overhead_residual` / `per_round_advance` (the `sim_rate<1`
  clock-rate family → #12c); `gpu_budget_real` / `gpu_budget_sim` (mis-applied invariant, symmetric real/sim
  overrun — likely a rung-definition bug, not a divergence); `v1_iter_per_data_id` / `v5_variance_pass_ratio` /
  `g2_grad_pool_size` / `utility` / `convergence` (cadence/emergent). **The bin-1 logical check (below) splits
  these:** length-confound for fwdllm/fwdllm_plus (cadence identical on bin 1), but GENUINE for fluxtune (#1d — the
  async cohort/cadence diverges even on bin 1).
- **fwdllm only:** `phase_gpu_compute` (marginal KS, means match).
- **fwdllm_plus only:** `eligibility` / `avail_timebase` / `selection_detail` (the #7 selection divergence);
  `terminal_state` / `total_commits` (stop-mismatch from unmatched length).
- **fluxtune only:** `selection_detail` / `phase_gpu_compute` / `phase_weights_to_gpu`; `staleness`;
  `terminal_state` / `total_commits` / `convergence_loss` (stop-mismatch + length).

### STRATEGY — nail first-data-bin logical parity before any longer run
The right way to prove parity is **logical determinism, not aggregate curve-matching**: for a matched scope, the
sim must take **the same sequence of steps in the same order** as real — same trainers selected, same order of
update receipt, same aggregations/rollbacks — differing ONLY in wall-clock (the sim skips real waits). **Scope to
the first 1 data bin** (`--max-data-id 1` / `--max-bin 1`): bin 0 already contains many iterations, variance passes,
and aggregations, so it exercises the full cadence machinery while keeping the trace short, matched, and diffable.
Prove logical parity on bin 0/1 FIRST (from the already-banked telemetry — no re-run needed); only then chase the
time dimension. This is what separates the two concerns and isolates length-confound from genuine logic bugs.

### Logical-parity check (TIME-STRIPPED, all available bins) — CURRENT REFERENCE
Tool: `expt_scripts/logical_parity.py [--max-bin N]` — diffs the `agg_round` event stream (data_id, iteration,
receive-ordered contributors, variance decision) real vs sim with every timestamp removed. Real receive-order is
DETERMINISTIC in both real and sim by design (operator-confirmed) → exact match is the correct target.

| baseline | receive-SET (to real's max bin) | cadence | logical parity HOLDS TO | verdict |
|---|---|---|---|---|
| **fwdllm** | 41/41 identical (K=10=all) | 22/41 | **data_id 6** — breaks at **7** | **BREAKS @ bin 7** — but receive-ORDER now 41/41 (K-D31) → the break is NOT order |
| **fwdllm_plus** | 14/14 identical | **14/14 identical** | **real's max (~13)** — no break | **PARITY ✓** over its (shorter) real run |
| **fluxtune** | **3/272 identical** | 29/272 | **breaks at bin 0** | thin-margin overrun (§ #1d) — GPU tail > min D on the contended-doubled GPUs |

**ROOT — CORRECTED (2026-07-05 full-run investigation; supersedes the "order → var → RNG-desync" reading for the
SYNC full run).** With K-D31, fwdllm's receive-ORDER is now **41/41 identical** on the full run — yet cadence still
breaks at data_id 7. So order is NOT the sync full-run cause. The real residual is **grad non-reproducibility given
matched order**: `|Δvar|` is ~1e-3 through bin 6 with *every* `var_good`/force decision matching, then at bin 7
(all-10 cohort, identical order, identical RNG) grads diverge ~1e-3 (GPU fp16 non-reproducibility), which the
split-half variance — a ratio with a near-zero denominator at a bin's first iteration — **amplifies to a 0.26 var
swing**, flipping the `var<0.3` gate at (7,2). Both modes confirmed `jvp_perf_opt=False` (no config skew). This
**answers P0-2 empirically: grads are NOT bit-reproducible → exact cadence parity is unattainable past ~bin 6.**
⇒ **Parity target (operator decision): cohort SET = HARD; `var_good`/cadence = HARD to bin 1, DISTRIBUTIONAL beyond;
`var` VALUE = SOFT (tolerance/KS); receive-ORDER within a set = SOFT for sync (fedavg order-invariant + K-D31
canonicalizes). Keep `cohort_sequence` EXACT but scoped to `--max-bin 1`; add a distributional cadence/var rung for
the full run.** For **fluxtune** the SET still genuinely diverges (#1d) — a timing/pipelining cause, not
nondeterminism. Full diagnosis: [PARITY_LOGICAL_TASKS.md](PARITY_LOGICAL_TASKS.md).

> **STATUS (2026-07-05 — P2-7a VALIDATED + full-run bin-7 wall found).** Databin1 checks
> (`parity_fwdllm_syn_0_165256`, `_plus_165736`, `--max-bin 1`): sync `cohort_sequence` **ok=true,
> set/order/var/cadence = 1.0** — K-D31 closed the benign delay-tie; P2-7a DONE. Full runs then exposed the bin-7
> float-nondeterminism wall above (order matches 41/41, cadence still breaks) → the doc's exact-cadence target is
> valid only ≤bin 1; beyond it must be distributional (pending a two-real-run P0-2 confirmation — strong single-run
> evidence already). **fluxtune (`--delay-factor 1` full run):** `jvp_perf_opt=True` cut GPU 7.57→3.61s MEAN (under
> 4.0s min budget) but the TAIL (4.1–5.4s) still overruns on the two doubled GPUs (10 trainers / 8 GPUs) →
> `set_match=3/272`. The overrun is now a thin-margin contention effect, not the JVP algorithm.

### Open issues (OPEN only — closed items live in §G/§H)
| # | issue | baseline(s) | next step |
|---|---|---|---|
| **#15** ⭐⭐ (⏸ PAUSED) | **fluxtune `sim_rate<1` — phantom COMMIT-PATH STALL FIXED (`sim_compute_truthful_gate`, P3 VALIDATED 2026-07-06: `STUCK_EVICT=0`, `phantom_skip`→65 ~1/commit, 30s failsafes gone). But `sim_rate` still <1** (steady ~0.49; per-commit `Δvclock/Δwall=0.465`). **Residual root (NEW, non-drain-gate):** (a) the vclock OMITS `aggregate()` variance-compute wall — sim 32s / real 29s, symmetric, uncredited (`fwdllm_aggregator.py:1783` folds eval only); (b) GPU≈D no-headroom (`max(gpu,D)≈gpu`, sim blocks in `drain_ready` on the real GPU pass while `buf_depth=6` waits) = #1d. Net: sim wall 145s > real 91s. Full evidence+plan: PARITY_LOGICAL_TASKS.md top (PAUSED checkpoint) + #15. | fluxtune | **PAUSED** (operator doing real-experiment telemetry/impl first). On resume: (1) fold the *non-overlapped* `aggregate()` time into the vclock (measure GPU-overlap first; flag-gated, operator sign-off); (2) GPU-vs-D headroom — 1 trainer/GPU or `--delay-factor` up; (3) longer flag-on run + flag-off A/B for parity. |
| **#N (bin-7 nondeterminism)** ⭐ | **SYNC exact-cadence parity has a float-nondeterminism wall at ~bin 7.** Order matches 41/41 (K-D31) yet cadence breaks: ~1e-3 GPU fp16 grad jitter, amplified by the split-half variance ratio, flips the `var<0.3` gate at (7,2). Not a sim bug. | fwdllm (fwdllm_plus latent) | Confirm with a 2-real-run diff (P0-2). Then land the parity-target relaxation: `cohort_sequence` EXACT scoped to `--max-bin 1`; distributional cadence/var rung (mean-band + KS + `var_good` fraction) for the full run. |
| **#1d** ⭐ | **fluxtune cohort SET diverges (thin-margin overrun).** `set_match=3/272`. `jvp_perf_opt` cut GPU to 3.61s MEAN (<4.0s budget) but the tail (4.1–5.4s) still overruns on the two GPUs that carry 2 trainers each (10/8) + the aggregator's eval GPU. Order flips → wrong 3-of-K commit. | fluxtune | Aggregator-GPU pin landed (K-D33); with #15's pipelining the compute drops toward the ~2.4s uncontended floor (<4.0s). If a residual tail remains: `perturbation_count`↓ (P2-5) or 1-trainer/GPU. |
| **#7** | fwdllm_plus real ~4× slower/round; at syn_0 real sees only ~4.9 eligible vs sim ~9.6. Not a sim bug. | fwdllm_plus | Profile per-iteration reselection + oracular-read cost from the banked per-phase log; explain the eligible-count gap at 100% avail. |
| **#11** | real-mode critical-path waste (`sleep(0.1)` MQTT-settle busy-waits; one-grad-per-poll drain tail) — real-only. | fwdllm, fwdllm_plus (real) | **Deferred to a validated pass** — ZERO parity impact (sim already skips them); removing them changes the working real reference + needs a real run (principle #8/#11c). |

### Next roots — ranked (correctness before time; SHARED before per-baseline — principle #14)
1. **#15 fluxtune — make the commit path fast/non-stalling (time, per-baseline) — TOP, IN PROGRESS.** The one thing
   keeping fluxtune `sim_rate<1`. Fix the phantom `_sim_inflight_expected` so the drain gate never blocks real wall on
   a non-computing trainer → commits flow → held trainers freed promptly. Hold-to-commit STAYS (correctness). Diagnose
   D1/D2 first. Also lifts #1d. Sync `sim_rate` 2.9–3.0.
2. **bin-7 nondeterminism → relax the parity target (SHARED, correctness-of-CHECK).** Land the distributional
   cadence/var rung + scope `cohort_sequence` EXACT to bin 1, after the P0-2 two-real-run confirmation. Un-reds the
   sync full-run cadence fails that are float-noise, not bugs.
3. **#7 fwdllm_plus real speed / selection divergence (per-baseline).** Not a sim bug; explain from telemetry.
4. Then C1/C2 convergence (distributional target) at matched `data_id` per baseline → gate to Phase 2.

### SKIP audit (20–22 skips; ~17 legit)
Legit at Phase-1 syn_0 + `random` selector: 7 availability ground-truth rungs + 4 delivery/withheld (Phase-2
effect path, not built) + 3 DynamicKC (disabled by design) + 2 oort-only (`random` baselines) + `residence`
(async telemetry) + `timing_overrun` (banked logs predate its P2-6 telemetry; populates on the next run). The 12
former rigor-gap skips (4 advance, 8 phase-timing) are un-skipped (K-D21).

---

## §B  How fwdllm differs structurally

fwdllm aggregates **gradients** (JVPs) not weights; commit cadence is **endogenous** (variance-gated dynamic-K);
progress axis is committed **`data_id`** (variance passes), not update count. Gradient values are mode-invariant
given identical input+perturbation seed, so parity reduces to **clock + selection + ordering parity plus a
variance-cadence layer**. Anchors (`flame/mode/horizontal/syncfl/fwdllm_aggregator.py`): `aggregate()` var gate +
rollback/`cached_v` at each `_agg_goal` boundary; `total_data_bins=150`; force-commit cap
`max_iterations_per_data_id`; `reselect_each_iteration` (fwdllm++ per-iteration reselection); sync path
`_aggregate_grads_sync`. Full detail: PARITY.md §F.1.

### §B.1  Real↔sim design deltas vs async_cifar10 — **CURATED, KEEP CURRENT**
Separates an **intentional fwdllm choice** from an accidental discrepancy. The §K column points at the one-line
rationale.

| # | Axis | async_cifar10 | fwdllm | Why | §K |
|---|---|---|---|---|---|
| 1 | Aggregated object | model **weights** | **gradients** (JVPs) | grad values mode-invariant → parity = clock+order+selection + variance-cadence | §F.1 |
| 2 | Progress axis | update/round count | committed **`data_id`** | cadence (updates-per-data_id) is an **output to match**, not an input | principle #2 |
| 3 | Commit cadence | fixed `agg_goal` | endogenous **variance-gated dynamic-K** | the emergent layer cifar doesn't model; V/DK/G rungs verify it | §F.1 |
| 4 | sct delay model | `send + max(gpu, D)` | `send + max(gpu, D)` (**remainder-wait, was additive**) | K-D29: real now sleeps `max(0,D−gpu)` (device wall = D, GPU hidden), so update order = per-trainer D order = deterministic & real↔sim identical. Reverses K-D2 | K-D2/**K-D29** |
| 5 | Per-eval sct | distinct, ~20× faster | **collapses to train sct** | eval lives on the aggregator; forward-grad "train" IS a forward pass (no 20× factor) | K-D3 |
| 6 | Slot residence | per-commit release | **hold slot to COMMIT** (felix port, K-D17b) — a trainer is freed only when its update commits (guards: no same-version dispatch, none while computing, none while returned-but-uncommitted) | Correct for BOTH sync & async — it is a correctness check, not a lever. fluxtune's `sim_rate<1` is a COMMIT-PATH STALL (K-D34/#15), NOT this rule. Commit RATE = throughput | K-D17b/**K-D34**, principle #4 |
| 7 | Surplus grad on rollback | carried | **carried** for async (fluxtune, c≫agg_goal); **drop** stays correct for sync (c≈agg_goal) | drop was benign only for sync; fluxtune dropped ~7/cycle → 2× passes | K-D12 |
| 8 | Async drain primitive | `_sim_recv_min` verbatim | purpose-built `_sim_recv_min_grad` / sync `_sync_sim_recv_first_k` | cifar's per-commit release + withheld paths key on WEIGHTS semantics | K-D4 |
| 9 | `time_mode` default | `"simulated"` | `"real"` (getattr fallback) | fwdllm's whole config corpus is `real`; a `simulated` default risks half-activating an unbuilt path | K-D1 |
| 10 | Cadence telemetry | n/a | **pre-mutation** cycle snapshot (`cycle_data_id`/`cycle_iteration`/`grad_pool_size`/`cached_v_size`) | post-mutation `data_id` advances before emit → off-by-one; snapshot makes V1 exact | K-D9 |
| 11 | Availability tracking (v1) | all `trace_read` | **mixed**: fwdllm unaware, fwdllm_plus `oracular`, fluxtune `client_notify`→approx `trace_read` | baselines carry different models; first-class `client_notify` deferred to Phase 2 | D1 |
| 12 | Launch tooling | single parity template (`debug_run.sh`) | per-baseline yamls + `_sim` files (`run_sequential.sh`) | different config models; both source the shared harness `examples/scripts/expt_runner.{sh,py}` | this work |

---

## §C  Baseline matrix (resolved from the launcher configs)
| baseline | sync/async | selector | agg | tracking / avail | reselection | K (agg_goal) | dynamic_kc |
|---|---|---|---|---|---|---|---|
| **fluxtune** | async | `async_oort` | fedbuff (+server LR, JVP) | `client_notify`→`trace_read` v1 (aware-at-select, reactive-90s) | — | 3 | disabled |
| **fwdllm** | sync | `random` | fedavg | `default` unaware (reactive-90s) | per-round | 10 (=c) | — |
| **fwdllm_plus** | sync | `random` | fedavg | `oracular` (aware-at-select via trace read) | per-iteration | 10 (yaml is source of truth) | — |

Phase order (locked): Phase-1 syn_0 → Phase-2 unavailability → Phase-3 beyond syn_0. One baseline at a time.
Stage 3 oort rungs run **only** for fluxtune; sync-barrier rungs run for **fwdllm/fwdllm_plus**.

---

## §D  Parity ladder for fwdllm (rung defs live in PARITY.md §F)
- **Reused verbatim:** Stage 0 TC1/K10; Stage 1 P3/K1/K7; Stage 2 A1–A3 (+A6/A7/A8 once avail telemetry lands);
  Stage 3 S3/4, A2c + the oort stack (fluxtune only); Stage 4 T2/K6/T_mqtt; Stage 8 C1; Stage 9 budget/stop.
- **Modified** for variance-gated dynamic-K: K3a/K3b/K2/U3/K8/U2 → re-keyed to the variance-pass boundary /
  committed `data_id` (PARITY.md §F.3).
- **New** variance-cadence layer: V1–V5, DK1–DK3, G1–G2 (PARITY.md §F.4). Localize down; never fix an EMERGENT
  rung directly. `var_threshold` / `max_iterations_per_data_id` are baseline knobs, not parity levers.
- **Availability** rungs (A1–A5, A6/A7/A8/K11, withheld/abandon/starvation) inherited; apply once Phase-2 wires
  the effect path + telemetry.

---

## §E  Roadmap — remaining phases

**Phase 1 (syn_0) — CLOSE-OUT (near done):** #14/#1c/#13 fixed; the drain runs at speed. Remaining: (1) **#12c
`--delay-factor` run** to drive `sim_rate ≥ 1`, then re-run the checker + bank the three pairs; (2) **#7**
fwdllm_plus real slowness + selection divergence from telemetry; (3) C1/C2 convergence at matched `data_id` per
baseline → gate to Phase 2.

**Phase 2 — unavailability (syn_20/50/mobiperf).** Wire the ClientAvailability effect path into the grad loop:
send-time gate (real) / `delivery_ts = max(sct, next_avail)` buffering (sim); two ledgers; reactive-90s in-flight;
starvation vclock-advance; per-baseline `avail_select_filter`. Emit `EVENT_AVAIL_CHANGE` +
`agg_belief_change`/`send_gate_wait` (builders in `flame/telemetry/events.py`) → unlocks A6/A7/A8/K11. **D3 watch:**
does a withheld/late grad roll into `cached_v` on rollback, or a stale-version late grad inflate `var` (thus
dynamic-K)? Over-instrument before trusting cadence. Exit: A1–A5 + A6/A7/A8 PASS; self-stops; withheld grads
delivered not dropped.

**Phase 3 — beyond syn_0.** Full ladder under scarcity; bin V1/V2 by run-fraction to separate a constant mix bias
from a compounding variance-feedback loop. Exit: curves within tolerance at matched `data_id`; K8/U2 within bar;
V1/V2 binned residual flat.

---

## §F  Locked principles (from async_cifar10, carried over)
1. **Sim does real forward-grad compute, charges modeled time.** GPU runs the real JVP; the agg stamps
   `sct = sim_send_ts + max(real_gpu_s, D) + leg`. **Never** put overhead on the vclock (`vclock = max(vclock, sct)`).
2. **Progress axis is `data_id`.** Updates-per-data_id is the dynamic-K random variable V1 validates — an output
   to match, not an input to assume.
3. **Variance is an emergent gate; localize, never tune it.** `var_threshold` / `max_iterations_per_data_id` are
   baseline-defining config knobs.
4. **Slot residence must survive variance rollbacks.** Release keys on the agg-goal boundary; the sct reorder
   buffer must not strand a grad across a rollback.
5. **DynamicKC: validate the input before the policy** (DK3 CONTROL before DK1/DK2). The controller is shared;
   don't fork it per baseline.
6. **Real is the reference only after admissibility.** A real↔sim gap has two fix directions — check whether the
   **real** input is the divergent side before tuning sim.
7. **syn_0 byte-identical gate OFF; unavailability config-gated.** Reuse the async_cifar10 flag axes; don't delete
   eligibility plumbing.
8. **Fix the concept, not the symptom — never regress a working example.** Classify a mechanism as
   **real-transport artifact** (incremental drain, `recv_fifo` timeouts, queue persistence — no sim analog; gate
   `and not self.simulated`) vs **algorithmic property**. Scope-check before editing shared code:
   `fwdllm_aggregator.py` = fwdllm blast radius; `top_aggregator.py` / shared `parity` engine / `_sim_recv_min`
   can silently break async_cifar10.
9. **Match pytest scope to blast radius.** fwdllm-only edit → `pytest tests/mode -k fwdllm`; shared parity engine
   → add `examples/async_cifar10/scripts/parity` + `tests/mode -k parity`; shared stack → full `pytest tests/`.
10. **Comment the WHY, crisply.** A conceptual choice, a real↔sim divergence + rationale, or a failure mode + why
    the fix takes its shape — one or two tight sentences. Deeper rationale → §K.
11. **Telemetry-FIRST, then instrument, then (rarely) run.** (a) Validate/refute from telemetry ALREADY ON DISK
    first — name the exact field/line. (b) Ship telemetry + plot + pytest IN THE SAME CHANGE as any new mechanism.
    (c) A run is justified only to observe an EMERGENT quantity no stored telemetry can yield — then run the
    SHORTEST length that exhibits it, smoke first, one mechanism per round. (d) fluxtune surfaces roots first.
12. **Consult PARITY.md vclock rules BEFORE any sim-clock change.** Clock is a monotone `max`
    (`vclock = max(vclock, sct)`); NEVER put overhead on it; `sct = send + max(gpu,D) + leg` (leg NOT on
    `trainer_speed`/utility); sync charges MAX-of-K sct, async the K-th-fastest; the sim SKIPS real waits and
    reconstructs order from sct (`SimReorderBuffer`).
13. **The vclock is virtual wall-time; the sim MUST produce SPEEDUP (`sim_rate = vclock/wall ≥ 1`).** The vclock
    advances exactly as the real wall-clock WOULD after the trainer's simulated waits — but faster, because the sim
    **elapses through** the modeled time remaining after GPU compute instead of actually waiting it out. fwdllm
    nuance: the forward-grad "train" is a REAL GPU pass (~2s) that MUST run in sim for grad mode-invariance — that
    ONE GPU pass (parallel across trainers) is the only irreducible real wall; transport/inter-round/delay waits
    are vclock jumps, never process sleeps. `sim_rate < 1` means the sim under-charges the vclock OR is stalling on
    a real wait it should skip (#15). Emit it every run (`[VCLOCK_PROGRESS]`).
14. **Correctness before speed; SHARED roots before per-baseline.** Fix major logical-correctness divergences
    (wrong selection order, wrong receive order, wrong cadence) before any throughput/wall tuning. Rank a root by
    blast radius: a bug that fails rungs across ≥2 baselines outranks a single-baseline one — fix the shared cause
    once, then narrow to per-baseline residuals. Never chase an aggregate-curve rung while a logic rung is red.
15. **Logical determinism is the parity definition.** For a matched scope the sim must take the SAME sequence of
    steps in the SAME order as real — same trainers selected, same order of update receipt, same aggregations and
    rollbacks — differing ONLY in wall-clock. Prove this on the **first data bin** (`--max-data-id 1`: multiple
    iterations + variance passes + aggregations, short + diffable) before extending length. A matched, ordered
    trace is the proof; a matched summary statistic is not.
16. **Do the right thing — no hacks.** Ask as many conceptual questions as it takes to understand the real↔sim
    divergence, but solve it at the root. A hack that moves a number without a correct mechanism is a regression in
    disguise — it will not close parity and it will mask the real bug. When unsure, stop and ask.

**Open design decisions:** D2 (avail telemetry port — Phase 2); D3 (variance-cadence × withheld/late grads —
Phase 2); D4 (eval-delay factor — confirmed ~1× train cost, K-D3). D1/D6 resolved (§K).

---

## §G  Fixes landed (what worked — ≤20-word problem + ≤20-word fix; do not redo)
- **#12c sync `sim_rate` (delay-factor).** No delay-headroom starved the vclock. Fix: `--delay-factor 1` (full
  registry D) → sct = send + max(gpu,D) charges the skipped device wall → sync `sim_rate` 0.94→2.9–3.0.
- **K-D31 validated (P2-7a).** Databin1 checks: sync `cohort_sequence` ok=true, set/order/var/cadence=1.0 — the
  benign delay-tie is canonicalized away; bin-1 cohort order is bit-exact.
- **Pinning visibility + aggregator GPU pin (K-D33).** Trainer emitted no actual CPU/GPU; aggregator eval defaulted
  to GPU 0 (contended with trainers 1/9). Fix: trainer `[PIN]` self-report (`trainer/main.py`), `[LOAD_BALANCE]`
  post-proc check (`spawner.spawn_all` — warns on imbalance / under-provisioning), aggregator `gpu_id` pin (least-
  loaded/idle GPU). 120 launch tests green; async_cifar10 shares the launcher, unaffected.
- **#13 drain stall (K-D28/b/c).** fwdllm's `_sim_recv_min_grad` left a stuck straggler in the expected set →
  re-fired the full 30s `RECV_TIMEOUT` every cycle → pipeline starved (`sim_rate` 0.06). Fix: felix-port
  stuck-end eviction + recv-grace 2→5s + probe-ceiling/ready-gating + `drain_ready` direct ingest → `sim_rate`
  0.06→0.30, 30s stall gone, R1=0. (Step-4 staggering NEUTRAL, OFF — §H; residual is #12c-bound.)
- **#1c R1 two-ledger bridge (K-D27/b).** async_oort gated selection ONLY on `all_selected` (physical-event-pruned)
  and never on the aggregator's virtual in-flight set → slow-sim returned-but-uncommitted trainer re-dispatched
  (R1 62.9%). Fix: agg maintains `_sim_pending_commit` felix-style, binds it live to `sel._agg_pending_commit_ref`,
  async_oort's `filtered_ends` excludes it; `outstanding = inflight ∪ buffer` (K-D27b, NOT `− _sim_committed`).
  Sim-only → async_cifar10 byte-identical. `SIM_R1_DISPATCH` 238→0, confirmed on the latest full run.
- **#14 MQTT join-notify startup race.** `backend/mqtt.py` `join()` fired `notify(JOIN)` fire-and-forget; when the
  MainThread beat async `on_connect`, the JOIN + subscriptions dropped with no retry → end never registered →
  async_oort hung below `minInitialTrainers`. Fix: `join()` `_wait_for_connect()` (10s bound) before subscribe+notify;
  no-op when connected → all examples unchanged.
- **#6 clock-rate anchor + Root B phase rungs (K-D25).** The clock-rate rungs compared sim-vclock against real's
  FULL wall, which carries a ~7.7s/round localhost transport ARTIFACT — a checker-anchor bug, the sim vclock was
  correct. Fix: agg emits `intrinsic_span_s` (barrier+eval, fedavg excluded); rungs + `wall_disparity` anchor real
  on its intrinsic clock (async byte-identical). Validated throughput 0.47→0.048, wall_disparity 90→1.6.
- **Phase-4 stopping rule (K-D24).** `sim_wall_ceiling_s` was truncating a real-compute sim. Fix: decoupled to
  `max_runtime_s × 20`; matched `--max-data-id` is the primary stop. Fixed root S1.
- **Residence: commit-then-carry + felix realign (K-D12/K-D14/K-D17b).** Boundary-drop stranded ~7 async
  grads/cycle → 2× passes; slot-on-return undercounted in_flight 3×. Fix: carry the surplus, hold the compute slot
  to COMMIT (so `len(selected_ends)` = virtual-time in-flight), source R1 from an immutable echoed interval.
- **Clock family re-keyed to `data_id` (#2).** Rungs keyed on `round` mismeasured fwdllm's variance-pass progress.
  Fix: `_progress_axis`/`_per_progress_last_event`; async_cifar10 auto-detects `round` → byte-identical.
- **Foundational sim mechanisms (Batch 1–2, K-D2/K-D3/K-D4/K-D5/K-D9/K-D11).** Additive `sct = send + gpu + D`,
  no-sleep sim path; `_sim_recv_min_grad` sct reorder buffer + in-flight gate + agg-goal rollback cleanup; sync
  `_sync_sim_recv_first_k` first-k-smallest (real-only commit-1-per-pass clamp); V1–V5/DK1–DK3/G1–G2 rungs + the
  pre-mutation cycle snapshot that makes V1 exact.
- **Scaffolding (landed, pointer only).** Pre-run instrumentation A–E (K-D21); availability params end-to-end,
  print==run (K-D22); Phase-2 real-only speedup-leak skips (K-D23); `staleness_policy` wired from config (K-D13/15).

---

## §H  Dead-ends & corrections — do NOT retry
- **fluxtune #15 — three superseded framings (all 2026-07-06, same session; final root = COMMIT-PATH STALL).**
  (1) "pure GPU-PIPELINING loss; keep the gate, decouple dispatch" — WRONG, felix's arrival gate is INERT
  (gate_holds=0), not the culprit. (2) "the fix is re-dispatch on physical RETURN to keep GPUs busy" — WRONG, fedbuff
  never re-hands a returner the same version, and real's 3.37 concurrency is a duty cycle `gpu/max(gpu,D)`, not
  under-use. (3) "hold-to-commit is a SYNC barrier / over-restrictive; replace with model-advance re-dispatch" — WRONG,
  hold-to-commit is a CORRECTNESS check (a trainer is freed only when its update commits; guards no-same-version /
  not-while-computing / not-while-returned-uncommitted). *Final root:* the commit path STALLS (the drain gate blocks
  real wall on PHANTOM `_sim_inflight_expected` entries), keeping correctly-held trainers idle. Fix = fast/non-stalling
  commit path; hold-to-commit untouched. *Lesson:* commit RATE is the throughput lever, not the residence rule. See
  PARITY_LOGICAL_TASKS.md FELIX GROUNDING F5/F6.
- **"The sync cadence break is a sim ORDER bug (order → split-half var → RNG desync) — exact cadence parity is
  achievable once order matches."** CORRECT for bin ≤1, REFUTED for the full run (2026-07-05). K-D31 made
  receive-ORDER 41/41 identical, yet fwdllm cadence STILL breaks at bin 7. Root is grad NON-reproducibility given
  matched order: ~1e-3 GPU fp16 jitter, amplified by the split-half variance ratio, flips the `var<0.3` gate at a
  sensitive bin. *Lesson:* exact `var` is only a valid target ≤bin 1; beyond it the target is DISTRIBUTIONAL. Don't
  chase exact cadence past the nondeterminism wall (principle #16) — it reds on float noise, not a bug.
- **"The fluxtune runs are GPU under-provisioned (2–4 of 8 GPUs, 5 trainers/GPU) → contention is the root."** WRONG —
  a misread of `gpu=4.9s` (GPU compute SECONDS in the delay log) as device IDs. The spawn table is authoritative:
  **8 GPUs, balanced round-robin** (`spawner.py:305` `(tid-1)%num_gpus`), CPU `sched_setaffinity` 1 core/trainer,
  threads capped. Only structural imbalance is 10 trainers > 8 GPUs (GPU 0,1 carry 2 each). Compute IS mode-invariant
  at the floor (min 2.4s both modes). *Lesson:* verify a "device id" is a device id; confirm pinning from the spawn
  table, not a grep of timing logs.
- **"#13 step 4 (freed-slot staggered re-dispatch) closes the residual 11-12s drain holds."** NEUTRAL/REFUTED
  (K-D28d): `sim_rate` 0.289→0.274, holds persisted. The holds are the drain correctly waiting (strict sct order)
  for the earliest-sct in-flight straggler while higher-sct grads buffer — INHERENT to real-GPU + strict-order +
  trickle dispatch, not a dispatch-bunching artifact. Feeding an OLD freed-slot vclock as `send_ts` makes the gate
  expect trainers even earlier → more holds. *Lesson:* the residual after steps 1-3 is **#12c** (no delay-headroom),
  NOT the drain. Code kept flag-gated OFF; a reworked attempt would stamp the ACTUAL dispatch vclock.
- **"K-D19 fixed fluxtune R1 / the NONE reset is a red herring."** REFUTED (K-D26). R1 stayed 60.4%: K-D19 fixed
  the wrong release path and dismissed the NONE hypothesis by checking the *selection filter*, but `all_selected`
  MEMBERSHIP was deleted upstream (90s wall-timeout + async_oort reading `KEY_END_STATE=NONE` as "left").
  *Lessons:* (i) an R1 fix isn't done until the smoke banks R1≤2%; (ii) when a guard "should exclude" but doesn't,
  check whether the member is *deleted* upstream, not just whether the filter reads it.
- **"Free the compute slot on physical RETURN" (K-D16 Option-A).** WRONG for virtual time — a returned-but-
  uncommitted trainer is still in flight until the vclock reaches its sct; freeing undercounted `in_flight` 3×.
  Reverted to hold-to-COMMIT (K-D17b). (The K-D16 re-run also DEADLOCKED, K-D17: drain gated on channel RECV not
  the buffer; triplet stamped at DISPATCH froze the pool pre-commit.)
- **"Drop stranded grads at the agg-goal boundary" (K-D6).** REVERSED for async (K-D12) — the "|selected|≈agg_goal"
  premise holds only for sync; fluxtune (c=10≫agg_goal=3) dropped ~7/cycle → 2× passes. Drop stays correct for sync.
- **Adding a scalar overhead to the vclock to close #6.** Forbidden (principle #1/#12) — the vclock is
  `max(vclock, sct)`. Fold only genuine unmodeled compute (eval_s, straggler spread); artifacts stay off. #6 was a
  checker-anchor bug (K-D25), not a missing fold. (Including FedAvg in `intrinsic_span_s` over-charged it → excluded.)
- **Tuning `var_threshold` / `max_iterations_per_data_id` to close a cadence gap.** Rejected — baseline-defining
  knobs, not parity levers. A cadence gap is an upstream set/order/clock divergence.
- **"sim wall ≈ real, comparable" / "D=0 smoke ⇒ clock fails are artifacts".** WRONG — always check avg in-flight
  concurrency (not just pass counts); the runs are D>0 and the fails traced to `round`-vs-`data_id` keying (#2).

---

## §K  Deviation log — one line per decision (anchor + rationale; referenced from §B.1/§G)
- **K-D1** — `time_mode` default `"real"` (not cifar's `"simulated"`): fwdllm's whole corpus is `real`; a
  `simulated` default risks half-activating an unbuilt path.
- **K-D2** — additive `sct = send + gpu + D` (not `max`): fwdllm's real mode sleeps D on top of GPU time.
- **K-D3** — per-eval sct collapses to the train sct (D4): eval lives on the aggregator; forward-grad "train" is a
  forward pass (no 20× factor); the trainer eval message is a utility report, not a clocked commit.
- **K-D4** — purpose-built `_sim_recv_min_grad` (not `_sim_recv_min` verbatim): the latter's per-commit release +
  withheld paths key on WEIGHTS semantics; reuse the primitives, fork the orchestration.
- **K-D5** — slot release + buffer clear on the AGG-GOAL boundary (not per-commit): a `data_id` spans many cycles
  with rollbacks; per-commit release would strand a re-contributing trainer (principle #4).
- **K-D9** — cadence telemetry is a **pre-mutation** cycle snapshot: post-mutation `data_id` advances before emit
  → off-by-one; snapshot makes V1 exact. Existing post-mutation fields unchanged (byte-identical real).
- **K-D11** — `ends_not_selected_yet` "commit-1-per-pass" clamp gated real-only: a real-transport draining
  discipline; the sim barrier is single-pass (principle #8).
- **K-D12** — fluxtune async: commit-then-CARRY the surplus + hold residence (reverses K-D6); drop stays correct
  for sync (c≈agg_goal).
- **K-D13/K-D15** — fluxtune `staleness_policy = fedbuff` staleness-weighted accept (was silently `none`); set
  identically real+sim (a definition, not a lever).
- **K-D14** — R1/W1 sourced from an ECHOED per-contribution interval (not the agg's per-end dispatch stamp, which
  is overwritten on re-dispatch — exactly when residence is broken).
- **K-D17b** — hold the compute slot to COMMIT (felix port): `len(selected_ends)` = virtual-time in-flight
  (fixed in_flight 2.7→9.5). Reverts K-D16's slot-on-return. **CONFIRMED CORRECT for both sync AND async (K-D34):** a
  trainer is freed only when its update commits — a correctness check, not a throughput lever. fluxtune's `sim_rate<1`
  is a commit-path STALL (#15), NOT this rule; it stays untouched.
- **K-D21** — pre-run instrumentation A–E landed & pytest-green; un-skipped the 12 rigor-gap rungs (§G).
- **K-D22** — availability params respected end-to-end; Phase-1 syn_0 default; print==run (§G).
- **K-D24** — Phase-4 ceiling decouple (×20) + B1/B2 sct folds; fixed root S1 (§G).
- **K-D25** — #6 is a CHECKER-ANCHOR bug, not a sim under-charge: agg emits `intrinsic_span_s` (barrier+eval,
  fedavg excluded), rungs anchor real on its intrinsic clock; Root B stamps `post_train_s` after the delay and
  moves the B2 straggler into the sct only. Async byte-identical (§G).
- **K-D26** — #1c R1 root-caused to a physical-wall vs vclock desync in the shared `async_oort` re-pick guard
  (`all_selected`), exposed by the slow sim. Fix (a): `_abandon_clock_now()` runs the 90s timeout on the vclock in
  sim (None in real → byte-identical). Corrected K-D19. Sustained NONE-delete path → deeper root K-D27.
- **K-D27/b** — the two-ledger split: async_oort's selection eligibility never consulted the aggregator's virtual
  in-flight truth. Fix: agg maintains `_sim_pending_commit` felix-style + binds `sel._agg_pending_commit_ref`;
  `filtered_ends` excludes it; `outstanding = inflight ∪ buffer` (K-D27b — the `− _sim_committed` subtraction
  dropped re-dispatched-after-commit trainers, R1 stuck 67.6%). Sim-only. `SIM_R1_DISPATCH` 238→0 (§G).
- **K-D28/b/c** — #13 drain-stall felix port (3 steps landed): stuck-end eviction + recv-grace floor 2→5s;
  probe-ceiling + ready-gating; direct `drain_ready` ingest (flag `sim_sct_ordered_drain`). `sim_rate` 0.06→0.30,
  30s stall gone, R1=0, 0 recv_fifo timeouts (§G).
- **K-D28d** — #13 step-4 freed-slot staggered re-dispatch: IMPLEMENTED but NEUTRAL → flag DISABLED. The residual
  holds are inherent strict-sct-order straggler waits, not a dispatch artifact; residual `sim_rate` is #12c-bound
  (§H).
- **K-D29** — **REMAINDER-WAIT delay model (reverses K-D2's additive decision).** The order→split-half-var→RNG-
  desync root (§A) requires the sim's commit order to equal real's arrival order. That needs a deterministic
  per-trainer arrival order, which the flat-additive `gpu+D` model didn't give (order was GPU-jitter-dominated).
  Fix (aligned with async_cifar10): real sleeps `max(0, D−gpu)` so the device wall = D (GPU hidden inside); sct =
  `send + max(gpu, D)`; per-trainer registry delays supply the completion spread → order = D-order = deterministic
  & real↔sim identical. Overrun (gpu>D) is flagged (`training_overran`, `[TIMING_OVERRUN]`) — it un-determinises
  order, so it's the fluxtune watch (its JVP GPU 7.57s overruns D/2). Crc32 straggler offset disabled
  (`sim_straggler_spread_s=0`); `perturbation_count` made a knob (default 10). See PARITY_LOGICAL_TASKS.md P2.

- **K-D30** — **full-cohort determinism gate + `timing_overrun` signal (P1-5, closes the P1-4/P1-5 gap).** The
  selection set/sequence rungs (`selection`/`aggregation_sequence`/`utility`) gated to a TRIVIAL pass for every
  stochastic selector, so fwdllm's syn_0 selection was never actually checked. Fix: `_selection_is_deterministic`
  un-gates when `num_chosen==num_candidates` in BOTH modes (K≥pool → set-deterministic) — data-driven, so it
  enforces fwdllm (K=all), keeps fluxtune (agg_goal=3) + fwdllm_plus (#7 asymmetric eligible) gated, self-disables
  under Phase-2 scarcity, and falls back to the old selector-name rule on count-less legacy telemetry (no
  regression). `participation` EXCLUDED (round-keyed → mechanical KS on fwdllm's constant-`round`/`data_id` axis;
  cohort_sequence is its per-cycle enforcement). New `timing_overrun` DIAG rung surfaces the K-D29 overrun tell
  (P2-6 `training_overran` fraction + first-overrun bin → gpu>D flips order → cohort/var break is a timing-model
  limit not a sim bug). P1-4 (async_cifar10 cohort adapter) assessed redundant, not built. Sim/checker-only →
  async_cifar10 byte-identical; banked scoreboard stable.

- **K-D31** — **canonical `(D, trainer_id)` cohort commit order (P2-7a; operator chose canonicalize-by-id).** The
  databin1 run left ONE sync residual: two trainers sharing a registry delay (D=6.5) swap in receive order (real
  breaks the sct tie by physical arrival, sim by sct-sort) — benign (both in the same split-half → `var`/grads
  bit-identical) but the EXACT-order `cohort_sequence` rung flags it. Fix: trainer stamps pure `D`
  (`MODELED_DELAY_S`, both modes, deterministic from the registry — unlike sct which folds in GPU jitter); agg
  sorts each cohort by `(D, str(end))` in `_canonicalize_cohort_commit_order` before the telemetry snapshot +
  `aggregate()`, reordering `_per_agg_trainer_list` + the trailing cohort slice of the grad/jvp lists in lockstep.
  No-op when delays off / already canonical (real still receives in strict D-order, K-D29 — only ties move).
  Sim/agg-only → async_cifar10 byte-identical; 428 mode + 9 canon + 115 parity green.

- **K-D32** — **fluxtune JVP perf optimizations (`jvp_perf_opt`, §L; fluxtune-only, config-gated, all
  BIT-IDENTICAL).** fluxtune's forward-grad JVP was ~10× the sync compute (2P=20 selection passes + 5 removable),
  overrunning its modeled mobile delay → out-of-order commits (#1d). Fix (validated by
  `scripts/profile_jvp_opt.py`): (a) trainable-only finite difference — `calculate_jvp(trainable_idx=…)` skips
  `p−h·0=p` on the 98.5% frozen backbone (1.26× + −251 MB); (b) skip the 3 diagnostic-only forward passes; (c)
  reuse the selected perturbation's cached JVP. Combined fluxtune −37% (sync would be −68% but is left OFF).
  Gated on `jvp_perf_opt` (trainer_base default false = byte-identical; true in both fluxtune yamls, must match
  real↔sim; revertible per-config). Startup `[JVP_PERF_OPT]` log confirms 10/10 trainers active. NOT adopted: vmap
  (2× but fp32-diverges via FD cancellation), fwd-AD (slower). 13 pytests + 185 fwdllm mode + 115 parity green.

- **K-D33** — **pinning self-report + load-balance check + aggregator GPU pin.** Trainers now emit a `[PIN]` line
  (actual `CUDA_VISIBLE_DEVICES`/cuda device+name/`cpu_affinity`) at startup (`trainer/main.py` — NOT the dead
  `fl_main.py`); `spawner.spawn_all` emits `[LOAD_BALANCE]` (WARN on GPU imbalance, `num_gpus<visible`
  under-provisioning, or CPU imbalance); `aggregator_spawner.spawn(gpu_id=…)` + `runner.py` pin the aggregator to a
  dedicated (idle if `visible>num_gpus`, else least-loaded) GPU so its eval stops contending GPU 0. Confirmed the
  `client_idx%8` device arg is vestigial (`FedSgdTrainer:388` overwrites `self.device=torch.device("cuda")`=cuda:0 of
  the CVD-masked view) → the spawner's pin is authoritative. Shared launcher; 120 launch tests green.

- **K-D34** — **fluxtune #15 = a COMMIT-PATH STALL; `sim_compute_truthful_gate` fix LANDED (P1/P2), P3 pending
  (2026-07-06).** Supersedes three same-session mis-framings (§H): pipelining-loss, re-dispatch-on-return, and
  "hold-to-commit is over-restrictive." Grounding (PARITY_LOGICAL_TASKS.md FELIX GROUNDING F5/F6): hold-to-commit
  correctly frees a trainer only when its update COMMITS (guards: no same-version dispatch, none while computing, none
  while returned-but-uncommitted), so the commit RATE is the throughput lever. felix's gate is INERT and cifar's
  commit is instant (~0.4s) so held trainers barely idle. **D1/D2 CONFIRMED** (banked pair): fluxtune's
  `_sim_recv_min_grad` `earlier_stuck` gate holds already-arrived grads (`buf_depth=7`, 99.7%) to wait on trainers
  stamped-expected-at-DISPATCH but idle-in-recv (not computing) behind the single-threaded drain → 82% of wall
  (~1974s) burned → concurrency sim 1.65 vs real 7.69 → sim_rate 0.50. **Fix (flag `sim_compute_truthful_gate`,
  default off = byte-identical, fluxtune-yaml on):** stamp `_sim_dispatch_wall[end]` at the real `channel.send`; the
  gate skips any expected entry whose last dispatch is older than `sim_gate_compute_cap_s` (default 10s) — a
  stamped-but-idle phantom no longer blocks a ready commit, while a genuine in-window straggler is still held (commit
  order preserved). Hold-to-commit, the sct-ordered drain, K-D12, K-D27 UNTOUCHED; `fwdllm_aggregator`-only →
  async_cifar10 byte-identical. 25 pytests green. **P3 RAN 2026-07-06 (`run_20260706_112114` sim / `_110555` real):**
  phantom fix **VALIDATED** — `STUCK_EVICT=0`, `phantom_skip`→65 (~1/commit, load-bearing), 30s failsafes gone (max gap
  12.8s). **But `sim_rate` still <1** (steady ~0.49; per-commit `Δvclock/Δwall=0.465`) — the phantom stall was necessary
  but NOT the sim_rate ceiling. **Residual root (NEW):** (a) vclock omits `aggregate()` variance-compute wall (sim 32s /
  real 29s, symmetric, uncredited — `:1783` folds eval only); (b) GPU≈D no-headroom (#1d) → sim blocks in `drain_ready`
  on the real GPU pass. Sim wall (145s) > real (91s). **PAUSED** for operator real-experiment work; resume plan (fold
  non-overlapped aggregate into vclock + GPU-vs-D headroom + longer flag-on run & flag-off A/B) at PARITY_LOGICAL_TASKS.md
  top. jvp_perf_opt verified symmetric (trainers True, aggregator-eval False, both modes) — NOT a mismatch.

*Retired/superseded anchors (kept only as pointers): K-D6 (→K-D12), K-D7/K-D8/K-D10/K-D16/K-D18/K-D19/K-D20/K-D23
— landed scaffolding or corrections, folded into §G/§H; see git history for detail.*

---

## §L  Forward-grad JVP compute profile & retained fluxtune optimizations
*(2026-07-05, tool: `scripts/profile_jvp_opt.py` — reuses the real `create_model` + `calculate_jvp`; distilbert-base
+ AdapterHub adapters, batch 8, seq 192, A40, fp16. Absolute ms are a CLEAN single-trainer profile; the real run
multiplies by ~10× from GPU contention across the 10 concurrent trainers, but pass-counts/ratios/memory transfer.)*

**Mechanism.** Forward-grad trains via a **central finite-difference JVP** (`fwdgrad_utils.calculate_jvp`): each
perturbation = **2 forward passes** `f(θ±hv)`, h=0.01, autocast+no_grad → `jvp=(f(θ+hv)−f(θ−hv))/2h`. **fluxtune**
SELECTS the best of `perturbation_count`(=10) perturbations by |jvp| (2P=**20 passes**); **fwdllm/sync** selects by
cos-sim (**0 forward passes**) + 1 final JVP. Only **~1.5% of params are trainable** (bottleneck adapters in all 6
layers + head, 1.04M/67.4M); the backbone is frozen.

| path | fwd passes | ms/batch (clean) |
|---|---|---|
| sync fwdllm (current) | 5 | 50 |
| **sync fwdllm (opt)** | 2 | **16 (−68%)** |
| fluxtune P=1 (opt) | 2 | 16 (== sync) |
| fluxtune P=5 (opt) | 10 | 80 |
| fluxtune P=10 (current) | 25 | 251 |
| **fluxtune P=10 (opt)** | 20 | **159 (−37%)** |
| backprop ref (1 fwd+1 bwd) | — | 17 |

- **Compute vs sync:** fluxtune = `2P × per-pass` → **10× sync at P=10**, linear in P, **equals sync at P=1**. The
  JVP-selection is the entire fluxtune surcharge; sync's cos-sim selection is free.
- **Memory:** forward-grad peak is **FLAT in P** (~3.2–3.4 GB = model + one held forward; **no autograd graph**).
  Backprop stores activations (3.71 GB). fluxtune's extra JVP inferences cost **TIME, not memory** — same footprint
  as sync (this is forward-grad's design tradeoff: many cheap forward passes, no backward, low memory).
- **Per-pass:** full-param FD 10.0 ms; trainable-only FD 7.95 ms.

**RETAINED — bit-identical (fidelity-preserving; real↔sim parity untouched):**
1. **Trainable-only FD** — skip `p−h·0=p` on the 98.5% frozen params inside `calculate_jvp`: **1.26× + −251 MB**,
   `max|Δjvp|=0`.
2. **Drop the 3 diagnostic-only forward passes** (`_train_one_batch:646-648`, loss before/after-update logging —
   never feed grads/telemetry) **+ reuse the winner's cached JVP** (`:645`, fluxtune): fluxtune 25→20, sync 5→2.

Combined: **sync −68%, fluxtune −37%**, all bit-identical → should clear the `delay_factor=1` overrun
(4.2s → ~2.6s < min cohort D 4.0s) WITHOUT touching fidelity. **LANDED (K-D32), fluxtune-only & config-gated**
(`jvp_perf_opt`, default false = byte-identical; true in both fluxtune yamls; sync untouched). Startup log
`[JVP_PERF_OPT]`; 13 new pytests + 185 fwdllm mode + 115 parity green.

**NOT retained — change fidelity:**
- **vmap-batching the perturbations** — **2.0×** (biggest single win) and mathematically exact (fp64 seq==vmap
  BIT-IDENTICAL, deterministic run-to-run → *would* be real↔sim safe) BUT differs ~5% from the current sequential in
  fp16/fp32: the FD subtracts two O(1) losses (catastrophic cancellation floors precision), so any reduction-order
  change re-baselines the trajectory. Excluded per the fidelity bar; available if a re-baseline is ever accepted.
- **Forward-mode AD** (exact JVP) — slower (0.5×, needs eager attention; not impl for SDPA) + different math.
- **`perturbation_count`↓** — the direct lever, but changes the baseline algorithm.
