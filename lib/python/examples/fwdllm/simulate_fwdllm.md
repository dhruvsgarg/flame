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

**All three baselines now run end-to-end at the 2700s budget — the startup race (#14), the R1 re-dispatch churn
(#1c) and the 30s drain stall (#13) are all FIXED and validated. Two standing fronts remain: (1) LOGICAL PARITY —
prove the sim takes the same steps in the same order as real on a matched scope (first data bin); (2) `sim_rate < 1`
— the sim burns more wall than the virtual time it models, so the sim job is not faster than the real job.**

Latest pairs (`run_sequential.sh --mode both --delays on --max-runtime-s 2700`, `run_..._041051 → 080931`):

| baseline | real (data_id in ~2710s wall) | sim `sim_rate` | sim wall to real's data_id | R1 | verdict |
|---|---|---|---|---|---|
| **fwdllm** (sync) | 69 | **0.935** | 1345s (real 2707s → **sim ~2× faster/progress**) | 0% (sync) | runs clean; `sim_rate` just under 1 |
| **fwdllm_plus** (sync) | 18 | **0.948** | 346s (real 2719s → **sim ~8× faster/progress**) | 0% (sync) | runs clean; `sim_rate` just under 1 |
| **fluxtune** (async) | 69 | **0.303** ⛔ | 6699s (real 2704s → **sim ~2.5× slower**) | **0** ✓ (`SIM_R1_DISPATCH=0`, evict=1, 0 recv_fifo timeouts) | drain + R1 fixed; acute `sim_rate` case **+ a genuine logical divergence (#1d, async cohort order)** |

**Why `sim_rate < 1` (the #12c root, not the drain).** With no `--delay-factor`, `training_delay_factor=None` →
the hardcoded ÷10 → modeled delay D≈0.4–1.8s. So each commit charges ~only the real GPU pass (which the sim
genuinely runs) to the vclock — there is almost no skipped transport wait to compress → `sim_rate` ceilings just
below 1 for the sync baselines, and fluxtune's async strict-sct-order straggler holds drag it to 0.30. Per
principle #13 the sim MUST deliver speedup (`sim_rate ≥ 1`); it does not yet.

**The `sim_rate` lever — `--delay-factor` (#12c), applied within the first-data-bin scope below.** Re-run with the
full registry delay so `sct = send + gpu + D`, D≈4–18s → the vclock jumps by gpu+D per commit while wall pays only
the skipped GPU → expect `sim_rate` toward/past 1. Pair it with the #15 hard-wall-pause audit (the sim must not
sleep real seconds it should fast-forward). Sync baselines are already near-parity in rate; the same should push
them over 1. (Overall next step is first-data-bin logical parity — see STRATEGY / Next roots.)

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
| **fwdllm** | 194/194 identical (trivial: K=10=all) | 22/194 | **data_id 7** — breaks at **8** | **BREAKS @ bin 8** — sim inserts +1 iteration at data_id 7 |
| **fwdllm_plus** | 48/48 identical | **48/48 identical** | **data_id 18 (real's max)** — no break | **PARITY ✓ to bin 18** (extend when a longer real run exists) |
| **fluxtune** | **0/17 identical** | 7/17 | **breaks at bin 0** (cohort #1) | **DIVERGES from the first aggregation** — async cohort order |

**ROOT (2026-07-05 code+telemetry investigation — supersedes the earlier "order is benign" reading):** all three
break from ONE cause — **the sim's commit/receive ORDER ≠ real's actual arrival ORDER.** fwdllm's variance is a
**split-half statistic over the commit-ORDERED grad list** (`fwdgrad_utils.py:133-158`), so order matters even when
the cohort SET is identical; the sim commits in sct order, real in physical-arrival order → different `var` → the
`var<0.3` gate flips at a different iteration → and since each trainer's `torch.Generator` is seeded once and
**never reset**, one extra iteration desyncs every trainer's RNG → grads diverge ~1% thereafter (the fwdllm bin-8
break). For fluxtune (agg_goal=3<K) the order picks *which 3* commit → cohort wrong from aggregation #1. The order
isn't reproducible because the timing model is mis-set (D≈0.4s flat ≪ GPU 1–1.7s → order is GPU-jitter-dominated,
not delay-deterministic; fwdllm also uses a flat-additive delay, not the remainder-wait model). **Grads are
deterministic GIVEN matched order** (batch/seed mode-invariant) → this is a sim ORDER bug, NOT nondeterminism →
exact cadence parity IS achievable once order matches. **Full diagnosis + fix plan: [PARITY_LOGICAL_TASKS.md](PARITY_LOGICAL_TASKS.md).**

> **STATUS (2026-07-05 checkpoint):** the checker now GATES on this (enforced `cohort_sequence` EXACT rung +
> V2 mean-guard, P1) and the timing-model fix has LANDED (**K-D29** remainder-wait `max(gpu,D)` + per-trainer D +
> crc32 straggler off + `perturbation_count` knob). **P1-4/P1-5 now RESOLVED** (K-D30): full-cohort determinism
> gate un-gates fwdllm's selection/aggregation_sequence/utility to GENUINE enforcement (fluxtune/fwdllm_plus stay
> gated); new `timing_overrun` DIAG rung surfaces the K-D29 overrun tell; P1-4 assessed redundant. 419 mode + 115
> async_cifar10 parity tests green; banked scoreboard stable. **Not yet run.** Pending before close: the
> **databin1 validation run** (`--delay-factor 2 --max-data-id 1`). Resume steps: the tracker's SESSION CHECKPOINT.

### Open issues (OPEN only — closed items live in §G/§H)
| # | issue | baseline(s) | next step |
|---|---|---|---|
| **#1d** ⭐⭐ | **fluxtune LOGICAL divergence (not timing): async cohort composition differs every aggregation (0/17 receive-sets match, bin≤1).** The sim's sct-sorted receive order does not reproduce real's actual first-`agg_goal` arrival order → var<0.3 crossing shifts → sim needs +2 aggregations to clear bin 0 (V1 genuinely off). CORRECTNESS bug — fix BEFORE fluxtune's time dimension (principle #14/#16). | fluxtune | Diff per-trainer real-GPU time vs sim-sct order — does the sct use ACTUAL measured GPU or a nominal speed? First resolve the conceptual Q: is real's receive-order deterministic or wall-jittery (§A logical-parity check)? |
| **#12c** ⭐ | `sim_rate < 1` — no delay-headroom to compress. `--delays on` uses the hardcoded ÷10 shrink (D≈0.4–1.8s) unless `--delay-factor` is passed, so the vclock charges ~only GPU. | all (acute: fluxtune) | Re-run with `--delay-factor` (sct = send + gpu + D, D≈4–18s) → `sim_rate` toward/past 1. Then re-run the checker. Config-flow otherwise correct (K-D24/K-D21). |
| **#15** ⭐ | **Hard real-wall pauses in the sim.** fluxtune sim wall 8879s ≫ vclock 2694s (3× overhead) — the vclock must advance THROUGH the modeled wait after GPU compute (fast-forward), never make the sim PROCESS sleep real seconds. Suspect: drain holds waiting real-wall for the slowest straggler's GPU + any non-parallel GPU dispatch. | all (acute: fluxtune) | Audit every place the sim blocks real-wall (drain grace, recv_fifo, GPU dispatch serialization). Confirm the 10 trainers' GPU runs in PARALLEL and the only irreducible real wall is one GPU pass; everything else must be a vclock jump. |
| **#7** | fwdllm_plus real ~4× slower/round (156 vs 41 s/round); at syn_0 real sees only ~4.9 eligible vs sim ~9.6. Not a sim bug. | fwdllm_plus | Profile per-iteration reselection + oracular-read cost from the banked per-phase log; explain the eligible-count gap at 100% avail. |
| **#11** | real-mode critical-path waste (`sleep(0.1)` MQTT-settle busy-waits; one-grad-per-poll drain tail) — real-only. | fwdllm, fwdllm_plus (real) | **Deferred to a validated pass** — ZERO parity impact (sim already skips them); removing them changes the working real reference + needs a real run (principle #8/#11c). Never change grad values/cadence. |

### Next roots — ranked (correctness before time; SHARED before per-baseline — principle #14)
0. **DONE — first-data-bin logical-parity check ran.** fwdllm + fwdllm_plus are **logically clean on bin 1**
   (cadence + cohorts identical) → their whole-run cadence fails are length-confound, and they are **cleared to
   chase the time dimension**. fluxtune has a **genuine logical divergence (#1d)**.
1. **#1d fluxtune async cohort/receive-order (correctness, per-baseline) — TOP.** The one real logic bug. Root-cause
   the sct receive-order vs real arrival order; resolve the deterministic-vs-jittery conceptual question first.
2. **#12c `--delay-factor` + #15 hard-wall-pause audit (time, SHARED).** Give the vclock its delay-headroom AND
   stop the sim burning real wall it should skip → `sim_rate ≥ 1`. These are the `throughput`/`overhead`/`per_round`
   fail family; safe to pursue for the two sync baselines now, and for fluxtune once #1d lands.
3. **#7 fwdllm_plus real speed / selection divergence (per-baseline).** `eligibility`/`selection_detail` — real
   completes now; explain the 4× slowness + the syn_0 eligible-count gap from telemetry. Not a sim bug.
4. Then C1/C2 convergence at matched `data_id` per baseline → gate to Phase 2 (unavailability).

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
| 6 | Slot residence | per-commit release | **hold slot to COMMIT** (felix-aligned) — dispatched-but-uncommitted trainer held in `selected_ends`+`all_selected`, released on commit | a returned-but-uncommitted grad is still in flight in VIRTUAL time (commits when vclock reaches sct) | K-D5/K-D17b, principle #4 |
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
- **K-D17b** — hold the compute slot to COMMIT (felix-aligned): `len(selected_ends)` = virtual-time in-flight
  (fixed in_flight 2.7→9.5). Reverts K-D16's slot-on-return.
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

*Retired/superseded anchors (kept only as pointers): K-D6 (→K-D12), K-D7/K-D8/K-D10/K-D16/K-D18/K-D19/K-D20/K-D23
— landed scaffolding or corrections, folded into §G/§H; see git history for detail.*
