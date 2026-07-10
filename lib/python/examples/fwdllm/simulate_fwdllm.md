# High-Fidelity Simulator for FwdLLM — Real↔Sim Parity

**Active build (branch `dg/fwdllm_sim_unavail`).** A simulated-clock runner for the `fwdllm` example
(FedFwd / forward-gradient FL) reaching **real↔sim parity** across the **fluxtune / fwdllm / fwdllm++**
baselines — at **100% availability (syn_0)** first (Phase 1), then **unavailability** (syn_20/50/mobiperf,
Phase 2), then **beyond syn_0** (Phase 3). fwdllm has no native sim clock; the build wires flame-core's virtual
clock + availability substrate into fwdllm's variance-gated gradient loop. It reuses async_cifar10's virtual
clock, sct reorder buffer, in-flight gate, availability substrate, and parity ladder where they transfer, and
deviates where the workload demands (fwdllm aggregates **gradients** not weights; **variance-gated dynamic-K**
commit cadence; **`data_id`** progress axis; one-message-per-call grad loop; rollback across agg-goal cycles).

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
> - **Date hygiene, every revisit.** Any session that edits this doc sweeps the WHOLE file for prior-dated
>   narrative (e.g. "as of 2026-07-05", "(2026-07-09)"): fold what's still load-bearing into crisp, date-free
>   prose (or a one-liner in §G/§H/§K), delete what's superseded, and tag anything left unchanged-but-still-valid
>   `[checked YYYY-MM-DD]` so the next sweep re-examines it fresh rather than trusting an old tag forever. Run
>   identifiers (`run_YYYYMMDD_HHMMSS_...`) and JSON report filenames are evidence anchors, not narrative dates —
>   always keep those.
> - **Conceptual survives, mechanical gets purged.** A design choice, root cause, refuted hypothesis, or
>   real↔sim nuance is worth a permanent line — it changes future judgment. Instrumentation/plotting/test/code
>   scaffolding descriptions do NOT survive past landing, even crisply worded — git + the code ARE that record.
>   Concretely: drop pytest/test-green counts, field-by-field telemetry additions, and "which files changed"
>   narration once a fix is landed; keep the one-line problem+fix (§G) or before→after numbers that prove the fix
>   actually worked. Only OPEN/pending items earn a date stamp and expanded detail — closed ones get the terse
>   §G/§H/§K treatment or nothing at all.

> **Working checklist for every fix in this doc (already the §F ruleset — indexed here, not restated):**
> (a) Ground every claim in a metric that is actually captured and diffable — telemetry/banked-logs FIRST (§F#11),
>   logical-determinism traces over aggregate curve-matching (§F#15).
> (b) Isolate the true bottleneck, not its symptom — classify real-transport artifact vs algorithmic property
>   (§F#8), rank by blast radius / SHARED-before-per-baseline (§F#14), confirm which side (real or sim) is
>   actually divergent before tuning either (§F#6).
> (c) Design fixes from first principles at the root — no hack that moves a number without a correct mechanism
>   (§F#16); baseline-defining knobs are not parity levers (§F#3).
> (d) **Never launch an experiment run (GPU/cluster job) directly.** Print the exact command (cwd + flags) and
>   let the operator run it. Code edits, telemetry-only reads of already-banked logs, and pytest are fine
>   unattended; a new `run_sequential.sh`/`run_parity.py` invocation is not.

**Prerequisites:** [async_cifar10/PARITY.md](../async_cifar10/PARITY.md) — parity methodology (ladder,
roles/tiers/gating, run-length budget, landed sim mechanisms); **fwdllm's rung catalog is PARITY.md §F**.
[async_cifar10/UNAVAILABILITY_DESIGN.md](../async_cifar10/UNAVAILABILITY_DESIGN.md) — the availability substrate
fwdllm inherits via its aggregator class chain (ClientAvailability mixin, trace-read effect path, two-ledger
discipline, starvation self-termination, A6/A7/A8/K11 ground-truth rungs).

---

## §A  Current status

> ## ⭐⭐ fluxtune slowdown ROOT = **sim selector blind to modeled delay D** (RC1). Supersedes #1d as the driver.
> **Fix LANDED** (`fwdllm_aggregator.py:1337`; not flag-gated — correctness). **Operator validation run PENDING** (see NEXT).
> The `--delay-divisor 0.5` headroom run (the old #1d fix) did NOT close parity: fresh pair (real
> `run_20260710_112701` / sim `run_20260710_113854`) scores **46/14/19**, sim vclock **1050s vs real 560s (~1.9×)**,
> `cohort_sequence set_match=0.042`.
>
> **RC1.** `:1337` set `PROP_CLIENT_TASK_TRAIN_DURATION` (async_oort's speed-penalty input) from
> `real_client_task_train_duration` (=`WALL_SEND−WALL_RECV`) in BOTH modes. Real sleeps its budget so this = D; sim
> doesn't sleep so it = raw GPU (~3-6s) for every trainer → Oort can't tell fast from slow → selection flattens →
> over-selects slow trainers → cohort diverges → ~1.9× slow. Sole aggregator missing the sim-branch (asyncfl:1013 /
> oort:912 / syncfl:430 have it). Evidence: real chooses fast trainers heavily (chosen 370=99, 373=62), sim starves
> them (370=13, 373=2) at identical 100% AVL_TRAIN; `selection_bias` shows real `observed_selected=18.56s` (=D) vs sim
> `3.84s` (=GPU). Why headroom alone didn't help: it makes commit order (sct=D) deterministic, but SELECTION never sees D.
>
> **Fluxtune fails (14).** DIRECT(RC1): `selection_bias` · `selection_detail` · `participation` · `cohort_sequence`.
> CASCADE: `v1_iter_per_data_id` · `staleness` · `g2_grad_pool_size` · `throughput` · `per_round_advance` ·
> `overhead_residual` · `terminal_state` · `total_commits` · `convergence`. Noise: `step_timing_breakdown` (small-N KS).
>
> **fwdllm / fwdllm_plus — NO new root** (53/7/21 & 51/7/21). RC1's telemetry symptom is present (sim `trainer_speed`
> 1.4s vs real 11.6s) but BENIGN — `random` ignores speed. Their fails are already-explained non-bugs (#N n=10 float
> jitter; fwdllm_plus eligible-count gap §G#7; length confound).
>
> **NEXT — operator validation run** (print, don't launch; real reference exists → `--mode sim`):
> `./run_sequential.sh --only fluxtune --mode sim --delays on --num-gpus 8 --delay-divisor 0.5 --max-runtime-s 1800 --max-data-id 10`
> then `python run_parity.py --yes --baselines fluxtune`. GATE: `selection_bias`/`participation`/`cohort_sequence`
> pass-or-improve · `sim_selected_mean`≈real (~9.3s) · v1 real≈sim (~9.5) · `sim_rate`>1. Then re-assess the
> `staleness` gap + `sim_model_agg_compute_time` fold (likely cascade). Keep `--delay-divisor 0.5` (holds commit order clean).

Latest banked pairs (`run_sequential.sh --mode both --delays on`, n=10 smoke config, `run_20260709_1525 → 1601` —
supersedes the older n=100 pairs, which predate K-D36's per-stage wall-budget rungs and several since-landed fixes):

| baseline | sim `sim_rate` | sim wall / real wall | verdict |
|---|---|---|---|
| **fwdllm** (sync) | **2.62** ✓ | 212s / 776s | healthy; 3.7× wall speedup |
| **fwdllm_plus** (sync) | **2.70** ✓ | 213s / 1746s | healthy; 8.2× wall speedup (real slow — §G eligible-count-gap root cause) |
| **fluxtune** (async) | **0.52** ⛔ | 997s / 356s | RC1 selector-blind-to-D (§A); fix landed, validation run pending. This row predates the fix + `--delay-divisor 0.5` |

### Parity scoreboard — REFERENCE baseline (checker run on the pairs above; `expt_scripts/run_parity.py --yes`)
*Current-truth: supersedes the prior n=100-pair counts — do NOT reference old counts. Several of that older
scoreboard's SHARED fails (`gpu_budget_real`/`gpu_budget_sim`, `utility`, `v5_variance_pass_ratio`,
`overhead_residual` for the two sync baselines) no longer reproduce on this pair — closed, see §G.*

| baseline | pass / fail / skip | JSON |
|---|---|---|
| **fwdllm/syn_0** | **53 / 7 / 21** | `experiments/_parity_reports/parity_fwdllm_syn_0_20260709_154002.json` |
| **fwdllm_plus/syn_0** | **51 / 7 / 21** | `parity_fwdllm_plus_syn_0_20260709_155814.json` |
| **fluxtune/syn_0** (pre-RC1-fix, `--delay-divisor 0.5`) | **46 / 14 / 19** | `parity_fluxtune_syn_0_20260710_113854.json` — `selection_bias`+`participation` now surface RC1 directly |

**Fails, categorized by blast radius (fix the SHARED roots first — principle #14).**
- **SHARED — all 3:** `per_round_advance` (mean within 1–7%, KS noisy at n=10 — small-N artifact, not a
  divergence); `step_timing_breakdown` (checker-bug portion FIXED K-D37; residual funcs are the same small-N KS
  noise, means agree within ~8%, matching the `phase_gpu_compute` precedent); `cohort_sequence` var VALUE now
  fails inside bin ≤1 too (`var_match_frac` 0.5/0.6/0.0) — for fwdllm/fwdllm_plus this is the bin-7 float-jitter
  wall generalizing DOWN to bin 1 at n=10 scale (both modes agree on `var_good`, only the raw float differs
  ~0.1–0.2% > the 1e-3 tolerance — K-D35's "HARD to bin 1" premise is n-scale-sensitive); for fluxtune the SET
  itself diverges (#1d, unchanged, now quantified by `timing_overrun`'s 11% figure).
- **fwdllm only:** `phase_gpu_compute` (marginal KS, means match — unchanged); `drain_wall_budget` — sim's
  `barrier_wait_s` (2.67s) exceeds real's (1.43s), a new, small (~1.2s/cycle) fwdllm-only overrun (fwdllm_plus/
  fluxtune don't show it — sim's barrier cost looks fixed while real's scales with baseline); `terminal_state` /
  `total_commits` (sim 10 vs real 9 rounds at matched virtual budget — length confound from the healthy
  `sim_rate`, not a bug).
- **fwdllm_plus only:** `eligibility` / `avail_timebase` / `selection_detail` (real-transport-artifact divergence,
  root-caused — §G "eligible-count gap"; real 5.3 vs sim 10.0 mean eligible, expected not a bug); `throughput`
  (marginal, `rel_diff=0.058` vs `tol=0.05` — same root); `cohort_sequence` order/var (same n10 float-jitter class
  as fwdllm).
- **fluxtune only:** everything downstream of the #15 residual (`sim_rate≈0.52`) and #1d (cohort SET 0.011 match):
  `overhead_residual` / `throughput` / `drain_wall_budget` (drain_spread) / `staleness` / `v1_iter_per_data_id` /
  `v2_var_trajectory` / `g2_grad_pool_size` / `terminal_state` / `total_commits` / `convergence` — no new root, all
  cascade from #15/#1d.

### STRATEGY — nail first-data-bin logical parity before any longer run
Prove parity by **logical determinism, not aggregate curve-matching**: for a matched scope the sim must take
**the same sequence of steps in the same order** as real — same trainers selected, same update-receipt order,
same aggregations/rollbacks — differing ONLY in wall-clock. **Scope to the first 1 data bin** (`--max-data-id 1`
/ `--max-bin 1`): bin 0 already contains many iterations, variance passes, and aggregations, so it exercises the
full cadence machinery while staying short, matched, and diffable. Prove bin 0/1 parity FIRST (from banked
telemetry — no re-run); only then chase the time dimension. This isolates length-confound from genuine logic bugs.

### Logical-parity check (TIME-STRIPPED, all available bins) — CURRENT REFERENCE
Tool: `expt_scripts/logical_parity.py [--max-bin N]` — diffs the `agg_round` event stream (data_id, iteration,
receive-ordered contributors, variance decision) real vs sim with every timestamp removed. Real receive-order is
DETERMINISTIC in both real and sim by design (operator-confirmed) → exact match is the correct target.

| baseline | receive-SET (to real's max bin) | cadence | logical parity HOLDS TO | verdict |
|---|---|---|---|---|
| **fwdllm** | 41/41 identical (K=10=all) | 22/41 | **data_id 6** — breaks at **7** | **BREAKS @ bin 7** — but receive-ORDER now 41/41 (K-D31) → the break is NOT order |
| **fwdllm_plus** | 14/14 identical | **14/14 identical** | **real's max (~13)** — no break | **PARITY ✓** over its (shorter) real run |
| **fluxtune** | **3/272 identical** | 29/272 | **breaks at bin 0** | thin-margin overrun (§ #1d) — GPU tail > min D on the contended-doubled GPUs |

**ROOT (full-run investigation, n=100 pair).** With K-D31, fwdllm's receive-ORDER is **41/41 identical** on the
full run — yet cadence still breaks at data_id 7. So order is NOT the sync full-run cause. The residual is
**grad non-reproducibility given matched order**: `|Δvar|` ~1e-3 through bin 6 with *every* `var_good`/force
decision matching, then at bin 7 (all-10 cohort, identical order + RNG) grads diverge ~1e-3 (GPU fp16
non-reproducibility), which the split-half variance — a ratio with a near-zero denominator at a bin's first
iteration — **amplifies to a 0.26 var swing**, flipping the `var<0.3` gate at (7,2). Both modes confirmed
`jvp_perf_opt=False` (no config skew). **Answers P0-2 empirically: grads are NOT bit-reproducible → exact
cadence parity is unattainable past ~bin 6 at n=100.** ⇒ **Parity target (operator decision): cohort SET = HARD;
`var_good`/cadence = HARD to bin 1, DISTRIBUTIONAL beyond; `var` VALUE = SOFT (tolerance/KS); receive-ORDER
within a set = SOFT for sync (fedavg order-invariant + K-D31 canonicalizes).** Databin1 checks confirmed
`cohort_sequence` set/order/var/cadence = 1.0 at `--max-bin 1` (P2-7a DONE, K-D31) on the n=100 pair. For
**fluxtune** the SET still genuinely diverges (#1d) — a timing/pipelining cause, not nondeterminism. Full
diagnosis: **§J**.

**The bin-1 var-VALUE wall is n-scale-sensitive, not a fixed bin.** On a fresh n=10 pair, `cohort_sequence` breaks
as early as `data_id=1` for both fwdllm and fwdllm_plus (`var_match_frac` 0.5/0.6) — SET and `var_good` still
match, only the raw float crosses the 1e-3 tolerance sooner with a smaller cohort (same GPU fp16 jitter as the
n=100 bin-7 break, just earlier). `timing_overrun` confirms this isn't a timing/order issue: 0% GPU-vs-D overrun
in both modes for sync. Take-away: the onset bin scales down with n, so don't assume "HARD to bin 1" holds for
var VALUE at every scale — SET/CADENCE HARD-to-bin-1 does hold at both scales.

### Open issues (OPEN only — closed items live in §G/§H)
| # | issue | baseline(s) | next step |
|---|---|---|---|
| **#1e (RC1)** ⭐⭐ (TOP) | **fluxtune sim selector is BLIND to modeled delay D → wrong selection mix → ~1.9× slow.** `fwdllm_aggregator.py:1337` stamps `PROP_CLIENT_TASK_TRAIN_DURATION` (async_oort speed input) from the REAL wall duration in BOTH modes (no `if self.simulated`); in sim the trainer doesn't sleep so it = raw GPU (~3-6s) for all → Oort can't prefer fast trainers → cohort SET diverges (`set_match=0.042`) → selection_bias/participation/v1/staleness/throughput/… cascade. Sole aggregator missing the sim-branch (asyncfl:1013/oort:912/syncfl:430 all have it). | fluxtune (telemetry-shared, benign for sync) | **FIX LANDED** (sim-branch at :1337 → `SIM_CLIENT_TASK_TRAIN_DURATION_S`; not flag-gated — correctness). **Validation run PENDING** (§A NEXT; gate: selection_bias/participation/cohort pass-or-improve, sim_rate>1). |
| **#1d** (DEMOTED — necessary-not-sufficient; RC1/#1e is the driver) | **fluxtune GPU-vs-D headroom.** With `--delay-divisor 0.5` the commit order (sct=D) is now deterministic, so the old "min delay ≈ GPU → fastest-3 coin-flip" is resolved — yet cohort STILL diverges because selection (RC1) never sees D. Headroom stays a prerequisite (keeps commit order clean) but does NOT close parity alone; the headroom run confirmed this (46/14/19, set_match=0.042). | fluxtune | Fold into #1e: keep `--delay-divisor 0.5`; the real fix is RC1. Trace swap (real LLM-mobile runtimes) remains the principled long-term fix (§J.2). |
| **#15** (fold DEPRIORITIZED) | Phantom commit-path stall FIXED + P3-validated (K-D34). The `sim_model_agg_compute_time` vclock fold is NOT the residual driver — eval is per-databin (10× over the run), aggregate ~1.3s. Re-evaluate only AFTER #1d headroom lands (it changes cycle counts 139→~95). | fluxtune | Deferred behind #1d. `overlap_fraction=0.80` measured (`run_20260709_152612`) if the fold is revisited. |
| **#N (bin-7 nondeterminism)** ⭐ | **SYNC exact-cadence parity has a float-nondeterminism wall, onset bin is n-scale-sensitive (not fixed at ~7).** ~1e-3 GPU fp16 grad jitter, amplified by the split-half variance ratio, flips the `var` gate. Not a sim bug. **Checker fix K-D35 validated:** SET HARD/uncapped, CADENCE/VAR/ORDER capped to bin 1 works as designed at n=100 — but at n=10 the var-VALUE tolerance (1e-3) is *still* crossed inside bin 1 (`data_id=1`), confirmed via `timing_overrun` (0% GPU-vs-D overrun in both modes → not an order/timing issue). | fwdllm (fwdllm_plus latent) | P0-2 (2-real-run diff) still open to empirically bound the jitter magnitude vs n. Otherwise no action — DISTRIBUTIONAL target beyond bin 1 already covers it (v1/v2/v4/v5). |
| **#11** | real-mode critical-path waste (`sleep(0.1)` MQTT-settle busy-waits; one-grad-per-poll drain tail) — real-only. | fwdllm, fwdllm_plus (real) | **Deferred to a validated pass** — ZERO parity impact (sim already skips them); removing them changes the working real reference + needs a real run (principle #8/#11c). |
| **fwdllm `barrier_wait_s` overrun** (minor) | `drain_wall_budget`'s `barrier_wait_s`: sim 2.67s > real 1.43s, fwdllm-only (fwdllm_plus/fluxtune don't show it — sim's cost looks fixed while real's scales with baseline, so only the fastest real baseline gets outrun). ~1.2s/cycle, small next to the 2.6× overall speedup. | fwdllm | Not yet root-caused; low priority next to #15/#1d. |

### Next roots — ranked (correctness before time; SHARED before per-baseline — principle #14)
1. **#1e (RC1) fluxtune selector-speed — TOP.** Fix landed (§A); run the operator validation (`--mode sim`,
   keep `--delay-divisor 0.5`) → gate on selection_bias/cohort + sim_rate>1. #1d headroom stays a prerequisite
   (deterministic commit order); the principled long-term fix is the LLM-mobile runtime trace (§J.2).
2. **`sim_model_agg_compute_time` vclock fold — deferred** behind RC1 (re-measure overlap after the fix lands).
3. **fwdllm `barrier_wait_s` overrun (minor).** Root-cause only if it starts moving `sim_rate` materially.
4. Then C1/C2 convergence (distributional target) at matched `data_id` per baseline → gate to Phase 2.

### SKIP audit (19–21 skips; ~17 legit)
Legit at Phase-1 syn_0 + `random` selector: 7 availability ground-truth rungs + 4 delivery/withheld (Phase-2
effect path, not built) + 3 DynamicKC (disabled by design) + 2 oort-only (`random` baselines) + `residence`
(async telemetry). `timing_overrun` now POPULATES (no longer a skip) — see §A. The 12 former rigor-gap skips
(4 advance, 8 phase-timing) are un-skipped (K-D21).

---

## §B  How fwdllm differs structurally

fwdllm aggregates **gradients** (JVPs) not weights; commit cadence is **endogenous** (variance-gated dynamic-K);
progress axis is committed **`data_id`** (variance passes), not update count. Gradient values are mode-invariant
given identical input+perturbation seed, so parity reduces to **clock + selection + ordering parity plus a
variance-cadence layer**. Anchors (`flame/mode/horizontal/syncfl/fwdllm_aggregator.py`): `aggregate()` var gate +
rollback/`cached_v` at each `_agg_goal` boundary; `total_data_bins=150`; force-commit cap
`max_iterations_per_data_id`; `reselect_each_iteration` (fwdllm++ per-iteration reselection); sync path
`_aggregate_grads_sync`. Full detail: PARITY.md §F.1.

### §B.1  Real↔sim design deltas vs async_cifar10 — **CURATED, KEEP CURRENT** `[checked 2026-07-09]`
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

## §C  Baseline matrix (resolved from the launcher configs) `[checked 2026-07-09]`
*Canonical cross-cutting catalog + the planned 5-baseline restructure: [`../_metadata/BASELINES.md`](../_metadata/BASELINES.md).*
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
- **Per-stage wall-budget instrumentation (K-D36).** New ONE-SIDED (`sim<=real`) rungs where sim should collapse
  a real-transport phase to ~0, DISTRIBUTIONAL/EQUALITY where it's genuine shared compute: `drain_wall_budget`
  (Stage 6, + `processing_wall_ts`), `trainer_phase_wall_budget` + `step_timing_breakdown` (Stage 4),
  `aggregation_compute_wall` + `cycle_model_version` (Stage 6). First live-pair run surfaced a
  `step_timing_breakdown` checker bug (K-D37, fixed) plus the findings in §A/§J.

---

## §E  Roadmap — remaining phases

**Phase 1 (syn_0) — CLOSE-OUT (near done):** #14/#1c/#13/#12c/#7 fixed or explained; sync `sim_rate` is healthy
(§A). Remaining: (1) **#15** fluxtune commit-path residual (§J); (2) C1/C2 convergence at matched `data_id` per
baseline → gate to Phase 2.

**Phase 2 — unavailability (syn_20/50/mobiperf).** Wire the ClientAvailability effect path into the grad loop:
send-time gate (real) / `delivery_ts = max(sct, next_avail)` buffering (sim); two ledgers; reactive-90s in-flight;
starvation vclock-advance; per-baseline `avail_select_filter`. Emit `EVENT_AVAIL_CHANGE` +
`agg_belief_change`/`send_gate_wait` (builders in `flame/telemetry/events.py`) → unlocks A6/A7/A8/K11. **D3 watch:**
does a withheld/late grad roll into `cached_v` on rollback, or a stale-version late grad inflate `var` (thus
dynamic-K)? Over-instrument first. Exit: A1–A5 + A6/A7/A8 PASS; self-stops; withheld grads delivered not dropped.

**Phase 3 — beyond syn_0.** Full ladder under scarcity; bin V1/V2 by run-fraction to separate a constant mix bias
from a compounding variance-feedback loop. Exit: curves within tolerance at matched `data_id`; K8/U2 within bar;
V1/V2 binned residual flat.

---

## §F  Locked principles (from async_cifar10, carried over) `[checked 2026-07-09]`
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
- **#7 fwdllm_plus eligible-count gap (real 5.3 vs sim 10.0 mean eligible, ~4× wall) — root-caused, NOT a sim
  bug.** `profile_eligibility_gap.py` (telemetry-only, no re-run) on the banked n=10 pair: per-trainer
  `_train_one_batch`/`_perform_training` compute is mode-invariant (real 0.82–1.11s vs sim 0.88–1.18s), while
  real-only waits dominate the real per-cycle wall — `_fetch_weights`/`recv_wrapper` MQTT fetch (43.9s/call, #11)
  and `_emulate_training_delay` (10.4s/call, K-D29's intentional remainder-wait; sim charges both as 0 by design).
  Combined with `reselect_each_iteration=True` + K-D11's real-only "commit 1 per pass" anti-deadlock clamp, this
  exactly reproduces the eligibility telemetry: real's FIRST reselect after a full-cohort dispatch always shows
  eligible=1 (one grad drained before refilling); sim's `_sync_sim_recv_first_k` always drains the full cohort in
  one call, so its next reselect always shows eligible=10. The `eligibility`/`avail_timebase`/`selection_detail`/
  `throughput` rung fails for fwdllm_plus are this same expected real-transport artifact, not a new divergence —
  no code fix needed (checker-gating them like K-D37 did for `step_timing_breakdown` is a follow-up choice, not
  yet made).
- **`step_timing_breakdown` real-only-func checker fix (K-D37).** DIST-gated `_emulate_training_delay`/
  `pause_execution` (commented real-only sleeps) and `_fetch_weights`/`recv_wrapper` (dup of the already-DIAG'd
  `phase_mqtt_fetch`). Fix: `_STEP_TIMING_REAL_ONLY_FUNCS` reported but `gates_ok=False`.
- **Stale SHARED-fail list retired.** `gpu_budget_real`/`gpu_budget_sim`/`utility`/`v5_variance_pass_ratio`/
  `overhead_residual`(sync) no longer fail on a fresh pair — superseded by since-landed fixes above; do not
  re-investigate unless they reappear.
- **Per-stage wall-budget instrumentation (K-D36).** No rung caught "sim a little slower at exactly one stage"
  (only whole-run `sim_rate` / two-sided KS). Fix: `drain_wall_budget`/`trainer_phase_wall_budget`/
  `step_timing_breakdown`/`aggregation_compute_wall`, one-sided or distributional per phase.
- **#N bin-7 checker fix (K-D35).** `cohort_sequence` conflated 4 targets under one bin cap, exact past bin 6 was
  unachievable. Fix: SET uncapped/HARD, CADENCE/VAR/ORDER capped to bin 1 (order SOFT for sync).
- **#12c sync `sim_rate` (delay-factor).** No delay-headroom starved the vclock. Fix: `--delay-factor 1` (full
  registry D) → sct = send + max(gpu,D) charges the skipped device wall → sync `sim_rate` 0.94→2.9–3.0.
- **K-D31 validated (P2-7a).** Databin1 checks: sync `cohort_sequence` ok=true, set/order/var/cadence=1.0 — the
  benign delay-tie is canonicalized away; bin-1 cohort order is bit-exact.
- **Aggregator GPU pin (K-D33).** Aggregator eval defaulted to GPU 0, contending with trainers 1/9. Fix: pin the
  aggregator to a dedicated (idle/least-loaded) GPU. async_cifar10 shares the launcher, unaffected.
- **#13 drain stall (K-D28/b/c).** fwdllm's `_sim_recv_min_grad` left a stuck straggler in the expected set →
  re-fired the full 30s `RECV_TIMEOUT` every cycle → pipeline starved (`sim_rate` 0.06). Fix: felix-port
  stuck-end eviction + recv-grace 2→5s + probe-ceiling/ready-gating + `drain_ready` direct ingest → `sim_rate`
  0.06→0.30, 30s stall gone, R1=0. (Step-4 staggering NEUTRAL, OFF — §H; residual is #12c-bound.)
- **#1c R1 two-ledger bridge (K-D27/b).** async_oort gated selection ONLY on `all_selected` (physical-event-pruned)
  and never on the aggregator's virtual in-flight set → slow-sim returned-but-uncommitted trainer re-dispatched
  (R1 62.9%). Fix: agg maintains `_sim_pending_commit` felix-style, binds it live to `sel._agg_pending_commit_ref`,
  async_oort's `filtered_ends` excludes it; `outstanding = inflight ∪ buffer` (K-D27b, NOT `− _sim_committed`).
  Sim-only → async_cifar10 byte-identical. `SIM_R1_DISPATCH` 238→0.
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
- **fluxtune #15 — three superseded framings (same investigation; final root = COMMIT-PATH STALL).**
  (1) "GPU-PIPELINING loss; keep gate, decouple dispatch" — WRONG, felix's arrival gate is INERT (gate_holds=0).
  (2) "re-dispatch on physical RETURN to keep GPUs busy" — WRONG, fedbuff never re-hands a returner the same
  version, and real's 3.37 concurrency is a duty cycle `gpu/max(gpu,D)`, not under-use. (3) "hold-to-commit is a
  SYNC barrier / over-restrictive; replace with model-advance re-dispatch" — WRONG, hold-to-commit is a
  CORRECTNESS check (freed only on commit; guards no-same-version / not-while-computing /
  not-while-returned-uncommitted). *Final root:* the commit path STALLS (drain gate blocks real wall on PHANTOM
  `_sim_inflight_expected` entries) → correctly-held trainers idle. Fix = fast/non-stalling commit path;
  hold-to-commit untouched. *Lesson:* commit RATE is the throughput lever, not the residence rule. See
  **§I** F5/F6.
- **"The sync cadence break is a sim ORDER bug (order → split-half var → RNG desync) — exact cadence parity is
  achievable once order matches."** CORRECT for bin ≤1, REFUTED for the full run. K-D31 made
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

## §I  FELIX grounding (async substrate — durable; the reference model for #15)

*How async_cifar10's felix drain behaves (code + `../async_cifar10/PARITY.md`). Anchors: **asyncfl** =
`flame/mode/horizontal/asyncfl/top_aggregator.py`; **fwdllm** = `flame/mode/horizontal/syncfl/fwdllm_aggregator.py`.
Directly governs the new async baselines `fedbuff+round`/`fedbuff+Iter` (BASELINES.md §2), which inherit this substrate.*

- **F1 — felix's expected-sct arrival gate is INERT (gate_holds=0), not load-bearing.** cifar GPU ≈0.4s, so every
  in-flight update is arrived+buffered when the drain runs; effective behavior = sort arrived updates by sct and
  commit. A dormant safety net (asyncfl:359-448; PARITY.md:294-296/321-323).
- **F2 — felix re-dispatches on COMMIT; the trainer idles in `recv` — sub-second only because compute≈0.**
  `_sim_hold_busy_slots` (asyncfl:1453-1500) holds each busy trainer until its update commits (released :618-627).
  fwdllm's K-D17b is a FAITHFUL port.
- **F3 — two ledgers already SEPARATE in felix:** PHYSICAL (GPU/MQTT, `_sim_buffer`) vs VIRTUAL (selection
  eligibility, staleness gate, vclock). "Busy ≠ UN_AVL": a busy trainer holds a SLOT, released on commit.
- **F4 — regime table** (why the same mechanism transfers to sync fwdllm but breaks async fluxtune):

  | baseline | GPU | agg_goal vs c | return→commit idle | gate | outcome |
  |---|---|---|---|---|---|
  | cifar felix | ~0.4s | (varies) | sub-second | inert | sim_rate ≫ 1 |
  | fwdllm/plus (sync) | ~1.0s | K=c (barrier) | ~1 GPU pass, ALL commit | inert | `sim_rate` 2.6-3.0 ✓ |
  | **fluxtune (async)** | **~4s** | **3 ≪ 10** | **multi-sec → 30s** | **load-bearing → stall** | **`sim_rate` ~0.5 ⛔** |

  `[checked 2026-07-09 — figures hold across both the n=100 and n=10 banked pairs]`

- **F5 — hold-to-commit is a CORRECTNESS CHECK, not over-restriction.** A trainer is freed (re-selectable) only
  once its update is COMMITTED. Three guards: never dispatch (a) a version it already computed, (b) while it is
  still computing, (c) while its returned update is uncommitted. Do NOT weaken it. With N=C (fluxtune 10/10/3, no
  idle pool) real's 3.37 concurrency is a duty cycle (gpu/max(gpu,D)≈0.34 → N×0.34), NOT under-utilization.
- **F6 — the bug is purely the COMMIT PATH STALLING; commit RATE is the throughput lever.** Trainers freed only on
  commit ⇒ commit rate = throughput. cifar's commit path is instant; fluxtune's STALLS (#15/§J) → held trainers
  idle 30s. Fix = fast/non-stalling commit path; the gate must wait only on a genuinely-COMPUTING trainer, never a
  phantom. Hold-to-commit untouched.

---

## §J  fluxtune `sim_rate<1` root (#1d headroom) + resume plan

> **SUPERSEDED as the DRIVER by RC1 (§A): the sim selector was blind to D (`:1337`).** The #1d headroom below is
> still a prerequisite (keeps commit order sct=D deterministic) but does NOT close parity alone — the `--delay-divisor
> 0.5` run confirmed it (46/14/19). Keep §J's headroom/trace steps; the primary fix is RC1. Reconcile fully after the validation run.

**Root + fix: full narrative in K-D34 (§K).** Summary: hold-to-commit frees a trainer only on commit (F5/F6) →
commit RATE sets throughput; fluxtune's drain gate was blocking real wall on a PHANTOM `_sim_inflight_expected`
entry → 30s failsafe → correctly-held trainers idled ~30s instead of one GPU pass. **Fix LANDED + P3-VALIDATED**
(`sim_compute_truthful_gate`, flag-gated default off = byte-identical; validated on `run_20260706_112114` sim /
`_110555` real: `STUCK_EVICT=0`, `phantom_skip`→65 ~1/commit, 30s failsafes gone, max gap 12.8s).

**Residual RE-DIAGNOSED (2026-07-10) — the driver is #1d headroom, NOT a vclock fold:**
- **DOMINANT — no GPU-vs-D skip headroom (#1d).** min registry delay 4s ≈ JVP GPU 3.86s → 11% of passes overrun D
  (`timing_overrun` sim 11% / real 7.2%) → the async fastest-3 cohort is a GPU coin-flip, not deterministic D →
  cohort SET diverges at **cycle 0** (`set_match=0.011`, real {…370,375,373} vs sim {…379,372,378}) → sim grinds
  **13.9 vs real 9.5 iters/databin** → **139 vs 95 cycles** for the same 10 databins → inflated sim wall + every
  downstream rung. **Fix = config-only headroom** (`--delay-divisor 0.5 --num-gpus 10`); the principled fix is a
  realistic LLM-mobile runtime trace (delays naturally ≫ GPU). **The trace is NOT orthogonal to parity** — a
  min-delay ≈ GPU-time collision is what makes the async fastest-K nondeterministic across two independent runs.
- **NOT the driver — the aggregate/eval vclock fold.** Telemetry (`run_20260709_152612`): eval runs **10× over the
  whole run** (once per databin, gated on `var_good_enough` at `:1816`), not per cycle; `aggregate_fedavg` is ~1.3s.
  So the `overhead_residual` 30s/commit is the modeled delay D + extra cycles, not eval. `sim_model_agg_compute_time`
  is at most a small residual cleanup — re-measure `overlap_fraction` only AFTER headroom lands.

**RESUME steps — headroom FIRST (config-only), then re-assess the fold:**

1. **Headroom run (operator) — the #1d fix.** Restore D-dominance so the async fastest-K cohort is decided by the
   deterministic registry delay, not a GPU race:
   ```bash
   cd lib/python/examples/fwdllm/expt_scripts
   ./run_sequential.sh --only fluxtune --mode both --delays on --num-gpus 10 --delay-divisor 0.5
   python run_parity.py --yes --baselines fluxtune
   # GATE: cohort set_match ↑ (deterministic) · v1 iters real≈sim (~9.5) · timing_overrun <5% · sim_rate → >1
   ```
   `--delay-divisor 0.5` doubles the delay (4–18s → 8–36s, min 8s ≫ GPU 3.86s); `--num-gpus 10` drops GPU by
   removing the 2-trainers-on-GPU-0/1 contention. Either alone gives headroom; both is robust. `delay_divisor` is
   a DIVISOR — <1 LENGTHENS (K-D38); this is applied identically real+sim so it's admissible (principle #6), not a
   parity lever.
2. **Trace swap (principled fix; TODO).** The pre-LLM papaya/fedbuff 4–18s trace's min collides with the A40 JVP
   (3.86s). Real mobile LLM forward-grad is far slower → a real trace's delays would clear the GPU for free (no
   `delay_divisor` hack) AND improve realism. **Next session: pull fwdllm's codebase — it is believed to carry
   per-model/per-phone runtime numbers — and replace the delay distribution.** This subsumes the old "latency-trace
   is orthogonal" TODO (it isn't — see the residual re-diagnosis above).
3. **Re-assess the `sim_model_agg_compute_time` fold — only if `sim_rate` still <1 after headroom.** Re-measure
   `overlap_fraction` on the new pair first (headroom changes cycle counts 139→~95). NOT YET IMPLEMENTED; flag-gate
   default-off per [[flag-gate-ab-lifecycle]].

**Other open fronts (after #1d headroom; priority order — matches the §A open-issues table):**
1. **bin-7 nondeterminism → relax the parity target (SHARED).** ~1e-3 GPU fp16 grad jitter, amplified by the
   split-half variance ratio, flips the `var` gate — not a sim bug; onset bin is n-scale-sensitive (n=10 hits it
   inside bin 1 already, §A's logical-parity-check ROOT). Keep `cohort_sequence` SET/CADENCE EXACT to `--max-bin 1`,
   var VALUE DISTRIBUTIONAL beyond it (already true past bin 1; n=10 shows it's needed AT bin 1 too for `var`).
   Confirm with a 2-real-run diff (P0-2) first.
2. **#11 real-mode critical-path waste** (`sleep(0.1)` MQTT-settle, one-grad-per-poll drain tail) — real-only, ZERO
   parity impact; deferred to a validated pass. (#7's eligible-count gap is this same class, root-caused — §G.)
4. **NEW async baselines** `fedbuff+round`/`fedbuff+Iter` (BASELINES.md §2): when they land, add them to the parity
   ladder — they inherit the felix async substrate (§I) and this same commit-path/vclock machinery, so re-run the
   fluxtune gates above per baseline.

**Anchors.** Split-half var `aggregator/.../fwdgrad_utils.py:133-158`; commit-order append
`fwdllm_aggregator.py:718-721`; RNG-once seed `.../tc_transformer_trainer_distribute.py:222-225`; delay/sct
`FedSgdTrainer.py:510-538`(sleep)/`:623-629`(sct); sim drain `fwdllm_aggregator.py:_sim_recv_min_grad`(~752-970);
dispatch stamp `:2819`; send `:2836`; compute-truthful gate `:856`; eval fold `:1783`; agg-overlap window
`fwdllm_aggregator.py:1745`(`agg_compute_start/end_wall`)/`FedSgdTrainer.py:601`(`gpu_pass_start/end_wall`);
checker rungs `async_cifar10/scripts/parity/checks.py`; standalone diff `expt_scripts/logical_parity.py`; #7
eligible-count-gap diagnostic `expt_scripts/profile_eligibility_gap.py` (§G); real commit-1-per-pass clamp
`fwdllm_aggregator.py:2081-2083` (K-D11); sim full-cohort barrier drain `fwdllm_aggregator.py:_sync_sim_recv_first_k`
(~2099-2126); reselect gate `_select_ends_respecting_reselect_gate` (~2524-2571); selector eligible-pool compute
`flame/selector/random.py:220-253`.

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
- **K-D21** — pre-run instrumentation A–E landed; un-skipped the 12 rigor-gap rungs (§G).
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
  (`sim_straggler_spread_s=0`); `perturbation_count` made a knob (default 10). See **§J**.

- **K-D30** — **full-cohort determinism gate + `timing_overrun` signal.** The selection set/sequence rungs gated
  to a TRIVIAL pass for every stochastic selector, so fwdllm's syn_0 selection was never actually checked. Fix:
  un-gate (enforce as EXACT) whenever `num_chosen==num_candidates` in both modes — data-driven, so it enforces
  fwdllm (K=all) while keeping fluxtune/fwdllm_plus properly gated (they aren't set-deterministic). New
  `timing_overrun` DIAG rung surfaces the K-D29 overrun tell (gpu>D flips order → a cohort/var break downstream is
  a timing-model limit, not a sim bug). Checker-only, async_cifar10 unaffected.

- **K-D31** — **canonical `(D, trainer_id)` cohort commit order.** Two trainers sharing a registry delay could
  swap receive order (real breaks the tie by physical arrival, sim by sct-sort) — benign (same split-half, grads
  bit-identical) but flagged by the exact-order `cohort_sequence` rung. Fix: canonicalize each cohort by
  `(D, trainer_id)`, since `D` (unlike sct) is deterministic and mode-identical — only benign ties move, real
  order is untouched. Sim/agg-only → async_cifar10 byte-identical.

- **K-D32** — **fluxtune JVP perf optimizations (`jvp_perf_opt`, §L; fluxtune-only, config-gated, all
  BIT-IDENTICAL).** fluxtune's forward-grad JVP was ~10× the sync compute (2P=20 selection passes + 5 removable),
  overrunning its modeled mobile delay → out-of-order commits (#1d). Fix (validated by
  `scripts/profile_jvp_opt.py`): (a) trainable-only finite difference — `calculate_jvp(trainable_idx=…)` skips
  `p−h·0=p` on the 98.5% frozen backbone (1.26× + −251 MB); (b) skip the 3 diagnostic-only forward passes; (c)
  reuse the selected perturbation's cached JVP. Combined fluxtune −37% (sync would be −68% but is left OFF).
  Gated on `jvp_perf_opt` (trainer_base default false = byte-identical; true in both fluxtune yamls, must match
  real↔sim; revertible per-config). NOT adopted: vmap (2× but fp32-diverges via FD cancellation), fwd-AD (slower).

- **K-D33** — **aggregator GPU pin.** Aggregator eval defaulted to GPU 0, contending with trainers 1/9. Fix: pin
  the aggregator to a dedicated (idle if oversubscribed, else least-loaded) GPU. Gotcha: `FedSgdTrainer`'s
  `client_idx%8` device arg is vestigial — `self.device` is overwritten to cuda:0 of the CVD-masked view, so the
  spawner's pin (not that arg) is what's actually authoritative for trainer placement.

- **K-D34** — **fluxtune #15 = a COMMIT-PATH STALL; `sim_compute_truthful_gate` fix LANDED (P1/P2), P3
  VALIDATED.** Supersedes three same-session mis-framings (§H). Grounding (§I FELIX GROUNDING F5/F6): hold-to-commit frees a trainer only on COMMIT (guards: no same-version dispatch, none while
  computing, none while returned-but-uncommitted), so commit RATE is the throughput lever; felix's gate is INERT
  and cifar's commit is instant (~0.4s) so held trainers barely idle. **D1/D2 CONFIRMED** (banked pair):
  fluxtune's `_sim_recv_min_grad` `earlier_stuck` gate holds already-arrived grads (`buf_depth=7`, 99.7%) to wait
  on trainers stamped-expected-at-DISPATCH but idle-in-recv (not computing) behind the single-threaded drain →
  82% of wall (~1974s) burned → concurrency sim 1.65 vs real 7.69 → sim_rate 0.50. **Fix (flag
  `sim_compute_truthful_gate`, default off = byte-identical, fluxtune-yaml on):** stamp `_sim_dispatch_wall[end]`
  at the real `channel.send`; the gate skips any expected entry whose last dispatch is older than
  `sim_gate_compute_cap_s` (default 10s) — a stamped-but-idle phantom no longer blocks a ready commit, a genuine
  in-window straggler is still held (commit order preserved). Hold-to-commit, sct-ordered drain, K-D12, K-D27
  UNTOUCHED; `fwdllm_aggregator`-only → async_cifar10 byte-identical. **P3 validated on
  `run_20260706_112114` sim / `_110555` real:** phantom fix **VALIDATED** — `STUCK_EVICT=0`, `phantom_skip`→65
  (~1/commit, load-bearing), 30s failsafes gone (max gap 12.8s). **But `sim_rate` still <1** (steady ~0.49;
  per-commit `Δvclock/Δwall=0.465`) — the phantom stall was necessary but NOT the ceiling. **Residual root
  (NEW):** (a) vclock omits `aggregate()` variance-compute wall (sim 32s / real 29s, symmetric, uncredited —
  `:1783` folds eval only); (b) GPU≈D no-headroom (#1d) → sim blocks in `drain_ready` on the real GPU pass. Sim
  wall (145s) > real (91s). **PAUSED** for operator real-experiment work; resume plan (fold non-overlapped
  aggregate into vclock + GPU-vs-D headroom + longer flag-on run & flag-off A/B) at **§J**.

- **K-D35** — **#N bin-7 checker fix: `cohort_sequence_parity` split into 4 independently-scoped targets.**
  One `max_bin` cap conflated SET/CADENCE/VAR/ORDER, so a full-run check demanded exact cadence past the bin-~7
  nondeterminism wall (§A ROOT) — impossible, not a bug. Fix: SET stays HARD/uncapped (fluxtune's #1d must still
  fail); CADENCE/VAR gate only within `max_bin` (default 1); ORDER gates only when `is_async` (sync's fedavg is
  order-invariant, K-D31 canonicalizes ties). Checker-only, async_cifar10 unaffected. **Validated on a fresh
  n=10 pair (§A):** the split works as designed, but at n=10 scale the bin-1 var-VALUE cap is itself crossed
  — the onset bin is n-scale-sensitive, not fixed at ~7. P0-2 confirmation still open.

- **K-D36** — **per-stage wall-budget instrumentation (`drain_wall_budget`/`trainer_phase_wall_budget`/
  `step_timing_breakdown`/`aggregation_compute_wall`).** No rung caught "sim a LITTLE slower at exactly one
  stage" (only whole-run `sim_rate` or two-sided KS) — the #15 shape, generalized. New rungs are ONE-SIDED
  (`sim<=real`) for phases sim should collapse to ~0 (transport/dispatch), DISTRIBUTIONAL/EQUALITY (KS/mean) for
  genuine shared compute that should match, not just bound. `step_timing`/`aggregation_compute_wall` start as
  DIAG (new sources, not yet strict-enforced) until proven noise-free on a real run. Deliberately no
  model-version-trajectory rung — it's 1:1 with the already-checked `cycle_data_id` cadence, would add no signal.

- **K-D37** — **`step_timing_breakdown` real-only-func exemption (first live-pair run of K-D36).**
  The rung DIST-gated every `step_timing` function as "genuine shared compute," but `_emulate_training_delay`
  (K-D29 remainder-wait, commented "real sleeps, sim skips") and `pause_execution` (commented "real-transport
  artifact... gate off in sim") are real-only sleeps by design; `_fetch_weights`/`recv_wrapper` duplicate the
  same MQTT recv `phase_mqtt_fetch` already treats as DIAG-only. Fix: `_STEP_TIMING_REAL_ONLY_FUNCS` reported,
  `gates_ok=False`, mirrors `trainer_phase_wall_budget_ok`'s `mqtt_fetch_s` treatment. Residual fails on the
  remaining funcs are small-N KS noise (means agree within ~8%) — the `phase_gpu_compute` precedent generalizes.
  Checker-only, both `checks.py` and its `parity_checks.py` re-export shim needed the update.

- **K-D38** — **fluxtune `sim_rate<1` root RE-DIAGNOSED = #1d headroom (supersedes the K-D34 "vclock fold"
  residual framing); `training_delay_factor` clarified as a DIVISOR.** Banked telemetry (`run_20260709_152612`
  real / `_153340` sim): the min registry delay (4s) ≈ JVP GPU (3.86s) → 11% overrun → async fastest-3 cohort is a
  GPU coin-flip → SET diverges at cycle 0 → sim 13.9 vs real 9.5 iters/databin (139 vs 95 cycles) → all downstream
  rungs cascade. The `sim_model_agg_compute_time` fold is NOT the driver (eval is per-databin 10×, aggregate ~1.3s).
  Fix = config headroom (`--delay-divisor 0.5 --num-gpus 10`); principled fix = real LLM-mobile runtime trace (the
  trace is NOT orthogonal to parity). Code: `training_delay_factor` is a DIVISOR (effective = delay/factor; <1
  lengthens) — trainer now reads it as `self.training_delay_divisor`, CLI flag `--delay-divisor` (alias
  `--delay-factor`), warning + docs corrected (was backwards). Wire key unchanged (back-compat); async_cifar10
  trainer doesn't consume it. Tests green.
*Retired/superseded anchors (kept only as pointers): K-D6 (→K-D12), K-D7/K-D8/K-D10/K-D16/K-D18/K-D19/K-D20/K-D23
— landed scaffolding or corrections, folded into §G/§H; see git history for detail.*

---

## §L  Forward-grad JVP compute profile & retained fluxtune optimizations
*(tool: `scripts/profile_jvp_opt.py` — reuses real `create_model` + `calculate_jvp`; distilbert-base
+ AdapterHub adapters, batch 8, seq 192, A40, fp16. Absolute ms are a CLEAN single-trainer profile; the real run
is ~10× from GPU contention across the 10 concurrent trainers, but pass-counts/ratios/memory transfer.)*
`[checked 2026-07-09]`

**Mechanism.** Forward-grad trains via a **central finite-difference JVP** (`fwdgrad_utils.calculate_jvp`): each
perturbation = **2 forward passes** `f(θ±hv)`, h=0.01, autocast+no_grad → `jvp=(f(θ+hv)−f(θ−hv))/2h`. **fluxtune**
selects the best of `perturbation_count`(=10) perturbations by |jvp| (2P=**20 passes**); **fwdllm/sync** selects
by cos-sim (**0 forward passes**) + 1 final JVP. Only **~1.5% of params trainable** (bottleneck adapters in all 6
layers + head, 1.04M/67.4M); backbone frozen.

| path | fwd passes | ms/batch (clean) |
|---|---|---|
| sync fwdllm (current) | 5 | 50 |
| **sync fwdllm (opt)** | 2 | **16 (−68%)** |
| fluxtune P=1 (opt) | 2 | 16 (== sync) |
| fluxtune P=5 (opt) | 10 | 80 |
| fluxtune P=10 (current) | 25 | 251 |
| **fluxtune P=10 (opt)** | 20 | **159 (−37%)** |
| backprop ref (1 fwd+1 bwd) | — | 17 |

- **Compute vs sync:** fluxtune = `2P × per-pass` → **10× sync at P=10**, linear in P, **equals sync at P=1**.
  JVP-selection is the entire fluxtune surcharge; sync's cos-sim selection is free.
- **Memory:** forward-grad peak is **FLAT in P** (~3.2–3.4 GB = model + one held forward; **no autograd graph**).
  Backprop stores activations (3.71 GB). fluxtune's extra JVP inferences cost **TIME, not memory** — same
  footprint as sync (forward-grad's design tradeoff: many cheap forward passes, no backward, low memory).
- **Per-pass:** full-param FD 10.0 ms; trainable-only FD 7.95 ms.

**RETAINED — bit-identical (fidelity-preserving; real↔sim parity untouched):**
1. **Trainable-only FD** — skip `p−h·0=p` on the 98.5% frozen params inside `calculate_jvp`: **1.26× + −251 MB**,
   `max|Δjvp|=0`.
2. **Drop the 3 diagnostic-only forward passes** (`_train_one_batch:646-648`, loss before/after-update logging —
   never feed grads/telemetry) **+ reuse the winner's cached JVP** (`:645`, fluxtune): fluxtune 25→20, sync 5→2.

Combined: **sync −68%, fluxtune −37%**, all bit-identical → should clear the `delay_factor=1` overrun (4.2s →
~2.6s < min cohort D 4.0s) WITHOUT touching fidelity. **LANDED (K-D32), fluxtune-only & config-gated**
(`jvp_perf_opt`, default false = byte-identical; true in both fluxtune yamls; sync untouched).

**NOT retained — change fidelity:**
- **vmap-batching the perturbations** — **2.0×** (biggest single win), mathematically exact (fp64 seq==vmap
  BIT-IDENTICAL, deterministic → *would* be real↔sim safe) BUT differs ~5% from sequential in fp16/fp32: the FD
  subtracts two O(1) losses (catastrophic cancellation floors precision), so any reduction-order change
  re-baselines the trajectory. Excluded per the fidelity bar; available if a re-baseline is accepted.
- **Forward-mode AD** (exact JVP) — slower (0.5×, needs eager attention; not impl for SDPA) + different math.
- **`perturbation_count`↓** — the direct lever, but changes the baseline algorithm.
