# High-Fidelity Simulator for FwdLLM — Real↔Sim Parity

**Active build (branch `dg/sim_parity_fwdllm`).** A simulated-clock runner for the `fwdllm` example
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

**Last landed:** consolidated re-baseline (all 3 baselines, `run_20260710_1820 → 1955`,
`--mode both --delays on --num-gpus 8 --delay-divisor 0.5 --max-runtime-s 5400`) **validates RC1/RC3/§M**:
sync `sim_rate` 5.6–5.8 (healthy, up from 2.6–2.7), fluxtune `sim_rate` rung now **PASSES** at 0.91 (up from 0.52),
and logical parity is clean (fwdllm/fwdllm_plus full to bin ≤1; fluxtune cadence 24/24 + first ~8 cohorts identical,
up from 3/272). No regressions — every sync/fwdllm_plus fail is #N float-nondeterminism, length-confound, or a
known-artifact bucket.
**Next (RESUME HERE — see #S1):** fluxtune async cohort-SET divergence is **root-caused to a sim STALENESS
under-modeling bug** (not headroom, not #N). The `--delay-divisor 0.25` diagnostic (`run_20260710_2242`) proved it.
Fix = stamp compute-time `version_key` through the sim reorder buffer so staleness matches real; verify at 0.5.

Latest banked pairs (`run_sequential.sh --mode both --delays on --num-gpus 8 --delay-divisor 0.5
--max-runtime-s 5400`, n=10 smoke; real runs all hit the 5400s runtime cap):

| baseline | sim `sim_rate` | sim wall / real wall | verdict |
|---|---|---|---|
| **fwdllm** (sync) | **5.62** ✓ | 959s / ~5400s | healthy; 5.6× wall speedup |
| **fwdllm_plus** (sync) | **5.77** ✓ | 943s / ~5460s | healthy; 5.8× wall speedup |
| **fluxtune** (async) | **0.91** ✓(rung) | 5967s / ~5400s | near-1, sim_rate rung passes (was 0.52); NO wall speedup — compute-bound floor (open issues). Cohort-SET divergence is #S1 (staleness), not sim_rate |

### Parity scoreboard (checker on the pairs above; `expt_scripts/run_parity.py --yes`)

| baseline | pass / fail / skip | JSON |
|---|---|---|
| **fwdllm/syn_0** | **51 / 8 / 21** | `experiments/_parity_reports/parity_fwdllm_syn_0_20260710_195554.json` |
| **fwdllm_plus/syn_0** | **48 / 10 / 21** | `parity_fwdllm_plus_syn_0_20260710_195312.json` |
| **fluxtune/syn_0** | **42 / 17 / 19** | `parity_fluxtune_syn_0_20260710_195238.json` |

Slightly fewer passes than the shorter prior run — expected: the 5400s length feeds more cycles into DIST/length
rungs. Every fail is explained below; none is a new bug.

**Fails, categorized by blast radius (fix the SHARED roots first — principle #14).**
- **SHARED — all 3:** `cohort_sequence` (SET/ORDER/CADENCE all 1.0 for sync; the ONLY failing component is var
  VALUE at bin ≤1 — pure #N, see below); `v2_var_trajectory` (var-mean 3.8–5.4% > 2% tol but KS passes — #N
  accumulation); `step_timing_breakdown`/`per_round_advance` (small-N KS, K-D37).
- **fwdllm only:** `drain_wall_budget` `barrier_wait_s` overrun (sim 2.5s vs real 0.017s; drain-tail/spread PASS);
  `utility` (RandomSelector, max_ks 0.247 vs 0.2 = 10-sample KS noise); `terminal_state`/`total_commits` (length
  confound from the healthy `sim_rate`, not a bug).
- **fwdllm_plus only:** `eligibility`/`avail_timebase`/`selection_detail`/`throughput`/`overhead_residual` —
  real-transport artifact, root-caused §G "eligible-count gap", not a bug.
- **fluxtune only:** `cohort_sequence` SET collapses, and everything keyed off it (`v1`/`g2`/`staleness`/`v2`/
  `convergence`) — ALL downstream of **#S1** (sim staleness under-modeling → variance bias → cadence desync). NOT a
  selection bug (the `run_20260710_2242` 0.25 diagnostic shows cohorts bit-identical iteration-for-iteration until
  the variance-driven cadence desync misaligns them). RC1/RC3/§M are landed (cadence + first ~7 cohorts identical).

### STRATEGY — nail first-data-bin logical parity before any longer run
Prove parity by **logical determinism, not aggregate curve-matching**: for a matched scope the sim must take
**the same sequence of steps in the same order** as real — same trainers selected, same update-receipt order,
same aggregations/rollbacks — differing ONLY in wall-clock. **Scope to the first 1 data bin** (`--max-data-id 1`
/ `--max-bin 1`) before chasing the time dimension; isolates length-confound from genuine logic bugs.

### Logical-parity check (TIME-STRIPPED, all available bins) — CURRENT REFERENCE
Tool: `expt_scripts/logical_parity.py [--max-bin N]` — diffs the `agg_round` event stream (data_id, iteration,
receive-ordered contributors, variance decision) real vs sim with every timestamp removed. Real receive-order is
DETERMINISTIC in both real and sim by design (operator-confirmed) → exact match is the correct target.

Default scope is bin ≤1 (STRATEGY's first-data-bin target). Numbers below from the `run_20260710_1955` sim pairs.

| baseline | receive-SET | cadence | verdict |
|---|---|---|---|
| **fwdllm** | 3/3 identical (K=10=all) | 3/3 identical | **LOGICAL PARITY** (bin ≤1) |
| **fwdllm_plus** | 3/3 identical | 3/3 identical | **LOGICAL PARITY** (bin ≤1) |
| **fluxtune** | 8/24 identical | **24/24 identical** (at 0.5) | cohorts identical iter-for-iter then desync — root = **#S1** staleness bug (was 3/272 pre-RC3). At 0.25 cadence breaks earlier (real clears bin 0 in 7 iters, sim 10) as the staleness→variance gap widens |

**Root (n=100 pair, #N):** grad non-reproducibility given matched order — ~1e-3 GPU fp16 jitter, amplified by the
split-half variance ratio, flips the `var<0.3` gate at a sensitive bin (K-D31 already made receive-ORDER 41/41
identical, so order itself is not the cause). **Parity target (operator decision):** cohort SET = HARD; `var_good`/
cadence = HARD to bin 1, DISTRIBUTIONAL beyond; `var` VALUE = SOFT; receive-ORDER within a set = SOFT for sync.
**n-scale-sensitive:** at n=10 the var-VALUE wall moves as early as `data_id=1` (same jitter, smaller cohort) — the
onset bin is not fixed at ~7; for SYNC, SET/CADENCE stay HARD-to-bin-1 at every scale (SET 3/3, CADENCE 3/3).
**#N is SYNC-only.** fluxtune's async cohort-SET divergence is NOT #N — it is **#S1** (sim staleness under-modeling
→ variance bias → cadence desync → SET misalignment), proven by the 0.25 diagnostic (cohorts bit-identical
iter-for-iter; the divergence tracks staleness/delay, not fp16 jitter). See §H for the refuted "async #N" framing.

### Open issues (OPEN only — closed items live in §G/§H)
| # | issue | baseline(s) | next step |
|---|---|---|---|
| **#S1 fluxtune sim STALENESS under-modeling (async cohort-SET root)** | Sim grads too FRESH: `staleness` rung real 1.14 vs **sim 0.218**. `staleness = self._round − tres.version` ([asyncfl/top_aggregator.py:1108](../../flame/mode/horizontal/asyncfl/top_aggregator.py#L1108)); sim's ≈0 ⇒ `tres.version` is stamped at COMMIT not compute/dispatch, so fedbuff's staleness decay never shrinks sim grads → sim variance systematically HIGHER → sim clears bin 0 in 10 iters vs real's 7 → cadence desyncs → cohort-SET collapses (downstream, NOT a selection bug). Delay-coupled: `v2_var_trajectory` gap 5%→11% as delay 0.5→0.25. This is D3 arriving in Phase 1. | fluxtune (async only) | (1) telemetry+code confirm `tres.version` is commit-time in sim; (2) FLAG-GATED fix: carry compute-time `version_key` (K-D39) through the SimReorderBuffer to commit; (3) verify at `--delay-divisor 0.5` — v2 gap collapses, staleness matches, iters-to-clear re-syncs, cohort-SET recovers. |
| **fluxtune `sim_rate`<1 (compute-bound floor)** | At 0.5, headroom already adequate (sct D≥8s > JVP gpu ~3.5–5s, `phantom_skip=0`, `STUCK_EVICT=0`) — NOT the K-D38 D≈gpu collision. Residual = compute floor: P=10 JVP (10× sync) + ~8.9s unskippable aggregator eval + agg_goal=3 low GPU parallelism; sim already ~4× faster per commit (advance 79s vs 308s). | fluxtune | Real LLM-mobile trace (Next roots #1). Shrinking divisor (0.25/0.1) pushes the NUMBER >1 by doing fewer data_ids per vclock ceiling — baseline modeling knob (principle #3), not a parity fix. |
| **#N (var-VALUE nondeterminism wall)** | Float-nondeterminism flips the `var<0.3` gate → cohort SET (async) / var VALUE (sync) diverge past a sensitive bin (onset n-scale-sensitive). Not a sim bug. | fwdllm, fluxtune (fwdllm_plus latent) | DISTRIBUTIONAL target beyond bin 1 already covers it; P0-2 (2-real-run diff) open only to bound jitter magnitude vs n. |
| **#11** | Real-mode critical-path waste (`sleep(0.1)` MQTT-settle; one-grad-per-poll drain tail) — real-only, zero parity impact. | fwdllm, fwdllm_plus (real) | Deferred to a validated pass — needs a real run to touch (principle #8/#11c). |
| **fwdllm `barrier_wait_s` overrun** (minor) | sim 2.5s > real 0.017s, fwdllm-only; drain-tail/spread PASS. | fwdllm | Not yet root-caused; low priority next to `sim_rate`'s 5.6× overall speedup. |

### Next roots — ranked (correctness before time; SHARED before per-baseline — principle #14)
1. **#S1 fluxtune staleness fix — TOP (RESUME HERE).** Sim under-models grad staleness (0.218 vs real 1.14) → the
   whole async cohort-SET divergence. Confirm `tres.version` is commit-time in the sim async path, then carry the
   compute-time `version_key` (K-D39) through the SimReorderBuffer so `self._round − tres.version` matches real.
   Flag-gated (default off). Verify at `--delay-divisor 0.5`: v2 gap collapses, staleness matches, cohort-SET
   recovers. Open-issues #S1 has the full chain; diagnostic pair = `run_20260710_2242` (0.25/900s).
2. **LLM-mobile runtime trace swap (principled).** Only after #S1: the papaya/fedbuff 4–18s trace is a modeling
   choice; a real mobile-LLM forward-grad trace would give honest headroom (`sim_rate`>1) AND set the delay regime
   the staleness fix must hold under. Pull fwdllm's codebase (believed to carry per-model/per-phone runtime numbers)
   and replace the delay distribution. Divisor tuning (0.25/0.1) is NOT this — it's a diagnostic knob (principle #3).
3. **`sim_model_agg_compute_time` vclock fold — deferred** (re-measure overlap now that the re-baseline landed; #15).
4. **fwdllm `barrier_wait_s` overrun (minor).** Root-cause only if it starts moving `sim_rate` materially.
5. Then C1/C2 convergence (distributional target) at matched `data_id` per baseline → gate to Phase 2.

### SKIP audit (19–21 skips; ~17 legit)
Legit at Phase-1 syn_0 + `random` selector: 7 availability ground-truth rungs + 4 delivery/withheld (Phase-2
effect path, not built) + 3 DynamicKC (disabled by design) + 2 oort-only (`random` baselines) + `residence`
(async telemetry). `timing_overrun` now POPULATES (no longer a skip). The 12 former rigor-gap skips (4 advance, 8
phase-timing) are un-skipped (K-D21).

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
  `step_timing_breakdown` checker bug (K-D37, fixed).

---

## §E  Roadmap — remaining phases

**Phase 1 (syn_0) — CLOSE-OUT (near done):** #14/#1c/#13/#12c/#7 fixed or explained; sync `sim_rate` is healthy
(§A). Remaining: (1) fluxtune `sim_rate`<1 / no wall speedup — needs the real LLM-mobile trace (§A Next roots #1),
vclock fold deferred (§A Next roots #2); (2) C1/C2 convergence at matched `data_id` per baseline → gate to Phase 2.

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
    a real wait it should skip. Emit it every run (`[VCLOCK_PROGRESS]`).
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

**Open design decisions:** D2 (avail telemetry port — Phase 2); **D3 (variance-cadence × late/stale grads —
ELEVATED to Phase 1: it IS the #S1 fluxtune root; the sim must reproduce real's grad staleness so the variance
trajectory matches)**; D4 (eval-delay factor — confirmed ~1× train cost, K-D3). D1/D6 resolved (§K).

---

## §G  Fixes landed (what worked — ≤20-word problem + ≤20-word fix; do not redo)
- **fluxtune selection-mix collapse (RC1+RC3).** Selector was blind to modeled delay D (RC1) and a same-tuple
  double-pick starved the pool via a stale `_trainer_state_dict` prune (RC3, the dominant driver). Fix: stamp
  the speed signal from modeled duration in sim; re-key the no-repeat guard by `version_key`, pruned only on
  advance, not commit (K-D39).
- **§M `version_key` unification (K-D39).** version/staleness/no-repeat used 3 inconsistent shapes across
  trainer/aggregator/selector. Fix: one shared `version_key` property + vocabulary.
- **#7 fwdllm_plus eligible-count gap** (real 5.3 vs sim 10.0 mean eligible) — root-caused, NOT a sim bug: real's
  first reselect after a full-cohort dispatch always sees eligible=1 (drain-then-refill), sim's barrier drain
  always sees eligible=10. Real-transport artifact, no code fix needed.
- **`step_timing_breakdown` real-only-func exemption (K-D37).** Rung DIST-gated real-only sleeps as if shared
  compute. Fix: `_STEP_TIMING_REAL_ONLY_FUNCS` reported but `gates_ok=False`.
- **Stale SHARED-fail list retired.** `gpu_budget_*`/`utility`/`v5_variance_pass_ratio`/`overhead_residual`(sync)
  no longer reproduce — superseded by since-landed fixes; don't re-investigate unless they reappear.
- **Per-stage wall-budget instrumentation (K-D36).** No rung caught "sim a little slower at one stage." Fix:
  `drain_wall_budget`/`trainer_phase_wall_budget`/`step_timing_breakdown`/`aggregation_compute_wall`.
- **#N bin-7 checker fix (K-D35).** `cohort_sequence` conflated 4 targets under one bin cap. Fix: SET
  uncapped/HARD, CADENCE/VAR/ORDER capped to bin 1.
- **#12c sync `sim_rate` (delay-factor).** No delay-headroom starved the vclock. Fix: `--delay-factor 1` → sync
  `sim_rate` 0.94→2.9-3.0.
- **K-D31 validated.** Bin-1 cohort order is bit-exact once benign delay-ties are canonicalized.
- **Aggregator GPU pin (K-D33).** Aggregator eval contended with trainers on GPU 0. Fix: pin to a dedicated GPU.
- **#13 drain stall (K-D28/b/c).** A stuck straggler re-fired the full 30s `RECV_TIMEOUT` every cycle. Fix: felix
  stuck-end eviction + recv-grace + ready-gating → `sim_rate` 0.06→0.30, 30s stall gone.
- **#1c R1 two-ledger bridge (K-D27/b).** Selector eligibility never consulted the aggregator's virtual in-flight
  set. Fix: agg maintains `_sim_pending_commit`, bound live into the selector's filter.
- **#14 MQTT join-notify startup race.** `join()`'s fire-and-forget notify could drop before `on_connect`. Fix:
  `_wait_for_connect()` before subscribe+notify.
- **#6 clock-rate anchor (K-D25).** Checker compared sim-vclock against real's FULL wall (carries a localhost
  transport artifact). Fix: anchor real on its own `intrinsic_span_s`.
- **Phase-4 stopping rule (K-D24).** `sim_wall_ceiling_s` truncated a real-compute sim. Fix: decouple to
  `max_runtime_s × 20`.
- **Residence: commit-then-carry + felix realign (K-D12/14/17b).** Boundary-drop stranded async grads; slot-on-
  return undercounted in-flight. Fix: carry the surplus, hold the compute slot to COMMIT.
- **Clock family re-keyed to `data_id` (#2).** Rungs keyed on `round` mismeasured fwdllm's variance-pass progress.
- **Foundational sim mechanisms (K-D2-5/9/11).** `sct = send + gpu + D`, no-sleep sim path, sct reorder buffer +
  in-flight gate + agg-goal rollback cleanup, V1-V5/DK1-3/G1-2 rungs.
- **Scaffolding (landed, pointer only).** Pre-run instrumentation (K-D21); availability params end-to-end (K-D22);
  Phase-2 skips (K-D23); `staleness_policy` wired from config (K-D13/15).

---

## §H  Dead-ends & corrections — do NOT retry
- **"fluxtune's async cohort-SET divergence is an #N nondeterminism wall / a GPU-vs-D headroom collision, closed by
  more delay headroom."** REFUTED by the `run_20260710_2242` `--delay-divisor 0.25` diagnostic. With D≫gpu (sct
  104–144s vs gpu 3.77s) the SET still diverges at the same iter (~8), so headroom is not the cause; and the
  divergence is systematic + delay-coupled (`v2` gap 5%→11% as delay 0.5→0.25), not random fp16 jitter. Cohorts are
  bit-identical iteration-for-iteration — the real root is **#S1** (sim staleness under-modeling → variance bias →
  cadence desync → SET misalignment). *Lesson:* when async cohorts diverge but cadence/first-cohorts match, suspect
  the variance INPUT (staleness/grad values), not the selector; amplify with a diagnostic knob to expose it.
- **fluxtune commit-path stall — three superseded framings (same investigation; final root = COMMIT-PATH
  STALL, K-D34).** (1) "GPU-PIPELINING loss; keep gate, decouple dispatch" — WRONG, felix's arrival gate is
  INERT (gate_holds=0). (2) "re-dispatch on physical RETURN to keep GPUs busy" — WRONG, fedbuff never re-hands a
  returner the same version, and real's 3.37 concurrency is a duty cycle `gpu/max(gpu,D)`, not under-use. (3)
  "hold-to-commit is a SYNC barrier / over-restrictive; replace with model-advance re-dispatch" — WRONG,
  hold-to-commit is a CORRECTNESS check (freed only on commit; guards no-same-version / not-while-computing /
  not-while-returned-uncommitted; §B.1 row 6). *Final root:* the commit path STALLS (drain gate blocks real wall
  on PHANTOM `_sim_inflight_expected` entries) → correctly-held trainers idle. Fix = fast/non-stalling commit
  path; hold-to-commit untouched. *Lesson:* commit RATE is the throughput lever, not the residence rule.
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

## §K  Deviation log — one line per decision (anchor + rationale; referenced from §B.1/§G)
- **K-D1** — `time_mode` default `"real"` (not cifar's `"simulated"`): fwdllm's whole corpus is `real`; a
  `simulated` default risks half-activating an unbuilt path.
- **K-D2** — additive `sct = send + gpu + D` (not `max`): fwdllm's real mode slept D on top of GPU time.
  Reversed by K-D29 (remainder-wait) once determinism required it.
- **K-D3** — per-eval sct collapses to the train sct: eval lives on the aggregator; forward-grad "train" IS a
  forward pass (no 20× factor).
- **K-D4** — purpose-built `_sim_recv_min_grad` (not `_sim_recv_min` verbatim): cifar's per-commit release +
  withheld paths key on WEIGHTS semantics.
- **K-D5** — slot release + buffer clear on the AGG-GOAL boundary (not per-commit): a `data_id` spans many cycles
  with rollbacks; per-commit release would strand a re-contributing trainer.
- **K-D9** — cadence telemetry is a pre-mutation cycle snapshot: post-mutation `data_id` advances before emit,
  which would off-by-one V1.
- **K-D11** — `ends_not_selected_yet` "commit-1-per-pass" clamp gated real-only: a real-transport draining
  discipline; the sim barrier is single-pass.
- **K-D12** — fluxtune async: commit-then-CARRY the surplus + hold residence (reverses K-D6); drop stays correct
  for sync (c≈agg_goal).
- **K-D13/K-D15** — fluxtune `staleness_policy = fedbuff` staleness-weighted accept (was silently `none`); set
  identically real+sim.
- **K-D14** — R1/W1 sourced from an ECHOED per-contribution interval, not the agg's per-end dispatch stamp
  (overwritten on re-dispatch, exactly when residence breaks).
- **K-D17b** — hold the compute slot to COMMIT (felix port): `len(selected_ends)` = virtual-time in-flight.
  Confirmed correct for BOTH sync and async (K-D34) — a correctness check, not a throughput lever.
- **K-D21** — pre-run instrumentation A–E landed; un-skipped the 12 rigor-gap rungs.
- **K-D22** — availability params respected end-to-end; Phase-1 syn_0 default; print==run.
- **K-D24** — Phase-4 ceiling decouple (×20) + B1/B2 sct folds; fixed root S1.
- **K-D25** — #6 was a CHECKER-ANCHOR bug, not a sim under-charge: real's clock-rate rungs carried a localhost
  transport artifact. Fix: agg emits `intrinsic_span_s`, rungs anchor real on it.
- **K-D26** — #1c R1 root-caused to a physical-wall vs vclock desync in `async_oort`'s re-pick guard
  (`all_selected`). Fix: `_abandon_clock_now()` runs the 90s timeout on the vclock in sim. Corrected K-D19;
  sustained NONE-delete path → deeper root K-D27.
- **K-D27/b** — the two-ledger split: async_oort's selection eligibility never consulted the aggregator's virtual
  in-flight truth. Fix: agg maintains `_sim_pending_commit` felix-style, bound live to the selector's filter;
  `outstanding = inflight ∪ buffer` (K-D27b — NOT `− _sim_committed`, which dropped re-dispatched-after-commit
  trainers).
- **K-D28/b/c** — #13 drain-stall felix port: stuck-end eviction + recv-grace floor + probe-ceiling/ready-gating +
  direct `drain_ready` ingest. `sim_rate` 0.06→0.30, 30s stall gone.
- **K-D28d** — #13 step-4 freed-slot staggered re-dispatch: IMPLEMENTED but NEUTRAL → flag DISABLED. Residual
  holds are inherent strict-sct-order straggler waits, not a dispatch artifact.
- **K-D29** — REMAINDER-WAIT delay model (reverses K-D2): deterministic commit order needs a deterministic
  per-trainer arrival order, which flat-additive `gpu+D` didn't give. Fix: real sleeps `max(0, D−gpu)` so device
  wall = D (GPU hidden); sct = `send + max(gpu, D)`; order = D-order = real↔sim identical. Overrun (gpu>D) flagged
  as `training_overran`.
- **K-D30** — full-cohort determinism gate + `timing_overrun` signal: selection set/sequence rungs gated to a
  trivial pass for every stochastic selector, so fwdllm's syn_0 selection was never checked. Fix: enforce EXACT
  whenever `num_chosen==num_candidates`.
- **K-D31** — canonical `(D, trainer_id)` cohort commit order: two trainers sharing a registry delay could swap
  receive order (benign, same split-half) but flagged the exact-order rung. Fix: canonicalize by `(D, trainer_id)`.
- **K-D32** — fluxtune JVP perf optimizations (`jvp_perf_opt`, §L; config-gated, bit-identical): trainable-only
  finite difference + drop 3 diagnostic-only forward passes + reuse cached JVP → fluxtune −37% compute.
- **K-D33** — aggregator GPU pin: eval defaulted to GPU 0, contending with trainers. Fix: pin to a dedicated
  (idle/least-loaded) GPU.
- **K-D34** — fluxtune commit-path stall (supersedes 3 mis-framings, §H): the drain gate blocked real wall on a
  PHANTOM `_sim_inflight_expected` entry (stamped-expected-at-dispatch but idle-in-recv). Fix
  (`sim_compute_truthful_gate`, default off = byte-identical): skip any expected entry whose last dispatch is
  older than `sim_gate_compute_cap_s`. Validated: `STUCK_EVICT=0`, 30s failsafes gone. Residual (vclock omits
  `aggregate()` compute wall; GPU≈D no-headroom) re-diagnosed by K-D38 as headroom, not a vclock fold.
- **K-D35** — bin-7 checker fix: `cohort_sequence_parity` conflated SET/CADENCE/VAR/ORDER under one bin cap,
  demanding exact cadence past the nondeterminism wall. Fix: SET stays HARD/uncapped, CADENCE/VAR/ORDER cap to
  bin 1. n=10 pair shows the var-VALUE wall is n-scale-sensitive, not fixed at ~bin 7.
- **K-D36** — per-stage wall-budget instrumentation: no rung caught "sim a little slower at one stage" (only
  whole-run `sim_rate`/two-sided KS). Fix: `drain_wall_budget`/`trainer_phase_wall_budget`/`step_timing_breakdown`/
  `aggregation_compute_wall`, one-sided or distributional per phase.
- **K-D37** — `step_timing_breakdown` real-only-func exemption: the rung DIST-gated real-only sleeps
  (`_emulate_training_delay`/`pause_execution`) as genuine shared compute. Fix: report but don't gate on them.
- **K-D38** — fluxtune `sim_rate<1` root RE-DIAGNOSED = GPU-vs-D headroom (supersedes K-D34's "vclock fold"
  framing): min registry delay (4s) ≈ JVP GPU (3.86s) → 11% overrun → async fastest-3 cohort is a GPU coin-flip,
  not deterministic D. Fix = config headroom (`--delay-divisor 0.5 --num-gpus 10`); principled fix = a real
  LLM-mobile runtime trace (§A Next roots #1). Also: `training_delay_factor` clarified as a DIVISOR (<1
  lengthens, was documented backwards). **PARTLY SUPERSEDED (see #S1/§H):** the `run_20260710_2242` 0.25 diagnostic
  showed that at 0.5 headroom is already adequate and the cohort-SET divergence is NOT a D-collision coin-flip but
  the #S1 staleness bug; the compute-floor + divisor-clarification parts of K-D38 still stand.
- **K-D39** — §M `version_key` unification: version/staleness/no-repeat were named + compared inconsistently
  across trainer/aggregator/selector/async_cifar10. Fix: one shared `version_key` property
  (`syncfl/top_aggregator.py` base → `(round, 0)`; fwdllm overrides → `(model_version, iteration_per_data_id)`;
  asyncfl inherits it unoverridden) — folds RC3's 3-tuple guard into `(model_version, iteration)` once
  `model_version` bumps unconditionally per data-bin (`inc_model_version_per_data_id` purged). Also fixed a
  latent cross-round dedup false-abort in the trainer, and added the same optional no-repeat plumbing to sync
  `oort.py` for symmetry (left unwired — no driving bug there). Landed + tested.

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
