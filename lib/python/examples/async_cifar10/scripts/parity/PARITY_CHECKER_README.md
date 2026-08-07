# Parity checker — implementation reference

Scope: **how `checks.py` grades a rung**, not the methodology or any example's live
status. For those, read first:
- [`../../PARITY.md`](../../PARITY.md) §1/§2 — the causal ladder, roles (CONTROL/
  MECHANISM/EMERGENT), stages (0–9), tiers (INV/EXACT/DIST/DIAG), dependency gating, and the
  async_cifar10 rung catalog. §F is fwdllm's rung catalog.
- `examples/fwdllm/simulate_fwdllm.md` — fwdllm's live pass/fail scoreboard and open issues.

This file exists because a class of rungs (wall-budget / step-timing / cohort-composition
checks, added after PARITY.md §F was written) shares one comparison primitive
(`pctl_band_ok`) across several files and isn't yet folded into PARITY.md's tables. Fold a
rung's entry into PARITY.md §2/§F once it stabilizes; this file is the interim + implementation
-level reference, and stays even after that (calibration notes belong here, not in the
methodology doc).

## Tiers, in one line each
- **INV** — sim invariant; a genuine impossibility if it fails (e.g. vclock going backwards). Hard FAIL.
- **EXACT** — tight tolerance on a quantity both modes should reproduce closely. Hard FAIL.
- **DIST** — distributional match (KS / percentile band / mean-rel); FAIL unless `--lenient`.
- **DIAG** — informational, feeds root-cause, never gates `overall_verdict`.

Roles: **CONTROL** (an upstream input matches — failing means fix the input model, not the
mechanism), **MECHANISM** (one transformation is modeled — the localized bug when its
upstream CONTROLs pass), **EMERGENT** (an aggregate outcome — never fix directly, walk down
to the lowest failing dependency). See PARITY.md §1 for the full definitions and the
dependency-gating (ROOT-CAUSE vs DOWNSTREAM) algorithm.

## The `pctl_band_ok` primitive
`pctl_band_ok(real, sim, qs=(50, 90, 95), tol_rel, min_abs)` — central + upper-percentile
band agreement for a WALL/timing distribution. Passes iff **every** quantile in `qs` agrees
within `tol_rel` (relative) OR `min_abs` (absolute floor). P99/max are reported as `p99_diag`
but never gate.

**Why it exists:** sim charges a *modeled* completion time and does its *real* compute
alongside it (PARITY.md §F-1). For genuine SHARED compute (both modes do the same work), the
sim host's raw physical wall can develop a fatter upper tail than real from GPU/memory
contention (many trainers' JVP compute packed concurrently) — a tail-sensitive stat (raw KS,
mean) then false-fails a distribution whose *center* is fine. `pctl_band_ok` grades the shape
that matters (central tendency + the near-tail) and explicitly ignores the far tail, which is
a contention artifact, not a parity gap. It is NOT for logical-determinism rungs (set/order/
cadence) — those need exact or tie-window logic, not a timing band.

**Calibrating `min_abs` — read this before adding a new call site.** `min_abs` must be picked
in the *metric's own scale*, not copied from another rung. A single sub-noise floor
(`band_min_abs_s=0.5`, copied from `drain_wall_budget_parity`'s ~1s-scale drain events) was
briefly the default for `_step_timing_compare` too, whose functions run in the 1–100ms range
— it silently absorbed genuine 5x regressions (0.01s → 0.05s, well within the 0.5s floor) and
several `TestStepTimingBreakdown` tests passed for the wrong reason (2026-07-23). Anchor
`min_abs` to an already-vetted "this magnitude is noise" constant for that metric family (here,
`_STEP_TIMING_NEAR_ZERO_ABS_DIFF_S = 3e-4`, the same floor the near-zero-mean special case
uses) rather than a fresh guess. When adding a `pctl_band_ok` call: write a test with a genuine
divergence sized just *above* your chosen `min_abs`, and confirm it fails — not just the happy
path.

## Wall-budget / timing rung catalog
Not yet in PARITY.md §2/§F. `[C]` = component of a composite rung's `components` dict, not a
separate `CHECK_META` entry.

| ID / key | Stage | Role | Tier | Isolates | Deps |
|---|---|---|---|---|---|
| `drain_wall_budget` | 6 | MECHANISM | EXACT (composite) | see components below | `vclock_telemetry`, `commit_visibility` |
| `[C]` barrier_wait_s | — | — | one-sided (`sim<=real*(1+tol)`) | real-only transport wait sim should collapse toward 0 | — |
| `[C]` drain_spread | — | — | one-sided | drain loop through an already-ready cohort runs wider in one mode | — |
| `[C]` drain_tail_s | — | — | DIST (`pctl_band_ok`) | SHARED cohort-merge replay compute — genuine work both modes do, modeled cost charged to vclock separately, so graded on shape not a one-sided budget. **Not exempted by "sim is faster"** — sim dropping to ~0 while real measures real compute is a miss, not health. | — |
| `trainer_phase_wall_budget` | 4 | MECHANISM | one-sided per phase | trainer-side dispatch/local-copy overhead sim should collapse; `mqtt_fetch_s` reported (`gates_ok=False`) — real `channel.recv()` wall wait in BOTH modes (not sim in-mem cache), apples-to-oranges because dispatch cadence/aggregator-side overhead aren't vclock-modeled | — |
| `step_timing_breakdown` | 4 | DIAG | DIST (`_step_timing_compare`) | fine-grained per-`@timer_decorator`-function GPU-compute decomposition, trainer-side. Genuine shared compute (mode-invariant per PARITY.md principle #1) — target is a match. `_STEP_TIMING_REAL_ONLY_FUNCS` / `_STEP_TIMING_OFF_CRITICAL_PATH_FUNCS` reported but excluded from gating (`gates_ok=False`) | `phase_gpu_compute` |
| `agg_step_timing_breakdown` | 6 | DIAG | DIST (`_step_timing_compare`, `mean_tol_rel=0.5`) | aggregator-side analog; wider mean tolerance — aggregator CPU/memory bookkeeping tracks ambient contention from sim's continuously-active trainer pool, not an algorithmic divergence | `aggregation_compute_wall` |
| `aggregation_compute_wall` | 6 | DIAG | — | rollup of aggregator-side wall spent computing (vs waiting) | `drain_wall_budget` |

`_step_timing_compare` (shared by both `step_timing_breakdown_parity` and
`agg_step_timing_breakdown_parity`) also SKIPs a function cleanly when both sides' P99 sit
below `_STEP_TIMING_DEGENERATE_MAX_S` (1ms) — timer-quantization dither, not a measurement —
and short-circuits to PASS on a near-zero point mass (`_STEP_TIMING_NEAR_ZERO_MEAN_S`,
`_STEP_TIMING_NEAR_ZERO_ABS_DIFF_S`) where KS/mean-rel are uninformative. A function passes on
`ks <= ks_tol` OR `mean_rel <= mean_tol_rel` OR `pctl_band_ok(...).ok`.

## fwdllm-specific rungs not yet in PARITY.md §F
(§F has V1-V5/DK1-DK3/G1-G2; these were added later — see simulate_fwdllm.md §G for when.)

| ID / key | Stage | Role | Tier | Isolates | Deps |
|---|---|---|---|---|---|
| `cohort_sequence` (L1) | 6 | EMERGENT | EXACT, scoped (see below) | the ordered per-aggregation logical sequence: SET/CADENCE/VAR/ORDER hard only through `max_bin` (default 1, the float-non-determinism wall); `composition` grades the FULL sequence distributionally (index-paired cohort-membership overlap, `composition_tol=0.8`); `count` compares cohort counts **filtered to the matched LOGICAL budget N** (`_matched_logical_budget`, progress <= N) — rolled-up V1, so it deps on `v1_iter_per_data_id` | `participation`, `inter_arrival_order`, `r1_inflight_overlap`, `v1_iter_per_data_id` |
| `r1_inflight_overlap` (R1) | 3 | MECHANISM | — | overlapping dispatch→commit intervals per trainer — a real one-in-flight violation vs sim's slot-hold model | `participation`, `retask_before_close` |
| `w1_compute_conservation` (W1) | 3 | DIAG | — | forward-pass count vs committed-grad count conservation | `r1_inflight_overlap` |
| `concurrency_cap` | 4 | CONTROL | INV, **per mode** | outstanding dispatched-not-committed ends vs that mode's own selector `c` | — |
| `retask_before_close` | 4 | MECHANISM | INV, **per mode** | dispatching an end that already contributed to the still-OPEN agg cycle | `concurrency_cap` |
| `overlap_factor` (K4) | 1 | MECHANISM | EXACT | **pipelining depth**: mean per-cycle barrier (`intrinsic_span_s`) ÷ mean per-cycle clock advance, per side, over the matched budget. The one rung that says WHY a throughput residual exists — `throughput`/`per_round_advance`/`overhead_residual` all restate the same clock-per-work number. Passes on abs band `tol` OR relative `tol_rel` (the factor is ~1.2 on async_cifar10, ~5 on fwdllm) | `trainer_speed`, `sim_commit_monotone` |
| `v2b_var_drift` | 6 | DIAG | — | is V2's `var` gap a LEVEL offset (per-cycle mechanism → chase it) or a PROGRESSIVE drift (diverging training trajectories → `var` is the readout, not the cause)? Bins the matched budget by progress ordinal and reports the sim/real ratio per bin + `trend_rho` | `v2_var_trajectory` |

**`overlap_factor` was promoted from DIAG (2026-07-30)** and its numerator
replaced. It previously divided `_per_round_max_speed` by `_per_round_advances` —
but that helper keys on FL `round`, which fwdllm holds static for a whole lap, so
on the `data_id` axis it collapsed to ONE entry holding the run-global max trainer.
Both modes then read an identical constant and the rung silently restated its own
denominator (`fluxtune` reported `sim_mean_speed_s == real_mean_speed_s == 97.92`).
Use `_per_cycle_barrier_s`, never `_per_round_max_speed`, for anything per-cycle.

**`cohort_sequence.count` on the logical axis (2026-07-23):** the raw full-run cohort count
(`len(rc_full)` vs `len(sc_full)`) false-failed whenever one mode's run was simply still going past
the other's shared prefix (fwdllm 109 vs 118, ~7.5%, despite `composition` reading a perfect 1.0).
Now filtered to the matched LOGICAL budget N (`_matched_logical_budget`, cohorts whose progress
`<= N`, the common data_id prefix) — **not** a clock window V, which conflated sim's vclock with
real's wall (PARITY.md §1.5). On this axis `count` is rolled-up V1 (aggregation cycles per N
data_ids): it fires only when a side does MORE cycles to reach the SAME data_ids — a same-sign drift
V1's per-unit distributional tolerance absorbs — so it deps on `v1_iter_per_data_id` (a fail with V1
failing is downstream). See `TestCohortSequenceCountMatchedBudget` in `tests/mode/test_parity_checks.py`.

**Index-paired IDENTITY gating for stochastic-async selectors (2026-07-23).** For an async,
stochastic-subset, path-dependent selector (fluxtune's `AsyncOortSelector`), the marginal cohort
slot is a physical-FIFO-arrival (real) vs modeled-sct (sim) BOUNDARY RACE that cascades: index-paired
membership decorrelates to the **independent-draw floor** (matched marginals, zero index-correlation;
`_independent_draw_overlap_floor`) while `participation_parity` (S2) still enforces the marginal
invariant. Observed overlap AT the floor ⇒ two independent samples of the same process, not a bias
(observed << floor would be a real anti-correlation). Index-paired identity is then unattainable
(0.8 target vs a ~0.24 floor at 100-trainer scale), so the identity checks GATE to diagnostic when
`is_async and not _selection_is_deterministic` — mirroring `selection_parity`/S1's stochastic gating:
- `cohort_sequence` (`identity_gated`): composition + first-bin SET → diagnostic; **COUNT stays
  enforced** (throughput), reporting `composition.independent_draw_floor` + `at_independent_draw_floor`.
- `trainer_speed_identity` (`utility.gated_stochastic`): per-trainer UTILITY (loss-on-current-model,
  path-dependent) → diagnostic; **`speed_s` (registry-assigned) stays enforced**; the utility
  DISTRIBUTION (`utility_parity`) is the criterion.
- `iters_per_data_id_moving_avg` / V1b (`ma_shadow_gated`): the MA-shadow bounds → diagnostic (the
  per-data_id retry sequence is decorrelated by the same cascade); the **cumulative-mean guard stays
  enforced** (catches a real systematic drift). Sync fwdllm (is_async=False) is never gated. Tests:
  `TestCohortSequence`, `TestTrainerSpeedIdentityGating`, `TestV1bItersMovingAvg` (stochastic cases).

**The two dispatch-loop tripwires (`concurrency_cap` / `retask_before_close`, 2026-07-29)** are
graded **per mode independently** off each side's own `redispatch_decomp` events — not as a
real/sim diff — and name the offending side in `offending_modes`. Both are single-side
decidable (PARITY.md §D-9): a mode that dispatches past its own `c`, or re-tasks a trainer
whose cycle hasn't closed, is broken on its own terms. `retask_before_close` is invariant 1 at
DISPATCH level and is upstream of `r1_inflight_overlap`, which grades CONTRIBUTIONS: those stay
clean even while dispatch violates, because the round-trip outruns the cycle — a green R1 does
**not** clear the dispatch path. When `retask_before_close` fails, the Stage-1 throughput rungs
(`throughput`, `per_round_advance`, `total_commits`) and the cadence rungs (`v1*`,
`cohort_sequence`) are downstream of it by construction; the dep edges aren't wired backwards
across stages, so read them in that order manually.

## The matched LOGICAL budget (`_matched_logical_budget`)

Every windowed rung fixes the WORK and measures TIME (§D-4). The primitive that
defines "same work" differs per axis, and the `data_id` axis is **not** a ported
copy of the `round` one:

| axis | N | `prog_fn` | why |
|---|---|---|---|
| `round` (async_cifar10) | `min(final_round)` | raw `round` | dense, monotone, self-verifying — a `round` event's presence proves the close |
| `data_id` (fwdllm family) | length of the position-wise **common prefix** of both sides' chronological commit sequences | 1-based ordinal in that prefix | see below |

Two independent reasons the `round`-style max-key ceiling is unsound on `data_id`:

1. **The key isn't monotone.** `round` bumps one bin BEFORE `cycle_data_id` wraps,
   so a lap runs `(1,148) -> (2,149) -> (2,0) -> (2,1)`. Sorting the tuple puts
   `(2,149)` *last* when it happened *first*, so `max()` returns the first bin of
   the new lap and `<= N` then admits the entire lap. Order by event `ts`
   (`_verified_progress_order`), never by sorting keys.
2. **"Reached the same key" ≠ "did the same work."** `felix_round` had 9 bins sim
   committed and real never did, all sitting under a budget the checker called
   matched — `total_commits` compared two different workloads and graded the
   resulting 4.7% (really sim's vclock deadline vs real's wall span) as a PASS.

The common-prefix construction fixes both: same bins, same order, same count on
both sides by construction. `matched_logical_budget_n` is therefore an integer
bin COUNT on the `data_id` axis, not a `"round:data_id"` string.

## Calibrating a DIST tolerance — measure the floor, don't pick a number

`examples/fwdllm/expt_scripts/replicate_floor.py` measures the pipeline's own
run-to-run spread from same-mode, same-seed, same-config replicate runs already on
disk. A tolerance at or below that floor grades noise and can never be closed by
any code change. Measured at n=100/3600s: committed bins **3.4–4.5%**, cycles
0.6–1.4%, iters/bin 2.8–3.9%, mean var 0.6–1.2%. That is why
`_THROUGHPUT_FAMILY_TOL_REL` is 0.08 and not the historical 0.05. Re-run it after
any change to scale, hardware, or run length — the floor grows as runs shorten,
which is also how you decide whether a short verification run can grade a rung at
all.

## Adding a new rung
1. Implement in `checks.py`, register in `CHECK_META` (stage/role/deps) and wire into
   `run_all_parity`.
2. Ship telemetry (if new) + its `analyze_run.py` plot + pytest in the SAME change (PARITY.md
   principle #8 — a field with no reader is dark data).
3. Add its table row here (implementation-level) and, once it stabilizes across a real run, to
   PARITY.md §2/§F (methodology-level catalog).
4. If it reuses `pctl_band_ok`, read the calibration note above before picking `min_abs`.
