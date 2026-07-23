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
| `trainer_phase_wall_budget` | 4 | MECHANISM | one-sided per phase | trainer-side dispatch/local-copy overhead sim should collapse; `mqtt_fetch_s` reported (`gates_ok=False`) — apples-to-oranges real network I/O vs sim in-mem cache | — |
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
| `r1_inflight_overlap` (R1) | 3 | MECHANISM | — | overlapping dispatch→commit intervals per trainer — a real one-in-flight violation vs sim's slot-hold model | `participation` |
| `w1_compute_conservation` (W1) | 3 | DIAG | — | forward-pass count vs committed-grad count conservation | `r1_inflight_overlap` |

**`cohort_sequence.count` on the logical axis (2026-07-23):** the raw full-run cohort count
(`len(rc_full)` vs `len(sc_full)`) false-failed whenever one mode's run was simply still going past
the other's shared prefix (fwdllm 109 vs 118, ~7.5%, despite `composition` reading a perfect 1.0).
Now filtered to the matched LOGICAL budget N (`_matched_logical_budget`, cohorts whose progress
`<= N`, the common data_id prefix) — **not** a clock window V, which conflated sim's vclock with
real's wall (PARITY.md §1.5). On this axis `count` is rolled-up V1 (aggregation cycles per N
data_ids): it fires only when a side does MORE cycles to reach the SAME data_ids — a same-sign drift
V1's per-unit distributional tolerance absorbs — so it deps on `v1_iter_per_data_id` (a fail with V1
failing is downstream). See `TestCohortSequenceCountMatchedBudget` in `tests/mode/test_parity_checks.py`.

## Adding a new rung
1. Implement in `checks.py`, register in `CHECK_META` (stage/role/deps) and wire into
   `run_all_parity`.
2. Ship telemetry (if new) + its `analyze_run.py` plot + pytest in the SAME change (PARITY.md
   principle #8 — a field with no reader is dark data).
3. Add its table row here (implementation-level) and, once it stabilizes across a real run, to
   PARITY.md §2/§F (methodology-level catalog).
4. If it reuses `pctl_band_ok`, read the calibration note above before picking `min_abs`.
