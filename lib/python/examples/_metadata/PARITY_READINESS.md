# Parity readiness — real↔sim parity, Felix and FluxTune

> **C0 · Rule zero: extract, then climb (operator). Time is the scarcest resource.**
> 1. **Mine what exists first.** Stored logs, telemetry and the code answer most questions. A launch that only
>    reproduces known failures wastes the time it takes.
> 2. **Short runs to find and fix.** Debug with the shortest run that shows the mechanism (single pairs, 3-5 min
>    legs, one or two shapes). Verify each fix across several short settings (both datasets, stacks, traces) before
>    believing it.
> 3. **Common roots first.** Group reds across baselines and cells by mechanism. One root that turns many cells
>    green comes before any single-cell fix.
> 4. **Lower rungs clean before higher ones.** A higher rung launches only when the lower rungs have no open shared
>    root. Higher rungs inherit lower roots and compound them into noise.
> 5. **Long runs only to confirm, or for effects that exist only at length** (convergence, drift, compounding).
>    Launch one only when confident; state what confirms and what refutes.

## Preamble
- **Read [ROBUST_FL_READINESS.md](ROBUST_FL_READINESS.md) first** (doc rules, R-rules, shared L/T). This doc owns
  real↔sim parity: climbing rules, method, axes, tools, the ladder, each track's scoreboard and the **run queue**
  (Next steps below). Track docs ([FELIX_READINESS.md](FELIX_READINESS.md), [FLUXTUNE_READINESS.md](FLUXTUNE_READINESS.md))
  keep their `FX-N`/`FT-N` items, lessons and built features.
- **Deprecated parity docs** ([PARITY.md](../async_cifar10/PARITY.md), [simulate_fwdllm.md](../fwdllm/simulate_fwdllm.md),
  [PARITY_CHECKER_README.md](../async_cifar10/scripts/parity/PARITY_CHECKER_README.md)) only shrink.
- **IDs:** `C#` climbing rules (C0 above all) · `Q#` ladder tasks · `PR#` run-queue steps.

---

## Climbing rules (operator)

- **C1 Every run closes roots.** Before the next launch, trace every red cell from stored logs to a root; each root
  gets a fix in tree or an item with a hypothesis and a prediction.
- **C2 Lowest red check per cell first.** Higher fails are presumed downstream. A root failing 2+ cells outranks one.
- **C3 No cell blocks another within a rung.** Climb per cell (baseline × dataset × avail/unavail); always
  `--keep-going`. Across rungs, C0.4 governs.
- **C4 Logical before timing.** Logical = same steps, same order, timestamps ignored. Timing closes through profiled
  charges, never knob tuning (T3, FX-T31). A logical miss on a timing-red cell is first checked for clock coupling.
- **C5 Loop per run.** Regrade the run and rewrite the scoreboard; group each cell's lowest red by root; apply C1,
  C6-C8; the next launch states what confirms and refutes each fix.
- **C6 Correct first, equal second.** Decide which side is wrong from first principles (FL semantics, `third_party/`,
  the stated invariant). A check green because both sides share a bug is a bug.
- **C7 Instrument what you can't see.** When logs can't name a root, add the telemetry in the same change as the next fix.
- **C8 Launch only when fixes are maximized:** every red cell has a fix in tree or an item the run will answer, full
  pytest green, smoke green (R19).
- **C9 Correct fixes default on (operator)** with a knob to revert; R9 default-off is for unproven changes.
- **C10 Resolve, don't just file (operator).** Every session (and every wait on a run) picks up as many independent
  Next-steps items as it can and drives each to a fix in tree or a closed item; a new item is filed only for a root found
  this session that can't be fixed in it. Report items closed vs opened at the end of each session.

---

## Method

- **Pipeline** `clock → availability → selection → dispatch/train → return/order → aggregation → utility → emergent`;
  the root is the **lowest failing check whose inputs match**. Checker stages (`CHECK_META`): 0 telemetry, 1 clock,
  2 availability, 3 selection, 4 dispatch, 5 return/order, 6 aggregation, 7 utility, 8 emergent, 9 budget.
- **Tiers.** INV/EXACT hard fail · DIST fails only above a replicate floor (L12, Q2) · DIAG informational.
- **Growth rule.** Every root leaves behind the finest check that would have localized it.
- **Logical budget (L10)**; stochastic selectors graded on marginals (L14).

| axis | graded on | checks |
|---|---|---|
| **Logical** (first) | EV (both legs) + time-stripped quantities | `eligibility` (+ `excluded_by_{real,sim}`), `selection_detail`, `participation`, `selection_bias`, `residence`, `agg_goal_cycles_*`, `aggregation_sequence`, `staleness`, `withheld_delivery`, `utility`, `convergence*`; EV17 |
| **Timing** (after) | stage-1 clock + time-to-N | `overhead_residual`, `per_round_advance`, `throughput`, `overlap_factor`, `matched_budget_coverage`, `total_commits`, `terminal_state`; `agg_timing_split` (DIAG) |

A timing red on a pair with < ~20 commits is noise.

## Tools

```
L="conda run --no-capture-output -n dg_flame python lib/python/examples/scripts/parity_ladder.py"
$L --rungs L1-L4 --datasets all --keep-going --deadline-h 6     # CPU
$L --grade <pool> --max-stage 1                                  # offline re-gate
python lib/python/examples/scripts/logical_diff.py <pool>        # first diverging selection/commit per pair
conda run -n dg_flame python lib/python/examples/async_cifar10/scripts/parity/event_invariants.py <run_dir>
```
- Pools on one node are safe (`harness_pool.Leases`, L28); progress prints every 5 min and after each leg (R23).
  One Ctrl+C tears everything down (the ladder waits for the pool's teardown).
- All runs on jayne only (parent R4).

---

## Felix scoreboard — run 5, 2026-09-30 (cifar L1-L4 only)

`ladder_20260930_054655` (jayne). Speech L1-L4 and both L5 never ran (kaylee launches absent; jayne L5 not relaunched).

| rung | cifar | reds → root |
|---|---|---|
| L1 sim EV | 11 green, 1 known (oort syn_50 EV1) | — |
| L2/L3 pairs | 19/24 green; syn_0, syn_0b, mobiperf all green | syn_50 × felix/fedbuff/feddance/oort_star/refl stage-1 timing: felix FX-N54, fedbuff FX-N55+N56, feddance FX-N57, oort_star/refl matched-window 13-14% (whole run 6-7%; no floor, Q2) |
| L4 campaign | 25/25 green; P11a-c CAUGHT | — |

Stage 2 on every syn_50 cell: A7 commit-belief + A2 (FX-N41); A5 was a checker artifact (fixed). fedbuff syn_50 EV14
both legs (FX-N15).

## FluxTune scoreboard
Parked with the track; its board still sits in simulate_fwdllm.md §A.

---

## Next steps (run queue — top item is next)

- **PR6 · Verify FX-N54/N55/N56 on short pairs · running.** `pool_fxn54_verify` (jayne): T3 felix + fedbuff × syn_50,
  syn_0 × both datasets, `max_experiment_runtime_s=600`. *Confirms:* felix syn_50 K2/K3b/K4 green and EV16 green; fedbuff
  picks UN_AVL on both legs, K2/K3b green; syn_0 unchanged. *Refutes:* any syn_0 red, or felix EV16 `commit_late` > 0.
- **PR7 · Run 6: L1-L4 ∥ L5, both datasets, jayne only · after PR6.** Speech L1-L4 and every L5 cell are unrun since the
  run-4 roots landed. GPU charge profiles: cifar from kaylee, speech from jayne (same hardware, T9). Launch (repo root; `$L` as in Tools):
  ```
  $L --rungs L1-L4 --datasets all --keep-going --deadline-h 6
  $L --rungs L5 --datasets all --keep-going --deadline-h 6
  ```
  *Predictions:* L2 syn_50 felix/fedbuff stage 1 green; feddance syn_50 stays red (FX-N57); speech mirrors cifar; L5
  K3b green with the profiles. *Refutes:* a GPU cell K3b red → re-profile from run 6's own L5 real legs.
- **PR8 · Regrade run 6 (C5), then L6 only if L2-L5 have no open shared root.**

**Blocked on the operator**
- FX-N57 sync wait-K: does a late (older-version) update count toward K? Recommended: no — K is the version's quorum of
  updates trained on it; a late one commits as a bonus (sim syncfl today); real syncfl and the oort stack then change.
- FX-N15 fedbuff server lr 40.9 (cifar, `fedbuff.py` table): keep or change (baseline-defining, T5).
- S5 knob layout: (a) `baselines.yaml` = what a baseline is; `datasets.yaml` gets `defaults` + per-baseline tuned
  values (trainer lr, batch, epochs, server lr, c, aggGoal, round_threshold); code tables move to config; a tool prints
  the resolved baseline × dataset matrix with each value's source and diffs it against a run's config. (b) the same
  in one new `knobs.yaml`. Recommended: (a).

---

## Active build — parity ladder (FX-N42)

`examples/scripts/parity_ladder.py` (`LADDER`; known misses `KNOWN`, each citing an item; real-only phases
`REAL_ONLY`). Each rung is one pool under `<out>/<rung>/`; `LADDER.txt` lists green/known/red per rung.

| rung | legs | est. wall (cifar / speech) |
|---|---|---|
| L0 | static: pytest collect, data, knob preflight (pool gate) | 4 min |
| L1 | T1: sim only, 120s, syn_0 + syn_50, EV | 6 / 6 min |
| L2 | T3: real+sim pairs × 4 shapes (oort mobiperf 960s) | ~70 / ~90 min |
| L3 | L2 re-graded to stage 3 | 0 |
| L4 | T4 extras: P4-P11c, EV | 37 / 48 min |
| L5 | GS: GPU pairs 10 min, G0 cohort, syn_0 + syn_20 | 122 / 238 min |
| L6 | G0C + G0: 30 min + real↔real control | > 6h per dataset |
| L7 | G1/G2 at reference n, 90 min → 3h | ~5h per dataset |

- Q2 · todo: real↔real floors into `floor_gated_tol` (parent S2) so DIST gates.
- Q3 · todo: `--grade` re-runs the checker on stored pairs.
- Q4 · todo: runner reports the two axes per cell (today it gates timing INV/EXACT and leaves logical DIST ungated).
- Q5 · todo: per-cell scheduling (C3).
- Q6 · wip: `logical_diff.py`; next: per-round marginals for async pairs.

Run length follows the residual's shape (R6): telemetry 5-10 min · clock family 15-30 min · one mechanism 30 min ·
participation 60 min+ · convergence 2h+.
