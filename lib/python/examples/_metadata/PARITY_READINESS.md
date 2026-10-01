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
> 6. **Long-run phase (operator).** Once short runs stop finding roots, launch the needed long legs (e.g. ~3h oort
>    syn_50) together: several small-but-reasonable legs in parallel on the GPUs, to surface any remaining real↔sim
>    blockers at once.

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

## Felix scoreboard — run 6, 2026-10-01 (both datasets, L1-L5)

`ladder_20261001_051724` (L1-L4) and `ladder_20261001_051427` (L5), jayne. "regraded" = the stored pairs re-run through
this session's checker (FX-D33); predicted board until run 7 confirms (PR8 closed FX-N59/N60).

| rung | run 6 | regraded | reds → root |
|---|---|---|---|
| L1 sim EV | 22 / 2 / 0 | — | known: oort syn_50 EV1 (P3 oort) |
| L2/L3 pairs | 35 / 0 / 13 | 40 / 2 / 6 | fedbuff syn_50 + mobiperf ×2 datasets, felix cifar syn_50: FX-D34; feddance cifar mobiperf K2 9.5% (no floor, Q2); known: oort syn_50 timing FX-N62 |
| L4 campaign | 48 / 2 / 0 | — | known: speech P7o EV12 FX-N30; P11a-c CAUGHT |
| L5 GPU pairs | 22 / 0 / 2 | 22 / 0 / 2 | speech refl syn_20 FX-D35; cifar oort_star syn_20 K2 12.6% with A2c selected-speed bias on 35 rounds (G0C control, L6) |

Stage 2-3 (not gated, no floors): sync syn_50 A3 progress-binned trajectories follow the stall counts; A7 selection is
green after FX-D33.

## FluxTune scoreboard
Parked with the track; its board still sits in simulate_fwdllm.md §A.

---

## Next steps (run queue — top item is next)

- **PR9 · Run 7: full L1-L5, both datasets, jayne · next (operator, ~6.5h: run 6 L1-L4 6h26m ∥ L5 3h32m).** PR8/PR8b
  (`pool_fxn59_verify`, `pool_fxn63_verify2`) closed FX-N59/N60/N63 (FX-D34/D35); FX-D35 changes every sync real leg's
  receive path, so every cell reruns. Commands as run 6 (`$L --rungs L1-L4 …` ∥ `$L --rungs L5 …`, `$L` in Tools). *Predictions:* L2 red only on
  feddance cifar mobiperf (no floor); L5 red only on cifar oort_star (selection draw); ungated DIST: refl U6 (real 0.17s lag, sim 0). Then Q2 floors, then L6.

**Operator decisions / open**
- S5 knob layout: (a) adopted (operator 2026-10-01): `datasets.yaml` holds dataset defaults + `by_baseline` tuned values
  with sources; left: move the `fedbuff.py` server-lr table into config, and a tool printing the resolved matrix.

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
- Q3 · todo: `--grade` re-runs the checker on stored pairs (done by hand this session: `scripts/parity_check.py --real --sim
  --agg-goal --budget-s` per `*_grade/summary.txt`, ~5 min for 72 pairs at 20 parallel).
- Q4 · todo: runner reports the two axes per cell (today it gates timing INV/EXACT and leaves logical DIST ungated).
- Q5 · todo: per-cell scheduling (C3).
- Q6 · wip: `logical_diff.py`; next: per-round marginals for async pairs.

Run length follows the residual's shape (R6): telemetry 5-10 min · clock family 15-30 min · one mechanism 30 min ·
participation 60 min+ · convergence 2h+.
