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
- **C5 Loop per run.** Regrade both nodes and rewrite the scoreboard; group each cell's lowest red by root; apply C1,
  C6-C8; the next launch states what confirms and refutes each fix.
- **C6 Correct first, equal second.** Decide which side is wrong from first principles (FL semantics, `third_party/`,
  the stated invariant). A check green because both sides share a bug is a bug.
- **C7 Instrument what you can't see.** When logs can't name a root, add the telemetry in the same change as the next fix.
- **C8 Launch only when fixes are maximized:** every red cell has a fix in tree or an item the run will answer, full
  pytest green, smoke green (R19).
- **C9 Correct fixes default on (operator)** with a knob to revert; R9 default-off is for unproven changes.

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
- Pull a node's ladder to jayne:
```
R=/home/dgarg39/flame/lib/python/examples; N=kaylee; LAD=<ladder_dir>
rsync -a $N:$R/experiments/$LAD $R/experiments/${LAD}_$N/
ssh $N "cat $R/experiments/$LAD/*/*/*/runs/*/legs.txt" | sort -u > /tmp/legs.txt
rsync -a --files-from=/tmp/legs.txt -r $N:/ /
```

---

## Felix scoreboard — short runs, 2026-09-30

T3 syn_0/syn_0b/mobiperf × six × both datasets (`pool_fixB_verify_s1`, `_s2_kaylee`, `pool_clamp_verify_*`):
K3b green on all 36 cells (max rel 0.038), EV green on all legs. Open reds: A6r on oort/oort_star/refl mobiperf
(PR5, fix in tree). Run 4's board (`ladder_20260929_*`) is superseded.

Landed 2026-09-30 (each default on, knob reverts): async sim commit order exact (`simOrderSlackSeconds` 0, EV11
absolute); sim charges profiled per (dataset, harness, stack) from real legs (`profile_felix_charges.py` →
`async_cifar10/sim_charge_profiles/`, `simDispatchLatencySeconds`, `SIM_CHARGES=legacy`); real async frees an
ingested slot at SEND (`release_recvd_at_send`); clock clamp uses an arrived end's exact sct; checker times real from
run start (K2/K8/U2).

## FluxTune scoreboard
Parked with the track; its board still sits in simulate_fwdllm.md §A.

---

## Next steps (run queue — top item is next)

- **PR5 · Real trainer trace clock starts at join · fix in tree, verify overnight.** An undispatched real trainer
  never started its trace clock (origin rode the first task) → A6r red on sync mobiperf. The aggregator now sends the
  origin at the join barrier (`send_origin_at_join`; fwdllm off). *Confirms:* A6r green on oort/oort_star/refl mobiperf.
- **PR4 · Overnight 2026-09-30: L1-L4 ∥ L5, split by node · running.** GPU charge profiles come from each node's
  run-4 L5 real legs (T9), hence jayne = cifar L1-L4 ∥ speech L5, kaylee = speech L1-L4 ∥ cifar L5 (~3.5h). Launch
  (repo root; `$L` as in Tools):
  ```
  # jayne
  $L --rungs L1-L4 --datasets cifar10 --keep-going --deadline-h 5
  $L --rungs L5 --datasets google_speech --keep-going --deadline-h 5
  # kaylee
  $L --rungs L1-L4 --datasets google_speech --keep-going --deadline-h 5
  $L --rungs L5 --datasets cifar10 --keep-going --deadline-h 5
  ```
  *Predictions:* L2/L3 stage 1 green on every cell; syn_50 EV10/EV11/EV16 green (→ delete FX-N38 `KNOWN`); EV17 green
  on real legs; P11a-c CAUGHT; L5 K3b green with the GPU profiles. *Refutes:* a GPU cell K3b red → re-profile from
  run 5's own L5 real legs.
- **PR6 · Regrade run 5 (C5), then L6 only if L2-L5 have no open shared root.**

**Blocked on the operator**
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
