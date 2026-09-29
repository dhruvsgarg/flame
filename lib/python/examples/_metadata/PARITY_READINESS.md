# Parity readiness — real↔sim parity, Felix and FluxTune

## Preamble
- **Read [ROBUST_FL_READINESS.md](ROBUST_FL_READINESS.md) first** (doc rules, R-rules, shared L/T). This doc owns
  real↔sim parity: climbing rules, method, axes, tools, the ladder, each track's scoreboard and the **run queue**
  (Next steps below). Track docs ([FELIX_READINESS.md](FELIX_READINESS.md), [FLUXTUNE_READINESS.md](FLUXTUNE_READINESS.md))
  keep their `FX-N`/`FT-N` items, lessons and built features.
- **Deprecated parity docs** ([PARITY.md](../async_cifar10/PARITY.md), [simulate_fwdllm.md](../fwdllm/simulate_fwdllm.md),
  [PARITY_CHECKER_README.md](../async_cifar10/scripts/parity/PARITY_CHECKER_README.md)) only shrink.
- **IDs:** `C#` climbing rules · `Q#` ladder tasks · `PR#` run-queue steps.

---

## Climbing rules (operator)

- **C1 Every run closes roots.** Before the next launch, trace every red cell from stored logs to a root; each root
  gets a fix in tree or an item with a hypothesis and a prediction.
- **C2 Lowest red check per cell first.** Higher fails are presumed downstream. A root failing 2+ cells outranks one.
- **C3 No cell blocks another.** Climb per cell (baseline × dataset × avail/unavail); always `--keep-going`.
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

## Felix scoreboard — after run 4 (2026-09-29)

Sources: jayne `experiments/ladder_20260929_045715` (L1-L4), jayne `ladder_20260929_044231` (speech L5-L6),
kaylee `ladder_20260929_044146_kaylee` (cifar L5, partial L6: 9 real legs, stopped).

- **L1** green (known: oort T1 syn_50 EV1). **L4** green except injected gs_P11a (collision) and fedbuff P8 real EV17.
- **L2/L3/L5** logical: felix, fedbuff, feddance near green on both datasets. Remaining logical reds: real EV17
  (fedbuff/oort/oort_star syn_50) and speech sync oort/oort_star/refl `selection_detail`/`residence` (sim ~2 picks
  in flight at commit vs real ~1.1). Timing (`overhead_residual`, `throughput`) red almost everywhere (FX-N43).
- **Void legs (port collision, two ladders on jayne; fixed by leases):** gs_P11a sim, cifar T3_mobiperf_3st oort real,
  gs_GS_syn_0_refl sim, gs_G0C_syn_0 felix real. Don't read those cells from run 4.
- **L6** partial only: kaylee fedbuff cifar G0 syn_0 went NaN at round 150 (G0C twin finite; FX-N15). Convergence
  can't be judged below 90 min (heterogeneous start is slow; operator).

**Roots found in run 4 and fixed in tree (unmeasured until run 5)**

| symptom | root | fix |
|---|---|---|
| felix/fedbuff sim syn_50 EV10 (6/3 legs), dup buffer adds | a late eval reply freed a trainer whose newer train was in flight | eval branch keeps it busy (`_last_task_sent`) |
| same, plus EV11 past-dating (79/80 felix, 82/87 fedbuff while 2 tasks out) | due withheld update left `pending_withheld` at reinject, before commit; trainer re-picked | hold until commit (`sim_hold_withheld_until_commit`, default on) |
| EV16 delivery_ts mismatch (felix 0370) | eviction estimate kept for an update finished while available | deliver at true completion (`sim_withheld_true_delivery`, default on) |
| EV16 922 false unheld (hand re-grade) | checker read trace scale from its env | reads the run's `trace_time_scale` (L29) |
| real EV17, real pool < sim | real selector timeout stamp re-based as epoch; evicted end never popped | avail clock; `commit_withheld` on receipt |
| 4 void legs, gs_P11a join 3/15 | two pools took port 18830 two seconds apart | `Leases` + broker duplicate-id fail-fast |
| oort mobiperf graded nothing | 0 commits in 240s (unaware oort waits out 90s) | CPU oort mobiperf legs 960s; coverage checks SKIP on 0 commits |
| L6 G0C cells "sim MISSING" | report expected a sim leg | G0C graded real-only |
| Ctrl+C left GPU legs alive | `subprocess.run` SIGKILLed the pool after 0.25s | ladder waits for pool teardown |

**Measured, not a root:** speech sync real `aggregate()` 11.4s is recv wait for K; the commit itself is 0.065s
median (p90 0.077s). The speech sync in-flight gap needs another root; run 5's `agg_timing` + `excluded_by` answer it.

**New telemetry for run 5:** `agg_timing` per commit (recv wait / ingest / commit / other, both modes);
selection `excluded_by` + per-trainer `excl`; `[JOIN_BARRIER]` joins by id.

## FluxTune scoreboard
Parked with the track; its board still sits in simulate_fwdllm.md §A.

---

## Next steps (run queue — top item is next)

- **PR1 · Read the operator's smoke · todo.** The smoke ran on jayne after the run-4 fixes. Green = gate PASS and the
  felix pair commits on both legs on both datasets. Red → fix, re-smoke; nothing else launches.
- **PR2 · Full pytest (R10) · todo.** Four suites (parent doc); must pass before PR3.
- **PR3 · Run 5: 6h daytime, both nodes · todo (after PR1-PR2).** Hand the operator (from repo root):
  ```
  L="conda run --no-capture-output -n dg_flame python lib/python/examples/scripts/parity_ladder.py"
  # jayne (both lines in parallel; leases keep them apart)
  $L --rungs L1-L4 --datasets all --keep-going --deadline-h 6                 # CPU, ~3.5-4h
  $L --rungs L5 --datasets google_speech --keep-going --deadline-h 6          # 4 GPUs, ~4h
  # kaylee
  $L --rungs L5-L6 --datasets cifar10 --keep-going --deadline-h 6             # L5 ~2h, then L6 pairs until the deadline
  ```
  Before handing over: `--dry-run` each and confirm the makespan line; L6 cifar will not finish in 6h (pairs are
  ordered so the deadline drops whole pairs). *Predictions:* EV10/EV11/EV16 green on felix/fedbuff sim syn_50
  (T3, L1) → delete the FX-N38 `KNOWN` row; EV17 green on every real leg; no broker `already connected` anywhere;
  gs_P11a sim CAUGHT (EV10 FAIL, EV0 PASS, 15/15 joined); oort mobiperf ≥ 4 commits both legs; `agg_timing`
  commit_s ≪ recv_wait_s on every stack; `excluded_by` names which hold differs on feddance/oort_star syn_50.
  *Refutes:* any EV10 on syn_50 sims → a third re-pick path (read `[LATE_EVAL]` and `excl` of the re-picked end).
- **PR4 · Regrade run 5 (C5) · after PR3.** Rewrite the scoreboard above; roots → items; then FX-N43 timing.

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
