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
$L --grade <pool> --max-stage 1 [--regrade]                      # offline re-gate (--regrade re-runs the checker, ~30s/48 pairs)
python lib/python/examples/scripts/logical_diff.py <pool>        # first diverging selection/commit per pair
conda run -n dg_flame python lib/python/examples/async_cifar10/scripts/parity/event_invariants.py <run_dir>
python lib/python/examples/async_cifar10/scripts/parity_check.py --real <A> --sim <B> --control --json-out c.json   # real<->real floor (Q2); --floors <json> sizes a pair's gates
```
- Pools on one node are safe (`harness_pool.Leases`, L28); progress prints every 5 min and after each leg (R23).
  One Ctrl+C tears everything down (the ladder waits for the pool's teardown).
- All runs on jayne only (parent R4).

---

## Felix scoreboard — run 10b, 2026-10-04 (L7 cifar G1 + G2, speech G1 on top of run 8's L1-L6)

L1-L5 (`ladder_20261001_164350`, `_164408`, jayne): L1 22/2/0 · L2/L3 41/1/6 (regraded 43/3/2 with FX-D36 + FX-N62 KNOWN) · L4 48/2/0 ·
L5 22/0/2 (cifar felix/oort syn_0 K2 8.1/8.2%, opposite signs). Reds there are no-floor tolerances (Q2) or FX-N62.

L6 `ladder_20261002_033358` (G0C + G0, 691 min), green / known / red: 31/0/5 raw → **35/0/1** with FX-D37/D38 (floor SKIPs, stall-free K8, EV16 re-run offline, refl BN fix).

| cell | raw red | root |
|---|---|---|
| speech feddance syn_20 | K3b/K2/K3/K8 (real 28.1 vs sim 18.5 s/round) | real↔real itself spans 18.0-22.2 s/round at syn_0: selections match to round ~21, then utility feedback diverges; floor 17% swallows the 8% tol → SKIP |
| cifar + speech oort syn_20 | K8/U2 8.3% (sim 11 stalls, real 9) | 90s stall draw owned by K3s: K8/U2 now stall-free; cifar green |
| cifar oort_star syn_20 | sim EV16 | one withheld delivery at the 1800s budget committed 12.9s later by the final sync round: not late (FX-D37) |
| speech refl syn_0 (G0C real) | EV14 test-loss NaN at round 600 | fixed (FX-D38): BN `running_var` went negative from stale deltas; fresh-only BN stats |
| speech oort syn_20 | K8 trainers_at_n 49 vs 46 (6.1% vs 5%) | open: a 2-leg floor reads 0 (selections seeded), so no floor sizes a ±3 count (Q2, FX-N62) |

Real↔real K2 spread (G0 vs G0C real, one pair per cell: T8 lower bound): cifar ≤ 5% on all six; speech ≤ 3% except feddance 17-19%.
EV green on every leg bar the two above.

**L7 cifar G1** (`ladder_20261002_203715`, 273 min, syn_0, n=300, 5400s, 7 GPUs): felix + fedbuff **2/0/0**, 67/67 and 66/66 enforced checks, EV0-EV18 green both modes.

| check | felix | fedbuff |
|---|---|---|
| K2 / K3b / K3 / K8 | 0.1% / 0.001 / KS 0.03 / 0.2% (tol 8/10/20/8%) | 0.3% / 0.003 / KS 0.01 / 0.3% |
| S2 / S3/4 / A2c | KS 0.017 / 0.1% / KS 0.005 | KS 0.007 / 0% / KS 0 |
| U3 staleness mean | 2.894 vs 2.893 | 2.895 vs 2.894 |
| real queue_wait p99 / max | 0.127s / 0.37s (0 skips) | 0.054s / 0.12s (0 skips) |
| C1/C2 convergence | LOWC: acc diff 1.5%, loss 0.013 (15 evals) | LOWC: 1.1%, 0.014 (13 evals) |
| sim wall vs vclock | 1212s for 5405s (4.5×) | 727s for 5401s (7.4×) |

Not green or not graded: C1/C2 are low-confidence at a 5400s budget (checker floor 7200s, FX-N65); felix U5 WARN ρ = -0.20 (chronic on utility selectors
in L5-L7, selection-set noise, FX-N66); fedbuff sim aggregator exit abort (FX-N33, data intact). G2 (oort, oort_star, refl, feddance) did not run: the first
launch (`ladder_20261002_202440`, G1+G2, est. 15.3h > 12h deadline) was stopped after its gate.

**L7 cifar G2** (`pr12_cifar_20261003_0322` + run 10b `block_20261003_1311`, syn_0, n=300, 5400s), **3/0/0**:

| cell | result | evidence |
|---|---|---|
| refl | green, 0 fails; residence hist and carried ages = real to 2 dp | run 10 |
| oort | green 69/69: carried 3.71 vs 3.84, K2 1.0%, K3b 0.01, K8 1.6% (trainers 4.7%), A2c bias -2.53/-2.18 s, queue_wait p99 0.014s | `g2_oort`, FX-D39 confirmed |
| feddance | 58/59 on its floor: K2/K3/K3b/K8/U2/A2c SKIP; U6 2.1 s vs 2.0 s gate (sim↔sim 1.8 s) = KNOWN draw | sim↔sim s/round 24.0-27.75 (n=4), real↔real 28.10/28.55 (n=2); locked sets overlap 4-8/12 within either mode (FX-D37) |

oort_star syn_0 G2 dropped (graded with unavailability). C1/C2 LOWC at 5400s (FX-N65).

**L7 speech G1** (`block_20261003_1311/gs_g1`, syn_0, n=100, 5400s, 4 GPUs per leg), **0/0/2**:

| cell | red | root |
|---|---|---|
| fedbuff (64/66) | U6 real 0.80 s vs sim 0 (real queue_wait p99 5.9 s, 2330 waits > 1 s) | aggregator OMP stall: 0.33 s per 29 MB update at 2.4 updates/s (FX-D40, fixed; smoke p90 0.05 s) |
| fedbuff | speed identity: 8 trainers (D 2-3 s) +11-13% | per-commit real mean 3.58 vs 3.12 s on D=3: heavier real overrun tail (GPU excess 42 vs 20 s) + ~0.1 s post-compute span; hypothesis: aggregator spin on HT siblings of trainer cores (FX-D40). Checker now per commit |
| felix | real leg CUDA OOM at 9 min (memory 4 → 43 GB/GPU) | `evaluate()` kept the utility's autograd graph and chained it across evals (FX-D41, fixed) |

fedbuff otherwise green: K2 5.1%, K3b 0.051, K8 5.1%, U3 2.895/2.895, P3 support 0.998, C1 LOWC acc diff 0.5%.

## FluxTune scoreboard
Parked with the track; its board still sits in simulate_fwdllm.md §A.

---

## Next steps (run queue — top item is next)

- **PR13+14 · Run 11, one ~10.5h block on jayne · running (tmux `dg_flame`, `experiments/block_20261004_*`).** `lib/python/examples/scripts/run_block_20261004.sh` (smoked with `BLOCK_EXTRA=--smoke`; Ctrl+C stops it):
  A. speech G1 felix + fedbuff, then G2 oort/refl/feddance, 2 legs × 4 GPUs (~5h), beside cifar oort syn_50 3h CPU pair (`N62`). B. felix cifar syn_0 7500s pair +
  a real replicate (~4.8h), then `--control` floors and both reals graded against the sim.
  *Confirms FX-D40:* speech real queue_wait p99 < 1 s, U6 green, speech sim wall ≥ 3× faster than real; speed identity green (refutes the HT-sibling hypothesis if still red:
  then charge the post-compute span). *Confirms FX-D41:* felix speech real runs 90 min, GPU memory flat; `[TRAINER_HP]` lr 0.000195 (FX-N10).
  *Speech G2:* EV green, INV/EXACT green or floor-SKIP (feddance lock-in may need G2S). *N62:* oort syn_50 K2/K3b graded on ≥ 60 stall-free rounds.
  *G1L (FX-N65):* C1/C2 not LOWC; sim acc diff inside the real↔real spread.

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
| L7 | G1/G2 at reference n, 90 min → 3h | cifar G1 4.5h (done) + G2 ~9h / speech ~5h |

- Q2 · wip: `--control`/`--floors` + `parity_ladder` regrade feed G0C real↔real floors into `floor_gated_tol` (SKIP only, never tighten: n=2, T8). Next: ≥3 legs per syn_0 cell, then tighten + gate DIST.
- Q4 · todo: runner reports the two axes per cell (today it gates timing INV/EXACT and leaves logical DIST ungated).
- Q5 · todo: per-cell scheduling (C3).
- Q6 · wip: `logical_diff.py`; next: per-round marginals for async pairs.

Run length follows the residual's shape (R6): telemetry 5-10 min · clock family 15-30 min · one mechanism 30 min ·
participation 60 min+ · convergence 2h+.
