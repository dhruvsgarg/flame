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
- **C5 Loop per run.** Regrade the run, rewrite the scoreboard and the FELIX accuracy table (`accuracy_table.py`); group each cell's lowest red by root; apply C1,
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
- **C11 New features go short, small, parallel first (operator).** Streaming and unavailability are screened with short
  legs (≤ 15-30 min), small cohorts (n ≈ 50, 1-2 GPUs) and many legs packed in parallel; long, low-parallelism GPU legs
  only confirm what the screen already passed.
- **C12 Early termination only by rule (operator).** A leg is killed early only when it stops progressing: S1 no new
  committed round for `--stall-min` (15) min, S2 no log growth for 10 min (FX-D45); fatal lines abort the pool (FX-D22).
  A slow but progressing leg (low sim_rate, long rounds) is never cut.

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

## Felix scoreboard — runs 12-14 (`experiments/block_20261005_0401`), regraded 2026-10-05 with FX-D50-52

Green / known / red on INV/EXACT (`parity_ladder.py --grade <pool> --regrade --max-stage 9`; DIST gated only by floors, Q4).

**Settled (earlier runs).** L1-L5 (run 8, `ladder_20261001_164350`/`_164408`): 22/2/0 · 43/3/2 · 48/2/0 · 21/0/3, reds = no-floor tolerances.
L6 G0C + G0 (`ladder_20261002_033358`): 35/0/1 (speech oort syn_20 trainers_at_n 49 vs 46). Real↔real K2 spread at syn_0: cifar ≤ 5%,
speech ≤ 3% except feddance 17-19%. Cifar L7 at n=300 syn_0, 5/0/0: G1 felix/fedbuff (`ladder_20261002_203715`) K2 ≤ 0.3%, U3 equal to 3 dp,
sim 4.5×/7.4× real; G1L felix 7500s (`block_20261004_0543/g1l`) C1/C2 acc diff 1.09/1.11% vs real↔real 1.29%; G2 refl/oort/feddance
(`pr12_cifar_20261003_0322`, `block_20261003_1311`) green, feddance on its sim↔sim floor (FX-D37). N62 oort syn_50 3h CPU 69/69 (FX-D48).

| run 12-14 pool | g / k / r | reds and roots |
|---|---|---|
| `gs` G1S felix+fedbuff (n=100, 45 min, D ×5) | 2 / 0 / 0 | K2 0.4/1.3%, U6 0.08 s, speed identity green, queue_wait p99 0.17/0.30 s; no OOM (FX-D43) |
| `gs_g2` G2 oort+refl (90 min, D ×5) | 2 / 0 / 0 | DIST: refl U6 0.60 s, ingest 168 vs oort 35 ms/update (FX-N70); oort 2/47 speed identity (FX-N73) |
| `cifar` G0U (n=50, 15 min) on `cifar_uc` G0UC floors | 9 / 0 / 3 | fedbuff syn_50 sim EV16, fedbuff mobiperf K4 (both FX-D50); oort mobiperf EV1, 2 rounds in both modes (FX-D52) |
| `gs` G0U (n=50, 30 min) | 0 / 0 / 4 | felix + fedbuff sim EV16 → clock family (FX-D50); fedbuff mobiperf 5 vs 8 rounds (FX-D52) |
| `cifar_t0` / `cifar_t50` G0T, `cifar_o` G0To | 10/2/0 · 9/0/3 · 8/0/0 | feddance syn_0 lock-in KNOWN (adj. residual ≤ 0.04 s); fedbuff syn_50 EV16 (FX-D50); oort_star lin syn_50 K3b 21.8 vs 16.6 s (FX-N9) |
| `gs_t` G0T felix syn_0 | 1 / 0 / 1 | lin sim EV10: gate failsafe dropped a 50 s-wall task (FX-D50) |
| `cifar_g1u` G1U | aborted | CUDA OOM at trainer init, n=300 on 3 GPUs (FX-D52) |

G0UC real↔real floors at n=50 syn_50 (throughput rel): oort 0.90, refl 0.20, fedbuff 0.18, feddance 0.13, oort_star 0.06, felix 0.02; mobiperf
feddance 0.56, others ≤ 0.07. Unaware/sync cells are stall-placement chaos at this size (FX-L53).

**Accuracy (correctness, not parity):** see FELIX_READINESS → Accuracy table. Streaming-era rows: only speech oort reached its target
(60.7% at 80 min); speech felix/fedbuff sat at chance (server lr, fixed); refl had 0-5% fresh commits (FX-D53). Run 15 G1A gives full-data rows.

## FluxTune scoreboard
Parked with the track; its board still sits in simulate_fwdllm.md §A.

---

## Next steps (run queue — top item is next)

- **PR16 · Run 15, ~6h on jayne (8 GPUs) · ready: `lib/python/examples/scripts/run_block_20261006.sh`** (smoked: `smoke_run15b_*`; Ctrl+C
  stops it). A (~45 min, CPU): T3 felix/fedbuff syn_50 + mobiperf (FX-D50) ∥ T3 refl syn_0 + syn_50 (FX-D53), both datasets. B (~5h, no
  start past T0+4.4h): G1A (reference n, syn_0, full data, 90 min) — cifar GPUs 0-3: felix → refl; speech GPUs 4-7: felix → refl → fedbuff.
  *Confirms FX-N74:* speech felix ≥ 30% by 60 min and rising on both sides (G1S was 3.7%); cifar felix ≥ 50% by 90 min on full data; refl
  never degrades and > 10% of its commits are fresh. *Refutes:* speech felix flat → the async step (rate, staleness) not the lr; refl
  accuracy still falls → the stale weighting itself. *FX-D50:* T3 EV16 commit_late 0, K11 green. *FX-D53:* refl rounds close on K fresh
  (agg_round staleness has ≥ K zeros). Afterwards: `scripts/accuracy_table.py <OUT>/*` fills the accuracy table; regrade each pool.
- **PR17 · next:** GPU G0U/G0UC confirms of FX-D50 (fedbuff, oort mobiperf), G1U at n=300, speech oort/feddance/fedbuff G1A rows.

**Operator decisions / open**
- Decided (operator 2026-10-05): cifar GPU legs at 0.2 CPU/trainer, sized per baseline (FX-D49); the second block of a run starts
  automatically after the screens (operator analyzes separately).
- Decided (operator 2026-10-05): streaming design → FELIX Active build (streaming).
- Decided (operator 2026-10-04): jayne has 7 healthy GPUs; fix the real aggregator, not the wire format (FX-D44); speech D ×5 (FX-D46); C11, C12.
- Open: FX-N73 — should selector speed include the transfer leg in sim, as real Oort's duration does?
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

- Q2 · wip: `--control`/`--floors` + `parity_ladder --regrade` feed G0C (syn_0/20) and G0UC (G0U, G0T) real↔real floors into `floor_gated_tol`
  (SKIP only, never tighten: n=2, T8). Next: ≥3 legs per cell, then tighten + gate DIST.
- Q4 · todo: runner reports the two axes per cell (today it gates timing INV/EXACT and leaves logical DIST ungated).
- Q5 · todo: per-cell scheduling (C3).
- Q6 · wip: `logical_diff.py`; next: per-round marginals for async pairs.

Run length follows the residual's shape (R6): telemetry 5-10 min · clock family 15-30 min · one mechanism 30 min ·
participation 60 min+ · convergence 2h+.
