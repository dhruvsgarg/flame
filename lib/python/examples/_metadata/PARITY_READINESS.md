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
> 6. **Long-run phase (operator).** Once short runs stop finding roots, launch the needed long legs together:
>    several small-but-reasonable legs in parallel on the GPUs, to surface any remaining real↔sim blockers at once.

## Preamble
- **Read [ROBUST_FL_READINESS.md](ROBUST_FL_READINESS.md) first** (doc rules, R-rules, shared L/T). This doc owns
  real↔sim parity: climbing rules, method, tools, the pre-launch checklist, each track's scoreboard and the **run
  queue** (Next steps). Track docs ([FELIX_READINESS.md](FELIX_READINESS.md), [FLUXTUNE_READINESS.md](FLUXTUNE_READINESS.md))
  keep their `FX-N`/`FT-N` items, lessons and built features.
- **Deprecated parity docs** ([PARITY.md](../async_cifar10/PARITY.md), [simulate_fwdllm.md](../fwdllm/simulate_fwdllm.md),
  [PARITY_CHECKER_README.md](../async_cifar10/scripts/parity/PARITY_CHECKER_README.md)) only shrink.
- **IDs:** `C#` climbing rules · `PL#` pre-launch checks · `Q#` ladder tasks · `PR#` run-queue steps.

---

## Climbing rules (operator)

- **C1 Every run closes roots.** Before the next launch, trace every red cell from stored logs to a root; each root
  gets a fix in tree or an item with a hypothesis and a prediction.
- **C2 Lowest red check per cell first.** Higher fails are presumed downstream. A root failing 2+ cells outranks one.
- **C3 No cell blocks another within a rung.** Climb per cell (baseline × dataset × avail/unavail); always
  `--keep-going`. Across rungs, C0.4 governs.
- **C4 Logical before timing.** Logical = same steps, same order, timestamps ignored. Timing closes through profiled
  charges, never knob tuning (T3, FX-T31). A logical miss on a timing-red cell is first checked for clock coupling.
- **C5 Loop per run.** Regrade the run, rewrite the scoreboard and the FELIX accuracy table (`accuracy_table.py`);
  group each cell's lowest red by root; apply C1, C6-C8; the next launch states what confirms and refutes each fix.
- **C6 Correct first, equal second.** Decide which side is wrong from first principles (FL semantics, `third_party/`,
  the stated invariant). A check green because both sides share a bug is a bug.
- **C7 Instrument what you can't see.** When logs can't name a root, add the telemetry in the same change as the next fix.
- **C8 Launch only when fixes are maximized:** every red cell has a fix in tree or an item the run will answer, and
  the pre-launch checklist (PL1-PL8) passes.
- **C9 Correct fixes default on (operator)** with a knob to revert; R9 default-off is for unproven changes.
- **C10 Resolve, don't just file (operator).** Every session (and every wait on a run) drives as many independent
  Next-steps items as it can to a fix in tree or a closed item; a new item is filed only for a root found this session
  that can't be fixed in it. Report items closed vs opened at the end of each session.
- **C11 New features go short, small, parallel first (operator).** Streaming and unavailability are screened with short
  legs (≤ 15-30 min), small cohorts (n ≈ 50, 1-2 GPUs) and many legs packed in parallel; long, low-parallelism GPU legs
  only confirm what the screen already passed.
- **C12 Early termination only by rule (operator).** A leg is killed early only when it stops progressing: S1 no new
  committed round for `--stall-min` (15) min, S2 no log growth for 10 min (FX-D45); fatal lines abort the pool (FX-D22).
  A slow but progressing leg is never cut.

## Pre-launch checklist (operator: never repeat a failure; every long run passes all of these)

Each check names the incident it prevents. A new failure mode adds a check here in the same session.

| ID | check | how | incident |
|---|---|---|---|
| PL1 | full pytest green | ROBUST → Full pytest | — (R10) |
| PL2 | gate: collect + smoke pair per dataset | automatic in each pool | — (R19) |
| PL3 | **scale smoke**: every new GPU leg shape at its production n, c, GPUs and data for 5 min, with the lanes that will share the node running together; no OOM, ABORT, STALLED, MISSING, `GPU_TIGHT` / `RAM_TIGHT` (> 85% of a GPU / host RAM) | Claude, before hand-off (`harness_pool.py --scale-smoke 300`); the block's phase 0 repeats it for the densest leg and aborts itself | run 15: speech felix G1A OOM at 2.5 min (c=30, 45/45 GB); the 60 s n≤12 smoke can't see density. The scale smoke then caught 39.4 GB on 4 GPUs (FX-D58), an exit-time traceback (FX-D57) and host RAM at 100% (FX-D59: run 15's 70-min death was the kernel OOM killer, RAM + swap full at 20:48) |
| PL4 | detached launch | `run_block_*.sh --detach` (or inside tmux); stop `kill -INT -- -$(cat OUT/PGID)` | a lost terminal must not end a 10h run |
| PL5 | no code or script edits while a run is live | T15 | `pool_smoke_fxn52` ran a half-edited selector |
| PL6 | GPUs idle and ECC-clean, disk free | `nvidia-smi --query-gpu=index,memory.used,ecc.errors.uncorrected.volatile.total --format=csv`; `df -h` | S0 GPU 1 ECC |
| PL7 | a new knob reaches the trainer | dry-run + grep a short leg (L9) | FX-D32 |
| PL8 | +10 min after launch: `BLOCK.log` advancing, each pool's first PROGRESS line, no ABORT; a real asyncfl leg's aggregator telemetry grows < 100 MB/min | read `OUT/`; `ls -l <run>/telemetry/aggregator_*` | run 15 G1A sim OOM went unseen for 6h; run 16 real spin wrote 310 MB/min (FX-D60) |
| PL9 | every leg of the last block graded (no MISSING) before its numbers are cited | `SUMMARY.txt`; else `parity_check.py` by hand | run 16: 5 finished legs ungraded (FX-D62) |

## Method

- **Pipeline** `clock → availability → selection → dispatch/train → return/order → aggregation → utility → emergent`;
  the root is the **lowest failing check whose inputs match**. Checker stages (`CHECK_META`): 0 telemetry, 1 clock,
  2 availability, 3 selection, 4 dispatch, 5 return/order, 6 aggregation, 7 utility, 8 emergent, 9 budget.
- **Tiers.** INV/EXACT hard fail · DIST fails only above a replicate floor (L12, Q2) · DIAG informational.
- **Growth rule.** Every root leaves behind the finest check that would have localized it.
- **Logical budget (L10)**; stochastic selectors graded on marginals (L14). A timing red on < ~20 commits is noise.

| axis | graded on | checks |
|---|---|---|
| **Logical** (first) | EV (both legs) + time-stripped quantities | `eligibility` (+ `excluded_by_{real,sim}`), `selection_detail`, `participation`, `selection_bias`, `residence`, `agg_goal_cycles_*`, `aggregation_sequence`, `staleness`, `withheld_delivery`, `utility`, `convergence*`; EV17 |
| **Timing** (after) | stage-1 clock + time-to-N | `overhead_residual`, `per_round_advance`, `throughput`, `overlap_factor`, `matched_budget_coverage`, `total_commits`, `terminal_state`; `agg_timing_split` (DIAG) |

Run length follows the residual's shape (R6): telemetry 5-10 min · clock family 15-30 min · one mechanism 30-45 min ·
participation 60 min+ · compounding clock (K2/K3b) 90 min-2h · convergence / C1-C2 sign-off 2-4h.

## Tools

```
L="conda run --no-capture-output -n dg_flame python lib/python/examples/scripts/parity_ladder.py"
$L --rungs L1-L4 --datasets all --keep-going --deadline-h 6     # CPU
$L --grade <pool> --regrade --max-stage 9                        # offline re-gate (~30s/48 pairs)
python lib/python/examples/scripts/logical_diff.py <pool>        # first diverging selection/commit per pair
conda run -n dg_flame python lib/python/examples/async_cifar10/scripts/parity/event_invariants.py <run_dir>
python lib/python/examples/async_cifar10/scripts/parity_check.py --real <A> --sim <B> --control --json-out c.json   # real<->real floor (Q2)
```
- Pools on one node are safe (`harness_pool.Leases`, L28); progress prints every 5 min and after each leg (R23); each
  GPU leg's DONE line carries its peak GPU memory (FX-D56). One Ctrl+C tears everything down. All runs on jayne (R4).

---

## Felix scoreboard

Green / known / red on INV/EXACT (`--grade <pool> --regrade --max-stage 9`; DIST gated only by floors, Q4).

**Settled.** L1-L5 (run 8): 22/2/0 · 43/3/2 · 48/2/0 · 21/0/3, reds = no-floor tolerances. L6 G0C + G0: 35/0/1 (speech oort
syn_20 trainers_at_n 49 vs 46). Real↔real K2 spread at syn_0: cifar ≤ 5%, speech ≤ 3% except feddance 17-19%. Cifar L7 n=300
syn_0: G1 felix/fedbuff K2 ≤ 0.3%; G1L felix C1/C2 acc diff 1.09/1.11% vs real↔real 1.29%; G2 refl/oort/feddance green (feddance on
its sim↔sim floor, FX-D37). N62 oort syn_50 3h CPU 69/69. Speech G1S felix+fedbuff 2/0/0, G2 oort+refl 2/0/0 (DIST: refl U6 FX-N70,
oort speed identity FX-N73).

| pool (run) | g / k / r | reds and roots |
|---|---|---|
| run 16 A `block_20261006_run16/a_async`: T3 felix+fedbuff syn_50 + mobiperf, both datasets | 6 / 2 / 0 | FX-D50 holds; known = fedbuff mobiperf S1 timing (FX-N62) |
| run 16 A `…/a_refl`: T3 refl syn_50 + mobiperf, both datasets | 3 / 0 / 1 | **FX-D55 confirmed** (real gates 3 updates at the 150 s flip); cifar syn_50 S1/S8 red: a 0.45 s offset forks one task across the flip (FX-L53) → T3C floor |
| run 16 B G1A (accuracy, full data) | regraded by hand | **cifar felix 67/67 green** (K2 0.6%, both reach 50%); speech fedbuff K2 0.4%, K3b/U3 green, DIST phase timings + U6 red; speech refl green but U6 (FX-N70); speech felix sim EV14 = NaN eval (FX-D61); 5 legs MISSING: S2 cut in post-run analysis (FX-D62) |
| runs 12-14 G0U cifar (n=50) on G0UC floors | 9 / 0 / 3 | fedbuff syn_50 EV16 + mobiperf K4 (FX-D50, CPU-confirmed); oort mobiperf EV1, 2 rounds (FX-D52) |
| runs 12-14 G0U speech (n=50) | 0 / 0 / 4 | felix + fedbuff EV16 (FX-D50, CPU-confirmed); fedbuff mobiperf 5 vs 8 rounds (FX-D52) |
| runs 12-14 G0T cifar syn_0 · syn_50 · G0To | 10/2/0 · 9/0/3 · 8/0/0 | feddance syn_0 lock-in KNOWN; fedbuff EV16 (FX-D50); oort_star lin syn_50 K3b 21.8 vs 16.6 s (FX-N9) |
| runs 12-14 G0T speech felix syn_0 · G1U cifar | 1/0/1 · aborted | lin EV10 (FX-D50) · CUDA OOM at init, n=300 on 3 GPUs (FX-D52) |

G0UC real↔real floors at n=50 syn_50 (throughput rel): oort 0.90, refl 0.20, fedbuff 0.18, feddance 0.13, oort_star 0.06, felix 0.02;
mobiperf feddance 0.56, others ≤ 0.07. n=50 unaware/sync cells are stall-placement chaos (FX-L53). Accuracy: FELIX → Accuracy table.

## FluxTune scoreboard
Parked with the track; its board still sits in simulate_fwdllm.md §A.

---

## Next steps (run queue — top item is next)

- **PR17 · Run 17, ~5.5h on jayne · `run_block_20261007.sh` in tmux `dg_flame`** (phase 0 smoked: `smoke_run17_s0`, cifar
  pair 1.0, real telemetry 46 MB/min; speech sim EV12 = 5-min wall ceiling, as run 16). 0 (40 min): PL3 felix scale smoke, aborts the block.
  A (~25 min, CPU): T3 + T3C refl syn_50, T3 felix+fedbuff syn_0, both datasets. B: G1A speech felix + fedbuff pairs (~3h, 2 legs at a
  time), then G1AS cifar refl + fedbuff sim legs (~45 min, graded by hand against run 16's real legs). No leg starts past T0+5.8h.
  *Confirms:* FX-D60 — real felix/fedbuff aggregator telemetry ≈ G1L's 50 MB/min, `slot_starvation` < 5/s, speech U6 smaller;
  FX-D61 — 0 `[FEDBUFF_BN]` clamps, finite test loss, speech felix ≥ 30% by 60 min and rising; FX-D62 — no MISSING/STALLED after `run_end`;
  FX-D63 — evals every 20 rounds; T3C floor covers the cifar refl syn_50 fork. *Refutes:* speech felix finite but flat → felix's rate ×
  server lr 1.0 is too strong under staleness (in-process lr sweep, operator call); telemetry still > 100 MB/min → a second spin source.
  Afterwards: `accuracy_table.py --runs …`, regrade each pool (C5).
- **PR18 · next:** cifar felix/fedbuff real G1A re-run (their run 16 real legs ran the spin); GPU G0U + G0UC all six (FX-N9); G1U n=300;
  speech oort/feddance G1A.

**Operator decisions / open**
- Decided (2026-10-06): Claude runs the PL3 scale smoke and launches the block in tmux `dg_flame`; every block opens with phase 0 and aborts itself on failure.
- Decided (2026-10-05): cifar GPU legs at 0.2 CPU/trainer (FX-D49); a run's second block starts automatically after
  the screens; streaming design → FELIX Active build.
- Decided (2026-10-04): fix the real aggregator, not the wire format (FX-D44); speech D ×5 (FX-D46); C11, C12.
- Open: FX-N73 — should selector speed include the transfer leg in sim, as real Oort's duration does?
- S5 knob layout (a) adopted (2026-10-01): `datasets.yaml` holds dataset defaults + `by_baseline` values with sources;
  left: the `fedbuff.py` server-lr table into config, and a resolved-matrix printer.

---

## Active build — parity ladder (FX-N42)

`examples/scripts/parity_ladder.py` (`LADDER`; known misses `KNOWN`, each citing an item; real-only phases `REAL_ONLY`).
Each rung is one pool under `<out>/<rung>/`; `LADDER.txt` lists green/known/red per rung.

| rung | legs | est. wall (cifar / speech) |
|---|---|---|
| L0 | static: pytest collect, data, knob preflight (pool gate) | 4 min |
| L1 | T1: sim only, 120s, syn_0 + syn_50, EV | 6 / 6 min |
| L2 | T3: real+sim pairs × 4 shapes (oort mobiperf 960s) | ~70 / ~90 min |
| L3 | L2 re-graded to stage 3 | 0 |
| L4 | T4 extras: P4-P11c, EV | 37 / 48 min |
| L5 | GS: GPU pairs 10 min, G0 cohort, syn_0 + syn_20 | 122 / 238 min |
| L6 | G0C + G0: 30 min + real↔real control | > 6h per dataset |
| L7 | G1/G2 at reference n, 90 min → 3h | cifar G1 4.5h + G2 ~9h / speech ~5h |

- Q2 · wip: `--control`/`--floors` + `--regrade` feed G0C (syn_0/20) and G0UC (G0U, G0T) real↔real floors into
  `floor_gated_tol` (SKIP only, never tighten: n=2, T8). Next: ≥3 legs per cell, then tighten + gate DIST.
- Q4 · todo: runner reports the two axes per cell (today it gates timing INV/EXACT and leaves logical DIST ungated).
- Q5 · todo: per-cell scheduling (C3).
- Q6 · wip: `logical_diff.py`; next: per-round marginals for async pairs.
