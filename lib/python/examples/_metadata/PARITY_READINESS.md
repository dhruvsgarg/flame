# Parity readiness — real↔sim parity, Felix and FluxTune

> **C0 · Rule zero: extract, then climb (operator). Time is the scarcest resource.**
> 1. **Mine what exists first:** stored logs, telemetry and code answer most questions.
> 2. **Short runs find and fix:** shortest run showing the mechanism; verify across several settings.
> 3. **Common roots first:** one root greening many cells beats any single-cell fix.
> 4. **Lower rungs clean first:** higher rungs launch only with no open lower root.
> 5. **Long runs only confirm** or test length-only effects; state confirm/refute up front.
> 6. **Long-run phase:** once short runs stop finding roots, launch many small long legs in parallel.

## Preamble
- **Read [ROBUST_FL_READINESS.md](ROBUST_FL_READINESS.md) first.** This doc owns parity rules, method, tools, scoreboards and run queue.
- Track docs ([FELIX](FELIX_READINESS.md), [FLUXTUNE](FLUXTUNE_READINESS.md)) keep their items, lessons and built features.
- Deprecated parity docs ([PARITY.md](../async_cifar10/PARITY.md), [simulate_fwdllm.md](../fwdllm/simulate_fwdllm.md),
  [PARITY_CHECKER_README.md](../async_cifar10/scripts/parity/PARITY_CHECKER_README.md)) only shrink.
- **IDs:** `C#` climbing rules · `PL#` pre-launch checks · `Q#` ladder tasks · `PR#` run-queue steps.

---

## Climbing rules (operator)

- **C1 Every run closes roots:** each red cell gets a fix in tree or a predicted item.
- **C2 Lowest red check per cell first;** a root failing 2+ cells outranks one.
- **C3 Climb per cell** (baseline × dataset × avail); always `--keep-going`.
- **C4 Logical before timing;** timing closes via profiled charges, never knob tuning (FX-T31).
- **C5 Loop per run:** regrade, rewrite scoreboard, dashboard and FELIX accuracy table.
- **C6 Correct first, equal second:** a check green on a shared bug is a bug.
- **C7 Instrument what you can't see,** in the same change as the next fix.
- **C8 Launch only when fixes are maximized** and PL1-PL11 pass.
- **C9 Correct fixes default on** with a revert knob; unproven ones default off.
- **C10 Resolve, don't just file:** drive max items to closure; report closed vs opened.
- **C11 New features screen short, small, parallel** (≤ 30 min, n ≈ 50) before long legs.
- **C12 Kill early only by rule:** S1 no commit 15 min, S2 no log 10 min.
- **C13 Model every significant real cost in sim,** on the clock and in selector measurements.
- **C14 Baselines stay themselves:** source-faithful values from `baseline_reference.yaml` (ROBUST L33-L37); only Felix/FluxTune innovate.
- **C15 Quick runs before long runs:** no overnight until short runs stop finding roots.
- **C16 Batch fixes per run:** root many offline, test together in one ≤ 30-min run.
- **C17 Close Felix fast (operator 10-08):** goal = baselines, availability, streaming, both datasets closed.
  - Pick the top open item; fix the root system-wide: every Felix stack and baseline, not one cell.
  - Shared roots that FluxTune also has get one line in FT-N12-style FLUXTUNE Next steps, fixed later.
  - Land a few small, independently checkable fixes (pytest / stored runs), then one short CPU smoke batch.
  - Smoke after every few fixes so a new red names its cause; runs only as long as the signal needs.
- **C18 Short runs gate long runs (operator 10-08).** While any issue is fixable and testable by a short run, no long run.
  Short = pytest, stored telemetry, CPU tiers, ≤ 30-min GPU screens (G0U/G0UC/G0T/G0To), in-process `fl_lr_check.py`.
  Long (G1*/G2/blocks) only to find new issues once the short queue is empty. No long run before 2026-10-09 02:00.

- **C19 Minimum experiment first (operator 10-08).** Every block, before launch, states in its doc line:
  - *Claim:* the one hypothesis it validates or refutes.
  - *Unknowns:* only what is unmeasured; anything measured or reliably modelled is not re-run.
  - *Shortest real test per unknown:* micro-benchmark before end-to-end, one probe before a sweep, one repetition first,
    the case the model says matters before calibration or edge cases.
  - *Decision + stop rule,* written before the data: what each result decides; the threshold to stop or rethink.
  - *Duration:* expected wall time; > 1 h states why it can't be shorter. Sweeps, repeats and grids only confirm.
- **C20 Invalidated configs stop (operator 10-08).** When a baseline's definition changes, its queued and running legs are
  dropped (not graded, not cited) and resume on the new config; other baselines continue.

## Pre-launch checklist (operator: never repeat a failure)

A new failure mode adds a check here in the same session.

| ID | check | how | incident |
|---|---|---|---|
| PL1 | full pytest green | ROBUST → Full pytest | R10 |
| PL2 | gate: collect + smoke pair per dataset | automatic per pool | R19 |
| PL3 | scale smoke: new GPU shapes at production n/c/GPUs, 5 min, co-located | `harness_pool.py --scale-smoke 300`; block phase 0 repeats | run 15 GPU OOM, host RAM OOM (FX-D56-59) |
| PL4 | detached launch | `run_block_*.sh --detach`; stop `kill -INT -- -$(cat OUT/PGID)` | lost terminal |
| PL5 | no code edits while a run is live | T15 | half-edited selector ran |
| PL6 | GPUs idle, ECC-clean, disk free | `nvidia-smi --query-gpu=index,memory.used,ecc.errors.uncorrected.volatile.total --format=csv`; `df -h` | S0 GPU 1 ECC |
| PL7 | new knob reaches the trainer | dry-run + grep a short leg (L9) | FX-D32, FX-D72 |
| PL8 | +10 min: `BLOCK.log` advancing, PROGRESS lines, telemetry < 100 MB/min | read `OUT/`; `ls -l <run>/telemetry/aggregator_*` | run 15 unseen OOM; run 16 310 MB/min |
| PL9 | every leg graded before citing numbers | `SUMMARY.txt`, else `parity_check.py` | run 16 MISSING legs |
| PL10 | stall rules never cut a leg matching its sim | compare STALLED real with sim timeline | run 18 oort cut pre-round 1 |
| PL11 | every sim leg has a charge profile for its harness × dataset × stack | no `no profiled sim charges` WARNING in the leg's `shell.log` | FX-N71: tiny_cpu ran a 0.6 s placeholder |

## Method

- **Pipeline** `clock → availability → selection → dispatch/train → return/order → aggregation → utility → emergent`;
  root = lowest failing check whose inputs match. Checker stages (`CHECK_META`) 0-9 follow it, plus budget.
- **Tiers:** INV/EXACT hard fail · DIST fails only above a replicate floor (L12, Q2) · DIAG informational.
- **Growth rule:** every root leaves the finest check that would have localized it.
- Logical budget (L10); stochastic selectors graded on marginals (L14); timing on < 20 commits is noise.

| axis | graded on | checks |
|---|---|---|
| **Logical** (first) | EV + time-stripped quantities | `eligibility`, `selection_detail`, `participation`, `selection_bias`, `residence`, `agg_goal_cycles_*`, `aggregation_sequence`, `staleness`, `withheld_delivery`, `utility`, `convergence*`; EV17 |
| **Timing** (after) | stage-1 clock + time-to-N | `overhead_residual`, `per_round_advance`, `throughput`, `overlap_factor`, `matched_budget_coverage`, `total_commits`, `terminal_state`; `agg_timing_split` (DIAG) |

Run length by residual (R6): telemetry 5-10 min · clock 15-30 · mechanism 30-45 · participation 60+ ·
compounding clock 90 min-2 h · convergence sign-off 2-4 h.

## Tools

```
L="conda run --no-capture-output -n dg_flame python lib/python/examples/scripts/parity_ladder.py"
$L --rungs L1-L4 --datasets all --keep-going --deadline-h 6     # CPU
$L --grade <pool> --regrade --max-stage 9                        # offline re-gate (~30s/48 pairs)
python lib/python/examples/scripts/logical_diff.py <pool>        # first diverging selection/commit per pair
conda run -n dg_flame python lib/python/examples/async_cifar10/scripts/parity/event_invariants.py <run_dir>
python lib/python/examples/async_cifar10/scripts/parity_check.py --real <A> --sim <B> --control --json-out c.json   # real<->real floor (Q2)
```
- Pools on one node are safe (L28); progress every 5 min; DONE lines carry GPU peak; one Ctrl+C stops all. jayne only (R4).

---

## Progress dashboard (rewrite every run, C5)

Open = FELIX FX-N items; built = FX-D lines; target = G1A full-data leg reaching cifar 50% / speech 60% both sides.

| run / commit | date | pytest | open FX-N | built FX-D | INV/EXACT green / known / red | at target |
|---|---|---|---|---|---|---|
| runs 6-8 (`798a0e435`) | 10-01 | n/a | 20 | 31 | L1-L5: 134 / 7 / 5 | — |
| runs 12-14 (`e174fbed6`) | 10-05 | n/a | 17 | 47 | G0U/G0T: 37 / 2 / 11 | — |
| run 16 (`11152ac33`) | 10-06 | 2414 / 0 | 17 | 61 | 9 / 2 / 1 (+5 MISSING) | 1 / 12 |
| run 17 (`67959e542`) | 10-07 | 2416 / 0 | 17 | 62 | 14 / 0 / 0 | 1 / 12 |
| run 18 (`1abe482a3`) | 10-07 | 2420 / 0 | 16 | 63 | T3 5/0/7 · G1A speech 1/0/3 (timing) | 3 / 12 |
| run 19 | 10-07 | 2423 / 0 | 16 | 69 | speech G0U 6/0/2 · cifar G0U 9/0/1 | 3 / 12 |
| runs 20-24 (`bd5fc954d`) | 10-07 | 2427 / 0 | 16 | 76 | speech felix G0U ✅ (120 / 119 rounds) | 3 / 12 |
| run 25 | 10-08 | 2445 / 0 | 16 | 86 | speech G0U 10/0/2 · cifar 10/0/1 (+11 MISSING) | 3 / 12 |
| run 26 | 10-08 | 2447 / 0 | 16 | 88 | cifar felix/fedbuff/oort 3/0/0 · speech fedbuff 1 red | 3 / 12 |
| run 27 (FX-D89-91) | 10-08 | 2450 / 0 | 17 | 91 | speech felix + fedbuff ✅ · cifar felix + oort ✅, fedbuff floor-bound | 3 / 12 |
| CPU smokes (FX-D92-99) | 10-08 | 2456 / 0 | 16 | 99 | T1 24/24 · T3 mobiperf 2/0/0 · P6 2/0/0 · P7 6/0/0 | 3 / 12 |
| PR22/23 + FX-D100-103 | 10-08 | 2491 / 0 | 16 | 103 | T3 cifar 27/30 · speech 16/30 (stale stub profile → FX-D103; syn_0b re-check 2/2) · G0U fedbuff cifar ✅ (FX-N77 closed) | 4 / 12 |

**Per baseline × dataset (latest grade).** Parity = INV/EXACT on the latest pair; accuracy numbers in FELIX accuracy table.

| baseline | dataset | parity syn_0 | parity unavailability | target |
|---|---|---|---|---|
| felix | cifar | ✅ G1A 67/67 (run 16) | ✅ T3 (run 16); ✅ G0U syn_50 (run 27); mobiperf ⬚ PR21 | ✅ 57 / 60 min |
| felix | speech | 🟡 G1A timing reds (run 18; FX-N70) | ✅ T3; ✅ G0U syn_50 (runs 24-27) | ✅ 72 / 72 min |
| fedbuff | cifar | ✅ G1A 66/66 (run 17) | ✅ T3; 🟡 G0U syn_50 floor-bound (run 27) | ❌ (FX-D73 sim 56%) |
| fedbuff | speech | 🟡 G1A timing reds (run 18) | ✅ T3; ✅ G0U syn_50 (runs 26-27); mobiperf ⬚ PR21 | ✅ 82 / 88 min |
| refl | cifar | ✅ G1A 68/68 (run 17) | ✅ T3 syn_50 on T3C floor; 🟡 G0U A2 KS 0.24 | ❌ rising |
| refl | speech | ✅ G1A; U6 DIST (FX-N70) | ✅ T3 syn_50 (run 17) | ❌ |
| oort | cifar | ✅ G2 (L7) | ✅ G0U mobiperf + syn_50 (run 27) | n/a (stream) |
| oort | speech | 🟡 G1A throughput 8.1% (FX-N76) | ✅ G0U syn_50; mobiperf ⬚ PR21 | ❌ |
| oort_star | cifar | ✅ L6 G0 | 🔴 G0T lin syn_50 K3b (FX-N9) | ⬚ |
| oort_star | speech | ✅ L6 G0 | 🟡 G0U syn_50 replicate chaos (FX-L53); ⬚ PR21 | ⬚ |
| feddance | cifar | ✅ G2 on sim↔sim floor | ✅ G0U (runs 12-14) | ❌ |
| feddance | speech | ✅ G1A 1.0 (run 18) | 🟡 T3 mobiperf overhead_residual | ❌ round-bound if audit faithful (FX-N74) |

refl, oort, oort_star, feddance rows predate FX-D100 (source-faithful configs): void for accuracy, re-graded under FX-N80 (C20).

## Felix scoreboard

INV/EXACT green / known / red via `--grade <pool> --regrade --max-stage 9`; DIST gated by floors only (Q4).

**Settled.**
- L1-L5 (run 8) green apart from no-floor tolerances; L6 G0C + G0 35/0/1.
- Real↔real K2 at syn_0: cifar ≤ 5%, speech ≤ 3% (feddance 17-19%).
- Cifar L7 n=300: G1 felix/fedbuff K2 ≤ 0.3%; G1L acc diff within real↔real.
- Cifar G2 refl/oort/feddance green; N62 oort syn_50 3 h 69/69.
- Speech G1S felix + fedbuff, G2 oort + refl green on INV/EXACT.

| pool (run) | g / k / r | reds and roots |
|---|---|---|
| PR23 `pool_20261008_1845_G0Tgs`: speech G0T felix + fedbuff | 8/0/0 · EV 8/8 | DIST only: phase_gpu_compute, speed identity (FX-N76); fedbuff syn_50 eligibility |
| PR22/23 `pool_20261008_1628_{T3c,T3s,T1}`, `_2000_wt{T1,P7}`, `_2010_wtT3s`, `_1628_G0Ufb` | T3 cifar 27/0/3 · speech 16/0/14 · T1 24/24 · wtP7 6/0/0 · G0Ufb 1/0/0 | speech timing = stale stub profile → FX-D103 (syn_0b 2/0/0 after); cifar syn_50 oort-family terminal_state (pre-FX-D100, FX-N9) |
| CPU smokes `pool_20261008_15{0148,2447}_*` | T3 mob 2/0/0 · P6 2/0/0 · P7 6/0/0 (feddance DIST) | FX-D92-D99; T1 24/24 EV |
| run 27 `block_20261008_1231_run27`: G0U syn_50 both datasets | speech 2/0/0 · cifar 2/0/1 | cifar fedbuff red vs own real, green vs run 26 real |
| run 26 `block_20261008_1138_run26` | 3/0/0 · speech 1/0/1 | fedbuff overhead_residual 12.5% → FX-D90 |
| run 25 `block_20261008_0508_run25`: G0U | speech 10/0/2 · cifar 10/0/1 | early async commits → FX-D88; late abandons → FX-D89; exit race → FX-D87 |
| run 19 `block_20261007_1456_run19`: speech G0U/G0UC, cifar G0U, T3C | 6/0/2 · 9/0/1 · 6/0/0 | delivery-order roots → FX-D69-71 |
| run 18 `block_20261007_0424_run18`: T3, G1A speech | 5/0/7 · 1/0/3 | T3 syn_50 timing reds; G1A timing only (FX-N70, FX-N76) |
| run 17 `block_20261006_1943`: T3 + G1A(S) | 8/0/0 · 4/0/0 · 2/0/0 | FX-D60/61/63 confirmed |
| run 16 `block_20261006_run16`: T3 + G1A | 6/2/0 · hand-graded | cifar felix 67/67; fedbuff mobiperf S1 known |

G0UC real↔real floors (n=50 syn_50, throughput rel): oort 0.90 · refl 0.20 · fedbuff 0.18 · feddance 0.13 · oort_star 0.06 · felix 0.02.
Mobiperf: feddance 0.56, others ≤ 0.07. n=50 syn_50 stall placement is chaotic (FX-L53).

## FluxTune scoreboard
Parked with the track; board in simulate_fwdllm.md §A.

---

## Next steps (run queue — top item is next)

- **PR24 · FELIX "Next steps → Resume here", in order FX-N82 → N83 → N80 → N84.** Then FX-N80's C19 steps (in-process `fl_lr_check --baseline`, then T3 + G0U screens on FX-D100 configs).
- **PR23 · GPU screens (C18) · partly done.** Done: G0UC fedbuff cifar ×3 + G0U pair at HEAD ✅ (FX-N77 closed). Done: speech G0T
  felix + fedbuff INV/EXACT 8/0/0 (`pool_20261008_1845_G0Tgs`). Dropped by C20 (rerun on FX-D100): G0T oort_star, G0U mobiperf oort_star/refl/feddance,
  speech G0T for the other four.
- **PR21 · GPU long block · deferred by C18** until the short queue is empty: G1A accuracy pairs, G1U, G0T full grid.

**Operator decisions**
- Claude runs PL3 and launches blocks in tmux `dg_flame`; phase 0 aborts on failure (10-06).
- Cifar GPU legs at 0.2 CPU/trainer; a run's second block auto-starts (10-05).
- Fix the real aggregator, not the wire format; speech device time ×5 (10-04).
- C13, C14, C15 adopted; FX-N73 folded into C13 / FX-N76 (10-07).
- Flat weight codec default-on; lean MQTT keeps QoS 2; CPU aggregation, GPU eval (10-08).
- QoS 1 rejected: per-message seqno restart could splice late duplicates (10-08).
- Short runs gate long runs (C18); GPU screens ≤ 30 min and `fl_lr_check.py` count as short; speech streaming on GPU, model unchanged (10-08).
- A baseline slow by its own algorithm is round-bound, not a bug; disparity vs its reference is fixed (10-08).
- S5: per-baseline values moved to `baseline_reference.yaml` (FX-D100); `datasets.yaml` keeps dataset-only values.
- Baselines: code wins over paper; nearest own config when a dataset is missing; sync K/N >= 5%; model fixed, adapt the fewest
  knobs and record them (ROBUST L33-L37) (10-08 evening).
- D = our trainer runtime distribution for every baseline, never per-sample; compute > D = rethink for all (ROBUST L36) (10-08).
- Implement REFL `adapt_selection` (FX-N82); vendor Oort into third_party (FX-N83); trainer memory audit to raise n (FX-N84) (10-08).

---

## Active build — parity ladder (FX-N42)

`examples/scripts/parity_ladder.py` (`LADDER`; `KNOWN` misses cite items; `REAL_ONLY` phases). One pool per rung; `LADDER.txt` summarizes.

| rung | legs | wall (cifar / speech) |
|---|---|---|
| L0 | static: pytest collect, data, knob preflight | 4 min |
| L1 | T1 sim only, 120 s, EV | 6 / 6 min |
| L2 | T3 real+sim × 4 shapes | ~70 / ~90 min |
| L3 | L2 regraded to stage 3 | 0 |
| L4 | T4 extras P4-P11c, EV | 37 / 48 min |
| L5 | GS GPU pairs 10 min, G0 cohort | 122 / 238 min |
| L6 | G0C + G0: 30 min + real↔real | > 6 h per dataset |
| L7 | G1/G2 at reference n, 90 min → 3 h | ~5-13 h |

- Q2 · wip: real↔real floors feed `floor_gated_tol` (SKIP only). Next: ≥ 3 legs per cell, then tighten DIST.
- Q4 · todo: runner reports logical and timing axes per cell.
- Q5 · todo: per-cell scheduling (C3).
- Q6 · wip: `logical_diff.py`; next: per-round marginals for async pairs.
