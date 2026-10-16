# Parity readiness — real↔sim parity, Felix and FluxTune

> **Golden rule: ROBUST R0: short screens first; long runs only for what needs length.**
>
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
- **C5 Loop per run:** regrade, then rewrite the FELIX Scoreboard (now + expected after the next run); never append a results log.
- **C6 Correct first, equal second:** a check green on a shared bug is a bug.
- **C7 Instrument what you can't see,** in the same change as the next fix.
- **C8 Launch only when fixes are maximized** and the PL checklist passes.
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
- **C21 Measured packing before any batch config (operator 10-15).** A batch (PR36, PR37, any FluxTune block) is configured only after
  a 5-10 min screen at production density shows its packing on measured CPU, RAM, GPU memory and GPU SM use, not formula estimates
  or lease counts. Name the binding resource and the number that binds it, and record both here and in the track doc.
- **C22 Maximum signal per unit time (operator 10-15).** Time is the scarcest resource and nodes are shared.
  - Find issues with the shortest run that shows them; long runs only confirm a config already shown correct (C15, C18).
  - Run as many legs at once as measured resources allow without interference (Throughput below); a stalled or idle resource is
    waste to root, like a red cell.
  - Every batch ends with `resource_report.py --util`; its numbers go into the Throughput record.

## Pre-launch checklist (operator: never repeat a failure)

A new failure mode adds a check here in the same session.

| ID | check | how | incident |
|---|---|---|---|
| PL1 | full pytest green | ROBUST → Full pytest | R10 |
| PL2 | gate: collect + smoke pair per dataset | automatic per pool | R19 |
| PL3 | scale smoke: new GPU shapes at production n/c/GPUs, 5 min, co-located | `harness_pool.py --scale-smoke 300`; block phase 0 repeats | run 15 GPU OOM, host RAM OOM (FX-D56-59) |
| PL4 | detached launch | `run_prNN.py` in tmux `dg_flame` (survives a lost terminal); stop: one Ctrl+C in its pane | lost terminal |
| PL5 | no code edits while a run is live, incl. this session's own smokes | T15; `pgrep -af 'debug_run|harness_pool'` empty before editing | half-edited selector ran; 10-14 `debug_run.sh` edited under a live G3 smoke |
| PL6 | GPUs idle, ECC-clean, disk free | `nvidia-smi --query-gpu=index,memory.used,ecc.errors.uncorrected.volatile.total --format=csv`; `df -h` | S0 GPU 1 ECC |
| PL7 | new knob reaches the trainer | dry-run + grep a short leg (L9) | FX-D32, FX-D72 |
| PL8 | +10 min: `BLOCK.log` advancing, PROGRESS lines, telemetry < 100 MB/min | read `OUT/`; `ls -l <run>/telemetry/aggregator_*` | run 15 unseen OOM; run 16 310 MB/min |
| PL9 | every leg graded before citing numbers | `SUMMARY.txt`, else `parity_check.py` | run 16 MISSING legs |
| PL10 | stall rules never cut a leg matching its sim | compare STALLED real with sim timeline | run 18 oort cut pre-round 1 |
| PL11 | every sim leg has a charge profile for its harness × dataset × stack | no `no profiled sim charges` WARNING in the leg's `shell.log` | FX-N71: tiny_cpu ran a 0.6 s placeholder |
| PL12 | co-located pools share the node by leases (R24); size each pool's lease to its real use | GPU pool `--gpu-cpus-per-trainer 0.1` beside a CPU tier; watch `LOAD foreign` and `INTERFERENCE` | N86: G0U leased 112 CPUs using ~3, T3 idle 45 min |
| PL13 | no leg starts without the host RAM it needs, idle pool or not | automatic (`ram_blocks` + node-wide `ram_*` ledger, FX-D131/D136); `RAM_TIGHT` in a DONE line = rerun its neighbours | PR28: G5 at 285 of 398 GB OOM-killed 3 legs; PR29: 398 + 222 GB co-ran |
| PL14 | measured RAM within estimate, no growth | DONE lines / runner probe `RAM_OVER`, `LEAK?`; `resource_report.py <pool>` (FX-D140) | PR29 OOMs on formula estimates |
| PL15 | selector comparisons have a choice; real↔sim equality legs run deterministic kernels | `fd_sanity.py` choice check; `trainerDeterministic=true` (FX-D149) | F3: both arms picked the same ends; F4 losses drifted from r2 |
| PL16 | foreign GPU jobs: pin pools to GPUs nobody else uses (`GPU_IDS`), not to "free right now" | `nvidia-smi --query-compute-apps=pid --format=csv` + `ps -o user=`; watch restarts | PR31 A1: vLLM on 4-7 restarts every few min, pool took GPU 6 between restarts, trainer OOM aborted the pool |
| PL17 | a stage's gate must not queue behind its siblings | the gate's smoke is per dataset, timed from its first leg (FX-D154); `ABORT gate` in a stage's pool.log = read P00/pool.log for BLOCKED before blaming code | PR31 A4/A5: smoke timed out waiting on 112 sibling CPUs |
| PL18 | the launch node runs the planned commit and code key | `git log -1` on that node = the commit the doc names; `code_state.txt` HEAD; each pool's `code_key.txt` = the doc's key (FX-D162) | PR32: jayne launched without the shepherd commit (no G0UC, ECC GPU 1 used) |
| PL19 | ≤ 2% of train tasks overrun D per leg, real and sim; above it, fewer trainers per GPU | DONE lines / trainer `overran` count (FX-N98) | PR35: speech G1AS sim 10%, real G1A 3.5% |
| PL20 | AVL_EVAL is eval-only and Felix-only (FX-D165) | EV21 in every leg's events (prefix-safe → DOOMED) | 10-14: baselines trained on mobiperf_3st EVAL devices |
| PL22 | the batch's packing is measured (C21): its last screen's `--util` names the binding resource, legs/h, overruns 0-2%, no slow legs; the plan uses `--gpu-share`/`--lease-from-telemetry` as measured | `resource_report.py <screen pools> --util` | first PR36: 3.3 legs at once, CPU 6%, GPU memory 39%, GPUs idle 25% for 14 h |
| PL23 | packed GPU footprints are per nominal GPU, floored per trainer, and at most FX-T36 trainers share a GPU | `gpu|` history ≈ exclusive peaks (cifar n50 ~20 GB, speech ~40 GB/GPU); no `CUDA error: out of memory` in a smoke's trainers.log (FX-D175) | PR36-short: B1 + S1 OOM-aborted, 137 trainers on one A40 |
| PL21 | a new or changed prefix-safe EV passes on stored legs of all six × every trace before launch | run it over the last batch's legs (`event_invariants.load_run` + the check) | PR36: EV21 DOOMED 17 oort_star/fedbuff legs mid-batch |

## Throughput (both tracks; operator 10-15)

Goal: most legs (and budget-hours) finished per wall hour, each leg behaving as if alone (R18). Felix numbers: FELIX Active build
"throughput"; FluxTune adopts the same tools when it resumes.
- **Measure (`harness_pool` ledger, `resources.jsonl`, 30 s):** node CPU busy cores and RAM used, per-GPU SM % and memory, per leg cores,
  PSS by role, GPU GB and SM % (`nvidia-smi pmon`). The DONE line and end event carry cores p95, `CPU_SAT`, RAM peak vs estimate, GPU
  peak, SM %, overran/train tasks. History (`_harness_durations.json`) keeps per leg key: duration, cores p95, RAM peak, [GPU GB, SM %]
  per GPU.
- **Report:** `resource_report.py POOLS --util [--gpus ids]` gives legs/h, budget-h/h, mean concurrency, CPU/RAM/GPU SM/memory use, GPU idle
  share, overruns, and slow legs (> 1.25× solo estimate).
- **Pack (flags, default off; byte-identical off, R9):** `--lease-from-telemetry` (measured CPU lease for real CPU legs, RAM reservation
  = 1.15 × measured peak, admission ≤ 70% node CPU, back-pressure on saturated slots); `--gpu-share K` (legs share a GPU while committed
  measured peaks ≤ 90% of memory and SM ≤ 90%, ≤ FX-T36 trainers per GPU; unmeasured legs take a whole GPU); `run_pr28 --measured --one-queue` (all stages at once,
  stage order = GPU priority).
- **Buffers (operator 10-15):** no resource past 90% of capacity (node hangs and failed runs cost more than the packing gains): committed
  GPU memory ≤ 90% and live use + newcomer ≤ 90%, GPU SM ≤ 90%, host RAM headroom 10% of MemTotal (`--mem-headroom-gb` default), CPU
  admission ≤ 70%.
- **Static reservations (operator 10-15):** a leg reserves 1.15 × its largest measured peak upfront and keeps it to the end. Dynamic
  grow/release was rejected: allocations can't wait for a reservation (an OOM, not a pause), and early release only frees the last ~25%
  of a leg.
- **Bottleneck → optimize it:** once a batch's binding resource is named (C21), add a next step to cut that resource's use in our own
  processes (aggregator, trainers), so it binds less.
- **Interference signals (each must match the solo baseline):** real overruns of D (PL19, ≤ 2%), `CPU_SAT`/`TIMING_SUSPECT`, slow legs, sim
  wall slowdown, EV/parity verdicts vs the same jobs packed exclusively.
- **Experiment loop:** same jobs, two packings, 5-10 min legs at production density (`--scale-smoke 240`), run back to back; adopt a packing
  only if legs/h rises and every interference signal holds.

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
python lib/python/examples/scripts/readiness_score.py <pools>    # parity score matrix (C5)
python lib/python/examples/scripts/insights.py <run_dirs> [--json J] [--plot P]   # FELIX_INSIGHTS metrics, real/sim twins paired
python lib/python/examples/scripts/cleanup_experiments.py [--apply] [--fwdllm]   # free storage: keeps pooled/banked/cited/<7d runs
conda run -n dg_flame python lib/python/examples/async_cifar10/scripts/parity/event_invariants.py <run_dir>
python lib/python/examples/async_cifar10/scripts/parity_check.py --real <A> --sim <B> --control --json-out c.json   # real<->real floor (Q2)
```
- Pools on one node are safe (L28); progress every 5 min; DONE lines carry GPU peak; one Ctrl+C stops all. Nodes per R4.
- **Early exits (all automatic in `harness_pool`):** fatal log line → pool aborts in ~30 s (ABORT.txt); no new round 15 min / silent
  10 min → leg killed (STALLED.txt); prefix-safe EV FAIL (EV6/8/9/11/14-19) → leg + pair partner killed at the next 5-min check
  (DOOMED.txt, FX-D133). Runner probes print PASS/FAIL per claim as stages finish. Not early-killable: EV1/5/7/10/12, timing/DIST.

---

## Felix scoreboard

Single source: [FELIX_READINESS.md](FELIX_READINESS.md) → Scoreboard (cell matrix, open reds → root, expected after the next run,
accuracy table, convergence board A/C/U/S for long runs: FX-N93). `scripts/readiness_score.py <pools>`: baseline × dataset × scenario cells green / applicable, worst pair wins, a later
pool supersedes, untested counts as not green. INV/EXACT via `--grade <pool> --regrade --max-stage 9`; DIST gated by floors only (Q4).

**Settled.**
- L1-L5 (run 8) green apart from no-floor tolerances; L6 G0C + G0 35/0/1.
- Real↔real K2 at syn_0: cifar ≤ 5%, speech ≤ 3% (feddance 17-19%). Rows below that name syn_20/syn_50 ran the deleted 2-state traces.
- Cifar L7 n=300: G1 felix/fedbuff K2 ≤ 0.3%; G1L acc diff within real↔real.
- Cifar G2 refl/oort/feddance green; N62 oort syn_50 3 h 69/69.
- Speech G1S felix + fedbuff, G2 oort + refl green on INV/EXACT.

G0UC real↔real floors (n=50 2-state syn_50, throughput rel; syn_50_3st floors come from PR36 G0UC): oort 0.90 · refl 0.20 · fedbuff 0.18 · feddance 0.13 · oort_star 0.06 · felix 0.02.
Mobiperf: feddance 0.56, others ≤ 0.07. n=50 syn_50 stall placement is chaotic (FX-L53).

## FluxTune scoreboard
Parked with the track; board in simulate_fwdllm.md §A.

---

## Next steps (run queue — top item is next)

- **PR36-short rescreen (FELIX Next steps; first screen OOM-aborted, FX-D175):** jayne `--only B1,C5,S1S`, shepherd `--only S1C,S1S0` (~2.5 / 2.3 h).
- **PR36 · full run after a clean rescreen (first run stopped 10-15 by EV21 false kills; FELIX PR36):** full re-screen on one code key (G0U/G0UC, T3/T3C, G0T/G0TC, G1AS, third reals) → 76/76 +
  accuracy + FX-N98 speed confirm. `run_pr36.py --node jayne|shepherd` (~14.8 / 13.8 h). Screen shapes stay per tier.
- **PR37 · blocked: PR36 complete:** all six × syn_0 + syn_50_3st + stream_lin × both datasets, 2 h real + 2 h-vclock sim, real replicate on
  syn_0 + lin, at one n:c:K per dataset (G3: cifar 75:15:10, speech 50:15:10) → sim-only experiments. `run_pr37.py` (~24-28 h).
- **Long blocks (G1U, G1AS grid) · deferred by C18** until PR37: FELIX FX-N93.

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
- Cifar Oort/oort_star clip 1.0 on (recorded deviation); speech G1AS accuracy recorded round-bound; debug every red on short runs
  before any long run (10-10).
- D = our trainer runtime distribution for every baseline, never per-sample; compute > D = rethink for all (ROBUST L36) (10-08).
- Trainer memory audit to raise n (FX-N84) (10-08).
- One code state = one `code_key` over run-affecting files (FX-D162); a re-screen may span batches on it (PR35 + PR36) (10-14).
- Skip PR35b; FX-N97/N96/N98 before PR36; ≤ 1-2% overruns; PR37 (all six, 3 h) gates sim-only experiments (10-14).
- 2-state syn traces deleted; 3-state `syn_*_3st` everywhere (10-14, FX-D166).
- Sim speed: never slower than real; 1-2× needs review; 2-5× and beyond is the target; always look for more speedup (10-14).
- Batches span jayne GPUs 0,2-7 + shepherd GPUs 0-3; scale n/c/K at fixed ratios to run more legs in parallel (10-14).
- Speech trains in bf16 (FX-D168); GPU placement balanced by default (FX-D169); PR37 shape G3 (10-14 late).
- Throughput is the central goal; batch configs only after measured packing (C21, C22); no long run at low utilization (10-15).

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

- Q2 · wip: real↔real floors feed `floor_gated_tol` (SKIP only). Floors take every same-code replicate pair (FX-D159); ≥ 3 reals vote per check
  (FX-D161, L13); next: tighten DIST.
- Q4 · todo: runner reports logical and timing axes per cell.
- Q5 · todo: per-cell scheduling (C3).
- Q6 · wip: `logical_diff.py`; next: per-round marginals for async pairs.
