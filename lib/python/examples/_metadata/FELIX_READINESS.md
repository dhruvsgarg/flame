# Felix readiness — backprop FL (async_cifar10, google_speech)

## Preamble
- **Read [ROBUST_FL_READINESS.md](ROBUST_FL_READINESS.md) first.** Its doc rules, operating rules (R1-R13),
  shared lessons (L) and tripwires (T) apply here and are not repeated. This doc holds only what is specific
  to Felix: weight-aggregating backprop FL, round/`agg_goal` progress axis, Oort-family selectors.
- **CURRENT FOCUS.** Goal: Felix feature-complete (syn_0 + unavailability, both datasets) and paper
  experiments running sim-only. Per run: root-cause every red cell (or as many as the logs allow) and fix it, so
  the whole matrix reaches parity in the fewest runs. **Parity work follows
  [PARITY_READINESS.md](PARITY_READINESS.md)** (climbing rules C1-C9, method, tools, scoreboard, ladder build).
- **Scope:** `felix`, `oort`, `oort_star`, `refl`, `feddance`, `fedbuff` (+ each one's `*_oracle` arm for the
  streaming experiment, FX-N13). Out of scope: `fedavg`, `oracle`.
- **Traces (both datasets, both papers):** `syn_0`, `syn_20`, `syn_50` (synthetic) and `mobiperf_3st` (the
  real-world 3-state trace). `syn_10` is the pre-2026-09-27 "syn_20" (10.8%, used by the EuroSys'26 runs; FX-D18).
  Levels (300 trainers, instantaneous TRAIN/EVAL/UN %) are flat from 1 min to 3 h: syn_10 89/0/11, syn_20 ~80/0/20,
  syn_50 ~50/0/50, mobiperf_2st 10/0/90, 3st_50 10/25/65, 3st_75 10/10/80; mobiperf's trainable share rises to 18-22%
  by 6-24 h (diurnal).
- **IDs:** `FX-N` next steps · `FX-L` lessons · `FX-T` tripwires · `FX-D` built features. Shared work is `S#` in the parent.
- **Reference (read only for detail):** rung catalog and derivations → [PARITY.md](../async_cifar10/PARITY.md)
  §2-§5 · checker internals → [PARITY_CHECKER_README.md](../async_cifar10/scripts/parity/PARITY_CHECKER_README.md)
  · availability design → [UNAVAILABILITY_DESIGN.md](../async_cifar10/UNAVAILABILITY_DESIGN.md) · streaming
  experiment → [EXPERIMENT_felix_streaming.md](../async_cifar10/docs/EXPERIMENT_felix_streaming.md) · identity →
  [BASELINES.md](BASELINES.md) (Felix section).

---

## Status grid (scoreboard)

**At a glance (2026-10-01)** — green / known / red, detail in [PARITY_READINESS.md](PARITY_READINESS.md) → Felix scoreboard:

| test | cifar | speech |
|---|---|---|
| pytest (4 suites) | 2325 passed · 0 failed · 7 skipped | (shared) |
| L1 sim EV | 11 / 1 / 0 (run 5) | not run since run 4 |
| L2/L3 pairs | 19 / 0 / 5 (run 5; all 5 = syn_50) | not run since run 4 (21 red of 24) |
| L4 campaign | 25 / 0 / 0, P11a-c caught (run 5) | not run since run 4 |
| L5 GPU pairs | not run since run 4 (6 red of 12, both datasets) | — |
| syn_50 felix+fedbuff verify (T3, 600s) | felix: stage 1 green, A2+A3 red; fedbuff: EV10 (fixed, T1 green) | felix: only A2 red; fedbuff: EV10 (fixed) |

**Cross-cutting capabilities**

| capability | status | evidence / item |
|---|---|---|
| checker catches injected sim bugs (P11a-c) | ✅ both datasets | run 5 cifar, `pool_fxn32_p11b` (P11b on fedbuff) |
| fail fast on fatal errors | ✅ | FX-D22 |
| parity ladder runner + offline re-gate | ✅ | FX-N42 |
| GPU join / warm-up at n=100 | ✅ 44-99s / ≤1.35s | FX-D19 |
| sync wait-K, round livelock | ✅ EV7 green all sync legs | FX-D20, FX-D21 |
| async real ingest latency (< 2s) | ✅ run 3: max < 2s outside P7/P7o except 4 legs with one 2.2-3.2s update | FX-D16 |
| GPU speech legs start (no OOM) | ✅ run 3 L5: EV green on all 24 GPU legs | FX-N34 |
| sim clock charges profiled per dataset/platform | ✅ K3b green 36/36 short cells (CPU); GPU in run 6 (PR7) | FX-D23 |
| DIST tolerances sized by a replicate floor | ⚠ nominal everywhere ("no floor yet") | FX-N42, parent S2 |
| streaming / oracle experiment (FX-N13) | 🟡 cifar arms agree real/sim; speech arms starved | FX-N30, FX-N13 |
| sim speedup vs real | ⬚ not measured at REF | S6, FX-N22 P8 |
| paper experiments sim-only | ⬚ | FX-N12 |

Stored June REF grades (🟡, pre PRs #72-#85): `async_cifar10/experiments/parity_{felix,oort,refl}_20260624_{5400,3h}.json`,
`parity_feddance_20260623_3h.json`.

---

## Active build — parity ladder (FX-N42)

Moved to [PARITY_READINESS.md](PARITY_READINESS.md) → Active build (rungs, tasks Q1-Q6).

## Active build — fast parallel harness (FX-N22)

**Why.** Campaign 3 ran ~9h serially at n=60-120, with the real leg re-run on every pair. The harness exists
to catch logic bugs (event invariants, injected bugs); distribution parity is decided on GPU.

**Isolation contract: a parallel leg must behave as if it ran alone.** Each slot gets:
- disjoint physical cores via `taskset`. The launcher pins the aggregator (`FLAME_AGG_CORES`: 2 at n≤30,
  8 for GPU legs) and the trainers inside that affinity;
- a private mosquitto (`FLAME_MQTT_BROKER`). MQTT client ids are fixed task ids, so two runs on one broker
  kick each other off;
- a run tag (`FLAME_RUN_TAG`) scoping every kill and clean-slate sweep, and an exact run-dir handoff
  (`FLAME_RUN_DIR_FILE`);
- GPU legs: every healthy GPU (volatile-ECC GPUs are skipped), CPUs = 8 + 0.4·n. cifar n=300 therefore
  takes the whole node; speech n=100 takes 48 CPUs and runs beside CPU slots.

**Test shapes** (`harness_pool.SHAPES`): small cohorts with varied c and aggGoal. aggGoal goes through the
launcher's `agg_goal`, which is also the sync K (FX-L36).

| shape | n | aggGoal | c | runtime | trace scale |
|---|---|---|---|---|---|
| syn_0 | 12 | 3 | 5 | 180s | – |
| syn_0b | 15 | 2 | 8 | 180s | – |
| syn_20 | 15 | 3 | 6 | 240s | 4 |
| syn_50 | 15 | 3 | 6 | 1200s | 4 (FX-L43) |
| mobiperf_3st | 45 | 2 | 4 | 240s | 4 (FX-L34) |

google_speech CPU sims get a 2× wall ceiling: with 29 MB updates on 2 aggregator cores they are
aggregator-bound (EV12 still grades the vclock budget).

**Tiers** (`lib/python/examples/scripts/harness_pool.py --tier A[,B] --datasets cifar10|google_speech|all`; runs
from any cwd; `EXAMPLE_OF` maps each dataset to the example that runs it; roots and per-node state in
`lib/python/examples/experiments/`). Every run starts with a
gate: pytest collection plus one felix smoke pair per dataset; it aborts in ~4 min.

| tier | jobs | est. wall, one node (cpt 1 / 0.5) |
|---|---|---|
| T1 | sim-only, affected × {syn_0, syn_50}, 120s, EV only | 6 / 3 min |
| T2 | sim-only × 4 shapes vs the banked real legs + P11a-c | 38 / 22 min (both datasets) |
| T3 | real+sim pairs × 4 shapes; refreshes the bank | 82 / 47 min (both datasets) |
| T4 | campaign P1-P11 (`harness_campaign.sh` = `--tier T4 --pytest`) | 70 / 40 min per dataset |
| G0 / G0C | GPU 30 min screen: all six × syn_0 + syn_20, n 100/50 (FX-N34) / a second real leg per syn_0 cell (R7) | T4+G0+G0C ~6.9h per 8-GPU node |
| G1 / G2 | GPU 90 min: felix+fedbuff / the other four, reference config | ~2.5h per pair (cifar), in parallel with CPU (speech) |

`--changed <ref>` picks the affected baselines; `--exclude-phases` drops exact phase
ids. Whole-node jobs run last. Between leg starts (never mid-leg) the pool skips GPUs and cores another process is
using and checks live free memory, so it scales down under foreign load and back up when it clears (`LOAD` lines in
`pool.log`); `--gpu-ids` pins the usable GPUs.

**One node.** All pools run on jayne (parent R4); run dirs for both datasets live in `async_cifar10/experiments/`, and the
P7/P7o checkpoints are FX-N13 replay input.

**Tasks**
- P1-P5 · done. Smokes: `pool_smoke_T3/T2/G1` (cifar), `pool_smoke_ds` (both datasets, 8 legs in parallel),
  `pool_smoke_gsG1` (speech GPU), `pool_smoke_gate`; SIGINT tore down 4 slots in 11s. Their SUMMARY files remain;
  21 of their 30 run dirs were pruned on jayne (2026-09-27). Tests:
  `tests/harness/test_{slot_isolation,harness_pool,fl_data}.py`, `tests/launch/test_debug_run_dataset_profile.py`.
- P6 · isolation control · todo (operator, ~15 min): `$P --tier ISO --max-parallel 1 --output-dir
  $E/iso_solo`, then `$P --tier ISO_FILL --cpus-per-trainer 0.5 --output-dir $E/iso_packed`, then
  `examples/scripts/harness_iso_compare.py $E/iso_solo $E/iso_packed` ($P = `conda run --no-capture-output -n dg_flame python lib/python/examples/scripts/harness_pool.py`; $E = `lib/python/examples/experiments`).
  EQUIVALENT → 0.5 becomes the default. Measured: a small slot averages 1-2 cores (p95 4-6) with its aggregator.
- P7 · done: T4 on the pool, gate + `--pytest`; `harness_campaign.sh` is now a shim.
- P8 · in-process fast sim: fake trainer replies, no MQTT and no processes (~100×). It is the
  paper-experiment engine (parent S6). After P6.

## Next steps (persistent queue — top item is next)

**Counter (2026-10-01):** 24 open — 10 wip (fixed in tree, awaiting a run) · 7 todo · 5 blocked · 2 other (FX-N13 design,
FX-N8 likely closed). This session: 16 closed (FX-N26/N32/N35/N38/N44/N45/N46/N48/N50/N51/N52/N53/N54 + PR4/PR5), 5 opened (FX-N55/N56/N57/N58 + C10).

Work rule: PARITY C10 — each session resolves as many independent items below as it can, not just files new ones.

**Test levels.** pytest (P0/T0) = in-process unit/integration tests, no FL processes, ~4 min. CPU tests =
pool tiers T1-T4: real aggregator + trainer processes over MQTT on CPU (stub/tiny_cpu data, n 12-30), graded
by the event checker and the parity battery. GPU tests = G0 (30 min screen at n 100/50, + G0C real replicates) and
G1/G2 (the production path at the reference n, cifar 300 / speech 100, 90 min). "CPU+GPU" = one pool run with both.

**Unblock map.** Run 6 (PARITY PR7: speech L1-L4 never ran in run 5, L5 on both datasets) → Q2 floors (gate DIST) →
L6 = FX-N34 G0 screen → L7 = FX-N4 (G1) + FX-N5 (G2), + FX-N7 → FX-N6 + FX-N9 (GPU unavailability) → FX-N11 (speech GPU
parity) → FX-N12. FX-N13 design can start any time.

- **FX-N55 `[C]` · Unaware fedbuff selected on the oracle trace (C6) · wip: fixed in tree, verify (PARITY PR6).**
  `AsyncSelectorBase._eligible_candidates` filtered on the telemetry-only `PROP_AVL_STATE` stamp, so fedbuff
  (`avail_select_filter` false) never picked UN_AVL (5300/5300 AVL_TRAIN, both legs). → the asyncfl aggregator clears
  `filter_by_avl_state` for an unaware baseline (`unaware_ignores_trace`). Exposed a real race (EV17 7/14 on
  `pool_fxn54_verify`): a selector timeout reclaim freed an owed end and re-picked it in the same `select()` →
  a reclaimed end is not a candidate that pass. *Exit:* fedbuff syn_50 EV17 green; UN_AVL picks on both legs.
- **FX-N56 `[C]` · Sim freed an unaware async withheld slot at sct · wip: fixed in tree, verify (PR6).** Real can't
  see a send-gate, so the slot stays until the 90s timeout; sim freed it at the withhold. → asyncfl holds it in
  `_withheld_slot_held` until dispatch+90s (`sim_hold_withheld_slot`). `pool_fxn54_verify` fedbuff syn_50: in flight
  real 6.0 / sim 5.83, timeouts 21 / ~19 match, yet sim 1.75 vs real 3.62 s/round: real spent 3.75 slots on dead
  dispatches vs sim 2.85 (27 vs 19), the extra ≈ the 7 EV17 re-picks (FX-N55 race). `pool_fxn55b_verify`: EV17 green,
  speech stage 1 green, cifar K2 matched-window 9.1% (tol 8%), K3b/K4 green. A held pick counted as a recv end (its
  update is in the delivery ledger) livelocked a speech sim at vclock 599.7 → excluded from recv; its dispatch+90s is a
  starvation wake-up. `pool_fxn56d_verify`: stage 1 green on 3 of 4 syn_50 cells; fedbuff sim EV10 (a selector-timeout
  reclaim left a held end out of both slot and identity hold) → held set synced to `all_selected` (T1 EV green).
  *Exit:* fedbuff syn_50 K2/K3b/K4 green (run 6).
- **FX-N57 `[C]` · Sync baselines took stale updates · wip: fixed in tree, verify (`pool_fxn57_t3`, run 6).** Real syncfl
  counted a withheld delivery toward K (feddance round 101: staleness 31), sim aggregated it as a bonus. Operator: accept a
  late update only if the baseline's own rule allows it (REFL ≤ 5, oort none); FedDance is sync and never mentions stale
  updates → discard, like oort. → syncfl discards any stale update on both sides (`sync_accept_stale`, default false).
  *Exit:* feddance syn_50 K2 green; real in flight ≈ sim.
- **FX-N41 `[C][S]` · Unavailability parity family · wip: stage-2 checker roots fixed, verify (PR6).** syn_50 stage 2
  was red on every cell; three were checker artifacts: A5 (normalized bins on the 150s trace steps → absolute time,
  graded where both sides observed within 30s), A7 commit (each commit vouched 30s past a pre-boundary commit; 0 of
  ~5100 commit beliefs actually wrong → point accuracy, ±1s), and offline grades loading ground truth at the shell's
  trace scale (→ the run's `trace_time_scale`). fedbuff A2: sim listed an in-flight held pick as withheld, real as in
  flight (num_eligible 5.6-7.5 vs 9.2) → identity hold starts when the slot frees. Remaining: felix A2 KS 0.24 (means
  7.1/6.6), A3 late-run bins. *Exit:* each rung green or its root has an item.
- **FX-N49 `[C]` · REFL exploration port · wip: confirm on GPU.** CPU L2 refl EV/parity green; *Exit:* GPU refl distinct
  picks ≈ oort's (L5/L6).
- **FX-N42 `[S]` · Parity ladder · wip (PARITY_READINESS Active build: Q2-Q6).** *Exit:* Q2-Q6 done; a run graded per cell on both axes.
- **FX-N33 `[C][S]` · Aggregator aborts at interpreter exit · todo (root open).** After a clean channel leave:
  `terminate called without an active exception` → `Fatal Python error: Aborted` (a C++ thread destroyed while
  joinable; no Python frame). 16 of 661 run dirs from 09-27/28, 15 of them sim aggregators, CPU and GPU alike. Data is
  intact (post-leave); FX-D22 allowlists exactly this signature. Loaded natives: grpc, pyarrow (+ s3fs), torch, wandb,
  mlflow. 0 aborts in the 140 run dirs since `c2e4ac286` (3/119 earlier on 09-30, 43/577 on 09-29); that commit
  touches no thread/exit code, so the root is unnamed. *Exit:* root named, abort gone.
- **FX-N30 `[S]` · Speech tiny_cpu is too heavy for CPU slots · todo.** P7/P7o sims run at sim_rate 0.2-0.5 and are
  killed at 63-141s of 180 (EV0/EV12, `KNOWN`). In P7o real, the oracle's `select` blocks the MQTT thread 20-84s on 2
  aggregator cores. Options: shrink the speech tiny_cpu model/data, more aggregator cores for oracle legs, or run
  P7/P7o speech on GPU/P8. *Exit:* speech P7/P7o EV green; FX-N13 speech arms usable.
- **FX-N22 · Fast parallel harness (Active build) · wip: P6 isolation control next.** *Exit:* P6
  EQUIVALENT at cpt ≤ 0.5, and T2 for both datasets under 25 min on one node.
- **FX-N34 `[C][S]` · G0 GPU screen, all six × both datasets × syn_0/syn_20 · todo: L6, after run 6 L5.** 30 min,
  cohorts scaled with the reference c/n and aggGoal/c (cifar n=100 c=10 aggGoal 3, 3 GPUs; speech n=50 c=15 aggGoal 5,
  4 GPUs). G0C adds a second real leg per syn_0 cell for the real↔real floor (R7). *Predictions:* EV green on every leg
  except cifar fedbuff EV14 (FX-N15); syn_20 legs withhold and deliver. *Exit:* per cell INV/EXACT green → G1/G2; each
  red cell gets an item.
- **FX-N4 · First GPU block: felix + fedbuff, syn_0, 90 min · blocked: FX-N34.** `$P --tier G1 --datasets cifar10`
  (whole node, ~5h) and `--datasets google_speech` (~2.5h). jayne runs on 7 GPUs (GPU 1 ECC); each pair stays on one
  layout (L18). *Predictions:* EV green; felix real queue_wait p99 < 1s; `trainer_speed_identity` green. *Exit:*
  INV/EXACT green, convergence inside the replicate band, DIST residuals common-mode.
- **FX-N10 · google_speech on the launcher · wip: graded runs next.** Reference = 2024 SoCC n=100. *Next:* run 6
  speech L1-L4; GPU lr check in the first speech G1 (0.000195 vs 0.001 via `--trainer-hp learningRate=…`); target
  accuracy + stop rule (2024: 20 evals ≥ 60%); then S4 removes the 2024 JSON/scripts and the import script. *Exit:*
  all six real+sim graded on speech (T4 CPU + G1/G2 GPU).
- **FX-N19 · asyncfl sim serializes more than real at small n · todo (recheck on run 6).** Run 5 cifar syn_0/syn_0b/
  mobiperf felix/fedbuff K3b/K4 green; reopen only if run 6 or G1 shows it. *Exit:* root-caused, or graded against a
  real↔real floor (L12).
- **FX-N5 · syn_0 GPU block (G2): oort, oort_star, refl, feddance · blocked: FX-N4.** Same protocol.
  oort's open root: per-round `relative_change` of the exploited utility, binned by quartile, in both modes
  (don't touch the pacer). refl: confirm at 3h.
- **FX-N15 · fedbuff training diverges · wip: root fixed in tree, confirm in run 6.** Root: config drift. Server lr 40.9
  (`fedbuff.py` table) was tuned with client lr 0.000195, no decay (`30e20a3f6`, Feb 2024); the cifar template ran fedbuff
  trainers at 0.01 (~50×), so real went NaN at the first eval. → `datasets.yaml` `by_baseline` (a baseline's tuned values
  per dataset, over the dataset defaults); cifar fedbuff = 0.000195, no decay: `pool_fxn15_lr` syn_0/syn_0b pairs all
  green, real test loss 2.30 at round 50 (was NaN). Launch logs every lr/batch/optimizer knob with its source (`[HP]`);
  trainers log `[TRAINER_HP]`, fedbuff `[SERVER_LR]`. *Exit:* run 6 fedbuff EV14 green on every leg; then drop the `KNOWN` row.
- **FX-N58 `[C]` · Baseline hyperparameters from their sources · wip: fixed in tree, verify (`pool_hpval_t1`, run 6).**
  `lrDecay*`/`minLearningRate` had no alias (decay never ran); every baseline ran the template's lr 0.01 / batch 10.
  → aliases (decay default off) + `trainerOptimizer`; `datasets.yaml` `by_baseline` carries each baseline's
  paper/repo/our-2024 values per dataset, each field tagged with its source; `[HP]`/`[TRAINER_HP]`/`[SERVER_LR]` log them.
  *Exit:* every leg's `[TRAINER_HP]` equals its `by_baseline` row (both datasets).
- **FX-N13 · Streaming motivation experiment, both datasets · design after FX-N22 + FX-N10.** Show (a) per-trainer
  statistical utility changes as data streams in, (b) an unaware aggregator mis-selects, (c) one that tracks but
  mis-estimates utility still mis-selects. Pipeline: P7/P7o + `scripts/oracle_misselection.py` +
  `scripts/felix_streaming_figures.py`; cifar arms agree real/sim in sign (`pool_20260926_223155_T4/P7_figures`); speech
  arms starved (FX-N30). Next: arms onto the launcher (S3), horizon, operator's design, sweep.
- **FX-N7 · Remove legacy `trackTrainerAvail` (oort, oort_star, refl) · todo: unblocked.** refl EV green on every run-5
  leg. Delete the dead check and legacy branch (S4), then T12 retires. *Exit:* code gone, pytest green.
- **FX-N2 · Parent S2 (parity pipeline) for async_cifar10 · todo.** *Exit:* the stored Jun 23-24 pairs
  re-grade through it to within the floor, or each difference is explained.
- **FX-N6 · Unavailability design re-audit · todo.** Keep v1 semantics; check against the fwdllm invariants,
  logical-budget grading and the drain primitives. *Exit:* one audit table (item · keep/change · evidence).
- **FX-N8 · Concurrent-run confound on fedbuff · likely closed by FX-N22.** *Exit:* P6 EQUIVALENT.
- **FX-N9 · GPU unavailability: syn_20 → syn_50 → mobiperf_3st, all six · blocked: FX-N5, FX-N6.**
  *Exit:* A1-A8/K11 + the syn_0 ladder green; clean self-stop; withheld updates delivered; AVL_EVAL and the
  empty-pool cleanup exercised live on mobiperf_3st.
- **FX-N11 · google_speech GPU parity · blocked: FX-N9, FX-N10.** Reuse the FX-N4/5/9 protocol.
- **FX-N12 · Felix paper experiments, sim-only · blocked: FX-N11, FX-N22 P8.** List: open question below.

---

## Baseline hyperparameters (note, operator 2026-10-01)

Source of truth: `datasets.yaml` `by_baseline` (each field tagged [paper]/[repo]/[ours]); shared defaults are listed in its
header. Rules: a baseline uses its paper's values per dataset; lr decay only where the baseline configures it (REFL); a late
update is accepted only under the baseline's own rule (REFL staleness ≤ 5; oort, oort_star, feddance none). Judgement calls:
- REFL speech lr: paper Table 1 0.005 used; its repo config says 0.05.
- REFL staleness: paper default is no bound (≤ 5 only in its §3.2 study); kept `stale_update: 5` (operator).
- Oort/FedDance/REFL defer other knobs to FedScale defaults, which include lr decay 0.98/10; decay kept off except REFL (operator rule).
- Speech felix: the 2024 runs used Adam 0.04 (`_48h_oort` trainers), which stays at chance in a centralized check (FX-T27); kept 0.000195.
- The SGD papers' speech values run with `trainerOptimizer: sgd`; felix/fedbuff keep the dataset's Adam.
- Models differ from some papers (REFL/FedDance use ResNet18 on CIFAR-10; ours is CifarNet), so their lrs are a starting point, not a guarantee.

## Felix lessons (dos)

**Selectors (Oort family)**
- **FX-L1** Oort's pacer has two branches: a flat trend (`|Δ| ≤ 0.1·last`) raises `round_threshold`, a
  sharp one (`≥ 5·last`) lowers it. It fires on TRAIN only. Log `round_threshold` every round.
- **FX-L2** The UCB temporal term keys on the round of the last RECEIPT (`PROP_LAST_RETURNED_ROUND`),
  initialised at registration — never None, never the dispatch round.
- **FX-L3** Once a faithful controller still diverges, instrument its INPUT by quartile; stop touching the
  controller.
- **FX-L4** Async and sync selector knobs differ: Oort paper defaults are sync values. Per-round terms
  fire 2-3× more often in async. Felix uses `exploration_decay` 0.999.
- **FX-L5** Real records a stale-but-returned trainer's speed and utility too; skipping it makes Oort treat
  slow trainers as unexplored and re-pick them forever.

**Aggregation, ordering, clock**
- **FX-L6** Async (felix): drain each in-flight end's queue directly (`simSctOrderedDrain`) and hold busy
  slots until commit (`_sim_hold_busy_slots`). This one root cleared K3b/K2/U3/U6/K8/U2 together.
- **FX-L7** Sync oort over-selects ×1.3: a prior-round straggler with `sct` past the pinned round start is
  held in `selected_ends` and commits later (`simInflightCarryover`).
- **FX-L8** refl/oort: a trainer still computing (`vclock < sct`) is kept out of the eligible pool via the
  unavailable path, not `selected_ends`.
- **FX-L9** For a strict-barrier baseline (feddance, fedavg), real visibility lag is anchored on the barrier
  (`max_dur − dur_i`); streaming oort/refl stay per-message.
- **FX-L31** A committed end is RECVD so it leaves RECV; a buffered one reset to NONE then committed stays a
  phantom that blocks the starvation wake-up (P3 felix/fedbuff 6-min livelock).
- **FX-L32** A dispatch consumes the end's earlier receipt (eval reply, commit) and its RECVD state; else cleanup or
  recv frees the in-flight trainer and it is re-dispatched (FX-D16 22s tail; late withheld commits, FX-D16).
- **FX-L10** Split eval from train in any check that reads `agg_rounds`. Each eval gets its own `sct`,
  never the last train `sct`.
- **FX-L30** Anything replaying trainer data (oracle, replay) reads the trainers' stream clock: vclock in sim,
  wall since the broadcast `AGG_START_TS` in real — never 0.

**Availability**
- **FX-L11** Two separate axes per baseline: knowledge at selection (`avail_select_filter`: felix, oort_star,
  refl, feddance) and in-flight slot release (`proactive_inflight_evict`: felix only; others use the 90s
  vclock abandon).
- **FX-L12** If a trainer drops mid-flight, its compute still completes; the send is gated (real), or
  buffered to `delivery_ts = max(sct, next_avail)` (sim), and it commits late as stale. Nothing is dropped.
- **FX-L13** Under scarcity, sim jumps the vclock to the next availability transition; size `--runtime-s`
  for it rather than subtracting the jumps.
- **FX-L14** In-flight count differs per baseline (oort over-selects, async is bound by concurrency, refl
  and feddance clear each round). Size n ≈ threshold / (1 − unavailable fraction).
- **FX-L33** With the substrate on, only withhold/evict/abandon release an in-flight slot; the channel's MQTT
  UN_AVL release re-dispatched trainers mid-update and a same-end eval overwrote the buffered train (P2 EV10/EV5).
- **FX-L34** Size a harness phase for its trace: mobiperf_3st has ~10% AVL_TRAIN from t=0, so n ≈ select / 0.1 × 1.5
  (P3: n=45 for sync select 3; select 13 needs n ≈ 200).
- **FX-L35** Parallel legs need a slot each: own physical cores (`taskset`), private broker, run tag, the
  aggregator's solo core share (`FLAME_AGG_CORES`). Prove density with the ISO control before raising it.
- **FX-L36** Set a test shape through the launcher's `experiment.aggregator.agg_goal`: it fans into
  `aggGoal` and sync `aggr_num` as the last merge layer, so K = aggGoal for sync baselines.
- **FX-L37** A dataset differs by `datasets.yaml` + `fl_data.py` only; baselines, selectors and the sim/real
  machinery stay shared (google_speech = n=100, ResNet34-1D, Adam, 29 MB updates).
- **FX-L38** The first CUDA touch in a process (incl. the first CPU backward: autograd queries the device
  count) initializes the driver, serialized across processes (0.26s alone, 9-35s in a busy pool). Pay it in
  `initialize()` (`_warmup_device`), never inside a timed task.
- **FX-L39** Time sim-speed guards (wall ceiling) from the join barrier, like real's trace origin.
- **FX-L25** A sim starvation wake-up is the earliest FUTURE event that frees a slot: availability
  transition, withheld delivery, residence-held `sct`. A due entry is no wake-up (it stops the run).
- **FX-L43** Size a leg for its trace: over runtime × trace scale it must see ≥ 80% of the named unavailability
  (`effective_unavailability`; pytest over T4/G0) and a few outage/up cycles per trainer for the mechanism under test.
- **FX-L44** Trace time 0 is the join barrier (real re-anchors `agg_start_time_ts`; sim's vclock starts at round 0). Spawn
  stagger and registration fall before it (an UN_AVL trainer still joins), so a trace needs no all-available head; a
  trainer holds trace time at 0 until its first dispatch brings the origin (FX-D18).
- **FX-L15** The ramp is syn_0 (byte-identical to availability off) → syn_20 → syn_50 → mobiperf. 2-state
  traces collapse AVL_EVAL (`_trace_has_avl_eval`); only mobiperf exercises it.

**One task per version, rounds**
- **FX-L26** One task per (trainer, model version): train at v blocks train and eval at v (a train already
  returned the utility); eval at v still allows train at v (operator). Default `taskRetryPolicy: none`: the 90s
  timeout frees the slot and the trainer waits for the next version. `fixed`/`exponential` are A/B only.
- **FX-L27** Identity of an update is (trainer, version): dedup commits on it; the trainer drops a request it
  already answered; a late update from an older dispatch never answers the end's newer one.
- **FX-L28** A round advances the model version only when `agg_goal` updates aggregate (FX-N37); a starved, empty or
  all-stale iteration keeps it and still frees every consumed slot. Test the cache, not the optimizer's result.
- **FX-L40** Time out each trainer 90s after its own dispatch, both modes, applied at dispatch time; a timeout never
  closes a round (FX-N37). A per-recv timeout waits (1+K)× and overruns the budget (FX-D17).
- **FX-L41** Accumulate model state in float and cast back to each tensor's dtype once; apply baseline-specific
  server steps to parameters (float) only. Integer buffers exist only in some models (BatchNorm) (FX-D17).
- **FX-L42** A sync round that commits nothing re-dispatches at the same version: select afresh, excluding ends
  already tasked at it. A selector's round cache serves only the same round's RECV lookups (FX-N31).

**Reading the checker**
- **FX-L29** Stub legs charge a seeded compute span (`flame.harness.stub_compute_s`, fit to run_20260702 real). Those
  runs fell back to CPU (CUDA failed to start before S0), so refit it from a G0/G1 real leg (L17).
- **FX-L16** A2 failing (KS) while S3/4 passes is one in-flight gap graded at two tolerances; walk to
  residence, not eligibility.
- **FX-L17** U6 KS on a sub-ms point mass is signal-free; read `mean_diff`. P3 sub-second opposite-sign
  offsets are wall-capture; trust P3 only when `grid_KS` also fails.
- **FX-L18** `gate_holds = 0` over a whole run means the gate is structurally inert, so suspect the
  accounting upstream. Measure one-in-flight from overlapping dispatch→commit intervals, not warning
  counters.
- **FX-L19** `phase_gpu_compute` and refl K2 are sensitive to run length; re-check at ≥2.5h before acting
  on a short-run FAIL.
- **FX-L20** Run lengths: telemetry sanity 5-10 min · one mechanism rung 45 min · compounding clock residual
  (K2/K3b) 90 min-2h · refl low-frequency K2 drift 3h · C1/C2 sign-off 3-4h. Smoke 5 min first.
- **FX-L21** `Sdet` triage: eligible set differs + aggregates match = stochastic, PASS; + clock diverges = fix
  the clock; eligible set matches but decisions differ = the score VALUES diverge.
- **FX-L22** Past-dating comes in two streams (train-only U6 vs all-commit SIM_CLOCK_DIAG); always say which
  one a number came from.
- **FX-L23** `sim_committed_fresh == agg_goal` confirms oort's block-for-K fix; a later fresh-count gap is a
  different root.
- **FX-L24** Only re-run real when the real path changed; sim-only changes grade against the stored real dir.

## Felix tripwires (don'ts)

**Selectors (Oort family)**
- **FX-T3** Don't add a `system_util` recency guard or widen the slow-speed tail for oort carry-over decay.
- **FX-T6** Don't key oort latency by task type (sync oort sends no eval tasks).
- **FX-T8** Don't expect seeding to align per-round selection sets; judge S2 by speed class.
- **FX-T17** Don't expect oort carry-over decay to be a run-length transient (structural), or the felix
  min-budget seed alone to fix past-dating.

**Aggregation, ordering, clock**
- **FX-T1** Don't enable `simStaggeredRedispatch` (falsified; kept off) or retune `simRedispatchGapSeconds`.
- **FX-T2** Don't add `mqtt_fetch` (~20-57s) to `sct`; it is the wait before re-selection, not transfer.
- **FX-T4** Don't tune the `_sim_recv_min` gate predictor; `exp == sct` exactly.
- **FX-T7** Don't clamp felix's clock jump or pace dispatch to fix "fresh" past-dating; that was eval
  reusing a stale `sct`.
- **FX-T14** Don't add prediction-only gates with no real blocking (they never fire), or a `version_at(sct)`
  staleness relabel (inert).
- **FX-T15** Don't read high wall-clock commit density as sim "running fast"; judge per-round vclock
  advance and `commit_gap`.
- **FX-T18** Don't add a scalar fudge for P3's ~1s `mean_overhead` offset (wall-capture).
- **FX-T20** Don't gate sim ingestion on "committed this agg cycle"; a re-dispatch starts a new outstanding
  update. Stranding it livelocked P3 felix/fedbuff (FX-D6).
- **FX-T21** Don't call `ends()` for anything but a real dispatch: RECV or state-less calls run the selector
  (feddance phantom picks, oracle crash). List the pool with `all_ends()`.
- **FX-T22** Don't skip a trainer after the selector chose it; the selector keeps a slot nothing was sent to.
  Exclude at eligibility (the removed `[SELECTION_CHECK] Skipping`, 46-143 per felix run).

**Availability**
- **FX-T5** Don't hold ALL buffered ends out of refl's pool (`pending_ends`); it over-holds.
- **FX-T9** Don't express "busy" through the UN_AVL list, including in any unavailability redesign.
- **FX-T10** Don't use per-tick MQTT availability broadcasts (comms storm); v1 reads the trace.
- **FX-T11** Don't exclude AVL_TRAIN from eval on a 2-state trace (it empties the pool and wipes
  `selected_ends`).
- **FX-T12** Don't zero the legacy `trackTrainerAvail` block before `simUnavailability` is set
  statically (FX-N7).
- **FX-T19** Don't fork withhold/abandon logic per stack (one shared `ClientAvailability`), or grade A4 on
  the bare transition fraction (use A4dur).

**One task per version, rounds, baselines**
- **FX-T24** Don't add a staleness cutoff to fedbuff: its 1/√(1+s) discount is the baseline (operator).
- **FX-T26** Don't set `selector.kwargs.aggr_num` or `aggGoal` directly in an experiment: `agg_goal`
  overwrites them (a small-n oort smoke kept K=10 > n and starved).

**Harness and runs**
- **FX-T31** Don't retune `simCompletionLegSeconds`, `simRedispatchGapSeconds` or `simCommitOverheadSeconds` to close a
  clock rung: they are single-config fits; profiled charges replaced them (FX-D23).
- **FX-T13** Don't grade a fedbuff/felix real run while another n=300 real run shares the broker (FX-N8).
- **FX-T16** Don't re-chase GPU contention (overrun 0), SEND_TIMEOUT or MQTT drops at n=300 cifar; all
  measured 0.
- **FX-T23** Don't run more stub trainers than node cores (1 core each): n=200 on 128 cores crawled the sim to
  its wall ceiling at vclock 6.
- **FX-T25** Don't kill FL workers by process name alone; scope by `FLAME_RUN_TAG` (`_expt_pids`,
  `slot_pids`). The runner's pre-leg `pkill -9` would have killed every neighbouring slot.
- **FX-T28** Don't derive a leg's health from the shared `experiments/` dir; parallel neighbours' logs leak
  in (FX-D13). Read the leg's own run dirs.
- **FX-T29** Don't write a replay input in place or from a daemon thread: a writer killed at exit truncates it (FX-N33).

**Datasets**
- **FX-T27** Don't take trainer lr from the 2024 speech "_oort" configs (Adam 0.04 stays at chance).
- **FX-T30** Don't name a trace by intent; name it by its measured full-day unavailable fraction (the old "syn_20" was
  10.8%, FX-D18).

---

## Built (current capabilities; one line each — details live in code, tests and `git log`)

IDs are kept because code comments cite them.

**Simulator fidelity**
- **FX-D23** Sim non-compute charges are profiled per (dataset, harness, stack) from real legs
  (`scripts/profile_felix_charges.py` → `async_cifar10/sim_charge_profiles/`), applied to all six by `debug_run.sh`:
  completion leg + dispatch latency (`SIM_CHARGES=legacy` reverts); async commit order exact (`simOrderSlackSeconds` 0).
- **FX-D24** Real async frees an ingested trainer's slot at SEND, not one arrival later (`release_recvd_at_send`).
- **FX-D26** (code cites FX-N26/N38/N44/N45/N46/N50) Run-4 roots, confirmed by run 5 cifar L1-L4
  (`ladder_20260930_054655`): warm-up builds an optimizer (first-task cold start); sim syn_50 ordering (EV10/11/16 green);
  real duration excludes `SEND_GATE_WAIT_S`; trace origin sent at the join barrier (A6r green on mobiperf); sim holds an
  ingested trainer's identity until aggregate; join barrier = whole cohort and `real_hold_owed_ends` (EV17 green).
- **FX-D28** (code cites FX-N54) asyncfl sim re-injects due withheld deliveries before starving (`sim_reinject_when_idle`);
  EV16 fails a delivery committed > 10s late. felix syn_50 K2/K3b/K4 green on both datasets (`pool_fxn54_verify`).
- **FX-D29** (code cites FX-N35) Pool SUMMARY lists each leg's UN_AVL share at selection (`leg_unavailability.py`); the
  real bank hashes the trace store, so a regenerated trace stales its banked real legs.
- **FX-D31** (code cites FX-N32) P11b (`order_by_sct`) runs on fedbuff, which withholds: CAUGHT on both datasets with
  607/703 unheld UN_AVL commits (`pool_fxn32_p11b`; felix had 0-1).
- **FX-D30** Grader reads the run's own `max_experiment_runtime_s` (L29; K5 used the CLI budget).
- **FX-D27** Oort-family selection follows the REFL fork's `getTopK` (exploit ≤ explored−1, explore the remaining slots,
  random pad) and keeps below-cut-off ends in the exploit pool (oort/oort_star/refl EV + syn_0 parity green, run 5).
- **FX-D25** Checker times both sides from run start (first train selection), not the first commit (K2/K8/U2).
- **FX-D1** Core sim fidelity: sct-ordered drain, one-in-flight residence, oort carry-over, refl pool exclusion,
  intrinsic selector duration, UCB temporal term, faithful pacer, feddance barrier-anchored U6.
- **FX-D4/D8** Cold-start gate (`simColdStartGate`, default on) and uncapped busy hold, freed on abandon/evict;
  one-in-flight holds off syn_0 (felix/fedbuff sims EV10 = 0 on syn_20/50/mobiperf_3st, cifar T4; P4 control fails).
- **FX-D6** asyncfl sim: no stranded same-cycle re-dispatch; inner loop honours `_work_done`; refl residence
  starvation wake-up.
- **FX-D9** One task per version: dispatch ledger + no-repeat guard (a train at v also blocks eval at v),
  `taskRetryPolicy` (default none), (trainer, version) commit dedup, trainer-side discard, EV15.
- **FX-D10** Sync rounds advance only on a committed aggregation (non-empty cache); an all-stale round frees its slots.
- **FX-D20** (code cites FX-N37) `syncWaitForK` (oort, oort_star, refl, feddance): a version advances only on `agg_goal`
  accepted updates; each pick times out 90s after its own dispatch and is topped up at the same version. Run 2: EV7 green
  on every sync leg, P3 oort 3 vs 3 commits on both datasets (was 2 vs 84).
- **FX-D21** (code cites FX-N31) A sync round that commits nothing re-dispatches from the FX-D9 ledger, never the round
  cache; `run_end` records the stop point for EV12 (sim vclock, real ts). Run 2: P3 refl/feddance EV1/EV12 green on both.
- **FX-D17** Sync stacks (syncfl, oort) abandon a trainer 90s after its dispatch in real too (shared
  `_abandon_stalled`); a real round's recvs stop at the latest awaited trainer's timeout (`recv_fifo(deadline=)`;
  the first one under `syncWaitForK`, FX-N37);
  an abandoned trainer's late update is still received, stale-gated. refl sums integer buffers in float and applies
  YoGi/QFedAvg and its staleness norm to float tensors only (reference REFL: `model.parameters()`).

**Availability**
- **FX-D2** Unavailability v1 substrate for all six: send-gate / deliver-late, two ledgers, proactive evict,
  starvation advance.
- **FX-D5** Real withheld updates are delivered (bool send-gate fix); `mobiperf_3st` launchable.
- **FX-D12** Under the substrate only withhold/evict/abandon free an in-flight slot; committed ends leave RECV;
  a dispatch consumes the end's earlier receipt.
- **FX-D16** (code cites FX-N18) Real asyncfl ingests updates on arrival: no settle sleep for felix/fedbuff (`baselines.yaml`);
  recv also reads ends whose update arrived or is still owed (timed out / withheld, `_owed_ends`, FX-L27).
  T4 real queue_wait p99: cifar ≤ 0.06s on every felix/fedbuff leg (0.1s-settle control P10: 0.227s); speech ≤ 0.43s outside FX-N30.

**Harness, checkers, datasets**
- **FX-D19** (code cites FX-N36) flame processes poll NVML only with `FLAME_STAT_THREADS=1`: run 2 GPU legs joined 44-99s
  after their first trainer, max `[WARMUP]` 1.35s (was 10-min joins).
- **FX-D22** (code cites FX-N40) Fail fast: `examples/scripts/fail_fast.py` flags EV0's fatal lines (Traceback, Fatal
  Python error, Segmentation fault; the FX-N33 abort allowlisted by signature). `harness_pool.py` scans live legs every
  ~30s and on completion → teardown, `ABORT.txt`, partial SUMMARY, rc 3; standalone `harness_suite.sh` stops after the
  pair; `--no-fail-fast` opts out. `--inject-bug trainer_crash` stopped a smoke pool 61s in (`pool_smoke_fxn40_abort2`);
  on run 2 it flags exactly the 9 speech OOM legs. Tests: `tests/harness/test_fail_fast.py`.
- **FX-D7** Trainer availability thread stops at EOT or shutdown (SIGTERM/atexit); clean teardown.
- **FX-D11** Streaming + oracle harness (P7/P7o) with offline replay and figures.
- **FX-D13** Event checker EV0-EV16 + injected bugs (P11a-c); parallel isolated pool with tiers T1-T4/G1/G2,
  real bank, `--changed`, `--shard`, gate (FX-N22). Run dirs are `run_<ts>_<phase>_<name>` (`FLAME_RUN_LABEL`) and never reused; health reads
  `FLAME_RUN_DIR_FILE`; the leg watchdog budgets the sim wall ceiling; sub-0.2s phases grade on mean (50 ms).
- **FX-D15** No cold start in timed tasks: startup warm-up (CPU + GPU; frees its activation cache), CUDA-only sync in the weights phase,
  sim wall ceiling from the join barrier (first-task compute 9-14s → 0.1s; fedbuff EV11 fixed).
- **FX-D18** Traces sit at their named level from t=0: `syn_10` (11%, the old syn_20, origin +600s), `syn_20` (20%, new,
  stationary, `_metadata/scripts/gen_synthetic_trace.py` seed 20), `syn_50` (50%, origin +2400s), mobiperf (origin
  +300s: the injected 5-min head) via `_metadata/scripts/shift_trace_origin.py`; syn_* extended from 24 h (then all
  UN_AVL) to mobiperf's 149 h with the same chain (`gen_synthetic_trace.py --extend`; 13 MB, parsed once per process
  with the C loader by `trace.read_trace_file`, 9 s); `effective_unavailability()`; any `syn_*`
  reaches real trainers (spawner + trainer); a real trainer's trace clock holds at 0 until the aggregator's origin
  arrives (it used to run from process start and pop transitions up to ~5 min early). T4 syn_50 legs 1200s (~2 cycles per trainer), P3 n=45, G0 uses syn_20.
  Tests: `tests/availability/test_synthetic_trace_fractions.py`, `test_harness_pool.py`.
- **FX-D14** Dataset switch (`fl_data.py`, `datasets.yaml`, `data_roots` = /coc/scratch/dgarg/fl_datasets); google_speech on the
  launcher (FX-N10).

## Open questions (operator)

- Felix paper experiment list, and the exact streaming-experiment design (FX-N13), once FX-N22 and FX-N10 land.
- Make `real_drain_ready_ingest` the default (R9)? Cifar T4 P9 vs P1: RECV_FIFO skips 887→0 (fedbuff) and 908→0
  (felix); queue_wait p99 0.028 vs 0.035s (fedbuff) and 0.019 vs 0.025s (felix); fedbuff parity 0.984 vs 0.952; no new EV fail.
  Speech P9 is mixed: skips go to 0, but felix p99 is 0.286s vs 0.186s (P1), and speech P1 fedbuff is void (shared run dir, FX-D13).
