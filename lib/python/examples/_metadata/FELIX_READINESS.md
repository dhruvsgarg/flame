# Felix readiness — backprop FL (async_cifar10, google_speech)

## Preamble
- **Read [ROBUST_FL_READINESS.md](ROBUST_FL_READINESS.md) first.** Its doc rules, operating rules (R1-R13),
  shared lessons (L) and tripwires (T) apply here and are not repeated. This doc holds only what is specific
  to Felix: weight-aggregating backprop FL, round/`agg_goal` progress axis, Oort-family selectors.
- **CURRENT FOCUS.** Goal: Felix feature-complete (syn_0 + unavailability, both datasets) and paper
  experiments running sim-only, with **every baseline correct on both sides**: reference semantics and the target accuracy
  (cifar 50%, speech 60%) reached in its window on full data (Accuracy table), not only real↔sim agreement. Per run: root-cause every red cell (or as many as the logs allow) and fix it, so
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

Run-by-run green/known/red, open/closed items and per-baseline parity + accuracy: [PARITY_READINESS.md](PARITY_READINESS.md) → Progress dashboard, Felix scoreboard.
pytest: 2416 passed · 0 failed · 7 skipped (2026-10-07, with FX-D64).

**Accuracy table (FX-N74; `scripts/accuracy_table.py --runs <run dirs>`)** — test accuracy % at time on the leg's own clock (real
wall, sim vclock), syn_0, reference n. Targets: cifar 50%, speech 60%. G1A = full data (run 16, `block_20261006_run16`; evals every 40
rounds; run 17 `block_20261006_1943` speech + cifar G1AS sims, evals every 20 rounds, FX-D63); G2 rows stream data (0 → 100% over 3 h: lower bounds). `-` = no eval in that window.

| dataset · baseline | leg | real: @30 / @60 / @90 min · max | sim: same | note |
|---|---|---|---|---|
| cifar · felix | G1A | 36.2 / 51.8 / 56.9 · 57.0 | 40.6 / 50.6 / 53.7 · 56.3 | **target at 57 min real, 60 min sim** |
| cifar · refl | G1A | 24.9 / 32.2 / 44.5 · 44.5 | 29.1 / 35.7 / 44.1 · 44.1 | rising at 90 min; K2 2% |
| cifar · fedbuff | G1A | 20.0 / 24.0 / 32.7 · 34.7 | 23.1 / 31.5 / 31.7 · 38.1 | below target both sides; K2 0.3% |
| cifar · oort, feddance | G2 stream (`block_20261003_1311`, `pr12_cifar_20261003_0322`) | oort 24.8 @60 · 37.7; feddance · 16.6 | oort · 31.4; feddance · 35.4 | |
| speech · fedbuff | G1A | 24.5 / 41.4 / 48.3 · 48.3 | 30.0 / 42.5 / 52.1 · 52.1 | rising at 90 min |
| speech · refl | G1A | 3.9 / 29.3 / 38.7 · 38.7 | 3.9 / 31.0 / - · 39.7 | 82 rounds in 90 min (65 s/round, K fresh) |
| speech · felix | G1A | 9.2 / 38.7 / 35.7 · 40.4 | 21.8 / 37.9 / 50.7 · 50.7 | finite (FX-D61); ±10-pt eval swings, real 21 → 9% at 30 min; K2 1.2% |
| speech · oort, feddance | G2 stream (`block_20261004_0543/gs_g2`) | oort **60.7 at 80 min**; feddance · 21.5 | oort · 57.2; feddance · 45.2 | |

**Cross-cutting capabilities**

| capability | status | evidence / item |
|---|---|---|
| checker catches injected sim bugs (P11a-c) | ✅ both datasets | run 6 L4 (`ladder_20261001_051724`) |
| fail fast on fatal errors | ✅ | FX-D22 |
| parity ladder runner + offline re-gate | ✅ | FX-N42 |
| GPU join / warm-up at n=100 | ✅ 44-99s / ≤1.35s | FX-D19 |
| sync wait-K, round livelock | ✅ EV7 green all sync legs | FX-D20, FX-D21 |
| async real ingest latency (< 2s) | ✅ run 3: max < 2s outside P7/P7o except 4 legs with one 2.2-3.2s update | FX-D16 |
| GPU legs start (no OOM), EV green | ✅ run 6 L5: EV green on all 24 GPU legs | FX-D19 |
| sim clock charges profiled per dataset/platform | ✅ K3b green on every syn_0 cell, CPU + GPU, both datasets (run 6) | FX-D23 |
| DIST tolerances sized by a replicate floor | 🟡 real↔real floors (n=2) from G0C (syn_0/20) and G0UC (per G0U cell, reused by G0T) gate the regrade; nothing tightened yet | FX-D37, FX-D52, PARITY Q2 |
| streaming / oracle experiment (FX-N13) | 🟡 ST1-ST5 cifar: EV19 green on every G0T/G0To leg (after FX-D51 origin fix); parity per cell = its G0U cell; oracle advantage ungraded (1 eval per 15-min leg) | FX-N13 |
| sim speedup vs real | 🟡 cifar n=300 syn_0: 4.5× (felix) / 7.4× (fedbuff); 🔴 speech n=100: 1.1× fedbuff, 0.6× refl (GPU-bound: the sim does the real compute, ≈ D for fast trainers) | S6, FX-N22 P8 |
| paper experiments sim-only | ⬚ | FX-N12 |

Stored June REF grades (🟡, pre PRs #72-#85): `async_cifar10/experiments/parity_{felix,oort,refl}_20260624_{5400,3h}.json`,
`parity_feddance_20260623_3h.json`.

---

## Active build — fast parallel harness (FX-N22)

The CPU harness catches logic bugs (event invariants, injected bugs); distribution parity is decided on GPU.

**Isolation contract: a parallel leg must behave as if it ran alone.** Each slot gets:
- disjoint physical cores via `taskset`. The launcher pins the aggregator (`FLAME_AGG_CORES`: 2 at n≤30,
  8 for GPU legs) and the trainers inside that affinity;
- a private mosquitto (`FLAME_MQTT_BROKER`). MQTT client ids are fixed task ids, so two runs on one broker
  kick each other off;
- a run tag (`FLAME_RUN_TAG`) scoping every kill and clean-slate sweep, and an exact run-dir handoff
  (`FLAME_RUN_DIR_FILE`);
- GPU legs: the healthy GPUs they ask for (FX-T36 density), CPUs = 8 + 0.4·n (cifar 0.2·n, FX-D49): speech n=100 takes 48.

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
| G0 / G0C | GPU 30 min screen: all six × syn_0 + syn_20, n 100/50 / a second real leg per syn_0 cell (R7) | T4+G0+G0C ~6.9h per 8-GPU node |
| T3S | oort/oort_star/refl pairs with aggGoal 10 (stragglers cross rounds), syn_0 + syn_20, FX-D39 | 24 min (6 pairs) |
| G2S / G2C | one G2 cell replicated sim-only (5400s, ~8 min) / real-only (97 min): floor for a chaotic selector (FX-N68, FX-D37) | 8 / 97 min |
| G1L / N62 | felix syn_0 7500s pair + a real replicate (FX-N65) / unaware oort syn_50 CPU pair at 3h (FX-N62) | ~4.8h whole node / ~5h |
| TS / TSo | streaming CPU pairs (tiny_cpu) × {linear, events} × {syn_0, syn_50}; TSo adds the oracle arms (FX-N13) | ~7 min smoke |
| G0UC / G1U | a real replicate per G0U cell (floors G0U + G0T) / felix+fedbuff syn_50 at the reference n, 90 min, whole node (FX-D52) | cifar 33 min / ~5h serial |
| G0T / G0To | streaming GPU screen, G0U cohort × {linear, events} × {syn_0, syn_50}, horizon = leg's trace span; G0To = oracle arms | cifar ~2h at 0.2 CPU/trainer |
| N64 / N64S | FX-D38 refl speech health: 25 min real+sim pair + a real replicate / 40 min sim-only (EV18) | 45 / 27 min |
| G1 / G2 | GPU 90 min: felix+fedbuff / the other four, reference config | ~2.5h per pair (cifar), in parallel with CPU (speech) |
| G1S | felix + fedbuff at the reference n, 45 min, 4 GPUs (fix confirmation, C11) | speech ~100 min |
| G1A / G1AS | accuracy: reference n, syn_0, full data, eval every 20 rounds, 90 min, 4 GPUs per leg (FX-N74) / its sim legs only | ~2.5h per pair (cifar) / 25 min |
| T3C | a real replicate per T3 cell (real↔real floor, read by `--regrade`) | ~22 min |
| G0U | unavailability screen: syn_50 + mobiperf_3st, n=50, trace scale 4; cifar 15 min on 1 GPU, speech 30 min on 2 (FX-N9, C11) | cifar all six ~90 min |

`--changed <ref>` picks the affected baselines; `--exclude-phases` drops exact phase
ids. Whole-node jobs run last. Between leg starts (never mid-leg) the pool skips GPUs and cores another process is
using and checks live free memory, so it scales down under foreign load and back up when it clears (`LOAD` lines in
`pool.log`); `--gpu-ids` pins the usable GPUs.

**One node.** All pools run on jayne (parent R4); run dirs for both datasets live in `async_cifar10/experiments/`, and the
P7/P7o checkpoints are FX-N13 replay input.

Tests: `tests/harness/test_{slot_isolation,harness_pool,fl_data}.py`, `tests/launch/test_debug_run_dataset_profile.py`.

## Active build — streaming time-to-accuracy experiments (FX-N12, FX-N13)

**Experiment (operator).** All six baselines × {cifar10, google_speech} × {syn_0, syn_20, syn_50, mobiperf_3st} × two
stream modes; metric = time-to-accuracy (targets: cifar 50%, speech 60%); each trainer's data reaches 100% at 6 days of
trace time; sim-only once streaming parity holds. Claim: Felix's eval selector finds clients whose data (hence utility) just
grew; unaware or mis-estimating selectors don't.
- **linear:** every trainer starts at 10% of its data and grows linearly to 100% at the horizon.
- **events:** starts at 10%; 9 chunks of 10% land at seeded uniform times in [0, horizon], out of sync across trainers.
- Decided (operator 2026-10-05): stream clock = trace time with its own horizon (6 days for paper runs, the leg's trace span
  for screens); chunks land regardless of availability; new data is more of the trainer's own shard (same distribution).

**Built (FX-D47).** `async_cifar10/stream_schedule.py` is the one schedule for trainer, oracle injection, offline replay
(`oracle_misselection.py`) and EV19. Harness tiers TS/TSo (CPU), G0T/G0To (GPU). Telemetry: `trainer_round` carries the
count trained on and its `stream_clock_s`.

**Steps left:** ST5 speech (G0T felix re-check, PR17; other baselines after FX-N70; P7 after FX-N30) · ST6 sim-only sweep (needs
FX-N22 P8) with time-to-accuracy and misselection-gap figures; the oracle-advantage check lives there (screens evaluate every 50 rounds).

## Unavailability v1 audit (FX-N6, 2026-10-04): keep v1; change and open rows are follow-ups

| item | verdict | evidence |
|---|---|---|
| trace-read knowledge; select filter per baseline (aware: felix, oort_star, refl, feddance) | keep | EV17; unaware fedbuff picks UN_AVL 0.51/0.53 (FX-D32) |
| compute completes, send gated, delivered late and stale | keep | FX-D5, EV16 (FX-D28), withheld slot held (FX-D34); async sim delivers before a later sct (FX-D50); real gate reads the trace at send (FX-D55) |
| two ledgers; commit order `(delivery_ts, end_id)` | keep | P11b CAUGHT on both datasets (FX-D31) |
| busy ≠ unavailable ≠ withheld; slot freed only by withhold/evict/abandon | keep | P11a CAUGHT; FX-L33, FX-D12 |
| all availability time on the vclock; origin = join barrier | keep | P11c CAUGHT; A3, A6r green (FX-D26) |
| 90s abandon per dispatch, never closes a round | keep | EV7, K3s (FX-D20, FX-D33); real async wakes at it, not the next 30s poll (FX-D50) |
| felix-only proactive evict; evicted update ingested | keep | felix syn_50 A2 6.9/6.9 (FX-D34) |
| starvation wake-up = earliest future slot-freeing event | keep | FX-L25, FX-D28 |
| one task per (trainer, version) under unavailability | keep | EV15 (FX-D9); fwdllm port is S8 (FluxTune parked) |
| logical-budget grading; timing on stall-free rounds | keep | sync stalls by cause (FX-D48); n=50 syn_50 stall placement is chaotic, graded on G0UC floors (FX-L53) |
| AVL_EVAL + empty-pool cleanup (land-mines 8, 13) | open | G0U mobiperf ran on GPU (EV green); counts not yet read |
| A2 two-tolerance shape (land-mine 3) | open | n=50 G0U: only refl syn_50 A2 red (KS 0.24, stall placement); n=300 = G1U |
| streaming clock vs trace clock | done (FX-D47) | stream clock × `FLAME_TRACE_TIME_SCALE` = trace time, same origin as availability |
| legacy `trackTrainerAvail` | change | FX-N7 (delete) |

## Next steps (persistent queue — top item is next)

Work rule: PARITY C10. Run queue and pre-launch checklist: PARITY_READINESS.

**Test levels.** pytest (P0/T0) = in-process unit/integration tests, no FL processes, ~4 min. CPU tests =
pool tiers T1-T4: real aggregator + trainer processes over MQTT on CPU (stub/tiny_cpu data, n 12-30), graded
by the event checker and the parity battery. GPU tests = G0 (30 min screen at n 100/50, + G0C real replicates) and
G1/G2 (the production path at the reference n, cifar 300 / speech 100, 90 min). "CPU+GPU" = one pool run with both.

**Unblock map.** FX-D50 + FX-D55 (CPU-confirmed, runs 15-16) → GPU G0U/G0UC, then G1U → speech FX-N9. FX-N70 (refl) + FX-N5 (speech feddance floor) →
FX-N11. FX-N12 also needs FX-N13 (ST6) and P8 (speech sim is GPU-bound).

- **FX-N74 `[C]` · Every baseline reaches its target on full data, both sides · wip: run 18 G1A speech on SGD (PR18).** Accuracy table above.
  Cifar felix reaches 50% (57/60 min); every other G1A row is below target at 90 min (speech 40-52%, cifar fedbuff 32-38%, refl 44%).
  Speech oort (SGD 0.04, b16) reached 60.7% at 80 min even streaming. In-process (`fl_lr_check.py --staleness 5 --flame-opt felix`, 40
  rounds, then 120): at r120 Adam 0.000195 × 1.0 = felix 26.8 / fedbuff 33.6%; SGD 0.04 b16 × 0.5 = 49.2 / 39.5%, × 1.0 = 41.3 (swings) /
  49.5%. Streamed oort's 60.7% is SGD + sync, not streaming (no same-baseline stream vs full pair yet). *Next:* run 18 G1A speech felix/fedbuff
  on SGD (operator 2026-10-07) and oort/feddance (PR18). *Exit:* G1A real and sim reach target per baseline, or a named root; real↔sim acc
  diff within the real↔real floor.
- **FX-N9 · GPU unavailability: syn_50 → mobiperf_3st, all six · wip: CPU confirmed (run 16 T3: async 6/2/0, FX-D55 gates real at flips); GPU G0U + G1U next.**
  Cifar G0U 9/0/3 on G0UC floors; syn_20 G0 green (L6). Open: cifar refl T3 syn_50 S1/S8 (a 0.45 s clock offset after 72 rounds puts one
  0.75 s task on either side of the 150 s flip, FX-L53: run 17 T3C floors it); fedbuff K4 + oort mobiperf length (FX-D52); oort_star G0T lin
  syn_50 K3b 21.8 vs 16.6 s; refl syn_50 A2 KS 0.24. *Exit:* A1-A8/K11 + the syn_0 ladder green; withheld updates delivered; AVL_EVAL and
  empty-pool cleanup counted on mobiperf_3st; G1U n=300 INV/EXACT green.
- **FX-N70 `[C][S]` · Real refl ingest on 29 MB updates · wip: FX-D64 may explain it (PR18).** Run 16 speech G1A refl: queue_wait p99 0.74 s (was 4.3; 5 updates > 1 s),
  ingest 138 ms/update (13/round), K2 green; U6 red: real visibility lag 0.18 s vs sim 0. *Next:* profile it into a sim charge or floor it
  (G2C). *Exit:* U6 green or floor-SKIP.
- **FX-N5 · syn_0 GPU block (G2), speech feddance · todo: speech G2 feddance + G2S floor (PR17).** Speech oort/refl INV/EXACT green at D ×5 (run 13); cifar
  done. *Exit:* speech feddance INV/EXACT green or floor-SKIP.
- **FX-N73 `[C]` · Selector speed includes the transfer leg in real only · todo (operator decision).** Real oort/refl duration = dispatch→receipt
  (speech ~1.4 s of 29 MB transfer); sim speed = max(gpu, D), leg kept out by FX-D23. Speech oort: preferred-duration median 16.35 vs 15.0 s,
  2/47 fast trainers outside speed identity; selection bias/participation green. C6: real Oort's duration includes comm. *Exit:* decided; speed identity green.
- **FX-N10 · google_speech on the launcher · wip: lr fixed (FX-N74); stop rule next.** *Next:* target accuracy + stop rule (2024: 20 evals
  ≥ 60%); then S4 removes the 2024 JSON/scripts and the import script. *Exit:* all six real+sim graded on speech (T4 CPU + G1/G2 GPU).
- **FX-N13 · Streaming experiments (linear + events), both datasets · wip: ST5 cifar done; speech G0T felix eve green, lin EV10 fixed (FX-D50).**
  *Exit:* ST5 speech done; ST6 shows the oracle advantage.
- **FX-N71 · tiny_cpu cifar felix K4 4.1 vs 4.9 in flight · todo.** P6 and P7 (`ladder_20261001_164350` L4, ungated: L4 grades EV) and
  the TS smoke: speeds (2.72/2.73 s), barrier (2.7 s) and in-flight (5/5) match, but sim's per-commit clock is 0.66 vs 0.55 s. Not the
  charges (no tiny_cpu profile; stub's total 0.05 s); speech P6/P7 and every GPU felix leg green. *Next:* `logical_diff.py` on the P6 pair,
  then per-commit `agg_timing`. *Exit:* K4 green on cifar P6/P7.
- **FX-N42 `[S]` · Parity ladder · wip (PARITY_READINESS Active build: Q2-Q6).** *Exit:* Q2-Q6 done; a run graded per cell on both axes.
- **FX-N33 `[C][S]` · Aggregator aborts at interpreter exit · todo (root open).** After a clean channel leave:
  `terminate called without an active exception` → `Fatal Python error: Aborted` (a C++ thread destroyed while joinable; only the main
  thread, no Python frame). Runs 16-17: every cifar G1A aggregator, real and sim (exit -6, so the runner skips the trainers' EOT grace), and 2 T3 legs; a bare
  torch+CUDA+pyarrow+grpc import doesn't reproduce it.
  Data is intact (post-leave); FX-D22 allowlists exactly this signature. Loaded natives: grpc, pyarrow (+ s3fs), torch, cuda.bindings,
  zstandard. Next: diff what sim alone starts at exit. *Exit:* root named, abort gone.
- **FX-N30 `[S]` · Speech tiny_cpu is too heavy for CPU slots · todo.** P7/P7o sims run at sim_rate 0.2-0.5 and are
  killed at 63-141s of 180 (EV0/EV12, `KNOWN`). In P7o real, the oracle's `select` blocks the MQTT thread 20-84s on 2
  aggregator cores. Options: shrink the speech tiny_cpu model/data, more aggregator cores for oracle legs, or run
  P7/P7o speech on GPU/P8. *Exit:* speech P7/P7o EV green; FX-N13 speech arms usable.
- **FX-N22 · Fast parallel harness (Active build) · wip: P6 isolation control next.** *Exit:* P6
  EQUIVALENT at cpt ≤ 0.5, and T2 for both datasets under 25 min on one node.
- **FX-N7 · Remove legacy `trackTrainerAvail` (oort, oort_star, refl) · todo: unblocked.** refl EV green on every run-5
  leg. Delete the dead check and legacy branch (S4), then T12 retires. *Exit:* code gone, pytest green.
- **FX-N2 · Parent S2 (parity pipeline) for async_cifar10 · todo.** *Exit:* the stored Jun 23-24 pairs
  re-grade through it to within the floor, or each difference is explained.
- **FX-N8 · Concurrent-run confound on fedbuff · likely closed by FX-N22.** *Exit:* P6 EQUIVALENT.
- **FX-N11 · google_speech GPU parity · blocked: FX-N9, FX-N10.** Reuse the FX-N5/9 protocol.
- **FX-N12 · Felix paper experiments, sim-only · blocked: FX-N11, FX-N13, FX-N22 P8.** = ST6 of the streaming design.

---

## Baseline hyperparameters (note, operator 2026-10-01)

Source of truth: `datasets.yaml` `by_baseline` (each field tagged [paper]/[repo]/[ours]); shared defaults are listed in its
header. Rules: a baseline uses its paper's values per dataset; lr decay only where the baseline configures it (REFL); a late
update is accepted only under the baseline's own rule (REFL staleness ≤ 5; oort, oort_star, feddance none). Judgement calls:
- REFL speech lr: paper Table 1 0.005 used; its repo config says 0.05.
- REFL staleness: paper default is no bound (≤ 5 only in its §3.2 study); kept `stale_update: 5` (operator).
- Oort/FedDance/REFL defer other knobs to FedScale defaults, which include lr decay 0.98/10; decay kept off except REFL (operator rule).
- 2024 pairs (git 97cede899): cifar felix SGD 0.04 × server 0.3, fedbuff SGD 0.000195 (batch 32) × 40.9 — kept; speech felix Adam
  0.04 × 0.065, fedbuff Adam 0.000195 × 0.075 — replaced (operator 2026-10-07) by oort's speech client, SGD 0.04 b16, × server 0.5 (felix) / 1.0
  (fedbuff), as cifar felix shares oort's client (FX-T27, FX-N74).
- Every speech baseline now trains with `trainerOptimizer: sgd`; the dataset default (Adam) is unused on speech.
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
- **FX-L7** Sync oort over-selects ×1.3: an in-flight straggler stays in `selected_ends` and is received stale in the next round by sct-ordered pop, as real (FX-D39).
- **FX-L8** refl/oort: a trainer still computing (`vclock < sct`) is kept out of the eligible pool via the
  unavailable path, not `selected_ends`.
- **FX-L9** For a strict-barrier baseline (feddance, fedavg), real visibility lag is anchored on the barrier
  (`max_dur − dur_i`); streaming oort/refl stay per-message.
- **FX-L31** A committed end is RECVD so it leaves RECV; a buffered one reset to NONE then committed stays a
  phantom that blocks the starvation wake-up (P3 felix/fedbuff 6-min livelock).
- **FX-L32** A dispatch consumes the end's earlier receipt (eval reply, commit) and its RECVD state; else cleanup or
  recv frees the in-flight trainer and it is re-dispatched (FX-D16 22s tail; late withheld commits, FX-D16).
- **FX-L45** In sim, arrival is not receipt: an update real can't have received yet (withheld, evicted) must leave
  channel and selector state as real's unanswered dispatch (FX-D34).
- **FX-L54** A withheld delivery is an event in the sct order: reinject every delivery due before the next buffered sct, or the clock
  jumps past it (EV16, FX-D50).
- **FX-L60** Keep real-side measurement off the ingest path: a Python-heavy eval thread shares the GIL with ingest and stalls real
  commits sim never charges (FX-D64). Read queue_wait spikes against eval times first.
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
- **FX-L36** Set a test shape through the launcher's `experiment.aggregator.agg_goal`: it fans into
  `aggGoal` and sync `aggr_num` as the last merge layer, so K = aggGoal for sync baselines.
- **FX-L37** A dataset differs by `datasets.yaml` + `fl_data.py` only; baselines, selectors and the sim/real
  machinery stay shared (google_speech = n=100, ResNet34-1D, Adam, 29 MB updates).
- **FX-L57** Check a client/server lr pair in-process (`scripts/fl_lr_check.py`, minutes) before a GPU run; grade accuracy on full data
  (G-tier legs stream 0 → 100% over 3 h, so a 45-min leg trains on ≤ 25%) (FX-N74).
- **FX-L58** A decision taken at an instant (send gate, select filter) reads the trace at that instant: the 1 s availability poller's
  cached state let real updates finishing just past a flip go out ungated (FX-D55).
- **FX-L49** A forward pass with no backward runs under `no_grad`, else a kept loss tensor holds its graph (FX-D41); free cached CUDA blocks after every
  GPU step on an idle path, telemetry included (FX-D43).
- **FX-L50** Under `max(gpu, D)` a trainer whose host GPU time nears D takes its speed from GPU contention, which differs real vs sim; check overrun
  share per trainer before reading a speed-identity red (FX-D46).
- **FX-L52** Classify a sync stall by cause: a round holding an abandon, or taking a fresh send-gated update, is a stall at any
  length (N62: 40-60 s gated rounds read as stall-free, K3b 0.175 → stall-free 3.16 vs 3.15 s, FX-D48).
- **FX-L53** syn_* traces flip on shared slot boundaries (150 s at scale 4): seconds of clock offset move 90 s stalls between rounds. Grade n=50
  timing on G0UC floors (oort 90%, felix 2%).
- **FX-L51** Profile the aggregator per update with fresh buffers, as production allocates them: a 29 MB copy costs ~30 ms of page faults, the
  math 3 ms (FX-D40 inert, FX-D44). A reused-buffer microbenchmark hides it.
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
- **FX-L55** A real idle loop wakes at its earliest FUTURE deadline, read on the clock that stamped it, never a fixed poll: the 30 s poll
  quantized real's 90 s reclaim to 103 s (FX-D50); epoch vs avail-clock stamps made every wait 10 ms (FX-D60).
- **FX-L59** Watch a real leg's selection and telemetry rate, not just its rounds: a busy-loop commits normally while writing 30 GB (run 16
  felix real: 309k selections, 13 per dispatch vs 2 in G1L).
- **FX-L48** BN `running_mean`/`running_var` are non-additive state: set them to a convex mean of absolute stats (fresh updates, or global
  at the update's version + its delta), never base + rate × server lr × stale deltas (FX-D61); the reference REFL never aggregates buffers. Scan every checkpoint for a negative `running_var` when a loss goes NaN with finite weights (FX-D38).
- **FX-L41** Accumulate model state in float and cast back to each tensor's dtype once; apply baseline-specific
  server steps to parameters (float) only. Integer buffers exist only in some models (BatchNorm) (FX-D17).
- **FX-L42** A sync round that commits nothing re-dispatches at the same version: select afresh, excluding ends
  already tasked at it. A selector's round cache serves only the same round's RECV lookups (FX-N31).

**Reading the checker**
- **FX-L46** Decompose a sync clock residual first: 2-9 ninety-second timeout stalls dominate a syn_50 mean; compare
  stall-free advance and stall count separately (FX-N62). Count a stall served late as one episode (FX-D36).
- **FX-L47** Audit real's receive path per leg: arrivals vs processed, and `active_task_skips`. An abandoned end whose
  update arrived is a lost message; growing skips mean a leaked reader slot (FX-D35).
- **FX-L56** Align real and sim on trace time or matched work, never round fraction or per-event means: those follow round count and loop
  cadence, not the mechanism (A3, S3/4, FX-D51).
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
- **FX-T34** Don't mark the round a stale gated straggler lands in as a stall: an over-selecting sync round never waited for it (FX-D48).
- **FX-T33** Don't enable `simInflightCarryover`: it holds a straggler whose sct falls inside the next round, so it is received two rounds late (real: one) and sim carries 6.6 vs 3.8 (FX-D39).
  Exclude at eligibility (the removed `[SELECTION_CHECK] Skipping`, 46-143 per felix run).

**Availability**
- **FX-T35** Don't let a wall failsafe drop a sim update before its task timeout: speech sim tasks reached 50 s wall and were re-dispatched (EV10, FX-D50).
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
- **FX-T37** Don't count a stale update toward a REFL round's K: reference REFL closes on K fresh clients and adds stale ones on top;
  counting them let real refl close every round on stale backlog (0 fresh after round 750, acc 22 → 9%, FX-D53).
- **FX-T24** Don't add a staleness cutoff to fedbuff: its 1/√(1+s) discount is the baseline (operator).
- **FX-T26** Don't set `selector.kwargs.aggr_num` or `aggGoal` directly in an experiment: `agg_goal`
  overwrites them (a small-n oort smoke kept K=10 > n and starved).

**Harness and runs**
- **FX-T32** Don't clamp `running_var` at 0 as the fix for a negative one: variance pinned at 0 makes BN divide by ~√eps and the eval loss explodes (6678, acc 0.04); the clamp is a guard only (FX-D38).
- **FX-T31** Don't retune `simCompletionLegSeconds`, `simRedispatchGapSeconds` or `simCommitOverheadSeconds` to close a
  clock rung: they are single-config fits; profiled charges replaced them (FX-D23).
- **FX-T13** Don't grade a fedbuff/felix real run while another n=300 real run shares the broker (FX-N8).
- **FX-T16** Don't re-chase GPU contention (overrun 0), SEND_TIMEOUT or MQTT drops at n=300 cifar; all
  measured 0.
- **FX-T23** Don't run more stub trainers than node cores (1 core each): n=200 on 128 cores crawled the sim to
  its wall ceiling at vclock 6.
- **FX-T36** Don't pack more than ~75 cifar or ~25 speech trainers per A40 (idle ~0.4-0.47 GB each), or let more than 7 speech tasks
  (~3.1 GB each) train on one at once (FX-D58): cifar n=300 on 3 GPUs OOMed at init; speech felix on 3 GPUs at 45/45 GB (FX-D52/D56).
- **FX-T25** Don't kill FL workers by process name alone; scope by `FLAME_RUN_TAG` (`_expt_pids`,
  `slot_pids`). The runner's pre-leg `pkill -9` would have killed every neighbouring slot.
- **FX-T28** Don't derive a leg's health from the shared `experiments/` dir; parallel neighbours' logs leak
  in (FX-D13). Read the leg's own run dirs.
- **FX-T29** Don't write a replay input in place or from a daemon thread: a writer killed at exit truncates it (FX-N33).
- **FX-T38** Don't key a liveness watchdog on run-dir log growth past `run_end`: teardown and grading are silent; run 16 cut 5 finished G1A
  legs 10 min after their last line (FX-D62).
- **FX-T39** Don't stack two eval gates (round modulo, then a commit stride): evals landed every 2N rounds (FX-D63).

**Datasets**
- **FX-T27** Don't take speech lr pairs from the 2024 configs as-is (server lr 0.065/0.075 stays at chance), or run speech felix/fedbuff on
  Adam: under staleness it oscillates (felix 27% at r120 vs SGD 49%, `fl_lr_check.py`, FX-N74).
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
- **FX-D32** (code cites FX-N15/N55/N56/N57/N58) Run-6 confirmed: unaware fedbuff ignores the oracle stamp and picks
  UN_AVL at the trace level (0.51/0.53 at selection, EV17 green); asyncfl holds a withheld pick's slot to dispatch+90s;
  syncfl discards stale updates (`sync_accept_stale`); each baseline's `[TRAINER_HP]` equals its `datasets.yaml`
  `by_baseline` row on every leg; fedbuff EV14 green on all CPU+GPU legs (`ladder_20261001_051724`, `_051427`).
- **FX-D38** (code cites FX-N64) REFL BN running stats average over fresh updates only (`bn_fresh_only`; REFL itself aggregates
  parameters only) + a `clamp_running_var` guard (refl, fedbuff): stale deltas drove speech `running_var` negative → NaN at round ~600. EV18 +
  `model_health` telemetry; 2400s refl speech sim: 0 clamps, acc 0.28 (`block0_n64_20261002_1809`).
- **FX-D37** (code cites Q2, FX-N68) `parity_check.py --control` grades two legs of one mode; `--floors` feeds `control_floors` into
  `floor_gated_tol` (never tightens; n=2 floors are lower bounds, T8), A2c bias included. Feddance lock-in: locked sets overlap 4-8/12 within
  either mode, cifar G2 SKIPs on its sim↔sim floor (11.7%; real↔real 1.6%). K8/U2 drop timeout-stall excess; EV16 exempts a delivery at the budget.
- **FX-D36** (code cites FX-N62) Checker: a timeout stall is an episode (one round ≥ 72s, or consecutive ≥ 3× median rounds summing past
  it); K4 drops stall time from its clock span.
- **FX-D35** (code cites FX-N63) Real `recv_fifo` releases every reader slot when its streamer exits (a reader cancelled
  before aiostream's merge started it leaked its end forever) and enqueues on dequeue: speech refl syn_0/syn_20 real 223
  rounds (PR8 42), 0 `active_task_skips`, arrivals = processed + too-stale, K2/K3b/K4/A2 green (`pool_fxn63_verify2`).
- **FX-D39** (code cites FX-N67) `simInflightCarryover` off for oort/oort_star: sim stale receipts land one round after dispatch, as real
  (G2 oort carried 6.6 vs 3.8 → 3.71 vs 3.84). Tier `T3S` (aggGoal 10) reproduces it on CPU in 7 min. K3b adds `mix_adjusted_residual_s`; U5
  ranks only trainers on both sides (FX-N66); the P3/T2 support guard counts the sim tail past real's p99.
- **FX-D40** Aggregator math libs get a quarter of its pinned cores (`FLAME_AGG_MATH_THREADS` overrides). Inert in production: speech fedbuff real ingest
  0.143 s/update in run 10b and run 11 alike (the cost is pickling, FX-N70). `trainer_speed_identity` grades one sample per commit (`agg_observed_s`).
- **FX-D42** (code cites FX-D42) asyncfl caches the dispatch payload per (version, task, stamp), cleared at commit: async re-pickled the 29 MB speech
  model on every dispatch (85 ms, ~0.9 s per 10-update version). `tests/mode/test_dispatch_payload_cache.py`.
- **FX-D44** Spawned aggregators (trainers: FX-D59) keep freed buffers in the glibc heap (`MALLOC_MMAP_THRESHOLD_` 32 MB, trim 4 GB, top pad 256 MB;
  `FLAME_MALLOC_TUNE=0` reverts): speech ingest chain 114 → 30 ms, dispatch pickle 35 → 16 ms (bench); run 12 G1S real queue_wait p99 0.17/0.30 s
  (fedbuff/felix). `tests/launch/test_malloc_env.py`.
- **FX-D45** Pool stall rules (C12): `StallWatch` kills a leg alone after S1 no committed round for `--stall-min` (15) min or S2 no log growth for
  10 min → `STALLED.txt`; slow legs are never cut. `tests/harness/test_stall_watch.py`.
- **FX-D46** Speech device time ×5 (operator): `datasets.yaml` `device_time_scale: 5` → `training_delay_factor` 0.2 and `send_timeout_wait_s`
  450 on GPU legs; a CPU harness `--delay-factor` wins. The availability abandon, sync recv deadline and the checker's stall cut (0.8 × timeout)
  read that one knob (`_task_timeout_s`, `_stall_cut`). Solo full-data task: speech 2.3 s vs cifar 8.5 ms. Run 12-13: overrun 0.5% of tasks,
  speed identity green on felix/fedbuff/refl, refl K3b 1.1% (was 3.74 vs 2.46 s/round). `test_task_timeout.py`, `test_debug_run_dataset_profile.py`.
- **FX-D49** GPU slots below the 0.4 CPU/trainer default (operator 2026-10-05: cifar at 0.2) are sized per baseline: each pair gets max(formula,
  1.5 × its own measured cores p95, history `cores|<key>`), capped at 0.4, real and sim equal; a leg throttled (≥ 90% busy) in > 5% of samples
  is flagged `CPU_SAT` in `pool.log`, `jobs.tsv` and SUMMARY (timing suspect). `test_harness_pool.py`.
- **FX-D47** (code cites FX-N13) Streaming ST1-ST4: `stream_schedule.py` (linear `initial_frac`, events `n_chunks`/`seed`, legacy stagger;
  `clock: trace` = stream clock × trace scale) shared by trainer, oracle and replay; EV19 (visible = schedule, monotone, stream clock = run
  clock); `stream_growth` DIAG (visible and fresh share per train task, real vs sim); `debug_run.sh` mirrors the trainers' `data_streaming`
  into the aggregator config. `test_stream_schedule.py`, EV19 in `test_event_invariants.py`.
- **FX-D48** (code cites FX-N62) Checker stalls by cause: `_mark_stall_causes` stamps sync rounds holding an abandon or a fresh gated update
  (sim `withheld_delivery` at staleness 0, real `task_send` gated > 0.5 s, among the round's contributors); K4/K8 drop only the excess over a
  stall-free round; K2/K3/K3b SKIP below 10 stall-free rounds when stalls removed any. N62 69/69 (was K3b 0.175); regrade of 169 stored
  pairs: 165 same, 2 newly red (speech GS oort syn_20 K2 9% stall-free, pre-D46; an A3 the old K3b masked). `test_stall_causes.py`.
- **FX-D50** Async sim pops a withheld delivery due before the next buffered sct first (`sim_reinject_lookahead`; run 12-14 EV16 commit_late
  10-41 per syn_50 leg); the sim gate holds until the task timeout (not 64 passes ≈ 32 s; cold-start cap unset = task timeout, was 10 s:
  full-data speech round 1 past-dated 7/10) and logs `[SIM_GATE_FAILSAFE]`; real asyncfl's idle wait ends at the
  selector's earliest reclaim (`real_wake_at_timeout`; fedbuff mobiperf reclaim 103 → 90 s). `test_live_wiring.py`, `test_asyncfl_real_drain_ready.py`.
- **FX-D51** Checker: EV16 lets a sync delivery commit at its round's close; A3 bins eligible counts by trace time; S3/4 grades async selectors'
  total picks; K8 allows ±1 trainer; the trainer wall budget is DIAG (off the vclock); EV19 takes the real origin from the `trace_origin`
  event (log fallback). Run 12-14 regrade cleared 4 sync EV16, 5/6 A3, 5/6 S3/4, EV19 and a ±1 K8. `test_parity_checks.py`, `test_ladder.py`.
- **FX-D52** Ladder `--regrade` re-runs the event checker and grades reference-config cells (no agg-goal); G0UC replicates floor their G0U and
  matching G0T/G0To cells across a block's pools; G1U cifar takes the whole node; `G0U_RUNTIME_X` lengthens G0U oort mobiperf ×4 and speech
  fedbuff mobiperf ×3; feddance G0T syn_0 clock family is KNOWN (FX-D37).
- **FX-D53** REFL: only fresh updates fill a round's K (`_counts_toward_k`, `refl_fresh_k`); accepted stale updates still aggregate, as
  `third_party/REFL` (`total_updates = tasks_round + round_stale_updates`). `_inflight_commit_fresh` now records fresh. `test_refl_fresh_k.py`.
- **FX-D54** Accuracy tooling: `scripts/accuracy_table.py` (accuracy at 15-120 min and time to target per leg, real wall / sim vclock),
  `scripts/fl_lr_check.py` (in-process FL lr pairs on the real model and split), tier `G1A` (reference n, syn_0, full data, eval every 20 rounds).
- **FX-D55** Real send gate refreshes availability from the trace at send time and polls every 0.1 s while gated (`FLAME_SEND_GATE_REFRESH=0`
  reverts); trace transitions pop under one lock (poller + gate). Run 15 cifar refl syn_50: real sent at 150.65 s past a 150.0 flip, so sim
  withheld what real sent (S1/S8 red from round 72). `tests/mode/test_send_gate_wait.py`.
- **FX-D56** Density guard: speech G1A/G1S/G1U legs take 4 GPUs (c=30 on 3 peaked 45/45 GB); `harness_pool.py --scale-smoke S` runs every
  leg S seconds at its production n/c/GPUs; each GPU leg's DONE line prints `gpu peak U/T GB`, `GPU_TIGHT` above 85%. Block phase 0 = PL3.
- **FX-D57** `flame/monitor/runtime.py` GC pause callback binds its clock at def: a GC at interpreter exit ran it after `time` was cleared, and
  the `Exception ignored` traceback fail-fasted a healthy pool (run 16 phase-0 smoke, speech felix real). `tests/test_gc_pause_callback.py`.
- **FX-D58** Per-GPU training cap: `datasets.yaml` `gpu_train_slots` (speech 7) → trainers hold one of N `flock` slots of their GPU from
  compute start through the cache release; the wait counts as GPU time, as contention does. Speech felix's seeded c=30 first wave put 9 of 30
  on one GPU (39.4/45 GB, both modes; 11 OOMs). GPU legs only. `test_gpu_train_slots.py`.
- **FX-D59** Host RAM: trainers keep glibc defaults (`FLAME_MALLOC_TUNE_TRAINERS=1` opts back into FX-D44's knobs, which stay on aggregators):
  the 4 GB trim threshold kept each trainer's load-time peak (+290 MB at data load alone; 1.17 GB PSS per cifar trainer), so cifar n=300 +
  speech n=100 reached RAM + swap 100% and the kernel killed run 15. DONE lines print `ram peak`, `RAM_TIGHT` above 85%. `test_malloc_env.py`.
- **FX-D60** Real asyncfl's idle wait reads selector stamps on their own clock (avail clock once a select saw `vclock_now`) and wakes only
  at a future reclaim: FX-D50 compared avail-clock stamps to epoch, so every drain waited 10 ms (run 16: 30 Hz loop, 30 GB telemetry per
  cifar felix real leg). `test_asyncfl_real_drain_ready.py`.
- **FX-D61** FedBuff/Felix BN running stats = the mean of each update's absolute stats (global BN at its version, from a 64-version ring,
  + its delta), outside rate and server lr (`bn_absolute_mean`, default on): speech felix clamped a negative `running_var` on 296/300 commits
  (test loss 1e17, chance accuracy). In-process: off → loss 3e9. `tests/optimizer/test_bn_clamp.py`; `fl_lr_check.py --staleness --flame-opt`.
- **FX-D62** Post-run: `StallWatch` disarms at the aggregator's `run_end`; pool legs skip the runner's plot analysis (`FLAME_POST_ANALYSIS=0`,
  else capped at 300 s); `accuracy_table.py` streams telemetry. Run 16 lost 5 finished G1A legs to the S2 cut in that analysis. `test_stall_watch.py`.
- **FX-D63** Example aggregators evaluate on every `evalEveryNRounds` round (commit stride 1 under the round gate; it was 2 → every 2N).
  `trainer/pytorch/test_eval_cadence.py`.
- **FX-D64** Aggregators decode the test set into tensors once at load (`fl_data.in_memory`; `FLAME_EVAL_PRELOAD=0` reverts) and don't pin it:
  the eval thread's per-wav decode held the GIL ~7.5 s per speech eval, so real ingest waited up to 19.5 s at each eval (run 17 G1A felix,
  148 waits > 1 s, 96 within 90 s of an eval); now 4.7 s once at load, 0.08 s per pass. `tests/harness/test_fl_data.py`.
- **FX-D43** Trainer `_release_gpu_cache` also runs after the util_cf telemetry forward (256 samples left 2.4 GB reserved per idle trainer; felix's
  skewed picks left ~10 hoarders per GPU → speech felix real OOM at 35 min, run 11; run 12 G1S ran 45 min clean). `test_release_gpu_cache.py`.
- **FX-D41** Trainer `evaluate()` resets the utility and runs under `no_grad`: it summed onto the last task's utility (eval utilities read ~5% high) and
  its graph chained across evals until the next train (speech felix real CUDA OOM at 9 min, `block_20261003_1311/gs_g1`). `test_eval_utility.py`.
- **FX-D34** (code cites FX-N59/N60) Sim leaves an unanswered dispatch's channel/selector state as real does: an unaware
  withheld pick holds its slot (`sim_hold_withheld_slot`; mobiperf fedbuff real 3 vs sim 2 rounds, was 3 vs 122), and a
  felix-evicted end's update is ingested (`sim_ingest_evicted`; felix syn_50 A2 6.9/6.9, K2/K4 green, both datasets;
  `pool_fxn59_verify`).
- **FX-D33** (code cites FX-N41/N61/N62) Checker: A7 commit and selection beliefs are graded at their own instant (±1s);
  K2/K3/K3b grade stall-free rounds (< 72s) and K3s grades the ≥72s timeout-stall rate (Poisson 2σ + 1). Run-6 regrade:
  sync syn_50 stage 1 green on feddance/refl/oort_star, both datasets.
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
  Real asyncfl ingests via streamer-free `drain_ready` by default (operator 2026-10-01; `real_drain_ready_ingest=false`
  = recv_fifo, kept as campaign control P9); P9-vs-P1 run 4: RECV_FIFO skips 900 → 0, p99 no worse on cifar.

**Harness, checkers, datasets**
- **FX-D19** (code cites FX-N36) flame processes poll NVML only with `FLAME_STAT_THREADS=1`: run 2 GPU legs joined 44-99s
  after their first trainer, max `[WARMUP]` 1.35s (was 10-min joins).
- **FX-D22** (code cites FX-N40) Fail fast: `examples/scripts/fail_fast.py` flags EV0's fatal lines (Traceback, Fatal
  Python error, Segmentation fault; the FX-N33 abort and a teardown SIGTERM's `SystemExit: 0` traceback allowlisted by signature, EV0 alike). `harness_pool.py` scans live legs every
  ~30s and on completion → teardown, `ABORT.txt`, partial SUMMARY, rc 3; standalone `harness_suite.sh` stops after the
  pair; `--no-fail-fast` opts out. `--inject-bug trainer_crash` stopped a smoke pool 61s in (`pool_smoke_fxn40_abort2`);
  on run 2 it flags exactly the 9 speech OOM legs. Tests: `tests/harness/test_fail_fast.py`.
- **FX-D7** Trainer availability thread stops at EOT or shutdown (SIGTERM/atexit); clean teardown.
- **FX-D11** Streaming + oracle harness (P7/P7o) with offline replay and figures.
- **FX-D13** Event checker EV0-EV19 (EV18 = global model finite, BN `running_var` >= 0; EV19 = stream schedule) + injected bugs (P11a-c); parallel isolated pool with tiers T1-T4/G1/G2,
  real bank, `--changed`, `--shard`, gate (FX-N22). Run dirs are `run_<ts>_<phase>_<name>` (`FLAME_RUN_LABEL`) and never reused; health reads
  `FLAME_RUN_DIR_FILE`; the leg watchdog budgets the sim wall ceiling; sub-0.2s phases grade on mean (50 ms).
- **FX-D15** No cold start in timed tasks: startup warm-up (CPU + GPU; frees its activation cache), CUDA-only sync in the weights phase,
  sim wall ceiling from the join barrier (first-task compute 9-14s → 0.1s; fedbuff EV11 fixed).
- **FX-D18** Traces sit at their named level from t=0 (`gen_synthetic_trace.py`, `shift_trace_origin.py`): `syn_10` (the old syn_20, 11%),
  `syn_20` (stationary, seed 20), `syn_50`, mobiperf (5-min injected head), all extended to 149 h and parsed once per process (C loader, 9 s);
  `effective_unavailability()`; a real trainer's trace clock holds at 0 until the aggregator's origin arrives. Tests:
  `tests/availability/test_synthetic_trace_fractions.py`, `test_harness_pool.py`.
- **FX-D14** Dataset switch (`fl_data.py`, `datasets.yaml`, `data_roots` = /coc/scratch/dgarg/fl_datasets); google_speech on the
  launcher (FX-N10).

## Open questions (operator)

