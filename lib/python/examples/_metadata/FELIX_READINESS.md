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

**At a glance (2026-10-03, run 9)** — green / known / red, detail in [PARITY_READINESS.md](PARITY_READINESS.md) → Felix scoreboard:

| test | both datasets (run 7-9) | regraded with FX-D36/D37 + FX-N62 KNOWN row |
|---|---|---|
| pytest | 2333 passed · 0 failed · 7 skipped (2026-10-02, with FX-D36) | — |
| L1 sim EV | 22 / 2 / 0 | — |
| L2/L3 pairs | 41 / 1 / 6 | 43 / 3 / 2 (feddance mobiperf K2 8.6-9.5%, no floor) |
| L4 campaign | 48 / 2 / 0, P11a-c caught | — |
| L5 GPU pairs | 22 / 0 / 2 (EV green on all 24; cifar felix/oort syn_0 K2 8.1/8.2% vs 8%, opposite signs) | — |
| L6 G0C + G0 (`ladder_20261002_033358`) | 31 / 0 / 5 | 35 / 0 / 1 (speech oort syn_20 trainers_at_n 49 vs 46; speech refl NaN fixed, FX-D38) |
| L7 cifar G1 (`ladder_20261002_203715`) | 2 / 0 / 0: felix + fedbuff syn_0 n=300 5400s, EV0-18 green, K2 0.1/0.3%, K3b 0.001/0.003, K8 0.2/0.3%, queue_wait p99 0.13/0.05s; C1/C2 LOWC (FX-N65) | — |
| L7 cifar G2 (`pr12_cifar_20261003_0322`, oort_star not run) | 1 / 0 / 2: refl green (residence hist = real to 2 dp); oort S3/4+Sr (gate, FX-D39, fixed, CPU-confirmed); feddance K3b/A2c (lock-in draw, FX-N68) | — |

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
| DIST tolerances sized by a replicate floor | 🟡 real↔real floors (n=2, G0C) mark ungradeable cells; nothing tightened yet | FX-D37, PARITY Q2 |
| streaming / oracle experiment (FX-N13) | 🟡 cifar arms agree real/sim; speech arms starved | FX-N30, FX-N13 |
| sim speedup vs real | 🟡 cifar n=300 syn_0: sim wall 4.5× (felix) / 7.4× (fedbuff) under vclock; leg wall 2.7-4.3× incl. startup | S6, FX-N22 P8 |
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
| G0 / G0C | GPU 30 min screen: all six × syn_0 + syn_20, n 100/50 / a second real leg per syn_0 cell (R7) | T4+G0+G0C ~6.9h per 8-GPU node |
| T3S | oort/oort_star/refl pairs with aggGoal 10 (stragglers cross rounds), syn_0 + syn_20, FX-D39 | 24 min (6 pairs) |
| G2S / G2C | one G2 cell replicated sim-only (5400s, ~8 min) / real-only (97 min): floor for a chaotic selector (FX-N68) | 8 / 97 min |
| N64 / N64S | FX-D38 refl speech health: 25 min real+sim pair + a real replicate / 40 min sim-only (EV18) | 45 / 27 min |
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

## Parked design — streaming time-to-accuracy experiments (FX-N12, FX-N13)

Starts when L1-L5 have no open shared root (operator: don't add noise to the parity path before then).

**Experiment (operator).** All six baselines × {cifar10, google_speech} × {syn_0, syn_20, syn_50, mobiperf_3st} × two
stream modes; metric = time-to-accuracy; each trainer's data reaches 100% at 6 days of trace time (the trace horizon);
sim-only once streaming parity holds. Claim: Felix's eval selector finds clients whose data (hence utility) just grew;
unaware or mis-estimating selectors don't.
- **linear:** every trainer starts at 10% of its data and grows linearly to 100% at 6 days.
- **events:** starts at 10%; each trainer uploads preset chunks at its own, out-of-sync times (flat stretches, then jumps),
  totalling 100% at 6 days.

**Exists.** Trainer streaming in the shared trainer (`async_cifar10/trainer/pytorch/main.py` `data_streaming`): a seeded
prefix of the local pool grows linearly 0→100% over `full_data_available_after_s`, optional per-client onset/span
(`stagger`); clock = vclock (sim) / wall since `AGG_START_TS` (real, FX-L30). P7/P7o harness, oracle selector,
`oracle_misselection.py`, `felix_streaming_figures.py`; cifar arms agree in sign real/sim; speech arms starved (FX-N30).

**Steps** (each with pytest + a harness smoke; EV/parity before any sweep):
- ST1 linear: `initial_frac` (0.1): visible = initial + (1−initial)·t/T.
- ST2 events: per-trainer chunk schedule, deterministic from (trainer_id, seed), identical in real and sim; `data_growth`
  telemetry per chunk.
- ST3 clock: the stream clock is trace time (same scale as the availability trace), so 6 days lines up with its horizon.
- ST4 checker: EV — visible count monotone and equal to the schedule at every task, both modes; parity — per-task visible
  count, utility trajectories, oracle-vs-baseline misselection.
- ST5 parity: P7/P7o × {linear, events} × both datasets, CPU then GPU short legs; FX-N30 (speech P7) first.
- ST6 sweep: sim-only matrix (needs FX-N22 P8 for scale); figures: time-to-accuracy, misselection gap.

**Open (operator, at ST2/ST6).** Event mode: chunk size and gap distribution; uploads only while AVL_TRAIN (tie to the
trace)? New data from the same distribution (today: a random-permutation prefix) or drifting labels? Target accuracy per
dataset (speech: 2024 used 60%).

## Next steps (persistent queue — top item is next)

**Counter (2026-10-03, after run 10 G2 cifar):** 19 open — 3 wip · 2 next · 8 todo · 4 blocked · 2 other (FX-N13 parked design; FX-N8 likely closed).
This session: 2 closed (FX-N67 oort carry-over gate → FX-D39; FX-N66 U5 now ranks common trainers), 1 opened (FX-N68); FX-N5 narrowed (oort_star not yet run).

Work rule: PARITY C10 — each session resolves as many independent items below as it can, not just files new ones.

**Test levels.** pytest (P0/T0) = in-process unit/integration tests, no FL processes, ~4 min. CPU tests =
pool tiers T1-T4: real aggregator + trainer processes over MQTT on CPU (stub/tiny_cpu data, n 12-30), graded
by the event checker and the parity battery. GPU tests = G0 (30 min screen at n 100/50, + G0C real replicates) and
G1/G2 (the production path at the reference n, cifar 300 / speech 100, 90 min). "CPU+GPU" = one pool run with both.

**Unblock map.** run 9 (L1-L6 + cifar G1) done → L7 = FX-N4 (speech G1) + FX-N5 (G2), + FX-N7 → FX-N6 + FX-N9 (GPU unavailability) → FX-N11
(speech GPU parity) → FX-N12. FX-N13 design can start any time.

- **FX-N62 `[C]` · Unaware short legs can't grade timing · blocked: long-run phase (PARITY C0.6).** Unaware oort waits
  out a 90s timeout on most syn_50 rounds: 11-18 rounds per 1200s leg, ~6 stall-free; stall counts match (5/6, 6/7).
  fedbuff mobiperf (240s): 3 real vs 2 sim rounds, all stalls (K3b/S3/4 red on < 20 commits, `pool_fxn59_verify`). With
  2-3 stalls a leg, one alive pick splitting a stall moves K3b's stall-free mean 16% (speech fedbuff syn_50). KNOWN in the
  ladder for oort syn_50 and fedbuff mobiperf. K8/U2 grade stall-free (FX-D37); speech oort syn_20 trainers_at_n 49 vs 46 (5% tol) needs a
  ≥3-leg floor. Operator: run ~3h oort syn_50 + fedbuff mobiperf legs once short-run roots are exhausted, beside the
  other long legs. *Exit:* graded unaware unavail timing cells.
- **FX-N42 `[S]` · Parity ladder · wip (PARITY_READINESS Active build: Q2-Q6).** *Exit:* Q2-Q6 done; a run graded per cell on both axes.
- **FX-N33 `[C][S]` · Aggregator aborts at interpreter exit · todo (root open).** After a clean channel leave:
  `terminate called without an active exception` → `Fatal Python error: Aborted` (a C++ thread destroyed while
  joinable; only the main thread, no Python frame). Run 6: 9 of 244 run dirs, all sim aggregators, CPU and GPU (09-27/28:
  15 of 16 sim; run 9 G1: fedbuff sim, 1 of 2 sim legs). Data is intact (post-leave); FX-D22 allowlists exactly this signature. Loaded natives: grpc, pyarrow
  (+ s3fs), torch, cuda.bindings, zstandard. Next: diff what sim alone starts at exit. *Exit:* root named, abort gone.
- **FX-N30 `[S]` · Speech tiny_cpu is too heavy for CPU slots · todo.** P7/P7o sims run at sim_rate 0.2-0.5 and are
  killed at 63-141s of 180 (EV0/EV12, `KNOWN`). In P7o real, the oracle's `select` blocks the MQTT thread 20-84s on 2
  aggregator cores. Options: shrink the speech tiny_cpu model/data, more aggregator cores for oracle legs, or run
  P7/P7o speech on GPU/P8. *Exit:* speech P7/P7o EV green; FX-N13 speech arms usable.
- **FX-N22 · Fast parallel harness (Active build) · wip: P6 isolation control next.** *Exit:* P6
  EQUIVALENT at cpt ≤ 0.5, and T2 for both datasets under 25 min on one node.
- **FX-N4 · GPU block: felix + fedbuff, syn_0, 90 min · next: speech (PARITY PR13); cifar done.** Cifar G1 green (`ladder_20261002_203715`: EV green, queue_wait p99
  0.13/0.05s, `trainer_speed_identity` P3 green, K2/K3b/K8 ≤ 0.3%). Speech: `harness_pool.py --tier G1 --datasets google_speech` (~2.5h per pair). jayne runs on 7 GPUs
  (GPU 1 ECC); each pair stays on one layout (L18). *Exit:* speech INV/EXACT green, DIST residuals common-mode; convergence needs FX-N65.
- **FX-N65 `[C]` · Convergence can't be graded at 90 min · todo.** C1/C2 read LOWC below 7200s (`SHORT_RUN_CONFIDENCE_S`); G1 legs are 5400s (acc diff 1.5/1.1%, 15/13 evals) and no
  real↔real replicate sizes the band. *Next:* felix cifar syn_0 ≥7500s, real ×2 + sim (PARITY PR14). *Exit:* C1/C2 not LOWC and acc diff inside the real↔real spread, or the
  G1/G2 length raised to 2h.
- **FX-N10 · google_speech on the launcher · wip: graded runs next.** Reference = 2024 SoCC n=100. Run 6 speech:
  L1/L4 green (P7o = FX-N30), L2 reds = the shared unavail roots above, L5 EV green. *Next:* GPU lr check in the first speech G1 (0.000195 vs 0.001 via `--trainer-hp learningRate=…`); target
  accuracy + stop rule (2024: 20 evals ≥ 60%); then S4 removes the 2024 JSON/scripts and the import script. *Exit:*
  all six real+sim graded on speech (T4 CPU + G1/G2 GPU).
- **FX-N5 · syn_0 GPU block (G2): oort, oort_star, refl, feddance · next (PARITY PR12b).** Run 10 cifar: refl 0 fails; oort fixed by FX-D39 (verify at n=300); feddance FX-N68. oort_star runs only with unavailability.
  *Exit:* oort G2 EV green, K2/K3b/K8 inside tolerance, Sr/S3/4 green, queue_wait p99 < 1s. refl: confirm at 3h.
- **FX-N68 `[C]` · feddance cifar G2 selection is a lock-in draw; no 5400s floor · todo.** Utilities tie to ±0.005 (loss ≈ chance), so top-N locks onto ~12 trainers from round ~60-100. Real and sim agree to round 60 (slowest pick 27.8/28.2 s, 26.2/25.8 s) then lock different sets (28 vs 21 s): K2 28.0 vs 24.0 s/round (14%), A2c bias 0 vs -1.9 s, S2 per-trainer max_KS 0.88; K3b's mix-adjusted residual is 0.04 s (clock charges match). *Predict:* sim↔sim and real↔real replicates at 5400s spread ≥ 14% (the 30-min G0 floor ≤ 5% predates lock-in). *Next:* `G2S` ×3 (sim, ~25 min) + `G2C` (real, 97 min); `parity_check --control`, `--floors`. *Exit:* a 5400s floor sizes K2/K3b/K8/A2c for feddance (SKIP or green).
- **FX-N13 · Streaming experiments (linear + events), both datasets · parked: design in "Parked design" above.**
  *Exit:* ST1-ST5 done; streaming EV + parity green on both datasets.
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
- **FX-N12 · Felix paper experiments, sim-only · blocked: FX-N11, FX-N13, FX-N22 P8.** = ST6 of the streaming design.

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
- **FX-L48** BN `running_mean`/`running_var` are non-additive state: average them over FRESH updates only (convex); the reference REFL never
  aggregates buffers (`model.parameters()` only). Scan every checkpoint for a negative `running_var` when a loss goes NaN with finite weights (FX-D38).
- **FX-L41** Accumulate model state in float and cast back to each tensor's dtype once; apply baseline-specific
  server steps to parameters (float) only. Integer buffers exist only in some models (BatchNorm) (FX-D17).
- **FX-L42** A sync round that commits nothing re-dispatches at the same version: select afresh, excluding ends
  already tasked at it. A selector's round cache serves only the same round's RECV lookups (FX-N31).

**Reading the checker**
- **FX-L46** Decompose a sync clock residual first: 2-9 ninety-second timeout stalls dominate a syn_50 mean; compare
  stall-free advance and stall count separately (FX-N62). Count a stall served late as one episode (FX-D36).
- **FX-L47** Audit real's receive path per leg: arrivals vs processed, and `active_task_skips`. An abandoned end whose
  update arrived is a lost message; growing skips mean a leaked reader slot (FX-D35).
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
- **FX-T33** Don't enable `simInflightCarryover`: it holds a straggler whose sct falls inside the next round, so it is received two rounds late (real: one) and sim carries 6.6 vs 3.8 (FX-D39).
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
- **FX-T32** Don't clamp `running_var` at 0 as the fix for a negative one: variance pinned at 0 makes BN divide by ~√eps and the eval loss explodes (6678, acc 0.04); the clamp is a guard only (FX-D38).
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
- **FX-D32** (code cites FX-N15/N55/N56/N57/N58) Run-6 confirmed: unaware fedbuff ignores the oracle stamp and picks
  UN_AVL at the trace level (0.51/0.53 at selection, EV17 green); asyncfl holds a withheld pick's slot to dispatch+90s;
  syncfl discards stale updates (`sync_accept_stale`); each baseline's `[TRAINER_HP]` equals its `datasets.yaml`
  `by_baseline` row on every leg; fedbuff EV14 green on all CPU+GPU legs (`ladder_20261001_051724`, `_051427`).
- **FX-D38** (code cites FX-N64) refl speech NaN: its staleness-weighted delta average drove BatchNorm `running_var` negative (round 550 sim,
  600-650 real; REFL itself aggregates `model.parameters()` only, never buffers). BN running stats now average over fresh updates only
  (`bn_fresh_only`, convex, `optimizer/refl.py`), plus a `clamp_running_var` guard (refl, fedbuff; both default on). `model_health` telemetry +
  EV18 (finite weights, `running_var` >= 0). Block 0: 2400s refl speech sim, 0 clamps, var_min 4e-5, acc 0.28 at round 800 (clamp alone: loss 6678,
  acc 0.04); real+sim 1500s legs EV18 green (`block0_n64_20261002_1809`, `block0c_n64s_20261002_1934`).
- **FX-D37** (code cites Q2) Q2 floors: `parity_check.py --control` grades two real legs, `--floors` feeds `control_floors` into
  `floor_gated_tol` (never tightens: 2-leg floors are lower bounds, T8); `parity_ladder` regrades L6 against its G0C legs
  (speech feddance syn_20 → SKIP, floor 17%). K8/U2 time-to-N drops timeout-stall excess (`_stall_s_to_n`); EV16 exempts a
  delivery at the budget. L6 regrade 33 → 34 green of 36 (`ladder_20261002_033358`).
- **FX-D36** (code cites FX-N62) Checker: a timeout stall is an episode (one round ≥ 72s, or consecutive rounds ≥ 3× median
  summing past it), so a late alive pick can't leak a 70s piece into the stall-free mean; K4 drops stall time from its
  clock span. Regrade of the 48 L2 pairs: refl cifar syn_50 K4 0.54/0.86 → 0.98/0.98x, speech fedbuff syn_50 K2 0.105 →
  0.02 (stalls 3/3), no other cell moved.
- **FX-D35** (code cites FX-N63) Real `recv_fifo` releases every reader slot when its streamer exits (a reader cancelled
  before aiostream's merge started it leaked its end forever) and enqueues on dequeue: speech refl syn_0/syn_20 real 223
  rounds (PR8 42), 0 `active_task_skips`, arrivals = processed + too-stale, K2/K3b/K4/A2 green (`pool_fxn63_verify2`).
- **FX-D39** (code cites FX-N67) `simInflightCarryover` off for oort/oort_star (`..._parity.yaml`): sim stale receipts land one round after dispatch, as real (G2 oort carried 6.6 vs 3.8; refl, gate off, matched real to 2 dp). Harness shape `syn_0s`/`syn_20s` + tier `T3S` (aggGoal 10, n 30-40, oort/oort_star/refl) reproduces it on CPU in 7 min: gate on carried 5.95 vs 3.39 (rel 0.43, as GPU 0.43), off 3.33 vs 3.30, EV + all checks green. K3b adds `mix_adjusted_residual_s`; U5 ranks only trainers on both sides (disjoint stochastic picks no longer read ρ < 0, FX-N66); P3/T2 support guard counts the sim tail past real's p99 (binomial edge) instead of a bare p99 ratio.
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
  Python error, Segmentation fault; the FX-N33 abort allowlisted by signature). `harness_pool.py` scans live legs every
  ~30s and on completion → teardown, `ABORT.txt`, partial SUMMARY, rc 3; standalone `harness_suite.sh` stops after the
  pair; `--no-fail-fast` opts out. `--inject-bug trainer_crash` stopped a smoke pool 61s in (`pool_smoke_fxn40_abort2`);
  on run 2 it flags exactly the 9 speech OOM legs. Tests: `tests/harness/test_fail_fast.py`.
- **FX-D7** Trainer availability thread stops at EOT or shutdown (SIGTERM/atexit); clean teardown.
- **FX-D11** Streaming + oracle harness (P7/P7o) with offline replay and figures.
- **FX-D13** Event checker EV0-EV18 (EV18 = global model finite, BN `running_var` >= 0, from `model_health`) + injected bugs (P11a-c); parallel isolated pool with tiers T1-T4/G1/G2,
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

- None open; the streaming questions sit in its parked design.
