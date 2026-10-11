# Felix readiness — backprop FL (async_cifar10, google_speech)

## Preamble
- **Read [ROBUST_FL_READINESS.md](ROBUST_FL_READINESS.md) first**; its rules, L and T apply here unrepeated.
- **CURRENT FOCUS:** Felix feature-complete (syn_0 + unavailability, both datasets), paper runs sim-only.
- **Correct, not just equal:** every baseline follows its reference and hits target (cifar 50%, speech 60%) on full data.
- Root and fix every red cell per run, so the matrix reaches parity in fewest runs.
- **Parity work follows [PARITY_READINESS.md](PARITY_READINESS.md)**: rules, method, scoreboard, run queue.
- **Scope:** `felix`, `oort`, `oort_star`, `refl`, `feddance`, `fedbuff` (+ `*_oracle` arms, FX-N13). Out: `fedavg`, `oracle`.
- **Traces:** `syn_0`, `syn_20`, `syn_50`, `mobiperf_3st`. `syn_10` = the old "syn_20" (10.8%, EuroSys'26; FX-D18).
- Levels (n=300, TRAIN/EVAL/UN %) flat 1 min-3 h: syn_10 89/0/11 · syn_20 80/0/20 · syn_50 50/0/50 · mobiperf_2st
  10/0/90 · 3st_50 10/25/65 · 3st_75 10/10/80. Mobiperf trainable share rises to 18-22% by 6-24 h.
- **IDs:** `FX-N` next steps · `FX-L` lessons · `FX-T` tripwires · `FX-D` built. Shared work is `S#` in the parent.
- **Deprecated reference:** [PARITY.md](../async_cifar10/PARITY.md) §2-5 · [PARITY_CHECKER_README.md](../async_cifar10/scripts/parity/PARITY_CHECKER_README.md)
  · [UNAVAILABILITY_DESIGN.md](../async_cifar10/UNAVAILABILITY_DESIGN.md) · [EXPERIMENT_felix_streaming.md](../async_cifar10/docs/EXPERIMENT_felix_streaming.md)
  · [BASELINES.md](BASELINES.md).

---

## Status grid

Parity per run and per baseline: [PARITY_READINESS.md](PARITY_READINESS.md) → Progress dashboard, Felix scoreboard.
**Readiness score** (`scripts/readiness_score.py <pools>`, baseline x dataset x scenario, worst pair wins, later pool supersedes): **parity 52 / 76** (PR29 over PR28,
regraded on FX-D139; syn_0 12/12 · syn_20 12/12 · syn_50 8/12 · mobiperf 7/10 · stream_lin 3/12 · stream_eve 4/12 · stream_cpu 6/6; 16 untested = PR30) · **accuracy 5 / 12** (table below).
pytest: 2572 passed, 7 skipped (10-11, four suites, FX-D136-D140).

**Accuracy table (FX-N74; `scripts/accuracy_table.py --runs <dirs>`).** Test acc % on the leg's own clock, syn_0, reference n.
G1A = full data (runs 16-18); G2 streams data 0 → 100% over 3 h (lower bound). `-` = no eval in window.

| dataset · baseline | leg | real @30 / @60 / @90 · max | sim same | note |
|---|---|---|---|---|
| cifar · felix | G1A | 36.2 / 51.8 / 56.9 · 57.0 | 40.6 / 50.6 / 53.7 · 56.3 | **target at 57 / 60 min** |
| cifar · refl | G1A | 24.9 / 32.2 / 44.5 · 44.5 | 29.1 / 35.7 / 44.1 · 44.1 | rising at 90 min |
| cifar · fedbuff | G1A (2024 lr) | 20.0 / 24.0 / 32.7 · 34.7 | 23.1 / 31.5 / 31.7 · 38.1 | superseded by FX-D73 row |
| cifar · fedbuff | G1AS SGD 0.04 × 1.0 (run 21) | — | 39.6 / 46.0 / 56.4 · 56.4 | **sim target at 63 min** |
| cifar · oort, feddance | G2 stream | oort 24.8 @60 · 37.7; feddance · 16.6 | oort · 31.4; feddance · 35.4 | |
| speech · felix | G1A SGD (run 18) | 41.5 / 58.8 / 67.5 · 67.5 | 46.0 / 56.0 / 68.1 · 68.1 | **target at 72 min both** |
| speech · fedbuff | G1A SGD (run 18) | 40.8 / 54.6 / 59.1 · 60.2 | 42.5 / 55.9 / 64.0 · 64.0 | **target at 82 / 88 min** |
| speech · refl | G1A Adam (run 16) | 3.9 / 29.3 / 38.7 · 38.7 | 3.9 / 31.0 / - · 39.7 | 82 rounds, 65 s/round |
| speech · refl | G1AS SGD (runs 19, 21) | — | lr 0.005: 43.7 max; lr 0.05: 53.0 max | rising, below 60% |
| speech · oort | G1A SGD (run 18) | 25.8 / 44.1 / 50.8 · 50.8 | 18.8 / 45.3 / 55.1 · 55.1 | 102 / 113 rounds |
| speech · feddance | G1A SGD (run 18) | 3.7 / 28.8 / 28.8 · 28.8 | 3.7 / 27.7 / 27.7 · 27.7 | round-bound: 38 rounds, 142 s/round |
| in-process `fl_lr_check` (FX-N74, N80) | reference k, full data | cifar fedbuff 53 @r1000 · oort 60 · feddance 45 · refl (lr 0.1) 43 | speech @r150: oort (eta 0.002) 25 · feddance 18 · refl (FedAvg) 17 | 10-08 |
| speech · oort, feddance | G2 stream | oort **60.7 at 80 min**; feddance 21.5 | oort 57.2; feddance 45.2 | |
| cifar · oort_star, refl, oort, feddance | G1AS source-faithful (PR28 G5) | — | oort_star **50.8, target at 82 min** · refl 29.5 @90 · oort 43.7 @30 (OOM-cut) · feddance 18.5 @30 (FX-D127-cut) | rerun oort + feddance (PR29) |
| speech · oort_star, oort, refl, feddance | G1AS source-faithful (PR28 G4) | — | 16.7 · 14.8 · 3.7 · 4.0 @90 | **round-bound (operator 10-10)**: 49-69 rounds (K=5, 78-110 s/round); r60 matches in-process |

**Cross-cutting capabilities**

| capability | status | item |
|---|---|---|
| checker catches injected sim bugs (P11a-c) | ✅ both datasets, speech P11b now CAUGHT (PR28 C3) | FX-D125 kept sensitivity |
| fail fast; ladder runner + offline re-gate | ✅ | FX-D22, FX-N42 |
| GPU join/warm-up at n=100; sync wait-K | ✅ joins 44-99 s; EV7 green | FX-D19, FX-D20/21 |
| real async ingest < 2 s | ✅ G0U queue_wait max 0.37 s (run 25); U6 felix ≤ fedbuff both datasets (run 27) | FX-D16 |
| profiled sim charges per dataset/platform | ✅ K3b green every syn_0 cell | FX-D23 |
| DIST tolerances from replicate floors | 🟡 n=2 floors gate regrade; none tightened | FX-D37, PARITY Q2 |
| streaming / oracle experiment | 🟡 cifar EV19 green; oracle advantage ungraded | FX-N13 |
| sim speedup vs real | 🟡 cifar 4.5-7.4×; 🔴 speech 0.6-1.1× (GPU-bound); wait-K feddance 0.3-0.7× was FX-D127 | FX-N22 P8 |
| paper experiments sim-only | ⬚ | FX-N12 |

**Validation matrix (goal: real↔sim parity on correct behaviour).** Short = pytest, stored telemetry, CPU tiers, ≤ 30-min GPU screens, in-process `fl_lr_check.py` (PARITY C18).

| axis | cells | latest short-run evidence | left to test (short) |
|---|---|---|---|
| syn_0 × six, cifar | 6 | T3 syn_0/0b all six ✅ (PR28 C2); P7 4/4 graded ✅ (PR28 C5; felix/oort grades cut by deadline) | grade C5 P7 felix/oort offline |
| syn_0 × six, speech | 6 | T3 syn_0/0b all six ✅ (PR28 C2); P7 dropped (FX-D135) | — |
| syn_20 × six, both | 12 | T3 (PR28 C2): cifar 4/6 (oort, oort_star syn_20s timing); speech 5/6 (oort_star syn_20s timing) | Oort floors (FX-N85 s2) |
| syn_50 × six, both | 12 | T3 (PR28 C2): cifar 3/6 (oort real EV17 → FX-D129; feddance timing), speech 4/6; G0U (G1, B1): 9/12 (speech feddance EV16 → FX-D130; cifar fedbuff/feddance timing) | PR29 confirm |
| mobiperf_3st × six, both | 12 | cifar T3 6/6 ✅; speech T3 4/6; G0U (PR28 G3, B5): 5/8 + cifar oort (terminal_state only); oort_star sim not run | G0U cifar oort_star mobiperf (PR29); T3 (C7 skipped) |
| streaming lin/events × syn_0/syn_50 (+oracle) | 6 × 2 × 4 | cifar G0T non-Oort EV19 16/16, INV/EXACT 13/16 (PR28 B4); speech G0T felix+fedbuff 8/0/0 | PR29 B2 cifar G0TC floors; PR30 speech G0T six + Oort `G0T_*s` (FX-D134); G0To |
| checker sensitivity P11a-c | 2 | ✅ 6/6 CAUGHT both datasets (PR28 C3) | — |
| correctness: accuracy to target | 12 | 5 / 12 (G1A 3 + cifar fedbuff, oort in-process); cifar oort_star sim 50.8% (PR28 G1AS) | G1A reals (PR21); speech round-bound decision |
| DIST replicate floors | per cell | n = 2 | T3C ≥ 3 legs per cell (FX-N42 Q2) |

---

## Active build — fast parallel harness (FX-N22)

CPU harness catches logic bugs; distribution parity is decided on GPU.

**Isolation contract: a parallel leg behaves as if alone.** Each slot gets:
- disjoint cores via `taskset` (`FLAME_AGG_CORES`: 2 at n≤30, 8 on GPU legs);
- a private mosquitto (`FLAME_MQTT_BROKER`); fixed client ids collide on shared brokers;
- a run tag (`FLAME_RUN_TAG`) scoping kills, and exact run-dir handoff (`FLAME_RUN_DIR_FILE`);
- GPU legs: healthy GPUs (FX-T36), CPUs = 8 + 0.4·n (cifar 0.2·n, FX-D49).

**Shapes** (`harness_pool.SHAPES`; aggGoal via launcher `agg_goal` = sync K, FX-L36):

| shape | n | aggGoal | c | runtime | trace scale |
|---|---|---|---|---|---|
| syn_0 | 12 | 3 | 5 | 180s | – |
| syn_0b | 15 | 2 | 8 | 180s | – |
| syn_20 | 15 | 3 | 6 | 240s | 4 |
| syn_50 | 15 | 3 | 6 | 1200s | 4 (FX-L43) |
| mobiperf_3st | 45 | 2 | 4 | 240s | 4 (FX-L34) |
| syn_0s / syn_20s / syn_50s | 30 / 40 / 40 | 10 | 13 | 300 / 300 / 1200s | – / 4 / 4 (Oort family, FX-D106) |
| mobiperf_3sts | 165 | 10 | 13 | 240s | 4 (Oort family, cpt 0.4) |

**Tiers** (`scripts/harness_pool.py --tier A[,B] --datasets cifar10|google_speech|all`). Every pool opens with a ~4-min gate
(pytest collect + felix smoke pair per dataset). Speech CPU sims get a 2× wall ceiling (aggregator-bound).

| tier | jobs | wall |
|---|---|---|
| T1 | sim-only, syn_0 120 s + syn_50 360 s, EV only | 13 min |
| T2 | sim-only × 4 shapes vs banked reals + P11a-c | 38 min |
| T3 / T3C | real+sim pairs × 4 shapes / a real replicate per cell (floor) | 82 / 22 min |
| T3S | oort/oort_star/refl at aggGoal 10 (stragglers cross rounds) | 24 min |
| T4 | campaign P1-P11 (`harness_campaign.sh`) | 70 min per dataset |
| TS / TSo | streaming CPU pairs × {linear, events} × {syn_0, syn_50}; +oracle | 7 min |
| PROF | n=10, aggGoal = c = 10, `FLAME_PYSPY` profiles (FX-D77) | 4 min |
| G0 / G0C | GPU 30-min screen, syn_0 + syn_20 / real replicate | ~6.9 h node |
| G0U / G0UC | unavailability screen syn_50 + mobiperf, n=50 / real replicate | cifar 15 min, speech 30 min |
| G0T / G0To | streaming GPU screen × {lin, events} × {syn_0, syn_50} / oracle | cifar ~2 h |
| G1 / G2 | 90 min reference config: felix+fedbuff / other four | ~2.5 h per pair |
| G1S | felix + fedbuff, reference n, 45 min, 4 GPUs (fix confirm) | speech ~100 min |
| G1A / G1AS | accuracy: syn_0, full data, eval every 20 rounds, 90 min / sims only | ~2.5 h / 25 min |
| G1U / G1L | felix+fedbuff syn_50 at reference n / felix syn_0 7500 s + replicate | ~5 h each |
| G2S / G2C | one G2 cell replicated sim-only / real-only (chaotic-selector floor) | 8 / 97 min |
| N62 / N64 | unaware oort syn_50 3 h CPU / refl speech health (EV18) | ~5 h / 45 min |

- `--changed <ref>` picks affected baselines; `--exclude-phases` drops phase ids; `--gpu-ids` pins GPUs.
- Pool skips busy GPUs/cores between legs, scaling with foreign load (`LOAD` in `pool.log`).
- All pools run on jayne; run dirs live in `async_cifar10/experiments/`.
- Tests: `tests/harness/test_{slot_isolation,harness_pool,fl_data}.py`, `tests/launch/test_debug_run_dataset_profile.py`.

## Active build — streaming time-to-accuracy (FX-N12, FX-N13)

- **Experiment:** six baselines × two datasets × four traces × two stream modes; metric time-to-target.
- **Claim:** Felix's eval selector finds clients whose data just grew; others don't.
- **linear:** each trainer grows 10% → 100% of its shard by the horizon.
- **events:** starts 10%; nine 10% chunks land at seeded uniform times, unsynchronized.
- Stream clock = trace time; horizon 6 days (paper) or leg span (screens); chunks ignore availability.
- **Built (FX-D47):** one `stream_schedule.py` for trainer, oracle, replay and EV19; tiers TS/TSo, G0T/G0To.
- **Left:** ST5 speech on GPU screens (FX-N30) · ST6 sim-only sweep + oracle advantage (needs P8).

## Unavailability v1 (FX-N6 audit, 2026-10-04): kept; open rows below

Kept rules live in Built (Availability) and FX-L11-L15, L25, L33.

| item | verdict | note |
|---|---|---|
| A2 two-tolerance shape (land-mine 3) | open | refl syn_50 A2 KS 0.30 = stall placement: per-instant UN_AVL sequence identical (run 25, FX-L62) |

## Plan of action — 1-day close (operator 10-10; scope: syn_0/20/50 first, mobiperf last)

*Claim:* all six baselines x both datasets reach INV/EXACT green and DIST within replicate floor on syn_0/20/50, and streaming
screens are green, within ~24 h. *Out of today:* FX-N12, FX-N22 P8 speech speedup, FX-N84, FX-N33, mobiperf if time runs out.
*Done =* INV/EXACT green; DIST within floor; chaotic cells (Oort stall placement, feddance speech A_m near-ties, n=50 stall
placement: FX-L53) graded on >= 3-leg real floors, no sim change (PARITY R7).

| step | when | what | exit |
|---|---|---|---|
| P0 | done 10-10 | PR27 A-D triage (R1-R5 below) | red list with root or "floor" per cell |
| P1-P3, P5 | done 10-10 (PR28, below) | combined pool on frozen code: T3 x six x both, G0U syn_50 + mobiperf, T3C/G0UC floors, cifar G0T, G1AS sims | per-cell INV/EXACT + floor-gated DIST |
| P4 | now | PR28 roots fixed (FX-D127-D131, uncommitted); one confirm pool PR29 (PARITY run queue); remaining reds floor-bound or named items | scoreboard rewritten (C5) |

- **Rules:** one combined pool per code state; every red gets a root or a floor label before a rerun (C1, C10).

**PR28 results (10-10; `experiments/pr28_20261010_0215`, pools `pool_20261010_0215_PR28_<stage>`; `--grade --max-stage 9`, regraded).**
g/k/r = INV/EXACT cells green / known / red; EV fails count red.

| stage | result | reds → root or label |
|---|---|---|
| C1 T3 cifar Oort syn_50s + clip | 0/0/2; 0 `update_rejected` | oort real coverage 0.34, oort_star `terminal_state` → FX-D129 class (confirm PR29) |
| C2 T3 syn_0/0b/20/50 x six x both | 37/0/7; syn_0/0b 20/20 | cifar oort syn_50s real EV17 → FX-D129; timing EXACT 3-17% mixed sign: cifar feddance syn_50 (picks fork at near-ties, PR27 R3 class), Oort syn_20s both datasets, speech refl syn_50 → floors |
| C3 T4 P11a-c both | 6/6 CAUGHT | — |
| C4 T3C speech floors | syn_20s Oort, feddance syn_20/50: 3rd real leg ✅ | syn_50s Oort real legs OOM-killed (G5, FX-D131) → rerun |
| C5 P7 both | cifar 4/4 graded ✅ (felix/oort grades cut by deadline); speech 0/6 | speech = FX-N86 |
| N1-N4 gdb | 4/4 clean exit | FX-N33 abort not reproduced |
| G1 G0U syn_50 x six x both | 7/0/3 | speech feddance sim EV16 → FX-D130; cifar fedbuff 7.8%, feddance 10% throughput (< run-27 floors 0.18 / 0.13; no cifar G0UC floor file) |
| G2 G0UC speech feddance, Oort | EV 3/3; floor clears speech feddance `terminal_state` | — |
| G3 G0U mobiperf non-Oort | 5/0/3 | cifar fedbuff 12.7% (> mobiperf floor 0.07: open), feddance 13.6% (< 0.56), speech feddance 8.1% |
| G4/G5 G1AS sims | accuracy table | feddance both EV12 → FX-D127; cifar oort EV0 = OOM (FX-D131) |
| B1-B3 cifar Oort clip G0U/G0UC | B1 2/0/0; EV 6/6; 0 rejects in 14 clip legs | R1 decision (FX-N85) |
| B4 G0T cifar non-Oort | 13/0/3; EV19 16/16 | felix lin syn_0 9.7%, fedbuff eve syn_50 11.4%, feddance lin syn_50 `terminal_state` → floors (G0UC cohort) |
| B5 G0U cifar Oort mobiperf_3sts clip | oort: `terminal_state` only | oort_star sim + grade never ran (pool hang, FX-D131) |

- **PR29 results (10-10; `experiments/pr29_20261010_1606`, regraded; 9 stages, 9 h).** Closed: cifar Oort clip (0 rejects in 26 legs, B1 syn_50s green),
FX-D127/D128 (G3 speech feddance sim_rate, G4 feddance + oort sims reach budget), FX-D129 (cifar Oort real EV17 PASS, C1), cifar G0T feddance EV16 (FX-D136), P7 6/6.
Open reds after the FX-D139 regrade (9 cells), every one rooted from stored telemetry (10-11):

| cell | red | root |
|---|---|---|
| speech Oort T3 syn_50s real; cifar oort_star G0U mobiperf_3sts real | EV0 (+EV1/EV12) | host RAM 504/504: G4's 398 GB leg + B1's 222 GB leg co-ran (620 > 504); FX-D136 ledger |
| speech feddance G0U syn_50 | MISSING | DOOMED EV16 false positive; FX-D136 (stored prefix now EV16 PASS) |
| cifar fedbuff G0U mobiperf (14.6%), G0T lin syn_50 (8.5%) | throughput | FX-D137: abandon wake left buffered ends RECVD, sim picked 3 where real picked 1 (reals agree 39 commits) |
| cifar feddance T3 syn_50 | throughput | FX-D138: stub overran D, barrier dropped the pick, wait-K jumped to the flip (3 reals identical, sims identical) |
| speech feddance G0U mobiperf | throughput 9.7% | chaos: utilities differ by GPU numerics, real<->real forks one pick later (R3); needs n>=3 floor |
| speech oort_star T3 syn_50s | trainers_at_n 37 vs 39 | draw: 0403 always an overcommit straggler (real committed 1 of 18), 0372's rare eligibility windows |
| (cleared) cifar oort T3 syn_50s | overlap_factor | FX-D139: graded on 1 stall-free round; reals 26 vs 8 rounds (FX-L53) |

*Ops:* G5's 398 GB legs started with 272-285 GB free (idle-pool RAM exemption): RAM 504/504 OOM-killed C4 syn_50s reals and G5 oort sim.
  B5's pool spun 3.5 h in `simulate_makespan`. Both fixed (FX-D131). Skipped by the 8 h stop: C6-C8, B6, B7.

## Next steps (persistent queue — top item is next)

Work rule: PARITY C10, C17, C18. Run queue + pre-launch checklist: PARITY_READINESS.
**Test levels:** pytest (~4 min) · CPU tiers T1-T4/TS (real processes, MQTT, n 12-45) · GPU ≤ 30-min screens (G0U/G0T/G0UC) · long (G1*/G2/blocks).
**Unblock map:** short-run queue empty → long runs (PR21) → FX-N11 → FX-N12 (also needs FX-N13 ST6 + FX-N22 P8).

*PR27 root causes (10-10; PR27 A-D done, graded offline):*
- **R1 cifar Oort/Oort_star NaN, real AND sim (training instability, not a sim bug).** Source config (SGD momentum 0.9, 20 steps, lr 0.04)
  on 1-8-sample streaming shards blows up one delta; the finite-but-huge update (weight norm 16.8 -> 52.5, stat_utility 3238) poisons
  version 3-4, then ~35-55% of tasks return NaN. NaN legs since the K=10 shapes: real 13/94 (mobiperf_3sts oort), 12/55 (syn_50s
  oort_star), 2/819, 2/359; sim 18/21, 8/43, 31/91. FX-D111/D120 reject-and-replace then re-picks until the pool drains (82 rejects in one
  round): sim rounds 131 s vs real 16 s, in_flight 7.9 vs 1.0, so eligibility, overlap, total_commits, matched_budget_coverage, selection_bias
  all go red downstream. FX-D107 (server first step) does not reach it. Fix landed opt-in (FX-D126); *decision for operator:* enable
  `trainerClipGradNorm: 1.0` for Oort/oort_star cifar as a recorded deviation (ROBUST L37 ladder rung), else grade those cells on real replicates.
- **R2 sampled availability checks were noise on short legs.** A2 KS 0.213-0.245, A3 0.203-0.225, A4dur 0.06-0.10 on 13-17-selection legs
  (real<->real feddance speech syn_50: A2 KS 0.213 between two reals); A4dur's block of identical 0.2977 errors was a 90 s real stall
  (no selections 150-239 s) forward-filled as the old state, trace and trainer `avail_change` were right (150.5 s). Fixed: FX-D125.
- **R3 feddance speech syn_50 U6 (commit_visibility 48 vs 27 s) = chaos, not mechanism.** Rounds 1-3 identical (lags and speeds), picks fork
  at round 4 (A_m near-ties, R7); real<->real mean lag 48.2 vs 39.1 s, p90 113 vs 134 s. Floor-gate it (no sim change).
- **R4 cifar T3 syn_50s Oort `trainer_speed_identity` = harness artifact:** 11/177 real tasks hit TIMING_OVERRUN (stub 20 real CPU steps
  0.64 s > D 0.5 s under pool load), real speed = gpu time by C13; also its `utility` (stub loss on 8 samples blows up, R1).
  Open: cap cifar stub local steps like speech (FX-D110), needs the cifar stub charge profile re-derived first.
- **R5 not rooted, all on 7-14-round legs:** speech Oort syn_20/syn_50 `total_commits`, `participation`, `inter_arrival_order`,
  `selection_detail` (n_selections 13-15; underpowered) and cifar T3 oort_star syn_50 `state_timeline_agreement` (0.9135, same stall-hole
  class as R2 in the A5 builder). Next: T3C replicates for the floor before any change.
- **Offline regrade after FX-D125 (`parity_regrade/`), cells red on any DIST/EXACT/INV:** A 2 -> 2 (commit_visibility only, R3); C 7 -> 5;
  D 4 -> 4 (R1). Stored `parity/` JSON and SUMMARY.txt are unchanged (pre-fix).

*Resume here (10-11). State: FX-D136-D140 uncommitted in tree, pytest green, `run_pr29b.py` smoked; next action = operator launches
PR29b (PARITY run queue), then PR30; after each, regrade + `readiness_score.py` + `resource_report.py` (C5).*
- **FX-N85 · Close PR27 roots R1-R5 and PR28 roots · wip.** Order:
  1. **R1 done (operator 10-10):** clip 1.0 is the cifar oort/oort_star default (`baseline_reference.yaml` deviation; full-data r300 48.9/54.5%
     -> 48.9/49.9%, r200 equal, `experiments/lrcheck_20261010_clipfull`; tiny-shard NaN 3/6 -> 0/6). C20: older cifar Oort legs are void.
  2. **PR29b (`run_pr29b.py`):** confirm FX-D136-D140 on PR29's reds (table above) and re-screen the baselines whose sim changed.
  3. **Floors (R3, R5):** chaotic cells (speech feddance mobiperf, speech oort_star syn_50s): a second real replicate in the next batch if still red (operator 10-11).
  4. **R4:** re-derive the cifar stub charge profile, then cap cifar stub local steps (`harness_stub_max_steps=1`, as speech FX-D110) for the Oort family
     in `harness_pool.shaped()`; PL11 check; rerun T3 cifar oort syn_50.
  5. **R5 A5 builder:** apply FX-D125's observed-window rule to `state_timeline_agreement` (cifar T3 oort_star syn_50 0.9135).
  6. **PR30 (`run_pr30.py`):** the 16 untested streaming cells (speech G0T six, Oort `G0T_*s` both datasets) with G0TC floors.
- **FX-N80 · Verify source-faithful configs (FX-D100, D104, D105) · todo.** C19:
  - *Claim:* every baseline learns on our model with its source knobs, or with the fewest recorded adaptations (ROBUST L37).
  - *Step 1 done (`experiments/lrcheck_20261008_fxn80`):* cifar oort 60% peak (target ~r250); feddance cifar 45% r1000, speech
    36% peak r600, slow but learning (1 step); ladder adaptations recorded (`baseline_deviations.py --md`): refl cifar lr 0.01 -> 0.1,
    refl speech YoGi -> FedAvg server, oort speech YoGi eta 0.005 -> 0.002. Client momentum is not the speech slowdown (A/B).
  - *Step 2 · wip:* T3 + G0U syn_50/mobiperf screens for refl, oort, oort_star, feddance. *Stop:* any INV/EXACT red or NaN.
    N88 (`pool_20261009_N88_T3` 4/14 red, `_N88_G0U` 6/14 red after FX-D119 regrade; oort_star T3 6/6 green) rooted to
    FX-D116 (speech sync sim +0.4 s/round), FX-D117 (feddance speech syn_50 sim rounds 150 s), FX-D118 (refl speech real 303 s
    round), FX-D119 (checker graded all-stall legs), FX-D120 (NaN rejects deadlocked G0U oort_star syn_50 sim to the budget).
    PR26 (`pool_20261009_PR26_{G0U,T3C}`): refl speech 2/2 green; Oort T3C 14 real replicates all PASS; feddance speech reds rooted
    to FX-D121-D124 (landed, unverified on cluster: PR27). Open: (a) cifar Oort NaN: FX-D107 did not remove it; rooted in PR27 R1, FX-D126 clip opt-in (FX-N85).
    (b) Oort stall placement and trainers_at_n: real<->real forks too (gs syn_20 oort: R/C stall at rounds 3+10, sim 3+13; rounds 11/11
    vs 23); selection forked on a 3.5% pref-duration bias (real wall = D + weight staging) -> FX-D123, rerun T3 Oort. (c) feddance
    speech picks fork at near-ties of A_m (relu clamp 1e-6 vs k/32: a 10^4 utility step on one sample of local accuracy): chaos by
    design, needs the G0UC replicate (R7) before any sim change. A6r 0.94 vs 0.95 on one real leg: n=50 sampling, floor-gate it.
  - *Step 3 done:* 0 TIMING_OVERRUN in 563 Oort GPU tasks (PL3 + N80_G0U legs, both datasets); 3-5% overran on tiny_cpu only.
- **FX-N84 · Trainer memory audit (GPU + host RAM) · todo.** Per-trainer CUDA context, model, optimizer state, allocator cache and
  RSS at n = 50/100/300 (cifar + speech); find waste that scales with n or model size; fix; re-measure trainers per A40
  (FX-T36: ~75 cifar / ~25 speech today; n=200 speech also needs a new split). *Exit:* footprint table, fixes, new caps.

*Short-run queue (fix + verify with pytest, stored telemetry, CPU tiers or GPU screens):*
- **FX-N9 · Unavailability, all six x syn_50 / mobiperf · wip.** PR28: G0U syn_50 x six x both 9/12 green (G1 + B1; reds: speech
  feddance EV16 → FX-D130, cifar fedbuff/feddance timing → floors); G0U mobiperf non-Oort 5/8 (G3), cifar oort `terminal_state` only (B5);
  A1-A8/K11 green in every PR28 cell. Left: PR29 confirm; cifar oort_star mobiperf sim (B5); T3 mobiperf (C7 skipped); FX-N85 s6.
  *Exit:* A1-A8/K11 green per cell.
- **FX-N13 · Streaming, both datasets · wip.** Cifar G0T non-Oort: EV19 16/16, INV/EXACT 13/16 (PR28 B4; timing → FX-N85 s6).
  Left: PR30 Oort `G0T_*s` (FX-D134), G0To (B6 skipped); speech = FX-N30.
  *Exit:* EV19 + INV/EXACT green both datasets; ST6 oracle advantage graded.
- **FX-N30 `[S]` · Speech streaming on GPU screens · wip.** felix + fedbuff G0T lin/events × syn_0/syn_50 INV/EXACT 8/0/0, EV
  8/8 (`pool_20261008_1845_G0Tgs`; DIST reds = FX-N76 gpu_compute/speed identity). Left: PR30 (all six on HEAD, G0TC floors); G0To. *Exit:* speech G0T/G0To lin + events x syn_0/syn_50 EV19 + INV/EXACT green.
- **FX-N74 `[C]` · Every baseline reaches target on full data · wip: 5 / 12 (G1A cifar felix, speech felix + fedbuff; cifar fedbuff + oort in-process).**
  PR28 G1AS sims (accuracy table): cifar oort_star 50.8% at 82 min; refl 29.5% @90 rising; oort, feddance cut (OOM, FX-D127): PR29 G4.
  Speech refl/feddance/oort/oort_star: recorded round-bound (operator 10-10); longer runs only once correctness + parity are green.
  *Exit:* each cell at target, or audited faithful and named round-bound.
- **FX-N76 `[C][S]` · C13 real-cost audit · wip.** weights_to_ram matches (run 27). Open: speech felix sim gpu_compute 3.7 vs
  real 1.9 s (sim GPU contention; off the clock while < D). *Exit:* speed identity + phase DIST green both datasets.
- **FX-N79 · feddance P7 DIST borderline · rooted, confirm.** selection_bias KS 0.234, commit_visibility 0.202 (tol 0.20, n≈120).
  Root: real per-commit overhead 0.305 s (A2 diag) skewed real's picks slower (13.7 vs 12.5 s); FX-D92-D99 cut it to 0.064 s
  (`pool_20261008_2000_wtP7`: KS 0.031 / 0.119). Today's checker still fails the old pair, so the fix is runtime, not checker.
  Replicate 1 on HEAD green (PR28 C5 cifar P7 feddance); replicate 2 (C6) skipped. *Exit:* both green.
- **FX-N33 `[C][S]` · Aggregator abort at interpreter exit · todo.** `terminate called without an active exception` after channel
  leave; Python dump shows one thread, no frame (C++ static teardown). Frequent on oort T1 syn_50 sim. *Next:* gdb backtrace via
  `FLAME_AGG_CMD_PREFIX` (FX-D102) = `/coc/scratch/dgarg/gdb_env/bin/gdb -q -batch -ex run -ex 'thread apply all bt' --args`.
  PR28 N1-N4: 4/4 exited normally under gdb (not reproduced; gdb may perturb teardown timing). *Next:* grep PR29 legs for the
  line; if absent there too, close. *Exit:* root or a clean join.
- **FX-N42 `[S]` · Parity ladder · wip (PARITY Q2-Q6).** *Exit:* a run graded per cell on both axes, DIST on ≥ 3-leg floors.
- **FX-N10 · google_speech on the launcher · wip.** Stop rule wired. *Next:* S4 cleanup. *Exit:* all six graded on speech, CPU + GPU.
- **FX-N22 · Fast parallel harness · wip.** *Exit:* ISO P6 EQUIVALENT at cpt ≤ 0.5; T2 both datasets < 25 min. Absorbs FX-N8.
  Measured: T3 legs use p95 ≤ 2.7 of 44 CPUs (N86/N88), so pools starve each other (PL12). ISO now takes `--baselines`
  (Oort family on syn_0s) and CPU legs record cores p95. *Next (post-N88):* ISO oort,oort_star solo `--max-parallel 1` at cpt 1
  vs packed at cpt 0.5 and 0.25; `harness_iso_compare.py`. EQUIVALENT ⇒ T3 default cpt 0.5 and GPU default 0.1.
- **FX-N2 · Parent S2 pipeline for async_cifar10 · todo.** *Exit:* stored Jun pairs regrade within floor.

*Long (only after the short queue is empty, PARITY C18):*
- **PR21** GPU block: G1A accuracy pairs (FX-N74; speech feddance longer window if round-bound), G1U n=300, G0T full grid.
- **FX-N11 · google_speech GPU parity · blocked: short queue.** **FX-N12 · paper experiments sim-only · blocked: FX-N11, FX-N13, P8.**

---

## Baseline hyperparameters (operator 10-08)

Single source: `_metadata/baseline_reference.yaml` (value + citation per baseline x dataset; rules ROBUST L33-L37).
- REFL: 1 mini-batch/round [code]; cifar FedAvg + equal stale weight [code]; speech YoGi + SAA -4, lr 0.05 [code]; random sampler;
  `adapt_selection` cifar 1 / speech 0, picks round(K x 1.3) [code].
- Oort/oort_star: 20 mini-batches, YoGi (no momentum), equal-weight average, lr decay 0.95, FedScale utility, upstream getTopK
  [code] (`third_party/Oort`); cifar borrows Oort's CV config.
- FedDance: 1 mini-batch, FedScale defaults (equal-weight FedAvg, decay 0.98/10) [paper]; I_m = mean local training loss (Eq. 6).
- FedScale-family clients (REFL, Oort, FedDance): SGD momentum 0.9, weight decay 5e-4, fresh per task [code].
- FedBuff: 1 epoch, lr normalization [paper]; client/server lr [ours] (paper tunes by sweep). Felix: [ours].
- Sync K = 5% of n (cifar 15, speech 5); async keeps aggGoal 10, c 10% / 30%.

## Felix lessons (dos)

**Selectors (Oort family)**
- **FX-L1** Oort pacer fires on TRAIN only; flat trend raises, sharp drops `round_threshold`.
- **FX-L2** UCB temporal term keys on last receipt round, initialised at registration.
- **FX-L3** If a faithful controller still diverges, instrument its input by quartile.
- **FX-L4** Async fires per-round terms 2-3× more; felix uses `exploration_decay` 0.999.
- **FX-L5** Record stale-but-returned trainers' speed and utility, else Oort re-picks them.
- **FX-L63** Screen Oort at K ≥ 9: below, upstream exploitLen int(K·0.1) = 0, so it only explores (unavailable picks stall 90 s;
  cifar K=2 averages single-class overfits to NaN). FX-D106.

**Aggregation, ordering, clock**
- **FX-L6** Async sim drains per-end queues in sct order; holds busy slots until commit.
- **FX-L7** Sync oort over-selects ×1.3; stragglers land stale next round, as real.
- **FX-L8** refl/oort: a still-computing trainer leaves eligibility via the unavailable path.
- **FX-L9** Strict-barrier baselines anchor visibility lag on the barrier (`max_dur − dur_i`).
- **FX-L10** Split eval from train in `agg_rounds` checks; each eval has its own sct.
- **FX-L30** Replay reads the stream clock: sim vclock, real wall since `AGG_START_TS`.
- **FX-L31** Committed ends leave RECV; a phantom buffered end blocks starvation wake-up.
- **FX-L32** A dispatch consumes the end's earlier receipt and RECVD state.
- **FX-L45** In sim, arrival ≠ receipt: unreceivable updates keep real's unanswered-dispatch state.
- **FX-L54** Reinject every withheld delivery due before the next buffered sct.
- **FX-L60** Keep real measurement off the ingest path; read queue_wait spikes against evals.

**Availability**
- **FX-L11** Two axes: select filter (felix, oort_star, refl, feddance); proactive evict (felix only).
- **FX-L12** Mid-flight dropout: compute completes, send gated or buffered, commits late and stale.
- **FX-L13** Sim jumps the vclock under scarcity; size `--runtime-s` for it.
- **FX-L14** Size n ≈ threshold / (1 − unavailable fraction); in-flight differs per baseline.
- **FX-L15** Ramp syn_0 → syn_20 → syn_50 → mobiperf; only mobiperf exercises AVL_EVAL.
- **FX-L25** Sim starvation wake-up = earliest future slot-freeing event, never a due one.
- **FX-L33** Under the substrate only withhold, evict or abandon free an in-flight slot.
- **FX-L34** mobiperf_3st is ~10% AVL_TRAIN: size n ≈ select / 0.1 × 1.5.
- **FX-L43** A leg must see ≥ 80% of the named unavailability and several cycles.
- **FX-L44** Trace time 0 = join barrier; trainers hold trace time 0 until first dispatch.
- **FX-L58** Instant decisions (send gate, select filter) read the trace at that instant.

**One task per version, rounds**
- **FX-L26** One task per (trainer, version); train at v blocks eval at v; retry policy `none`.
- **FX-L27** Update identity is (trainer, version): dedup commits, drop answered requests.
- **FX-L28** Version advances only on `agg_goal` aggregated updates; empty rounds still free slots.
- **FX-L40** Time out each trainer 90 s after its own dispatch; timeouts never close rounds.
- **FX-L41** Accumulate state in float, cast once; server steps touch float parameters only.
- **FX-L42** An empty sync round re-dispatches the same version, excluding ends already tasked.
- **FX-L48** BN stats = convex mean of absolute stats, never base + scaled stale deltas.
- **FX-L55** Real idle loops wake at the earliest future deadline, on its stamping clock.
- **FX-L59** Watch real selection and telemetry rates; busy-loops commit normally while writing GBs.

**Harness, compute, datasets**
- **FX-L29** Stub compute span was fit on CPU-fallback runs; refit from a GPU real leg.
- **FX-L36** Set shapes via `experiment.aggregator.agg_goal`; it sets aggGoal and sync K.
- **FX-L37** Datasets differ only in `datasets.yaml` + `fl_data.py`; machinery is shared.
- **FX-L38** Pay first CUDA touch in `initialize()` (`_warmup_device`), never in timed tasks.
- **FX-L39** Time sim-speed guards (wall ceiling) from the join barrier.
- **FX-L49** Run no-backward forwards under `no_grad`; free CUDA cache after idle GPU steps.
- **FX-L50** Near-D GPU time makes speed contention-driven; check overrun share first.
- **FX-L51** Profile the aggregator with fresh buffers; reused buffers hide page-fault cost.
- **FX-L57** Check lr pairs in-process (`fl_lr_check.py`; `--flame-opt refl`, `--lr-decay`) first; grade accuracy on full data.
- **FX-L61** A per-task sim span of exactly D + constant is a charge, not a mechanism: check the leg's profile first.
- **FX-L65** A sim leg slower than real time is a bug: read `agg_timing.recv_wait_s` and `[SIM_BARRIER] barrier_wait_s` before blaming load (FX-D127).
- **FX-L64** Size a multi-pool batch by CPU leases and host RAM, not GPUs: `run_pr28.py --plan` list-schedules the stages first;
  RAM estimates come from measured peaks (`resource_report.py`, FX-D140), never the formula alone.
- **FX-L66** Diff a red against its own real replicates first: identical reals + a forked sim = mechanism (FX-D137/D138); forked reals = chaos.

**Reading the checker**
- **FX-L16** A2 red with S3/4 green is one in-flight gap; walk to residence.
- **FX-L17** U6 KS on a sub-ms point mass is noise; read `mean_diff`. P3 needs `grid_KS`.
- **FX-L18** `gate_holds = 0` all run means an inert gate; suspect upstream accounting.
- **FX-L19** Recheck `phase_gpu_compute` and refl K2 at ≥ 2.5 h before acting.
- **FX-L21** `Sdet`: eligible differs + aggregates match = PASS; same eligible, different picks = score values.
- **FX-L22** Past-dating has two streams (U6 train-only, SIM_CLOCK_DIAG); name the source.
- **FX-L23** `sim_committed_fresh == agg_goal` confirms oort block-for-K; later gaps are other roots.
- **FX-L24** Re-run real only when the real path changed.
- **FX-L46** Split sync clock residual into stall-free advance and stall count.
- **FX-L47** Audit real receive: arrivals vs processed, and growing `active_task_skips`.
- **FX-L52** A sync round holding an abandon or gated update is a stall at any length.
- **FX-L53** syn_* flips share slot boundaries; grade n=50 timing on G0UC floors.
- **FX-L56** Align real and sim on trace time or matched work, not round fraction.
- **FX-L62** Split A2 by `excluded_by` and trace instant first: same UN_AVL sequence + different stall placement = sampling, not mechanism.

## Felix tripwires (don'ts)

**Selectors (Oort family)**
- **FX-T3** Don't add a `system_util` recency guard or widen oort's slow-speed tail.
- **FX-T6** Don't key oort latency by task type; sync oort sends no evals.
- **FX-T8** Don't expect seeding to align per-round selections; judge S2 by speed class.
- **FX-T17** Oort carry-over decay is structural; felix min-budget seed alone won't fix past-dating.

**Aggregation, ordering, clock**
- **FX-T1** Don't enable `simStaggeredRedispatch` or retune `simRedispatchGapSeconds`.
- **FX-T2** Don't add `mqtt_fetch` to sct; it is pre-selection wait, not transfer.
- **FX-T4** Don't tune the `_sim_recv_min` gate predictor; `exp == sct` exactly.
- **FX-T7** Don't clamp felix's clock jump for past-dating; it was eval reusing stale sct.
- **FX-T14** Don't add prediction-only gates or a `version_at(sct)` relabel; both inert.
- **FX-T15** Don't read wall-clock commit density as sim speed; judge per-round vclock advance.
- **FX-T18** Don't add a scalar fudge for P3's ~1 s `mean_overhead` (wall-capture).
- **FX-T20** Don't gate sim ingest on "committed this cycle"; it livelocked P3.
- **FX-T21** Don't call `ends()` except for a real dispatch; use `all_ends()` to list.
- **FX-T22** Don't skip a trainer after the selector chose it.
- **FX-T33** Don't enable `simInflightCarryover`; exclude stragglers at eligibility instead.
- **FX-T34** Don't mark a stale gated straggler's landing round as a stall.

**Availability**
- **FX-T5** Don't hold all buffered ends out of refl's pool; it over-holds.
- **FX-T9** Don't express "busy" through the UN_AVL list.
- **FX-T10** Don't broadcast availability per tick over MQTT; read the trace.
- **FX-T11** Don't exclude AVL_TRAIN from eval on 2-state traces; the pool empties.
- **FX-T19** Don't fork withhold/abandon per stack; grade A4 via A4dur.
- **FX-T35** Don't let a wall failsafe drop a sim update before its task timeout.

**One task per version, rounds**
- **FX-T24** Don't add a staleness cutoff to fedbuff; its 1/√(1+s) discount is the baseline.
- **FX-T26** Don't set `aggr_num` or `aggGoal` directly; `agg_goal` overwrites them.
- **FX-T32** Don't fix negative `running_var` by clamping at 0; eval loss explodes.
- **FX-T37** Don't count stale updates toward REFL's K; reference closes on K fresh.

**Harness, compute, datasets**
- **FX-T13** Don't grade fedbuff/felix real while another n=300 real shares the broker.
- **FX-T16** Don't re-chase GPU contention, SEND_TIMEOUT or MQTT drops at cifar n=300.
- **FX-T23** Don't run more stub trainers than node cores.
- **FX-T25** Don't kill FL workers by process name; scope by `FLAME_RUN_TAG`.
- **FX-T27** Don't use 2024 speech lr pairs or Adam on speech; staleness makes it oscillate.
- **FX-T28** Don't read leg health from shared `experiments/`; read the leg's own dirs.
- **FX-T29** Don't write replay input in place or from a daemon thread.
- **FX-T30** Name traces by measured unavailable fraction, not intent.
- **FX-T31** Don't retune sim charge knobs to close a clock rung; use profiles.
- **FX-T36** Don't exceed ~75 cifar / ~25 speech trainers or 7 speech trainings per A40.
- **FX-T38** Don't key liveness on log growth past `run_end`; teardown is silent.
- **FX-T39** Don't stack two eval gates; evals land every 2N rounds.
- **FX-T40** Don't treat a "replied/returned" set as version-free: a return answers only its end's latest dispatch (FX-D129, FX-D127).
- **FX-T42** Don't read a regraded red as floor-gated unless its grade dir holds `floors.json` from the same code state (FX-D132).
- **FX-T43** Don't bound a sim barrier by D alone: a compute overrun reads as absent and the clock jumps past it (FX-D138).
- **FX-T44** Don't return from a sim drain with buffered ends RECVD: the selector frees their slots (FX-D137).
- **FX-T41** Don't exempt an idle pool from the RAM check: PR28 G5 started 398 GB legs at 285 GB free and OOM-killed three legs (FX-D131).

---

## Built (one line each; details in code, tests, `git log`; IDs kept because code cites them)

**Simulator fidelity**
- **FX-D1** Core sim: sct-ordered drain, one-in-flight residence, faithful oort/refl/feddance mechanics.
- **FX-D4/D8** Cold-start gate (`simColdStartGate`) and uncapped busy hold, freed on abandon/evict.
- **FX-D6** asyncfl sim: no stranded same-cycle re-dispatch; refl starvation wake-up.
- **FX-D9** One task per version: dispatch ledger, no-repeat guard, commit dedup, EV15.
- **FX-D10** Sync rounds advance only on a committed aggregation; all-stale rounds free slots.
- **FX-D17** Sync stacks abandon 90 s after dispatch in real too; refl math on float only.
- **FX-D20** `syncWaitForK`: version advances only on `agg_goal` accepted updates; picks topped up.
- **FX-D21** Empty sync rounds re-dispatch from the ledger; `run_end` marks the stop point.
- **FX-D23** Sim non-compute charges profiled per (dataset, harness, stack) (`profile_felix_charges.py`).
- **FX-D100** Baselines are source-faithful from `baseline_reference.yaml` (launcher layer, cited, tested): mini-batch `localSteps`,
  FedScale YoGi (`fedscale_yogi.py`, parameters only), `fedavg_yogi`, REFL `sample_mode`, Oort `overcommitment`, FedScale-form Oort
  utility (`statUtility`), FedDance I_m from `TRAIN_LOSS_MEAN`, FedBuff `lrBatchNormalize`.
- **FX-D99** `tiny_cpu_cifar10` charge profile (P6 + P7 reals); a leg without one prints a WARNING (FX-N71: 0.6 s placeholder).
- **FX-D26** Run-4 roots: warm-up optimizer, syn_50 ordering, trace origin at join barrier.
- **FX-D27** Oort-family selection follows REFL fork's `getTopK`.
- **FX-D28** asyncfl sim re-injects due withheld deliveries before starving (`sim_reinject_when_idle`).
- **FX-D34** Sim mirrors unanswered dispatches: withheld picks hold slots; evicted updates ingested.
- **FX-D39** `simInflightCarryover` off for oort/oort_star; stale receipts land one round late.
- **FX-D50** Async sim pops withheld deliveries before next sct (`sim_reinject_lookahead`); gate holds to timeout.
- **FX-D66** Oort sync never awaits an end already on the cleanup queue.
- **FX-D69** FX-D50 lookahead also counts earliest in-flight expected completion (`_sim_next_event_ts`).
- **FX-D70** Oort sync stack reinjects withheld deliveries with the FX-D50 lookahead.
- **FX-D71** Felix D.1 eviction skips a buffered update only once complete (`sct <= now`).
- **FX-D74** C13: sim adds measured pre + post-train time to duration (`sim_charge_trainer_overhead`).
- **FX-D75** D.1 eviction also frees a send-gated buffered update (real never receives it).
- **FX-D76** Late withheld deliveries of evicted/abandoned ends never retake freed slots.
- **FX-D88** Async sim withholds send-gated buffer heads first (`sim_withhold_before_gate`).
- **FX-D89** Async sim abandon deadline is a clock event (`sim_abandon_wakes`).
- **FX-D90** Send-gated held slot frees at delivery commit, not reinjection (`sim_hold_delivering_slot`).

**Availability**
- **FX-D2** Unavailability v1 for all six: send-gate/deliver-late, two ledgers, evict, starvation advance.
- **FX-D5** Real withheld updates are delivered; `mobiperf_3st` launchable.
- **FX-D12** Only withhold/evict/abandon free slots; dispatch consumes the end's earlier receipt.
- **FX-D18** Traces at named level from t=0, 149 h, C loader; `effective_unavailability()`.
- **FX-D55** Real send gate rereads the trace at send, polls 0.1 s (`FLAME_SEND_GATE_REFRESH`).
- **FX-D94** One aggregator trace key `availability_trace`, runner-fanned; Felix configs carry no `trackTrainerAvail` (FX-N7).

**Real path and wire**
- **FX-D16** Real asyncfl ingests on arrival via `drain_ready`; no settle sleep.
- **FX-D24** Real async frees an ingested slot at SEND (`release_recvd_at_send`).
- **FX-D35** Real `recv_fifo` releases reader slots when its streamer exits.
- **FX-D40** Aggregator math libs get a quarter of pinned cores (`FLAME_AGG_MATH_THREADS`).
- **FX-D42** asyncfl caches dispatch payload per (version, task, stamp).
- **FX-D44** Aggregators tune glibc malloc to keep freed buffers (`FLAME_MALLOC_TUNE`).
- **FX-D59** Trainers keep glibc malloc defaults; host RAM no longer fills.
- **FX-D60** Real asyncfl idle wait compares stamps on their own clock.
- **FX-D64** Aggregators decode the test set once at load (`FLAME_EVAL_PRELOAD`).
- **FX-D77** `FLAME_PYSPY` profiles every role; `profile_report.py` ranks; tier PROF.
- **FX-D78** Flat tensor weight codec, zero-copy out-of-band frames (`FLAME_WEIGHT_CODEC`).
- **FX-D79** Aggregation on CPU, GPU only for eval/oracle (`FLAME_AGG_MODEL_DEVICE`); eval off ingest.
- **FX-D80** MQTT sends await PUBCOMP asynchronously; whole messages FIFO; one chunk copy.
- **FX-D81** No log line formats a whole payload (`test_no_eager_payload_logs.py`).
- **FX-D82** GPU cache release runs inside real's D padding, outside sim overhead.
- **FX-D83** Waits wake on deliveries (`Channel.wait_arrival`); no polling sleeps.
- **FX-D84** Zero-copy MQTT client `paho_fast.FastClient` (`FLAME_MQTT_FAST`).
- **FX-D85** QoS-2 chunks never re-published; partial message dropped only on new seqno 0.
- **FX-D86** Teardown flushes sends; JOIN/LEAVE ordered after in-assembly chunks.
- **FX-D87** Trainer atexit stops and joins availability/heartbeat threads.
- **FX-D92** Trainer atexit ignores SIGTERM/SIGINT before joining (a late SIGTERM raised there).
- **FX-D93** Real asyncfl stops at the wall budget while waiting, not at its next commit.

**Training correctness**
- **FX-D38** REFL BN stats average fresh updates only; `clamp_running_var` guard.
- **FX-D41** Trainer `evaluate()` resets utility and runs under `no_grad`.
- **FX-D43** Trainer frees GPU cache after the util_cf telemetry forward.
- **FX-D53** REFL fills K with fresh updates only; stale still aggregate (`refl_fresh_k`).
- **FX-D61** FedBuff/Felix BN = mean of absolute stats (`bn_absolute_mean`).
- **FX-D63** Aggregators evaluate every `evalEveryNRounds` round exactly.
- **FX-D72** `--trainer-hp` overrides by_baseline `trainer.hyperparameters` keys.
- **FX-D73** Cifar fedbuff SGD 0.04 b32 × server 1.0 (r1000 53% vs 32%).
- **FX-D97** FedBuff server lr is a required config `learning_rate`; the dataset table is gone (S5).

**Harness, checkers, datasets**
- **FX-D7** Trainer availability thread stops at EOT or shutdown.
- **FX-D11** Streaming + oracle harness (P7/P7o) with offline replay.
- **FX-D13** Event checker EV0-EV19, injected bugs P11a-c, isolated parallel pool.
- **FX-D14** Dataset switch (`fl_data.py`, `datasets.yaml`); speech on the launcher.
- **FX-D15** No cold start in timed tasks: startup warm-up on CPU and GPU.
- **FX-D19** NVML polled only with `FLAME_STAT_THREADS=1`; GPU joins under 100 s.
- **FX-D22** Fail fast on fatal lines (`fail_fast.py`); pool aborts with `ABORT.txt`.
- **FX-D25** Checker times both sides from first train selection.
- **FX-D29** SUMMARY lists UN_AVL share at selection; real bank hashes the trace store.
- **FX-D30** Grader reads the run's own `max_experiment_runtime_s`.
- **FX-D31** P11b `order_by_sct` runs on fedbuff; CAUGHT on both datasets.
- **FX-D32** Run-6 confirms: unaware fedbuff picks UN_AVL; `[TRAINER_HP]` matches config.
- **FX-D33** A7 graded at its instant; K2/K3 stall-free rounds; K3s grades stall rate.
- **FX-D36** Timeout stall = an episode; K4 drops stall time.
- **FX-D37** `parity_check.py --control/--floors` feed `floor_gated_tol` (never tightens).
- **FX-D45** `StallWatch` S1/S2 rules kill only stalled legs (`STALLED.txt`).
- **FX-D46** Speech `device_time_scale: 5` drives delay factor and task timeout.
- **FX-D47** Streaming `stream_schedule.py`, EV19, `stream_growth` DIAG.
- **FX-D48** Checker stamps sync stalls by cause (abandon, gated update).
- **FX-D49** GPU slot CPUs sized per baseline from measured p95; `CPU_SAT` flag.
- **FX-D51** Checker fixes: EV16 sync close, A3 by trace time, S3/4 total picks.
- **FX-D52** `--regrade` reruns event checker; G0UC floors G0U/G0T; `G0U_RUNTIME_X`.
- **FX-D54** Accuracy tooling: `accuracy_table.py`, `fl_lr_check.py`, tier G1A.
- **FX-D56** `--scale-smoke`, DONE lines print GPU peak, `GPU_TIGHT` above 85%.
- **FX-D57** GC pause callback binds its clock at def; no exit traceback.
- **FX-D58** Per-GPU training cap via `flock` slots (`gpu_train_slots`, speech 7).
- **FX-D62** `StallWatch` disarms at `run_end`; pool legs skip plot analysis.
- **FX-D65** Stall-killed legs skip fail-fast; `merge()` skips unstarted phases.
- **FX-D67** StallWatch S1 counts `abandon_timeout` as progress.
- **FX-D68** EV1 needs two train rounds, not 2 × agg_goal.
- **FX-D91** Checker keeps zero per-round advances (same-vclock commits).
- **FX-D95** Timing checks SKIP below 2 matched units, never grade unmatched full-run means.
- **FX-D96** asyncfl selector reclaims emit `abandon_timeout` (real abandons were invisible).
- **FX-D98** A2 grades dispatching selections only; no-op wakes sample at each side's own cadence.
- **FX-D103** Stub charge profiles re-derived after the FX-N77 transport fixes (10-08 T3 reals; speech charges were 2-3x high).
- **FX-D101** Pool gate: one collect at a time per node, one smoke per code state; timeouts ABORT, never crash the pool.
- **FX-D102** `FLAME_AGG_CMD_PREFIX` wraps the aggregator command (FX-N33 gdb diagnostics).
- **FX-D104** REFL `adapt_selection` (cifar 1): per version, picks = max(cap·N, N − stale due within the mean round length); the
  version closes on min(K, picks) fresh; N = round(K × 1.3) as REFL. Mode 2 rejected (no zero-pick round under K-fresh close).
- **FX-D115** C1/C2 SKIP when neither side's accuracy gains ≥ 0.05 over its first eval (stub legs; loss is noise).
- **FX-D116** Charge profiles split `download_leg` (agg -> trainer) from `completion_leg`; sim charges it only when FX-D108's
  measured lag is absent (both had charged it: speech sync sim +0.4 s/round). Profiles re-derived from their recorded runs.
- **FX-D117** Sim wait-K wake counts a FX-D113 carried update as a delivery (else the next trace flip closed the round).
- **FX-D118** Real wait-K recv stops once every awaited pick replied, so distribute tops up (`realRecvUntilAwaited`).
- **FX-D119** Checker: a side with no stall-free round SKIPs K8's time half, U2 and K4 (K8 still grades trainers);
  stall excess is never negative.
- **FX-D120** A non-finite-rejected pick owes nothing at its version: distribute replaces it (`_sync_failed_ends`).
- **FX-D121** K11 late slack counts only if an agg_round closed between delivery and commit (a barrier applies at its close).
- **FX-D122** Sim wait-K drain commits at most K per version (co-due deliveries carry), as real closes at the K-th arrival.
- **FX-D123** Real trainer stamps `CLIENT_TASK_TRAIN_INTRINSIC_S` = sim's duration formula; the selector speed excludes weight
  staging (real read D + 0.1 s, sim D + 0.01). `realIntrinsicClientDuration=false` reverts. Real-vs-sim speed gap 70 ms -> 2 ms.
- **FX-D125** Selection-sampled availability checks (PR27 roots): A4dur integrates both sides over the common trace horizon and only
  over windows both sides sampled (a real 90 s stall hole was credited to the old state: 12 trainers, TV 0.30); A3/A4dur SKIP below
  `MIN_SEL_SAMPLED` = 20 selections; A2 KS tolerance = max(0.2, 1.22 sqrt((n+m)/nm)) (`ks_sample_tol`, never tightens). Offline regrade of
  PR27 A/C/D: cells with a red A2/A3/A4dur 9 -> 3; real signal stays red (D mobiperf, R1).
- **FX-D126** Opt-in client grad-norm clip `trainerClipGradNorm` (default 0 = off; source Oort's clip line is commented out,
  `third_party/Oort/training/learner.py:284`); `fl_lr_check --clip-grad-norm/--shard-cap`; reference key `clip_grad_norm`. Default 1.0 for
  cifar oort/oort_star (recorded deviation, operator 10-10): tiny-shard NaN 3/6 -> 0/6, full-data accuracy unchanged within seeds.
- **FX-D127** Sim wait-K barrier probes only picks that owe a reply (`_sync_owes_nothing`); a stale-replied pick blocked max(D) +
  margin of wall per pass (PR28 feddance G1AS sims 0.28-0.71x real time, 5395 of 5415 s in that wait; vclock unchanged).
- **FX-D128** Example aggregators evaluate an eval round only on its committing pass (`_round_committed`); wait-K re-ran it per pass
  (feddance r80 x6), the first copy scoring the previous version.
- **FX-D129** A queued return answers only its end's latest dispatch (`_note_returned_version`, `_owes_newer_dispatch`,
  `_drop_superseded_returns`, both sync stacks): a superseded stale return freed the newer task's slot at round end (real Oort re-pick
  while gated, EV17) and hid it from the 90 s abandon.
- **FX-D130** A withheld delivery carried past K (FX-D122) commits as a withheld delivery (event, ready ts, slot held until commit,
  FX-D90); the fresh path re-gated it and emitted nothing (speech feddance EV16).
- **FX-D131** `harness_pool`: `ram_blocks` makes an idle pool wait for foreign RAM (only a leg larger than the node starts unchecked);
  `simulate_makespan` returns inf when nothing can start (was an endless loop on a GPU-count hiccup: B5 hung 3.5 h).
- **FX-D132** `parity_ladder` floors pair only within one code state (`code_key.txt` per pool; else the batch stamp) and with the
  regraded pool's own real; Oort-family `*s` cells (`{phase}_{trace}_{b}_grade`) were never found, so no Oort floor ever applied.
  `readiness_score.py` scores baseline x dataset x scenario cells from graded pools.
- **FX-D133** Early exit: `harness_pool` runs `event_invariants.doomed()` (prefix-safe EV6/8/9/11-backwards/14-19) on each live leg
  every `--doom-min` (5) min; a FAIL kills the leg + its pair partner (DOOMED.txt; P11/P4 exempt). PR28 replay, 245 legs cut at 10-75%: 0 false positives; G1 speech feddance EV16 caught at 25%.
- **FX-D134** G0T runs the Oort family at its K=10 G0U shape as `G0T_*s` (FX-L63; G0U_syn_50s floors map onto it); `G0TC` = real
  replicate per G0T cell, paired by `parity_ladder` as G0UC.
- **FX-D136** `harness_pool` RAM ledger: each started leg takes a node-wide `ram_*` lease holding its GB, and a start needs `min(MemAvailable, MemTotal - sum(leases))`
  (PR29 C3/B1 reals hit 504/504 GB again: pools probed before neighbours' footprints ramped). `readiness_score`: a later pool's grade of a pair supersedes an earlier one.
  EV16 never flags a sync delivery whose closing round is not logged yet (prefix kill false positive, speech feddance G0U syn_50).
- **FX-D135** Speech P7/P7o/TS dropped (FX-N86 option c, operator 10-10): tiny_cpu speech compute 5-46 s > D 1.25-4.75 s made the harness
  invalid; speech streaming is G0T only; `readiness_score` marks speech stream_cpu n/a (76 cells).
- **FX-D137** Async sim keeps buffered ends' slots on commit-free returns (abandon wake, no committable pop): left RECVD, the selector's
  FX-D24 release freed them (cifar fedbuff mobiperf picked 3 where real picked 1). `sim_keep_slots_on_wake=false` reverts.
- **FX-D138** Sync sim barriers (`_sim_barrier_recv`, syncfl + Oort) wait up to the task timeout for picks whose latest dispatch is unanswered,
  and `_sim_sync_wait` never jumps past one (a stub overrun past D read as absent; wait-K jumped to the next flip). `sim_barrier_awaits_picks=false` reverts.
- **FX-D139** Checker: `overlap_factor`, `total_commits`, `terminal_state` time SKIP below min(10, rounds) stall-free rounds, as `throughput`.
- **FX-D140** Resource ledger: every pool writes `resources.jsonl` (leg start/end, 30 s node + per-leg PSS by role and GPU memory); DONE
  lines show RAM peak/estimate, `RAM_OVER`, `LEAK?`; the next estimate is >= 1.15 x the measured peak; `resource_report.py [--timeline]`.
- **FX-D124** Sim sync drain stamps a delivered update's speed (was 0) and anchors U6 lag at the round close from its delivery.
- **FX-D114** A sim sync version waiting for K also wakes on the next availability change (real's poll tops up).
- **FX-D113** Sync sim barrier carries over-quota updates to the next barrier, as real's rxq does (dropping them
  deadlocked feddance once a new end made the recv timeout unbounded).
- **FX-D112** A stale-dropped Oort return records its version too; else FX-D109 awaits it forever (N86 T3 speech sim stalls).
- **FX-D111** Aggregators drop an update with NaN/inf weights or utility (failed task: not aggregated, not counted to K, not
  scored; `update_rejected` telemetry); `rejectNonfiniteUpdates=false` reverts.
- **FX-D110** Speech stub legs run one real local step (`harness_stub_max_steps`); the stub's span is `stub_compute_s`.
- **FX-D109** Oort awaits a re-dispatched end whose older stale return sits in the cleanup queue (FX-D66 deadlocked real).
- **FX-D108** Sim starts each recipient after its measured fan-out delivery lag (`SIM_WALL_SEND_TS`; EV3 subtracts
  `sim_delivery_lag_s`); `simChargeDeliveryLag=false` reverts.
- **FX-D107** `yogi_normalize_first`: first YoGi step eta·g/(|g|+tau) (cifar oort/oort_star; recorded deviation).
- **FX-D106** Oort-family screens at K=10 (`harness_pool.OORT_SHAPE`, `G0U_OORT`; speech mobiperf GPU left to T3 at n=165);
  exploration decays once per round, FX-N37 top-ups included; speech mobiperf Oort cells dropped (n=165 > 100 partitions).
- **FX-D105** Oort audit vs `third_party/Oort@05a3aa1`: getTopK (exploitLen, decay-then-size, cut-off pool, size-weighted explore,
  exploration off once all explored), pacer on returned exploits, preferred duration over all measured arms, dropped stragglers
  score the version's mean utility, equal-weight FedAvg (also FedDance), client SGD momentum 0.9 / wd 5e-4 (also REFL, FedDance);
  YoGi and FedScale utility already matched. Deviations: no a-priori size/speed before first contact; stragglers run, then drop.

## Open questions (operator)

- REFL `stale_update`: code -1 (unbounded, run_exps.sh:84) replaced the earlier operator choice 5. Confirm.
