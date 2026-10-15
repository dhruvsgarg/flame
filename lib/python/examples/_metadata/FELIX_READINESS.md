# Felix readiness — backprop FL (async_cifar10, google_speech)

Read [ROBUST_FL_READINESS.md](ROBUST_FL_READINESS.md) and [PARITY_READINESS.md](PARITY_READINESS.md) first; their rules, L and T apply
here unrepeated. **Focus (operator 10-12/13): Felix-wide parity to 76 / 76, then convergence parity (FX-N93).** Order: Next steps →
Scoreboard (now, and expected after the next run) → Decisions → reference sections.

**At a glance (10-14 night):** parity **76 / 76 on mixed commits**; PR35 one code key **47 / 52 tested** (stream 24 untested) · accuracy
**6 / 12** · convergence board **0 / 12** · next (new context): implement FX-N97 → FX-N96 → FX-N98 (Active build: PR36 prerequisites),
then PR36 full re-screen on one code key, then PR37 3 h convergence pairs → sim-only experiments (FX-N12).

## Next steps (persistent queue — top item is next)

Work rule: PARITY C10, C17, C18. Test levels: pytest (~3 min) · CPU tiers T1-T4/TS · GPU ≤ 30-min screens (G0U/G0T/G0UC) · long
(G1*/G2). Unblock map: FX-N97 + FX-N96 + FX-N98 → PR36 (76/76 on one code key) → PR37 (FX-N93.2/.3) → FX-N12.

- **FX-N97 · AVL_EVAL is unavailable for every baseline but Felix · todo (correctness, first; operator 10-14).** Design + checks:
  Active build "PR36 prerequisites" §A. *Exit:* EV check green on mobiperf_3st + syn_*_3st legs of all six; baselines never train on,
  dispatch to, or deliver from an AVL_EVAL trainer; Felix evals use AVL_EVAL trainers.
- **FX-N96 · 3-state synthetic traces, new default · todo (operator 10-14).** Active build §B. *Exit:* `syn_{10,20,50}_3st` generated,
  self-checked and documented; every run defaults to them; 2-state syn traces DEPRECATED (delete once PR36 grades green on 3st).
- **FX-N98 · Sim GPU scheduling + no-overrun rule · todo (operator 10-14).** Active build §C. *Exit:* overrun share ≤ 2% per leg in real and
  sim (PL19); speech G1AS felix sim ≥ 1× real (target ≥ 2× on 4 GPUs) with accuracy within noise of PR35 A1.
- **PR36 · full re-screen on one code key · blocked: FX-N96/N97/N98** (PR35b skipped, operator 10-14). G0U/G0UC + T3/T3C (syn_*_3st,
  mobiperf_3st), G0T/G0TC both datasets, G1AS six × both, a third real on the 4 PR35 draw cells (Scoreboard). Size with `--plan`
  (~14-16 h). Commit `FX-D163/D164` landed with the PR35 analysis. *Exit:* `readiness_score.py` on one code key = 76/76; G1AS on the board.
- **PR37 · 3 h convergence parity (FX-N93.2/.3) · blocked: PR36.** Per dataset, three cells × (real, real replicate, sim), 3 h each:
  syn_0 (baseline), syn_50_3st (unavailability), stream_lin syn_0 (streaming); felix in every cell plus one sync baseline (proposal:
  oort_star; confirm at `--plan`). Deterministic kernels. *Decision rule:* all cells inside the real↔real band on the convergence board's
  C and U criteria ⇒ paper experiments run sim-only (FX-N12, operator 10-14); a red ⇒ root on short runs first (C18).
- **PL-LOG · harness_pool contention messages · todo (apply after PR36 grades; changes the code key, so not between PR35 and PR36).**
  `harness_pool.py` BLOCKED reasons say "N foreign-busy" even when our own legs hold the slots. Rename to `resources-busy` when the
  block is this pool's own running legs. Re-print `LOAD foreign` (CPU/GPU contention) to the terminal every 10 min while any foreign
  load exists (other users or pools, incl. dgarg39's own), not only on change. Each line lists, per foreign owner (user + pool/command,
  e.g. `pbehnam3 run_pr30: 2 GPUs, 48 CPUs`), the GPUs/CPUs it holds, so we know whom to contact to speed up our runs.
- **FX-N94 · tiny_cpu first-contact sim delivery lag · todo.** `smoke_fxd160_0010`: first-contact trainers measured 1.1-1.7 s FX-D108 lag
  (real ~0.03 s), pushing sim's clock 1.7 s ahead by round 3; GPU legs stay ≤ 0.5 s. Check whether the trainer is still initializing when
  its first task arrives in sim. *Exit:* first-contact lag ≈ later tasks' on tiny_cpu.
- **FX-N95 · speech sim round close → train start ~0.1 s short · todo.** PR35 gs G0U feddance syn_50 r1→r2: real close→start 0.30 s,
  sim 0.20 (dispatch 0.09 + measured lag 0.11); arrival→close real 0.165, sim 0.128 (its two reals agree to 0.002). Not on stub: gs T3 oort
  syn_20s sim drifts −2.1 s by r10, its real replicate −1.8 s (lag order is a draw). *Exit:* mean sim−real per-round offset inside real↔real.
- **FX-N85 · Close PR27/PR28 roots · wip.** Left:
  - (b) Cap cifar stub local steps for the Oort family (`harness_stub_max_steps=1` in `harness_pool.shaped()`, as speech FX-D110):
    11/177 real stub tasks overran D under pool load and 8-sample stub losses blow up utility; PL11 check, rerun T3 cifar oort syn_50.
  - (c) A5 builder: apply FX-D125's observed-window rule to `state_timeline_agreement` (cifar T3 oort_star syn_50 0.9135).
  - *Exit:* every listed cell green or floor-gated.
- **FX-N92 · Felix insights: telemetry-backed mechanisms for the paper (operator 10-12) · wip.** Model, catalog, telemetry (FX-D153) and
  `insights.py` in [FELIX_INSIGHTS.md](FELIX_INSIGHTS.md); its Next steps IN1, IN2 (PR31 pools ran clean 10-13; findings not yet written), IN4 (long legs).
  *Exit:* FELIX_INSIGHTS findings for every catalog row, real↔sim agreed.
- **FX-N90 · FedDance coverage under over-commit · todo.** Literal Alg. 1 l.3 scores a picked-but-never-aggregated device I = 0 forever
  (F3 syn_0: 104/300 at U = 0, 199 vs 281 committers, rounds to 40% +17%). `straggler_score: mean` (FedScale: REFL aggregator.py:611-616)
  restores coverage but re-biases to slow ends (in-process: 34% slowest-quartile picks, 18.2 vs 15.5 s/round): A_m rewards rare engagement,
  and one-in-flight makes slow ends rare. *Exit:* a cited rule with coverage ≈ random and no speed bias, F3 rerun; else record as FedDance's own.
- **FX-N91 · EuroSys plot debts (no runs now, operator 10-12) · todo.** In order:
  - (a) Controls: G1X cifar syn_0 `fedavg` random sim 20 h (~25 min wall): plateau ≈ 64% → shared config, else FedDance's coverage;
    G1X speech mobiperf `felix` sim (n=100, 70%; ~5 h). `$PY scripts/harness_pool.py --tier G1X --datasets <ds> --baselines <b> --traces <t>`.
  - (b) A `straggler_score: mean` arm shows what literal Alg. 1 l.3 costs (reviewer question; FX-N90).
  - (c) REFL label: operator calls REFL an original baseline, notebook labels it `REFL*` (logs `refl_n300_oracular_*_prob0.9`); curves
    scaled ×1.06 (`accuracy_adjustment_factor`; raw syn_0 max 64.6%). Decide label and factor; ratios use REFL.
  - (d) Hard-coded MobiPerf e2e table vs logs: OORT* 15.6 h (table) vs 9.45 h (curve's first 70%); OORT+Async* 15.9 vs 14.97; REFL
    mobiperf has no log anywhere (table row only).
  - (e) Speech e2e: May-2025 rows have no logs/configs on any node; comm unit per speech message unknown (FedDance uses 13.5 = 29.0/2.15 MB).
  - (f) Comm to 70% scales whole-log counts by time share; exact = count `[MSG_ARRIVAL]`/send lines up to the 70% timestamp.
- **FX-N84 · Trainer memory audit (GPU + host RAM) · wip.** Host PSS/trainer (n=10 real): cifar 1195 → 1052 MB, speech 1318 → 1160
  MB via lazy mlflow import (81 MB) and own-satellite coords (49 MB); `TRAINER_RAM_GB` lowered to match. Left per trainer: `import
  torch` ~280 MB anon, cuDNN/cuBLAS host ~390 MB, dynamo 74 MB; GPU ~0.37 GB context, speech idle ~0.6 GB (cache already released).
  FluxTune's O1-O5 don't apply (no frozen params/functorch; trainer malloc_trim, CUDA env knobs measured ~0). Aggregator (1-8 GB/leg,
  speech creeps): speech test set kept int16 (336 vs 672 MiB, bit-identical eval), `malloc_trim` after each eval returns 0.4-0.9 GB
  in ≤ 62 ms (`FLAME_AGG_MALLOC_TRIM=0` reverts). Fork-server sharing of imports saves ~280 MB/trainer more but PR35 is
  GPU/CPU-lease-bound (RAM-unbounded plan = same 10.8 h): defer. Next: trainers per A40 (FX-T36), which sets GPUs per leg.
  *Exit:* footprint table, fixes, new caps.

*Short-run queue (fix + verify with pytest, stored telemetry, CPU tiers or GPU screens):*
- **FX-N9 · Unavailability, all six × syn_50 / mobiperf · wip.** PR35 key: syn_50 10/12 (draw cells), mobiperf 10/10; left: FX-N96/N97 (3-state), PR36.
  *Exit:* both columns green on one code state.
- **FX-N13 · Streaming, both datasets · wip.** stream columns 12/12 on mixed commits (PR34); left: PR36, G0To (oracle arms), ST6.
  *Exit:* EV19 + INV/EXACT green both datasets; ST6 graded.
- **FX-N30 `[S]` · Speech streaming on GPU screens · wip.** All six G0T green (PR31 A2/B3, PR34 B1); left: G0To. *Exit:* speech G0T/G0To lin + events × syn_0/syn_50 EV19 + INV/EXACT green.
- **FX-N74 `[C]` · Every baseline reaches target on full data · wip: 5 / 12.** Accuracy table below. *Exit:* each cell at target, or
  audited faithful and named round-bound.
- **FX-N76 `[C][S]` · C13 real-cost audit · wip.** weights_to_ram matches. Open: speech felix sim gpu_compute 3.7 vs real 1.9 s (sim GPU
  contention; off the clock while < D); speech `trainer_speed_identity` DIST (V2). *Exit:* speed identity + phase DIST green both datasets.
- **FX-N33 `[C][S]` · Aggregator abort at interpreter exit · todo.** `terminate called without an active exception` after channel leave
  (C++ static teardown); frequent on oort T1 syn_50 sim, seen on F3 fedavg sim (results intact); 4/4 clean under gdb. *Next:* gdb via
  `FLAME_AGG_CMD_PREFIX` (FX-D102) = `/coc/scratch/dgarg/gdb_env/bin/gdb -q -batch -ex run -ex 'thread apply all bt' --args`.
  *Exit:* root or a clean join.
- **FX-N42 `[S]` · Parity ladder · wip (PARITY Q2-Q6).** *Exit:* a run graded per cell on both axes, DIST on ≥ 3-leg floors.
- **FX-N10 · google_speech on the launcher · wip.** *Next:* S4 cleanup. *Exit:* all six graded on speech, CPU + GPU.
- **FX-N22 · Fast parallel harness · wip.** T3 legs use p95 ≤ 2.7 of 44 CPUs, so pools starve each other (PL12). *Next:* ISO
  oort,oort_star solo `--max-parallel 1` at cpt 1 vs packed at 0.5 and 0.25; `harness_iso_compare.py`. EQUIVALENT ⇒ T3 default cpt 0.5,
  GPU 0.1. *Exit:* ISO P6 EQUIVALENT at cpt ≤ 0.5; T2 both datasets < 25 min. Absorbs FX-N8.
- **FX-N2 · Parent S2 pipeline for async_cifar10 · todo.** *Exit:* stored Jun pairs regrade within floor.

*After 76/76 — convergence and accuracy (operator 10-13: the screens are 4-30 min mechanism parity; accuracy is next):*
- **FX-N93 · Convergence parity at the reference config · todo (starts at 76/76).** Screens prove per-cycle mechanisms; the paper needs the
  sim's accuracy-vs-time (ROBUST exit criterion 2, R6). One C19 block per step, in order:
  1. *Accuracy correctness, sim-only (cheap):* G1AS six × both datasets on the 76/76 commit; each reaches target or is audited round-bound
     (FX-N74; cifar refl/oort/feddance never reached it on source-faithful configs; speech open question below).
  2. *Real↔sim convergence pairs (was PR21):* G1A syn_0 at reference n, deterministic kernels, felix + fedbuff + one sync stack per dataset,
     plus a real replicate; grade accuracy at matched rounds and time-to-target inside the real↔real band. Add that rung if `convergence*`
     can't grade curve shape (C7).
  3. *Unavailability over hours:* G1U syn_50 + mobiperf at reference n, 3 h (compounding clock): felix/fedbuff, then the sync four.
  4. *Sim speed (FX-N22 P8, S6 unblocks at a green board):* speech sim is 0.6-1.1× real at n=50; reach ≥ 5× at reference n before sweeps.
  5. *Then FX-N12:* paper experiments sim-only: every baseline re-run on the final commit (EuroSys decision), streaming TTA (ST6).
  Graded on the Scoreboard's convergence board (A/C/U/S columns).
  *Exit:* per baseline × dataset, sim accuracy curve inside the real↔real band at reference n on syn_0 and syn_50; target reached or named
  round-bound. Absorbs FX-N11 (speech GPU parity at reference n). **FX-N12 · paper experiments sim-only · blocked: FX-N93, FX-N13, P8.**

---

## Scoreboard (rewrite after every run, PARITY C5)

**Parity: 76 / 76 on mixed commits (PR28-PR34); one code key: PR35 `5e041d393b8a` 47 / 52 tested, stream 24 untested (PR36).**
`readiness_score.py` on PR35 B1/C1/C2/C3 after the FX-D164 regrade (10-14, jayne). INV/EXACT + EV, worst pair wins; 2 reals per cell, so
FX-D161 keeps any fail red. **Accuracy 6 / 12** (cifar felix, fedbuff, oort, oort_star; speech felix, fedbuff). ✅ green · ❌ red · – n/a ·
· untested.

| baseline · dataset | syn_0 | syn_20 | syn_50 | mobiperf | stream_lin | stream_eve | stream_cpu |
|---|---|---|---|---|---|---|---|
| felix · cifar | ✅ | ✅ | ✅ | ✅ | · | · | ✅ |
| felix · speech | ✅ | ✅ | ✅ | ✅ | · | · | – |
| fedbuff · cifar | ✅ | ✅ | ✅ | ✅ | · | · | ✅ |
| fedbuff · speech | ✅ | ✅ | ✅ | ✅ | · | · | – |
| refl · cifar | ✅ | ✅ | ✅ | ✅ | · | · | ✅ |
| refl · speech | ✅ | ✅ | ✅ | ✅ | · | · | – |
| oort · cifar | ✅ | ❌ | ✅ | ✅ | · | · | ✅ |
| oort · speech | ✅ | ❌ | ✅ | – | · | · | – |
| oort_star · cifar | ✅ | ❌ | ❌ | ✅ | · | · | ✅ |
| oort_star · speech | ✅ | ✅ | ✅ | – | · | · | – |
| feddance · cifar | ✅ | ✅ | ✅ | ✅ | · | · | ✅ |
| feddance · speech | ✅ | ✅ | ❌ | ✅ | · | · | – |

**Open reds → root** (PR35; `logical_diff.py`, per-round clocks):
- feddance · speech G0U syn_50 (trainers 31 vs 29, 2/2 reals): sim delivered send-gated updates at the flip, real 0.2-0.4 s later
  (r4 −0.44 s, r7 −0.27 s), utilities forked at sel #7. **Fixed, FX-D163:** local re-sim vs both PR35 reals has no selection or
  commit fork, K8 0.1% / 32 vs 32.
- oort · speech T3 syn_20s (K8 10-11%, S1): trainer 0377 finishes 0.3 s either side of the t=150 flip; its real replicate forks the same
  way (real↔real first fork at commit #5). Draw: needs a third real (FX-D161).
- oort · cifar T3 syn_20s (trainers 39 vs 37): 0385/0396 share D = 3.25 s and arrive 8 ms apart in real, reversed in sim. K-th-slot
  tie, FX-L72: third real.
- oort_star · cifar T3 syn_20s, G0U syn_50s: red against 1 of 2 reals, and the two reals differ (raw 688 vs 787 s). Third real.
- Cleared by FX-D164 (grader): fedbuff · cifar G0U syn_50 K8 11.2% → 1.1% (raw 1.1%; async gaps had been classified as stalls).
- FX-D163 local re-screen (18 sims vs PR35 reals: speech syn_50/syn_50s/mobiperf G0U + T3, cifar fedbuff G0U): EV 18/18 PASS, no new
  fork. Only the draw cells above stay red.

**Expected after PR36:** 76/76 on one code key, with syn_20/syn_50 = 3-state and FX-N97 enforced. Draw cells need their third real
(L13); mobiperf cells may move, since baselines lose EVAL devices.

**Cross-cutting capabilities**

| capability | status | item |
|---|---|---|
| checker catches injected sim bugs (P11a-c) | ✅ 6/6 CAUGHT both datasets (PR28 C3) | FX-D125 kept sensitivity |
| fail fast; ladder runner + offline re-gate; early DOOMED kill | ✅ | FX-D22, FX-D133, FX-N42 |
| GPU join/warm-up at n=100; sync wait-K | ✅ joins 44-99 s; EV7 green | FX-D19-D21 |
| real async ingest < 2 s | ✅ G0U queue_wait max 0.37 s; U6 felix ≤ fedbuff both datasets | FX-D16 |
| profiled sim charges per dataset/platform | ✅ re-derived on FX-D141/D142 code, T3 syn_0 32/32 | FX-D143 |
| DIST tolerances from replicate floors | 🟡 n=2 floors gate regrade; none tightened | FX-D37, PARITY Q2 |
| real == sim with deterministic kernels | ✅ V2: losses bit-identical, commits identical to r41-48 cifar, all rounds speech; G0U/G0T screens use them | FX-D144/D145/D149/D158 |
| streaming / oracle experiment | 🟡 cifar EV19 green; oracle advantage ungraded | FX-N13 |
| sim speedup vs real | 🟡 cifar 4.5-7.4× at n=50, ~18× at n=300 G1XQ; 🔴 speech 0.6-1.1× at n=50, 11.2× at n=100 sim-only | FX-N22 P8 |
| paper experiments sim-only | ⬚ | FX-N12 |

**Accuracy table (FX-N74; `scripts/accuracy_table.py --runs <dirs>`).** Parity target cifar 50%, speech 60% on full data, syn_0, reference
n, the leg's own clock. G1A = full data (runs 16-18); G2 streams data 0 → 100% over 3 h (lower bound). `-` = no eval in window.
Sync-four numbers are G1AS on source-faithful configs (FX-D100/D104/D105); older G1A rows were void and are dropped.

| dataset · baseline | leg | real @30 / @60 / @90 · max | sim same | note |
|---|---|---|---|---|
| cifar · felix | G1A | 36.2 / 51.8 / 56.9 · 57.0 | 40.6 / 50.6 / 53.7 · 56.3 | **target at 57 / 60 min** |
| cifar · felix, fedbuff | G1AS PR35 A2 (key `5e041d393b8a`) | — | felix 40.0 / 51.1 / 57.0 · 57.5 · fedbuff 39.4 / 46.4 / 54.9 · 56.2 | **sim target at 52 / 65 min**; no regression vs runs 18/21 |
| cifar · oort, oort_star | G1AS PR35 A2 | — | oort 44.3 / 51.6 / 53.7 · 54.5 · oort_star 36.1 / 51.6 / 52.0 · 52.0 | **sim target at 32 / 58 min** |
| cifar · refl, feddance | G1AS PR35 A2 | — | refl 20.0 / 25.5 / 26.9 · 29.5 · feddance 19.7 / 27.4 / 35.6 · 35.6 (first uncut) | below 50%: audit or round-bound (FX-N74) |
| speech · felix | G1A SGD (run 18) | 41.5 / 58.8 / 67.5 · 67.5 | 46.0 / 56.0 / 68.1 · 68.1 | **target at 72 min both** |
| speech · fedbuff | G1A SGD (run 18) | 40.8 / 54.6 / 59.1 · 60.2 | 42.5 / 55.9 / 64.0 · 64.0 | **target at 82 / 88 min** |
| speech · felix, fedbuff | G1AS PR35 A1 | — | felix 44.1 / 59.2 / 66.2 · 67.9 · fedbuff 42.5 / 55.1 / 64.6 · 64.6 | **sim target at 65 / 87 min** |
| speech · oort_star, oort, refl, feddance | G1AS PR35 A1 | — | 13.3 · 16.0 · 4.0 · 7.7 max @90 (3-4 evals) | **round-bound (operator 10-10)**; PR28 G4 16.7 · 14.8 · 3.7 · 4.0 |
| speech · refl | G1AS SGD (runs 19, 21) | — | lr 0.005: 43.7 max; lr 0.05: 53.0 max | rising, below 60% |
| in-process `fl_lr_check` | reference k, full data | cifar fedbuff 53 @r1000 · oort 60 · feddance 45 · refl (lr 0.1) 43 | speech @r150: oort (eta 0.002) 25 · feddance 18 · refl (FedAvg) 17 | 10-08 |
| both · oort, feddance | G2 stream | cifar oort 37.7, feddance 16.6; speech oort **60.7 at 80 min**, feddance 21.5 | cifar oort 31.4, feddance 35.4; speech 57.2, 45.2 | lower bound |
| EuroSys config: feddance (70% target) | G1X sims, 20 h vclock, FX-D148/D150 (`fd_20261012_0503`) | — | cifar syn_0 64.4 · syn_10 64.2 · syn_50 65.2 @12.9 h · mobiperf 49.0; speech mobiperf 60 @9.85 h, 68.9 max | plateaus: FX-N90 |

**Convergence board (FX-N93; long runs beyond the 76 screens).** Cell = baseline × dataset at reference n, syn_0 unless noted; green when
all four columns pass. Thresholds are nominal until a real↔real replicate sets the band (L12).
- **A · accuracy:** G1AS sim reaches target (cifar 50%, speech 60%) within 90 min on its clock. Where the task can't (speech, streaming):
  round-bound, and real and sim must reach the same accuracy at the same time, whatever the length (60, 90, 120 min) (FX-N93.1).
- **C · curve parity:** G1A real + sim + real replicate, deterministic kernels: accuracy at matched rounds within max(2 pp, real↔real) and
  time-to-target sim/real within max(10%, real↔real) (FX-N93.2; time-to-target rung still to build, C7).
- **U · hours under unavailability:** syn_50_3st + mobiperf_3st, 3 h: INV/EXACT green and C's criteria (FX-N93.3, PR37).
- **S · sim speed:** sim ≥ 1× real wall at reference n on every cell, never slower (FX-N98).

| baseline · dataset | A | C | U | S |
|---|---|---|---|---|
| felix · cifar / speech | ✅ / ✅ | ⬚ / ⬚ (run 18 pre-76/76) | ⬚ / ⬚ | ✅ 4.7× / ✅ 1.09× (fix: FX-N98) |
| fedbuff · cifar / speech | ✅ / ✅ | ⬚ / ⬚ | ⬚ / ⬚ | ✅ / ✅ 1.5× |
| oort, oort_star · cifar | ✅ ✅ | ⬚ | ⬚ | ✅ |
| refl, feddance · cifar | ❌ ❌ (29.5, 35.6%) | ⬚ | ⬚ | ✅ |
| oort, oort_star, refl, feddance · speech | round-bound: needs C (same accuracy, same time) | ⬚ | ⬚ | ✅ |

**Convergence 0 / 12** (A 6/12 · C 0 · U 0 · S cifar 6/6; speech felix 1.09×, fedbuff 1.5×, sync four ≥ 13× at G1AS).

---

## Decisions (operator; standing until changed here)

- **Node (10-13):** runs on jayne GPUs 0,2-7 (GPU 1 out) or shepherd GPUs 0-3 (R4); score on the node holding every scored pool's runs (jayne).
  Shepherd = jayne hardware (EPYC 7513, A40, driver 575.57.08): charge profiles and floors are shared (T9).
- **EuroSys FedDance (10-11/12): FedDance's algorithm on our setup, apples to apples.**
  - *Common to all baselines:* model, Dirichlet α=0.1, n=300 cifar / 100 speech, K=10 aggregated (c=13 dispatched), the May-2025 shared
    trainer (1 epoch, bs 32, lr 0.04, plain SGD; `trainer/config_dir0.1_num300_*` @ `966797558`), traces, simulator, target 70% both
    datasets (`EUROSYS_TARGET`), time-to-target on the run's clock, comm = msgs × 1 MB (eval reply 0.01 MB).
  - *Per baseline (its paper):* selection, availability handling, aggregation. FedDance: V·I·A utility + MAB exploration (K_h 50, K 5,
    β 5, overcommit 1.3), own check-in predictor (no trace oracle), FedAvg equal weights, server lr 1.
  - *Not carried over:* FedDance's paper setup (LS=1, bs 24, lr 0.1, 1,000 clients, 45%/31% targets); its claim is relative
    round-to-accuracy in one shared harness, as it re-ran REFL in its own setup.
  - *Comparability:* only FedDance runs on the new sim code today; all baselines (FedDance again) re-run after parity. EuroSys curves stay
    the May-2025 real runs (`agg_logs/`).
  - *Plots:* time to 70% = first eval ≥ 70, else log-time fit past the run's end, capped 48 h (`curve_tta`); comm scaled to that time;
    FeLiX speedup vs REFL and FedDance (`speedup_refs`), comm ratio vs REFL only (`ref_strategy`).
  - Speech F5 used a fixed vclock budget from V3's sim speed; no F4 rerun (V2 covers real == sim).
- **Baselines (10-08, 10-10):** source-faithful values in `baseline_reference.yaml` (ROBUST L33-L37); cifar Oort/oort_star clip 1.0
  is a recorded deviation (FX-D126: tiny-shard NaN 3/6 → 0/6, full-data r300 48.9/54.5 → 48.9/49.9%; C20 voids older cifar Oort legs);
  speech Oort/refl/feddance accuracy recorded round-bound; debug every red on short runs before any long run.
- **FedScale baselines (10-12):** FX-D148 partial dispatch is correct behaviour for refl, oort, oort_star, feddance (FedScale `resampleClients`);
  REFL staleness unbounded (`stale_update: -1`): paper §5 "no maximum threshold is applied to staleness"; the 5-round cap is only the
  SAFA comparison (§5.2.2, `run_exps.sh` SAFA branch). Stored REFL runs already used -1 (reference applied last), so no C20 from it.
- **PR36 prerequisites (10-14):** skip PR35b. Implement FX-N97, FX-N96, FX-N98 in a new context first, then PR36 re-runs everything on
  one code key.
  - AVL_EVAL is eval-only and Felix-only; for every other baseline it equals UN_AVL (FX-N97).
  - 3-state syn traces (EVAL share from `mobiperf_3st_50`) become the default; 2-state syn traces are deprecated, then deleted (FX-N96).
  - At most 1-2% of tasks may overrun D, in real and sim; above that, fewer trainers per GPU (FX-N98).
  - Sim must never be slower than real.
- **Sim-only experiments (10-14):** after PR37's 3 h pairs (baseline, unavailability, streaming × both datasets) sit inside the real↔real
  band, paper experiments run sim-only.
- **Run scope (10-12):** the 76/76 must stand on one code state, so a batch reruns felix/fedbuff G0U (A4) beside the others. One code
  state = one `code_key` (FX-D162, 10-14): it may span batches (PR35 + PR36) while no run-affecting file changes.

**FedDance in the paper** (eval text; evidence `fd_20261012_0503` telemetry, FX-N90). FedDance climbs fast, then plateaus below 70% on
both datasets. Its utility favours fast, high-loss devices, so early rounds are short (12 s vs 15.5 s for random selection on CIFAR-10)
and accuracy rises steeply (≈40% within 2 h). But FedDance gives a selected device whose update was dropped as a straggler zero
statistical utility (Alg. 1, l. 3), and over-commit (13 picks for 10 updates) drops three devices every round. Within two hours a third
of the CIFAR-10 population (99-105 of 300) is zeroed and never contributes again; under MobiPerf, where few devices are online at once,
only 91 of 300 ever contribute and 20% of devices supply 96% of updates. With non-IID data (Dirichlet α = 0.1) the excluded devices'
classes never reach the model: CIFAR-10 plateaus at 64-65% on the synthetic traces and 49% on MobiPerf; Speech Commands shows the same
skew (55 of 100 devices ever contribute, 20% supply 84% of updates) and reaches 68.9% after 20 h. Extrapolated, FedDance needs > 48 h
to 70% everywhere except CIFAR-10 syn_50 (39 h).

---

## Scope and conventions

- **Correct, not just equal:** every baseline follows its reference and hits target on full data; root and fix every red per run.
- **Scope:** `felix`, `oort`, `oort_star`, `refl`, `feddance`, `fedbuff` (+ `*_oracle` arms, FX-N13; `fedavg` = FedDance's random arm,
  FX-D146). Out: `oracle`.
- **Traces:** `syn_0`, `syn_20`, `syn_50`, `mobiperf_3st`. `syn_10` = the old "syn_20" (10.8%, EuroSys'26; FX-D18).
- Levels (n=300, TRAIN/EVAL/UN %) flat 1 min-3 h: syn_10 89/0/11 · syn_20 80/0/20 · syn_50 50/0/50 (2-state; `_3st` targets in Active
  build §B) · mobiperf_2st
  10/0/90 · 3st_50 10/25/65 · 3st_75 10/10/80. Mobiperf trainable share rises to 18-22% by 6-24 h.
- **IDs:** `FX-N` next steps · `FX-L` lessons · `FX-T` tripwires · `FX-D` built · `PR#` run batches (PARITY). Shared work is `S#` in the parent.
- **Deprecated reference:** [PARITY.md](../async_cifar10/PARITY.md) §2-5 · [PARITY_CHECKER_README.md](../async_cifar10/scripts/parity/PARITY_CHECKER_README.md)
  · [UNAVAILABILITY_DESIGN.md](../async_cifar10/UNAVAILABILITY_DESIGN.md) · [EXPERIMENT_felix_streaming.md](../async_cifar10/docs/EXPERIMENT_felix_streaming.md)
  · [BASELINES.md](BASELINES.md).

## Active build — PR36 prerequisites (FX-N97, FX-N96, FX-N98; operator 10-14)

Implement in this order in a fresh context; delete each subsection when its item lands (its Built line replaces it).

**§A · AVL_EVAL semantics (FX-N97, correctness).**
- *Rule:* AVL_TRAIN may train and eval; AVL_EVAL may only eval; UN_AVL does nothing. Only Felix classifies trainers by state and sends
  eval tasks. For every other baseline AVL_EVAL ≡ UN_AVL in selection, dispatch and delivery. A train result is delivered only in
  AVL_TRAIN (all baselines, Felix included); an eval result in AVL_EVAL or AVL_TRAIN.
- *Suspected leak (verify on stored mobiperf_3st legs first):* unaware baselines (oort, fedbuff: `avail_select_filter` false) dispatch
  training to AVL_EVAL trainers. The trainer send gate holds only UN_AVL (`syncfl/trainer.py` `_send_weights`). Sim `compute_delivery_ts`
  and `trace.next_avail_after` count AVL_EVAL as deliverable. Net effect: baselines train and deliver on EVAL devices (too generous).
  Aware baselines filter EVAL at train selection (`get_curr_task_ineligible_trainers`) but can still deliver a train result during EVAL.
- *Touch points:*
  - availability core: `trace.next_avail_after` (task-aware), `client_availability.compute_delivery_ts`,
    `get_curr_task_ineligible_trainers`;
  - trainer: send gate + `_refresh_avl_state`;
  - selection plumbing: `channel.py` AVL_TRAIN→AVL_EVAL release, `selector/async_base.py`, the oort/asyncfl `top_aggregator` D.2 sites;
  - checker: `parity/ground_truth.py`, EV16.
  - Unaware baselines still pick without a filter (their algorithm). The picked EVAL device computes, and its upload is held until
    AVL_TRAIN (the send-gate model).
- *New check EV21 (real + sim, prefix-safe → DOOMED):*
  - an aware baseline never dispatches a train task to a trainer that is not AVL_TRAIN at dispatch;
  - no train result commits while its trainer is not AVL_TRAIN at delivery;
  - Felix on a 3-state trace sends > 0 eval tasks to AVL_EVAL trainers.
  Add as a PARITY pre-launch check.
- *Impact:* every mobiperf_3st leg so far (76/76 mobiperf column, EuroSys G1X mobiperf) gave baselines EVAL devices. Void them (C20)
  and re-run in PR36.

**§B · 3-state synthetic traces (FX-N96).** Method, written for the paper:
- *Source:* MobiPerf `mobiperf_3st_50`, 300 devices, 149 h. It equals `mobiperf_2st` with part of UN_AVL relabeled AVL_EVAL; the
  AVL_TRAIN timeline is identical (6742 periods, 15.7% of time).
  - State shares: TRAIN 15.7%, EVAL 34.0%, UN 50.3%. EVAL is 40.3% of non-train time.
  - Exits: TRAIN → EVAL 55% / UN 45%; EVAL → TRAIN 60% / UN 40%; UN → TRAIN 95% / EVAL 5%.
  - Dwell medians: TRAIN 614 s, EVAL 5666 s, UN 543 s.
- *Construction:* `gen_synthetic_trace.py --three-state <syn_X> --like mobiperf_3st_50` writes `syn_X_3st`.
  - Keep syn_X's AVL_TRAIN intervals exactly: training eligibility, churn time scale and the trace name (= non-trainable share) are unchanged.
  - Relabel each non-train window as one of three patterns, with MobiPerf's exit probabilities:

    | pattern | probability |
    |---|---|
    | all AVL_EVAL (TRAIN→EVAL→TRAIN) | 0.55 × 0.60 = 0.33 |
    | AVL_EVAL then UN_AVL (TRAIN→EVAL→UN→TRAIN) | 0.55 × 0.40 = 0.22 |
    | all UN_AVL | 0.45 |

  - The split point in EVAL-then-UN windows is one per-trace fraction, solved so EVAL = 40.3% of non-train time. Seeded per (trace, trainer).
  - Expected shares (TRAIN / EVAL / UN): syn_50 50 / 20 / 30; syn_20 80 / 8 / 12; syn_10 89 / 4.5 / 6.5.
- *Self-checks (pytest + generator output):*
  - TRAIN intervals equal the source's;
  - each state's share within ±1 pp of target at 149 h and in 0-3 h;
  - window-exit rates within ±5 pp of MobiPerf's;
  - deterministic for a seed, no repeated states, final UN_AVL marker kept.
- *Default switch:* new keys; `harness_pool` shapes/tiers, run scripts, `debug_run.sh` defaults and the Scoreboard syn_20/syn_50 columns
  move to `_3st` (grep every `syn_10`/`syn_20`/`syn_50` literal). Old 2-state keys are marked DEPRECATED in `synthetic_traces.yaml`,
  deleted once PR36 grades green on 3st.
- *Trainer check (CPU T3 pairs, felix/fedbuff/oort on syn_50_3st, ≤ 15 min):*
  - each trainer's `avail_change` telemetry equals the trace at the same trace times, all 3 states;
  - EV21 green;
  - an eval result delivers in AVL_EVAL, a train result waits for AVL_TRAIN;
  - real↔sim green.

**§C · Sim GPU scheduling and the no-overrun rule (FX-N98).**
- *Measured:*
  - speech G1AS felix sim (n=100, 4 GPUs) took 82.9 min of wall time for 90 min of simulated time (1.09×); fedbuff 1.5×.
  - Sim runs ~28 trainings at once (it never sleeps D), real ~3. Per-task compute: sim mean 11.5 s vs real 5.9 s (G1A 10-07).
  - Overruns of D: sim 10%, real 3.5%. Slot waits count as compute.
  - The aggregator waits 4536 of 4749 s at the sim barrier. cifar G1AS felix 4.7×; speech sync stacks fine.
- *Rule (PL19):* at most 2% of train tasks per leg overrun D, in real and sim. Above that, fewer trainers per GPU: real via more GPUs per
  leg or a lower `gpu_train_slots` (FX-T36), sim via K. Real speech G1A at 3.5% already breaks it.
- *Design (sim only, real unchanged):*
  - `simGpuTrainSlots` K per GPU;
  - waiting tasks take a slot in earliest-deadline order (deadline = vclock send + D, the earliest the barrier can need them);
  - sim charges slot-held time only (slot waiting is a sim artifact; in real it stays contention).
  - First a ≤ 15-min micro-benchmark of one speech task at K = 1-4 per GPU: pick the K with best throughput and per-task time ≤ real's.
- *Verify:*
  - speech G1AS felix sim ≥ 1× (target ≥ 2× on 4 GPUs);
  - per-task compute KS vs real G1A, and overruns ≤ 2%;
  - accuracy within noise of PR35 A1 (67.9%);
  - speech G0U felix/fedbuff sims green vs PR35 reals;
  - cifar not slower.

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
| G0T / G0TC / G0To | streaming GPU screen × {lin, events} × {syn_0, syn_50} / real replicate / oracle | cifar ~2 h |
| G1 / G2 | 90 min reference config: felix+fedbuff / other four | ~2.5 h per pair |
| G1S | felix + fedbuff, reference n, 45 min, 4 GPUs (fix confirm) | speech ~100 min |
| G1A / G1AS | accuracy: syn_0, full data, eval every 20 rounds, 90 min / sims only | ~2.5 h / 25 min |
| G1U / G1L | felix+fedbuff syn_50 at reference n / felix syn_0 7500 s + replicate | ~5 h each |
| G1XS / G1XQ / G1XP / G1X / G1XF(S) | EuroSys config (FX-D152) | 15 min / 2 h vclock / 60 min / 20 h vclock / 6000-3000 s |
| G2S / G2C | one G2 cell replicated sim-only / real-only (chaotic-selector floor) | 8 / 97 min |
| N62 / N64 | unaware oort syn_50 3 h CPU / refl speech health (EV18) | ~5 h / 45 min |

- `--changed <ref>` picks affected baselines; `--exclude-phases` drops phase ids; `--gpu-ids` pins GPUs.
- Pool skips busy GPUs/cores between legs, scaling with foreign load (`LOAD` in `pool.log`).
- Run dirs live in `async_cifar10/experiments/`, pools in `experiments/` of the node that ran them (PR28-PR30, FD, PR32: jayne; PR31 ran on
  shepherd, runs + pools copied to jayne 10-13).
- Tests: `tests/harness/test_{slot_isolation,harness_pool,fl_data}.py`, `tests/launch/test_debug_run_dataset_profile.py`.

## Active build — streaming time-to-accuracy (FX-N12, FX-N13)

- **Experiment:** six baselines × two datasets × four traces × two stream modes; metric time-to-target.
- **Claim:** Felix's eval selector finds clients whose data just grew; others don't.
- **linear:** each trainer grows 10% → 100% of its shard by the horizon.
- **events:** starts 10%; nine 10% chunks land at seeded uniform times, unsynchronized.
- Stream clock = trace time; horizon 6 days (paper) or leg span (screens); chunks ignore availability.
- **Built (FX-D47):** one `stream_schedule.py` for trainer, oracle, replay and EV19; tiers TS/TSo, G0T/G0To.
- **Left:** ST5 speech on GPU screens (FX-N30) · ST6 sim-only sweep + oracle advantage (needs P8).

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
- **FX-L67** Size a selector comparison so eligible >> picks: `fd_sanity` "selector had a choice" before reading any arm difference (F3/S4 n=50 mobiperf: 202/218 picks took every eligible end).
- **FX-L68** A selector learns only from updates the server used; feedback from dropped stale returns re-picks stragglers (FX-D150).
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
- **FX-L69** List both sides' long rounds before rooting a stall-count residual: twins straddling a hard cut are classification (FX-D157).
- **FX-L70** Before trusting a sync real fork, check real's own timeouts (EV20): real is reference only once admissible (R2, FX-D155).
- **FX-L71** Equal eligible sets + scores < 2% apart = numerics: equality legs run deterministic kernels before a mechanism hunt (FX-D158).
- **FX-L72** Co-released withheld updates at a shared flip land in a run-dependent order (29 MB uploads finish 0.2-0.4 s after release);
  the K-th slot is a draw: grade such cells on n ≥ 3 real floors, never a tie-break tweak.
- **FX-L73** Linear streams fork on ms stream-clock jitter at a sample boundary: read sim−real stream clock pre-fork against real↔real
  before calling it a stream mechanism (PR34 lin: −0.10 vs −0.06…+0.03 s).
- **FX-L75** If sim runs a constant early at every flip-closed round, look for an uncharged release cost: diff sim `withheld_delivery` times
  against real's first arrival after the flip (FX-D163: 0 vs 0.36 s).
- **FX-L76** Before rooting a sim fork at a flip boundary, diff the real replicate's clock: speech oort T3 reals drifted as far as sim.
- **FX-L74** "Sim outside all n reals" is no refute: under exchangeability it holds with probability 2/(n+1) (50% at n=3); vote per real (FX-D161).

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
- **FX-T45** Don't snapshot node state once per pool (GPU list) or let a probe pass on 0 cells: PR29b B1 idled 5.5 h and read PASS (FX-D141).
- **FX-T46** Don't leave first-use device init (CUDA context, lazy copies) on the round path: real pays it on the trace clock, sim doesn't (FX-D142).
- **FX-T47** Don't trust a sim charge profile older than a dispatch-path change: compare `_meta.commit` with HEAD, re-run `CHG` + T3 syn_0 (FX-D143).
- **FX-T41** Don't exempt an idle pool from the RAM check: PR28 G5 started 398 GB legs at 285 GB free and OOM-killed three legs (FX-D131).
- **FX-T48** Don't `--regrade` a pool copied without its run dirs: missing runs rewrite its grades as EV0 reds (PR31 copies on jayne, 10-13).
- **FX-T49** Don't classify a pair's rounds with per-side thresholds or drop one side's zero advances: twin rounds split (FX-D159).

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
- **FX-D156** Sync sim barrier lands a due stale withheld return past K this pass, before the next selection (it can't count to K);
  carried, its feedback missed that selection (F1 cifar feddance eve). `simIngestStalePastK=false` reverts.

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
- **FX-D155** Real wait-K recv returns at once when an awaited pick is past its timeout but unabandoned (`_real_overdue_pick`); the
  deadline filter dropped it, so picks dispatched ms apart were abandoned ~60-90 s late. EV20 grades it. `realRecvReturnsOverdue=false` reverts.

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
- **FX-D141** `harness_pool`: GPU list refreshed every 30 s (a failed `nvidia-smi` returns [] and the start-time snapshot starved PR29b B1
  5.5 h on an idle node); ready legs blocked > 10 min log `BLOCKED <jid>: <reason>`; history merged under a lock on save (concurrent pools
  erased each other's RAM peaks; 53 backfilled from ledgers); RAM formula per (dataset, GPU) from ledger maxima (speech GPU 1.55, CPU 1.05
  GB/trainer). Batch probe: 0 graded cells = `EMPTY`, skipped legs = `PARTIAL` (PR29b B1 printed PASS).
- **FX-D142** Example aggregators build the eval model copy at `initialize` (before the trace origin): the lazy first-eval copy to the GPU
  created the CUDA context (0.8 s) between round 1's commit and round 2's dispatch, on real's trace clock only; G0T stream chunks then
  landed on different sides of a trainer's train time (0374: 55 vs 27 samples), forking Oort utilities and picks (cifar oort eve syn_0s,
  oort_star lin syn_50s, PR30 B1).
- **FX-D148** `syncPartialDispatch` (feddance + its fedavg arm via syncfl `_sync_wait_k_plan`; refl, oort, oort_star via Oort `_wait_k_need`): a wait-K version dispatches to whoever is eligible, topping up later, as FedScale
  `resampleClients` takes all online when < c; it waited for c eligible (F3 n=50 mobiperf: no round closed after 2750 s). Off = byte-identical.
- **FX-D149** `trainerDeterministic` (cuDNN deterministic kernels; G1XP, G1XS): default kernels drift 1e-6 per epoch, chaotic on tiny shards
  (F4 losses 1e-3 by r5, commit fork r35); on: 0 drift (single-process check).
- **FX-D150** FedDance learns only from aggregated updates (`feedback: aggregated`, `baseline_reference.yaml`; `all` reverts): a dropped
  return repeats its last accuracy (zero increment), never-selected devices take last round's mean factors, selected-never-aggregated keep
  I = 0 (Alg. 1 l.3-15; FedScale never runs stragglers; `straggler_score: mean` = FX-N90). Stale feedback had locked F3 syn_0 onto stragglers (0550: 121 picks, 0 commits).
- **FX-D151** Real sync U6 lag anchors on absolute `WALL_SEND_TS`: dispatch-relative durations under-read FX-D148 top-up rounds (r11: real 4 s, sim 89 s).
- **FX-D152** EuroSys FedDance tooling: `scripts/run_fd.py` (`--plan`, `--only`), tiers G1XS/G1XQ/G1XP/G1X/G1XF(S) (`EUROSYS_*`,
  `FEDDANCE_PAPER_HP`), `scripts/fd_sanity.py` (per-run invariants incl. "selector had a choice"; `--pair`: commits and losses equal up to
  the first fork), `scripts/eurosys_export.py`; `all_plots.ipynb` loads `eurosys26_expts/feddance_logs/*.json` (`curve_tta`).
- **FX-D153** Insights telemetry (FELIX_INSIGHTS §4, read by no checker): async `selection` in-flight split by task + train slots held by
  eval (`async_base._occupancy`); `wire` event per message, both roles (`channel._emit_wire`); felix `explore_ids`/`exploit_ids`; metrics in
  `scripts/insights.py`.
- **FX-D154** Pool gate stamps the smoke per dataset (a combo reuses them) and smokes only unstamped ones; its 15-min timeout starts at
  the smoke's first leg START (`run_smoke`; 60 min to get leases) and it honours `--gpu-ids` (PR31 A4/A5 aborted queued behind siblings).
  `logical_diff.py` drops no-op selections (FX-D98; one shifted PR31 fedbuff's first fork from #22 to #6).
- **FX-D157** Stall-free timing rungs match stall-episode counts per pair (`_pair_stall_sets`): the side with fewer promotes its longest
  slow-band rounds; twins at 70 vs 74 s flipped the 72 s cut and moved every stall-free mean ~20%. K3s keeps the raw counts.
- **FX-D158** G0U/G0T/G0UC/G0TC legs run `trainerDeterministic` (`harness_pool.DETERMINISTIC`; PL15): default kernels forked FedDance
  near-ties on 1% loss differences (speech G0U syn_50).
- **FX-D159** K8/U2 keep zero advances (L32), classify stalls with one pooled slow band per pair and pair-matched counts (FX-D157);
  floors take every same-code real replicate pair (max). `PAIR_SLOW_BAND=False` reverts the band. PR33 fedbuff·cifar syn_50 13% → 1.3%.
- **FX-D160** Sim reads the data stream at modeled train start (send + FX-D108 lag), as real reads it after its download; EV19 accepts
  both. `simStreamAtTrainStart=false` reverts. Real−sim stream clock was +0.1-0.6 s on every clean sync pair (speech: 13/67 tasks forked).
- **FX-D161** `parity_ladder` regrade grades each sim against every same-code real replicate (`<cell>.rep<i>.json`, L13); with ≥ 3 reals a
  check red on a minority is a draw (`why`: `draw S8 terminal_state 1/3`), on a majority red; n < 3 keeps any fail red (L12).
- **FX-D162** `harness_pool.code_key()` hashes tracked run-affecting files only (`NOT_RUN_AFFECTING`: docs, tests, graders, `run_*.py`): a
  doc/grader commit keeps gate stamps and floor pairing (L18).
- **FX-D163** Sim delivers a send-gated update at flip + profiled `gate_release` (trainer flip→send + upload, per stack; `profile_felix_charges.py
  --only gate_release` from syn_50/mobiperf reals; speech 0.11-0.59 s, cifar 0.02-0.04 s); EV16 expects it. `simGateReleaseSeconds=0` reverts.
- **FX-D164** Stall stripping (K8, U2, K2, K4, overlap) applies to sync legs only; async legs grade raw time (`ASYNC_NO_STALLS=False` reverts).
  A/B over 49 PR32-PR35 async pairs: 2 red → green, 0 green → red.
- **FX-D147** Leak check (`ram_growth_gb`) starts after a leg's process count peaks: a short sim's spawn ramp read as LEAK? (S3 +113 GB).
- **FX-D146** `fedavg` arm = feddance with selector `utility=random` (uniform over the same eligible set); only selection differs (S4/F3).
- **FX-D145** FedAvg sums updates in sorted end-id order: arrival order differs sim vs real, so float sums did.
- **FX-D144** Trainer seeds torch from logical identity: `(trainer, "data")` before load (GPU-preload augmentation) and
  `(trainer, round, task#)` before each pass (batch shuffle). No seed was set, so results rode PyTorch's fixed default seed and any
  mode-specific draw forked sim from real (cifar feddance G0U mobiperf: round-1 loss differs, picks fork at r14). Also a candidate root
  for "utilities differ by GPU numerics" forks (feddance speech R3, Oort) in the parked parity work.
- **FX-D143** Sim charge profiles re-derived on FX-D141/D142 code (`pool_20261011_1249_PR31_PROF`: 12 CHG + 22 T3 syn_0 reals; T3 syn_0 32/32 green).
  GPU fedavg dispatch 38 → 17 ms (old value carried the round-1 eval cold start); stub fedavg dispatch 13 → 18 ms, download 17 → 18 ms
  (half the cifar feddance T3 syn_50 drift); stub oort_sync dispatch/download ~2x (K=10 shapes fan out to 13): Oort T3 cells need a re-screen.
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

- Speech Oort/refl/feddance accuracy: round-bound (10-10) vs a per-baseline target vs long sims after P8?
- PR37's sync baseline: oort_star (proposal) or another?
- Delete the 2-state syn traces after PR36: does any EuroSys plot still need `syn_10`/`syn_50` 2-state (G1X FedDance `fd_20261012_0503`)?
