# Felix readiness — backprop FL (async_cifar10, google_speech)

Read [ROBUST_FL_READINESS.md](ROBUST_FL_READINESS.md) and [PARITY_READINESS.md](PARITY_READINESS.md) first; their rules, L and T apply
here unrepeated. **Focus (operator 10-12/13): Felix-wide parity to 76 / 76, then convergence parity (FX-N93).** Order: Next steps →
Scoreboard (now, and expected after the next run) → Decisions → reference sections.

**At a glance (10-16):** golden rule ROBUST R0 (short screens first). PR36-short screens are done: no OOM (FX-D175 holds), one sim bug
(Felix eval past-dated, FX-D176), two checker artifacts (FX-D177) and the exit abort (FX-D178), all fixed. Next: **PR37 with the PR36 tail in one batch** (pass 1 ~20.6 h, then replicate pass ~8.2 h; Next steps).
First PR36 run regrades 60 / 76 · accuracy **6 / 12** · convergence **0 / 12** · then FX-D171, PR37 → FX-N12.

## Next steps (persistent queue — top item is next)

Work rule: PARITY C10, C17, C18. Test levels: pytest (~3 min) · CPU tiers T1-T4/TS · GPU ≤ 30-min screens (G0U/G0T/G0UC) · long
(G1*/G2). Unblock map: PR37 + PR36 tail (one batch, mixed-key board; confirms FX-N98) → FX-N12.

- **FX-N100 · Throughput: utilization-aware packing (central goal, operator 10-15) · wip: packing adopted (E3/E3b); FX-D175 confirmed
  on the rescreen (0 OOM, overruns 3 / 10,748); full PR36 is the next measurement.** Design: Active build "throughput".
  *Exit:* on a 10-min mixed batch (CPU + GPU legs, both datasets), legs finished per hour ≥ 2× the PR36 packing, with node CPU busy
  and GPU SM / memory use reported per minute. Interference (overruns > 2% (PL19), `CPU_SAT`, sim slowdown > 1.25×, EV/parity reds vs
  solo) must stay at the solo baseline. The PR36 `--plan` uses the measured packing.
- **FX-N98 · Sim speed + no-overrun rule · wip: sim side met in PR36 A1 (felix ~2.0× incl. startup, fedbuff 2.9×, 0 overruns, accuracy
  within 2 pp); close when the rerun's real legs show ≤ 2% overruns.**
  Active build "sim speed". *Exit:* overrun share ≤ 2% per leg in real and sim (PL19); speech G1AS felix sim ≥ 2× real (never < 1×) with
  accuracy within 2 pp of PR35 A1 (felix 67.9%, fedbuff 64.6%).
- **PR37 + PR36 tail · one batch, both nodes (operator 10-16: no full PR36 rerun) · ready: G3 packing screen, then commit, push,
  `git pull` on shepherd, launch.**
  - *Why one batch:* PR36-short left no logic root; a full PR36 would add only untested mobiperf / B1 cells (mobiperf is not a PR37
    trace) and replicate floors for the draw class. Those run as tail stages (`T*`, `run_pr37.py TAIL_J/TAIL_S`) first in the same queue,
    on the PR37 code key. The 76/76-on-one-key gate is dropped: the board mixes the PR36 / rescreen keys with the tail's key.
  - *Launch (operator, tmux `dg_flame` on each node, same commit, PL18):* `cd ~/flame/lib/python/examples &&
    /coc/scratch/dgarg/miniconda3/envs/dg_flame/bin/python scripts/run_pr37.py --node jayne --hours 30 --measured --one-queue
    --gpu-share 4` (shepherd: `--node shepherd`). Commit: the FX-D171/D176-D178 commit after `0af166d6a`.
  - *Two passes (operator 10-16): never two reals on possibly buggy code.* Pass 1 = tail + one real + one sim per G3 cell. Pass 2 =
    the G3C replicate real per syn_0 + lin cell (`--node jayne-rep` / `shepherd-rep`, ~8.2 h each), launched only if pass 1 found no bug
    and the code key is unchanged; a fix in between reruns pass 1 for the affected cells first.
  - *Pass 1 split (`--plan`, measured packing; 7 vs 4 GPUs):*

    | node | tail (first) | PR37 | end |
    |---|---|---|---|
    | jayne (GPUs 0,2-7) | TB1 speech G0U syn_50_3st felix/feddance/refl/oort_star (~3.2 h); TC5 oort T3 + T3C syn_50_3sts both (~1.7 h); TMP G0U mobiperf_3st six × both (planner places its long oort leg late, ~19.7 h; not a PR37 trace) | A1 speech G3 felix/fedbuff/refl/oort/oort_star | ~19.8 h |
    | shepherd (GPUs 0-3) | TDU cifar G0U + G0UC oort/fedbuff syn_50_3st; TDT cifar G0T + G0TC oort syn_50_3sts (< 1 h) | S1 cifar G3 six; S2 speech G3 feddance | ~20.6 h |

  - *Watching (tmux pane):* every 5 min a `STATUS` board per stage (bar, legs done, running, ETA) plus one `!!` line per failed
    (rc≠0), STALLED (no round 15 min / silent 10 min, auto-killed), DOOMED (prefix-safe EV fail) leg or ABORTED pool; probe FAIL/RED lines
    print live every 10 min (`run_pr28.status_board`). Silence past ETA with no `!!` = check `<sid>.out`.
  - *Early signal:* tail done by ~3 h. A tail red on a PR37 baseline × trace (oort syn_50_3st, felix/feddance/refl speech syn_50_3st)
    voids those PR37 legs (C20); mobiperf reds don't touch PR37.
  - *Analysis when logs land (copy shepherd's `pr37_shepherd_<ts>/` + its `pool_*` + run dirs to jayne with `rsync -a -r`, FX-T48):*
    1. Tail: `parity_ladder.py --grade <T* pool> --regrade --max-stage 9`; then `readiness_score.py` over the first PR36 pools (A1-D1,
       S1-S2), the rescreen pools (Scoreboard) and the T* pools. A later pool supersedes per cell; draw cells grade on the T* replicate
       floor (G0UC/T3C/G0TC). Report the board as mixed-key (PR36 `c9f2…`/`deec…`, rescreen, PR37 key).
    2. PR37 pass 1: per cell, sim vs its one real at matched rounds within 2 pp + time-to-target (convergence board A/C/U/S, provisional);
       pass 2 adds the G3C band (decision rule below); `accuracy_table.py`;
       `resource_report.py <all pools> --util` per node into Throughput.
    3. FX-N98: real G3 overrun share ≤ 2% (PL19) closes it; Felix sim/real speed from the G3 pairs.
  - *First PR36 run (10-15, kept on the board):* G1AS speech felix sim 68.1% / fedbuff 65.9%, 90 min vclock in 45.5 min wall, 0 overruns;
    cifar felix 56.0 / fedbuff 56.1 / oort 52.2 / oort_star 52.9%.
- **PR37 · 2 h convergence parity, all six baselines (FX-N93.2/.3) · launches with the tail (item above).**
  - Cells, both datasets: syn_0, syn_50_3st, stream_lin syn_0. Legs: real + sim at 2 h of vclock (tier G3); pass 2 adds a real replicate
    (G3C) of the syn_0 and lin cells; syn_50_3st reuses its baseline's syn_0 band (operator 10-15).
  - Shape (`harness_pool.G3_SHAPE`, all six alike): cifar n:c:K 75:15:10 on 1 GPU, speech 50:15:10 on 2 GPUs; cohort = the reference
    split's first n trainers (same shard, compute and D). K=10 keeps Oort's exploitLen ≥ 1 (FX-L63), sync K/n ≥ 5% (L34), c > K keeps
    async overlap, ≤ 75 / 25 trainers per A40 (FX-T36). Deterministic kernels; eval every 10 rounds.
  - *Decision rule:* every cell inside the real↔real band on the convergence board's C and U criteria ⇒ paper experiments run sim-only
    (FX-N12); a red ⇒ root it on short runs first (C18).
  - *FX-D171 (in tree 10-16):* Felix utility = Oort's (`stat_utility: fedscale` on both `felix` rows of `baseline_reference.yaml`). Was
    `legacy`: first mini-batch only, √(|D|·Σ_{b0} L²); now |D|·√(EMA of batch-mean L²) over the whole first pass.
  - *Watch (PR37 vs PR36 Felix):* a Felix-only shift. U grows ~√(|D|/B) (~2.3× at |D|=167, B=32), so fedbuff's β rises a few % (0.89 →
    0.94); ranking weights |D| linearly; a train-pass U reads lower than an eval U (check `utility_belief`). **Revert (drop the two lines)
    if Felix falls outside its real↔real band.**
- **FX-N94 · tiny_cpu first-contact sim delivery lag · todo.** `smoke_fxd160_0010`: first-contact trainers measured 1.1-1.7 s FX-D108 lag
  (real ~0.03 s), pushing sim's clock 1.7 s ahead by round 3; GPU legs stay ≤ 0.5 s. Check whether the trainer is still initializing when
  its first task arrives in sim. *Exit:* first-contact lag ≈ later tasks' on tiny_cpu.
- **FX-N95 · speech sim round close → train start ~0.1 s short · todo.** PR35 gs G0U feddance syn_50 r1→r2: real close→start 0.30 s,
  sim 0.20 (dispatch 0.09 + measured lag 0.11); arrival→close real 0.165, sim 0.128 (its two reals agree to 0.002). Not on stub: gs T3 oort
  syn_20s sim drifts −2.1 s by r10, its real replicate −1.8 s (lag order is a draw). *Exit:* mean sim−real per-round offset inside real↔real.
- **FX-N85 · Close PR27/PR28 roots · wip.** Left:
  - (b) cifar Oort-family stub legs run one real step (FX-D110 extended, `harness_pool.shaped()`); PR36 C1/C3 are its rerun.
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
- **FX-N9 · Unavailability, all six × syn_50_3st / mobiperf_3st · wip.** PR35 key (2-state syn_50): 10/12 (draw cells), mobiperf 10/10; left: PR36.
  *Exit:* both columns green on one code state.
- **FX-N13 · Streaming, both datasets · wip.** stream columns 12/12 on mixed commits (PR34); left: PR36, G0To (oracle arms), ST6.
  *Exit:* EV19 + INV/EXACT green both datasets; ST6 graded.
- **FX-N30 `[S]` · Speech streaming on GPU screens · wip.** All six G0T green (PR31 A2/B3, PR34 B1); left: G0To. *Exit:* speech G0T/G0To lin + events × syn_0/syn_50 EV19 + INV/EXACT green.
- **FX-N74 `[C]` · Every baseline reaches target on full data · wip: 5 / 12.** Accuracy table below. *Exit:* each cell at target, or
  audited faithful and named round-bound.
- **FX-N76 `[C][S]` · C13 real-cost audit · wip.** weights_to_ram matches; `phase_gpu_compute` is DIAG and overruns of D are graded
  (FX-D175, operator 10-16). Open: `trainer_speed_identity` red only where real overran D (0405: 11.8 vs 10.0 s). *Exit:* graded on what
  reaches the clock, green both datasets. Also open: real speech FedBuff spends 0.12 s mean (p90 0.31 s) arrival → commit on
  1800 s legs (sim 0, U6 DIST red on gs syn_50_3st, rescreen S1S/B1); profile and charge it if it moves K2.
- **FX-N101 · Eval delay: real sleeps floor(D/20) (0 s for every D < 20 s), sim charges D/20 · todo.** Pick the faithful one (D/20) for
  both sides, A/B on a G0T Felix pair. *Exit:* real and sim eval spans equal.
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

**Parity: 60 / 76: PR36's finished pools (keys `c9f29966340b` + `deec34d15ea6`, same run code), regraded on scratch copies with
FX-D174's checker** (10-15, jayne; `--grade --regrade --max-stage 9`, then `readiness_score.py` over A1-D1 + S1-S2). FX-D172/D173 change run
code, so the PR36 rerun re-measures every cell. PR36-short doesn't replace this board: rescreen pools `pool_20261016_0057_PR36_JAYNE_{B1,C5,S1S}`
+ `pool_20261016_0056_PR36_SHEPHERD_{S1C,S1S0}` (copied to jayne) regrade with FX-D177 to S1S 12/12, S1S0 12/12, S1C 23/24, B1 9/22 (9
MISSING: FX-D178 abort; 3 mobiperf EV1: 1 commit in 240 s), C5 0/2; every left red is the draw class below. **Accuracy 6 / 12** (cifar felix, fedbuff, oort, oort_star; speech felix, fedbuff).
✅ green · ❌ red · – n/a.

| baseline · dataset | syn_0 | syn_20_3st | syn_50_3st | mobiperf | stream_lin | stream_eve | stream_cpu |
|---|---|---|---|---|---|---|---|
| felix · cifar | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ |
| felix · speech | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | – |
| fedbuff · cifar | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ |
| fedbuff · speech | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ | – |
| refl · cifar | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| refl · speech | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | – |
| oort · cifar | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ |
| oort · speech | ✅ | ✅ | ❌ | – | ✅ | ✅ | – |
| oort_star · cifar | ✅ | ❌ | ❌ | ✅ | ✅ | ❌ | ✅ |
| oort_star · speech | ✅ | ✅ | ❌ | – | ❌ | ❌ | – |
| feddance · cifar | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| feddance · speech | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | – |

**Open reds → root (16):**
- *Killed by EV21, rerun (9):* oort_star syn_50_3sts (cifar T3, G0U, G0T eve; speech T3, G0U, G0T lin/eve), fedbuff syn_50_3st (cifar T3;
  speech G0U, G0T lin). With the real leg MISSING the regrade keeps the stored pre-fix EV; FX-D174's EV21 passes on those sims.
- *Real draws: trainers-at-N off by 2 of ~35 (5.4% vs 5% tol) (5):* oort T3 syn_20/50_3sts cifar, G0U syn_50_3sts cifar, T3 syn_50_3sts
  speech; oort_star T3 syn_20_3sts cifar; feddance speech G0T eve syn_50_3st. Real↔real forks the same way (cifar oort T3 syn_50_3sts: first
  commit fork #0 in both pairs; speech: reals 33 vs 35, sim = replicate). Needs the rerun's T3C/G0UC floors (C3 died, so none existed).
- *Leg too short, fixed:* oort cifar T3 mobiperf_3sts: 1 commit in 960 s, both legs identical (unaware sync Oort abandons ~10 of 11 picks
  per 90 s now that AVL_EVAL counts as UN_AVL). `MIN_RUNTIME_S` 3000 (G0U sim's 4th commit at 2508 s).
- *Sim bug, fixed (FX-D173):* fedbuff speech G0U mobiperf_3st EV16. A 0.75 s gate release crossed a short AVL_TRAIN window, the async loop
  re-gated the released update, and the second withhold overwrote its sct.
- *Checker artifact, fixed (FX-D174; green above):* fedbuff cifar G0U mobiperf_3st throughput: the sim matched the replicate real to 0.1 s per commit;
  the stop cut real's round 6 after 1 of 2 commits.
- *Open:* felix cifar G0T eve syn_0 (first graded run). Sim = both reals to ≤ 0.5 s through round 40; after the round-37 selection fork it
  falls 9% behind (picks mean D 9.76 vs 9.35/9.24 s), while the two reals (forked at commit #31) agree within 2%. Mix-adjusted residual
  −0.02 s. One sim trajectory can't separate a post-fork draw from a mechanism; the rerun gives a second.

**Expected after the PR37 batch's tail stages:** 76/76 is possible only if the 5 draw cells clear on the new floors and felix eve is a draw. The 9 killed
cells and the 2 fixed cells should go green.

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
| cifar · felix, fedbuff | G1AS PR36 A2 | — | felix 37.8 / 51.5 / 53.4 · 56.0 · fedbuff 39.5 / 46.3 / 55.3 · 56.1 | **sim target at 56 / 63 min** (PR35 52 / 65) |
| cifar · oort, oort_star | G1AS PR36 A2 | — | oort 27.4 / 44.9 / 52.2 · 52.2 · oort_star 44.1 / 50.8 / 52.9 · 52.9 | **sim target at 47 / 59 min** |
| cifar · refl, feddance | G1AS PR36 A2 | — | refl 19.7 / 25.8 / 25.7 · 25.8 · feddance 17.3 / 27.0 / 33.3 · 33.3 | below 50%: audit or round-bound (FX-N74) |
| speech · felix | G1A SGD (run 18) | 41.5 / 58.8 / 67.5 · 67.5 | 46.0 / 56.0 / 68.1 · 68.1 | **target at 72 min both** |
| speech · fedbuff | G1A SGD (run 18) | 40.8 / 54.6 / 59.1 · 60.2 | 42.5 / 55.9 / 64.0 · 64.0 | **target at 82 / 88 min** |
| speech · felix, fedbuff | G1AS PR36 A1 (bf16) | — | felix 45.2 / 57.4 / 68.1 · 68.1 · fedbuff 42.3 / 55.3 / 65.9 · 65.9 | **sim target at 64 / 87 min** |
| speech · oort_star, oort, refl, feddance | G1AS PR36 A1 | — | 18.0 · 16.5 · 3.7 · 5.0 max @90 (3-4 evals) | **round-bound (operator 10-10)**; PR28 G4 16.7 · 14.8 · 3.7 · 4.0 |
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
- **S · sim speed (operator 10-14):** never slower than real.
  - 1-2×: okay, but needs a review and sanity check (where the wall goes).
  - 2-5× and beyond: the goal.
  - Speedup is the simulator's purpose, so every change keeps looking for more of it (FX-N98, S6).

| baseline · dataset | A | C | U | S |
|---|---|---|---|---|
| felix · cifar / speech | ✅ / ✅ | ⬚ / ⬚ (run 18 pre-76/76) | ⬚ / ⬚ | ✅ 4.3× / ✅ ~2.0× (PR36 A1) |
| fedbuff · cifar / speech | ✅ / ✅ | ⬚ / ⬚ | ⬚ / ⬚ | ✅ / ✅ 2.9× (PR36 A1) |
| oort, oort_star · cifar | ✅ ✅ | ⬚ | ⬚ | ✅ |
| refl, feddance · cifar | ❌ ❌ (29.5, 35.6%) | ⬚ | ⬚ | ✅ |
| oort, oort_star, refl, feddance · speech | round-bound: needs C (same accuracy, same time) | ⬚ | ⬚ | ✅ |

**Convergence 0 / 12** (A 6/12 · C 0 · U 0 · S 12/12: speech felix ~2.0×, fedbuff 2.9×, sync four ≥ 13× at G1AS).

---

## Decisions (operator; standing until changed here)

- **Node (10-13, 10-14):** batches use jayne GPUs 0,2-7 (GPU 1 out) and shepherd GPUs 0-3 together (R4); score on the node holding
  every scored pool's runs (copy shepherd pools and runs to jayne, FX-T48).
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
- **PR36 prerequisites (10-14):** skip PR35b; FX-N97, FX-N96, FX-N98 first, then PR36 re-runs everything on one code key.
  - AVL_EVAL is eval-only and Felix-only; for every other baseline it equals UN_AVL (FX-D165).
  - 3-state syn traces (EVAL share from `mobiperf_3st_50`) are the default; 2-state syn traces deleted (FX-D166, `b1c53e940`).
  - At most 1-2% of tasks may overrun D, in real and sim; above that, fewer trainers per GPU (FX-N98).
  - Sim must never be slower than real.
- **Sim-only experiments (10-14):** after PR37 (all six baselines × syn_0 + syn_50_3st + stream_lin × both datasets, 3 h, n:c:K equal across
  baselines) sits inside the real↔real band, paper experiments run sim-only.
- **Sim speed (10-14):** sim never slower than real; 1-2× needs review; 2-5× and beyond is the target; always look for more.
- **Speed levers (10-14 late):** bf16 for speech if accuracy stays within 2 pp (FX-D168); balanced GPU placement without an A/B
  (FX-D169); PR37 at one scaled n:c:K per dataset that satisfies every baseline's constraints (G3_SHAPE); jayne 7 GPUs + shepherd 4.
- **Nodes (10-14):** batches use jayne GPUs 0,2-7 and shepherd GPUs 0-3 together; more parallel legs via scaled n/c/K at fixed ratios.
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
- **Traces:** `syn_0`, `syn_20_3st`, `syn_50_3st`, `mobiperf_3st` (FX-D166). `syn_10_3st` = the old "syn_20" level (10.8%, EuroSys'26;
  FX-D18). The 2-state `syn_{10,20,50}` keys are deleted (`b1c53e940`); scoreboard columns keep the short names.
- Levels (n=300, TRAIN/EVAL/UN %) flat 1 min-3 h: syn_10_3st 89/4.5/6.5 · syn_20_3st 80/8/12 · syn_50_3st 50/20/30 · mobiperf_2st
  10/0/90 · 3st_50 10/25/65 · 3st_75 10/10/80. Mobiperf trainable share rises to 18-22% by 6-24 h.
- **IDs:** `FX-N` next steps · `FX-L` lessons · `FX-T` tripwires · `FX-D` built · `PR#` run batches (PARITY). Shared work is `S#` in the parent.
- **Deprecated reference:** [PARITY.md](../async_cifar10/PARITY.md) §2-5 · [PARITY_CHECKER_README.md](../async_cifar10/scripts/parity/PARITY_CHECKER_README.md)
  · [UNAVAILABILITY_DESIGN.md](../async_cifar10/UNAVAILABILITY_DESIGN.md) · [EXPERIMENT_felix_streaming.md](../async_cifar10/docs/EXPERIMENT_felix_streaming.md)
  · [BASELINES.md](BASELINES.md).

## Active build — sim speed and the no-overrun rule (FX-N98; operator 10-14)

- *Root (speech G1AS felix sim, n=100, 4 GPUs, `run_20261014_214734`: 1.03×, 1.10× after warm-up):* the barrier waits 566 of 621 s for
  the earliest-deadline in-flight task, i.e. the fast (D 10-20 s) trainers' compute + slot wait (FX-L77). Round-robin had put 7 of 16
  fast trainers on one GPU (76% busy, 5-13 s slot waits). The sim needs 1.9 GPU-s of compute per vclock s (1277 / 680).
- *Levers landed:* bf16 autocast for speech, real and sim (FX-D168: 2.61 → 1.62 s per task); balanced placement default (FX-D169;
  operator 10-14: no A/B). A 2-GPU A/B (0.82× balanced vs 0.90× round-robin, both capacity-bound) did not test the 4-GPU case.
- *Rule (PL19):* ≤ 2% of train tasks per leg overrun D, in real and sim; above that, fewer trainers per GPU (real: more GPUs per leg or lower
  `gpu_train_slots`, FX-T36; sim: K). Real speech G1A at 3.5% (fp32) broke it; bf16 should drop it.
- *Confirm in PR36 A1:* speech G1AS felix sim ≥ 2×, overruns ≤ 2%, felix/fedbuff accuracy within 2 pp of PR35 A1, cifar not slower.
  Next lever if short: more GPUs per speech sim leg.

## Active build — throughput (FX-N100, absorbs FX-N99; operator 10-15)

- *Goal:* finish the most legs per wall hour at high CPU/RAM/GPU use, with each leg behaving as if alone (R18, R24). Proven on 5-10 min
  runs, then every long batch uses it.
- *Evidence (first PR36 run, jayne, 14 h; ledger `resources.jsonl`):* load 1-12 of 128 CPUs (~5%); GPU memory 100-195 of 315 GB; 2-15 legs
  at once, mostly 3-4 after 4 h; GPUs idle for the last ~2.5 h while CPU tiers finished. Peak GPU memory per leg: cifar G0U ~20 GB (one
  GPU of 45), speech G0U ~25 GB per GPU, G1AS ~31-35 GB per GPU. Mean busy cores: real legs 0.3-0.5, sims 2.2-4.6. GPU compute (SM %) is
  not recorded.
- *Plan, each step tested on a 5-10 min batch:*
  1. *Instrument:* per-minute node CPU busy cores, RAM, per-GPU SM % and memory in the ledger. `resource_report.py --util` gives time-
     weighted utilization, idle capacity, legs and leg-budget-hours finished per wall hour, and per-leg interference (overrun share, `CPU_SAT`,
     duration vs solo).
  2. *GPU sharing:* place legs on GPUs by measured peak memory and SM % (history per leg key), up to ~85% of memory and ~90% SM. Sim
     vclock is unaffected (L1); real timing is unaffected while compute < D, so the overrun share (PL19) is the interference signal.
  3. *Measured CPU leases + admission (FX-N99, built):* real CPU legs lease measured cores; sims keep the formula (their wall is their
     throughput).
  4. *One queue across stages and nodes:* no lane waits on another, and the GPU tail fills with GPU work, not CPU stragglers.
- *Binding constraints (first PR36 run, BLOCKED counts):* CPU leases held idle by other pools (478), GPUs held exclusively (~600), pool
  RAM budget (261), lanes serial (C3 began at 13:02 after C1). Baseline `resource_report.py --util`: 15.9 legs/h, 3.3 legs running on
  average, CPU 6%, GPU memory 39%, GPUs idle 25% of GPU-time. Real cifar compute is 0.18 s per 4 s task, so GPU memory, not SM, binds GPU
  legs; measured leg RAM (cifar G0U 43-59 GB, speech 65-80 GB) ≈ formula, so RAM caps jayne at ~8 cifar or ~6 speech GPU legs.
- *Built (10-15, uncommitted):* ledger samples per-GPU SM % (`gpu_sm_pct`), per-leg SM (`nvidia-smi pmon`), node CPU busy and RAM; each
  GPU leg's DONE line and ledger end event carry SM % and overran/train tasks (`OVERRUN` > 2%); history `gpu|<key>` = [peak GB, SM %] per
  GPU. `--gpu-share K` places legs on GPUs by measured footprint (`place_gpus`, `gslot_<g>_<i>` leases carrying GB,SM): ≤ 85% memory
  committed, ≤ 90% SM, live use + peak ≤ 90%; unmeasured legs claim a whole GPU. Under `--lease-from-telemetry` the measured RAM peak
  ×1.15 sizes the reservation both ways. `resource_report.py --util`: utilization, legs/h, budget-h/h, concurrency, overruns, slow legs.
- *Measured footprints (E1 + first PR36 run; per leg, n=50 G0U, n=15-40 T3; refresh as screens add history):*

  | leg | cores mean / p95 | RAM peak GB | GPU GB per GPU | SM % | wall vs budget |
  |---|---|---|---|---|---|
  | cifar G0U real (1 GPU) | 0.9 / 2.7 | 49-53 | 19.6-19.8 | 0-1 | 240 s budget → 7.8 min (join + teardown ~3.5 min) |
  | cifar G0U sim (1 GPU) | 1.8-2.0 / 4.6-5.6 | 24 (PR36: 43-59) | 19.4-19.8 | 0-1 | 3.7-4.2 min |
  | speech G0U real (2 GPUs) | 0.8-1.0 / 2.6-4.2 | 59-65 (PR36 ≤ 78) | 21.6-28.3 | ~1 | 6.7-7.9 min |
  | speech G0U sim (2 GPUs) | 2.2-2.4 / 4.9 | 34-47 (PR36 ≤ 79) | 14.8-20.9 | ~1 | 2.3-2.7 min |
  | G1AS sim n=300 cifar / n=100 speech (4 GPUs) | 4.6 / 3.7 | 133-163 | ~31-35 | n/a | 90 min vclock: cifar ~20 min, speech 31-46 min |
  | T3 real / sim (stub) | 0.3-0.9 / ≤ 1.2; sim 0.6-1.7 / ≤ 3.3 | 6-13 | 0 | 0 | real 5.6 min; sim 1.8-2.9 min |

  GPU memory is ~0.4 GB per trainer process (CUDA context + preloaded shard), so GPU legs bind on GPU memory and host RAM, never SM.
- *E1, exclusive GPUs (`pool_20261015_tput_E1`; G0U + T3 syn_50_3st, felix + fedbuff, both datasets, `--scale-smoke 240`):* 68.9 legs/h,
  4.59 budget-h/h, concurrency 5.4; CPU 5%; RAM peak 232 GB of 504; GPU SM 2%, memory 13%, GPUs idle 62%; overruns 0/565; 3 slow legs
  (G0U reals 470 s vs 342 s est: join + teardown). Parity reds on G0U fedbuff (both) and gs felix are 240 s-leg artifacts and the control
  for E2.
- *E2 attempt 1 (aborted):* the speech felix sim CUDA-OOMed on GPU 0 (44.7 of 44.4 GiB usable) beside a cifar real. Its recorded footprint
  (14.8 GB/GPU) was a missed peak: a 2.7-min sim gets few 30 s samples, while its real twin (same trainers) peaked at 28.3. Fixed: a
  footprint is the max over the key's real and sim legs; `GPU_GB` = 44.4; caps 90% (operator 10-15).
- *GPU memory per trainer (micro-benchmark, idle GPU 7):* 312 MiB of bare CUDA context per process; cuDNN kernels add ~4 MiB (lazy loading
  is already on). MPS does not shrink it (20 processes: 6,257 vs 6,319 MiB); its pipe dir must be short (108-char socket limit), and clients
  index the server's visible GPUs. So a 50-trainer leg holds ~15 GB of contexts before data. With caps, only two cifar G0U legs (19.8 GB)
  share a GPU; speech legs (21.6-28.3 GB/GPU) stay alone.
- *E2, whole-leg sharing (`pool_20261015_tput_E2`):* 74.5 legs/h (+8%), concurrency 6.0, GPUs idle 57%, overruns 0/570, 0 slow legs,
  parity verdicts = E1. Only cifar legs paired: speech legs (≥ 21.6 GB/GPU at their even 2-GPU split) never fit beside another leg.
- *Trainer-granular packing (built 10-15):* the placer spreads a leg's trainers over any GPUs with room (`place_trainers`; per-trainer GB =
  per-GPU peak / even split; partly used GPUs first; ≥ 5 trainers per GPU; aggregator's 3 GB on a GPU the leg already uses, listed
  last), passed as `FLAME_GPU_TRAINER_CAPS` to the spawner (`balanced_gpu_placement(caps)`, rate-balanced within caps; off =
  byte-identical). Read → choose → commit runs under a node-wide `flock` (`gpu_place.lock`), and a leg reserves its whole footprint
  before launch (all or nothing), so two legs can never each hold part of the memory they need. Footprints are keyed by dataset ·
  harness · baseline · n · GPUs (memory doesn't depend on mode, trace or runtime), seeded from PR36's exclusive legs: cifar n=50 20.6-22.4
  GB, speech n=50 38-42 GB per GPU (long legs; E1's 4-min legs read 28), G1AS n=300/100 30-39 GB per GPU.
- *E3, trainer packing (`pool_20261015_tput_E3`):* 81.7 legs/h (+19% vs E1), concurrency 7.2, GPU memory 20%, GPUs idle 34%, overruns 0/572,
  no OOM (shared GPUs peaked 28-34 GB), 1 slow leg. Packed legs ran 5-10% longer than E1's. New red: cifar G0U felix
  `trainer_trace_fidelity_sim` (DIST) at 94% of trainers within tolerance vs a 95% gate (3 of 50, 83 spurious transitions). Speech fedbuff sim
  used 60 GB RAM vs 42 GB reserved: a short E1 sim had lowered it. Fixed: the RAM reservation is 1.15 × the max over every run of that
  baseline at that n (cifar 71, speech 92-93 GB).
- *E3b (E3 repeated with the RAM fix, `pool_20261015_tput_E3b`):* the felix cell is green (100% within tolerance): E3's red was a draw. Verdicts
  = E1, overruns 0/568, no OOM. Throughput fell to 69.8 legs/h: measured RAM reservations (cifar 71, speech 92-93 GB) admit only 5 GPU legs
  at once, while actual RAM peaked at 253 GB (4-min legs use less than the full-length peaks). At production length **host RAM binds the
  speech-heavy stages and GPU memory the rest.** Trainer packing is adopted.
- *PR36-short (packed, 10-15; `--util`):* jayne 70.2 legs/h, concurrency 5.3, CPU 5%, GPUs idle 88% (B1 aborted at 21 min); shepherd
  50.9 legs/h, concurrency 4.9, GPU memory 27%, idle 24%; overruns 2/5047. Both GPU pools OOM-aborted: packed legs had recorded footprints
  ÷2-4 (FX-T51), so placement put 95-137 cifar trainers on one A40. Packed real compute rose 0.05 → 0.2-0.4 s (< D, 0 overruns). Real
  legs ran 1.3-1.5× their estimate (concurrent startup). Fixed in FX-D175; history purged on both nodes (entries < 0.75 × key max).
- *PR36-short rescreen (10-16, `--util`):* jayne 17.8 legs/h, concurrency 2.6, GPUs idle 37%; shepherd 26.8 legs/h, concurrency 2.4, idle
  42%; 0 OOM; overruns 3 / 10,748. Binding: the tail, not a resource. Few cells, and SHORT_KEEP legs run 900-1800 s (S1S 195 min).
- *Sim CPU leases:* T1 (24 sims) at 2 / 4 cores / formula (7-9 cores): 3.07 / 2.26 / 1.62 min per sim, but 282 / **337** / 181 legs/h.
  Sims lease ≥ 4 cores (`SIM_LEASE_MIN_CORES`), real CPU legs ≥ 2; core history falls back across runtimes (`cores_seen`).
- *PR36 plans (`run_pr36.py --plan`, planner models trainer packing + measured RAM):*

  | | old packing (jayne / shepherd) | `--measured --one-queue --gpu-share 4` |
  |---|---|---|
  | full | 13.9 / 11.9 h | 12.5 / 12.3 h (RAM-bound: measured speech peaks > formula) |
  | short, all stages (`--short 240`) | 9.0 / 3.7 h | 4.0 / 3.3 h |
  | short, bug-finding stages only (G0U, T3, P7 / G0T) | – | 2.1 / 3.0 h |
  | rescreen, FX-D175 footprints (B1, C5, S1S / S1C, S1S0) | – | **2.5 / 2.3 h** |

  The short screen grades EV + INV/EXACT logic; G1AS, replicate and third-real stages run only at full length (240 s legs give artifact
  DIST/timing reds, E1).
- *FX-N99 parts (built, flagged):*

- *Root (PR36, jayne 10-15):* `slot_cpus` leases aggregator cores + 1 CPU/trainer (14-74 CPUs/leg); measured `cores_p95` is 0.1-1.2, `cpu_sat`
  0.000 on every finished leg. C3 sat `BLOCKED … 62 held by this pool's legs` with load ~2 on 128 CPUs and all 8 GPUs idle (PL12/N86 again).
  Lanes also run stages serially (A2 waited for A1; two GPU lanes idle from ~08:00).
- *Built (10-15, in tree, uncommitted; `harness_pool.py --lease-from-telemetry [--priority N]`, `run_pr36.py --measured --one-queue`):*
  - *1 Measured lease (`measured_lease`):* a real CPU leg leases ceil(1.25 × margin × max of its key's last-3 `cores_p95`) physical cores,
    at least 2, never above the formula. Sims, GPU legs and legs without history keep the formula. The aggregator gets at most half the
    slot (`FLAME_AGG_CORES`). `plan()` uses the same numbers.
  - *2 Live admission (`NodeLoad`):* a leg starts only while smoothed node busy cores + legs started < 60 s ago + its predicted p95 ≤ 70%
    of CPUs. Measured pools drop the CPU lane's `--reserve-cores`; `plan()` drops `LANE_CPU_CAP`.
  - *3 Back-pressure:* a leg whose slot saturates in > 5% of its last 10 samples, or a run queue > 1 per CPU, pauses starts. The leg's DONE
    line reads `TIMING_SUSPECT` (`run_pr28` lists it under INTERFERENCE → rerun solo), and its key's margin ×1.5 (cap 4).
  - *4 One queue:* every stage starts at once (after the first pool's gate stamp), stage order = priority. A lower-priority pool takes
    GPUs only while enough stay free for a higher-priority pool's blocked GPU leg (`want_p<N>` leases). CPU legs never defer.
  - Tests: `test_measured_lease_sizes_pairs_from_their_p95` (incl. byte-identical off), `test_gpu_priority_wants_span_pools`,
    `test_node_load_reads_busy_cores`.
- *Plan (`--plan --hours 20`):* jayne 13.9 → 13.1 h, shepherd 11.9 h either way. GPU-bound: 5,027 GPU-min on 7 GPUs, ~90% busy. CPU leases
  80k → 60k CPU-min, and the CPU tiers no longer queue behind each other. Shorter needs more GPU legs per GPU (an isolation change, R18).
- *A/B (10-15):* T3 syn_0 felix/fedbuff cifar, old vs measured lease, co-run (`pool_20261015_fxn99_AB{old,new}`): 4/4 PASS. Real↔real
  control old vs measured: throughput 1.4% / 0.3%, trainers-at-N equal. T1 both datasets measured (`pool_20261015_fxn99_T1M`): 24/24 EV
  PASS, but sims ran 1.5-2.2× slower on 4-CPU slots, so sims keep formula leases (only real CPU legs are measured).

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
- **FX-L15** Ramp syn_0 → syn_20_3st → syn_50_3st → mobiperf_3st; every `_3st` trace exercises AVL_EVAL.
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
- **FX-L77** Read sim speed from the critical chain: the barrier waits on the earliest-deadline in-flight task, so per-GPU load of the
  fastest trainers (not mean GPU use) paces the sim (FX-N98: GPU 2 held 7 of 16 fast trainers).
- **FX-L74** "Sim outside all n reals" is no refute: under exchangeability it holds with probability 2/(n+1) (50% at n=3); vote per real (FX-D161).
- **FX-L78** Per-trainer memory is at its floor: 312 MiB CUDA context per process (MPS doesn't shrink it) + ~1.0-1.5 GB host RAM after
  FX-N84's 10-12% cut. Gain throughput by packing on measured footprints (FX-N100), not by trimming trainers (operator 10-15).

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
  `rsync -a --files-from` does not recurse: add `-r` (10-16 copied 72 empty run dirs).
- **FX-T49** Don't classify a pair's rounds with per-side thresholds or drop one side's zero advances: twin rounds split (FX-D159).
- **FX-T50** Don't key a prefix-safe EV on (round, trainer): async selectors re-list held picks every round, so one stale entry DOOMs
  the leg (PR36 EV21, 17 legs). Key on the dispatch.
- **FX-T51** Don't record a packed leg's GPU peak per placed GPU or trust one 30 s sample: footprints read ÷2-4, 137 trainers shared an
  A40, two pools OOM-aborted (PR36-short, FX-D175).

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
- **FX-D165** AVL_EVAL is eval-only (FX-N97): a train result delivers only in AVL_TRAIN, an eval result in AVL_EVAL or AVL_TRAIN
  (`trace.deliverable`), in the real send gate, sim withhold/delivery (`compute_delivery_ts(task)`) and Felix's boundary eviction (both
  modes); unaware baselines still pick EVAL devices, compute, and hold the upload until AVL_TRAIN. EV21 (prefix-safe): aware train picks
  AVL_TRAIN (a held pick re-listed until it commits or is abandoned is no dispatch, FX-D174), real train sends leave in AVL_TRAIN, only Felix evals. The sim's
  starvation wake-up skips AVL_EVAL flips unless the selector sends evals (`_wake_task`). `test_avl_eval_semantics.py`; for non-Felix
  every decision equals the trace with AVL_EVAL relabeled UN_AVL (`test_avl_eval_is_unavail_for_baselines.py`).
- **FX-D166** 3-state syn traces `syn_{10,20,50}_3st` (FX-N96): syn_X's AVL_TRAIN intervals kept, non-train windows relabeled with
  MobiPerf exits (all EVAL 0.33 / EVAL→UN 0.22 / UN 0.45; EVAL = 40.3% of non-train), seeded per (trace, trainer)
  (`gen_synthetic_trace.py --three-state`; self-checks in `test_three_state_traces.py`). Every tier, script and default uses them.
  Verified 10-14 vs the deleted 2-state source: TRAIN intervals 300/300 identical per trace; shares within 0.5 pp (149 h) / 0.6 pp (0-3 h);
  windows all-EVAL 0.32 / EVAL→UN 0.21 / all-UN 0.47; trainer `avail_change` = trace (fidelity 1.0, T3 3st). A 2-state name raises
  a KeyError naming its `_3st` twin.
- **FX-D167** Sim GPU slots (FX-N98): `sim_gpu_train_slots` per GPU (speech 1; `gpu_slot_bench.py`: K=1 11.4 vs 10.1 steps/s at K=2-3)
  replaces real's `gpu_train_slots` in sim; waiters take a slot earliest-deadline first (vclock send + D); sim charges slot-held time only.
- **FX-D168** `trainerAmp: bf16` (speech, `datasets.yaml`, every baseline, real and sim): bf16 autocast on the trainer's forward + loss
  (train, eval, warm-up). Per task 2.61 → 1.62 s; in-process FL (`fl_lr_check` felix, r5-r30) within 2.4 pp of fp32. `""` reverts.
- **FX-D169** Balanced GPU placement (launcher `execution.gpu_placement`, default `balanced`; `FLAME_GPU_PLACEMENT=round_robin` reverts):
  trainers fastest-first onto the GPU with the least task rate 1/D, ≤ ceil(n/G) each (`spawner.balanced_gpu_placement`).
- **FX-D170** `harness_pool` contention log: a CPU block held mostly by this pool's own legs reads `resources-busy`; every 10 min while
  foreign load exists, `LOAD foreign owners:` lists user + run label/script with its GPUs and CPUs (`foreign_owners`, PL-LOG).
- **FX-D172** Async selectors send no evals unless they override `_concurrency_for_task` (Felix's `AsyncOortSelector`, evalGoalFactor > 0):
  FedBuff took a free slot at the eval tasklet ~once per leg, real only (PR36 EV21). `test_no_eval_dispatch_without_eval_goal`.
- **FX-D173** Sim never re-gates a released withheld delivery (`_sim_withhold_if_unavail`, as oort's commit loop): a gate release past a
  short AVL_TRAIN window withheld it twice and lost its sct (PR36 gs fedbuff G0U mobiperf_3st EV16). `test_released_delivery_past_a_short_window_is_not_regated`.
- **FX-D178** Aggregator mains end with `exit_after_run` (`flame/launch/cli.py`): join threads, atexit, flush, `os._exit`; native
  static teardown aborted ~2.5% of sim aggregators after results (FX-N33; killed PR36-short B1). `FLAME_HARD_EXIT=0` reverts.
- **FX-D177** Checker: A6/A7 score observations only up to the run span and match transitions on gt's (t_start, t_end] (short legs read
  exact trainers as err 0.28 on both sides); U6 grades the rounds both legs reached (L10; sim's extra round 11 failed feddance).
- **FX-D176** Sim gate: an eval dispatch's expected completion is its send stamp (was send + train D), so train commits can't lap an
  in-flight eval (Felix evals past-dated 2-3 s since PR32). `sim_eval_expect_at_send=false` reverts.
- **FX-D175** Packing + grading fixes (PR36-short): GPU history per nominal GPU, per-trainer floor (`TRAINER_GPU_GB_MIN`), FX-T36 cap
  across legs (`TRAINERS_PER_GPU`, share in `gslot_*`); `--scale-smoke` keeps `SHORT_KEEP` legs full length and records no GPU footprint
  (short legs read 25-55% of the full peak); Felix grading:
  `phase_gpu_compute` DIAG, `timing_overrun` INV at max(1, 2%) per side (`PL19_OVERRUN_MAX_FRAC`; fwdllm unchanged).
- **FX-D174** Checker: EV21 counts a train pick once per dispatch (async selectors re-list in-flight/withheld picks every round; PR36
  killed 17 legs); throughput/K3/K3b drop the last common round when one side's stop cut it short (`_tail_cut`).
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

- **Q-EV21 · Felix "reaches AVL_EVAL" arms only at ≥ 20 eval picks:** Felix's 35-version eval gate gives 2-3 picks per G0U leg, so screens
  can't fail it. Accept (graded on G1/PR37 legs), or arm at ≥ 1?
