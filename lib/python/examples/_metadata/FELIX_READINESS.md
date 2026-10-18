# Felix readiness — backprop FL (async_cifar10, google_speech)

Read [ROBUST_FL_READINESS.md](ROBUST_FL_READINESS.md) and [PARITY_READINESS.md](PARITY_READINESS.md) first; their rules, L and T apply
here unrepeated. **Focus (operator 10-12/13): Felix-wide parity to 76 / 76, then convergence parity (FX-N93).** Order: Next steps →
Scoreboard (now, and expected after the next run) → Decisions → reference sections.

**At a glance (10-17):** golden rule ROBUST R0 (short screens first). PR37 pass 1 (key `1e6004b6c305`) found no sim logic bug but every
Felix claim failed (Scoreboard "Felix claims"); the **fix bundle** landed (Felix server lr 1.0 = FedBuff's, FX-D185; sync-sim abandon
wake, FX-D182; real eval delay, FX-D183; pool RAM cap, FX-D179; U6 scale, FX-D184) and the **central dispatcher** (`dispatch.py`,
FX-D186) now launches every long run; its live smoke ran on both nodes. Next: **PR38** = PR37 + G3C replicates on key `44d44bd2d67b`,
balanced over jayne GPUs 0,2-7 + shepherd GPUs 0-3 → FX-N108 gate.

## Next steps (persistent queue — top item is next)

Work rule: PARITY C10, C17, C18. Test levels: pytest (~3 min) · CPU tiers T1-T4/TS · GPU ≤ 30-min screens (G0U/G0T/G0UC) · long
(G1*/G2). Unblock map: PR38 (one key: G3 pairs + G3C band) → Felix claims gate (FX-N108) + convergence board C/U → FX-N12.

- **PR38 · PR37 on the fix-bundle key + G3C replicates, one dispatcher batch · wip: running since 10-17 04:57 (commit `2f8dca198`,
  key `44d44bd2d67b`, tmux `dg_flame` window `pr38`, log `experiments/pr38_dispatch.out`; gates ok on both nodes 05:00; ETA ~01:30
  10-18).** The bundle changes the key, so
  every PR37 G3 leg reruns (floors pair within one key, FX-D132): six × {syn_0, syn_50_3st, lin} × both datasets as real + sim pairs,
  plus one G3C real per syn_0 + lin cell (syn_50_3st takes its baseline's syn_0 band, FX-D181). 60 units, 141 leg-h.
  - *Launch (one command, one tmux window on jayne; the dispatcher drives shepherd over ssh):* tmux `dg_flame`, window `pr38`:
    `cd /home/dgarg39/flame/lib/python/examples && /coc/scratch/dgarg/miniconda3/envs/dg_flame/bin/python scripts/dispatch.py
    batches/pr38.yaml`. Writes `experiments/dispatch_pr38_<ts>/DISPATCH.log`; pools `experiments/pool_<ts>_PR38_{JAYNE,SHEPHERD}_{A,B,RA,RB}`
    on their nodes. Stop: one Ctrl+C in that window (ABORT to every pool, both nodes).
  - *Plan (`--plan`, 10-17, commit-time history):* jayne (7 GPUs, 120 CPUs, 475 GB) 38 units, finish ~20.2 h; shepherd (4 GPUs) 22
    units, ~19.2 h. Binding: host RAM (a G3 real reserves ~90 GB, its sim ~70 GB until it ends). Units are released ≤ 30 min ahead of
    their planned start, so the split adapts if one node runs slow.
  - *Watch:* `STATUS <node>` every 5 min (legs running / done, predicted finish), `!!`/`!` health lines (aborted pool, STALLED / DOOMED
    leg, leg > 1.25 × estimate, host RAM < 10% free, foreign process on a batch GPU). Shepherd GPUs 4-7 run another user's jobs
    (pbehnam3, 21 GB each): not ours, ignored.
  - *Claims:* (a) every G3 cell's sim inside its real↔real band (C criteria); (b) FX-N103 classified by 4 reals of the oort ≡ oort_star
    lin process; (c) Felix claims C1-C4 (`felix_claims.py`), now that Felix and FedBuff share the server lr.
  - *Decision:* (a) + (b) green ⇒ parity side of FX-N12 done; (c) red ⇒ FX-N108 (Felix design), not a parity root.
  - *Analysis:* the dispatcher copies shepherd's pools + run dirs here; then `parity_ladder.py --grade <pool> --regrade --max-stage 9` per
    pool, `readiness_score.py`, `felix_claims.py <pools>`, `accuracy_table.py`, `resource_report.py --util` per node, and the
    Scoreboard's dynamics / comm tables.
  - *Known short-leg artifact (not a root):* Felix speech S3/4 `num_chosen` totals 77 vs 84 on 240 s legs (sim vclock 267 s vs real
    237 s at equal rounds; packing screen and dispatcher smoke alike); the 2 h PR37 Felix speech cells were green.
- **FX-N103 · Cifar Oort-family pacer under linear streaming · wip (draw class pending PR38).** G3 lin oort / oort_star red on
  S1/S8 + A2c/T2/P3: real 551 / 568 versions, sim 665 / 383. Root: Oort's pacer relaxes `round_threshold` only when the 20-round utility
  sum changes ≤ 10%; FedScale utility ∝ visible |D|, which grows with the stream, so window ratios sit at 0.05-0.5 (syn_0: < 0.1 → all
  four syn_0 legs ramp T alike, 391-395 versions). Each borderline window flips T's ramp, which sets the preferred duration (4-21 s),
  hence picks' speed and round length (positive feedback). Stream growth is identical on all legs (visible 0.16 → 0.94). Four extra
  sim replicates (`experiments/rep_20261017_oortlin`): 455, 380, 495, 361 ⇒ sims are not run-to-run reproducible here; spread 361-665
  brackets both reals. Accuracy is insensitive (max 39-45% across all). *Exit:* PR38 reals classify it (claim b); a mechanism ⇒
  diff the pacer's utility input real vs sim (`exploitation_util_history` telemetry, C7).
- **FX-N108 · Felix claims gate (operator 10-17) · todo, blocks FX-N12.** Scoreboard "Felix claims" C1-C4 all ✅ on one key, both
  datasets, syn_0 / syn_50_3st / lin (+ mobiperf, α sweep); graded by `scripts/felix_claims.py` (exit 1 + one WARNING line per miss).
  - *Done:* server lr (FX-D185); rate type checked irrelevant at lr 1.0 (in-process ±0.5 pp).
  - *Left:* Felix's selector runs Oort's paper defaults (`scoring.OORT_PAPER_DEFAULTS`: exploration, pacer, round_penalty α 2,
    cut_off_util), never tuned for Felix; its top-20% clients supply 57% of updates (FedBuff 23%). A G3 sim screen per knob (~8 min
    each, cifar lin + syn_0, where C2' fails) after PR38 shows the claims at lr 1.0.
  - *Heterogeneity:* G3 sims at Dirichlet α 0.1 / 0.5 / 1.0 (C4). *Decide with the operator:* C4 scored on time-to-target, or only among
    baselines that reach target (1-mini-batch refl / feddance never use the full shard, so streaming can't hurt them).
  - *Comm:* Felix ships a full model with every eval task (+11% downlink cifar); C3' (to 40%) fails vs sync Oort until Felix reaches 40%
    clearly sooner.
- **FX-N100 · Throughput: utilization-aware packing (central goal, operator 10-15) · wip.** Design: Active build "throughput". PR37
  pass 1 (`--util`): jayne 3.7 legs/h, concurrency 2.7, CPU 2%, RAM peak 377 / 504 GB, GPU memory 48%, idle 17%; shepherd 2.3 legs/h,
  concurrency 2.0, RAM peak 329, GPU memory 34%, idle 36%. Binding: host RAM reservations of G3 reals (73-95 GB, n=50-75 trainers ×
  ~1.1 GB); FX-D179 removed the stale per-pool cap, FX-D186 the static node split. Next lever: FX-N84's fork-server import sharing
  (~280 MB/trainer). *Exit:* on a 10-min mixed batch (CPU + GPU legs, both datasets), legs finished per hour ≥ 2× the PR36 packing, with
  node CPU busy and GPU SM / memory use reported per minute. Interference (overruns > 2% (PL19), `CPU_SAT`, sim slowdown > 1.25×,
  EV/parity reds vs solo) must stay at the solo baseline.
- **FX-N106 · Unaware baselines under syn_50_3st on speech are near-degenerate · todo (operator decision).** In 2 h: oort 13 versions
  (3.7%), oort_star 44, refl 53, feddance 76, fedbuff 83 (vs 95-204 at syn_0); cifar loses only 0-38%. Speech tasks run ~5× longer
  (~50 s) against the same 1 min-3 h windows, so far more cross a flip; abandons are 17-35% of dispatches (cifar fedbuff 3%) and each
  holds a slot or a sync round for the 450 s scaled timeout. Faithful to each baseline as configured, but these cells give the paper
  ~no accuracy signal and only weak parity signal. *Decide:* longer legs for speech unavailability cells, or report them round-bound.
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
- **FX-N9 · Unavailability, all six × syn_50_3st / mobiperf_3st · wip.** PR37 tail + G3 (key `1e6004b6c305`): syn_50_3st 9/12 (draw class: Scoreboard), mobiperf 10/10.
  *Exit:* both columns green on one code state.
- **FX-N13 · Streaming, both datasets · wip.** stream_eve 12/12, stream_lin 10/12 (cifar Oort pair: FX-N103); left: G0To (oracle arms), ST6.
  *Exit:* EV19 + INV/EXACT green both datasets; ST6 graded.
- **FX-N30 `[S]` · Speech streaming on GPU screens · wip.** All six G0T green (PR31 A2/B3, PR34 B1); left: G0To. *Exit:* speech G0T/G0To lin + events × syn_0/syn_50 EV19 + INV/EXACT green.
- **FX-N74 `[C]` · Every baseline reaches target on full data · wip: 5 / 12.** Accuracy table below. *Exit:* each cell at target, or
  audited faithful and named round-bound.
- **FX-N76 `[C][S]` · C13 real-cost audit · wip.** weights_to_ram matches; `phase_gpu_compute` is DIAG and overruns of D are graded
  (FX-D175, operator 10-16). Open: `trainer_speed_identity` red only where real overran D (0405: 11.8 vs 10.0 s). *Exit:* graded on what
  reaches the clock, green both datasets. Also open: real speech FedBuff spends 0.12 s mean (p90 0.31 s) arrival → commit on
  1800 s legs (sim 0, U6 DIST red on gs syn_50_3st, rescreen S1S/B1); profile and charge it if it moves K2.
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

**Parity: 69 / 76, mixed key** (10-17, jayne; `parity_ladder.py --grade --regrade --max-stage 9`, then `readiness_score.py`, a later
pool superseding per pair). PR37 key `1e6004b6c305`: tail `pool_20261016_0542_PR37_{JAYNE_TMP,TB1,TC5; SHEPHERD_TDU,TDT}` + G3
`…_JAYNE_A1`, `…_SHEPHERD_{S1,S2}`, `pool_20261016_2056_PR37_JAYNE_S1_S1J`, `pool_20261016_2345_PR37_JAYNE_S1B_S1K`, on top of the
PR36 rescreen pools (`pool_20261016_005{6,7}_PR36_*`); syn_20_3st and stream_cpu are still the first PR36 run's (60 / 76 board, keys
`c9f29966340b` + `deec34d15ea6`). INV/EXACT + EV only; DIST via floors. ✅ green · ❌ red · – n/a.

| baseline · dataset | syn_0 | syn_20_3st | syn_50_3st | mobiperf | stream_lin | stream_eve | stream_cpu |
|---|---|---|---|---|---|---|---|
| felix · cifar | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| felix · speech | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | – |
| fedbuff · cifar | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ |
| fedbuff · speech | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | – |
| refl · cifar | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| refl · speech | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | – |
| oort · cifar | ✅ | ❌ | ✅ | ✅ | ❌ | ✅ | ✅ |
| oort · speech | ✅ | ✅ | ❌ | – | ✅ | ✅ | – |
| oort_star · cifar | ✅ | ❌ | ✅ | ✅ | ❌ | ✅ | ✅ |
| oort_star · speech | ✅ | ✅ | ✅ | – | ✅ | ✅ | – |
| feddance · cifar | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| feddance · speech | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | – |

**Open reds → root (7; no sim logic bug):**
- *Pacer draw, pending PR38 (2):* oort, oort_star · cifar stream_lin (G3 S1/S8; FX-N103).
- *Draw class without a floor (3):* oort, oort_star · cifar syn_20_3st (first PR36 T3 K8: 2 of ~35 trainers-at-N; never rerun);
  feddance · speech syn_50_3st (TB1 G0U S8 K8; no replicate; its G3 cell is INV/EXACT green).
- *Draw at screen length (1):* fedbuff · cifar syn_50_3st (TDU G0U 15 min: matched-window throughput 0.086 vs tol 0.08, its G0UC
  real↔real control reads 0.070). The 2 h G3 cell agrees to 0.3%: 823 sim / 825 real / 822 spare real versions.
- *Degenerate cell (1):* oort · speech syn_50_3st (G3 S8 on 11 matched versions; 13 vs 11 in 2 h; FX-N106).

**Expected after pass 2:** the lin pair either clears as a draw (reals spread like sims) or opens a pacer-input root. 76 / 76 also needs
T3C floors for the two cifar syn_20_3st cells and a replicate for speech feddance syn_50_3st (not in pass 2).

**PR37 G3 parity, full grade incl. DIST (pass 1, one real per cell): 26 / 36.** EV 72 / 72 PASS on every leg.

| baseline · dataset | syn_0 | syn_50_3st | lin |
|---|---|---|---|
| felix · cifar / speech | ✅ / ✅ | ✅ / ✅ | ✅ / ✅ |
| fedbuff · cifar / speech | ✅ / ✅ | ✅ / ❌ U6 | ✅ / ✅ |
| refl · cifar / speech | ✅ / ✅ | ✅ / ✅ | ✅ / ✅ |
| oort · cifar / speech | ❌ C2 / ✅ | ❌ C2 / ❌ A3, S8 | ❌ FX-N103 / ✅ |
| oort_star · cifar / speech | ❌ C2 / ✅ | ❌ C2 / ✅ | ❌ FX-N103 / ✅ |
| feddance · cifar / speech | ✅ / ✅ | ❌ T2 / ❌ U6 | ✅ / ❌ C2 |

- *C2 loss by round (cifar Oort family, 4):* common-mode noise, not a mechanism. YoGi test loss jumps 0.24-0.84 between consecutive
  evals on both sides (other baselines 0.02-0.09; max 8.98 on both) against a 0.15 tolerance; accuracy agrees within 1-2 pp. Needs the
  pass 2 floor (L12).
- *U6 (2):* speech fedbuff: real's 0.10 s arrival → commit has no sim charge (FX-N76; versions 83 / 83). Speech feddance: KS 0.022,
  mean gap 2.1 s on ~46 s lags vs an absolute 2 s tolerance (fixed: FX-D184).
- *T2 (cifar feddance syn_50_3st):* p99 of picked D 20 vs 16 s (means 8.0 / 7.7 s, KS 0.027): tail draw. *A3 (speech oort syn_50_3st):*
  one decile 0.207 vs 0.20 on 13 rounds.

### Felix claims (operator 10-17): the paper's acceptance criteria, graded per run

Every baseline must reach the target (cifar 50%, speech 60%); Felix's time-to-target (TTA) must be **significantly lower than every
baseline in every scenario**; Felix's trainer→aggregator and aggregator→trainer traffic to target must be the **lowest**; Felix must
be the **least affected** by streaming, data heterogeneity and high unavailability. A ❌ here is a release blocker, not a footnote.

> ⚠️ **WARNING (PR37 pass 1, 10-17): every Felix claim fails at the run's config.** Root FX-D185 (Felix server lr 0.3 / 0.5, never
> re-tuned). At server lr 1.0 (cifar G3 sims only) TTA-to-target and comm-to-target pass; TTA-to-40% under streaming, comm-to-40%,
> and least-affected still fail.

| claim | cifar syn_0 | cifar syn_50_3st | cifar lin | speech syn_0 | speech syn_50_3st | speech lin |
|---|---|---|---|---|---|---|
| C1 all six reach target in 2 h | ❌ fedbuff only (felix @1.0 sim too) | ❌ fedbuff, oort_star peak at 50.0 without crossing (felix @1.0 sim ✅) | ❌ none (felix @1.0 sim ✅) | ❌ felix, fedbuff only | ❌ none | ❌ none |
| C2 Felix TTA lowest (target) | ❌ fedbuff 107 min, felix never · ✅ @1.0: 53 | ❌ never · ✅ @1.0: 57 | ❌ never · ✅ @1.0: 101 | ❌ felix 115 vs fedbuff 112 | ⬚ nobody reaches 60 | ⬚ nobody |
| C2' Felix TTA lowest (40%) | ❌ 72 vs oort 31 · ✅ @1.0: 24 | ❌ 78 vs oort_star 43 · ✅ @1.0: 39 | ❌ 114 vs oort_star 47 · ❌ @1.0: 57 | ✅ 36 vs fedbuff 42 | ✅ 52 vs fedbuff 102 | ❌ 78 vs fedbuff 65 |
| C3 Felix comm lowest (to target) | ❌ never reaches · ✅ @1.0: 20.4 GB vs fedbuff 36.6 | ✅ @1.0 only one there | ✅ @1.0 only one there | ❌ 124 vs fedbuff 111 GB | ⬚ | ⬚ |
| C3' Felix comm lowest (to 40%) | ❌ 28.1 vs oort 8.5 GB · ❌ @1.0: 9.9 | ❌ 30.5 vs oort_star 8.1 · ❌ @1.0: 16.1 | ❌ 45.0 vs oort_star 16.1 · ❌ @1.0: 22.3 | ❌ 41.5 vs fedbuff 41.3 | ❌ 59.1 vs fedbuff 41.5 | ❌ 89.1 vs fedbuff 64.5 |
| C4 Felix least affected (Δ max vs syn_0) | – | ❌ −3.4 pp vs oort_star +0.4 | ❌ −7.3 (@1.0 −3.9) vs refl −0.2, feddance +1.9 | – | ✅ −3.5 vs fedbuff −16.1 | ❌ −11.8 vs sync four −1.5…+3.3 |

- *Heterogeneity:* untested ⬚. Every run is Dirichlet α 0.1; no α sweep exists. Needed for C4 (G3 sims, α ∈ {0.1, 0.5, 1.0}).
- *Why Felix's comm is high:* traffic grows with time for async stacks (c = 15 always in flight) and Felix ships a full model with
  every eval task (cifar 1,098 eval dispatches = +11% downlink, replies ~0.2 KB). Sync Oort moves 11 models per ~18 s round, so it is
  cheaper per minute; Felix wins comm only by reaching the target much sooner (C3 at @1.0).
- *Why C4 fails:* 1-mini-batch baselines (refl, feddance) never see the full shard, so streaming can't hurt them; they also never
  reach target (33-43% cifar, 15-22% speech). C4 should be scored among baselines that reach target, or on TTA, not on max.

**Communication (real legs; model 2.15 MB cifar, 29.0 MB speech; GB total down / up, GB down+up to 40% @ min):**

| baseline | cifar syn_0 | cifar syn_50_3st | cifar lin | speech syn_0 | speech syn_50_3st | speech lin |
|---|---|---|---|---|---|---|
| felix | 24.1 / 21.8 · 28.1 @72 | 23.9 / 23.2 · 30.5 @78 | 24.8 / 22.3 · 45.0 @114 | 65.9 / 63.5 · 41.5 @36 | 67.9 / 66.4 · 59.1 @52 | 68.8 / 65.9 · 89.1 @78 |
| felix @1.0 (sim) | 22.2 / 20.2 · 9.9 @24 | 23.2 / 22.6 · 16.1 @39 | 23.5 / 21.2 · 22.3 @57 | – | – | – |
| fedbuff | 20.5 / 20.5 · 18.5 @54 | 17.8 / 17.7 · 16.5 @56 | 20.5 / 20.5 · 34.5 @101 | 59.6 / 59.1 · 41.3 @42 | 25.0 / 23.9 · 41.5 @102 | 59.6 / 59.1 · 64.5 @65 |
| oort | 9.3 / 9.3 · 8.5 @31 | 6.0 / 5.9 · 10.0 @108 | 13.1 / 13.1 · 17.1 @59 | 30.6 / 30.4 · – | 6.4 / 5.9 · – | 34.5 / 34.3 · – |
| oort_star | 9.3 / 9.3 · 9.0 @35 | 9.8 / 9.8 · 8.1 @43 | 13.5 / 13.4 · 16.1 @47 | 31.3 / 30.9 · – | 15.6 / 15.3 · – | 35.1 / 34.8 · – |
| feddance | 16.1 / 16.1 · 31.9 @119 | 17.8 / 17.8 · – | 15.6 / 15.6 · 29.1 @112 | 46.0 / 45.9 · – | 30.3 / 29.8 · – | 46.0 / 45.8 · – |
| refl | 9.4 / 9.4 · – | 8.7 / 8.6 · – | 9.4 / 9.4 · – | 39.6 / 39.2 · – | 21.8 / 21.2 · – | 39.6 / 39.3 · – |

To 50% (cifar): fedbuff 36.6 GB @107 (syn_0); felix @1.0 20.4 @53 / 22.7 @57 / 38.2 @101. To 60% (speech syn_0): felix 124.3 @115,
fedbuff 110.8 @112. EuroSys plots count comm as msgs × 1 MB (eval reply 0.01 MB); bytes here are measured on the wire.

### Running config (PR37 G3) and invariants checked

- *Shape (`harness_pool.G3_SHAPE`, all six):* cifar n:c:K 75:15:10 on 1 A40, speech 50:15:10 on 2; the reference split's first n
  trainers (Dirichlet α 0.1); 2 h real / 2 h vclock; eval every 10 versions on the full test set; `trainerDeterministic`; speech bf16
  (FX-D168) and `device_time_scale` 5 (D ~5× cifar's; FX-D46); Oort-family over-commit 1.1, REFL/FedDance picks round(K × 1.3).
- *Task timeout:* `send_timeout_wait_s` 90 s cifar / 450 s speech for every non-Felix baseline. Verified on all syn_50_3st legs: real
  abandons at 90.0-90.1 s / 450.0-450.2 s after dispatch, then the slot re-selects. Felix never times out there: it evicts at the
  availability flip (`aware_boundary_eviction`, median 5 s cifar / 31 s speech after dispatch).
- *Availability handling:* select filter = felix, oort_star, refl, feddance (pick only AVL_TRAIN); unaware = oort, fedbuff (pick
  anyone, upload held until AVL_TRAIN or abandoned at the timeout). Proactive eviction = Felix only (FX-L11). AVL_EVAL is eval-only
  and Felix-only (FX-D165).
- *Training/server knobs (`baseline_reference.yaml`; final = source-faithful or recorded deviation, L33-L37):*

  | baseline · dataset | local work | batch | client lr / opt | mom · wd | lr decay | server | utility | status |
  |---|---|---|---|---|---|---|---|---|
  | felix · cifar / speech | 1 epoch | 32 / 16 | 0.04 SGD | 0 · 0 | none | fedbuff, **lr 1.0** (FX-D185; PR37 ran 0.3 / 0.5), rate `new` (scale 0.4, a 0.25, b 0.1) | fedscale | lr final; selector knobs untuned (FX-N108) |
  | fedbuff · cifar / speech | 1 epoch | 32 / 16 | 0.04 SGD | 0 · 0 | none | fedbuff, lr 1.0, rate 1/√(1+s), lr batch-normalize | – | final (FX-D73, FX-N74) |
  | refl · cifar / speech | 1 step | 10 / 20 | 0.1 / 0.05 SGD | 0.9 · 5e-4 | 0.98 / 10 | REFL SAA (speech: FedAvg server, recorded) | – | final (deviations recorded) |
  | oort, oort_star · cifar / speech | 20 steps | 30 / 15 | 0.04 SGD | 0.9 · 5e-4 | 0.95 / 5 · 15 | FedAvg-YoGi η 0.005 / 0.002 (cifar clip 1.0) | fedscale | final (deviations recorded) |
  | feddance · cifar / speech | 1 step | 24 / 32 | 0.1 / 0.075 SGD | 0.9 · 5e-4 | 0.98 / 10 | FedAvg equal | – | final |

  Felix vs FedBuff (FX-D185): client side and server lr identical, so the pair
  differs only in Felix's selection (utility + speed, availability-aware, eval tasks), its rate (staleness + utility, mean 0.90 vs
  0.69) and FedBuff's last-batch lr normalization. Felix's rate constants and exploration knobs are 2024 values, never tuned on this
  model.

### PR37 training dynamics (pass 1; `run_*_G3_*`, per-leg extract of aggregator telemetry)

Leg = 2 h on its own clock (real wall since trace origin, sim vclock). Versions = committed model versions. Contributors = trainers
whose update was aggregated (of n). Top-20% = share of all updates from the 20% most-used trainers. Staleness = mean versions behind.
t→X = first eval ≥ X (min). Cells read real / sim. Targets: cifar 50%, speech 60%.

*cifar10 (n:c:K 75:15:10, 1 GPU):*

| baseline · cell | versions | max acc % | acc @60 min % | t→40% | t→50% | contrib. | top-20% | stale | sim × |
|---|---|---|---|---|---|---|---|---|---|
| felix · syn_0 | 1012 / 1015 | 48.9 / 47.8 | 36.2 / 34.8 | 72 / 68 | – | 75 | 57% | 1.40 | 26× |
| felix · syn_50_3st | 1081 / 1084 | 45.5 / 45.7 | 35.5 / 35.0 | 78 / 83 | – | 75 | 43% | 2.30 | 27× |
| felix · lin | 1037 / 1028 | 41.7 / 40.6 | 26.3 / 24.8 | 114 / 116 | – | 75 | 58% | 1.40 | 26× |
| fedbuff · syn_0 | 954 / 954 | 52.1 / 53.3 | 37.1 / 37.0 | 54 / 54 | 107 / 107 | 75 | 23% | 1.40 | 40× |
| fedbuff · syn_50_3st | 823 / 825 | 50.0 / 50.1 | 37.0 / 36.9 | 56 / 57 | – / 104 | 75 | 31% | 4.38 | 44× |
| fedbuff · lin | 953 / 954 | 44.5 / 45.2 | 26.3 / 26.5 | 101 / 96 | – | 75 | 23% | 1.40 | 41× |
| refl · syn_0 | 379 / 381 | 32.9 / 33.6 | 23.8 / 23.8 | – | – | 75 | 23% | 0.14 | 136× |
| refl · syn_50_3st | 346 / 361 | 30.6 / 30.5 | 23.2 / 24.8 | – | – | 75 | 30% | 0.52 | 147× |
| refl · lin | 379 / 382 | 32.8 / 33.7 | 22.7 / 21.0 | – | – | 75 | 23% | 0.14 | 135× |
| oort · syn_0 | 391 / 395 | 48.3 / 49.3 | 46.3 / 44.1 | 31 / 34 | – | 74 | 61% | 0 | 53× |
| oort · syn_50_3st | 242 / 200 | 43.5 / 42.2 | 24.7 / 25.8 | 108 / 117 | – | 73 | 60% | 0 | 92× |
| oort · lin | 551 / 665 | 41.4 / 39.0 | 40.2 / 35.9 | 59 / – | – | 72 | 69% | 0 | 33× |
| oort_star · syn_0 | 394 / 390 | 49.6 / 48.4 | 45.9 / 44.8 | 35 / 38 | – | 74 | 59% | 0 | 54× |
| oort_star · syn_50_3st | 410 / 407 | 50.0 / 47.9 | 40.7 / 40.1 | 43 / 43 | – | 74 | 54% | 0 | 50× |
| oort_star · lin | 568 / 383 | 43.2 / 45.4 | 41.3 / 39.3 | 47 / 46 | – | 71 | 68% | 0 | 56× |
| feddance · syn_0 | 576 / 576 | 40.6 / 39.5 | 30.9 / 28.0 | 119 / – | – | 53 | 68% | 0 | 82× |
| feddance · syn_50_3st | 634 / 605 | 39.0 / 41.2 | 26.5 / 28.5 | – / 110 | – | 49 | 63% | 0 | 83× |
| feddance · lin | 557 / 569 | 42.5 / 40.9 | 33.7 / 31.4 | 112 / 107 | – | 53 | 69% | 0 | 87× |

*google_speech (n:c:K 50:15:10, 2 GPUs):*

| baseline · cell | versions | max acc % | acc @60 min % | t→40% | t→60% | contrib. | top-20% | stale | sim × |
|---|---|---|---|---|---|---|---|---|---|
| felix · syn_0 | 219 / 222 | 62.5 / 63.4 | 43.5 / 48.3 | 36 / 36 | 115 / 113 | 50 | 39% | 1.39 | 4.3× |
| felix · syn_50_3st | 229 / 228 | 59.0 / 56.7 | 44.9 / 44.5 | 52 / 52 | – | 50 | 36% | 2.37 | 4.4× |
| felix · lin | 227 / 221 | 50.7 / 51.6 | 32.2 / 34.0 | 78 / 80 | – | 50 | 40% | 1.39 | 7.9× |
| fedbuff · syn_0 | 204 / 204 | 63.0 / 65.2 | 42.6 / 41.7 | 42 / 42 | 112 / 118 | 50 | 25% | 1.39 | 5.0× |
| fedbuff · syn_50_3st | 83 / 83 | 46.9 / 44.8 | 28.3 / 29.4 | 102 / 103 | – | 50 | 29% | 2.45 | 9.6× |
| fedbuff · lin | 204 / 204 | 49.8 / 50.9 | 31.0 / 31.8 | 65 / 77 | – | 50 | 25% | 1.39 | 8.6× |
| refl · syn_0 / 50_3st / lin | 104 / 105 · 53 / 56 · 104 / 105 | 16.1 / 18.5 · 5.8 / 4.8 · 14.5 / 11.5 | ≤ 7.3 | – | – | 50 | 26-31% | 0.25-0.54 | 46-74× |
| oort · syn_0 / 50_3st / lin | 95 / 97 · 13 / 11 · 107 / 109 | 18.7 / 22.0 · 3.7 / 1.5 · 19.3 / 22.1 | ≤ 17.2 | – | – | 45-49 | 32-42% | 0 | 16-76× |
| oort_star · syn_0 / 50_3st / lin | 97 / 97 · 44 / 51 · 109 / 115 | 17.6 / 22.1 · 6.3 / 6.5 · 20.9 / 20.6 | ≤ 15.1 | – | – | 47-49 | 34-41% | 0 | 15-31× |
| feddance · syn_0 / 50_3st / lin | 121 / 122 · 76 / 76 · 121 / 124 | 19.3 / 19.3 · 12.7 / 8.1 · 21.7 / 18.2 | ≤ 10.5 | – | – | 34-36 | 52-55% | 0 | 31-50× |

Real↔real band, the one measured cell (cifar fedbuff syn_50_3st, spare shepherd real): versions 822 vs 823, max 48.9 vs 50.0%, final
46.6 vs 49.0%, t→40% 65 vs 56 min (2.4 pp, 16%). Sims outside that spread: the FX-N103 pair, cifar oort syn_50_3st versions −17%,
cifar oort_star syn_50_3st final −4.2 pp (Oort loss spikes; max −2.1), cifar feddance lin final −2.6 pp, speech fedbuff lin t→40% +18%.
Pass 2 sets the per-cell bands that grade them.

**What moves accuracy and time to accuracy (real legs; sims agree):**
- *Unavailability (syn_50_3st vs syn_0) hurts by awareness and task length.* Aware selectors keep their pace: Felix +5-7% versions on
  both datasets (accuracy −3.4 / −3.5 pp), oort_star +4% on cifar. Unaware ones lose rounds to abandons and gated uploads: cifar oort
  −38% versions (−4.8 pp), fedbuff −14% (−2.1 pp, staleness 1.4 → 4.4), refl −9%. Speech is far harsher (FX-N106): fedbuff −59%
  versions and −16 pp (t→40% 42 → 102 min), refl −49%, oort −86% (13 versions, 3.7%). Selection eligibility falls to 45-52% of n.
- *Dataset.* Speech tasks are ~5× longer (`device_time_scale` 5), so 2 h gives ~5× fewer versions (felix 1012 vs 219) but more accuracy
  per version (62.5% at 219). The sync four are round-bound on speech (15-22% max at 95-121 versions) while cifar Oort reaches 48-50%.
- *Streaming (lin, data 10% → 100% over 2 h) costs accuracy, not versions,* for multi-step learners: max −7 pp cifar (felix, fedbuff,
  Oort family) and −12 to −13 pp speech (felix, fedbuff); acc @60 min ~−10 pp. 1-mini-batch baselines barely change (refl ±0, feddance
  +1.9 cifar): they never use the full shard. Cifar Oort-family under lin commits more versions (pacer holds a short preferred duration,
  FX-N103).
- *Baselines.* Cifar: fedbuff alone hits 50% in 2 h (107 min); Oort / oort_star are fastest early (40% at 31-35 min vs fedbuff 54, felix
  72) then plateau at 48-50%; felix 48.9% at its untuned server lr 0.3 (at 1.0: 55.9%, 50% at 53 min; FX-D185); feddance 40.6% with 53 / 75 contributors (FX-N90); refl 33% (1 mini-batch per
  round). Speech syn_0: felix ≈ fedbuff (60% at 115 / 112 min); syn_50_3st: felix 59.0% vs fedbuff 46.9%, Felix's clearest win.
- *Selection concentration.* Felix's exploit set gives its top 20% of clients 57% of updates (cifar) vs FedBuff's 23%; Oort family
  59-69%; feddance 63-69%. Unavailability spreads Felix's picks (43%).
- *Sim speed (S).* Never < 4×: cifar 26× (felix) - 147× (refl); speech felix 4.3×, fedbuff 5.0× (both scale with commit count), sync
  four 15-76×.

**Cross-cutting capabilities**

| capability | status | item |
|---|---|---|
| checker catches injected sim bugs (P11a-c) | ✅ 6/6 CAUGHT both datasets (PR28 C3) | FX-D125 kept sensitivity |
| fail fast; ladder runner + offline re-gate; early DOOMED kill | ✅ | FX-D22, FX-D133, FX-N42 |
| GPU join/warm-up at n=100; sync wait-K | ✅ joins 44-99 s; EV7 green | FX-D19-D21 |
| real async ingest < 2 s | ✅ G3 queue_wait p99 ≤ 0.53 s, max 1.1 s | FX-D16 |
| profiled sim charges per dataset/platform | ✅ re-derived on FX-D141/D142 code, T3 syn_0 32/32 | FX-D143 |
| DIST tolerances from replicate floors | 🟡 n=2 floors gate regrade; none tightened | FX-D37, PARITY Q2 |
| real == sim with deterministic kernels | ✅ V2; PR37 G3 felix/fedbuff versions within 3% on 12/12 cells | FX-D144/D145/D149/D158 |
| no-overrun rule (PL19) | ✅ PR37 0 / 311k train tasks | FX-D180 |
| streaming / oracle experiment | 🟡 EV19 green both datasets; oracle advantage ungraded | FX-N13 |
| sim speedup vs real | ✅ PR37 G3: cifar 26-147×, speech 4.3-76× | FX-D180 |
| paper experiments sim-only | ⬚ | FX-N12 |

**Accuracy table (FX-N74; `scripts/accuracy_table.py --runs <dirs>`).** Parity target cifar 50%, speech 60% on full data, syn_0, reference
n, the leg's own clock. G1A = full data (runs 16-18); G2 streams data 0 → 100% over 3 h (lower bound). `-` = no eval in window.
Sync-four numbers are G1AS on source-faithful configs (FX-D100/D104/D105); older G1A rows were void and are dropped. PR37 G3 (n=75 / 50,
2 h): dynamics table above.

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
| felix · cifar / speech | ✅ / ✅ | 🟡 / 🟡 PR37 pass 1 inside band | 🟡 / 🟡 G3 syn_50_3st INV/EXACT ✅ | ✅ 26× / ✅ 4.3× |
| fedbuff · cifar / speech | ✅ / ✅ | 🟡 / 🟡 (speech lin t→40% +18%) | 🟡 / 🟡 | ✅ 40× / ✅ 5.0× |
| oort, oort_star · cifar | ✅ ✅ | ❌ lin (FX-N103); syn_0 🟡 | 🟡 (oort versions −17%) | ✅ 53× 54× |
| refl, feddance · cifar | ❌ ❌ (29.5, 35.6%) | 🟡 | 🟡 | ✅ 136× 82× |
| oort, oort_star, refl, feddance · speech | round-bound: needs C (same accuracy, same time) | 🟡 (accuracy ≤ 22%, 10-13 evals) | ❌ degenerate (FX-N106) | ✅ ≥ 15× |

🟡 = provisional on PR37 pass 1 (one real per cell; exceptions under the dynamics tables); pass 2 grades each cell on its band.
**Convergence 0 / 12** (A 6/12 · C 0 · U 0 · S 12/12: PR37 G3 sims 4.3-147×).

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
- **Long runs launch through `scripts/dispatch.py` (operator 10-17):** a batch YAML of harness_pool stages; never hand-split
  stages per node again (PR36/PR37 shepherd ran long twice and was offloaded by hand).
- **Felix claims (10-17):** all six reach target; Felix's time-to-target is significantly lowest and its up/down traffic to target
  lowest in every scenario; Felix is least affected by streaming, heterogeneity and unavailability. Any miss is a ⚠️ warning on the
  Scoreboard ("Felix claims") and blocks FX-N12.
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
| G3 / G3C | PR37: six × {syn_0, syn_50_3st, lin}, 2 h real + 2 h-vclock sim, eval every 10 rounds, deterministic; `G3_SHAPE` cifar n:c:K 75:15:10 on 1 GPU, speech 50:15:10 on 2 / real replicate | ~2.2 h per real |
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
- **FX-L79** Grade Oort-family streaming cells on ≥ 3-real floors: utility ∝ visible |D| grows, so the pacer's ≤ 10% flat test is borderline
  every window and its T ramp is a draw (PR37 cifar lin: six sims 361-665 versions; FX-N103).
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
- **FX-T45** Don't snapshot node state once per pool (GPU list, RAM budget) or let a probe pass on 0 cells: PR29b B1 idled 5.5 h and read
  PASS (FX-D141); PR37 shepherd S1 kept a 205 GB start-time RAM budget for 20 h (FX-D179).
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
- **FX-T53** Don't re-tune one async stack's server lr alone: FX-D73 moved FedBuff to 1.0, Felix stayed at 0.3 and ran 8 pp behind
  for a week of batches (FX-D185). Tune Felix's own knobs by the same method in the same change.
- **FX-T52** Don't pass `--gpu-share` for legs with no footprint history: an unmeasured leg claims a whole GPU and `place_gpus` rejects
  any GPU with a live context (4 MiB > the live cap), so the pool idles ("no GPU room", 26 min lost, S1J 10-16). Drop the flag, or fix
  the live-cap tolerance in a shared change (changes the code key).

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
- **FX-D187** A grade reserves `GRADE_RAM_GB` (8 GB): with `--lease-from-telemetry` it inherited its leg's measured peak (~90 GB in
  PR37) and blocked waiting legs.
- **FX-D186** `scripts/dispatch.py BATCH.yaml [--plan]`: one batch over jayne + shepherd. Each node expands every stage itself
  (`harness_pool --jobs-json`: own history, cores, GPUs, RAM budget); units (a pair + its grade) go longest-first to the node with the
  earliest finish, list-scheduled per leg (CPU, GPU, RAM; a sim frees its share when it ends, a grade waits for its pair); a unit is released to that node's pool (`--queue`) only within 30 min of its planned
  start and the plan is redone each minute from every pool's `status.json`, so a slow node gets fewer remaining units. Health per 5 min:
  aborted / STALLED / DOOMED, legs > 1.25 × estimate, foreign GPU processes, host RAM < 10% free. Ctrl+C sends ABORT to every pool; at
  the end remote pools + run dirs are copied here. Preflight refuses differing commit / code key (PL18). `tests/harness/test_dispatch.py`.
  Smoke 10-17 (`dispatch_dsmoke`, 8 G3 pairs at `--scale-smoke 240`, both nodes): pools launched remotely, units released per plan,
  11 legs rc 0, 4 pairs graded (EV PASS), no OOM / RAM / overrun flag; the 15-min cap exercised Ctrl+C (ABORT, clean teardown on both);
  `collect()` copied shepherd's pools + run dirs.
- **FX-D185** Felix server lr 1.0 on both datasets = FedBuff's (FX-D73 / FX-N74; was the 2024 0.3 / 0.5): Felix and FedBuff now differ
  only in Felix's selection, its staleness+utility rate and FedBuff's last-batch lr normalization (evidence in
  `baseline_reference.yaml`).
- **FX-D184** U6 mean tolerance × max(1, task timeout / 90 s): speech lags scale with its device time (`mean_tol` in the JSON).
- **FX-D183** Real eval spans max(gpu, D/20) as sim charges (sleep D/20 − measured eval time; was floor(D/20), 0 s for D < 20 s).
  `realEvalDelayFloor=true` reverts.
- **FX-D182** Starved sync sims (syncfl, Oort) wake at min(next flip, next pick timeout) (`_sim_starvation_wake`): picks were abandoned at
  the next flip, up to 80 s late (cifar oort syn_50_3st sim 200 vs real 242 versions → 232). `sim_abandon_wakes=false` reverts.
- **FX-D179** `harness_pool` gates RAM only on the live node-wide check (MemAvailable ∧ MemTotal − all leases); the per-pool budget fixed
  at start capped PR37 shepherd S1 at 205 GB for 20 h. `felix_claims.py` (FX-N108 grader) and `fl_lr_check.py` (pair server lr now
  honoured under `--baseline`) sit outside the code key.
- **FX-D181** `parity_ladder` pairs G3C reals with their G3 cell for floors and per-real votes (G3_syn_50_3st takes the syn_0 band); G3C is
  real-only; `readiness_score` files G3 lin cells under stream_lin (`test_g3c_replicate_floors_its_g3_cell_and_syn_50`).
- **FX-D180** No-overrun rule + sim speed (FX-N98): bf16 speech (FX-D168), balanced placement (FX-D169), PL19 graded; PR37 G3: 0 / 311k
  train tasks overran D (real + sim); sims cifar 26-147×, speech 4.3-76× real.
- **FX-D171** Felix utility = Oort's (`stat_utility: fedscale`, both `felix` rows): |D|·√(EMA of batch-mean L²) over the first pass
  (was `legacy`: first mini-batch only). A/B G3 cifar sims: accuracy unchanged (FX-N104).
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
  can't fail it. Accept (graded on G1/PR37 legs: G3 felix syn_50_3st 264 / 277 eval picks, EV21 PASS), or arm at ≥ 1?
