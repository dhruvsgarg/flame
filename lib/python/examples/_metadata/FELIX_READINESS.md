# Felix readiness — backprop FL (async_cifar10, google_speech)

Read [ROBUST_FL_READINESS.md](ROBUST_FL_READINESS.md) and [PARITY_READINESS.md](PARITY_READINESS.md) first; their rules, L and T apply
here unrepeated. **Focus (operator 10-19): improve Felix itself (FX-N109: syn_0, then syn_50_3st; α 0.1; PR39a / PR39b done, PR39c1 next), parity parked at 76 / 76;
the next PR39 runs on the optimized Felix, then the claims gate (FX-N108) and sim-only experiments (FX-N12).** Order: Next steps → Scoreboard (now, and expected after the next run) → Decisions → reference sections.

**At a glance (10-18):** golden rule ROBUST R0 (short screens first). PR38 (key `44d44bd2d67b`, 60 units, both nodes via the dispatcher,
18.7 h) ran clean: EV PASS on all 96 legs, 0 crashes, 3 / 390k train tasks overran D. Felix and FedBuff are real == sim on every cell
(versions within 1%, time to target inside the 2-real band); Felix now has the lowest time to target in all six scenarios (FX-D185).
10-18: **parity 76 / 76** (mixed key): G3S sim-only votes, then FX-D194 (sync barrier lands a gated delivery due mid-barrier)
closed feddance speech syn_50_3st on G0US (9 / 9 votes). Open: PR39 re-votes the feddance cells on key `b5bb7b240c90`, a real > sim
draw-like lean on cifar lin oort (FX-N103), Felix claims C2' / C3' on cifar (FX-N108; C1 ✅ 10-19).

## Next steps (persistent queue — top item is next)

Work rule: PARITY C10, C17, C18. Test levels: pytest (~3 min) · CPU tiers T1-T4/TS · GPU ≤ 30-min screens (G0U/G0T/G0UC) · long
(G1*/G2/G3). Unblock map: 76 / 76 (10-18) → FX-N109 optimized Felix → PR39 on it → FX-N108 → FX-N12.

- **FX-N109 · Optimize Felix (Active build "Felix design") · wip.** E1 (PR39a) and E2 (PR39b) done: TTA is set by the delay-adjusted
  step κ = η w̄ c / K up to a per-dataset edge (Formulation). The sensing audit found Felix picks by shard size, on a duration model
  where size was free (fixed by FX-D197), and that eval and train utilities measure different losses. *Built (pytest 2761 green):* I5 AIMD on K
  (K = B refuted, FX-T59), I7 `norm` rate, I8 update-counted eval gate, I9 utility on receive, `utilityPerSample` (FX-D195);
  `selection_audit.py`, tiers F39C / F39S, `batches/pr39c1.yaml` / `pr39c2.yaml` / `pr39s.yaml` (FX-D196). *Smoke (10-20, 10-min vclock,
  seed 1, `async_cifar10/experiments/run_20261020_00*_F39C_*` (pool dirs lost)):* every leg EV PASS after the EV3 fix. f2's K walks 10 → 2, backs off on a loss spike,
  settles at 3-4; dkc (old weight) goes 10 → 15 (an early false backoff) → 6. I9 cuts cifar eval-belief bias 1.19 → 0.96 (error
  0.21 → 0.09). Speech smokes are too short to reach an eval or a stable AIMD window. *Next:* (1) operator launches
  PR39c1 on shepherd GPUs 0-3 (4-7: pbehnam3) (`dispatch.py batches/pr39c1.yaml --nodes shepherd`, tmux `dg_flame`; 33 Felix legs,
  plan ~2.8 h); (2) read it with `knob_sweep.py` + `selection_audit.py`; (3) PR39c2 (the remainder + FedBuff controls), then PR39s. *Exit:* E1-E5 of the design.

- **PR39 · Hold 76 / 76 on the FX-D194 key + band strengtheners · blocked: FX-N109 (operator 10-19: re-target on the optimized Felix).** Commit, push; shepherd pulls; then
  `dispatch.py batches/pr39.yaml --plan`, `dispatch.py batches/pr39.yaml` (tmux `dg_flame`). Stages: S speech G3S1 all six × syn_0 /
  syn_50_3st / lin; C cifar G3S1 lin for felix, fedbuff, refl, feddance; CF cifar feddance G3S1 syn_50_3st (the one G3 cell FX-D194
  moves that S/C miss; syn_0 and lin have no gated deliveries); D felix, fedbuff G3 syn_50_3st both datasets (re-derived
  gate_release); RA a third G3C syn_50_3st real for cifar oort, oort_star. Key `b5bb7b240c90` (linked to the G0US key in
  `REAL_SAME_CODE`); a further run-affecting edit before launch needs a new link. *Claim:* every cell stays green, 76 / 76 on mixed
  keys. Refute: a G3S red on a cell green today. *Then:* rewrite the Scoreboard, FX-N108.

- **FX-N103 · Cifar lin oort: real > sim lean · wip (cell green).** G3S `pool_20261018_033638_G3S`: sims 455 / 543 / 669 versions vs 2 h
  reals 551 / 625 / 694 (floor 17-40%), 7 of 9 pairs real > sim. 10-18 audit, no biased input found: (a) sim S3 ≡ real 145243 for 15
  rounds (observed durations Δ 0.001 s, utilities identical); (b) fork at v16 is a ms tie at a whole-second budget (0420 vs 0375, both
  5.001 s), as reals fork among themselves (v17); (c) aggregator cost real 54-66 vs sim 30 ms/cycle leans the wrong way; (d) 10-min pair
  `pool_20261018_031320_G3` 79 / 79 rounds, advance 7.34 / 7.39 s. Post-fork sims drift to slower picks (mean observed 4.2-11.4 s vs real
  3.8-9.1 s). *Next:* more draws, not fixes: ≥ 3 more lin oort sims in the sim-only phase. *Exit:* no one-sided lean over ≥ 6 sims × 3 reals.
- **FX-N108 · Felix claims gate (operator 10-17) · wip, blocks FX-N12.** PR38 (Felix server lr 1.0): C2 ✅ in all six scenarios; C4 ✅;
  left (Scoreboard "Felix claims"):
  - *C1 · every baseline reaches target · ✅ 10-19:* all six scenarios under the round-bound rule (Decisions; operator extended it to
    cifar 10-19, `felix_claims.ROUND_BOUND`): every below-target leg reads `learning` (cifar refl 31-34%, feddance 40-42%, oort /
    oort_star 39-50%), unaware speech syn_50_3st legs `degenerate` (abandon at 450 s, re-dispatch ≤ 0.1 s). Regraded on PR38.
  - *C2' / C3' · to 40% on cifar:* sync Oort reaches 40% at 25-43 min moving 11 models per ~18 s round (5.5-15 GB); Felix 28-59 min,
    11-24 GB (c = 15 always in flight + a full model per eval task, +11% downlink). Lever: FX-N109 (K5c15 flips C2'; E3c targets C3).
  - *C4 heterogeneity:* untested; out of scope while every run is α 0.1 (operator 10-19).
- **FX-N100 · Throughput: utilization-aware packing (central goal, operator 10-15) · wip.** PR38 (`--util`): jayne 3.4 legs/h,
  concurrency 4.6, CPU 2%, RAM peak 437 / 504 GB, GPU memory 56%, idle 17%; shepherd 1.8 legs/h, concurrency 2.4, RAM peak 394, idle 3%.
  Binding: host RAM (G3 legs reserve 80-100 GB). Speech sims are GPU-slot-bound (FX-L81): PR38 gave each 2 GPUs (no `--gpu-share`),
  felix 44 min (2.7×) vs PR37's 30 min on 5 GPUs. *Next:* `--gpu-share` for speech sims; FX-N84's fork-server import sharing
  (~280 MB/trainer); default `--gpu-cpus-per-trainer` 0.4 → 0.1 after PR39 (changes the key): G0US n=50 GPU sims leased 28 cores,
  used avg 0.9-2.4 / p95 ≤ 5.5, so 4 legs filled 120 CPUs with GPU 7 idle. *Exit:* a 10-min mixed batch at ≥ 2× PR36's legs/h with
  interference at the solo baseline (PL19, `CPU_SAT`, sim slowdown ≤ 1.25×, EV/parity = solo).
- **FX-N94 · tiny_cpu first-contact sim delivery lag · todo.** `smoke_fxd160_0010`: first-contact lag 1.1-1.7 s (real ~0.03 s), sim
  clock +1.7 s by round 3; GPU legs ≤ 0.5 s. *Exit:* first-contact lag ≈ later tasks' on tiny_cpu.
- **FX-N85 · Close PR27/PR28 roots · wip.** (b) cifar Oort-family stub legs run one real step (FX-D110 extended); (c) A5 builder:
  apply FX-D125's observed-window rule to `state_timeline_agreement`. *Exit:* every listed cell green or floor-gated.
- **FX-N92 · Felix insights for the paper (operator 10-12) · wip.** [FELIX_INSIGHTS.md](FELIX_INSIGHTS.md) Next steps IN1, IN2, IN4.
  *Exit:* findings for every catalog row, real↔sim agreed.
- **FX-N90 · FedDance coverage under over-commit · todo.** Literal Alg. 1 l.3 zeroes picked-but-never-aggregated devices forever (F3
  syn_0: 104/300); `straggler_score: mean` restores coverage but biases to slow ends. *Exit:* a cited rule with coverage ≈ random and
  no speed bias, or recorded as FedDance's own.
- **FX-N91 · EuroSys plot debts (no runs now, operator 10-12) · todo.** (a) controls: G1X cifar syn_0 `fedavg` sim 20 h; G1X speech
  mobiperf `felix`; (b) a `straggler_score: mean` arm (FX-N90); (c) REFL label + ×1.06 factor; (d) MobiPerf e2e table vs logs (OORT* 15.6
  vs 9.45 h; REFL mobiperf has no log); (e) speech e2e May-2025 rows have no logs; (f) comm to 70% from counted messages, not time share.
- **FX-N84 · Trainer memory audit · wip.** PSS/trainer cifar 1052 MB, speech 1160 MB after lazy mlflow + own coords; aggregator
  `malloc_trim` after eval. Left: `import torch` ~280 MB, cuDNN/cuBLAS host ~390 MB (fork-server sharing: FX-N100). *Exit:* footprint
  table, fixes, new caps.

*Short-run queue (fix + verify with pytest, stored telemetry, CPU tiers or GPU screens):*
- **FX-N110 · Speech sim gate failsafe under GPU contention · todo.** PR39a `gs_F39A_c30_r1`: trainer 0388 took 11 min wall for a ~4 s
  train (06:55-07:06, its GPUs ~98% SM with packed legs at `--gpu-share 2`); `SIM_GATE_FAILSAFE` dropped it at 457 s (round 143), then
  128 pastdated commits (EV11 2.8%), 1 redispatch (EV10); c30_r2 same; EV13 stalls on ctl_r1 (304 s), c10_r1 (141 s), and PR39b
  `gs_F39B_K10c15_r1` fedbuff (the only non-PASS of 69 legs). Root the 11-min hang, then a solo vs packed c30 screen. *Exit:* speech
  c30 EV-clean when packed.
- **FX-N9 · Unavailability, all six × syn_50_3st / mobiperf_3st · wip.** syn_50_3st 12/12, mobiperf 10/10 (mixed key; PR39). *Exit:* both
  columns green on one code state.
- **FX-N13 · Streaming, both datasets · wip.** stream_eve 12/12, stream_lin 12/12; left: G0To (oracle arms), ST6.
- **FX-N30 `[S]` · Speech streaming on GPU screens · wip.** G0T + G3 lin green on all six; left: G0To.
- **FX-N74 `[C]` · Every baseline reaches target on full data · wip: 5 / 12.** Accuracy table below; folds into FX-N108 C1.
- **FX-N76 `[C][S]` · C13 real-cost audit · wip.** `trainer_speed_identity` red only where real overran D; real speech FedBuff
  arrival → commit 0.12 s mean (sim 0): profile and charge it if it moves K2.
- **FX-N42 `[S]` · Parity ladder · wip (PARITY Q2-Q6).** *Exit:* a run graded per cell on both axes, DIST on ≥ 3-leg floors.
- **FX-N10 · google_speech on the launcher · wip.** *Next:* S4 cleanup.
- **FX-N22 · Fast parallel harness · wip.** ISO oort,oort_star solo vs packed at cpt 0.5 / 0.25 (`harness_iso_compare.py`).
- **FX-N2 · Parent S2 pipeline for async_cifar10 · todo.**

*After 76/76 — convergence and accuracy:*
- **FX-N93 · Convergence parity at the reference config · todo.** One C19 block per step: (1) G1AS six × both datasets: target or
  audited round-bound; (2) G1A real↔sim pairs + real replicate at reference n, deterministic kernels (time-to-target rung, C7); (3) G1U
  syn_50 + mobiperf 3 h; (4) sim speed ≥ 5× at reference n (S6); (5) FX-N12. Graded on the convergence board.
  **FX-N12 · paper experiments sim-only · blocked: PR39, FX-N108.**

---

## Scoreboard (rewrite after every run, PARITY C5)

**Reset 10-20 (FX-D197, operator): task time is now the FedScale device model. Every earlier result (parity 76 / 76, PR38 G3 cells,
Felix claims, accuracy, convergence, PR39a / PR39b fits) ran on work-independent durations and is void; none carries over (git keeps it).**

| board | state on the FX-D197 key |
|---|---|
| parity, 76 cells (PARITY C5) | ⬚ re-graded by PR39 |
| Felix claims C1-C4 (Decisions 10-17) | ⬚ after FX-N109 |
| accuracy (FX-N74), convergence (FX-N93) | ⬚ |

**Expected after the duration smoke (10-20, shepherd):** Felix real == sim on a 15-min cifar G3 pair; EV PASS on cifar / speech sims.

### Running config (G3) and invariants checked

- *Shape (`harness_pool.G3_SHAPE`, all six):* cifar n:c:K 75:15:10 on 1 A40, speech 50:15:10 on 2; the reference split's first n
  trainers (Dirichlet α 0.1); 2 h real / 2 h vclock; eval every 10 versions on the full test set; `trainerDeterministic`; speech bf16
  (FX-D168); Oort-family over-commit 1.1, REFL/FedDance picks round(K × 1.3).
- *Task time:* FedScale device model (Scope, "Task-time model").
- *Task timeout:* `send_timeout_wait_s` 903 s cifar / 2560 s speech (2 × the longest FedScale task in the reference population,
  `datasets.yaml`) for every non-Felix baseline;
  Felix evicts at the availability flip (`aware_boundary_eviction`).
- *Availability handling:* select filter = felix, oort_star, refl, feddance; unaware = oort, fedbuff (upload held until AVL_TRAIN or
  abandoned). Proactive eviction = Felix only (FX-L11). AVL_EVAL is eval-only and Felix-only (FX-D165).
- *Training/server knobs (`baseline_reference.yaml`; source-faithful or recorded deviation, L33-L37):*

  | baseline · dataset | local work | batch | client lr / opt | mom · wd | lr decay | server | utility | status |
  |---|---|---|---|---|---|---|---|---|
  | felix · cifar / speech | 1 epoch | 32 / 16 | 0.04 SGD | 0 · 0 | none | fedbuff, lr 1.0 (FX-D185), rate `new` (scale 0.4, a 0.25, b 0.1) | fedscale | lr final; K / c / rate under FX-N109 (Formulation) |
  | fedbuff · cifar / speech | 1 epoch | 32 / 16 | 0.04 SGD | 0 · 0 | none | fedbuff, lr 1.0, rate 1/√(1+s), lr batch-normalize | – | final (FX-D73, FX-N74) |
  | refl · cifar / speech | 1 step | 10 / 20 | 0.1 / 0.05 SGD | 0.9 · 5e-4 | 0.98 / 10 | REFL SAA (speech: FedAvg server, recorded) | – | final (deviations recorded) |
  | oort, oort_star · cifar / speech | 20 steps | 30 / 15 | 0.04 SGD | 0.9 · 5e-4 | 0.95 / 5 · 15 | FedAvg-YoGi η 0.005 / 0.002 (cifar clip 1.0) | fedscale | final (deviations recorded) |
  | feddance · cifar / speech | 1 step | 24 / 32 | 0.1 / 0.075 SGD | 0.9 · 5e-4 | 0.98 / 10 | FedAvg equal | – | final |

  Felix vs FedBuff differ only in Felix's selection (utility + speed, availability-aware, eval tasks), its rate (staleness + utility)
  and FedBuff's last-batch lr normalization. Felix's rate constants are 2024 values; its rate is ~staleness-blind (FX-T56); exploration knobs don't bind on syn_0 (PR39a).

**Cross-cutting capabilities**

| capability | status | item |
|---|---|---|
| checker catches injected sim bugs (P11a-c) | ✅ 6/6 CAUGHT both datasets (PR28 C3) | FX-D125 |
| fail fast; ladder runner + offline re-gate; early DOOMED kill | ✅ | FX-D22, FX-D133, FX-N42 |
| real async ingest < 2 s | ✅ (re-measure on FX-D197) | FX-D16 |
| profiled sim charges per dataset/platform | ✅ | FX-D143 |
| DIST tolerances from replicate floors | 🟡 2-real bands gate regrade (FX-D188 keeps reds past 3× floor); none tightened | FX-D37, PARITY Q2 |
| real == sim with deterministic kernels | ✅ mechanism; re-measure on FX-D197 | FX-D144/D149/D158 |
| no-overrun rule (PL19) | ✅ mechanism; re-measure on FX-D197 | FX-D180 |
| streaming / oracle experiment | 🟡 EV19 green both datasets; oracle advantage ungraded | FX-N13 |
| sim speedup vs real (leg wall) | re-measure on FX-D197 (speech async GPU-slot-bound, FX-L81) | FX-D180 |
| one dispatcher batch over both nodes | ✅ PR38: 60 units, 18.7 h vs 20.7 h plan, auto-collect | FX-D186 |
| paper experiments sim-only | ⬚ | FX-N12 |

**Accuracy table (FX-N74; `scripts/accuracy_table.py`).** Target cifar 50%, speech 60%, syn_0, full data, the leg's own clock. ⬚ on FX-D197.

**Convergence board (FX-N93; long runs beyond the 76 screens).** Green when all four columns pass; thresholds nominal until a
real↔real replicate sets the band (L12).
- **A · accuracy:** sim reaches target within 90-120 min on its clock, or is round-bound and real and sim reach the same accuracy at
  the same time.
- **C · curve parity:** real + sim + real replicate, deterministic kernels: accuracy at matched rounds within max(2 pp, real↔real),
  time to target within max(10%, real↔real).
- **U · hours under unavailability:** syn_50_3st + mobiperf_3st, ≥ 2 h: INV/EXACT green and C's criteria.
- **S · sim speed (operator 10-14):** never slower than real; 1-2× needs a review; 2-5× and beyond is the goal.

**Convergence ⬚ / 12 on FX-D197.**

---

## Decisions (operator; standing until changed here)

- **Nodes (10-13, 10-14):** batches use jayne GPUs 0,2-7 (GPU 1 out) and shepherd GPUs 0-3 together (R4); scored pools and runs live on
  jayne (the dispatcher copies shepherd's). Shepherd = jayne hardware: charge profiles and floors are shared (T9).
- **Task time = FedScale device model (10-20, FX-D197):** permanent for cifar and speech; the work-independent per-client D never
  returns, and results made with it are void. Screens stay at n 75 / 50; claims and paper runs at the reference n 300 / 100.
- **FedBuff K (10-20):** FedBuff keeps K = 10, as its paper's static buffer-size analysis chose; K / κ tuning is Felix's only.
- **Felix optimization (10-19, FX-N109):** α 0.1 only; syn_0 first, then syn_50_3st; one change at a time. Signals Felix may use must be
  general across datasets, models and unavailability (update norm / direction allowed). Knob flags are temporary: keep a flag only
  while its knob is undecided or proven useful; delete it (code, config, F39A arm) once the knob is dropped or fixed.
- **Long runs launch through `scripts/dispatch.py` (10-17):** a batch YAML of harness_pool stages; never hand-split stages per node.
- **EuroSys FedDance (10-11/12): FedDance's algorithm on our setup, apples to apples.**
  - *Common:* model, Dirichlet α=0.1, n=300 cifar / 100 speech, K=10 aggregated (c=13), the May-2025 shared trainer (1 epoch, bs 32, lr
    0.04, plain SGD; `trainer/config_dir0.1_num300_*` @ `966797558`), traces, simulator, target 70% (`EUROSYS_TARGET`), time to target on
    the run's clock, comm = msgs × 1 MB (eval reply 0.01 MB).
  - *Per baseline:* selection, availability handling, aggregation. FedDance: V·I·A utility + MAB exploration (K_h 50, K 5, β 5,
    overcommit 1.3), own check-in predictor, FedAvg equal weights, server lr 1. Not carried over: its paper setup (LS=1, bs 24, lr 0.1).
  - *Comparability:* all baselines (FedDance again) re-run after parity; EuroSys curves stay the May-2025 real runs (`agg_logs/`).
  - *Plots:* time to 70% = first eval ≥ 70, else log-time fit, capped 48 h (`curve_tta`); speedup vs REFL and FedDance, comm vs REFL.
- **Baselines (10-08, 10-10):** source-faithful values in `baseline_reference.yaml` (ROBUST L33-L37); cifar Oort/oort_star clip 1.0 is a
  recorded deviation (FX-D126); speech Oort/refl/feddance accuracy recorded round-bound; debug every red on short runs first.
- **FedScale baselines (10-12):** FX-D148 partial dispatch is correct for refl, oort, oort_star, feddance (`resampleClients`); REFL
  staleness unbounded (`stale_update: -1`, paper §5).
- **Traces and overruns (10-14):** AVL_EVAL is eval-only and Felix-only (FX-D165); 3-state syn traces only (FX-D166); ≤ 1-2% of tasks may
  overrun D, real and sim, else fewer trainers per GPU (FX-N98).
- **Felix claims (10-17):** all six reach target; Felix's TTA significantly lowest and its traffic to target lowest in every scenario;
  Felix least affected by streaming, heterogeneity and unavailability. Any miss is a ⚠️ on the Scoreboard and blocks FX-N12.
- **Sim-only experiments (10-14):** after the G3 batch (all six × syn_0 + syn_50_3st + stream_lin × both datasets) sits inside the
  real↔real band, paper experiments run sim-only.
- **Sim speed (10-14):** never slower than real; 1-2× needs review; 2-5× and beyond is the target; always look for more.
- **Speed levers (10-14 late):** speech bf16 (FX-D168); balanced GPU placement (FX-D169); one scaled n:c:K per dataset for all baselines
  (`G3_SHAPE`); more parallel legs via scaled n/c/K at fixed ratios.
- **C1 and degenerate cells (10-18; cifar 10-19):** a speech or cifar baseline below target in a 90-120 min leg is round-bound if its eval loss falls and
  accuracy rises; near-degenerate unaware speech cells under unavailability are fine while every abandon fires at the task timeout
  (the dataset's `send_timeout_wait_s`) and its slot is re-selected; simulator studies give them longer legs (FX-N106 closed).
- **Launches (10-18):** Claude runs short tests on jayne itself; PR39 and other long batches launch only on the operator's go.
- **Real-path-identical keys (10-18):** reals pair across code keys whose diff leaves the real path unchanged
  (`parity_ladder.REAL_SAME_CODE`, evidence per link), so sim-only fixes re-grade against existing reals instead of new 2 h reals.
- **Run scope (10-12, 10-14):** the 76/76 stands on one code state = one `code_key` (FX-D162), which may span batches; a band pairs
  only within one key; syn_50_3st takes its baseline's syn_0 band (10-15; FX-L80 asks to revisit for Oort).

---

## Scope and conventions

**Task-time model (FX-D197, operator 10-20; paper methodology).** Every task's modeled time is FedScale's, as Oort and REFL compute
it (`getCompletionTime`):

    t = 3 · samples · ms_per_sample / 1000  +  transfers · model_kbit / bandwidth_kbps      (then ÷ harness delay factor)

- *Devices:* FedScale's `client_device_capacity` traces (500k devices; `third_party/REFL/dataset/data/device_info`); trainer k gets
  device k, REFL's `clientId % len(profiles)`. Registry fields `compute_ms_per_sample`, `bandwidth_kbps`. Across all 500k devices:
  compute p10 / p50 / p90 21 / 41 / 175 ms per sample, bandwidth 1.8 / 6.1 / 32.5 Mbps.
- *Work:* samples = local steps × batch (Oort 20 × 30 / 15, REFL 1 × 10 / 20, FedDance 1 × 24 / 32), else epochs × visible shard
  (Felix, FedBuff: 1 epoch, so it grows with streamed data). 3× = forward + backward. Model 2.15 MB cifar, 29.0 MB speech.
- *Transfers:* train = download + upload (2); eval = one forward pass (1×) + download; Felix's I9 utility pass = one forward pass.
- *Timeout:* each dataset's abandon timeout is 2 × the longest task in its reference population (cifar 903 s, speech 2560 s), the
  same ratio as the old 90 s / 450 s.
- *Mean task time (G3 populations, cifar / speech):* Felix / FedBuff 50 / 324 s, Oort 168 / 185 s, REFL 11 / 112 s, FedDance
  15 / 115 s, eval 18 / 126 s.
- *What it replaced:* one per-client delay (2024 speed classes, mean 12 s cifar / 58 s speech) for every task, whatever its work.
  That let Felix train 1.7× bigger shards at no cost, undercharged Oort's 600-sample tasks 3.4× relative to Felix's epochs, and
  charged REFL / FedDance a full epoch's time for one minibatch.
- *Validation:* EV3 recomputes t for every train task (registry device, reported samples and model size); `task_time_audit.py`
  reports mismatches and per-baseline train / eval times per leg. 10-20 shepherd smokes (G3 syn_0, 20 / 30 min vclock, all six):
  0 mismatches over 4,212 train tasks, every eval span equal to its formula, EV PASS on every leg, and the cifar Felix real + sim
  pair grades parity PASS (396 vs 400 tasks, same 33.1 s mean). Measured mean task time, cifar / speech: Felix 33 / 292 s
  (its speed term picks devices fast for their work), FedBuff 42 / 299 s, Oort and Oort* 151 / 165 s (44 / 66 tasks in the leg),
  REFL 11 / 95 s, FedDance 12 / 83 s.

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

## Active build — Felix design: ideas and execution (FX-N109, operator 10-19)

**Goal:** minimize time to accuracy and bytes to accuracy. α 0.1; syn_0 first, then syn_50_3st. **Every number in this section
(PR39a, PR39b, the 10-20 smokes) predates FX-D197; the mechanisms stand, magnitudes (κ edge, R, U*, belief bias) re-measure in PR39c1.**
**Frame:** TTA = U* / R (FELIX_INSIGHTS I-G1): U* = aggregated updates to target (update quality: selection, staleness, coverage),
R = aggregated updates per hour (system: c, durations, availability, waste).

**Today (PR38 G3, n:c:K 75:15:10 cifar / 50:15:10 speech):**
- *Train pick:* Oort score (normalized loss utility + UCB bonus) × speed penalty, `scoring.OORT_PAPER_DEFAULTS` untuned; round_threshold 70,
  exploration_decay 0.999. Top 20% of clients supply 42% (sim) to 57% of cifar updates.
- *Eval pick:* budget evalGoalFactor · K = 10 per version, but eligible only if trained before and ≥ 35 versions since the last eval or
  train (`evalStaleGate`, a train return also sets `last_eval_round`); on 3-state traces AVL_EVAL devices only. Used: cifar 10% syn_0,
  2% syn_50_3st; speech 3-4% syn_0, 0.5% syn_50_3st.
- *Aggregation:* FedBuff, server lr 1.0, weight w = 0.4 · (1 + τ)^-0.25 + 0.6 · (1.5 − (1 + u)^-0.1) (`fedbuff.weight_factor`). FedScale
  u is ~10²-10³, so the β term sits at ≈ 0.95 and w ≈ 0.58 + 0.4 · (1 + τ)^-0.25. Measured (PR39b): 0.80-0.94 at every K × c, while
  FedBuff's 1/√(1+τ) falls from 0.69 to 0.44 as c / K grows. c, K, lr fixed.
- *Unavailability:* aware dispatch, proactive eviction, evals only to AVL_EVAL devices; no prediction of a device leaving mid-task.

**Formulation (fit on PR39b, cifar syn_0 unless noted; `hyp` = untested):**
- *Server step at version v:* x_{v+1} = x_v + (η/K) Σ_{i∈B_v} w_i Δ_i(x_{v−τ_i}), |B_v| = K. The per-update step is γ = η w̄ / K. With
  c tasks in flight, an update's delay in updates is δ ≈ c, so its staleness in versions is τ ≈ c / K (Felix measures 0.6-0.75 c / K,
  because it favors fast clients).
- *Rate (Little's law):* R = c / (d̄ + o). Measured R ≈ 5.5 · c updates/min for c 15-30, the same at every K (d̄ + o ≈ 11 s).
- *Objectives:* TTA(ε) = U*(ε) / R. Bytes B(ε) ≈ m · (2 U* + E + c), with m = model size and E = eval downloads. Goal: minimize TTA
  subject to B ≤ the best baseline's (claims C2, C3).
- *Step law:* U* falls as γ grows until the delay-adjusted step κ = γ · c = η w̄ c / K reaches a per-dataset edge κ*_D. Past the edge
  U* rises, then the run diverges. Columns: U* = updates to 40% (R × min); min / GB to 40% · to 50% (seeds reaching 50%).

  | Felix arm | γ | κ | U* | to 40% | to 50% |
  |---|---|---|---|---|---|
  | K10c15 (ctl) | .089 | 1.3 | 2.3k | 27.7 / 11.3 | 52.5 / 20.4 (2/3) |
  | K5lr05 | .086 | 1.3 | 1.9k | 23.3 / 10.1 | 51.9 / 21.1 |
  | K10c20 | .088 | 1.8 | 2.4k | 21.1 / 11.7 | 46.9 / 24.3 |
  | **K5c15** | .172 | 2.6 | **1.4k** | 17.3 / **7.5** | 45.6 / **17.9** |
  | K10c30 | .085 | 2.6 | 2.5k | **14.3** / 12.1 | **35.2** / 28.5 |
  | K10lr2 | .178 | 2.7 | 2.0k | 24.3 / 9.9 | 52.6 / 19.4 (1/3) |
  | K5c20 | .168 | 3.4 | 1.7k | 15.6 / 9.1 | 42.1 / 22.8 |
  | K3c15 | .277 | 4.2 | 2.1k | 26.1 / 11.6 | 55.1 / 22.9 |
  | K5c30 | .160 | 4.8 | 2.4k | 14.4 / 12.7 | 39.7 / 32.4 |
  | K3c20 / K3c30 | .27 | 5.5 / 8.0 | 2.6k / 3.2k | 24.3 / 14.7 · 19.0 / 17.9 | 52.0 / 29.3 · 45.0 / 39.3 (2/3) |
  | K2c15 / c20 / c30 | .41 | 6.2 / 8.2 / 12 | 4.1k / ≥ 6.7k / ∞ | 50.7 (2/3) · 59.5 (1/3) · – | – |

- *Edge:* on cifar κ* ≈ 3-4: κ ≥ 5 costs +50-140% U*, and at κ ≥ 8 the test loss spikes to 3-7 and 40% is never reached. On speech
  κ* < 2.7: Felix K5 (κ 2.7) and K3 (4.4) have test-loss spikes of 13-16 and peak at 37% / 27% (ctl 51%), while FedBuff K5 (κ 1.7)
  stays smooth.
- *(1) K and c substitute, they don't stack:* at fixed κ, γ ∝ 1 / c cancels R ∝ c, so time stays roughly flat (K5c15 17.3 vs K10c30 14.3
  min) while U* and bytes grow with c (K5c30 ≈ K10c30). So run the lowest c that keeps κ at the edge.
- *(2) K matters beyond γ:* at equal γ, smaller K wins (K5lr05 vs K10c15: −16% min to 40%; K5c15 vs K10lr2: −29%, and K10lr2 reaches
  50% on 1 / 3 seeds). K sets how coarse each jump is.
- *(3) γ wants a schedule:* a large γ helps early and hurts late (K10lr2: 40% sooner, max 47.8% vs 50.2%).
- *(4) Nothing in Felix shrinks γ as κ grows:* w ≥ 0.58 at any staleness; FedBuff's 1/√(1+τ) does shrink it.
- *Selection's share of the gain:* at matched γ and κ (Felix K10c30 vs FedBuff K5c30: γ .085 / .088, κ 2.6, equal R), Felix needs 2.5k
  updates to 40% and FedBuff 4.1k. Utility + speed selection is a real 1.65× U* gain, separate from step size. FedBuff gains about as
  much from K5 as Felix does (−35% vs −37% min to 40%), and Felix stays 1.6-2× faster at every matched (K, c).
- *Eval cost:* the gate counts versions, so eval picks track versions (1.0-1.5 per version), and their share of downloads grows as K
  falls: 10% at K10, 25% at K5, 73% at K2 (~11% of bytes at K5c15). On syn_0 eval buys no time (eval0 = ctl, PR39a).
- *Control problem (`hyp`):* choose (η, K, c) online so κ stays just under κ*_D. Read κ*_D from update geometry (gradient noise scale B,
  cos(Δ_stale, Δ_fresh), `update_stats.py`), not a per-dataset table. Take c from the bytes budget. Keep K as small as the noise of a
  mean over K allows (K ≥ 3-5 at α 0.1).
- *Runtime K (I5):* K = B (the gradient noise scale tr Σ / |G|², McCandlish 2018) is **refuted** at α 0.1. In the 10-20 smoke
  (ctl / dkc legs), the per-cycle B is mostly unbounded (the estimate of |G|² ≤ 0) or 10²-10³, because
  1-epoch updates from different clients barely agree, so K pins at c = 15, opposite to PR39b's K5. B measures client disagreement,
  not SGD noise; it stays as telemetry only. **Built instead: AIMD on K** (TCP congestion control, `aimd_k`). Per population sweep
  (n updates), if the clients' loss on the received model (I9) didn't rise by more than 2 SE, K −= 1 (larger step); if it rose,
  K ×= 2. This finds the edge κ*_D per dataset and re-finds it as the edge moves late in training (the schedule of (3)). Its only
  constants are AIMD's conventional 1 / ×2, a window of one sweep, and a 2-SE test.

**Sensing audit (PR39b telemetry, `selection_audit.py`; cifar / speech K10c15):**
- *Felix picks big shards, not fast or poorly fit clients:* Spearman of picks vs shard size is +0.95 / +0.80, vs speed +0.13 / +0.40,
  vs per-sample loss −0.88 / −0.39. Felix averages 263 / 1005 samples per update against a population mean of 152 / 837 (FedBuff:
  picks follow speed, +0.69 / +0.79).
- *Cause 1, the setup:* task duration is the client's trace D, independent of shard size (ρ 0.01; `main.py` `training_delay_s`), so a
  big shard costs no time. Fixed by FX-D197 (FedScale device model). Part of Felix's 1.65× U* gain at matched γ
  comes from this.
- *Cause 2, the estimand:* utility = |D| · √EMA₀.₈(batch loss²) over the training pass (FedScale's, `syncfl/trainer.oort_loss`). On
  1-epoch tasks that EMA reads the loss after training, so big shards look well fit. Eval runs the same EMA on the unchanged model,
  i.e. the loss before training. Eval-set beliefs overstate the next train's utility 1.2-1.4× on cifar and 2.1-2.9× on speech
  (train-set ×1.0); median error is 0.23-0.41 vs 0.12-0.22 on cifar and 1.1-1.9 vs 0.15 on speech. Fix: I9.
- *Not a class skew:* the label mass of Felix's updates is within TV 0.04 of the population (FedBuff 0.08). Concentration is
  inherited from Oort's |D|-weighted utility (Oort family top-20% share 0.59-0.69 in PR38; Felix 0.39-0.53).
- *The weight discards the signal:* u varies with CV 0.37-0.50 across clients, but β = 1.5 − (1+u)^-0.1 squashes it to 0.71-1.02. Fix: I7.
- *Streaming:* G3's linear stream grows every client at the same rate (no `stagger`), so ranks never change and eval has nothing to
  discover. There, eval-set beliefs are no better than train-set ones (error 0.15 vs 0.14). Eval's value needs heterogeneous growth
  (`events`, F39S). On syn_0, eval buys no time (eval0 = ctl) and adds 10-25% to downloads.

**Signals (general, privacy-safe: scalars the server already gets, or update geometry it already sees):** per client: duration mean /
variance d_i, availability history → p_i = P(finishes before leaving), loss utility u_i, data size n_i, belief age (versions since last
report), ‖Δ_i‖ and cos(Δ_i, Δ̄) (not secure-aggregation compatible). Per aggregator: accuracy gained per update (I-G2), staleness, buffer
depth, slot occupancy, train time lost to eval (I-F1), update disagreement, belief error (I-A4).

**Ideas (state `hyp` until a screen measures them):**
- *I1 · Train value per slot-second.* V_i = p_i · u_i · s(τ_i) / d_i, s = the optimizer's own staleness weight, τ_i ≈ d_i × commit rate;
  plus a coverage term against repeat picks. Replaces the Oort score × speed penalty.
- *I2 · Eval by value of information.* Eval m iff P(fresh belief moves m into the train top-k) × gain > λ · model bytes; uncertainty grows
  with belief age / model drift since m's last report. AVL_EVAL devices cost bytes only. Replaces the fixed gate.
- *I3 · Eval → train on the same model.* An eval that ranks its client into the train top-k trains on the model it already holds:
  one download saved per conversion (C3').
- *I4 · Finish-probability (unavailability).* p_i from the client's own session history; prefer tasks that fit the remaining window.
- *I5 · Adaptive K (built, `aimd_k` + I9).* AIMD on K against the clients' loss on receive (Formulation, "Runtime K"); c is fixed as
  the resource budget. `noise_scale_k` stays for telemetry (refuted as a controller).
- *I6 · Exploration by uncertainty,* not a decay clock (same quantity as I2).
- *I7 · Utility-normalized weight (built, `agg_rate_conf.type: norm`).* w_i = (u_i / ū_cycle) / √(1+τ_i): utility reweights within the
  cycle (mean 1) and the staleness discount is FedBuff's own. No constants.
- *I8 · Version-free eval gate (built, `evalGateUnit: updates`).* The gate counts committed updates (gate × configured K), so changing K
  keeps the eval cadence. Still to do: `round_threshold` and exploration decay.
- *I9 · Utility on receive (built, trainer `utilityOnReceive`).* Before training, a no-grad pass with the received model measures
  |D| · rms loss, which is eval's quantity, so train and eval beliefs are comparable. Charged one forward pass; BN buffers restored.
- *I10 · Value per second (`hyp`; FX-D197 makes it meaningful).* With duration ∝ samples, utility per expected second replaces Oort's speed
  penalty (I1).

**Execution (one change at a time):**
- *E1 · PR39a: one-knob syn_0 sweep · done* (`--tier F39A`, `pool_20261019_0326_PR39A_JAYNE_{C,S}`, 3 seeds, sim-only). Only K and c
  move cifar (FX-L84). Worse: K 20, c 10, lr 0.7, gate 15 / 5 (bytes). No effect: exploration, round_penalty, round_threshold,
  cut_off_util, a_exp, scale, lr 1.5, eval off. Top-20% share stays ~0.41, so selection knobs don't bind on syn_0 α 0.1. Speech: c 10
  cuts GB to target 100 → 70 at equal time; c 30 is EV-untrusted (FX-N110). Overrides: `--agg-hp 'sel.<k>=v opt.<k>[.<sub>]=v'`.
- *E2 · PR39b: K × c grid · done* (`--tier F39B`, 60-min vclock sims, 3 seeds, 69 legs, 68 EV PASS). Felix K 2 / 3 / 5 / 10 × c 15 / 20 /
  30, plus K10lr2 / K5lr05 and FedBuff K 10 / 5 × c 15 / 30; speech K10c15 / K5c15 / K3c15 (felix) and K10c15 / K5c15 (fedbuff); no
  speech arm reaches 60% in 60 min. Results are in the Formulation table. `knob_sweep.py pool_20261019_1912_PR39B_*`.
- *E3 · PR39c1 (`batches/pr39c1.yaml`, priority, shepherd) · ready.* Felix only, 60-min syn_0 sims, 3 seeds. Cifar: ctl (K10c15, + B
  telemetry), K5, uK10 (I9), nK10 (I7), pK10 (per-sample utility + I9: does the gain need free shard size?), dkc (I5 AIMD + I9), f2
  (I5 + I7 + I8 + I9). Speech: ctl, uK10, dkc, f2. *Claim:* f2 (no per-dataset values) matches each dataset's best fixed arm (cifar K5,
  speech ctl) on time and bytes; dkc's K settles at 3-6 on cifar and higher on speech; pK10 stays within the seed range of uK10.
  *Refute:* dkc's K stays at a bound or random-walks; f2 trails the best fixed arm by more than the seed range; pK10 loses Felix's
  edge on FX-D197 (then selection still leans on size).
- *E3' · PR39c2 (`batches/pr39c2.yaml`) · ready, after PR39c1.* Speech K5 / K5lr05 / K10lr2 (is speech's K5 failure κ or K?), nK10,
  nK5, pK10; cifar K3lr06, K2lr04, nK5, nK10c30, gK5 (I8); FedBuff K10c15 controls on both datasets.
- *F39S · PR39s (`batches/pr39s.yaml`) · built, operator launches after PR39c2.* Does the eval selector sense heterogeneous growth and
  pay its bytes? Cifar syn_0 `events` stream, 2 h vclock sims, 3 seeds: ctl (felix, fedbuff, oort), eval0, gateU, uK10, ugU, oracle (true
  current utility). *Claim:* uK10 / ugU beat eval0 on TTA by more than the seed range and close part of the gap to the oracle.
  *Refute:* eval0 ≈ ctl here too, which means drop eval tasks (C3 gains its bytes) or redesign them as I2.
- *E4 · syn_50_3st.* Repeat E3 where availability matters (I4, eval on AVL_EVAL); a session cut mid-task adds waste to R.
- *E5 · Confirm.* Real legs for the kept changes (sim parity is graded on today's config only), then PR39 on the optimized Felix.

## Active build — throughput (FX-N100, absorbs FX-N99; operator 10-15)

- *Goal:* the most legs per wall hour at high CPU/RAM/GPU use, each leg behaving as if alone (R18, R24); proven on 5-10 min screens,
  then every long batch uses it. Tools and buffers: PARITY "Throughput". Built: FX-D189 (measured leases, admission, one queue),
  FX-D190 (GPU sharing, trainer-granular packing), FX-D175 (footprint fixes), FX-D186 (dispatcher).
- *Measured footprints (per leg; refresh as history grows):*

  | leg | cores mean / p95 | RAM peak GB | GPU GB per GPU | SM % | wall |
  |---|---|---|---|---|---|
  | cifar G3 real (n=75, 1 GPU) | 0.2-0.3 / ≤ 0.5 | 80-90 | 34-35 | 0-2 | 2 h + ~3 min join/teardown |
  | cifar G3 sim | ~2 / 4-5.7 | ~70 | ~35 | – | 4-8 min |
  | speech G3 real (n=50, 2 GPUs) | 0.5-1 / 1.5-3.5 | 77-82 | 36-44 | 1-32 | 2 h + ~3 min |
  | speech G3 sim | ~2 / 3-5 | 72-82 | 36-41 | 17-81 | 4-44 min (async: GPU-slot-bound, FX-L81) |
  | cifar / speech G0U real (n=50) | 0.9 / 2.7-4.2 | 49-65 | 20 / 22-28 | ~1 | 7-8 min at 240 s |
  | G1AS sim n=300 cifar / n=100 speech (4 GPUs) | 4.6 / 3.7 | 133-163 | 31-35 | – | cifar ~20 min, speech 31-46 min |
  | T3 real / sim (stub) | 0.3-0.9 / ≤ 3.3 | 6-13 | 0 | 0 | 2-6 min |

  GPU memory is ~0.4 GB per trainer process (312 MiB CUDA context; MPS doesn't shrink it, FX-L78), so GPU legs bind on GPU memory and
  host RAM, never SM.
- *Results (`resource_report.py --util`):*

  | batch | legs/h | concurrency | CPU | RAM peak | GPU mem | GPU idle | note |
  |---|---|---|---|---|---|---|---|
  | first PR36 (jayne, 14 h) | 15.9 | 3.3 | 6% | – | 39% | 25% | idle leases, exclusive GPUs, serial lanes |
  | E1 / E2 / E3 screens (240 s) | 68.9 / 74.5 / 81.7 | 5.4 / 6.0 / 7.2 | 5% | 232-253 | 13-20% | 62 / 57 / 34% | trainer packing +19%, 0 overruns, verdicts = solo |
  | PR37 pass 1 (jayne / shepherd) | 3.7 / 2.3 | 2.7 / 2.0 | 2% | 377 / 329 | 48 / 34% | 17 / 36% | RAM-capped (FX-D179) |
  | PR38 (jayne / shepherd, 18.7 h) | 3.4 / 1.8 | 4.6 / 2.4 | 2-5% | 437 / 394 | 56 / 61% | 17 / 3% | host-RAM-bound; overruns 24 / 390k |
| PR39b sims (both nodes, 3 h) | 23.0 | 4.3 | 4% | 342 | 25% | 53% | RAM-capped at 4 legs/node; 0 overruns / 400k |

- *Binding now:* host RAM (a G3 leg reserves 1.15 × its 70-90 GB peak). *Levers:* fork-server import sharing (FX-N84, ~280 MB/trainer),
  `--gpu-share` for speech sims (FX-L81), short legs for bug-finding (R0).

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
- **FX-L54** Reinject every withheld delivery due before the next buffered sct, in every commit loop (sync barrier: FX-D194).
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
- **FX-L80** Give a chaotic cell its own real band, not syn_0's: deterministic Oort syn_0 reals agree to 0.2%, its syn_50_3st reals ~10% (stall
  placement, FX-L53); measure the cell's own band before calling a sim red (PR38 cifar oort syn_50_3st).
- **FX-L81** Async speech sims are GPU-slot-bound (`sim_gpu_train_slots` 1 per GPU, FX-D167): their wall scales with GPUs spanned, not
  load (PR38 felix 2 GPUs 44 min vs PR37 5 GPUs 30 min, same 2.1 s per task). Spread their trainers (`--gpu-share`).
- **FX-L82** Charge transport from real's profile, not the sim's own measurement: a busy sim node delivers a fan-out ~1.6× faster than
  an idle real node (cifar 8-47 vs 17-77 ms by rank), and the gap compounds into the stream clock (FX-D191).
- **FX-L83** Read a real↔sim fork at its first differing commit set before blaming the clock: equal-D picks racing for the K-th slot
  is a draw (FX-L72), cifar lin oort round 13 (384 vs 437, both D = 2 s); a version closed late on a straggler is a bug (FX-D194 v12).
- **FX-L84** Tune Felix's server step through κ = η w̄ c / K, not K, c or lr alone: on cifar, U* is lowest at κ 2.6-3.4 (K5c15, K5c20),
  and K and c substitute for each other (PR39b, Formulation).
- **FX-L85** At equal per-update step γ, prefer the smaller K (more frequent, smaller jumps): K5lr05 −16% and K5c15 −29% min to 40% vs
  K10 at the same γ (PR39b).
- **FX-L86** Count eval gates and selector clocks in updates or seconds: counted in versions, eval's share of downloads grows 10% → 73%
  as K drops 10 → 2 (PR39b).
- **FX-L87** Read what drives a selector's picks (`selection_audit.py`) before tuning its knobs: Felix's picks track shard size
  (ρ +0.95), not speed (+0.13) or per-sample loss (−0.88) (PR39b).
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
- **FX-T54** Don't read a floor measured at another matched N: G0U feddance speech floors ran at N=14, the red pair at N=13 (reals
  25-30 there); fix the round the sim lost first.
- **FX-T55** Don't carry a cifar-tuned K or lr to speech: Felix K5 (κ 2.7) is cifar's best arm and diverges on speech (test loss 13-16,
  max 37% vs 51%; PR39b).
- **FX-T57** Don't merge eval and train utilities into one belief unless they measure the same loss: eval reads pre-training loss and
  train the post-epoch EMA, so eval beliefs overstate 1.2-2.9× (PR39b; I9).
- **FX-T58** Don't test eval-driven sensing on an unstaggered linear stream: every client grows at one rate, so ranks never change
  (G3 lin). Use `events` or `stagger`.
- **FX-T60** Don't charge a task a time independent of its work: one per-client D let Felix train 1.7× bigger shards free, and gave
  Oort's 600-sample tasks and REFL's 10-sample tasks the same time (FX-D197).
- **FX-T59** Don't set K from the within-cycle gradient noise scale at α 0.1: B is unbounded or 10²-10³ and K pins at c (smoke
  10-20). It measures client disagreement, not SGD noise.
- **FX-T56** Don't read Felix's a_exp / scale as a staleness discount: w = 0.4 α(τ) + 0.6 β with β ≈ 0.95 stays ≥ 0.58, so at K10
  sweeping them shows nothing (PR39a no-effect).

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
- **FX-D195** Felix mechanisms behind flags (FX-N109; default off = today): I5 `selector.kwargs.dynamic_kc` (asyncfl `_dynamic_kc_step`,
  `update_stats.py` noise scale B and stale-vs-fresh cos, policy `aimd_k` on the cycle's per-sample loss (`noise_scale_k` kept),
  `stats` = telemetry only, EV7 per-round K); I7 fedbuff rate `norm`; I8 `evalGateUnit: updates` (aggregator `n_updates`, end `last_report_updates`); I9 trainer
  `utilityOnReceive` (charges one forward pass, reported as `utility_pass_s`; EV3 subtracts it); selector `utilityPerSample`. Tests: `test_felix_dynamic_kc.py`, `test_fedbuff_lr.py::TestFedBuffNormRate`, `test_oort_selector.py::TestEvalStaleGate`.
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
- **FX-D197** Task time = FedScale device model (Oort / REFL `getCompletionTime`): registry `fedscale_device_id`, `compute_ms_per_sample`,
  `bandwidth_kbps` (`_metadata/scripts/add_fedscale_devices.py`); trainer `_task_time_s` (train 3 × samples + 2 transfers, eval 1 × + download,
  I9 pass 1 ×; reported as `model_samples / compute_s / comm_s / kbit`); EV3 recomputes it; dataset timeouts in `datasets.yaml`;
  parity speed diagnostics use device speed. `training_delay_s` / `speed_class` stay for FluxTune only.
- **FX-D196** `selection_audit.py`: per sim leg, what drives picks (size, speed, loss per sample), samples per update, cifar class
  TV, belief error and bias by source (eval / train), eval per train, I5 K / B trajectory. Tiers F39C (E3 arms, trainer hp via `thp`)
  and F39S (events-stream eval microbenchmark); `batches/pr39c1.yaml`, `pr39c2.yaml`, `pr39s.yaml`.
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
- **FX-D162** `harness_pool.code_key()` hashes on-disk blobs of tracked run-affecting files only (`NOT_RUN_AFFECTING`: docs, tests,
  graders, `run_*.py`): a doc/grader commit keeps gate stamps and floor pairing (L18); a pre-commit screen keys as its commit (FX-D194).
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
- **FX-D183** Real eval spans max(gpu, modeled eval time) as sim charges (sleeps the difference).
  `realEvalDelayFloor=true` reverts.
- **FX-D182** Starved sync sims (syncfl, Oort) wake at min(next flip, next pick timeout) (`_sim_starvation_wake`): picks were abandoned at
  the next flip, up to 80 s late (cifar oort syn_50_3st sim 200 vs real 242 versions → 232). `sim_abandon_wakes=false` reverts.
- **FX-D179** `harness_pool` gates RAM only on the live node-wide check (MemAvailable ∧ MemTotal − all leases); the per-pool budget fixed
  at start capped PR37 shepherd S1 at 205 GB for 20 h. `felix_claims.py` (FX-N108 grader) and `fl_lr_check.py` (pair server lr now
  honoured under `--baseline`) sit outside the code key.
- **FX-D181** `parity_ladder` pairs G3C reals with their G3 cell for floors and per-real votes (G3_syn_50_3st takes the syn_0 band, also
  when that cell ran in a sibling same-key pool: PR38 split them over nodes); G3C is real-only; `readiness_score` files G3 lin cells under
  stream_lin (`test_g3c_replicate_floors_its_g3_cell_and_syn_50`, `test_syn_50_band_from_a_sibling_pools_syn_0_cell`).
- **FX-D188** Checker: a rung whose replicate floor swallowed its tolerance SKIPs only while every gap stays ≤ `_FLOOR_TOL_K` (3) × its
  floor; past that it is red with `beyond_floor` (PR38 cifar lin oort: 57% time gap on a 17% floor read SKIP → green).
- **FX-D189** Measured CPU leases (`--lease-from-telemetry`: real CPU legs lease 1.25 × margin × last-3 `cores_p95`, ≥ 2 cores), live
  admission ≤ 70% node CPU (`NodeLoad`), back-pressure on saturated slots (`TIMING_SUSPECT`), one queue across stages (`want_p<N>` GPU
  priority leases); sims keep formula leases ≥ 4 cores (`SIM_LEASE_MIN_CORES`). A/B `pool_20261015_fxn99_AB{old,new}` 4/4 PASS.
- **FX-D190** GPU sharing (`--gpu-share K`): ledger samples per-GPU SM % and memory; legs (and their trainers, `place_trainers` →
  `FLAME_GPU_TRAINER_CAPS`) are placed on GPUs by measured footprint under a node-wide `flock`, all or nothing, ≤ 90% memory and SM;
  unmeasured legs claim whole GPUs. E3 screen +19% legs/h, 0 overruns, verdicts = solo.
- **FX-D191** Sync sims charge real's profiled fan-out download by send rank (`SIM_FANOUT` → `simDownloadByRank`; cifar 13-76 ms,
  speech 66-903 ms over ranks) and real's send-prep (D end → send minus pre/post-train; `simSendPrepSeconds`) at completion; async
  dispatch keeps the measured lag (FX-D108). Profiles re-derived from PR38 G3 reals (`profile_felix_charges.py` labels `send_prep`,
  `download_by_rank`). Screen: cifar lin oort sim−real stream clock −0.12 → +0.03 s by round 6, first fork round 5 → 13 (reals: 17).
  Closed FX-N95: speech feddance G0U syn_50_3st round close → first task receipt sim 0.137 s, real 0.138 (0.08-0.17).
- **FX-D192** An exclusive GPU leg takes only GPUs with no live `gslot_*` lease of any pool, chosen under `gpu_place.lock`
  (`exclusive_gpus`): an exclusive speech sim and a packed cifar sim OOM-aborted two pools on GPU 0.
- **FX-D193** `G3S<i>_<cell>` (harness_pool tier G3S, `G3S_REPS` 3): sim-only G3 replicates, each graded by `parity_ladder` against every
  real of that cell on a real-path-identical key (`REAL_SAME_CODE`, transitive per baseline); `readiness_score` takes the cell's
  majority over its G3S sims; a cross-key real votes only at the leg's `--runtime-s` (L18, `_same_length`). `felix_claims` C1 applies the 10-18 round-bound rule (`round_bound`).
- **FX-D194** Sync sim barrier commits a send-gated delivery due before the next fresh sct first, counting to K under wait-K
  (`_sync_sim_commit_due_withheld`; `sim_reinject_lookahead=false` reverts): speech feddance syn_50_3st v12 closed on a 235 s straggler
  16 s late, carrying the gated update stale. Tier G0US = sim-only G0U replicates, graded and voted as G3S.
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

- **Q-EV21 · Felix "reaches AVL_EVAL" arms only at ≥ 20 eval picks · parked: FX-N109 I2.** The eval budget (evalGoalFactor · K = 10 per
  version) is a cap: the 35-version gate, train returns resetting `last_eval_round`, and AVL_EVAL-only on 3-state traces leave 0.5-10%
  used (PR38). Re-decide once FX-N109 settles the eval policy.
