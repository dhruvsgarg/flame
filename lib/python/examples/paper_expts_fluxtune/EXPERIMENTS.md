# FluxTune paper evaluation plan

The live plan for the FluxTune paper's evaluation. The doc rules in
[ROBUST_FL_READINESS.md](../_metadata/ROBUST_FL_READINESS.md) apply: every line is true now, no changelog, a
finished item is deleted. Infra and parity status live in [FLUXTUNE_READINESS.md](../_metadata/FLUXTUNE_READINESS.md);
figures come from `fwdllm/expt_scripts/make_paper_figs.py` (§7). Plot list, run list and measured run times:
[RUN_PLAN.md](RUN_PLAN.md). Task cost D per model (calculation and values): [D_CALIBRATION.md](D_CALIBRATION.md).

---

## 1. Claims, metrics, baselines

**Headline claim.** FluxTune reaches target accuracy faster and with less communication than FwdLLM, on every
model/dataset pair we test.

- **Primary metrics:** time to target accuracy (TTA, simulated hours, lower is better) and communication cost
  (GB uplink + downlink until target, lower is better).
- **Secondary metrics:** commits to target, peak accuracy, client engagement time, selection diversity.
- **Primary comparison:** FluxTune (§2 recipe, = C1's A6) vs FwdLLM. Only these two appear in [A] and [B].
- **Secondary baselines:** FedBuff(P) and Felix(P) (`fedbuff_round`, `felix_round`, forward perturbation). They
  appear only in C4, which answers "are the gains just async FL?" (FwdLLM as the sync reference) without diluting
  the main comparison.
- **Availability is not a FluxTune theme.** [A]–C3 run at 100% availability (`syn_0`). Availability appears only
  in C4, against the availability-aware Felix.

## 2. Setup

Common to every run: sim mode, `syn_0` (except C4), highly non-IID partition (Dirichlet α=1), 1 seed per arm, the
same stop rule in every arm (§5). Clients n, pool size N, ρ0, wall cap and eval sampling per cell:
[RUN_PLAN.md §2](RUN_PLAN.md#2-cells-pool-size-and-eval); every arm in a cell uses them. We run one model per dataset (no full model × dataset grid): small, mid and large, the largest on the
hardest task (Yahoo: 10 classes, FwdLLM paper target 0.65). Trainable p rises with model size (0.45M → 0.59M →
0.81M), keeping the probe dimension in one band.

| Cell (manifest id) | Model (total), PEFT, trainable params | Dataset (seq len) | Clients n | Pool N | Target | Prior evidence (FluxTune, single seed) |
|---|---|---|---|---|---|---|
| `distilbert_agnews` | DistilBERT (67M), adapters rf=16, 450,340 | AG News (192) | 100 | 50 if the N50 check passes, else 90 | 0.86 | 0.873 peak in 3 h (full controller); 0.8726 in 1.74 h (simplified) |
| `roberta_large_yelp` | RoBERTa-large (355M), LoRA q,v r=6, `classifier.dense` dropped, 591,874 | Yelp Polarity (256) | 100 | 50 | 0.82 | none; the rf=16 smoke (p 4.2M) sat at chance after 2 h; needs lr 3e-4 |
| `smollm2_1p7b_yahoo` | SmolLM2-1.7B (1.7B), LoRA q,v r=4, 806,912 | Yahoo Answers (256) | 75 (60 if 75 OOMs) | 50 | 0.65 | DistilBERT ≈ 0.73 on Yahoo; SmolLM2-1.7B 0.832 on AG News after 8 h (`run_20261014_034646`, N=50, α=1) |

DistilBERT stays on AG News so the [C] ablations share the [A] cell (A0/A6 reuse its runs).

**Sim mode (every figure run).** Trainers do the real compute; each task's device delay D is charged to the vclock instead
of slept, so a cell runs 3.4× (DistilBERT) to 33× (RoBERTa-L) faster than real ([RUN_PLAN §4](RUN_PLAN.md#4-run-time-basis)).
Every time axis and TTA is vclock. A sim number stands for real only when (1) the cell's D divisor (below, measured 10-24)
and charge profile (G0, pending for RoBERTa-L and 1.7B) hold and (2) real and sim agree at matched commits (checked on
DistilBERT · AG News, 10-24). Run rules: [RUN_PLAN §2](RUN_PLAN.md#2-cells-pool-size-and-eval) "Sim and real legs".

**Eval:** DistilBERT scores the full test set every 4th commit. Yelp-P and Yahoo score 10% of the test set (3,800 / 6,000
rows, ≈ ±1.2 pt at target), resampled per eval; stride = ⌈t_eval / s_commit⌉ + 1 from S.

**Task cost D per model** (`--delay-divisor`; D = device delay / divisor). Calculation, measurements, full D tables and
how to re-derive for a new model: **[D_CALIBRATION.md](D_CALIBRATION.md)**.

| Cell | bs | `fluxtune` divisor | `fwdllm` divisor | `fluxtune` D: min – median – max (100 trainers) |
|---|---|---|---|---|
| DistilBERT · AG News | 8 | 0.478 | 4.78 | 8.4 s – 23 s – 1.6 min |
| RoBERTa-L · Yelp-P | 32 | 0.0169 | 0.169 | 3.9 min – 10.8 min – 46 min |
| SmolLM2-1.7B · Yahoo | 32 | 0.0047 | 0.047 | 14 min – 39 min – 2.8 h |

**Per-cell settings from S (operator 10-22):** the `fwdllm` lr is screened on RoBERTa and 1.7B at {1e-4, 3e-4, 1e-3} (DistilBERT keeps 0.01); 1.7B's n 75 gets a
short memory screen at bs 32.

**FluxTune recipe** (every FluxTune arm: [A], A6, C4). The simplified controller
([cheatsheet §2](../fwdllm/fl_fwd_ft_pipeline_cheatsheet.md#2-flowcharts)) plus two concepts it dropped:

| Restored | Setting | Flag | Why |
|---|---|---|---|
| ρ decline | ρ_t = ρ0·exp(−(t−1)/(2·T_res)), T_res = 2·lnΦ\*/ρ0² (283 at ρ0 0.07; P0 ran 300 at 0.067); landing's decay without B_max, stop or rail | `--rho-schedule exp --t-res {T_res}` (P0 chose it 10-18) | constant ρ grows the absolute step 3× (0.67 → 2.0); the simplified arm overshot its peak (Φ 2.68), and SmolLM2 sat at ρ 0.037 throughout |
| GL decay stop | on (fires on 20 evals past the warm-up, 0.45 × `--max-commits`, > 0.5% below best) | `--saturation-stop` without `--no-sat-decay` | free collapse guard; decline untested on Yelp-P and Yahoo |

Unchanged from the simplified controller: P=10 `mean`, trust ratio, fixed pool N, uniform ω (G1 kept it 10-18;
`grad_aware` is a C3 arm), stall rule 0.003 (leaky streak), Φ rail. Train batch size 8 on AG News, 32 on Yelp-P and Yahoo (P0b);
a 12 h cell drops to bs 8 if F2 shows bs 32 can't reach Λ_need in its cap (TTA is scored on vclock, but a DNF scores nothing).
Async: C = 0.3·n clients in flight, K = C/3 (30 / 10 at n 100). s, ρ0 and T_res are per cell (§2.1). Under `exp`, Φ is bounded by exp(ρ0²·T_res/2) = Φ\* = 2 in every cell, so the Φ rail
never fires and the commit cap (§5) ends the run.
Stop-rule warm-up and horizon are 450 / 150 at a 1,000-commit cap and scale with `--max-commits` (`expts/saturation_stop.py`).

```
export FWDLLM_FD_SCALE_INVARIANT=1   # required when p != 450,340 (FT-T9)
--probe-combine mean --perturbation-count 10 --server-step-rule trust_ratio --rho0 {ρ0}
--commit-gate fixed --pool-target {N} --rho-schedule exp --t-res {T_res} --agg-rate-type uniform
--train-batch-size {8 AG News | 32 Yelp-P, Yahoo}
--var-stopping-policy off --budget-stop-frac 0 --b-max-probe-every 0 --phi-stop halt
--saturation-stop --sat-stall-frac 0.003 --server-update-audit --no-cos-ground-truth-audit
--max-commits {cap} --final-full-eval [--eval-max-samples {rows} --eval-resample]   # rows: Yelp-P, Yahoo
```

Runs from before the ρ anneal (§3 P0), including all July runs, are FluxTune v1 or pre-anneal and do not feed
any figure.

### 2.1 Pool N, step ρ0 and T_res (ρ_max and N = 50 runs open: §3 P0c)

Three knobs, none tuned per setting. Rule, derivation and measurements:
[cheatsheet §8](../fwdllm/fl_fwd_ft_pipeline_cheatsheet.md#8-choosing-n-ρ0-and-t_res); runs: RUN_PLAN §3.1.

| Knob | Value | Rests on | Gap |
|---|---|---|---|
| N | 50, every cell | `N* ≈ 29·Λ_need²` per dataset · bs: AG News 34–42, Yahoo bs 32 38 (F0); p cancels across models | N 50 at ρ0 0.07 never run (N50 check); Yelp-P unmeasured; p cancellation rests on one point |
| ρ0 | 0.07 DistilBERT; 0.14 RoBERTa-L, 1.7B (operator 10-24) | 0.07 is the largest step that has run cleanly (DistilBERT, 0.067–0.068 on all three datasets); accuracy depends on Λ alone below it | nothing above 0.07 run yet (R1/R2 on DistilBERT; big cells via the P3/P4 abort rule, fallback 0.07); N\* scales with (ρ0/0.07)² (N row) |
| Φ\* | 2 | P0 `exp` peaked at Φ 1.90, ended 1.93 | not swept |
| T_res | 2·lnΦ\*/ρ0²: 283 at ρ0 0.07 (DistilBERT), 71 at 0.14 (RoBERTa-L, 1.7B; T ~380 ≫ T_res, so horizon-aware ≈ 71) | derived; capped runs reach Φ\* | none at 0.14 |

ρ0 is passed directly (`--rho0`); s = ρ0/cos is derived (2.10 / 4.82 / 5.62 at N 50) and logged. `--gate-safety-s` still works when `--rho0` is unset.
**n_eff cannot set N.** `n_eff = N·(1 + O(P/p))`: all six recent runs read n_eff/N = 1.000 median (≥ 0.956), since
each upload is ~p/P ≈ 45,000× more probe noise than gradient. It detects only correlated probes; log it, don't gate on it.

**Fixed N, decaying ρ.** Each commit reads one data bin; the `exp` tail keeps reading bins with small steps while Φ
barely moves. A pool shrinking with ρ is time-optimal (solution §4.6) but reads less data per tail commit.

In the paper: N is the number of client gradients per aggregation, set once per dataset; ρ0 is the largest relative
step for which accuracy still depends on progress alone; T_res spends the same Φ\* in every setting.

## 3. Run queue (top = next)

Each item: action · exit · state. [B] adds no runs: it reads the [A] runs.

- **P0c · finalise N, ρ0, T_res (§2.1) · exit: ρ_max decided and N 50 confirmed · F0 done 10-21.** Runs:
  [RUN_PLAN §3.1](RUN_PLAN.md#31-gating-runs-no-figure-they-block-the-rest) R1/R2 (ρ_max), N50 (N 50 vs P0's N 90),
  F1 (Yelp-P N check, optional), F2 (compute check after S), F3 (transfer and weight-norm checks inside S, P3, P4).
  Dropped: the `decay` gate, s sweeps and s fitting (s is derived), the cost cap R_max.
  - [ ] **T_res:** optional C3 sweep {150, 300, 600}; report as a decay constant (parked).
- **G0 · v2 charge profile (FT-N1) · exit: v2 sim block graded 0-fail · todo. Blocks every sim run.** Paired
  v2 real legs with the §2 recipe (`uniform`) (FT-L6, FT-T8).
- **P1 · [A] `distilbert_agnews`: `fwdllm` + `fluxtune`, 1 seed · exit: `e2e_distilbert_agnews.pdf` from
  real runs · blocked: G0.** These runs are also C1's A0 and A6.
- **P2 · C1 Phase A (A1–A4, 1 seed) · blocked: P1.** Its result picks the C3 groups worth running.
- **S · smoke `roberta_large_yelp` + `smollm2_1p7b_yahoo` · exit: both cells pass · running (jayne, operator, 10-18/19);
  `classifier.dense` drop is in code (10-22); RoBERTa runs before it read p +1,049,600. Feeds no figure.** `fwdllm` + `fluxtune`, 1 seed each, 2 h cap
  (4 runs). Pass: `[ProbeDim]` p = 591,874 / 806,912, no OOM or trainer death, FluxTune accuracy off chance
  (0.50 Yelp-P, 0.10 Yahoo), ρ not pinned at its cap. Record s/commit and vclock/wall to set N, the charge
  profile and the per-run time in P3/P4.
  - 1.7B · Yahoo dry run, 10-20, not an S leg (`run_20261020_015616`, 1.23 h; details and files [RUN_PLAN §3.3](RUN_PLAN.md#33-run-log-smollm2-17b--yahoo)):
    learns. p 806,912; 0 OOM / dead trainers; acc 0.141 → 0.266 at commit 92; ρ 0.0373 fixed (`pool`, not capped);
    43 s/commit at N 50, vclock/wall ~17×; distinct clients / pool 0.74. Ran at N 50 / `pool` / bs 8 (s 1.5), not the cell's `exp` / bs 32 / ρ0 0.07.
  - [ ] 1.7B · Yahoo both arms at the §2 recipe (`exp`, bs 32, n 75; measures bs 32 s/commit for F2 and the n 75 memory fit); RoBERTa · Yelp-P both arms; charge profiles (the dry run used the DistilBERT `fluxtune_yahoo.yaml`).
  - [ ] Per big cell: `fwdllm` lr screen, F2 bs check (§2).
- **P3 · [A] `roberta_large_yelp` · blocked: S.** 12 h wall cap, scored on TTA.
- **P4 · [A] `smollm2_1p7b_yahoo` · blocked: S.** 12 h wall cap (~380 commits at bs 32, est. from the measured bs 8 pace; RUN_PLAN §4), scored on TTA.
- **P5 · C2 cumulative ladder · blocked: P2.** Reuses P1/P2 arms where the config is identical.
- **P6 · C3 design choices · blocked: P2.** Only the concepts Phase A shows matter.
- **P7 · C4 async SOTA + availability · blocked: FT-N5** (unavailability wiring for the grad loop).
- **P8 · C5 generalisation · deferred (operator 10-22); blocked: P2, P3, P4.**

## 4. Experiments

### [A] End-to-end comparison (`figs_e2e.yaml`)

One figure row per cell, FwdLLM vs FluxTune:

| Panel | Shows | Callout |
|---|---|---|
| (a) | test accuracy vs simulated time | "N× faster" between the two arms' mean TTA |
| (b) | GB to target, downlink and uplink stacked, min–max whiskers over seeds | "N× less" |
| (c) | test accuracy vs commits (server model updates) | separates better steps from more steps per hour |

Outputs: `e2e_<cell>.pdf` per row, `e2e_grid.pdf` with all rows stacked.

### [B] How FluxTune gains over FwdLLM (same runs as [A])

- **B1 Client engagement** (`b1_<cell>.pdf`): CDF of how long a selected client must stay engaged. FluxTune
  works per iteration, so a client is needed for one task (dispatch → commit). FwdLLM holds its cohort for a
  whole round (every iteration of a data bin until the variance gate passes). Expected: minutes vs much longer.
- **B2 Selection diversity** (`b2_<cell>.pdf`): (a) share of clients that have contributed vs time; (b) Lorenz
  curve of contributions per client, with Gini.
- **B3 Progression** (`b3_<cell>.pdf`): (a) uploads pooled per commit; (b) cumulative commits vs time. Shows
  FluxTune committing at a small, steady, SNR-sized pool while FwdLLM's variance gate needs large pools.

### [C] Ablation (DistilBERT · AG News, `figs_ablation.yaml`)

**What separates FluxTune from FwdLLM.** E1–E3 should carry the per-commit gain and S1 the wall-clock gain.
These are predictions from code reading and single-seed runs, not controlled results.

| # | Concept | FwdLLM | FluxTune | Expected impact | Evidence |
|---|---|---|---|---|---|
| E1 | per-client gradient estimate | 1 of 10 probes, the one most aligned with the last gradient | mean of P=10 probes | high | ~10× better step direction and size at equal compute and bytes |
| E2 | server step | SGD, η=0.01 | trust ratio, exponential ρ decline | high | the fix for v1's peak-then-collapse |
| E3 | when to commit | variance ≤ 0.3 | fixed pool of N uploads (§2.1) | high | sets each step's noise; uploads independent (n_eff/N 1.000 median, §2.1) |
| S1 | execution | sync rounds | async (C = 0.3·n, K = C/3) | wall-clock only | removes the sync barrier |

Stale-update acceptance and Oort selection come from Felix, not FluxTune; FluxTune inherits them and they are not
ablated. FwdLLM's probe rule is in `tc_transformer_trainer_distribute.py:451-462`. FluxTune runs uniform weights
(§2): G1's `grad_aware` arm was within 0.6 pts at equal commits and 14% slower per commit (RUN_PLAN §0b);
C3 `agg_weights` tests it and Felix's `new` against uniform. Dynamic K/C is off in every run.

Each [C] group renders `c_<group>.pdf`: TTA bars (seed mean, min–max, DNF hatched), accuracy vs time, and
accuracy vs commits.

**C1 · Phase A: FwdLLM + one concept** (`phase_a`). If single arms gain little but A6 jumps, the concepts
depend on each other: add A1+A2, A1+A3, A2+A3.

| Arm | Adds to FwdLLM | Flags |
|---|---|---|
| A0 | nothing | `--only fwdllm` |
| A1 | E1 mean of 10 probes | `--probe-combine mean --perturbation-count 10` |
| A2 | E2 trust-ratio step | `--server-step-rule trust_ratio --rho-schedule exp --rho0 0.07 --t-res 283` (one run, no ρ grid) |
| A3 | E3 commit every N uploads | `--commit-gate fixed --pool-target {N} --var-stopping-policy off` |
| A4 | S1 async | `fedbuff_it_unaware` with FwdLLM's estimator settings |
| A6 | all = FluxTune | §2 recipe flags, `--pool-target {N}` |

**C2 · Phase B: cumulative ladder** (`phase_b`, headline ablation figure): FwdLLM → + E1 → + E2 → + E3 (still
sync) → + S1 async = FluxTune. Reorder E1–E3 if Phase A shows a dependency.

**C3 · Design choices** (one group each; *needs code* = not runnable today):

| Group | Arm | Standard name | Flags |
|---|---|---|---|
| `agg_criteria` | SNR target | closed-form pool size | `--commit-gate n_target` |
| | fixed pool N ∈ {10,20,90} (50 = A6) | batch-size sweep | `--commit-gate fixed --pool-target N` |
| `update_utility` (all P=10) | aligned with last gradient | FwdLLM rule, guided ES | A0 |
| | cos-sim weighted | — | *needs code* |
| | top-1 probe | — | *needs code* |
| | mean | FluxTune | A6 |
| `step_size` | trust ratio, constant ρ | LARS/LAMB | `pool`: P0's run reused (N 90, ρ 0.067) |
| | trust ratio, ρ0·t^−a | LR decay | `--rho-schedule rm --rho0 0.07 --rho-exp {a}`, a solved so Φ = Φ\* at the commit cap (≈ 0.1 at 1,000) |
| | trust ratio, ρ0·exp(−t/2T_res) | FluxTune | A6 |
| | trust ratio, ρ0/Φ | 1/Φ alternative | `--rho-schedule phi` (needs code) |
| `agg_weights` | uniform (ω = 1) | FedAvg-style mean | A6 |
| | staleness ω only | Felix | `--agg-rate-type new` |
| | staleness ω × alignment gate | FluxTune | `grad_aware`: G1's run reused (N 90, ρ 0.067) |

No plain-SGD arm: C1's A0 → A2 already compares FwdLLM's SGD with the trust ratio. No P sweep, variance-gate
sweep or iteration cap.

`step_size` also renders `c_step_size_step_size.pdf`: ρ and Φ per commit (needs `server_update_audit: true`).
If time allows: server momentum 0.5 (`--server-momentum 0.5`; 0.9 diverged), server Adam/Yogi (*needs code*),
normalised SGD and clipping (*needs code*), antithetic or orthogonal probes (*needs code*).

**C4 · Sync vs async, and availability** (`figs_availability.yaml`, `c_avail_<trace>.pdf`): DistilBERT · AG News,
one row per trace: `syn_0`, `mobiperf_3st_50`. Same [A] panels and metrics. The `syn_0` row shows async
is not the only source of FluxTune's gain: if it were, FedBuff(P) and Felix(P) would match FluxTune. The other rows
show how FluxTune holds up when clients come and go, against the availability-aware Felix.

| Arm | Execution | Gradient estimator and step | Traces |
|---|---|---|---|
| FwdLLM | sync | FwdLLM's | `syn_0` (reuses the P1 runs) |
| FedBuff(P) (`fedbuff_round`) | async | FwdLLM's | all |
| Felix(P) (`felix_round`) | async, availability-aware | FwdLLM's | all |
| FluxTune | async | E1–E3 | all (`syn_0` reuses the P1 runs) |

FedBuff(P) and Felix(P) keep their `baselines.yaml` lr (Felix 0.075 placeholder; operator 10-22).

**C5 · Generalisation (Phase D)** (`c5_generalisation.pdf`, `figs_ablation.yaml`). Rerun the 2–3 winning C1 concepts on
Yahoo and Yelp-P (DistilBERT), plus FwdLLM + the largest concept on SmolLM2-1.7B (P4 supplies both references).
Deferred (operator 10-22).

## 5. Scoring rules

- **TTA** = native time of the first eval that ends `tta_consecutive` evals in a row ≥ target (manifest key,
  2 in every cell). Native time is the sim vclock (wall time for real runs).
- **Bytes and commits to target** are counted up to that eval. A seed that never reaches target is DNF: its
  bytes and commits are whole-run, drawn faded and labelled DNF.
- **Seeds:** 1 per arm; same-seed replicate noise is 2.3 points (`parity_floors/fluxtune.yaml`),
  so a single-seed gap under that is unresolved. Bars show the seed mean with min–max. Curves show the arm's median-TTA seed, or its highest-peak seed if none reached.
- **Same stop rule in every arm:** fixed stall rule + GL decay + commit cap + wall cap. Prior runs stopped for three
  different reasons and are not comparable.
- **`server_update_audit: true` in every arm of a comparison** (it feeds ρ/Φ; its wall cost perturbs arrival
  order, so keep it symmetric). The cos ground-truth audit is off in every figure run.

## 6. Telemetry → metric map

All read by `fwdllm/expt_scripts/plotlib/reducers.py` (`load_run` → `RunResult`).

| Metric | Event.field | `RunResult` |
|---|---|---|
| accuracy curve | `agg_eval.test-accuracy` | `evals` |
| native time | `agg_round.vclock_now` | `native_s()` |
| commits, uploads per commit | `agg_round.commit_reason` ∈ {natural, cap, plateau}; `contributing_trainers` | `commit_ts`, `commit_uploads` |
| GB down / up | `comm{agg_to_trainer}` (aggregator) / `comm{trainer_to_agg}` (trainer files) | `bytes_until()` |
| engagement, async | `agg_round.contributor_intervals` (sim: vclock) | `session_durs` |
| engagement, sync | `agg_round` same-`data_id` span on the vclock | `round_span_durs` |
| selection diversity | `agg_round.contributing_trainers` | `contribs` |
| ρ, Φ, n_eff | `server_update` (needs `server_update_audit`) | `server_updates` |

## 7. Figure pipeline

```bash
cd lib/python/examples/fwdllm/expt_scripts
python make_paper_figs.py --manifest figs_e2e.yaml            # [A] + [B]
python make_paper_figs.py --manifest figs_ablation.yaml       # C1–C3
python make_paper_figs.py --manifest figs_availability.yaml   # C4
python make_paper_figs.py --manifest figs_e2e.yaml --synthetic   # layout preview, watermarked
```

- **Adding data:** append each seed's run dir (relative to `examples/fwdllm/`) to its arm's list in the
  manifest. Empty arms are skipped, so partial manifests render.
- **Outputs:** `paper_figs/<manifest>/*.pdf` (gitignored, overwritten each render), `summary.csv` (per seed:
  reached, TTA, GB down/up, commits, peak/final accuracy — the source of every number in the text) and
  `manifest.json`.
- **Code:** `plotlib/paper.py` (figures, metrics), `plotlib/reducers.py` (telemetry), `plotlib/synthetic.py`
  (preview data); tests in `plotlib/test_paper.py`. Colours and markers come from `plotlib/baselines.py`;
  ablation arms use FluxTune's blue ramp.

## 8. Predictions, gaps, caveats

**Predictions**
- E1 and E3 are both variance reduction (more probes vs more uploads per step), so their combined gain is less
  than the sum.
- S1 improves TTA, not accuracy per commit.

**Code gaps**
- [ ] `check_arm_health.py`: skip the `B_max origin` gate when `b_max_probe_every` is 0 (B_max is then a stale ln 2
  prior; A0's and `pool`'s FAILs are false alarms).
- [ ] Launcher flag for `stalenessPolicy` (today a yaml override).
- [ ] cos-sim and top-1 combine; normalised SGD; clipping; antithetic/orthogonal probes; server Adam/Yogi.
- [ ] Unavailability for the grad loop (FT-N5) for C4.

**Caveats**
- In fixed-pool mode the log line `[Variance=GOOD] var=0.79 <= thr=0.3` is misleading: the pool count, not
  the variance, triggered the commit.
- The prior 1.7B run used N=50 (milder non-IID) and the October DistilBERT runs used N=50 (simplified) or the
  `n_target` gate (full controller, mean pool 49.5), not today's N=90; their accuracies are not like for like.
- `learning_rate` is inert under trust ratio; the RoBERTa lr 3e-4 applies to the `fwdllm` arm only.
- RoBERTa-L and 1.7B vclock hours rest on pre-recipe dataset profiles until G0; quote their TTA in hours only after
  that (FLUXTUNE FT-T11). 1.7B runs before 10-24 used divisor 0.0192, so their vclock under-charges D 4.1×; their
  accuracy per commit stands.
- The big-model D assumes A40 cost ratios transfer to the phone NPU ([D_CALIBRATION §6](D_CALIBRATION.md#6-rules-and-caveats)); 1.7B's is likely a lower bound.

## 9. Legacy IDs cited in code

The docs that defined these are deleted; comments still cite them.

- **EXPTS_CHARTER A5:** α=1 for all FluxTune runs so far (§2 keeps α=1).
- **Issue I-1:** v1's peak-then-collapse at the lap boundary, caused by an undamped optimizer. Fixed by the
  trust-ratio step; FT-N4 tracks it.
- **BRIDGE_DESIGN decision #3:** new launches are sim-only. **§2b N3/N5/N6:** reducer fixes (converge.json,
  idle split, sync round span), tested in `plotlib/test_reducers.py`.
- **PLOT_TRACKER open-work #3/#4:** auto broken y-axis and CDF percentile marks in `plotlib/figures.py`.

Decided 2026-10-15: fixed pool (`n_target` is a C3 arm); α=1.
Decided 2026-10-16: cells are DistilBERT · AG News, RoBERTa-large · Yelp-P, SmolLM2-1.7B · Yahoo (§2); [A]/A6
FluxTune is the §2 recipe (simplified + exponential ρ decline + GL).
Decided 2026-10-17: targets 0.86 / 0.82 / 0.65; n 100 / 100 / 75; pool N rule and eval sampling in RUN_PLAN §2;
12 h wall cap on RoBERTa and 1.7B; 1 seed per arm; no SmolLM2-360M runs; C4 traces `syn_0` + `mobiperf_3st_50`.
Decided 2026-10-17 (later): pool N 90 / 100 / 70; DistilBERT eval = full test set, stride 4. Eval sample size and stride for Yelp-P and Yahoo stay open (RUN_PLAN §2).
Decided 2026-10-17 (latest): ρ = exponential decay (`exp`, T_res 300) in the baseline; N final, N-check run dropped.
Decided 2026-10-18: P0 A/B → `exp`, T_res 300 is the FluxTune recipe; `pool` is a C3 `step_size` arm (RUN_PLAN §0).
Decided 2026-10-18: G1 → uniform ω is the recipe; `grad_aware` is a C3 `agg_weights` arm (RUN_PLAN §0b).
Decided 2026-10-19: P0b → train batch size 32 for Yelp-P and Yahoo: same plateau as bs 8, reached in fewer vclock h and commits (RUN_PLAN §0c).
Proposed 2026-10-21: N 50 every cell, ρ0 0.07 every model, Φ\* 2 → T_res 283; s derived (§2.1). Open: ρ_max (R1/R2), N 50 check, DistilBERT figure runs at N 90 or 50, P4 compute (RUN_PLAN §2).
Decided 2026-10-22: 1.7B · Yahoo eval = 6,000 rows resampled per eval (stride set from S's bs 32 pace); `exp` for every model; every 1.7B run passes
`--no-cos-ground-truth-audit` (the audit's CPU backward hung and killed the 10-19 aggregator).
Decided 2026-10-22: `fwdllm` runs `jvp_perf_opt: true` (no diagnostic passes: 2 passes/task, not 5). Its D divisor (1.63, §O of FWDLLM_DESIGN.md, sized for 5) must be re-derived; A0 is stale.
Decided 2026-10-22: D = reference-NPU cost per task (passes × 0.3086 s/sample × bs / 2), no +1.5 s buffer, both arms; the floor alone guards overruns. `fluxtune` divisor 0.48 → 0.507; `fwdllm` → 5.07 (lands with its floor).
Decided 2026-10-22 (later): +6% buffer on both arms (D × 1.06, keeps the arm ratio; ≈ fluxtune's old +1.5 s); data alone needs none (A0's 5-pass GPU times, an upper bound on 2 passes: 0.15% overrun at the unfloored min D, 0% at floor 4.0). Divisors: `fluxtune` 0.478, `fwdllm` 4.78; `fwdllm` floor 7.0 → 4.0 (fluxtune's, same D shape). Confirm 0 overruns on the first 2-pass leg.
Decided 2026-10-22 (operator review): recipe, ρ0 0.07 (+ R1/R2, abort rule), Φ\* 2, stop rule, caps, targets, PEFT confirmed; T_res horizon-aware on
both 12 h cells; bs 32 unless F2 fails; C = 0.3·n, K = C/3; 1.7B n 75 → 60 on OOM; big-cell eval 10% of test; per-model D; `fwdllm` lr screens;
1 seed; `tta_consecutive` 2 everywhere; cos audit off everywhere; C3 trimmed (§4); `B_max origin` gate not applied to A0/`pool`; C5 deferred.
Decided 2026-10-24 (operator): ρ0 0.14 on RoBERTa-L and 1.7B `fluxtune`, T_res 71 (Φ\* 2); DistilBERT stays 0.07. Open: N 50 vs N\* × 4 (RUN_PLAN §6).
