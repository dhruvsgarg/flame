# FluxTune paper evaluation plan

The live plan for the FluxTune paper's evaluation. The doc rules in
[ROBUST_FL_READINESS.md](../_metadata/ROBUST_FL_READINESS.md) apply: every line is true now, no changelog, a
finished item is deleted. Infra and parity status live in [FLUXTUNE_READINESS.md](../_metadata/FLUXTUNE_READINESS.md);
figures come from `fwdllm/expt_scripts/make_paper_figs.py` (§7). Plot list, run list and measured run times:
[RUN_PLAN.md](RUN_PLAN.md).

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

Common to every run: sim mode, `syn_0` (except C4), highly non-IID partition (Dirichlet α=1), ≥ 3 seeds per arm, the
same stop rule in every arm (§5). Clients N depend on model size; once set for a cell, every arm in that cell
uses it. We run one model per dataset (no full model × dataset grid): small, mid and large, the largest on the
hardest task (Yahoo: 10 classes, FwdLLM paper target 0.65). Trainable p rises with model size (0.45M → 0.59M →
0.81M), keeping the probe dimension in one band.

| Cell (manifest id) | Model (total), PEFT, trainable params | Dataset (seq len) | N | Target | Prior evidence (FluxTune, single seed) |
|---|---|---|---|---|---|
| `distilbert_agnews` | DistilBERT (67M), adapters rf=16, 450,340 | AG News (192) | 100 | 0.80 (pencilled) | 0.873 peak in 3 h (full controller); 0.8726 in 1.74 h (simplified) |
| `roberta_large_yelp` | RoBERTa-large (355M), LoRA q,v r=6, `classifier.dense` dropped, 591,874 | Yelp Polarity (256) | TBD | 0.80 (pencilled; paper 0.82) | none; the rf=16 smoke (p 4.2M) sat at chance after 2 h; needs lr 3e-4 |
| `smollm2_1p7b_yahoo` | SmolLM2-1.7B (1.7B), LoRA q,v r=4, 806,912 | Yahoo Answers (256) | TBD (50 in the prior run) | 0.65 (pencilled; paper 0.65) | DistilBERT ≈ 0.73 on Yahoo; SmolLM2-1.7B 0.832 on AG News after 8 h (`run_20261014_034646`, N=50, α=1) |

DistilBERT stays on AG News so the [C] ablations share the [A] cell (A0/A6 reuse its runs).

**FluxTune recipe** (every FluxTune arm: [A], A6, C4). The simplified controller
([cheatsheet §2](../fwdllm/fl_fwd_ft_pipeline_cheatsheet.md#2-flowcharts)) plus three concepts it dropped:

| Restored | Setting | Flag | Why |
|---|---|---|---|
| ρ decline | ρ_t = ρ0/Φ_t (constant absolute step) | `--rho-schedule phi` (needs code, P0) | constant ρ grows the absolute step 3× (0.67 → 2.0); the simplified arm overshot its peak (Φ 2.68), and SmolLM2 sat at ρ 0.037 throughout |
| grad-aware weights | staleness ω (Felix `new`) × alignment gate | `--agg-rate-type grad_aware` | measured near-inert on AG News (≤ 0.6% weight change); restored as recipe, `uniform` is the C3 control |
| GL decay stop | on (fires on 20 evals past commit 450, > 0.5% below best) | `--saturation-stop` without `--no-sat-decay` | free collapse guard; decline untested on Yelp-P and Yahoo |

Unchanged from the simplified controller: P=10 `mean`, trust ratio, s=1.5, fixed pool N, stall rule 0.003, Φ rail.
Under `phi`, Φ=3 takes 8/ρ0² commits, so the commit cap (§5) usually ends the run.

```
--probe-combine mean --perturbation-count 10 --server-step-rule trust_ratio --gate-safety-s 1.5
--commit-gate fixed --pool-target {N} --rho-schedule phi --agg-rate-type grad_aware
--var-stopping-policy off --budget-stop-frac 0 --b-max-probe-every 0 --phi-stop halt
--saturation-stop --sat-stall-frac 0.003 --server-update-audit
```

Runs from before the ρ anneal (§3 P0), including all July runs, are FluxTune v1 or pre-anneal and do not feed
any figure.

## 3. Run queue (top = next)

Each item: action · exit · state. [B] adds no runs: it reads the [A] runs.

- **P0 · ρ anneal + stop-rule fix · exit: A/B decided · todo. Blocks everything below.**
  Today `--rho-schedule pool` fixes ρ = s·√(P·N/p) for the whole run (`FedSgdAggregator.py:278`; s=1.5, P=10,
  N=50 → ρ 0.050 DistilBERT, 0.037 SmolLM2-360M, 0.038 1.7B). Each step is sideways (|cos(θ, Δθ)| < 0.0006),
  so ‖θ_t‖² = ‖θ_{t−1}‖²(1+ρ²) and Φ grows from ρ alone; late constant-ρ commits mostly add noise (simplified
  arm peaked at Φ 2.68, then dipped). Proposal: ρ_t = ρ0/Φ_t, so Φ_t = √(1 + t·ρ0²); no new knob.
  - [ ] `--rho-schedule phi` in `_rho_star_now` (ρ0 from the `pool` formula or `--rho-star`).
  - [ ] Fix the stall-rule streak reset (one noisy eval zeroes it); add a commit cap.
  - [ ] A/B on `distilbert_agnews`: §2 recipe (`phi`, ρ0 0.07, N 98) vs the same with `pool`, cap ~1,000
    commits, ≥ 2 seeds; compare TTA at 0.80/0.86/0.87, peak accuracy, Φ at peak. `phi` is the recipe; `pool` is a
    C3 arm. If `pool` wins, revisit the recipe.
  - [ ] Decide how ρ0 scales with p (hold N, ρ0 ∝ 1/√p; or hold ρ0, N ∝ p) with one SmolLM2-360M pair.
- **P0b · Finalise how pool size N is chosen per model · exit: procedure validated and frozen · todo. Blocks
  every experiment below, S excepted.** The draft procedure ([cheatsheet §8](../fwdllm/fl_fwd_ft_pipeline_cheatsheet.md#8-choosing-the-pool-size-n-per-model-draft-not-final))
  is not final. Pool size N (gradients per commit) is not the §2 table's N (clients). Open: the √p scaling of N
  (subsumes P0's ρ0-scaling item), the ρ band 0.03–0.07, the distinct-data threshold, and raising bs where the run
  sees under one pass of the data (Yelp-P, Yahoo).
  - [ ] SmolLM2-360M pair: N = 50 vs 70, compared at matched gradients.
  - [ ] Yahoo bs A/B (8 vs 32) at fixed N.
  - [ ] Freeze the procedure and record each cell's N in §2.
- **P1 · [A] `distilbert_agnews`: `fwdllm` + `fluxtune` × 3 seeds · exit: `e2e_distilbert_agnews.pdf` from
  real runs · blocked: P0.** These runs are also C1's A0 and A6.
- **P2 · C1 Phase A (A1–A4 × 3 seeds) · blocked: P0.** Its result picks the C3 groups worth running.
- **S · smoke `roberta_large_yelp` + `smollm2_1p7b_yahoo` · exit: both cells pass · blocked (RoBERTa only):
  `classifier.dense` drop. Not blocked by P0; feeds no figure.** `fwdllm` + `fluxtune`, 1 seed each, 2 h cap
  (4 runs). Pass: `[ProbeDim]` p = 591,874 / 806,912, no OOM or trainer death, FluxTune accuracy off chance
  (0.50 Yelp-P, 0.10 Yahoo), ρ not pinned at its cap. Record s/commit and vclock/wall to set N, the charge
  profile and the per-run time in P3/P4.
- **P3 · [A] `roberta_large_yelp` · blocked: P0, ρ0 scaling, S.** Per run between DistilBERT (~3 h) and SmolLM2
  (~10 h); S sets it.
- **P4 · [A] `smollm2_1p7b_yahoo` · blocked: P0, ρ0 scaling, S.** ~10–11 h per run (seq 256 ≈ 1.33× AG News's
  ~8 h).
- **P5 · C2 cumulative ladder · blocked: P2.** Reuses P1/P2 arms where the config is identical.
- **P6 · C3 design choices · blocked: P2.** Only the concepts Phase A shows matter.
- **P7 · C4 async SOTA + availability · blocked: FT-N5** (unavailability wiring for the grad loop).
- **P8 · C5 generalisation · blocked: P2, P3, P4.**

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
| E2 | server step | SGD, η=0.01 | trust ratio, ρ0/Φ decline | high | the fix for v1's peak-then-collapse |
| E3 | when to commit | variance ≤ 0.3 | fixed pool of 50 uploads | high | sets each step's noise; pool ~99% effective (n_eff 49.5 of 50) |
| S1 | execution | sync rounds | async (C=30, K=10) | wall-clock only | removes the sync barrier |

Stale-update acceptance and Oort selection come from Felix, not FluxTune; FluxTune inherits them and they are not
ablated. FwdLLM's probe rule is in `tc_transformer_trainer_distribute.py:451-462`. FluxTune runs `grad_aware` weights
(§2), though they measured near-inert (≤ 0.6% weight change; [FLUXTUNE_CODE_QA.md](../fwdllm/FLUXTUNE_CODE_QA.md):203-217);
C3 `agg_weights` tests them against uniform. Dynamic K/C is off in every run.

Each [C] group renders `c_<group>.pdf`: TTA bars (seed mean, min–max, DNF hatched), accuracy vs time, and
accuracy vs commits.

**C1 · Phase A: FwdLLM + one concept** (`phase_a`). If single arms gain little but A6 jumps, the concepts
depend on each other: add A1+A2, A1+A3, A2+A3.

| Arm | Adds to FwdLLM | Flags |
|---|---|---|
| A0 | nothing | `--only fwdllm` |
| A1 | E1 mean of 10 probes | `--probe-combine mean --perturbation-count 10` |
| A2 | E2 trust-ratio step | `--server-step-rule trust_ratio --rho-schedule const --rho-star {grid}` |
| A3 | E3 commit every 50 uploads | `--commit-gate fixed --pool-target 50 --var-stopping-policy off` |
| A4 | S1 async | `fedbuff_it_unaware` with FwdLLM's estimator settings |
| A6 | all = FluxTune | §2 recipe flags, `--pool-target 50` |

**C2 · Phase B: cumulative ladder** (`phase_b`, headline ablation figure): FwdLLM → + E1 → + E2 → + E3 (still
sync) → + S1 async = FluxTune. Reorder E1–E3 if Phase A shows a dependency.

**C3 · Design choices** (one group each; *needs code* = not runnable today):

| Group | Arm | Standard name | Flags |
|---|---|---|---|
| `agg_criteria` | variance ≤ thr | FwdLLM / adaptive batch | `--commit-gate var --var-threshold {0.1,0.3,1.0}` |
| | hard stop after I iterations | fixed iteration budget | `max_iterations_per_data_id` cap |
| | SNR target | closed-form pool size | `--commit-gate n_target` |
| | fixed pool N ∈ {10,20,50,100} | batch-size sweep | `--commit-gate fixed --pool-target N` |
| `update_utility` | aligned with last gradient | FwdLLM rule, guided ES | A0 |
| | cos-sim weighted | — | *needs code* |
| | top-1 probe | — | *needs code* |
| | 1 random probe | SPSA / MeZO | `--perturbation-count 1 --probe-combine mean` |
| | mean of P ∈ {2,5,10,20} | multi-sample estimate | `--perturbation-count P --probe-combine mean` |
| `step_size` | SGD, tuned η grid | plain SGD | `--server-step-rule raw_sgd --learning-rate {grid}` |
| | trust ratio, constant ρ | LARS/LAMB | `--rho-schedule const` / `pool` |
| | trust ratio, ρ0·t^−a | LR decay | `--rho-schedule rm` |
| | trust ratio, ρ0/Φ | FluxTune | `--rho-schedule phi` |
| `agg_weights` | uniform (ω = 1) | FedAvg-style mean | `--agg-rate-type uniform` |
| | staleness ω only | Felix | `--agg-rate-type new` |
| | staleness ω × alignment gate | FluxTune | `--agg-rate-type grad_aware` |

`step_size` also renders `c_step_size_step_size.pdf`: ρ and Φ per commit (needs `server_update_audit: true`).
If time allows: server momentum 0.5 (`--server-momentum 0.5`; 0.9 diverged), server Adam/Yogi (*needs code*),
normalised SGD and clipping (*needs code*), antithetic or orthogonal probes (*needs code*).

**C4 · Sync vs async, and availability** (`figs_availability.yaml`, `c_avail_<trace>.pdf`): DistilBERT · AG News,
one row per trace: `syn_0`, `syn_50_3st`, `mobiperf_3st`. Same [A] panels and metrics. The `syn_0` row shows async
is not the only source of FluxTune's gain: if it were, FedBuff(P) and Felix(P) would match FluxTune. The other rows
show how FluxTune holds up when clients come and go, against the availability-aware Felix.

| Arm | Execution | Gradient estimator and step | Traces |
|---|---|---|---|
| FwdLLM | sync | FwdLLM's | `syn_0` (reuses the P1 runs) |
| FedBuff(P) (`fedbuff_round`) | async | FwdLLM's | all |
| Felix(P) (`felix_round`) | async, availability-aware | FwdLLM's | all |
| FluxTune | async | E1–E3 | all (`syn_0` reuses the P1 runs) |

**C5 · Generalisation (Phase D).** Rerun the 2–3 winning C1 concepts on Yahoo and Yelp-P (DistilBERT), plus one
FwdLLM-vs-FluxTune pair on SmolLM2-1.7B for the largest concept.

## 5. Scoring rules

- **TTA** = native time of the first eval that ends `tta_consecutive` evals in a row ≥ target (manifest key,
  default 1). Native time is the sim vclock (wall time for real runs).
- **Bytes and commits to target** are counted up to that eval. A seed that never reaches target is DNF: its
  bytes and commits are whole-run, drawn faded and labelled DNF.
- **Seeds:** ≥ 3 per arm; same-seed replicate noise is 2.3 points (`parity_floors/fluxtune.yaml`). Bars show
  the seed mean with min–max. Curves show the arm's median-TTA seed, or its highest-peak seed if none reached.
- **Same stop rule in every arm:** fixed stall rule + GL decay + commit cap + wall cap (P0). Prior runs stopped for three
  different reasons and are not comparable.
- **Step-rule arms:** tune η for plain SGD on a grid, so a comparison is not just a learning-rate mismatch.
- **`server_update_audit: true` in every arm of a comparison** (it feeds ρ/Φ; its wall cost perturbs arrival
  order, so keep it symmetric).

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
- Plain SGD with tuned η may match the trust ratio early, then drift as the trainable weights' norm grows.
- S1 improves TTA, not accuracy per commit.

**Code gaps**
- [ ] `--rho-schedule phi`, stall-rule fix, commit cap (P0).
- [ ] Seed-matched probe arms: changing P changes the random draws consumed, so trajectories diverge for
  reasons unrelated to the combine rule.
- [ ] Launcher flag for `stalenessPolicy` (today a yaml override).
- [ ] cos-sim and top-1 combine; normalised SGD; clipping; antithetic/orthogonal probes; server Adam/Yogi.
- [ ] Unavailability for the grad loop (FT-N5) for C4.
- [ ] Drop `classifier.dense` for `roberta-large`, as `pre_classifier` is for DistilBERT
  (`tc_transformer_trainer_distribute.py:216-217`); kept, it adds 1,049,600 to p.

**Caveats**
- In fixed-pool mode the log line `[Variance=GOOD] var=0.79 <= thr=0.3` is misleading: the pool count, not
  the variance, triggered the commit.
- The prior 1.7B run used N=50 (milder non-IID) and DistilBERT used N=100; their accuracies are not like for
  like.

## 9. Legacy IDs cited in code

The docs that defined these are deleted; comments still cite them.

- **EXPTS_CHARTER A5:** α=1 for all FluxTune runs so far (§2 keeps α=1).
- **Issue I-1:** v1's peak-then-collapse at the lap boundary, caused by an undamped optimizer. Fixed by the
  trust-ratio step; FT-N4 tracks it.
- **BRIDGE_DESIGN decision #3:** new launches are sim-only. **§2b N3/N5/N6:** reducer fixes (converge.json,
  idle split, sync round span), tested in `plotlib/test_reducers.py`.
- **PLOT_TRACKER open-work #3/#4:** auto broken y-axis and CDF percentile marks in `plotlib/figures.py`.

## Open questions (operator)

- Final target accuracy per cell (pencilled until checked against the paper), and N for the RoBERTa-large and
  SmolLM2-1.7B cells.
- C4 traces: `syn_50_3st` + `mobiperf_3st`, or also `syn_20_3st`?

Decided 2026-10-15: fixed pool (`n_target` is a C3 arm); α=1.
Decided 2026-10-16: cells are DistilBERT · AG News, RoBERTa-large · Yelp-P, SmolLM2-1.7B · Yahoo (§2); [A]/A6
FluxTune is the §2 recipe (simplified + ρ decline + grad-aware weights + GL).
