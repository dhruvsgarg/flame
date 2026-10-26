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
- **FwdLLM plateaus on non-IID data** (the FwdLLM paper also reports it never reaches the IID maximum), so the
  target may be DNF for it; speedups then follow the §5 rule (time to its peak, plus "≥ X×" at target).
- **Primary comparison:** FluxTune (§2 recipe, = C1's A6) vs FwdLLM. Only these two appear in [A] and [B].
- **Secondary baselines:** FedBuff(P) and Felix(P) (`fedbuff_round`, `felix_round`, forward perturbation). They
  appear only in C4, which answers "are the gains just async FL?" (FwdLLM as the sync reference) without diluting
  the main comparison.
- **Availability is not a FluxTune theme.** [A]–C3 run at 100% availability (`syn_0`). Availability appears only
  in C4, against the availability-aware Felix.

## 2. Setup

Common to every run: sim mode, `syn_0` (except C4), highly non-IID partition (Dirichlet α=1), 1 seed per arm, the
same stop rule in every arm (§5). Clients n, pool size N, s (→ ρ0), Φ_budget (→ T_res), wall cap and eval sampling per cell:
[RUN_PLAN.md §2](RUN_PLAN.md#2-cells-pool-size-and-eval); every arm in a cell uses them. We run one model per dataset (no full model × dataset grid): small, mid and large, the largest on the
hardest task (Yahoo: 10 classes, FwdLLM paper target 0.65). Trainable p rises with model size (0.45M → 0.59M →
0.81M), keeping the probe dimension in one band.

| Cell (manifest id) | Model (total), PEFT, trainable params | Dataset (seq len) | Clients n | Pool N | Target | Prior evidence (FluxTune, single seed) |
|---|---|---|---|---|---|---|
| `distilbert_agnews` | DistilBERT (67M), adapters rf=16, 450,340 | AG News (192) | 100 | 50 if the N50 check passes, else 90 | 0.86 | 0.873 peak in 3 h (full controller); 0.8726 in 1.74 h (simplified) |
| `roberta_large_yelp` | RoBERTa-large (355M), LoRA q,v r=6, `classifier.dense` dropped, 591,874 | Yelp Polarity (256) | 100 | 50 | 0.82 | 0.732 at c369 (10-24 shepherd, ρ0 0.14, T_res 71, 380-commit cap; budget-bound at Φ 2.0) |
| `smollm2_1p7b_yahoo` | SmolLM2-1.7B (1.7B), LoRA q,v r=2, 413,696 (`fwdllm` arm r=4) | Yahoo Answers (256) | 60 | 50 | 0.65 | DistilBERT ≈ 0.73 on Yahoo; SmolLM2-1.7B 0.832 on AG News after 8 h (`run_20261014_034646`, N=50, α=1) |

DistilBERT stays on AG News so the [C] ablations share the [A] cell (A0/A6 reuse its runs).

**Sim mode (every figure run).** Trainers do the real compute; each task's device delay D is charged to the vclock instead
of slept, so a cell runs 3.4× (DistilBERT) to 33× (RoBERTa-L) faster than real ([RUN_PLAN §4](RUN_PLAN.md#4-run-time-basis)).
Every time axis and TTA is vclock. A sim number stands for real only when (1) the cell's D divisor (below, measured 10-24)
and charge profile (G0: AG News, Yahoo final 10-25; Yelp-P stale) hold and (2) real and sim agree at matched commits (checked on
DistilBERT · AG News, 10-24). Run rules: [RUN_PLAN §2](RUN_PLAN.md#2-cells-pool-size-and-eval) "Sim and real legs".

**Eval:** DistilBERT scores the full test set every 4th commit. Yelp-P and Yahoo score 10% of the test set (3,800 / 6,000
rows, ≈ ±1.2 pt at target), resampled per eval; stride 4 / 3 (= ⌈t_eval / s_commit⌉ + 1; on the bench an eval adds ~10% wall and 0 vclock).

**Task cost D per model** (`--delay-divisor`; D = device delay / divisor). Calculation, measurements, full D tables and
how to re-derive for a new model: **[D_CALIBRATION.md](D_CALIBRATION.md)**.

| Cell | bs | `fluxtune` divisor | `fwdllm` divisor | `fluxtune` D: min – median – max (100 trainers) |
|---|---|---|---|---|
| DistilBERT · AG News | 8 | 0.478 | 4.78 | 8.4 s – 23 s – 1.6 min |
| RoBERTa-L · Yelp-P | 32 | 0.0169 | 0.169 | 3.9 min – 10.8 min – 46 min |
| SmolLM2-1.7B · Yahoo | 32 | 0.0047 | 0.047 | 14 min – 39 min – 2.8 h |

**FluxTune recipe** (every FluxTune arm: [A], A6, C4). The simplified controller
([cheatsheet §2](../fwdllm/fl_fwd_ft_pipeline_cheatsheet.md#2-flowcharts)) plus two concepts it dropped:

| Restored | Setting | Flag | Why |
|---|---|---|---|
| ρ decline | ρ_t = ρ0·exp(−(t−1)/(2·T_res)), T_res = 2·lnΦ_budget/ρ0² (§2.1); landing's decay without B_max, stop or rail | `--rho-schedule exp --t-res {T_res}` (P0 chose it 10-18) | constant ρ grows the absolute step 3× (0.67 → 2.0); the simplified arm overshot its peak (Φ 2.68), and SmolLM2 sat at ρ 0.037 throughout |
| GL decay stop | DistilBERT only (fires on 20 evals past the warm-up, 0.45 × `--max-commits`, > 0.5% below best); big cells stop on target | `--saturation-stop` without `--no-sat-decay` | free collapse guard |

**Big cells add (decision 15, RUN_PLAN):** server momentum 0.9 (`--server-momentum 0.9`); stop only at target ×2 (`--target-stop {target} --target-stop-window 2`;
commit cap 100,000 and Φ rail 1e9 are inert, saturation stop off); bench, one leg per 4 GPUs.

Unchanged from the simplified controller: P=10 `mean`, trust ratio, fixed pool N, uniform ω (G1 kept it 10-18;
`grad_aware` is a C3 arm), stall rule 0.003 (leaky streak; DistilBERT), Φ rail (DistilBERT). Train batch size 8 on AG News, 32 on Yelp-P and Yahoo (P0b).
Async: C = 0.3·n clients in flight, K = C/3 (30 / 10 at n 100). s, ρ0 and T_res are per cell (§2.1). Under `exp`, Φ is bounded by Φ_budget = exp(ρ0²·T_res/2): 2 on DistilBERT (the Φ rail
never fires; the commit cap (§5) ends the run), 4 on RoBERTa-L, ≈ 210 on 1.7B (rail off on both).
Stop-rule warm-up and horizon are 450 / 150 at a 1,000-commit cap and scale with `--max-commits` (`expts/saturation_stop.py`).

```
export FWDLLM_FD_SCALE_INVARIANT=1   # required when p != 450,340 (FT-T9)
--probe-combine mean --perturbation-count 10 --server-step-rule trust_ratio {--gate-safety-s {s} | --rho0 {ρ0}}  # §2.1
--commit-gate fixed --pool-target {N} --rho-schedule exp --t-res {T_res} --agg-rate-type uniform
--train-batch-size {8 AG News | 32 Yelp-P, Yahoo}
--var-stopping-policy off --budget-stop-frac 0 --b-max-probe-every 0 --phi-stop halt
--saturation-stop --sat-stall-frac 0.003 --server-update-audit --no-cos-ground-truth-audit
--max-commits {cap} --final-full-eval [--eval-max-samples {rows} --eval-resample]   # rows: Yelp-P, Yahoo
# big cells: --server-momentum 0.9 --target-stop {target} --target-stop-window 2 --max-commits 100000 --phi-stop-threshold 1e9, no --saturation-stop
```

Runs from before the ρ anneal (§3 P0), including all July runs, are FluxTune v1 or pre-anneal and do not feed
any figure.

**FwdLLM arm (`fwdllm`): settings changed from our earlier runs so it trains (operator 10-25).** Upstream reference
(github.com/UbiquitousLearning/FwdLLM): lr 0.01 for every model and dataset; var threshold per model (roberta-large 0.2, distilbert
0.1, none for llama); BitFit for non-DistilBERT models; a fresh client sample after each var-passing update (`client_sampling(round_idx)`
in `FedAvgServerManager_var.py`); 1,000 uniform clients. Evidence: [RUN_PLAN §3.5](RUN_PLAN.md#35-sanity-and-smoke-runs-no-figure).

| Change | Before | Now | Why | Upstream-faithful? |
|---|---|---|---|---|
| Cohort cadence | `round`: one 10-client cohort for a full lap (163 commits on Yelp-P) | `reselect_cadence: data_bin`: new cohort per commit; clients fixed through the var iterations of that commit | the pinned cohort's label mix (58% class 1 vs 50% across the 20 clients) drove RoBERTa to predict one class (class-1 share 0.73 → 0.996, MCC ~0); a different cohort drove it to class 0 | yes (upstream resamples per update) |
| RoBERTa-L PEFT | LoRA q,v r 6 (as FluxTune) | BitFit | FwdLLM's own choice for roberta-large; LoRA at lr 0.01 sat at chance (rb60) | yes; the arms differ in PEFT, so p differs |
| 1.7B var threshold | 0.2 (RoBERTa's value) | grid {15, 8.4, 4.6}, screened | 1.7B var is ~45× RoBERTa's (113 at iteration 1, levelling off at 0.45), so 0.2 never passed: 0 commits in 287 iterations. The grid matches RoBERTa's 8 / 13 / 27 iterations per commit at 0.4 / 0.2 / 0.1 | no upstream value for llama |
| Stuck-client eviction | 300 s without an accepted update (vclock in sim) | max(300 s, `send_timeout_wait_s`); 3600 s on big cells | 1.7B tasks charge D up to ~400 s, so live clients were evicted mid-iteration (bench: crash; harness: 0 commits) | bug fix (not in upstream) |
| lr | 0.01 | screened per cell (RoBERTa {1e-3, 3e-3, 1e-2, 3e-2}; 1.7B {3e-4, 1e-3, 3e-3}) | best of a screen, not a fixed value | upstream used 0.01 everywhere |

**Current choices (10-25 06:20; smoke legs at n 20 / 10, `data_bin`):**

| Cell | lr | var threshold | Status | Evidence |
|---|---|---|---|---|
| SmolLM2-1.7B · Yahoo (LoRA q,v r 4) | **3e-3** | **8.4** | final (10-25 13:58: var 15 and lr 5e-3 both worse) | lr 3e-3 beats 1e-3 and 3e-4 at matched commits (c13: acc 0.305 vs 0.199; all at var 8.4); 1e-2 overshoots (c1 loss 4.10 vs 2.57). Smoke at the final values (n 10, `data_bin`, 2 h, 67 commits, ~10 iterations per commit): acc 0.108 → 0.222 (c9) → 0.306 (c25) → 0.368 (c49) → **0.384** (c65); loss 2.57 → **2.03** (c57); MCC +0.32 |
| RoBERTa-L · Yelp-P (BitFit) | **1e-2** | **0.2** | final (10-25 07:15) | the only setup with a steady gain (best loss 0.678, MCC +0.13 at c82); s ≈ 1.76 at p 273,410; 1e-3 / 3e-3 under-step, 3e-2 noise-dominated; var 0.4 the same per commit but noisier; LoRA r 6 (lr 2.5e-3) flatter on α 1 and α 100. Progress per commit is small (Λ ≈ 0.05 by c90): expect `fwdllm` to miss 0.82 in 12 h |
| DistilBERT · AG News | 0.01 | 0.3 (`baselines.yaml`; upstream 0.1) | final (RUN_PLAN decision 14) | A0 |

Closed 07:15: RoBERTa lr / var; LoRA vs BitFit (LoRA flatter, no LoRA grid).
Open diagnostic: RoBERTa-L `fwdllm` on AG News (model vs task). Details: [RUN_PLAN §3.5](RUN_PLAN.md#35-sanity-and-smoke-runs-no-figure).

Kept from before: sync, K = c = 10 clients per iteration, var gate on (`var_control`), `perturbation_sampling`, no iteration cap,
`FWDLLM_FD_SCALE_INVARIANT=1`, Dirichlet α 1 (upstream: uniform). The DistilBERT · AG News `fwdllm` arm (run 1) gets the same cadence
change; A0 (peak 0.851) ran pinned cohorts.

### 2.1 Pool N, step s → ρ0, and Φ_budget → T_res

Inputs are N, s and Φ_budget; ρ0 and T_res follow per model. Rule, derivation and measurements:
[cheatsheet §8](../fwdllm/fl_fwd_ft_pipeline_cheatsheet.md#8-choosing-n-ρ0-and-t_res); runs: RUN_PLAN §3.1.

- **s** = ρ0 / cos(pool, g), step size in units of the pool's direction quality, cos = √(P·N/p). Unitless, so it transfers
  across models where a fixed ρ0 does not (ρ0 0.14 is s 3.1 on DistilBERT at N 90, 5.6 on 1.7B at N 50).
- **Φ** = ‖θ_t‖/‖θ_0‖ = e^B, B = ½Σln(1+ρ_t²) = 1/cos(θ_t, θ_0); exact from step sizes alone. Φ_budget = exp(ρ0²·T_res/2) is
  the most Φ an `exp` run can reach; every schedule fixes one, so pick it instead of inheriting it.
- **Derived:** ρ0 = s·√(P·N/p); T_res = 2·lnΦ_budget/ρ0²; commits to Φ = 3 ≈ T_res·ln(1/(1 − ln3/lnΦ_budget)) (1.57·T_res at Φ_budget 4).
- **Accuracy follows Λ = 2B/s**, schedule-independent (buildplan §5.3, §5.12). At the Φ ≈ 3 peak, Λ\* = 2·ln3/s: smaller s banks
  more progress per unit Φ but costs pooling ∝ 1/s² (wall ∝ lnΦ/s²).

| Knob | Value | Rests on | Gap |
|---|---|---|---|
| N | 50, every cell | `N* ≈ 29·Λ_need²` per dataset · bs: AG News 34–42, Yahoo bs 32 38 (F0); p cancels across models; DistilBERT run 2 reached 0.86 at N 50 | Yelp-P unmeasured; p cancellation rests on one point |
| s | **2.4 RoBERTa-L, 2.81 1.7B** (operator 10-26, decision 15); DistilBERT keeps ρ0 0.07 (s 2.10) | 1.5 validated on DistilBERT's three tasks (0.872 / 0.664 / 0.810, within 0.008 of the FL targets); without momentum, s > 2 under-trains (s 2.9 reaches the rail with half the Λ, buildplan §5.12). 1.7B at s 2.81: 0.598 at Φ_budget 2 (budget-bound, §3.4); with momentum 0.9 it led every smoke (§2.2) | s 2.4 / 2.81 + momentum unproven at full n until P3 / P4 |
| ρ0 | s·√(P·N/p) at P 10, N 50: **0.0698 RoBERTa-L, 0.0978 1.7B (r 2, passed as `--rho0`)**; 0.07 DistilBERT | derived | — |
| Φ_budget | **4 RoBERTa-L; ≈ 210 1.7B** (T_res 1,119 at ρ0 0.0978; ρ only halves by c1,500); 2 on DistilBERT | DistilBERT peaks at Φ\* 2.82 / 3.00 / 2.91 (22-run ledger: 2.4–3.3); 4 sits above every peak. Φ\* is knowable only from a run: the Φ_knee probe tracks the trajectory, not Φ\* (buildplan §5.6). The run checks itself: a peak before Φ 4 found Φ\*; accuracy still rising near Φ 4 means raise it | Φ\* unmeasured beyond DistilBERT |
| T_res | **570 RoBERTa-L** (= 2·ln4/ρ0²), **1,119 1.7B** (set, operator); 283 DistilBERT (Φ_budget 2) | derived / set | momentum 0.9 makes the realised step larger than ρ; Φ read from the run |

RoBERTa-L passes `--gate-safety-s 2.4 --t-res 570` without `--rho0`; 1.7B passes `--rho0 0.0978`; under `exp` with `--pool-target` the aggregator derives ρ0
and logs `[ServerStep] … rho*= s=`. DistilBERT passes `--rho0 0.07` (overrides `--gate-safety-s`).
**n_eff cannot set N.** `n_eff = N·(1 + O(P/p))`: all six recent runs read n_eff/N = 1.000 median (≥ 0.956), since
each upload is ~p/P ≈ 45,000× more probe noise than gradient. It detects only correlated probes; log it, don't gate on it.

**Fixed N, decaying ρ.** Each commit reads one data bin; the `exp` tail keeps reading bins with small steps while Φ
barely moves. A pool shrinking with ρ is time-optimal (solution §4.6) but reads less data per tail commit.

In the paper: N is the number of client gradients per aggregation, set once per dataset; s is the step relative to
the pool's direction quality; T_res spends a chosen Φ budget.

### 2.2 Concepts: tried, kept, tuned, general

Single seed throughout (noise 2.3 pt); "kept" = in the recipe today. Runs: RUN_PLAN §0–§0c, §3.4–§3.7.

| Concept | Verdict | Evidence | Tune per cell? |
|---|---|---|---|
| Mean of P = 10 probes (E1) | kept | ag60: fastest single concept (A1 0.31 vclock h to 0.60 vs FwdLLM 2.27) | no |
| More probes per client (P 40) | dropped | 1.7B: 4× cost per commit for ~4× progress (§3.7 `sm_k40s`) | no: compute-neutral |
| Trust-ratio step, `exp` ρ decline (E2) | kept | P0: `exp` 0.878 vs constant ρ 0.872, no collapse over 7 lap wraps | form no; s, Φ_budget yes |
| Step s = ρ0 / cos(pool, g) | kept as the unit | fixed ρ0 across p handicaps arms (§3.7: s 2.01 / 1.40 vs 2.81) | yes: 2.10 / 2.4 / 2.81 |
| Φ_budget (→ T_res) | set explicitly | budget-bound plateaus: 1.7B 0.598 at budget 2 (85% spent); RoBERTa-L 0.732 at Φ 2.0 (budget 2) | yes: 2 / 4 / ≈ 210 |
| Fixed pool N (E3) | kept, N 50 | F0: N\* 34–42 (AG News), 38 (Yahoo); p cancels | per dataset · bs only |
| Async C = 0.3·n, K = C/3 (S1) | kept | wall-clock gain only | no |
| Uniform ω (vs `grad_aware`) | kept | G1: within 0.6 pt at equal commits, +14% s/commit | no |
| bs 32 on seq-256 tasks | kept | P0b: same plateau as bs 8, 35–40% fewer vclock h | per task |
| Smaller LoRA rank | 1.7B r 2 (r 4 before) | same s: r 2 +3–6 pt by c30, same per-task cost (probe forwards dominate) | yes; RoBERTa-L r 6 untested lower |
| Server momentum 0.9 | big cells | 1.7B: best 0.432 (c28) vs 0.357 plain r 2 (c43); lead appears once the buffer fills (~c9) | yes: untested on DistilBERT at ρ0 0.07; weight norm grows faster |
| Saturation / GL stop | DistilBERT only | big cells stop on target | — |
| Cos ground-truth audit | off everywhere | stalls 1.7B at commit 1 (CPU backward) | no |

**Transfers to a new cell unchanged:** P 10 mean, trust ratio with `exp`, uniform ω, C / K ratios, N 50 (re-check N\* per dataset),
target stop ×2, cos audit off, `FWDLLM_FD_SCALE_INVARIANT=1`. **Set per cell:** s (→ ρ0 from p, P, N), Φ_budget (→ T_res), LoRA r,
momentum, bs, eval rows / stride, D divisor (D_CALIBRATION), bench workers per GPU. **FwdLLM arm per cell:** lr and var threshold
(var sets the pooled probe count, so match iterations per commit across models), cohort reselected per commit (`data_bin`).

## 3. Run queue (top = next)

Each item: action · exit · state. [B] adds no runs: it reads the [A] runs.

- **FT-N21 · big-model real↔sim transfer check · todo.** One 1–2 h real leg per big model beside a same-seed sim leg
  (FLUXTUNE FT-L19); a sim figure on RoBERTa-L or 1.7B cites it.
- **P1 · [A] `distilbert_agnews`: `fwdllm` + `fluxtune`, 1 seed · exit: `e2e_distilbert_agnews.pdf` from
  real runs · runs done 10-25 (FluxTune 0.86 at 4.34 vclock h; FwdLLM DNF, best 0.855; RUN_PLAN §3.6); manifest + figure todo.** These runs are also C1's A0 and A6.
- **P2 · C1 Phase A (A1–A4, 1 seed) · blocked: P1.** Its result picks the C3 groups worth running.
- **P3 · [A] `roberta_large_yelp` · `fluxtune` (run 4) running on jayne bench GPUs 4–7 since 10-26 01:50 (decision 15); `fwdllm` todo.** Stop: acc ≥ 0.82 on 2 evals or 30.5 h wall, scored on TTA.
- **P4 · [A] `smollm2_1p7b_yahoo` · `fluxtune` (run 6b, r 2 + momentum) running on jayne bench GPUs 0–3 since 10-26 01:50 (decision 15; run 6 stopped at c398, peak 0.576); `fwdllm` (run 5) ready (lr 3e-3, var 8.4).** Stop: acc ≥ 0.65 on 2 evals or 30.5 h wall, scored on TTA.
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
| E1b | probe selection by measured slope (studied variant) | — | max-\|JVP\| 1 of 10 vs mean (RUN_PLAN run 26, low priority) | low | selection raises aim and length together, so stability is unchanged (b²/a = 1); same compute and bytes as mean |
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
| | top-1 probe by \|JVP\| | max-\|JVP\| selection | `--probe-combine select` on the `fluxtune` path (RUN_PLAN run 26) |
| | mean | FluxTune | A6 |
| `step_size` | trust ratio, constant ρ | LARS/LAMB | `pool`: P0's run reused (N 90, ρ 0.067) |
| | trust ratio, ρ0·t^−a | LR decay | `--rho-schedule rm --rho0 0.07 --rho-exp {a}`, a solved so Φ = Φ\* at the commit cap (≈ 0.1 at 1,000) |
| | trust ratio, ρ0·exp(−t/2T_res) | FluxTune | A6 |
| | trust ratio, ρ0/Φ | 1/Φ alternative | `--rho-schedule phi` (needs code) |
| `agg_weights` | uniform (ω = 1) | FedAvg-style mean | A6 |
| | staleness ω only | Felix | `--agg-rate-type new` |
| | staleness ω × alignment gate | FluxTune | `grad_aware`: G1's run reused (N 90, ρ 0.067) |
| | alignment gate only (no staleness × utility base) | — | `grad_aware` + `agg_rate_conf.base: neutral` (yaml override; RUN_PLAN run 27, low priority) |

No plain-SGD arm: C1's A0 → A2 already compares FwdLLM's SGD with the trust ratio. No P sweep, variance-gate
sweep or iteration cap.

`step_size` also renders `c_step_size_step_size.pdf`: ρ and Φ per commit (needs `server_update_audit: true`).
If time allows: server momentum 0.5 (`--server-momentum 0.5`; 0.9 diverged on DistilBERT, is used on the big cells), server Adam/Yogi (*needs code*),
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
- **Speedup** = FwdLLM TTA / FluxTune TTA, in sim time only (wall time depends on simulator packing). If FwdLLM is
  DNF at the target: headline = time to FwdLLM's peak (FluxTune's first eval ≥ that peak); also "≥ X×" at the target
  (FwdLLM's whole run / FluxTune TTA) and the peak gain. Never extrapolate a DNF curve or quote a hand-picked
  threshold; same rule in every cell. Worked example: RUN_PLAN §3.6.
- **Curves are EMA-smoothed (0.6, `--smooth`), raw curve faint underneath.** Smoothing is visual only: TTA, peaks and
  markers use the raw evals.
- **Peak gaps under the 2.3-pt noise floor are not claims.** A DNF and a speedup are; say "plateaus at X" rather than
  "can't reach".
- **Bytes and commits to target** are counted up to that eval. A seed that never reaches target is DNF: its
  bytes and commits are whole-run, drawn faded and labelled DNF.
- **Seeds:** 1 per arm; same-seed replicate noise is 2.3 points (`parity_floors/fluxtune.yaml`),
  so a single-seed gap under that is unresolved. Bars show the seed mean with min–max. Curves show the arm's median-TTA seed, or its highest-peak seed if none reached.
- **Same stop rule in every arm of a cell:** DistilBERT: stall rule + GL decay + commit cap + wall cap; big cells: target ×2 or the
  30.5 h wall. Prior runs stopped for three different reasons and are not comparable.
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
- Max-|JVP| (E1b) lands between FwdLLM's pick and mean: better aim than one random probe (G_rule ≈ 3 vs 10), no stability gain.
- Server momentum shortens TTA: ag60 `Smom` 0.5 (0.35 vs 0.53 vclock h to 0.60, 1 seed, FT-T15); 1.7B smokes at 0.9 (§2.2).
  P3 / P4 test 0.9 at full n.

**Code gaps**
- [ ] `check_arm_health.py`: skip the `B_max origin` gate when `b_max_probe_every` is 0 (B_max is then a stale ln 2
  prior; A0's and `pool`'s FAILs are false alarms).
- [ ] Launcher flag for `stalenessPolicy` (today a yaml override).
- [ ] cos-sim weighted combine; normalised SGD; clipping; antithetic/orthogonal probes; server Adam/Yogi.
- [ ] Unavailability for the grad loop (FT-N5) for C4.

**Caveats**
- In fixed-pool mode the log line `[Variance=GOOD] var=0.79 <= thr=0.3` is misleading: the pool count, not
  the variance, triggered the commit.
- `learning_rate` is inert under trust ratio; the screened RoBERTa / 1.7B lr applies to the `fwdllm` arm only (§2 "FwdLLM arm").
- RoBERTa-L vclock hours rest on the pre-recipe `fluxtune_yelp-p.yaml` (Yahoo re-profiled 10-25); its charges are ≪ D, so the error is
  small (FLUXTUNE FT-T11). 1.7B runs before 10-24 used divisor 0.0192, so their vclock under-charges D 4.1×; their
  accuracy per commit stands.
- The big-model D assumes A40 cost ratios transfer to the phone NPU ([D_CALIBRATION §6](D_CALIBRATION.md#6-rules-and-caveats)); 1.7B's is likely a lower bound.

## 9. Legacy IDs cited in code

The docs that defined these are deleted; comments still cite them.

- **EXPTS_CHARTER A5:** α=1 for all FluxTune runs so far (§2 keeps α=1).
- **Issue I-1:** v1's peak-then-collapse at the lap boundary, caused by an undamped optimizer. Fixed by the
  trust-ratio step (P0 crossed 7 lap wraps; FLUXTUNE FT-N23 checks the big cells).
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
Decided 2026-10-25 (operator; supersedes 10-24 and horizon-aware T_res): big cells are set by s and Φ_budget, not ρ0. s 1.75 (between the
validated 1.5 and the 2 compute limit) → ρ0 0.0509 RoBERTa-L, 0.0436 1.7B; Φ_budget 4 → T_res 1,072 / 1,461 (§2.1). 10-25 07:40 (operator): 1.7B moves to s 2.0 → ρ0 0.0498, T_res 1,119. Paper run (RUN_PLAN run 6, [config](configs/paper_fluxtune_smollm2_yahoo.yaml)): n 60, C 30, K 10, stop at acc ≥ 0.65 on 2 evals
(in-process `--target-stop`) or 20 h wall, Φ rail 4 (never fires below Φ_budget 4), no commit cap, saturation off, then one full-test eval. Open: 12 h caps vs ~1,700–2,300 commits to Φ 3 (RUN_PLAN §6).
Decided 2026-10-26 (operator; RUN_PLAN decision 15): big-cell `fluxtune` on the bench, 4 GPUs each, momentum 0.9, stop only at target ×2. RoBERTa-L s 2.4, T_res 570 (Φ_budget 4);
1.7B LoRA r 2, ρ0 0.0978 (s 2.81), T_res 1,119 (from the §3.7 smokes). Supersedes 10-25's s 1.75 / 2.0 and 12 / 20 h caps (now 30.5 h wall).
