# FluxTune paper: plots and run list

Every plot and every run the paper needs, derived from [EXPERIMENTS.md](EXPERIMENTS.md) (the source of truth for
the recipe, queue and scoring). Run times are from measured run speeds (§3); **(est.)** marks an extrapolation.

## 1. Plots

All rendered by `fwdllm/expt_scripts/make_paper_figs.py` into `paper_figs/<manifest>/`.

| # | File | Manifest | Shows |
|---|---|---|---|
| 1 | `e2e_distilbert_agnews.pdf` | `figs_e2e.yaml` | [A] (a) accuracy vs sim time + "N× faster"; (b) GB to target, down/up stacked + "N× less"; (c) accuracy vs commits |
| 2 | `e2e_roberta_large_yelp.pdf` | `figs_e2e.yaml` | [A] same three panels |
| 3 | `e2e_smollm2_1p7b_yahoo.pdf` | `figs_e2e.yaml` | [A] same three panels |
| 4 | `e2e_grid.pdf` | `figs_e2e.yaml` | plots 1–3 stacked |
| 5–7 | `b1_<cell>.pdf` × 3 cells | `figs_e2e.yaml` | B1: CDF of client engagement time |
| 8–10 | `b2_<cell>.pdf` × 3 cells | `figs_e2e.yaml` | B2: share of clients contributed vs time; Lorenz curve + Gini |
| 11–13 | `b3_<cell>.pdf` × 3 cells | `figs_e2e.yaml` | B3: uploads per commit; cumulative commits vs time |
| 14 | `c_phase_a.pdf` | `figs_ablation.yaml` | C1: FwdLLM + one concept (every C plot: TTA bars, accuracy vs time, accuracy vs commits) |
| 15 | `c_phase_b.pdf` | `figs_ablation.yaml` | C2: cumulative ladder (headline ablation) |
| 16 | `c_agg_criteria.pdf` | `figs_ablation.yaml` | C3: when to commit |
| 17 | `c_update_utility.pdf` | `figs_ablation.yaml` | C3: how probes are combined |
| 18 | `c_step_size.pdf` + `c_step_size_step_size.pdf` | `figs_ablation.yaml` | C3: server step rule; ρ and Φ per commit |
| 19 | `c_agg_weights.pdf` | `figs_ablation.yaml` | C3: aggregation weights |
| 20–22 | `c_avail_syn_0.pdf`, `c_avail_syn_50_3st.pdf`, `c_avail_mobiperf_3st.pdf` | `figs_availability.yaml` | C4: sync vs async, and availability |
| 23 | not named yet | none | C5: generalisation |

## 2. Runs

**Common to every run:** sim mode, `syn_0` (except C4), Dirichlet α=1, ≥ 3 seeds per arm, `server_update_audit:
true`, the same stop rule (stall 0.003 + GL decay + ~1,000-commit cap + wall cap), eval at a higher stride than the measured runs (§3). **FluxTune** = the
[EXPERIMENTS.md §2](EXPERIMENTS.md#2-setup) recipe with `--pool-target {N}`.

### 2.1 Gating runs (no figure; they block the rest)

| ID | Model · Dataset | Arms / parameters | Runs | Time per run | Feeds |
|---|---|---|---|---|---|
| P0 A/B | DistilBERT · AG News | FluxTune `phi` (ρ0 0.07, N 98) vs same with `--rho-schedule pool`; cap ~1,000 commits | 2 × 2 seeds = 4 | `phi` ~2–3.5 h; `pool` ~1 h (Φ=3 at ~450 commits) | the recipe; `pool` arm reusable in C3 `step_size` (#18) |
| P0 ρ0 scaling | SmolLM2-360M · AG News (dataset assumed) | one pair: hold N, ρ0 ∝ 1/√p; or hold ρ0, N ∝ p | 2 | ~9 h | ρ0 for RoBERTa and SmolLM2-1.7B |
| P0b pool size | SmolLM2-360M · AG News (assumed) | `--pool-target` 50 vs 70, at matched gradients | 2 | ~9 h each | the N-selection rule |
| P0b batch size | Yahoo (model unstated; DistilBERT assumed) | bs 8 vs 32 at fixed N | 2 | bs 8 ~3.5 h; bs 32 ~6–10 h (est.) | bs for Yelp-P and Yahoo |
| S smoke | RoBERTa-large · Yelp-P (LoRA q,v r=6, p 591,874, seq 256); SmolLM2-1.7B · Yahoo (LoRA q,v r=4, p 806,912, seq 256) | FwdLLM + FluxTune, 1 seed each | 4 | 2 h cap | s/commit for P3/P4. RoBERTa needs the `classifier.dense` drop first |

### 2.2 Figure runs

| ID | Model · Dataset | Algorithm / arms | Key parameters | Runs | Time per run | Plots |
|---|---|---|---|---|---|---|
| **P1** | DistilBERT (adapters rf=16, p 450,340) · AG News (seq 192) | `fwdllm`, FluxTune | 100 clients, target 0.80, N from P0b | 6 | ~2–3.5 h (3 h wall cap binds) | #1, 4, b1–b3 DistilBERT; also C1 A0/A6, C2 endpoints, C4 `syn_0` FwdLLM + FluxTune |
| **P3** | RoBERTa-large · Yelp-P | `fwdllm`, FluxTune | clients TBD, target 0.80, lr 3e-4 | 6 | ~13 h (est.; S sets it) | #2, 4, b1–b3 RoBERTa |
| **P4** | SmolLM2-1.7B · Yahoo | `fwdllm`, FluxTune | clients TBD, target 0.65 | 6 | ~33 h to the commit cap | #3, 4, b1–b3 SmolLM2 |
| **P2 (C1)** | DistilBERT · AG News | A1 `--probe-combine mean --perturbation-count 10`; A2 `--server-step-rule trust_ratio --rho-schedule const --rho-star {grid}`; A3 `--commit-gate fixed --pool-target 50 --var-stopping-policy off`; A4 `fedbuff_it_unaware` with FwdLLM's estimator | rest as FwdLLM | 12 + ρ\* grid for A2 | ~3 h (wall cap) | #14; picks the C3 groups |
| **P5 (C2)** | DistilBERT · AG News | FwdLLM → +E1 → +E1+E2 → +E1+E2+E3 (sync) → FluxTune; only the two middle rungs are new | order may change after C1 | 6 new | ~3 h | #15 |
| **P6 `agg_criteria`** | DistilBERT · AG News | `--commit-gate var --var-threshold {0.1,0.3,1.0}`; iteration cap; `--commit-gate n_target`; `--commit-gate fixed --pool-target {10,20,50,100}` (50 = A6) | | ~27 | ~2–3 h | #16 |
| **P6 `update_utility`** | DistilBERT · AG News | A0 (reused); `--perturbation-count 1 --probe-combine mean`; P ∈ {2,5,20} mean (10 = A6); cos-sim and top-1 need code | | 12 now, +6 after code | ~2–3 h | #17 |
| **P6 `step_size`** | DistilBERT · AG News | `--server-step-rule raw_sgd --learning-rate {grid}`; `--rho-schedule const`; `pool` (from P0); `rm`; `phi` (= A6) | tune SGD η | ~15 | ~2–3 h | #18 |
| **P6 `agg_weights`** | DistilBERT · AG News | `--agg-rate-type uniform`, `new` (`grad_aware` = A6) | | 6 | ~2–3 h | #19 |
| **P7 (C4)** | DistilBERT · AG News | FedBuff(P) `fedbuff_round` and Felix(P) `felix_round` on 3 traces; FluxTune on `syn_50_3st`, `mobiperf_3st` | traces `syn_0`, `syn_50_3st`, `mobiperf_3st` | 24 | ~3 h (wall cap) | #20–22; blocked on FT-N5 |
| **P8 (C5)** | DistilBERT · Yahoo, Yelp-P | 2–3 winning C1 concepts + FwdLLM and FluxTune references | | ~25–30 | ~3.5 h | #23 |
| **P8 (C5)** | SmolLM2-1.7B · Yahoo | FwdLLM vs FluxTune with the largest concept | | 3–6 | ~33 h | #23 |

## 3. Run-time basis

Under `phi`, Φ=3 takes 8/ρ0² commits (> 1,000 in every cell), so the commit cap ends a run: **time ≈ 1,000 ×
s/commit**, unless the stall rule or GL decay fires first. FwdLLM wall time is unmeasured; under the shared stop
rule it runs to the wall cap, so it gets FluxTune's time.

**Eval stride.** The measured runs below evaluated every 2 commits (`eval_every_n_commits: 2`, the yaml default).
Paper runs use a higher stride (`--eval-every-n-commits`), at least ⌈t_eval / s per commit⌉ + 1 per cell
([cheatsheet §8](../fwdllm/fl_fwd_ft_pipeline_cheatsheet.md#8-choosing-the-pool-size-n-per-model-draft-not-final)
step 5). The s/commit below include the stride-2 eval cost, so the run times here are upper bounds. A higher
stride also coarsens TTA to one eval per stride, so every arm in a comparison uses the same stride.

| Run | Model · data | Pool N | Wall | Commits | s/commit |
|---|---|---|---|---|---|
| `run_20261014_034646` | SmolLM2-1.7B · AG News, seq 192 | 50 | 8.0 h (wall cap) | 399 | 72 |
| `run_20261012_032648` | SmolLM2-360M · AG News | 50 | 6.0 h (wall cap) | 651 | 33 |
| `run_20261011_223407` | SmolLM2-360M · AG News | 10 | 4.4 h | 351 | 45 |
| `run_20261002_000757` (dir deleted) | DistilBERT · AG News, simplified | 98 | 1.74 h (Φ=3) | 881 | 7 |
| `run_20261001_120715` (dir deleted) | DistilBERT · AG News, full controller | 98 | 3.0 h (wall cap) | 855 | 13 |
| buildplan §4.5 smokes | DistilBERT · Yahoo / Yelp-P, eval capped 10k | — | — | — | 13 / 12 |

Scaling to planned cells uses the [cheatsheet §8](../fwdllm/fl_fwd_ft_pipeline_cheatsheet.md#8-choosing-the-pool-size-n-per-model-draft-not-final)
cost model (t_commit 9.2 s, t_iter 5.1 s per K=10 gradients) and ×1.33 for seq 192 → 256:

- **SmolLM2-1.7B · Yahoo:** 72 s × (N 70/50 ≈ 1.3) × 1.33 ≈ 120 s/commit → ~33 h for 1,000 commits. A 10 h wall cap
  buys ~300 commits (Φ ≈ 1.3).
- **RoBERTa-large · Yelp-P:** no run yet; 360M's 33 s × N 60/50 × 1.33 ≈ 50 s/commit × ~960 commits ≈ 13 h.

## 4. Compute (3 seeds)

| Group | Runs | Node-hours |
|---|---|---|
| Gating (P0, P0b, S) | ~14 | ~65 |
| [A]/[B] (P1, P3, P4) | 18 | ~295 (P4 ≈ 200, full 8-GPU node) |
| C1 + C2 | 18 | ~55 |
| C3, all groups | ~60 | ~150–180 |
| C4 | 24 | ~70 |
| C5 | ~30 + 3–6 | ~95 DistilBERT + 100–200 for 1.7B |

## 5. Open

- P4: accept ~33 h/run, or cap wall at ~10–12 h and score TTA to 0.65 only (1.7B reached 0.832 on AG News in
  8 h). Lowering N does not help: total gradients to Φ\* have no N in them.
- RoBERTa time stays an estimate until S records its s/commit.
- Eval stride value per cell (S records t_eval and s/commit to set it).
- Final targets per cell, client counts for RoBERTa and 1.7B, whether C4 adds `syn_20_3st`, and C5's output file.
- Dataset for the 360M gating pairs and model for the Yahoo bs A/B are not stated in EXPERIMENTS.md.
