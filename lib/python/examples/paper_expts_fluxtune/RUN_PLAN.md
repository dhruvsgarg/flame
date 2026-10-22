# FluxTune paper: plots and run list

Every plot and every run the paper needs, derived from [EXPERIMENTS.md](EXPERIMENTS.md) (the source of truth for
the recipe, queue and scoring). Run times are from measured run speeds (§4); **(est.)** marks an extrapolation.
**Status:** `ready` = runnable now · `code` = needs code first · `decide` = optional, operator decides per step.

## 1. Plots

All rendered by `fwdllm/expt_scripts/make_paper_figs.py` into `paper_figs/<manifest>/`.

| # | File | Manifest | Question | Expected |
|---|---|---|---|---|
| 1–3 | `e2e_distilbert_agnews.pdf`, `e2e_roberta_large_yelp.pdf`, `e2e_smollm2_1p7b_yahoo.pdf` | `figs_e2e.yaml` | [A] Does FluxTune reach target faster and with fewer bytes than FwdLLM? (a) acc vs sim time + "N× faster"; (b) GB to target, down/up stacked + "N× less"; (c) acc vs commits | FluxTune wins TTA and GB in every cell; (c) shows higher accuracy per commit, not just more commits/h |
| 4 | `e2e_grid.pdf` | `figs_e2e.yaml` | plots 1–3 stacked | — |
| 5–7 | `b1_<cell>.pdf` | `figs_e2e.yaml` | B1: How long must a selected client stay engaged? (CDF) | FluxTune: minutes (one task); FwdLLM: much longer (whole round) |
| 8–10 | `b2_<cell>.pdf` | `figs_e2e.yaml` | B2: How widely is work spread? Share of clients contributed vs time; Lorenz + Gini | FluxTune covers the population sooner, lower Gini (guess; EXPERIMENTS states none) |
| 11–13 | `b3_<cell>.pdf` | `figs_e2e.yaml` | B3: How does training progress? Uploads per commit; cumulative commits vs time | FluxTune: steady pool of N, steady commit rate; FwdLLM: large, variable pools, few commits |
| 14 | `c_phase_a.pdf` | `figs_ablation.yaml` | C1: What does each concept add to FwdLLM alone? (every C plot: TTA bars, acc vs time, acc vs commits) | E1, E2, E3 gain per commit; S1 gains wall clock only. Small singles + big A6 jump ⇒ concepts interact |
| 15 | `c_phase_b.pdf` | `figs_ablation.yaml` | C2: Cumulative ladder (headline ablation) | Each rung lowers TTA; E1 + E3 less than the sum (both are variance reduction) |
| 16 | `c_agg_criteria.pdf` | `figs_ablation.yaml` | C3: When to commit? | Fixed pool ≥ variance gate; the var gate's iterations per bin vary widely (hitting time, FT-L4) |
| 17 | `c_update_utility.pdf` | `figs_ablation.yaml` | C3: How to combine probes? | Mean of P beats FwdLLM's aligned 1-of-10; gain flattens with P (variance ∝ 1/P); P=1 worst |
| 18 | `c_step_size.pdf` + `c_step_size_step_size.pdf` | `figs_ablation.yaml` | C3: Which server step rule? ρ and Φ per commit | Tuned SGD matches the trust ratio early, then drifts as ‖θ‖ grows; const ρ overshoots its peak; `exp` holds |
| 19 | `c_agg_weights.pdf` | `figs_ablation.yaml` | C3: Do aggregation weights matter? | Near-inert (≤ 0.6% weight change): the three arms fall within noise |
| 20–21 | `c_avail_syn_0.pdf`, `c_avail_mobiperf_3st_50.pdf` | `figs_availability.yaml` | C4: Are the gains just async? Does FluxTune hold up under real availability? | `syn_0`: FedBuff(P), Felix(P) fall short of FluxTune. `mobiperf_3st_50`: FluxTune holds vs the availability-aware Felix(P) |
| 22 | `c5_generalisation.pdf` | `figs_ablation.yaml` | C5: Do the winning concepts transfer to other data and a larger model? | Same ranking on Yahoo, Yelp-P and on 1.7B |

## 2. Cells, pool size and eval

**Common to every run:** sim mode, `syn_0` (except C4), Dirichlet α=1, **1 seed per arm** (3 later if time
allows), `server_update_audit: true`, the same stop rule (stall 0.003 + GL decay + ~1,000-commit cap + wall cap).
**FluxTune** = the [EXPERIMENTS.md §2](EXPERIMENTS.md#2-setup) recipe with `--pool-target {N}`. With 1 seed, a gap
under the 2.3-pt replicate noise is unresolved.

| Cell | p | Clients n | Pool N | ρ0 | Target | Wall cap | Eval sample (rows) | Stride |
|---|---|---|---|---|---|---|---|---|
| `distilbert_agnews` | 450,340 | 100 | 90 | 0.067 | 0.86 | 3 h | full test (7,600) | 4 |
| `roberta_large_yelp` | 591,874 | 100 | 100 | 0.062 | 0.82 | 12 h | open: sample size TBD | open: TBD |
| `smollm2_1p7b_yahoo` | 806,912 | 75 | 70 | 0.044 | 0.65 | 12 h | 6,000 (operator 10-20; fixed draw today) | 7 |

Every DistilBERT run uses N 90 (final): [A], C1–C5. C5 on Yelp-P / Yahoo uses that dataset's eval setting (open).

**Pool size N** = client gradients pooled into one server commit (`--pool-target`). It is not clients n (the
population). Gradients needed to reach Φ\* (`G* = 2·lnΦ*·p/(s²·P)`) have no N in them, so N is a systems choice,
and both caps favour a large N: the commit cap counts commits (larger N = more gradients per commit) and the
wall cap pays the per-commit overhead (fewer commits). Limits: step ρ ≤ 0.07 (top of the band); N ≤ n (distinct
clients per pool measured 0.77 at N = n, ≥ 0.7 required); N ≥ max(C, 2K) = 30 (staleness ≤ 1).

- **Rule:** `N = min(p·(0.07/s)²/P, n)`, rounded down to a multiple of K = 10, then `ρ0 = s·√(P·N/p)` (s = 1.5,
  P = 10). Rounding down keeps ρ ≤ 0.07 and N ≤ n, and makes I = N/K whole. DistilBERT: 98.1 → 90.
- s stays fixed, so ρ0 scales with p through the formula. This replaces the ρ0-scaling pair.
- N is decided (no N-check run). S checks distinct-clients/N ≥ 0.7 at RoBERTa's N = n and 1.7B's N.

**Eval.** Each eval scores a fresh random sample of the test set: a new draw each eval, seeded by commit index so
a rerun repeats it (*code*: today `eval_max_samples` is one fixed draw, `fwdllm_aggregator.py:3142`).
DistilBERT scores the full test set (7,600 rows) every 4th commit, so it needs no resampling. **Open:** sample size and stride for Yelp-P and Yahoo (below are the old 3,000–3,800-row, stride-4 estimates). ~3,000–3,800 rows per eval gives a standard error ≤ 0.9 pt at target, below the 2.3-pt replicate noise. A fresh
draw can cross the target by luck, so `tta_consecutive: 2`. Each run ends with one full-test eval (*code*), which
supplies the final accuracy in the text.

| Cell | t_eval (sample) | s/commit, no eval | Eval share at stride 4 | Evals per run |
|---|---|---|---|---|
| DistilBERT · AG News | ~7 s (full 7,600 rows; ~1 ms/row) | ~10 s | ~15% | ~250 |
| RoBERTa · Yelp-P | ~23 s at 3,800 rows (est., ~6 ms/row); open | ~88 s (est.) | ~7% at 3,800 rows; open | ~120; open |
| SmolLM2-1.7B · Yahoo | **220–294 s at 6,000 rows (measured 10-20, median 251 s; eval batch 8, 7 trainers share the aggregator's GPU)** | **~43 s at N 50 (measured)** | 0 at stride 7 (stride 5 held commits 0.6–16 s; stride 4 would slow commits ~22%) | ~1 per 5 min |

Stride 4 meets ⌈t_eval / s_commit⌉ + 1 at the DistilBERT and RoBERTa sizes; 1.7B · Yahoo needs ⌈220 / 43⌉ + 1 = 7. Every arm in
a cell uses the same stride and sample size. 6,000 fixed rows give ±1.2 pt (95%) at acc 0.65; the gap to the full set is a
constant offset.

## 3. Runs

### 3.1 Gating runs (no figure; they block the rest)

| ID | Model · Dataset | Arms / parameters | Runs | Time per run | Status | Feeds |
|---|---|---|---|---|---|---|
| G0 (FT-N1) | DistilBERT · AG News | paired v2 real legs with the §2 recipe → `sim_charge_profiles/fluxtune.yaml`, graded 0-fail (FT-T8: no v2 sim on the v1 profile) | ≥ 2 real | TBD | `code` (after the P0 code) | every sim run |
| P0 A/B | DistilBERT · AG News | FluxTune `exp` (T_res 300, ρ0 0.067, N 90) vs `--rho-schedule pool`; compare TTA at 0.80/0.86/0.87, peak, Φ at peak | 2 | `exp` 4 h (no commit cap: FT-N4 exit, holds past the lap boundary); `pool` ~1 h | `code` | the recipe; `pool` reused in C3 `step_size` |
| G1 grad-aware | DistilBERT · AG News | simplified recipe, same ρ schedule/N 90/seed: `--agg-rate-type grad_aware` vs `uniform`; compare TTA at 0.80/0.86/0.87, peak, ω spread (`agg_round.agg_weight`) | 2 | ~3 h | `code` (after P0 code) | decides grad-aware in the recipe or C3 control only; set `--agg-rate-type` in §2 |
| P0b bs | DistilBERT · Yahoo | bs 8 vs 32 at N 90 | 2 | bs 8 ~3.5 h; bs 32 ~6–10 h (est.) | blocked: P0, G1 | bs for Yelp-P and Yahoo (both under one pass) |
| S smoke | RoBERTa-large · Yelp-P (LoRA q,v r=6, seq 256); SmolLM2-1.7B · Yahoo (LoRA q,v r=4, seq 256) | FwdLLM + FluxTune at §2's N | 4 | 2 h cap | RoBERTa `code` (`classifier.dense` drop) | s/commit, t_eval, charge profile, distinct-clients/N for P3/P4 |

### 3.2 Figure runs

All DistilBERT arms run at N 90, 3 h cap, 1 seed.

| ID | Model · Dataset | Arms | Runs | Time per run | Status | Plots |
|---|---|---|---|---|---|---|
| **P1** | DistilBERT · AG News | `fwdllm`, FluxTune | 2 | 3 h | blocked: G0, P0, G1 | #1, 4, 5, 8, 11; also C1 A0/A6, C2 endpoints, C4 `syn_0` |
| **P3** | RoBERTa-large · Yelp-P | `fwdllm`, FluxTune; lr 3e-4 | 2 | 12 h (cap) | blocked: P0b, S | #2, 4, 6, 9, 12 |
| **P4** | SmolLM2-1.7B · Yahoo | `fwdllm`, FluxTune | 2 | 12 h (cap; ~720 commits, §4) | blocked: P0b, S | #3, 4, 7, 10, 13; C5 references |
| **P2 (C1)** | DistilBERT · AG News | A1 `--probe-combine mean --perturbation-count 10`; A2 `--server-step-rule trust_ratio --rho-schedule const --rho-star {0.03,0.05,0.07,0.10}`; A3 `--commit-gate fixed --pool-target 90 --var-stopping-policy off`; A4 `fedbuff_it_unaware` with FwdLLM's estimator | 7 | 3 h | blocked: P1 | #14; picks the C3 groups |
| P2 pairs | DistilBERT · AG News | A1+A2, A1+A3, A2+A3, only if singles gain little but A6 jumps | 3 | 3 h | `decide` | #14 |
| **P5 (C2)** | DistilBERT · AG News | FwdLLM → +E1 → +E1+E2 → +E1+E2+E3 (sync) → FluxTune; the two middle rungs are new | 2 | 3 h | blocked: P2 | #15 |
| **P6 `agg_criteria`** | DistilBERT · AG News | `--commit-gate var --var-threshold {0.1,0.3,1.0}`; iteration cap (`max_iterations_per_data_id`, value TBD); `--commit-gate n_target`; `--commit-gate fixed --pool-target {10,20,50}` (90 = A6) | 8 | 3 h | blocked: P2 | #16 |
| **P6 `update_utility`** | DistilBERT · AG News | A0 (reused); `--perturbation-count 1 --probe-combine mean`; P ∈ {2,5,20} mean (10 = A6) | 4 | 3 h | blocked: P2 | #17 |
| P6 `update_utility` + | DistilBERT · AG News | cos-sim weighted; top-1 probe | 2 | 3 h | `code` · `decide` | #17. Expected: between aligned-1 and mean-of-P |
| P6 `update_utility` + | DistilBERT · AG News | antithetic; orthogonal probes | 2 | 3 h | `code` · `decide` | #17. Expected: lower estimator variance at equal P, like raising P (guess) |
| **P6 `step_size`** | DistilBERT · AG News | `--server-step-rule raw_sgd --learning-rate {0.003,0.01,0.03,0.1}`; `--rho-schedule const`; `pool` (from P0); `rm`; `exp` (= A6) | 6 | 3 h | blocked: P2 | #18 |
| P6 `step_size` + | DistilBERT · AG News | `phi` (ρ0/Φ) | 1 | 3 h | `code` · `decide` | #18. Alternative decay; expected ≈ `exp` (guess) |
| P6 `step_size` + | DistilBERT · AG News | `--server-momentum 0.5` (0.9 diverged) | 1 | 3 h | `decide` | #18. Expected: small or no gain (guess) |
| P6 `step_size` + | DistilBERT · AG News | server Adam; Yogi; normalised SGD; clipping | 4 | 3 h | `code` · `decide` | #18. Expected: normalised SGD and clipping ≈ trust ratio (both bound the step); Adam/Yogi untested (guess) |
| **P6 `agg_weights`** | DistilBERT · AG News | `--agg-rate-type new`; `uniform` = G1's run, `grad_aware` = A6 | 1 | 3 h | blocked: P2 | #19 |
| **P7 (C4)** | DistilBERT · AG News | FedBuff(P) `fedbuff_round`, Felix(P) `felix_round` on `syn_0` + `mobiperf_3st_50`; FluxTune on `mobiperf_3st_50` (FwdLLM and FluxTune `syn_0` = P1) | 5 | 3 h | blocked: FT-N5 | #20–21 |
| **P8 (C5)** | DistilBERT · Yahoo, Yelp-P | 2–3 winning C1 concepts + FwdLLM + FluxTune, per dataset | 8–10 | ~3.5 h | blocked: P2 | #22 |
| **P8 (C5)** | SmolLM2-1.7B · Yahoo | FwdLLM + the largest C1 concept (P4 supplies both references) | 1 | 12 h | blocked: P2, P4 | #22 |

### 3.3 Run log: SmolLM2-1.7B · Yahoo

Config for all three (unless noted): FluxTune, sim, 50 trainers on 8 A40s (6–7 per GPU), shards 0–49 of
`niid_label_clients=100_alpha=1`, K 10 / C 30, LoRA q,v r=4 (p 806,912), seq 256, `--commit-gate fixed --pool-target 50
--rho-schedule pool`, delays `--delay-divisor 0.0192 --delay-floor 4.0`, `--send-timeout-s 3600`,
`--no-cos-ground-truth-audit`, eval 6,000 fixed rows. Deviates from the §2 cell (n 75, N 70, `exp`): an S-type run, not P4.
Charge profile is the DistilBERT-derived `fluxtune_yahoo.yaml` (`--allow-stale-profile`).

| Date | Run dir (`fwdllm/experiments/`) | Wall | Result |
|---|---|---|---|
| 10-19 | `run_20261019_020059` (r=8, catalog cos audit on) | 43 min | **void**: 0 commits. The cos audit's fp32 backward on 1,024 × 256 rows ran on the aggregator's CPU model; it hung 38 min, then the aggregator was SIGKILLed. The launcher still printed `COMPLETED crash=0`. Pass `--no-cos-ground-truth-audit` for any 1.7B run. |
| 10-20 01:26 | `run_20261020_012606` (screen, stride 5) | 15 min (cap) | **pass**: 50/50 trainers, 17 commits (~43 s each), 0 OOM/Traceback/overrun/abandon; ≤ 4.1 GB GPU and ≤ 2.3 GB host per trainer, aggregator 9.8 GB; acc 0.141 → 0.149 (commit 15); sim clock ~14× wall |
| 10-20 01:56 | `run_20261020_015616` (stride 7) | 1.23 h (stopped by hand, 03:10; planned 3 h) | **pass**: 99 commits, 43 s median (84/h), 0 errors / dead trainers / eval waits; acc 0.141 → **0.266** (commit 92), loss 2.84 → 2.08, rising every eval; Φ 1.071, Λ 0.092; sim clock 74,410 s (~17× wall) |

**3 h run (10-20 01:56), stopped by hand at 1.23 h once systems and trend were confirmed.** Launch:
`HOURS=3 fwdllm/expt_scripts/launch_smollm17b_yahoo_n50.sh` (3 h wall ceiling; sim-clock cap 432,000 s = 40× wall, so the
wall ceiling binds; `--phi-stop halt` can't fire in 3 h).

| Measure | Result |
|---|---|
| Pace | 99 commits in 69.7 min of commits; 42.7 s mean, 43.0 s median; 4.99 trips/commit (expected 5.0) |
| Eval | 14 evals, 6,000 rows, stride 7; 251 s median, 294 s max (gap ~300 s), 0 `eval still running` |
| Accuracy (commit: acc) | 1: 0.141 · 22: 0.159 · 43: 0.188 · 64: 0.228 · 78: 0.250 · 92: **0.266**; top-class share 0.33 → 0.15 |
| Memory | trainers ≤ 4,108 MiB GPU, host RSS 2.12 → 2.27 GB in the first 15 min then flat (105 GB total); aggregator 9.0 → 10.1 GB; GPUs 20.6–35.9 of 46 GB (≤ 78%); node 117 of 503 GB |
| Pool spread | distinct clients / contributions per pool: median 0.74, min 0.68 (S needs ≥ 0.7) |

Files: run dir `fwdllm/experiments/run_20261020_015616_fluxtune_yahoo_n50_smoke_syn_0_sim/` (489 MB; logs, telemetry,
configs); in `fwdllm/`: launcher `smollm17b_yahoo_n50_20261020_0156.log`, watchdog `…_watch.log` (`watch_arm.py`),
per-eval accuracy `…_acc.log` (`track_acc.py`); scripts `fwdllm/expt_scripts/launch_smollm17b_yahoo_n50.sh`,
`track_acc.py`. Screen: run dir `run_20261020_012606_…`, launcher `fwdllm/smollm17b_yahoo_n50_screen_20261020_0125.log`.

**Reading.** Progress is Λ = 2·lnΦ/s at fixed ρ 0.0373. DistilBERT · Yahoo read 0.657 at Λ ≈ 1.0 (`125003`, dir deleted);
this run stopped at Λ 0.092. Λ 1.0 needs ~1,080 commits (~12.9 h at 84/h); Φ\* = 3 ~1,580 (~19 h). No Yahoo curve at low
Λ survives, so this run confirms systems and trend, not the accuracy level.

## 4. Run-time basis

Under `exp`, Φ is bounded by exp(ρ0²·T_res/2) (≈ 1.96 for DistilBERT), so the Φ rail never fires and a cap ends every run: the wall cap on all three
cells (DistilBERT ~1,000 commits in 3 h at ~10 s). FwdLLM wall time is unmeasured; under the shared stop rule it
runs to the wall cap, so it gets FluxTune's time.

The measured runs below evaluated the full test set every 2 commits, so their s/commit include that eval cost. The
360M runs serve only as scaling inputs; no paper run uses 360M.

| Run | Model · data | Pool N | Wall | Commits | s/commit |
|---|---|---|---|---|---|
| `run_20261020_015616` | SmolLM2-1.7B · Yahoo, seq 256, eval 6,000 rows / stride 7 | 50 | 1.23 h (stopped by hand) | 99 | 43 |
| `run_20261020_012606` | SmolLM2-1.7B · Yahoo, seq 256, eval 6,000 rows / stride 5 | 50 | 0.25 h (wall cap) | 17 | ~43 |
| `run_20261014_034646` | SmolLM2-1.7B · AG News, seq 192 | 50 | 8.0 h (wall cap) | 399 | 72 |
| `run_20261012_032648` | SmolLM2-360M · AG News | 50 | 6.0 h (wall cap) | 651 | 33 |
| `run_20261011_223407` | SmolLM2-360M · AG News | 10 | 4.4 h | 351 | 45 |
| `run_20261002_000757` (dir deleted) | DistilBERT · AG News, simplified | 98 | 1.74 h (Φ=3) | 881 | 7 |
| `run_20261001_120715` (dir deleted) | DistilBERT · AG News, full controller | 98 | 3.0 h (wall cap) | 855 | 13 |
| buildplan §4.5 smokes | DistilBERT · Yahoo / Yelp-P, eval capped 10k (12.5 s/eval) | — | — | — | 13 / 12 |

Scaling is linear in N, plus ×1.33 for seq 192 → 256:

- **SmolLM2-1.7B · Yahoo, N 70:** measured 43 s at N 50 (10-20) × 1.4 ≈ 60 s/commit → ~720 commits in 12 h. The old
  134 s estimate scaled the AG News run, whose s/commit included a full eval every 2 commits.
- **RoBERTa-large · Yelp-P, N 100:** 360M's 33 s × 2 × 1.33 ≈ 88 s/commit (est.) → ~490 commits in 12 h. S replaces
  this estimate.

## 5. Compute (1 seed; one run = one node)

| Group | Runs | Node-hours |
|---|---|---|
| Gating (G0, P0, G1, P0b bs, S) | ~12 | ~31 + G0 |
| [A]/[B] (P1, P3, P4) | 6 | ~54 |
| C1 + C2 | 9 | ~27 |
| C3 committed | 19 | ~57 |
| C4 | 5 | ~15 |
| C5 | 9–11 | ~30–35 DistilBERT + 12 for 1.7B |
| **Committed total** | ~61 | ~232 |
| `decide` arms (P2 pairs, C3 extras incl. `phi`) | 13 | ~39 |

3 seeds later ≈ 3× the figure-run rows.

## 6. Open (closed by data, not by decision)

- RoBERTa s/commit and t_eval, and both charge profiles: S records them, then the §2 and §4 estimates are replaced.
  1.7B · Yahoo s/commit and t_eval are measured (§3.3).
- N is decided; S confirms distinct-clients/N ≥ 0.7 at RoBERTa's N = n.
- C3 `agg_criteria` iteration cap value: set from A0's measured iterations per bin.
- Code before the first figure run: `--rho-schedule exp` smoke, stall-rule fix, commit cap, per-eval resampled eval,
  final full-test eval, `classifier.dense` drop.
