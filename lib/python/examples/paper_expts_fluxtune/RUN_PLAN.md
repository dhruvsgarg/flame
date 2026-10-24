# FluxTune paper: plots and run list

Every plot and every run the paper needs, derived from [EXPERIMENTS.md](EXPERIMENTS.md) (the source of truth for
the recipe, queue and scoring). Run times are from measured run speeds (§4); **(est.)** marks an extrapolation.
**Status:** `ready` = runnable now · `code` = needs code first · `decide` = optional, operator decides per step.

**Rules (every agent, every run):**
1. When a run starts or finishes, update its status on the board below, with its node, config file, and log file paths.
2. When a plot is made, add its path (and the script that made it) next to the run(s) it shows.

## Status board (keep current: update on every launch, finish and decision)

Last updated 2026-10-24 (sim vs real checks, §2 sim rules; FT-N18 B2 launch; 10-22 operator review of every setting). Paths relative to `lib/python/examples/fwdllm/`; run dirs under `experiments/`.

**Running**

| ID | Node · GPUs · tmux | Started → ETA | Run dir · logs | Check |
|---|---|---|---|---|
| S smoke RoBERTa-L · Yelp-P | jayne | 10-18 (operator) | — | — |
| rb60: RoBERTa-L · Yelp-P to 0.60, `fwdllm` (lr 0.01) → FluxTune (§2 recipe, T_res 615, max-commits 380, bs 32) | jayne · GPUs 0–7 · `dg_flame:rb60` | 10-23 16:06 → each arm ≤ 12 h | script `fwdllm/expt_scripts/nodes/run_node_roberta_yelp_to60_queue.sh`; queue log `fwdllm/rb60_queue_20261023_1606.log`; arm → run dir `fwdllm/rb60_20261023_160630/runs.tsv`; figures `fwdllm/rb60_20261023_160630/paper_figs/` (script `fwdllm/expt_scripts/plot_roberta_yelp60.py`) | n 100, drops to 75 on OOM; D est. (fluxtune 0.0169, fwdllm 0.169) |

**Done**

| ID | Outcome | Detail |
|---|---|---|
| D divisor per model (10-24, shepherd, 16:15 → 16:27) | measured A40 cost per task, one trainer alone per GPU: RoBERTa-L 0.0171 (0.0169 in use, kept), **SmolLM2-1.7B 0.0047 (0.0192 was 4.1× too large)**, DistilBERT 0.478 unchanged; table in [EXPERIMENTS §2](EXPERIMENTS.md#2-setup). `launch_smollm17b_yahoo_n50.sh` still passes 0.0192 | `fwdllm/ft_profile_20261024/` on shepherd (`run.sh`, 4 run dirs); FT-L18, FT-T13, FT-T14 |
| Sim vs real, big models (10-24, shepherd, 14:30 → 15:00) | `fluxtune` n 16 / C 8 / K 5, 10 min per leg, §2 recipe: RoBERTa-L sim 5 commits, 5.3 vclock h (33×), acc 0.53 → 0.57; SmolLM2-1.7B sim 2 commits, 2.2 vclock h (14×); both real legs 0 iterations (D 4–19 min). 0 errors. Not paper runs | `fwdllm/ft_models_check_20261024/` on shepherd (`run.sh`, 4 run dirs; `first_try/` = the 8-per-GPU SmolLM OOM); §4 table; FT-L11, L12, L17 |
| Sim vs real, DistilBERT n 50 (10-24, shepherd, 13:33 → 14:06) | `fluxtune` n 50 / C 20 / K 10, 30 min per leg: real 103 commits, acc 0.33–0.34; sim 345 commits, 6,202 vclock s (3.4×), acc 0.54 and rising, no collapse past the lap boundary; matched at c100. Not paper runs | `fwdllm/ft_n50_check_20261024/` on shepherd (`run.sh`, both run dirs); FT-L13 |
| FT-N18 B2 (10-24, jayne, 02:58 → 03:58) | fwdllm bench gate PASS on DistilBERT · AG News n 100 `syn_0`: `fwdllm` (3 bench vs 3 harness seeds, every same-seed pair identical) and `fluxtune` (3 vs A6, AUC 44.6 / 43.4). Not paper runs | `fwdllm/bench_B2_20261024/` (`b2.json`, `bench/`, `harness/`, `run_harness_seeds.sh`); harness run dirs `experiments/run_20261024_033641_b2h_s123[56]_*`; FLUXTUNE FT-D3 |
| ag60 preview (10-22, jayne, 2 lanes, 07:24 → 10:34) | 22 of 23 DistilBERT · AG News arms reached 0.60; A3 (FwdLLM + fixed pool) diverged, stopped by hand. TTA (vclock h): FluxTune 0.51 vs FwdLLM 2.27 (4.5×); A1 0.31, A12 0.67, A123 1.95, A2 5.9; N 10/20/50/90 0.23/0.37/0.51/0.71. Not scored (1 seed, target 0.60, window 1) | script `fwdllm/expt_scripts/nodes/run_node_agnews_to60_queue.sh`; arm → run dir `fwdllm/ag60_20261022_{072444,073816,074438}/runs.tsv`; figures `fwdllm/ag60_20261022_074438/paper_figs/` (script `fwdllm/expt_scripts/plot_ag60.py`); fixed on the way: mean-of-10 on FwdLLM (A1), `exp` without pool (A2), converge watcher after a lap wrap. Figure manifests' `target_acc` still 0.60 (restore 0.80) |
| F0 Λ readout | no run: Λ_need from P0 `exp`, G1, P0b → N\* 34–42 (AG News), 38 (Yahoo bs 32) → N 50 every cell | §3.1 |
| 1.7B · Yahoo dry run | learns: 99 commits, 43 s/commit at N 50 bs 8, acc 0.141 → 0.266 (`pool`, stopped by hand 1.23 h); not an S leg | §3.3 |
| P0b bs 32 | **bs 32 is the recipe** (decided 10-19). Stopped 10-19 02:00 at 717 commits; same plateau as bs 8 (0.669, max 0.671), 35–40% fewer vclock h and commits to 0.5/0.6/0.65. Real wall is reversed (18.6 vs 7.0 s/commit) | §0c |
| P0b bs 8 | stopped 22:15 at plateau: 0.65 at ~1:05 wall, 0.669 by 2:10 (max 0.671, 1178 commits) | §0c |
| P0 A/B | `exp`, T_res 300 is the recipe; `pool` → C3 arm | §0 |
| G1 grad-aware | `uniform` is the recipe; `grad_aware` → C3 arm (no gain at equal commits, +14% s/commit) | §0b |
| FwdLLM A0 (preliminary) | peak 0.851; stopped on vclock budget at 43 min, not a matched P1 arm | §0 |
| FT-N4 (4 h stability) | closed without a run: P0 `exp` crossed 7 lap wraps, no collapse (DistilBERT only; check P3/P4 at their first wrap) | §0 |
| Stall-rule fix | leaky streak, `SAT_STALL_LEAK = 1` | [EXPERIMENTS.md](EXPERIMENTS.md) §3 P0 |

**Remaining gating runs** (§3.1)

| ID | Needs | State |
|---|---|---|
| G0 v2 charge profile (FT-N1) | ≥ 2 real legs with the §2 recipe; `--agg-rate-type uniform` | todo (last gating run on shepherd); blocks every sim figure run |
| S smoke SmolLM2-1.7B · Yahoo | — | todo: both arms at the §2 recipe (`exp`, ρ0 0.14, T_res 71, s 5.62, bs 32, n 75); measures bs 32 s/commit for F2 and the eval stride, and whether n 75 fits (OOM → n 60). Launch: `ARM=fluxtune HOURS=2 fwdllm/expt_scripts/launch_smollm17b_yahoo_n50.sh`, then `ARM=fwdllm` |
| S smoke RoBERTa-L · Yelp-P | — | running (jayne); its s/commit feeds the Λ compute check (F2) |
| `fwdllm` lr screen | RoBERTa-L · Yelp-P, 1.7B · Yahoo; lr {1e-4, 3e-4, 1e-3} | todo after S: 6 × ~1.5 h; best lr feeds P3/P4's `fwdllm` arm (DistilBERT keeps 0.01) |
| R1/R2 ρ_max | shepherd, 2 × 2 h | todo; decides whether DistilBERT's ρ0 rises above 0.07 (big cells run 0.14, operator 10-24) |
| N50 check | shepherd, 2 h | todo; confirms N 50 at ρ0 0.07 against P0 (N 90) |
| F1 Yelp-P N check | shepherd, ~4 h | optional; P3 reaching target also confirms N 50 on Yelp-P |

**Stop and eval flags (in code 10-22):** `--max-commits N` (halts, `stop_reason` `commit_cap`; scales the stop-rule warm-up
450 → 0.45·N, so set it near the commits a 12 h run reaches: ~380 on RoBERTa and 1.7B, est.); `--eval-resample` (fresh
`--eval-max-samples` draw per eval, seeded by commit); `--final-full-eval` (one full-test eval at run end, event `agg_eval_final`).

**Operator decisions (closed 10-22)**
1. P1 `fwdllm` runs to the 3 h wall cap (`--max-runtime-s` ≥ 40 × wall cap), not equal vclock.
2. `B_max origin` gate applies only when BmaxProbe is on; A0 and `pool` are not invalidated (code fix: EXPERIMENTS §8).
3. DistilBERT eval stride 4.
4. No iteration-cap arm: C3 `agg_criteria` is trimmed to `n_target` + N {10, 20, 90}.
5. 12 h cells (RoBERTa, 1.7B): bs 32 if F2 shows reachable Λ ≥ Λ_need, else bs 8 (TTA is vclock, but a DNF scores nothing).
6. DistilBERT figure runs at N 50 if the N50 check passes, else N 90.
7. Horizon-aware T_res on both 12 h cells.
8. (10-24) ρ0 0.14 on RoBERTa-L and 1.7B `fluxtune`, T_res 71 (Φ\* 2); DistilBERT stays 0.07.

Closed by data, not by decision: grad-aware (G1), bs (P0b → 32, subject to F2), Yelp-P/Yahoo eval stride (S), v1 profile (G0).

**Then the figure runs:** P1 → P2 → P5, P6; P3/P4 after S and the lr screens; P7 after FT-N5; P8 deferred.

## 0. P0 + A0 queue (done 2026-10-18)

**Script:** [run_node_p0_queue.sh](../fwdllm/expt_scripts/nodes/run_node_p0_queue.sh), node **shepherd**, GPUs 0–3,
tmux `dg_flame:p0q`, 04:41 → 08:42. Queue log: `fwdllm/p0_queue_20261018_0441.log` (per-arm gate verdicts at each
`--- [p0q]` line). Paths below are relative to `lib/python/examples/fwdllm/`; run dirs sit under `experiments/`.
Each run dir holds `*_aggregator.log` (evals: `results after eval are`), `*_trainers.log`, `plots/`,
`telemetry/` (aggregator + 100 trainer jsonl; 5.1 / 2.4 / 1.3 GB).

| Arm | Run dir | Launcher dir (`expt_scripts/smoke_logs/`; config `*.yaml`, log `*.out`) | Ended | Commits | s/commit | Peak acc | 0.86 / 0.87 at commit | Φ end |
|---|---|---|---|---|---|---|---|---|
| P0 `exp` | `run_20261018_044202_fluxtune_agnews_n100_smoke_syn_0_sim` | `20261018_044200/fluxtune_agnews_n100_smoke_syn_0_sim.yaml` | 2 h wall cap | 1071 | 6.7 | **0.878** (~c904) | 344 / 560 | 1.93 |
| P0 `pool` | `run_20261018_064840_fluxtune_agnews_n100_smoke_syn_0_sim` | `20261018_064838/fluxtune_agnews_n100_smoke_syn_0_sim.yaml` | Φ=3 rail (`phi_fixed halt`), 65 min | 490 | 6.7 | 0.872 (~c364) | 268 / 364 | 3.00 |
| FwdLLM A0 | `run_20261018_075342_fwdllm_agnews_n100_smoke_syn_0_sim` | `20261018_075340/fwdllm_agnews_n100_smoke_syn_0_sim.yaml` | `max_runtime_s` 86,400 s vclock, 43 min wall | 251 | 9.3 | 0.851 (~c224) | DNF / DNF | — |

Superseded attempts (P0 `exp` config, no result): `run_20261018_030629` (0 evals), `run_20261018_030938` (6 evals),
`run_20261018_032613` (261 commits, ended 04:40 when the queue took over); launcher dirs `20261018_030627`, `_030936`, `_032611`.

Commit numbers for accuracy = eval index × 4 (stride 4). Plot: `fwdllm/p0_exp_vs_pool_acc_phi.png`
(script `fwdllm/scripts/plot_p0_exp_vs_pool.py`; Φ computed from each schedule, matches the logged endpoints).

**P0 decision (operator 10-18): `exp`, T_res 300, is the FluxTune recipe; `pool` becomes a C3 `step_size` arm.**
`pool` led by ≤ 1 pt over c150–490 (inside the 2.3-pt noise), then plateaued at ~0.867 from c340 (no real dip
before the rail). `exp` overtook at ~c560 and climbed to 0.876. RUN_PLAN rule: gap < 2.3 pts throughout → keep `exp`.
`exp` check closed: ρ_last 0.01127 = 0.0671·exp(−1070/600); Φ 1.927 = Π√(1+ρ²); every gate held.

**Caveats for later scoring:**
- `pool` and A0 flagged `B_max origin` FAIL (B > B_max on 182 / 164 commits): `pool` runs Φ to 3 by design; A0 is
  a FwdLLM control. Read §6 before reusing either as a scored arm.
- A0 stopped on its 86,400 s vclock budget, not the 3 h wall cap, so it is not yet P1's matched `fwdllm` arm.
- Fwdllm runs on this branch need the uncommitted trainer fix in `FedSgdTrainer.py:241-249` (launcher sends
  `avl_events_syn_{20,50}_3st`; without it every trainer crashes at startup).
- Queue mechanics (reusable): an arm starts only when GPUs 0–3 are free; `FLAME_SLOT_GPUS` pins the aggregator
  to GPU 3; `watch_arm.py` kills an arm after 20 min without a commit or 10 trainer deaths.

## 0b. G1 grad-aware (done 2026-10-18)

**Script:** [run_node_g1_queue.sh](../fwdllm/expt_scripts/nodes/run_node_g1_queue.sh) (P0 `exp` config with
`--agg-rate-type grad_aware`, 2 h cap), node **shepherd**, GPUs 0–3, tmux `dg_flame:g1q`, 17:09 → 19:16.
Queue log: `fwdllm/g1_queue_20261018_1709.log` (verdict: every readable gate holds). The `uniform` arm is P0 `exp` (§0).

| Arm | Run dir | Launcher dir (`expt_scripts/smoke_logs/`; config `*.yaml`, log `*.out`) | Commits | s/commit | Peak acc | 0.80 / 0.86 / 0.87 at commit (wall) | Φ end | ω max/min |
|---|---|---|---|---|---|---|---|---|
| `uniform` (P0 `exp`) | `run_20261018_044202_fluxtune_agnews_n100_smoke_syn_0_sim` | `20261018_044200/fluxtune_agnews_n100_smoke_syn_0_sim.yaml` | 1071 | 6.5 | **0.878** (c905) | c153 / c343 / c561 (17 / 39 / 62 min) | 1.927 | 1 |
| `grad_aware` | `run_20261018_170949_fluxtune_agnews_n100_smoke_syn_0_sim` | `20261018_170946/fluxtune_agnews_n100_smoke_syn_0_sim.yaml` | 941 | 7.4 | 0.874 (c782) | c146 / c342 / c558 (19 / 44 / 70 min) | 1.907 | median 1.105, max 1.29 |

Plot: `fwdllm/g1_uniform_vs_grad_aware.png` (acc vs commits, acc vs wall, ω spread; script
`fwdllm/scripts/plot_g1_uniform_vs_grad_aware.py`). Commits and wall come from aggregator telemetry (`server_update`, `agg_eval`).

**G1 result: `uniform` is the recipe; `grad_aware` becomes a C3 `agg_weights` arm.** At equal commits the arms stay
within 0.6 pts (inside the 2.3-pt noise). ω spread grows 1.02 → 1.14 over the run without an accuracy gain.
`grad_aware` runs 0.9 s/commit slower (gating cost or node load, not separated), so it is 2–8 min behind at each target.

## 0c. P0b batch size (done 2026-10-19)

**Script:** [run_node_p0b_queue.sh](../fwdllm/expt_scripts/nodes/run_node_p0b_queue.sh) (§2 recipe, `uniform`, Yahoo,
eval 3,000 rows stride 4; caps bs 8 3.5 h, bs 32 6 h; `--train-batch-size`), node **shepherd**, GPUs 0–3,
tmux `dg_flame:p0bq`, 10-18 19:55 → 10-19 02:00 (bs 32 stopped by operator). Queue log: `fwdllm/p0b_queue_20261018_1955.log`.

| Arm | Run dir | Launcher dir (`expt_scripts/smoke_logs/`; config `*.yaml`, log `*.out`) | Commits | s/commit (real) | Peak acc | 0.50 / 0.60 / 0.65 at commit (vclock h) | Φ end |
|---|---|---|---|---|---|---|---|
| bs 8 | `run_20261018_195537_fluxtune_yahoo_n100_smoke_syn_0_sim` | `20261018_195534/fluxtune_yahoo_n100_smoke_syn_0_sim.yaml` | 1178 | 7.0 | 0.671 | c250 / c350 / c510 (2.9 / 4.5 / 7.45) | 1.92 |
| bs 32 | `run_20261018_221609_fluxtune_yahoo_n100_smoke_syn_0_sim` | `20261018_221606/fluxtune_yahoo_n100_smoke_syn_0_sim.yaml` | 717 | 18.6 | 0.671 | c185 / c241 / c365 (2.1 / 2.8 / 4.9) | 1.83 |

Plot: `fwdllm/p0b_bs8_vs_bs32.png` (acc vs vclock, acc vs commits; script `fwdllm/scripts/plot_p0b_bs8_vs_bs32.py`).
vclock = telemetry `vclock_now`; it runs past `--sim-wall-ceiling-h` (bs 8: 19.5 h vs 3.5), so check its unit before quoting hours.

**P0b decision (operator 10-19): bs 32 for Yelp-P and Yahoo.** Same plateau (0.669); bs 32 gets there in fewer commits
and vclock h. On real wall bs 8 is faster (0.65 at 1.0 h vs 1.84 h).

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
| 16 | `c_agg_criteria.pdf` | `figs_ablation.yaml` | C3: How sensitive is FluxTune to N, and does an adaptive pool help? | Flat for N ≥ N\* (50, 90); N 10/20 noisier per step; `n_target` ≈ fixed N |
| 17 | `c_update_utility.pdf` | `figs_ablation.yaml` | C3: How to combine 10 probes? | Mean beats FwdLLM's aligned 1-of-10; cos-sim and top-1 fall between |
| 18 | `c_step_size.pdf` + `c_step_size_step_size.pdf` | `figs_ablation.yaml` | C3: Which ρ schedule? ρ and Φ per commit | const ρ (`pool`) plateaus and hits the Φ rail; budget-matched `rm` vs `exp` isolates the decay shape |
| 19 | `c_agg_weights.pdf` | `figs_ablation.yaml` | C3: Do aggregation weights matter? | Near-inert (≤ 0.6% weight change): the three arms fall within noise |
| 20–21 | `c_avail_syn_0.pdf`, `c_avail_mobiperf_3st_50.pdf` | `figs_availability.yaml` | C4: Are the gains just async? Does FluxTune hold up under real availability? | `syn_0`: FedBuff(P), Felix(P) fall short of FluxTune. `mobiperf_3st_50`: FluxTune holds vs the availability-aware Felix(P) |
| 22 | `c5_generalisation.pdf` | `figs_ablation.yaml` | C5 (deferred): Do the winning concepts transfer to other data and a larger model? | Same ranking on Yahoo, Yelp-P and on 1.7B |

## 2. Cells, pool size and eval

**Common to every run:** sim mode, `syn_0` (except C4), Dirichlet α=1, **1 seed per arm**,
`server_update_audit: true`, cos audit off, `tta_consecutive: 2`, the same stop rule (stall 0.003 + GL decay + commit cap + wall cap),
async C = 0.3·n, K = C/3.
**FluxTune** = the [EXPERIMENTS.md §2](EXPERIMENTS.md#2-setup) recipe with `--pool-target {N}`. With 1 seed, a gap
under the 2.3-pt replicate noise is unresolved.

| Cell | p | Clients n | Pool N | ρ0 (`--rho0`) | T_res | implied s | Target | Wall cap | Eval sample (rows) | Stride |
|---|---|---|---|---|---|---|---|---|---|---|
| `distilbert_agnews` | 450,340 | 100 | 50 if N50 check passes, else 90 | 0.07 | 283 | 2.10 (1.57 at N 90) | 0.86 | 3 h | full test (7,600) | 4 |
| `roberta_large_yelp` | 591,874 | 100 | 50 | **0.14** | 71 (Φ\* 2) | 4.82 | 0.82 | 12 h | 10% = 3,800, resampled per eval | set from S (4, est.) |
| `smollm2_1p7b_yahoo` | 806,912 | 75 (60 if OOM) | 50 | **0.14** | 71 (Φ\* 2) | 5.62 | 0.65 | 12 h | 10% = 6,000, resampled per eval | set from S's bs 32 s/commit (4, est.) |

C = 0.3·n, K = C/3: 30 / 10 at n 100, 23 / 8 at n 75, 18 / 6 at n 60. Big-cell bs: 32 unless F2 fails (then 8).

**Per-cell launch flags** (on top of the [EXPERIMENTS §2](EXPERIMENTS.md#2-setup) recipe; `{}` = still open):

```
# distilbert_agnews (bs 8, full test set, so no --eval-max-samples)
--dataset agnews --num-trainers 100 --pool-target {50 | 90} --rho0 0.07 --t-res 283 --train-batch-size 8
--max-commits 1000 --final-full-eval --eval-every-n-commits 4 --sim-wall-ceiling-h 3
# roberta_large_yelp (fwdllm arm: --learning-rate {lr screen})
--dataset yelp-p --model-type roberta-large --peft-method lora --lora-r 6 --num-trainers 100
--pool-target 50 --rho0 0.14 --t-res 71 --train-batch-size {32 | 8 per F2}
--max-commits {from S, ~370} --eval-max-samples 3800 --eval-resample --final-full-eval
--eval-every-n-commits {stride} --sim-wall-ceiling-h 12 --delay-divisor 0.0171   # fwdllm arm: 0.171
# smollm2_1p7b_yahoo (all of this is fwdllm/expt_scripts/launch_smollm17b_yahoo_n50.sh: ARM, HOURS, MAX_COMMITS, STRIDE env)
--dataset yahoo --model-type llama --model-name HuggingFaceTB/SmolLM2-1.7B --peft-method lora --lora-r 4 --num-trainers {75 | 60}
--pool-target 50 --rho0 0.14 --t-res 71 --train-batch-size {32 | 8 per F2}
--max-commits {from S, ~380} --eval-max-samples 6000 --eval-resample --final-full-eval
--eval-every-n-commits {stride, 4 est.} --sim-wall-ceiling-h 12 --delay-divisor 0.0047   # fwdllm arm: 0.047
# fwdllm arm on 1.7B: --learning-rate {lr screen}
# every cell: --no-cos-ground-truth-audit
# every cell: --max-runtime-s >= 40 × wall cap in s, so the wall ceiling, not the vclock, binds
```

How to set the new flags:
- **`--max-commits`** = ⌊wall cap / S's measured s/commit⌋, so the cap and the wall cap land together. It also sets the
  stall/GL warm-up (0.45 × cap) and horizon (0.15 × cap); at the default 450 they never arm on RoBERTa or 1.7B.
  DistilBERT: 1,000 (reached at ~1.9 h). Same value in both arms of a cell.
- **`--eval-max-samples` + `--eval-resample`**: only where the full test set is too slow to score per eval (Yelp-P, Yahoo).
  Pair with `tta_consecutive: 2` in the figure manifest. Stride = ⌈t_eval / s_commit⌉ + 1.
- **`--final-full-eval`**: every figure run; its `agg_eval_final` accuracy is the number quoted in the text. It runs after the
  loop exits: ~30 s on DistilBERT, ~40 min on 1.7B · Yahoo (60,000 rows, est.). Give external watchdogs that much margin
  (`launch_smollm17b_yahoo_n50.sh` kills at `HOURS` + 0.5 h).

**Sim and real legs (every run; why: [FLUXTUNE_READINESS](../_metadata/FLUXTUNE_READINESS.md) FT-L10–L17, FT-T10–T12):**
- **What sim is.** Real compute, modeled time: each task's delay D (`training_delay_s / --delay-divisor`) is charged to the
  vclock, not slept. TTA and every figure's x axis are vclock. Speed vs real in §4.
- **Calibration.** vclock = D per task (`--delay-divisor`, measured per model: [EXPERIMENTS §2](EXPERIMENTS.md#2-setup)) +
  aggregator charges (`sim_charge_profiles/fluxtune_<dataset>.yaml`, < 1 s per op). Yelp-P and Yahoo profiles predate the
  §2 recipe (`--allow-stale-profile`); G0 re-profiles them from a full-budget real run, never a smoke.
- **Caps.** Sim: `--max-runtime-s` ≥ 40 × wall cap plus `--sim-wall-ceiling-{s,h}`; the ceiling ends the leg. Real:
  `--max-runtime-s` is wall from aggregator start, join included (~3.5 min at n 100).
- **Real legs are slow by design.** At recipe D a RoBERTa / 1.7B iteration takes 4–19 min and the first commit ~1 h; a
  check of accuracy needs ≥ 2 h. Compare real and sim at matched commits, never matched wall.
- **Side by side.** Real and sim legs in parallel run as lanes (own GPUs, cores, mosquitto, `FLAME_RUN_LABEL`), as in
  `fwdllm/ft_models_check_20261024/run.sh`. 1.7B at bs 32: ≤ 4 trainers per GPU on a lane that holds the aggregator.
- **After a real smoke,** move its run dir out of `experiments/`, or the profile-staleness preflight blocks the next
  `fluxtune` sim.
- **Bench** (`fl_bench.py` / `bench_pool.py`) is sim only and stands for the harness only on cells that passed B2
  (FLUXTUNE FT-N18): DistilBERT · AG News `fwdllm`, `fluxtune`.

**Pool size N** = client gradients pooled into one server commit (`--pool-target`), not clients n. Rule and derivation:
[cheatsheet §8](../fwdllm/fl_fwd_ft_pipeline_cheatsheet.md#8-choosing-n-ρ0-and-t_res).

- **N 50, every cell.** `N ≥ N* ≈ 29·Λ_need²·(ρ0/0.07)²`, set per dataset · bs; p cancels across models. Measured N\*:
  AG News 34–42, Yahoo bs 32 38 (F0). Client availability sets C (clients in flight), with C ≤ N.
- **ρ0 0.07 on DistilBERT, 0.14 on RoBERTa-L and 1.7B (operator 10-24); T_res = 2·lnΦ\*/ρ0² (Φ\* 2): 283 at 0.07, 71 at 0.14.**
  At 0.14 the 12 h cells' T (~380) ≫ T_res, so horizon-aware T_res ≈ 71 too. 0.07 is the largest step that has run cleanly;
  R1/R2 test 0.10 and 0.14 on DistilBERT. Pass it as `--rho0` (overrides `--gate-safety-s`; the aggregator logs the implied s = ρ0·√(p/(P·N))).
- **First-hours abort rule (RoBERTa-L, 1.7B):** Φ off (1+ρ²)^(t/2) by > 1%, no accuracy above chance by ~c50, or GL decay
  before Φ ≈ 1.5 → halve ρ0 to 0.07 (`--rho0 0.07`), T_res 283; N stays.

**Eval.** Each eval scores a fresh random sample of the test set: a new draw each eval, seeded by commit index so
a rerun repeats it (`--eval-resample`).
DistilBERT scores the full test set (7,600 rows) every 4th commit, so it needs no resampling. Yelp-P and Yahoo score 10% of
the test set (3,800 / 6,000 rows), which gives the same noise at each target (≈ ±1.2 pt, 95%), below the 2.3-pt replicate noise;
stride is set from S. `tta_consecutive: 2` in every cell. Each run ends with one full-test eval (`--final-full-eval`), which
supplies the final accuracy in the text.

| Cell | t_eval (sample) | s/commit, no eval | Eval share at stride 4 | Evals per run |
|---|---|---|---|---|
| DistilBERT · AG News | ~26 s measured on P0, aggregator pinned (full 7,600 rows) | — | evals back to back: 6.7 s/commit incl. eval (P0) | ~270 in 2 h (P0 `exp`) |
| RoBERTa · Yelp-P | ~23 s at 3,800 rows (est., ~6 ms/row) | ~117 s at bs 32 (est., §4) | ~5% at 3,800 rows (est.) | ~90 (est.) |
| SmolLM2-1.7B · Yahoo | **220–294 s at 6,000 rows (measured 10-20, median 251 s; eval batch 8, 7 trainers share the aggregator's GPU)** | **~43 s at N 50, bs 8 (measured)**; ~114 s at bs 32 (est., §4) | 0 at stride 7, bs 8 (stride 5 held commits 0.6–16 s); stride 4 at bs 32 (est.) | ~1 per 5 min |

Stride = ⌈t_eval / s_commit⌉ + 1: 4 for RoBERTa; 7 for 1.7B at bs 8, 4 at bs 32 (est.); DistilBERT (P0: 26 s eval,
6.7 s/commit) runs evals back to back at stride 4; the rule gives 5 (open). Every arm in a cell uses the same stride and
sample size. 6,000 rows give ±1.2 pt (95%) at acc 0.65.

## 3. Runs

### 3.1 Gating runs (no figure; they block the rest)

| ID | Model · Dataset | Arms / parameters | Runs | Time per run | Status | Feeds |
|---|---|---|---|---|---|---|
| G0 (FT-N1) | DistilBERT · AG News | paired v2 real legs with the §2 recipe (`uniform`) → `sim_charge_profiles/fluxtune.yaml`, graded 0-fail (FT-T8: no v2 sim on the v1 profile) | ≥ 2 real | TBD | todo | every sim run |
| P0 A/B | DistilBERT · AG News | FluxTune `exp` (T_res 300, ρ0 0.067, N 90) vs `--rho-schedule pool`; compare TTA at 0.80/0.86/0.87, peak, Φ at peak, at equal commits. Config: [p0_exp_distilbert_agnews.yaml](configs/p0_exp_distilbert_agnews.yaml) (`pool` = same with `--rho-schedule pool`); both arms: stall rule and GL off, cos audit off, uniform ω, v1 charge profile, 2 h wall cap | 2 | 2 h wall each (FT-N4 closed without a 4 h run) | **done 10-18: `exp` chosen** (§0) | the recipe; `pool` reused in C3 `step_size` |
| G1 grad-aware | DistilBERT · AG News | P0 `exp` config with `--agg-rate-type grad_aware` vs `uniform`; TTA at 0.80/0.86/0.87, peak, ω spread | 1 new (`uniform` = P0 `exp`) | 2 h | **done 10-18: `uniform` chosen** (§0b) | the recipe; `grad_aware` reused in C3 `agg_weights` |
| P0b bs | DistilBERT · Yahoo | bs 8 vs 32 at N 90 | 2 | bs 8 2.3 h (plateau); bs 32 3.7 h (operator stop) | **done 10-19: bs 32** (§0c) | bs for Yelp-P and Yahoo |
| S smoke | RoBERTa-large · Yelp-P (LoRA q,v r=6, seq 256); SmolLM2-1.7B · Yahoo (LoRA q,v r=4, seq 256) | FwdLLM + FluxTune at §2's N | 4 | 2 h cap | running (jayne, operator) | s/commit, t_eval, charge profile, distinct-clients/N for P3/P4 |
| F0 Λ readout | DistilBERT · AG News, Yahoo | `expt_scripts/lambda_need.py` on P0 `exp`, G1, P0 `pool`, P0b bs 8/32: Λ at target and at peak − 0.5 pt | 0 | — | **done 10-21** ([cheatsheet §8](../fwdllm/fl_fwd_ft_pipeline_cheatsheet.md#8-choosing-n-ρ0-and-t_res)) | N\* 34–42 (AG News), 38 (Yahoo bs 32), 53 (Yahoo bs 8) → N 50; accuracy vs Λ schedule-independent |
| R1/R2 ρ_max | DistilBERT · AG News, bs 8 | P0 `exp` config (N 90, same seed) at `--rho0 0.10` (T_res 139) and `--rho0 0.14` (T_res 71); overlay accuracy vs Λ on P0 at matched Λ (not peak: Λ_∞ shrinks as 1/ρ0) | 2 | 2 h cap | todo | ρ_max = largest ρ0 that overlaps P0 within ±0.005. Raise it only if R1 overlaps; at 0.10, N\* ≈ 80 |
| N50 check | DistilBERT · AG News, bs 8 | §2 recipe, N 50, `--rho0 0.07`, T_res 283 vs P0 `exp` (N 90): TTA 0.86/0.87, peak, samples read | 1 | 2 h cap | todo | N 50 adopted if peak within 2.3 pt of P0 |
| F1 Yelp-P N check | DistilBERT · Yelp-P, bs 32, seq 256 | §2 recipe as P0b (N 90, `--rho0 0.067`, T_res 300), eval 3,000 rows stride 4; run to plateau; `lambda_need.py <run> --targets 0.80 0.82` | 1 | ~4 h cap | optional | N\* for Yelp-P; N 50 holds if Λ_need ≤ 1.32. Reusable as P8's DistilBERT · Yelp-P FluxTune arm if eval settings match |
| F2 compute check | RoBERTa-L · Yelp-P; SmolLM2-1.7B · Yahoo | no run: S s/commit scaled to N 50 → T_wall = 12 h ÷ s/commit → Λ reachable = Λ_∞·(1 − e^{−T_wall/566}), Λ_∞ = 4·ln2·√(P·N/p)/0.07, vs Λ_need × √(450,340/p) | 0 | — | after S | bs 32 if reachable Λ ≥ need, else bs 8 (decision 5). Est.: 1.7B Λ ≈ 0.48 vs ≈ 0.62; RoBERTa Λ ≈ 0.55 |
| F3 transfer checks | RoBERTa-L · Yelp-P; SmolLM2-1.7B · Yahoo | no extra run: (a) S's first `trainable_weight_norm` vs 13.35·√(p/450,340) = 15.3 / 17.9; (b) P3/P4 `lambda_need.py`: Λ at milestones vs DistilBERT × √(450,340/p) | 0 | — | (a) after S, (b) after P3/P4 | if either fails, N picks up a model factor; raise N for C5 and the 3-seed reruns |

### 3.2 Figure runs

All DistilBERT arms run at the N the N50 check picks (50, else 90), 3 h cap, 1 seed.

| ID | Model · Dataset | Arms | Runs | Time per run | Status | Plots |
|---|---|---|---|---|---|---|
| **P1** | DistilBERT · AG News | `fwdllm`, FluxTune | 2 | 3 h cap (FluxTune hits the ~1,000-commit cap at ~1.9 h) | blocked: G0 | #1, 4, 5, 8, 11; also C1 A0/A6, C2 endpoints, C4 `syn_0` |
| **P3** | RoBERTa-large · Yelp-P | `fwdllm` (screened lr), FluxTune | 2 | 12 h (cap) | blocked: S, F2, lr screen | #2, 4, 6, 9, 12 |
| **P4** | SmolLM2-1.7B · Yahoo | `fwdllm` (screened lr), FluxTune | 2 | 12 h (cap; ~380 commits at N 50, bs 32 (est., §4)) | blocked: S, F2, lr screen | #3, 4, 7, 10, 13 |
| **P2 (C1)** | DistilBERT · AG News | A1 `--probe-combine mean --perturbation-count 10`; A2 `--server-step-rule trust_ratio --rho-schedule exp --rho0 0.07 --t-res 283`; A3 `--commit-gate fixed --pool-target {N} --var-stopping-policy off`; A4 `fedbuff_it_unaware` with FwdLLM's estimator | 4 | 3 h | blocked: P1 | #14; picks the C3 groups |
| P2 pairs | DistilBERT · AG News | A1+A2, A1+A3, A2+A3, only if singles gain little but A6 jumps | 3 | 3 h | `decide` | #14 |
| **P5 (C2)** | DistilBERT · AG News | FwdLLM → +E1 → +E1+E2 → +E1+E2+E3 (sync) → FluxTune; the two middle rungs are new | 2 | 3 h | blocked: P2 | #15 |
| **P6 `agg_criteria`** | DistilBERT · AG News | `--commit-gate n_target`; `--commit-gate fixed --pool-target {10,20,90}` (50 = A6; if N stays 90, 50 replaces 90) | 4 | 3 h | blocked: P2 | #16 |
| **P6 `update_utility`** | DistilBERT · AG News | A0, A6 (reused); cos-sim weighted; top-1 probe; all P=10 | 2 | 3 h | `code`; blocked: P2 | #17. Expected: between aligned-1 and mean |
| P6 `update_utility` + | DistilBERT · AG News | antithetic; orthogonal probes | 2 | 3 h | `code` · `decide` | #17. Expected: lower estimator variance at equal P, like raising P (guess) |
| **P6 `step_size`** | DistilBERT · AG News | `pool` (P0's run reused); `--rho-schedule rm --rho0 0.07 --rho-exp {a}`, a solved so Φ = Φ\* at the commit cap (≈ 0.1 at 1,000); `exp` (= A6). No SGD arm | 1 | 3 h | blocked: P2 | #18 |
| P6 `step_size` + | DistilBERT · AG News | `phi` (ρ0/Φ) | 1 | 3 h | `code` · `decide` | #18. Alternative decay; expected ≈ `exp` (guess) |
| P6 `step_size` + | DistilBERT · AG News | `--server-momentum 0.5` (0.9 diverged) | 1 | 3 h | `decide` | #18. Expected: small or no gain (guess) |
| P6 `step_size` + | DistilBERT · AG News | server Adam; Yogi; normalised SGD; clipping | 4 | 3 h | `code` · `decide` | #18. Expected: normalised SGD and clipping ≈ trust ratio (both bound the step); Adam/Yogi untested (guess) |
| **P6 `agg_weights`** | DistilBERT · AG News | `--agg-rate-type new`; `uniform` = A6, `grad_aware` = G1's run (N 90, ρ 0.067) | 1 | 3 h | blocked: P2 | #19 |
| **P7 (C4)** | DistilBERT · AG News | FedBuff(P) `fedbuff_round`, Felix(P) `felix_round` on `syn_0` + `mobiperf_3st_50`; FluxTune on `mobiperf_3st_50` (FwdLLM and FluxTune `syn_0` = P1) | 5 | 3 h | blocked: FT-N5 | #20–21 |
| P8 (C5) | DistilBERT · Yahoo, Yelp-P | 2–3 winning C1 concepts + FwdLLM + FluxTune, per dataset | 8–10 | ~3.5 h | deferred | #22 |
| P8 (C5) | SmolLM2-1.7B · Yahoo | FwdLLM + the largest C1 concept (P4 supplies both references) | 1 | 12 h | deferred | #22 |

### 3.3 Run log: SmolLM2-1.7B · Yahoo

Config for all three (unless noted): FluxTune, sim, 50 trainers on 8 A40s (6–7 per GPU), shards 0–49 of
`niid_label_clients=100_alpha=1`, K 10 / C 30, LoRA q,v r=4 (p 806,912), seq 256, `--commit-gate fixed --pool-target 50
--rho-schedule pool`, delays `--delay-divisor 0.0192 --delay-floor 4.0`, `--send-timeout-s 3600`,
`--no-cos-ground-truth-audit`, eval 6,000 fixed rows. Dry run to check that 1.7B learns (operator 10-22): not an S leg, not P4; deviates from the §2 cell (n 75, `exp`, ρ0 0.07, bs 32).
Charge profile is the DistilBERT-derived `fluxtune_yahoo.yaml` (`--allow-stale-profile`).

| Date | Run dir (`fwdllm/experiments/`) | Wall | Result |
|---|---|---|---|
| 10-19 | `run_20261019_020059` (r=8, catalog cos audit on) | 43 min | **void**: 0 commits. The cos audit's fp32 backward on 1,024 × 256 rows ran on the aggregator's CPU model; it hung 38 min, then the aggregator was SIGKILLed. The launcher still printed `COMPLETED crash=0` (fixed 10-22: a non-zero aggregator exit now reads `AGG_KILLED`). Pass `--no-cos-ground-truth-audit` for any 1.7B run. |
| 10-20 01:26 | `run_20261020_012606` (screen, stride 5) | 15 min (cap) | **pass**: 50/50 trainers, 17 commits (~43 s each), 0 OOM/Traceback/overrun/abandon; ≤ 4.1 GB GPU and ≤ 2.3 GB host per trainer, aggregator 9.8 GB; acc 0.141 → 0.149 (commit 15); sim clock ~14× wall |
| 10-20 01:56 | `run_20261020_015616` (stride 7) | 1.23 h (stopped by hand, 03:10; planned 3 h) | **pass**: 99 commits, 43 s median (84/h), 0 errors / dead trainers / eval waits; acc 0.141 → **0.266** (commit 92), loss 2.84 → 2.08, rising every eval; Φ 1.071, Λ 0.092; sim clock 74,410 s (~17× wall) |

**3 h run (10-20 01:56), stopped by hand at 1.23 h once systems and trend were confirmed.** Launch:
`HOURS=3 fwdllm/expt_scripts/launch_smollm17b_yahoo_n50.sh` as of 10-20 (since moved to the §2 recipe; 3 h wall ceiling; sim-clock cap 432,000 s = 40× wall, so the
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

Under `exp`, Φ is bounded by exp(ρ0²·T_res/2) (≈ 1.96 for DistilBERT), so the Φ rail never fires and a cap ends every run:
the ~1,000-commit cap on DistilBERT (6.7 s/commit → ~1.9 h), the 12 h wall cap on RoBERTa and 1.7B. FwdLLM ran 9.3 s/commit
(A0, 251 commits); under the shared stop rule it runs to the wall cap.

Pre-P0 runs evaluated the full test set every 2 commits; P0, G1 and P0b used stride 4. Each s/commit includes its eval cost.
The 360M runs serve only as scaling inputs; no paper run uses 360M.

| Run | Model · data | Pool N | Wall | Commits | s/commit |
|---|---|---|---|---|---|
| `run_20261018_044202` (P0 `exp`, shepherd) | DistilBERT · AG News, 4 GPUs, eval stride 4 | 90 | 2.0 h (wall cap) | 1071 | 6.7 |
| `run_20261018_064840` (P0 `pool`, shepherd) | DistilBERT · AG News, 4 GPUs, eval stride 4 | 90 | 1.08 h (Φ=3) | 490 | 6.7 |
| `run_20261018_075342` (FwdLLM A0, shepherd) | DistilBERT · AG News, 4 GPUs, eval stride 4 | — | 0.81 h (vclock budget) | 251 | 9.3 |
| `run_20261018_170949` (G1 `grad_aware`, shepherd) | DistilBERT · AG News, 4 GPUs, eval stride 4 | 90 | 2.0 h (wall cap) | 941 | 7.4 |
| `run_20261018_195537` (P0b bs 8, shepherd) | DistilBERT · Yahoo, seq 256, eval 3,000 rows stride 4 | 90 | 2.3 h (plateau) | 1178 | 7.0 |
| `run_20261018_221609` (P0b bs 32, shepherd) | DistilBERT · Yahoo, seq 256, eval 3,000 rows stride 4 | 90 | 3.7 h (operator stop) | 717 | 18.6 |
| `run_20261020_015616` | SmolLM2-1.7B · Yahoo, seq 256, eval 6,000 rows / stride 7 | 50 | 1.23 h (stopped by hand) | 99 | 43 |
| `run_20261020_012606` | SmolLM2-1.7B · Yahoo, seq 256, eval 6,000 rows / stride 5 | 50 | 0.25 h (wall cap) | 17 | ~43 |
| `run_20261014_034646` | SmolLM2-1.7B · AG News, seq 192 | 50 | 8.0 h (wall cap) | 399 | 72 |
| `run_20261012_032648` | SmolLM2-360M · AG News | 50 | 6.0 h (wall cap) | 651 | 33 |
| `run_20261011_223407` | SmolLM2-360M · AG News | 10 | 4.4 h | 351 | 45 |
| `run_20261002_000757` (dir deleted) | DistilBERT · AG News, simplified | 50 | 1.74 h (Φ=3) | 881 | 7 |
| `run_20261001_120715` (dir deleted) | DistilBERT · AG News, full controller | `n_target` (mean 49.5) | 3.0 h (wall cap) | 855 | 13 |
| buildplan §4.5 smokes | DistilBERT · Yahoo / Yelp-P, eval capped 10k (12.5 s/eval) | — | — | — | 13 / 12 |

Scaling is linear in N, ×1.33 for seq 192 → 256, and ×2.66 for bs 8 → 32 (P0b: 18.6 / 7.0 s/commit):

- **SmolLM2-1.7B · Yahoo, N 50, bs 32:** measured 43 s at N 50, bs 8 (10-20, §3.3) × 2.66 ≈ 114 s/commit (est.) →
  ~380 commits in 12 h. The old 255 s estimate scaled the AG News run, whose s/commit included a full eval every 2 commits.
- **RoBERTa-large · Yelp-P, N 50, bs 32:** 360M's 33 s × 1.33 × 2.66 ≈ 117 s/commit (est.) → ~370 commits in 12 h.
- Both sit below a 1,000-commit cap's 450-commit warm-up; `--max-commits ~380` scales it to ~170 so stall and GL arm. S replaces these

**Sim speed vs real** (10-24, shepherd, `fluxtune`, real and sim legs side by side on the same node; Done rows above).
Gain = vclock / wall; it tracks D / GPU time per task (FT-L11). These legs ran the old divisors: at SmolLM's measured 0.0047,
D and its gain are 4.1× larger (~57×).

| Cell (check shape) | Wall per leg | Real: commits | Sim: commits · vclock | Gain | D per task · GPU per task |
|---|---|---|---|---|---|
| DistilBERT · AG News (n 50, C 20, K 10) | 30 min | 103 | 345 · 1.72 h | 3.4× | — (smoke-YAML delays) |
| RoBERTa-L · Yelp-P (n 16, C 8, K 5, §2 recipe) | 10 min | 0 (0 iterations) | 5 · 5.3 h | 33× | 237–1,124 s (mean 536) · 9 s |
| SmolLM2-1.7B · Yahoo (n 16, C 8, K 5, §2 recipe) | 10 min | 0 (0 iterations) | 2 · 2.2 h | 14× | 208–990 s (mean 483) · 21 s |

## 5. Compute (1 seed; one run = one node)

| Group | Runs | Node-hours |
|---|---|---|
| Gating (G0, P0, G1, P0b bs, S, R1/R2, N50, `fwdllm` lr screens) | ~21 | ~46 + G0 |
| [A]/[B] (P1, P3, P4) | 6 | ~54 |
| C1 + C2 | 6 | ~18 |
| C3 committed (2 need code) | 8 | ~24 |
| C4 | 5 | ~15 |
| **Committed total** | ~46 | ~157 + G0 |
| `decide` arms (P2 pairs, C3 extras incl. `phi`) | 11 | ~33 |
| C5 (deferred) | 9–11 | ~30–35 DistilBERT + 12 for 1.7B |

## 6. Open (closed by data, not by decision)

- RoBERTa s/commit and t_eval, and both charge profiles: S records them, then the §2 and §4 estimates are replaced.
  1.7B · Yahoo s/commit and t_eval are measured at bs 8 (§3.3); bs 32 is scaled.
- N 50 and ρ_max 0.07: closed by the N50 check and R1/R2 (§3.1).
- N 50 at ρ0 0.14 (big cells): the §2 rule `N* ≈ 29·Λ_need²·(ρ0/0.07)²` gives 4× the 0.07 N\* (Yahoo bs 32: 38 → ~152 > 50).
  Closed by S's first-hours abort rule and the cells' Λ readout, or by raising N.
- 12 h-cell bs (F2), 1.7B n (75 or 60), `fwdllm` lr: closed by S and the lr screens.
