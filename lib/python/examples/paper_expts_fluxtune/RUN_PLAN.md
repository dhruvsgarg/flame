# FluxTune paper: plots and run list

Every plot and every run the paper needs, derived from [EXPERIMENTS.md](EXPERIMENTS.md) (the source of truth for
the recipe, queue and scoring; task cost D per model: [D_CALIBRATION.md](D_CALIBRATION.md)). Run times are from measured run speeds (§4); **(est.)** marks an extrapolation.
**Status:** `ready` = runnable now · `code` = needs code first · `decide` = optional, operator decides per step.

**Rules (every agent, every run):**
1. When a run starts or finishes, update its status on the board below, with its node, config file, and log file paths.
2. When a plot is made, add its path (and the script that made it) next to the run(s) it shows.

## Status board (keep current: update on every launch, finish and decision)

Last updated 2026-10-25 07:50 (G0 profiles final, decision 12; 1.7B run 6 ready: s 2.0, target stop 0.65 or 20 h, decision 13, [config](configs/paper_fluxtune_smollm2_yahoo.yaml); earlier 03:45: fwdllm setup smokes running, §3.5; big cells set by s 1.75 and Φ_budget 4, §2; 12 h cap gap, §6). 2026-10-24: plot list and figure runs reworked with priorities, §1 / §3.2; retired runs §3.3; 0.60-target runs moved to §7. Paths relative to `lib/python/examples/fwdllm/`; run dirs under `experiments/`.

**Running**

| ID | Node · GPUs · tmux | Started → ETA | Run dir · logs | Check |
|---|---|---|---|---|
| fwdllm setup smokes + bench parity (§3.5 ledger): `fluxtune` 1.7B pair, 1.7B `fwdllm` lr 3e-3, RoBERTa `fwdllm` on AG News, RoBERTa LoRA α 1; next `fluxtune` RoBERTa pair | shepherd · all 8 GPUs (tmux `dg_flame:fwd*`) | 10-25 03:42 → ~08:30 (est.) | `fwdllm/fwd_smoke_20261025/` (`pool_*.log`, `queue_*.log`, `bench_*/`, `harness_*.out`); scripts `gen*.sh`, `harness_leg.sh`, `queue_harness.sh`, `status.py` | §3.5 ledger |

The 10-24 RoBERTa S `fluxtune` run (`rbflux_20261024/`) never launched (dry run only, no run dir); its recipe (ρ0 0.14) is superseded by decision 7. Full FluxTune runs wait until the `fwdllm` setup is final (operator 10-25).

**Done**

| ID | Outcome | Detail |
|---|---|---|
| G0 charge profiles (FT-N1, 10-25, jayne, 06:55 → 07:28) | 30-min real legs, §2 recipe (pool 50, K 10, `exp`, `uniform`), decision 12. **DistilBERT** (n 30 / C 20, GPUs 0–3): 178 rounds, `drain_tail` 0.278 → **0.032 s**, `fedavg` 0.056 → **0.031 s**. **SmolLM2-1.7B** (n 16 / C 12, s 2.5, divisor 0.3, GPUs 4–7): 37 rounds / 7 commits, `drain_tail` 0.434 → **0.120 s**, `fedavg` 0.075 → **0.917 s** (median 0.024 s; the 7 commit rounds take 4.5–5.3 s, so ≈ 4.6 s per commit). 0 errors / OOM; SmolLM's 235 `[TIMING_OVERRUN]` are from the shortened D (expected). Old profiles overcharged DistilBERT ~0.3 s / round and undercharged 1.7B ~4 s / commit. Cross-check: the 10-24 shepherd `ft_n50` leg (n 50, `rm`) read DistilBERT `drain_tail` 0.09 s (host / n). Final for all future runs (decision 12); `fluxtune_yelp-p.yaml` still stale | `fwdllm/charge_profile_20261025/` (`run.sh`), `experiments/run_20261025_065511_g0_distil_*_real`, `run_20261025_065655_g0_smollm_*_real` → `sim_charge_profiles/fluxtune.yaml`, `fluxtune_yahoo.yaml` |
| D divisor per model (10-24, shepherd, 16:15 → 16:27) | measured A40 cost per task, one trainer alone per GPU: RoBERTa-L 0.0171 (0.0169 in use, kept), **SmolLM2-1.7B 0.0047 (0.0192 was 4.1× too large)**, DistilBERT 0.478 unchanged; launchers updated (SmolLM per arm 0.0047 / 0.047). Calculation: [D_CALIBRATION.md](D_CALIBRATION.md) | `fwdllm/ft_profile_20261024/` on shepherd (`run.sh`, 4 run dirs); re-run with `fwdllm/expt_scripts/profile_task_cost.sh`; FT-L18, FT-T13, FT-T14 |
| Sim vs real, big models (10-24, shepherd, 14:30 → 15:00) | `fluxtune` n 16 / C 8 / K 5, 10 min per leg, §2 recipe: RoBERTa-L sim 5 commits, 5.3 vclock h (33×), acc 0.53 → 0.57; SmolLM2-1.7B sim 2 commits, 2.2 vclock h (14×); both real legs 0 iterations (D 4–19 min). 0 errors. Not paper runs | `fwdllm/ft_models_check_20261024/` on shepherd (`run.sh`, 4 run dirs; `first_try/` = the 8-per-GPU SmolLM OOM); §4 table; FT-L11, L12, L17 |
| Sim vs real, DistilBERT n 50 (10-24, shepherd, 13:33 → 14:06) | `fluxtune` n 50 / C 20 / K 10, 30 min per leg: real 103 commits, acc 0.33–0.34; sim 345 commits, 6,202 vclock s (3.4×), acc 0.54 and rising, no collapse past the lap boundary; matched at c100. Not paper runs | `fwdllm/ft_n50_check_20261024/` on shepherd (`run.sh`, both run dirs); FT-L13 |
| FT-N18 B2 (10-24, jayne, 02:58 → 03:58) | fwdllm bench gate PASS on DistilBERT · AG News n 100 `syn_0`: `fwdllm` (3 bench vs 3 harness seeds, every same-seed pair identical) and `fluxtune` (3 vs A6, AUC 44.6 / 43.4). Not paper runs | `fwdllm/bench_B2_20261024/` (`b2.json`, `bench/`, `harness/`, `run_harness_seeds.sh`); harness run dirs `experiments/run_20261024_033641_b2h_s123[56]_*`; FLUXTUNE FT-D3 |
| F0 Λ readout | no run: Λ_need from P0 `exp`, G1, P0b → N\* 34–42 (AG News), 38 (Yahoo bs 32) → N 50 every cell | §3.1 |
| 1.7B · Yahoo dry run | learns: 99 commits, 43 s/commit at N 50 bs 8, acc 0.141 → 0.266 (`pool`, stopped by hand 1.23 h); not an S leg | §3.4 |
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
| G0 v2 charge profile (FT-N1) | short real legs at the §2 recipe (`uniform`, pool 50, agg goal 10), decision 12 | **done, final** 10-25 (decision 12; Done table): DistilBERT 0.032 / 0.031 s, SmolLM 0.120 / 0.917 s (`drain_tail` / `fedavg`); Yelp-P not re-profiled |
| S smoke SmolLM2-1.7B · Yahoo | — | **folded into run 6** (decision 13): n 60 fits (10-23 run: 14.6 h on 7 GPUs, 0 OOM); s/commit ~99 s and t_eval 104 s from the 10-23 run set stride 3; no commit cap, so `--max-commits` is not needed |
| S smoke RoBERTa-L · Yelp-P | — | running 10-24 19:13 (Running table above); launched as `ARMS=fluxtune HOURS=12 bash fwdllm/expt_scripts/nodes/run_node_roberta_yelp_to60_queue.sh` (defaults: TARGET 0.82, no target stop) at the §2 recipe (ρ0 0.14, T_res 71); its s/commit feeds F2 and `--max-commits`. 10-18 attempt at the old recipe (bs 8, pool 100) learned to 0.633 in 2 h (`run_20261018_175532`) |
| Big-model real↔sim check (FLUXTUNE FT-N21) | 1–2 h real leg + same-seed sim leg per model, paper packing | todo; confirms DistilBERT's parity transfers (0 overruns, accuracy within band at matched commits) |
| `fwdllm` lr × var screen | RoBERTa-L · Yelp-P (BitFit), 1.7B · Yahoo | running 10-25 on the bench (§3.5 W legs); best lr / var feed runs 3 and 5 (DistilBERT keeps 0.01) |

**Stop and eval flags (in code 10-22):** `--max-commits N` (halts, `stop_reason` `commit_cap`; scales the stop-rule warm-up
450 → 0.45·N, so set it near the commits a 12 h run reaches: ~380 on RoBERTa and 1.7B, est.); `--eval-resample` (fresh
`--eval-max-samples` draw per eval, seeded by commit); `--final-full-eval` (one full-test eval at run end, event `agg_eval_final`).

**Operator decisions (closed 10-22)**
1. P1 `fwdllm` runs to the 3 h wall cap (`--max-runtime-s` ≥ 40 × wall cap), not equal vclock.
2. `B_max origin` gate applies only when BmaxProbe is on; A0 and `pool` are not invalidated (code fix: EXPERIMENTS §8).
3. DistilBERT eval stride 4.
4. No iteration-cap arm: pool size is `tune_commit_rule` (N 10 / 20 / 90 + `n_target`).
5. 12 h cells (RoBERTa, 1.7B): bs 32 if F2 shows reachable Λ ≥ Λ_need, else bs 8 (TTA is vclock, but a DNF scores nothing).
6. DistilBERT figure runs at N 50; N 90 is run 19 (`tune_commit_rule`).
7. (10-25, supersedes horizon-aware T_res and 10-24's ρ0 0.14 / T_res 71) RoBERTa-L and 1.7B `fluxtune`: s 1.75, Φ_budget 4 →
   ρ0 0.0509 / 0.0436, T_res 1,072 / 1,461 (§2). DistilBERT stays ρ0 0.07.
8. (10-24) 12 h cells run on the harness, not the bench; the RoBERTa S run has no server momentum.
9. (10-25) RoBERTa-L `fwdllm` arm uses BitFit (FwdLLM's own choice for roberta-large); `fluxtune` keeps LoRA r 6. Big-cell
   runs may use the bench once §3.5 V1/V2 pass (revisits 8). `fwdllm` setup is finalised before any full FluxTune run.
10. (10-25) `fwdllm` changes so it trains, every cell (run 1 included): new cohort per commit (`reselect_cadence: data_bin`);
    per-model var threshold (1.7B screened at {15, 8.4, 4.6}); screened lr; stuck-eviction timeout ≥ `send_timeout_wait_s`.
    Table and reasons: [EXPERIMENTS §2](EXPERIMENTS.md#2-setup) "FwdLLM arm".
11. (10-25 07:15) Final `fwdllm` values: **1.7B lr 3e-3, var 8.4**; **RoBERTa-L BitFit lr 1e-2, var 0.2**; LoRA dropped for `fwdllm`.
    Choices table: [EXPERIMENTS §2](EXPERIMENTS.md#2-setup) "Current choices"; ledger: §3.5.

12. (10-25 06:50) G0 profiles come from short real legs (30 min, fewer trainers), not full-budget runs: the charged ops
    (`drain_tail`, `fedavg`) are per agg round (10 updates) server work, set by model, update size and pool, not by D or n.
    SmolLM's leg shortens D (divisor 0.3) so rounds happen in 30 min. Sim runs after a profile change can be re-timed from
    their `vclock_charge` events. **Final (operator 10-25 07:40): the 10-25 profiles (`fluxtune.yaml`, `fluxtune_yahoo.yaml`) are used
    for every future run; G0 is closed (no sim grade needed).** 1.7B `fluxtune`: **s 2.0, Φ_budget 4 → ρ0 0.0498, T_res 1,119**
    (Φ = 3 at ~1,760 commits; supersedes 7 for 1.7B and the 2.5 used by the G0 leg).
13. (10-25 07:50) 1.7B `fluxtune` stop rule: stop once acc ≥ 0.65 on 2 consecutive evals (`--target-stop 0.65 --target-stop-window 2`,
    in-process), else at the 20 h wall cap (`--sim-wall-ceiling-h 20`). No commit cap, saturation stop off; Φ rail at 4 (`--phi-stop-threshold 4`,
    operator 10-25; Φ → Φ_budget 4 from below, so in practice it never fires). Every one of these exits cleanly, so `--final-full-eval` scores the full test set at the end (`agg_eval_final`).
    `--target-acc` (watcher) kills the run and skips it, so this launcher doesn't use it. Launcher defaults: n 60, C 30, K 10, stride 3
    (`launch_smollm17b_yahoo_n50.sh`).

Closed by data, not by decision: grad-aware (G1), bs (P0b → 32, subject to F2), Yelp-P/Yahoo eval stride (S), v1 profile (G0).

**Then the figure runs** in §3.2 priority order (P1 → P2 → P3); big-model runs 3–6 after S and the lr screens; 23–25 after FT-N5.

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

All rendered by `fwdllm/expt_scripts/make_paper_figs.py` into `paper_figs/<manifest>/`. The names below are the paper
names; plots marked *new* need a renderer or manifest group first. Runs are §3.2 numbers. Every run goes to peak accuracy (§2 stop rule), not to a preview target.

**End to end** (`figs_e2e.yaml`; one panel per cell: DistilBERT · AG News, RoBERTa-L · Yelp-P, SmolLM2-1.7B · Yahoo)

| Plot | Shows | Place | Runs | Renderer today |
|---|---|---|---|---|
| `E2E_acc_vs_time` | test accuracy vs sim time, "N× faster" at target | main | 1–6 | `e2e_<cell>` panel (a) |
| `E2E_acc_vs_commits` | test accuracy vs server commits (better steps vs more steps per hour) | main | 1–6 | `e2e_<cell>` panel (c) |
| `E2E_comm_to_target` | GB to target, downlink + uplink stacked, "N× less" | main | 1–6 | `e2e_<cell>` panel (b) |
| `E2E_participation_time_cdf` | CDF over clients of total engaged time in the run (sum of a client's sessions) | main | 1–6 | *new* (sum `session_durs` / `round_span_durs` per client) |
| `E2E_session_length_cdf` | CDF of one engagement: FluxTune one task (dispatch → commit), FwdLLM one round | main | 1–6 | `b1_<cell>` |

**Concept contribution** (`figs_ablation.yaml`, *new* group; DistilBERT · AG News; FluxTune vs FluxTune without one concept: acc vs time, acc vs commits, "x× slower without")

| Plot | Concept removed | Place | Runs |
|---|---|---|---|
| `Contri_probe_mean` | E1: mean of 10 probes → FwdLLM's aligned 1-of-10 (run 7), or the max-\|JVP\| 1-of-10 (run 26) | main | 2, 7, 26 |
| `Contri_trust_ratio` | E2: trust-ratio step with `exp` decline → FwdLLM's SGD | main | 2, 8 |
| `Contri_fixed_pool` | E3: fixed pool N → FwdLLM's variance gate | main | 2, 9 |
| `Contri_async` | S1: async → sync rounds | main | 2, 10 |

**Hyperparameter tuning** (`figs_ablation.yaml`; DistilBERT · AG News)

| Plot | Arms | Place | Runs | Renderer today |
|---|---|---|---|---|
| `tune_rho0` | ρ0 0.07 / 0.10 / 0.14 / 0.28 (Φ\* 2, so T_res 283 / 139 / 71 / 18) | main | 2, 14, 15, 16 | `c_rho_max` |
| `tune_commit_rule` | when to commit: variance gate (FwdLLM) vs SNR target (`n_target`) vs fixed N 10 / 20 / 50 / 90 | main | 2, 9, 17, 18, 19, 20 | `c_agg_criteria` |
| `tune_rho_schedule` | constant ρ vs `exp` vs `exp` + server momentum 0.5; plus ρ and Φ per commit | main | 2, 21, 22 | `c_step_size` + `c_step_size_step_size` (without `rm`) |
| `tune_batch_size` | bs 8 vs 32 (DistilBERT · Yahoo, N 90) | appendix | P0b (§0c), no new run | `fwdllm/scripts/plot_p0b_bs8_vs_bs32.py` |
| `tune_agg_weights` | uniform ω vs alignment-only ω (`grad_aware` without the staleness × utility base) | appendix | 2, 27 | `c_agg_weights` |

**Strategy comparison** (`figs_availability.yaml`; DistilBERT · AG News; `syn_0` unless named)

| Plot | Strategy vs FwdLLM and FluxTune | Place | Runs |
|---|---|---|---|
| `strategy_fwdllm_async` | FwdLLM's estimator, async per iteration (`fedbuff_it_unaware`) | main | 1, 2, 11 |
| `strategy_fedbuff` | FedBuff(P) (`fedbuff_round`) | main | 1, 2, 13 |
| `strategy_felix` | Felix(P) (`felix_round`, availability-aware) | main | 1, 2, 12 |
| `strategy_availability_mobiperf` | FluxTune, Felix(P), FedBuff(P) on `mobiperf_3st_50` | main if FT-N5 lands | 23, 24, 25 |

**Not plotted:** `b2` (diversity), `b3` (progression), `c_phase_a`, `c_phase_a_pairs`, `c_phase_b`, `c_update_utility`,
`c5_generalisation` (deferred, operator 10-22).

## 2. Cells, pool size and eval

**Common to every run:** sim mode, `syn_0` (except C4), Dirichlet α=1, **1 seed per arm**,
`server_update_audit: true`, cos audit off, `tta_consecutive: 2`, the same stop rule (stall 0.003 + GL decay + commit cap + wall cap),
async C = 0.3·n, K = C/3.
**FluxTune** = the [EXPERIMENTS.md §2](EXPERIMENTS.md#2-setup) recipe with `--pool-target {N}`. With 1 seed, a gap
under the 2.3-pt replicate noise is unresolved.

| Cell | p | Clients n | Pool N | ρ0 (`--rho0`) | T_res | implied s | Target | Wall cap | Eval sample (rows) | Stride |
|---|---|---|---|---|---|---|---|---|---|---|
| `distilbert_agnews` | 450,340 | 100 | 50 | 0.07 | 283 | 2.10 | 0.86 | 3 h | full test (7,600) | 4 |
| `roberta_large_yelp` | 591,874 | 100 | 50 | 0.0509 (from s) | **1,072** (Φ_budget 4) | **1.75** (set) | 0.82 | 12 h | 10% = 3,800, resampled per eval | set from S (4, est.) |
| `smollm2_1p7b_yahoo` | 806,912 | 75 (60 if OOM) | 50 | 0.0498 (from s) | **1,119** (Φ_budget 4) | **2.0** (set, decision 12) | 0.65 | 12 h | 10% = 6,000, resampled per eval | set from S's bs 32 s/commit (4, est.) |

C = 0.3·n, K = C/3: 30 / 10 at n 100, 23 / 8 at n 75, 18 / 6 at n 60. Big-cell bs: 32 unless F2 fails (then 8).

**Per-cell launch flags** (on top of the [EXPERIMENTS §2](EXPERIMENTS.md#2-setup) recipe; `{}` = still open):

```
# distilbert_agnews (bs 8, full test set, so no --eval-max-samples)
--dataset agnews --num-trainers 100 --pool-target 50 --rho0 0.07 --t-res 283 --train-batch-size 8
--max-commits 1000 --final-full-eval --eval-every-n-commits 4 --sim-wall-ceiling-h 3
# roberta_large_yelp (fwdllm arm: --learning-rate {lr screen})
--dataset yelp-p --model-type roberta-large --peft-method lora --lora-r 6 --num-trainers 100
--pool-target 50 --gate-safety-s 1.75 --t-res 1072 --train-batch-size {32 | 8 per F2}   # no --rho0
--max-commits {from S, ~370} --eval-max-samples 3800 --eval-resample --final-full-eval
--eval-every-n-commits {stride} --sim-wall-ceiling-h 12 --delay-divisor 0.0171   # fwdllm arm: 0.171
# smollm2_1p7b_yahoo (all of this is fwdllm/expt_scripts/launch_smollm17b_yahoo_n50.sh: ARM, HOURS, MAX_COMMITS, STRIDE env)
--dataset yahoo --model-type llama --model-name HuggingFaceTB/SmolLM2-1.7B --peft-method lora --lora-r 4 --num-trainers {75 | 60}
--pool-target 50 --gate-safety-s 2 --t-res 1119 --train-batch-size {32 | 8 per F2}   # no --rho0
--target-stop 0.65 --target-stop-window 2 --eval-max-samples 6000 --eval-resample --final-full-eval   # no --max-commits (decision 13)
--eval-every-n-commits 3 --sim-wall-ceiling-h 20 --delay-divisor 0.0047   # fwdllm arm: 0.047
# fwdllm arm on 1.7B: --learning-rate {lr screen}
# every cell: --no-cos-ground-truth-audit
# every cell: --max-runtime-s 1000000000 --force, so the wall ceiling, not the vclock, binds (D_CALIBRATION §6)
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
- **Calibration.** vclock = D per task (`--delay-divisor` per model and arm: [D_CALIBRATION.md](D_CALIBRATION.md)) +
  aggregator charges (`sim_charge_profiles/fluxtune_<dataset>.yaml`, < 1 s per op). AG News and Yahoo re-profiled 10-25 from
  30-min real legs (decision 12); Yelp-P still predates the §2 recipe (`--allow-stale-profile`).
- **Caps.** Sim: `--max-runtime-s 1000000000` plus `--sim-wall-ceiling-{s,h}` and `--force` (the wall-budget preflight
  expects the vclock cap to bind); the ceiling ends the leg. 40 × wall cap ended rb60 at 4.7 h of 12. Real:
  `--max-runtime-s` is wall from aggregator start, join included (~3.5 min at n 100).
- **Real legs are slow by design.** At recipe D a RoBERTa-L iteration takes 4–46 min, 1.7B 14 min–2.8 h; first commit ≥ 1 h; a
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
- **RoBERTa-L and 1.7B: s 1.75, Φ_budget 4 (operator 10-25); ρ0 = s·√(P·N/p), T_res = 2·lnΦ_budget/ρ0².**
  s 1.75 sits between 1.5 (validated on DistilBERT's three tasks) and 2 (compute limit; above it under-trains). Results:
  ρ0 0.0509 / 0.0436, T_res 1,072 / 1,461, Φ = 3 at ~1,685 / ~2,300 commits. **1.7B now s 2.0 → ρ0 0.0498, T_res 1,119** (decision 12).
  Pass `--gate-safety-s {s} --t-res {T_res}`,
  no `--rho0`; the aggregator logs ρ0. Why s and Φ, not ρ0: [EXPERIMENTS §2.1](EXPERIMENTS.md#21-pool-n-step-s--ρ0-and-φ_budget--t_res-ρ_max-and-n--50-runs-open-3-p0c).
- **DistilBERT: ρ0 0.07, T_res 283 (s 2.10, Φ_budget 2).** `tune_rho0` tests 0.10 / 0.14 / 0.28 (§3.2).
- **First-hours abort rule (RoBERTa-L, 1.7B):** Φ off (1+ρ²)^(t/2) by > 1%, no accuracy above chance by ~c50, or GL decay
  before Φ ≈ 1.5 → lower s to 1.5 (T_res from Φ_budget 4); N stays.

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
| G0 (FT-N1) | DistilBERT · AG News; SmolLM2-1.7B · Yahoo | v2 real legs with the §2 recipe (`uniform`) → `sim_charge_profiles/fluxtune.yaml`, `fluxtune_yahoo.yaml` (FT-T8: no v2 sim on the v1 profile) | 1 + 1 (10-25; DistilBERT cross-checked by the 10-24 `ft_n50` leg) | 30 min | done, final 10-25 | every sim run |
| P0 A/B | DistilBERT · AG News | FluxTune `exp` (T_res 300, ρ0 0.067, N 90) vs `--rho-schedule pool`; compare TTA at 0.80/0.86/0.87, peak, Φ at peak, at equal commits. Config: [p0_exp_distilbert_agnews.yaml](configs/p0_exp_distilbert_agnews.yaml) (`pool` = same with `--rho-schedule pool`); both arms: stall rule and GL off, cos audit off, uniform ω, v1 charge profile, 2 h wall cap | 2 | 2 h wall each (FT-N4 closed without a 4 h run) | **done 10-18: `exp` chosen** (§0) | the recipe; `pool` reused in C3 `step_size` |
| G1 grad-aware | DistilBERT · AG News | P0 `exp` config with `--agg-rate-type grad_aware` vs `uniform`; TTA at 0.80/0.86/0.87, peak, ω spread | 1 new (`uniform` = P0 `exp`) | 2 h | **done 10-18: `uniform` chosen** (§0b) | the recipe; `grad_aware` reused in C3 `agg_weights` |
| P0b bs | DistilBERT · Yahoo | bs 8 vs 32 at N 90 | 2 | bs 8 2.3 h (plateau); bs 32 3.7 h (operator stop) | **done 10-19: bs 32** (§0c) | bs for Yelp-P and Yahoo |
| S smoke | RoBERTa-large · Yelp-P (LoRA q,v r=6, seq 256); SmolLM2-1.7B · Yahoo (LoRA q,v r=4, seq 256) | FwdLLM + FluxTune at §2's N | 4 | 2 h cap | RoBERTa ran 10-24 19:13; 1.7B folded into run 6 (decision 13) | s/commit, t_eval, charge profile, distinct-clients/N for P3/P4 |
| F0 Λ readout | DistilBERT · AG News, Yahoo | `expt_scripts/lambda_need.py` on P0 `exp`, G1, P0 `pool`, P0b bs 8/32: Λ at target and at peak − 0.5 pt | 0 | — | **done 10-21** ([cheatsheet §8](../fwdllm/fl_fwd_ft_pipeline_cheatsheet.md#8-choosing-n-ρ0-and-t_res)) | N\* 34–42 (AG News), 38 (Yahoo bs 32), 53 (Yahoo bs 8) → N 50; accuracy vs Λ schedule-independent |
| F2 compute check | RoBERTa-L · Yelp-P; SmolLM2-1.7B · Yahoo | no run: S s/commit scaled to N 50 → T = 12 h ÷ s/commit → Λ reachable = (2·ln4/s)·(1 − e^{−T/T_res}) at s 1.75, vs Λ_need | 0 | — | after S | bs 32 if reachable Λ ≥ need, else bs 8 (decision 5). Est.: RoBERTa Λ ≈ 0.46, 1.7B ≈ 0.41 (§6) |
| F3 transfer checks | RoBERTa-L · Yelp-P; SmolLM2-1.7B · Yahoo | no extra run: (a) S's first `trainable_weight_norm` vs 13.35·√(p/450,340) = 15.3 / 17.9; (b) P3/P4 `lambda_need.py`: Λ at milestones vs DistilBERT × √(450,340/p) | 0 | — | (a) after S, (b) after P3/P4 | if either fails, N picks up a model factor; raise N for C5 and the 3-seed reruns |

### 3.2 Figure runs

Run in priority order: **P1** before **P2** before **P3**; within a priority, top to bottom. G0 closed 10-25 (decision 12).
A 4-value tuning sweep runs 3 values at P2 and the 4th at P3 (3 points still plot the trend).
DistilBERT · AG News runs: §2 recipe (N 50, ρ0 0.07, T_res 283, bs 8), 3 h cap, 1 seed, to peak accuracy. Flags below are the change from that.

| Pri | # | Run | Model · Dataset | Flags (change from the cell recipe) | Time | Status | Plots |
|---|---|---|---|---|---|---|---|
| P1 | 1 | FwdLLM | DistilBERT · AG News | `fwdllm` arm, lr 0.01 | 3 h | ready (G0 closed 10-25) | `E2E_*`; reference in `strategy_*` |
| P1 | 2 | FluxTune | DistilBERT · AG News | none | 3 h (hits the 1,000-commit cap at ~1.9 h) | ready (G0 closed 10-25) | `E2E_*`, every `Contri_*`, every `tune_*` except `tune_batch_size`, `strategy_*` |
| P1 | 7 | FluxTune − E1 | DistilBERT · AG News | drop `--probe-combine mean` (FwdLLM's aligned 1-of-10, P 10) | 3 h | on `fluxtune` `select` alone gives max-\|JVP\| (= run 26): also set `select_perturbation_using_jvp: false` (yaml override) for the cosine pick | `Contri_probe_mean` |
| P1 | 8 | FluxTune − E2 | DistilBERT · AG News | drop `--server-step-rule trust_ratio --rho-schedule exp --rho0 --t-res` (SGD, lr 0.01) | 3 h | ready (G0 closed 10-25) | `Contri_trust_ratio` |
| P1 | 9 | FluxTune − E3 | DistilBERT · AG News | `--commit-gate var` with FwdLLM's variance policy, no `--pool-target` | 3 h | variance gate under async never run | `Contri_fixed_pool`, `tune_commit_rule` |
| P1 | 10 | FluxTune − S1 (= A123) | DistilBERT · AG News | `fwdllm` sync arm + E1, E2, E3 flags | 3 h | ready (G0 closed 10-25) | `Contri_async` |
| P1 | 3 | FwdLLM | RoBERTa-L · Yelp-P | `fwdllm` arm, screened lr | 12 h | blocked: S, F2, lr screen | `E2E_*` |
| P1 | 4 | FluxTune | RoBERTa-L · Yelp-P | cell recipe (§2: s 1.75, T_res 1,072, bs 32) | 12 h | blocked: S, F2 | `E2E_*` |
| P1 | 5 | FwdLLM | SmolLM2-1.7B · Yahoo | `fwdllm` arm, screened lr | 12 h | blocked: S, F2, lr screen | `E2E_*` |
| P1 | 6 | FluxTune | SmolLM2-1.7B · Yahoo | cell recipe (§2, decisions 12–13: n 60, C 30, K 10, s 2.0, T_res 1,119, bs 32, divisor 0.0047; stop at 0.65 ×2 or 20 h, Φ rail 4, then full-test eval). Config [paper_fluxtune_smollm2_yahoo.yaml](configs/paper_fluxtune_smollm2_yahoo.yaml); launch `ARM=fluxtune bash lib/python/examples/fwdllm/expt_scripts/launch_smollm17b_yahoo_n50.sh` (jayne, 8 GPUs) | ≤ 20 h | **ready** (10-25) | `E2E_*` |
| P2 | 11 | FwdLLM async (A4) | DistilBERT · AG News | `fedbuff_it_unaware` with FwdLLM's estimator | 3 h | ready (G0 closed 10-25) | `strategy_fwdllm_async` |
| P2 | 12 | Felix(P) | DistilBERT · AG News | `felix_round`, `baselines.yaml` lr (0.075 placeholder) | 3 h | ready (G0 closed 10-25) | `strategy_felix` |
| P2 | 13 | FedBuff(P) | DistilBERT · AG News | `fedbuff_round`, `baselines.yaml` lr | 3 h | ready (G0 closed 10-25) | `strategy_fedbuff` |
| P2 | 15 | ρ0 0.14 | DistilBERT · AG News | `--rho0 0.14 --t-res 71` | 3 h | ready (G0 closed 10-25) | `tune_rho0` |
| P2 | 16 | ρ0 0.28 | DistilBERT · AG News | `--rho0 0.28 --t-res 18` (upper limit; N\* scales as ρ0², so expected to break at N 50) | 3 h | ready (G0 closed 10-25) | `tune_rho0` |
| P2 | 17 | N 10 | DistilBERT · AG News | `--pool-target 10` | 3 h | ready (G0 closed 10-25) | `tune_commit_rule` |
| P2 | 19 | N 90 | DistilBERT · AG News | `--pool-target 90` (also confirms N 50 against P0's N 90) | 3 h | ready (G0 closed 10-25) | `tune_commit_rule` |
| P2 | 21 | constant ρ | DistilBERT · AG News | `--rho-schedule pool` (ρ 0.07; ends on the Φ rail) | 3 h | ready (G0 closed 10-25) | `tune_rho_schedule` |
| P3 | 14 | ρ0 0.10 | DistilBERT · AG News | `--rho0 0.10 --t-res 139` | 3 h | ready (G0 closed 10-25) | `tune_rho0` (4th value) |
| P3 | 18 | N 20 | DistilBERT · AG News | `--pool-target 20` | 3 h | ready (G0 closed 10-25) | `tune_commit_rule` (4th value) |
| P3 | 20 | adaptive pool | DistilBERT · AG News | `--commit-gate n_target` (no `--pool-target`) | 3 h | ready (G0 closed 10-25) | `tune_commit_rule` |
| P3 | 22 | momentum 0.5 | DistilBERT · AG News | `--server-momentum 0.5` (0.9 diverged) | 3 h | ready (G0 closed 10-25) | `tune_rho_schedule` |
| P3 | 23 | FluxTune, `mobiperf_3st_50` | DistilBERT · AG News | trace `mobiperf_3st_50` | 3 h | blocked: FT-N5 | `strategy_availability_mobiperf` |
| P3 | 24 | Felix(P), `mobiperf_3st_50` | DistilBERT · AG News | as 12, trace `mobiperf_3st_50` | 3 h | blocked: FT-N5 | `strategy_availability_mobiperf` |
| P3 | 25 | FedBuff(P), `mobiperf_3st_50` | DistilBERT · AG News | as 13, trace `mobiperf_3st_50` | 3 h | blocked: FT-N5 | `strategy_availability_mobiperf` |
| P3 | 26 | FluxTune, max-\|JVP\| upload | DistilBERT · AG News | `--probe-combine select` (keeps `select_perturbation_using_jvp: true`: upload the 1 of 10 probes with the largest \|JVP\|; same 2P passes, D and bytes as run 2) | 3 h | ready (G0 closed 10-25) | `Contri_probe_mean` |
| P3 | 27 | FluxTune, alignment-only ω | DistilBERT · AG News | `--agg-rate-type grad_aware` + yaml override `agg_rate_conf.base: neutral` (ω = alignment gate only, base 1.0; the launcher hardcodes `base: new`) | 3 h | ready (G0 closed 10-25) | `tune_agg_weights` |

Deferred (operator 10-22): C5 generalisation (EXPERIMENTS §4).

### 3.3 Retired runs (done, not used in any paper plot)

Kept for the record; none feeds a §1 plot. Paths relative to `lib/python/examples/fwdllm/`, run dirs under `experiments/`.

| Run(s) | What it was | Why retired | Where |
|---|---|---|---|
| P0 `exp`, P0 `pool`, FwdLLM A0 preliminary (10-18) | recipe decision runs (N 90, ρ0 0.067, v1 profile) | old config; superseded by runs 2, 21, 1 | §0 |
| G1 `grad_aware` (10-18) | uniform vs grad-aware ω | decided `uniform`; cited in the text, no plot | §0b |
| RoBERTa-L 10-18 attempt (`run_20261018_175532`) | old recipe (bs 8, pool 100) | old recipe | — |
| 1.7B · Yahoo dry run, screen, void 10-19 | systems check at the old recipe | not the cell recipe | §3.4 |
| ag60 preview (10-22), rb60 `fwdllm` (10-23) | 0.60-target runs | preview target only | §7 |
| Sim vs real checks, D divisor profile, FT-N18 B2 (10-24) | calibration and parity | not paper runs | status board Done |
| §4 scaling runs (360M, 1.7B · AG News, Oct DistilBERT) | run-time inputs | not paper runs | §4 |

**Dropped from the plan before running:** FT-N22 momentum seeds (run 22 covers momentum); F1 Yelp-P N check (P3 reaching
target confirms N); N50 check (run 19); R1/R2 as gating runs (now runs 14/15 at N 50); C1 singles, P2 pairs, C2 ladder
(replaced by `Contri_*`); `update_utility` arms (cos-sim, top-1, antithetic, orthogonal); `agg_weights` `new`; `step_size`
`rm`, `phi`, Adam/Yogi, normalised SGD, clipping; T_res sweep (ρ0 sweep covers it).

### 3.4 Run log: SmolLM2-1.7B · Yahoo

Config for all three (unless noted): FluxTune, sim, 50 trainers on 8 A40s (6–7 per GPU), shards 0–49 of
`niid_label_clients=100_alpha=1`, K 10 / C 30, LoRA q,v r=4 (p 806,912), seq 256, `--commit-gate fixed --pool-target 50
--rho-schedule pool`, delays `--delay-divisor 0.0192 --delay-floor 4.0`, `--send-timeout-s 3600`,
`--no-cos-ground-truth-audit`, eval 6,000 fixed rows. Dry run to check that 1.7B learns (operator 10-22): not an S leg, not P4; deviates from the §2 cell (n 75, `exp`, ρ0 0.07, bs 32).
Charge profile is the DistilBERT-derived `fluxtune_yahoo.yaml` (`--allow-stale-profile`).

| Date | Run dir (`fwdllm/experiments/`) | Wall | Result |
|---|---|---|---|
| 10-19 | `run_20261019_020059` (r=8, catalog cos audit on) | 43 min | **void**: 0 commits. The cos audit's fp32 backward on 1,024 × 256 rows ran on the aggregator's CPU model; it hung 38 min, then the aggregator was SIGKILLed. The launcher still printed `COMPLETED crash=0` (fixed 10-22: a non-zero aggregator exit now reads `AGG_KILLED`). Pass `--no-cos-ground-truth-audit` for any 1.7B run. |
| 10-20 01:26 | `run_20261020_012606` (screen, stride 5) | 15 min (cap) | **pass**: 50/50 trainers, 17 commits (~43 s each), 0 OOM/Traceback/overrun/abandon; ≤ 4.1 GB GPU and ≤ 2.3 GB host per trainer, aggregator 9.8 GB; acc 0.141 → 0.149 (commit 15); sim clock ~14× wall |
| 10-23 22:00 (shepherd) | `run_20261023_220053_fluxtune_yahoo_n60_smoke_syn_0_sim` (n 60, N 50, bs 32, `exp` ρ0 0.07, T_res 283 → s 2.81, Φ_budget 2; eval 6,000 rows stride 4) | 14.6 h (ended without a stop reason) | 530 commits, ~99 s/commit; peak **0.598**, 11-eval mean 0.562 → 0.581 → 0.590 over the last ~250 commits; Φ 1.80 at the end, 85% of its ln 2 budget spent. Budget-bound: the rest adds ≲ +0.015. Basis for s 1.75 / Φ_budget 4 (§2) |
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

### 3.5 Sanity and smoke runs (no figure)

**Done.** Paths relative to `lib/python/examples/fwdllm/`.

| Date · node | Run | What it checked | Result | Where |
|---|---|---|---|---|
| 10-18 shepherd | P0 / G1 / P0b queues | recipe choices (DistilBERT) | §0–§0c | §0–§0c |
| 10-19 / 10-20 shepherd | 1.7B · Yahoo void, screen, dry run | 1.7B systems: memory, pace, learns | void (cos audit hang); screen pass; dry run 0.141 → 0.266 | §3.4 |
| 10-22 jayne | ag60 preview (23 arms) | every DistilBERT arm runs and learns | 22 / 23 to 0.60 | §7 |
| 10-23 jayne | rb60 `fwdllm` LoRA lr 0.01, var 0.3 (JSON override, not 0.2) | RoBERTa `fwdllm` learns | no: acc 0.54 → 0.49, eval loss 0.69 → 0.80; logged var 0.36–9.8 > 0.3, so the gate rarely passed | `rb60_20261023_160630/` |
| 10-23 shepherd | 1.7B · Yahoo n 60 `fluxtune` | 1.7B budget at s 2.81 | peak 0.598, budget-bound | §3.4 |
| 10-24 shepherd | sim vs real, DistilBERT n 50 | real ↔ sim at matched commits | matched at c100 | status board Done |
| 10-24 shepherd | sim vs real, RoBERTa-L, 1.7B (10 min legs) | real ↔ sim, big models | not closed: real legs 0 commits (FT-N21 todo) | `ft_models_check_20261024/` |
| 10-24 shepherd | D divisor profile | task cost per model | SmolLM 0.0047 | `ft_profile_20261024/` |
| 10-24 jayne | FT-N18 B2 | bench ≡ harness, DistilBERT `fwdllm`, `fluxtune` | pass, same-seed pairs identical | `bench_B2_20261024/` |

**FwdLLM reference values** (github.com/UbiquitousLearning/FwdLLM): lr 0.01 for every model and dataset
(`fedFwd_roberta-large_yelp-p_lr0.01_client_num_1000`); var threshold per model, not dataset (`FedSgdAggregator.py`:
roberta-large 0.2, bert 0.2, distilbert 0.1, albert 0.1; none for llama); BitFit for non-DistilBERT; warmup 0, linear decay over
3,000 rounds (ours logs lr 0.01 constant at c365); 1,000 uniform clients. Our runs add `FWDLLM_FD_SCALE_INVARIANT=1` (kept, as every run).

**Running 10-25 (shepherd, `dg_flame:fwdsm`, `fwd_smoke_20261025/`).** Every leg: `fwdllm` sim, n 20 (shards 0–19 of
`niid_label_clients=100_alpha=1`), c 10 / K 10 (= n 100's), bs 32, eval 1,000 rows (resampled) stride 4, no final eval, seed 1234,
D 0.169 (RoBERTa) / 0.047 (1.7B). RoBERTa-L: BitFit. 1.7B: LoRA q,v r 4. Bench legs 4 workers.

*Why n 20 gives the same answer here:* `fwdllm` is sync with 10 clients per iteration at any n, so per-commit compute, gradient
noise and the var gate are n-independent; 20 shards hold ~5× the samples 100 commits use. n 20 lets a harness twin fit on 2–4 GPUs.
It does not hold for `fluxtune` (pool N 50 > n 20 repeats clients), so FluxTune runs keep n 100 / 75.

| ID | Where | Arms | Gives | Pass |
|---|---|---|---|---|
| V1 | bench GPU 2 + harness GPUs 0–1 | RoBERTa BitFit, lr 0.01, var 0.2, 25 commits | bench ≡ harness for RoBERTa + BitFit; bench leg memory → `bench_profiles.yaml` `yelp-p` | same losses and commit sequence as the harness twin; 0 OOM / Traceback |
| V2 | bench GPU 3, then harness GPUs 0–3 | 1.7B LoRA, lr 1e-3, var 0.2, 25 commits | the same for the `llama` path; memory → `yahoo` profile | as V1 |
| W_sm lr | bench GPUs 4–7 | 1.7B lr {1e-4, 3e-4, 1e-3, 3e-3}, var 0.2, 100 commits / 2 h | 1.7B lr | best eval-loss slope; acc > 0.15 (chance 0.10) |
| W_sm var | bench GPUs 4–7 | 1.7B lr 1e-3, var {0.1, 0.4} | gate pass rate vs commits / h; logged var distribution (`characterize_variance_curve.py`) | — |
| W_rb lr | bench GPUs 4–7 | RoBERTa BitFit lr {1e-3, 3e-3, 1e-2, 3e-2}, var 0.2 | does BitFit at FwdLLM's lr learn; lr bracket | acc > 0.55 (chance 0.50), eval loss falling 3 evals |
| W_rb var | bench GPUs 4–7 | RoBERTa lr 0.01, var {0.1, 0.4} | as W_sm var | — |

Kill a W leg at c50 if eval loss rises. W legs count only once V1 / V2 pass. Status: `python fwd_smoke_20261025/status.py`.

**Ledger (10-25, shepherd; all paths under `fwdllm/fwd_smoke_20261025/`).** The log below the ledger has the detail.

*Verified*

| When | What | Evidence | Where |
|---|---|---|---|
| 04:20 | bench ≡ harness, RoBERTa `fwdllm` (BitFit, `round` cadence) | V1r: 25 / 25 commits and every eval identical, once the harness draws its first cohort from all n | `bench_identity/V1*`, `experiments/run_*fwdsm_V1r*` |
| 04:25 | 1.7B `fwdllm` stuck-eviction bug found and fixed | 162 → 0 evictions; regression test; 444 tests pass | `fwdllm_aggregator.py`, `random.py`, `test_fwdllm_reselection.py` (uncommitted, jayne + shepherd) |
| 04:40 | pinned cohort causes RoBERTa's one-class collapse | 1 selection for the run; cohort 58% class 1 → predicted class 1 0.996; another cohort → class 0 | `bench_sweep/W_rb*` |
| 05:00 | `data_bin` cadence works | selections = commits + 1; 19 / 20 clients by c4; collapse gone | `bench_databin/` |
| 06:10 | bench ≡ harness, 1.7B `fwdllm` (`data_bin`) | V2c: 12 / 12 commits and 3 / 3 evals identical; bench 2.4× slower | `bench_sm_databin/`, `experiments/run_*fwdsm_D_sm_lr0.001*` |
| 07:15 | bench ≈ harness, 1.7B `fluxtune` (n 16, C 8, K 5, N 10), interim at c6 | per-commit gap 1e-6 to 3e-5 relative, not growing; eval acc identical; loss gap 0.0016 → 0.0001. Weights already differ by 1e-5 at c1 (load or LoRA-init precision). Bench 2× slower | `bench_F_sm/`, `experiments/run_*fwdsm_F_sm_n16_N10*` |
| 07:30 | **bench ≈ harness, 1.7B `fluxtune`: pass** | harness reached c25 (07:28); over the 14 shared commits the update-norm gap is ≤ 3.2e-5 relative and not growing; 4 / 4 evals agree to ≤ 0.005 loss and ≤ 0.002 acc. Float-order differences in the async pool, not logic | `bench_F_sm/`, `experiments/run_*fwdsm_F_sm_n16_N10*` |
| 07:30 | RoBERTa-L `fwdllm` is flat on AG News too, early (c22) | loss 1.385–1.397 ≈ ln 4, acc 0.25–0.29, MCC +0.02. Points to the model / head, not the binary task (1.7B moved by c5–9) | `bench_A_rb/` |
| 07:15 | LoRA doesn't beat BitFit on RoBERTa `fwdllm` | loss change over the run −0.002 (LoRA α 1, c42; LoRA α 100, c100) vs −0.005 / −0.009 (BitFit α 1 / α 100) | `bench_lora/`, `bench_lora_a100/` |
| 07:15 | non-IID isn't RoBERTa's bottleneck | near-IID BitFit (α 100) makes about the same progress per commit as α 1 (best loss 0.680 vs 0.678) | `bench_uniform/` |

*Final values (`fwdllm` arm)*

| Cell | Value | Final since | Basis |
|---|---|---|---|
| all cells | `reselect_cadence: data_bin` | 05:00 | collapse fix; matches upstream |
| 1.7B · Yahoo | **lr 3e-3** | 06:40 | beats 3e-4 / 1e-3 at matched commits; 1e-2 overshoots (c1 loss 4.10) |
| 1.7B · Yahoo | var **8.4** (working; var 15 optional check) | 06:20 | ~10 iterations per commit; learns: the final-values smoke `D_sm_lr0.003_v8.4` (05:46 → 07:45, 67 commits) reached acc 0.384 (c65), best loss 2.03, MCC +0.32. Var is a speed knob |
| RoBERTa-L · Yelp-P | **BitFit, lr 1e-2, var 0.2** | 07:15 | only setup with a steady gain (best loss 0.678, MCC +0.13 at c82); lr ≤ 3e-3 under-steps, 3e-2 noisy; var 0.4 the same per commit but noisier; LoRA flatter. Weak in absolute terms: Λ ≈ 0.05 by c90 |

*Running (07:25)*

| Run | GPU | Gives |
|---|---|---|
| `fluxtune` 1.7B pair to c25 | harness 0, 1, 7; bench 5 | final bench ≈ harness call for 1.7B `fluxtune` |
| 1.7B `fwdllm` lr 3e-3, var 8.4 | harness 2–3 | the 1.7B curve to c100 / 2 h |
| RoBERTa `fwdllm` BitFit lr 1e-2, var 0.2 on **AG News** (`A_rb_*`, n 20) | bench 4 | model vs task: does RoBERTa-L `fwdllm` learn on a 4-class task (DistilBERT reached 0.851)? |
| RoBERTa LoRA α 1 | bench 6 | to c65; flat so far |
| `fluxtune` RoBERTa pair (n 16, C 8, K 5, N 10, s 1.75, T_res 1,072, LoRA r 6), started 07:32 | harness 0–1; bench 7 | bench ≈ harness for RoBERTa `fluxtune` |

**Interim results (10-25 04:20).**
- **V1 pass.** The bench and harness legs match exactly once the harness draws its first cohort from all n (iterations per commit 12, 8, 9, 16, …;
  evals 0.691/0.526 → 0.686/0.546 → 0.700/0.489). The first harness try (`--min-initial-frac 0.9`) selected from 18 of 20 joined trainers
  and diverged from c0. Same-seed checks need `minInitialTrainers = n` (`harness_leg.sh` sets it). Bench RoBERTa leg: ~10 GB GPU at 4 workers.
- **V2 fail: 1.7B `fwdllm` is broken on bench and harness alike (not a bench bug).** Cause: `fwdllm_aggregator.py`
  `_prune_departed_from_round_cache` evicts a cohort member after `ROUND_CACHE_STUCK_TIMEOUT_S` = 300 s (hardcoded) without an accepted
  contribution. In sim that age is vclock, and 1.7B `fwdllm` tasks are charged D ≈ 85–400 s (divisor 0.047), so live clients are evicted
  mid-iteration and re-selected. The bench then over-counts in-use ends (11 > c 10), and `random.py` `sample(…, -1)` raises; the crashed leg
  hangs. The harness doesn't crash but thrashes: `H_sm_lr0.0001` had 162 evictions, 81 iterations and 0 commits in 15 min (stopped, with the `H_sm` queue).
  RoBERTa (D < 300 s) is unaffected. Real runs would hit the same thing (wall D > 300 s). Fix proposed, not applied: timeout = max(300 s,
  `send_timeout_wait_s`; launchers pass `--send-timeout-s 3600` for big models), plus clamp `required_trainers` ≥ 0 in `random.py`.
- **W_rb (c22–36, 04:30): collapse to one class at every lr**: `top_class_share` 0.65–0.83 → 0.91–1.00, acc 0.50, eval loss 0.69 → 0.70–0.76;
  lower lr only collapses more slowly. Cause: 1 selection for the whole run. `fwdllm` pins its cohort per round (`cadence=round`) and every
  commit so far is in round 1, so the same 10 non-IID clients (α 1, 2 classes) set every update. Upstream FwdLLM appears to draw fresh clients each
  comm round (confirmed: `FedAvgServerManager_var.py` resamples via `client_sampling(round_idx)` after each var-passing update ≈ our
  `reselect_cadence: data_bin`). The DistilBERT `fwdllm` baseline uses the same cadence.
  Cohort = clients 0, 1, 2, 3, 5, 10, 11, 12, 14, 15: 58.4% class 1 overall (the other 10 clients: 42.1%); every bin 0–35 is 54–64% class 1
  (no shuffle in the local loader). The model drifts to **class 1**: predicted class-1 share 0.735 → 0.996 while true is ~0.50; MCC ~0,
  entropy 0.690 → 0.671 (≈ ln 2), logit norm 0.15 → 0.33. So it learns the cohort's prior, not features.
  var 0.1 → 27 iterations / commit (208 s); 0.2 → 13; 0.4 → 8.
- **W_rb stopped 04:55; rerun as `D_rb_*` (10-25 04:57, bench GPUs 4–7, tmux `dg_flame:fwddb`, `bench_databin/`, `pool_databin.log`).**
  Only change: `reselect_cadence: data_bin` (a new cohort per commit; configs `yaml_databin/`). Checked at c1–4: selections = commits + 1, and 19 of 20
  clients used by c4 (the pinned run used 10 in 36 commits).
- **D_rb at 05:05 (c4–12, 13 min):** acc 0.51–0.57, MCC +0.05–0.14, entropy ~0.69, eval loss flat at 0.686–0.699. The class drift is gone
  (lr 0.01 top-class share 0.73 → 0.59 → 0.55, vs 0.73 → 0.80 → 0.90 pinned). No lr or var chosen yet: read at c50.
- **D_rb 05:50 (operator):** stopped the 2 weakest at 58 min: lr 3e-2 (c26, loss 0.688–0.699, the noisiest) and var 0.1 (c14, 3× the cost per commit,
  loss 0.687–0.693). Kept lr 1e-3 / 3e-3 / 1e-2 at var 0.2, and lr 1e-2 at var 0.4.
- **Why RoBERTa `fwdllm` is flat at c35–57 (05:55, from `server_update`):** BitFit p = **273,410** at run time (`[ProbeDim]`; the
  initializer's `trainable_p=1323010` is logged before the trainer drops `classifier.dense`) = biases 271,360 + `out_proj` 2,050, the same
  as FwdLLM upstream, and below LoRA's 591,874. One commit pools k ≈ 130 client probes, so cos(estimate, true gradient) ≈ √(k/p) ≈ 0.022.
  ρ per commit = 0.004 / 0.012 / 0.038 / 0.11 at lr 1e-3 / 3e-3 / 1e-2 / 3e-2 → s = ρ/√(k/p) ≈ 0.18 / 0.54 / **1.76** / 5.0. lr 1e-2 (FwdLLM's value)
  sits at FluxTune's s 1.75; 3e-3 and 1e-3 under-step. Progress Λ = Σρ²/s ≈ s·k/p per commit ≈ 8e-4, so lr 1e-2 is at Λ ≈ 0.05 by c56;
  FluxTune's big-cell Λ_need is ~1.26, i.e. ~1,500 `fwdllm` commits (FluxTune pools k ≈ 500, about 4× more per commit). Flat at c56 is expected.
  The var gate checks a 1,024-dim slice and falls ~1/k with pooled probes, so it sets k (var 0.1 → 280, 0.2 → 130, 0.4 → 80), not direction quality;
  Λ per unit of compute is about the same across var values. Control `U_rb_a100_lr0.01_v0.2`
  (near-IID `niid_label_clients=100_alpha=100`, since `uniform_client_1000` fails `check_partitions.py`: 191,297 duplicate rows), bench GPU 4,
  started 05:53 (`bench_uniform/`, tmux `dg_flame:fwdunif`).
- **1.7B `fwdllm` var gate never passes at 0.2.** V2b: 0 commits after 287 iterations on data_id 0 (no evictions, so the fix holds); var
  113 → 8.4 (iteration 13) → 4.6 (27) → 0.45 (287), plateauing above 0.2, and `max_iters_per_data_id` is unset. RoBERTa's var is ~45× smaller (2.8 → 0.18 by iteration 13).
  Matched by iterations per commit to RoBERTa's 0.4 / 0.2 / 0.1 (8 / 13 / 27 iterations), the 1.7B grid is var {15, 8.4, 4.6}.
- **1.7B sweep `D_sm_*` (10-25 05:12, `data_bin`, n 10, 100 commits / 2 h):** lr {1e-4, 3e-4, 1e-3, 3e-3} at var 8.4, plus var 4.6 and 15 at lr 1e-3. Harness
  queue on GPU pairs (`queue_sm_databin.log`, tmux `dg_flame:fwdsmq`); bench twin of lr 1e-3 / var 8.4 on GPU 7 (`bench_sm_databin/`, tmux
  `dg_flame:fwdsmb`) = the 1.7B bench ≡ harness check (V2c). V2b stopped (it had answered the eviction question).
  05:46 (operator): shortened; no 2 h legs. lr 3e-4 stopped at c13 (loss 2.55 vs 1e-3's 2.38 at c9, on the same eval rows), lr 1e-4 dropped, queue
  stopped. lr 3e-3 started on GPUs 2–3 (`queue_sm_lr3e-3.log`, tmux `dg_flame:fwdsm3`), compared with 1e-3 at ~c15. Then var 4.6 / 15 at the
  winner for ~15 commits each. lr 1e-3 at c13: loss 2.72 → 2.24, acc 0.115 → 0.199. V2c twin matches the harness through c5.
- **1.7B lr (06:15, operator): use 3e-3 for now.** Every lr leg ran at var 8.4. Same eval rows at each commit: c13 loss / acc 2.178 / 0.305 (3e-3) vs 2.236 / 0.199 (1e-3); 3e-4 stopped
  behind. **Later tuning:** 3e-3 is the grid's top value, so the upper bracket is untested. Run lr 1e-2 (then 3e-2 if better) at var 8.4, ~15 commits
  each, and compare at matched commits. The ρ/s step rule doesn't transfer to 1.7B LoRA (ρ 0.16–0.31 per commit, s 13–27, yet the largest
  learns fastest; LoRA B starts at 0, so ρ is inflated early). Pick from eval curves.
  1.7B var: 8.4 is a placeholder matched to RoBERTa's iterations per commit. 4.6 / 15 not run yet.
  Update 06:40: the deferred lr 1e-2 leg had in fact started at 06:18 (the swap command ran although it was reported as rejected). It overshoots: c1 eval
  loss 4.10 vs 2.57 (3e-3) / 2.72 (1e-3); 5 commits in 22 min. Stopped. **lr 3e-3 confirmed** (upper bracket closed). lr 1e-3 stopped at c32 (2.097 / 0.268).
- **06:35 (operator): stopped what ran long without a useful trend:** RoBERTa BitFit lr 1e-3 (c70) and 3e-3 (c63), flat; the near-IID BitFit control (c71,
  answered: no better per commit than α 1). Kept BitFit lr 1e-2 at var 0.2 / 0.4 and LoRA α 1. Started **LoRA r 6 on near-IID** (α 100, lr 2.5e-3, var 0.2,
  bench GPU 4, `bench_lora_a100/`, tmux `dg_flame:fwdulora`). A LoRA α 1 lr grid runs only if LoRA beats BitFit (LoRA α 1 at c17: 0.685 / 0.563,
  same as BitFit so far).
- **Bench parity left (FT-N18) before big cells use the bench:**
  1. ~~1.7B `fwdllm` bench ≡ harness~~ **V2c pass (06:10):** 12 / 12 commits identical (iterations, update norm, var at commit) and 3 / 3 evals,
     under `data_bin`; twin stopped at c12. The bench is 2.4× slower for 1.7B here (290 vs 119 s/commit at 4 workers on a shared GPU), so 1.7B
     `fwdllm` gains nothing on the bench yet. Bench 1.7B leg ~27 GB GPU.
  2. RoBERTa `fwdllm` under `data_bin`: V1 passed under `round`; the `data_bin` path is model-independent and passed on 1.7B (V2c), so it's covered
     unless a RoBERTa-specific path differs (optional same-seed pair).
  3. `fluxtune` on RoBERTa-L and 1.7B: never run on the bench (async pool N 50, n 100 / 75); one same-seed pair each.
  4. `bench_profiles.yaml` entries: `yelp-p` (~10 GB GPU at 4 workers, measured), `yahoo` (~16–18 GB at 4 workers; confirm from V2c).
  5. Re-run B2 on DistilBERT `fwdllm` / `fluxtune` (1 seed) after the shared `fwdllm_aggregator.py` / `random.py` fix (rule: re-run B2 on
     touched cells after any shared change; the fix should be inert there, D < 300 s).
  6. Same result at 2 and 4 workers (B2 gate item; not checked on big models).
  Separate from bench parity and still open: real ↔ sim for big models (FT-N21).
- **Fix applied 10-25 (jayne + shepherd, uncommitted):** stuck timeout = max(300 s, `send_timeout_wait_s`); `random.py` clamps
  `required_trainers` ≥ 0; regression test `test_send_timeout_extends_stuck_window` (444 pass). Recheck V2b (1.7B lr 1e-3 n 10, bench GPU 0 vs
  harness GPUs 2–3, tmux `dg_flame:fwdv2b`) running.

## 4. Run-time basis

Under `exp`, Φ is bounded by exp(ρ0²·T_res/2): ≈ 1.96 on DistilBERT, so its Φ rail never fires and the ~1,000-commit cap
ends it (6.7 s/commit → ~1.9 h); 4 on RoBERTa and 1.7B, where the 12 h wall cap ends the run long before the Φ = 3 rail (§6). FwdLLM ran 9.3 s/commit
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

- **SmolLM2-1.7B · Yahoo, N 50, bs 32:** measured 43 s at N 50, bs 8 (10-20, §3.4) × 2.66 ≈ 114 s/commit (est.) →
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
| Gating (G0, P0, G1, P0b bs, S, `fwdllm` lr screens) | ~18 | ~40 + G0 |
| P1 figure runs (1–10) | 10 | ~66 (6 × 3 h + 4 × 12 h) |
| P2 figure runs | 8 | ~24 |
| P3 figure runs (23–25 need FT-N5) | 9 | ~27 |
| **Figure total** | 27 | ~117 |
| C5 (deferred) | 9–11 | ~30–35 DistilBERT + 12 for 1.7B |

## 6. Open (closed by data, not by decision)

- RoBERTa s/commit and t_eval, and both charge profiles: S records them, then the §2 and §4 estimates are replaced.
  1.7B · Yahoo s/commit and t_eval are measured at bs 8 (§3.4); bs 32 is scaled.
- N 50 and ρ0 0.07 on DistilBERT: shown by `tune_commit_rule` and `tune_rho0` (§3.2).
- **12 h caps vs s 1.75 / Φ_budget 4 (big cells).** Φ = 3 needs ~1,685 commits (RoBERTa-L) / ~2,300 (1.7B); at ~117 / ~99 s/commit
  (est. / 1.7B bs 32 measured, `run_20261023_220053`) that is ~55 / ~63 h. In 12 h they reach Φ ≈ 1.5 / 1.4 (Λ ≈ 0.46 / 0.41 vs
  Λ\* 1.26 at Φ 3). Operator decides: longer caps, or accept a capped run.
- 12 h-cell bs (F2), 1.7B n (75 or 60), `fwdllm` lr: closed by S and the lr screens.

## 7. Preview runs (0.60 target only)

Reference runs stopped at 0.60, not at peak; they feed no paper plot. Final runs go to peak accuracy (§3.2). Paths relative to `lib/python/examples/fwdllm/`.

| ID | Outcome | Detail |
|---|---|---|
| ag60 preview (10-22, jayne, 2 lanes, 07:24 → 10:34) | 22 of 23 DistilBERT · AG News arms reached 0.60; A3 (FwdLLM + fixed pool) diverged, stopped by hand. TTA (vclock h): FluxTune 0.51 vs FwdLLM 2.27 (4.5×); A1 0.31, A12 0.67, A123 1.95, A2 5.9; N 10/20/50/90 0.23/0.37/0.51/0.71. Not scored (1 seed, target 0.60, window 1) | script `fwdllm/expt_scripts/nodes/run_node_agnews_to60_queue.sh`; arm → run dir `fwdllm/ag60_20261022_{072444,073816,074438}/runs.tsv`; figures `fwdllm/ag60_20261022_074438/paper_figs/` (script `fwdllm/expt_scripts/plot_ag60.py`); fixed on the way: mean-of-10 on FwdLLM (A1), `exp` without pool (A2), converge watcher after a lap wrap. Figure manifests now at the cell targets (0.86 / 0.82 / 0.65, `tta_consecutive` 2) |
| rb60 (10-23, jayne, 16:06 → 20:53) | `fwdllm` lr 0.01 at chance (0.50–0.55 over 365 commits; lr screen needed); stopped on its 40× vclock cap at 4.7 h of 12 (cap now 1e9); FluxTune arm never started (queue waited on GPUs) | `fwdllm/rb60_20261023_160630/runs.tsv`, `fwdllm/rb60_queue_20261023_1606.log` |

| Run(s) | What it was | Why retired | Where |
|---|---|---|---|
| ag60 preview, all 23 arms (10-22) | DistilBERT · AG News, stopped at 0.60 | preview target only; every surviving arm is rerun to peak (A0 → 1, A6 → 2, A123 → 10, A4 → 11, C4fx → 12, C4fb → 13, R2 → 15, CN10/20/90 → 17/18/19, Cnt → 20, Spool → 21, Smom → 22; R1 → 14) | `ag60_20261022_{072444,073816,074438}/runs.tsv`; figures `ag60_20261022_074438/paper_figs/` |
| ag60 A1, A2, A3 (singles), A12 (ladder), A13, A23 (pairs) | FwdLLM + one or two concepts | add-one ablation replaced by leave-one-out (`Contri_*`) | as above |
| ag60 Srm, Wnew, Wga | `rm` schedule; staleness ω; grad-aware ω | `rm` repeats what constant vs `exp` shows; ω arms within noise | as above |
| rb60 `fwdllm` (10-23) | RoBERTa-L · Yelp-P, lr 0.01 | at chance; lr screen replaces it | `rb60_20261023_160630/runs.tsv` (target 0.60) |
