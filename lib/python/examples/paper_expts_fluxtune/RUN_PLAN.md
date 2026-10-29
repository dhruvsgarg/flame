# FluxTune paper: plots and run list

Every plot and every run the paper needs, derived from [EXPERIMENTS.md](EXPERIMENTS.md) (the source of truth for
the recipe, queue and scoring; task cost D per model: [D_CALIBRATION.md](D_CALIBRATION.md)). Run times are from measured run speeds (§4); **(est.)** marks an extrapolation.
**Status:** `ready` = runnable now · `code` = needs code first · `decide` = optional, operator decides per step.

**Rules (every agent, every run):**
1. When a run starts or finishes, update its status on the board below, with its node, config file, and log file paths.
2. When a plot is made, add its path (and the script that made it) next to the run(s) it shows.

## Status board (keep current: update on every launch, finish and decision)

Last updated 2026-10-29 11:50. Paths relative to `lib/python/examples/fwdllm/`; run dirs under `experiments/`.

**Running**

**SmolLM2-1.7B · Yahoo near-IID (`alpha_100`, decision 18), `fluxtune` + `fwdllm`, LoRA r 2: relaunched 10-29 11:50, pgid 2967650 (kill -TERM -- -2967650).** Dir `paper_su_20261029/`
(`launch.sh`, `jobs.tsv`, two yamls, `bench.log`, `POOL/`); jayne bench, 4 GPUs per leg: `fwdllm` GPUs 0–3 (20 workers; GPU 0 shares 8.5 GB with another user), `fluxtune` GPUs 4–7 (24 workers); stop 0.65 ×2 or 15 h timeout (54,000 s, kill, no final eval); `sim_wall_ceiling_s` now 1e9 (off). Try 1 (05:48 → 07:49) hit the 2 h default ceiling: final full-test `fluxtune` 0.434, `fwdllm` 0.419 → `try1_2h_ceiling/`.
`fluxtune`: the run 6c recipe (ρ0 0.0978, T_res 290, no momentum, n 60, C 30 / K 10). `fwdllm`: lr 3e-3, var 8.4, `data_bin`, divisor 0.047, n 60, C = K = 10;
its charge profile is the agnews `fwdllm.yaml` (preflight `--force`; vclock mispriced as in run 5).

**DistilBERT · Yahoo (4th cell, operator 10-27): runs 28 / 29, one queued pipeline on jayne; **finished 10-27 13:57** (`pipeline done`).**
Dir `paper_dy_20261027/`: `pipeline.sh` (driver), `pick_screen.py`, `pipeline.log` (stage log). Stages, in order:

| Stage | What | Stop | Output |
|---|---|---|---|
| `real_fwdllm`, `real_flux` | 30-min real charge legs (n 30, bench-free, own broker) | 30 min each | `experiments/run_*_dy_{fwd,flux}_real_*_real` → `sim_charge_profiles/{fwdllm,fluxtune}_yahoo_distilbert.yaml` (model-suffixed; SmolLM's `fluxtune_yahoo.yaml` untouched) |
| `screen` | 2 `fwdllm` sim legs in parallel, 1 GPU each: A lr 0.01 / var 0.3, B lr 0.003 / var 0.3 | fixed 1.5 h wall, no target stop | `POOL_screen/`; winner (best acc by the common vclock) in `winner.txt` |
| `full_fwdllm` → `full_flux` | paper legs (run 28, 29), bench, 2 GPUs each | **only 0.65 ×2**, then full-test eval; no commit cap, wall ceiling 720 h (the code default is 2 h) | `POOL_full_fwdllm/`, `POOL_full_flux/` |

**Run 28 (`fwdllm`, lr 0.01 / var 0.3, screen winner scrA 52.0% vs scrB 45.0%): DNF, stopped by hand (SIGTERM, no final eval) 10-27 11:28** after 5.5 h wall / ~46.7 vclock h, 243 evals: peak **0.616**, last 0.568; plateau 0.57–0.60 since ~07:00 (1.1 h in), loss flat ~1.28 from 08:00 then drifting up to ~1.31. Pipeline logged `exit=1` (rc −15) and marked the stage done. `POOL_full_fwdllm/P_fwdllm/runs/syn_0_fwdllm_s1/run_20261027_055652_P_fwdllm_fwdllm_yahoo_n100_smoke_syn_0_sim`.  
**Run 29 (`fluxtune`): DONE 10-27 13:57** — see Done table.

*Then:* add both run dirs to `figs_e2e.yaml` (`distilbert_yahoo`) — todo.

**Running (shepherd, 10-29): cos-audit / pitch runs, not in the figure plan.** `paper_pitch_20261029a/` (q4 `fwdllm`, GPUs 2,3, since 00:18, 6 h timeout) and `paper_pitch_20261029b/` (q1 `pool`, q2 `exp` T_res 566, q3 `exp` T_res 849, GPUs 0,1,4,5,6,7, since 02:08; q5a/b queued). Cos audit on (every 25–50 commits) adds a ~440–500 s `[SIM_OVERHEAD]` pause per audit: slow, not stalled (checked 03:30: telemetry current, c328–373).

The 10-24 RoBERTa S `fluxtune` run (`rbflux_20261024/`) never launched (dry run only, no run dir); its recipe (ρ0 0.14) is superseded by decision 15.

**Done**

| ID | Outcome | Detail |
|---|---|---|
| AG News ablation + strategy queue (10-28 18:30 → 10-29 02:06, shepherd bench, `expt_scripts/queue_ablations_bench.sh`, 2 GPUs / leg, 4 h timeout) | Run ids = `aNN`. **Reached 0.86 ×2:** 8 (c501, 46 min wall, 4.15 vclock h), 10 (c377, 41 min, 32.6 h), 14 (c501, 47 min, 4.24 h), 19 (c369, 58 min, 4.93 h), 20 (c745, 50 min, 4.19 h), 21 (c365, 33 min, 2.69 h). **DNF:** 17 N 10 (cap 1000, peak 0.8355), 18 N 20 (cap 1000, peak 0.8593), 26 max-\|JVP\| (cap 1000, peak 0.8514), 22 momentum 0.8 (diverged to 0.25), baselines 11 / 12 / 13 (saturation stop, peaks 0.857 / 0.8555 / 0.8491, final 0.8553 / 0.8464 / 0.8295). **Stopped by hand 20:41** (first queue, no final eval): 15 (peak 0.8583), 16 (peak 0.7924). Full per-run numbers: §3.2 | first queue `paper_ab_20261028/` (a08, a10, a15, a16; `queue.log`, `bench.log`, `autostop.log`); second queue `paper_ab_20261028b/` (the rest; `queue.log`, `bench.log`); both under `lib/python/examples/fwdllm/` on **shepherd**, runs under `POOL/<arm>/runs/` |
| P1 run 29: DistilBERT · Yahoo `fluxtune` (10-27 11:29 → 13:57, jayne bench, 8 workers) | **Target reached: 0.65 ×2 at c557** (`[BudgetStop] reason=target`, clean exit), 2.5 h wall (8,841 s), 17,200 vclock s (4.8 h), 27,979 tasks, Φ 1.82. 140 evals, peak **0.662**, **final full-test eval 0.652 / loss 1.341**. Run 28 `fwdllm` DNF (peak 0.616 at 46.7 vclock h) → FluxTune reaches 0.65 where FwdLLM never does | run dir `paper_dy_20261027/POOL_full_flux/P_fluxtune/runs/syn_0_fluxtune_s1/run_20261027_112923_P_fluxtune_fluxtune_yahoo_n100_smoke_syn_0_sim` (`telemetry/aggregator_*.jsonl`, `bench.log`, `bench_summary.json`); `paper_dy_20261027/{pipeline.log,bench_full_flux.log,P_fluxtune.yaml}` |
| P1 run 4b: RoBERTa-L · Yelp-P `fluxtune`, LoRA r 6, no momentum (decision 16; 10-26 02:55 → 08:56, jayne bench GPUs 4–7, 30 workers) | **Target reached: 0.82 ×2 at c401** (`[BudgetStop] reason=target`, clean exit), 404 commits, **6.0 h wall (364 min), 97.9 vclock h** (16×), 20,229 tasks. Peak 0.834, last periodic eval 0.833, **final full-test eval 0.8264 / loss 0.408**. Φ 2.02 (budget 4 not used up); trainable weight norm 9.9 → 19.9 (2.0×), no divergence. First P3 paper point | run dir `paper_rb_20261026b/POOL/rb_paper/runs/syn_0_fluxtune_s1/run_20261026_025521_fluxtune_yelp-p_n100_smoke_syn_0_sim` (`telemetry/aggregator_*.jsonl`, `bench.log`, `bench_summary.json`); `paper_rb_20261026b/bench.log`; config `paper_rb_20261026b/fluxtune_yelp-p_n100_smoke_syn_0_sim.yaml`; launcher `paper_rb_20261026b/launch.sh` |
| P1 run 6c: SmolLM2-1.7B · Yahoo `fluxtune`, r 2, T_res 290, no momentum (decision 17; 10-26 04:07 → 10-27 02:25, jayne bench GPUs 0–3, 24 workers) | **DNF, stopped by hand (SIGTERM, no final eval) at c~450 / 22.3 h wall**: peak **0.606**, acc 0.597 (c448), plateau 0.59–0.60 since c~350 (+0.5 pt / 100 commits). Lap wrap (c~439) crossed with no collapse. Φ 2.89 of cap 4 = ‖θ‖ growth 6.4 → 18.4 (matches a pure noise random walk, cos 0.035); Λ 0.755 / 0.987 → schedule ceiling ≈ 0.62, so 0.65 unreachable. Next: forward-gradient emulator screens (‖θ₀‖-scaled step, momentum, r 1) before another FL run | `paper_sm_20261026b/POOL/sm_paper/runs/syn_0_fluxtune_s1/run_20261026_040708_fluxtune_yahoo_n60_smoke_syn_0_sim`; `paper_sm_20261026b/bench.log`; config `paper_sm_20261026b/fluxtune_yahoo_n60_smoke_syn_0_sim.yaml` |
| P1 run 6b: SmolLM2-1.7B `fluxtune` r 2 + momentum 0.9 (10-26 01:50 → 04:06, jayne bench) | **Stopped by hand at c40: diverging.** Peak acc 0.331 (c28), best loss 2.144 (c22); by c40 acc 0.267, loss 4.155, top-class share 0.52; trainable weight norm 6.4 → 40.0 (6.3×). Same failure as run 4 | `paper_sm_20261026/` (config, `bench.log`, POOL) |
| P1 run 4: RoBERTa-L `fluxtune` + momentum 0.9 (10-26 01:50 → 02:44, jayne bench) | **Stopped by hand at c50: diverging.** Acc 0.551 (c1) → 0.473 (c49), one-class evals (top share 0.99–1.00), loss 0.69 → 0.71–0.79; trainable weight norm 9.9 → 37.0 (step 0.69 → 2.40), vs 13.9 at c49 without momentum (10-24). Logged Φ 1.12 is not ‖θ‖ growth under momentum (FT-T16) → decision 16 | `paper_rb_20261026/` (config, `bench.log`, POOL) |
| §3.7 smokes (10-25, jayne bench, n 12) | **Stopped by hand 10-26** (operator; 01:27 `sm_base`, `sm_k40s`; 01:49 `sm_r2s`, `sm_r2m`). Best acc: `sm_r2m` **0.432** (c28; stopped c31), `sm_r2s` 0.357 (c43; c51), `sm_base` 0.328 (c42; c55), `sm_k40s` 0.288 (c12). `sm_r2m` led at matched commits, vclock and wall → decision 15 | `bench_sm_20261025/POOL/sm_base/`, `POOL_s/sm_{k40s,r2s}/`, `POOL_m/sm_r2m/`; `bench{3,4,5}.log`; §3.7 |
| P1 run 6: SmolLM2-1.7B · Yahoo `fluxtune` (10-25, jayne, 07:57 → ~20:02, 8 GPUs) | **Stopped by hand at c398 / 12.1 h**: peak **0.576**; loss still falling ~0.09 per 100 commits (not plateaued), heading to the ~0.59 plateau of the 10-23 run, so 0.65 is out of reach at cos 0.025; backprop ceiling ~0.77 (r=4, `fwdllm/bp_ceiling_20261025/`). Led to the §3.7 smokes | `run_20261025_075720_fluxtune_yahoo_n60_smoke_syn_0_sim`; `smollm17b_yahoo_n50_fluxtune_20261025_0756{.log,_watch.log,_acc.log}` |
| P1 runs 1 + 2: DistilBERT · AG News (10-25, shepherd bench, 4 GPUs per leg, 12:05 → 13:55) | **FluxTune 0.86 ×2 at c525 / 4.34 vclock h**, peak 0.873 (c1,097), stopped by hand 13:23 at c1,193 (plateau). **FwdLLM DNF**: peak 0.855 (c333), plateau 0.82–0.85, Φ 4.31 rising, stopped by hand 13:55 at c493 / 27.0 vclock h. Manual stops; full-test evals every 4 commits, so the last eval is final. Decision 14 caps. **Paper: 4.8× to FwdLLM's peak, ≥ 6.2× at 0.86, +1.8 pt peak** | §3.6 (files, logs, paper points, plot) |
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
| Big-model real↔sim check (FLUXTUNE FT-N21) | 1–2 h real leg + same-seed sim leg per model, paper packing | todo; confirms DistilBERT's parity transfers (0 overruns, accuracy within band at matched commits) |
| `fluxtune_yelp-p.yaml` charge profile | 30-min real RoBERTa-L leg (decision 12 method) | todo; charges ≪ D, so P3's vclock is close |

**Stop and eval flags (in code 10-22):** `--max-commits N` (halts, `stop_reason` `commit_cap`; scales only the saturation-stop warm-up
450 → 0.45·N, so a huge cap is inert when that stop is off, as on the big cells); `--target-stop A --target-stop-window W` (clean halt); `--eval-resample` (fresh
`--eval-max-samples` draw per eval, seeded by commit); `--final-full-eval` (one full-test eval at run end, event `agg_eval_final`).

**Operator decisions (closed 10-22)**
1. P1 `fwdllm` runs to the 3 h wall cap (`--max-runtime-s` ≥ 40 × wall cap), not equal vclock.
2. `B_max origin` gate applies only when BmaxProbe is on; A0 and `pool` are not invalidated (code fix: EXPERIMENTS §8).
3. DistilBERT eval stride 4.
4. No iteration-cap arm: pool size is `tune_commit_rule` (N 10 / 20 / 90 + `n_target`).
5. Big cells: bs 32 (F2 closed by decision 15).
6. DistilBERT figure runs at N 50; N 90 is run 19 (`tune_commit_rule`).
7. Superseded by 15 (was s 1.75, Φ_budget 4 on both big cells). DistilBERT stays ρ0 0.07.
8. Superseded by 15 (was: 12 h cells on the harness).
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
    for every future run; G0 is closed (no sim grade needed).** (Its 1.7B s 2.0 is superseded by 15.)
13. (10-25 07:50) Big-cell stop rule: stop once acc ≥ target on 2 consecutive evals (`--target-stop`, in-process; clean exit, so
    `--final-full-eval` scores the full test set, `agg_eval_final`), not `--target-acc` (watcher kill, no final eval). Caps: decision 15.
14. (10-25 12:00) DistilBERT · AG News cell: **6 h wall cap and `--max-commits 2000` in both arms** (was 3 h / 1,000; FwdLLM at ~10 s/commit
    hit 1,000 commits at ~2.8 h); `fwdllm` var 0.3 (the `baselines.yaml` value A0 ran, not upstream's 0.1); `fwdllm` charge profile kept
    (`fwdllm.yaml`, 08-05; charges ≪ D); B2 re-check after the 10-25 shared fix skipped (fix inert at D < 300 s). Runs 1–2 on the bench,
    4 GPUs and 8 workers per leg (1 GPU per leg ran FluxTune ~12 s/commit vs 6.7 on the 4-GPU harness).
15. (10-26 01:45, operator) Big-cell `fluxtune` runs move to the bench (parity passed for both models), one leg per 4 GPUs on jayne, and add
    server momentum 0.9 (§3.7: `sm_r2m` led every matched point). **RoBERTa-L:** LoRA r 6, s 2.4 (ρ0 0.0698), T_res 570 (Φ_budget 4).
    **1.7B:** LoRA r 2 (p 413,696), ρ0 0.0978 (s 2.81), T_res 1,119. Both: stop only at target ×2 (`--max-commits 100000`, `--phi-stop-threshold 1e9`,
    no saturation stop), `--final-full-eval`, cos audit off; strides 4 / 3 (eval adds ~10% wall, 0 vclock). The bench ignores `sim_wall_ceiling_s`:
    `bench_pool.py --timeout-s 109800` (30.5 h) is the wall cap and a kill (no final eval). Workers 30 (RoBERTa) / 24 (1.7B).
17. (10-26 04:07, operator) 1.7B drops momentum too (run 6b diverged at c40); r 2, ρ0 0.0978 (s 2.81) kept; T_res 1,119 → 290 so
    Φ_budget = 4 (without momentum, 1,119 leaves ρ near constant with the rail off).
16. (10-26 02:50, operator) RoBERTa-L drops server momentum (run 4 diverged: norm 3.7× in 50 commits); s 2.4, T_res 570 kept (Λ max 1.16
    vs 0.29 at the 10-24 ρ0 0.14 / T_res 71, which plateaued at 0.732). 1.7B keeps momentum 0.9; watch its weight norm.
18. (10-29, operator) "Uniform" partition runs use near-IID `niid_label_clients=100_alpha=100`: Yahoo's `uniform_client_1000` / `_10000` fail
    `check_partitions.py` (515k duplicate train rows, test/train overlap); `100_alpha=1` and `100_alpha=100` pass. Both arms LoRA r 2
    (`fwdllm` matches `fluxtune`; was r 4). Replaces the α=1 `fwdllm` n50 harness leg (stopped by hand 10-29 at c108 / 10.6 h: acc 0.49, best 0.523).

Closed by data, not by decision: grad-aware (G1), bs (P0b → 32, subject to F2), Yelp-P/Yahoo eval stride (S), v1 profile (G0).

**Then the figure runs** in §3.2 priority order (P1 → P2 → P3); 23–25 after FT-N5.

**What the runs so far established** (concept table, knobs to tune per cell, what transfers: [EXPERIMENTS §2.2](EXPERIMENTS.md#22-concepts-tried-kept-tuned-general)):
- Kept everywhere: mean of 10 probes, trust ratio with `exp` decline, fixed pool N 50, uniform ω, async C = 0.3·n / K = C/3, cos audit off.
- Set per cell: s (ρ0 = s·√(P·N/p)), Φ_budget (T_res = 2·lnΦ_budget/ρ0²), LoRA r, momentum, bs, eval rows / stride, D divisor.
- Dropped: more probes per client (compute-neutral), constant ρ (C3 arm), `grad_aware` ω (C3 arm), fixed ρ0 across models.
- Open: does P3 (s 2.4, no momentum) reach target at full n; P4 run 6c (r 2, s 2.81, no momentum, T_res 290 → Φ_budget 4).
- Momentum 0.9 diverged on every cell at full n: RoBERTa-L (norm 3.7× by c50), 1.7B (6.3× by c40, peak 0.331 at c28), DistilBERT (ag60). The n 12 smoke lead ended where it was stopped (c31).

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
| `tune_rho_schedule` | constant ρ vs `exp` vs `exp` + server momentum 0.8; plus ρ and Φ per commit | main | 2, 21, 22 | `c_step_size` + `c_step_size_step_size` (without `rm`) |
| `tune_batch_size` | bs 8 vs 32 (DistilBERT · Yahoo, N 90) | appendix | P0b (§0c), no new run | `fwdllm/scripts/plot_p0b_bs8_vs_bs32.py` |
| `tune_agg_weights` | uniform ω vs alignment-only ω (`grad_aware` without the staleness × utility base) | appendix | 2, 27 | `c_agg_weights` |

**Strategy comparison** (`figs_availability.yaml`; DistilBERT · AG News; `syn_0` unless named)

| Plot | Strategy vs FwdLLM and FluxTune | Place | Runs |
|---|---|---|---|
| `strategy_fwdllm_async` | FwdLLM's estimator, async per iteration (`fedbuff_it_unaware`) | main | 1, 2, 11 |
| `strategy_fedbuff` | FedBuff(P) (`fedbuff_round`) | main | 1, 2, 13 |
| `strategy_felix` | Felix(P) (`felix_round`, availability-aware) | main | 1, 2, 12 |
| `strategy_availability_mobiperf` | FluxTune, Felix(P), FedBuff(P) on `mobiperf_3st_50` | **not for MLSys** (dropped) | 23, 24, 25 |

**Not plotted:** `b2` (diversity), `b3` (progression), `c_phase_a`, `c_phase_a_pairs`, `c_phase_b`, `c_update_utility`,
`c5_generalisation` (deferred, operator 10-22).

## 2. Cells, pool size and eval

**Common to every run:** sim mode, `syn_0` (except C4), Dirichlet α=1, **1 seed per arm**,
`server_update_audit: true`, cos audit off, `tta_consecutive: 2`, one stop rule per cell (DistilBERT: stall 0.003 + GL decay + commit cap
+ wall cap; big cells: target ×2 or 30.5 h wall), async C = 0.3·n, K = C/3.
**FluxTune** = the [EXPERIMENTS.md §2](EXPERIMENTS.md#2-setup) recipe with `--pool-target {N}`. With 1 seed, a gap
under the 2.3-pt replicate noise is unresolved.

| Cell | p | Clients n | Pool N | ρ0 (`--rho0`) | T_res | implied s | Target | Wall cap | Eval sample (rows) | Stride |
|---|---|---|---|---|---|---|---|---|---|---|
| `distilbert_agnews` | 450,340 | 100 | 50 | 0.07 | 283 | 2.10 | 0.86 | 6 h | full test (7,600) | 4 |
| `roberta_large_yelp` | 591,874 | 100 | 50 | 0.0698 (from s) | **570** (Φ_budget 4) | **2.4** (set, decision 15) | 0.82 | 30 h | 10% = 3,800, resampled per eval | 4 |
| `distilbert_yahoo` | 450,340 | 100 | 50 | 0.07 | 283 | 2.10 | 0.65 | 12 h | 10% = 6,000, resampled per eval | 4 |
| `smollm2_1p7b_yahoo` | 413,696 (LoRA r 2) | 60 | 50 | **0.0978** (`--rho0`) | **290** (Φ_budget 4) | **2.81** (decision 15) | 0.65 | 30 h | 10% = 6,000, resampled per eval | 3 |

C = 0.3·n, K = C/3 gives 30 / 10 at n 100; the 1.7B cell runs C 30 / K 10 at n 60 (decision 15). Big-cell bs 32.

**Per-cell launch flags** (on top of the [EXPERIMENTS §2](EXPERIMENTS.md#2-setup) recipe; `{}` = still open):

```
# distilbert_agnews (bs 8, full test set, so no --eval-max-samples)
--dataset agnews --num-trainers 100 --pool-target 50 --rho0 0.07 --t-res 283 --train-batch-size 8
--max-commits 2000 --final-full-eval --eval-every-n-commits 4 --sim-wall-ceiling-h 6   # decision 14
# roberta_large_yelp (fwdllm arm: --learning-rate {lr screen}); fluxtune = decisions 15–16, launcher fwdllm/paper_rb_20261026b/launch.sh
--dataset yelp-p --model-type roberta-large --peft-method lora --lora-r 6 --num-trainers 100 --c 30 --agg-goal 10
--pool-target 50 --gate-safety-s 2.4 --t-res 570 --train-batch-size 32   # no --rho0, no momentum (decision 16)
--target-stop 0.82 --target-stop-window 2 --max-commits 100000 --phi-stop-threshold 1e9 --eval-max-samples 3800 --eval-resample --final-full-eval
--eval-every-n-commits 4 --sim-wall-ceiling-h 30 --delay-divisor 0.0169   # fwdllm arm: 0.169
# smollm2_1p7b_yahoo: fluxtune = decisions 15, 17, launcher fwdllm/paper_sm_20261026b/launch.sh (fwdllm arm: launch_smollm17b_yahoo_n50.sh, r 4)
--dataset yahoo --model-type llama --model-name HuggingFaceTB/SmolLM2-1.7B --peft-method lora --lora-r 2 --num-trainers 60 --c 30 --agg-goal 10
--pool-target 50 --rho0 0.0978 --t-res 290 --train-batch-size 32   # decision 17
--target-stop 0.65 --target-stop-window 2 --max-commits 100000 --phi-stop-threshold 1e9 --eval-max-samples 6000 --eval-resample --final-full-eval
--eval-every-n-commits 3 --sim-wall-ceiling-h 30 --delay-divisor 0.0047   # fwdllm arm: 0.047
# fwdllm arm on 1.7B: --learning-rate {lr screen}
# every cell: --no-cos-ground-truth-audit
# every cell: --max-runtime-s 1000000000 --force, so the wall ceiling, not the vclock, binds (D_CALIBRATION §6)
```

How to set the new flags:
- **`--max-commits`** (DistilBERT) = ⌊wall cap / measured s/commit⌋, so the cap and the wall cap land together. It also sets the
  stall/GL warm-up (0.45 × cap) and horizon (0.15 × cap); at the default 450 they never arm on RoBERTa or 1.7B.
  DistilBERT: 2,000 (decision 14). Same value in both arms of a cell.
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
  `fwdllm` has only the AG News profile (`fwdllm.yaml`, 08-05); big-cell `fwdllm` runs use it (operator 10-25, `--force`). Slight
  error: charges are ~0.1–1 s per agg op vs D 85–400 s per task. Adjust if needed by re-timing from the run's `vclock_charge` events.
- **Caps.** Sim: `--max-runtime-s 1000000000` plus `--sim-wall-ceiling-{s,h}` and `--force` (the wall-budget preflight
  expects the vclock cap to bind); the ceiling ends the leg. 40 × wall cap ended rb60 at 4.7 h of 12. Real:
  `--max-runtime-s` is wall from aggregator start, join included (~3.5 min at n 100).
- **Real legs are slow by design.** At recipe D a RoBERTa-L iteration takes 4–46 min, 1.7B 14 min–2.8 h; first commit ≥ 1 h; a
  check of accuracy needs ≥ 2 h. Compare real and sim at matched commits, never matched wall.
- **Side by side.** Real and sim legs in parallel run as lanes (own GPUs, cores, mosquitto, `FLAME_RUN_LABEL`), as in
  `fwdllm/ft_models_check_20261024/run.sh`. 1.7B at bs 32: ≤ 4 trainers per GPU on a lane that holds the aggregator.
- **After a real smoke,** move its run dir out of `experiments/`, or the profile-staleness preflight blocks the next
  `fluxtune` sim.
- **Bench** (`fl_bench.py` / `bench_pool.py`) is sim only and stands for the harness only on cells that passed a bench ≡ harness
  check (FLUXTUNE FT-N18): DistilBERT · AG News (B2), RoBERTa-L (`fwdllm` V1, `fluxtune` pair), 1.7B (`fwdllm` V2c, `fluxtune` F_sm).
  The bench ignores `sim_wall_ceiling_s`; `bench_pool.py --timeout-s` is the wall cap and kills (no final eval).

**Pool size N** = client gradients pooled into one server commit (`--pool-target`), not clients n. Rule and derivation:
[cheatsheet §8](../fwdllm/fl_fwd_ft_pipeline_cheatsheet.md#8-choosing-n-ρ0-and-t_res).

- **N 50, every cell.** `N ≥ N* ≈ 29·Λ_need²·(ρ0/0.07)²`, set per dataset · bs; p cancels across models. Measured N\*:
  AG News 34–42, Yahoo bs 32 38 (F0). Client availability sets C (clients in flight), with C ≤ N.
- **RoBERTa-L: s 2.4, Φ_budget 4 → ρ0 0.0698 (derived), T_res 570. 1.7B (r 2): ρ0 0.0978 (s 2.81), T_res 290 (Φ_budget 4).**
  Neither runs momentum (decisions 16–17). Why s and Φ, not ρ0: [EXPERIMENTS §2.1](EXPERIMENTS.md#21-pool-n-step-s--ρ0-and-φ_budget--t_res).
- **DistilBERT: ρ0 0.07, T_res 283 (s 2.10, Φ_budget 2).** `tune_rho0` tests 0.10 / 0.14 / 0.28 (§3.2).
- **First-hours abort rule (RoBERTa-L, 1.7B):** Φ off (1+ρ²)^(t/2) by > 1%, no accuracy above chance by ~c50, or GL decay
  before Φ ≈ 1.5 → lower s to 1.5 (T_res from Φ_budget 4); N stays.

**Eval.** Each eval scores a fresh random sample of the test set: a new draw each eval, seeded by commit index so
a rerun repeats it (`--eval-resample`).
DistilBERT scores the full test set (7,600 rows) every 4th commit, so it needs no resampling. Yelp-P and Yahoo score 10% of
the test set (3,800 / 6,000 rows), which gives the same noise at each target (≈ ±1.2 pt, 95%), below the 2.3-pt replicate noise;
stride 4 / 3. `tta_consecutive: 2` in every cell. Each run ends with one full-test eval (`--final-full-eval`), which
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
| S smoke | RoBERTa-large · Yelp-P (LoRA q,v r=6, seq 256); SmolLM2-1.7B · Yahoo | FluxTune at §2's N | — | — | **closed**: RoBERTa 10-24 (shepherd `run_20261024_191344`, 0.732 at c369, budget-bound); 1.7B → run 6, §3.7 | P3 / P4 recipe (decision 15) |
| F0 Λ readout | DistilBERT · AG News, Yahoo | `expt_scripts/lambda_need.py` on P0 `exp`, G1, P0 `pool`, P0b bs 8/32: Λ at target and at peak − 0.5 pt | 0 | — | **done 10-21** ([cheatsheet §8](../fwdllm/fl_fwd_ft_pipeline_cheatsheet.md#8-choosing-n-ρ0-and-t_res)) | N\* 34–42 (AG News), 38 (Yahoo bs 32), 53 (Yahoo bs 8) → N 50; accuracy vs Λ schedule-independent |
| F2 compute check | RoBERTa-L · Yelp-P; SmolLM2-1.7B · Yahoo | — | 0 | — | **closed**: big cells stop on target with a 30.5 h wall (decision 15), bs 32 | — |
| F3 transfer checks | RoBERTa-L · Yelp-P; SmolLM2-1.7B · Yahoo | no extra run: (a) S's first `trainable_weight_norm` vs 13.35·√(p/450,340) = 15.3 / 17.9; (b) P3/P4 `lambda_need.py`: Λ at milestones vs DistilBERT × √(450,340/p) | 0 | — | (a) after S, (b) after P3/P4 | if either fails, N picks up a model factor; raise N for C5 and the 3-seed reruns |

### 3.2 Figure runs

Run in priority order: **P1** before **P2** before **P3**; within a priority, top to bottom. G0 closed 10-25 (decision 12).
A 4-value tuning sweep runs 3 values at P2 and the 4th at P3 (3 points still plot the trend).
AG News ablation / strategy runs (7, 8, 10–22, 26) queue on the jayne bench with [queue_ablations_bench.sh](../fwdllm/expt_scripts/queue_ablations_bench.sh): 2 GPUs per leg, 4 legs in parallel on 8 GPUs, **4 h bench timeout** (kill, no final eval), stop at **0.86 ×2** (`--target-stop`, then full-test eval), 2,000-commit cap. Order: 8, 10, 15, 16, 17, 19, 21, 14, 18, 20, 22, 26, then 13, 11, 12 last (baselines, never run). Runs 7, 9, 27 are not in the script (yaml override / gate not ready).
**Launch** (when all 8 GPUs are free; from `lib/python/examples/fwdllm/`; output `paper_ab_20261028/` (first queue; second `paper_ab_20261028b/`), queue log `queue.log`, per-leg `START` / `DONE` / `FAIL` lines in `bench.log`, runs under `POOL/<arm>/runs/`; a failed leg does not stop the queue):
```
mkdir -p paper_ab_20261028
OUT=paper_ab_20261028 setsid nohup bash expt_scripts/queue_ablations_bench.sh > paper_ab_20261028/queue.log 2>&1 < /dev/null &
```
Stop: `kill -TERM -- -<pgid>`. Subset: pass arm names (`a08 a15 …`); `DRY=1` renders the yamls only. Arm `aNN` = run NN.
DistilBERT · AG News runs: §2 recipe (N 50, ρ0 0.07, T_res 283, bs 8), 6 h / 2,000-commit cap (decision 14), 1 seed, to peak accuracy. Flags below are the change from that.

| Pri | # | Run | Model · Dataset | Flags (change from the cell recipe) | Time | Status | Plots | Logs (done runs; under `fwdllm/`) | Accuracy reached |
|---|---|---|---|---|---|---|---|---|---|
| P1 | 1 | FwdLLM | DistilBERT · AG News | `fwdllm` arm, lr 0.01, var 0.3, `data_bin` | ≤ 6 h | **done** 10-25: DNF, best 0.855 (Done table) | `E2E_*`; reference in `strategy_*` | `p1_distil_20261025/bench/P1_fwdllm/runs/syn_0_fwdllm_s1/leg.out`, `p1_distil_20261025/pool.log` (shepherd) | peak 0.855 (DNF) |
| P1 | 2 | FluxTune | DistilBERT · AG News | none | ≤ 6 h | **done** 10-25: 0.86 at 4.34 vclock h (Done table) | `E2E_*`, every `Contri_*`, every `tune_*` except `tune_batch_size`, `strategy_*` | `p1_distil_20261025/bench/P1_fluxtune/runs/syn_0_fluxtune_s1/leg.out`, `p1_distil_20261025/pool.log` (shepherd) | peak 0.873 |
| P1 | 7 | FluxTune − E1 | DistilBERT · AG News | drop `--probe-combine mean` (FwdLLM's aligned 1-of-10, P 10) | ≤ 4 h | on `fluxtune` `select` alone gives max-\|JVP\| (= run 26): also set `select_perturbation_using_jvp: false` (yaml override) for the cosine pick | `Contri_probe_mean` |  |  |
| P1 | 8 | FluxTune − E2 | DistilBERT · AG News | drop `--server-step-rule trust_ratio --rho-schedule exp --rho0 --t-res` (SGD, lr 0.01) | ≤ 4 h | **done** 10-28 (shepherd bench): target 0.86 ×2 at c501, 46.0 min wall, 14,948 vclock s (4.15 h), 25,279 tasks; final full-test 0.8589. Raw SGD (lr 0.01) matches the trust-ratio run 2 (c525 / 4.34 h): E2 shows no gain at this cell, check before plotting | `Contri_trust_ratio` | `paper_ab_20261028/bench.log`, `paper_ab_20261028/POOL/a08/runs/syn_0_*_s1/run_*_a08_*/bench.log` | peak 0.8607, final 0.8589 |
| P1 | 9 | FluxTune − E3 | DistilBERT · AG News | `--commit-gate var` with FwdLLM's variance policy, no `--pool-target` | ≤ 4 h | variance gate under async never run | `Contri_fixed_pool`, `tune_commit_rule` |  |  |
| P1 | 10 | FluxTune − S1 (= A123) | DistilBERT · AG News | `fwdllm` sync arm + E1, E2, E3 flags | ≤ 4 h | **done** 10-28: target 0.86 ×2 at c377, 41.0 min wall, 117,295 vclock s (32.6 h), 19,000 tasks; final 0.8589. Sync + E1/E2/E3 flags is 7.5× slower in vclock than run 2 (4.3 h) | `Contri_async` | `paper_ab_20261028/bench.log`, `paper_ab_20261028/POOL/a10/runs/syn_0_*_s1/run_*_a10_*/bench.log` | peak 0.8611, final 0.8589 |
| P1 | 3 | FwdLLM | RoBERTa-L · Yelp-P | `fwdllm` arm, screened lr | 12 h | blocked: S, F2, lr screen | `E2E_*` |  |  |
| P1 | 4 | FluxTune | RoBERTa-L · Yelp-P | decisions 15–16: LoRA r 6, s 2.4 (ρ0 0.0698), `exp` T_res 570, no momentum (0.9 diverged, Done table), n 100 / C 30 / K 10, bs 32; stop at 0.82 ×2 (no commit cap, Φ rail and saturation off), then full-test eval. Bench, 4 GPUs: `bash lib/python/examples/fwdllm/paper_rb_20261026b/launch.sh` | ≤ 30.5 h | **done** 10-26 (4b): target 0.82 ×2 at c401, 6.0 h wall / 97.9 vclock h, final full-test 0.8264 (Done table) | `E2E_*` | `paper_rb_20261026b/bench.log`, `paper_rb_20261026b/POOL/rb_paper/runs/syn_0_fluxtune_s1/run_20261026_025521_*/bench.log` | peak 0.834, final 0.8264 |
| P1 | 5 | FwdLLM | SmolLM2-1.7B · Yahoo | `fwdllm` arm: lr 3e-3, var 8.4 (decision 11), LoRA q,v r 4, n 60, C = K = 10, `data_bin`, divisor 0.047; stop as run 6 (0.65 ×2 or 20 h). Launch `ARM=fwdllm bash lib/python/examples/fwdllm/expt_scripts/launch_smollm17b_yahoo_n50.sh` (8 GPUs, harness) | ≤ 20 h | ready (10-25); expect DNF (never reaches target) at 0.65 (smoke 0.384 at c65, ~107 s/commit) | `E2E_*` |  |  |
| P1 | 6 | FluxTune | SmolLM2-1.7B · Yahoo | run 6 (r 4, s 2.0) peak 0.576; run 6b (r 2 + momentum 0.9) diverged at c40 (Done). **6c (decision 17):** LoRA r 2, ρ0 0.0978 (s 2.81), `exp` T_res 290 (Φ_budget 4), no momentum, n 60 / C 30 / K 10, bs 32; stop at 0.65 ×2, then full-test eval. `bash lib/python/examples/fwdllm/paper_sm_20261026b/launch.sh` | ≤ 30.5 h | **done as DNF** 10-27 (6c): stopped by hand at c~450 / 22.3 h wall, peak 0.606 (Done table). **Backup for the figure if not re-run:** `fwdllm/paper_sm_20261026b/POOL/sm_paper/runs/syn_0_fluxtune_s1/run_20261026_040708_fluxtune_yahoo_n60_smoke_syn_0_sim` (plots from `agg_eval` + vclock; scores DNF) | `E2E_*` | `paper_sm_20261026b/bench.log`, `paper_sm_20261026b/POOL/sm_paper/runs/syn_0_fluxtune_s1/run_20261026_040708_*/bench.log` | peak 0.606 (DNF) |
| P1 | 28 | FwdLLM | DistilBERT · Yahoo | `fwdllm` arm, n 100, C = K = 10, bs 32, `data_bin`, divisor 4.78; lr / var from the 1.5 h screen (0.01 vs 0.003, var 0.3); run until 0.65 ×2. `paper_dy_20261027/pipeline.sh` | until target | **done as DNF** 10-27: stopped by hand 11:28 after 5.5 h wall, peak 0.616, no final eval (Running) | `E2E_*` | `paper_dy_20261027/pipeline.log`, `paper_dy_20261027/POOL_full_fwdllm/P_fwdllm/runs/syn_0_fwdllm_s1/run_20261027_055652_*/bench.log` | peak 0.616 (DNF) |
| P1 | 29 | FluxTune | DistilBERT · Yahoo | §2 recipe (N 50, ρ0 0.07, T_res 283), n 100 / C 30 / K 10, bs 32, divisor 0.478; run until 0.65 ×2. Same pipeline, after 28 | until target | **done** 10-27 (jayne bench, 11:29 → 13:57): target 0.65 ×2 at c557, 2.5 h wall (8,841 s), 17,200 vclock s (4.8 h), 27,979 tasks; final full-test 0.652 (Done table) | `E2E_*` |  |  | `paper_dy_20261027/pipeline.log`, `paper_dy_20261027/POOL_full_flux/P_fluxtune/runs/syn_0_fluxtune_s1/run_20261027_112923_P_fluxtune_fluxtune_yahoo_n100_smoke_syn_0_sim/bench.log` | peak 0.662, final 0.652 |
| P2 | 11 | FwdLLM async (A4) | DistilBERT · AG News | `fedbuff_it_unaware` with FwdLLM's estimator | ≤ 4 h | **done as DNF** 10-29 (`reason=stall`, SatStop at c553, acc 0.8488 vs best 0.8546): 171 min wall, 174,280 vclock s (48.4 h), 221,489 tasks; final 0.8553 | `strategy_fwdllm_async` | `paper_ab_20261028b/bench.log`, `paper_ab_20261028b/POOL/a11/runs/syn_0_*_s1/run_*_a11_*/bench.log` | peak 0.8570, final 0.8553 |
| P2 | 12 | Felix(P) | DistilBERT · AG News | `felix_round`, `baselines.yaml` lr (0.075 placeholder) | ≤ 4 h | **done as DNF** 10-29 (`reason=stall`, SatStop at c573, acc 0.8397 vs best 0.8501): 118 min wall, 107,203 vclock s (29.8 h), 153,930 tasks; final 0.8464 | `strategy_felix` | `paper_ab_20261028b/bench.log`, `paper_ab_20261028b/POOL/a12/runs/syn_0_*_s1/run_*_a12_*/bench.log` | peak 0.8555, final 0.8464 |
| P2 | 13 | FedBuff(P) | DistilBERT · AG News | `fedbuff_round`, `baselines.yaml` lr | ≤ 4 h | **done as DNF** 10-29 (`reason=stall`, SatStop at c637, acc 0.8212 vs best 0.8367): 234 min wall, 243,761 vclock s (67.7 h), 303,910 tasks; final 0.8295 | `strategy_fedbuff` | `paper_ab_20261028b/bench.log`, `paper_ab_20261028b/POOL/a13/runs/syn_0_*_s1/run_*_a13_*/bench.log` | peak 0.8491, final 0.8295 |
| P2 | 15 | ρ0 0.14 | DistilBERT · AG News | `--rho0 0.14 --t-res 71` | ≤ 4 h | **stopped by hand** 10-28 20:41 (first queue stopped to relaunch on 8 GPUs; no final eval): 89 min wall, ~c890, 32,006 vclock s (8.9 h); flat at 0.858 for the last 5 evals, never reached 0.86. Not rerun in `paper_ab_20261028b`: **rerun todo** | `tune_rho0` | `paper_ab_20261028/bench.log`, `paper_ab_20261028/POOL/a15/runs/syn_0_*_s1/run_*_a15_*/bench.log` | peak 0.8583 (no final eval) |
| P2 | 16 | ρ0 0.28 | DistilBERT · AG News | `--rho0 0.28 --t-res 18` (upper limit; N\* scales as ρ0², so expected to break at N 50) | ≤ 4 h | **stopped by hand** 10-28 20:41 (same stop; no final eval): 84 min wall, ~c885, 30,088 vclock s (8.4 h); plateau 0.787. Usable as the expected-to-break point (DNF) | `tune_rho0` | `paper_ab_20261028/bench.log`, `paper_ab_20261028/POOL/a16/runs/syn_0_*_s1/run_*_a16_*/bench.log` | peak 0.7924 (no final eval) |
| P2 | 17 | N 10 | DistilBERT · AG News | `--pool-target 10` | ≤ 4 h | **done as DNF** 10-28 (`commit_cap` 1000): 41.0 min wall, 7,058 vclock s (1.96 h), 10,029 tasks; flat 0.832–0.835; final 0.8328 | `tune_commit_rule` | `paper_ab_20261028b/bench.log`, `paper_ab_20261028b/POOL/a17/runs/syn_0_*_s1/run_*_a17_*/bench.log` | peak 0.8355, final 0.8328 |
| P2 | 19 | N 90 | DistilBERT · AG News | `--pool-target 90` (also confirms N 50 against P0's N 90) | ≤ 4 h | **done** 10-28: target 0.86 ×2 at c369, 57.6 min wall, 17,746 vclock s (4.93 h), 33,509 tasks; final 0.8611 | `tune_commit_rule` | `paper_ab_20261028b/bench.log`, `paper_ab_20261028b/POOL/a19/runs/syn_0_*_s1/run_*_a19_*/bench.log` | peak 0.8612, final 0.8611 |
| P2 | 21 | constant ρ | DistilBERT · AG News | `--rho-schedule pool` (ρ 0.07; ends on the Φ rail) | ≤ 4 h | **done** 10-28: target 0.86 ×2 at c365, 33.0 min wall, 9,693 vclock s (2.69 h), 18,429 tasks; final 0.8607. Fastest arm | `tune_rho_schedule` | `paper_ab_20261028b/bench.log`, `paper_ab_20261028b/POOL/a21/runs/syn_0_*_s1/run_*_a21_*/bench.log` | peak 0.8625, final 0.8607 |
| P3 | 14 | ρ0 0.10 | DistilBERT · AG News | `--rho0 0.10 --t-res 139` | ≤ 4 h | **done** 10-28: target 0.86 ×2 at c501, 47.2 min wall, 15,258 vclock s (4.24 h), 25,329 tasks; final 0.8605 | `tune_rho0` (4th value) | `paper_ab_20261028b/bench.log`, `paper_ab_20261028b/POOL/a14/runs/syn_0_*_s1/run_*_a14_*/bench.log` | peak 0.8605, final 0.8605 |
| P3 | 18 | N 20 | DistilBERT · AG News | `--pool-target 20` | ≤ 4 h | **done as DNF** 10-28 (`commit_cap` 1000): 53.1 min wall, 14,073 vclock s (3.91 h), 20,029 tasks; 0.1 pt short of 0.86, still rising slowly; final 0.8592. Rerun with a higher cap if N 20 needs a TTA | `tune_commit_rule` (4th value) | `paper_ab_20261028b/bench.log`, `paper_ab_20261028b/POOL/a18/runs/syn_0_*_s1/run_*_a18_*/bench.log` | peak 0.8593, final 0.8592 |
| P3 | 20 | adaptive pool | DistilBERT · AG News | `--commit-gate n_target` (no `--pool-target`) | ≤ 4 h | **done** 10-28: target 0.86 ×2 at c745, 49.6 min wall, 15,069 vclock s (4.19 h), 22,529 tasks; final 0.8601 | `tune_commit_rule` | `paper_ab_20261028b/bench.log`, `paper_ab_20261028b/POOL/a20/runs/syn_0_*_s1/run_*_a20_*/bench.log` | peak 0.8605, final 0.8601 |
| P3 | 22 | momentum 0.8 | DistilBERT · AG News | `--server-momentum 0.8` (0.9 diverged) | ≤ 4 h | **done as DNF, diverged** 10-28 (`reason=stall` at c534): 48.1 min wall, 12,721 vclock s (3.53 h), 26,729 tasks; peak 0.8214 early, then collapsed to 0.25 (one class), final loss 12.4. Momentum 0.8 diverges at AG News too | `tune_rho_schedule` | `paper_ab_20261028b/bench.log`, `paper_ab_20261028b/POOL/a22/runs/syn_0_*_s1/run_*_a22_*/bench.log` | peak 0.8214, final 0.2500 |
| P3 | 23 | FluxTune, `mobiperf_3st_50` | DistilBERT · AG News | trace `mobiperf_3st_50` | ≤ 4 h | **not for MLSys** (dropped); was: blocked: FT-N5 | `strategy_availability_mobiperf` |  |  |
| P3 | 24 | Felix(P), `mobiperf_3st_50` | DistilBERT · AG News | as 12, trace `mobiperf_3st_50` | ≤ 4 h | **not for MLSys** (dropped); was: blocked: FT-N5 | `strategy_availability_mobiperf` |  |  |
| P3 | 25 | FedBuff(P), `mobiperf_3st_50` | DistilBERT · AG News | as 13, trace `mobiperf_3st_50` | ≤ 4 h | **not for MLSys** (dropped); was: blocked: FT-N5 | `strategy_availability_mobiperf` |  |  |
| P3 | 26 | FluxTune, max-\|JVP\| upload | DistilBERT · AG News | max-\|JVP\| upload (same 2P passes, D and bytes as run 2); `select_perturbation_using_jvp` default | ≤ 4 h | **done as DNF** 10-28 (`commit_cap` 1000): 87.0 min wall, 34,047 vclock s (9.46 h), 50,029 tasks; peak 0.8514 then slipped; final 0.8475 | `Contri_probe_mean` | `paper_ab_20261028b/bench.log`, `paper_ab_20261028b/POOL/a26/runs/syn_0_*_s1/run_*_a26_*/bench.log` | peak 0.8514, final 0.8475 |
| P3 | 27 | FluxTune, alignment-only ω | DistilBERT · AG News | `--agg-rate-type grad_aware` + yaml override `agg_rate_conf.base: neutral` (ω = alignment gate only, base 1.0; the launcher hardcodes `base: new`) | ≤ 4 h | ready (G0 closed 10-25) | `tune_agg_weights` |  |  |

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
| 13:58 | **1.7B lr 3e-3 / var 8.4 confirmed** (run 5) | at ~15–18 vclock h: 3e-3 / 8.4 acc 0.305, loss 2.18 (c13) vs var 15 0.195 / 2.60 (c21; 7.9 vs 10.1 iterations per commit, loss stalled) and lr 5e-3 0.156 / 2.81 (c9; 16.7 iterations per commit, loss rising). Both stopped by hand at ~32 min | `queue_T_sm.log`, `harness_T_sm_*.{yaml,out}`, `experiments/run_*fwdsm_T_sm_*` |

*Final values (`fwdllm` arm)*

| Cell | Value | Final since | Basis |
|---|---|---|---|
| all cells | `reselect_cadence: data_bin` | 05:00 | collapse fix; matches upstream |
| 1.7B · Yahoo | **lr 3e-3** | 06:40 | beats 3e-4 / 1e-3 at matched commits; 1e-2 overshoots (c1 loss 4.10) |
| 1.7B · Yahoo | var **8.4** (var 15 worse, 13:58) | 06:20 | ~10 iterations per commit; learns: the final-values smoke `D_sm_lr0.003_v8.4` (05:46 → 07:45, 67 commits) reached acc 0.384 (c65), best loss 2.03, MCC +0.32. Var is a speed knob |
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
  3. ~~`fluxtune` on RoBERTa-L and 1.7B~~ **pass**: 1.7B F_sm (07:30, above); RoBERTa-L pair (operator 10-26).
  4. `bench_profiles.yaml` entries: `yelp-p` (~10 GB GPU at 4 workers, measured), `yahoo` (~16–18 GB at 4 workers; confirm from V2c).
  5. Re-run B2 on DistilBERT `fwdllm` / `fluxtune` (1 seed) after the shared `fwdllm_aggregator.py` / `random.py` fix (rule: re-run B2 on
     touched cells after any shared change; the fix should be inert there, D < 300 s).
  6. Same result at 2 and 4 workers (B2 gate item; not checked on big models).
  Separate from bench parity and still open: real ↔ sim for big models (FT-N21).
- **Fix applied 10-25 (jayne + shepherd, uncommitted):** stuck timeout = max(300 s, `send_timeout_wait_s`); `random.py` clamps
  `required_trainers` ≥ 0; regression test `test_send_timeout_extends_stuck_window` (444 pass). Recheck V2b (1.7B lr 1e-3 n 10, bench GPU 0 vs
  harness GPUs 2–3, tmux `dg_flame:fwdv2b`) running.

### 3.6 Run log: DistilBERT · AG News P1 (runs 1, 2)

Shepherd bench (`bench_pool.py`, decision 14): one leg per 4 GPUs (`fwdllm` 0–3, `fluxtune` 4–7), 8 workers, seed 1234. §2 recipe; caps 6 h wall / 2,000 commits, saturation stop on (warm-up 900); full test set (7,600) every 4 commits. Both legs
were stopped by hand once they plateaued (operator), so neither reached a cap; the bench logs both as FAIL rc −15.
All paths under `fwdllm/p1_distil_20261025/`.

| Arm | Run dir (`bench/P1_<arm>/runs/syn_0_<arm>_s1/`) | Commits · wall · vclock | s/commit (wall) | 0.86 ×2 | Peak | Last | Φ end |
|---|---|---|---|---|---|---|---|
| `fluxtune` (run 2) | `run_20261025_120448_fluxtune_agnews_n100_smoke_syn_0_sim` | 1,193 · 79 min · 11.6 h | 4.0 | **c525, 4.34 vclock h** (34.6 min wall) | 0.873 (c1,097) | 0.871 | 1.98 |
| `fwdllm` (run 1) | `run_20261025_120447_fwdllm_agnews_n100_smoke_syn_0_sim` | 493 · 108 min · 27.0 h | 4.7 → ~20 (slows as iterations per commit grow) | DNF | 0.855 (c333) | 0.837 | 4.31 |

| File | What |
|---|---|
| `jobs.tsv` | bench jobs (one line per leg: phase, baseline, trace, YAML, gpu_gb / ram_gb / workers) |
| `fwdllm_agnews_n100_smoke_syn_0_sim.yaml`, `fluxtune_agnews_n100_smoke_syn_0_sim.yaml` | leg configs (`run_sequential.sh --dry-run`; flags in §2 and decision 14) |
| `pool.log` | queue log: START / FAIL lines per leg |
| `bench/P1_<arm>/runs/syn_0_<arm>_s1/leg.out` | leg log (trainer + aggregator stdout; no `*_aggregator.log` on the bench) |
| `bench/P1_<arm>/runs/syn_0_<arm>_s1/_bench_input_<pid>.yaml` | YAML as the bench ran it |
| `<run dir>/telemetry/*aggregator*.jsonl` | evals (`agg_eval`), commits (`server_update`, with ρ, Φ), `vclock_now`; source for figures |
| `acc_fluxtune.tsv`, `acc_fwdllm.tsv` | per eval: commit, wall min, vclock h, acc, loss, Φ (extracted from telemetry) |
| `aborted_1gpu_bench/`, `aborted_1gpu_pool.log` | first start, 1 GPU per leg (FluxTune ~12 s/commit), stopped at ~3 min; not used |

Reading: FluxTune peaks 0.5 pt below P0's 0.878 (N 90, 1 seed, inside the 2.3-pt noise). FwdLLM matches A0's plateau (0.851) under
`data_bin`; plain SGD keeps growing ‖θ‖ (Φ 4.3) without gain. Figure: add both run dirs to `figs_e2e.yaml` (`distilbert_agnews`), todo.
Preview plot (acc vs wall and vclock, EMA 0.6 over faint raw): `fwdllm/p1_distil_fwdllm_vs_fluxtune.png`, script
`fwdllm/scripts/plot_p1_distil_fwdllm_vs_fluxtune.py <out.png>` (reads the two TSVs).

**Paper points** (sim time only; speedup rule in [EXPERIMENTS.md §5](EXPERIMENTS.md#5-scoring-rules)):

| Point | FwdLLM | FluxTune | Claim |
|---|---|---|---|
| Time to FwdLLM's peak (0.855) | 14.8 h | 3.1 h (first eval ≥ 0.8546) | **4.8× faster** (headline speedup) |
| Time to 0.86 target | DNF in 27.0 h | 4.34 h (×2) | **≥ 6.2×** (lower bound) |
| Peak accuracy | 0.855 | 0.873 | **+1.8 pt** |
| Accuracy at 4.3 h | 0.782 | 0.860 | +7.8 pt at equal time |
| Max drop after peak | 9.6 pt (0.759) | 0.4 pt (0.869) | FluxTune stable at plateau |

- FwdLLM plateaus at 0.82–0.85 from ~7 h on (non-IID ceiling, as the FwdLLM paper reports for non-IID); FluxTune passes it.
- Don't extrapolate FwdLLM to 0.86 (flat curve) or quote a hand-picked threshold (0.84 gives 3.2×, 0.85 gives 5.4×).
- Wording: "FwdLLM plateaus at 85.5%. FluxTune reaches it 4.8× faster in simulated time and continues to 87.3%; FwdLLM does
  not reach the 86% target within 27 h."
- 1 seed per arm: the +1.8-pt peak gap is inside the 2.3-pt replicate noise; the speedup and the DNF are not.

**3.7 Run log: SmolLM2-1.7B · Yahoo rank / probes / momentum smokes (10-25, jayne bench)**

Smoke ablation on the n 60 recipe (not a paper figure); full notes also in [fl_fwd_ft_smollm2_smoke_plan.md](../fwdllm/fl_fwd_ft_smollm2_smoke_plan.md) §14. Paths relative to `lib/python/examples/fwdllm/`. Snapshot tables taken 2026-10-26 00:31; all four legs stopped by hand 10-26 (finals: status board Done).

**Summary**
- **Backprop ceiling (r=4, centralised, pooled test sample): ~0.77** (plateau from step 750; single seed). Best FluxTune SmolLM so far: 0.598 (shepherd, n=60). The 0.65 target was inherited from DistilBERT's FL target (0.660 vs its 0.734 ceiling, a 7-pt gap); for SmolLM a 5-8 pt drop from 0.77 suggests **0.69-0.72**.
- **Best so far: LoRA r=2 + server momentum 0.9 at the same s (2.81).** At matched commits it is ahead of plain r=2 (c12: 0.249 / 2.119 vs 0.212 / 2.237) and of base (0.182 / 2.302). Promising, not proven: one seed, n=12 smoke, and only ~4 evals past the point where the momentum buffer filled.
- **r=2 beats r=4 at the same s** (loss ~0.05-0.09 lower, acc +3-6 pts by c30), once rho is scaled with the cosine.
- **K=40 probes is compute-neutral at best** (4x cost per commit for ~4x progress per commit), so it is not worth running.
- **Decay (`t_res`) is irrelevant in the first ~30 commits**; only the combined r=2+momentum leg uses the slower t_res 1119, so that confound is untested.

**What we learned (order of discovery)**
1. `--cos-ground-truth-audit` stalls every leg at commit 1 (aggregator at 100% CPU, ~60-70 GB RAM, GPUs idle), even with `--cos-probe-batch-size 64 --cos-probe-every 5`. Keep it `--no-cos-ground-truth-audit` for SmolLM.
2. In this harness `s = rho0 / cos_theory`, `cos_theory = sqrt(G*N/p)` (G probes, N pool, p trainable). A fixed `rho0` across arms lets s vary (first r=2 / K=40 legs had s = 2.01 / 1.40 against base 2.81) and handicaps them. Hold **s fixed**: rho0 = 0.0698 (r=4,k=10), 0.0978 (r=2,k=10), 0.1387 (r=4,k=40).
3. Under `trust_ratio` the step is `rho * ||theta||`; ||theta|| scales ~sqrt(p), so with a fixed rho0 the cosine gain from a smaller p is cancelled (step x cos was 0.0148 base vs 0.0155 r=2).
4. LoRA rank barely changes per-task cost (probe forwards dominate): ~21 s r=4 vs ~23 s r=2 per task; commit gap 5.1 vs 5.4 min.
5. Momentum is heavy-ball applied before the trust-ratio normalisation, so the step norm stays `rho*||theta||`; its benefit appears after the buffer fills (~10 commits). It started behind plain r=2 (c3, c6) and crossed at c9.

**Shared arm settings (`expt_scripts/run_sequential.sh`, sim mode)**
`--only fluxtune --dataset yahoo --partition-method niid_label_clients=100_alpha=1 --num-trainers 12 --c 6 --c-async 6 --agg-goal 5 --min-initial-frac 0.9 --model-type llama --model-name HuggingFaceTB/SmolLM2-1.7B --peft-method lora --adapter-reduction-factor 16 --fd-scale-invariant --delay-divisor 0.0047 --delay-floor 4.0 --send-timeout-s 3600 --train-batch-size 32 --probe-combine mean --server-step-rule trust_ratio --gate-safety-s 2 --commit-gate fixed --pool-target 50 --rho-schedule exp --agg-rate-type uniform --var-stopping-policy off --phi-stop-threshold 4 --budget-stop-frac 0 --b-max-probe-every 0 --phi-stop halt --server-update-audit --no-cos-ground-truth-audit --eval-max-samples 2000 --eval-resample --eval-every-n-commits 3 --max-commits 150 --max-runtime-s 1000000000 --sim-wall-ceiling-h 24 --force --yes`
Launched through `scripts/bench_pool.py --gpus-per-leg 2 --per-gpu 1` (jobs files in `bench_sm_20261025/`), 2 GPUs per leg, `workers=6`.

**Arms (per-arm values)**
| arm | k | LoRA r | p | rho0 | s | t_res | momentum | GPUs | config yaml | run dir |
|---|---|---|---|---|---|---|---|---|---|---|
| base | 10 | 4 | 806,912 | 0.07 | 2.81 | 560 | 0 | 0,1 | expt_scripts/smoke_logs/20261025_204018/ | bench_sm_20261025/POOL/sm_base/runs/syn_0_fluxtune_s1/run_20261025_204054_* |
| r2_s | 10 | 2 | 413,696 | 0.0978 | 2.81 | 560 | 0 | 4,5 | expt_scripts/smoke_logs/20261025_211919/ | bench_sm_20261025/POOL_s/sm_r2s/runs/syn_0_fluxtune_s1/run_20261025_211930_* |
| k40_s | 40 | 4 | 806,912 | 0.14 | 2.81 | 560 | 0 | 2,3 | expt_scripts/smoke_logs/20261025_211907/ | bench_sm_20261025/POOL_s/sm_k40s/runs/syn_0_fluxtune_s1/run_20261025_211930_* |
| r2_mom | 10 | 2 | 413,696 | 0.0978 | 2.81 | 1119 | 0.9 | 6,7 | expt_scripts/smoke_logs/20261025_224827/ | bench_sm_20261025/POOL_m/sm_r2m/runs/syn_0_fluxtune_s1/run_20261025_224838_* |

Logs inside each run dir: `bench.log` (aggregator + trainers), `telemetry/aggregator_*.jsonl` (`server_update`, `agg_eval`, `agg_round`), `telemetry/trainer_*.jsonl`; the bench driver log is `bench_sm_20261025/bench{3,4,5}.log` and `leg.out` under each `POOL*/sm_*/runs/*/`.

**Per-arm controller values (first commit; mean commit gap)**
| arm | commits | rho (c1) | cos_theory | s=rho/cos | step norm (c1) | weight norm now | min / commit |
|---|---|---|---|---|---|---|---|
| base | 45 | 0.0698 | 0.0249 | 2.81 | 0.595 | 9.44 | 5.1 |
| r2_s | 36 | 0.0973 | 0.0348 | 2.80 | 0.620 | 7.50 | 5.3 |
| k40_s | 9 | 0.1387 | 0.0498 | 2.79 | 1.190 | 9.27 | 19.6 |
| r2_mom | 18 | 0.0973 | 0.0348 | 2.80 | 0.620 | 12.40 | 5.6 |

**Matched evals: accuracy / test loss (wall minutes since leg start)**
| eval | base | r=2 same s | K=40 same s | r=2 + momentum |
|---|---|---|---|---|
| c0 | 0.131 / 2.765 (0 m) | 0.135 / 2.698 (0 m) | 0.144 / 2.506 (0 m) | 0.132 / 2.699 (0 m) |
| c3 | 0.158 / 2.464 (16 m) | 0.153 / 2.407 (16 m) | 0.180 / 2.281 (58 m) | 0.142 / 2.480 (18 m) |
| c6 | 0.143 / 2.408 (31 m) | 0.155 / 2.367 (32 m) | 0.196 / 2.189 (117 m) | 0.170 / 2.510 (34 m) |
| c9 | 0.149 / 2.336 (47 m) | 0.171 / 2.288 (48 m) | – | 0.202 / 2.207 (51 m) |
| c12 | 0.182 / 2.302 (63 m) | 0.212 / 2.237 (65 m) | – | 0.249 / 2.119 (68 m) |
| c15 | 0.203 / 2.246 (77 m) | 0.214 / 2.209 (81 m) | – | 0.283 / 2.065 (85 m) |
| c18 | 0.210 / 2.217 (92 m) | 0.226 / 2.170 (97 m) | – | – |
| c21 | 0.228 / 2.168 (107 m) | 0.266 / 2.115 (112 m) | – | – |
| c24 | 0.233 / 2.128 (122 m) | 0.263 / 2.065 (128 m) | – | – |
| c27 | 0.241 / 2.088 (137 m) | 0.285 / 1.999 (143 m) | – | – |
| c30 | 0.243 / 2.118 (152 m) | 0.300 / 2.029 (158 m) | – | – |
| c33 | 0.266 / 2.071 (167 m) | 0.319 / 1.983 (174 m) | – | – |
| c36 | 0.290 / 2.031 (183 m) | – | – | – |
| c39 | 0.277 / 2.035 (198 m) | – | – | – |
| c42 | 0.328 / 1.963 (212 m) | – | – | – |

Eval noise: 2000 resampled test rows, accuracy ~±0.8 pt, loss ~±0.02.

**Backprop ceiling (`expt_scripts/backprop_ceiling.py`, AdamW lr 2e-4, bs 32, bf16 autocast, seed 1233, 3000-row test sample)**
- r=4: step 250 0.751, 500 0.757, 750 0.770, 1000 0.761, 1250 0.775, 1500 0.767, 1750 0.769 (stopped). Trainable p = 806,912, same as the harness. Curve `bp_ceiling_20261025/r4/curve.jsonl`, log `bp_ceiling_20261025/r4.log`.
- r=2: only step 250 logged, 0.746 / 0.816 (stopped by request). p = 413,696. `bp_ceiling_20261025/r2/curve.jsonl`, `r2.log`.

**Earlier / discarded runs (for reference)**
| run | what | result | location |
|---|---|---|---|
| n=60 baseline (stopped at c398) | rho0 0.0497 (s=2), t_res 1119, k 10, r 4, pool 50, agg-goal 10, c 30, eval 6000 | acc ~0.56 (best 0.576 by c398); loss still falling ~0.09 per 100 commits when stopped, heading to the 10-23 run's ~0.59 plateau | experiments/run_20261025_075720_fluxtune_yahoo_n60_smoke_syn_0_sim/ |
| yesterday n=60 (shepherd, 14.4 h, 531 commits) | rho0 0.07, t_res 283 | best 0.598, plateau 0.57-0.595 from c350 | shepherd:experiments/run_20261023_220053_fluxtune_yahoo_n60_smoke_syn_0_sim/ |
| audit stalls | cos audit batch 1024 / 64 | stuck at commit 1 | bench_sm_20261025/POOL_audit1024_stalled/, POOL_audit64_stalled/ |
| first k40 / r2 legs (fixed rho0 0.07, s 1.40 / 2.01) | killed ~21:19 | r2 c3 2.455 vs base 2.464 (equal) | bench_sm_20261025/POOL/sm_k40, POOL/sm_r2 |

**Caveats**
- n=12 trainers, 150-commit cap, single seed; do not compare with the n=60 runs (different n, agg-goal, s, eval size).
- r=2+momentum also carries `t_res 1119` (the others 560); both changes were requested together.
- Momentum's trainable weight norm grows faster (11.15 at c16 vs 7.43 for plain r=2 at c34, from the same 6.4 start); watch for instability.
- Finals (stopped by hand 10-26): best acc r2_mom 0.432 (c28), r2_s 0.357 (c43), base 0.328 (c42), k40_s 0.288 (c12) → run 6b (decision 15).

**Next (optional, after P4):** r = 1 at the same s; r 2 + momentum at `t_res 560` to separate the decay confound.

## 4. Run-time basis

Under `exp`, Φ is bounded by exp(ρ0²·T_res/2): ≈ 1.96 on DistilBERT, so its Φ rail never fires and the commit cap
ends it; 4 on RoBERTa-L and 1.7B, both with the rail off, ending on target or the 30.5 h wall (decisions 15–17). FwdLLM ran 9.3 s/commit
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
- Superseded by measured bench pace (§3.7: ~5 min/commit at n 12 on 2 GPUs; P3 / P4 record the full-n pace).

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

- **Do P3 / P4 reach target within 30.5 h?** Pace at full n on 4 GPUs is measured from their first hours; if the wall binds first,
  the operator raises `--timeout-s` or accepts a DNF (no final eval on a kill).
- **Momentum under `exp`:** Φ ≠ e^B (FT-T16); read Φ from `server_update`, not from the schedule.
- N 50 and ρ0 0.07 on DistilBERT: shown by `tune_commit_rule` and `tune_rho0` (§3.2).

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
