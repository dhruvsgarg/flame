# Felix readiness — backprop FL (async_cifar10, google_speech)

Read [ROBUST_FL_READINESS.md](ROBUST_FL_READINESS.md) and [PARITY_READINESS.md](PARITY_READINESS.md) first; their rules, L and T apply
here unrepeated. **Focus (operator 10-25 late): the paper's first plots = priorities 1-2, seed 1 (End-to-end, Ladder, swaps:
concurrency and availability; 116 legs). Follow the runbook in Next steps A; the next command is its first unchecked step.**
Paper story and plan: `felix_paper_outline.md` (Table 2 figures → runs, Table 1 run sets); methods FELIX_INSIGHTS §6; run rules
PAPER_TEMPLATE Part E. Trimmed 10-25 late: superseded screens (F43-F48 tables, FL1 motivation draft, contribution ledger, harness
tier tables) live in `git log` and FELIX_INSIGHTS §5.

**Only runs from F41 (10-21) onward count** (trace-tied `usage` streams, FedScale task time, R27 availability knowledge). Pre-F47
cifar numbers ran frozen augmentation draws: relative evidence only. **Felix MobiPerf numbers from F47-F49 ran 1-7% physically
impossible extra tasks (FX-N157): re-read on FB2 data before quoting.**

**At a glance (10-27).** Felix = `fx` (frozen, Decisions): notify / evict + utility selection + rate `new` (s 0.4 / a 0.25 / b 0.2)
+ c ramp on FedBuff's buffer. FB2 cifar (seed 1, 80%): Felix 1.13-1.23× FedBuff, GB −33% / −12% / **+34%** (`s50u`). **Main sync stacks = lr decay every 50 (2e ✅ 10-27):
no sync baseline (incl. oracle arms) reaches 80% / 75% in 120 h on any cell (final 61-78%), so Felix and FedBuff win every cell; decay every 100 / 120 (2d / 2c) are ablation, where FedDance / REFL still beat Felix on `smu` (FX-N166).** Targets: cifar **80%** (FL bound; reported at 75% on MobiPerf), speech **88%** (`targets.json`);
Felix hits cifar at 65-72 h, so paper legs run to the hit or 120 h.

**Terms.** Each aggregation makes a version. *Current model* = the latest version; training continues from it. *Averaged model* = EMA of
recent versions (β = 1 − K/n), served and scored for every stack, never trained from. Arm `X_srv` = averaged, `X_srv:raw` = current.
*Felix old* = arm `ctl` (Oort score + UCB + eval tasks); *Felix new* = `fx` (Decisions; `fn_chr` = it at b 0.1). Both use
rate `new` and FedBuff's buffer.

**Cohort and participation (the one place for n / K / c; operator 10-23: each stack at its own paper's ratios, no stack favoured).**
Rule (`baseline_reference.yaml`): sync K = ceil(max(paper K / N, 5%) × n); async stacks keep FedBuff's buffer K 10 (paper §6.1:
"K = 10 is a good setting across benchmarks") and its c / n (paper Table 1: c 1000 of 5,000 CIFAR-10 clients = 20%).

| stack | paper setting | cifar n 300: K · in flight | speech n 100: K · in flight |
|---|---|---|---|
| FedBuff · Felix | K 10, c 1000 of 5,000 (20%) | 10 · c 60 (Felix: K fixed, c ramps in [max(K, ½ trainable), 60]) | 10 · c 20 (Felix: same rule, ≤ 20; Q-SPEECHC) |
| Oort · Oort* | 100 of 1.3K = 3.8% → 5% | 15 · 16 (× 1.1, truncated) | 5 · 5 (rule; F43 ran 10 · 11: Q-SPEECHK) |
| REFL · REFL* | 10 of 1,000 = 1% → 5% | 15 · 20 (× 1.3, rounded) | 5 · 6 (rule; F43 10 · 13) |
| FedDance · FedDance* · FedAvg | 10 of 1,000 = 1% → 5% | 15 · 19 (× 1.3, truncated) | 5 · 6 (rule; F43 10 · 13) |

Async stacks hold c / K = 6 (cifar) / 3 (speech) tasks in flight per buffer (staleness ≈ c / K); sync stacks 1.1-1.3 × K per round
(they wait instead of going stale). Screens: n:c:K 75:15:10 cifar, 50:15:10 speech (G3_SHAPE). **F43-F45 ran every cifar stack at
K 15 (async off its paper's K); F46 onward uses this table.** Cohort = the reference split's first n trainers, Dirichlet α 0.1.


## Next steps (persistent queue — top item is next)

Work rule: PARITY C10, C17, C18; ROBUST R0. Launches: tmux on shepherd GPUs 0-3 (jayne only on GPUs the operator names; Decisions "Nodes"), committed tree (scripts refuse a
dirty one), from `lib/python/examples`, `PY="conda run --no-capture-output -n dg_flame python"`. Paper runs are all bench
(`paper_runs.py` → `bench_pool`); the harness is only the B2 / EV24 reference. Final runs on `/coc/scratch`.

**Node plan (10-27; rewrite on every change).** Ours: shepherd GPUs 0-3, jayne GPUs 0-7 (operator 10-27; FluxTune leg on 0-1 done 13:57; shepherd
4-7 not ours). Each node has its own `/home/dgarg39/flame` checkout (`git pull` on jayne after a push); both share `$R=/coc/scratch/dgarg/mlsys27_felix_runs_20261026`.

| when | node · GPUs | what | legs · per GPU | wall | why now |
|---|---|---|---|---|---|
| 10-27 02:47 → 07:25 | shepherd 0-3 | 2c sync rerun, every 120 | 32 / 32 ok, 0 diverged | 4 h 20 | ✅ recorded |
| 10-27 04:05 → 06:47 | jayne 2-7 | B2 cifar run 3 | 3 / 5 rows PASS | 2 h 40 | step 3 |
| 10-27 10:52 → 15:38 | shepherd 0-3 | 2d sync rerun, every 100 (tmux `n163h`) | 32 / 32 ok, 0 diverged | 4 h 46 | ✅ recorded |
| 10-27 10:54 → 12:04 | jayne 2-7 | B2 cifar oort (RT 12 h) + fedbuff reruns | fedbuff PASS, oort upd +5.3% | 1 h 10 | ✅ recorded |
| 10-27 17:00 → ~21:45 | shepherd 4-7 | 2e · cifar sync at lr decay every 50 = main (tmux `e50`, `$R/cifar_e50`, worktree `~/flame_e50`) | 32 / 32 ok, 0 diverged | 4 h 48 | ✅ recorded |
| 10-27 14:44 → 10-28 01:22 | shepherd 0-3 | 4 · speech pool S: `fx` × 4 cells | 4 / 4 ok, 0 diverged | 10 h 38 | ✅ recorded; GPUs 0-3 idle until M |
| 10-27 14:22 → ~10-28 12:00 (10-28 02:38: 12 / 30 done, 8 running, 10 queued; the queue self-launches as GPUs free, no action) | jayne 0-7 | 4 · speech pool J: every non-`smu` non-Felix leg, oracle arms first | 30 · 1 | ~21 h | E2E only; do not mark stale |
| 10-28 ~02:45 (operator launches) → ~17:00-19:00 | shepherd 0-3 | 4 · speech pool M: the other 10 `smu` legs at the new target, oracle first | 10 · 1 | ~16 h | target set; run command 4's tmux line |
| after speech E2E | jayne 2-7 | 3 · B2 speech sync (4 rows) | harness 4 + bench 2 | ~1.75 h | after, as cifar did (operator 10-27) |
| then | freed GPUs | future TODO (operator 10-28: no time): seeds 2-3, then speech ablations (priority 3, below seeds; skipped, cifar covers them) | — | — | — |

Two pools must never share a GPU with a harness leg (FX-T91 OOM); bench pools may (1 speech + 4 cifar = 34 GB, FX-N161).
**Run order rule (operator 10-27; standing):** seed 1 of every End-to-end cell (both datasets; every final sync / async config) first,
for the one End-to-end plot of numbers and gains; everything else waits. Within it, Felix, then the oracle `*` arms, then the
deployable ones, so every baseline is on the plot early (cifar seed 1: oracle and check-in arms close for REFL 86-95 vs 88-94 h,
FedBuff 78-81 vs 72-82 h; not for FedDance: FedDance\* ✗ 77%, so deployable FedDance still matters). Ablations (Ladder, swaps,
rejected ideas) are cifar-only in the paper (cifar seed 1 done); speech ablations skipped (priority 3, below seeds 2-3; operator 10-28). Then seed 1 of priorities
2 → 3 → 4; seeds 2-3 are a future TODO (operator 10-28).
**Sequencing (operator 10-27 13:00):** no idle GPUs, no reruns, running past 10 h is fine: one queue per node, longest legs first, so
a GPU takes the next leg the moment one ends. No cifar seeds beside speech legs (they would slow seed 1). B2 speech runs after the
speech E2E legs (operator 10-27: cifar ran B2 after too); a clear B2 miss means rerunning that baseline's speech legs. Speech E2E =
12 async × ~11 h + 32 sync × ~4 h ≈ 260 GPU-h on 12 GPUs: done ~10-28 evening.

**Commands (10-27 14:15; tmux, from `/home/dgarg39/flame/lib/python/examples`, `R=/coc/scratch/dgarg/mlsys27_felix_runs_20261026`).**
Split scripts are pre-built (`$R/split_launch.py` from `$R/launch_p2_speech.sh`): S 4 + J 30 now, M 10 after the speech MobiPerf
target = all 44 speech End-to-end legs, no overlap. Every speech pool runs `--gpu-reserve-gb 20`: placement reads real free memory,
and with 45 GB GPUs (44.4 GiB free when empty) a speech leg (22) then starts only on an empty GPU, never beside another speech leg or
2d's cifar legs. Jayne 0-1 are free (FluxTune leg done 13:57), so J uses all 8.
*Tripwire (10-27 14:20):* pools sharing `--out` share `_yaml/`; S's first launch crashed (`dry run gave 0 yaml`) when J's dry-run
rewrote the same dirs. `split_launch.py` now keeps each pool's `--phases` to its own legs; relaunch S (command 1) as is.
*Why M waits:* speech `smu` 88% is out of reach (F49 Felix 52.8% at 12 h; cifar `smu` topped at 77.8% vs 80%), so it gets a per-trace
target like cifar's 75% (Decisions "Reporting target per trace"). A leg's target is fixed in its YAML when its pool starts, so Felix
`smu` (pool S) sets it, and the other `smu` legs start in pool M at that target instead of running 120 h. M goes on shepherd because S
ends there just as the target lands.

| # | node · when | command | est. wall |
|---|---|---|---|
| 1 | ✅ shepherd · pool S | done 10-28 01:22, 4 / 4 (`$R/speech_s1_S.log`) | 10 h 38 |
| 2 | jayne · running (10-28 01:35: 10 / 30) | `tmux new -s sp_J` → `GPUS=0,1,2,3,4,5,6,7 bash $R/launch_p1_speech_J.sh` (log `$R/speech_s1_J.log`); 15 oracle legs first (FedBuff\*, REFL\*, FedDance\*, FedAvg\*, Oort\*), then FedBuff + 12 deployable sync | ~20 h (6 async ~11 h, 24 sync ~4 h) |
| 3 | ✅ done 10-28 (`159efc904`; `smu` `fx` re-digested: re-set to 84% (operator 10-28): hit 97.8 h, 1,378 GB; pushed, jayne pulled) | speech MobiPerf target: peak 5-eval mean of `smu` `fx` (`leg_digest`), rounded down to 5 pp → add `"google_speech": {"mobiperf_3st": 0.80}` to `_metadata/trace_targets.json` together with the uncommitted `leg_digest.report_target` fix (it read speech as cifar via job name "cifar-10", so the `smu` `fx` digest says hit 31.4 h at 75%; the real hit is 80%), commit, push, `git pull` on both nodes; re-digest `smu` `fx` (hit = 80%). The key change marks S / J legs `stale: code key` (as cifar's 75%): accept | ~15 min |
| 4 | shepherd · **launch 10-28 ~02:45, tmux `sp_M`; log `$R/speech_s1_M.log`; ends ~17:00-19:00 (ctl ~10 h, FedBuff\* ≤ 11 h, 8 sync 3-5 h on 4 GPUs)** | pool M (block below; the split must print 10 legs) | ~16 h (2 async ≤ 11 h + 8 sync ~4 h on 4 GPUs) |
| 5 | jayne 2-7 · after speech E2E | `tmux new -s b2s` → `HGPUS=2,3,4,5 BGPUS=6,7 DS=google_speech TAG=1027_s ARMS=sync_srv BLS=refl,feddance,fedavg,oort bash scripts/run_b2.sh` | ~1.75 h |
| 6 | **future TODO (operator 10-28: no time; cifar and speech seeds 2-3 not planned)** | Run order rule: `$PY scripts/paper_runs.py launch --prio 2 --seeds 2 3 --datasets cifar10 --gpus <gpus> --root $R && bash $R/launch_p2_cifar.sh` (sync at every 100); speech priority 3 later | ~1 day (cifar 4 / GPU) |

Command 4 (pool M, shepherd):
```
$PY scripts/paper_runs.py launch --prio 2 --datasets google_speech --root $R   # rewrites the script only; pool M's YAMLs take X when it starts
python3 $R/split_launch.py $R/launch_p2_speech.sh $R/launch_p1_speech_M.sh M 0,1,2,3 '^gs_FB2E2E_smu_(?!fx )' \
    '^gs_FB2E2E_\S+_((fbstar|rstar|fdstar|favstar)_srv \S+|sync_srv oort_star) ' '--gpu-reserve-gb 20'
tmux new -s sp_M   # then: bash $R/launch_p1_speech_M.sh
```

*Never run a full `launch_p*_<dataset>.sh` while that dataset's pools run: running legs count as missing and would run twice;
split off only legs no pool holds.* Checks while running: 3c.2-3.4 on the first speech sync legs (~12 h in).

**A · Paper first plots (operator 10-25 late).** Runbook; tick each step here with its pool path when it passes.
1. ✅ **Gates (10-26, `6dbd6e535`).** EV24 5 / 5 (EV16 0, EV24 0); cifar smoke 30 / 30 + figures render; speech smoke 28 / 30 (step 3b).
2. ✅ **Cifar run, priorities 1-3 (70 legs, done 10-26 ~21:00; 69 ok + `fn` killed at 75%, digested by hand).** Root `$R`; status:
   `paper_runs.py status --root $R --datasets cifar10`; a hand-killed leg is `FAIL rc=-15`, digest with `scripts/leg_digest.py <run dir>`.
2a. ✅ **Divergence fix for the FedScale family (FX-N166b, 10-26): scaled lr decay alone fixes it; no clip.** Decay-off, 10 / 32 sync legs
   diverged (fixed lr 0.1, 1 step, no clip, no normalization; step doubles per round once loss ~0.3 → NaN). In-process FedAvg, 20k
   rounds (`$R/n163_divcheck/`): no decay NaN at 14,750; decay 0.98 / 200 stable to 20-25k (3 / 3 seeds, ~70%); clip 1.0 hurts (61%).
   Rerun: `$PY scripts/fl_lr_check.py --dataset cifar10 --pairs 0.1:1 --rate 1 --k 15 --batch 24 --local-steps 1 --momentum 0.9
   --weight-decay 0.0005 --rounds 20000 --eval-every 250 --gpus 0 [--lr-decay 0.98:200:0.00005] [--clip-grad-norm 1.0]`.
   Rounds / 120 h: all sync stacks 17-27k (one bucket), Oort 0.8-1.4k (own bucket).
2b. ✅ **Cifar sync rerun at every 200 (FedDance\* 300, Oort 9), 32 / 32, 10-27 01:10; superseded.** `$R/launch_n163b_*.sh`; older `run_*`
   (appendix sensitivity). Hit 80% (`smu` 75%): REFL 66-85 h, REFL\* 65-74, FedDance 64-69 (`smu` 73), FedAvg 78-94, FedAvg\* 84-88;
   Oort family ✗ 59-66%, FedDance\* ✗ 78.6-79.5%. FedDance beat Felix on `s20u` / `s50u`. Live accuracy: last `"agg_eval"` in
   `telemetry/aggregator_*.jsonl` (`tail -c 60000000`).
2c. ✅ **Cifar sync rerun at every 120 (target anchor), 32 / 32, 10-27 07:25 (shepherd 0-3, 260 min), code `ccb25f6fc`.** Script
   `$R/launch_n163t_cifar.sh`, log `$R/cifar_n163t.log`, status `$R/status_n163t.md`; 0 `NONFINITE_REJECT`, 0 diverged, `lrDecayEpoch: 120`
   in every leg. Numbers: Scoreboard (2c table). Its `run_*` dirs stay on disk when 2d writes newer ones.
2d. ✅ **(done 10-27 ~15:38, table in Scoreboard; superseded 10-27 by 2e: every 50 is main, 100 = ablation)** **Cifar sync rerun at every 100 (operator 10-27), same 32 legs. Operator 10-27 13:30: every 100 is the paper value for all
   further cifar runs (2c = appendix sensitivity).** Yaml cifar REFL / FedDance (FedAvg reads it) / Oort (Oort\*
   `same_as`) every 100; speech unchanged (20 / 30); 102 reference / harness tests pass. Launch (shepherd, after commit; script asserts 100
   and a clean tree):
   ```
   tmux new -s n163h
   cd /home/dgarg39/flame/lib/python/examples && bash /coc/scratch/dgarg/mlsys27_felix_runs_20261026/launch_n163h_cifar.sh
   ```
   Log `$R/cifar_n163h.log`, status `$R/status_n163h.md`. *Checks:* 0 `NONFINITE_REJECT` / diverged; `lrDecayEpoch: 100` in each new
   `debug_run.yaml`; trainer `lr` = 0.1 × 0.98^((round − 1) // 100) at 2-3 rounds of one leg; then add a 2d table beside 2c's and pick one
   for the paper (the other is appendix sensitivity). *Open from 2c:* `paper_runs.py status` shows End-to-end cifar `stale: code key` (key
   predates `08b094f8c`'s `harness_pool.py` edit): rebase the key or accept before `paper_plots.py`.
2e. ✅ **(done 10-27 ~21:45, table in Scoreboard: main cifar sync result; 2d / 2c = ablation)** **Cifar sync at lr decay every 50 (operator 10-27), same 32 legs, shepherd GPUs 4-7.** Not committed: the `every: 50` edit lives
   only in the worktree `~/flame_e50` (branch `e50`), because a push / pull would change the trees under the running speech pools. Output goes to
   its own `$R/cifar_e50` (2d untouched). Launch (shepherd; script `$R/launch_e50_cifar.sh`, log `$R/cifar_e50.log`):
   ```
   cd /home/dgarg39/flame && git worktree add -b e50 ~/flame_e50 HEAD && cd ~/flame_e50/lib/python/examples/_metadata \
     && sed -i '50s/every: 100/every: 50/;92s/every: 100/every: 50/;148s/every: 100/every: 50/' baseline_reference.yaml \
     && git commit -qam "cifar sync lr decay every 50 (sensitivity)" && tmux new -s e50 "bash /coc/scratch/dgarg/mlsys27_felix_runs_20261026/launch_e50_cifar.sh"
   ```
   *Checks (all ✅):* 32 / 32 ok in 288 min, 0 `NONFINITE_REJECT`, 0 diverged, `lrDecayEpoch: 50` in `debug_run.yaml`. *Result:* all 32 sync legs miss the
   target. **TODO (deferred on purpose: speech runs are live, a commit would change their code hash):** `every: 50` lives only on branch `e50` (main still
   says 100). After the speech E2E finishes, commit it to main and rebase the cifar code key (`$R/cifar*/code_key.txt`). Until then cifar figures (10-27 ~22:30)
   come from a scratch root `$SCRATCH/root` that symlinks `$R/cifar` with the sync / oracle pools (`sync_srv`, `rstar`, `fdstar`, `favstar`) pointed at
   `$R/cifar_e50` and `code_key.txt` forced to this tree's key; rebuild it the same way for re-plots (`paper_plots.py --root <it>`).
3. [ ] **B2 on jayne 2-7 (node to itself; cifar harness takes 4 GPUs, FX-D52).** *Pass:* `bench_validate.md` all rows within bars, cifar
   then speech (FX-N151). `run_b2.sh` refuses a dirty tree.
   - Run 1-2 (`1027_c`, `_c2`): harness CUDA OOM; root `F46_SHAPE` gave the n 300 harness leg 1 GPU; fixed `08b094f8c`.
   - Run 3 (`1027_c3`, `08b094f8c`): refl / feddance / fedavg PASS (AUC Δ −0.3 / 0.7 / −0.0 pp). **oort FAIL** (AUC +2.1 > 1.5 pp; upd / h
     bench 135 vs harness 114, task s 102 vs 104): the harness leg ran alone (04:10-04:18), so not contention; at RT 4 h Oort makes ~30
     rounds, n = 1, so a few slow rounds swing it (FX-N151 sampling noise). **fedbuff DOOMED** by the live EV23 check (3 of 22,759 picks
     without a `picked` check-in); EV23 on the full telemetry PASSes (23,018 picks = 23,018 `picked`): a live-prefix read race, not a
     protocol bug. Fix (10-27): `doomed()` skips EV23's pick-count FAIL (protocol breaks stay live); test added.
   - Run 4-5 (`1027_c4` oort RT 12 h, `_c5` fedbuff RT 4 h; `0c67829f9`, jayne 10:54-12:04): **fedbuff PASS** (AUC 39.6 / 39.1, Δ 0.5;
     upd / h 23,549 / 23,623; GB 406 / 407). **oort FAIL on upd only, marginal** (AUC 29.2 / 30.7, Δ −1.5 at the bar; upd / h 147 / 140
     = +5.3% vs 5% bar; GB 8.1 / 7.8; task s 93.5 / 94.4): more rounds shrank run 3's 18% gap to 5%, so sampling noise, not a bench bug.
   - **Cifar closed (operator 10-27: no time to rerun):** 4 / 5 PASS + oort marginal (accepted; seeds 2-3 for its bar not run).
   - *Felix in B2:* not in runs 1-5 (`fx` is not an F49 arm; `ctl_srv` × fedbuff = FedBuff). Felix's bench ⇄ harness match is
     FX-N149 (speech `fx` + FedBuff, 10-25) and cifar B2 ✅; the later Felix fixes (N157, N160) change both sides and passed EV24, so no
     rerun. Optional (+~1 h): add `fn_chr` to the speech B2 `ARMS`.
   - **Speech B2 (`1027_s`): after the speech E2E legs (operator 10-27; command 5).** Sync rows only (Felix / FedBuff green,
     FX-N149), 12 h vclock, cell `smu`. A marginal miss (like cifar oort) → accept and record; a clear miss → root it and rerun that
     baseline's speech legs. Also read rounds / h per baseline from the digests (3c.3).
3b. [ ] **Speech smoke re-run: skipped (proposal 10-27; operator to confirm).** 10-26 smoke passed 28 / 30 (2 OOMs,
   fixed FX-N161); since then only the sync lr overlay changed, which B2 speech exercises. Was:
   `R2=$R/FB2S_speech; $PY scripts/paper_runs.py launch --smoke --prio 4 --datasets google_speech --gpus 0,1,2,3 --root $R2 && bash
   $R2/launch_p4_smoke_speech.sh` (30 speech legs, 1 / GPU, ~1.5 h; script name ends `_speech`). *Pass:* 30 / 30.
3c. [ ] **Speech sync sanity, before step 4 (operator 10-26: do not repeat cifar's misses).** Follow the decay *Procedure*
   (Decisions, "Sync stacks"), with speech's own rounds, not cifar's (speech rounds are ~10× fewer).
   Current overlay (FX-N163, from F49's ~14 rounds / h ≈ 1.7k): every 20, Oort 30. Run after 2a so its fix is in the test.
   **Decision 10-27: keep every 20 / Oort 30 as final for speech; cifar's 100 does not carry over.** `every` counts rounds, and speech
   runs ~10× fewer (~1.7k in 120 h vs cifar ~18-23k): every 100 would end speech at 0.98^17 = 71% of base lr, i.e. almost no decay
   (safe: 3c.1 saw no divergence without decay, but not the source schedule). Same rule, per-dataset value. The target anchor needs a decay-off speech crossing of 88%; none
   exists (3c.1 in-process peaked 62-69% at 3,500 rounds), so the budget fallback holds (every ≈ 10 × 1.7k / 1,000 → 20; lr ends at
   0.98^85 = 18% of base vs the source's 13%). 3c.1: no divergence, decay ≥ no decay. Re-check only 3c.3 (rounds within 2× of 1.7k)
   on B2 speech and pool J's first 12 h; 3c.2 (`NONFINITE_REJECT` = 0) live.
   Re-derive with the target anchor (Decisions): needs speech decay-off crossing rounds; a family with none takes the budget fallback.
   0. *Status 10-27 01:15:* 3c.1 launched 23:38 on shepherd GPUs 2,3 (idle after 2b C), tmux `n163c_speech`, `$R/n163c_speech/run.sh`,
      logs `{refl,fed}_{none,decay,clip,both}.log`: 8 sims (REFL recipe + FedDance / FedAvg recipe × 4 arms), K 5, SGD, 3,500 rounds, 4 / GPU,
      decay 0.98 / 20. **REFL recipe done (3,500 rounds, 0 NaN): decay 61.9%, none 59.5%, clip 43.8%, both 32.8%.** FedDance / FedAvg recipe at round 2.3-3.0k, 0 NaN so far (decay 62.7%, none 45.3%); ETA ~01:45. Clip 1.0 hurts (rejected again); no divergence without decay yet. Must end before 3b (needs GPUs 2,3).
   *3c.1 result 10-27 02:00:* 8 / 8 sims done at 3,500 rounds, 0 NaN: FedDance / FedAvg recipe decay 68.5%, none 68.6%, clip 49.9%,
   both 39.7%; REFL recipe 61.9 / 59.5 / 43.8 / 32.8% (decay / none / clip / both). No speech divergence; clip rejected.
   1. *Divergence, in-process:* `fl_lr_check.py --dataset google_speech` with each speech recipe (Baseline hyperparameters: REFL
      batch 20 lr 0.05; FedDance / FedAvg batch 32 lr 0.075; 1 step, mom 0.9, wd 5e-4), arms = no decay / decay / clip / both, to
      ≥ 3k rounds (> 120 h of speech rounds). Any non-finite → apply the 2a fix to speech too.
   2. *Divergence, live:* every speech sync leg's newest `bench.log` has 0 `NONFINITE_REJECT`; digest `updates.versions_per_h` never
      drops to 0 while the leg runs (cifar's diverged legs ran to 120 h frozen and looked like plain misses).
   3. *Rates:* rounds / h per baseline × cell (incl. `*` arms) from stable legs (3b smoke = early rate; confirm on the first full
      legs); rounds at 120 h → `every`; one bucket per ~2× spread, Oort separate. Change the overlay only if rounds differ > 2× from 1.7k.
      Oort: place its floor at the end of the run as cifar did (every = 15 × rounds / ~584 for speech's 0.95 / 15 source), not at
      58% of it; check the oracle arms' rounds separately (cifar FedDance\* ran ~1.5× FedDance).
   4. *Schedule check:* trainer `lr` in `trainer_round` telemetry = base × factor^((round − 1) // every) at 2-3 rounds; lr at the leg's
      last round 6-20% of base (source 13%), not at the floor early.
   5. *Fairness:* oracle `*` arms run about the same rounds / h as their non-oracle arm; a big gap means a stall or divergence, not speed.
   6. Then the speech run (step 4); after the first ~12 h, re-run checks 2-3 on the live legs before trusting the rest.
4. [ ] **Speech run (B2 after, operator 10-27):** full set `$PY scripts/paper_runs.py launch --prio 2 --datasets google_speech && bash <root>/launch_p2_speech.sh`
   (~26 async × ~11 h + 32 sync × ~4 h on 8 GPUs ≈ 2-2.5 days; less with target hits).
   *Now (operator 10-27): End-to-end only (44 legs), commands 1-4; Ladder / swaps for speech moved to priority 3.*
   - Pool S (shepherd 0-3, 4): `fx` × 4 cells.
   - Pool J (jayne 0-7, 30): every non-`smu` non-Felix leg, oracle arms first.
   - Pool M (shepherd 0-3, after the speech MobiPerf target; 10): every other `smu` leg, oracle arms first.
   *Cifar seeds 2-3:* future TODO (operator 10-28: no time); sync at every 50 if ever run (2e main).
5. [ ] **Tables as data lands:** `paper_runs.py status` + `paper_plots.py` (finished legs only); fill the outline's ⚠ numbers.
   **Paper table data:** `paper_felix/figs/expts/results_{cifar10,google_speech}.json` (`scripts/paper_json.py --root $R`; per cell × stack: time / GB / updates to target, accuracy at fixed hours, GB by fate, staleness, slot use, ratios vs Felix, ablations; `meta.complete`). Cifar complete (44 / 44); **speech: re-run after pools J and M finish (~10-28 evening) and commit**.
   GPUs free for the operator only in the speech tail (< 8 legs left).
6. Then (FX-N131, Run order rule): static (appendix); silent departures (FX-N155) need code first. **Future TODO (operator 10-28, no time), in order:** seeds 2-3 of End-to-end and Ladder, both datasets; then speech Ladder / swaps / rejected ideas (priority 3, skipped: cifar seed 1 has them).

Facts for the runbook: smoke = 2 h vclock, stops at `FB1S_TARGETS` (cifar 35%, speech 15%); paper legs stop at the 5-eval mean ≥
target (`targetAccuracy`, `targetWindow` 5; every stack, sync and async) or at 120 h vclock (6-day trace, legs start +24 h so each
client has its day-1 data); wall timeout 24 h / leg (smoke 1 h). Wall per 2 h vclock, speech bench: sync ~4 min, async 10-11 min.
Cifar bench (10-26): Felix ~6-7 min wall per vclock hour solo (12 h leg 72 min, `fn` 133 min; F49 the same), packed 4 / GPU
×1.5-1.8; output on NFS +9% (n = 1). A bench leg is not faster than a harness leg (speech 12 h: harness 33 min on 4 GPUs, bench
75 min on 1); the bench's gain is parallel legs (cifar 32 vs 1, speech 8 vs ~2). Layout `<root>/<cifar|speech>/<phase>/runs/<trace>_<baseline>_s<k>/`, phase `FB2<set>_<cell>_<arm>`.

- **FX-N157 · Felix dispatched to clients still computing · fixed (FX-D237); EV24 0 on run 1's 5 legs; exit = step 1 EV24.**
  The async clock-jump clamp committed an update before its sct and freed the client; a second path popped a newer dispatch's
  in-flight entry. After the exit: add EV24 to `EARLY` once it passes stored legs of all stacks (PL21).
- **FX-N160 · Evicted update committed late · fixed (FX-D238); exit = step 1 EV16.** Felix evicts a client going unavailable
  (`aware_boundary_eviction`); `free_stalled_slot` drops it from the gate's in-flight set, so the gate stopped waiting for its update.
  When that update arrived late in wall time the clock had passed its delivery: speech `smu` `fx` commit 59 s (bench) / 12.8 s
  (harness) after delivery, 1 of ~58 withheld per 12 h; cifar 0. Verify: the speech legs of `EV24_1025_ev2` (`N160_1025` was stopped unfinished).
- **FX-N151 · Sync-stack bench gap · fixed root (FX-D234); exit = step 3 speech B2.** Root: check-in reconnect jitter from one
  shared RNG in end order. Rest = sampling noise (REFL `smu` 4 h bench +12% updates / h, heavy straggler tail); `bench_validate.py`
  bars widen to either side's seed range (operator 10-25). Data: `experiments/N151_1025/`.
- **FX-N153 / N154 · Run pipeline and run sets · built (FX-D235, D236); exit = step 1 smoke.** `FB2E2E` … `FB2STA` = Table 1
  (152 legs / seed). Digests gzip raw logs; `bench_report`, `config_audit` read `.gz`, other readers don't yet.
- **FX-N159 · Per-dataset launch · built** (`paper_runs.py launch --datasets`).
- **FX-N161 · Speech bench legs OOM when packed · fixed (10-26).** A speech bench leg holds ~18-20 GB GPU (5 procs) but was
  declared `--leg-gpu-gb 10`, so pools stacked legs on one 44 GB GPU; startup OOM'd 2 speech smoke legs. Now 22 in `paper_runs.py`,
  `run_ev24.sh`, `run_b2.sh`; the FX-D239 watchdog also covers worker startup. Each pool sees only its own ramping legs: worst case
  per GPU 1 speech (20) + 4 cifar (4 × 3.4) = 34 GB fits, but no harness leg beside bench legs (FX-T91).
- **FX-N162 · Cifar run ETA · measured.** Wall per leg (DONE lines): sync 120 h 4-7 h; Felix / FedBuff hits 5-8 h; `uo_o` 9.6 h;
  `smu` 120 h ~18-22 h. Every leg fits the 24 h timeout, so FX-N158 is not needed for cifar; re-check for speech (async ~11 h / 12 h
  vclock, so a 120 h leg would exceed 24 h: raise `--timeout-s` or plateau-stop before step 4).
- **FX-N163 · Sync cifar stacks stalled at 48-56%: source lr decay too fast for our rounds · fixed in config (target-anchored decay, Decisions); exit = step 2c rerun ✅ (10-27).**
  Decay is per aggregator round (`main.py:884`): unscaled FedAvg lr 0.1 → 2% by 12 h (1,979 rounds), floor from ~22 h; Oort at 10% by
  12 h. Screen `experiments/N163_1026` (24 h, FedAvg · REFL): source decay 48.2 / 48.6%, none 72.9 / 74.3%, stretched (`horizon`) 71.8 / 72.7%.
  *Sources (read 10-26 from the repos and PDFs):* REFL `core/aggregator.py:599` ×0.98 every 10 rounds, floor 5e-5, `epochs=1000`
  rounds (`run_exps.sh:50`) → lr ends at 13% of base. Oort `learner.py:457` ×0.95 every 5 (speech 15), floor 1e-4, "1000 rounds in our
  paper" (configs 500) → floor by ~600 rounds. FedDance / FedAvg: "FedScale defaults" (1,000 rounds; its hyper-parameters are in a
  technical report, not the PDF) = REFL's 0.98 / 10; REFL's paper also "FedScale defaults, no tuning". FedBuff: none described.
  *Rule (operator 10-26, one for every sync stack):* each source schedule spans ~1,000 rounds, ours run longer, so `every` is scaled by
  (rounds a leg completes in 120 h) / 1,000. Measured on the decay-off cifar digests (versions / h × 120): REFL / FedDance / FedAvg
  ~18-23k → ×20 (every 200); Oort 0.9-1.2k → ×1 (every 5, source-faithful). Speech (F49: ~14 rounds / h ≈ 1.7k): ×2 (every 20; Oort 30).
  Speech reads the same `baseline_reference.yaml` overlay, so it needs no extra step; re-derive its factor from the FB2 speech digests
  and rerun only if the rounds differ by > 2×. Paper line: "we scale FedScale's lr schedule to our longer round horizon, uniformly
  for every baseline". *Superseded 10-27 by the target anchor (Decisions): every 120 (2c), then 100 (2d, operator).* *Checks:* trainer lr = base × factor^((round − 1) // every) (`main.py:884`); verified against the unscaled legs'
  `trainer_round` lr (round 1,698: 0.00329 = 0.1 × 0.98^169); the overlay gives every = 200 / 20 / 5 / 30 (10-26). Cifar ×20 lr at
  rounds 5k / 10k / 20k / 23k: 60 / 36 / 13 / 10% of base; the floor (5e-5) needs ~75k rounds, never reached in 120 h. Past a leg's
  budget the decay simply continues. Oort cifar (0.04, ×0.95 / 5) floors at round ~585 (~60-80 h at 7-10 rounds / h), as its source does.
  Speech floors are far off (REFL ~6.8k, Oort ~3.5k rounds vs ~1.7k). *One factor per config, not per stack:* REFL / FedDance / FedAvg
  share FedScale's one schedule and differ in rounds (REFL ~190 / h, FedDance ~146, FedAvg ~150), so ×20 ends at 10 / 17 / 17% of base
  lr at 120 h; a per-stack factor is a tuned knob per baseline, and rounds / h swing 3-5× across cells (FedDance 33-205 / h), so no
  fixed value equalizes them. *Left:* REFL's ~90% needs ResNet18 (ours: CifarNet, ceiling 89.0%) and "1 epoch" ≈ 17 steps vs our 1 step.
- **FX-N166 · FedDance / REFL match Felix to 80% on cifar syn (interim decay-off run) · open: claim decision after step 2b; FedAvg divergence rooted, gate pending.**
  (a) Time to 80% on `s0u / s20u / s50u`: FedDance 60 / 59 / 63 h, REFL 62 / 68 / 71 h vs Felix 65.3 / 65.8 / 72.0 h (GB 776 / 731
  / 724 vs 2,334 / 2,549 / 3,249). Felix is no longer first and spends 3-4.5× the GB there; `smu` none reaches 80% (Felix 75.1,
  REFL 76.5, FedDance 76.1%). Seed 1, n = 1: confirm on seeds 2-3 before the paper claim changes. (b) **FedAvg's 3 early stops = divergence, rooted
  (10-26):** `s0u` round 9,723 (58.8 h; last eval 10% = chance, best 78.7%), `s20u` 5,979 (37.8 h), `s50u` 10,070 (71 h). Client
  `stat_utility` grows 4 → 300 → 10⁴ → 10¹² over ~8 rounds (a feedback blow-up, not one outlier client), then the model is non-finite,
  every update is rejected (`nonfinite_utility`, ~294 per leg, FX-D111) and the run ends as `work_done`, which the digest read as a
  plain early stop. Cause: constant lr 0.1 + momentum 0.9, 1 step, no clip and no decay (the decay-off run); correction 10-26 21:00: 10 / 32 legs
  diverged (FedAvg 4 / 4, FedAvg\* 2, FedDance\* 3, REFL\* `smu`; step 2a), so the Scoreboard's ✗ for those are frozen-at-divergence. Fix: the scaled decay lowers lr (0.1 → 0.04 at 9k rounds, 0.01
  at 23k); unproven, so step 2b gates it on the 4 FedAvg legs first, with Oort's recorded clip as the fallback. Divergence is now
  flagged by the digest (patch outside the tree, step 2b). (d) *Why sync stacks do not stall on unavailability (FedDance `s50u`, digest):* a round is ~20 s (1 step × 24 samples,
  2.15 MB model, ~10 s compute + ~10 s comm), so only 1.5% of tasks see the client leave mid-task (2,593 of 168k) and 1.8% time out;
  the 1.3× over-commit absorbs those (34k extra picks discarded, 20% of the GB); 8,750 rounds in 63 h (~140 / h) vs the papers'
  1,000. Not a check-in margin: the oracle `*` arms are no better (REFL\* 61 / 64 / 71 h vs REFL 62 / 68 / 71; FedDance\* ✗). The
  setup favors short tasks (tiny model, one step) while Felix / FedBuff run 1 epoch per task and so meet departures far more often;
  speech (29 MB model, round 256 s) is where sync stacks are round-bound (Decisions). (c) Oort / Oort\* stay 73-76% at 120 h with decay off (cost 61-97 GB); redo on the rerun.
- **FX-N165 · FB2 cifar swaps beat `fx` on `s50u` · open: operator decision (`fx` is frozen).** Same run, seed 1, n = 1:
  - c: `fx_cfix` (c 30) and `fx_ca` reach 80% at 63.8 h vs `fx` 72.0 h, with 1.2k vs 3.2k GB (`fx` holds c 60 all run: ½ of ~150
    eligible > 60, so the ramp never binds on `s50u`). `smu`: `fx_ca` (mean c 11) ends highest, 77.4% vs 75.1% at 589 vs 2,522 GB.
  - Availability: Felix by check-in `fx_ck` hits at 73.5 h (+2%) with −79% GB.
  - vs FedBuff on `s50u`: Felix 1.13× sooner but +34% GB. The `−27-55% GB` claim (F49) no longer holds at 80%.
  - *Next:* seeds 2-3 of `fx` / `fx_cfix` / `fx_ca` on `s50u` before any change; read speech (step 4) for the same pattern.
- **FX-N164 · Post-run tool fixes · todo (after the cifar run; T15).** (a) `config_audit.py:169` strips `_r\d+$` as a replicate
  suffix, so arm `fx_r1` is audited as `fx` → `SWG smu fx_r1` stale on `rp=1.0/1.0/0.2` (the arm's intended rate; leg fine): exempt
  arm names, then re-digest. (b) `bench_pool.py` never reaps finished children (59 zombies). (c) `paper_plots.py` swap figures: y
  labels and x ticks overlap the neighbour panel. (d) `paper_plots.py`: the 60-80% target-sweep figure (the per-trace
  hit is in the digest now, Decisions "Reporting target per trace"; plots read `hit`).
- **FX-N155 · Robustness: silent departures · todo (code, priority 3).** With probability p a client's flip to not eligible is not
  reported; Felix holds the slot until `send_timeout_wait_s`. p ∈ {1, 5, 10}% on the ablation cells. Lost / late reports, lost
  uploads, sensitivity: after the first version.
- **FX-N156 · Speech 2 legs / GPU · todo (optional).** Measure `--per-gpu 2` wall and accuracy-identity on one cell; raise
  `--timeout-s` (24 h) with it.
- **FX-N158 · Plateau stop · parked (operator 10-25 late: only if long legs become a problem).** Must use the leg's clock (a 120 h leg
  is ≤ 13 h wall): stop when the best 5-eval mean has not risen ≥ 0.5 pp in 24 h vclock; ✗ + plateau accuracy; same rule for all.
- **Availability swap forecast arm · not built** ("if cheap", outline Table 1); first plots run check-in and oracle only.

**B · Standing.**
- **FX-N132 · General mechanisms only (operator 10-22).** No per-dataset / n / K / c constants in Felix's mechanisms.
- **FX-N147 · Eligibility is hard to forecast · ✅ (outline §0, Fig 2).** `scripts/eligibility_forecast.py` →
  `paper_felix/data/fx_n147_*.json`. *Left:* log FedDance V_m per pick; re-check on the paper trace window.

**C · Parked / backlog (after the paper runs; one line each).**
- **D5** parity re-grade on the streaming setup (G3 reals + sims) · **FX-N108** claims gate · **FX-N93** convergence parity ·
  **FX-N12** sim-only sign-off (all behind D5).
- **FX-N129** bench gate: B3 (`predictor_audit.py`) left · **FX-N135** bytes via local work (speech only) · **FX-N136** speech
  current-model swing · **FX-N137** buffer composition · **FX-N127** pacer · **FX-N128** aggregation V4 · **FX-N100** throughput ·
  **FX-N94**, **FX-N85**, **FX-N90**, **FX-N92**, **FX-N91** (EuroSys plot debts), **FX-N42**, **FX-N10**, **FX-N22**, **FX-N2**.

---

## Scoreboard (rewrite after every run, PARITY C5)

**Now (10-28 01:40):** speech E2E seed 1: pool S done (Felix 4 / 4), pool J 10 / 30 (table below); cifar priorities 1-3 70 / 70 + sync reruns 2c / 2d / 2e (every 120 / 100 / 50; **2e = main**), 0 diverged;
B2 cifar 4 / 5 PASS + oort marginal (upd +5.3% vs 5%); speech B2 after speech E2E. **Open reds:** REFL / FedDance beat Felix on `smu` by 16-21 h at ~½ the GB (FX-N166);
swaps beat `fx` on `s50u` (FX-N165); FedDance\* ✗ 66-78%; Oort family ✗ 69-76% (6-11 rounds / h); End-to-end cifar `stale: code key`.
**Remaining speech E2E (10-28 02:38): 16 / 44 done, 8 running + 10 queued (pool J), 10 pool M (launching). Everything else = future TODO (seeds 2-3, then speech ablations; static run set not scheduled). Next:** launch pool M now (`tmux new -s sp_M`, `bash $R/launch_p1_speech_M.sh`; shepherd idle; target 84%, Felix hits it at 97.8 h); pool J ends ~10-28 12:00; then B2 speech, seeds 2-3.

**Latest result: FB2 speech End-to-end, partial (10-28 01:40, seed 1, 88% target, `smu` target pending, same root `$R/speech`; pool S done, pool J 10 / 30).**
h · GB to 88%, else ✗ final % at 120 h; (↻ n %) = running, accuracy at that sim hour. Sync stacks K 5 (rule), lr decay unchanged (20 / 30). `*` = oracle.

| cell | Felix `fx` | FedBuff\* (FedBuff) | Oort\* | REFL\* | FedDance\* | FedAvg\* |
|---|---|---|---|---|---|---|
| `s0u` | 70.9 · 1,499 | 101.2 · 2,081 (↻ 85% @ 63 h) | ✗ 75.6 | ✗ 41.0 | ✗ 55.2 | ✗ 54.3 |
| `s20u` | 68.6 · 1,575 | ↻ 86% @ 92 h | ✗ 68.9 | ✗ 39.0 | ✗ 53.0 | ↻ 51% @ 116 h |
| `s50u` | 66.2 · 1,762 | ↻ 84% @ 81 h | ✗ 53.5 | ✗ 39.2 | ↻ 48% @ 97 h | ↻ 39% @ 58 h |
| `smu` (84%) | 97.8 · 1,378 (peak 5-mean 84.2%) | queued in M | M | M | M | M |

**Trends:** Felix hits 88% on all 3 synthetic cells in 66-71 h, ahead at every checkpoint (63 / 77 / 84 / 87% at 10 / 20 / 40 / 60 h vs FedBuff\* 52 / 71 / 81 / 84% on `s0u`):
**1.43× sooner, −28% GB** on `s0u`; `s20u` / `s50u` FedBuff\* is on track to hit later than 92 / 81 h (the 1.3-1.5× window again). No sync stack nears 88%: Oort\* is best (76 / 69 / 54%) and
still climbing at 120 h (not plateaued, ~1 pp / 4 h); REFL\* / FedDance\* / FedAvg\* sit at 39-55% despite 2-3× Oort\*'s rounds. Sync speech is slow rather than diverged (0 diverged; K 5 = 3-18k updates / 120 h). Deployable arms
(Oort, REFL, FedDance, FedAvg, FedBuff = `ctl_srv` × fedbuff, needed) still running. Oracle and deployable sync arms share one schedule per stack (Oort 0.95 / 30, others 0.98 / 20; operator 10-28: final, no retune), so the deployable arms should land at or below their `*` arm.
Figures `curves` / `headline` / `efficiency` redrawn with speech partial (PRELIM; re-run `paper_plots.py` for the `smu` Felix hit).

**Previous result: FB2 cifar End-to-end (10-27, seed 1, 80% target, `smu` 75%, root `/coc/scratch/dgarg/mlsys27_felix_runs_20261026`).**
h · GB to target, else ✗ final % at 120 h. **Main = 2e, sync stacks at lr decay every 50** (`$R/cifar_e50`, 32 / 32 ok, 0 diverged, 10-27 17:00-21:45, shepherd 4-7);
2d (every 100) and 2c (every 120) below are ablation. `*` = oracle arm. Felix / FedBuff are async (same in all three tables).

| cell | Felix `fx` | FedBuff · FedBuff\* | REFL · REFL\* | FedDance · FedDance\* | FedAvg · FedAvg\* | Oort · Oort\* |
|---|---|---|---|---|---|---|
| `s0u` | 65.3 · 2,334 | 77.8 · 3,464 · 72.3 · 3,443 | ✗ 75.0 · 74.9 | ✗ 75.8 · 73.0 | ✗ 74.3 · 74.6 | ✗ 77.7 · 73.6 |
| `s20u` | 65.8 · 2,549 | 80.8 · 2,900 · 81.7 · 3,080 | ✗ 75.2 · 74.4 | ✗ 75.9 · 73.4 | ✗ 74.6 · 74.5 | ✗ 76.1 · 73.5 |
| `s50u` | 72.0 · 3,249 | 81.0 · 2,422 · 80.5 · 2,542 | ✗ 74.8 · 74.7 | ✗ 76.2 · 73.0 | ✗ 74.7 · 74.2 | ✗ 74.8 · 74.7 |
| `smu` (75%) | 95.2 · 2,277 | ✗ 73.6 · 73.9 | ✗ 68.3 · 70.0 | ✗ 70.9 · 60.9 | ✗ 68.5 · 67.5 | ✗ 69.0 · 71.3 |

**Takeaway:** every 50 over-decays (lr 0.98^(round / 50) falls ~2× faster than 2d): sync stacks plateau at 73-78% on syn and 61-71% on `smu`, so no sync
baseline reaches the target and Felix is first on every cell (vs FedBuff 1.1-1.2×, GB as in 2c). Oracle arms do no better (FedDance\* 60.9-73.4%). **Caveat:**
this rests on a harsher decay than 2c / 2d, where FedDance / REFL are within 1.1-1.4× of Felix on syn and beat it on `smu`; report 2d / 2c beside it so the sync
baselines are not read as crippled.

**2c, every 120 (ablation; code `ccb25f6fc`, 10-27 02:47-07:25).**

| cell | Felix `fx` | FedBuff · FedBuff\* | REFL · REFL\* | FedDance · FedDance\* | FedAvg · FedAvg\* | Oort · Oort\* |
|---|---|---|---|---|---|---|
| `s0u` | 65.3 · 2,334 | 77.8 · 3,464 · 72.3 · 3,443 | 86 · 1,237 · 88 · 1,309 | 82 · 1,054 · ✗ 77.7 | ✗ 79.9 · 112 · 1,724 | ✗ 76.1 · 72.8 |
| `s20u` | 65.8 · 2,549 | 80.8 · 2,900 · 81.7 · 3,080 | 95 · 1,171 · 94 · 1,340 | 74 · 921 · ✗ 77.1 | 93 · 1,208 · 103 · 1,573 | ✗ 75.3 · 72.7 |
| `s50u` | 72.0 · 3,249 | 81.0 · 2,422 · 80.5 · 2,542 | 117 · 1,515 · 98 · 1,391 | 81 · 934 · ✗ 77.2 | 114 · 1,352 · ✗ 79.8 | ✗ 74.7 · 74.7 |
| `smu` (75%) | 95.2 · 2,277 | ✗ 73.6 · 73.9 (6.0-6.2k GB) | 78 · 1,199 · 74 · 1,193 | 79 · 1,164 · ✗ 66.4 | 96 · 1,431 · ✗ 72.7 | ✗ 69.1 · 70.7 |

**2d, every 100 (ablation; code `0c67829f9`, 32 / 32 ok, 0 diverged, 10-27 10:52-15:38 shepherd 0-3, 286 min).** Felix / FedBuff rows unchanged (async).

| cell | Felix `fx` | FedBuff · FedBuff\* | REFL · REFL\* | FedDance · FedDance\* | FedAvg · FedAvg\* | Oort · Oort\* |
|---|---|---|---|---|---|---|
| `s0u` | 65.3 · 2,334 | 77.8 · 3,464 · 72.3 · 3,443 | 111 · 1,592 · ✗ 79.8 | 91 · 1,169 · ✗ 76.3 | ✗ 79.2 · ✗ 79.3 | ✗ 77.2 · 74.1 |
| `s20u` | 65.8 · 2,549 | 80.8 · 2,900 · 81.7 · 3,080 | 90 · 1,128 · 120 · 1,721 | 91 · 1,120 · ✗ 77.0 | ✗ 79.3 · ✗ 79.4 | ✗ 76.0 · 74.7 |
| `s50u` | 72.0 · 3,249 | 81.0 · 2,422 · 80.5 · 2,542 | ✗ 79.0 · ✗ 79.9 | 96 · 1,110 · ✗ 77.0 | ✗ 79.6 · ✗ 79.1 | ✗ 75.0 · 74.4 |
| `smu` (75%) | 95.2 · 2,277 | ✗ 73.6 · 73.9 | ✗ 73.6 · 83 · 1,335 | 80 · 1,173 · ✗ 65.1 | ✗ 72.2 · ✗ 71.9 | ✗ 69.6 · 70.6 |

Felix fastest on the 3 syn cells vs every baseline incl. oracle arms (vs FedDance 1.40× / 1.38× / 1.33×; vs FedBuff 1.1-1.2×), at 2-3× the GB
of sync. **Not on `smu`:** FedDance 80 h and REFL\* 83 h beat Felix 95 h at ~½ its GB (FX-N166, open). Every 100 slowed sync vs 2c (REFL, FedDance +5-24 h;
REFL / FedAvg now miss 80% on `s50u`).

Felix vs best sync (2c; FedDance): `s0u` 1.26× sooner, `s20u` 1.13×, `s50u` 1.13×, at 2.2-3.5× the GB; on `smu` FedDance / REFL / REFL\* hit
75% at 74-79 h vs Felix 95 h at ~½ its GB (2b had FedDance ahead on `s20u` / `s50u`; FX-N166). Felix vs FedBuff: `s0u` 1.19×, −33% GB;
`s20u` 1.23×, −12%; `s50u` 1.13×, **+34%**; on `smu` FedBuff never reaches 75%. Swaps and ladder: FX-N165, `paper_felix/figs/expts/rejected.md`.

**Previous: F49 (10-24; `experiments/F49_1024_1530`, code `0f6858e70`; bench, 12 h, seed 1, averaged model; old 90 /
95%-of-ceiling targets; MobiPerf Felix tainted by FX-N157).** final % · min / GB to the top target Felix and FedBuff both reach.

| cell | Felix `fn_chr` | FedBuff | best sync / Oort-family | verdict |
|---|---|---|---|---|
| cifar `s0u` | 74.4 · 68.3%: 270 / 260 | 71.2 · 530 / 569 | Oort 52.4 | PASS (1.96×, −54% GB) |
| cifar `s50u` | 73.2 · 67.8%: 250 / 303 | 69.0 · 650 / 465 | Oort* 51.3 | PASS (2.6×, −35%) |
| cifar `smu` ⚠ | 59.1 · 55%: 230 / 314 | 56.3 · 350 / 571 | FedDance 47.4 | PASS (1.52×, −45%) |
| speech `s0u` | 63.5 · 50%: 376 / 177 | 53.3 · 580 / 242 | FedDance* 17.1 | PASS (1.54×, −27%) |
| speech `s50u` | 64.5 · 40%: 256 / 145 | 44.0 · 613 / 139 | FedDance* 15.0 | PASS (2.4×, +4% GB) |
| speech `smu` ⚠ | 52.8 · 45%: 535 / 265 | 46.4 · 676 / 295 | FedDance* 21.1 | PASS (1.26×, −10%) |

- Below ~50% Oort and the 1-step sync stacks beat Felix on bytes at 4-12× its time; speech sync stacks sit near chance for 4-6 h.
- Speech FedBuff led Felix for the first ~2.5 h on `s0u` (F49); the FB1S smoke had Felix 22% vs FedBuff 16% at 2 h: re-read on FB2.
- FX-N149: speech `fx` and FedBuff bench match the harness (AUC 39.3 / 39.0, updates / h −3%, GB −3%).

### Running config and invariants checked

- *Shapes:* n / K / c in "Cohort and participation" (top). GPUs: cifar screens 1 A40, n 300 bench 1 GPU / leg; speech 2-4. Eval
  every 10 min of the leg's clock on the full test set; speech bf16 (FX-D168); FB1 legs stop at the target's 5-eval mean (`targetAccuracy`, FX-D229; `targets.json`).
- *Task time:* FedScale device model (FELIX_INSIGHTS §6). *Timeout:* `send_timeout_wait_s` per baseline (FX-D198; cifar 903 s
  n 75, FedBuff n 300 677 s; speech 2560 s) for every non-Felix baseline; Felix evicts at the flip. *Knowledge (R27):* Felix notify; non-`*` baselines check-in; `*` oracle. AVL_EVAL is
  eval-only and Felix-only (FX-D165).
- *Training / server knobs (`baseline_reference.yaml`; source-faithful or recorded deviation, L33-L37):*

  | baseline · dataset | local work | batch | client lr / opt | mom · wd | lr decay | server | utility | status |
  |---|---|---|---|---|---|---|---|---|
  | felix · cifar / speech | 1 epoch | 32 / 16 | 0.04 SGD | 0 · 0 | none | fedbuff, lr 1.0 (FX-D185), rate `new` (scale 0.4, a 0.25, b 0.2) | fedscale | final (`fx`, Decisions) |
  | fedbuff · cifar / speech | 1 epoch | 32 / 16 | 0.04 SGD | 0 · 0 | none | fedbuff, lr 1.0, rate 1/√(1+s), lr batch-normalize | – | final (FX-D73, FX-N74) |
  | refl · cifar / speech | 1 step | 10 / 20 | 0.1 / 0.05 SGD | 0.9 · 5e-4 | 0.98 / 100 · 20 (source 10; FX-N163) | REFL SAA (speech: FedAvg server, recorded) | – | final (deviations recorded) |
  | oort, oort_star · cifar / speech | 20 steps | 30 / 15 | 0.04 SGD | 0.9 · 5e-4 | 0.95 / 100 · 30 (source 5 · 15; FX-N163) | FedAvg-YoGi η 0.005 / 0.002 (cifar clip 1.0) | fedscale | final (deviations recorded) |
  | feddance, fedavg · cifar / speech | 1 step | 24 / 32 | 0.1 / 0.075 SGD | 0.9 · 5e-4 | 0.98 / 100 · 20 (source 10; FX-N163) | FedAvg equal | – | final (deviation recorded) |

  Felix (`fx`) vs FedBuff differ only in availability (notify / evict), selection (`prop` utility), rate (`new`) and c (ramp);
  buffer, client recipe, lr normalization, augmentation and the averaged model are FedBuff's / every stack's.

---

## Decisions (operator; standing until changed here)

- **Reference cohort (10-23):** targets at n 300 cifar / 100 speech.
- **Nodes (operator 10-26, supersedes 10-25 late):** our default is **shepherd GPUs 0-3** (4 of 8; the rest are Aish's fwdllm runs,
  `~/aish_test`: check `nvidia-smi` first, don't interfere). **Jayne: only GPUs the operator names, ask first** (its 0-3 are Aish's too;
  jayne needs the same pushed commit). Launch scripts default to shepherd 0-3 and refuse other hosts without `GPUS` (`HGPUS` +
  `BGPUS` for B2 / EV24). Run from `/home/dgarg39/flame` (no extra worktrees); leave flame clones we did not create alone (ROBUST L38).
- **Felix new = `fx`, final and frozen (operator 10-25; F44-F49).** Never edit it: a new idea runs as its own arm or ablation row
  against `fx`. One config for cifar and speech; it lives in `baseline_reference.yaml` (felix server) + `harness_pool.FELIX_FINAL`
  (arm `fx`; `config_audit` row `rp 0.4/0.25/0.2`). Felix old (`ctl`) shares the rate constants.

  | layer | `fx` | rejected (evidence) |
  |---|---|---|
  | availability | notify on every flip + evict mid-task (C1) | check-in: FedBuff ~33 of 60 slots busy on `s50u` (F44); Felix by check-in −4.4 / −10 AUC (F42b) |
  | selection | `prop`: never-trained first, then ∝ \|D\| × Oort EMA loss q among online clients | speed term −1.5-2 pp final (F45); + balance loses 25 / 28 pairs (F46); Oort cutoff lowest final (F45); UCB ±0.3; eval tasks tie at +257k starvation (F48L) |
  | aggregation | FedBuff buffer K 10; rate `new` w = 0.4 (1+τ)^-0.25 + 0.6 (1 − (1+loss)^-0.2) | 1/√(1+τ) 2× slower to 50%, 1/(1+τ) slowest (F46); 2024 b 0.1 1-3 ticks slower; a 0 as fast but no staleness term (F48R) |
  | step | server lr 1.0; lr batch-normalize (equal per-sample weight) | φ-scaled step −0.9-3.2 pp (FX-T85) |
  | concurrency | c ramp: 60 (FedBuff's c / n) → min(60, ½ trainable) as loss phase φ falls, ≥ 10 (`felix_c` `pool=0.5 ramp`); speech ≤ 20 | c 60: −1.1 pp at 2.2× GB on MobiPerf (F48L); c ∝ availability / ≤ ⅓-½ pool −3-15 pp early (F48a); c by φ alone floors at 10 (FX-T89); c 30, K 20 (F46-F47) |
  | local work | 1 epoch, client SGD lr 0.04, batch 32 / 16, no wd | e2 / e3 −1.3-2.7 pp (F47); wd ±0.5 (F44) |
  | data · serving | per-batch augmentation and the averaged model (EMA β = 1 − K/n), both on every stack | frozen draws plateau at ~60% (F47) |
- **Augmentation (10-24, FX-D233, FX-N143; adopted on principle: the baselines' own pipelines do it, so every stack gets it):**
  default on for all six stacks, both datasets, and the ceiling. Paper defense:
  - *What:* each time a device trains on a sample it draws a fresh random variant ("fresh view") of the dataset's standard train
    augmentation, on its own copy, with the label unchanged: cifar crop (pad 4) + horizontal flip (162 variants per image); speech
    FedScale's pipeline on the waveform (amplitude ×0.7-1.1, speed / pitch ±20%, shift ≤ 0.256 s, background noise ≤ 45%, each
    p 0.5; `fl_data.augment`; FedScale's STFT stretch omitted, speed covers duration). The test set is never augmented.
  - *Standard practice:* the baselines' own code augments exactly so (FedScale / REFL `third_party/REFL/core/utils/utils_data.py:31-33`
    cifar, `fllibs.py:234` speech), as does FedAvg's CIFAR setup (McMahan et al. 2017; verify cite). Our loader always declared it;
    the GPU preload froze one draw per sample (FX-D233), so this is a bug fix applied to everyone, not a Felix choice.
  - *Realistic on devices:* a few tensor ops per batch on the device's own data: 0.8 ms (cifar, 32) / 0.6 ms (speech, 16) on a GPU,
    small next to a training step; no extra bytes, no server role, nothing new leaves the device. Uncharged in the FedScale task
    time for every stack alike.
  - *Why it helps:* async FL revisits streamed samples hundreds of times (F47 `s0u` Felix: ~230 passes per visible sample in 4 h);
    with one fixed view clients memorize (received-model loss 0.03-0.13, test loss 1.3 → 2.2-3.3 at flat accuracy). A fresh view
    per revisit acts like a new example, so test loss stays < 1 and accuracy rises +10-17 pp. Felix gains most in time because it
    revisits most.
  - *Privacy:* no claim. It changes no message and gives no formal guarantee; less memorization narrows the generalization gap
    membership inference exploits (Yeom et al. 2018; verify cite), but attacks adapt to augmentation.
  - *Targets:* the ceiling rises too: `ceiling.py` trains with the same augmentation (`--no-augment` = old), so targets re-derive
    from it and every stack is judged against the same, higher bar.
- **Holdout (10-22):** every stack trains on 100% (F49 audit: holdout off on all 56 legs); an arm that reads held-out data holds out
  10% of its own (none today). Ceilings use 100% shards.
- **Data stream (10-22):** 6-day `usage` stream tied to each leg's own trace; screens keep it and raise n instead of compressing time.
- **Targets and comparison (10-24, supersedes 10-22's 80 / 90 / 95%):** one final target per dataset = the FL upper bound (ideal
  FedAvg, FX-N146); the converged centralized ceiling is reported beside it; no lowered fractions. Hit = first 5-eval mean ≥ target.
  Every experiment (FB1, ladder) runs to this plateau, not to a published target (operator 10-25): the cohort holds ~all train
  samples (cifar 49.3k of 50k, speech 84.0k of 84.8k), so the bound is not inflated by small n; it is stricter than FedBuff's CIFAR-10 60% (its Table
  2; other model). Plain SGD as every stack (no momentum / server averaging). Oort / REFL speech targets: unverified.
  Paper runs (FB1): every stack on the 6-day trace (legs start +24 h, so ≤ 120 h) until it hits the target; one that never does
  reports accuracy, time and GB at the trace end. Report at the hit, not at a horizon: time and GB vs the fastest deployed
  (check-in) baseline (× sooner, × fewer GB), tables / bars normalized to the best stack, and each stack's accuracy and GB at Felix's
  hit. Oort et al. are compared once long runs let them reach the target. **Values:** cifar n 300 **80.0%** (FL bound, 10-25;
  centralized 89.0%); speech n 100 **88%** (operator; FL bound 87.5% at round 835 and rising; centralized 94.5%). Both in
  `_metadata/targets.json` (the legs' stop target).
- **Reporting target per trace (operator 10-26; refines the above for tables, bars, speedups and the legs' stop):**
  - *Rule (fixed before the sync rerun, applied to every cell):* the highest accuracy any stack holds (5-eval mean) within the 120 h
    trace, rounded down to 5 pp, capped at the dataset's FL bound. Baselines reach the FL bound on the less difficult deployments;
    on MobiPerf (high heterogeneity, trainable 18-22%) only a lower level is attainable in the trace window, so that level is the
    representative target. Paper wording: "on MobiPerf we report the time and data to 75%, the highest level any system sustains".
  - *Values:* cifar syn traces **80%**; cifar MobiPerf **75%**; speech **84%** (operator 10-28: Felix `fx` peak 84.2%, no 5 pp rounding; legs stop at the hit, a miss reports 120 h accuracy).
  - *Evidence (10-26, peak 5-eval means on cifar `smu`, seed 1):* Felix `fx` 76.0% (hit 75% at 95.2 h), best Felix variant 77.8%
    (`fx_ca`), FedBuff 74.1%, FedBuff\* ≤ 74.6%; sync stacks void until the scaled-decay rerun (interim decay-off: REFL 76.5%, FedDance
    76.1%, Oort 69%, none near 80%). Best stack 77.8% → 75%, below the 80% bound. Re-check on seeds 2-3; if 75% is missed, 72%.
  - *Why stop there:* MobiPerf legs otherwise ran the full 120 h (16-19 h wall each) for a target nobody reaches; the stop saves wall
    on the 8 MobiPerf legs of the sync rerun and any seed 2-3 runs. Same 5-eval-mean rule as every other leg (`targetWindow` 5).
  - *Code (10-26):* `_metadata/trace_targets.json` ({dataset: {trace: target}}); `harness_pool.trace_target(ds, trace)` sets each
    FB1 / FB2 leg's `targetAccuracy` (smoke tiers keep `FB1S_TARGETS`); `leg_digest.report_target` lowers a digest's hit to the trace
    target, so the 12 finished `smu` legs were re-digested (hit = 75%) and killed legs count as hits. Syn traces and speech unchanged.
    The 80% `targets.json` stays the FL-bound target and the 60-80% target-sweep figure (FX-N164d) is still todo. Already-run
    `smu` legs ran at the 80% stop (full 120 h), the same trajectory the 75% stop would have followed to its hit (stopped legs aside).
  - *Open:* `uo_o` / `b1` / `fn` were stopped at the hit, so their end-of-trace accuracy and GB are not available (ladder reads time and
    GB at the 75% hit only).
- **Streams = setting, not a claim (10-25):** title names availability only. F47 65%: `s0` 1.9× / −58% GB ≈ `s0u` 2.0× / −55%;
  gain grows with unavailability (`s50u` 2.7×); c by phase, the one stream mechanism, failed (FX-T89).
- **Fast bench (10-22, 10-25):** every paper run is bench, on cells green against the harness (B2): cifar ✅; speech `fx` + FedBuff
  ✅ (FX-N149), speech sync = runbook step 3.
- **Task time = FedScale device model (10-20, FX-D197):** permanent; results with the per-client D are void.
- **FedBuff K (10-20):** K = 10 (its paper's choice); K / κ tuning is Felix's only.
- **Baseline cadence (10-21):** a baseline with no code change and no open question is not re-run (FB1 reuses nothing from screens).
- **Baseline availability knowledge (10-21, R27):** each baseline learns availability only by its deployed mechanism (check-in, Bonawitz
  2019 / Papaya; REFL adds its on-device Prophet forecaster, hold-off off at our n); `*` = oracle, labelled inflated; FedBuff* and Felix
  by check-in (K1) are extra arms. Fairness is a guardrail, not a claim. Prophet 1.1.6 + cmdstanpy 1.2.4 in `dg_flame`.
- **Baselines (10-08, 10-10, 10-12):** source-faithful values in `baseline_reference.yaml` (ROBUST L33-L37); cifar Oort clip 1.0 recorded
  (FX-D126); FX-D148 partial dispatch for refl / oort / feddance; REFL staleness unbounded.
- **Sync stacks: client lr decay scaled to our round budget (operator 10-26, FX-N163; L37 deviation, called out in the evaluation):**
  REFL, Oort / Oort*, FedDance / FedDance*, FedAvg / FedAvg*, both datasets. The FedScale schedule (×0.98 per 10 rounds; Oort ×0.95 per
  5) spans the sources' ~1,000 rounds; our 120 h legs run ~18-23k (cifar REFL family), so `every` is scaled by rounds / 1,000, one rule
  for every baseline (FX-N163 has the factors). Unscaled it stalled every sync stack at 48-56%. Felix and FedBuff run no decay (FedBuff's
  paper describes none), as in their designs. An interim decay-off run is in the Scoreboard; the rerun (2b) gives the main numbers.
  *Target anchor (operator 10-27; supersedes the budget rule below for any family with a decay-off crossing):* the source schedule
  ends at 13% of base lr (0.98^100) after its 1,000-round run; we place that end where a constant-lr (decay-off) run of our workload
  reaches target: per leg every = (decay-off rounds to target) / 100, one value per family (median), legs without a crossing take it.
  *Why not the budget (every 200):* it ties the schedule to our 120 h horizon, not the workload. *Why not epochs:* the source runs only
  ~2 CIFAR epochs (1,000 rounds × K 10 × batch 10-24), so an epoch-faithful translation is every ~7 and stalls (like unscaled). *Why
  decay-off:* it removes the circularity (a decayed leg's crossing depends on its own every); constant lr crosses earliest, so this is
  the fast-decay end. *Cifar (decay-off digests 10-26 15:06, `$R/n163_every/off_cross.py $R/cifar`):* REFL 118 / 112 / 121 / 139 (`s0u` / `s20u` /
  `s50u` / `smu` 75%), REFL\* 120 / 120 / 132 / 132, FedDance 94 / 89 / 87 / 123, FedAvg\* 132 / 149 / ✗ / ✗; FedAvg diverged before
  every crossing, FedDance\* never crossed → median **every 120** for REFL, REFL\*, FedDance, FedDance\*, FedAvg, FedAvg\* (15 / 17
  within ±25%; FedDance 87 / 89 just below); **operator 10-27: every 100 (2d) for every sync stack.** *Oort / Oort\*: same every (operator 10-27):* no decay-off leg reached target in
  120 h (Oort 74.8-76.1%, Oort\* 72.7-73.6% syn; 0 diverged), so its own anchor only bounds every ≥ 9 (floor at the run end); 120 is
  inside that bound, so one every for every sync stack follows the rule with no per-stack exception. Oort keeps its factor (0.95) and
  floor; at its ~0.8-1.4k rounds lr ends at 57-70% of base, near decay-off (the most generous; every 9 ended at 59-66%). *Sensitivity (in-process FedAvg recipe, cifar n300 K 15,
  20k rounds, seed 1, `$R/n163_every`; sim lacks augmentation, plateaus ~70%):* every 20 / 50 / 100 / 200 / 400 → plateau 62 / 68.5 /
  68.5 / 69.5 / 70.0%, rounds to 65% ✗ / 2,750 / 2,750 / 2,250 / 2,250, 0 NaN: faster decay never reaches target sooner, only lowers
  the plateau. Report it in the appendix with 2b's every-200 legs.
  *Procedure (operator 10-26; cifar did it in 2a, speech repeats it in 3c):* (1) prove no baseline diverges (in-process arms: no
  decay / decay / clip; grep live legs for `NONFINITE_REJECT` and 0-rounds / h stretches, which make a leg look slow); (2) rounds in
  120 h per baseline × cell = digest `updates.versions` / `end.h` × 120, stable legs only; (3) per-leg `every` = source `every` ×
  rounds / 1,000 (lr then ends at the source's end fraction, 13% for FedScale); (4) one value per family when the per-leg values sit
  within ~±25% of it, else split the family; (5) after the run, re-check (2) on its digests. Cifar result: REFL, REFL\*,
  FedDance, FedAvg, FedAvg\* per-leg 167-273 → **every 200** (lr ends at 6-19%); **FedDance\* 27-38k rounds (oracle avoids departing
  clients, fewer timeouts; read off the unscaled-decay legs, which never diverged) → every 300** (`harness_pool` FB2 `fdstar_srv`,
  cifar only); **Oort / Oort\* every 9, not the source 5:** the source floors at round ~584 (58% of its 1,000 rounds; conf "may converge w/ ~400 rounds"), our legs have not converged by then (every 5: 52-57% vs 73-76% decay-off), so the floor goes at the end of our 0.7-1.4k rounds: 5 × rounds / 584 = 6-12 per leg → 9 (floor ~1,050; recorded deviation; FX-N163 rule extended: decay spans the run). Same value for oracle and non-oracle arms is preferred, not required: it
  held by coincidence elsewhere. Clip 1.0 rejected for the family (in-process: 55-61% vs 70%).
  *Why decay is required, not tuning (operator 10-26):* a config is deployable only if it never diverges; one that diverges on
  some runs and not others is not. Constant lr 0.1 diverged in 10 / 32 cifar legs (FedAvg 4 / 4, FedAvg\* 2, FedDance\* 3, REFL\*
  1; in-process NaN at 14.75k rounds), so the FedScale-family sync baselines cannot run without lr decay, and the decay-off
  (afternoon) results are invalid, including the legs that hit 80% before diverging (their speed came from an lr that is not stable over a
  full run). The source decay unscaled is not usable either (stalls at 47-56%, no divergence); the scaled decay is stable (3 / 3 seeds).
- **Traces and overruns (10-14):** 3-state syn traces only (FX-D166); ≤ 1-2% of tasks may overrun D, else fewer trainers per GPU.
- **Speed levers (10-14):** speech bf16, balanced GPU placement, one n per dataset for all baselines (K / c: Cohort table).
- **Degenerate cells (10-18):** a baseline below target in a short leg is round-bound if loss falls and accuracy rises.
- **Speech sync stacks are round-bound, reported as is (operator 10-25; closes F42-T12):** REFL / FedDance / FedAvg / Oort end 8-16%
  at 12 h on speech `s0u`, where nobody is ever unavailable, and their `*` oracles are no higher (REFL* 9.4, FedDance* 17.1, Oort*
  16.5%; F49): availability timing is not the cause. Accuracy rises every hour (FedAvg 3.7 → 13.7%); 160-186 rounds × K 5 × 1 local
  step ≈ 0.35 passes of the data in 12 h. Two causes, both their design: the round barrier waits for the slowest of K picks (median
  256 s, 93% of it the 29 MB model transfer) and each round carries one local step (FELIX_INSIGHTS §5). Configs stay source-faithful
  (C14); the paper states this; FB1's ≤ 120 h legs give them ~10× the rounds (bench vclock, no wall cost per round). Reopen only for
  a correctness bug. F49 ran the unscaled per-round lr decay (FX-N163): re-read on FB2 speech before quoting.
- **Launches (10-17, 10-25):** long batches on the operator's go (runbook commands); Claude runs short tests itself.
- **Parity keys (10-12, 10-18):** a board stands on one `code_key` (FX-D162); reals pair across real-path-identical keys
  (`parity_ladder.REAL_SAME_CODE`).


---

## Scope and conventions

- **Correct, not just equal:** every baseline follows its reference and hits target on full data; root and fix every red per run.
- **Scope:** `felix`, `oort`, `oort_star`, `refl`, `feddance`, `fedbuff` (+ `*` oracle arms, R27; `fedavg` = FedDance's random arm,
  FX-D146). Out: `oracle`.
- **Traces:** `syn_0`, `syn_20_3st`, `syn_50_3st`, `mobiperf_3st` (FX-D166); `syn_10_3st` = EuroSys'26's "syn_20" (FX-D18).
  Levels (n=300, TRAIN/EVAL/UN %): syn_10_3st 89/4.5/6.5 · syn_20_3st 80/8/12 · syn_50_3st 50/20/30; mobiperf trainable 18-22% by 6-24 h.
- **Methods (task time, streaming model, availability protocol, baselines table):** FELIX_INSIGHTS §6.
- **IDs:** `FX-N` next steps · `FX-L` lessons · `FX-T` tripwires · `FX-D` built · `PR#` batches. Shared work is `S#` in the parent.
- **Deprecated reference:** [PARITY.md](../async_cifar10/PARITY.md) §2-5 · [PARITY_CHECKER_README.md](../async_cifar10/scripts/parity/PARITY_CHECKER_README.md)
  · [UNAVAILABILITY_DESIGN.md](../async_cifar10/UNAVAILABILITY_DESIGN.md) · [BASELINES.md](BASELINES.md).


## Active build — fast bench and paper pipeline

**Fast bench (FX-N129): the simulator without its transport.** `scripts/fl_bench.py` runs one sim leg in one process: the
harness's own sim-mode aggregator class (the baseline's `aggregator_main`) and the real `PyTorchCifar10Trainer` objects, MQTT and the
trainer processes replaced by an in-memory channel. A dispatch hands its payload to that trainer's `_fetch_weights → train /
evaluate → _send_weights`; `--workers 0` runs it inline, `--workers W` in W trainer processes (trainer i → worker i mod W), and
the aggregator's own sim gate (expected completion, cold-start hold) waits for any reply that can precede the next commit.
- *Same code, by construction:* vclock, sct-ordered commits, `ClientAvailability` (notify / check-in / oracle, eviction, abandon,
  withheld delivery), streams, holdout, FedScale task time, selectors, optimizers, dynamic K / c, eval cadence and telemetry are the
  framework's. Configs: the leg's `debug_run.yaml` (a harness leg's, or `harness_suite.sh --dry-run`'s) through the launcher's own
  merge (`runner._build_aggregator_config`, `ConfigGenerator`). Output = a harness sim run dir (`aggregator_*.jsonl`,
  `trainer_<id>.jsonl` per trainer, `aggregator_config.json`), so `accuracy_table`, `felix_plots`, `pacing_plot`,
  `predictor_audit`, `config_audit`, `task_time_audit` read it unchanged. Logs: WARNING+ in one `bench.log`.
- *Carries forward automatically:* any change under `flame/` (aggregator stacks, selectors, availability, optimizers, telemetry),
  the trainer `main.py`, and every config layer (`baselines.yaml`, `datasets.yaml`, `baseline_reference.yaml`, harness_pool arms
  via the dry-run yaml). *Does not:* MQTT channel / backends, real-only paths (send-gate wait, real STATE_UPDATE notify, real
  check-in client), spawner GPU slots / pinning, env set by `harness_suite.sh` (the bench copies `FLAME_TRACE_OFFSET_S`; trace time
  scale not yet). A framework call into a channel method the bench lacks fails loudly (AttributeError), never silently.
- *Not modeled (the harness's job):* real-mode legs and real↔sim parity, MQTT delivery and per-process costs (charge profiles are
  profiled on the harness and read from config), process isolation (a trainer crash ends the bench leg).
- *Gain:* RAM, not leg speed. Leg host RAM ~4 GB vs 84-99 GB harness; 12 h vclock cifar Felix ~12 min solo (harness 17 min);
  setup 8 s (launcher YAML pickle-cached by file version, one train-set object per process). Speech is trainer-compute-bound:
  workers needed (timing todo).
- *Validity chain (operator 10-22):* real ⇄ harness sim (D5 parity) ⇄ bench (B2). The bench is never graded against real; a bench
  result stands for real FL only for cells green on both links. Until D5 re-grades parity on the streaming setup, bench and
  harness-sim results are sim-internal evidence alike. Use the harness for every sim-mechanism change (real↔sim pair first), the
  bench for screens, seeds, sweeps and long paper runs once B2 holds for the cells touched; re-run B2 on the affected cells after
  any change under `flame/` or the trainer.
- *Validation gate (Decisions "Fast bench"; `scripts/bench_validate.py --bench POOL --harness POOLS`, matched vclock):* B1
  checkpoint replay of real commits within 1 pp (done: 44.3 vs 44.1); B2 per baseline × dataset × trace, 12 h vs the F42b harness
  sim legs: |AUC Δ| ≤ max(1.5 pp, either side's seed range), train updates / h ±10%, bytes ±5%, mean train task time ±5% (and
  `task_time_audit` 0 formula mismatches); B3 `predictor_audit.py` check-in / REFL numbers match the harness legs'; B4 3 bench seeds
  (aggregator seed 1233 + i, the harness `--seeds` rule) for its own floors. Cells voided by F42-T12 validate after that fix.
  Cells with no F42b leg (Oort, Oort* both datasets; FedDance, FedDance*, FedAvg speech) first get harness F42 sim legs.
- *Packing rules (`scripts/bench_pool.py`, harness pool layout, one Ctrl+C, per-leg process group + hard timeout):* one math
  thread per process (`--threads`, FX-T73); CUDA contexts per GPU are the binding resource (FX-T74): legs × (1 + workers) per GPU.
  Seed i = aggregator seed 1233 + i; worker count does not change results (same task counts at 2 and 4 workers).
- *FluxTune:* FLUXTUNE FT-N18 (fwdllm stack on the bench).


### Common ideas we tested and rejected (defense table; one row per idea, evidence in the cited ID)

For the paper's "why not X" answers. Every row was measured on our setup (n 75 cifar / 50 speech, α 0.1, trace-tied streams, R27
knowledge); details stay at the cited ID.

| family | idea | what happened | evidence |
|---|---|---|---|
| selection | pick by individual update value / pre-step loss (Oort-style statistical utility alone) | top-K buffer loses 0.04-0.4 nats to random K; updates interact, so value is in the buffer's composition | FX-T78, `value_headroom.py` |
| selection | proxies for a good buffer: delta-cosine diversity, label balance (classifier-bias deltas), \|D\|, \|D\| q | ≈ random; the greedy buffer oracle's +0.10-0.18 nats stays uncaptured | FX-N137 |
| selection | uniform picks (FedBuff's) inside Felix | worse: AUC −1.6 / −1.9, final −0.6 / −1.7 vs l3b | FX-T79 |
| selection | stronger participation balance (γ 3) | slower to 40% (124 vs 94 min), final mixed | FX-T79 |
| selection | fresher utility / eval-task refresh of beliefs | stale beliefs flip ≤ 6-14% of ranks; eval tasks cost slots for no gain (C6; F48L `fn_ev` −0.2 pp, one tick later) | FX-T66, FX-L94 |
| selection | novelty loss (new vs trained samples) as a value signal | ratio 0.9-1.2×: no signal | FX-T67 |
| selection | an extra measurement pass per train task (I9) | cut throughput 14-20% and accuracy 1.4-7 pp | FX-T64 |
| aggregation | weight updates by \|D\| q_pre or the client's own held-out gain | −0.9 to −3.3 / −1.1 to −15 pp vs uniform (replay) | FX-T77 |
| aggregation | consensus / leave-one-out cosine, update norm, samples^0.65 (C5 value rate) | ≈ 0 correlation with oracle weights / ≤ uniform; C5 −2.1 / −1.2 AUC on speech | V4 block, F42b ladder |
| aggregation | FedBuff's staleness rate 1/√(1+τ) instead of Felix's rate | 40% +26 / +30 min, AUC −0.4 / −0.8, same final | FX-T79, `bench_F42g_ro_*` |
| step size | smaller fixed server lr (0.5 / 0.25), picked from a replay | −1.2 to −6.4 AUC closed-loop (slow early) | FX-T76 |
| step size | always normalize sub-batch steps (FedBuff §5, `l3bn`) | same code: final −0.4-0.8 / +0.5-1.4 pp, 40% +30 / +40 min, +40% GB | FX-T79, `bench_F42h_drift_1023_0418` |
| step size | switch full ↔ normalized steps on an inferred phase (buffer coherence r, \|w − w̄\| / \|Δw\|, step cosine) | no signal is monotone: all incoherent early, peak at plateau approach | FX-T80, FX-T81 |
| step size | stronger staleness discount 1/(1+τ) (n 300) | slowest and lowest on every cell: static −2.7 pp, 50% at 210-280 vs 100-130 min | F46 `uo_p1` |
| step size | scale the server step by the loss phase φ | static −0.9 / −3.2 pp: φ tracks fit, not learning left | FX-T85 |
| selection | speed term (utility per second) at n 300 | +3-4 pp at 1 h, −1.5-2 pp final on streams; loses to utility alone 25 / 28 pairs | F45, F46 `vub` |
| local work | 2 / 3 local epochs with an update-norm guard | −30-58% GB to 50%, but −1.3 to −2.7 pp final and a 55% stall on streams | F47 `fn_e2` / `e3` |
| pacing | fixed lower concurrency (c 30) or a bigger buffer (K 20) | c 30: `cp`'s final at 2.2-2.4× its GB; K 20 −0.4 pp at 2.3× GB | F46, F47 |
| pacing | c bounded by availability (c ∝ trainable share, ≤ ½ / ⅓ of the pool) | restores choice but 3-15 pp behind at 3 h: early throughput beats choice | F48a, F47 `fn_ca` |
| pacing | c by loss phase as a self-pacing controller | c hits floor 10 by 6-8 h and stays to 48 h as data arrives | FX-T89 |
| step size | lr decay schedules, adaptive server optimizers (FedAdam / Yogi) | not run: need a horizon or add β / ε / η constants (pruned, FX-N134) | FX-N134 |
| pacing | pace c on held-out gain (pace_c, pacer v2 snr / ratio) | signal noise at n 75 (z < 1.5 even in hour 1): c cut to 2-5, −4 to −10 AUC | FX-T72, FX-T75 |
| pacing | AIMD K on loss-on-receive (C4) | walks K to k_min or saturates at K = c; no gain | FX-T65, F42b ladder |
| pacing | training-loss change, per-client sample counts, hard plateau tests | memorization / privacy / ≥ 1 h to confirm | V3 block |
| availability | Felix learning availability by check-in instead of notify (K1, `fck`) | −4.4 AUC cifar syn_50_3st, −10.0 speech syn_50_3st | F42b verdict |
| serving | serve the averaged model (EMA, β = 1 − K/n) | kept, but generic: helps every stack (+0.6-2.0 AUC cifar, +4-8 speech), so not a Felix edge | FX-D217, FX-D219 |
| setup | per-client task time independent of work | gave Felix free bigger shards; void (FedScale task time instead) | FX-T60 |


## Harness (reference for B2 / EV24 only; FX-N22)

`scripts/harness_pool.py --tier T --datasets D [--phases …] --gpu-ids …`: isolated parallel legs (own cores, mosquitto, run tag);
tier list in `--help`, shapes in `harness_pool.SHAPES`. Run dirs in `async_cifar10/experiments/`, pools in `experiments/`. Speech n 100
harness legs: ~150 GB RAM, 4 GPUs each. Tests: `tests/harness/test_{slot_isolation,harness_pool,fl_data}.py`.

## Baseline hyperparameters (operator 10-08)

Single source: `_metadata/baseline_reference.yaml` (value + citation per baseline x dataset; rules ROBUST L33-L37).
- REFL: 1 mini-batch/round [code]; cifar FedAvg + equal stale weight [code]; speech YoGi + SAA -4, lr 0.05 [code]; random sampler;
  `adapt_selection` cifar 1 / speech 0, picks round(K x 1.3) [code].
- Oort/oort_star: 20 mini-batches, YoGi (no momentum), equal-weight average, lr decay 0.95, FedScale utility, upstream getTopK
  [code] (`third_party/Oort`); cifar borrows Oort's CV config.
- FedDance: 1 mini-batch, FedScale defaults (equal-weight FedAvg, decay 0.98/10) [paper]; I_m = mean local training loss (Eq. 6).
- FedScale-family clients (REFL, Oort, FedDance): SGD momentum 0.9, weight decay 5e-4, fresh per task [code].
- FedBuff: 1 epoch, lr normalization [paper]; client/server lr [ours] (paper tunes by sweep). Felix: [ours].
- n / K / c: "Cohort and participation" (top).

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
- **FX-L95** A real client's small post-task notify can beat its own update's commit; a ledger release at commit must keep state the client sent after dispatch (F42-T9: the check-in pool emptied).
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

- **FX-L97** Serve the version sequence's average, train on the raw model: EMA β ≈ 1 − K/n is +0.7-1.6 AUC paired at equal training, for
  Felix and FedBuff alike (F42e); report its baseline ablation beside any Felix claim.
- **FX-L98** Score a serving or eval change paired inside one leg (served and raw on the same training) before spending seeds.
- **FX-L96** Screen on the bench only cells green on both links, real ⇄ harness sim (D5) and harness sim ⇄ bench (B2); re-run B2 on
  the touched cells after any `flame/` or trainer change (V5).

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
- **FX-L86** Count eval gates and selector clocks in updates or seconds: in versions, eval's download share grows 10% → 73% as K drops
  10 → 2 (PR39b), and a 35-version gate outlasts AVL_EVAL episodes (0 speech evals, PR40a).
- **FX-L99** Cut bytes by raising accuracy per update, not by dropping tasks: GB to target = updates × model size; 3 local epochs
  gave −40-48% GB to 40% at ≤ 1 pp final (`bench_F42h_ep_1023_0441`).
- **FX-L101** Compare Felix to Oort on bytes, not only to FedBuff: Oort reaches cifar 50% on 15-18 GB vs Felix 98-112 (F43), at 15-26 h.
- **FX-L102** Read a stream plateau against test loss and clients' loss of the received model: rising test loss at flat accuracy
  with client loss ≤ 0.1 is memorization; F47 per-batch augmentation keeps test loss < 1 and lifts +10-17 pp.
- **FX-L103** Size c against the trainable count, not n: c 60 of ~148 trainable dilutes `uo`'s picks (picks~size +0.67 vs +0.93 at
  c 30, 33 vs 40 samples / update; F46 `s50u`).
- **FX-L104** Check selectivity on MobiPerf before crediting selection: c 60 ≥ its ~50-67 trainable, so `fn` picks like FedBuff
  (picks~size −0.45, τ 15); c ≈ 12 restores +0.48 (F47 `smu`).
- **FX-L91** Size a one-seed screen by its draw floor: four cifar ctl sims (3 seeds + a same-seed rerun, forked at commit 1) span
  36.9-40.5% and 150 → > 180 min to 40% (PR40a).
- **FX-L92** Trust a sign repeated across cells over one cell's gap: I9 lost to ctl and eval0 in all 13 cells while single gaps sat
  inside the floor (PR40a).
- **FX-L94** Ask whether a belief can reorder picks before refreshing it: |D| · loss is size-driven at α 0.1 (q 2× vs |D| 25× across
  clients), so stale vs fresh loss and jagged growth move ranks ≤ 1-7% (offline, 10-21).
- **FX-L90** Scale every liveness rule to the dataset's task timeout: S1's 15 min sat under cifar's 903 s abandon and killed 4 legit
  reals (PR39, FX-N111).
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
- **FX-T72** Don't read a held-out plateau on a data stream as convergence: F42b pace_c cut cifar c to 4-5 and lost 1-7 pp final accuracy as data kept arriving.
- **FX-T75** Don't pace c on 10-min held-out gain at n 75: paired per-report d_nll p10 / p90 −1.27 / +1.17 vs mean −0.013, so z < 1.5
  even in hour 1; pace_gain snr cut c to 2 and lost ~10 AUC, ratio −4 AUC at −49% GB with no better GB to 40% (F42d).
- **FX-T76** Don't pick a server step from a staleness-0 replay of mid-run checkpoints: replay chose α 0.125-0.25 (+2-4 pp, cifar and
  speech) but a fixed 0.5 / 0.25 lost 1.2-6.4 AUC closed-loop for Felix and 4-12 for FedBuff (slow early; F42d).
- **FX-T82** Don't run > 1 local epoch on cifar without an update-norm guard: F43 e2 / e3 hit local loss 0.000 on 45-111 samples, one
  finite step 241 (typical 0.02-0.3) passed `NONFINITE_REJECT` and the model went NaN at 5-11 h.
- **FX-T84** Don't read the population through `channel.ends()` outside dispatch: it runs a selection and returns the in-flight
  set; `_dynamic_kc_step` did, so `n_eligible_train` = in-flight and F46 smoke's `ca` c fell to its floor 15 (fixed: `all_ends()`).
- **FX-T85** Don't scale the server step by the loss phase φ: φ tracks how well clients' data is fit, not learning left; static
  `ps` −0.9 pp, `pscp` −3.2 (φ 0.4-0.5 at 12 h while accuracy still climbs; F46).
- **FX-T86** Don't call two times to target a tie by minutes: an eval lands on the first commit after its tick, up to 2 min late at
  c ≈ K (F46 `uo_cp`); `fl1_verdict` compares eval ticks.
- **FX-T87** Don't test availability-scaled c on syn_50_3st: its trainable share stays ≈ 0.5, so `ca` ≡ c 30 (F46: c 29-30, same
  final); use a MobiPerf cell.
- **FX-T88** Don't sense new data from a client's own loss jump once its data is fit: F46 static legs (no new data) jump ×8 on
  5.8% of tasks at 9-12 h, streams 3.4-3.6%; a 0.01-0.05 loss swings on a few samples (FX-N142 pivoted, R26 offline).
- **FX-T83** Don't size a run or a per-round array by a round cap: 20000 versions cut F43 n 300 async legs at 17-35 h; a 20000-slot list crashed F45 `s3v` at 10 h (FX-D230).
- **FX-T80** Don't gate the step on buffer coherence r = |mean Δ|² / mean |Δ|²: K·r ≈ 0.8-1.25 all run (signal ≤ 3% of update energy),
  lowest early, peaks before the plateau (F42f / F42g l3b vs l3bn legs, `dynamic_kc` stats).
- **FX-T81** Don't gate on |w − w̄| / |Δw| or consecutive-step cosine: d rises with step decay, cos peaks +0.1-0.2 at plateau approach,
  ≈ 0 early and late (`bench_F42h_drift_1023_0418`, `agg_drift`).
- **FX-T79** Don't blame the cifar stream plateau on selection, rate or sub-batch steps: frozen crop draws set it; per-batch
  augmentation removes it for every stack (F47 +10-17 pp; FX-D233).
- **FX-T89** Don't expect c by loss phase to self-pace: F47 streams hold c at floor 10 (φ ≤ 0.12) from 6-8 h to 48 h while data
  keeps arriving; it saves bytes, not pace.
- **FX-T91** Don't put bench legs on GPUs a harness leg uses: a speech harness leg grows to 44 / 45 GB per GPU after the bench
  placed its legs; 8 smoke legs OOM'd (10-25). Split GPUs (`--gpus`), or run in sequence.
- **FX-T90** Don't rank legs that hit the target stop by final or total GB: F47 `aug` legs stopped at 70% in 4-13 h; compare time /
  GB to target, or run to the horizon.
- **FX-T78** Don't pick by individual update value or pre-step loss q: the top-K buffer loses 0.04-0.4 nats to random K
  (`value_headroom.py`, cifar); updates interact, so the gain is in buffer composition.
- **FX-T77** Don't weight buffer updates by |D| q_pre or a client's own held-out gain: replay vs uniform −0.9 to −3.3 / −1.1 to −15 pp.
- **FX-T71** Don't key a probe / eval gate on train-report times: a busy client reports every task, so it never looks stale and probes stop (F42S01: 0 probes after the first burst).
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
- **FX-T73** Don't pack bench legs with torch's default threads: 24 legs × 3 processes × 128 threads put load at ~1000 on 128 cores
  and every leg timed out (bench wave 1, 10-22); one math thread per process.
- **FX-T74** Don't count GPU memory alone when packing bench legs: 12 CUDA contexts per GPU (4 legs × 3 processes) slowed Felix legs
  ~2.5× vs solo; count contexts per GPU.
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
- **FX-T60** Don't charge a task a time independent of its work: one per-client D let Felix train 1.7× bigger shards free, and gave
  Oort's 600-sample tasks and REFL's 10-sample tasks the same time (FX-D197).
- **FX-T59** Don't set K from the within-cycle gradient noise scale at α 0.1: B is unbounded or 10²-10³ and K pins at c (smoke
  10-20). It measures client disagreement, not SGD noise.
- **FX-T61** Don't compare TTA across stacks on a version eval cadence: in 2 h Oort got 8 evals, FedDance 44, every speech leg 1-4
  (PR39); eval on the leg's clock (FX-N112).
- **FX-T64** Don't add a measurement pass to train tasks without pricing its R: I9's forward pass cut R 14-20% and accuracy 1.4-7 pp
  (PR40a).
- **FX-T65** Don't adapt K by AIMD on loss-on-receive: it walks K to k_min, past κ* (cifar 7 → 3, f2 → 2; speech dkc 18.9% vs
  46.7%, PR40a).
- **FX-T66** Don't sell utility freshness or eval refresh: offline 48 h (`rank_sim`), stale beliefs flip ≤ 6% cifar / ≤ 14% speech
  picks at ≤ 1.1% regret, flat 12 → 48 h; eval refresh recovers ~0 (10-21).
- **FX-T67** Don't build on novelty loss: global-model loss on a client's unseen vs trained samples is 1.00 (p10-p90 0.88-1.2), also
  for class-by-class arrival, on PR40a / F40S checkpoints (10-21). Re-confirmed F42b streams: loss on newly arrived vs trained samples 0.9-1.2× (cifar, speech).
- **FX-T70** Don't let a per-trainer helper parse a population-wide file: 75 trainers × 0.56 GB of trace YAML doubled leg RAM (FX-D210).
- **FX-T68** Don't bound a pool with `timeout --foreground conda run …`: it kills conda, not the python child (F41 smoke ran past
  its 15 min); run `timeout` on the env's python directly.
- **FX-T69** Don't test progress across clients without pairing: client hardness (1.45× p10-p90) swamps a window's decline, so an
  unpaired AIMD read every window as flat and walked K 10 → 15 (F41 smoke); pair each client with its own last loss (FX-D205).

---

## Built (one line each; details in code, tests, `git log`; IDs kept because code cites them)
- **FX-N141** FL1 tooling: `F44`-`F49` tiers (`harness_pool.py`), `run_fl1.sh` / `run_f48.sh` / `run_f49.sh`, `fl1_verdict.py` (updates,
  staleness, starvation, lead; ties by eval tick, FX-T86), `config_audit.py` (… `rp` = s / a / b, `dkcp`, `cmin`), `selection_audit.py`,
  selector `prop` (`async_oort.py`), `value_headroom.py --holdout`, `bench_pool.py` (packs by GPU memory + host RAM).
- **FX-N144** c policy `felix_c` (`flame/selector/loss_phase.py` φ): `pool` (c ≤ pool × trainable), `ramp` (blend c → that bound by 1 − φ),
  `phase`, `avail`; `dynamic_kc` telemetry carries c, φ, step scale (`dynamic_c_report.py`). `fx` = `pool=0.5 ramp`.
- **FX-N146** `ceiling.py --full --cosine` (converged centralized ceiling on full shards); `fl_bound.py` (ideal-FedAvg FL bound,
  `--gpu a,b,..` splits clients per round); `target_report.py` (time / GB at the 5-eval-mean hit, normalized tables, curves, bars).
- **FX-D235** Paper run sets (FX-N154): tiers `FB2<set>` / smoke `FB2<set>S` from `harness_pool.FB2_SETS` (E2E, LAD, SWC, SWA, SWG,
  SWS, REJ, STA = outline Table 1; cells + `s20u`; FedBuff\* `fbstar_srv`, FedAvg\* `favstar_srv`; swaps = `fx` with one layer
  replaced); `config_audit` rows per arm (`test_fb2_run_sets_match_table_1`, `test_every_fb2_arm_has_an_expected_row`).
- **FX-D236** Paper run pipeline (FX-N153): `paper_runs.py` (manifest: Table 1 + Table 2 figure → legs; `status` done / missing /
  stale per figure; `launch` writes one script, one bench pool per dataset × seed), `leg_digest.py` (`digest.json.gz` per leg, every
  plot series with its source event, then pigz of raw logs; ~10 KB, 5-15 s per 12 h leg), `paper_plots.py` (one function per Table 2
  row → `paper_felix/figs/expts/`, MISSING / PRELIM panels), `bench_pool --digest` + repeatable `--harness-args`, `run_b2.sh`
  (harness ⇄ bench B2 on one cell); readers accept gzipped logs (`test_paper_pipeline.py`).

**Simulator fidelity**
- **FX-D212** Holdout per arm (V1): `validation_holdout` 0.1 only on arms that read held-out data (harness_pool `_PROBE`), the rest
  train on 100%; the trainer refuses `validation_probe` without a holdout; `config_audit.py` checks holdout per arm, `_r<i>` rows and
  P3 / P4 (trained = visible − visible // k, from `trainer_round`).
- **FX-D213** Fast bench (V5): `fl_bench.py` (one leg, in-memory channel, `--workers`), `bench_pool.py` (packing, timeouts, one
  Ctrl+C), `bench_validate.py` (gate B2; `--upd` tolerance); `test_fl_bench.py`.
- **FX-D237** Busy until sct (FX-N157): an async client committed before its sct (clock-jump clamp) stays busy until sct
  (`sim_busy_until_sct`, default on) and is a sim wake-up; EV24 `busy_until_completion` (dispatch before the previous task's sct,
  not abandoned) in `event_invariants.py`, not yet prefix-safe.
- **FX-D238** Gate on evicted updates (FX-N160): the async commit gate probes and holds for an evicted end whose update has not
  arrived, at its `pending_withheld` delivery (a lower bound); the failsafe gives it up (`sim_gate_evicted`, default on;
  `test_gate_waits_for_evicted_in_transit_update`).
- **FX-D239** Bench fail-fast: `fl_bench` exits a leg (rc 70) when a worker dies, not at the wall timeout
  (`test_dead_worker_fails_the_leg`); EV0 accepts telemetry `run_end` (bench legs have no aggregator log); `run_ev24.sh` grades the
  harness run dirs in `async_cifar10/experiments` and skips the `.log` files; `scripts/watch_runs.py` (pool watcher, ALERT lines).
- **FX-D240** Two-tier selection telemetry (default; `FLAME_TELEMETRY_FULL=1` = old): `selection.per_trainer` keeps the chosen ends,
  the whole pool once per 600 s clock per task (`per_trainer_full` true; trimmed events false, so pool readers skip them; old logs
  have no flag). 2 h cifar Felix leg: 1.76 GB → 78 MB raw (lean 71); EV9 / EV21 still run. `paper_runs` no longer exports
  `FLAME_TELEMETRY_LEAN` (drops the field entirely). EV12 passes a target stop (5-eval mean ≥ `targetAccuracy`).
- **FX-D234** Path-independent check-in (FX-N151): `CheckinLedger(replay="time")` replays sim check-ins in time order (ties by end)
  with one jitter stream per client, so end order and `advance` cadence (harness vs bench) no longer change a run; `checkin.replay=end`
  reverts (`test_time_replay_is_path_independent`). `fl_bench` sets `FLAME_TRACE_OFFSET_S` before generating trainer configs (bench
  trainers saw the trace 24 h early: telemetry-only in sim, the send gate is real-only).
- **FX-D215** V2 held-out report (FX-N127): trainer `validation_report` scores the received model on its held-out samples in eval mode
  (buffers kept) on every train task and probe, paired client-side with its previous report (`d_nll`, `d_correct`, `n_paired`, prev
  version); charged as `val_pass_s` (EV3, `task_time_audit` subtract it); aggregator `_ingest_val_report` adds updates / seconds between
  the paired versions (version → (U, t) map); `val_probe` telemetry carries `src`. Tests: `test_stream_schedule.py`, `test_event_invariants.py`.
- **FX-D216** `pace_gain` policy (V3; laws snr / ratio, τ the only setting) + controller `k_follows_c` (false keeps K when c falls);
  `dynamic_kc` events carry `pace_*` state. Arms v2p / v2pr / v2p_s025. Tests: `test_pace_gain.py`.
- **FX-D217** Serve-EMA (FX-N128): aggregator `evalEma` scores EMAs beside the model (`agg_eval_ema`); `serveEma=sweep` (β = 1 − K / n
  contributors) or a float serves the EMA as `agg_eval` and logs `agg_eval_raw`; training unchanged. Arms *_ema, ctl_srv / l3b_srv
  (felix, fedbuff). Tests: `test_example_eval_gate.py`.
- **FX-D219** Averaged model on every stack (FX-N133): `serveEma=sweep` on sync stacks reads K = `aggregation_goal` and n = clients
  seen in round caches (syncfl and oort stacks); harness arms `ctl_srv` (all baselines), `rstar_srv`, `fdstar_srv`, `fbstar_srv`.
- **FX-D220** EV16 sync rule (F42-T12 b, h): a withheld delivery is graded against the first close that could take it: due before its
  intake cut (last arrival) and, if fresh for that round, before its K-th fresh arrival (`_sync_closes`).
- **FX-D230** `debug_run.sh` round cap 10^7 (was 20000), so the vclock budget binds; asyncfl per-trainer participation is a set of
  rounds (was a 20000 list); `bench_report.py --ceiling` adds end h and the centralized ceiling there.
- **FX-D233** `trainerAugmentEachPass` (FX-N143: default on, every stack, both datasets, ceilings too; off in stub mode): plain tensors +
  `fl_data.augment` per batch (cifar crop pad 4 + flip; speech FedScale's waveform pipeline) (`test_augment_each_pass.py`).
- **FX-D232** FedBuff `norm_clip` (`opt.norm_clip=k`, 0 off): an update whose L2 (BN stats excluded) exceeds k × the median of the last
  200 is scaled to it after 10 norms; emits `update_clipped` (`tests/optimizer/test_fedbuff_lr.py`). Tier F44 = FL1 (`F44_ARMS`, static
  `s0` + streams, 12 h), `scripts/run_fl1.sh`, `scripts/fl1_verdict.py` (pass rule vs FedBuff, same eval tick = tie, top-20% share, coverage);
  `bench_report` keeps static `s0` apart from `s0u`.
- **FX-D231** R27 availability knowledge (F42a, F42-T1..T5): `avail_knowledge` notify / checkin / oracle / none
  (`ClientAvailability._believed_state`; `flame/availability/checkin.py` ledger + `checkin_client.py`; EV22, EV23); REFL
  `ProphetForecaster` + `avail_predictor: prophet`, `holdoff_rounds`; FedDance `checkin_source: selection`; arms `rstar`, `fdstar`,
  `fbstar`, `fck`; CPU gates `--tier F42C` / `F42CR`; `avail_view_plot.py`; trainer `validation_holdout` / `validation_probe`;
  `config_audit.py`, `predictor_audit.py`.
- **FX-D221** REFL with Prophet (F42-T12 c): picks = the fork's priority-fill (p < 0.5 shuffled first, random fill), not a strict p order;
  Prophet `growth` flat (selector `prophet_growth` reverts; deviation in `baseline_reference.yaml`).
- **FX-D222** Sync check-in (F42-T12 d): an overdue early arrival polls within a window (`checkin.overdue: round` reverts), and a sim
  sync drain stops at the next check-in while the version misses picks (`simCheckinDrainCap=false` reverts).
- **FX-D223** Analysis: `felix_why.py POOL...` (bench or harness, per dataset, all-seed summary); `value_headroom.py` (per-update value
  vs buffer-composition headroom, R26); `utilityPrefixSamples=-1` = one training batch.
- **FX-D225** `agg_drift` telemetry (FX-N134 G): per version |w − w̄| (served EMA), |Δw|, cos of consecutive steps, on every `serveEma` leg
  (`agg_common._emit_drift`).
- **FX-D226** F42 arms `l3b_e2_srv`, `l3b_e3_srv` (l3b + `epochs` 2 / 3; FX-N135).
- **FX-D227** `FLAME_TELEMETRY_LEAN=1` drops `selection.per_trainer` (99% of bytes at n 300; `test_lean_drops_selection_per_trainer`).
- **FX-D228** Tier F43 (`F43_ARMS` at `F43_SHAPE`, `F43_RUNTIME_S`; `test_f43_tier_reference_cohort`), `scripts/run_f43_jayne.sh` (cifar bench +
  speech harness, one TERM stops all; Ctrl+C in tmux), arm `l3b_e4_srv`.
- **FX-D229** `targetWindow` (config): stop when the mean of the last w evals ≥ `targetAccuracy`; unset = the consecutive rule
  (`test_target_window_stop.py`). F43 legs set 0.7 / 5 (`F43_STOP`).
- **FX-D224** Felix screen knobs: `sel.selectType=random` (uniform picks); F42 arms `l3b_b3`, `l3b_pb`, `l3b_ro`, `l3bn`, `rnd`,
  `rnd_ro`, `rnd_ron` (all `_srv`).
- **FX-D218** Bench from harness plans: `bench_pool.py --harness-args` (harness_pool `_parser` / `plan`, dry-run YAMLs), `--gpus-per-leg`
  (fl_bench `--gpu A,B` round-robins workers); `bench_report.py` (AUC, TTA, GB to target, seed range, `_srv` raw rows);
  `ceiling.py`; `config_audit.py` reads bench pools and audits lr / kfc / vrep / serve; `dispatch.py` survives a timed-out GPU probe.
- **FX-D214** Lean trainers: satellite coords load only when a config sets `satellite_coordinates_path` (LEO runs; not in the cifar
  base); per-batch grad-norm and delta-L2 telemetry behind `trainerUpdateNorms` (off; GPU syncs nothing read); parity template's
  `util_counterfactual` off (no reader of `util_disparity`); `ClientAvailability` scores beliefs for the leg's connected clients
  only (selection unchanged: the selector sees connected ends; was ~790 `_believed_state` calls per commit at n 75).
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
- **FX-D211** F42-T9: `CheckinLedger.release` keeps a check-in pooled while its own update commits (real trainers check in right
  after upload). F42CR real 10 min: FedBuff 5 → 429 versions, REFL 2 → 81; F42R GPU 8 min: Felix 53, REFL 24, FedDance 14, FedBuff 13
  (sims 52 / 29 / 16 / 27); EV PASS (REFL EV12: F42-T11). `notify_recv` receipts (`Channel.update_state`) feed
  `predictor_audit.real_notify_fidelity` (F42-T7): F42R GPU real Felix 8 min, flip → receipt 0.13 s p50 / p95 (8 flips; a
  partial packed run 1.36 / 2.69 s, 7 flips), missed 0 / false 0.4 s per client-h, EV PASS; D5's 2 h reals re-measure it.
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
- **FX-D198** Task timeout per baseline (FX-N115): `flame/launch/task_timeout.py` = 1.5 × the longest modeled task of the baseline's
  local work over `datasets.yaml` `reference_split` (`model_kbit`); `debug_run.sh` sets `send_timeout_wait_s` from it. `test_task_timeout.py`.
- **FX-D199** Stall rule S1 waits max(15 min, 1.25 × the leg's task timeout) (`harness_pool.stall_limit_s`, FX-N111).
- **FX-D200** Test eval on the leg's clock (`evalEverySeconds`, `agg_common.evaluate` via `_avail_now`, background thread unchanged);
  G3-family tiers pass `evalEverySeconds=auto` = runtime / `EVAL_POINTS` (24) (FX-N112). `test_example_eval_gate.py`.
- **FX-D201** Felix local work (FX-N116): trainer `localWork` `cap` (localSteps capped at an epoch) | `split` (compute = β / (1 − β) ×
  transfer, `localWorkBeta`), charged at the capped samples (`_local_steps`, `_task_time_s`). Off = byte-identical. `test_local_work.py`.
- **FX-D203** C3 `sel.utilityPerSecond`: stat utility ÷ last intrinsic task seconds, speed penalty off (`async_oort`; pytest).
- **FX-D204** Balance `sel.participationDecay` γ: utility × (1 + picks / mean picks)^-γ (`async_oort._participation_factor`; pytest).
- **FX-D205** C4 `aimd_k` `stall`: paired per-client loss change per window (`cycle_client_losses` from the asyncfl aggregator); fall → K − 1,
  rise → 2K (+ memory floor), flat → K + 1; floor `k_min` (pytest; F41 smoke).
- **FX-D206** C5 FedBuff `agg_rate_conf.type: value`: rate = samples^β (1+τ)^-a (β 0.65, a 0.5), lr ÷ the cycle's mean rate (pytest).
- **FX-D207** Q-STREAM: `data_streaming.trace` ties `usage` growth to the trainer's own leg trace (in use = not AVL_TRAIN; none →
  linear over 6 days); `harness_pool.usage_hp`, tier F41, `batches/f41.yaml` (pytest; EV19 on F41 smoke).
- **FX-D209** FX-N125: `--scale-smoke` gives `SHORT_KEEP` legs (unaware Oort, GPU FedBuff on syn_50_3st) `SHORT_KEEP_S` 3600 s, not
  their full length (F41's 24 h Oort smoke legs ran in full; pytest).
- **FX-D210** FX-N126: trace-tied usage reads `availability_traces/usage_index_<trace>.json` (per-trainer in-use flips, source sha256,
  `stream_schedule.build_usage_index`) instead of parsing the 14 MB synthetic YAML per trainer: +561 → +31 MB per process; F41 cifar
  usage leg 109 → 51 GB (pytest; smoke `pool_*_F41SMOKE_E`).
- **FX-D208** 10-21 Felix mechanisms (FX-N118-N123): eval gate on the clock (`evalGateUnit: seconds`), prefix utility
  (`utilityPrefixSamples`), usage stream (`stream_schedule.usage_sessions`), AIMD memory, trace offset (`FLAME_TRACE_OFFSET_S`),
  `utility_bench.py` (in-process utility bench), `rank_sim.py` (offline closed-loop ranking model); F40S smoke checks passed.
- **FX-D196** `selection_audit.py`: per sim leg, what drives picks (size, speed, loss per sample), samples per update, cifar class
  TV, belief error and bias by source (eval / train), eval per train, I5 K / B trajectory. Tier F39C (E3 arms, trainer hp via `thp`);
  `batches/pr39c2.yaml`.
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
- **Q-SPEECHK / Q-SPEECHC · Speech n 100 shape (recommended 10-24; F48S runs it):** scale by n / 3 under the Cohort rules: async c 60 → 20 (FedBuff's
  20%, not F43's 30), K stays 10 (FedBuff's K is absolute, paper §6.1; c / K 2, so lower staleness); sync K 15 → 5 (5% rule), c 5-6.
  The alternative K 3 keeps cifar's c / K 6 but drops FedBuff's K.
- **Q-PHASE · Felix c bounds:** F46 bounds dynamic c to [K, c_config] = [10, 60] (20% of n, FedBuff's c). Keep the upper bound at
  FedBuff's c for fairness, or allow Felix above it when availability is high?

- **Q-ARMS · Confirm the current arms table (Next steps "Current arms").**
- **Q-CHECKIN · Open: whether sync check-in may keep its window open until K × 1.3 have checked in (F42-T12 d).**
- **Q-PAPAYA · Section / page of FedBuff-Papaya's 1.5× task-timeout rule** (FX-N115 cites it).
