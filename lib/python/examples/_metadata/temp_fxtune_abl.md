# Paper runs (outline §10, R1–R9), 10-29 evening

## What each R needs (existing data first)

| R | needs a run? | why |
|---|---|---|
| R1a DistilBERT · AG News | **no** | q2 (Φ_b 4, c1493) and run 1 (FwdLLM) ran on the same shepherd bench, same divisors (0.478 / 4.78) and recipe; q2's cos audit costs wall, not vclock. L10 answered: run 1 *is* on the paper engine |
| R1b DistilBERT · Yahoo | SwiFT arm only (= r3b) | FwdLLM arm = run 28 (DNF, peak 0.616, divisor 4.78) |
| R1c RoBERTa-L · Yelp-P | FwdLLM arm only | SwiFT = run 4b (Φ_b 4, T_res 570). FwdLLM (run 3: BitFit lr 1e-2, var 0.2, divisor 0.169) never ran; 12–30 h, 4 GPUs. **Open: where/when** |
| R1d 1.7B | no | 6c + jayne FwdLLM |
| R2 | r2s (run 8 −step), r2b (run 9 −buffer) | a08 = run 8 but stopped at target 0.86 (c505); need final accuracy |
| R3 | r3a, r3b | |
| R4 | r4a | no lr ≥ 0.03 FwdLLM screen with Φ exists |
| R5 | **can't run** | fwdllm aggregator has no unavailability wiring (FT-N5); runs 23–25 already dropped → delete the trace clauses |
| R6 | not a queue run | needs a new offline script + saved q2 checkpoints (none saved). Later |
| R7 | r7a, r7b | `--server-weight-decay` exists (FedSgdAggregator) |
| R8 | r8a, r8b | L5 answered: q5a/q5b used `--rho0 0.07` (not rescaled) |
| R9 | not yet | needs new code (cohort held per shared minibatch in SwiFT). Later |

**Bug found:** q4 ran FwdLLM at divisor 0.478 (FluxTune's); FwdLLM's is 4.78 (D_CALIBRATION: 2 vs 20 forward passes per task).
That is the "27 s vs 3 s" task time: q4 vclock is 10× inflated. Its per-commit data (Φ, var, cos, accuracy) is fine.

## Arms (shepherd bench; `queue_ablations_bench.sh` r-arms; one seed; `--server-update-audit`; no target / stall stop)

| arm | R | change from recipe (DistilBERT · AG News, N 50, ρ0 0.07, bs 8) | cap | cos every | watcher stop |
|---|---|---|---|---|---|
| r2s | R2 run 8 −step | a08: raw SGD lr 0.01, SwiFT buffering | 1500 | — | plateau |
| r2b | R2 run 9 −buffer | `fwdllm` (pick, var 0.3, held cohort, divisor 4.78) + trust ratio `exp` T_res 566 | 1500 | — | plateau |
| r3a | R3a | **Yahoo** (bs 32, 6,000-row resampled eval, run 29 cell), constant ρ 0.07 | 1500 (Φ 39) | 50 | fall / hold |
| r3b | R3b = R1b SwiFT | **Yahoo**, `exp` T_res 566 (Φ_b 4) | 1500 | 50 | plateau, only after r3a ends and r3b matches its commits |
| r4a | R4a | `fwdllm` lr 0.03 (3×), var 0.3, divisor 4.78 | 1500 | 25 | fall / hold |
| r7a | R7a | `fwdllm` lr 0.01 + WD 0.0026 (= q4's mean ρ²/2, pins Φ ≈ 1) | 1500 | — | plateau |
| r7b | R7b | constant ρ 0.07 + WD `auto` (ρ²/2): WD instead of a budget | 1500 | — | plateau |
| r8a | R8 | N 10, ρ0 0.0313 = 0.07√(10/50), T_res 2829 (Φ_b 4) | 300 | 10 | cap |
| r8b | R8 | N 90, ρ0 0.0939, T_res 314 | 300 | 10 | cap |

## Stop rules (`expt_scripts/watch_paper_runs.py <OUT>`)

Acc = 5-eval trailing mean; peak = its running max; commit = `server_update` count. The watcher touches `<OUT>/stop/<arm>`;
the aggregator halts cleanly at the next commit and runs the final full-test eval (`stop_file`, new, default off).
- **fall** (r3a, r4a; the claim is "falls ≥ 2.3 pt"): FELL when acc ≤ peak − 10 pt for 10 evals (40 commits), c ≥ 300.
  10 pt = 4× the replicate floor, so the figure shows a fall, not a dip (q1 took ~100 commits from −5 to −10).
  HOLDS when Φ ≥ 20 and acc within 2.3 pt of peak (q1 was −25 pt at Φ 21).
- **plateau** (r2s, r2b, r7a, r7b, r3b): PLATEAU when peak gained < 0.3 pt over the last 300 commits, c ≥ 800 (r3b ≥ 1000).
  On q2 this fires ~c1100–1200 at ≤ 0.3 pt under its c1493 value.
- **guard** (all non-fall arms): FELL when acc ≤ peak − 5 pt for 10 evals, c ≥ 200.
- **hang**: no telemetry for 45 min → SIGTERM. Stop file ignored for 20 min → SIGTERM.
- Backstops: `--max-commits` (clean), bench timeout 16 h (kill).

## Launched 10-29 16:19 (shepherd `fwdllm/paper_runs_20261029/`)

8 legs on GPUs 0–7 (r2s r2b r3a r3b r4a r7b r7a r8a), r8b queued; watcher pid 530442 (`watch.log`, `verdicts.json`).
Stop-file halt smoke-tested 10-29 15:05 (r2s, c10: `reason=external`, final full eval, DONE). Shepherd-only, uncommitted:
`aggregator/FedSgdAggregator.py` (`stop_file`), `expt_scripts/{queue_ablations_bench.sh,watch_paper_runs.py}`.
R1c (FwdLLM RoBERTa-L): operator decides later.

**Cos audit costs ~7.4 min wall per fire on the bench** (q1: median gap 446 s vs 5.4 s, 3.6 of its 5.8 h), not 85 s.
So r3a / r3b were killed at c7 (16:37) and relaunched with cos every 150 in `paper_runs_20261029y/` (GPUs 3, 7; own
watcher, pid 535971); their rows in `paper_runs_20261029/` read FAIL rc=-15. r8b took GPU 2. Yahoo ~28 s / commit →
≤ ~13 h. r4a (cos every 25) and r8a / r8b (every 10) keep their audits; r8 legs take ~4 h each because of them.
Measured logged Φ is the plan (exp ΣB), not ‖θ‖: on WD arms read `trainable_weight_norm` (r7b pinned 6.73 → 6.72).

## Launch command (for reference)

```bash
cd /home/dgarg39/aish_test/flame/lib/python/examples/fwdllm && D=paper_runs_20261029 && mkdir -p $D
OUT=$D GPUS_PER_LEG=1 TIMEOUT_H=16 setsid nohup bash expt_scripts/queue_ablations_bench.sh r2s r2b r3a r3b r4a r7b r7a r8a r8b \
  > $D/queue.log 2>&1 < /dev/null &
setsid nohup python expt_scripts/watch_paper_runs.py $D > $D/watch.log 2>&1 < /dev/null &   # after jobs.tsv exists
```
r8b waits for the first free GPU. Est. wall at 1 GPU / leg: AG News SwiFT ~5–6 h to c1500, Yahoo ~8 h, r8 ~1–4 h,
FwdLLM unknown (q4: 605 commits in 6 h; r4a's 3× lr reaches Φ 8 in far fewer commits).
Monitor (tmux on shepherd):
```bash
cd /home/dgarg39/aish_test/flame/lib/python/examples/fwdllm && source /coc/scratch/dgarg/miniconda3/etc/profile.d/conda.sh && conda activate test_fwdllm && watch -n 120 'for d in paper_runs_20261029 paper_runs_20261029y; do python3 expt_scripts/check_ablations.py $d | head -13; tail -4 $d/watch.log; echo; done'
```

---

# Pitch evidence runs (Q1–Q8), 10-29

## Running now

| where | what | answers |
|---|---|---|
| shepherd `paper_ab_20261028b` | a11 / a12 / a13 (FedBuff-async, Felix(P), FedBuff(P) baselines); `TIMEOUT_H=4` ends them by ~02:20 | none of Q1–Q8 |
| jayne `smollm17b_yahoo_n50_fwdllm_20261028_1827` (pgid 2426406) | FwdLLM 1.7B, lr 3e-3, var 8.4, LoRA r 2 (same r as FluxTune 6c) | Q2: Φ 8.6 at c272, no peak yet. **Keep running**; record Φ at peak if it turns |

## What each question needs

| Q | already answered by logs | new run |
|---|---|---|
| Q1 collapse without budget | no: a21 (constant ρ) stopped on target at Φ 2.46 | **q1** |
| Q2 Φ at peak | FwdLLM DistilBERT·AG News 3.6; FwdLLM 1.7B > 8.6 (running) | **q1** (our side), **q4** (FwdLLM repeat, with Φ) |
| Q3 two-sided budget | only "too small caps" (a14–a16, 1.7B / RoBERTa Φ_b 2) | **q2** (Φ_b 4), **q3** (Φ_b 8); Φ_b 2 = run 2 |
| Q4 gate proxy | variance ∝ ‖g‖² on 3 FwdLLM runs; cosine never logged | **q4** (FwdLLM + cos audit); q1 for cos vs Φ on ours |
| Q5 cos ÷ √(PN/p) | only Aug data (DistilBERT, 0.05 init → 0.10–0.15 trained) | **q1 + q4 + q5a / q5b** (N 10 / 50 / 90 and FwdLLM's N 100–470, P 1-of-10 vs mean-10) |
| Q6 growth law | **yes**, every no-momentum run within 1.3%; 6c / 4b fixed at launch, no momentum | none |
| Q7 budget faster? | no: a21 constant ρ 0.07 hits 0.86 at 2.65 h vs 3.68–4.30 h budgeted | none; reread after q1 (if q1 collapses, budget = safety, not speed) |
| Q8 cohort hold | yes: 2.8 min DistilBERT, ~20 min RoBERTa-L, 75 min 1.7B (median vclock) | none |

Not covered (cost): cosine on another model (the 1.7B audit stalled at commit 1, `bench_sm_20261025/POOL_audit*_stalled`);
Q2 on a second dataset (constant-ρ DistilBERT · Yahoo, ~7 h). Decide after q1.

## Arms (DistilBERT · AG News, §2 recipe, bs 8, n 100, shepherd bench)

All: no target stop, no stall stop, Φ rail off, `--server-update-audit --cos-ground-truth-audit --cos-probe-batch-size 1024`
(`SIM_LONG` in `fwdllm/expt_scripts/queue_ablations_bench.sh`). Dry-rendered 10-29 00:11, keys checked.

| arm | change from recipe | cap | cos every | Φ reached | est. wall |
|---|---|---|---|---|---|
| q1 | constant ρ 0.07 (`--rho-schedule pool`) | 1,500 | 50 | 4 @ c566, 8 @ c849, ~39 @ c1500 | ~3 h |
| q2 | `exp`, T_res 566 (Φ_budget 4) | 1,500 | 50 | ~3.6 | ~3 h |
| q3 | `exp`, T_res 849 (Φ_budget 8) | 2,000 | 50 | ~6.6 | ~4 h |
| q4 | `fwdllm`, lr 0.01, var 0.3 (= run 1) | 700 | 25 | ~5+ (run 1: 4.3 @ c494) | ~4 h |
| q5a | N 10 | 300 | 10 | — | ~0.5 h |
| q5b | N 90 | 300 | 10 | — | ~0.7 h |

Wall estimates assume the audit costs ~85 s per fire (harness, Aug); unmeasured on the bench.

## Launched 10-29 00:18 (shepherd; script synced, md5 de6c01ab on both nodes)

| dir | arms | GPUs | state |
|---|---|---|---|
| `paper_pitch_20261029a` | q4 | 2,3 (idle ones) | started 00:18 |
| `paper_pitch_20261029b` | q3 q1 q2 q5a q5b (3 at once) | 0,1,4,5,6,7 | waiter (`pgrep -f "queue_ablations_bench.sh a1[7]"`) starts it when the a11–a13 queue exits, ~02:20 |

Separate dirs: each launch empties its `jobs.tsv`. One bench pool on all 8 would stack legs on a11–a13's GPUs (it places by free memory).
Waiter pattern is `a1[7]` so it can't match its own `bash -c` line (the first try did and was killed).

**Smoke check (~30 min after each start): per-commit Φ logged and cos audit fires**
```bash
ssh shepherd.cc.gatech.edu 'cd /home/dgarg39/aish_test/flame/lib/python/examples/fwdllm && for f in paper_pitch_20261029{a,b}/POOL/q*/runs/*/run_*/telemetry/aggregator_*.jsonl; do echo "$f server_update=$(grep -c "\"server_update\"" $f) cos=$(grep -c cos_ground_truth $f)"; done'
```
cos = 0 after the first stride → audit broken on the bench; kill and relaunch without `--cos-ground-truth-audit` (Q1–Q3 still answered).

**Monitor**
```bash
ssh shepherd.cc.gatech.edu 'cd /home/dgarg39/aish_test/flame/lib/python/examples/fwdllm && source /coc/scratch/dgarg/miniconda3/etc/profile.d/conda.sh && conda activate test_fwdllm && for d in paper_pitch_20261029a paper_pitch_20261029b; do python3 expt_scripts/check_ablations.py $d; done; pgrep -af "a1\[7\]" | head -1'
```

## Readout (per question)

- **Q1:** q1 accuracy (5-eval smoothed) vs Φ. Drop ≥ 2 pt past its peak while q2 / run 2 hold → budget is the mechanism.
- **Q2:** Φ at smoothed peak for q1, q4, run 1 (3.6), 1.7B FwdLLM. Spread ≤ ~±20% → "known in advance"; else "set once per model".
- **Q3:** peak and final accuracy at Φ_budget 2 (run 2: 0.873) / 4 (q2) / 8 (q3). Two-sided if q3 declines and run 2 caps.
- **Q4:** q4 per commit: first-iteration variance, ‖g‖², Φ, `cos_ground_truth`. Flat cos while var rises → proxy argument holds.
- **Q5:** `cos_ground_truth / cos_theory` (= `aim_d`) across q1, q5a, q5b, q4; constant → planned step quality holds.
- **Q7:** time to 0.85 / 0.86 for q1 vs q2 / run 2, and whether q1 stays above 0.86 afterwards.

Analysis scripts (stdin to `ssh shepherd python3 - <run dirs>`):
`/tmp/claude-1170639/-home-dgarg39-flame/94e20b78-f337-48be-9c3b-7271a77da086/scratchpad/pitch_stats.py` (Φ, acc vs Φ, var),
`.../gate_stats.py` (FwdLLM variance vs ‖g‖², cohort hold), `.../abl_stats.py` (legs without `server_update`).

## SmolLM fwdllm on jayne (status 10-28 23:38)

c258 / 5.2 h wall / 406 vclock h: best 0.48, last-10 mean 0.46; Φ 8.6, no decline. Behind FluxTune 6c on time
(0.45 at 306 vs 64 vclock h). Recheck at c~400; if it peaks, log Φ at peak (Q2).
```bash
tail -f /home/dgarg39/aish_test/flame/lib/python/examples/fwdllm/smollm17b_yahoo_n50_fwdllm_*_acc.log
```

## Status check 10-29 05:30 (shepherd)

No fails or stalls. q5a/q5b queued (start when q1/q2 free GPUs, ~08:10). All legs done ~10:30.

| arm | commits | best acc | now (smooth) | Φ now | ETA |
|---|---|---|---|---|---|
| q4 (FwdLLM) | 550 / 700 | 0.855 | 0.844 | 4.55 | ~07:30 |
| q1 | 845 / 1500 | 0.871 | 0.828 | 7.89 | ~08:10 |
| q2 | 847 / 1500 | 0.876 | 0.871 | 2.93 | ~08:10 |
| q3 | 805 / 2000 | 0.874 | 0.870 | 3.57 | ~10:30 |
| q5a, q5b | queued | – | – | – | ~08:10 → 09:00 |

- **Q1:** collapse confirmed. q1 smoothed 0.868 @ Φ 3.67 → 0.828 @ Φ 7.89 (−4.0 pt; Φ 5 0.858, Φ 6 0.854); q2 / q3 hold ~0.870. Budget is the mechanism.
- **Q2:** smoothed peak Φ 3.67 (q1), 3.66 (q4), 3.6 (run 1). Consistent; q4 now 0.5 pt below peak at Φ 4.5. 1.7B FwdLLM (jayne): no peak, Φ > 8.6.
- **Q3:** q2 (Φ 2.93) 0.871 ≈ q3 (Φ 3.57) 0.870; q3 hasn't passed Φ 4, no two-sided verdict yet.
- **Q4:** q4 var flat 0.29 while Φ 1.35 → 4.5 (corr 0.53). cos not re-read.
- **Q5:** `cos_theory` 0.065 on FluxTune arms; ratio needs q5a/q5b.
- **Q6, Q8:** answered. **Q7:** q1 not faster than q2 per commit (0.858 @ c376 vs 0.857 @ c397) and collapses later → budget is safety, not speed.
- Pending after q1: cosine on another model, Q2 on a second dataset.

## Status check 10-29 07:10 (shepherd)

q4 **FAIL rc=124**: hit the 360 min timeout at c605 / 700 (no-op for the question; Φ 4.76, past the peak). q3 / q1 / q2 at c1200–1250, ETA ~08:10 (q1, q2), ~10:30 (q3). q5a / q5b still queued. Jayne SmolLM log's last line is 05:05 (c108 of the new run; best 0.523), ~2 h stale: check the process.

| arm | commits | smooth peak (Φ) | last (smooth) | Φ now |
|---|---|---|---|---|
| q4 (FwdLLM) | 605 | 0.849 (3.66) | 0.842 | 4.76 |
| q1 (const ρ) | 1250 | 0.868 (3.67) | **0.621** | 21.2 |
| q2 (Φ_b ≈ 3.4) | 1250 | 0.878 (3.39) | 0.878 | 3.43 |
| q3 (Φ_b ≈ 4.9) | 1218 | 0.875 (4.70) | 0.872 | 4.86 |

- **Q1: confirmed, strongly.** q1 smoothed 0.868 → 0.621 (−24.7 pt) by Φ 21; Φ 8: 0.814. q2 / q3 hold within 0.3 pt of peak. Budget is the mechanism, not just a safety margin.
- **Q2: holds across methods.** Peak Φ 3.67 (q1), 3.66 (q4), 3.6 (run 1), 3.39 (q2, no decline yet). Spread ≈ ±5% on the three uncapped runs. "Known in advance" supported for DistilBERT · AG News; second model still open (1.7B FwdLLM had no peak by Φ 8.6).
- **Q3: not two-sided yet.** q3 (Φ_b 4.9) sits at 0.872, 0.3 pt below its peak; q2 (3.4) 0.878 is the best final. Φ_b 2 (run 2, 0.873) vs 3.4 (0.878) vs 4.9 (0.872): flat within ~0.6 pt, so the budget is a plateau, not a sharp optimum. Only the "too large collapses" side is shown (q1 at Φ ≥ 8). Say "a wide safe band around Φ ≈ 3–5", not "tuned optimum".
- **Q4: proxy argument is mixed.** q4 (FwdLLM) var flat 0.29 while Φ 1.4 → 4.7, so FwdLLM's variance gives no early warning. FluxTune var does rise with Φ (corr 0.86–0.93 on q1; 0.045 → 0.24), but it tracks ‖g‖ growth (52 → 118), as expected. `cos_ground_truth` is tiny and noisy on 25 fires: q4 0.000–0.015, q1 0.000–0.017 rising slightly with Φ; no clear collapse signature in cos. Don't claim cos is a usable collapse gate from this data.
- **Q5: not matching theory.** `cos_theory` 0.065 vs measured `cos_ground_truth` ~0.003–0.017 (ratio ~0.05–0.25, drifting up with Φ). Needs q5a / q5b before saying anything; as is, the planned-step-quality claim (`aim_d` constant) is not supported.
- **Q7:** q1 is not faster than q2 per commit (0.858 @ c376 vs 0.857 @ c397) → budget is safety, not speed (unchanged).
- **Open:** q5a / q5b; recheck q3 past Φ 5 once it finishes; cosine on a second model; Q2 on a second dataset.

## Final readout 10-29 11:40 (all runs finished or stopped)

### Run status and data caveats

- q1 and q5a/q5b ran to their caps (c1500, c300, c300). q2 (c1493/1500), q3 (c1450/2000) and q4 (c605/700) hit the 6 h
  timeout (rc=124). All reached the Φ they were meant to test except q3 (5.48 of ~6.6 planned).
- jayne 1.7B FwdLLM (SmolLM · Yahoo, LoRA r 2) was INTERRUPTed at 05:08 at c551, Φ 12.3. Its `_acc.log` `commit=` counter wraps
  at ~435; use the telemetry (`experiments/run_20261028_182751_fwdllm_yahoo_n60_smoke_syn_0_sim`).
- **q4 replays run 1 exactly** (same seed; peak 0.8546 at c333 in both, identical Φ trajectory). It adds the cos audit and
  80 more commits, not a second independent FwdLLM point.
- Every arm is a single seed. Eval noise: raw vs 5-eval-smoothed accuracy differs by up to ~0.5 pt. Treat gaps < ~1 pt as ties.
- Setup for every q arm: DistilBERT · AG News, §2 recipe, bs 8, n 100, syn_0, no target or stall stop, Φ rail off.
  The cos audit uses a fixed shuffled held-out batch of 1024 (post-B17).
  q1/q2/q3 share the seed and are identical until their ρ schedules diverge.

| arm | Φ_budget | smooth peak (Φ, c) | final smooth (Φ) | Φ reached |
|---|---|---|---|---|
| run 2 (P1_fluxtune) | 2 | 0.872 (1.97, c1097) | 0.872 (1.98) | 1.98 |
| q2 | 4 | **0.880** (3.56, c1407) | 0.879 (3.62) | 3.62 |
| q3 | 8 | 0.875 (4.70, c1160) | 0.866 (5.48) | 5.48 |
| q1 | ∞ (const ρ 0.07) | 0.868 (3.67, c532) | **0.296** (39.1) | 39.1 |
| q4 / run 1 (FwdLLM) | none | 0.849 (3.66, c349) | 0.842 (4.76) | 4.76 |
| 1.7B FwdLLM · Yahoo (jayne) | none | 0.508 (11.4, c473) | 0.496 (12.3) | 12.3 |

### Q1. Without the budget, does training collapse?

**Answer: yes, completely.** With ρ held at 0.07, weight norm grows without bound and accuracy collapses to chance.
The budgeted arms, on the same seed and recipe, hold their peak.

- q1 smoothed accuracy vs Φ: 0.845 (Φ 2), 0.864 (3), **0.868 peak (3.67, c532)**, 0.866 (4), 0.858 (5), 0.854 (6), 0.814 (8).
- It falls below 0.86 at Φ 4.7 (c631), 0.85 at 5.8 (c716), 0.80 at 9.6 (c925) and 0.70 at 14.6 (c1097).
  It ends at 0.296 smoothed / 0.259 raw at Φ 39 (c1500). AG News is 4-class, so chance is 0.25.
- Controls: q2 never goes below 0.877 after its peak (final 0.879). q3 never goes below 0.865 (final 0.866).
- The decline is slow at first (−1 pt by Φ 5, −1.4 pt by Φ 6) and then accelerates (−5.4 pt by Φ 8, −57 pt by Φ 39).
  A short run, or one that stops on a target, never sees it. This is why a21 (constant ρ, stopped at target, Φ 2.46) looked fine.
- Mechanism: the step direction does not get worse (cos *rises*, see Q4). The failure is norm growth: Φ = |θ|/|θ₁| reaches 39×.

**Writeup:** "Without a budget, the same recipe peaks at 0.868 and then decays to chance (0.30 at Φ = 39). Budgeted runs hold
within 0.1–0.9 pt of their peak over the same number of commits."

### Q2. Is the Φ at peak accuracy known in advance?

**Answer: it is stable within a model (across methods and inits) but not across models. Set it once per model.**

- DistilBERT · AG News: FluxTune constant ρ peaks at Φ **3.67**, FwdLLM at Φ **3.66** (run 1 = q4).
  That is two different optimizers and two different starting norms (|θ₁| 6.75 vs 13.6), with peak absolute norms of 24.8 vs 49.7.
  The ratio Φ matches; |θ| does not. So Φ is the right variable for the budget.
- 1.7B · Yahoo (FwdLLM): no peak at Φ 3–4. Smoothed accuracy is still climbing at Φ 8 (0.456) and peaks at Φ 11.4 (0.508).
  It is 0.496 at Φ 12.3, a −1.2 pt dip that is within noise. Its Φ* is ≥ 11, about 3× DistilBERT's.
  Caveat: LoRA r 2 starts from a small |θ₁| (7.4), so part of that is the init.
- The peak also depends on the path. Arms that reach a given Φ more slowly are better there:
  at Φ 5, q1 (fast) is at 0.858 vs q3 (annealed) at 0.868. q3's smoothed peak is at Φ 4.7, not 3.67.
  So Φ* moves up when ρ is annealed, which the budget does by construction.
- Not covered: a second dataset on the same model (constant-ρ DistilBERT · Yahoo, ~7 h) was not run.

**Writeup:** "For a given model, the norm ratio at which accuracy peaks is the same under both FluxTune and FwdLLM (3.66 vs 3.67)
despite a 2× difference in initial norm. It differs across models (≥ 11 for a 1.7B LoRA model), so Φ_budget is a per-model
constant set once, not a universal value."

### Q3. Is the budget two-sided (too small hurts, too large hurts)?

**Answer: yes, weakly, with a broad optimum near Φ_budget ≈ 4.** Peak accuracy by budget:

| Φ_budget | arm | peak | final | notes |
|---|---|---|---|---|
| 2 | run 2 | 0.872 | 0.872 | slow: 0.87 at c1061 |
| 4 | q2 | **0.880** | 0.879 | 0.87 at c609, 0.875 at c889; still rising at its cap |
| 8 | q3 | 0.875 | 0.866 | peaks at Φ 4.7, then −0.9 pt by Φ 5.5 (timed out before Φ 6) |
| ∞ | q1 | 0.868 | 0.296 | collapses |

- Too small: −0.8 pt in peak, and about 75% more commits to reach 0.87.
- Too large: −0.5 pt in peak, and already trending down at timeout. Its accuracy-vs-Φ curve matches q1's early decline (~1 pt by Φ 5–5.5).
  q3 was planned to reach Φ ≈ 6.6, so the decline would likely have continued, but the data stop at Φ 5.5.
- Single seed, gaps ≤ 1.2 pt (≥ 2× eval noise only for 4 vs 2). The large-budget side rests mostly on the q1 extreme.
- Earlier "too small" evidence (a14–a16: Φ_b 2 on 1.7B and RoBERTa) also showed capped accuracy.

**Writeup:** "Accuracy is flat over a broad band of budgets (0.872–0.880 for Φ_b 2–8). Too small a budget slows convergence and
caps accuracy slightly; too large starts the same decay as no budget. We use Φ_b = 4 (the peak Φ for this model)."
Avoid "tuned optimum".

### Q4. Is the gate signal a good proxy for step quality / collapse?

**Answer: no, for both methods. Variance measures gradient *scale*, and cos does not flag collapse.**

- FwdLLM (q4): `var_at_commit` stays at 0.29 throughout only because the gate commits once variance drops below
  `var_threshold` 0.3. It is flat by construction and carries no information.
- The real FwdLLM signal is first-iteration variance per bin: 2.6 → 9.9 → 14.5 (×5.7) from the first to the last 10% of bins,
  while ‖g‖² grows ×5.6 (‖g‖ 2.4k → 5.7k). The gate answers by spending more iterations: 10 → 35 → 57 per bin.
  So the gate tracks ‖g‖², costs time as gradients grow, and has no notion of direction quality.
- FluxTune: `var_at_commit` ∝ ‖g‖² exactly. In q1, var/‖g‖² = 1.66e-5 at the start and 1.67e-5 at c1500 (var 0.045 → 0.51,
  ‖g‖ 52 → 174). It rises with Φ (corr(log var, log Φ) 0.90 in q1) only because ‖g‖ grows.
- cos (`cos_ground_truth`, 25–30 audits per arm) does not warn of collapse. In q1 it *rises* from 0.005 (Φ < 1.6) to 0.017–0.019
  (Φ 12–35), because the true gradient grows 0.4 → 5.2 as the model degrades.
  In the stable arms it stays noisy around 0.002–0.007 (q2 corr with Φ −0.04, q3 −0.32).
- Cost: FwdLLM uses 33 iterations per commit (median) vs FluxTune's fixed 4, for about the same measured aim
  (cos 0.003 median in q4 vs 0.002–0.007 in FluxTune arms).

**Writeup:** "Neither the variance gate nor gradient alignment detects the collapse. Variance tracks ‖g‖² (ratio constant to
1%), and alignment improves as the model degrades. FwdLLM's variance gate therefore adds iterations (10 → 57 per commit) as
gradients grow, without improving step direction. Collapse comes from norm growth, which is what Φ_budget bounds directly."

### Q5. Does step quality follow cos_theory = √(g·N/P)?

**Answer: cos increases with N at least as fast as √N, but its level is ~10–15× below theory.**
Don't claim "planned step quality is achieved". Early phase (Φ ≤ 1.6, 26–30 audits per N; mean ± se):

| N (arm) | cos_theory | measured cos | aim_d = measured / theory | acc at c300 |
|---|---|---|---|---|
| 10 (q5a) | 0.029 | 0.0010 ± 0.0010 | 0.03 ± 0.03 | 0.54 (peak 0.64) |
| 50 (q1–q3 pooled) | 0.065 | 0.0038 ± 0.0011 | 0.06 ± 0.02 | ~0.84 |
| 90 (q5b) | 0.087 | 0.0085 ± 0.0010 | 0.10 ± 0.01 | 0.852 |

- Scaling: measured cos goes 1 : 3.8 : 8.5 for N 10 : 50 : 90, against √N's 1 : 2.2 : 3.0. That is steeper than √N, closer to
  linear in N. A constant aim_d is only marginally consistent (50 vs 90 differ by 1.9σ, 10 vs 50 by 0.6σ).
- The shortfall is not client disagreement: n_dir = aim_d²·N ≈ 0.2 at N 50, and n_dir < 1 is impossible under that explanation
  (`replay_scoring.py` test).
- Within a run, aim_d rises with the true-gradient norm (q1: ~0.02–0.08 early → 0.26–0.28 at Φ > 11). That fits a fixed noise
  floor that cos_theory does not model: cos is low when ‖g‖ is small and recovers as ‖g‖ grows.
- Part of the gap could be the probe: cos is measured against a 1024-sample gradient, not the full gradient, which lowers the
  measured value. Not quantified.
- Practical effect of N: N 10 barely learns (0.64 peak, var 0.23 vs 0.045 at N 50). N 90 matches N 50 per commit
  (0.852 at c300 vs q2's 0.85 at c325) but costs more time per commit (3.7 h vclock for 300 commits vs ~2.3 h at N 50).
  N ≈ 50 is the efficient point.
- Not run: P 1-of-10 vs mean-10.

**Writeup:** "Measured alignment grows with pool size as predicted, at least as fast as √N (cos 0.001 / 0.004 / 0.009 for
N = 10 / 50 / 90). Its absolute level is an order of magnitude below the isotropic bound, so we use cos_theory for scaling, not
as a calibrated step-quality estimate."

### Q6. Does the norm-growth law hold?

**Answer: yes, within 1.6% on every run, up to Φ 39.** Predicted exp(½Σln(1+ρ²)) vs measured |θ|/|θ₁|:

| run | predicted | measured | error |
|---|---|---|---|
| q1 | 38.4 | 39.0 | 1.5% |
| q2 | 3.61 | 3.66 | 1.3% |
| q3 | 5.46 | 5.54 | 1.6% |
| q4 (FwdLLM) | 4.76 | 4.75 | 0.3% |
| q5a | 1.515 | 1.511 | 0.3% |
| q5b | 1.453 | 1.458 | 0.3% |
| 1.7B FwdLLM | 12.27 | 12.13 | 1.2% |

It holds for both methods, two models, N 10–90 and three ρ schedules. All runs are without momentum.

**Writeup:** "Weight norm follows Φ = Π√(1+ρ_t²) to within 1.6% across all runs, including a 39× growth, so a ρ schedule fixes Φ in advance."

### Q7. Does the budget make training faster?

**Answer: no, it costs at most ~15% of the time to a given accuracy. What it buys is a higher ceiling and no collapse.**

| arm | to 0.85 (vclock / c) | to 0.86 | to 0.87 | to 0.875 | stays ≥ 0.86 |
|---|---|---|---|---|---|
| q1 (no budget) | 2.2 h / c307 | 2.9 h / c385 | never | never | c385 to c631 only |
| q2 (Φ_b 4) | 2.3 h / c325 | 3.4 h / c429 | 5.3 h / c609 | 8.4 h / c889 | to the end |
| q3 (Φ_b 8) | 2.4 h / c332 | 3.3 h / c417 | 5.8 h / c660 | 11.0 h / c1157 | to the end |
| run 2 (Φ_b 2) | 2.9 h / c381 | 4.4 h / c533 | 10.1 h / c1061 | never | to the end |

- Constant ρ is fastest to 0.85 / 0.86 by 5–15%, because it takes larger steps early.
- Then it tops out 1.2 pt below q2 and stays above 0.86 for only ~250 commits.
- Note: a21 (constant ρ, Aug harness) reached 0.86 in 2.65 h vs 3.68–4.30 h budgeted. That gap is larger than here.

**Writeup:** "The budget is not a speedup: an unbudgeted run reaches 0.86 about 15% sooner. It then peaks lower (0.868 vs 0.880)
and collapses, so whether its result is usable depends on stopping at the right commit. The budget removes that dependence."

### Q8. How long is a cohort held per commit?

Median vclock per commit:
- FluxTune DistilBERT: 0.6 min (4 iterations, fixed).
- FwdLLM DistilBERT: 2.9 min (run 1, ~31 iterations).
- FwdLLM 1.7B: 83 min (p90 150 min, ~8 iterations of 272 s tasks). The earlier 75 min estimate was from a shorter window.
- Don't use q4's 32 min/commit: its bench task timing came out 27 s vs run 1's 3 s for the same trajectory, so q4 vclock is inflated.
