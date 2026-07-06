# TEMP task tracker — logical real↔sim parity (delete when folded into simulate_fwdllm.md §G)

**Goal.** Prove the sim takes the SAME logical steps in the SAME order as real (same cohorts, same receive
order, same variance cadence, same grads) up to data bin 1 across fwdllm / fwdllm_plus / fluxtune — and make the
parity CHECKS + PYTESTS actually *exhibit* these properties (they don't today). Only then chase the time
dimension. Correctness before speed; no hacks (simulate_fwdllm.md principles #14/#16).

---

## ⏸ SESSION CHECKPOINT (2026-07-05 — P2-7a VALIDATED + full-run roots nailed; resume at #15)

**P2-7a (K-D31) VALIDATED and full runs analyzed.** Databin1 checks (`--max-bin 1`): sync `cohort_sequence`
ok=true, set/order/var/cadence=1.0 — the delay-tie is closed. Full runs (`--delay-factor 1`) then settled BOTH
open roots:
- **SYNC:** `sim_rate` 2.9–3.0 (#12c resolved). But receive-ORDER now 41/41 identical yet cadence still breaks at
  **bin 7** → the "order→var→RNG-desync" root is REFUTED for the full run. Real root = **grad non-reproducibility
  given matched order** (~1e-3 GPU fp16 jitter amplified by the split-half variance → gate flip at (7,2)). This
  **answers P0-2**: exact cadence parity is unattainable past ~bin 6. ⇒ relax the target: `cohort_sequence` EXACT
  scoped to `--max-bin 1`; add a DISTRIBUTIONAL cadence/var rung (mean-band + KS + `var_good` fraction) for the full
  run. (Confirm with a 2-real-run diff before landing — strong single-run evidence already.)
- **fluxtune:** `sim_rate=0.50` is a **GPU-PIPELINING loss (#15)**, not #12c and not the commit gating (which is
  correct, per-grad). Objective telemetry: sim 1.54× GPU concurrency vs real 3.37×; per-commit real 4.30s wall, sim
  6.60s wall / 3.27s vclock. The overrun (#1d, `set_match=3/272`) is a thin-margin contention tail (10 trainers /
  8 GPUs doubling + aggregator eval GPU) — `jvp_perf_opt` already put the MEAN (3.61s) under the 4.0s budget.
- **Pinning is clean** (earlier "under-provisioned" read was a misread of `gpu=4.9s` compute-time as device IDs):
  8 GPUs, balanced round-robin, CPU-pinned. Aggregator GPU pin + trainer `[PIN]` self-report + `[LOAD_BALANCE]`
  check landed (K-D33, 120 launch tests green).

**NEXT: #15 — decouple real-GPU dispatch from the sct-ordered commit drain** (keep GPUs full like real; drain orders
commits by sct for the vclock only). This is the one thing keeping fluxtune `sim_rate<1`, and it also lifts #1d.

---

## ⭐ #15 fluxtune `sim_rate=0.50` — ROOT CAUSE FOUND (resume implementation here)

Banked evidence run: `experiments/run_20260705_204619_fluxtune_n10_smoke_syn_0_sim` (+ `_real` pair
`run_20260705_202448`). Primary drain diagnostic = the `[SIM_GRAD_RECV]` log line
(`fwdllm_aggregator.py:926`, format `end= sct= T_v= buf_depth= inflight_exp= sel_ends=`).

**ROOT CAUSE (evidence-backed): a circular wait between the drain's `earlier_stuck` gate and hold-to-commit.**
The drain (`_sim_recv_min_grad`, `fwdllm_aggregator.py:752`) blocks REAL WALL to keep sct-ordered commits: it won't
commit a buffered grad while an in-flight trainer has a smaller **expected** sct (`_sim_inflight_expected[end]`). But
that trainer is itself blocked in `recv_wrapper` (hold-to-commit: `_release_end_on_return:1136` returns early on the
residence path, so its slot frees only when its PREVIOUS grad commits) → it is NOT computing → its grad never comes →
30s `RECV_TIMEOUT_WAIT_S` failsafe evict. Circular. Steady state (startup is fine — 10 dispatched fresh → parallel;
collapses at the first commit when trainers start getting held).
```
hold-to-commit blocks trainers in recv → gate waits real-wall for NON-COMPUTING in-flight trainers
  → circular stall (30s failsafe) → ~1974s (81% of 2425s wall) burned → GPU 1.54x (real 3.37x) → sim_rate 0.50
```

**EVIDENCE:** per-commit real 4.30s wall vs sim 6.60s wall / 3.27s vclock (sim_rate 0.50). GPU concurrency real
3.37x (98% busy) / sim 1.54x (85%); same GPU work (~3.7-4.0k trainer-s), same ~480s 8-way floor, `max=10` at
startup (HW sustains 10-wide). Trainer 371 `recv_wrapper` mean **30.9s** sim (max 55) vs **0.01s** median real;
compute mode-invariant (~3-4s, `_emulate_training_delay=0.000s`). Smoking gun: `[SIM_GRAD_STUCK_EVICT] end=0379
exp=24.0 bmin=27.0` — drain blocked ~30s for 379 while **6 grads sat ready**; 379's telemetry: blocked in
`recv_wrapper` **60.39s**, got weights the instant after evict. `buf_depth` **constantly 6**; **524/1093 commits
(48%) stall >2s = 1974s**. Committed scts already go out of order (11,13,7,5.4,5.5) → the strict order the gate
blocks for isn't even preserved. Eval is once/data-bin (23/24), not per-iter, but 8.5s BLOCKING (~195s, secondary).

**PROPER FIX — re-dispatch on RETURN** (match real, recv 0.01s), decoupling physical GPU pipelining from the
virtual in-flight ledger. Returned trainer gets next weights + computes immediately → never idle in recv → the
in-flight trainer the gate waits for produces its grad in ~4s IN PARALLEL with 7 others → drain commits a BATCH per
GPU-pass. Keep the gate + sct-order commits (preserve #1d cohort parity); keep the virtual in-flight COUNT /
selection eligibility (R1) held-to-commit — that's a SEPARATE ledger from physical compute (K-D17b wrongly welded
them). **Hard constraint (principle #16):** the compute-ahead grad must use the model version real dispatched
(fedbuff staleness `V'-V`) — real also cycles on return so versions come from the same deterministic commit order
(K-D29); VERIFY from telemetry, don't assume. **Belt-and-suspenders:** `_sim_inflight_expected[end]` is stamped at
DISPATCH (`:2819`, `_sst + _budget`, `_sst=self._vclock.now`) assuming immediate compute-start; a held trainer
hasn't started → expected sct is a fiction. Tie it to actual compute-START and/or bound the wait << 30s.

**IMPLEMENTATION PLAN (resume tomorrow):**
- **P0 (verify, read-only):** diff per-trainer `MessageType.MODEL_VERSION` (grad's dispatch version) vs the commit
  sequence, real vs sim (`inc_model_version_per_data_id=True`; staleness log ~`fwdllm_aggregator.py:1170`). Confirm
  re-dispatch-on-return in sim yields the SAME dispatch-version sequence as real. If not, pin version to the
  trainer's virtual completion (sct), not physical dispatch — revisit before coding.
- **P1 (core):** in `_release_end_on_return:1128` async-sim-residence branch, let PHYSICAL re-dispatch (weights send
  → compute next grad) happen on return, while the SELECTOR's virtual in-flight set (`selected_ends`/`all_selected`,
  driving R1 + `extra = c - inflight`) stays held to commit via `_sim_hold_busy_slots`. Re-dispatched grad enters
  the sct buffer with its own `SIM_COMPLETION_TS`; drain keeps committing in sct order. Keep K-D27 two-ledger
  discipline (`_sim_pending_commit`) — do NOT reintroduce the R1 regression. Fix the `_sim_inflight_expected` stamp.
- **P2 (tests+telemetry, same change):** `pytest tests/mode -k fwdllm` + `-k parity`; async_cifar10 byte-identical
  (fwdllm_aggregator-only edit; `_sim_recv_min` untouched, principle #8/#9). Assert `recv_wrapper`→~0, concurrency
  up; R1 must stay ~0 (bank it — K-D19: not done until smoke shows R1<=2%).
- **P3 (validate):** `run_sequential.sh --only fluxtune --mode both --delays on --delay-factor 1 --max-data-id 2`.
  Expect `recv_wrapper`→~0, concurrency→~3.37x+, `sim_rate`→>1, no `[SIM_GRAD_STUCK_EVICT]`, and `cohort_sequence`/
  `var` parity UNCHANGED (if it moves, the P0 version assumption was wrong).
- **P4 (docs):** fold into `simulate_fwdllm.md` §G/§K (new K-D34, the two-ledger physical/virtual dispatch split);
  update this checkpoint.

**Repro (read-only, from `lib/python/examples/fwdllm`):**
```bash
FS=$(ls -td experiments/run_*_fluxtune_n10_smoke_syn_0_sim | head -1); AGG=$(ls "$FS"/*aggregator.log|head -1)
grep "SIM_GRAD_RECV" "$AGG" | python3 -c "import sys;from datetime import datetime as D;p=None;b=w=t=0
for l in sys.stdin:
 s=D.strptime(l.split(' | ')[0],'%Y-%m-%d %H:%M:%S,%f').timestamp()
 if p is not None:
  d=s-p;t+=1
  if d>2:b+=1;w+=d
 p=s
print(f'commits={t+1} gaps>2s={b} wall_in_waits={w:.0f}s')"          # -> 48% / ~1974s
grep "SIM_GRAD_STUCK_EVICT" "$AGG"                                    # the head-of-line evict
grep "SIM_GRAD_RECV" "$AGG" | grep -oE 'buf_depth=[0-9]+' | sort | uniq -c   # buf_depth stuck at 6
```

---

**DONE (landed + tested):**
- **P1-1/P1-2/P1-3** — enforced `cohort_sequence` rung (EXACT, ungated), V2 mean-guard, `--max-bin` window. Both
  new rungs correctly FAIL the banked pairs (were invisible). New enforced ref: fwdllm 41/13/21, fwdllm_plus
  36/17/21, fluxtune 35/19/19.
- **P1-5 / P1-4** — full-cohort determinism gate (un-gates fwdllm selection/aggregation_sequence/utility;
  fluxtune/fwdllm_plus stay gated); `timing_overrun` DIAG signal rung added; P1-4 assessed redundant. See the
  DEFERRED list below for the full rationale. +9 pytests; banked scoreboard stable (only a new SKIP for the
  overrun rung, which predates the banked logs).
- **P1-6 (partial)** — 11 rung/guard pytests. *Still missing: a LIVE sim==real grad-determinism test (needs P0-2).*
- **P1-8** — banked logs re-run through the upgraded checker.
- **P2-1/P2-3/P2-5/P2-6** — remainder-wait delay model (K-D29: real sleeps `max(0,D−gpu)`, sct `max(gpu,D)`,
  overrun telemetry), crc32 straggler disabled, `perturbation_count` knob (default 10), overrun watch.
- **P2-2 (config)** — factor=2 will be applied via the launch flag (below), not yet run.
- **P2-4 (partial)** — GPU profiled: fwdllm/plus ~1.0s, **fluxtune 7.57s** (JVP 20 passes). Optimization deferred.

**NOT DONE / DEFERRED (pick up here) — in priority order:**
1. **✅ P1-4 / P1-5 RESOLVED (this session):**
   - **P1-5 DONE** — data-driven **full-cohort determinism gate** (`_selection_is_deterministic`:
     `num_chosen==num_candidates` in both modes, else the DETERMINISTIC_SELECTORS name rule as a legacy fallback).
     Un-gates `selection`/`aggregation_sequence`/`utility` for fwdllm (syn_0, K=all → GENUINELY enforced, no
     longer a trivial gated pass) while self-keeping fluxtune (agg_goal=3) + fwdllm_plus (#7 asymmetric eligible)
     gated. **participation deliberately NOT un-gated** — it keys on `round` (constant for fwdllm's data_id axis)
     so it degenerates to a mechanical KS=1.0; cohort_sequence is fwdllm's per-cycle enforcement. Banked scoreboard
     unchanged (41/13, 36/17, 35/19) — the un-gated rungs pass genuinely; utility's fail is a pre-existing pooled
     KS=0.45 (bin-8 desync), not new.
   - **P1-4 ASSESSED REDUNDANT (not built)** — extending `cohort_sequence` to async_cifar10's per-commit shape
     yields a GATED trivial pass (stochastic subset, no `var`, exact order unattainable by design); its set+order
     are already covered there by `aggregation_sequence` (gated) + `inter_arrival_order` + `first_divergence`.
     Building the adapter adds a muddying no-signal rung (principle #16). Revisit only if async_cifar10 ever needs
     an exact-ordered cohort rung.
   - **NEW SIGNAL — `timing_overrun` DIAG rung** (`checks.py`, wired + tested): surfaces the P2-6
     `training_overran` fraction per mode + earliest `(data_id, iter)` overrun — the K-D29 order-determinism tell
     (gpu > modeled D → arrival order can flip → cohort/var break is a TIMING-MODEL limit, not a sim bug). SKIPs on
     the banked logs (predate P2-6); will populate on the databin1 run and tell us WHY fluxtune breaks.
2. **✅ P2-7 DONE (databin1 run + `--max-bin 1` check).** Cascade verified — see P2-7 in PHASE 2. Sync RNG-desync
   root broken (`var` bit-identical, `timing_overrun=0%`); fluxtune overruns 38% as predicted.
3. **✅ P2-7a CODE LANDED (K-D31; operator chose canonicalize-by-trainer_id) — PENDING VALIDATION RUN.** Trainer
   stamps pure `D` (`MODELED_DELAY_S`, both modes); agg `_canonicalize_cohort_commit_order` sorts each cohort by
   `(D, str(end))` before the telemetry snapshot + `aggregate()` (reorders `_per_agg_trainer_list` + trailing
   grad/jvp slice in lockstep). No-op when delays off / already canonical. 428 mode + **9 new**
   `test_fwdllm_commit_canon.py` + 115 async_cifar10 parity green; async byte-identical. **NEXT: the operator runs
   the validation command below** → confirm sync `cohort_sequence` order_match→1.0 (var already bit-identical).
   A dry `--only fwdllm,fwdllm_plus` attempt crashed on a **missing `h5py`** (environment, not code) → operator
   will run in the correct env.
4. **✅ P2-4 GPU optimization — LANDED (K-D32, `jvp_perf_opt`, fluxtune-only, config-gated, bit-identical).**
   Implemented in `calculate_jvp` (trainable_idx) + `tc_transformer_trainer_distribute._train_one_batch`
   (skip 3 diagnostic passes + reuse winner JVP), threaded via main.py/fl_main.py, enabled in both fluxtune yamls
   (default false in trainer_base). Startup `[JVP_PERF_OPT]` confirms 10/10 trainers True (aggregator's eval-only
   trainer False, harmless). 13 pytests + 185 fwdllm mode + 115 parity green. NOT retained: vmap (fp32 FD
   cancellation), fwd-AD (slower). **NEXT: the batch run** confirms fluxtune `timing_overrun`→0 + cohort set
   recovers + real↔sim var parity at multi-bin.
5. **P2-5 (tune fluxtune `perturbation_count`)** — only if the bit-identical cuts above don't fully clear overrun.
   Lower it so GPU < min cohort D. Changes the baseline algorithm (deferred, operator call).
5. **P0-2** — controlled grad-determinism-given-order confirmation (also unblocks the P1-6 live test).
6. **Phase 3** — extend beyond bin 1 once bin-1 parity holds.

**IMMEDIATE NEXT COMMAND (P2-7a validation — confirm K-D31 closes the sync tie):**
```
cd lib/python/examples/fwdllm/expt_scripts
# sync baselines only (fluxtune still overruns → deferred to P2-5):
bash run_sequential.sh --mode both --delays on --delay-factor 2 --max-data-id 1 \
     --max-runtime-s 600 --only fwdllm,fwdllm_plus --yes
python run_parity.py --yes --max-bin 1 --baselines fwdllm fwdllm_plus
# EXPECT: cohort_sequence order_match_frac → 1.0 (was 0.5 / 0.0); var still bit-identical.
# grep the agg log for [COMMIT_CANON] to see the tie reorder fire.
```
(The dry attempt crashed on a missing `h5py` — environment, not code. Run in the env that has the fwdllm deps.)
Expected: fwdllm/fwdllm_plus (GPU≈1s < D/2 of 2–9s) → order deterministic → `cohort_sequence` improves; fluxtune
(GPU 7.57s > D/2) → `[TIMING_OVERRUN]` fires, order still flips → tune `perturbation_count` down next.

**KEY LEARNINGS (durable):**
- The divergence root is **commit ORDER**, not nondeterminism: fwdllm var is a split-half stat over the
  commit-ordered grad list → wrong order → wrong var → threshold flips → per-trainer RNG (seeded once, never
  reset) desyncs → grads diverge ~1%. Grads ARE deterministic given matched order ⇒ exact parity is achievable.
- The order wasn't deterministic because D≈0.4s (÷10) ≪ GPU 1–1.7s → GPU-jitter-dominated. Fix = remainder-wait
  `max(gpu,D)` + per-trainer D + D≫GPU (D/2) ⇒ order = D-order = deterministic.
- fwdllm is **forward-only** (FedFwd, no backprop); the GPU lever is `perturbation_count` (JVP passes), not a
  backprop optimization. fluxtune's 20 passes (7.57s) is the acute cost.
- The old checker was **blind** to all this (cohort rung gated off, V2 KS-only). The new `cohort_sequence` +
  V2-guard are the spec the model fix must satisfy.

---

## ROOT DIAGNOSIS (from the 2026-07-05 investigation — supersedes "order is benign")

**One root explains all three baselines: the sim's commit/receive ORDER ≠ real's actual arrival ORDER.**
- fwdllm's variance is a **split-half statistic over the commit-ORDERED grad list** (`fwdgrad_utils.py:133-158`,
  appended in commit order at `fwdllm_aggregator.py:718-721`). Order matters even for an identical cohort SET.
- Sim commits in **sct order** (`_sim_recv_min_grad` / `_sync_sim_recv_first_k`); real commits in **physical
  arrival order**. Different order → different split-half `var` → `var_good_enough` flips at a different
  `iteration_per_data_id`.
- Each trainer's `torch.Generator` is seeded ONCE and **never reset** (`tc_transformer_trainer_distribute.py:
  222-225`); the perturbation for a given `(data_id, iter)` depends on how many prior forward passes that trainer
  did. One extra iteration → every trainer's RNG desyncs from that point → all later grads differ (~1%). This is
  the fwdllm bin-8 cadence break AND the fluxtune 0/17 cohort break — same cause.
- **fluxtune** (async, agg_goal=3<K): order picks *which 3* commit → cohort wrong from aggregation #1.
- **fwdllm/fwdllm_plus** (sync, K=all): SET always all-10, but order feeds the split-half var → cadence break.

**Why the order isn't reproducible right now (timing model mis-set):**
- These runs: `enable_training_delays: true` but `training_delay_factor: null` → D≈**0.4s flat** (not the
  per-trainer registry 4–18s). GPU compute is **1–1.7s** (`trainer_round.real_gpu_time_s`), i.e. **D ≪ GPU**.
- So arrival order is dominated by **GPU jitter** (run-to-run, non-deterministic), NOT by a deterministic
  per-trainer mobile delay. Operator's intended model: mobile delay ≫ GPU, so order = delay order = deterministic.
- fwdllm's real delay is **flat-additive** (`_delay_s` slept ON TOP of GPU, `FedSgdTrainer.py:510-538`), not the
  **remainder-wait** (`delay − gpu`) that async_cifar10 / async_google_speech use. No `gpu > delay` handling.
- Sim sct adds a crc32 `_sim_straggler_offset_s` (spread 0.9s) + `_wan_s` that **real has no counterpart for**
  (`FedSgdTrainer.py:550-565`, `:623-624`) → an extra sim-only reorder vs real.

**Foundational question — likely ANSWERED by the code:** grads are deterministic GIVEN matched commit order
(batch selection is deterministic in `data_id`; seeds are mode-invariant; JVP is fixed). The ONLY divergence
sources are commit-order + retry-count, both downstream of order. ⇒ **it's a sim ORDER bug, not nondeterminism.**
Needs one controlled confirmation (P0-2). If confirmed, exact cadence parity IS achievable.

**Checker is blind to all of this (Agent A):** `aggregation_sequence` (cohort set) is gated `ok=True` for
stochastic selectors (`DETERMINISTIC_SELECTORS=∅`, `checks.py:882`) and keys on `round` not `cycle_data_id`;
`inter_arrival_order` is WARN-only; `v2_var_trajectory` is **KS-only, no mean guard** (a 1% offset passes); no
`--max-bin`. Both divergences pass today.

---

## PHASE 0 — settle decisions + the foundational question (BLOCKING)
- [x] **P0-1 Operator decisions — RESOLVED 2026-07-05** (see "Resolved decisions" below): remainder-wait delay;
      D configurable, start at **D/2** (not full registry); **optimize GPU now** (prove forward < backward);
      checker rungs **hard-FAIL**. New scope added: fluxtune JVP forward-pass-count knob + budget-overrun ordering
      watch.
- [ ] **P0-2 Confirm grad-determinism-given-order.** Controlled check: force identical commit order in a sim and
      a real short run (or two real runs) and diff per-iteration `var` + a grad norm. Expect bit-match if order
      matches. Decides sim-bug (exact target) vs nondeterminism (distributional target).

## PHASE 1 — make the CHECKS/PYTESTS exhibit the properties (TOP PRIORITY, test-driven)
*Build the failing check FIRST, confirm it catches the current divergence on banked logs, then fix the model.*
- [x] **P1-1 DONE — enforced `cohort_sequence` rung landed** (`parity/checks.py::cohort_sequence_parity`, tier
      EXACT, ungated, keyed on the `_fwd_cadence_cycles` stream). Per cycle asserts cohort SET + receive-ORDER +
      cadence tuple + `var` value (rel-tol 1e-3). Wired into `run_all_parity` + `CHECK_META` + the `run_parity.py`
      headline. **Now FAILS all 3 banked pairs** (was invisible).
- [x] **P1-2 DONE — V2 mean-guard added** (`var_trajectory_parity`, `mean_tol_rel=0.02`). **Now FAILS all 3**
      (KS-only used to pass the ~1% offset).
- [x] **P1-3 DONE — `--max-bin` window** threaded through `_fwd_cadence_cycles` → V1/V2/V3/V4/V5 + `cohort_sequence`
      + `run_all_parity` + `run_parity.py` + `cli.py`. `run_parity.py --max-bin 1` verified.
- [x] **P1-4 ASSESSED REDUNDANT (not built).** Extending `cohort_sequence` to async_cifar10's per-commit shape is
      a GATED trivial pass (stochastic subset, no `var`); async_cifar10's set+order are already covered by
      `aggregation_sequence` + `inter_arrival_order` + `first_divergence`. No new signal → not built (principle #16).
- [x] **P1-5 DONE — data-driven full-cohort determinism gate** (`_selection_is_deterministic` /
      `_full_cohort_selection` / `_has_cohort_counts`, `checks.py`). Un-gates `selection`/`aggregation_sequence`/
      `utility` when `num_chosen==num_candidates` in both modes (fwdllm syn_0) → GENUINELY enforced; fluxtune +
      fwdllm_plus stay gated (subset / #7 asymmetric eligible); legacy no-count telemetry falls back to the old
      selector-name rule (no regression). participation EXCLUDED (round-keyed → mechanical KS on fwdllm's data_id
      axis). `decision_determinism` left DIAG (a localizer by design, not an enforce target). 6 gate pytests +
      banked-log validation (scoreboard stable). NEW `timing_overrun` DIAG rung surfaces the P2-6 overrun signal.
- [~] **P1-6 PARTIAL — rung + guard pytests landed** (`TestCohortSequence` ×9, `TestVarTrajectoryMeanGuard` ×2 in
      `tests/mode/test_parity_checks.py`). STILL TODO: a LIVE sim==real grad-determinism test given matched order
      (needs P0-2 infra).
- [ ] **P1-7 Cross-example + cross-baseline coverage.** Confirm the rung runs on async_cifar10's per-commit shape
      (P1-4). fwdllm_plus & fluxtune already covered by the banked-log run.
- [x] **P1-8 DONE — banked logs re-run through the upgraded stack.** New ENFORCED reference (full run):
      **fwdllm 41/13/21, fwdllm_plus 36/17/21, fluxtune 35/19/19** — each now lists `cohort_sequence` +
      `v2_var_trajectory` in FAILS. At **`--max-bin 1`**: cadence (V1) passes but `cohort_sequence` still fails —
      fwdllm shows `order_match_frac=0.0`, `var_match_frac=0.5` (real `var=0.371605` vs sim `0.371067` at cycle 0),
      proving the order→split-half-var→grad-desync mechanism. Regression: 337 mode/telemetry/selector +115
      async_cifar10 parity tests green.

## PHASE 2 — fix the timing/order MODEL until the (now-failing) checks pass
- [x] **P2-1 DONE — remainder-wait delay model** (`FedSgdTrainer._emulate_training_delay(gpu_time_s)` →
      `(modeled_delay, remaining, overran)`; real sleeps `max(0, delay−gpu)`, sim skips; `[TIMING_OVERRUN]` log).
      sct is now `max(gpu, delay)` (not `gpu+delay`). Supersedes K-D2 → **K-D29**. Tests rewritten
      (`test_fwdllm_trainer_sim_duration.py`), full suite green.
- [x] **P2-2 DONE (config) — factor=2 (D/2)** via launch flag `--delay-factor 2` (knob stays live; `training_delay_
      factor` per-config). Applied at launch below.
- [x] **P2-3 DONE — crc32 straggler offset disabled** (`sim_straggler_spread_s: 0.0` in all 3 sim yamls). The
      per-trainer registry delays now supply the completion spread; the offset would re-noise the deterministic
      order. `_wan_s` already 0. (Helper kept flag-gated for its unit tests.)
- [~] **P2-4 PARTIAL — GPU profiled + per-step telemetry LANDED.** Coarse (from `trainer_round`): fwdllm/
      fwdllm_plus **~1.0s** (cos-sim path), **fluxtune 7.57s mean** (JVP, 20 passes). NEW **per-step telemetry**
      (`EVENT_STEP_TIMING` via `timer_decorator` — no-op unless telemetry on) + plotter
      `expt_scripts/plot_step_timing.py` (fine `step_timing` funcs + coarse `trainer_round` phases; PNG + table).
      **Finding from banked logs:** the dominant REAL wall is `mqtt_fetch_s` (fwdllm 4.2s/round, **fluxtune
      18.3s/round**) ≫ `gpu_compute_s` (1.1 / 3.6s) — that's the #11 real-transport cost the SIM already skips, not
      a GPU lever. The FINE GPU sub-step split (perturbation-selection vs per-batch JVP) populates on the next run
      (telemetry now in place). fwdllm is forward-ONLY → the GPU lever is `perturbation_count` (P2-5).
- [x] **P2-5 DONE (knob) — `perturbation_count` config knob** (default 10 = byte-identical) threaded
      config→`main.py`/`fl_main.py`→`tc_transformer_trainer_distribute.py`, replacing the hardcoded `1*10` /
      `range(0,10)` in all 4 sites. Lowering it cuts fluxtune's forward-pass cost. LEFT AT 10 for this run
      (operator: fluxtune tuning is a next step). True per-baseline JVP cost measured (P2-4).
- [x] **P2-6 DONE — budget-overrun telemetry** (`training_overran` + `remaining_time_s` on `trainer_round`,
      `[TIMING_OVERRUN]` warning). This is the "keep a tab" watch: if actual GPU > modeled sct, the update arrives
      after the vclock passed its sct → out-of-order commit. Expect it to fire on fluxtune this run.
- [x] **P2-7 DONE — cascade VERIFIED on the databin1 run** (`--delay-factor 2 --max-data-id 1`,
      `smoke_logs/20260705_150145` → `experiments/run_20260705_15*`; `run_parity.py --yes --max-bin 1`).
      **RNG-desync root BROKEN for the sync baselines:** on bin 1 `cohort_sequence` shows `var` **bit-identical**
      real↔sim (fwdllm `0.21480107…`, fwdllm_plus `0.37160512…`; banked had `var_match=0.5`, real≠sim) +
      `set_match=1.0` + `cadence_match=1.0` + **`timing_overrun=0%`** ⇒ grads mode-invariant on bin 1. RESIDUAL:
      `order_match` 0.5 (fwdllm) / 0.0 (fwdllm_plus) from ONE benign **delay-TIE** — trainers 3 & 9 (…372/…378)
      both drew `training_delay_s=13.0` (`_metadata/trainer_registry.yaml`, → D=6.5); their sct ties, real breaks
      it by physical arrival & sim by sct-sort, both in the SAME split-half ⇒ var/grads unchanged. **fluxtune:**
      `timing_overrun=38%` (GPU≈4.2s > min cohort D 2.0/2.5/3.5s), `set_match=0.22`, `var` mean off 29% → genuine
      break, timing-model-limited → P2-5 (`perturbation_count`↓ so GPU<minD, and/or `delay_factor`=1). Parity
      counts (`--max-bin 1`): fwdllm 45/4/27, fwdllm_plus 43/6/27, fluxtune 42/8/24 — each lists `cohort_sequence`
      in FAILS. **OPEN FORK (P2-7a):** close the benign sync tie — canonicalize equal-D commit order by trainer_id
      in BOTH modes, or relax the order rung to var-equivalent. Operator call (touches real commit path / rung
      contract).

## PHASE 3 — validate on data bin 1, then extend
- [ ] **P3-1 Fresh databin1 loop** (`--max-data-id 1`, delays configured per P2-2) across the 3 baselines.
- [ ] **P3-2 Run parity + sanity checks** on the databin1 pairs; require the logical rung GREEN for all three.
- [ ] **P3-3 Extend length** only after bin-1 parity holds; re-check where (if) it breaks; iterate.
- [ ] **P3-4 Fold outcomes** into simulate_fwdllm.md (§G one-liners, retire closed issues) and DELETE this tracker.

---

## RESOLVED DECISIONS (operator, 2026-07-05)
1. **Delay model → remainder-wait.** `wall = delay` (GPU hidden inside), aligning fwdllm to async_cifar10. (P2-1)
2. **Delay magnitude → configurable, start at D/2** (factor=2), not full registry, not ÷10. Raise only if order is
   still GPU-dominated. Tied to proving forward < backward and to the true JVP cost. (P2-2)
3. **GPU overhead → optimize NOW.** Prove forward-pass < backprop; root-cause the 5–8× vs cifar10. Forward latency
   grows with model size, so this is a standing invariant to hold with margin. (P2-4)
4. **fluxtune JVP cost is under-counted → new scope.** 2 passes/perturbation × ~10 = ~20 passes; make the count a
   configurable knob and measure the real full-JVP GPU cost into the sct. (P2-5) Watch budget-overrun→ordering as
   the next fluxtune step. (P2-6)
5. **Checker enforcement → hard FAIL.** Cohort/order rung + V2 mean-guard are EXACT & enforced (ungated). The
   banked pairs will fail until the model is fixed — that's the gate. (P1-1/P1-2/P1-8)

## ANCHORS (for implementers)
- Split-half var: `examples/fwdllm/aggregator/.../fwdgrad_utils.py:133-158`; order append `fwdllm_aggregator.py:718-721`.
- RNG-once seed: `.../tc_transformer_trainer_distribute.py:222-225`, draws `:306/:345/:414`.
- Delay/sct: `FedSgdTrainer.py:510-538` (sleep), `:623-629` (sct), `:550-565` (straggler offset).
- Sim commit order: `fwdllm_aggregator.py:747-927` (`_sim_recv_min_grad`), sync `top_aggregator.py:360+`.
- Checker rungs: `async_cifar10/scripts/parity/checks.py` — `aggregation_sequence:1071`, `inter_arrival_order:1323`,
  `v2 var_trajectory:3753`, `v1:3716`, gating `DETERMINISTIC_SELECTORS:882`, wiring `run_all_parity:4155`.
- Standalone logical diff: `expt_scripts/logical_parity.py`.
