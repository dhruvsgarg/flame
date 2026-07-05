# TEMP task tracker — logical real↔sim parity (delete when folded into simulate_fwdllm.md §G)

**Goal.** Prove the sim takes the SAME logical steps in the SAME order as real (same cohorts, same receive
order, same variance cadence, same grads) up to data bin 1 across fwdllm / fwdllm_plus / fluxtune — and make the
parity CHECKS + PYTESTS actually *exhibit* these properties (they don't today). Only then chase the time
dimension. Correctness before speed; no hacks (simulate_fwdllm.md principles #14/#16).

---

## ⏸ SESSION CHECKPOINT (2026-07-05, paused for a break — resume here)

**Code + tests are landed and GREEN; NO run has been launched yet.** Full suite: 606 mode/telemetry/selector +
115 async_cifar10 parity green; trainer files compile. NOTHING is half-edited. Nothing committed yet at pause →
**commit + push done at end of session** (this checkpoint is the resume anchor).

**DONE (landed + tested):**
- **P1-1/P1-2/P1-3** — enforced `cohort_sequence` rung (EXACT, ungated), V2 mean-guard, `--max-bin` window. Both
  new rungs correctly FAIL the banked pairs (were invisible). New enforced ref: fwdllm 41/13/21, fwdllm_plus
  36/17/21, fluxtune 35/19/19.
- **P1-6 (partial)** — 11 rung/guard pytests. *Still missing: a LIVE sim==real grad-determinism test (needs P0-2).*
- **P1-8** — banked logs re-run through the upgraded checker.
- **P2-1/P2-3/P2-5/P2-6** — remainder-wait delay model (K-D29: real sleeps `max(0,D−gpu)`, sct `max(gpu,D)`,
  overrun telemetry), crc32 straggler disabled, `perturbation_count` knob (default 10), overrun watch.
- **P2-2 (config)** — factor=2 will be applied via the launch flag (below), not yet run.
- **P2-4 (partial)** — GPU profiled: fwdllm/plus ~1.0s, **fluxtune 7.57s** (JVP 20 passes). Optimization deferred.

**NOT DONE / DEFERRED (pick up here) — in priority order:**
1. **⚠ P1-4 and P1-5 were SKIPPED before jumping to Phase 2** (operator flagged this):
   - **P1-4** — uniform cohort-record ADAPTER so the `cohort_sequence` rung also runs on **async_cifar10**'s
     per-commit `agg_round` shape (fwdllm's 3 baselines already work; async_cifar10 does not yet).
   - **P1-5** — enforce `selection`/`decision_determinism`/`selection_detail` where order is deterministic
     (populate `DETERMINISTIC_SELECTORS` / per-baseline gate; today WARN-only via the empty set at `checks.py:882`).
2. **P2-7 — LAUNCH THE databin1 RUN** (the immediate experimental next step; command below), then re-run the
   checker `--max-bin 1` and confirm the cascade: fwdllm/fwdllm_plus `cohort_sequence` should extend past bin 8;
   fluxtune expected to still break (GPU overrun) → confirms P2-5 tuning is the fluxtune next step.
3. **P0-2** — controlled grad-determinism-given-order confirmation (also unblocks the P1-6 live test).
4. **P2-4 (optimize)** / **P2-5 (tune fluxtune `perturbation_count`)** — reduce fluxtune GPU below its D budget.
5. **Phase 3** — extend beyond bin 1 once bin-1 parity holds.

**IMMEDIATE NEXT COMMAND (P2-7):**
```
cd lib/python/examples/fwdllm/expt_scripts
bash run_sequential.sh --mode both --delays on --delay-factor 2 --max-data-id 1 --max-runtime-s 1800 --yes
python run_parity.py --yes --max-bin 1     # then inspect cohort_sequence per baseline
```
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
- [ ] **P1-4 Uniform cohort-record adapter** so async (per-commit `agg_round`, `contributing=[end]`) and sync
      (per-cycle cohort) run ONE cohort/order diff. NOTE: fluxtune already emits per-cycle cadence fields (it's the
      fwdllm aggregator with `is_async`), so `cohort_sequence` ALREADY runs on all 3 fwdllm baselines. The adapter
      is only needed to extend the rung to async_cifar10's per-commit shape — lower priority.
- [ ] **P1-5 Enforce selection/determinism rungs** where order is deterministic: populate
      `DETERMINISTIC_SELECTORS` (or a per-baseline determinism gate) so `selection`/`decision_determinism`/
      `selection_detail` stop being WARN-only.
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
- [~] **P2-4 PARTIAL — GPU profiled** (from telemetry): fwdllm/fwdllm_plus **~1.0s** (cos-sim path, ~1 forward
      pass); **fluxtune 7.57s mean, max 59.6s** (JVP path, 2×10 = 20 passes). fwdllm is forward-ONLY (FedFwd, no
      backprop) so "forward<backprop" is moot — the real GPU lever is the **perturbation count** (P2-5). fluxtune's
      7.57s WILL overrun a D/2 budget (2–9s) → tracked via P2-6, tuned via P2-5 next step.
- [x] **P2-5 DONE (knob) — `perturbation_count` config knob** (default 10 = byte-identical) threaded
      config→`main.py`/`fl_main.py`→`tc_transformer_trainer_distribute.py`, replacing the hardcoded `1*10` /
      `range(0,10)` in all 4 sites. Lowering it cuts fluxtune's forward-pass cost. LEFT AT 10 for this run
      (operator: fluxtune tuning is a next step). True per-baseline JVP cost measured (P2-4).
- [x] **P2-6 DONE — budget-overrun telemetry** (`training_overran` + `remaining_time_s` on `trainer_round`,
      `[TIMING_OVERRUN]` warning). This is the "keep a tab" watch: if actual GPU > modeled sct, the update arrives
      after the vclock passed its sct → out-of-order commit. Expect it to fire on fluxtune this run.
- [ ] **P2-7 Verify the cascade closes** — the launch below is the test: fwdllm/fwdllm_plus (GPU≈1s < D/2)
      should extend cohort_sequence parity past bin 8; fluxtune (GPU overrun) is expected to still break → confirms
      P2-5 is the fluxtune next step.

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
