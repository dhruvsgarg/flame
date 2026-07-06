# TEMP task tracker — logical real↔sim parity (delete when folded into simulate_fwdllm.md §G)

**Goal.** Prove the sim takes the SAME logical steps in the SAME order as real (same cohorts, receive order, variance
cadence, grads) up to data bin 1 across fwdllm / fwdllm_plus / fluxtune, and make the parity CHECKS + PYTESTS exhibit
these properties. Correctness before speed; no hacks (simulate_fwdllm.md principles #14/#16).

---

## ⏸ RESUME HERE (2026-07-06) — #15 fix LANDED, P3 run is the gate

**#15 fluxtune `sim_rate=0.50` fix (`sim_compute_truthful_gate`) is CODE-COMPLETE + unit-green; the P3 validation run
is the only remaining step and needs the GPU + h5py env (operator runs it).** Everything else below is either DONE or
a lower-priority open task.

### ▶ IMMEDIATE NEXT COMMAND (P3 — validate the #15 fix)
```bash
cd lib/python/examples/fwdllm/expt_scripts
bash run_sequential.sh --only fluxtune --mode both --delays on --delay-factor 1 --max-data-id 2 --yes
python run_parity.py --yes --max-bin 1 --baselines fluxtune
# then inspect the new sim agg log (see "Repro" at the end of the #15 section):
#   FS=$(ls -td ../experiments/run_*_fluxtune_n10_smoke_syn_0_sim | head -1); AGG=$(ls "$FS"/*aggregator.log|head -1)
#   grep -c SIM_GRAD_STUCK_EVICT "$AGG"                       # expect ~0
#   grep SIM_GRAD_RECV "$AGG" | grep -oE 'phantom_skip=[0-9]+' | tail -1   # expect > 0 (guard firing)
```
**EXPECT (PASS):** no/near-zero `[SIM_GRAD_STUCK_EVICT]`, `phantom_skip` rising, the `gap>2s` wall (~1974s baseline)
collapsing, concurrency↑, **`sim_rate` → >1** — AND `cohort_sequence`/`var`/`staleness` parity **UNCHANGED** vs the
flag-off banked run. **If parity MOVES:** the compute cap is too tight (skipping a genuine straggler) → raise
`sim_gate_compute_cap_s` (yaml, default 10.0) and re-run; if it still moves, the guard is reordering commits →
revisit (the fix must only remove dead wall, not change WHICH grad commits WHEN).

### Other OPEN fronts (after P3), priority order
1. **bin-7 nondeterminism → relax the parity target (SHARED, correctness-of-CHECK).** SYNC full-run cadence breaks at
   bin 7 even with order 41/41 matched (K-D31): ~1e-3 GPU fp16 grad jitter, amplified by the split-half variance
   ratio, flips the `var<0.3` gate at (7,2) — NOT a sim bug. **Task:** keep `cohort_sequence` EXACT but scoped to
   `--max-bin 1`; ADD a DISTRIBUTIONAL cadence/var rung (mean-band + KS + `var_good` fraction) for the full run.
   Confirm with a 2-real-run diff first (= **P0-2**, below).
2. **P0-2 — grad-determinism-given-order confirmation.** Force identical commit order in a sim + a real short run (or
   two real runs), diff per-iteration `var` + a grad norm. Decides exact (sim-bug) vs distributional (nondeterminism)
   target beyond bin 1. Also unblocks the P1-6 LIVE sim==real grad-determinism pytest (still TODO).
3. **#1d / P2-5 — fluxtune cohort SET diverges (thin-margin overrun), fluxtune only.** `set_match=3/272`. `jvp_perf_opt`
   (K-D32) cut GPU mean 7.57→3.61s (<4.0s budget) but the TAIL (4.1–5.4s) overruns on the two doubled GPUs (10/8) →
   order flips → wrong 3-of-K commit. **Task:** once #15's faster commits land, re-check; if a residual tail remains,
   tune `perturbation_count`↓ (P2-5, a knob already wired, default 10; LOWERING changes the baseline algorithm → operator
   call) so GPU < min cohort D, and/or `--delay-factor` up.
4. **#7 — fwdllm_plus real ~4× slower/round; at syn_0 real sees ~4.9 eligible vs sim ~9.6.** Not a sim bug. Profile the
   per-iteration reselection + oracular-read cost from the banked per-phase log; explain the eligible-count gap.
5. **#11 — real-mode critical-path waste** (`sleep(0.1)` MQTT-settle, one-grad-per-poll drain tail), real-only, ZERO
   parity impact (sim already skips). Deferred to a validated pass (removing it needs a real run; principle #8/#11c).
6. **Phase 3 — extend beyond bin 1** once bin-1 parity holds; re-check where it breaks; iterate. Then C1/C2
   convergence (distributional target) at matched `data_id` → gate to Phase 2 (unavailability).

### D3 (optional, secondary) — N=20/C=10/K=3 run
Adds a 10-trainer idle pool; confirms real fills C from fresh idle trainers while a returner waits for its commit
(validates F5) and the sim reproduces the selection sequence. `run_sequential.sh --only fluxtune --mode both`,
`num_trainers=20`, selector `c: 10`, `agg_goal: 3`.

---

## 📚 FELIX GROUNDING (async_cifar10) — the reference model for #15 (durable; keep)

*How async_cifar10's felix drain ACTUALLY behaves (read from code + `async_cifar10/PARITY.md`). Anchors: **asyncfl** =
`flame/mode/horizontal/asyncfl/top_aggregator.py`; **fwdllm** = `flame/mode/horizontal/syncfl/fwdllm_aggregator.py`.*

- **F1 — Felix's expected-sct arrival gate is INERT (gate_holds=0), not load-bearing.** It DOES stamp
  `_sim_inflight_expected[end]=dispatch_vclock+budget` (asyncfl:1668-71, a LOWER BOUND) and CAN block (`earlier_stuck`,
  asyncfl:430-446), but over a cifar run `gate_holds=0` (asyncfl:157-162; PARITY.md:294-296/321-323): cifar GPU ≈
  **0.4s**, so every in-flight update is already arrived+buffered when the drain runs. Felix's EFFECTIVE behavior =
  sort arrived updates by sct and commit. The gate is a dormant safety net (min-budget lower bound + clock-jump clamp).
- **F2 — Felix re-dispatches on COMMIT; the felix trainer idles in recv — but sub-second, only because compute≈0.**
  `_sim_hold_busy_slots` (asyncfl:1453-1500) at the agg-goal boundary holds each busy trainer in
  `selected_ends`/`all_selected`/`_sim_pending_commit` until its update commits (released asyncfl:618-627); the trainer
  blocks in `_fetch_weights`→`channel.recv` (`syncfl/trainer.py:187`) meanwhile. So felix IS hold-to-commit —
  **fwdllm's K-D17b is a FAITHFUL port.** Precondition: return→commit is sub-second in cifar → ≈0 idle wall.
- **F3 — Two ledgers already SEPARATE in felix.** PHYSICAL (GPU/MQTT, `_sim_buffer`, `cleanup_recvd_ends`) vs VIRTUAL
  (selection eligibility `_sim_pending_commit`/`all_selected`/`selected_ends`, staleness gate `_sim_inflight_expected`,
  vclock). "Busy ≠ UN_AVL": a busy trainer holds a SLOT (`extra = c − len(selected_ends)`), released on commit
  (asyncfl:1459-62; PARITY.md:324-328).
- **F4 — Regime table (why the same mechanism transfers to sync fwdllm but breaks fluxtune):**

  | baseline | GPU | agg_goal vs c | return→commit idle | gate | outcome |
  |---|---|---|---|---|---|
  | cifar felix | ~0.4s | (varies) | sub-second | inert | sim_rate ≫ 1 |
  | fwdllm/plus (sync) | ~1.0s | K=c (barrier) | ~1 GPU pass, ALL commit | inert | `sim_rate` 2.9-3.0 ✓ |
  | **fluxtune (async)** | **~4s** | **3 ≪ 10** | **multi-sec → 30s** | **load-bearing → stall** | **`sim_rate` 0.50 ⛔** |

- **F5 — Hold-to-commit is a CORRECTNESS CHECK, not over-restriction.** The aggregator marks a trainer busy/idle by
  whether it RESPONDED to its task; a trainer is freed (re-selectable) ONLY once its returned update is
  PROCESSED/COMMITTED. That enforces three guards: **(a)** never dispatch a version it already computed
  (same-shard×same-version = wasted grad); **(b)** never dispatch while still computing; **(c)** never dispatch while
  its returned update is uncommitted. Do NOT weaken it. Concurrency: with **N>C** a fast returner is held while fresh
  idle trainers keep C busy; with **N=C** (fluxtune 10/10/3, NO idle pool) concurrency is agg-cadence-limited and
  real's **3.37 is a duty cycle** (gpu/max(gpu,D) ≈ 4/12 ≈ 0.34 → N×0.34), NOT under-utilization.
- **F6 — The bug is purely the COMMIT PATH STALLING; commit RATE is the throughput lever.** Because trainers are freed
  only on commit, commit rate = free rate = throughput. cifar's commit path is instant → held trainers barely idle;
  fluxtune's STALLS (see #15) → held trainers idle 30s. Fix = fast/non-stalling commit path; the gate must only wait
  on a genuinely-COMPUTING trainer, never a phantom. Hold-to-commit untouched.
- **F7 — Anchors.** Felix gate/HOLD asyncfl:359-448; clamp 461-476; `_sim_hold_busy_slots` 1453-1500; commit-release
  618-627; expected-sct seed 128-137/1668-1671; sct formula (trainer) `async_cifar10/trainer/pytorch/main.py:838-846`;
  PARITY.md gate-inert 294-296/321-323, residence §3.resid, past-dating 537-563.

---

## ⭐ #15 fluxtune `sim_rate=0.50` — COMMIT-PATH STALL (root confirmed; fix LANDED, P3 pending)

**Superseded framings (do not revisit):** (1) "GPU-pipelining loss / decouple dispatch" — felix's gate is INERT (F1).
(2) "re-dispatch on return / hold-to-commit is over-restrictive" — hold-to-commit is a CORRECTNESS check (F5). The
final root is a commit-path stall.

**ROOT (D1/D2 CONFIRMED on `run_20260705_204619` sim / `_202448` real):** hold-to-commit correctly frees a trainer
only on commit, so commit RATE sets throughput (F5/F6). fluxtune's `_sim_recv_min_grad` `earlier_stuck` gate blocks
real wall on a PHANTOM `_sim_inflight_expected` entry — a trainer stamped expected-at-DISPATCH that isn't computing
(idle-in-recv, waiting for weights the gate-blocked single-threaded aggregator can't send) → 30s failsafe → the
correctly-held trainers idle ~30s instead of ~one GPU pass → sim_rate 0.50.

**EVIDENCE:**
- Gate holds ALREADY-ARRIVED grads: every top stall commits a grad that arrived (`[MSG_ARRIVAL]`) 20-23s before the
  stall; `buf_depth=7` on 99.7% of commits (buffer pinned full). Awaited trainers provably `recv_wrapper`-IDLE
  (0370 idle 13s, 0371 44.6s spanning the stall).
- Dominant tax: inter-commit gap mean 2.20s; 496 commits pay ~2s + 24 gaps ≥10s + 1×31s failsafe = **1974s = 82% of
  the 2398s wall**. `sim_rate` 0.494.
- Closed self-throttle: arrival rate 0.459/s == commit rate 0.456/s. Concurrent compute **sim 1.65 vs real 7.69**.
  `recv_wrapper` idle: sim mean 17.9s vs real median 0.01s. Compute mode-invariant (~3.6s).
- Mechanism = **(c1)** optimistic `exp` stamped at DISPATCH (`:2819`, contention-free lower-bound budget) + **(c3)**
  `sim_staggered_redispatch=False` → `_sst=_round_now` (`:2813`) collapses fresh-cohort exp onto the round frontier so
  it always looks earlier than buffered later-sct grads → `earlier_stuck` stays armed + **(c2)** single-threaded agg
  BLOCKS in the `for _pass` loop (`:772`), can't run `_distribute_weights_async` → can't send weights to the trainer
  it waits on → deadlock to grace/30s failsafe. `SIM_R1_DISPATCH=0` — hold-to-commit is clean, NOT implicated.
- Also: fluxtune's self-throttle is the variance-gating keep-training loop — after the initial `Sent 10 WEIGHTS`,
  distributes are `0 WEIGHTS + 1–2 VAR=bad` (only 1–2 trainers told to keep training/cycle). Real frees on grad
  PROCESS (recv 0.01s, non-residence path) → ~8 compute; sim defers to commit → the slow commit starves re-dispatch.

**FIX — ✅ P1/P2 LANDED (`sim_compute_truthful_gate`, flag-gated, default off = byte-identical; fwdllm_aggregator-only
→ async_cifar10 untouched):**
- **P1:** `_sim_dispatch_wall[end]` stamped at the real `channel.send` (`fwdllm_aggregator.py:2836`). The
  `earlier_stuck` gate (`:856`) skips any `_sim_inflight_expected` entry whose last dispatch is older than
  `sim_gate_compute_cap_s` (default **10.0s** > ~5.6s max JVP compute) OR never dispatched → a stamped-but-idle phantom
  no longer blocks a ready commit; a genuine in-window straggler is STILL held (sct order preserved). Hold-to-commit,
  the sct-ordered drain, K-D12 carried surplus, K-D27 `_sim_pending_commit` — all UNTOUCHED. `[SIM_GRAD_RECV]` now
  emits `phantom_skip=`. Enabled in `expt_scripts/fluxtune_n10_smoke_sim.yaml` (`sim_compute_truthful_gate: true`,
  `sim_gate_compute_cap_s: 10.0`).
- **P2:** 4 new `TestComputeTruthfulGate` (phantom skipped; stale-dispatch=phantom; in-window straggler STILL held;
  flag-off byte-identical) in `tests/mode/test_fwdllm_sim_grad_loop.py`; +189 fwdllm mode tests green.
- **P3 = the gate (RESUME HERE, command at top).** Preserve the committed LOGICAL trace (grads/versions/order/
  staleness = real; #1d/cohort parity) — the fix must only remove dead wall. **P4:** §G one-liner after P3 passes.

**Repro (read-only, from `lib/python/examples/fwdllm`):**
```bash
FS=$(ls -td experiments/run_*_fluxtune_n10_smoke_syn_0_sim | head -1); AGG=$(ls "$FS"/*aggregator.log|head -1)
grep SIM_GRAD_RECV "$AGG" | python3 -c "import sys;from datetime import datetime as D;p=None;b=w=t=0
for l in sys.stdin:
 s=D.strptime(l.split(' | ')[0],'%Y-%m-%d %H:%M:%S,%f').timestamp()
 if p is not None:
  d=s-p;t+=1
  if d>2:b+=1;w+=d
 p=s
print(f'commits={t+1} gaps>2s={b} wall_in_waits={w:.0f}s')"    # baseline sim: 48% / ~1974s -> expect near-0
grep SIM_GRAD_STUCK_EVICT "$AGG"; grep SIM_GRAD_RECV "$AGG" | grep -oE 'phantom_skip=[0-9]+' | tail -1
```

---

## DONE (landed + tested — terse; do not redo)
- **#14/#1c/#13/#12c fixed:** MQTT join-notify race (#14); R1 two-ledger bridge (K-D27, `SIM_R1_DISPATCH` 238→0, #1c);
  drain-stall felix port (K-D28, `sim_rate` 0.06→0.30, #13); `--delay-factor 1` → sync `sim_rate` 0.94→2.9-3.0 (#12c).
- **K-D29 remainder-wait delay model:** real sleeps `max(0,D−gpu)`, sct = `send + max(gpu,D)`, `[TIMING_OVERRUN]`
  telemetry (P2-1/6). crc32 straggler offset disabled (P2-3). `perturbation_count` knob, default 10 (P2-5, wired).
- **K-D30 full-cohort determinism gate + `timing_overrun` DIAG rung** (P1-5): un-gates
  selection/aggregation_sequence/utility for fwdllm (K=all); fluxtune/fwdllm_plus stay gated. P1-4 assessed redundant.
- **K-D31 canonical `(D, trainer_id)` cohort commit order — VALIDATED (P2-7/P2-7a).** Databin1 `--max-bin 1`: sync
  `cohort_sequence` ok=true, set/order/var/cadence=1.0 (the benign delay-tie closed; var already bit-identical). 428
  mode + 9 canon + 115 async parity green; async byte-identical.
- **K-D32 fluxtune JVP perf-opt (`jvp_perf_opt`, config-gated, bit-identical):** trainable-only FD + skip 3 diagnostic
  passes + reuse winner JVP → fluxtune −37% (7.57→3.61s mean). NOT retained: vmap (fp32 FD cancellation), fwd-AD.
- **K-D33 pinning:** trainer `[PIN]` self-report, `[LOAD_BALANCE]` check, aggregator GPU pin (least-loaded/idle). 8
  GPUs balanced round-robin (earlier "under-provisioned" read was a misread). 120 launch tests green.
- **Checks/pytests (P1-1/2/3/8):** `cohort_sequence` rung (EXACT, ungated), V2 mean-guard, `--max-bin` window. Banked
  ENFORCED ref (full run): fwdllm 41/13/21, fwdllm_plus 36/17/21, fluxtune 35/19/19.
- **#15 P1/P2** (this session) — see the #15 section above.

## KEY LEARNINGS (durable)
- **fluxtune #15 = commit-path stall, not residence/pipelining/GPU.** Hold-to-commit is correct (F5); commit RATE is
  the throughput lever (F6). Felix's gate is inert (F1) — cifar hides the cost via sub-second compute.
- **SYNC divergence root = commit ORDER via split-half var, but exact cadence has a bin-7 float-nondeterminism wall.**
  fwdllm var is a split-half stat over the commit-ordered grad list → wrong order → wrong var → `var<0.3` gate flips →
  per-trainer RNG (seeded once, never reset) desyncs → grads diverge ~1%. Grads ARE deterministic given matched order
  ⇒ exact parity achievable ≤bin 1; beyond ~bin 6, ~1e-3 GPU fp16 jitter amplified by the variance ratio breaks it →
  target must go DISTRIBUTIONAL (open task 1).
- **Order made deterministic** by remainder-wait `max(gpu,D)` + per-trainer D ≫ GPU (K-D29) + canonical tie-break
  (K-D31). fwdllm is forward-ONLY → the GPU lever is `perturbation_count`, not backprop.

## ANCHORS
- Split-half var `aggregator/.../fwdgrad_utils.py:133-158`; commit-order append `fwdllm_aggregator.py:718-721`.
- RNG-once seed `.../tc_transformer_trainer_distribute.py:222-225`.
- Delay/sct `FedSgdTrainer.py:510-538`(sleep)/`:623-629`(sct). Sim drain `fwdllm_aggregator.py:_sim_recv_min_grad`
  (~752-970); dispatch stamp `:2819`; send `:2836`; compute-truthful gate `:856`.
- Checker rungs `async_cifar10/scripts/parity/checks.py` — `aggregation_sequence`, `inter_arrival_order`,
  `v2 var_trajectory`, gating `DETERMINISTIC_SELECTORS`, `cohort_sequence_parity`. Standalone diff
  `expt_scripts/logical_parity.py`.

## RESOLVED DECISIONS (operator, 2026-07-05/06)
1. Delay model → remainder-wait (`wall = max(gpu,D)`), aligning to async_cifar10.
2. Delay magnitude → configurable; full registry (`--delay-factor 1`) for the current sync runs.
3. GPU overhead → optimize now (K-D32 landed); forward < backprop held with margin.
4. Checker rungs → hard-FAIL (EXACT + V2 mean-guard, ungated).
5. #15 fix → **compute-truthful gate** (NOT re-dispatch-on-return / NOT weaken hold-to-commit); **flag-gated + P3
   parity gate**; raise `sim_gate_compute_cap_s` if parity moves.
