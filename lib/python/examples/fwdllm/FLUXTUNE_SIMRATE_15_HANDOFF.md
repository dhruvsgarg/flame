# #15 fluxtune `sim_rate` — ROOT CAUSE FOUND, resume-here handoff

**Status (2026-07-05):** Root cause of fluxtune `sim_rate = 0.50` is IDENTIFIED and evidence-backed. The fix
(re-dispatch-on-return) is designed and de-risked but NOT yet implemented. Pick up at "IMPLEMENTATION PLAN" below.
This is a TEMP tracker for the #15 work — fold into `simulate_fwdllm.md` §G/§K and delete when the fix lands.

> Prereqs / cross-refs: `simulate_fwdllm.md` (§A open issue #15, §H corrections, principles #13/#15/#16),
> `PARITY_LOGICAL_TASKS.md` (checkpoint). The banked evidence run is
> `experiments/run_20260705_204619_fluxtune_n10_smoke_syn_0_sim` (+ its `_real` pair `run_20260705_202448`).

---

## 1. THE ROOT CAUSE (definitive, evidence-backed)

**A circular wait between the sim drain's `earlier_stuck` gate and the hold-to-commit slot residence.**

The drain (`_sim_recv_min_grad`, `flame/mode/horizontal/syncfl/fwdllm_aggregator.py:752`) blocks in REAL WALL to
preserve sct-ordered commits: it will not commit a buffered grad while an *in-flight* trainer has a smaller
**expected** sct (`_sim_inflight_expected[end]`). But that in-flight trainer is itself **blocked in `recv_wrapper`**
(hold-to-commit: `_release_end_on_return:1136` returns early on the residence path, so the trainer's slot only frees
when its PREVIOUS grad commits) — so it is NOT computing and cannot produce the grad the drain is waiting for. Its
previous grad won't commit because the drain is stalled. Circular; broken only by the 30s `RECV_TIMEOUT_WAIT_S`
failsafe evict.

This is the STEADY STATE, not an edge case. The startup is fine (all 10 dispatched fresh → compute in parallel,
concurrency=10 for ~5s) and it collapses right after the first commit, when trainers start getting held.

### Causal chain (fully closed)
```
hold-to-commit blocks trainers in recv (recv_wrapper mean 30.9s, max 60s)
  → drain earlier_stuck gate waits real-wall for these NON-COMPUTING in-flight trainers
  → circular stall, broken only by the 30s failsafe evict
  → ~1974s (81% of the 2425s sim wall) burned in gate-waits
  → GPU concurrency 1.54x (vs real 3.37x)
  → sim_rate = vclock/wall = 1191/2425 = 0.50
```

## 2. THE EVIDENCE (all from the banked sim/real pair; reproduce with the scripts in §5)

- **Per-commit:** real reaches agg-goal in **4.30s wall**; sim models **3.27s vclock** but spends **6.60s wall** →
  per-commit sim_rate 0.50.
- **GPU concurrency:** real avg **3.37x** (98% busy), sim avg **1.54x** (85% busy). Same total GPU work
  (~3.7-4.0k trainer-s), same ~480s 8-way pipeline floor. `max=10` at startup (proves the HW sustains 10-wide).
- **Trainer recv-block (trainer 371):** sim `recv_wrapper` mean **30.9s** / max 55s per iteration; real **0.28s**
  (median **0.01s**). Compute is mode-invariant (~3-4s, delay correctly skipped `_emulate_training_delay=0.000s`).
- **The smoking-gun stall:** `[SIM_GRAD_STUCK_EVICT] round=1 end=0379 exp=24.0 bmin=27.0` — drain blocked ~30s
  waiting for trainer 379 (expected sct 24) while **6 grads sat ready** (bmin 27). Trainer 379's own telemetry:
  blocked in `recv_wrapper` for **60.39s** (20:47:19→20:48:19), got weights the instant AFTER the evict.
- **Steady-state:** `buf_depth` is **constantly 6** (six grads always ready, held behind the deadlocked one);
  **524/1093 commits (48%) stall >2s** = **1974s** total wall (~81%).
- **Out-of-order already happens:** committed scts go 11,13,7,5.4,5.5,10,13,16… (non-monotonic) — so the strict
  ordering the gate blocks for is not even being preserved; it pays full real-wall cost for nothing.
- **Eval:** once per data-bin (23 evals / 24 data_ids), NOT per iteration. But it's an 8.5s BLOCKING call on the
  aggregator (~195s / 8% of wall). Secondary; overlap later.

## 3. THE PROPER FIX

The gate is not wrong to want sct order — the bug is it **waits for trainers that aren't running**. Fix = ensure
the trainer the gate waits for is ALWAYS actually computing → **re-dispatch on RETURN** (match real, whose recv is
0.01s), decoupling physical GPU pipelining from the virtual in-flight ledger.

- A returned trainer immediately gets the next weights and computes → never idle-blocked in recv.
- The in-flight trainer the gate waits for then produces its grad in ~4s IN PARALLEL with 7 others → the drain
  commits a BATCH per GPU-pass instead of stalling on one held trainer. `buf_depth` stays deep, gate-waits become
  short and legitimate, 30s failsafe never fires. Expected: concurrency 1.54x → ~3.37x+, sim_rate 0.50 → ~1.3-2+.
- **Keep the gate + sct-order commits** (they preserve #1d cohort parity) — only remove the recv-block that starves
  the pipeline. The virtual in-flight COUNT / selection eligibility (R1) stays held-to-commit; that is a SEPARATE
  ledger from the trainer's physical compute. Conflating them is what caused this (K-D17b held both together).

### The hard constraint to preserve (principle #16 — no hacks)
The re-dispatched (compute-ahead) grad must be computed against the **model version real dispatched** — fluxtune's
fedbuff down-weights by staleness `V'-V`, so a grad against the wrong version silently changes the result. Since
REAL also re-dispatches on return and cycles continuously, the versions come from the same deterministic commit
order (K-D29 remainder-wait). **This should hold but MUST be verified from telemetry, not assumed** — see the P0
gate below.

### Belt-and-suspenders (independently correct, do alongside)
`_sim_inflight_expected[end]` is stamped at DISPATCH (`fwdllm_aggregator.py:2819`, `_sst + _budget` where
`_sst = self._vclock.now`, `_budget = self._sim_trainer_budget.get(end, self._sim_budget_min)`) assuming the
trainer starts computing immediately. A HELD trainer hasn't started, so the expected sct is a fiction (too early) →
the gate waits for a grad that won't come. Tie the expected clock to actual compute-START, and/or bound the gate's
real-wall wait far below the 30s failsafe.

## 4. IMPLEMENTATION PLAN (resume here tomorrow)

**P0 — GATE (verify before coding, read-only):** Pull per-trainer `MODEL_VERSION` (the version each grad was
computed against) vs the commit sequence, real vs sim. Confirm real absorbs the staleness (grad version = dispatch
version, fedbuff down-weights) and that re-dispatch-on-return in sim yields the SAME dispatch-version sequence as
real. Anchor: `MessageType.MODEL_VERSION` on grad msgs (`fwdllm_aggregator.py:_process_single_trainer_message`,
staleness log ~line 1170); `inc_model_version_per_data_id=True`. If versions DON'T align, the design needs the
version pinned to the trainer's virtual completion (sct) rather than physical dispatch — revisit before coding.

**P1 — CORE CHANGE (once P0 passes):** Decouple physical re-dispatch from the virtual commit-hold.
- `_release_end_on_return` (`:1128`): on the async sim residence path, currently `return`s (holds slot+guard to
  commit). Change so the trainer's PHYSICAL re-dispatch (weights send → compute next grad) happens on return,
  while the SELECTOR's virtual in-flight set (`selected_ends`/`all_selected`, driving R1 + `extra = c - inflight`)
  stays held to commit via `_sim_hold_busy_slots`. Two ledgers: physical-dispatch (release on return) vs
  virtual-inflight (release on commit).
- The re-dispatched grad enters the sct reorder buffer with its own `SIM_COMPLETION_TS`; the drain keeps committing
  in sct order for the vclock. Verify the buffer/`_sim_inflight_expected`/`_sim_pending_commit` bookkeeping stays
  consistent (K-D27 two-ledger discipline — do NOT reintroduce the R1 regression).
- Fix `_sim_inflight_expected` stamp to reflect actual compute-start (belt-and-suspenders above).

**P2 — TESTS + TELEMETRY (same change, principle #11):** pytest `tests/mode -k fwdllm` (residence/R1/drain) +
`tests/mode -k parity`; assert async_cifar10 byte-identical (shared `_sim_recv_min` untouched — this is a
fwdllm_aggregator-only edit, principle #8/#9). Add a telemetry assert that `recv_wrapper` mean drops to ~0 and
concurrency rises. R1 must stay ~0 (bank it — K-D19 lesson: an R1 fix isn't done until the smoke shows R1<=2%).

**P3 — VALIDATE:** re-run `run_sequential.sh --only fluxtune --mode both --delays on --delay-factor 1
--max-data-id 2` (short). Confirm: `recv_wrapper`→~0, concurrency→~3.37x+, `sim_rate`→>1, `[SIM_GRAD_STUCK_EVICT]`
gone, and — critically — `cohort_sequence`/`var` parity UNCHANGED or improved (the fix must not alter which grads
commit in which cohort; if it does, the version-parity assumption (P0) was wrong).

**P4 — DOCS:** fold the root cause into `simulate_fwdllm.md` §A (#15) + §H (the "gate waits on held trainers" root)
+ new §K deviation (K-D34, the two-ledger physical/virtual dispatch split); update `PARITY_LOGICAL_TASKS.md`
checkpoint; delete this handoff file.

## 5. REPRO SCRIPTS (paste-ready, read-only, run from `lib/python/examples/fwdllm`)

```bash
FS=$(ls -td experiments/run_*_fluxtune_n10_smoke_syn_0_sim | head -1)   # banked evidence run
AGG=$(ls "$FS"/*aggregator.log | head -1)

# (a) the stall: 48% of commits stall >2s = ~1974s wall
grep "SIM_GRAD_RECV" "$AGG" | python3 -c "
import sys,re; from datetime import datetime
prev=None;big=0;tot=0;wait=0
for l in sys.stdin:
    ts=datetime.strptime(l.split(' | ')[0],'%Y-%m-%d %H:%M:%S,%f').timestamp()
    if prev is not None:
        d=ts-prev; tot+=1
        if d>2: big+=1; wait+=d
    prev=ts
print(f'commits={tot+1} gaps>2s={big} ({100*big/tot:.0f}%) wall_in_waits={wait:.0f}s')"

# (b) the evict + buf_depth stuck at 6
grep -E "SIM_GRAD_STUCK_EVICT|SIM_GRAD_RECV" "$AGG" | grep -oE "buf_depth=[0-9]+" | sort | uniq -c
grep "SIM_GRAD_STUCK_EVICT" "$AGG"

# (c) trainer recv-block sim vs real (trainer 371)
for m in sim real; do D=$(ls -td experiments/run_*_fluxtune_n10_smoke_syn_0_$m|head -1)
  tf=$(ls "$D"/telemetry/trainer_*371.jsonl|head -1)
  python3 - "$tf" "$m" <<'PY'
import json,sys,statistics as st
f,m=sys.argv[1],sys.argv[2]; recv=[];tr=[]
for l in open(f):
    e=json.loads(l)
    if e.get('event')=='step_timing':
        if e.get('func')=='recv_wrapper' and e.get('duration_s') is not None: recv.append(e['duration_s'])
        if e.get('func')=='_train_one_batch' and e.get('duration_s') is not None: tr.append(e['duration_s'])
print(f"{m}: recv_wrapper mean={st.mean(recv):.1f}s median={st.median(recv):.2f}s | train mean={st.mean(tr):.2f}s")
PY
done
```

Aggregator timeline decoder (per-step wall) + GPU-concurrency scripts were run ad hoc during the investigation;
the key numbers are all captured in §2 above. The `[SIM_GRAD_RECV]` log line
(`fwdllm_aggregator.py:926`, format `end= sct= T_v= buf_depth= inflight_exp= sel_ends=`) is the primary drain
diagnostic — grep it to watch the gate.

## 6. LANDED THIS SESSION (already committed separately / in this commit)
- **K-D33 pinning (in this branch):** trainer `[PIN]` self-report (`examples/fwdllm/trainer/main.py` — NOT the dead
  `fl_main.py`); `[LOAD_BALANCE]` post-proc check (`flame/launch/spawner.py::_assert_load_balanced`); aggregator GPU
  pin (`flame/launch/aggregator_spawner.py` `gpu_id` param + `flame/launch/runner.py` picks idle/least-loaded GPU).
  120 launch tests green. Corrected earlier "under-provisioned GPUs" misread — pinning IS clean (8 GPUs, balanced
  round-robin, CPU sched_setaffinity 1 core/trainer). `client_idx%8` device arg is vestigial (`FedSgdTrainer:388`
  overwrites `self.device=torch.device("cuda")`).
- **Docs updated:** `simulate_fwdllm.md` §A/§G/§H/§K, `PARITY_LOGICAL_TASKS.md` checkpoint,
  `fluxtune_contributions.md` §5 — current status (sync sim_rate 2.9-3.0, K-D31/P2-7a validated, bin-7 float-
  nondeterminism wall, fluxtune #15 = pipelining).

## 7. RELATED OPEN ITEMS (context, not part of #15)
- **bin-7 float-nondeterminism wall (sync):** exact cadence parity unattainable past ~bin 6 (GPU fp16 grad jitter
  amplified by split-half variance); relax `cohort_sequence` EXACT to `--max-bin 1` + distributional rung beyond.
  Confirm with a 2-real-run P0-2 diff. (simulate_fwdllm.md open issue.)
- **#1d fluxtune cohort SET (thin-margin overrun):** `jvp_perf_opt` put GPU MEAN (3.61s) under the 4.0s budget but
  the TAIL (4-5.4s) overruns on the 2 doubled GPUs. The #15 fix (keep compute near the ~2.4s uncontended floor via
  pipelining) should also lift this.
