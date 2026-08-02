# HANDOFF — H11 / H12 hypothesis state

**TEMPORARY. Delete when both close**; findings move to `simulate_fwdllm.md` §A/§E/§G. This file exists so
the validate/invalidate state of two hypotheses survives a session boundary without re-deriving it.

---

## H11 — sim over-dispatches at the round boundary

**Status: root-caused, FIXED, unit-tested — LIVE VALIDATION PENDING (the one run still owed).**

**Defect.** `_release_sim_slots_at_agg_goal`'s legacy path clears `_sim_inflight_expected` AND
`_sim_pending_commit`, then calls `_sim_hold_busy_slots`, which rebuilds `outstanding` from those now-empty
sets and deletes everything not in it from `all_selected`. Both re-pick guards hit zero at one instant, so the
boundary's successive top-up `select()` calls re-pick ends that are still training.

**Evidence (production telemetry, `felix_round` sim `run_20260801_163513`).** End `…0395` picked 5x and
`…0409` 2x at a frozen vclock, `in_pending_commit: false` across every repeat → 35 picks against c=30. Real
issued 30 unique. `felix_it` is a clean NEGATIVE (0 over-dispatch both modes, all 24652 re-picks are true
churn), which localizes the defect to the round-boundary batch path, NOT AsyncOort's `select()`.

**Fix.** Fold `_trainer_inflight_dispatch_version` (dispatched-not-yet-returned, maintained in BOTH modes,
and the only in-flight record the boundary does not clear) into sim's `outstanding` — the same half real's
`_PendingCommitUnion` already carries, which is why real never had the bug.
`flame/mode/horizontal/syncfl/fwdllm_aggregator.py`, 3 tests in
`tests/mode/test_fwdllm_sim_grad_residence.py::TestBoundaryDropKeepsDispatchedGuard` (2 fail without it).

**What the unit tests do NOT cover.** They drive `_sim_hold_busy_slots` with a fake channel, so they pin the
mechanism. They do not confirm that in a live run `_trainer_inflight_dispatch_version` is populated at the
right moment relative to the boundary. That is exactly what the pending run tests.

### PENDING LAUNCH — `felix_round` real+sim, **7200s** (blocked: node in use by the H12 probe)

```bash
cd lib/python/examples/fwdllm/expt_scripts
bash run_sequential.sh --mode both --max-runtime-s 7200 --only felix_round --yes
python trace_boundary_repicks.py ../experiments/<new felix_round sim dir>
python run_parity.py --yes --baselines felix_round
```

**⚠ MUST be 7200s, NOT 3600s** — an earlier plan said 3600s and it would have been a wasted run. The defect
only fires at the round-1→2 boundary, and that boundary arrives late:

| leg | round-2 re-draw first fires at |
|---|---|
| real `run_20260802_104547` | wall **4823s** |
| real `run_20260801_084040` | wall **4270s** |
| sim `run_20260801_163513` | **vclock 4441s** (wall 1390s) |

`--max-runtime-s` caps WALL in real mode and VCLOCK in sim mode, so 3600s stops both legs before the boundary
exists and the rung grades a run in which the defect cannot occur. (§C's "one MECHANISM rung → 1800s" is a
general rule that does not hold for a BOUNDARY-gated defect — duration must reach the boundary.)

**Exit criteria.**
- `trace_boundary_repicks.py` on the new sim dir reads **OVER-DISPATCH=0**.
- `selection_detail` goes green on `felix_round`.
- Boundary picks read **30 unique on both sides** (real already does).
- No regression in `concurrency_cap` / `r1_inflight_overlap` / `slot_utilization`.

If over-dispatch persists, the guard is being cleared somewhere else as well — re-run
`trace_boundary_repicks.py` and check `in_all_selected` / `in_pending_commit` in the `selection` event's
`per_trainer` block at the repeats, which is how the original was localized.

---

## H12 — what supplies the replicate floor

**Status: SPLIT. Amplifier CONFIRMED. Source FALSIFIED. No standing candidate — this is the open question.**

### CONFIRMED — the JVP is ill-conditioned by construction

`jvp = (L(p+hv) − L(p−hv)) / 2h` at `h=0.01`, so its condition number is ~`|L|/(2h·|ΔL|)`. Measured on the
bench, fp16 vs fp32 on the SAME input (needs no nondeterminism at all):

| model | rel Δloss | rel Δjvp | amplification |
|---|---|---|---|
| proxy (4-layer, 256d) | 1.61e-05 | 1.12e-03 | **70x** |
| **real DistilBERT+adapter** (67.4M / 1.04M trainable) | 2.24e-05 | 4.22e-03 | **189x** |

Production telemetry independently gave a **72x** median. This is the standing, run-independent case for
`FWDLLM_JVP_FP32` — but it AMPLIFIES noise, it does not CREATE it.

### FALSIFIED — fp16 / GPU kernel nondeterminism is NOT the source

Real DistilBERT stack, 8 co-located processes, one A40, torch 2.12: **every loss and jvp bit-identical**,
within and across processes. `FWDLLM_STRICT_DETERMINISM` came back inert (nothing to pin) — not evidence it
is broken.

### FALSIFIED on disk — every other input hypothesis

| hypothesis | how it died |
|---|---|
| trainer data differs | all **100** `CLIENT n DATA HASH` lines identical across runs |
| first task saw mid-bin updated weights | all 30 first tasks are `iteration 0`, `model_version 0` in BOTH runs |
| arrival/dispatch order | dispatch ranks identical (2 adjacent swaps in 30); the 8 reproducing trainers scatter across ranks 1-29 |
| seeding / `client_idx` / partition / cohort | `client_idx=(trainer_id−1)%modulo` is registry-fixed; `perturbations_total`, `forward_passes_total`, `(data_id, iteration, model_version)` all match; cohorts bit-identical (Jaccard 1.000) |
| the model weights | all 30 trainers receive the SAME weights at iteration 0 — if weights were the differing input, all 30 would differ. **8 do not.** |

### The contradiction, stated plainly

Data, dispatch order, iteration, model_version, RNG stream position, and arithmetic-under-identical-inputs
are **all verified identical** — and **22 of 30 trainers still differ by ~1.6e-3** on their first task,
which the 189x amplifier then turns into O(10%) gradient differences and a 13.3% iters/bin floor.

**There is no standing candidate.** Do not adopt one without evidence.

### PENDING — the last arithmetic hypothesis

The probe ran 8 processes doing **identical work in lockstep**. Production runs ~12 per GPU doing
**different** work at different times — different allocation patterns, different cuBLAS workspace
availability. `--hetero` gives each replica its own seed; `--compare` diffs two launches replica-by-replica,
which is the same-trainer-across-two-runs comparison production actually makes (comparing neighbours *within*
one launch is meaningless once seeds differ).

```bash
cd lib/python/examples/fwdllm/expt_scripts
python probe_jvp_determinism.py --sweep --model real --hetero --replicas 8 --out-dir probe_A
python probe_jvp_determinism.py --sweep --model real --hetero --replicas 8 --out-dir probe_B
python probe_jvp_determinism.py --compare probe_A probe_B
```

**If it reproduces** → the arm whose spread collapses is the fix; promote the flag, re-measure the floor.
**If it does not** → the tool says so explicitly (*"stop probing and hunt the differing INPUT"*), and the
next step is instrumentation, not another probe: a weights/logits hash emitted at INFO on the trainer's first
task, then ONE short run. Every hypothesis testable on existing telemetry is exhausted.

### Consequence for the roadmap either way

The floor is real and measured (`felix_round` 13.3% iters/bin, 11.16 accuracy points at peak;
`fedbuff_round` 3.9%). With its cause unknown, **the IRREDUCIBLE branch is the more likely one** — tolerance
recalibration against per-baseline measured floors, not a code fix. That path needs node 2's
`fluxtune`/`fwdllm` floors regardless of how H12 resolves, which is why those runs are not blocked on this.

---

## Node status at handoff

| node | job | state |
|---|---|---|
| 1 | `felix_round` real+sim **7200s** — H11 live validation | **PENDING — blocked on the probe node** |
| 2 | `fluxtune` + `fwdllm` real replicates (floors) | launched |
| 3 | `fwdllm_it_unaware` + `fwdllm_it_oracular` 7200s pairs | launched |
| bench | H12 `--hetero` / `--compare` probe | running |
