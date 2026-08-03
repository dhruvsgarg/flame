# HANDOFF — H11 / H13 hypothesis state

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

### PENDING LAUNCH — `felix_round` real+sim, **7200s** (blocked: node in use by the probe)

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

## H13 — what supplies the replicate floor

**Status: SOURCE FOUND. Live dropout inside `calculate_jvp`. The fix is unbuilt.**

### CONFIRMED — dropout is on during the finite difference

`create_model` → `train_adapter("rotten tomato")` leaves **13 of DistilBERT's 20 `nn.Dropout` modules
training at p=0.1**, while the root `model.training` reads **False**. Nothing on the training path calls
`.eval()`. So `calculate_jvp` runs `L(p−hv)` and `L(p+hv)` as two forward passes under **two different
dropout masks**, drawn from the process-global RNG that no per-task seed pins.

Bench, real stack, CPU, no autocast, one input + one perturbation repeated 6x:

| model mode | loss exact | loss spread | jvp values |
|---|---|---|---|
| as production builds it | 17% | **5.1e-3** | −0.91, −0.83, −0.88, −0.94, **−1.50**, **−0.31** |
| `.eval()` | 100% | 0.00e+00 | −0.9245 x6 |

A **5x** swing in the "gradient" for identical inputs, with no GPU and no fp16 involved.

### Why every earlier audit came back clean

The per-client `torch_rng` (seeded `client_idx`) is a **different generator** from the global one dropout
draws on, so the RNG-position fingerprint matches. Data, dispatch order, iteration and model_version match
because the divergence enters *below* all of them. And a trainer reproduces exactly when its global stream
happens to sit at the same offset — advanced by prior forward passes, a count async timing decides. That is
the 8-of-30 split.

### Runs 1-2 were INCONCLUSIVE, not negative

`probe_jvp_determinism.py::_build_real` called `.to(device).eval()` — silencing the mechanism. Every
bit-exact null from those runs is void. The probe is fixed (`evalmode` arm, `live drop` census, verdict read
off the WITHIN-process column, `--hetero`'s cross-process table suppressed). Two results survive:

| model | rel Δloss | rel Δjvp | amplification |
|---|---|---|---|
| proxy (4-layer, 256d) | 1.61e-05 | 1.12e-03 | **70x** |
| **real DistilBERT+adapter** (67.4M / 1.04M trainable) | 2.24e-05 | 4.22e-03 | **189x** |

`jvp = (L(p+hv) − L(p−hv)) / 2h` at `h=0.01` is ill-conditioned by construction (~`|L|/(2h·|ΔL|)`);
production telemetry independently gave a 72x median. It AMPLIFIES noise, it does not create it — and what
it amplifies is dropout, not fp16. `FWDLLM_STRICT_DETERMINISM` stays inert and unrelated.

### It is a correctness bug, not only a reproducibility one

Two masks means the estimator is not a directional derivative of any one function. Separately,
`eval_model()` sets `.eval()` and **never restores train mode**, so a trainer trains with dropout on until
its first eval and off afterwards.

### Open questions, answered

**Is the bug "eval", and should forward-gradient tuning run with dropout or without?**
The bug is not dropout's presence, it is that the TWO passes of one finite difference draw DIFFERENT masks.
`(L_b(p+hv) - L_a(p-hv))/2h` is not a directional derivative of anything. Two fixes are defensible: run the
difference in eval (differentiate the deterministic loss, lose the regularizer) or share one mask across the
+/- passes (`L_m` for both -- keeps the regularizer and is still a correct derivative *of L_m*). Only the
first is built. "Dropout is a training construct" is true but does not settle it alone: the JVP passes are
inference-SHAPED, yet what they compute is a training gradient. The 1h A/B decides.

**What does probe C measure -- does dropout change the perturbations?**
No: it changes their SCORE, not their draw. Candidates come from the per-client CPU `torch_rng`, which
dropout never touches, and probe C holds `v` FIXED and repeats the same call. What moves is `jvp`, and
through it `grad += jvp*v` and `stat_utility`. Selection flips a round LATER: `old_grad` arrives noisy from
the aggregator, and the cos-sim top-1 over 10 candidates is a discrete pick over near-ties (§D-42).
Expect `base` (live drop 13) within-process jvp spread > 0, `evalmode` (live drop 0) exactly 0. It says
nothing about accuracy -- that is the 1h A/B's job.

**Did `stat_utility` use eval? Does anything in the code?**
`_compute_batch_stat_utility` runs `self.model(x)` under `torch.no_grad()` only -- `no_grad` is not `eval`,
so dropout was live there too, and `agg_rate_type: new` weights it via `beta(stat_utility)` (H12a). The ONLY
`.eval()` on the trainer path is `eval_model()`, and **it never fires**: `evaluate_during_training` is
hardcoded False (`trainer/main.py:125`), and `test_on_the_server` evaluates the AGGREGATOR's model. Verified
on disk -- 0 `len(test_dl)` lines in a 100-trainer `felix_round` real log. Its missing `.train()` restore is
therefore latent, not active.

**So can we assume eval-mode training is fine, since baselines already reach 75%?**
No -- that inference would hold only if those runs had been in eval mode, and they were not. Dropout was live
for 100% of every training pass on record, so **75-77% is the DROPOUT-LIVE number and eval-mode accuracy is
unmeasured.** That is exactly what the A/B buys, and it is the reason the flag defaults OFF.

**Runtime cost of dropout, GPU and NPU?**
GPU: small and measurable, not modelled -- mask RNG plus one elementwise multiply per site, memory-bound, no
GEMM. Eval mode should be marginally FASTER; `tb_forward_jvp` / `tb_stat_utility` in the A/B give the number,
so do not guess one. If it moves materially, the four charge profiles need regenerating (§B). NPU: the
mobile delay is MODELLED, not measured, so nothing here reaches the sim clock on its own, and a deployed
inference graph folds dropout out entirely.

**Flag across the trainer files?**
There is only one: `tc_transformer_trainer_distribute.py` is the sole caller of `calculate_jvp`, and
`trainer/main.py` wires the knob. One code path, one yaml key.

### Next -- the A/B

`jvp_eval_mode` is LANDED, config-gated, default False = today's behavior. Set it in the `felix_round`
yamls' `hyperparameters` (real AND sim -- they must match, §F-18) and run the two 1h legs.

```bash
cd lib/python/examples/fwdllm/expt_scripts
python probe_jvp_determinism.py --sweep --model real --replicas 8 --out-dir probe_C   # bench, minutes
# then, per leg: jvp_eval_mode false vs true, same duration
grep JVP_EVAL_MODE ../experiments/<run dir>/*trainers.log | head -1                   # confirm it took
```

Read, in order: (1) does `[JVP_EVAL_MODE] jvp_eval_mode=True` appear -- a knob that never reached the trainer
is the failure mode §F-18 exists for; (2) peak accuracy, ON vs OFF -- the part that is NOT a parity question;
(3) the replicate floor, once there are two ON legs.
**Exit criterion: the 13.3% iters/bin / 11.16-point floor drops.** If it does not, H13 is falsified as the
dominant term and the tolerance-recalibration branch (§F-28) is back. If accuracy drops materially with it
ON, prefer the shared-mask fix over eval mode and re-run.

Node 2's `fluxtune`/`fwdllm` floors are still worth having either way.

---

## Node status at handoff

| node | job | state |
|---|---|---|
| 1 | `felix_round` real+sim **7200s** — H11 live validation | **PENDING — blocked on the probe node** |
| 2 | `fluxtune` + `fwdllm` real replicates (floors) | launched |
| 3 | `fwdllm_it_unaware` + `fwdllm_it_oracular` 7200s pairs | launched |
| bench | H13 `--sweep --model real` on the fixed probe | **PENDING** |
