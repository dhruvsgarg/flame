# Real / Sim Parity — Methodical Causal Ladder

Living document for the async_cifar10 parity checker.
Kept in sync with `scripts/parity/checks.py` (check functions),
`scripts/parity/report.py` (stage grouping + verdict), and the pytest suite.

**Real/sim comparator — give the two run dirs, get a report JSON:**
```bash
cd lib/python/examples/async_cifar10
# single baseline: point at the real + sim run dirs
PYTHONIOENCODING=utf-8 python scripts/parity_check.py \
  --real experiments/<real_run_dir> \
  --sim  experiments/<sim_run_dir> \
  --agg-goal 10 --budget-s <runtime_s> --json-out experiments/parity_<baseline>_<tag>.json

# all baselines at once: auto-discovers the latest real/sim pair per tag
PYTHONIOENCODING=utf-8 python scripts/parity_check.py --batch \
  --experiments-dir experiments --baselines felix oort refl feddance \
  --agg-goal 10 --budget-s <runtime_s> --json-out experiments/parity.json
```
`--budget-s` = the run's `--runtime-s` (e.g. 12600 for 3.5h). Add `--lenient` to
demote DIST fails to warnings; prints a stage-grouped report + root-cause banner.
Per-run plots: `python ../../../scripts/analysis/analyze_run.py <run>/telemetry`.

**Readiness/regression tests** (no cluster; run under lib/python):
`pytest tests/mode/test_baseline_readiness.py tests/mode/test_sim_barrier.py
tests/mode/test_async_sim_ordering.py tests/mode/test_sync_sim_ordering.py
tests/mode/test_sim_commit_overhead.py` — guards baseline wiring, the in-memory
cache, serialize-once, sim-recv barrier ordering, and the overhead model.

### Current status (Jun 9 — evening run 201709; oort/feddance from 094917)

**Speedup: done.** Removed across all baselines: recv-poll, distribute stagger,
per-send model re-serialize, disk-backed update cache (→ in-memory MemCache).
Eval/checkpoint off the critical path. Floor = per-commit 2 MB deserialize + optimizer.

**Cross-baseline scorecard** (`parity_batch.py`, 3.5h / n=300 / syn_0):

| baseline | run | sim_rate | rounds real→sim | advance real/sim | staleness real/sim | acc diff | overhead | blocking fail |
|---|---|---|---|---|---|---|---|---|
| feddance | 094917 | 18.4x | 365→370 (+1.4%) | 33.85/34.11s | 183.0/185.5 (ks=0.01) | 0.029 ✓ | 0.0 | advance KS-shape (0.57), trainer_speed absent in real |
| oort     | 094917 | 12.0x | 666→701 (+5.3%) | 18.49/17.98s | 0/0 | 0.033 ✓ | 0.0 | terminal trainers 255/286 (10.8%); total_commits 2.6% (tol 2%) |
| refl     | 201709 | **1.18x** | 8161→8435 (+3.4%) | 1.51/1.49s | 3.07/3.02 ✓ | 0.021 ✓ | 0.10 | eligibility KS=0.48, loss_diff=0.16, speedup low (§4/#2) |
| felix    | 201709 | **2.28x** | 2719→4230 (+55%) | 4.52/2.98s | 2.79→**12.97** | 0.073 ✗ | 0.16→**0.315** | advance 34% low → rounds+staleness cascades (§3b) |

**Per-baseline status:**

- **feddance**: essentially passing. Per-round advance KS high (0.57) despite
  mean match (33.85≈34.11); distribution shape differs — low priority. Real
  telemetry missing `trainer_speed_s` field (real_mean=0.0) — telemetry gap, not
  a sim bug. No action before overnight run.
- **oort**: near-pass. terminal_state trainers at matched V: 255 sim vs 286 real
  (10.8%, tol 5%); total_commits 2.6% over tol. Rounds themselves 5.3% off (tol
  10%). Likely noise at this run length (~700 rounds). Recheck after overnight.
- **refl**: rounds/staleness/advance pass. Remaining DIST fails: eligibility KS=0.48
  (eligible set shifts — downstream of availability trace time-base mismatch A3),
  loss_diff=0.16 (marginal vs tol 0.15), participation/aggregation_sequence/utility
  diverge (downstream of eligibility). **Speedup 1.18x is the main gap** — §4/#2
  (lazy deserialize) would bring it to ~2.2x with no parity cost.
- **felix**: all major metrics blocked on overhead under-charge (0.16 → **0.315**
  applied tonight). See §3b. Expect overnight run to pass rounds + staleness + acc.

---

## §1  Philosophy: the parity ladder

An FL run is a **pipeline**. Each round flows through the same stages in both
real and sim mode:

```
clock/time-base → availability → selection → dispatch+training
   → update-return+ordering → aggregation → utility → emergent outcomes
```

Parity must hold at *every* stage. If it breaks at stage N, every stage above N
also diverges — but those upper failures are **consequences, not bugs**. The job
of the checker is to find the **lowest broken rung**: the earliest stage whose
own inputs are sound but whose output diverges. That stage holds the root cause.

This replaces the old "severity-ordered symptom list." Severity tells you what
hurts; the ladder tells you *why*, and does so automatically.

### Three roles every check plays

Tag each check with the role it serves in localization:

- **CONTROL** — confirms an *input* to a stage is identical across modes
  (e.g. trainer_speed_s, training_budget_s, telemetry coverage). A failing
  control means the sim's inputs differ; fix the input model, not the stage.
- **MECHANISM** — confirms a *single transformation* inside a stage is modeled
  (e.g. per-commit overhead, inter-round overlap, availability time-base). A
  failing mechanism with passing controls is a *localized* bug — the prize.
- **EMERGENT** — an aggregate outcome (throughput, terminal state, convergence,
  utility). These are what we ultimately care about, but they never localize on
  their own; they only tell you *something* below them broke.

Debugging rule: an EMERGENT failure is a prompt to walk *down* the ladder to the
mechanism/control checks beneath it. Never fix an emergent symptom directly.

### Two-axis classification

Every check has two orthogonal labels:

- **STAGE** (0–9 below): where in the causal pipeline it sits. Determines
  ordering and dependency.
- **TIER** (enforcement strictness, unchanged from today):
  - `INV`  — sim-mode invariant; FAIL is always a hard FAIL.
  - `EXACT`— must match within tight tolerance; hard FAIL.
  - `DIST` — distributional match; FAIL unless `--lenient`.
  - `DIAG` — diagnostic only; never FAILs (informational), but feeds root-cause.

STAGE drives diagnosis; TIER drives the pass/fail verdict. They are independent.

### Dependency gating (the part that makes checks build on each other)

Each check declares its **upstream prerequisites** — the checks whose passing is
required for this check to be *meaningful*. The verdict engine then:

1. Walks rungs bottom-up.
2. Finds the lowest stage with an enforced FAIL whose upstreams all PASS →
   labels it **ROOT-CAUSE**.
3. Tags every higher enforced FAIL whose upstream chain contains a failed check
   as **DOWNSTREAM (of <root>)**, demoted from the headline failure list.

Result: one run prints "ROOT-CAUSE: stage-1 overhead residual (K3b); 7 downstream
failures suppressed" instead of nine equally-loud FAILs you have to triage by hand.

`deps` must name the *strongest causal link*, not a generic base. In particular
TC1 (coverage) is **not** a universal ancestor — it gates only K10, because a
missing field makes a downstream check SKIP (handled locally), not FAIL. Wiring
every check to depend on TC1 would wrongly demote independent failures (e.g. a
real trainer_speed gap) to "downstream" whenever any *unrelated* field is absent.

### Growth rule

Every time a parity bug is root-caused, leave behind the **most fine-grained
check that would have localized it to the responsible mechanism**, placed at its
causal stage with its upstream dependencies declared. Checks are append-only:
never delete one to "clean up." A check that is currently redundant becomes a
regression guard the next time the simulator changes.

When a single coarse check can be split into independent mechanisms, **split it**
— one assertion per mechanism. A blob KS over six timing phases tells you "timing
is off"; six per-phase KS checks tell you "the MQTT-fetch phase is off, the rest
match." Always prefer the latter.

---

## §2  The ladder

Stages run foundational → emergent. Within a stage, controls/mechanisms precede
the emergent rollup. `[NEW]` = to implement; everything else exists in checks.py.
"Isolates" = the one thing this check tells you when it fails *and its upstreams
pass*. "Dep" = upstream prerequisites.

### Stage 0 — Telemetry coverage  *(gate for everything)*
| ID | Check | Role/Tier | Isolates | Dep |
|---|---|---|---|---|
| TC1 `[NEW]` | Field coverage matrix | CONTROL/INV | A field a downstream check reads is missing/sparse in one mode — explains every downstream SKIP at once | — |
| K10 | vclock_now present (sim) | CONTROL/INV | Sim path never stamps vclock (sync aggregator today) | TC1 |

> TC1 generalizes K10: for *each* field consumed downstream (vclock_now,
> staleness, trainer_speed_s, avail_composition, num_eligible, the phase fields,
> stat_utility, sim_send_ts), report presence count + density per mode. One table
> turns "9 mysterious SKIPs" into "these 3 fields are absent in sim."

### Stage 1 — Clock / time-base  *(the foundation; most parity bugs live here)*
| ID | Check | Role/Tier | Isolates | Dep |
|---|---|---|---|---|
| K1 | vclock monotone (sim) | MECHANISM/INV | vclock goes backwards | K10 |
| K7 | sim_rate in [0.01,100] | MECHANISM/INV | vclock/wall absurd | K10 |
| P3 | trainer_speed_s distribution | CONTROL/DIST | The *input* to the clock model differs (speed model itself wrong) | — |
| K3a `[NEW]` | Modeled-compute advance | MECHANISM/EXACT | sim Δvclock vs the speed order-statistic the round-close formula *should* produce (K-th fastest in-flight for async; max-of-K for sync) — tests the advance **formula** with overhead excluded | P3,K1 |
| K3b `[NEW]` | Overhead residual | MECHANISM/EXACT | `real_advance − sim_advance` per round ≈ 0 — the missing per-commit MQTT/dispatch overhead (CRITICAL-1). Pass once `sim_commit_overhead_s` is modeled | K3a |
| K4 | Overlap factor | MECHANISM/DIAG | Sim doesn't model inter-round async pipelining | P3,K1 |
| K3 | Per-round advance distribution | EMERGENT/EXACT | Sum of K3a+K3b+K4 diverges (rollup) | K3a,K3b,K4 |
| K2 | Rounds-per-virtual-second | EMERGENT/EXACT | Throughput diverges (rollup) | K3 |

> The decomposition is the whole point. Today K3 (per_round_advance) lumps
> formula + overhead + overlap into one FAIL. Split it: if **P3 passes, K3a
> passes, K3b fails, K4 passes** → the bug is *pure missing overhead*, nothing
> else. That single sentence is what CRITICAL-1 took a paragraph of prose to say.
> K3b is also cross-validated at Stage 4 (mqtt_fetch phase): real overhead seen
> at the trainer level should equal K3b residual × agg_goal.

### Stage 2 — Availability  *(indexed by the clock — so gated on Stage 1)*
| ID | Check | Role/Tier | Isolates | Dep |
|---|---|---|---|---|
| A1 | avail_composition parity | MECHANISM/DIST | Per-state available counts diverge | — |
| A2 | num_eligible / num_candidates | MECHANISM/DIST | Eligible-set size diverges | A1 |
| A3 `[NEW]` | Trace time-base consistency | CONTROL/DIST | Availability trace indexed by *different* clocks (sim=vclock, real=wall) — the REFL HIGH-1 bug. Compare each trainer's first/last-available time mapped through its mode's clock | K3 |
| A4 `[NEW]` | Per-trainer duty-cycle | MECHANISM/DIST | A trainer's on/off fraction differs even when set sizes match; needs avail_change events | A3 |

> A2's failure on REFL is *downstream* of the clock (sim runs at vclock_rate
> 0.274 → hits different trace windows). A3 makes that explicit: it fails only
> when the time-base mapping itself is wrong, so A2-fail + A3-pass = "fix the
> clock first," A2-fail + A3-fail = "fix the trace lookup."

### Stage 3 — Selection  *(given the eligible set)*
| ID | Check | Role/Tier | Isolates | Dep |
|---|---|---|---|---|
| S3/4 | num_chosen / in_flight / effective_c | MECHANISM/DIST | Selector picks a different count | A2 |
| S2 | Participation frequency | EMERGENT/DIST | Per-trainer chosen-count diverges | S3/4 |
| S1 | Per-round Jaccard | DIAG | Exact set identity (gated WARN for stochastic selectors) | A2 |

### Stage 4 — Dispatch & training  *(per-trainer timing; the overhead source)*
| ID | Check | Role/Tier | Isolates | Dep |
|---|---|---|---|---|
| T2 `[NEW]` | training_budget_s distribution | CONTROL/DIST | The *input* to the speed model differs | — |
| T_pre `[NEW]` | pre_train_s phase | MECHANISM/DIST | one phase | — |
| T_w2g `[NEW]` | weights_to_gpu_s phase | MECHANISM/DIST | one phase | — |
| T_gpu `[NEW]` | gpu_compute_s phase | MECHANISM/DIST | one phase | T2 |
| T_mqtt `[NEW]` | mqtt_fetch_s phase | MECHANISM/DIST | per-commit MQTT overhead at trainer level (cross-checks K3b) | — |
| T_w2r `[NEW]` | weights_to_ram_s phase | MECHANISM/DIST | one phase | — |
| T_post `[NEW]` | post_train_s phase | MECHANISM/DIST | one phase | — |
| T3 | GPU budget respected | MECHANISM/INV | real GPU time overruns modeled budget | T2 |
| K6 | sim_send_ts correctness | CONTROL/INV | sim dispatch timestamps not stamped/advancing | K10 |

> Today `trainer_phase` is one DIAG blob. Split into one DIST sub-check per phase
> so the report says exactly which phase diverges. Keep the blob's combined table
> in the report for at-a-glance reading, but each phase asserts independently.

### Stage 5 — Update return & ordering
| ID | Check | Role/Tier | Isolates | Dep |
|---|---|---|---|---|
| U5 | Inter-arrival order (Spearman) | MECHANISM/DIST (gated WARN) | Arrival rank within a round diverges | K3,S3/4 |
| U4 | agg_goal_count cycles 1..K | MECHANISM/INV | Lost/double-counted update per round | — |

### Stage 6 — Aggregation
| ID | Check | Role/Tier | Isolates | Dep |
|---|---|---|---|---|
| U3 | Staleness distribution | MECHANISM/DIST | Staleness diverges (async: directly downstream of clock under-charge) | K3,U5 |
| P1 | Aggregation sequence | EMERGENT/DIST (gated for stochastic) | Per-round contributing set diverges | S2,U5 |

### Stage 7 — Statistical utility
| ID | Check | Role/Tier | Isolates | Dep |
|---|---|---|---|---|
| F1-3 | Per-trainer utility distributions | EMERGENT/DIST | Utility diverges (downstream of selection+training+staleness) | S2,T_gpu,U3 |

### Stage 8 — Emergent outcomes  *(the headline numbers; depend on ~everything)*
| ID | Check | Role/Tier | Isolates | Dep |
|---|---|---|---|---|
| K8 | Terminal-state parity at matched V | EMERGENT/EXACT | rounds/trainers at matched virtual budget diverge | K2,S2 |
| U2 | Total commits at matched V | EMERGENT/EXACT | commit count at V diverges | K2,U4 |
| C1 | Accuracy curve by FL round | EMERGENT/DIST | accuracy diverges | F1-3,K8 |
| C2 `[NEW]` | Loss curve by FL round | EMERGENT/DIST | loss diverges (tracked separately from acc) | F1-3,K8 |

### Stage 9 — Budget / stop sanity  *(meta; orthogonal to causal chain)*
| ID | Check | Role/Tier | Isolates | Dep |
|---|---|---|---|---|
| K9 | Stopped by budget, not rounds cap | INV (WARN) | Comparison truncated by `rounds` cap | — |
| K5 | Failsafe ceiling | INV | Sim wall overshoot > 20% of budget | — |

---

---

## §3  Root cause: overhead-driven buffer backup (felix staleness + rounds)

### §3a — Over-charge (Jun9 morning, overhead=0.50)

Jun9 morning trace: felix overnight sim found rounds gap AND staleness gap share
one cause. `_advance_sim_clock(sct) = max(now, sct) + sim_commit_overhead_s`. In
async the `sct` frontier advances only ~0.31 s/commit (overlap-compressed:
vclock/round 3.1 / agg_goal 10). felix overhead **0.50 > 0.31**, so every commit
adds more vclock than the frontier moves → the reorder buffer backs up: updates
commit long AFTER their completion ts. Evidence: `T_v - sct` median **118 s**
(mean 548), **100%** of commits late; overhead alone (0.50 × 25199 commits) =
12600 s = the entire vclock span. Result: staleness sim **9.3** vs real 2.8, and
the vclock over-charges → sim does 1557 rounds vs real 2595 → lower final
accuracy (39% vs 51%). refl (0.24 ~= 0.31) borderline; oort/feddance (0) unaffected.

**Fixes applied (Jun9 morning):**
- **Overhead re-tuned** — felix 0.50 → 0.16, refl 0.24 → 0.10 (`_metadata/baselines.yaml`).
- **Staleness telemetry** — async `agg_round` now carries `commit_gap_s`
  (`vclock - sct`; >0 = buffer backed up), `buf_depth`, `residence_rounds`, `inflight`.
- **Eval off the critical path** — `evaluate()` snapshots weights and runs the
  test-set forward pass in a daemon thread, so async isn't penalised by per-eval pauses.

### §3b — Under-charge (Jun9 evening, overhead=0.16 → 0.315)

Jun9 evening 201709 run showed overhead=0.16 **under-charges**: advance fell to
2.98s vs real 4.52s (overhead_residual: residual=1.55s, rel=0.342,
implied_per_commit=0.155s). The `overhead_residual` check directly gives the fix:
`0.16 + 1.55 / K=10 = 0.315`.

**Root cause cascade from under-charge:**

1. **advance 2.98s << real 4.52s** → too many rounds (4230 sim vs 2719 real, +55%).
2. **Staleness inflates in two ways:**
   - Baseline: `training_budget / advance = 12.16 / 2.98 = 4.1` rounds vs real
     `12.0 / 4.52 = 2.65`; even without buffer backup, sim staleness is 55% higher.
   - Partial buffer backup: with heterogeneous budgets (5–56s), slow trainers' scts
     fall behind the vclock by many rounds, further inflating mean staleness.
   - Combined: sim staleness 12.97 vs real 2.79.
3. **`queue_wait_s` P50 = 13.1s in sim (vs real 0.7s) is a physical-time artifact,
   not a cause.** `queue_wait_s = _t_msg_start (datetime.now()) − recv_wts_ts (MQTT
   arrival)`. In sim, ALL updates drain into `_sim_buffer` in one barrier call (~0.3s
   physical after dispatch), but are popped one-per-commit over K rounds. An update
   with staleness S rounds has `queue_wait_s ≈ S × wall_per_round ≈ 13 × 1.31 ≈ 17s`.
   Fixing advance (not queue draining logic) is the correct lever.
4. **Accuracy diverges** (0.073) because wrong round count → different aggregation
   trajectory.

**Fix applied (Jun9 evening):** felix overhead 0.16 → **0.315** in `_metadata/baselines.yaml`.

**Predicted outcomes for overnight run:**
- advance: 4.52s ≈ real ✓ → rounds: ~2788 ≈ real 2719 ✓
- staleness: 12.16 / 4.52 ≈ 2.7 rounds ≈ real 2.79 ✓
- buffer backup: inter-sct gap ≈ 1.9s (het. budgets) >> 0.315 → no backup
- sim_rate: ~3.4x (up from 2.28x; fewer rounds × same wall/round)
- queue_wait_s: ~2.7 × 1.31 ≈ 3.5s (residual ~2.8s gap vs real 0.7s from
  `SIM_RECV_GRACE_FLOOR_S=2.0` — expected artifact, not a bug)

## §4  Next tasks (sim-real parity)

1. **[OVERNIGHT] Validate felix overhead=0.315 (3.5h run, tonight).** Checks to pass:
   - `overhead_residual`: residual_s ≈ 0, rel < 0.10 ✓
   - `per_round_advance`: sim ≈ 4.52s, ks < 0.2 ✓
   - `throughput`: sim rounds ≈ real 2719 (tol 10%) ✓
   - `staleness`: sim_mean ≈ 2.79 (tol KS < 0.2) ✓
   - `terminal_state` + `total_commits`: pass ✓
   - `convergence`: acc_diff < 0.05 ✓
   - `commit_gap_s` in telemetry: should collapse to ~0 (no backup)
   - `queue_wait_s` P50 in `[LAG_DECOMP]`: expect ~3–4s sim (down from 13.1s);
     residual gap vs real 0.7s is the `SIM_RECV_GRACE_FLOOR_S=2.0` artifact.
   - `sim_rate`: expect ~3.4x (up from 2.28x)

2. **[OVERNIGHT] Validate oort/feddance no regression.** Recheck:
   - oort: terminal_state trainers ratio (was 10.8% at 094917 — recheck after longer run)
   - feddance: advance KS-shape, add `trainer_speed_s` telemetry to real feddance
     runs (field missing from real, causing DIST fail on a missing-data artifact)

3. **[DONE] Lazy weight-deserialize on the sync recv barrier (refl speedup).**
   Trainer (sim mode): pre-serializes weights as `WEIGHTS_BYTES` (raw cloudpickle
   bytes) in outer message dict instead of the live tensor. Aggregator barrier drains
   all N outer dicts cheaply (bytes-copy, not tensor-reconstruct), builds SCT priority
   queue, pops K minimum → calls `cloudpickle.loads(WEIGHTS_BYTES)` only for K.
   Expected: refl barrier 688ms × (1 - (N-K)/N) ≈ 688ms × 0.2 + fixed = ~200ms;
   per-round wall ~1.5s → ~1.0s; refl speedup 1.18x → **~1.8–2.2x**.
   Implementation: `MessageType.WEIGHTS_BYTES=41`, `syncfl/trainer.py` (pre-serialize),
   `syncfl/top_aggregator.py` (lazy deserialize in `_sync_sim_recv_first_k`).
   TODO: validate refl sim_rate after overnight run.

   TODO: Also check oort trainer (oort uses its own `_distribute_weights` via
   `oort/top_aggregator.py`) — does oort sim barrier also benefit? oort uses
   `syncfl._sync_sim_recv_first_k` via inheritance so YES, it benefits too.

4. **[DONE] Feddance `trainer_speed_s` telemetry fix.**
   Base `syncfl` stack never set `PROP_ROUND_DURATION` (only oort overlay did).
   Fix: `syncfl/top_aggregator.py` sets it from `wall_lag_s` (recv − dispatch ts) in
   real mode when not already set. Feddance real telemetry `trainer_speed_s` was 0.0;
   after fix it will reflect actual round duration.
   TODO: verify feddance `trainer_speed` DIST check passes after overnight run.

5. **[AFTER OVERNIGHT] Refl eligibility drift (KS=0.48).** Eligible set diverges between
   real and sim. Likely downstream of availability trace time-base mismatch (A3):
   sim indexes the trace by vclock, real by wall clock — at lower vclock rate (sim)
   different trace windows are hit. Fix: remap trace lookup to virtual time in sim.
   Recheck once speedup improves (more vclock coverage → smaller A3 mismatch).

6. **[DONE] Log level cleanup.** Moved to DEBUG: all `channel.py` recv_fifo trace
   logs, `asyncfl/syncfl` per-commit/per-recv details (`[AGG_RECV_WEIGHTS]`,
   `[AGG_RECV_EVAL]`, `[AGG_START]`, "proceeding to agg weights", "agg_goal reached",
   "aggregation finished", "_agg_training_stats", "Avg training time",
   `[SIM_PENDING]`, "sending weights to {end}", "received data from {end}").
   Kept at INFO: `[AGG_ROUND]`, `[LAG_DECOMP]`, `[SEND_RECV_LAG]`, `[SIM_BARRIER]`,
   `[SYNC_SIM_RECV]`, `[AGG_COMMIT_TIMING]`, `[DISTRIBUTE_TIMING]`, `[TRAIN_CYCLE]`,
   and staleness/participation summaries every 100 rounds.
   TODO: plot improvements — reduce CDFs to ~8 parity-relevant plots, batch
   per-trainer figures into one call; profile `analyze_run.py` to find slow path.

Telemetry to read after overnight: `agg_round.commit_gap_s/buf_depth/residence_rounds`
(sim), `[SIM_BARRIER]` (`sct`/`T_v`), `[LAG_DECOMP]` (`queue_wait_s`, both modes),
`system/agg_commit_timing_cdf.pdf`.
