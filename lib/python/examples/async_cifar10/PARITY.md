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

### Current status (Jun 10 — felix/refl run 150131, 45-min; oort/feddance from Jun 9)

**Aggregation correctness fixed (Jun 10).** The lazy-deserialize change had broken
the sim weight path for the oort + asyncfl aggregators (refl crashed with
`UnboundLocalError`; felix silently never aggregated). Unified on `WEIGHTS_BYTES` +
`common.util.materialize_weights` across all up-path reads — see §4/#3. Smoke
confirms both baselines aggregate in sim **and** real; the Jun-10 scorecard below is
the first post-fix parity run.

**Speedup: done + extended.** Removed across all baselines: recv-poll, distribute
stagger, per-send re-serialize, disk-backed cache (→ in-memory MemCache). Eval/
checkpoint off the critical path. Lazy weight-deserialize now applies to **both real
and sim** (trainer always ships `WEIGHTS_BYTES`; aggregator reconstructs only the
committed K). refl sim_rate 1.18 → **2.31x**; felix 2.28 → **3.59x**.

**Cross-baseline scorecard** (felix/refl: `parity_check.py`, 45-min / 3000s / n=300 /
syn_0; oort/feddance still Jun-9 3.5h):

| baseline | run | sim_rate | rounds real→sim | advance real/sim | staleness real/sim | acc diff | overhead | top remaining fail |
|---|---|---|---|---|---|---|---|---|
| feddance | 094917 | 18.4x | 365→370 (+1.4%) | 33.85/34.11s | 183.0/185.5 (ks=0.01) | 0.029 ✓ | 0.0 | advance KS-shape (0.57), trainer_speed absent in real |
| oort     | 094917 | 12.0x | 666→701 (+5.3%) | 18.49/17.98s | 0/0 | 0.033 ✓ | 0.0 | terminal trainers 255/286 (10.8%); total_commits 2.6% |
| refl     | 150131 | **2.31x** | 2016→1887 (−6.4%) | 1.48/1.74s | 3.02/3.00 ✓ | 0.034 ✓ | 0.10→**0.074** | overhead over-charge (advance +15%) → throughput/terminal/commits; eligibility KS=0.27 |
| felix    | 150131 | **3.59x** | 664→667 (+0.5%) | 4.52/4.95s | 2.79/**7.19** ✗ | **0.024** ✓ | 0.315 | **staleness — reorder-buffer residence (§3c)**; advance KS 0.37; total_commits 8.4% |

**Per-baseline status:**

- **feddance** (Jun 9): essentially passing. Advance KS-shape high (0.57) despite mean
  match; real telemetry `trainer_speed_s` gap. No new run yet.
- **oort** (Jun 9): near-pass. terminal_state 10.8%, total_commits 2.6% over tol;
  likely noise at ~700 rounds. No new run yet.
- **refl** (Jun 10): **speedup goal hit** (1.18 → 2.31x via lazy deserialize).
  staleness (3.00≈3.02), avail_timebase (A3, max_rel 0.015), convergence (acc 0.034,
  loss 0.143 — both now pass) all PASS. eligibility KS improved 0.48 → 0.27 (still
  >0.2). **New, expected**: the faster barrier dropped the per-commit floor, so the old
  `overhead=0.10` now **over-charges** (advance 1.74 vs 1.48, residual −0.26) →
  throughput/terminal/total_commits fail. Retuned 0.10 → **0.074** (= 0.10 + −0.26/K);
  revalidate next run.
- **felix** (Jun 10): **rounds (+0.5%), overhead, and convergence (0.073✗ → 0.024✓)
  now PASS** — the §3b overhead=0.315 retune worked. **The one real remaining gap is
  staleness: 7.19 vs real 2.79.** Root-caused (§3c) to reorder-buffer **residence**,
  not the clock (advance/overhead now match). Secondary: advance KS-shape (0.37,
  mean matches), total_commits 8.4% (downstream of staleness), utility (downstream).

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

### §3c — Residual felix staleness = reorder-buffer residence (Jun10, overhead=0.315)

The overhead retune fixed the **clock** (advance 4.95 ≈ real 4.52; `overhead_residual`
PASS), and rounds now match (667 vs 664). **But staleness is still 7.19 vs real 2.79.**
The §3b prediction (staleness → ~2.7 once advance matched) was wrong because it assumed
"no buffer backup." The Jun-10 telemetry shows the buffer **is** chronically backed up —
this is now the lowest broken rung for felix.

**Evidence (felix sim 123632 vs real 125815):**

| metric | SIM | REAL |
|---|---|---|
| staleness mean / p90 / p99 | **7.19 / 18 / 33** | 2.79 / 5 / 9 |
| `commit_gap_s` (= vclock − sct) mean / p90 / max | **25.3 / 82 / 172 s** | n/a |
| `buf_depth` (reorder buffer occupancy) | **~28 steady** | n/a |
| `residence_rounds` mean / max | **9.75 / 392** | n/a |
| `concurrency` / `in_flight` | 30 / 31 | 30 / 31 |

**Mechanism — NOT buffer residence; it is late PHYSICAL arrival vs a decoupled
clock.** Bucketing the felix-sim commits by `commit_gap_s` settles it: the
high-staleness commits have `residence_rounds ≈ 0` but `commit_gap_s = 40–300s`, and
`corr(commit_gap, residence) ≈ −0.06`. So they did **not** sit in the buffer — they
were enqueued and committed in the **same** round, but their completion time was
40–300 virtual-seconds in the past. They are **late physical arrivals**:

| `commit_gap_s` | `residence_rounds` (p50) | staleness |
|---|---|---|
| 0–2s | 2 | 2.6 |
| 5–15s | 4 | 4.0 |
| 40–100s | **0** | 15.1 |
| 100–300s | **0** | 27.9 |

Why does a sim update arrive "late" relative to its own completion? Because **the sim
trainer does not sleep its budget** — it computes fast (real GPU ms) and stamps a
*future* `sim_completion_ts = dispatch_vclock + budget`. The message therefore lands
physically ~immediately, normally *before* its sct (buffered, committed when the vclock
reaches sct, `commit_gap ≈ 0`). But under GPU contention (300 trainers / few GPUs) a
trainer's real round-trip can exceed the wall time it takes the fast vclock to reach its
sct (≈ `budget/advance × wall_per_round` ≈ 15s for a 56s budget). Then its message is
drained **after** the clock already advanced past its sct — committed out of completion
order, with `staleness = current_version − trained_version` inflated by `commit_gap/advance`.

`concurrency=30` is identical in sim and real, so this is **not** over-selection, and
**not** the reorder buffer (residence ≈ 0 for the bad commits). It is the virtual clock
advancing **past in-flight updates that have not arrived yet**, because the sim's
physical execution (real wall-time) is decoupled from the virtual clock.

**Why real doesn't have it:** in real the trainer *actually takes* its budget wall-time,
so each update arrives at its true completion and the model version is exactly right at
that moment — real "waits" implicitly. Charging `current_version − trained_version` is
correct there.

**Dead end — the ordering gate (reverted).** First attempt: make the sim wait for a
not-yet-arrived in-flight trainer expected to complete earlier. The diagnostics killed
it: `[SIM_GATE_DIAG]` showed `mean_n_outstanding = 0` on every line — there is NEVER an
un-arrived trainer to wait for (messages arrive physically fast and are already
buffered), so the gate never fired (`barrier_wait` p50 = 0) and staleness was unchanged.
A `version_at(sct)` "re-label `trained_version`" band-aid was also tried and reverted
(it masks the number; staleness must stay `current_version − trained_version`).

**Root cause — IMPLEMENTED FIX (Jun10): the per-commit overhead had taken over the
virtual clock.** `_advance_sim_clock` did `vclock = max(vclock, sct) + overhead`. The
clock-split diagnostic is damning: by end-of-run `vclock = 3189s` decomposes into
`overhead_cum = 3150` + `sct_adv_cum = 39` — i.e. **98.8% of the clock is accumulated
overhead**, and `sct_adv_cum` froze at ~39 after round 43. The 0.315 overhead (×~10
commits/round = 3.15 s/round) shoved the clock ahead of every completion, so the
committed `sct` was always behind it (`max(vclock, sct) = vclock`, a no-op) and the
trainers' actual completion times stopped driving the clock entirely. The clock became a
pure overhead ramp with no relation to when trainers finish → staleness (clock-position
at commit − at dispatch) was measured against a meaningless axis → inflated AND drifting
(staleness 3.7→9.2, commit_gap 10→36 across rounds; real is flat ~2.8). `dup_buffer_adds
= 0` ruled out re-selection.

**Fix:** set felix `simCommitOverheadSeconds = 0` so `_advance_sim_clock` is just
`vclock = max(vclock, sct)` — the clock TRACKS completions. The dead gate is removed; the
clock-split + buffer past/future logging is kept as `[SIM_CLOCK_DIAG]` to verify (expect
`overhead_cum ≈ 0`, `sct_adv_cum ≈ vclock`). 35 readiness/ordering tests pass.

**Known consequence to validate:** the overhead was masking a real ~2.98 vs 4.52
per-round-advance gap (the natural sim completion spacing is lower than real's). With
overhead=0, staleness stops drifting and becomes `budget/advance` (consistent), but if
advance lands ~3.0 the *throughput* will mismatch — that residual is a completion-spacing
/ dispatch-timing issue (likely the `mqtt_fetch` 60s-real vs 23s-sim gap, plus whether
the buffer holds FUTURE completions — `buf_future` in the diag), to be modeled in the
trainer `sct`, NOT re-faked on the clock. The next 45-min felix run + `[SIM_CLOCK_DIAG]`
tells us whether throughput needs that follow-up.

### §3d — Validation: overhead=0 + virtual-completion gate (Jun11, run 012226) — STILL DRIFTS

Two fixes were live in this run: (a) `simCommitOverheadSeconds = 0` (§3c), and (b) a
re-implemented **virtual-completion gate** in `asyncfl._sim_recv_min` that tracks
per-trainer expected completion (`_sim_inflight_expected[end] = send_ts + dur`) and is
supposed to make the clock wait for a still-in-flight trainer whose expected completion
precedes the buffered minimum, rather than committing a far-future update and racing the
clock past the laggard.

**Result — the staleness drift is NOT fixed.** Telemetry (`agg_round.staleness`, n=13250),
mean staleness by round-decile: `3.65 → 6.10 → 7.85 → 8.65 → 9.73 → 9.94 → 10.75 → 11.30
→ 11.80 → 12.26`. Overall mean **9.21 vs real ~2.79** — monotonic, no plateau, same shape
as before the fix. `commit_gap_s` rises 2 → 13 s and plateaus; `residence_rounds` ~11–12;
final test-accuracy ~0.24 at round ~1300.

**The §3c overhead fix held** (good): every `[SIM_CLOCK_DIAG]` shows `overhead_cum = 0`,
`sct_adv_cum = vclock`, `dup_buffer_adds = 0`. So the clock is no longer an overhead ramp.
But staleness still drifts, which means the overhead was a *contributing* inflator, not
the whole mechanism.

**The gate (b) is inert.** Every diag line: `barrier_wait_s ≈ 0`, `gate_failsafe = 0`. It
never waits and never fires. Yet `buf_past` is routinely 1–10 per line and `commit_gap_s`
spikes to 91 / 372 / 720 / 1625 / 2747 s with `vclock_lead_over_buf` spiking in lockstep.
So updates with `sct << vclock` keep being committed — exactly what the gate was meant to
prevent — but the gate's `_sim_inflight_expected` never flags them, because the per-trainer
`dur` estimate (running mean ~12 s) places their expected completion in the near future,
not "earlier than buffered_min." A straggler re-dispatched many rounds ago has a true sct
deep in the past, but the gate predicts it near-now → no wait. The gate only catches
laggards whose *predicted* completion is early, which is never the stragglers that matter.

**Refined diagnosis (the actual mechanism).** Staleness drift is driven by **out-of-order
commits of past-dated updates advancing the round/version counter without advancing the
clock**: when a buf_past update commits, `vclock = max(vclock, sct) = vclock` (no-op), but
`round` still increments. Many such commits pile version increments at a frozen clock, so
every subsequently-measured update's `staleness = current_version − trained_version`
inflates, and it compounds over rounds. The clock and the version counter are decoupled in
the *opposite* direction from §3c: there it was clock-runs-ahead-of-completions; here it is
version-runs-ahead-of-clock. The real fix must **keep version and clock coupled** — either
the clock advances on every commit at the rate implied by completion spacing, or past-dated
updates are committed at the clock position where their sct actually fell (so they do not
retroactively inflate everyone else). The prediction-based gate cannot do this because it
needs accurate per-trainer durations it does not have.

**Status: PAUSED by user (Jun11)** to run experiments; revisit after. The gate code
(993ff450) and `[SIM_CLOCK_DIAG]` instrumentation remain in tree — harmless (inert), and
the diag is what diagnosed this. Decide on removal vs. fix when resuming (see §4.1).

### §3e — Fix attempt: the gate predictor learned the WRONG quantity (Jun11, awaiting run)

Resumed §3d. Re-tracing end-to-end pinned **why the gate is inert** to a concrete code
bug, not a design dead-end: the gate's per-trainer expected-completion predictor
(`_sim_inflight_expected[end] = sim_send_ts + budget`) learned `budget` from
**`SIM_ROUND_DURATION`** — which is `max(gpu_time, modeled_budget)` (trainer
`main.py:804`), i.e. **inflated by GPU contention**. Under contention a trainer's observed
"budget" spikes, so its *next* expected completion is pushed into the future → the gate's
`min_stuck < buffered_min` test never trips → the clock laps the in-flight straggler →
its update commits **past-dated** (`sct << vclock`) → `_round` increments while
`_advance_sim_clock` is a no-op → version drifts ahead of the clock → staleness inflates
and compounds (the §3d drift).

**The right predictor is `TRAINING_BUDGET_S`** (= the stable, contention-free modeled
`training_delay_s`, already on the wire — the async aggregator reads it at
`asyncfl/top_aggregator.py:606`). Since real `sct = send_ts + max(gpu, budget) ≥ send_ts +
budget`, the modeled budget is a true **lower bound** on completion, so a gate clamped to
it can hold the clock until a straggler can *plausibly* have finished, and never overshoot
a real completion. This is what makes the EXISTING gate machinery fire on genuine
stragglers (dispatched-long-ago, deep-past true sct) instead of being silenced by a GPU
spike — no new ordering logic needed, just the correct input.

**Changes (this commit):**
- Predictor learns from `TRAINING_BUDGET_S` (fallback to `SIM_ROUND_DURATION` for old
  msgs). Renamed `_sim_trainer_dur`→`_sim_trainer_budget`, `_sim_dur_*`→`_sim_budget_*`.
- New `[SIM_CLOCK_DIAG]` fields to confirm/refute the hypothesis from the next logs:
  `gate_holds` (gate should now actually fire — was ~0), `pastdated_commits`,
  `pastdated_gap_cum`, `pastdated_gap_max` (should fall toward 0 if the fix works),
  `budget_mean` (sanity: should ≈ real mean training_budget_s, NOT the contention-inflated
  SIM_ROUND_DURATION mean).

**What the next felix run tells us:**
- **Fix works:** `gate_holds` > 0, `pastdated_commits`/`pastdated_gap_cum` collapse,
  staleness stops drifting and lands ≈ `budget/advance` ≈ 2.7 (real 2.79). Watch
  `barrier_wait_s` rises modestly (genuine holds) and whether `advance`/rounds stay matched
  — if holding the clock pushes advance down and rounds up, that's the §4.1b throughput
  residual surfacing, to model in the trainer `sct`, NOT re-fake on the clock.
- **Fix insufficient:** if `gate_holds` > 0 but `pastdated_commits` stays high, the
  stragglers arrive *physically after* the gate's failsafe deadline (`RECV_TIMEOUT_WAIT_S`)
  → the clamp can't wait that long without killing speedup → next attempt is candidate (ii)
  proper (commit-at-sct reorder / version_at(sct) incorporation), not prediction.

### §3f — Run 221402 result + the principled fix: completion-frame staleness (Jun12)

**The §3e predictor fix did its narrow job but did NOT fix the drift** — this is the
"Fix insufficient" branch above. Run 221402 (`overhead=0`, modeled-budget predictor):
- Predictor works: `gate_holds` 0 → **391k** (gate now fires), `budget_mean=11.8` (the
  stable modeled budget, not the inflated `SIM_ROUND_DURATION`). `gate_failsafe=0`.
- But staleness still drifts: decile mean **3.53 → 11.74**, overall **8.75** (real 2.79) —
  ~unchanged from run 012226 (3.65 → 12.26). `barrier_wait_s ≈ 0` on 25/36 diag lines:
  the gate **spins without waiting** (pass-cap, not failsafe), so it can't prevent the
  past-dating — and it *fundamentally can't*, because waiting wall-time for a still-running
  straggler every commit would destroy the speedup.

**New, decisive diagnostics that relocate the root cause:**
1. **Completions are CLEAN.** Trainer telemetry: `overran=0.0%`, `real_gpu_time≈0.1s`,
   `sim_round_duration == training_budget` exactly. So `sct = sim_send_ts + modeled_budget`
   with NO contention inflation. The earlier "late physical arrival under GPU contention"
   story (§3c) does **not** apply to these runs — messages arrive in ~0.1s.
2. **`commit_gap` (vclock − committed sct) grows monotonically** across the run:
   round-band means 30 → 230 s, max 214 → 2911 s, **59% of all 18 239 commits past-dated**
   (`gap>2`), while `buf_depth ≈ concurrency (30)` stays flat and `probed ≈ 1/commit`.
   Round-36 smoking gun: `T_v=139` while 12 updates with `sct=3..36` are committed in one
   sweep (`probed=12`, `buf_depth=39`) — i.e. **completed updates enter the reorder buffer
   long AFTER the clock passed their sct**, so they commit massively past-dated.
3. Staleness tracks `commit_gap` exactly: `staleness ≈ budget/advance + commit_gap/advance`.

**Root cause (relocated):** not GPU contention, not the clock formula — it is the
**receive/reorder pipeline**: a completed update sits un-received (not yet in the buffer)
while the clock advances via other commits, then is drained late and committed out of
completion order. Because `staleness = self._round − trained_version` counts versions up
to the **physical commit**, this late/out-of-order commit inflates the number — and it
compounds, because every late commit pushes `self._round` further ahead of the version
that was actually current when the update completed. (Note `staleness_factor=0.0` in
`optimizer.do` → staleness is a **measurement**, not a model input; the model trajectory
is unaffected, so this is purely a fidelity-of-measurement bug.)

**The principled fix (IMPLEMENTED, Jun12) — measure staleness in the completion frame.**
An update's staleness is, by definition, the number of model versions published during its
**actual training interval** `[sim_send_ts, sct]`. So charge it against the version that was
current when it *truly completed* (`vclock = sct`), not the version inflated by the receive
pipeline at physical-commit time:
`staleness = version_at(sct) − trained_version`.
This is the physically-correct async semantics and is **identical to the raw value in real
mode** (commits are in-order there, so `version_at(sct) == self._round`) and whenever
`commit_gap == 0`; it strictly removes the receive/reorder inflation otherwise. It is NOT
the §3c reverted band-aid (which relabeled `trained_version` — a fact); here `trained_version`
is untouched and the *incorporation reference* is moved to completion time.

Implementation (`asyncfl/top_aggregator.py`): a monotone `_version_vclock_log` of
`(vclock, version)` appended on each version bump (`_record_version_vclock`), and
`_version_at(t)` bisects it. The raw value is kept in telemetry as `agg_round.staleness_raw`
so the next run quantifies exactly how much of the 8.75 was pipeline artifact vs. genuine
`budget/advance` baseline. Guarded by `TestCompletionFrameStaleness` (3 cases).

**What this fixes / does NOT fix (the two causes are now cleanly separated):**
- **Fixes the DRIFT.** Expected: `staleness` flat at ≈ `budget/advance` (no upward decile
  trend), `staleness_raw` still drifts (showing the artifact removed). Predicted level
  ≈ `11.7/2.64 ≈ 4.4`.
- **Does NOT fix the advance/throughput gap** (`advance` 2.64 vs real 4.52). That residual
  baseline (4.4 vs real 2.65 ≈ the 1.7× advance ratio) is §4.1b — model the real per-commit
  round-trip/dispatch cost in the trainer `sct` so completions are spaced like real, NOT on
  the clock. After §3f lands, the remaining staleness gap should be *exactly* the advance
  ratio, confirming the decomposition.

## §4  Next tasks (sim-real parity)

1. **[OPEN — felix staleness STILL drifts after §3c+gate (run 012226, §3d); PAUSED] Overhead off the
   clock (§3c).** Root cause: the per-commit overhead (0.315) had taken over the virtual
   clock (`vclock = 98.8% overhead_cum`). The overhead fix (`simCommitOverheadSeconds = 0`)
   **landed and held** — run 012226 confirms `overhead_cum = 0`, `sct_adv_cum = vclock`.
   **But staleness still drifts** (decile mean 3.65 → 12.26, overall 9.2 vs real 2.79; §3d).
   So overhead was only a contributing inflator. The re-added virtual-completion gate
   (993ff450) is **inert** (`barrier_wait_s ≈ 0`, `gate_failsafe = 0`) — its per-trainer
   duration prediction never flags the deep-past stragglers, so buf_past updates keep
   committing and racing version ahead of the clock.

   **Refined root cause (§3d):** out-of-order commit of a past-dated update increments the
   round/version counter while `vclock = max(vclock, sct)` is a no-op → version drifts ahead
   of the clock → `staleness = current_version − trained_version` inflates and compounds.
   When resuming, the fix must **keep version and clock coupled**, NOT predict durations.
   Two honest candidate designs to evaluate (do NOT band-aid the staleness number):
   - **(i) Couple the clock to commits:** advance `vclock` by the empirical inter-completion
     spacing on *every* commit (incl. past-dated ones), so N commits = N units of clock, and
     version/clock stay locked. Risk: distorts the time-base for availability indexing.
   - **(ii) Commit-at-sct ordering:** do not let a past-dated update increment the *current*
     version; account it at the version that was current when its `sct` actually fell (a true
     reorder, not a relabel of `trained_version`). This is the physically correct async
     semantics and keeps staleness = real versions-elapsed.
   Decide gate's fate then (remove if (ii), since reorder subsumes it). Keep `[SIM_CLOCK_DIAG]`.

   **UPDATE (§3e/§3f, Jun12):** predictor fix (§3e) made the gate fire (`gate_holds` 0→391k)
   but did NOT fix the drift (run 221402). New diagnostics relocated the root cause: completions
   are CLEAN (0% overrun, `sct=send_ts+budget`), but the **receive/reorder pipeline** commits
   updates long after their `sct` (`commit_gap` 30→230s, 59% past-dated), inflating
   `self._round − trained_version`. **Implemented candidate (ii) as completion-frame staleness:**
   `staleness = version_at(sct) − trained_version` (`_version_at` bisects a monotone
   `_version_vclock_log`). Awaiting the next run to confirm the drift flattens (`staleness` flat,
   `staleness_raw` still drifts). The gate is now redundant for staleness but left in (inert,
   harmless); remove once §3f is confirmed. See §3f.

1b. **[STILL OPEN — felix throughput; now the PRIMARY residual after §3f] Close the 2.64 vs 4.52
   per-round-advance gap honestly.** After §3f removes the drift, the remaining staleness gap
   should be exactly the advance ratio (`budget/advance` ≈ 4.4 vs real 2.65 ≈ 1.7×). The residual
   advance gap is a completion-spacing / dispatch-timing issue (the `mqtt_fetch` 60s-real vs
   23s-sim phase divergence). Model the real round-trip cost in the trainer `sct`, NOT on the
   clock. Diagnose with `[SIM_CLOCK_DIAG]` `buf_past` vs `buf_future` before designing.

2. **[APPLIED — revalidate] Refl overhead retune 0.10 → 0.074.** The lazy-deserialize
   speedup (1.18 → 2.31x) made the sync barrier faster, so 0.10 now over-charges (advance
   1.74 vs 1.48, residual −0.26 → `0.10 + (−0.26/K=10) = 0.074`). Applied in
   `_metadata/baselines.yaml`. Next refl run should pass `per_round_advance`,
   `overhead_residual`, `throughput`, `terminal_state`, `total_commits`.

3. **[OPEN] Refl eligibility drift, KS 0.48 → 0.27 (still >0.2).** Improved as `avail_timebase`
   (A3) now passes (max_rel 0.015), but the eligible set still shifts. Recheck after the
   0.074 retune (better throughput → more matched coverage); if it persists, it is a
   genuine eligible-set divergence to localize at Stage 2, not a clock artifact.

4. **[OPEN] Re-run oort + feddance** on the post-fix code at 45-min to refresh their
   Jun-9 numbers (oort terminal_state 10.8%, total_commits 2.6%; feddance advance
   KS-shape + the real `trainer_speed_s` telemetry, item 6 below).

5. **[DONE — Jun10] Lazy weight-deserialize, now unified across all paths.**
   Trainers ship the weight update as `WEIGHTS_BYTES` (raw cloudpickle bytes) instead
   of a live tensor, so the aggregator reconstructs the tensor only for the K updates
   it commits, not the N-K it discards. (The channel's recv otherwise eagerly
   `cloudpickle.loads` every received tensor — even surplus/stale ones.)

   **Regression found Jun10 (095102/095139):** the original change only handled bytes
   in `syncfl._sync_sim_recv_first_k`. The earlier note's claim that "oort benefits via
   inheritance" was WRONG — refl/oort and felix/async use their *own* recv/handle paths
   (`_oort_sim_recv`, `_sim_recv_min`, their own `_handle_weights_msg`), which still read
   `MessageType.WEIGHTS`. Result: **refl** raised `UnboundLocalError` (weights only bound
   inside `if WEIGHTS in msg`) → aggregator died on the first round-1 update; **felix**
   silently never counted updates (async read skipped) → 0 aggregations, no accuracy,
   60 GB spin telemetry. Real mode unaffected (the bytes swap was `if simulated`).

   **Fix (Option B, unified):** trainers always send `WEIGHTS_BYTES` (both real+sim, so
   real overcommit also skips discarded reconstructions); every up-path aggregator read
   goes through `common.util.materialize_weights(msg)` (bytes→WEIGHTS in place,
   idempotent, backward compatible with live-tensor messages, returns None for eval-only
   updates) + a defensive `weights = None` default; the async train-vs-eval router now
   keys off `WEIGHTS or WEIGHTS_BYTES`. Touches: `common/util.py`, `syncfl/trainer.py`,
   and the up-path reads in syncfl/asyncfl/oort tops + syncfl/asyncfl/eager middles +
   eager top. Guarded by `tests/mode/test_weights_bytes_roundtrip.py`.
   Validated Jun10: smoke (felix+refl, sim+real) aggregates with 0 errors; the 150131
   parity run is the first clean post-fix scorecard (sim_rate 2.31x/3.59x).

6. **[DONE] Feddance `trainer_speed_s` telemetry fix.**
   Base `syncfl` stack never set `PROP_ROUND_DURATION` (only oort overlay did).
   Fix: `syncfl/top_aggregator.py` sets it from `wall_lag_s` (recv − dispatch ts) in
   real mode when not already set. Feddance real telemetry `trainer_speed_s` was 0.0;
   after fix it will reflect actual round duration.
   TODO: verify feddance `trainer_speed` DIST check passes after a fresh run.

7. **[DONE] Log level cleanup.** Moved to DEBUG: all `channel.py` recv_fifo trace
   logs, `asyncfl/syncfl` per-commit/per-recv details (`[AGG_RECV_WEIGHTS]`,
   `[AGG_RECV_EVAL]`, `[AGG_START]`, "proceeding to agg weights", "agg_goal reached",
   "aggregation finished", "_agg_training_stats", "Avg training time",
   `[SIM_PENDING]`, "sending weights to {end}", "received data from {end}").
   Kept at INFO: `[AGG_ROUND]`, `[LAG_DECOMP]`, `[SEND_RECV_LAG]`, `[SIM_BARRIER]`,
   `[SYNC_SIM_RECV]`, `[AGG_COMMIT_TIMING]`, `[DISTRIBUTE_TIMING]`, `[TRAIN_CYCLE]`,
   and staleness/participation summaries every 100 rounds.

8. **[DONE — Jun10] Plotting overhaul** (`scripts/analysis/analyze_run.py`, plan in
   `PLOTTING.md`): single-pass aggregator-log parser (was 6 full scans of the 2.2 GB
   log), `binned_line` for the noisy/slow series, and new deep-dive subdirs
   `plots/{availability,aggregation,selection/why}/`. The `aggregation/` set
   (`buffer_health_over_rounds`, `commit_gap_cdf`, `residence_rounds_cdf`,
   `staleness_vs_speed`) is exactly what §3c needs to watch the felix buffer fix land.

Telemetry to read for the felix buffer fix (§3c): `agg_round.commit_gap_s/buf_depth/
residence_rounds` (sim; should collapse toward 0/K/~1), `[SIM_BARRIER]` (`sct`/`T_v`),
`[LAG_DECOMP]` (`queue_wait_s`, both modes), `plots/aggregation/*`.
