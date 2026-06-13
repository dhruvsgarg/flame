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

### Status: refreshed from the Jun 13 4h n=300 runs

All four baselines re-run real+sim for a 14400s (4h) budget at n=300, α=0.1. Numbers below are
from those runs (`experiments/parity_<baseline>.json`; real/sim dirs dated `20260613`). Eval
cadence = every 50 FL rounds; accuracy = the **last reported eval** per run.

**A. Cross-baseline performance** (same 4h budget, so rounds + attained accuracy are comparable):

| baseline | FL rounds r/s | final test-acc r/s | final loss r/s |
|---|---|---|---|
| refl     | 9054 / 8587 | **0.579** / 0.565 | 3.20 / 3.06 |
| felix    | 3402 / 3913 | **0.579** / 0.578 | **1.18** / 1.17 |
| oort     |  777 /  839 | 0.508 / 0.480 | 1.34 / 1.51 |
| feddance |  504 /  579 | 0.428 / 0.494 | 1.79 / 1.67 |

**Read:** felix ≈ refl at the top (~0.579 real acc, within eval noise — felix does *not* strictly
out-accuracy refl), but felix reaches it in ~⅓ the rounds and at the **lowest loss** (1.18);
refl's accuracy comes with a high/miscalibrated loss (3.2). oort (0.51) and feddance (0.43) trail.
So felix is co-best on accuracy and best on loss + round-efficiency.

**B. Parity status** (lowest broken rung; full report per-baseline JSON):

| baseline | speedup | enforced pass | clock (K2/K3) | staleness r/s | root-cause FAIL(s) | verdict |
|---|---|---|---|---|---|---|
| refl     | 1.73x | **31/36** | PASS (terminal ✓) | 2.93 / 2.87 ✓ | trainer_speed, participation | FAIL (inputs) |
| felix    | 1.32x | 28/36 | K3b rel 0.112 | 2.80 / 4.75 (KS 0.166) | overhead_residual, participation | 1 ROOT (clock under-charge) |
| oort     | 14.2x | 28/36 | PASS | 0 / 0 ✓ | trainer_speed, selection_detail, training_budget | FAIL (inputs) |
| feddance | 15.8x | 26/36 | K3 KS 0.58 | 0 / 0 ✓ | trainer_speed, training_budget, participation | FAIL (inputs) |

> **felix is "1 root from green," not "outcomes match."** The +15% rounds gap (sim 3913 vs real
> 3402) and the staleness/terminal/commits FAILs are **all downstream of one mechanism**: K3b
> under-charges 0.046s/commit, which compounds (0.046 × ~10 commits/round × ~3900 rounds ≈ the 11%).
> Fix K3b (§4.1) and throughput/terminal/commits/staleness close together. The §3m latency-knob
> zeroing (`simRedispatchGapSeconds=0`, `simCommitOverheadSeconds=0`) is what removed that 0.046s.

**C. Full ladder** (✓/✗ + the actual margin vs tolerance; from `parity_<baseline>.json`):

| stage·check (tol) | felix | oort | refl | feddance |
|---|---|---|---|---|
| 1c P3 trainer_speed (KS≤.10) | ✓ .086 | ✗ .125 | ✗ .153 | ✗ .197 |
| 1m K3b overhead (rel≤.10) | ✗ .112 | ✓ .053 | ✓ .076 | ✗ .111 |
| 1e K3 advance (KS≤.20) | ✗ .214 | ✓ .193 | ✓ .128 | ✗ .581 |
| 1e K2 throughput (rel≤.10) | ✗ .112 | ✓ .052 | ✓ .072 | ✗ .110 |
| 2 A2 eligibility | ✓ | ✓ (mean) | ✓ .226 | ✓ |
| 2 A3 timebase | ✓ | ✓ | ✓ | ✓ |
| 3 S3/4 num_chosen (rel≤.05) | ✓ .018 | ✗ .177 | ✓ .000 | ✓ .000 |
| 3 S2 participation (avg_diff) | ✗ 18.5 | ✗ 11.1 | ✗ 227 | ✗ 17.4 |
| 4 T2 budget (KS≤.10) | ✓ .005 | ✗ .208 | ✓ .061 | ✗ .133 |
| 4 T gpu_compute (KS≤.25) | ✓ .025 | ✓ .029 | ✓ .025 | ✓ .073 |
| 5 U5 arrival (WARN, rho) | −.47 | −.44 | −.47 | −.31 |
| 6 U3 staleness r/s | ✗ 2.8/4.75 | ✓ 0/0 | ✓ 2.9/2.9 | ✓ 0/0 |
| 6 P1 agg_seq (gated WARN) | ✓ | ✓ | ✓ | ✓ |
| 7 F1-3 utility (pooled KS) | ✓ .022 | ✓ .030 | ✓ .040 | ✓ .065 |
| 8 K8 terminal (rel≤.10) | ✗ .110 | ✗ .054 | ✓ .072 | ✗ .114 |
| 8 U2 commits (rel≤.02) | ✗ .110 | ✗ .054 | ✗ .072 | ✗ .114 |
| 8 C1 accuracy (≤.05) | ✓ .032 | ✓ .042 | ✓ .024 | ✓ .024 |
| 8 C2 loss (≤.15) | ✓ .065 | ✗ .182 | ✗ .194 | ✗ .155 |

Reading the table: **refl** clock GREEN, only CONTROL inputs (P3) + participation + loss left.
**felix** every FAIL traces to K3b (the one cell to fix). **oort/feddance** root at the speed/budget
input (P3+T2) which propagates to advance (feddance K3 .581) and convergence (loss). **S2
participation and U5 arrival fail on all four** — path-dependent for stochastic selectors (§4 item 4).
**Speedup** is GPU-bound, not clock-bound: felix/refl run the most commits so compress least (1.3–1.7x);
oort/feddance run few commits → 14–16x. **pytest ladder guards** (`tests/mode/`, 40 tests) all PASS —
these are code regression guards, separate from this comparator.

### Verification tracker — current metric → post-fix hypothesis (next run)

For every FAIL / needs-improvement row, the **current number** (Jun13 4h run) and the **predicted
number/trend after the staged fix lands**. Fill the "next-run actual" mentally against the hypothesis:
a hit confirms the mechanism; a miss is a regression to localize. Rows with no staged fix are expected
**unchanged** — any movement there is a regression signal (the fix touched something it shouldn't).

**A. felix — staged fix: `simRedispatchGapSeconds 0→1.0` (§4.1); + new `task_send` concurrency check (§4.0)**

| check (tol) | current (Jun13) | hypothesis after fix (next run) |
|---|---|---|
| §4.0 concurrency (NEW) | not measured (old check vacuous) | mean ≈ 27–28 computing of c≈30, peak ≤ ~40 (1.3× overcommit), double_dispatch=0 |
| K3b overhead_residual (rel≤.10) | ✗ .112 — sim advance 3.68 vs real 4.14 | ✓ <.10 — gap spaces completions, sim advance →~4.1 |
| K3 advance (KS≤.20) | ✗ .214 | ✓ ≤.20 |
| K2 throughput (rel≤.10) | ✗ .112 — sim 3913 vs real 3402 rounds (+15%) | ✓ ≤.10 — rounds gap shrinks as advance rises |
| U3 staleness (r/s) | ✗ sim 4.75 / real 2.80 (KS .166) | ✓ sim →~2.8 — slot-hold drops L/C (de-bunches overlap) |
| K8 terminal (rel≤.10) | ✗ .110 | ✓ ≤.10 — closes with throughput |
| U2 commits (rel≤.02) | ✗ .110 | →~0 — closes with throughput (downstream of K2) |
| P3 trainer_speed (KS≤.10) | ✓ .086 | unchanged (✓) — not touched by the gap |

**B. oort / refl / feddance — NO fix staged this run (§4.2 selection-mix not yet coded)**

Expected **unchanged**; recorded so the next run flags any unexpected drift. §4.0 now shows real
selection is sound (redispatch=0), so the divergence is a **sim-selection** question to localize.

| baseline · check (tol) | current (Jun13) | hypothesis after fix (next run) |
|---|---|---|
| oort P3 trainer_speed (KS≤.10) | ✗ .125 | unchanged until §4.2 — commit-weighted by selection mix |
| oort T2 budget (KS≤.10) | ✗ .208 — real 16.4 vs sim 12.3 (per-trainer identical) | unchanged until §4.2 (selection-mix, not input) |
| oort S3/4 num_chosen (rel≤.05) | ✗ .177 | unchanged until §4.2 |
| oort C2 loss (≤.15) | ✗ .182 | unchanged until §4.2 (downstream of selection mix) |
| refl P3 (KS≤.10) / participation (share-KS≤.2) | ✗ .153 / ✗ .497 | unchanged — genuine selection-mix divergence (§4.3 not suppressed) |
| refl C2 loss (≤.15) | ✗ .194 | unchanged until §4.2 |
| feddance P3 / K3 advance (KS) | ✗ .197 / ✗ .581 | unchanged — speed-mix drives the advance KS |
| feddance T2 / participation / C2 | ✗ .133 / ✗ .527 / ✗ .155 | unchanged until §4.2 |

**What landed (validated this run set):**
- **refl overhead retune 0.10→0.074 (§4.2):** K3b residual −0.12 (rel 0.076), K3/K2 PASS, **terminal_state
  now PASS**, eligibility KS 0.48→0.226. refl clock is GREEN; only CONTROL inputs + participation remain.
- **feddance/fedavg staleness bug fix:** staleness now **0/0** (was the bogus ~183/185). oort also 0/0.
- **felix §3m (speed real up):** removing the real settle-sleep brake brought real advance 4.32→4.14,
  staleness 2.81→2.80; **P3 trainer_speed now PASS** (KS 0.086). §5 checker fixes hold (P1/utility/
  phase_mqtt gated → PASS). Only K3b (0.046s/commit) + S2 participation remain.

**What's still open:**
- **felix — K3b overhead_residual rel 0.112** (just over 0.1): sim under-advances (3.68 vs 4.14) and
  over-stales (4.75 vs 2.80). §3m's `simRedispatchGapSeconds=0` **over-overlaps** — completions bunch.
  Re-enabling a small redispatch gap (§3k/§3L) or modeling the 0.046s/commit overhead closes it.
- **trainer_speed / training_budget is the common root** for refl/oort/feddance — the real trainer
  speed model fed to sim differs (oort: budget real 16.42 vs sim 12.27; feddance: 12.44 vs 10.95),
  which then drives feddance's advance KS 0.58. Stage-1/4 CONTROL **inputs**, not the clock formula.
- **S2 participation** fails on all four (avg_diff 18–227) — for stochastic selectors per-trainer
  chosen-counts are path-dependent over 4h. Likely the next §5-style gating fix (mirror P1/S1), not a
  dynamics bug; S3/4 in-flight + S2 means match where checked.

**Ruled out (do not re-chase):** GPU contention (T3 overrun=0); re-dispatch-invariant violation
(SEND_TIMEOUT fires 0×; **and now confirmed at the trainer level — §4.0 `task_send` intervals give
double-dispatch=0 on all baselines; the old "redispatch 4549/26988" was a validator bookkeeping
artifact, not real**); `commit_gap` as a staleness proxy; MQTT drops (0, both directions); the felix
post-compute leg as a "bug" (it's serial-aggregator scheduling, not network — §3).

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

## §3  Felix sim parity — root-cause log

**Real pipeline (n=300, ~50-min):** trainer computes ~11.7s, then a ~1.0s
post-compute leg before its next dispatch → real **cycle ~12.96s**, advance 4.32s,
all_selected ~30 (of which ~27.6 actively computing), **staleness 2.81, commits in true
completion order**. The whole job is to reproduce that in sim where the trainer does
*not* sleep its budget (computes in real GPU ms, stamps a modeled completion `sct`).

**Why all_selected=30 but only ~27.6 compute — verified from LAG_DECOMP (Jun12).** The
post-compute leg is **serial-aggregator scheduling, NOT network.** `wall_lag` (send→commit) =
compute 11.92 + push/mqtt/post ~0.1 = **12.01s** (MQTT round trip <0.1s); + `queue_wait` 0.55s
(serial recv-queue backlog, pre-commit, counts toward staleness: 12.01/4.32≈2.78≈real 2.81) +
re-selection/send/`sleep(0.1)` ~0.4s (post-commit, doesn't count). Cycle 12.96 → computing
fraction 11.92/12.96 ≈ 0.92 → ~27.6 of 30. Drops ruled out (receptions ≥ dispatches,
SEND_TIMEOUT=0). The leg is not a bug — §3m speeds real up by removing the artificial part.

### §3m  Speed real up to ~c computing instead of slowing sim (Jun12, preferred)

The dominant slice of the post-compute leg is removable: a real-only `time.sleep(0.1)` before
selection, hit twice per commit (~0.2s ≈ 46% of the 0.43s/commit budget), which inflates
`queue_wait` and the re-dispatch gap. New knob `realDistributeSettleSeconds` (default 0.1 =
legacy), set to **0** for felix → aggregator compute-bound, real holds ~30, advance ~3.97,
staleness ~3.0. **Validated (Jun13):** real came down to advance 4.14, staleness 2.80 ✓.
**But the sim-side prediction was wrong** — with `simRedispatchGapSeconds=0` sim *over-overlaps*
(advance 3.68, staleness 4.75; K3b rel 0.112), so the §3k/§3L gap was load-bearing after all.
See §4.1: restore a small gap.
**CPU pinning:** aggregator already pinned to a reserved set (`runner.py`); do NOT pin to 1 core
(its MQTT thread shares CPU with the FL loop). The loop is latency-bound, not CPU-bound, so
pinning is second-order — the sleep removal is the win.

### Older state (pre-§3j, stale — kept for the dead-ends)

**What is NOT the cause (verified Jun12, do not re-derive):**
- **NOT GPU contention.** T3 PASSES (overrun=0, gpu_compute ≈ 0.08s); messages arrive
  physically in ~0.1s. Do **not** attribute late commits to contention without new data.
- **NOT a re-dispatch / selection-invariant violation.** A trainer is excluded from
  selection while in `all_selected` (`async_oort.py:1526`); it stays there until its
  update is received & cleaned up, and the §3g code (`top_aggregator.py:393-401`) keeps
  drained-but-uncommitted trainers in-slot. The 90s `SEND_TIMEOUT` (which could free an
  in-flight trainer) **fires 0× in both modes**. Invariant holds and matches sim≈real.
- **`commit_gap_s` is a red herring** for staleness: it is `vclock − sct` where `sct =
  sim_send_ts + budget`; past-dated `commit_gap` runs to 2922s yet the actual recorded
  staleness on those commits is only ~23. Track **staleness**, not `commit_gap`.

So the residual is (a) the **§3i leg is a tuned scalar** (`simCompletionLegSeconds=1.6`)
that inflates the staleness baseline, and (b) a **heavier-than-real past-dated tail**.
The advance gap (pre-leg sim 3.68 vs real 4.32) is sim **over-overlap**: sim re-dispatches
a finished trainer with ~zero latency, so completions bunch; real has real queue_wait +
re-selection latency between finish and next dispatch, spreading them. **Open: model that
real latency principledly (not the leg scalar), and characterise the past-dated tail —
why 8% of sim commits land out of order when arrivals are ~0.1s.**

### Changelog (done / tried-and-failed — do not repeat)

| # | change | result |
|---|---|---|
| §3a | overhead 0.50→0.16 | 0.50 over-charged (clock = pure overhead ramp, staleness 9.3, rounds −40%); 0.16 then under-charged |
| §3b | overhead 0.16→0.315 | advance 2.98→4.95 matched real 4.52; staleness still 7.19 |
| §3c | **overhead → 0** (`simCommitOverheadSeconds=0`) | clock now TRACKS completions (`vclock=max(vclock,sct)`); overhead had been 98.8% of the clock. **KEPT.** |
| §3d | prediction gate (wait for early in-flight) | **inert** — predictor learned contention-inflated `SIM_ROUND_DURATION`; never fired |
| §3e | gate predictor → `TRAINING_BUDGET_S` | gate fires (`gate_holds`>0) but staleness still drifts |
| §3f | `version_at(sct)` relabel band-aid | **REVERTED** — felix `fedbuff` consumes staleness for `alpha` (40% of rate); relabel only rewrites telemetry. Proved root cause = commit ORDER (run 150131: advance+throughput matched, staleness still 7.19). |
| §3g | **probe the LIVE in-flight set** (not the stale `recv_ends` snapshot) | **WORKS by reordering** — `recv_fifo` pops min-`sct` buffered msg first → staleness 7.19→3.57 PASS, acc/loss PASS. **KEPT.** |
| §3i | **`simCompletionLegSeconds=1.6s`** added to `sct` only (not compute/budget) | advance 3.68→4.00 (mean PASS, throughput/terminal PASS) but **staleness 3.57→4.09 FAIL**. Leg is a **tuned scalar** (calibrated to real W−compute) → models real latency but as a fudge; raised the staleness baseline. UNDER REVIEW — replace with a principled latency model or back out (see §4.1). |
| §3k | **split the 1.6s leg into pre-commit holding (0.6) + post-commit re-dispatch gap (1.0)** | **PARTIAL on run 175309: staleness 3.45→3.16 (KS 0.078→0.042) BUT advance 4.24→4.03, KS 0.172→0.208 FAIL.** The gap was implemented as pure availability-exclusion → with a 300-trainer pool the selector **refills the freed slot from idle trainers**, so the gap is INERT on throughput (computing concurrency pinned at c=30, `inflight_tracked`≈30). Effective cycle collapses to holding → advance low, staleness ≈ F/g. Corrected by §3L. |
| §3L | **cooling trainers hold a concurrency slot (no idle-pool refill)** | **implemented; inert in the Jun13 run because §3m set the gap to 0.** `extra = max(0, c − in_flight − cooling_count)` in `async_oort._handle_send_state`; aggregator stamps `channel.properties["sim_cooling_count"] = len(_cooling)` in `_distribute_weights`. Predicted advance ≈4.4 / staleness ≈2.77 **only with a non-zero gap** — the Jun13 run ran gap=0 (§3m) and got advance 3.68 / staleness 4.75, so §3L never engaged. §4.1: re-test §3L with the gap restored. Guarded by `TestCoolingHoldsConcurrency`. |
| §3j | **drain by physical READINESS, not predicted completion** (`_sim_recv_min`) | **VALIDATED on run 161419: staleness 4.09→3.451 PASS** (KS 0.078, tail collapsed), advance KS 0.218→0.172 PASS, terminal/commits/acc/loss PASS. §3g drained in-flight ends only within `buffered_min + slack`, so a SLOW trainer (budget 38-56s, far-future `exp`) was excluded even though its message had physically arrived (wall_lag ~0.1s) → it sat undrained until the clock passed its `sct` → committed past-dated, staleness 33 vs real ~9 (same budget). Tell: tail commits have **residence≈0** (NOT buffer-resident) but **gap≈45**, corr(res,gap)=−0.27. Fix: also admit any in-flight end with a non-empty rxq (`_sim_end_has_ready_msg`), so slow trainers buffer as FUTURES and commit in `sct` order. recv_fifo on a ready end returns immediately → no added blocking. Guarded by `test_drains_ready_inflight_above_ceiling`. |

### §3k  Residual staleness gap → post-commit re-dispatch gap (Jun12)

After §3j, staleness PASSES but sim still sits 0.6 high at the mean (3.45 vs 2.81) and
wider at the tail (p99 19 vs 9). Root-caused from the real LAG_DECOMP, not tuned away.

**The model (verified on real):** in steady state `staleness = holding / advance`, where
`holding` = send→commit time. Real check: holding 11.79s / advance 4.32 = 2.73 ≈ measured
2.81 ✓. Equivalently `staleness = (L/C)·(F/g)` with cycle `C`, in-flight `F`, agg_goal `g`.
`F/g` is matched (in_flight 30.8 both → 3.08).

**The bug:** the §3i leg put the WHOLE real post-compute cycle (1.6s) *before* the commit
(in `sct`). But the real cycle splits into a **pre-commit holding** part and a **post-commit
re-dispatch** part, and only the former counts toward staleness. Measured real decomposition
(LAG_DECOMP, full run): holding = compute 11.71 + queue_wait 0.61 + delivery 0.06 + post/mqtt
0.03 = **11.79s**, then a **~1.0–1.15s post-commit** re-dispatch leg (re-selection + model
push). So real L/C = 11.79/12.94 = **0.91** → staleness 0.91·3.08 = **2.81** ✓. Sim folded
everything into `sct` (instant re-dispatch) → L/C ≈ 1.0 → 3.08 + tail = **3.45**.

**The fix (principled, cycle-preserving):** split the 1.6 leg:
- `simCompletionLegSeconds` 1.6 → **0.6** (pre-commit holding = real queue_wait+delivery; in `sct`).
- `simRedispatchGapSeconds` 0 → **1.0** (post-commit; aggregator-side). A just-committed end is
  held out of selection until `vclock ≥ sct + gap` (`_sim_cooldown_until`, consumed in
  `_distribute_weights` via `set_curr_unavailable_trainers`), so it returns with a FRESHER
  model_version. The selector refills to keep c=30 in-flight, so the gap spaces completions
  (raises advance, de-bunches overlap) WITHOUT counting toward the committed update's staleness.

Total cycle compute+0.6+1.0 = compute+1.6 is **unchanged**, so advance/throughput/terminal stay
matched (they currently PASS); predicted staleness → (compute+0.6)/advance ≈ **2.84** vs real
2.81. Not a relabel (§3f dead-end): the update genuinely commits earlier (smaller `sct`), so the
real number `fedbuff` consumes drops. Guarded by `TestRedispatchGap` in
`tests/mode/test_async_sim_ordering.py` (3 tests: cooldown recorded at sct+gap; off-by-default
inert; commit order/clock unchanged). **Status (Jun13):** not exercised — §3m turned the gap off
(`simRedispatchGapSeconds=0`), and that run confirms the gap is needed (sim over-overlaps without
it: advance 3.68 vs 4.14, staleness 4.75 vs 2.80). §4.1: turn the gap back on and re-run.

### §5  Checker category-errors (Jun12) — the real "final disparity"

Once §3j made the sim **dynamics** match, the residual FAILs were all the checker
enforcing *exact identity* on quantities a stochastic / in-memory simulator cannot
reproduce. Diagnostic tell: each was byte-identical across sim 120543→161419 despite a
large staleness change → not downstream of the dynamics. Fixes (in `scripts/parity/`):

| # | check | was | fix | why it's correct |
|---|---|---|---|---|
| §5a | **P1 `aggregation_sequence`** | hard DIST FAIL (exact_set_match 0.0) | **gate to WARN for stochastic selectors** (mirror S1's `DETERMINISTIC_SELECTORS` idiom) | The ladder spec (§2 Stage 6) already says P1 is "gated for stochastic"; the impl never did. AsyncOort picks different rounds across modes → exact per-round set identity is unattainable. S2 participation (PASS) is the enforced invariant. |
| §5b | **F1-3 `utility`** | hard FAIL (max_KS **1.0**, per-trainer) | **enforce the POOLED utility KS** (0.037 ✓); per-trainer max_KS becomes a *gated* diagnostic over trainers with ≥10 commits both modes | The 43 KS=1.0 trainers were all 1–6-sample (KS=1.0 is mechanical for n≤2); their means were identical (~2.2–2.4). Pooled distribution is the path-independent fidelity measure; per-trainer identity is path-dependent for a streaming selector. |
| §5c | **`phase_mqtt_fetch`** | enforced MECHANISM/DIST FAIL (real 57.5s vs sim 24.8s) | **reclassify to DIAG** (WARN) with a note | Pure network-I/O wall time; sim serves weights from an in-mem cache and folds the trainer cycle into budget+leg, so this phase is deliberately *not* part of the virtual clock (§3h). Comparing it is apples-to-oranges. gpu_compute + other modeled phases stay enforced DIST. |

Result: felix **ALL CHECKS PASSED**. Guard: `test_ladder.py` clean-pair + phase-split
tests still pass (per-phase `ok` unchanged; only tiers/gating moved). These are
append-only checker corrections, not tolerance loosening — a future *deterministic*
selector still gets exact-set + per-trainer enforcement via `DETERMINISTIC_SELECTORS`.

**Dead ends — do not retry:** overhead > 0 on the clock (§3a/b — masks & drifts);
prediction-only gates with no real blocking (§3d/e — never wait); `version_at(sct)`
relabel (§3f — fedbuff uses the real number); adding `mqtt_fetch` (57s) to `sct`
(§3h — not version-relevant, would inflate staleness ~6×). `phase_mqtt_fetch` FAIL is
a benign wall-time artifact (in-mem cache is faster), orthogonal to the virtual clock
— NOT chased. A2 eligibility point-mass handled (means match + CV<0.01 rescue).

## §4  Plan: close the remaining parity gaps (proposed Jun13)

Three fronts, ranked by ROI, each tagged with what already worked/failed so we don't re-walk
dead-ends. **Overriding principle (§4.0):** parity ≠ goal. The goal is a *correct* simulator.
Real is the reference **only after it is itself shown correct** — never tune sim to match a wrong
real, and never add a scalar fudge where a mechanism is called for.

### ⏸ SESSION CHECKPOINT — resume here (Jun13b, real-validator made sound)

**Done this session (working tree, NOT yet run/committed to git):**
- **§4.0 validate_real — REWRITTEN ON A SOUND BASIS.** The first cut was *unsound*: it checked
  `effective_c` (absent in **all** selection events → concurrency check was vacuous, printed `[OK]`
  having compared nothing) and cleared in-flight on `contributing_trainers` (which omits
  overcommit-discarded completions → every later legit re-selection miscounted as a re-dispatch
  breach: the bogus oort 4549 / refl 26988). Root telemetry facts: aggregator `in_flight` **==
  num_chosen** (per-round selected count, NOT concurrency); `trainer_round.ts` is emitted *before*
  the real-mode budget sleep (≈ task-start, not completion). **Neither agg field can measure real
  concurrency or re-dispatch.** Fix: new **`task_send`** trainer event (fired from `_send_weights`,
  after the sleep + upload) carrying `[wall_recv_ts, wall_send_ts]` — the true busy window.
  `validate_real` now computes peak/mean concurrency = interval overlap, and a double-dispatch = two
  overlapping intervals for one trainer (serial loop ⇒ exactly 0 unless telemetry is corrupt).
  **Open §4.0 diagnostic RESOLVED:** the high redispatch count was a bookkeeping artifact, NOT a real
  breach. Guarded by `scripts/parity/test_validate_real.py` (11 tests). No graceful fallback — assumes
  the run carries `task_send` (every run from this code onward does); concurrency populates on re-run.
- **Speedup / logging readout — NEW (§4.4).** `validate_real` now prints a speedup/headroom block
  (`intrinsic_speedup_x = virtual/wall`, `compute_floor_wall_s = GPU work / mean concurrency`,
  `pct_of_floor`, `ceiling_speedup_x`). On old oort sim: intrinsic 14.2× (matches table); floor needs
  `task_send` concurrency so it lands next run. **Logging is the headroom:** the aggregator `.log` is
  ~220k lines/run (oort 777 rounds), 105k of it `[RECV_FIFO]` hot-path traces — **the parity checker
  reads only the JSONL telemetry, never the `.log`, so this spam serves zero parity purpose.** Kept
  AS-IS this run (consistent real baseline for the felix calibration); demote it in the speedup phase
  for a clean before/after (§4.4).
- **§4.1 felix gap — STAGED (config), AWAITING SIM RUN.** `_metadata/baselines.yaml` felix
  `simRedispatchGapSeconds 0.0 → 1.0`. Mechanism in code + test-guarded. **NEXT: user runs felix
  sim**, then re-check K3b/K3/K2/staleness/terminal. If advance overshoots 4.14, set gap to real's
  *measured* LAG_DECOMP leg (not whatever passes K3b).
- **Knob retirement — `sendStaggerSeconds` REMOVED.** Off in every baseline (default 0, dead `>0`
  branch) and its rationale (MQTT send-burst drops) is ruled out (drops 0 both directions). Dropped
  the config field + the dead sleep branch + `_stag_acc` accounting in both top_aggregators'
  `[DISTRIBUTE_TIMING]`. Behavioral no-op (knob was always 0). The other sim knobs (leg/overhead/gap/
  settle) are all in active use across baselines — NOT inert, kept.
- **§4.2 speed/budget — LOCALIZED, NO CODE; localization DATA already captured.** Per-trainer assigned
  budget is **byte-identical** real vs sim; the T2/P3 FAILs are **commit-weighted artifacts of a
  divergent selection mix** (oort real selects slow trainers more: weighted budget 16.4 vs sim 12.3).
  NOT an input-model bug — it's Stage-3 selection, same root as refl/feddance participation.
  **Confirmed the selector telemetry is sufficient to localize it — no instrumentation gap:** the
  `selection` event's `per_trainer` dict already records each candidate's `utility`, `speed_s`,
  `selected`, plus selector score terms (oort `believed_I`/`temporal`/`system_util`, feddance
  `V/I/A/U`) in BOTH modes. NEXT (post-run): compare the real vs sim per-candidate scores at matched
  rounds to see why the chosen mix diverges. Unblocked — §4.0 shows real selection is sound.
- **§4.3 participation — DONE (checker).** `checks.py::participation_parity` scale-free (per-trainer
  **share** KS, tol 0.2). felix 0.057 ✓, oort 0.184 ✓; refl 0.497 ✗, feddance 0.527 ✗ (**genuine**
  selection-mix divergence — correctly NOT suppressed). `report.py` updated.

**To continue (in order):**
1. **Correctness run (logging AS-IS):** 1h real+sim per baseline. `task_send` lands in fresh runs →
   `parity_check.py --validate-real <real_dir>` validates concurrency soundly + prints speedup/floor;
   `--batch` re-checks felix parity against the verification tracker.
2. §4.2: localize the selection-mix divergence from the run's `per_trainer` scores (real vs sim at
   matched rounds). Real is admissible (§4.0 sound), so it's a sim-selection question.
3. **Speedup phase (after correctness locked):** demote the hot-path `.log` traces — `[RECV_FIFO]`
   (channel.py, ~105k lines/run) + per-commit `[MSG_*]` (oort top_aggregator) — to DEBUG. None are read
   by the parity checker. The wall-time delta vs step-1 = the logging overhead; re-tune felix gap if
   real's leg shifts (the §4.1 plan already re-measures real's leg). Compare `pct_of_floor` before/after.

### §4.0  Validate that REAL is correct before treating it as ground truth  [PREREQUISITE]
Before matching sim to real on any axis, prove real obeys its own invariants — otherwise we'd be
fitting sim to a bug. Check each from existing logs + a targeted test; only then is the real number
admissible as the parity target.

| axis | real invariant to confirm | sound signal | status |
|---|---|---|---|
| **concurrency** | computing fraction = compute/cycle; no in-flight pile-up | peak/mean overlap of `task_send` `[wall_recv_ts, wall_send_ts]` intervals (true busy window) | needs a fresh run (event is new); no static `c` to assert `≤ c` — concurrency is emergent, so it's reported not bounded |
| **selection** | chosen ⊆ eligible; no double-select while in-flight | `num_chosen ≤ num_eligible` (agg); double-dispatch = overlapping intervals for one trainer (trainer telemetry) | sound; re-dispatch breach == 0 by construction (serial trainer loop) |
| **aggregation** | agg_goal cycles 1..K; staleness ≥ 0 & matches LAG_DECOMP holding/advance | agg_round agg_goal_count + staleness | sound; real staleness 2.80 = holding 11.59/advance 4.14 ✓ |

**DONE (sound rewrite).** `scripts/parity/validate_real.py` + `parity_check.py --validate-real <dir>`,
guarded by `test_validate_real.py`. The **key correction**: the aggregator's `in_flight` is just
`num_chosen` (not concurrency), `effective_c` is absent, and `trainer_round.ts` is pre-sleep — so the
first cut's concurrency/re-dispatch checks were vacuous/artifactual (the bogus oort 4549/refl 26988
redispatch counts came from clearing in-flight only on `contributing_trainers`, which omits
overcommit-discarded completions). The sound basis is the new **`task_send`** event:
`[wall_recv_ts, wall_send_ts]` brackets the trainer's true busy window (it fires after the real-mode
budget sleep). **Open §4.0 redispatch diagnostic is RESOLVED — 0 real breaches; it was bookkeeping.**
Concurrency now reports honestly: existing (pre-`task_send`) runs get `task_send_present=False` and
do **not** falsely pass. Re-run a baseline to validate its real concurrency.

### §4.1  felix — restore the post-commit redispatch gap  [config; mechanism already in code]
**Problem:** felix's only enforced root is K3b (0.046 s/commit under-charge); it cascades to
throughput (+15% rounds), terminal, commits, and **staleness 2.80→4.75**. One mechanism, six FAILs.
**Worked:** §3j drain-by-readiness (staleness 7.19→3.45); §3k/§3L leg(0.6)+gap(1.0)+slot-hold
(predicted advance 4.4 / staleness 2.77). **Failed:** §3m zeroed the gap on the theory that
speeding real up (settle=0) lets sim match at gap=0 — the Jun13 run **refutes** it (gap=0 → sim
over-overlaps: advance 3.68, staleness 4.75). The gap is a *mechanism* (real has a measured ~1 s
post-commit re-dispatch leg, §3k LAG_DECOMP), not a fudge — so restoring it is principled, **provided
§4.0 confirms real's leg is real and not an artifact.**
**Change (config only — `_metadata/baselines.yaml` felix):** `simRedispatchGapSeconds 0.0 → 1.0`
(keep leg 0.6, overhead 0, settle 0). Mechanism guarded by `TestRedispatchGap` +
`TestCoolingHoldsConcurrency`.
**Expected:** slot-hold engages → computing ≈ c−rate·gap → completions de-bunch → advance ↑~4.1–4.4,
staleness ↓~2.8. **Validation:** one n=300 sim run. The gap value is *calibrated to real's measured
re-dispatch leg* (LAG_DECOMP), not fitted to pass the check — if real's leg ≠ ~1 s on the new
settle=0 runs, set the gap to the measured leg, not to whatever makes K3b green.

### §4.2  oort/refl/feddance — trainer_speed/training_budget input gap  [LOCALIZE FIRST, no hack]
**Problem:** P3 trainer_speed FAILS on all three (KS .125/.153/.197); T2 budget on oort/feddance
(real 16.4 vs sim 12.3; 12.4 vs 11.0) → drives feddance advance K3 KS .581 + the loss curves.
felix's T2 matches (KS .005), proving the model *can* match — so these three have a genuine input
divergence. **This is the clearest place the principle bites:** the speed model is a *shared input*;
if real and sim draw it differently, the bug could be in **either** mode. Do **not** bend sim's draw
to match real until we know which side is wrong.
**LOCALIZED (Jun13, from trainer telemetry — no new run):** per-trainer assigned `training_budget_s`
is **byte-identical** across modes (0370→4.0, 0371→16.0, … both modes; per-trainer diff = 0.00 for
all). The T2/P3 divergence is purely **commit-weighted**: the *pooled* (per-commit) budget differs
(oort real 16.42 vs sim 12.27; feddance 12.36 vs 10.95) only because the **selection mix** differs —
real over-commits slow (high-budget) trainers, sim picks faster ones. **Verdict: NOT an input-model
bug (the speed draw is identical). The root is Stage-3 selection** — the same divergence as the
refl/feddance participation FAIL (§4.3). Do **not** alter sim's (identical) speed draw.
**NEXT:** determine which mode selects correctly. The oort selector scores by utility (loss + system
terms); the mix shift means sim and real score/avail-window trainers differently. Localize at Stage 3
(why does the chosen mix diverge?), gated by §4.0 (is real's selection itself correct? — its
`redispatch_while_inflight` diagnostic is high, resolve that first).

### §4.3  S2 participation — make the metric scale-free  [checker fix]
**Problem:** S2 fails on all four (avg_diff 18.5/11.1/227/17.4 vs fixed warn=10). `avg_diff` is a
**raw count** that grows with run length, so long runs (refl 8587 rounds) fail mechanically even
when the participation *distribution* matches. S3/4 in-flight + means already PASS.
**Why not just gate it (as §5 did for P1/S1):** S2 is *by design* the enforced selection invariant
for stochastic selectors — gating it leaves no enforced selection check. Principled fix: keep it
enforced but **scale-free**.
**Change (`scripts/parity/checks.py::participation_parity`):** replace `avg_diff <= 10` with a
length-independent test — KS on per-trainer counts, or `avg_diff / mean_participation`. Still catches
a genuine *shape* divergence (refl 227 likely persists = a real selection difference to localize, not
to suppress). Add a `test_ladder.py` regression case.
**Validation:** re-run; felix/oort/feddance pass on shape if their distributions truly match; any
remaining FAIL is a real signal, consistent with §4.0.

### §4.4  Speedup / logging overhead  [measure first, then cut — AFTER correctness]
**Principle:** correctness first. Sim fidelity is calibrated against real's *measured* latency, and the
hot-path logging inflates that latency — so we do NOT change logging during the correctness run (keeps
the real reference consistent with the felix §3m/§4.1 calibration). We *measure* the headroom now and
*cut* in a separate phase for a clean before/after.

**Measured (the readout, in `validate_real`):** `intrinsic_speedup_x = virtual_s/wall_s` (oort sim
14.2×, matches the table); `compute_floor_wall_s = Σ gpu_compute_s / mean_concurrency` (the GPU-bound
minimum wall); `pct_of_floor` (100% = at the floor; lower = overhead-bound); `ceiling_speedup_x =
virtual_s/floor` (best achievable). The floor needs `task_send` concurrency, so it populates next run.
Early signal: sim per-commit GPU is ~0.01s but wall is overhead-heavy → sim is **overhead-bound, not
GPU-bound** → large headroom.

**The overhead (not read by the parity checker — JSONL only):** aggregator `.log` ≈ 220k lines/run
(oort, 777 rounds), of which `[RECV_FIFO]` (channel.py `_get_inner`/`recv_fifo`, `logger.info`/`warning`
on every recv poll) is **~105k**, plus per-commit `[MSG_ARRIVAL/PROCESSING/ACCEPTED/SKIP]` and
`[SEND_RECV_LAG]`/`[LAG_DECOMP]` (oort `top_aggregator`) at ~8k each. refl (9054 rounds) is ~10× worse.

**Plan (speedup phase):** demote those hot-path traces to DEBUG (keep `[AGG_ROUND]`, `[DISTRIBUTE_TIMING]`,
and the JSONL events — the parity inputs). Re-run; the wall delta vs the correctness run = the logging
cost, and `pct_of_floor` should jump. Then re-measure real's LAG_DECOMP leg (logging removal lowers it)
and re-tune felix's `simRedispatchGapSeconds` to the new leg. Applies to **both** real and sim, but the
real win matters most — it de-contaminates the very reference sim is matched to.

**Done (one-liners):**
5. **[Jun10] Lazy weight-deserialize, unified.** Trainers always ship `WEIGHTS_BYTES`;
   every up-path read goes through `common.util.materialize_weights` (idempotent,
   eval-safe). Fixed the regression where refl crashed (`UnboundLocalError`) and felix
   silently never aggregated (each baseline has its own recv/handle path). Guarded by
   `tests/mode/test_weights_bytes_roundtrip.py`.
6. **Feddance `trainer_speed_s`.** `syncfl/top_aggregator.py` sets `PROP_ROUND_DURATION`
   from `wall_lag_s` in real mode. Verify the DIST check passes next run.
7. **Log-level cleanup.** Per-commit/recv traces → DEBUG; `[AGG_ROUND]`, `[LAG_DECOMP]`,
   `[SIM_BARRIER]`, staleness/participation summaries stay INFO.
8. **[Jun10] Plotting overhaul** (`analyze_run.py`, `PLOTTING.md`): single-pass log
   parser, `binned_line`, deep-dive subdirs incl. `plots/aggregation/*` (buffer/commit_gap/
   residence/staleness) for watching the felix order fix land.

Telemetry for the felix order fix: `agg_round.commit_gap_s/buf_depth/staleness` (sim;
past-dated commits → 0, staleness → ~3), `[SIM_CLOCK_DIAG]` (`gate_holds`, `barrier_wait_s`,
`pastdated_commits`), `[LAG_DECOMP]` (`queue_wait_s`, both modes), `plots/aggregation/*`.
