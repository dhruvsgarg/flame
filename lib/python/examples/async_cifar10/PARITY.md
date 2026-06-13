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

### ⚠️ Status: MAJOR UPDATE PENDING after the 4h overnight runs (Jun 12)

All four baselines (felix / oort / refl / feddance, real + sim) are queued for fresh 4h
n=300 runs. The numbers below predate them and will be **rewritten** once the logs land —
treat them as the hypotheses those runs validate, not current truth.

**Done this cycle (code landed):**
- **felix staleness (§3g/§3j):** drain in-flight by physical rxq readiness, not predicted
  completion → past-dated tail collapsed (staleness 7.19→~3.45, KS 0.078).
- **felix advance/concurrency (§3L):** cooling (committed, not-yet-redispatched) ends hold a
  concurrency slot so the idle pool can't refill — makes the redispatch gap actually bite.
- **real speed-up to ~c=30 (§3m):** drop the real-only settle-sleep brake
  (`realDistributeSettleSeconds=0`) so the aggregator is compute-bound; sim then matches at
  c=30 with `simRedispatchGapSeconds=0` (§3L slot-hold goes inert).
- **feddance/fedavg staleness BUG fixed:** base sync aggregator built `TrainResult` without a
  version → staleness reported the round number (the bogus ~183). Now stamps the trained-on
  version. oort/refl unaffected (different aggregator).
- **MQTT drop plot:** rewritten to per-round drops + CDF (flat 0 = healthy); old plot was
  silently dead (dispatch log → DEBUG). Verified drops=0 on real felix.
- **checker category-errors (§5):** P1/utility/phase_mqtt gated for stochastic/in-mem sim.
- **speed-up (Jun10):** in-mem cache, no recv-poll/stagger/re-serialize, lazy weight-deserialize.

**To evaluate after the overnight runs:**
- **felix:** does real hold ~30 computing (queue_wait collapses) at advance ~3.97 / staleness
  ~3.0, and does sim match? Retune `simCompletionLegSeconds` (0.6 placeholder → ~0.1–0.3) vs the
  new real `LAG_DECOMP` if advance KS misses. Watch the slow-trainer far-future-`sct` sawtooth.
- **feddance:** real staleness now meaningful — confirm it's small and real≈sim; revisit the
  advance KS-shape (0.57) and `trainer_speed_s` gap.
- **oort:** refresh the Jun-9 numbers (terminal 10.8%, commits 2.6% — likely noise).
- **refl:** revalidate overhead 0.074 retune + eligibility KS 0.27.

**Ruled out (do not re-chase):** GPU contention (T3 overrun=0); re-dispatch-invariant violation
(SEND_TIMEOUT fires 0×); `commit_gap` as a staleness proxy; MQTT drops (0, both directions);
the felix post-compute leg as a "bug" (it's serial-aggregator scheduling, not network — §3).

**Pre-overnight scorecard (stale — to be replaced):**

| baseline | run | sim_rate | rounds r→s | advance r/s | staleness r/s | acc |
|---|---|---|---|---|---|---|
| feddance | 094917 | 18.4x | 365→370 | 33.85/34.11 | ~~183/185~~ bug-fixed | 0.029 |
| oort | 094917 | 12.0x | 666→701 | 18.49/17.98 | 0/0 | 0.033 |
| refl | 150131 | 2.31x | 2016→1887 | 1.48/1.74 | 3.02/3.00 | 0.034 |
| felix | 100106/175309 | 2.7x | 626→671 | 4.32/4.03 | 2.81/3.16 | 0.018 |

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
staleness ~3.0. Sim then matches at c=30 with `simRedispatchGapSeconds=0` (§3L slot-hold inert).
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
| §3L | **cooling trainers hold a concurrency slot (no idle-pool refill)** | **implemented, awaiting run.** `extra = max(0, c − in_flight − cooling_count)` in `async_oort._handle_send_state`; aggregator stamps `channel.properties["sim_cooling_count"] = len(_cooling)` in `_distribute_weights`. Now computing ≈ c − rate·gap ≈ 27.7 (mirrors real's 30-slot cap that includes ~3 re-dispatching trainers) → predicted advance ≈ 4.4 (real 4.32), staleness ≈ (F/g)·(L/C) = 3.0·(12.3/13.3) = **2.77** (real 2.81). Guarded by `TestCoolingHoldsConcurrency`. |
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
inert; commit order/clock unchanged). **Awaiting an n=300 run to confirm the predicted landing.**

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

## §4  Next tasks (sim-real parity)

1. **[DONE — §3j validated on run 161419] staleness FAIL (4.09 → 3.451 PASS).** Tail
   collapsed exactly as predicted (KS 0.078, all non-negative, past-dated fraction → ~0).
   Felix sim parity is now GREEN end-to-end. Residual checker FAILs were category-errors,
   fixed in §5. The §3i leg scalar can stay (advance mean PASSES at 4.00 vs 4.32); item 1a
   below (replace the leg with a measured latency model) is now purely optional polish.
1a. **[§3k PARTIAL → §3L implemented, awaiting run] advance-KS + the §3i leg scalar.** §3k (leg
   0.6 + gap 1.0) tightened staleness (run 175309: 3.45→3.16) but FAILED advance (KS 0.208) — the
   gap was inert on throughput because the idle pool refilled cooled slots (computing concurrency
   stayed at c=30). §3L holds the cooled slot against concurrency (`extra = max(0, c−in_flight−
   cooling_count)`), so computing drops to ~27.7 like real. Predicted advance ≈4.4, staleness
   ≈2.77. Next n=300 run validates both; if advance KS still >0.2, characterise the slow-trainer
   far-future-`sct` sawtooth (big `vclock_lead_over_buf` spikes in `[SIM_CLOCK_DIAG]`).

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
