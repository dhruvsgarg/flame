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

### Status: **Jun 14 — FULL 14400 s (4 h) runs (n=300, α=0.1)** — the run-length caveat RESOLVED

> **Read first.** These are the full **14400 s** runs the original tables were built for; they
> supersede the Jun-13 2700 s section. The 2700 s caveat predicted that several oort/refl/feddance
> selection/CONTROL checks *passed only because the horizon was too short for the selection-mix to
> accumulate* — **the 4 h runs bear this out**: `--batch` now roots **feddance, oort, and refl all at
> `trainer_speed` (P3)**, the commit-weighted footprint of the same selection-mix / in-flight-residence
> divergence. feddance's "40/40 ALL GREEN" at 2700 s was the horizon artifact — at 4 h it fails
> P3·K3·K2·terminal·commits. Pairs (`--batch` auto): felix sim 010018 / real 033735, refl sim 010048 /
> real 034934, oort sim 075449 / real 081648, feddance sim 081900 / real 083925. `--budget-s 14400`.
> `pytest tests/mode/ + scripts/parity/` = **58 PASS** (no code changed; checker run only).

**`--batch` summary (4 h, 14400 s, agg-goal 10):**

| baseline | rounds r/s | K2 | K3 | P3 | overall | ROOT (banner) | reduced to |
|---|---|---|---|---|---|---|---|
| **felix**    | 3421 / 3454 | ✓ | ✓ | ✓ | FAIL | **staleness** | **ONE miss — staleness** (gap landed) |
| **oort**     |  747 /  748 | ✓ | ✓ | ✗ | FAIL | trainer_speed | in-flight **residence** (quantified) |
| **refl**     | 10158 / 9574 | ✗ | ✓ | ✗ | FAIL | trainer_speed | eligible-**pool** comp + genuine loss |
| **feddance** |  464 /  513 | ✗ | ✗ | ✗ | FAIL | trainer_speed | selection-mix (masked at 2700 s) |

**felix — `simRedispatchGapSeconds 1.0→0.6` LANDED; whole throughput family GREEN, one miss left.**
At 4 h: K3b residual −0.05 (rel .011 ✓), K3 advance .075 ✓ (sim 4.17 ≈ real 4.12 — the 2700 s
over-advance 4.35 is gone), K2 rounds 3454/3421 (rel .013 ✓), terminal .013 ✓, commits .012 ✓, P3 .084 ✓,
participation .03 ✓, loss .016 ✓. **The single FAIL is staleness: sim 3.94 vs real 2.80** (KS shape .086
— close; a mean/tail shift, not a shape divergence). The post-commit gap is **over-loaded** — it sets
*both* advance (K3b) and the past-dated bunching that drives the staleness tail, and they want opposite
values: gap **1.0** → advance 4.35 (K3b ✗) / staleness **2.96 ✓**; gap **0.6** → advance 4.17 (K3b ✓) /
staleness **3.94 ✗**. Lowering the gap to fix advance re-bunched completions and regrew the §3j
past-dated tail (steady-state 12.54/4.17 ≈ 3.0; measured 3.94 = +0.9 tail). **They must be decoupled.**

**oort — in-flight RESIDENCE QUANTIFIED (the `inflight_residence` event populated):**

| metric (oort, per round) | sim | real | read |
|---|---|---|---|
| **`in_flight_after`** (carried stragglers) | **0.09** (median 0) | **2.40** (median 2, max 10) | sim drains to ~0; real holds ~2.4 |
| `residence_rounds` (rounds in-flight) | mean 0.01, max 2 | mean 0.17, max 8 | real stragglers reside longer |
| `carried_over_ages` count | 66 | **1792** | real carries ~27× more across rounds |
| selection_detail `mean_chosen`/`in_flight` | 13.09 | **15.39** | the 2.3 gap **== the 2.40 carried** |

Mid-run sim every round `before=13 → after=0`; real `before=13–16 → after=0–3`. **⚠ SUPERSEDED (see
§4.5):** this `in_flight_after` gap turned out to be a **false signal** — the round-300 log trace showed
sim's slow stragglers live in the persistent `_sim_buffer` (committing late as stale), not in
`selected_ends`, so 0.09-vs-2.40 compares different structures. The selected-vs-pool speed check then
showed oort's **pool already ≈ real** (12.12 vs 11.35) and the divergence is the **selector** picking
~average (12.21) vs real fast (8.81). **oort is a selector-scoring case, not a residence/carryover one**
— it gets §4.8-style localization, not the §4.5 slot fix. (The §4.5 fix applies to **refl**, below.)

**refl — eligible-POOL composition, worse at full horizon; loss now genuinely diverges.** A2b
`eligible_speed` sim pool **12.32 s** vs real **6.43 s** (KS .429, up from .363 at 2700 s — accumulates
as predicted). num_chosen/in_flight and staleness still match → it's the pool's speed mix (sim frees
busy clients back to the pool too early → slow re-enter → sim pool skews slow). participation
matched-count-KS **.316 ✗** (genuine), **convergence_loss now FAILS .1735** (>.15; the 2700 s .092 was
the ~2-eval weak pass — 192 evals at 4 h show the real divergence). This is the genuine `sct`-gated
pool-exclusion case (§4.5) — confirmed by selected-vs-pool speed: refl's *selected* matches real
(6.90/6.30) but its *pool* is full-population (12.41) while real's pool ≈ selected (6.56).

**feddance — the 2700 s "all-green" was the horizon artifact.** At 4 h: P3 .147 ✗ (root), K3 advance
**KS .522** (mean_rel .077 — shape diverges hard while means stay close), K2 rounds 513/464 (sim
**+10.6%**, 28.11 s/round vs real 30.4), terminal/commits +7.4%. **But feddance is NOT the residence
mechanism:** it selects exactly k=10 with **no overcommitment** (in_flight 10/10 both modes, no
`inflight_residence` telemetry). Its root is the **selection mix** — *which* 10 are picked — which drives
both the max-of-K advance **shape** (K3 KS .522) and the commit-weighted speed (P3 .147). See §4.8.

> **The roots are different per baseline (do not over-unify):** **refl** = `sct`-gated **pool composition**
> (selected matches 6.90/6.30, pool 12.41 vs 6.56) → §4.5 pool-exclusion fix (IMPLEMENTED, refl only).
> **oort** = **selector scoring** (pool ≈ real 12.12/11.35, but sim selects ~average 12.21 vs real fast
> 8.81) → §4.8-style utility-score localization, **not** §4.5. **feddance** = no overcommit → **selection
> mix** in the `feddance_I` (loss-utility) term, likely path-dependent (§4.8). **felix** = solved on
> throughput; residual gap↔staleness coupling held for a dedicated pass (§4.6). Only **refl** has a coded
> fix + re-run this round; oort/feddance need score localization first; felix is held.


### Distilled history — what worked / what didn't (Jun 12–14)

**Worked (kept, validated):**
- **felix §3j drain-by-readiness** — admit any in-flight end with a physically-ready msg into the
  reorder buffer so slow trainers commit in `sct` order (staleness 7.19→3.45). `_sim_recv_min`.
- **felix §3g probe the LIVE in-flight set** (recv_fifo pops min-`sct`) — reorders the past-dated tail.
- **felix §3m `realDistributeSettleSeconds=0`** — removes a real-only 2×`sleep(0.1)`/commit so real
  holds ~c computing (advance 4.14, staleness 2.80) instead of slowing sim to a wrong reference.
- **felix §4.1 redispatch gap** — post-commit re-dispatch leg as `simRedispatchGapSeconds`, slot-held
  by §3L cooling (freed slot isn't refilled from idle). `1.0` fixed K3/K2/terminal; **`1.0→0.6` (Jun 14)
  zeroed the over-advance** → throughput family GREEN. Residual: staleness (see Status).
- **Checker corrections (principled, not relaxation):** throughput family → one 5% bar (K2/K8/U2 are the
  same rounds/V quantity); S2 share-KS → **matched-round count-KS** (share folds in the throughput
  offset); A2b `eligible_speed` pool-composition check; P1/S1 gated WARN for stochastic selectors;
  F1-3 enforce **pooled** utility KS (per-trainer KS=1.0 was n<=2 mechanical); `phase_mqtt_fetch` → DIAG
  (in-mem cache wall-time, off the virtual clock). `validate_real` + the `task_send` event prove real is
  admissible (concurrency 28.8/c30, **double-dispatch 0**).

**Didn't work / dead ends (do NOT re-try):**
- overhead>0 on the clock (§3a/b — masks & drifts; clock must `=max(vclock,sct)`, overhead 0).
- prediction-only gates with no real blocking (§3d/e — never fire).
- `version_at(sct)` relabel (§3f — fedbuff consumes the *real* staleness number; relabel is inert).
- adding `mqtt_fetch` (57 s) to `sct` (§3h — not version-relevant, inflates staleness ~6×).
- §3m's theory that settle=0 lets sim match at **gap=0** — refuted (gap=0 → sim over-overlaps:
  advance 3.68, staleness 4.75). The gap is a real mechanism, not a fudge.
- `sendStaggerSeconds` (always 0; MQTT drops ruled out) — REMOVED.
- the bogus "redispatch 4549/26988" breach — a `validate_real` bookkeeping artifact (cleared in-flight
  on `contributing_trainers`, which omits overcommit-discarded completions), NOT a real violation.

**Ruled out (do not re-chase):** GPU contention (T3 overrun=0); SEND_TIMEOUT (fires 0×); `commit_gap`
as a staleness proxy (track staleness, not commit_gap); MQTT drops (0 both directions); the felix
post-compute leg as a "bug" (serial-aggregator scheduling, not network).


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
| A2c `[Jun14c]` | selected-vs-pool speed bias | MECHANISM/DIST | Selector's revealed *speed preference* (`bias=selected−pool`) diverges with the pool matched → **selector-scoring** (oort), vs pool itself diverging → **composition** (A2b, refl) | A2b |
| Sx `[Jun14c]` | selector score-term localize | DIAG | *Which* utility-score term drives a mix split (oort believed_I/temporal/system_util; feddance V/I/A/U) — pinpoints e.g. oort `system_util` | A2b |
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


## §4  Plan — code changes for the next 4 h run (Jun 14)

**Overriding principle (§4.0):** parity ≠ goal; a *correct* simulator is. Real is the reference only
after `validate_real` shows it admissible (done: concurrency 28.8/c30, double-dispatch 0). Never tune
sim to a wrong real; never add a scalar fudge where a mechanism is called for.

**Already landed (config/checker/instrumentation):** §4.0 `validate_real` + `task_send`; §4.1 felix
`simRedispatchGapSeconds 1.0→0.6` (throughput family green); §4.3 participation matched-round count-KS;
A2b `eligible_speed`; oort `inflight_residence` event (now populated — gave the calibration target).

**Rule (from this session): one change per baseline per run; the change is either a strong-candidate FIX
or instrumentation to GET MORE DATA; only re-run baselines we actually change.** The roots are
*different per baseline* (verified from this run's telemetry), so the changes are independent code paths
and can ship together without cross-contamination:

| baseline | stack | root (from data) | one change | type | re-run? |
|---|---|---|---|---|---|
| **felix** | async (async_oort/fedbuff) | gap↔staleness coupling; staleness shifted at every quantile (med 3 vs 2, **p99 28 vs 9**), `pastdated=0` so it's **buffer-depth aging** (buf_depth 26.6), not past-dating | **HOLD** (no confident single knob; see §4.6) | — | NO |
| **oort** | sync (oort/fedavg) | pool already ≈ real (12.12 vs 11.35); **selector picks ~average (12.21) vs real fast (8.81)** → selector-scoring/path issue, NOT pool | localize oort utility scores (like §4.8) | DATA | NO (no §4.5) |
| **refl** | sync (refl_oort/refl) | selected matches (6.90/6.30); busy client re-enters the pool before `sct` → **pool skews slow (A2b 12.41 vs 6.56)** | §4.5 sct-gated pool exclusion | FIX | YES (sim) |
| **feddance** | syncfl (feddance/fedavg) | **no overcommit** (in_flight 10/10) → not residence; selection-mix drives advance **shape** (K3 KS .522) + commit-speed (P3 .147) | §4.8 selector-score localization | DATA | YES |

### §4.5  refl — `sct`-gated pool exclusion (IMPLEMENTED)  [FIX]
**Principle (from the user, Jun 14): an update must become *available to the aggregator* at the time it
would in real — at the trainer's modeled completion `sct`, not at its (instant) physical arrival in sim.**
A trainer that has physically sent but is modeled as still computing (`vclock < sct`) must NOT re-enter
the eligible pool — in real it is busy that whole time. **The data scoped this to refl, not oort** (oort
note below): refl's *selected* speeds already match (sim 6.90 vs real 6.30) but its eligible **pool**
skews slow — sim **12.41 s** (≈ full population) vs real **6.56 s** (≈ selected), because in real the slow
clients are busy/out-of-pool while in sim they re-enter the pool before `sct`. Its in-flight *count* even
matches (53.55/53.65); it is purely *pool composition*.

**Commit *order* is already `sct`-gated** (`SimReorderBuffer.pop_min` commits the K-smallest-`sct` =
real's K-fastest). What is not is the **pool re-entry** of a still-computing trainer.

**Change (IMPLEMENTED, config-gated `simInflightResidence`, default off):** in
`oort/top_aggregator._distribute_weights`, in sim, compute the **still-computing set** =
`self._sim_buffer.pending_after(vclock)` (buffered ends with `sct > vclock`) and add them to the
**`trainer_unavail_list`** passed to `set_curr_unavailable_trainers` — the selector (`refl_oort`:145,
`oort`:443) excludes the unavailable list from candidates, so they stay out of the pool until `vclock ≥
sct`, then commit normally via the buffer. **Crucially via the *unavailable* path, NOT `selected_ends`** —
adding to `selected_ends` would put them in `select()`'s return and **re-dispatch** them (the send loop
weights every returned id), resetting their `sct`. Bounded: released at `vclock ≥ sct` (budget ≤ ~56 s ≈
≤3 rounds), so no "177 accumulate". Files: `config.py` (`sim_inflight_residence`),
`sim/virtual_clock.py` (`SimReorderBuffer.pending_after`), `oort/top_aggregator.py`, `baselines.yaml`
(refl only). **Target:** refl A2b pool 12.41→~6.5 → closes P3/participation/loss. **Guard:**
`test_virtual_clock.py::test_pending_after_*` + `test_sync_sim_ordering.py::TestSimInflightResidence`
(held set = `sct>vclock`, released after; merge is additive; off-by-default inert).

**oort note (NOT §4.5 — selector scoring; LOCALIZED to `system_util`).** oort's eligible **pool already ≈
real** (sim 12.12 vs real 11.35), so pool exclusion is the wrong lever and is left **off** for oort. The
new Stage-3 checks (below) localize it from the *existing* run: **A2c selection-bias** — real picks 2.54 s
faster than its pool (`bias −2.54`) while sim picks at pool-average (`bias +0.09`), pools matched →
selector, not composition; **Sx selector-score** — `believed_I` matches (KS .03) and `temporal` matches
(.01), but **`system_util` diverges (KS .132, sim 0.921 vs real 0.865)**. The Oort score is
`(believed_I + temporal)·system_util`; `system_util` is the speed/staleness penalty. Sim's is **higher**
(less penalty) → sim under-penalizes slow trainers → picks ~average instead of fast. **oort's root is the
`system_util` computation in sim** (`oort_system_utility(round_duration, round_preferred_duration,
alpha)`). Both inputs are duration-derived and differ across modes: **`PROP_ROUND_DURATION` is the MODELED
`sct`-leg in sim but WALL time in real**, and **`round_preferred_duration` is recomputed DYNAMICALLY each
round** (a percentile of candidate durations), so the target itself differs. **Instrumented (Jun14c):**
`oort.py` now emits `round_preferred_duration_s` + `alpha` on the selection event so the next oort sim run
pinpoints whether the `system_util` gap is the *target* (pref) or the *round_duration input*.

### Jun 14c — selection localizers (A2c, Sx) + oort instrumentation [DONE for oort/refl; EXTEND to others]
Turned this session's ad-hoc diagnostics into **standing parity checks** (the growth rule, §1) so any run
auto-localizes a selection divergence — and instrumented oort's missing score input. **All landed,
guarded (`test_ladder.py` +5, 79 parity/sim/mode tests PASS); no run needed (work on existing telemetry):**
- **A2c `selection_speed_bias_parity`** (`checks.py`, MECHANISM/DIST, dep A2b): selected-vs-pool speed,
  per mode, with `bias=selected−pool`. Distinguishes **pool composition** (A2b fails → refl) from
  **selector scoring** (pool matches but bias diverges → oort). Validated on the existing runs: oort bias
  **real −2.54 / sim +0.09** (pools 11.35/12.12 matched) → selector; refl selected matched (KS .098), pool
  **6.43/12.32** → composition.
- **Sx `selector_score_parity`** (DIAG, dep A2b): per-score-term KS over selected candidates (oort
  believed_I/temporal/system_util; feddance V/I/A/U). Localizes the diverging term: **oort → `system_util`
  KS .132**; refl → all terms match. `participation` now also deps on A2c.
- **oort instrumentation:** `round_preferred_duration_s` + `alpha` on the selection event (above).

**Status across baselines (these localizers are general; only oort/refl exercised so far):**
| baseline | A2c / Sx read | done? |
|---|---|---|
| **refl** | pool composition (A2b) → §4.5 fix IMPLEMENTED | ✅ fix in; run validates |
| **oort** | selector scoring → `system_util`; pref/alpha now instrumented | ✅ localized; **fix pending** the round_duration/pref input |
| **feddance** | Sx should localize `feddance_I` (matches the §4.8 manual finding); A2c not yet run on it | ⬜ wire feddance through A2c/Sx; confirm `feddance_*` keys land |
| **felix** | async stack — A2c/Sx untested; its open issue is staleness (§4.6), not selection | ⬜ run A2c/Sx on felix to confirm selection is clean |

> **What the oort+refl sim run is FOR (and isn't):**
> - **refl — VALIDATE the §4.5 fix** (the only coded change). Pass criteria: `[SIM_RESIDENCE]` log fires;
>   A2b pool **12.41 → ~6.5**; A2c selected still matches; P3/participation/convergence_loss flip to PASS.
> - **oort — DIAGNOSTIC only, not a fix run.** No oort behavior changed; the run just *populates the new
>   `round_preferred_duration_s`/`alpha` instrumentation* so we can localize the `system_util` gap to the
>   target vs the round_duration input. (If you'd rather not spend the run, the *existing* oort pair already
>   localizes to `system_util` via Sx — the re-run only adds the pref/alpha breakdown.)
>
> **Is everything instrumented?** **refl: yes** — fully covered (A2b + A2c + Sx + the fix), run validates.
> **oort: now yes** — the one gap (`round_preferred_duration`/`alpha`) is added; after the run, the
> remaining question is purely *which* duration input to align (modeled `sct`-leg vs wall), which the
> breakdown answers. **feddance/felix: not yet** — extend A2c/Sx to them next (table above).

### §4.6  felix — HOLD (no confident single change this run)
Throughput family is GREEN; the only miss is staleness (3.94 vs 2.80). It is a **gap↔staleness coupling**:
larger gap ↑advance AND ↓staleness, so the one knob can't hit advance 4.12 (wants gap≈0.6) and staleness
2.80 (wants gap>1.0) together. `pastdated=0` rules out the §3j tail; the excess is **buffer-depth aging**
(buf_depth 26.6, p99 28 vs 9) — i.e. the cooling/slot-hold dynamics, not a scalar. The decoupling knob
(leg, in `sct`, shifts holding not advance) is too weak (0.6→0 only buys ~0.14 of 1.14). **No one-line
fix; do not re-run felix** until a dedicated investigation of the buffer-aging vs gap is done. Dead ends
that re-break the green family: raise the gap; add a leg scalar; revert to gap 1.0 (re-fails throughput).

### §4.7  (merged into §4.5)
The refl pool fix is §4.5 above; oort is reclassified there as a selector-scoring case (§4.8-style),
not a slot/pool change. Nothing separate here.

### §4.8  feddance — selection-mix localized: same pool, real picks slower  [DATA → analysis done]
feddance has no overcommit (in_flight 10/10), so residence does **not** apply. **Localized from existing
`selection.per_trainer` telemetry (matched first 400 rounds):** the candidate **pool speed matches**
(sim 12.13 s vs real 12.20 s), but **real selects slower trainers (mean 13.32 s) than sim (12.42 s) from
that same pool** — the FedDance selector *ranks the identical pool differently*. That slower-pick mix is
exactly what lengthens real's max-of-K advance (30.4 vs 28.1 s/round, K3 KS .522) and raises commit-speed
(P3 .147). **Term localized (matched 400 rounds, V/I/A/U):** `V` is identical (0.967 both); A/U are tiny
(~0–2). The **`feddance_I` term dominates** (magnitude 82–90) and is where the modes split — real selects
**I=90.0** vs sim **82.6** (pool-I real 30.3 vs sim 28.1). `I` is the path-dependent age term
(`last_engaged_round`): real picks higher-I = staler = slower trainers. **Resolved — `I` is the trainer's last-round training LOSS (Oort
stat_utility), not an age term** (`feddance.py` docstring `I_m`; set via `on_update_received` → `PROP_I`).
That event fires **on commit in both modes** (`syncfl/top_aggregator` calls it identically), and sim does
**real** GPU compute so the loss is drawn from the same process — so the I update point is **not** biased;
the divergence is **path-dependent stochastic accumulation** (real picks higher-loss trainers; over 4 h
the two selection histories drift). **Implication: feddance is likely NOT a sim correctness bug** — it is
the same stochastic-selector class as §5 (P1/utility): per-round / commit-weighted *mix identity* is
unattainable for a chaotic selector, so P3/K3 (commit-weighted EMERGENT outcomes) should be judged
**distributionally**, like participation (matched-count-KS, which feddance PASSES). **One thing left to
rule out:** the `mean_I`/`prev_round_mean_I` normalization buffer accumulates over a *different round
count* at matched time (sim 513 vs real 464) — check it doesn't horizon-bias the I ranking. If clean,
the feddance fix is a **checker reclassification (P3/K3 → DIST for stochastic selectors)**, not a sim
change, and feddance needs **no re-run**.

### Sequencing
1. **felix:** no change, no re-run.
2. **refl (IMPLEMENTED):** §4.5 `sct`-gated pool exclusion, config-gated `simInflightResidence=True` (refl
   only), guarded. `pytest tests/sim/ tests/mode/ scripts/parity/` green under `dg_flame` (76 pass).
3. **oort:** NOT §4.5 (pool already matches). Localize the oort selector utility scores (§4.5 oort note,
   §4.8-style) before any oort change — **not re-run** this round.
4. **feddance:** no code/re-run — first rule out the `mean_I` horizon bias (§4.8); likely a checker
   reclassification, not a sim change.
5. Launch **refl sim only** (real is unchanged — pair against the existing 4 h real `034934`). 2 h reads
   the pool engagement (A2b per-round); 4 h reads the downstream verdicts (P3/participation/loss).
   `parity_check.py --real experiments/<refl_real_034934> --sim experiments/<new_refl_sim> --budget-s
   <runtime>`; compare A2b pool 12.41→~6.5 and watch `[SIM_RESIDENCE]` log lines confirm the hold fires.

**Telemetry to watch:** `inflight_residence` (`in_flight_after`, `residence_rounds`, `carried_over_ages`),
`agg_round.staleness/vclock_now`, `selection.per_trainer.{speed_s,selected}`, A2b pool speed,
`plots/aggregation/*`. Run per-baseline plots: `python ../../../scripts/analysis/analyze_run.py <run>/telemetry`.

**Done (one-liners, kept for reference):** lazy weight-deserialize via `materialize_weights`
(`test_weights_bytes_roundtrip.py`); feddance `trainer_speed_s` from `wall_lag_s`; hot-path logs → DEBUG
(parity reads JSONL only); plotting overhaul (`analyze_run.py`, `PLOTTING.md`, `plots/aggregation/*`).
**Logging headroom (§4.4, after correctness):** aggregator `.log` ~220k lines/run (oort), ~105k
`[RECV_FIFO]`; none read by the checker — demote to DEBUG in a separate phase, measure wall delta +
`pct_of_floor`, then re-tune felix's gap to real's de-contaminated leg.
