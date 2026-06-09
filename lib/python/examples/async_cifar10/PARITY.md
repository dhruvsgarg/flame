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

### Current status (Jun 8 EOD)
- **Speedup done** (felix sim_rate 0.27 -> 2.5x; refl -> 1.7x): removed recv-poll,
  distribute stagger, per-send model re-serialize, disk-backed cache; eval/
  checkpoint -> 50, checkpoint off the critical path. All shared (help real+sim),
  fidelity-neutral, in the shared base -> cover all 6 baselines.
- **Floor reached**: remaining sim wall is the necessary per-commit 2 MB
  deserialize + optimizer aggregation.
- **Deferred to final fine-tune** (after the real baseline stops moving):
  re-tune `sim_commit_overhead_s` to the new real per-round advance (recompute
  from `overhead_residual`); then chase DIST-tier parity (felix staleness 2x,
  refl/feddance selection drift).

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

## §3  Implementation plan

Three coordinated changes. No behavior change to existing check *logic* except
the splits noted; this is mostly added structure + new checks.

**a) `checks.py` — dependency metadata + verdict engine**
- Give every check a stable registry entry: `id`, `stage` (0–9), `role`
  (CONTROL/MECHANISM/EMERGENT/DIAG), `tier` (existing), and `deps` (list of
  upstream ids). A small module-level dict keyed by the result name is enough;
  the check functions stay as-is and just gain a registry row.
- Rewrite `overall_verdict` to consume the registry: compute the set of failed
  checks, then for each enforced FAIL determine whether any transitive upstream
  also failed. Lowest stage with an enforced FAIL and a fully-passing upstream
  chain = ROOT-CAUSE; the rest with a failed upstream = DOWNSTREAM. Return
  `(passed, root_causes, downstream, warnings)`.

**b) New / split checks** (crisp hints, no code)
- `TC1` field-coverage matrix: iterate a declared list of (field → which event
  type → which modes expect it); emit per-field presence counts and density.
  FAIL when an expected field is absent in a mode. Subsumes K10's note.
- `K3a` modeled-compute advance: per round, compute the speed order-statistic the
  close formula predicts (async: K-th smallest of the in-flight speeds; sync:
  max of the K committed speeds) and compare to sim Δvclock. KS + mean-rel.
- `K3b` overhead residual: per round, `real_advance − sim_advance`; report mean,
  distribution, and implied `per_commit_overhead = mean/agg_goal`. PASS when ≈ 0.
- `A3` trace time-base: per trainer, map first/last availability transition
  through each mode's clock and compare; FAIL on systematic offset.
- `A4` duty-cycle: per trainer, fraction-of-time-available; KS across trainers
  (requires avail_change telemetry — note the dependency if absent → SKIP).
- Split `trainer_phase` into `T_pre/T_w2g/T_gpu/T_mqtt/T_w2r/T_post`, each its
  own DIST KS result; keep the combined table for display.
- `T2` training_budget_s distribution KS (control).
- `C2` loss curve: add a `loss_tol` branch to convergence and emit as a separate
  named result so it asserts independently of accuracy.

**c) `report.py` — stage-grouped output + root-cause header**
- Regroup `_SECTIONS` by STAGE 0–9 (not the old 0–8 symptom sections); add a
  role tag column next to the tier tag so CONTROL/MECHANISM/EMERGENT is visible
  per line.
- Print a top banner: `ROOT-CAUSE: <id> (stage N) — <one-line isolates>` plus
  `N downstream failures suppressed: [...]`. The full per-check table still
  prints below, with downstream FAILs rendered `[~~] DOWNSTREAM` instead of
  `[XX] FAIL` so the eye goes to the root.
- Roll-up table (`--batch`): add a `ROOT` column naming the lowest broken rung
  per baseline.

**d) tests** — one assertion per check id; add fixtures that inject a single
mechanism fault (e.g. zero overhead, wrong trace time-base) and assert the
verdict names exactly that rung as ROOT-CAUSE with the rest demoted to downstream.
This is the regression net that makes the ladder self-reinforcing.

---

## §4  Current status  *(last run: 2026-06-08, Jun7 runs — mapped onto rungs)*

Run dirs:
- felix_real: `run_20260607_135610_dbg_felix_n300_alpha0.1_syn0_stream_real`
- felix_sim:  `run_20260607_105426_dbg_felix_n300_alpha0.1_syn0_stream_sim`
- refl_real:  `run_20260607_135535_dbg_refl_n300_alpha0.1_syn0_stream_real`
- refl_sim:   `run_20260607_105421_dbg_refl_n300_alpha0.1_syn0_stream_sim`

Config: n=300, alpha=0.1, syn_0, agg_goal=10, 3h budget.
Felix: c=30 (async_oort + fedbuff). REFL: aggr_num=10 1.3x overcommit, stale_update=5.

**FELIX — root cause at Stage 1 (clock).** Once the ladder is built, the expected
verdict is: P3 PASS (control: speed model identical, KS=0.093) → K3a PASS
(formula right) → **K3b FAIL (root: overhead residual ≈ 5.8s/round = 0.58s/commit
missing)** → everything above demoted to downstream: K3/K2 (2.17 vs 7.97 s/round),
K8/U2 (1328 vs 364 rounds at V), U3 staleness (10.32 vs 2.53), F1-3, C1. Nine
loud FAILs collapse to one root.

**REFL — two independent roots.** Stage 0: **K10/TC1 FAIL** (0/4777 sim events
carry vclock_now → all Stage-1/8 clock checks SKIP). Stage 2: **A3 FAIL** (trace
time-base: sim=vclock vs real=wall) driving A2 (KS=0.384), S2 (avg_diff=165), P1,
F1-3 as downstream. P3 KS=0.268 is the one remaining independent Stage-1 control gap.

| Rung | Check | FELIX | REFL |
|---|---|---|---|
| S0 | TC1/K10 vclock present | PASS | **FAIL (root)** |
| S1 | P3 speed (control) | PASS | FAIL (1.15s gap) |
| S1 | K3a formula | n/a (NEW) | n/a |
| S1 | **K3b overhead (root)** | **FAIL (root)** | SKIP (S0) |
| S1 | K3/K2 throughput | FAIL (downstream) | SKIP (S0) |
| S1 | K4 overlap | WARN | SKIP (S0) |
| S2 | A1 composition | PASS | PASS |
| S2 | A2 eligibility | PASS | **FAIL** |
| S2 | A3 time-base | n/a (NEW) | **FAIL (root) expected** |
| S3 | S3/4 selection_detail | FAIL (downstream) | PASS |
| S3 | S2 participation | PASS | FAIL (downstream) |
| S5 | U5 inter-arrival | WARN | WARN |
| S6 | U3 staleness | FAIL (downstream) | PASS |
| S6 | P1 aggregation seq | FAIL (stochastic) | FAIL (downstream) |
| S7 | F1-3 utility | FAIL (downstream) | FAIL (downstream) |
| S8 | K8 terminal_state | FAIL (downstream) | SKIP (S0) |
| S8 | U2 total_commits | FAIL (downstream) | SKIP (S0) |
| S8 | C1 convergence | FAIL (downstream) | PASS |

---

## §5  Open issues  *(now named by rung)*

### ✅ [ROOT · Stage 1 · FELIX] Overhead residual not modeled — K3b  *(IMPLEMENTED)*
**Shipped:** new hyperparameter `sim_commit_overhead_s` (YAML alias
`simCommitOverheadSeconds`, default 0.0). Charged once per committed update on
the virtual clock via the shared `TopAggregator._advance_sim_clock` helper
(`flame/mode/horizontal/syncfl/top_aggregator.py`), used by the asyncfl, oort,
and syncfl commit paths. Serialized model: K clustered commits add ~K·overhead;
spread commits add ~overhead (no-op in real mode). Felix value set to **0.58**
in `metadata/baselines.yaml` (= (7.97−2.17)/10 from Jun7). Unit tests:
`tests/mode/test_sim_commit_overhead.py`.
**Verifies via:** re-run parity → K3b residual → 0, then K3/K2/K8/U2/U3/C1/F1-3
clear as downstream. Re-tune the value if the residual moves.

### ✅ [ROOT · Stage 0 · REFL] vclock_now absent in sync path — K10/TC1  *(IMPLEMENTED)*
**Shipped:** `build_agg_round(..., extra={"vclock_now": self._vclock.now if
self.simulated else None})` added to both the oort
(`flame/mode/horizontal/oort/top_aggregator.py`) and syncfl
(`flame/mode/horizontal/syncfl/top_aggregator.py`) telemetry-emit sites, matching
the asyncfl path. The sync vclock already advances per commit (now also charging
`sim_commit_overhead_s`). **Verifies via:** re-run → TC1 coverage 100%,
K1/K2/K3/K7/K8/U2 become runnable for REFL.

### [ROOT · Stage 2 · REFL] Availability trace time-base — A3
**Fix in:** `main_oort_sync_agg.py` availability lookup. Index the syn_0 trace by
the same basis in both modes (or by FL round). **Sequence:** fix the two roots
above first, re-run, then confirm A3; A2/S2/P1 should follow.

### ✅ [Stage 1 · REFL] trainer_speed gap — P3 (1.15s; real 6.29 vs sim 5.14)  *(IMPLEMENTED — same knob)*
Addressed by the same `sim_commit_overhead_s` mechanism; refl value set to
**0.115** in `metadata/baselines.yaml` (Jun7-derived ~0.115s/commit). Re-run and
re-check P3; tighten `trainer_speed_parity` ks_tol to 0.05 once it lands.

### [Stage 3 · FELIX] num_chosen 7.9% off — S3/4
Mild downstream of the clock under-charge (in_flight builds up differently).
Expect it to clear after K3b; re-run to confirm before treating as independent.

### Backlog — checks to tighten once roots are fixed
| Tighten | Why | Where |
|---|---|---|
| K2 tol → 5% | sharper throughput gate | `throughput_parity` tol_rel |
| P3 ks_tol → 0.05 | speed should match tightly post-fix | `trainer_speed_parity` |
| A4 duty-cycle | per-trainer on/off windows | new check; needs avail_change |
| K3a/K3b enforced | formula vs overhead split | new Stage-1 checks |

---

## §6  Simulator wall-speedup — decouple sim-time ordering from wall-waiting

**Status: IMPLEMENTED (validate wall-speedup on next run).** Scope: felix (asyncfl)
+ refl (oort/syncfl). Goal: make sim mode actually faster than real *without
changing any logical result* (commit order, staleness, throughput, vclock,
telemetry all byte-identical — only wall-clock drops).

**Shipped:** completion-barrier set-drain in the three sim recv helpers
(`asyncfl._sim_recv_min`, `oort._oort_sim_recv`, `syncfl._sync_sim_recv_first_k`):
one event-driven `recv_fifo(set, first_k=len, timeout=grace)` instead of per-end
0.5s polling; shared adaptive grace + `[SIM_BARRIER]` instrumentation in the
syncfl base (`_sim_recv_grace_s`, `_note_sim_fill`); the 0.5s empty-spin removed.
Guarded by `test_async/sync_sim_ordering` (commit-order invariant, unchanged) and
new `test_sim_barrier.py` (asserts the single set-drain across all 3 stacks +
adaptive grace). Measure on next run: `[SIM_BARRIER] barrier_wait_s` should track
`wall_lag_s` (≈0.16s), sim `queue_wait_s` → ~0.

### The problem (measured, Jun7 — `[LAG_DECOMP]` agg log, both modes)
Sim wall ≈ real wall (felix: 10,494s ≈ 10,486s) → **zero speedup**, which defeats
the point of simulating. The decomposition pinpoints why:

| component | real | sim | meaning |
|---|---|---|---|
| `wall_lag_s` (round-trip) | 12.2s | **0.16s** | agg-send → agg-recv |
| `mqtt_lag_s` | 0.02s | 0.02s | trainer-send → agg-receive |
| `compute_s` (modeled) | 12.1s | 12.2s | reported max(gpu,D), *not* slept in sim |
| `queue_wait_s` | 8.25s | **85.7s** | agg-**received** → agg-**processed** |

Key reading: in sim the trainer round-trip is **0.16s** (no sleep — updates
physically arrive almost instantly), but each update then **sits 85s in the
reorder buffer** before being committed. So arrival is NOT the bottleneck — the
**commit/pop rate is**. With ~10 commits/round over 7.9s/round that is **~0.79s
of wall per commit**, which is exactly the `_sim_recv_min` pacing:
- async `_sim_recv_min` calls `recv_fifo([e], 1, timeout=0.5)` **per end**,
  sequentially — each un-arrived probe burns 0.5s, repeated per commit.
- plus `time.sleep(0.5)` on an empty buffer.
- sync/oort use the same 0.5s fill timeout (`*_SIM_RECV_FILL_TIMEOUT_S`).

The updates are **already in the buffer at 0.16s**; the 0.5s pacing meters how
fast we drain them and buys **zero** correctness (everything needed is present).
That metering is the entire ~8s/round.

### The principle
In simulated mode **physical arrival time carries no information** — correctness
depends only on `sim_completion_ts = sim_send_ts + max(gpu, D)`, which the trainer
computes independently of when its message physically lands. Therefore the
aggregator must:
1. order/commit purely by `sim_completion_ts` (it already does — KEEP), and
2. wait in wall-clock **only long enough to have the information for the next
   correct commit** — i.e. until every in-flight update that *could* carry a
   smaller `sim_completion_ts` has been received — **never on a fixed clock**.

The current code violates (2): it waits fixed 0.5s windows regardless of whether
the needed messages already arrived. Since sim trainers don't sleep, the whole
in-flight set lands within a tiny wall window (~MQTT delivery); the fix is to
wait on the **set**, not the clock.

### The fix (both stacks, minimal + local to the 3 recv helpers)
1. **Event-driven set-drain.** Replace per-end polling with a single
   `recv_fifo(in_flight_set, first_k=len(set), timeout=SHORT)` call — `recv_fifo`
   already yields messages FIFO as they arrive across the whole set and returns
   after a short silence. Stop misusing it one-end-at-a-time.
2. **Completion barrier, not a timer.** Commit by `sim_completion_ts` once the
   barrier holds:
   - async (felix): pop min sct; it is safe to commit S once every in-flight end
     is buffered (or provably has `sim_send_ts ≥ S`). Repeat to `agg_goal`,
     refilling `c` after each pop.
   - sync (refl/oort): once all (overcommitted) selected ends are buffered, take
     the `first_k` smallest sct. (sync already drains the set; just make the wait
     barrier-based instead of a fixed 0.5s.)
3. **Remove the 0.5s empty-spin.** Keep only a SHORT bounded grace (the lone
   fallback for genuinely non-responding ends: unavailable/departed) so
   availability-aware traces still terminate. For syn_0 (all available) the
   barrier completes immediately. The grace is sized to *max plausible real
   compute* (e.g. a running max of observed `wall_lag_s` × a factor), NOT a fixed
   0.5s — it never paces the common path, only catches dead ends.
4. **Instrument the barrier (so we can prove it).** Today `queue_wait_s` conflates
   buffer residency with pop pacing. Add a per-commit log of the *actual barrier
   wait* (wall from "drain started" to "all expected in-flight buffered") and the
   buffer depth at pop. This is the metric the fix must drive toward ~`wall_lag_s`
   (≈0.16s) and away from the ~0.79s/commit pacing — measured, not assumed.

### Why correctness/parity is preserved (the argument)
- The commit **sequence** is a pure function of `{sim_send_ts, sim_completion_ts,
  in-flight set}` — none of which this change touches. Same min-sct (async) /
  first-k-smallest-sct (sync) → identical logical output.
- The barrier makes "is the current min final?" **exact** (wait for the actual
  set) instead of **approximate** (wait 0.5s and hope). It is therefore *more*
  correct under MQTT jitter, not less — the old timer could in principle commit
  before a smaller-sct straggler arrived; the barrier cannot.
- vclock advance (+`sim_commit_overhead_s`), staleness, participation, selection,
  stat_utility, and all telemetry derive from the commit sequence → unchanged.
- Cross-round straggler carry (refl overcommitment persistent buffer) is
  preserved; the barrier counts already-buffered carried updates.

### Why this is the right fix (vs alternatives)
- Changes only **wall-pacing**, not logic → near-zero parity risk, and the
  existing ordering tests (`test_sync_sim_ordering`, `test_async_sim_ordering`)
  are the exact regression guard for the invariant we rely on.
- Minimal/local (3 recv helpers); no new simulator architecture.
- Uses the recv API as intended (set-drain) instead of fighting it (per-end poll).
- A full discrete-event rewrite (no broker, jump to next completion) would be
  faster still but is a large change with real parity risk — unjustified when the
  measured bottleneck is purely the poll loop.

### Expected outcome
sim wall/round 8s → ~MQTT-bound (~0.1–0.5s) ⇒ ~10–30× sim speedup; `sim_rate`
flips from ~1 to ≫1 (sim finally earns its name) and the `--sim-wall-ceiling-s`
caveat disappears. Real mode untouched.

### Validation
1. `test_sync_sim_ordering` + `test_async_sim_ordering` pass unchanged
   (commit-order invariant).
2. New test: identical commit sequence + vclock trajectory given the same inputs,
   with materially lower wall — i.e. logic-invariant, wall-variant.
3. Re-run felix+refl real/sim; parity checker shows the SAME vclock/staleness/
   throughput as a pre-fix sim run, only wall drops (K-tier checks unchanged).
4. Direct metric: the new per-commit barrier-wait should fall from ~0.79s to
   ~`wall_lag_s` (≈0.16s), and sim `queue_wait_s` from ~86s to ~0; sim wall
   collapses ~10–30×. A `selected-but-correct` guard: assert the committed
   `sim_completion_ts` sequence is non-decreasing and identical to a pre-fix
   replay on the same telemetry inputs.

### Jun8 re-run findings — barrier worked, distribute stagger was next

The barrier fix landed: sim recv is now ~0.019s/commit (was ~0.79s), total ~89s.
But sim wall only improved ~5x (felix sim_rate 0.27 -> 1.28) because the bottleneck
moved to `_distribute_weights`, which slept a fixed `time.sleep` between every
weight send (async 0.2s/send, sync 0.5s/send, not shortened for sim). On felix
that was 4744 sends x 0.2s = 949s = 45% of sim wall. Same wall-pacing anti-pattern
as the old recv poll; same fix: `sim_send_stagger_s` (default 0.0 in sim, real
keeps 0.5s). Guarded by `system/mqtt_delivery_accounting.pdf` (dispatched vs
received; a growing positive gap = broker drops at stagger=0).

Per-baseline parity after the Jun8 run:
- felix: near parity. throughput/terminal/convergence PASS; overhead retuned
  0.58 -> 0.50 (it slightly over-charged). Remaining: staleness 2x, per_round
  advance KS (shape).
- refl: under-charges (advance sim 1.64 < real 2.91) -> ran 1380 rounds vs real
  824 -> selection drift (faster trainers picked) -> trainer_speed/eligibility/
  participation diverge. Overhead bumped 0.115 -> 0.24 (=0.115 + measured 0.127
  residual) to close the advance gap and the round-count cascade.

Next bottleneck (post-stagger): the remaining ~3.7s/round is the real MQTT publish
of the 2 MB model, re-serialized per send (~15 identical sends/round). The new
`[DISTRIBUTE_TIMING]` log isolates it; the likely fix is to serialize the model
once per round and reuse the payload for all sends.

### Wall-time accounting (felix sim, Jun8: 2115s / 316 rounds = 6.7s/round)

| component | wall | % | status |
|---|---|---|---|
| distribute send-stagger (0.2s x 4744 sends) | 949s | 45% | FIXED (send_stagger_s=0) |
| recv barrier (0.019s x 4661 commits) | 89s | 4% | already fast (barrier fix) |
| distribute settle sleep (0.1s/round) | ~31s | 1.5% | FIXED (gated to real) |
| weights_to_device per send (re-converted ~15x/round) | TBD | TBD | FIXED (hoisted once/round) |
| MQTT publish + cloudpickle of 2MB model, per send | TBD | TBD | OPEN |
| deepcopy(self.weights) + optimizer.do, per commit | TBD | TBD | OPEN |
| _update_weights/distribute, selection, scheduling | TBD | TBD | OPEN |
| **remaining (unmeasured)** | **~1046s** | **~49%** | split by [DISTRIBUTE_TIMING] on re-run |

We account for ~51% explicitly (and fixed the stagger+settle). The other ~49%
is the distribute send-loop + aggregate region; the `[DISTRIBUTE_TIMING]` log
(send_wall_s, excl stagger) splits it on the next run. Principle: only
aggregator-side serialized costs (hidden behind the trainer sleep in real, but
on the critical path in sim) raise sim_rate; target those.

### Next optimization tasks (ranked; status before re-run)

1. **[MEASURE] split the remaining ~49%** — re-run, then
   `grep DISTRIBUTE_TIMING <agg log>`: if send_wall_s dominates -> MQTT/pickle is
   next; else -> aggregate (deepcopy/optimizer). Gates tasks 2/3. *PENDING RE-RUN.*
2. **[OPT] serialize the 2MB model once/round** — channel.send cloudpickles per
   send; pre-serialize once and reuse the payload for all sends. Expected: cuts
   per-send pickle. Risk: channel-API change. *PLANNED (gated on #1=send-loop).*
3. **[OPT] avoid per-commit deepcopy(self.weights)** in optimizer.do — copy once
   per round or use a preallocated buffer. Risk: optimizer semantics. *PLANNED
   (gated on #1=aggregate).*
4. **[INVESTIGATE] _update_weights per distribute** — confirm it isn't redundant
   work each distribute call. *PLANNED.*
5. **[BACKLOG] reduce MQTT payload** — 2MB/model; compression or delta-encoding.
   Bigger change, helps both modes. *BACKLOG.*
6. **[BACKLOG] trainer fetch-pickup latency** — notify_trainer_avail polls avail
   every 1s; may add up to 1s to task pickup. Low priority for syn_0. *BACKLOG.*

Parity-correctness tasks (separate from speedup):
- **felix staleness 2x** (sim 5.06 vs real 2.48) — investigate after speedup lands.
- **felix per_round_advance KS** — distribution shape (means match); constant
  per-commit overhead can't reproduce real's variable spread. Low priority.
- **refl selection drift** — should shrink once the 0.24 overhead aligns round
  counts; re-check trainer_speed/eligibility/participation after re-run.

### Jun8 18:00 re-run: speedup landed; real baseline shifted

Speedup worked: felix sim_rate 1.28 -> 2.49 (wall 2115 -> 1085s); refl 0.95 ->
1.70 (2396 -> 1590s). Distribute 949 -> 54s, barrier ~284s, no mqtt drops.

NEW fidelity regression (expected mid-optimization): removing the stagger from
*real* too sped REAL up (felix advance 7.77 -> 5.68 s/round; real rounds
308 -> 423). So the real reference moved and the fitted overhead is now
mis-tuned -> throughput/terminal/per_round_advance regressed (were passing).
Lesson: the per-commit overhead is a constant fitted to the real baseline, so
**re-tune it LAST**, after all shared-cost optimizations stop moving real.
Pending re-tune (do last): felix 0.50 -> ~0.32, refl 0.24 -> ~0.21.
refl round-count cascade is fixed (860 vs 850; throughput rel 0.43 -> 0.105);
its trainer_speed/eligibility selection drift persists (separate, DIST-tier).

Wall now (felix sim 1085s / 3640 commits = 0.30s/commit): barrier 52ms +
distribute 15ms + ~230ms/commit aggregate. The 230ms is the new target:

1. **[OPT] self.cache is diskcache (disk-backed)** -- `self.cache[end]=tres`
   writes the 2 MB update to disk every commit and the optimizer reads it back.
   Swap to an in-memory dict (only `.reset()` is diskcache-specific; verify
   cache is cleared per round so memory stays bounded). Fidelity-neutral, shared
   cost, on the sim critical path. *TOP NEXT.*
2. **[OPT] per-round eval/checkpoint** -- eval every 10 rounds (real test-set
   forward pass; needed for convergence check, keep) and checkpoint every 10
   (2 MB disk write; analysis-only -> disable in sim). *checkpoint: easy win.*
3. **[FLOOR] per-commit 2 MB deserialize + optimizer.do** -- necessary ML work
   for convergence fidelity; the floor for real-execution-with-vclock.

Order: do shared fidelity-neutral cuts (#1, checkpoint) -> re-run -> re-tune
overhead to the final real baseline -> then chase remaining DIST-tier parity
(felix staleness 2x, refl selection drift).

### Jun8 (post-1800) - exhausted shared aggregator optimizations

All fidelity-neutral, shared (help real + sim), tuned for sim's exposed path:
- **DONE in-memory cache** - self.cache: diskcache -> MemCache (dict with
  iterkeys/pop/reset). Removes per-commit 2 MB disk I/O; optimizers already
  pop after consuming, so memory stays bounded (no leak).
- **DONE serialize-once sends** - the same model goes to every recipient in a
  distribute call; build + `channel.dumps()` once, `channel.send_payload()` per
  end (skips re-pickling 2 MB per send). Big for sync/refl (13-67 sends/call).
- **DONE eval/checkpoint/util_counterfactual -> every 50** (was 10): config
  defaults + OVERNIGHT node1/node2 yamls. Cuts the periodic episodic stalls.
- **DONE checkpoint off critical path** - snapshot state_dict to CPU on the main
  thread, torch.save in a daemon thread. Disk write no longer blocks the round.
- **FLOOR** per-commit 2 MB deserialize + optimizer.do - left as-is (necessary
  ML work; convergence fidelity depends on it).

Deferred to the END (after the next run stabilizes the real baseline):
- **re-tune sim_commit_overhead_s** to the new real per-round advance. The
  constant is fitted to real, which keeps moving as shared costs are removed, so
  this is the last step. From Jun8-1800 (pre these cuts): felix ~0.32,
  refl ~0.21 - recompute from the next run's overhead_residual, do not apply yet.
- **refl selection drift** (trainer_speed/eligibility) - DIST-tier, revisit after.
- **felix staleness 2x** - DIST-tier, revisit after.
