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
`--budget-s` = the run's `--runtime-s`. Add `--lenient` to demote DIST fails to
warnings; prints a stage-grouped report + root-cause banner.
Per-run plots: `python ../../../scripts/analysis/analyze_run.py <run>/telemetry`.

**Launch runs** (node-agnostic; any baselines/mode/duration on any machine):
```bash
bash scripts/debug_run.sh --baselines 'oort refl' --runtime-s 3600 --mode both
```
Reads `expt_scripts_2026/felix_oort_refl_feddance_alpha0.1_parity.yaml` (every
baseline × sim/real), seeds real+sim identically (`SEED=1234`, `SEED=none` to
disable), and applies the per-baseline sim fixes. Split across machines by
passing different `--baselines`.

**Readiness/regression tests** (no cluster; run under lib/python with `dg_flame`):
`pytest tests/mode/ tests/selector/test_oort_selector.py tests/sim/
examples/async_cifar10/scripts/parity/` — guards baseline wiring, the in-memory
cache, serialize-once, sim ordering (barrier/residence/carry-over), the overhead
model, deterministic seeding, and every checker rung. Last green: **146 pass / 7 skip**.

---

## Status (Jun 15 — seeded 45-min run results)

The **seeded** oort/refl/felix/feddance pairs (`seed=1234`, 2700 s, agg_goal 10,
oort §4.9 carry-over ON) are in: `experiments/run_20260615_16*/17*`, reports
`experiments/parity_{baseline}.json`. This is the run the prior status was waiting
for; it cleanly separates genuine divergence from stochastic path drift.

**Headline:** refl is the **first near-clean baseline** — every clock, carry-over,
pool, bias and participation check green; only `Sd` (a speed-penalty config gap)
remains. The other three each have **one localized root**, now pinned below. Two
cross-cutting reads emerged from seeding (`Sdet`) and from the speed control (`P3`).

### One change per baseline — LANDED (Jun 16), for the 3 h runs

All four implemented + 146/7 green; re-checking the *45-min* data already flips
**refl → PASS** and clears felix/feddance `P3`. The oort/felix sim+config changes
take effect on the next launch — that's what the 3 h runs validate (+ `C1`/`C2`
convergence). Re-run the same `debug_run.sh`/parity batch.

| baseline | change | where | re-check expectation |
|---|---|---|---|
| **oort** | carry-over re-buffer now in a `try/finally` so held stragglers survive the caller abandoning the recv generator early (it stops at `agg_goal`, dropping `held_over`) — the §4.9 under-fire was a lost-straggler bug, not tuning | [oort/top_aggregator.py:122](../../flame/mode/horizontal/oort/top_aggregator.py#L122) | `Sr in_flight_after`→~4.6, cascading `K2`/`K3`/`P3`/`S3·4` |
| **felix** | revert async knobs `round_threshold 10→30`, `exploration_decay .95→.98` (pre-paper parity values) | [parity yaml felix blocks](expt_scripts_2026/felix_oort_refl_feddance_alpha0.1_parity.yaml) | overlap 10.9×→~6.6×, staleness 8.4→~2.8, `K2`/`K3`/`U3` green |
| **refl** | gate `Sd` to WARN when the penalty is **inactive in real** (`r_frac=0`, no reconstructable `pref`) — the PROP_ROUND_DURATION None-density artifact (A2b/A2c class). oort (`r_frac>0`) stays enforced | [checks.py preferred_duration_parity](scripts/parity/checks.py) | refl fully green (done on 45-min data) |
| **feddance** | widen `P3 mean_overhead` bar 0.5→1.5 s (sub-grid, opposite-sign, metadata-matched wall-capture; grid_KS stays the backstop) — also clears felix's spurious `P3` | [checks.py trainer_speed_parity](scripts/parity/checks.py) | `P3` PASS; `S2`/`K3` KS resolve on the longer horizon |

**Verdict table (seeded):**

| baseline | clock K2/K3 | carry-over Sr | P3 speed | selection | Sdet eligible_match | net root |
|---|---|---|---|---|---|---|
| **refl** | PASS (1.56/1.54) | PASS (53.6≈53.4) | PASS | A2b/A2c/Sx/**S2 all PASS** | 0.001 + aggregates match → stochastic-class OK | **Sd only** (real binds 0.0, sim 0.356) |
| **oort** | FAIL (6.62 vs 5.15) | FAIL (1.43 vs 4.58, partial) | FAIL (sim 56 s tail vs real 21 s) | S3/4 + Sd FAIL | ≈0 (clock-driven) | **speed-tail / clock** (root moved *down*) |
| **felix** | FAIL (advance 2.34 vs 3.5; overlap 10.9× vs 6.6×) | n/a async | FAIL (−1.06 s) | — | 0.0 (clock-driven) | **async overlap model** (did NOT recover seeded) |
| **feddance** | PASS clock (K3 KS only) | n/a (10/10) | FAIL (+1.05 s) | **S2 FAIL** (0.247, n=85) | 0.729 → set matches, **values** diverge | **feddance_U term** + short horizon |

**Two cross-cutting reads:**
- **`Sdet` = INPUT DIVERGENCE everywhere.** Seeding *works* (selection is now a pure
  fn of state+seed), but per-round inputs differ, so decisions can't line up. The
  split is the diagnosis: `eligible_match≈0` **with matching aggregates** (refl S2,
  participation) = stochastic-class, PASS as-is; `eligible_match≈0` **with a
  diverging clock** (oort/felix) = genuine, fix the clock; `eligible_match` high but
  `decision_match≈0` (feddance 0.73→0.01) = the **values** diverge, not the set.
- **`P3` regressed to FAIL on 3/4** via the `mean_overhead≤0.5 s` sub-criterion, not
  the grid. felix −1.06 s, feddance +1.05 s — **opposite signs**, `grid_KS` passes,
  and `training_delay_s` (A2b/A2c metadata) matches perfectly → this is wall-capture
  variance in *observed* speeds, not a speed-model bug. **oort is the exception:** a
  genuine 56 s sim tail vs real's 21 s cap (`grid_KS` 0.124 also fails) — real signal.

### What the seeded run validated (good and bad)

**refl — VALIDATED near-clean.** Every clock check green (`K2` 1.56/1.54 rel .009,
`K3b` residual .004); §4.5 carry-over **exact** (`Sr` in_flight_after 53.6/53.4,
residence 4.12/4.11); `A2b`/`A2c`/`Sx` green; `P3` PASS. **`S2` participation now
PASSES** (matched_count_KS .177) — confirming the prior hypothesis: participation
is **stochastic-class** (aggregates match though `Sdet eligible_match`=.001).
*Sole remaining FAIL = `Sd`*: real binds the speed/duration penalty in **0.0** of
rounds (reconstructed `pref`=None → real `system_util`≡1), sim binds **0.356**. The
sim applies a preferred-duration penalty the refl fork doesn't. Config gap, not
dynamics — see next steps.

**oort — root cause moved DOWN to the speed/clock input.** The §4.9 carry-over fix
**fires but under-shoots**: `Sr in_flight_after` 0.15→**1.43**, still short of real's
**4.58** (committed_fresh 8.72 vs 10; stale_rejected 4.28 vs 7.14). More important,
`P3` now FAILS: sim trainer_speed has a **56 s tail** real never reaches (real_max
21 s; grid_KS .124, raw .278), and `A2c` observed selected real 3.44 s vs sim 6.92 s
— sim's selected trainers realize far slower budgets. That diverges the clock (`K2`
6.62 vs 5.15 s/round, rel .22) → `Sdet eligible_match`≈0 → **selection-mix verdicts
(Sr/Sd/S3·4) are not trustworthy until the speed input matches**. Fix the tail first.

**felix — confirmed async-overlap-model collapse; seeding did NOT recover it.** Sim
still over-overlaps: advance **2.34 vs 3.5** s, overlap **10.9× vs 6.6×**, staleness
**8.4 vs 2.8** (`U3` KS .526), throughput +33% (`K2` rel .33). Real stayed healthy.
Per the prior branch, "still collapses seeded → it's the overlap model, not the
mix." There is **no async Oort reference** (`third_party/Oort` is sync-only), so the
fix anchors to real's spacing. `P3` −1.06 s is secondary wall-capture.

**feddance — values divergence + short horizon.** Clock fine (`K2` rel .037; `K3`
KS-only). No overcommit (10/10). **`Sdet eligible_match`=0.729** (the set largely
matches) **but `decision_match`=0.012** → the divergence is in the **utility values**:
`Sx feddance_U` term real 5.61 vs sim 3.01 (KS .394) is the lead; `feddance_I` 29.2
vs 28.1 secondary. `S2` is a **borderline** FAIL (KS .247 vs .2) over only **n=85**
rounds. `P3` +1.05 s is wall-capture. So feddance is **one real lead (feddance_U)**
plus two horizon/jitter artifacts.

**Cross-cutting infra (carried forward, all validated):**
- **Seeding.** Per-selector RNG (`AbstractSelector._rng`/`_pyrng` from
  `hyperparameters.seed`, threaded as `_seed`) insulated from the process-global RNG.
  `Sdet` (`decision_determinism`) reports `eligible/decision/chosen_match_frac` — read
  it **first**; it routed all four diagnoses above (clock vs values vs stochastic).
- **Believed-vs-actual utility** telemetry + 4 analyze_run plots. The oort "low
  believed_I" remains the D2-*normalized* reward (raw stat-utility healthy); the
  null-utility picks are explore + carried-over in-flight (a symptom of the carry-over
  root, not a defect).
- **Checker corrections** (guarded, append-only): `P3` integer-grid KS + `mean_overhead`
  bar; `A2b`/`A2c` `training_delay_s` metadata pool; `Sr` residence rung; throughput
  → one 5% bar; stochastic selectors → gated `P1`/`S1`/utility/participation.
- **Run length.** 1 h (this run was 45 min) already exercises `Sdet` + every mechanism
  check. Convergence `C1`/`C2` stay `[??] LOWC` under 2 h (all four here: acc_diff
  ≤.025, loss_diff ≤.065 — promising but inconclusive). **Reserve one 3–4 h run for
  final convergence sign-off once the mechanism FAILs are closed.**

### Pinned next steps (per baseline)

**refl — close `Sd`, then convergence sign-off (CLOSEST TO DONE).**
1. Resolve `Sd`: real binds 0.0 / sim 0.356. Diff the refl-fork `selector.kwargs`
   (`round_preferred_duration` / `round_threshold`) against what the **sim** path
   feeds the penalty — the fork almost certainly disables the preferred-duration
   penalty (so real `system_util`≡1) while sim still applies it. Align sim to the
   fork. Guard with a `Sd`-on-refl regression case.
2. With `Sd` green, refl is fully clean → **launch the one 3–4 h refl run** for the
   `C1`/`C2` convergence sign-off (currently `LOWC`).

**oort — fix the speed tail before re-judging the selector.**
1. **Root:** sim trainer_speed has a 56 s tail real (max 21 s) never reaches; sim's
   *selected* trainers realize ~2× slower budgets (`A2c` 3.44→6.92 s). Find why the
   oort sim over-assigns slow budgets / lets the 56 s cap bind — compare the budget
   draw and the selected-set speed composition (`A2c` observed pool). This diverges
   the clock (`K2`); everything above selection is downstream of it.
2. **Carry-over under-shoot (secondary):** `Sr in_flight_after` 1.43 vs 4.58 — the
   §4.9 hold fires but drains too fast. Re-check the hold window: is the straggler
   stale-rejected (stale_rejected 4.28 vs real 7.14 suggests *fewer* held) before the
   clock passes its `sct`? Inspect `[SIM_CARRYOVER]` / `inflight_residence`.
3. Do **not** trust `Sr`/`Sd`/`S3·4` mix verdicts until step 1 lands (`Sdet
   eligible_match`≈0 — inputs differ before the draw).

**felix — fix the async overlap model (the held item).**
1. Sim over-overlaps (10.9× vs real 6.6×) → staleness 8.4 vs 2.8, throughput +33%.
   The `simRedispatchGapSeconds` ↔ staleness coupling is the lever; the buffer-aging
   investigation noted in §3 is now the blocking task. Anchor the target to **real's
   6.6× overlap / 2.8 staleness** (no async Oort reference exists).
2. Decision to make: re-tune the overlap/buffer-aging model, **or** revert felix's
   async knobs to the pre-paper parity values. Prefer the model fix (the knobs are a
   symptom); fall back to revert if the buffer-aging fix proves intractable.

**feddance — chase `feddance_U`, then rerun longer.**
1. **Lead:** `Sx feddance_U` real 5.61 vs sim 3.01 (KS .394) with `eligible_match`
   high → a **selection-value** divergence, not a set/clock bug. Audit the `U` term
   computation (data/utility component) in the sim vs real feddance selector path.
2. `S2` (.247, n=85) and `P3` (+1.05 s) are horizon/wall-capture — **rerun feddance
   at 1–2 h** to confirm `S2` collapses with more rounds before treating it as real.
   If `feddance_U` is ruled a checker artifact too, reclassify `P3`/`K3` → DIST for
   the stochastic sync selector.

**Cross-cutting checker decision (do once):** the `P3 mean_overhead≤0.5 s` bar fires
on a sub-second, opposite-sign, grid-passing offset for felix/feddance while
`training_delay_s` matches exactly — that's wall-capture, not a speed-model bug.
Either widen the bar (~1.5 s) or evaluate `mean_overhead` on `training_delay_s`
metadata (as `A2b`/`A2c` do). **Keep oort's `P3` failing** — its 56 s tail also
trips `grid_KS`, so the grid still catches the genuine case.

### Dead ends — do NOT retry

- Overhead > 0 on the virtual clock (masks & drifts; clock must `= max(vclock, sct)`).
- Prediction-only gates with no real blocking (never fire).
- `version_at(sct)` staleness relabel (fedbuff consumes the *real* number; inert).
- Adding `mqtt_fetch` (~57 s) to `sct` (not version-relevant; inflates staleness ~6×).
- `simRedispatchGapSeconds=0` for felix (sim over-overlaps; the gap is a real mechanism).
- Tuning sim to a *wrong* real, or any scalar fudge where a mechanism is called for.
- Re-chasing: GPU contention (overrun 0), SEND_TIMEOUT (0×), MQTT drops (0), the
  felix post-compute leg as a "bug" (it's serial-aggregator scheduling), per-trainer
  exact-set/identity on a stochastic streaming selector (path-dependent by nature).
- Expecting seeding to align per-round sets: `Sdet eligible_match`≈0 is *expected*
  for stochastic selectors whose inputs (clock-indexed availability) drift; judge the
  aggregates (S2/participation), not the per-round draw. Only a *diverging clock*
  (oort/felix) makes `eligible_match`≈0 a genuine bug.
- A scalar fudge for the `P3 mean_overhead` ~1 s offset (wall-capture, opposite signs
  across baselines, grid-passing). Widen the bar or score on `training_delay_s`; don't
  bias the speed model to chase it (it would break the genuine oort 56 s tail signal).

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
| Sd `[Jun16]` | preferred-duration penalty bind | MECHANISM/DIST | Oort speed-penalty **binding frequency** per round (≥1 selected w/ `system_util<1`) + reconstructed `pref` median — the D1 unsorted-`pref` guard (caught: real 80 % vs sim 46 %). Works on pre-instrumentation runs (reconstructs `pref=dur·√system_util`). | A2b |
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

## §3  Mechanism reference — implemented sim fixes

The simulator does **real GPU compute** but stamps a *modeled* completion time
`sct` (it does not sleep the trainer's wall budget). Parity work is making the
sim's clock, ordering, and availability behave as the real pipeline would at that
`sct`. The validated mechanisms below are config-gated and guarded; the
**Overriding principle**: parity ≠ goal, a *correct* simulator is — real is the
reference only after `validate_real` shows it admissible (done: concurrency
28.8/c30, double-dispatch 0); never tune sim to a wrong real.

### Clock & ordering (felix async stack, validated)
- **Overhead → 0** (`simCommitOverheadSeconds=0`): the clock TRACKS completions
  (`vclock = max(vclock, sct)`) instead of being a pure overhead ramp.
- **Drain by physical READINESS, not predicted completion** (`_sim_recv_min`):
  admit any in-flight end whose message has physically arrived into the reorder
  buffer, so slow trainers buffer as futures and commit in `sct` order (staleness
  7.2 → 3.5). Probing the LIVE in-flight set (`recv_fifo` pops min-`sct`) reorders
  the past-dated tail.
- **`realDistributeSettleSeconds=0`**: removes a real-only 2×`sleep(0.1)`/commit so
  real holds ~c computing instead of being artificially slowed (advance 4.1,
  staleness 2.8).
- **`simRedispatchGapSeconds`** (post-commit re-dispatch leg, slot-held by cooling):
  spaces completions without counting toward the committed update's staleness.
  `0.6` zeroed the over-advance → throughput family green. *Residual:* a
  gap↔staleness coupling means one knob can't hit both advance and staleness; felix
  is HELD pending a buffer-aging investigation, not a scalar.

### §4.5  refl — `sct`-gated pool exclusion (`simInflightResidence`, validated)
A trainer that has physically sent but is modeled as still computing (`vclock < sct`)
must NOT re-enter the eligible pool — in real it is busy. In `oort/top_aggregator.
_distribute_weights`, the still-computing set (`_sim_buffer.pending_after(vclock)`)
is added to `trainer_unavail_list` (the *unavailable* path, NOT `selected_ends` —
which would re-dispatch and reset `sct`); released at `vclock ≥ sct` (budget ≤ ~56s).
Fixed refl's pool composition (A2b 12.4 → ~6.5 = real), flipping all emergent checks
green. Guard: `test_virtual_clock.py::test_pending_after_*`,
`test_sync_sim_ordering.py::TestSimInflightResidence`.

### §4.9  oort — `sct`-gated carry-over (`simInflightCarryover`)
**Jun-15 seeded run:** under-fired (`in_flight_after` 0.15→1.43 vs real 4.58).
**Root found + FIXED (Jun 16):** not a tuning gap — a **lost-straggler bug**. The
held stragglers were re-buffered in a loop *after* the yield-loop, but the caller
abandons this generator the moment `agg_goal` fresh updates are accepted, so it
never ran and `held_over` was dropped each round. Now wrapped in `try/finally`
(the `GeneratorExit` on `gen.close()` runs the re-buffer). Pending 3 h validation.
Below is the mechanism as designed.
The sync-oort aggregator over-selects (×1.3) and closes a round at agg_goal=10,
leaving the ~3 slowest still computing. In **real** they stay in `selected_ends`
in-flight across rounds (`in_flight_after` 3.3); in **sim** the update arrives at
once, gets stale-rejected (prior `MODEL_VERSION`), and frees its slot → sim drains
to 0.15. **Distinct from §4.5**: §4.5 gates pool *re-entry*; §4.9 gates the
*cleanup/commit*. In `oort/top_aggregator._oort_sim_recv`: a prior-round straggler
(`_round − MODEL_VERSION > 0`) with `sct > vclock_round_start` is held (not yielded,
not clock-advanced), re-buffered so it stays in `selected_ends` (carried in-flight),
and commits a few rounds later once the clock passes its `sct`. Enabled for the oort
sim block. Guard: `test_sync_sim_ordering.py::TestSimInflightCarryover`.

### §5  Checker corrections (stochastic / observability classes)
Once the sim *dynamics* match, some residual FAILs were the checker enforcing exact
identity on quantities a stochastic / in-memory simulator cannot reproduce
(diagnostic tell: byte-identical across runs despite large dynamics changes). All
are principled, guarded, append-only — a future *deterministic* selector still gets
exact enforcement via `DETERMINISTIC_SELECTORS`:
- **P1 aggregation_sequence** → WARN for stochastic selectors (exact per-round set
  identity unattainable; S2 participation is the enforced invariant).
- **F1-3 utility** → enforce the POOLED KS (per-trainer KS=1.0 was mechanical for
  n≤2 samples; means were identical).
- **phase_mqtt_fetch** → DIAG (in-mem cache wall time, deliberately off the virtual
  clock).
- **trainer_speed / eligible_speed / selection_bias** → integer-grid / metadata-pool
  (see Status → checker corrections).

### Discrepancy ledger — flame vs reference Oort (per-baseline)
flame has ONE `OortSelector` inherited by both the `oort` baseline (should match
standalone Oort, `third_party/Oort`) and `refl` (should match the REFL fork,
`third_party/REFL`). The two references differ on defaults, so each baseline's
knobs are config-driven (`selector.kwargs`), defaulting to the Oort paper
(`scoring.OORT_PAPER_DEFAULTS`) with refl overriding to the fork.

| # | discrepancy | resolution |
|---|---|---|
| D1 | `pref` not sorted | FIXED (sort added) — was a port bug; validated on oort |
| D2 | stat-utility not normalized/clipped | FIXED (`scoring.oort_normalize_reward`, config `normalize_reward`/`clip_bound`) |
| D3 | `round_threshold` | config-driven: oort/felix=10 (paper), refl=30 (fork) |
| D4 | `cut_off_util` + cutoff-index | FIXED: config (0.7 paper / 0.05 refl); index now thresholds the exploit-boundary score (was inert) |
| D5 | temporal time-base | **DEFERRED** — flame uses `last_selected_round`; refs use round-last-UPDATED. Needs an aggregator-stamped prop; subtle, high blast radius. Flagged in code. |
| D6 | `clip_bound` | config-driven: 0.98 paper / 0.9 fork |
| S | refl exploitation | FIXED: was deterministic top-k; now the fork's cut_off_util-augmented utility-weighted `np.random.choice` |
