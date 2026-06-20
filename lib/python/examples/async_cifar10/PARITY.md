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
model, deterministic seeding, the pass/total scoreboard, and every checker rung.
Last green: **160 pass / 7 skip** (adds `eval_commit_timeliness` + the eval-stale-`sct` guard).

---

## Workflow policy: minimize time & runs to parity

The objective is parity in the fewest wall-hours and cluster runs. Rules:

1. **Run real only when needed.** Real is the reference; once a baseline's real
   run is admissible and stored, re-run real only when a change affects the *real*
   path. Sim-only changes validate against the stored real dir.
2. **Over-instrument telemetry deliberately.** Emit more signals than any single
   check needs if they capture runtime behavior and speed up root-causing — cheap
   to log, expensive to re-run for. (E.g. per-round `inflight_residence`, the
   `SIM_CLOCK_DIAG` past-dating counters localized oort + felix from stored runs.)
3. **Root-cause per baseline, then scope the fix to its blast radius.** Common
   cause across baselines → fix once. Independent fix that cannot affect others →
   land it. Fix that *could* perturb another baseline → serialize it (one baseline
   per run round) so attribution stays clean.
4. **A fix touching the real path → re-run both** real and sim; a sim-only fix →
   re-run sim, reuse stored real.
5. **Shortest run that exhibits the issue.** Don't default to 3–4h. Use the
   minimum duration that surfaces the check under test; reserve long runs for
   `C1`/`C2` convergence or round-count-compounding residuals only.
6. **Crisp code comments** — one sentence at most; let tests document behavior.
7. **Context-free variable names** (see Naming discipline below).

---

## Run-length budget: minimum time per fix (don't burn GPU for nothing)

**Standing instruction:** whenever a run is proposed for a fix/measurement, state the
**minimum duration up front**, keyed to the table below, and *why* — never default to
3–4h. Max principled insight per GPU-hour; run the shortest duration that makes the
check-under-test statistically read, then stop.

| what you're validating | min run | why |
|---|---|---|
| telemetry field present / instrument sane | **5–10 min** | a few hundred commits populate any per-commit field — grep the jsonl |
| a single MECHANISM rung (gate hold, `commit_visibility`, `residence`, `selection_detail`) | **45 min** | every mechanism check fires; per-commit distributions stabilize |
| clock/advance residuals that compound (`K2`/`K3b`, past-dating cumulative) | **90 min – 2 h** | round-count-compounding drift needs the rounds; felix past-dating is fully visible by ~hr 1 |
| low-frequency eligibility-shape / round-count drift (refl `K2`) | **3 h** | only surfaced at 3h despite a clean 45min pass |
| `C1`/`C2` convergence (accuracy/loss sign-off) | **full budget (3–4 h+)** | terminal-state + curve parity only — nothing else needs this |

Rules: **smoke (5 min) before any multi-hour run** (confirm the field lands); **one
mechanism per run round** when a fix could perturb another baseline (serialize); a
**sim-only** change validates against the stored real dir (don't re-run real).

> **These pending commit-visibility runs are MEASUREMENT** → the *90 min – 2 h* band.
> 2 h is the ceiling (margin on past-dating compounding), **not** convergence; anything
> past 2 h here is wasted GPU. The two confirmation reads (oort straggler-lag, felix
> clock-jump signature) are stable well before then.

---

## Doc policy: ONE status section, not one per run

This doc used to grow a new `## Status (date — ...)` section per check-in,
left in place "for the record." That's bloat — old scoreboards are
regenerable from `parity_check.py --batch` and old hypothesis tables just
restate what the next rerun already overwrote. **From now on: there is
exactly one `## Status` section, dated to the most recent rerun, fully
replacing the previous one.** Durable lessons that don't expire with the next
rerun (triage rules, dead ends, validated mechanisms) live in their own named
sections below status and get *updated in place*, not appended to.

---

## Status (Jun 20 — felix past-dating ROOT-CAUSED from stored logs: EVAL ships a stale train `sct`; fix landed, smoke pending)

**The "fresh past-dating" is an EVAL-task artifact, not a clock/clamp problem.**
Mining the stored clamp-run sim log + per-trainer telemetry settled it without a
rerun:

- The trainer computes its modeled completion `_sim_completion_ts = sim_send_ts +
  duration` **only in the train path** (`trainer/pytorch/main.py:822`). `evaluate()`
  never recomputed it. But the send path stamps `msg[SIM_COMPLETION_TS] =
  self._sim_completion_ts` onto **every** simulated message (`syncfl/trainer.py:418`).
  So **every eval message carries the trainer's LAST TRAIN round's `sct`** — long
  past by the time it commits.
- Direct evidence (trainer `…0544`): trained at rounds 1 (`sct=24.2`) and 33
  (`sct=165.6`), then was dispatched **eval 65×** for the rest of the run. All 65
  evals shipped `sct=165.6`, committing at rounds 2217/2253/2289/2325 with
  `commit_gap_s` up to **5234s**. The frozen `sct=166` repeating every ~36 rounds in
  the SIM_BARRIER log is this one trainer's evals.
- They are mislabeled **"fresh"** because the `pastdated_by_source` classifier keys
  on `MODEL_VERSION` (= current round for a freshly-dispatched eval), not on the
  stale `sct`. Eval commits never emitted the `commit_gap_s` telemetry (they exit at
  the STAT_UTILITY branch before the train-only emit), so **U6 (train-only) reads
  16.8s straggler-dominated** while the **SIM_BARRIER/CLOCK_DIAG stream (incl. eval)
  reads "fresh=95%"** — two populations, one bug.
- A stale eval `sct` is also the *minimum* of the reorder buffer, so it poisons
  `peek_min_ts` (the gate/probe-ceiling key) — degrading train ordering too.

**Why the clamp was inert (now explained, not hypothesized):** the clamp caps
*forward* clock advance; this bug delivers a far-*past* `sct` on the wire, which the
clamp cannot touch. The `exp > vclock` guard discussion is moot.

**Corroboration:** oort (sync) dispatches **0 eval tasks** and shows **no
past-dating**; felix dispatches both. Eval-dispatch is the felix/oort differentiator.

### Fix (landed, sim-only)
`evaluate()` now stamps its OWN completion ts, `sct = send_ts + max(real_eval_gpu,
training_delay_s/20)` (`trainer/pytorch/main.py:1072`), so eval commits at ≈now
(`commit_gap≈0`) and stops poisoning the buffer key. Real path unchanged (still
sleeps `floor(D/20)`) → validates against the stored real dir. Eval commits now emit
their own `commit_gap_s`/`update_visibility_lag_s` telemetry tagged
`task_to_perform="eval"` (`asyncfl/top_aggregator.py`, eval branch), and
`analyze_run.py` splits the commit-gap / visibility-lag plots train-vs-eval (eval
series absent for no-eval baselines). **Smoke pending** (5–10 min: confirm eval
`commit_gap_s`→~0 and `pastdated_by_source` fresh→~0).

| baseline | score (pre-fix) | run dirs | U6 `commit_visibility` read |
|---|---|---|---|
| **felix** | **33/42** (clamp run, pre-eval-fix) | `run_20260620_002022…real` / `run_20260620_002029…sim` (1.5h) | real_mean **0.017s** vs sim_mean **16.819s** (train-only); eval stream past-dated to 5234s → **eval-stale-`sct` CONFIRMED; fix landed, rerun pending** |
| **oort**  | **41/45** | `run_20260619_124438…real` / `run_20260619_122953…sim` (2.5h) | real_mean **0.004s** vs sim_mean **0.001s** (point-mass) → **no past-dating; 0 eval dispatched** |
| **refl** | 38/44 (stored, no rerun) | Jun 18 dirs | — |
| **feddance** | 41/43 (stored, no rerun) | Jun 18 dirs | — |

### The `update_visibility_lag_s` instrument (landed prior session)
One metric, both modes, all baselines: **`committed_ts − ready_ts` in the aggregator's
own clock.** sim = `vclock.now − sct`; real = `wall(commit) − wall(MQTT arrival)`. Async
target ≈0 (independent commits at own readiness); sync = barrier wait (matches in both).
Rung **U6 `commit_visibility`** (Stage 6, MECHANISM/DIST, KS + mean-gap), declared
**upstream of `staleness`**; self-SKIPs when the field is absent.

### Settled roots (the measurement runs resolved both open hypotheses)
| baseline | root — now settled with evidence |
|---|---|
| **felix** | **ROOT-CAUSED (Jun 20): EVAL tasks ship a stale train `sct`.** `evaluate()` never recomputes `_sim_completion_ts`, so the send path stamps every eval message with the trainer's last TRAIN completion ts (`trainer/pytorch/main.py:822` set in train only; `syncfl/trainer.py:418` stamps unconditionally). Eval then commits with `commit_gap` up to 5234s, mislabeled "fresh" (classifier keys on `MODEL_VERSION`, not `sct`). This is the "fresh=95%" signature; the train-only U6 16.8s tail is the *secondary* effect of the stale eval `sct` poisoning the reorder-buffer minimum. **Fix landed** (`evaluate()` stamps `send_ts + max(gpu, D/20)`); clamp was inert because it caps forward advance, not a past `sct`. Smoke pending. The K3b/K4/U3 up-ladder residuals are expected to shrink once eval stops poisoning ordering — re-measure after the rerun. |
| **oort** | **Carry-over decay = A2c stochastic, NOT a real-timing gap — verification complete.** U6: both modes commit immediately (sim lag **0.001s** ≤ real **0.004s**); sim if anything drains *faster*, so the `in_flight_after` decay (Sr: real 3.91 vs sim 0.69) is **not** late/early commit timing — it is the selection mix tightening. Direct mix evidence: **Sd preferred-duration penalty binds real 0.822 vs sim 0.495** (sim under-penalizes slow trainers → selects fewer slow → fewer overcommit slots → fewer carry-overs), worst Sx term `system_util` (real 0.942 vs sim 0.959). **NOT a train/eval-overwrite bug** — sync oort dispatches 0 eval tasks (verified in both run logs). Same A2c class as refl/feddance; closed only by the speed-model work, not an oort-specific mechanism. |
| **refl / feddance** | A2c stochastic speed-tail. Unchanged, deprioritized. One speed-model fix may close both, *and* the oort `system_util`/Sd mix (now a single A2c family). |

### Implementation steps — status
1. **U6 point-mass guard (checker-side) — ✅ LANDED.** Both-modes mean lag ≤ `NEAR_ZERO_LAG_S`
   (50 ms) ⇒ pass on mean, annotate "point-mass: KS uninformative" (the A2 `num_candidates`
   precedent). oort U6 now PASS (sim 0.001s / real 0.004s) → **41/45** against the stored dir;
   felix's genuine 14.8s gap still FAILs. Guard: `test_commit_visibility_parity` case (5).
2. **felix eval-stale-`sct` — ✅ ROOT-CAUSED + FIX LANDED (Jun 20), smoke pending.**
   `evaluate()` now stamps its own `_sim_completion_ts = send_ts + max(gpu, D/20)`
   (`trainer/pytorch/main.py:1072`); eval commits emit task-tagged
   `commit_gap_s`/`update_visibility_lag_s`; plots + checker split train/eval. The
   clock-jump clamp is now understood as inert *by construction* (it caps forward
   advance; the bug delivers a past `sct`) — leave it enabled, don't re-tune
   `_SIM_ORDER_SLACK_S`. Smoke (5–10 min) to confirm eval `commit_gap`→~0; then a
   90-min run to re-read U6/K3b/K4/U3 (expected to tighten once the buffer key is
   clean). Only after that is the clamp/buffer-aging direction even worth revisiting.
3. ~~oort task-type-keyed latency~~ — **DROPPED, premise falsified** (sync oort = 0 eval dispatches;
   the Sd residual is A2c, see step 5).
4. **felix D5** (stamp-at-commit, `asyncfl/top_aggregator.py:516`) — after the clamp rerun confirms
   past-dating resolved; mind `round_nudge_type`.
5. **refl + feddance + oort speed-tail** (A2c) last — one speed-model fix may close all three
   stochastic-mix residuals (incl. oort's Sd 0.822/0.495).

### Durable lessons (kept; update in place, don't append)

- **`Sdet` triage rule.** `eligible_match≈0` **with matching aggregates** =
  stochastic-class, PASS as-is; `eligible_match≈0` **with a diverging clock**
  = genuine, fix the clock; `eligible_match` high but `decision_match≈0` = the
  **values** diverge, not the set. Localized all four baselines' roots.
- **`participation`/S2 vs per-round Jaccard.** Round-to-round draw mismatch
  (`Sdet`/`S1`) is expected for a stochastic selector and not itself a bug.
  But a **systematic per-trainer skew over the full run** (S2 `matched_count_ks`
  large, `max_diff` not averaging out by run's end) is not noise — it means
  something is biasing *which* trainers win, not just *when* (this is how
  refl's `participation` finding was distinguished from normal stochastic
  variance — see Status table above).
- **`U6 commit_visibility` KS is a false-positive on a sub-ms point mass.** When
  both modes commit immediately (sync oort: real_mean 0.004s, sim_mean 0.001s) the
  lag is a near-degenerate point mass at ~0, so KS→1.0 carries no signal — identical
  to A2 `num_candidates` KS=0.999. The substantive read is `mean_diff` (3 ms ≪ the
  2.0s bar). A real divergence (felix past-dating) shows up as a **large mean_diff**
  (14.8s), not just KS. Trust U6 only when `mean_diff` clears the bar; both-near-zero
  ⇒ pass on mean (point-mass guard, Status → Next steps #1).
- **`P3 mean_overhead` is wall-capture, not a speed-model bug**, when
  sub-second, opposite-sign across baselines, and `grid_KS`/`training_delay_s`
  metadata match — only trust a `P3` FAIL when `grid_KS` also fails.
- **`K3b` ≠ "missing overhead" when `implied_per_commit_overhead_s`≈0.** The check
  is *named* overhead_residual, but its residual is `real_advance − sim_advance`,
  which is also moved by the per-round **max-of-K speed** order statistic. If the
  implied per-commit overhead is sub-0.05s yet `K3b` fails, read the K3a `max_speed`
  line: a sim/real gap there (refl: sim 7.75 vs real 8.64; `trainer_speed` max 21 vs
  29 with p99 matching) means the advance gap is a **thinner sim speed-tail / selection
  mix**, not omitted overhead. Do NOT add a `simCommitOverheadSeconds` scalar to chase
  it (dead end); fix the speed-tail model or accept it as the feddance A2c class.
- **Oort `in_flight_after` decay ≠ "carry-over gate broken" once decile-0 matches.**
  If `inflight_residence` telemetry shows decile-0 `in_flight_after` ≈ real but
  decaying over the run, the gate itself (pinned `_round_start_vclock`) is correct;
  the decay is a *selection-mix tail* effect, not a gate bug.
  **The `system_util` recency guard was the wrong fix (Jun 19):** with intrinsic
  per-task latency, a trainer's last-observed duration == its current duration, so
  returning `system_util=1` for a "stale" value removes a *correct* speed penalty —
  observationally identical to the forbidden speed-tail widening, and there is no
  principled round threshold (the staleness premise is false). Evidence it isn't a
  clean `system_util` divergence anyway: `selection_bias` passes and sim selects
  *slower* than real early (decile-0 5.6 vs 3.8; trajectories cross). Correct path:
  **instrument with `commit_visibility`** to classify the decay (real timing gap vs
  A2c stochastic), and land the **task-type-keyed latency** correctness fix (score
  `system_util` on the train-task duration, not whichever task committed last).
- **`phase_gpu_compute` gap is run-length-sensitive.** At 1h (felix), sim=0.18s vs
  real=0.42s (FAIL, KS=0.334). At 2.5h, sim=0.422s vs real=0.356s (PASS, KS=0.08) —
  gap closed and reversed. Don't treat a 1h `phase_gpu_compute` FAIL as a permanent
  structural gap; re-check at 2.5h before acting.
- **`pastdated_by_source=[fresh=...]` was an EVAL artifact, not "fast trainers lapped
  by the clock" (Jun 20 — supersedes the prior reading of this counter).** The
  classifier keys on `MODEL_VERSION` (= current round for a freshly-dispatched eval),
  so eval commits carrying a *stale train* `sct` are labeled "fresh" with a huge
  per-commit gap. Root: `evaluate()` never recomputes `_sim_completion_ts`; the send
  path stamps the last train value (`syncfl/trainer.py:418`). **Tell:** a single
  trainer's eval commits repeat the *same* `sct` for hundreds of rounds (e.g. `…0544`
  `sct=166` × 65). Cross-check the "fresh" log counter against the **train-only U6
  telemetry** (which excludes eval) and per-trainer `sct` cardinality before trusting
  a "fresh dominates" read — they told opposite stories here. Fixed by stamping a
  per-eval `sct`.
- **Two past-dating populations, two telemetry streams.** `commit_gap_s`/U6 telemetry
  is emitted **only in the train (WEIGHTS) branch**; eval (STAT_UTILITY) exits before
  it. So the train-only U6 mean and the all-commits SIM_BARRIER/CLOCK_DIAG stream can
  diverge wildly (16.8s vs "fresh=95%/5234s"). After the Jun-20 fix, eval emits its
  own task-tagged `commit_gap_s` so both are visible and the analyzer/checker split
  train vs eval. Always disambiguate which stream a "past-dating" number came from.
- **oort `sim_committed_fresh` = agg_goal confirms block-for-K fix; don't re-examine.**
  At 2.5h, `sim_committed_fresh=10` matches real exactly. The block-for-K mechanism is
  closed; any future `committed_fresh` gap is a different root.
- **Run length matters.** 45min exercises every mechanism check but isn't
  enough for `C1`/`C2` convergence sign-off or to catch round-count-compounding
  clock residuals / low-frequency eligibility-shape drift (refl's 3h-only
  `K2` regression, surfaced only at 3h despite a clean 45min pass).
- **Checker-side fixes validate instantly against stored run dirs; only sim
  *mechanism* changes need a new cluster rerun.** Land and verify all checker
  corrections against existing dirs first, then batch mechanism changes into
  one rerun.

### Dead ends — do NOT retry

- Overhead > 0 on the virtual clock (masks & drifts; clock must `= max(vclock, sct)`).
- Prediction-only gates with no real blocking (never fire).
- `version_at(sct)` staleness relabel (fedbuff consumes the *real* number; inert).
- Adding `mqtt_fetch` (~57 s) to `sct` (not version-relevant; inflates staleness ~6×).
- `simRedispatchGapSeconds=0` for felix (sim over-overlaps; the gap is a real mechanism).
- `simRedispatchGapSeconds=0.6` for felix — tested Jun 16, no measurable effect; don't retune this scalar further, go to the buffer-aging model.
- Expecting the felix seed fix (min budget default) alone to eliminate past-dating — confirmed Jun 17: past-dating drops from 73%→14% initially but recovers to 59% by commit 5000 with `gate_holds=0` throughout. Other cascade sources remain active; need source-level instrumentation, not scalar tuning.
- Expecting oort carry-over decay to be a run-length transient — confirmed Jun 18 2.5h: `in_flight_after` sim=0.47 vs real=3.65 at 2.5h; zero by decile 2 of 10. Structural, not transient. Don't re-test duration.
- "Widen the oort slow-speed tail" to fix carry-over decay — `trainer_speed` already passes; sim tail is if anything wider (max 29 vs 21 at 2.5h). It's a selection-mix tail effect, not a speed-model gap.
- **`system_util` recency guard for oort carry-over decay (Jun 19)** — with intrinsic per-task latency the last-observed duration is *correct*, so returning `system_util=1` for a "stale" value is a value-fudge identical in effect to the (forbidden) speed-tail widening, with no principled threshold. The decay is the A2c selection-mix class (`commit_visibility` confirms no past-dating); close it with the speed-model work, not an oort-specific knob.
- **oort task-type-keyed latency / `PROP_ROUND_DURATION` train-vs-eval split (Jun 20)** — premise FALSIFIED by the run: sync oort dispatches **0 eval tasks** (real & sim; eval is bundled into the train commit, `oort/top_aggregator.py:901-903`), so nothing overwrites `PROP_ROUND_DURATION`. felix *does* dispatch both, but its `system_util` penalty is inert (`round_threshold=70`, Sd passes, `system_util≡1.0`) so the split is a no-op there too. Don't key the duration by task; the oort Sd gap is A2c.
- **felix clock-jump clamp to fix "fresh" past-dating (Jun 20)** — wrong target. The
  clamp caps *forward* clock advance, but the "fresh" past-dating is EVAL committing a
  *past* stale `sct` (root-caused Jun 20: `evaluate()` reused the last train
  `_sim_completion_ts`). No forward-advance cap can fix a stale-past wire timestamp.
  Leave the clamp enabled (cheap, correct for genuine straggler jumps) but stop
  attributing "fresh=95%" to it. Don't re-tune `_SIM_ORDER_SLACK_S`. Fix is in
  `evaluate()` (stamp a per-eval `sct`).
- **felix dispatch-timestamp pacing for past-dating (Jun 19)** — pacing/inflating the effective dispatch `sct` so fewer commits *look* past-dated falsifies fast trainers' modeled completion and pushes staleness the wrong way (same family as the redispatch scalar). The root is the inert arrival-gate + clock jump; fix it with the modeled-completion clock-jump clamp (`simClockJumpClamp`, landed Jun 20), not the dispatch ts.
- Tuning sim to a *wrong* real, or any scalar fudge where a mechanism is called for.
- Re-chasing: GPU contention (overrun 0), SEND_TIMEOUT (0×), MQTT drops (0), the
  felix post-compute leg as a "bug" (it's serial-aggregator scheduling), per-trainer
  exact-set/identity on a stochastic streaming selector (path-dependent by nature).
- Expecting seeding to align per-round sets: `Sdet eligible_match`≈0 is *expected*
  for stochastic selectors whose inputs (clock-indexed availability) drift; judge the
  aggregates (S2/participation), not the per-round draw. Only a *diverging clock*
  or a *systematic per-trainer skew* (not round noise) makes it a genuine bug.
- A scalar fudge for the `P3 mean_overhead` ~1 s offset (wall-capture, opposite signs
  across baselines, grid-passing). Widen the bar or score on `training_delay_s`; don't
  bias the speed model to chase it.

### Naming discipline (instruction — apply when touching baseline code)

State and variable names must be **context-free**: a reader should not need the
surrounding code to know what a name refers to. Round/version/time confusion has
caused real bugs here (D5), so:
- A name ending `_round` is a **round index** (int), never a timestamp. Use
  `_ts`/`_time_s` for times. Don't name a round int `_stamp`/`timestamp`.
- Qualify *whose* round: aggregator's global counter vs the selector's last-run
  round vs a per-trainer property are different things. (Done for oort+refl:
  selector `self.round` → `self._last_selection_round`; the per-trainer property
  read is `end_last_selection_round`. **Deferred:** the base aggregator
  `self._round` → `self._agg_round` — it's a `TopAggregator` attribute, so that's
  a dedicated all-baseline pass, not scopeable to one baseline.)
- A local should say *what it is*, not just its type-shape — `trainer_model_version`
  (the version an update was trained on) is kept precisely because it names the
  provenance; a vaguer `trained_round` was rejected.
- Don't paper over an ambiguous name with a comment — rename it. Comments explain
  *why*, names carry *what*.
- Scope renames to the baseline you're in (oort+refl share `OortSelector`); felix
  (`AsyncOortSelector`) and others are separate passes.

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
| U6 `[Jun19]` | Commit visibility lag (`update_visibility_lag_s`) | MECHANISM/DIST | The aggregator-clock delay between an update becoming READY and being COMMITTED diverges — sim commits late (past-dating) relative to real. Same metric both modes (sim `vclock−sct`, real `wall commit−arrival`); the **upstream** cause of a staleness divergence. Self-SKIPs if the field is absent. | K3 |
| U3 | Staleness distribution | MECHANISM/DIST | Staleness diverges (async: directly downstream of clock under-charge) | K3,U5,U6 |
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

### §3.async  Async ≠ sync selector knobs — do NOT inherit the Oort *paper* defaults
`third_party/Oort` is **sync-only**; there is no async Oort reference, so the paper
defaults (`OORT_PAPER_DEFAULTS`, e.g. `round_threshold 10`, `exploration_decay .95`)
are SYNC values. Applying them to the async `AsyncOortSelector` (felix) regressed it
(overlap 10.9× vs real 6.6×, staleness 8.4 vs 2.8) because the sim overlap model is
calibrated to the selected MIX, and the sync knobs narrow that mix. Root theme:
**many Oort knobs are parameterized *per round*, but "a round" is a different unit in
async (one `agg_goal` batch) than sync (a full barrier), and async runs ~2–3× more of
them.** Inheriting sync values therefore misbehaves:

- **`round_threshold` (speed penalty)** — exists to protect a SYNC barrier (round =
  max-of-K; a straggler blocks everyone). Async/fedbuff has no barrier (stragglers
  commit stale later) → the penalty should be largely **inert**. Felix uses **70**
  (broad mix). NB the pacer ([async_oort.py:546](../../flame/selector/async_oort.py#L546))
  only ever *raises* it toward 100, so the start value washes out over a long run.
- **`exploration_decay`** — applied once **per round**; async's higher round count
  collapses a sync-tuned decay almost immediately (0.95 → exploration floored in ~29
  rounds). Felix uses **0.999** (still reaches an exploitation phase across ~1150
  rounds). `0.9999` ≈ permanent exploration (never exploits) — rejected.
- **temporal/UCB** `√(0.1·log(round_num)/last_selected_round)` — `log(round_num)`
  inflates with async's round count (more exploration pressure, automatically).
- **pacer cadence** (`pacer_step` rounds) — fires more often in wall-time in async.
- **staleness weighting** — async-only (sync has none); confirm fedbuff down-weights.
- **D5 temporal time-base** (FIXED Jun 16, oort+refl) — the issue was *write
  timing*, not value: stamping `PROP_LAST_SELECTED_ROUND` at commit let its
  visible value ride commit ordering (sim-regular vs real-jittery). Now stamped
  at selection (value = selection round, unchanged). Matters more in async,
  where selection and commit decouple. felix (`AsyncOortSelector`) deferred.

Principled generalization (not yet done): re-parameterize the per-round terms by
**wall-time or samples-seen** so they're invariant to round semantics. Until then,
async knobs are config-driven and anchored to the real run's spacing, NOT the paper.
**Felix's 70/0.999 is a new operating point** — the first 3 h run is its first parity
test there; if overlap/staleness still diverge, the next move is the overlap-model
(buffer-aging) re-tune, not more knob changes.

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
(the `GeneratorExit` on `gen.close()` runs the re-buffer).
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

**Third mechanism, FIXED (Jun 17): carry-over threshold creep.** The 1h rerun
showed carry-over still under-firing (`inflight_after` sim 0.82 vs real 4.24).
Mining per-round `in_flight_after` from the existing sim run showed it **starts at
4.31 (≈ real 4.24) and decays to ~0 by mid-run** — a progressive collapse, not a
uniform absorb. Root: the gate held a prior-round straggler when `sct >
vclock_round_start`, but `vclock_round_start` was re-read as `self._vclock.now`
*inside each* `_oort_sim_recv` call. The block-for-K-fresh retry loop creates a
fresh generator per pass *after* earlier passes advanced the clock, so the
threshold crept forward within a single round and committed stragglers whose `sct`
fell between the true round start and the advanced clock. Early rounds (few
retries) barely creep → match real; later rounds creep hard → drain. Fix:
`_aggregate_weights` pins `self._round_start_vclock = self._vclock.now` once, before
the first `_oort_sim_recv` call; the gate reads that pinned value. Guard:
`TestSimInflightCarryover::test_pinned_threshold_holds_straggler_across_retry`.

**Fourth mechanism, OPEN (Jun 17 eve): speed-tail selection decay.** The Jun 17
rerun (post threshold-creep fix) confirmed the gate is correct — decile-0
`in_flight_after` = 4.02 ≈ real 4.24. But the carry-over decays to ~0 by decile 4
despite the pinned threshold. Mining `inflight_residence` events shows
`carried_over_ages` growing (ages 0→1→2→3→4→5 per trainer across deciles 0–2)
then hitting zero: the SAME slow trainers persist in carry-over for many rounds
but no NEW slow trainers are seeded into carry-over. Root: Oort's
`system_util` speed penalty penalizes slow trainers as utility is learned;
over ~200 rounds, the selection mix tightens to fast trainers → fewer
overcommitment slots with `sct >> vclock_round_start` → steady-state carry-over
count → 0. In real, physical timing variability means even "fast" selections
sometimes carry over (real `preferred_duration` binding 0.869 vs sim 0.522).
This is the **same A2c class** as refl/feddance selection-mix; carry-over is one
manifestation. Fix direction: widen the slow-speed tail in sim so Oort selects
enough slow trainers to sustain ~3 carry-overs at steady state.

**Second mechanism, FIXED (Jun 16) and CONFIRMED (Jun 18 2.5h): block-for-K-fresh.**
Even with carry-over correct, a prior rerun showed `committed_fresh` sim 7.24 vs real 10
— a *starvation*, not a carry-over bug. Root: `_aggregate_weights`'s second
poll loop only ran `while not self.simulated`, so a sim round got exactly one
`_oort_sim_recv` pass; if a fresh (this-round) trainer's message wasn't ready
within the adaptive `grace` window (`4× EMA of past full-drain time`,
`syncfl/top_aggregator.py`), it was silently skipped this round and re-probed
next round — by which point `self._round` had advanced, so it commits **stale**
instead of fresh. Fix: removed the `not self.simulated` gate so sim also retries.
**Confirmed Jun 18 2.5h**: `sim_committed_fresh=10` matches real exactly. Mechanism closed.

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
| D5 | temporal time-base | **FIXED (Jun 16) for oort + refl** — the VALUE (selection round == `MODEL_VERSION`) was always correct and matches both refs' `time_stamp` (engagement-round, which in sync == selection round). The bug was the *write timing*: the aggregator wrote it at **commit**, making a candidate's visible value depend on commit ordering (sim sct-regular vs real FIFO-jittery). Fix: stamp at **selection** in the selector (`oort.py::_record_last_selected_round`), remove the commit-write. Guard `TestLastSelectedRoundStamp`. **felix not yet done** — same commit-write at `asyncfl/top_aggregator.py:516`, separate `AsyncOortSelector`; deferred (mind `round_nudge_type`). |
| D6 | `clip_bound` | config-driven: 0.98 paper / 0.9 fork |
| S | refl exploitation | FIXED: was deterministic top-k; now the fork's cut_off_util-augmented utility-weighted `np.random.choice` |
