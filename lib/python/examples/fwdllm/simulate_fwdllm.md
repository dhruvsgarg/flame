# FwdLLM — Real↔Sim Parity

**Scope: real↔sim parity ONLY**, for the **fluxtune / fwdllm / fwdllm_plus** baselines — at **100%
availability (syn_0)** first (Phase 1), then unavailability (Phase 2), then beyond syn_0 (Phase 3). Everything
else about the fwdllm build — how it differs structurally from async_cifar10, the baseline matrix, the phased
roadmap, the JVP compute/perf profile, the sim receive/barrier redesign, the NPU delay-factor calibration, and
open (non-parity) design decisions — lives in **[FWDLLM_DESIGN.md](FWDLLM_DESIGN.md)**. This doc mirrors
[async_cifar10/PARITY.md](../async_cifar10/PARITY.md)'s focus and discipline, scoped to fwdllm; that doc owns
the shared parity methodology (ladder, roles/tiers/gating, run-length budget, landed sim mechanisms) and
fwdllm's rung catalog (§F) — read it first if you're new to this track.

> ## PREAMBLE — how to maintain this doc (READ BEFORE EDITING)
> **Parity only.** If what you're adding is a design decision, build-plan step, roadmap item, performance
> optimization, or calibration derivation rather than a real↔sim parity finding or fix, it belongs in
> [FWDLLM_DESIGN.md](FWDLLM_DESIGN.md), not here — don't let non-parity content creep back in.
>
> This is a **living status doc**, not a changelog. §A describes the state **right now** — rewrite it in place,
> never stack dated "UPDATE" blocks. Per-run history lives in git + the parity JSONs; the code is the source of
> truth for *what* a mechanism is. **One line per landed item** (§G: ≤20 words problem + ≤20 words fix). **Keep
> only what teaches** — a root cause or conceptual correction that would otherwise be re-litigated; drop
> mechanical/instrumentation/test-count narration once landed (git has that). When a chain of hypotheses gets
> superseded, keep only the FINAL correct one plus a one-word lesson — don't keep every wrong turn. An issue
> lives in exactly one place: OPEN (§A) xor CLOSED (§G, one liner). Never both.
>
> **Working checklist for every fix:** (a) ground every claim in a metric actually captured and diffable —
> telemetry/banked logs first, logical-determinism traces over aggregate curve-matching; (b) isolate the true
> bottleneck, not its symptom — verify claims against code, not against what a docstring/comment says it does;
> (c) design fixes from first principles at the root, no hack that moves a number without a correct mechanism;
> (d) **never launch an experiment run directly** — print the exact command and let the operator run it. Code
> edits, telemetry reads of already-banked logs, and pytest are fine unattended; (e) **always use conda env
> `dg_flame`** for any python/pytest/analyze_run.py invocation in this repo — running in the wrong env (e.g.
> `base`) silently skips deps (`sortedcontainers`, etc.) and produces misleading collection errors, not a real
> signal; (f) **new debugging telemetry ships with its plot in the same change** — a `build_*`/`emit()` field
> added without a reader in `scripts/analysis/analyze_run.py` is dark data (2026-07-13 audit found several
> rounds' worth of already-emitted phase/residence/comm telemetry with zero plots). Reuse the existing plot
> style for that data's shape (binned_line over progress for a per-round series, cdf_multi for a distribution,
> bar_plot for a per-category summary — see `scripts/analysis/plot_helpers.py`); only introduce a new plot
> shape if the telemetry is a genuinely new kind of quantity nothing existing already renders.

**Prerequisites:** [async_cifar10/PARITY.md](../async_cifar10/PARITY.md) — parity methodology (ladder,
roles/tiers/gating, run-length budget, landed sim mechanisms); fwdllm's rung catalog is PARITY.md §F.
[async_cifar10/UNAVAILABILITY_DESIGN.md](../async_cifar10/UNAVAILABILITY_DESIGN.md) — the availability substrate
fwdllm inherits via its aggregator class chain. [FWDLLM_DESIGN.md](FWDLLM_DESIGN.md) — build plan, structural
deltas, roadmap, performance work, calibration. [fluxtune_contributions.md](fluxtune_contributions.md) §8 — the
LIVE ledger for fluxtune training-stability/convergence issues (oscillation, collapse, accuracy degradation);
check it BEFORE opening a new stability investigation here (§G, 07-14 session 7).

**Comparator — discovers the latest real/sim pair per baseline and runs the shared parity battery:**
```bash
cd lib/python/examples/fwdllm/expt_scripts
python run_parity.py                       # all 3 baselines, latest pairs, confirm
python run_parity.py --baselines fluxtune   # one baseline
python run_parity.py --yes                  # skip the confirm prompt
python run_parity.py --validate             # + live-run checks (staleness/vclock_now)
```
Rung catalog: PARITY.md §F. **Not redefined there:** per-stage wall-budget instrumentation
(`drain_wall_budget`, `trainer_phase_wall_budget`, `step_timing_breakdown`, `aggregation_compute_wall`) is
ONE-SIDED (`sim<=real`) where sim should collapse a real-transport phase to ~0, DISTRIBUTIONAL where it's
genuine shared compute.

---

## §A  Current status — 2026-07-15, gated on the R1 residence-check fix (no new run this session)

**Basis:** all findings below are re-analysis of the SAME banked 07-15 2h run (`run_20260715_000843`–`_050735`)
the prior session flagged as a fluxtune regression, plus fixes landed against that banked telemetry — no new run
happened this session (per PREAMBLE (d)). Net: **two of the three original "regression" symptoms are now
root-caused with a fix landed (TIMING_OVERRUN; `r1_inflight_overlap`, root-caused THIS session after the prior
session's checker fix + REVERTED code-fix attempt); fluxtune's `sim_rate` collapse and accuracy-floor collapse
remain unexplained.** All landed items need a fresh run to validate — that run is the right next step for a new
context.

| item | status | detail |
|---|---|---|
| TIMING_OVERRUN (357×/169×/0× fluxtune/fwdllm_plus/fwdllm) | **ROOT-CAUSED, fix mechanism LANDED, NOT YET VALIDATED** | §O's margin used the fast-class MEAN (3.00s) as its reference trainer; the actual binding constraint is the class FLOOR (2.00s, 5 of 100 trainers) — every overrun traces to exactly those trainers. `training_delay_floor_s` (new hyperparameter, `FedSgdTrainer.py`) clamps the raw registry delay before dividing; 0.0 = no-op. **Refutes both prior hypotheses** (GPU contention, RNG-fix/GPU-assignment correlation) — same-GPU concurrency at overrun time ≈ random baseline, per-GPU compute distributions are uniform. Needs a run with `--delay-floor 7.0` (fluxtune) / `--delay-floor 11.0` (fwdllm+plus) to validate — see command below. |
| `r1_inflight_overlap` | **ROOT-CAUSED + FIXED, NOT YET VALIDATED against a live run** | Checker fix (prior session) traced the violation to `async_oort.py`'s reactive re-dispatch, but the version_key/triplet guard should have caught that -- it doesn't, because the eviction happens one layer BELOW it. `_sim_hold_busy_slots`'s `outstanding` set (which drives `all_selected`/`selected_ends` release) only counted `_sim_inflight_expected ∪ buffered`, and `_sim_inflight_expected` deliberately gets NO entry until a trainer's delay is LEARNED from a prior message (§M) -- so a trainer's FIRST-EVER dispatch in a run is invisible to it, and gets wiped from `all_selected` the instant ANY OTHER trainer's commit triggers a reconcile, seconds before its own grad can return. Traced to an exact instance in the 07-15 banked telemetry: trainer `...0449`, dispatched t=48.36s, evicted+re-dispatched t=59.68s, its own grad didn't commit until t=61.48s. Fixed by folding `_sim_pending_commit` (added unconditionally at dispatch, independent of delay-learning) into `outstanding`; the SEND_TIMEOUT_WAIT_S abandon path (`async_oort.py`) now also discards from `_agg_pending_commit_ref` so a genuinely-abandoned trainer doesn't stay stuck forever. See §G for the full trace + fix. 2 new regression tests (confirmed fail pre-fix, pass post-fix); 662 fwdllm/parity/selector tests pass, 0 regressions. |
| `_sim_gate_compute_cap_s` | **LANDED** (documented + derived, not architecture) | Was a blind `10.0` constant for "is a dispatched trainer still plausibly computing." Real compute time doesn't scale with registry speed_class (it's ~uniform regardless of class), so `10.0` was already thinner than observed max (10.647s, this run) for fluxtune. Now `16.0` in fluxtune's sim yamls (= observed max × 1.5), documented as baseline-derived not universal; code fallback (`10.0`) explicitly marked "not correct for any specific baseline, config should override." |
| fluxtune `sim_rate` collapse (1.82×→0.60×) + accuracy floor (both modes →~0.25) | **STILL UNEXPLAINED** | Neither TIMING_OVERRUN (narrow — ~5-9/100 trainers) nor `r1_inflight_overlap` (now fixed, effect on sim_rate/accuracy not yet measured against a live run) has been shown sufficient to explain a run-wide slowdown this large. Top priority for the next session, after validating the fixes above on a fresh run. |

### TIMING_OVERRUN — root cause and fix, ready to validate

Traced every overrun event (via the registry's `task_id`→hash mapping) against `trainer_registry.yaml`: **347/357
fluxtune and 133/169 fwdllm_plus overruns trace to exactly the same 5 trainers** (raw registry delay = 2.0s, the
literal floor of the "fast" class `[2,4]s`), zero involvement from any other speed class. §O's margin table
checked a 3.00s reference trainer (fluxtune → 6.25s budget, "comfortable +0.63s"); the actual floor trainer's
budget is 4.17s (fluxtune) / 1.227s (fwdllm/plus) — for fwdllm/plus, mean observed compute (1.215s, §O's own
number) is already ≈ the budget, a coin flip by construction, not a margin. fwdllm's own 0-overrun run is that
coin landing the other way in a different time window, not a systemic difference.

**Refutes the two hypotheses design doc §O's amendment left open:** same-physical-GPU concurrency at the moment
of each overrun (mean 0.57 other trainers active) is statistically indistinguishable from a random baseline
(mean 0.49) — not GPU contention. Per-GPU aggregate compute-time distributions are uniform across all 8 GPUs
(mean 4.2–4.4s everywhere) — not the RNG-order fix correlating dispatch order with GPU assignment either.

**Fix landed** (`training_delay_floor_s`, `flame/config.py` + `FedSgdTrainer.py::resolve_training_delay_s`):
floors the raw registry delay before dividing by `training_delay_factor`, so only the handful of floor-adjacent
trainers get a wider budget — the other ~90+ trainers' calibration to real per-sample JVP cost is untouched (a
uniform-divisor rescale would have pushed EVERY trainer's modeled delay well above that target; user call:
clamp the floor only). `--delay-floor F` wired through `run_sequential.sh` (mirrors `--delay-divisor`). Derived
values (max-observed × 1.3 safety, this run's data): **fluxtune 7.0** (budget 14.58s, was 4.17s), **fwdllm+plus
11.0** (budget 6.75s, was 1.227s). `resolve_training_delay_s` unit-tested (4 tests, `test_fwdllm_trainer_sim_
duration.py`). **NOT YET VALIDATED against a live run** — do that first in the next session:
```
lib/python/examples/fwdllm/expt_scripts/run_sequential.sh --only fluxtune \
    --mode both --delays on --delay-divisor 0.48 --delay-floor 7.0 --num-gpus 8 --max-runtime-s 7200 \
    --num-trainers 100 --c-async 30 --after parity,sanity,plot

lib/python/examples/fwdllm/expt_scripts/run_sequential.sh --only fwdllm,fwdllm_plus \
    --mode both --delays on --delay-divisor 1.63 --delay-floor 11.0 --num-gpus 8 --max-runtime-s 7200 \
    --num-trainers 100 --c 10 --min-initial-trainers 10 --after parity,sanity,plot
```
Read after: zero `[TIMING_OVERRUN]`; `cohort_sequence` SET/ORDER/CADENCE for fwdllm_plus (should now match near
100%, matching fwdllm's clean run, since the precondition that broke it this run is fixed).

### `r1_inflight_overlap` — checker fixed, root-caused, FIXED (this session)

**What changed in the checker** (`checks.py`, prior session): the old `_overlap_fraction` compared each trainer's
dispatch→COMMIT `contributor_intervals` entries, one per variance-gate ATTEMPT (every iteration, pass or fail,
per the field's actual construction — not "one per commit" as its old docstring claimed). fluxtune's
variance-gated design legitimately re-queries the same trainer multiple times within one still-open data_id
(confirmed by tracing trainer `…0416`'s full timeline: `var_bad` dispatches ARE genuine new-work requests —
"submit another round of perturbations against the current model," not passive pings), so consecutive retry
attempts were flagged as fake "overlaps." Rewrote to dispatch→RESOLVE, where RESOLVE = the `agg_round` variance
evaluation that actually consumes the contribution (pass or fail) — traced via the SAME `contributor_intervals`
field but keyed to the CYCLE, not the message. Gated to async-only (`is_async` flag on `agg_round`): naively
applying the same model to fwdllm_plus (sync, barrier-dispatch) reported a nonsensical 90% "violation" on its
real leg — sync's per-lap broadcast dispatch doesn't fit a per-trainer dispatch/resolve timeline the same way,
so it SKIPs cleanly instead. 269/269 parity tests pass (`test_parity_checks.py` + `scripts/parity/`).

**Re-run against banked telemetry: fluxtune real=0.0% / sim=19.4%.** A real, reproducible asymmetry. The prior
session's first pass (REVERTED) assumed the fix belonged at the version_key/triplet-guard layer
(`_trainer_state_dict`, async_oort.py) — that guard's OWN logic is actually correct (it keys on
`(model_version, iteration_per_data_id)`, which is frozen for the whole commit→resolve window), so touching it
would have been the wrong layer. Broke 71 tests for exactly that reason: it tried to redefine what "commit"
releases (the compute-slot guard) instead of finding the real gap.

**Root cause (traced to one exact instance in the 07-15 banked telemetry, trainer `...0449`, fluxtune sim):**
```
t=48.36s   DISPATCH #1 (weights)      trainer starts 10.4s of real JVP compute
t=59.30s   [a different trainer commits, triggers _sim_hold_busy_slots reconcile]
t=59.68s   DISPATCH #2 (var_bad)      <- VIOLATION: re-selected while #1 is still outstanding
t=61.48s   #1's grad is finally processed/committed
t=62.51s   cohort RESOLVE (var_bad, FAIL)
```
At `t=59.30`, `_handle_send_state`'s own log shows `count_ineligible: 0` and an almost-empty triplet-guard dict
— `...0449` is excluded by NEITHER guard. The reason: `_sim_hold_busy_slots` (fwdllm_aggregator.py) defines
`outstanding = _sim_inflight_expected ∪ buffered` and deletes from `all_selected`/`selected_ends` anyone not in
it. `_sim_inflight_expected[end]` only gets an entry once a trainer's delay has been LEARNED from a PRIOR
message (§M, deliberately — `test_train_staggered_unseen_trainer_gets_no_gate_entry`), so a trainer's
FIRST-EVER dispatch in the whole run is invisible to `outstanding` and gets wiped out of `all_selected` the very
next time ANY OTHER trainer's commit triggers this reconcile — long before its own first grad can possibly
return. At `c=30`/`n=100` this fires continuously throughout the run (most trainers' "first dispatch" happens
well after t=0), matching the steady ~19.4% rate rather than a one-time startup blip.

**Fix landed:** fold `_sim_pending_commit` into `_sim_hold_busy_slots`'s `outstanding` (fwdllm_aggregator.py) —
it's added unconditionally at dispatch and discarded only at actual commit, so it's delay-independent and
already the correct ledger for "dispatched but not committed." Reading its pre-reconcile value here is safe
because the commit path always discards from it immediately before this function runs for that same event. Also
fixed the SEND_TIMEOUT_WAIT_S abandon path in `async_oort.py` to discard from `_agg_pending_commit_ref` too —
otherwise, once `outstanding` depends on that set, a genuinely-abandoned (300s+) trainer would stay stuck in it
forever, undoing its own timeout reclaim. Neither the delay-gate's "no fallback" semantics
(`_sim_inflight_expected`) nor the version_key/triplet guard were touched.

2 new regression tests added (`test_unseen_delay_trainer_stays_held_until_commit` in
`test_fwdllm_sim_grad_residence.py`; `test_abandoned_end_dropped_from_pending_commit_ref` in
`test_send_timeout_frees_selected_ends.py`) — both confirmed to FAIL on pre-fix code and PASS post-fix.
`test_committed_trainer_drops_out_not_starved` updated (its hand-rolled setup now also discards from
`_sim_pending_commit`, mirroring what the real commit path always does before this function runs, since
`outstanding` reads that set as an input). 662 fwdllm/parity/selector/async_oort tests pass, 0 regressions (3
pre-existing unrelated failures in `test_config_generator.py`, stale `num_trainers==10` assertions predating the
n=100 yaml scale-up). **NOT YET VALIDATED against a live run** — the next fluxtune sim run should show
`r1_inflight_overlap` at ~0%.

### Superseded this session — for the record, so it isn't re-litigated

The prior session's TIMING_OVERRUN/residence-violation/`sim_rate`-collapse narrative (RNG-order fix
correlating GPU load, `var_bad` as a passive ping, dispatch→REPLY as the residence boundary) is WRONG in each
of those specific claims — superseded by the root causes above. One-line lesson: **when a checker rung and a
narrative both come from the same investigation, re-derive the checker's ground truth from raw telemetry before
trusting either — this session's first "fix" (excluding `var_bad`, using REPLY as the release signal) looked
internally consistent and still validated against 269 tests, and was still wrong.**

### `overhead_residual` — fwdllm/fwdllm_plus mechanism, still open (unrelated to the above, real-only `num_min_req=1` clamp)

`v1_iter_per_data_id` — sim needs more variance-gate retries per `data_id` than real. Confirmed NOT a
clock-formula defect: `max(trainer_speed)` itself is mode-invariant once iteration-count is held fixed (real
11.23s vs sim 11.04s, 1.6% apart, n=72 matched cycles) — purely a retry-COUNT divergence. The RNG-order fix
(prior session) closed this for fwdllm_plus but not fwdllm — still traces to the real-only `num_min_req=1`
clamp calling `sync_collect_and_accumulate_grads`/`_distribute_weights_sync` once per LAP (10 laps/cycle)
instead of once per cycle. **KEPT, not removed** (§G): the sync compose loop couples dispatch+collect per lap,
and bulk-collecting risks stranding messages (same hazard `drain_ready` dodges) plus breaks
`reselect_each_iteration`'s per-arrival re-selection — a real fix needs a compose-loop refactor + live
validation, not attempted blind.

### Next decisions — priority order

1. **NEXT.** Launch the two TIMING_OVERRUN-fix validation runs above (`--delay-floor 7.0`/`11.0`), which now also
   validate the `r1_inflight_overlap` fix in the same run (same fluxtune sim leg). Read for: zero
   `[TIMING_OVERRUN]`, fwdllm_plus's `cohort_sequence` recovering toward fwdllm's clean match, `r1_inflight_
   overlap` at ~0% (was 19.4%), and whether fluxtune's `sim_rate`/accuracy collapse recovers now that a
   trainer's slot is genuinely held for its full dispatch→resolve window (this was the leading candidate
   mechanism but is NOT yet proven sufficient — don't assume it explains everything before re-measuring).
2. If `r1_inflight_overlap` does NOT drop to ~0% on the fresh run: re-open with a live trace (same method as
   this session — find one violating instance, walk its dispatch/commit/reconcile timeline against `all_
   selected`/`_sim_pending_commit`/`_trainer_state_dict` state) rather than re-guessing from first principles.
3. fwdllm's `overhead_residual`/`num_min_req=1` compose-loop refactor — the one mechanism the RNG-order fix
   (prior session) didn't touch; still the fwdllm-specific blocker for Phase-1 close-out.
4. `sim_sct_ordered_drain` A/B (fluxtune only, below) — unexercised since the Bug-A gate fix; blocked on #1/#2
   above so it isn't confounded by either.
5. felix (async_cifar10) 46/46 reconfirmation (deferred repeatedly) + C1/C2 convergence at matched
   `data_id` — gates Phase 2.
6. Operator-run seeded real↔real pair (`*_seeded.yaml`s) — GPU-nondeterminism floor, now meaningful once #1/#2
   stop breaking dispatch-order determinism.
7. **DEFERRED, not dropped:** momentum (S1-S3) and fluxtune server-optimizer retry (Adam-style, not
   heavy-ball) — see `fluxtune_contributions.md` §8.2. Resume only after Phase-1 parity (design doc §E) closes.

### Accuracy reached at matched duration (7200s), across baselines and real/sim — from the SAME unfixed run, re-read after validating the fixes above

Final accuracy = mean of the last 5 evals (single-point accuracy is noisy). `C1` below is the checker's
matched-checkpoint average |diff|. Numbers below predate both fixes above (§A "TIMING_OVERRUN" and
"`r1_inflight_overlap`") — kept for reference, not as a current parity verdict.

| baseline | real final acc | sim final acc | real − sim | C1 `avg_accuracy_diff` |
|---|---|---|---|---|
| fluxtune | **0.25** ⚠ | **0.25** ⚠ | ~0.00 | 0.057 (FAIL, tol 0.05 — misleadingly close, see below) |
| fwdllm | 0.56 | 0.77 | **−0.21** | 0.071 (FAIL, was 0.097) |
| fwdllm_plus | 0.57 | 0.65 | −0.08 | 0.093 (FAIL, was 0.085) |

**fluxtune's 0.25/0.25 is the 4-way AgNews random-guess floor, not agreement** — both legs collapsed
independently (see "fluxtune `sim_rate` collapse... STILL UNEXPLAINED" above); its small C1 is an artifact of
two broken runs landing on the same floor, not fidelity. Ignore fluxtune's C1/accuracy numbers until that's
fixed and re-run.

**Direction flipped vs. the 07-14 basis** for fwdllm/fwdllm_plus: real used to end higher than sim
(+0.14/+0.15/+0.11); now sim ends higher (fwdllm −0.21, fwdllm_plus −0.08). Consistent with sim's now-larger
throughput edge post-RNG-fix (fwdllm: 161 vs 104 rounds in budget) — sim simply completes more training
progress in the same nominal window. The C1 *gap* still fails tolerance regardless of direction.

### vclock vs. wall-clock gap, across baselines

Two distinct numbers answer this, depending on which gap is meant:

**Within-sim (`sim_rate` = vclock ÷ sim's own physical wall-clock)** — "how much faster than 1:1 does the
sim compute": fwdllm 2.27× (was 2.46×), fwdllm_plus 2.80× (was 2.47×) — both still healthy speedups.
**fluxtune 0.60× (was 1.82×) — SLOWDOWN**, see "fluxtune `sim_rate` collapse... STILL UNEXPLAINED" above;
`is_speedup: False` this run.

**Cross-mode (`wall_disparity`, DIAG) — cumulative `|real-genuine-elapsed −
sim-vclock|` at matched progress units**, from the first matched unit to the
last:

| baseline | mean cumulative gap | max cumulative gap | matched units | vs. 07-14 basis |
|---|---|---|---|---|
| fluxtune | **1706s** | **3207s** | 161 | ~3× worse (was 614s/1305s) |
| fwdllm | 93s | 338s | 104 | improved (was 140s/394s) |
| fwdllm_plus | 220s | 509s | 106 | ~1.5× worse (was 143s/385s) |

fluxtune's cross-mode gap tripling and fwdllm_plus's moderate worsening both track the TIMING_OVERRUN
finding above (357×/169× fires respectively); fwdllm's improvement tracks its 0 overruns + clean
`cohort_sequence` match. Consistent single story across every rung, not independent noise.

**Old fluxtune-specific detail (Bug A / reactive gate) is CLOSED — see §G.**

### fluxtune: `sim_sct_ordered_drain` A/B — open, priority raised

Schema default is `false` (`config.py:245` = legacy `recv_fifo`); felix (async_cifar10) never sets it, so
uses that default. **All of fluxtune's yamls override it to `true`** (`drain_ready`, comment-justified:
fluxtune's higher per-grad probe frequency vs felix's per-round one risks the shared queue stranding a
delivered update the readiness probe can't see) — not evidence-justified, since no run has been made with
it `false` since the Bug-A gate fix landed. A stranded message under `drain_ready` and the confirmed
`r1_inflight_overlap` residence violation are adjacent hazard classes (both about a grad/dispatch getting
lost-or-duplicated relative to the reorder buffer) — worth ruling in/out together once that's fixed.
`fluxtune_n10_smoke_sim_no_sct_drain.yaml` exists for the A/B
(`sim_sct_ordered_drain: false`, otherwise identical to `fluxtune_n10_smoke_sim.yaml`); re-run at the
current 7200s+/factor-0.48 basis, not the stale 1800s/factor-0.5 one — and only after both TIMING_OVERRUN
(validated) and `r1_inflight_overlap` (fixed) land, so this A/B isn't confounded by either:

```
python -m flame.launch.run_experiment \
    lib/python/examples/fwdllm/expt_scripts/fluxtune_n10_smoke_sim_no_sct_drain.yaml
```

Read after: `sim_rate` (does it recover off the 07-15 0.60× low?); `r1_inflight_overlap` (does it move off
19.4%?); `pastdated_n`/`pastdated_gap_max` in `[SIM_GRAD_RECV]` (materially higher with the flag OFF means
`recv_fifo`'s background streamer really does strand/lap messages under fluxtune's probe frequency — the
flag's actual justification).

(Roadmap after this closes: see "Next decisions" above.)

---

## §F  Locked principles (from async_cifar10, carried over)
1. **Sim does real forward-grad compute, charges modeled time.** GPU runs the real JVP; the agg stamps
   `sct = sim_send_ts + max(real_gpu_s, D) + leg`. **Never** put overhead on the vclock (`vclock = max(vclock, sct)`).
   **RESOLVED 07-14 (session 7, §A/§G item 2)** — session 6 questioned whether the "never" is too broad for
   genuinely non-trivial aggregator work. Settled: fwdllm's ~10s/cycle aggregator overhead is compose-loop-
   coupling + (now-trimmed) logging volume — real-harness implementation waste, not genuine FL work — so
   engineering it away (already done for the logging half) is the right direction, not folding it onto the
   vclock. "Never put overhead on the vclock" stands as originally written.
2. **Progress axis is `data_id`.** Updates-per-data_id is the dynamic-K random variable — an output to match,
   not an input to assume.
3. **Variance is an emergent gate; localize, never tune it.** `var_threshold` / `max_iterations_per_data_id`
   are baseline-defining config knobs.
4. **Slot residence must survive variance rollbacks.** Release keys on the agg-goal boundary; the sct reorder
   buffer must not strand a grad across a rollback.
5. **Real is the reference only after admissibility.** A real↔sim gap has two fix directions — check whether
   the **real** input is the divergent side before tuning sim.
6. **Fix the concept, not the symptom.** Classify a mechanism as **real-transport artifact** (no sim analog,
   gate `and not self.simulated`) vs **algorithmic property**. Scope-check before editing shared code:
   `fwdllm_aggregator.py` = fwdllm blast radius; `top_aggregator.py` / shared parity engine / `_sim_recv_min`
   can silently break async_cifar10.
7. **Match pytest scope to blast radius.** fwdllm-only edit → `pytest tests/mode -k fwdllm`; shared parity
   engine → add `examples/async_cifar10/scripts/parity` + `tests/mode -k parity`; shared stack → full `pytest tests/`.
8. **Telemetry-FIRST, then instrument, then (rarely) run.** Validate/refute from telemetry ALREADY ON DISK
   first — name the exact field/line. Ship telemetry + plot + pytest IN THE SAME CHANGE as any new mechanism.
   A run is justified only to observe an EMERGENT quantity no stored telemetry can yield.
9. **Consult PARITY.md vclock rules BEFORE any sim-clock change.** Clock is a monotone `max`; NEVER put
   overhead on it; the sim SKIPS real waits and reconstructs order from sct (`SimReorderBuffer`).
10. **The vclock is virtual wall-time; the sim MUST produce SPEEDUP (`sim_rate = vclock/wall ≥ 1`).** The
    forward-grad "train" pass is the only irreducible real wall (parallel across trainers); transport/
    inter-round/delay waits are vclock jumps, never process sleeps. `sim_rate < 1` means the sim is stalling
    on a real wait it should skip, OR (§A, fluxtune) its per-commit processing throughput can't keep pace with
    arrivals — check BOTH before assuming it's a wait-modeling gap.
11. **Correctness before speed; SHARED roots before per-baseline.** Fix major logical-correctness divergences
    before any throughput/wall tuning. A bug that fails rungs across ≥2 baselines outranks a single-baseline one.
12. **Logical determinism is the parity definition.** For a matched scope the sim must take the SAME sequence
    of steps in the SAME order as real — same trainers selected, same order of update receipt, same
    aggregations and rollbacks — differing ONLY in wall-clock. Prove this on the first data bin before
    extending length.
13. **Do the right thing — no hacks.** A hack that moves a number without a correct mechanism is a regression
    in disguise. When unsure, stop and ask.
14. **`version_key` is the ONLY version-identity vocabulary.** 2-tuple: base `(round, 0)`, fwdllm
    `(model_version, iteration_per_data_id)`. Any new code comparing/stamping a version goes through it — no
    bare-scalar shortcut "for now."
15. **Verify claims against code, not documentation/comments.** A docstring saying two functions are
    "the analog of" each other is a claim about intent, not a guarantee of behavioral equivalence — diff them
    (§A, fluxtune vs felix).
16. **Don't blame GPU/resource contention at n=10** — checked and refuted once already; won't apply until
    ≥100-trainer scale. Any unexplained real-wall gap should be assumed closeable by measurement (a wall-clock
    + vclock phase timer around the suspect stage), not guessing.
17. **A bounded rotating in-flight cohort settling at `c − agg_goal` surplus is the correct steady state for
    a `c ≫ agg_goal` fedbuff pool, not a backlog to eliminate.** Total concurrency is held constant by
    construction: a boundary that closes on `agg_goal` commits frees exactly `agg_goal` slots and dispatches
    exactly `agg_goal` replacements, so surplus fixed-points around `c − agg_goal` (matches fluxtune's own
    measured `buf_depth` sitting at 7-8/10 for `c=10, agg_goal=3`, exactly). `carried_surplus_commits` will be
    the MAJORITY commit-source bucket in steady state (~70% at fluxtune's ratio) — don't drive it toward 0;
    only `pastdated_commits` (genuine scheduling anomalies, distinct bucket) should read ~0.

---

## §G  Landed fixes, refuted hypotheses, and deviations — durable lessons only

*(Collapsed from the former §G/§H/§K. Superseded hypothesis chains keep only the final correct answer + a
one-word lesson; pure scaffolding/telemetry-only entries dropped — git has that record. Full history:
`git log -- lib/python/examples/fwdllm/simulate_fwdllm.md`.)*

**TIMING_OVERRUN — §O's margin checked the fast-class MEAN, not the FLOOR (07-15) — ROOT-CAUSED + FIX
LANDED, not yet validated.** 347/357 fluxtune and 133/169 fwdllm_plus overruns traced (via `trainer_registry.
yaml`'s `task_id`) to exactly the same 5 trainers (raw delay = 2.0s, the class floor); §O's margin table used a
3.00s reference trainer. Refuted both live hypotheses first: same-GPU concurrency at overrun time ≈ random
baseline (not contention); per-GPU compute distributions uniform across all 8 GPUs (not RNG-fix/GPU-assignment
correlation). Fix: `training_delay_floor_s` (new hyperparameter) floors the raw registry delay before dividing
— only the floor-adjacent trainers get a wider budget, the other ~90+ trainers' calibration to real per-sample
JVP cost is untouched (a divisor-wide rescale would have moved everyone's delay well above that target). Wired
through `run_sequential.sh --delay-floor`. 4 unit tests (`resolve_training_delay_s`). See §A for the
not-yet-run validation command + derived per-baseline values (fluxtune 7.0, fwdllm+plus 11.0).

**`r1_inflight_overlap` checker used the wrong "commit" signal (07-15) — FIXED, reveals a real unfixed sim
bug.** Old check compared dispatch→COMMIT intervals from `contributor_intervals`, which includes every
variance-retry attempt (pass or fail) — false-positived on fluxtune's legitimate multi-iteration resampling of
the same trainer. Two WRONG intermediate "fixes" attempted and rejected first (dispatch→REPLY as the release
signal; excluding `var_bad` as a "passive ping") — both looked internally consistent and passed tests, both
were refuted by tracing one trainer's full dispatch/reply/eval timeline end to end (`var_bad` IS new work;
real always resolves before its next dispatch, sim doesn't). Final fix: dispatch→RESOLVE (the `agg_round`
variance evaluation that actually consumes the contribution), gated async-only (`is_async` — sync's barrier
dispatch reported a nonsensical 90% on fwdllm_plus's real leg when naively applied). Reveals fluxtune
real=0.0%/sim=19.4%. 269/269 parity tests pass on the checker fix.

**`r1_inflight_overlap` root-caused + FIXED (07-15, this session) — a compute-slot bug, not a version_key
bug.** The first fix attempt (REVERTED, prior session) assumed the version_key/triplet guard
(`_trainer_state_dict`) needed a third state; wrong layer — that guard is correct (keyed on
`(model_version, iteration_per_data_id)`, frozen for the whole commit→resolve window) and was never reached,
because the eviction happens BELOW it. Traced to one exact instance (trainer `...0449`, fluxtune sim,
07-15 telemetry): `_sim_hold_busy_slots`'s `outstanding = _sim_inflight_expected ∪ buffered` gives NO entry to a
trainer's FIRST-EVER dispatch in a run (`_sim_inflight_expected` deliberately withholds an entry until a
trainer's delay is LEARNED from a prior message, §M) — so it's wiped from `all_selected` the instant ANY OTHER
trainer's commit triggers this reconcile, seconds before its own grad can return. Fix: fold `_sim_pending_
commit` (added unconditionally at dispatch, delay-independent, discarded only at actual commit) into
`outstanding`; also made the SEND_TIMEOUT_WAIT_S abandon path (`async_oort.py`) discard from `_agg_pending_
commit_ref`, else a genuinely-abandoned trainer would stay stuck in it forever once `outstanding` depends on
it. 2 new regression tests (fail pre-fix, pass post-fix); 662 fwdllm/parity/selector tests pass, 0 regressions.
**NOT YET VALIDATED against a live run** — see §A for the validation command.

**`_sim_gate_compute_cap_s` was a blind constant, not derived (07-15) — LANDED (config + docs, not
architecture).** `10.0` (hardcoded default) was already thinner than fluxtune's observed max real compute
(10.647s, 07-15 run) — real GPU compute time is roughly uniform across trainers regardless of registry
`speed_class` (unlike the modeled delay `D`), so this needed the same "derive from observed data" treatment as
the delay-floor fix above, not a runtime-adaptive rewrite (too risky to ship untested this session). Fluxtune's
sim yamls now set `16.0` (= observed max × 1.5), documented as baseline-derived; code fallback stays `10.0`
labeled explicitly as "not correct for any specific baseline."

**`select_random` dispatch order leaked through PYTHONHASHSEED (07-14) — FIXED.** Seeded RNG sample was correct
(same trainer set) but wrapped in a bare `set()` before return, so iteration order followed per-process string
hashing, not the seed. Fix: `set()` → `dict.fromkeys()` in `async_oort.py`/`oort.py`/`async_random.py`; new
cross-process order test. 193/193 selector tests pass. **VALIDATED 07-15** on a fresh 2h re-run: fwdllm's
`cohort_sequence` SET/ORDER/CADENCE hit 100% match (was drifting); fwdllm_plus's `overhead_residual`/
`per_round_advance`/`v1_iter_per_data_id` flipped FAIL→PASS. Also surfaced that dispatch-order identity alone
doesn't guarantee `overhead_residual` parity (fwdllm's stayed open/worsened — separate real-only
`num_min_req=1` clamp mechanism, above) and that TIMING_OVERRUN — not this fix — governs whether the identity
holds at all (§A); see §A for the 07-15 regression this run also surfaced in fluxtune.

**fluxtune `preferred_duration` (oort utility-binding gap) — FIXED, VALIDATED 07-15.** Real's utility term was
binding selection 80.1% of rounds vs sim's 29.3% (50.7pp gap). Root: the earlier-landed oort pacer fix
(faithful both-branch `OortSelector.pacer()` port) closes it — confirmed on the 07-15 2h re-run: `frac_diff`
50.7pp→9.3pp (in the 20pp tolerance), `selector_score`'s `believed_I` component KS 0.662→0.04.

**Parity-CLI progress-axis bugs (Bug C, 07-14) — FIXED.** `_progress_axis()` chose round-vs-data_id per side
independently, and raw `cycle_data_id` collided across laps; batch-CLI glob matched `fwdllm_plus` for `fwdllm`
too. Fix: prefer `data_id` whenever present; composite `(round, data_id)` key; glob anchored on `_{tag}_n<N>_`.
517 tests pass (`tests/mode` + `scripts/parity/`).

**Aggregator `step_timing` was parsed nowhere (07-14) — FIXED.** Aggregator's own per-function overhead was
emitted but read by zero checks. Fix: loader captures it; new `agg_step_timing_breakdown` DIST rung (DIAG,
no real-only exemption). 118+401 tests pass.

**fwdllm had no seeded yaml (07-14) — FIXED.** Seed plumbing (selector RNG via `ChannelManager.join`, per-client
JVP `torch_rng`) was already correct, just never exercised, and a comment wrongly said otherwise. Fix: 6 new
seeded yaml pairs (one per baseline) + corrected the comment. 549 tests pass.

**Real-only `num_min_req=1` clamp (07-14) — KEPT, no code change.** Sync compose loop couples dispatch+collect
per lap; bulk-collecting risks stranding messages and breaks per-iteration reselection. Refactor needed, too
risky to attempt without a live run.

**`recv_fifo` hot path logged at INFO, ~10x/cycle (07-14) — FIXED.** 425k lines / 112MB per run, unread by any
check or plot. Fix: downgraded 9 mechanical trace lines to DEBUG in `channel.py`; anomaly warnings kept.
Full `lib/python/tests/` suite (shared code) passes.

**`overhead_residual`'s `max(trainer_speed)` divergence (07-14) — RESOLVED, no code change.** Held iteration-
count fixed on banked telemetry: real/sim means differ only 1.6% (n=72 matched cycles). Gap is retry-COUNT,
not the clock formula — principle #1 stands.

**fluxtune accuracy drop — ALREADY ROOT-CAUSED (07-14), cross-referenced.** `fluxtune_contributions.md` §8
already refuted the round-reset hypothesis (F10) and found the real cause (undamped SGD + frozen bin order,
F1-F15). No new fix; check that doc before re-opening fluxtune stability questions here.

**Server-momentum (S1) landed, flag-gated, not yet run (07-14).** F8's undamped direct-SGD update is shared
code across all 3 baselines (confirmed via log grep), so no baseline should be structurally disadvantaged by a
fluxtune-only fix without justification. Fix: `_server_update_step` (heavy-ball momentum,
`hyperparameters.server_momentum`, default 0.0 = byte-identical) in `FedSgdAggregator.aggregate()` — 5 unit
tests. Short (8min) real-only A/B pairs for **all 3 baselines** (`{fluxtune,fwdllm,fwdllm_plus}_n10_smoke_
short[_momentum].yaml`, momentum=0.9 vs 0), all 12 short/sim/momentum yamls now seeded (`seed: 1234`, same both
legs, no separate seeded run needed) — plus 2 node-split launcher scripts
(`run_momentum_ab_{without,with}.sh`). fwdllm's own telemetry shows the same collapse signature early in round
1 (not yet confirmed position-locked like fluxtune's) — fluxtune-only is NOT settled; the A/B pairs exist for
all 3 baselines specifically so this is testable, not assumed. 262 tests pass (`tests/mode -k fwdllm`).

**Reactive gate blocked real wall (Bug A, 07-13) — FIXED, VALIDATED 07-14.** Gate re-checked `earlier_stuck`
after an unconditional blocking call even when already safe. Fix: `_sim_gate_is_safe` checks first, near-zero
timeout when safe. `sim_rate` 0.97×→1.82× confirmed on re-run.

**Carried-surplus commits misclassified as "round1" (Bug B, 07-13) — FIXED.** felix's round-axis classifier
wasn't re-keyed to fwdllm's `data_id` axis. Fix: ingest-time stamp detects carry-over, buckets it separately.

**`eval_model()` backgrounded (Part 6, 07-13) — FIXED.** Sync eval stalled dispatch; naive backgrounding would
race on shared `self.fmodel/params/buffers`. Fix: deleted the dead assignment, backgrounded on a daemon thread.

**fwdllm_plus livelock — FIXED, validated 07-13.** `RandomSelector` freed only `k=5` of a `c=10` cohort per
cleanup (stale batch-size knob). Fix: removed `k` from the selector entirely, not retuned.

**fluxtune commit-path stall — three superseded framings, final root: phantom `_sim_inflight_expected` entry**
(stamped-expected but idle, not computing). Fix: `sim_compute_truthful_gate` skips stale-dispatch entries.
Superseded again 07-13: not the actual `sim_rate<1` driver (see Bug A).

**fluxtune cohort-SET divergence (#S1) — FIXED.** Real released a busy trainer's re-pick guard on RETURN not
commit; `async_oort`'s abandon had no liveness check. Fix: hold-to-commit + `send_timeout_wait_s=300`.

**`version_key` unification — FIXED.** Version identity was a bare int in some places, a 3-tuple in others.
Fix: one shared 2-tuple property across trainer/aggregator/selector, both examples.

**Remainder-wait delay model — FIXED.** Additive `send+gpu+D` gave nondeterministic arrival order. Fix: real
sleeps `max(0,D-gpu)` so device wall=D; sim never sleeps D (vclock-only).

**Slot residence: hold to COMMIT — FIXED.** Release-on-RETURN undercounted in-flight state 3×. Fix: hold the
slot until commit, for both sync and async.

**Async surplus-grad handling — FIXED.** Dropping at the agg-goal boundary was fine for sync, wrong for
`c≫agg_goal` async (fluxtune dropped ~7 grads/cycle). Fix: carry the surplus + hold busy trainers.

**Checker/telemetry-only fixes (not mechanism bugs):**
- Async `total_commits`/`throughput` summed overlapping cycles as sequential → 76-86% spurious diff. Fix: fall
  back to raw wall for async.
- Clock-rate rungs anchored on full wall (localhost transport artifact) instead of `intrinsic_span_s`.
- `cohort_sequence_parity` conflated SET/CADENCE/VAR/ORDER under one cap, tripping on real GPU fp16 jitter.
  Fix: SET stays hard/uncapped; CADENCE/VAR/ORDER cap to bin 1, distributional beyond.
- "GPU under-provisioned at n=10" — refuted by the spawn table (balanced round-robin, 8 GPUs, 1 core/trainer).
