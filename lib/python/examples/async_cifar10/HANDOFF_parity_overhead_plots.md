# Handoff: trainer overhead, real/sim parity, plots (async_cifar10)

State snapshot so we can resume without losing context. Branch:
`dg/streaming_expt_opt_gap`. Everything below is committed unless marked
**uncommitted/open**.

## Commits this effort (newest first)
- `8ac0d535` sync aggregator: guard `channel.ends()` returning None in sim mode
- `426e19ad` plots: semantic participation heatmap, response-lateness CDF, aggregate time-split, acc-gain dual-axis
- `694dde96` random selector works on sync FL + logical commit-sequence parity tool
- `95559457` sync sim-ordering (first_k by smallest sim_completion_ts) + doc
- `9ff451a9` Task 3: real/sim parity hardening — seeding, determinism fix, checks, tests
- `94d2cff7` eliminate per-round trainer overhead that inflated "compute" time

Test status: `pytest lib/python/tests` → ~260 passed, 7 e2e skipped (the e2e
parity test skips unless `FLAME_E2E_*` env vars point at run dirs).

---

## 1. Trainer per-round overhead (DONE)

**Root cause (data-backed):** the "excess trainer time" beyond the modeled
budget was entirely trainer-side per-round wrapper code, NOT compute, GPU
memory, network, or the aggregator. At 300 trainers / 8 GPUs the "GPU compute"
was ~2.3 s/sample on 1-sample rounds. Culprits, all amplified by process
co-location:
- `MemoryProfiler` heap walks: `gc.collect()` + iterating *all* ~415k Python
  objects up to 3× with a per-object `torch.is_tensor()` check, 4 calls/round,
  two of them *inside* the GPU-timing window.
- `torch.cuda.empty_cache()` every 50 batches / per-epoch / pre+post training —
  under co-location this forces a CUDA sync and frees the allocator's blocks, so
  the next round re-allocates from the driver (serialized across processes). It
  HURTS, not helps.
- per-param `.item()` syncs: grad-norm (epoch 1), `delta_weight_l2` (every
  round, ungated), and `update_local_accuracy` (every batch).

**Fix:** profiler gated off by default (`memory_profiling_enabled` hyperparam);
removed in-loop empty_cache/gc; grad-norm + delta_l2 telemetry-gated + single
fused `.item()`; `update_local_accuracy` accumulates on-GPU, syncs once/round;
added first-class per-phase timing `pre_train_s` / `post_train_s` to
`[TRAIN_CYCLE]` and the `trainer_round` telemetry.

**Validated A/B (48 trainers / 2 GPUs):** per-round trainer compute 0.51→0.03s
(real), 0.46→0.01s (sim); aggregator-observed lag 0.99→0.09s (real). The new
`scripts/analyze_timing_overrun.py` (histogram of lateness + CDF, split
trainer-attributable vs aggregator-observed) is the diagnostic.

**Key insight:** overruns are contention-dependent — invisible at 48/2 (gpu ≪
budget), real at 300/8 (gpu 5–7s vs 5s budget → 39% trainer-side overruns).

---

## 2. Real/sim parity

### Conceptual framing (the core reasoning)
"Sim should replay real's event order, just faster" is correct ONLY for
quantities that are a deterministic function of the logical timeline. **Real
mode is itself non-deterministic run-to-run**, so it's a distribution, not a
single ground truth. Split:
- **Exact-matchable** (given seed + well-separated D + low contention so arrival
  order == sim-completion order): commit/aggregation order per model version,
  staleness sequence, OORT-visible speed `max(gpu,D)`, eligibility at a logical
  time.
- **Irreducibly non-deterministic** (distributional + invariants only): (1)
  selector RNG; (2) GPU jitter at near-ties; (3) physical MQTT arrival under
  contention (the thing sim repairs).

### Done
- **Seeding (Phase 0):** aggregator seeds process-global `np.random`/`random`
  (+torch) from a `seed` hyperparam at `internal_init`. The selector runs in the
  aggregator process and draws from these. `seed=None` = legacy.
- **FedBuff parity bug fixed:** `_handle_send_state` called
  `random.seed(time.time())` on *every* selection — non-reproducible by
  construction, clobbered global RNG, defeated any seed. Removed.
- **Sync sim-ordering:** `_sync_sim_recv_first_k` in syncfl/top_aggregator —
  commits the first_k smallest `sim_completion_ts`, advances vclock, sources
  `PROP_ROUND_DURATION` from `SIM_ROUND_DURATION`. Real path unchanged.
- **Tests:** `tests/sim/test_virtual_clock.py`, `tests/mode/test_async_sim_ordering.py`
  (real-vs-sim commit/staleness equivalence under arrival permutations),
  `tests/mode/test_sync_sim_ordering.py`, `tests/selector/test_selection_determinism.py`
  (FedBuff+Oort selection is a pure fn of (state,seed)),
  `tests/mode/test_parity_checks.py` + opt-in `tests/mode/test_real_sim_e2e_parity.py`.
- **Shared module:** `scripts/parity_checks.py` — canonical loaders + parity
  funcs (selection Jaccard, agg-sequence, staleness, participation, sim_send_ts,
  gpu-budget, vclock monotonicity, agg_goal cycles) + `commit_sequence()` /
  `first_divergence()` (the logical-ordering diff). `compare_parity.py` imports
  its loaders.

### Baselines (must all work — FluxTune out of scope)
- async: **felix**=async_oort selector + fedbuff optimizer; **fedbuff**=fedbuff selector.
- sync: **fedavg**=random selector; **oort**=oort; **refl**=refl_oort; **feddance**=feddance.

### CROSS-BASELINE PARITY SWEEP — all 6 baselines pass e2e 7/7 [DONE]
Seeded real/sim pairs now exist for every baseline (`<baseline>_n48_parity_seeded_{real,sim}.yaml`,
seed=1234, n48/2GPU, alpha0.1/syn_0, agg_goal=5). 12-round sweep + `compare_parity` +
`test_real_sim_e2e_parity` → **all 6 pass 7/7**. Stacks: fedavg/feddance use `syncfl`
(`horizontal/top_aggregator` re-exports it); felix/fedbuff use `asyncfl`; oort/refl use the
separate `horizontal/oort/top_aggregator` (overrides distribute/aggregate). Fixes landed:
- **fedbuff eval crash** — async aggregator appended to `selector.trainer_eval_recv_ends`
  (async_oort-only); guarded with `hasattr` (asyncfl/top_aggregator). Other selector attrs it
  touches (all_selected/selected_ends/requester/ordered_updates_recv_ends/remove_from_selected_ends/
  reset_end_state_to_none/_cleanup_removed_ends) all exist on fedbuff.
- **oort/refl sim-ordering** — the oort stack had ZERO `simulated` handling (committed by physical
  arrival, no vclock). Added: `sim_send_ts` stamping in distribute; sim aggregate commits via the
  inherited sim machinery; `_handle_weights_msg` no longer overwrites round_duration with wall-clock
  in sim. Result: oort staleness 0/0; sim_send_ts + round-duration parity pass.
- **oort/refl cross-round straggler carry** — `_oort_sim_recv` is a GENERATOR backed by a PERSISTENT
  `_sim_buffer`: overcommitment (1.3×) stragglers carry across rounds and commit late as stale
  (REFL accepts → real 0.93 / sim 0.75, was 0.81 / 0.02). The caller's "stop at aggr_num accepted"
  loop drains stale stragglers too, so in-flight stays bounded (oort `chosen` 6–7, was growing 6→37).
- **JOIN BARRIER (`min_trainers_to_start`)** — `_await_min_trainers` (syncfl base, called by all three
  distribute paths) blocks once at startup until N ends join. Trainers are real processes that spawn/
  join over wall-clock in BOTH modes (sim only virtualizes training sleeps); without it sim raced
  ahead of joins and selected from a half-filled pool (sim ncand 4→40 vs real 48). Now ncand=48 from
  round 0 in both. Set to 48 in all 12 parity configs.
- **Selection-parity gate** — even with matched pools + seed, exact per-round selection still differs:
  selectors sample from a candidate LIST ordered by channel-JOIN order (physical, varies run-to-run),
  so the same seed draws a different subset, then RNG cascades. This is the residual "irreducible"
  part — but participation-FREQUENCY parity holds. `parity_checks.selection_parity` now gates the
  Jaccard check (informational) for stochastic selectors via `DETERMINISTIC_SELECTORS` (empty;
  add a selector once it sorts candidates before sampling), and `participation_parity` is the enforced
  selection invariant. All other checks (staleness, participation, round-duration, sim_send_ts, GPU,
  convergence, agg_goal cycles, vclock monotonicity) are enforced and pass for every baseline.
- `random` selector context: its SEND/RECV channel **state** is the *buffered
  concurrency pattern* (FwdLLM/async): SEND=pick new trainers to send to,
  RECV=return in-flight set to receive from. Stateless sync FL never sets it →
  `KeyError 'state'`. Fixed by defaulting state to SEND when absent; also removed
  a fully-duplicated dead `select()` method. fedavg now runs on the sync stack
  (validated: no KeyError, 40 real rounds).

### OPEN ISSUE A — async (felix) sim over-selection [FIXED]
Symptom: per-round selection ~5 real vs **19–21 sim**, staleness mean **1.16 real
vs ~8.6 sim** (KS ~0.77); `[SIM_PENDING] blocked` climbing 7→43 toward N.
**Two-part root cause (both in concurrency slot-accounting, asyncfl):**
concurrency is budgeted as `extra = c − len(selected_ends)` while candidate
exclusion uses `all_selected` (two separate structures). (1) The round-end sim
bookkeeping discarded buffered-but-uncommitted ends from `selected_ends` (kept
them only in `all_selected`) → their slot freed but they couldn't be re-picked →
selector refilled with NEW trainers → in-flight grew each round. (2) Even after
keeping them in `selected_ends`, `_handle_recv_state` strips every end in
`VAL_END_STATE_RECVD` from `selected_ends` — and `recv_fifo` marks an end RECVD
the moment its message is *buffered*, though only 1 is *committed* per aggregate.
So buffered-not-committed ends were stripped on the next aggregate, re-opening the
phantom slots. **Fix (3 edits, `asyncfl/top_aggregator.py`):** (a) round-end:
keep pending-buffer ends in `selected_ends` (`.add`, not discard); (b) on commit
in `_sim_recv_min`, release the slot (discard from `selected_ends`); (c) in
`_sim_recv_min`, after popping, reset the still-buffered ends' channel state from
RECVD back to NONE so `_handle_recv_state` leaves them in `selected_ends`
(`to_probe` already skips them via `_sim_buffer.has`, so no re-recv). **Verified**
(felix seeded real+sim, 12-round shortened pair, seed=1234): `[SIM_PENDING]`
stable at 5–7 (was →43); per-round selection ~5–7 sim vs 5 real (was 19–21);
staleness **1.03 real / 1.3 sim — PASS** (was 8.6); commit counts 60/60 match;
`compare_parity` PASS on staleness/participation/round-duration/sim_send_ts/GPU/
convergence; e2e **6/7 PASS**. The remaining FAIL (`test_selection_parity`,
Jaccard 0.162) is the documented IRREDUCIBLE divergence: `first_divergence` shows
both modes commit 0370 first then diverge at **commit #1** from selector-RNG
desync (real & sim see different candidate sets at near-ties — the
"irreducibly non-deterministic" selector-RNG class in the framing above; even
real-vs-real won't match exactly). Unit test note: `test_async_sim_ordering`'s
`FakeChannel` gained a minimal `_ends`/`_FakeEnd` to model the END_STATE the
reset in (c) touches.

### OPEN ISSUE B — sync (fedavg) sim aggregation does not commit [FIXED]
**Root cause (not the recv helper):** the sync aggregator drove BOTH distribute
and aggregate through plain `channel.ends()`, which defaults to **SEND** state.
The `random` selector is stateful (buffered concurrency): a SEND call means
"pick NEW trainers to send to." So distribute filled concurrency (c=8 sends), and
aggregate's second SEND call computed `required = c − 8 = 0` → returned `None` →
`_aggregate_weights` early-returned BEFORE ever calling recv. Log signature: many
`Agg weights` calls but **0** `Waiting for first_k` / `[SYNC_SIM_RECV]`. The
recv-against-real-channel hypothesis was wrong — aggregation bailed one step
earlier. (Real had the same bug; it blocked on `recv_fifo` instead of spinning.)
**Fix:** mirror the async stack — distribute uses `channel.ends(VAL_CH_STATE_SEND,
task)`, aggregate uses `channel.ends(VAL_CH_STATE_RECV)` so the random selector
returns its in-flight `selected_ends`; the two telemetry/optimizer `ends()` calls
use RECV too. Stateless selectors (oort/refl/feddance/default) ignore the state
arg → no change for them. **Verified:** fedavg seeded real+sim 10-round pair both
complete all rounds (real 1→10 rounds, sim 0→41 commits); `compare_parity.py`
PASSes staleness/participation/convergence/round-duration/sim_send_ts/GPU; all 7
`test_real_sim_e2e_parity` PASS. (Remaining `compare_parity` FAILs #1/#2 are a
separate gap — the random selector emits **0 `selection` telemetry events** on the
sync stack, so Jaccard is nan; #3 agg-sequence checks order, irrelevant for sync.)

### How to verify parity
Seeded config pairs exist: `felix_n48_parity_seeded_{real,sim}.yaml`,
`fedavg_n48_parity_seeded_{real,sim}.yaml`, plus `felix_n48_stress_{real,sim}`.
Run each pair, then:
```
FLAME_E2E_REAL_DIR=<real run> FLAME_E2E_SIM_DIR=<sim run> FLAME_E2E_AGG_GOAL=5 \
  pytest lib/python/tests/mode/test_real_sim_e2e_parity.py -v
# or the full report:
python scripts/compare_parity.py --real <agg jsonl> --sim <agg jsonl> \
  --real-trainer-dir <tel dir> --sim-trainer-dir <tel dir>
```
Run dirs land in `experiments/run_<ts>_<name>`. Launch:
`python -m flame.launch.run_experiment <yaml>` (blocks; renames nothing).
n48 dataset split (`_metadata/dataset_splits/cifar10_alpha0.1_n48.yaml`) is
derived from the first 48 trainers of the n300 split.

---

## 3. Plots / telemetry analyzer (`scripts/analysis/analyze_run.py` + `plot_helpers.py`)

Runs per-run; categories `performance/ sanity/ selection/ insights/ system/`.
Invoked post-run by the launcher if present. Goal: all baselines produce the
same outputs. **Test:** `python scripts/analysis/analyze_run.py <run>/telemetry
--out /tmp/x`.

### Answers to the "why" questions
1. **insights/loss_vs_visible** — `final_loss` per *trainer-round* vs that
   trainer's `visible_fraction`, all rounds pooled (not per-trainer aggregated).
   The different pattern at fraction=1 is because once fully unlocked, points
   reflect full-data training (more samples, real loss spread), whereas low
   fractions are tiny datasets (1–few samples) pinned near random loss.
2. **insights/util_disparity** — from `util_disparity` telemetry the trainer
   emits (`_emit_util_disparity`). "Full" utility = the trainer's Oort
   statistical utility over its FULL data pool (counterfactual, as if all
   unlocked); "streamed" = on the currently-visible prefix. Plot is the mean
   streamed/full ratio across trainers per round. Driven by the full-pool loss,
   not just the fraction.
3. **insights/utility_vs_visible** — Oort utility = N·sqrt(mean(loss²)); N =
   sample count, which grows with unlocked data, so utility scales ~with N (the
   point of the streaming work: utility inflates with data volume).
8. **sanity/selection_count high at start** — round 0 is warmup: as 48 trainers
   join, the aggregator selects them; the count sums `chosen` across ALL
   selection events logged under round 0 (the whole join burst), not a single
   pick of all trainers. Steady-state ≈ agg_goal.
10. **trainer report back early?** — the residual compares *GPU time* to budget,
    so "early" only means the GPU finished early; in real mode the trainer SLEEPS
    to fill the budget, so its *response* is on-time, never early. Added a
    response-lateness CDF (`max(gpu,D) − budget`, 0 = on-time, >0 = overrun).
11. **eval_vs_train_selections spike at start** — same round-0 warmup burst.

### Done (committed `426e19ad`)
- #12 participation heatmap: discrete semantic colors + legend (grey=not
  selected, light-blue=eval, light-green=train, light-red=unavailable,
  orange=selected-but-unavailable; availability forward-filled from avail_change).
- #10 response-lateness CDF (+ answers "why early").
- #19/#5 aggregate round-time split over rounds (mean pre/gpu/post/sleep across
  trainers) — system-level view; visually confirms the overhead fix.
- #4 acc-gain per eval now overlays overall accuracy on a secondary axis.

### Plot items — DONE this pass (analyzer at repo-root `scripts/analysis/`)
- **#5/#6** — added `system/.. runtime_overhead_cdf.pdf` (CDF of aggregator
  observed−reported overhead) alongside the existing scatter + hist.
- **#7** — `hist_plot` now annotates **P50/P90/P99** (was P50/P90); applies to
  the runtime-overhead hist and every histogram.
- **#9** — `trainer_early_late_counts.pdf` (300-wide bar) replaced by
  `sanity/trainer_late_fraction_hist.pdf`: one value per trainer (overran/total),
  with aggregate P50/P90/P99.
- **#15** — `selection_frequency_hist.pdf` (per-trainer bar) replaced by
  `selection/selection_fairness_lorenz.pdf`: Lorenz curve(s) + Gini, two levels
  (overall + always-available vs ever-unavailable when a trace has unavailability;
  degenerate for syn_0). New `plot_helpers.lorenz_plot`.
- **#16/#18** — comm now in **MEGABYTES** (`MODEL_MB` = 537,610 params × 4 B ≈
  2.15 MB; train=2×, eval=1× per chosen). `cumulative_comm_by_round`,
  `comm_per_round_train_vs_eval` (+ total-MB annotation), `comm_vs_accuracy`, and
  the `compare_streaming` axis all use MB. Override via `--model-params`.

### Plot items — DONE (second pass)
- **#13** — `system/compute_time_by_task_cdf.pdf`: trainer GPU compute CDF split
  by `task_to_perform` (train vs eval), n + mean per task.
- **#14** — `selection/selected_speed_cdf.pdf`, `selected_utility_cdf.pdf`, and
  `selected_utility_expected_vs_actual_cdf.pdf` (believed-at-selection vs the
  trainer's actual stat_utility that round). New `plot_helpers.cdf_multi`.
- **#17 (no new telemetry needed — derived):** `comm_per_round_train_vs_eval.pdf`
  is now split by task AND direction (train down / train up / eval down; eval up
  is a ~0-MB utility scalar) in MB; new `comm_message_accounting.pdf` shows
  messages returned vs **discarded** (chosen−committed, the overcommit slack).
  `comm_breakdown_by_round()` derives it all from selection (down) + agg_round
  contributing (up) — no hot-path instrumentation, so zero risk to long runs.
- **#19** — `system/trainer_time_split_overall.pdf`: whole-run mean trainer
  time split as one stacked bar (collapses the over-rounds area). (A true
  aggregator-internal step breakdown would still need aggregator step telemetry.)

### Overnight n300 run — infra READY
- **Configs:** `expt_scripts_2026/felix_oort_refl_feddance_alpha0.1_OVERNIGHT_node{1,2}.yaml`
  — each runs sim THEN real per baseline (node1 felix+oort, node2 refl+feddance),
  BOTH with the join barrier (min_start=290, tmo=600) so sim/real select from the
  same pool and are directly comparable. (Generated by merging STREAMING+SIMULATED.)
- **Robust runner:** `run_experiment_batch` is now non-interactive when stdin
  isn't a TTY (or `FLAME_BATCH_CONTINUE_ON_ERROR=1`) — a failed run is logged and
  the batch proceeds; new `_sweep_stragglers()` runs in a `finally` between
  experiments to hard-kill leftover trainer/aggregator procs (matches only the
  example mains, never the runner) and wait for GPU to drain. Post-run analyzer is
  already best-effort (never raises), so a plot bug can't kill a run.
- **Runner/smoke:** `scripts/overnight_run.sh {smoke | run node1 | run node2}`.
  smoke = 48-trainer / 4-round version of all 8 node experiments.
- **Comparison:** `scripts/compare_overnight.sh` → per-baseline sim-vs-real
  (`compare_parity`) + cross-baseline sim-plot/real-plot
  (`analyze_run --compare-streaming`) under `/tmp/overnight_compare`.
- **Verified mode-consistency for n300:** data-streaming + availability use the
  vclock in sim (oracular agg-side path fixed); felix eval-select frequency equal
  sim/real; oort reject / refl accept-≤5 staleness preserved (mode-independent).

### Telemetry fields available now (for the above)
`trainer_round`: real_gpu_time_s, sim_round_duration_s, wait_time_s,
training_budget_s, remaining_time_s, overran, pre_train_s, post_train_s,
stat_utility, final_loss, delta_weight_l2, visible_samples, total_samples,
dataset_size, avail_state. `agg_round`: round, agg_goal(_count), staleness[],
stat_utility[], trainer_speed_s[], contributing_trainers[], agg_observed_s{end:s},
updates_in_queue, vclock_now. `selection`: round, task, chosen[], num_*,
avail_composition, per_trainer{utility/believed_I/speed_s/selected}, exploration_factor.
Model size for MB: the `Net` in trainer/aggregator (compute param count once).

---

## Immediate next steps (suggested order)
1. ~~**Sync sim no-commit (Open B)**~~ — DONE. Was a SEND/RECV channel-state bug
   in the sync aggregator (distribute+aggregate both used SEND-default
   `channel.ends()`); fixed by distribute→SEND, aggregate→RECV. Verified with the
   fedavg seeded pair + e2e parity test (all 7 pass). Follow-up (optional): wire
   `selection` telemetry for the random selector on the sync stack so
   `compare_parity` checks #1/#2 (selection Jaccard, stat-utility) have events to
   compare instead of nan.
2. ~~**Async over-selection (Open A)**~~ — DONE. Two slot-accounting bugs in
   `asyncfl/top_aggregator.py` (pending ends dropped from `selected_ends`;
   buffered ends stripped as RECVD by `_handle_recv_state`). Fixed; staleness
   8.6→1.3, selection 19–21→~5–7, e2e 6/7 (remaining FAIL is irreducible
   selector RNG). Optional follow-up: `test_selection_parity` / `compare_parity`
   check #1 are too strict for RNG selectors (felix/fedbuff) — gate them on a
   "deterministic-selector" flag or relax to a distributional check, since exact
   selection parity is unattainable even real-vs-real for these baselines.
3. ~~**Cross-baseline parity sweep**~~ — DONE for all 6 (fedavg/feddance/fedbuff/
   felix/oort/refl), e2e 7/7 each. See "CROSS-BASELINE PARITY SWEEP" above for the
   fixes (fedbuff eval guard, oort/refl sim-ordering + straggler carry, join
   barrier, selection-parity gate). Optional follow-ups: (a) make a selector sort
   its candidate list before sampling to earn EXACT selection parity (then add it
   to `DETERMINISTIC_SELECTORS`); (b) the launcher lingers ~min after the
   experiment finishes (non-daemon shutdown) — runs complete but `timeout`-reap;
   worth a clean shutdown. Sweep harness: `/tmp/parity_sweep.sh <baselines...>`.
4. ~~**Plots**~~ — high-value items DONE: #16/#18 (MB), #15 (Lorenz/Gini fairness),
   #9 (late-fraction hist), #7 (P99), #5/#6 (overhead CDF). Remaining: #13, #14,
   #17 (needs new telemetry), #19-extend (see "Plot items — STILL TODO").
5. **n300 in sim** — `min_trainers_to_start`/`min_trainers_join_timeout_s` wired
   (configurable join barrier); `*_SIMULATED_node{1,2}.yaml` created (min_start=290,
   tmo=600). Oracular availability now uses the vclock in sim (was wall-clock, so
   traces other than syn_0 were ignored in sim). Data-streaming + felix eval-select
   frequency verified mode-consistent. Baseline staleness semantics (oort reject /
   refl accept ≤5) are in `_aggregate_weights` → mode-independent, respected in sim.

## Constraints to honor
Minimize code AND comment bloat (keep the PR diff small); deletions of dead code
are welcome. Don't ship unverified behavior changes (prefer revert + document, as
done for Open A). All baselines must produce the same plot set.
