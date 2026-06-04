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
- `random` selector context: its SEND/RECV channel **state** is the *buffered
  concurrency pattern* (FwdLLM/async): SEND=pick new trainers to send to,
  RECV=return in-flight set to receive from. Stateless sync FL never sets it →
  `KeyError 'state'`. Fixed by defaulting state to SEND when absent; also removed
  a fully-duplicated dead `select()` method. fedavg now runs on the sync stack
  (validated: no KeyError, 40 real rounds).

### OPEN ISSUE A — async (felix) sim over-selection [characterized, NOT fixed]
Seeded e2e on felix: invariants hold, but per-round selection ~5 real vs **19–21
sim**, staleness mean **1.16 real vs ~8.6 sim** (KS ~0.77). Root cause: under
no-sleep, every selected trainer returns immediately and `_sim_recv_min` buffers
*all* returned recv-state ends while committing only `agg_goal`/round, so the
in-flight buffer (and `[SIM_PENDING] blocked` count) grows toward N across rounds
(observed climbing 7→43), inflating staleness. A targeted slot-accounting fix
(keep pending in `selected_ends` so `extra=c-len(selected_ends)` doesn't
over-refill) did NOT resolve it — the growth is the buffer-fill draining slower
than it fills, not just slot accounting. **Reverted, not shipped.** `first_divergence`
on the felix runs localizes the first control divergence to **commit #1**
(selection-level RNG desync from differing candidate sets), before staleness
blows up. NEXT: bound the sim in-flight/buffer to `c` (don't buffer beyond the
concurrency the real path would have in flight), then re-verify with the seeded
e2e test + `first_divergence`.

### OPEN ISSUE B — sync (fedavg) sim aggregation does not commit [new, NOT fixed]
After the None-guards (`8ac0d535`), the sync sim run no longer crashes and
distribute works (8+ sends), but `_sync_sim_recv_first_k` produced **0
`[SYNC_SIM_RECV]` commits** in ~5 min and no rounds completed. Likely the
fill-loop isn't receiving the selected ends' updates against the real channel
(fake-channel test passes, live differs — same class of gap as the crash). NEXT:
reproduce, add a debug log of `len(buf)` / `pending` per pass in
`_sync_sim_recv_first_k`, and check whether `channel.recv_fifo(pending,
first_k=len(pending), timeout=...)` is delivering against the real channel state
(the real channel needs ends in RECV state; the random/sync distribute may leave
them in a different state than the helper assumes).

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

### TODO plot items (with intended approach)
- **#5/#6 (deeper):** per-step overhead distributions at trainer AND aggregator;
  agg-observed vs actual trainer compute+response (a `runtime_agg_vs_trainer`
  scatter already exists — make it an explicit overhead CDF + aggregate stats).
- **#7** runtime-overhead histogram → add aggregate P50/P90/P99 annotation; make
  readable.
- **#9** per-trainer early/late counts (giant bar) → replace with a histogram of
  per-trainer late-fraction + aggregate stats.
- **#13** participation: add per-trainer AND aggregate compute-time stats split
  by eval/train.
- **#14** speed_utility: split into separate CDFs of utility and of speed; and
  split expected (selection-criteria value) vs actual trainer stat_utility at
  that time.
- **#15** selection-frequency histogram (unreadable) → drop the aggregator;
  Lorenz curve / Gini for fairness, at two levels (overall + by availability),
  related to each trainer's data-split size.
- **#16** comm `comm_per_round_train_vs_eval`: use **megabytes** (model param
  count × 4 bytes × #ends; train=down+up=2×, eval=down=1×) instead of
  "model-equivalents"; add total annotation + aggregate stats.
- **#17 (needs NEW telemetry):** track MB moved + message counts agg→trainer and
  trainer→agg; sanity counts of weights sent, responses received, discarded,
  unused. Add to `agg_round`/a new event. The selector mostly doesn't know these.
- **#18** comm_vs_accuracy: replace "model-equivalents" x-axis with MB (same
  conversion as #16).
- **#19 (extend):** also an aggregator-side step time breakdown + an overall
  (whole-run) stacked bar, not only the over-rounds area.

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
1. **Sync sim no-commit (Open B)** — add per-pass debug to `_sync_sim_recv_first_k`,
   reproduce on `fedavg_n48_parity_seeded_sim.yaml`, fix recv against the real
   channel, then run the fedavg seeded pair + e2e parity test. (Sync is the
   simpler/structurally-bounded case; fixing it first informs the async one.)
2. **Async over-selection (Open A)** — bound sim in-flight/buffer to `c`; verify
   with seeded e2e + `first_divergence`.
3. **Cross-baseline parity sweep** — seeded real/sim pairs for oort/refl/feddance
   (configs needed; oort/refl need oracular traces, feddance needs use_oort_loss_fn).
4. **Plots** — work the TODO list above; #16/#18 (MB) and #15 (fairness) and #9
   (aggregate) are high-value/low-risk; #17 needs new telemetry.

## Constraints to honor
Minimize code AND comment bloat (keep the PR diff small); deletions of dead code
are welcome. Don't ship unverified behavior changes (prefer revert + document, as
done for Open A). All baselines must produce the same plot set.
