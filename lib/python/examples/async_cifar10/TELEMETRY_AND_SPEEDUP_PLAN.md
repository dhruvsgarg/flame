# Plan: Trainer/Aggregator Telemetry + Virtual-Clock Speedup (async_cifar10)

## Context

Two related needs in the `async_cifar10` FL example. **Order of work: Task 1 (telemetry) first, then Task 2 (speedup) later.**

1. **Telemetry & visualization.** Today we only see loss/accuracy/round-time. We can't see *how the selector chooses* (utility vs. speed tradeoff, what it deems available/eligible, how it uses current resources) or *how the aggregator aggregates* (staleness, agg-goal progress, participation). The existing `scripts/plotters/` is a manual, multi-step, hardcoded-path pipeline tuned for an old "fwdllm" log schema; it scrapes regexes out of free-text logs and is not run automatically post-run. We want robust, structured telemetry that supports apples-to-apples comparison **across selector implementations** and auto-generated plots after a run.

2. **Speedup that actually speeds up.** `speedup_factor=2` did not yield ~2x wall-clock. Root cause: slow-trainer dynamics are simulated with a literal `time.sleep()` after the GPU finishes, and several fixed waits/polls ignore `speedup_factor` entirely. We want to **keep the dynamics** (slow trainers delay their updates; selection still trades off utility vs. speed; staleness still accrues) while **eliminating real wall-clock spent sleeping**. The chosen approach is a **virtual (simulated) clock**: trainers do real GPU compute but report a *simulated completion time* instead of sleeping; the aggregator orders/commits updates by simulated time.

**Scope / reuse principle:** implement for `async_cifar10` first, but place all shared logic in the `flame` library base classes (aggregator / selector / trainer mixins) so other examples inherit it without duplication. The example only wires config + example-specific hooks.

---

# Task 1 — Structured telemetry + plots (do first)

### Structured emission (library layer, generic)
- New **`flame/telemetry/` module**: a lightweight `TelemetryWriter` that appends **JSONL event records** to a per-run file (one schema, typed events). Inject via base aggregator/selector/trainer so every example emits the same schema → cross-selector comparison is free, no regex scraping.
- **Event types to emit** (most state already exists; we structure it, not recompute):
  - *Selector*: per-`select()` — candidates, eligible set, availability composition (counts of `AVL_TRAIN`/`AVL_EVAL`/`UN_AVL` from `PROP_AVL_STATE`), per-trainer utility/speed used, chosen set, explore-vs-exploit split. Hook the base `AbstractSelector` ([selector/__init__.py](../../flame/selector/__init__.py)) + each concrete selector's decision point.
  - *Aggregator*: per-round loss/accuracy/round-time (already logged), plus staleness distribution, agg-goal progress, per-trainer participation, in-flight count (async).
  - *Trainer*: per-round real GPU time vs. `sim_round_duration`, wait time, availability state, samples visible (streaming). This is also the data that proves the Task 2 speedup works.
  - *Streaming utility disparity (new)*: with data streaming enabled, emit at fine resolution the trainer's **statistical utility on the currently-unlocked prefix** (the real, time-T value) **alongside a counterfactual utility computed over the full dataset** as if it were unlocked from the start, plus their ratio and the visible-sample fraction. Reuse the existing utility computation (Oort `I_m` / `reset_stat_utility` / `fetch_statistical_utility`) and the streaming machinery (`_visible_sample_count`, `_rebuild_stream_loader`, [trainer/pytorch/main.py:428](trainer/pytorch/main.py#L428)) — build a full-pool loader for the counterfactual forward pass. **Configurable** (`util_counterfactual: {mode, every_n_rounds, sample_size}`), **default subsample + every-N-rounds**, gated off otherwise, so the extra forward pass is opt-in. This surfaces the hypothesized gap between real-world streamed utility and the full-dataset assumption.
- Keep `wandb` as-is for live dashboards; JSONL is the source of truth for offline/paper plots.

### Plotting (fresh module, reuse helpers)
- New **`scripts/analysis/analyze_run.py`** (or `flame/telemetry/plots.py`): a single `analyze_run.py <run_dir>` that reads the JSONL and emits a PNG bundle + a small summary. **Reuse the existing matplotlib helpers** rather than rewriting: CDF/percentile rendering from `scripts/plotters/cdf_plot.py` (`generateCDF`), multi-run band/interpolation logic from `scripts/plotters/comparative_plotter.py`, and stacked-bar (train vs. stall) from `scripts/plotters/stall_stacked_bar_plot.py`.
- **Plots:** loss/accuracy/round-time over time; selector availability composition over rounds (stacked); utility-vs-speed scatter of selected vs. eligible; selection-frequency/fairness per trainer; staleness CDF; agg-goal/in-flight timeline; trainer real-GPU vs. simulated-time stacked bar (the speedup evidence); **streamed-vs-full utility disparity over time** (per-trainer real vs. counterfactual utility + ratio). A `--compare run_dirs...` mode overlays selectors.
- **Auto post-run:** invoke `analyze_run.py` from the example's launch/teardown so plots land in `<run_dir>/plots/` automatically.

---

# Task 2 — Virtual-clock speedup (do later)

## Why virtual-clock is the right call (correctness of reordering)

Concern: the async aggregator pops updates FIFO **by physical arrival** (`channel.recv_fifo`, [channel.py:430](../../flame/channel.py#L430)) — if we stop sleeping, all trainers finish ~together and arrival order no longer reflects simulated speed, corrupting staleness/ordering. Verified against both loops:

- **Async** ([asyncfl/top_aggregator.py:176](../../flame/mode/horizontal/asyncfl/top_aggregator.py#L176)): pops **one** update per iteration via `recv_fifo(..., 1)`; staleness = `agg_round − trainer_version`, so commit order *does* matter.
- **Sync** ([syncfl/top_aggregator.py:231](../../flame/mode/horizontal/syncfl/top_aggregator.py#L231)): waits for `first_k` then aggregates via a weighted average (`optimizer.do`), which is **order-independent**; only *which* k complete and the round duration matter.

**Decoupling constraint (important):** the aggregator must **not** hold a global profile of client runtimes — that is unrealistic FL. Instead it acts only on information **trainers report about themselves**. A trainer's simulated round duration *is* a local quantity (its own `training_delay_s`, eval delay, streaming reveal — all known to the trainer at start). So:

- **Upfront ETA announce:** when a trainer is selected and begins (real) compute, it immediately sends a cheap control message announcing `sim_completion_ts = sim_send_ts + its own modeled duration`. This is local, observation-based reporting (like Oort/FedScale clients reporting expected speed), not aggregator coupling. Because the announcement is sent at *start* and is tiny, it arrives well before the heavy weight update — so by the time any update lands, the aggregator already knows every in-flight trainer's reported ETA.
- **Async commit rule:** the aggregator orders in-flight ends by reported `sim_completion_ts`, picks the **smallest**, and does a *targeted* blocking `channel.recv(end_id)` ([channel.py:384](../../flame/channel.py#L384)) on exactly that end. It commits it and advances the virtual clock `T_v` to that ETA. Ordering is by reported sim time, **not physical arrival**, so GPU-contention jitter (different trainers sharing a GPU finishing out of sim-order) cannot mis-order updates.
- **Bounded, correct waiting:** the only wait is physical — blocking until the earliest-ETA update actually arrives. If a simulated-faster trainer is physically slow due to contention, the aggregator waits real compute time for it (unavoidable for correctness; this is the "wait a bit" — bounded by one trainer's real compute, **never** by simulated delays). A real-time budget on the targeted recv maps onto the existing `SEND_TIMEOUT_WAIT_S` path so a trainer that announces then goes unavailable doesn't block forever (drop/stale → advance).
- **Sync:** the k committed updates = the k selected trainers with smallest reported `sim_completion_ts`; round duration = the k-th smallest. Aggregation result is unchanged (order-independent); we stop sleeping and set round wall-time from virtual time.

This is more efficient than scaling sleeps: wall-clock is bounded by *real GPU time only*, not by the modeled delays, so large `training_delay_s` values cost nothing — and it keeps the aggregator decoupled from any client profile.

### Shared library layer (generic, reused by all examples)
- **A new `flame/sim/virtual_clock.py` mixin** holding `T_v`, `advance(ts)`, and a per-in-flight-end map of **trainer-reported** `sim_completion_ts`. Aggregators compose this in; examples inherit. The aggregator stores only what trainers announce — no client profile.
- **Message contract** in [flame/common/constants.py](../../flame/common/constants.py): (a) a lightweight **ETA-announce control message** (`MessageType.SIM_COMPLETION_TS`) the trainer sends at start of compute; (b) the same field echoed on the final weight update for validation. Backward-compatible: absence ⇒ real-time behavior (non-sim examples untouched).
- **New selector property** `PROP_SIM_COMPLETION_TS` in [selector/properties.py](../../flame/selector/properties.py), populated from the trainer's announcement; aggregator records `sim_send_ts` at distribute-time (reuse existing `PROP_ROUND_START_TIME` / `_track_trainer_version_duration_s`).

### Trainer changes ([trainer/pytorch/main.py](trainer/pytorch/main.py))
- At start of a selected round, compute `sim_round_duration` locally (currently `training_delay_s` + eval delay, scaled by `speedup_factor`) and **announce** `sim_completion_ts` to the aggregator before/at the moment heavy compute begins. **Do not sleep** the delay. Remove/guard the post-GPU sleeps at [main.py:571](trainer/pytorch/main.py#L571) and [main.py:708](trainer/pytorch/main.py#L708); echo the same `sim_completion_ts` on the outgoing update for validation.
- Fix the fixed real-time waits that ignore `speedup_factor`: availability wait loops ([main.py:485](trainer/pytorch/main.py#L485), [main.py:668](trainer/pytorch/main.py#L668)), polling thread (line ~735), heartbeat thread, and the 20s channel-not-ready fallback ([main.py:295](trainer/pytorch/main.py#L295)). Drive these from the virtual clock (availability transitions already scale at [main.py:263](trainer/pytorch/main.py#L263); align the rest).
- A **`real` mode** flag preserves today's literal-sleep behavior for regression comparison.

### Aggregator changes
- **Async** ([asyncfl/top_aggregator.py](../../flame/mode/horizontal/asyncfl/top_aggregator.py)): consume ETA-announce messages into the in-flight ETA map. Replace the single `recv_fifo(...,1)` pop with "select in-flight end with min reported `sim_completion_ts` → targeted `channel.recv(end_id)` → commit → advance `T_v`." Targeted recv carries a real-time timeout that reuses the existing `SEND_TIMEOUT_WAIT_S` drop/stale path. Keep all existing staleness/agg-goal/participation logic; staleness now keys off `T_v` ordering instead of `datetime.now()` arrival.
- **Sync** ([syncfl/top_aggregator.py](../../flame/mode/horizontal/syncfl/top_aggregator.py)): pick the `first_k` smallest reported `sim_completion_ts` responders; set round duration from virtual time; no sleeping. Aggregation math unchanged (order-independent weighted average).
- Gate the whole thing behind a config switch (`time_mode: simulated|real`) so `real` mode is byte-for-byte the current path.

---

## Discovered during Task 1 testing (carry into Task 2)

Running the telemetry smoke (`felix_n10_alpha100_syn20_telemetry_smoke.yaml`, async_oort/fedbuff) surfaced several real bugs. Some were fixed inline to unblock telemetry validation; the deeper ones are flagged for Task 2.

**Already fixed (keep, but revisit under the virtual clock):**
- **`speedup_factor` never reached the trainer.** The launcher built the trainer command without `--speedup_factor`/`--battery_threshold`, so the trainer always ran at `1.0` regardless of YAML — this is almost certainly why the original "speedup_factor=2 gave no 2× speedup" observation happened. Fixed in `flame/launch/spawner.py` (+ `runner.py` passes `exp.trainer.speedup_factor`). **Task 2 must still confirm the *intended* speedup semantics** (GPU compute time does not scale; only `time.sleep` delays + availability-event timestamps do — the virtual clock is what actually decouples wall-clock from sim-time).
- **Aggregator hung forever on a quiet in-flight trainer.** `recv_fifo` had no timeout; when every in-flight end went silent (all unavailable, or a stale ghost) the async aggregator blocked indefinitely. Band-aided with `recv_fifo(timeout=...)` + `RECV_TIMEOUT_WAIT_S=30` in asyncfl `_aggregate_weights`, plus a ghost filter (`channel.has`) and `selected_ends` cleanup in `async_oort._cleanup_removed_ends`. **The virtual-clock redesign should replace this band-aid** with the targeted, ETA-ordered receive (which has principled per-end completion times and timeouts).
- UTF-8 stdio in the launcher (latin-1 locales crashed on status glyphs) and an async `channel.ends()==None` guard.

**Open issues for Task 2:**
- **In-flight accounting leaks (`freed=0`).** Telemetry showed a persistent `in_flight=1` ghost every round: an end gets selected, never returns an update, and is never freed from `selected_ends` (channel cleanup logged `in_flight_after=1, freed=0` each round). The proper fix is explicit in-flight↔availability reconciliation: on a client-notify `AVL_TRAIN→UN_AVL` transition (felix) drop the end from in-flight immediately; for non-notify baselines (fedbuff) a real timeout must drop it — and a returning trainer's late update should be explicitly accepted-or-discarded by version/staleness checks. This is core to the async correctness work.
- **Correlated availability is unrealistic.** All trainers share one `syn_20` "pattern" trace, so they go UN_AVL/AVL in lockstep (the whole system stalls together when the pattern is down). Decorrelate with per-trainer phase offsets (or per-trainer traces) so availability is independent — this also removes the global-stall failure mode at low speedup.
- **Model not learning in the smoke (accuracy flat at ~0.10 = random, loss pinned at ln 10).** Even after data fully unlocked, 100 async rounds produced no learning. Likely a hyperparameter/optimizer issue (client LR `0.001`, fedbuff server LR/`use_oort_lr`, delta-weight scaling) compounded by a very small per-trainer partition (~164 samples in this split). Needs a convergence sanity pass (validate felix actually learns on a known-good config) before drawing conclusions from streamed-vs-full utility plots. Telemetry is correct — it is what surfaced this.
- **`np.str_` end-ids leak into participation/selection dicts** (from `np.random.choice` in oort sampling). Harmless today (subclass of `str`) but a smell; normalize to `str` to avoid subtle set/dict-key surprises.

---

## Tests (enhance pytest; cover simulated + real modes)

Existing selector tests live in `lib/python/tests/selector/` with `lib/python/tests/conftest.py`. Add:
- **`tests/telemetry/test_event_schema.py`** (Task 1): schema/round-trip + that each selector emits the required event fields (parametrized over selectors for cross-comparability). Include a streaming-utility case: with a known prefix vs. full pool, assert the counterfactual utility ≥/≠ the streamed utility as expected and that disparity events are emitted at the configured cadence.
- **`tests/sim/test_virtual_clock.py`** (Task 2): virtual-clock advance/ordering unit tests; assert min-`sim_completion_ts` selection reproduces a hand-computed staleness sequence.
- **`tests/mode/test_async_aggregation_ordering.py`** (Task 2): feed a scripted set of in-flight ends with known sim durations + shuffled physical arrival; assert commit order and per-update staleness are **identical** to a reference computed by virtual time, and **independent of arrival order** (the jitter guarantee).
- **`tests/mode/test_sync_aggregation_equivalence.py`** (Task 2): assert sync aggregate output is identical in `simulated` vs. `real` mode (order-independence guard / regression).
- **Mode parity / regression** (Task 2): parametrize a tiny end-to-end run over `time_mode ∈ {simulated, real}`; assert correctness parity (final weights/accuracy within tolerance) and that `simulated` wall-clock < `real` (performance guard).

---

## Critical files

| Area | File |
|---|---|
| Async agg loop | `lib/python/flame/mode/horizontal/asyncfl/top_aggregator.py` |
| Sync agg loop | `lib/python/flame/mode/horizontal/syncfl/top_aggregator.py` |
| Channel recv | `lib/python/flame/channel.py` (`recv` L384, `recv_fifo` L430) |
| Selector base/props | `lib/python/flame/selector/__init__.py`, `lib/python/flame/selector/properties.py` |
| Trainer | `lib/python/examples/async_cifar10/trainer/pytorch/main.py` |
| Msg types | `lib/python/flame/common/constants.py` |
| New: telemetry (Task 1) | `lib/python/flame/telemetry/` |
| New: analysis (Task 1) | `scripts/analysis/analyze_run.py` (reuses `scripts/plotters/` helpers) |
| New: virtual clock (Task 2) | `lib/python/flame/sim/virtual_clock.py` |

---

## Verification

**Task 1**
1. Run produces `<run_dir>/events.jsonl` and `<run_dir>/plots/*.png` automatically; eyeball selector availability-composition and streamed-vs-full utility-disparity plots.
2. `pytest lib/python/tests/telemetry` green.
3. Cross-selector compare: run two selectors (e.g. async_oort vs. feddance), `analyze_run.py --compare` overlays utility/speed/staleness/accuracy.

**Task 2**
4. `pytest lib/python/tests/sim lib/python/tests/mode` green; ordering/parity tests pass.
5. End-to-end async_cifar10 at `speedup_factor=2`, `time_mode=simulated`: wall-clock materially lower than `real` mode for the same #rounds, while final accuracy curve matches `real` within tolerance.
6. Regression: `real` mode output unchanged vs. pre-change baseline (same seed) — final weights/accuracy parity.
