# High-Fidelity Simulator for FwdLLM — Real↔Sim Parity

**Active build (branch `dg/sim_parity_fwdllm`).** A simulated-clock runner for the `fwdllm` example
(FedFwd / forward-gradient FL) reaching **real↔sim parity** across the **fluxtune / fwdllm / fwdllm_plus**
baselines — at **100% availability (syn_0)** first (Phase 1), then **unavailability** (syn_20/50/mobiperf,
Phase 2), then **beyond syn_0** (Phase 3). fwdllm has no native sim clock; the build wires flame-core's virtual
clock + availability substrate into fwdllm's variance-gated gradient loop. It reuses async_cifar10's virtual
clock, sct reorder buffer, in-flight gate, availability substrate, and parity ladder where they transfer, and
deviates where the workload demands (fwdllm aggregates **gradients** not weights; **variance-gated dynamic-K**
commit cadence; **`data_id`** progress axis; one-message-per-call grad loop; rollback across agg-goal cycles).

> ## PREAMBLE — how to maintain this doc (READ BEFORE EDITING)
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
fwdllm inherits via its aggregator class chain. [fluxtune_contributions.md](fluxtune_contributions.md) §8 — the
LIVE ledger for fluxtune training-stability/convergence issues (oscillation, collapse, accuracy degradation);
check it BEFORE opening a new stability investigation here (§G, 07-14 session 7).

---

## §A  Current status — 2026-07-14, 7200s baseline-calibrated runs (all 3, one scoreboard)

**Basis:** 7200s real+sim pairs, per-baseline `training_delay_factor` from §O
(`--delays on --num-gpus 8`; fluxtune `--delay-divisor 0.48`, fwdllm/fwdllm_plus
`--delay-divisor 1.63`) — `run_20260714_002901` through `run_20260714_052349`.
Parity reports: `experiments/_parity_reports/parity_{fwdllm,fwdllm_plus,fluxtune}
_syn_0_20260714_LATEST.json`, generated with **two parity-CLI bugs fixed** this
pass (both landed, tested, see §G) — the scoreboard below is post-fix and
trustworthy:
1. `scripts.parity.cli --batch`'s glob matched `fwdllm_plus` for the `fwdllm` tag
   too (`fwdllm` is a strict prefix), silently double-reporting fwdllm_plus's
   numbers under the fwdllm label whenever both baselines' run dirs coexisted.
2. `_progress_axis()` (`scripts/parity/checks.py`) picked `round` vs `data_id`
   independently per side, and raw `cycle_data_id` collided across laps — see the
   §G entries for the full mechanism. This was corrupting `overhead_residual`/
   `per_round_advance`/`throughput`/`terminal_state`/`total_commits`/`convergence`
   for fwdllm and fwdllm_plus specifically (fluxtune was unaffected — it never
   completes a lap at this scale). **Re-run after the fix**: those rungs now FAIL
   with sane, comparable numbers instead of the earlier degenerate ones (e.g.
   fwdllm's `overhead_residual` went from a nonsensical `sim=572.8s/real=37.7s`
   to a genuine `sim=45.5s/real=37.7s`, 20.6% rel) — a real finding, not an
   artifact (below).

| baseline | `sim_rate` | enforced | verdict |
|---|---|---|---|
| **fluxtune** (async) | **1.82×** (was 0.97× pre-fix) | 52/64 pass · 12 fail · 3 warn · 17 skip | Bug-A gate fix (§G) confirmed real: sim_rate roughly doubled. Still trails fwdllm's (lower, recalibrated) rate. Root: `overhead_residual` 19.2%, `preferred_duration` 50.7pp, `step_timing_breakdown`, `g1_grad_norm`. |
| **fwdllm** (sync) | **2.46×** (was 5.71× at 1200s/factor 0.5) | 51/63 pass · 12 fail · 1 warn · 20 skip | sim_rate drop is EXPECTED (§O recalibration, not a regression — below). Zero `[TIMING_OVERRUN]` in 3350 real commits. Root: `overhead_residual` 20.6% (shared with the other two, see below), `phase_weights_to_ram` (ms-scale, low practical weight), `step_timing_breakdown`. |
| **fwdllm_plus** (sync) | **2.47×** | 52/63 pass · 11 fail · 1 warn · 20 skip | Same recalibration-driven sim_rate drop as fwdllm. Root: `overhead_residual` 21.5% (shared), `step_timing_breakdown`. |

### `overhead_residual` — RESOLVED (sessions 5-7, 07-14)

All three show ~17-21%, two unrelated mechanisms:
- **fluxtune** (19.2%, real>sim): the classic gap this rung was built for — real pays per-commit MQTT/dispatch
  overhead sim skips by design (§F #1). Just over tolerance; same class as felix/refl/oort's smaller gaps.
- **fwdllm/fwdllm_plus** (20-21%, sim>real): fully explained by `v1_iter_per_data_id` — real/sim are two
  independently-unseeded runs whose trajectories drift apart late in the run, so sim needs more variance-gate
  retries per `data_id`, and each retry costs a fixed ~11s barrier cycle. Confirmed: `max(trainer_speed)` itself
  is mode-invariant once iteration-count is held fixed (real 11.23s vs sim 11.04s, 1.6% apart, n=72 matched
  cycles) — the gap is purely the retry-COUNT variable, not a clock-formula defect. No code change; principle #1
  ("never put overhead on the vclock") stands.

`step_timing` breakdown (aggregator's own, now checked via `agg_step_timing_breakdown`, §G):

| function | real | sim |
|---|---|---|
| `sync_collect_and_accumulate_grads` | 16.9s/cycle | 3.5s/cycle |
| `_distribute_weights_sync` | 3.7s/cycle | 0.45s/cycle |
| `aggregate` (FedAvg merge) | 0.51s/cycle | 0.58s/cycle |

Real's ~10s/cycle excess is the real-only `num_min_req=1` clamp calling both functions once per LAP (10
laps/cycle) instead of once per cycle. **KEPT, not removed** (§G): the sync compose loop couples dispatch+collect
per lap, and bulk-collecting risks stranding messages (same hazard `drain_ready` dodges) plus breaks
`reselect_each_iteration`'s per-arrival re-selection — a real fix needs a compose-loop refactor + live
validation, not attempted blind. The log-volume half of the gap (not the lap coupling) was cheap to cut and is
fixed (§G).

### Next decisions before the next run

1. ~~fluxtune server-optimizer, code~~ — **LANDED (07-14), see §G.** `server_momentum` (heavy-ball momentum,
   default 0.0 = byte-identical) gated in the shared `FedSgdAggregator.aggregate()`. Code + tests landed, no
   run yet. Short (8min) with/without pairs exist for **all 3 baselines** now (`*_n10_smoke_short.yaml` vs
   `*_n10_smoke_short_momentum.yaml`, momentum=0.9), not just fluxtune — S1's scope is UNDECIDED
   (`fluxtune_contributions.md` §8.3 caveat: fwdllm's own telemetry shows the same early acc=0.25/mcc=0
   collapse). **Short runs can only test the early-cold-start severity** (`data_id` ~0-15, the only range
   an 8min budget reaches) — confirming fluxtune's specific **position-locked, recurs-every-epoch** pattern
   (F5) needs ≥2 full laps (`total_data_bins=150` each), which at the per-`data_id` iteration cost observed
   (~11-20s × several retries) realistically needs a 7200s-class run, not a short one.
2. Re-run the 7200s scoreboard with `agg_step_timing_breakdown` active + trimmed logging for a clean post-fix
   baseline before drawing conclusions from the momentum A/B.
3. Operator-run seeded real↔real pair (new `*_seeded.yaml`s, §G) to measure the GPU-nondeterminism floor.
4. Already open before this session, still open: `sim_sct_ordered_drain` A/B (below), felix (async_cifar10)
   46/46 re-confirmation (deferred repeatedly), C1/C2 convergence at matched `data_id` — then gate to Phase 2
   (unavailability).

### Short-run iteration loop (now) — short run → check logs/telemetry → fix → repeat

Do this before any 7200s commitment. **Real only** — the momentum question is "does it stabilize training",
not real<->sim parity, so sim adds nothing here (the `_sim_short[.yaml]`/`_sim_short_momentum.yaml` siblings
still exist for later parity work, just not part of this A/B). All 12 short/momentum yamls (real+sim, all 3
baselines, both legs) now carry `hyperparameters.seed: 1234` — same seed on both A/B legs, so selector/model-init
RNG is controlled and momentum is the only thing that can differ between them; no separate seeded run needed.
6 real configs total (3 baselines × without/with momentum=0.9), split across 2 nodes by treatment group:

```
bash lib/python/examples/fwdllm/expt_scripts/run_momentum_ab_without.sh   # node A: momentum=0, all 3 baselines
bash lib/python/examples/fwdllm/expt_scripts/run_momentum_ab_with.sh      # node B: momentum=0.9, all 3 baselines
```

Read after each pair: `agg_eval`'s `test-accuracy`/`test-loss` (any exact 0.25/mcc=0 collapse in the momentum
run vs without, at matched `data_id`s) and run `scripts/analysis/analyze_run.py` for the plots. Then, whenever
touching the parity ladder itself:

```
python -m pytest lib/python/tests/mode -k fwdllm -q
python -m pytest lib/python/examples/async_cifar10/scripts/parity/ -q
```

### Long-running overnight commands (later, once the short-run loop looks solid)

Not to be launched yet — full 7200s real+sim parity pairs, one baseline at a time, `--delays on` +
per-baseline `--delay-divisor` from §O:

```
lib/python/examples/fwdllm/expt_scripts/run_sequential.sh --only fluxtune \
    --delays on --delay-divisor 0.48 --max-runtime-s 7200

lib/python/examples/fwdllm/expt_scripts/run_sequential.sh --only fwdllm,fwdllm_plus \
    --delays on --delay-divisor 1.63 --max-runtime-s 7200
```

Plus, once the seeded-yaml real↔real floor and the momentum decision are both ready to check at full scale:

```
python -m flame.launch.run_experiment lib/python/examples/fwdllm/expt_scripts/fwdllm_n10_smoke_seeded.yaml
python -m flame.launch.run_experiment lib/python/examples/fwdllm/expt_scripts/fwdllm_n10_smoke_seeded.yaml  # run twice, diff the two real run dirs
```

### Per-baseline items NOT already covered above

`overhead_residual`/`v1_iter_per_data_id`/`step_timing_breakdown`'s shared `_fetch_weights`/`recv_wrapper`
pattern is folded into the resolution above — don't re-litigate per baseline. What's left, baseline-specific:
- **fluxtune**: `preferred_duration` (Sd, oort-only) — real's utility term is the *binding* selection constraint
  80.1% of rounds vs sim's 29.3%, a 50.7pp gap isolated to the oort selector.
- **fwdllm**: `phase_weights_to_ram` (ms-scale: real 6ms vs sim 2ms) — low practical weight, sanity-check only.

### GPU-overrun check: clean across all 3

Zero `[TIMING_OVERRUN]` in any of the three baselines' real aggregator+trainer
logs (6,000+ combined real commits) — i.e. no instance of a trainer's actual GPU
compute time exceeding its modeled per-trainer delay budget `D` (§O). This
directly validates §O's fast-class margin (+0.13s, flagged as "thin, watch
first") at 5× the sample count it was computed from — see §O.

### Accuracy reached at matched duration (7200s), across baselines and real/sim

Final accuracy = mean of the last 5 evals (single-point accuracy is noisy —
fluxtune's sim leg swings 0.44–0.66 in its last 5 evals alone; the windowed mean
is the more honest number). `C1` below is the checker's matched-checkpoint
average |diff|, now computed correctly post-fix (composite `(round, data_id)` key
excludes sim's lap-2 evals from being compared against real's lap-1 ones).

| baseline | real final acc | sim final acc | real − sim | C1 `avg_accuracy_diff` |
|---|---|---|---|---|
| fluxtune | 0.72 | 0.59 | +0.14 | 0.073 (FAIL, tol 0.05) |
| fwdllm | 0.66 | 0.51 | +0.15 | 0.097 (FAIL) |
| fwdllm_plus | 0.62 | 0.51 | +0.11 | 0.085 (FAIL) |

Two consistent patterns: (1) **real always ends higher than sim** across all
three baselines, by a similar ~0.11-0.15 absolute margin — not yet root-caused,
plausibly downstream of the same `overhead_residual` gap (sim's per-commit
cadence models more/less elapsed vclock-time than real actually took, so at a
matched *vclock* budget sim has done a different amount of *actual* gradient
work than real did in the same *wall* budget); needs checking against gradient
step counts, not assumed. (2) **fluxtune (async) reaches the highest real
accuracy** (0.72 vs fwdllm's 0.66 and fwdllm_plus's 0.62) despite far fewer total
commits (66 vs 100-101) at the same 7200s wall budget — async's overlap lets more
wall-clock-parallel gradient work happen per commit; not an apples-to-apples
comparison (different `agg_goal`/optimizer config per §C) but a real, notable
outcome-level gap worth keeping in view. Aside: fwdllm's and fwdllm_plus's SIM
legs track near-bit-identically for their first ~54/146 evals (confirmed exact
floating-point match on several early points) — expected, not a bug: under syn_0
(100% availability) there's no forced-reselection trigger, so fwdllm_plus's only
distinguishing knob (`reselect_each_iteration`) is close to a no-op early on: both
legs pick the same cohort/order until small GPU nondeterminism accumulates.

### vclock vs. wall-clock gap, across baselines

Two distinct numbers answer this, depending on which gap is meant:

**Within-sim (`sim_rate` = vclock ÷ sim's own physical wall-clock)** — "how much
faster than 1:1 does the sim compute": fluxtune 1.82×, fwdllm 2.46×, fwdllm_plus
2.47× (table above; `sim_speedup` DIAG rung, all three `is_speedup: True`, no
slowdowns).

**Cross-mode (`wall_disparity`, DIAG) — cumulative `|real-genuine-elapsed −
sim-vclock|` at matched progress units**, from the first matched unit to the
last:

| baseline | mean cumulative gap | max cumulative gap | matched units | as % of budget |
|---|---|---|---|---|
| fluxtune | 614s | 1305s | 66 | ~8.6% of 7148s |
| fwdllm | 140s | 394s | 100 | ~3.7% of 3756s |
| fwdllm_plus | 143s | 385s | 101 | ~3.8% of 3764s |

fluxtune's cross-mode gap is proportionally over 2× fwdllm/fwdllm_plus's — consistent
with its larger `overhead_residual` rel_diff (19.2% vs ~21%, similar) compounding
over fewer, larger commit cycles (`agg_goal=3` vs `10`) so each individual
mismatch is a bigger absolute vclock-second.

**Old fluxtune-specific detail below (Bug A / reactive gate) is CLOSED — see §G.
The `sim_sct_ordered_drain` A/B question is still open, not exercised by this run
(both legs used the config default `true`).**

### fluxtune: still-open `sim_sct_ordered_drain` A/B (unexercised by this run)

Both fluxtune legs used the config default `sim_sct_ordered_drain: true`. Whether the flag is load-bearing
(felix never sets it, uses `recv_fifo` unconditionally — `grep -rl sim_sct_ordered_drain
lib/python/examples/*/expt_scripts/*.yaml` matches only fluxtune's yaml) is still unsettled — no run has been
made with it `false` since the Bug-A gate fix landed. `fluxtune_n10_smoke_sim_no_sct_drain.yaml` exists for this
(`sim_sct_ordered_drain: false`, otherwise identical to `fluxtune_n10_smoke_sim.yaml`); re-run at the current
7200s/factor-0.48 basis, not the stale 1800s/factor-0.5 one, for a matched-vintage pair:

```
python -m flame.launch.run_experiment \
    lib/python/examples/fwdllm/expt_scripts/fluxtune_n10_smoke_sim_no_sct_drain.yaml
```

Read after: `sim_rate` (does it move off 1.82×?); `pastdated_n`/`pastdated_gap_max` in `[SIM_GRAD_RECV]`
(materially higher with the flag OFF means `recv_fifo`'s background streamer really does strand/lap messages
under fluxtune's probe frequency — the flag's actual justification).

(Roadmap after this closes: see "Next decisions before the next run" above.)

### Open follow-up: real-mode sync visibility-lag anchor (not yet decided)

`update_visibility_lag_s` is now populated for fwdllm/fwdllm_plus's sync path in SIM mode (surfaces the
pre-existing `_barrier_anchored_lags` computation in `sync_collect_and_accumulate_grads`, previously computed
but never reaching structured telemetry — same gap `commit_gap_s` had for fluxtune, §G). **REAL mode has no
equivalent wiring at all** — `sync_collect_and_accumulate_grads`'s real branch never computed a `_barrier_durs`
list the way its sim branch (or the base class's own `_aggregate_weights`) does. Undecided: whether streaming
per-message `_update_visibility_lag` or barrier-anchored `_barrier_anchored_lags` (adapted to fwdllm's
variance-gated dynamic-K cadence, not a fixed round) is the right anchor — needs deciding by reading how
`sync_collect_and_accumulate_grads`'s collection loop actually shapes arrival vs. commit for fwdllm's dynamic-K.
Left `None` deliberately rather than guessed at.

---

## §B  How fwdllm differs structurally

fwdllm aggregates **gradients** (JVPs) not weights; commit cadence is **endogenous** (variance-gated dynamic-K);
progress axis is committed **`data_id`** (variance passes), not update count. Gradient values are mode-invariant
given identical input+perturbation seed, so parity reduces to **clock + selection + ordering parity plus a
variance-cadence layer**. Anchors (`flame/mode/horizontal/syncfl/fwdllm_aggregator.py`): `aggregate()` var gate +
rollback/`cached_v` at each `_agg_goal` boundary; `total_data_bins=150`; force-commit cap
`max_iterations_per_data_id`; `reselect_each_iteration` (fwdllm_plus per-iteration reselection); sync path
`_aggregate_grads_sync`. Full detail: PARITY.md §F.1.

### §B.1  Real↔sim design deltas vs async_cifar10

| # | Axis | async_cifar10 | fwdllm | Why |
|---|---|---|---|---|
| 1 | Aggregated object | model **weights** | **gradients** (JVPs) | grad values mode-invariant → parity = clock+order+selection + variance-cadence |
| 2 | Progress axis | update/round count | committed **`data_id`** | cadence (updates-per-data_id) is an **output to match**, not an input |
| 3 | Commit cadence | fixed `agg_goal` | endogenous **variance-gated dynamic-K** | the emergent layer cifar doesn't model |
| 4 | sct delay model | `send + max(gpu, D)` | `send + max(gpu, D)` (remainder-wait) | real sleeps `max(0,D−gpu)` (device wall = D) so update order = per-trainer D order = deterministic, real↔sim identical |
| 5 | Per-eval sct | distinct, ~20× faster | collapses to train sct | eval lives on the aggregator; forward-grad "train" IS a forward pass |
| 6 | Slot residence | per-commit release | **hold slot to COMMIT** (felix port) — freed only when its update commits | correctness check, not a lever, for BOTH sync & async |
| 7 | Surplus grad on rollback | carried | **carried** for async (c≫agg_goal); **drop** stays correct for sync (c≈agg_goal) | drop was benign only for sync |
| 8 | Async drain primitive | `_sim_recv_min` (+`recv_fifo` default) | `_sim_recv_min_grad` (+opt-in `drain_ready` via `sim_sct_ordered_drain`) | fluxtune's higher per-grad probe frequency vs felix's per-round — **necessity of the opt-in is the open §A question** |
| 9 | `time_mode` default | `"simulated"` | `"real"` (getattr fallback) | fwdllm's whole config corpus is `real`; a `simulated` default risks half-activating an unbuilt path |
| 10 | Availability tracking (v1) | all `trace_read` | mixed: fwdllm unaware, fwdllm_plus `oracular`, fluxtune `client_notify` | baselines carry different models; first-class `client_notify` deferred to Phase 2 |
| 11 | Aggregator-side eval | backgrounded (daemon thread, off critical path, `eval_every_n_rounds`) since inception | now ALSO backgrounded (was synchronous, needed `sim_model_eval_time`'s vclock fold; §G Part 6) | the axis that matters is synchronous/blocking vs. backgrounded, not centralized vs. decentralized — cifar was only ever exempt from a fold by implementation choice, not a structural guarantee (§F #1/#10) |

---

## §C  Baseline matrix

| baseline | sync/async | selector | agg | tracking / avail | reselection | K (agg_goal) | dynamic_kc |
|---|---|---|---|---|---|---|---|
| **fluxtune** | async | `async_oort` | fedbuff (+server LR, JVP) | `client_notify`→`trace_read` v1 | — | 3 | disabled |
| **fwdllm** | sync | `random` | fedavg | `default` unaware | per-round | 10 (=c) | — |
| **fwdllm_plus** | sync | `random` | fedavg | `oracular` | per-iteration | 10 | — |

Phase order (locked): Phase-1 syn_0 → Phase-2 unavailability → Phase-3 beyond syn_0. One baseline at a time.

---

## §D  Parity ladder for fwdllm (rung defs live in PARITY.md §F)
- **Reused verbatim:** Stage 0 TC1/K10; Stage 1 P3/K1/K7; Stage 2 A1–A3; Stage 3 S3/4, A2c + oort stack
  (fluxtune only); Stage 4 T2/K6/T_mqtt; Stage 8 C1; Stage 9 budget/stop.
- **Modified** for variance-gated dynamic-K: K3a/K3b/K2/U3/K8/U2 → re-keyed to the variance-pass boundary /
  committed `data_id` (PARITY.md §F.3).
- **New** variance-cadence layer: V1–V5, DK1–DK3, G1–G2 (PARITY.md §F.4). Localize down; never fix an EMERGENT
  rung directly. `var_threshold` / `max_iterations_per_data_id` are baseline knobs, not parity levers.
- **Availability** rungs (A1–A5, A6/A7/A8/K11) inherited; apply once Phase-2 wires the effect path + telemetry.
- **Per-stage wall-budget instrumentation:** `drain_wall_budget`, `trainer_phase_wall_budget`,
  `step_timing_breakdown`, `aggregation_compute_wall` — ONE-SIDED (`sim<=real`) where sim should collapse a
  real-transport phase to ~0, DISTRIBUTIONAL where it's genuine shared compute.

---

## §E  Roadmap

**Phase 1 (syn_0) — in close-out.** The parity-CLI checker bugs are fixed (§A/§G) and the 7200s scoreboard is
trustworthy. `overhead_residual`'s ~17-21% is root-caused to TWO DIFFERENT mechanisms, not one shared root
(fwdllm/fwdllm_plus: `v1_iter_per_data_id` retry-count divergence; fluxtune: classic unmodeled-overhead
direction, just over tolerance) — see §A for the full breakdown and the session-6 next-steps checklist (§A,
"Next-session plan"), which is the authoritative punch list for closing Phase 1: aggregator-side overhead
instrumentation, the `num_min_req=1` real-only clamp, `overhead_residual`'s construction, seeding, and
fluxtune's post-boundary accuracy drop. Also open: `preferred_duration` (fluxtune, oort-specific) and
`sim_sct_ordered_drain` A/B (unexercised since the Bug-A fix landed).

Exit: all 3 baselines' `sim_rate`/throughput/terminal_state pass, then C1/C2 convergence at matched `data_id`.

**Phase 2 — unavailability (syn_20/50/mobiperf).** Wire the ClientAvailability effect path into the grad loop:
send-time gate (real) / `delivery_ts = max(sct, next_avail)` buffering (sim); two ledgers; starvation
vclock-advance; per-baseline `avail_select_filter`. Exit: A1–A5 + A6/A7/A8 PASS; self-stops; withheld grads
delivered not dropped.

**Phase 3 — beyond syn_0.** Full ladder under scarcity. Exit: curves within tolerance at matched `data_id`;
K8/U2 within bar; V1/V2 binned residual flat.

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

---

## §L  Forward-grad JVP compute profile & retained fluxtune optimizations
*(tool: `scripts/profile_jvp_opt.py` — reuses real `create_model` + `calculate_jvp`; distilbert-base
+ AdapterHub adapters, batch 8, seq 192, A40, fp16. Absolute ms are a CLEAN single-trainer profile; the real run
is ~10× from GPU contention across the 10 concurrent trainers, but pass-counts/ratios/memory transfer.)*

**Mechanism.** Forward-grad trains via a **central finite-difference JVP** (`fwdgrad_utils.calculate_jvp`): each
perturbation = **2 forward passes** `f(θ±hv)`, h=0.01, autocast+no_grad → `jvp=(f(θ+hv)−f(θ−hv))/2h`. **fluxtune**
selects the best of `perturbation_count`(=10) perturbations by |jvp| (2P=**20 passes**); **fwdllm/sync** selects
by cos-sim (**0 forward passes**) + 1 final JVP. Only **~1.5% of params trainable** (bottleneck adapters in all 6
layers + head, 1.04M/67.4M); backbone frozen.

| path | fwd passes | ms/batch (clean) |
|---|---|---|
| sync fwdllm (current) | 5 | 50 |
| sync fwdllm (opt) | 2 | 16 (−68%) |
| fluxtune P=10 (current) | 25 | 251 |
| fluxtune P=10 (opt) | 20 | 159 (−37%) |
| backprop ref (1 fwd+1 bwd) | — | 17 |

- **Compute vs sync:** fluxtune = `2P × per-pass` → 10× sync at P=10, linear in P, equals sync at P=1.
  JVP-selection is the entire fluxtune surcharge; sync's cos-sim selection is free.
- **Memory:** forward-grad peak is FLAT in P (~3.2–3.4 GB = model + one held forward; no autograd graph).
  fluxtune's extra JVP inferences cost TIME, not memory.

**LANDED, fluxtune-only & config-gated** (`jvp_perf_opt`, default false = byte-identical; true in both fluxtune
yamls; sync untouched): trainable-only finite-difference (skip the 98.5% frozen params inside `calculate_jvp`)
+ drop 3 diagnostic-only forward passes + reuse the winner's cached JVP. Bit-identical, real↔sim parity
untouched.

**NOT retained (changes fidelity, excluded per the fidelity bar):** vmap-batching (2.0× win, but ~5% different
in fp16/fp32 from catastrophic-cancellation reduction-order sensitivity); forward-mode AD (slower, different
math); `perturbation_count`↓ (changes the baseline algorithm).

---

## §M  Sim receive/barrier redesign — event-driven, zero-hardcoded-wait

**Status: code landed 2026-07-12 (all 9 subtasks), 723 tests green.** Live validation ran 07-13 (§A) —
partial: fixed fwdllm_plus's livelock (confirmed), fwdllm stayed healthy, but fluxtune's `sim_rate<1` persists
under a **different, more precise root** than what motivated this redesign (§A: one-grad-per-tick throughput
backlog, not a hardcoded-wait/grace-ceiling problem — the shared `_sim_known_delay_s` cache this redesign built
is not obviously the bottleneck, see §A's A/B test).

**Design (still current):** one canonical delay-report field `MessageType.MODELED_DELAY_S`; one shared
per-trainer delay cache in `syncfl.TopAggregator` (`_sim_known_delay_s` / `_note_sim_known_delay` /
`_sim_recv_timeout_s`) replacing three previously-divergent per-subclass EMA/budget-fallback copies (syncfl,
asyncfl, fwdllm_aggregator). No hardcoded seed, no cross-trainer fallback — an unseen trainer gets no bound
(the barrier blocks genuinely via `recv_fifo(timeout=None)`, confirmed non-CPU-polling); a known trainer gets
an exact deterministic wait bound. `drain_ready(timeout=None)` returns immediately-empty (can't block like
`recv_fifo` can) — this is why fluxtune's opt-in `sim_sct_ordered_drain` path needs a poll-tick fallback that
felix's default `recv_fifo` path doesn't (§A/§B.1#8).

**2026-07-13 addition:** `_sim_recv_min_grad` now tracks past-dated-commit telemetry
(`_sim_pastdated_commits`/`_sim_pastdated_gap_max`/`_sim_pastdated_by_source`, ported from felix's
`_sim_recv_min`/`_sim_pop_committable` path which fluxtune's grad loop never went through) — folded into the
`[SIM_GRAD_RECV]` log line. Purpose: make the pending `sim_sct_ordered_drain` A/B (§A) legible on the
correctness dimension, not just `sim_rate`.

---

## §O  NPU-calibrated `training_delay_factor` per baseline

**Problem.** `lib/python/examples/_metadata/trainer_registry.yaml`'s `training_delay_s` (4–19s, Papaya/FedBuff
mobile-CNN traces) is one shared constant scaled by one shared `training_delay_factor` (0.5, all three
baselines) — a CNN training-round budget, not calibrated to fwdllm's actual forward-grad JVP cost, and (§L)
fluxtune and fwdllm/fwdllm_plus don't cost the same: fluxtune's `perturbation_count`=10 selection is 20
fwd-pass-units/data-bin vs fwdllm/fwdllm_plus's 5 (1 JVP + 3 diagnostic passes) — **~4×, not the ~10× the raw
`perturbation_count` alone would suggest**. One shared divisor can't be right for both.

**Ground data.**
- Real per-sample forward-grad JVP cost for distilbert, measured on the FwdLLM paper's reference NPU device
  (`third_party/ae/fig15/b&c-energy&network.ipynb`, `train_time_dict_dict["distilbert"]["ours"] = 0.3085584`
  s/sample) — matches our config exactly (`use_adapter: false`, `fl_algorithm: FedFwd`,
  `configs/aggregator_base.json:38-40`).
- Per-baseline fwd-pass-unit counts: §L's clean single-trainer profile (`scripts/profile_jvp_opt.py`, A40) —
  fwdllm/fwdllm_plus 5 units, fluxtune(opt) 20 units — cross-validated against the banked 07-12/07-13 real runs'
  `forward_passes_iter`/`perturbations_iter` telemetry (`FedSgdTrainer.py:740-745`): both agree exactly
  (fwdllm/plus 5/1 constant, fluxtune 20/10 constant across all iterations in both runs). A static trace of the
  `select_perturbation_using_jvp=False` code path suggested 1 JVP for all three baselines — **this is wrong,
  discard it; the telemetry+profile agreement is ground truth.**
- 100-trainer registry stats (`trainer_id` 1–100, the pool `client_idx_modulo` draws from): mean=12.51s,
  median=11.0s, stdev=8.49s, range=[2,47]s. By `speed_class`: fast (n=16) mean 3.00s [2,4]; medium (n=22) mean
  6.32s [5,8]; slow (n=19) mean 10.53s [9,12]; very_slow (n=43) mean 20.09s [13,47].

**Reference-device target cost per data bin** (`= fwd_pass_units × per-sample-JVP-time × batch_size / 2`,
batch_size=8; `/2` because 1 JVP = 2 fwd-pass-units by `fwdgrad_utils`' own counting convention):
```
1 fwd-pass-unit (NPU) = 0.3085584 × 8 / 2 = 1.2342 s
fwdllm / fwdllm_plus:  5 units × 1.2342  = 6.171 s / data bin
fluxtune:              20 units × 1.2342 = 24.685 s / data bin
```

**`training_delay_factor` (÷ on `training_delay_s`, `FedSgdTrainer.py:546`, config-only, no code change).**
Anchor: `divisor = registry_mean / target_cost`, uniform across all 100 trainers (preserves the Papaya/FedBuff
relative fast:medium:slow:very_slow spread; only re-anchors the absolute magnitude). A flat **+1.5s buffer**
(midpoint of the 1–2s asked for) is added to each baseline's target cost before deriving the divisor, so the
gap between modeled delay and real GPU compute doesn't run to zero:

```
fwdllm / fwdllm_plus:  target 6.171+1.5=7.671s → divisor = 12.51/7.671 ≈ 1.63
fluxtune:               target 24.685+1.5=26.185s → divisor = 12.51/26.185 ≈ 0.48
```

| | old (shared) | new fwdllm/fwdllm_plus | new fluxtune |
|---|---|---|---|
| `training_delay_factor` | 0.5 | **1.63** | **0.48** |
| registry-mean delay | 25.02s | 7.67s | 26.06s |
| fast-class delay | 6.00s | 1.84s | 6.25s |

**Fast-class headroom (the binding constraint — smallest budget, so checked explicitly, not just the mean).**
Real observed GPU compute (07-12/13 banked runs, this dev GPU, not the NPU): fwdllm/plus mean 1.215s max
1.712s; fluxtune mean 3.630s max 5.618s.
```
fwdllm/plus fast-class: 3.00/1.63 = 1.840s vs observed max 1.712s → margin +0.13s (THIN — watch first)
fluxtune fast-class:    3.00/0.48 = 6.250s vs observed max 5.618s → margin +0.63s (comfortable)
```
fwdllm/fwdllm_plus's fast class is the one to watch for `[TIMING_OVERRUN]` (`FedSgdTrainer.py:549-556`) —
re-tighten (raise the divisor slightly) or accept per that warning's own guidance if it fires.

**Caveats (unchanged from the derivation discussion):** the NPU number is a single benchmark point from one
unnamed device, not a distribution; per-sample × batch_size is an upper-bound linear approximation (NPU
batching may parallelize part of this in reality); this recalibrates delay *magnitude* only — it does not
give LLM-specific heterogeneity *shape* (no data exists on whether cheap phones degrade disproportionately more
on transformer ops than CNN ops).

**Action — update configs to use these, not the old shared divisor.** `run_sequential.sh`'s `--delay-divisor`
is a single value per invocation (§ "Usage"), so the three baselines now need **separate invocations**, not one
shared `--delay-divisor 0.5 --delays on` run across all of them:
```
run_sequential.sh --only fluxtune               --delays on --delay-divisor 0.48
run_sequential.sh --only fwdllm,fwdllm_plus      --delays on --delay-divisor 1.63
```
Any future parity/smoke run that passes `--delay-divisor` must use the baseline-appropriate value above, not
the old 0.5 default. **Validated 07-14** by the 7200s real runs (`run_20260714_002901`/`_003007`/`_032159`,
6,000+ real trainer-commits combined): zero `[TIMING_OVERRUN]` across all three baselines' aggregator+trainer
logs, including fwdllm/fwdllm_plus's thin (+0.13s) fast-class margin — holds even at 5× the sample count of the
07-12/13 runs the margin was computed from.
