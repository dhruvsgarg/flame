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
fwdllm inherits via its aggregator class chain.

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

### `overhead_residual` root-caused (07-14 session 5): TWO DIFFERENT mechanisms, same-size symptom

All three baselines show a ~17-21% `overhead_residual`, but investigating the actual
per-round telemetry (per-trainer `agg_observed_s`/`contributor_intervals`, binned by
quartile of the run) shows **two structurally different mechanisms that happen to
produce similar-magnitude numbers** — not one shared root. Correcting the earlier
"shared root" framing from this morning's pass.

**fluxtune (19.2%, real > sim — the DESIGNED direction) is the classic mechanism
this rung was built to catch.** Real pays per-commit MQTT/dispatch/aggregator
overhead that sim (by design, §F #1) never charges to the vclock — `implied_per_commit_overhead_s=6.93s`
lines up with real's own `recv_wrapper`/`_fetch_weights` phase costs
(`step_timing_breakdown`, §A above). The magnitude just exceeds the 10% tolerance;
same class of gap async_cifar10's felix/refl/oort carry (below), just bigger here.

**fwdllm/fwdllm_plus (20-21%, sim > real — the OPPOSITE, unexpected direction) is a
different mechanism, specific to fwdllm's variance-gated dynamic-K, with no
counterpart in any fixed-`agg_goal` baseline.** Method (per-stage delta binning, as
suggested): pulled every `agg_round` event's `iteration_per_data_id` and `grad_norm`
for fwdllm, binned into quartiles by `data_id` order:

```
iters/data_id (retries before a commit)     grad_norm (mean over the 10 committers)
        real      sim                              real        sim
Q1      2.84      2.68   (sim slightly LOWER)       897         904   (<1% apart)
Q2      2.84      2.61                               995         920
Q3      3.44      4.00                              1147        1171
Q4      4.15      5.95   (sim 43% HIGHER)            1272        1525  (~20% apart)
```

Both signals are **flat/matched at the start of the run and diverge increasingly
toward the end** — not a fixed offset present from `data_id=0`. That rules out a
static config/modeling bug (which would show up immediately) and points at
**compounding trajectory drift**: real and sim are two independently-unseeded
stochastic training runs (confirmed already, `Sdet`: "UNSEEDED — decisions are
independent stochastic paths"); each round's cohort pick and each GPU's fp-jitter
is a small independent perturbation, and by data_id ~75+ enough of these have
accumulated that real's and sim's actual model weights sit at measurably different
points, so the actual computed gradients (grad_norm) differ more, which feeds the
variance-gate (`var_good_enough` = split-half variance < `var_threshold`) — a
*higher* grad_norm spread trips the gate more often, needing more `iteration_per_data_id`
retries before a commit, and **every retry costs another full ~11s barrier cycle**
(the max of 10 modeled per-trainer delays) stacked onto that `data_id`'s measured
advance. The arithmetic closes almost exactly: `overhead_residual`'s
`sim_mean_advance_s`/`real_mean_advance_s` ≈ `v1_iter_per_data_id`'s
`sim_mean_iters`/`real_mean_iters` × a **matched** ~11-11.3s-per-iteration constant
on both sides (real 37.71s÷3.35 iters = 11.26s/iter; sim 45.46s÷4.133 iters =
11.0s/iter — the per-iteration COST is fine; only the iteration COUNT diverges).
**`overhead_residual` is fully explained by `v1_iter_per_data_id` for fwdllm/
fwdllm_plus — it is not an independent bug to chase, it's the downstream
consequence of the retry-count gap.** Per §F #3 ("variance is an emergent gate;
localize, never tune it directly") the real question is one level up: why does
sim's grad_norm diverge from real's FASTER/MORE over the run than the reverse
would need to be true for parity — needs checking whether real and sim take
statistically the SAME cohort-selection-driven walk (a `cohort_sequence`-family
question) before concluding this is irreducible unseeded drift vs. a fixable bias.

### `overhead_residual`'s real side is DEFINED to mirror sim's vclock — it cannot see aggregator overhead (session 6, 07-14)

Code, not inference: `_intrinsic_span_s = max(_cycle_speed_s)` (`fwdllm_aggregator.py:2241`), and the comment
says so outright — *"mirrors the sim vclock composition exactly... the FedAvg merge is excluded because the sim
does not charge it to the vclock."* Real's side of `overhead_residual`/`per_round_advance`/`wall_disparity` is
**not** real's measured wall-clock experience — it's the same `max(trainer_speed)` formula sim's vclock uses,
by construction. So §F #1 ("never put overhead on the vclock") currently means the checker never LOOKS for
aggregator overhead either — the rule and the measurement are the same decision, made twice. **Open question,
disagreed with this session, not yet resolved:** should genuinely non-trivial aggregator-side wall time (not
network arrival wait) be (a) folded into the vclock, or (b) engineered away if it's implementation waste rather
than real FL work? Evidence below suggests mostly (b) for fwdllm's specific case, but the general rule question
stands.

**Quantified via the aggregator's own `step_timing` telemetry (emitted, but read by zero existing checks):**

| function | real (real-only clamp) | sim (bulk drain) |
|---|---|---|
| `sync_collect_and_accumulate_grads` | 3350 calls × 1.69s = **16.9s/cycle** | 620 calls × 3.48s = **3.5s/cycle** |
| `_distribute_weights_sync` | 0.37s/cycle | 0.45s/cycle |
| `aggregate` (FedAvg merge) | 0.51s/cycle | 0.58s/cycle |
| measured total (ts-delta) | **~21.4s/cycle** | — |
| what `intrinsic_span_s` counts | **~11.3s/cycle** (`max(trainer_speed)` only) | (vclock is separately modeled) |

Real calls `sync_collect_and_accumulate_grads` **once per individual trainer message** (3350 = 335 cycles × 10),
not once per cycle, because of a real-only clamp: `if self.ends_not_selected_yet and not self.simulated:
num_min_req = min(num_min_req, 1)` (`fwdllm_aggregator.py:2477`) — its own comment: *"commit 1 per pass relies
on uncommitted msgs persisting in the queue... clamping to 1 strands the cohort -> deadlock."* Sim's barrier
(`_sync_sim_recv_first_k`) bulk-drains all 10 in one call. ~10s/cycle (about half the total) is this
architectural difference — Python/logging/telemetry-emission overhead repeated 10× per cycle, not GPU compute
and not network wait. Not yet determined whether it's mostly logging (cheap to cut) or something structural.

### Next-session plan (incorporates the above + follow-up questions raised this session)

1. **Add an aggregator-side `step_timing_breakdown` check** (mirrors the existing trainer-side one, which only
   reads trainer telemetry today) so this ~10s/cycle stops being invisible to the ladder.
2. **Investigate why `overhead_residual` is asymmetric by construction, and why it still diverges even where
   real and sim track "the same" quantity.** Two distinct sub-questions, not yet separated: (a) is
   `max(trainer_speed)`-as-real's-clock the right design at all, given it structurally cannot see real overhead
   (above); (b) even granting that formula, WHY do real and sim disagree on `max(trainer_speed)` itself (not
   just on retry count) — re-check with iteration-count held fixed.
3. **Determine whether the real-only `num_min_req=1` clamp is still load-bearing.** Was it ever needed in
   async_cifar10 (felix/oort don't have it — their sync/async loops bulk-collect)? If fwdllm's queue-drain
   deadlock case doesn't actually require per-message clamping, fix the root cause (bulk-collect like sim does)
   instead of modeling the tax it creates.
4. **Identify what's actually inside the ~10s/cycle**: profile `sync_collect_and_accumulate_grads`'s per-message
   cost — is it logging volume, per-message `build_comm`/telemetry emission, or genuine per-message Python work?
   Trim whatever isn't needed by a parity check or a plot.
5. **Wire seeds everywhere** (all 3 baselines' yamls + selectors) so real→real and real→sim both become
   reproducible — currently NEITHER is: `RandomSelector` supports a `_seed` kwarg
   (`lib/python/flame/selector/__init__.py:56-67`) but no fwdllm yaml passes one, and `FedSgdTrainer.py`'s JVP
   perturbation draw has no seed call anywhere (`trainer/main.py`'s `set_seed` is a separate, flame-unused
   entry point). async_cifar10 has a `*_parity_seeded_{real,sim}.yaml` convention
   (`lib/python/examples/async_cifar10/expt_scripts_2026/`, `hyperparameters.seed: 1234`) to model this on —
   caveat: felix's own seeded-config comment admits oort selection itself may still be unseeded even there, so
   verify what the existing convention actually covers before assuming it's a complete answer. GPU kernel
   non-determinism (cuDNN, atomic-add reductions) is a separate, harder floor even with matched seeds — measure
   the residual gap after seeding before assuming exact match is reachable.
6. **fluxtune: investigate an accuracy/loss degradation after a "round reset" boundary** — raised this session,
   not yet scoped. Hypothesis to check first: confirm model state is NOT being incorrectly reset/rolled back at
   a commit/version boundary (`cached_v` rollback, §F #4) before looking elsewhere. Supporting signal already in
   hand: fluxtune's sim accuracy trajectory is noisy and drops sharply in its last few evals (§A accuracy table
   above, last-5 sim = `[0.658, 0.625, 0.591, 0.617, 0.444]`) — not yet tied to a specific boundary event.

### Per-baseline items NOT already covered by the next-session plan above

`overhead_residual`/`v1_iter_per_data_id`/`step_timing_breakdown`'s shared `_fetch_weights`/`recv_wrapper`
pattern are all folded into the 6-point plan above (items 1-4) — don't re-litigate per baseline. What's left,
baseline-specific:
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

### After fluxtune's genuine residual closes
Fix the parity-CLI progress-axis bug (above) first — it's blocking a trustworthy read on fwdllm/fwdllm_plus's
`overhead_residual`. Then: fluxtune's real 19.2% `overhead_residual` gap → `sim_sct_ordered_drain` A/B →
fwdllm/fwdllm_plus re-measured post-checker-fix → felix (async_cifar10) 46/46 re-confirmation (deferred
repeatedly, do this before trusting felix numbers again) → C1/C2 convergence at matched `data_id` → gate to
Phase 2 (unavailability).

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
   **UNDER REVIEW as of 07-14 (§A, session 6)** — disagreed with this session: the "never" may be too broad:
   genuine non-trivial aggregator work (not network-arrival wait) may belong on the vclock, or should be
   engineered away rather than excluded from measurement. Not yet resolved — see §A's next-session plan.
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
