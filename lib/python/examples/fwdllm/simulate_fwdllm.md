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

## §A  Current status — 2026-07-13, one baseline at a time (fluxtune first)

**Basis:** 1200s real+sim pairs, `--delays on --delay-divisor 0.5 --num-gpus 8`
(`run_20260712_234246` through `run_20260713_005918`). Parity reports:
`experiments/_parity_reports/parity_{fwdllm,fwdllm_plus,fluxtune}*_20260713_LATEST.json`.

| baseline | `sim_rate` | verdict |
|---|---|---|
| **fwdllm** (sync) | **5.71×** | Healthy. Only fail: `per_round_advance` (K3, 3.9% mean gap) — minor, not investigated this pass. |
| **fwdllm_plus** (sync) | **5.69×** | The 07-12 livelock (see old §G) is CONFIRMED FIXED — real progress throughout, no stall. Residual: `throughput`/`terminal_state`/`total_commits` all fail ~13% (sim completes more rounds than real at matched budget); `overhead_residual` fails (sim advances *faster* than real per round, i.e. sim under-counts overhead real pays). Root not yet isolated — **next baseline after fluxtune**. |
| **fluxtune** (async) | **0.97× at last measurement — fix LANDED, not yet re-validated by a run** | Root: reactive gate blocked real wall unnecessarily (below). Fix landed 2026-07-13 (session 3); the A/B run below is now also the validation run for it — re-run before treating this row as closed. |

### fluxtune: verified root (code-checked, not docstring-trusted)

**Not primarily the `drain_ready` poll fallback.** Both real and sim commit **exactly one grad per
composer-loop tick** by architecture — real's `_aggregate_grads_async` does
`next(channel.recv_fifo(ends, 1, timeout=30))` ([fwdllm_aggregator.py:1246](../../flame/mode/horizontal/syncfl/fwdllm_aggregator.py#L1246));
sim's `_sim_recv_min_grad` also returns exactly one via `_sim_buffer.pop_min()`. This is true regardless of
which channel-drain primitive fills the buffer.

**Verified via telemetry:** `buf_depth` (grads already arrived, sitting in `_sim_buffer` waiting to be popped)
is **7-8 out of ~10 at 592/598 commits** — the buffer is almost always near-full. `SIM_GRAD_STUCK_EVICT`
(30s failsafe) never fires; `phantom_skip` (the compute-truthful gate) is always 0. So the gate/hold logic
isn't the driver either. The real mechanism: sim's arrivals are faster/more synchronized than real's (real GPU
compute ~3.7s, no modeled-delay sleep in sim mode) — faster than the aggregator's own per-tick overhead
(`aggregate()` mean 1.8s, tail to 16s) can drain them one at a time — so a persistent backlog forms and a
trainer waits behind however many others are ahead of it. `recv_wrapper` (trainer wait for next dispatch,
pure real wall, zero vclock credit) confirms it: fluxtune mean **15.8s**, p50 **14.2s**, max **48.3s** (~3×
one 16s cycle) vs fwdllm/fwdllm_plus mean ~6s, max ~18s.

**Redispatch batching divergence (correctness trace, data_id=0):** real batches all 3 freed slots from one
agg-goal cycle into **one** `_distribute_weights_async` call (`|ends|=3`); sim fires **three separate**
`|ends|=1` calls, 0.2–4.7 real-seconds apart. The variance-recompute cadence itself is identical in both
(var correctly held constant across exactly 3 redispatches, matching `agg_goal=3`) — this is a
packaging/timing artifact of the one-grad-per-tick loop, not a variance-check bug.

**`_sim_recv_min_grad` vs felix's `_sim_recv_min` — verified diff, not doc-trusted.** Core buffer+gate+pop+
clock-jump control flow is structurally parallel. Real differences: fluxtune adds a compute-truthful
phantom-skip gate felix lacks (inert here, `phantom_skip=0`); fluxtune popped `_sim_buffer` directly instead
of through felix's `_sim_pop_committable`, so it inherited **none** of felix's past-dated-commit tracking —
**now ported** (`_sim_pastdated_commits`/`_sim_pastdated_gap_max`/`_sim_pastdated_by_source`, folded into the
`[SIM_GRAD_RECV]` log line) so the open question below is actually measurable.

**Open question — is `sim_sct_ordered_drain` (drain_ready) even needed?** It's the only place fluxtune
diverges from felix's default: felix never sets this flag, uses `recv_fifo` unconditionally (verified —
`grep -rl sim_sct_ordered_drain lib/python/examples/*/expt_scripts/*.yaml` matches only fluxtune's yaml). The
stated justification (`recv_fifo`'s background streamer can strand a message under heavy async churn) is
plausible for fluxtune specifically — it re-probes the same end every composer tick (~600×/run) vs felix's
per-round cadence — but there's no direct evidence in the current (drain_ready-ON) run that stranding would
actually occur without it. Config default kept `true` for both baselines (maintainability) pending the re-run
below — this is a config choice, not settled by evidence for fluxtune specifically.

**Root cause, fully isolated (session 3, 2026-07-13) — the reactive gate, not the drain primitive.** Both
`_sim_recv_min`/`_sim_recv_min_grad` re-derive `earlier_stuck` AFTER an unconditional blocking ingest call,
every pass, even when the buffered minimum is already provably safe to commit from in-memory state alone (the
deterministic per-trainer delay cache, §M). Measured: commit-to-commit wall delta when something was already
buffered-and-ready — mean 2.09s, p90 4.1s, max 12.4s, on 99% of commits. **Fixed**: `_sim_gate_is_safe`
(shared, `syncfl/top_aggregator.py`) checks safety BEFORE the ingest call; when safe, the call still happens
(preserves eager-drain of a physically-ready/near-ceiling straggler) but with a near-zero timeout
(`_SIM_GATE_FAST_PROBE_TIMEOUT_S=0.01`) instead of the full per-trainer bound. Landed on both felix's
`_sim_recv_min` and fwdllm's `_sim_recv_min_grad`; 1072 tests pass. **Not yet validated by a live run** (no
broker in the dev environment) — this is what the A/B run below now also validates.

### Next step: A/B run (ready to launch, not yet run — now doubles as the Bug-A-fix validation run)

`fluxtune_n10_smoke_sim_no_sct_drain.yaml` — identical to `fluxtune_n10_smoke_sim.yaml` except
`sim_sct_ordered_drain: false` (legacy `recv_fifo` path, matching felix's default), delays/runtime matched to
the 07-13 baseline (`enable_training_delays: true`, `training_delay_factor: 0.5`, `max_runtime_s: 1800`).

```
python -m flame.launch.run_experiment \
    lib/python/examples/fwdllm/expt_scripts/fluxtune_n10_smoke_sim_no_sct_drain.yaml
```

Read after (both legs, drain ON and OFF): **`sim_rate`** (now expected to recover materially above 0.97×,
toward fwdllm's ~5.7× — this is the actual prediction the reactive-gate fix makes, superseding the older
"stays the same" prediction below that assumed the drain primitive was irrelevant); **`buf_depth`** (expected
to trend toward 0 between arrival bursts instead of sitting at 7-8/10); **`pastdated_commits`** (corrected,
carried-surplus-excluded counter — expected near 0); **`carried_surplus_commits`** (expected: the MAJORITY
commit source, stepping once per data_id boundary — §F #17, this is healthy, not a regression);
`pastdated_n`/`pastdated_gap_max` in `[SIM_GRAD_RECV]` still answers the drain-primitive question (a
materially higher count with the flag OFF means `recv_fifo`'s streamer really does strand/lap messages under
fluxtune's probe frequency). Re-run the baseline (`sim_sct_ordered_drain: true`) too for a matched pair — same
instrumentation is new on both legs.

### After fluxtune closes
One baseline at a time: fluxtune → fwdllm_plus's `overhead_residual`/throughput gap → fwdllm's minor
`per_round_advance` residual → felix (async_cifar10) 46/46 re-confirmation (deferred twice, do this before
trusting felix numbers again) → C1/C2 convergence at matched `data_id` → gate to Phase 2 (unavailability).

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

**Phase 1 (syn_0) — in close-out.** fwdllm healthy. fwdllm_plus's livelock fixed, throughput-gap residual open.
fluxtune's `sim_rate<1` root isolated to one-grad-per-tick throughput (§A); A/B pending to settle whether
`sim_sct_ordered_drain` is load-bearing. Exit: all 3 baselines' `sim_rate`/throughput/terminal_state pass, then
C1/C2 convergence at matched `data_id`.

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

**Reactive gate blocked real wall unnecessarily (Bug A, session 3, 07-13) — FIXED, not yet run-validated.**
`_sim_recv_min`/`_sim_recv_min_grad` re-derived `earlier_stuck` AFTER an unconditional blocking ingest call,
every pass, even when the buffered minimum was already provably safe from in-memory state (the deterministic
per-trainer delay cache, §M) — mean 2.09s/p90 4.1s/max 12.4s wasted per commit, 99% of commits, the root of
fluxtune's `sim_rate<1` (§A). Fix: `_sim_gate_is_safe` (shared, `syncfl/top_aggregator.py`) checks safety
BEFORE the ingest call; when safe, the call still happens (preserves eager-drain of a physically-ready/
near-ceiling straggler — narrowing to "skip the call outright" broke pre-existing tests guarding that
behavior) but with a near-zero timeout instead of the full per-trainer bound. Shared by both felix and
fluxtune. *Lesson:* an optimization that only reasons about ONE of a gate's two justifications (here:
`earlier_stuck` HOLD, vs. the separate "eager-drain a ready/near-ceiling straggler" reason) will break tests
guarding the other one — read every existing test on the touched function before narrowing its behavior.

**Carried-surplus commits 100% misclassified as "round1" (Bug B, session 3, 07-13) — FIXED.** felix's
round-axis past-dated-source classifier was copy-pasted into fwdllm without re-keying to fwdllm's actual
progress axis (`data_id`, not `round` — `round` only advances once per 150-`data_id` lap, never reached in any
smoke run). Every past-dated commit at a data_id boundary (the carried-surplus case, deliberate by design for
`c ≫ agg_goal` — see §F #17) fell into the `_cur_round <= 1` "round1" bucket, hiding the real mechanism. Fix:
`_sim_enqueue_data_id` stamp at ingest time detects a carried-over item; bucketed `carried_surplus` and
excluded from the primary `pastdated_commits`/`gap_cum`/`gap_max` (which should read ~0) — tracked separately,
non-alarming. *Lesson:* a classifier ported between two baselines that share code but not a progress axis is a
straight bug, not a design choice — re-verify axis, not just shape, when reusing across baselines.

**fwdllm's `eval_model()` backgrounded; a real race fixed first (Part 6, session 3, 07-13).** `eval_model()`
ran synchronously, inline, on every data_id boundary — stalling both aggregation dispatch and trainer idle
time, and requiring `sim_model_eval_time`'s vclock fold (real paid the wall, sim didn't). Backgrounding it
naively would have raced: `eval_model()` reassigned `self.fmodel/self.params/self.buffers` via
`fc.make_functional_with_buffers(self.model)` — the SAME attributes the main training path assigns right
before `self.aggregate()` on every cycle — a call whose (fmodel, params, buffers) OUTPUT was never read
anywhere in `eval_model()`'s own body (confirmed by reading
`torch._functorch.make_functional.FunctionalModuleWithBuffers._create_from`: it deep-copies internally, never
mutates the model passed in — the assignment only existed to clobber shared state). Fix: deleted that dead
assignment, added a `model=` param (default `self.model`), and backgrounded `eval_model()` on a daemon thread
mirroring async_cifar10's `evaluate()` — reusing the already-shared `_eval_snapshot_model()`/eval-inflight
guard, no new snapshot machinery needed. `sim_model_eval_time` removed entirely (field, both call sites, all 5
smoke yamls) — the asymmetry it corrected for no longer exists once neither mode pays eval on the critical
path. *Lesson:* "backgrounding" isn't just wrapping a call in a thread — grep every attribute the backgrounded
function writes for a second writer on the main path first.

**Shared agg/eval vclock-fold scope correction.** `sim_model_agg_compute_time` (and the now-removed
`sim_model_eval_time`) live in shared `_process_aggregation_goal_met`, active in ALL THREE fwdllm-family
baselines' configs, not fluxtune-specific. Worth revisiting fwdllm's own `per_round_advance` (K3) residual
(§A) against this now that eval no longer contributes to `intrinsic_span_s` at all (Part 6) — unconfirmed
follow-up lead, not yet investigated.

**fwdllm_plus livelock (was the leading §A item pre-07-13) — FIXED, validated 07-13.** `RandomSelector`
freed only `min(N, k=5)` of a full `c=10` cohort per cleanup call — a stale absolute batch-size knob from the
selector's original n≈150 design; `async_oort` had already hit and fixed this exact class
(`_cleanup_recvd_ends` drains ALL received ends). Fix: `k` was redundant (no other selector has the concept) —
removed from `RandomSelector` + the 4 smoke yamls entirely, not just retuned. *Lesson:* when a selector-level
batch cap silently mismatches a full-cohort config, check whether the cap is even a real concept elsewhere
before retuning its value.

**fluxtune commit-path stall — three superseded framings before the real root** (GPU-pipelining loss / re-
dispatch-on-RETURN / hold-to-commit-is-too-restrictive — all wrong, see git history). Final root at the time:
the drain gate blocked real wall on a phantom `_sim_inflight_expected` entry (stamped-expected-at-dispatch but
idle-in-recv, not genuinely computing). Fix: `sim_compute_truthful_gate` skips an expected entry whose last
dispatch predates `sim_gate_compute_cap_s`. **Superseded again, 07-13** (§A): with this gate landed,
`phantom_skip=0` throughout the current runs — it's not what's driving `sim_rate<1` anymore; the actual driver
is one-grad-per-composer-tick throughput vs sim's faster/more-synchronized arrival rate. *Lesson:* a fix that
demonstrably worked for its own symptom (STUCK_EVICT→0) can still coexist with a different, larger unfixed
bottleneck — re-measure the top-line metric (`sim_rate`) after every "fix," don't just confirm the specific
telemetry the fix targeted.

**fluxtune async cohort-SET divergence (#S1) — real dispatching to still-busy trainers.** Real released a busy
trainer's re-pick guard on physical RETURN, not commit; `async_oort`'s 90s abandon had no liveness check, so
real re-dispatched fresh work to still-busy trainers, desyncing cadence. Fix: unconditional hold-to-commit +
configurable `send_timeout_wait_s` (300 for fluxtune, vs default 90 tuned for felix's shorter rounds).
*Refuted along the way:* "GPU-vs-D headroom collision" and "more delay headroom closes it" — a 0.25-divisor
diagnostic kept D≫gpu and the SET still diverged at the same iteration, ruling out headroom.

**`version_key` unification.** version/staleness/no-repeat identity was a bare `model_version` int in some
places, an inconsistent 3-tuple in others. Fix: one shared `version_key` property (2-tuple) across
trainer/aggregator/selector, both examples. Any new code comparing/stamping a version must go through it.

**Remainder-wait delay model (sct = `send + max(gpu, D)`, not additive `send + gpu + D`).** Additive didn't
give a deterministic per-trainer arrival order (needed for real↔sim identical commit order). Fix: real sleeps
`max(0, D−gpu)` so device wall = D with GPU hidden inside it; sim never sleeps D (charged to vclock only).
Overrun (gpu > D) flagged as `[TIMING_OVERRUN]`.

**Slot residence: hold to COMMIT, not release-on-RETURN.** A returned-but-uncommitted trainer is still in
flight in virtual time (its grad only commits when vclock reaches its sct); releasing on physical return
undercounted in-flight state 3×. Correctness check, not a throughput lever — confirmed for both sync and async.

**Async surplus-grad handling: carry, don't drop, when `c ≫ agg_goal`.** Dropping at the agg-goal boundary was
benign only when `c≈agg_goal` (sync). fluxtune (`c=10`, `agg_goal=3`) dropped ~7 grads/cycle → ~2× the passes
needed. Fix: commit-then-carry the surplus + hold busy trainers across the boundary.

**Checker/telemetry-only fixes worth remembering (not mechanism bugs):**
- `total_commits`/`throughput` async checker cumulative-summed overlapping async cycles as if sequential →
  76-86% spurious rel_diff. Fix: fall back to raw wall for async (same as async_cifar10 already does).
- Clock-rate rungs anchored real's elapsed time on full wall (carries a localhost transport artifact) instead
  of `intrinsic_span_s`. *Lesson:* verify a checker's own anchor before trusting a real↔sim gap it reports.
- `cohort_sequence_parity` conflated SET/CADENCE/VAR/ORDER under one bin cap, demanding exact cadence past a
  genuine nondeterminism wall (~1e-3 GPU fp16 jitter, amplified by the split-half variance ratio, flips the
  `var<0.3` gate at a sensitive bin — not a bug). Fix: SET stays HARD/uncapped; CADENCE/VAR/ORDER cap to bin 1;
  beyond bin 1 the target is DISTRIBUTIONAL, not exact.
- "GPU under-provisioned at n=10" — refuted by the spawn table (`spawner.py`: balanced round-robin, 8 GPUs,
  1 core/trainer pinned). A misread of a GPU-compute-seconds log field as a device id. *Lesson:* confirm
  pinning from the spawn table, not a grep of timing logs.

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
the old 0.5 default. **Not yet validated** — next run after landing should read `training_overran`/
`remaining_time_s` from telemetry (per baseline, per speed_class) before trusting the calibration, per the
fast-class margin flagged above.
