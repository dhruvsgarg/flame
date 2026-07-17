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
> This is a **living status doc**, not a changelog. §A/§B describe the state **right now** — rewrite in place,
> never stack dated "UPDATE" blocks. Per-run history lives in git + the parity JSONs; the code is the source of
> truth for *what* a mechanism is.
>
> **Score-tracking trigger.** Whenever `run_parity.py` is run over a real/sim pair with wall duration >3600s
> (1h) for one or more baselines, refresh §A's scoreboard in the SAME edit — for EVERY baseline, not just the
> one(s) freshly run (carry the others' last-known numbers forward, tagged STALE, rather than leaving them
> silently outdated).
>
> **§A is a scoreboard, never prose:** per-baseline pass/fail/skip + key-rung ✓/✗, nothing else. **§B is
> next-steps/open-issues, per baseline** — short (1-3 line) entries only, no essays. **§G is CLOSED items,
> ONE LINE each, under ~20 words** (problem → fix, terse) — no paragraphs, no trace dumps, no multi-sentence
> justification; that reasoning belongs in the code comment/commit that landed the fix, not here. **An issue
> lives in EXACTLY ONE place: open (§B) xor closed (§G, one line).** Never both, never neither, never repeated
> across sections in different states (open in one place, closed in another, "deferred" in a third) — when you
> close something, DELETE its §B entry as part of the same edit and add the §G one-liner; don't leave a stale
> copy anywhere. When a chain of hypotheses gets superseded, keep only the FINAL correct one — no wrong turns,
> no "superseded" sections.
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

## §A  Score — refreshed 2026-07-16 (see PREAMBLE's score-tracking trigger)

> **Fresh 1h pairs for ALL THREE baselines**, all at `agg_goal=10` (validates the §G `agg_goal` 3→10 fix —
> see below). fluxtune divisor 0.48/floor 4.0 (min-init=c=30); fwdllm/fwdllm_plus divisor 1.63/floor 11.0
> (min-init=c=10) — first non-STALE fwdllm/fwdllm_plus pair since the delay-floor+seed fixes landed. This
> session's deep-dive is fluxtune-only (operator focus); fwdllm/fwdllm_plus fails are listed but untriaged.

**Latest run per baseline** (`run_parity.py`, `lib/python/examples/fwdllm/expt_scripts`):

| baseline | run pair | duration | pass | fail | skip |
|---|---|---|---|---|---|
| fluxtune/syn_0 | `run_20260716_202633`/`_212850` (delay-floor 4.0, divisor 0.48, min-init=c=30, agg_goal=10) | ~3600s | 62 | 5 | 18 |
| fwdllm/syn_0 | `run_20260716_202655`/`_212840` (delay-floor 11.0, divisor 1.63, min-init=c=10, agg_goal=10) | ~3600s | 55 | 7 | 22 |
| fwdllm_plus/syn_0 | `run_20260716_202850`/`_213102` (delay-floor 11.0, divisor 1.63, min-init=c=10, agg_goal=10) | ~3600s | 56 | 7 | 21 |

**Key-rung status** (✓ pass · ✗ fail · – skip; catalog: `async_cifar10/PARITY.md` §F):

| baseline | cohort | vclock | thru | commits | terminal | R1 | V1 | V2 | U3 | S2 | conv | conv_loss |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fluxtune | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✗ | ✓ |
| fwdllm | ✗ | ✓ | ✗ | ✓ | ✓ | – | ✓ | ✓ | ✗ | – | ✓ | ✓ |
| fwdllm_plus | ✗ | ✓ | ✓ | ✓ | ✓ | – | ✓ | ✗ | ✓ | ✓ | ✗ | ✓ |

**All failing rungs, this run:**
- **fluxtune** (5): `cohort_sequence`, `v2_var_trajectory`, `v1b_iters_moving_avg`, `agg_step_timing_breakdown`,
  `convergence`. **`agg_goal` 3→10 VALIDATED** (§G) — `throughput`/`total_commits`/`terminal_state` all now
  PASS (rel_diff 0.0-0.005, were ~6% over tol) and `convergence_loss` now passes (0.007 vs prior 0.172).
  Remaining 5 are a tight cluster downstream of `cohort_sequence`, re-characterized this session (§B item 1):
  the SET divergence is NOT a tie-window arrival race (only 8/10 members shared at cycle 0, the 2 differing
  pairs are 14.6s apart in registry-expected delay — 14x the 1.0s tie window) and it COMPOUNDS over the run
  (mean cohort overlap decays from 8/10 at cycle 0 to a 520-cycle average of 1.8/10, below the ~3.3/10
  expected from two independent random draws off the same `c=30` pool). `v2_var_trajectory` misses by a hair
  (mean_rel_diff 0.0229 vs 0.02 tol; KS 0.025 well inside 0.2 tol) and `convergence` misses by a hair
  (acc_diff 0.06 vs 0.05 tol, down from 0.153) — both plausibly clear once the cohort cascade is fixed.
  `agg_step_timing_breakdown` re-diagnosed (§B item 2): NOT `_distribute_weights_async` (already exempted,
  real-only sleep) — the actual gating failures are `_process_aggregation_goal_met` (KS 0.365, sim 28% slower)
  and `aggregate` (KS 0.385, sim 32% slower), the same aggregator queue-bound gap as before, now isolated
  cleanly since it no longer also fails the throughput/commits/terminal EXACT rungs.
- **fwdllm** (7): `overhead_residual`, `per_round_advance`, `throughput`, `step_timing_breakdown`,
  `agg_step_timing_breakdown`, `cohort_sequence`, `utility` — first fresh (non-STALE) pair since the
  delay-floor+seed fixes; down from 16 fails on the stale pair. Untriaged this session (fluxtune was focus).
- **fwdllm_plus** (7): `eligibility`, `step_timing_breakdown`, `agg_step_timing_breakdown`, `cohort_sequence`,
  `v1b_iters_moving_avg`, `v2_var_trajectory`, `convergence` — first fresh (non-STALE) pair since the
  delay-floor+seed fixes. Untriaged this session (fluxtune was focus).

See §B for what's actively being worked per baseline; see §G for what's already closed.

---

## §B  Next steps / open issues — per baseline, as of the §A runs above

### fluxtune (~3600s, delay-floor 4.0, divisor 0.48, min-init=c=30, agg_goal=10 — VALIDATED)
1. **`cohort_sequence` SET divergence — ROOT-CAUSED + FIXED, LANDED UNVALIDATED (pending next live pair).**
   Root: `_sim_inflight_expected`'s gate is only armed for a trainer already cached in `_sim_known_delay_s` (a
   reactive, first-observation-only cache) — round 1 is every trainer's first contact, so the gate starts
   empty and cycle 0 commits whichever grad physically lands first instead of the true sct-minimum; 2/10
   objectively-faster trainers (registry rank 5/9 of 30) miss the window and land in cycle 1, and the
   divergence cascades (mean cohort overlap 8/10→1.8/10 over 520 cycles). Fix: `unknown_stuck`, a new gate
   condition in `_sim_recv_min_grad` — hold while a probed end is still delay-unknown and within
   `sim_gate_compute_cap_s` of its dispatch time (adaptive, unconditional, costs wall time once per trainer
   ever). `sim_gate_compute_cap_s` re-derived 16.0→11.0 (stale, pre-overhead-removal). **Fluxtune (async)
   only** — fwdllm/fwdllm_plus's sync `_sync_sim_recv_first_k` is a full-barrier, immune by construction; see
   §G and their own §B entries below.
2. **`agg_step_timing_breakdown` re-diagnosed: the aggregator queue-bound gap, not `_distribute_weights_async`.**
   `_distribute_weights_async` (KS=0.985) is already exempted (`gates_ok=False`, real-only sleep) — the
   `worst_func` field just reports it regardless of exemption, which previously obscured the real gating
   failures: `_process_aggregation_goal_met` (KS=0.365, real 0.226s/sim 0.289s, 28% slower) and `aggregate`
   (KS=0.385, real 0.100s/sim 0.132s, 32% slower). Same GPU-contention aggregator gap as the old §B item 3 —
   now cleanly isolated since throughput/total_commits/terminal_state (the EXACT-tier symptoms it used to also
   trip) all PASS post `agg_goal=10`.
3. `v2_var_trajectory` (mean_rel_diff 0.0229 vs 0.02 tol, KS 0.025 ≪ 0.2 tol) and `convergence` (acc_diff 0.06
   vs 0.05 tol, down from 0.153 pre-`agg_goal=10`) — both near-miss by a hair, plausibly downstream of #1's
   cohort cascade rather than independent bugs. Re-measure once #1 is fixed before treating either as its own
   root cause.
4. `v1b_iters_moving_avg` — `ma_max_abs_dev=0.85` vs `0.75` tol; worst drift at `data_id=81` (near the end of
   the run), consistent with #1's cascade compounding late rather than a fixed offset from the start.
5. `sim_sct_ordered_drain` A/B — unblocked. Run `fluxtune_n10_smoke_sim_no_sct_drain.yaml` against next pair.
6. **Accuracy drop after reaching 81%** — known, deferred by operator (07-15). Not yet triaged.
7. **Real↔real admissibility (§F-5)** — rungs now finalized (tie-window + tiered dep graph, `CHECK_META`
   `deps`); unblocked but deferred until #1's cascade root is found (may change what admissibility should test).

### fwdllm (~3600s, delay-floor 11.0, divisor 1.63, min-init=c=10, agg_goal=10 — first fresh pair)
> First non-STALE pair since the delay-floor+seed fixes landed. 7 fails (down from 16 on the stale pair);
> none individually triaged this session (fluxtune was the operator's stated focus) — next session's queue:
1. **`cohort_sequence` — fluxtune #1's diagnosis does NOT transfer, needs its own investigation.** fwdllm/
   fwdllm_plus are SYNC baselines (`_sync_sim_recv_first_k`, `top_aggregator.py:447`) — a genuine full-cohort
   barrier that drains every selected end before sorting by sct, structurally immune to the async
   `_sim_recv_min_grad` round-1 admission-race bug fluxtune had (§G 07-16). Root cause here is unknown; start
   from a `[SYNC_SIM_BARRIER]`/`[SYNC_SIM_RECV]` log trace of the first cycle, same method as fluxtune's
   `[SIM_GRAD_RECV]` trace, not by assuming the async fix applies.
2. `overhead_residual`/`per_round_advance` — previously root-caused (real-only `num_min_req=1` clamp calls the
   sync collect path once per LAP, not per cycle) but unfixed; needs a compose-loop refactor, risks stranding
   messages if done blind. Re-check the root cause still holds on this fresh pair before refactoring.
3. `throughput`, `step_timing_breakdown`, `agg_step_timing_breakdown`, `utility` — failing, UNEXAMINED this
   session, no root cause yet.

### fwdllm_plus (~3600s, delay-floor 11.0, divisor 1.63, min-init=c=10, agg_goal=10 — first fresh pair)
> First non-STALE pair since the delay-floor+seed fixes landed. 7 fails; none individually triaged this
> session (fluxtune was the operator's stated focus) — next session's queue:
1. **`cohort_sequence` / `v2_var_trajectory` / `v1b_iters_moving_avg` / `convergence`** — same near-miss cluster
   shape as fluxtune #3/#4, but fluxtune's cycle-0-cascade fix does NOT apply (fwdllm_plus is sync, same
   `_sync_sim_recv_first_k` full-barrier as fwdllm — see fwdllm #1 above). Needs its own trace, not inherited.
2. `eligibility`, `step_timing_breakdown`, `agg_step_timing_breakdown` — failing, UNEXAMINED this session.

### Cross-baseline / shared
- **fwdllm/fwdllm_plus `cohort_sequence`, still failing on fresh pairs — root cause is UNRELATED to fluxtune's
  (§G 07-16 async fix doesn't apply to their sync `_sync_sim_recv_first_k` barrier)**, needs independent
  investigation from their own `[SYNC_SIM_BARRIER]` trace.
- **felix (async_cifar10) likely has the same round-1 cold-start gap fluxtune had** — `asyncfl/top_aggregator.py`
  `_sim_recv_min` uses the identical `_sim_inflight_expected`/reactive-`_sim_known_delay_s` gate shape (no
  fallback for unseen ends), same theoretical blind spot on first contact. Felix's own code comment claims the
  gate is empirically "inert" there (real GPU compute ~0.4s wall, felix's `_SIM_RECV_MARGIN_S=0.5s` fallback
  probe already wide enough to catch it) — plausible but UNVERIFIED, not data-checked this session. See
  `simulate_fwdllm.md` §G 07-16 / `fwdllm_aggregator.py`'s `unknown_stuck` gate for the fix pattern if felix's
  own data later shows it's not actually inert. Not implemented for felix — flagged only, per operator (out of
  this session's blast radius: `async_cifar10/PARITY.md` owns felix).
- felix (async_cifar10) 46/46 reconfirmation — deferred repeatedly, gates Phase 2.
- Operator-run seeded real↔real pairs (`*_seeded.yaml`) — GPU-nondeterminism floor; the seed fix (§G) makes
  the default yamls seeded, so these now measure only the GPU-jitter floor.
- Momentum (S1-S3) / fluxtune server-optimizer retry — roadmap item, not parity; see
  `fluxtune_contributions.md` §8.2 / FWDLLM_DESIGN.md. Resume only after Phase-1 parity closes.

---

## §F  Locked principles (from async_cifar10, carried over)
1. **Sim does real forward-grad compute, charges modeled time.** GPU runs the real JVP; the agg stamps
   `sct = sim_send_ts + max(real_gpu_s, D) + leg`. **Never** put overhead on the vclock (`vclock = max(vclock, sct)`).
   **RESOLVED 07-14** — an earlier session questioned whether the "never" is too broad for
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
    on a real wait it should skip, OR (§B, fluxtune) its per-commit processing throughput can't keep pace with
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
    (fluxtune vs felix, §B/§G).
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
18. **Any important knob is logged CONSISTENTLY everywhere, or it's a trap.** If a value is configurable OR an
    always-on correctness path (seed, delay floor, agg_goal, c, availability trace, flag-gate state), it MUST
    surface identically across the yaml, the run snapshot, AND both roles' telemetry — the same resolved value,
    not None on one side. Divergent/missing logging silently breaks run reproducibility and wastes a session
    chasing a phantom (07-16: trainer `seed` logged `None` while the run was genuinely seeded, sparking a false
    "seed regression" hunt; snapshot recorded no seed at all). When you add or wire a knob, add its log on every
    surface in the SAME change and diff a real run to confirm it reads the resolved value, not a default.
19. **No compute on the critical path for a log the run doesn't need.** Logging is for monitoring; the real
    experiment must spend wall time on compute, not on building log strings. Any log whose ARGUMENTS are
    non-trivial (`_calculate_hash`/GPU→CPU `.cpu()`/`.item()`/`.tolist()`, `torch.allclose`/`.norm()`/`stack`,
    a comprehension or repr over params/grads/state_dict) MUST be gated behind `logger.isEnabledFor(logging.DEBUG)`
    (or a purpose flag like `_perturb_audit`) so it computes ONLY when explicitly enabled — an f-string evaluates its
    args even when the level would drop the line, so an ungated `logger.debug(f"...{hash(x)}")` still pays the
    cost. Determinism/correctness audits belong here: verify once with DEBUG on, then run with it off at zero cost.
    The high-perf run keeps at INFO only what plotting/sanity scripts parse (`extract_sanity_checks.py`'s regexes:
    selector `select()`/`_select_candidates`, `_distribute_weights…data_id`, `eval_model(): results after eval`,
    trainer PID, client data hash/samples) + telemetry `emit()`; everything else is DEBUG or deleted. This holds
    for ALL baselines and BOTH roles — grep `logger.(info|debug).*_calculate_hash|format_hash|\.item\(\)` before
    a perf run. Never change a value inside a gated log (reads are inert; gating must stay correctness-neutral).

---

## §G  Landed fixes — one line each (problem → fix). Full history: `git log -- lib/python/examples/fwdllm/simulate_fwdllm.md`.

- **Round-1 cold-start cohort scramble (async only)** (07-16) — `_sim_recv_min_grad`'s gate was blind on a
  trainer's first contact (`_sim_known_delay_s` reactive, no fallback); added `unknown_stuck`, a wall-clock-cap
  hold, instead of oracle-seeding the delay. `sim_gate_compute_cap_s` re-derived 16.0→11.0 (stale). Unvalidated.
- **`cohort_sequence` tie-window for arrival-race divergence** (07-16) — SET/ORDER divergence now granted a TIE, not a fail, when every differing trainer's registry-derived expected delay is within `tie_window_s=1.0` of the others'; ungrantable (no delay model/unknown trainer) stays strict. 6 new tests.
- **`agg_goal` 3→10 for fluxtune's `c=30` pool** (07-16) — K=3 was too tight for a `c=30` near-degenerate fast class, admitting only a coin-flip subset per cycle; `fluxtune_n10_smoke[_sim].yaml`. VALIDATED on the `_202633`/`_212850` 1h pair: `throughput`/`total_commits`/`terminal_state` all PASS (was ~6% over tol) and `convergence_loss` 0.172→0.007. `cohort_sequence` itself still fails — re-diagnosed as a cycle-0 seed cascade, not the `agg_goal=3` contention (§B fluxtune #1).
- **`step_timing_breakdown` degenerate-noise skip was max-gated, not p99-gated** (07-16) — a single GC/cache-miss outlier in a multi-thousand-sample near-zero function (`tb_batch_to_device`) defeated the skip and scored KS=0.3+ on quantization noise; gated on p99 instead. `_emulate_training_delay`/`_fetch_weights` were already correctly exempted since 07-10 — the doc previously misattributed the fail to them.
- **fwdllm/fwdllm_plus yamls renamed `_n10_smoke*` → `_n100_smoke*`** (07-16) — filename only; `num_trainers` was already 100. `run_sequential.sh`'s baseline map + `test_config_generator.py` updated.
- **`minInitialTrainers` defaults to N** (07-16) — `run_sequential.sh` now waits for ALL trainers before the first selection (removes the `>=threshold` pool-size race: real fired at 98, sim at 99 → divergent seeded draw). Set-exact initial cohort; `--min-initial-frac <1` opts back into straggler tolerance. Caveat: blocks forever if a trainer never joins.
- **AVL_TRAIN stamped at registration** (07-16) — `Channel.add` now sets `PROP_AVL_STATE=AVL_TRAIN` so a just-joined end is never read UNKNOWN before the first selection stamp; trace/tracker overwrites per selection (aware) or it stays (unaware). Kills the startup UNKNOWN transient in avail_composition; 4 tests (init→aware-override→unaware-persist).
- **seed added to snapshot.yaml** (07-16) — serializer dropped `hyperparameters.seed` while aggregator_config.json carried it; snapshot's `aggregator` block now records the resolved seed (§F-18 completeness).
- **`eval_model` exempted from `agg_step_timing_breakdown`** (07-16) — daemon-backgrounded, off-vclock, off-critical-path; real↔sim wall gap is pure GPU contention (sim trainers never sleep → GPUs 3.5× denser). Excluded from gating; the still-enforced trainer-side `phase_gpu_compute`/`step_timing_breakdown` catch any bleed into trainer compute.
- **All selectors leaked trainer JOIN order into the seeded draw** (07-16) — async_oort built its candidate list from raw `ends.keys()` (join order) before `_rng.choice`; canonicalized to `sorted(ends.keys())` in oort/async_oort/fedbuff/refl_oort/feddance; selectors now default seed 1234. Root of fluxtune cohort cycle-0 divergence (→ v2/utility/pref). 199 selector + 156 parity tests pass; validate on next fluxtune rerun.
- **`v1b_iters_moving_avg` rung added** (07-16) — moving-avg iters-per-data_id over the whole run within a tight bound; catches trajectory DRIFT that v1's pooled KS+mean cancels. FAILs the current pre-fix fluxtune pair (sim 14.5 vs real 13.6).
- **Trainer compute re-measured post overhead-removal** (07-16) — genuine JVP is mean 0.47s / max 6.1s (was 3.63s; ~87% was harness hashing/gc/logging), matches real↔sim ~1%; drove the §O floor re-derivation 7.0→4.0 (divisor unchanged). Detail in FWDLLM_DESIGN §O.
- **Trainer wall-time attribution — READ, no anomaly** (07-16) — n100 `_train_one_batch` 354ms real ≈ 349ms sim, `tb_*` account ~99%; prior 3952ms didn't reproduce.
- **Eval is NOT the residual (corrects a same-day wrong call)** (07-16) — eval is daemon-backgrounded both modes; `_process` not slowed during eval (0.4×), cadence +11%; residual is dispatch-order + raw-wall/vclock, not eval.
- **Aggregator is queue-bound, not 350ms** (07-16) — send→ingest 6.8s real/26s sim = queue wait (70% busy, serial commits, in-flight cap c=30), not MQTT transit; gates `sim_rate`.
- **Perturbations validated deterministic across modes** (07-16) — utility matches to 5 d.p. when aligned; divergence is dispatch ORDER (which data_id per trainer), onset median 1 update.
- **Trainer `seed` telemetry logged `None`** (07-16) — seed lived only in aggregator config; added `seed: 1234` to trainer `config_overrides` in all 6 base yamls (inert: trainer self-selects), confirms next run.
- **Aggregator per-commit waste removed** (07-16) — `FedSgdAggregator.aggregate` did the SAME full-model `deepcopy` block ×3/commit (copy-paste; only last used) → collapsed to 1; gated ~8 eager `_calculate_hash` GPU→CPU sha256 (debug strings built even at INFO) across `aggregate`+`fwdllm_aggregator`. 267 fwdllm tests pass; next run measures the `aggregate`/`_process` wall drop.
- **`--min-initial-frac` startup-barrier lever added** (07-16) — `run_sequential.sh` computes `minInitialTrainers=floor(frac·N)` into `selector.kwargs` (all selectors); opt-in A/B for the dispatch-order root (§B fluxtune #4), unchanged when unset.
- **fluxtune `preferred_duration` + `avail_composition` PASS post-fix** (07-16) — seed/pacer fix cleared pref (real 0.273→match); AVL-at-registration drove avail UNKNOWN→0. Confirmed on the `_161235`/`_161255` rerun.
- **Sim ran UNSEEDED while real had `seed=1234`** (07-15) — sim yamls omitted the key; added to all 3 + `config.py` default `None`→`1234`; drove `cohort_sequence` set-match to 0.0 at cycle 0.
- **`_handle_recv_state` leaked dispatch order via PYTHONHASHSEED** (07-15) — `select_random`'s `dict.fromkeys` fix never reached it; ported to async_oort/fedbuff/async_random.
- **Trainer batch interior emitted zero telemetry** (07-15) — `timer_decorator` keys off `args[0].fwd_llm_stage`; nested helpers pass `device`; added `_stage_timer` + 10 `tb_*` phases + phase/unaccounted CDFs.
- **`agg_step_timing_breakdown` false positives** (07-15) — KS on all-zero + tight distributions; degenerate-skip, 5% mean escape, exempt `_distribute_weights_async` (real-only `sleep(0.1)`).
- **`aggregation_plots` dead on NameError** (07-15) — `pc_x`/`pc_y`/`pgm_y` collection loop missing; restored, `pastdated_commits_over_rounds.pdf` renders again.
- **TIMING_OVERRUN** (07-15) — §O's margin used fast-class MEAN not FLOOR; `training_delay_floor_s` fix; VALIDATED 0 overruns (5400s run).
- **fluxtune accuracy floor** (07-14/15) — cross-refs `fluxtune_contributions.md` §8's tracked H0/F1-F15 collapse; not a parity bug, both legs match.
- **`r1_inflight_overlap`** (07-15) — checker flagged FedBuff's legit stale-accept redispatch as a violation; rescoped per `version_key`; FIXED, 0.0%/0.0%.
- **`_sim_gate_compute_cap_s`** (07-15) — blind `10.0` thinner than observed max; derived `16.0` for fluxtune, `10.0` fallback documented as non-universal.
- **`select_random` order leaked via PYTHONHASHSEED** (07-14) — `set()`→`dict.fromkeys()`; VALIDATED, fwdllm `cohort_sequence` 100% match.
- **fluxtune `preferred_duration`** (07-14) — oort pacer one-branch port bug; faithful both-branch port; 50.7pp→9.3pp gap (see §B for a 07-15 borderline re-open).
- **Parity-CLI progress-axis bugs** (07-14) — axis picked per-side independently, glob collided fwdllm/fwdllm_plus; prefer `data_id`, anchor glob on `_{tag}_n<N>_`.
- **Aggregator `step_timing` unparsed** (07-14) — zero checks read it; added loader capture + `agg_step_timing_breakdown` rung.
- **fwdllm had no seeded yaml** (07-14) — seed plumbing was already correct, just unexercised; added 6 seeded yaml pairs.
- **`recv_fifo` hot path logged at INFO** (07-14) — 425k lines/run, unread; downgraded 9 mechanical lines to DEBUG.
- **Server-momentum (S1)** (07-14) — landed flag-gated (`server_momentum`, default 0.0 no-op) + A/B yamls; NOT run, resume deferred to `fluxtune_contributions.md` §8.2.
- **Reactive gate blocked real wall** (Bug A, 07-13) — re-checked stale state after a blocking call; `_sim_gate_is_safe` checks first; `sim_rate` 0.97→1.82×.
- **Carried-surplus commits misclassified "round1"** (Bug B, 07-13) — classifier wasn't re-keyed to `data_id`; ingest-time carry-over stamp, separate bucket.
- **`eval_model()` blocked dispatch** (07-13) — sync eval stalled the loop; backgrounded on a daemon thread, dead assignment removed.
- **fwdllm_plus livelock** (07-13) — `RandomSelector` freed only `k=5` of `c=10`; removed the stale `k` knob entirely.
- **fluxtune commit-path stall** (07-13) — phantom `_sim_inflight_expected` entry (stamped, not computing); `sim_compute_truthful_gate` skips stale dispatches.
- **fluxtune cohort-SET divergence** (07-13) — real released a busy trainer's guard on RETURN not commit; hold-to-commit + `send_timeout_wait_s=300`.
- **`version_key` unification** (07-13) — version identity was bare-int in some places, 3-tuple in others; one shared 2-tuple property everywhere.
- **Remainder-wait delay model** (07-13) — additive `send+gpu+D` gave nondeterministic arrival order; real sleeps `max(0,D-gpu)`, sim never sleeps D.
- **Slot residence hold-to-COMMIT** (07-13) — release-on-RETURN undercounted in-flight state 3×; hold slot until commit, sync+async.
- **Async surplus-grad handling** (07-13) — dropping at agg-goal boundary wasted ~7 grads/cycle at `c≫agg_goal`; carry surplus + hold busy trainers.
- **Async `total_commits`/`throughput` overlap bug** — summed overlapping cycles as sequential (76-86% spurious diff); fall back to raw wall for async.
- **Clock-rate rungs anchored on transport artifact** — used full wall (localhost-only latency) instead of `intrinsic_span_s`.
- **`cohort_sequence_parity` conflated SET/CADENCE/VAR/ORDER** — one cap tripped on real GPU fp16 jitter; SET hard/uncapped, rest cap to bin 1.
- **"GPU under-provisioned at n=10"** — refuted; spawn table is balanced round-robin, 8 GPUs, 1 core/trainer.
