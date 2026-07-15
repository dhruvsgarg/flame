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

## §A  Score — refreshed 2026-07-15 (see PREAMBLE's score-tracking trigger)

**Latest run per baseline** (`run_parity.py`, `lib/python/examples/fwdllm/expt_scripts`):

| baseline | run pair | duration | pass | fail | skip |
|---|---|---|---|---|---|
| fluxtune/syn_0 | `run_20260715_154328`/`_154352` | 5400s | 53 | 10 | 18 |
| fwdllm/syn_0 | `run_20260715_000924`/`_021109` — **STALE**, pre-delay-floor-fix | 7200s | 47 | 16 | 21 |
| fwdllm_plus/syn_0 | `run_20260715_030542`/`_050735` — **STALE**, pre-delay-floor-fix | 7200s | 53 | 8 | 21 |

**Key-rung status** (✓ pass · ✗ fail · – skip; catalog: `async_cifar10/PARITY.md` §F):

| baseline | cohort | vclock | thru | commits | terminal | R1 | V1 | V2 | U3 | S2 | conv | conv_loss |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fluxtune | ✗ | ✓ | ✗ | ✗ | ✗ | ✓ | ✓ | ✗ | ✓ | ✓ | ✗ | ✓ |
| fwdllm | ✗ | ✓ | ✗ | ✗ | ✗ | – | ✗ | ✗ | ✓ | ✓ | ✗ | ✓ |
| fwdllm_plus | ✗ | ✓ | ✗ | ✗ | ✗ | – | ✓ | ✗ | ✓ | ✓ | ✗ | ✓ |

**All failing rungs, this run:**
- **fluxtune** (10): `overhead_residual`, `per_round_advance`, `throughput`, `preferred_duration`,
  `agg_step_timing_breakdown`, `cohort_sequence`, `v2_var_trajectory`, `terminal_state`, `total_commits`,
  `convergence`
- **fwdllm** (16): `trainer_speed`, `overhead_residual`, `per_round_advance`, `throughput`,
  `avail_composition`, `eligibility`, `training_budget`, `step_timing_breakdown`, `agg_step_timing_breakdown`,
  `cohort_sequence`, `v1_iter_per_data_id`, `v2_var_trajectory`, `utility`, `terminal_state`, `total_commits`,
  `convergence`
- **fwdllm_plus** (8): `throughput`, `step_timing_breakdown`, `agg_step_timing_breakdown`, `cohort_sequence`,
  `v2_var_trajectory`, `terminal_state`, `total_commits`, `convergence`

See §B for what's actively being worked per baseline; see §G for what's already closed.

---

## §B  Next steps / open issues — per baseline, as of the §A runs above

### fluxtune (5400s, post-delay-floor-fix)
1. **`cohort_sequence` total divergence — top priority.** Real/sim pick entirely different trainers from
   cycle 0 (0% set match) despite sharing the RNG-order fix that gives fwdllm/fwdllm_plus ~100%. Not traced.
2. **`throughput`/`per_round_advance`/`terminal_state`/`total_commits`/`overhead_residual` — one gap.** Sim
   completes 139 rounds vs real's 103-117 in the same matched vclock budget (~24-26% rel diff); `sim_rate`
   itself is 0.88× (<1). `r1_inflight_overlap` is ruled out as a cause (§G) — look elsewhere, start with #4.
3. **`preferred_duration` — borderline, re-check.** frac_diff exactly at the 20pp tolerance (real 23.9% vs sim
   3.9%) — worse than the 9.3pp this rung hit before. Could be this run's shorter length (5400s vs 7200s) or a
   genuine drift; re-check at matched length before concluding either way.
4. `agg_step_timing_breakdown` — worst offender `_distribute_weights_async` (real 0.166s vs sim 0.046s,
   KS=0.976); `_release_end_on_return`/`eval_model`/`_force_cuda_memory_cleanup` also diverge. Candidate
   explanation for #2, unexamined.
5. `v2_var_trajectory` — sim's mean post-perturbation variance 11.94 vs real's 10.21 (14.5% rel, tol 2%).
   Unexamined.
6. `convergence` — both legs at the ~0.25 accuracy floor; tracked as an ML-stability issue in
   `fluxtune_contributions.md` §8 (H3/S2 fixes pending there), not a parity bug — don't re-open here.
7. `sim_sct_ordered_drain` A/B — unblocked (TIMING_OVERRUN + `r1_inflight_overlap` both closed, §G); behind
   #1 so it isn't confounded. Command + what to read: `fluxtune_n10_smoke_sim_no_sct_drain.yaml`.

### fwdllm (STALE run — pre-delay-floor-fix)
1. **TIMING_OVERRUN validation not yet launched** (`--delay-floor 11.0`) — everything below is stale until it
   lands:
   ```
   lib/python/examples/fwdllm/expt_scripts/run_sequential.sh --only fwdllm,fwdllm_plus \
       --mode both --delays on --delay-divisor 1.63 --delay-floor 11.0 --num-gpus 8 --max-runtime-s 7200 \
       --num-trainers 100 --c 10 --min-initial-trainers 10 --after parity,sanity,plot
   ```
2. `overhead_residual`/`v1_iter_per_data_id` — root-caused (real-only `num_min_req=1` clamp calls the sync
   collect path once per LAP, not per cycle) but unfixed; needs a compose-loop refactor, risks stranding
   messages if done blind.
3. `cohort_sequence` — failing; not cross-checked against the RNG-order fix's actual effect at this run/scale.
4. `trainer_speed`, `avail_composition`, `eligibility`, `training_budget`, `utility` — failing, UNEXAMINED
   this session, no root cause yet.
5. `per_round_advance`, `throughput`, `step_timing_breakdown`, `agg_step_timing_breakdown`, `terminal_state`,
   `total_commits`, `convergence` — failing, not individually triaged; re-check after #1 lands before digging
   into any of these (may resolve or reshuffle once the run is fresh).

### fwdllm_plus (STALE run — pre-delay-floor-fix)
1. **TIMING_OVERRUN validation not yet launched** (same command as fwdllm above) — everything below is stale
   until it lands.
2. `cohort_sequence` — failing; not cross-checked against the RNG-order fix's actual effect at this run/scale.
3. `throughput`, `step_timing_breakdown`, `agg_step_timing_breakdown`, `terminal_state`, `total_commits`,
   `convergence` — failing, not individually triaged; re-check after #1 lands.

### Cross-baseline / shared
- felix (async_cifar10) 46/46 reconfirmation — deferred repeatedly, gates Phase 2.
- Operator-run seeded real↔real pairs (`*_seeded.yaml`) — GPU-nondeterminism floor; blocked on fluxtune #1
  above (dispatch-order determinism).
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

---

## §G  Landed fixes — one line each (problem → fix). Full history: `git log -- lib/python/examples/fwdllm/simulate_fwdllm.md`.

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
