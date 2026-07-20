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
> no "superseded" sections. **The moment a rung flips fail→pass, or a hypothesis is confirmed/refuted, move it
> to §G in the SAME edit** — under 30 words, mechanism + outcome only, no investigation narrative. Don't let a
> closed item linger described in §A/§B prose "for context"; §G is where it lives now.
>
> **§B bullets are updated IN PLACE, never appended-to.** A bullet is a max-75-word CURRENT STATE, not a log —
> when new evidence lands, rewrite the bullet to fold in whatever from the prior text still matters, drop what's
> superseded, and state plainly if the new evidence REFUTES the old claim (don't just tack the refutation onto
> the end and leave the refuted claim standing). If it can't fit in 75 words, it's not open-issue tracking
> anymore — the extra detail belongs in the code comment/commit, and the bullet should just point at it.
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

## §A  Score — refreshed 2026-07-20 pm (see PREAMBLE's score-tracking trigger)

> **Third data point: fresh 7200s (2h) triple**, same launch script/knobs as the two prior sessions, first run
> at this scale since 07-18. This session: (1) `agg_step_timing_breakdown`'s thread-local `time.thread_time()`
> cpu/wall diagnostic (landed 07-20 am, previously unexercised) finally has a real signal, cross-baseline: real
> cpu/wall ≈0.90-1.00 (thread busy ~the whole window) vs sim ≈0.82-0.89 for every residual function, on ALL 3
> baselines — reopens contention as a candidate via a DIFFERENT mechanism (thread scheduling, not the
> already-refuted GPU-pass-window overlap) — see the §B tracker table. (2) fluxtune's `pacer()` once-per-round
> fix (§G, 07-20 am) VALIDATED live: `preferred_duration` passes (frac_diff 0.152 vs 0.2 tol). (3) fwdllm_plus's
> `throughput`/`per_round_advance` 1h marginal fails CONFIRMED as sample-size variance, not a regression — both
> pass clean at 2h (4.9%/matched 0.5%). (4) fwdllm's `throughput` now PASSES clean for the first time (matched-
> window gate holds at 2h scale, 4.9%/matched 0.5%; `per_round_advance`'s KS sub-check still fails, see the §B
> tracker table). (5) Two NEW fluxtune fails at this scale: `sim_send_ts` and `trainer_speed_identity`'s `utility`
> component (23/100 trainers >10% deviation, previously closed 07-19 as non-reproducing — now reproduces).
> (6) fwdllm's `v2_var_trajectory` regresses (9.93% vs 2% tol, was passing at 1.4% at 1h scale). **Follow-up
> same day (pm-2)**: root-caused 3 of these from telemetry already on disk and landed 2 fixes (§G) —
> `sim_send_ts` was the EOT/shutdown broadcast omitting the stamp by design when avail-tracking is off, not a
> mid-run gap; FIXED (stamped unconditionally now), 591/591 `tests/mode` pass. `var_calc`'s DEBUG telemetry was
> also mis-gated at INFO since 07-19, silently taxing `_compute_var`'s own measured time in every run; FIXED.
> fwdllm/fwdllm_plus's `v2_var_trajectory` is the SAME sim-outruns-real population artifact as `overhead_
> residual`/`throughput`, not a separate mystery (`var_calc` shows 0 mismatches on every shared call); fluxtune's
> tie-cascade "confirmed legitimate" framing (§G 07-17d) does NOT hold on this run's own tie-check
> (`set_tie_frac=0.0`) — reopened. See the §B tracker table for the full breakdown.

**Latest run per baseline** (`run_parity.py`, `lib/python/examples/fwdllm/expt_scripts`):

| baseline | run pair | duration | pass | fail | skip |
|---|---|---|---|---|---|
| fluxtune/syn_0 | `run_20260720_014454`/`_034713` (delay-floor 4.0, divisor 0.48, min-init=N=100, agg_goal=10) | ~7200s | 61 | 6 | 18 |
| fwdllm/syn_0 | `run_20260720_014507`/`_034623` (delay-floor 7.0, divisor 1.63, min-init=N=100, agg_goal=10) | ~7200s | 57 | 5 | 22 |
| fwdllm_plus/syn_0 | `run_20260720_014523`/`_034736` (delay-floor 7.0, divisor 1.63, min-init=N=100, agg_goal=10) | ~7200s | 60 | 3 | 21 |

**Key-rung status** (✓ pass · ✗ fail · – skip; catalog: `async_cifar10/PARITY.md` §F):

| baseline | cohort | vclock | thru | commits | terminal | R1 | V1 | V2 | U3 | S2 | conv | conv_loss |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fluxtune | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ |
| fwdllm | ✓ | ✓ | ✓ | ✓ | ✓ | – | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ |
| fwdllm_plus | ✓ | ✓ | ✓ | ✓ | ✓ | – | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ |

**All failing rungs, this run:**
- **fluxtune** (6, was 5 — `agg_step_timing_breakdown` flips fail→pass for the first time this session, but
  TWO new fails open it back up): `cohort_sequence` SET cascade at the same cycle_index=2, same signature as
  every prior session (4th confirmation, not a bug, §F-17). `v1b_iters_moving_avg`/`v2_var_trajectory` still
  fail, downstream of the SET cascade. `drain_wall_budget`'s `drain_tail_s` still fails even though
  `agg_step_timing_breakdown` (its presumed same-root sibling) now passes — the coupling is looser than
  assumed. **NEW**: `trainer_speed_identity`'s `utility` component reopens (23/100 trainers >10% dev, closed
  07-19 as non-reproducing). **NEW→FIXED**: `sim_send_ts` failed on this run (EOT broadcast omitting the stamp
  by design); root-caused and fixed same day, §G — should read clean on the next run.
- **fwdllm** (5, was 3 pre-session): `throughput` now PASSES clean (matched-window gate, first clean pass).
  `per_round_advance`'s matched-window KS still fails, now at n=65 (no longer small-N; ratios still ~1.0, mean
  diff 0.5%) — see §B table. **NEW**: `overhead_residual` crosses its own un-gated tol (12.8% vs 10%) — same
  raw population-mismatch artifact as `per_round_advance`/`throughput`, not independent. **NEW**:
  `v2_var_trajectory` regresses (9.93% vs 2%, was passing 1.4% at 1h) — ROOT-CAUSED same day, same population
  artifact (`var_calc` 0 mismatches on shared calls). `agg_step_timing_breakdown`/`drain_wall_budget` residual
  persists — new cpu/wall signal, confounded by a `var_calc` instrumentation-cost artifact, see §B table.
- **fwdllm_plus** (3, was 4 at 1h scale): `throughput`/`per_round_advance`'s 1h marginal fails are CONFIRMED
  sample-size variance — both pass clean at 2h. `agg_step_timing_breakdown`/`drain_wall_budget` residual
  persists, same cpu/wall signal as fwdllm/fluxtune. `v2_var_trajectory` stays marginal (3.2% vs 2% tol),
  stable across both scales.

See §B for what's actively being worked per baseline; see §G for what's already closed.

---

## §B  Next steps / open issues — per baseline, as of the §A runs above

### Failing-rung tracker — refreshed 2026-07-20 pm-2 (rows combined across baselines when issue + hypothesis match)

Run pairs: FT `run_20260720_014454`/`_034713`, FW `run_20260720_014507`/`_034623`,
FW+ `run_20260720_014523`/`_034736` (all ~7200s). One row per open failing rung (or combined cluster); ≤20
words/cell. Ordered by next-step readiness (trivial fix → needs one more diagnostic → genuinely open).
Cross-cutting principles this feeds into: §F-13 (no un-rooted tolerance loosening), §F-16 (contention claims
need measurement, not guessing).

| Baseline | Rung(s) | Evidence | Read | Hypothesis / root cause | Next step |
|---|---|---|---|---|---|
| FW, FW+, FT | `agg_step_timing_breakdown`; `drain_wall_budget` (`drain_tail_s`) | real cpu/wall 0.90-1.00, sim 0.82-0.89; sim concurrency to 17 vs real 5-6; `_compute_var` is one of the flagged funcs | Sim threads idle more per wall-window despite 0 GPU-pass overlap | Thread-scheduling/GIL pressure from sim's denser thread pool (plausible, unconfirmed) — `_compute_var`'s own measured time was ALSO contaminated by a mis-gated `var_calc` GPU-sync, fixed 07-20 pm-2 (§G) | Re-run the cpu/wall density script on a clean pair (post `var_calc` fix) before drawing further conclusions |
| FW, FW+ | `per_round_advance` (matched KS); `overhead_residual`; `v2_var_trajectory` | matched n=65 KS=0.231; real/sim per-round advances stack in 5 discrete tiers, sim ~0.15s wider + right-skewed per tier; `var_calc` shows 0 mismatches on ALL shared (round,data_id,iter) keys, sim's extra unmatched calls average higher | Shape gap within matched rounds, not a magnitude gap; `v2_var_trajectory` is the SAME sim-outruns-real population effect as `overhead_residual`/`throughput`, not a separate mystery | `overhead_residual`/`v2_var_trajectory` root-caused: un-gated siblings of the already-fixed `throughput`/`per_round_advance` pattern. Residual KS gap: GPU-overrun ruled out (0% both sides) — dispatch-stagger jitter under thread contention is the live hypothesis, ties to row 2 | Extend `matched_window_*` gating to `overhead_residual` + `v2_var_trajectory`; correlate per-round tail outliers against that round's aggregator `step_timing` wall cost |
| FT | `cohort_sequence` (SET); `v1b_iters_moving_avg`; `trainer_speed_identity` (`utility`); `v2_var_trajectory` | Cycles 0-1 match exactly (10/10); cycle 2 forks 4/10; real-vs-real fingerprint pairing (2 independent real runs) shows the SAME divergence pattern | Real=raw FIFO arrival order (jitter); sim=clean sct-order by design (no jitter) — proven inherent, not sim-specific, via real↔real (§G) | **ROOT-CAUSED**: confined to a trainer's first-ever exploring transition, not an RNG/value bug (utility bit-identical once both sides explore) | `first_commit_race_diagnostic` landed on `cohort_sequence_parity` (§G), diagnostic-only. Observe `explained` on the next live pair; promote to gating only after validation |

**Other open items (not a failing rung):**
- FT: `sim_sct_ordered_drain` A/B unblocked — run `fluxtune_n10_smoke_sim_no_sct_drain.yaml` against next pair.
- FT: accuracy drop after reaching 81% — known, deferred by operator (07-15), not yet triaged.

**P3 — infra robustness, not parity-blocking:**
- Dynamic GPU health filtering — `CUDA_DEVICE_ORDER=PCI_BUS_ID` only fixes *which* physical card a given
  ordinal maps to; it does not detect or skip a genuinely broken card. Not attempted, lower priority.

---

**Q: will the system dynamically filter out non-working GPUs and keep going on the healthy ones?**
No, not yet — today's fix only makes CUDA's ordinal numbering match `nvidia-smi`'s (see the crash writeup,
previous turn), so assignment is *deterministic and reasoned-about-able*, but there is still no health check
anywhere in `flame/launch/`. If GPU 0 (nvidia-smi) is still broken next launch, whichever role's ordinal maps
to it (a trainer, under round-robin, or the aggregator, under the fixed "spare Nth GPU" rule) will still hit it
and crash the same way. Proposed design (P3, not yet built): a preflight pass in `runner.py` before spawning —
for each candidate CUDA index 0..`num_gpus`, attempt a cheap op (`torch.zeros(1, device=f'cuda:{i}')` or check
`nvidia-smi --query-gpu=index,memory.used --format=csv` for a card reporting `[Insufficient Permissions]`/`ERR!`);
build a `healthy_indices` list; remap both the trainer round-robin (`spawner.py:315`,
`gpu_id = (trainer_id-1) % self.num_gpus`) and the aggregator's spare-GPU pick (`runner.py:277-282`) to index
into `healthy_indices` instead of raw `range(num_gpus)`. Reduces `num_gpus` effective capacity by 1 per bad
card found (log it loudly) rather than crashing. Not attempted this session — infra work, lower priority than
the actual parity bugs above.

---

### Per-baseline settled state (fails are in the tracker table above, not repeated here)

- **fluxtune** (`run_20260720_014454`/`_034713`, delay-floor 4.0, divisor 0.48, agg_goal=10) — TOP PRIORITY.
  `_distribute_weights_async` stays exempted (`gates_ok=False`, real-only sleep). `tb_prepare_perturbation`
  exemption held, no regression (§G).
- **fwdllm** (`run_20260720_014507`/`_034623`, delay-floor 7.0, divisor 1.63, agg_goal=10) —
  `cohort_sequence`/`step_timing_breakdown`/`throughput` all PASS.
- **fwdllm_plus** (`run_20260720_014523`/`_034736`, delay-floor 7.0, divisor 1.63, agg_goal=10) —
  `step_timing_breakdown`/`per_round_advance`/`throughput` all PASS clean at both 1h and 2h scale.

### Cross-baseline / shared

- **felix (async_cifar10) may have the same round-1 cold-start gap fluxtune had** — `_sim_recv_min` uses the
  same reactive gate shape, no fallback for unseen ends. Felix's own code comment claims it's empirically inert
  (compute ~0.4s wall, `_SIM_RECV_MARGIN_S=0.5s` margin) — plausible but UNVERIFIED, not data-checked. See §G
  07-16 / `unknown_stuck` gate for the fix pattern if felix's own data later shows otherwise. Not implemented —
  out of this session's scope (`async_cifar10/PARITY.md` owns felix).
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

> **RULE: closed = here, in ≤30 words, immediately.** The instant a rung flips fail→pass or a hypothesis is
> confirmed/refuted, write ONE terse line below (mechanism + outcome, no narrative) and delete it from §A/§B in
> the same edit. Full reasoning lives in the commit/code comment, not this doc.

- **fluxtune `sim_send_ts` FIXED** (07-20 pm-2) — root-caused to the EOT/shutdown broadcast: `top_aggregator.py`'s
  `inform_end_of_training` only stamped `SIM_SEND_TS` when `trainer_event_dict` was set (avail-tracking on);
  99/100 trainers' null event was their final `task_recv`, not a mid-run gap. Now stamped unconditionally on
  `simulated`. `test_eot_avail_catchup.py` updated; 591/591 `tests/mode` pass.
- **`agg_step_timing_breakdown`'s `_compute_var` residual had a self-inflicted confound** (07-20 pm-2) —
  `var_calc` DEBUG telemetry (07-19 pm) was mis-gated at `logging.INFO` instead of `DEBUG`
  (`FedSgdAggregator.py:226`), so its `.norm().item()` GPU-sync ran on every `_compute_var` call in every run
  since, inflating exactly the function flagged as the residual's worst offender. Fixed; 282/282 fwdllm tests pass.
- **fwdllm/fwdllm_plus's `v2_var_trajectory` ROOT-CAUSED** (07-20 pm-2, via the still-banked `var_calc`
  telemetry) — 0 mismatches on every shared (round,data_id,iteration) key real vs sim; the fail is entirely
  sim's extra unmatched (further-into-training) calls averaging higher, pulling the population mean up. Same
  mechanism as `overhead_residual`/`throughput` (§G 07-19 pm), not a separate bug. fluxtune's stays open — 1583
  of 1585 shared keys genuinely mismatch (cohort-fork downstream, not population length).
- **fluxtune's cohort-fork mechanism ROOT-CAUSED precisely** (07-20 pm-3, supersedes the pm-2 "does NOT hold"
  entry) — real commits in raw FIFO arrival order (`channel.py:recv_fifo`, network/OS jitter, uncorrected); sim
  commits in clean sct-order by design (`asyncfl/top_aggregator.py:_sim_recv_min`, no jitter term, intentional).
  Proven via `eligible_fingerprint`/`decision_fingerprint` pairing (already-existing `emit_selection`
  instrumentation) that TWO INDEPENDENT REAL runs diverge the same way at the same point — inherent
  independent-process noise confined to a trainer's first-ever exploring transition, not a sim defect, not an
  RNG desync (values are bit-identical once both sides explore a trainer). `_cohort_set_tie_ok`'s boundary-tie
  concept simply doesn't cover this mechanism (a different kind of "tie") — see the new diagnostic below.
- **New DIAGNOSTIC-ONLY `first_commit_race_diagnostic` field on `cohort_sequence_parity`** (07-20 pm-3) — two
  gates, both required: (1) structural — a differing trainer's exploring transition landed within `tie_window_s`
  of the cohort boundary; (2) margin — the utility gap vs. the mode's own chosen-cohort cutoff is within that
  run's own observed rank-gap noise floor (not a fixed constant). Does NOT gate `ok`/`set_ok` yet — reports
  `explained: bool` for review. 4 new unit tests (near-tie explained; large-margin NOT explained; stale-explore
  NOT explained; absent when cohorts match) + 175/175 `test_parity_checks.py`, 595/595 `tests/mode` pass. Not yet
  validated against a live run — tune/promote to gating only after the next fluxtune pair.

- **fluxtune `pacer()` once-per-round fix VALIDATED live** (07-20 pm, fresh 7200s pair) — `preferred_duration`
  passes (frac_diff 0.152 vs 0.2 tol). Confirms the fix holds at scale; felix's own parity still needs a
  separate re-check (shared class, out of this session's scope).
- **fwdllm_plus's 1h `throughput`/`per_round_advance` marginal fails CONFIRMED sample-size variance** (07-20 pm)
  — both pass clean on the fresh 7200s pair (4.9%/matched 0.5%), not a regression as hypothesized 07-19.
- **fwdllm's `throughput` PASSES clean for the first time** (07-20 pm, 7200s pair) — matched-window gate holds
  at 2h scale (4.9%, matched 0.5%). `per_round_advance`'s KS sub-check remains open (§B item 1).
- **`tb_accumulate_grads` borderline fail did not reproduce on a 2nd independent run** (07-20 pm) — confirmed
  noise, closing the 07-19 flag.
- **`agg_step_timing_breakdown`'s `time.thread_time()` cpu/wall diagnostic exercised on a live run for the first
  time** (07-20 pm) — real cpu/wall ≈0.90-1.00, sim ≈0.82-0.89 for every residual function, consistent across
  all 3 baselines (`analyze_agg_step_timing_density.py`). Sim threads idle/blocked more of their wall window
  than real despite zero GPU-pass overlap — reopens contention via thread-scheduling pressure, not the already-
  refuted mechanism. Residual itself still open (§B item 2).

- **fwdllm/fwdllm_plus's `throughput` chronic fail FIXED** (07-20 am) — `checks.py`'s `throughput_parity`/
  `per_round_advance_parity` now gate `ok` on `matched_window_*` (same population, not full-vs-truncated) when
  `real_coord is not None` (sync baselines only — the intrinsic clock that makes the comparison meaningful).
  `throughput` flips fail→pass on the 1h fwdllm pair (matched 3.4% vs raw 8.3%), fwdllm_plus clean both scales.
  fluxtune untouched (async, `real_coord` is None, stays on the raw gate). `per_round_advance` flips clean for
  fwdllm_plus; fwdllm's matched KS still fails (small-N artifact, §B item 1) — not force-closed.
- **`agg_step_timing_breakdown`'s cpu/wall diagnostic was measuring the wrong thing** (07-20 am) — `timer_
  decorator` used `time.process_time()` (process-wide), silently summing the aggregator's own backgrounded
  eval-thread CPU into every call's measurement (confirmed via eval-window overlap, ~7-8x on real AND sim
  symmetrically). Switched to `time.thread_time()` (thread-local). Residual itself still open — needs a fresh
  run to get a real signal.
- **fluxtune's `preferred_duration` gap ROOT-CAUSED + FIXED** (07-20 am) — `async_oort.py`'s `pacer()` fired
  once per `select()` call, not once per round: reference Oort's `pacer()`/`getTopK()` are the same call,
  which its SYNCHRONOUS loop only invokes once/round by construction (no explicit guard needed there); flame's
  own sync `oort.py` already added an explicit `_last_selection_round` guard for exactly this reason, but the
  async port never did, and async's `select()` genuinely fires many times per round (once per freed trainer
  slot). Confirmed on BOTH real and sim 07-19 telemetry: an 18-call same-round burst ratchets `round_threshold`
  10→100 within 3-7 real seconds, permanently disabling the speed penalty — present on both sides, just at
  different wall-time offsets (item 1's timing-mismatch root), which is what inflated `preferred_duration`'s
  short-run gap. **Fixed**: `pacer(current_round)` now takes an explicit round (mirrors sync's signature) and
  the call site gates on a new `_last_pacer_round` (fires only on an actual round change), same pattern as
  sync's guard. 202/202 selector + 451/451 fwdllm/mode + 118/118 async_cifar10-parity tests pass (shared class
  with felix). Needs live validation next run (§B item 5); felix parity should be re-checked too. **Forward
  note**: fwdllm doesn't use this selector (`random`, no pacer) — if one ever is added, key the guard on each
  AGGREGATION event, which for fwdllm is `iteration_per_data_id` (fwdllm aggregates every iteration attempt,
  not just the final commit) — NOT `data_id`/round, which is too coarse and would repeat this exact bug.
- **`terminal_state`/`total_commits` newly failing at 15min scale is NOT an independent bug** (07-20 am) —
  same signal as `throughput`'s round-count-at-matched-V (checks.py's own docstring: rel_diffs match to 3
  decimals by construction). Confirmed: identical real/sim round-count gap (off by exactly 1 round) passes
  cleanly at 1h scale (2.9%, n=34) and fails only from 1-round discretization at 15min scale (n=7-8) — pure
  small-N boundary artifact riding on an already-clean per-round match, not a new mechanism. No fix needed.
- **fwdllm/fwdllm_plus's `throughput`/`per_round_advance` gap VERDICT: sim captures every stage correctly**
  (07-20 am) — round-by-round decomposition on real's `intrinsic_span_s` clock shows per-round advance ratio
  0.998-1.021 for every matched round regardless of iteration count; the aggregate 8-15% gap is entirely a
  population-length comparator artifact (sim's round count outruns real's wall-capped count) riding on a
  genuine FL dynamic (mean iters/data_id rises 1.606→1.923 later in training). Not a modeling gap. Gating-fix
  decision pending (§B item 1).
- **fwdllm/fwdllm_plus's `training_delay_floor_s`=11.0 predated the harness-overhead-removal fix, unlike
  fluxtune's already-re-derived 4.0** (07-19 pm, FWDLLM_DESIGN.md §O) — re-derived on a fresh compute read
  (max 2.72s/3.18s) with the same formula: `1.3 × 1.63 × 3.18 = 6.74s` → **7.0**. Landed in `run_sequential.sh`'s
  `BASELINE_DELAY_DEFAULTS`. **VALIDATED 07-20 am**: fresh ~900s triple shows 0 `[TIMING_OVERRUN]` across all 3
  baselines at the tighter floor.
- **`suppress_redundant_weights` reconfirmed on a 2nd, independent real pair** (07-20 am, `run_20260719_2326*`)
  — `audit_weight_redundancy.py` still 0% redundant weight-sends on both fwdllm and fwdllm_plus. Invariant holds.
- **fwdllm's `throughput`/`per_round_advance` gap had no mechanism, just a fwdllm_plus cross-baseline
  comparison that never explained a within-baseline real-vs-sim gap** (07-19 pm) — ROOT-CAUSED: per-trainer
  modeled delay is statistically identical real/sim (11.41s both); agg compute is never on the vclock (§F-1,
  confirmed via `phase_vclock_bottlenecks`, zero bottleneck phases). The 8.3% is sim's round population running
  25 rounds past real's own wall-capped population, and those extra rounds averaging 15% slower. Matched-window
  diagnostic fields (`matched_window_*`, DIAG-only) added to `throughput_parity`/`per_round_advance_parity` —
  confirm 3.4%/1.7% gap once the population mismatch is removed (was 8.3%/7.6%).
- **No check verified `agg_step_timing` telemetry actually covers 100% of a cycle's wall time** (07-19 pm) —
  summed `_aggregate_grads_sync` + `_distribute_weights_sync` against the raw `agg_round`-to-`agg_round` wall
  gap: real 100.1%, sim 99.8% accounted (fwdllm). Confirms the `agg_step_timing_breakdown` residual is real, not
  a missing-instrumentation gap.
- **`sync_collect_and_accumulate_grads`/`_aggregate_grads_sync` were the reported `worst_func` in
  `agg_step_timing_breakdown` at 23-76x real/sim ratio** (07-19 pm) — root-caused: real blocks in
  `channel.recv_fifo` under the `num_min_req=1` clamp, sim's branch is a non-blocking vclock computation; a
  real-transport wait, not compute. Exempted in `_AGG_STEP_TIMING_REAL_ONLY_FUNCS`, same class as
  `_distribute_weights_async`.
- **`_step_timing_compare` had no point-mass guard** (07-19 pm) — sub-ms functions (e.g. fwdllm's `_send_grads`,
  0.8ms real vs 0.9ms sim) failed on relative-% noise once p99 crossed the degenerate-skip threshold. Added a
  near-zero-mean + near-zero-absolute-diff guard (mirrors `trainer_phase_split`'s existing one); flips fwdllm's
  `step_timing_breakdown` fail→pass.
- **`suppress_redundant_weights` was never validated against a live pair** (07-19 pm) — `audit_weight_
  redundancy.py` on the fresh real legs confirms 0% redundant weight-sends for both fwdllm and fwdllm_plus (was
  ~88-90% before the fix). Confirmed correct.
- **`suppress_redundant_weights` was still an opt-in flag after validation** (07-19 pm, operator follow-up) —
  removed; `_should_send_full_weights` now unconditionally suppresses an intra-databin re-send. Config key
  deleted from all 6 yamls + `baselines.yaml`. Added `_warn_if_redundant_weights_resend` at both send sites as a
  regression tripwire (WARNs if a future call site bypasses the decision function).
- **`agg_step_timing_breakdown`'s residual had no way to separate "waiting on contention" from "genuinely more
  compute"** (07-19 pm) — `timer_decorator` now also captures `cpu_duration_s` (`time.process_time()`) alongside
  wall `duration_s`; `analyze_agg_step_timing_density.py` prints cpu/wall per function. Also checked: sim isn't
  batching more items/call (`contributing_trainers` cohort size is exactly 10 every cycle, both modes) — ruling
  out volume as the cause. Verdict needs the next pair's data (this run predates the field).
- **fluxtune's iters-per-data_id mismatch (67-68% across 2 runs) had no known root cause** (07-19 pm) —
  ROOT-CAUSED: traced data_id=0 cycle-by-cycle, cohort AND `var` are identical real/sim through cycle 1, both
  diverge together at cycle 2 (the known SET tie), pushing the threshold-crossing iteration 4→5. Later data_ids
  diverge from cycle 0 because the shared seeded RNG stream never resynchronizes. Same root as `cohort_sequence`
  SET cascade, not an independent bug — merged, no separate fix needed.
- **`_distribute_weights_sync` never logged `redundant_weights_suppressed_total`** (07-19 pm) — rule #18 trap:
  the async path already logged it, sync silently tracked the counter with no visibility. Added a matching
  `[Distribute] Done...` summary log.
- **`agg_step_timing_breakdown`'s contention-burstiness hypothesis** (07-19 pm) — REFUTED for all 3 baselines:
  fwdllm/fwdllm_plus sim show zero measured trainer-GPU-pass overlap yet the gap persists; fluxtune's own
  concurrency-duration correlation runs backwards. Not contention; cause still open (§B P1 item 2).
- **fwdllm's redundant-fetch-cost-asymmetry hypothesis for the `throughput`/`per_round_advance` gap** (07-19 pm)
  — REFUTED: 0% redundancy confirmed both sides, yet fwdllm's `_fetch_weights` still costs ~2x fwdllm_plus's
  per call. Gap CONFIRMED STABLE (11%→8.3% across 2 runs) but cause reopened (§B P1 item 1).
- **fluxtune's iters-per-data_id mismatch** (07-19 pm) — RECONFIRMED stable on an independent run (68% vs
  prior 67%), still the top candidate for the `cohort_sequence` cascade; root mechanism still open.
- **fwdllm/fwdllm_plus/fluxtune parity yamls never enabled `suppress_redundant_weights`** (07-19) — flag exists,
  is unit-tested, and was already `true` in `baselines.yaml`; the `expt_scripts/` yamls just never inherited it.
  Added to all 6. Root fix (eliminates the resend), not an instrumentation workaround.
- **`_process_aggregation_goal_met`'s buffered-replay loop had no wall cost of its own** (07-19) — extracted to
  `_replay_buffered_cohort_contribs`, `@timer_decorator`-wrapped; shows up in `agg_step_timing_breakdown` and
  sub-phases `drain_tail_s`. Tests contention-burst hypothesis (P1).
- **`agg_step_timing_breakdown`'s contention hypothesis had no density correlation** (07-19) — new
  `analyze_agg_step_timing_density.py` (same overlap method as `analyze_tb_prepare_perturbation.py`) buckets
  `_compute_var`/`_apply_weighted_update`/etc by concurrent-trainer GPU density from existing telemetry.
- **`v2_var_trajectory` divergence had no way to localize input vs reduction** (07-19) — new DEBUG-gated
  `var_calc` telemetry (`build_var_calc`) logs per-tensor input grad norms + output var per `_compute_var` call.
- **`drain_tail_s` was one opaque number** (07-19) — split into `drain_tail_canonicalize_s`/`drain_tail_replay_s`/
  `drain_tail_residual_s`, diagnostic-only (not yet gated).
- **No repeatable, whole-run, cross-baseline check for per-data_id iteration-count mismatches** (07-19) — new
  `analyze_iters_per_data_id.py`. On the fresh triple: fwdllm/fwdllm_plus are EXACT on every shared data_id (0
  mismatches, P0-1 validated); fluxtune mismatches on 74/111 (67%) shared data_ids — localizes the previously
  "data_id=0 only" finding to a run-wide, async-only phenomenon, not an isolated early divergence.
- **P0-1/P0-2 validated ≥3600s** (07-19) — `cohort_sequence` exact 1.0 (fwdllm/fwdllm_plus); fwdllm_plus's rate
  rungs all pass. fwdllm's own residual reopened, unexplained (§B).
- **`tb_prepare_perturbation` root-caused + exempted** (07-19) — branch 100% `cached` both sides (not
  branch-rate); same GPU-density artifact as `eval_model`. Added to `checks.py`'s exemption set; VALIDATED
  (`gates_ok=False` confirmed against the banked pair). Unmasked a smaller, unexamined `tb_accumulate_grads`
  borderline fail (KS 0.256 vs 0.25 tol, real 2.9ms vs sim 3.1ms) — new, not yet triaged, see §B.
- **fluxtune `trainer_speed_identity` utility outlier** (07-19) — didn't reproduce on the fresh 2h pair. Closed.
- **`tb_prepare_perturbation` had no branch/density telemetry** (07-18o) — added `extra` param to `_stage_timer`;
  density derived from existing GPU-pass windows; `analyze_tb_prepare_perturbation.py`.
- **Sim's one-shot sync collect couldn't match real's incremental `num_min_req=1`** (07-18o) — new fwdllm-scoped
  `_sim_sync_recv_incremental` (persistent SimReorderBuffer) replaces one-shot drain; clamp gate removed.
- **Real's `num_min_req=1` clamp isn't re-selection** (07-18o) — it re-sends full weights to still-computing
  trainers (88% of fwdllm real fetches were wasted duplicates); fluxtune unexposed (async re-pick guard).
- **P0-1 grad-merge fix validated** (07-18n) — `cohort_sequence` `var_match_frac` 0.25/0.5→exact 1.0 (fwdllm/
  fwdllm_plus); fluxtune's separate SET-cascade fail unaffected, as expected.
- **07-18 fluxtune relaunch omitted `--delays`, ran D=0, collapsed sim throughput 22x** (07-18m) — added
  per-baseline `BASELINE_DELAY_DEFAULTS` to `run_sequential.sh`; explicit flags still override.
- **`self.grad`'s per-cycle merge summed in raw arrival order, not canonical** (07-18l) — non-associative float
  add order-dependent; buffer contributions, replay in canonical (D, trainer_id) order.
- **Aggregator crashed hard on CUDA-unavailable machines** (07-18k) — CUDA-only diagnostics ran even after
  resolving `device='cpu'`; gated behind `device.type == 'cuda'`.
- **fluxtune `agg_step_timing_breakdown` gap** (07-18g/h/i/j) — CPU/memory contention from sim's dense trainers;
  fixed via NUMA-aware placement + dead-deepcopy removal + 50% tolerance widen. Validated PASS at 2h (07-18);
  **regressed 07-19, cause unclear, see §B**.
- **SET tie-window blind to mid-run dispatch timing** (07-17d) — late-fast vs early-slow trainers can tie;
  `_cohort_set_tie_ok` now checks actual commit proximity to the cohort boundary.
- **SET rung compared uncapped over the full run** (07-17d) — a legitimate tie cascades unboundedly once
  triggered; capped to the same `max_bin` window CADENCE/VAR/ORDER already use.
- **S2 (`participation_parity`) windowed on `round`, degenerating to n=1 for fwdllm** (07-17d) — now windows on
  cycle position (`n_rounds_matched` 1→13); validated `speed_class_tvd=0.031` (tol 0.15).
- **fluxtune's remaining `cohort_sequence` divergence** (07-17d) — confirmed legitimate stochastic noise
  (seeded draw over a timing-dependent candidate set), not a bug. Closed.
- **`minInitialTrainers=c` (not N) reopened the join-order race, all 3 baselines** (07-17) — sim's cadence
  outruns real's; fixed `minInitialTrainers=N` in all parity yamls.
- **Recv-side resample fallback deadlocked ALL dispatch at minInitialTrainers=N** (07-17b) — fallback
  contradicted its own docstring; removed entirely. Validated: 0 stalls, both 6-min and 5400s pairs.
- **Round-1 cold-start: `_sim_recv_min_grad`'s gate blind on first contact** (07-16) — added `unknown_stuck`
  wall-clock-cap hold. Validated: cycle 0 10/10 (was 8/10).
- **SET/ORDER divergence had no tie tolerance** (07-16) — now granted a TIE when every differing trainer's
  expected delay is within `tie_window_s=1.0`.
- **fluxtune `agg_goal=3` too tight for `c=30` pool** (07-16) — coin-flip admission; raised to 10. Validated:
  `throughput`/`total_commits`/`terminal_state` PASS, `convergence_loss` 0.172→0.007.
- **Degenerate-noise skip was max-gated not p99-gated** (07-16) — one GC outlier defeated it; switched to p99.
- **fwdllm/fwdllm_plus yamls renamed** `_n10_smoke*`→`_n100_smoke*` (07-16) — filename only, `num_trainers` was
  already 100.
- **`minInitialTrainers` now defaults to N** (07-16) — waits for ALL trainers before first selection, removes
  the pool-size race (real fired at 98, sim at 99).
- **AVL_TRAIN not stamped at registration** (07-16) — `Channel.add` now stamps it; kills startup UNKNOWN
  transient in `avail_composition`.
- **`snapshot.yaml` dropped `hyperparameters.seed`** (07-16) — now recorded in the aggregator block.
- **`eval_model` false-failed `agg_step_timing_breakdown`** (07-16) — daemon-backgrounded, off-vclock; exempted
  (pure GPU-density artifact, sim trainers never sleep).
- **All selectors leaked trainer JOIN order into the seeded draw** (07-16) — raw `ends.keys()` before
  `_rng.choice`; canonicalized to `sorted(ends.keys())`, default seed 1234 everywhere.
- **`v1b_iters_moving_avg` rung added** (07-16) — catches trajectory drift that v1's pooled KS+mean cancels out.
- **Trainer compute re-measured post overhead-removal** (07-16) — genuine JVP mean 0.47s (was 3.63s, ~87%
  harness overhead) — drove floor re-derivation 7.0→4.0s.
- **Trainer wall-time attribution read, no anomaly** (07-16) — n100 `_train_one_batch` 354ms real ≈ 349ms sim.
- **Eval wrongly blamed as the residual** (07-16) — eval is daemon-backgrounded both modes, doesn't slow
  `_process`; residual is dispatch order, not eval. Correction of a same-day wrong call.
- **Aggregator wrongly blamed as 350ms-slow** (07-16) — actually queue-bound (serial commits, `c=30` cap), not
  MQTT transit.
- **Perturbations validated deterministic across modes** (07-16) — utility matches to 5 d.p. when aligned;
  divergence is dispatch order, not compute.
- **Trainer `seed` telemetry logged `None`** (07-16) — seed lived only in aggregator config; added to trainer
  `config_overrides` in all 6 base yamls.
- **Aggregator did the same full-model deepcopy 3x/commit** (07-16) — copy-paste bug; collapsed to 1; gated 8
  eager `_calculate_hash` debug calls.
- **`--min-initial-frac` startup-barrier lever added** (07-16) — opt-in A/B for the dispatch-order root,
  unchanged when unset.
- **fluxtune `preferred_duration`+`avail_composition` PASS post-fix** (07-16) — seed/pacer fix cleared pref;
  AVL-at-registration drove avail UNKNOWN→0.
- **Sim ran UNSEEDED while real had `seed=1234`** (07-15) — sim yamls omitted the key; added to all 3 +
  default `None`→`1234`.
- **`_handle_recv_state` leaked dispatch order via PYTHONHASHSEED** (07-15) — ported the `dict.fromkeys` fix to
  async_oort/fedbuff/async_random.
- **Trainer batch interior emitted zero telemetry** (07-15) — `timer_decorator` keyed off the wrong arg; added
  `_stage_timer` + 10 `tb_*` phases.
- **`agg_step_timing_breakdown` false positives on tight distributions** (07-15) — added degenerate-skip, 5%
  mean escape, exempted `_distribute_weights_async`.
- **`aggregation_plots` dead on a NameError** (07-15) — missing collection loop; restored, plots render again.
- **TIMING_OVERRUN** (07-15) — §O's margin used fast-class MEAN not FLOOR; fixed `training_delay_floor_s`.
  Validated 0 overruns.
- **fluxtune accuracy floor** (07-14/15) — cross-refs `fluxtune_contributions.md` §8's tracked collapse; not a
  parity bug, both legs match.
- **`r1_inflight_overlap` flagged FedBuff's legit stale-accept redispatch** (07-15) — rescoped per
  `version_key`; fixed.
- **`_sim_gate_compute_cap_s`'s blind 10.0 too thin** (07-15) — derived 16.0 for fluxtune.
- **`select_random` order leaked via PYTHONHASHSEED** (07-14) — `set()`→`dict.fromkeys()`. Validated: fwdllm
  `cohort_sequence` 100% match.
- **fluxtune `preferred_duration`** (07-14) — oort pacer was a one-branch port; faithful both-branch port
  closed 50.7pp→9.3pp gap.
- **Parity-CLI progress-axis picked per-side independently** (07-14) — glob collided fwdllm/fwdllm_plus; prefer
  `data_id`, anchor glob on `_{tag}_n<N>_`.
- **Aggregator `step_timing` unparsed by any check** (07-14) — added loader capture + `agg_step_timing_
  breakdown` rung.
- **fwdllm had no seeded yaml** (07-14) — seed plumbing was already correct, just unexercised; added 6 pairs.
- **`recv_fifo` hot path logged 425k lines/run at INFO** (07-14) — downgraded 9 mechanical lines to DEBUG.
- **Server-momentum (S1) landed flag-gated** (07-14) — default 0.0 no-op + A/B yamls; not run, deferred to
  `fluxtune_contributions.md` §8.2.
- **Reactive gate re-checked stale state after a blocking call** (07-13) — `_sim_gate_is_safe` now checks
  first; `sim_rate` 0.97→1.82×.
- **Carried-surplus commits misclassified as round1** (07-13) — classifier wasn't re-keyed to `data_id`;
  ingest-time carry-over stamp, separate bucket.
- **`eval_model()` blocked dispatch (sync stall)** (07-13) — backgrounded on a daemon thread.
- **fwdllm_plus livelock** (07-13) — `RandomSelector` freed only k=5 of c=10; removed the stale `k` knob.
- **fluxtune commit-path stall** (07-13) — phantom `_sim_inflight_expected` entry; `sim_compute_truthful_gate`
  skips stale dispatches.
- **fluxtune cohort-SET divergence** (07-13) — real released a busy trainer's guard on RETURN not commit;
  hold-to-commit fix.
- **`version_key` identity was bare-int in some places, 3-tuple in others** (07-13) — one shared 2-tuple
  property everywhere.
- **Additive send+gpu+D delay gave nondeterministic arrival order** (07-13) — real sleeps `max(0,D-gpu)`, sim
  never sleeps D.
- **Release-on-RETURN undercounted in-flight state 3x** (07-13) — hold slot until commit, both sync+async.
- **Dropping surplus grads at agg-goal boundary wasted ~7/cycle** (07-13) — carry surplus + hold busy trainers.
- **Async cycles summed as sequential** (76-86% spurious diff) — fall back to raw wall for async.
- **Clock-rate rungs used full wall** (localhost-only latency) — switched to `intrinsic_span_s`.
- **`cohort_sequence_parity` conflated SET/CADENCE/VAR/ORDER** — one cap tripped on real GPU fp16 jitter; SET
  now hard/uncapped, rest capped to bin 1.
- **"GPU under-provisioned at n=10"** — refuted; spawn table is balanced round-robin, 8 GPUs, 1 core/trainer.
