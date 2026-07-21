# FwdLLM — Real↔Sim Parity

**Scope: real↔sim parity only**, for **fluxtune / fwdllm / fwdllm_plus** at 100% availability (syn_0, Phase 1),
then unavailability (Phase 2), then beyond syn_0 (Phase 3). Non-parity content (structural deltas, baseline
matrix, roadmap, JVP perf, sim barrier redesign, delay-factor calibration) lives in
[FWDLLM_DESIGN.md](FWDLLM_DESIGN.md). Shared parity methodology (ladder, roles/tiers/gating, run-length budget)
and fwdllm's rung catalog (§F) live in [async_cifar10/PARITY.md](../async_cifar10/PARITY.md) — read it first if
new to this track.

> ## PREAMBLE — maintaining this doc
> **Parity findings/fixes only** — design decisions, roadmap items, and calibration derivations belong in
> FWDLLM_DESIGN.md.
> **Living doc, not a changelog** — §A/§B describe the state *right now*, rewritten in place, never stacked as
> dated "UPDATE" blocks. Full history is `git log` on this file + the parity JSONs; §G keeps only the recent
> handful still load-bearing for current work.
> **§A**: scoreboard only (pass/fail/skip + key-rung table) — no prose essays. Refresh whenever `run_parity.py`
> runs a >3600s pair, for every baseline (carry stale numbers forward, tagged STALE).
> **§B**: open issues, per baseline, short and current-state only — updated in place, not appended to. An issue
> lives in exactly one place: open (§B) xor closed (§G, one line) — never both, never a stale copy left behind.
> **§G**: closed items, one line each (problem → fix, ≤30 words). The moment a rung flips or a hypothesis
> resolves, write the line and delete the §B entry in the same edit.
> **Every fix**: ground claims in telemetry already on disk before instrumenting or running; fix root causes,
> not symptoms; never launch an experiment directly (print the command for the operator to run); always use the
> `dg_flame` conda env for python/pytest/analyze_run.py; ship new telemetry with its plot + pytest in the same
> change (a field with no reader in `analyze_run.py` is dark data).

**Prerequisites:** [async_cifar10/PARITY.md](../async_cifar10/PARITY.md) (methodology + rung catalog §F),
[UNAVAILABILITY_DESIGN.md](../async_cifar10/UNAVAILABILITY_DESIGN.md) (availability substrate),
[FWDLLM_DESIGN.md](FWDLLM_DESIGN.md) (build plan/roadmap/calibration),
[fluxtune_contributions.md](fluxtune_contributions.md) §8 (training-stability/convergence ledger — check before
opening a new stability investigation here).

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

## §A  Score — refreshed 2026-07-21

fwdllm/fwdllm_plus are settled: only `drain_wall_budget`/`agg_step_timing_breakdown` fail, unchanged across
sessions. **fluxtune regressed sharply this session — 6→19 fails** — root-caused to an asymmetry in the
`_release_end_on_return` buffered-slot-release fix (9344422a): it freed real's channel slot the instant a
contribution buffered, but only sim had a selector-level guard to also keep that trainer un-re-pickable until
commit. Real had none, so it wasted ~9% of its dispatch capacity on duplicate redispatches
(`grep -c "Duplicate contribution from"`: real 1058, sim 0), widening the cohort-composition gap that cascades
into throughput/commits/utility/convergence. **Fix landed same session** (§B) — binds real's selector to the
same `_per_agg_trainer_list` already used for the dedup guard, no new data structure. 617/617 `tests/mode`
pass. A follow-up 900s smoke pair (`run_20260721_100808`/`_103232`) already shows 19→12 fails with this fix
alone. Remaining `cohort_sequence` divergence root-caused + a second fix landed same session (§B) —
**not yet validated live.**

**Latest run per baseline** (`run_parity.py`, `lib/python/examples/fwdllm/expt_scripts`):

| baseline | run pair | duration | pass | fail | skip |
|---|---|---|---|---|---|
| fluxtune/syn_0 | `run_20260721_013455`/`_033709` (agg_goal=10) | ~7200s | 47 | 19 | 16 |
| fwdllm/syn_0 | `run_20260721_013501`/`_033648` (agg_goal=10) | ~7200s | 60 | 2 | 22 |
| fwdllm_plus/syn_0 | `run_20260721_035015`/`_055227` (agg_goal=10) | ~7200s | 61 | 2 | 21 |

**Key-rung status** (✓ pass · ✗ fail · – skip; catalog: `async_cifar10/PARITY.md` §F):

| baseline | cohort | vclock | thru | commits | terminal | R1 | V1 | V2 | U3 | S2 | conv | conv_loss |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fluxtune | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ |
| fwdllm | ✓ | ✓ | ✓ | ✓ | ✓ | – | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| fwdllm_plus | ✓ | ✓ | ✓ | ✓ | ✓ | – | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

fluxtune's 19 fails are all downstream of the one root cause above (§B): `cohort_sequence`,
`trainer_speed_identity`, `overhead_residual`, `per_round_advance`, `throughput`, `selection_detail`,
`phase_weights_to_gpu`, `staleness`, `v1_iter_per_data_id`, `v1b_iters_moving_avg`, `v2_var_trajectory`,
`v5_variance_pass_ratio`, `g2_grad_pool_size`, `r1_inflight_overlap`, `utility`, `terminal_state`,
`total_commits`, `convergence`, `convergence_loss`.

---

## §B  Next steps / open issues — per baseline

> **RULE: every tracker cell ≤20 words.** State the claim/number, cut qualifiers. If it needs more, it's not
> tracker material — shorten it or point at the code comment/commit.

| Baseline | Rung(s) | Hypothesis / root cause | Next step |
|---|---|---|---|
| FW, FW+ | `agg_step_timing_breakdown` (`drain_wall_budget` PASSES) | cpu/wall 0.84-0.88 both sides (busy, not blocked) — raw CPU-seconds gap, not scheduling. `_replay_buffered_cohort_contribs` worst (sim 5.6x real `cpu_duration_s`) | Needs a ≥2h pair to separate small-N noise from real widening. Isolate `_replay_buffered_cohort_contribs` |

### fluxtune — TOP PRIORITY (`run_20260721_013455`/`_033709`, agg_goal=10)

**Root cause, confirmed via code trace + telemetry.** `_release_end_on_return`'s `buffered=True` fast-path
(9344422a) frees a trainer's channel slot the instant its grad is buffered, on both real and sim. But
re-selection is gated by a SECOND set too: `async_oort.py`'s `_agg_pending_commit_ref`. That ref was bound
only in sim (`_sim_hold_busy_slots` → `_sim_pending_commit`, gated `if not self.simulated: return`) — empty in
real by construction. So sim's cadence was unaffected by the fix (already correct); real genuinely re-dispatched
a trainer whose prior contribution was still buffered-but-uncommitted, wasting a full GPU/JVP pass that the
dedup guard then silently discarded — 1058 wasted dispatches this run (0 sim), ~9.1% of all real grad-receives,
matching `r1_inflight_overlap` (real 3.73% vs sim 0.0%, tol 2%). Cohort SIZE integrity stayed intact (always
exactly 10 unique contributors) — this is wasted capacity, not corruption. The waste widened the pre-existing
`cohort_sequence` composition drift (below), and since gradient variance is a property of WHICH 10 trainers
land in a cohort, sim needed far more attempts per commit than real (`v1_iter_per_data_id` 7.0 vs 12.6,
matching the `agg_round` cycle-count ratio 1052 vs 1894 exactly) — cascading into every other failing rung.

**[LANDED, unvalidated live] Fix.** `_per_agg_trainer_list` was already the unconditional, real+sim-shared
"buffered but not yet committed" list (used for the dedup guard). Bound
`channel._selector._agg_pending_commit_ref = self._per_agg_trainer_list` for real only (`if not
self.simulated`, right after the append) — no new/parallel structure, sim's own `_sim_hold_busy_slots` binding
untouched. Two supporting changes since the selector holds a live (never-rebound) reference: `_per_agg_trainer_
list`'s two reset points switched from rebinding to in-place `.clear()`/slice-assign; new `_OrderedContributor
List(list)` adds `.discard()` so it duck-types against the set-based binding sim uses. 8 new tests
(`test_fwdllm_real_pending_commit.py`), 617/617 `tests/mode` pass. **Needs the next live pair** to confirm the
1058 wasted duplicates → ~0 and the 19-fail regression closes.

**`cohort_sequence` background (the mechanism the fix above targets).** Real released a trainer's slot/guard
only at full-cohort commit (batched — an "atomic burst"); sim's `_sim_hold_busy_slots` reconciles against a
virtual in-flight set and stays flat. Real and sim were running genuinely different algorithms for "who's free,"
not just experiencing timing jitter. **Sim's flat concurrency is correct** (true async fedbuff decouples
dispatch from the batch boundary) — real's batched release was the bug, since P0-1 already buffers each
contribution safely on receipt (nothing is lost by releasing early). `_set_tie` was separately audited and
confirmed correctly calibrated (not miscalibrated) — no fix needed there.

**`cohort_sequence`'s residual divergence, root-caused via SELECT_TRACE forensics (07-21).** On the 900s
smoke pair above, real/sim's `select_random` candidate pools first diverge by exactly one trainer (an
ordinary async timing difference), then the gap grows monotonically all round (1→15 trainers by call #41) —
never self-healing. Two stacked causes: (1) `select_random`/`sample_by_speed` drew via `self._pyrng.sample()`/
`self._rng.choice()`, a single persistent RNG stream whose consumption is population-size-dependent — one
incidental pool-size difference desyncs every later draw, permanently, instead of affecting only that one pick;
(2) sim's `_sim_pending_commit` guard was only reconciled by `_sim_hold_busy_slots` on OTHER commit/boundary
events, not on a trainer's own receipt, so it stayed wrongly eligible for many calls after buffering its own
grad (real's guard is a live reference, always current — no such gap). **[LANDED, unvalidated live] Fix.**
(1) new `_keyed_topk` (`async_oort.py`): each candidate's rank key = a fresh `Random(f"{seed}|{salt}|
{agg_version_key}|{id}")` draw — depends only on its own identity, never on pool membership/size/call order —
replacing `.sample()`/`.choice()` in both call sites; `agg_version_key` threaded through as the round-scoping
key. (2) sim now also does `self._sim_pending_commit.add(end)` synchronously on receipt (mirroring real's
live-reference guard), additive only — commit still discards it. 14 new tests (`test_selection_determinism.py`,
`test_fwdllm_real_pending_commit.py`), full `tests/` suite green. **Needs the next live pair** to confirm
`cohort_sequence` (and its downstream-gated `v2_var_trajectory`/`utility`/`v1b_iters_moving_avg`) clear.

**Contributing factor, still open.** Every trainer's first-ever GPU/JVP pass takes ~8-11x longer than its
later, steady-state passes (real median 4.11s vs 0.40s; sim 3.75s vs 0.48s) — a CUDA-context/kernel-compile
warmup cost, symmetric both sides, landing exactly in round 1's highest-leverage dispatch burst. GPU kernel
pre-warm at trainer startup **[LANDED, unvalidated live]** (`ForwardTextClassificationTrainer._warmup_gpu_
kernels`, called from `trainer/main.py` only — the aggregator's internal eval model must NOT warm up, that
crashed the first attempt). Needs a live pair to confirm it reduces round-1 jitter.

**Cleanup debt.** Exhaustive `[SELECT_TRACE]` debug logging landed in `async_oort.py` for the pm-13/14/15
localization above (every `select()` call logs full inputs at INFO) — deliberately verbose, not meant to ship
long-term; flag-gate behind DEBUG or delete now that the divergence is localized.

**Other open items (not a failing rung):**
- `trainer_speed_identity`'s `utility` sub-check reopened at 7200s scale (23/100 >10% dev) but failed to
  reproduce on 3 independent 30min pairs — leans flaky/noise, stays open until a ≥2h run adjudicates.
- `sim_sct_ordered_drain` A/B unblocked — run `fluxtune_n10_smoke_sim_no_sct_drain.yaml` against next pair.
- Accuracy drop after reaching 81% — known, deferred by operator, not yet triaged (see
  `fluxtune_contributions.md` §8).

### fwdllm / fwdllm_plus (`run_20260721_013501`/`_033648`, `_035015`/`_055227`, agg_goal=10)

Settled — only `agg_step_timing_breakdown`/`drain_wall_budget` fail (tracker table above), unaffected by
fluxtune's regression. `cohort_sequence`/`step_timing_breakdown`/`throughput`/`per_round_advance` all PASS.

### Cross-baseline / shared

- **felix (async_cifar10) may share fluxtune's round-1 cold-start gap** — `_sim_recv_min` uses the same
  reactive gate shape, no fallback for unseen ends. Felix's own comment claims it's empirically inert but
  UNVERIFIED. Out of this session's scope (`async_cifar10/PARITY.md` owns felix).
- felix 46/46 reconfirmation — deferred repeatedly, gates Phase 2.
- Momentum (S1-S3) / fluxtune server-optimizer retry — roadmap item, not parity; see
  `fluxtune_contributions.md` §8.2 / FWDLLM_DESIGN.md. Resume only after Phase-1 parity closes.

**P3 — infra robustness, not parity-blocking:** Dynamic GPU health filtering — `CUDA_DEVICE_ORDER=PCI_BUS_ID`
fixes *which* card an ordinal maps to, but doesn't detect/skip a genuinely broken one. No health check exists
in `flame/launch/` yet. Not attempted, lower priority.

---

## §F  Locked principles (from async_cifar10, carried over)
1. **Sim does real forward-grad compute, charges modeled time.** GPU runs the real JVP; the agg stamps
   `sct = sim_send_ts + max(real_gpu_s, D) + leg`. Never put overhead on the vclock (`vclock = max(vclock, sct)`).
2. **Progress axis is `data_id`.** Updates-per-data_id is the dynamic-K output to match, not an input to assume.
3. **Variance is an emergent gate; localize, never tune it.** `var_threshold`/`max_iterations_per_data_id` are
   baseline-defining knobs, not parity levers.
4. **Slot residence must survive variance rollbacks.** Release keys on the agg-goal boundary; the sct reorder
   buffer must not strand a grad across a rollback.
5. **Real is the reference only after admissibility.** Check whether the real input is the divergent side
   before tuning sim.
6. **Fix the concept, not the symptom.** Classify a mechanism as real-transport artifact (`and not
   self.simulated`) vs algorithmic property. Scope-check before editing shared code — `top_aggregator.py` /
   `_sim_recv_min` can silently break async_cifar10.
7. **Match pytest scope to blast radius.** fwdllm-only → `pytest tests/mode -k fwdllm`; shared parity engine →
   add async_cifar10 parity tests too; shared stack → full `pytest tests/`.
8. **Telemetry-FIRST, then instrument, then (rarely) run.** Validate/refute from telemetry already on disk
   first. Ship telemetry + plot + pytest in the same change as any new mechanism.
9. **Consult PARITY.md vclock rules before any sim-clock change.** Clock is a monotone `max`; sim skips real
   waits and reconstructs order from sct (`SimReorderBuffer`).
10. **The vclock is virtual wall-time; sim MUST produce speedup (`sim_rate = vclock/wall ≥ 1`).** `sim_rate < 1`
    means sim is stalling on a wait it should skip, or its commit throughput can't keep pace with arrivals.
11. **Correctness before speed; shared roots before per-baseline.** A bug failing rungs across ≥2 baselines
    outranks a single-baseline one.
12. **Logical determinism is the parity definition.** Same trainers selected, same order of update receipt,
    same aggregations/rollbacks — differing ONLY in wall-clock. Prove it on the first data bin before extending.
13. **Do the right thing — no hacks.** A hack that moves a number without a correct mechanism is a regression
    in disguise. When unsure, stop and ask.
14. **`version_key` is the ONLY version-identity vocabulary.** 2-tuple: base `(round, 0)`, fwdllm
    `(model_version, iteration_per_data_id)`. No bare-scalar shortcut.
15. **Verify claims against code, not documentation/comments.** A docstring claiming two functions are
    equivalent is a statement of intent, not a guarantee — diff them.
16. **Don't blame GPU/resource contention at n=10** — refuted once already; won't apply until ≥100-trainer
    scale. Any unexplained real-wall gap should be measured (wall-clock + vclock phase timer), not guessed.
17. **A bounded rotating in-flight cohort settling at `c − agg_goal` surplus is the correct steady state** for a
    `c ≫ agg_goal` fedbuff pool, not a backlog to eliminate — don't drive `carried_surplus_commits` toward 0.
18. **Any important knob is logged CONSISTENTLY everywhere, or it's a trap.** A configurable/correctness-path
    value (seed, delay floor, agg_goal, c, trace, flag state) must surface identically across yaml, snapshot,
    and both roles' telemetry — divergent/missing logging wastes sessions chasing phantoms.
19. **No compute on the critical path for a log the run doesn't need.** Any log with non-trivial arguments
    (`.item()`, hashing, `torch.allclose`/`.norm()`) MUST be gated behind `logger.isEnabledFor(logging.DEBUG)` —
    an f-string evaluates its args even when the level would drop the line.
20. **When real and sim disagree on timing, optimize REAL toward determinism — never inject noise into sim.**
    Sim's per-speed-class modeled duration must stay clean/reproducible (that's what makes `cohort_sequence`
    checkable at all). Fix real's measured completion time at its source instead.

---

## §G  Landed fixes — recent, load-bearing for current work only. Full history: `git log -- lib/python/examples/fwdllm/simulate_fwdllm.md`.

> **RULE: closed = here, ≤30 words, immediately.** The instant a rung flips or a hypothesis resolves, write
> ONE line (mechanism + outcome) and delete it from §A/§B in the same edit.

- **Real's `_agg_pending_commit_ref` was unbound, letting a buffered-but-uncommitted trainer get re-dispatched**
  (07-21) — bound to `_per_agg_trainer_list` (real-only), mirroring sim's own `_sim_pending_commit` binding.
  See §B — this session's dominant fix.
- **`_release_end_on_return` held real's slot to full-cohort commit, causing sawtooth (not flat) concurrency**
  (07-21 am) — `buffered=True` releases as soon as P0-1 has safely buffered the contribution. Exposed the
  asymmetry above; superseded by it, not reverted.
- **fwdllm trainer's remainder-wait sleep only compensated `gpu_time_s`, not real's total elapsed overhead**
  (07-20 pm-10) — now sleeps against elapsed-since-dispatch (`_wall_recv_ts`), closing an avoidable real-side
  noise source (§F-20).
- **`_compute_var`'s stop-the-world GC pause hypothesis REFUTED** (07-20 pm-11) — new `gc_pause_s` telemetry
  shows ~0ms GC time both sides.
- **`_distribute_weights_sync` missing from the real-only timing exemption set** (07-20 pm-11) — added,
  matching its already-exempted async twin.
- **`v2_var_trajectory`/`utility`/`throughput`/`per_round_advance` false-failed on population-length, not a
  real gap** (07-20 pm-5/pm-7) — gated on matched VIRTUAL BUDGET (each event's own commit timestamp filtered to
  `<= V`), not index-count or raw population. General pattern behind most "sim outruns real" false fails.
- **fluxtune's `pacer()` fired once per `select()` call instead of once per round**, ratcheting
  `round_threshold` to max and disabling the speed penalty (07-20 am) — fixed with an explicit
  `_last_pacer_round` guard, closing `preferred_duration`'s gap.
- **fluxtune `sim_send_ts` was the EOT/shutdown broadcast skipping the stamp by design, not a mid-run gap**
  (07-20 pm-2) — now stamped unconditionally.
- **fluxtune's `selection_train.vclock_now` was never stamped in sim**, blinding sim-side selection
  diagnostics without affecting the real side (07-20 pm-6) — fixed.
- **P0-1: buffer each trainer's contribution on receipt, merge into `self.grad` in canonical (D, trainer_id)
  order at commit** (07-18) — the deferred-merge foundation this session's fixes build on.
