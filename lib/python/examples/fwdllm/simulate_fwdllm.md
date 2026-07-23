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
genuine shared compute. Implementation-level reference (tiers, the `pctl_band_ok` band-escape primitive
and its `min_abs` calibration rule, full wall-budget/timing rung table): `async_cifar10/scripts/parity/PARITY_CHECKER_README.md`.

---

## §A  Score — refreshed 2026-07-23

**fluxtune 3→0 fails; 69/0/16 (checker regrade, this session).** All three fails
(`cohort_sequence.composition`, `v1b_iters_moving_avg`, `trainer_speed_identity.utility`) were ONE irreducible
boundary-race cascade — NOT a sim bug (§G/§H). The marginal cohort slot is a physical-FIFO-arrival (real) vs
modeled-sct (sim) near-tie that cascades, decorrelating index-paired IDENTITY to the independent-draw floor
(observed overlap 0.239 = floor 0.237) while EVERY marginal criterion matches (participation S2 tvd 0.023,
utility distribution KS 0.036, count 4.5%, v1/v2, speed_s 0/100). Index-paired identity is unattainable (0.8
target vs 0.237 ceiling), so those checks now GATE to diagnostic for stochastic-async selectors (mirrors S1/S2);
count, cum_mean_rel, speed_s stay enforced. sct-order-membership lever REJECTED (unrealistic; breaks under
Phase-2 unavailability). Details §H.

**fwdllm / fwdllm_plus — shared-compute timing family root-caused: co-location contention, NOT sim over-compute
(§G).** Only `drain_wall_budget` GATES (MECHANISM); `step_timing_breakdown` + `agg_step_timing_breakdown` are DIAG
(non-gating). Input sizes byte-identical (agg_goal 10, grad_pool 2.07, cached_v 25.92); sim's per-commit drain
floor (p10 35ms) equals real's typical (37ms) in EVERY decile with ~19% of commits real-matched throughout —
bursty contention from 100 co-located trainer threads, not more work. **Fix-1 `_flat_grad_norm`** (per-param
GPU→host sync → single on-device reduce, bit-identical, 55 tests) **re-measured at the 3600s re-run (07-23):
`drain_tail_s` p90 rel UNCHANGED** (fwdllm 0.935→0.942, fwdllm_plus 0.912→0.919) — fix-1 alone does not close the
vclock-charge gap; `sim_model_agg_compute_time` still charges the full contention-inflated drain wall onto the
vclock. Decision needed next: charge-the-floor vs relax (§B).

**Flag inventory (promotion call in §B):** `sim_model_agg_compute_time` is ON for all three;
`sim_sct_ordered_drain` + `sim_model_dispatch_queue` are fluxtune-yaml-only.

**Latest run per baseline** (`run_parity.py`, `lib/python/examples/fwdllm/expt_scripts`):

| baseline | run pair | duration | pass | fail | skip |
|---|---|---|---|---|---|
| fluxtune/syn_0 | `run_20260723_044359`/`_064615` (agg_goal=10) | ~7200s | 69 | 0 | 16 |
| fwdllm/syn_0 | `run_20260723_161459`/`_171648` (agg_goal=10) | ~3600s | 59 | 3 | 22 |
| fwdllm_plus/syn_0 | `run_20260723_161647`/`_171829` (agg_goal=10) | ~3600s | 61 | 2 | 21 |

(fwdllm/fwdllm_plus: post-`_flat_grad_norm`-fix re-run, 3600s. Same pass/fail/skip counts as the pre-fix 7200s
pair — the timing family persists (§B). fluxtune re-graded on stored dirs with this session's checker change.)

**Key-rung status** (✓ pass · ✗ fail · – skip; catalog: `async_cifar10/PARITY.md` §F):

| baseline | cohort | vclock | thru | commits | terminal | R1 | V1 | V2 | U3 | S2 | conv | conv_loss |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fluxtune | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| fwdllm | ✓ | ✓ | ✓ | ✓ | ✓ | – | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| fwdllm_plus | ✓ | ✓ | ✓ | ✓ | ✓ | – | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

fluxtune CLEAN (all rungs pass). fwdllm/fwdllm_plus's remaining `drain_wall_budget` (gating) +
`step_timing_breakdown`/`agg_step_timing_breakdown` (DIAG) are ONE family — co-location contention (root-caused
§G), fix-1 landed; re-run + vclock-charge re-measure pending (§B).

---

## §B  Next steps / open issues — per baseline

> **RULE: every tracker cell ≤20 words.** State the claim/number, cut qualifiers. If it needs more, it's not
> tracker material — shorten it or point at the code comment/commit.

**fluxtune: all 3 fails RESOLVED this session** (boundary-race cascade → stochastic-async identity gating; →
§G/§H). No open fluxtune parity gap except the deferred 81% accuracy drop below.

**Shared-compute timing family ROOT-CAUSED (this session → §G): co-location contention, NOT sim over-compute.**
Only `drain_wall_budget` GATES (MECHANISM); `step_timing_breakdown` + `agg_step_timing_breakdown` are DIAG
(non-gating). Evidence: input sizes byte-identical (agg_goal 10, grad_pool 2.07, cached_v 25.92); sim drain floor
p10 35ms = real 37ms in EVERY decile, ~19% commits real-matched throughout (bursty, not a warm-up leak);
thread-local `cpu_duration_s` tracks wall (on-CPU burn, not deschedule).

| Baseline | Rung | State | Next |
|---|---|---|---|
| FW, FW+ | `drain_wall_budget` (GATING) | fix-1 `_flat_grad_norm` LANDED + RE-RUN done (3600s, 07-23): p90 rel unchanged (0.94/0.92) | fix-1 insufficient alone — decide charge-floor vs relax |
| FW, FW+ | `agg_step_timing_breakdown` (DIAG) | same contention; does NOT gate verdict | informational |
| FW | `step_timing_breakdown` (DIAG) | same; does NOT gate | informational |

**vclock-charge re-measure — DONE (3600s re-run, 07-23).** `sim_model_agg_compute_time: true` still charges the
RAW measured drain+fedavg wall onto the sim vclock; fix-1 did not shrink it — `drain_tail_s` p90 real/sim rel
unchanged (fwdllm 0.935→0.942, fwdllm_plus 0.912→0.919), `per_round_advance` central-escape still needed (KS
0.371/0.108), `drain_wall_budget` still fails. Fix-1 (`_flat_grad_norm`) was NOT the dominant contention source.
Next: decide charge-the-floor vs relax (§F-20, don't inject sim-host noise into the clock), or keep digging for
the actual amplifier first.

**Flag-promotion decision (next step, operator call per [[flag-gate-ab-lifecycle]]).**
`sim_model_agg_compute_time` is effectively default (ON all three baselines). `sim_sct_ordered_drain` +
`sim_model_dispatch_queue` are in fluxtune's sim yaml ONLY, but both model GENERAL async-transport artifacts
(sct-ordered drain, serial-dispatch queue), not fluxtune-specific mechanics. Next: run fwdllm/fwdllm_plus sim
smoke with both flags ON to confirm inert-or-better (they already pass cohort/throughput/per_round), then
promote all three to code-level default-on and delete the gates.

**Other open (not the cascade):**
- Accuracy drop after reaching 81% — known, deferred by operator (`fluxtune_contributions.md` §8).

### fwdllm / fwdllm_plus (`run_20260723_161459`/`_171648`, `_161647`/`_171829`, agg_goal=10, 3600s)

Throughput parity CLOSED, validated 7200s (→ §G). The remaining timing family is root-caused above (co-location
contention, `drain_wall_budget` gating; DIAG step-timing checks non-gating); fix-1 `_flat_grad_norm` landed and
re-measured — gap unchanged, fix-1 alone insufficient (charge-floor-vs-relax decision open above). No open
sync/throughput gap.

### Cross-baseline / shared

- **felix (async_cifar10) may share fluxtune's round-1 cold-start gap** — `_sim_recv_min` uses the same
  reactive gate shape, no fallback for unseen ends. Felix's own comment claims it's empirically inert but
  UNVERIFIED. Out of this session's scope (`async_cifar10/PARITY.md` owns felix).
- felix 46/46 reconfirmation — deferred repeatedly, gates Phase 2.
- Momentum (S1-S3) / fluxtune server-optimizer retry — roadmap item, not parity; see
  `fluxtune_contributions.md` §8.2 / FWDLLM_DESIGN.md. Resume only after Phase-1 parity closes.

**Tech debt — sim/real in-flight bookkeeping is over-complex; simplify AFTER this fix validates.**
The §F.1-23 deadlock was a "too many sources of truth" bug: sim tracks the same virtual in-flight set across
`_sim_pending_commit`, `_sim_inflight_expected`, `_sim_buffer`, `_sim_committed`, `selected_ends`,
`all_selected`, reconciled by `_sim_hold_busy_slots` — and one add at the wrong seam desynced them. Two smells:
(a) those sets should be ONE authoritative per-end state (`dispatched → returned/buffered → committed`) with
the slot/guard sets DERIVED, not maintained in parallel; (b) `_process_single_trainer_message` means RECEIPT in
real but COMMIT in sim — the exact ambiguity that bit here — so the receipt vs commit responsibilities should
split. If we keep hitting deep bugs here, that refactor becomes the priority. Do it as its own scoped step
behind the new loop-level characterization tests (`test_fwdllm_sim_grad_loop.py::TestCommitThenProcessFreesTheSlot`),
never bundled with a correctness fix (would muddy live parity validation).

**P3 — infra robustness, not parity-blocking:** `_check_gpu_health()` now aborts pre-spawn on a broken ordinal
(§G), and `execution.gpu_ids` lets the operator exclude one manually. Still no *automatic* skip-and-remap of a
broken card — the operator must pass `--gpu-ids`. Lower priority.

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

### §F.1 Version & commit invariants (confirmed real+sim in code + real logs, 2026-07-21)

21. **`model_version` bumps once per COMPLETED data-bin** (variance PASS → global model update, `+= 1` at the
    data_id advance) — constant across all iterations of one data-bin. `iteration_per_data_id` bumps on every
    variance-FAIL retry, resets to 0 on data-bin advance. `version_key = (model_version, iteration_per_data_id)`
    therefore changes EVERY iteration and is the sole step identity (§F-14).
22. **Commit == the update being used for aggregation, and it happens at that instant — no lag.** Real: on
    ordered arrival (trainer already waited). Sim: when the vclock reaches the update's `sct` (buffer-unlock IS
    the commit). An update drained for aggregation must be committed in the same step, never on a later
    event/aggregation.
23. **Commit partially unlocks the trainer: it frees the compute slot immediately, but a version_key re-pick
    guard (`_trainer_state_dict`) keeps it un-pickable for the SAME `(model_version, iteration)`.** It re-enters
    the pool once the version_key advances (next iteration or next data-bin). Sim's slot-hold
    (`_sim_pending_commit`) must be cleared at commit — never re-added after — or the slot never frees and
    re-dispatch across variance-retry iterations starves.
24. **Within a data-bin the global weights are constant; a re-picked trainer gets a RETRY, not a re-send.** The
    aggregator sends full WEIGHTS to a trainer only for a `model_version` it has not yet received this data-bin
    (`_weights_sent_this_cycle`, cleared on the `model_version` bump). A re-pick at the same `model_version`
    (still on this data-bin) is told VAR=bad — recompute new perturbations — never a redundant weight re-send.

---

## §G  Landed fixes — recent, load-bearing for current work only. Full history: `git log -- lib/python/examples/fwdllm/simulate_fwdllm.md`.

> **RULE: closed = here, ≤30 words, immediately.** The instant a rung flips or a hypothesis resolves, write
> ONE line (mechanism + outcome) and delete it from §A/§B in the same edit.

- **fluxtune 3→0 (69/0/16): all fails were ONE boundary-race cascade, not a sim bug** (07-23) — marginal cohort
  slot is a physical-FIFO vs modeled-sct near-tie; index overlap 0.239 = independent-draw floor 0.237; every
  marginal criterion matches (S2/utility-dist/count/v1/v2/speed). Checker: gate index-identity for stochastic-async
  (`cohort_sequence.composition`+first-bin, `trainer_speed_identity.utility`, `v1b` MA-shadow) → diagnostic; count/
  cum_mean_rel/speed_s enforced; added `independent_draw_floor` diagnostic. 204 tests. sct-order lever rejected (§H).
- **fwdllm timing family: co-location contention, NOT over-compute** (07-23) — input sizes byte-identical, sim
  drain floor p10 35ms = real 37ms every decile, thread-local cpu tracks wall. Only `drain_wall_budget` gates;
  step-timing checks are DIAG. `_flat_grad_norm` per-parameter GPU sync → single on-device reduce (bit-identical,
  55 tests). vclock-charge re-measure pending re-run (§B).
- **fwdllm/fwdllm_plus throughput CLOSED, validated at 7200s** (07-23) — recv_fifo→`drain_ready` + var_bad
  dedup/`pause_execution` removal held: fwdllm 3.2%, fwdllm_plus mw 4.8%, both PASS. Was the §H sync gap.
- **fluxtune `v2_var_trajectory` + `drain_wall_budget` PASS under logical-N** (07-23) — v2 real 0.921/sim 0.919
  (0.3%); fluxtune drain_tail real 0.434/sim 0.69 in-band. fluxtune 5→3 fails; both dropped from the cascade.
- **`cohort_sequence.count` PASSES on the matched logical budget** (07-23) — real 1750/sim 1833 rel 4.5% <5%.
  (`composition` since resolved as a boundary-race cascade → gated, see top of §G.)
- **fwdllm `per_round_advance` PASSES via central-tendency escape** (07-23) — mean_rel 3.2% (tol 15%), KS 0.345
  tolerated by the `pctl_band_ok` central escape. Round-1 tail no longer trips it.
- **Checker: `matched_virtual_budget` deleted → grade on the LOGICAL budget N** (07-23) — `V=min(sim
  vclock, real wall)` conflated the two clocks (the axis `sim_rate` tests). New `_matched_logical_budget`
  (progress ≤ N); U2/K8 reshaped count→**time-to-N**, v2/utility/`cohort_sequence.count` swapped to
  N-truncation, `cohort_sequence` deps V1. 666 tests pass. Design: PARITY.md §1.5.
- **Checker: `pctl_band_ok` DIST-band escape landed + tested** (07-23) — `_step_timing_compare`,
  `drain_tail_s`, `per_round_advance` central-tendency escape. 24 new tests; README (`parity/PARITY_CHECKER_README.md`).
- **Checker: `_step_timing_compare`'s `band_min_abs_s` (0.5s) silently passed 5x step-timing regressions**
  (07-23) — 10-100ms-scale functions vs a 500ms floor copied from drain's ~1s scale; anchored to the
  metric's own noise constant (`_STEP_TIMING_NEAR_ZERO_ABS_DIFF_S`=3e-4s) instead.
- **fluxtune 19→5 validated at 7200s** (07-22) — selector rebind + `_keyed_topk` + sim-deadlock fix (§F.1-23)
  + dispatch-queue/commit-fold vclock charging all held; `preferred_duration`/`terminal_state`/`total_commits`
  now PASS. Remaining fails all downstream of a residual ~9% vclock under-charge (§B).
- **Sim deadlock: `_process_single_trainer_message`'s `else: _sim_pending_commit.add` re-pinned committed
  trainers** (07-21 pm) — dropped the commit-time add; dispatch-time add + version_key guard already cover
  re-pick. Cleared `sim_rate` 0.07 stall.
- **Startup crashes (unhealthy GPU 0 + unconditional CUDA RNG init)** (07-21) — device-gated `torch_cuda_rng`,
  `_check_gpu_health()` preflight allocation, `execution.gpu_ids` ordinal allowlist; 7200s pairs ran clean.
- **`cohort_sequence` grading made distributional (set-overlap ≥0.8)** (07-21) — absorbs boundary-race
  cascades; fluxtune now fails it only via the upstream `data_id` drift, not the grade (§B).
- **`[SELECT_TRACE]` debug logging removed** (07-22) — divergence localized, verbose tracing deleted.
- **GPU kernel pre-warm landed & ran** (07-21) — `_warmup_gpu_kernels` at trainer startup (trainer only, not
  agg eval model); round-1 tail still visible in fwdllm `per_round_advance`, tracked in §B.
- **Real's `_agg_pending_commit_ref` was unbound, letting a buffered-but-uncommitted trainer get re-dispatched**
  (07-21) — bound to `_per_agg_trainer_list` (real-only), mirroring sim's own `_sim_pending_commit` binding;
  validated at 7200s (19→5, line above).
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

---

## §H  Throughput parity — sync fixes VALIDATED at 7200s (fwdllm/fwdllm_plus → §G); fluxtune composition RESOLVED (2026-07-23)

> Sync-path throughput CLOSED, validated at 7200s (→ §G): fwdllm 3.2%, fwdllm_plus mw 4.8%. The fluxtune "async
> composition skew" is RESOLVED — it was a boundary-race cascade (index identity unattainable, not a sim skew),
> now graded on marginals (see RESOLVED subsection below). This section is the handoff + negative-result record.

### Validated sync fixes (fwdllm / fwdllm_plus) — both held (2700s A/B, reconfirmed at 7200s → §G)
Both were real-side transport artifacts on the sync `distribute→collect(1)` loop; real must match sim's
`decision + max(D)` per round (§F-20). Mechanism detail is in `git log`; summary + current-run numbers:
- **Fix 1 — recv_fifo streamer stall → `drain_ready`.** `recv_fifo`'s fire-and-forget per-end streamers stranded
  already-arrived grads to a 30s timeout (89×30s = 40% of collect wall). Real twin `_real_sync_recv_incremental`
  (flag `real_drain_ready_ingest`, ON in `fwdllm_n100_smoke.yaml`). Gap 1.57× → 1.31×.
- **Fix 2 — var_bad flood + 1s trainer poll.** `_distribute_weights_sync` re-sent VAR=bad to the whole cohort
  ~agg_goal×/iteration; the trainer drained them FIFO at 1/sec, delaying its next compute ∝ D (~10s/round).
  Aggregator one-instruction-per-version_key dedup (`_end_served_version_key`) + removed the `pause_execution`
  `sleep(1)`. Default-on, no flag. Pytest: `test_fwdllm_instruction_dedup.py`, `test_fwdllm_sim_speedup_waits.py`.

**Result — the ~10s/round residual is GONE and CONFIRMED at 7200s (→ §G):** real per-round wall sits at/below
sim's `max(D)`. fwdllm real 60.55s/round vs sim 62.54 (**3.2%**, PASS); fwdllm_plus matched-window **4.8%** PASS
(the 2700s mw 5.6% small-N marginal cleared on the longer pair). Both sync fixes migrated to §G.

### Latest run — 7200s pair per baseline (`run_parity.py`, agg_goal=10)
Real `_001159`/`_022604`/`_044359`, sim `_021350`/`_042757`/`_064615` (fwdllm / fwdllm_plus / fluxtune).

| baseline | pass/fail/skip | throughput | cohort_seq | v2 | other fails |
|---|---|---|---|---|---|
| fwdllm | 59 / 3 / 22 | ✓ 3.2% | ✓ | ✓ | step_timing, drain_wall_budget, agg_step_timing (contention, §G) |
| fwdllm_plus | 61 / 2 / 21 | ✓ mw 4.8% | ✓ | ✓ | drain_wall_budget, agg_step_timing (contention, §G) |
| fluxtune | 69 / 0 / 16 | ✓ 0.1% | ✓ | ✓ | — (composition/v1b/utility resolved: boundary-race cascade → gated) |

Reading the fails:
- **FW/FW+ `cohort_sequence` now PASSES** (07-23) — `count` grades at the matched LOGICAL budget N
  (`_matched_logical_budget`, PARITY.md §1.5) and `composition` is perfect; the comparison-AXIS artifact is gone.
- **FW+ small-N marginals CLEARED at 7200s:** throughput mw 4.8% and v2 full-run 1.1% both PASS. The 2700s
  mw 5.6% / v2 mw 2.55% were length artifacts, as expected.
- **fluxtune composition skew (the open work — see below):** `cohort_sequence.composition` genuinely fails
  (mean_overlap 0.251), `v1b_iters_moving_avg` (real 11.67/sim 12.22), `trainer_speed_identity.utility` 29/100
  >10% tol. `speed_s` PERFECT (0/100). `v2` now PASSES (0.3%). The cohort surplus narrowed to ~4.7%.
- **Shared, pre-existing (§B) — ONE shared-compute wall-inflation family:** the transport funcs are ALREADY
  exempt (`gates_ok=False`); the fails are NON-exempt genuine shared-compute funcs where sim's physical wall
  exceeds real's beyond tolerance. `agg_step_timing_breakdown` (`_compute_var` 5→33ms 6.5x, `_replay_buffered_cohort_contribs`
  44→369ms, `_process_aggregation_goal_met` 176→680ms; FW/FW+); `drain_wall_budget` drain_tail (measured
  cohort-merge replay, p90 real 0.10/sim 1.55s; fluxtune PASSES); FW `step_timing_breakdown` (`_make_model_functional`).
  Leading hypothesis §F-10 host contention (README lines 65/68 already attribute agg/drain to it), but observed
  2-15x exceeds the sized-for-2x tolerances — adjudicate contention vs sim over-compute from logs (next context).

### RESOLVED (07-23) — fluxtune "composition skew" was a BOUNDARY-RACE CASCADE, not a sim skew
The old "D=8.33 skew (real 49.5%/sim 56.8%)" is NOT visible at cohort-slot level in this run (46.6%/48.7%,
+2.1%); speed-class shares AND marginal participation match (per-trainer r=0.979, S2 tvd 0.023). Mechanism:
each cohort's MARGINAL 10th slot is a sub-100ms arrival tie among ~16 interchangeable D=8.33 trainers — real
admits physical-FIFO-first, sim the lowest-sct — a coin-flip that cascades (the excluded trainer fronts the next
cohort) and Oort path-dependence amplifies. Index overlap decays 0.9→0.24 over ~11 cohorts then PLATEAUS at the
independent-draw floor (**observed 0.239 = floor 0.237**): real and sim are two independent samples of the SAME
process. Every selection CRITERION verified matched (eligible_fingerprint 100%, speed 0/100, utility distribution
KS 0.036, S2, counts) — and since fluxtune's selector is `AsyncOortSelector` (speed×utility, not speed-only), the
ID/fingerprint comparison — not a speed projection — is what confirmed it.

- **The two A/B fixes were inert because there is NO chargeable per-cycle mechanism** — the divergence is a
  stochastic tie-break, not an aggregator leg. Confirmed, not a loose end.
- **sct-order-membership lever REJECTED** (was "highest-leverage untried"): admitting the lowest-sct 10 instead
  of the first-arrived means the aggregator BLOCKS on future arrivals / holds slots for possibly-offline
  trainers — FIFO-violating and DEADLOCKS under Phase-2 unavailability. Do not pursue.
- **Resolution LANDED:** index-paired IDENTITY is unattainable (0.8 target vs 0.237 floor) → gate it for
  stochastic-async selectors (`cohort_sequence.composition`+first-bin, `trainer_speed_identity.utility`, `v1b`
  MA-shadow) to diagnostic; count/cum_mean_rel/speed_s stay enforced, S2 owns the mix-bias catch. Added the
  `independent_draw_floor` diagnostic. 204 checker tests. fluxtune 69/0/16.

(Historical A/B / verification detail below is superseded background — kept for the negative-result record.)

**Verification of the operator's hypothesis ("agg sends at T, update commits at T+D"), pair `_222013`/`_222057`:**
- **SIM commits at ≈ T+D** — `commit_gap_s = vclock−sct` median 0.0, not speed-correlated. Clean.
- **REAL commits at T+D + read-wait**, but the read-wait is dominated by *genuine* cohort-fill (real physical
  arrival spread 3.55s ≈ sim sct-spread 3.40s — the fill is faithfully modeled). The `post-fill stall`
  (commit−last-arrival, 0.5s) is aggregator compute, and sim already OVER-charges its `drain_tail`/`fedavg`
  (0.92 vs real 0.75). So neither fill nor commit-compute is the gap.
- **The gap is `_distribute_weights_async`:** real **0.143s/call** vs sim **0.0345s/call** — the **0.108s delta is
  the real-only settle sleep**. Per cohort (~10 distributes): real distribute **1.43s** vs sim **0.34s**, Δ
  **~1.08s/cohort** of real wall sim neither runs nor charges. Being a per-cohort quasi-constant it compresses
  real's fast/slow cycle ratio → sim over-weights fast trainers (D=8.33 share **49.5%(real)→56.8%(sim)**,
  slow(≥25) 2.7%→2.1%), the one skew behind all four fails. Per-cohort: real 4.02s vs sim 3.70s (~8%), ratio 1.087.
- **Why sync (fwdllm/plus) was fine:** the SAME sleep is in `_distribute_weights_sync`, ~10×/round — but a sync
  round is ~57s so 10×0.1s ≈ **1.7%** (within fwdllm's validated residual). Async cohorts are ~4s → the same 1s
  lands as ~25%. Present on both; material only on async.

**recv_fifo/drain_ready fix — LANDED but INERT on the gap (negative result).** `_real_async_recv_min_grad`
(streamer-free `drain_ready`, commit at T+D) replaced the async recv_fifo path; it dropped the 181,703
"already has active task" log lines and aligns async with the sync collect, but the D-skew (49.5/56.8) and
per-cohort wall (4.02 vs 3.70) were **unchanged** — the recv_fifo streamer was NOT the mechanism. Kept as a
cleanup (flag `real_drain_ready_ingest` ON), not a parity fix. `leg` is a uniform additive constant so it cancels
in the per-cohort delta (grow-leg hypothesis REFUTED).

**ENABLING CHANGE LANDED:** the two hardcoded `time.sleep(0.1)` pads (`_distribute_weights_sync/_async`) now read
the existing `real_distribute_settle_s` knob (code-default 0.1 = byte-identical). `fluxtune_n10_smoke.yaml` sets it
to **0.0** for the A/B.

### `matched_virtual_budget` deleted — grade on the logical axis (LANDED 07-23; design PARITY.md §1.5)
This section IS the evidence. The sync leg above proves V **masks**: FW/FW+ read at-parity under V while a
real **1.57× throughput gap** hid, found only via raw databins/wall and fixed on the real side. The fluxtune
skew proves V **fails to grade**: sim still runs 1795 vs 1648 cycles *inside* the matched V. So V is neither
necessary nor sufficient for parity — it conflates sim vclock with real wall, the very thing `sim_rate` tests.
**Landed:** `_matched_virtual_budget` deleted, `_matched_logical_budget`/`_time_to_progress` added; U2/K8
reshaped count→time-to-N, v2/utility/`cohort_sequence.count` swapped to progress-≤-N, `cohort_sequence`
deps V1; report.py + README updated; 666 tests pass. **Still open:** launcher still ends on a wall budget
(checker truncates to N post-hoc — correct, but a fixed-N termination would drop the wasted tail).

### OTHER OPEN (independent of the fluxtune skew above)
- **FW+ marginals CLEARED at 7200s** (07-23) — throughput mw 4.8%, v2 full-run 1.1%, both PASS. Closed.
- **Settle sleep:** real ran clean at `real_distribute_settle_s: 0.0` (droppable dead weight), but it's NOT the
  parity cause — don't expect it to move fluxtune. Verify selection determinism before dropping code-wide.

**Checker invariants I1-I6** (drafted in chat, not yet written up) — re-derive AFTER the
vclock/throughput root-cause lands (they hinge on it).
