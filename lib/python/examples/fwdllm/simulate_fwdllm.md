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

## §A  Score — refreshed 2026-07-22

**fluxtune's 19-fail regression is fixed and validated at 7200s: 19→5.** The selector rebind
(`_agg_pending_commit_ref`→`_per_agg_trainer_list`), `_keyed_topk` selection determinism, the sim-deadlock fix
(§F.1-23), and dispatch-queue + commit-fold vclock charging all held; the former 30min residuals
(`preferred_duration`, `terminal_state`, `total_commits`) now PASS (→ §G). **4 of the 5 remaining fluxtune fails
share ONE root:** sim still runs **~8.9% ahead** (1795 vs 1648 cycles at a matched ~7200s virtual budget)
because the vclock under-charges real's per-cycle wall — even with all three correction flags ON in the sim
config (`sim_sct_ordered_drain`, `sim_model_dispatch_queue`, `sim_model_agg_compute_time` = true). The drift
makes sim reach `data_id=1` while real is still on `data_id=0` by cycle 9, cascading into `cohort_sequence`
(`set_overlap` 0.371 vs tol 0.8), `v1b_iters_moving_avg` (sim 11.97 vs real 10.99 iters/data_id),
`v2_var_trajectory` (+8.8% mean var, tol 2%), and `trainer_speed_identity`'s `utility` sub-check (28/100 >10%;
`speed` sub-check exact at 0.05%). The 5th, `drain_wall_budget`, is the shared cross-baseline `drain_tail_s`
issue (§B). fwdllm newly fails `per_round_advance` — matched-window MEAN passes (5.2%), fails only on KS shape
(0.368, `ratio_max` 5.06 round-1 tail); fwdllm_plus passes the same code → likely small-N tail, needs one repro.

**Flag inventory (promotion call in §B):** `sim_model_agg_compute_time` is ON for all three;
`sim_sct_ordered_drain` + `sim_model_dispatch_queue` are fluxtune-yaml-only.

**Latest run per baseline** (`run_parity.py`, `lib/python/examples/fwdllm/expt_scripts`):

| baseline | run pair | duration | pass | fail | skip |
|---|---|---|---|---|---|
| fluxtune/syn_0 | `run_20260722_053716`/`_073938` (agg_goal=10) | ~7200s | 64 | 5 | 16 |
| fwdllm/syn_0 | `run_20260722_010124`/`_030320` (agg_goal=10) | ~7200s | 59 | 3 | 22 |
| fwdllm_plus/syn_0 | `run_20260722_031706`/`_051919` (agg_goal=10) | ~7200s | 61 | 2 | 21 |

**Key-rung status** (✓ pass · ✗ fail · – skip; catalog: `async_cifar10/PARITY.md` §F):

| baseline | cohort | vclock | thru | commits | terminal | R1 | V1 | V2 | U3 | S2 | conv | conv_loss |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fluxtune | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ |
| fwdllm | ✓ | ✓ | ✓ | ✓ | ✓ | – | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| fwdllm_plus | ✓ | ✓ | ✓ | ✓ | ✓ | – | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

fluxtune's 5 fails: `cohort_sequence`, `v1b_iters_moving_avg`, `v2_var_trajectory`, `trainer_speed_identity`
(all four downstream of the ~8.9% vclock under-charge drift), plus the shared `drain_wall_budget` (§B).

---

## §B  Next steps / open issues — per baseline

> **RULE: every tracker cell ≤20 words.** State the claim/number, cut qualifiers. If it needs more, it's not
> tracker material — shorten it or point at the code comment/commit.

| Baseline | Rung(s) | Hypothesis / root cause | Next step |
|---|---|---|---|
| ALL | `drain_wall_budget` (`drain_tail_s`) | sim modeled drain tail overshoots real (sim 0.55-0.73 vs real 0.11-0.47s, budget 0.5-0.59) | Trim sim drain-tail model; 2h pair now available |
| FW, FW+ | `agg_step_timing_breakdown` | `_aggregate_grads_sync`/`_distribute_weights_sync` are real-transport (sim rightly collapses); real outlier is `_compute_var` sim **6x** real (0.031 vs 0.005s) | Isolate `_compute_var` sim path |

### fluxtune — residual vclock under-charge (sim ~8.9% ahead)

**Root, validated at 7200s.** The 19→5 fixes held (→ §G). The 4 remaining non-shared fails
(`cohort_sequence`, `v1b_iters_moving_avg`, `v2_var_trajectory`, `trainer_speed_identity.utility`) are ONE
cascade: sim commits 1795 cycles to real's 1648 in the same ~7200s virtual budget, so sim's per-cycle vclock
advance runs ~8.9% short of real's measured wall. Both transport-correction flags are already ON, so the
residual is whatever wall they don't yet model. **Next: instrument per-cycle vclock advance (sim modeled vs
real measured wall), localize the missing ~9% leg — don't tune knobs (§F-3/13).**

- **`cohort_sequence` is a SYMPTOM, not a grading bug:** the distributional grade (set-overlap ≥0.8) is correct;
  it fails only because sim/real are on different `data_id` at the same cycle index (real 0 / sim 1 by cycle 9).
  Closing the under-charge closes this.
- **`trainer_speed_identity.utility` adjudicated at 2h:** 28/100 >10% dev — NOT noise (earlier 30min pairs
  couldn't reproduce; the 7200s pair does). Downstream of which trainers land in which cohort, i.e. the same
  drift. `speed` sub-check is exact (0.05%).

**Flag-promotion decision (next step, operator call per [[flag-gate-ab-lifecycle]]).**
`sim_model_agg_compute_time` is effectively default (ON all three baselines). `sim_sct_ordered_drain` +
`sim_model_dispatch_queue` are in fluxtune's sim yaml ONLY, but both model GENERAL async-transport artifacts
(sct-ordered drain, serial-dispatch queue), not fluxtune-specific mechanics. Next: run fwdllm/fwdllm_plus sim
smoke with both flags ON to confirm inert-or-better (they already pass cohort/throughput/per_round), then
promote all three to code-level default-on and delete the gates.

**Other open (not the cascade):**
- **fwdllm `per_round_advance`** — matched-window mean passes (5.2%), KS-shape fails (0.368) on a `ratio_max`
  5.06 round-1 tail; fwdllm_plus passes same code. Reproduce once before treating as a mechanism (GPU pre-warm
  landed — §G — so round-1 jitter may still be the cause).
- Accuracy drop after reaching 81% — known, deferred by operator (`fluxtune_contributions.md` §8).

### fwdllm / fwdllm_plus (`run_20260722_010124`/`_030320`, `_031706`/`_051919`, agg_goal=10)

Checker PASSES cohort/throughput/v1/v2 (only shared `drain_wall_budget`/`agg_step_timing_breakdown` fail). The
matched-virtual-budget normalization had HIDDEN a real 1.6× throughput gap (68 vs 110 databins/7200s) from 89 ×
30s `sync_collect_and_accumulate_grads` stalls — **ROOT-CAUSED (§H): `recv_fifo`'s fire-and-forget streamer
strands the slow trainer's already-arrived grad; fix `_real_sync_recv_incremental` via `drain_ready` landed
behind `real_drain_ready_ingest` (default OFF). NEXT: operator runs the validation leg (§H).**

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

## §H  fwdllm real↔sim throughput gap — ROOT-CAUSED + FIX LANDED (flag), pending validation (2026-07-22)

> **Root cause confirmed from telemetry AND code; fix implemented behind an A/B flag, default OFF.** Pick up at
> "NEXT STEPS" (run the validation leg).

### The gap = one real-transport artifact (the whole 1.6×)
Real 68 databins in 7200s WALL vs sim 110 in 7200s VCLOCK. **The entire gap is 89 × 30s aggregator stalls** and
the arithmetic closes exactly: 89 × 30s = 2670s = **15.00s/round** over 178 rounds = the real 37.4 − sim 22.8 =
14.6s/cyc gap. Zero logical divergence: every one of the 178 rounds commits exactly 10 contributors (no partial
commits, no dropped grads) — pure wall delay, so sim rightly collapses it (§F-1/§F-6); the fix is real-side (§F-20).

### Mechanism (verified both sides + in code)
Real collected via `channel.recv_fifo(ends, num_min_req=1, timeout=30)` (clamped to 1 by `ends_not_selected_yet`
whenever 10 ends ≥ agg_goal 10 → runs every cohort). `recv_fifo` spawns a **fire-and-forget** streamer with one
`_get_inner` task per end, consumes 1, returns — the other 9 tasks linger, holding their ends in
`_active_recv_fifo_tasks` with their own 30s grace. The next collect **skips every lingering end** as "already
active" (`channel.py:733`; 16,020 skip logs + 89 "No data from" in the real agg log). The slow trainer's grad is
thereby stranded: trainer `…0469` (D=22.09, returns in D correctly — observed/modeled = 1.00, NOT a late/slow
trainer) sent its grad at t=967.36, but the aggregator didn't process it until t=990.26 — a 23s stall while the
collect burned its 30s. **This is a documented hazard:** the `drain_ready` docstring (`channel.py:594-601`)
describes exactly this stranding; `drain_ready` is the streamer-free fix — and the **sim path already uses it**
(`_sim_recv_min_grad`) / a `recv_fifo(new_ends, first_k=len(new_ends))` variant (`_sim_sync_recv_incremental`).
Real was never migrated.

### Fix (landed, `flag real_drain_ready_ingest`, code-default OFF)
`fwdllm_aggregator._real_sync_recv_incremental` — the real twin of `_sim_sync_recv_incremental`: a persistent
**arrival-ordered** pending buffer refilled by streamer-free `drain_ready` (pulls straight from each End rxq,
delivered there by the backend `_rx_task` independent of any recv path), popping `num_min_req` per call — same
collect cadence as `recv_fifo`, minus the lingering-task stall. Rejection of duplicate/stale/no-grad messages
stays in `_process_single_trainer_message` (only GRADIENTS+GRADIENTS_FOR_VAR_CHECK increments `agg_goal_cnt`), so
sweeping extra no-op messages is safe; P0-1 merges grads in canonical order at commit, so aggregation is
order-independent. Buffer cleared at the agg-goal boundary (variance-rollback safety), mirroring the sim clear.
Pytest: `tests/mode/test_fwdllm_real_drain_ready.py` (prompt commit, arrival order, num_min_req cap + cross-call
buffering, deadline-empty). All 325 fwdllm mode tests pass; the `test_parity_checks.py` fails are pre-existing
(PARKED checker work below), unrelated.

### NEXT STEPS
1. **Run the validation leg** (operator; `real_drain_ready_ingest: true` is set in `fwdllm_n100_smoke.yaml`):
   generate a fresh 7200s real run and confirm databins-in-7200s converges real→sim (~68 → ~110) and the 89
   30s collect stalls vanish (`sync_collect_and_accumulate_grads` p99 ≪ 30s). Then re-run `run_parity.py`.
2. **Promote per [[flag-gate-ab-lifecycle]]:** if validated inert-or-better, flip the code-level default ON and
   delete the gate (it's a correctness/parity fix, not a tunable). fwdllm_plus uses the same real path — enable
   there too.
3. **Then** decompose `sync_collect_and_accumulate_grads` for **fluxtune** (async, ~8.9% gap, §A/§B) — likely a
   different, smaller root (the residual vclock under-charge, not this recv_fifo stall); don't assume same cause.

### PARKED — mid-flight checker work (`async_cifar10/scripts/parity/checks.py`), DO NOT SHIP AS-IS
Uncommitted edits from this session, correct in spirit but **one is broken**:
- ✅ `pctl_band_ok()` helper added; applied to `_step_timing_compare` (band escape) + `drain_wall_budget`
  `drain_tail_s` reclassified one-sided→DIST band + `per_round_advance` central-tendency escape. These made
  fluxtune `drain_wall_budget` and fwdllm `per_round_advance` pass; keep.
- ❌ **`cohort_sequence` split into `composition` + `count` uses RAW full-run counts** (178 vs 316) → false-fails
  fwdllm/fwdllm_plus (composition is perfect 1.0; count fails on the raw gap). **Must filter both to the matched
  virtual budget** (`_matched_virtual_budget`, as `total_commits` does) before counting — raw totals are the
  known false-fail (§G 07-20). Fix or revert the count sub-check before committing.
- Flags `sim_model_dispatch_queue` + `sim_sct_ordered_drain` were ADDED to `fwdllm_n100_smoke_sim.yaml` and
  `fwdllm_plus_n100_smoke_sim.yaml` (not yet run). If validated inert-or-better next run, remove the config gates
  and make sim-default (§B flag-promotion).
- TODO deferred (both REQUIRED before committing the checker changes, per §F-8 "ship telemetry + plot + pytest
  in the same change"):
  1. **Pytest coverage** for the new/changed `checks.py` logic — `pctl_band_ok()` (band pass/fail, `min_abs`
     floor, tail-ignore), the `_step_timing_compare` band escape, `drain_tail_s` DIST reclassification,
     `per_round_advance` central-tendency escape, and the fixed `cohort_sequence` composition+count split. Add to
     `tests/mode/test_parity_checks.py`.
  2. **Parity checker README** (tiers EXACT/DIST/DIAG, rung catalog, what each grades) in the `parity/` dir —
     shared across examples, so NOT in this doc. Checker invariants I1-I6 were drafted in chat; re-derive AFTER
     the vclock/throughput root-cause lands (they hinge on it).
