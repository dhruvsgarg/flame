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

## §A  Score — refreshed 2026-07-19 (see PREAMBLE's score-tracking trigger)

> **All three rows are fresh 7200s (2h) pairs, launched together this session** — fwdllm is no longer stale.
> **P0-1 (`cohort_sequence` grad-merge order) is now VALIDATED at ≥3600s scale**: exact 1.0 on SET/ORDER/CADENCE/
> VAR for both fwdllm and fwdllm_plus (§G). **P0-2 (sim incremental collect) is VALIDATED for fwdllm_plus**
> (13→2 fails: `overhead_residual`/`per_round_advance`/`throughput`/`total_commits`/`terminal_state`/
> `step_timing_breakdown` all clean) but only PARTIALLY for fwdllm (13→6 fails: counts-at-matched-time pass, but
> the *rate* rungs reopened — new asymmetry, not yet root-caused, see §B fwdllm). fluxtune is flat at 6 fails
> (was 5) but the composition shifted: `trainer_speed_identity` closed, `tb_prepare_perturbation` ROOT-CAUSED
> this session (same GPU-density-artifact class as `eval_model`, ready to exempt), but `agg_step_timing_
> breakdown` **regressed** from its 07-18j-validated PASS and a new `drain_wall_budget` fail appeared — cause
> unclear, needs a repeat data point before re-diagnosing (see §B fluxtune).

**Latest run per baseline** (`run_parity.py`, `lib/python/examples/fwdllm/expt_scripts`):

| baseline | run pair | duration | pass | fail | skip |
|---|---|---|---|---|---|
| fluxtune/syn_0 | `run_20260719_012533`/`_032751` (delay-floor 4.0, divisor 0.48, min-init=N=100, agg_goal=10) | ~7200s | 61 | 6 | 18 |
| fwdllm/syn_0 | `run_20260719_012633`/`_032815` (delay-floor 11.0, divisor 1.63, min-init=N=100, agg_goal=10) | ~7200s | 56 | 6 | 22 |
| fwdllm_plus/syn_0 | `run_20260719_034113`/`_054255` (delay-floor 11.0, divisor 1.63, min-init=N=100, agg_goal=10) | ~7200s | 61 | 2 | 21 |

**Key-rung status** (✓ pass · ✗ fail · – skip; catalog: `async_cifar10/PARITY.md` §F):

| baseline | cohort | vclock | thru | commits | terminal | R1 | V1 | V2 | U3 | S2 | conv | conv_loss |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fluxtune | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ |
| fwdllm | ✓ | ✓ | ✗ | ✓ | ✓ | – | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ |
| fwdllm_plus | ✓ | ✓ | ✓ | ✓ | ✓ | – | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

**All failing rungs, this run:**
- **fluxtune** (6, up from 5 — composition shifted, see §B/§G): `cohort_sequence` SET cascade (unchanged,
  identical divergence point every session — full determinism confirmed). `v2_var_trajectory` grew (0.0334→
  0.0637). `v1b_iters_moving_avg` still fails. `agg_step_timing_breakdown` REOPENED (was validated PASS 07-18j).
  `drain_wall_budget`'s `drain_tail_s` NEW fail. `tb_prepare_perturbation` root-caused, exemption pending (§G).
- **fwdllm** (6, was 13 pre-fix): `cohort_sequence` now PASSES (P0-1 validated, §G). `overhead_residual`/
  `per_round_advance`/`throughput` REOPENED with a new fetch-cost asymmetry vs fwdllm_plus (§B). `step_timing_
  breakdown`/`agg_step_timing_breakdown` unchanged. `v2_var_trajectory` improved but still fails (11.0%→7.19%).
- **fwdllm_plus** (2, was 13 pre-fix): P0-2 validated (§G) — only `drain_wall_budget`'s `drain_tail_s` and
  `agg_step_timing_breakdown`'s aggregator-contention residual remain, see §B.

See §B for what's actively being worked per baseline; see §G for what's already closed.

---

## §B  Next steps / open issues — per baseline, as of the §A runs above

### Priority plan — 2026-07-19 (analysis only; implement next session, in this order)

This session root-caused `tb_prepare_perturbation` to a GPU-density artifact, validated P0-1/P0-2 at ≥3600s
scale, and surfaced two new open items (fwdllm's fetch-cost asymmetry, fluxtune's `agg_step_timing_breakdown`
regression) — no new runs needed for any of it beyond the fresh triple already on disk. Ordered by
(confirmed-bug > cheap-verification > new-instrumentation > infra):

~~**P0 — sim sync barrier can't do incremental (`num_min_req=1`) collection, + grad-merge-order fix**~~ —
**LANDED + VALIDATED at ≥3600s scale, see §G.** P0-1 fully closed. P0-2 fully closed for fwdllm_plus; fwdllm
still has an open, unexplained residual (see fwdllm §B item 1).

**P1 — cheap verifications against telemetry already on disk, no new runs, do alongside/before P0:**
1. **fwdllm's `overhead_residual`/`per_round_advance`/`throughput` reopened at 7200s scale with a NEW
   asymmetry** (07-19): fwdllm and fwdllm_plus show near-identical redundant-fetch abort rates (90.0% vs 89.8%)
   but fwdllm's redundant fetches cost ~2x more per call (real mean 35.5ms vs 18.6ms) — investigate why the
   SAME wasted-fetch mechanism costs more per call for fwdllm specifically (payload/config diff? real-run
   variance? need a second data point to know if this is stable).
2. **Instrumentation LANDED 07-19, not yet run**: `agg_step_timing_breakdown`'s contention-burstiness hypothesis
   (fwdllm + fwdllm_plus) — `_replay_buffered_cohort_contribs` now separately timed (was folded into
   `_process_aggregation_goal_met`); `analyze_agg_step_timing_density.py` buckets any agg-side function's
   duration by concurrent-trainer GPU density. Run both against the next pair to get the actual verdict.
3. **Instrumentation LANDED 07-19, RE-SCOPED by first run**: `analyze_iters_per_data_id.py` shows fwdllm/
   fwdllm_plus EXACT on every shared data_id (P0-1 validated) but fluxtune mismatches on 74/111 (67%) —
   run-wide, not just data_id=0 as first thought. Async-only; root cause still open, this is now the top
   candidate explanation for fluxtune's `cohort_sequence` cascade (needs its own check next).
4. **fluxtune's `agg_step_timing_breakdown` regressed from its 07-18j-validated PASS** (07-19) — same 4
   functions fail again at 53-99% mean_rel on a fresh 2h pair. Cross-baseline GPU contention was checked and
   ruled out (confirmed single-run-per-node); likely machine-load variance between runs. Get a second data
   point before re-diagnosing — don't assume the fix regressed.
5. **`v2_var_trajectory` now checked at ≥3600s scale (07-19)**: fwdllm_plus RESOLVED (0.43% mean_rel, P0-1
   closed it); fwdllm still fails but improved (11.0%→7.19%); fluxtune got WORSE (3.34%→6.37%), contradicting
   last session's "flat" call. **Instrumentation LANDED 07-19**: DEBUG-gated `var_calc` telemetry logs
   per-tensor input grad norms + output var per `_compute_var` call — enable DEBUG on the next pair to localize
   input drift vs the reduction itself.
6. **`drain_wall_budget`'s `drain_tail_s` sub-check fails for fluxtune + fwdllm_plus** (sim 0.55-0.72s vs a
   fixed 0.5s floor budget) — small magnitude. **Instrumentation LANDED 07-19**: split into
   `drain_tail_canonicalize_s`/`drain_tail_replay_s`/`drain_tail_residual_s` on the next pair's telemetry.
7. **NEW, small (07-19)**: `tb_accumulate_grads` (trainer-side) — KS 0.256 vs 0.25 tol, real 2.9ms vs sim
   3.1ms, unmasked once `tb_prepare_perturbation` was exempted. Not yet examined; likely same density-artifact
   class but not verified the same way (no branch data for this func) — don't blind-exempt without checking.

**P2 — landed this session:**
8. ~~`tb_prepare_perturbation` branch-taken + concurrent-trainer-density A/B~~ — **ROOT-CAUSED + EXEMPTED
   07-19** (§G). Also: ~~real-only redundant weight resend~~ — **`suppress_redundant_weights: true` added to all
   6 parity yamls 07-19** (§G); NOT yet validated against a live pair, next run should confirm the
   `redundant_weights_suppressed_total` counter fires and fwdllm's abort rate drops from ~90%.
   **Follow-up (operator, 07-19)**: this should end up as unconditional default behavior, not a flag — a
   trainer that already has the current version_key/model_version's weights should never be re-sent them,
   full stop, as an invariant of the version-tracking logic itself, not an opt-in. Once the counter/abort-rate
   check above confirms it's correct and byte-identical elsewhere, flip `_should_send_full_weights`'s legacy
   branch away (remove `_suppress_redundant_weights` gating, not just default it `True`) — separate change,
   don't do it pre-emptively without that confirmation.

**P3 — infra robustness, not parity-blocking:**
9. Dynamic GPU health filtering — `CUDA_DEVICE_ORDER=PCI_BUS_ID` only fixes *which* physical card a given
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

### fluxtune (~7200s, delay-floor 4.0, divisor 0.48, min-init=N=100, agg_goal=10) — TOP PRIORITY
1. **`cohort_sequence` SET cascade — quantified this session, user hypothesis about its historical cause
   PARTIALLY REFUTED.** Pulled the raw `agg_round` telemetry (both legs, `run_20260718_015337`/`_035554`):
   cycle_index 2 (real `data_id=0,iter=3`) diverges by exactly **4/10 members (60% overlap)** — real cohort
   `{458,461,420,467,387,372,414,456,371,448}` vs sim `{405,434,463,457,387,372,414,456,371,448}`. Operator
   hypothesis was that this traces to a historical period when aggregation was slow + eval was on the critical
   path (both since fixed, §G 07-13/07-16), and that with those fixed, real/sim should now release-and-select
   at the same rate, tightening the candidate set. **Checked directly: doesn't hold as stated.** `aggregate()`'s
   own wall cost is now ~70-140ms/call (§G 07-18g/i) — two orders of magnitude below the observed ~10s
   inter-round cadence (dominated by trainer JVP compute, 8-15s/trainer per the same telemetry), so
   aggregator speed was never the pacing bottleneck at this `agg_round` cadence and the eval/aggregate fixes
   couldn't have meaningfully closed this gap. The real driver is structural: fluxtune's `c=30 ≫ agg_goal=10`
   fedbuff pool has a designed-in surplus/carry-over (§F-17 — NOT a bug, `carried_surplus_commits` is supposed
   to be the majority bucket), and real's round-4 inter-round gap is anomalously fast (4.46s vs a ~10s
   baseline) exactly where the divergence appears — consistent with a surplus-drain burst, but one driven by
   the `c≫agg_goal` pool's inherent queueing dynamics, not by aggregator/eval slowness. **Also newly surfaced**:
   real and sim don't even take the same NUMBER of iterations to clear data_id=0 — real advances to data_id=1
   after iteration 4, sim is still on data_id=0 at iteration 5 — so cadence isn't matched even in aggregate,
   independent of the cycle-2 SET tie. **Next step (P1 item 3 above):** extend this same telemetry-diff to the full run
   (not just the first 5 rounds) to get a run-wide divergence-magnitude trend, and separately investigate the
   iteration-count mismatch at data_id=0 (4 vs 5) — that's a distinct, possibly more tractable, target than the
   SET tie itself for "tightening the case," since it's an aggregate cadence question, not a per-cycle stochastic
   one.
2. `_distribute_weights_async` still exempted (`gates_ok=False`, real-only sleep) — unrelated, unaffected.
3. `sim_sct_ordered_drain` A/B — unblocked. Run `fluxtune_n10_smoke_sim_no_sct_drain.yaml` against next pair.
4. **Accuracy drop after reaching 81%** — known, deferred by operator (07-15). Not yet triaged.
5. **Real↔real admissibility (§F-5)** — rungs now finalized (tie-window + tiered dep graph, `CHECK_META`
   `deps`); unblocked, ready to resume (the `cohort_sequence` admission investigation that deferred it is
   closed, §G 07-17d).
6. **`agg_step_timing_breakdown` REOPENED** — see P1 item 4 above.
7. **`drain_wall_budget`'s `drain_tail_s` NEW fail** — see P1 item 6 above (shared with fwdllm_plus).
8. **`v2_var_trajectory` GREW, not flat** — see P1 item 5 above.
9. **`tb_prepare_perturbation` exemption** — diagnosis closed (§G), action item pending in P2 above.

### fwdllm (`run_20260719_012633`/`_032815`, ~7200s — score-trigger run, see §A)
> `cohort_sequence` now PASSES at scale (exact 1.0, P0-1 confirmed) — no longer an open item.
1. **`overhead_residual`/`per_round_advance`/`throughput` REOPENED at this scale** (07-19) — sim's modeled
   per-round cost runs ~11% over real's (68.36s vs 61.19s intrinsic-second). New finding: fwdllm_plus shows a
   near-identical redundant-fetch abort rate (89.8% vs fwdllm's 90.0%, so P0-2's own mechanism isn't the
   differentiator) but fwdllm's redundant fetches cost ~2x more per call (real mean 35.5ms vs fwdllm_plus's
   18.6ms) — see P1 item 1. Not yet root-caused; get a second data point to check stability before deeper triage.
2. `step_timing_breakdown`/`agg_step_timing_breakdown` — same functions fail as the earlier short pair, now at
   full run-length sample size; unchanged mechanism, rides the shared contention hypothesis (P1 item 2, shared
   with fwdllm_plus).
3. `v2_var_trajectory` still fails but improved (11.0%→7.19% mean_rel) — see P1 item 5, needs its own
   investigation, not fully explained by P0-1.

### fwdllm_plus (`run_20260719_034113`/`_054255`, ~7200s — score-trigger run, see §A)
> P0-2 VALIDATED here: `overhead_residual`/`per_round_advance`/`throughput`/`total_commits`/`terminal_state`/
> `step_timing_breakdown` (incl. the previously-failing `_send_grads`) all PASS clean. 13→2 fails.
1. `agg_step_timing_breakdown` — still a genuine residual gap (`_aggregate_grads_sync`/`_compute_var`/
   `_distribute_weights_sync`/`_prepare_round_state`/`_process_aggregation_goal_met`/`sync_collect_and_
   accumulate_grads`, 53-97% mean_rel). Same functions and same `c=agg_goal=10` zero-slack contention
   hypothesis as fwdllm's own residual (P1 item 2) — not yet measured directly.
2. `drain_wall_budget`'s `drain_tail_s` — new small fail, shared with fluxtune (P1 item 6).

### Cross-baseline / shared

- **P0-1/P0-2 — VALIDATED at ≥3600s scale, see §G 07-18l/n and §G 07-19.** P0-1 (`cohort_sequence` grad-merge
  order) fully closed for both fwdllm and fwdllm_plus. P0-2 (sim incremental collect) fully closed for
  fwdllm_plus; fwdllm has a new, still-open residual (fwdllm §B item 1, P1 item 1) — same redundant-fetch abort
  rate as fwdllm_plus (~90% both) but ~2x the per-call cost, unexplained.

- **`minInitialTrainers=c` (not N) reopened the post-barrier join-order race for ALL THREE baselines —
  ROOT-CAUSED+FIXED, §G 07-17.** Sim's cycle cadence legitimately outruns real's (transport-collapse), so
  identical wall-clock-bound trainer spawn timing lands in different cycles per mode — no algorithmic bug.
  Fixed in the yamls `run_sequential.sh`'s `BASE_YAML_MAP` actually generates from: `fwdllm_n100_smoke[_sim]`,
  `fwdllm_plus_n100_smoke[_sim]`, and (despite the name) `fluxtune_n10_smoke[_sim].yaml` — NOT
  `fluxtune_n100_smoke_4h.yaml`, which the first pass mistakenly targeted and which this pipeline never reads.
  VALIDATED on all three at 5400s scale (see above) — no longer suspect for any remaining gap.
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

> **RULE: closed = here, in ≤30 words, immediately.** The instant a rung flips fail→pass or a hypothesis is
> confirmed/refuted, write ONE terse line below (mechanism + outcome, no narrative) and delete it from §A/§B in
> the same edit. Full reasoning lives in the commit/code comment, not this doc.

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
