# FwdLLM — Real↔Sim Parity

**Scope: real↔sim parity only**, for **fluxtune / fwdllm / fwdllm_plus** (+ the 6 ported fedbuff/felix-lineage
baselines) at 100% availability (syn_0, Phase 1), then unavailability (Phase 2), then beyond syn_0 (Phase 3).
Non-parity content (structural deltas, baseline matrix, roadmap, JVP perf, sim barrier redesign, delay-factor
calibration) lives in [FWDLLM_DESIGN.md](FWDLLM_DESIGN.md). Shared parity methodology (ladder, roles/tiers/
gating, run-length budget) and fwdllm's rung catalog (§F) live in
[async_cifar10/PARITY.md](../async_cifar10/PARITY.md) — read it first if new to this track.

> ## PREAMBLE — how to use this doc
>
> **Fresh session? Read in this order:** §F (locked invariants — don't re-derive) → §E (dead ends — don't
> retry) → §A (current scoreboard) → §B's per-baseline table + "Next session" block → §D (only the lessons
> relevant to the rung you're chasing) → §C if still stuck on ladder-walk mechanics.
>
> | section | contents | update rule |
> |---|---|---|
> | §A | scoreboard: pass/fail per baseline, ≤2-line caption | rewrite in place on every >3600s run |
> | §B | per-baseline open issues + ONE "Next session" block | current-state only; an issue lives here XOR §G, never both |
> | §C | ladder/decomposition method, run-length budget | timeless; edit only if the method itself changes |
> | §D | durable lessons — positive, transferable invariants | ≤30 words each; update in place, never append near-dupes |
> | §E | dead ends — falsified hypotheses, do not retry | append-only, one line each |
> | §F | locked invariants — always-true / always-do | numbers are cited elsewhere; keep stable |
> | §G | closed items, one line each | move here the instant a §A/§B issue resolves, delete the source in the same edit |
>
> **Living doc, not a log — no dated annotations.** Every claim here must read as true right now (or believed
> true until evidence says otherwise), never as "as of <date>". §G is the one intentional exception: a
> recent-fixes ledger ordered newest-first by position, not by date stamp — full history is `git log` on
> this file.
>
> **Non-negotiables:**
> - Correctness per mode first; parity is the consequence, never the goal — for a MATCH or a DIVERGENCE
>   alike. A rung passing because both sides are equally wrong is a regression dressed as a green rung
>   (§D-5). A DIVERGENT rung names two disagreeing sides, never which is at fault — verify each side's own
>   absolute signal before deciding which to change (§D-9).
> - Parity findings/fixes only here — design decisions, roadmap items, calibration derivations belong in
>   FWDLLM_DESIGN.md.
> - Ground every claim in telemetry already on disk before instrumenting or running; fix root causes, not
>   symptoms; always use the `dg_flame` conda env for python/pytest/analyze_run.py; ship new telemetry with
>   its plot + pytest in the same change.
> - Runs happen on a separate operator-controlled node: never launch or babysit one yourself — print the
>   command instead. Assume no run is in flight unless told otherwise, and that each baseline's real/sim
>   pair runs ONE AT A TIME per node (different baselines may run in parallel on different nodes, so
>   cross-baseline directory timestamps can legitimately interleave — only same-baseline real→sim ordering
>   is meaningful).
> - Run artifacts live under `lib/python/examples/fwdllm/experiments/run_<timestamp>_<name>_<syn>_<real|sim>/`.

**Prerequisites:** [async_cifar10/PARITY.md](../async_cifar10/PARITY.md) (methodology + rung catalog §F),
[UNAVAILABILITY_DESIGN.md](../async_cifar10/UNAVAILABILITY_DESIGN.md) (availability substrate),
[FWDLLM_DESIGN.md](FWDLLM_DESIGN.md) (build plan/roadmap/calibration),
[fluxtune_contributions.md](fluxtune_contributions.md) §8 (training-stability/convergence ledger — check
before opening a new stability investigation here).

**Comparator — discovers the latest real/sim pair per baseline and runs the shared parity battery:**
```bash
cd lib/python/examples/fwdllm/expt_scripts
python run_parity.py                        # all baselines, latest pairs, confirm
python run_parity.py --baselines fluxtune   # one baseline
python run_parity.py --yes                  # skip the confirm prompt
python run_parity.py --validate             # + live-run checks (staleness/vclock_now)
```
Rung catalog: PARITY.md §F. **Not redefined there:** per-stage wall-budget instrumentation
(`drain_wall_budget`, `trainer_phase_wall_budget`, `step_timing_breakdown`, `aggregation_compute_wall`) is
ONE-SIDED (`sim<=real`) where sim should collapse a real-transport phase to ~0, DISTRIBUTIONAL where it's
genuine shared compute. Implementation-level reference (tiers, the `pctl_band_ok` band-escape primitive
and its `min_abs` calibration rule, full wall-budget/timing rung table):
`async_cifar10/scripts/parity/PARITY_CHECKER_README.md`.

---

## §A  Score

**Latest run per baseline** (`run_parity.py`; ✓/✗/– = pass/fail/skip; PARITY.md §F). `fluxtune`, `fwdllm`,
`fedbuff_round`, `felix_round` have a 5400s pair; the other 6 remain 3600s+ n=100/c=30 except `fwdllm_plus`
(STALE). Open fails and root-cause analysis: §B.

| baseline | run pair | dur | pass/fail/skip | cohort | vclock | thru | commits | terminal | R1 | V1 | V2 | U3 | S2 | conv | conv_loss |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fluxtune/syn_0 | `run_20260729_080111`/`run_20260730_122621` | 5400s | 61/9/16 | ✓ | ✓ | ✗ | ✗ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| fwdllm/syn_0 | `run_20260729_020102`/`_033249` | 5400s | 59/3/22 | ✓ | ✓ | ✓ | ✓ | ✓ | – | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| fwdllm_plus/syn_0 STALE | `run_20260723_161647`/`_171829` | ~3600s | 61/2/21 | ✓ | ✓ | ✓ | ✓ | ✓ | – | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| felix_it/syn_0 | `run_20260728_000201`/`_010416` | 3600s | 63/6/16 | ✗ | ✓ | ✓ | ✗ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ |
| fwdllm_it_unaware/syn_0 | `run_20260728_011134`/`_021316` | 3600s | 59/4/21 | ✓ | ✓ | ✓ | ✗ | ✗ | – | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| fedbuff_it_unaware/syn_0 | `run_20260728_015339`/`_025553` | 3600s | 67/1/18 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ |
| fwdllm_it_oracular/syn_0 | `run_20260728_022355`/`_032547` | 3600s | 58/5/21 | ✓ | ✓ | ✓ | ✗ | ✗ | – | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| fedbuff_it_oracular/syn_0 | `run_20260728_032938`/`_043152` | 3600s | 64/4/18 | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | ✓ | ✗ | ✓ | ✓ | ✗ | ✓ |
| fedbuff_round/syn_0 | `run_20260729_034223`/`run_20260730_111334` | 5400s | 57/9/21 | ✗ | ✓ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✓ | ✓ | ✓ | ✓ |
| felix_round/syn_0 | `run_20260729_055151`/`run_20260730_115035` | 5400s | 64/6/16 | ✗ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✗ | ✓ | ✓ | ✗ | ✓ |

Per-pair numeric detail: `experiments/_parity_reports/parity_<baseline>_syn_0_<sim-ts>.json`. Open fails: §B.
Net vs the pre-fix legs: fedbuff 62/4→**57/9** (regressed), felix 56/10→**64/6**, fluxtune 60/10→**61/9**.
Both moves are the same root in opposite directions (§B) — **do not read felix's row as progress.**

---

## §B  Next steps / open issues — per baseline

> **RULE: every tracker cell ≤20 words.** State the claim/number, cut qualifiers. If it needs more, it's
> not tracker material — shorten it or point at the code comment/commit.

| baseline | open fails | next step |
|---|---|---|
| `fedbuff_round` (57/9/21) | `throughput`/`overhead_residual`/`total_commits`/`terminal_state`/`v1`/`v1b`/`v2_var_trajectory` — ALL Root C, sim now **11.6% FAST** (was 2.2% slow) · `cohort_sequence` count 14.8% · `step_timing_breakdown` (D-1) | Sign-flipped by the charge fix (§D-14). One root with felix — Root C block below. Do NOT re-tune the charge back |
| `felix_round` (64/6/16) | `v2_var_trajectory` (2.6%, tol 2%) · `cohort_sequence` count 7.0% · `terminal_state` (time 4.7% ✓, trainers 53 vs 57) · `selection_detail`/`preferred_duration` (1-vs-2 lap events, underpowered) · `convergence` (5.06% vs 5% tol, hairline) | throughput family green is **coincidence, not a fix** (Root C). Re-grade after Root C, not before |
| `fluxtune` (61/9/16) | `throughput`/`per_round_advance`/`overhead_residual`/`total_commits`/`terminal_state` — s/cycle 1.134, cadence 1.001 · `drain_wall_budget` (now pure D-1, decoupled from the clock) · `preferred_duration`/`phase_gpu_compute` | Charge fix bought only 2pp of 13 (§D-18 sizing falsified). ~90% of the gap is unattributed — re-split per §D-16 |
| `fwdllm`/`fwdllm_plus` | `drain_wall_budget` (GATING)/`step_timing_breakdown`/`agg_step_timing_breakdown` (DIAG) — pre-existing, D-1 co-location contention | The "charge-the-floor vs relax" call is DECIDED and landed (§G); these are residual measurement-only signals. `throughput`/`terminal_state`/`total_commits` CLOSED (§D-13, §G) |
| `fwdllm_it_unaware`/`fwdllm_it_oracular` | `terminal_state`/`total_commits`, new at 3600s | likely same §D-1 contention family (shares fwdllm's sync dispatch path) — not independently diagnosed |
| `felix_it` | `total_commits`/`terminal_state`/`cohort_sequence`/`convergence` | run §D-16's two-factor split on its own pair FIRST; profile a charge only if `s/cycle` is the failing factor (§D-3, §F-13) |
| `fedbuff_it_oracular` (64/4/18) | `total_commits`/`terminal_state`/`v2_var_trajectory`/`convergence` | same: split before charging. Iteration cadence, so §D-17 does not apply |
| `fedbuff_it_unaware` (67/1/18) | `convergence` only | duration-gated (§C bar 2h+), not a bug |

### Next session

> **Update this block in place on every run — overwrite Part 1/Part 2, never stack a new dated block below.**

**Part 1 — changes landed, and the standing hypothesis.** The round-boundary backfill is landed and
validated: `concurrency_cap` green on all three baselines, self-overlap 553→0, inert on throughput. The
profiled `drain_tail`/`fedavg` charge is landed and validated AS A CHARGE — it lands exactly at the
predicted s/cycle — but it moved a factor a charge should never touch: sim's `cycles/bin` shifted 12-18pp on
both round baselines, sliding sim's cadence off its own real and onto the sibling baseline's real instead
(felix went green by coincidence, fedbuff went red). **Hypothesis (Root C):** the charge feeds
`SimReorderBuffer.pop_min()` (`flame/sim/virtual_clock.py:91`) — sim buffers ~12.6 deep and admits the 10
lowest-`sct` per cycle, resolved against the vclock, so charging less virtual time per cycle changes WHICH
gradients pool, hence `var`, hence cadence. Real has no such buffer (`buf_depth` 0) and no equivalent
feedback path. Already established, don't re-derive: cohort composition is bit-identical real↔sim on the
round-1 pinned cohort; participation entropy matches to 4 digits; iteration-cadence baselines are
unaffected — Root C is specific to `reselect_cadence: round`.

**Part 2 — what to do next, by outcome.** Probe first, no run needed: diff the admitted-vs-buffered set per
cycle across the two sim legs (`fedbuff_round`, `felix_round`). **If the admitted set moved and the delta
concentrates in the re-charged cycles (hypothesis holds):** resolve on sim's own terms whether
`sim_sct_ordered_drain` should select from a reorder buffer at all when real has none (§D-9) — decide that
before touching either side — then confirm with a sim-only pair (`--mode sim --only
fedbuff_round,felix_round`; reals already grade clean). **If the admitted set is unchanged (hypothesis
fails):** Root C is not the buffer — re-decompose rather than widening the probe. Either way: do not re-tune
the charge back toward its old value — that is parity-imitation of a number the mechanism doesn't produce
(§D-21, preamble).

### Other open items

- `fluxtune` `throughput`/`per_round_advance` residual (~12%): cadence is at parity, so Root C isn't its
  story. Charging profiled instead of live bought only 29% of the removed charge back as speed (§D-18) —
  next probe is what bounds the vclock for the other 71% of a cycle (`max(vclock, sct)` on the slowest
  modeled trainer is the lead).
- Accuracy: `sim_charge_registry` is profiled family-wide from `fedbuff_round`/`felix_round` reals only, and
  real `drain_tail` varies 25% across baselines — `fluxtune` is under-charged. `sim_charge_profile_path` is
  already a per-baseline yaml field; generate one profile per baseline with `profile_sim_charges.py`, no
  code change needed.
- `felix_round`'s newly-lapping fails (`selection_detail`, `preferred_duration`, `terminal_state`'s trainer
  count, hairline `convergence`) are underpowered lap-boundary measurement artifacts, not defects.
- `aggregation_compute_wall` (DIAG) fails on all three (`aggregate_fedavg_s` sim mean 0.073-0.090s vs real
  0.059-0.068s) — ungated, pre-existing, logged so it isn't re-discovered as new.
- Do not re-tune `redispatch_turnaround` — its totals check already ruled it out (§D-14), and Root C is not
  a charge-magnitude problem.
- `retask_before_close` is real-UNGRADED (the dispatch tripwire postdates older real legs); sim reads 0.0%,
  not blocking — closes on the next `--mode both` pair.
- `cohort_sequence` `count` is downstream of Root C (it counts cohorts, which scales with how far each side
  got) — re-grade after Root C, don't chase separately.
- Flag promotion: `sim_model_agg_compute_time` defaults ON. `sim_sct_ordered_drain` +
  `sim_model_dispatch_queue` are fluxtune-yaml-only but model general async-transport artifacts — smoke
  fwdllm/fwdllm_plus with both ON, confirm inert-or-better, then promote to code-level default-on.
- Checker invariants I1-I6 were drafted in a prior session and never committed anywhere (checked git log +
  repo-wide grep — genuinely unrecoverable). No longer blocked, but needs operator input on intended
  semantics before drafting fresh ones.
- Confirm the `async_oort`→`AsyncSelectorBase` re-basing (§G) at the INTEGRATION level — unit tests confirm
  the mechanism, not a live comparison:
  ```bash
  cd lib/python/examples/fwdllm/expt_scripts
  bash run_sequential.sh --mode both --only felix_round,felix_it
  ```
- felix (async_cifar10) may share fluxtune's round-1 cold-start gap (`_sim_recv_min`, no fallback for unseen
  ends) — unverified, out of scope here (`async_cifar10/PARITY.md` owns felix).
- felix 46/46 reconfirmation — deferred repeatedly, gates Phase 2.
- Momentum (S1-S3) / server-optimizer retry — roadmap, not parity; resume after Phase-1 closes.
- Accuracy drop after 81% — known, deferred by operator (`fluxtune_contributions.md` §8).
- Sim's in-flight bookkeeping is split across `_sim_pending_commit`/`_sim_inflight_expected`/`_sim_buffer`/
  `_sim_committed`/`selected_ends`/`all_selected` — should be one authoritative per-end state machine.
  Simplify once Root C closes; scope behind `test_fwdllm_sim_grad_loop.py`'s two commit/residence tests,
  never bundled with a correctness fix.
- P3/infra: no automatic GPU skip-and-remap on a broken ordinal (manual `execution.gpu_ids` exclude works).

---

## §C  How we debug here — the ladder, the fwdllm decomposition tree, run-length budget

**Ladder walk** (full method: PARITY.md §1 + rung catalog §F). An FL run is a pipeline —
`clock → availability → selection → dispatch/train → return/order → aggregation → variance-cadence →
utility → emergent`. Parity must hold at every stage; break at stage N and every stage above diverges as a
*consequence, not a bug*. The checker labels the **lowest broken rung with sound (matched) inputs** the
ROOT and demotes higher fails to DOWNSTREAM. Tag every rung a role — CONTROL (input identical), MECHANISM
(one transform modeled — the prize), EMERGENT (aggregate; never fix directly, walk *down*) — and a tier —
INV/EXACT (hard fail), DIST (fail unless `--lenient`), DIAG (informational).

**fwdllm decomposition tree** (which rung fails → where to walk):
- `K2`✗ (throughput) but `K3a`✓ (per-pass advance) → clock is fine, commit COUNT diverged → walk to
  `V1`/`V5` (variance cadence), not the clock.
- `V1`✗ (iterations-per-data_id) but `V2`✓ given matched inputs → the variance *inputs* differ → walk to
  `U5`/`S2` (ordering/selection), not the variance gate.
- `V2`✗ with `V1` inputs matched → a true grad-pool accumulation-order bug.
- `drain_wall_budget`✗ but input byte-sizes identical → co-location contention, not over-compute → §D-1
  (don't tune sim compute).
- `cohort_sequence.composition`✗ but every marginal (S2/utility/count/v1/v2/speed) matches → boundary-race
  cascade = stochastic identity → §D-2 (gate index-identity, keep marginals).
- **Never touch `var_threshold` / `max_iterations_per_data_id`** (§F-3): baseline-defining config, not
  parity levers. A cadence gap is ALWAYS an upstream set/order/clock divergence.

**Run-length budget (fwdllm) — state the min duration up front; never default to 3-4h.** Every run is
operator-launched, so pick the shortest length that exhibits the issue. A run below a rung's duration bar
can still surface a mechanism bug (wrong dispatch pattern, missing guard, RNG desync) worth chasing
immediately — don't wait for a multi-hour run to reveal something a short run already showed. Conversely,
don't grade duration-gated rungs (`terminal_state`, `conv`, cadence distributions) off a short run.

| validating | min run | why |
|---|---|---|
| telemetry field present / instrument sane | 5-10 min | a few hundred commits populate any per-commit field |
| one MECHANISM rung (`drain_wall_budget`, `selection_detail`, `eligibility`) | 45 min | the mechanism fires; per-commit dists stabilize |
| variance-cadence rungs (`V1`/`V2`/`V5`, iterations-per-data_id) | ~90 min | enough committed data_ids for the cadence dist to stabilize |
| throughput / `per_round_advance` (`K2`/`K3`) compounding residual | 3600s (~1h) | round-count-compounding drift needs the data_ids |
| stochastic identity / participation (`cohort_sequence`, `S2`) | 3600s+ | index overlap must reach its independent-draw floor to read as identity-not-bias (§D-2) |
| convergence sign-off (`terminal_state`, `conv`, `conv_loss`) | full 2h+ | terminal-state + curve parity only |

Smoke (5-10 min) before any multi-hour run. One mechanism per run when a fix could perturb another baseline.

---

## §D  Durable lessons — fwdllm diagnostic patterns

> Positive, transferable invariants — things learned that must be followed. ≤30 words each; update in
> place, never append near-duplicates. A falsified hypothesis belongs in §E, not here. Shared (non-fwdllm)
> patterns live in [async_cifar10/PARITY.md](../async_cifar10/PARITY.md) "Durable lessons."

**D-1.** A shared-compute wall rung failing with byte-identical inputs is co-location contention, not sim
over-compute. Fix by charging the vclock, never by tuning sim's compute.

**D-2.** A boundary-race cascade on a stochastic-async selector is core-IDENTITY, not skew. Gate
index-identity to diagnostic; keep count/marginals enforced.

**D-3.** Porting a baseline's selector class does not port its real↔sim timing parity — aggregator/trainer
classes are separate. Diff the destination against the shared base first.

**D-4.** Grade parity on the logical work budget N, never a matched virtual-time window — normalizing along
the axis under test is circular.

**D-5.** A green parity rung is not a correctness claim — common-mode bugs pass differential tests. Always
pair with an absolute, mode-independent sanity check.

**D-6.** A selector's parity record belongs to the SELECTOR+AGGREGATOR pair, not the selector alone. Verify
which side owns each guard before citing it as a reference.

**D-7.** A statistic computed over rate-scaled samples measures the rate, not the samples. Check both
sides' input to a shared formula before suspecting the formula itself.

**D-8.** A cache/reuse fast path can silently skip a guard the slow path enforces. Diff its return against
what the bypassed call would have filtered.

**D-9.** A parity DIVERGENCE names two disagreeing sides, never which is wrong. Find each side's own
absolute, same-side self-consistency signal before choosing which to change.

**D-10.** `reselect_cadence` changes which candidate POOL fills a freed dispatch slot, not whether a
just-committed trainer personally waits — the version_key re-pick guard is cadence-agnostic.

**D-11.** A per-round wall residual can be a per-cycle cost compounded many times over. Count actual
cycles-per-progress-unit before assuming a small measured gap can't explain a bigger one.

**D-12.** A telemetry span measured from a shared batch-start timestamp is cumulative across the whole
batch. Pool by first-difference between sorted events, never a flat mean.

**D-13.** A run truncated by a wall-clock deadline can emit one event for progress it never finished.
Require verified-completion evidence per progress-axis key, not event presence.

**D-14.** After fixing a proven overcharge, re-verify the residual's sign and mechanism, not just that the
old bug is gone — a flipped sign means re-decompose, not re-tune.

**D-15.** When sim runs faster than real with no charge category to explain it, suspect a
concurrency/scheduling policy divergence. Measure per-trainer idle time, not server wall.

**D-15a.** Anchor "what was sent when" on `processing_wall_ts` (arrival), not `dispatch_version_key` (read
at close) or a cycle's own close timestamp — both mis-date a mid-cycle send.

**D-16.** Factor a per-round residual into `(s/cycle) × (cycles/round)` before naming a mechanism — a
shared symptom across baselines can still have disjoint roots.

**D-17.** An event-count rung is only meaningful over matched WORK, not the whole run. Compare a divergent
event's progress key against the matched budget, not wall time.

**D-18.** A charge ledger prices the CHARGE, not the residual it removes. Never size a throughput
prediction off the charge delta — confirm the clock is actually charge-limited first.

**D-19.** Never normalize a per-bin statistic by distinct `data_id` — it wraps. Key on
`(round, cycle_data_id)`, count only completed visits, truncate to the matched budget.

**D-20.** Grade an invariant on its own quantity and axis, never a proxy sampled at the wrong instant. Check
for an existing per-entity span before instrumenting a new one.

**D-21.** A vclock charge is not a neutral accounting term — if sim selects work against the clock, a
charge change can shift sim's selections and cadence too.

---

## §E  Dead ends — do NOT retry

> Falsified hypotheses, one line each, append-only. A dead end never un-dies; re-listing one wastes a
> session. Landed-but-inert cleanups belong in §G, not here.

- **sct-order-membership lever** (admit the lowest-sct 10 instead of first-arrived) — REJECTED: the
  aggregator would BLOCK on future arrivals / hold slots for possibly-offline trainers → FIFO-violating,
  DEADLOCKS under Phase-2 unavailability; and the divergence is a stochastic tie-break, not a chargeable
  mechanism (§D-2).
- **recv_fifo→drain_ready as the async fluxtune D-skew fix** — LANDED but INERT: dropped 181k "already has
  active task" log lines but the D-skew (49.5/56.8) and per-cohort wall (4.02 vs 3.70) were unchanged. Kept
  as cleanup (flag `real_drain_ready_ingest` ON), NOT a parity fix; the recv_fifo streamer was not the
  mechanism.
- **`real_distribute_settle_s = 0.0` as the fluxtune parity cause** — NOT the cause: real ran clean at 0.0
  (droppable dead weight) but the skew is a stochastic tie-break; don't expect dropping it to move fluxtune.
- **`_compute_var` stop-the-world GC pause** — REFUTED: `gc_pause_s` telemetry shows ~0ms GC both sides.
- **`_flat_grad_norm` as the drain-wall contention amplifier** — landed (bit-identical, correct
  optimization) but INSUFFICIENT: p90 `drain_tail_s` rel unchanged (0.94/0.92); not the dominant contention
  source (§D-1).
- **GPU/resource contention blamed at n=10** — REFUTED once; held below ~100 trainers. (At n=100 contention
  IS a root — §D-1; this dead end is scale-bounded.)
- **`matched_virtual_budget` (V = min(vclock, wall))** — DELETED, don't reintroduce: conflates the two
  clocks (the axis `sim_rate` tests), masks throughput + fails to grade (§D-4).
- **"surplus idle" as the post-§D-15 residual (felix/fluxtune)** — REFUTED: the `c/(busy+idle)` identity is
  MATCHED on all three post-fix legs (felix 2.93 vs 2.87 s/cycle, fluxtune 9.59 vs 9.73). The residual is
  `cycles/bin` (felix) and `s/cycle` charges (fluxtune) — §D-16. Don't re-open the idle term.
- **`reselect_cadence` pool-size theory as the round-cadence throughput driver** — SUPERSEDED: the per-cycle
  charge-compounding root (§D-11) explains the gap. Pool-size may still matter for `slot_starvation`-style
  idling, just not this gap.
- **`felix_round`/`fluxtune` sharing one "overshoot root"** — REFUTED: disjoint factors, §D-16.
- **the round-cadence cohort pin as a defect ("never rotates / 70 of 100 never train")** — NOT a defect,
  operator ruling: `reselect_cadence: round` pins for a whole round by design and `rounds: 50` would rotate
  it 50×; a 5400s run just covers ~1 round. Don't re-key it on `_model_version` (§D-17).
- **cohort COMPOSITION as `felix_round`'s `v2_var_trajectory` driver** — REFUTED: both modes commit from the
  same frozen 30 for 89% of the run with the gap fully present; widening real's set moves `var` the wrong
  way (§D-2). The trajectory diverges, the contributor set doesn't.
- **"67% of `fluxtune`'s throughput residual is charged contention" (§D-18 sizing)** — REFUTED by the
  validation leg: 0.320 s/cycle of charge removed bought 0.092 s/cycle, 13.2%→11.9% not →7%. The charge fix
  is correct and stays; its predicted magnitude was wrong. Don't re-price a charge without checking the
  clock is charge-limited.
- **Root A′ as a `felix_round`-specific cadence defect ("chase what changes at bin ~30")** — REFUTED: the
  entire +14.7% vanished on a charge change that never touched felix's code path. It was sim's
  charge-coupled cadence level (Root C), not a felix mechanism. Don't reopen the bin-30 onset.
- **`fedbuff_round`'s 553 same-end dispatches / the round-boundary backfill as a throughput term** — FIXED
  and confirmed inert on throughput: cap green, self-overlap 0, dispatches/cycle unchanged at 10.02. It was
  a real INV breach worth fixing; it moved no timing rung.
- **cohort composition as a real↔sim divergence on the round baselines** — the round-1 pinned 30 is
  BIT-IDENTICAL real↔sim on both (30/30), and participation entropy matches to 4 digits across all six legs.
  Both modes run the same trainers on the same shards; stop looking for a set difference (§D-2).
- **`aggregate_grad_pool()` summation order as the `v2_var_trajectory` driver** — REFUTED: it's an
  element-wise SUM over ≤`max_iterations_per_data_id` items (float-reorder effect ~1e-6 relative, not the
  observed 3%) feeding the outgoing `GRAD_POOL` payload, not the variance gate. The gate is
  `calculate_var()`'s split-half over `grad_for_var_check_list`.

---

## §F  Locked invariants (from async_cifar10, carried over)

> Always-true / always-do rules. Numbers are cited across this doc — keep them stable, don't renumber.
> Diagnostic *patterns* (see X → means Y) live in §D, not here.

1. **Sim does real forward-grad compute, charges modeled time.** Agg stamps
   `sct = sim_send_ts + max(real_gpu_s, D) + leg`. Never put overhead on the vclock (`vclock = max(vclock, sct)`).
2. **Progress axis is `data_id`; identity/caching axis is `model_version`.** `data_id` wraps every
   `total_data_bins` lap — never key a cache/identity on it, use monotone `model_version` (§F-21). `_round`
   (the lap counter) equals `model_version` only in regular FL; don't carry a round-keyed construct over
   without re-deriving the axis.
3. **Variance is an emergent gate; localize, never tune it.** `var_threshold`/`max_iterations_per_data_id`
   are baseline-defining knobs, not parity levers.
4. **Slot residence must survive variance rollbacks.** Release keys on the agg-goal boundary; the sct
   reorder buffer must not strand a grad across a rollback.
5. **Real is the reference only after admissibility.** Check whether real is the divergent side before
   tuning sim.
6. **Fix the concept, not the symptom.** Classify a mechanism as real-transport artifact (`and not
   self.simulated`) vs algorithmic property; scope-check shared code first — `top_aggregator.py`/
   `_sim_recv_min` can silently break async_cifar10.
7. **Match pytest scope to blast radius.** fwdllm-only → `pytest tests/mode -k fwdllm`; shared parity
   engine → add async_cifar10 tests too; shared stack → full `pytest tests/`.
8. **Telemetry-first, then instrument, then (rarely) run.** Validate/refute from telemetry already on disk
   before running anything. Ship telemetry + plot + pytest together with any new mechanism.
9. **Consult PARITY.md's vclock rules before any sim-clock change.** Clock is a monotone `max`; sim skips
   real waits and reconstructs order from sct (`SimReorderBuffer`).
10. **Sim MUST produce speedup: `sim_rate = vclock/wall ≥ 1`.** `< 1` means sim is stalling on a wait it
    should skip, or its commit throughput can't keep pace with arrivals.
11. **Correctness before speed; shared roots before per-baseline.** A bug failing rungs across ≥2
    baselines outranks a single-baseline one.
12. **Logical determinism is the parity definition.** Same trainers selected, same receipt order, same
    aggregations/rollbacks — differing ONLY in wall-clock. Prove it on the first data bin first.
13. **Do the right thing — no hacks.** A hack that moves a number without a correct mechanism is a
    regression in disguise. When unsure, stop and ask.
14. **`version_key` is the ONLY version-identity vocabulary.** 2-tuple: base `(round, 0)`, fwdllm
    `(model_version, iteration_per_data_id)`. No bare-scalar shortcut.
15. **Verify claims against code, not comments/docstrings.** A docstring claiming two functions are
    equivalent is a statement of intent, not a guarantee — diff them.
16. **Contention at scale → §D-1.** Refuted below ~100 trainers (§E); genuine root at n=100.
17. **A rotating cohort settling at `c − agg_goal` surplus is the correct steady state** for `c ≫ agg_goal`
    fedbuff — don't drive `carried_surplus_commits` toward 0.
18. **Any important knob is logged CONSISTENTLY everywhere, or it's a trap.** A correctness-path value
    (seed, delay floor, agg_goal, c, trace, flag) must match across yaml, snapshot, and both roles'
    telemetry — divergent logging wastes sessions chasing phantoms.
19. **No compute on the critical path for a log the run doesn't need.** Gate any log with non-trivial
    args (`.item()`, hashing, `.norm()`) behind `logger.isEnabledFor(logging.DEBUG)` — an f-string
    evaluates its args even when the level would drop the line.
20. **Real/sim timing disagreement → fix real toward determinism, never inject noise into sim.** Sim's
    per-speed-class duration must stay clean (what makes `cohort_sequence` checkable). Fix real's
    measured completion time at the source.

### §F.1 Version & commit invariants (confirmed in code, both modes)

21. **`model_version` bumps once per COMPLETED data-bin** (variance PASS, `+= 1` at the data_id advance) —
    constant across one data-bin's iterations. `iteration_per_data_id` bumps on variance-FAIL retry, resets
    on data-bin advance. `version_key = (model_version, iteration_per_data_id)` changes every iteration —
    the sole step identity (§F-14).
22. **Commit == the update used for aggregation, at that instant — no lag.** Real: on ordered arrival. Sim:
    when vclock reaches the update's `sct` (buffer-unlock IS the commit). Never commit on a later event.
23. **Commit frees the compute slot immediately, but a version_key re-pick guard keeps the trainer
    un-pickable for the SAME `(model_version, iteration)`** until the version_key advances. Sim's slot-hold
    (`_sim_pending_commit`) must clear at commit and never re-add after, or re-dispatch starves across
    variance-retry iterations.
24. **Within a data-bin the global weights are constant; a re-picked trainer gets a RETRY, not a re-send.**
    Full WEIGHTS go out only for a `model_version` not yet received this data-bin
    (`_weights_sent_this_cycle`, cleared on the bump); a same-`model_version` re-pick gets VAR=bad, never a
    redundant weight re-send.
25. **One instruction per version_key: never dispatch to a trainer with an unresolved outstanding message
    for the CURRENT `version_key`.** Busy = silence, not a second message, until it returns or the
    version_key advances. Enforced by `_already_served_current_instruction`/`_mark_instruction_served`
    (end_id → last-served version_key). SYNC distribute always had this guard; ASYNC didn't (`fedbuff_round`'s
    `r1_inflight_overlap` was ~90% `VAR=bad` re-sends to an already-busy trainer — §G). Any new distribute
    call site must call both.
26. **Reuse the existing construct; don't duplicate per baseline.** New per-trainer/version state almost
    certainly needs an EXISTING mechanism (`_end_served_version_key`, `_keyed_topk`/`_keyed_draw`,
    `AsyncSelectorBase`), not a new one parallel to it — duplicate logic is duplicate bug surface. Tests
    too: extend a contract suite (`test_async_selector_base.py`, `test_selector_contract.py`,
    `test_selection_determinism.py`) before writing a bespoke one. Tell: about to add a variable/method/
    test whose name rhymes with an existing one (`_keyed_weighted_topk` next to `_keyed_topk`) — check
    whether the existing one should just take a parameter instead (already done: both now share one
    `_keyed_draw` primitive, §G).

### §F.2 Porting a SELECTOR ≠ porting TIMING parity → §D-3

Moved to §D-3 (it's a diagnostic pattern, not an invariant). Kept here as a stub because prior sessions cite
"§F.2" — the class-hierarchy detail and the "diff destination aggregator/trainer against the shared base"
rule now live in §D-3.

---

## §G  Landed fixes — recent, load-bearing for current work only. Full history: `git log -- lib/python/examples/fwdllm/simulate_fwdllm.md`.

> **RULE: closed = here, immediately.** The instant a rung flips or a hypothesis resolves, write ONE line
> (mechanism + outcome) and delete it from §A/§B in the same edit. Newest first.

- **Round-boundary BACKFILL validated** — `fedbuff_round` sim peak 35→30/30, self-overlap 553→0,
  `[ConcurrencyBackfill]` 29×, 0 `slot_starvation`; felix/fluxtune stay green. `concurrency_cap` CLOSED on
  all three. Inert on throughput (dispatches/cycle 10.02 unchanged).
- **Profiled `drain_tail`/`fedavg` charge validated as a charge, falsified as a sizing** — charge landed
  exactly (0.663→0.343 s/cycle, `charge_source: profiled`) but fluxtune throughput only 13.2%→11.9% (29%
  pass-through, §D-18), and it moved variance cadence 12-18pp on both round baselines — Root C, §B/§D-21.
- **`drain_tail`/`fedavg` moved from LIVE-span to REAL-PROFILED vclock charge (§D-18)** — sim's own span is
  contention-inflated 1.77x at n=100; charging it violated §F-1/§F-20. 2 tests.
- **Round-boundary BACKFILL landed** — `_cap_dispatch_to_concurrency()` + `_exclude_pending_commit` now on
  the cohort accumulate branch, not just the reuse branch. 5 tests.
- **Root A′ (`felix_round` +14.7% cycles/bin) CLOSED as a misattribution** — it was never felix-specific;
  the whole residual moved on a charge change that never touched felix's path. Absorbed into Root C (§B).
  felix throughput family now green, but for the wrong reason — do not bank it.
- **`concurrency_cap` re-based onto peak DISTINCT in-flight ends from `contributor_intervals` (§D-20)** —
  grades REAL on every baseline with no fresh run; `fluxtune`'s 31/30 was a phantom (1218 false positives).
  Same-end concurrent dispatches now graded separately (§F-25). 8 tests.
- **`_outstanding_dispatch_count()` cap arithmetic subtracts `_sim_committed`** — a committed end whose slot
  the boundary still holds is not in flight; the pending set is unchanged.
- **`selection_detail` windowed to matched WORK + event COUNT now graded** — was pooling each side's whole
  run, so it graded which side lapped. Green on all three; full-run counts still reported. 5 tests.
- **`v1_iter_per_data_id`/`v1b` normalizer FIXED (§D-19)** — keyed per data-bin VISIT `(round,
  cycle_data_id)`, completed visits only, truncated to the matched budget. Ends felix_round's false 4.4%
  PASS (true 14.7%); fedbuff `v1b` green. 4 tests. 1092 `pytest lib/python/tests` pass.
- **§D-15 fully validated, "overshoot" reclassified** — `busy+idle` matched on all three; the felix/fluxtune
  residual is two unrelated roots (§D-16/17/18), not an over-corrected barrier.
- **`selection_detail`'s "inverted granularity" closed as a non-finding** — lap-boundary crossing, not two
  selection policies. felix = checker windowing (21/22 events out-of-budget); fedbuff = duration-gated coin
  flip. Cohort pinning is CORRECT (operator ruling, §D-17).
- **`felix_round`'s "why does composition diverge when selection is deterministic" ANSWERED** — selection is
  deterministic; one side crossed the round boundary inside the graded window, the other didn't (§D-17).
- **`fedbuff_round`'s throughput family CLOSED by the §D-15 boundary-release fix** — validation pair
  `_034223`/`_220025`: `throughput`/`total_commits`/`terminal_state`/`per_round_advance`/`overhead_residual`/
  `cohort_sequence` all flip green, residual 14.0%→2.2%.
- **Mid-cycle redispatch ELIMINATED in sim, all three round-cadence baselines** (§D-15) —
  `retask_before_close` 78-90%→0.0%. Root confirmed; magnitude correct on all three.
- **`redispatch_turnaround.weights` cumulative-batch overcharge FIXED** (§D-12, 0.488→0.0598s, 7.3-8.3x) and
  RULED OUT as the throughput residual's source (§D-14 totals check). Value is now settled — do not re-tune
  (§F-13).
- **`fluxtune` sim yaml was missing `sim_charge_profile_path`** — every `redispatch_turnaround` row read
  `charge_source: "none"`, ~180s uncharged. Added; `convergence`/`selector_score`/`utility` green.
- **fwdllm's `throughput`/`terminal_state`/`total_commits` 5400s fails ROOT-CAUSED as a CHECKER bug, FIXED +
  VALIDATED** (§D-13). `_matched_logical_budget`/`_per_progress_last_event` treated an uncommitted trailing
  `data_id` (real hit `max_runtime_s` one variance-check attempt into it) as reached progress, comparing
  real's ~22s partial cycle against sim's ~205s completed one. Fixed to require `var_good_enough == True`
  evidence per `data_id`-axis key. Re-run: 56/6/22 → 59/3/22, only the pre-existing D-1 contention group
  remains. 1392 `pytest lib/python/tests` pass. CLOSED.
- **`redispatch_decomp`/`vclock_charge` dark-data gap closed** — `analyze_run.py` gained
  `redispatch_decomp_plots`/`vclock_charge_plots` (CDFs by mode/payload_kind + a `post_close_overhead` mean
  bar + a "real span vs sim charged" uncharged-gap bar), wired into `_PLOT_GROUPS`. Verified against live
  `fedbuff_round` real+sim telemetry (1.9M records) + 5 new pytest cases
  (`tests/analysis/test_redispatch_vclock_charge_plots.py`). CLOSED.
- **`redispatch_turnaround.weights` charge fix VALIDATED live on `fedbuff_round`/`felix_round`, 3600s.**
  `throughput`/`per_round_advance`/`overhead_residual`/`total_commits`/`terminal_state` all flip fail→pass,
  residual 1.5-4.8% (well under 10% tol) — the round-cadence dispatch-path root closed with no meaningful
  overshoot. CLOSED.
- **`async_oort.py` re-based onto `AsyncSelectorBase`** (2193→897 lines) — was the last selector carrying
  its own full copy of the send/recv concurrency mechanism, drifted from fixes landed only on the base.
  Utility-scoring POLICY (pacer, exploration split, `select_type` strategies) kept unchanged, routed through
  shared `_choose`/`_pre_choose` hooks; eval-task state and the `_cleanup_removed_ends` ghost-sweep override
  kept as genuine Oort-only behavior, not assumed away. `slot_starvation` telemetry promoted to the shared
  base; `_keyed_draw` extracted as a shared primitive. Full suite 1387/1392 pass (net +5 tests), 0
  regressions. Integration-level real+sim confirmation still open (§B).
- **General profiled-charge mechanism landed (FWDLLM_DESIGN.md §P).** `sim_charge_registry.yaml` +
  `get_profiled_charge_s()` loader + `charge_sim_vclock_overhead(..., profiled_s=...)` — a registry entry's
  `charge: true` flips a category on, independent of `sim_model_agg_compute_time`. `redispatch_turnaround.weights`
  seeded `charge: true, mean_s: 0.4365` (pooled n=3264) from a `run_20260728_151722`/`_151831` real pair;
  `.var_bad` left `charge: false` (measured negligible, ~2x not ~13x). `vclock_charge` ledger gained a
  `charge_source` field (`live`/`profiled`/`none`). 8 new tests.
- **`redispatch_decomp` widened to cover `VAR=bad` dispatches (not just fresh `weights` sends) + new
  `vclock_charge` ledger event added.** `_pk in ("weights", "var_bad")` gate + `payload_kind` field
  (`fwdllm_aggregator.py:4135`); `charge_sim_vclock_overhead` (the ONE shared fold-onto-vclock function, used
  by every fwdllm-family baseline) now emits `vclock_charge` on every call in both modes, plus a
  `charge: bool` param so a candidate category can be measured before being charged. Root-cause
  reconciliation (§D-11) found the round-cadence throughput gap is a per-cycle cost compounding ~10x — the
  old telemetry only measured the 1-of-10 terminal cycle. 4 new tests, 573 `-k "fwdllm or telemetry or
  parity"` pass.
- **`felix_round` sim + `fluxtune` (both sides) missing-run episode resolved by relaunch, not a code fix.**
  Both pairs re-ran clean (`run_20260728_102750`/`_113005`, `_102831`/`_113046`); the `DIRTY_ABORT`
  hypothesis was never confirmed on the run node and is now moot for this batch.
- **`selection_detail` CONFIRMED at production scale** — `_exclude_pending_commit` passes on both
  `fedbuff_round`/`felix_round` at n=100/c=30/3600s, not just n15. CLOSED.
- **`fedbuff_it_unaware` 6→1 fails confirmed duration-gated, not a bug** — running the same config 3600s
  instead of 1800s alone resolved 5 of 6 fails; no code change needed.
- **`run_sequential.sh`'s `fedbuff_round`/`felix_round` yaml mapping briefly mis-set to the n15 debug-scale
  files, then reverted.** The "n10"-named files are actually production n=100 scale (misleading filename
  only); n15 is a deliberate reduced-scale repro, not a substitute. `ALL_RUNS` now correct.
- **`fedbuff_it_oracular` 12→4 fails confirmed at scale** — `get_curr_unavail_trainers`/
  `get_curr_task_ineligible_trainers` DEBUG-gating fix (below) held at n=100/3600s; remaining 4 fails tracked
  in §B (profile-own-charge group).
- **`slot_starvation` telemetry landed** — surfaces a freed dispatch slot with no eligible candidate
  (`feasible_extra < extra`), shared by every async baseline.
- **All 9 baselines' smoke yamls bumped 1800s→3600s** + real/sim wall-ceiling watchdogs, matching the §C
  run-length bar (enabled the full-scale batch, §A).
- **`[LAG_DECOMP]` + `redispatch_decomp` telemetry landed for fwdllm** — splits redispatch wall gap into
  peer-wait vs post-close overhead, ported from asyncfl's shared version.
- **2 more stale "sim uses in-mem cache" doc claims found + fixed** (`checks.py` docstring, PARITY.md,
  PARITY_CHECKER_README.md) — no cache exists in either mode; real reason is dispatch-cadence /
  aggregator-side overhead.
- **`staleness` (U3) root-caused, FIXED, VALIDATED — CLOSED** (§D-9). Real's `_agg_pending_commit_ref` only
  covered return-time state, missing a still-training end and flooding it with re-dispatches (one end: 526
  sent vs 284 processed). Fixed via `_PendingCommitUnion` (dispatch-time + return-time, mirroring sim's
  `_sim_pending_commit` span). Confirmed on the post-fix pair: real 0.056/0.064 vs sim 0.095/0.12 (was
  real≫sim 7.62/9.10 pre-fix). Unmasked a separate `throughput`/`per_round_advance` gap, not a regression
  from this fix.
- **`[RecvBootstrap]` deadlock FIXED + VALIDATED.** A bootstrap added for a trainer-side crash also raced
  the aggregator's own first SEND tick, phantom-filling every dispatch slot and permanently freezing sim's
  vclock at 0.0 (reproduced at n=15 with zero GPU sharing possible — also falsified the GPU-contention
  hypothesis for the same eviction). Fixed by gating the bootstrap behind `allow_recv_bootstrap`, set only by
  single-parent callers (`channel.one_end()`); `channel.ends()` dispatchers default to no-bootstrap. Re-run
  confirms no repeat of the freeze.
- **`_exclude_pending_commit` FIXED + VALIDATED** (§D-8). Round-cadence's cohort-reuse cache-hit dispatch
  path bypassed the busy-exclusion guard entirely (called once per run, not per tick). Fixed by filtering
  the cache-hit list every tick. Flips `r1_inflight_overlap`/`selection_detail`/`participation`/
  `training_budget`/`overhead_residual`/`per_round_advance` fail→pass. NOT also a `staleness` fix — that was
  a separate, later-fixed bug (above).
- **Variance-check pool stopped rate-scaling stale contributions** (§D-7) — was scaling round-cadence's
  genuinely-stale carried-surplus entries toward zero before the variance gate, faking convergence (real
  committed data_ids in ~1.7 JVP samples vs fluxtune's ~16). Fixed: stop scaling the var-check pool by
  `rate`; model-update merge untouched. Validated for cadence effects; also exposed a previously-masked
  `fluxtune` compounding effect (§B).
- **R-D landed + VALIDATED** — async distribute never had the sync path's one-instruction-per-`version_key`
  guard (§F-25); `fedbuff_round`'s `r1_inflight_overlap` was ~90% `VAR=bad` re-sends to already-busy
  trainers. Fixed via `_already_served_current_instruction`. Confirmed: `r1_inflight_overlap` real 0.0%/sim
  0.0% both baselines (was ~90-91%); the flood's `phase_gpu_compute` contention side-effect also resolved
  (13.5s→4.8s fedbuff, 7.9s→3.6s felix).
- **`sample_by_util` reproducibility fix landed + VALIDATED** — `np.random.choice(p=...)` was pool/order-
  dependent; replaced with `_keyed_weighted_topk` (Efraimidis-Spirakis keys). Confirmed on `felix_it`:
  `cohort_sequence`/`v1b_iters_moving_avg`/`utility` all flip fail→pass.
- **`get_curr_unavail_trainers`/`get_curr_task_ineligible_trainers` INFO-logged every call, ungated** — a
  300-entry trace scan + log fired every iteration-cadence tick in ORACULAR mode (4735×/1243s run); gated
  behind DEBUG (§F-19). Confirmed effective at scale (`fedbuff_it_oracular` 12→4 fails, above).
- **R-C landed** — round-cache stuck-timeout now clocked on vclock in sim / wall in real
  (`_round_cache_clock_now`); sim evicted 0 stuck ends vs real's 6.
- **R-A landed** — `FedBuffSelector` re-based onto shared `AsyncSelectorBase` (832→54 lines), inheriting the
  version_key re-pick guard, R1 guard, vclock timeout, avl filter, full drain, `_keyed_topk` sampling. Real
  had been re-picking the same-version trainer on 34.4% of commits vs sim's 0.9% pre-fix.
- **R-B landed** — pinned round cohort was sized/checked against `agg_goal` instead of `c` (under-fill:
  `fedbuff_round` froze at 10/100 with 20 idle slots; over-fill: `felix_round` 30 real vs 40 sim). Fixed:
  `_round_cohort_target`/`_trim_round_cohort` target exactly `c`.
- **`reselect_cadence` knob added** (round/data_bin/iteration); `AsyncRandomSelector` collapsed 850→31 lines
  (zero methods of its own); selector stats de-duplicated ×4 into `AbstractSelector`; dead heartbeat
  mechanism deleted (no sender ever existed).
- **Async-baseline decisions settled** — extraction reference is `async_oort.py` (not `async_random.py`,
  collapsed to a stub); cohort target is `c`, trimmed exactly (`agg_goal` is only the aggregation trigger);
  `reselect_cadence` is a first-class knob.
- **fluxtune 3→0 fails (69/0/16)** — the 3 remaining fails were ONE boundary-race cascade on a
  stochastic-async selector, not a sim bug (§D-2); checker now gates index-identity for stochastic-async
  selectors, keeps marginals enforced.
- **fwdllm timing family root-caused** — co-location contention (byte-identical inputs, sim's per-op floor
  matches real's typical every decile), not sim over-compute (§D-1). `_flat_grad_norm` fix landed but
  insufficient alone (§E).
- **fwdllm/fwdllm_plus `throughput` CLOSED at 7200s** — `recv_fifo`→`drain_ready` + var_bad dedup held
  (3.2%/4.8%, both PASS).
- **Checker overhaul** — `matched_virtual_budget` deleted, graded on logical budget N instead (§D-4);
  `pctl_band_ok` DIST-band escape added; `_step_timing_compare`'s `band_min_abs_s` floor fixed (was masking
  5x regressions at ms-scale, copied from a 1s-scale metric).
- **Early foundation** (compressed — full detail via `git log` on this file): P0-1 deferred-merge buffering
  landed; fluxtune's sim-side vclock/pacer/EOT-stamping bugs fixed and real's `_agg_pending_commit_ref`
  bound, taking fluxtune 19→5 fails at 7200s; startup GPU-health crashes fixed; `cohort_sequence` grading
  made distributional; `[SELECT_TRACE]` debug logging added then removed once the divergence was localized.
