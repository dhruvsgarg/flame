# Felix insights — data-backed mechanisms behind time-to-accuracy and communication

Read [ROBUST_FL_READINESS.md](ROBUST_FL_READINESS.md) first; its doc rules apply here. **Purpose (operator 10-12):** turn end-to-end
numbers (time-to-accuracy, communication) into arguments backed by telemetry: *why* Felix wins or loses against each baseline, with
a formal model where one exists and a measured figure for every claim. **Status:** §4 instrumentation ✅ (10-12); metrics in `scripts/insights.py`; findings grow per run (§5). Iterative: each run adds evidence; a claim lives here only with its metric, source and evidence.

**Entry format.** `ID · claim · metric (formula) · source (event.field) · baselines · figure · evidence (run dir) · state`.
States: `hyp` (stated, unmeasured) · `measured` (number + run) · `proved` (bound derived + measured inside it) · `refuted`.
An insight counts only if real and sim agree on it (same metric on a parity pair, PARITY C6); sim-only figures say so.

## Next steps (top item is next)

- **IN1 · Verify the model against code · todo (no runs).** Confirm §1 notation line by line: eval span `s_E` (trainer `task_recv →
  task_send` for `task=eval`), eval budget reset rule (`curr_round_eval_slots_left` on train commit only, `async_oort._cleanup_recvd_ends`),
  cooling slots, and where `PROP_LAST_EVAL_ROUND` is first set (the eval gate needs it non-null). *Exit:* every §1 symbol cites a code line.
- **IN2 · Run `insights.py` on every PR31 pool · todo (after PR31).** I-A2/A3/A5, B1, C1, D1, E1, F1, G1 today; add I-A1 (pick share by
  speed quartile), I-A4, I-C2 (sync barrier wait), I-G2 next. *Exit:* one JSON + figure per row, real↔sim agreement per metric.
- **IN4 · Felix train/eval interference on long runs · blocked: a long Felix leg (PR21 G1A / G1U).** Short screens cannot show it
  (§1 evidence). *Exit:* I-F1..F4 measured on cifar + speech, syn_0 and mobiperf, real and sim.

## 1. Model — Felix's train and eval selectors on one slot pool

One aggregator thread runs `put_train → put_eval → get_weights` per inner-loop pass (`asyncfl/top_aggregator.compose`); the two
selector hands share one in-flight set (`async_base.selected_ends`), so they run concurrently only as tasks in flight.

**Notation.** n trainers; c train concurrency; K = aggGoal (updates per version v); E = ⌊evalGoalFactor·K⌋ eval budget per version
(felix: evalGoalFactor 1 → E = K); g = 35 (re-eval gate, versions). A_i(t) ∈ {AVL_TRAIN, AVL_EVAL, UN_AVL}. S_T(t), S_E(t) = trainers
with an in-flight train / eval task (disjoint: one task in flight per trainer, FX-L26). ω(t) = cooling slots. e(v) = eval budget left in v.

**Free slots** (`async_base._handle_send_state`: `extra = concurrency − |selected_ends| − cooling`; `async_oort._concurrency_for_task`):
- train: F_T(t) = max(0, c − |S_T| − |S_E| − ω)
- eval: F_E(t) = max(0, min(n, c + e(v)) − |S_T| − |S_E| − ω), capped at e(v)

**Eligibility** (`client_availability.get_curr_task_ineligible_trainers`, `async_oort._task_extra_eligible`):
- train: A_i = AVL_TRAIN (or AVL_EVAL-free 2-state traces), i ∉ S_T ∪ S_E, not pending commit / withheld / cooling.
- eval: i ∉ S_T ∪ S_E, trained before, v − last_eval_i ≥ g; on 3-state traces A_i = AVL_EVAL only; on 2-state traces (syn_*) the
  same AVL_TRAIN pool as train (FX-T11).

**Coupling channels** (each becomes a metric in §3 F):
1. *Slot coupling (train waits on eval).* Eval may fill up to c + e(v) slots, so as train tasks return, F_T stays 0 while
   |S_T| + ω < c ≤ |S_T| + |S_E| + ω. Train is **blocked by eval** at t iff that holds and Elig_T(t) ≠ ∅.
2. *Candidate coupling.* A trainer in S_E is not train-eligible; on 2-state traces eval and train compete for the same trainers.
3. *Eval waits on train.* An eval-eligible trainer in S_T is skipped; e(v) resets only when train commits advance v, so eval cadence is
   paced by train throughput.
4. *Belief coupling.* An eval return refreshes the trainer's believed utility (per_trainer `last_eval_round`, belief source), which then
   steers train picks.

**P1 (bound, hyp).** Fraction of train slot-time lost to eval ρ ≤ evalGoalFactor · s̄_E / s̄_T. *Derivation:* per version, E evals hold
≤ E·s̄_E slot-seconds; a version lasts ≈ K·s̄_T / c with c slots, so ρ ≤ E·s̄_E·c / (c·K·s̄_T) = (E/K)·(s̄_E/s̄_T). *Measure:* ρ =
∫ min(|S_E|, (c − |S_T| − ω)⁺, |Elig_T|) dt / (c·T) from §4 occupancy. *Performance:* lost commit rate ≈ ρ·c/s̄_T; time-to-accuracy cost
via the run's accuracy-per-commit slope (I-G2).

**Evidence so far.** Cifar G0U mobiperf felix real n=50 (`run_20261010_050908_G0U_mobiperf_3st_*_felix_*_real`, jayne): 503 eval
selection calls chose 3 tasks (gate g = 35 ≫ the leg's versions), so ρ ≈ 0 on ≤ 30-min screens; IN4 needs long legs.

## 2. Telemetry inventory (measured: event counts in one G0U syn_50 cifar real leg per baseline, PR28-PR30, jayne)

| aggregator event | felix | fedbuff | refl (Oort stack) | feddance | what it carries |
|---|---|---|---|---|---|
| `selection` | ✅ train + eval | ✅ | ✅ | ✅ | eligible/chosen counts, `excluded_by`, `in_flight`, `per_trainer` (utility, speed, avl_state, excl, last_train/eval round); Oort stack adds explore/exploit ids, blacklist, priority, `stale_due`; felix adds pacer + `sys_util_mean`, `concurrency`, `effective_c` |
| `agg_round` | ✅ | ✅ | ✅ | ✅ (no `agg_goal*`) | per commit: `staleness[]`, `stat_utility[]`, `trainer_speed_s[]`, contributors, visibility lag; async adds `buf_depth`, `commit_gap_s`, ready/commit ts |
| `agg_timing` | ✅ | ✅ | ✅ | ✅ | per commit wall split: recv_wait, ingest, commit, other |
| `dispatch` | ✅ | ✅ | ✅ | ✅ | end, task, round, `ts` (wall only: **no vclock**) |
| `utility_belief` | ✅ | ✅ | ✅ | ✅ | believed vs actual utility per return |
| `abandon_timeout`, `agg_belief_change`, `run_end`, `trace_origin` | ✅ | ✅ | ✅ | ✅ | slot reclaims, avail beliefs, stop point |
| `slot_starvation` | ✅ only | — | — | — | freed slot with fewer eligible than slots |
| `inflight_residence`, `model_health` | — | — | ✅ only | — | sync drain accounting; weight norm |
| `comm` | — | — | — | — | fwdllm only; per-message bytes now come from `wire` (§4 #2) |
| `version_bump_census`, `step_timing`, `server_update`, `withheld_delivery` | — | — | — | — | defined, not emitted in these legs |

Trainer side, all six: `avail_change`, `task_recv`, `task_send`, `trainer_round`, `selection` (felix also `util_disparity`).
**Gaps (pre-§4):** no per-message bytes, in-flight not split by task, felix had no explore/exploit tags; task outcomes need
reconstruction (§4 #4).

## 3. Insight catalog (questions → metric → source; "existing" = computable today, IN2)

| ID | question | metric | source | baselines | state |
|---|---|---|---|---|---|
| **A · Selection** | | | | | |
| I-A1 | whom does each selector pick? | pick share by speed quartile, utility quartile, avail state | `selection.per_trainer` + `chosen` (existing) | all | hyp |
| I-A2 | coverage and fairness | distinct contributors / n over time; Gini of updates per trainer; never-picked share | `agg_round.contributing_trainers` (existing) | all | measured (feddance, FX-N90) |
| I-A3 | does the selector have a choice? | choice ratio = eligible / chosen per selection | `selection.num_eligible`, `num_chosen` (existing) | all | measured (F3, FX-L67) |
| I-A4 | how wrong are utility beliefs? | believed − actual by staleness and quartile | `utility_belief` (existing) | utility selectors | hyp |
| I-A5 | explore vs exploit share and payoff | explore/exploit picks; their aggregated share | `selection.explore_ids` / `exploit_ids` (all utility selectors) | oort family, felix | hyp |
| **B · Availability and waste** | | | | | |
| I-B1 | how much compute is wasted? | train-seconds dispatched but not aggregated, by outcome (stale-dropped, abandoned, evicted, rejected) / total | reconstructed (§4 #4) | all | hyp |
| I-B2 | picks lost to unavailability | selections later abandoned or withheld; mean slot-seconds held | `abandon_timeout`, `withheld_delivery` (partly existing) | all | hyp |
| **C · Aggregation** | | | | | |
| I-C1 | how stale are aggregated updates? | staleness histogram; weight mass by staleness | `agg_round.staleness` (existing) | all | measured (REFL, this doc §5) |
| I-C2 | how long does a sync round wait for stragglers? | version start → K-th fresh arrival; idle slot-seconds at the barrier | trainer `task_recv.sim_send_ts` + `agg_round` (§4 #4) | refl, oort family, feddance | hyp |
| I-C3 | async buffer dynamics | `buf_depth`, `commit_gap_s`, visibility lag | `agg_round` (existing) | felix, fedbuff | hyp |
| **D · Overheads** | | | | | |
| I-D1 | where does server wall time go? | recv_wait / ingest / commit / other per commit | `agg_timing` (existing) | all | hyp |
| I-D2 | client overhead share | (task span − D) / task span | trainer `task_recv` / `task_send` (existing) | all | hyp |
| **E · Communication** | | | | | |
| I-E1 | bytes per aggregated update | sent + received bytes / aggregated updates; wasted share (I-B1 outcomes) | `wire` (§4 #2) | all | hyp |
| I-E2 | communication to accuracy | cumulative bytes at each eval point | `wire` + `agg_eval` | all | hyp (EuroSys uses message counts, FX-N91 f) |
| **F · Felix train/eval interplay (§1)** | | | | | |
| I-F1 | how often does train wait on eval? | blocked-time share; ρ vs P1 bound | `selection` occupancy (§4 #1) | felix | hyp |
| I-F2 | does eval steal good train picks? | trainers in S_E that rank in train's top-k at that selection | `selection.per_trainer` + occupancy (§4 #1) | felix | hyp |
| I-F3 | is eval starved by train? | eval budget used per version; eval-eligible but busy count | occupancy + `selection(task=eval)` | felix | hyp |
| I-F4 | does eval feedback improve train picks? | utility-belief error with vs without a recent eval | `utility_belief` + per_trainer belief source | felix | hyp |
| **G · Time-to-accuracy decomposition** | | | | | |
| I-G1 | why is X faster to target? | TTA = U* / R: U* = aggregated updates to target (statistical efficiency), R = aggregated updates per hour (system efficiency); R = c_eff · (1 − waste) / s̄_T | `agg_round`, `agg_eval`, §4 #4 | all | hyp |
| I-G2 | marginal value of an update | d acc / d aggregated update per phase of the run | `agg_eval` + `agg_round` (existing) | all | hyp |

## 4. Instrumentation (telemetry only, no behaviour change, no checker reads it)

| # | what | where | serves | state |
|---|---|---|---|---|
| 1 | `selection` (async: felix, fedbuff) gains `in_flight_train`, `in_flight_eval`, `cooling`, `free_slots`, `idle_eligible`, `train_slots_held_by_eval`, `train_blocked_by_eval` | `async_base._occupancy` (task per end recorded at dispatch) | I-F1, I-F3 | ✅ `test_occupancy_telemetry.py` |
| 2 | `wire` event `{dir, peer, bytes}` per message, both roles | `channel._emit_wire` at send/recv byte accounting | I-E1, I-E2 | ✅ `test_wire_event.py` |
| 3 | felix train `selection` gains `explore_ids`, `exploit_ids` (Oort stack had them) | `async_oort._selection_extra` | I-A5 | ✅ |
| 4 | task outcomes (dispatch → aggregated / stale / abandoned / never returned) | reconstructed, no new event: trainer `task_recv` (round, `sim_send_ts`), `trainer_round` (`sim_completion_ts`), `task_send` (task), `agg_round` contributors + staleness, `abandon_timeout` | I-B1, I-C2, I-G1 | reconstruction in `insights.py` (count-based today; span-based next) |
| — | dropped: `agg_goal*` on syncfl `agg_round` (checkers sort and count cycles on it: U4, EV); new `dispatch` sim events (EV checks read `dispatch`) | — | read K from `aggregator_config.json` | — |

**Analysis:** `scripts/insights.py RUN... [--json] [--plot]` (one function per ID, real/sim twins paired and printed side by side;
`tests/harness/test_insights.py`). Volume: `wire` = 2 events per message (≈ dispatches + updates), the rest are fields.

## 5. Findings (measured; each line cites its evidence)

- **REFL staleness (I-C1, 10-12):** aggregated updates > 5 rounds stale are 0% on syn_0 (G1A/G1AS cifar + speech, up to 496 rounds,
  max staleness 5), 0-2.4% on syn_50 and 0-1.1% on mobiperf (G0U, max 12); stale share 14-33%. Source: `agg_round.staleness`, jayne
  `async_cifar10/experiments/*refl*{G1A,G1AS,G0U}*`. Why it matters: a 5-round cap (SAFA's) would change nothing on syn_0.
- **Telemetry smoke (§4, 10-12, stub, cifar):** real↔sim agree on every logical metric: felix mobiperf T3 choice ratio 5.75 vs 5.73, explore
  share 0.42% vs 0.42%, waste 0.8% vs 0.6%; refl mobiperf T3 waste 0.4% vs 0.8%, 4.31 vs 4.34 MB per aggregated update
  (`pool_20261012_183234_T3`). Felix chose 0 eval tasks on these ≤ 6-min legs (gate g = 35), as §1 predicts.
- **FedDance coverage (I-A2; paper eval text, evidence `fd_20261012_0503` telemetry, FX-N90):** FedDance climbs fast, then plateaus below 70% on
both datasets. Its utility favours fast, high-loss devices, so early rounds are short (12 s vs 15.5 s for random selection on CIFAR-10)
and accuracy rises steeply (≈40% within 2 h). But FedDance gives a selected device whose update was dropped as a straggler zero
statistical utility (Alg. 1, l. 3), and over-commit (13 picks for 10 updates) drops three devices every round. Within two hours a third
of the CIFAR-10 population (99-105 of 300) is zeroed and never contributes again; under MobiPerf, where few devices are online at once,
only 91 of 300 ever contribute and 20% of devices supply 96% of updates. With non-IID data (Dirichlet α = 0.1) the excluded devices'
classes never reach the model: CIFAR-10 plateaus at 64-65% on the synthetic traces and 49% on MobiPerf; Speech Commands shows the same
skew (55 of 100 devices ever contribute, 20% supply 84% of updates) and reaches 68.9% after 20 h. Extrapolated, FedDance needs > 48 h
to 70% everywhere except CIFAR-10 syn_50 (39 h).
- **What moves accuracy and time to accuracy (G3, n:c:K 75:15:10 cifar / 50:15:10 speech, 2 h; real legs, sims agree; PR37 + PR38):**
  - *Unavailability (syn_50_3st vs syn_0) hurts by awareness and task length.* Aware selectors keep their pace (Felix +4-7% versions);
    unaware ones lose rounds to abandons and gated uploads: cifar oort −44% versions, fedbuff −13% (staleness 1.4 → 4.4). Speech is far
    harsher (tasks ~5× longer vs the same 1 min-3 h windows): fedbuff −59% versions, oort −85% (FX-N106).
  - *Dataset.* Speech gives ~5× fewer versions in 2 h (felix 1031 vs 220) but more accuracy per version; the sync four are round-bound
    on speech (≤ 23% at 97-121 versions) while cifar Oort reaches ~50%.
  - *Streaming (lin, data 10% → 100% over 2 h) costs accuracy, not versions,* for multi-step learners (cifar −3 to −8 pp max, speech
    −6 to −11 pp); 1-mini-batch REFL / FedDance barely change (they never use the full shard).
  - *Server lr (FX-D185).* Felix at 0.3 (PR37) reached 48.9% cifar in 2 h; at FedBuff's 1.0 (PR38) 53.8% and 50% at 63 min vs FedBuff 107.
  - *Selection concentration.* Felix's top 20% of clients supply 57% of cifar updates (FedBuff 23%, Oort family 59-69%, FedDance 63-69%);
    unavailability spreads Felix's picks (43%). *Driver (PR39b, `selection_audit.py`):* shard size, not speed. Picks vs size ρ +0.95 cifar / +0.80 speech,
    vs speed +0.13 / +0.40, vs per-sample loss −0.88 / −0.39. Oort's |D|-weighted utility does this on a duration model where size costs
    no time (FELIX Q-DUR). Class mass of updates stays within TV 0.04 of the population (FedBuff 0.08).
