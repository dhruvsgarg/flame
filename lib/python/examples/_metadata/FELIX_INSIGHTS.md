# Felix insights — data-backed mechanisms behind time-to-accuracy and communication

Read [ROBUST_FL_READINESS.md](ROBUST_FL_READINESS.md) first; its doc rules apply here. **Purpose (operator 10-12):** turn end-to-end
numbers (time-to-accuracy, communication) into arguments backed by telemetry: *why* Felix wins or loses against each baseline, with
a formal model where one exists and a measured figure for every claim. **Status:** §4 instrumentation ✅ (10-12); metrics in `scripts/insights.py`; findings grow per run (§5); reader takeaways in §7. Iterative: each run adds evidence; a claim lives here only with its metric, source and evidence.

**Entry format.** `ID · claim · metric (formula) · source (event.field) · baselines · figure · evidence (run dir) · state`.
States: `hyp` (stated, unmeasured) · `measured` (number + run) · `proved` (bound derived + measured inside it) · `refuted`.
An insight counts only if real and sim agree on it (same metric on a parity pair, PARITY C6); sim-only figures say so.

## Next steps (top item is next)

- **IN6 · Re-read F49 at the final targets · ✅ 10-25 (FELIX FX-N146).** No stack reaches cifar 80% / speech 88% in 12 h, so §5 F49
  and §7 compare at the highest 5-eval mean Felix and FedBuff both reach (`experiments/F49_1024_1530/n146_*`); numbers at the hit
  come from FB1. Dropped (operator 10-24): tuning the rate for speech's first 2.5 h (judge = time to target) and explaining the
  speech gap to F43 (accepted: setup changed).
- **IN7 · Bytes frontier vs Oort · todo (after FB1).** Where Oort reaches the target on long runs, compare GB to it; probe whether
  Felix's late phase can cut bytes (`cp`, per-client local work) without losing time to target.
- **IN1 · Verify the model against code · todo (no runs).** Confirm §1 notation line by line: eval span `s_E` (trainer `task_recv →
  task_send` for `task=eval`), eval budget reset rule (`curr_round_eval_slots_left` on train commit only, `async_oort._cleanup_recvd_ends`),
  cooling slots, and where `PROP_LAST_EVAL_ROUND` is first set (the eval gate needs it non-null). *Exit:* every §1 symbol cites a code line.
- **IN2 · Run `insights.py` on every PR31 pool · todo (after PR31).** I-A2/A3/A5, B1, C1, D1, E1, F1, G1 today; add I-A1 (pick share by
  speed quartile), I-A4, I-C2 (sync barrier wait), I-G2 next. *Exit:* one JSON + figure per row, real↔sim agreement per metric.
- **IN5 · Watch in every long run (D5, 48 h rungs, F47) · standing (F46).** Per leg: hour the clients' loss of the received model
  falls below ~0.3 (data-bound onset) vs the accuracy plateau; pick dilution (picks~size, samples / update) vs c / trainable on
  MobiPerf; whether rate `new`'s 12 h lead survives to 48 h; held-upload share of τ at K 10. *Exit:* each measured on cifar +
  speech, syn_50 and MobiPerf.
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
| I-C4 | where does staleness come from? | τ split by visibility lag (ready → commit > 60 s = held upload) vs in-flight work: share of updates, share of Σ τ, task span per source | `agg_round.staleness`, `update_visibility_lag_s`, `trainer_speed_s` (existing) | all async | measured (F45, §5) |
| I-C3 | async buffer dynamics | `buf_depth`, `commit_gap_s`, visibility lag | `agg_round` (existing) | felix, fedbuff | hyp |
| **D · Overheads** | | | | | |
| I-D1 | where does server wall time go? | recv_wait / ingest / commit / other per commit | `agg_timing` (existing) | all | hyp |
| I-D2 | client overhead share | (task span − D) / task span | trainer `task_recv` / `task_send` (existing) | all | hyp |
| **E · Communication** | | | | | |
| I-E1 | bytes per aggregated update | sent + received bytes / aggregated updates; wasted share (I-B1 outcomes) | `wire` (§4 #2) | all | hyp |
| I-E2 | communication to accuracy | cumulative bytes at each eval point | `wire` + `agg_eval` | all | measured (F49, §5) |
| **F · Felix train/eval interplay (§1)** | | | | | |
| I-F1 | how often does train wait on eval? | blocked-time share; ρ vs P1 bound | `selection` occupancy (§4 #1) | felix | hyp |
| I-F2 | does eval steal good train picks? | trainers in S_E that rank in train's top-k at that selection | `selection.per_trainer` + occupancy (§4 #1) | felix | hyp |
| I-F3 | is eval starved by train? | eval budget used per version; eval-eligible but busy count | occupancy + `selection(task=eval)` | felix | hyp |
| I-F4 | does eval feedback improve train picks? | utility-belief error with vs without a recent eval | `utility_belief` + per_trainer belief source | felix | hyp |
| **G · Time-to-accuracy decomposition** | | | | | |
| I-G1 | why is X faster to target? | TTA = U* / R: U* = aggregated updates to target (statistical efficiency), R = aggregated updates per hour (system efficiency); R = c_eff · (1 − waste) / s̄_T | `agg_round`, `agg_eval`, §4 #4 | all | measured (F49, §5) |
| I-G2 | marginal value of an update | d acc / d aggregated update per phase of the run | `agg_eval` + `agg_round` (existing) | all | hyp |
| I-G3 | how long does an early lead last? | lead = ∫(acc − acc_FedBuff) dt in pp·h, split 0-4 h / 4-12 h; hour FedBuff is ahead for good; break-even ≈ end + lead / late gap | `agg_eval` (existing) | all vs FedBuff | measured (FL1 F44 / F45, §5) |

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

**F49 end to end (re-read 10-25 at the final targets, IN6: nobody reaches cifar 80% / speech 88% in 12 h; compared at the highest
5-eval mean Felix and FedBuff both reach, = FedBuff's 12 h peak, so × sooner is capped by the horizon): Felix new (`fn_chr` = `fx` at rate b 0.1) vs every baseline (`experiments/F49_1024_1530`, code `0f6858e70`; cifar n 300 / speech
n 100 × `s0u` / `s50u` / `smu`, 12 h, augmentation on, 1 seed, averaged model; 56 / 56 legs ok, config audit 56 / 56 ✅).**
I-G1 split from `agg_round` (cumulative updates) + `agg_eval` + `wire`: TTA = U* / R, U* = updates to target, R = updates / h.

| cell | highest common 5-eval mean: Felix vs FedBuff h · GB | U* ratio · R ratio (FedBuff / Felix, Felix / FedBuff) | final Felix · FedBuff · best sync |
|---|---|---|---|
| cifar `s0u` | 71.4%: 6.7 vs 12.0 h (1.80×) · 375 vs 753 GB (−50%) | 65%: 2.24× fewer updates · 0.89× the rate | 74.4 · 71.2 · 52.4 (Oort) |
| cifar `s50u` | 69.3%: 5.2 vs 12.0 (2.32×) · 369 vs 513 (−28%) | 65%: 1.65× · 1.70× | 73.2 · 69.0 · 51.3 (Oort*) |
| cifar `smu` | 56.8%: 5.0 vs 12.0 (2.40×) · 371 vs 1184 (−69%) | 50%: 1.67× · 0.94× | 59.1 · 56.3 · 47.4 (FedDance) |
| speech `s0u` | 53.7%: 8.1 vs 12.0 (1.48×) · 227 vs 294 (−23%) | 50%: 1.33× · 1.25× | 63.5 · 53.3 · 17.1 (FedDance*) |
| speech `s50u` | 44.3%: 4.9 vs 12.0 (2.45×) · 167 vs 162 (+3%) | 40%: 1.0× · 2.6× | 64.5 · 44.0 · 15.0 (FedDance*) |
| speech `smu` | 46.9%: 9.6 vs 11.9 (1.24×) · 286 vs 315 (−9%) | 40%: 1.19× · 1.5× | 52.8 · 46.4 · 21.1 (FedDance*) |

- *Time: Felix is first to every rung (5 pp up to the final target) on every cell vs every baseline (6 / 6 PASS, `n146_*_verdict.md`),* and the only stack past 50% on speech /
  past 55% on cifar except FedBuff. Expectations held: cifar 65% 2.0 / 2.8× sooner on syn traces (F47: 1.9-2.7×), MobiPerf 1.5×
  (≥ 1.2×), sync stacks ≥ 5× later to 50% or never, speech ≥ 1.4× wherever FedBuff gets there at all.
- *Where the time comes from differs by cell, and that is the paper's structure.* No unavailability (`s0u`): all of it is
  statistical efficiency (2.24× fewer updates at 11% *lower* rate: big-shard picks run longer tasks, picks~size +0.92, 40 vs 25
  samples / update). Unavailability (`s50u`): half each (1.65× × 1.70×); FedBuff idles on check-ins (98k starvation events vs 0).
  Speech `s50u`: all rate (same U*, 2.6× R): there awareness alone is the win. MobiPerf: U* only (R 0.94×).
- *Bytes vs FedBuff follow U*, never R:* every update costs the same model round trip (cifar 4.3 MB, so GB to target = U* × 4.3 MB
  to ±2%). Felix saves 27-55% GB where it needs fewer updates and nothing where it wins by rate (speech `s50u` +4%). Time and bytes
  are separate claims with separate mechanisms: selection + step buy both; awareness buys time only.
- *Bytes vs the sync stacks and Oort: not a win, and the paper must say so.* At 40% the 1-step sync stacks spend 45-60 GB vs
  Felix's 50 (syn) and 44-51 vs 98 GB (MobiPerf); Oort reaches 40-50% on 5-13 GB (Felix 50-100 GB, 7-14× more) at 4-12× the time.
  At the sync stacks' own 12 h budget (139-184 GB) Felix is +10-17 pp ahead on syn traces but ties on MobiPerf (46-48 vs 47-48%);
  at Oort's budget (7-16 GB) Felix is 27-37 pp behind. Claim = fewest GB to every target ≥ 55% (cifar) / ≥ 45% (speech), the
  only targets no baseline but FedBuff reaches; below that Felix trades bytes for 4-12× time.
- *Why the sync stacks are byte-frugal early (cifar: 14 s tasks, 59% transfer, rounds wait ~31 s):* a fresh 1-step update (τ 0) is worth about one stale 1-epoch Felix update (U* to 40%:
  9-12k vs 11.5k on syn traces); on MobiPerf Felix needs 22.6k (τ 6.1, held uploads), so it ties per byte. Their limit is the round
  barrier: 2000 rounds × 15 picks × 10-24 samples in 12 h (R 2-4k / h vs Felix 14-17k), flat at 46-49% from 6 h. Oort is frugal for
  a different reason: 600 local samples per 4.3 MB (U* to 40% 1.1-1.5k updates, 8-10× fewer) at 214-237 rounds in 12 h.
- *Speech sync stacks sit at chance (3.7% of 35 classes) for 4-6 h and end 4-21%:* 160-186 rounds in 12 h (cifar 2000). Not bytes
  (async moves more: FedBuff 295 vs FedAvg 64 GB) but waiting: a 1-step speech task is 123 s, 93% of it moving the 29 MB model, and
  each round waits for its slowest of 6 picks (median 256 s ≈ the 238 s between rounds), so 12 h trains ~2 passes of the data.
  Async overlaps 20 tasks, never waits on the slowest, and each task carries an epoch (FedBuff 28% compute vs FedAvg 7%). Mechanical, not broken (cifar
  same code: 47-52%), but it is the setup's weakest point for a reviewer (K 5 by the 5% rule, 1 local step from FedDance's config).
- *Oracle knowledge does not rescue a sync stack:* REFL* = REFL ±0.6 pp everywhere; FedDance* is faster early (cifar MobiPerf 40%
  at 120 vs 200 min) but ends 1.5 pp lower and concentrates (top-20% 91% vs 67%). The bottleneck is the barrier and local work, not
  knowing who is online, which is why Felix's awareness only pays inside an async engine.
- *The c ramp is a strict win on cifar MobiPerf (12 h):* `fn_chr` vs `fn` same eval tick at 40-50%, 55% 30 min sooner, +0.5 pp
  final, **−50% GB** (659 vs 1308), τ 6.1 vs 7.8, starvation 5k vs 209k (c 60 > trainable ~55-68 leaves slots waiting), picks~size
  −0.13 vs −0.37, top-20% 50 vs 63%. c follows φ 0.81 → 0.06 to ~35 (= its pool term). On speech φ stays 0.56-0.91 (clients never fit
  their data in 12 h), so c 19 → 16 and `fn` ≈ `fn_chr` (53.2 / 52.8%, −9% GB): the ramp acts only once data is fit, as designed.
- *Against expectation: speech `s0u` FedBuff leads the first ~2.5 h* (1 h 18.9 vs 8.8%, lead 0-4 h −4.9 pp·h); raw models agree (13.9
  vs 9.9%), so it is training, not the served average. Felix makes 10% fewer updates in hour 1 (489 vs 542, tasks 142 vs 104 s from
  big-shard picks) yet loses 2× in accuracy, so per-update value is lower early. Hypothesis: rate `new`'s large step (0.7) on early
  speech updates; untested (F48R sweeps cifar only).
- *Against expectation: speech is 10-14 pp below F43 for both stacks* (Felix 38.5 / 53.7% at 4 / 8 h vs F43 56.5 / 67.8; FedBuff
  36.7 / 47.6 vs 46.7 / 57.8). Three changes are confounded: augmentation (speech's centralized model needs 30 vs 15 epochs with
  it), c 30 → 20, harness → bench. Felix ends 11 pp under the 75.6% speech ceiling (F43: 2 pp). F48S (harness, aug, c 20) separates them.
- *Server averaging matters most where steps are big:* averaged − current model at 12 h: speech Felix +4.2-13.0 pp, FedBuff
  +1.6-10.4; cifar −0.4 to +3.0; eval-to-eval swing 1.4-4.5 → 0.2-1.0 pp. Felix's lead survives without it (speech `s0u` current 56.4
  vs 44.1%, cifar `s50u` 72.2 vs 68.5%), so it is not an averaging artifact.
- *Staleness is not the binding cost once data is not memorized:* Felix runs τ 11.7 vs FedBuff 7.3 on cifar `s50u` and still ends
  +4.2 pp (F44-F46 without augmentation: the same staleness gap cost 1.2 pp). Overfitting made staleness look expensive.
- *Coverage on MobiPerf is the trace's, not the selector's:* 57-62% of devices ever train for every stack (the rest never charge
  in the 12 h window). Fairness metrics on MobiPerf must be over the reachable set.
- *Sanity:* cifar anchors repeat F47 within one eval tick (FedBuff 65% at 420 min `s0u` both; Felix 50% at 80 / 80 / 140 min);
  speech does not (above). **The cifar ceiling is under-trained, not a bound:** Felix 74.4% > the 71.9% `s0u` 12 h ceiling (40
  epochs, fixed lr on 12.2k samples vs Felix's ~480 sample-passes per sample), so 90 / 95%-of-ceiling targets understate the top.
  Converged on full shards (= trace end; cosine lr, 200 epochs) the centralized cifar ceiling is 89.0% (FELIX FX-N146).

**FL1 log (FELIX FX-N141; cifar bench n 300, 12 h; standing: one crisp line per worked / failed mechanism, for motivation / eval).**
- *Unaware async idles, and the idling hides staleness (F44 `s50u`, `FL1_1023_1700`).* FedBuff waits on check-ins (98k
  slot-starvation events) and holds slots of departed clients to its 677 s timeout: ~33 of 60 slots busy, 116k updates. Notify /
  evict (`b1`) keeps all 60 busy: 213k updates, 50-55% 80 min sooner, but staleness 7.8 vs 4.8 under the same 1/√(1+τ) rate costs
  1.2 pp final. *Confirmed (F45 `b1_c33`):* aware at c 33 ends 57.1 vs FedBuff 56.3% (staleness 4.2) but at FedBuff's pace (lead −8
  vs +23 pp·h). Motivation: awareness buys speed; aggregation must absorb the staleness it creates.
- *Utility alone keeps accuracy at fewer bytes (F45 `s3u` / `s3uo`, `FL1_1023_2130`):* |D| q picks big shards (picks~size +0.84-0.96),
  41 vs 25 samples / update on `s0u`: FedBuff's final with 20-27% fewer updates and GB. q from the training pass ties a pre-step batch.
- *Speed buys early and costs late (F45 `s3vu`, `s3v`, `s3vubx_oc`):* across arms each pp·h gained before 4 h comes back 0.6-1.0×
  after; the speed term costs 1.5-2 pp final on streams. Speed alone picks small shards (83 vs 164 samples / update, class TV 0.117 vs
  FedBuff 0.051) and ends −3.8 pp. Balance wins back about half; UCB does nothing (±0.3 pp); Oort's cutoff is fastest to 50%, lowest final.
  *F46 (K 10):* under every aggregation utility alone ends higher in 25 / 28 pairs (+0.6 pp, −19% GB); under rate `new` the speed
  term buys ≤ 10 min to 50%, so the FedScale cost of big shards is worth paying.
- *The server step sets the pace, not the staleness rule (F46 `uo_{o,n,nl,p1}`, same selection, K 10, τ 5.9):* rate `new`
  (weight 0.61-0.78, F45) vs 1/√(1+τ) (0.51-0.53) vs 1/(1+τ): 50% at 60-80 / 120-130 / 210-280 min; static final 69.6-69.9 / 68.2 / 65.5%.
  A bigger step wins everywhere and costs nothing final; the stronger discount loses everywhere. Felix new's early lead is its step.
- *Sub-batch lr normalization buys stability, not accuracy (F46 `uo_n` vs `uo_nl`):* a partial batch at full lr weighs each sample
  lr / m instead of lr / B; normalizing restores equal weights, halves the stream eval swing (0.22-0.48 vs 0.30-0.85 pp per eval)
  and leaves finals within 0.4 pp. Static shards (few partial batches) don't change.
- *Streams overfit once the window's data is learned (F46 `fn`, hourly evals + trainer `visible_samples`):* the federation sees
  only ~8.4k → 12.1k samples over the 12 h window (+3% / h; 1.5-3% of tasks see new data), and a centralized model on the same
  data gains just 59.8 → 61.8% (that ceiling froze draws too; F47 refutes "mostly data"). Overfitting sits on top: test loss
  bottoms at 1.24 at 2 h and climbs to 3.1 at flat accuracy while clients' loss of the received model falls to 0.03. Arms that train less late (c by phase,
  c 30) hold test loss at 1.3-1.8 and end +0.3-1.5 pp. Cause found: the trainer's GPU preload freezes one crop / flip draw per
  sample, so every stack re-fits identical images (FX-D233). Open: whether an overfit model absorbs day 2's data (F47, 24 h).
- *On streams a client's loss says how well its data is fit, not how much learning is left (F46 `ps`, `pscp`, FX-T85).* Oort and
  REFL read loss as learning left because their shards are static, so a fit shard is a spent one. With streaming data a fit
  client may get new data next hour, and no client can know what is left. Measured: φ falls with clients' loss on static data
  too (0.41 at 12 h) while accuracy still climbs, so a φ-scaled step loses 0.9-3.2 pp static and gains ≤ 0.6 on streams.
  Using φ to cut c is safer than to cut the step (static −0.2 pp): c sets bytes, the step sets progress.
- *A fit client's loss cannot tell new data from noise; pacing c can (F46 offline, FX-T88):* jumps ×8 vs a client's own last task
  hit 5.8% of tasks on static data (none new) vs 3.4-3.6% on streams. But c by phase regulates itself: as c falls, each pick waits
  longer and carries more new data (`s0u` `uo_cp`: tasks with new data 5.8 → 12.3 → 20.3% at 3-6 / 6-9 / 9-12 h vs a flat 3% at
  c 60), which lifts the received-model loss and so φ and c. *Refuted over 48 h (F47):* c sits at its floor 10 (φ ≤ 0.12) from
  6-8 h to 48 h while data keeps arriving; c by phase saves bytes, it does not pace.
- *Frozen crop draws, not data, made the stream plateau (F47 `aug` A/B, `FL1_1024_0432`):* per-batch crop / flip keeps test loss
  < 1 and clients' received-model loss at 1.2-1.3 (vs 0.03-0.13) and lifts Felix +10-12 pp at 4 h, FedBuff +11-13 at 8-12 h, MobiPerf
  +16-17 pp; streams reach 70% in 5.5-5.8 h (Felix) past the old frozen-draw ceiling. With it Felix's lead no longer fades: 65%
  1.9-2.7× sooner than FedBuff at 36-58% fewer GB on static / syn traces, ahead at every eval.
- *MobiPerf removes choice before it removes slots (F47 `smu`):* ~50-67 of 300 trainable vs c 60, so Felix's picks equal FedBuff's
  (picks~size −0.45 vs −0.46, τ 15) and its edge to 65% falls to 1.24× (−9% GB). c × trainable share (≈ 12) restores choice
  (+0.48, τ 3.3) and ends +3.8 pp on 92% fewer GB, but reaches 40% 4× later: the case for a pool-aware c that keeps early throughput.
- *Augmentation is setup hygiene, not a Felix claim (FX-D233):* it lifts every multi-pass learner (Felix, FedBuff, Oort's 600-sample
  tasks); 1-step REFL / FedDance barely revisit a sample and their test loss stays ~1.35 (F43), so they gain little. F47: Felix's
  gap over FedBuff holds with augmentation on both, without any pacing (65% 1.9-2.7× sooner on syn traces).
- *Dynamic c: throughput first, choice once data is fit (F48a, `FL1_1024_F48c`, cifar MobiPerf, augmentation, 3 h;
  `dynamic_c_report.py`).* Every fixed bound that restores choice loses early: c ≤ ½ / ⅓ trainable or c × share lift picks~size
  from −0.36 to +0.18 / +0.42 / +0.54 but end 48.6 / 43.8 / 36.5% vs c 60's 51.8% (26k / 14k / 7k vs 77k updates). Early every
  update helps and the model is far from fit, so updates beat selection. The ramp, c = φ · 60 + (1 − φ) · min(60, ½ trainable),
  keeps c 60's pace (same eval tick at 40 / 45 / 50%, 52.1%) on 19% fewer GB.
  - *Ramp down:* φ = mean client loss over the last n updates / its running peak, recomputed every commit (K updates), so c follows
    the fit within minutes: 57 → 52 → 47 → 43 → 37 → 32 at 0.5-3 h as φ goes 0.88 → 0.28 (about −5 per half hour). It starts as soon
    as clients fit their data, which is fastest where the pool is small and retrained often (MobiPerf).
  - *Ramp up:* φ divides by a running peak, so it barely recovers (F47 c by phase: floor 10 from 6-8 h to 48 h while data kept
    arriving). c rises through the pool term instead: as φ → 0, c → ½ trainable, so c tracks availability (F48a `ch2`: 21 → 24
    when trainable rose 43 → 49 at 3 h). Over a MobiPerf day (F48L) trainable stays 40-66, so c holds 25-38: no night surge to follow.
  - *Never binds on syn traces:* ½ trainable ≥ 72 > 60 on syn_0 / syn_50_3st, so the ramp is a no-op there by construction.
  - *24 h confirm (F48L, `experiments/F48_1025_0114/cifar_L`):* ramp 64.8 vs c 60 63.7 vs FedBuff 61.2%, the only arm to 65% (1390
    min), on 1056 vs 2319 / 2152 GB; τ 8.4 vs 11.7; picks~size −0.40 → −0.12. The late gain is less staleness and fewer wasted
    slots (starvation 5k vs 417k), not yet selection (picks~size stays < 0).
- *Eval tasks still cost more than they buy (F48L `fn_ev`):* refreshing beliefs with Felix old's eval tasks on top of the ramp ties
  (64.6%, one tick later at 65%, same GB) while 262k slot-starvation events appear: the selector's stale beliefs are good enough.
- *The 2024 rate constants over-damp (F48R, one knob at a time on `fn_chr`, cifar `s0u` / `smu`, 12 h):* every less-damped arm
  (a 0, b 0.2, s 0.2) is 1-3 eval ticks sooner with ≤ +0.5 pp; every more-damped arm (a 0.5, b 0.05, s 0.7) is 2-4 ticks slower and
  −0.2-1.0 pp. a 0 is fastest but drops the staleness term; b 0.2 keeps the staleness–utility trade-off at ≤ 1 tick of it and
  becomes `fx`. Same story as F46: the step size, not the staleness rule, sets the pace.
- *Why K stays fixed while c moves (reviewer question):* K is the buffer, the number of updates averaged per server step; c / K
  sets staleness (τ ≈ c / K). Lowering c at fixed K lowers staleness (F48a τ 5.5 → 3.2), which is what unavailability needs.
  Lowering K with c would keep τ but average fewer updates per step at rate `new`'s large weight (0.6-0.8), so noisier steps.
  Measured: adaptive K (AIMD on loss, C4) walked K to its floor or to c with no gain (FX-T65); K 20 at c 60 lost to c 30 at K 10 at
  the same τ (−0.5 pp at 2.3× GB, F46). And K is FedBuff's own choice (paper §6.1: K 10 across benchmarks), so only Felix's c
  moves; c ≥ K keeps one full buffer in flight.
- *Selectivity is c / trainable, not c / n (F46 `s50u`, FX-L103):* at c 60 of ~148 trainable, 40% of the pool is in flight and
  big-shard (long-task) clients are the ones held busy, so utility picks dilute (picks~size +0.67, 33 samples / update) vs c 30
  (+0.93, 40). Availability shrinks the selector's effective choice before it shrinks its slots: the case for c ∝ trainable count.
- *Felix's rate `new` is a big constant step, not a staleness rule (F44 / F45 `agg_round`):* mean weight 0.73-0.78 early, 0.61-0.65
  late vs 0.51-0.53 for 1/√(1+τ); its utility term is ~constant and its staleness exponent (0.25) is weak.
- *Speed arms lose by update noise more than staleness (F45 `s0u`):* mean τ 3.9 for every c 60 arm (τ > 8: 10.6 vs 8.0%), but 30%
  more updates of 12% fewer samples at the same step. `s50u`'s staleness is mostly unavailability (next line).
- *Unavailability makes staleness that straggler fixes cannot see (I-C4 `insights.py`, F45 `agg_round` `update_visibility_lag_s`; paper argument
  for staleness ≠ unavailability):* staleness has two sources. (1) In-flight work: τ ≈ c / K, its tail = slow tasks; without
  unavailability (`s0u`) it is all of it (τ > 20 from 89-92 s tasks vs 11-14 s median, 0 held uploads). (2) Held uploads: a client
  that leaves mid-task finishes, but its upload waits until it returns (our availability model: compute completes, the send waits).
  On `s50u` these are 1.1-1.2% of updates but **49-51% of all staleness**, mean τ 184-373 versions (hours), from clients with
  19-21 s tasks (median 10-11 s, not 10× stragglers); 99-100% of τ > 20 updates are held for FedBuff / `b1_c33`, 62-71% for the
  Felix arms. Same for every stack (FedBuff 1337 withheld deliveries, Felix arms 2094-2140). So async FL's straggler machinery
  (staleness-weighted rates, deadlines, speed-aware selection) targets source 1: a staleness rule only down-weights a held update
  once it lands, and a speed rule cannot predict who will leave. Availability-awareness acts on source 2 before it happens (who to
  pick, whether to evict). Next evidence: τ by source per stack on mobiperf, and how much accuracy the held updates cost
  (drop-on-return ablation).
- *Loss utility: the pass-averaged EMA is the one signal that builds a good buffer (q audit, `experiments/qaudit_1024`):* early no
  |D|-weighted loss predicts an update's value; once shards are memorized, top-15 by |D| · Oort EMA cuts test loss 4× random (0.155
  vs 0.035), while one pre-step batch (−0.30) and held-out loss (−0.13) are worse than random.
- *An early lead fades only where Felix trains more on the same data (I-G3; F46 `fn` vs FedBuff, gap per 2 h block):* static
  +6.5 / +10.3 / +6.0 / +2.9 / +1.6 / +1.4 pp (never behind); `s0u` +9.0 / +6.1, then ±0.5 (tie); `s50u` +11.6 / +7.7 / +0.3 / −0.9
  / −1.0 / −0.4 (behind at 88% of evals after 4 h). On `s50u` Felix makes 179k updates vs FedBuff's 120k (FedBuff idles ~half its
  slots) on the same ~12k visible samples, and its test loss ends 3.26 vs 2.26. Overfitting, not staleness: `k20` (τ 5.9, 179k
  updates) ends −0.4 pp vs FedBuff, `c30` (τ 5.8, 72k) +0.2. The fix is to stop training fit data (c by phase, augmentation; F47).
- *More local epochs = drift, not staleness (F44 `b1gw_e3`).* Same staleness (3.9), 45-60% fewer updates, equal accuracy to 4 h
  then −3 pp by 8 h on non-IID α 0.1; GB −28 to −58%. Local work needs per-client control, not a global epoch count.
- *Weight decay on clients does nothing for async cifar (F44 `b1gw`):* ±0.5 pp final vs `b1`.
- *Static data has no plateau (F44 `s0`):* every arm 64-68.5% at 12 h; the 54-57% stream plateau was frozen-draw overfitting (F47).

**Every finding below ran on work-independent task durations (before FELIX FX-D197, 10-20) and is void for the paper;
kept only as hypotheses to re-measure.**

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
    no time (fixed by FELIX FX-D197). Class mass of updates stays within TV 0.04 of the population (FedBuff 0.08).

## 6. Paper methods and story (moved from FELIX_READINESS 10-23; code is the source of truth)

**Task-time model (FX-D197, operator 10-20; paper methodology).** Every task's modeled time is FedScale's, as Oort and REFL compute
it (`getCompletionTime`):

    t = 3 · samples · ms_per_sample / 1000  +  transfers · model_kbit / bandwidth_kbps      (then ÷ harness delay factor)

- *Devices:* FedScale's `client_device_capacity` traces (500k devices; `third_party/REFL/dataset/data/device_info`); trainer k gets
  device k, REFL's `clientId % len(profiles)`. Registry fields `compute_ms_per_sample`, `bandwidth_kbps`. Across all 500k devices:
  compute p10 / p50 / p90 21 / 41 / 175 ms per sample, bandwidth 1.8 / 6.1 / 32.5 Mbps.
- *Work:* samples = local steps × batch (Oort 20 × 30 / 15, REFL 1 × 10 / 20, FedDance 1 × 24 / 32), else epochs × visible shard
  (Felix, FedBuff: 1 epoch, so it grows with streamed data). 3× = forward + backward. Model 2.15 MB cifar, 29.0 MB speech.
- *Transfers:* train = download + upload (2); eval = one forward pass (1×) + download; Felix's I9 utility pass = one forward pass.
- *Timeout:* each dataset's abandon timeout is 2 × the longest task in its reference population (cifar 903 s, speech 2560 s), the
  same ratio as the old 90 s / 450 s.
- *Mean task time (G3 populations, cifar / speech):* Felix / FedBuff 50 / 324 s, Oort 168 / 185 s, REFL 11 / 112 s, FedDance
  15 / 115 s, eval 18 / 126 s.
- *What it replaced:* one per-client delay (2024 speed classes, mean 12 s cifar / 58 s speech) for every task, whatever its work.
  That let Felix train 1.7× bigger shards at no cost, undercharged Oort's 600-sample tasks 3.4× relative to Felix's epochs, and
  charged REFL / FedDance a full epoch's time for one minibatch.
- *Validation:* EV3 recomputes t for every train task (registry device, reported samples and model size); `task_time_audit.py`
  reports mismatches and per-baseline train / eval times per leg. 10-20 shepherd smokes (G3 syn_0, 20 / 30 min vclock, all six):
  0 mismatches over 4,212 train tasks, every eval span equal to its formula, EV PASS on every leg, and the cifar Felix real + sim
  pair grades parity PASS (396 vs 400 tasks, same 33.1 s mean). Measured mean task time, cifar / speech: Felix 33 / 292 s
  (fast clients are free again sooner and re-picked; its speed penalty fades out), FedBuff 42 / 299 s, Oort and Oort* 151 / 165 s (44 / 66 tasks in the leg),
  REFL 11 / 95 s, FedDance 12 / 83 s.

**Design record: streaming data model (FX-N120, FX-N123; operator 10-21; paper methodology).**
- *Chosen:* `data_streaming.mode: usage` + `trace: <the leg's trace>` (Q-STREAM, operator 10-21; `harness_pool.usage_hp`) with
  `FLAME_TRACE_OFFSET_S=86400`.
  - *Who generates data, when:* a device generates data while it is in use = not AVL_TRAIN on **its own leg trace** (syn_X_3st or
    MobiPerf; in MobiPerf AVL_TRAIN is the charging state), accruing continuously through each in-use session. A trace with no in-use
    time (syn_0) grows linearly over the span. syn_0 runs two variants: static full data (`s0`) and linear growth (`s0u`).
  - *How much:* visible(t) = in-use seconds so far / all in-use seconds in the 6-day span, × the client's Dirichlet shard. Within a
    device, data ∝ time in use (a 3 h session brings 6× a 30 min one). Across devices the totals stay the shard: a device in use
    more gets its data sooner, not more of it. Without `trace`, the trainer's MobiPerf device drives growth (FX-N120's original).
  - *Clock:* the trace's own (real time, no compression), shared with availability. Every leg starts 24 h into the trace, so
    availability and data both read trace time t + 24 h, and a client starts with the data its usage generated by then (no
    seeded initial share).
  - *Assumption stated in the paper:* a device off the charger is in use, and its data generation is proportional to time in
    use. The trace records no screen state; battery drain (B below) supports that usage intensity is jagged.
- *Rejected, and why:*
  - *events* (the old stream): 9 equal chunks at uniform random times. Every client grows alike in expectation, so even an oracle
    gains nothing (PR40a), and it is not derived from any trace.
  - *One lump at the end of each off-charger session* (the first `usage` draft): new data appears only when charging starts, the
    moment the device becomes trainable. An eval in AVL_EVAL then never sees new data (0% of sessions vs 64% with continuous
    accrual), which rules out sensing by construction, and only 37 / 75 clients grow in a 12 h leg (vs 71).
  - *B, battery drain:* one equal lump per observed 75 → 50% drain (the gap between the two variants' threshold crossings; median
    156 min, p25-p90 68-495 min, within-device CV 0.69). Too sparse: no device gets data in the first 6 h, 11 / 75 none in 6
    days, 3 lumps per device.
  - *C, session length × drain rate:* behaves like the session-length model (only 213 / 1600 sessions have a measured drain, and
    per-device scaling cancels against the fixed shard), at the cost of an extra assumption.
  - *Trainable time as generation time:* charging devices are mostly idle, so this inverts the usage signal.
  - *A seeded initial share (10%, or lower):* a tuned value. At 0-1% all 75 clients start with < 5 samples; at any seeded share
    starting at trace time 0, the median client gets no new data in 12 h (devices sit in one long off-charger stretch early on).
    The 24 h offset derives the start instead: median 16% visible.
  - *Other offsets:* 0 h leaves every client with ~1 sample; 48 h / 72 h start at 27% / 43% visible with less growth left
    (median ×1.14 / ×1.05 in 12 h, end-lump numbers). 24 h is the first day boundary with enough data to train on.
  - *Compressing the trace into the leg (earlier draft: 24 trace hours per 3 h leg):* distorts the burst structure; legs run
    ≥ 12 h of trace time instead.
- *Measured properties (75 cifar clients, offline, start +24 h, continuous):* median visible 16% at the start; 71 / 75 clients
  receive new data within 12 h (median growth ×1.5); a mid-session eval sees new data in 64% of off-charger sessions; arrivals
  are bursty (81% of a client's data in its busiest 10% of hours) and unpredictable from history (hour-to-hour correlation −0.02);
  who-has-new-data rank agreement, this window vs the next, 0.65 / 0.48 / 0.26 at 1 / 3 / 6 h (session-length model). Total size
  rankings barely move (everyone grows in proportion), so the case for sensing rests on what is new since a client's last train
  (the novelty utility), not on |D|.


*Setup (paper methodology):* CIFAR-10 (n 75 screens / 300 paper) and Google Speech (50 / 100), Dirichlet α 0.1; FedScale task-time model
(compute ∝ samples × device speed + model transfer over device bandwidth, Scope); availability traces syn_0, syn_50_3st, MobiPerf
3-state (AVL_TRAIN / AVL_EVAL / UN_AVL); data grows while a device is in use (`usage` stream tied to its own trace, Scope "Design record");
legs start 24 h into the trace so each client starts with its day-1 data (and the trace's start artifact, one long in-use stretch,
is skipped); every stack trains on 100% of visible data (no holdout, F49 audit); clients draw a fresh random crop + flip per batch (cifar, FX-D233); test eval every 10 min of the leg's clock; per-baseline
task timeout (2 × the longest FedScale task, FX-D198); sims graded against reals for parity (PARITY).

*Baselines (source-faithful, `baseline_reference.yaml`; availability per Decisions "Baseline availability knowledge"):*

| stack | mode | selection | aggregation | local work | client keeps | server keeps | knows of availability |
|---|---|---|---|---|---|---|---|
| **Felix** (`fx`) | async | utility (\|D\| × Oort EMA q, `prop`) among online clients; c ramps 60 → ½ trainable as clients fit their data | FedBuff buffer K 10, rate `new` (staleness + utility; s 0.4 / a 0.25 / b 0.2), averaged model served | 1 epoch | current state (OS signals) | believed state per client (3 states) | now, pushed on every flip; evicts mid-task; no forecast |
| FedBuff | async | uniform | buffered, 1/√(1+τ) | 1 epoch | – | check-in set (T1) | at check-in; departure at timeout |
| FedAvg (ablation) | sync | uniform | FedAvg | 1 step (FedDance's config) | – | check-in set (T1) | same |
| Oort | sync | utility × speed penalty, UCB | YoGi | 20 steps | – | check-in set (T1) | same |
| REFL | sync, stale updates kept | least available first by p (Alg. 1); hold-off off at our n | SAA | 1 step | availability log + Prophet model | p sent at check-in | check-in + forecast |
| REFL* | as REFL | as REFL | SAA | 1 step | – | future trace, 10% of each list dropped (fork's "0.9 accuracy": kept flags always right) | oracle, inflated |
| FedDance | sync | V·I·A + MAB | FedAvg equal | 1 step | loss / accuracy with the update | ≤ 50 check-in rounds per client; V = 1 − exp(−5λ) | check-in rate (T4) |
| FedDance* | as FedDance | as FedDance | as FedDance | 1 step | – | trace online set per round | oracle |

*Availability protocol: implementation record (F42-T1, T3, T4; paper methods text; code is the source of truth):*
- **Who knows what (R27, `avail_knowledge`, `ClientAvailability._believed_state`).** `notify` (Felix): the client pushes every flip
  (real: STATE_UPDATE notify; sim: the trace flip `notify_latency_s` late, 0 by default); Felix filters and evicts mid-task.
  `checkin` (every non-`*` baseline, and Felix's K1 ablation `fck`): the server knows only who is waiting in its check-in pool; busy
  picks are known busy; a departure is learned only at task failure / the per-baseline timeout (FX-D198); no eviction. `oracle` (`*`):
  the trace's online set at selection (FedScale `getFeasibleClients`, REFL client_manager.py:256), no eviction. `none`: blind.
- **Check-in protocol (`flame/availability/checkin.py` `CheckinLedger`; Bonawitz et al. 2019 §2-3, pace steering).** A client checks
  in when it turns AVL_TRAIN, when a task ends while AVL_TRAIN, or at its reconnect time if AVL_TRAIN then (else a *missed
  opportunity*: it waits for its next flip). The server pools it while a selection is open, else rejects it with a reconnect time.
  - *Sync stacks (REFL, Oort, FedDance, FedAvg; syncfl / oort engines):* a selection window opens when the version still misses picks
    (`_sync_wants_picks`: no window while wait-K needs none; a mid-round top-up opens one too), lasts W = 0.1 × the round estimate
    μ, and stays open past W until the
    stack's selector has enough pool members (starved windows keep collecting); the selector picks only from the pool at close
    (REFL Alg. 1 "End_Selection_Window"); the rest are rejected. μ = EWMA of round-start gaps (top-up windows excluded), μ ← 0.75 × last gap + 0.25 × μ (REFL
    `deadline_alpha` 0.25, aggregator.py:636), prior μ = the stack's task timeout. Reconnect = next window start estimate (last
    start + μ; if already past, now + μ) + U(0, W).
  - *Async stacks (FedBuff, Felix `fck`; asyncfl engine):* no windows; a check-in is pooled while a concurrency slot is free
    (free at leg start), else rejected. Reconnect = now + m × g × U(0.5, 1.5), m = clients on hold (incl. this one), g = EWMA of the
    time between dispatches (α 0.25, prior = timeout / c): together the held clients return at about the dispatch rate (Bonawitz pace
    steering's goal), and the jitter avoids a thundering herd.
  - *Sim* replays the protocol from each client's trace on the vclock (`advance`); the sync window's W is charged on the vclock
    (selection latency). *Real*: the trainer runs `flame/availability/checkin_client.py` (CHECK_IN on the STATE_UPDATE notify,
    json {p}); the aggregator replies to a rejection with a directed `MessageType.CHECKIN_RECONNECT` {at, mu} on the shared trace
    clock; check-ins received before the first selection are queued (`Channel.set_checkin_handler`).
  - *Telemetry:* `checkin` events (pooled / rejected + reconnect_at / missed / picked, trace clock). *Checks:* EV23 (picks only from the
    pool; sim check-ins only in AVL_TRAIN, misses only outside; nobody back before its reconnect), EV22 (knowledge per arm),
    `tests/availability/test_checkin.py`.
- **REFL as deployed (paper §4.1 Alg. 1, §7; F42-T3).** Each device fits Prophet (1.1.6, cmdstanpy 1.2.4; daily seasonality, no
  weekly / yearly) on its own AVL_TRAIN share in 5-min bins from trace start to the start of the current trace day, refit once per
  trace day (`refl_predictors.ProphetForecaster`); p = mean forecast over [now + μ, now + 2μ], clipped to [0, 1], sent with the
  check-in (sim: computed agg-side from the same per-client history; no future data is read). The selector sorts the pool by p
  ascending (no forecast = 1.0, "assumed available", §4.1 fn. 5; ties by a seeded draw) and takes the top N. Hold-off: off
  (`holdoff_rounds: 0`; the paper's 5 rounds × 13 picks holds out 65 of our 75 clients vs ~6.5% of its 1,000, and the fork has none;
  F42C REFL stalled with every pool member held off). Deviations: paper over fork code for the deployed variant
  (the fork simulates IPS with the future trace: that is REFL*); Prophet trained per device on its own trace instead of on Stunner;
  refit daily (paper: "periodically"); p uses the server's μ as the slot (§7 query a = (μ, 2μ)). REFL* = the fork's oracle
  (`TraceOracle`, avail_priority 1, 0.9 knockout, no hold-off).
- **FedDance (F42-T4):** its predictor V_m = 1 − exp(−5λ) counts per-round check-ins (paper Alg. 1); deployed = the check-in pool
  at each selection (`checkin_source: selection` under `checkin` knowledge); FedDance* = the trace's online set.
- **FedAvg (ablation):** FedDance's experiment and training config with `utility: random` (debug_run.sh), i.e. uniform selection
  from the check-in pool + FedAvg equal weights, 1 local step.
- **FedBuff*:** FedBuff with the trace's online set at selection (labelled oracle).

*Story (proposal):* availability changes faster than any forecast; a system that **reacts** to it at each layer beats one that
predicts it. Felix (1) learns state the moment it changes and frees the slot (vs check-in staleness, forecasts, timeouts), (2) fills
each slot with the most learning per sample among online clients, (3) takes a large server step weighted by staleness and
utility, (4) shrinks concurrency to the trainable pool once clients fit their data, so bytes buy choice, not staleness. Each layer is one ablation (K1-K4); the end-to-end
claim (K5) is time and bytes to accuracy.

*Candidate claims (each names its experiment, metric and ablation):*
- **K1 · React, don't predict (C1):** notify + evict wastes fewer slots and misses fewer opportunities than check-in (+ REFL's forecast,
  FedDance's V), matching the oracles, with O(1) state. Metrics: belief mismatch (T2), pick survival, missed opportunities, timeouts / h,
  wasted slot-h / h, Felix's own cost (evicted partial work), messages / h, TTA. Ablations: Felix with check-in only, Felix blind.
- **K2 · Learning per sample (C3, `prop`):** U* and TTA vs uniform picks; top-20% share, worst-10% client accuracy (`client_accuracy.py`).
- **K3 · Step (rate `new`):** time to 50% / target vs 1/√(1+τ) and 1/(1+τ) at the same selection (F46, F48R).
- **K4 · Concurrency follows the pool (c ramp):** GB and time to target vs c 60 on MobiPerf (F48L: +1.1 pp at 46% GB); picks~size, τ.
- **K5 · End to end:** time and bytes to target vs every baseline; s0 static, s0u, s50u, MobiPerf × both datasets; Felix's loss from
  s0u → s50u smallest.
- **K6 · Robustness (F42e):** silent drops p ∈ {1, 5, 10}%, notify delay / loss.
- **K7 · Overheads:** notify messages / h vs check-ins / h, probe bytes share, aggregator ms per selection.
- **K8 · Scale:** reference n 300 cifar / 100 speech on long sims (D5).

Sources: K1-K4 from S1 + F43; K5, K7 from F43, then D5's 48 h runs; K6 from F42e; K8 from D5.

**Signals (general, privacy-safe: scalars the server already gets, or update geometry it already sees):** per client: task time split
(compute s / sample, transfer s), availability history, loss utility u_i, data size n_i, belief age, ‖Δ_i‖ and cos(Δ_i, Δ̄) (not
secure-aggregation compatible). Per aggregator: held-out loss / accuracy (probes), accuracy per update, staleness, buffer depth, slot
occupancy, update disagreement, belief error.

## 7. Takeaways for the reader (each backed by §5; the paper's teaching points)

1. **Time to accuracy = updates needed ÷ update rate (I-G1).** Every gain is one or the other, and the cells separate them:
   without unavailability Felix wins only by needing fewer updates (2.24×); with it, half comes from keeping slots busy (1.70×);
   on speech `s50u` all of it does. A method that only raises throughput cannot save bytes.
2. **Bytes follow updates, not time.** Each update costs one model round trip, so bytes to target = U* × model size. Felix saves
   9-69% GB vs FedBuff at the highest common accuracy exactly where it needs fewer updates (F49). The byte floor belongs to whoever puts the most learning in one
   round trip: Oort (600 local samples) reaches 40-50% on 7-14× fewer bytes, 4-12× later. Time and bytes are a frontier, not one axis.
3. **Awareness pays only inside an async engine.** FedBuff idles (98-239k starvation events) waiting for check-ins; Felix's notify +
   evict keeps slots busy. But giving a sync stack the oracle (REFL*, FedDance*) changes ≤ 1.5 pp: the round barrier, not knowledge,
   is what holds it back.
4. **Selection decides which data the federation learns from.** Utility per |D| picks big shards (picks~size +0.92) and runs fewer,
   longer, more valuable tasks; speed-first picks small shards and caps accuracy (F45). Choice exists only when c ≪ trainable:
   at c ≥ the online pool every selector is uniform (MobiPerf `fn` picks = FedBuff's).
5. **Throughput first, choice once the data is fit (c ramp).** Early every update helps; once clients fit their data (φ falls) extra
   slots buy staleness and bytes, so c shrinks toward ½ the trainable pool: −50% GB on MobiPerf at equal or better time. Where
   clients never fit their data (speech, 12 h) the ramp stays out of the way by itself.
6. **The server step sets the pace; averaging cleans the noise.** Rate `new`'s large step is Felix's early lead (F46); serving the
   EMA of versions turns a noisy fast iterate into a stable model (+4-13 pp on speech, swing 2-4× smaller): Polyak averaging, for FL.
7. **Staleness was a symptom of overfitting.** Without augmentation the same τ gap cost 1.2 pp; with it Felix runs 1.6× FedBuff's
   staleness and ends +4.2 pp. Fix memorization before blaming asynchrony.
8. **What does not work, and why:** more local epochs (drift on α 0.1 shards), a staleness-discounted step 1/(1+τ) (too small),
   φ-scaled steps (φ falls while accuracy climbs), fixed c ∝ availability (gives up early throughput), forecasting availability
   (REFL ≈ REFL*), UCB exploration (±0.3 pp).
9. **Compare at the target, not at a horizon.** A horizon snapshot ranks curves mid-flight; the paper's question is when each
   stack reaches the accuracy that matters and what it paid by then (time and bytes at the hit, FELIX Decisions).
10. **Honest limits:** below ~50% the 1-step sync stacks and Oort spend fewer bytes; speech sync baselines are transfer-bound and
   near chance for hours under the 5% K rule; speech Felix lags FedBuff in its first 2.5 h; 1 seed, 12 h; no stack reaches the
   final target (cifar 80%, speech 88%) in 12 h, so F49's ratios are at the highest common accuracy until FB1 reports at the hit.

