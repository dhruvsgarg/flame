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

**FL1 log (FELIX FX-N141; cifar bench n 300, 12 h; standing: one crisp line per worked / failed mechanism, for motivation / eval).**
- *Unaware async idles, and the idling hides staleness (F44 `s50u`, `FL1_1023_1700`).* FedBuff waits on check-ins (98k
  slot-starvation events) and holds slots of departed clients to the 903 s timeout: ~33 of 60 slots busy, 116k updates. Notify /
  evict (`b1`) keeps all 60 busy: 213k updates, 50-55% 80 min sooner, but staleness 7.8 vs 4.8 under the same 1/√(1+τ) rate costs
  1.2 pp final. Motivation: awareness buys speed; aggregation must absorb the staleness it creates. Test: F45 `b1_c33`.
- *More local epochs = drift, not staleness (F44 `b1gw_e3`).* Same staleness (3.9), 45-60% fewer updates, equal accuracy to 4 h
  then −3 pp by 8 h on non-IID α 0.1; GB −28 to −58%. Local work needs per-client control, not a global epoch count.
- *Weight decay on clients does nothing for async cifar (F44 `b1gw`):* ±0.5 pp final vs `b1`.
- *Static data has no plateau (F44 `s0`):* every arm 64-68.5% at 12 h; the 54-57% stream plateau is the stream's data, not Felix.
- *Speed alone selects small shards (F45 smoke, 1 h):* picks~size ρ −0.40, 18 samples / update vs population 27; adding |D| q
  flips it (+0.53); utility alone is all size (+0.91) at 36% fewer updates. Speed alone starves big-data clients; utility
  alone starves fast ones.

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
is skipped); 10% of each client's arrivals held out (no stack trains on them); test eval every 10 min of the leg's clock; per-baseline
task timeout (2 × the longest FedScale task, FX-D198); sims graded against reals for parity (PARITY).

*Baselines (source-faithful, `baseline_reference.yaml`; availability per Decisions "Baseline availability knowledge"):*

| stack | mode | selection | aggregation | local work | client keeps | server keeps | knows of availability |
|---|---|---|---|---|---|---|---|
| **Felix** | async | learning per slot-second among clients that can finish (C3 + bal) | FedBuff buffer, rate `new` (staleness + utility), averaged model served | 1 epoch | current state (OS signals) | believed state per client (3 states) | now, pushed on every flip; evicts mid-task; no forecast |
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
each slot with the most learning per second among clients that can finish, (3) sizes the server step to staleness and stability,
(4) stops spending bytes once held-out data says the model stopped learning. Each layer is one ablation (K1-K4); the end-to-end
claim (K5) is time and bytes to accuracy.

*Candidate claims (each names its experiment, metric and ablation):*
- **K1 · React, don't predict (C1):** notify + evict wastes fewer slots and misses fewer opportunities than check-in (+ REFL's forecast,
  FedDance's V), matching the oracles, with O(1) state. Metrics: belief mismatch (T2), pick survival, missed opportunities, timeouts / h,
  wasted slot-h / h, Felix's own cost (evicted partial work), messages / h, TTA. Ablations: Felix with check-in only, Felix blind.
- **K2 · Learning per slot-second (C3 + bal):** R × s̄ and TTA vs e0; top-20% share, worst-10% client accuracy (`client_accuracy.py`).
- **K3 · Step control (C4, C5):** eval swing and AUC on speech vs l3b; K trajectory.
- **K4 · Bytes only while learning (C8 pacing):** bytes to matched accuracy and final accuracy vs l5 / e0; held-out loss; probe cost
  (~1% bytes). Later: a dynamic probe rate.
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
