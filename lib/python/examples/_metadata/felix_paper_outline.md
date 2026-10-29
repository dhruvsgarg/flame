# Felix paper outline (MLSys) — arguments only

Built on [PAPER_TEMPLATE.md](PAPER_TEMPLATE.md) Parts A-B (P# = its
moves) and Part E (shared rules); neither is repeated here. Evidence:
[FELIX_INSIGHTS.md](FELIX_INSIGHTS.md) §5 (findings), §6 (methods,
baselines table), §7 (takeaways);
[FELIX_READINESS.md](FELIX_READINESS.md) Decisions (`fx` layer table),
Scoreboard, rejected-ideas table. **Figures (10-29):** all one column,
FB2 seed 1, built and indexed in `paper_felix/README.md`; style rules
in [plot_guideline.md](plot_guideline.md). Plots say **FLuX**
(`SYS_NAME` in `paper_felix/plot_config.py`, one line to rename); text
may say Felix until the name is final. Figure callouts use % for
accuracy deltas (absolute points): the prose must match. **Scope
now:** title → §5 design. Theory, implementation, evaluation,
ablations come after the front matter is final. **Framing:** one
system, one paper; the current implementation is Felix, no old-vs-new
comparison. **Numbers:** end to end = FB2 seed 1, both datasets, all
11 stacks (FELIX_READINESS Scoreboard; data
`paper_felix/figs/expts/results_{cifar10,google_speech}.json`); seeds
2-3 ⚠ future. Every other number is F47-F49 (12 h, old targets;
MobiPerf Felix tainted by FX-N157) ⚠ re-read on FB2.

## 0. Open choices (decide before drafting prose)

Decided (10-25): **eligible clients**, **client selection**,
**Felix-X** mechanism names, **2 eligibility states**, **claim:
eligibility is hard to forecast**, used in every section. Decided
10-29: accuracy-first hierarchy, no system changes (Pace stays), bytes
phrasing, speech MobiPerf target 84%. Title: "FLuX: Eligibility-Driven
Federated Learning Under Churn" (closed).

| item | proposal | alternatives |
|---|---|---|
| title (decided, closed) | **FLuX: Eligibility-Driven Federated Learning Under Churn** (as in `main_mlsys.tex`) | — |
| eligible clients (decided) | a client is **eligible** while its device meets the training criteria: idle, charging, unmetered network (Bonawitz et al. 2019). Two states only: eligible or not (online but not eligible = not eligible). The **eligible pool** = clients eligible now. Define once in §2 ¶2; never "available" for this | — |
| decomposition | time to accuracy = updates to accuracy ÷ update throughput (statistical × system efficiency, Oort §2); bytes to accuracy = dispatched tasks × model round-trip bytes = update throughput × (1 + discard share) × round trip × time to accuracy | — |
| mechanism names (decided scheme; ablation legends Felix−X) | **Felix-Notify** (push-based availability + eviction) · **Felix-Select** (utility-proportional client selection, \|D\| × EMA loss) · **Felix-Pace** (availability-adaptive concurrency) · **Felix-Agg** (staleness-tolerant aggregation, large server step) | descriptive only: push-based availability tracking · utility-proportional selection · adaptive concurrency · staleness-tolerant aggregation |
| streams (decided 10-25) | **setting, not hero:** part of the realistic setup (usage-tied streams) and methodology, kept wherever the setting is described; not in the title, the tension or the contributions, no stream-specific motivation subsection. Evidence (⚠ FX-N157 taints MobiPerf; F47 `fn_aug` vs `ctl_aug`, time / GB to 65%): `s0` static 1.9× / −58%, `s0u` stream 2.0× / −55%, `s50u` stream + unavailability 2.7× / −36%, `smu` 1.24× / −9%: the gain grows with unavailability, not streaming; the one stream-specific mechanism (c by loss phase) was refuted as a pacer (FX-T89). Caveats: 1 seed, at 65% not the final target, `fn` (≈ `fn_chr` on syn traces). Eval claim: "gains hold on static and streaming data" (add `s0` to the 48 h ladder / FB1 cells) | revisit only if a final-target run shows a clearly larger edge on streams |
| eligibility states (decided 10-25) | **eligible / not eligible** everywhere; no tiers. Code's AVL_EVAL = not eligible in text (feeds only unused on-device evals, FX-D165). One methods line: "traces also mark online-not-eligible time; every stack treats it as not eligible" | — |
| eligibility predictability (decided 10-25: claim it; FX-N147) | **Claim:** the deployed forecasters (REFL's per-device Prophet, FedDance's check-in Poisson V_m, both as run in our baselines) add nothing over the naive guess "eligible now stays eligible". **Method (MobiPerf, n 300, days 1-5, h 5-60 min):** ground truth = the trace (did the client stay eligible?); each predictor is scored against it: AUC (ranking; truth = 1.0, coin = 0.5) and Brier (probability error; truth = 0). **Q1, who is eligible in [t+h, t+2h]:** naive AUC 0.97-0.84, FedDance 0.92-0.84, Prophet 0.68-0.69. **Q2, which eligible client stays for h:** naive can't rank (0.5), Prophet 0.57-0.58, FedDance 0.65-0.67; both Brier worse than the base rate (0.37-0.53 / 0.12-0.29 vs 0.05-0.24). **Why:** eligibility memory fades within hours (corr 0.64 at 1 h, 0.12-0.19 at 12-24 h). syn_50 is memoryless by design (all AUC 0.50): control only. **Wording:** "forecasts add nothing over eligible now", not "unpredictable" | if a forecaster beats the naive guess: drop the claim, keep "scarce and shifting" |
| availability knowledge per stack (measure: FX-N147) | §3.2 table: what each stack knows and how accurate it is: check-in (exact at check-in, blind to departures until timeout), REFL Prophet and FedDance V_m forecasts (error from FX-N147), oracle `*` (exact at selection, blind mid-task). Component microbenchmark (P21) in eval | — |
| bytes claim (decided 10-29, closed; FB2: cifar `s50u` +34% GB vs FedBuff, speech −3% to −18%) | phrase it by the metric hierarchy: accuracy attainment first, then time, then bytes. "FLuX is the only stack to reach the target in all 8 settings; FedBuff, the strongest baseline, misses the 3 hardest (CIFAR-10 MobiPerf, speech syn_50 and MobiPerf). Where FedBuff also reaches it, FLuX gets there 1.1-1.7× sooner and moves up to 33% less data in 4 of the 5 settings." The one cell with more bytes (cifar `s50u`, +34%) is named in the evaluation: same accuracy, 1.13× sooner. Never lead with bytes | — |
| target per trace (decided 10-26; rule: FELIX_READINESS Decisions "Reporting target per trace") | report at the FL bound where it is attainable (cifar syn traces 80%) and at the highest level any stack holds within the 120 h trace on harder deployments (cifar MobiPerf 75%; speech MobiPerf **84%**, decided 10-29, FX-N168 closed): "baselines reach 80% on the less difficult deployments; on MobiPerf with high heterogeneity only 75% is attainable, so we use it as the representative target"; appendix: speedup / GB vs target 60-80% | one target for all cells (all miss on MobiPerf) |
| sync baselines' lr decay (decided 10-26, FX-N163; call out in the evaluation setup + deviations table) | off for REFL, Oort, FedDance, FedAvg (and `*`), as FedBuff and Felix: their per-round decay is tuned for ~1-2k rounds and stalls them at 48-56% over our 120 h traces (≥ 10× the rounds); off lifts them +24-26 pp at 24 h | keep source decay (baselines look broken) |
| not contributions (applied to every stack) | averaged (EMA) model served; per-batch augmentation (bug fix, FX-D233); FedBuff's buffer K 10 | — |
| **no system changes (decided 10-29)** | FLuX is frozen as run (FB2). Swap arms that match or beat it on a MobiPerf accuracy (c ∝ eligible 77.8%, fixed c 30 77.0%, check-in 76.5%, uniform 76.3% vs FLuX 76.0%; cifar only, seed 1) are reported, then justified. **Framing for §5.3 and the ablation text:** (1) every swap arm keeps FLuX's other mechanisms and beats every baseline on accuracy, so the result shows the insight is robust: sizing participation from the live pool wins in more than one form, not only with one tuned rule. (2) c ∝ eligible itself needs the live pool, which only Notify's reports provide; no baseline can run it. (3) Show what φ buys at early accuracy levels from F10b (time to 50-70%). (4) If F10b shows no early advantage, state it as a limit: one seed, cifar only (speech ablations were not run), and the simpler pool-proportional rule is a valid variant of the same design | change Pace (rejected: no new implementation for this submission) |
| **success-metric hierarchy (decided 10-29)** | **accuracy reached > time to it > bytes to it**, in every figure, table, callout and sentence. Higher accuracy justifies more time or bytes, unless a non-FLuX stack reaches the same accuracy in less time and fewer bytes. Callouts read in that order: "+x% accuracy, N× faster, M× fewer GB". Accuracy is comparable only where runs go the full 120 h (MobiPerf, both datasets, and every miss); syn_* runs stop at the target, so there every hit ties on accuracy and time, then bytes, decide | — |

## 1. The spine (every section restates a slice of this)

- **Setting:** FL serves user-facing models whose accuracy target
  moves with users; devices train only while eligible (~20% at a time,
  MobiPerf); data is generated while devices are in use, i.e. when
  they are not eligible.
- **Metric decomposition (Oort's statistical × system efficiency,
  P19):** time to accuracy = updates to accuracy ÷ update throughput;
  measured after the run, unknown to the server at training time.
- **Root cause (P2):** existing systems fix participation (cohort or
  concurrency) at a static, population-sized N while the eligible pool
  changes; async systems learn departures only at timeout; forecasters
  (REFL, FedDance) guess an eligible pool that is hard to predict;
  selection, concurrency and aggregation are tuned separately though
  they are coupled.
- **Tension (P1):** keeping devices busy (system efficiency) vs
  choosing whom to train and keeping updates fresh (statistical
  efficiency). More concurrency fills slots but, as it nears the
  eligible pool, selection degenerates to uniform, staleness grows (τ
  ≈ c / K) and more tasks sit on devices about to leave; less
  concurrency keeps choice but idles capacity when every update still
  helps. The balance point moves on two clocks: the eligible pool
  (minutes to hours) and the training phase (quantity early, quality
  late). Fixing one layer shifts the cost to another (removing idle
  slots raised staleness 4.8 → 7.8, −1.2 pp under 1/√(1+τ)), so the
  fix is cross-stack.
- **Insight:** drive selection, concurrency and aggregation together
  from the live eligible pool, reported by the clients.
- **Evidence the insight pays: no slot waits on a departed client (FB2
  seed 1, `stall` figure + table, `results_*.json` `slot_time_h`).**
  Every stack's slot time splits into useful compute, discarded
  compute, held by a departed client, finished-but-waiting (sync
  barrier) and idle.
  - *Felix* loses 0-49 s of slot time per departure (it evicts on the
    client's report) and computes in 98.9-100% of the slot time it
    holds; 0-2% of its slot time is lost on every cell. Speech MobiPerf
    also shows 18% of free capacity whose eligible clients all await
    their own update's commit (FedBuff\* 13%; F-table "slot use").
  - *FedBuff* learns a departure only at its timeout, so each one
    holds a slot 6-21 min (378-1,262 s). That is 30-59% of its slot
    time held or idle: 1,678 of 4,861 slot-h on cifar `s50u`, 978 of
    2,400 on speech `s50u`. The oracle (FedBuff\*) is no better
    (37-49%), because clients leave after selection.
  - *Sync stacks* lose 51-1,367 s per departure. They also spend about
    as much slot time finished and waiting at the barrier (or on the
    1.3× over-commit, FedDance / FedAvg: 485-592 slot-h discarded on
    cifar) as computing.
  - On syn traces Felix turns the same eligible time into 1.6-2.2×
    FedBuff's useful compute per hour (cifar `s50u` 59 vs 37 slot-h /
    h; speech `s50u` 18 vs 8): the system-efficiency half of its
    time-to-accuracy gain. On MobiPerf Felix deliberately runs fewer
    tasks (c ramps to ½ the eligible pool: cifar 27 vs 42 slot-h / h)
    and still reaches the target first. This is where its bytes saving
    comes from (2,277 vs FedBuff's 6.0k GB at 120 h).
  - Candidate contribution wording: "Felix never waits on a client
    that has left: departures cost it seconds of slot time, not the
    6-21 min timeout every check-in stack must pay."

| symptom (§3) | factor hit | mechanism (§5) | ablation arm (later) |
|---|---|---|---|
| round waits for slowest pick; oracle doesn't help sync | system | async engine (FedBuff base) | sync baselines, `*` arms |
| async idles on check-ins and dead slots | system | Felix-Notify | A1 `b1`, K1 `fck` |
| held uploads = half of staleness | statistical | Felix-Notify (evict) + Felix-Agg | A1, A3 |
| keeping slots busy raises staleness | statistical | Felix-Agg | A3 `fn` |
| availability removes choice before slots | statistical, bytes | Felix-Pace | A4 `fn_chr` |
| speed-first picks lower value per update | statistical, bytes | Felix-Select | A2 `uo_o` |

## Figures: placement and what the text must explain (10-29; supersedes the per-section figure notes below)

All built (FB2 seed 1, overlap check silent); regenerate with `scripts/paper_plots.py` (data) and `paper_felix/make_figs.py`
(trace, schematics, analytic); index `paper_felix/README.md`. Each `.tex` slot keeps its `\figplaceholder{F-id}` until wired in.
Rules the prose follows: accuracy before time before bytes (§0); §3 shows baselines only (FLuX from §4); trace names as in
`table_headline.tex`: **Syn-0% / Syn-20% / Syn-50% / MobiPerf**; callout deltas are absolute accuracy (%).

| F-id · slot | file | message | the text / caption must explain |
|---|---|---|---|
| F1 · §1 `fig:teaser` | `expts/teaser` | only FLuX reaches 75% on cifar MobiPerf: +1.8%, 1.4× faster, 2.8× fewer GB vs FedBuff; +3.7%, 1.7× vs FedDance | "+1.8%" = absolute accuracy (76.0 vs 74.1 peak); arrows = time / GB for FLuX to reach each baseline's peak; curves smoothed |
| F2 · §3.1 `figure*` (a-e) | `paper/s3_motivation/fig2_trace` | the pool's size is steady but its members churn; windows are short next to a task; forecasts ≤ "eligible now" | (a) two denominators: eligible = % of all clients (16%), new = % of the eligible pool not eligible 1 h earlier (30%). (b) strip: dot = median epoch task, tick = slowest, × = task timeout per dataset (`task_timeout.py`); 60% of MobiPerf windows end before the slowest speech task: backs intro ¶2 "a task can outlast an eligible window". (c) 0.64 at 1 h, 0.12 at 12 h. (d, e) AUC; Syn-50% is memoryless by design (control); Brier in text |
| F3 · §3.2 | `expts/slots_base` | eligible slot time is lost to barriers and departed clients: FedBuff 30-58%, FedBuff\* 27-48%, sync 51-77% | on MobiPerf FedBuff's loss is mostly **idle (no check-in yet)**; "held until timeout" dominates only on Syn traces. Striped = clients awaiting their own update's commit (one task per client per version, every async stack): counted in the bar; the text explains it is inherited, not a FLuX loss. Add the per-departure cost from F8 and: oracle ≈ deployable for async |
| F4 · §3.3 | `expts/fill_slots` | filling every slot buys no accuracy: same 80%, 1.8× the updates / h, 1.1× slower, 2.0× GB, staleness 7.4 → 11.9 | on MobiPerf the bound peaks 74.3% vs FedBuff 74.1% (ladder `b1`); replace the old F44 numbers in §3.3 |
| F5a · §3.4 | `expts/concurrency_base` | FedBuff's fixed c exceeds the eligible pool 81% (cifar) / 69% (speech) of the time | — |
| F5b · §3.4 | `expts/choice_replay` | selection's choice is gone before the slots run out: size-weighted picks beat uniform by +0.45 → +0.02 (MobiPerf), +1.30 → 0.00 (Syn-50%) as c / eligible goes 0.2 → 1 | metric = **Pearson** correlation of picks with data size (digest `corr_size`), not Spearman. Method: trace replay, no training (`scripts/choice_replay.py`), p ∝ \|D\| (q = 1), static shards, device-model task times, slots refilled at once. Markers = measured runs ("validated against runs"); they are FLuX runs at four c, labelled generically so §3 shows no FLuX |
| F6 · §4 `figure*` | `paper/s4_overview/fig_architecture` | one live signal (the pool) drives three coupled controllers; blue = new, grey = FedBuff engine | the figure writes Agg's weight as (1+τ)^−¼ (staleness term only): keep consistent with F11a's full rule. Reports are bytes, models MB (0.35-10 reports / client-h vs FedBuff's 6-125 check-ins, `overhead.md`). Steps 1-8 match §4 ¶2 |
| F7 · §4 `fig:timeline` | `paper/s4_overview/fig_toy_timeline` | FedBuff loses (i) idle slots before check-in, (ii) slots held to timeout, (iii) picks with pool ≤ c; FLuX evicts and refills at once and runs fewer tasks as the pool shrinks | keep mechanism names out (they are §5) |
| F8 · §5.1 `fig:notify` | `expts/stall` | a departure costs FLuX ≤ 1-49 s of slot time vs FedBuff's 6-21 min | accuracy contribution is in F12 (+ Notify) |
| F9 · §5.2 `fig:select` | `expts/picks` (cifar); `expts/picks_speech` (eval) | each slot carries more data: ρ +0.41 vs −0.56 (Syn-50%), ≈ 0 vs −0.30 (MobiPerf); 1.3× / 1.6× samples per update | cifar-only claim: speech is 0.76× / 0.97× samples per update (ρ +0.39 vs −0.18, −0.15 vs −0.17); say so in eval. Fix the F-era §5.2 numbers (40 vs 25, +0.92) |
| F10a · §5.3 `fig:pace` | `expts/concurrency` | FLuX's c starts at c_max, ramps to ½ the pool, then follows it | — |
| F10b · §5.3 | `expts/pace_levels` | the best c moves with the training phase: high c is first to early levels, low c first to the target | **contradicts "FLuX tracks the best c"**: MobiPerf c ∝ eligible reaches 75% at 61.5 h vs FLuX 95.2 h (383 vs 2,277 GB, peak 77.8%); Syn-50% c 30 / c ∝ eligible reach 80% at 63.8 vs 72.0 h. Use the §0 "no system changes" framing; never claim the ramp is optimal. `pace_bytes` backs bytes vs c 60 (+0.7%, 1.1×, 2.5× fewer GB on MobiPerf; identical on Syn-50%) |
| F11 · §5.4 `fig:agg` | (a) `paper/s5_design/fig5_4_agg` · (b) `expts/swap_aggregation` | a milder discount gives higher accuracy, sooner: MobiPerf 76.0% vs 75.2% (1/√(1+τ)) vs 72.6% miss (1/(1+τ)); at τ = 10, 1.7× FedBuff's weight | fix the caption todo (τ = 0 values match the formula) |
| F12 · eval ladder | `expts/ladder` | accuracy climbs with every mechanism (MobiPerf −1.8 / −1.7 / −1.4 / −0.7% vs FLuX's 76.0%) | on Syn-50% (all reach 80%) + Notify / + Select are 1.3× slower than FLuX while FedBuff is 1.1×: the gain arrives with Agg. `fn` (+ Agg, c 60) on MobiPerf stopped at 103 h: its 75.3% peak is not strictly comparable |
| F13 · eval swaps | `expts/swap_availability`, `swap_selection`, `swap_concurrency` | FLuX variants vs FLuX, accuracy first | swaps match or beat FLuX: availability / selection +0.1 to +0.4% (one seed, within noise; say so), concurrency +1.0% (c 30) and +1.9% (c ∝ eligible), not noise. Report, then justify with §0 "no system changes" |
| eval · slot use | `expts/slots` (with FLuX) | FLuX loses 0-2% of slot time on every cell except speech MobiPerf (20% = 2% lost + 18% awaiting own commit) | FLuX computes 98.9-100% of the slot time it occupies. Striped "awaiting own commit" (speech MobiPerf: FLuX 18%, FedBuff\* 13%): free slots whose eligible clients all have an uncommitted update; FLuX then has 0.15 dispatchable clients on average. It is the inherited one-task-per-version rule, binds when the pool (~17) is barely above c (~13), and is not model transfer (transfer is inside the task, counted busy) |
| appendix | `expts/oracle_gain`, `rounds`, `staleness`, `curves` | — | `oracle_gain` shows +3 to +19% for speech sync (don't say "barely helps"). `staleness`: 51-75% of FLuX's staleness is uploads withheld while the client is offline (FedBuff 53-79%): not slot holding, and eviction does not remove it |

## Title and abstract (≈ 170 words; B.0)

- **S1 context (25 w):** FL trains user-facing models on devices that
  are eligible only intermittently and generate data while in use.
- **S2 problem (25 w):** keeping devices busy and choosing whom to
  train conflict, and the balance shifts with the eligible pool and
  the training phase.
- **S3 why prior work fails (35 w):** they fix participation at a
  static size, learn departures at timeout or forecast an eligibility
  that is hard to predict, and tune selection, concurrency and
  aggregation separately; even oracle availability changes little.
- **S4 insight (25 w):** drive all three from the live eligible pool,
  reported by the clients.
- **S5 system (30 w):** Felix: Felix-Notify, Felix-Select, Felix-Pace,
  Felix-Agg, one clause each.
- **S6 results (30 w, P3 ranges on time and bytes):** Felix is the
  only stack to reach the target on all 2 datasets × 4 traces (FB2,
  seed 1); no sync stack (incl. oracle arms) reaches it anywhere.
  - cifar (80%; MobiPerf 75%): **1.13-1.23× sooner than FedBuff**
    (65.3-72.0 h vs 77.8-81.0 h) at **−33% to +34% GB** (`s0u` −33%,
    `s20u` −12%, `s50u` +34%); vs FedBuff\* 1.11-1.24×, −32% to +28%.
    MobiPerf: Felix 95.2 h · 2,277 GB; FedBuff ✗ 73.6%.
  - speech (88%; MobiPerf 84%, FX-N168 closed): **1.45-1.69× sooner
    than FedBuff** on `s0u` / `s20u` at −18% / −3% GB (vs FedBuff\*
    1.43-1.71×, −28% / −19%); FedBuff ✗ on `s50u` (85.7%) and `smu`
    (80.7%). At 80% on `smu`: 1.35×, −16% GB.
  - Range for the abstract: **1.1-1.7× sooner than FedBuff, −33% to
    +34% GB**; sync stacks plateau 61-78% (cifar) / 26-76% (speech).

## §1 Introduction (≈ 1.25 pages, ~900 words + Fig 1; argument first, ≤ 3 numbers)

- **Fig 1 teaser (`figs/expts/teaser`, cifar MobiPerf `smu`, built
  10-29):** accuracy vs time and vs GB (log), deployable stacks. No
  baseline reaches 75%, so it compares at each baseline's peak: vs
  FedBuff **1.4× faster, +1.8%, 2.8× fewer GB**; vs FedDance (best
  sync) **1.7× faster, +3.7%**. Sync stacks' bytes left out (fewer GB,
  never reach the accuracy: not a fair ratio).
- **¶1 Stakes (P4) (150 w):** FL trains user-facing models on phones:
  keyboard prediction, speech, ranking (Gboard, Apple, Meta Papaya).
  Their quality is visible to users, and users change: new words,
  apps, habits. So high accuracy is a moving target, not a fixed goal;
  a deployment must reach it fast and reach it again as data shifts ⚠
  cite (continual / periodic retraining in deployed FL).
- **¶2 The constraint (110 w):** devices train only while eligible:
  idle, charging, unmetered (Bonawitz et al. 2019), a small and
  shifting share of the population; and data is generated while
  devices are in use, when they are not eligible. Two resources are
  scarce: eligible time, so every eligible device should be doing
  useful work, and bytes moved (each task downloads and uploads the
  model; users and operators pay for traffic and server bandwidth), so
  work that is discarded or adds little is a cost. Introduce both axes
  here, since ¶7 reports both. Trend: bigger models make tasks longer
  than eligible windows and each task costlier in bytes.
- **¶3 Tension (P1) (140 w), bolded sentence:** **keeping eligible
  devices busy and choosing whom to train pull in opposite directions,
  and the balance shifts as the eligible pool and the training phase
  change.** Argue the two costs and the two clocks (spine); no
  numbers.
- **¶4 Why existing work falls short, by assumption (200 w, one
  number):**
  - *"A round can wait for its slowest participant" (sync):* the
    barrier, not availability knowledge, limits them; even an oracle
    changes little (the one number: ≤ 1.5 pp).
  - *"Participation can be sized once":* sync cohorts and async
    concurrency are fixed for the population; when the eligible pool
    shrinks, selection has no one left to choose from, and when it
    grows, capacity idles.
  - *"Eligibility can be forecast":* REFL and FedDance predict who
    stays eligible; on real traces the forecast is no better than
    "eligible now" (no number here; §3.1).
  - *"Departures are learned at timeout":* slots stay held by devices
    that left, and their late updates carry most of the staleness,
    which straggler fixes cannot see.
  - *"Layers can be tuned separately":* fixing idleness raises
    staleness; aggressive selection lowers throughput.
- **¶5 Insight (P2) (90 w):** one root cause behind the symptoms:
  participation and its consequences are set statically and per layer,
  while eligibility is dynamic. Structural, so no baseline knob fixes
  it.
- **¶6 Ideas (130 w), each ↔ a ¶4 assumption, no mechanism detail:**
  clients push eligibility changes instead of the server forecasting
  them, so the server dispatches only to eligible clients and reclaims
  slots at departure; concurrency follows the eligible pool once early
  throughput stops paying; selection spends each slot on the client
  with the most data left to learn; aggregation takes large steps so
  the staleness of a busy system costs little.
- **¶7 Results (80 w):** headline ranges on time and bytes (the second
  and third numbers). No limits here; they go to the evaluation's
  limits paragraph (Later sections).
- **¶8 Contributions (100 w), ideas only:**
  1. *Insight and analysis:* under dynamic eligibility the
     throughput-vs-selectivity trade-off in FL is dynamic and
     cross-stack; a static participation size cannot track it. A study
     of sync and async stacks shows where the loss goes: the round
     barrier, slots idled or held by departed clients, staleness from
     held uploads, choice lost as concurrency nears the eligible pool
     (§3).
  2. *Systems:* push-based availability with eviction and
     availability-adaptive concurrency, with no forecasting
     (eligibility is hard to forecast, §3.1).
  3. *ML:* utility-proportional client selection and
     staleness-tolerant aggregation for async FL.
  4. *Result:* first to every target, faster than the strongest
     deployed baseline on 2 datasets × 4 traces, with the speedup
     split into statistical × system efficiency; gains hold on static
     and streaming data.

### Task timeout per stack (evaluation setup + deviations table)

**Rule (FX-D198, `flame/launch/task_timeout.py`):** timeout = 1.5 ×
the slowest modeled task of that stack's local work over the reference
population. 1.5× is FedBuff / Papaya's factor (our old 90 s = 1.5 × 60
s). We use the maximum rather than a percentile so that every device
can finish a task. Task time = 3 × samples × device ms/sample + 2 ×
model size / device bandwidth (FedScale model).

| stack · local work | cifar: median task · slowest · timeout | speech: median · slowest · timeout |
|---|---|---|
| Felix, FedBuff · 1 epoch of the client's shard | 26 s · 451 s · **677 s** | 276 s · 1,280 s · **1,920 s** |
| REFL · 1 step (batch 10 / 20) | 8 · 39 · **58 s** | 92 · 457 · **685 s** |
| FedDance, FedAvg · 1 step (batch 24 / 32) | 13 · 46 · **69 s** | 94 · 463 · **695 s** |
| Oort · 20 steps | 80 · 374 · **561 s** | 180 · 611 · **916 s** |

- *Why async is longer:* async stacks train a full local epoch per
  task. With Dirichlet shards, the slowest device holding the largest
  shard sets the maximum, 17× the median task on cifar. A 1-step sync
  task is a few samples plus two model transfers.
- *Why it depends on the dataset:* model size (cifar 2.15 MB vs speech
  29 MB, so transfer dominates speech) and per-sample compute.
- *Why it depends on the stack:* each stack's own local work
  (source-faithful, L34). A shorter common timeout would cut off the
  slow devices of the epoch-based stacks, which could then never
  finish a task.
- *Consequence:* a check-in stack holds a departed client's slot for
  its timeout minus the time already spent. That is the 6-21 min per
  departure above, and it grows with model size and local work,
  exactly where devices are slowest. Felix's push reports make the
  timeout irrelevant (it stays only as a fallback for silent
  departures, FX-N155).

## §2 Background (≈ 0.5 page, ~350 words, text only)

- **¶1 The FL setting (40 w), for a reader new to FL:** a server
  trains one model over many devices without collecting their data: it
  sends the model to selected clients, each trains on its local data
  and returns an update, and the server aggregates the updates into a
  new model version. Cite McMahan et al. 2017, Bonawitz et al. 2019,
  Kairouz et al. 2021.
- **¶2 Terms, briefly (70 w):** client selection; sync rounds (cohort,
  barrier: the round ends when its last pick reports) vs async
  buffered aggregation (FedBuff: concurrency c = tasks in flight,
  buffer K = updates per version); staleness τ = versions an update
  lags, ≈ c / K; local work (steps or epochs per task); task timeout.
  Cited, no figure. Every term reappears in §3-§5.
- **¶3 Device eligibility (110 w):** define *eligible* (§0) and
  contrast it with *online*: eligibility criteria (idle, charging,
  unmetered; Bonawitz et al. 2019); check-in and pace steering; what
  the server knows when: eligible clients at check-in, a departure
  only at timeout. Eligible share from traces cited.
- **¶4 Device data arrives over time (40 w, setting only):** data is
  generated during use, so a client's dataset grows between its
  eligible windows; our evaluation models this (usage-tied streams)
  and also runs static shards ⚠ cite online / continual FL.
- **¶5 Metrics (50 w):** time and bytes to accuracy; statistical ×
  system efficiency (Oort).
- **¶6 Where prior systems stand (40 w + Table 1, P8):** rows = the
  concepts Felix builds on, columns = FedBuff, Oort, REFL, FedDance,
  Felix; each cell says what the system does for that concept, or "–"
  when it has none. No descriptions beyond the cell.

  | concept | FedBuff | Oort | REFL | FedDance | Felix |
  |---|---|---|---|---|---|
  | round barrier | none (async) | yes | yes, stale updates kept | yes | none (async) |
  | how eligibility is learned | check-in; departure at timeout | check-in | check-in + on-device Prophet forecast | check-in counts → Poisson forecast | client reports every change |
  | participation sized by | fixed c | fixed cohort | fixed cohort | fixed cohort | live eligible pool + training phase |
  | selection signal | uniform | loss × speed, UCB | least available first | availability × loss × accuracy gain | data size × loss |
  | staleness rule | 1/√(1+τ) | – | SAA | – | large step, mild discount |

  Shared blind spot: none sizes participation to the live eligible
  pool, and none couples the layers. (Felix's column may name ideas,
  not mechanisms; the mechanisms come in §5.)
- *Not here:* interactions (availability → staleness, availability →
  lost choice) are §3 findings.

## §3 Motivation: the case for cross-stack, availability-driven FL (≈ 1.5 pages, ~800 words + 5 figs)

**Restructured 10-29 (Figure plan F2-F5).** §3.1 eligibility (F2) ·
§3.2 slot time lost to barriers and departed clients (F3; the old §3.2
sync/oracle and §3.4 staleness findings become sentences here) · §3.3
filling slots (F4) · §3.4 choice (F5). The §3.2 and §3.4 bullets below
are evidence for those sentences, not separate subsections. Each
subsection: claim title → setup in one sentence (cell, stacks) →
figure → the "aha" vs what the reader knew → Takeaway (P6). All
figures carry the FL upper bound (P5) ⚠ FX-N146. Baselines get one
clause each, only as needed. **No solution here:** show each gap and
its size; never name or describe a Felix mechanism (notify, eviction,
ramp, rate). Idealized arms are framed as bounds ("FedBuff with every
slot refilled instantly"), not as our system. The reader sees the
opportunity in §3 and learns how it is taken in §4-§5. **Figures:**
small multi-panel rows (Figs 2-6 as 2-3 rows of column-width
subfigures, ~1.2 in tall); one message per panel; no standalone
full-width figure for a single curve.

- **§3.1 Eligibility is scarce, shifting and hard to forecast (P7)
  (160 w + Fig 2 `fig2_trace`, F2).**
  - *Figures (10-29, full 6-day MobiPerf):* **16%** eligible on
    average (not 18-22%: that was the 12 h window), **30%** of
    eligible clients gone 1 h later; autocorrelation **0.64** at 1 h,
    **0.12** at 12 h. Forecast AUC vs "eligible now": Prophet **−0.16
    to −0.30**; for who stays eligible, the best forecaster reaches
    **0.67** and "eligible now" is a coin flip (0.5). Brier stays in
    text / table.
  - *Setup:* MobiPerf and syn_50 traces, 12 h. *Known:* devices are
    often offline; REFL and FedDance forecast who stays eligible.
  - *Aha 1:* the eligible pool is a fifth of the population and
    changes membership within hours; 4 in 10 devices never become
    eligible in 12 h (eligible 18-22%; 57-62% ever train in 12 h).
  - *Aha 2:* eligibility is sticky for minutes but forgets within
    hours (corr 0.64 at 1 h, 0.12-0.19 at 12-24 h), so a daily model
    has little to learn. Scored against the trace, REFL's Prophet
    ranks who will be eligible worse than the naive "eligible now
    stays eligible" (AUC 0.68 vs 0.84-0.97); among eligible clients
    neither forecaster tells well who leaves (AUC 0.57-0.67), and both
    are worse calibrated than the base rate (FX-N147). 30% of eligible
    clients are gone 1 h later.
  - Data arrives in bursts: 81% of a client's data in its busiest 10%
    of hours (setting, one clause). Its hour-to-hour −0.02 is for
    **data arrivals**; never cite it for eligibility.
  - **Takeaway:** the server must work from the live eligible pool,
    not the population or a forecast of it.
- **§3.2 Knowing who is eligible does not help a sync stack; the
  barrier is the limit (150 w + Fig 3 + Table 2).**
  - *Setup:* speech, sync stacks with their deployed availability
    knowledge vs an oracle (`*`). *Aha:* perfect knowledge of who is
    eligible barely helps (REFL* ± 0.6 pp, FedDance* ≤ 1.5 pp; F49 12
    h). ⚠ FB2 120 h contradicts it for sync: oracle arms end 1-18 pp
    higher, yet still 35-76%, far below target; async FedBuff\* ≈
    FedBuff holds (±1.5 h). **Revised claim (operator 10-28):** "in
    most cells oracle knowledge does not help: async and cifar sync
    (±3 pp; FedBuff\* ±1.5 h), because the oracle is exact only at
    selection and the loss comes from clients leaving mid-task (equal
    with and without it) and, for sync, from the round barrier. Where
    it helps (speech sync, +1-18 pp) it is not foresight but protocol
    time: a check-in stack waits 41-109 s between rounds for picks to
    check in (also on syn_0, where no one ever leaves), 20-50% of a
    110-220 s round, so the oracle runs 1.2-2.3× the rounds; on cifar
    that wait is 1-4 s of 18 s." Felix gets the same effect from push
    notifications: no wait, and departures seen at once. REFL ≈ REFL*
    says nothing about REFL's forecaster: the oracle itself gains
    nothing, because rounds wait for the slowest of 6 picks (median
    256 s ≈ 238 s between rounds; 160-186 rounds in 12 h).
  - *Why rounds are long though tasks are short:* the "≤ 60 s per
    client" from FedBuff's setup was our old per-client delay,
    replaced by FedScale's device model (FX-D197). A 1-step speech
    task is 123 s, 93% of it moving the 29 MB model at device
    bandwidth (p50 6.1 Mbps), and the barrier takes the slowest pick's
    tail. Async tasks are longer (an epoch, ~300 s) but overlap 20 at
    a time and never wait for the slowest.
  - *Table 2, knowledge per stack (answers "how good is each
    source"):* check-in = exact at check-in, blind to departures until
    timeout; forecast (REFL Prophet, FedDance V_m) = measured error vs
    trace (FX-N147, from §3.1); oracle = exact at selection, blind
    mid-task. Felix's row waits for §5.
  - *Figures (10-29):* `oracle_gain` (oracle − deployable peak
    accuracy per stack × trace): cifar mostly within ±3% (FedDance\*
    on MobiPerf −10%), speech sync up to +19%, FedBuff ≈ 0 everywhere;
    `rounds` (median round by component, ratio to its oracle: 1.1-2.0×
    on speech MobiPerf, the check-in wait; cifar REFL 0.7×).
  - **Takeaway:** availability knowledge pays only in an engine
    without a barrier.
- **§3.3 Async FL idles half its slots, and filling them does not buy
  accuracy (160 w + Fig 4 slot occupancy).**
  - *Setup:* cifar `s50u`, FedBuff (c 60); FedBuff* (oracle at
    selection); and an idealized bound, FedBuff with every slot
    refilled the instant a client leaves or becomes eligible (arm
    `b1`; not described as a mechanism).
  - *Aha 1:* ~33 of 60 slots are busy: slots wait for check-ins and
    stay held by departed clients until timeout. An oracle at
    selection does not fix it (FedBuff* ≈ FedBuff, ±0.6 pp final, ⚠
    F42 numbers, re-read on FL1 cells): clients leave mid-task.
  - *Aha 2, the bigger gap:* the idealized bound runs 1.8× the updates
    (213k vs 116k) and reaches 50-55% 80 min sooner, yet ends 1.2 pp
    *lower*: staleness rises 4.8 → 7.8 under FedBuff's 1/√(1+τ).
    Doubling useful-looking work buys no final accuracy, and shrinking
    c to stay fresh gives back the speed (c 33: +0.8 pp, FedBuff's
    pace). Lead with this, not with the 1.2 pp.
  - *Figure (`fill_slots`, FB2 cifar `s50u`, replaces the F44 numbers
    above):* the bound `b1` runs **1.8×** FedBuff's updates per hour,
    staleness 7.4 → 11.9, yet is **1.1×** slower to 80% at **2.0×**
    the GB. Claim holds. Slot loss by cause (`slots`) and per
    departure (`stall`, FLuX 26-626× less than FedBuff) back Aha 1.
  - **Takeaway:** fixing idleness alone moves the cost into staleness;
    the layers are coupled.
- **§3.4 Unavailability, not slow devices, causes most staleness (130
  w + Fig 5 τ by source).**
  - *Setup:* FedBuff, staleness split by cause. *Known:* staleness
    comes from stragglers. *Aha:* updates held across a departure are
    1.1-1.2% of updates but 49-51% of total staleness (mean τ
    184-373), from median-speed clients.
  - ⚠ *FB2 contradicts "Felix evicts them" (`staleness`, 10-29):* held
    uploads are 53-74% of FedBuff's staleness but also **51-70% of
    FLuX's** (cifar `s50u` 12,992 late deliveries vs 54 evictions): a
    client that finishes and leaves before upload is delivered on
    return, not evicted. Reword to the FedBuff half only, or drop the
    figure.
  - **Takeaway:** discounting late updates treats the symptom; the
    cause is the departure, which the server learns too late.
- **§3.5 Availability removes choice before it removes capacity (150 w
  + Fig 6 picks~size vs c / eligible).**
  - *Setup:* MobiPerf, c 60 vs ~50-67 eligible of 300. *Known:*
    selection beats random. *Aha:* when c nears the eligible pool,
    every selector picks what FedBuff picks (picks~size −0.45 vs
    −0.46). A smaller fixed c restores choice (+3.8 pp, −92% GB) but
    starts 4× slower.
  - *Figures (FB2 cifar; now F5b `choice_replay`):* ρ(picks, size) falls as c /
    eligible rises (MobiPerf: 0.38 at 0.2 → −0.28 at 1.26, where
    FLuX's selector picks like FedBuff); `concurrency`: FedBuff's
    fixed c exceeds the eligible pool **81%** (cifar) / **69%**
    (speech) of the time on MobiPerf, FLuX's c tracks ½ the pool.
  - **Takeaway:** no fixed concurrency serves both early and late
    training.
- *Dropped §3.6 "streams make loss a stale selection signal" (10-25):*
  no stream-specific mechanism answers it (streams row, §0); the F46
  finding predates FX-D233. At most a discussion line.

## §4 Overview (≈ 0.5 page, ~350 words + architecture fig + toy timeline)

Figures: F6 (architecture redraw) and F7 (toy timeline) in the Figure
plan. ¶2's numbered steps must match F6's scenario numbering.

- **¶1 Design goals, each measured later (90 w):**
  - G1 no idle slot while an eligible client waits (starvation
    events).
  - G2 no forecast, O(1) state per client.
  - G3 only signals the server already sees; no dataset constants.
  - G4 FedBuff-compatible buffer (secure aggregation over K).
  - Non-goals: forecasting, raw data access, tuning K, per-client
    local work, per-client or hardware-specific models (one global
    model for every device).
- **¶2 Architecture (P10) (130 w), numbered steps:** ① client monitors
  its eligibility criteria (OS signals) and reports every change
  (Felix-Notify) → ② server's eligibility ledger (eligible / not, per
  client) → ③ Felix-Select picks an eligible client for each free
  slot, under the concurrency c set by ④ Felix-Pace → ⑤ client trains
  1 epoch, returns update + mean loss (or is evicted if it stops being
  eligible) → ⑥ Felix-Agg buffers updates and commits a version every
  K → ⑦ averaged model served.
- **¶3 What each party sends + privacy (70 w):** one small message per
  eligibility change; the loss scalar and |D| that Oort already sends;
  no new data leaves the device.
- **¶4 Overview by example (P11) (60 w + fig):** 6 clients over 4
  versions, FedBuff vs Felix: an idle slot waiting for a check-in, a
  slot held by a departed client, a pick from a shrunken pool.

## §5 Design (≈ 2.5 pages, ~1500 words + Algorithm 1; each subsection P13: question → observation → mechanism → why not → cost)

Figures: F8 Notify (`stall`), F9 Select (`picks`), F10 Pace
(`concurrency` + per-level time to accuracy), F11 Agg (weight + swap
with accuracy) in the Figure plan. Every "why not" sentence states
accuracy first, then time, then GB.

- **§5.1 Felix-Notify: how does the server learn who can train in time
  to use it? (350 w)**
  - *Observation:* §3.3 idling; Felix learning by check-in loses −4.4
    AUC cifar / −10.0 speech `s50u` (K1, F42b).
  - *Mechanism:* the client reports every eligibility change; the
    ledger keeps eligible / not per client; dispatch to eligible
    clients only; evict a running task when its client stops being
    eligible.
  - *Why not:* check-in (idles, §3.3); forecast (no better than
    "eligible now", §3.1, FX-N147; REFL ≈ REFL* is a barrier effect,
    not evidence about forecasting); oracle at selection (not
    deployable, and FedBuff* ≈ FedBuff because clients leave
    mid-task).
  - *Cost:* 1 message per flip; evicted partial work (measure: K7).
- **§5.2 Felix-Select: whom to put on a free slot? (400 w)**
  - *Observation:* the speed term buys +3-4 pp at 1 h and costs 1.5-2
    pp final; utility alone wins 25 / 28 pairs (F45-F46). The
    pass-averaged EMA loss is the only signal that builds a good
    buffer (top-15 test loss 4× better than random; pre-step batch,
    held-out loss worse than random).
  - *Mechanism:* p_i ∝ |D_i| · q_i, q = Oort's EMA loss over the
    training pass; proportional sampling, no cutoff, no UCB.
    Result: picks~size +0.92, 40 vs 25 samples per update.
  - *Why not:* Oort's cutoff + speed + UCB (fastest to 50%, lowest
    final; UCB ± 0.3 pp); uniform (−1.6 / −1.9 AUC);
    buffer-composition proxies ≈ random.
  - *Cost:* no extra pass (an extra measurement pass cost −14-20%
    throughput, FX-T64).
- **§5.3 Felix-Pace: how many tasks in flight when the eligible pool
  shrinks? (400 w)**
  - *Observation:* §3.5; early every update helps, so every fixed
    bound that restores choice loses early.
  - *Mechanism:* φ = mean received-model loss over recent updates /
    running peak, recomputed every commit; c = φ · c_max + (1 − φ) ·
    min(c_max, ½ eligible), within [K, c_max]. Ramps down within
    minutes; rises with the pool, not φ.
  - *FB2 (`pace_bytes`, 10-29):* vs fixed c = 60 (`fn`) on MobiPerf
    **2.5× fewer GB, 1.1× faster**; on syn_50 the same GB and time
    (pool stays above c). *Old evidence:* MobiPerf −50% GB (659 vs
    1308), same tick to 40-50%, +0.5 pp, starvation 5k vs 209k; a
    no-op on syn traces and on speech (φ stays 0.56-0.91), as
    designed.
  - *Why not:* fixed c ∝ eligible (4× slower start); c by loss phase
    alone (floor 10 from 6-8 h to 48 h); held-out pacing (noise); AIMD
    K (walks to floor). K stays fixed: τ ≈ c / K, and K 10 is
    FedBuff's own choice.
- **§5.4 Felix-Agg: how large a server step under staleness? (300 w)**
  - *Observation:* same selection, K 10: rate `new` / 1/√(1+τ) /
    1/(1+τ) reach 50% at 60-80 / 120-130 / 210-280 min (F46).
    Staleness was a symptom of overfitting: τ 11.7 vs 7.3 still ends
    +4.2 pp with augmentation.
  - *Mechanism:* rate `new` (`fedbuff.weight_factor`: polynomial
    staleness with a 0.25 exponent × loss-utility term; mean weight
    0.61-0.78) + per-sample lr normalization of partial batches
    (halves the stream eval swing).
  - *Why not:* stronger discount, φ-scaled step (−0.9 / −3.2 pp),
    smaller fixed lr (−1.2 to −6.4 AUC), FedAdam / Yogi (extra
    constants).
  - *Figure:* `fig5_4_agg` (analytic weight vs τ): at τ = 10 FLuX-Agg
    gives **1.7×** FedBuff's weight.
- **§5.5 Algorithm 1 (P16) + robustness (50 w):** one box for
  Felix-Notify → Felix-Pace → Felix-Select → Felix-Agg with line-cited
  prose; missed or delayed notifies degrade gracefully (K6, F42e ⚠ to
  measure).

## Evaluation plan (operator 10-25; rules: PAPER_TEMPLATE Part E "Evaluation plan and run pipeline")

**How to read:** Table 2 = every figure / table → claim → runs it
reads; Table 1 = each run set once; a run feeds every figure that
lists it. Execution (tiers, commands, status, paths) lives in
FELIX_READINESS. **First version** = seed 1 of every run a main-text
figure needs; seeds 2-3 and appendix rows after, re-plotting as data
lands.

**Shared setup:** cifar n 300 / speech n 100, Dirichlet α 0.1;
usage-tied data streams on every trace (static only in the appendix);
traces syn_0, syn_20, syn_50, MobiPerf; each stack at its own paper's
K / c ("Cohort and participation"); start +24 h into the 6-day trace
(each client starts with its day-1 data; 0 h leaves ~1 sample per
client, FELIX_INSIGHTS streams design); stop at the target (cifar 80%,
speech 88%; 5-eval mean) or the trace end (120 h); report time and GB
at each cell's reporting target (§0 "target per trace": cifar MobiPerf
75%). Sync baselines run without client lr decay (§0; deviations
table). Bench (B2-validated per dataset: cifar ✅, speech = FX-N149).
**Deployable** = FedBuff, Oort (check-in), REFL (Prophet forecast),
FedDance (check-in Poisson forecast), FedAvg; **oracle** = FedBuff*,
Oort*, REFL*, FedDance*, FedAvg* (the trace's eligible set at
selection); Felix = `fx`. **Ablation cells** = cifar × {syn_50,
MobiPerf} (operator 10-27: ablations on cifar only; speech ablations
skipped, priority 3 below seeds 2-3, operator 10-28).

### Table 2: figures and tables → claims → runs

| figure / table | where | claim it backs | content | cells | runs |
|---|---|---|---|---|---|
| **End to end** | | | | | |
| Teaser figure | Introduction | Felix reaches the target N× sooner at N% fewer GB than every deployable stack | accuracy vs time and vs GB; Felix vs deployable | cifar MobiPerf | End-to-end |
| Headline table | Evaluation: end to end | Felix is first to target on every cell, vs deployable and vs oracle stacks | one `table*`, layout + pipeline in "Headline table plan" below | all 8 | End-to-end |
| Convergence curves | Appendix | Felix's lead holds over the whole run, not only at the target | accuracy vs time per cell; deployable solid, oracle dashed | all 8 | End-to-end |
| Static data table | Appendix | Gains hold without data streams | headline-table row, static shards | syn_0 static × 2 datasets | End-to-end static |
| **Why Felix wins (same runs)** | | | | | |
| Efficiency split table | Evaluation: where the time goes | Each speedup = fewer updates to target (statistical) × more updates per hour (system); which one wins depends on the trace | per stack × cell: updates to target, updates / h, ratios vs Felix | all 8 (main: 4 ablation cells) | End-to-end |
| Slot use figure | Evaluation: where the time goes; Motivation: async idles | Async baselines idle slots waiting for check-ins or held by departed clients; sync stacks wait at the barrier; Felix keeps slots busy | share of slot time busy / idle / held by a departed client, per stack | cifar syn_50, MobiPerf; speech MobiPerf | End-to-end |
| Stall figure + table (`stall`) | Evaluation: where the time goes; Motivation: async idles | Felix loses seconds per departure; check-in stacks hold a slot until timeout (6-21 min), sync stacks also wait at the barrier | raw slot-hours per stack: useful / discarded compute, held by departed, barrier wait, idle; per-departure loss, releases | cifar + speech × syn_50, MobiPerf | End-to-end ✅ |
| Round time figure (10-29: appendix; §3.2 uses F3) | Appendix | Sync rounds are set by the slowest pick, mostly model transfer; an oracle can't shorten them | round time: compute / transfer / waiting for the slowest; sync stacks ± oracle | speech + cifar MobiPerf | End-to-end |
| Staleness by source figure (10-29: appendix; FLuX's staleness is also 51-70% held, so "Felix evicts them" is dropped) | Appendix | In FedBuff most staleness comes from uploads held across a departure | staleness split by cause; FedBuff, FedBuff*, Felix | cifar syn_50, MobiPerf | End-to-end |
| Client picks figure (⚠ FB2: samples / update 1.3-1.6× FedBuff on cifar, 0.8-1.0× on speech) | Evaluation: which clients train | Felix picks clients with the most left to learn; speed-first or uniform picks give less per update | picks vs data size, samples / update, coverage, top-20% share; per stack | 4 ablation cells (rest appendix) | End-to-end |
| Concurrency figure | Evaluation: concurrency; Motivation: choice lost | Felix's c follows the eligible pool and the training phase; a fixed c can't serve both early and late | c, eligible pool, loss phase over time (Felix) vs fixed c | cifar MobiPerf, syn_50; speech MobiPerf | End-to-end |
| Wasted bytes table | Evaluation: communication | Bytes follow updates; timed-out, evicted and late tasks are paid traffic | GB by task fate (aggregated / timed out / evicted / late) per stack | 4 ablation cells (rest appendix) | End-to-end |
| Eligibility knowledge table | Motivation: knowledge per stack | Check-in is blind to departures, forecasts are no better than "eligible now", the oracle is blind mid-task; Felix's reports are exact | per stack: share of dispatches to clients that leave mid-task, departure detection lag; forecast AUC | MobiPerf, syn_50 | End-to-end + forecast study |
| Forecast accuracy figure | Motivation: hard to forecast | Deployed forecasters add nothing over "eligible now" | AUC / Brier vs horizon | MobiPerf, syn_50 | forecast study ✅ |
| Eligibility trace figure | Motivation: scarce and shifting | Eligible pool ≈ 20%, membership churns within hours | pool size, sessions, autocorrelation | all traces | trace only ✅ |
| Overhead table | Evaluation: cost (or appendix) | Eligibility reports cost ≪ 1% of traffic; server work per commit is small | messages per client-hour, report vs model bytes, server time per commit | all 8 | End-to-end |
| **Ablations: design choices** | | | | | |
| Ladder figure (⚠ FB2: on `s50u` + Notify / + Select are 1.3× slower than FLuX, behind FedBuff's 1.1×; the gain arrives with Agg) | Evaluation: what each mechanism adds | Each mechanism adds on top of FedBuff; none alone gives the full gain | **peak accuracy (first column)**, then time / GB to target: FedBuff → + Notify → + Select → + Agg → + Pace (= Felix); see F12 | ablation cells | Ladder + End-to-end |
| Availability swap | Ablations: Felix-Notify | Inside Felix, client reports + eviction beat check-in, a forecast and an oracle at selection | Felix with knowledge = check-in / forecast / oracle vs reports | ablation cells | Swap: availability |
| Selection swap | Ablations: Felix-Select | Utility-proportional selection beats uniform and Oort's (speed + UCB + cutoff) | Felix with each selector | ablation cells | Swap: selection |
| Aggregation swap | Ablations: Felix-Agg | A large step with a mild staleness discount beats 1/√(1+τ) and 1/(1+τ) | Felix with each rule | ablation cells | Swap: aggregation |
| Concurrency swap | Ablations: Felix-Pace | The ramp beats fixed c 60, a small fixed c and c ∝ eligible pool | Felix with each c rule | ablation cells | Swap: concurrency (c 60 = Ladder `fn`) |
| Rejected ideas table | Ablations (rows with an aha); rest appendix | The obvious fixes lose: more local work per task (fewer bytes, slower to target), UCB exploration (ties), forecasting, a stronger staleness discount, c ∝ eligible | one row per idea: Δ time / GB vs Felix, one-line cause | ablation cells | Rejected ideas + the swaps |
| **Ablations: robustness and cost** | | | | | |
| Silent departures figure | Ablations: robustness | Clients that leave without reporting cost Felix little: the task timeout reclaims the slot | time / GB to target vs silent share p ∈ {0, 1, 5, 10}% | ablation cells | Robustness: silent |
| Lost reports / uploads figure | Appendix (after first version) | Lost or late reports and lost uploads degrade Felix gracefully | time to target vs loss p, delay d | ablation cells | Robustness: lost |
| Sensitivity table | Appendix (after first version) | One config for both datasets; results hold under ± changes to the step knobs, c_max, K | time to target per knob ± | 2 cells | Sensitivity |
| Parked | — | new model / dataset (PAPER_TEMPLATE C.1-C.2) | only if time remains | — | — |

### Headline table plan (operator 10-28; to decide → build)

**What each cell reports (priority order):** accuracy first, then
time, then bytes.
- *Reference per cell* = the fastest stack to the cell's target (Felix
  in all 8 cells today; the rule stays generic). Its row shows
  absolute **h · GB**; every other row is relative to it.
- *Stack hit the target:* **slowdown** = stack h / reference h (≥ 1×)
  and **GB ratio** = stack GB / reference GB (< 1 = fewer bytes, e.g.
  cifar `s50u` FedBuff 1.13× · 0.75×).
- *Stack missed:* time and GB ratios are undefined (it never got
  there), so the cell shows **✗ −Δ pp** spanning both sub-columns, Δ =
  target − the stack's peak 5-eval accuracy over 120 h (`peak_acc5`):
  what it gave up, even with more time than Felix had (e.g. cifar
  `s50u` FedAvg ✗ −5.2). Not "−Δ at Felix's hit time"
  (`acc_gain_pp_at_felix_hit_h`): larger, but unfair to stacks that
  were still climbing; use it in prose only.
- *Optional, decide after a mock:* for a miss, a small "(N×)" =
  Felix's time to that stack's peak ÷ the stack's time to it
  (Oort-style "reaches FedAvg's best N× sooner"). Keeps a speed number
  on misses; drop it if the table gets busy. Needs a finer Felix curve
  than the 50/60/70/… buckets in `time_h_to_acc`.
- Not reported: a stack vs the best *baseline* (one reference per cell
  is enough; FedBuff comparisons go in prose).

**Layout (one table, MLSys `table*`, full width):**
- Rows = stacks (12): Felix (absolute) · then 5 baselines, each
  deployable row with its oracle `*` row directly below (FedBuff /
  FedBuff\*, …), so "oracle barely helps" reads per pair.
  `\addlinespace` between baseline pairs; no vertical line for oracle.
- Columns = CIFAR-10 {syn_0, syn_20, syn_50, MobiPerf} │ Speech
  {same}, one vertical rule between datasets only; each trace has 2
  sub-columns (h× · GB×). Header row 2 gives each trace's target (80 /
  80 / 80 / 75 · 88 / 88 / 88 / 84%).
- Style (operator 10-28: no ✗ / grey, the table must tell a story):
  full `\textwidth`, `\small`; Felix row tinted blue; misses = −Δ
  tinted red on one absolute scale (none < 2 pp, then < 5, < 10, < 30,
  ≥ 30), worst per column bold; fewest GB bold; last column "targets
  reached" (Felix 8/8, FedBuff 5/8, sync 0/8). Story: FedBuff near
  misses, cifar sync ~5 pp short, speech sync collapses 20-58 pp.
  Fallback if too wide: two `table` (one per dataset), same rows.

**Pipeline (built 10-28; from the existing results JSONs, no new
runs):** `scripts/paper_json.py` → `results_*.json` → `python
paper_felix/make_tables.py` → `paper_felix/figs/expts/`:
- `table_headline.json`: per cell × stack `ref`, `time_ratio`,
  `gb_ratio`, `acc_gap_pp` (target − `peak_acc5`, misses only).
- `table_headline.tex`: `\input` as is; needs `booktabs, xcolor,
  pifont`. Not compiled here (no LaTeX on the server).
- `numbers.tex`: prose macros (`\fxSpeedupFedBuffMin/Max`,
  `\fxGbChangeFedBuffMin/Max`, per-dataset `\fxCifar…` / `\fxSpeech…`,
  `\fxCellsFirst`), so the abstract and §1 ¶7 never hard-code a value.
- Ratios are computed in `make_tables.py`; it asserts they match
  `paper_json`'s `felix_vs_this.speedup`. Open: the optional "(N×)" on
  misses needs FB2 eval series (the `data/runs/` cache is F49).

### Table 1: run sets (seed 1 = first version; seeds 2-3 for End-to-end and Ladder after every seed-1 run)

| run set | arms | cells | legs / seed | needs before launch | priority |
|---|---|---|---|---|---|
| End-to-end | `fx`, FedBuff, Oort, REFL, FedDance, FedAvg, FedBuff*, Oort*, REFL*, FedDance*, FedAvg* | 2 datasets × 4 traces | 88 | FedAvg* arm + 2 h smoke; syn_20 cell; digest + plots on a smoke | 1 |
| Ladder | `b1` (+ Notify), `uo_o` (+ Select), `fn` (+ Agg, c 60) | ablation cells | 6 | — (arms exist) | 2 (speech: 3, skipped: cifar covers; below seeds 2-3) |
| Swap: concurrency | small fixed c (c 30), c ∝ eligible (`ca`) | ablation cells | 4 | port to `fx` base | 2 (speech: 3, skipped: cifar covers; below seeds 2-3) |
| Swap: availability | `fx` + check-in (no eviction), `fx` + oracle; forecast if cheap | ablation cells | 4-6 | arms on `fx` (forecast = new code ❓) | 2 (speech: 3, skipped: cifar covers; below seeds 2-3) |
| Swap: aggregation | `fx` + 1/√(1+τ), `fx` + 1/(1+τ) | ablation cells | 4 | arms on `fx` | 3 |
| Swap: selection | `fx` + uniform, `fx` + Oort's selector | ablation cells | 4 | arms on `fx` | 3 |
| Rejected ideas | `fx` + UCB, `fx` + 2 local epochs | ablation cells | 4 | arms on `fx` | 3 |
| Robustness: silent | `fx` at p 1 / 5 / 10% silent departures (p 0 = End-to-end) | ablation cells | 6 | new code (F42e) + smoke | 3 |
| End-to-end static | `fx` + 5 deployable, static shards | syn_0 × 2 datasets | 12 | — | 4 (appendix) |
| Seeds 2-3 | End-to-end + Ladder, both datasets | as seed 1 | 100 / seed | every seed-1 run of priorities 1-4 | 5 |
| Robustness: lost · Sensitivity | — | — | — | after first version | 6 |

**First version ≈ 166 legs** (manifest: 152 / seed incl. 12 static;
silent departures not yet in it). **First plots (operator 10-25 late)
= priorities 1-2, 116 legs.** Speech is the bottleneck (1 leg / GPU,
async 10-13 h for 120 h): measure 2 legs / GPU on the speech smoke;
use jayne when free. **Run order (operator 10-27):** seed 1 of every
End-to-end cell (both datasets) first, then seed 1 of priorities 2 → 3
→ 4, then seeds 2-3; within a priority, cifar ahead of speech. Status
10-27: first version done (10-28): End-to-end seed 1 on both datasets
(88 legs) + cifar priorities 2-3; figures and results JSONs populated.
**Seeds 2-3, static and speech ablations: closed for the submission
(operator 10-28);** the paper reports seed 1.
## Later sections (claims they must back; outline after the front matter is final)

- Theory: inherit FedBuff's rate under bounded staleness; state where
  held uploads break the bound (P17 slogan TBD).
- Evaluation (K5): time and GB at the hit vs the fastest deployed
  baseline, all cells, seed 1 (seeds 2-3 closed, operator 10-28), FB2
  6-day runs; U* × R table (P19).
- Ablations: ladder A0-A4 + K1, swap ablations (FX-N145), why-not
  table.
- Limits paragraph at the end of the evaluation (P24; moved from intro
  ¶7): below ~50%, Oort and 1-step sync stacks spend fewer bytes at
  several times the time; speech sync stacks transfer-bound under the
  5% K rule; speech FedBuff leads Felix only in the first hour of
  `s0u` (15.7 vs 9.1% at 1 h; Felix ahead from 2 h, FB2); seeds.
