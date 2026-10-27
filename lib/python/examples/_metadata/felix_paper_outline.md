# Felix paper outline (MLSys) — arguments only

Built on [PAPER_TEMPLATE.md](PAPER_TEMPLATE.md) Parts A-B (P# = its moves) and Part E (shared rules); neither is repeated here.
Evidence: [FELIX_INSIGHTS.md](FELIX_INSIGHTS.md) §5 (findings), §6 (methods, baselines table), §7 (takeaways); [FELIX_READINESS.md](FELIX_READINESS.md)
Decisions (`fx` layer table), Scoreboard, rejected-ideas table.
**Scope now:** title → §5 design. Theory, implementation, evaluation, ablations come after the front matter is final.
**Framing:** one system, one paper; the current implementation is Felix, no old-vs-new comparison.
**Numbers:** cifar end to end = FB2 (seed 1, 80% FL-bound target, FELIX_READINESS Scoreboard); MobiPerf FedBuff, speech and
seeds 2-3 ⚠ pending. Every other number is F47-F49 (12 h, old targets; MobiPerf Felix tainted by FX-N157) ⚠ re-read on FB2.

## 0. Open choices (decide before drafting prose)

Decided (10-25): **eligible clients**, **client selection**, **Felix-X** mechanism names, **2 eligibility states**, **claim:
eligibility is hard to forecast**, used in every section. Title open.

| item | proposal | alternatives |
|---|---|---|
| title (open; availability only, see streams row) | **Felix: Jointly Adapting Client Selection, Concurrency and Aggregation to Intermittent Device Eligibility** | "Felix: Optimizing the Federated Learning Stack for Intermittently Eligible Devices"; "Felix: Fast Federated Learning When Devices Come and Go" |
| eligible clients (decided) | a client is **eligible** while its device meets the training criteria: idle, charging, unmetered network (Bonawitz et al. 2019). Two states only: eligible or not (online but not eligible = not eligible). The **eligible pool** = clients eligible now. Define once in §2 ¶2; never "available" for this | — |
| decomposition | time to accuracy = updates to accuracy ÷ update throughput (statistical × system efficiency, Oort §2); bytes to accuracy = dispatched tasks × model round-trip bytes = update throughput × (1 + discard share) × round trip × time to accuracy | — |
| mechanism names (decided scheme; ablation legends Felix−X) | **Felix-Notify** (push-based availability + eviction) · **Felix-Select** (utility-proportional client selection, \|D\| × EMA loss) · **Felix-Pace** (availability-adaptive concurrency) · **Felix-Agg** (staleness-tolerant aggregation, large server step) | descriptive only: push-based availability tracking · utility-proportional selection · adaptive concurrency · staleness-tolerant aggregation |
| streams (decided 10-25) | **setting, not hero:** part of the realistic setup (usage-tied streams) and methodology, kept wherever the setting is described; not in the title, the tension or the contributions, no stream-specific motivation subsection. Evidence (⚠ FX-N157 taints MobiPerf; F47 `fn_aug` vs `ctl_aug`, time / GB to 65%): `s0` static 1.9× / −58%, `s0u` stream 2.0× / −55%, `s50u` stream + unavailability 2.7× / −36%, `smu` 1.24× / −9%: the gain grows with unavailability, not streaming; the one stream-specific mechanism (c by loss phase) was refuted as a pacer (FX-T89). Caveats: 1 seed, at 65% not the final target, `fn` (≈ `fn_chr` on syn traces). Eval claim: "gains hold on static and streaming data" (add `s0` to the 48 h ladder / FB1 cells) | revisit only if a final-target run shows a clearly larger edge on streams |
| eligibility states (decided 10-25) | **eligible / not eligible** everywhere; no tiers. Code's AVL_EVAL = not eligible in text (feeds only unused on-device evals, FX-D165). One methods line: "traces also mark online-not-eligible time; every stack treats it as not eligible" | — |
| eligibility predictability (decided 10-25: claim it; FX-N147) | **Claim:** the deployed forecasters (REFL's per-device Prophet, FedDance's check-in Poisson V_m, both as run in our baselines) add nothing over the naive guess "eligible now stays eligible". **Method (MobiPerf, n 300, days 1-5, h 5-60 min):** ground truth = the trace (did the client stay eligible?); each predictor is scored against it: AUC (ranking; truth = 1.0, coin = 0.5) and Brier (probability error; truth = 0). **Q1, who is eligible in [t+h, t+2h]:** naive AUC 0.97-0.84, FedDance 0.92-0.84, Prophet 0.68-0.69. **Q2, which eligible client stays for h:** naive can't rank (0.5), Prophet 0.57-0.58, FedDance 0.65-0.67; both Brier worse than the base rate (0.37-0.53 / 0.12-0.29 vs 0.05-0.24). **Why:** eligibility memory fades within hours (corr 0.64 at 1 h, 0.12-0.19 at 12-24 h). syn_50 is memoryless by design (all AUC 0.50): control only. **Wording:** "forecasts add nothing over eligible now", not "unpredictable" | if a forecaster beats the naive guess: drop the claim, keep "scarce and shifting" |
| availability knowledge per stack (measure: FX-N147) | §3.2 table: what each stack knows and how accurate it is: check-in (exact at check-in, blind to departures until timeout), REFL Prophet and FedDance V_m forecasts (error from FX-N147), oracle `*` (exact at selection, blind mid-task). Component microbenchmark (P21) in eval | — |
| bytes claim (⚠ FB2 cifar breaks it: `s50u` +34% GB vs FedBuff; FX-N165) | "fewest GB to every target only FedBuff also reaches"; below that Felix trades bytes for 4-12× time | report time first and bytes as a range (−33% to +34%); or drop bytes from the abstract (weakens P3) |
| target per trace (decided 10-26; rule: FELIX_READINESS Decisions "Reporting target per trace") | report at the FL bound where it is attainable (cifar syn traces 80%) and at the highest level any stack holds within the 120 h trace on harder deployments (cifar MobiPerf 75%): "baselines reach 80% on the less difficult deployments; on MobiPerf with high heterogeneity only 75% is attainable, so we use it as the representative target"; appendix: speedup / GB vs target 60-80% | one target for all cells (all miss on MobiPerf) |
| sync baselines' lr decay (decided 10-26, FX-N163; call out in the evaluation setup + deviations table) | off for REFL, Oort, FedDance, FedAvg (and `*`), as FedBuff and Felix: their per-round decay is tuned for ~1-2k rounds and stalls them at 48-56% over our 120 h traces (≥ 10× the rounds); off lifts them +24-26 pp at 24 h | keep source decay (baselines look broken) |
| not contributions (applied to every stack) | averaged (EMA) model served; per-batch augmentation (bug fix, FX-D233); FedBuff's buffer K 10 | — |

## 1. The spine (every section restates a slice of this)

- **Setting:** FL serves user-facing models whose accuracy target moves with users; devices train only while eligible (~20% at a
  time, MobiPerf); data is generated while devices are in use, i.e. when they are not eligible.
- **Metric decomposition (Oort's statistical × system efficiency, P19):** time to accuracy = updates to accuracy ÷ update throughput;
  measured after the run, unknown to the server at training time.
- **Root cause (P2):** existing systems fix participation (cohort or concurrency) at a static, population-sized N while the eligible
  pool changes; async systems learn departures only at timeout; forecasters (REFL, FedDance) guess an eligible pool that is hard to
  predict; selection, concurrency and aggregation are tuned separately though they are coupled.
- **Tension (P1):** keeping devices busy (system efficiency) vs choosing whom to train and keeping updates fresh (statistical
  efficiency). More concurrency fills slots but, as it nears the eligible pool, selection degenerates to uniform, staleness grows
  (τ ≈ c / K) and more tasks sit on devices about to leave; less concurrency keeps choice but idles capacity when every update still
  helps. The balance point moves on two clocks: the eligible pool (minutes to hours) and the training phase (quantity
  early, quality late). Fixing one layer shifts the cost to another
  (removing idle slots raised staleness 4.8 → 7.8, −1.2 pp under 1/√(1+τ)), so the fix is cross-stack.
- **Insight:** drive selection, concurrency and aggregation together from the live eligible pool, reported by the clients.

| symptom (§3) | factor hit | mechanism (§5) | ablation arm (later) |
|---|---|---|---|
| round waits for slowest pick; oracle doesn't help sync | system | async engine (FedBuff base) | sync baselines, `*` arms |
| async idles on check-ins and dead slots | system | Felix-Notify | A1 `b1`, K1 `fck` |
| held uploads = half of staleness | statistical | Felix-Notify (evict) + Felix-Agg | A1, A3 |
| keeping slots busy raises staleness | statistical | Felix-Agg | A3 `fn` |
| availability removes choice before slots | statistical, bytes | Felix-Pace | A4 `fn_chr` |
| speed-first picks lower value per update | statistical, bytes | Felix-Select | A2 `uo_o` |

## Title and abstract (≈ 170 words; B.0)

- **S1 context (25 w):** FL trains user-facing models on devices that are eligible only intermittently and generate data while in use.
- **S2 problem (25 w):** keeping devices busy and choosing whom to train conflict, and the balance shifts with the eligible pool
  and the training phase.
- **S3 why prior work fails (35 w):** they fix participation at a static size, learn departures at timeout or forecast an
  eligibility that is hard to predict, and tune selection, concurrency and aggregation separately; even oracle availability changes little.
- **S4 insight (25 w):** drive all three from the live eligible pool, reported by the clients.
- **S5 system (30 w):** Felix: Felix-Notify, Felix-Select, Felix-Pace, Felix-Agg, one clause each.
- **S6 results (30 w, P3 ranges on time and bytes):** Felix is first to every target on 2 datasets × 4 traces.
  - cifar (FB2, seed 1, to 80%): **1.13-1.23× sooner than FedBuff** (65.3-72.0 h vs 77.8-81.0 h) at **−33% to +34% GB**
    (`s0u` −33%, `s20u` −12%, `s50u` +34%); vs FedBuff\* 1.11-1.24×, −32% to +28%. MobiPerf at its 75% target: Felix 95.2 h ·
    2,277 GB; FedBuff ⚠ running (72.1% at 80.8 h). Speech ⚠ pending; sync stacks ⚠ rerun with decay off (FX-N163). For reference,
    all cells at 75%: 1.45-1.84× FedBuff, −44% to +10% GB. F49's 1.26-2.6× / −55% to +4% were at the old, lower targets.

## §1 Introduction (≈ 1.25 pages, ~900 words + Fig 1; argument first, ≤ 3 numbers)

- **Fig 1 teaser (hardest realistic cell: cifar MobiPerf `smu`):** accuracy vs time and vs GB, all stacks; annotated "N× sooner, N% fewer GB" ⚠.
- **¶1 Stakes (P4) (150 w):** FL trains user-facing models on phones: keyboard prediction, speech, ranking (Gboard, Apple, Meta
  Papaya). Their quality is visible to users, and users change: new words, apps, habits. So high accuracy is a moving target, not a
  fixed goal; a deployment must reach it fast and reach it again as data shifts ⚠ cite (continual / periodic retraining in deployed FL).
- **¶2 The constraint (110 w):** devices train only while eligible: idle, charging, unmetered (Bonawitz et al. 2019), a small and
  shifting share of the population; and data is generated while devices are in use, when they are not eligible. Two resources are
  scarce: eligible time, so every eligible device should be doing useful work, and bytes moved (each task downloads and uploads the
  model; users and operators pay for traffic and server bandwidth), so work that is discarded or adds little is a cost. Introduce
  both axes here, since ¶7 reports both. Trend: bigger models make tasks longer than eligible windows and each task costlier in bytes.
- **¶3 Tension (P1) (140 w), bolded sentence:** **keeping eligible devices busy and choosing whom to train pull in opposite
  directions, and the balance shifts as the eligible pool and the training phase change.** Argue the two costs and
  the two clocks (spine); no numbers.
- **¶4 Why existing work falls short, by assumption (200 w, one number):**
  - *"A round can wait for its slowest participant" (sync):* the barrier, not availability knowledge, limits them; even an oracle
    changes little (the one number: ≤ 1.5 pp).
  - *"Participation can be sized once":* sync cohorts and async concurrency are fixed for the population; when the eligible pool
    shrinks, selection has no one left to choose from, and when it grows, capacity idles.
  - *"Eligibility can be forecast":* REFL and FedDance predict who stays eligible; on real traces the forecast is no better than
    "eligible now" (no number here; §3.1).
  - *"Departures are learned at timeout":* slots stay held by devices that left, and their late updates carry most of the staleness,
    which straggler fixes cannot see.
  - *"Layers can be tuned separately":* fixing idleness raises staleness; aggressive selection lowers throughput.
- **¶5 Insight (P2) (90 w):** one root cause behind the symptoms: participation and its consequences are set statically and per
  layer, while eligibility is dynamic. Structural, so no baseline knob fixes it.
- **¶6 Ideas (130 w), each ↔ a ¶4 assumption, no mechanism detail:** clients push eligibility changes instead of the server
  forecasting them, so the server dispatches only to eligible clients and reclaims slots at departure; concurrency follows the eligible pool once early throughput stops paying;
  selection spends each slot on the client with the most data left to learn; aggregation takes large steps so the staleness of a busy
  system costs little.
- **¶7 Results (80 w):** headline ranges on time and bytes (the second and third numbers). No limits here; they go to the
  evaluation's limits paragraph (Later sections).
- **¶8 Contributions (100 w), ideas only:**
  1. *Insight and analysis:* under dynamic eligibility the throughput-vs-selectivity trade-off in FL is dynamic and cross-stack; a
     static participation size cannot track it. A study of sync and async stacks shows where the loss goes: the round barrier, slots
     idled or held by departed clients, staleness from held uploads, choice lost as concurrency nears the eligible pool (§3).
  2. *Systems:* push-based availability with eviction and availability-adaptive concurrency, with no forecasting (eligibility is
     hard to forecast, §3.1).
  3. *ML:* utility-proportional client selection and staleness-tolerant aggregation for async FL.
  4. *Result:* first to every target, faster than the strongest deployed baseline on 2 datasets × 4 traces, with the speedup split
     into statistical × system efficiency; gains hold on static and streaming data.

## §2 Background (≈ 0.5 page, ~350 words, text only)

- **¶1 The FL setting (40 w), for a reader new to FL:** a server trains one model over many devices without collecting their data:
  it sends the model to selected clients, each trains on its local data and returns an update, and the server aggregates the updates
  into a new model version. Cite McMahan et al. 2017, Bonawitz et al. 2019, Kairouz et al. 2021.
- **¶2 Terms, briefly (70 w):** client selection; sync rounds (cohort, barrier: the round ends when its last pick reports) vs async
  buffered aggregation (FedBuff: concurrency c = tasks in flight, buffer K = updates per version); staleness τ = versions an update
  lags, ≈ c / K; local work (steps or epochs per task); task timeout. Cited, no figure. Every term reappears in §3-§5.
- **¶3 Device eligibility (110 w):** define *eligible* (§0) and contrast it with *online*: eligibility criteria (idle, charging, unmetered; Bonawitz et al. 2019); check-in and pace
  steering; what the server knows when: eligible clients at check-in, a departure only at timeout. Eligible share from traces cited.
- **¶4 Device data arrives over time (40 w, setting only):** data is generated during use, so a client's dataset grows between its
  eligible windows; our evaluation models this (usage-tied streams) and also runs static shards ⚠ cite online / continual FL.
- **¶5 Metrics (50 w):** time and bytes to accuracy; statistical × system efficiency (Oort).
- **¶6 Where prior systems stand (40 w + Table 1, P8):** rows = the concepts Felix builds on, columns = FedBuff, Oort, REFL,
  FedDance, Felix; each cell says what the system does for that concept, or "–" when it has none. No descriptions beyond the cell.

  | concept | FedBuff | Oort | REFL | FedDance | Felix |
  |---|---|---|---|---|---|
  | round barrier | none (async) | yes | yes, stale updates kept | yes | none (async) |
  | how eligibility is learned | check-in; departure at timeout | check-in | check-in + on-device Prophet forecast | check-in counts → Poisson forecast | client reports every change |
  | participation sized by | fixed c | fixed cohort | fixed cohort | fixed cohort | live eligible pool + training phase |
  | selection signal | uniform | loss × speed, UCB | least available first | availability × loss × accuracy gain | data size × loss |
  | staleness rule | 1/√(1+τ) | – | SAA | – | large step, mild discount |

  Shared blind spot: none sizes participation to the live eligible pool, and none couples the layers. (Felix's column may
  name ideas, not mechanisms; the mechanisms come in §5.)
- *Not here:* interactions (availability → staleness, availability → lost choice) are §3 findings.

## §3 Motivation: the case for cross-stack, availability-driven FL (≈ 1.5 pages, ~800 words + 5 figs)

Each subsection: claim title → setup in one sentence (cell, stacks) → figure → the "aha" vs what the reader knew → Takeaway (P6).
All figures carry the FL upper bound (P5) ⚠ FX-N146. Baselines get one clause each, only as needed.
**No solution here:** show each gap and its size; never name or describe a Felix mechanism (notify, eviction, ramp, rate). Idealized
arms are framed as bounds ("FedBuff with every slot refilled instantly"), not as our system. The reader sees the opportunity in §3
and learns how it is taken in §4-§5.
**Figures:** small multi-panel rows (Figs 2-6 as 2-3 rows of column-width subfigures, ~1.2 in tall); one message per panel; no
standalone full-width figure for a single curve.

- **§3.1 Eligibility is scarce, shifting and hard to forecast (P7) (160 w + Fig 2: pool, sessions, memory | forecast AUC / Brier by horizon; `paper_felix/`).**
  - *Setup:* MobiPerf and syn_50 traces, 12 h. *Known:* devices are often offline; REFL and FedDance forecast who stays eligible.
  - *Aha 1:* the eligible pool is a fifth of the population and changes membership within hours; 4 in 10 devices never become
    eligible in 12 h (eligible 18-22%; 57-62% ever train in 12 h).
  - *Aha 2:* eligibility is sticky for minutes but forgets within hours (corr 0.64 at 1 h, 0.12-0.19 at 12-24 h), so a daily model
    has little to learn. Scored against the trace, REFL's Prophet ranks who will be eligible worse than the naive "eligible now stays
    eligible" (AUC 0.68 vs 0.84-0.97); among eligible clients neither forecaster tells well who leaves (AUC 0.57-0.67), and both are
    worse calibrated than the base rate (FX-N147).
    30% of eligible clients are gone 1 h later.
  - Data arrives in bursts: 81% of a client's data in its busiest 10% of hours (setting, one clause). Its hour-to-hour −0.02 is for
    **data arrivals**; never cite it for eligibility.
  - **Takeaway:** the server must work from the live eligible pool, not the population or a forecast of it.
- **§3.2 Knowing who is eligible does not help a sync stack; the barrier is the limit (150 w + Fig 3 + Table 2).**
  - *Setup:* speech, sync stacks with their deployed availability knowledge vs an oracle (`*`). *Aha:* perfect knowledge of who is
    eligible barely helps (REFL* ± 0.6 pp, FedDance* ≤ 1.5 pp). REFL ≈ REFL* says nothing about REFL's forecaster: the oracle itself
    gains nothing, because rounds wait for the slowest of 6 picks (median 256 s ≈ 238 s between rounds; 160-186 rounds in 12 h).
  - *Why rounds are long though tasks are short:* the "≤ 60 s per client" from FedBuff's setup was our old per-client delay, replaced
    by FedScale's device model (FX-D197). A 1-step speech task is 123 s, 93% of it moving the 29 MB model at device bandwidth (p50 6.1
    Mbps), and the barrier takes the slowest pick's tail. Async tasks are longer (an epoch, ~300 s) but overlap 20 at a time and
    never wait for the slowest.
  - *Table 2, knowledge per stack (answers "how good is each source"):* check-in = exact at check-in, blind to departures until
    timeout; forecast (REFL Prophet, FedDance V_m) = measured error vs trace (FX-N147, from §3.1); oracle = exact at selection, blind
    mid-task. Felix's row waits for §5.
  - **Takeaway:** availability knowledge pays only in an engine without a barrier.
- **§3.3 Async FL idles half its slots, and filling them does not buy accuracy (160 w + Fig 4 slot occupancy).**
  - *Setup:* cifar `s50u`, FedBuff (c 60); FedBuff* (oracle at selection); and an idealized bound, FedBuff with every slot refilled
    the instant a client leaves or becomes eligible (arm `b1`; not described as a mechanism).
  - *Aha 1:* ~33 of 60 slots are busy: slots wait for check-ins and stay held by departed clients until timeout. An oracle at
    selection does not fix it (FedBuff* ≈ FedBuff, ±0.6 pp final, ⚠ F42 numbers, re-read on FL1 cells): clients leave mid-task.
  - *Aha 2, the bigger gap:* the idealized bound runs 1.8× the updates (213k vs 116k) and reaches 50-55% 80 min sooner, yet ends
    1.2 pp *lower*: staleness rises 4.8 → 7.8 under FedBuff's 1/√(1+τ). Doubling useful-looking work buys no final accuracy, and
    shrinking c to stay fresh gives back the speed (c 33: +0.8 pp, FedBuff's pace). Lead with this, not with the 1.2 pp.
  - **Takeaway:** fixing idleness alone moves the cost into staleness; the layers are coupled.
- **§3.4 Unavailability, not slow devices, causes most staleness (130 w + Fig 5 τ by source).**
  - *Setup:* FedBuff, staleness split by cause. *Known:* staleness comes from stragglers. *Aha:* updates held across a departure are
    1.1-1.2% of updates but 49-51% of total staleness (mean τ 184-373), from median-speed clients.
  - **Takeaway:** discounting late updates treats the symptom; the cause is the departure, which the server learns too late.
- **§3.5 Availability removes choice before it removes capacity (150 w + Fig 6 picks~size vs c / eligible).**
  - *Setup:* MobiPerf, c 60 vs ~50-67 eligible of 300. *Known:* selection beats random. *Aha:* when c nears the eligible pool, every
    selector picks what FedBuff picks (picks~size −0.45 vs −0.46). A smaller fixed c restores choice (+3.8 pp, −92% GB) but starts
    4× slower.
  - **Takeaway:** no fixed concurrency serves both early and late training.
- *Dropped §3.6 "streams make loss a stale selection signal" (10-25):* no stream-specific mechanism answers it (streams row, §0);
  the F46 finding predates FX-D233. At most a discussion line.

## §4 Overview (≈ 0.5 page, ~350 words + architecture fig + toy timeline)

- **¶1 Design goals, each measured later (90 w):**
  - G1 no idle slot while an eligible client waits (starvation events).
  - G2 no forecast, O(1) state per client.
  - G3 only signals the server already sees; no dataset constants.
  - G4 FedBuff-compatible buffer (secure aggregation over K).
  - Non-goals: forecasting, raw data access, tuning K, per-client local work, per-client or hardware-specific models (one global
    model for every device).
- **¶2 Architecture (P10) (130 w), numbered steps:** ① client monitors its eligibility criteria (OS signals) and reports every change
  (Felix-Notify) → ② server's eligibility ledger (eligible / not, per client) → ③ Felix-Select picks an eligible client for each
  free slot, under the concurrency c set by ④ Felix-Pace → ⑤ client trains 1 epoch, returns update + mean loss (or is evicted if it
  stops being eligible) → ⑥ Felix-Agg buffers updates and commits a version every K → ⑦ averaged model served.
- **¶3 What each party sends + privacy (70 w):** one small message per eligibility change; the loss scalar and |D| that Oort already sends;
  no new data leaves the device.
- **¶4 Overview by example (P11) (60 w + fig):** 6 clients over 4 versions, FedBuff vs Felix: an idle slot waiting for a check-in,
  a slot held by a departed client, a pick from a shrunken pool.

## §5 Design (≈ 2.5 pages, ~1500 words + Algorithm 1; each subsection P13: question → observation → mechanism → why not → cost)

- **§5.1 Felix-Notify: how does the server learn who can train in time to use it? (350 w)**
  - *Observation:* §3.3 idling; Felix learning by check-in loses −4.4 AUC cifar / −10.0 speech `s50u` (K1, F42b).
  - *Mechanism:* the client reports every eligibility change; the ledger keeps eligible / not per client; dispatch to eligible
    clients only; evict a running task when its client stops being eligible.
  - *Why not:* check-in (idles, §3.3); forecast (no better than "eligible now", §3.1, FX-N147; REFL ≈ REFL* is a barrier effect,
    not evidence about forecasting); oracle at selection (not deployable, and FedBuff*
    ≈ FedBuff because clients leave mid-task).
  - *Cost:* 1 message per flip; evicted partial work (measure: K7).
- **§5.2 Felix-Select: whom to put on a free slot? (400 w)**
  - *Observation:* the speed term buys +3-4 pp at 1 h and costs 1.5-2 pp final; utility alone wins 25 / 28 pairs (F45-F46). The
    pass-averaged EMA loss is the only signal that builds a good buffer (top-15 test loss 4× better than random; pre-step batch,
    held-out loss worse than random).
  - *Mechanism:* p_i ∝ |D_i| · q_i, q = Oort's EMA loss over the training pass; proportional sampling, no cutoff, no UCB.
    Result: picks~size +0.92, 40 vs 25 samples per update.
  - *Why not:* Oort's cutoff + speed + UCB (fastest to 50%, lowest final; UCB ± 0.3 pp); uniform (−1.6 / −1.9 AUC); buffer-composition proxies ≈ random.
  - *Cost:* no extra pass (an extra measurement pass cost −14-20% throughput, FX-T64).
- **§5.3 Felix-Pace: how many tasks in flight when the eligible pool shrinks? (400 w)**
  - *Observation:* §3.5; early every update helps, so every fixed bound that restores choice loses early.
  - *Mechanism:* φ = mean received-model loss over recent updates / running peak, recomputed every commit;
    c = φ · c_max + (1 − φ) · min(c_max, ½ eligible), within [K, c_max]. Ramps down within minutes; rises with the pool, not φ.
  - *Evidence (⚠ F48, FX-N157: re-read on FB2):* MobiPerf −50% GB (659 vs 1308), same tick to 40-50%, +0.5 pp, starvation 5k vs 209k; a no-op on syn traces and on
    speech (φ stays 0.56-0.91), as designed.
  - *Why not:* fixed c ∝ eligible (4× slower start); c by loss phase alone (floor 10 from 6-8 h to 48 h); held-out pacing (noise);
    AIMD K (walks to floor). K stays fixed: τ ≈ c / K, and K 10 is FedBuff's own choice.
- **§5.4 Felix-Agg: how large a server step under staleness? (300 w)**
  - *Observation:* same selection, K 10: rate `new` / 1/√(1+τ) / 1/(1+τ) reach 50% at 60-80 / 120-130 / 210-280 min (F46).
    Staleness was a symptom of overfitting: τ 11.7 vs 7.3 still ends +4.2 pp with augmentation.
  - *Mechanism:* rate `new` (`fedbuff.weight_factor`: polynomial staleness with a 0.25 exponent × loss-utility term; mean weight
    0.61-0.78) + per-sample lr normalization of partial batches (halves the stream eval swing).
  - *Why not:* stronger discount, φ-scaled step (−0.9 / −3.2 pp), smaller fixed lr (−1.2 to −6.4 AUC), FedAdam / Yogi (extra constants).
- **§5.5 Algorithm 1 (P16) + robustness (50 w):** one box for Felix-Notify → Felix-Pace → Felix-Select → Felix-Agg with line-cited prose; missed or delayed
  notifies degrade gracefully (K6, F42e ⚠ to measure).

## Evaluation plan (operator 10-25; rules: PAPER_TEMPLATE Part E "Evaluation plan and run pipeline")

**How to read:** Table 2 = every figure / table → claim → runs it reads; Table 1 = each run set once; a run feeds every figure
that lists it. Execution (tiers, commands, status, paths) lives in FELIX_READINESS. **First version** = seed 1 of every run a
main-text figure needs; seeds 2-3 and appendix rows after, re-plotting as data lands.

**Shared setup:** cifar n 300 / speech n 100, Dirichlet α 0.1; usage-tied data streams on every trace (static only in the appendix);
traces syn_0, syn_20, syn_50, MobiPerf; each stack at its own paper's K / c ("Cohort and participation"); start +24 h into the
6-day trace (each client starts with its day-1 data; 0 h leaves ~1 sample per client, FELIX_INSIGHTS
streams design); stop at the target (cifar 80%, speech 88%; 5-eval mean) or the trace end (120 h); report time and GB at each
cell's reporting target (§0 "target per trace": cifar MobiPerf 75%). Sync baselines run without client lr decay (§0; deviations
table). Bench
(B2-validated per dataset: cifar ✅, speech = FX-N149). **Deployable** = FedBuff, Oort (check-in), REFL (Prophet forecast),
FedDance (check-in Poisson forecast), FedAvg; **oracle** = FedBuff*, Oort*, REFL*, FedDance*, FedAvg* (the trace's eligible set at
selection); Felix = `fx`. **Ablation cells** = cifar × {syn_50, MobiPerf} (operator 10-27: ablations on cifar only; speech ablations priority 3, appendix if run).

### Table 2: figures and tables → claims → runs

| figure / table | where | claim it backs | content | cells | runs |
|---|---|---|---|---|---|
| **End to end** | | | | | |
| Teaser figure | Introduction | Felix reaches the target N× sooner at N% fewer GB than every deployable stack | accuracy vs time and vs GB; Felix vs deployable | cifar MobiPerf | End-to-end |
| Headline table | Evaluation: end to end | Felix is first to target on every cell, vs deployable and vs oracle stacks | rows dataset × trace (8); cols Felix │ 5 deployable ║ 5 oracle; h · GB to target, or ✗ + end accuracy; best bold | all 8 | End-to-end |
| Convergence curves | Appendix | Felix's lead holds over the whole run, not only at the target | accuracy vs time per cell; deployable solid, oracle dashed | all 8 | End-to-end |
| Static data table | Appendix | Gains hold without data streams | headline-table row, static shards | syn_0 static × 2 datasets | End-to-end static |
| **Why Felix wins (same runs)** | | | | | |
| Efficiency split table | Evaluation: where the time goes | Each speedup = fewer updates to target (statistical) × more updates per hour (system); which one wins depends on the trace | per stack × cell: updates to target, updates / h, ratios vs Felix | all 8 (main: 4 ablation cells) | End-to-end |
| Slot use figure | Evaluation: where the time goes; Motivation: async idles | Async baselines idle slots waiting for check-ins or held by departed clients; sync stacks wait at the barrier; Felix keeps slots busy | share of slot time busy / idle / held by a departed client, per stack | cifar syn_50, MobiPerf; speech MobiPerf | End-to-end |
| Round time figure | Motivation: the barrier | Sync rounds are set by the slowest pick, mostly model transfer; an oracle can't shorten them | round time: compute / transfer / waiting for the slowest; sync stacks ± oracle | speech + cifar MobiPerf | End-to-end |
| Staleness by source figure | Motivation: staleness; Evaluation | In FedBuff most staleness comes from uploads held across a departure; Felix evicts them | staleness split by cause; FedBuff, FedBuff*, Felix | cifar syn_50, MobiPerf | End-to-end |
| Client picks figure | Evaluation: which clients train | Felix picks clients with the most left to learn; speed-first or uniform picks give less per update | picks vs data size, samples / update, coverage, top-20% share; per stack | 4 ablation cells (rest appendix) | End-to-end |
| Concurrency figure | Evaluation: concurrency; Motivation: choice lost | Felix's c follows the eligible pool and the training phase; a fixed c can't serve both early and late | c, eligible pool, loss phase over time (Felix) vs fixed c | cifar MobiPerf, syn_50; speech MobiPerf | End-to-end |
| Wasted bytes table | Evaluation: communication | Bytes follow updates; timed-out, evicted and late tasks are paid traffic | GB by task fate (aggregated / timed out / evicted / late) per stack | 4 ablation cells (rest appendix) | End-to-end |
| Eligibility knowledge table | Motivation: knowledge per stack | Check-in is blind to departures, forecasts are no better than "eligible now", the oracle is blind mid-task; Felix's reports are exact | per stack: share of dispatches to clients that leave mid-task, departure detection lag; forecast AUC | MobiPerf, syn_50 | End-to-end + forecast study |
| Forecast accuracy figure | Motivation: hard to forecast | Deployed forecasters add nothing over "eligible now" | AUC / Brier vs horizon | MobiPerf, syn_50 | forecast study ✅ |
| Eligibility trace figure | Motivation: scarce and shifting | Eligible pool ≈ 20%, membership churns within hours | pool size, sessions, autocorrelation | all traces | trace only ✅ |
| Overhead table | Evaluation: cost (or appendix) | Eligibility reports cost ≪ 1% of traffic; server work per commit is small | messages per client-hour, report vs model bytes, server time per commit | all 8 | End-to-end |
| **Ablations: design choices** | | | | | |
| Ladder figure | Evaluation: what each mechanism adds | Each mechanism adds on top of FedBuff; none alone gives the full gain | time / GB to target: FedBuff → + Notify → + Select → + Agg → + Pace (= Felix) | ablation cells | Ladder + End-to-end |
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

### Table 1: run sets (seed 1 = first version; seeds 2-3 for End-to-end and Ladder after every seed-1 run)

| run set | arms | cells | legs / seed | needs before launch | priority |
|---|---|---|---|---|---|
| End-to-end | `fx`, FedBuff, Oort, REFL, FedDance, FedAvg, FedBuff*, Oort*, REFL*, FedDance*, FedAvg* | 2 datasets × 4 traces | 88 | FedAvg* arm + 2 h smoke; syn_20 cell; digest + plots on a smoke | 1 |
| Ladder | `b1` (+ Notify), `uo_o` (+ Select), `fn` (+ Agg, c 60) | ablation cells | 6 | — (arms exist) | 2 (speech: 3) |
| Swap: concurrency | small fixed c (c 30), c ∝ eligible (`ca`) | ablation cells | 4 | port to `fx` base | 2 (speech: 3) |
| Swap: availability | `fx` + check-in (no eviction), `fx` + oracle; forecast if cheap | ablation cells | 4-6 | arms on `fx` (forecast = new code ❓) | 2 (speech: 3) |
| Swap: aggregation | `fx` + 1/√(1+τ), `fx` + 1/(1+τ) | ablation cells | 4 | arms on `fx` | 3 |
| Swap: selection | `fx` + uniform, `fx` + Oort's selector | ablation cells | 4 | arms on `fx` | 3 |
| Rejected ideas | `fx` + UCB, `fx` + 2 local epochs | ablation cells | 4 | arms on `fx` | 3 |
| Robustness: silent | `fx` at p 1 / 5 / 10% silent departures (p 0 = End-to-end) | ablation cells | 6 | new code (F42e) + smoke | 3 |
| End-to-end static | `fx` + 5 deployable, static shards | syn_0 × 2 datasets | 12 | — | 4 (appendix) |
| Seeds 2-3 | End-to-end + Ladder, both datasets | as seed 1 | 100 / seed | every seed-1 run of priorities 1-4 | 5 |
| Robustness: lost · Sensitivity | — | — | — | after first version | 6 |

**First version ≈ 166 legs** (manifest: 152 / seed incl. 12 static; silent departures not yet in it). **First plots
(operator 10-25 late) = priorities 1-2, 116 legs.** Speech is the bottleneck (1 leg / GPU, async 10-13 h for 120 h):
measure 2 legs / GPU on the speech smoke; use jayne when free. **Run order (operator 10-27):** seed 1 of every End-to-end cell
(both datasets) first, then seed 1 of priorities 2 → 3 → 4, then seeds 2-3; within a priority, cifar ahead of speech. Status 10-27:
cifar seed 1 priorities 1-3 done; speech seed 1 End-to-end (44 legs) running, one End-to-end plot first; everything else waits
(FELIX_READINESS node plan).
## Later sections (claims they must back; outline after the front matter is final)

- Theory: inherit FedBuff's rate under bounded staleness; state where held uploads break the bound (P17 slogan TBD).
- Evaluation (K5): time and GB at the hit vs the fastest deployed baseline, all cells, 3 seeds, FB1 6-day runs; U* × R table (P19).
- Ablations: ladder A0-A4 + K1, swap ablations (FX-N145), why-not table.
- Limits paragraph at the end of the evaluation (P24; moved from intro ¶7): below ~50%, Oort and 1-step sync stacks spend fewer
  bytes at several times the time; speech sync stacks transfer-bound under the 5% K rule;
  speech FedBuff leads Felix for the first ~2.5 h (⚠ FB1S smoke: Felix 22% vs FedBuff 16% at 2 h; re-read); seeds.
