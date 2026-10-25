# Felix paper outline (MLSys) — arguments only

Built on [PAPER_TEMPLATE.md](PAPER_TEMPLATE.md) Parts A-B (P# = its moves) and Part E (shared rules); neither is repeated here.
Evidence: [FELIX_INSIGHTS.md](FELIX_INSIGHTS.md) §5 (findings), §6 (methods, baselines table), §7 (takeaways); [FELIX_READINESS.md](FELIX_READINESS.md)
Decisions, contribution ledger, "Paper: motivation and evaluation from FL1", rejected-ideas table.
**Scope now:** title → §5 design. Theory, implementation, evaluation, ablations come after the front matter is final.
**Framing:** one system, one paper; the current implementation is Felix, no old-vs-new comparison.
**Numbers:** F49 (12 h, 1 seed) at the old 90 / 95%-of-ceiling targets ⚠ re-read at the FL-bound targets (FX-N146, FELIX_INSIGHTS
IN6) before any number is final.

## 0. Open choices (decide before drafting prose)

Decided (10-25): **eligible clients**, **client selection**, **Felix-X** mechanism names. Title open. §4-§5 still use old names.

| item | proposal | alternatives |
|---|---|---|
| title (open; availability only, see streams row) | **Felix: Jointly Adapting Client Selection, Concurrency and Aggregation to Intermittent Device Eligibility** | "Felix: Optimizing the Federated Learning Stack for Intermittently Eligible Devices"; "Felix: Fast Federated Learning When Devices Come and Go" |
| eligible clients (decided) | a client is **eligible** while its device meets the training criteria: idle, charging, unmetered network (Bonawitz et al. 2019). Online but not eligible = can evaluate, not train. The **eligible pool** = clients eligible now. Define once in §2 ¶2; never "available" for this | — |
| decomposition | time to accuracy = updates to accuracy ÷ update throughput (statistical × system efficiency, Oort §2); bytes to accuracy = dispatched tasks × model round-trip bytes = update throughput × (1 + discard share) × round trip × time to accuracy | — |
| mechanism names (decided scheme; ablation legends Felix−X) | **Felix-Notify** (push-based availability + eviction) · **Felix-Select** (utility-proportional client selection, \|D\| × EMA loss) · **Felix-Pace** (availability-adaptive concurrency) · **Felix-Agg** (staleness-tolerant aggregation, large server step) | descriptive only: push-based availability tracking · utility-proportional selection · adaptive concurrency · staleness-tolerant aggregation |
| streams (decided 10-25) | **setting, not hero:** part of the realistic setup (usage-tied streams) and methodology, kept wherever the setting is described; not in the title, the tension or the contributions, no stream-specific motivation subsection. Evidence (F47 `fn_aug` vs `ctl_aug`, time / GB to 65%): `s0` static 1.9× / −58%, `s0u` stream 2.0× / −55%, `s50u` stream + unavailability 2.7× / −36%, `smu` 1.24× / −9%: the gain grows with unavailability, not streaming; the one stream-specific mechanism (c by loss phase) was refuted as a pacer (FX-T89). Caveats: 1 seed, at 65% not the final target, `fn` (≈ `fn_chr` on syn traces). Eval claim: "gains hold on static and streaming data" (add `s0` to the 48 h ladder / FB1 cells) | revisit only if a final-target run shows a clearly larger edge on streams |
| bytes claim | "fewest GB to every target only FedBuff also reaches"; below that Felix trades bytes for 4-12× time | drop bytes from the abstract (weakens P3) |
| not contributions (applied to every stack) | averaged (EMA) model served; per-batch augmentation (bug fix, FX-D233); FedBuff's buffer K 10 | — |

## 1. The spine (every section restates a slice of this)

- **Setting:** FL serves user-facing models whose accuracy target moves with users; devices train only while eligible (~20% at a
  time, MobiPerf); data is generated while devices are in use, i.e. when they are not eligible.
- **Metric decomposition (Oort's statistical × system efficiency, P19):** time to accuracy = updates to accuracy ÷ update throughput;
  measured after the run, unknown to the server at training time.
- **Root cause (P2):** existing systems fix participation (cohort or concurrency) at a static, population-sized N while the eligible
  pool changes; async systems learn departures only at timeout; selection, concurrency and aggregation are tuned separately though
  they are coupled.
- **Tension (P1):** keeping devices busy (system efficiency) vs choosing whom to train and keeping updates fresh (statistical
  efficiency). More concurrency fills slots but, as it nears the eligible pool, selection degenerates to uniform, staleness grows
  (τ ≈ c / K) and more tasks sit on devices about to leave; less concurrency keeps choice but idles capacity when every update still
  helps. The balance point moves on two clocks: the eligible pool (minutes to hours, unpredictable) and the training phase (quantity
  early, quality late). Fixing one layer shifts the cost to another
  (removing idle slots raised staleness 4.8 → 7.8, −1.2 pp under 1/√(1+τ)), so the fix is cross-stack.
- **Insight:** drive selection, concurrency and aggregation together from the live eligible pool, reported by the clients.

| symptom (§3) | factor hit | mechanism (§5) | ablation arm (later) |
|---|---|---|---|
| round waits for slowest pick; oracle doesn't help sync | system | async engine (FedBuff base) | sync baselines, `*` arms |
| async idles on check-ins and dead slots | system | Notify | A1 `b1`, K1 `fck` |
| held uploads = half of staleness | statistical | Notify (evict) + Agg | A1, A3 |
| keeping slots busy raises staleness | statistical | Agg | A3 `fn` |
| availability removes choice before slots | statistical, bytes | Pace | A4 `fn_chr` |
| speed-first picks lower value per update | statistical, bytes | Select | A2 `uo_o` |

## Title and abstract (≈ 170 words; B.0)

- **S1 context (25 w):** FL trains user-facing models on devices that are eligible only intermittently and generate data while in use.
- **S2 problem (25 w):** keeping devices busy and choosing whom to train conflict, and the balance shifts with the eligible pool
  and the training phase.
- **S3 why prior work fails (35 w):** they fix participation at a static size, learn departures at timeout, and tune selection,
  concurrency and aggregation separately; even oracle availability changes little.
- **S4 insight (25 w):** drive all three from the live eligible pool, reported by the clients.
- **S5 system (30 w):** Felix: four mechanisms, one clause each (names pending §0).
- **S6 results (30 w, P3 ranges on time and bytes):** Felix is first to every target on 2 datasets × 3 traces.
  - 1.26-2.6× sooner than FedBuff at −55% to +4% GB ⚠ re-read at final targets.

## §1 Introduction (≈ 1.25 pages, ~900 words + Fig 1; argument first, ≤ 3 numbers)

- **Fig 1 teaser (hardest realistic cell: cifar MobiPerf `smu`):** accuracy vs time and vs GB, all stacks; annotated "N× sooner, N% fewer GB" ⚠.
- **¶1 Stakes (P4) (150 w):** FL trains user-facing models on phones: keyboard prediction, speech, ranking (Gboard, Apple, Meta
  Papaya). Their quality is visible to users, and users change: new words, apps, habits. So high accuracy is a moving target, not a
  fixed goal; a deployment must reach it fast and reach it again as data shifts ⚠ cite (continual / periodic retraining in deployed FL).
- **¶2 The constraint (110 w):** devices train only while eligible: idle, charging, unmetered (Bonawitz et al. 2019), a small and
  shifting share of the population; and data is generated while devices are in use, when they are not eligible. Eligible time is the
  scarce resource, so every eligible device should be doing useful work. Trend: bigger models make tasks longer than eligible windows.
- **¶3 Tension (P1) (140 w), bolded sentence:** **keeping eligible devices busy and choosing whom to train pull in opposite
  directions, and the balance shifts as the eligible pool and the training phase change.** Argue the two costs and
  the two clocks (spine); no numbers.
- **¶4 Why existing work falls short, by assumption (200 w, one number):**
  - *"A round can wait for its slowest participant" (sync):* the barrier, not availability knowledge, limits them; even an oracle
    changes little (the one number: ≤ 1.5 pp).
  - *"Participation can be sized once":* sync cohorts and async concurrency are fixed for the population; when the eligible pool
    shrinks, selection has no one left to choose from, and when it grows, capacity idles.
  - *"Departures are learned at timeout":* slots stay held by devices that left, and their late updates carry most of the staleness,
    which straggler fixes cannot see.
  - *"Layers can be tuned separately":* fixing idleness raises staleness; aggressive selection lowers throughput.
- **¶5 Insight (P2) (90 w):** one root cause behind the symptoms: participation and its consequences are set statically and per
  layer, while eligibility is dynamic. Structural, so no baseline knob fixes it.
- **¶6 Ideas (130 w), each ↔ a ¶4 assumption, no mechanism detail:** clients push eligibility changes, so the server dispatches only
  to eligible clients and reclaims slots at departure; concurrency follows the eligible pool once early throughput stops paying;
  selection spends each slot on the client with the most data left to learn; aggregation takes large steps so the staleness of a busy
  system costs little.
- **¶7 Results + one honest limit (100 w):** headline ranges (the second and third numbers); limit: below ~50%, Oort and 1-step sync
  stacks spend fewer bytes at several times the time.
- **¶8 Contributions (100 w), ideas only:**
  1. *Insight:* under dynamic eligibility the throughput-vs-selectivity trade-off in FL is dynamic and cross-stack; a static
     participation size cannot track it.
  2. *Systems:* push-based availability with eviction and availability-adaptive concurrency, with no forecasting.
  3. *ML:* utility-proportional client selection and staleness-tolerant aggregation for async FL.
  4. *Result:* first to every target, faster than the strongest deployed baseline on 2 datasets × 3 traces, with the speedup split
     into statistical × system efficiency; gains hold on static and streaming data.

## §2 Background (≈ 0.5 page, ~350 words, text only)

- **¶1 Terms, briefly (70 w):** client selection; sync rounds (cohort, barrier) vs async buffered aggregation (FedBuff: concurrency c,
  buffer K, a model version per aggregation); staleness τ, ≈ c / K. Cited, no figure.
- **¶2 Device eligibility (110 w):** define *eligible* (§0) and contrast it with *online*: eligibility criteria (idle, charging, unmetered; Bonawitz et al. 2019); check-in and pace
  steering; what the server knows when: eligible clients at check-in, a departure only at timeout. Eligible share from traces cited.
- **¶3 Device data arrives over time (40 w, setting only):** data is generated during use, so a client's dataset grows between its
  eligible windows; our evaluation models this (usage-tied streams) and also runs static shards ⚠ cite online / continual FL.
- **¶4 Metrics (50 w):** time and bytes to accuracy; statistical × system efficiency (Oort).
- **¶5 Where prior systems stand (40 w + Table 1, P8):** system × {round barrier · how availability is learned · participation sized
  by · selection signal · staleness rule}. Shared blind spot: none sizes participation to the live eligible pool. No descriptions.
- *Not here:* interactions (availability → staleness, availability → lost choice) are §3 findings.

## §3 Motivation: the case for cross-stack, availability-driven FL (≈ 1.5 pages, ~800 words + 5 figs)

Each subsection: claim title → setup in one sentence (cell, stacks) → figure → the "aha" vs what the reader knew → Takeaway (P6).
All figures carry the FL upper bound (P5) ⚠ FX-N146. Baselines get one clause each, only as needed.

- **§3.1 Eligibility is scarce, bursty and unpredictable (P7) (130 w + Fig 2 trace CDFs).**
  - *Setup:* MobiPerf and syn_50 traces, 12 h. *Known:* devices are often offline. *Aha:* not just scarce but uncorrelated hour to
    hour (corr −0.02), so forecasting has little to learn: REFL with its forecaster ≈ REFL with an oracle.
  - Data: eligible 18-22%; 57-62% ever train in 12 h; 81% of a client's data lands in its busiest 10% of hours.
  - **Takeaway:** react to reported state; don't forecast.
- **§3.2 For sync FL, the barrier, not availability knowledge, is the limit (110 w + Fig 3).**
  - *Setup:* speech, sync stacks with and without oracle availability. *Aha:* perfect knowledge of who is eligible barely helps
    (REFL* ± 0.6 pp, FedDance* ≤ 1.5 pp); rounds wait for the slowest of 6 (median 256 s).
  - **Takeaway:** availability awareness pays only in an async engine.
- **§3.3 Async FL idles because it learns departures late, and fixing that alone raises staleness (160 w + Fig 4 slot occupancy).**
  - *Setup:* cifar `s50u`, FedBuff (c 60) vs FedBuff + notify / evict. *Aha:* about half the slots sit idle (~33 / 60 busy; 116k vs
    213k updates); filling them reaches 50-55% 80 min sooner, but staleness rises 4.8 → 7.8 and costs 1.2 pp final under 1/√(1+τ).
  - **Takeaway:** awareness buys system efficiency; aggregation must absorb the staleness it creates (layers are coupled).
- **§3.4 Unavailability, not slow devices, causes most staleness (130 w + Fig 5 τ by source).**
  - *Setup:* FedBuff, staleness split by cause. *Known:* staleness comes from stragglers. *Aha:* updates held across a departure are
    1.1-1.2% of updates but 49-51% of total staleness (mean τ 184-373), from median-speed clients.
  - **Takeaway:** act at the departure (reclaim the slot), not after (discount the update).
- **§3.5 Availability removes choice before it removes capacity (150 w + Fig 6 picks~size vs c / eligible).**
  - *Setup:* MobiPerf, c 60 vs ~50-67 eligible of 300. *Known:* selection beats random. *Aha:* when c nears the eligible pool, every
    selector picks what FedBuff picks (picks~size −0.45 vs −0.46); c scaled to the pool restores choice (+3.8 pp, −92% GB) but starts
    4× slower.
  - **Takeaway:** concurrency must follow the eligible pool, but only once early throughput stops paying.
- *Dropped §3.6 "streams make loss a stale selection signal" (10-25):* no stream-specific mechanism answers it (streams row, §0);
  the F46 finding predates FX-D233. At most a discussion line.

## §4 Overview (⚠ awaiting terminology; old names) (≈ 0.5 page, ~350 words + architecture fig + toy timeline)

- **¶1 Design goals, each measured later (90 w):**
  - G1 no idle slot while a trainable client waits (starvation events).
  - G2 no forecast, O(1) state per client.
  - G3 only signals the server already sees; no dataset constants.
  - G4 FedBuff-compatible buffer (secure aggregation over K).
  - Non-goals: forecasting, raw data access, tuning K, per-client local work.
- **¶2 Architecture (P10) (130 w), numbered steps:** client state monitor (OS signals → notify on flip) → server availability ledger
  (3-state belief) → LMS picks for free slots under PTC's c → trainer runs 1 epoch, returns update + mean loss → BSA buffer
  commits every K → averaged model served.
- **¶3 What each party sends + privacy (70 w):** one small message per state flip; the loss scalar and |D| that Oort already sends;
  no new data leaves the device.
- **¶4 Overview by example (P11) (60 w + fig):** 6 clients over 4 versions, FedBuff vs Felix: an idle slot waiting for a check-in,
  a slot held by a departed client, a pick from a shrunken pool.

## §5 Design (≈ 2.5 pages, ~1500 words + Algorithm 1; each subsection P13: question → observation → mechanism → why not → cost)

- **§5.1 LA: how does the server learn who can train in time to use it? (350 w)**
  - *Observation:* §3.3 idling; Felix learning by check-in loses −4.4 AUC cifar / −10.0 speech `s50u` (K1, F42b).
  - *Mechanism:* notify on every flip; 3-state belief per client; dispatch to AVL_TRAIN only; evict mid-task on departure.
  - *Why not:* check-in (idles), forecast (REFL ≈ REFL*), oracle (not deployable).
  - *Cost:* 1 message per flip; evicted partial work (measure: K7).
- **§5.2 LMS: whom to put on a free slot? (400 w)**
  - *Observation:* the speed term buys +3-4 pp at 1 h and costs 1.5-2 pp final; utility alone wins 25 / 28 pairs (F45-F46). The
    pass-averaged EMA loss is the only signal that builds a good buffer (top-15 test loss 4× better than random; pre-step batch,
    held-out loss worse than random).
  - *Mechanism:* p_i ∝ |D_i| · q_i, q = Oort's EMA loss over the training pass; proportional sampling, no cutoff, no UCB.
    Result: picks~size +0.92, 40 vs 25 samples per update.
  - *Why not:* Oort's cutoff + speed + UCB (fastest to 50%, lowest final; UCB ± 0.3 pp); uniform (−1.6 / −1.9 AUC); buffer-composition proxies ≈ random.
  - *Cost:* no extra pass (an extra measurement pass cost −14-20% throughput, FX-T64).
- **§5.3 PTC: how many tasks in flight when the pool shrinks? (400 w)**
  - *Observation:* §3.5; early every update helps, so every fixed bound that restores choice loses early.
  - *Mechanism:* φ = mean received-model loss over recent updates / running peak, recomputed every commit;
    c = φ · c_max + (1 − φ) · min(c_max, ½ trainable), within [K, c_max]. Ramps down within minutes; rises with the pool, not φ.
  - *Evidence:* MobiPerf −50% GB (659 vs 1308), same tick to 40-50%, +0.5 pp, starvation 5k vs 209k; a no-op on syn traces and on
    speech (φ stays 0.56-0.91), as designed.
  - *Why not:* fixed c ∝ trainable (4× slower start); c by loss phase alone (floor 10 from 6-8 h to 48 h); held-out pacing (noise);
    AIMD K (walks to floor). K stays fixed: τ ≈ c / K, and K 10 is FedBuff's own choice.
- **§5.4 BSA: how large a server step under staleness? (300 w)**
  - *Observation:* same selection, K 10: rate `new` / 1/√(1+τ) / 1/(1+τ) reach 50% at 60-80 / 120-130 / 210-280 min (F46).
    Staleness was a symptom of overfitting: τ 11.7 vs 7.3 still ends +4.2 pp with augmentation.
  - *Mechanism:* rate `new` (`fedbuff.weight_factor`: polynomial staleness with a 0.25 exponent × loss-utility term; mean weight
    0.61-0.78) + per-sample lr normalization of partial batches (halves the stream eval swing).
  - *Why not:* stronger discount, φ-scaled step (−0.9 / −3.2 pp), smaller fixed lr (−1.2 to −6.4 AUC), FedAdam / Yogi (extra constants).
- **§5.5 Algorithm 1 (P16) + robustness (50 w):** one box for LA → PTC → LMS → BSA with line-cited prose; missed or delayed
  notifies degrade gracefully (K6, F42e ⚠ to measure).

## Later sections (claims they must back; outline after the front matter is final)

- Theory: inherit FedBuff's rate under bounded staleness; state where held uploads break the bound (P17 slogan TBD).
- Evaluation (K5): time and GB at the hit vs the fastest deployed baseline, all cells, 3 seeds, FB1 6-day runs; U* × R table (P19).
- Ablations: ladder A0-A4 + K1, swap ablations (FX-N145), why-not table.
- Honest limits section (P24): bytes vs Oort / 1-step sync below ~50%; speech sync stacks transfer-bound under the 5% K rule;
  speech FedBuff leads Felix for the first ~2.5 h; seeds.
