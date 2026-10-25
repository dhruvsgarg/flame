# Felix paper outline (MLSys) — arguments only

Built on [PAPER_TEMPLATE.md](PAPER_TEMPLATE.md) Parts A-B (P# = its moves) and Part E (shared rules); neither is repeated here.
Evidence: [FELIX_INSIGHTS.md](FELIX_INSIGHTS.md) §5 (findings), §6 (methods, baselines table), §7 (takeaways); [FELIX_READINESS.md](FELIX_READINESS.md)
Decisions, contribution ledger, "Paper: motivation and evaluation from FL1", rejected-ideas table.
**Scope now:** title → §5 design. Theory, implementation, evaluation, ablations come after the front matter is final.
**Framing:** one system, one paper; the current implementation is Felix, no old-vs-new comparison.
**Numbers:** F49 (12 h, 1 seed) at the old 90 / 95%-of-ceiling targets ⚠ re-read at the FL-bound targets (FX-N146, FELIX_INSIGHTS
IN6) before any number is final.

## 0. Open choices (decide before drafting prose)

| item | proposal | alternatives |
|---|---|---|
| title | **Felix: React, Don't Predict — Asynchronous Federated Learning on Intermittently Available Devices** | "Felix: Fast FL When Devices Come and Go"; "Federated Learning at the Speed of Availability" |
| mechanism 1 (notify + evict, C1) | **Live Availability (LA)** | Presence Tracking; React-Evict Dispatch |
| mechanism 2 (\|D\| × EMA loss, `prop`, C3) | **Learning-Mass Selection (LMS)** | Value Sampling; Size-Unfitness Selection |
| mechanism 3 (c ramp, `fn_chr`) | **Pool-Tracking Concurrency (PTC)** | Fit-Paced Concurrency; c-Ramp |
| mechanism 4 (rate `new` + lr normalize) | **Bold-Step Aggregation (BSA)** | Large-Step Buffer; Step-over-Discount |
| streams | setup + motivation only (none of the compared systems model data arriving over time); a headline claim only if a mechanism responds to them (PTC's loss phase is the candidate) | headline "first FL eval on usage-generated data" — check continual / online FL work first |
| bytes claim | "fewest GB to every target only FedBuff also reaches"; below that Felix trades bytes for 4-12× time | drop bytes from the abstract (weakens P3) |
| not contributions (applied to every stack) | averaged (EMA) model served; per-batch augmentation (bug fix, FX-D233); FedBuff's buffer K 10 | — |

## 1. The spine (every section restates a slice of this)

- **Setting:** phones train only when idle, charging, unmetered; trainable share ~20% (MobiPerf); data arrives while devices are in use.
- **Accounting (organizing lens, P19):** time to accuracy = U* ÷ R (updates needed ÷ update rate); bytes = U* × model round trip.
- **Root cause (P2):** existing systems learn availability *late* (check-in, timeout, forecast) and size participation by the
  population n, not by the live trainable pool. This hurts both factors.
- **Tension (P1):** fill every slot (R) vs keep choice and freshness (U*); the trainable pool bounds both.
- **Insight:** react to state now, spend each slot on the most learning per update, size concurrency to the pool that can train.

| symptom (§3) | factor hit | mechanism (§5) | ablation arm (later) |
|---|---|---|---|
| round waits for slowest pick; oracle doesn't help sync | R | async engine (FedBuff base) | sync baselines, `*` arms |
| async idles on check-ins and dead slots | R | LA | A1 `b1`, K1 `fck` |
| held uploads = half of staleness | U* | LA (evict) + BSA | A1, A3 |
| availability removes choice before slots | U*, bytes | PTC | A4 `fn_chr` |
| speed-first picks lower value per update | U*, bytes | LMS | A2 `uo_o` |
| staleness discount slows progress | U* | BSA | A3 `fn` |

## Title and abstract (≈ 170 words; B.0)

- **S1 context (25 w):** FL trains on devices that can participate only intermittently.
  - MobiPerf: 18-22% of devices trainable at a time; 57-62% ever train in 12 h.
- **S2 problem (25 w):** unavailability slows FL twice: idle or wasted slots, and fewer useful updates.
- **S3 why prior work fails (35 w):** sync systems (Oort, REFL, FedDance) wait at a round barrier; async FedBuff learns availability at
  check-in; forecasting (REFL) and even oracle knowledge change little.
  - REFL* = REFL ± 0.6 pp; FedDance* ≤ 1.5 pp.
- **S4 insight (25 w):** time to accuracy = updates needed ÷ update rate; late knowledge and n-sized participation hurt both.
- **S5 system (30 w):** Felix: LA, LMS, PTC, BSA, one clause each.
- **S6 results (30 w, P3 ranges on time and bytes):** Felix is first to every target on 2 datasets × 3 traces.
  - 1.26-2.6× sooner than FedBuff at −55% to +4% GB; sync stacks ≥ 5× later to 50% or never ⚠ re-read at final targets.

## §1 Introduction (≈ 1.25 pages, ~950 words + Fig 1)

- **Fig 1 teaser (hardest realistic cell: cifar MobiPerf `smu`):** accuracy vs time and vs GB, all stacks; annotated "1.5× sooner, 45% fewer GB" ⚠.
- **¶1 Stakes (P4) (130 w):** FL is deployed on phones, and phones train only when idle, charging and unmetered.
  - Gboard, Apple, Meta (Papaya) deployments; Bonawitz et al. 2019 eligibility rules.
  - Trend: bigger models make tasks longer than sessions (speech 29 MB: 93% of a 1-step task is transfer); data is generated during use.
- **¶2 Tension (P1) (90 w), bolded sentence:** **an FL system must keep its slots busy and keep choosing whom to train, but the
  trainable pool limits both.**
  - MobiPerf: ~50-67 of 300 clients trainable vs FedBuff's c 60 (20% of n).
- **¶3 Why existing work falls short, by assumption (220 w):**
  - *"A round can wait for its slowest pick" (sync):* speech rounds wait for the slowest of 6 (median 256 s); sync stacks ≥ 5× later
    to 50% or never; an oracle changes ≤ 1.5 pp, so the barrier binds, not knowledge.
  - *"Availability is learned at check-in or timeout" (async):* FedBuff keeps ~33 of 60 slots busy (98k check-in waits, 677 s
    timeouts): 116k vs 213k updates. Held uploads are 1.1% of updates but ~50% of staleness, which straggler fixes cannot see.
  - *"Availability can be forecast, and participation scales with n":* REFL ≈ REFL*; data arrivals hour-to-hour corr −0.02; at
    c ≥ trainable pool every selector degenerates to uniform (Felix's edge 2.0-2.7× → 1.24× on MobiPerf, F47).
- **¶4 Insight (P2) (110 w):** one root cause, late knowledge and n-sized participation, hurts both U* and R.
  - Qualitative: structural, so no baseline knob fixes it (oracle arms, K, c sweeps all measured).
- **¶5 System (150 w), mechanism ↔ limitation:**
  - LA: clients push every state flip; server dispatches only to trainable clients, evicts on departure; no forecast, O(1) state.
  - LMS: sample ∝ |D_i| × EMA loss from the training pass; no speed term, no extra pass.
  - PTC: c = φ · c_max + (1 − φ) · min(c_max, ½ trainable): throughput first, choice once clients fit their data.
  - BSA: large, near-constant server step on FedBuff's buffer instead of staleness discounting.
- **¶6 Results + one honest limit (130 w):**
  - Felix is first to every target on all 6 cells vs every baseline (F49) ⚠.
  - The decomposition differs by cell: `s0u` all U* (2.24× fewer updates); `s50u` half each (1.65× × 1.70×); speech `s50u` all R (2.6×).
  - Limit: below ~50%, Oort and the 1-step sync stacks spend fewer bytes (Oort 7-14× fewer GB at 4-12× the time).
- **¶7 Contributions (120 w), 4 bullets:**
  1. *Characterization:* unavailability hurts FL through three channels (idle slots, hidden staleness, lost choice), measured on
     real traces with source-faithful baselines that learn availability by their deployed mechanisms.
  2. *Design:* Felix, four mechanisms, no forecasting, no dataset constants, only signals the server already sees.
  3. *Methodology:* FedScale task-time model, trace-tied usage streams, at-target comparison, simulator graded against real runs.
  4. *Evaluation:* time and bytes to target on 2 datasets × 3 traces, a U* × R decomposition, and a "why not X" table (25 rejected ideas).

## §2 Background (≈ 0.5 page, ~380 words + 1 small fig + notation table)

- **¶1 FL loop, sync vs async (110 w), fig in FedBuff Fig 2 style:** a version per aggregation; buffer K vs concurrency c; staleness τ ≈ c / K.
- **¶2 Participation states and check-in (110 w):** AVL_TRAIN / AVL_EVAL / UN_AVL; Bonawitz check-in + pace steering; what the server
  knows when (pool at check-in, departure at timeout).
- **¶3 Accounting (100 w):** TTA = U* / R; R = c_eff (1 − waste) / s̄_T; B = U* · b; per-task time d_i = 3 s_i ms_i + 2 m / bw_i (FedScale).
- **¶4 Baselines, one line each (60 w):** Oort (utility × speed, UCB), REFL (least-available first + stale aggregation), FedDance
  (V · I · A), FedBuff (uniform async buffer).

## §3 Motivation: the case for reacting to availability (≈ 1.5 pages, ~850 words + 4 figs + 1 table)

- **¶0 Setup (60 w):** same cells as the evaluation (cifar n 300 / speech n 100, α 0.1, syn_50 / MobiPerf, usage streams); FL
  upper bound on every curve (P5) ⚠ FX-N146.
- **Table 1 (P8), assumptions of prior systems:** system × {round barrier · availability learned by · forecast · data static ·
  participation sized by}. Shared blind spot: none sizes participation to the live trainable pool, and none models streaming data.
- **§3.1 Availability is scarce, bursty and unpredictable (P7) (150 w + Fig 2 trace CDFs).**
  - MobiPerf trainable 18-22%; 57-62% of devices ever train in 12 h; 81% of a client's data lands in its busiest 10% of hours; hour-to-hour corr −0.02.
  - **Takeaway:** react to state; forecasts have little to learn from (REFL ≈ REFL*).
- **§3.2 The round barrier, not knowledge, limits sync FL (150 w + Fig 3 acc vs time, sync vs oracle arms).**
  - Speech: 160-186 rounds in 12 h; sync stacks at chance (3.7%) for 4-6 h; REFL* ± 0.6 pp, FedDance* ≤ 1.5 pp.
  - **Takeaway:** awareness pays only inside an async engine.
- **§3.3 Unaware async idles (150 w + Fig 4 slot occupancy over time).**
  - FedBuff `s50u`: ~33 / 60 slots busy, 98k starvation events, 116k vs 213k updates with notify / evict; 50-55% 80 min sooner.
  - **Takeaway:** knowing state now buys update rate R.
- **§3.4 Unavailability makes staleness straggler fixes cannot see (150 w + Fig 5 τ by source).**
  - Held uploads: 1.1-1.2% of updates, 49-51% of Σ τ, mean τ 184-373 versions, from median-speed clients (19-21 s tasks).
  - **Takeaway:** act before the departure (whom to pick, evict), not after (discount the update).
- **§3.5 Availability removes choice before it removes slots (170 w + Fig 6 picks~size vs c / trainable).**
  - MobiPerf at c 60: picks equal FedBuff's (picks~size −0.45 vs −0.46, τ 15); c × trainable share restores choice (+0.48, +3.8 pp,
    −92% GB) but starts 4× slower; fixed c bounds end 36.5-48.6 vs 51.8% at 3 h (F48a).
  - **Takeaway:** c must follow the trainable pool, but only once early throughput stops paying.

## §4 Overview (≈ 0.5 page, ~350 words + architecture fig + toy timeline)

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
