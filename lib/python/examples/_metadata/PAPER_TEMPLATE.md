# Paper template — how to write an FL systems paper for MLSys (Felix and FluxTune)

**Purpose.** A writing guide shared by two independent MLSys submissions, Felix and FluxTune. Part A distills what six accepted
papers do well: Oort (OSDI'21), REFL (EuroSys'23), FedBuff (AISTATS'22), FwdLLM (ATC'24), FLINT (MLSys'23) and FedDance
(SoCC'25). Part B turns that into a section-by-section template: purpose, dos, don'ts, arguments, data, citations, figures,
math, experiments. Each paper's authors use Parts A and B to write their own paper-specific markdown; Parts C and D are only
starting points (see their disclaimers). Part E holds rules both papers share, and Part F the open items.

**Owners.** Felix: Dhruv. FluxTune: Aish (with Dhruv); its latest solution and experiment docs live on Aish's branch.

---

## Part A — What the six accepted papers do well

### A.1 Each paper in one line: the pitch, and what made it convincing

| paper | pitch | why a reader believes it by page 2 |
|---|---|---|
| **Oort** (OSDI'21) | Random participant selection wastes FL; selecting by statistical × system utility gives 1.2-14.1× time to accuracy. | The tension (statistical vs system efficiency) is drawn as a 2-D plot with random, opt-stat, opt-sys and Oort on it (Fig 7). Real-data CDFs show heterogeneity is an order of magnitude (Figs 1-2). Prior work is plotted against a centralized upper bound (Fig 3). |
| **REFL** (EuroSys'23) | FL wastes most device work; prioritizing least-available learners and keeping stale updates halves resource use at equal accuracy. | It changes the metric (resources consumed to reach accuracy), which makes SAFA look 5× wasteful and Oort look biased (Figs 2-4). Each motivation result ends in a bold "Opportunity." The trace statistic "70% of learners are available ≤ 10 min" anchors everything. |
| **FedBuff** (AISTATS'22) | Sync FL cannot use more than a few hundred concurrent clients and fully async breaks secure aggregation; a buffer of K updates gets both. | One scaling-limit figure: rounds to target barely fall past concurrency 100, "analogous to large-batch training" (Fig 1). Privacy compatibility is a design goal, not an afterthought. |
| **FwdLLM** (ATC'24) | Backprop is why federated LLM fine-tuning is stuck (memory wall, no NPU, no device scaling); perturbed inference removes all three. | Pilot experiments give three obstacles, each with a number (7.7 GB for RoBERTa-large; NPUs 30× faster but training-incapable; saturation at tens of clients). Table 3 shows every prior FL paper uses ≤ 100 devices per round. |
| **FLINT** (MLSys'23) | Bringing cross-device FL to production is risky; a platform that measures real constraints before deployment de-risks it. | Industry measurements: 8,000 device types, availability swings 14× over a day, eligibility filters leave 22% of devices. Honest per-domain accuracy loss (−0.18% to −1.85%). |
| **FedDance** (SoCC'25) | Participant selection fails in dynamic environments; combining a server-side availability predictor, a loss-based importance and an accuracy-increment term fixes it. | A three-histogram trace characterization (periods per device, period length, gap between periods; online 20.26% of the time, Fig 3). Controlled experiments correlate data heterogeneity with availability, so each baseline's specific blind spot shows up (REFL −2.0%, Oort −3.1% vs random, Fig 5). |

### A.2 What each paper does well, section by section

**Introduction**
- Oort ¶1: deployments (Google, Apple, NVIDIA, smart cities, video streaming), then the FL life cycle the reader already knows,
  then "unfortunately" + three facts (availability, heterogeneity, too many participants). Then a contributions list of exactly three.
- REFL ¶3: resource wastage "deters users from participating in FL and makes scaling problematic". It states the cost to the
  ecosystem, not only to a metric.
- FwdLLM ¶2-4: "a salient feature of LLMs is scalability", so the problem grows with the trend. Three obstacles get bold labels.
  Then "This work leverages a crucial observation: all above issues can be traced to backpropagation". One root cause for three
  symptoms is the strongest intro shape.
- FedDance ¶2-3: two challenges, each with a measured number (20.26% online; availability-first is 12.5% worse than random), then
  the two landmark baselines named and their failure measured (REFL −2.0%, Oort −3.1%).
- FedBuff: two numbered challenges (scalability, privacy) under bold labels, each with one paragraph. The proposal fixes both at once.

**Background and motivation**
- Oort §2: a primer (training vs testing), then "challenges" (data, systems, scale, privacy) backed by real-dataset CDFs, then
  "limitations of existing solutions" with a figure against an upper bound. Background → challenge → limitation is the cleanest order.
- REFL §3: "The Case for X". Each subsection is one experiment, one figure annotated with run time, and one "Opportunity." line.
- FedDance §2.2-2.3: challenges C1-C2, then limitations of REFL and Oort. Each limitation gets a controlled experiment built to expose
  it (data heterogeneity set inversely to availability). The setting is stylized, but it isolates a mechanism the way a unit test does.
- FwdLLM §2.2: "preliminary experiments". Figures of memory per model, NPU speedup across chip generations (a trend line, so the
  problem worsens over time), and accuracy vs number of clients.
- FLINT §2-3: measurement tables a practitioner recognizes (criteria → % devices still eligible).

**Overview and architecture**
- Oort §3: an architecture figure with circled steps 1-4 plus sub-steps 2a / 2b, then a 20-line API snippet for training and testing.
  The reader knows where Oort sits and how a developer calls it before any algorithm appears.
- FedDance §3: key design ideas I1-I3, each mapped to a challenge. An architecture figure splits client (Local Controller: Status
  Tracer, Runtime Monitor) from server (Global Coordinator: Utility Optimizer with three named estimators, Participant Selector).
- REFL §4: the two components get acronyms (IPS, SAA) used everywhere after. An "overview by example" figure shows 9 colored learners
  over 4 rounds under Oort vs REFL (Fig 5). It is the single most explanatory figure in the six papers.
- FwdLLM §3.1: a numbered workflow figure, a memory figure (what is released when), then "advantages by design", "implications on
  privacy" and "unique challenges", each one paragraph.

**Design**
- Oort §4: opens with the trade-off, then three challenges as questions, each answered by one subsection. It derives the ideal (importance
  sampling ∝ gradient norm), explains why it is impractical, gives the pragmatic proxy (loss), then privacy and robustness. Algorithm 1's
  prose cites line numbers.
- FwdLLM §3.2-3.4: every subsection is "observation → figure → mechanism". Mechanisms are named (VPP, DPS) and reappear in the ablation.
- FedDance §4: one subsection per utility factor, each with its formula (Poisson availability V = 1 − e^(−λK), loss importance I,
  accuracy increment A), then the combined utility U = V · I · A and Algorithm 1.
- REFL §4.2.3: lists the prior scaling rules (Equal, DynSGD, AdaSGD) before proposing its own. The reader sees the design space first.

**Theory**
- REFL §4.2: assumptions, Theorem 1, lemma chain, then "Significance of results": the rate improves with local steps, and asynchrony
  only affects the faster-decaying term, so it is "asynchrony for free". Every theorem should end in a one-line slogan like this.
- FedBuff §4: Theorem 1 → Corollary 1 → bullets on what it implies (worst-case complexity, communication cost vs K, local work vs
  drift, staleness dissipates as O(1/T)). The bound is read as a set of design trade-offs.
- FedDance §4.5: the bound is split into a non-vanishing bias (from selection skew) and a vanishing error. Its interpretation says
  which selection choices shrink each term, so the theory justifies the design.
- Oort §5: uses a known concentration bound (Hoeffding) to size participants and puts the proof in a tech report. Not every paper
  needs a new proof; a correct, cited bound that sizes a knob is enough.

**Implementation**
- Oort §6: LOC, framework integration (PySyft), fault tolerance (checkpointed metadata), solver. Half a column.
- REFL §5.1 and FLINT §3.4: how the emulator computes time (samples × latency per sample + bytes / bandwidth on a virtual clock). The
  reader can judge the simulation's realism.

**Evaluation**
- Oort §7 and REFL §5 open with the questions the evaluation answers and a bullet summary of the answers.
- Oort Table 2 splits every speedup into statistical × system factors per dataset. It makes the result explainable, not just large.
- Oort §7.2.2 and FedDance §6.3.2: named leave-one-out variants ("Oort w/o Pacer", "FedDance−V / −A / −I / −U") on the same axes, for
  both time to target and final accuracy.
- Oort §7.2.3: sensitivity to every knob (K, α, outliers, noise, fairness), and it shows the system self-corrects across α.
- FedDance §6.3.1: the predictor is benchmarked as a component (accuracy over time, precision / recall / F1, an overhead CDF). A
  mechanism with its own accuracy deserves its own microbenchmark.
- FedDance §6.4 and FwdLLM §5.3: system cost as CDFs and bars (device-side compute per selection, memory, energy, network).
- REFL §6: "future-proof". It triples the population and simulates faster hardware, then asks whether the gain grows.
- FwdLLM §5.2: a "for the first time" demonstration (LLaMA-7B on a phone) and one qualitative example.
- FedBuff §6: explains its metric (client trips) because rounds don't compare across sync and async. FLINT reports virtual-clock
  time and trains several trials per configuration to give error bounds.

**Discussion, related work, artifact**
- REFL §6: deployment considerations and adversarial clients that misreport availability. REFL §7: integration with real frameworks.
- FLINT §5: a call to the community ("report model size and on-device resource usage with benchmarks").
- Related work in all six: grouped by approach, one sentence per group on what breaks in the new setting.
- REFL appendix: claims C1 / C2 → experiments E1 / E2 → scripts → expected result and compute hours.

### A.3 The moves, numbered (Part B cites them as P#)

- **P1 Name the tension:** one trade-off with two named ends, drawn as a plot (Oort Fig 7).
- **P2 One root cause for several symptoms** (FwdLLM: backprop → memory, NPU, scaling).
- **P3 Ranges in the abstract** over every setting ("1.2×-14.1×").
- **P4 Stakes first:** real deployments in ¶1, and who loses if the problem stays.
- **P5 Prior work against a bound:** a centralized or oracle ceiling on the same figure (Oort Fig 3).
- **P6 Each motivation result ends with "Opportunity." or "Takeaway."** (REFL).
- **P7 Trace characterization as evidence:** a few histograms or CDFs a reader can repeat as a sentence (FedDance Fig 3; REFL "70% ≤ 10 min").
- **P8 A literature-settings table** that exposes a shared blind spot (FwdLLM Table 3).
- **P9 A controlled experiment per baseline blind spot** (FedDance Fig 5), labelled as stylized and kept apart from the realistic eval.
- **P10 Architecture figure with numbered steps, client vs server split, plus a short API or pseudocode listing** (Oort Figs 5-6).
- **P11 Overview by example:** a toy timeline of a few colored clients, baseline vs ours (REFL Fig 5).
- **P12 Challenges as questions → design ideas I1..In → one subsection each** (Oort §4.1; FedDance §3.1).
- **P13 Each design subsection: key observation → measured figure → mechanism → alternative and why not → cost / privacy.**
- **P14 Named mechanisms and acronyms, reused verbatim in ablation legends** (IPS / SAA, VPP / DPS, −V / −A / −I).
- **P15 Show the design space before the choice** (REFL's list of scaling rules).
- **P16 One algorithm box whose prose cites line numbers.**
- **P17 Theorem → slogan:** each bound ends in one line of design guidance ("asynchrony for free"; bias vs vanishing error).
- **P18 Evaluation opens with questions and answer bullets.**
- **P19 Decompose the headline speedup** into independent factors (Oort Table 2).
- **P20 Leave-one-out variants on one axis**, for both time to target and final accuracy.
- **P21 Component microbenchmarks** for any mechanism that has its own accuracy (FedDance's predictor).
- **P22 Sensitivity to every knob, robustness to adversity**, and self-correction across the knob.
- **P23 System-cost CDFs** (device-side and server-side), against baselines.
- **P24 Say where the gain shrinks and why** (Oort Speech 1.2×; FwdLLM non-IID −3.8%).
- **P25 Future-proofing:** a larger population and faster hardware (REFL §6).
- **P26 Justify the metric** whenever the field's default hides the effect (FedBuff client trips; REFL resource usage).
- **P27 Artifact appendix:** claims → experiments → scripts → expected output.
- **P28 A call to the community** that outlives the system (FLINT).

### A.4 Where they are weak (reviewers now expect better)

| weakness | in | what we do instead |
|---|---|---|
| Simulator never checked against a real run | Oort, REFL, FedDance, FLINT | grade the simulator against real runs (PARITY) and report it as a fidelity result |
| Rounds, not time: round-to-accuracy hides round length | FedDance (Table 2), FedBuff main text | time and bytes to target on a virtual clock driven by device traces |
| Fixed horizon ("after 1000 rounds") ranks curves mid-flight | FedDance Table 3, REFL | compare at the target; report accuracy at the end only for stacks that never hit |
| Sensitivity deferred to future work | REFL §5.1 | sweep every exposed knob |
| Component claim contradicted by its own table (FedDance's predictor "outperforms REFL" while Table 4 shows lower accuracy and F1) | FedDance §6.3.1 | every sentence matches its table; state losses as losses |
| Assumed forecaster accuracy (90%) instead of measuring it | REFL | run each baseline's deployed mechanism on the real trace (R27) |
| Stylized correlations presented as general | FedDance §2 | keep controlled experiments in motivation, labelled; the eval uses only realistic traces |
| Convex / strongly-convex theory for deep nets | FedDance Thm 1 | non-convex stationarity bounds (FedBuff, REFL style), or an exact accounting validated on runs |
| Constant arrival rates or synthetic delays | FedBuff | real availability traces and device compute / bandwidth traces |

---

## Part B — The template, section by section

**MLSys format.** 10 pages of main text plus unlimited references and appendix (check the year's call). Reviewers weigh ML rigor
(baselines, seeds, statistical care) and systems realism (cost model, overheads, deployability) equally. Artifact evaluation earns
badges; plan for it from day one. Rough page split: abstract + intro 1.25 · background + motivation 1.5 · overview 0.5 · design 2.5
· theory 0.75 · implementation 0.25 · evaluation 2.75 · related work + discussion + conclusion 0.5.

**Every section entry below has the same fields.** *Purpose* · *Dos* · *Don'ts* · *Arguments: qualitative / quantitative* ·
*Data* · *Citations* · *Figures / tables* · *Concepts / math* · *Experiments*.

### B.0 Title and abstract

- **Purpose:** the whole paper in about 150 words; reviewers decide here whether to read sympathetically.
- **Dos:** title = name + what it does + the condition it targets. Abstract in six sentences: context → problem and why it matters
  (P4) → why existing work fails, named → key insight → system and its 2-3 mechanisms → results as ranges over every setting, on both
  time and bytes (P3).
- **Don'ts:** no undefined acronyms, no "novel" or "efficient" without a number, no best case without its range, no claim that some
  evaluated cell contradicts.

### B.1 Introduction (≈1.25 pages)

- **Purpose:** convince the reader by the end of page 1 that the problem is real, widespread and growing, that existing systems
  cannot fix it by tuning, and that one insight fixes it.
- **Paragraph recipe:**
  1. *Stakes (P4):* 3-4 deployments plus one survey; the trend that makes it worse over time (more devices, bigger models, stricter
     privacy); who loses (developers, users, operators).
  2. *Problem as a tension (P1)*, one bolded sentence, general across models and datasets.
  3. *Why existing work falls short:* 2-3 limitations, each with one measured number from §3, grouped by assumption ("they assume
     clients stay online for a round") rather than by paper.
  4. *Insight:* one sentence a reader can repeat. If possible, one root cause behind the symptoms (P2).
  5. *System:* 2-3 mechanisms, one line each, each mapped to a limitation in ¶3.
  6. *Results:* ranges vs the strongest deployed baseline on both axes, and one honest limit.
  7. *Contributions:* 3-4 bullets: characterization / insight · design · analysis · implementation and evaluation (plus artifact).
- **Arguments:** qualitative = why the root cause is structural, so it survives every baseline knob; quantitative = one number per
  limitation, the headline ranges.
- **Figures:** Figure 1 is a teaser that shows the problem and the win at once, on the hardest realistic setting, annotated "N×
  sooner, N× fewer GB".
- **Don'ts:** no related-work survey; no design detail; no generality the evaluation lacks; never cite a number here that a later
  section qualifies without the qualifier.

### B.2 Background (≈0.5 page)

- **Purpose:** only what motivation and design need.
- **Dos:** the FL loop (check-in → select → dispatch → train → upload → aggregate) as a small figure contrasting sync and async
  timing (FedBuff Fig 2); a notation table with every symbol used later; one line per baseline mechanism; the device conditions for
  participation (idle, charging, unmetered network), cited.
- **Concepts:** version, staleness, concurrency vs buffer size, availability states, time to accuracy, bytes to accuracy.
- **Don'ts:** no textbook FL, no claims, no term that is never used again.

### B.3 Motivation and characterization ("The case for X", ≈1.5 pages)

- **Purpose:** prove the premise with measurements before the design appears.
- **Structure:** 3-4 subsections. Each = one observation (the subsection title is the claim) → one figure → one "Opportunity." /
  "Takeaway." sentence that the design later cashes (P6).
- **Quantitative arguments:** trace characterization (P7) such as session-length CDFs, share of time online, gaps between sessions,
  eligible share after device criteria, daily swing; pilot runs of the strongest baselines against an upper bound (P5); a decomposition
  of where the loss goes (idle slots, wasted work, staleness by source, choice ratio).
- **Qualitative arguments:** why the cause is structural; which assumption each baseline family makes that the trace breaks.
- **Data:** the same traces, models and datasets as the evaluation, so motivation numbers reappear later. A controlled experiment
  (P9) is allowed if it is labelled as such.
- **Figures / tables:** trace histograms or CDFs, a baseline-vs-bound curve, a literature-settings table if the field shares a blind
  spot (P8).
- **Citations:** trace sources (MobiPerf, the 136k-user trace of Yang et al. WWW'21, Stunner), deployment papers for the conditions.
- **Don'ts:** no strawman baselines; no setting you don't evaluate; no result from a setup later declared void.

### B.4 Overview (≈0.5 page)

- **Purpose:** the reader's map before details.
- **Dos:** design goals (3-4, each testable and later measured); non-goals (what is deliberately not done: no forecasting, no raw-data
  access, no per-dataset constants); architecture figure split into client and server parts with numbered steps (P10); what each party
  sends, and its privacy footprint; a short API or pseudocode listing; an overview-by-example timeline (P11).
- **Don'ts:** don't repeat the design; don't hide what the server must know.

### B.5 Design (≈2.5 pages; one subsection per mechanism)

- **Purpose:** each mechanism as an answer to one challenge (P12).
- **Per subsection (P13):**
  1. the challenge as a question;
  2. the key observation in one sentence, with its measured figure;
  3. the mechanism: rule, inputs (only signals the server already sees or the client can send privately), output; its formula;
  4. the design space and why not the obvious alternative (P15), citing the alternative's measured failure;
  5. cost (bytes, compute, memory, server state) and privacy footprint.
- **Dos:** named mechanisms reused in ablation legends (P14); one algorithm box for the whole system with line-cited prose (P16); state
  for every constant whether it is derived, sensed at runtime, or a deployment property; robustness to misreporting clients.
- **Don'ts:** no dataset-specific constants; no mechanism without an ablation; no "we found empirically" without a figure.

### B.6 Theory and analysis (≈0.75 page; proofs in the appendix)

- **Purpose:** turn the mechanism into a guarantee, a sizing rule or an exact accounting the reader can reuse.
- **Dos:** assumptions in words first, then the statement; a "what it means" paragraph ending in a slogan (P17); split the bound into
  interpretable terms (bias vs vanishing error; staleness term vs variance term); validate it on runs (predicted vs measured scatter).
  Where the system is a known algorithm plus bounded changes, inherit the known rate and say so.
- **Don'ts:** no assumption the system violates without saying so (e.g. bounded staleness under multi-hour held uploads); no convex
  theory for deep nets as the main result; no bound so loose it guides nothing.
- **Citations:** FedBuff Thm 1, REFL Thm 1, Stich & Karimireddy (error feedback), Mania et al. (perturbed iterates), Cho et al.
  (biased client selection), Li et al. (FedAvg on non-IID), Little's law for queueing arguments.

### B.7 Implementation (≈0.25 page)

- **Dos:** framework; LOC; components; message types added; how real and simulated runs share code; how baselines were ported
  (source-faithful, from their own code); how simulated time is computed (P26); anonymized artifact link.
- **Don'ts:** no engineering diary.

### B.8 Evaluation methodology (≈0.5 page)

- **Dos (one item each, in this order):** hardware; datasets, partition and α; models and parameter counts; device compute and bandwidth
  traces (cited); availability traces and their statistics; data-arrival model if data streams; baselines with their source configs and a
  deviation table; how each baseline learns availability; metrics and why (P26); targets and how they are derived; seeds; how the
  simulator is validated against real runs.
- **Table:** workload table (dataset · model · params · clients · labels · local work · batch · lr), as FedDance Table 1.
- **Don'ts:** no knob tuned for us that the baseline did not choose; no metric chosen after seeing results; no target picked to flatter
  one stack.

### B.9 End-to-end results

- **Dos:** open with questions and answer bullets (P18); one grid figure (rows = datasets, columns = scenarios; accuracy vs time and
  accuracy vs bytes); a table of time and bytes to target normalized to the best stack; the decomposition table (P19); seeds as bands or
  min-max whiskers.
- **Don'ts:** no horizon snapshot as the headline; no averaging that hides a losing cell.

### B.10 Where the gain comes from

- **Dos:** factor the headline into independent measured terms (statistical × system; updates needed × update rate) per cell; link each
  factor to a mechanism.

### B.11 Ablations

- **Dos:** a cumulative ladder from the strongest baseline to the full system; leave-one-out variants (P20); swap ablations (each layer
  replaced by the baselines' way); every named mechanism in at least one, on time and on final accuracy.

### B.12 Component microbenchmarks

- **Dos:** any mechanism with its own accuracy or latency gets its own measurement (P21): predictor accuracy, signal-vs-ground-truth
  correlation, controller trajectory over time, decision latency.

### B.13 Sensitivity and robustness

- **Dos:** sweep every exposed knob (P22); population scale; heterogeneity (α); adversity: message loss, delayed or missing signals,
  misreporting clients, noisy utilities; show self-correction.

### B.14 Overheads and scalability

- **Dos:** server time per decision; messages per hour vs baselines; device-side compute, memory and energy proxy as CDFs (P23); server
  state per client; scaling with population.

### B.15 Discussion and limitations

- **Dos:** where the gain shrinks and why (P24); future-proofing (P25); privacy (secure aggregation, DP compatibility); deployment notes
  (OS signals needed, permissions); what is simulated vs measured on real hardware.

### B.16 Related work (≈0.4 page)

- **Dos:** group by approach; per group, the assumption that breaks in our setting; position the closest 2-3 works explicitly; list
  complementary work (compression, secure aggregation, DP) as composable.
- **Don'ts:** no unpositioned lists; never misstate a baseline.

### B.17 Conclusion and artifact appendix

- **Conclusion:** one paragraph; restate the insight, not the numbers; optionally a call to the community (P28).
- **Artifact (P27):** claims C1..Cn → experiment → script → expected output → compute hours; the scaled-down variant that runs in a day.

### B.18 Cross-cutting style rules

- **Figures:** one message per figure, stated in the caption's first sentence; one color and marker per stack across the paper; a
  caption that says how to read it ("lower is better"); regenerable by one script from run directories.
- **Numbers:** every number traces to a run ID (a draft comment until camera-ready); units always; no more precision than the seed noise.
- **Names:** one vocabulary across text, figures and legends; define a term once and reuse it exactly.
- **Claims:** each claim names the experiment that tests it; losses are stated as losses; "against expectation" findings and rejected
  ideas go into a "why not X" appendix.

### B.19 Reviewer checklist (run on every draft)

- [ ] Problem is real, widespread and growing by the end of page 1; ¶1 cites deployments.
- [ ] Each limitation of prior work has a measured number.
- [ ] Insight fits in one sentence; mechanisms map one-to-one to limitations.
- [ ] Baselines are source-faithful, with a deviation table; the strongest deployed baseline is the reference.
- [ ] Time and bytes to target, at the target, with seeds; decomposition of the speedup.
- [ ] Every mechanism has an ablation; every knob has a sensitivity curve.
- [ ] Overheads measured; privacy compatibility stated.
- [ ] Simulator fidelity shown against real runs.
- [ ] Limitations section says where the gain shrinks.
- [ ] Artifact appendix maps claims to scripts.

---

## Part C — Felix (starting point only)

> **Disclaimer.** Dhruv will overhaul or replace this part in Felix's own paper markdown. It only points to the docs that hold
> Felix's evidence and records the third-workload recommendation. It deliberately does not decide Felix's narrative, evaluation or
> wording; the authors derive those from Parts A and B.

- **Evidence lives in:** [FELIX_INSIGHTS.md](FELIX_INSIGHTS.md) §5 (findings), §6 (methods, protocol, baselines table), §7 (reader
  takeaways); [FELIX_READINESS.md](FELIX_READINESS.md) Scoreboard, Decisions ("Targets and comparison", "Augmentation"), "Paper:
  motivation and evaluation from FL1", and the rejected-ideas table.
- **Framing note:** Felix is one system and one paper. The current implementation supersedes the earlier one, and the paper presents
  it as Felix, with the added baselines; no old-vs-new comparison.
- **For Felix's markdown (to defend there, not here):** the bytes trade-off vs Oort and the 1-step sync stacks below ~50%; the speech sync
  baselines near chance for hours; claims that rest on one seed or the old targets.

### C.1 Third workload for Felix: recommendations

**Constraints any third workload must meet.** It trains with backprop on a device, so it has to fit device memory: FluxTune's
premise is that backprop on sub-billion LLMs does not, and Felix must not contradict it. It needs a natural per-user split, a
deployment story, and a modeled task time (FedScale's per-sample traces are for small CNNs, so scale ms per sample by FLOPs relative
to a profiled model). Baselines should run with their own configs (ROBUST L34), and legs must fit the bench (n 100-300, ≤ 12 h vclock in
minutes to an hour of wall time). Bonus: data that is generated while the device is in use, which matches our `usage` stream (FELIX_INSIGHTS
§6) exactly.

| rank | workload | model | why it fits | cost and risk |
|---|---|---|---|---|
| **1 (recommended)** | **Language modeling on Reddit**, FedScale's per-user split (or StackOverflow). FedScale / REFL train ALBERT with *masked* LM (15% masking, `third_party/REFL/core/client.py:116`), reported as perplexity; a decoder trains next-token on the same split | **ALBERT-base (11M)** first (Oort and REFL ran it on Reddit; FedScale has the configs, so baselines are source-faithful), then a **small decoder, SmolLM2-135M with LoRA**, as the generative-era variant | keyboard LMs are FL's flagship deployment (Gboard); text is typed while the device is in use, so `usage` streams are literally right; natural non-IID per user; perplexity is the standard metric; a transformer adds a new model class | ALBERT: 45 MB on the wire (it shares weights across layers, not compute: per-token FLOPs ≈ BERT-base); SmolLM2 + LoRA: adapters of a few MB on the wire, but ~270 MB (fp16) of frozen weights per client in memory and per-token FLOPs ~1.5× ALBERT's, so check n 100 fits the bench before committing |
| 2 | **On-device assistant intent and slot classification**, MASSIVE (51 locales, split by locale + user) or Amazon reviews by user | **MobileBERT (25M) or DistilBERT (66M)**, full fine-tune or adapters | an "agentic" front end (routing a request to an app action) that runs on phones today; a BERT-class model is backprop-feasible on device; classification keeps eval cheap | MASSIVE has no real user ids (locale split only); reviews are not a mobile-native task |
| 3 | **Tool-call / function-calling fine-tuning for an on-device agent** (e.g. xLAM / Glaive function-calling, Android in the Wild UI actions) | SmolLM2-135M/360M or Qwen2.5-0.5B with LoRA | the most current story (agentic workloads on phones) | no natural user partition (synthetic or per-episode); generative eval is expensive; backprop on 0.36-0.5B on phones collides with FluxTune's memory-wall premise; long tasks (minutes) stress the trace model. Treat as a demo or discussion, not the third workload |
| baseline-safe fallback | **FEMNIST** (per writer, LEAF) or **OpenImage** (per author, FedScale) | ResNet-18 / ShuffleNet / MobileNetV2 | native to Oort, REFL and FedDance (EMNIST, Tiny-ImageNet), so it is easy to defend | not a new model class; adds breadth, not novelty |

**Recommendation.** Reddit language modeling with ALBERT-base as the third workload. Baselines run their own Reddit configs, the
deployment story is the strongest in FL, and the data-generation model matches our streams. Add SmolLM2-135M + LoRA on the same Reddit
split only if the bench budget allows; it answers "does Felix hold for a generative transformer" without leaving backprop-feasible scale.

### C.2 Plan outline: adding Reddit + ALBERT-base (and optionally SmolLM2-135M + LoRA)

> An outline to move into Felix's paper doc or FELIX_READINESS. Nothing here is implemented or decided; every item is a proposal
> with the open choice named. Code pointers are from a read of the tree at commit `96cfc10bf`.

**C.2.1 Workload variants (choose one or both)**

| variant | task / loss | metric | partition | why |
|---|---|---|---|---|
| A · ALBERT-base-v2 (11M params, ~45 MB) | masked LM, 15% masking, FedScale's block size | perplexity (lower is better) | FedScale `reddit/client_data_mapping` (per user), first n users after FedScale's `filter_less` | Oort and REFL ran exactly this, so baselines use their own configs |
| B · SmolLM2-135M + LoRA | causal next-token LM on the same Reddit split | perplexity (+ optional next-token top-1 accuracy) | identical to A | generative-transformer variant; only the model changes |
| alternative text source | StackOverflow, per user, next-word | perplexity | FedScale / TFF split | if Reddit's 25 GB download or cleaning is a blocker |

Open choices: n (300 like cifar, or 100 like speech); sequence / block length (sets cost); for B, LoRA rank and target modules.

**C.2.2 What carries over unchanged (dataset-agnostic, in `flame/`)**

Selectors (Felix, Oort, REFL, FedDance, FedBuff, FedAvg), aggregators and optimizers (FedBuff buffer, rate `new`, YoGi, SAA, FedAvg),
availability (notify, check-in ledger, REFL Prophet, oracle arms), `usage` data streams tied to each trace, the vclock / harness / bench,
telemetry, the launcher and `baselines.yaml` wiring. Speech was added the same way: one `DatasetSpec` in
[fl_data.py](../async_cifar10/fl_data.py) used by the shared trainer `async_cifar10/trainer/pytorch/main.py`.

**C.2.3 What has to be built (dev checklist)**

1. **Data:** download FedScale Reddit (`third_party/fedscale_dataset/download.sh reddit`, ~25 GB) under the data root; tokenizer
   (ALBERT's / SmolLM2's); a reference split file from `client_data_mapping` for the chosen n; held-out test set as FedScale builds it.
2. **`DatasetSpec` in `fl_data.py`:** model builder (HF `albert-base-v2`; SmolLM2 + LoRA via PEFT), train / test loaders, loss
   (masked-LM masking for A, causal shift for B), optimizer sort for the server table.
3. **Trainer loop (`main.py`):** token batches instead of image / waveform tensors; masking per batch for A; per-sample loss for the
   loss-based utilities (Oort, Felix, FedDance's importance factor) computed on LM loss; no augmentation (text: none, matching the
   baselines; record it as FT-N20 / FX-D233 did); bf16 autocast as speech.
4. **Eval metric everywhere it is read:** perplexity, lower is better. Today the pipeline assumes accuracy: `agg_eval`,
   `targetAccuracy` / `stableEvalsAboveTarget`, `fl1_verdict.py`, `target_report.py`, `felix_plots.py`, `accuracy_table`, `ceiling.py`,
   `fl_bound.py`, FedDance's accuracy-increment factor (decide: next-token or masked-token accuracy as its input). Likely the largest
   hidden cost.
5. **Task time:** the FedScale device model needs a per-model compute factor. FedScale scales ALBERT at 6.05× MobileNet
   (`model_factor`, `third_party/REFL/core/argParser.py:232`); our `compute_ms_per_sample` path (`flame/launch/spawner.py:159`) applies
   none today. Add a model factor (for B, scale by FLOPs vs ALBERT), keep EV3 / `task_time_audit.py` at 0 mismatches. Side check: whether
   speech's ResNet34 should have carried FedScale's resnet factor (2.4×).
6. **`datasets.yaml` entry:** `reference_split`, `model_kbit` (measured), GPU train slots, trainer hyperparameters, aggregator overrides.
7. **`baseline_reference.yaml` rows per baseline × reddit:** Oort / REFL from their Reddit configs (REFL `core/evals/configs/reddit/conf_exp.yml`:
   lr 1e-4, min lr 1e-5, batch 40, `filter_less` 41, YoGi); FedDance and FedBuff have no Reddit config, so their nearest config with a
   recorded deviation (ROBUST L34, L37); Felix ours. Sync K / async c by the Cohort rule.
8. **Per-baseline task timeout** recomputed for the new split (`flame/launch/task_timeout.py`).
9. **Tests and smokes:** stub mode for the harness, `test_baseline_reference.py` rows, one EV-graded harness pair per stack, then
   bench B2 for the new cells before any screen (Fast bench decision).

**C.2.4 Configs, targets and bounds**

- **Baseline configs:** from the papers' code first (code wins over paper text, L34), cited in `baseline_reference.yaml`.
- **Targets:** the papers' targets are for their setups (Oort: perplexity 39 on 1.6M clients; REFL Table 2: 43.6 semi-centralized with
  YoGi), not ours (n users, our traces, streams). Following FELIX_READINESS Decisions "Targets and comparison", derive our own:
  centralized ceiling (`ceiling.py`) and FL upper bound (`fl_bound.py`, ideal FedAvg) on our split, both in perplexity.
- **Sanity anchor:** our ceiling and bound should land in the papers' ballpark (~40 perplexity for ALBERT on Reddit); far off means the
  training recipe, tokenizer or masking differs, not the FL systems.

**C.2.5 Cost estimate (rough, from FLOPs; confirm with a 1 h bench smoke per stack)**

- ALBERT shares weights across layers but not compute: ≈ BERT-base, ~33 GFLOP per 64-token training sample (3× a ~11 GFLOP forward).
- At ~60 TFLOP/s achieved on an A40, Felix at cifar-like volume (~14-17k updates / h × ~40 samples / update ≈ 0.6M samples / h) costs
  ~1-1.5 GPU-h per 12 h-vclock leg, plus eval; Reddit clients hold few samples each, so samples per update may be lower.
- SmolLM2-135M + LoRA: ~1.5-2× ALBERT per sample (LoRA cuts bytes on the wire, not the backward pass).
- FB1-length runs (6 trace days, until the target) ≈ 12× the 12 h figure per leg; B roughly doubles the batch budget.
- Unknowns that move this: block length, local work per baseline (Oort 20 steps vs REFL / FedDance 1 step vs 1 epoch), eval set size and
  cadence, GPU memory per trainer (bench packing).

**C.2.6 Streaming data: how it could enter the paper (framing options, not decided)**

- *Realism and motivation:* the stacks compared (Oort, REFL, FedBuff, FedDance) assume static shards. Our `usage` streams (FELIX_INSIGHTS
  §6) make data appear while a device is in use, which fits Reddit (comments are typed in use) better than any other workload.
- *Evidence it makes FL harder* (FELIX_INSIGHTS §5): loss-based utility misreads a fit client as spent; multi-pass learners revisit few
  samples many times and overfit; streaming costs multi-step learners accuracy that 1-step stacks don't notice.
- *Claim scope:* "none of the systems we compare against model data arriving over time", not "no FL system does"; check continual /
  online FL work before writing related work.
- *Headline only with a mechanism:* streams can be a headline claim only if a Felix mechanism responds to them (candidate: the c ramp's
  loss phase); otherwise they stay setup and motivation.
- *Bounds under streams:* FX-N146 bounds use full shards (trace end), so a stream run approaches them only as data arrives. Options: report
  the gap to the bound over time; keep the static `s0` cell beside the stream cells to show what streaming costs.

**C.2.7 Suggested order (each step gated by the previous)**

1. Data + split + `DatasetSpec` + perplexity eval, stub-mode harness pass.
2. Task-time model factor + EV3 audit.
3. Baseline rows + timeouts; one harness pair per stack.
4. Centralized ceiling and FL bound on the split (targets).
5. Bench B2 on the new cells; 1 h smokes for cost (C.2.5).
6. Screens, then the FB1-length runs alongside cifar and speech.
7. Variant B only after A's cost is measured.

---

## Part D — FluxTune (starting point only)

> **Disclaimer.** Aish (with Dhruv) owns FluxTune's paper and will overhaul or replace this part. The latest solution, experiments,
> models and target accuracies are in markdowns on Aish's branch; this repo's FluxTune docs (FLUXTUNE_READINESS, EXPERIMENTS.md,
> the deprecated `fwdllm/fl_fwd_ft_*` and `fluxtune_contributions.md`) may be stale. Nothing here decides FluxTune's narrative,
> evaluation or wording.

- **Next step (Aish):** take the solution, experiment plan, models (FwdLLM's model set) and target accuracies from Aish's branch markdowns
  into FluxTune's paper markdown, written against Parts A and B.
- **Material in this repo that may help, if still current:** the scale-free accounting (budget, progress, time laws) in
  `fwdllm/fl_fwd_ft_solution.md` Part 3-4; the figure set in `fwdllm/figs/`; the figure pipeline `fwdllm/expt_scripts/make_paper_figs.py`;
  the baselines' naming firewall in [BASELINES.md](BASELINES.md) ("Felix ↔ FluxTune disambiguation").

---

## Part E — Rules both papers follow

- **Firewall** ([BASELINES.md](BASELINES.md) "Felix ↔ FluxTune disambiguation"): no shared results table; Felix-derived policies on the
  forward-gradient substrate are always labelled `Felix(P)`; Felix's paper never runs or cites FluxTune results.
- **Every number has a run ID** in a draft comment until camera-ready.
- **Compare at the target, not at a horizon;** report time and bytes to target; a stack that never hits reports its end accuracy, time and bytes.
- **Baselines are source-faithful;** every deviation sits in one generated table (`baseline_reference.yaml`, `baseline_deviations.py --md`).
- **Simulator fidelity is a result:** both papers carry a fidelity paragraph and an appendix table against real runs.

---

## Part F — Open items

1. **FluxTune:** models, targets and the system definition come from Aish's branch markdowns (Part D next step).
2. **Felix third workload:** confirm Reddit + ALBERT (C.1), and whether the SmolLM2 + LoRA variant is in budget.
3. **Real-device measurement:** a small on-phone measurement (Felix: notify / check-in cost; FluxTune: forward-pass time on an NPU)
   would close the most common systems-venue objection; decide per paper.
