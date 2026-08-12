# Forward-gradient fine-tuning — the problem, the model, and where it stands

> **This is one of two documents.** Here: *why* — the failure, its mechanism, the symbols, the model
> that predicts it, and what is still unknown. Companion: **[fl_fwd_ft_practice.md](fl_fwd_ft_practice.md)**
> — *what and how*: the shipped configuration, every knob and arm measured, dead ends, instruments,
> launch and reproduction. Its sections are `P1…P9`; **every number a run produced is owned there**,
> and this document cites rather than restates it.
>
> **Scope.** The measurements are FL (FluxTune / FwdLLM on agnews + DistilBERT), but the object of
> study is **backprop-free fine-tuning by directional derivatives**, of which FL is one deployment.
> §1–§7 are the FL instance; **§8 is the extension path** to single-device, centralized-ZO and
> datacenter settings.

---

> ## Start here
>
> | you want… | go to |
> |---|---|
> | **what we are building and why** | **§0.0** — goal, metric order, the autonomy test |
> | **the current state in one screen** | **§0** — load-bearing · settled · open · dead |
> | **what to do next** | [P5.3a](fl_fwd_ft_practice.md#p53a-execution-plan--from-here-to-a-zero-input-run) — the execution plan |
> | the three laws | §4.1 budget `B` · §4.2 progress `A` · **§4.6 time** |
> | the algorithm | §5.1 |
> | how a deployment self-calibrates | §5.5 |
> | a number a run produced | the practice doc — **never here** (R2) |
>
> **Parts 1–3 are background** — read them when a mechanism is in question, not to get oriented.

## §0.0 — What we are building

**The goal.** A backprop-free fine-tuning method that **minimises time-to-accuracy on a model and task
it has never seen**, by sensing every quantity it needs from the workload and the runtime. Nothing is
profiled in advance; nothing is a per-model constant an operator has to supply.

**The three success metrics, in strict order.** Later ones are never bought at the expense of earlier
ones:

| # | metric | the model's coordinate |
|---|---|---|
| **1** | **time to accuracy** | `t ≥ Λ²·p / (2·B·G_rule·K)` round trips (§4.6) |
| **2** | **stability** — reach a peak and hold it | `B ≤ B_max` (§4.1, §4.2) |
| **3** | **overheads** — communication, then compute | uploads `= K·t = Λ²·p/(2·B·G_rule)` |

**Why these do not have to be traded by hand.** §4.6 puts all three in one expression. Everything that
makes a run faster also makes it cheaper in bytes — except `K`, which is *free* in bytes and paid for
only in staleness. Metric 2 is not a trade against metric 1 but a **ceiling on it**: you run as fast as
the model's tolerance for accumulated junk allows. **The whole method therefore reduces to one decision
— how close to `B_max` to run — and `B_max` is sensed, not supplied** (§5.5).

**The autonomy requirement, stated as a test.** *Every constant in the loop must be either (a) exact
arithmetic on quantities the run already logs, (b) a hill-climb on something the runtime measures, or
(c) a property of the deployment the operator genuinely owns.* Anything that is none of these is a
profiling dependency and is a defect. §5.5 is the audit; **the surviving debts are `B_max` (sensed,
§5.5b) and `D` (forecast-only, §5.5a).**

> **Class (c) is deliberately narrow: the model, the PEFT scheme, and `p`.** Nothing else. In
> particular **`P` is sensed, not supplied** (§5.5f · D4) — the algorithm discovers the client compute
> budget by hill-climbing `P/τ(P)` rather than being told it. "No input" means **no tuning knob**: no
> learning rate, no variance threshold, no cohort width, no horizon, no safety factor, no probe count.

**What this rules out, and it is most of what the field does.** A learning rate. A tuned variance
threshold. A cohort width chosen offline. An anneal exponent fitted to a known horizon. A safety factor.
Each of those is a number that works on agnews/DistilBERT and silently means something else elsewhere —
which is §5.3's ratio principle applied to the *configuration* rather than to the arithmetic.

---

## §0 — Where the work stands

**In one line:** *we know why it diverges, we have four fixes that work, the sizing formula is 20×
optimistic because every reading's target is an 8-sample minibatch gradient — and G-1b has now closed
the last route by which a **pooling** knob could have bought stability, leaving the controller as the
only open engineering.*

*Read top to bottom: **what holds**, then **what is left**, then **what we tried and abandoned**.*

### Load-bearing — the six every decision is checked against

*If you read nothing else, read these. Everything in the rest of the document either derives from one of
them or is a consequence of one.*

| what | where |
|---|---|
| **The design principle.** Nothing may be compared against a fixed constant unless it is scale-invariant. The original defect was exactly this: estimator, step and gate all inflate together, so no quantity anywhere could be judged against a threshold | §2.4, §5.3 |
| **The autonomy principle.** Every constant must be exact arithmetic on logged quantities, a hill-climb on something the runtime measures, or a deployment property the operator owns. Anything else is a profiling dependency — the same defect one level up, in the *configuration* rather than the arithmetic | §0.0, §5.5 |
| **The time law.** `t ≥ Λ²·p/(2·B·G_rule·K)` round trips — **time-to-accuracy depends only on the budget spent, not on how it is split into `ρ`, `N`, `T`.** Equality **iff `s` is held constant**, which makes the commit gate a *time-optimality* mechanism. Achieved to 0.4%; the free `var` gate misses by 24% | §4.6 |
| **The budget law.** `Φ = ‖θ_T‖/‖θ_0‖ = e^B`, `B = ½·Σ ln(1+ρ_t²)` — closed form, **no free parameter**, 21 arms. Exact because every step is orthogonal to `θ` (ratio **1.000 ± 0.005**); generalises to `e^{((1+β)/(1−β))·B}` once anything correlates steps in time | §4.1, §2.1 |
| **The progress law.** Peak accuracy is monotone in `A = Σ ρ_t·cos_t·‖θ_tr,t‖`, validated out of sample at **2.5× better than `Λ`**. `Λ` is its fixed-`p` special case and is still the right read inside one model | §4.2 |
| **The criterion.** `ρ ≤ s·cos` at **`s` ≈ 2.9** — discriminates **16/16** over the portfolio, and the two laws independently bracket `s` at 2.6–4.3. **It sizes a pool; it does not bound a trajectory** — `s` cannot enter `B`, so no `s` stops a constant-`ρ` arm from turning (G-1b) | §4.4, §4.5 |
| **The estimator.** `cos = D·√(G_rule·N/p)`. The **scaling** holds to 1% across rule, `p` and `N`, and the gap is entirely **data-side** and not poolable. **`D`'s value is no longer settled**: 0.045–0.051 near init, **0.10–0.15** on the two arms that trained to peak, and within those it tracks `Φ` rather than accuracy — see D-2 | §6, §6.3 |
| **The shipped stack** — four fixes, all enacting to spec: average all `P` probes · trust-ratio step · Robbins–Monro anneal at `exp` = 0.25 · shrink `p` for **memory and compute only** | §5.1, [P2](fl_fwd_ft_practice.md#p2--the-recommended-stack), [P3](fl_fwd_ft_practice.md#p3--knob-ledger--what-each-lever-actually-did) |

### Also settled — do not re-measure

| what | where |
|---|---|
| `ρ ∝ η` and `ρ ∝ 1/√(K·I)`; the naive `ρ ∝ 1/√K` is false because the gate hands a `K` rise back as fewer iterations | §2.2 |
| The variance gate is a dimensionally-wrong `N`-controller — and is the *accidental* stabiliser | §2.3 |
| The defect is **shared with sync `fwdllm`**, not a FluxTune bug | §2.5 |
| Heterogeneity (α) acts **only** as a step-size multiplier, through the gradient scale | §2.6 |
| Collapse is **directional degeneracy** of the head, not logit saturation | §2.7 |
| Probe selection by `\|d\|` is stability-neutral **by construction**; the candidates carry no other signal | §3.2 |
| **At a pinned `ρ*`, `p` is inert** — `‖θ_tr‖ ∝ √p` cancels the `cos` gain and `B` never saw `p` | §4.2 |
| **Momentum is `ρ` in disguise** — `√x` progress for `x` budget, `x = (1+β)/(1−β)` | §5.2 |
| **The commit gate converts pool into commit rate at a net gain** — `2.13×` the control's progress per unit federated time, and the `√N` price per commit is exact to 3 decimals | §4.5, [P5.2](fl_fwd_ft_practice.md#p52-scored-nodes--g-1b-g-1-q3-s1-h-p) |
| The `cos` probe's reference batch was one client's non-IID shard — every number it logged is void, **the norm ratio included** | §6 |
| **`s` is an efficiency knob, not a safety one.** Both `const` arms turn; a 4× pool doubles `Λ/B` exactly and buys a higher peak, sooner, decaying 2–3× slower at matched `Φ` — but not a held one | §4.5 |
| **Q1** — the collapse budget is **relative** (`Φ`), not absolute (`‖θ_tr‖`) | §7.1 |
| **Q2** — `‖θ_tr‖` is mostly a **symptom**; perfect decay recovers only 20–25% of the damage | §7.2 |
| Backprop reaches **0.90** where the best forward-gradient arm reaches **0.876** — the cost of going backward-free | §7.3 |

### Open — and this is the entire list

| # | question | blocking what | how it closes |
|---|---|---|---|
| **D-2** | **`D` is 2–3× larger on the two arms that trained than near init** — 0.148 ± 0.020 and 0.104 ± 0.017 against 0.051 ± 0.011 and 0.045 ± 0.005. But it is **not "training state"**: pooled across both trained arms `D` is *non-monotone* in accuracy and the highest-accuracy bin reads the **lowest** `D`, while pre-turn 0.103 ± 0.021 steps to post-turn 0.163 ± 0.025. `const` vs `rm` is the leading remaining confound | the sizing formula and §8.1's generality — `N_req ∝ 1/D²`, so 2× is 4× in pool | **on the rig, not on an arm**: single-commit `cos` has SNR ≈ 1 (16–18% of fires come back negative), so only a fixed **trained checkpoint** with unlimited probes can separate `const` from `rm` |
| **H-S** | **A 3.5× shadow loss that is not data-side.** The rig reproduces `L` and gets `S` = 1.68 where the arms read 0.48. Prime suspect: the FD chord, `h‖v‖` = 6.71 against `‖θ_tr‖` = 6.75 — every probe steps a full parameter-norm, so `d` is a chord-averaged slope, not `⟨g,v⟩` | the last unexplained factor in `cos` | rig: true `⟨g,v⟩` vs the central FD at the shipped `h` (§6.3) |
| **C-1** | **The controller — now an implementation task, not a research one.** The stopping rule is **validated**: on replay, a `Φ` trigger at 2.7–3.0 banks the peak to within **0.005 mean / 0.014 worst over 10 arms**, against **0.141 / 0.595** for running to the end ([P5.2a](fl_fwd_ft_practice.md#p52a-the-φ-stop-counterfactual--c-1s-core-rule-validated-on-replay-measured-10-arms)). What remains is wiring: sense `B_max`, adapt `K`, stop on saturation | every arm's ending, and metric 1 of §0.0 | build it. The three unbuilt pieces are listed in §5.5-1; none needs new science |
| **B-1** | **Does `B_max` transfer?** It is the **one** constant left on the operating path (§5.5) — `ρ*`, `s`, `N` and the stopping rule all follow from it by arithmetic, and `D` and the `Λ` curve are needed only to *forecast*. Three arguments for transfer (geometric reading, §7.1's `p`-invariance, 7/7 arms) and **zero cross-model measurements** | whether any of this is a method rather than a result | run `probe_inflation_damage.py` on a second model and a second task — ~6 evals each, forward passes only, no training. Then calibrate its known conservative bias (§5.5b) |

**What is *not* open: the optimizer, or any pooling knob.** The four fixes enact to spec and the
dynamics are closed by the two laws. **What moved is where the remaining loss lives** — not in the step
rule, the combination rule or the gate, but in the *data* each reading is taken on. G-1b adds the
sharper statement: **every pooling stage buys progress per unit budget and none of them buys budget**,
so the only quantities that can stop a collapse are the two that enter `B` — `ρ` and `T`. That is a
controller, and it is C-1.

### Dead — what we tried and abandoned. Do not re-propose

*Every row is a belief this program actually held and paid for (R5: append-only). Superseded **numbers**
live in [P8.1](fl_fwd_ft_practice.md#p81-superseded-numbers--quote-check); the process lessons in
[P9.3](fl_fwd_ft_practice.md#p93-process-lessons).*

| we believed | what killed it |
|---|---|
| the collapse is intermittent — a bad seed or flaky infra | runs cut at ~95 commits stop *at the peak*; two 4 h runs agree on every constant (§1.2) |
| it is an aggregation bug — pool harder, or lower `η` | `K` ≥ 20 reaches 0.860 where `η` = 0.002 reaches 0.601 at the same commit: same stability, one free and one at full 1:1 cost |
| lowering `η` is the fix | pays 1:1 in progress and still has `Σρ² = ∞` |
| the variance gate is dead, or live only at `K` ≥ 20 | `var ∝ b²‖g‖²/n`, so it is live wherever it can reach its threshold (§2.3) |
| "there is a critical `ρ` ≈ 0.09" | horizon artifact; **every** `ρ > 0` inflates geometrically (§4.3). *"Walk `ρ*` up until it breaks"* is withdrawn — this program's own recommendation, one revision earlier |
| safety factor `s` = 0.3–0.5 | derived wrong; the two laws give 2.6–4.3 (§4.4). It was silently absorbing a shortfall of unknown size *and direction* |
| the `K` ≥ 30 / ≥ 51 cohort-width requirement | an artifact of `gate_safety_s` = 0.4 left in the yaml after §4.4 superseded `s`. Cost three inconclusive gate runs and a wasted node |
| keep selecting probes by `\|d\|` | `b²/a = 1` — selection raises the shadow quadratically and the length linearly, and they cancel **exactly**, for any `E` (§3.2). C1 refuted as written |
| "`p` first" in the priority order | `‖θ_tr‖ ∝ √p` cancels the `cos` gain and `B` never saw `p` — inert at a pinned `ρ*` (§4.2). The ladder's apparent win was raw SGD lowering `ρ` |
| server momentum as a free pooling lever | correlating steps costs budget at `(1+β)/(1−β)`; orthogonality ratio 1.007 / 2.981 / 6.615 against 1 / 3 / 7 (§5.2). S1 dropped |
| probe noise dominates data noise by ~40× | never measured, and false: `L` is flat over **64×** in bin size where `1/√B` predicts an 8× fall (§6.3). Inverts the whole "spend compute on probes" instinct |
| bigger bins as a way to recover `D` | same measurement — `D` is a constant of the setting, not a pooling deficiency (§6.3) |
| every `cos_ground_truth` logged before 2026-08-10 | the reference batch was **one client's non-IID shard**, anti-correlated (−0.46) with held-out truth (§6). Three failed consistency checks were written up as findings about the optimizer, for a full revision — **the norm ratio included** |
| "the `cos` shortfall is instrumental; a clean reference restores the closed form" | the clean reference reproduces it exactly — `D` = 0.050, invariant to rule, `p` and `N` (§6.1) |
| client disagreement, and reference noise, as the cause of the shortfall | both quantitatively excluded (§6) |
| "`Φ` decides whether *any* arm holds its peak" | true only for arms that actually learned (§4.2) — the missing "peak ≥ 0.80" qualifier made G-1's sinking condition unreadable |
| testing a setpoint through a schedule that spends it | `gate_rho_ref=setpoint` sizes from `ρ*₀` while `rm` anneals the step, so realised `ρ/cos` fell 2.84 → 0.95 in 79 commits. Cost G-1's whole setpoint half |
| **a lower `gate_safety_s` will hold the peak** — arm 1's own pre-registered fallback | `B` contains no `N`, no `cos` and no `s`, so both `const` arms spend identically and **both turn**, at `Φ` = 3.43 / 3.79 (§4.5). Cost one arm, and the belief that the gate was a stability mechanism at all |
| **extrapolating `A` per vclock-hour into an accuracy ranking** | `A` accumulates *through* the turn while accuracy falls, so G-1's projection ordered its two arms backwards (§4.6) |
| split-half cosine as a gate or an online `cos` estimator | measurability wall at this `p` — per-commit SNR 0.07. B9 built and dead; B10 parked behind it |
| `n_eff` as a controller input | an identity, blind to directional disagreement (§5.3). B11 demoted to an audit |
| the `‖θ‖²` log-log slope as a stability score | bounded by 1 under trust-ratio *by construction* (§4.3) — every "sub-linear ⇒ safe" reading is vacuous |
| an audit flag is free because it is emit-only | the `cos` probe backprops 1024 samples per commit: **85 s** against 1.64 s for the rest of the commit path. Cost eight arms across two nights at 4–15% of their vclock budget |

**Two patterns account for most of this table, and both are cheap to check for.** The `p` census, the
`ρ`/orthogonality reconstruction and the `cos` reference were each *an instrument whose arithmetic was
right and whose input was not*. `gate_safety_s` = 0.4 and the annealed-schedule row are each *a value
that stayed in a config after this document superseded it*. **Every one was caught by a ratio that came
out wrong, never by an accuracy curve** — which is the argument for the standing preflights
([P9.1](fl_fwd_ft_practice.md#p91-preflight)).

### How to maintain these two documents

| # | rule |
|---|---|
| **R1** | **Edit in place.** Rewrite the row or paragraph that changed. No changelogs, no dated "update:" notes. Git holds history. |
| **R2** | **One number, one home.** Model and mechanism live here; run results, flags and commands live in the practice doc. If you restate a number across the two, delete the copy and cite. |
| **R3** | **Evidence, model and recommendation move together.** A new finding updates the model here *and* re-checks the recommended stack ([P2](fl_fwd_ft_practice.md#p2--the-recommended-stack)) in the same edit. Never leave a corrected model beside a stale recommendation. |
| **R4** | **Tag claims** `MEASURED` (telemetry + run id) · `DERIVED` (arithmetic on measured inputs) · `ANALYSIS` (rests on a stated assumption) · `HYPOTHESIS`. Untagged prose is not evidence. |
| **R5** | **Dead ends are permanent.** [P6](fl_fwd_ft_practice.md#p6--dead-ends--do-not-retry) is append-only and is the first thing to read before proposing anything. |
| **R6** | **Prediction before run.** An arm enters the queue with a number *and* a sinking condition written down first. |
| **R7** | **Cheapest instrument first** ([P7](fl_fwd_ft_practice.md#p7--the-instrument-ladder)). A full run *confirms* a hypothesis; it never explores one. |
| **R8** | **§8 is frozen** until the open questions close. |
| **R9** | **Prose budget.** A paragraph that does not change a decision is deleted. Accuracy is a *result*, never the stability evidence. |

---

# Part 1 — The problem

## §1.1 What the method is, and what it costs

Fine-tune an LLM across federated clients **without a backward pass**. Each client perturbs the
trainable weights along random directions, measures the loss change with forward passes only, and
uploads a scalar-scaled direction. The server pools them and steps. Nothing stores an activation
graph, so peak memory is a forward pass and the operator set is inference-only — that is the whole
point, and the part nobody disputes.

The price is that **one measurement returns one number instead of `p`**:

| | backprop | forward-gradient |
|---|---|---|
| cost of one measurement | 1 fwd + 1 bwd ≈ 3 forward-equivalents | **2 forward passes**, no backward |
| what it returns | all `p` components of `g`, exactly | **one scalar** `d = ⟨g,v⟩` — the slope along `v` |
| memory | stores activations | nothing beyond a forward pass |
| the update produced | `−η·g` — right direction, known length | `−η·d·v` — right **on average**, ~0.1% aligned individually |
| noise sources | data sampling | data sampling **+ which direction you asked about** |

`ĝ = d·v` is unbiased (`E[d·v] = g`) with enormous variance. *On a hillside in fog:* backprop feels
the slope in every direction at once; forward-gradient picks one random direction, takes a test step,
feels whether it went up or down, steps back. **One reading is nearly worthless; a thousand averaged
readings are a slope meter.** Everything here is about whether enough readings can be averaged, fast
enough, to train — and about the specific way our implementation failed to.

## §1.2 The failure

Real 3.93 h run, 189 commits, agnews / DistilBERT-base + AdapterHub adapters:

```
 0.00h  acc 0.398  loss 1.380     <- init, ln(4) = 1.386
 2.13h  acc 0.846  loss 0.554     <- PEAK
 2.95h  acc 0.738  loss 0.692     <- degradation starts INSIDE round 1
 3.87h  acc 0.250  loss 2.371     <- one class, worse than chance
```

**Rise, peak, collapse to a single class.** A second 4 h run agrees on every derived constant, so this
is deterministic, not a seed. Runs cut at ~95 commits stop *at the peak*, which is why it looked
intermittent for months.

## §1.3 What it is not

| ruled out | evidence |
|---|---|
| overfitting | **train** loss diverges too (1.355 → 0.66 → 1.72) |
| plateauing / running out of signal | final loss **2.37 > ln 4** — answering *wrongly and systematically*, worse than "I don't know" |
| a round-boundary or data-order effect | degradation starts inside round 1 |
| the finite-difference approximation | the FD displacement gets *relatively smaller* over the run while the failure gets worse |
| a bad seed or flaky infra | two independent 4 h runs agree on every constant |
| something specific to async FluxTune | sync `fwdllm` shows the identical signature (§2.5) |

## §1.4 Definition of done — tracked

| # | criterion | status | what closes it |
|---|---|---|---|
| **1** | An arm reaches **peak ≥ 0.86** and **ends within 0.015 of its peak** at ≥300 commits, without hand-tuning per model or per α | **MET** — `065837` peaks 0.865 and ends 0.862 over 715 commits. `145729` sets a new best **peak** (0.876) but ends 0.793, so it meets the first clause only | — |
| **2** | The setpoint is **computed, not searched** — from the config, before the run | **MET for `ρ*`, by a different route.** `ρ* = √(2·B_max/T)` (§4.4) needs no `cos` and no `D`, and returns 0.062 where the portfolio searched its way to 0.06. **Still open for the *forecast*** — predicting the accuracy that `ρ*` will reach needs `cos`, hence `D` (D-2), and the `Λ` curve | the operating half is closed; the forecasting half is D-2. §5.5a draws the line |
| **3** | Stability readable in **~20 commits** from `B` and `A`, not a 4 h accuracy curve | **MET** — both are exact at any horizon, and `B` predicted `112201`'s `Φ` to 1.1% over 1,364 commits | — |
| **4** | Every constant in the loop is dimensionless (§5.3) | **PARTIAL** — met for the step rule (α moves `ρ` by 0 to 6 s.f.); the gate is dimensionless in form but its `s` is still an empirical number. G-1b fixed `s` = 1.5 as a *value* and reclassified what it is for | making `s` derivable is item 2; it is no longer on item 5's path |
| **5** | *(new)* The controller **holds** the trajectory rather than merely sizing it once | **OPEN, but the rule is validated** — a `Φ` stop at 2.7–3.0 banks the peak to 0.005 mean / 0.014 worst over 10 arms on replay (P5.2a). Unbuilt, not unknown | wire it: sense `B_max`, adapt `K`, stop on saturation (§5.5-1) |

**Item 5 is the one this program has been circling without naming.** Every arm to date either anneals on
a schedule chosen offline (`rm`, `exp` = 0.25) or holds `ρ` fixed — and §4.4's design rule assumes the
horizon `T` is known in advance. The two `const` arms are the clean demonstration that a *correct*
one-shot setpoint still collapses if the horizon outruns it, **at any pool size**.

> **Item 5 has its sensor, and the sensor has been scored.** `Φ = e^B` is exact per commit from `ρ`
> alone, carries no free parameter and is readable in ~20 commits. Replayed as a stopping rule over 10
> arms, a trigger at **`Φ` = 2.7–3.0 gives up 0.005 of peak on average and 0.014 at worst**, against
> **0.141 / 0.595** for running to the end (P5.2a). **It has still never been run as a controller** —
> it was measured on arms that were not trying to obey it, which is the weaker of the two claims but the
> cheaper one to have made.

---

# Part 2 — Mechanism: why it diverges

## §2.1 Every step is orthogonal to `θ`. **MEASURED.**

`‖θ+Δ‖² = ‖θ‖² + 2⟨θ,Δ⟩ + ‖Δ‖²`, so the ratio of observed norm growth to step energy has an exact null
at **1.000 = the step carries no component toward or away from where the model stands**. Measured per
25-commit block: **1.000 ± 0.005 in every block of every arm** — including arms that reach 0.86 and
hold it.

**Orthogonality is not itself the defect** — `g` is nearly perpendicular to `θ` at high `p`; any
optimizer shows this. What it establishes is that the norm grows by the *full* `‖Δθ‖` every commit,
with no cancellation, which turns the trajectory into an **exact difference equation** (§4.1). Growth
per se is normal (adapters init near zero); what is atypical is that it is geometric and unbounded.

## §2.2 The step never anneals, and it feeds itself. **MEASURED.**

`ρ ∝ η` holds to 1% across a 20× sweep in `η`. `ρ ∝ 1/√N` holds to 4% over a 1.9× range in `N`, where
`N = K·I` is the *pooled upload count*, not the cohort width. **`ρ ∝ 1/√(K·I)` is the most load-bearing
measured law here.** The naive `ρ ∝ 1/√K` is **falsified**, because raising `K` makes the gate commit
sooner and cuts `I` — it is `N`, not `K`, that sets the aim (§2.3).

Whether growth is geometric or arithmetic is decided by the *absolute* step. At `K = 50` the absolute
step is flat, so `‖θ‖²` grows linearly and safely. At `K = 10` it grows, so `‖θ‖²` is geometric with a
doubling time of ~68 commits. **Divergence is present at commit 1** and takes ~150 commits to become
visible: steps do not become more wrongly aimed over time; what grows is the absolute step and the norm
it is applied to.

**Why the absolute step grows: the gradient genuinely grows.** A backprop probe on a fixed batch shows
`‖g‖` tracking `‖θ_tr‖` with exponent ≈0.9 in the rising phase, matching the ≈0.8 inferred from `rms|d|`
through a completely different instrument. So the loop

> *noise inflates the norm → bigger gradient → bigger absolute step → more noise*

is **real and closed**. Norm control is therefore potentially **curative, not cosmetic** — which is what
raised the bar for the weight-decay question (§7.2). In the stable `mean` arms this loop is absent:
`‖θ_tr‖` and `‖g‖` both sit flat for the whole run.

## §2.3 The variance gate is dimensionally wrong — and is the accidental stabiliser. **MEASURED.**

The gate asks the right question — *have I pooled enough readings to trust this direction?* — with a
statistic that carries units. What `var` computes is the per-coordinate variance between two half-means,
which for two samples is exactly `‖G_A − G_B‖²/(2p)`; with `u = d·v` and `‖v‖² ≈ p`:

```
var  =  ||G_A - G_B||^2 / (2p)   ~=   2 * b^2 * ||g||^2 / n
```

**Both legs of the gate's behaviour fall out of that one line.**

1. **`var ∝ 1/n` is why it is an `N`-controller at all.** Across `K` = 20/30/50 the realised
   `N = K·I` lands in a narrow band regardless of `K` — the gate hands most of a `K` increase back as
   fewer iterations. At `K` = 10 the target is unreachable, `max_iter` binds, and `N` pins low.
2. **`var ∝ ‖g‖²` is the units bug.** The achievable variance floor drifts 36× over a run — *exactly*
   `(6.0×)²`, the square of the `‖θ‖` growth. Because the ruler carries units of `‖θ‖²`, holding
   `var ≤ 0.3` forces `N ∝ ‖θ‖²`, hence `ρ ∝ 1/‖θ‖`, hence a **constant absolute step**.

> **The dimensional bug is the stabiliser.** Wrong units are exactly what turn a fixed threshold into a
> `ρ ∝ 1/‖θ‖` anneal — and since `‖θ‖ ∝ √t` under a constant absolute step, that is Robbins–Monro
> `ρ_t ∝ 1/√t` *by accident*. **It is not a fix:** `Σρ_t² = Σc/t` diverges logarithmically, so collapse
> is deferred (extrapolated to commit 1,200–1,700), not prevented — and the threshold hard-codes one
> trajectory. Three objections stand against the statistic itself: it has units, it cannot see
> directions (it is computed from scalars), and it never references the step.

This is the precise departure from FwdLLM, whose central pooling mechanism this is. The replacement is
not "a better commit test" but **an explicit `N`-controller with `ρ` as its sensor** — which is what the
var gate turns out to have been all along.

## §2.4 The single defect

**Nothing in the pipeline is scale-invariant.** The estimator (`|d|` tracks `‖θ‖`), the step
(`‖Δθ‖ ∝ |d|`) and the gate (`var ∝ |d|²`) all inflate together, so **no quantity anywhere can be
meaningfully compared against a fixed constant.** All three legs are one violation of the ratio
principle (§5.3), with the twist that decides the fix: **the third violation partly cancels the first
two**, so the replacement must supply that anneal *deliberately* rather than merely remove the bug.

## §2.5 It is cross-baseline, not FluxTune-specific. **MEASURED.**

Sync `fwdllm` over 131 commits: orthogonality ratio 1.000 ± 0.008, same `ρ` magnitude at commit 1,
geometric norm growth with a ~106-commit doubling time. **Same signature, same magnitude, shared
`_server_update_step`.** So the step rule and the anneal are **cross-baseline hygiene, not a FluxTune
contribution** (§7.4).

One consolation prize (**ANALYSIS**): at matched commits `ρ·√N` is 1.68 for fluxtune vs 1.01 for
fwdllm, which at matched `‖g‖` puts FluxTune's `|JVP|` selection at `G_rule ≈ 3` and **FwdLLM's
cosine-similarity probe selection at `G_rule ≈ 1`, i.e. no better than random.**

## §2.6 Heterogeneity is a step-size multiplier, and nothing more. **MEASURED over 1000× in α.**

**(a) α enters through the gradient scale, and only through it.** At the same `‖θ_tr‖`, `rms|d|` spans
2.5× across α = 0.1 / 1 / 100, propagating straight into `ρ`. **Under raw SGD, heterogeneity is a hidden
×1.6 on the relative step** — and the diverging arms are exactly the low-α ones.

**(b) The pooling invariant is not `ρ·√N`** (it spans 2.3× across α). What is invariant to ±1.5% across
all seven α×K arms is the form with the gradient scale divided out:

```
rho * sqrt(N) * ||theta_tr|| / ( eta * rms|d| * sqrt(p) )  =  0.53 +- 0.01
```

**(c) The gate's liveness is set by `‖g‖`, so α decides it as much as `K` does.** `var_threshold = 0.3`
sits *exactly* on the α=1, `K`=20 floor — tuned to one point of a two-dimensional surface. `K` = 10
survives at α = 100 while `K` = 20 does **not** stabilise at α = 0.1, so **"`K ≥ 20` is safe" is false
as stated**; the true statement is *"the gate is safe while it can reach its threshold"*.

**(d) The trust-ratio step removes the α multiplier by construction.** At a pinned `ρ*`, α = 1 and
α = 0.1 give `ρ` **identical to six significant figures at every commit**, while `var` still shows the
1.6× gradient-scale ratio. **The mechanism is still there; the step no longer feels it.** The residual
3-point accuracy gap is data difficulty at a matched step — what a portable optimizer should look like.

## §2.7 Why it looks healthy for three hours, and what the collapse is

Early on the coherent term (`T·ρ·cos`) beats the `√T` noise, so **the accuracy climb is real learning**;
the inflation term is exponential the whole time but invisible until it is not. `cos` does not degrade
over time. **We never reached a minimum and walked back out** — accuracy peaked because a rising linear
term and a falling term crossed. (*A convergence-detection rule still does not exist.*)

**The collapse is directional degeneracy, not logit saturation.** **MEASURED:** at collapse
`top_class_share → 1.00`, but `logit_norm` is the *same* as in the healthy arm and prediction entropy
stays high. **The head's decision direction is destroyed while its scale is unremarkable.** The
objection *"some of the weight increase was in the right direction"* is answered by arithmetic: ≤1.5% of
each step is aligned, so the coherent part accumulates to a fraction of one norm over ~190 commits while
the norm grows 7×.

## §2.8 The three production monitors

α-independent across every arm measured:

| monitor | reading |
|---|---|
| **`‖θ_tr‖`** | peak accuracy at **28–39**; −5 points at **47–52**; `top_class_share > 0.9` at **62–73**. *The one worth wiring to an alarm.* |
| **`ρ = ‖Δθ‖/‖θ_tr‖`** | exact per commit; feeds `B` and `Λ` (§4) |
| **`top_class_share`** | 1.00 is the collapse fingerprint; `logit_norm` does **not** discriminate |

**Score `B` and `A` — never accuracy, never the `‖θ‖²` log-log slope.** Both are exact at any horizon
and readable in ~20 commits. Use `Λ` only to compare arms at one `p` (§4.2).

---

# Part 3 — Symbols and estimator shape

## §3.1 Symbols

Per-commit unless stated. "Dimensionless" means a pure ratio, so comparing it against a fixed constant
is legitimate — that property is the whole fix (§5.3).

| symbol | what it is | value here |
|---|---|---|
| `θ_tr` | the **trainable** slice (adapters + classifier) | `‖θ_tr‖` = **13.35** at init |
| `p` | trainable dims = probe dimension | **450,340** (`rf`=16); 229,012 (`rf`=32); 118,348 (`rf`=64) |
| `v` | one probe direction, raw Gaussian draw, **not** normalised | `‖v‖ = √p ≈ 671` |
| `h` | finite-difference spacing | 0.01 → displacement `h‖v‖ = 6.71` — **an instrument, not a step size** |
| `d` | the JVP: **one scalar** per probe, `d ≈ ⟨g,v⟩` | rms 3.4 → 15.9 over a diverging run |
| `g` | true gradient at `θ` | measured by the server-side backprop probe (§6) |
| `G` | server's pooled update before the step | the direction actually taken |
| `P` | probes per trainer per iteration | 10 |
| `K` | trainers pooled per commit (`agg_goal`) | 10 shipped, and sufficient — the ≥30 requirement is withdrawn (§5.6) |
| `I` | iterations over the **same data bin** before committing | 18.5 at `K`=10 (capped), 8.2 at `K`=50 |
| `N = K·I` | uploads pooled server-side. **Not** the client population | 185 at `K`=10; 300–500 at `K`≥20 |
| `C` | concurrency pool — caps `K` | 30 |
| `η` | server learning rate (a **knob**, superseded by `ρ*`) | 0.01 |
| `ω` | per-upload aggregation weight | 0.70–0.87 — a re-weighting, **not** a step size |
| **`ρ`** | **relative step `‖Δθ‖/‖θ_tr‖`** — an *outcome* under raw SGD, a *knob* under trust-ratio | **0.16 at commit 1** shipped |
| `ρ*` | the relative step an operator *sets* under the trust-ratio rule | the knob `ρ` should have been |
| **`cos(G,g)`** | fraction of the step aligned with the true gradient | **0.036–0.067 closed form; never validly measured** (§6) |
| `E[v∥²]` | probe-selection gain | 2.988 at `P`=10, 4.744 at `P`=30 |
| `G_rule` | pooling gain of the combination rule: `E[v∥²]` if selecting, `P` if averaging | 2.988 → 10 |
| `a`, `b` | estimator shape constants (§3.2) | properties of the **rule**, not the data |
| **`B`** | **budget spent** = `½Σln(1+ρ_t²)`; `‖θ_T‖/‖θ_0‖ = e^B` | §4.1 |
| **`Λ`** | **progress banked** = `Σρ_t·cos_t` | §4.2 |
| `Φ` | inflation `= e^B` | ≤3.63 holds, ≥4.2 degrades (§4.2) |
| `var` | commit gate's statistic: spread of `d` across the pool | drifts with `‖θ‖²` (§2.3) |

**`v` is a raw Gaussian draw**, never normalised. Three consequences: `‖v‖` concentrates at `√p` to
0.07%, so **normalising `v` is a no-op for `cos`**; `v` sets the probe displacement `h√p`, so **changing
`p` silently changes the FD spacing** (hence the `FWDLLM_FD_SCALE_INVARIANT` rescale); and **isotropy is
exact by construction**, which licenses the `1/√p` overlap below.

Where the dimensions actually live matters, and cost a revision: `create_model` builds 1.04M trainable
params, but the trainer replaces `pre_classifier` with an empty module *before any probe is drawn*, so
production `p` is **450,340** and `‖θ_tr‖` is 13.35, not 20.36. Consequences: `trainable_scope:
adapters_only` is a **no-op**, the FD displacement is 50% of `‖θ_tr‖` rather than 76%, and every `cos`
prediction rose ×1.52 when this was found. (**`h` itself is pinned** between truncation error above and
fp16 catastrophic cancellation below; it cannot be reduced.)

## §3.2 Estimator shape: why "pick the best probe" cannot work

Split any upload `u` into its component along `ĝ = g/‖g‖` and the rest: `u = α·ĝ + u⊥`.

- **`a` measures the shadow**: `E[α] = a·‖g‖` — how much of the true gradient survives.
- **`b` measures the total length**: `E‖u‖ = b·‖g‖·√p`, so **`b = 1` is "as long as one raw probe"**.

A single upload's aim is `cos(u,g) = (a/b)/√p`. For the three rules that matter:

| rule | `a` (shadow) | `b` (length) | aim `a/b` | **stability `b²/a`** |
|---|---|---|---|---|
| one raw probe `d·v` | 1 | 1 | 1 | 1 |
| **select best of `P` by `\|d\|`** | `E` | `√E` | `√E` | **1** |
| **average all `P`** | 1 | `1/√P` | `√P` | **1/P** |

The middle row answers *"why doesn't picking the best probe help?"* Selecting on `|d|` raises the
shadow **quadratically** and the length **linearly**. Aim genuinely improves — but stability depends on
`b²/a`, in which the two cancel **exactly, for any `E`**. `a` and `b` are properties of the rule, known
in closed form before the run; the one empirical input is that `d` is Gaussian, which is verified
(**MEASURED** — see [P3](fl_fwd_ft_practice.md#p3--knob-ledger--what-each-lever-actually-did)).

**Pooling `n` uploads:** the shadow is identical in each and survives untouched; the perpendicular junk
shrinks by `√n`.

```
shadow of pooled G  ~  a * ||g||                    <- unchanged by pooling
||G||               ~  b * ||g|| * sqrt(p/n)        <- shrinks as 1/sqrt(n)
cos(G, g)           =  (a/b) * sqrt(n/p)
rho                 ~  eta * ||G|| / ||theta||      <- also shrinks as 1/sqrt(n)
  =>  rho / cos     ~  (b^2/a) * p / n
```

Read the last line as **safety = (a property of the rule) × (dimensions per pooled reading)**. Improve
it via a better rule (`b²/a`), fewer dimensions (`p`), or more readings (`n`). This is the whole lever
table (§5.2).

**Status: both halves are now MEASURED, and the rule-dependence of both is confirmed** — `L` and `S` are
each invariant across a 5.5× swing in `b` and a 3× swing in `a` (§6). What the table omits is that `g` is
not the same object for every upload: each `u` measures its own client's 8-sample gradient. That
substitution costs a constant factor `D` ≈ 0.05 and is the whole remaining gap (§6).

Four consequences of the estimator worth stating plainly:

**(a) The parameter count is the adversary.** A random direction in `p` dimensions overlaps a fixed
target by ≈ `1/√p`; at `p = 450k` one probe is ~0.15% signal. No tuning removes this — and `p` is itself
a *lever*.

**(b) Averaging is the only free lever, and it pays twice.** Over `n` independent readings signal adds
**linearly** while near-orthogonal noise adds **in quadrature**. The average is simultaneously **better
aimed** *and* **shorter** — safety improves as `1/n`, not `1/√n`.

**(c) Misaim never cancels in length.** A step perpendicular to `θ` gives `√(‖θ‖²+‖Δ‖²) > ‖θ‖` for
*every* perpendicular direction; sign does not matter, only length. This is §2.1, and it is what makes
§4.1 exact.

**(d) Inflated weights destroy a classifier by randomising which class wins**, not by saturating it
(§2.7).

---

# Part 4 — The model: two conserved quantities

*This is the current model, incorporating all evidence. It is scored against, not decorated with,
history.*

## §4.1 The norm law — how much budget a trajectory spends

§2.1 makes the trajectory an exact difference equation `‖θ_{t+1}‖² = ‖θ_t‖²(1+ρ_t²)`. Integrating gives,
with **no free parameter**:

```
INFLATION   Phi = ||theta_T|| / ||theta_0||  =  exp( B ),   B = (1/2) * sum_t ln(1 + rho_t^2)
PROGRESS    Lambda = sum_t rho_t * cos_t,     cos_t = sqrt( G_rule * N_t / p )
EFFICIENCY  Lambda / B  =  2 * cos / rho      <- exact by construction
```

**The norm law holds across 21 arms** spanning `ρ` 0.0002–0.22, `N` 10–200, α 0.1–1, both combination
rules, both step rules, both gates, `p` 118k–450k, and 177–1,273 commits. **It contains no `cos`, no
`N`, no rule, no α and no `p`: `‖θ_T‖` is a function of the `ρ` trajectory and nothing else.** The
sharpest case is an arm where `N` fell 200 → 40 under an annealed gate and the law still fit to 0.02%.

> **It holds only while steps are independent. MEASURED.** Server momentum correlates consecutive steps,
> so `⟨θ,Δ⟩` no longer vanishes and the orthogonality ratio of §2.1 leaves 1.000: it reads
> **1.007 / 2.981 / 6.615** at β = 0 / 0.5 / 0.75, against `(1+β)/(1−β)` = **1 / 3 / 7**. The law
> generalises with that one factor, `Φ = exp( ((1+β)/(1−β))·B )`, which fits the two momentum arms to
> 0.5% and 2%. **Any mechanism that correlates steps in time enters here, and it enters multiplicatively.**

> **Its accuracy is `ρ`-dependent** — the flat "<0.3%" once claimed is wrong. Replayed over all arms:
> **<0.3% on the low-`ρ` trust-ratio arms**, **~2% on the high-`ρ` raw-SGD arms**, **−5.8% on the
> collapsed α=0.1 arm.** §2.1's ratio drifts as `ρ` grows and the error compounds over `T`. Sum `B` over
> the steps that actually lie between the first and last norm sample — the mis-aligned window costs 1.2%
> vs 0.8% mean error.

## §4.2 The progress law — how much learning a trajectory banks

21 arms sorted by `Λ`: **peak accuracy is monotone from 0.377 to 0.876 with no exception outside
replicate noise**, flattening around `Λ` ≈ 0.95 and creeping after. Read `Λ` as **accumulated aligned
displacement in units of `‖θ_tr‖`** — travel ~0.6 of your own length in the right direction to reach
0.85, ~1.3 to reach 0.876. Calibration is P4's ledger; do not restate it here.

> **`Λ` is the *relative* coordinate and does not transfer across `p`. MEASURED.** Adapters init at a
> `p`-independent per-element scale, so **`‖θ_tr‖ ∝ √p`** (ratio constant to ±1.3%), and therefore
>
> ```
> A  =  Lambda * ||theta_tr||  =  sum_t rho_t * sqrt( G_rule * N_t ) * (||theta_tr||/sqrt(p))
> ```
>
> **`p` cancels — `A`, not `Λ`, is the coordinate accuracy tracks.** Validated out of sample: a curve
> fitted on 2,278 `rf`=16 points predicts 208 held-out `rf`=32/64 points to **0.0246 in `A` against
> 0.0623 in `Λ`**, winning on each held-out arm separately.

**Corollary — at a pinned `ρ*`, `p` does nothing.** `A` is `p`-free and `B` depends only on the `ρ`
trajectory, so shrinking `p` moves neither progress nor budget. The old `p` ladder's apparent win was
raw SGD lowering the *realised* `ρ`. `p` remains a memory, compute and comms lever; `cos ∝ 1/√p` is real
and simply buys nothing that `‖θ_tr‖` does not give straight back.

**Among arms that learned, whether an arm *holds* its peak is decided by `Φ` and nothing else.**

> **The peak sits at a fixed `Φ` across the whole portfolio. MEASURED, 7/7.** Every arm that learned
> (peak ≥ 0.80), turned, and still had >15% of its run left afterwards peaks at **`Φ` = 2.41–3.11, mean
> 2.71** — across both combination rules, both step rules, `rf` = 16/32/64, α = 0.1 and 1, `T` from 177
> to 1,364 and `ρ` from 0.06 to 0.20. Arms that *held* peaked at `Φ` = 2.82–3.54 and simply had no
> budget left to walk past it.
>
> **This is the most transferable number in the document, and it is not fitted** — `Φ` comes from `ρ`
> alone and the peak location was never used to choose anything. Read it geometrically: steps are
> orthogonal to `θ` (§2.1), so **`Φ` is the reciprocal of the un-junked fraction of `‖θ_tr‖`**.
> `Φ` = 2.7 says a head fails when it is more than ~68° off the direction that earned its accuracy —
> a statement about classifiers, not about agnews. Scored as a stopping rule in P5.2a.

> **The `Φ` ≤ 3.63 harbour is marginal, and it was fitted only on annealed arms.** The two `const` arms
> peak at `Φ` = 2.82 / 2.51 and cross −0.015 at **3.43 / 3.79** (P5.2). Read `Φ ≤ 3.6` as the edge of the
> cliff, not a safe operating band. **Pool moves the *damage* at a given `Φ`, not the `Φ` at which damage
> starts** — 4× the pool halves the cost of the budget clock without slowing it, which is §4.1's
> statement that `B` contains no `N`.

> **The "peak ≥ 0.80" qualifier is load-bearing and was missing.** Four arms sit at `Φ` = 1.00–1.01 —
> *no* inflation — and still fall 0.06–0.12 below their peak, all with `Λ` ≤ 0.068. **`Φ` governs
> *losing what you learned*; it says nothing about arms that never learned.** Independently, this shows
> accuracy loss does not require norm growth (§7.2).

**The two failure modes are separately diagnosable ~20 commits in: too little `Λ` = never learned; too
much `B` = learned and then lost it.**

## §4.3 Three consequences that restructure everything downstream

**(a) There is no critical `ρ`.** Under a pinned `ρ`, `Φ = (1+ρ*²)^{T/2}` for *any* `ρ* > 0` — geometric
always, with doubling time `1.4/ρ*²`. The apparent boundary at `ρ ≈ 0.09` was **a horizon artifact**:
the locus where doubling time falls below run length. **`ρ` is not a threshold; it is the rate at which
a fixed budget is spent.**

**(b) The `‖θ‖²` log-log slope carries no stability information under trust-ratio** — it is bounded
above by 1 *by construction*, so every "sub-linear ⇒ safe" reading on such an arm is vacuous. Under raw
SGD a slope > 1 *does* mean something: the absolute step is outgrowing `‖θ‖`, i.e. the `|JVP|` feedback
of §2.2 is live.

**(c) The closed form's *scaling* is right and its *constant* is 20× high. MEASURED.** With a clean
reference, `D = cos_measured/cos_predicted` = **0.050**, and it is invariant across a 3.57× swing in the
prediction — 0.0506 at (`select`, `p`=450k) vs 0.0501 at (`mean`, `p`=118k), agreeing to 1%. So
`cos ∝ √(G_rule·N/p)` is confirmed as a *shape* and refuted as a *magnitude*. The reason `Λ` could order
21 arms while being 20× wrong is exactly this: a constant factor cannot change an ordering. **What that
constant costs is the sizing formula** (§4.5), not the rankings.

## §4.4 The criterion

Two clocks. **The deadline:** coherent progress grows as `T·ρ·cos`, noise displacement as `√T·ρ`;
signal overtakes noise at `T ≈ 1/cos²`. **The budget:** the norm inflates by `(1+ρ²)^{T/2}`, doubling at
`T ≈ 1.4/ρ²`. You survive iff `1/cos² ≤ 1.4/ρ²`:

> ## ρ ≲ s · cos(G, g)
> **The relative step must not exceed the fraction of it that is actually aligned with the gradient.**

The O(1) constants that derivation drops are now measured. Eliminating `T` between the two laws gives
`B = Λ·ρ/(2cos)`, so fitting a target `Λ_req` inside a budget `B_max` requires
`ρ/cos ≤ 2·B_max/Λ_req` = **2.6–4.3** — not the 0.3–0.5 originally derived, a factor of ~8 that is
exactly the 5–10× seen empirically. **`s` = 2.6–4.3 is stated against the closed-form `cos`**; against
the measured one it is 20× larger, and the two must never be mixed in one expression (§6).

**The design rule.** For a fixed learning target `Λ_req` over `T` commits, Cauchy–Schwarz makes constant
`ρ = Λ_req/(T·cos)` the budget-minimising trajectory, at cost `B = Λ_req²/(2T·cos²)`:

> **Budget cost for a fixed amount of learning is ∝ `ρ`; wall-clock cost is ∝ `1/ρ`. Pick the smallest
> `ρ` whose commit count you can afford.** There is nothing to tune — `ρ` is the exchange rate between
> wall clock and safety margin, and `cos` (via `P`, `K`, `I`, `p`) sets the exchange rate itself.

Note what (a) already killed: *"walk `ρ*` up until it breaks"* is withdrawn. There is no cliff to find,
only an exchange rate to pick.

## §4.5 Sizing a pool — the formula, and the two conventions that must never be mixed

```
G_rule = E[v_par^2] for select-one-of-P  |  = P for average-all-P
cos           = D * sqrt( G_rule * N / p )        N = K*I uploads pooled server-side
N_req( rho* ) = p * (rho*/s)^2 / G_rule           <- pool needed to make rho* safe
```

`D` rescales `cos`, `Λ` and `s` **together**, so every ratio, ordering and design rule is untouched by
it. The one thing that breaks is **mixing conventions** — pairing the closed-form `cos` with the
superseded `s` = 0.4 is what made every gate A/B unrunnable, and it is a config error, not a physical
limit (P6).

**The criterion is empirically validated, and sharply. MEASURED over 17 arms.** Scoring `ρ/cos` against
the closed-form `cos`, among arms that learned: **`ρ/cos` ≤ 2.67 → all 12 end within 0.015 of peak;
≥ 3.11 → all 4 degrade.** No overlap, and the threshold falls inside the 2.6–4.3 the two laws predict,
so **`s` ≈ 2.9 is the operating constant** and `ρ ≤ s·cos` discriminates 16/16. (The `ρ/cos` = 0.03 arm
that also collapsed never learned — §4.2's qualifier, not a counterexample.)

> **What the criterion is *for*, sharpened by G-1b. MEASURED.** It sizes a pool so an arm **arrives**
> before it spends its budget. It says nothing about **staying** — and the two were conflated every time
> `s` was called a safety factor. Two arms at `ρ` = 0.06 `const` and `s` = 2.9 / 1.5 spend `B` at an
> identical rate and **both turn**, because `B` contains no `N`, no `cos` and no `s`. What the lower `s`
> buys is efficiency `Λ/B` = 2`cos`/`ρ` (P5.2).
>
> **So no gate setting is a stability mechanism.** The only quantities entering the budget are `ρ` and
> `T`, which is why §1.4 item 5 is the whole of what is left.

## §4.6 The time law — a lower bound on time-to-accuracy. **DERIVED, MEASURED to 0.4%.**

*This is the law that ranks configurations when **time to accuracy** is the objective, rather than peak
accuracy or budget. It is the third conserved statement, and it is a **bound**, not an identity.*

Per commit: `ΔΛ = ρ·cos`, `ΔB = ½ρ²`, and the wall-clock cost is `I = N/K` **serial round trips**. Write
`cos_t = c·√N_t` with `c = D·√(G_rule/p)`. Then over any trajectory, Cauchy–Schwarz on
`Λ = Σ ρ_t · c√N_t` gives `Λ² ≤ (Σρ_t²)·(c²ΣN_t)`, i.e. with `t` = total round trips:

> ## t ≥ Λ² · p / ( 2 · B · G_rule · K )
> **Round trips to reach a given `Λ` depend only on the budget you spend getting there — not on how you
> split it into `ρ`, `N` and `T`.** Equality **iff `ρ_t ∝ √N_t`**, i.e. **iff `s` is held constant.**

**MEASURED over 12 arms.** The bound is achieved to **0.4%** by both `const` arms — the only two that
held `ρ` and `N` fixed together — and missed by exactly the arms whose `s` drifted:

| arm | regime | miss vs bound |
|---|---|---|
| `145729`, `112201` | `const` ρ, `const` `N` — `s` pinned | **+0.4%, +0.4%** |
| `013917` | annealed gate: `N` falls **with** `ρ` | +1.7% |
| `200242`/`200325`/`013806`/`200358` | raw SGD, `N` pinned, `ρ` drifts slowly | +0.2% to +1.1% |
| `212009`, `222817` | `select`, `p` ladder | +3.2%, +3.6% |
| `035045` | `rm` anneal at **pinned** `N` — `s` drifts by `ρ_t` | **+8.4%** |
| `065837`, `223446` | free `var` gate — `N` swings independently of `ρ` | **+24%, +23%** |

> **So holding `s` constant is not a safety choice — it is the time-optimality condition**, and the free
> `var` gate leaves **24%** of the wall clock on the table by violating it. That is the sharpest
> justification C2 has ever had, and it is stronger than the throughput number G-1 measured.

**Four consequences, in the order they matter.**

**(a) Spend the whole budget: `t ∝ 1/B`.** Unspent budget is wasted time, and `B` is capped at `B_max`.
There is no speed-versus-stability trade *except* through `B_max` — you go as fast as the model's
tolerance for junk allows, and nothing else.

**(b) `ρ` and `N` are degenerate, so neither is a speed knob.** Given `(Λ, B)`, every split of the work
into steps and pools costs the same round trips **and the same uploaded bytes** —
`Σ N_t = K·t = Λ²p/(2·B·G_rule)`, which is `ρ`-free. Choose `ρ` for *control resolution* and `N` for
gate reachability; neither buys speed.

**(c) The speed levers are `K` and `G_rule`, and `K` is free in bytes.** `t ∝ 1/K` while total uploads
are **`K`-independent** — wider cohorts buy wall clock at zero communication cost, paying only in
staleness. `t ∝ 1/G_rule` makes `select → mean` a **3.35×** cut in time-to-accuracy (10 / 2.988), which
reframes C1: it is a *speed* result, not only a stability one.

**(d) `p` remains inert, and for the same reason as ever.** A fixed accuracy needs fixed `A`, and
`‖θ_tr‖ ∝ √p` means `Λ ∝ 1/√p` for that accuracy — so `Λ²p` is `p`-free. `p` does not enter time any
more than it enters progress (§4.2).

### §4.6a The setpoint, and why `T` is not an input

`ρ` is degenerate for cost, so it is chosen for **control resolution** — how many commits the controller
gets to make decisions over. Setting `B = B_max` in `B ≈ ½Tρ²`:

```
rho*  = sqrt( 2 * B_max / T_res )    T_res = commits of resolution wanted, NOT a budget
```

At `B_max` = ln 2.7 and `T_res` = 500 this returns **0.062**, against the 0.06 the portfolio reached by
search. **`T` was never a budget input; it is a controller hyperparameter with an obvious default.**

Two further readings, both `s`-side. `Λ_peak = 2·ln(Φ_peak)/s` — the efficiency identity `Λ/B = 2cos/ρ`
at `B = ln Φ_peak` — scores `s`=2.9 → 0.63 against **0.72** observed and `s`=1.5 → 1.22 against **1.30**.
And `T_peak = 2·ln(Φ_peak)/ρ²` contains **no `s`**, so a lower `s` buys better commits, never more of
them. The floor is gate reachability, `I = ⌈n_req/K⌉ ≤ max_iter`, which at `K` = 10 is **`s` ≥ 0.90**.

# Part 5 — What the model prescribes

## §5.1 The algorithm

**(a) On one device — the general form, nothing federated:**

```
given: theta, trainable slice theta_tr of dimension p, step schedule rho*_t, budget P
loop t:
  draw P Gaussian probes v_1..v_P            # scale h by sqrt(p) so h||v|| is p-independent
  for each i: d_i <- ( L(theta + h v_i) - L(theta - h v_i) ) / (2h)   # 2 fwd passes, no_grad
  u <- (1/P) * sum_i d_i * v_i               # ASSIMILATE ALL -- never select on |d|  (3.2)
  theta_tr <- theta_tr - rho*_t * ||theta_tr|| * u / ||u||            # TRUST-RATIO step (4.4)
  B += 0.5*ln(1 + rho_t^2);  if exp(B) > 2.5: anneal harder or stop   # BUDGET (4.1, 4.2)
  rho*_{t+1} <- rho*_0 * t^-exp              # exp = 0.25, sized to the HORIZON
```

**(b) The same loop in FL:**

```
SIZING (from the model and the runtime -- NOTHING profiled):
  p, ||theta_tr|| <- read off the model
  PHASE A: B_max  <- ln 2      # safe prior: "weights may double". NOT a fitted number (5.5f D1)
           rho*   <- sqrt( 2 * B_max / T_res ),  T_res ~ 500 = control resolution (4.6a)
           run ~150 commits -- spends B ~ 0.21, Phi ~ 1.23, negligible
  PHASE B: B_max  <- noise-injection probe, ~6 evals, forward-only        (5.5b)  [NOT BUILT]
           rho*   <- re-derived from the measured B_max
  K, P            <- hill-climb K/tau(K) and P/tau(P)                     (5.5e)  [NOT BUILT]
  N               <- from the gate, holding s constant = time-optimal     (4.6)

PER COMMIT (server):
  dispatch to C clients, wait for K uploads   # async: K arrives, stragglers roll into the next
  G <- sum_k omega_k u_k / sum_k omega_k      # omega re-weights; it must NOT set magnitude
  commit when the pool reaches N_target       # NOT a cos_hat test -- unmeasurable today (P6)
  theta_tr <- theta_tr - rho*_t * ||theta_tr|| * G / ||G||
  log rho, ||theta_tr||, top_class_share      # the three monitors (2.8)
  B += 0.5*ln(1 + rho_t^2)                    # exact, no free parameter (4.1)
  anneal rho so that B LANDS on B_max         # not Robbins-Monro: unspent budget = wasted time (4.6a)
  stop when dAcc/dLambda flattens, or B -> gamma*B_max, whichever first    [NOT BUILT]
  periodically re-sense B_max, re-climb K and P   # 5.5b, 5.5e             [NOT BUILT]
  # NO accuracy target: the run finds its own ceiling (5.5f D2)
```

**The last four lines are the entire remaining build (C-1).** Everything above them enacts to spec. They
are grouped here because they are the only places a *decision* is made from *sensed* state rather than
from arithmetic — which is the §0.0 autonomy requirement, and the reason `s`, `ρ`, `N`, `I` and `p` are
no longer decisions at all.

Flags, defaults and status for each element: [P1](fl_fwd_ft_practice.md#p1--what-is-built)
and [P2](fl_fwd_ft_practice.md#p2--the-recommended-stack).

## §5.2 The lever table — this ranks every possible fix

**`ρ/cos` is "will this survive"; `ρ·cos` is "how fast does it learn".** A good lever improves the first
without hurting the second.

| lever | `ρ/cos` (stability) | progress **per unit `B`** | who owns it |
|---|---|---|---|
| **P-averaging** (trainer pooling) | **∝ 1/P** | **∝ √P — free in wall clock** | trainer |
| **K** (cohort width) | **∝ 1/K** | **∝ √K** — parallel across devices | client selection |
| **I** (iterations per bin) | **∝ 1/I** | **∝ √I** — **but serial: one round trip each** | aggregation gate |
| **p** (trainable dimension) | ∝ √p | **invariant — `‖θ_tr‖ ∝ √p` cancels it** (§4.2) | model/PEFT design |
| `η` learning rate | ∝ η | **invariant — pays 1:1** | aggregation |
| server momentum `β` | ∝ `√((1−β)/(1+β))` | **invariant — `√x` progress for `x` budget** | aggregation |
| probe-selection gain `E[v∥²]` | **invariant** (`b²/a = 1`) | invariant | trainer probe selection |
| step normalization | sets `ρ` to an operator constant | decoupled | aggregation |

**Read the second column, not the first.** Four rows improve `ρ/cos` and *do not* improve progress per
unit budget: `p`, `η`, `β` and probe selection. Only `P`, `K` and `I` — the three stages that pool
**independent readings of independent data** — are free, and they all buy `√n`. **A lever is free if and
only if it raises `cos` without correlating steps and without shrinking `‖θ_tr‖`.** That single sentence
now replaces four separately-discovered dead ends.

### §5.2a Every pooling stage is the same currency. **DERIVED from §6.**

Fold `D` into the estimator. With `B` the bin size, `σ` the per-sample gradient spread and `g*` the
population gradient, `rms‖g_bin‖ ≈ σ/√B`, so `D = ‖g*‖√B/σ` and — for `mean`, where `b = 1/√P` —

```
cos     =  (||g*||/sigma) * sqrt( P * B * K * I / p )
compute =  2 * P * B * K * I     forward passes over one sample, per commit
      =>  cos  =  (||g*||/sigma) * sqrt( compute / (2p) )
```

> **`cos ∝ √compute`, with the same constant for all four stages.** `P`, `B`, `K` and `I` are not four
> levers; they are four spellings of one. This is why every sweep came back with the same exchange rate,
> and why raising any one alone was never enough.

**Choose the spelling by latency and bytes, since the FLOPs are fixed** — `P` and bin size cost no round
trips and no bytes; `K` is parallel but brings staleness and costs `∝ K` in bytes per commit; `I` costs
**one serial round trip each**. §4.6 turns this into the time law and §5.5e into the two hill-climbs.

**Only one thing escapes the `√compute` law, and it is already spent.** `b²/a` is a property of the
arithmetic rather than the budget, and `select → mean` bought 10× of it at zero compute; §3.2 says there
is no second one. Both apparent escapes are dead: `D` does not move with bin size or bin count (§6.3),
and `‖g*‖/σ` is a property of the task and partition, not a knob.

> **There is no lever left to find.** Everything is `√compute`, spent wherever it is cheapest in latency
> and bytes — which makes the **controller**, not the estimator, the remaining engineering.

## §5.3 The ratio principle

Every quantity compared against a fixed constant must be **scale-invariant**, because anything carrying
units silently changes meaning as training proceeds. The test: *re-parameterise so `‖θ‖` doubles; the
loss surface shape is unchanged, so the trajectory should be.* Any rule that fails needs re-tuning every
time the model, adapter rank or round index changes. Three replacements follow:

| unit-carrying | scale-free replacement |
|---|---|
| absolute step `η` | relative step `ρ*` |
| absolute variance threshold | an `N` target |
| absolute iteration cap | a measured adequacy condition |

**Why the criterion does not have to enumerate its failure modes.** Anything not thought of enters
through exactly **two** channels: it changes the step taken → shows up in **`ρ`**, logged exactly; or it
degrades the pool → shows up as **`n_eff < n`**, hence in `cos`. **Channel 2 is the unpaid half.**
`n_eff` detects *correlation* among uploads and returns 1.00 ± 0.01 in every arm ever replayed, so it
carries no information; it does **not** detect *directional* disagreement, because at `p = 4.5e5`
differing `g_k` move `var` by `O(n/p)` and are invisible. **Scale-free setpoints therefore have to come
from the step rule, not from channel 2.**

## §5.4 Priority order — and why it is a hill-climb, not a list

**The time law replaces the old ranked list.** `t ≥ Λ²·p/(2·B·G_rule·K)` says only three things move
time-to-accuracy, and each has a *sensed* objective rather than a fixed rank:

| lever | effect on `t` | bytes | how the controller decides it |
|---|---|---|---|
| **`B`** (budget spent) | `∝ 1/B` | `∝ 1/B` | spend to `B_max`; `B_max` is sensed (§5.5b) |
| **`K`** (cohort width) | `∝ 1/K` | **free — uploads are `K`-independent** | hill-climb `K/τ(K)`, both measured at runtime (§5.5e) |
| **`G_rule`** (= `P` under `mean`) | `∝ 1/G_rule` | `∝ 1/G_rule` | **sensed** — hill-climb `P/τ(P)` (§5.5e, D4); cancels only if client compute dominates `τ` |
| `ρ`, `N`, `I`, `T` | **none — degenerate** | **none** | pick `ρ` for control resolution, `N` for gate reachability |
| `p` | **none** — `Λ ∝ 1/√p` at fixed accuracy cancels the `Λ²p` | `∝ p` | memory and comms budget alone |

**Two things the old list got wrong.** `I` was ranked "last, and only what the gate demands" — but `I`
and `K` are the *same* quantity `N` split two ways, and §4.6 shows the split is free. And "`K` first
because it pools independent data, which is where `D` lives" was a D-1 conjecture that was **refuted**:
`D` is flat over 64× in bin size (§6.3). `K` still leads, for a completely different and now-measured
reason — it is the only lever that buys wall clock at zero communication cost.

**The one unmeasured cost in this table is staleness.** `K` is free in bytes and free in the time law,
because the law assumes every pooled upload reads the *current* `θ`. Stale uploads read an older one, so
the cost should appear as a fall in `D` with `K`. `D(K)` has never been measured, and it is what turns
the `K` hill-climb from a throughput heuristic into an optimum: **`t ∝ τ(K)/(D(K)²·K)`**.

## §5.5 Deploying without a profiling run

**The objection this section answers.** Every constant in Parts 1–7 was obtained by profiling — 30 arms
on one model and one task. A real deployment gets no such budget, so *"it works after we sweep it"* is
not a method. The question is how many independent numbers a new setting actually has to discover, and
how each one can be got.

**The knob count collapses, and most of it is already banked.** Each row below is a knob this program
started with and no longer has to set:

| knob | fate | how it is set now |
|---|---|---|
| `η` learning rate | **eliminated** | subsumed by `ρ*`; the trust-ratio step enacts `ρ = ρ*` to 8.7e-5 |
| α heterogeneity | **neutralised** | the step rule removes it — α moves `ρ` by **0 to 6 s.f.** (§2.6d) |
| `p` / PEFT rank | **inert for learning** | memory and compute budget alone (§4.2) |
| probe selection rule | **eliminated** | `b²/a = 1` — always assimilate all `P` (§3.2) |
| server momentum `β` | **eliminated** | refuted: `√x` progress for `x` budget (§5.2) |
| `var_threshold` | **replaced** | by a scale-free `N` target (§2.3) |
| `P`, `K`, `I`, bin size | **one lever, not four** | `cos ∝ √compute` with one constant (§5.2a) — choose the *spelling* by latency and bytes, never by tuning |
| `ρ*` | **derived** | `√(2·B_max/T)` — §4.4 |
| `s`, hence `n_req` | **derived from a latency choice** | `n_req = p(ρ*/s)²/G_rule`; `s` = 1.5 is near its floor (§4.4c) |
| anneal exponent | **replaceable** | a horizon-sized `exp` becomes a budget-feedback law once `B` is tracked (C-1) |

> **What is left is one constant.** **`B_max`** — and it is sensed online, not supplied (§5.5b).
> `T` and `s` are *not* budget inputs: §4.6 shows time and bytes are `ρ`-degenerate, so `T` is only a
> control-resolution choice and `s` follows from gate reachability. Everything else is arithmetic.

### §5.5-0 What the objective implies for the controller

§0.0 fixes the metric order and §4.6 puts all three in one expression, so they need no hand-trading:
everything that makes a run faster also makes it cheaper in bytes, `K` is free in bytes outright, and
stability is a **ceiling** rather than a trade. **The single decision is how close to `B_max` to run.**

**What follows for the controller, and it changes the shipped recommendation:**

- **Hold `s` constant.** It is the Cauchy–Schwarz equality condition (§4.6); the free `var` gate gives up
  **24%** of wall clock, and `rm` at a pinned `N` gives up 8.4%. Under an anneal this means `N` must fall
  **with** `ρ`, which is `gate_rho_ref=annealed` — the setting P3 recorded as "progress-per-wall-clock"
  against `setpoint`'s "progress-per-commit". **With time-to-accuracy as the objective, `annealed` is the
  right one**, and `013917` is the arm that achieves the bound to 1.7%.
- **Anneal to *land* on `B_max`, never to stay under it.** Robbins–Monro converges to some `B_∞`, and
  `t ∝ 1/B_∞`, so every unit of unspent budget is wall clock given away.
- **Stop on saturation, not on a schedule.** Track `dAcc/dΛ` over a window — an aggregate over 100+
  commits, which is the only regime where this program's statistics are readable (§5.3). Stop when it
  flattens, or when `B → γ·B_max`, whichever comes first.
- **The `const` arms did not need an anneal; they needed a stop.** `145729` peaked 0.876 at commit 514
  and then ran 431 further commits into the ground. A `Φ` trigger would have kept the portfolio's best
  accuracy, at the portfolio's best time-to-accuracy, with no schedule at all.

### §5.5-1 The sensing audit — every quantity, and where it comes from

**The autonomy test of §0.0 applied to the whole loop.** A quantity is legitimate only if it is (a)
exact arithmetic on what the run logs, (b) a runtime hill-climb, or (c) a deployment property the
operator genuinely owns.

| quantity | class | how it is obtained | status |
|---|---|---|---|
| `ρ` | (a) | it *is* the knob — trust-ratio enacts `ρ = ρ*` to 8.7e-5 | **built** |
| `B`, `Φ` | (a) | `½Σln(1+ρ²)` — exact, no free parameter, readable in ~20 commits | **built** |
| `Λ` | (a) | `Σρ·√(G_rule·N/p)` — closed form, exact | **built** |
| `p`, `‖θ_tr‖` | (a) | read off the model at init (`[ProbeDim]`) | **built** |
| `τ` round-trip time | (a) | the aggregator already times every round trip | **built, unused** |
| `N`, `I` | (a) | the gate's own closed form, given `s` | **built** |
| `ρ*` | (a) | `√(2·B_max/T_res)`, `T_res` = control resolution ≈ 500 (§4.6a) | **not wired** |
| `s` | (a) | held constant — the time-optimality condition (§4.6) | **built** (`n_target` + `annealed`) |
| **`B_max`** | **(b)** | **noise-injection probe, ~6 evals, forward-only, periodic** (§5.5b) | **exists offline only** |
| **`K`, `C`** | **(b)** | **hill-climb `K/τ(K)` against availability** (§5.5e) | **not built** |
| `dAcc/dΛ` | (b) | eval slope over a 100+ commit window — the stopping signal | **not built** |
| `D` | (b) | `cos` audit on a stride. **Forecast-only** — never on the operating path (§5.5a) | **built** |
| **`P`, bin size** | **(b)** | **hill-climb `P/τ(P)`** — same rule shape as `K` (§5.5e) | **not built** |
| `p` / PEFT rank | (c) | device memory budget. Inert for learning and for time (§4.6d) | operator |
| model, PEFT scheme | (c) | the deployment. α is **neutralised**, not sensed (§2.6d) | operator |

> **Four gaps, and they are the entire remaining build**: `B_max` is not sensed online, neither `K` nor
> `P` is adaptive, and there is no saturation-based stop. All four are C-1. **Nothing else in the loop
> needs a number an operator has to know.**

### §5.5a The operation / prediction split — where `D` actually bites

`D` is the one badly-behaved constant (D-2), and the reason it is survivable is that **it never appears
on the operating path**:

| you want to… | needs | blocked on `D`? |
|---|---|---|
| set `ρ*` | `B_max`, `T` | **no** — §4.4 |
| set `N`/`s`, and enact them | the closed form, exact | **no** |
| know how much budget you have spent | `ρ` — exact, parameter-free | **no** |
| know *when to stop* | `B_max` | **no** |
| **predict the accuracy you will reach** | `cos`, hence `D`, plus the `Λ` curve | **yes** |

**So `D` and the dose-response curve are needed to *forecast* a run, never to *run* one.** A deployment
that cannot profile can still be operated safely and optimally; what it loses is the ability to promise
a number in advance. That is the right thing to lose.

### §5.5b Sensing `B_max` — one shot, forward passes only

The remaining constant needs no trajectory, because **inflation can be injected instead of waited for**
(the method that settled §7). Take the current model, add isotropic Gaussian noise to the trainable
slice scaled so `‖θ_tr‖` grows by `Φ`, read accuracy back at `Φ` ∈ {1.5 … 4}. The knee is `Φ_peak`; set
`B_max = ln Φ_peak`.

- **~6 evals**, on a copy of the model — no training, no gradients, no backward pass, i.e. the same
  operator set the method already restricts itself to.
- **Already built** as `scripts/probe_inflation_damage.py`; it has never been run *inside* a live run.
- **Runs on the model being trained**, so it calibrates the deployment rather than a proxy.
- **Biased conservative** — it noises a model that cannot re-fit, reading the knee ~0.6–1.2 low
  (`Φ` ≈ 3.0 on the rig against 3.6–4.2 on arms, §7.1). It under-spends budget, never over-spends.

**Why `B_max` is the right constant to bet on.** Three independent lines: *(1)* it is **geometric, not
task-shaped** — `Φ` is the reciprocal of the un-junked fraction of the weight vector, so `Φ_peak` ≈ 2.7
says a head fails past ~68° off the direction that earned its accuracy; *(2)* it **already transferred**
across the one axis tested — two models differing 1.77× in norm share a knee at the same `Φ` (§7.1);
*(3)* it **held over the whole portfolio unfitted** — 7/7 turned arms at `Φ` = 2.41–3.11 (§4.2), and
scored as a stopping rule it gives up 0.005 of peak on average (P5.2a). Falsified by a model family
where `‖θ_tr‖ ∝ √p` fails, or a head that is not a linear readout — both checkable in one probe.

**What this does not solve.** `B_max`'s transfer has three arguments and **zero cross-model
measurements** (B-1). The probe's conservative bias is uncalibrated. The controller does not exist
(C-1). And forecasting a new deployment's final accuracy remains impossible — §5.5a is the line.

### §5.5e Adaptive `K` — the one lever with an availability coupling

`K` is the top time lever and the only one whose cost is not in the time law. The controller's objective
falls straight out of §4.6, since wall time is round trips × round-trip duration:

```
minimise   t_wall  ~  tau(K) / ( D(K)^2 * K )        <- both factors MEASURED at runtime
```

- **`τ(K)` is measured directly.** It rises with `K` because the server waits for `K` of `C`: P3 measured
  8.9 → 12.3 s for `K` = 30 → 50, so `K/τ(K)` improves **sublinearly**, roughly `K^0.7`.
- **`D(K)` is the staleness price and has never been measured.** Staleness is genuine at `K` ≥ 30 (max 2
  at 30, 4 at 50; `pastdated_commits` max 34 at 50), so there is finally something to weigh.
- **Availability sets the ceiling**, not a config: `K ≤ C ≤` trainers actually available. Raise `K` while
  `K/τ(K)` improves *and* clients are there to fill it; back off when either fails.

**Uploads are `K`-independent** (`= K·t`, and `t ∝ 1/K`), so this hill-climb spends no communication
budget — it trades staleness for wall clock and nothing else. That makes `K` the **first** thing the
controller should move and the **last** thing it should be conservative about, which is the opposite of
how `dynamic_kc` was configured (`k_max` = 15, disabled).

**`P` takes the identical rule, and it is the one that also cuts bytes.** `G_rule = P` under `mean`, so
`t ∝ 1/P` *and* uploads `∝ 1/P`, at **invariant total client compute** (§5.2a). The only thing that can
cancel it is the round trip becoming compute-bound:

```
minimise   t_wall  ~  tau(P) / P          <- hill-climb, identical shape to K
                                             cancels exactly if tau ~ P
```

> **`K` and `P` are the same rule applied to the two stages that cost nothing to widen.** `K` is free in
> bytes and paid in staleness; `P` is free in compute and paid in latency *only if* the client is
> compute-bound. Both are measured, not configured — which is D4.

**One correctness consequence of a moving `P`.** `G_rule` becomes time-varying, so every closed form
must use the *current* value: `Λ = Σ ρ_t·√(G_rule_t·N_t/p)`, and the time law's pooling variable is
`Σ G_rule_t·N_t` — total **probe readings**, not uploads. `replay_scoring.py` takes one `G_rule` per arm
from the config and will mis-score any run whose `P` moves.

### §5.5f The four controller decisions, settled 2026-08-12

*These were open design questions; they are now fixed and the build follows them.*

| # | question | **decision** | why |
|---|---|---|---|
| **D1** | `B_max` needs a partly-trained model to measure, but `ρ*` needs `B_max` to start | **Two-phase from a safe prior.** Phase A runs at `B_max` = **ln 2** — *"the model survives its weights doubling"* — giving `ρ*` = 0.053 at `T_res` = 500. After ~150 commits that has spent `B` ≈ 0.21 (`Φ` = 1.23). Phase B injection-probes, re-derives `ρ*`, runs to the stop | ln 2 is the **weakest non-trivial claim**, not a fitted number, and Phase A's spend is negligible against *any* plausible `B_max` — so a wrong prior costs almost nothing and is corrected within 150 commits |
| **D2** | what is "the target accuracy"? | **No target.** Stop when `dAcc/dΛ` flattens over a 100+ commit window, or `B → γ·B_max`, whichever fires first. Report the plateau found | a supplied target is itself an operator input, and §0.0 admits none. The run finds its own ceiling |
| **D3** | is the communication budget a constraint? | **A tiebreak.** Minimise bytes *subject to* time-first | uploads `= Λ²p/(2·B·G_rule)`, so spending `B` and raising `P` cut time and bytes **together**, and `K` is free in bytes outright. Most of the comms win falls out of optimising time |
| **D4** | which knobs may the operator still supply? | **Model, PEFT scheme and `p` only.** `P` is **sensed** by hill-climbing `P/τ(P)` (§5.5e) | the strongest autonomy claim the evidence supports. `p` stays operator-owned because it is measurably **inert** (§4.2, §4.6d) — sensing it would buy optics, not performance |

> **What D4 costs.** It promotes **P-1** from an ablation to a **build prerequisite** — the controller
> cannot hill-climb `P` without knowing `τ(P)` — and it requires the trainer to accept a **mid-run `P`
> change**, which nothing does today.

## §5.6 What is settled, and what is not

**Settled — do not re-measure.** The step rule, the anneal, the combination rule and the `p` lever all
enact to ≤4e-3 with mechanisms measured. On top of them, §4 closes the *dynamics*: `‖θ_T‖` is a
closed-form function of the `ρ` trajectory alone, and peak accuracy is a monotone function of `Λ`.

**The remaining unknown is not about the optimizer, and it is not about its constants either.** It is
that every reading is taken on an 8-sample bin, so the quantity the estimator is unbiased *for* is not
the quantity we want to descend. That costs `D` ≈ 0.05 in `cos` and 400× in `N_req` (§6).

**Two structural gaps:** the controller — §5.1 still stops on a schedule chosen offline, though `Φ` now
gives it a sensor (§1.4 item 5, C-1); and the `h`/`p` coupling, unanalysed beyond the FD-rescale
interaction and owned by nobody.

**The cohort-width requirement is withdrawn, and now empirically.** It was an artifact of
`gate_safety_s` = 0.4. At the measured `s` ≈ 2.9, `ρ*` = 0.06 with `mean` needs `N` = 19 — `I` = 2 at
`K` = 10, which G-1 then ran: the gate enacted `n_req` = 19.3 off the cap on 100% of commits and netted
**2.13×** the control's progress per unit federated time ([P5.2](fl_fwd_ft_practice.md#p52-scored-nodes--g-1b-g-1-q3-s1-h-p)).
Every cohort this harness runs is already wide enough.

---

# Part 6 — `cos`, now measured: the formula is the right shape and 20× too big

*This section carried the program's largest unknown for months. It is closed. The answer is not the one
the model bet on, and the correction is not in the optimizer at all.*

**What was built.** A fp32 backward pass on a fixed held-out batch, server-side, at `θ_t` before the
step, emitting `cos_ground_truth` per commit. Index alignment was unit-tested (pool = ±`g` → cos = ±1).

**What was wrong with it.** The probe sliced the first `n` rows of the raw test tensor, bypassing the
DataLoader's sampler. The test index list is built by iterating clients in partition order and extending
with each client's shard, **with no shuffle** — so under a Dirichlet partition the probe's 64-sample
"global reference" was **one client's shard**, 48/4/5/7 across four classes against a test set that is
exactly balanced.

**Everything it reported is void — including the number that was thought to have survived it.** The
claim that `‖G‖/‖g_probe‖` matched `b·√(p/N)` to 5–10% was itself an artifact: a 64-sample reference
gradient has norm 3.1–8.0, which is roughly the norm of an 8-sample *client* gradient, so the ratio came
out at 1.0 for the wrong reason. Replayed against a clean 1024-sample reference (norm 0.27–0.34), the
same three arms read 0.95 / 1.11 / 1.01 where they should read ~9. **Both halves of §3.2 were
mismeasured, not one** — and the half that "survived" was the one nobody re-checked, because a
direction-free ratio felt safe.

Measured offline at a backprop-trained model, `cos(g_batch, g_full_test)`:

| `n` | **first `n` (what shipped)** | random `n` (the fix) |
|---|---|---|
| 64 | **−0.457** | 0.480 |
| 256 | +0.493 | 0.794 |
| 1024 | +0.613 | 0.935 |
| 2048 | +0.628 | **0.975** |

**The shipped reference was anti-correlated with held-out truth** (−0.46 trained, −0.42 at init, +0.69
at accuracy 0.38): **its sign swings with model state.** That one fact reproduces everything the probe
appeared to show — a near-zero pooled magnitude, an 11× drift at fixed `N`/`p`/rule, and an
anti-ordering of the arms — with none of it being real.

**Fixed and re-run.** Fixed-seed shuffled batch over the whole test set, default size 1024, plus a
preflight that fails the launch on a class-skewed reference. Six arms carried it on 2026-08-10; the
logged dominant-class share was **0.27–0.29** against a balanced 0.25 on every one.

## §6.1 The result: the shape is right, the constant is not

`D ≡ cos_measured / cos_predicted` = **0.050**. The formula is 20× optimistic. But:

> **`D` is invariant to the combination rule, to `p`, and to `N`. MEASURED.** 0.0506 at (`select`,
> `p` = 450k) versus 0.0501 at (`mean`, `p` = 118k) — **1% apart across a 3.57× swing in the
> prediction** — and 0.0506 at `N` = 20 versus 0.0454 at `N` = 80, **10% apart across a 10× swing in
> `N`** (20 / 80 / 200) from the two G-1 arms. Pooling more uploads does not buy any of it back, which
> is the same negative result §6.3 reports for bin size and bin count.

So `cos ∝ √(G_rule·N/p)` is **confirmed as a scaling law and refuted as a magnitude**, which is exactly
why `Λ` could order 21 arms while being 20× wrong: a constant factor cannot reorder anything. The
estimator-shape table of §3.2 is likewise confirmed on both halves — `L` and `S` below are each
invariant across a 5.5× swing in `b` and a 3× swing in `a`.

## §6.2 Where the factor of 20 comes from, and it is not the probes

Split the shortfall into the two things §3.2 predicts separately. Both are measured per arm:

```
L = (||G|| / ||g||) / ( b * sqrt(p/N) )      length excess   -- measured 8.5 .. 9.5
S = cos * (||G|| / ||g||) / a                shadow deficit  -- measured 0.48
D = S / L                                                    -- measured 0.050
```

**`L` ≈ 9 is a data-sampling fact, not an estimator fact.** Because `v` is a raw Gaussian draw,
`d = ⟨g_k,v⟩` has `rms|d| = ‖g_k‖` *identically* — so `rms|d|` measures the norm of the client's
**8-sample** gradient. Measured: `rms|d|` = 3.2–3.4 against `‖g_heldout(1024)‖` = 0.27–0.34, a ratio of
**10.0–10.9**, which is `L` within its scatter. **Nine tenths of every upload's length is data noise the
pooling was never sized for**, because `‖G‖` normalises by a quantity ~10× larger than the gradient we
mean to descend.

**`S` ≈ 0.48 is the other half:** the pooled `K` = 10 × 8-sample gradient carries only half of the
held-out gradient's shadow. Zero-mean sampling noise alone predicts `S` = 1, so the missing half is
*bias* — Dirichlet-skewed client draws, and train-vs-held-out gradient disagreement, which this probe
cannot separate.

> **The correction is entirely on the data side.** Not the probe direction, not the combination rule,
> not `p`, not the step. `cos = D·(a/b)·√(n/p)` with `D` ≈ 0.05, and **`D` is the price of the estimator
> being unbiased for the wrong gradient.** §6's old caveat named this exactly — "*`cos = (a/b)√(n/p)`
> assumes the `n` readings are independent and share one target `g`; they do not*" — and it turns out to
> be worth a factor of 20, not a footnote.

## §6.3 `D` is a constant of the setting. **MEASURED, and it is a negative result.**

D-1 swept bin size `B` ∈ {8,32,128,512} × distinct bins `M` ∈ {1,10,30} on the real model and the real
Dirichlet partition, against both references, 10 reps.

> **`L` is flat: 9.47–10.85 over 64× in `B` and 30× in `M`**, where `L ∝ 1/√B` predicts an **8× fall.**
> `S` is flat in `M` too. **Bigger bins and more bins do not recover `D`.**

So `D` is not a pooling deficiency and there is nothing to pool away. **That is the benign outcome**: a
constant is absorbed into `s`, which is exactly why `ρ ≤ 2.9·cos` works (§4.5). The `1/D²` in `N_req`
never had to be paid — it was double-counting a factor `s` already carries.

> **The invariance is established *at one training state*, and that qualifier was invisible until now.**
> This sweep ran on the rig at init, and every arm-side `D` behind it came from a run cut inside ~75
> commits. The two arms that trained to peak read `D` = **0.148 ± 0.020** and **0.104 ± 0.017** against
> their 0.045–0.051. Bin size and bin count are still refuted as levers; what is open is whether the
> *constant* they are flat against is a constant of the **setting** or of the **checkpoint**. That is
> D-2 (§0).
>
> **The second arm rules out the obvious answer.** Pooled over both, `D` is **non-monotone in accuracy**
> and its *highest*-accuracy bin reads its *lowest* `D` (0.092 ± 0.022), while the pre-turn/post-turn
> split reads 0.103 ± 0.021 → 0.163 ± 0.025. So the elevation tracks `Φ` — the model coming apart — and
> not "having learned". The near-init windows of the two `const` arms (0.079 / 0.091, ~1σ above the `rm`
> arms' 0.045–0.051) leave **`const` vs `rm` as the leading confound**. It will not be settled on an arm:
> single-commit `cos` has SNR ≈ 1, with 16–18% of fires returning a *negative* cosine.

**The rig half-passes its gate, and the residual is the last open item.** It reproduces `L` (10.2 against
the arms' 8.5–9.5) but returns `S` = 1.68 where the arms read 0.48 — and its own `S` is arithmetically
clean (`‖g_train‖/‖g_test‖` = 2.01 × `cos(g_train,g_test)` = 0.80 → 1.61). **So a 3.5× shadow loss lives
in the FL pipeline, not in the data.** Prime suspect is the FD chord: `h‖v‖` = 6.71 against
`‖θ_tr‖` = 6.75, so each probe steps roughly a full parameter-norm and `d` is a chord-averaged slope
rather than `⟨g,v⟩`. That attenuates alignment while leaving `d` Gaussian — which is why the
distributional check in [P3](fl_fwd_ft_practice.md#p3--knob-ledger--what-each-lever-actually-did) did not
catch it, and why §1.3's ruling-out of the FD (as a cause of *divergence*) does not apply. This is H-S.

**Isotropy of `v` is exact by construction and was never the caveat.** `n_eff` excludes the
*correlation* half of the pooling assumption; the *homogeneity* half is what broke, and it broke at 20×.

**The methodological lesson, twice over:** the probe's arithmetic was unit-tested and correct, and
nothing tested that its *input* was what we thought — one `Counter(labels[:64])` would have caught it.
Then the correction itself was under-applied: one number was allowed to survive the defect on a
plausibility argument ("a ratio of norms cannot depend on the reference's direction" — true, and
irrelevant, because it depends on the reference's *norm*). **When an instrument's input is wrong,
every number it produced is void, including the ones that still look reasonable.**

---

# Part 7 — What inflation actually does

Both of the old blockers reduce to one question: what does growing `‖θ_tr‖` do to a *trained* model?
The inflation is a sum of steps in random directions, so it can be **injected** rather than waited for:
train to a realistic peak, add Gaussian noise to the trainable slice scaled so `‖θ_tr‖` grows by `Φ`,
read accuracy back. **MEASURED on the offline rig, 2026-08-10.**

## §7.1 The budget is relative, not absolute (Q1 — settled)

The accuracy knee sits at the **same `Φ`** in two models whose norms differ 1.77× (`rf`=16 vs `rf`=64):
both hold near 0.9 up to `Φ` ≈ 2.5, break at 3.0, and are destroyed by 3.6. Absolute norms at the knee
differ by exactly the ratio of the base norms. **`B_max` transfers across `p`**, as the ratio principle
predicted.

> The rig's own threshold (`Φ`≈3.0) is lower than the real arms' 3.6–4.2, because it dumps noise on a
> model that never adapted while real training re-fits continuously. That bias applies to both `p`
> equally, so it cancels in the comparison — **take the *relative* verdict from the rig and the
> *number* from the arm ledger: `B_max` ≈ 3.6–4.2 in `Φ`.**

## §7.2 The norm is mostly a symptom, and decay will not rescue it (Q2 — settled)

Weight decay scales the whole vector, preserving direction. Renormalizing an inflated model back to its
original norm is exactly what perfect decay leaves behind:

| `Φ` | noise (norm grows) | noise + renorm (norm pinned) | recovered |
|---|---|---|---|
| 2.5 | 0.856 | 0.803 | — |
| 3.0 | 0.608 | 0.685 | +0.08 |
| 3.6 | 0.262 | 0.403 | +0.14 |

**~20–25% of the damage is magnitude; ~75–80% is the junk:signal ratio, which decay cannot touch.** The
mechanism is visible in the renormalized rows: `logit_norm` falls and entropy rises to ≈ `ln 4` — the
signal has been diluted below the noise and the model predicts nearly uniformly. This confirms the
pre-registered prediction and demotes the weight-decay arm to an optional confirmation.

**Two controls separate the mechanisms, and only one matches reality.** Isotropic *noise* reproduces the
observed collapse signature (`top_class_share`→1.0 at unremarkable `logit_norm`, §2.7); pure *scale*
growth does not (`logit_norm` explodes, entropy → 0), and *coherent* displacement destroys accuracy at
`Φ` = 1.2 already. **Isotropic junk accumulation is the right model of the real mechanism.**

## §7.3 The cost of going backward-free

Backprop on the same model, `p` and data reaches **0.90** where the best forward-gradient arm ever
recorded is **0.876** (`145729`). That **2.4-point** gap is the measured price at this scale — down from
3.5 points, bought by pooling 4× harder per commit and paid for in a peak the arm then loses.

*Caveat on all of Part 7:* injection jumps to the endpoint instead of walking there, and omits the
learning happening alongside. It settles the comparisons; it does not replace a trajectory for
calibration.

## §7.4 What is standard, what is ours, and what to claim

**None of the ingredients are ours:** `1/√p` probe overlap (classical ZO); signal-linear / noise-`√T`
(SGD noise ball); `Σρ = ∞`, `Σρ² < ∞` (Robbins–Monro); step relative to `‖θ‖` (trust-region, LARS/LAMB);
a pool beyond which pooling buys nothing (critical batch size). **Ours is the packaging** — collapsing
them into an inequality between quantities the server already logs, turning an asymptotic rate statement
into an online control law. Do not write it up as a new theorem.

| claim | verdict |
|---|---|
| **C1 · `\|JVP\|` probe selection** | **Refuted as written.** `b²/a` = 1 — selection is stability-neutral by construction. The **combination rule** is the contribution, and §4.6 makes it a *speed* result: `t ∝ 1/G_rule`, so `select → mean` is **3.35× less wall clock** at zero extra cost |
| **C2 · async aggregation / gate** | **A measured win, and stronger than throughput.** §4.6 shows holding `s` constant is the **time-optimality condition** — the free `var` gate gives up 24% of wall clock. G-1 priced the trade exactly (P5.2). What C2 is *not* is a stability mechanism: `B` contains no `s`, so no gate setting prevents collapse |
| **C3 · aggregation weighting (ω)** | **Magnitude half: park** (ω spans 0.70–0.87 against a ≥10× gap, and trust-ratio removes it from magnitude). **Freshness half: now on the critical path** — staleness is the only cost of `K`, the top time lever, and `D(K)` has never been measured |
| **S1 · server momentum** | **Refuted.** `√x` progress for `x` budget — the same exchange rate as raising `ρ`, i.e. a re-parameterisation. Do not claim it |
| **S2 · variance-gate recalibration** | **Superseded** — replace the loop, do not re-tune the threshold |
| **S3 · aggregation-rate tempering** | **Subsumed** — under trust-ratio `ρ` = `ρ*_t` regardless of ω |
| **Systems: flat memory, inference-only operators** | **Untouched, and the strongest part of the paper.** The `cos` probe's backward pass is an audit flag, off by default |

**Net effect: C1 shrinks** to a compute-budget claim that only averaging redeems, **C2 grows** and
relocates from "async" to "the controller async makes possible", **C3 splits** into a parked half and a
now-load-bearing half. Systems contributions are unaffected.

**Four things the work added that were not on the claim list:**

1. **A free-lever test subsuming four dead ends.** A lever is free iff it raises `cos` without
   correlating steps and without shrinking `‖θ_tr‖`. `p`, `η`, `β` and probe selection all fail; only
   `P`, `K`, `I` pass.
2. **Scale invariance as a design principle** (§5.3) — cleanest demonstration: under trust-ratio,
   heterogeneity moves `ρ` by *zero to six significant figures* while still moving `var` by 1.6×.
   And its configuration-level twin, the **autonomy requirement** (§0.0).
3. **The time law** (§4.6): `t ≥ Λ²p/(2·B·G_rule·K)`, which makes time-to-accuracy depend only on budget
   spent and turns the commit gate into an optimality mechanism.
4. **`D`** — for forward-gradient FL the binding constraint is the gradient of the *bin*, not the
   variance of the *probe*. Unplanned, and it inverts the "spend compute on probes" instinct.

---

# Part 8 — Generality beyond FL

*Speculative relative to Parts 1–7, and **frozen** (R8): revisit only when the open questions close.*

**Nothing in the model is federated.** `cos = D·√(G_rule·n/p)`, `ρ ≤ s·cos`, `B` and `Λ` apply to any
optimizer estimating gradients from directional derivatives. FL enters only in **how `n` decomposes and
what each factor costs**: single-device and centralized ZO are `P` × accumulation; data-parallel adds
workers; **federated is `P` × `K` × `I`**, where `I` costs round trips and `K` costs staleness. `D` is
not federated either — any setting taking directional derivatives on minibatches pays it; only
full-batch ZO has `D` = 1.

**A falsifiable prediction against centralized ZO.** It uses `P` = 1 and a very small fixed LR. With
`G_rule` = 1 and no `K` or `I`, `cos` = `√(1/p)`, so the stable relative step is ~10⁻³ and the method
needs ~`p` steps for coherent progress — matching the very long step counts such methods report. Check
published curves before claiming it.

**What must be checked before generalizing.** Each row is an assumption, not a result:

| # | assumption | why it might not survive |
|---|---|---|
| 1 | **steps orthogonal to `θ`** (§2.1), which makes `B` exact | measured only at `p` ≥ 118k on adapter+head slices. **Check the orthogonality ratio first on any new model — one line of replay** |
| 2 | **`‖θ_tr‖ ∝ √p`**, which makes `p` inert (§4.2, §4.6d) | holds for adapter-style init. A PEFT scheme initialising otherwise makes `p` a live lever again — **the assumption most likely to break** |
| 3 | **`B_max`** | settled as *relative* (§7.1), value calibrated on one model family. This is B-1, and it is the **only** constant on the operating path (§5.5) |
| 4 | `Λ`'s calibration and **`D` ≈ 0.05–0.15** | fitted here — but both are **forecast-only** (§5.5a), so neither blocks operation |
| 5 | **isotropic probes / `1/√p` overlap** | exact given Gaussian `v` over the whole slice. Structured or block probes break it deliberately |
| 6 | **`h` pinned by fp16 cancellation** | a precision property, not a math one; bf16/fp32 move the usable window |

**Larger models and datacenter.** On-device, forward-gradient wins because it stores no activations. In
a datacenter with backprop available the argument narrows to "2 forward passes beat 1 fwd + 1 bwd, with
no activation memory" — and `cos ∝ 1/√p` makes it *worse* at scale, which constraining `p` does not
rescue (§4.2). What would help is the one thing FL cannot buy: **large bins**, where `D` lives.
**MoE is structurally interesting** — only active experts contribute to a forward pass, so `p_effective`
is per-token active params: block-coordinate probing for free, and the one structural idea that beats
the `√(n/p)` barrier. Unanalysed.

**Is there a general ML contribution**, in descending confidence:

1. **The scale-invariance requirement** (§5.3) and its configuration-level twin, the **autonomy
   requirement** (§0.0) — step, gate and estimator must be ratios, and no constant may be profiled.
2. **`D`.** The `√(n/p)` scaling is confirmed and its constant is 20× off, entirely because each
   directional derivative is taken on a minibatch. **The binding constraint is data noise in the target,
   not variance in the probe** — which inverts the "spend compute on probes" instinct.
3. **The three conserved quantities** (§4). `B` exact and parameter-free; `A` orders arms across `p`;
   `t ≥ Λ²p/(2·B·G_rule·K)` bounds time-to-accuracy and makes constant `s` the optimality condition.
   Transfer outside agnews/DistilBERT is untested.
4. **No longer: `p` as a gradient-quality parameter.** True of `cos`, false of learning and of time.
