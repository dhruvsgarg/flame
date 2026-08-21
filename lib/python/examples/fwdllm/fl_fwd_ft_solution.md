# Forward-gradient fine-tuning — the problem, the model, and where it stands

> **One of two documents.** Here: *why* — the failure, its mechanism, the symbols, the model, and what is
> still unknown. Companion **[fl_fwd_ft_practice.md](fl_fwd_ft_practice.md)** (`P1…P10`) owns *what and
> how*: every number a run produced, every flag, every arm, the dead ends, the launch procedure.
> **Numbers are cited here, never restated** (R2, appendix).
>
> **Scope.** The measurements are FL (FluxTune / FwdLLM, DistilBERT + adapters, on agnews / yahoo /
> yelp-p), but the object of study is **backprop-free fine-tuning by directional derivatives**. §1–§7 are
> the FL instance; §8 is the extension path. **One model, one PEFT scheme, one `p`** — the three datasets
> vary the task, nothing else (§0).

| you want… | go to |
|---|---|
| **the goal, and the autonomy test** | **§0.0** |
| **the state in one screen** | **§0** |
| **what to do next** | [P5](fl_fwd_ft_practice.md#p5--the-queue) |
| the three laws | §4.1 budget `B` · §4.2 progress `A` · §4.6 time `t` |
| the algorithm · self-calibration | §5.1 · §5.5 |
| what died, or what a superseded number became | [P6](fl_fwd_ft_practice.md#p6--dead-ends--do-not-retry) · [P8.1](fl_fwd_ft_practice.md#p81-superseded-numbers--quote-check) |
| a number a run produced | the practice doc — **never here** |

*Parts 1–3 are background. Read them when a mechanism is in question, not to get oriented.*

---

## §0.0 — What we are building

**The goal.** A backprop-free fine-tuning method that **minimises time-to-accuracy on a model and task it
has never seen**, sensing every quantity it needs from the workload and the runtime.

| # | metric, in strict order | the model's coordinate |
|---|---|---|
| **1** | **time to accuracy** | `t ≥ Λ²·p / (2·B·G_rule·K)` round trips (§4.6) |
| **2** | **stability** — reach a peak and hold it | `B ≤ B_max` (§4.1, §4.2) |
| **3** | **overheads** — communication, then compute | uploads `= K·t = Λ²·p/(2·B·G_rule)` |

§4.6 puts all three in one expression, so they need no hand-trading: everything that makes a run faster
also makes it cheaper in bytes. **The cohort pair is the one exception** (K-C, closed): `K` is
free in *total* bytes only while `K ≤ n_req`, it is never free in *peak* rate, and its price is **not**
staleness — staleness is set by `C/n_req`, which contains no `K` (§5.4). Metric 2 is not a trade against
metric 1 but a **ceiling on it**. **The method reduces to one decision — how close to `B_max` to run —
and `B_max` is sensed, not supplied** (which is why 3.1, the sensor's grid, is now the top open item).

**The autonomy requirement, as a test.** *Every constant in the loop must be either (a) exact arithmetic
on quantities the run already logs, (b) a hill-climb on something the runtime measures, or (c) a property
of the deployment the operator genuinely owns.* Anything else is a profiling dependency and is a defect.
§5.5 is the audit; the surviving debts are **`B_max`** (sensed) and **`D`** (forecast-only).

> **Class (c) is deliberately narrow: the model, the PEFT scheme, and `p`. Nothing else** — in particular
> **`P` is sensed** (D4). "No input" means **no tuning knob**: no learning rate, no variance threshold, no
> cohort width, no horizon, no safety factor, no probe count. Each is a number that works on
> agnews/DistilBERT and silently means something else elsewhere — §5.3's ratio principle applied to the
> *configuration* rather than the arithmetic.

---

## §0 — Where the work stands

**In one line:** *we know why it diverges, we have four fixes that work, the sizing formula's shape is
right and its constant is 20× high because every reading's target is an 8-sample minibatch gradient —
and the controller is now **built and run**: it beats its hand-set control on all three datasets, and on
yelp-p it stopped itself at its own peak. What is not yet sound is the sensor that sets its target.*

### Load-bearing — the eight every decision is checked against

*Index only. Each is stated, derived and evidenced where it points. Numbers live in the practice doc (R2).*

| what | where |
|---|---|
| **Design principle** — nothing is compared against a fixed constant unless it is scale-invariant | §2.4, §5.3 |
| **Autonomy principle** — every constant is logged arithmetic, a runtime hill-climb, or a deployment property | §0.0, §5.5 |
| **Budget law** — `Φ = e^B`, `B = ½Σln(1+ρ_t²)`: what a trajectory spends | §4.1 |
| **Progress law** — `A = Σρ_t·cos_t·‖θ_tr,t‖`: what it banks, and peak accuracy is monotone in it | §4.2 |
| **Time law** — `t ≥ Λ²p/(2·B·G_rule·K)`, equality **iff `s` is constant** | §4.6 |
| **Criterion** — `ρ ≤ s·cos` at `s` ≈ 2.9. **Sizes a pool; does not bound a trajectory** | §4.4, §4.5 |
| **Estimator** — `cos = D·√(G_rule·N/p)`: shape right, constant 20× high, shortfall entirely data-side | §6 |
| **Shipped stack** — mean-combine · trust-ratio step · **law-C landing** on a sensed `B_max` · `p` for memory only | §5.1, [P2](fl_fwd_ft_practice.md#p2--the-shipped-stack) |
| **The result it produced** — controller vs hand-set control, three datasets | [P4.11](fl_fwd_ft_practice.md#p411-the-2026-08-20-p-4-pairs--the-law-wins-on-all-three-one-pair-is-valid) |

### Also settled — do not re-measure

`ρ ∝ η` and `ρ ∝ 1/√(K·I)`; the naive `ρ ∝ 1/√K` is false (§2.2) · the variance gate is a
dimensionally-wrong `N`-controller and the *accidental* stabiliser (§2.3) · the defect is shared with sync
`fwdllm` (§2.5) · heterogeneity acts **only** as a step-size multiplier (§2.6) · collapse is **directional
degeneracy** of the head, not logit saturation (§2.7) · probe selection by `|d|` is stability-neutral **by
construction** (§3.2) · at a pinned `ρ*`, **`p` is inert** (§4.2) · **momentum is `ρ` in disguise** (§5.2)
· the commit gate converts pool into commit rate at a net gain, and **`s` is an efficiency knob, not a
safety one** (§4.5) · the collapse budget is **relative**, not absolute (§7.1) · `‖θ_tr‖` is mostly a
**symptom** — perfect decay recovers 20–25% (§7.2) · backprop reaches **0.90** where the best
forward-gradient arm reaches **0.876** (§7.3) · **the controller is built and run** — it beats its
hand-set control on all three datasets and lands on its own peak (§1.4 item 5) · hill-climb **`C`, not
`K`**: commit throughput is flat in `K` at fixed `C` (closes K-C).

### Open — and this is the entire list

| # | question | blocking what | how it closes |
|---|---|---|---|
| **D-2** | **`D` is 2–3× larger on the two arms that trained than near init** — but it is not "training state": pooled across both it is *non-monotone* in accuracy, and its highest-accuracy bin reads the **lowest** `D` while pre-turn steps to post-turn. `const` vs `rm` is the leading confound. Numbers: §6.3 | the sizing formula and §8's generality — `N_req ∝ 1/D²`, so 2× is 4× in pool | **on the rig, not on an arm**: single-commit `cos` has SNR ≈ 1, so only a fixed **trained checkpoint** with unlimited probes can separate `const` from `rm` |
| **H-S** | **A 3.5× shadow loss that is not data-side.** The rig reproduces `L` but gets `S` = 1.68 where the arms read 0.48. Prime suspect: the FD chord — `h‖v‖` ≈ `‖θ_tr‖`, so `d` is a chord-averaged slope, not `⟨g,v⟩` | the last unexplained factor in `cos` | rig: true `⟨g,v⟩` vs the shipped central FD at the shipped `h` (§6.3) |
| **3.1** | **The `B_max` sensor is mis-ranged, so what it senses is mostly `B`.** Its Φ grid starts at 1.5 and the live knee is below that on **every one of 19 fires across three datasets**, so `knee()` extrapolates from its synthetic `Φ`=1 anchor to one reading, `ln Φ_knee` is pinned into 0.21–0.42, and `B_max = B + ln Φ_knee` recedes as budget is spent. Numbers: [P4.11](fl_fwd_ft_practice.md#p411-the-2026-08-20-p-4-pairs--the-law-wins-on-all-three-one-pair-is-valid) | **the central claim** — that `B_max` is a task property the run discovers. The controller still works without it, but "it senses the task's ceiling" is unsupported until this closes | **re-range the grid** below 1.5 and re-read the knee (buildplan row **P**, ~30 min). Nothing on disk answers it: no arm checkpoints a model |
| **3.1c** | **Is `B_max` a fixed total at all?** The probe reads headroom *from `θ_t`* on a model that cannot re-fit; a run re-fits continuously, and measured `B_rem` does not shrink as budget is spent. That is the signature of a **rate limit that is re-earned**, not a tank that drains | the shape of the whole method: if budget is a rate, "spend `B_max` then stop" is the wrong stopping rule and only the `Φ` rail is load-bearing | buildplan rows **F′** (run past the rail with nothing halting) and **P** |
| **3.1b** | **Does `B_max` differ by task at all?** The three combined values (0.795 / 0.805 / 1.023) order by **fire count, not dataset**, so B-1's "neither invariant nor monotone in `num_labels`" is not confirmed by the live probe | whether `ρ*`, `s`, `N` and the stop could ever ship as a fixed constant | falls out of **3.1** — with a grid that brackets the knee, one fire per dataset settles it |
| **G-1** | **Every arm ever run is DistilBERT + adapters at `rf`=16, `p`≈4.5e5.** The three datasets vary the *task*; `p` varies by 1.4% across them. The one time `p` moved for real — `rf`=64, `p`=118,348 — law C + the annealed gate did not compose under **any** `T_res` | "same controller, new model", the §8 extension, and `T_res`=300 / `f`=0.95, which are pinned to one `p` | a second model. Not started; [buildplan §1 hole 3](fl_fwd_ft_buildplan.md) holds the `rf`=64 numbers |

**What is *not* open: the optimizer, any pooling knob, or the controller.** The four fixes enact to spec,
the dynamics are closed by the two laws, and the controller is built and beating its control. What moved
is *where the remaining loss lives* — not in the step rule, the combination rule or the gate, but in the
**data** each reading is taken on, and in the **instrument** that sets the budget (3.1). G-1b sharpens the
first: **every pooling stage buys progress per unit budget and none of them buys budget**, so the only
quantities that can stop a collapse are `ρ` and `T`.

---

# Part 1 — The problem

## §1.1 What the method is, and what it costs

Fine-tune an LLM across federated clients **without a backward pass**. Each client perturbs the trainable
weights along random directions, measures the loss change with forward passes only, and uploads a
scalar-scaled direction. Nothing stores an activation graph, so peak memory is a forward pass and the
operator set is inference-only — that is the whole point, and the part nobody disputes. The price is that
**one measurement returns one number instead of `p`**:

| | backprop | forward-gradient |
|---|---|---|
| cost of one measurement | 1 fwd + 1 bwd ≈ 3 forward-equivalents | **2 forward passes**, no backward |
| what it returns | all `p` components of `g`, exactly | **one scalar** `d = ⟨g,v⟩` |
| memory | stores activations | nothing beyond a forward pass |
| the update produced | `−η·g` — right direction, known length | `−η·d·v` — right **on average**, ~0.1% aligned individually |
| noise sources | data sampling | data sampling **+ which direction you asked about** |

`ĝ = d·v` is unbiased (`E[d·v] = g`) with enormous variance. *On a hillside in fog:* backprop feels the
slope in every direction at once; forward-gradient picks one random direction, takes a test step, feels
whether it went up or down, steps back. **One reading is nearly worthless; a thousand averaged readings
are a slope meter.** Everything here is about whether enough readings can be averaged, fast enough.

## §1.2 The failure

Real 3.93 h run, 189 commits, agnews / DistilBERT-base + AdapterHub adapters:

```
 0.00h  acc 0.398  loss 1.380     <- init, ln(4) = 1.386
 2.13h  acc 0.846  loss 0.554     <- PEAK
 2.95h  acc 0.738  loss 0.692     <- degradation starts INSIDE round 1
 3.87h  acc 0.250  loss 2.371     <- one class, worse than chance
```

**Rise, peak, collapse to a single class.** A second 4 h run agrees on every derived constant, so this is
deterministic, not a seed. Runs cut at ~95 commits stop *at the peak*, which is why it looked intermittent
for months.

## §1.3 What it is not

| ruled out | evidence |
|---|---|
| overfitting | **train** loss diverges too (1.355 → 0.66 → 1.72) |
| plateauing / running out of signal | final loss **2.37 > ln 4** — answering *wrongly and systematically* |
| a round-boundary or data-order effect | degradation starts inside round 1 |
| the finite-difference approximation | the FD displacement gets *relatively smaller* while the failure gets worse |
| a bad seed or flaky infra | two independent 4 h runs agree on every constant |
| something specific to async FluxTune | sync `fwdllm` shows the identical signature (§2.5) |

## §1.4 Definition of done — tracked

| # | criterion | status |
|---|---|---|
| **1** | peak ≥ **0.86** and ends within **0.015** of it at ≥300 commits, without hand-tuning per model or α | **MET, and now on a controller arm across task** — yelp-p `125010` ends **0.0006** below its peak over 1,348 commits; agnews `152215` peaks 0.868 |
| **2** | the setpoint is **computed, not searched** | **MET for `ρ*`** by a different route: `ρ* = √(2·B_max/T_res)` needs no `cos` and no `D` (§4.6a). **Open for the *forecast*** — predicting the accuracy `ρ*` reaches needs `D` (§5.5a) |
| **3** | stability readable in **~20 commits** from `B` and `A` | **MET** — both exact at any horizon; `B` predicted `112201`'s `Φ` to 1.1% over 1,364 commits |
| **4** | every constant in the loop is dimensionless (§5.3) | **PARTIAL** — met for the step rule; the gate is dimensionless in form but its `s` is still empirical |
| **5** | the controller **holds** the trajectory rather than sizing it once | **MET 2026-08-20.** Run as a controller on all three datasets. yelp-p `125010` halted itself on `[BudgetStop] reason=budget` at commit 1,348 — 58% of the vclock it was given — ending 0.0006 below peak, above a control that spent its whole budget and never got there |

**Item 5 was the one this program circled without naming, and it is the one that just closed.** Every
earlier arm either annealed on a schedule chosen offline or held `ρ` fixed, and §4.4's design rule assumes
the horizon `T` is known in advance; the two `const` arms showed that a *correct* one-shot setpoint still
collapses if the horizon outruns it, **at any pool size**. Law C removes the horizon from the inputs.

**Two things item 5 does not yet show.** *(a)* **Holding is demonstrated once.** agnews and yahoo were
killed mid-climb by a watchdog bug, so "reaches and holds" has n=1 (buildplan §2). *(b)* **No arm in this
set ever diverged**, so the stop demonstrated *efficiency* — stopping early at no cost — not the collapse
avoidance that motivated it. Those are different claims and should not be merged.

---

# Part 2 — Mechanism: why it diverges

## §2.1 Every step is orthogonal to `θ`. **MEASURED.**

`‖θ+Δ‖² = ‖θ‖² + 2⟨θ,Δ⟩ + ‖Δ‖²`, so the ratio of observed norm growth to step energy has an exact null at
**1.000 = the step carries no component toward or away from where the model stands**. Measured per
25-commit block: **1.000 ± 0.005 in every block of every arm**, including arms that reach 0.86 and hold.

**Orthogonality is not itself the defect** — `g` is nearly perpendicular to `θ` at high `p`. What it
establishes is that the norm grows by the *full* `‖Δθ‖` every commit with no cancellation, turning the
trajectory into an **exact difference equation** (§4.1). Growth per se is normal (adapters init near
zero); what is atypical is that it is geometric and unbounded.

## §2.2 The step never anneals, and it feeds itself. **MEASURED.**

`ρ ∝ η` holds to 1% across a 20× sweep in `η`. `ρ ∝ 1/√N` holds to 4% over a 1.9× range in `N`, where
`N = K·I` is the *pooled upload count*, not the cohort width. **`ρ ∝ 1/√(K·I)` is the most load-bearing
measured law here.** The naive `ρ ∝ 1/√K` is **falsified**: raising `K` makes the gate commit sooner and
cuts `I` (§2.3).

Whether growth is geometric or arithmetic is decided by the *absolute* step. At `K` = 50 it is flat, so
`‖θ‖²` grows linearly and safely; at `K` = 10 it grows, so `‖θ‖²` is geometric with a ~68-commit doubling
time. **Divergence is present at commit 1** and takes ~150 commits to become visible: steps do not become
more wrongly aimed over time; what grows is the absolute step and the norm it is applied to.

**Why the absolute step grows: the gradient genuinely grows.** A backprop probe on a fixed batch shows
`‖g‖` tracking `‖θ_tr‖` with exponent ≈0.9 in the rising phase, matching the ≈0.8 inferred from `rms|d|`
through a different instrument. The loop *noise inflates the norm → bigger gradient → bigger absolute step
→ more noise* is **real and closed**, which is why norm control could have been curative rather than
cosmetic (§7.2 settles that it is not). In the stable `mean` arms the loop is absent: `‖θ_tr‖` and `‖g‖`
both sit flat all run.

## §2.3 The variance gate is dimensionally wrong — and is the accidental stabiliser. **MEASURED.**

The gate asks the right question — *have I pooled enough readings to trust this direction?* — with a
statistic that carries units. `var` is the per-coordinate variance between two half-means, which for two
samples is `‖G_A − G_B‖²/(2p)`; with `u = d·v` and `‖v‖² ≈ p`:

```
var  =  ||G_A - G_B||^2 / (2p)   ~=   2 * b^2 * ||g||^2 / n
```

1. **`var ∝ 1/n` is why it is an `N`-controller at all.** Across `K` = 20/30/50 the realised `N` lands in
   a narrow band regardless of `K` — the gate hands most of a `K` increase back as fewer iterations. At
   `K` = 10 the target is unreachable, `max_iter` binds, and `N` pins low.
2. **`var ∝ ‖g‖²` is the units bug.** The achievable variance floor drifts 36× over a run — exactly
   `(6.0×)²`, the square of the `‖θ‖` growth. Because the ruler carries units of `‖θ‖²`, holding
   `var ≤ 0.3` forces `N ∝ ‖θ‖²`, hence `ρ ∝ 1/‖θ‖`, hence a **constant absolute step**.

> **The dimensional bug is the stabiliser.** Wrong units are what turn a fixed threshold into a
> `ρ ∝ 1/‖θ‖` anneal — and since `‖θ‖ ∝ √t` under a constant absolute step, that is Robbins–Monro *by
> accident*. **It is not a fix:** `Σρ_t² = Σc/t` diverges logarithmically, so collapse is deferred
> (extrapolated to commit 1,200–1,700), not prevented, and the threshold hard-codes one trajectory. Three
> objections stand against the statistic itself: it has units, it cannot see directions (it is computed
> from scalars), and it never references the step.

This is the precise departure from FwdLLM, whose central pooling mechanism this is. The replacement is not
"a better commit test" but **an explicit `N`-controller with `ρ` as its sensor** — which is what the var
gate turns out to have been all along.

## §2.4 The single defect

**Nothing in the pipeline is scale-invariant.** The estimator (`|d|` tracks `‖θ‖`), the step
(`‖Δθ‖ ∝ |d|`) and the gate (`var ∝ |d|²`) all inflate together, so **no quantity anywhere can be
meaningfully compared against a fixed constant.** All three are one violation of the ratio principle
(§5.3), with the twist that decides the fix: **the third violation partly cancels the first two**, so the
replacement must supply that anneal *deliberately* rather than merely remove the bug.

## §2.5 It is cross-baseline, not FluxTune-specific. **MEASURED.**

Sync `fwdllm` over 131 commits: orthogonality ratio 1.000 ± 0.008, same `ρ` magnitude at commit 1,
geometric norm growth with a ~106-commit doubling time. **Same signature, same magnitude, shared
`_server_update_step`.** So the step rule and the anneal are **cross-baseline hygiene, not a FluxTune
contribution** (§7.4).

One consolation prize (**ANALYSIS**): at matched commits `ρ·√N` is 1.68 for fluxtune vs 1.01 for fwdllm,
which at matched `‖g‖` puts FluxTune's `|JVP|` selection at `G_rule ≈ 3` and **FwdLLM's
cosine-similarity probe selection at `G_rule ≈ 1`, i.e. no better than random.**

## §2.6 Heterogeneity is a step-size multiplier, and nothing more. **MEASURED over 1000× in α.**

**(a) α enters through the gradient scale, and only through it.** At the same `‖θ_tr‖`, `rms|d|` spans
2.5× across α = 0.1 / 1 / 100, propagating straight into `ρ`. **Under raw SGD, heterogeneity is a hidden
×1.6 on the relative step** — and the diverging arms are exactly the low-α ones.

**(b) The pooling invariant is not `ρ·√N`** (it spans 2.3× across α). Invariant to ±1.5% across all seven
α×K arms is the form with the gradient scale divided out:

```
rho * sqrt(N) * ||theta_tr|| / ( eta * rms|d| * sqrt(p) )  =  0.53 +- 0.01
```

**(c) The gate's liveness is set by `‖g‖`, so α decides it as much as `K` does.** `var_threshold = 0.3`
sits *exactly* on the α=1, `K`=20 floor — tuned to one point of a two-dimensional surface. `K` = 10
survives at α = 100 while `K` = 20 does **not** stabilise at α = 0.1, so **"`K ≥ 20` is safe" is false as
stated**; the true statement is *"the gate is safe while it can reach its threshold"*.

**(d) The trust-ratio step removes the α multiplier by construction.** At a pinned `ρ*`, α = 1 and α = 0.1
give `ρ` **identical to six significant figures at every commit**, while `var` still shows the 1.6×
gradient-scale ratio. **The mechanism is still there; the step no longer feels it.** The residual 3-point
accuracy gap is data difficulty at a matched step.

## §2.7 Why it looks healthy for three hours, and what the collapse is

Early on the coherent term (`T·ρ·cos`) beats the `√T` noise, so **the accuracy climb is real learning**;
the inflation term is exponential the whole time but invisible until it is not. `cos` does not degrade
over time. **We never reached a minimum and walked back out** — accuracy peaked because a rising linear
term and a falling term crossed.

**The collapse is directional degeneracy, not logit saturation. MEASURED:** at collapse
`top_class_share → 1.00`, but `logit_norm` is the *same* as in the healthy arm and prediction entropy
stays high. **The head's decision direction is destroyed while its scale is unremarkable.** The objection
*"some of the weight increase was in the right direction"* is answered by arithmetic: ≤1.5% of each step
is aligned, so the coherent part accumulates to a fraction of one norm over ~190 commits while the norm
grows 7×.

## §2.8 The three production monitors

α-independent across every arm measured:

| monitor | reading |
|---|---|
| **`‖θ_tr‖`** | peak accuracy at **28–39**; −5 points at **47–52**; `top_class_share > 0.9` at **62–73**. *The one worth wiring to an alarm.* |
| **`ρ = ‖Δθ‖/‖θ_tr‖`** | exact per commit; feeds `B` and `Λ` |
| **`top_class_share`** | 1.00 is the collapse fingerprint; `logit_norm` does **not** discriminate |

**Score `B` and `A` — never accuracy, never the `‖θ‖²` log-log slope.** Both are exact at any horizon and
readable in ~20 commits. Use `Λ` only to compare arms at one `p`.

---

# Part 3 — Symbols and estimator shape

## §3.1 Symbols

Per-commit unless stated. "Dimensionless" means a pure ratio, so comparing it against a fixed constant is
legitimate — that property is the whole fix (§5.3).

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
| `C` | uploads in flight — caps `K`, and sets throughput and staleness (§5.4) | 30 |
| `η` | server learning rate (a **knob**, superseded by `ρ*`) | 0.01 |
| `ω` | per-upload aggregation weight | 0.70–0.87 — a re-weighting, **not** a step size |
| **`ρ`** | **relative step `‖Δθ‖/‖θ_tr‖`** — an *outcome* under raw SGD, a *knob* under trust-ratio | **0.16 at commit 1** shipped |
| `ρ*` | the relative step an operator *sets* under trust-ratio | the knob `ρ` should have been |
| **`cos(G,g)`** | fraction of the step aligned with the true gradient | 0.036–0.067 closed form; **20× high** (§6) |
| `E[v∥²]` | probe-selection gain | 2.988 at `P`=10, 4.744 at `P`=30 |
| `G_rule` | pooling gain of the rule: `E[v∥²]` if selecting, `P` if averaging | 2.988 → 10 |
| `a`, `b` | estimator shape constants (§3.2) | properties of the **rule**, not the data |
| **`B`** | **budget spent** = `½Σln(1+ρ_t²)`; `‖θ_T‖/‖θ_0‖ = e^B` | §4.1 |
| **`Λ`** | **progress banked** = `Σρ_t·cos_t` | §4.2 |
| `A` | **absolute** progress = `Σρ_t·cos_t·‖θ_tr,t‖` — the `p`-portable coordinate | §4.2 |
| `Φ` | inflation `= e^B` | peak at 2.41–3.11; ≥4.2 destroys |
| `var` | commit gate's statistic: spread of `d` across the pool | drifts with `‖θ‖²` (§2.3) |
| `D` | `cos_measured/cos_predicted` — the data-side shortfall | 0.045–0.051 near init, 0.10–0.15 trained (§6) |

**`v` is a raw Gaussian draw**, never normalised. Three consequences: `‖v‖` concentrates at `√p` to 0.07%,
so **normalising `v` is a no-op for `cos`**; `v` sets the probe displacement `h√p`, so **changing `p`
silently changes the FD spacing** (hence `FWDLLM_FD_SCALE_INVARIANT`); and **isotropy is exact by
construction**, which licenses the `1/√p` overlap below.

**Where the dimensions live cost a revision.** `create_model` builds 1.04M trainable params, but the
trainer replaces `pre_classifier` with an empty module *before any probe is drawn*, so production `p` is
**450,340** and `‖θ_tr‖` is 13.35, not 20.36
([P1](fl_fwd_ft_practice.md#p1--the-system-under-test)). Consequences: `trainable_scope: adapters_only` is
a **no-op**, the FD displacement is 50% of `‖θ_tr‖` rather than 76%, and every `cos` prediction rose ×1.52
when this was found. **`h` itself is pinned** between truncation error above and fp16 catastrophic
cancellation below.

## §3.2 Estimator shape: why "pick the best probe" cannot work

Split any upload `u` into its component along `ĝ = g/‖g‖` and the rest: `u = α·ĝ + u⊥`. **`a` measures the
shadow** (`E[α] = a·‖g‖`); **`b` measures the total length** (`E‖u‖ = b·‖g‖·√p`, so `b = 1` is "as long as
one raw probe"). A single upload's aim is `cos(u,g) = (a/b)/√p`:

| rule | `a` (shadow) | `b` (length) | aim `a/b` | **stability `b²/a`** |
|---|---|---|---|---|
| one raw probe `d·v` | 1 | 1 | 1 | 1 |
| **select best of `P` by `\|d\|`** | `E` | `√E` | `√E` | **1** |
| **average all `P`** | 1 | `1/√P` | `√P` | **1/P** |

The middle row answers *"why doesn't picking the best probe help?"* Selecting on `|d|` raises the shadow
**quadratically** and the length **linearly**. Aim genuinely improves — but stability depends on `b²/a`,
in which the two cancel **exactly, for any `E`**. `a` and `b` are known in closed form before the run; the
one empirical input is that `d` is Gaussian, which is verified
([P3](fl_fwd_ft_practice.md#p3--knob-ledger)).

**Pooling `n` uploads:** the shadow is identical in each and survives untouched; the perpendicular junk
shrinks by `√n`.

```
shadow of pooled G  ~  a * ||g||                    <- unchanged by pooling
||G||               ~  b * ||g|| * sqrt(p/n)        <- shrinks as 1/sqrt(n)
cos(G, g)           =  (a/b) * sqrt(n/p)
rho                 ~  eta * ||G|| / ||theta||      <- also shrinks as 1/sqrt(n)
  =>  rho / cos     ~  (b^2/a) * p / n
```

Read the last line as **safety = (a property of the rule) × (dimensions per pooled reading)**. Improve it
via a better rule (`b²/a`), fewer dimensions (`p`), or more readings (`n`) — that is the whole lever table
(§5.2). **Both halves are MEASURED and rule-dependent as predicted**: `L` and `S` are each invariant
across a 5.5× swing in `b` and a 3× swing in `a` (§6). What the table omits is that `g` is not the same
object for every upload — each `u` measures its own client's 8-sample gradient, and that substitution
costs the constant `D`.

Four consequences worth stating plainly:

**(a) The parameter count is the adversary.** A random direction in `p` dimensions overlaps a fixed target
by ≈ `1/√p`; at `p` = 450k one probe is ~0.15% signal. No tuning removes this.

**(b) Averaging is the only free lever, and it pays twice.** Over `n` independent readings signal adds
**linearly** while near-orthogonal noise adds **in quadrature** — the average is simultaneously better
aimed *and* shorter, so safety improves as `1/n`, not `1/√n`.

**(c) Misaim never cancels in length.** A step perpendicular to `θ` gives `√(‖θ‖²+‖Δ‖²) > ‖θ‖` for *every*
perpendicular direction; sign does not matter, only length. This is §2.1, and it makes §4.1 exact.

**(d) Inflated weights destroy a classifier by randomising which class wins**, not by saturating it.

---

# Part 4 — The model: three conserved statements

## §4.1 The norm law — how much budget a trajectory spends

§2.1 makes the trajectory an exact difference equation `‖θ_{t+1}‖² = ‖θ_t‖²(1+ρ_t²)`. Integrating gives,
with **no free parameter**:

```
INFLATION   Phi = ||theta_T|| / ||theta_0||  =  exp( B ),   B = (1/2) * sum_t ln(1 + rho_t^2)
PROGRESS    Lambda = sum_t rho_t * cos_t,     cos_t = D * sqrt( G_rule * N_t / p )
EFFICIENCY  Lambda / B  =  2 * cos / rho      <- exact by construction
```

**The norm law holds across 21 arms** spanning `ρ` 0.0002–0.22, `N` 10–200, α 0.1–1, both combination
rules, both step rules, both gates, `p` 118k–450k, and 177–1,273 commits. **It contains no `cos`, no `N`,
no rule, no α and no `p`: `‖θ_T‖` is a function of the `ρ` trajectory and nothing else.** The sharpest
case is an arm where `N` fell 200 → 40 under an annealed gate and the law still fit to 0.02%.

> **It holds only while steps are independent. MEASURED.** Server momentum correlates consecutive steps,
> so §2.1's ratio leaves 1.000: it reads **1.007 / 2.981 / 6.615** at β = 0 / 0.5 / 0.75 against
> `(1+β)/(1−β)` = **1 / 3 / 7**. The law generalises with that one factor,
> `Φ = exp( ((1+β)/(1−β))·B )`, fitting the two momentum arms to 0.5% and 2%. **Anything that correlates
> steps in time enters here, multiplicatively.**

> **Its accuracy is `ρ`-dependent** — the flat "<0.3%" once claimed is wrong. Replayed: **<0.3%** on
> low-`ρ` trust-ratio arms, **~2%** on high-`ρ` raw-SGD arms, **−5.8%** on the collapsed α=0.1 arm. Sum
> `B` over the steps that actually lie between the first and last norm sample — the mis-aligned window
> costs 1.2% vs 0.8% mean error.

## §4.2 The progress law — how much learning a trajectory banks

21 arms sorted by `Λ`: **peak accuracy is monotone from 0.377 to 0.876 with no exception outside replicate
noise**, flattening around `Λ` ≈ 0.95 and creeping after. Read `Λ` as **accumulated aligned displacement
in units of `‖θ_tr‖`** — travel ~0.6 of your own length in the right direction to reach 0.85, ~1.3 to
reach 0.876. Calibration is [P4](fl_fwd_ft_practice.md#p4--arm-ledger).

> **`Λ` is the *relative* coordinate and does not transfer across `p`. MEASURED.** Adapters init at a
> `p`-independent per-element scale, so **`‖θ_tr‖ ∝ √p`** (ratio constant to ±1.3%), and therefore
>
> ```
> A  =  Lambda * ||theta_tr||  =  sum_t rho_t * sqrt( G_rule * N_t ) * (||theta_tr||/sqrt(p))
> ```
>
> **`p` cancels — `A`, not `Λ`, is the coordinate accuracy tracks.** Out of sample, a curve fitted on
> 2,278 `rf`=16 points predicts 208 held-out `rf`=32/64 points to **0.0246 in `A` against 0.0623 in `Λ`**,
> winning on each held-out arm separately.

**Corollary — at a pinned `ρ*`, `p` does nothing.** `A` is `p`-free and `B` depends only on the `ρ`
trajectory. The old `p` ladder's apparent win was raw SGD lowering the *realised* `ρ`. `p` remains a
memory, compute and comms lever; `cos ∝ 1/√p` is real and buys nothing that `‖θ_tr‖` does not give back.

**Among arms that learned, whether an arm *holds* its peak is decided by `Φ` and nothing else.**

> **The peak sits at a fixed `Φ` across the whole portfolio. MEASURED, 7/7.** Every arm that learned
> (peak ≥ 0.80), turned, and still had >15% of its run left peaks at **`Φ` = 2.41–3.11, mean 2.71** —
> across both combination rules, both step rules, `rf` = 16/32/64, α = 0.1 and 1, `T` from 177 to 1,364
> and `ρ` from 0.06 to 0.20. Arms that *held* peaked at `Φ` = 2.82–3.54 and had no budget left to walk
> past it.
>
> **This is the most transferable number in the document, and it is not fitted** — `Φ` comes from `ρ`
> alone and the peak location was never used to choose anything. Read it geometrically: steps are
> orthogonal to `θ`, so **`Φ` is the reciprocal of the un-junked fraction of `‖θ_tr‖`**. `Φ` = 2.7 says a
> head fails when it is more than ~68° off the direction that earned its accuracy — a statement about
> classifiers, not about agnews.

Two qualifiers, both load-bearing. **The `Φ` ≤ 3.63 harbour is marginal and was fitted only on annealed
arms** — the two `const` arms peak at `Φ` = 2.82 / 2.51 and cross −0.015 at 3.43 / 3.79, so read
`Φ ≤ 3.6` as the edge of the cliff, not a safe band. **Pool moves the *damage* at a given `Φ`, not the `Φ`
at which damage starts.** And **"peak ≥ 0.80" is required**: four arms sit at `Φ` = 1.00–1.01 with *no*
inflation and still fall 0.06–0.12 below their peak, all with `Λ` ≤ 0.068. `Φ` governs *losing what you
learned*; it says nothing about arms that never learned — which independently shows accuracy loss does not
require norm growth (§7.2).

**The two failure modes are separately diagnosable ~20 commits in: too little `Λ` = never learned; too
much `B` = learned and then lost it.**

## §4.3 Three consequences that restructure everything downstream

**(a) There is no critical `ρ`.** Under a pinned `ρ`, `Φ = (1+ρ*²)^{T/2}` for *any* `ρ* > 0` — geometric
always, doubling at `1.4/ρ*²`. The apparent boundary at `ρ ≈ 0.09` was **a horizon artifact**: the locus
where doubling time falls below run length. **`ρ` is not a threshold; it is the rate at which a fixed
budget is spent.**

**(b) The `‖θ‖²` log-log slope carries no stability information under trust-ratio** — bounded above by 1
*by construction*, so every "sub-linear ⇒ safe" reading is vacuous. Under raw SGD a slope > 1 *does* mean
something: the absolute step is outgrowing `‖θ‖`, i.e. §2.2's feedback is live.

**(c) The closed form's *scaling* is right and its *constant* is 20× high** (§6). A constant factor cannot
change an ordering, which is why `Λ` could rank 21 arms while being 20× wrong. What the constant costs is
the sizing formula (§4.5), not the rankings.

## §4.4 The criterion

Two clocks. **The deadline:** coherent progress grows as `T·ρ·cos`, noise displacement as `√T·ρ`; signal
overtakes noise at `T ≈ 1/cos²`. **The budget:** the norm inflates by `(1+ρ²)^{T/2}`, doubling at
`T ≈ 1.4/ρ²`. You survive iff `1/cos² ≤ 1.4/ρ²`:

> ## ρ ≲ s · cos(G, g)
> **The relative step must not exceed the fraction of it that is actually aligned with the gradient.**

The O(1) constants that derivation drops are now measured. Eliminating `T` between the two laws gives
`B = Λ·ρ/(2cos)`, so fitting a target `Λ_req` inside a budget `B_max` requires
`ρ/cos ≤ 2·B_max/Λ_req` = **2.6–4.3** — not the 0.3–0.5 originally derived, a factor of ~8 that is exactly
the 5–10× seen empirically. **`s` = 2.6–4.3 is stated against the closed-form `cos`**; against the
measured one it is 20× larger, and the two must never be mixed in one expression.

**The design rule.** For a fixed learning target `Λ_req` over `T` commits, Cauchy–Schwarz makes constant
`ρ = Λ_req/(T·cos)` the budget-minimising trajectory, at cost `B = Λ_req²/(2T·cos²)`:

> **Budget cost for a fixed amount of learning is ∝ `ρ`; wall-clock cost is ∝ `1/ρ`. Pick the smallest `ρ`
> whose commit count you can afford.** There is nothing to tune — `ρ` is the exchange rate between wall
> clock and safety margin, and `cos` (via `P`, `K`, `I`, `p`) sets the exchange rate itself.

Note what (a) already killed: *"walk `ρ*` up until it breaks"* is withdrawn. There is no cliff to find,
only an exchange rate to pick.

## §4.5 Sizing a pool — and the two conventions that must never be mixed

```
G_rule = E[v_par^2] for select-one-of-P  |  = P for average-all-P
cos           = D * sqrt( G_rule * N / p )        N = K*I uploads pooled server-side
N_req( rho* ) = p * (rho*/s)^2 / G_rule           <- pool needed to make rho* safe
```

`D` rescales `cos`, `Λ` and `s` **together**, so every ratio, ordering and design rule is untouched by it.
The one thing that breaks is **mixing conventions** — pairing the closed-form `cos` with the superseded
`s` = 0.4 is what made every gate A/B unrunnable (P6).

**The criterion is empirically validated, and sharply. MEASURED over 17 arms.** Scoring `ρ/cos` against
the closed-form `cos`, among arms that learned: **`ρ/cos` ≤ 2.67 → all 12 end within 0.015 of peak;
≥ 3.11 → all 4 degrade.** No overlap, and the threshold falls inside the predicted 2.6–4.3, so **`s` ≈ 2.9
is the operating constant** and `ρ ≤ s·cos` discriminates 16/16. (The `ρ/cos` = 0.03 arm that also
collapsed never learned — §4.2's qualifier, not a counterexample.)

> **What the criterion is *for*, sharpened by G-1b.** It sizes a pool so an arm **arrives** before it
> spends its budget. It says nothing about **staying** — and the two were conflated every time `s` was
> called a safety factor. Two arms at `ρ` = 0.06 `const` and `s` = 2.9 / 1.5 spend `B` at an identical
> rate and **both turn**, because `B` contains no `N`, no `cos` and no `s`. What the lower `s` buys is
> efficiency `Λ/B` = 2`cos`/`ρ`, doubled exactly
> ([P4.2](fl_fwd_ft_practice.md#p42-g-1b--s-is-efficiency-not-safety)). **So no gate setting is a stability
> mechanism** — the only quantities entering the budget are `ρ` and `T`, which is why §1.4 item 5 is the
> whole of what is left.

## §4.6 The time law — a lower bound on time-to-accuracy. **DERIVED, MEASURED to 0.4%.**

Per commit `ΔΛ = ρ·cos`, `ΔB = ½ρ²`, and the wall-clock cost is `I = N/K` **serial round trips**. Write
`cos_t = c·√N_t` with `c = D·√(G_rule/p)`. Cauchy–Schwarz on `Λ = Σ ρ_t·c√N_t` gives
`Λ² ≤ (Σρ_t²)·(c²ΣN_t)`, i.e. with `t` = total round trips:

> ## t ≥ Λ² · p / ( 2 · B · G_rule · K )
> **Round trips to reach a given `Λ` depend only on the budget you spend getting there — not on how you
> split it into `ρ`, `N` and `T`.** Equality **iff `ρ_t ∝ √N_t`**, i.e. **iff `s` is held constant.**

**MEASURED over 12 arms.** Achieved to **0.4%** by both `const` arms — the only two holding `ρ` and `N`
fixed together — and missed by exactly the arms whose `s` drifted:

| arm | regime | miss vs bound |
|---|---|---|
| `145729`, `112201` | `const` ρ, `const` `N` — `s` pinned | **+0.4%, +0.4%** |
| `013917` | annealed gate: `N` falls **with** `ρ` | +1.7% |
| `200242`/`200325`/`013806`/`200358` | raw SGD, `N` pinned, `ρ` drifts slowly | +0.2% to +1.1% |
| `212009`, `222817` | `select`, `p` ladder | +3.2%, +3.6% |
| `035045` | `rm` anneal at **pinned** `N` — `s` drifts by `ρ_t` | **+8.4%** |
| `065837`, `223446` | free `var` gate — `N` swings independently of `ρ` | **+24%, +23%** |

> **Holding `s` constant is not a safety choice — it is the time-optimality condition**, and the free
> `var` gate leaves **24%** of the wall clock on the table by violating it. That is the sharpest
> justification C2 has ever had.

**Four consequences.** **(a) Spend the whole budget: `t ∝ 1/B`** — unspent budget is wasted time, and `B`
is capped at `B_max`, so there is no speed-versus-stability trade *except* through `B_max`. **(b) `ρ` and
`N` are degenerate**: given `(Λ, B)`, every split costs the same round trips **and the same bytes**
(`Σ N_t = K·t = Λ²p/(2·B·G_rule)`, which is `ρ`-free), so choose `ρ` for control resolution and `N` for
gate reachability. **(c) `G_rule` is a speed lever; `K` is open (K-C).** `t ∝ 1/G_rule` makes
`select → mean` a **3.35×** cut in time-to-accuracy (10 / 2.988), reframing C1 as a *speed* result, and
that half is untouched. The `K` half is not: `t` counts **gate iterations**, which are serial wall-clock
waits only when `C = K`. The dispatcher refills to `C` and never consults `K`
(`flame/selector/async_base.py:405`), so at `C > K` the run is arrival-limited and `t_wall ∝ Σ N·τ/C` —
`K`-free. Total uploads are `K`-independent either way. **(d) `p` remains inert** — a fixed
accuracy needs fixed `A`, and `‖θ_tr‖ ∝ √p` means `Λ ∝ 1/√p`, so `Λ²p` is `p`-free.

### §4.6a The setpoint, and why `T` is not an input

`ρ` is degenerate for cost, so it is chosen for **control resolution** — how many commits the controller
gets to decide over. Setting `B = B_max` in `B ≈ ½Tρ²`:

```
rho*  = sqrt( 2 * B_max / T_res )    T_res = commits of resolution wanted, NOT a budget
```

At `B_max` = ln 2.7 this returns 0.062 at `T_res` = 500 and **0.081 at `T_res` = 300**, against the 0.06
the portfolio reached by search. **`T` was never a budget input; it is a controller hyperparameter with an
obvious default.** *`T_res` = 500 was chosen only to reproduce that searched 0.06 and was refuted on
replay (T5, 2026-08-15) — it floors the commit gate. **The shipped value is 300**; both the value and the
sense in which `T_res` is used (a rate, not a deadline) are settled in `fl_fwd_ft_buildplan.md` §5.*

Two `s`-side readings. `Λ_peak = 2·ln(Φ_peak)/s` scores `s`=2.9 → 0.63 against **0.72** observed and
`s`=1.5 → 1.22 against **1.30**. And `T_peak = 2·ln(Φ_peak)/ρ²` contains **no `s`**, so a lower `s` buys
better commits, never more of them. The floor is gate reachability, `I = ⌈n_req/K⌉ ≤ max_iter`, which at
`K` = 10 is **`s` ≥ 0.90**.

**`Λ = 2B/s` is an identity, not an approximation — MEASURED out of sample (T5, 2026-08-15).** Under the
`n_target` gate `N = p(ρ/s)²/G_rule`, so `cos = √(G_rule·N/p) = ρ/s` and `Λ = Σρ·cos = Σρ²/s = 2B/s`.
Replayed over five arms it holds to **−0.3%** on the two that pin `s` (`145729`, `112201`) and misses
**+21.5 to +23.3%** on the three whose `s` drifts (`035045`, `084554`, `003648`) — the same arms, in the
same order, as the miss table above. Three consequences:

- **The `ρ` schedule is `Λ`-neutral at fixed `B`.** Two landing laws that spend the same budget bank the
  same learning; they differ only in how many commits they take. Comparing schedules on `Λ` without
  matching `B` compares nothing.
- **`s`, not the schedule, is the only lever on `Λ` per unit budget** — which is why G-1b's `s` verdict
  (efficiency, not safety) is load-bearing and the anneal's shape is not.
- **`T_res` is therefore free to be chosen for well-posedness rather than for yield.** That is what settles
  the landing law: `T_res` is a **rate** (`ρ*_t = √(2·(B_max − B_t)/T_res)`, `T_res` never decremented),
  not a deadline. A decrementing `T_res` would make `T` an operator input again — the thing this section is
  named for — and buys nothing, since `Λ` does not depend on it. Full decision, the two refuted
  alternatives, and the constants (`T_res` = 300, `f` = 0.95, `ρ_max` = `s·√(max_iter·K·G_rule/p)`):
  `fl_fwd_ft_buildplan.md` §5.

---

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
           rho*   <- min( rho_max, sqrt( 2*(B_max - B) / T_res ) )     # law C (4.6a, D5)
                     T_res = 300, a RATE never decremented; rho_max = s*sqrt(max_iter*K*G_rule/p)
           run ~150 commits -- spends B ~ 0.21, Phi ~ 1.23, negligible
  PHASE B: B_max  <- noise-injection probe, ~6 evals, forward-only        (5.5b)  [BUILT + LIVE;
           rho*   <- re-derived from the measured B_max                            its GRID is wrong, 3.1]
           re-fires every 150 commits; senses combine by `anchor` (latest)
  K/C, P          <- hill-climb (K,C)/tau and P/tau(P)                    (5.5e)  [NOT BUILT]
  N               <- from the gate, holding s constant = time-optimal     (4.6)

PER COMMIT (server):
  dispatch to C clients, wait for K uploads   # async: K arrives, stragglers roll into the next
  G <- sum_k omega_k u_k / sum_k omega_k      # omega re-weights; it must NOT set magnitude
  commit when the pool reaches N_target       # NOT a cos_hat test -- unmeasurable today (P6)
  theta_tr <- theta_tr - rho*_t * ||theta_tr|| * G / ||G||
  log rho, ||theta_tr||, top_class_share      # the three monitors (2.8)
  B += 0.5*ln(1 + rho_t^2)                    # exact, no free parameter (4.1)
  anneal rho so that B LANDS on B_max         # not Robbins-Monro: unspent budget = wasted time (4.6a)
  stop when held-out accuracy SATURATES        # PRIMARY [NOT BUILT -- buildplan row E]
  stop when smoothed Phi crosses 2.7           # the damage rail (P4.1); is 2.7 right? row F'
  # B >= f*B_max is NO LONGER a termination rule -- B_max drives rho* only
  # NO accuracy target: the run finds its own ceiling (5.5f D2)
```

**Everything above enacts to spec and has run end to end on three datasets.** The two remaining
`[NOT BUILT]` lines — the `K`/`P` hill-climb and the saturation stop — are *additions*, not gaps: the loop
lands and terminates correctly without either. Every place a *decision* is made from *sensed* state rather
than arithmetic is the §0.0 autonomy requirement being met, and is why `s`, `ρ`, `N`, `I` and `p` are no
longer decisions at all.

**The one sensed quantity that is not yet sound is `B_max`** (3.1). The probe fires and the loop lands on
what it returns, but its grid does not bracket the knee, so what it returns tracks `B` rather than the
task. **The controller's behaviour is validated; its target is not.**

**Two decisions follow from that, taken 2026-08-20** and reflected above. *(1)* Senses combine by
**`anchor`**, not `mean`: across 19 fires the measured headroom `ln Φ_knee` shows **no downward trend**
while `mean`-minus-`B` collapses (yelp-p 0.246 measured against 0.084 used), which annealed `ρ*` to
**1.7× below** what the current measurement supported and is why that arm halted 0.060 short of its
reference while still climbing. *(2)* **Termination moves to saturation**; `B ≥ f·B_max` is demoted, and
`B_max` survives only as the input to law C's `ρ*`. `anchor`'s standing objection — that it never
terminates — is void once saturation is what terminates.

## §5.2 The lever table — this ranks every possible fix

**`ρ/cos` is "will this survive"; `ρ·cos` is "how fast does it learn"; `t` is "how long until it does".**

| lever | `ρ/cos` | progress per unit `B` | effect on `t` | bytes | who decides it |
|---|---|---|---|---|---|
| **`P`-averaging** | **∝ 1/P** | **∝ √P** | **∝ 1/P** | **∝ 1/P** | **sensed** — hill-climb `P/τ(P)` |
| **`K`** (cohort width) | **∝ 1/K** | **∝ √K** | **open — `∝ 1/K` only if `C = K`** (K-C) | total free while `K ≤ n_req`; peak rate ∝ `K` | the gate, for reachability |
| **`C`** (uploads in flight) | — | — | **∝ 1/C** while arrival-limited | none | **sensed** — hill-climb `C/τ(C)` against `D(C)` and availability |
| **`I`** (iterations/bin) | **∝ 1/I** | **∝ √I** | none — degenerate with `ρ` | none | the gate, for reachability |
| **`B`** (budget spent) | — | — | **∝ 1/B** | ∝ 1/B | spend to `B_max`, which is sensed |
| `p` | ∝ √p | **invariant** — `‖θ_tr‖ ∝ √p` cancels it | **none** | ∝ p | operator: memory/comms |
| `η` | ∝ η | **invariant — pays 1:1** | none | none | eliminated by `ρ*` |
| momentum `β` | ∝ `√((1−β)/(1+β))` | **invariant — `√x` for `x`** | none | none | refuted |
| probe selection `E[v∥²]` | **invariant** (`b²/a` = 1) | invariant | none | none | eliminated (§3.2) |
| step normalization | sets `ρ` to an operator constant | decoupled | — | — | shipped |

**Read the second and third columns, not the first.** `C` is in the table for the third column only — it
pools nothing, it just decides how fast the same uploads arrive. Four rows improve `ρ/cos` and *do not*
improve progress per unit budget: `p`, `η`, `β` and probe selection. Only `P`, `K` and `I` — the three
stages that pool **independent readings of independent data** — are free, and they all buy `√n`. **A lever is free if
and only if it raises `cos` without correlating steps and without shrinking `‖θ_tr‖`.** That one sentence
replaces four separately-discovered dead ends.

### §5.2a Every pooling stage is the same currency. **DERIVED from §6.**

With `B` the bin size, `σ` the per-sample gradient spread and `g*` the population gradient,
`rms‖g_bin‖ ≈ σ/√B`, so `D = ‖g*‖√B/σ` and — for `mean`, where `b = 1/√P` —

```
cos     =  (||g*||/sigma) * sqrt( P * B * K * I / p )
compute =  2 * P * B * K * I     forward passes over one sample, per commit
      =>  cos  =  (||g*||/sigma) * sqrt( compute / (2p) )
```

> **`cos ∝ √compute`, with the same constant for all four stages.** `P`, `B`, `K` and `I` are not four
> levers; they are four spellings of one. This is why every sweep came back with the same exchange rate.

**Choose the spelling by latency and bytes, since the FLOPs are fixed** — `P` and bin size cost no round
trips and no bytes; `K` is parallel but brings staleness and costs `∝ K` in bytes per commit; `I` costs
**one serial round trip each**.

> **There is no lever left to find.** `b²/a` is the one escape from `√compute`, and `select → mean` already
> spent it (10× at zero compute); §3.2 says there is no second. Both apparent escapes are dead: `D` does
> not move with bin size or bin count (§6.3), and `‖g*‖/σ` is a property of the task and partition.
> **That makes the controller, not the estimator, the remaining engineering.**

## §5.3 The ratio principle

Every quantity compared against a fixed constant must be **scale-invariant**, because anything carrying
units silently changes meaning as training proceeds. The test: *re-parameterise so `‖θ‖` doubles; the loss
surface shape is unchanged, so the trajectory should be.* Three replacements follow: absolute step `η` →
relative step `ρ*`; absolute variance threshold → an `N` target; absolute iteration cap → a measured
adequacy condition.

**Why the criterion does not have to enumerate its failure modes.** Anything not thought of enters through
exactly **two** channels: it changes the step taken → shows up in **`ρ`**, logged exactly; or it degrades
the pool → shows up as **`n_eff < n`**, hence in `cos`. **Channel 2 is the unpaid half.** `n_eff` detects
*correlation* among uploads and returns 1.00 ± 0.01 in every arm ever replayed; it does **not** detect
*directional* disagreement, because at `p` = 4.5e5 differing `g_k` move `var` by `O(n/p)` and are
invisible. **Scale-free setpoints therefore have to come from the step rule, not from channel 2.**

## §5.4 Why the priority order is a hill-climb, not a list

§5.2's third column is the whole ranking: only `B`, `K` and `G_rule` move time, and each has a *sensed*
objective rather than a fixed rank. **Two things the old ranked list got wrong.** `I` was ranked "last,
and only what the gate demands" — but `I` and `K` are the *same* quantity `N` split two ways, and §4.6
shows the split is free. And "`K` first because it pools independent data, which is where `D` lives" was a
conjecture **refuted** by D-1: `D` is flat over 64× in bin size (§6.3).

**The one unmeasured cost is staleness, and it is priced against `C`, not `K`. ANALYSIS.** The time law
assumes every pooled upload reads the *current* `θ`; a stale one reads an older `θ`, and its staleness in
commits is its flight time over the commit interval — with `C` uploads in flight and a commit consuming
`n_req` of them, that is **`C/n_req`**, in which `K` does not appear. Raising `K` at fixed `C` cannot add
staleness and may remove it, since a contributor is parked from re-dispatch until a release boundary. So
the hill-climb the controller wants is **`t ∝ τ(C)/(D(C)²·C)`**, `K`'s remaining job is gate
reachability, and **`D(·)` has never been measured against either.** This is K-C; K-1 is its test.

## §5.5 Deploying without a profiling run

**The objection this answers.** Every constant in Parts 1–7 was obtained by profiling — 30 arms on one
model and one task. A real deployment gets no such budget, so *"it works after we sweep it"* is not a
method.

**§0.0's autonomy test applied to the whole loop.** Rows marked *was a knob* are ones this program started
with and no longer has to set.

| quantity | class | how it is obtained now | status |
|---|---|---|---|
| `ρ` | (a) | it *is* the knob — trust-ratio enacts `ρ = ρ*` to 8.7e-5. *`η` subsumed* | **built** |
| `B`, `Φ` | (a) | `½Σln(1+ρ²)` — exact, readable in ~20 commits | **built** |
| `Λ`, `A` | (a) | `Σρ·√(G_rule·N/p)`, ×`‖θ_tr‖` | **built** |
| `p`, `‖θ_tr‖` | (a) | read off the model at init (`[ProbeDim]`) | **built** |
| `τ` round-trip time | (a) | the aggregator already times every round trip | **built, unused** |
| `N`, `I` | (a) | the gate's closed form given `s`. *`var_threshold` replaced by an `N` target* | **built** |
| `s`, hence `n_req` | (a) | held constant — the time-optimality condition. `n_req = p(ρ*/s)²/G_rule` | **built** |
| `ρ*` | (a) | `min(ρ_max, √(2·(B_max−B)/T_res))`, `T_res`=300 fixed (§4.6a, D5). *The anneal exponent folds in once `B` is tracked* | **live** |
| **`B_max`** | **(b)** | **noise-injection probe, ~6 evals, forward-only, periodic** (§5.5b) | **offline only** |
| **`K`, `C`** | **(b)** | **hill-climb the pair** against availability (§5.5e). *Which one carries the wall clock is K-C* | **not built** |
| **`P`, bin size** | **(b)** | **hill-climb `P/τ(P)`** — *one lever, not four: `cos ∝ √compute`* | **not built** |
| saturation signal | (b) | Prechelt generalization-loss / patience on *smoothed* held-out accuracy — a raw `dAcc/dΛ` slope false-triggers on a dip and misses a masked plateau (buildplan §3 row E) | **not built** |
| `D` | (b) | `cos` audit on a stride. **Forecast-only** (§5.5a) | **built** |
| `p` / PEFT rank | (c) | device memory budget; inert for learning and for time | operator |
| model, PEFT scheme | (c) | the deployment. α is **neutralised**, not sensed | operator |
| *probe selection rule* · *momentum `β`* | — | **eliminated** — `b²/a` = 1 · `√x` progress for `x` budget | — |

> **Four gaps, and they are the entire remaining build**: `B_max` is not sensed online, neither `K` nor
> `P` is adaptive, and there is no saturation stop. All four are C-1. **Nothing else in the loop needs a
> number an operator has to know.**

**What follows for the controller, and it changed the shipped recommendation.** *Hold `s` constant* — the
Cauchy–Schwarz equality condition, which under an anneal means `N` must fall **with** `ρ`
(`gate_rho_ref=annealed`). *Anneal to **land** on `B_max`*, never to stay under it: Robbins–Monro
converges to some `B_∞` and `t ∝ 1/B_∞`, so unspent budget is wall clock given away. *Stop on saturation, not on a
schedule* — a Prechelt GL/patience criterion on smoothed held-out accuracy, superseding the raw `dAcc/dΛ`
slope this section once named (§5.5f D2). **The `const` arms did not need an anneal; they needed a stop** — `145729` peaked 0.876 at
commit 514 and ran 431 further commits into the ground.

### §5.5a The operation / prediction split — where `D` actually bites

| you want to… | needs | blocked on `D`? |
|---|---|---|
| set `ρ*` | `B_max`, `T_res` | **no** |
| set `N`/`s`, and enact them | the closed form, exact | **no** |
| know how much budget you have spent | `ρ` — exact, parameter-free | **no** |
| know *when to stop* | `B_max` | **no** |
| **predict the accuracy you will reach** | `cos`, hence `D`, plus the `Λ` curve | **yes** |

**So `D` and the dose-response curve are needed to *forecast* a run, never to *run* one.** A deployment
that cannot profile can still be operated safely and optimally; what it loses is the ability to promise a
number in advance. That is the right thing to lose.

### §5.5b Sensing `B_max` — one shot, forward passes only

**Inflation can be injected instead of waited for** (the method that settled §7). Add isotropic Gaussian
noise to the trainable slice scaled so `‖θ_tr‖` grows by `Φ`, read accuracy back on a grid of `Φ`, and take
the knee on **chance-normalized** accuracy. **~6 evals on a copy** — no training, no gradients, the same
operator set the method already restricts itself to.

> **The shipped grid `Φ` ∈ {1.5 … 4} is wrong, and this is open question 3.1.** It was sized off B-1's
> knees (2.0–3.5), measured offline. The *live* knee is far below 1.5 — this section already said so
> ("biased conservative … reads the knee ~0.6–1.2 low") and the grid was never moved to match. Across 19
> in-run fires on three datasets, **every point read chance**, so `knee()` falls through to a two-point
> extrapolation from its synthetic `(Φ=1, 1.0)` anchor to the Φ=1.5 reading alone. The grid must bracket
> the knee from **below**; until it does, `B_max = B + ln Φ_knee` recedes as `B` is spent and the sensed
> value is not a task property.

> **The probe measures a REMAINING budget, so `B_max = B + ln Φ_peak`.** `Φ` is
> read against the norm of the model *as it stands now*, while `B` accumulates from `θ_0`; taking
> `ln Φ_peak` as the total makes the two incomparable, and in practice makes `B_max` land *below* the
> spend on the first fire — `ρ*` = 0 for 23–48% of commits on all four P-4 arms
> ([P4.7](fl_fwd_ft_practice.md#p47-p-4--the-law-beats-the-control-the-implementation-had-four-defects)).
> Anchored this way `B_max` > `B` by construction, so a re-sense can only ever move the landing point, not
> stop the run — which is what makes 3.3's stop a genuine backstop rather than a race with the sensor.

**Built** as `probe_inflation_damage.py` and wired into the aggregator as `[BmaxProbe]`
(`b_max_probe_every`); it fires live every 150 commits and 19 fires are on record. It runs on the
model being trained, so it calibrates the deployment rather than a proxy, and it is **biased conservative** — it noises a model that cannot re-fit, reading the knee
~0.6–1.2 low (§7.1), so it under-spends budget, never over-spends. **`b_max_probe_phis` is read from
`args` and set by nothing**, so the grid is the module constant in practice — and `condition_fp` does not
cover it.

**Why `B_max` is the right constant to bet on — and why it must still be sensed, not shipped fixed.**
*(1)* It is **geometric, not task-shaped in its mechanism** — `Φ_peak` ≈ 2.7 says a head fails past ~68°
off the direction that earned its accuracy; the collapse floor confirms this (B-1: every dataset's
post-collapse accuracy lands at its own chance level `1/K`, the same directional-degeneracy signature
everywhere). *(2)* It **transfers across model capacity** — two models differing 1.77× in norm share a
knee at the same `Φ` (§7.1). *(3)* It **held over the whole agnews portfolio unfitted** — 7/7 turned arms
at `Φ` = 2.41–3.11, and as a stopping rule it gives up 0.005 of peak on average (P4.1). **It does NOT
transfer across task** — B-1's three *offline* knees are neither invariant nor monotone in `num_labels`
(§7.1 has the numbers). So the *mechanism* is geometric and general; the *value* is not derivable from a
dataset property known in advance, which is what makes the probe mandatory rather than a fallback.
**The live probe has not yet confirmed the across-task difference** (3.1b): its three combined values
order by fire count, not by dataset — which is exactly what 3.1 predicts a mis-ranged grid would do.

### §5.5e Adaptive `K` and `P` — the same rule twice

```
minimise   t_wall  ~  tau(X) / ( D(X)^2 * X )        X = C if arrival-limited, K if C == K  (K-C)
```

**The variable is not yet known to be `K`.** The 8.9 → 12.3 s round trip at `K` = 30 → 50 was measured on
arms that moved `C` 60 → 100 alongside, holding `C/K` = 2, so it prices `K` and `C` together and cannot
separate them; the same is true of every staleness reading (max 1/2/4 at `C` = 30/60/100). **`D(·)` has
never been measured against either.** **Availability sets the ceiling**, not a config:
`K ≤ C ≤` trainers actually available. Uploads are `K`-independent, so this hill-climb spends no total
communication budget — it trades staleness for wall clock and nothing else, which makes the pair the
**first** thing the controller should move and the **last** thing it should be conservative about; the
opposite of `dynamic_kc` (`k_max` = 15, disabled). **K-1 settles which of the two it is climbing.**

**`P` takes the identical rule, and it also cuts bytes.** `G_rule = P` under `mean`, so `t ∝ 1/P` *and*
uploads `∝ 1/P` at invariant total client compute. The only thing that can cancel it is the round trip
becoming compute-bound: `minimise t_wall ~ tau(P)/P`, which cancels exactly if `τ ∝ P`.

**One correctness consequence of a moving `P`.** `G_rule` becomes time-varying, so every closed form must
use the *current* value: `Λ = Σ ρ_t·√(G_rule_t·N_t/p)`, and the time law's pooling variable becomes
`Σ G_rule_t·N_t` — total **probe readings**, not uploads.

### §5.5f The four controller decisions, settled 2026-08-12

| # | question | **decision** | why |
|---|---|---|---|
| **D1** | `B_max` needs a partly-trained model to measure, but `ρ*` needs `B_max` to start | **Two-phase from a safe prior.** Phase A at `B_max` = **ln 2** — *"the model survives its weights doubling"* — giving `ρ*` = **0.068 at `T_res` = 300**; after ~150 commits that has spent `B` ≈ 0.27 (`Φ` = 1.31). Phase B injection-probes and re-derives `ρ*` | ln 2 is the **weakest non-trivial claim**, not a fitted number, and Phase A's spend is negligible against *any* plausible `B_max` |
| **D2** | what is "the target accuracy"? | **No target.** Stop when `B → B_max` (the `Φ`-threshold rule, 3.3, replay-validated P4.1) crosses first, **or** the saturation criterion (3.5, Prechelt GL/patience) does, whichever fires first — model §5.5f, buildplan §3 row E | a supplied target is itself an operator input, and §0.0 admits none |
| **D3** | is the communication budget a constraint? | **A tiebreak.** Minimise bytes *subject to* time-first | spending `B` and raising `P` cut time and bytes together; `K` is free in *total* bytes while `K ≤ n_req` (K-C) |
| **D4** | which knobs may the operator still supply? | **Model, PEFT scheme and `p` only.** `P` is **sensed** by hill-climbing `P/τ(P)` | the strongest autonomy claim the evidence supports. `p` stays operator-owned because it is measurably **inert** |

> **What D4 costs.** It promotes **P-1** from an ablation to a **build prerequisite**, and it requires the
> trainer to accept a **mid-run `P` change**, which nothing does today.

**Two further decisions, settled 2026-08-15 by T5's replay** (`expt_scripts/replay_landing_law.py`):

| # | question | **decision** | why |
|---|---|---|---|
| **D5** | when the periodic re-sense (3.1) moves `B_max` mid-run, does the landing target `T_res` move with it? | **No — `T_res` is a rate, never run state.** `ρ*_t = min(ρ_max, √(2·(B_max_t − B_t)/T_res))` with `T_res` constant, so `B → B_max` as `B_max(1 − e^{−t/T_res})` | `B_max` is *measured* and drifts; `T_res` is *chosen* and no probe speaks to it. Resetting per re-sense recedes the horizon forever; decrementing reinstates `T` as an input (§4.6a). Costs 3.1× the commits of a fixed-horizon law for **identical `Λ`** — cheap, because §4.6a's identity makes `Λ` schedule-free |
| **D6** | what does the run *do* when the stop fires? | **Halt** — `_work_done = True`, the path `max_runtime_s` already takes. Gated `phi_stop: off \| log_only \| halt` | freezing `θ` and continuing to evaluate is dominated: a frozen model's accuracy is fixed, so further evals cost GPU and return eval noise. `log_only` preserves the past-the-stop counterfactual that validated the rule (P4.1) without keeping a bad run alive |

**Both make the controller safe by construction rather than by backstop.** Under D5, `B` approaches
`B_max` monotonically *from below* and never crosses it — so the `Φ` stop fires only when `B_max` was
re-sensed downward or the momentum correction applies, which is what "backstop" was always supposed to
mean. Contrast `rm`, whose `Σρ²` diverges logarithmically and therefore *always* eventually needs it.

## §5.6 The cohort-width requirement is withdrawn

`K` ≥ 30 / ≥ 51 was an artifact of `gate_safety_s` = 0.4. At `s` ≈ 2.9, `ρ*` = 0.06 with `mean` needs
`N` = 19 — `I` = 2 at `K` = 10, which G-1 then ran off the cap on 100% of commits, netting **2.13×** the
control's progress per unit federated time. **Every cohort this harness runs is already wide enough.**

Two structural gaps remain: the controller (C-1), and the `h`/`p` coupling, unanalysed beyond the
FD-rescale interaction and owned by nobody.

---

# Part 6 — `cos`, now measured: the right shape, 20× too big

**What was built.** A fp32 backward pass on a fixed held-out batch, server-side, at `θ_t` before the step,
emitting `cos_ground_truth` per commit. Index alignment was unit-tested (pool = ±`g` → cos = ±1).

**What was wrong with it.** The probe sliced the first `n` rows of the raw test tensor, bypassing the
DataLoader's sampler. The test index list is built by iterating clients in partition order with **no
shuffle**, so the probe's 64-sample "global reference" was **one client's shard** — 48/4/5/7 across four
classes against an exactly balanced test set. Measured offline at a backprop-trained model,
`cos(g_batch, g_full_test)`:

| `n` | **first `n` (what shipped)** | random `n` (the fix) |
|---|---|---|
| 64 | **−0.457** | 0.480 |
| 256 | +0.493 | 0.794 |
| 1024 | +0.613 | 0.935 |
| 2048 | +0.628 | **0.975** |

**The shipped reference was anti-correlated with held-out truth** (−0.46 trained, −0.42 at init, +0.69 at
accuracy 0.38): **its sign swings with model state**, and that one fact reproduces everything the probe
appeared to show. **Everything it reported is void, including the `‖G‖/‖g‖` ratio thought to have
survived** — a 64-sample reference gradient has norm 3.1–8.0, roughly that of an 8-sample *client*
gradient, so the ratio came out at 1.0 for the wrong reason (P8.1).

**Fixed and re-run.** Fixed-seed shuffled batch over the whole test set, default size 1024, plus a
preflight that fails the launch on a class-skewed reference; the logged dominant-class share was 0.27–0.29
against a balanced 0.25 on every arm.

## §6.1 The result: the shape is right, the constant is not

`D ≡ cos_measured / cos_predicted` = **0.050**. The formula is 20× optimistic. But:

> **`D` is invariant to the combination rule, to `p`, and to `N`. MEASURED.** 0.0506 at (`select`,
> `p` = 450k) versus 0.0501 at (`mean`, `p` = 118k) — **1% apart across a 3.57× swing in the prediction**
> — and 0.0506 at `N` = 20 versus 0.0454 at `N` = 80. Pooling more uploads does not buy any of it back.

So `cos ∝ √(G_rule·N/p)` is **confirmed as a scaling law and refuted as a magnitude**, which is exactly
why `Λ` could order 21 arms while being 20× wrong.

## §6.2 Where the factor of 20 comes from, and it is not the probes

```
L = (||G|| / ||g||) / ( b * sqrt(p/N) )      length excess   -- measured 8.5 .. 9.5
S = cos * (||G|| / ||g||) / a                shadow deficit  -- measured 0.48
D = S / L                                                    -- measured 0.050
```

**`L` ≈ 9 is a data-sampling fact, not an estimator fact.** Because `v` is a raw Gaussian draw,
`d = ⟨g_k,v⟩` has `rms|d| = ‖g_k‖` *identically* — so `rms|d|` measures the norm of the client's
**8-sample** gradient. Measured: `rms|d|` = 3.2–3.4 against `‖g_heldout(1024)‖` = 0.27–0.34, a ratio of
**10.0–10.9**, which is `L` within its scatter. **Nine tenths of every upload's length is data noise the
pooling was never sized for.**

**`S` ≈ 0.48 is the other half:** the pooled `K` = 10 × 8-sample gradient carries only half of the
held-out gradient's shadow. Zero-mean sampling noise alone predicts `S` = 1, so the missing half is *bias*
— Dirichlet-skewed client draws and train-vs-held-out disagreement, which this probe cannot separate.

> **The correction is entirely on the data side.** Not the probe direction, not the combination rule, not
> `p`, not the step. **`D` is the price of the estimator being unbiased for the wrong gradient.**

## §6.3 `D` is a constant of the setting. **MEASURED, and it is a negative result.**

D-1 swept bin size `B` ∈ {8,32,128,512} × distinct bins `M` ∈ {1,10,30} on the real model and the real
Dirichlet partition, against both references, 10 reps.

> **`L` is flat: 9.47–10.85 over 64× in `B` and 30× in `M`**, where `L ∝ 1/√B` predicts an **8× fall.**
> `S` is flat in `M` too. **Bigger bins and more bins do not recover `D`.**

So `D` is not a pooling deficiency and there is nothing to pool away. **That is the benign outcome**: a
constant is absorbed into `s`, which is why `ρ ≤ 2.9·cos` works. The `1/D²` in `N_req` never had to be
paid — it was double-counting a factor `s` already carries.

> **The invariance is established *at one training state*, and that qualifier was invisible until now.**
> The sweep ran on the rig at init, and every arm-side `D` behind it came from a run cut inside ~75
> commits. The two arms that trained to peak read `D` = **0.148 ± 0.020** and **0.104 ± 0.017** against
> their 0.045–0.051. Bin size and count are still refuted as levers; what is open is whether the
> *constant* they are flat against belongs to the **setting** or to the **checkpoint** — D-2.
>
> **The second arm rules out the obvious answer.** Pooled over both, `D` is **non-monotone in accuracy**
> and its *highest*-accuracy bin reads its *lowest* `D` (0.092 ± 0.022), while pre-turn/post-turn reads
> 0.103 ± 0.021 → 0.163 ± 0.025. The elevation tracks `Φ` — the model coming apart — not "having
> learned". The near-init windows of the two `const` arms (0.079 / 0.091) leave **`const` vs `rm` as the
> leading confound**. It will not be settled on an arm: single-commit `cos` has SNR ≈ 1, with 16–18% of
> fires returning a *negative* cosine.

**The rig half-passes its gate, and the residual is H-S.** It reproduces `L` (10.2 against 8.5–9.5) but
returns `S` = 1.68 where the arms read 0.48 — and its own `S` is arithmetically clean
(`‖g_train‖/‖g_test‖` = 2.01 × `cos(g_train,g_test)` = 0.80 → 1.61). **So a 3.5× shadow loss lives in the
FL pipeline, not in the data.** Prime suspect is the FD chord: `h‖v‖` = 6.71 against `‖θ_tr‖` = 6.75, so
each probe steps roughly a full parameter-norm and `d` is a chord-averaged slope rather than `⟨g,v⟩`. That
attenuates alignment while leaving `d` Gaussian — which is why P3's distributional check did not catch it,
and why §1.3's ruling-out of the FD (as a cause of *divergence*) does not apply.

> **When an instrument's input is wrong, every number it produced is void — including the ones that
> still look reasonable.** One `Counter(labels[:64])` would have caught this.

---

# Part 7 — What inflation actually does

The inflation is a sum of steps in random directions, so it can be **injected** rather than waited for:
train to a realistic peak, add Gaussian noise to the trainable slice scaled so `‖θ_tr‖` grows by `Φ`, read
accuracy back. **MEASURED on the offline rig.**

## §7.1 The budget is relative, not absolute (Q1 — settled)

The accuracy knee sits at the **same `Φ`** in two models whose norms differ 1.77× (`rf`=16 vs `rf`=64):
both hold near 0.9 up to `Φ` ≈ 2.5, break at 3.0, and are destroyed by 3.6. Absolute norms at the knee
differ by exactly the ratio of base norms. **`B_max` transfers across `p`**, as the ratio principle
predicted.

> The rig's threshold (`Φ` ≈ 3.0) is lower than the real arms' 3.6–4.2 because it dumps noise on a model
> that never adapted while real training re-fits continuously. That bias applies to both `p` equally, so
> it cancels in the comparison — **take the *relative* verdict from the rig and the *number* from the arm
> ledger.**

**It does NOT transfer across task (B-1, MEASURED 2026-08-13, `probe_inflation_damage.py`, rf=16,
agnews/yahoo/yelp-p, mode=noise, reps≥3).** Reading the knee on chance-normalized accuracy
(`(acc−1/K)/(base−1/K)`, since post-collapse every dataset floors at its own chance level `1/K` — 0.24/4,
0.11/10, 0.51/2, confirming the collapse mechanism itself is identical, just the floor differs):

| dataset | classes `K` | normalized knee (`Φ`) |
|---|---|---|
| agnews | 4 | ~3.0–3.3 |
| yelp-p | 2 | ~2.1–2.3 |
| yahoo | 10 | ~2.0–2.3 |

**Neither invariant nor monotone in `K`.** The pre-registered prediction (§P5.1) was monotone-in-classes
(yelp-p latest, agnews middle, yahoo earliest); what landed instead is agnews alone at the late end, with
yelp-p and yahoo — opposite ends of the class-count range — landing at nearly the *same*, earlier knee.
Yahoo's low base accuracy (0.73 at 3 epochs) is not an undertraining artifact: a 9-epoch confirmatory run
peaked at epoch 2 (0.732) and *degraded* with more training (0.692 at epoch 8, overfitting the 30k-example
proxy set) while collapsing if anything earlier (Φ=2.0 already at 0.084 vs the 3-epoch run's 0.485) — the
early knee is robust to the training-budget confound, not an artifact of it. `~0.73` also matches the
DistilBERT/yahoo accuracy the FwdLLM paper itself reports (peak reported ~0.76) — this rig's base model
is in the right ballpark, not a broken one. **On the B-1 decision table
(P5.1), this is the "erratic" outcome: the online injection probe (3.1) is not optional infrastructure,
it is now the load-bearing path** — a fixed or `num_labels`-derived `B_max` would misprice the budget on
at least two of three datasets tested. **Replicated 2026-08-13: 2 independent training runs per dataset,
all agreeing tightly.** agnews knee 3.0–3.5 both runs (normalized 0.732→0.003 / 0.752→0.021 at Φ=3.0→3.5);
yelp-p crosses 0.5-normalized at Φ=2.0–2.5 both runs (0.567→0.405 / 0.517→0.409); yahoo agrees across a
3-epoch and a 9-epoch run. Below this doc's 10-arm bar (P4.1) but no longer single-run.

## §7.2 The norm is mostly a symptom, and decay will not rescue it (Q2 — settled)

Renormalizing an inflated model back to its original norm is exactly what perfect decay leaves behind:

| `Φ` | noise (norm grows) | noise + renorm (norm pinned) | recovered |
|---|---|---|---|
| 2.5 | 0.856 | 0.803 | — |
| 3.0 | 0.608 | 0.685 | +0.08 |
| 3.6 | 0.262 | 0.403 | +0.14 |

**~20–25% of the damage is magnitude; ~75–80% is the junk:signal ratio, which decay cannot touch.** In the
renormalized rows `logit_norm` falls and entropy rises to ≈ `ln 4` — the signal has been diluted below the
noise. This demotes the weight-decay arm to an optional confirmation.

**Two controls separate the mechanisms, and only one matches reality.** Isotropic *noise* reproduces the
observed collapse signature (`top_class_share` → 1.0 at unremarkable `logit_norm`, §2.7); pure *scale*
growth does not (`logit_norm` explodes, entropy → 0); *coherent* displacement destroys accuracy at
`Φ` = 1.2 already. **Isotropic junk accumulation is the right model.**

## §7.3 The cost of going backward-free

Backprop on the same model, `p` and data reaches **0.90** where the best forward-gradient arm ever
recorded is **0.876** (`145729`). That **2.4-point** gap is the measured price at this scale — down from
3.5 points, bought by pooling 4× harder per commit and paid for in a peak the arm then loses.

*Caveat on all of Part 7:* injection jumps to the endpoint instead of walking there, and omits the
learning happening alongside. It settles the comparisons; it does not replace a trajectory.

## §7.4 What is standard, what is ours, and what to claim

**None of the ingredients are ours:** `1/√p` probe overlap (classical ZO); signal-linear / noise-`√T` (SGD
noise ball); `Σρ = ∞`, `Σρ² < ∞` (Robbins–Monro); step relative to `‖θ‖` (trust-region, LARS/LAMB); a pool
beyond which pooling buys nothing (critical batch size). **Ours is the packaging** — collapsing them into
an inequality between quantities the server already logs, turning an asymptotic rate statement into an
online control law. Do not write it up as a new theorem.

| claim | verdict |
|---|---|
| **C1 · `\|JVP\|` probe selection** | **Refuted as written** (`b²/a` = 1). The **combination rule** is the contribution, and §4.6 makes it a *speed* result: **3.35× less wall clock** at zero extra cost |
| **C2 · async aggregation / gate** | **A measured win, stronger than throughput.** Holding `s` constant is the **time-optimality condition** (§4.6), not merely a throughput heuristic. Now carried by a controller that beats its hand-set control on three datasets. What C2 is *not* is a stability mechanism |
| **C3 · aggregation weighting (ω)** | **Magnitude half: park** (ω spans 0.70–0.87 against a ≥10× gap). **Freshness half: on the critical path** — staleness is the only cost of running the cohort wide, and `D(·)` has never been measured against `C` or `K` |
| **S1 · server momentum** | **Refuted.** `√x` progress for `x` budget — a re-parameterisation of `ρ` |
| **S2 · variance-gate recalibration** | **Superseded** — replace the loop, do not re-tune the threshold |
| **S3 · aggregation-rate tempering** | **Subsumed** — under trust-ratio `ρ` = `ρ*_t` regardless of ω |
| **Systems: flat memory, inference-only operators** | **Untouched, and the strongest part of the paper.** The `cos` probe's backward pass is an audit flag, off by default |

**Net effect: C1 shrinks** to a compute-budget claim only averaging redeems, **C2 grows** and relocates
from "async" to "the controller async makes possible", **C3 splits** into a parked half and a
now-load-bearing half. Systems contributions are unaffected.

**Four things the work added that were not on the claim list:** *(1)* **a free-lever test subsuming four
dead ends** — a lever is free iff it raises `cos` without correlating steps and without shrinking
`‖θ_tr‖`; *(2)* **scale invariance as a design principle** (§5.3) and its configuration-level twin, the
**autonomy requirement**, cleanest demonstration being that under trust-ratio heterogeneity moves `ρ` by
*zero to six significant figures* while still moving `var` by 1.6×; *(3)* **the time law** (§4.6);
*(4)* **`D`** — the binding constraint is the gradient of the *bin*, not the variance of the *probe*,
which inverts the "spend compute on probes" instinct.

---

# Part 8 — Generality beyond FL

*Speculative relative to Parts 1–7, and **frozen** (R8): revisit only when the open questions close.*

**Nothing in the model is federated.** FL enters only in **how `n` decomposes and what each factor
costs**: single-device and centralized ZO are `P` × accumulation; data-parallel adds workers;
**federated is `P` × `K` × `I`**, where `I` costs round trips and `K` costs staleness. `D` is not
federated either — any setting taking directional derivatives on minibatches pays it; only full-batch ZO
has `D` = 1.

**A falsifiable prediction against centralized ZO.** At `P` = 1 with no `K` or `I`, `cos` = `√(1/p)`, so
the stable relative step is ~10⁻³ and the method needs ~`p` steps for coherent progress — matching the
very long step counts such methods report. Check published curves before claiming it.

**What must be checked before generalizing.** Each row is an assumption, not a result:

| # | assumption | why it might not survive |
|---|---|---|
| 1 | **steps orthogonal to `θ`**, which makes `B` exact | measured only at `p` ≥ 118k on adapter+head slices. **Check the orthogonality ratio first on any new model — one line of replay** |
| 2 | **`‖θ_tr‖ ∝ √p`**, which makes `p` inert | holds for adapter-style init. A PEFT scheme initialising otherwise makes `p` live again — **the assumption most likely to break** |
| 3 | **`B_max`** | settled as *relative* (§7.1), value calibrated on one model family. B-1, and the **only** constant on the operating path |
| 4 | `Λ`'s calibration and **`D` ≈ 0.05–0.15** | fitted here — but both are **forecast-only**, so neither blocks operation |
| 5 | **isotropic probes / `1/√p` overlap** | exact given Gaussian `v` over the whole slice. Structured or block probes break it deliberately |
| 6 | **`h` pinned by fp16 cancellation** | a precision property, not a math one; bf16/fp32 move the window |

**Larger models and datacenter.** Where backprop is available the argument narrows to "2 forward passes
beat 1 fwd + 1 bwd, with no activation memory", and `cos ∝ 1/√p` makes it *worse* at scale. What would
help is the one thing FL cannot buy: **large bins**, where `D` lives. **MoE is structurally interesting**
— `p_effective` is per-token active params, i.e. block-coordinate probing for free. Unanalysed.

*The general-ML contributions, ranked, are §7.4's closing list — not repeated here.*

---

## Appendix — how to maintain these two documents

| # | rule |
|---|---|
| **R1** | **Edit in place.** Rewrite the row or paragraph that changed. No changelogs, no dated "update:" notes. Git holds history. |
| **R2** | **One number, one home.** Model and mechanism here; run results, flags and commands in the practice doc. If a number is restated across the two, delete the copy and cite. |
| **R3** | **Evidence, model and recommendation move together.** A new finding updates the model *and* re-checks [P2](fl_fwd_ft_practice.md#p2--the-shipped-stack) in the same edit. |
| **R4** | **Tag claims** `MEASURED` (telemetry + run id) · `DERIVED` (arithmetic on measured inputs) · `ANALYSIS` (rests on a stated assumption) · `HYPOTHESIS`. Untagged prose is not evidence. |
| **R5** | **Dead ends are permanent.** [P6](fl_fwd_ft_practice.md#p6--dead-ends--do-not-retry) is append-only and is the first thing to read before proposing anything. |
| **R6** | **Prediction before run.** An arm enters the queue with a number *and* a sinking condition written down first. |
| **R7** | **Cheapest instrument first** ([P7](fl_fwd_ft_practice.md#p7--the-instrument-ladder)). A full run *confirms* a hypothesis; it never explores one. |
| **R8** | **§8 is frozen** until the open questions close. |
| **R9** | **Prose budget.** A paragraph that does not change a decision is deleted. Accuracy is a *result*, never the stability evidence. |
