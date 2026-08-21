# FluxTune: fine-tuning without backprop, and the controller that makes it hold

*Read this top to bottom. Every number is from a logged run; the run id is given so it can be checked.
The companion documents hold the full derivations and the complete evidence — this one holds the argument.*

**Three systems appear throughout.** **FwdLLM** is the prior work whose failure §2 explains.
**FluxTune-v2** is the previous version of our system — right step rule, but a step *size* hand-searched
on one dataset. **FluxTune** is this work: the same stack with the step size set from a budget it measures
for itself. The target is a **backprop reference** — exact gradients on the same rig.

---

## 1. What we are doing, and why it is hard

We fine-tune a language model across many clients **without ever running a backward pass**. Clients can
only evaluate the model forward. This buys flat activation memory and lets training run on an
inference-only runtime.

To get a gradient without autograd, a client picks a **random direction** `v`, nudges the weights both
ways, and measures how the loss changed:

```
d  =  [ L(θ + h·v) − L(θ − h·v) ] / (2h)     ≈  how much the loss slopes along v
u  =  d · v                                   ≈  a one-sample guess at the gradient
```

Averaged over many random directions this points the right way. The catch is **how many is "many."** The
trainable slice here has `p ≈ 450,000` dimensions, and a random direction in 450,000 dimensions is
almost perpendicular to the one you want. Pooling `N` such guesses recovers alignment only as a square
root:

```
cos( our step , true gradient )  ≈  sqrt( N · P / p )     — with N ≈ 200, P = 10, this is ≈ 0.07
```

> **The one fact everything follows from: each step is ~7% signal and ~93% noise.**
> Not "noisy but roughly right" — *mostly sideways*, every single step.

**Symbols.** Only four are needed.

| | |
|---|---|
| `ρ` | **step size relative to the model**: `‖Δθ‖ / ‖θ‖`. `ρ = 0.05` means "move 5% of your own length" |
| `Φ` | **how much the model has grown**: `‖θ_now‖ / ‖θ_start‖` |
| `B` | **budget spent** — defined in §4, and `Φ = e^B` |
| `Λ` | **progress banked** — the useful part of all that movement |

---

## 2. The failure: noise has nowhere to go, so it piles up

Because each step is almost perpendicular to where the model already is, it cannot shorten the model — it
can only lengthen it. Pythagoras, exactly:

```
‖θ + Δθ‖²  =  ‖θ‖²  +  ‖Δθ‖²          (the cross-term is zero; measured at 1.000 ± 0.005, every arm)
```

So **the weights grow, every commit, forever.** The 93% that was noise does not cancel out over time — it
accumulates as *length*.

That gives `Φ` its meaning, and it is the intuition worth keeping:

> **`Φ` is the reciprocal of how much of the model is still the part that earned its accuracy.**
> At `Φ = 2.7`, roughly **1/2.7 ≈ 37%** of the model's length is signal and the rest is accumulated
> junk. The classifier head is then sitting ~68° away from the direction that used to work, and it fails.

**And in the old system this ran away.** The step was `Δθ ∝ η·G`, so `ρ` was an *outcome*, not something
anyone chose. A longer weight vector produces a larger gradient (measured: `‖g‖` tracks `‖θ‖` with
exponent ≈0.9), a larger gradient produces a longer step, and a longer step adds more length:

> noise inflates the model → the gradient grows → the step grows → more noise

The loop is closed. It is present at commit 1 and becomes visible around commit 150 — the steps never
become *more wrongly aimed*, they just become *bigger*, applied to a model that keeps getting longer.
Left alone it ends the same way every time: an arm reaches 0.85 and then **hands it all back**, landing at
its dataset's chance level.

![Noise compounds](figs/fig1_noise_compounds.png)

**And when it grows too far, the run gives everything back.** Of the 22 arms on record that actually
learned, **every one below `Φ` = 3.63 held its peak** — worst loss 0.014 — and **every one above
`Φ` = 4.23 lost it**, by 0.083 to 0.604. Two arms ended at half the accuracy they had reached
(`112201`: 0.849 → 0.250; `013806`: 0.855 → 0.251). The cliff is real, it is sharp, and it is located
in a quantity computable from the step sizes alone.

### 2.1 The old system survived by accident

The prior system (FwdLLM) decides when to commit by pooling until the readings agree: `var(d) ≤ threshold`.
Working that out, `var ≈ 2·b²·‖g‖²/N`, which says two things:

1. It is really a **pool-size controller** — `var ∝ 1/N`.
2. **It carries units.** `var ∝ ‖g‖²`, and `‖g‖` grows with the model, so the variance floor drifted **36×**
   over one run — exactly the square of the 6× norm growth.

Because the ruler stretches while the threshold does not, holding `var ≤ c` forces the pool to grow like
`‖θ‖²`, which forces `ρ ∝ 1/‖θ‖`, which is a **decaying step schedule — by accident.**

> **The units bug was the only thing holding the system up.** And it is not a fix: the decay is too slow
> to prevent collapse (it defers it to ~commit 1,200–1,700), and because the threshold has units it means
> something different on every model, dataset and run length. **This is why nothing transferred.**

**The defect in one line: nothing in the pipeline is scale-invariant** — the readings, the step, and the
gate all inflate together, so no quantity anywhere can be compared against a fixed number.

---

## 3. The fix, in one idea

**Stop measuring anything in units that stretch.** Three replacements make every quantity a pure ratio:

| what was wrong | replacement | what it buys |
|---|---|---|
| step size was an *outcome* of the gradient magnitude | **trust-ratio step**: `θ ← θ − ρ·‖θ‖·G/‖G‖` — we *set* `ρ` directly | `ρ` becomes a knob. Client heterogeneity now moves it by *zero to six decimal places* |
| commit gate compared a quantity with units to a constant | **`n_target` gate**: pool until `N ≥ p·(ρ/s)²/(P)` | the same pool-size controller, stated dimensionlessly |
| probes were *selected* by `|d|` | **average all of them** | more signal per commit, for free |

That removes the runaway. It does not answer the remaining question, which is the whole rest of the
problem: **how big should `ρ` be, and when should we stop?**

---

## 4. Two numbers describe any run

With steps perpendicular to `θ`, the trajectory is an exact recursion `‖θ_{t+1}‖² = ‖θ_t‖²(1+ρ_t²)`.
Summing it gives one number for what a run **spends** and one for what it **earns**:

```
SPEND     B  =  ½ · Σ_t ln( 1 + ρ_t² )        and       Φ = e^B
EARN      Λ  =  Σ_t  ρ_t · cos_t              ( the ~7% of each step that pointed the right way )
```

Both are dimensionless, both are exact at any point in the run, and **`B` is computable from the step
sizes alone** — no gradients, no accuracy, no model-specific constant.

Two empirical facts turn this into a control problem:

**(a) Accuracy rises with `Λ`.** Over 21 runs sorted by `Λ`, peak accuracy climbs monotonically from 0.377
to 0.876. `Λ` is how far you travelled *usefully*, measured in multiples of your own length.

**(b) Whether you keep what you learned is decided by `Φ` alone.** Two readings of the same portfolio,
and it is worth keeping them apart:

- **where accuracy peaks** — every run that learned, turned over, and still had time left peaked at
  **`Φ` = 2.41–3.11** (mean 2.71), across two step rules, three model sizes, α from 0.1 to 1, and run
  lengths from 177 to 1,364 commits;
- **where it is lost** — below **`Φ` = 3.63** every such run held its peak to within 0.014; above
  **`Φ` = 4.23** not one did (Figure 1).

Neither number was fitted to anything: `Φ` is computed from the step sizes, and the peak location was
never used to choose a single constant.

![The budget law is exact](figs/fig2_budget_law.png)

**`B` is not a model — it is an identity that the runs obey.** Predicting `Φ` from the step sizes alone,
across every arm on record, the median miss is **0.09%**. The one visible exception is the pair of
server-momentum arms: correlated steps inflate faster, exactly as the law says they must
(`Φ = exp(((1+β)/(1−β))·B)`, checked at β = 0.5 and 0.75).

![Accuracy rises with progress banked](figs/fig3_accuracy_vs_progress.png)

The relationship was **pre-registered before the three-dataset runs**: if yahoo reached 0.6–0.7 by
`Λ ≈ 1.0`, the curve transfers across task; if it stalled near 0.35 with `Λ > 1.0`, it does not.
**yahoo read 0.657 at `Λ` = 0.994.** agnews reads 0.868 at `Λ` = 1.001.

### The problem, restated

> There is a budget. Spending it buys accuracy; overspending destroys what you bought.
> **How fast do we spend, and when do we stop?**

And there is a clean answer to the first half. Under a gate that keeps `ρ ≤ s·cos`, the exchange rate is
fixed:

```
Λ  =  2B / s          — an identity, not a fit  (verified to −0.3% out of sample)
```

> **This is the load-bearing simplification.** It says **the schedule cannot buy accuracy.** Two step
> schedules that spend the same budget earn the same accuracy and differ only in how many commits they
> take. So there is *no schedule to tune* — we are free to pick one for stability rather than for yield,
> and the only real lever on accuracy-per-budget is `s`.

---

## 5. FluxTune

**Spending rule (law C).** Aim the step at whatever budget is left:

```
ρ*_t  =  sqrt( 2 · (B_max − B_t) / T_res )
```

`T_res` is a **rate — how many commits of control resolution we want — not a deadline.** That distinction
is what removes the run length from the inputs: nobody has to say how long training will take. The step
naturally shrinks as the budget is consumed, and approaches `B_max` from below.

**Where `B_max` comes from — it is measured, not supplied.** Add random noise to the weights, scaled so the
model grows by exactly `Φ`; read held-out accuracy back; find the `Φ` at which accuracy falls apart. Six
forward passes on a copy, no gradients. Repeat every 150 commits.

Because the probe measures headroom *from where the model stands now*, while `B` accumulates from the
start:

```
B_max  =  B_now  +  ln Φ_knee
```

**What an operator supplies: no learning knob at all.** No learning rate, no variance threshold, no cohort
width, no run length, no target accuracy. Only a description of the dataset and a compute budget.

---

## 6. It works: FluxTune beats its hand-tuned predecessor on every dataset

Three datasets, one model (DistilBERT + adapters), 100 clients, non-IID. Three systems in the comparison,
and it is worth being precise about which is which:

| name | what it is |
|---|---|
| **FwdLLM** | the prior system — variance-gate commits, raw SGD step. §2 is about why it falls over |
| **FluxTune-v2** | the previous FluxTune: trust-ratio step and dimensionless gate, but a **static step size `ρ` = 0.06 hand-searched on agnews**, decayed on a fixed schedule |
| **FluxTune** | this work: the same stack with `ρ` set by law C from a **sensed** budget. No `ρ`, no `B_max`, no run length supplied |
| **backprop reference** | exact gradients on the same rig — the target |

v2 is run unchanged on all three datasets, so the experiment asks exactly: **does a hand-tuned constant
transfer to a new task, and does a sensed one?**

| | FluxTune | FluxTune-v2 (full budget) | compute to match v2's *best ever* | backprop reference |
|---|---|---|---|---|
| agnews | **0.868** | 0.843 | **5.5× less** | 0.850 — **cleared** |
| yahoo | **0.657** | 0.428 | **6.5× less** | 0.734 |
| yelp-p | **0.814** | 0.728 | **8.5× less** | 0.874 |

**v2 never reaches FluxTune's accuracy**, on any dataset, given its entire budget. And on agnews FluxTune
is already **past the backprop reference**.

![FluxTune versus FluxTune-v2](figs/fig4_fluxtune_vs_v2.png)

### Why it wins — one column explains it

| budget actually spent | agnews | yahoo | yelp-p |
|---|---|---|---|
| FluxTune-v2, after its **full** budget | 0.107 | 0.119 | 0.111 |
| FluxTune | 0.697 | 0.690 | 0.970 |

A fixed `ρ` = 0.06 spends **≈0.11 of budget regardless of the dataset** — that is what a hand-set constant
does, by definition. FluxTune spends 6–9× more of the same budget in the same wall clock.

And the *penalty* for under-spending is set by the task, not by the guess: agnews saturates early, so v2
still reaches 0.843; yahoo needs far more budget and v2 stalls at 0.428.

> **The claim in one sentence: the right amount of budget is not a constant, so it has to be sensed — and
> what it costs you to guess wrong is decided by the task you haven't seen yet.**

---

## 7. What we found broken, and it is the interesting part

FluxTune stops itself. **But it throttles its own step on the way there, and we now know exactly why.**

**The probe was not the problem. The averaging was.** Every 150 commits the probe reports remaining
headroom. Across all 19 firings on all three datasets that reading **shows no downward trend** — it does
not shrink as budget is spent:

| yelp-p, headroom reported by the probe | fire 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| **measured** `ln Φ_knee` | 0.250 | 0.595 | 0.683 | 0.415 | 0.253 | 0.236 | 0.217 | **0.246** |
| **what FluxTune used** (mean of all senses, minus `B`) | 0.250 | 0.373 | 0.378 | 0.276 | 0.184 | 0.132 | 0.100 | **0.084** |

The probe kept saying *"you have ~0.25 of road left."* The `mean` combiner turned that into *"you have
0.08 left"* — a **2.9× understatement** — and since `ρ* = √(2·headroom/T_res)`, the step was annealed to
**1.7× below what the current measurement supported.** `B` then crossed `0.95·B_max` and the run halted.

> **The run did not stop because it was out of road. It stopped because the odometer was averaged.**

![The combiner throttles the step](figs/fig5_the_combiner_throttles.png)

### 7.1 What that costs is not the same on every dataset

This is the question that decides what to fix. `Λ = 2B/s` says the schedule cannot buy accuracy — so
un-throttling `ρ*` buys **commits**, not learning. It raises accuracy only if the run was stopping while
accuracy still had slope in `B`. So: **did it?**

![Does more budget still buy accuracy?](figs/fig7_does_more_budget_help.png)

| | accuracy at the end | tail slope `dAcc/dB` | verdict |
|---|---|---|---|
| **agnews** | 0.868 vs a 0.850 reference | 0.06 | **already past it, and flat.** Fixing the throttle buys time only |
| **yahoo** | 0.657 vs 0.734 | **0.23** | **still climbing.** Needs only `ΔB` ≈ 0.33 → `Φ` ≈ 2.8. The throttle cost it *accuracy* |
| **yelp-p** | 0.814 vs 0.874 | 0.03 | **saturated.** Closing 0.060 at this slope would take `Φ` ≈ 17. Not a budget problem at all |

Three conclusions follow, and they are different from what the stopping behaviour alone suggested:

1. **Un-throttling `ρ*` makes FluxTune reach its plateau faster on every dataset** — same `Λ` for the same
   `B`, in fewer commits.
2. **On yahoo it also raises the accuracy reached**, because that curve still has slope. yahoo is the one
   dataset where the bug cost a result rather than a few hours.
3. **yelp-p's 0.060 gap is not a stopping problem and not a budget problem.** Its curve is flat. That gap
   is a property of the estimator or of adapter capacity on that task, and no controller change addresses
   it. Saying otherwise would be the easiest mistake in this document to make.

### 7.2 Two deeper problems, and a rail that is probably too tight

**(a) The probe's search range barely contains the answer.** It tests `Φ ∈ {1.5, 2, 2.5, 3, 3.5, 4}`, a
range sized from offline measurements taken before any of this. Live, **the very first point it tests is
already below the knee**, so the crossing lies to the left of the entire grid: sampling 8 of the 19
firings, **5 put the knee below the whole grid** — the knee-finder then extrapolates between a synthetic
anchor at `Φ`=1 and that single reading — and 3 put it barely inside the first interval. **The four points
at 2.5–4.0 do no work on any of them**, two thirds of a 2–3 minute probe spent where the answer is not.
Consequence: what it reports is pinned into a narrow band regardless of the model, so
`B_max = B + (roughly a constant)` **recedes as budget is spent**.

![The ruler starts past the mark](figs/fig6_ruler_starts_past_the_mark.png)

**(b) The whole "fixed budget" picture may be wrong.** The probe noises a model that **cannot re-fit**. A
training run re-fits continuously. If remaining headroom genuinely stays ~0.25 as training proceeds — which
is what the measurements show — then budget is not a tank that drains. It is closer to a **rate limit that
is continually re-earned**, and stopping when a cumulative total is reached is the wrong stopping rule.

**(c) And the rail is probably too tight — which is exactly what yahoo needs.** The shipped damage
backstop is `Φ` = 2.7. The portfolio says arms hold their peak up to `Φ` = 3.63 and only lose it past
4.23 (Figure 1). yahoo's projected landing is `Φ` ≈ 2.8 — **just past the shipped rail, and far inside the
measured cliff.** So the rail, set from the old diverging dynamics, is the second thing standing between
FluxTune and the backprop reference on yahoo.

## 8. What we are changing

| | decision | why |
|---|---|---|
| **What ends a run** | **Saturation, not budget.** Stop when smoothed held-out accuracy stops improving. `B_max` stays, but only to drive `ρ*` — never as a termination rule | the run must end because *learning* stopped, not because a running total was reached. §7.1 shows all three datasets do saturate; the rule should detect that rather than approximate it |
| **How senses combine** | **Latest, not mean** (`b_max_policy=anchor`) | §7's table. `anchor`'s standing objection — it never terminates — is void once saturation is what terminates |
| **The `Φ` ≈ 2.7 rail** | **Raise it toward the measured cliff, after testing.** One run with both stops in log-only mode passes straight through 2.7 and shows whether the head turns | 2.7 comes from the old diverging dynamics; the portfolio puts the cliff at 3.63–4.23, and yahoo's projected landing at 2.8 falls in that gap |
| **Target accuracy** | The backprop reference, unchanged — **and never an input** | stopping *at* a supplied target would make the target a knob and break the zero-input claim. The rule stops on saturation; whether saturation lands at or above the reference is then a **result**, not a setting |

**On that last row, explicitly.** The goal is not "terminate once the backprop number is reached." It is
"terminate when the model has stopped learning, and *then observe* where that lands." agnews lands above
the reference; yahoo is projected to land at it; yelp-p lands 0.060 below it and no stopping rule changes
that. Building the target into the stop would hide exactly the fact that is most worth reporting.

## 9. Honest boundaries

- **Two of three runs are below the backprop reference** (yahoo by 0.077, yelp-p by 0.060). Of those, only
  yahoo has slope left; yelp-p has saturated (§7.1), so its gap is a statement about forward-gradient
  estimation on that task, not about the controller.
- The "ends within 0.015 of its peak" check is weaker than it sounds: **a run that is still climbing
  passes it trivially**, because its peak is its last point. It shows *no divergence*, not convergence.
- **`Φ` ≈ 2.7 is not confirmed under the new dynamics.** It comes from the old, diverging regime; nothing
  on record tests it under a controlled `ρ`. §8 tests it directly.
- **The tail-slope projections in §7.1 are extrapolations**, fitted over the last 20% of each run. They
  are the basis for a pre-registered prediction, not a result.
- **No run in *this* set collapsed**, so on these six arms the stopping rule demonstrated efficiency —
  stopping early at almost no cost — rather than the damage avoidance it exists for. The damage itself is
  not in doubt: six arms in the wider portfolio lost 0.083–0.604 of accuracy past `Φ` = 4.23 (Figure 1).
- **One model throughout.** All three datasets use DistilBERT + adapters at the same `p`. The one time the
  model size really changed, the controller's two remaining constants stopped composing with the gate.
  Generality is demonstrated **across task, not across model.**

---

## 10. Summary

**The problem.** Forward-gradient steps are ~93% noise and perpendicular to the weights, so noise
accumulates as length and cannot cancel. Under a raw step rule this runs away. The previous system
survived only because its commit gate compared a quantity carrying units of `‖θ‖²` against a fixed
threshold — an accidental decay schedule that deferred collapse instead of preventing it, and that meant
something different on every task.

**The formulation.** Perpendicularity makes the trajectory exact, giving two dimensionless numbers: a
spend `B` (with `Φ = e^B`, the fraction of the model that is still signal) and an earning `Λ`. Accuracy
rises with `Λ`; retention is governed by `Φ`. Under a dimensionless gate, `Λ = 2B/s` identically — so the
schedule cannot buy accuracy, and there is nothing in it to tune.

**FluxTune.** Set the step from the budget remaining, at a *rate* rather than against a deadline, so
run length is never an input; and measure the budget on the running model with forward passes only.

**The evidence.** On three datasets FluxTune beats FluxTune-v2 — the same stack with a step size
hand-searched on agnews — by 5.5–8.5× in compute to equal accuracy, and v2 never catches it. The budget law predicts measured weight growth to
within 0.23% across all six runs.

**The open problem, and it is smaller than it looked.** FluxTune throttles its own step, because a flat
headroom measurement is averaged into a shrinking one. That costs *time* on agnews and yelp-p, both of
which have saturated — agnews above its reference. It costs *accuracy* on yahoo alone, which still has
slope and is projected to reach its reference with `ΔB` ≈ 0.33. yelp-p's remaining 0.060 is not a
controller problem at all. The next runs fix the combiner, move termination to saturation, and test
whether the `Φ` rail can move out to the measured cliff.
