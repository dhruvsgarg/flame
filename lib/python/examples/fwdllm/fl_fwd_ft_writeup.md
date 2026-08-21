# Backprop-free federated fine-tuning: why it diverged, and the control law that fixes it

*A self-contained account. No familiarity with the other documents in this directory is assumed; where a
number is quoted, the run that produced it is named so it can be checked against the logs.*

---

## 1. The setting

We fine-tune a pre-trained transformer across many clients **without ever computing a backward pass**.
Each client holds private data and can only run the model forward.

Let `θ` be the model and `θ_tr ⊂ θ` the trainable slice — LoRA-style adapters plus a classifier head, of
dimension `p`. A client estimates the gradient by **directional derivatives**: draw `P` isotropic Gaussian
probes `v_i ~ N(0, I_p)` and take a central finite difference,

```
d_i  =  [ L(θ + h·v_i) − L(θ − h·v_i) ] / (2h)   ≈  ⟨g, v_i⟩        (2 forward passes, no autograd)
u    =  (1/P) · Σ_i d_i · v_i                                       (E[⟨g,v⟩v] = g, so u is aimed at g)
```

The server pools `N = K·I` such uploads — `K` clients per commit, `I` iterations over the same data bin —
into one direction `G`, and takes a step. **Nothing but forward evaluation is ever required**, which is
the point: activation memory is flat, and the operator set is the one an inference runtime already has.

The price is estimator quality. In `p ≈ 4.5·10⁵` dimensions a single probe is almost orthogonal to the
true gradient, and pooling recovers alignment only as a square root:

```
cos(G, g)  =  D · sqrt( G_rule · N / p )                         (G_rule = P when averaging probes)
```

with `D ≈ 0.05–0.15` measured. Every commit therefore moves the model **mostly sideways**. That is the
fact the rest of this document is about.

**Notation used throughout.**

| symbol | meaning |
|---|---|
| `ρ_t = ‖Δθ_t‖ / ‖θ_t‖` | the **relative step** at commit `t` |
| `Φ = ‖θ_T‖ / ‖θ_0‖` | **norm inflation** over the run |
| `B` | **budget spent** (defined in §3) |
| `Λ` | **progress banked** (defined in §3) |
| `s` | the commit gate's safety ratio, `ρ ≤ s·cos` |

---

## 2. The problem: the previous system diverges, and its stabiliser is an accident

### 2.1 Every step is orthogonal to the model, so the norm can only grow

Because `‖θ + Δ‖² = ‖θ‖² + 2⟨θ,Δ⟩ + ‖Δ‖²`, the ratio of observed norm growth to step energy has an exact
null at **1.000** — meaning the step carries no component toward or away from where the model already
stands. Measured per 25-commit block, that ratio reads **1.000 ± 0.005 in every block of every arm**,
including arms that reach 0.86 accuracy and hold it.

This is not itself pathological: a true gradient is also nearly perpendicular to `θ` at high `p`. What it
establishes is that the norm grows by the **full** `‖Δθ‖` every commit, with no cancellation:

```
‖θ_{t+1}‖²  =  ‖θ_t‖² · ( 1 + ρ_t² )                             — an exact difference equation
```

### 2.2 Under a raw SGD step, `ρ` is an outcome, and the loop closes on itself

The step is `Δθ ∝ η·G`, so `ρ` is whatever the gradient magnitude happens to make it. And the gradient
magnitude **grows with the norm**: a backprop probe on a fixed batch shows `‖g‖` tracking `‖θ_tr‖` with
exponent ≈0.9. So:

> noise inflates the norm → the gradient grows → the absolute step grows → more noise

is a **closed, self-reinforcing loop**. Two consequences that are easy to get wrong:

- **Divergence is present at commit 1** and takes ~150 commits to become *visible*. Steps do not become
  more wrongly aimed over time; what grows is the absolute step and the norm it acts on.
- The collapse is **directional degeneracy of the head**, not logit saturation: post-collapse accuracy
  lands at each dataset's own chance level (0.25 on 4 classes, 0.10 on 10, 0.50 on 2).

### 2.3 The existing variance gate stabilises it — by having the wrong units

The prior system (FwdLLM) commits when the pooled readings agree: `var(d) ≤ threshold`, where `var` is the
per-coordinate variance between two half-means. Writing `u = d·v` with `‖v‖² ≈ p`:

```
var  =  ‖G_A − G_B‖² / (2p)   ≈   2·b²·‖g‖² / N
```

Two readings of that expression:

1. **`var ∝ 1/N` is why the gate is really an `N`-controller.** Across `K` = 20/30/50 the realised `N`
   lands in a narrow band regardless of `K` — the gate hands most of a `K` increase straight back as
   fewer iterations.
2. **`var ∝ ‖g‖²` is a units bug.** The achievable variance floor drifts **36×** over a run — exactly
   `(6.0×)²`, the square of the norm growth. Because the ruler carries units of `‖θ‖²`, holding
   `var ≤ c` forces `N ∝ ‖θ‖²`, hence `ρ ∝ 1/‖θ‖`, hence a roughly **constant absolute step**.

> **The bug is the stabiliser.** Wrong units are what turn a fixed threshold into a `ρ ∝ 1/‖θ‖` anneal —
> and since `‖θ‖ ∝ √t` under a constant absolute step, that is a Robbins–Monro schedule *by accident*.
>
> **It is not a fix.** `Σρ_t² = Σc/t` diverges logarithmically, so collapse is **deferred, not
> prevented** (extrapolated to commit 1,200–1,700), and the threshold hard-codes one trajectory: it is a
> constant with units, so it means something different on every model, dataset and run length.

### 2.4 The defect, stated once

**Nothing in the pipeline is scale-invariant.** The estimator (`|d|` tracks `‖θ‖`), the step
(`‖Δθ‖ ∝ |d|`) and the gate (`var ∝ |d|²`) all inflate together, so **no quantity anywhere can be
meaningfully compared against a fixed constant.**

The three violations are not independent: the third partly cancels the first two. So the fix cannot merely
remove the bug — it must **supply that anneal deliberately**, from a quantity that has no units.

---

## 3. Mathematical formulation: two conserved quantities

Integrating the difference equation of §2.1 gives, with **no free parameter**:

```
BUDGET     B  =  ½ · Σ_t ln( 1 + ρ_t² )            and      Φ  =  e^B
PROGRESS   Λ  =  Σ_t ρ_t · cos_t                   with     cos_t = D·sqrt(G_rule·N_t/p)
```

`B` is what a trajectory **spends**; `Λ` is what it **banks**. Both are dimensionless, both are exact at
any horizon, and both are computable per commit from `ρ` alone.

**The budget law is the strongest empirical statement here.** It holds across **21 arms** spanning `ρ`
0.0002–0.22, `N` 10–200, α 0.1–1, both combination rules, both step rules, both gates, `p` 118k–450k, and
177–1,273 commits. It contains **no `cos`, no `N`, no rule, no α and no `p`**: `‖θ_T‖` is a function of
the `ρ` trajectory and nothing else. (It requires steps to be *independent*; server momentum correlates
them and enters multiplicatively as `Φ = exp(((1+β)/(1−β))·B)`, verified at β = 0/0.5/0.75.)

**Two facts turn these into a control problem.**

**(a) Accuracy is monotone in `Λ`.** Over 21 arms sorted by `Λ`, peak accuracy rises monotonically from
0.377 to 0.876 with no exception outside replicate noise. Read `Λ` as *accumulated aligned displacement in
units of the model's own length*.

**(b) Whether an arm keeps what it learned is decided by `Φ`, and by nothing else.** Every arm that
learned, turned, and still had run left peaks at **`Φ` = 2.41–3.11, mean 2.71** — across both combination
rules, both step rules, `p` = 118k/229k/450k, α = 0.1 and 1, `T` from 177 to 1,364 commits, and `ρ` from
0.06 to 0.20.

That constant is **not fitted**: `Φ` is computed from `ρ` alone and the peak location was never used to
choose anything. It has a geometric reading. Steps are orthogonal to `θ`, so **`Φ` is the reciprocal of
the un-junked fraction of `‖θ_tr‖`**; `Φ ≈ 2.7` says a classifier head fails once it sits more than ~68°
off the direction that earned its accuracy. That is a statement about *classifier heads*, not about a
particular dataset.

### 3.1 The problem, restated

Define `B_max := ln Φ_peak`. Then the entire method reduces to **one scalar decision**:

> There is a finite budget `B_max`. Spending it buys progress `Λ`; overspending destroys what was bought.
> **How fast should the budget be spent, and when should the run stop?**

And the efficiency of the exchange is fixed by construction:

```
Λ / B  =  2·cos / ρ
```

so under a gate that holds `ρ ≤ s·cos` with `s` constant, `cos = ρ/s` and

```
Λ  =  Σ ρ_t² / s  =  2B / s                                       — an IDENTITY, not an approximation
```

Verified out of sample: **−0.3%** on the two arms that pin `s`, and **+21.5 to +23.3%** on three arms
whose `s` drifts — the same arms, in the same order, as the independent error table.

**This identity is what makes the method tunable-free.** It says the `ρ` *schedule* is `Λ`-neutral at
fixed `B`: two schedules that spend the same budget bank the same learning and differ only in how many
commits they take. So the schedule can be chosen for **well-posedness rather than for yield** — there is
no schedule to tune, because no schedule wins.

---

## 4. The solution

Four replacements, each removing one violation of §2.4, plus one control law.

| # | what | replaces | why it is scale-free |
|---|---|---|---|
| 1 | **Average all probes**, never select on `|d|` | `|d|`-based probe selection | selection is stability-neutral by construction; averaging raises `G_rule` from `E[v‖²]`≈3.0 to `P`=10 for free |
| 2 | **Trust-ratio step** `θ_tr ← θ_tr − ρ*_t·‖θ_tr‖·G/‖G‖` | `Δθ ∝ η·G` | makes `ρ` a **knob instead of an outcome**. Enacts to 8.7e-5; client heterogeneity moves `ρ` by *zero to six significant figures* |
| 3 | **`n_target` commit gate**: pool until `N ≥ N_req = p(ρ/s)²/G_rule` | `var(d) ≤ threshold` | the same `N`-controller the variance gate secretly was, but stated in a dimensionless quantity |
| 4 | **Law C landing** (below) | a hand-chosen decay exponent | derives the schedule from the budget, with no horizon input |
| 5 | **Sense `B_max`** by noise injection (below) | shipping `Φ_peak` as a constant | the value is not derivable in advance; the mechanism is |

### 4.1 Law C — spending the budget without being told the horizon

```
ρ*_t   =  min( ρ_max , sqrt( 2·(B_max − B_t) / T_res ) )
ρ_max  =  s·sqrt( max_iter·K·G_rule / p )                        (gate reachability, mechanical)
stop when   B_t  ≥  f · B_max ,   f = 0.95
```

`T_res` is a **rate — commits of control resolution — and is never decremented.** It is not a deadline,
and this is the crux of needing no horizon input. Two alternatives were considered and rejected:

- **Law A** (`ρ*` set once from `B_max` and `T`) makes `ρ*` constant under perfect tracking and smuggles
  the horizon `T` back in as an operator input.
- **Law B** (recompute against a receding horizon) never terminates.
- **Law C** approaches `B_max` monotonically **from below**, so the stop is a genuine backstop rather than
  a race with the sensor.

Because of the `Λ = 2B/s` identity, choosing `T_res` costs no learning — only commits. It is therefore
free to be set to a round default (300) rather than searched.

### 4.2 Sensing `B_max` — forward passes only

`B_max` cannot be shipped as a constant, so it is **measured on the model being trained**. Add isotropic
Gaussian noise to `θ_tr`, scaled so `‖θ_tr‖` grows by exactly `Φ`; read held-out accuracy back at several
`Φ`; take the knee on **chance-normalised** accuracy (post-collapse every dataset floors at its own `1/K`,
so a raw threshold would mean three different things on three datasets).

```
B_max  =  B  +  ln Φ_knee
```

The `+B` matters: the probe measures **remaining headroom from `θ_t`**, while `B` accumulates from `θ_0`.
Anchoring this way makes `B_max > B` by construction, so a re-sense can only move the landing point, never
retroactively stop the run.

The probe costs ~6 forward-only evaluations on a copy, needs no gradients, and re-fires every 150 commits;
successive senses combine by `mean`.

### 4.3 What an operator supplies

Nothing that is a learning knob:

| supplied by hand | derived mechanically | universal constant |
|---|---|---|
| a description of the dataset (paths, `num_labels`, sequence length) | `p`, bin count, `num_labels` from the data | `probe_combine = mean`, `server_step_rule = trust_ratio`, `commit_gate = n_target` |
| a compute budget | `ρ*_t` from law C; `ρ_max` from gate reachability; `N_req` closed-form | `s` = 1.5, `T_res` = 300, `f` = 0.95, `P` = 10 |
| an evaluation-cost cap | **`B_max` — sensed** | `B_max` *prior* `ln 2`, replaced by the first sense |

No learning rate, no variance threshold, no cohort width, no horizon, no target accuracy, no safety factor.

---

## 5. Results

### 5.1 The experiment

Three datasets differing in class count, sequence length and shard size; one model (DistilBERT +
adapters, `p` ≈ 4.5·10⁵); 100 simulated clients, Dirichlet α = 1; identical configuration everywhere.

| | agnews | yahoo | yelp-p |
|---|---|---|---|
| classes | 4 | 10 | 2 |
| data bins per round | 150 | 1,750 | 650 |
| sequence length | 192 | 256 | 256 |
| backprop reference (10 clients, 3 epochs, exact gradients) | 0.850 | 0.734 | 0.874 |

**Controller** = the stack of §4 with law C and a sensed `B_max`, given **no** `ρ*` and **no** `B_max`.
**Control** = the same stack with the *hand-set* configuration that was tuned on agnews — `ρ*` = 0.06 with
a Robbins–Monro decay — run on all three datasets unchanged. The comparison therefore asks exactly:
**does a hand-tuned constant transfer to a new task, and does a sensed one?**

The two arms are distinguishable in one log line each:

```
controller   [ServerStep] trust_ratio rho_star=0.06 schedule=landing exp=0.25
             [CommitGate] n_target s=1.5 p=448802 probe_combine=mean P=10 G_rule=10.0 rho_ref=annealed
             [Landing]    law=C B_max=0.693147 T_res=300 rho*_0=0.0679778 rho_max=0.10013 stop_frac=0.95

control      [ServerStep] trust_ratio rho_star=0.06 schedule=rm exp=0.25
             [CommitGate] n_target s=1.5 p=448802 probe_combine=mean P=10 G_rule=10.0 rho_ref=setpoint
```

### 5.2 The controller wins on every dataset

Peak held-out accuracy (peak, never final — byte-identical replicates differ by ±0.045 past a turn against
±0.0009 at peak):

| | controller | control, full budget | vclock at which the controller passes the control's **best-ever** accuracy | reference |
|---|---|---|---|---|
| agnews | **0.8676** | 0.8432 | 8,257 — **5.5×** | 0.850 |
| yahoo | **0.6571** | 0.4275 | 9,017 — **6.5×** | 0.734 |
| yelp-p | **0.8141** | 0.7280 | 5,494 — **8.5×** | 0.874 |

**No control ever reaches its controller's peak**, on any dataset, over its entire budget.

### 5.3 Why it wins — the mechanism is visible in one column

The controls ended having spent:

| | control `B` at end of its full budget | controller `B` |
|---|---|---|
| agnews | 0.1075 | 0.6972 |
| yahoo | 0.1187 | 0.6900 |
| yelp-p | 0.1109 | 0.9699 |

A **fixed `ρ*` = 0.06 spends ≈0.11 of budget regardless of the dataset** — that is what a hand-set
constant does, by definition. The controller spends 6–9× more of the same budget in the same wall clock.

And the *cost* of under-spending is task-dependent, which is why the margin varies: agnews saturates early
(its control reaches 0.843 of a 0.850 reference on `B` = 0.11), while yahoo needs far more budget and its
control stalls at 0.428 of 0.734. **This is the claim in one sentence: the right amount of budget is not a
constant, so it must be sensed — and the penalty for guessing is set by the task, not by the guess.**

### 5.4 The formulation is confirmed by the runs, not merely consistent with them

Predicted vs observed norm inflation, all six arms, `Φ_predicted = e^B` computed from the `ρ` sequence
alone against `Φ` measured from the weights:

| arm | commits | `B` | `Φ` predicted | `Φ` observed | error |
|---|---|---|---|---|---|
| agnews controller `152215` | 899 | 0.6972 | 2.01 | 2.01 | **0.02%** |
| agnews control `021843` | 938 | 0.1075 | 1.11 | 1.11 | **0.00%** |
| yahoo controller `125003` | 964 | 0.6900 | 1.99 | 2.00 | **−0.23%** |
| yahoo control `151619` | 1,138 | 0.1187 | 1.13 | 1.12 | **0.13%** |
| yelp-p controller `125010` | 1,348 | 0.9699 | 2.64 | 2.64 | **0.09%** |
| yelp-p control `161751` | 997 | 0.1109 | 1.12 | 1.12 | **0.16%** |

Six arms, three datasets, two schedules, up to 1,348 commits, **worst error 0.23%**. The quantity the
controller steers by is the quantity the system actually obeys.

### 5.5 The controller terminates on its own budget

yelp-p `125010`, the one arm that ran to its own stopping condition:

```
[BudgetStop] reason=budget action=halt commit=1348 B=0.972057 B_max=1.02312 Phi=2.643
```

It stopped at **95.0% of a budget it measured itself**, having used **28,885 of the 50,000** vclock-seconds
it was allowed — 58%. It ended **0.0006 below its own peak**. Its control, given the full 50,000, peaked
0.086 lower and drifted 0.014 down from there.

### 5.6 The Λ-curve transfers across task

Pre-registered before the runs: *if yahoo reaches ~0.6–0.7 by `Λ` ≈ 1.0, the agnews `Λ`-curve transfers;
if it plateaus near 0.35 with `Λ` > 1.0, `Λ` does not transfer across task.*

**yahoo reached 0.6571 at `Λ` = 0.994** — against 0.30 on earlier arms and a 0.734 reference. agnews reads
0.868 at `Λ` = 1.001. The progress coordinate means the same thing on both tasks.

---

## 6. What is not yet settled

Stated plainly, because each of these bounds what the results above can be claimed to show.

### 6.1 The `B_max` sensor's grid does not bracket the knee

The injection probe reads accuracy at `Φ ∈ {1.5, 2, 2.5, 3, 3.5, 4}`, a range sized from *offline*
measurements of the knee (2.0–3.5). The **live** knee is far lower. Across all **19 in-run fires on the
three datasets, every grid point returned chance accuracy**:

| | chance | Φ=1.5 | 2.0 | 2.5 | 3.0 | 3.5 | 4.0 |
|---|---|---|---|---|---|---|---|
| agnews, fire 1 | 0.250 | 0.277 | 0.248 | 0.232 | 0.244 | 0.264 | 0.264 |
| yahoo, fire 6 | 0.100 | 0.318 | 0.125 | 0.080 | 0.105 | 0.092 | 0.100 |
| yelp-p, fire 8 | 0.500 | 0.531 | 0.482 | 0.482 | 0.482 | 0.518 | 0.490 |

The knee-finder therefore never brackets the crossing; it falls through to a two-point extrapolation
between its synthetic `(Φ=1, normalised 1.0)` anchor and the single `Φ=1.5` reading. Recomputed over the
fires, **5 of 8 sampled use one grid point and 3 use two — the points at 2.5–4.0 are consulted on none.**

Consequences:

1. `ln Φ_knee` is pinned into ≈0.21–0.42 regardless of the model, so **`B_max = B + ln Φ_knee` recedes as
   budget is spent**. The sensed value rises monotonically from ≈0.50 at the first fire on all three
   datasets.
2. The three combined values (0.795 / 0.805 / 1.023) **order by fire count (5/6/8), not by dataset** — so
   no claim that `B_max` differs *by task* survives this instrument.
3. The stop still fires, because `mean` lags a rising sequence: yelp-p halted at `B` = 0.9721 ≥ 0.95 ×
   mean 1.0231, while its *latest* sense was 1.186, which would not have stopped it. **The termination is
   arithmetic on the combiner, not `B_max` converging.**

The fix is to re-range the grid below 1.5 and re-read the knee. It does not affect §5.2–§5.6, which are
measured accuracies, but it does mean **"the controller senses the task's ceiling" is not yet supported** —
what is supported is that it senses *something* that produces a working landing.

### 6.2 Two of three do not reach the backprop reference, and all three were still climbing

| | peak | reference | gap | still improving at the end? |
|---|---|---|---|---|
| agnews | 0.8676 | 0.850 | **−0.018 (exceeds)** | yes, +0.0028 over the last 10% of vclock |
| yahoo | 0.6571 | 0.734 | +0.077 | yes, **+0.0123** — the fastest of the three |
| yelp-p | 0.8141 | 0.874 | +0.060 | yes, +0.0041 |

**No arm had plateaued.** For agnews and yahoo the reason is external — a monitoring bug killed both
mid-run — but **yelp-p stopped itself**, on `B ≥ 0.95·B_max`, while accuracy was still rising and with 42%
of its vclock unused. Under a criterion that values reaching the target above stopping early, that is a
**miss, not a success**: the budget stop is a bound on *damage*, and it fired well before the model had
extracted what the task allows.

The "ends within 0.015 of peak" check is therefore weaker than it looks: **an arm that is still climbing
passes it trivially**, because its peak is its last point. It demonstrates *no divergence*; it does not
demonstrate convergence.

**One qualifier that cuts the other way, and it matters.** yelp-p halted at **`Φ` = 2.643**, against an
independent safety stop at `Φ` = 2.7 that would have fired within ~30–60 more commits. The two nearly
coincided, so the arm was at the ceiling the *model* imposes, not at an arbitrary early cut. Simply
running it longer is therefore not available — the budget really is nearly spent. Since `Λ = 2B/s`, the
only lever on learning **per unit budget** is `s` (1.5 here, with a gate-reachability floor near 0.9);
running past `Φ` ≈ 2.7 is the regime every arm on record says destroys the head. Whether that ceiling is
real on yelp-p is an open experiment, not a settled fact.

Two further caveats on the reference itself: it is a 10-client, 3-epoch backprop run, loose by an unknown
and probably unequal amount per dataset — agnews' controller **exceeded** it — so "93% of reference" and
"90% of reference" are not comparable statements across columns.

### 6.3 No arm in this set ever diverged

The stop rule was motivated by collapse (one earlier arm fell 0.874 → 0.296). In these six arms that never
happened, so the stop demonstrated **efficiency** — stopping early at almost no cost — and not the
collapse avoidance it exists for. Those are different claims and should not be merged.

### 6.4 One model, one `p`

Every arm ever run is DistilBERT + adapters at `p` ≈ 4.5·10⁵; the three datasets differ in `p` by 1.4%.
So the generality demonstrated is **across task, holding the learner fixed**. The one time `p` moved for
real — adapter reduction factor 64, `p` = 118,348 — law C and the annealed gate **did not compose** under
any `T_res`. `T_res` = 300 and `f` = 0.95 are the two constants that are neither sensed nor derived, and
they are pinned to one `p`.

---

## 7. Summary

**The problem.** Backprop-free fine-tuning by directional derivatives takes steps that are orthogonal to
the model, so the weight norm grows monotonically and, under a raw step rule, geometrically — with the
gradient magnitude feeding back into the step. The prior system survives this only because its variance
gate carries units of `‖θ‖²`, which converts a fixed threshold into an accidental Robbins–Monro anneal
that defers collapse rather than preventing it, and hard-codes one trajectory.

**The formulation.** Orthogonality makes the trajectory an exact difference equation, giving two
dimensionless conserved quantities: a spend `B = ½Σln(1+ρ²)` with `Φ = e^B`, and a gain `Λ = Σρ·cos`.
Accuracy is monotone in `Λ`; whether it is *retained* is governed by `Φ` alone, which peaks at a fixed
2.41–3.11 across every arm on record for a geometric reason. Under a gate that holds `ρ ≤ s·cos`,
`Λ = 2B/s` identically — so the schedule cannot buy learning, only spend time, and there is nothing in it
to tune.

**The solution.** Make `ρ` a knob (trust-ratio), make the gate dimensionless (`n_target`), average the
probes, and spend the budget by `ρ*_t = √(2(B_max − B_t)/T_res)` with `T_res` a rate rather than a
deadline — which removes the horizon from the inputs. `B_max` itself is sensed on the running model by
forward-only noise injection.

**The evidence.** Across three datasets the controller beats the same stack with a hand-set, agnews-tuned
step by 5.5–8.5× in budget-to-equal-accuracy, and no control reaches its controller's peak. The budget law
predicts observed norm inflation to within **0.23% on all six arms**, up to 1,348 commits. On yelp-p the
controller halted itself on its own measured budget at 58% of the compute it was allowed, ending 0.0006
below its peak.

**The honest boundary.** The sensor that sets the target is mis-ranged, so what it returns tracks spend
rather than the task; two of three arms end below the backprop reference and all three were still
improving when they ended; and the whole result holds one model fixed.
