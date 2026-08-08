# FluxTune — divergence root cause, the stability criterion, and the probe-budget question

**Status:** analysis complete, no code changed. Written 2026-08-06 on `dg/fluxtune_expts_sim_init`.
**Purpose:** hand off a finished diagnosis so the next session can go straight to building.

**How to read this.** Part I is the model — pure theory, no reference to our config. Part II is what
the shipped system measurably does, scored against that model. Part III is what to change. Part IV is
generality and future directions. Part V is reference. Nothing in Parts I–II is a proposal; nothing in
Part III is a measurement.

Claims are tagged **MEASURED** (telemetry/logs/model builder; reproduction in §21), **DERIVED**
(arithmetic on MEASURED inputs), **ANALYSIS** (rests on a stated modelling assumption), or
**HYPOTHESIS** (untested; ledger in §18).

**The one-paragraph summary.** The committed update is a pure random walk whose *relative* size is
pinned at ρ = 0.115, so the adapter norm compounds geometrically and the model is destroyed by logit
saturation at ~110 commits. The stability condition is `ρ ≤ cos(G,g)`; the shipped config sits **5.0×
over budget**, at commit 1, by construction. The cheapest fixes are already paid for: averaging the P
probes we already compute instead of selecting one is **10×** at identical cost, and freezing one
768×768 layer is another **2.31×**.

---

# Part I — The model

*Nothing in this part depends on our configuration. It is the frame everything else is scored in.*

## 1. What this optimizer is actually doing

### 1.1 Weights, gradients, and what one probe actually returns

**Weights `θ` are a place; the gradient `g` is a direction.** Both are `p`-vectors — one number per
trainable parameter — but they answer different questions. `θ` says where the model currently sits in
`p`-dimensional space. `g` says, at that exact spot, which way the loss increases fastest; training
moves along `−g`. The gradient is not a property of the model, it is a property of the model *and* the
current data bin, and it changes at every step.

The whole difficulty is how much of `g` one unit of compute can tell you:

| | backprop (traditional) | forward-gradient (here) |
|---|---|---|
| cost of one measurement | 1 forward + 1 backward ≈ 3 forward-equivalents | **2 forward passes**, no backward |
| what it returns | **all `p` components of `g`, exactly** | **one scalar** `d = ⟨g,v⟩` — the slope along one direction `v` |
| memory | stores activations for the backward pass | nothing beyond a forward pass |
| the update it produces | `−η·g` — right direction, known length | `−η·d·v` — right **on average**, ~0.1% aligned individually |
| sources of noise | data sampling | data sampling **+ which direction you happened to ask about** |

So backprop learns `p` numbers per measurement; we learn **one**. That ratio — a million to one on this
model — is the entire cost of not having a backward pass, and every technique in this document is a way
of buying some of it back by averaging. `ĝ = d·v` is unbiased (`E[d·v] = g`, because the probe's
overlap with the gradient enters twice — once in `d`, once in `v`) and has enormous variance.

### 1.2 The hillside in fog — the image used throughout

You are on a hillside in thick fog and want to reach the bottom.

- **Backprop** is feeling the slope under your feet in every direction at once. You know which way is
  downhill, exactly, before you move.
- **A compass is not enough**, and this is the point of the analogy: a compass tells you *north*, and
  north is a direction in the horizontal plane. Downhill is a *different* horizontal direction that
  depends on the terrain, and no fixed reference tells you what it is.
- **Forward-gradient** is what you actually have: pick a direction at random, take a small test step,
  feel whether you went up or down and by how much (`d`), step back. **One reading. One number.**
- **One reading is nearly worthless** — in a million dimensions the direction you happened to try is
  almost perpendicular to downhill, so `d` is small and its sign tells you almost nothing about where
  the valley is. **A thousand averaged readings are a slope meter.**

Recurring vocabulary from here on: *one reading* = one probe; *the fog* = no backward pass; *how steep
did that feel* = `d`; *how much of your step was genuinely downhill* = `cos(G,g)`.

### 1.3 Four consequences

**(a) The parameter count is the adversary.** A random direction in `p` dimensions overlaps any fixed
target by only ≈ `1/√p`; at `p ≈ 10⁶`, one probe is 0.1% signal. No tuning removes this — it is the
price of the fog. It is why `p` appears in every formula below and is itself a *lever*.

**(b) Averaging is the only real lever, and it pays twice.** Over `n` independent readings the signal
is identical in all of them so it adds **linearly** (`×n`), while the near-orthogonal noise adds **in
quadrature** (`×√n` — see §3.2). The average is simultaneously **better aimed** *and* **shorter**,
which is why pooling improves safety as `1/n`, not `1/√n`.

**(c) Aim matters more than step size, because misaim never cancels in length.** Split each step into
the part that points downhill and the part that does not. The downhill parts point the same way every
time and accumulate **linearly** in `T`. The random parts point differently each time, so as
*displacement* they accumulate only as `√T` — but as **distance from the origin** they always add.
That last clause is the Pythagoras point and it is worth being concrete:

> Take a step `Δ` perpendicular to your current position `θ`. The new distance from the origin is
> `√(‖θ‖² + ‖Δ‖²)`, which is **larger than `‖θ‖` for every perpendicular direction** — the sign and
> orientation of `Δ` do not matter, only its length. So while random steps partly cancel each other as
> *displacement*, they can never cancel in *length*: each one strictly inflates the weight vector.
>
> With our numbers: `‖θ‖ = 20.36` and `‖Δ‖ = 0.115·20.36 = 2.34` gives a new norm of
> `√(20.36² + 2.34²) = 20.49` — **+0.66% per commit, unconditionally**. Compounded, ×2 in 105 commits.

**The race** is between those two accumulations. The useful part gets you `T·ρ·cos` of the way to a
solution (in units of your own norm); the wasted part inflates you by `(1+ρ²)^{T/2}`. Either you
arrive before you inflate, or you don't. `ρ ≤ cos` is that race written down (§3.3).

**(d) Inflated weights destroy a classifier by making it confident, not by making it forget.** The
logits are `z = W·h + b`, linear in the weights, so scaling `W` by 4.2 scales the *gaps between
logits* by 4.2 and the softmax saturates toward one-hot. Two separate things are going on:

- **Magnitude sets confidence.** It grew 4.2×.
- **Direction sets which class wins.** It was randomized by 185 mutually orthogonal kicks.

So the model becomes maximally confident about an arbitrary answer. The natural objection — *but some
of the weight increase was in the right direction, why doesn't that carry the day?* — is answered by
arithmetic: only 2.3% of each step is aligned, and the aligned parts add up to roughly **0.5 norms** of
coherent progress over 185 commits while the norm itself grows **4.2×**. The learning is real and it is
outvoted about 8:1. Test loss above `ln(num_classes)` is the fingerprint: **worse than answering "I
don't know."**

## 2. Vocabulary

### 2.1 Two different things are called "selection"

The codebase and this document both use the word for two unrelated mechanisms. They are kept distinct
throughout:

- **probe selection** — trainer-side: given `P` perturbation directions, which one(s) to turn into an
  upload. This is what C1 is about. Governs `E[v∥²]`.
- **client selection** — the FLAME `selector` component: which clients to dispatch to and how many to
  wait for. Governs `K` and `C`.

Where an unqualified "selection" appears in a quoted claim from another document, it means client
selection.

### 2.2 Symbols

Everything is per-commit unless stated. "Dimensionless" means the quantity is a pure ratio, so
comparing it against a fixed constant is legitimate — that property is the whole fix (§5).

| symbol | what it is | value here | read it as |
|---|---|---|---|
| `θ` | the full weight vector | 67.4M entries | where the model sits |
| `θ_tr` | the **trainable** slice (adapters + heads); the rest is frozen and never moves | `p` = 1,040,932; `‖θ_tr‖` = 20.36 at init | the only thing that can diverge |
| `p` | trainable dimensions = the probe dimension | 1,040,932 | "how many directions I could have asked about" |
| `v` | one probe direction, a raw Gaussian draw, **not** normalised | `‖v‖ = √p ≈ 1020` | the question being asked |
| `h` | finite-difference spacing | 0.01 → probe displacement `h‖v‖ = 10.2` | **a measuring instrument, not a step size** |
| `d` | the JVP: a **scalar** per probe, `d ≈ ⟨g,v⟩` | rms 3.6 → 21.5 over the run | "how steep did that feel" |
| `ĝ = d·v` | one trainer's gradient **guess** — a `p`-vector | — | unbiased, enormous variance |
| `g` | the true gradient at `θ` | measured by the §15.1 probe | the target; **not observable in production** |
| `u_k` | one trainer's uploaded update | — | one pooled reading |
| `G` | server's pooled update before the step | — | the direction actually taken |
| `P` | probes per trainer per iteration | 10 | trainer-side pool |
| `K` | trainers pooled per commit (`agg_goal`) | 10 | cohort-side pool |
| `I` | iterations over **the same data bin** before committing | ≈18.5 | gate-side pool |
| `n = P·K·I` | **total independent readings behind one commit** | ≈1,850 (only 185 of them used, §11) | the number that sets aim |
| `N = K·I` | uploads pooled server-side | ≈185 | **not** the client population (100) |
| `C` | concurrency pool (`c`) — caps how large `K` can be | 30 | — |
| `η` | server learning rate (a **knob**) | 0.01 | — |
| `ω` | per-upload aggregation weight | 0.70–0.87 | a re-weighting, not a step size |
| **`ρ`** | **relative step: `‖Δθ‖ / ‖θ_tr‖`** — an **outcome**, not a knob | **0.115, flat all run** | "what fraction of myself do I move each commit" |
| **`cos(G,g)`** | **fraction of the taken step aligned with the true gradient** | **0.0231** | "how much of that step was genuinely downhill" — **not** model accuracy |
| `E[v∥²]` | **probe-selection gain** (below) | 2.991 shipped | dimensionless multiplier |
| `G_rule` | pooling gain of the combination rule: `E[v∥²]` if selecting, `P` if averaging | 2.991 → 10 | §3.5 |
| `a`, `b` | estimator shape constants (§3.1) | — | properties of the **rule**, not of the data |
| `var` | the commit gate's statistic: spread of `d` **across probes**, not across data | drifts 0.4 → 15 | has units of `‖θ‖²` — that is the bug |
| `ρ*` | the relative step an operator would *set* under S-A | proposed | the knob `ρ` should have been |

### 2.3 Probe-selection gain `E[v∥²]`

Draw `P` probes and keep the one with the largest `|d|`. The kept probe's **squared overlap with the
gradient direction** is on average `E[v∥²]` times that of a probe kept at random. Normalised so
*random* = 1.000, it is the expected value of the largest of `P` draws from a chi-square with one
degree of freedom — a pure order statistic, fixed by `P` and the rule, independent of model and data.
Top-1 of 10 gives 3.811. It is a *gain*, not a probability, and it says nothing about stability (§11.1).

### 2.4 `v` is a raw Gaussian draw, and what that implies

Each of the `p` coordinates of `v` is drawn independently from `N(0,1)` — `torch.randn_like`, never
normalised (`tc_transformer_trainer_distribute.py:416`). Three consequences:

- **`‖v‖` is essentially a constant**: it concentrates at `√p ≈ 1020` to within 0.07%. So normalising
  `v` would not change the *direction* quality at all — it is a no-op for `cos`.
- **But it does set the probe displacement**, `h‖v‖ = h√p = 10.2`. Nobody chose 10.2; it fell out of
  the parameter count. **Change `p` and the finite-difference spacing silently changes with it** — a
  live hazard for S-I, which cuts `p` by 57% and would move `h‖v‖` from 10.2 to 6.7 as a side effect.
- **Isotropy**: a Gaussian draw has no preferred direction, so `⟨v,x⟩` has the same distribution for
  every unit vector `x`. This is exact, by construction, and is what licenses `1/√p` in §1.3(a).

## 3. The stability criterion

### 3.1 One upload: two shape constants

Take any single upload `u` (a `p`-vector) and split it into the part lying along the true gradient
direction `ĝ = g/‖g‖` and everything else: `u = α·ĝ + u⊥`.

- **`a` measures the shadow.** Defined by `E[α] = a·‖g‖`. Project the upload back onto the gradient
  direction; `a` is the length of that shadow in units of `‖g‖`. **"How much true gradient survives"**
  means exactly this: `a = 1` says the average upload carries a gradient-component as long as the
  gradient itself; `a = 3` says three times as long.
- **`b` measures the total length.** Defined by `E‖u‖ = b·‖g‖·√p`. The natural unit here is
  `‖g‖·√p`, not `‖g‖`, because the plain one-probe estimator `d·v` has length `|d|·‖v‖ ≈ ‖g‖·√p`.
  So **`b = 1` is "as long as one raw probe"**, and `b` says how much longer or shorter your rule makes
  the upload than that reference.

Everything follows from the pair, because a single upload's aim is `cos(u,g) = a‖g‖ / (b‖g‖√p) =
(a/b)/√p`. **`a/b` is aim per upload; `b` alone is length per upload.**

Worked, for the two rules that matter (setting `‖g‖ = 1`, so `d = v∥`, the probe's component along the
gradient):

| rule | shadow `⟨u,ĝ⟩` | length `‖u‖` | `a` | `b` | `a/b` | `b²/a` |
|---|---|---|---|---|---|---|
| one raw probe `d·v` | `v∥²` | `\|v∥\|·√p` | 1 | 1 | 1 | 1 |
| **select best of P by `\|d\|`** | `v∥²` of the winner | `\|v∥\|·√p` of the winner | `E` | `√E` | `√E` | **1** |
| **average all P** | `1` (unbiased) | `√(p/P)` | 1 | `1/√P` | `√P` | **1/P** |

The middle row is the lockstep that answers "why doesn't picking the best probe help?" — selecting on
`|d|` raises the **shadow quadratically** (`∝ d²`, hence `a = E`) and the **length linearly**
(`∝ |d|`, hence `b = √E`). Aim `a/b = √E` genuinely improves. But the step length also grows by `√E`,
and stability depends on `b²/a`, in which the two cancel **exactly, for any `E`**.

`a` and `b` are **properties of the combination rule, known in closed form before the run starts**.
They are not measured online and they do not drift. They change only if you change the rule. The one
empirical input they rest on is that `d` is Gaussian — verified in §11.2.

### 3.2 Pooling: signal linear, noise "in quadrature"

**"In quadrature" means lengths combine by Pythagoras rather than by addition.** Add two vectors of
length 1 pointing the same way and you get length 2; add two pointing at right angles and you get
`√(1²+1²) = √2`. In `p` dimensions independent random vectors are very nearly at right angles, so `n`
of them sum to length `√n`, not `n`.

Pool `n` uploads. The shadow is present and identical in every one of them, so it adds **linearly** and
survives averaging untouched. The perpendicular junk is different in each, so it adds **in quadrature**
and averaging shrinks it by `√n`:

```
shadow of the pooled G   ~  a * ||g||                    <- unchanged by pooling
||G||                    ~  b * ||g|| * sqrt(p/n)        <- shrinks as 1/sqrt(n)
cos(G, g)                =  shadow / length = (a/b) * sqrt(n/p)
rho                      ~  eta * ||G|| / ||theta||      <- also shrinks as 1/sqrt(n)
  =>  rho / cos          ~  (b^2/a) * p / n
```

Read the last line as **safety = (a property of the combination rule) × (dimensions per pooled
reading)**. Improve it by choosing a better rule (`b²/a`), probing fewer dimensions (`p`), or pooling
more readings (`n`).

### 3.3 The criterion

Two clocks run from the moment training starts.

- **The deadline.** Coherent progress grows as `T·ρ·cos`; accumulated noise displacement grows as
  `√T·ρ`. Signal overtakes noise at `T ≈ 1/cos²`. Before that, you are mostly wandering.
- **The budget.** The norm inflates by `(1+ρ²)^{T/2}`, doubling at `T ≈ 1.4/ρ²`. After that, the
  classifier is saturated and gone.

You survive iff the deadline arrives before the budget runs out: `1/cos² ≤ 1.4/ρ²`. Dropping the
constant:

> ## ρ ≤ cos(G, g)
> **The relative step must not exceed the fraction of it that is actually aligned with the gradient.**
> Take a step bigger than you can aim, and the norm blows up before the aim pays off.

**What to target.** Progress per commit is `ρ·cos`, increasing in `ρ`, so you want `ρ` **as large as
the budget allows** — the boundary, not zero. But the derivation drops O(1) constants and `cos` is
estimated, and the risk is violently asymmetric: over-budget fails *exponentially*, under-budget is
merely *linearly* slow. So run at a safety factor: **`ρ/cos ≈ 0.3–0.5` early**, then anneal `ρ` toward
zero (S-B) so the iterate settles instead of hovering in a noise ball. It is a schedule, not a setpoint:
start just inside the boundary, end far inside it.

### 3.4 Measuring `cos` without a backward pass

`cos(G,g)` mentions `g`, which forward-gradient FL never computes. There are three separate ways to get
at it, and the design needs all three:

| | how | what it is for |
|---|---|---|
| **predicted** | `cos = (a/b)·√(n/p)` — closed form from the config alone, no measurement | sizing a configuration before you run it (§3.5) |
| **estimated online, gradient-free** | **split-half cosine**: split the pool in two, average each half, take the cosine between the two averages. Independent halves share only the signal, so `cos(half_a, half_b) ≈ cos²/2`, giving `cos ≈ √(2·cos(a,b))` | the production control law — this is what a gate or controller reads (S-E) |
| **ground truth** | manufacture `g` with a real backward pass on a small fixed held-out batch, **server-side only**, once per commit | validating the other two; one-time instrumentation, no protocol change (§15.1) |

The ground-truth probe is affordable precisely because it is not on-device and not per-client: one
backward pass on a few hundred held-out samples, on the server that already holds `G`.

### 3.5 Sizing a configuration

Collapse the rule into a single **pooling gain** `G_rule`: the factor by which the rule multiplies the
effective number of readings. From the §3.1 table, `(a/b)² = G_rule` with

```
G_rule = E[v_par^2]   for select-one-of-P     (P does not appear -- see §11.1)
G_rule = P            for average-all-P
```

Then the whole model is one line, and its inversion is the sizing formula:

```
cos = sqrt( G_rule * N / p )          N = K*I uploads pooled server-side
N_req( rho* ) = p * rho*^2 / G_rule   <- pool needed to make rho* safe
```

Sizing table (**DERIVED**), for our `p`:

| config | `G_rule` | `p` | `N` needed at ρ\*=0.115 | at ρ\*=0.05 | at ρ\*=0.02 |
|---|---|---|---|---|---|
| shipped (select 1 of 10) | 2.99 | 1,040,932 | 4,600 | 870 | 139 |
| + S-H (average all 10) | 10 | 1,040,932 | 1,376 | 260 | 42 |
| + S-H + S-I (freeze `pre_classifier`) | 10 | 450,340 | 595 | **113** | 18 |

Today's `N ≈ 185`. Two readings of the same fact, and it is worth being explicit about which is meant:

- **Holding `η` fixed** (so `ρ` falls as you pool more, and `cos` rises): `ρ/cos ∝ 1/N`, they meet at
  `N ≈ 925` — that is the "5.0× over budget ⇒ 5× more pooling" statement.
- **Holding `ρ` fixed at 0.115** (what S-A makes possible): only `cos` moves, so you need `N ≈ 4,600`.

Under S-A the operator sets `ρ*`, so the second column is the operative one — and the table's punchline
is that **with S-H and S-I in place, a modest `ρ* = 0.05` is already satisfied by `N = 113`, below
today's 185.** The pool does not need to grow at all; the rule and the dimension do the work.

Translating `N` into deployment knobs: `N = K·I`, and `C ≥ K` must hold or the cohort cannot fill.
Wall-clock per commit is set by `I` (each iteration is a serial round trip), not by `K` (clients run in
parallel), so **choose the smallest `I` that the gate allows and buy the rest with `K`** (§13).

### 3.6 What is standard, and what is ours

`ρ ≤ cos` as a single inequality is **ours** — derived here, not quoted. None of its ingredients are:

| ingredient | where it is standard |
|---|---|
| a random probe in `p` dims aligns with a target only as `1/√p` | classical ZO / forward-gradient analysis (Nesterov–Spokoiny; Duchi et al.; Baydin et al.) |
| signal accumulates linearly, noise as `√T` | the standard SGD noise-ball argument |
| convergence needs `Σρ_t = ∞`, `Σρ_t² < ∞` | Robbins–Monro stochastic approximation (1951) |
| bound the step **relative to `‖θ‖`**, not in absolute units | trust-region methods; LARS/LAMB |
| "there is a pool size beyond which more pooling buys nothing" | the **gradient noise scale** / critical batch size (McCandlish et al.) — the closest established relative in shape |

**What is ours** is the packaging: collapsing those into an inequality **between two quantities the
server already logs or can estimate gradient-free**, which turns an asymptotic rate statement into an
*online control law* — a gate can literally evaluate "is `ρ ≤ cos` right now?". That is what makes S-C
and S-E possible and is the honest form of the contribution claim. Do not write it up as a new theorem.

## 4. The lever table — this ranks every possible fix

Two columns answering different questions. **`ρ/cos` is "will this survive"** (under 1 = safe).
**`ρ·cos` is "how fast does it learn"**. A good lever improves the first without hurting the second; a
bad one trades them 1:1. Pooling is the only lever that is free, because it shortens the step and
improves the aim by the same `√n` — **ratio improves by `n`, product unchanged**.

| lever | `ρ/cos` (stability) | `ρ·cos` (per-commit progress) | who owns it |
|---|---|---|---|
| **P-averaging** (trainer-side pooling) | **∝ 1/P** | **invariant — and free in wall clock** | trainer |
| **K** (cohort width) | **∝ 1/K** | invariant — parallel across devices | client selection |
| **I** (iterations per bin) | **∝ 1/I** | invariant — **but serial: one round trip each** | aggregation gate |
| **p** (trainable dimension) | **∝ p** | ∝ 1/p | model/PEFT design |
| `η` learning rate | ∝ η | ∝ η — **pays 1:1** | aggregation |
| probe-selection gain `E[v∥²]` | **invariant** (`b²/a = 1`) | ∝ E | trainer probe selection |
| step normalization | sets ρ to an operator constant | decoupled | aggregation |

Per unit of **compute**: progress ∝ `C/n`, norm inflation ∝ `C/n²`, so the ratio improves ∝ `n`.
**Larger pools are strictly better for stability per FLOP and slower only in absolute wall-clock
progress.** That is the honest trade, and the whole argument for pool control being a contribution
rather than a tuning detail.

## 5. The ratio principle

Every quantity this algorithm compares against a fixed constant must be **scale-invariant** — a ratio
of two quantities that grow the same way — because anything with units silently changes meaning as
training proceeds.

The test is: *re-parameterise the model so `‖θ‖` doubles. The shape of the loss surface is unchanged,
so the trajectory should be unchanged.* Any rule that fails this test needs re-tuning every time the
model, the adapter rank, or the round index changes.

This is the design principle to claim, and it is stronger than any individual fix: **a training rule
whose constants are dimensionless does not need to be re-tuned when the model changes.** In practice it
means three replacements — an absolute step `η` becomes a relative step `ρ*`; an absolute variance
threshold becomes a cosine; an absolute iteration cap becomes a measured adequacy condition. ("Scale-
free" was the earlier wording in this document; *scale-invariant* is the accurate term, since the
quantities do have a scale — the rule just must not depend on it.)

## 6. Traps

1. **`ρ` is not `η`.** `η` is a config number; `ρ` is what the system *ends up* doing, pinned by a
   feedback loop (§9.2). The whole of S-A is "make `ρ` the knob instead of `η`."
2. **`h` is not a step size.** `h‖v‖` is how far you displace the weights *to take a measurement*, and
   you step back afterwards. `ρ` is how far you actually move. They are independent, and shrinking one
   does nothing for the other.
3. **`cos(G,g)` is not a quality score for the model.** It is the aim of *one server step*, and can sit
   at 0.023 while accuracy climbs happily (§10).
4. **"Unbiased" does not mean "accurate."** `ĝ = d·v` has the right *average* and catastrophic
   *variance*. Probe selection makes it lower-variance-looking but **biased in scale** by ≈3× — the
   unmatched-effective-LR confound in every guided-vs-random A/B so far.
5. **`N` is a pooling count, not a population.** `N ≈ 185` uploads per commit; the 100-client
   population is irrelevant to the noise arithmetic.
6. **A drifting `var` is not a data effect.** `var` is the spread of `d`, and `d ∝ ‖θ_tr‖`, so the
   gate's own ruler grows as `‖θ‖²` (§9.3).

---

# Part II — Diagnosis of the shipped system

*All MEASURED or DERIVED against the model above. No proposals here.*

## 7. Scope and configuration

| run | mode | duration | commits | outcome |
|---|---|---|---|---|
| `run_20260804_003042_fluxtune_n100_smoke_syn_0_real` | real | 3.93 h | 189 | peak acc 0.846 @ r1/did106 → **0.250 / mcc 0.000 / loss 2.37** @ r2/did38 |
| `run_20260804_043301_fluxtune_n100_smoke_syn_0_real` | real | 3.93 h | 185 | peak acc 0.853 @ r1/did122 → **0.252 / loss 1.54** @ r2/did34 |
| `run_20260805_{110016,130231,150446}_fluxtune_..._real` | real | 1.86–1.88 h | ~95 | stop at did 92–94 — **at the peak, before the collapse** |

Both 4 h runs contain the failure and agree on every derived constant. Deterministic, not a bad seed.

**Why it looked intermittent.** The 2 h runs terminate on `max_runtime_s` after ~95 commits; the
norm-doubling time is 109–116 commits (§9.2). **The short runs end at 0.85 doublings — exactly where
accuracy peaks.** Hence the standing rule: **4 h minimum, or the result is uninformative. Any A/B
shorter than ~3.2 h will report a win for a config that diverges.**

**No usable baseline control exists yet.** The `fwdllm` runs from the same period
(`run_2026080[45]*fwdllm_n100_smoke_syn_0_real`) only reach `data_id=38` in 1.84 h and never leave
acc 0.25 — the sync path is too slow at this horizon. Its `|jvp|` is flat (rms 3.1–4.0 over 3,101
merges) because its weights barely move, which is not evidence of stability. **A fair cross-baseline
comparison at the horizon where fluxtune fails does not exist** (H-F).

The shipped config, from `aggregator_config.json` of the reference run and
`expt_scripts/fluxtune_n10_smoke.yaml` (launched by `expt_scripts/run_sequential.sh`, entry
`fluxtune:`). **MEASURED.**

```
population N_clients=100, alpha=1 (Dirichlet), agnews, DistilBERT-base + AdapterHub adapters
trainable p = 1,040,932  (1.5% of 67.4M; backbone frozen)
train_batch_size = 8          -> one "data bin" = 8 samples, 150 bins per round
perturbation_count P = 10, h = 0.01, central finite difference, fp16/autocast, no_grad
probe selection: rank by |JVP|, coin-flip between top-2  (tc_transformer_trainer_distribute.py:481-483)
client selection: async_oort, c = 30, agg_goal K = 10, dynamic_kc.enabled = FALSE
commit gate: var_threshold = 0.3, var_stopping_policy = plateau (patience 3, rel_delta 0.15),
             max_iterations_per_data_id = 20
server step: theta <- theta - eta * (1/N_acc) * sum_k omega_k * d_k * v_k   (raw in-place SGD)
             eta = 0.01, measured constant to 4 digits for the whole run
             server_momentum absent => 0.0 (no-op). No optimizer state of any kind.
omega: grad_aware / base=new / align_gate=true / align_floor=0.0 / inverse_var=false
       scale=0.4, a_exp=0.25, b_exp=0.1        [measured omega in 0.702..0.865, median 0.818]
```

**Where the 1.04M probe dimensions live** (**MEASURED**, from the real builder) — this matters a great
deal (§11.4):

```
pre_classifier   590,592   56.7%      <- ONE 768x768 layer is over half the probe dimension
adapters         447,264   43.0%
classifier         3,076    0.3%
```

`‖θ_trainable‖` at init = 20.356 (from `build_model(4, 192)`, the production path), so the probe
displacement `h‖v‖ = 10.203` is **50% of the trainable norm** — a chord across half the parameter
space, not a derivative. Two things make this less alarming than it looks, and one thing keeps it
unfixable:

- The resulting `d` values are still consistent with the exactly-linear model to 0.3% (§11.2), so the
  truncation error is not showing up.
- The displacement gets *relatively smaller* as the run proceeds (`h‖v‖/‖θ_tr‖`: 0.50 → 0.12) while
  the failure gets *worse*, so the finite difference is not the villain (H-C).
- `h` cannot be reduced anyway: at fp16 the two forward losses agree to 1–2 significant figures at
  `h = 0.01` (QA §4.2), so **`h` is pinned between truncation error above and catastrophic cancellation
  below** — a real and publishable tension.

The instinct that the *update* should shrink as learning progresses is correct and is S-B; it is a
separate knob from `h` (trap 2), and it must be *derived* rather than hand-set, because premature
shrinking stalls you before you arrive. Robbins–Monro's `Σρ_t = ∞` is precisely the formal guarantee
that you never run out of travel (§15.6).

## 8. What failure looks like

Test-set trajectory, `run_20260804_003042` (**MEASURED**, `agg_eval`, eval every 2 commits):

```
 0.00h r1 did=  0   acc 0.398  loss 1.380     <- init, ln(4) = 1.386
 1.38h r1 did= 70   acc 0.816  loss 0.993
 2.13h r1 did=106   acc 0.846  loss 0.554     <- PEAK
 2.95h r1 did=146   acc 0.738  loss 0.692     <- degradation starts INSIDE round 1
 3.04h r2 did=  0   acc 0.818  loss 0.547
 3.34h r2 did= 14   acc 0.527  loss 1.464     <- knee
 3.56h r2 did= 24   acc 0.250  loss 1.688     <- single-class collapse
 3.87h r2 did= 38   acc 0.250  mcc 0.000  loss 2.371
```

Three observations rule out the obvious explanations:

1. **Final loss 2.37 > ln(4) = 1.386.** The model is *confidently wrong*, not uncertain — the
   fingerprint of §1.3(d). A model that had merely lost its signal would sit **at** ln 4.
2. **Train loss diverges too.** `stat_utility` (Oort utility = `8·rms(batch loss)`) goes
   **10.84 → 5.31 (did 120) → 13.74 (r2 did 35)** = train loss 1.355 → 0.66 → 1.72. Train and test
   move together — **not overfitting**.
3. **Degradation starts inside round 1** (did 138–148), before the round boundary. The boundary is not
   the trigger. (Consistent with F10, which already REFUTED the staleness-reset bug.)

## 9. Root cause — three measured legs of one defect

### 9.1 Leg 1 — the committed update is a pure random walk. **MEASURED.**

**A random walk** is a sequence of steps whose directions are uncorrelated with each other and with
where you already are. Its signature is that expected *displacement* is zero while expected *distance
travelled* grows as `√T` — you go nowhere, slowly, but you get further from where you started. Descent
is the opposite: steps correlated with a fixed direction, displacement growing linearly in `T`.

The test is assumption-free. Expanding one step, `‖θ+Δ‖² = ‖θ‖² + 2⟨θ,Δ⟩ + ‖Δ‖²`, the ratio of observed
norm growth to step energy is `1 + 2⟨θ,Δθ⟩/‖Δθ‖²`. It has an exact null: **1.000 means the step carries
no component either toward or away from where the model stands.**

```
sum over all commits of observed d(||W||^2)  /  sum of ||dTheta||^2
      run_003042 : 1.0326        run_043301 : 1.0323
```

**100% of the step energy goes into inflating the parameter norm, to within 3%.** The residual 3.2%
corresponds to a mean cosine between `θ` and `Δθ` of `1.9e-3` — about twice the `1/√p = 9.8e-4` scale
of pure randomness, so it is a small but statistically real *outward* tilt, consistent with the
norm-growth feedback below. Pythagoras accounts for ~97% of the inflation and the tilt for ~3%.

Three clarifications this measurement invites:

- **The origin is not the target.** A ratio below 1 would mean steps lean back toward zero, which is
  what weight decay does — it is not the goal, and the minimum sits at some nonzero `‖θ‖`. The ratio is
  a **null test for the presence of descent**, not an objective to optimise. Systematic descent would
  show as a *deficit* below 1; there is none.
- **Orthogonal to `θ` is not the same as orthogonal to `g`.** A step can be 2.3% aligned with the
  gradient and still ~perpendicular to the weight vector, because `g` itself is nearly perpendicular to
  `θ` in high dimensions. Both facts hold here simultaneously; they are not in tension.
- **This is not "the parameters moved the right way and later went wrong."** Every commit is orthogonal,
  including commit 1. But the 1.032 figure is a *sum over the whole run* and cannot by itself rule out
  a drift — **we have not measured the ratio per 20-commit block, and we should**; it is a free log
  replay (H-I, §18).

**Is norm growth typical in training?** Growth, yes — adapters initialize near zero, so `‖θ_tr‖` must
rise for anything to be learned. Healthy training grows the norm quickly and then *flattens*, because
gradients shrink as the loss flattens and because weight decay or normalization layers pull back. What
is atypical here is that it is geometric, unbounded, and is the *entire* content of the update.

### 9.2 Leg 2 — the relative step is pinned, so the norm inflates geometrically. **MEASURED + DERIVED.**

A **relative step** is `‖Δθ‖/‖θ_tr‖` — the fraction of your own current size that you move. It is the
right thing to look at because the *absolute* step here grows 6× over the run while the relative step
does not move at all. **Geometric growth** means multiplying by a constant factor each commit
(×1.0066) rather than adding a constant amount; compounding, it doubles every ~110 commits.

This leg **depends on Leg 1**: `‖θ_{t+1}‖ = ‖θ_t‖·√(1+ρ_t²)` is only valid because the step is
orthogonal. That the resulting prediction lands within 2.5% is therefore also an independent
confirmation of Leg 1.

Reconstructing `‖θ_tr,t‖` from telemetry using only *differences* of `‖W‖` anchored at the measured
init (this cancels the frozen-norm constant; both runs independently imply `‖θ_fz‖ = 415.929`, matching
the directly measured 416.213 to 0.07% — a strong consistency check):

```
rho = ||dTheta|| / ||theta_tr||        first-20 commits   mid-run   last-20
                       run_003042            0.113          0.118     0.163
                       run_043301            0.116          0.117     0.159

||theta_tr||:  20.36 (init)  ->  86.16 (commit 189)     = 4.2x

geometric-walk prediction  ||theta_{t+1}|| = ||theta_t|| * sqrt(1 + rho_t^2):
      predicts 83.98 vs 86.16 observed   (-2.5%)   [run_003042]
      predicts 81.01 vs 82.85 observed   (-2.2%)   [run_043301]
norm doubling: predicted 109 commits, observed 114  |  predicted 104, observed 116
```

**A two-parameter model — "every step is a random direction of relative size 0.115" — predicts the
final adapter norm after 189 commits to within 2.5%.** Nothing else is happening.

**Where 0.115 comes from, given that nobody set it.** `ρ = η·‖G‖/‖θ_tr‖`, and `‖G‖ ∝ rms|d| ∝ ‖θ_tr‖`,
so the `‖θ_tr‖` cancels and what is left is `η` times a dimensionless constant of the estimator (≈11.5
for this model and rule). The proportionality `|JVP| ∝ ‖θ_tr‖` is the loop that pins it, and it is
**MEASURED** across all 34,447 probe-selection events: `rms|d|` goes 3.59 → 21.5 (6.0×) while `‖θ_tr‖`
goes 13.6 → 81.4 (6.0×). So: **noise inflates the norm → bigger norm produces bigger `|JVP|` → bigger
absolute step → more inflation.** Multiplicative, unstable at initialization (ρ₀ = 0.113), with no
restoring force. *Why* `|d| ∝ ‖θ_tr‖` is the one part not yet proven — H-B; the §15.1 probe settles it.

Consequences worth stating plainly:

- **Turning `η` down does not fix it.** It scales `ρ` proportionally and immediately, but the loop then
  compounds at the new constant rate. Smaller `η` **delays** divergence; it cannot prevent it.
- **The steps do not become more wrongly aimed over time.** `cos` stays ≈0.023 throughout, because it
  depends on `n` and `p` and neither changes. What grows is the *absolute* step (6×) and the norm it is
  applied to. "Bigger and bigger steps in the wrong direction" is right in absolute terms; the
  *fraction* that is misdirected is constant.
- **`ρ` should be close to `cos`, and it is 5× too big.** That intuition is exactly the criterion: `cos`
  is how much of the step is real, so the step should be no larger than that. We *are* capturing `ρ`
  today — it is computable per commit from `weight_norm` and `update_delta_norm`, already logged. We are
  **not** capturing `cos`, which is why the boundary it should respect is invisible in every existing
  run, and why §15.1 is the first thing to land.
- **Divergence is present at commit 1** and merely takes ~110 commits to become visible. Practically
  this is good news: a config can be scored in minutes rather than 4 hours, and no seed or schedule
  change can rescue it.

> **This retro-explains the S1 NaN.** Heavy-ball at β = 0.9 multiplies the effective relative step by
> `1/(1−β) = 10` → ρ_eff = 1.13 → the norm doubles *every step*. S1 was not the wrong idea; it was
> applied to an already multiplicatively-unstable process. The lesson is ordering, not rejection.

### 9.3 Leg 3 — the variance gate is dimensionally wrong and 100% dead. **MEASURED.**

The gate asks the right question — *have I pooled enough readings to trust this direction yet?* — with
the wrong statistic. From 3,441 `[IterProgress]` lines covering all 186 bins:

```
commit reasons:  natural (var < 0.3) = 0     plateau = 105     cap(max_iter=20) = 81
bins that EVER reached var < 0.3 at any iteration:  0 of 186

achievable variance floor (median per-bin minimum), by 20-bin block:
   bins   0- 20 : 0.415        bins 100-120 : 1.462
   bins  60- 80 : 0.674        bins 140-160 : 3.911
                               bins 180-186 : 14.97      <- 36x drift over the run
```

**36× is exactly (6.0×)².** `var` is a second moment of `d`, and `d ∝ ‖θ_tr‖`, so the gate's ruler grows
as `‖θ‖²` while `var_threshold` stays at 0.3. Three independent reasons the statistic cannot be
repaired by choosing a better number:

1. **It has units.** `var` carries `‖θ‖²`, so no constant is correct for more than an instant. Fixing it
   by scheduling the threshold would be re-tuning per model and per run — exactly what §5 rejects.
2. **It cannot see the directions.** `var(d)` is computed from `P` scalars. The noise this system
   suffers from lives in the `v`'s, and `cos` is *entirely* about directions. A pool can have perfectly
   agreeing `d` values and completely disagreeing `v` directions.
3. **It does not reference the step.** "Enough pooling" is only meaningful relative to how far you
   intend to move. `var < 0.3` knows nothing about `ρ`.

`ρ ≤ cos` is the right criterion on all three counts: dimensionless, direction-aware, and step-aware —
and the split-half cosine (§3.4) is its gradient-free estimator, so the gate and the controller can read
the same number.

**This is a core departure from FwdLLM's design, and should be written up as one.** Variance-controlled
aggregation — pool client updates until the variance of their gradient estimates falls below a
threshold — is FwdLLM's central mechanism for deciding how much to pool. Our position is not that the
threshold is mistuned but that **the statistic cannot support a threshold at all.** Second consequence:
since the natural exit never fires, the two escape hatches *are* the commit policy, which makes
**`max_iterations_per_data_id = 20` — a wall-clock convenience knob — the de-facto controller of half of
`N`.** Not a defensible design position to write up. E-1 (§18) is the ablation that establishes this at
the accuracy level rather than by inference.

### 9.4 The single defect

**Nothing in the pipeline is scale-invariant.** The estimator (`|d| ∝ ‖θ‖`), the step (`‖Δθ‖ ∝ |d|`) and
the gate (`var ∝ |d|²`) all inflate together, so no quantity anywhere can be meaningfully compared
against a fixed constant: `η = 0.01` and `var_threshold = 0.3` mean something different at commit 150
than at commit 1. All three legs are one violation of §5, seen from three places. The fix is not better
constants; it is to make every controlled quantity a ratio.

## 10. Why it looks healthy for three hours

```
rho (MEASURED)         = 0.115
cos(G,g)               = 0.0231       = sqrt(2.991 * 185 / 1040932)
rho / cos              = 5.0x  OVER BUDGET
1/cos^2 = 1881 commits to signal dominance ; norm doubles every 109 -> ~17x too slow to escape
```

`cos = 0.0231` is **ANALYSIS** in its pooling-independence assumption (§12); the `E[v∥²] = 2.991` input
is MEASURED. It says 2.3% of every step is progress and 97.7% is a random kick — **from commit 1, not
from some later point.** The noise is not something that arrives late; it is present in identical
proportion the whole way. Nor does the signal-to-noise *ratio* degrade: `cos` depends on `n` and `p`,
and neither changes during the run. What changes is the accumulated damage.

Three quantities accumulate on three timescales — useful displacement **linearly** in `T` (rate
`ρ·cos`), noise displacement as **√T** (rate `ρ`), norm inflation **exponentially** as `(1+ρ²)^{T/2}`
(rate `ρ²/2`). Early on linear beats √T: the 2.3% that is aligned adds coherently while the noise
partially cancels, so **the accuracy climb is real learning, not a coincidence.** But exponential beats
everything eventually.

**Why a plunge rather than a plateau.** A plateau is the signature of a *noise floor*: steps stop
producing progress and the loss flattens. That is not this failure. Here the damage is multiplicative —
`‖θ‖` compounds, the logits scale with `‖θ‖`, and once the argmax is wrong the cross-entropy of a
saturated softmax grows roughly *linearly in the logit scale*. So loss tracks an exponential. Smooth
rise, rounded peak, accelerating fall is the signature of a linear term overtaken by an exponential one.

**We never reached a minimum and walked back out.** Signal dominance needs ~1,881 commits; we ran 185.
Accuracy peaked at 0.846 because the rising linear term and the falling exponential term crossed —
the peak is a crossover, not an optimum. Classical training does three things we do none of: large
steps far from the optimum, small steps near it, and a stopping rule at it. Here `ρ` is constant (no
decay), its size is set by an accident of `‖θ‖` rather than by distance to the optimum, and **there is
no stopping rule at all.** S-B supplies the decay and S-E supplies a principled "have I pooled enough";
**a convergence-detection rule is still missing from the proposed design** and is an open item (§18).

> **How to read a fluxtune learning curve.** The climb is *not* evidence the configuration is sound,
> and the collapse is *not* a late-appearing bug — both are the same two terms in the order arithmetic
> requires. The only honest health indicators are `ρ` vs `cos` at commit 1, and `‖θ_tr‖`.

## 11. Audit of probe selection: what the probe budget actually buys

**The 2P forward passes are well spent; the probe-selection rule spends them badly.** One probe = one
random direction `v` = the **two** forward passes at `θ±hv` = one scalar `d`. So `P = 10` probes cost 20
forward passes and yield **10 scalars and 10 directions** — 10 independent gradient guesses, of which
the trainer today uploads exactly one.

### 11.1 Magnitude selection is provably stability-neutral. **DERIVED.**

> Picking the probe with the largest `|d|` finds the direction that happened to overlap the gradient
> most — better aim. But `|d|` is also the *scale factor* on the uploaded vector `ĝ = d·v`, so the same
> pick makes the step proportionally longer. You aim 3× better and step 3× further: exact cancellation.

From the §3.1 table, selecting one probe gives `a = E`, `b = √E`, hence `a/b = √E` and **`b²/a = 1`**:

```
cos       ~ sqrt(E[v_par^2]) * sqrt(N/p)      <- probe selection DOES improve alignment
rho/cos   ~ (b^2/a) * p/N = 1 * p/N           <- probe selection does NOT improve stability, at all
```

**Exactly stability-neutral for any `E[v∥²]`, and independent of `P`.** Under the shipped raw-SGD step,
C1 is a genuine ~3× per-commit progress multiplier *and* a ~3× accelerant of the blow-up.

### 11.2 Probe selection has nothing to find. **MEASURED.**

Here is the argument in full, because it is the load-bearing one for C1. At the moment of measurement
`g` is a fixed vector and `v_i` is a Gaussian draw, so `d_i = ⟨g, v_i⟩` is **a fixed linear function of
a Gaussian** — which is exactly Gaussian: `d_i ~ N(0, ‖g‖²)`, independently for each `i`. Every one of
the `P` candidates is an independent draw from the same bell curve. There is no "good probe" hiding in
the batch; there is only the largest sample from a bell curve. And a selection rule sees only the `d`
values, which carry no information about the `v` directions beyond their overlap with `g` — so no
cleverer function of `(d_1..d_P)` can exist.

That is a proof, conditional on `d` actually being Gaussian, which is the thing that could fail if the
finite difference were operating non-linearly over its 50%-of-norm chord. So we checked, over 34,447
events / 344,470 candidate JVPs, by running the *same* order-statistic estimator on the observed data
and on synthetic iid `N(0,1)`:

```
                          observed     synthetic iid Gaussian
random (any 1 of 10)        1.000            —                 (normalisation, by construction)
top-1 of 10                 3.811            3.798            <- match to 0.3%
coin-flip top-2 of 10       2.991            2.987            <- match to 0.1%
pooled skew                +0.0011            0                (kurtosis 2.51 vs 3.0 is an artifact
                                                                of standardising by each event's own
                                                                10-sample rms, not real structure)
```

The test therefore does double duty: **it closes off cleverer probe-selection rules, and it validates
that the finite difference is behaving linearly** despite the large chord (§7).

### 11.3 The k-sweep, settled offline. **MEASURED.**

`FLUXTUNE_CODE_QA.md` §D2 records that the top-k sweep "cannot run today" (k hardcoded to 2, no config
knob). **It does not need to run** — the objective is computable from the already-logged JVPs. For a
rule that *averages* the top-k of P: `a = E_k`, `b = √(E_k/k)`, so `a/b = √(E_k·k)` and `b²/a = 1/k`.
Raising `k` *lowers* `E_k` (weaker probes join) but the `×k` from averaging always wins. All rows cost
the same 20 forward passes and the same dense `p`-vector upload:

| rule | `E_k` | `E_k·k` | `cos` gain | `ρ/cos` improvement |
|---|---|---|---|---|
| random 1 of 10 | 1.000 | 1.00 | 1.00× | 1× |
| **coin-flip top-2, i.e. k=1 with E=2.991 (SHIPPED)** | **2.991** | **2.99** | **1.73×** | **1×** |
| top-1 of 10 | 3.811 | 3.81 | 1.95× | 1× |
| average top-2 | 2.991 | 5.98 | 2.45× | 2× |
| average top-3 | 2.469 | 7.41 | 2.72× | 3× |
| average top-5 | 1.806 | 9.03 | 3.00× | 5× |
| **average all 10 (S-H)** | **1.000** | **10.00** | **3.16×** | **10×** |

**Monotone in k on both objectives; the optimum is k = P: use every candidate.** The shipped rule is
*not* the "average top-2" row — it is a coin flip reporting **one** of the top two, i.e. k=1, with zero
stability improvement. Holding `η`, `N`, `p` fixed and changing only the rule:

```
SHIPPED:      rho = 0.115   cos = 0.0231   rho/cos = 5.0    <- diverges
AVERAGE-ALL:  rho = 0.021   cos = 0.0422   rho/cos = 0.50   <- 2x INSIDE the stability budget
```

> **Averaging the probes you have already computed is a 10× stability improvement for zero additional
> compute, zero additional communication, and no change to η, N or p.** It alone takes the system from
> 5.0× over budget to 2× under it. Among *stable* configurations (ρ normalised to the boundary in
> both) it is **3.34× faster per commit** than selecting.

This reframes the §3.3 cost story in `fluxtune_contributions.md` favourably: the `2P` forward passes
stop being *overhead for a heuristic* and become a **variance-reduction budget with linear, measured
return** (`ρ/cos ∝ 1/P`). "10× the compute of sync fwdllm" is much easier to defend when it buys a
measured 10× in the quantity that governs convergence.

**Caveats, stated honestly:**
- P-averaging attacks **probe noise only** — the P probes share one bin and one θ, whereas `K` attacks
  probe *and* data noise. But probe noise dominates by ~40× (`√(p/N_eff) ≈ 43` vs O(1) for an 8-sample
  bin) — **ANALYSIS** — so P-averaging captures nearly all the available gain. *This also settles
  H2/M1: spend compute on probes, not bigger bins.*
- Memory: accumulate one running sum and discard each `v`, preserving the "peak memory independent of
  P" claim. The code already materialises all P candidates in `v_buffer`.
- The averaged estimator is the **unbiased** one; the shipped one carries a ~3× scale bias — precisely
  the unmatched-effective-LR confound `FLUXTUNE_CODE_QA.md` §E3.1 warns about, now with a number.

### 11.4 Is more `P` worth it, and does it make commits *faster*?

Yes, and the trade is better than it first appears — but only after S-H, because under select-one,
changing `P` moves `cos` and `ρ` by the same factor and does nothing for stability (§11.1). A `P` sweep
on the shipped code would measure nothing.

**More `P` is not slower learning.** Averaging `P` probes does shorten the step, but that is not a cost
— under S-A you then *raise* `ρ` back to the new budget. At the boundary `ρ = cos`, per-commit progress
is `ρ·cos = cos² ∝ P`. **So at the stability boundary, `P = 30` learns 3× faster per commit than
`P = 10`, not slower.** It only looks like a slowdown if `ρ` is left pinned by `η` instead of being set.

**And it makes commits faster in wall clock, which is the sharper point.** `P` and `I` are
*substitutes*: both pool over the same data bin at the same `θ`, so both reduce **probe noise only**.
But their costs are wildly different — `P` costs two extra forward passes on-device, in parallel across
all clients, with no communication; `I` costs a **full round trip** (dispatch, train, upload, aggregate)
per unit. So the pooling stages rank cleanly:

| stage | reduces | marginal cost | verdict |
|---|---|---|---|
| **`P`** | probe noise | 2 forward passes, on-device, parallel, no comms | **buy here first** |
| **`K`** | probe **and** data noise | one more device, parallel; more comms and staleness | **buy here second** |
| **`I`** | probe noise **only** | one serial round trip | **the worst deal: redundant with `P` and the only one that costs wall clock** |

**Raise `P`, lower `I`, hold `n = P·K·I` — commits get strictly faster and no less well-aimed.** This
is the concrete argument for S-D and against the current design, in which `I ≈ 18.5` is set by a
wall-clock cap and is doing work `P` could do for free. The one real ceiling on `P`: past the point
where probe noise falls below data noise (currently ~40× away), `P` stops helping and only `K` does.

### 11.5 The lever nobody has looked at: `p` itself

`cos = (a/b)·√(N/p)`. Every idea above moves `a`, `b`, or `N`. **`p` is the largest number in the
expression and it is a free design choice.** `ρ/cos ∝ p`, so halving `p` is worth exactly as much as
doubling `N` — and unlike `N` it costs no wall clock. From the census in §7, one 768×768
`pre_classifier` layer is **56.7% of the entire probe dimension**:

```
freeze pre_classifier:  p 1,040,932 -> 450,340    cos x1.52    rho/cos improves 2.31x
```

**Half the 5.0× gap closes by removing one layer from the trainable set**, at zero compute cost, with
less communication and less memory. Whether that layer is needed for accuracy is an empirical question
nobody has asked — it is trainable by HuggingFace default, not by design (H-G). Note the side effect
flagged in §2.4: cutting `p` also changes `h‖v‖` from 10.2 to 6.7, so the S-I arm must either normalise
`v` or scale `h` by `√(p_old/p_new)` to keep the finite difference comparable.

> **The general statement, and it is contribution-grade:** for backprop FL, PEFT rank is a *memory and
> communication* knob and gradient quality is unaffected by it. For forward-gradient FL,
> **`cos ∝ 1/√p` — PEFT rank is the primary determinant of gradient quality.** Adapter width is not an
> efficiency detail here; it is an algorithmic parameter. This ties directly to the memory/hardware
> thesis in `fluxtune_contributions.md` §3 and strengthens it: the same choice that makes the method
> fit on the device also makes its gradients better.

### 11.6 The stand this forces on C1

**Do not abandon C1 — redirect it.** Three claims, in descending confidence:

1. **The probe budget is justified.** `2P` forward passes buy a measured `P`-fold improvement in the
   quantity that governs convergence. Linear return, no diminishing.
2. **The *combination* rule is the contribution, not the probe-selection rule.** "Compute P directional
   derivatives and assimilate all of them" is the trainer-side analogue of server-side pooling. The
   estimator itself (multi-point ZO averaging) is known; **the criterion `ρ ≤ cos` and the design rule
   it implies are the contribution.**
3. **If a probe-selection stage survives, it should select on curvature or split-half SNR, not
   magnitude** — magnitude is provably stability-neutral (§11.1) and structure-free (§11.2), while the
   curvature signal is *already computed and discarded*.

## 12. Proven, assumed, and not yet measured

| claim | status |
|---|---|
| step is orthogonal to weights (random walk) | **MEASURED**, ratio 1.032, assumption-free |
| ρ ≈ 0.115 constant; norm growth geometric | **MEASURED**, model predicts final norm to 2.5% |
| `\|d\| ∝ ‖θ_tr‖` (6.0× vs 6.0×) | **MEASURED** (the *why* is open — H-B) |
| variance floor drifts 36× = (6.0×)²; 0/186 natural commits | **MEASURED** |
| `E[v∥²]` = 3.811 (top-1) / 2.991 (shipped coin-flip) | **MEASURED**, n = 34,447 events |
| candidate JVPs indistinguishable from iid Gaussian | **MEASURED**, matches synthetic to 0.3% |
| top-k averaging is monotone in k → use all P | **MEASURED**, offline sweep |
| `cos(G,g) ≈ 0.0231` | **ANALYSIS** — see the assumption below; §15.1 measures it directly |
| criterion `ρ ≤ cos`; `ρ/cos = 5.0×`; the §3.5 sizing table | **DERIVED**, inherits that one caveat |
| does the orthogonality ratio drift within a run? | **NOT MEASURED** — free log replay, H-I |
| does `cos` drift within a run? | **NOT MEASURED** — needs §15.1 |

**The one assumption, stated precisely.** Isotropy of `v` is exact by construction (§2.4) and is not the
caveat. The caveat is **independent, homogeneous pooling**: `cos = (a/b)√(n/p)` assumes the `n` pooled
readings are independent and share one target `g`. In reality the `I` iterations over a bin share the
bin's gradient (so they reduce probe noise but not bin bias) and the `K` trainers have *different*
gradients (so pooling them averages different targets). Both push the effective `n` below the nominal
one. `‖G‖/‖g_probe‖` from §15.1 is the direct test: if it disagrees with `√(p/n)`, the model needs
revisiting before anything is built.

---

# Part III — What to do

## 13. The three pooling stages and who owns them

The natural reading ("it's an aggregation bug") is half right, and acting on that half alone costs a 5×
slowdown.

```
rho      = ||dTheta|| / ||theta_tr||        <-- AGGREGATION owns this (the step rule)
cos(G,g) = (a/b) * sqrt(N/p)                <-- everything else owns this (via P, K, I, and p)
the failure is the RATIO
```

The estimator is pooled at three levels, and each sits with a different component:

| stage | owner | cost structure | reduces |
|---|---|---|---|
| `P` probes per iteration | **trainer** | on-device, parallel, free in wall clock | probe noise |
| `K` trainers per commit | **client selection** (`agg_goal`, `c`) | parallel across devices | probe + data noise |
| `I` iterations per bin | **aggregation** (the variance gate, `FedSgdAggregator.py:450-534`) | **serial round trips** | probe noise only |

> **`N = K × I` is a client-selection knob times an aggregation knob.** That is why C2 and C3 cannot be
> claimed as independent contributions — and why the gate, which decides `I`, is a client-selection
> decision made from aggregation telemetry.

**The asymmetry.** Aggregation can only shrink `ρ`; it can never raise `cos` — nothing on the server
improves the quality of the estimate it was handed, it can only avoid wasting it. Clamping `ρ` from
0.115 → 0.0231 at today's pool buys guaranteed stability but costs **5.0×** in per-commit progress.
Pooling raises `cos`, which raises the `ρ` *budget* for free: growing any pool by 5× gives
`ρ = 0.115/√5 = 0.0514` and `cos = 0.0231·√5 = 0.0516` — they meet, at **exactly** today's `ρ·cos`.
**But pooling alone runs out.** Even at `K = 100`, `I = 20`, `P = 10`, `cos ≈ 0.13`, and with `ρ` held
constant the iterate never converges — it hovers in a noise ball of fixed relative radius.

> **In one line: pooling sets how large `ρ` is *allowed* to be; aggregation *spends* within that budget
> and drives it to zero.**

Three places the coupling is required, not merely convenient:

**(a) The controller's sensor and actuator are on opposite sides.** `ρ` is measured from
`server_update` telemetry (aggregation) and actuated on `K`/`C`/`P`. A pure-selection controller has
nothing to measure — which is why today's `dynamic_kc_policy.py` targets `target_iter_per_data_id: 15`,
a heuristic with no connection to the estimator; a pure-aggregation controller can clamp `ρ` but cannot
tell whether it is clamping harder than necessary. **The strongest argument that C2 + C3 are one
contribution.**

**(b) Normalisation is a prerequisite for the pool sweeps to be *interpretable*.** Under raw SGD,
growing any pool changes `ρ` **and** `cos` at once, so a K- or P-sweep is an unmatched-effective-LR
comparison — the same objection `FLUXTUNE_CODE_QA.md` §E3.1 raises against guided-vs-random. Under a
trust-ratio step `ρ` is pinned and the sweep moves `cos` alone. **You currently cannot run a clean C1 or
C2 ablation, and the blocker is an aggregation-side defect.**

**(c) The commit gate is the `I` half of `N`.** S-E is not a third thing — it is the seam itself.

| | trainer | client selection (C, K) | aggregation (step rule, gate, ω) |
|---|---|---|---|
| S-A trust-ratio step | — | — | **pure** |
| S-B ρ annealing | — | — | **pure — nothing else can do this** |
| S-C closed-loop controller | actuator (P) | actuator (K, C) | sensor (ρ) — **needs both** |
| S-D widen K, shrink I | — | **pure** — only possible because async | releases `I` |
| S-E split-half-cosine gate | — | it *is* the I half of N | statistic is aggregation-side — **the seam** |
| S-H average all P probes | **pure** | — | benefits from S-A to be measurable |
| S-I shrink `p` | model/PEFT design — upstream of all three | | |
| S-J adaptive P per client | **pure** | — | — |
| S-F select on direction, not magnitude | probe selection | — | normalisation — **needs both** |
| S-G weight decay | — | — | pure (hygiene, not a contribution) |

## 14. The algorithm

Three layers, each a specialisation of the one above. This is the constructive statement of everything
in Parts I–II, and it doubles as the target end-state design.

**(a) Perturbation-based training on one device — the core loop.**

```
given: theta, trainable slice theta_tr of dimension p, step schedule rho*_t, budget P
loop t:
  draw P Gaussian probes v_1..v_P            # normalise, or scale h by sqrt(p), so h||v|| is p-independent
  for each i: d_i <- ( L(theta + h v_i) - L(theta - h v_i) ) / (2h)        # 2 forward passes, no_grad
  u <- (1/P) * sum_i d_i * v_i               # ASSIMILATE ALL -- never select on |d| (11.1, 11.2)
  cos_hat <- splithalf_cosine(u)             # gradient-free adequacy estimate (3.4)
  if cos_hat < rho*_t: raise P (or accumulate another bin) and repeat     # criterion enforced online
  theta_tr <- theta_tr - rho*_t * ||theta_tr|| * u / ||u||                # TRUST-RATIO step (3.3)
  rho*_{t+1} <- Robbins-Monro( rho*_t )      # sum rho = inf, sum rho^2 < inf
```

Two invariants make it scale-invariant (§5): the step is a *fraction of `‖θ_tr‖`*, and the stopping
test compares two *dimensionless* quantities. No constant in the loop carries units.

**(b) The same loop in FL, with `N`, `C`, `K`.**

```
SIZING (offline, before the run):
  choose rho*_0 and a safety factor s ~ 0.3-0.5      # 3.3: sit inside the boundary, not on it
  n_req  <- p * (rho*_0 / s)^2 / G_rule              # G_rule = P under averaging   (3.5)
  n_req  = P * K * I  ->  pick I as small as the gate allows, then K = n_req/(P*I)
  require C >= K                                     # concurrency must admit the cohort

PER COMMIT (server):
  dispatch to C clients, wait for K uploads          # async: K arrives, stragglers roll into the next
  G <- sum_k omega_k u_k / sum_k omega_k             # omega re-weights; it must NOT set magnitude
  cos_hat <- splithalf_cosine over the K*I uploads
  if cos_hat < rho*_t and I < I_max: request one more iteration on this bin; continue
  theta_tr <- theta_tr - rho*_t * ||theta_tr|| * G / ||G||
  log rho = ||dTheta||/||theta_tr||, cos_hat, ||theta_tr||     # the three production monitors
  N_target <- N_now * (rho_now / rho*_t)^2           # closed loop; spend it on K before I  (S-C, S-D)
```

**(c) The interplay, as a priority order.** Every decision reduces to *how to reach the required `n`
most cheaply*, and the three stages are not interchangeable:

1. **Shrink `p` first.** It is the only lever that improves aim at *negative* cost, and `ρ/cos ∝ p`.
2. **Then spend `P`.** Free in wall clock, on-device, parallel; reduces probe noise, which dominates.
3. **Then spend `K`.** Parallel across devices; the only stage that reduces *data* noise, so it is what
   you need once probe noise is beaten down. Costs communication and creates staleness.
4. **Spend `I` last, and only what the gate demands.** It buys exactly what `P` buys, and it is the
   only stage that costs a serial round trip.
5. **Set `ρ*` from the achieved `cos`, not from `η`.** Aggregation spends the budget the other three
   created, and anneals it.
6. **Never let probe selection set step magnitude.** It cannot improve stability and it silently biases
   the effective learning rate by ~3×.

## 15. The solution set

Each lands behind a named flag, default = old / byte-identical off, per the flag-gate discipline.

### 15.1 The measurement to land first — one probe answers three questions

`cos(G,g) ≈ 0.0231` is the **only** load-bearing number still resting on an assumption. Every sizing
above depends on it. `FLUXTUNE_CODE_QA.md` §E1 says measuring it needs `v_k` uploaded server-side;
**for this quantity it does not.** `G` is already server-side; you only need *some* `g` to compare
against, and a backprop gradient on a fixed held-out probe batch suffices.

- ~20 lines beside the existing `server_update` telemetry (`FedSgdAggregator.py:338-356`, which already
  carries `‖Δ‖` and `‖W‖`). **No protocol change, server-side only.**
- Log per commit: `cos(G, g_probe)`, `‖G‖/‖g_probe‖`, `‖g_probe‖`, `ρ`, and the split-half cosine.

It settles four things at once: `cos` becomes MEASURED, so every sizing firms up; `‖g_probe‖` vs
`rms|d|` **resolves H-B** (genuine gradient growth vs FD artifact), deciding whether weight decay is
curative; `‖G‖/‖g_probe‖` tests the independent-pooling assumption (§12); and logging the split-half
cosine beside the true `cos` **validates the production estimator against ground truth**, which is what
S-E and S-C will run on.

**Decision rule:** if measured `cos ≈ 0.023`, everything below holds as written. If materially higher,
requirements relax as `∝ 1/cos²` — recompute before building.

### 15.2 S-H. Average all P probes instead of selecting one — **TRAINER · best gain per effort**

```
now:      upload  d_sel * v_sel                (one probe; 9 measurements discarded)
proposed: upload  (1/P) * sum_i d_i * v_i      (all P; same passes, same bytes)
```

**10× improvement in `ρ/cos` at zero additional cost** (§11.3). Takes the system from 5.0× over budget
to 2× under it by itself. Also makes the estimator unbiased, removing the ~3× scale confound that
contaminates every guided-vs-random comparison.

- Site: `tc_transformer_trainer_distribute.py:456-487` + the `g_k = d_k·v_k` emit at `:625-631`.
- Flag: `probe_combine: {select | mean}`, default `select`.
- Sanity check: `ρ` drops by `√(E[v∥²]·P) = 5.5×` on the first commit, and `‖G‖` by the same factor.
- Also fixes the `P=1` crash (`sorted_indices[-2]` on a 1-element list) and the RNG-stream mismatch
  that `FLUXTUNE_CODE_QA` §C3 flags as blocking the C1 ablation — `mean` consumes all P draws.

### 15.3 S-I. Shrink `p` — freeze `pre_classifier` — **MODEL DESIGN · cheapest 2.31× available**

`ρ/cos ∝ p`, and one 768×768 layer is 56.7% of the trainable dimension (§11.5). Freezing it gives
`p: 1,040,932 → 450,340`, `cos ×1.52`, `ρ/cos ×0.433` — at *negative* cost.

- Flag: extend `freeze_layers` / add `trainable_scope: {adapters_head | adapters_only}`.
- **Must ship with `v` normalisation or an `h` rescale**, or the finite difference silently changes with
  `p` (§2.4, §11.5) and the arm becomes uninterpretable.
- Gated on **H-G**: does it cost accuracy?
- Generalises to an adapter-rank sweep — for forward-grad, rank is an *algorithmic* parameter.

### 15.4 S-A. Trust-ratio (relative) server step — **AGGREGATION · highest structural leverage**

```
now:      theta <- theta - eta * G / N_acc
proposed: theta <- theta - rho_star * ||theta_tr|| * G / ||G||
```

*Take a step of a fixed fraction `ρ*` of my own size, in the direction `G` points* — `G/‖G‖` keeps the
direction and discards the magnitude. Makes `ρ` an **operator constant** rather than an emergent
quantity, removing the `|JVP|` scale from the update entirely, which is why it works **regardless of how
H-B resolves**. (Same idea as LARS / trust-region.)

- Site: `FedSgdAggregator.py:322-336` (`_server_update_step`).
- Flag: `server_step_rule: {raw_sgd | trust_ratio}`, default `raw_sgd`.
- Subsumes **S3** — ω can no longer influence step *magnitude*, only relative weighting.
- Sanity check: `‖Δθ‖/‖θ_tr‖` from telemetry equals `rho_star` on every commit.
- **Prerequisite for evaluating S-H, S-I and S-D** (§13b), so land it early even though S-H is the
  bigger single win.

### 15.5 S-E. Scale-invariant commit gate — **THE SEAM**

Replace `var < var_threshold` with the **split-half cosine** of the pool (§3.4). Dimensionless, so
immune to the 36× drift that kills any absolute threshold; direction-aware, unlike `var`; and it **is**
an estimator of `cos(G,g)`, so the gate and the ρ-controller read the *same* number and the commit rule
becomes literally **"commit when `ρ ≤ cos`"**. Gives adaptive `N` per bin and retires plateau + cap from
being the de-facto policy — three heuristics collapse into the criterion.

> **Measurability constraint — S-E cannot land first.** The sampling noise on a *single* commit's
> split-half cosine is `≈ 1/√p = 1.0e-3`, while at today's `cos = 0.023` the signal is only
> `cos_half² ≈ 2.7e-4` — **SNR ≈ 0.3, so one commit measures nothing** (verified by simulation, which
> also reproduces `cos = √(N/p)` to 0.1%). Useful for *analysis* pooled over ~100 commits, useless as a
> per-commit gate. After S-H (`cos ×3.16`) and S-I (`×1.52`), `cos ≈ 0.11` ⇒ signal `≈ 6e-3` against the
> same 1e-3 floor ⇒ **SNR ≈ 6, and the gate becomes measurable.** Gated on S-H + S-I by arithmetic, not
> by preference.

### 15.6 S-B. Anneal ρ\* on a Robbins–Monro schedule — **AGGREGATION · the theory-shaped claim**

Measured `ρ` is **constant**, so `Σρ² = ∞`. A stochastic-approximation method with a non-square-summable
step sequence **provably cannot converge** — it can only random-walk. This is *the* convergence
condition, and the telemetry measures the exact quantity it is about.

> `Σρ_t = ∞`: steps must not shrink so fast that their total length is finite, or you stall before
> arriving — **this is the formal answer to "won't shrinking the step prematurely slow us down?"** It is
> the condition that guarantees you can still reach any distance. `Σρ_t² < ∞`: the *noise* contributions
> (which add as squares) must total to something finite, or the accumulated jitter never settles.
> Together: **go far enough, but eventually go quietly.** A constant `ρ` satisfies the first and fails
> the second — precisely a random walk that never converges.

`ρ*_t = ρ_0/√t` satisfies both. Composes with S-A. **This is the rigorous answer to "why doesn't
lowering η fix it":** lowering `η` scales `ρ` by a constant, and a *constant* `ρ` of any size still has
`Σρ² = ∞`.

### 15.7 S-C. Dynamic K/C as a closed-loop stability controller — **NEEDS BOTH SIDES**

`selector/dynamic_kc_policy.py` is `LANDED-OFF` and targets `target_iter_per_data_id: 15` — a
load-balancing heuristic with no connection to the estimator. **The control target is wrong.** The
server already logs `‖Δθ‖` and `‖W‖`, so `ρ` is free; since `ρ ∝ 1/√N`, `N_target = N_now·(ρ_now/ρ*)²`.
Closed loop, measured setpoint. Genuinely fluxtune-specific: **only the async forward-gradient path has
a free `N` to spend.**

### 15.8 S-D. Widen K, shrink I — **CLIENT SELECTION**

`I` and `P` are substitutes that reduce the same noise at wildly different cost (§11.4), so the current
`I ≈ 18.5` is paying round trips for something `P` gives away. If a pool increase is still needed after
S-H and S-I (it may not be — check measured `ρ/cos` first), buy it with `K`:

| route | wall-clock per commit | commits in 4 h |
|---|---|---|
| more iterations (K=10, `max_iter`↑) | **×the multiplier** | proportionally fewer — strictly worse |
| **wider cohort (K↑, c↑)**, I unchanged | **1×** | unchanged |

**K is device-parallel, I is serial.** In real FL, 50 phones compute simultaneously; each extra
*iteration* is another full round trip. Only the 8-GPU emulation harness pays for K — a strong argument
for running this sweep **in sim**, where parity is established (`simulate_fwdllm.md`). `c = 30`
currently caps K. Larger C/K produces real staleness → H-E and the C3-freshness reversal.

### 15.9 S-J. Adaptive `P` per client, gated on split-half cosine — **TRAINER · C1 redirected**

Raise `P` until the trainer's own split-half cosine across its `P` guesses reaches `ρ*`, then stop. Same
probe budget, spent where the bin is hard instead of uniformly — heterogeneous clients (α = 1 Dirichlet)
have bins of very different difficulty, so a fixed `P = 10` over-spends on easy bins and under-spends on
hard ones. **Strictly after S-H** (§11.4).

**Do not hand-write a decreasing schedule.** "Far from the optimum ⇒ coarse aim is fine" is the standard
argument for **increasing** the pool late (the batch-size ramp-up result), not decreasing it. And `cos`
is **independent of `‖g‖`** — signal and probe noise both scale with the gradient and cancel — so probe
noise does not get relatively worse near a minimum. Data noise does, so the late-stage response is more
`K`. Decreasing `P` late is defensible only as a *consequence* of annealing `ρ` under S-B. **A
hand-written schedule and a measured controller may produce similar curves here; only the controller
survives a change of model, dataset, or `p`.**

Statistics that could set `P`, in order of preference. All must be **dimensionless** (§5).

| statistic | how | assessment |
|---|---|---|
| **split-half cosine across the `P` probes** | split the P guesses in two, average each, cosine between the averages | **preferred.** Gradient-free; free (probes already computed); per-client and per-bin; an estimator of the same `cos` the criterion is written in |
| relative spread of the `d_i` — `var(d)/mean(d²)` | already computed by the gate's machinery | cheapest, but blind to directions (§9.3). Must be the *normalised* form |
| server-side `ρ` and measured `cos` | already logged / §15.1 | ground truth, but arrives one commit late and is global, not per-client |

### 15.10 S-F. Restate C1: direction, not magnitude — **NEEDS BOTH SIDES**

Under S-A the magnitude is discarded by construction. If a probe-selection stage is retained, select on
properties other than magnitude — two of which are already paid for:

| metric | why it is interesting | cost |
|---|---|---|
| **curvature `vᵀHv`, from the sum of the two FD passes** (how sharply the loss curves along `v`; small = a long straight valley, large = a narrow one you overshoot) | the central difference uses only the **difference** of `L(θ+hv)` and `L(θ−hv)`. Their **sum** is the second derivative: `L(+)+L(−)−2L(θ) ≈ h²vᵀHv` — **the curvature is already computed and thrown away.** Select for high `\|d\|` *and* low `vᵀHv` = steep *and* safe to travel far along. Directly serves the trust-region step | **≈ free** — one extra `L(θ)` per iteration, amortised over all P |
| **split-half SNR within the bin** | CE is mean-reduced, but a forward pass yields per-sample losses at no extra cost. Compute `d` on each half of the 8-sample bin and select on *agreement*, not magnitude → directions that generalise rather than fit bin noise | **free** |
| **actual loss decrease at the step scale** — pick `v` minimising `L(θ − ρ*‖θ‖v̂)` | under S-A the step size is known in advance, so select the probe that genuinely lowers the loss *at the displacement you will take*. Trust-region selection rather than derivative selection | 1 extra pass per candidate |

### 15.11 S-G. Weight-decay control arm — **AGGREGATION · hygiene, not a contribution**

`λ ≈ ρ²/2` exactly cancels the measured inflation. ~3 lines. Include as a **control** so the stack has
to prove it beats "just add weight decay". If H-B resolves as *genuine* gradient growth, this is curative
rather than cosmetic and the bar it sets goes up.

### 15.12 Structural ideas not yet costed

Only these can beat the `√(n/p)` barrier; neither is ready to build.

| idea | assessment |
|---|---|
| **block-coordinate probing** — probe one adapter layer at a time, `p → p/L` per probe | promising: `ρ/cos` improves ~`L` while each commit updates `1/L` of the params. Known in ZO optimisation. **Needs analysis before building** |
| **low-rank / subspace probing** — sample `v` in a `q ≪ p` subspace | the only structural route past `√(n/p)`, but needs a good subspace *and* a way to broadcast it (`q·p` floats is prohibitive) |

## 16. Explicitly do NOT do

- **Lower η alone.** Pays 1:1 (§4) and does not restore square-summability (§15.6).
- **Retune `var_threshold`.** Dimensionally impossible, direction-blind, step-blind (§9.3).
- **Retry momentum before ρ is bounded.** `ρ_eff = ρ/(1−β)` — the NaN is arithmetic.
- **Keep selecting probes by `|d|`.** Stability-neutral by construction (§11.1) and the candidates
  provably carry no other structure in `|d|` (§11.2). Top-k average with k < P is strictly worse.
- **Build the least-squares / min-norm gradient solve** over `{(v_i,d_i)}`. At `P,N ≪ p` it equals the
  average up to scale (`FLUXTUNE_CODE_QA` §E2) — no gain in `cos`.
- **Orthogonalise the P probes or coordinate probes across trainers.** No-ops at `p = 10⁶`: random
  probes are already orthogonal to `1/√p ≈ 0.001`, and `K·P = 500 ≪ p`.
- **Normalise `v` expecting a variance win.** `‖v‖` concentrates to 0.07%, so it is a no-op for `cos`
  and `|d|/‖v‖ ≡ |d|`. *Do* normalise it (or rescale `h`) when `p` changes, for the separate reason in
  §2.4 — that is hygiene, not a fix.
- **Chase momentum in the probe distribution** (bias `v` toward the EMA of committed updates). Elegant —
  momentum where it helps aim, not where it inflates the step — but **computed negative**: the
  accumulated trajectory has `cos ≈ √T·0.023 ≈ 0.23` after 100 commits, so as a control variate it
  removes only `1−cos² ≈ 5%` of the variance.
- **Invest further in ω-direction / inverse-var.** Two orders of magnitude below the problem
  (ω ∈ [0.702, 0.865] against a 5.0× gap). ω-*freshness* is different — critical path after S-D.
- **Shrink `h`.** Pinned between truncation and fp16 cancellation (§7), and H-C says the probe gets
  *relatively smaller* as the run proceeds anyway.

## 17. Order of operations

> **How to execute this without spending a GPU-day per question: `FLUXTUNE_PROBE_PLAN.md`.** Four
> independent workstreams — log replay (no GPU), an offline measurement rig, a single-process trajectory
> replica, and real runs. Items 1–7 map to its workstreams B and C.

1. **Land the `cos(G,g)` probe** (§15.1). Everything is sized off it; it resolves H-B and H-C and
   validates the split-half estimator for free; no protocol change.
2. **Run E-1 and H-I as log replay** (§18) — both are free and both firm up Part II.
3. **Land S-H (average all P probes)** — 10× for ~10 lines, and it unblocks the C1 ablation.
4. **Land S-A (trust-ratio step) behind a flag**, default off — it is what makes S-H's, S-I's and S-D's
   measurements interpretable.
5. **Test S-I / H-G**: freeze `pre_classifier` (with the `h` rescale), 4 h run, compare peak accuracy.
6. **Re-measure ρ/cos with S-H + S-I on**, and re-derive `N_req` from §3.5. If it is already under
   budget, **S-D's pool increase is unnecessary** — spend the margin on a larger `ρ*` or on cutting `I`.
7. **Land S-E (split-half cosine gate)**, which also supplies S-C's measured setpoint, then S-B, S-C,
   the S-D sweep with the staleness histogram (H-E → C3-freshness), S-J, and the curvature probe (H-H).
8. **In parallel, settle H-F**: 4 h `fwdllm` / `fwdllm_plus` runs with `server_update_audit`, ρ computed
   identically. This decides contribution vs. hygiene, which changes the paper's claim structure — do
   not leave it until the end.

**Standing rule for every run from here: 4 h minimum** (§7).

## 18. Hypothesis and experiment ledger

| ID | Hypothesis | Status | Experiment | What changes |
|---|---|---|---|---|
| **H-A** | Root cause is the scale-invariance violation → geometric random walk | **VERIFIED** (3 legs, §9) | — | — |
| **H-B** | `\|d\| ∝ ‖θ_tr‖` is **genuine gradient-norm growth**, not an FD artifact | **OPEN** | log `‖g_backprop‖` on a fixed probe batch beside `rms\|d\|` — **free, same probe as §15.1** | If genuine: norm control is *curative*. If artifact: fix is normalising `v` / relative `h`. **S-A works either way** |
| **H-C** | The FD is not the driver — the probe gets **relatively smaller** over the run (`h‖v‖/‖θ_tr‖`: 0.50 → 0.12) | **SUSPECTED**, favours H-B-genuine | same probe | If confirmed, drop the "shrink h" thread entirely |
| **H-D** | The collapse endpoint is **logit saturation** from the inflated norm | **SUSPECTED** (loss 2.37 ≫ ln 4; acc exactly 0.250, mcc exactly 0.000) | log prediction entropy + logit norm at `agg_eval` | Confirms norm→logits→collapse; makes norm the one production monitor |
| **H-E** | Staleness stays ≤ 1 **only because N is small**; raising K/C produces genuine staleness | **SUSPECTED** (`FLUXTUNE_CODE_QA` §A4) | staleness histogram during the S-D sweep | If true, **C3-freshness moves from "inert" to load-bearing** |
| **H-F** | The instability is **not** fluxtune-specific — `_server_update_step` is shared code | **OPEN** | 4 h `fwdllm`/`fwdllm_plus` runs, ρ computed identically | Decides **contribution vs. cross-baseline hygiene** — materially changes the paper's claim structure |
| **H-G** | `pre_classifier` (56.7% of `p`) is **not needed** for accuracy | **OPEN** | freeze it; 4 h run; compare peak accuracy | If true, **2.31× of the 5.0× gap closes for free** |
| **H-H** | The FD's discarded curvature term `vᵀHv` carries usable signal | **OPEN** | log `L(+)+L(−)−2L(θ)` per candidate; correlate with realised loss decrease | If true, unlocks the only probe-selection metric that is free and not stability-neutral |
| **H-I** | The orthogonality ratio and `ρ` are **stationary within a run** — no early aligned phase | **OPEN** | **free log replay**: report the Leg-1 ratio and `ρ` per 20-commit block instead of run-summed | If the ratio drifts below 1 early, some real descent existed and §9.1's aggregate is hiding it |

**E-1 — the variance gate carries no signal (the ablation §9.3 needs).** Three phases, increasing cost:

1. **Free, log replay.** Regress `log var_at_commit` on `log ‖θ_tr‖` across all 186 bins. Prediction:
   slope ≈ 2, high `R²` — i.e. the gate is measuring the parameter norm, not pooling adequacy. This is
   decisive on its own and needs no GPU.
2. **Free, log replay.** Correlate `var_at_commit` against the realised test-loss change over the next
   eval interval, with `‖θ_tr‖` partialled out. Prediction: ≈ zero partial correlation.
3. **Three 4 h arms.** (a) shipped gate; (b) **no gate** — fixed `I = 18`, matching today's mean `N`;
   (c) dimensionless gate — threshold scaled by `‖θ_tr‖²`. Prediction: **(a) ≈ (b) on both accuracy and
   realised `I`**, establishing that the gate is inert rather than merely mistuned; (c) diverges from
   both, showing the statistic only becomes responsive once it is made scale-invariant.

**Open items with no experiment yet**, carried so they are not lost:

- **No convergence-detection rule exists**, even in the §14 design. S-E answers "have I pooled enough
  for this step"; nothing answers "am I done".
- **The `h`/`p` coupling** (§2.4) is newly identified and unanalysed beyond the S-I interaction.
- **Whether `cos` itself drifts within a run** is unmeasured until §15.1 lands.

---

# Part IV — Generality and future directions

*Speculative relative to Parts I–III. Kept separate so it can be updated as findings land, and so no
claim here leaks into the ones that are measured.*

## 19. What generalises beyond FluxTune

### 19.1 The general model, and where FL enters

Write the criterion with the pooling stages left abstract:

```
cos = sqrt( G_rule * n / p )        n = product of all INDEPENDENT pooling stages
rho <= cos                          the step must not exceed the aim
n_req = p * rho*^2 / G_rule         what any configuration must pool
```

**Nothing in those three lines is federated.** They apply to any optimizer that estimates a gradient
from directional derivatives. FL enters only in *how `n` decomposes and what each factor costs*:

| setting | `n` decomposes as | the expensive factor |
|---|---|---|
| **single device** | `P` × (gradient-accumulation steps) | wall clock, serially |
| **centralized ZO fine-tuning** (e.g. MeZO-style) | `P` × accumulation | wall clock; no communication at all |
| **data-parallel / distributed** | `P` × workers × accumulation | all-reduce bandwidth |
| **federated (here)** | `P` × `K` × `I` | `I` (round trips) and `K` (staleness, heterogeneity) |

So: **the mathematical model is general and the controller is where the paradigm matters.** The
FL-specific parts are precisely (i) that `I` costs a round trip while `P` is free, which is what makes
"raise P, shrink I" a real result rather than an accounting identity; (ii) staleness, which has no
analogue in centralized training; and (iii) client heterogeneity, which is what makes per-client
adaptive `P` (S-J) worth building.

### 19.2 A concrete, testable prediction against centralized ZO

Centralized zeroth-order fine-tuning uses `P = 1` and a very small fixed learning rate. Our criterion
predicts *why*: with `G_rule = 1` and no `K` or `I`, `cos = √(1/p)` ≈ 10⁻³, so the stable relative step
is ~10⁻³ and the method needs ~`1/cos² = p` steps to make coherent progress — which matches the very
long step counts such methods report. **That is a falsifiable prediction about an existing published
method, derivable from our model with no new experiments**, and it would be the strongest single piece
of evidence that the criterion is not an artifact of our stack. Worth checking against published curves
before claiming it.

### 19.3 Is there a general ML contribution here

Honest assessment, in descending confidence:

1. **Yes: the scale-invariance requirement for forward-gradient training** (§5) — the step, the gate and
   the estimator must all be ratios, or the method needs re-tuning per model. This is a *design
   principle*, defensible on its own, and the sharpest general claim we have.
2. **Yes: `p` is a gradient-quality parameter, not just a memory knob** (§11.5). This inverts the
   standard PEFT intuition and applies anywhere forward gradients are used.
3. **Probably: the online control law** — `ρ ≤ cos` with a gradient-free estimator for `cos` — turns an
   asymptotic result into something a scheduler can act on. Novel as *packaging*, not as theory (§3.6).
4. **No: the scaling law itself.** `cos ∝ √(n/p)` is known ZO analysis.

To be a standalone paper this would need three things we do not have: the criterion predicting the
divergence point across **≥3 models and ≥2 tasks**; the split-half estimator shown to **track measured
`cos`** (§15.1 is step one); and the derived controller **beating hand-tuned schedules without
re-tuning** when the model changes. All three are buildable; none is in hand.

### 19.4 Larger models and datacenter fine-tuning

Two observations, both speculative:

- **The case changes from memory to throughput.** On-device, forward-gradient wins because it stores no
  activations. In a datacenter with backprop available, the only argument is that 2 forward passes are
  cheaper than 1 forward + 1 backward and need no activation memory, so a much larger batch fits. That
  is a real but much narrower claim, and `cos ∝ 1/√p` makes it *worse* at scale unless `p` is
  aggressively constrained — which is exactly what PEFT does.
- **MoE is structurally interesting.** Only the active experts contribute to a forward pass, so a probe
  naturally lives in the active subspace: `p_effective` is per-token active parameters, not total. That
  is **block-coordinate probing for free** (§15.12), which is the one structural idea that beats the
  `√(n/p)` barrier. Unanalysed; if it holds it is the most interesting extension on this list.

---

# Part V — Reference

## 20. Claims this supersedes

| Existing claim | Verdict |
|---|---|
| `fluxtune_contributions.md` §8 headline: "never converges — it *oscillates*" | **Superseded.** At 4 h it is a monotone rise then monotone divergence. F4/F5's position-locked collapses are an early *symptom* of the same noise, not the mechanism |
| S1 momentum "REFUTED as-designed" | **Re-framed.** Correctly refuted *at ρ = 0.115*; ρ_eff = 1.13 explains the NaN exactly. Testable once ρ is bounded |
| S2 "variance-gate recalibration" | **Structurally unfixable as scoped** (§9.3) |
| S3 "aggregation-rate tempering / cap ω ≤ 1" | **Correct but ~2 orders of magnitude too small.** ω ∈ [0.702, 0.865]; the problem is 5.0× |
| H1 shuffle, H3 bin-order permutation | **Do not address the mechanism.** Park |
| H2 bin size / M1 sweep | **Direction settled: spend compute on probes, not bigger bins** — probe noise dominates data noise ~40× (§11.3) |
| C1 "guided selection improves accuracy" | **Substantially revised (§11).** Stability-neutral by construction; dominated 3.34× by averaging the same probes. Redirect, don't abandon |
| C3-direction (alignment gate) | **Park.** ≤0.6% weight perturbation vs a 5.0× problem |
| C3-**freshness** (staleness) | **REVERSED — do not park.** The ≤8%-on-15.6% verdict is *conditional on N being small*; `FLUXTUNE_CODE_QA` §A4 says staleness stays ≤1 for a structural reason. **S-D creates precisely the high-C/K regime §A4 names as the one where freshness becomes load-bearing.** Untestable today; critical path immediately after S-D |
| QA §D2 "the k sweep cannot run today" | **Answered offline** (§11.3) — computable from already-logged JVPs. Monotone; optimum k = P |
| QA §E1 "measuring `cos(G,g)` needs `v_k` uploaded" | **Not for this quantity** (§15.1) — a backprop gradient on a probe batch suffices, no protocol change |
| FwdLLM's variance-controlled aggregation | **Core departure.** Not mistuned — the statistic cannot support a threshold at all (§9.3). E-1 is the ablation |

**Telemetry bug — FIXED 2026-08-07.** `tc_transformer_trainer_distribute.py:485` used to log the argmax
under the label `chosen jvp` while the actual pick is the coin-flip result. It now logs the coin-flip
winner as `chosen jvp`, the argmax as `max jvp`, and the index as `chosen idx`. **Runs before that date
carry the old, mislabelled field.** The coin flip itself is real and matches `FLUXTUNE_CODE_QA.md` §D2.

## 21. Reproducing every number

All from `lib/python/examples/fwdllm/experiments/`. Nothing needs a GPU except the model probe.

```bash
RUN=run_20260804_043301_fluxtune_n100_smoke_syn_0_real
```

**Accuracy / stat_utility / staleness** — `agg_eval` and `agg_round` events in
`$RUN/telemetry/aggregator_*.jsonl`.

**Leg 1 (random walk)** — `server_update` events; compare `Σ (‖W_{t+1}‖² − ‖W_t‖²)` against `Σ ‖Δθ_t‖²`
(`weight_norm`, `update_delta_norm`). Expect 1.032. **For H-I, report this per 20-commit block, not
run-summed.**

**Leg 2 (relative step)** — same events, anchored at the measured init trainable norm:

```python
T0   = 20.356                                  # ||theta_tr|| at init, from build_model
base = W[0]**2 - (T0**2 + dW[0]**2)            # ||theta_frozen||^2; cancels out of all deltas
theta_tr_t = sqrt(W[t]**2 - base)
rho_t      = dW[t] / theta_tr_t                # expect ~0.115, flat
pred       = T0 * prod(sqrt(1 + rho_t**2))     # expect within 2.5% of theta_tr_final
```

**`|JVP|` growth, selection gain, Gaussianity, and the k-sweep** — the trainer log carries all 10
candidates per selection:

```bash
grep 'All JVPs sorted by magnitude' $RUN/*trainers.log   # 34,447 lines
# format (runs from 2026-08-07): "All JVPs sorted by magnitude: [...] and chosen
#          jvp: X and max jvp: Y and chosen idx: I for trainer : T for model
#          version: R data-id: D. iteration: I"
# GOTCHAS: "model version" here is actually the ROUND.
#          In runs BEFORE 2026-08-07 there is no `max jvp`/`chosen idx`, and
#          `chosen jvp` holds the ARGMAX, not the coin-flip winner (§20).
#
# per (round,data_id): rms|d| 3.59 -> 21.5 ; mean(max|d|)/rms|d| = 1.81 flat
# per EVENT, normalise the 10 values by that event's own rms, then:
#   E[v_par^2 | top-1]      = mean( max(d)^2 / mean(d^2) )              -> 3.811
#   E[v_par^2 | coin top-2] = mean( (d1^2+d2^2)/2 / mean(d^2) )         -> 2.991
#   top-k average objective = mean( mean(top-k d^2) / mean(d^2) ) * k   -> monotone, 3.81 .. 10.00
# compare against synthetic iid N(0,1) with the SAME estimator -> 3.798 / 2.987 (match to 0.3%)
```

**Leg 3 (variance gate)** — the aggregator log:

```bash
grep -o '\[IterProgress\] data_id=.* force_commit_planned=[A-Za-z]*' $RUN/*aggregator.log
# 3,441 lines / 186 bins. Split on data_id change; last row of each bin is the commit.
# reason = CAP if iter>=19 else natural if var<0.3 else plateau  ->  0 / 105 / 81
# For E-1 phase 1, join var-at-commit against the reconstructed ||theta_tr|| from Leg 2.
```

**Model constants and the `p` census** (needs the `test_fwdllm` env, ~1 min):

```bash
cd /home/dgarg39/flame/lib/python
/coc/scratch/dgarg/miniconda3/envs/test_fwdllm/bin/python -c "
import sys, torch, math; sys.path.insert(0,'.')
from examples.fwdllm.scripts.profile_jvp_opt import build_model
m = build_model(4, 192)
tr = [(n,p) for n,p in m.named_parameters() if p.requires_grad]
fz = [p for _,p in m.named_parameters() if not p.requires_grad]
p  = sum(x.numel() for _,x in tr)
tn = torch.sqrt(sum((x.detach().float()**2).sum() for _,x in tr)).item()
fn = torch.sqrt(sum((x.detach().float()**2).sum() for x in fz)).item()
print(p, tn, fn, 0.01*math.sqrt(p), 0.01*math.sqrt(p)/tn)
g={}
for n,x in tr:
    k='pre_classifier' if 'pre_classifier' in n else ('classifier' if n.startswith('classifier') else 'adapters')
    g[k]=g.get(k,0)+x.numel()
print(g)"
# -> 1040932  20.3562  416.2133  10.203  0.501
# -> {'adapters': 447264, 'pre_classifier': 590592, 'classifier': 3076}
```
