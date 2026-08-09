# FluxTune — divergence root cause, the stability criterion, and the probe-budget question

**Status:** diagnosis verified and revised against the 2026-08-07 ten-arm sim portfolio (§7.1). Written
2026-08-06, updated 2026-08-08 on `dg/fluxtune_opts`. No fix has been built yet.

**Structure.** Part I is the model (pure theory). Part II is what the shipped system measurably does.
Part III is what to change. Part IV is generality. Part V is reference. **Part VI is the live
implementation tracker.** Nothing in I–II is a proposal; nothing in III is a measurement. **If you are
here to build, start at §22.**

Claims are tagged **MEASURED** (telemetry/logs; reproduction in §21), **DERIVED** (arithmetic on
MEASURED inputs), **ANALYSIS** (rests on a stated assumption), or **HYPOTHESIS** (ledger in §18).

**Summary.** The server step carries no scale: `‖G‖` tracks `‖θ_tr‖`, so the *relative* step `ρ` does
not fall as the model inflates, `‖θ_tr‖` compounds geometrically, and at `‖θ_tr‖ ≈ 85` the classifier
degenerates to predicting one class. The stability condition is `ρ ≤ cos(G,g)`; measured `ρ = 0.16` at
commit 1 against `cos ≤ 0.015`, so the shipped config is **≥10× over budget from commit 1**. The sim
portfolio confirmed the two load-bearing scaling laws — `ρ ∝ 1/√(K·I)` to 5%, `ρ ∝ η` to 1% — and found
the mechanism that stops the blow-up *by accident*: at **`K ≥ 20` the variance gate stops hitting its
iteration cap and starts controlling `N`, and because its threshold has units of `‖θ‖²` it grows
`N ∝ ‖θ‖²`, pinning the absolute step and turning geometric norm growth into arithmetic.** Those arms
hold **0.860 accuracy** where `K = 10` collapses to 0.250. That is Robbins–Monro `ρ_t ∝ 1/√t`
implemented by a dimensional bug: marginally non-square-summable, so it defers collapse to ~1,200
commits rather than preventing it. The work is to replace the accident with an explicit step rule and
schedule (S-A + S-B), and to measure `cos` directly (§15.1) before any sizing number is trusted again.

**The open question this leaves — and the answer.** Every replacement so far still ends in an operator
constant, so the method would keep needing a per-model, per-dataset re-tune; `var_threshold` already had
to move 0.1 → 0.3 when the data went heterogeneous. Reading the gate's actual code settles why:
`var = ‖G_A−G_B‖²/(2m) ≈ 2b²‖g‖²/n`, so the setpoint is really a **gradient-scale** constant, and both
`‖θ‖` drift within a run and α across configs move it (§9.3). Inverting that identity yields
**`n_eff = 2·mean(d²)/var`** (§15.13): dimensionless, rule-agnostic, free of `p` *and* of the check
layer, computable from telemetry already in the aggregator. Synthetic pools with known `n` confirm it
recovers `n` to 3%, detects redundancy, and stays **flat over a 100× spread in `‖g‖`** — exactly the
dependence that makes `var_threshold` non-portable. **Measured limit:** it is blind to *directional*
client disagreement (`O(n/p)` on `var`), so it closes the correlation half of the problem and leaves the
≥2.4× `cos` shortfall to §15.1. What it buys is a **scale-free setpoint**: every unit-carrying constant
dies, and the survivors are `s ≈ 0.3–0.5` and the anneal margin `ε` — both O(1) from the derivation —
plus a pooling budget, which is a resource, not a knob.

---

# Part I — The model

*Nothing here depends on our configuration. It is the frame everything else is scored in.*

## 1. What this optimizer is actually doing

### 1.1 One probe, one scalar

| | backprop | forward-gradient (here) |
|---|---|---|
| cost of one measurement | 1 fwd + 1 bwd ≈ 3 forward-equivalents | **2 forward passes**, no backward |
| what it returns | all `p` components of `g`, exactly | **one scalar** `d = ⟨g,v⟩` — the slope along `v` |
| memory | stores activations | nothing beyond a forward pass |
| the update it produces | `−η·g` — right direction, known length | `−η·d·v` — right **on average**, ~0.1% aligned individually |
| noise sources | data sampling | data sampling **+ which direction you asked about** |

Backprop learns `p` numbers per measurement; we learn **one**. That million-to-one ratio is the entire
cost of not having a backward pass, and every technique here buys some of it back by averaging.
`ĝ = d·v` is unbiased (`E[d·v] = g`; the probe's overlap with `g` enters twice, once in `d` and once in
`v`) and has enormous variance.

*The image used throughout:* you are on a hillside in fog. Backprop feels the slope in every direction
at once. Forward-gradient picks one random direction, takes a test step, feels whether it went up or
down (`d`), steps back. **One reading is nearly worthless** — in a million dimensions your direction is
almost perpendicular to downhill. **A thousand averaged readings are a slope meter.** Vocabulary: *one
reading* = one probe; *how steep did that feel* = `d`; *how much of your step was genuinely downhill* =
`cos(G,g)`.

### 1.2 Four consequences

**(a) The parameter count is the adversary.** A random direction in `p` dimensions overlaps any fixed
target by only ≈ `1/√p`; at `p ≈ 10⁶`, one probe is 0.1% signal. No tuning removes this — it is why `p`
appears in every formula below, and `p` is itself a *lever* (§11.5).

**(b) Averaging is the only real lever, and it pays twice.** Over `n` independent readings the signal is
identical in all of them so it adds **linearly** (`×n`), while near-orthogonal noise adds **in
quadrature** (`×√n`). The average is simultaneously **better aimed** *and* **shorter** — safety improves
as `1/n`, not `1/√n`.

**(c) Misaim never cancels in length.** Downhill parts of each step point the same way and accumulate
linearly in `T`. Random parts accumulate only as `√T` as *displacement* — but as *distance from the
origin* they always add:

> A step `Δ` perpendicular to `θ` gives new norm `√(‖θ‖² + ‖Δ‖²) > ‖θ‖` for **every** perpendicular
> direction; sign and orientation do not matter, only length. With our numbers, `‖θ‖ = 13.35` and
> `‖Δ‖ = 0.16·13.35 = 2.14` → `13.52`: **+1.3% per commit, unconditionally**; ×2 in 55 commits. Whether
> that compounds is decided entirely by whether `‖Δ‖` stays fixed or grows with `‖θ‖` (§9.2).

**The race:** the useful part gets you `T·ρ·cos` of the way there; the wasted part inflates you by
`(1+ρ²)^{T/2}`. `ρ ≤ cos` is that race written down (§3.3).

**(d) Inflated weights destroy a classifier by randomising which class wins, not by saturating it.**
Logits `z = W·h + b` are linear in the weights, so magnitude sets confidence (`logit_norm` 0.07 → ~3)
and *direction* sets which class wins — and direction is randomised by hundreds of orthogonal kicks.
Only the second is the failure. **MEASURED** (§10): at collapse `top_class_share = 1.00` while
prediction entropy stays *high* (0.8–1.3 vs `ln 4 = 1.386`) and `logit_norm` is indistinguishable from a
healthy run's. Degenerate, not confident. The obvious objection — *some of the weight increase was in
the right direction* — is answered by arithmetic: ≤1.5% of each step is aligned, so the coherent part
accumulates to a fraction of one norm over ~190 commits while the norm grows 7×. The learning is real
and it is outvoted. Test loss above `ln(num_classes)` is the fingerprint: **worse than answering "I
don't know."**

## 2. Vocabulary

### 2.1 Two different things are called "selection"

- **probe selection** — trainer-side: given `P` perturbation directions, which to upload. This is what
  C1 is about. Governs `E[v∥²]`.
- **client selection** — the FLAME `selector`: which clients to dispatch to and how many to wait for.
  Governs `K` and `C`.

An unqualified "selection" quoted from another document means client selection.

### 2.2 Symbols

Per-commit unless stated. "Dimensionless" means a pure ratio, so comparing it against a fixed constant
is legitimate — that property is the whole fix (§5).

| symbol | what it is | value here | read it as |
|---|---|---|---|
| `θ` | full weight vector | 67.4M entries | where the model sits |
| `θ_tr` | the **trainable** slice (adapters + heads); rest frozen | `p` = 1,040,932; `‖θ_tr‖` = **13.35** at init | the only thing that can diverge |
| `p` | trainable dims = probe dimension | 1,040,932 | how many directions I could have asked about |
| `v` | one probe direction, raw Gaussian draw, **not** normalised | `‖v‖ = √p ≈ 1020` | the question being asked |
| `h` | finite-difference spacing | 0.01 → displacement `h‖v‖ = 10.2` | **an instrument, not a step size** |
| `d` | the JVP: a **scalar** per probe, `d ≈ ⟨g,v⟩` | rms 3.4 → 15.9 over the run | how steep did that feel |
| `ĝ = d·v` | one trainer's gradient guess (`p`-vector) | — | unbiased, enormous variance |
| `g` | true gradient at `θ` | measured by the §15.1 probe | **not observable in production** |
| `u_k` | one trainer's uploaded update | — | one pooled reading |
| `G` | server's pooled update before the step | — | the direction actually taken |
| `P` | probes per trainer per iteration | 10 | trainer-side pool |
| `K` | trainers pooled per commit (`agg_goal`) | 10 | cohort-side pool |
| `I` | iterations over the **same data bin** before committing; equals `pool_size` exactly | 18.5 at `K`=10 (capped), 8.2 at `K`=50 | gate-side pool |
| `n = P·K·I` | total independent readings behind one commit | ≈1,850 (only 185 used, §11) | the number that sets aim |
| `N = K·I` | uploads pooled server-side | 185 at `K`=10; 300–500 at `K`≥20 | **not** the client population (100) |
| `C` | concurrency pool (`c`) — caps `K` | 30 | — |
| `η` | server learning rate (a **knob**) | 0.01 | — |
| `ω` | per-upload aggregation weight | 0.70–0.87 | a re-weighting, not a step size |
| **`ρ`** | **relative step `‖Δθ‖/‖θ_tr‖`** — an **outcome**, not a knob | **0.16 at commit 1**, then set by whether `N` grows (§9.2) | what fraction of myself I move per commit |
| **`cos(G,g)`** | **fraction of the step aligned with the true gradient** | **≤ 0.015** (95%, split-half; 0.0231 predicted) | how much of that step was downhill — **not** accuracy |
| `E[v∥²]` | probe-selection gain (§2.3) | 2.99 at `P`=10, 4.74 at `P`=30 | dimensionless multiplier |
| `G_rule` | pooling gain of the combination rule: `E[v∥²]` if selecting, `P` if averaging | 2.991 → 10 | §3.5 |
| `a`, `b` | estimator shape constants (§3.1) | — | properties of the **rule**, not the data |
| `var` | commit gate's statistic: spread of `d` across the pool | drifts with `‖θ‖²` | units are the bug — **and the accidental fix at `K`≥20** (§9.3) |
| `ρ*` | the relative step an operator would *set* under S-A | proposed | the knob `ρ` should have been |

### 2.3 Probe-selection gain `E[v∥²]`

Draw `P` probes, keep the largest `|d|`. The kept probe's **squared overlap with the gradient
direction** averages `E[v∥²]` times that of a random probe. Normalised so random = 1.000, it is the
expected largest of `P` draws from a chi-square with one dof — a pure order statistic, fixed by `P` and
the rule, independent of model and data. Top-1 of 10 gives 3.811. It is a *gain*, not a probability, and
it says nothing about stability (§11.1).

### 2.4 `v` is a raw Gaussian draw

Each coordinate ~ `N(0,1)`, `torch.randn_like`, never normalised
(`tc_transformer_trainer_distribute.py:416`). Three consequences:

- **`‖v‖` is essentially constant**, concentrating at `√p ≈ 1020` to 0.07%. Normalising `v` is a **no-op
  for `cos`**.
- **But it sets the probe displacement**, `h‖v‖ = h√p = 10.2`. Nobody chose 10.2; it fell out of the
  parameter count. **Change `p` and the FD spacing silently changes with it** — a live hazard for S-I,
  which cuts `p` 57% and would move `h‖v‖` from 10.2 to 6.7 as a side effect.
- **Isotropy** is exact by construction, and is what licenses `1/√p` in §1.2(a).

## 3. The stability criterion

### 3.1 One upload: two shape constants

Split any upload `u` into its component along `ĝ = g/‖g‖` and the rest: `u = α·ĝ + u⊥`.

- **`a` measures the shadow**: `E[α] = a·‖g‖`. `a = 1` means the average upload carries a
  gradient-component as long as the gradient itself.
- **`b` measures the total length**: `E‖u‖ = b·‖g‖·√p`. The unit is `‖g‖·√p` because the plain one-probe
  estimator `d·v` has that length, so **`b = 1` is "as long as one raw probe"**.

A single upload's aim is `cos(u,g) = (a/b)/√p`. **`a/b` is aim per upload; `b` alone is length.** For
the two rules that matter (setting `‖g‖ = 1`, so `d = v∥`):

| rule | shadow `⟨u,ĝ⟩` | length `‖u‖` | `a` | `b` | `a/b` | `b²/a` |
|---|---|---|---|---|---|---|
| one raw probe `d·v` | `v∥²` | `\|v∥\|·√p` | 1 | 1 | 1 | 1 |
| **select best of P by `\|d\|`** | `v∥²` of winner | `\|v∥\|·√p` of winner | `E` | `√E` | `√E` | **1** |
| **average all P** | `1` (unbiased) | `√(p/P)` | 1 | `1/√P` | `√P` | **1/P** |

The middle row answers "why doesn't picking the best probe help?" — selecting on `|d|` raises the shadow
**quadratically** (`a = E`) and the length **linearly** (`b = √E`). Aim `a/b = √E` genuinely improves,
but stability depends on `b²/a`, in which the two cancel **exactly, for any `E`**.

`a` and `b` are **properties of the rule, known in closed form before the run**. They do not drift. The
one empirical input is that `d` is Gaussian — verified in §11.2.

### 3.2 Pooling: signal linear, noise in quadrature

In `p` dimensions independent random vectors are nearly orthogonal, so `n` of them sum to length `√n`,
not `n`. Pool `n` uploads: the shadow is identical in each and survives untouched; the perpendicular
junk shrinks by `√n`.

```
shadow of the pooled G   ~  a * ||g||                    <- unchanged by pooling
||G||                    ~  b * ||g|| * sqrt(p/n)        <- shrinks as 1/sqrt(n)
cos(G, g)                =  shadow / length = (a/b) * sqrt(n/p)
rho                      ~  eta * ||G|| / ||theta||      <- also shrinks as 1/sqrt(n)
  =>  rho / cos          ~  (b^2/a) * p / n
```

Read the last line as **safety = (a property of the rule) × (dimensions per pooled reading)**. Improve
it via a better rule (`b²/a`), fewer dimensions (`p`), or more readings (`n`).

### 3.3 The criterion

Two clocks run from the start. **The deadline:** coherent progress grows as `T·ρ·cos`, noise
displacement as `√T·ρ`; signal overtakes noise at `T ≈ 1/cos²`. **The budget:** the norm inflates by
`(1+ρ²)^{T/2}`, doubling at `T ≈ 1.4/ρ²`. You survive iff `1/cos² ≤ 1.4/ρ²`. Dropping the constant:

> ## ρ ≤ cos(G, g)
> **The relative step must not exceed the fraction of it that is actually aligned with the gradient.**

**What to target.** Progress per commit is `ρ·cos`, increasing in `ρ`, so you want `ρ` **as large as the
budget allows** — the boundary, not zero. But the derivation drops O(1) constants, `cos` is estimated,
and the risk is violently asymmetric (over-budget fails *exponentially*, under-budget is merely
*linearly* slow). So: **`ρ/cos ≈ 0.3–0.5` early**, then anneal `ρ` toward zero (S-B) so the iterate
settles instead of hovering in a noise ball. A schedule, not a setpoint.

### 3.4 Measuring `cos` without a backward pass

| | how | what it is for |
|---|---|---|
| **predicted** | `cos = (a/b)·√(n/p)` — closed form from config alone | sizing before you run (§3.5) |
| **estimated online, gradient-free** | **split-half cosine**: split the pool, average each half, cosine between them. Halves share only signal, so `cos(a,b) ≈ cos²/2` → `cos ≈ √(2·cos(a,b))` | *intended* as the production control law (S-E) — but **measured unusable at `p = 10⁶`**: per-commit SNR 0.07, run-pooled ≈1 (§9.5). Analysis only, pooled over ~1,000+ commits |
| **ground truth** | manufacture `g` with a real backward pass on a small fixed held-out batch, **server-side only**, once per commit | validating the other two; one-time instrumentation, no protocol change (§15.1) |

The ground-truth probe is affordable precisely because it is not on-device and not per-client.

### 3.5 Sizing a configuration

Collapse the rule into a **pooling gain** `G_rule = (a/b)²`:

```
G_rule = E[v_par^2]   for select-one-of-P     (P does not appear -- see 11.1)
G_rule = P            for average-all-P

cos           = sqrt( G_rule * N / p )        N = K*I uploads pooled server-side
N_req( rho* ) = p * rho*^2 / G_rule           <- pool needed to make rho* safe
```

Sizing table (**DERIVED**), for our `p`:

| config | `G_rule` | `p` | `N` needed at ρ\*=0.16 | at ρ\*=0.05 | at ρ\*=0.02 |
|---|---|---|---|---|---|
| shipped (select 1 of 10) | 2.99 | 1,040,932 | 8,900 | 870 | 139 |
| + S-H (average all 10) | 10 | 1,040,932 | 2,660 | 260 | 42 |
| + S-H + S-I (freeze `pre_classifier`) | 10 | 450,340 | 1,150 | **113** | 18 |

> **This table is a lower bound, not a sizing.** §9.5 puts `cos` at **≤ 0.015 (95%) against the 0.0231
> this formula predicts**, i.e. effective `n` is ≥2.4× below nominal `N = K·I`. Requirements scale as
> `1/cos²`, so every entry is optimistic by ≥2.4× until §15.1 lands. The portfolio *did* confirm the
> **scaling** (`ρ ∝ 1/√N` to 5% over a 1.9× range, §9.2), so **ratios between rows survive; absolute `N`
> does not.**

Today `N = K·I ≈ 185` at `K = 10` (`I` capped at 20), rising to ≈500 at `K ≥ 20` where the gate rather
than the cap sets `I` (§9.3). Two readings: holding `η` fixed, `ρ/cos ∝ 1/N`; holding `ρ` fixed (what
S-A makes possible), `N ∝ ρ*²`. Under S-A the operator sets `ρ*`, so the second is operative.

Translating to deployment knobs: `N = K·I`, and `C ≥ K` must hold. Wall-clock per commit is set by `I`
(serial round trips), not `K` (parallel), so **choose the smallest `I` the gate allows and buy the rest
with `K`** (§13).

### 3.6 What is standard, and what is ours

`ρ ≤ cos` as a single inequality is **ours**. None of its ingredients are:

| ingredient | where it is standard |
|---|---|
| a random probe aligns with a target only as `1/√p` | classical ZO / forward-gradient (Nesterov–Spokoiny; Duchi et al.; Baydin et al.) |
| signal accumulates linearly, noise as `√T` | the standard SGD noise-ball argument |
| convergence needs `Σρ_t = ∞`, `Σρ_t² < ∞` | Robbins–Monro (1951) |
| bound the step **relative to `‖θ‖`** | trust-region; LARS/LAMB |
| "a pool size beyond which more pooling buys nothing" | gradient noise scale / critical batch size (McCandlish et al.) |

**Ours is the packaging:** collapsing those into an inequality **between two quantities the server
already logs or can estimate**, turning an asymptotic rate statement into an *online control law* a
controller can literally evaluate. That is what makes S-C possible. Do not write it up as a new theorem.

> **Half that packaging is unpaid for.** `ρ` is logged exactly per commit; `cos` is **not** measurable
> online at `p = 10⁶` (§9.5). So today the law is enforceable only against a `cos` fixed offline from
> `(a/b)√(N/p)` and validated by a server-side backprop probe (§15.1). Closing that gap is the
> difference between a design rule and a control law, and the single most valuable open problem here.

## 4. The lever table — this ranks every possible fix

**`ρ/cos` is "will this survive"** (under 1 = safe); **`ρ·cos` is "how fast does it learn"**. A good
lever improves the first without hurting the second. Pooling is the only free lever, because it shortens
the step and improves the aim by the same `√n`.

| lever | `ρ/cos` (stability) | `ρ·cos` (progress) | who owns it |
|---|---|---|---|
| **P-averaging** (trainer pooling) | **∝ 1/P** | **invariant — and free in wall clock** | trainer |
| **K** (cohort width) | **∝ 1/K** | invariant — parallel across devices | client selection |
| **I** (iterations per bin) | **∝ 1/I** | invariant — **but serial: one round trip each** | aggregation gate |
| **p** (trainable dimension) | **∝ p** | ∝ 1/p | model/PEFT design |
| `η` learning rate | ∝ η | ∝ η — **pays 1:1** | aggregation |
| probe-selection gain `E[v∥²]` | **invariant** (`b²/a = 1`) | ∝ E | trainer probe selection |
| step normalization | sets ρ to an operator constant | decoupled | aggregation |

Per unit of **compute**: progress ∝ `C/n`, inflation ∝ `C/n²`, so the ratio improves ∝ `n`. **Larger
pools are strictly better for stability per FLOP and slower only in absolute wall-clock progress.**

## 5. The ratio principle

Every quantity compared against a fixed constant must be **scale-invariant**, because anything with
units silently changes meaning as training proceeds. The test: *re-parameterise so `‖θ‖` doubles; the
loss surface shape is unchanged, so the trajectory should be.* Any rule that fails needs re-tuning every
time the model, adapter rank, or round index changes.

**A training rule whose constants are dimensionless does not need re-tuning when the model changes.** In
practice, three replacements: absolute step `η` → relative step `ρ*`; absolute variance threshold →
cosine; absolute iteration cap → measured adequacy condition.

### 5.1 Why the criterion does not have to enumerate its failure modes

The objection to any control law is *"you tuned for the effects you thought of."* `ρ ≤ cos` is closed
against that, because anything not thought of can enter through exactly **two** channels:

- it changes the step actually taken → shows up in **`ρ`**, logged exactly, every commit;
- it degrades the pool → shows up as **`n_eff < n`**, hence in `cos`.

**Channel 2 is only half-covered, and the limit is measured (§15.13).** `n_eff` detects *correlation*
among uploads — duplicated, stale, or otherwise redundant readings — and is exactly invariant to
gradient **scale**. It does **not** detect *directional* disagreement between clients: at `p = 10⁶` the
signal is `1/√p` of an upload's length, so differing `g_k` move `var` by `O(n/p)` and are invisible.
Synthetic pools with 4, 20 and 100 distinct gradient directions all return `n_eff/n ≈ 1.00`.

So the honest scope: **`ρ` is exact, `n_eff` closes the correlation half of channel 2 and makes every
setpoint scale-free, and the disagreement half is still open** — it needs `cos` measured directly
(§15.1), because it is the same signal-under-noise problem that makes the split-half cosine unusable.
Predicting `cos` from `(a/b)√(N/p)` assumes homogeneous independent pooling and is blind to both halves.

## 6. Traps

1. **`ρ` is not `η`.** `η` is a config number; `ρ` is what the system *ends up* doing (§9.2). S-A is
   "make `ρ` the knob instead of `η`."
2. **`h` is not a step size.** `h‖v‖` displaces the weights *to take a measurement*, then steps back.
   Independent of `ρ`; shrinking one does nothing for the other.
3. **`cos(G,g)` is not a model quality score.** It is the aim of *one server step*, and can sit at
   0.01–0.02 while accuracy climbs (§10).
4. **"Unbiased" ≠ "accurate."** `ĝ = d·v` has the right average and catastrophic variance. Probe
   selection is lower-variance-looking but **biased in scale by ≈3×** — the unmatched-effective-LR
   confound in every guided-vs-random A/B so far.
5. **`N` is a pooling count, not a population.** 185 at `K = 10`; the 100-client population is
   irrelevant. `K` is *not* `N`: raising `K` shortens `I` (§9.3).
6. **A drifting `var` is not a data effect.** `var` is the spread of `d`, `d` grows with `‖θ_tr‖`, so the
   gate's ruler grows as `‖θ‖²` (§9.3).

---

# Part II — Diagnosis of the shipped system

## 7. Scope and configuration

| run | mode | duration | commits | outcome |
|---|---|---|---|---|
| `run_20260804_003042_fluxtune_n100_smoke_syn_0_real` | real | 3.93 h | 189 | peak acc 0.846 @ r1/did106 → **0.250 / mcc 0.000 / loss 2.37** @ r2/did38 |
| `run_20260804_043301_fluxtune_n100_smoke_syn_0_real` | real | 3.93 h | 185 | peak acc 0.853 @ r1/did122 → **0.252 / loss 1.54** @ r2/did34 |
| `run_20260805_{110016,130231,150446}_fluxtune_..._real` | real | 1.86–1.88 h | ~95 | stop at did 92–94 — **at the peak, before the collapse** |

Both 4 h runs contain the failure and agree on every derived constant. Deterministic, not a bad seed.

**Why it looked intermittent.** The 2 h runs terminate on `max_runtime_s` after ~95 commits, and
`‖θ_tr‖` doubles every ~68 commits (§9.2) — so they stop just past the first doubling, **exactly where
accuracy peaks.** The old "4 h minimum" rule is replaced by a cheaper, sharper one: **score `‖θ_tr‖²` vs
commit index — linear is safe, super-linear diverges** — readable in ~20 commits at any horizon.

Shipped config, from `aggregator_config.json` of the reference run and
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

**Where the 1.04M probe dimensions live** (**MEASURED**) — this matters a great deal (§11.5):

```
pre_classifier   590,592   56.7%      <- ONE 768x768 layer is over half the probe dimension
adapters         447,264   43.0%
classifier         3,076    0.3%
```

`‖θ_trainable‖` at init = **13.35** (`trainable_weight_norm`, agreeing across all ten arms). The offline
`build_model(4, 192)` probe reports 20.356 and is **not** the production path — every number previously
anchored on 20.356 has been re-derived. `p` is confirmed at 1,040,932, so the probe displacement
`h‖v‖ = 10.203` is **76% of the trainable norm** — a chord across most of the parameter space. Three
observations:

- The resulting `d` values match the exactly-linear model to 0.3% (§11.2), so truncation error is not
  showing up.
- The displacement gets *relatively smaller* over the run (`h‖v‖/‖θ_tr‖`: 0.76 → 0.11) while the failure
  gets *worse* — so the FD is not the villain (H-C).
- `h` cannot be reduced anyway: at fp16 the two forward losses agree to 1–2 sig figs at `h = 0.01`
  (QA §4.2), so **`h` is pinned between truncation error above and catastrophic cancellation below** — a
  real and publishable tension.

The instinct that the *update* should shrink as learning progresses is correct and is S-B — a separate
knob from `h` (trap 2), and it must be *derived*, because premature shrinking stalls you before you
arrive (§15.6).

## 7.1 The 2026-08-07 sim portfolio — ten arms, all falsification, no fix

Run overnight across three nodes per `FLUXTUNE_PROBE_PLAN.md` §4b. Every arm is `--mode sim` with
`--server-update-audit --pool-split-half-audit`, at commit `f03ef09a` / `57993c80`.

**Enactment audit.** `agg_goal`/`c`, `learning_rate` and `perturbation_count` all landed; `P = 30` is
confirmed *trainer-side* by counting candidates in the `All JVPs sorted by magnitude` lines (a trainer
override, by design never in `aggregator_config.json`). Two traps cleared:

- **`select_perturbation_using_jvp: false` in every `aggregator_config.json` is a decoy.** The trainer
  reads its own copy, and `_metadata/baselines.yaml` sets it **true** for `fluxtune`. The |JVP|-selection
  path is what ran. The aggregator-side copy is dead except for `run_sequential.sh` §O's forward-pass
  accounting.
- **Four of the fourteen 08-07 directories are aborted launches** (`003434`, `003555`, `003618`,
  `003842`, at the earlier commit `873d0332`) with ≤1 commit of telemetry. Ignore them.

**Not run:** the `fwdllm_plus` leg — the node-3 chained command carried only two arms. H-F is answered
by `fwdllm` alone.

Scorecard (**MEASURED**; `ρ`, `‖θ_tr‖` from `server_update`, accuracy from `agg_eval`). `I` is read off
`pool_size`, which equals the committed iteration count exactly in every record. `I`, `N` and the second
`ρ` are averaged over **commits 40–80**, a window every arm reaches:

| arm | run | `K`/`c` | `η` | `P` | commits | `I` | `N` | `ρ` c1 → c40-80 | `‖θ_tr‖` end | peak acc | final acc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **anchor** | `004203` | 10/30 | .01 | 10 | 198 | 18.3 | 182 | 0.200 → 0.124 | 91.4 | 0.853 | **0.250** |
| **anchor** | `020030` | 10/30 | .01 | 10 | 191 | 18.5 | 185 | 0.200 → 0.122 | 94.8 | 0.844 | **0.254** |
| K=20 | `031606` | 20/40 | .01 | 10 | 125 | 16.1 | 322 | 0.157 → 0.101 | 28.8 | 0.860 | 0.858 |
| K=30 | `042626` | 30/60 | .01 | 10 | 131 | 10.0 | 299 | 0.159 → 0.099 | 27.1 | 0.860 | 0.857 |
| K=50 | `054135` | 50/100 | .01 | 10 | 148 | 6.8 | 339 | 0.149 → 0.098 | 28.1 | **0.861** | 0.857 |
| η, 8 h | `004228` | 10/30 | .01 | 10 | 346 | 18.8 | 188 | 0.198 → 0.124 | 152.8 | 0.843 | **0.285** |
| η=.002 | `025016` | 10/30 | .002 | 10 | 327 | 18.5 | 184 | 0.040 → 0.036 | 16.0 | 0.815 | 0.808 |
| η=.0005 | `045646` | 10/30 | .0005 | 10 | 338 | 19.2 | 192 | 0.010 → 0.009 | 13.5 | 0.590 | 0.487 |
| **P=30** | `011351` | 10/30 | .01 | 30 | 196 | 18.7 | 187 | 0.248 → 0.163 | 146.0 | 0.848 | **0.276** |
| `fwdllm` | `004453` | 10/10 | .01 | 10 | 131 | 8.9 | 89 | 0.175 → 0.107 | 31.8 | 0.496 | 0.342 |

**`I` is not constant within a run at `K ≥ 20`** — it climbs as `‖θ_tr‖` grows (§9.3), which is why those
arms survive. Run-mean `I` is 16.4 / 12.1 / 8.2 for `K` = 20 / 30 / 50.

**Sim reproduces real.** The `K = 10` anchors land on the reference real run within replicate spread:
peak 0.844–0.853 @ r1/did100–112 → 0.250 @ r2/did40–47 in 191–198 commits, vs real 0.846 @ did106 →
0.250 @ r2/did38 in 189. **Sim is the right place to run this**, and the two anchors give the replicate
spread (±0.005 peak accuracy, ±4% commits) any A/B has to beat.

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

1. **Final loss 2.37 > ln(4).** Answering *wrongly and systematically*, not merely losing signal. What
   kind of wrong is §10.
2. **Train loss diverges too.** `stat_utility` (Oort = `8·rms(batch loss)`) goes **10.84 → 5.31 (did 120)
   → 13.74 (r2 did 35)** = train loss 1.355 → 0.66 → 1.72. **Not overfitting.**
3. **Degradation starts inside round 1** (did 138–148). The round boundary is not the trigger
   (consistent with F10, which already REFUTED the staleness-reset bug).

## 9. Root cause — three legs of one defect, plus two measurements they forced

### 9.1 Leg 1 — every step is orthogonal to `θ`. **MEASURED.**

`‖θ+Δ‖² = ‖θ‖² + 2⟨θ,Δ⟩ + ‖Δ‖²`, so the ratio of observed norm growth to step energy is
`1 + 2⟨θ,Δθ⟩/‖Δθ‖²`, with an exact null at **1.000 = the step carries no component toward or away from
where the model stands**, so its whole length adds by Pythagoras.

```
per 25-commit block, from trainable_weight_norm / trainable_delta_norm (L3):
  K=10   0.999 1.003 1.002 0.998 1.000 1.005 1.001 1.000
  K=50   0.989 1.002 1.009 1.006 1.003 1.007
  fwdllm 1.000 0.993 0.994 1.002 1.006 1.018
```

**1.000 ± 0.005 in every block of every arm — including arms that reach 0.86 accuracy and hold it.**
Closes **H-I** (stationary; no early aligned phase) and corrects the earlier 1.032, which came from
reconstructing `‖θ_tr‖` out of the *total* `‖W‖`.

**Orthogonality is therefore not the defect.** `g` is itself nearly perpendicular to `θ` at `p = 10⁶`,
so `⟨θ,Δθ⟩ ≈ 0` is what *any* optimizer would show — the K=50 arm proves it by learning well with the
identical ratio. The test tells us only that the norm grows by the full `‖Δθ‖` every commit; **whether
that compounds is §9.2.** Do not read Leg 1 as "the update is pure noise."

*Is norm growth typical?* Growth yes — adapters init near zero. Healthy training grows the norm quickly
then *flattens* (gradients shrink; weight decay / normalization pull back). What is atypical at `K = 10`
is that it is geometric and unbounded.

### 9.2 Leg 2 — `ρ ∝ η/√N`, and nothing anneals it. **MEASURED.**

`Δθ = η·G` so `ρ = η‖G‖/‖θ_tr‖`, and §3.2 predicts two exact proportionalities. Both hold:

```
rho ∝ eta   (commit 1, before any drift):   eta .01 -> .002 -> .0005
            rho              0.2004    0.0404    0.0101
            predicted        0.2004    0.0401    0.0100        <- 1% and 0%

rho ∝ 1/sqrt(N)  (commits 40-80, N = K * pool_size):
            K                10        20        30        50
            N               182       322       299       339
            rho * sqrt(N)   1.68      1.80      1.70      1.81  <- invariant to 4% over 1.9x in N
```

**`ρ ∝ 1/√(K·I)` is the most load-bearing claim in this document and it is now MEASURED**, over a curve.
Note what falsified the *naive* version: `ρ ∝ 1/√K` fails badly (0.16 → 0.14 over `K` 10 → 50, not 0.07)
because raising `K` makes the gate commit sooner, cutting `I` from 18.5 to 8.2. It is `N`, not `K`, that
sets the aim (§9.3).

**Where `ρ` at commit 1 comes from, given nobody set it.** `‖G‖ ∝ rms|d|`, and `rms|d|` grows with
`‖θ_tr‖`, so `‖θ_tr‖` largely cancels and what remains is `η` times an estimator constant. The loop is
**MEASURED but sub-linear**: over the `K = 10` run `rms|d|` goes 3.35 → 15.87 (4.7×) while `‖θ_tr‖` goes
13.6 → 94.8 (7.0×) — exponent ≈0.8; in the `η = 0.002` arm, where `‖θ_tr‖` barely moves, `rms|d|` is flat
at 3.34. So the feedback *noise inflates the norm → bigger norm → bigger `|JVP|` → bigger absolute step*
is real but has log-log gain < 1, which is why `ρ` drifts *down* within a run. *Why* `|d|` tracks
`‖θ_tr‖` at all is open — H-B; §15.1 settles it.

**Geometric vs arithmetic growth is decided by the absolute step, visible in one column.** With
`‖θ_{t+1}‖² = ‖θ_t‖² + ‖Δθ_t‖²` (Leg 1), constant `‖Δθ‖` gives `‖θ‖²` **linear** in `t`; growing `‖Δθ‖`
gives geometric. Measured `ρ·‖θ_tr‖ = ‖Δθ‖` by commit window:

```
K=50   2.04  2.04  2.03  2.05  2.03  2.02     <- constant to 1%  => ||theta||^2 linear, +4.16/commit
K=10   2.49  2.53  2.95  4.22  4.81  6.77  11.4  12.4   <- grows  => geometric, doubling ~68 commits
```

Consequences:

- **Turning `η` down does not fix it.** It scales `ρ` immediately, but the loop compounds at the new
  constant rate. The `η = 0.002` arm survives 327 commits at `‖θ_tr‖ = 16.0` but its `‖θ‖²` is still
  linear-with-drift and it pays the full 1:1 progress cost (§13).
- **Steps do not become more wrongly aimed over time.** `cos` depends on `N` and `p`; what grows is the
  *absolute* step and the norm it is applied to.
- **`ρ` should be close to `cos`, and it is ≥10× too big.** `ρ` is captured per commit; `cos` is not
  (§9.5), which is why §15.1 is first.
- **Divergence is present at commit 1** and takes ~150 commits to become visible. A config is scored by
  `‖Δθ‖` and `‖θ_tr‖²` in ~20 commits, not 4 hours.

> **This retro-explains the S1 NaN.** Heavy-ball at β = 0.9 multiplies the effective relative step by
> `1/(1−β) = 10` → ρ_eff = 1.13 → the norm doubles *every step*. S1 was not the wrong idea; it was
> applied to an already multiplicatively-unstable process. The lesson is ordering, not rejection.

### 9.3 Leg 3 — the variance gate is dimensionally wrong, dead at `K = 10`, and the accidental stabiliser at `K ≥ 20`. **MEASURED.**

The gate asks the right question — *have I pooled enough readings to trust this direction?* — with a
statistic that carries units. At `K = 10` it never fires; from 3,441 `[IterProgress]` lines covering all
186 bins of the reference run:

```
commit reasons:  natural (var < 0.3) = 0     plateau = 105     cap(max_iter=20) = 81
bins that EVER reached var < 0.3 at any iteration:  0 of 186

achievable variance floor (median per-bin minimum), by 20-bin block:
   bins   0- 20 : 0.415        bins 100-120 : 1.462
   bins  60- 80 : 0.674        bins 140-160 : 3.911
                               bins 180-186 : 14.97      <- 36x drift over the run
```

**36× is exactly (6.0×)².** `var` is a second moment of `d`, `d` grows with `‖θ_tr‖`, so the gate's ruler
grows as `‖θ‖²` while `var_threshold` stays at 0.3.

**What `var` actually computes — read off the code, not inferred.** `calculate_var`
(`fwdgrad_utils.py:186-211`) splits the pool in half, averages each half, and takes the per-coordinate
variance between the two half-means, averaged over coordinates. For two samples that is exactly
`‖G_A − G_B‖²/(2p)`. Since `cos(G_A,G_B) ≈ 3e-5` (§9.5) the cross term vanishes, and with `u = d·v`,
`‖v‖² ≈ p`:

```
var  =  ||G_A - G_B||^2 / (2p)   ~=   2 * b^2 * ||g||^2 / n
```

**Both legs of §9.3 fall out of that one line.** `var ∝ ‖g‖²` is the units (and, via H-B, the `‖θ‖²`
drift and the accidental anneal); `var ∝ 1/n` is why it is an `N`-controller at all. It is also the
*same two half-means* the §9.5 split-half cosine is built from — the gate measures their **distance**,
L1 measures their **angle**. One statistic, two readings, and only the un-normalised one is measurable
(§15.13).

**The threshold absorbs data heterogeneity too — through scale, not disagreement.** The tempting story
is that `‖G_A − G_B‖²` picks up a between-client disagreement term when the halves hold different `g_k`.
**That is measured false** (§15.13): at `p = 10⁶` the signal is `1/√p` of an upload's length, so
disagreement moves `var` by `O(n/p)`; synthetic pools with 4, 20 or 100 distinct gradient directions are
indistinguishable from one. What actually happens is simpler: `var ≈ 2b²‖g‖²/n`, and a client holding
one of four classes has a **larger local gradient** than a near-IID one, so heterogeneity raises `var`
by raising `‖g‖`. Same conclusion, different mechanism — and a more useful one, because a scale effect
is exactly what a dimensionless statistic cancels.

So `0.3` is compensating for two expressions of one thing: a gradient scale that drifts 36× *within* a
run (via `‖θ‖`) and shifts *across* configs (via α). **That is why a re-tuned threshold cannot port
across `N`, `C`, `K`, dataset, or model, and why `n_eff` — invariant to `‖g‖` over 100× in test — is the
replacement.** Note this is *not* the same phenomenon as the ≥2.4× shortfall in §9.5 and §12: that one
is about disagreement degrading `cos`, which `var` cannot see at all.

**But "dead" is only true at `K = 10`:**

```
                       K=10    K=20    K=30    K=50   |  eta=.002 (K=10)
natural (var<0.3)         0      77     128     148   |      1
plateau                  67      20       4       1   |     31
cap (max_iter=20)        83      28       0       0   |    118
mean commit iteration  17.8    15.4    11.1     7.2   |   18.7
```

Two mechanisms follow, and together they are the finding of the portfolio:

1. **The gate targets `N`, not `I`.** Early in every K-arm realised `N = K·I` lands at 264–296 regardless
   of whether `K` is 20, 30 or 50 — the gate hands back most of a `K` increase as fewer iterations. That
   is **S-D happening automatically**: same aim, `I` cut from 18.5 to 5.9. At `K = 10` the target is
   unreachable, `max_iter` binds, and `N` is pinned at 185.
2. **Because the ruler has units of `‖θ‖²`, holding `var ≤ 0.3` forces `N ∝ ‖θ‖²`, hence `ρ ∝ 1/‖θ‖`,
   hence a constant absolute step and `‖θ‖²` linear in `t`.** Measured at `K = 50`: `I` 5.9 → 10.3 and
   `N` 296 → 513 as `‖θ_tr‖` goes 15.2 → 27.3, with `‖Δθ‖` flat at 2.03.

> **The dimensional bug is the stabiliser.** The wrong units are exactly what turns a fixed threshold
> into a `ρ ∝ 1/‖θ‖` anneal — which, since `‖θ‖ ∝ √t` under a constant absolute step, is Robbins–Monro
> `ρ_t ∝ 1/√t` (S-B) by accident. That is why `K ≥ 20` holds 0.86.
>
> **It is not a fix, for two reasons.** `Σρ_t² = Σc/t` diverges *logarithmically* — the marginal case, so
> growth is deferred not stopped. Extrapolating measured `d(‖θ_tr‖²)/dcommit` (6.07 / 4.27 / 4.16 for
> `K` = 20/30/50) puts degradation onset (`‖θ_tr‖ ≈ 50`) at commit **400–560** and collapse (`≈ 85`) at
> **1,200–1,700** — 8× beyond any arm run so far. And `0.3` hard-codes the trajectory: not portable
> across model, `p`, or adapter rank.

So the earlier verdict — *the statistic cannot support a threshold at all* — is **withdrawn**. What
survives:

1. **It has units**, so the setpoint is a hidden per-model constant even when the loop it closes is the
   right one. That is the §5 violation, and why the good behaviour is an accident.
2. **It cannot see directions.** `var(d)` is computed from scalars; `cos` is entirely about the `v`'s.
3. **It does not reference the step.** `var < 0.3` knows nothing about `ρ`.

`ρ ≤ cos` fixes all three. But the runs also say the target of the replacement is not "a better commit
test" — it is **an explicit `N`-controller with `ρ` as its sensor**, because that is what the var gate
turns out to have been all along. That promotes S-C and demotes S-E.

**Where this leaves the FwdLLM comparison.** Variance-controlled aggregation is FwdLLM's central pooling
mechanism, and the departure is sharper than "the statistic is unusable": it *is* a usable
`N`-controller, but only in a regime FwdLLM never enters (`K ≥ 20`), and only because its constant is
dimensionally wrong in the direction that happens to help. **E-1's prediction — the gate is inert, so
(a) ≈ (b) — is FALSIFIED at `K ≥ 20` and holds only at `K = 10`.**

### 9.4 The single defect

**Nothing in the pipeline is scale-invariant.** The estimator (`|d|` tracks `‖θ‖`), the step
(`‖Δθ‖ ∝ |d|`) and the gate (`var ∝ |d|²`) all inflate together, so no quantity anywhere can be
meaningfully compared against a fixed constant: `η = 0.01` and `var_threshold = 0.3` mean something
different at commit 150 than at commit 1. All three legs are one violation of §5. The K-sweep adds the
twist that decides the fix: **the third violation partly cancels the first two.** When the gate is live
it converts its own drift into a step anneal — which is why the replacement must supply that anneal
*deliberately* (S-A + S-B) rather than merely remove the bug.

### 9.5 `cos` is bounded, not measured — and the bound excludes the prediction. **MEASURED.**

L1 logs the raw split-half components per commit, so `cos ≈ √(2·Σdot / Σ‖a‖‖b‖)` is computable:

```
per-commit split_half_cos:  mean 1.0e-4, sd 1.5e-3   (sd = 1.5 / sqrt(p) -- the sampling floor)
                            => per-commit SNR ~ 0.07, run-pooled SNR ~ 1.0

pooled over all 1,400 K=10 commits (5 arms):  split_half_cos = 3.0e-5 +- 4.0e-5
                                              => cos = 0.008, 95% upper bound 0.015
prediction from sec 3.5:  cos = sqrt(2.991 * 185 / 1040932) = 0.0231  => split_half = 2.7e-4
```

**The prediction sits ~6 SEM above the measurement.** Either the effective `n` is ≥2.4× below the nominal
`N = K·I` (the §12 independence caveat — the more likely of the two), or the split-half estimator is
biased low and cannot be S-E's gate statistic. §15.1 separates them. Until then treat `cos ≤ 0.015` as
operative: **`ρ/cos ≥ 10` at commit 1, not 5.0.**

Also: **the probe plan's "pool over ~100 commits and it is meaningful" is wrong** — a 190-commit run
gives SNR ≈ 1; it takes ~1,400 commits to bound `cos`, and even that is one-sided (§15.5).

### 9.6 H-F — the defect is cross-baseline. **MEASURED, closes H-F.**

`fwdllm` (sync, `K = 10`, `c = 10`), scored identically over 131 commits: orthogonality ratio
**1.000 ± 0.008** per block, `ρ` **0.175** at commit 1, `‖θ_tr‖` 13.55 → 31.8 with `‖Δθ‖` growing, i.e.
geometric, doubling every ~106 commits. **Same signature, same magnitude, shared `_server_update_step`.**

So **S-A and S-B are cross-baseline hygiene, not a fluxtune contribution** — the claim structure moves to
the criterion and the controller (§19.3). One consolation prize in the same data: at matched commits
`ρ·√N` is 1.68 for fluxtune vs 1.01 for fwdllm, which — at matched `‖g‖`, so **ANALYSIS** — puts
fluxtune's `|JVP|` selection at `G_rule ≈ 3` and **FwdLLM's cosine-similarity probe selection at
`G_rule ≈ 1`, i.e. no better than random.** Worth a line in the paper; confirm on the rig (B1) first.

## 10. Why it looks healthy for three hours

```
rho (MEASURED)         = 0.16 at commit 1
cos(G,g)               <= 0.015  (95%, sec 9.5; 0.0231 predicted)
rho / cos              >= 10x  OVER BUDGET
1/cos^2 >= 4400 commits to signal dominance ; norm doubles every ~68 -> far too slow to escape
```

`cos ≤ 0.015` says under 1.5% of every step is progress — **from commit 1, not from some later point.**
The noise does not arrive late, and the SNR *ratio* does not degrade, because `cos` depends only on `N`
and `p`. Of §3.3's three accumulations, early on the linear one (`T·ρ·cos`) beats the `√T` noise, so
**the accuracy climb is real learning**; the inflation term is exponential *only while `ρ` is held up* —
exactly what the gate stops doing at `K ≥ 20`.

**`‖θ_tr‖` is the state variable and it predicts accuracy across every arm.** Pooling the ten arms at
matched commits: healthy at `‖θ_tr‖ ≲ 30`; degradation begins at **45–60**; single-class collapse by
**80–95**. Holds for `P` = 10 and 30, every `K`, every `η`, and `fwdllm`. **The one production monitor
worth wiring to an alarm.**

**What the collapse is — H-D revised.** Not logit saturation. At collapse `top_class_share → 1.00` but
`logit_norm` is 2.5–3.6 — *the same as the healthy `K = 50` arm at 2.86* — and prediction entropy stays
**high** at 0.8–1.3 vs `ln 4 = 1.386`. The model is **degenerate, not confident**: the head's decision
direction is destroyed while its scale is unremarkable (§1.2d). Loss > `ln 4` follows from answering one
class at moderate probability.

**We never reached a minimum and walked back out.** Accuracy peaked because the rising linear term and
the falling term crossed — a crossover, not an optimum. Classical training does three things we do none
of: large steps far from the optimum, small steps near it, and a stopping rule at it. The `K ≥ 20` arms
accidentally acquire the second; **a convergence-detection rule is still missing** (§18).

**So the climb is not evidence the config is sound, and the collapse is not a late-appearing bug.**
Score runs by §17's rule, never by the accuracy curve.

## 11. Audit of probe selection: what the probe budget buys

One probe = one random `v` = **two** forward passes at `θ±hv` = one scalar `d`. So `P = 10` costs 20
forward passes and yields **10 scalars and 10 directions** — 10 independent gradient guesses, of which
the trainer uploads exactly one. **The 2P passes are well spent; the selection rule spends them badly.**

### 11.1 Magnitude selection is provably stability-neutral. **DERIVED, now MEASURED.**

> Picking the largest `|d|` finds the direction that overlapped the gradient most — better aim. But
> `|d|` is also the *scale factor* on `ĝ = d·v`, so the same pick makes the step proportionally longer.
> Aim 3× better, step 3× further: exact cancellation.

From §3.1, `a = E`, `b = √E`, hence **`b²/a = 1`**:

```
cos       ~ sqrt(E[v_par^2]) * sqrt(N/p)      <- probe selection DOES improve alignment
rho/cos   ~ (b^2/a) * p/N = 1 * p/N           <- probe selection does NOT improve stability, at all
```

Under the shipped raw-SGD step, C1 is a genuine ~3× per-commit progress multiplier *and* a ~3×
accelerant of the blow-up. **The `P = 30` arm made this MEASURED** — the cleanest single result in the
portfolio, because `b = √E` is a point prediction with no free parameter:

```
E[coin-flip top-2]   P=10: 2.988   P=30: 4.744    (37k selection events; P=10 matches the
                                                   reference run's 2.991 to 0.1%)
predicted rho ratio  sqrt(4.744 / 2.988) = 1.260
measured rho ratio   0.2477 / 0.2004     = 1.236   <- 2%
```

And the consequence the prediction demanded — *tripling the probe budget must hurt* — is what happened:
`P = 30` learns **faster** per commit (acc 0.79 at commit 40 vs 0.58–0.66) and **collapses sooner**
(doubling 57 vs 68 commits; peak at did 62 vs 112; acc 0.25 by commit 120). Same `ρ/cos`, more of both.
**Do not run a `P` sweep on the shipped selection rule again** — it measures `E`, nothing else.

### 11.2 Probe selection has nothing to find. **MEASURED.**

At the moment of measurement `g` is fixed and `v_i` is Gaussian, so `d_i = ⟨g,v_i⟩` is a fixed linear
function of a Gaussian — exactly Gaussian, `d_i ~ N(0, ‖g‖²)`, iid. There is no "good probe" hiding in
the batch, only the largest sample from a bell curve. And a selection rule sees only the `d` values,
which carry no information about the `v` directions beyond their overlap with `g` — so **no cleverer
function of `(d_1..d_P)` can exist.**

That is a proof conditional on `d` being Gaussian, which could fail if the FD were operating
non-linearly over its 76%-of-norm chord. Checked over 34,447 events / 344,470 candidate JVPs, running
the *same* order-statistic estimator on observed data and on synthetic iid `N(0,1)`:

```
                          observed     synthetic iid Gaussian
random (any 1 of 10)        1.000            —                 (normalisation, by construction)
top-1 of 10                 3.811            3.798            <- match to 0.3%
coin-flip top-2 of 10       2.991            2.987            <- match to 0.1%
pooled skew                +0.0011            0                (kurtosis 2.51 vs 3.0 is an artifact of
                                                                standardising by each event's own
                                                                10-sample rms, not real structure)
```

Double duty: **closes off cleverer probe-selection rules, and validates that the FD behaves linearly**
despite the large chord.

### 11.3 The k-sweep, settled offline. **MEASURED.**

`FLUXTUNE_CODE_QA.md` §D2 records that the top-k sweep "cannot run today" (k hardcoded to 2). **It does
not need to run** — the objective is computable from already-logged JVPs. For averaging the top-k of P:
`a = E_k`, `b = √(E_k/k)`, so `a/b = √(E_k·k)` and `b²/a = 1/k`. Raising `k` lowers `E_k` but the `×k`
from averaging always wins. All rows cost the same 20 forward passes and the same dense upload:

| rule | `E_k` | `E_k·k` | `cos` gain | `ρ/cos` improvement |
|---|---|---|---|---|
| random 1 of 10 | 1.000 | 1.00 | 1.00× | 1× |
| **coin-flip top-2, i.e. k=1 with E=2.991 (SHIPPED)** | **2.991** | **2.99** | **1.73×** | **1×** |
| top-1 of 10 | 3.811 | 3.81 | 1.95× | 1× |
| average top-2 | 2.991 | 5.98 | 2.45× | 2× |
| average top-3 | 2.469 | 7.41 | 2.72× | 3× |
| average top-5 | 1.806 | 9.03 | 3.00× | 5× |
| **average all 10 (S-H)** | **1.000** | **10.00** | **3.16×** | **10×** |

**Monotone in k on both objectives; the optimum is k = P.** The shipped rule is *not* "average top-2" —
it is a coin flip reporting **one** of the top two, i.e. k=1, zero stability improvement. Holding `η`,
`N`, `p` fixed and changing only the rule:

```
SHIPPED:      rho = 0.160   cos <= 0.015   rho/cos >= 10    <- diverges
AVERAGE-ALL:  rho = 0.029   cos <= 0.027   rho/cos >= 1.1   <- ~10x better, but only ~at the boundary
```

> **Averaging probes you have already computed is a 10× improvement in `ρ/cos` for zero additional
> compute, zero communication, and no change to η, N or p.** Among *stable* configurations it is
> **3.34× faster per commit** than selecting. **But** at the measured `cos ≤ 0.015` rather than the
> predicted 0.0231, S-H lands *at* the budget, not inside it: **necessary, no longer sufficient** — ship
> with S-A + S-B or S-I. The 10× ratio is unaffected (it is a property of the rule); only the margin
> moved.

**Caveats:**
- P-averaging attacks **probe noise only** (the P probes share one bin and one θ), whereas `K` attacks
  probe *and* data noise. But probe noise dominates by ~40× (`√(p/N_eff) ≈ 43` vs O(1) for an 8-sample
  bin) — **ANALYSIS** — so P-averaging captures nearly all available gain. *This settles H2/M1: spend
  compute on probes, not bigger bins.*
- Memory: accumulate one running sum and discard each `v`, preserving "peak memory independent of P".
  The code already materialises all P candidates in `v_buffer`.
- The averaged estimator is the **unbiased** one; the shipped one carries a ~3× scale bias — the
  unmatched-effective-LR confound `FLUXTUNE_CODE_QA.md` §E3.1 warns about, now with a number.

### 11.4 Is more `P` worth it, and does it make commits faster?

Yes — but only after S-H, because under select-one, changing `P` moves `cos` and `ρ` by the same factor
(§11.1). Confirmed at cost: the `P = 30` arm bought a 1.24× larger step, 1.24× better aim, and an 8%
shorter time to collapse.

**More `P` is not slower learning.** Averaging shortens the step, but under S-A you then *raise* `ρ` back
to the new budget. At the boundary `ρ = cos`, progress is `ρ·cos = cos² ∝ P`, so **`P = 30` learns 3×
faster per commit than `P = 10`.** It only looks like a slowdown if `ρ` is left pinned by `η`.

**And it makes commits faster in wall clock.** `P` and `I` are *substitutes* — both pool over the same
bin at the same `θ`, both reduce probe noise only — but their costs differ wildly:

| stage | reduces | marginal cost | verdict |
|---|---|---|---|
| **`P`** | probe noise | 2 forward passes, on-device, parallel, no comms | **buy here first** |
| **`K`** | probe **and** data noise | one more device, parallel; more comms and staleness | **buy here second** |
| **`I`** | probe noise **only** | one serial round trip | **worst deal: redundant with `P`, and the only one that costs wall clock** |

**Raise `P`, lower `I`, hold `n = P·K·I` — commits get strictly faster and no less well-aimed.** That is
the concrete argument for S-D. The one real ceiling on `P`: past the point where probe noise falls below
data noise (currently ~40× away), only `K` helps.

### 11.5 The lever nobody has looked at: `p` itself

`cos = (a/b)·√(N/p)`. Everything above moves `a`, `b`, or `N`. **`p` is the largest number in the
expression and it is a free design choice.** `ρ/cos ∝ p`, so halving `p` is worth exactly as much as
doubling `N` — and costs no wall clock. One 768×768 `pre_classifier` layer is **56.7% of `p`**:

```
freeze pre_classifier:  p 1,040,932 -> 450,340    cos x1.52    rho/cos improves 2.31x
```

**2.31× of the ≥10× gap closes by removing one layer from the trainable set**, at zero compute cost, less
communication, less memory. Whether it is needed for accuracy is unasked (H-G) — it is trainable by
HuggingFace default, not by design. Side effect (§2.4): cutting `p` moves `h‖v‖` from 10.2 to 6.7, so the
S-I arm must normalise `v` or scale `h` by `√(p_old/p_new)`.

> **Contribution-grade general statement:** for backprop FL, PEFT rank is a *memory and communication*
> knob and gradient quality is unaffected. For forward-gradient FL, **`cos ∝ 1/√p` — PEFT rank is the
> primary determinant of gradient quality.** The same choice that makes the method fit on the device
> also makes its gradients better (ties to `fluxtune_contributions.md` §3).

### 11.6 The stand this forces on C1

**Do not abandon C1 — redirect it.** In descending confidence:

1. **The probe budget is justified — but only under averaging.** `2P` passes buy a `P`-fold improvement
   in `ρ/cos` *if all P are assimilated*; under selection they buy `√E` of aim and `√E` of step, i.e.
   nothing in stability. **The budget is justified by S-H, not by C1.**
2. **The *combination* rule is the contribution, not the probe-selection rule.** "Compute P directional
   derivatives and assimilate all of them" is the trainer-side analogue of server-side pooling. The
   estimator (multi-point ZO averaging) is known; **the criterion `ρ ≤ cos` and the design rule it
   implies are the contribution.**
3. **If a probe-selection stage survives, select on curvature or split-half SNR, not magnitude** —
   magnitude is provably stability-neutral (§11.1) and structure-free (§11.2), while the curvature signal
   is *already computed and discarded* (§15.10).

## 12. Proven, assumed, and not yet measured

| claim | status |
|---|---|
| step orthogonal to `θ`; ratio 1.000 ± 0.005, stationary, all arms | **MEASURED** — supersedes 1.032; closes H-I |
| `ρ ∝ η` | **MEASURED** at commit 1, to 1%, over 20× in `η` |
| `ρ ∝ 1/√(K·I)` | **MEASURED** to 4%, over 1.9× in `N` — *the* load-bearing law |
| `ρ ∝ 1/√K` (naive form) | **FALSIFIED** — the gate returns a `K` rise as fewer iterations |
| `‖Δθ‖` constant ⇒ `‖θ‖²` linear; growing ⇒ geometric | **MEASURED**, both branches, 1% |
| `rms\|d\|` tracks `‖θ_tr‖` **sub-linearly** (exponent ≈0.8) | **MEASURED** (the *why* is open — H-B) |
| gate live at `K ≥ 20` (77–148 natural commits), dead at `K = 10` (0) | **MEASURED** — supersedes "100% dead" |
| gate targets `N ≈ 280`, and grows `N ∝ ‖θ‖²` | **MEASURED**, `K` = 20/30/50 |
| `E[v∥²]` = 2.988 (P=10 coin-flip) / 4.744 (P=30) | **MEASURED**, 37k events per arm |
| `b = √E` — probe selection scales the step by `√E` | **MEASURED** via P=30, to 2% |
| candidate JVPs indistinguishable from iid Gaussian | **MEASURED**, matches synthetic to 0.3% |
| top-k averaging monotone in k → use all P | **MEASURED**, offline sweep |
| collapse at `‖θ_tr‖` 80–95, onset 45–60, arm-independent | **MEASURED**, 10 arms |
| collapse is single-class **degeneracy**, not logit saturation | **MEASURED** (L4) — revises H-D |
| `cos(G,g) ≤ 0.015` (95%); 0.0231 predicted | **MEASURED** (bound only) — the gap is the open item |
| criterion `ρ ≤ cos`; `ρ/cos ≥ 10×`; the §3.5 sizing table | **DERIVED**, inherits the caveat below |
| does `cos` drift within a run? | **NOT MEASURED** — SNR too low; needs §15.1 |
| `var = ‖G_A−G_B‖²/(2p) ≈ 2b²‖g‖²/n` | **DERIVED from the code** (`fwdgrad_utils.py:186-211`) — gives both the units bug and the `N`-control in one line |
| heterogeneity raises the `var` floor ⇒ forced `var_threshold` 0.1 → 0.3 | **ANALYSIS** on that identity, consistent with the ≥2.4× `n_eff` shortfall; B12 is the test |
| `n_eff` is measurable online, dimensionless, `~1/√n` error | **DERIVED** (§15.13) — replay-validatable today, no new telemetry |

**The one assumption, now the leading suspect.** Isotropy of `v` is exact by construction and is *not*
the caveat. The caveat is **independent, homogeneous pooling**: `cos = (a/b)√(n/p)` assumes the `n`
readings are independent and share one target `g`. In reality the `I` iterations over a bin share the
bin's gradient, and the `K` trainers have *different* gradients. Both push effective `n` below nominal —
and §9.5 measures a shortfall of ≥2.4×, consistent in size. **`‖G‖/‖g_probe‖` from §15.1 is the direct
test**, and it is no longer a precaution: the model already disagrees with the data at ~6σ.

**It is also no longer only an assumption to be validated — it is a quantity to be *used*.** §15.13
turns the shortfall into an online, dimensionless measurement (`n_eff = 2·mean‖u_k‖²/(p·var)`) from
telemetry the aggregator already computes. That converts the caveat from a threat to the sizing tables
into the sensor that makes the control law self-tuning, and it is the only route by which data
heterogeneity stops being a hand-set constant (§5.1, §9.3).

---

# Part III — What to do

## 13. The three pooling stages and who owns them

The natural reading ("it's an aggregation bug") is half right, and acting on that half alone costs a ≥10×
slowdown. **The portfolio settled this empirically:** node 1 bought stability with `N`, node 2 with `η`,
and at matched commits `K ≥ 20` reaches **0.860** while `η = 0.002` reaches **0.601 at commit 120 / 0.815
after 327**. Same stability, one at zero progress cost and one at the full 1:1 cost — exactly §4's table.

```
rho      = ||dTheta|| / ||theta_tr||        <-- AGGREGATION owns this (the step rule)
cos(G,g) = (a/b) * sqrt(N/p)                <-- everything else owns this (via P, K, I, and p)
the failure is the RATIO
```

| stage | owner | cost structure | reduces |
|---|---|---|---|
| `P` probes per iteration | **trainer** | on-device, parallel, free in wall clock | probe noise |
| `K` trainers per commit | **client selection** (`agg_goal`, `c`) | parallel across devices | probe + data noise |
| `I` iterations per bin | **aggregation** (variance gate, `FedSgdAggregator.py:450-534`) | **serial round trips** | probe noise only |

> **`N = K × I` is a client-selection knob times an aggregation knob.** That is why C2 and C3 cannot be
> claimed as independent contributions — and why the gate, which decides `I`, is a client-selection
> decision made from aggregation telemetry.

**The asymmetry.** Aggregation can only shrink `ρ`; it can never raise `cos`. Clamping `ρ` to `cos` at
today's pool buys guaranteed stability at the full ≥10× progress cost. Pooling raises `cos`, which raises
the `ρ` *budget* for free. **But pooling alone runs out:** at `K ≥ 20` it brings the system inside budget
for ~500 commits, then the linearly-growing norm walks it back out (§9.3).

> **In one line: pooling sets how large `ρ` is *allowed* to be; aggregation *spends* within that budget
> and drives it to zero.**

Three places the coupling is required, not merely convenient:

**(a) The controller's sensor and actuator are on opposite sides.** `ρ` is measured from `server_update`
(aggregation) and actuated on `K`/`C`/`P`. A pure-selection controller has nothing to measure — which is
why today's `dynamic_kc_policy.py` targets `target_iter_per_data_id: 15`, a heuristic unconnected to the
estimator; a pure-aggregation controller can clamp `ρ` but cannot tell whether it is clamping too hard.
**The strongest argument that C2 + C3 are one contribution.**

**(b) Normalisation is a prerequisite for pool sweeps to be *interpretable as accuracy A/Bs*.** Under raw
SGD, growing any pool changes `ρ` **and** `cos` at once — an unmatched-effective-LR comparison. Under a
trust-ratio step `ρ` is pinned and the sweep moves `cos` alone. Such a sweep is still fully interpretable
as a *stability* measurement without S-A, since `ρ·√N` and `‖Δθ‖` are readable directly — which is how
the K-sweep settled the pooling law. **So S-A is a prerequisite for a clean C1/C2 accuracy claim, not for
the physics.**

**(c) The commit gate is the `I` half of `N`**, and §9.3 shows it is *already* an `N`-controller, so the
seam is real and occupied. What changes is which fix owns it: not S-E (unmeasurable statistic, §15.5) but
S-C, whose sensor `ρ` is measured exactly per commit.

| | trainer | client selection (C, K) | aggregation (step rule, gate, ω) |
|---|---|---|---|
| S-A trust-ratio step | — | — | **pure** |
| S-B ρ annealing | — | — | **pure — nothing else can do this** |
| S-C closed-loop controller | actuator (P) | actuator (K, C) | sensor (ρ) — **needs both** |
| S-D widen K, shrink I | — | **pure** — only possible because async | releases `I` |
| S-E split-half-cosine gate | — | it *is* the I half of N | aggregation-side — **demoted: not measurable today** |
| S-H average all P probes | **pure** | — | benefits from S-A to be measurable |
| S-I shrink `p` | model/PEFT design — upstream of all three | | |
| S-J adaptive P per client | **pure** | — | — |
| S-F select on direction, not magnitude | probe selection | — | normalisation — **needs both** |
| S-G weight decay | — | — | pure (hygiene, not a contribution) |

## 14. The algorithm

**(a) Perturbation-based training on one device.**

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

Two invariants make it scale-invariant (§5): the step is a *fraction of `‖θ_tr‖`*, and the stopping test
compares two *dimensionless* quantities. No constant carries units. **Caveat: the `cos_hat` line is
aspirational** — at `p = 10⁶` the split-half estimate is below its own sampling floor (§15.5), so until
`p` falls or §15.1 rehabilitates it, the adequacy test must be a target `n` set from `ρ`.

**(b) The same loop in FL.**

```
SIZING (offline, before the run):
  choose rho*_0 and a safety factor s ~ 0.3-0.5      # 3.3: sit inside the boundary, not on it
  n_req  <- p * (rho*_0 / s)^2 / G_rule              # G_rule = P under averaging   (3.5)
  n_req  = P * K * I  ->  pick I as small as the gate allows, then K = n_req/(P*I)
  require C >= K                                     # concurrency must admit the cohort

PER COMMIT (server):
  dispatch to C clients, wait for K uploads          # async: K arrives, stragglers roll into the next
  G <- sum_k omega_k u_k / sum_k omega_k             # omega re-weights; it must NOT set magnitude
  commit when the pool reaches N_target              # NOT a cos_hat test -- unmeasurable today (15.5)
  theta_tr <- theta_tr - rho*_t * ||theta_tr|| * G / ||G||
  log rho = ||dTheta||/||theta_tr||, ||theta_tr||, top_class_share    # the three production monitors
  N_target <- N_now * (rho_now / rho*_t)^2           # closed loop; spend it on K before I  (S-C, S-D)
  rho*_{t+1} <- rho*_0 * t^-(1/2+eps)                # strictly inside Robbins-Monro, not on its edge
```

**(c) Priority order.** Every decision reduces to *how to reach the required `n` most cheaply*, and per
§11.4 the stages are not interchangeable: **`p` first** (the only lever that improves aim at *negative*
cost), **then `P`** (free in wall clock), **then `K`** (the only stage that touches data noise; costs
comms and staleness), **`I` last and only what the gate demands** (buys what `P` buys, at a serial round
trip). Then two rules: **set `ρ*` from the achieved `cos`, not from `η`, and anneal faster than `1/√t`** —
not optional, since a pool large enough to be safe *now* is not safe 1,000 commits later; and **never let
probe selection set step magnitude** — it cannot improve stability and silently biases the effective LR
by ~3×.

## 15. The solution set

Each lands behind a named flag, default = old / byte-identical off.

### 15.1 The measurement to land first — one probe answers four questions

`cos(G,g)` is the **only** load-bearing number still resting on an assumption, and §9.5 puts that
assumption under real strain (bound `≤ 0.015` vs closed form 0.0231, ~6 SEM apart). Every sizing depends
on which is right. `FLUXTUNE_CODE_QA.md` §E1 says measuring it needs `v_k` uploaded server-side; **for
this quantity it does not.** `G` is already server-side; you only need *some* `g` to compare against, and
a backprop gradient on a fixed held-out probe batch suffices.

- ~20 lines beside the existing `server_update` telemetry (`FedSgdAggregator.py:338-356`). **No protocol
  change, server-side only.**
- Log per commit: `cos(G, g_probe)`, `‖G‖/‖g_probe‖`, `‖g_probe‖`, `ρ`, and the split-half cosine.

It settles four things: `cos` becomes MEASURED, so every sizing firms up; `‖g_probe‖` vs `rms|d|`
**resolves H-B**, deciding whether weight decay is curative; `‖G‖/‖g_probe‖` tests the independent-pooling
assumption (§12); and comparison against the split-half number **decides whether the L1 estimator is
usable at all**, which S-E and S-C depend on.

**Decision rule.** If ground-truth `cos ≈ 0.023`, the split-half estimator is biased low and **S-E is
dead**. If ground-truth `cos ≈ 0.008–0.015`, the estimator is sound and **`n_eff` is ≥2.4× below `N`**, so
every sizing here is optimistic by that factor and must be re-derived first. **No branch leaves the
current numbers unchanged.**

### 15.2 S-H. Average all P probes instead of selecting one — **TRAINER · best gain per effort**

```
now:      upload  d_sel * v_sel                (one probe; 9 measurements discarded)
proposed: upload  (1/P) * sum_i d_i * v_i      (all P; same passes, same bytes)
```

**10× improvement in `ρ/cos` at zero additional cost** (§11.3) — the largest single lever, premise now
measured (§11.1). At the measured `cos` it lands the system *at* the budget: **necessary, not sufficient**
— ship with S-A+S-B or S-I. Also makes the estimator unbiased, removing the ~3× scale confound.

- Site: `tc_transformer_trainer_distribute.py:456-487` + the `g_k = d_k·v_k` emit at `:625-631`.
- Flag: `probe_combine: {select | mean}`, default `select`.
- Sanity check: `ρ` drops by `√(E[v∥²]·P) = 5.5×` on the first commit; `‖G‖` by the same factor.
- Also fixes the `P=1` crash (`sorted_indices[-2]` on a 1-element list) and the RNG-stream mismatch that
  `FLUXTUNE_CODE_QA` §C3 flags as blocking the C1 ablation — `mean` consumes all P draws.

### 15.3 S-I. Shrink `p` — freeze `pre_classifier` — **MODEL DESIGN · cheapest 2.31×**

`ρ/cos ∝ p`, and one 768×768 layer is 56.7% of `p` (§11.5). Freezing it: `p: 1,040,932 → 450,340`,
`cos ×1.52`, `ρ/cos ×0.433` — at *negative* cost. The only lever the portfolio did not test.

- Flag: extend `freeze_layers` / add `trainable_scope: {adapters_head | adapters_only}`.
- **Must ship with `v` normalisation or an `h` rescale**, or the FD silently changes with `p` (§2.4) and
  the arm becomes uninterpretable.
- Gated on **H-G**: does it cost accuracy?
- Generalises to an adapter-rank sweep — for forward-grad, rank is an *algorithmic* parameter.

### 15.4 S-A. Trust-ratio (relative) server step — **AGGREGATION · highest structural leverage**

```
now:      theta <- theta - eta * G / N_acc
proposed: theta <- theta - rho_star * ||theta_tr|| * G / ||G||
```

*Take a step of a fixed fraction `ρ*` of my own size, in the direction `G` points.* Makes `ρ` an
**operator constant** rather than an emergent quantity, removing the `|JVP|` scale from the update
entirely — which is why it works **regardless of how H-B resolves**. (Same idea as LARS / trust-region.)

- Site: `FedSgdAggregator.py:322-336` (`_server_update_step`).
- Flag: `server_step_rule: {raw_sgd | trust_ratio}`, default `raw_sgd`.
- Subsumes **S3** — ω can no longer influence step *magnitude*.
- Sanity check: `‖Δθ‖/‖θ_tr‖` equals `rho_star` on every commit.
- **Prerequisite for evaluating S-H, S-I and S-D** (§13b), so land it early.

> **The portfolio adds one important correction: S-A alone does not bound `‖θ‖`.** A constant `ρ*` still
> gives `‖θ‖ ×(1+ρ*²)^{T/2}` — geometric, just slower. The surviving arms held the **absolute** step
> constant (`‖Δθ‖ = 2.03`), i.e. `ρ_t ∝ 1/‖θ_t‖`. So S-A's value is making the step an *operator* quantity
> and removing the `|JVP|` feedback; **boundedness must come from S-B**, and the two must ship together.

### 15.5 S-E. Scale-invariant commit gate — **DEMOTED by the measurement**

The idea is still right in principle: replace `var < var_threshold` with the **split-half cosine** of the
pool — dimensionless, direction-aware, and an estimator of the same `cos` the criterion is written in, so
the commit rule becomes literally *"commit when `ρ ≤ cos`"*.

> **But the measurability constraint is now measured, and worse than estimated.** Per-commit split-half
> cosine has sd `1.5e-3` against a signal of `1e-4`: **SNR ≈ 0.07**, and whole-run pooling gives ≈1
> (§9.5). The earlier arithmetic promised SNR ≈ 6 after S-H + S-I; redone at the measured `cos` and noise
> floor, the same stack gives **SNR ≈ 2 — still unusable as a per-commit gate.**

**So S-E is not the next thing to build, and the seam is not a commit test.** §9.3 shows the var gate is
already a working implicit `N`-controller; the replacement should be an **explicit `N`-controller with `ρ`
as its sensor** (S-C), because `ρ` is measured exactly per commit while `cos` is not measurable at all
today. Revisit S-E only if §15.1 validates the split-half estimator *and* `p` has come down.

> **The wall is specific to the cosine, not to the split-half pair.** Normalising the gate's own
> statistic gives `‖G_A−G_B‖²/(‖G_A‖²+‖G_B‖²) = 1 − cos(G_A,G_B) ≈ 1 − 1e-4` against a `1.5e-3` floor —
> the same wall, because the dimensionless content of that pair *is* the cosine. **§15.13 escapes it by
> taking a different ratio of the same two vectors**, one that is `O(n)` rather than `1 + O(1e-4)`.

### 15.6 S-B. Anneal ρ\* on a Robbins–Monro schedule — **AGGREGATION · now the load-bearing fix**

At `K = 10` measured `ρ` is constant, so `Σρ² = ∞`. A stochastic-approximation method with a
non-square-summable step sequence **provably cannot converge** — it can only random-walk.

> `Σρ_t = ∞`: steps must not shrink so fast that their total length is finite, or you stall before
> arriving — **the formal answer to "won't shrinking the step prematurely slow us down?"**. `Σρ_t² < ∞`:
> the *noise* contributions (which add as squares) must total to something finite, or the jitter never
> settles. **Go far enough, but eventually go quietly.** A constant `ρ` satisfies the first and fails the
> second — precisely a random walk that never converges. **This is also the rigorous answer to "why
> doesn't lowering η fix it":** lowering `η` scales `ρ` by a constant, and a constant `ρ` of any size
> still has `Σρ² = ∞`.

> **The portfolio promoted this from "theory-shaped" to "demonstrated, and demonstrated insufficient at
> the obvious schedule."** The `K ≥ 20` arms implement `ρ_t ∝ 1/‖θ_t‖` by accident, and since a constant
> absolute step makes `‖θ_t‖ ∝ √t`, that *is* `ρ_t = ρ_0/√t`. It works — 0.860 held. But
> `Σ(ρ_0/√t)² = ρ_0²Σ1/t` diverges **logarithmically**, and the arithmetic `‖θ‖²` growth reaches collapse
> at commit ~1,200–1,700. **So `1/√t` defers rather than converges; the schedule must be
> `t^{-(1/2+ε)}`** — inside the Robbins–Monro window on both sides, not on its edge. That design
> conclusion did not exist before the runs. Composes with S-A.

### 15.7 S-C. Dynamic K/C as a closed-loop stability controller — **NEEDS BOTH SIDES**

`selector/dynamic_kc_policy.py` is `LANDED-OFF` and targets `target_iter_per_data_id: 15` — a
load-balancing heuristic with no connection to the estimator. **The control target is wrong.** The server
already logs `ρ` per commit, and since `ρ ∝ 1/√N` (measured to 4%), the law is
`N_target = N_now·(ρ_now/ρ*)²`. Closed loop, measured setpoint. Genuinely fluxtune-specific: **only the
async forward-gradient path has a free `N` to spend.**

> **Promoted to first among the fixes by §9.3.** The var gate is *already* an `N`-controller and the
> `K ≥ 20` arms are its output; S-C is the same loop with the setpoint made explicit and portable instead
> of hidden in `var_threshold = 0.3`. Cheapest real fix: the sensor `ρ` is measured exactly per commit
> (unlike `cos`), the actuator is a selector knob needing no trainer or protocol change, and the predicted
> behaviour is already validated — the gate's realised `N` tracked `‖θ‖²` in three independent arms.

### 15.8 S-D. Widen K, shrink I — **CLIENT SELECTION**

`I` and `P` are substitutes at wildly different cost (§11.4), so the current `I ≈ 18.5` is paying round
trips for something `P` gives away. If a pool increase is still needed after S-H and S-I (check measured
`ρ/cos` first), buy it with `K`:

| route | wall-clock per commit | commits in 4 h |
|---|---|---|
| more iterations (K=10, `max_iter`↑) | **×the multiplier** | proportionally fewer — strictly worse |
| **wider cohort (K↑, c↑)**, I unchanged | **1×** | unchanged |

**K is device-parallel, I is serial.** In real FL, 50 phones compute simultaneously; each extra
*iteration* is another full round trip. Only the 8-GPU emulation harness pays for K — a strong argument
for running this sweep **in sim**, where parity is established (`simulate_fwdllm.md`). Larger C/K produces
real staleness → H-E and the C3-freshness reversal.

> **Already validated, and cheaper than expected.** The K-sweep *is* S-D: at `K = 50` the gate cut `I`
> from 18.5 to 5.9 — **3× fewer serial round trips for a slightly larger `N`** — and the arm reached the
> best peak accuracy in the portfolio (0.861) and held it. Caveat: `K` is not a knob on `N` (the gate
> absorbs most of the increase), so `K` and the gate must be set together — precisely S-C's job.

### 15.9 S-J. Adaptive `P` per client, gated on split-half cosine — **TRAINER · C1 redirected**

Raise `P` until the trainer's own split-half cosine across its `P` guesses reaches `ρ*`, then stop. Same
budget, spent where the bin is hard — heterogeneous clients (α = 1) have bins of very different
difficulty, so fixed `P = 10` over-spends on easy bins. **Strictly after S-H**, and **blocked by the same
measurability wall as S-E**, worse: a split-half over `P = 10` guesses is far noisier than the server's
over `N = 185`. Park until `p` falls or a better statistic appears.

**Do not hand-write a decreasing schedule.** "Far from the optimum ⇒ coarse aim is fine" is the standard
argument for **increasing** the pool late (the batch-size ramp-up result), not decreasing it. And `cos` is
**independent of `‖g‖`** — signal and probe noise both scale with the gradient and cancel — so probe noise
does not get relatively worse near a minimum. Data noise does, so the late-stage response is more `K`.
Decreasing `P` late is defensible only as a *consequence* of annealing `ρ` under S-B. **A hand-written
schedule and a measured controller may produce similar curves here; only the controller survives a change
of model, dataset, or `p`.**

Statistics that could set `P`, in order of preference. All must be **dimensionless**:

| statistic | how | assessment |
|---|---|---|
| **split-half cosine across the `P` probes** | split the P guesses, average each, cosine between | **preferred.** Gradient-free; free; per-client and per-bin; estimates the same `cos` the criterion uses |
| relative spread of the `d_i` — `var(d)/mean(d²)` | already computed by the gate's machinery | cheapest, but blind to directions (§9.3). Must be the *normalised* form |
| server-side `ρ` and measured `cos` | already logged / §15.1 | ground truth, but arrives one commit late and is global, not per-client |

### 15.10 S-F. Restate C1: direction, not magnitude — **NEEDS BOTH SIDES**

Under S-A the magnitude is discarded by construction. If a probe-selection stage is retained, select on
properties other than magnitude — two of which are already paid for:

| metric | why it is interesting | cost |
|---|---|---|
| **curvature `vᵀHv`** | the central difference uses only the **difference** of `L(θ±hv)`. Their **sum** is the second derivative: `L(+)+L(−)−2L(θ) ≈ h²vᵀHv` — **already computed and thrown away.** Select for high `\|d\|` *and* low `vᵀHv` = steep *and* safe to travel far along. Directly serves the trust-region step | **≈ free** — one extra `L(θ)` per iteration, amortised over all P |
| **split-half SNR within the bin** | a forward pass yields per-sample losses at no extra cost. Compute `d` on each half of the 8-sample bin and select on *agreement* → directions that generalise rather than fit bin noise | **free** |
| **actual loss decrease at the step scale** — pick `v` minimising `L(θ − ρ*‖θ‖v̂)` | under S-A the step size is known in advance, so select the probe that genuinely lowers the loss *at the displacement you will take*. Trust-region rather than derivative selection | 1 extra pass per candidate |

### 15.11 S-G. Weight-decay control arm — **AGGREGATION · hygiene, not a contribution**

`λ ≈ ρ²/2` exactly cancels the measured inflation. ~3 lines. Include as a **control** so the stack has to
prove it beats "just add weight decay". If H-B resolves as *genuine* gradient growth, this is curative
rather than cosmetic and the bar goes up.

### 15.12 Structural ideas not yet costed

Only these can beat the `√(n/p)` barrier; neither is ready to build.

| idea | assessment |
|---|---|
| **block-coordinate probing** — probe one adapter layer at a time, `p → p/L` per probe | promising: `ρ/cos` improves ~`L` while each commit updates `1/L` of the params. Known in ZO optimisation. **Needs analysis before building** |
| **low-rank / subspace probing** — sample `v` in a `q ≪ p` subspace | the only structural route past `√(n/p)`, but needs a good subspace *and* a way to broadcast it (`q·p` floats is prohibitive) |

### 15.13 S-K. Measure `n_eff` — the sensor that removes the last threshold — **AGGREGATION · server-side, ~3 lines**

Everything above still needs *one* number the system cannot see: how much pooling it actually got.
`cos = (a/b)√(N/p)` assumes it equals `N`. **Invert the §9.3 identity and it becomes observable:**

```
n_eff  =  2 * mean_k( ||u_k||^2 ) / ( m * var )  =  2 * mean_k( d_k^2 ) / var
```

`m` is the numel of the **check layer** — `calculate_var` averages over that layer's coordinates, and
`grad_for_var_check` carries one layer only (`tc_transformer_trainer_distribute.py:633`), not the whole
trainable slice. Since `u_k = d_k·v_k` with `‖v_k‖² ≈ m`, `m` cancels: **the sensor is independent of
which layer is checked, and `p` never enters.**

**Why this escapes §15.5's wall.** The split-half *cosine* fails because it extracts a `1e-4` signal
from `O(1)` noise, floored at `1/√p`. `n_eff` is a ratio of **two noise energies** — per-upload against
pooled-difference — which is `O(n)`, not `1 + O(1e-4)`. Relative error is `~1/√n ≈ 7%` per commit, with
no `p` in it. Same two half-means; a different, well-conditioned functional of them.

**Properties — MEASURED on synthetic pools with known `n`, before any run:**

| property | result |
|---|---|
| recovers true `n` on an iid pool | **holds** — 198.7/200, 413.5/400 |
| detects redundancy (duplicated uploads) | **holds** — ×2 → 0.49, ×4 → 0.26 |
| invariant to gradient **scale** | **holds** — `n_eff` flat over a 100× spread in `‖g‖` while `var` moves |
| detects **directional** client disagreement | **FAILS** — 4/20/100 distinct gradient directions all give `n_eff/n ≈ 1.00` |

Row 3 is the one that matters for portability: it is exactly the `‖g‖²` dependence that makes
`var_threshold` a per-model, per-α constant, and `n_eff` cancels it. `b²` cancels too, so the estimator
is **rule-agnostic** — valid under `select`, `mean` or any top-k with no constant to re-derive when S-H
lands.

**Row 4 is a real limit and bounds the claim.** At `p = 10⁶` the signal is `1/√p` of an upload's length,
so between-client disagreement perturbs `var` by `O(n/p)` — the same signal-under-noise wall that makes
the split-half cosine unusable (§15.5). **`n_eff` therefore closes the *correlation* half of channel 2,
not the *disagreement* half.** The ≥2.4× shortfall in §9.5 lives in the half `n_eff` cannot see, so
§15.1 is still needed to explain it; `n_eff` does not substitute for B1.

- **Zero new plumbing.** `var` is already computed at `FedSgdAggregator.py:438`; the `u_k` are already
  in `grad_for_var_check_list`. Server-side, no protocol change, no trainer change, no backward pass —
  so unlike §15.1 it does **not** trade away the inference-only-operator-set claim.

- Flag: `n_eff_audit` (emit-only) → then `rho_target_source: {predicted | n_eff}` for the controller.
- **Validate by log replay before trusting it**: `n_eff` must reproduce the measured `ρ·√N` invariant
  (1.68–1.81) and must fall as `K` rises if the pool is genuinely less independent at wider cohorts.

> **What this leaves.** With `n_eff` measured, every unit-carrying constant is gone: `η`,
> `var_threshold`, `max_iterations_per_data_id`, and the plateau pair all die. What survives is `s ≈
> 0.3–0.5` and the anneal margin `ε`, both **O(1) constants from the derivation itself** rather than
> from the data — the character of "use 3σ", not of "`var_threshold = 0.3`" — plus a pooling **budget**,
> which is a resource decision the operator legitimately owns. Note also that `ρ* = s·√(G_rule·N/p)` and
> `N_req = p·ρ*²/G_rule` are the *same equation*: stability fixes a **curve**, not an operating point.
> Where you sit on it is set by devices and round trips, not by tuning.

## 16. Explicitly do NOT do

- **Lower η alone.** Pays 1:1 (§4) and does not restore square-summability (§15.6). Measured: §24.
- **Retune `var_threshold`.** Not because it is inert — it controls `N` at `K ≥ 20` — but because its
  units make the setpoint a per-model constant, *and* it is simultaneously absorbing a heterogeneity
  floor (§9.3), so one number is paying for two unrelated drifts. Retuning fixes whichever one you
  measured against and silently mis-sets the other. Replace the loop (S-C) with a measured sensor
  (S-K); do not re-tune it.
- **Run a `P` sweep under the shipped selection rule.** Measures `E` and nothing else; the GPU-day is
  already spent (§24).
- **Retry momentum before ρ is bounded.** `ρ_eff = ρ/(1−β)` — the NaN is arithmetic.
- **Keep selecting probes by `|d|`.** Stability-neutral by construction (§11.1); the candidates provably
  carry no other structure in `|d|` (§11.2). Top-k average with k < P is strictly worse.
- **Build the least-squares / min-norm gradient solve** over `{(v_i,d_i)}`. At `P,N ≪ p` it equals the
  average up to scale (`FLUXTUNE_CODE_QA` §E2) — no gain in `cos`.
- **Orthogonalise the P probes or coordinate probes across trainers.** No-ops at `p = 10⁶`: random probes
  are already orthogonal to `1/√p ≈ 0.001`, and `K·P = 500 ≪ p`.
- **Normalise `v` expecting a variance win.** `‖v‖` concentrates to 0.07%, so it is a no-op for `cos`. *Do*
  normalise it (or rescale `h`) when `p` changes, for the separate reason in §2.4 — hygiene, not a fix.
- **Chase momentum in the probe distribution** (bias `v` toward the EMA of committed updates). Elegant but
  **computed negative**: the accumulated trajectory has `cos ≈ √T·cos ≈ 0.1–0.23` after 100 commits, so as
  a control variate it removes only `1−cos² ≈ 5%` of the variance.
- **Invest further in ω-direction / inverse-var.** Two orders of magnitude below the problem
  (ω ∈ [0.702, 0.865] against a ≥10× gap). ω-*freshness* is different — critical path after S-D.
- **Shrink `h`.** Pinned between truncation and fp16 cancellation (§7); H-C says the probe gets
  *relatively smaller* over the run anyway.

## 17. Order of operations

> **How to execute without spending a GPU-day per question: `FLUXTUNE_PROBE_PLAN.md`.** Four independent
> workstreams — log replay (no GPU), an offline measurement rig, a single-process trajectory replica, and
> real runs. Its §4b portfolio is done (§7.1). **Per-feature status lives in §22, not here.**

0. **Replay `n_eff` on the ten arms already on disk** (§15.13). No GPU, no code, hours not days. It
   either reproduces the `ρ·√N` invariant — in which case the last unit-carrying constant has a
   measured replacement and every downstream controller can be built against it — or it does not, and
   §15.1 is the only route left. **Cheapest decision in the document; do it before writing anything.**
1. **Land the `cos(G,g)` probe** (§15.1). The only thing that can resolve the 6-σ gap in §9.5; decides
   whether S-E is dead or the sizing tables are. Also resolves H-B and H-C; no protocol change.
   With step 0 done it gains a second job: **`n_eff` vs ground-truth `cos` is the calibration**, and
   `a/b` is then the only closed-form input left.
2. **Land S-H (average all P probes)** — 10× for ~10 lines, premise measured, unblocks the C1 ablation.
   Sanity check on commit 1: `ρ` drops by `√(E·P) = 5.5×`.
3. **Land S-A + S-B together behind one flag**, default off. Not sequentially: S-A alone leaves `‖θ‖`
   geometric (§15.4), and the schedule must be `t^{-(1/2+ε)}`, not the `1/√t` the gate already achieves.
4. **Land S-C** — the explicit `N`-controller, `N_target = N_now·(ρ_now/ρ*)²`, sensor `ρ`. Cheapest real
   fix: no trainer or protocol change, control law already validated by the K-sweep.
5. **Test S-I / H-G**: freeze `pre_classifier` (with the `h` rescale), compare peak accuracy.
6. **Confirm the deferral prediction**: one long `K = 50` arm at ≥32 h vclock (~8 h wall). §9.3 predicts
   degradation onset at commit 400–560, collapse at 1,200–1,700. **The cheapest falsification of the whole
   `‖θ‖`-drives-collapse story** — if that arm holds 0.86 past commit 2,000, the model is wrong somewhere.
7. **Then**: the S-D sweep with the staleness histogram (H-E → C3-freshness), S-J, the curvature probe
   (H-H), `fwdllm_plus`, and S-E only if §15.1 rehabilitates it.

**Standing rule for every run:** score `‖θ_tr‖²` vs commit index and `‖Δθ‖ = ρ·‖θ_tr‖` — both readable in
~20 commits, valid at any horizon. **Accuracy is not a stability signal** at any runtime: the `K ≥ 20`
arms look identical to a fix and are an 8× deferral.

## 18. Hypothesis and experiment ledger

| ID | Hypothesis | Status | Experiment | What changes |
|---|---|---|---|---|
| **H-A** | Root cause is the scale-invariance violation → unbounded norm growth | **VERIFIED**, refined: it is the *absolute* step failing to be held, not `ρ` being pinned (§9.2) | — | — |
| **H-B** | `\|d\|` tracks `‖θ_tr‖` because of **genuine gradient-norm growth**, not an FD artifact | **OPEN, narrowed**: coupling is sub-linear (≈0.8), and `\|d\|` is flat in arms where `‖θ‖` is flat | log `‖g_backprop‖` on a fixed probe batch beside `rms\|d\|` — **free, same probe as §15.1** | If genuine: norm control is *curative*. If artifact: fix is normalising `v` / relative `h`. **S-A works either way** |
| **H-C** | The FD is not the driver — the probe gets **relatively smaller** (`h‖v‖/‖θ_tr‖`: 0.76 → 0.11) | **SUSPECTED**, favours H-B-genuine | same probe | If confirmed, drop the "shrink h" thread |
| **H-D** | The collapse endpoint is logit saturation | **REVISED — saturation half REFUTED.** `‖θ_tr‖` 80–95 predicts collapse in all 10 arms, but `logit_norm` at collapse (2.5–3.6) equals the healthy arm's 2.86 and entropy stays 0.8–1.3 | — | Collapse is **directional degeneracy of the head**; monitor `‖θ_tr‖` + `top_class_share`, not logit norm |
| **H-E** | Staleness stays ≤ 1 **only because N is small**; raising K/C produces genuine staleness | **OPEN** — the K-sweep ran at `c` up to 100 but the histogram was not scored | staleness histogram from the 08-07 K-sweep — **free log replay, data on disk** | If true, **C3-freshness moves from "inert" to load-bearing** |
| **H-F** | The instability is **not** fluxtune-specific | **RESOLVED: shared** (§9.6). `fwdllm` shows ratio 1.000, `ρ` 0.175, doubling ~106 | done | **S-A/S-B are cross-baseline hygiene.** The claim moves to the criterion and controller (§19.3) |
| **H-G** | `pre_classifier` (56.7% of `p`) is **not needed** for accuracy | **OPEN** | freeze it; compare peak accuracy | If true, **2.31× of the ≥10× gap closes for free** |
| **H-H** | The FD's discarded curvature term `vᵀHv` carries usable signal | **OPEN** | log `L(+)+L(−)−2L(θ)` per candidate; correlate with realised loss decrease | If true, unlocks the only probe-selection metric that is free and not stability-neutral |
| **H-I** | The orthogonality ratio is **stationary within a run** | **RESOLVED: stationary**, 1.000 ± 0.005 per block in every arm | done | The ratio is *not* a defect signature — healthy arms score identically |
| **H-J** | The `K ≥ 20` arms **defer** collapse rather than prevent it; `‖θ_tr‖²` linear ⇒ collapse at commit 1,200–1,700 | **PREDICTED** from measured `d(‖θ_tr‖²)/dcommit` | one `K = 50` arm at ≥32 h vclock | If it holds past ~2,000 commits, the `‖θ‖`-threshold model is wrong and the fix ordering changes |
| **H-K** | α raises `var` by raising `‖g‖` (scale), not by client disagreement — so `n_eff` is **flat across α** and a setpoint expressed in `n_eff` needs no per-dataset re-tune | **NARROWED**: the disagreement leg is REFUTED synthetically (§15.13 row 4); the scale leg is confirmed synthetically and untested on real data | B12: α = 0.1 / 1 / 100 at K = 10 and 20. `var` tracks `mean(d²)`; `n_eff` invariant | If `n_eff` is flat over a 1000× α span, **the last per-dataset constant is gone**. If it drifts, something outside gradient scale is moving `var` and the sensor is incomplete |
| **H-L** | `n_eff` from §15.13 is a sound pooling-adequacy estimate (not just an identity) | **PARTLY RESOLVED**: recovers `n` to 3% and detects redundancy on synthetic pools; blind to directional disagreement | B11 replay: does `n_eff` reproduce `ρ·√N` = 1.68–1.81 and fall as `K` rises | Sound ⇒ every setpoint becomes scale-free. It does **not** explain the §9.5 shortfall, so B1 stays on the critical path |

**E-1 — REWRITTEN.** The original hypothesis (*the gate is inert; (a) ≈ (b)*) is **falsified at `K ≥ 20`**
and holds only at `K = 10`. The question now is the opposite one — *how good an `N`-controller is it, and
what does the setpoint cost?*

1. **Free, log replay.** Regress realised `N` on `‖θ_tr‖²` across `K` = 20/30/50. Prediction: slope ≈ 1,
   high `R²` — the gate holds `N/‖θ‖²` fixed, which is why `‖Δθ‖` is constant. Makes the "accidental S-B"
   claim measured rather than inferred.
2. **Free, log replay.** Correlate `var_at_commit` against the realised test-loss change over the next
   eval interval, with `‖θ_tr‖` partialled out. Prediction: ≈ zero partial correlation — the gate controls
   `N` without carrying information about step quality.
3. **Two arms.** (a) `K = 50` with the gate; (b) `K = 50` with `I` frozen at its early value (5.9), which
   removes the anneal while holding `N`. Prediction: **(b) diverges like `K = 10`**, isolating the anneal
   as the cause rather than the larger pool. If (b) survives, credit belongs to `N` alone and S-B's
   urgency drops.

**Open items with no experiment yet:**

- **No convergence-detection rule exists**, even in the §14 design. Nothing answers "am I done".
- **The `h`/`p` coupling** (§2.4) is newly identified and unanalysed beyond the S-I interaction.
- **Whether `cos` drifts within a run** is unmeasured — split-half SNR far too low; needs §15.1.
- **Why measured `cos` falls ≥2.4× short of `(a/b)√(N/p)`** — the §12 caveat, quantified and unexplained.
  `fwdllm_plus` was never run.
- **Why the offline builder's `‖θ_tr‖` (20.356) differs from production's (13.35)** while `p` matches
  exactly. Cheap to settle (dump per-group norms both ways) and worth doing before B5.

---

# Part IV — Generality and future directions

*Speculative relative to Parts I–III.*

## 19. What generalises beyond FluxTune

### 19.1 The general model, and where FL enters

```
cos = sqrt( G_rule * n / p )        n = product of all INDEPENDENT pooling stages
rho <= cos                          the step must not exceed the aim
n_req = p * rho*^2 / G_rule         what any configuration must pool
```

**Nothing in those three lines is federated.** They apply to any optimizer estimating a gradient from
directional derivatives. FL enters only in *how `n` decomposes and what each factor costs*:

| setting | `n` decomposes as | the expensive factor |
|---|---|---|
| **single device** | `P` × gradient-accumulation steps | wall clock, serially |
| **centralized ZO fine-tuning** (MeZO-style) | `P` × accumulation | wall clock; no communication |
| **data-parallel / distributed** | `P` × workers × accumulation | all-reduce bandwidth |
| **federated (here)** | `P` × `K` × `I` | `I` (round trips) and `K` (staleness, heterogeneity) |

**The mathematical model is general; the controller is where the paradigm matters.** The FL-specific parts
are (i) that `I` costs a round trip while `P` is free, which makes "raise P, shrink I" a real result
rather than an accounting identity; (ii) staleness; (iii) client heterogeneity, which is what makes S-J
worth building.

### 19.2 A testable prediction against centralized ZO

Centralized ZO fine-tuning uses `P = 1` and a very small fixed learning rate. Our criterion predicts
*why*: with `G_rule = 1` and no `K` or `I`, `cos = √(1/p)` ≈ 10⁻³, so the stable relative step is ~10⁻³
and the method needs ~`1/cos² = p` steps for coherent progress — matching the very long step counts such
methods report. **A falsifiable prediction about an existing published method, derivable with no new
experiments**, and the strongest single piece of evidence that the criterion is not an artifact of our
stack. Check against published curves before claiming it.

### 19.3 Is there a general ML contribution here

In descending confidence:

1. **Yes: the scale-invariance requirement for forward-gradient training** (§5) — the step, gate and
   estimator must all be ratios, or the method needs re-tuning per model. A *design principle*, and the
   sharpest general claim we have.
2. **Yes: `p` is a gradient-quality parameter, not just a memory knob** (§11.5). Inverts the standard PEFT
   intuition; applies anywhere forward gradients are used.
3. **Probably: the online control law** — `ρ ≤ cos` with a gradient-free estimator for `cos`. Novel as
   *packaging*, not as theory (§3.6).
4. **No: the scaling law itself.** `cos ∝ √(n/p)` is known ZO analysis.

A standalone paper would need three things we do not have: the criterion predicting the divergence point
across **≥3 models and ≥2 tasks**; the split-half estimator shown to **track measured `cos`** (§15.1 is
step one; §9.5 says it currently tracks nothing measurable); and the derived controller **beating
hand-tuned schedules without re-tuning** when the model changes. All buildable; none in hand.

**What the portfolio changed.** H-F closing as *shared* removes the fix itself from the contribution:
S-A/S-B repair the shared `_server_update_step`, so they are hygiene for both baselines. What survives —
now backed by a measured curve rather than a derivation — is the pair of **scaling laws that let a
configuration be scored before it is run** (`ρ ∝ η`, `ρ ∝ 1/√(K·I)`) and the `ρ/cos`-vs-`ρ·cos` split that
says pooling is free and `η` is not. **That is the publishable core; the fix is the demonstration, not the
claim.**

### 19.4 Larger models and datacenter fine-tuning

Both speculative:

- **The case changes from memory to throughput.** On-device, forward-gradient wins because it stores no
  activations. In a datacenter with backprop available, the only argument is that 2 forward passes are
  cheaper than 1 fwd + 1 bwd and need no activation memory, so a much larger batch fits — a real but much
  narrower claim, and `cos ∝ 1/√p` makes it *worse* at scale unless `p` is aggressively constrained
  (which is exactly what PEFT does).
- **MoE is structurally interesting.** Only active experts contribute to a forward pass, so a probe
  naturally lives in the active subspace: `p_effective` is per-token active parameters, not total. That is
  **block-coordinate probing for free** (§15.12) — the one structural idea that beats the `√(n/p)`
  barrier. Unanalysed; the most interesting extension on this list.

---

# Part V — Reference

## 20. Claims this supersedes

| Existing claim | Verdict |
|---|---|
| `fluxtune_contributions.md` §8 headline: "never converges — it *oscillates*" | **Superseded.** At 4 h it is a monotone rise then monotone divergence. F4/F5's position-locked collapses are an early *symptom*, not the mechanism |
| S1 momentum "REFUTED as-designed" | **Re-framed.** Correctly refuted *at ρ = 0.16*; ρ_eff = 1.6 explains the NaN exactly. Testable once ρ is bounded |
| S2 "variance-gate recalibration" | **Structurally unfixable as scoped** (§9.3) — but the gate is a working `N`-controller with a non-portable setpoint, not a dead statistic |
| S3 "aggregation-rate tempering / cap ω ≤ 1" | **Correct but ~2 orders of magnitude too small.** ω ∈ [0.702, 0.865]; the problem is ≥10× |
| H1 shuffle, H3 bin-order permutation | **Do not address the mechanism.** Park |
| H2 bin size / M1 sweep | **Settled: spend compute on probes, not bigger bins** — probe noise dominates ~40× (§11.3) |
| C1 "guided selection improves accuracy" | **Substantially revised (§11).** Stability-neutral by construction; dominated 3.34× by averaging the same probes. Redirect, don't abandon |
| C3-direction (alignment gate) | **Park.** ≤0.6% weight perturbation vs a ≥10× problem |
| C3-**freshness** (staleness) | **REVERSED — do not park.** The ≤8%-on-15.6% verdict is *conditional on N being small*; `FLUXTUNE_CODE_QA` §A4 says staleness stays ≤1 for a structural reason. **S-D creates precisely the high-C/K regime §A4 names as the one where freshness becomes load-bearing.** Critical path immediately after S-D |
| QA §D2 "the k sweep cannot run today" | **Answered offline** (§11.3) — computable from logged JVPs. Monotone; optimum k = P |
| QA §E1 "measuring `cos(G,g)` needs `v_k` uploaded" | **Not for this quantity** (§15.1) — a backprop gradient on a probe batch suffices |
| FwdLLM's variance-controlled aggregation | **Core departure, restated.** Not "the statistic cannot support a threshold" — it holds 0.86 accuracy at `K ≥ 20`. The departure is that the setpoint carries units, so it works by accident and does not port (§9.3) |

**Claims this document itself supersedes, after the portfolio** — so nothing older is quoted by mistake:

| Superseded | Replaced by |
|---|---|
| `‖θ_tr‖` at init = 20.36 (offline `build_model`) | **13.35**, telemetry, all ten arms (§7) |
| ρ = 0.115, flat all run | **0.16 at commit 1**, then falling iff `N` grows (§9.2) |
| orthogonality ratio 1.032 with a real 3% outward tilt | **1.000 ± 0.005**, stationary; the tilt was a reconstruction artifact (§9.1) |
| `\|d\| ∝ ‖θ_tr‖` exactly (6.0× vs 6.0×) | **sub-linear**, exponent ≈0.8 (§9.2) |
| the variance gate is 100% dead | **dead only at `K = 10`**; binding `N`-controller at `K ≥ 20` (§9.3) |
| `cos = 0.0231` (ANALYSIS) | **≤ 0.015** (95%, measured); closed form excluded at ~6 SEM (§9.5) |
| ρ/cos = 5.0× over budget | **≥ 10×** (§10) |
| collapse = logit saturation | **directional degeneracy**; `logit_norm` does not discriminate (§10, H-D) |
| "4 h minimum or the result is uninformative" | score `‖θ_tr‖²` vs commit and `‖Δθ‖` — ~20 commits, any horizon (§17) |
| S-E measurable after S-H + S-I (SNR ≈ 6) | **SNR ≈ 2**; S-E demoted, S-C promoted (§15.5, §15.7) |
| probe plan: split-half "meaningful pooled over ~100 commits" | needs **~1,400**, and even then one-sided (§9.5) |

**Telemetry bug — FIXED 2026-08-07.** `tc_transformer_trainer_distribute.py:485` used to log the argmax
under the label `chosen jvp` while the actual pick is the coin-flip result. It now logs the coin-flip
winner as `chosen jvp`, the argmax as `max jvp`, and the index as `chosen idx`. **Runs before that date
carry the old, mislabelled field.** The coin flip itself is real.

## 21. Reproducing every number

All from `lib/python/examples/fwdllm/experiments/`. Nothing needs a GPU except the model probe.

```bash
RUN=run_20260807_020030_fluxtune_n100_smoke_syn_0_sim     # the K=10 anchor
# telemetry files are multi-GB; slice first:
grep -hE '"event": "(server_update|agg_eval)"' $RUN/telemetry/aggregator_*.jsonl > /tmp/$RUN.jsonl
```

**Everything in Part II comes from `server_update` and `agg_eval` directly** — L3 logs
`trainable_weight_norm` / `trainable_delta_norm` / `rho`, L1 the split-half components, L4 `logit_norm` /
`pred_entropy` / `top_class_share`. **The reconstruction from `‖W‖` and the `build_model` anchor are
retired.** Only `p` and the layer census survive from that probe.

**Legs 1 and 2** — from `server_update` alone:

```python
ortho = (tw[b]**2 - tw[a]**2) / sum(dn[a+1:b+1]**2)   # per 25-commit block; expect 1.000 +- 0.005
rho_t = dn[t] / tw[t]                                 # == the logged `rho`; ~0.16 at commit 1
step  = dn[t]                                         # THE diagnostic: flat => ||theta||^2 linear
N_t   = K * pool_size[t]                              # pool_size == iterations, exactly, every record
# rho * sqrt(N) is invariant across the K-sweep (1.68-1.81); rho / eta is invariant at commit 1
```

**`|JVP|` growth, selection gain, Gaussianity, k-sweep** — the trainer log carries all candidates:

```bash
grep 'All JVPs sorted by magnitude' $RUN/*trainers.log   # 34,447 lines
# format (runs from 2026-08-07): "All JVPs sorted by magnitude: [...] and chosen
#          jvp: X and max jvp: Y and chosen idx: I for trainer : T for model
#          version: R data-id: D. iteration: I"
# GOTCHAS: "model version" here is actually the ROUND.
#          In runs BEFORE 2026-08-07 there is no `max jvp`/`chosen idx`, and
#          `chosen jvp` holds the ARGMAX, not the coin-flip winner (§20).
#
# count `tensor(` in the list to read P off the log (10 vs 30) -- P is a trainer override and
# never appears in aggregator_config.json.
# rms|d| over the K=10 run: 3.35 -> 15.87  while ||theta_tr|| 13.6 -> 94.8  (exponent ~0.8)
# per EVENT, normalise the P values by that event's own rms, then:
#   E[v_par^2 | top-1]      = mean( max(d)^2 / mean(d^2) )        -> 3.805 (P=10) / 5.619 (P=30)
#   E[v_par^2 | coin top-2] = mean( (d1^2+d2^2)/2 / mean(d^2) )   -> 2.988 (P=10) / 4.744 (P=30)
#   top-k average objective = mean( mean(top-k d^2)/mean(d^2) )*k -> monotone, 3.81 .. 10.00
# compare against synthetic iid N(0,1) with the SAME estimator -> 3.798 / 2.987 (match to 0.3%)
# b = sqrt(E) check: rho(P=30)/rho(P=10) at commit 1 = 1.236 vs sqrt(4.744/2.988) = 1.260
```

**Leg 3 (variance gate)** — the aggregator log:

```bash
grep -o '\[IterProgress\] data_id=.* force_commit_planned=[A-Za-z]*' $RUN/*aggregator.log
# Split on data_id change; the last row of each bin is the commit.
# reason = CAP if iter >= max_iter-1 else natural if var < 0.3 else plateau
#   K=10: 0 natural / 67 plateau / 83 cap      K=30: 128 / 4 / 0
#   K=20: 77 / 20 / 28                         K=50: 148 / 1 / 0
# For E-1 phase 1, regress realised N = K*pool_size on ||theta_tr||^2 across the K>=20 arms.
```

**`cos` from the split-half components** — never average per-commit `split_half_cos`:

```python
cos = sqrt( 2 * sum(split_half_dot) / sum(split_half_norm_a * split_half_norm_b) )
# per-commit sd is 1.5e-3 (= 1.5/sqrt(p)); one run gives SNR ~ 1. Pool ARMS, not just commits:
# all five K=10 arms (1,400 commits) -> split_half_cos = 3.0e-5 +- 4.0e-5 => cos <= 0.015 (95%)
```

**The `p` census** (needs the `test_fwdllm` env, ~1 min). **Use this for `p` and the layer split only** —
its `‖θ_tr‖ = 20.356` is **wrong for production**, which starts at 13.35. Reconciling that gap is a live
discrepancy, not a known offset: `p` matches exactly, the norm does not.

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
# -> 1040932  20.3562  416.2133  10.203  0.501     <- p OK; 20.3562 is NOT the production init norm
# -> {'adapters': 447264, 'pre_classifier': 590592, 'classifier': 3076}
```

---

# Part VI — Implementation tracker (LIVE)

*Parts I–V are the diagnosis and the plan. This is the record of what we **built and ran**, and the
working surface from here on. The running verdict in §25 is what eventually gets written into
`fluxtune_contributions.md`.*

> **DOC DISCIPLINE — STRICT.**
> 1. **Edit in place, do NOT append.** Rewrite the row that changed. No dated "update:" notes, no
>    changelog, no sibling rows. Git holds history.
> 2. **One row per feature.** Superseded/refuted → overwrite.
> 3. **Status tags:** `TODO` · `WIP` · `WORKS` · `FAILED` · `INCONCLUSIVE` · `PARKED`. Nothing reaches
>    `WORKS` without predicted-vs-observed numbers and its run id in the same row.
> 4. **Every feature is flag-gated, default = old / byte-identical off.** Name the flag. Terminal state
>    (`PERMANENT` / `FLAGGED` / `REVERTED`) is the **operator's** call — ask, with the A/B evidence.
> 5. **A prediction before the run, always.** A feature with no falsifiable prediction is not ready.
> 6. **Score every run the same way** (§17): `‖θ_tr‖²` vs commit, `‖Δθ‖`, `ρ·√N`, `top_class_share`.
>    Accuracy is a *result*, never the stability evidence.

## 22. Feature ledger

**Config arms — already run** (2026-08-07 portfolio, §7.1; no code change, launcher flags only).

| # | What | Flag / knob | Run | Predicted | Observed | Status |
|---|---|---|---|---|---|---|
| A1 | `K`-sweep 10→20→30→50 | `--agg-goal` / `--c` | `031606` `042626` `054135` | `ρ ∝ 1/√K`: 0.16 → 0.07 | `ρ ∝ 1/√(K·I)` instead: 0.16 → 0.14. **`ρ·√N` invariant to 4%**; K≥20 holds 0.860 | **WORKS** (stability); prediction *refined*, not met |
| A2 | `η`-sweep 0.01→0.002→0.0005 | `--learning-rate` | `004228` `025016` `045646` | `ρ ∝ η` exactly at commit 1 | 0.2004 / 0.0404 / 0.0101 — **1%** | **WORKS** as a measurement; **FAILED** as a fix (1:1 progress cost) |
| A3 | `P = 30` under the shipped selection rule | `--perturbation-count` | `011351` | `ρ/cos` unchanged, `E` up ⇒ *faster* divergence | `E` 2.99→4.74, `ρ` ratio 1.236 vs 1.260 predicted; doubling 68→57 commits | **WORKS** as a falsification; **FAILED** as a fix, as predicted |
| A4 | `fwdllm` cross-baseline at matched scoring | `--only fwdllm` | `004453` | either signature | Same: ratio 1.000, `ρ` 0.175, doubling ~106 | **WORKS** — closes H-F: defect is shared |
| A5 | `fwdllm_plus` | `--only fwdllm_plus` | — | — | never launched (node-3 loop carried 2 arms) | **TODO** |

**Code features — to build.** Ordered as §17. Nothing here has been written yet.

| # | Feature | Flag (default = old) | Site | Prediction to check at commit ~10 | Status |
|---|---|---|---|---|---|
| B1 | `cos(G,g)` ground-truth probe | `cos_ground_truth_audit` | `FedSgdAggregator._emit_server_update` | `cos` lands at 0.023 **or** 0.008–0.015; §15.1's two branches diverge from there | **TODO** — blocks every sizing |
| B2 | S-H · average all `P` probes | `probe_combine: {select\|mean}` | `tc_transformer_trainer_distribute.py:456-487`, `:625-631` | `ρ` drops **5.5×** on commit 1 (`√(E·P)`); `‖G‖` by the same factor | **TODO** |
| B3 | S-A + S-B · trust-ratio step with a `t^{-(1/2+ε)}` anneal | `server_step_rule: {raw_sgd\|trust_ratio}` + `rho_schedule` | `FedSgdAggregator._server_update_step` | logged `ρ` **equals** `ρ*_t` every commit; `‖θ_tr‖²` grows **sub-linearly** | **TODO** — must land together (§15.4) |
| B4 | S-C · explicit `N`-controller | `dynamic_kc.policy: rho_target` | `selector/dynamic_kc_policy.py` | realised `N` tracks `N_now·(ρ_now/ρ*)²`; `ρ` converges to `ρ*` within ~20 commits | **TODO** |
| B5 | S-I · freeze `pre_classifier` (+ `h` rescale) | `trainable_scope: {adapters_head\|adapters_only}` | model builder | `p` 1,040,932 → 450,340; `cos ×1.52`; **peak accuracy unchanged** is the real test (H-G) | **TODO** |
| B6 | H-J · long `K=50` arm, ≥32 h vclock | none (runtime only) | — | degradation onset commit 400–560, collapse 1,200–1,700 | **TODO** |
| B7 | S-G · weight decay `λ ≈ ρ²/2` (control arm) | `server_weight_decay` | `_apply_weighted_update` | `‖θ_tr‖` plateaus instead of growing | **TODO** — the bar the stack must beat |
| B8 | H-E · staleness histogram from the K-sweep | none (log replay) | `agg_round` telemetry | staleness > 1 appears at `c = 100` | **TODO** — free, data on disk |
| B9 | S-E · split-half commit gate | `commit_gate: {var\|cos}` | gate | — | **PARKED** — unmeasurable at `p = 10⁶` (§15.5) |
| B10 | S-J · adaptive `P` per client | `probe_budget: adaptive` | trainer | — | **PARKED** — same wall as B9, worse (§15.9) |
| B11 | S-K · `n_eff` sensor, emit-only | `n_eff_audit` | `FedSgdAggregator` (var site, `:438`) | `n_eff` reproduces `ρ·√N` = 1.68–1.81 across the K-sweep; `n_eff/N` ≈ 0.3–0.4, matching the §9.5 shortfall | **TODO** — replay first (no GPU), then wire |
| B12 | H-K · α-sweep 1 → 10 → 100 at one fixed setpoint | `--partition-method niid_label_clients=100_alpha={1,10,100}` | launcher only | under the **var gate**: achievable var floor rises with heterogeneity ⇒ commit behaviour shifts. Under `n_eff`: `n_eff/N` falls with α ⇒ **the sensor absorbs it with no knob change** | **TODO** — partitions confirmed present in `agnews_partition.h5` |

## 23. What worked, and why

*Only entries with numbers. "Why" must name the mechanism, not the outcome.*

- **Widening `K` (A1).** Holds 0.860 for the whole run where `K = 10` collapses to 0.250. **Why:** not `K`
  itself — `K` raises `N` only until the gate absorbs it. The mechanism is that at `K ≥ 20` the gate
  escapes its `max_iter` cap and starts controlling `N`; since `var` carries `‖θ‖²`, holding it at 0.3
  forces `N ∝ ‖θ‖²`, so `‖Δθ‖` is pinned at 2.03 and `‖θ‖²` grows linearly instead of geometrically.
  **It is an accident and it only defers** — see §24.
- **The measurement stack (L1–L4).** `trainable_weight_norm`, `rho` and `top_class_share` turned a 4-hour
  accuracy verdict into a 20-commit one, and L3 alone corrected four constants (§20). **Why:** the
  quantities that govern the dynamics live on the trainable slice; anything derived from total `‖W‖` is
  swamped by the frozen 416.2.
- **Sweeps over single points.** Every surviving law was confirmed by a *slope* (`ρ` vs `η`, `ρ` vs `N`),
  and the one claim that died — `ρ ∝ 1/√K` — died because a curve exposed the gate's feedback that a
  single point would have hidden. **Why:** a slope cannot be rescued by a fudge factor.

## 24. What did not work, and why

- **Lowering `η` (A2).** Survives, but reaches 0.601 at commit 120 where `K = 20` reaches 0.860. **Why:**
  `η` scales `ρ` and leaves `cos` untouched, so `ρ·cos` falls 1:1 with the stability gained. Pooling moves
  `ρ` and `cos` in opposite directions and is free; `η` is not.
- **Raising `P` under the shipped selection rule (A3).** Faster learning per commit *and* faster collapse;
  peak at did 62 instead of 112, gone by commit 120. **Why:** selecting on `|d|` sets `a = E`, `b = √E`,
  so `b²/a = 1` exactly — aim gain and step growth cancel for *any* `E`. Confirmed to 2%.
- **The split-half cosine as an online statistic.** Per-commit SNR 0.07; a whole 190-commit run gives ≈1.
  **Why:** at `p = 10⁶` the sampling floor on a cosine is `1/√p = 1e-3` and the signal is `cos²/2 ≈ 1e-4`.
  Arithmetic, not tuning — it kills S-E and S-J at the current `p`, and is the strongest practical
  argument for S-I.
- **Treating `‖θ_tr‖` orthogonality as the defect.** The ratio is 1.000 in the arms that *work* too.
  **Why:** `g` is itself near-perpendicular to `θ` at `p = 10⁶`; the test says the norm grows by the full
  step length, not that the step is noise. The defect is `ρ` vs `cos`, and only that.
- **`K ≥ 20` as a fix.** `Σρ_t²` diverges logarithmically under the accidental `1/√t`, and `‖θ_tr‖²` grows
  linearly at 4.2/commit — collapse at commit ~1,200–1,700, an 8× deferral. **Why it must not ship as the
  answer:** `var_threshold = 0.3` hard-codes the trajectory and does not port across model, `p`, or
  adapter rank. B6 is the falsification.

## 25. Running verdict on the claimed contributions

*Not yet landed in `fluxtune_contributions.md` — that happens once the B-features have run. Two naming
schemes are in play: `C1–C3` (this doc, `FLUXTUNE_CODE_QA.md`) and `S1–S3` (contributions §8.2).*

| claim | as written today | after the portfolio | what would settle it |
|---|---|---|---|
| **C1 · informed JVP-magnitude probe selection** (contributions §2, §4.1, §6) | "keeps the steepest — a better gradient estimate per round" | **Half-refuted.** True for *aim* (`cos ×1.73`), **exactly zero** for stability (`b²/a = 1`, measured), dominated **3.34×** by averaging the same probes. The `2P` budget is justified by S-H, not by selection | B2: does `mean` beat `select` at matched `ρ` |
| **C2 · async aggregation / K-C control** | "async, straggler-tolerant; `agg_goal < K` commits as stragglers arrive" | **Reframed and strengthened.** The pooling law `ρ ∝ 1/√(K·I)` is measured, and only the async path has a free `N` to spend — but **raising `K` is not the claim**, because the gate absorbs it. The claim is the *controller* (S-C) | B4: `ρ` converges to `ρ*` under a measured setpoint |
| **C3 · aggregation weighting (ω: direction + freshness)** | grad-aware rate, align gate, staleness weighting | **Direction: park** — ω spans 0.70–0.87 against a ≥10× gap. **Freshness: open and now testable** — the K-sweep ran at `c = 100` and the histogram is unscored | B8 (free replay), then the S-D sweep |
| **S1 · server momentum** (contributions §8.2) | "REFUTED as-designed; retry with a variance-normalised step" | **Re-framed, not refuted.** `ρ_eff = ρ/(1−β) = 1.6` explains the NaN as arithmetic. Testable only after B3 bounds `ρ` | retry after B3 |
| **S2 · variance-gate recalibration** | "commit on the plateau, not a noise dip" | **Superseded.** The gate is a working `N`-controller at `K ≥ 20`; the problem is that its setpoint carries units, not that it commits early. Replace the loop, do not re-tune the threshold | B4 |
| **S3 · aggregation-rate tempering (cap ω ≤ 1)** | "cap rate ≤ 1; damp, not amplify" | **Correct and ~2 orders of magnitude too small.** Subsumed by B3: under a trust-ratio step ω cannot set magnitude at all | B3 |
| **Systems: flat memory, inference-only operator set** (contributions §3.1–3.2, §4.3) | structural claims about the absence of an autograd graph | **Untouched.** Nothing in the divergence work bears on them; they remain the strongest part of the paper | — |
| **Cost framing** (contributions §3.3) | "10× sync compute at P=10, collapsing to parity at P=1" | **Needs rewriting.** Under selection, raising `P` is *harmful* (A3). Under S-H the same `2P` buys `ρ/cos ∝ P` — a linear, measured return. Defensible only after B2 | B2 |

**Does this change the contributions?** Yes — all three ML claims move, in different directions: **C1
shrinks** to a compute-budget justification that only S-H redeems, **C2 grows** but relocates from "async"
to "the controller that async makes possible", and **C3 splits** into a parked half and an untested half.
The systems contributions are unaffected; the §3.3 cost framing is not. What the portfolio *adds* are two
candidates that were not on the list: **`p` as a gradient-quality parameter** (§11.5), and the
**`ρ ≤ cos` criterion with `ρ` as an exactly-measured online sensor** (§3.6). The second is only
half-paid-for — `cos` has no usable online estimator at `p = 10⁶` — so until B1/B5 it is a *design rule*,
not the control law the paper would want to claim.
