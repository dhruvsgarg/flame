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

## §0 — Where the work stands

**In one line:** *we know why it diverges and we have four fixes that work; what we still cannot do is
compute the right step size from the config rather than reading it off a dose-response curve.*

### Settled — do not re-measure

| what | where |
|---|---|
| **Root cause:** nothing in the pipeline is scale-invariant — estimator, step and gate all inflate together | §2.4 |
| Every step is orthogonal to `θ`, so the norm grows by the full `‖Δθ‖` every commit — this makes the norm law *exact* | §2.1 |
| `ρ ∝ η` and `ρ ∝ 1/√(K·I)`; the naive `ρ ∝ 1/√K` is false because the gate hands a `K` rise back as fewer iterations | §2.2 |
| The variance gate is a dimensionally-wrong `N`-controller — and is the *accidental* stabiliser | §2.3 |
| The defect is **shared with sync `fwdllm`**, not a FluxTune bug | §2.5 |
| Heterogeneity (α) acts **only** as a step-size multiplier, through the gradient scale | §2.6 |
| Collapse is **directional degeneracy** of the head, not logit saturation | §2.7 |
| **Norm law** `‖θ_T‖/‖θ_0‖ = e^B` — closed form, no free parameter, 21 arms | §4.1 |
| **Progress law** — peak accuracy is monotone in `Λ` and saturates | §4.2 |
| Probe selection by `\|d\|` is stability-neutral **by construction**; the candidates carry no other signal | §3.2 |
| **Fix 1** — average all `P` probes instead of selecting one (5/5 arms) | §5.1, [P3](fl_fwd_ft_practice.md#p3--knob-ledger--what-each-lever-actually-did) |
| **Fix 2** — trust-ratio step: `ρ` becomes an operator constant, the α multiplier disappears | §5.1, [P3](fl_fwd_ft_practice.md#p3--knob-ledger--what-each-lever-actually-did) |
| **Fix 3** — Robbins–Monro anneal at `exp` = 0.25 | §5.1, [P3](fl_fwd_ft_practice.md#p3--knob-ledger--what-each-lever-actually-did) |
| **Fix 4** — shrink `p` via adapter rank; `p` is a *gradient-quality* knob (3/3) | §5.2, [P3](fl_fwd_ft_practice.md#p3--knob-ledger--what-each-lever-actually-did) |
| The `cos` probe's reference batch was one client's non-IID shard — every `cos` it logged is void | §6 |
| **Q1** — the collapse budget is **relative** (`Φ`), not absolute (`‖θ_tr‖`) | §7.1 |
| **Q2** — `‖θ_tr‖` is mostly a **symptom**; perfect decay recovers only 20–25% of the damage | §7.2 |
| Backprop reaches **0.90** where the best forward-gradient arm reaches **0.865** — the cost of going backward-free | §7.3 |

### Dead — refuted, do not re-propose

| what | why it died |
|---|---|
| "There is a critical `ρ` ≈ 0.09" | horizon artifact; **every** `ρ > 0` inflates geometrically (§4.3) |
| Safety factor `s` = 0.3–0.5 | derived wrong; the two laws give 2.6–4.3 (§4.4) |
| Lowering `η` as the fix | pays 1:1 in progress and still has `Σρ² = ∞` |
| Split-half cosine as a gate or an online `cos` estimator | measurability wall at this `p` |
| `n_eff` as a controller input | an identity, blind to directional disagreement (§5.4) |
| The `‖θ‖²` log-log slope as a stability score | bounded by 1 under trust-ratio *by construction* (§4.3) |
| Client disagreement, and reference noise, as the cause of the `cos` shortfall | both quantitatively excluded (§6) |
| Every `cos_ground_truth` number ever logged | **void** — broken probe reference (§6) |
| "`Φ` decides whether *any* arm holds its peak" | true only for arms that actually learned (§4.2) |

Full reasoning and the rest of the list: [P6 · dead ends](fl_fwd_ft_practice.md#p6--dead-ends--do-not-retry).

### Open — and this is the entire list

| # | question | blocking what | how it closes |
|---|---|---|---|
| **H-P** | With a clean reference, does measured `cos` match the closed form? | whether `ρ ≤ s·cos` is a real *control law* or only a *design rule* | probe is fixed; rides free on any arm |
| **Q3** | Does `Λ` predict **in the steep region** (it is currently fitted where accuracy has saturated)? | turning `Λ` from fitted into predictive | 3 arms at pinned `ρ*`, `N` |
| **C2** | Cohort width — the commit gate has never produced a clean A/B | the async/controller contribution | gate re-run at `K` = 30 |
| **H-E** | Does high `K`/`C` produce real staleness? | whether ω-freshness matters | K/C sweep (old logs deleted) |
| **S1** | Does a gradient component persist across commits, so momentum pools *in time*? | the only lever that raises `Λ` at zero `B` cost | in flight |
| **M1** | "probe noise dominates data noise by ~40×" is **analysis, never measured** — and it is what puts `P` ahead of `K` in the priority order | the whole priority order (§5.5) | rig: split-half over probes vs over data |
| — | **No convergence-detection rule exists**, and no eval-side monitor works as one | knowing when to stop | unowned |

**What is *not* open: the optimizer.** The four fixes enact to spec and the dynamics are closed by the
two laws. Everything above is about the *constants* those laws are scored against.

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

## §1.4 Definition of done

1. An arm reaches **peak ≥ 0.86** and **ends within 0.015 of its peak** at ≥300 commits, without
   hand-tuning per model or per α.
2. The setpoint that achieves it is **computed, not searched** — from the config, before the run.
3. Stability is readable in **~20 commits** from `B` and `Λ` (§4), not from a 4 h accuracy curve.
4. Every constant in the loop is dimensionless (§5.3), so nothing needs retuning when `p`, α, `K` or
   the round index changes.

Items 1 and 3 are met by several arms today. **Item 2 is the open one** — the sizing formula's `cos`
has never been cleanly measured (§6), so `ρ*` is read off a dose-response curve. Item 4 is met for the
step rule and unmet for the commit gate.

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

**Score `B` and `Λ` — never accuracy, never the `‖θ‖²` log-log slope.** Both are exact at any horizon
and readable in ~20 commits.

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
| `K` | trainers pooled per commit (`agg_goal`) | 10 shipped; ≥30 required (§5.5) |
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

**Status: the length half is MEASURED and holds; the shadow half has never been validly measured** (§6).

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

> **Its accuracy is `ρ`-dependent** — the flat "<0.3%" once claimed is wrong. Replayed over all arms:
> **<0.3% on the low-`ρ` trust-ratio arms**, **~2% on the high-`ρ` raw-SGD arms**, **−5.8% on the
> collapsed α=0.1 arm.** §2.1's ratio drifts as `ρ` grows and the error compounds over `T`. Sum `B` over
> the steps that actually lie between the first and last norm sample — the mis-aligned window costs 1.2%
> vs 0.8% mean error.

## §4.2 The progress law — how much learning a trajectory banks

The same 21 arms sorted by `Λ`: **peak accuracy is monotone from 0.377 to 0.865 with no exception
outside replicate noise**, saturating at `Λ ≈ 0.95`. Read `Λ` as **the accumulated aligned displacement,
in units of `‖θ_tr‖`** — you must travel about 0.6 of your own length in the right direction to reach
0.85, and about one length to saturate.

Calibration (agnews/DistilBERT): `Λ` 0.07 → 0.49 · 0.20 → 0.70 · 0.40 → 0.80 · 0.60 → 0.85 · 0.95 → 0.865.

**Among arms that learned, whether an arm *holds* its peak is decided by `Φ` and nothing else.** Every
arm with peak ≥ 0.80 and `Φ ≤ 3.63` ends within 0.015 of its peak; every arm at `Φ ≥ 4.2` degrades,
monotonically in `Φ`.

> **The "peak ≥ 0.80" qualifier is load-bearing and was missing.** Four arms sit at `Φ` = 1.00–1.01 —
> *no* norm inflation — and still fall 0.06–0.12 below their peak. All four have `Λ ≤ 0.068`; their
> "peak" is barely the init accuracy and what follows is directional drift at constant norm. **`Φ`
> governs *losing what you learned*; it says nothing about arms that never learned.** Independently,
> this is direct evidence that accuracy loss does **not** require norm growth (§7.2).

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

**(c) The closed-form `cos` is functionally validated, and the probe that appeared to contradict it was
broken.** `Λ` is built from the *predicted* `cos`, and it collapses arms spanning `G_rule` 2.988 vs 10,
`p` 118k–450k and `N` 10–200. **A formula that wrong could not order 21 arms** — which raises the prior
sharply that the probe's reference was the problem, not the formula (§6). *Caveat that sets the next
experiment:* the rule and `p` variation all sits at `Λ > 0.8`, where accuracy has saturated, so
**`Λ`'s rule- and `p`-scaling is untested in the steep region** (Q3).

## §4.4 The criterion

Two clocks. **The deadline:** coherent progress grows as `T·ρ·cos`, noise displacement as `√T·ρ`;
signal overtakes noise at `T ≈ 1/cos²`. **The budget:** the norm inflates by `(1+ρ²)^{T/2}`, doubling at
`T ≈ 1.4/ρ²`. You survive iff `1/cos² ≤ 1.4/ρ²`:

> ## ρ ≲ s · cos(G, g)
> **The relative step must not exceed the fraction of it that is actually aligned with the gradient.**

The O(1) constants that derivation drops are now measured. Eliminating `T` between the two laws gives
`B = Λ·ρ/(2cos)`, so fitting a target `Λ_req` inside a budget `B_max` requires
`ρ/cos ≤ 2·B_max/Λ_req` = **2.6–4.3** — not the 0.3–0.5 originally derived, a factor of ~8 that is
exactly the 5–10× seen empirically.

**The design rule.** For a fixed learning target `Λ_req` over `T` commits, Cauchy–Schwarz makes constant
`ρ = Λ_req/(T·cos)` the budget-minimising trajectory, at cost `B = Λ_req²/(2T·cos²)`:

> **Budget cost for a fixed amount of learning is ∝ `ρ`; wall-clock cost is ∝ `1/ρ`. Pick the smallest
> `ρ` whose commit count you can afford.** There is nothing to tune — `ρ` is the exchange rate between
> wall clock and safety margin, and `cos` (via `P`, `K`, `I`, `p`) sets the exchange rate itself.

Note what (a) already killed: *"walk `ρ*` up until it breaks"* is withdrawn. There is no cliff to find,
only an exchange rate to pick.

## §4.5 Sizing a configuration — the formula, and why it is not yet usable

```
G_rule = E[v_par^2]   for select-one-of-P     (P does not appear -- sec 3.2)
G_rule = P            for average-all-P

cos           = sqrt( G_rule * N / p )        N = K*I uploads pooled server-side
N_req( rho* ) = p * (rho*/s)^2 / G_rule       <- pool needed to make rho* safe
```

| config | `G_rule` | `p` | `N` needed at ρ\*=0.16 | at ρ\*=0.05 | at ρ\*=0.02 |
|---|---|---|---|---|---|
| shipped (select 1 of 10) | 2.99 | 450,340 | 3,860 | 377 | 60 |
| + average all 10 | 10 | 450,340 | 1,150 | 113 | 18 |
| + average + adapter `rf`=64 | 10 | 118,348 | 303 | **30** | 5 |

> **The absolute values in this table are unverified.** The direct `cos` measurement that appeared to
> refute them was an instrument artifact (§6), so the formula is *unrefuted but unconfirmed*. **Ratios
> between rows survive** (`ρ ∝ 1/√N` is measured to 5%); **treat absolute `N_req` as untested** until
> H-P closes, and size `ρ*` from the dose-response curve in
> [P4](fl_fwd_ft_practice.md#p4--arm-ledger--every-arm-ever-run) instead.

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
  rho*_{t+1} <- rho*_0 * t^-exp              # exp = 0.25, sized to the HORIZON
```

**(b) The same loop in FL:**

```
SIZING (offline, before the run):
  choose rho*_0 from the DOSE-RESPONSE CURVE (P4), not from p*(rho/s)^2/G_rule   <- 4.5 unusable
  pick I as small as the gate allows, then K = n_req/(P*I);  require C >= K

PER COMMIT (server):
  dispatch to C clients, wait for K uploads   # async: K arrives, stragglers roll into the next
  G <- sum_k omega_k u_k / sum_k omega_k      # omega re-weights; it must NOT set magnitude
  commit when the pool reaches N_target       # NOT a cos_hat test -- unmeasurable today (P6)
  theta_tr <- theta_tr - rho*_t * ||theta_tr|| * G / ||G||
  log rho, ||theta_tr||, top_class_share      # the three monitors (2.8)
  rho*_{t+1} <- rho*_0 * t^-exp
```

Flags, defaults and status for each element: [P1](fl_fwd_ft_practice.md#p1--what-is-built)
and [P2](fl_fwd_ft_practice.md#p2--the-recommended-stack).

## §5.2 The lever table — this ranks every possible fix

**`ρ/cos` is "will this survive"; `ρ·cos` is "how fast does it learn".** A good lever improves the first
without hurting the second.

| lever | `ρ/cos` (stability) | `ρ·cos` (progress) | who owns it |
|---|---|---|---|
| **P-averaging** (trainer pooling) | **∝ 1/P** | **invariant — and free in wall clock** | trainer |
| **K** (cohort width) | **∝ 1/K** | invariant — parallel across devices | client selection |
| **I** (iterations per bin) | **∝ 1/I** | invariant — **but serial: one round trip each** | aggregation gate |
| **p** (trainable dimension) | **∝ √p** | ∝ 1/p | model/PEFT design |
| `η` learning rate | ∝ η | ∝ η — **pays 1:1** | aggregation |
| probe-selection gain `E[v∥²]` | **invariant** (`b²/a = 1`) | ∝ E | trainer probe selection |
| step normalization | sets `ρ` to an operator constant | decoupled | aggregation |

Per unit of **compute**: progress ∝ `C/n`, inflation ∝ `C/n²`, so the ratio improves ∝ `n`. **Larger
pools are strictly better for stability per FLOP, and slower only in absolute wall-clock progress.**

**The `p` row is `√p`, not `p`** (**MEASURED**): `ρ` does not fall as `√p`, because both halves of
`ρ = ‖Δθ‖/‖θ_tr‖` scale as `√p` and cancel. **`ρ` is already dimensionless in `p`**, so shrinking `p`
buys `cos` alone — which is still the best cost/benefit in the stack, since it costs *negative* compute.

**Three pooling stages, and who owns each:**

| stage | owner | cost structure | reduces |
|---|---|---|---|
| `P` probes per iteration | **trainer** | on-device, parallel, free in wall clock | probe noise |
| `K` trainers per commit | **client selection** | parallel across devices | probe + data noise |
| `I` iterations per bin | **aggregation** (commit gate) | **serial round trips** | probe noise only |
| `p` | **model/PEFT design**, upstream of all three | negative | — |

`P` and `I` are *substitutes* — both pool over the same bin at the same `θ`, both reduce probe noise
only — at wildly different cost. **Raise `P`, lower `I`, hold `n = P·K·I`: commits get strictly faster
and no less well-aimed.** Wall clock per commit is set by `I` (serial), not `K` (parallel), so **choose
the smallest `I` the gate allows and buy the rest with `K`.**

> **In one line: pooling sets how large `ρ` is *allowed* to be; aggregation *spends* within that budget.
> Aggregation can only shrink `ρ`; it can never raise `cos`.**

The natural reading — *"it's an aggregation bug"* — is **half right, and acting on that half alone costs
a ≥10× slowdown**: `K` ≥ 20 reaches 0.860 while `η` = 0.002 reaches 0.601 at commit 120 — same
stability, one at zero progress cost and one at the full 1:1 cost. **And pooling alone runs out**: at
`K` ≥ 20 it buys ~500 commits before the linearly-growing norm walks back out.

**Why the controller spans two subsystems.** `N = K × I` is a client-selection knob times an aggregation
knob, so the gate (which decides `I`) is a client-selection decision made from aggregation telemetry —
**the controller's sensor and actuator sit on opposite sides.** A pure-selection controller has nothing
to measure; a pure-aggregation controller can clamp `ρ` but cannot tell whether it is clamping too hard.

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

## §5.4 Priority order for any new configuration

Every decision reduces to *how to reach the required `n` most cheaply*:

1. **`p` first** — the only lever that improves aim at *negative* cost.
2. **then `P`** — free in wall clock, and only pays off if all `P` are assimilated.
3. **then `K`** — the only stage that touches data noise.
4. **`I` last**, and only what the gate demands — it is the one that costs serial round trips.

Then two rules: **set `ρ*` from the dose-response curve, not from `η`, and anneal it** — a pool large
enough to be safe now is not safe 1,000 commits later; and **never let probe selection set step
magnitude.**

*Caveat, and it is open:* the `P`-before-`K` ordering rests on "probe noise dominates data noise by
~40×", which is **ANALYSIS and has never been measured** (M1).

## §5.5 What is settled, and what is not

**Settled — do not re-measure.** The step rule, the anneal, the combination rule and the `p` lever all
enact to ≤4e-3 with mechanisms measured. On top of them, §4 closes the *dynamics*: `‖θ_T‖` is a
closed-form function of the `ρ` trajectory alone, and peak accuracy is a monotone function of `Λ`.

**The remaining unknowns are not about the optimizer.** They are about the two constants those laws are
scored against — `B_max` and `Λ_req` — and about whether the closed-form `cos` inside `Λ` is real (§6).

**Two structural gaps with no owner:** no convergence-detection rule exists, even in §5.1; and the
`h`/`p` coupling is unanalysed beyond the FD-rescale interaction.

**One requirement, not a preference:** `K` ≥ 30 with `C` ≥ `K`. At `ρ*` = 0.06 the criterion demands
five times what `K` = 10 can pool, which is why the commit-gate A/B has come back inconclusive twice.

---

# Part 6 — The open hole: `cos` has never been validly measured

*Everything else in the model is either closed-form or measured. `cos` is neither. This section is
also the sharpest methodological lesson in the program.*

**What was built.** A fp32 backward pass on a fixed held-out batch, server-side, at `θ_t` before the
step, emitting `cos_ground_truth` per commit. Index alignment was unit-tested (pool = ±`g` → cos = ±1).
Three arms carried it.

**What survives.** `‖G‖/‖g_probe‖` matches `b·√(p/N)` **to 5–10% in all three arms**, across 6× in `b`
and 6.7× in `N`. That is a ratio of *norms* and does not depend on the reference's direction, so it
stands: **the length half of §3.2 is measured and holds**, and so does the assumption that client
gradients have roughly the norm of a held-out one.

**What does not.** **Every `cos` value the probe reported is void**, because the reference batch was
wrong. The probe sliced the first `n` rows of the raw test tensor, bypassing the DataLoader's sampler.
The test index list is built by iterating clients in partition order and extending with each client's
shard, **with no shuffle** — so under a Dirichlet partition the probe's 64-sample "global reference" was
**one client's shard**, 48/4/5/7 across four classes against a test set that is exactly balanced.

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
anti-ordering of the arms — with none of it being real. Both rival explanations are dead: **client
disagreement** would need fewer than one effective upload, and **reference noise** is only a 1.1–1.6×
effect by a two-batch control.

**Fixed and shipped:** fixed-seed shuffled batch over the whole test set, default size 1024, plus a
preflight that fails the launch if the reference is ever class-skewed again. **Nothing has been
re-measured with it yet — that is H-P, and it rides free on any arm.**

**Operative reading.** There is **no trustworthy measurement of `cos` in either direction.** The closed
form is *unrefuted* — §4.3(c) shows `Λ` built from it orders 21 arms across both rules, 3.8× in `p` and
20× in `N` — and the instrument that appeared to contradict it was broken. So: **size `ρ*` from the
dose-response curve, not from §4.5.** `ρ` is exact per commit, but the quantity it must be compared
against **has no usable online estimator** (the gradient-free split-half route is dead). **Closing that
gap is the difference between a design rule and a control law** — still the most valuable open problem
here.

**What is and is not assumed.** Isotropy of `v` is exact by construction and is **not** the caveat. The
caveat is **independent, homogeneous pooling**: `cos = (a/b)√(n/p)` assumes the `n` readings are
independent and share one target `g`. They do not — the `I` iterations share a bin's gradient and the
`K` trainers have different gradients. `n_eff` excludes the *correlation* half and the replay above
excludes the *disagreement* half, so **neither half of the pooling model is currently suspect.**

**The lesson, which generalises:** the probe's arithmetic was unit-tested and correct; nothing tested
that its *input* was what we thought. Three consistency checks failed and were written up as findings
about the optimizer for a full revision before anyone asked what the reference batch actually contained.
One `Counter(labels[:64])` would have caught it.

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
recorded is **0.865**. That 3.5-point gap is the measured price at this scale.

*Caveat on all of Part 7:* injection jumps to the endpoint instead of walking there, and omits the
learning happening alongside. It settles the comparisons; it does not replace a trajectory for
calibration.

## §7.4 What is standard, what is ours, and what to claim

`ρ ≤ s·cos` **as a single inequality is ours**. None of its ingredients are: `1/√p` probe overlap
(classical ZO — Nesterov–Spokoiny, Duchi et al., Baydin et al.); signal linear / noise `√T` (standard
SGD noise ball); `Σρ_t = ∞`, `Σρ_t² < ∞` (Robbins–Monro 1951); step relative to `‖θ‖` (trust-region;
LARS/LAMB); a pool size beyond which pooling buys nothing (critical batch size, McCandlish et al.).

**Ours is the packaging:** collapsing those into an inequality between two quantities the server already
logs or can estimate, turning an asymptotic rate statement into an *online control law*. **Do not write
it up as a new theorem**, and **do not claim the control law while §6 is open.**

Where that leaves the paper claims:

| claim | verdict |
|---|---|
| **C1 · informed JVP-magnitude probe selection** | **Refuted as written; redirect it.** `b²/a = 1` is measured — selection is stability-neutral by construction. The **combination rule** is the contribution: "compute `P` directional derivatives and assimilate all of them" is the *trainer-side analogue of server-side pooling*. Rewrite C1 as a compute-budget claim |
| **C2 · async aggregation / K-C control** | **Reframed; controller still not demonstrated.** `ρ ∝ 1/√(K·I)` is measured and only the async path has a free `N` to spend, but the gate A/B has failed to produce a contrast twice. Honest current form: a cohort-width **requirement** — a stronger claim than a gate parameter |
| **C3 · aggregation weighting (ω)** | **Magnitude half: park** — ω spans 0.70–0.87 against a ≥10× gap, and the trust-ratio step removes it from magnitude entirely. **Freshness half: open and testable** (H-E) |
| **S1 · server momentum** | **Re-framed and now testable.** The historical NaN was arithmetic (`ρ_eff = ρ/(1−β)`), and the same bug sat in the code until momentum was moved *before* the trust-ratio scale. Hypothesis: junk is fresh each commit, signal persists, so temporal pooling raises `cos` by `√(1/(1−β))` at **zero** cost in `B` |
| **S2 · variance-gate recalibration** | **Superseded.** The gate is a working `N`-controller whose setpoint carries units. Replace the loop; do not re-tune the threshold |
| **S3 · aggregation-rate tempering** | **Subsumed and measured.** Under trust-ratio, `ρ` = `ρ*_t` regardless of ω |
| **Systems: flat memory, inference-only operator set** | **Untouched, and the strongest part of the paper.** The `cos` probe's backward pass is an audit flag, off by default, and does not touch the operator-set claim |
| **Cost framing** ("10× sync compute at P=10") | **Needs rewriting.** Under selection, raising `P` is *harmful*; under averaging the same `2P` buys `ρ/cos ∝ P` |

**Net effect:** all three ML claims move. **C1 shrinks** to a compute-budget justification that only
averaging redeems; **C2 grows** but relocates from "async" to "the controller that async makes
possible"; **C3 splits** into a parked half and an untested half. The systems contributions are
unaffected.

**Three things the work added that were not on the claim list:**

1. **`p` is a gradient-quality parameter, not just a memory knob.** For backprop FL, PEFT rank is a
   memory/communication knob and gradient quality is unaffected; for forward-gradient FL, **`cos ∝ 1/√p`
   — PEFT rank is the primary determinant of gradient quality.** Measured, and the only result here
   about the *model* rather than the optimizer, so it does not compete with FwdLLM's execution
   primitives.
2. **Scale-invariance as a design principle** (§5.3), with its cleanest demonstration: under the
   trust-ratio step, heterogeneity moves `ρ` by *zero to six significant figures* while still moving
   `var` by 1.6×.
3. **The `ρ ≤ s·cos` criterion is *unpaid*, not refuted.** What is publishable today is the **empirical
   dose-response curve**, which was not planned.

---

# Part 8 — Generality beyond FL

*Speculative relative to Parts 1–7, and **frozen** (R8): revisit only when the open questions close.*

## §8.1 The model is general; the controller is where the paradigm enters

```
cos   = sqrt( G_rule * n / p )      n = product of all INDEPENDENT pooling stages
rho  <= s * cos                     the step must not exceed the aim
n_req = p * (rho*/s)^2 / G_rule     what any configuration must pool
```

**Nothing in those three lines is federated.** They apply to any optimizer estimating a gradient from
directional derivatives. FL enters only in *how `n` decomposes and what each factor costs*: single
device `P` × accumulation (wall clock); centralized ZO `P` × accumulation (no comms); data-parallel
`P` × workers × accumulation (all-reduce); **federated `P` × `K` × `I`**, where `I` costs round trips
and `K` costs staleness and heterogeneity.

Likewise §4's two conserved quantities contain nothing federated — `B` is a property of the `ρ`
trajectory alone, and `Λ` needs only `ρ_t` and `cos_t`. **If they hold outside FL, they are the
transferable result.**

## §8.2 A testable prediction against centralized ZO

Centralized ZO fine-tuning uses `P = 1` and a very small fixed learning rate. The criterion predicts
*why*: with `G_rule = 1` and no `K` or `I`, `cos = √(1/p)`, so the stable relative step is ~10⁻³ and the
method needs ~`p` steps for coherent progress — matching the very long step counts such methods report.
**A falsifiable prediction about an existing published method, derivable with no new experiments.**
Check published curves before claiming it.

## §8.3 What must be checked before generalizing

| # | assumption this work rests on | why it might not survive outside our setting |
|---|---|---|
| 1 | **Orthogonality of steps to `θ`** (§2.1), which makes `B` exact | measured only at `p` ≥ 118k on adapter+head slices. At small `p`, or with layers of large initial norm and structured gradients, `⟨θ,Δθ⟩` need not vanish. **Check the orthogonality ratio first on any new model — one line of replay** |
| 2 | **`Λ`'s calibration** (0.6 → 0.85, saturating at 0.95) | fitted on agnews/DistilBERT. The *monotonicity* should transfer; the numbers are task- and model-specific |
| 3 | **`B_max`** | settled as *relative* here (§7.1), but its value is calibrated on one model family |
| 4 | **isotropic probes and the `1/√p` overlap** | exact by construction *given* Gaussian `v` over the whole trainable slice. Structured or block probes break it deliberately — that is their point, and they need their own analysis |
| 5 | **probe noise dominates data noise by ~40×** (**ANALYSIS**, M1) | this is what justifies "spend compute on `P`, not on bigger bins". At larger batch sizes or lower `p` the balance moves |
| 6 | **`h` pinned by fp16 cancellation** | a precision property, not a math one. bf16/fp32 move the usable `h` window and therefore the FD's faithfulness |
| 7 | **The closed-form `cos`** | functionally validated but never cleanly measured (§6). **No generalization claim about `cos` until H-P closes** |

## §8.4 Larger models and datacenter fine-tuning

**The case changes from memory to throughput.** On-device, forward-gradient wins because it stores no
activations. In a datacenter with backprop available, the only argument is that 2 forward passes are
cheaper than 1 fwd + 1 bwd and need no activation memory — real, but much narrower, and **`cos ∝ 1/√p`
makes it *worse* at scale** unless `p` is aggressively constrained (which is what PEFT does — and that
is now a gradient-quality decision, not an efficiency one).

**MoE is structurally interesting:** only active experts contribute to a forward pass, so `p_effective`
is per-token active parameters — **block-coordinate probing for free**, the one structural idea that
beats the `√(n/p)` barrier. Unanalysed, and the most interesting extension on this list.

## §8.5 Is there a general ML contribution here

In descending confidence:

1. **Yes: `p` is a gradient-quality parameter, not just a memory knob.** Inverts the standard PEFT
   intuition; measured.
2. **Yes: the scale-invariance requirement for forward-gradient training** (§5.3) — step, gate and
   estimator must all be ratios, or the method needs re-tuning per model.
3. **Maybe: the two conserved quantities** (§4). `B` is exact and parameter-free; `Λ` orders 21 arms.
   Transfer outside agnews/DistilBERT is untested (§8.3 rows 1–3).
4. **Maybe: the online control law** — `ρ ≤ s·cos` with a gradient-free estimator for `cos`. Novel as
   *packaging*, not theory, and **weakened by the probe defect**: what is publishable today is the
   dose-response curve.
5. **No: the scaling law itself.** `cos ∝ √(n/p)` is known ZO analysis.

A standalone paper needs three things we do not have: the criterion predicting the divergence point
across **≥3 models and ≥2 tasks**; a usable estimator for `cos`; and the derived controller **beating
hand-tuned schedules without re-tuning** when the model changes.
