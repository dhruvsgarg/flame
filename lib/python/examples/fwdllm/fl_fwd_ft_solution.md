# Forward-gradient fine-tuning — the divergence problem, the model, and the state of the fix

> **Scope.** The measurements here are FL (FluxTune / FwdLLM on agnews + DistilBERT), but the object of
> study is **backprop-free fine-tuning by directional derivatives**, of which FL is one deployment.
> Parts 1–6 are the FL instance; **Part 7 is the extension path** to single-device, centralized-ZO and
> datacenter settings, with the assumptions that must be re-checked to get there.
>
> Supersedes and replaces `FLUXTUNE_DIVERGENCE_HANDOFF.md` and `FLUXTUNE_PROBE_PLAN.md` (both deleted
> 2026-08-10; all content carried here — the probe plan's durable half is the instrument ladder in §6.2,
> and its superseded constants are listed in §4.5).

---

## §0 — How this document works

**Read order.** §1 what we are trying to solve · §2 what the shipped system measurably does · §3 the
model that explains it · §4 every arm and knob ever tried, with verdicts · §5 the fix as it currently
stands · §6 what runs next · §7 generality beyond FL.

**§3, §4 and §5 move together.** Evidence changes the model, the model changes the recommended
solution. Any edit that adds a finding to §4 must, in the same edit, update §3 if the model moved and
re-check §5 if the recommendation moved. Never leave a corrected model beside a stale recommendation.

**Maintenance rules.**

| # | rule |
|---|---|
| **R1** | **Edit in place.** Rewrite the row or paragraph that changed. No changelogs, no dated "update:" notes, no appended addenda. Git holds history. |
| **R2** | **One number, one home.** Every quantity has exactly one owning section; everywhere else cites `§x.y`. If you find yourself restating a number, delete the copy. |
| **R3** | **Status lives in two tables only** — §4.1 (arms run) and §4.3 (features built). No status tags anywhere else. Every run appears in §4.1 exactly once, with its run id. |
| **R4** | **Tag every claim** `MEASURED` (telemetry + run id) · `DERIVED` (arithmetic on measured inputs) · `ANALYSIS` (rests on a stated assumption) · `HYPOTHESIS` (§6.3). Untagged prose is not evidence. |
| **R5** | **Dead ends are permanent.** §4.4 is append-only and is the first thing to read before proposing anything. A knob that failed does not get re-proposed elsewhere in the doc; it gets a §4.4 row citing why. |
| **R6** | **Prediction before run.** An arm enters §6.4 with a number and a *sinking condition* written down first. No prediction ⇒ not ready to launch. |
| **R7** | **Cheapest instrument first** (§6.2). Log replay beats an offline rig beats a single-process replica beats a sim run beats a real run — by ~3 orders of magnitude in cost each step. A full run is what *confirms* a hypothesis, never what explores it. |
| **R8** | **Supersede by overwriting**, and add a row to §4.5 only if the stale number is quoted in another document or a paper draft. §4.5 is a quote-checking aid, not a history. |
| **R9** | **§7 is frozen** until §6.1 closes. Add to it only when something in Parts 1–6 forces a generality caveat. |
| **R10** | **Prose budget.** A paragraph that does not change a decision is deleted. No section summarises another section. Accuracy is a *result*, never the stability evidence (§2.7). |

---

# Part 1 — The problem

## §1.1 What FluxTune is for

Fine-tune an LLM across federated clients **without a backward pass**. Each client perturbs the
trainable weights along random directions, measures the loss change with forward passes only, and
uploads a scalar-scaled direction. The server pools them and steps. Nothing stores an activation
graph, so peak memory is a forward pass and the operator set is inference-only — that is the whole
point, and it is the part of the system nobody disputes (§5.5).

The cost of giving up backprop is that one measurement returns **one number instead of `p`** (§3.1).
Everything in this document is about whether enough of those numbers can be averaged, fast enough, to
train — and about the specific way our implementation fails to.

## §1.2 The failure

`run_20260804_003042` (real, 3.93 h, 189 commits), agnews / DistilBERT-base + AdapterHub adapters:

```
 0.00h r1 did=  0   acc 0.398  loss 1.380     <- init, ln(4) = 1.386
 2.13h r1 did=106   acc 0.846  loss 0.554     <- PEAK
 2.95h r1 did=146   acc 0.738  loss 0.692     <- degradation starts INSIDE round 1
 3.34h r2 did= 14   acc 0.527  loss 1.464     <- knee
 3.87h r2 did= 38   acc 0.250  mcc 0.000  loss 2.371
```

**Rise, peak, collapse to a single class.** A second 4 h run agrees on every derived constant, so this
is deterministic, not a seed. Runs cut at ~95 commits stop *at the peak*, which is why it looked
intermittent for months.

## §1.3 What it is not

| ruled out | evidence |
|---|---|
| overfitting | **train** loss diverges too: `stat_utility` 10.84 → 5.31 → 13.74, i.e. train loss 1.355 → 0.66 → 1.72 |
| losing signal / plateauing | final loss **2.37 > ln 4 = 1.386** — answering *wrongly and systematically*, worse than "I don't know" |
| a round-boundary or data-order effect | degradation starts inside round 1 |
| the finite-difference approximation | the FD displacement gets *relatively smaller* over the run (`h‖v‖/‖θ_tr‖`: 0.50 → 0.07) while the failure gets worse (H-C, §4.4) |
| a bad seed / flaky infra | two independent 4 h runs agree on every constant |
| something specific to async FluxTune | `fwdllm` (sync) shows the identical signature (§2.4) |

## §1.4 Definition of done

1. An arm reaches **peak ≥ 0.86** on agnews/DistilBERT and **ends within 0.015 of its peak** at ≥300
   commits, without hand-tuning per model or per α.
2. The setpoint that achieves it is **computed, not searched** — from the config, before the run.
3. Stability is readable in **~20 commits** from `B` and `Λ` (§3.4), not from a 4 h accuracy curve.
4. Every constant in the loop is dimensionless (§3.7), so nothing needs retuning when `p`, α, `K` or
   the round index changes.

Items 1 and 3 are met by several arms today (§4.1). **Item 2 is the open one** — the sizing formula's
`cos` disagrees with the measured one by 25–80× (§3.9), so `ρ*` is currently read off a dose-response
curve. Item 4 is met for the step rule and unmet for the commit gate.

---

# Part 2 — Diagnosis of the shipped system

## §2.1 The system under test

**MEASURED**, from `aggregator_config.json` + `expt_scripts/fluxtune_n10_smoke.yaml`:

```
population N_clients=100, alpha=1 (Dirichlet), agnews, DistilBERT-base + AdapterHub adapters
trainable p = 450,340  (0.67% of 67.4M; backbone frozen AND pre_classifier dropped)
train_batch_size = 8          -> one "data bin" = 8 samples, 150 bins per round
perturbation_count P = 10, h = 0.01, central finite difference, fp16/autocast, no_grad
probe selection: rank by |JVP|, coin-flip between top-2  (tc_transformer_trainer_distribute.py:481-483)
client selection: async_oort, c = 30, agg_goal K = 10, dynamic_kc.enabled = FALSE
commit gate: var_threshold = 0.3, var_stopping_policy = plateau (patience 3, rel_delta 0.15),
             max_iterations_per_data_id = 20
server step: theta <- theta - eta * (1/N_acc) * sum_k omega_k * d_k * v_k   (raw in-place SGD)
             eta = 0.01, constant to 4 digits for the whole run; no optimizer state of any kind
omega: grad_aware / base=new / align_gate=true / scale=0.4 [measured 0.702..0.865, median 0.818]
```

**Where the probe dimensions actually live** (**MEASURED**), and it matters everywhere (§4.2 `p` row).
`create_model` builds 1,040,932 trainable params, but for distilbert the trainer's `__init__` then runs
`self.model.add_module("pre_classifier", nn.Sequential())`
(`tc_transformer_trainer_distribute.py:217`) — **replacing that layer with an empty module before any
probe is drawn** — and `self.params` comes from `make_functional_with_buffers` *after* that:

```
                         create_model     PRODUCTION (post-:217)
pre_classifier              590,592          0        <- dropped, not frozen
adapters                    447,264    447,264
classifier                    3,076      3,076
p                         1,040,932    450,340        <- the number every cos uses
||theta_tr|| at init          20.33        13.33
h*||v|| = h*sqrt(p)          10.203        6.711
```

Consequences: `trainable_scope: adapters_only` is a **no-op** (the layer is already gone); the FD
displacement is 50% of `‖θ_tr‖`, not 76%; every `cos` prediction rose ×1.52 when this was found.

**`h` is pinned.** At fp16 the two forward losses agree to 1–2 significant figures at `h = 0.01`, so
`h` sits between truncation error above and catastrophic cancellation below. It cannot be reduced.

## §2.2 The defect, in three legs

### Leg 1 — every step is orthogonal to `θ`. **MEASURED.**

`‖θ+Δ‖² = ‖θ‖² + 2⟨θ,Δ⟩ + ‖Δ‖²`, so the ratio of observed norm growth to step energy has an exact null
at **1.000 = the step carries no component toward or away from where the model stands**. Measured per
25-commit block: **1.000 ± 0.005 in every block of every arm** — including arms that reach 0.86 and
hold it.

**Orthogonality is not itself the defect** (`g` is nearly perpendicular to `θ` at high `p`; any
optimizer shows this). What it establishes is that the norm grows by the *full* `‖Δθ‖` every commit,
with no cancellation — which turns the trajectory into the exact difference equation integrated in
§3.4. Growth per se is normal (adapters init near zero); what is atypical at `K = 10` is that it is
geometric and unbounded.

### Leg 2 — `ρ ∝ η/√N`, and nothing anneals it. **MEASURED.**

```
rho ∝ eta   (commit 1):   eta .01 -> .002 -> .0005
            rho              0.2004    0.0404    0.0101
            predicted        0.2004    0.0401    0.0100        <- 1% and 0%

rho ∝ 1/sqrt(N)  (commits 40-80, N = K * pool_size):
            K                10        20        30        50
            N               182       322       299       339
            rho * sqrt(N)   1.68      1.80      1.70      1.81  <- invariant to 4% over 1.9x in N
```

**`ρ ∝ 1/√(K·I)` is the most load-bearing measured law here.** The naive `ρ ∝ 1/√K` is **FALSIFIED**
(0.16 → 0.14 over `K` 10 → 50, not 0.07) because raising `K` makes the gate commit sooner, cutting `I`
from 18.5 to 8.2. It is `N`, not `K`, that sets the aim.

Geometric vs arithmetic growth is decided by the *absolute* step, visible in one column
(`‖θ_{t+1}‖² = ‖θ_t‖² + ‖Δθ_t‖²` from Leg 1):

```
K=50   2.04  2.04  2.03  2.05  2.03  2.02     <- constant to 1%  => ||theta||^2 linear, +4.16/commit
K=10   2.49  2.53  2.95  4.22  4.81  6.77  11.4  12.4   <- grows  => geometric, doubling ~68 commits
```

Divergence is **present at commit 1** and takes ~150 commits to become visible. Steps do not become
more wrongly aimed over time; what grows is the absolute step and the norm it is applied to.

**Why `|d|` grows — the gradient genuinely grows** (**MEASURED**, B1 probe). Backprop gradient norm on
a fixed batch tracks `‖θ_tr‖` with exponent ≈0.9 in the rising phase:

```
||theta_tr|| / ||g_backprop||, 50-commit block means:
  select N=200 (diverging) 18/3.20  27/4.11  40/5.04  59/8.78  92/13.93   then 113/11.26 125/10.95 (post-collapse)
  mean   N=200 (stable)    14/3.09  14/3.05  14/3.00  15/3.00  15/3.00  15/3.14
  mean   free  (mid)       16/3.16  22/3.37  28/4.00  33/4.92  40/4.94  47/4.61   <- saturating
```

This matches the ≈0.8 inferred from `rms|d|` through a completely different instrument. So the loop
*noise inflates the norm → bigger gradient → bigger absolute step* is **real**, and norm control is
potentially **curative, not cosmetic** (which raises the bar for the weight-decay control arm, §6.1 Q2).

### Leg 3 — the variance gate is dimensionally wrong, and is the accidental stabiliser. **MEASURED.**

The gate asks the right question — *have I pooled enough readings to trust this direction?* — with a
statistic that carries units. From 3,441 `[IterProgress]` lines over the reference run at `K = 10`:

```
commit reasons:  natural (var < 0.3) = 0     plateau = 105     cap(max_iter=20) = 81
bins that EVER reached var < 0.3:  0 of 186
achievable variance floor (median per-bin minimum): 0.415 (bins 0-20) -> 14.97 (bins 180-186)  = 36x drift
```

**36× is exactly (6.0×)².** What `var` computes (`fwdgrad_utils.py:186-211`) is the per-coordinate
variance between two half-means, which for two samples is exactly `‖G_A − G_B‖²/(2p)`; with `u = d·v`
and `‖v‖² ≈ p`:

```
var  =  ||G_A - G_B||^2 / (2p)   ~=   2 * b^2 * ||g||^2 / n
```

**Both legs of the gate's behaviour fall out of that one line.** `var ∝ ‖g‖²` is the units bug (hence
the `‖θ‖²` drift); `var ∝ 1/n` is why it is an `N`-controller at all.

```
                       K=10    K=20    K=30    K=50   |  eta=.002 (K=10)   |  K=10, a=100
natural (var<0.3)         0      77     128     148   |      1             |     96
plateau                  67      20       4       1   |     31             |    102
cap (max_iter=20)        83      28       0       0   |    118             |     38
mean commit iteration  17.8    15.4    11.1     7.2   |   18.7             |    8.7
```

1. **The gate targets `N`, not `I`.** Early in every K-arm realised `N = K·I` lands at 264–296
   regardless of whether `K` is 20, 30 or 50 — it hands back most of a `K` increase as fewer
   iterations. At `K = 10` the target is unreachable, `max_iter` binds, `N` pins at 185.
2. **Because the ruler has units of `‖θ‖²`, holding `var ≤ 0.3` forces `N ∝ ‖θ‖²`, hence `ρ ∝ 1/‖θ‖`,
   hence a constant absolute step.** Measured at `K = 50`: `I` 5.9 → 10.3, `N` 296 → 513 as `‖θ_tr‖`
   goes 15.2 → 27.3, with `‖Δθ‖` flat at 2.03.

> **The dimensional bug is the stabiliser.** Wrong units are exactly what turn a fixed threshold into a
> `ρ ∝ 1/‖θ‖` anneal — which, since `‖θ‖ ∝ √t` under a constant absolute step, is Robbins–Monro
> `ρ_t ∝ 1/√t` by accident. **It is not a fix:** `Σρ_t² = Σc/t` diverges *logarithmically*, so growth is
> deferred, not stopped (extrapolated collapse at commit 1,200–1,700), and `0.3` hard-codes one
> trajectory. Three objections stand against the statistic itself: it has units, it cannot see
> directions (it is computed from scalars), and it does not reference the step.

This is the precise departure from FwdLLM, whose central pooling mechanism this is: the replacement is
not "a better commit test" but **an explicit `N`-controller with `ρ` as its sensor** — which is what the
var gate turns out to have been all along.

## §2.3 The single defect

**Nothing in the pipeline is scale-invariant.** The estimator (`|d|` tracks `‖θ‖`), the step
(`‖Δθ‖ ∝ |d|`) and the gate (`var ∝ |d|²`) all inflate together, so no quantity anywhere can be
meaningfully compared against a fixed constant. All three legs are one violation of the ratio principle
(§3.7), with the twist that decides the fix: **the third violation partly cancels the first two**, so
the replacement must supply that anneal *deliberately* rather than merely remove the bug.

## §2.4 It is cross-baseline, not FluxTune-specific. **MEASURED — closes H-F.**

`fwdllm` (sync, `K = 10`, `c = 10`) over 131 commits: orthogonality ratio **1.000 ± 0.008**, `ρ`
**0.175** at commit 1, `‖θ_tr‖` 13.55 → 31.8 with `‖Δθ‖` growing — geometric, doubling every ~106
commits. **Same signature, same magnitude, shared `_server_update_step`.**

So the step rule and the anneal are **cross-baseline hygiene, not a FluxTune contribution** (§5.5). One
consolation prize: at matched commits `ρ·√N` is 1.68 for fluxtune vs 1.01 for fwdllm, which at matched
`‖g‖` (**ANALYSIS**) puts FluxTune's `|JVP|` selection at `G_rule ≈ 3` and **FwdLLM's cosine-similarity
probe selection at `G_rule ≈ 1`, i.e. no better than random.**

## §2.5 Heterogeneity is a step-size multiplier. **MEASURED**, 1000× in α.

Means over commits 40–80:

| arm | run | `I` | `N` | `ρ` c1 | `ρ`·√N | `‖Δθ‖` | `‖θ_tr‖` end | `var` floor | nat/plat/cap | peak | final |
|---|---|---|---|---|---|---|---|---|---|---|---|
| α=0.1 K=10 | `225718` | 18.9 | 190 | 0.237 | 1.95 | 4.39 | **149.7** | 1.33 | 0/108/87 | 0.828 | **0.250** |
| α=1 K=10 | `004203` | 18.2 | 182 | 0.200 | 1.67 | 3.07 | 91.4 | 0.64 | 0/114/85 | 0.853 | **0.250** |
| α=100 K=10 | `012201` | 8.7 | 87 | 0.147 | 0.91 | 2.02 | 45.7 | 0.28 | 96/102/38 | **0.865** | 0.838 |
| α=0.1 K=20 | `001241` | 19.4 | 388 | 0.163 | 2.13 | 2.38 | 28.6 | 0.38 | 1/38/67 | 0.848 | 0.848 |
| α=1 K=20 | `031606` | 16.1 | 322 | 0.157 | 1.79 | 2.11 | 28.8 | 0.30 | 76/20/29 | 0.860 | 0.858 |
| α=100 K=20 | `023623` | 4.6 | 92 | 0.148 | 0.93 | 1.95 | 31.8 | 0.26 | 205/3/2 | **0.868** | 0.865 |

**(a) α enters through the gradient scale, and only through it.** At commits 0–10, where
`‖θ_tr‖ = 13.5` in every arm, `rms|d|` is **7.7 / 6.1 / 3.1** at α = 0.1 / 1 / 100 — a 2.5× spread from
data alone, propagating straight into `ρ` (0.237 / 0.200 / 0.147). **Under raw SGD, heterogeneity is a
hidden ×1.6 on the relative step.**

**(b) The invariant is not `ρ·√N`** (that spans 0.91–2.13 across α). What is invariant to ±1.5% across
all seven arms is the form with the gradient scale divided out:

```
rho * sqrt(N) * ||theta_tr|| / ( eta * rms|d| * sqrt(p) )  =  0.53 +- 0.01
      a=0.1/1/100 x K=10/20:  0.533 0.535 0.535 0.527 0.538 0.537 0.523
```

**(c) The gate's liveness is set by `‖g‖`, so α decides it as much as `K` does.** `var_threshold = 0.3`
sits *exactly* on the α=1, `K`=20 floor — tuned to one point of a two-dimensional surface. `K = 10`
survives at α = 100 and `K = 20` does **not** stabilise at α = 0.1, so **"`K ≥ 20` is safe" is false as
stated**; the true statement is "the gate is safe while it can reach its threshold".

**(d) The trust-ratio step removes the α multiplier by construction — MEASURED.** At `ρ* = 0.06`,
α = 1 vs α = 0.1 give `ρ` **identical to six significant figures at every commit** (0.059892 vs
0.059882 at c1; 0.014404 vs 0.014404 at c300) while `var` still shows the 1.6× gradient-scale ratio
(0.0122 vs 0.0195). **The mechanism is still there; the step no longer feels it.** The residual 3-point
accuracy gap (0.775 vs 0.804) is data difficulty at a matched step — what a portable optimizer should
look like.

## §2.6 Why it looks healthy for three hours, and what the collapse is

Early on the coherent term (`T·ρ·cos`) beats the `√T` noise, so **the accuracy climb is real learning**;
the inflation term is exponential the whole time but invisible until it is not. `cos` does not degrade
over time. **We never reached a minimum and walked back out** — accuracy peaked because a rising linear
term and a falling term crossed. *A convergence-detection rule still does not exist* (§6.3).

**The collapse is directional degeneracy, not logit saturation** (**MEASURED** — revises H-D). At
collapse `top_class_share → 1.00` but `logit_norm` is 2.5–3.6 — *the same as the healthy `K = 50` arm at
2.86* — and prediction entropy stays high at 0.8–1.3 vs `ln 4 = 1.386`. **The head's decision direction
is destroyed while its scale is unremarkable.** The objection *"some of the weight increase was in the
right direction"* is answered by arithmetic: ≤1.5% of each step is aligned, so the coherent part
accumulates to a fraction of one norm over ~190 commits while the norm grows 7×.

## §2.7 The three production monitors

**MEASURED**, α-independent across all six §2.5 arms and every arm in §4.1:

| monitor | reading |
|---|---|
| **`‖θ_tr‖`** | peak accuracy at **28–39**; −5 points at **47–52**; `top_class_share > 0.9` at **62–73**. *The one worth wiring to an alarm.* |
| **`ρ = ‖Δθ‖/‖θ_tr‖`** | exact per commit, logged in `server_update`. Feeds `B` and `Λ` (§3.4) |
| **`top_class_share`** | 1.00 is the collapse fingerprint; `logit_norm` does **not** discriminate |

Scoring rule: **`B` and `Λ` (§3.4), never accuracy, never the `‖θ‖²` log-log slope** (dead, §4.4).
Both are exact at any horizon and readable in ~20 commits.

---

# Part 3 — The model

*This is the current model, incorporating everything in §4. It is scored against, not decorated with,
history — corrections are folded in, not annotated.*

## §3.1 One probe, one scalar

| | backprop | forward-gradient (here) |
|---|---|---|
| cost of one measurement | 1 fwd + 1 bwd ≈ 3 forward-equivalents | **2 forward passes**, no backward |
| what it returns | all `p` components of `g`, exactly | **one scalar** `d = ⟨g,v⟩` — the slope along `v` |
| memory | stores activations | nothing beyond a forward pass |
| the update it produces | `−η·g` — right direction, known length | `−η·d·v` — right **on average**, ~0.1% aligned individually |
| noise sources | data sampling | data sampling **+ which direction you asked about** |

`ĝ = d·v` is unbiased (`E[d·v] = g`) with enormous variance. *The image:* on a hillside in fog,
backprop feels the slope in every direction at once; forward-gradient picks one random direction, takes
a test step, feels whether it went up or down, steps back. **One reading is nearly worthless; a thousand
averaged readings are a slope meter.**

Four consequences:

**(a) The parameter count is the adversary.** A random direction in `p` dimensions overlaps any fixed
target by ≈ `1/√p`; at `p = 450,340` one probe is ~0.15% signal. No tuning removes this — and `p` is
itself a *lever* (§4.2).

**(b) Averaging is the only free lever, and it pays twice.** Over `n` independent readings the signal
adds **linearly** while near-orthogonal noise adds **in quadrature**. The average is simultaneously
**better aimed** *and* **shorter** — safety improves as `1/n`, not `1/√n`.

**(c) Misaim never cancels in length.** A step `Δ` perpendicular to `θ` gives `√(‖θ‖²+‖Δ‖²) > ‖θ‖` for
**every** perpendicular direction; sign does not matter, only length. This is Leg 1 (§2.2) and it is
what makes §3.4 exact.

**(d) Inflated weights destroy a classifier by randomising which class wins, not by saturating it** —
see §2.6.

## §3.2 Symbols

Per-commit unless stated. "Dimensionless" means a pure ratio, so comparing it against a fixed constant
is legitimate — that property is the whole fix (§3.7).

| symbol | what it is | value here |
|---|---|---|
| `θ_tr` | the **trainable** slice (adapters + classifier; `pre_classifier` is dropped, §2.1) | `p` = **450,340**; `‖θ_tr‖` = **13.35** at init |
| `p` | trainable dims = probe dimension | **450,340** (`rf`=16); 229,012 (`rf`=32); 118,348 (`rf`=64) |
| `v` | one probe direction, raw Gaussian draw, **not** normalised | `‖v‖ = √p ≈ 671` |
| `h` | finite-difference spacing | 0.01 → displacement `h‖v‖ = 6.71` — **an instrument, not a step size** |
| `d` | the JVP: a **scalar** per probe, `d ≈ ⟨g,v⟩` | rms 3.4 → 15.9 over a diverging run |
| `g` | true gradient at `θ` | measured by the B1 probe (§3.9) |
| `G` | server's pooled update before the step | the direction actually taken |
| `P` | probes per trainer per iteration | 10 |
| `K` | trainers pooled per commit (`agg_goal`) | 10 |
| `I` | iterations over the **same data bin** before committing; equals `pool_size` exactly | 18.5 at `K`=10 (capped), 8.2 at `K`=50 |
| `N = K·I` | uploads pooled server-side | 185 at `K`=10; 300–500 at `K`≥20. **Not** the client population (100) |
| `C` | concurrency pool (`c`) — caps `K` | 30 |
| `η` | server learning rate (a **knob**, superseded by `ρ*`) | 0.01 |
| `ω` | per-upload aggregation weight | 0.70–0.87 — a re-weighting, not a step size |
| **`ρ`** | **relative step `‖Δθ‖/‖θ_tr‖`** — an **outcome** under raw SGD, a **knob** under trust-ratio | **0.16 at commit 1** shipped |
| `ρ*` | the relative step an operator *sets* under the trust-ratio rule | the knob `ρ` should have been |
| **`cos(G,g)`** | fraction of the step aligned with the true gradient | **0.036–0.067 closed form; 0.0004–0.003 measured** (§3.9) |
| `E[v∥²]` | probe-selection gain (§3.3) | 2.988 at `P`=10, 4.744 at `P`=30 |
| `G_rule` | pooling gain of the combination rule: `E[v∥²]` if selecting, `P` if averaging | 2.988 → 10 |
| `a`, `b` | estimator shape constants (§3.3) | properties of the **rule**, not the data |
| **`B`** | **budget spent** = `½Σln(1+ρ_t²)`; `‖θ_T‖/‖θ_0‖ = e^B` | §3.4 |
| **`Λ`** | **progress banked** = `Σρ_t·cos_t` | §3.4 |
| `Φ` | inflation `= e^B` | ≤3.63 holds, ≥4.2 degrades (§3.4) |
| `var` | commit gate's statistic: spread of `d` across the pool | drifts with `‖θ‖²` (§2.2 Leg 3) |

**`v` is a raw Gaussian draw** — each coordinate `N(0,1)`, `torch.randn_like`, never normalised
(`tc_transformer_trainer_distribute.py:416`). Three consequences: `‖v‖` concentrates at `√p` to 0.07%,
so **normalising `v` is a no-op for `cos`**; it sets the probe displacement `h√p = 6.71`, so **changing
`p` silently changes the FD spacing** (controlled by `FWDLLM_FD_SCALE_INVARIANT=1`); and **isotropy is
exact by construction**, which licenses `1/√p` above.

## §3.3 Estimator shape, and pooling

Split any upload `u` into its component along `ĝ = g/‖g‖` and the rest: `u = α·ĝ + u⊥`.

- **`a` measures the shadow**: `E[α] = a·‖g‖`.
- **`b` measures the total length**: `E‖u‖ = b·‖g‖·√p`, so **`b = 1` is "as long as one raw probe"**.

A single upload's aim is `cos(u,g) = (a/b)/√p`. For the three rules that matter (setting `‖g‖ = 1`):

| rule | shadow `⟨u,ĝ⟩` | length `‖u‖` | `a` | `b` | `a/b` | `b²/a` |
|---|---|---|---|---|---|---|
| one raw probe `d·v` | `v∥²` | `\|v∥\|·√p` | 1 | 1 | 1 | 1 |
| **select best of P by `\|d\|`** | `v∥²` of winner | `\|v∥\|·√p` of winner | `E` | `√E` | `√E` | **1** |
| **average all P** | `1` (unbiased) | `√(p/P)` | 1 | `1/√P` | `√P` | **1/P** |

The middle row answers *"why doesn't picking the best probe help?"* — selecting on `|d|` raises the
shadow **quadratically** (`a = E`) and the length **linearly** (`b = √E`). Aim `a/b = √E` genuinely
improves, but stability depends on `b²/a`, in which the two cancel **exactly, for any `E`**. `a` and `b`
are properties of the rule, **known in closed form before the run**; the one empirical input is that `d`
is Gaussian, verified in §4.2.

**Pooling `n` uploads:** the shadow is identical in each and survives untouched; the perpendicular junk
shrinks by `√n`.

```
shadow of the pooled G   ~  a * ||g||                    <- unchanged by pooling
||G||                    ~  b * ||g|| * sqrt(p/n)        <- shrinks as 1/sqrt(n)
cos(G, g)                =  shadow / length = (a/b) * sqrt(n/p)
rho                      ~  eta * ||G|| / ||theta||      <- also shrinks as 1/sqrt(n)
  =>  rho / cos          ~  (b^2/a) * p / n
```

Read the last line as **safety = (a property of the rule) × (dimensions per pooled reading)**. Improve
it via a better rule (`b²/a`), fewer dimensions (`p`), or more readings (`n`).

**Status: the length half is MEASURED and holds; the shadow half does not** — §3.9.

## §3.4 The two conserved quantities. **MEASURED, 21 arms — the core of the model.**

Leg 1 (§2.2) is an exact difference equation: `‖θ_{t+1}‖² = ‖θ_t‖²(1+ρ_t²)`. Integrating it gives, with
no free parameter:

```
INFLATION   Phi = ||theta_T|| / ||theta_0||  =  exp( B ),   B = (1/2) * sum_t ln(1 + rho_t^2)
PROGRESS    Lambda = sum_t rho_t * cos_t,     cos_t = sqrt( G_rule * N_t / p )
EFFICIENCY  Lambda / B  =  2 * cos / rho      <- exact by construction
```

**The norm law holds to <0.3% on every bounded arm** (<4% even on fully collapsed ones, where Leg 1's
ratio drifts), across 21 arms spanning `ρ` 0.0002–0.22, `N` 10–200, α 0.1–1, both combination rules,
both step rules, both gates, `p` 118k–450k, and 177–1,273 commits (§4.1). **It contains no `cos`, no
`N`, no rule, no α and no `p`: `‖θ_T‖` is a function of the `ρ` trajectory and nothing else.** The
sharpest case is `013917`, where `N` fell 200 → 40 under the annealed gate and the law still fit to
0.02%.

**The progress law** is the same 21 arms sorted by `Λ`: peak accuracy is monotone from 0.377 to 0.865
with no exception outside replicate noise, and saturates at `Λ ≈ 0.95`. Read `Λ` as **the accumulated
aligned displacement, in units of `‖θ_tr‖`** — you must travel about 0.6 of your own length in the right
direction to reach 0.85, and about one length to saturate. Calibration for agnews/DistilBERT:
`Λ` 0.07 → 0.49, 0.20 → 0.70, 0.40 → 0.80, 0.60 → 0.85, 0.95 → 0.865.

**Whether an arm *holds* its peak is decided by `Φ` and nothing else.** Every arm at `Φ ≤ 3.63` ends
within 0.015 of its peak; every arm at `Φ ≥ 4.2` degrades, monotonically in `Φ`: 0.672 (4.23), 0.761
(4.47), 0.353 (5.74), 0.251 (9.47). **The two failure modes are separately diagnosable before a run
ends: too little `Λ` = never learned; too much `B` = learned and then lost it.**

Three consequences that restructure everything downstream:

**(a) There is no critical `ρ`.** Under a pinned `ρ`, `Φ = (1+ρ*²)^{T/2}` for *any* `ρ* > 0` — geometric
always, with doubling time `1.4/ρ*²`. The apparent boundary at `ρ ≈ 0.09` was **a horizon artifact**:
the locus where doubling time falls below run length. **`ρ` is not a threshold; it is the rate at which
a fixed budget is spent.**

**(b) The `‖θ‖²` log-log slope carries no stability information under trust-ratio** — it is bounded
above by 1 *by construction*, so every "sub-linear ⇒ safe" reading on such an arm is vacuous. Under
`raw_sgd` a slope > 1 does mean something: the absolute step is outgrowing `‖θ‖`, i.e. the `|JVP|`
feedback of Leg 2 is live.

**(c) The closed-form `cos` is functionally validated even though the probe disagrees with it 25–80×.**
`Λ` is built from the *predicted* `cos` and it collapses arms spanning `G_rule` 2.988 vs 10, `p`
118k–450k and `N` 10–200. **A formula that wrong could not order 21 arms** — which raises the prior
sharply that the probe's reference is the problem, not the formula (§3.9). *Caveat that sets the next
experiment:* the rule and `p` variation all sits at `Λ > 0.8`, where accuracy has saturated, so
**`Λ`'s rule- and `p`-scaling is untested in the steep region** (§6.1 Q3).

## §3.5 The criterion

Two clocks. **The deadline:** coherent progress grows as `T·ρ·cos`, noise displacement as `√T·ρ`; signal
overtakes noise at `T ≈ 1/cos²`. **The budget:** the norm inflates by `(1+ρ²)^{T/2}`, doubling at
`T ≈ 1.4/ρ²`. You survive iff `1/cos² ≤ 1.4/ρ²`:

> ## ρ ≲ s · cos(G, g)
> **The relative step must not exceed the fraction of it that is actually aligned with the gradient.**

The O(1) constants that derivation drops are now measured. Eliminating `T` between the two laws of §3.4
gives `B = Λ·ρ/(2cos)`, so fitting a target `Λ_req` inside a budget `B_max` requires

```
rho / cos  <=  2 * B_max / Lambda_req   =   2.6 .. 4.3       (B_max = 1.22-1.28, Lambda_req = 0.6-0.95)
```

**`s = 2.6–4.3`, not the 0.3–0.5 originally derived** — a factor of ~8, which is exactly the 5–10×
measured empirically (§4.5). Progress per commit is `ρ·cos`, increasing in `ρ`, so you want `ρ` as large
as the budget allows; but per §3.4(a) there is no cliff, only an exchange rate.

**The design rule.** For a fixed learning target `Λ_req` over `T` commits, Cauchy–Schwarz makes constant
`ρ = Λ_req/(T·cos)` the budget-minimising trajectory, at cost `B = Λ_req²/(2T·cos²)`:

> **Budget cost for a fixed amount of learning is ∝ `ρ`; wall-clock cost is ∝ `1/ρ`. Pick the smallest
> `ρ` whose commit count you can afford.** There is nothing to tune — `ρ` is the exchange rate between
> wall clock and safety margin, and `cos` (via `P`, `K`, `I`, `p`) sets the exchange rate itself.

## §3.6 The lever table — this ranks every possible fix

**`ρ/cos` is "will this survive"; `ρ·cos` is "how fast does it learn".** A good lever improves the first
without hurting the second. Pooling is the only free lever, because it shortens the step and improves
the aim by the same `√n`.

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
pools are strictly better for stability per FLOP and slower only in absolute wall-clock progress.**

**The `p` row is `√p`, not `p`** (**MEASURED**, §4.2): `ρ` does not fall as `√p` because both halves of
`ρ = ‖Δθ‖/‖θ_tr‖` scale as `√p` and cancel. **`ρ` is already dimensionless in `p`**, so shrinking `p`
buys `cos` alone.

**Three pooling stages, and who owns each:**

| stage | owner | cost structure | reduces |
|---|---|---|---|
| `P` probes per iteration | **trainer** | on-device, parallel, free in wall clock | probe noise |
| `K` trainers per commit | **client selection** (`agg_goal`, `c`) | parallel across devices | probe + data noise |
| `I` iterations per bin | **aggregation** (commit gate, `FedSgdAggregator.py:450-534`) | **serial round trips** | probe noise only |
| the step itself | **aggregation** | free | — |
| `p` | **model/PEFT design**, upstream of all three | negative | — |

`P` and `I` are *substitutes* — both pool over the same bin at the same `θ`, both reduce probe noise
only — at wildly different cost. **Raise `P`, lower `I`, hold `n = P·K·I`: commits get strictly faster
and no less well-aimed.** And wall-clock per commit is set by `I` (serial), not `K` (parallel), so
**choose the smallest `I` the gate allows and buy the rest with `K`**.

> **In one line: pooling sets how large `ρ` is *allowed* to be; aggregation *spends* within that budget.
> Aggregation can only shrink `ρ`; it can never raise `cos`.**

The natural reading — *"it's an aggregation bug"* — is **half right, and acting on that half alone costs
a ≥10× slowdown**: `K` ≥ 20 reaches **0.860** while `η` = 0.002 reaches **0.601 at commit 120 / 0.815
after 327** — same stability, one at zero progress cost and one at the full 1:1 cost. **And pooling
alone runs out**: at `K` ≥ 20 it buys ~500 commits before the linearly-growing norm walks back out
(§6.3 H-J).

**Why the controller spans two subsystems.** `N = K × I` is a client-selection knob times an aggregation
knob, so the gate (which decides `I`) is a client-selection decision made from aggregation telemetry —
**the controller's sensor and actuator sit on opposite sides.** A pure-selection controller has nothing
to measure (which is why today's `dynamic_kc_policy.py` targets `target_iter_per_data_id: 15`, a
heuristic unconnected to the estimator); a pure-aggregation controller can clamp `ρ` but cannot tell
whether it is clamping too hard.

## §3.7 The ratio principle

Every quantity compared against a fixed constant must be **scale-invariant**, because anything with
units silently changes meaning as training proceeds. The test: *re-parameterise so `‖θ‖` doubles; the
loss surface shape is unchanged, so the trajectory should be.* Any rule that fails needs re-tuning every
time the model, adapter rank, or round index changes. Three replacements follow: absolute step `η` →
relative step `ρ*`; absolute variance threshold → an `N` target; absolute iteration cap → a measured
adequacy condition.

**Why the criterion does not have to enumerate its failure modes.** Anything not thought of enters
through exactly **two** channels: it changes the step taken → shows up in **`ρ`**, logged exactly; or it
degrades the pool → shows up as **`n_eff < n`**, hence in `cos`. **Channel 2 is the unpaid half.**
`n_eff` detects *correlation* among uploads and returns `1.00 ± 0.01` in every arm ever replayed, so it
carries no information; it does **not** detect *directional* disagreement, because at `p = 4.5e5`
differing `g_k` move `var` by `O(n/p)` and are invisible. **Scale-free setpoints therefore come from the
step rule, not from channel 2.**

## §3.8 Sizing a configuration

```
G_rule = E[v_par^2]   for select-one-of-P     (P does not appear -- see 4.2)
G_rule = P            for average-all-P

cos           = sqrt( G_rule * N / p )        N = K*I uploads pooled server-side
N_req( rho* ) = p * (rho*/s)^2 / G_rule       <- pool needed to make rho* safe
```

| config | `G_rule` | `p` | `N` needed at ρ\*=0.16 | at ρ\*=0.05 | at ρ\*=0.02 |
|---|---|---|---|---|---|
| shipped (select 1 of 10) | 2.99 | 450,340 | 3,860 | 377 | 60 |
| + average all 10 | 10 | 450,340 | 1,150 | 113 | 18 |
| + average + adapter `rf`=64 | 10 | 118,348 | 303 | **30** | 5 |

> **The absolute values in this table are not usable** — §3.9 measures `cos` 25–80× below the formula,
> so `N_req` (∝ `1/cos²`) is optimistic by ~3 orders of magnitude. **Ratios between rows survive
> (`ρ ∝ 1/√N` is measured to 5%); absolute `N` does not.** Size `ρ*` from the §4.1 dose-response curve
> until §6.1 Q3 closes.

## §3.9 What the model does not explain — the `cos` anomaly. **MEASURED 2026-08-10.**

Three arms carried `--cos-ground-truth-audit`: a fp32 backward pass on a **fixed 64-sample held-out
batch**, server-side, at `θ_t` *before* the step, emitted per commit
(`FedSgdAggregator._cos_probe_gradient`). Index alignment is unit-tested (`test_cos_probe.py`: pool =
±`g` → cos = ±1.0000). Trainers already run the FD with dropout off, so the probe's `model.eval()`
matches — that confound is closed.

| arm | run | `N` | commits | **measured `cos`** (run-pooled ± SEM) | closed form | meas/pred | **`‖G‖/‖g‖`** meas | pred `b√(p/N)` |
|---|---|---|---|---|---|---|---|---|
| `select`, pinned | `013806` | 200 | 328 | **0.00314 ± 0.00034** | 0.0364 | 0.086 | 90.9 | 82.0 |
| `mean`, pinned | `042027` | 200 | 312 | **0.00132 ± 0.00022** | 0.0666 | 0.020 | 14.3 | 15.0 |
| `mean`, free gate | `065837` | ~30 | 715 | **0.00044 ± 0.00011** | 0.025 | 0.018 | 41.6 | 39.8 |

**Confirmed: the length model.** `‖G‖/‖g_probe‖` matches `b·√(p/N)` to 5–10% in all three arms, across
6× in `b` and 6.7× in `N`. So §3.3's `b`, and the assumption that client gradients have roughly the
norm of the held-out one, both hold.

**Not confirmed: the shadow model.** Implied `a` = 0.093 / 0.007 / 0.007 instead of 1. Three independent
consistency checks fail:

1. **`cos` is not constant where the closed form says it must be.** In the stable pinned-`mean` arm
   (`N`, `p`, rule all fixed) it **rises 11×** over 300 commits: 0.00027 / 0.00053 / 0.00067 / 0.00145 /
   0.00191 / 0.00289 per 50-commit block — tracking accuracy 0.3 → 0.85, not anything in `(a/b)√(N/p)`.
2. **The direct `G_rule` measurement misses.** `mean`/`select` at matched `N` = 200 should give
   `√(10/2.988)` = 1.83. Over the first 30 commits (the only window where `‖θ_tr‖` matches, before
   `select` diverges) it is **0.00032 ± 0.00057 vs 0.00026 ± 0.00048** — indistinguishable from each
   other and from zero. **Inconclusive, not falsified**: needs a `select` arm that does not run away.
3. **`ρ/cos` on the measured `cos` anti-orders the arms.** The best arm in twenty (`mean`, free gate,
   0.865 peak) sits at `ρ/cos ≈ 200`; the arm that collapses to 0.251 sits at **90**.

**What is and is not assumed here.** **Isotropy of `v` is exact by construction and is not the
caveat.** The caveat is **independent, homogeneous pooling**: `cos = (a/b)√(n/p)` assumes the `n`
readings are independent and share one target `g`. They do not — the `I` iterations over a bin share
*that bin's* gradient, and the `K` trainers have genuinely *different* gradients. `n_eff` excludes the
**correlation** half of that (§4.2), which leaves the **disagreement** half tangled with the probe's own
reference error. **Separating those two is the next measurement.**

**The leading suspect is the reference, not the pooling** (H-M). `g_probe` is one 64-sample **test**
batch; the uploads estimate 100 clients' **training** bins. The measured quantity is
`cos(G, g_train)·cos(g_train, g_probe)` and the second factor is unmeasured — an attenuation of ~0.03
reconciles everything, and its rise during training explains check 1 exactly. §3.4(c) backs this
independently. **The control is ~5 lines (§6.4 B13): a second disjoint held-out batch logging
`cos(g_A, g_B)`, which bounds the attenuation as `≈ √cos(g_A,g_B)`.**

**Operative reading until then:** `cos` is bounded above by the closed form and below by ~0.0005;
the criterion is **not an enforceable online budget at either end**; `ρ*` comes from §4.1's
dose-response curve. Note what this costs: `ρ` is exact per commit, but the quantity it must be compared
against has **no usable online estimator** (the gradient-free split-half route is dead, §4.4).
**Closing that gap is the difference between a design rule and a control law** — the single most
valuable open problem here.

## §3.10 What is standard, and what is ours

`ρ ≤ cos` as a single inequality is **ours**. None of its ingredients are: `1/√p` probe overlap
(classical ZO — Nesterov–Spokoiny, Duchi et al., Baydin et al.); signal linear / noise `√T` (standard
SGD noise ball); `Σρ_t = ∞`, `Σρ_t² < ∞` (Robbins–Monro 1951); step relative to `‖θ‖` (trust-region;
LARS/LAMB); a pool size beyond which pooling buys nothing (critical batch size, McCandlish et al.).

**Ours is the packaging:** collapsing those into an inequality between two quantities the server already
logs or can estimate, turning an asymptotic rate statement into an *online control law*. **Do not write
it up as a new theorem**, and do not claim the control law while §3.9 is open.

---

# Part 4 — Evidence: everything tried, and how it scored

## §4.1 Master arm ledger — every arm ever run

**The single source of truth for run results (R3).** Sorted by `Λ` (progress banked, §3.4), which is the
ordering that predicts peak accuracy. `Φ` pred is `e^B`; `Φ` obs is `‖θ_T‖/‖θ_0‖` — their agreement is
the norm law. `‖θ_0‖` = 13.35 (`rf`=16), 9.6 (`rf`=32), 6.86 (`rf`=64). All arms are α=1, `mean`/`rf`=16
unless the row says otherwise.

| `Λ` | arm | run | `T` | `ρ` c1 | `ρ` c40-80 | `B` | `Φ` pred → obs | `‖θ_tr‖` end | peak | final |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.008 | `n_target` rm ρ*=.01 e=.55 **@ α=0.1** | `001008` | 1263 | 0.0100 | 0.0011 | 0.0002 | 1.000 → 1.000 | 13.35 | 0.388 | 0.264 |
| 0.008 | `n_target` s=0.4 rm ρ*=.01 | `220627` | 1273 | 0.0100 | 0.0011 | 0.0002 | 1.000 → 1.000 | 13.35 | 0.394 | 0.274 |
| 0.009 | `var` gate, rm ρ*=.01 | `200209` | 1249 | 0.0100 | 0.0011 | 0.0002 | 1.000 → 1.000 | 13.35 | 0.377 | 0.282 |
| 0.016 | `rm` ρ*=.02 e=.55 | `223510` | 186 | 0.0200 | 0.0021 | 0.0007 | 1.001 → 1.001 | 13.36 | 0.379 | 0.329 |
| 0.068 | `const` ρ*=.01 | `211736` | 186 | 0.0100 | 0.0100 | 0.0092 | 1.009 → 1.009 | 13.47 | 0.488 | 0.429 |
| 0.199 | `rm` ρ*=.03 e=.25 | `013843` | 318 | 0.0300 | 0.0108 | 0.0150 | 1.015 → 1.015 | 13.55 | 0.695 | 0.662 |
| 0.369 | `mean` raw, N=200 | `211800` | 179 | 0.0346 | 0.0317 | 0.0850 | 1.089 → 1.088 | 14.53 | 0.804 | 0.793 |
| 0.393 | gate `setpoint` ρ*=.06 | `035557` | 312 | 0.0599 | 0.0216 | 0.0592 | 1.061 → 1.061 | 14.18 | 0.804 | 0.804 |
| 0.398 | `rm` ρ*=.06 e=.25 **(Q3 anchor)** | `035045` | 317 | 0.0599 | 0.0216 | 0.0597 | 1.061 → 1.062 | 14.20 | 0.801 | 0.799 |
| 0.398 | gate `setpoint` ρ*=.06 **@ α=0.1** | `062213` | 317 | 0.0599 | 0.0216 | 0.0597 | 1.062 → 1.062 | 14.19 | 0.775 | 0.770 |
| 0.460 | gate `annealed` ρ*=.06 (`N` 200→40) | `013917` | 707 | 0.0599 | 0.0216 | 0.0913 | 1.096 → 1.096 | 14.65 | 0.821 | 0.819 |
| 0.591 | `rm` ρ*=.09 e=.25 | `060834` | 313 | 0.0896 | 0.0324 | 0.1332 | 1.142 → 1.145 | 15.34 | 0.846 | 0.843 |
| 0.631 | `mean` raw, N=200 **+cos** | `042027` | 312 | 0.0344 | 0.0317 | 0.1434 | 1.154 → 1.158 | 15.46 | 0.849 | 0.849 |
| 0.819 | `raw_sgd` control (`select`) | `200325` | 177 | 0.1837 | 0.1167 | 1.4172 | 4.12 → 4.23 | 57.43 | 0.855 | **0.672** |
| 0.837 | `select` raw, N=200 | `200242` | 179 | 0.1861 | 0.1225 | 1.4687 | 4.34 → 4.47 | 60.72 | 0.853 | **0.761** |
| 0.891 | `select` `rf`=16 (`p` ladder) | `200358` | 195 | 0.2008 | 0.1228 | 1.7146 | 5.55 → 5.74 | 78.11 | 0.852 | **0.353** |
| 0.937 | **`mean` raw, free gate +cos** | `065837` | 715 | 0.1202 | 0.0859 | 1.2810 | 3.60 → 3.63 | 48.83 | **0.865** | **0.862** |
| 0.953 | `mean` raw, free gate (replicate) | `223446` | 693 | 0.1202 | 0.0859 | 1.2776 | 3.59 → 3.62 | 48.60 | 0.864 | 0.850 |
| 1.014 | `select` `rf`=32 `p`=229012 | `212009` | 187 | 0.1941 | 0.1099 | 1.1647 | 3.20 → 3.28 | 31.59 | 0.857 | 0.851 |
| 1.353 | `select` raw, N=200 **+cos** | `013806` | 328 | 0.1861 | 0.1207 | 2.2086 | 9.10 → 9.47 | 128.59 | 0.855 | **0.251** |
| 1.368 | `select` `rf`=64 `p`=118348 | `222817` | 188 | 0.1863 | 0.1072 | 1.0939 | 2.99 → 3.04 | 20.83 | **0.859** | 0.852 |

*The α-sweep arms (`225718`, `004203`, `012201`, `001241`, `031606`, `023623`) and the 08-07 K/η/P sweeps
are scored in §2.5 and §4.2 respectively; they predate `Λ`/`B` scoring and have not been re-replayed.*

**Read three things off it:**

1. **Peak accuracy is monotone in `Λ`**, saturating at ≈0.865 by `Λ ≈ 0.95`, with no exception outside
   the ±0.005 replicate band.
2. **Whether an arm holds its peak is decided by `Φ`, and by nothing else** (§3.4).
3. **Efficiency `Λ/B = 2cos/ρ` spans 20× across the portfolio.** `mean` at `N`=200, `ρ`=0.03 banks
   `Λ` = 0.63 for `B` = 0.14 (11% of budget); `mean` under a free gate at `ρ`=0.086 spends **9× the
   budget for +0.016 accuracy**. **The best arm in the portfolio is the least efficient one** — only
   visible in these coordinates.

**Replicate spread** (`200242` vs `200325` are byte-identical configs; `223446` was repeated as
`065837`):

| | `ρ` c1 | `ρ` c40-80 | `‖θ_tr‖` end | peak acc | final acc |
|---|---|---|---|---|---|
| `200242` vs `200325` | 1.3% | 5.0% | 5.7% | **±0.0009** | **±0.045** |
| `223446` vs `065837` | 0.0% | 0.0% | 0.5% | **±0.0007** | ±0.012 |

**Score A/Bs on peak accuracy and the stability columns, never on final accuracy of a diverging arm** —
post-turnover trajectories are chaotic and the spread is 50× wider there than at the peak.

## §4.2 Knob ledger — what each lever does, and what it measured

One row per concept. `ρ/cos` and `ρ·cos` columns are the §3.6 predictions; "measured" is what happened.

| knob | predicted effect | **measured** | verdict |
|---|---|---|---|
| **`η` server LR** | `ρ ∝ η`, pays 1:1 in progress | `ρ` 0.2004 → 0.0404 → 0.0101 over `η` .01/.002/.0005, **to 1%**. `η`=0.002 reaches 0.601 @ c120 / 0.815 @ c327 vs `K`≥20's 0.860 | **works as physics, FAILED as a fix** — full 1:1 cost, and `Σρ²` still diverges (§4.4) |
| **`K` cohort width** | `ρ/cos ∝ 1/K`, progress invariant | `ρ·√N` invariant to 4% over `K` 10→50; the *naive* `ρ ∝ 1/√K` **falsified** (gate returns the gain as fewer `I`). `K`≥20 holds **0.860** where `K`=10 collapses | **works**, but `K` is not a knob on `N` — the gate absorbs it. Set `K` and the gate together |
| **`I` iterations/bin** | `ρ/cos ∝ 1/I`, but serial | at `K`=50 the gate cut `I` 18.5 → 5.9 — 3× fewer round trips at slightly larger `N`, best peak of the 08-07 portfolio (0.861), held | **works — buy `N` with `K`, not `I`** |
| **`P` probes, under selection** | stability-**neutral** (`b²/a = 1`); more `E` ⇒ *faster divergence* | `E` 2.988 (P=10) → 4.744 (P=30) over 37k events; predicted `ρ` ratio `√(4.744/2.988)`=1.260, **measured 1.236 (2%)**. `P`=30 learns faster per commit (0.79 @ c40 vs 0.58–0.66) and **collapses sooner** (doubling 57 vs 68) | **prediction confirmed, including its harmful direction.** Never sweep `P` under selection again (§4.4) |
| **combination rule** (`select` → `mean`) | `ρ/cos ∝ 1/P`; `ρ` down `√(E·P)` = 5.466× | at matched `N`=200: `ρ` **5.386×**, `‖Δθ‖` **5.430×** (1.5%). `‖θ_tr‖` end 60.7 → 14.5; `‖θ‖²` growth rate **106×** lower (prediction was 30×; the effect compounds — 5.4× at c1-10, 15.0× by c140) | **WORKS — 10× in `ρ/cos` for zero extra compute, bytes, or `η`/`N`/`p` change** |
| **top-k averaging** | `a = E_k`, `b = √(E_k/k)` ⇒ `b²/a = 1/k`; monotone, optimum `k = P` | offline over 34,447 events: random 1.00× / coin-top-2 1.73× / top-1 1.95× / top-3 2.72× / **all-10 3.16×** in `cos` gain, at `ρ/cos` 1× / 1× / 1× / 3× / **10×**. All cost the same 20 passes | **settled offline — use all `P`.** The shipped rule is k=1 with E=2.991, i.e. **zero** stability gain |
| **probe distribution** | is there a "good probe" to find? | `d_i` indistinguishable from iid `N(0,‖g‖²)`: top-1 of 10 → 3.811 observed / 3.798 synthetic; coin top-2 → 2.991 / 2.987 | **NO — closes off all cleverer `|d|`-based selection**, and double-duty validates FD linearity at the large chord |
| **`p` (adapter `rf`)** | `cos ∝ 1/√p`; `ρ/cos ∝ √p` (`ρ` is `p`-invariant) | `rf` 16/32/64: peak **0.852 / 0.857 / 0.859**, `‖θ_tr‖` end 78.1 / 31.6 / **20.8**, `Φ` 5.74 / 3.28 / **3.04**. `ρ` ratio measured **1.00 / 0.967 / 0.928** (the `0.71/0.51` prediction is **falsified** — hence `√p`, §3.6) | **WORKS — best cost/benefit in the stack.** A quarter of `p` learns agnews at least as well. Must run with `FWDLLM_FD_SCALE_INVARIANT=1` |
| **step rule** (`raw_sgd` → `trust_ratio`) | makes `ρ` an operator constant; removes `|JVP|` scale and the α multiplier | `ρ = ρ*` to **8.7e-5** (`const`); `(1+ρ*²)^{T/2}` predicts `‖θ_tr‖` 13.4696 vs 13.4714 observed (**0.013%**); **α moves `ρ` by 0 to 6 s.f.** (§2.5d) | **WORKS.** But **alone it does not bound `‖θ‖`** — constant `ρ*` is still geometric. Ships with the anneal |
| **anneal** (`rho_schedule=rm`, `t^-exp`) | Robbins–Monro: `Σρ=∞`, `Σρ²<∞` | enacts to **4.5e-4** over 318 commits. `exp`=0.55 takes `ρ` **17× below setpoint** by c186 → arm flat at 0.34. **`exp`=0.25 is measured adequate** despite being formally outside the RM window — log-log 0.01/0.05/0.12 over three arms, 313–318 commits | **WORKS; `exp`=0.25.** The RM bound is conservative at `T`≈300. Size the exponent to the **horizon** |
| **`ρ*` setpoint** | peak monotone in `ρ*` | 0.03/0.06/0.09 → peak **0.695 / 0.801 / 0.846**, all still climbing at cutoff, all stable (`Φ` ≤ 1.145). **The anneal spends most of the setpoint**: realised mid-run `ρ` is 0.011/0.022/0.032, an order of magnitude below the 0.086 of the best free-gate arm | **band not closed.** Read `ρ*` off §4.1 by `Λ`, not by `ρ` — and note §3.4(a): the old "walk `ρ*` up" conclusion is withdrawn (§4.4) |
| **commit gate** (`var` → `n_target`) | scale-free `N` target replaces a unit-carrying threshold | `N_req` closed form **exact** (28.1 at ρ*=.01; **1,013.3** at ρ*=.06 = `450,340·(0.06/0.4)²/10`). Both `setpoint` arms sat at the `I` cap for all 312 commits ⇒ byte-equivalent to a hard pin | **INCONCLUSIVE twice.** `s` and `K` are **coupled**: at ρ*=.06 the criterion demands 5× what `K`=10 can pool. That is a **cohort-width requirement**, not a gate parameter. Re-run at `K`=30 |
| **gate `ρ` reference** | — | **composition bug**: `N_req ∝ ρ_t²`, so annealing `ρ` makes the gate demand *less* pooling and progress `ρ·cos` decays as `ρ²`. Observed: `N_req` 28.1 → **0.0** by c20, `I` floored at 1 on 1,271/1,273 commits, peak 0.394 *decaying* to 0.274 — **reproduced identically at α=0.1** over 1,263 commits | **fixed by `gate_rho_ref=setpoint`** (sizes `N_req` from `ρ*₀`). No-op under `const` and `raw_sgd` |
| **`annealed` vs `setpoint` gate** | — | at **matched commits** `setpoint` wins every column (c312: acc 0.804 vs 0.759, `I` 20 vs 6); over the same **vclock** `annealed` gets 2.3× more commits and ends higher (0.821 vs 0.804) | **not right vs wrong** — `annealed` is progress-per-wall-clock, `setpoint` is progress-per-commit. Belongs to the controller, not the gate |
| **α heterogeneity** | enters via `‖g‖` only | §2.5 — 1000× in α, `var` floor 1.33/0.64/0.28, new invariant 0.53 ± 0.01, both `K` predictions falsified | **understood and neutralised** by the step rule |
| **`n_eff` sensor** | dimensionless pooling-adequacy sensor | `n_eff = 2·mean(d²)/var`. Synthetic: recovers true `n` (198.7/200), detects redundancy (×2 → 0.49), scale-invariant over 100× in `‖g‖`, **FAILS on directional disagreement** (4/20/100 distinct directions all → 1.00). In 17 real arms: **`n_eff/N` = 1.00 ± 0.01** | **an identity, not a measurement.** In a gate it reduces to a counter. Its one real result is **negative and valuable**: pool *correlation* is excluded from the §3.9 shortfall |

## §4.3 Feature ledger — what is built

**The single source of truth for implementation status (R3).** Every feature is flag-gated, default =
old / byte-identical off. Terminal state (`PERMANENT` / `FLAGGED` / `REVERTED`) is the **operator's**
call — ask, with the A/B evidence. Nothing reaches `WORKS` without predicted-vs-observed numbers and a
run id.

| # | feature | flag (default = old) | status |
|---|---|---|---|
| **B1** | `cos(G,g)` ground-truth probe | `cos_ground_truth_audit` + `cos_probe_batch_size` | **RUN · HALF-ANSWERED** (§3.9). Emits `cos_ground_truth`/`pooled_norm`/`probe_grad_norm` per commit; index alignment unit-tested. Confirms `b` to 5–10% and resolves H-B. `cos` itself is 25–80× under the closed form and anti-orders the arms. **Blocked on B13** |
| **B2** | average all `P` probes | `probe_combine: {select\|mean}` | **WORKS · 5/5 arms** (§4.2). Site: `_accumulate_mean_over_probes`. Also fixes the `P=1` crash (`sorted_indices[-2]` on a 1-element list) and the RNG-stream mismatch that blocked the C1 ablation |
| **B3** | trust-ratio step + `t^-exp` anneal | `server_step_rule` + `rho_star`/`rho_schedule`/`rho_exp` | **WORKS mechanically and portably; setpoint open** (§4.2). Site: `FedSgdAggregator._apply_weighted_update` (pool-then-apply). Subsumes S3: ω can no longer influence step magnitude |
| **B4** | scale-free `n_target` commit gate | `commit_gate: {var\|n_target}` + `gate_safety_s` + `gate_rho_ref` | **INCONCLUSIVE, twice** (§4.2). Closed form exact; composition bug found and fixed; both A/Bs cap-bound. **Re-run at `K` = 30** |
| **B5** | shrink `p` via adapter bottleneck | `adapter_reduction_factor` (16) + `FWDLLM_FD_SCALE_INVARIANT` | **WORKS · 3/3 · H-G YES** (§4.2). Read by `expts/initializer.create_model` on **both** sides; `trainable_scope` stays, inert, so old configs parse |
| **B6** | long `K`=50 arm, ≥32 h vclock | none (runtime only) | **DEMOTED** — `B` now extrapolates the answer exactly (§3.4). Confirmation, not an open question |
| **B7/B15** | weight decay `λ ≈ ρ²/2` | `server_weight_decay` | **TODO · CHEAPEST DECISIVE ARM** — this is §6.1 Q2. 3 lines |
| **B8** | staleness histogram from the K-sweep | none (log replay) | **TODO** — free, data on disk (§6.3 H-E) |
| **B9** | split-half commit gate | `commit_gate: {var\|cos}` | **DEAD** at `p`=450k (§4.4) |
| **B10** | adaptive `P` per client | `probe_budget: adaptive` | **PARKED** — same wall as B9, worse (§4.4) |
| **B11** | `n_eff` sensor | rides on `server_update_audit` (emit-only) | **FAILED as a sensor, KEPT as an audit** (§4.2). Do not wire it to a controller |
| **B12** | α-sweep | `--partition-method ...alpha={0.1,1,100}` | **WORKS · 4/4** (§2.5) |
| **B13** | two-batch `cos` control | `cos_probe_batch_size` + second cached batch | **TODO (H-M)** — ~5 lines, rides on any arm. Now a *diagnosis of the probe*, since §3.4(c) validated the formula functionally |
| **B14** | `rf`=64 to `Φ` = 6.6 | `adapter_reduction_factor=64` + runtime | **TODO · TOP BLOCKER** — this is §6.1 Q1 |
| **B16** | `Λ` out of sample: rule × `p` at pinned `ρ*`,`N` | `probe_combine` × `adapter_reduction_factor` | **TODO** — this is §6.1 Q3, pre-registered |

## §4.4 Dead ends — do not retry

**Append-only (R5). Read before proposing anything.**

| do not | why it is dead |
|---|---|
| **Lower `η` alone** | Pays 1:1 in progress and does not restore square-summability — a constant `ρ` of any size has `Σρ² = ∞` |
| **Retune `var_threshold`** | Not because it is inert (it controls `N` whenever reachable) but because its units make the setpoint a per-model constant *and* it simultaneously absorbs a heterogeneity floor that moves 1.33 → 0.28 across α. Any retune fits one corner of an (α,`K`) grid |
| **Sweep `P` under the shipped selection rule** | Measures `E` and nothing else; and raising `P` under selection is *harmful* (§4.2) |
| **Keep selecting probes by `\|d\|`** | Stability-neutral by construction (`b²/a = 1`, measured); candidates provably carry no other structure in `\|d\|`. Top-k average with k < P is strictly worse |
| **Split-half cosine as a gate or a `cos` estimator (S-E / S-J)** | **Measurability wall, measured.** Per-commit sd 1.5e-3 against a 1e-4 signal (SNR 0.07); at matched `N`=200 over 491 commits it returns **−3e-5 ± 1.1e-4**. Even `mean` + `rf`=64 at the measured `cos` gives SNR ≈ 2. The wall is specific to the *cosine*, not the split-half pair: normalising the same statistic gives `1 − cos(G_A,G_B) ≈ 1 − 1e-4` against the same 1.5e-3 floor. Revisit only if `p` falls by orders of magnitude |
| **Wire `n_eff` to a controller** | An identity (`n_eff/N` = 1.00 ± 0.01 in 17 arms), blind to directional disagreement (§4.2) |
| **Retry momentum before `ρ` is bounded** | `ρ_eff = ρ/(1−β)`; at β=0.9 that is `ρ_eff` = 1.6 and the norm doubles *every step* — the historical NaN is arithmetic, not a bad idea. **Now unblocked** by the trust-ratio step: at `ρ*`=0.06, β ≤ 0.5 is the testable range |
| **Build the least-squares / min-norm gradient solve** over `{(v_i,d_i)}` | At `P,N ≪ p` it equals the average up to scale — no gain in `cos` |
| **Orthogonalise the `P` probes, or coordinate probes across trainers** | No-ops: random probes are already orthogonal to `1/√p`, and `K·P` = 500 ≪ `p` |
| **Normalise `v` expecting a variance win** | `‖v‖` concentrates to 0.07%. *Do* rescale `h` when `p` changes — hygiene (§3.2), not a fix |
| **Chase momentum in the probe distribution** | Computed negative: the accumulated trajectory has `cos ≈ 0.1–0.23` after 100 commits, so as a control variate it removes ~5% of the variance |
| **Invest further in ω-direction / inverse-variance weighting** | Two orders of magnitude below the problem, and the trust-ratio step removes ω from magnitude entirely. ω-*freshness* is different and still open (§6.3 H-E) |
| **Shrink `h`** | Pinned between truncation error and fp16 catastrophic cancellation (§2.1) |
| **Size `N` from §3.8's table** | 25–80× optimistic in `cos`, hence ~10³× in `N_req` (§3.9) |
| **Score stability by the `‖θ‖²` log-log slope** | Bounded above by 1 under trust-ratio **by construction** (§3.4b) — every "sub-linear ⇒ safe" reading on such an arm is vacuous. Score `B` and `Λ` |
| **Walk `ρ*` up to 0.12 / 0.15 / 0.20** | This document's own recommendation one revision ago, **withdrawn**: budget cost for fixed `Λ` is ∝ `ρ`, so raising `ρ*` buys wall clock and spends safety. Pick `ρ = Λ_req/(T·cos)` from the commit budget |
| **`trainable_scope: adapters_only` as a `p` lever** | No-op — `pre_classifier` is already dropped at `:217` (§2.1). The remaining `p` lever is adapter width |
| **H1 shuffle / H3 bin-order permutation** | Do not address the mechanism. Parked |
| **H2 bin size / M1 sweep** | Settled: spend compute on probes, not bigger bins — probe noise dominates data noise by ~40× (**ANALYSIS**) |

**Not costed, not dead — the only two ideas that beat the `√(n/p)` barrier.** **Block-coordinate
probing** (one adapter layer at a time, `p → p/L` per probe) improves `ρ/cos` by ~`L` while each commit
updates `1/L` of the params; known in ZO optimisation, needs analysis first. **Low-rank / subspace
probing** needs a good subspace *and* a way to broadcast it.

## §4.5 Superseded numbers — quote check

Only numbers that were quoted in other documents or drafts before being corrected (R8). If you see one
of these in `fluxtune_contributions.md`, `FLUXTUNE_CODE_QA.md`, or a paper draft, it is wrong.

| superseded | replaced by |
|---|---|
| `p = 1,040,932`, `‖θ_tr‖` init 20.356 | **`p` = 450,340, `‖θ_tr‖` = 13.35** — the trainer drops `pre_classifier` before probing (§2.1) |
| "freezing `pre_classifier` buys 2.31×" | **already banked** — the layer is not in `p` |
| `ρ` = 0.115 flat all run; orthogonality ratio 1.032 | **`ρ` = 0.16 at commit 1**, falling iff `N` grows; ratio **1.000 ± 0.005**. Both earlier numbers were reconstruction artifacts of using total `‖W‖` |
| "the variance gate is 100% dead" / "live at `K` ≥ 20" | **live wherever `2b²‖g‖²/n` can reach 0.3**: live at `K`=10/α=100, cap-bound at `K`=20/α=0.1 (§2.5c) |
| `ρ·√N` = 1.68–1.81 is the pooling invariant | true **only at fixed α**; the invariant is `ρ√N‖θ_tr‖/(η·rms\|d\|·√p)` = 0.53 ± 0.01 (§2.5b) |
| `n_eff` is the scale-free replacement for `var_threshold` | **an identity** — the step rule carries the portability claim (§2.5d, §4.2) |
| `cos` = 0.0351 predicted; `cos ≤ 0.015` from split-half | **measured 0.0004–0.003 against a held-out reference** (§3.9); both retired |
| `ρ/cos ≥ 10×` over budget | `ρ/cos` **against the measured `cos` does not order the arms at all**; against the closed form it is the efficiency `2cos/ρ` and it does (§3.4) |
| `ρ/cos ∝ p` | **∝ √p** — `ρ` is `p`-invariant (§3.6) |
| `s = 0.3–0.5` | **`s` = 2.6–4.3**, derived from the two conserved laws (§3.5) |
| "there is a critical `ρ ≈ 0.09`" | **debunked by arithmetic** (§3.4a) — a horizon artifact. Every `ρ > 0` inflates geometrically |
| `exp = 0.55` (strictly inside Robbins–Monro) | **0.25** — sized to the horizon; 0.55 spends the whole budget in the dead zone (§4.2) |
| "4 h minimum or the result is uninformative" | score `B` and `Λ` — exact at any horizon, readable in ~20 commits |
| collapse = logit saturation | **directional degeneracy**; `logit_norm` does not discriminate (§2.6) |
| "never converges — it *oscillates*" (`fluxtune_contributions.md` §8) | at 4 h it is a **monotone rise then monotone divergence** |
| QA §D2 "the k sweep cannot run today" | answered offline from logged JVPs — monotone, optimum `k = P` (§4.2) |
| QA §E1 "measuring `cos(G,g)` needs `v_k` uploaded" | not for this quantity — a server-side backprop gradient on a probe batch suffices, and it is built (B1) |

**Telemetry gotcha.** Before 2026-08-07, `tc_transformer_trainer_distribute.py:485` logged the argmax
under the label `chosen jvp` while the actual pick is the coin-flip result. It now logs the coin-flip
winner as `chosen jvp`, the argmax as `max jvp`, and the index as `chosen idx`. **Runs before that date
carry the old, mislabelled field** — any parser must handle both.

## §4.6 Process lessons — how to run the work

*Only entries about **method**. Results live in §4.1–§4.3.*

**Worked:**

- **Sweeps over single points.** Every surviving law was confirmed by a *slope*; the one claim that died
  (`ρ ∝ 1/√K`) died because a curve exposed feedback a single point would have hidden. A slope cannot be
  rescued by a fudge factor.
- **Pinning `N`.** Every fix moves a quantity the var gate immediately re-spends (§4.7), so an unpinned
  A/B measures the gate, not the fix.
- **Varying the axis every earlier sweep held fixed.** The α-sweep cost four arms and no code, and moved
  three things — two of them falsifications of our own predictions.
- **Measuring norms rather than cosines.** `‖G‖/‖g‖` and `‖g‖`-vs-`‖θ_tr‖` are `O(1)` ratios and gave
  clean answers from the same probe whose headline cosine is still contested. **At `p` = 4.5e5, build
  the instrument that reads an `O(1)` quantity.**
- **Pre-registering a sinking condition, not just a point prediction.** The ρ* node missed both point
  predictions and the read was still unambiguous, because "arm 3 diverges AND arm 1 under 0.6" was
  written down first.
- **Scoring `‖θ_tr‖²` and `‖Δθ‖` instead of accuracy.** Turned a 4-hour verdict into a 20-commit one
  (now superseded by `B`/`Λ`, which are better still).

**Did not work:**

- **Scale-free-by-measurement as a strategy.** The plan was to make each unit-carrying constant portable
  by measuring the units away; for the gate that produced an identity (`n_eff`), and for `cos` a number
  that fails its own consistency checks. **At this `p` the informative content of pool statistics is
  `O(1/√p)` below their noise, so the portable quantity has to come from the *step rule*, where `ρ` is
  exact.**
- **Deriving a setpoint from theory, twice.** `s` = 0.3–0.5 was never calibrated and was silently
  absorbing a shortfall of unknown size *and direction*. **A third re-derivation would repeat the
  error** — the operating point comes from §4.1 until H-M closes.
- **Scoring features without scoring compositions.** The gate and the anneal are each correct and
  multiply into a stall (`N_req ∝ ρ_t²`, §4.2). Nothing in either feature's own prediction could have
  caught it.
- **Launching a portability test without a napkin check on its own operating point.** `N_req` = 1,013
  against a cohort ceiling of 200 was computable in one line, and the risk was named before launch.
- **Final accuracy as an A/B statistic.** ±0.045 between byte-identical replicates vs ±0.0009 at the
  peak (§4.1).
- **A missing attribute costing a full night on three nodes** (08-08): twelve of sixteen arms died ~30 s
  in with `AttributeError: 'ClassificationArgs' object has no attribute
  'select_perturbation_using_jvp'` while the chain marched on. **08-09 was the same class, quieter:** a
  feature that emits a *wrong number* rather than crashing. Both are now preflighted (§6.5).

## §4.7 The interaction that shapes every A/B

`var ∝ b²‖g‖²/n`, so averaging's 30× cut in `b²` puts `var` under `var_threshold` = 0.3 at iteration 1:
the gate commits at `I ≈ 1–2` instead of 18.5 and `N` collapses from 185 to ~10–20. Since
`cos = √(G_rule·N/p)`, the 10× gain in `G_rule` and the ~10× loss in `N` **cancel** — what averaging
buys under the shipped gate is **wall clock (~6.5× fewer serial round trips), not aim.**

1. **"`ρ` drops 5.45×" holds only at matched `N`.** Score **`ρ·√N`**, which is `N`-free.
2. **`mean` and the var gate cannot both be free.** Either pin the pool (`--var-threshold 0
   --max-iter-per-data-id 20 --var-stopping-policy off` ⇒ `I` = 20, `N` = 200 exactly) or the arms are
   unmatched by ~10× in `N`.

**Scoring gotcha:** `_pool_split_half_stats` returns `None` for a pool of one, so at `I = 1` the
`server_update` record carries **no `pool_size` and no split-half components**. Reconstruct
`N = K·(iteration_per_data_id + 1)`, which is always present.

---

# Part 5 — The FluxTune approach as it currently stands

*Grounded in §3 and §4. Every element here is either measured to work or explicitly marked open.*

## §5.1 The algorithm

**(a) Perturbation-based training on one device.**

```
given: theta, trainable slice theta_tr of dimension p, step schedule rho*_t, budget P
loop t:
  draw P Gaussian probes v_1..v_P            # scale h by sqrt(p) so h||v|| is p-independent
  for each i: d_i <- ( L(theta + h v_i) - L(theta - h v_i) ) / (2h)     # 2 forward passes, no_grad
  u <- (1/P) * sum_i d_i * v_i               # ASSIMILATE ALL -- never select on |d| (4.2, 4.4)
  theta_tr <- theta_tr - rho*_t * ||theta_tr|| * u / ||u||              # TRUST-RATIO step (3.5)
  rho*_{t+1} <- rho*_0 * t^-exp              # exp = 0.25, sized to the HORIZON (4.2)
```

**(b) The same loop in FL.**

```
SIZING (offline, before the run):
  choose rho*_0 from the DOSE-RESPONSE CURVE (4.1), not from p*(rho/s)^2/G_rule   <- 3.8 is unusable
  pick I as small as the gate allows, then K = n_req/(P*I);  require C >= K

PER COMMIT (server):
  dispatch to C clients, wait for K uploads   # async: K arrives, stragglers roll into the next
  G <- sum_k omega_k u_k / sum_k omega_k      # omega re-weights; it must NOT set magnitude
  commit when the pool reaches N_target       # NOT a cos_hat test -- unmeasurable today (4.4)
  theta_tr <- theta_tr - rho*_t * ||theta_tr|| * G / ||G||
  log rho = ||dTheta||/||theta_tr||, ||theta_tr||, top_class_share   # the three monitors (2.7)
  rho*_{t+1} <- rho*_0 * t^-exp
```

## §5.2 Priority order for any new configuration

Every decision reduces to *how to reach the required `n` most cheaply*:

1. **`p` first** — the only lever that improves aim at *negative* cost (§4.2).
2. **then `P`** — free in wall clock, and only pays off if all `P` are assimilated.
3. **then `K`** — the only stage that touches data noise.
4. **`I` last**, and only what the gate demands — it is the one that costs serial round trips.

Then two rules: **set `ρ*` from the dose-response curve, not from `η`, and anneal it** — a pool large
enough to be safe now is not safe 1,000 commits later; and **never let probe selection set step
magnitude.**

## §5.3 The recommended stack, and what each element rests on

| element | setting | rests on | confidence |
|---|---|---|---|
| combination rule | `probe_combine=mean` | §4.2, 5/5 arms, prediction hit to 1.5% | **settled** |
| step rule | `server_step_rule=trust_ratio` | §4.2, enacts to 8.7e-5; α-portable to 6 s.f. | **settled** |
| anneal | `rho_schedule=rm`, `rho_exp=0.25` | §4.2, three arms, 313–318 commits | **settled** |
| `p` | `adapter_reduction_factor=64` **+ `FWDLLM_FD_SCALE_INVARIANT=1`** | §4.2, 3/3, 0.859 at a quarter of `p` | **settled**; headroom question is §6.1 Q1 |
| `ρ*₀` | read off §4.1 by target `Λ`, then `ρ = Λ_req/(T·cos)` | §3.5 design rule | **empirical** — the formula route is blocked on §3.9 |
| commit gate | `commit_gate=n_target`, `gate_rho_ref=setpoint`, `s`=0.4 | closed form exact; A/B inconclusive twice | **open** — needs `K`=30 (§6.4) |
| `K`, `C` | `K` ≥ 30, `C` ≥ `K` | §4.2 — `N_req` at ρ*=0.06 is 1,013, five times what `K`=10 can pool | **open**, and it is a *requirement*, not a preference |
| monitors | `‖θ_tr‖`, `ρ`, `top_class_share`; score `B` and `Λ` | §2.7, §3.4 | **settled** |

## §5.4 What is finalized, and what is not

**Settled — do not re-measure.** The step rule, the anneal, the combination rule and the `p` lever all
enact to ≤4e-3 with mechanisms measured (§4.2). On top of them, §3.4 closes the *dynamics*: `‖θ_T‖` is a
closed-form function of the `ρ` trajectory alone, and peak accuracy is a monotone function of `Λ`.

**The remaining unknowns are not about the optimizer.** They are about the two constants those laws are
scored against — `B_max` and `Λ_req` — and about whether the closed-form `cos` inside `Λ` is real. That
is exactly §6.1.

**Two structural gaps with no owner yet:** no convergence-detection rule exists, even in §5.1; and the
`h`/`p` coupling is unanalysed beyond the `FWDLLM_FD_SCALE_INVARIANT` interaction.

## §5.5 Claim status

Two naming schemes are in play: `C1–C3` (this doc, `FLUXTUNE_CODE_QA.md`) and `S1–S3`
(`fluxtune_contributions.md` §8.2). Not yet landed in that file.

| claim | as written today | after the portfolios | what would settle it |
|---|---|---|---|
| **C1 · informed JVP-magnitude probe selection** | "keeps the steepest — a better gradient estimate per round" | **Refuted as a contribution.** At matched `N`=200, `mean` beats `select` by 5.386× in `ρ` and 106× in `‖θ‖²` growth rate, at equal or better accuracy. `b²/a = 1` is measured, so the `2P` budget is justified by **averaging alone**. **Do not abandon C1 — redirect it: the *combination* rule is the contribution, not the probe-selection rule.** "Compute `P` directional derivatives and assimilate all of them" is the **trainer-side analogue of server-side pooling**. Rewrite C1 as a compute-budget claim | **DONE** |
| **C2 · async aggregation / K-C control** | "async, straggler-tolerant; `agg_goal < K` commits as stragglers arrive" | **Reframed; the controller is still not demonstrated.** `ρ ∝ 1/√(K·I)` is measured and only the async path has a free `N` to spend, but the gate A/B has failed to produce a contrast twice. **The honest current form is a cohort-width *requirement*, which is a stronger claim than a gate parameter** | gate re-run at `K`=30 |
| **C3 · aggregation weighting (ω)** | grad-aware rate, align gate, staleness weighting | **Direction: park** — ω spans 0.70–0.87 against a ≥10× gap, and the trust-ratio step removes it from magnitude. **Freshness: open and testable** | B8 (free replay), then the K-sweep |
| **S1 · server momentum** | "REFUTED as-designed" | **Re-framed, not refuted.** `ρ_eff = ρ/(1−β)` = 1.6 explains the NaN as arithmetic; `ρ` is now bounded to 4.5e-4, so the retry is unblocked at β ≤ 0.5 | after §6.1 fixes `ρ*` |
| **S2 · variance-gate recalibration** | "commit on the plateau, not a noise dip" | **Superseded.** The gate is a working `N`-controller whose setpoint carries units. Replace the loop, do not re-tune the threshold | B4 |
| **S3 · aggregation-rate tempering** | "cap rate ≤ 1; damp, not amplify" | **Subsumed and measured.** Under trust-ratio, `ρ` = `ρ*_t` to 8.7e-5 regardless of ω | **DONE** |
| **Systems: flat memory, inference-only operator set** | structural claims about the absence of an autograd graph | **Untouched and the strongest part of the paper.** *B1 uses a server-side backward pass — it is an audit flag, off by default, and does not touch the operator-set claim* | — |
| **Cost framing** | "10× sync compute at P=10, collapsing to parity at P=1" | **Needs rewriting.** Under selection, raising `P` is *harmful*; under averaging the same `2P` buys `ρ/cos ∝ P` | **DONE** |

**Net effect on the contributions:** all three ML claims move. **C1 shrinks** to a compute-budget
justification that only averaging redeems; **C2 grows** but relocates from "async" to "the controller
that async makes possible", now blocked on a cohort-width measurement rather than a gate design; **C3
splits** into a parked half and an untested half. The systems contributions are unaffected.

**What the portfolios added that was not on the list:**

- **`p` is a gradient-quality parameter, not just a memory knob** (§4.2) — measured, 0.859 peak at a
  quarter of `p`. The strongest ML result here, and the only one about the *model* rather than the
  optimizer, so it does not compete with FwdLLM's execution primitives. **Contribution-grade general
  statement:** for backprop FL, PEFT rank is a memory/communication knob and gradient quality is
  unaffected; for forward-gradient FL, **`cos ∝ 1/√p` — PEFT rank is the primary determinant of
  gradient quality.**
- **Scale-invariance as a design principle** (§3.7) with its cleanest demonstration: under the
  trust-ratio step, heterogeneity moves `ρ` by **zero to six significant figures** while still moving
  `var` by 1.6× (§2.5d).
- **The `ρ ≤ cos` criterion is *weaker* after B1, not stronger.** `ρ` is exact and every fix built on it
  enacts to 1e-4, but the quantity it must be compared against is 25–80× below its closed form, drifts
  11× where theory says it is constant, and anti-orders the arms. **Write it up as a design rule with a
  measured dose-response curve, and do not claim the control law until H-M closes.**

---

# Part 6 — What runs next

*In-place section. When an item resolves, move its finding into §3/§4 and delete the row here.*

## §6.1 The three questions that block finalizing

| # | question | why it blocks | decisive experiment |
|---|---|---|---|
| **Q1** | Is the collapse threshold **absolute** (`‖θ_tr‖ ≈ 45`) or **relative** (`Φ ≈ 3.6`)? | It *is* `B_max`, so it sets every sizing number in §3.4, and it decides what the `p` lever actually buys — headroom, or only `cos` | **`rf`=64 (`p`=118k, `‖θ_0‖`=6.86) run to `Φ` = 6.6.** Relative predicts degradation at `‖θ_tr‖` ≈ 24.7, absolute predicts it holds to 45. `B` = 1.89 at `ρ` ≈ 0.107 ⇒ **~330 commits**. The `p` ladder is the *only* lever that separates them, and §3.7 predicts the relative answer |
| **Q2** | Is `‖θ_tr‖` **causal**, or a symptom of accumulated misaim? | If causal, weight decay removes `B_max` and **uncaps `Λ`** — the largest available result in this program. If symptom, the budget is real and permanent and every fix must live inside it | **Weight decay, `λ = ρ²/2`** (3 lines), at `ρ*` = 0.09 with a no-decay control. Holds `Φ` = 1.00 exactly by construction. **Prediction: it does not help** — §2.6 makes the failure *directional*, and direction is scale-invariant, so rescaling `‖θ‖` cannot undo it. **A null is as valuable as a positive**, and it is the cheapest arm in the plan |
| **Q3** | Does `Λ` predict **out of sample in the steep region**? | `Λ` is the deliverable and it is currently *fitted*, not *predicted*; its `G_rule` and `p` scaling only appears where accuracy has saturated (§3.4c) | **Three arms, pinned `ρ*`=0.06 / `exp`=0.25 / `N`=200, varying only the rule and `p`** — pre-registered below |

**Q3's pre-registered predictions** (written before launch; `Λ` from `Σρ_t·√(G_rule·N/p)`, peak from
§4.1's calibration):

| arm | `G_rule` | `p` | `Λ` | **predicted peak** | status |
|---|---|---|---|---|---|
| `mean`, `rf`=16 | 10 | 450,340 | 0.398 | 0.801 | **anchor — already measured at 0.801** (`035045`) |
| **`select`, `rf`=16** | 2.988 | 450,340 | **0.218** | **≈ 0.70** | a −0.10 swing from the *rule alone*, at pinned `ρ` and `N` |
| **`mean`, `rf`=64** | 10 | 118,348 | **0.776** | **≈ 0.857** | a +0.056 swing from `p` alone, at pinned `ρ` and `N` |

**Sinking conditions.** Q1: none — either answer is a result. Q2: none — a null is the informative
branch. **Q3: if `select` lands near 0.80 rather than 0.70, `G_rule` does not enter progress**, `Λ` is
`Σρ_t·√(N/p)` at best, the `2P` probe budget buys nothing measurable, and averaging loses its
quantitative case.

## §6.2 The instrument ladder — test before you run

**R7 in practice.** Each rung is ~3 orders of magnitude cheaper than the next. *Every* new hypothesis
starts at the highest rung that can falsify it; a sim run is what **confirms** a hypothesis that already
survived a cheaper instrument.

| rung | what it is | cost | what it can answer |
|---|---|---|---|
| **1 · log replay** | Python over `aggregator_*.jsonl` and trainer logs already on disk. No GPU | minutes | anything expressible in `ρ`, `‖θ_tr‖`, `N`, `var`, `B`, `Λ`, the JVP distribution, staleness, commit reasons. **The k-sweep, `E[v∥²]`, `n_eff`, the norm law and the progress law were all settled here** |
| **2 · offline rig** | one GPU, real model + real JVP math, **no FL stack**. Import `build_model` (`scripts/profile_jvp_opt.py:64`), `calculate_jvp` / `functional_get_loss` (`fwdgrad_utils.py:105,66`), `stage1_vmap_fd` (batches all `P` probes, verified bit-identical to the production loop), the real agnews H5 partitions | minutes–1 h | anything needing a backprop ground truth or a scaled-`‖θ‖` sweep: `cos(G,g)` vs `N`, `‖G‖/‖g‖`, H-B, the collapse endpoint, curvature `vᵀHv`. **B1's design came from here** |
| **3 · single-process replica** | the whole optimization loop in one process — draw probes, compute JVPs, pool `K` uploads, run the real gate, commit with the real server arithmetic (`_server_update_step` / `_apply_weighted_update`, ~15 lines). No MQTT, no selector, no sim clock | hours per trajectory | **interactions**: the feedback loop, the gate drifting out from under itself, A/B ranking of fixes, and the `K`/`P` ablations that are structurally impossible in production because `ρ` and `cos` move at once. **Validation gate is non-negotiable:** reproduce the shipped arm's `ρ`, orthogonality ratio, doubling time and rise-peak-collapse first, or nothing downstream counts |
| **4 · sim run** | the real stack with the virtual clock. `sim_rate` 3.33 (fluxtune) / 12.6 (fwdllm) virtual-s per wall-s ⇒ 4 h vclock ≈ 1.2 h wall | ~1–3 h wall per arm | end-to-end confirmation, genuine staleness, anything needing real concurrency or real timing |
| **5 · real run** | the 8-GPU emulation harness | ~4 h wall per arm | confirming a *winner* only. Discover in sim, confirm on real |

**Two standing rules for rungs 2–3.** (1) **Probes import the production code** — never reimplement the
math. The discipline is `scripts/profile_jvp_opt.py`, which reuses `expts.initializer.create_model` and
`fwdgrad_utils.calculate_jvp`, so a validated result transfers into the trainer as a **config flag, not
a rewrite**. (2) **Nothing on rungs 1–3 modifies `trainer/`, `aggregator/`, or any yaml on the critical
path**, and no probe result lands as a production change until two independent instruments agree on the
number it produces.

**Reuse handles that already exist** — this is why rungs 2 and 3 are affordable at all:

| handle | where | what it buys |
|---|---|---|
| `create_model` | `expts/initializer.py:61` | the exact production DistilBERT + AdapterHub model, backbone frozen |
| `build_model(num_labels, seq)` | `scripts/profile_jvp_opt.py:64` | that call wrapped with the real `ClassificationArgs` (adapter PEFT, fp16, seq 192, batch 8) |
| `calculate_jvp`, `functional_get_loss` | `trainer/forward_training/fwdgrad_utils.py:105,66` | the *real* central-FD JVP math, importable standalone |
| `stage1_vmap_fd` | `scripts/profile_jvp_opt.py:143` | all `P` probes in one batched pass — **verified bit-identical** to the production loop, and the reason a replica is affordable |
| `calculate_var`, `calculate_snr`, `calculate_cos_sim` | `fwdgrad_utils.py:186,243,349` | the real commit-gate statistics |
| `TextClassificationDataManager.load_federated_data` | `data_manager/text_classification_data_manager.py` | the real agnews H5 partitions at `niid_label_clients=100_alpha=1` |
| `_server_update_step` / `_apply_weighted_update` | `aggregator/FedSgdAggregator.py:238,311` | the exact server arithmetic (~15 lines) to mirror in a replica |
| `_emit_server_update` | `FedSgdAggregator.py:338` | `‖Δθ‖`, `‖W‖`, `η` per commit under `server_update_audit` |
| `characterize_variance_curve.py`, `audit_weight_redundancy.py`, `plot_run.py` | `expt_scripts/` | the streaming-telemetry idiom to copy — **including the `data_id`-cycling workaround**: `data_id` cycles per round, so it must never be used as a reducer key |

> **There is no standing tooling on rungs 1–3.** Every replay to date was one-off shell/Python work, and
> four scripts proposed for this (`replay_step_geometry.py`, `analyze_jvp_selection.py`,
> `probe_gradient_quality.py`, `replica_fluxtune_loop.py`) were **never built** — B1 landed directly in
> the aggregator instead. **Rung 3 has never existed at all**, which is why every A/B in §4.1 cost a sim
> run. Budget for building it before the next fix, not after.

**Rung 3 cost, stated honestly.** Per upload the trainer does `2P` = 20 forward passes (batch 8, seq
192); one commit pools `N ≈ 185` ⇒ **≈ 3,700 passes/commit, ≈ 700k for a 189-commit trajectory.** Two
things make that tractable: `stage1_vmap_fd` batches all `P` probes into ~2 effective passes, and there
is no orchestration overhead (the reference run spent 75 s/commit across 8 GPUs). If it is still too
slow, shorten `seq` and/or reduce `N` and accept the trade explicitly: **the replica's constants will
shift, its mechanism and its ranking of fixes will not** — which is exactly what the validation gate
licenses.

**Sim caveats that survive.** Trainers do real forward-grad compute (gradient values are mode-invariant);
what sim removes is the *waiting*. (a) **Compare sim to sim** — include a sim baseline replicate, since
several reference constants were measured on real runs. (b) Ordering differs between modes, so pool
composition differs commit-to-commit; harmless for `ρ`, `cos`, `B` and `Λ` (aggregates over 100+
commits), **not** safe for a claimed A/B win of a few percent.

**Reproducing any number from logs.** All from `lib/python/examples/fwdllm/experiments/`; nothing needs
a GPU. **Everything in Parts 2–4 comes from two telemetry events**, `server_update` and `agg_eval`:

```
server_update : trainable_weight_norm, trainable_delta_norm, rho, pool_size, var_at_commit,
                n_eff_ratio, split_half_dot, split_half_norm_a, split_half_norm_b,
                cos_ground_truth, pooled_norm, probe_grad_norm
agg_eval      : acc, mcc, loss, logit_norm, pred_entropy, top_class_share
```

Emit flags, all off by default, emit-only, wrapped so they cannot fault training:
**`--server-update-audit`** (the base record) · **`--pool-split-half-audit`** (the split-half
components; its own flag because it adds a pass over params × uploads and must not change the base
record's cost profile) · **`--cos-ground-truth-audit`** (B1's backward pass). `logit_norm`,
`pred_entropy`, `top_class_share` and the direct `‖θ_tr‖` are correctness-or-free and **on by default
since 2026-08-07** — runs older than that lack them.

```bash
RUN=run_20260810_042027_fluxtune_n100_smoke_syn_0_sim   # the cos-probe mean/N=200 arm
# telemetry is ~700 MB per run; slice first (~1 min):
grep -hE '"event": "(server_update|agg_eval)"' $RUN/telemetry/aggregator_*.jsonl > /tmp/$RUN.jsonl

grep 'All JVPs sorted by magnitude' $RUN/*trainers.log     # all P candidates, per selection event
#   "model version" in that line is actually the ROUND; count `tensor(` to read P (a trainer override,
#   never in aggregator_config.json). Normalise each event by its own rms, then:
#   E[v_par^2 | coin top-2] = mean( (d1^2+d2^2)/2 / mean(d^2) )   -> 2.988 (P=10) / 4.744 (P=30)
#   top-k average objective = mean( mean(top-k d^2)/mean(d^2) )*k -> monotone, 3.81 .. 10.00

grep -o '\[IterProgress\] data_id=.* force_commit_planned=[A-Za-z]*' $RUN/*aggregator.log
#   split on data_id change; last row of each bin is the commit.
#   reason = CAP if iter >= max_iter-1 else natural if var < 0.3 else plateau
```

```python
ortho  = (tw[b]**2 - tw[a]**2) / sum(dn[a+1:b+1]**2)   # per 25-commit block; expect 1.000 +- 0.005
rho_t  = dn[t] / tw[t]                                 # == the logged `rho`
step   = dn[t]                                         # flat => ||theta||^2 linear
N_t    = K * pool_size[t]                              # pool_size == iterations, exactly, every record
cos_gt = mean(cos_ground_truth)   # SEM = sd/sqrt(n); sd ~ 3e-3 => ~300 commits for 3 sig figs
GG     = pooled_norm / probe_grad_norm    # vs b*sqrt(p/N); b = sqrt(E) select, 1/sqrt(P) mean

# NEVER average per-commit split-half cosines: one commit's cosine sits under the 1/sqrt(p) ~ 1e-3
# sampling floor while the signal is ~3e-4. The record carries the RAW components for this reason
# (split_half_dot, split_half_norm_a, split_half_norm_b, pool_size) -- pool them per ARM:
cos_sh = sqrt( 2 * sum(split_half_dot) / sum(split_half_norm_a * split_half_norm_b) )
# Simulation at the real p, N confirms this identity and cos = sqrt(N/p) to 0.1%. It is also why the
# split-half GATE is dead (sec 4.4) -- the same arithmetic, one commit at a time, has SNR 0.07.

# THE TWO SCORING NUMBERS (sec 3.4). Both exact at any horizon; G_rule = 2.988 select / P mean.
B   = 0.5*sum(log1p(rho[t]**2) for t in range(1,T))       # budget spent
Phi = exp(B)                                              # == tw[-1]/tw[0] to <0.3% on 21 arms
Lam = sum(rho[t]*sqrt(G_rule*N[t]/p) for t in range(T))   # progress banked
# N[t] = 10*(pool_size or iteration_per_data_id+1) -- pool_size is absent when I == 1 (sec 4.7)
```

**The `p` census** (needs the `test_fwdllm` env, ~1 min) — for `p` and the layer split only. In-run,
`[ProbeDim]` logs the same `p` from `self.params`, so this is usually unnecessary:

```bash
cd /home/dgarg39/flame/lib/python
/coc/scratch/dgarg/miniconda3/envs/test_fwdllm/bin/python -c "
import sys, torch, math; sys.path.insert(0,'.')
from examples.fwdllm.scripts.profile_jvp_opt import build_model
m = build_model(4, 192)
m.add_module('pre_classifier', torch.nn.Sequential())   # the line the trainer runs at :217
tr = [x for _,x in m.named_parameters() if x.requires_grad]
p = sum(x.numel() for x in tr)
print(p, torch.sqrt(sum((x.detach().float()**2).sum() for x in tr)).item(), 0.01*math.sqrt(p))"
# -> 450340  13.325  6.711   <- production. Without the add_module line: 1040932 / 20.356 / 10.203
```

## §6.3 Open hypotheses

*Resolved hypotheses are folded into §2–§4 and deleted from here. Resolved so far: H-A (root cause is
the scale-invariance violation — refined: it is the *absolute* step failing to be held, §2.2), H-B
(`|d|` growth is **genuine** gradient growth, §2.2), H-C (the FD is not the driver — confirmed by H-B),
H-D (collapse is **directional degeneracy**, not logit saturation, §2.6), H-F (the defect is **shared**
with sync fwdllm, §2.4), H-G (`p` can be cut without costing accuracy — **yes**, §4.2), H-I (the
orthogonality ratio is stationary, not a defect signature, §2.2), H-K (α raises `var` through `‖g‖`, not
disagreement — and the portability constant dies in the *step rule*, not the gate, §2.5), H-L (`n_eff`
is an identity, §4.2).*

| ID | hypothesis | how it gets tested | rung (§6.2) |
|---|---|---|---|
| **H-N** | The collapse threshold is **relative** (`Φ ≈ 3.6`), not absolute (`‖θ_tr‖ ≈ 45`) | **Q1.** Every arm to date is at `p` = 450k, where the two are indistinguishable; §3.7 predicts relative | 4 |
| **H-O** | `‖θ_tr‖` is a **symptom** of accumulated directional misaim, not the causal variable | **Q2.** Predicted not to help. If it *does*, `B_max` disappears and `Λ` is uncapped | 4 |
| **H-M** | The `cos` shortfall is **reference attenuation** (a 64-sample held-out batch), not client disagreement | Log `cos(g_A, g_B)` over two disjoint held-out batches (B13). ≈1e-3 confirms attenuation; ≈1 would mean real client disagreement. **Prior raised sharply by §3.4(c)** | 2, then rides free on any run |
| **H-J** | `K ≥ 20` **defers** collapse rather than preventing it; collapse at commit 1,200–1,700 | Largely *predicted* now — `B` extrapolates exactly (§3.4). **Demoted to a confirmation** unless a cheap node is free | 1 (extrapolation), 4 to confirm |
| **H-E** | Staleness stays ≤ 1 **only because `N` is small**; raising `K`/`C` produces genuine staleness | **Staleness histogram from the 08-07 K-sweep (`c` up to 100) — free log replay, data already on disk.** If true, ω-freshness moves from "inert" to load-bearing | **1 — do this one first, it costs nothing** |
| **H-H** | The FD's discarded curvature term `vᵀHv` carries usable signal | The central difference uses only the *difference* of `L(θ±hv)`; their **sum** is `L(+)+L(−)−2L(θ) ≈ h²vᵀHv` — **already computed and thrown away.** Log it, correlate with realised loss decrease at the step scale actually taken. Unlocks the only probe-selection metric that is free and *not* stability-neutral | 2 |

**If a probe-selection stage is retained at all**, select on something other than `|d|` — three
candidates, all already paid for: **curvature `vᵀHv`** (≈free, one extra `L(θ)` amortised over `P`);
**split-half SNR within the bin** (compute `d` on each half of the 8-sample bin, select on agreement —
free); **loss decrease at the step scale** (under trust-ratio the step size is known in advance, so pick
the `v` minimising `L(θ − ρ*‖θ‖v̂)` — trust-region rather than derivative selection; 1 pass per
candidate).

## §6.4 Run queue

**Order of operations**, cheapest-decisive first:

1. **H-E staleness replay** — rung 1, free, data on disk. Do it before anything is launched.
2. **Q2 (weight decay).** 3 lines, one node-hour, and a null result reshapes Part 5. Cheapest decisive
   arm.
3. **Q1 (`rf`=64 to `Φ` = 6.6).** ~330 commits, one arm. Sets `B_max` for everything.
4. **Q3 (rule × `p` at pinned `ρ`, `N`).** Three arms, pre-registered (§6.1). Converts `Λ` from fitted
   to predictive.
5. **B13's two-batch `cos` control (H-M).** ~5 lines, rides on any of the above for free.
6. **Then:** the gate re-run at `K` = 30 (a *throughput* question now, not a safety one); the K-sweep
   with the staleness histogram; the curvature probe (H-H); `fwdllm_plus`, which has never launched.

**Every arm reports `B` and `Λ`** (§3.4), `B` as a fraction of `B_max` and `Λ` against §4.1's
calibration. Both are exact at any horizon, so the two failure modes are diagnosable before the run
ends: **too little `Λ` = never learned; too much `B` = learned and then lost it.**

## §6.5 Preflight and launch

**Three preflights run in `run_sequential.sh`** (CPU-only, ~4 s total, refuses to launch on failure);
`_node_lib.sh` aborts a node on any arm producing <5 commits:

- `test_model_args_parity.py` — every arg the trainer reads unguarded is supplied by **both**
  `trainer/main.py` and `aggregator/main_fedfwd_agg.py`.
- `test_commit_gate.py` — `N_req`'s closed form and its annealed-vs-setpoint divergence.
- `test_cos_probe.py` — cos = ±1 on a pool built from `g` itself.

**Read the enactment checks before the science:**

```bash
grep -m1 '\[ServerStep\]'    $RUN/*aggregator.log   # rho* must equal the logged rho
grep -m1 '\[CommitGate\]'    $RUN/*aggregator.log   # n_have vs n_req per iteration
grep -m1 '\[CosProbe\]'      $RUN/*aggregator.log   # batch size, then per-commit cos
grep -m1 '\[probe_combine'   $RUN/*trainers.log     # trainer-side; NOT in aggregator_config.json
grep -m1 '\[ProbeDim\]'      $RUN/*trainers.log     # THE p (post pre_classifier drop)
grep -m1 '\[FD\] spacing'    $RUN/*trainers.log     # THE h -- read h*sqrt(p) HERE, not off ProbeDim
```

> **`[ProbeDim]`'s `h*sqrt(p)` is not the FD spacing** — it prints the *nominal* `h`, so the `p` ladder
> read 6.71 / 4.79 / 3.44 and looked like the FD had silently shrunk with `p`. It had not: `[FD]` showed
> `h` rescaled to 0.01 / 0.014023 / 0.019507, holding `h√p` at 6.7107 on all three arms.

**Launcher gotchas, each of which costs a wasted run:**

- The baseline flag is **`--only`**, not `--baselines` (unknown args abort).
- **`--yes`** — otherwise each invocation stops at an interactive `[y/N]` prompt.
- **`--clean`** — back-to-back runs otherwise `DIRTY_ABORT` on a prior run's stray workers.
- **`--force`** — the sim-charge-profile pre-flight blocks when audit-on reals are newer than the
  profile; re-profiling from them would bake diagnostic overhead into the vclock model. Confirm with
  **`--dry-run`** that it is the only `✗` first, since `--force` overrides every check.
- **`--num-trainers 100`** pins `minInitialTrainers`, so varying `--c` does not move the warmup
  threshold underneath a sweep.
- The baseline → yaml map is **hardcoded** in `ALL_RUNS`; there is no custom-yaml flag, and
  `fwdllm_plus` has no entry. Use config flags, never a hand-edited yaml.
- **Pin the pool for any A/B** — `--var-threshold 0 --max-iter-per-data-id 20 --var-stopping-policy
  off` ⇒ `I` = 20, `N` = 200 exactly (§4.7).

---

# Part 7 — Generality

*Speculative relative to Parts 1–6, and **frozen** (R9): revisit only when §6.1 closes. The question it
answers is whether this transfers from FL LLM fine-tuning to backprop-free fine-tuning in general —
including the datacenter, where the argument must be speed and memory rather than device constraints.*

## §7.1 The model is general; the controller is where the paradigm enters

```
cos = sqrt( G_rule * n / p )        n = product of all INDEPENDENT pooling stages
rho <= s * cos                      the step must not exceed the aim
n_req = p * (rho*/s)^2 / G_rule     what any configuration must pool
```

**Nothing in those three lines is federated.** They apply to any optimizer estimating a gradient from
directional derivatives. FL enters only in *how `n` decomposes and what each factor costs*: single
device `P` × accumulation (wall clock); centralized ZO `P` × accumulation (no comms); data-parallel
`P` × workers × accumulation (all-reduce); **federated `P` × `K` × `I`**, where `I` costs round trips and
`K` costs staleness and heterogeneity.

Likewise §3.4's two conserved quantities contain nothing federated — `B` is a property of the `ρ`
trajectory alone, and `Λ` needs only `ρ_t` and `cos_t`. **If they hold outside FL, they are the
transferable result.**

## §7.2 A testable prediction against centralized ZO

Centralized ZO fine-tuning uses `P = 1` and a very small fixed learning rate. The criterion predicts
*why*: with `G_rule = 1` and no `K` or `I`, `cos = √(1/p)`, so the stable relative step is ~10⁻³ and the
method needs ~`p` steps for coherent progress — matching the very long step counts such methods report.
**A falsifiable prediction about an existing published method, derivable with no new experiments.**
Check published curves before claiming it.

## §7.3 What must be checked before generalizing

| # | assumption this work rests on | why it might not survive outside our setting |
|---|---|---|
| 1 | **Leg 1 (steps orthogonal to `θ`)**, which makes `B` exact | measured only at `p` ≥ 118k on adapter+head slices. At small `p`, or when the trainable slice includes layers with large initial norm and structured gradients, `⟨θ,Δθ⟩` need not vanish. **Check the orthogonality ratio first on any new model — it is one line of replay** |
| 2 | **`Λ`'s calibration** (0.6 → 0.85, saturating at 0.95) | fitted on agnews/DistilBERT. The *monotonicity* should transfer; the numbers are task- and model-specific. Recalibrate per (model, task) before using `Λ` as a target |
| 3 | **`B_max` ≈ 1.22–1.28** | this is §6.1 Q1 and is unresolved even here. If it is relative (`Φ`), it may transfer; if absolute (`‖θ_tr‖`), it certainly will not |
| 4 | **isotropic probes and the `1/√p` overlap** | exact by construction *given* Gaussian `v` over the whole trainable slice. Structured or block probes (§4.4) break it deliberately — that is their point, and they need their own analysis |
| 5 | **probe noise dominates data noise by ~40×** (**ANALYSIS**) | this is what justifies "spend compute on `P`, not on bigger bins". At larger batch sizes or lower `p` the balance moves, and `K`/batch may become the better spend |
| 6 | **`h` pinned by fp16 cancellation** | a precision property, not a math one. bf16, fp32 or a different loss scale move the usable `h` window and therefore the FD's faithfulness |
| 7 | **The closed-form `cos` (§3.9)** | functionally validated but 25–80× off its direct measurement. **No generalization claim about `cos` should be made while H-M is open** |

## §7.4 Larger models and datacenter fine-tuning

**The case changes from memory to throughput.** On-device, forward-gradient wins because it stores no
activations. In a datacenter with backprop available, the only argument is that 2 forward passes are
cheaper than 1 fwd + 1 bwd and need no activation memory — real, but much narrower, and **`cos ∝ 1/√p`
makes it *worse* at scale** unless `p` is aggressively constrained (which is what PEFT does, and §4.2
says that is now a gradient-quality decision, not an efficiency one).

**MoE is structurally interesting:** only active experts contribute to a forward pass, so `p_effective`
is per-token active parameters — **block-coordinate probing for free**, the one structural idea that
beats the `√(n/p)` barrier. Unanalysed, and the most interesting extension on this list.

## §7.5 Is there a general ML contribution here

In descending confidence:

1. **Yes: `p` is a gradient-quality parameter, not just a memory knob** (§4.2). Inverts the standard
   PEFT intuition; measured. The strongest ML result here and the only one about the *model*.
2. **Yes: the scale-invariance requirement for forward-gradient training** (§3.7) — the step, gate and
   estimator must all be ratios, or the method needs re-tuning per model. With a direct demonstration
   (§2.5d).
3. **Maybe: the two conserved quantities** (§3.4). `B` is exact and parameter-free; `Λ` orders 21 arms.
   Their transfer outside agnews/DistilBERT is untested — §7.3 rows 1–3.
4. **Maybe: the online control law** — `ρ ≤ s·cos` with a gradient-free estimator for `cos`. Novel as
   *packaging*, not theory, and **weakened by B1**: what is publishable today is the *dose-response
   curve*, which is empirical and was not planned.
5. **No: the scaling law itself.** `cos ∝ √(n/p)` is known ZO analysis.

A standalone paper needs three things we do not have: the criterion predicting the divergence point
across **≥3 models and ≥2 tasks**; a usable estimator for `cos`; and the derived controller **beating
hand-tuned schedules without re-tuning** when the model changes.
