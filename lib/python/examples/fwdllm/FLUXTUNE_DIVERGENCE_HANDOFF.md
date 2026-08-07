# FluxTune — divergence root cause, the stability criterion, and the probe-budget question

**Status:** analysis complete, no code changed. Written 2026-08-06 on `dg/fluxtune_expts_sim_init`.
**Purpose:** hand off a finished diagnosis so the next session can go straight to building.

Every claim is tagged:

- **MEASURED** — read out of telemetry / logs / the real model builder. Reproduction in §14.
- **DERIVED** — arithmetic on MEASURED inputs, no modelling assumption.
- **ANALYSIS** — rests on a stated modelling assumption, flagged wherever it appears.
- **HYPOTHESIS** — not yet tested. Collected in §9 with the experiment that would settle each.

---

## 0. Read this first — the six facts that carry everything

1. **The committed update is a pure random walk.** Over 185 commits, 100% of the step energy goes
   into inflating the parameter norm (ratio 1.032). No measurable net descent component.
2. **The *relative* step is pinned at ρ = 0.115 all run**, so the adapter norm grows **geometrically**
   — `‖θ‖ ← ‖θ‖·√(1+ρ²)`, doubling every ~110 commits, predicted to within 2.5%. Divergence is
   structural and present at commit 1. It is not a tuning accident.
3. **The stability condition is `ρ ≤ cos(G,g)`.** Shipped config sits **5.0× over budget**.
4. **`ρ` is set by aggregation; `cos` is set by selection; `N = K × I` is a selection knob times an
   aggregation knob.** Neither side owns the bug, neither can fix it well alone (§10).
5. **C1 as shipped throws away 9 of the 10 measurements it pays for.** Averaging all P probes instead
   of selecting one improves `ρ/cos` by exactly **10×** at *identical* compute and communication —
   on its own that takes the system from 5.0× over budget to **2× under** it (§8).
6. **Any A/B shorter than ~3.2 h is uninformative** — the 2 h runs terminate at 0.85 norm doublings,
   which is exactly the accuracy peak. A short run will report a win for a config that diverges.

---

## 1. Scope — the runs this is based on

| run | mode | duration | commits | outcome |
|---|---|---|---|---|
| `run_20260804_003042_fluxtune_n100_smoke_syn_0_real` | real | 3.93 h | 189 | peak acc 0.846 @ r1/did106 → **0.250 / mcc 0.000 / loss 2.37** @ r2/did38 |
| `run_20260804_043301_fluxtune_n100_smoke_syn_0_real` | real | 3.93 h | 185 | peak acc 0.853 @ r1/did122 → **0.252 / loss 1.54** @ r2/did34 |
| `run_20260805_{110016,130231,150446}_fluxtune_..._real` | real | 1.86–1.88 h | ~95 | stop at did 92–94 — **at the peak, before the collapse** |

**Why it looked intermittent.** The 2 h runs terminate on `max_runtime_s` after ~95 commits. The
norm-doubling time is 109–116 commits (§5, Leg 2). **The short runs end at 0.85 doublings — exactly
where accuracy peaks.** That is why the failure looked sporadic, and why the 4 h floor is
non-negotiable.

Both 4 h runs contain the failure and agree on every derived constant. Deterministic, not a bad seed.

**No usable baseline control exists yet.** The `fwdllm` runs from the same period
(`run_2026080[45]*fwdllm_n100_smoke_syn_0_real`) only reach `data_id=38` in 1.84 h and never leave
acc 0.25 — the sync path is too slow at this horizon. Its `|jvp|` is flat (rms 3.1–4.0 over 3,101
merges) because its weights barely move, which is not evidence of stability. **A fair cross-baseline
comparison at the horizon where fluxtune fails does not exist** (see H-F).

---

## 2. The shipped configuration (the referent for everything below)

From `aggregator_config.json` of the reference run and `expt_scripts/fluxtune_n10_smoke.yaml`
(launched by `expt_scripts/run_sequential.sh`, entry `fluxtune:`). **MEASURED.**

```
population N_clients=100, alpha=1 (Dirichlet), agnews, DistilBERT-base + AdapterHub adapters
trainable p = 1,040,932  (1.5% of 67.4M; backbone frozen)
train_batch_size = 8          -> one "data bin" = 8 samples, 150 bins per round
perturbation_count P = 10, h = 0.01, central finite difference, fp16/autocast, no_grad
selection: rank by |JVP|, coin-flip between top-2   (tc_transformer_trainer_distribute.py:481-483)
selector: async_oort, c = 30, agg_goal K = 10, dynamic_kc.enabled = FALSE
commit gate: var_threshold = 0.3, var_stopping_policy = plateau (patience 3, rel_delta 0.15),
             max_iterations_per_data_id = 20
server step: theta <- theta - eta * (1/N_acc) * sum_k omega_k * d_k * v_k   (raw in-place SGD)
             eta = 0.01, measured constant to 4 digits for the whole run
             server_momentum absent => 0.0 (no-op). No optimizer state of any kind.
omega: grad_aware / base=new / align_gate=true / align_floor=0.0 / inverse_var=false
       scale=0.4, a_exp=0.25, b_exp=0.1        [measured omega in 0.702..0.865, median 0.818]
```

**Where the 1.04M probe dimensions actually live** (**MEASURED**, from the real builder) — this turns
out to matter a great deal (§8.5):

```
pre_classifier   590,592   56.7%      <- ONE 768x768 layer is over half the probe dimension
adapters         447,264   43.0%
classifier         3,076    0.3%
```

Two properties of the estimator that drive everything, both **MEASURED**:

- **`v` is a raw Gaussian draw, never normalized** (`tc_transformer_trainer_distribute.py:416`), so
  `‖v‖ = √p ≈ 1020` and the FD probe displacement is **`h·‖v‖ = 0.01 · 1020.3 = 10.203`**.
- **`‖θ_trainable‖` at init = 20.356** (from `build_model(4, 192)`, the production path).

> **Intuition to hold onto:** the finite difference displaces the *entire trainable parameter vector
> by 50% of its own norm*. That is a chord across half the parameter space, not a derivative. And its
> scale is an **accident of the parameter count** (`√p`) — nobody chose it. Change the adapter width
> and the effective step size of the whole algorithm changes with it.

---

## 3. What failure looks like

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

1. **Final loss 2.37 > ln(4) = 1.386.** The model is *confidently wrong*, not uncertain. A model that
   had merely lost its signal would sit **at** ln 4 with uniform predictions. Far above it means the
   logits are large and pointed the wrong way — blown-up weights, not forgetting.
2. **Train loss diverges too.** `stat_utility` (Oort utility = `8·rms(batch loss)`, so `loss ≈ U/8`)
   goes **10.84 → 5.31 (did 120) → 13.74 (r2 did 35)** = train loss 1.355 → 0.66 → 1.72. Train and
   test move together — **not overfitting**.
3. **Degradation starts inside round 1** (did 138–148), before the round boundary. The boundary is
   not the trigger. (Consistent with F10, which already REFUTED the staleness-reset bug.)

> **This supersedes the framing in `fluxtune_contributions.md` §8.** That section describes the
> failure as *oscillation with recurring single-class collapses*. At 4 h the actual behaviour is a
> clean **monotone rise, then a monotone, irreversible divergence**. F4/F5's position-locked
> collapses are an early *symptom* of the same noise, not the mechanism — and H3 (bin-order
> permutation) does not touch the mechanism at all.

---

## 4. Root cause — three measured legs

### Leg 1 — the committed update is a pure random walk. **MEASURED.**

If a step `Δθ` were doing useful descent it would have a component along `θ`. Test: does `‖W‖²` grow
by exactly `‖Δθ‖²`? (That happens **iff** `Δθ ⊥ θ`.)

```
sum over all commits of observed d(||W||^2)  /  sum of ||dTheta||^2
      run_003042 : 1.0326        run_043301 : 1.0323
```

**100% of the step energy goes into inflating the parameter norm, to within 3%.** F8 with a number,
and assumption-free — it uses only `weight_norm` and `update_delta_norm`, both already logged under
`server_update_audit`.

*Why this is the right test:* in 1.04M dimensions an isotropic random vector is orthogonal to any
fixed vector to within `1/√p ≈ 0.001`, so `Δθ ⊥ θ` is exactly what "random direction" predicts. Any
systematic descent would show as a deficit in this ratio. There is none.

### Leg 2 — the *relative* step is pinned, so the norm inflates geometrically. **MEASURED + DERIVED.**

Reconstruct `‖θ_tr,t‖` from telemetry using only *differences* of `‖W‖`, anchored at the measured
init (this cancels the frozen-norm constant; both runs independently imply `‖θ_fz‖ = 415.929`,
matching the directly measured 416.213 to 0.07% — a strong consistency check):

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

**Why ρ stays constant (the positive feedback loop).** Because `|JVP| ∝ ‖θ_tr‖`. From all 34,447
selection events (**MEASURED**):

```
rms|d| over the run:  3.59 (did 0)  ->  21.5 (r2 did 35)   = 6.0x
||theta_tr||       : 13.6           ->  81.4               = 6.0x
```

The loop closes: **noise inflates the norm → bigger norm produces bigger `|JVP|` → bigger step → more
inflation.** Multiplicative, exponential, unstable at initialization (ρ₀ = 0.113). No restoring force
anywhere in the system. **Why** `|d| ∝ ‖θ_tr‖` is the one part not yet proven — see H-B in §9; the
§13 probe settles it for free.

> **This retro-explains the S1 NaN.** Heavy-ball at β = 0.9 multiplies the effective relative step by
> `1/(1−β) = 10` → ρ_eff = 1.13 → the norm doubles *every step*. S1 was not the wrong idea; it was
> applied to an already multiplicatively-unstable process. The lesson is ordering, not rejection.

### Leg 3 — the variance gate is dimensionally wrong and 100% dead. **MEASURED.**

From 3,441 `[IterProgress]` lines covering all 186 bins:

```
commit reasons:  natural (var < 0.3) = 0     plateau = 105     cap(max_iter=20) = 81
bins that EVER reached var < 0.3 at any iteration:  0 of 186

achievable variance floor (median per-bin minimum), by 20-bin block:
   bins   0- 20 : 0.415        bins 100-120 : 1.462
   bins  60- 80 : 0.674        bins 140-160 : 3.911
                               bins 180-186 : 14.97      <- 36x drift over the run
```

**36× is exactly (6.0×)².** Variance is a second moment of `d`, and `d ∝ ‖θ_tr‖`, so the gate's own
measurement scale grows as `‖θ‖²` while `var_threshold` stays at 0.3.

> **S2 ("recalibrate `var_threshold`") is structurally unfixable as scoped.** No constant can be
> correct for more than an instant against a statistic that drifts 36× within one run. The problem is
> not the value — the quantity **has dimensions**. Any fix must replace it with a dimensionless one.
>
> Second consequence: the two escape hatches *are* the commit policy. Opt-2 was designed as a safety
> net and is in fact the entire mechanism — so **`max_iterations_per_data_id = 20`, a wall-clock
> convenience knob, is the de-facto variance controller** and therefore the de-facto setter of half
> of `N`. Not a defensible design position to write up.

### The single root cause

**Nothing in the pipeline is scale-free.** The estimator (`|d| ∝ ‖θ‖`), the step (`‖Δθ‖ ∝ |d|`) and
the gate (`var ∝ |d|²`) all inflate together, so no quantity anywhere can be meaningfully compared
against a fixed constant. All three legs are one defect seen from three places.

---

## 5. Why it looks like it's working for three hours

Three quantities accumulate on **three different timescales**:

| quantity | accumulates as | per-commit rate |
|---|---|---|
| useful (aligned) displacement | **linear** in T — aligned components add coherently | `ρ·cos` |
| noise displacement | **√T** — random directions add in quadrature | `ρ` |
| parameter-norm inflation | **exponential** — `(1+ρ²)^{T/2}` | `ρ²/2` |

Early on, linear beats √T: even though **97.7% of every step is noise**, the 2.3% that is aligned
adds up coherently while the noise partially cancels. That is the accuracy climb, and it is real
learning — the estimator's *direction* is unbiased (§6.1). The exponential term is invisible in
absolute terms at first.

But exponential eventually beats everything. Once the norm has inflated enough the logits saturate,
the model becomes confidently wrong, and the loss goes above ln 4. The curve's shape — smooth rise,
rounded peak, accelerating fall — is the signature of a linear term overtaken by an exponential one.

**Numerically:** signal overtakes noise only after `T ≈ 1/cos² = 1881` commits; the norm doubles every
109. **The system needs ~17× more time than it has before it destroys itself. Divergence is not a
risk here — it is arithmetic.**

---

## 6. The stability criterion

### 6.1 Derivation

Per commit, `G = (1/N) Σ_k ω_k u_k` where `u_k` is one trainer's uploaded update and `N` = number of
individual uploads accumulated (**MEASURED: N ≈ 185** = K=10 × 18.5 iterations/bin).

Write each upload as `E[u] = a·g` and `‖u‖ ≈ b·‖g‖·√p`. Then, since `p ≫ N` (noise-dominated):

```
||G||       ~ b * ||g|| * sqrt(p/N)          aligned part ~ a * ||g||
cos(G, g)   = (a/b) * sqrt(N/p)
rho         ~ eta * ||G|| / ||theta||     ->    rho / cos  ~  (b^2/a) * p / N
```

**These two ratios `a/b` and `b²/a` are the whole game** — every design choice in §8 is a choice of
`a` and `b`. For the shipped rule (select one probe of P by `|JVP|`): `a = E[v∥²]`, `b = √(E[v∥²])`,
so `a/b = √(E[v∥²])` and **`b²/a = 1`**.

`E[v∥²]` is now **MEASURED**, not assumed — from the 34,447 logged selection events, normalising
within each event (§14):

```
random (any 1 of 10)            E[v_par^2] = 1.000   (by construction)
top-1 of 10                                 = 3.811
coin-flip top-2 of 10 (SHIPPED)             = 2.991
```

So, with `N = 185`, `p = 1,040,932`:

```
cos(G, g) = sqrt(2.991 * 185 / 1040932) = 0.0231      <- ANALYSIS only in the isotropy assumption;
                                                         E[v_par^2] itself is now MEASURED
```

Composing with Leg 1 (`Δθ ⊥ θ`, so `‖θ‖² ← ‖θ‖²(1+ρ²)`): over `T` commits signal accumulates as
`T·ρ·cos` and noise as `√T·ρ`, so signal dominates only after `T ≳ 1/cos²`, and the norm must survive
that long, requiring `ρ²·T ≲ 1`. Substituting:

> ## ρ ≤ cos(G, g)
> **The relative step must not exceed the fraction of it that is actually aligned with the gradient.**
> Take a step bigger than you can aim, and the norm blows up before the aim pays off.

### 6.2 Where the shipped config sits

```
rho (MEASURED)         = 0.115
cos(G,g)               = 0.0231
rho / cos              = 5.0x  OVER BUDGET
1/cos^2 = 1881 commits to signal dominance ; norm doubles every 109 -> ~17x too slow to escape
```

### 6.3 The scaling table — this ranks every possible fix

The core identity: **averaging `n` near-orthogonal estimates shrinks the noise as `1/√n` but leaves
the aligned part untouched.** Hence `ρ ∝ 1/√n`, `cos ∝ √n`, product constant:

| lever | `ρ/cos` (stability) | `ρ·cos` (per-commit progress) | who owns it |
|---|---|---|---|
| **N = K × I** (server-side pooling) | **∝ 1/N** | **invariant — costs nothing** | **K,C: selection · I: aggregation** |
| **P-averaging** (trainer-side pooling) | **∝ 1/P** | invariant | selection (§8) |
| **p** (trainable dimension) | **∝ p** | ∝ 1/p | model/PEFT design (§8.5) |
| `η` learning rate | ∝ η | ∝ η — **pays 1:1** | aggregation |
| selection gain `E[v∥²]` | **invariant** (`b²/a = 1`) | ∝ E | trainer-side selection |
| step normalization | sets ρ to an operator constant | decoupled | aggregation |

Per unit of **compute** (T commits × N samples each): progress ∝ `C/N`, norm inflation ∝ `C/N²`, so
the ratio improves ∝ `N`. **Larger pools are strictly better for stability per FLOP and slower only
in absolute wall-clock progress.** That is the honest trade and the whole argument for pool control
being a contribution rather than a tuning detail.

---

## 7. What is proven vs. what is assumed

| claim | status |
|---|---|
| step is orthogonal to weights (random walk) | **MEASURED**, ratio 1.032, assumption-free |
| ρ ≈ 0.115 constant; norm growth geometric | **MEASURED**, model predicts final norm to 2.5% |
| `|d| ∝ ‖θ_tr‖` (6.0× vs 6.0×) | **MEASURED** (the *why* is open — H-B) |
| variance floor drifts 36× = (6.0×)²; 0/186 natural commits | **MEASURED** |
| `E[v∥²]` = 3.811 (top-1) / 2.991 (shipped coin-flip) | **MEASURED**, n = 34,447 events |
| candidate JVPs indistinguishable from iid Gaussian | **MEASURED**, matches synthetic to 0.3% (§8.2) |
| top-k averaging is monotone in k → use all P | **MEASURED**, offline sweep (§8.3) |
| `cos(G,g) ≈ 0.0231` | **ANALYSIS** — isotropy assumption only; §13 measures it directly |
| criterion `ρ ≤ cos`; `ρ/cos = 5.0×`; `N_req ≈ 925` | **DERIVED**, inherits that one caveat |

---

## 8. Is C1 (guided perturbation selection) the right use of the probe budget?

**Short answer: the 2P forward passes are well spent, but the selection rule spends them badly.**
Computing P probes is worth up to a 10× stability gain; selecting one of them and discarding the rest
captures **none** of it. This section is the evidence.

### 8.1 What selection provably can and cannot buy

From §6.1, a rule that selects one probe has `a = E[v∥²]`, `b = √(E[v∥²])`, therefore:

```
cos       ~ sqrt(E[v_par^2]) * sqrt(N/p)      <- selection DOES improve alignment
rho/cos   ~ (b^2/a) * p/N = 1 * p/N           <- selection does NOT improve stability, at all
```

**Selection is exactly stability-neutral, for any `E[v∥²]`.** It raises `cos` and raises `ρ` by the
identical factor. Everything it buys in aim, it spends in step size. Under the shipped raw-SGD step
that means C1 is a genuine ~3× per-commit progress multiplier *and* a ~3× accelerant of the blow-up.

### 8.2 Evidence: selection has nothing to find. **MEASURED.**

If the P candidate JVPs carried exploitable structure, their distribution would deviate from the
Gaussian that the linear model predicts (`d = ⟨g,v⟩` with `v` isotropic ⇒ `d ~ N(0,‖g‖²)` exactly).
Over 34,447 events / 344,470 candidate JVPs, comparing the *same* order-statistic estimator on the
observed data and on synthetic iid `N(0,1)`:

```
                          observed     synthetic iid Gaussian
top-1 of 10                 3.811            3.798            <- match to 0.3%
coin-flip top-2 of 10       2.991            2.987            <- match to 0.1%
pooled skew                +0.0011            0                (kurtosis 2.51 vs 3.0 is an artifact
                                                                of standardising by each event's own
                                                                10-sample rms, not real structure)
```

**The candidates are statistically indistinguishable from iid Gaussian draws.** Selection therefore
extracts *exactly* the order-statistic gain of sampling a Gaussian tail — and nothing more. This is
not an empirical accident to be tuned away: for a linear functional of an isotropic Gaussian probe,
there is provably no structure for a selection rule to exploit. (It also confirms the FD's
non-linear truncation term is small enough not to show up as non-Gaussianity.)

### 8.3 The k-sweep, settled offline. **MEASURED.**

`FLUXTUNE_CODE_QA.md` §D2 records that the top-k sweep "cannot run today" (k is hardcoded to 2, no
config knob). **It does not need to run** — the objective is computable from the already-logged JVPs.

For a rule that *averages* the top-k of P: `a = E_k`, `b = √(E_k/k)`, so `a/b = √(E_k·k)` and
`b²/a = 1/k`. That is, **`cos ∝ √(E_k·k)` and `ρ/cos ∝ 1/k`**:

```
  k   E[v_par^2|top-k]   E_k*k   cos gain   rho/cos improvement
  1        3.811         3.811     1.95x         1x   (none - selection is stability-neutral)
  2        2.991         5.982     2.45x         2x
  3        2.469         7.406     2.72x         3x
  5        1.806         9.030     3.00x         5x
 10        1.000        10.000     3.16x        10x
```

**Monotone in k on both objectives. The optimum is k = P: use every candidate.** Note the shipped
rule is *not* row k=2 — it is a coin flip that reports **one** of the top two, i.e. k=1 with
`E = 2.991`, so it sits at `cos` gain 1.73× and **zero** stability improvement.

### 8.4 The comparison that matters, at identical compute

All rows cost the same `2P = 20` forward passes and the same dense `p`-vector upload:

| rule | `cos` gain | `ρ/cos` improvement | progress at the stability boundary (`∝ E_k·k`) |
|---|---|---|---|
| random 1 of 10 | 1.00× | 1× | 1.00 |
| top-1 of 10 | 1.95× | 1× | 3.81 |
| **coin-flip top-2 (SHIPPED)** | **1.73×** | **1×** | **2.99** |
| **average all 10 (S-H)** | **3.16×** | **10×** | **10.00** |

Concretely, holding `η`, `N`, `p` fixed and only changing the rule:

```
SHIPPED:      rho = 0.115   cos = 0.0231   rho/cos = 5.0    <- diverges
AVERAGE-ALL:  rho = 0.021   cos = 0.0422   rho/cos = 0.50   <- 2x INSIDE the stability budget
```

> **Averaging the probes you have already computed is a 10× stability improvement for zero
> additional compute, zero additional communication, and no change to η, N or p.** It alone takes the
> system from 5.0× over budget to 2× under it.

And among *stable* configurations (i.e. after normalising ρ to the boundary in both), averaging is
**3.34× faster per commit** than selecting (`10.00 / 2.99`). Selection's only advantage was that it
took bigger steps — which under S-A is exactly the thing the operator sets by hand.

**Why this is good news, not bad.** It reframes the §3.3 cost story in `fluxtune_contributions.md`
favourably: the `2P` forward passes stop being *overhead for a heuristic* and become a
**variance-reduction budget with linear, measured return** (`ρ/cos ∝ 1/P`). "10× the compute of sync
fwdllm" is a much easier claim to defend when it buys a measured 10× in the quantity that governs
convergence.

**Caveats, stated honestly:**
- P-averaging attacks **probe noise only** — the P probes share one data bin and one θ. `K` attacks
  probe *and* data noise. But probe noise dominates by ~40× (relative probe noise `√(p/N_eff) ≈ 43`
  vs data noise O(1) for an 8-sample bin) — **ANALYSIS** — so P-averaging captures nearly all of the
  available gain. *This also settles the H2/M1 direction: spend compute on probes, not bigger bins.*
- Memory: accumulate one running sum and discard each `v` — the §3.1 "peak memory independent of P"
  claim is preserved. The current code already materialises all P candidates in `v_buffer`, so this
  is not a regression.
- The averaged estimator is the **unbiased** one (`E[(1/P)Σ d_i v_i] = g`); the shipped one carries a
  ~3× scale bias. That bias is precisely the unmatched-effective-LR confound `FLUXTUNE_CODE_QA.md`
  §E3.1 warns about, now with a measured number.

### 8.5 The lever nobody has looked at: `p` itself

`cos = (a/b)·√(N/p)`. Every idea above moves `a`, `b`, or `N`. **`p` is the largest number in the
expression and it is a free design choice.** `ρ/cos ∝ p`, so halving `p` is worth exactly as much as
doubling `N` — and unlike `N` it costs no wall clock.

From the trainable-parameter census in §2: **one 768×768 `pre_classifier` layer is 56.7% of the
entire probe dimension.**

```
freeze pre_classifier:  p 1,040,932 -> 450,340    cos x1.52    rho/cos improves 2.31x
```

**Half the 5.0× gap closes by removing one layer from the trainable set**, at zero compute cost, with
less communication and less memory. Whether that layer is needed for accuracy is an empirical
question nobody has asked — it is trainable by HuggingFace default, not by design.

> **The general statement, and it is a genuine contribution-grade insight:** for backprop FL, PEFT
> rank is a *memory and communication* knob and gradient quality is unaffected by it. For
> forward-gradient FL, **`cos ∝ 1/√p` — PEFT rank is the primary determinant of gradient quality.**
> Adapter width is not an efficiency detail here; it is an algorithmic parameter. This ties directly
> to the memory/hardware thesis in `fluxtune_contributions.md` §3 and strengthens it: the same choice
> that makes the method fit on the device also makes its gradients better.

### 8.6 Alternatives — what else could we select on, or sample from?

**Family A — what to do with the P measurements.** *(Settled: average them, §8.3.)*

| option | verdict |
|---|---|
| select top-1 / coin-flip top-2 | stability-neutral; discards 90% of the measurements |
| **average all P** | **dominant on both objectives — this is S-H** |
| top-k average, k < P | strictly worse than k = P on both axes (§8.3) |
| least-squares / min-norm solve over `{(v_i,d_i)}` | at `P ≪ p` this equals the average up to scale (`FLUXTUNE_CODE_QA` §E2) — **no gain in `cos`; do not build it** |
| orthogonalise the P probes before averaging | **no-op** — random probes in 10⁶ dims are already orthogonal to `1/√p ≈ 0.001` |
| `|d|/‖v‖` instead of `|d|` | **no-op** — `‖v‖` concentrates to 0.07% (`FLUXTUNE_CODE_QA` §C2) |

**Family B — what to select *on*, if a selection stage is kept for a different purpose.** Selection
cannot improve `ρ/cos`, but it can select for *properties other than magnitude*, and two of these are
already paid for:

| metric | why it is interesting | cost |
|---|---|---|
| **curvature `vᵀHv`, from the sum of the two FD passes** | the central difference computes `L(θ+hv)` and `L(θ−hv)` and uses only their **difference**. `L(+)+L(−)−2L(θ) ≈ h²vᵀHv` — **the curvature is already computed and thrown away.** Select for high `|d|` *and* low `vᵀHv` = directions supporting a large safe step. Directly serves the trust-region step (S-A) | **≈ free** — one extra `L(θ)` per iteration, amortised over all P |
| **split-half SNR within the bin** | CE is mean-reduced, but a forward pass yields per-sample losses at no extra cost. Compute `d` on each half of the 8-sample bin and select on *agreement*, not magnitude → selects directions that generalise rather than fit bin noise. Trainer-level analogue of S-E's split-half cosine gate | **free** |
| **actual loss decrease at the step scale** — pick `v` minimising `L(θ − ρ*‖θ‖v̂)` | under S-A the step size is known in advance, so select the probe that genuinely lowers the loss *at the displacement you will take*. Trust-region selection rather than derivative selection — more honest when `h‖v‖` is 50% of `‖θ‖` | 1 extra pass per candidate |

**Family C — what to sample *from* (the structural lever on `cos`).** Only these can beat the
`√(N/p)` barrier:

| idea | assessment |
|---|---|
| **reduce `p`** (PEFT rank; freeze `pre_classifier`) | **S-I — best value in the whole doc: 2.31× for free** (§8.5) |
| **block-coordinate probing** — probe one adapter layer at a time, `p → p/L` per probe | promising: `ρ/cos` improves ~`L` while each commit updates `1/L` of the params. Known in ZO optimisation. **Needs analysis before building** |
| **low-rank / subspace probing** — sample `v` in a `q ≪ p` subspace | the only structural route past `√(N/p)`, but needs a good subspace *and* a way to broadcast it (`q·p` floats is prohibitive) |
| **momentum in the probe distribution** — bias `v` toward the EMA of committed updates | elegant in principle (momentum where it helps aim, not where it inflates the step — sidestepping exactly why S1 NaN'd), **but it does not pay**: the accumulated trajectory has `cos ≈ √T·0.023 ≈ 0.23` after 100 commits, so as a control variate it removes only `1−cos² ≈ 5%` of the variance. **Computed negative — do not chase** |
| **cross-trainer probe coordination** (server assigns disjoint subspaces to the K trainers) | **near no-op** — at `K·P = 500 ≪ p = 10⁶` the probes are already effectively disjoint |

### 8.7 The stand to take on C1

**Do not abandon C1 — redirect it.** Three claims, in descending confidence:

1. **The probe budget is justified.** `2P` forward passes buy a measured `P`-fold improvement in the
   quantity that governs convergence. Linear return, no diminishing.
2. **The *combination* rule is the contribution, not the selection rule.** "Compute P directional
   derivatives and assimilate all of them" is the trainer-side analogue of the server-side pooling,
   and the criterion `ρ ≤ cos` is what makes that a principled design rule rather than a preference.
   The estimator itself (multi-point ZO averaging) is known; **the criterion and the resulting design
   rule are the contribution.**
3. **If a selection stage survives, it should select on curvature or split-half SNR, not on
   magnitude** — because magnitude selection is provably stability-neutral (§8.1) and the candidates
   provably carry no other structure in `|d|` (§8.2), while the curvature signal is *already being
   computed and discarded*.

---

## 9. Hypothesis ledger — what to test next and what each answer changes

| ID | Hypothesis | Status | Experiment | What changes |
|---|---|---|---|---|
| **H-A** | Root cause is the scale-free violation → geometric random walk | **VERIFIED** (3 legs, §4) | — | — |
| **H-B** | `|d| ∝ ‖θ_tr‖` is a **genuine gradient-norm growth**, not an FD artifact | **OPEN** | log `‖g_backprop‖` on a fixed probe batch beside `rms|d|` — **free, same probe as §13** | If genuine: norm control is *curative*. If artifact: fix is normalising `v` / relative `h`. **S-A works either way** |
| **H-C** | The FD is not the driver — the probe gets **relatively smaller** over the run (`h‖v‖/‖θ_tr‖`: 0.50 → 0.12), i.e. the estimator becomes *more* faithful as it diverges | **SUSPECTED**, favours H-B-genuine | same probe | If confirmed, drop the "shrink h" thread entirely |
| **H-D** | The collapse endpoint is **logit saturation** from the inflated head/adapter norm | **SUSPECTED** (loss 2.37 ≫ ln 4; acc exactly 0.250, mcc exactly 0.000) | log prediction entropy + logit norm at `agg_eval` | Confirms norm→logits→collapse; makes norm the one production monitor |
| **H-E** | Staleness stays ≤ 1 **only because N is small**; raising K/C produces genuine staleness | **SUSPECTED** (`FLUXTUNE_CODE_QA` §A4's own structural argument) | staleness histogram during the S-D sweep | If true, **C3-freshness moves from "inert" to load-bearing** (§11) |
| **H-F** | The instability is **not** fluxtune-specific — `_server_update_step` is shared code | **OPEN** | 4 h `fwdllm`/`fwdllm_plus` runs with `server_update_audit`, ρ computed identically | Decides whether S-A/S-B are a **contribution** or **cross-baseline hygiene** — materially changes the paper's claim structure |
| **H-G** | `pre_classifier` (56.7% of `p`) is **not needed** for accuracy | **OPEN** | freeze it; 4 h run; compare peak accuracy | If true, **2.31× of the 5.0× gap closes for free** (§8.5) |
| **H-H** | The FD's discarded curvature term `vᵀHv` carries usable signal for probe selection | **OPEN** | log `L(+)+L(−)−2L(θ)` per candidate; correlate with realised loss decrease | If true, unlocks the only selection metric that is both free and not stability-neutral (§8.6-B) |

---

## 10. Who owns what — selection vs. aggregation

The natural reading ("it's an aggregation bug") is half right, and acting on that half alone costs a
5× slowdown.

### 10.1 The seam

```
rho      = ||dTheta|| / ||theta_tr||        <-- AGGREGATION owns this (the step rule)
cos(G,g) = (a/b) * sqrt(N/p)                <-- SELECTION owns this (via P, N, and p)
the failure is the RATIO
```

One level down is the decisive fact:

> **N = K × I**, where **K** (`agg_goal`) and **C** (concurrency) are selector-owned, and **I**
> (iterations per data bin) is aggregator-owned — the redispatch decision lives in the variance gate
> at `FedSgdAggregator.py:450-534`.
>
> **The single quantity that determines gradient quality is a selection knob times an aggregation
> knob.** That is why C2 and C3 cannot be claimed as independent contributions.

And `P` is a *third* pooling stage that sits entirely inside the trainer (§8) — so the estimator is
pooled at three levels (`P` probes → `K` trainers → `I` iterations) of which only the middle one is
unambiguously "selection".

### 10.2 The asymmetry — the part worth internalizing

**Aggregation can only shrink ρ. It can never raise cos.** Nothing on the server can improve the
quality of the estimate it was handed; it can only avoid wasting it. Clamping ρ from 0.115 → 0.0231
at today's pool buys guaranteed stability, but per-commit progress `ρ·cos` falls **5.0×**. That is the
"η pays 1:1" row of §6.3, and *normalisation pays it too* if it acts alone.

**Selection can raise cos, which raises the ρ *budget* — for free.** Since `ρ ∝ 1/√n` and `cos ∝ √n`
for any pooling stage `n ∈ {P, K, I}`, growing the pool by 5×:

```
rho  = 0.115 / sqrt(5) = 0.0514        cos = 0.0231 * sqrt(5) = 0.0516     <- they meet
rho*cos unchanged                       <- EXACTLY today's per-commit progress. Zero cost.
```

**Selection is the only knob that buys stability without paying for it** — and §8 shows the cheapest
pool to grow is the one already being computed and discarded.

**But selection alone runs out.** Even at the realistic ceiling, `cos` saturates: `K = 100`, `I = 20`,
`P = 10` gives `cos ≈ 0.13`, and with ρ held *constant* the iterate never converges — it hovers in a
noise ball of fixed relative radius. **Selection can buy a budget; only aggregation can anneal
ρ → 0 and actually converge.**

> **The division in one line: selection sets how large ρ is *allowed* to be; aggregation *spends*
> within that budget and drives it to zero.**

### 10.3 Three places the coupling is required, not merely convenient

**(a) The controller's sensor and actuator are on opposite sides.** You measure ρ from
`server_update` telemetry (aggregation) and actuate on K/C/P (selection). A pure-selection controller
has nothing to measure — which is exactly why today's `dynamic_kc_policy.py` targets
`target_iter_per_data_id: 15`, a heuristic with no connection to the estimator. A pure-aggregation
controller can clamp ρ but cannot tell whether it is clamping harder than necessary. **Closed loop
needs both. This is the strongest argument that C2 + C3 are one contribution.**

**(b) Normalisation is a prerequisite for the pool sweeps to be *interpretable*.** Under raw SGD,
growing any pool changes ρ **and** cos simultaneously — so a K-sweep or a P-sweep is an
unmatched-effective-learning-rate comparison, the same objection `FLUXTUNE_CODE_QA.md` §E3.1 raises
against guided-vs-random. Under a trust-ratio step ρ is pinned by the operator, so the sweep moves
cos with the step size held constant. **You currently cannot run a clean C1 or C2 ablation, and the
blocker is an aggregation-side defect.**

**(c) The commit gate is a selection decision made from aggregation telemetry.** "Commit vs. take
another iteration" **is** the `I` half of `N`. S-E is not a third thing — it is the seam itself, and
it points at adaptive `N` per bin: **commit when the measured split-half cosine exceeds ρ.**

### 10.4 Ownership per solution

| | selection (P, C, K) | aggregation (step rule, gate, ω) |
|---|---|---|
| S-A trust-ratio step | — | **pure** |
| S-B ρ annealing (Robbins–Monro) | — | **pure — selection structurally cannot do this** |
| S-C closed-loop controller | actuator (K, C, P) | sensor (ρ) — **needs both** |
| S-D widen K, don't lengthen I | **pure** — only possible because async | — |
| S-E split-half-cosine gate | it *is* the I half of N | statistic is aggregation-side — **the seam** |
| **S-H average all P probes** | **pure — trainer-side pooling** | benefits from S-A to be measurable |
| **S-I shrink `p`** | model/PEFT design — upstream of both | — |
| S-F selection→direction, not magnitude | trainer-side selection | normalisation — **needs both** |
| S-G weight decay | — | pure (hygiene, not a contribution) |

---

## 11. Corrections and re-prioritizations this forces

| Existing claim | Verdict |
|---|---|
| `fluxtune_contributions.md` §8 headline: "never converges — it *oscillates*" | **Superseded.** At 4 h it is a monotone rise then monotone divergence |
| S1 momentum "REFUTED as-designed" | **Re-framed.** Correctly refuted *at ρ = 0.115*; ρ_eff = 1.13 explains the NaN exactly. Testable once ρ is bounded |
| S2 "variance-gate recalibration" | **Structurally unfixable as scoped** (Leg 3) |
| S3 "aggregation-rate tempering / cap ω ≤ 1" | **Correct but ~2 orders of magnitude too small.** ω ∈ [0.702, 0.865]; the problem is 5.0× |
| H1 shuffle, H3 bin-order permutation | **Do not address the mechanism.** Park |
| H2 bin size / M1 sweep | **Direction now settled: spend compute on probes, not bigger bins** — probe noise dominates data noise ~40× (§8.4) |
| C1 "guided selection improves accuracy" | **Substantially revised — see §8.** Stability-neutral by construction; dominated 3.34× by averaging the same probes. Redirect, don't abandon |
| C3-direction (alignment gate) | **Park.** ≤0.6% weight perturbation vs a 5.0× problem |
| C3-**freshness** (staleness) | **REVERSED — do not park.** The ≤8%-on-15.6% verdict is *conditional on N being small*; `FLUXTUNE_CODE_QA` §A4 says staleness stays ≤1 for a structural reason. **S-D creates precisely the high-C/K regime §A4 names as the one where freshness becomes load-bearing.** Untestable today; on the critical path immediately after S-D. Report the staleness histogram with the K-sweep (H-E) |
| QA §D2 "the k sweep cannot run today" | **Answered offline** (§8.3) — the objective is computable from already-logged JVPs. Monotone; optimum k = P |
| QA §E1 "measuring `cos(G,g)` needs `v_k` uploaded" | **Not for this quantity** (§13) — a backprop gradient on a probe batch suffices, no protocol change |

**Telemetry bug — fix before any selection ablation reads this field:**
`tc_transformer_trainer_distribute.py:485` logs `jvp_all_perturbations[sorted_indices[-1]]` (the
argmax) but labels it `chosen jvp`, while the actual pick is the coin-flip result `best_idx`
(`:481-483`). The coin flip itself is real and matches `FLUXTUNE_CODE_QA.md` §D2.

---

## 12. Solution space — stands, with reasoning

Each lands behind a named flag, default = old / byte-identical off, per the §8 flag-gate discipline.
Ownership tags per §10.4.

### S-H. Average all P probes instead of selecting one — **SELECTION · best ratio of gain to effort**

```
now:      upload  d_sel * v_sel                (one probe; 9 measurements discarded)
proposed: upload  (1/P) * sum_i d_i * v_i      (all P; same passes, same bytes)
```

**10× improvement in `ρ/cos` at zero additional cost** (§8.4). Takes the system from 5.0× over budget
to 2× under it by itself. Also makes the estimator unbiased, removing the ~3× scale confound that
currently contaminates every guided-vs-random comparison.

- Site: `tc_transformer_trainer_distribute.py:456-487` + the `g_k = d_k·v_k` emit at `:625-631`.
- Flag: `probe_combine: {select | mean}`, default `select`.
- Sanity check: `ρ` drops by `√(E[v∥²]·P) = 5.5×` on the first commit, and `‖G‖` by the same factor.
- Also fixes the `P=1` crash (`sorted_indices[-2]` on a 1-element list) and the RNG-stream mismatch
  that `FLUXTUNE_CODE_QA` §C3 flags as blocking the C1 ablation — `mean` consumes all P draws.

### S-I. Shrink `p` — freeze `pre_classifier` — **MODEL DESIGN · cheapest 2.31× available**

`ρ/cos ∝ p`, and one 768×768 layer is **56.7%** of the trainable dimension (§8.5). Freezing it gives
`p: 1,040,932 → 450,340`, `cos ×1.52`, `ρ/cos ×0.433` — at *negative* cost (less compute, less
communication, less memory).

- Flag: extend `freeze_layers` / add `trainable_scope: {adapters_head | adapters_only}`.
- Gated on **H-G**: does it cost accuracy? Nobody has asked; it is trainable by HF default.
- Generalises to an adapter-rank sweep — for forward-grad, rank is an *algorithmic* parameter.

### S-A. Trust-ratio (relative) server step — **AGGREGATION · highest structural leverage**

```
now:      theta <- theta - eta * G / N_acc
proposed: theta <- theta - rho_star * ||theta_tr|| * G / ||G||
```

Makes ρ an **operator constant** rather than an emergent quantity, removing the `|JVP|` scale from
the update entirely — which is why it works **regardless of how H-B resolves**.

- Site: `FedSgdAggregator.py:322-336` (`_server_update_step`).
- Flag: `server_step_rule: {raw_sgd | trust_ratio}`, default `raw_sgd`.
- Subsumes **S3** — ω can no longer influence step *magnitude*, only relative weighting.
- Sanity check: `‖Δθ‖/‖θ_tr‖` from telemetry equals `rho_star` on every commit.
- **Prerequisite for evaluating S-H, S-I and S-D** (§10.3b), so land it early even though S-H is the
  bigger single win.

### S-B. Anneal ρ\* on a Robbins–Monro schedule — **AGGREGATION · the theory-shaped claim**

Measured ρ is **constant**, so `Σρ² = ∞`. A stochastic-approximation method with a
non-square-summable step sequence **provably cannot converge** — it can only random-walk. This is
*the* convergence condition, and the telemetry measures the exact quantity it is about.

Set `ρ*_t` with `Σρ_t = ∞, Σρ_t² < ∞` (e.g. `ρ*_t = ρ_0/√t`). Composes with S-A.

**The rigorous answer to "why doesn't lowering η fix it":** lowering η scales ρ by a constant, but a
*constant* ρ of any size still has `Σρ² = ∞`. **Smaller η delays divergence; it cannot prevent it.**

### S-C. Dynamic K/C as a closed-loop stability controller — **BOTH**

`selector/dynamic_kc_policy.py` is `LANDED-OFF` and targets `target_iter_per_data_id: 15` — a
load-balancing heuristic with no connection to the estimator. **The control target is wrong.** The
server already logs `‖Δθ‖` and `‖W‖`, so ρ is free; since `ρ ∝ 1/√N`:

```
N_target = N_now * (rho_now / rho_star)^2
```

Closed loop, **measured setpoint**. Genuinely fluxtune-specific: **only the async forward-gradient
path has a free `N` to spend.** Sensor is aggregation-side, actuator selection-side (§10.3a).

### S-D. Widen K, don't lengthen I — **SELECTION**

If a pool increase is still needed after S-H and S-I (it may not be — check the measured ρ/cos
first), buy it with **K, not I**:

| route | wall-clock per commit | commits in 4 h |
|---|---|---|
| more iterations (K=10, `max_iter`↑) | **×the multiplier** | proportionally fewer — strictly worse |
| **wider cohort (K↑, c↑)**, I unchanged | **1×** | unchanged |

**K is device-parallel, I is serial.** In real FL, 50 phones compute simultaneously; each extra
*iteration* is another full round trip. Only the 8-GPU emulation harness pays for K — a strong
argument for running this sweep **in sim**, where parity is established (`simulate_fwdllm.md`).
`c = 30` currently caps K. Larger C/K produces real staleness → H-E and the C3-freshness reversal.

### S-E. Scale-free commit gate, identical to the controller's statistic — **THE SEAM**

Replace `var < var_threshold` with the **split-half cosine** of the pool:

- **Dimensionless** → immune to the 36× drift that kills any absolute threshold (Leg 3).
- It **is** an estimator of `cos(G,g)`, so the gate and the ρ-controller read the *same* measurement
  and the commit rule becomes literally **"commit when `ρ ≤ cos`"**.
- Gives **adaptive N per bin** instead of a static target, and retires plateau + cap from being the
  de-facto policy. Three heuristics collapse into the criterion itself.

### S-F. Restate C1: direction, not magnitude — **BOTH** (see §8.7)

Under S-A the magnitude is discarded by construction. If a selection stage is retained, select on
**curvature (free from the discarded FD sum) or split-half SNR (free from per-sample losses)** — not
on `|d|`, which §8.1–8.2 show is stability-neutral and structure-free.

### S-G. Weight-decay control arm — **AGGREGATION · hygiene, not a contribution**

`λ ≈ ρ²/2` exactly cancels the measured inflation. ~3 lines. Include as a **control** so the stack
has to prove it beats "just add weight decay". If H-B resolves as *genuine* gradient growth, this is
curative rather than cosmetic and the bar it sets goes up.

### Explicitly do NOT do

- **Lower η alone.** Pays 1:1 (§6.3) and does not restore square-summability (S-B).
- **Retune `var_threshold`.** Dimensionally impossible (Leg 3).
- **Retry momentum before ρ is bounded.** `ρ_eff = ρ/(1−β)` — the NaN is arithmetic.
- **Build the least-squares gradient solve.** At `P,N ≪ p` it equals the average up to scale (§8.6-A).
- **Orthogonalise probes, coordinate probes across trainers, or normalise `v`.** All no-ops at
  `p = 10⁶` (§8.6). Stated so nobody re-derives them.
- **Chase momentum-in-the-probe-distribution.** Computed negative: ~5% variance reduction (§8.6-C).
- **Invest further in ω-direction / inverse-var.** Two orders of magnitude below the problem.
  (ω-*freshness* is different — on the critical path after S-D.)
- **Shrink `h` to "fix" the FD.** `h‖v‖ = 10.2` is 50% of `‖θ_tr‖` at init, but shrinking `h` runs into
  fp16 catastrophic cancellation (QA §4.2: 1–2 significant figures at h = 0.01). **`h` is pinned
  between truncation and cancellation** — a real, publishable tension. And H-C says the probe gets
  *relatively smaller* as the run proceeds, so the FD is not obviously the villain.

---

## 13. Do this first — one probe answers three questions

`cos(G,g) ≈ 0.0231` is the **only** load-bearing number still resting on an assumption (isotropy);
its `E[v∥²]` input is now measured (§6.1). Every sizing in §12 depends on it.

`FLUXTUNE_CODE_QA.md` §E1 says this needs `v_k` uploaded server-side. **For this quantity it does
not.** `G` is already server-side; you only need *some* `g` to compare against, and a backprop
gradient on a fixed held-out probe batch suffices.

- ~20 lines beside the existing `server_update` telemetry (`FedSgdAggregator.py:338-356`, which
  already carries `‖Δ‖` and `‖W‖`). **No protocol change.**
- Log per commit: `cos(G, g_probe)`, `‖G‖/‖g_probe‖`, `‖g_probe‖`, and `ρ`.

**It settles three things at once:**

1. `cos(G,g)` → every §12 sizing becomes MEASURED.
2. `‖g_probe‖` vs `rms|d|` → **resolves H-B** (genuine gradient growth vs FD artifact), deciding
   whether weight decay is curative and whether the `v`-normalisation thread is worth opening.
3. `‖G‖/‖g_probe‖` → validates the `√(p/N)` noise model; if it disagrees, isotropy is wrong and §6
   needs revisiting **before** anything is built.

**Decision rule:** if measured `cos ≈ 0.023`, everything in §12 holds as written. If materially
higher, requirements relax as `∝ 1/cos²` — recompute before building.

---

## 14. Reproducing every number

All from `lib/python/examples/fwdllm/experiments/`. Nothing needs a GPU except the model probe.

```bash
RUN=run_20260804_043301_fluxtune_n100_smoke_syn_0_real
```

**Accuracy / stat_utility / staleness** — `agg_eval` and `agg_round` events in
`$RUN/telemetry/aggregator_*.jsonl`.

**Leg 1 (random walk)** — `server_update` events; compare `Σ (‖W_{t+1}‖² − ‖W_t‖²)` against
`Σ ‖Δθ_t‖²` (`weight_norm`, `update_delta_norm`). Expect 1.032.

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
# format: "All JVPs sorted by magnitude: [tensor(...), ...] and chosen jvp: X
#          for trainer : T for model version: R data-id: D. iteration: I"
# GOTCHAS: "model version" here is actually the ROUND.
#          "chosen jvp" is the argmax, NOT the coin-flip winner (telemetry bug, §11).
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

---

## 15. Target end-state design

```
MODEL        shrink p: freeze pre_classifier (56.7% of the probe dimension)          [S-I]
             for forward-grad, PEFT rank IS a gradient-quality parameter (cos ~ 1/sqrt(p))

TRAINER      draw P perturbations, compute all P JVPs                                [C1 budget kept]
             upload (1/P) * sum_i d_i * v_i     <- ASSIMILATE ALL, do not select     [S-H]
             (if a selection stage is kept, select on curvature or split-half SNR,
              never on |d| -- magnitude selection is stability-neutral)              [S-F]

GATE         per bin, accumulate updates; compute SPLIT-HALF COSINE of the pool      [S-E]
             commit when  cos_measured >= rho_star    <- the criterion, enforced online
             (replaces: absolute var threshold + plateau + iteration cap)

STEP         theta <- theta - rho_star * ||theta_tr|| * G/||G||                      [S-A]
             rho_star annealed on a Robbins-Monro schedule                           [S-B]

CONTROL      measure rho and cos every commit; actuate N_target = N*(rho/rho*)^2     [S-C]
             spend N on K (device-parallel) before I (serial); c scales with K       [S-D]

WEIGHTS      omega-freshness becomes load-bearing once C/K is large                  [C3-freshness]
             omega-direction / inverse-var stay parked
```

Read top to bottom: the model chooses how many dimensions must be probed; the trainer pools every
probe it pays for; the gate decides when enough probes have been pooled to aim reliably; the step
takes a bounded relative move in that direction; the controller keeps pool size matched to step size.
**Every constant in the shipped system that had dimensions is replaced by a dimensionless one.**

---

## 16. Suggested next actions, in order

1. **Land the `cos(G,g)` probe** (§13). Everything is sized off it, it resolves H-B and H-C for free,
   no protocol change.
2. **Land S-H (average all P probes)** — 10× for ~10 lines, and it fixes the `P=1` crash and RNG
   mismatch that currently block the C1 ablation. Flag `probe_combine`, default `select`.
3. **Land S-A (trust-ratio step) behind a flag**, default off — it is what makes S-H's and S-D's
   measurements interpretable (§10.3b).
4. **Test S-I / H-G**: freeze `pre_classifier`, 4 h run, compare peak accuracy. 2.31× for free if it
   holds.
5. **Re-measure ρ/cos with S-H + S-I on.** If it is already under budget, **S-D's pool increase may
   be unnecessary** — spend the margin on faster commits or a larger `ρ*` instead. That choice is
   exactly what S-C automates.
6. **Land S-E (split-half cosine gate)** — also supplies S-C's measured setpoint.
7. **Then** S-B (ρ annealing), S-C (closed loop), the S-D sweep with the staleness histogram (H-E →
   C3-freshness), and the free curvature probe (H-H).
8. **In parallel, settle H-F**: 4 h `fwdllm` / `fwdllm_plus` runs with `server_update_audit`, ρ
   computed identically. This decides contribution vs. hygiene, which changes the paper's claim
   structure — do not leave it until the end.

> **Standing rule for every run from here: 4 h minimum, or the result is uninformative.** The 2 h
> runs terminate at 0.85 norm doublings, exactly at the peak. Any A/B shorter than ~3.2 h will report
> a win for a configuration that diverges.
