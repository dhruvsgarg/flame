# FluxTune — the "peak then plunge" divergence: root cause, criterion, and solution space

**Status:** analysis complete, no code changed. Written 2026-08-06 on `dg/fluxtune_expts_sim_init`.
**Purpose:** hand off a finished diagnosis so the next session can go straight to building.

Every claim is tagged:

- **MEASURED** — read out of telemetry / logs / the real model builder. Reproduction command in §9.
- **DERIVED** — arithmetic on MEASURED inputs, no modelling assumption.
- **ANALYSIS** — rests on a modelling assumption (stated inline). Exactly one number in this doc is
  ANALYSIS and it is flagged every time it appears: `cos(G,g) ≈ 0.024`.

**The one-line summary.** FluxTune's committed update is a fixed-*relative*-size step in a random
direction, so the adapter weight norm inflates **geometrically** while the useful signal accumulates
only **linearly**. Divergence is not a tuning accident — it is structurally guaranteed at the shipped
operating point, present from commit 1, and the three hours of rising accuracy are just a race the
noise wins. The relative step is **4.8× larger than the stability budget** allows.

---

## 1. Scope — the runs this is based on

| run | mode | duration | commits | outcome |
|---|---|---|---|---|
| `run_20260804_003042_fluxtune_n100_smoke_syn_0_real` | real | 3.93 h | 189 | peak acc 0.846 @ r1/did106 → **0.250 / mcc 0.000 / loss 2.37** @ r2/did38 |
| `run_20260804_043301_fluxtune_n100_smoke_syn_0_real` | real | 3.93 h | 185 | peak acc 0.853 @ r1/did122 → **0.252 / loss 1.54** @ r2/did34 |
| `run_20260805_{110016,130231,150446}_fluxtune_..._real` | real | 1.86–1.88 h | ~95 | stop at did 92–94 — **at the peak, before the collapse** |

**Why it looked intermittent:** the 2 h runs terminate on `max_runtime_s` right at the accuracy peak.
The collapse needs ~3.2 h of wall clock to show up. Only the two 4 h runs
(`max_runtime_s=14400`) contain it. Both contain it, and they agree quantitatively — this is
deterministic behaviour, not a bad seed.

**Not a useful control:** the `fwdllm` baseline runs from the same period
(`run_2026080[45]*fwdllm_n100_smoke_syn_0_real`) only reach `data_id=38` in 1.84 h and never leave
acc 0.25 — the sync path is too slow at this horizon to compare against. Its `|jvp|` is flat
(rms 3.1–4.0 across 3,101 merges) but that is because its weights barely move, not evidence of
stability. **A fair baseline comparison does not exist yet at the horizon where fluxtune fails.**

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
       scale=0.4, a_exp=0.25, b_exp=0.1
```

Two facts about the estimator that matter later, both **MEASURED**:

- `v` is a **raw Gaussian draw**, never normalized (`tc_transformer_trainer_distribute.py:416`).
  So `‖v‖ = √p ≈ 1020`, and the finite-difference probe displacement is
  **`h·‖v‖ = 0.01 · 1020.3 = 10.203`**.
- `‖θ_trainable‖` at init = **20.356** (measured directly from `build_model(4, 192)` in
  `scripts/profile_jvp_opt.py`, which is the production construction path).

> **The FD probe displaces the entire trainable parameter vector by 50% of its own norm.**
> This is a chord across half the parameter space, not a directional derivative. And its scale is
> an *accident of the parameter count* (`√p`), not a design choice.

---

## 3. What failure looks like

Test-set trajectory, `run_20260804_003042` (**MEASURED**, `agg_eval` telemetry, eval every 2 commits):

```
 0.00h r1 did=  0   acc 0.398  loss 1.380     <- init, ln(4) = 1.386
 1.38h r1 did= 70   acc 0.816  loss 0.993
 2.13h r1 did=106   acc 0.846  loss 0.554     <- PEAK
 2.95h r1 did=146   acc 0.738  loss 0.692     <- degradation starts inside round 1
 3.04h r2 did=  0   acc 0.818  loss 0.547
 3.34h r2 did= 14   acc 0.527  loss 1.464     <- knee
 3.56h r2 did= 24   acc 0.250  loss 1.688     <- single-class collapse
 3.87h r2 did= 38   acc 0.250  mcc 0.000  loss 2.371
```

Three things rule out the obvious explanations:

1. **Final loss 2.37 > ln(4) = 1.386.** The model is *confidently wrong*, not uncertain. Weights
   have blown up; this is not forgetting.
2. **Train loss diverges too.** `stat_utility` (Oort utility = `8·rms(batch loss)`, so
   `loss ≈ U/8`) goes **10.84 → 5.31 (did 120) → 13.74 (r2 did 35)** — i.e. train loss
   1.355 → 0.66 → 1.72. Train and test move together. **Not overfitting.**
3. **The degradation starts inside round 1** (did 138–148), before the round boundary. The round
   boundary is not the trigger. (Consistent with F10, which already REFUTED the staleness-reset bug.)

> **This supersedes the framing in `fluxtune_contributions.md` §8.** That section describes the
> failure as *oscillation with recurring single-class collapses* around a non-converging global
> model. At the 4 h horizon the actual behaviour is a clean **monotone rise, then a monotone,
> irreversible divergence**. F4/F5's position-locked collapses are an early-training symptom of the
> same underlying noise; they are not the mechanism, and H3 (bin-order permutation) does not address
> the mechanism at all.

---

## 4. Root cause — three measured legs

### Leg 1 — the committed update is a pure random walk. **MEASURED.**

If a step `Δθ` were doing useful descent it would have a component along `θ`. Test: does
`‖W‖²` grow by exactly `‖Δθ‖²` (which is what happens iff `Δθ ⊥ θ`)?

```
sum over all 185/189 commits of observed d(||W||^2)  /  sum of ||dTheta||^2
      run_003042 : 1.0326
      run_043301 : 1.0323
```

**100% of the step energy goes into inflating the parameter norm, to within 3%.** Over 185 commits
there is no measurable net descent component. This is F8 ("undamped direct SGD → random walk") with
a number attached, and it is independent of any assumption — it uses only `weight_norm` and
`update_delta_norm`, both already logged by `server_update_audit`.

### Leg 2 — the *relative* step is pinned, so the norm inflates geometrically. **MEASURED + DERIVED.**

Reconstruct `‖θ_tr,t‖` from telemetry using only *differences* of `‖W‖` anchored at the measured
init (this cancels the frozen-norm constant; both runs independently imply `‖θ_fz‖ = 415.929`,
matching the directly measured 416.213 to 0.07% — a good consistency check):

```
rho = ||dTheta|| / ||theta_tr||        first-20 commits   mid-run   last-20
                       run_003042            0.113          0.118     0.163
                       run_043301            0.116          0.117     0.159

||theta_tr||:  20.36 (init)  ->  86.16 (commit 189)     = 4.2x
geometric-walk prediction  ||theta_t+1|| = ||theta_t||*sqrt(1+rho_t^2):
      predicts 83.98 vs 86.16 observed   (-2.5%)   [run_003042]
      predicts 81.01 vs 82.85 observed   (-2.2%)   [run_043301]
norm doubling time: predicted 109 commits, observed 114   [run_003042]
                    predicted 104 commits, observed 116   [run_043301]
```

A two-parameter model — *"every step is a random direction of relative size 0.115"* — predicts the
final adapter norm after 189 commits to within 2.5%. **There is nothing else happening.**

**Why ρ is constant** (this is the positive feedback loop): `|JVP| ∝ ‖θ_tr‖`. Parsing all 34,447
selection events (**MEASURED**):

```
rms|d| over the run:  3.59 (did 0)  ->  21.5 (r2 did 35)   = 6.0x
||theta_tr||       : 13.6           ->  81.4               = 6.0x
```

So: noise inflates the norm → a bigger norm produces a bigger `|JVP|` → a bigger step → more
inflation. **Multiplicative, exponential, and unstable at initialization** (ρ₀ = 0.113 already —
this is not something that develops mid-run).

> **This retro-explains the S1 NaN.** Heavy-ball at β = 0.9 multiplies the effective relative step
> by `1/(1−β) = 10` → ρ_eff = 1.13 → the norm doubles *every step*. S1 was not the wrong idea;
> it was applied to an already multiplicatively-unstable process. Do not retry momentum before ρ is
> controlled — and once ρ *is* controlled, momentum becomes safe and cheap to test.

### Leg 3 — the variance gate is dimensionally wrong and 100% dead. **MEASURED.**

From 3,441 `[IterProgress]` log lines covering all 186 bins:

```
commit reasons:  natural (var < 0.3) = 0     plateau = 105     cap(max_iter=20) = 81
bins that EVER reached var < 0.3 at any iteration:  0 of 186

achievable variance floor (median per-bin minimum), by 20-bin block:
   bins   0- 20 : 0.415
   bins  60- 80 : 0.674
   bins 100-120 : 1.462
   bins 140-160 : 3.911
   bins 180-186 : 14.97          <- 36x drift over the run
```

**36× is exactly (6.0×)².** Variance is a second moment of `d`, and `d ∝ ‖θ_tr‖`, so the gate's own
measurement scale grows as `‖θ‖²` while `var_threshold` stays at 0.3.

> **S2 ("recalibrate `var_threshold`") is structurally unfixable.** No constant threshold can be
> correct for more than an instant against a statistic that drifts 36× within one run. The problem
> is not the value; it is that the quantity has dimensions.
>
> This also means the two escape hatches (plateau, cap) are *the entire commit policy* — Opt-2 is
> not a safety net, it is the mechanism. And `max_iterations_per_data_id=20`, a wall-clock
> convenience knob, is the de-facto variance controller.

### The single root cause

**Nothing in the pipeline is scale-free.** The estimator, the step, and the commit gate all inflate
together with `‖θ_tr‖`, so there is no quantity anywhere in the system that a fixed constant can be
compared against. Fix that one property and all three legs resolve together.

---

## 5. The stability criterion

### 5.1 Derivation

Per commit, `G = (1/N) Σ_k ω_k d_k v_k` with `v_k ~ N(0, I_p)`, `d_k = ⟨g, v_k⟩`, and `N` = number of
individual trainer updates accumulated (**MEASURED: N ≈ 185** = K=10 × 18.5 iterations/bin).

Writing `v = v_∥ ĝ + v_⊥`:

```
E[d·v | selected]   = E[v_par^2] * g                     unbiased in DIRECTION, inflated in SCALE
||G||               ~ omega_bar * ||g|| * sqrt(E[v_par^2] * p / N)      (p >> N, noise dominates)
cos(G, g)           = sqrt( E[v_par^2] * N / p )
```

With `E[v∥²] ≈ 3.2` (order statistic for a coin flip between the top-2 of 10 half-normals),
`N = 185`, `p = 1,040,932`:

```
cos(G, g)  ~  sqrt(3.2 * 185 / 1040932)  =  0.0239        <-- *** ANALYSIS *** (assumes isotropy)
```

Now compose per commit. Leg 1 established `Δθ ⊥ θ`, so:

- norm inflation: `‖θ‖² ← ‖θ‖²(1 + ρ²)`
- useful displacement: `ρ · cos(G,g) · ‖θ‖`

After `T` commits, signal accumulates **linearly** (`T·ρ·cos`) and noise as a **random walk**
(`√T·ρ`). Signal dominates only after `T ≳ 1/cos²`. But the norm must survive that long, which needs
`ρ²·T ≲ 1`. Combining:

> ### ρ ≤ cos(G, g)
> **The relative step must not exceed the fraction of it that is actually aligned with the gradient.**

### 5.2 Where the shipped config sits

```
rho (MEASURED)              = 0.115
cos(G,g) (ANALYSIS)         = 0.0239
rho / cos                   = 4.8x  OVER BUDGET
1/cos^2  = 1756 commits to signal dominance;  norm doubles every 109 commits
                                              -> divergence is certain, ~16x too slow to escape
```

At 186 commits / 4 h, signal dominance would need ~37 h, by which time the norm has doubled 23 times.

### 5.3 The scaling table — this is what ranks every fix

Because `ρ ∝ 1/√N` and `cos ∝ √N`:

| lever | effect on `ρ/cos` (stability) | effect on `ρ·cos` (per-commit progress) |
|---|---|---|
| **N = K × iterations** | **∝ 1/N** — the only linear lever | **invariant** — costs nothing |
| `η` learning rate | ∝ η | ∝ η — **pays 1:1 for every unit of stability** |
| selection gain `E[v∥²]` (C1) | **invariant** — both terms ∝ √E | ∝ E — a 3.2× progress multiplier |
| step normalization | sets ρ to an operator constant | decoupled |

Per unit of *compute* (T commits × N samples): progress ∝ `C/N`, total norm inflation ∝ `C/N²`, so
the ratio improves ∝ `N`. **Larger N is strictly better for stability per FLOP, and slower in
absolute wall-clock progress.** That is the honest trade and it is the whole argument for
iteration-control.

---

## 6. Corrections and de-prioritizations this forces

| Existing claim | Verdict |
|---|---|
| `fluxtune_contributions.md` §8 headline: "never converges — it *oscillates*" | **Superseded.** At 4 h it is a monotone rise then monotone divergence. Oscillation is an early symptom. |
| S1 momentum "REFUTED as-designed" | **Re-framed.** Correctly refuted *at ρ = 0.115*; ρ_eff = 1.13 explains the NaN exactly. Becomes testable once ρ is bounded. |
| S2 "variance-gate recalibration" | **Structurally unfixable as scoped** (Leg 3). Needs a dimensionless statistic, not a better constant. |
| S3 "aggregation-rate tempering / cap ω ≤ 1" | **Correct but ~2 orders of magnitude too small.** ω ∈ [0.702, 0.865] measured; the problem is 4.8×. |
| H3 bin-order permutation, H1 shuffle, H2 bin size | **Do not address the mechanism.** H2 (bin size) touches N indirectly and is the only one worth revisiting, via §7. |
| C3-direction (alignment gate) | **Park.** ≤0.6% weight perturbation (QA §B1) against a 4.8× problem. Unmeasurable at this noise floor. |
| C3-freshness (staleness) | **Park.** ≤8% on 15.6% of updates, staleness never exceeded 1 (QA §A4). |
| C1 "guided selection improves accuracy" | **Needs restating** — see §7. It is stability-*neutral*, which is a stronger and more defensible claim than the current one. |

**Telemetry bug found, fix before any selection ablation:**
`tc_transformer_trainer_distribute.py:485` logs `jvp_all_perturbations[sorted_indices[-1]]` (the
argmax) but labels it `chosen jvp`, while the actual pick is the coin-flip result `best_idx`
(`:481-483`). The log line misreports the selected candidate. The coin flip itself is real and
matches `FLUXTUNE_CODE_QA.md` §D2.

---

## 7. Solution space — stands, with reasoning

Ordered by leverage. Each should land behind a named flag, default = old / byte-identical off, per
the §8 flag-gate discipline.

### S-A. Trust-ratio (relative) server step — **highest leverage, smallest diff**

```
now:      theta <- theta - eta * G / N_acc
proposed: theta <- theta - rho_star * ||theta_tr|| * G / ||G||
```

Makes ρ an **operator constant** instead of an emergent quantity. Removes the `|JVP|` scale from the
update entirely, which is what makes the whole system invariant to the 6× estimator drift. Kills the
geometric runaway structurally rather than by tuning.

- Site: `FedSgdAggregator.py:322-336` (`_server_update_step`).
- Flag: `server_step_rule: {raw_sgd | trust_ratio}`, default `raw_sgd`.
- Subsumes S3 (the rate cap becomes unnecessary — ω can no longer influence step magnitude,
  only relative weighting).
- Sanity check: `‖Δθ‖/‖θ_tr‖` from `server_update` telemetry equals `rho_star` every commit.

### S-B. Decay ρ\* on a Robbins–Monro schedule — **the theory-shaped claim**

Measured ρ is **constant**, therefore `Σρ² = ∞`. A stochastic-approximation method with a
non-square-summable step sequence **provably cannot converge** — it can only random-walk. That is
not an engineering observation, it is the convergence condition, and the telemetry measures the
exact quantity the condition is about.

Set `ρ*_t` with `Σρ_t = ∞, Σρ_t² < ∞` (e.g. `ρ*_t = ρ_0/√t`). Composes with S-A (S-A is what makes
ρ controllable at all; S-B is what makes it converge).

**This is the strongest thing in the analysis and the most defensible contribution.** It is also the
honest answer to "why doesn't lowering η fix it": lowering η lowers ρ by a constant factor, but a
*constant* ρ of any size still has `Σρ² = ∞`. Smaller η delays divergence; it does not prevent it.

### S-C. Dynamic K/C as a closed-loop stability controller — **the selection-side contribution**

`selector/dynamic_kc_policy.py` is `LANDED-OFF` and targets `target_iter_per_data_id: 15` — a
load-balancing heuristic with no connection to the estimator. **The control target is wrong.**

The server already logs `‖Δθ‖` and `‖W‖` every commit, so ρ is free. Since `ρ ∝ 1/√N`:

```
N_target = N_now * (rho_now / rho_star)^2
```

One line, closed loop, measured setpoint. This converts dynamic-KC from a scheduling heuristic into
a **stability controller**, and it is genuinely fluxtune-specific: only the async forward-gradient
path has a free `N` to spend (a sync baseline's N is fixed by its cohort).

### S-D. Widen K, do not lengthen iterations — **the concrete sizing**

Closing the 4.8× gap needs `N ≈ 890` vs 185 today.

| route | config | wall-clock per commit | commits in 4 h |
|---|---|---|---|
| more iterations | K=10, max_iter≈90 | **4.8×** | ~39 — strictly worse |
| **wider cohort** | **K=50, c=100** (full population), iterations unchanged | **1×** | ~185 |

**K parallelizes across devices; iterations serialize.** In the FL model and in the simulator, K is
free — only the 8-GPU emulation harness pays for it. That is a strong argument for running this
ablation **in sim**, where real↔sim parity is already established (`simulate_fwdllm.md`).

Note `c = 30` currently caps K; c must rise to ~100 with K. Also revisit H2 here — bin size is the
other multiplier on the effective sample count, and the M1 sweep now has a criterion to optimize
against (ρ/cos) rather than raw accuracy.

### S-E. Make the commit gate scale-free — and identical to the controller's statistic

Replace `var < var_threshold` with the **split-half cosine** of the pool.

- It is **dimensionless** → immune to the 36× drift that kills the absolute threshold (Leg 3).
- It **is** an estimator of `cos(G,g)` — so the commit gate and the ρ-controller (S-C) read the
  *same* measurement, and the commit rule becomes literally "commit when `ρ ≤ cos`".

That is a genuine unification rather than three bolted-on heuristics (gate + plateau + cap), and it
retires the plateau/cap escape hatches from being the de-facto policy.

### S-F. Restate C1: selection chooses the *direction*, not the *magnitude*

Guided `|JVP|` selection multiplies `cos` by 1.8× **and** ρ by 1.8× — the measured selection gain
`max|d|/rms|d| = 1.81 ± 0.03` is flat across the entire run (**MEASURED**; exactly the order
statistic of 10 iid Gaussians, i.e. selection is finding scale, not structure).

So as shipped, C1 is a real **3.2× per-commit progress multiplier** *and* a 3.2× accelerant of the
blow-up. Under S-A the magnitude is discarded, and C1 keeps the 1.8× alignment gain with **zero** ρ
inflation.

**The claim to make:** *"guided selection is unbiased in direction and inflated ~3× in norm; we take
the direction and normalize away the norm."* This is stronger than the current framing because it
survives the obvious reviewer objection that guided-vs-random is an unmatched-learning-rate
comparison (`FLUXTUNE_CODE_QA.md` §E3.1).

### S-G. Cheap hygiene control arm (not a contribution — a baseline)

Weight decay `λ` on the trainable params chosen so `λ ≈ ρ²/2` exactly cancels the measured
inflation. ~3 lines. Include it as a control so the S-A/S-B/S-C stack has to prove it beats
"just add weight decay". Reported as a fix, not a win.

### Explicitly do NOT do

- **Lower η alone.** Pays 1:1 (§5.3) and does not restore square-summability (§S-B).
- **Retune `var_threshold`.** Dimensionally impossible (Leg 3).
- **Retry momentum before ρ is bounded.** ρ_eff = ρ/(1−β) — the NaN is arithmetic, not bad luck.
- **Invest further in ω** (freshness / alignment / inverse-var). Two orders of magnitude below the
  problem until ρ is controlled.
- **Reduce `h` to fix the FD.** `h·‖v‖ = 10.2` is indeed 50% of `‖θ_tr‖`, but shrinking h runs into
  fp16 catastrophic cancellation (QA §4.2: only 1–2 significant figures survive at h = 0.01). h is
  **pinned between truncation and cancellation** — a real, quantifiable, publishable tension. The
  principled reformulation is to **normalize `v` to unit norm and define the probe as a fixed
  fraction of `‖θ_tr‖`**, which removes `√p` from the estimator entirely — but note this changes the
  estimator (`E[d·v] = g/p` for unit `v`) and re-baselines the trajectory, so it is a separate
  research thread, not a bug fix.

---

## 8. Do this first — measure `cos(G,g)` directly

`cos(G,g) ≈ 0.0239` is the **only** load-bearing number that is ANALYSIS, and every fix sizing in §7
depends on it (`N ≈ 890`, `ρ/cos = 4.8`, `ρ* ≤ 0.024`).

`FLUXTUNE_CODE_QA.md` §E1 says this measurement requires uploading `v_k` server-side. **For this
particular quantity it does not.** `G` is already server-side; you only need *some* `g` to compare
against, and a backprop gradient on a fixed held-out probe batch is sufficient to get the alignment.

- ~20 lines next to the existing `server_update` telemetry (`FedSgdAggregator.py:338-356`, which
  already carries `‖Δ‖` and `‖W‖`).
- **No protocol change**, unlike the full Group-E measurement.
- Log per commit: `cos(G, g_probe)`, `‖G‖/‖g_probe‖`, and `ρ`. Those three plus `N` close the loop.

**Decision rule:** if measured `cos ≈ 0.024`, the `N ≈ 890` / `K = 50, c = 100` sizing in §7-S-D
holds as written. If it comes back materially higher, every requirement relaxes proportionally
(`N_required ∝ 1/cos²`) — recompute before building.

---

## 9. Reproducing every number

All from `lib/python/examples/fwdllm/experiments/`. Nothing below needs a GPU except the model probe.

```bash
RUN=run_20260804_043301_fluxtune_n100_smoke_syn_0_real
```

**Accuracy trajectory / stat_utility / staleness** — `agg_eval` and `agg_round` events in
`$RUN/telemetry/aggregator_*.jsonl`.

**Leg 1 (random walk)** — `server_update` events; compare `Σ (‖W_{t+1}‖² − ‖W_t‖²)` against
`Σ ‖Δθ_t‖²` (`weight_norm`, `update_delta_norm`). Expect ratio 1.032.

**Leg 2 (relative step)** — same events, anchored at the measured init trainable norm:

```python
T0 = 20.356                                   # ||theta_tr|| at init, from build_model
base = W[0]**2 - (T0**2 + dW[0]**2)           # ||theta_frozen||^2; cancels out of all deltas
theta_tr_t = sqrt(W[t]**2 - base)
rho_t      = dW[t] / theta_tr_t               # expect ~0.115, flat
pred       = T0 * prod(sqrt(1 + rho_t**2))    # expect within 2.5% of theta_tr_final
```

**`|JVP|` growth + selection gain** — the trainer log carries all 10 candidates per selection:

```bash
grep 'All JVPs sorted by magnitude' $RUN/*trainers.log   # 34,447 lines
# parse: "All JVPs sorted by magnitude: [tensor(...), ...] and chosen jvp: X
#         for trainer : T for model version: R data-id: D. iteration: I"
# NOTE: "model version" in this line is actually the ROUND. "chosen jvp" is the argmax,
#       not the coin-flip winner (see the telemetry bug in §6).
# bucket by (round, data_id): rms|d| 3.59 -> 21.5 ; mean(max|d|)/rms|d| = 1.81 flat
```

**Leg 3 (variance gate)** — the aggregator log:

```bash
grep -o '\[IterProgress\] data_id=.* force_commit_planned=[A-Za-z]*' $RUN/*aggregator.log
# 3,441 lines / 186 bins. Split into bins on data_id change; the last row of each bin is the commit.
# reason = CAP if iter>=19 else natural if var<0.3 else plateau  ->  0 / 105 / 81
```

**Model constants** (needs the `test_fwdllm` env, ~1 min):

```bash
cd /home/dgarg39/flame/lib/python
/coc/scratch/dgarg/miniconda3/envs/test_fwdllm/bin/python -c "
import sys, torch, math; sys.path.insert(0,'.')
from examples.fwdllm.scripts.profile_jvp_opt import build_model
m = build_model(4, 192)
tr = [p for _,p in m.named_parameters() if p.requires_grad]
fz = [p for _,p in m.named_parameters() if not p.requires_grad]
p  = sum(x.numel() for x in tr)
tn = torch.sqrt(sum((x.detach().float()**2).sum() for x in tr)).item()
fn = torch.sqrt(sum((x.detach().float()**2).sum() for x in fz)).item()
print(p, tn, fn, 0.01*math.sqrt(p), 0.01*math.sqrt(p)/tn)"
# -> 1040932  20.3562  416.2133  10.203  0.501
```

---

## 10. Suggested next actions, in order

1. **Land the `cos(G,g)` probe** (§8). Everything else is sized off it. No protocol change.
2. **Land S-A (trust-ratio step) behind a flag**, default off. A/B at α=1, N=100, **4 h minimum** —
   a 2 h run cannot see the failure and will show a false pass.
3. **Land S-E (split-half cosine gate)**, which also gives S-C its measured setpoint.
4. **Run the S-D sizing sweep in sim** (K=10/20/50 at c=30/60/100), using ρ/cos as the objective,
   not final accuracy — the accuracy replicate floor is 2.3 points (`parity_floors/fluxtune.yaml`)
   and will not resolve this.
5. **Then** S-B (ρ decay schedule), S-C (closed loop), and only then revisit S-F/momentum/ω.

**Standing rule for every run from here:** **4 h minimum, or the result is uninformative.** The
2 h runs terminate exactly at the peak. Any A/B shorter than ~3.2 h will report a win for a
configuration that diverges.
