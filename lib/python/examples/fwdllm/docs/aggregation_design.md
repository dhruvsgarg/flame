# FluxTune aggregation (C3) — design starter

**What this is.** A starter design doc for FluxTune's **intelligent (gradient-aware) aggregation**
contribution (C3): (1) the async-FL aggregation regime *today* — FedBuff → FeLiX → current FluxTune, with
code anchors; (2) the **hypothesis** for why the current rule is probably wrong for FluxTune; (3) the
**design space** for weighing updates. Seed for the C3 investigation in
[`EXPERIMENTS.md`](../EXPERIMENTS.md) §4 and [`EXPTS_CHARTER.md`](../EXPTS_CHARTER.md) §2d; companion
efficiency contribution is [`docs/dynamic_kc_design.md`](dynamic_kc_design.md) (C2).

**Status:** design only — nothing here is implemented; current FluxTune uses the FeLiX rule (below).
Any change here is **baseline-affecting** → operator sign-off, flag-gated, byte-identical off, and must
preserve real↔sim parity ([`simulate_fwdllm.md`](../simulate_fwdllm.md)).

---

## 1. The regime today

All three run through the **FedBuff optimizer** ([`flame/optimizer/fedbuff.py`](../../../flame/optimizer/fedbuff.py)).
An update is a **delta** buffered until `agg_goal` arrive; each is multiplied by a **scalar rate**, summed, then
scaled by the server LR and added to base weights (`do()` → `aggregate_fn(tres, rate)` → `scale_add_agg_weights`
at `fedbuff.py:181-228`). The three differ only in **how `rate` is computed.**

| | rate formula | signals used | lineage |
|---|---|---|---|
| **FedBuff** | `rate = 1 / √(1 + Δv)`,  `Δv = version − trainer_version` | **staleness only** (round-based) | `agg_rate_conf.type = "old"` (`fedbuff.py:180`) |
| **FeLiX** | `rate = scale·α(Δv) + (1−scale)·β(u)` | staleness **+** statistical utility | `type = "new"` (`fedbuff.py:183-200`); async_cifar10 / REFL |
| **FluxTune (now)** | **same as FeLiX** + a **variance commit-gate** | staleness + utility, gated on grad variance | FeLiX rate + `var < var_threshold` gate |

Where, for the FeLiX/FluxTune "new" rate (`fedbuff.py:91-139`):
- `α(Δv) = 1 / (1 + Δv)^a_exp`  — **staleness decay** (polynomial; `alpha_exponential` also available).
- `β(u) = 1 − 1/(1+u)^b_exp + 0.5`  — **utility upshift**, increasing in `u`.
- `u = tres.stat_utility` = the **Oort statistical utility** (`I_m` ≈ the client's average training loss
  in its last engaged round — `flame/selector/feddance.py:7`). So β rewards high-loss (informative) clients.
- FluxTune config: `scale=0.4, a_exp=0.25, b_exp=0.1` (`_metadata/baselines.yaml:457-461`);
  `stalenessPolicy=none` (down-weight, never *reject*, `baselines.yaml:473`).

**The variance gate (FluxTune only).** Before an update is eligible, `FedSgdAggregator.aggregate()` computes
a batch of gradient statistics and commits the data-bin only when variance is under threshold
(`aggregator/FedSgdAggregator.py:194-202, 132`):
- `calculate_var(grads)`, `calculate_real_var(jvps)`, `calculate_snr(jvps)`,
  `calculate_snr_gradients(grads)`, `calculate_cv(grads)`.
- commit iff `self.var < self.var_threshold` (`var_threshold`: distilbert default **0.1**, overridden to
  **0.3** for the baseline; `FedSgdAggregator.py:73-89`, `baselines.yaml:475`).

**Key structural fact:** these rich per-update **gradient statistics are computed and then discarded for
weighting** — they gate whether the bin commits, but the *rate* that weights each update ignores them.

---

## 2. Why this is probably the wrong rule for FluxTune (the hypothesis)

Observed in N=100 smoke runs (E3/E4): FluxTune spends ~5–6× the forward-pass compute of the oracular sync
baseline for *less* Δloss per unit — its async concurrency does more *unproductive* work. The aggregation rule
is one suspect. Three reasons it likely under-differentiates FluxTune's updates:

**H1 — Weights ≠ gradients.** FedBuff/FeLiX target **weight-averaging** async FL: the buffered delta is a
*weight update*, so scalar down-weighting is sensible. FluxTune's update is a **forward-mode gradient estimate**
(a JVP-scaled direction). A **scalar** rate changes *magnitude*, not *direction quality*: two opposite-pointing
estimates are still **averaged**, not filtered — a wasteful/harmful combination scalar weighting can't express.
Gradients want a *direction-aware* combiner, not a magnitude knob.

**H2 — Staleness doesn't move fast enough to differentiate.** `Δv = agg_model_version − trainer_version`, and
`_model_version` advances **per data-bin completion**, itself **gated on the variance threshold**
(`fwdllm_aggregator.py:1422`, `fedbuff.py:194`). So version advances **slowly and in lockstep**, concurrent
in-flight updates carry **near-identical `Δv`** → `α(Δv) ≈ constant` across the buffer → the **staleness axis
adds almost no differentiation**. The rate collapses onto `β(u)` (Oort-loss) alone — one coarse, quantized
signal. Net: the "tradeoff" is mostly one-dimensional, and that dimension is weak.

**H3 — The best signals are computed but unused.** Each update's gradient/JVP variance/SNR/CV already exist
(§1) but only drive the binary gate. They are exactly *how trustworthy* an estimate is — the natural weighting
signal for noisy gradient estimates — and they're thrown away.

---

## 3. Design space — how to actually weigh gradient updates

Think along five axes; a concrete scheme picks one option per axis.

**Axis A — what signal(s) to weight by.**
- **Staleness** — but on *which clock*? (see Axis D). Current: `Δmodel_version` (coarse).
- **Statistical utility** — Oort `I_m` (now), or test-loss delta, or per-update loss reduction.
- **Gradient magnitude** — `|JVP|` (FluxTune already selects the max-|JVP| direction in C1; magnitude is a
  proxy for informative descent).
- **Gradient trust — variance / SNR / CV** (already computed): inverse-variance ≈ precision, the
  statistically optimal weight for combining noisy unbiased estimates.
- **Direction agreement** — cosine of the update against the *running aggregate* (or against the buffer's
  mean). Anti-aligned updates (negative cosine) are candidates to reject or down-weight.
- **Representativeness** — client sample count / class coverage (non-IID aware).

**Axis B — how to combine (the core "weights≠gradients" choice).**
- **B0 scalar rate** (current): one scalar per update, magnitude-only.
- **B1 scalar but gradient-aware**: keep scalar form, feed it the trust/alignment signals (cheap upgrade).
- **B2 inverse-variance / precision weighting**: `w_i ∝ 1/var_i` (or `SNR_i`) — principled for noisy JVP
  estimates; uses signals already on hand.
- **B3 direction-aware / alignment**: project updates onto the running descent direction; weight by (or
  gate on) cosine agreement; optionally drop anti-aligned components rather than average them.
- **B4 coordinate/block-wise**: per-parameter-block trust instead of one global scalar (adapters are
  small — feasible).

**Axis C — reject vs down-weight vs hybrid.** Current: a **variance gate** (hard reject at bin level) +
soft down-weight (rate). Options: loosen the gate and push the discrimination into the *weight*; or keep a
hard gate on *direction* (anti-aligned) and soft-weight on *trust*.

**Axis D — the staleness clock (fixes H2).** Replace/augment `Δmodel_version` with a clock that differentiates
concurrent updates:
- `#commits since dispatch` (aggregations while this update was in flight),
- wall-age or vclock-age of the update,
- `#iterations` since dispatch,
- or **normalize** `Δv` by the current data-bin progression rate so a slow, variance-gated clock still yields
  spread. Quantify the staleness *distribution* first (§4) before picking.

**Axis E — normalization & LR coupling.** Do the weights sum to 1 / to `agg_goal`? A change in weight scale
silently rescales the effective server LR (`scale_add`, `fedbuff.py:230+`), so re-normalize or re-tune LR together.

---

## 4. Candidate schemes to prototype

All flag-gated, A/B against S0, measured on E1 (time-to-acc) and E3 (Δloss per forward-pass).

| id | scheme | axes | rationale |
|---|---|---|---|
| **S0** | current FeLiX scalar rate | B0 | control |
| **S1** | **inverse-variance** weighting `w∝1/var` (or `SNR`) | A:trust, B2 | principled for noisy JVP estimates; reuses §1 signals; directly attacks H3 |
| **S2** | **alignment-gated**: reject/down-weight anti-aligned (cosine<0 vs running aggregate), weight by \|cos\| | A:agreement, B3, C | attacks H1 — stops averaging opposing gradients |
| **S3** | **magnitude×SNR** `w∝\|JVP\|·SNR`, staleness-free | A:magnitude+trust, B1 | isolates informativeness from the dead staleness axis (H2) |
| **S4** | **finer staleness clock** (Δcommits or wall-age) in the FeLiX form | A:staleness, D | tests whether H2 alone explains the collapse |
| **S5** | **hybrid**: inverse-variance × alignment × finer-staleness | B2+B3+D | the "everything" combiner if S1–S4 each help |

---

## 5. How to know if it's working (evaluate the *aggregation*, not just the run)

Before E1/E3 outcomes, instrument the aggregation itself:
- **Staleness spread** — distribution/entropy of `Δv` across the buffer at each commit. If ~degenerate,
  H2 confirmed → a finer clock (Axis D) is mandatory.
- **Weight spread** — distribution/entropy of the assigned `rate`s. If FeLiX assigns near-uniform weights,
  it's not differentiating regardless of formula.
- **Weight ↔ usefulness correlation** — does a higher assigned weight predict a larger realized Δloss
  contribution? (post-hoc, per committed update). The real test of a good rule.
- **Wasted-work fraction** — share of committed forward-pass compute in low-weight / anti-aligned / dropped
  updates (ties back to the E3/E4 root cause).

---

## 6. Interactions & guardrails
- **C1 variance gate** — already filters high-variance updates; a variance-based *weight* (S1) risks
  **double-counting**. Decide: loosen the gate and move discrimination into the weight, or keep the gate for
  commit-timing and weight on an orthogonal signal (alignment/magnitude).
- **C2 dynamic K/C** ([`dynamic_kc_design.md`](dynamic_kc_design.md)) — concurrency `C` sets how many
  stale/in-flight updates coexist, i.e. how much there is to differentiate. High C makes a good rule matter
  more; the two together set the wasted-work E3/E4 measure.
- **Fidelity / sim parity** — any rule change is baseline-affecting: flag (byte-identical off), operator
  sign-off, re-validate real↔sim parity (variance cadence / commit order are parity-checked — a new weight that
  reorders commits moves parity).
- **Server LR coupling** — re-normalize weights or re-tune LR together (Axis E).

---

## 7. Open questions
- Is the right object a **scalar weight** at all, or a **direction-aware combine** (B3/B4)? H1 says maybe not.
- Should stale/anti-aligned updates be **rejected** (waste the compute but protect the model) or **kept and
  corrected**? Interacts with C2 (whether to have dispatched them at all).
- Does the Oort `stat_utility` (training-loss proxy) even correlate with *gradient* usefulness here, or
  should utility be redefined in gradient terms (|JVP|, SNR)?
- What's the cheapest signal that recovers most of the differentiation — so C3 doesn't add aggregator cost
  that eats FluxTune's speed win?

---

## 8. Anchors
- FedBuff rates + apply: [`flame/optimizer/fedbuff.py`](../../../flame/optimizer/fedbuff.py) — `alpha_*`/`beta_*` `:91-139`, `weight_factor` `:110-139`, `do()` rate select `:178-207`, `scale_add`/LR `:215-270`.
- Variance/SNR/CV signals + gate: [`aggregator/FedSgdAggregator.py`](../aggregator/FedSgdAggregator.py) `:194-202` (compute), `:132` (`var<var_threshold`), `:73-89` (thresholds).
- Grad-stat helpers: [`trainer/forward_training/fwdgrad_utils.py`](../trainer/forward_training/fwdgrad_utils.py) — `calculate_var/snr/cv/real_var/snr_gradients`.
- Staleness clock: `fwdllm_aggregator.py:1422-1426` (`_model_version` advances per data-bin), `:1206-1230` (staleness gate).
- Config: `_metadata/baselines.yaml:457-475` (fluxtune `agg_rate_conf`, `var_threshold`, `stalenessPolicy`).
