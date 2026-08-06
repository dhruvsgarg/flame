# FluxTune — what the code actually does (Q&A)

**Scope.** Answers grounded in the code on `dg/fluxtune_expts_sim_init`, the shipped config
(`_metadata/baselines.yaml` → `fluxtune`, `expt_scripts/fluxtune_n10_smoke.yaml`, launched by
`expt_scripts/run_sequential.sh:295`), and one full real run
(`experiments/run_20260804_043301_fluxtune_n100_smoke_syn_0_real`, N=100, α=1, K=10, C=30).
Every claim is tagged:

- **LANDED** — in code and active in the shipped fluxtune config.
- **LANDED-OFF** — in code, but the shipped config disables it.
- **DESIGN-ONLY** — described in `fluxtune_contributions.md` / `EXPTS_CHARTER.md`, no code.
- **MEASURED** — a number read out of the run above.
- **ANALYSIS** — my derivation, not measured. Flagged everywhere it appears.

The headline: **C1 (guided perturbation selection) is real and load-bearing. C3's freshness half is
real but weak. C3's direction half is implemented, fires constantly, and is numerically inert as
configured — it changes weights by ≤0.6%. The novelty weight ωᵈⁱʳ described in the design does not
exist in code and *cannot*, because the server never sees `v`.**

---

## 0. The pipeline, precisely (so the rest of the answers have a referent)

### 0.1 What one trainer does per dispatch

Per dispatch a trainer trains on **one 8-sample bin** (`data_id`), not an epoch.

1. Draw `P = perturbation_count = 10` candidate perturbations. For each trainable tensor it draws a
   `(P, *shape)` block from the client-seeded RNG (`tc_transformer_trainer_distribute.py:409-441`).
   So `v⁽ⁱ⁾` is a **raw standard-normal draw over trainable coordinates only**; frozen backbone
   coordinates are exactly zero.
2. Score all `P` candidates with a **central finite difference**
   (`fwdgrad_utils.py:105-134`), `h = 0.01`, two forward passes each, under `no_grad` + autocast:
   `d⁽ⁱ⁾ = [ L(θ + h·v⁽ⁱ⁾) − L(θ − h·v⁽ⁱ⁾) ] / 2h`.
   `L` is mean-reduction cross-entropy over the 8-sample bin. Dropout is off (`jvp_eval_mode: true`).
3. Rank by **|d|**, take the **top two**, pick one of them by a **fair coin flip** from the seeded RNG
   (`:470-485`). That is the selection rule — see §D.
4. Emit `g_k = d_k · v_k` (`:625-631`) — one scalar times one perturbation, per bin.
5. Upload `{param_name → d_k·v_k}`, plus `|d_k|` (diagnostic), the loss-derived `stat_utility`, and the
   `model_version` it trained on (`fwdllm_trainer.py:543-620`). **`v_k` itself is never uploaded, and
   no seed or index that would let the server reconstruct it is uploaded either.**

Compute: `2P = 20` forward passes per bin; the winner's JVP is reused from the selection cache rather
than recomputed (`jvp_perf_opt`).

### 0.2 What the server does per cycle

1. Collect `K = agg_goal = 10` updates. Sort them into a canonical `(modeled delay D, trainer_id)`
   order (`fwdllm_aggregator.py:2079-2147`) — deterministic in both real and sim since P0-1.
2. Merge each one into a running sum with a scalar weight `ω_k`
   (`fwdllm_aggregator.py:850-999`): `G ← G + ω_k · g_k`.
3. Compute a **split-half variance** of the pool (on one probe layer only, `layer_id_for_check = 12`
   for this model) and compare to `var_threshold = 0.3`.
   - **pass** → commit;
   - **fail** → roll back weights, stash this cycle's `(K, G)` in `cached_v`, re-dispatch, repeat with
     the same bin (`FedSgdAggregator.py:450-534`).
   - **plateau/cap force-commit** (Opt-2) → commit anyway.
4. On commit, with `N` = total number of *individual trainer updates* accumulated for this bin across
   all its cycles (`FedSgdAggregator.py:310-336`):

   **θ ← θ − η · (1/N) · Σₖ ωₖ dₖ vₖ**

   applied in place, per-parameter, no optimizer state. `model_version += 1`
   (`fwdllm_aggregator.py:2423-2461`), `data_id += 1`.

### 0.3 The weight ω, exactly as shipped

`agg_rate_conf: {type: grad_aware, base: new, align_gate: true, align_floor: 0.0, inverse_var: false,
scale: 0.4, a_exp: 0.25, b_exp: 0.1}`

ω factorizes into a base and a direction factor:

- **base** (the FeLiX scalar, `fedbuff.py:110-139`):
  `base = 0.4·α(s) + 0.6·β(U)`, with `α(s) = (1+s)^-0.25` (freshness) and
  `β(U) = 1.5 − (1+U)^-0.1` (loss/utility, "upshift" variant).
- **direction factor** (`fwdllm_aggregator.py:3889-3909`): for `cos < align_floor`,
  multiply by `(1 + cos)/(1 + align_floor)`; otherwise 1. `cos` is
  `cosine(g_k, G_partial)` — the running partial sum of *this cycle's already-merged updates*.
- `inverse_var` (multiply by `min(1, var_ref/var_k)`) exists but is **off**.

**MEASURED** over 34,410 merges in the reference run: ω ∈ [0.702, 0.865], median **0.818**.
Staleness took only two values: **s = 0 on 84.4%, s = 1 on 15.6%**, never higher.
Commits: **185 total — 130 plateau, 55 cap, 0 natural**; 3,442 cycles → ≈18.6 cycles per bin,
so **N ≈ 186 individual updates are averaged into each committed step**.

---

## Group A — Does the freshness weight actually do anything?

### A1. Is the aggregate normalized by the weight sum?

**No. It is normalized by the *count*, not by Σω.** `FedSgdAggregator.py:329` divides by
`training_num`, which is the number of individual trainer updates accumulated
(`fwdllm_aggregator.py:2204` hands over `(self.grad, self._agg_goal_cnt)`; retries add their own
counts at `FedSgdAggregator.py:289-299`). Nothing anywhere divides by `Σ ωₖ`.

Consequences, and they are the opposite of what a "relative-only" reading would give:

1. **A uniform scaling of all ω does not cancel — it scales the step.** Because every shipped ω is
   strictly below 1 (median 0.818, max 0.865), the committed step is a **~0.82× shrink of the plain
   mean**, i.e. the aggregation rate silently multiplies the effective learning rate by ~0.82. If you
   ever retune `scale`/`b_exp`, you are retuning the learning rate at the same time. That coupling is
   worth stating in the paper, or removing by dividing by Σω.
2. Relative differences also act, as expected — a staler or lower-utility probe contributes less than
   its neighbour in the same window.
3. So the paper **can** say stale updates shrink the step. What it should *not* imply is that the
   shrink is large: see A4.

**Recommended wording:** "each contribution is scaled by ω ∈ (0,1] and the buffer is averaged by
count, so ω acts both as a relative reweighting within the window and as a bounded global damping of
the step."

### A2. Is G applied directly, or fed to a stateful server optimizer?

**Applied directly.** `FedSgdAggregator.py:331` is `param.sub_(η · G/N)` — a raw in-place SGD step,
no momentum, no Adam, no EMA, no per-coordinate normalization, no optimizer state carried across
commits. This is finding F8 of `fluxtune_contributions.md`, and it is still true.

- A heavy-ball momentum hook **does** exist (`_server_update_step`, `FedSgdAggregator.py:238-249`),
  gated on `hyperparameters.server_momentum`, **default 0.0 = exact no-op**, and it is **not set in
  the shipped fluxtune config**. It is marked REVERTED in the contributions doc: at momentum 0.9 the
  run went to NaN loss by `data_id` 73.
- So **the magnitude information ω carries survives intact into the weights**. No Adam washes it out.
  Nothing needs scoping on that account.
- η itself is `hyperparameters.learning_rate = 0.01` times a linear decay `(comm_round − round)/comm_round`
  with `comm_round = 3000` and `warmup_ratio = 0` (`main_fedfwd_agg.py:210` overrides the JSON's 1).
  **MEASURED:** η stayed within [0.009993, 0.01] for the entire run — effectively constant.
  Note `optimizer.kwargs.learning_rate = 0.075` in `baselines.yaml` is **dead code for fluxtune**:
  fwdllm never calls `fedbuff.scale_add_agg_weights`. Worth deleting or documenting; it currently
  reads like the shipped LR and is not.

### A3. What unit is age counted in?

**Committed global model updates.** `staleness = self._model_version − msg[MODEL_VERSION]`
(`fwdllm_aggregator.py:898`), and `_model_version` increments exactly once per completed data-bin,
i.e. once per applied weight update (`fwdllm_aggregator.py:2423-2461`). Not wall-clock, not merge
events, not dispatch events.

This *is* comparable to FedBuff's τ (server steps) — with one caveat a reviewer will find and you
should pre-empt: **one fluxtune "server step" is not one FedBuff round.** A FedBuff step consumes K
client updates; a fluxtune step consumes K × (iterations in that bin) ≈ 186 forward-gradient samples
(§0.2). So a staleness of 1 in fluxtune spans ~19 aggregation cycles of wall time. Comparing the raw
exponents of the two staleness functions without saying this would be an apples-to-oranges
comparison in fluxtune's favour.

### A4. Is γ ≠ 0 in the shipped config, and is a bounded-staleness cutoff active?

- **The freshness exponent is non-zero but small, and it is polynomial, not exponential.**
  `a_exp = 0.25` inside `α(s) = (1+s)^-0.25`, and α enters ω with weight only `scale = 0.4`
  (`fedbuff.py:91,120,139`). The exponential form exists in code (`alpha_exponential`) but the call
  site hardcodes `alpha_type="polynomial"` — it is unreachable from config.
- **There is no bounded-staleness cutoff.** `stalenessPolicy: fedbuff` in the shipped yaml means
  "accept everything, down-weight by version gap"; the `baselines.yaml` comment states the absence is
  a deliberate choice. Nothing is discarded on age. (`refl.py` has such a cutoff; the fwdllm
  aggregator does not.)
- **But the decay is doing almost nothing, for a systems reason rather than a tuning one.**
  **MEASURED:** staleness never exceeded 1 in the whole run. The only two ω values freshness can
  produce are:
  - `s=0` → `α=1` → contribution 0.400
  - `s=1` → `α=0.841` → contribution 0.336

  With `U ≈ 10.8` (β ≈ 0.720, contribution 0.432), that is **ω = 0.832 vs 0.768 — a 7.7%
  down-weight, applied to 15.6% of updates.** The observed ω spread [0.702, 0.865] is therefore
  driven mostly by the **utility** term β(U), not by freshness.

  The reason staleness stays ≤ 1 is structural: the model version only advances on a commit
  (~19 cycles), while a trainer's round trip is ~1 cycle. Under this K/C and this variance gate,
  in-flight updates essentially cannot get stale.

**Honest claim to make:** freshness weighting is *implemented and correct*, but under the shipped
operating point it is a ≤8% effect on a sixth of the updates. Claim it as a mechanism that keeps the
system safe under staleness, not as a source of the measured accuracy. If you want it to be
load-bearing, you need a regime that actually produces staleness (higher C/K ratio, real delays,
mobiperf availability), and you should report the observed staleness histogram alongside the claim.
Also note: the F9 amplification risk (`β_upshift` can push ω above 1, up to 1.3 at extreme loss) is
**live in the code but never triggered in this run** — max observed ω was 0.865.

---

## Group B — Is the novelty weight implemented as documented?

### B1. Does the aggregator have access to `v_k`?

**No, and this is structural, not an oversight.** The upload message
(`fwdllm_trainer.py:587-620`) contains `d_k·v_k` per parameter, `|d_k|` as a scalar for SNR
diagnostics, `stat_utility`, versions and timestamps. There is **no `v_k`, no RNG seed, no candidate
index**. The server cannot factor `d_k v_k` into its parts either: `d_k` is uploaded only as `abs(jvp)`
and only as a diagnostic, so even the sign is not recoverable from the message.

Therefore **ωᵈⁱʳ = 1 − maxⱼ|v_k·v_j| does not exist in code and cannot be computed with the current
protocol**. This is the concrete answer to the open "how does the server learn which candidate won"
item: it doesn't. **DESIGN-ONLY.**

What *does* exist (`fwdllm_aggregator.py:943-967`, `:3911-3930`) is a different quantity: the cosine
between the incoming **update** `g_k = d_k v_k` and the **running partial sum** of the current
cycle's already-merged updates. Call it an *alignment* gate, not a *novelty* gate — they point
opposite ways (alignment rewards agreement, novelty rewards disagreement), so the design text should
not be read as describing this code.

**And the measurement says the alignment gate is inert as configured. MEASURED:** across 34,410
merges the gate fired 15,414 times (44.8% — exactly the ~half you expect when cosines are symmetric
around zero), but every logged cosine was in **[−0.006, 0.000]**, median **−0.001**. With
`align_floor = 0.0` the multiplier is `(1 + cos)`, so the gate's actual effect was **≥ 0.994** —
a down-weight of at most **0.6%**, typically 0.1%.

This is not a tuning accident, it is dimensionality. `g_k = d_k v_k` with `v` an isotropic draw in
p ≈ 1.04M dimensions: two such vectors have cosine of order `1/√p ≈ 0.001`. The gate can only bite
when updates are genuinely collinear, which random probes never are. **`grad_aware_gated_total` is
therefore a counter of sign flips, not evidence of an effect** — it should not be reported as
"45% of anti-aligned updates were gated" without the magnitude.

Note also that the R1→R4 ablation ladder (83.00 → 84.08%) sits **inside the measured same-seed
replicate floor** for accuracy (`parity_floors/fluxtune.yaml`: `accuracy_diff: 0.023`, i.e. 2.3
points between two config-identical legs). Combined with the 0.6% weight perturbation above, the
R3 "grad-aware helps" reading is not supportable from that ladder. If C3 is to be claimed, it needs
either a configuration where it bites (see below) or a different metric than final accuracy.

**To make it bite** you would need `align_floor > 0` (e.g. 0.02, well above the `1/√p` noise band) —
but then it gates on noise, not on signal, which is worse. The defensible version of C3-direction in
this geometry is not a pairwise cosine at all; it is the variance/SNR path (`inverse_var`, currently
off) which operates on magnitudes that *do* separate.

### B2. Window-only or against merged history?

**Window-only, and narrower than that: within the current cycle.** `self.grad` is the reference, and
it is zeroed immediately after each commit (`fwdllm_aggregator.py:2209`). So the comparison is
against the partial sum of the ≤9 updates already merged in this cohort — not against merged
history, and not even against the whole bin's accumulated `cached_v`. Cheap and defensible; just
describe it accurately as "within the in-flight buffer".

### B3. Is it greedy in arrival order?

**Yes, greedy and order-dependent — but the order is canonicalized, so it is reproducible.**

- Greedy: the first update in a cycle has `cos = None` (zero-norm reference) and is merged ungated;
  each subsequent one is compared against the sum of its predecessors. Different orders → different
  weights → a different aggregate. This is finding F16.
- Reproducible: since P0-1, the merge order is canonically sorted by `(modeled delay D, trainer_id)`
  in **both** real and sim (`fwdllm_aggregator.py:2079-2147`), so it is no longer a function of
  nondeterministic physical arrival.

So the reproducibility caveat is **resolved** for run-to-run replication, but the *algorithmic*
caveat remains and the paper should own it: the aggregate is a function of an ordering that has no
mathematical justification (it is the delay ordering, not a statistical one). Given B1's finding
that the effect is ≤0.6%, this is currently a footnote rather than a threat.

### B4. Full parameter vector or trainable slices only?

**Trainable only, on both sides, with no leakage.** The uploaded `grad_dict` is built from
parameters with `p.grad is not None` (`fwdllm_trainer.py:543-548`), which for this model is the
adapter + head set; `_cosine_flat` iterates `named_parameters()` but only accumulates entries whose
name is present in the uploaded dict (`fwdllm_aggregator.py:3911-3930`). Frozen coordinates are in
neither the norms nor the dot product. Normalization is therefore over the 1.04M trainable
coordinates, consistent with the finite difference. Nothing to fix.

### B5. Does the sign-consistency gate exist, is it hard or soft, does it recompute G?

- **It exists, and it is soft.** `_grad_aware_rate` (`fwdllm_aggregator.py:3889-3909`) applies a
  *linear ramp*: weight 1 at `cos ≥ align_floor`, falling linearly to 0 at `cos = −1`. Nothing is
  ever dropped; the function is bounded above by `base` by construction, so C3 can only down-weight
  and can never inflate the effective LR. That bound is a genuine, checkable property worth stating.
- **There is no hard drop, so the "recompute G after dropping" question is moot.** The weight is
  applied at merge time in a single pass; `G` is never revisited, renormalized, or recomputed.
- **But per B1 it is soft to the point of inertness at the shipped `align_floor = 0.0`.** So the
  accurate description for the paper is: "a bounded soft down-weighting of anti-aligned updates,
  implemented and active, whose magnitude under isotropic high-dimensional probes is negligible" —
  or move it to the ablation section as an option with the measurement attached. What it should not
  be is a headline mechanism in the shipped-system description.

---

## Group C — The estimator / bias question

### C1. Is the update Σ dₖvₖ or (1/K)Σ dₖvₖ?

**A mean, not a sum**, and the divisor is the accumulated count `N`, not the cycle's `K`:

θ ← θ − η · (1/N) Σₖ₌₁ᴺ ωₖ dₖ vₖ,  **MEASURED** N ≈ 186 (K=10 × ~18.6 cycles/bin), η ≈ 0.01.

Since ω̄ ≈ 0.82 and the divisor is a count, the shipped estimator is `0.82 × mean(dₖvₖ)` (§A1).

**The unbiasedness argument holds for the mean form**, which is the one in code:
for `v ~ N(0, I_p)` and `d = ⟨g, v⟩`, we have `E[d·v] = E[vvᵀ]g = g` exactly. So the *random*-direction
estimator is unbiased, and averaging N of them keeps it unbiased while cutting variance by N.
Guided selection breaks that scaling — see C-analysis below.

### C2. Are `v` unit-norm, or raw Gaussian?

**Raw Gaussian**, `torch.randn`-drawn per trainable coordinate, never normalized anywhere
(`tc_transformer_trainer_distribute.py:416`, `:543-550`). So `‖v‖ ≈ √p ≈ 1020` for p ≈ 1.04M.

Two separate consequences, and it matters not to conflate them:

- **For the *ranking* (C1's selection), the length confound is negligible.** The relative standard
  deviation of `‖v‖` across draws is `1/√(2p) ≈ 0.07%`. In high dimension the norms concentrate, so
  ranking by `|d|` is ranking by `|⟨g, v̂⟩|` to within a tenth of a percent. C1's score is safe.
- **For the *estimator*, the √p scaling is fully baked in.** `‖d·v‖ ≈ |g|·√p ≈ 10³·|g|` per sample.
  Nothing normalizes it out; the small learning rate (0.01) is what absorbs it. If you ever state
  "the update is an unbiased estimate of g", state that it is unbiased *in expectation* while being
  ~10³× larger than g *per sample*, and ~10²× larger after averaging N ≈ 186 (see §E).

Nothing in `ω` (C3's weight) uses `‖v‖` either, so C3's weighting is not contaminated by length.

### C3. Are the P candidates drawn from the same distribution the unguided baseline uses?

**Yes for the distribution — but there is no unguided arm in the code to compare against.**
All arms draw the same `(P, *shape)` standard normal block from the same seeded generator
(`:409-441`). The `fwdllm` baseline then selects among them by **cosine similarity to the carried
server gradient** (`:424-436`); fluxtune selects by |JVP|. There is **no arm that skips selection**.

So the clean C1 ablation the paper needs — "same draws, selection is the only difference" — is
currently *unavailable*, and two code facts block it:

1. Setting `perturbation_count: 1` **crashes**: the top-2 pick indexes `sorted_indices[-2]` on a
   one-element list (`:481`). The contributions doc's "fluxtune P=1 ≡ sync, 16 ms" row (§3.3) is a
   cost-model projection, not a runnable configuration. Either guard that index or add an explicit
   `select: random` mode.
2. Changing `P` changes how many draws are consumed from the client RNG, so a P-sweep is **not**
   seed-matched across arms — the trajectories diverge for a reason unrelated to selection. A
   `select: random` mode that still draws all P and takes index 0 solves both problems at once and
   is a ~5-line change. **I'd land that before running the C1 ablation.**

### C4. Is `d` rescaled anywhere between client and server?

**No.** The chain is: `d` is produced already divided by `2h` in the finite difference
(`fwdgrad_utils.py:133`), multiplied by `v` (`:628`), scaled by the scalar `ω` at merge
(`fwdllm_aggregator.py:982`), summed, divided by the count, multiplied by η. There is **no** ε-rescale,
no batch-size rescale, no gradient clipping (`max_grad_norm: 1.0` exists in `model_args.py:51` but is
a HuggingFace-trainer field on a path fluxtune does not take), no loss-reduction switch — CE is
mean-reduction throughout, so `d` is the directional derivative of the **mean** batch loss.

So `d ≈ ⟨g, v⟩` survives to the server, and the estimator framing is premised correctly. Two
second-order caveats to state rather than hide:

- **Truncation:** the central difference is `⟨g,v⟩ + O(h²·‖v‖³·∇³L)`, and `h·‖v‖ ≈ 10` is not a small
  perturbation of the parameters. The estimator is exact only to the extent the loss is locally
  quadratic over that displacement. This is a real caveat and it is *not* the same as the fp16 one.
- **Cancellation:** two O(1) losses differing by O(h) are subtracted under autocast; the contributions
  doc §4.2 measures ~1–2 significant figures surviving in fp16, and `FWDLLM_JVP_FP32=1` exists as an
  escape hatch (off by default). So `d` carries multiplicative noise of order a few percent before it
  ever reaches the server.

---

## Group D — The selection rule

### D1. Signed JVP or |JVP|?

**|JVP| — magnitude, not signed** (`:470`, `key=lambda x: abs(x[1])`). Both the design text
("largest JVP") and the "perturbation that leads to the maximum loss" phrasing are wrong descriptions
of the code; the log line even reads `All JVPs sorted by magnitude`.

**Magnitude-max is the right choice here and the paper should say so on purpose, not by accident.**
Writing `v = v_∥ĝ + v_⊥`, we get `d = |g|·v_∥`, so `d·v = |g|(v_∥²ĝ + v_∥v_⊥)`. The update term is
**even in the sign of `d`** — flipping `v → −v` flips `d` and leaves `d·v` unchanged. So a *signed*
max would be selecting on a quantity the estimator is invariant to, and would waste half the
candidates. Magnitude-max maximizes `v_∥²`, which is exactly the useful quantity. The descent
direction is recovered by the minus sign in `θ ← θ − ηG`, not by the selection.

### D2. Is `k` a config knob with default 2, or hardcoded?

**Hardcoded, and it is not a top-k average — it is a coin flip between the top two**
(`:481-483`): `pair = [sorted_indices[-1], sorted_indices[-2]]`, then a uniform draw from the seeded
RNG picks one. Only that single winner's `d·v` is uploaded.

So the planned "sweep k" ablation **cannot run today**; it needs a `top_k` hyperparameter threaded
from config, plus a decision on semantics (pick-one-of-k, which is what the code does, versus
average-the-k, which is a different and probably better estimator). By contrast **`P` is a real knob**:
`hyperparameters.perturbation_count`, default 10, plumbed at `trainer/main.py:142-143`. The P sweep
is runnable (modulo C3's two caveats); the k sweep is not.

Worth noting the coin flip is undocumented in the design text and is not a tie-break — it is a
deliberate 50/50 between the best and second-best candidate every single time, presumably for
exploration. It costs about 15% of the achievable `E[v_∥²]` (see §E) and should either be justified
in the paper as exploration or removed.

### D3. Are all P candidates scored every iteration, or is there a schedule?

**All P, every iteration, unconditionally** (`:460-467`). No schedule exists.

A schedule was written and is **deliberately disabled in place**: `:474-477` holds a commented-out
condition (`if 0.8 * databin_best_jvp_val > abs(sorted_jvps[-1])`) that would have carried the bin's
best perturbation across iterations, behind a literal `if False:` with the comment *"carefully delete
this condition, we do not want to use best across iterations - this reduces exploration"*. The
supporting state (`databin_best_v_params`, `databin_best_jvp_val`, the `best_idx == -1` branch at
`:657`) is all still live.

So "the schedule is a knob" is **DESIGN-ONLY**; it should be marked as planned, and the doc should
note the previous attempt was reverted for an exploration-collapse reason — that is useful evidence,
not a gap.

---

## Group E — The one measurement that settles it

**It does not exist yet.** `scripts/profile_jvp_opt.py` has a `backprop_ref` (`:211-226`) but it is a
*cost/memory* reference — it never compares directions. Nothing in the tree logs `cos(G, g)` or
`‖G‖/‖g‖`. So this is a new probe. Below is what to build, and what I predict it will find, so you can
tell immediately whether the run is sane.

### E1. What to log

On a small model (DistilBERT + adapters is already small enough; p ≈ 1.04M trainable), at each
commit, alongside the existing `server_update` telemetry (`FedSgdAggregator.py:338-356`, already
carries `‖Δ‖` and `‖w‖` — extend it):

- the true backprop gradient `g` on the same accumulated bins,
- `cos(G, g)` and `‖G‖/‖g‖` for the merged `G` actually committed,
- the same two for the same collected `{(v_k, d_k)}` re-merged **without** ω (isolates C3 from C1),
- the min-norm least-squares solve over the same `{(v_k, d_k)}`.

To do this you must **retain `v_k` server-side**, which today you cannot (§B1). Cheapest route: have
the trainer upload the winning candidate's index plus its RNG state, and reconstruct `v_k` on the
server — that is also exactly the mechanism ωᵈⁱʳ would need, so it unblocks both.

Run it under (i) random directions (needs the `select: random` arm from §C3), (ii) guided, (iii) LS.

### E2. Predicted results — **ANALYSIS**, not measured

With `v_∥ ~ N(0,1)`, `d·v = |g|(v_∥²ĝ + v_∥v_⊥)`, and `E[v_∥v_⊥] = 0` because `v_⊥` is independent of
a selection rule that reads only `v_∥`:

- **Guided selection does not bias the direction. It inflates the scale.**
  `E[d·v | selected] = E[v_∥²]·g` — still parallel to `g`. Random gives `E[v_∥²] = 1` exactly;
  coin-flip-of-top-2-of-10 gives `E[v_∥²] ≈ 3.2` (top-1 alone would be ≈3.8).
  **So the guided estimator is unbiased in direction and inflated ≈3× in norm.** That is the precise,
  defensible statement, and it is stronger than "guided selection biases the estimator".
- **Noise:** the orthogonal part has norm ≈ `|g|·√(E[v_∥²]·p/N)`. With p = 1.04e6 and N ≈ 186:
  - random: `‖G‖/‖g‖ ≈ ω̄·√(p/N) ≈ 0.82 × 74.8 ≈ 61`, `cos(G,g) ≈ 1/74.8 ≈ 0.013`
  - guided: `‖G‖/‖g‖ ≈ 0.82 × 134 ≈ 110`, `cos(G,g) ≈ 3.2/134 ≈ 0.024`
  - **both cosine and norm ratio scale as `√(E[v_∥²])` ≈ 1.8×.**
- **Least squares (N ≪ p) fixes the scale, not the direction.** The min-norm solve is the projection
  of `g` onto `span{v_k}`, giving `cos = √(N/p) ≈ 0.013` and `‖G_LS‖/‖g‖ ≈ 0.013`. Note that cosine
  is *the same* as the plain mean's under random directions — at N ≪ p the mean and the LS solve
  point almost identically and differ only in norm (by a factor ≈ p/N ≈ 5600). **If the experiment
  shows LS improving the cosine materially, something in the setup is wrong.**

### E3. What this predicts about the paper's claims

1. **The C1/C3 coupling you suspected is real, and it is a scale coupling.** Guided selection
   multiplies the expected step by ≈3× at fixed η. So "fluxtune (guided) vs baseline (random)" is
   partly an unmatched-learning-rate comparison. **The clean ablation is guided vs random with η
   scaled by `E[v_∥²]`** (or with `G` renormalized). Under that control, the surviving C1 benefit is
   the ~1.8× SNR gain, which is real and worth claiming — but it is a smaller and much more
   defensible number than an uncontrolled A/B would show.
2. **It also explains §8's oscillation problem quantitatively.** At the predicted numbers the
   committed step has norm ≈ η·110·|g| ≈ 1.1|g|, of which only ≈2.4% is aligned with `g`: the aligned
   component is ≈0.026|g| while the orthogonal random-walk component is ≈1.1|g| — a ~40× noise-to-signal
   ratio *in weight space, per commit*. That is F8's "undamped random walk", with a number attached,
   and it says the fix is variance reduction (larger N, or the LS/normalized step), not a smaller η —
   shrinking η shrinks signal and noise equally.
3. **It gives the freshness claim its ceiling.** A weight that varies by 8% cannot matter against a
   40× noise-to-signal ratio. Report C3-freshness as a safety mechanism, not a convergence driver.

---

## Landed vs designed — the full ledger

| Item | Where | Status |
|---|---|---|
| C1 guided perturbation selection (|JVP|, top-2 coin flip, P=10) | `tc_transformer_trainer_distribute.py:456-487` | **LANDED**, active |
| Trainable-only finite difference + winner-JVP reuse (`jvp_perf_opt`) | `fwdgrad_utils.py:105-134` | **LANDED**, on in the yaml |
| Opt-1 redundant weight-resend suppression | `fwdllm_aggregator.py:3834-3866` | **LANDED**, unconditional |
| Opt-2 variance-plateau + iteration-cap force-commit | `fwdllm_aggregator.py:3868-3887`, `FedSgdAggregator.py:488-526` | **LANDED**, and **MEASURED to be the only commit trigger firing: 130 plateau + 55 cap, 0 natural** |
| C3 staleness×utility base weight (FeLiX scalar) | `fedbuff.py:110-139` | **LANDED**; freshness contributes ≤8% on 15.6% of updates |
| C3 alignment gate (soft, bounded ≤ base) | `fwdllm_aggregator.py:3889-3909` | **LANDED** but **numerically inert**: fires 44.8%, effect ≤0.6% |
| C3 inverse-variance reliability weight | same | **LANDED-OFF** (`inverse_var: false`) |
| S1 server momentum | `FedSgdAggregator.py:238-249` | **LANDED-OFF**, default 0.0; REVERTED (diverged to NaN at 0.9) |
| C2 dynamic K/C | `selector/dynamic_kc_policy.py` | **LANDED-OFF** (`dynamic_kc.enabled: false`) |
| Deterministic `(D, trainer_id)` merge order | `fwdllm_aggregator.py:2079-2147` | **LANDED** |
| ωᵈⁱʳ novelty weight `1 − max\|v_k·v_j\|` | — | **DESIGN-ONLY**; impossible without a protocol change (server never sees `v`) |
| Hard sign-consistency gate + G recompute | — | **DESIGN-ONLY** (only the soft ramp exists) |
| Least-squares gradient solve | — | **DESIGN-ONLY** |
| Per-iteration selection schedule | `:473-477` | **DESIGN-ONLY**, previously written and disabled with `if False:` for exploration collapse |
| `k` (top-k) as a knob | — | **DESIGN-ONLY**; hardcoded 2 |
| Unguided/random-direction arm for the C1 ablation | — | **DESIGN-ONLY**; `P=1` crashes today |
| `naive_avg` aggregation arm (for the C3 ablation) | — | **DESIGN-ONLY**, not implemented |
| H1/H2/H3 (data shuffle, bin size, bin-order permutation) | — | **DESIGN-ONLY** |
| S2 variance-gate recalibration, S3 rate cap | — | partially subsumed by Opt-2 / the `≤ base` bound; the threshold recalibration is **DESIGN-ONLY** |

## Smallest set of code changes that unblocks the paper

1. **`select: random` arm** — draw all P, take index 0. Unblocks the C1 ablation, fixes the `P=1`
   crash and the RNG-stream mismatch at once. (~5 lines)
2. **Upload the winner's candidate index + RNG state** — makes `v_k` reconstructible server-side.
   Unblocks the Group E measurement *and* is the prerequisite for ωᵈⁱʳ if you still want it.
3. **`top_k` as a hyperparameter**, and decide pick-one vs average-the-k. Unblocks the k sweep.
4. **Divide by `Σ ω` instead of by the count**, or state the ~0.82 LR coupling explicitly in the paper.
5. **Delete or document `optimizer.kwargs.learning_rate: 0.075`** — it is dead for fluxtune and reads
   like the shipped LR.
