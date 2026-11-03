# FluxTune paper outline (MLSys) — arguments only

Format: each bullet = one claim; sub-bullets = points that **must** appear when writing it. ⚠ = number or evidence pending.
Rules: [PAPER_TEMPLATE.md](PAPER_TEMPLATE.md) Parts A-B (P# = its moves), Part E; [FLUXTUNE_PAPER_WRITING_RULES.md](../fwdllm/FLUXTUNE_PAPER_WRITING_RULES.md)
(gaps are properties of forward-only training; FwdLLM is the instrumented case study; FwdLLM terms first).
Evidence: [EXPERIMENTS.md](../paper_expts_fluxtune/EXPERIMENTS.md), [RUN_PLAN.md](../paper_expts_fluxtune/RUN_PLAN.md),
[cheatsheet](../fwdllm/fl_fwd_ft_pipeline_cheatsheet.md) §8, [FLUXTUNE_READINESS.md](FLUXTUNE_READINESS.md), firewall [BASELINES.md](BASELINES.md).
**Scope now:** title → §5. **Status:** P1 DistilBERT · AG News done; P3 / P4 `fluxtune` running (decision 15), `fwdllm` arms next.
**Headline so far (P1, 1 seed):** 4.8× to FwdLLM's peak, ≥ 6.2× at 0.86, +1.8 pt peak; FwdLLM DNF (0.855). Never extrapolate a DNF.

## 0. Vocabulary, concepts, choices

| internal | paper term |
|---|---|
| upload / commit | client update / server update |
| pool, N | buffer, buffer size N |
| C in flight, K per iteration | concurrency C; K internal |
| probe, P | perturbation, P per client update |
| trust-ratio step, ρ | normalized server step; relative step size ρ = ‖Δθ‖/‖θ‖ |
| `exp`, T_res | step-size decay; decay constant |
| Φ / Λ | weight-norm growth / cumulative progress |
| data bin, `data_id` | data bin (defined in §2) |
| task, D / vclock, TTA | local task, device time per task / simulated time, time to accuracy |

**Concepts (only these are claimed):**

| # | concept | FwdLLM | FluxTune | ablation |
|---|---|---|---|---|
| E1 | Perturbations as samples, not candidates | keeps 1 of 10 | averages all 10 | runs 7, 26 |
| E2 | Normalized server step with decay | SGD, lr 0.01 | step = ρ_t‖θ‖, decays to a growth budget | run 8 |
| E3 | Fixed buffer size | aggregate when variance ≤ threshold | aggregate every N = 50 | run 9 |
| E4 | Data-bin client selection | one cohort held for all iterations of a bin | fresh clients each iteration, one task each | ⚠ none yet |

- **Async is the substrate, not a concept.**
  - FedBuff's engine, cited, unchanged; never called ours.
  - Appears only in §2 (one sentence), §4 (grey box), §5.5, evaluation.
  - Run 10 (sync control) and C4 (async with FwdLLM's estimator) are controls, not claims.
- **E4 credit.**
  - Ours: keying selection to the data bin, one task per engagement, many clients per buffer.
  - Felix's (cited): utility / availability scoring inside the selector.
  - ⚠ BASELINES.md says FwdLLM owns the data-bin primitive; settle what we claim.
- **Not claimed (no ablation):** LoRA rank, central-difference JVP, stale-update acceptance, stop rule, batch size, momentum.

**Design choices** (each is a "why not" or a setting in §5; 1 seed, 2.3-pt noise):

| choice | verdict | key evidence |
|---|---|---|
| step unit s, not fixed ρ0 | kept, per cell | fixed ρ0 gave arms s 2.01 / 1.40 vs 2.81 |
| explicit Φ_budget | kept, 2 / 4 / ≈ 210 | too small caps accuracy (1.7B 0.598 at 2) |
| N 50 everywhere | kept | N* 34-42 AG News, 38 Yahoo; p cancels |
| P 40 | dropped | 4× cost for ~4× progress |
| gradient-aware weights | dropped | within 0.6 pt, 14% slower |
| constant ρ | dropped | 0.867 vs 0.876 with decay |
| momentum 0.9 | big cells only | 1.7B 0.432 vs 0.357; diverged on DistilBERT |
| LoRA r 2 (1.7B), bs 32 | per model / task | +3-6 pt; 35-40% fewer vclock h |

**Open:** title (**FluxTune: Fast and Stable Backpropagation-Free Federated Fine-Tuning of LLMs**); second forward-only baseline
(MeZO-style) ⚠ not planned; availability = one C4 row only if FT-N5 lands.

## 1. Spine

- **Setting:** on-device LLM fine-tuning where backprop does not fit.
  - Memory wall, inference-only NPUs (FwdLLM's premise, cited).
- **Accounting (P19):** TTA = server updates needed × time per server update.
  - Concepts cut the first factor; the async engine sets the second.
- **Root cause (P2):** rules built for backprop gradients, applied to an estimate ~0.5% aligned with the gradient.
  - Each rule trusts one noisy reading: one perturbation, one variance, one cohort.
  - Fixed constants drift as weights grow.
- **Tension (P1):** update often vs make each update accurate.
- **Insight:** make every server rule relative to the weights and spread across clients.

| gap (§3) | hurts | concept |
|---|---|---|
| perturbations ≈ perpendicular noise; picking the best cannot help | updates needed, bytes | E1 |
| SGD step grows with the weights; peak then collapse | final accuracy | E2 |
| variance threshold drifts | updates needed | E3 |
| non-IID data: a held cohort imprints its label mix; holding it is unrealistic | final accuracy | E4 (+ E2) |

## Title and abstract (≈ 170 w)

- **Context:** on-device fine-tuning needs backprop-free training.
  - Forward gradients fit device memory and NPUs.
- **Problem:** real-world client data is heterogeneous (non-IID), and forward-only FL handles it poorly: slow convergence, accuracy lost late.
  - Each client update is near-noise, so a small held cohort's label skew dominates the model.
  - FwdLLM on heterogeneous AG News plateaus at 0.855, never reaching 0.86.
- **Unavailability (one line):** phones are online only in short sessions, so holding a cohort for a whole data bin stalls when clients drop out.
- **Why prior work fails:**
  - What works: the estimate is unbiased and needs inference memory only.
  - What's missing: ~0.5% of each estimate lies along the gradient, yet backprop-era rules are kept.
  - They break as weights grow and as client data differ.
- **Insight:** average perturbations, aggregate by count, size the step to the weights, spread updates across clients.
- **System:** FluxTune's four concepts on standard async aggregation.
  - Settings follow from model size and dataset: works on a new model or task without a learning-rate search.
- **Results (P3 ranges):** N-M× faster, N-M× fewer GB over 3 models × 3 datasets ⚠ runs 1-6.

## §1 Introduction (≈ 1.25 pages)

- **Fig 1:** accuracy vs time and vs GB on the hardest cell, "N× faster, N× fewer GB" ⚠ runs 3-6.
- **¶1 Stakes:** personal text lives on phones; backprop does not fit there.
  - 7.7 GB to backprop RoBERTa-large; NPUs 30× faster but inference-only.
  - Models grow faster than device memory.
- **¶2 Tension (bold):** forward-only FL must update often, yet each client update carries almost no signal.
  - Alignment ≈ √(P/p) = 0.5%; a buffer helps only as √N.
- **¶3 Why existing systems fall short (by assumption):** forward-only FL keeps backprop FL's rules, which assume each client update is a reliable gradient.
  - *Client update, "send the best perturbation":* every perturbation is mostly noise orthogonal to the gradient; selecting one scales signal and noise together, so update quality never improves.
  - *Aggregation, "aggregate once client updates agree":* update variance scales with the gradient norm, which shifts during training, so a fixed threshold means a drifting buffer size (36× within one run).
  - *Server step, "a fixed learning rate is safe":* noisy steps are near-perpendicular to the weights, so the weight norm grows every round; accuracy peaks, then falls.
  - *Participation, "a few clients represent the population and can stay":* under non-IID data, a small cohort reused across rounds pulls the model toward its own label mix.
  - Long participation is also unrealistic: it needs devices online for minutes to hours, but phones connect in short sessions (cite trace ⚠).
  - Async FL speeds these rules up but does not fix them (C4 ⚠).
- **¶4 Insight (P2):** one root cause: nothing is relative to the weights or spread across clients.
  - Follows from ĝ = (∇f·v)v, so any method using it inherits it.
- **¶5 Ideas (no formulas, no constants):**
  - Combine perturbations, don't pick.
  - Size the step to the weights and shrink it.
  - Aggregate when enough updates arrive, not when they agree.
  - Spread each update across many clients, one task each.
  - Closing clause: standard async engine (cited); settings carry to new models and tasks.
- **¶6 Results:** time and GB to target vs FwdLLM on 3 cells ⚠.
  - DNF rule: time to FwdLLM's peak plus "≥ X×" at target.
  - Removing any concept slows TTA N× ⚠.
- **¶7 Contributions (ideas, not "we built"):**
  - *Characterization:* four structural gaps (picking is stability-neutral, threshold drifts, SGD grows weights, held cohort imprints skew).
  - *Design:* rules relative to weights, spread across clients; settings in scale-free units (N, s, Φ_budget), unlike FwdLLM's per-model screens.
  - *Analysis:* weight growth predicted exactly; progress law; buffer rule independent of model size.
    - ⚠ growth-law accuracy: 0.09% here vs ≤ 0.3% in §5.2.2; pick one.
  - *Evaluation:* 3 models × 3 datasets, time and bytes, leave-one-out, simulator checked against real runs.

## §2 Background (≈ 0.6 page)

- **Forward gradients:** two forward passes → directional derivative along a random perturbation.
  - Unbiased; variance grows with trainable p; adapters keep p at 0.45-0.81M.
- **FwdLLM (fig: one round):** the system we instrument.
  - Server picks K clients; each keeps 1 of P perturbations and uploads.
  - Server repeats iterations with the same cohort until variance < threshold, then SGD step.
  - Defaults: lr 0.01, threshold per model.
- **Data bins:** each client's data cut into fixed batches by `data_id`.
  - Bin k = every selected client's own k-th batch (a slice across clients).
  - A lap visits every bin once (150 at bs 8, AG News).
  - In FwdLLM a bin ends when the threshold passes; iterations per bin vary (15 → 34).
- **Async FL (one sentence):** FedBuff buffered aggregation is standard; FluxTune uses it as is.
- **Notation table:** p, P, N, C, K, ρ, Φ, Λ, D, data bin.
- **Baselines:** FedBuff(P), Felix(P) = async with FwdLLM's estimator (firewall labels).

## §3 Motivation (≈ 1.5 pages)

- **Setup:** same cells as the evaluation; backprop ceiling on every curve (P5).
- **Table 1 (P8):** FwdLLM, MeZO, FedKSeed, FedMeZO × {perturbations, when to aggregate, server step} ⚠ verify.
  - Blind spot: none sizes the step to the weights.
- **§3.1 Every perturbation is mostly noise, so picking the best cannot help → E1.**
  - cos ≈ 1/√p (0.15%): ĝ is > 99.8% sideways noise.
  - Ranking by |JVP| raises signal and noise together: stability unchanged for any rule; mean improves it P×.
  - Cosine pick is biased toward the stale aggregate; mean is unbiased.
  - Picking discards 90% of computed forward passes.
  - Concede: MeZO averages; the claim targets forward-gradient FL's picking rule.
  - Takeaway: average perturbations. Fig: [C](../fwdllm/figs/concept_noise_optC.png) or [D](../fwdllm/figs/concept_noise_optD.png) ⚠ pick one.
- **§3.2 SGD steps grow with the weights and undo training → E2.**
  - Steps are orthogonal to θ, so ‖θ‖ grows every update.
  - Peak then collapse; FwdLLM weights 4.31× and rising.
  - Cohort mix moves the SGD step 1.6×.
  - Takeaway: step relative to the weights, and decay it.
- **§3.3 A variance threshold drifts within a run → E3.**
  - Variance ∝ gradient norm²: 36× drift in one run.
  - Does not transfer: RoBERTa's 0.2 gave 0 commits on 1.7B; re-screened per model.
  - Takeaway: aggregate a fixed count.
- **§3.4 On non-IID data, a held cohort decides where the model goes → E4.**
  - Held cohort with 58% class 1 drove RoBERTa-L to one class (share 0.996).
  - Reselecting per update fixed the collapse, not the plateau (FwdLLM's own paper reports it).
  - Holding a cohort across a bin: tasks of 4-46 min (RoBERTa-L) to 2.8 h (1.7B); unrealistic for phones (cite ⚠).
  - Availability gap: a held cohort stalls the whole bin when any member drops out, and phones are online only in short sessions.
  - Takeaway: spread each update across many clients, one task each.
  - ⚠ evidence is FwdLLM-only; needs a controlled FluxTune run.

## §4 Overview (≈ 0.5 page)

- **Design goals:**
  - G1 no constants tuned per model, dataset or setup: settings in model-independent units (N, s, Φ_budget), so one value carries across model sizes; no per-model learning-rate or threshold search.
  - G2 clients run inference only and upload one adapter-sized update per task (FwdLLM's bytes).
  - G3 unmodified async engine, with no update depending on one cohort or on a client staying past one task.
  - Out of scope (orthogonal, combinable with FluxTune; say so to pre-empt "why not X"): a new async protocol, predicting client availability, update compression, secure aggregation.
- **Architecture (P10):** select for the bin (E4) → average perturbations (E1) → buffer N (E3) → normalized, decayed step (E2).
  - Figure: shade the existing async engine (FedBuff dispatch, buffering) grey to mark it as borrowed; only the four E1-E4 boxes are FluxTune's.
- **What each party sends:** one adapter-sized update per task, as FwdLLM.
- **Overview by example (P11):** 6 clients, 3 server updates, FwdLLM's held round vs FluxTune.

## §5 Design (≈ 2.5 pages; each: question → observation → mechanism → intuition → why not → cost → leave-one-out → generalisation)

Evaluation of each concept is **leave-one-out**: full FluxTune vs FluxTune with only that concept swapped back to FwdLLM's rule; the TTA gap is the concept's contribution.

- **Contributions (claimed, argued in this section):** the four concepts E1-E4, plus two analyses: the growth law (backs E2) and the buffer rule (backs E3).
- **Design choices (used, not claimed; settings table + Appendix):** stop rules (when a run ends), finite-difference spacing, PEFT choice (adapter / LoRA rank), and cost-only changes (save compute without changing updates needed).

- **§5.1 E1 Perturbations as samples: how should a client combine its P perturbations?**
  - Mechanism: upload the mean of all P = 10.
  - Averaging cuts noise by the number of perturbations; any picking rule leaves signal-to-noise unchanged; and the mean is unbiased.
  - Every forward pass used (picking discards 90%).
  - Why not: max-|JVP| pick (run 26, same cost); cosine pick (run 7, biased); P 40 (compute-neutral).
  - Cost: same compute and bytes as max-|JVP|.
  - Intuition: each perturbation is a noisy sample of the gradient; averaging independent samples cancels noise, picking one only rescales signal and noise together.
  - Leave-one-out: FluxTune − E1 = max-|JVP| pick (run 26; cosine pick run 7) ⚠ TTA gap.
  - Generalises: yes, no value to find. The noise cut holds by construction for any model; P = 10 is a compute choice (A.3).
  - Preview: FwdLLM + E1 alone 7× faster to 0.60.
  - **Math (keep to these):**
    - Stability: 1/P for the mean vs 1 for any picking rule.
    - Estimator: v ~ N(0, I_p), JVP z = ∇f·v, ĝ = z·v; E[ĝ] = ∇f.
    - Alignment of one perturbation: cos(ĝ, ∇f) = z / (‖∇f‖‖v‖) ≈ ζ/√p, ζ ~ N(0, 1).
    - Mean of P: ĝ_P = (1/P)Σ z_i v_i; signal adds, noise cancels → cos ≈ √(P/p).
    - Best of P by |z|: cos ≈ E[max_i |ζ_i|]/√p (≈ 1.5/√p at P = 10) vs √10/√p for the mean ⚠ reconcile with the tex's 3.35×.
    - Picking scales the gradient component and the noise length by the same |z|, so their ratio (stability) is unchanged;
      averaging cuts noise variance by P.
    - Cosine pick: E[ĝ_pick] ∝ previous aggregate, not ∇f (biased).
- **§5.2 E2 Normalized step with decay: how far should each update move?**
  - Mechanism: the aggregate sets only the direction; the step length is a set fraction of the current weight size.
  - Every update is a pure turn: measured step is sideways to the weights, so it rotates them without stretching.
  - Growth law: weight growth then follows exactly from the step sizes, so growth is a budget we choose.
  - The step fraction decays over time; the decay speed is derived from the growth budget (Φ_budget 2 / 4 / ≈ 210).
  - Set the step unit s, not the starting fraction, so the step follows model size automatically.
  - Accuracy depends on the total useful movement made toward the gradient, not on how the decay spreads that movement over time: the schedule changes how fast we get there, not where we end up.
  - Step length is independent of the cohort mix (SGD's moved 1.6×, here 0×).
  - P1: FwdLLM drops 9.6 pt after peak, FluxTune 0.4 pt.
  - Why not: SGD; constant ρ (run 21); 1/t; Adam / Yogi; momentum as a concept (cell-dependent, breaks the law).
  - Fig: [perpendicular steps](../fwdllm/figs/fig_perpendicular_steps.pdf) ⚠ re-render on runs 1, 2, 22.
  - Cost: one norm per update.
  - Intuition: noisy steps are sideways to the weights, so each one lengthens them; making the step a fraction of ‖θ‖ turns that growth into a predictable, decaying budget.
  - Leave-one-out: FluxTune − E2 = SGD step, lr 0.01 (run 8) ⚠ TTA and final-accuracy gap.
  - Generalises: the form does (relative step, scale-free s, p-free movement); the values do not fully: s (2.10 / 2.4 / 2.81) and Φ_budget (2 / 4 / ≈ 210) are set per cell.
    - Finding them: start from s ≈ 2-3 and Φ_budget from the cell's accuracy target (too small caps accuracy); first-hours check (growth off e^B > 1% or no accuracy by ~50 updates → s 1.5).
    - ⚠ replace with a rule if one exists; otherwise state as the one per-cell knob.
  - **Math (keep to these):**
    - Buffer alignment: G = mean of N client updates, cos(G, ∇f) ≈ √(P·N/p).
    - Step: Δθ_t = −ρ_t‖θ_t‖·G/‖G‖.
    - Step-unit relation: ρ0 = s·√(P·N/p). Decay: ρ_t = ρ0·e^(−t/2T_res), T_res from Φ_budget.
    - Perpendicular (measured cos(θ, Δθ) ≈ 0) ⇒ ‖θ_{t+1}‖² = ‖θ_t‖²(1 + ρ_t²); each update turns θ by arctan ρ_t ≈ ρ_t rad.
    - Growth law: Φ_T = ‖θ_T‖/‖θ_0‖ = Π_t √(1 + ρ_t²) = e^B, B = ½Σ ln(1 + ρ_t²) ≈ ½Σρ_t².
    - SGD contrast: ‖Δθ‖ = η‖G‖ and ‖G‖ tracks ‖θ‖, so ρ_t^SGD = η‖G‖/‖θ‖ is never controlled; Φ grows without bound.
    - Decay: ρ_t = ρ0·e^{−(t−1)/(2T_res)} ⇒ B_∞ ≈ ρ0²T_res/2 ⇒ Φ_∞ = e^{ρ0²T_res/2}; set T_res = 2 lnΦ_budget/ρ0².
    - Step unit: s = ρ0/cos = ρ0/√(P·N/p) ⇒ ρ0 = s·√(P·N/p) (follows p).
    - Progress: Λ = Σ_t ρ_t·cos_t ⇒ Λ_∞ = 4 lnΦ_budget/s; accuracy is a function of Λ only.
    - Model-free movement: ‖θ‖ ∝ √p and cos ∝ 1/√p ⇒ useful movement ρ‖θ‖·cos is p-free.
    - Skip: Λ(t) closed form, momentum algebra (one sentence: momentum breaks cos(θ, Δθ) = 0, so Φ ≠ e^B).
- **§5.3 E3 Fixed buffer size: when should the server aggregate?**
  - Mechanism: server update every N client updates; no variance gate.
  - Updates are independent (effective count = N), so the count measures signal exactly.
  - Buffer rule: the buffer size needed grows with the square of the progress required, and model size cancels out, so N = 50 works for every model.
  - Each update consumes exactly one data bin.
  - Why not: variance threshold (drifts, re-screened per model); adaptive buffer (C3).
  - Cost: none.
  - Intuition: independent client updates add signal linearly, so counting them measures signal exactly; variance tracks the gradient norm, which drifts, so a threshold on it drifts too.
  - Leave-one-out: FluxTune − E3 = variance threshold (run 9) ⚠ TTA gap.
  - Generalises: across models yes (p cancels, N = 50 everywhere); across datasets via the buffer rule (N* 34-42 AG News, 38 Yahoo).
    - Finding it for a new task: N* ∝ Λ_need² from the target accuracy; N = 50 is a safe upper value on tested datasets.
  - **Math (keep to these):**
    - Buffer rule: N* ∝ Λ_need² (p cancels), so N = 50 for every model.
    - Why a threshold drifts: buffer variance ≈ p‖∇f‖²/(P·N) ∝ ‖∇f‖², and ‖∇f‖ tracks ‖θ‖ ⇒ a fixed threshold means a growing N.
    - Why a count suffices: n_eff = N(1 + O(P/p)) ≈ N (client updates are independent; measured 1.000 median).
    - Buffer rule: need Λ_∞ ≥ Λ_need ⇒ cos ≥ ρ0Λ_need/(4 lnΦ_budget) ⇒ N* = p·(ρ0Λ_need/(4 lnΦ_budget))²/P.
    - p cancels: Λ_need ∝ 1/√p (same absolute movement) ⇒ N* ∝ p·Λ_need² is model-free; N* 34-42 (AG News), 38 (Yahoo) → N 50.
- **§5.4 E4 Data-bin client selection: who builds an update, and for how long?**
  - Mechanism: every iteration, pick fresh K clients for the current bin; each leaves after one task.
  - A buffer holds ~N/K separate draws, not one cohort repeated.
  - Heterogeneity (main): averages out per-client label skew.
  - Availability (secondary): a dropout costs one task, not a stalled bin.
  - Credit: scoring inside the selector is Felix's (cited).
  - Why not: cohort held per bin (one-class collapse); fresh cohort per update only (still plateaus); larger K.
  - Cost: one selection per iteration.
  - Intuition: each client's label mix is skewed; averaging over more distinct clients cancels the skew (E1's logic at client level), and one task each means no one has to stay online.
  - Leave-one-out: FluxTune − E4 = cohort held per bin ⚠ run not yet planned.
  - Generalises: no value to tune (the gain depends only on buffer size and cohort size); its size scales with how non-IID the data is ⚠ check on one IID and one skewed dataset.
  - ⚠ needs "FluxTune − E4" run + `b2` diversity figure (add to RUN_PLAN).
  - **Math (keep to these):**
    - Client label mix π_k ~ Dirichlet(α·1_L) over L classes: Var[π_kc] = (1/L)(1 − 1/L)/(Lα + 1).
    - Buffer label mix π̄ = mean over m distinct clients: Var[π̄_c] = Var[π_kc]/m.
    - Held cohort: m = K (= 10); per-iteration selection: m up to N (= 50), i.e. √(N/K) ≈ 2.2× less skew.
    - Check: Yelp-P (L 2, α 1), sd of one cohort's class share = √(0.25/3 / 10) ≈ 0.09, matching the observed 58% cohort ⚠ verify.
    - Availability: a held cohort must stay for N/K tasks per bin, i.e. N/K·D online time; selection asks for D.
- **§5.5 Running on an async engine (≈ 120 w, not a concept).**
  - FedBuff engine, C = 0.3 n, cited.
  - Churn costs one task (follows from E1-E4).
  - Stale updates need no weighting: each server update turns the weights only slightly, so a stale direction is barely off (gradient-aware weights: no gain).
  - **Math (one line):** stale by τ server updates means off by ≤ τ·ρ_t rad; E[τ] ≈ C/N server updates; each update turns θ by ≈ ρ_t, so a τ-stale direction is off by ≤ τ·ρ_t (≤ 0.6 × 0.07 ≈ 0.04 rad at C 30, N 50), far below the ~π/2 between noise and gradient.
- **§5.6 Algorithm 1 + settings table.**

  | setting | DistilBERT / RoBERTa-L / 1.7B | kind |
  |---|---|---|
  | P | 10 | set |
  | N | 50 | rule, per dataset |
  | s | 2.10 / 2.4 / 2.81 | set, per cell |
  | ρ0 | 0.07 / 0.0698 / 0.0978 | derived |
  | Φ_budget | 2 / 4 / ≈ 210 | set, per cell |
  | T_res | 283 / 570 / 1,119 | derived |
  | C | 0.3 n | deployment |
  | server momentum | 0 DistilBERT; 0.9 big cells | set, per cell (Appendix A.1) |
  | adapter size / batch size | adapters rf 16, bs 8 / LoRA r 6, bs 32 / LoRA r 2, bs 32 | set, per model / task (A.2) |
  | stop rule | saturation stop (DistilBERT); target ×2 or 30.5 h wall (big cells) | run control, same in every arm (A.4) |

  - First-hours check: growth off e^B > 1% or no accuracy by ~50 updates → lower s to 1.5.

## Later sections (claims they must back)

- **Baseline faithfulness:** FwdLLM gets every fix it needs, none of our concepts.
  - Fresh cohort per update (upstream); BitFit on RoBERTa-L; threshold and lr screened per model.
  - Its non-IID plateau is in its own paper, so DNF is expected.
- **Simulator fidelity:** real ↔ sim graded on 9 baselines (0 fails); bench ≡ harness.
  - Big models rest on measured D; FT-N21 real leg open.
- **Theory:** growth law, progress law, buffer rule; predicted vs measured.
  - Slogan: "the step schedule sets speed, not accuracy."
- **Evaluation:** E2E on 3 cells, time and GB at target, engagement CDFs, async rows.
- **Ablations (leave-one-out):** FluxTune − Ek for each of E1-E4, plotted against full FluxTune and FwdLLM; `Contri_*` runs 7-9, 26 + E4 ⚠; run 10 in the gain decomposition; `tune_*` as sensitivity.
- **Honest limits (P24):**
  - 10× forward passes per task; gap to backprop (~2.4 pt AG News; 1.7B ceiling ~0.77).
  - 1 seed vs 2.3-pt noise; s and Φ_budget set per cell; momentum breaks the exact law.
  - Big-model time assumes A40 → NPU ratios.

## Appendix (design choices without an ablation; main text only points here)

- **A.1 Server momentum 0.9 (big cells).** One line in §5.2 and the settings table.
  - Expand: 1.7B smokes (0.432 vs 0.357 plain), divergence on DistilBERT, why it breaks Φ = e^B (read Φ from the run).
- **A.2 LoRA rank and batch size.**
  - Expand: r 2 vs r 4 at fixed s (+3-6 pt by c30, same task cost); bs 32 vs 8 (same plateau, 35-40% fewer simulated hours).
- **A.3 Perturbation count P.**
  - Expand: P 40 is compute-neutral (4× cost per update for ~4× progress), so P stays 10.
- **A.4 Stop and run-control rules.**
  - Expand: saturation / accuracy-decay stop on DistilBERT, target ×2 or wall cap on big cells; identical in every arm.
- **A.5 Client weighting.**
  - Expand: gradient-aware and staleness weights vs uniform (within 0.6 pt, 14% slower per update, G1).
- **A.6 Concurrency C = 0.3 n.**
  - Expand: why it is a deployment choice (C ≤ N) and how it sets staleness (E[τ] ≈ C/N).
- **A.7 Finite-difference spacing.**
  - Expand: h·√p fixed, so the finite difference is comparable across model sizes.
- **A.8 Rejected alternatives for E2 and E3.**
  - Expand: 1/t decay, Adam / Yogi, constant ρ, ρ refit from a periodic probe, adaptive buffer; one line of evidence each.
