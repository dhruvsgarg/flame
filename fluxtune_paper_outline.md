# SwiFT paper outline (MLSys): arguments only

Format: one claim per bullet, phrases not prose; sub-bullets **must**
appear in the text. ⚠ = number or evidence pending. Rules:
[paper_template.md](paper_template.md) Parts A-B (P# = its moves; B.20
for prose). Evidence for every number below:
[run_answers_to_story_qs.md](run_answers_to_story_qs.md) (Q1–Q8, final
readout 2026-10-29 11:40), plus EXPERIMENTS.md, RUN_PLAN.md and
BASELINES.md for end-to-end cells and baseline screens (not in this
repo or at `../paper_expts_fluxtune/` on the laptop; read them on the
node).

**Status (2026-10-30).** Pitch decided by runs q1–q5 (Q1–Q8 readout in
`_metadata/temp_fxtune_abl.md`). R-runs R1a/b, R2, R3, R4a, R7a/b, R8
finished 10-29/30 on shepherd; results in §1 (E19–E24) and §10.1. R1c,
R6, R9 not run; R5 cannot run. Title, abstract, intro, background and
motivation are written in 00-abstract.tex, 01-introduction.tex and
02-background.tex (draft, not committed). 03-design.tex is still a
skeleton. The audit fixed claims the data did not carry (§0, "audit").
**Three R results change the text:** −step ties SwiFT (E19), FwdLLM +
weight decay matches SwiFT's peak (E22), and the gated system does
fall at 3× lr (E21). §11 lists the holes left.

**Figures made (2026-10-30):** Fig 1 and Fig 2a–e, from the logs; files,
runs and printed numbers in §12. Numbers in this outline were refreshed
from the same digests: **1.7B peak gain is +9.4 pt, not +9.8** (smoothed
vs. smoothed), and **the growth law holds within 2.4% on all 13 runs,
r4a included** (the "13% off" was the logged ρ, which divides by the
post-step norm).

**System name: SwiFT** (`\sys`), decided 2026-10-29. FedSiFT and
FluxTune are retired. The name changes only together with a new title.

**Writing is not blocked by any pending run.** The argument stands on
the data in §1, scoped as §0 says. Remaining runs (R1c, R6, R9, §10.2)
fill a number or a sentence; none is needed. In the .tex:
- A pending *number* → `\ph{X}` (e.g. headline ranges from R1).
- A pending *strengthening* → write the claim at its current scope
  ("on DistilBERT · AG News") and add `\tbd{R1c: if FwdLLM on
  RoBERTa-L also trails, widen to four cells}` right after it. The
  prose must read as finished without it.
- If a run comes back against a claim, the claim narrows (§0 wording);
  the spine does not change.

## 0. Guardrails: claims the data do not support

- **The run-level budget leads; buffering comes second.** The growth
  law (E1) and the collapse (E2) are the strongest data; the per-step
  cosine supports a scaling claim only (E13).
- **Never write:**
  - That FwdLLM *at shipped settings* collapses. Smoothed, it goes
    0.849 → 0.842 (−0.7 pt) by Φ 4.83 (timed out; transient dip to
    0.819 at Φ ≈ 4.1, recovered); 1.7B FwdLLM dips −1.2 pt by Φ 14.2,
    within noise. At 3× lr it does fall to chance (R4a, E21), but from
    a low peak (0.43): write it as a controlled motivation run, never
    as the baseline's result.
  - That the relative step beats SGD on accuracy or speed on
    DistilBERT · AG News: −step ties (E19). Credit it for planned
    growth and unit-free settings, not for this cell's numbers.
  - That the budget beats weight decay on accuracy (E22: ties). The
    measured difference is time (2.3× vs. SwiFT + WD; 9× vs. FwdLLM +
    WD to 0.86).
  - That the noise, or the cosine's level, is predictable. Only how
    the cosine grows with averaging, and the norm growth, are.
  - "cos ≈ √(PN/p)" as a step-quality estimate; "planned step
    quality".
  - "Variance grows because the weights grow." Variance tracks ‖g‖²
    (E7).
  - That the growth at the peak is the same for every model (E4).
  - That the budget speeds training (E6), or that it has a sharp
    optimum (E5).
  - Momentum in any shipped cell (it breaks the growth law; 6b,
    RoBERTa run 4, a22 collapse).
  - "count", "count-based", "set by counts" in any form.
  - Non-IID as a headline noise source until R6 and −select (R9)
    exist; it stays in the eval and limits.
  - (audit) That growth *causes* the fall. Untested: ln Φ ≈ ½Σρ² is
    also the total squared relative distance the noise has moved the
    weights. Write "Φ measures how much noise the run has put into the
    weights, and past a point accuracy falls with it". This is also
    what makes "budget the noise" literal.
  - (audit) "Nothing observable warns of the fall." The server sees
    ‖θ‖ every step, so Φ is observable. The claim is narrower: the
    gate's signals (variance, cosine) do not warn, and measuring Φ
    adds nothing a plan does not already know (E1).
  - (audit) That FwdLLM "peaks" at Φ 3.72. After it, FwdLLM gives back
    only 0.7 pt, a tie. Write "both stop improving at Φ ≈ 3.7".
  - (audit) "The gradient grows ×5.6." ‖g‖² grows ×5.6, so ‖g‖ grows
    ×2.4, and ‖g‖ there is the norm of the noisy estimate, not of the
    true gradient (E7).
  - (audit) "Averaging cleans a step at a known rate", or the ratio 1
    : 3.8 : 8.5. N 10's cosine (0.0010 ± 0.0010) is 1σ from zero; only
    N 50 → 90 is resolved (E13).
  - (audit) That the budget raises the peak over constant ρ (0.880 vs.
    0.868), or any ranking inside the Φ_b 2–8 band. Those gaps (≤ 1.2
    pt) are below the ~2.3-pt replicate floor (CLAUDE.md).
  - (audit) That the run ends *at* Φ_budget. Φ_budget is an asymptote;
    q2 ends at Φ 3.67 (predicted 3.62) at c1493.
  - (audit) That collapse *to chance* is shown beyond one cell. On
    Yahoo the unbudgeted run (r3a) peaks at Φ 4.1 and falls −3.2 pt by
    Φ 6.4 before the 8 h wall cap stopped it (E2): "falls past its
    peak on two datasets, to chance on AG News".
  - That the growth law applies with momentum or weight decay (logged
    Φ there is the plan, not ‖θ‖). Without them it holds within 2.4%
    on all 13 runs, steps up to ρ 0.6 (E1).
  - That a SwiFT phone's task is "short" on 1.7B: dispatch to upload
    takes a median 28.4 min (p90 57), vs. the gated hold of 83 min
    (2.9× shorter, E12). Write "one task instead of a whole server
    update".
- **Replicate floor (audit).** One seed everywhere. Claims that clear
  ~2.3 pt: the collapse (−57 pt), 1.7B +9.4 pt, DistilBERT · Yahoo
  +6.4 pt, DistilBERT · AG News +3.1 pt (just over), r4a's fall (−18
  pt). Claims below it (−step, both WD arms, Φ_b 2–8, R8 at N 90) are
  "matches" or "within noise", never wins.
- **Settings:** Φ_budget = 4 and N 50 on all three models; s per
  model; no momentum. DistilBERT cells at Φ_b 4 exist (q2, r3b).
- **Cell grid (audit).** The eval draft has four cells: DistilBERT ×
  {AG News, Yahoo}, RoBERTa-L × Yelp-P, 1.7B × Yahoo. "3 × 3" appeared
  here and in CLAUDE.md; write "four cells over three models and three
  datasets" unless all nine are run (⚠ confirm with RUN_PLAN.md).

## 1. Evidence ledger (each number has one home)

All DistilBERT · AG News unless stated; bs 8, n 100, one seed. Gaps <
1 pt are ties (raw vs. 5-eval smoothed differs by up to 0.5 pt). q1–q3
share a seed and match until their ρ schedules split. q4 is a replay
of run 1, not a second FwdLLM seed.

| # | finding | numbers | home | wording limit |
|---|---|---|---|---|
| E1 | Growth law holds | predicted vs. measured Φ (Fig 2b) within 2.4% along the whole run on 13 runs: both methods, 2 models, N 10–90, 4 step rules, steps up to ρ 0.61, up to Φ 39. q-runs within 1.6%. Final pred / meas: q1 39.1 / 39.1; q2 3.62 / 3.67; q3 5.48 / 5.56; q4 4.84 / 4.83; q5a 1.57 / 1.57; q5b 1.57 / 1.59; 1.7B 14.35 / 14.21; r2s 3.36 / 3.42; r3a 6.27 / 6.40; r3b 2.76 / 2.82; r8a 1.15 / 1.15; r8b 2.34 / 2.36; r4a 10.25 / 10.08 | §3.2 (motivation), §5.2 (law) | "without momentum or weight decay"; one seed per run. Φ = ‖θ_t‖/‖θ_0‖ (initial weights), ρ_t = ‖Δθ_t‖/‖θ_{t−1}‖; the logged `rho` and `phi` use the post-step norm and the plan, so never quote them |
| E2 | No budget → collapse to chance | q1 (constant ρ 0.07): smoothed peak 0.868 at Φ 3.69 (c532) → 0.820 at Φ 8 → 0.696 at Φ 14.6 → 0.296 at Φ 39 (chance 0.25). Slow first (−1 pt by Φ 5), then fast. Yahoo (r3a, same ρ): peak 0.672 at Φ 4.11 (c569) → 0.640 at Φ 6.4 (−3.2 pt; q1 was −1.4 pt at Φ 6); stopped by the 8 h wall cap at c751. Budgeted r3b holds (−0.3 pt) | §3.2, Fig 2a; intro ¶3 (one clause) | "the same recipe without a budget"; to chance on AG News only |
| E3 | Budget holds the peak | q2 (Φ_b 4): peak 0.880, final 0.879, never < 0.877 after peak. q3 (Φ_b 8): never < 0.865. Same seed and commits as q1 | §5.2, ablation −budget | "within 0.1–0.9 pt of peak" |
| E4 | Where unbounded growth stops helping depends on the model, not the method | DistilBERT: both stop improving at Φ ≈ 3.7 (3.69 ours, constant ρ, a real peak; 3.72 FwdLLM, the argmax of a plateau), initial ‖θ‖ 6.73 vs. 13.3 (2×). DistilBERT · Yahoo (r3a): Φ 4.11, so ≈ 3.7–4.1 on two datasets. 1.7B · Yahoo FwdLLM: still rising at Φ 8 (0.435), best 0.508 at Φ 13.2, 0.496 at Φ 14.2 | §3.2; §5.2 (why Φ, not ‖θ‖) | "the ratio matches, the norm does not"; never "set once per model" as a tuning step (Φ_budget is 4 everywhere, and on 1.7B it sits far below 13 and still wins, E15). 1.7B caveat: LoRA r 2, small ‖θ_0‖ 6.3, and its first step alone grows it 17% (ρ 0.61). Path-dependent: annealed q3 is best at Φ 4.7 |
| E5 | Budget band is broad | peak by Φ_b: 2 → 0.872 (0.87 at c1061); 4 → 0.880 (0.87 at c609); 8 → 0.875, −0.9 pt by Φ 5.5; ∞ → 0.868, collapses | §5.2 sensitivity; eval sensitivity | "a broad band", never "tuned optimum"; large side rests on q1 |
| E6 | Budget is not a speedup | to 0.86: q1 2.9 h, q2 3.4 h, q3 3.3 h, run 2 4.4 h. q1 never reaches 0.87; stays ≥ 0.86 only c385–c631 | §5.2; eval | "costs ≤ ~15% time to 0.86; removes the dependence on stopping at the right commit" |
| E7 | Gate tracks the update's size; relative to that size the noise does not change | FwdLLM (q4, Fig 2c) first-iteration variance ×5.7 (2.55 → 14.5) while ‖g‖² ×5.7 (‖g‖ 2.4k → 5.7k, ×2.4); variance ÷ ‖g‖² flat (0.99–1.00 of its start); iterations per server update 9 → 34 → 56 (first / middle / last 10%). Ours: var/‖g‖² 1.66e-5 → 1.67e-5 over q1 while var ×11. Earlier: AG 4.9× / 4.8×; Yahoo 18× / 18×; 1.7B 0.57× / 0.57× | §3.3 (gate); §3.1 Opportunity and §5.1 (the noise share of a step is the same early and late, so its averaging can be fixed before training) | "variance tracks ‖g‖²"; ‖g‖ = norm of the client update / buffer mean, not the true gradient; never "tracks the weights" |
| E8′ | Gate's committed variance is flat by construction | q4 `var_at_commit` 0.29 throughout (threshold 0.3) | §3.3 (one clause) | it carries no information |
| E9 | Alignment does not flag collapse either | q1 cos 0.005 (Φ < 1.6) → 0.017–0.019 (Φ 12–35) as accuracy falls; true ‖∇f‖ 0.4 → 5.2. Stable arms 0.002–0.007, no trend | §3.3; §5.2 (why a budget, not a monitor) | 25–30 audits per arm, noisy; "does not warn", not "is useless" |
| E10 | Gate costs work for no better aim | forward passes per server update (iterations × 10 clients × 1 perturbation × 2; FwdLLM ranks its 10 candidates by cosine with the previous update on the CPU and runs inference on the top one only, as in FwdLLM's code): gated median 660, rising 180 → 1,140 (first / last 50 server updates, q4) vs. SwiFT's fixed 1,000 (N 50 × P 10 × 2). Cosine: gated median 0.0030 over 25 audits, no trend (0.0033 first half, 0.0030 second) vs. 0.002–0.007 ours | §3.3, §3.4 | "about the same measured cosine"; the gate's extra work is its ~6× growth over a run, not its level (it starts at 0.18× SwiFT's). Forward passes, never "iterations", across systems |
| E11 | Thresholds and lr do not transfer | FwdLLM shipped: DistilBERT var 0.3, lr 0.01; 1.7B var 8.4, lr 3e-3 (28× threshold). ⚠ RoBERTa-L's 0.2 → zero server updates on 1.7B (re-verify) | §3.3 | per-model screens are in BASELINES.md |
| E12 | Held cohort per server update | median hold (simulated, Fig 2e): gated DistilBERT 2.9 min (p90 6; run 1, ~30 iterations); gated 1.7B 83 min (p90 150, max 273; ~8 iterations). Reconciled: each iteration waits for the slowest of 10 phones, median 511 s vs. a 255 s median task, so a phone computes ~34 min of its 83 min hold. One task: DistilBERT 2 s, 1.7B 255 s (gated); SwiFT 1.7B dispatch → upload median 28.4 min (p90 57), so 2.9× shorter than the hold (Fig 1b). MobiPerf sessions (`states_2st`, n 6,742): median 10.2 min, 50% ≤ 10 min; shorter than the hold: 40% (DistilBERT), 83% (1.7B); shorter than SwiFT's 1.7B task: 64% | §3.4; Fig 1b, 2e | lead with 1.7B; never use q4's 32 min (vclock inflated); RoBERTa-L ~20 min is lr screens only, not plotted. Compare like with like: a gated phone is held the whole hold, a SwiFT phone one task. "50% of sessions ≤ 10 min", not REFL's "70%" |
| E13 | Averaging more raises the cosine, no slower than √N | early phase (Φ ≤ 1.6): N 10 / 50 / 90 → cos 0.0010 ± 0.0010 / 0.0038 ± 0.0011 / 0.0085 ± 0.0010 vs. theory 0.029 / 0.065 / 0.087. Resolved step: N 50 → 90 raises cos ×2.2 vs. √(90/50) = 1.34 | §5.1 only (one clause in §3.1) | "scaling, not level"; level 10–15× below the isotropic bound; part may be the 1024-sample probe; never print 1 : 3.8 : 8.5. Firm up with the offline cosine bench (§10, R6) |
| E14 | N has a cheap end; the expensive end pays off only with ρ0 rescaled | Same step size (q5a / q5b, ρ0 0.07, not rescaled): N 10 peak 0.64 raw, var 0.23 vs. 0.045; N 90 0.852 at c300 (≈ N 50 per server update) but 3.7 h vs. 2.1 h per 300. ρ0 = s√(PN/p) rescaled (R8): N 10 (ρ0 0.031) peak 0.589, final 0.46, no better; N 90 (ρ0 0.094) 0.868 at c300, to 0.85 / 0.86 in 1.9 / 2.6 h vs. N 50's 2.3 / 3.4 h | §5.1 (why N 50); tension (per-step ends) | DistilBERT only; R8 capped at 300 server updates, so N 90's peak is unknown. N 90 rescaled is 17–24% sooner, a tie in accuracy: N 50 is "enough", not "the efficient point" |
| E17 | (audit) The update is perpendicular to the weights | no new run: the law's residual bounds the radial part. ln Φ_meas − ln Φ_pred ≈ Σρ_t cos(θ,Δθ); q1: max residual 0.011 over Σρ = 105 → mean cos(θ,Δθ) ≲ 1e-4 | §3.2 (replaces the ⚠ re-measure of \|cos(θ,Δθ)\|) | "on average"; bounds the mean, not each step |
| E18 | (dropped) Real availability trace | R5 cannot run: the fwdllm aggregator has no unavailability wiring | — | delete the trace clauses (intro ¶7, contribution 4, §3.4) |
| E15 | 1.7B end to end (Fig 1a) | smoothed vs. smoothed: SwiFT 6c 0.602 (raw max 0.606; Φ_b 4, ρ0 0.0978, T_res 290, no momentum) vs. FwdLLM 0.508 (c473): **+9.4 pt**. To 0.508: SwiFT 117 h / 30.2 GB vs. FwdLLM 721 h / 90.4 GB → **6.2× sooner, 3.0× fewer GB**. FwdLLM ran 844 simulated h (c551, interrupted), SwiFT 423 h | eval; intro ¶7; Fig 1 | never +9.8 (raw vs. smoothed) |
| E16 | DistilBERT end to end (R1a, R1b) | AG News: q2 0.880 vs. FwdLLM run 1 0.849 smoothed (+3.1 pt); to 0.849: 2.3 h vs. 16.0 h (7.0×); 0.86: 3.4 h vs. never (DNF 27.0 h, ≥ 7.9×). Yahoo: r3b 0.680 (c721, still rising, wall cap c746) vs. run 28 0.616 (DNF 46.7 h): +6.4 pt; 0.65 at 3.5 h vs. never | eval; intro ¶7 | q2 and run 1 share bench, divisors, recipe (no rerun needed). ⚠ bytes to FwdLLM's peak (AG News, Yahoo); ⚠ Yahoo time to 0.616; check run 28 and r3b used the same eval rows (r3b: 6,000 resampled) |
| E19 | −step ties SwiFT (R2, r2s) | SwiFT buffering + SGD lr 0.01: peak 0.877 (c991, Φ 3.18), final 0.871 (full test 0.868), plateau stop c1123 at Φ 3.42; to 0.85 / 0.86 / 0.87: 2.5 / 3.5 / 6.0 h vs. q2 2.3 / 3.4 / 5.3 h. Its ρ = η‖G‖/‖θ‖ falls by itself 0.079 → 0.034 (‖θ‖ ×3.4, ‖G‖ ×1.4), close to q2's planned 0.070 → 0.019 | §5.2 ablation; §2 baseline argument | tie. lr 0.01 is FwdLLM's screened lr for this cell, so −step inherits a tuned lr; its Φ is unplanned and still growing. §2's "the others are our −step arm" now argues *for* FedKSeed / FedMeZO-like fixed-lr baselines being competitive here: reframe (§6) |
| E20 | −buffer learns far slower (R2, r2b) | pick 1 of 10 + variance gate 0.3 + held cohort, SwiFT's step (Φ_b 4): peak 0.603 smoothed (c151, raw 0.676), then ≥ 5 pt under peak → guard stop c210 (Φ 1.53); 3.35 h for 210 server updates, 8 iterations each. q2 at c200: 0.785 / 1.3 h | §5.1 ablation | stopped early by the guard and noisy (0.50–0.59 at the end): write "learns far slower", not "diverges". Trainable set is FwdLLM's (‖θ₁‖ 13.3 vs. 6.7) |
| E21 | The gate does not bound growth (R4a) | FwdLLM lr 0.03 (3×), var 0.3: first update ρ 0.58 (Φ 1.16 in one step); peak 0.432 smoothed (Φ 3.45, c33; raw 0.539) → chance (≤ 0.250) at Φ 7.8 (c181) → 0.257 at Φ 10.1 (c301, full test 0.250). ρ at Φ 2 / 4: 0.30 / 0.15 (shipped q4: 0.10 / 0.05). Iterations per server update 21 → 48; ‖g‖ 1.08k → 1.63k; cos 0.003 → 0.039 at Φ 4.2, 0.01–0.03 at chance | §3.3 brake paragraph; Fig 2a arm; intro ¶4 cost 1 | from a low peak; a controlled motivation run, not a baseline. Cos again does not warn (E9) |
| E22 | Weight decay is a brake, not a speedup (R7) | FwdLLM + WD 0.0026 (r7a): ‖θ‖ pinned at Φ 2.65 from c~600; peak 0.864 (c926), final 0.849 full test; 0.85 / 0.86 at 15.0 / 30.4 h (shipped run 1: never); iterations per server update 9 → 23 (shipped 9 → 56), ‖g‖ 730 → 1.15k (shipped → 1.75k). Constant ρ + WD ρ²/2 (r7b): Φ pinned 1.08, peak 0.864, holds; 0.86 at 7.7 h vs. q2 3.4 h (2.3×), never 0.87 | §5.2 design space; eval "obvious fixes" | peaks tie SwiFT (−1.6 pt); claim time only. WD also slows FwdLLM's iteration growth, so §3.2's 9 → 56 is "at shipped settings" |
| E23 | Settings transfer across datasets (E8, R3b) | DistilBERT · Yahoo with AG News's s, N, Φ_b, T_res: holds and beats FwdLLM by 6.4 pt (E16) | title rung 2; G2 | one model; s across models still per model |
| E24 | Averaging scaling, second read (R8) | early cos (Φ ≤ 1.6): N 10 rescaled 0.0027 ± 0.0009 (q5a 0.0010 ± 0.0010); N 90 rescaled 0.0080 ± 0.0012 (q5b 0.0085 ± 0.0010) | §5.1 (E13 caveat) | N 10's cosine depends on the run; still "scaling, not level" |

Not measured, and the prose must not lean on it: cosine on a model
other than DistilBERT (1.7B audit stalled); Φ at peak on a second
dataset; best-of-P vs. mean on the bench; cosine with the population
gradient, IID vs. skewed (R6); −select (R9); FwdLLM on the async
engine (C4); collapse to chance on a second cell; FedAdam; bytes to
FwdLLM's peak on the DistilBERT cells; FwdLLM at
shipped lr past Φ 4.8 (R4b); a second seed of anything.

## 2. Names and vocabulary

**Vocabulary** (internal → paper term → first defined). As CLAUDE.md,
with these updates:

| internal | paper term | first defined |
|---|---|---|
| probe, P | perturbation; P per client update | intro ¶2 (intuition); §2 |
| upload / commit | client update / server update | §2 |
| pool, N | buffer, buffer size N | §2 (FedBuff) |
| alignment, `cos_ground_truth` | cosine with the true gradient | intro ¶2; §2 |
| trust-ratio step, ρ | relative step ρ = ‖Δθ‖/‖θ‖ (trust ratio once, LARS / LAMB) | §2 |
| Φ | weight-norm growth Φ = ‖θ_t‖/‖θ_1‖ | intro ¶3 (words); §2 (symbol) |
| Φ_b, `budget` | growth budget Φ_budget | intro ¶6 (words); §5.2 |
| var threshold | **variance gate** (FwdLLM's variance threshold) | intro ¶4; §2 |
| gate vs. budget | *gated* control (measure, then step) vs. *budgeted* control (planned before training) | intro ¶4–¶5 |
| lr, threshold | gradient-unit settings | intro ¶4 |
| data bin | shared minibatch | §2 |
| vclock, TTA | simulated time, time to accuracy | §2 |
| `exp`, T_res | step decay; decay constant | §5.2 |

- **Banned:** "count(s)" in any form; "ledger"; "inflation";
  "alignment" (use cosine); "commit"; "pool"; "proactive vs. reactive"
  without the proxy argument (§3 C-core).
- **"Budget the noise"** has two levels, always named the same way:
  - *per step:* how many perturbations and client updates a step
    averages, fixed before training (the buffer);
  - *per run:* how much the weights may grow, fixed before training
    (the growth budget).

**Mechanisms (outline only; never a table in the paper).** Reused
verbatim in ablation legends (P14). The names were chosen to echo
"SiFT" in FedSiFT; with SwiFT the echo is gone (open decision, §11).

| level | mechanism | removes | FwdLLM | SwiFT | ablation |
|---|---|---|---|---|---|
| per run | **signal-calibrated steps** (relative step + growth budget) | unbounded growth; gradient-unit lr | SGD, fixed lr; growth unplanned | Δθ = −ρ_t‖θ‖G/‖G‖, ρ_t decays so Φ ends at Φ_budget | −budget (constant ρ, q1: to chance); −step (SGD lr 0.01, r2s: ties, E19) |
| per step | **signal-sized buffering** (mean of P + fixed buffer N) | variance gate; held cohort; best-of-P pick | pick 1 of P; step when variance < threshold; cohort held per shared minibatch | mean of P; step every N client updates; any arriving phone | −buffer (gate + pick + held cohort, r2b: far slower, E20); sub −avg (pick, runs 7 / 26, ⚠ L11); stability −N↓ (N 10, q5a / r8a) |
| (inherited) | cohort-free participation | — | — | FedBuff arrivals | −select (control, R9, not run) |

- −budget is the strongest ablation (q1). −buffer is credited (E20).
  −step is not, on this cell (E19): the step rule's case rests on
  planned growth (E2 vs. q2) and on no lr to screen, not on accuracy.
- −N↓ (q5a: N 10, peak 0.64) wastes progress without diverging: Φ
  1.51, law within 0.3%.

## 3. Spine

Top to bottom, the paper's argument. Each step stands without the
next.

1. **Setting.** On-device LMs should learn from their users; FL is
   how. Backprop does not fit phones; forward gradients do (inference
   only). FwdLLM showed it works.
2. **Problem.** It works under lab conditions only: a configuration
   tuned per model (threshold 0.3 vs. 8.4), phones held online for
   minutes to over an hour per server update (83 min on 1.7B), and a
   run that slows as it trains (9 → 56 iterations per server update) and stops
   improving well below what the same work could reach (0.849 vs.
   0.880; 0.616 vs. 0.680; 0.508 vs. 0.602).
3. **Root cause.** Two facts about forward-gradient noise:
   - (a) an update is almost all noise, and the noise is perpendicular
     to the weights (E17), so every step lengthens them. Growth thus
     measures how much noise the run has put into the weights;
   - (b) past some growth, accuracy falls; with nothing to bound it,
     all the way to chance (E2). Existing systems *gate* on a measured
   variance. Variance tracks the update's size (E7), not its direction
   or the weights' growth. So the gate neither plans nor bounds (b).
   What keeps FwdLLM short of the fall is its fixed lr, whose relative
   step shrinks as the weights grow (q4: 0.19 → 0.04); at 3× the lr
   the gate still passes every server update and the run falls to
   chance (E21, R4a). The gate also demands more work per step as
   updates grow (180 → 1,140 forward passes per server update; no
   brake: r4a's iterations grew 21 → 48 and it fell anyway), and it holds a cohort to take its measurement.
4. **Insight (C-core).** The noise a run spends does not need
   measuring: the growth it causes follows from the step sizes the
   server picks (E1: within 2.4% on 13 runs, up to 39× and ρ 0.6),
   and how much noise a step
   carries follows from how much it averages, the same early and late
   (E7 ratio). The signals a gate can measure tell the server neither:
   variance tracks ‖g‖², and the cosine *rises* as the model degrades
   (E9). So the noise can be budgeted before training, not gated
   during it.
   - Novelty: "known mechanism, new regime, new consequence". Norm
     growth from perpendicular steps is known for normalized networks
     (Arora et al. 2019; van Laarhoven 2017; Kosson et al. 2024 ⚠
     verify). New: at > 99% noise it is exact and set by ρ alone, it
     carries a run to chance, and it is why gradient-unit settings
     drift.
5. **Tension (P1), two levels, both ends measured.**
   - Per run: each step makes progress and also spends growth. Spend
     too little and the run is slow (Φ_b 2: 75% more server updates to
     0.87; its −0.8 pt is a tie). Spend without limit and accuracy
     collapses (E2).
   - Per step: average too little and a step is mostly noise (N 10
     peaks at 0.64); too much and each step waits on many phones (N
     90: 1.6× the time, no gain per step).
   - Dynamics: the update's norm grows ×2.4 within one run (‖g‖² ×5.6,
     E7), so any setting in gradient units means something different
     late than early.
   - Figure for P1 (audit): the per-step tension already has both ends
     measured. Accuracy at 300 server updates and simulated time per
     server update vs. N (10 / 50 / 90), one plot; candidate for §3 or
     §5.1.
6. **Two ideas, one per level.**
   - *Budget the run:* step a fraction ρ of the weights' size; ρ
     decays so growth ends at a set Φ_budget.
   - *Budget the step:* average every perturbation, update after a
     fixed N client updates from any available phone; no variance
     gate.
7. **Resolution.** Growth is planned, so the peak holds (E3). Work per
   step is fixed, so no client is held past one task and no setting
   drifts. Settings: N 50 and Φ_budget 4 on every model; s per model.
8. **Results.** Higher peak than FwdLLM, sooner, fewer bytes, four
   cells over three models and three datasets.

## 4. Title and abstract (≈ 160 w)

- **Title ladder** (name + what it does + condition; B.0):
  1. Now (in common/macros.tex): "SwiFT: Practical Federated
     Fine-Tuning with Forward Gradients by Budgeting, Not Gating, the
     Noise".
  2. E8 held on DistilBERT (r3b, E23); usable if the authors accept
     one model as the evidence: "SwiFT: Federated Fine-Tuning with
     Forward Gradients, Without Hand-Tuning or Long-Lived Clients".
  3. "Tuning-Free…" only if s also transfers across models. "LLM" only
     if the 1.7B cell is a headline.
- **S1 Context:** phones run language models that should learn from
  their users; backprop does not fit, forward gradients need only
  inference.
- **S2 Problem:** each forward-gradient update is almost all noise;
  noisy steps lengthen the weights, and past some growth accuracy
  falls.
- **S3 Why existing systems fail:** they gate each step on a measured
  variance, which tracks the update's size, not its quality or the
  weights' growth. The gate demands more work as training proceeds,
  needs per-model tuning, holds phones online for over an hour per
  step, and never bounds growth.
- **S4 Insight:** weight growth follows exactly from step sizes, so it
  can be budgeted before training, and so can how much each step
  averages.
- **S5 System:** SwiFT steps a set fraction of the weights' size
  within a growth budget, and updates after a fixed number of client
  updates from any available phone.
- **S6 Results (P3):** vs. FwdLLM on four cells (three models, three
  datasets): peak accuracy +3.1 to +9.4 pt (AG News +3.1, Yahoo +6.4,
  1.7B +9.4; RoBERTa-L ⚠ R1c); FwdLLM's peak 6.2–7.0× sooner (1.7B,
  AG News; Yahoo ⚠); 3.0× fewer bytes on 1.7B (DistilBERT ⚠). Without the budget, the same recipe collapses to chance.
- Written (176 words, six sentences); holes in §11.
- Don'ts: no "predict", no "count", no cosine number, no
  "tuning-free".

## 5. §1 Introduction (≈ 1.25 pages; ≤ 3 numbers)

**Numbers allowed in the intro:** (1) the collapse (0.87 → 0.30,
chance 0.25) or "to chance"; (2) the 1.7B hold (83 min); (3) headline
ranges. Nothing else (no "3B", no "1000×"). Written, ≈1.4 pages in the
draft build (budget 1.25). It follows Dhruv's earlier comments: reach
the problem sooner; one paragraph for "why forward-gradient FL"; start
from FwdLLM, not backprop; alternatives to backprop in §2; define
noise before using it; no analogy; tie each metric to who pays. FwdLLM
is named once in the intro body (¶1); later it is "the gated
baseline".

- **Fig 1** (`fig:intro-teaser`, made: `fluxtune_paper/figs/intro/fig1_teaser.pdf`,
  one column; 1.7B · Yahoo). (a) Accuracy vs. simulated hours, SwiFT
  vs. FwdLLM: "+9.4 pt peak", dashed line at FwdLLM's peak (50.8%) with
  an arrow to SwiFT's crossing, "6.2× sooner, 3.0× fewer GB". (b) CDF of
  time online per contribution: FwdLLM's hold (median 83 min) vs.
  SwiFT's one task (28.4 min), over MobiPerf sessions: "2.9× shorter".
  Caption: x cut at 3 h (max hold 273 min); both curves 5-eval
  smoothed.
- **¶1 Stakes, then the route (one paragraph; P4).**
  - Phones ship LMs (Apple ~3B + LoRA; Gemini Nano ⚠ cite); the data
    to adapt them is private and on the phone → FL. FL ships for
    keyboard LMs (Gboard); LLM fine-tuning is the next step.
  - Quality is user-visible and a moving target → reach accuracy fast,
    and again.
  - Backprop needs several× inference memory and NPUs run only
    inference (one clause, cite FwdLLM). Forward gradients need only
    inference: FwdLLM, MeZO, SPRY. Name FwdLLM here as the system that
    made it work.
  - Trend: models ~1000× larger than keyboard LMs; sessions short
    (idle, charging, Wi-Fi; REFL).
- **¶2 How forward-gradient FL learns, and its metrics.**
  - Topic sentence defines noise: "A forward gradient is a guess." A
    client nudges the weights along a random direction (a
    perturbation), runs inference twice, and multiplies the slope by
    the direction.
  - The guess is right on average but almost all noise: one random
    direction among ~a million weights is nearly perpendicular to the
    true gradient. No number here (the cosine level is §3.1, scaling
    only).
  - Averaging many guesses cleans the step: a client averages its
    perturbations into a client update; the server averages client
    updates. Define perturbation, client update, server update.
  - Metrics, each with who pays: time to accuracy (data drifts and
    sessions are short); bytes to accuracy (uplinks are metered);
    device work (forward passes per server update, the phone's
    battery); an accuracy peak that holds (accuracy given back is
    phones' work wasted; E2 makes this real, answering Aish's
    comment).
- **¶3 The problem as a tension (P1), bold sentence:** "Every noisy
  step makes some progress and also lengthens the weights, so a
  forward-gradient run must decide, before each step and over the
  whole run, how much noise to spend."
  - Why steps lengthen the weights, in words: the noise points in a
    random direction, which is nearly perpendicular to the weights, so
    each step walks outward (one clause; geometry in §3.2).
  - Over the run: too little growth and training is slow and peaks
    low; too much and accuracy falls, and without a bound it falls to
    chance (number 1).
  - Per step: average too little and the step is mostly noise; too
    much and each step waits on many phones.
  - Dynamics: the update grows during training, so the right setting
    in gradient units moves within a run and across models.
- **¶4 Why existing systems fall short: they gate (one assumption,
  three costs; B.1 ¶4).** Critique the choice, not FwdLLM.
  - Assumption: control must be measured at run time. They gate: run
    more perturbations until the variance across one cohort's updates
    falls below a threshold, then step with a fixed lr (inherited from
    backprop FL, where updates are mostly signal).
  - Cost 1, wrong quantity: variance tracks the update's size, not
    whether a step points the right way or how far the weights have
    grown. The gate neither plans growth nor bounds it; FwdLLM avoids
    the fall only because its next cost slows it down.
  - Cost 2, drift and tuning: as updates grow, the gate demands ever
    more work per step, and its threshold and lr do not carry over
    between models; each model needs its own search, on users' phones.
  - Cost 3, held phones: the variance is measured on the same phones,
    so they stay online until it passes, up to 83 min per step on a
    1.7B model (number 2), against sessions of minutes.
  - Async alone does not help: the failure is in what is measured, not
    in when updates arrive. (Defence detail §3.4 and eval; answers
    Dhruv's async comment.)
- **¶5 Insight (P2), one sentence:** "The noise a run puts into the
  weights shows up as growth that follows exactly from the step sizes
  the server chooses, so the noise can be budgeted before training
  instead of gated during it."
  - Contrast for the reader: a gate reacts to a proxy; a budget plans
    the quantity that matters. Name both: proxy = variance (the
    update's size); true quantities = growth (exact law) and how much
    a step averages (fixed in advance; more averaging, a cleaner
    step).
  - One clause on novelty: norm growth from perpendicular steps is
    known in normalized networks (cite); what is new is that in
    forward- gradient training it is exact, set by ρ alone, and
    decides accuracy.
  - Why it holds everywhere: it follows from the geometry of noisy
    steps, not from the model or dataset; the law held for both
    methods and two models (no number; §3.2).
- **¶6 Two ideas (no formulas, no constants beyond words; each mapped
  to a cost in ¶4):**
  - *Budget the run* (costs 1, 2): each step moves a fixed fraction of
    the weights' size, shrinking over the run so total growth ends at
    a chosen budget. Growth is planned, so the peak holds; the
    fraction is unit-free, so it does not drift with the gradient.
  - *Budget the step* (costs 2, 3): each client averages all its
    perturbations; the server updates after a fixed number of client
    updates, from whichever phones are available, each running one
    short task. Nothing is measured, so no phone is held and no
    threshold drifts.
  - Close: the settings are unit-free; N and Φ_budget are the same on
    every model, only s is set per model; the async engine is standard
    (cite FedBuff).
- **¶7 Results.** Four cells: DistilBERT × {AG News, Yahoo}, RoBERTa-L
  × Yelp-P, SmolLM2-1.7B × Yahoo. No trace clause (R5 cannot run).
  - Order: accuracy → speed → bytes, as ranges (number 3). Best cell
    1.7B: +9.4 pt peak, 6.2× sooner to FwdLLM's peak, 3.0× fewer GB.
    DistilBERT · AG News: 7.0× to FwdLLM's peak.
  - DNF rule: time to FwdLLM's peak, plus "≥ X×" at a target FwdLLM
    never reaches.
  - Accuracy holds after the peak; without the budget it collapses; a
    phone is held for one short task instead of a whole server update
    (no number in the intro; SwiFT's 1.7B task is 28.4 min median, so
    never "short task" on 1.7B; E12). Never "0.6 min" here.
- **¶8 Contributions (ideas, not "we built"; three plus results):**
  - **Diagnosis:** in forward-gradient FL, each step is almost all
    noise and perpendicular to the weights, so the noise a run spends
    shows up as weight-norm growth that follows an exact law of the
    step sizes; past some growth accuracy falls, to chance without a
    bound. The variance gate measures the update's size, so it never
    bounds growth, demands more work as training proceeds, and holds a
    cohort (§3).
  - **Signal-calibrated steps:** relative steps within a growth budget
    plan growth before training and keep the peak; the budget works
    over a broad band (§5.2).
  - **Signal-sized buffering:** averaging all perturbations and
    updating after a fixed number of client updates replaces the gate,
    the pick and the held cohort; the noise share of a step stays the
    same over a run, so its averaging can be fixed before training
    (§5.1).
  - **Results:** four cells; leave-one-out ablations, including the
    unbudgeted run (−step ties, so say "each mechanism is ablated",
    not "credited"); obvious fixes (weight decay); simulator checked
    against real runs (§6, ⚠ PARITY data for the current engine,
    §10.4).

## 6. §2 Background (≈ 0.4 page including Table 1)

**Budget (2026-10-30).** The ledger's rule 7 gives background and
motivation 1.5 pages together, not 0.5 + 1.5. Plan: background ≈ 0.4
page (≈ 350 words + Table 1), motivation ≈ 1.1 pages (≈ 650 words +
Fig 2, five panels in one row). The notation table moves to the appendix. The old
draft was ≈0.85 page with both tables. Follows Dhruv's earlier
comments: brief FL primer, focus on forward-gradient training; define
iterations, shared minibatch, server update here; justify FwdLLM as
the only baseline.

- **FL in two sentences:** server model; clients train locally and
  send updates; server aggregates. Terms (cited): client selection;
  sync vs. buffered async (FedBuff); concurrency C; buffer, buffer
  size N; server update; staleness τ; non-IID (Dirichlet α).
- **Why forward gradients** (moved from the intro): three routes
  around backprop memory, one clause each (fewer weights keeps
  backward; recompute / split keeps backward ops or ships
  activations); forward gradients: inference memory (7.7 GB backprop,
  RoBERTa-L), NPUs 30× faster but inference-only, short work units, no
  client drift, near-backprop accuracy (FwdLLM, MeZO).
- **Estimator (definitions only):** z = ∇f·v from two forward passes;
  ĝ = z·v, unbiased; cos ≈ 1/√p per perturbation (⚠ cite); averaging k
  independent → √(k/p). Contrast with backprop: backprop points along
  ∇f, ĝ mostly along v. This is "noise" in the paper.
- **Relative step and growth, one line each:** ρ = ‖Δθ‖/‖θ‖ (trust
  ratio, LARS / LAMB); Φ = ‖θ_t‖/‖θ_1‖. No law here (§3.2).
- **How gated forward-gradient FL runs (FwdLLM; no figure):** a cohort
  of K clients gets a shared minibatch; each *iteration* every client
  runs P perturbations and uploads its best; iterations repeat on the
  same cohort until the variance across uploads falls below a
  threshold; then one SGD step (server update). Define iteration here
  (Aish).
- **Shared minibatch:** bin k = every selected client's own k-th
  batch; all client updates in a server update use it.
- **Metrics:** peak accuracy that holds; time to accuracy on a
  simulated clock; bytes to accuracy; active computation time per
  phone (the ledger's fourth metric, replacing "device work"). The
  target is never a system input.
- **Table 1, positioning** (`tab:positioning`; rows grouped per step /
  per run; FedKSeed and FedMeZO cells are from the papers, ⚠ verify
  against their code):

  | concern | FwdLLM | FedKSeed | FedMeZO | MeZO (central) | FedBuff (backprop) |
  |---|---|---|---|---|---|
  | combine perturbations | pick 1 of P | 1 per local step | 1 per local step | average | – |
  | when to step | variance gate | after local steps | after local steps | – | every N |
  | who contributes | cohort held per minibatch | sampled per round | sampled per round | – | async arrivals |
  | step size | SGD, fixed lr | SGD, fixed lr | SGD, fixed lr | SGD, fixed lr | SGD |
  | weight growth | unplanned | unplanned | unplanned | unplanned | – |

  - (2026-10-30) Row "weight growth": "unplanned", not "unbounded". A
    fixed lr does bound growth in practice by shrinking ρ as ‖θ‖ grows
    (q4, r2s), and at a larger lr it does not (r4a). What no system
    does is choose where growth ends.
  - Blind spot: no system plans weight growth, or sets the averaging
    per server update before training. Why FwdLLM is the only
    baseline: it is the only forward-gradient FL system built for
    phones that sets each server update's averaging from the noise.
  - (2026-10-30, E19) Write in §2, plainly and without a claim:
    FedKSeed / FedMeZO step with fixed-lr SGD on a fixed average; the
    nearest design in our runs is the −step ablation (eval), so the
    reader is told where that design is tested. Do not write that it
    is weaker: −step ties SwiFT on DistilBERT · AG News at the lr
    screened for that cell (0.01), and that tie is stated in the
    ablation, not here. What §2 may say: fixed-lr systems leave growth
    unplanned and their lr is a gradient-unit setting (E11, E21).
    FedKSeed and FedMeZO run several local zeroth-order steps then
    FedAvg (⚠ verify), so they also carry client drift; one clause.
  - Remaining risk: a reviewer asks for FedKSeed / FedMeZO directly.
    The only answers are a run (§10.2 item 2: −step at the DistilBERT
    lr on a second model, or FedKSeed / FedMeZO themselves), or a
    limits sentence. No text can close it.
- **Notation (small table, `tab:notation`):** p, P, N, K, C, t (with
  shared minibatch t mod T_bins), ρ, Φ = ‖θ_t‖/‖θ_1‖, Φ_budget. Only
  terms used again. Candidate to move to the appendix if the front
  matter must shrink.
- **Not here:** any measurement (§3).

## 7. §3 Motivation: the case for planning (≈ 1.1 pages)

**Rewritten 2026-10-30 with the R results.** The previous version is
in git history. Budget: ≈ 650 words + Fig 2 (one full-width row of
five panels, made; §12); see §6 for the 1.5-page split.

**Labels.** `sec:motiv`; subsections `sec:motiv:growth`,
`sec:motiv:gate` and `sec:motiv:hold`, the three the approved intro
cites. `sec:motiv:noise` is dropped: its one number (cosine ≈ 0.003)
moves to the opening paragraph (⚠ ledger home change, needs approval).
Figure: `fig:motiv`, one full-width row of five panels: 2a
`fig:motiv-collapse`, 2b `fig:motiv-growth`, 2c `fig:motiv-gate`, 2d
`fig:motiv-rho`, 2e `fig:motiv-hold`. Files and numbers: §12.

**Order follows the intro:** growth (¶4) → gate (¶5, cost 2 and the
attribution) → hold (¶5, cost 1). Growth leads and is the longest
subsection, because it is the biggest gap and the newest finding
(B.3).

**Rules.** Each subsection has a claim title, then an inline setup,
its figure, and an "Opportunity." sentence that names the gap, never
our mechanism (B.3). Two runs are controlled experiments (P9), framed
as such and never as SwiFT or as a baseline result:
- q1 is "relative-step SGD with no bound";
- r4a is "the gated system at 3× its learning rate".

The ablations stay out of §3 because they reveal the mechanisms: r2s
(−step), r2b (−buffer), r7a/b (weight decay) and r8 (N with ρ0
rescaled). Setup for all panels: DistilBERT · AG News, one seed, bench
engine, unless stated. ⚠ State the partition (the ledger's D8 says all
evaluation is non-IID; confirm the q/r arms' Dirichlet α, or say IID
here).

**C-core (revised by E21).** The noise a run puts into the weights
shows up as growth, and the step sizes predict that growth. The gate's
signals (variance, cosine) track neither the growth nor the fall.
FwdLLM stays off the fall only because its fixed lr shrinks its
relative step; at a 3× lr the gate still passes, and the run falls to
chance. So growth must be planned; measuring does not control it.

- **Opening ¶ (no subsection, ~3 sentences).**
  - Every update is almost all noise: the gated step's median cosine
    with the true gradient is 0.003 (1024-sample backprop gradient,
    audited every 25–50 server updates; E9, E10).
  - So over 99.99% of each step's squared length is noise. That noise
    has two consequences, one per subsection: it grows the weights
    (§growth), and the gate's attempt to measure it holds phones
    (§gate, §hold).
  - No averaging-scaling result here (E13 and E24 are §5.1; they
    reveal the buffer).

- **§3.1 `sec:motiv:growth`: Noisy steps lengthen the weights by a law
  of the step sizes, and past some growth accuracy falls.**
  - Geometry, in words first: the noise is nearly perpendicular to θ,
    and a perpendicular step lengthens a vector. Then ‖θ+Δθ‖² ≈
    ‖θ‖²(1+ρ²), so Φ_T ≈ Π√(1+ρ_t²) and ln Φ ≈ ½Σρ_t². Prior work,
    once: "known mechanism, new regime, new consequence" (Arora 2019,
    van Laarhoven 2017, Kosson 2024 ⚠ verify). New here: at > 99.99%
    noise the step sizes alone predict the growth, and the growth
    carries a run past its peak.
  - **Fig 2b (law, `motiv_growth.pdf`).** Predicted vs. measured Φ:
    13 runs within 2.4% along the whole run, up to 39×, on both methods
    and two models, including steps up to ρ 0.6 (r4a, 1.7B's first
    step) (E1). One clause from E17: the residual bounds the mean
    cos(θ, Δθ) at ≲ 1e-4, so on average the steps have no radial part.
    Wording: "follows from" or "predicts within 2.4%", never "exact";
    "without momentum or weight decay".
  - **Fig 2a (fall, `motiv_collapse.pdf`).** With no bound,
    relative-step SGD peaks at 86.8% at Φ 3.7, then gives back 1 pt by
    Φ 5 and 4.8 pt by Φ 8, and ends
    at 29.6% at Φ 39 (chance 25%; E2). The fall is slow, then fast, so
    a short or target-stopped run never sees it. On Yahoo the same
    recipe falls 3.2 pt by Φ 6.4 before the wall cap (r3a). Scope:
    "falls past its peak on two datasets, to chance on AG News".
  - Where accuracy stops improving is set by the model, not the
    method: both arms on AG News at Φ ≈ 3.7 despite a 2× different
    initial norm; 4.1 on Yahoo (r3a); the 1.7B gated run still gains
    at Φ 8 and is best at Φ 13 (E4, D2). "The ratio matches, the
    norm does not."
  - Causality ban (§0): write "Φ measures how much noise the run has
    put into the weights, and past a point accuracy falls with it",
    never "growth causes the fall".
  - Not a direction failure: the cosine rises as accuracy falls (q1
    0.005 → 0.019, E9). One clause; the full point lands in §gate.
  - Weight decay, one forward pointer (no result): it is the known
    brake on norm growth (Kosson 2024); §5.2 says why it is not enough
    (E22: it holds the peak but costs time). It stays out of §3: it is
    a design-space result, and a reviewer reading §3.1 asks for it.
  - **Opportunity:** once the steps are chosen, the growth is known,
    so where a run ends can be chosen before training rather than
    found by running into the fall.

- **§3.2 `sec:motiv:gate`: The variance gate tracks the update's size,
  and it neither plans nor bounds growth.**
  - Its purpose, plainly (D4): it makes each server update clean
    enough by measuring. It does not guard against the fall. Never
    write "systems gate to prevent the fall".
  - **Fig 2c (size, not signal, `motiv_gate.pdf`).** At shipped
    settings the first-iteration variance and ‖g‖² both grow ×5.7, and
    iterations per server update 9 → 56, over one run; variance ÷ ‖g‖²
    stays flat (E7; also AG 4.9 / 4.8, Yahoo 18 / 18, 1.7B 0.57 /
    0.57). The gated step's cosine stays ≈ 0.003, no trend. So the gate
    asks for ~6× the work per server update without raising the signal
    share: 180 → 1,140 forward passes per server update (median 660),
    against a fixed 1,000 for a 50-update buffer (E10). Claim the growth,
    not the level.
    Scope: "at shipped settings", because weight decay slows that rise
    (E22). ‖g‖ is the norm of the noisy update, never the true
    gradient.
  - Secondary (D3), one clause: picking 1 of P keeps one draw's signal
    and discards the rest (analytic 1.9/√p vs. √10/√p; R6 not run, so
    analytic only).
  - **Fig 2d (no bound, `motiv_rho.pdf`).** With a fixed lr, the relative step ρ =
    η‖G‖/‖θ‖ shrinks as the weights grow: 0.19 → 0.04 at the shipped
    lr (q4). That shrinking, not the gate, keeps the shipped run near
    its plateau. At 3× the lr (r4a) the gate still passes (variance
    0.3) every server update, and iterations grow 21 → 48. But ρ
    starts at 0.58, the weights reach Φ 10 in 300 server updates, and
    accuracy peaks at 43.2% (Φ 3.45) and is at chance by Φ 7.8 (E21).
    This is D4, now measured, and it backs the intro ¶5 attribution of
    the escape.
  - **Do not attribute FwdLLM's lower peak to its shrinking step**
    (contradicted 2026-10-30). r2s has the same fixed lr and a
    similarly shrinking ρ (0.079 → 0.034), yet averages a fixed
    buffer, and it peaks at 87.7%. r2b has the gate, pick and held
    cohort with a planned step, and it learns far slower (E19, E20).
    The evidence puts the lower peak on the gated-averaging side
    (caveat: r2b's trainable set is FwdLLM's). That is an ablation
    finding for the eval, not §3. Intro ¶5's "which also stall it at a
    lower peak" must change (§11).
  - Neither gate signal warns of the fall: the committed variance is
    0.29 against a 0.3 threshold throughout, by construction (E8′).
    The cosine rises as accuracy collapses (q1 0.005 → 0.019; r4a
    0.003 → 0.039). Φ would warn, but §3.1 makes it known in advance,
    so measuring it adds nothing.
  - Gradient-unit settings (L2):
    - a 3× change in lr turns a run that plateaus at 84.9% into one
      that peaks at 43.2% and ends at chance (q4 vs. r4a; one cell);
    - the shipped thresholds are 0.3 on DistilBERT and 8.4 on 1.7B
      (28×), and the lrs 0.01 and 3e-3 (E11);
    - ⚠ re-verify that RoBERTa-L's 0.2 threshold gives zero server
      updates on 1.7B before citing it.
  - **Opportunity:** a control fixed before training needs no
    threshold, and it cannot drift with the gradient.

- **§3.3 `sec:motiv:hold`: The variance gate holds phones online for
  minutes to hours.** (Lead with 1.7B.)
  - Why the cohort must stay, as intuition first: the variance
    compares the same phones' updates on one shared minibatch at one
    model version, so a phone that leaves stalls the measurement.
  - Setup: hold = simulated time from a cohort's dispatch to its
    server update.
  - **Fig 2e (`motiv_hold.pdf`).** CDF of the hold per server update:
    gated DistilBERT median 2.9 min, gated 1.7B median 83 min (p90 150),
    over MobiPerf session lengths (median 10.2 min). 40% of sessions are
    shorter than the DistilBERT hold, 83% than the 1.7B hold. Ticks mark
    one task's time (2 s, 255 s), what a phone would need to stay if
    nothing were measured; not SwiFT's curve.
    - Why 83 min and not 8 × 255 s: each iteration waits for the
      slowest of the 10 phones (median 511 s), so a held phone computes
      ~34 min of its 83 (E12).
    - RoBERTa-L dropped (lr screens only, R1c not run).
  - An asynchronous engine alone does not remove the hold: the gate
    needs one shared minibatch at one model version, which async
    arrivals do not supply (argument). ⚠ a11–a13 results unread.
  - Non-IID, one sentence, with no figure and no measurement (R6 and
    R9 not run). A held cohort contributes only its K distinct clients
    to a server update, and client heterogeneity cancels only over
    distinct clients. FwdLLM reports losing accuracy on non-IID data
    (cite; up to 3.8% ⚠ check the exact values). This follows the
    ledger's D8 and the intro ¶5, which overrule the old outline's
    "drop".
  - **Opportunity:** if nothing is measured on a cohort, any available
    phone can contribute one task and leave.

- **Dropped from §3 (2026-10-30):**
  - the cosine panel (old 2b): one number in the opening instead;
  - the backprop ceiling: no run;
  - the trace-stall result (E18, R5);
  - the non-IID figure (R6, R9);
  - the averaging-scaling and per-step tension (E13, E14, E24): these
    move to §5.1, because they reveal the buffer.

## 8. Solution pitch: §4 Overview and §5 Design (headlines)

### §4 Overview (≈ 0.5 page)

- **Design goals (each testable):**
  - G1 *Holds its peak:* growth approaches a chosen Φ_budget and never
    passes it. Measured: post-peak drop; final Φ vs. Φ_budget; −budget
    collapses.
  - G2 *No gradient-unit settings:* N, s, Φ_budget are unit-free; N 50
    and Φ_budget 4 on all three models; s per model. Datasets: E23.
  - G3 *Fits a phone:* inference only; one adapter-sized upload per
    task; one task per engagement. Measured: hold CDF, bytes per task.
  - G4 *Composes:* unmodified async engine; secure-aggregation
    compatible (server reads only the aggregate and its own weights).
  - Non-goals: new async protocol, availability prediction,
    compression, DP itself, per-client models, run-time monitoring.
- **Architecture figure (P10):** client vs. server; borrowed async
  engine shaded grey. Names enter inline at their step:
  1. Server sends weights + shared-minibatch index to any available
     phone (cohort-free participation, grey).
  2. Client runs P perturbations, uploads their mean (signal-sized
     buffering, client half).
  3. Server buffers N client updates (signal-sized buffering, server
     half).
  4. Server steps ρ_t‖θ‖ along the buffer's mean, ρ_t from the plan
     (signal-calibrated steps).
- **Traffic on the figures (§10 bookkeeping):** per iteration and per
  server update, FwdLLM vs. SwiFT: messages and bytes each way, from
  the code.
- **Overview by example (P11):** 6 phones, 3 server updates. FwdLLM:
  one cohort per bin, iterations until the gate passes, fixed lr, Φ
  unplanned. SwiFT: fresh phones, one task each, ρ‖θ‖ steps along a
  planned Φ curve.
- **Plan before training (one box or sentence):** the server fixes N,
  s and Φ_budget; ρ0 = s√(PN/p) and T_res = 2 lnΦ_budget/ρ0² follow;
  the whole ρ schedule, and therefore Φ_t, is known at t = 0 (1.7B 6c:
  ρ0 0.0978, T_res 290 → Φ_budget 4, followed exactly). Φ_t approaches
  Φ_budget from below, Φ_t = Φ_budget^(1 − e^(−t/T_res)); q2 ends at
  3.67 (predicted 3.62).

### §5.1 Signal-sized buffering: how much does a step average, and from whom?

- Observation: §3.1 (the noise share of a step is set by its
  averaging, not by when it is taken), §3.3 (the gate measures size
  and costs work), §3.4 (the gate holds phones).
- Mechanism: client uploads the mean of its P slope-weighted
  perturbations; server updates every N client updates, no gate.
- Why the mean: minimum-variance unbiased combination of i.i.d.
  guesses; the pick keeps one draw's worth. Credit MeZO. ⚠ bench rerun
  of best-of-P if the −avg arm is kept.
- Why fixed N, and why 50 (E14): N 10 barely learns at either step
  size (peak 0.64 / 0.59); N 90 at the same ρ matches N 50 per server
  update but costs 1.6× the time; the cosine rises from N 50 to 90 no
  slower than √N (E13, level 10–15× below theory, scaling only).
- (audit, R8 done) The √(PN/p) factor: with ρ0 rescaled, N 90 gets the
  larger step and reaches 0.85 / 0.86 17–24% sooner than N 50 (ties in
  accuracy, 300 server updates only); N 10 gets a smaller step and is
  no better. So the factor is right in direction at the large end and
  moot at the small end. Write N 50 as "enough" (larger N with
  rescaled ρ0 is a little faster, at more phones per step), never
  "optimal".
- Stability without a variance bound: a small N wastes progress, it
  does not diverge, because the step length is ρ‖θ‖ whatever the noise
  and growth is planned. Evidence: N 10 (q5a): Φ 1.51 on plan (0.3%),
  no divergence.
- Cost: same compute and bytes per client update as the pick; 4 vs. 33
  iterations per server update; server needs only the aggregate.
- Ablations: −buffer (r2b: 0.60 peak, 0.50 at c200 vs. SwiFT 0.785,
  2.6× the time per server update, E20); −avg (pick, L11); −N↓ (q5a,
  r8a).

### §5.2 Signal-calibrated steps: how far does each step move, and for how long?

- Observation: §3.2 (exact law; per-model peak growth; collapse).
- Mechanism: Δθ_t = −ρ_t‖θ_t‖·G/‖G‖; ρ0 = s√(PN/p); ρ_t = ρ0
  e^{−(t−1)/(2T_res)}, so Φ → Φ_budget. Credit LARS / LAMB for the
  form.
- Why a budget, not a gate: the gate's signals do not warn of the fall
  (§3.3, E9); growth is exact, so bound it directly.
- Design space (P15, audit), each with why not:
  - *Monitor Φ and stop* (constant ρ + stop at Φ_budget; the code's Φ
    rail): Φ is known in advance, so a rail is the plan with a cliff
    in it; it ends training instead of slowing it, so a run cannot
    keep learning past the stop (FL accuracy is a moving target, ¶1).
    Data: q1 at Φ 4 is 0.866, q2 0.880, a tie, so argue this one,
    don't claim accuracy.
  - *Weight decay* (the textbook brake on norm growth; Kosson et al.
    rotational equilibrium): it bounds growth and holds the peak (both
    arms 0.864, a tie with SwiFT's 0.880), but by shrinking every step
    it slows training: constant ρ + WD ρ²/2 pins Φ at 1.08 and takes
    2.3× as long to 0.86; FwdLLM + WD pins Φ at 2.65 and takes 30 h
    (E22). λ is also a gradient-unit setting (0.0026 was chosen from
    q4's ρ). Argue time and tuning, not accuracy.
  - *A standard lr decay* (cosine, step): decays in iterations, not in
    growth, so its endpoint Φ depends on the gradient's units; the
    budget picks the decay that ends at a set Φ.
  - *Server optimizer* (FedAdam, momentum): momentum breaks the growth
    law (A.1); ⚠ FedAdam in the metric map, never run (R7c).
- Why Φ, not ‖θ‖: both methods stop improving at the same Φ with 2×
  different initial norms (E4).
- Setting Φ_budget: flat across a broad band, 0.872–0.880 for Φ_b 2–8
  (ties, E5); 4 on every model, including 1.7B, where unbounded growth
  stops helping only past Φ 13 and Φ_b 4 still wins by 9.4 pt. So the
  budget need not match where growth stops helping; it must sit below
  the fall. Too small: slower (75% more server updates to 0.87). Too
  large: q1's early decline. Never "optimum".
- What it costs and buys (E6): ≤ ~15% more time to 0.86; the peak
  holds for the whole run, so the result does not depend on stopping
  at the right server update. (Not "higher peak": 0.880 vs. 0.868 is a
  tie.)
- Ablations: −budget (constant ρ, q1: collapse to chance); −step (SGD
  lr 0.01, r2s: ties, E19; its ρ happens to decay into the band
  because ‖θ‖ outgrows ‖G‖; with 3× lr the same SGD step runs away,
  E21).
- Math: ‖θ_{t+1}‖² = ‖θ_t‖²(1 + ρ_t²); Φ_T = Π√(1 + ρ_t²) ≈ e^{½Σρ²};
  T_res = 2 lnΦ_budget/ρ0²; SGD's ρ = η‖G‖/‖θ‖ is unplanned.

### §5.3 Cohort-free participation (≈ 150 w; inherited, not claimed)

- Consequence: with no variance to measure, FedBuff's arrival model
  applies as is. Shared minibatch index per server update (⚠ credit
  vs. FwdLLM's data bin). Staleness E[τ] ≈ C/N; stale updates
  unweighted.
- Control: −select not run (R9); no claim.

### §5.4 Algorithm 1 + settings table

| setting | DistilBERT / RoBERTa-L / 1.7B | kind |
|---|---|---|
| P | 10 | set (compute) |
| N | 50 | set once (E14) |
| s | 2.10 / 2.4 / 2.81 | set, per model |
| ρ0 | 0.07 / 0.0698 / 0.0978 | derived |
| Φ_budget | 4 / 4 / 4 | set; broad band (E5) |
| T_res | 566 / 569 / 290 | derived (2 ln Φ_budget / ρ0²) |
| C | 0.3 n | deployment |
| server momentum | 0 everywhere | not used (breaks the law) |
| adapter / batch | rf 16, bs 8 / LoRA r 6, bs 32 / LoRA r 2, bs 32 | per model (A.2) |
| stop rule | ⚠ restate; same in all arms | run control (A.4) |

## 9. Later sections (claims they must back)

- **Theory:** growth law (E1, predicted vs. measured scatter); cosine
  scaling with N (E13, scaling only). Slogan: "the step schedule fixes
  growth before training".
- **Evaluation:** four cells end-to-end at Φ_budget 4 (DistilBERT ×2
  and 1.7B done; RoBERTa-L FwdLLM ⚠ R1c); time, bytes and device work
  to target; hold CDFs; E23 for G2; noise floor stated once.
- **Ablations:** −budget (q1), −step (r2s), −buffer (r2b), −avg ⚠ L11,
  −N↓ (q5a / r8a); each vs. full SwiFT and FwdLLM, on time and final
  accuracy. No −select.
- **Obvious fixes (metric map):** FwdLLM + WD (r7a), SwiFT + WD (r7b),
  E22. FedAdam not run.
- **Sensitivity:** Φ_budget 2 / 4 / 8 / ∞ (E5); N 10 / 50 / 90 at same
  and rescaled ρ0 (E14); s.
- **Honest limits (P24):** 10 forward passes per perturbation set; gap
  to backprop ⚠; one seed; s per model; peak Φ differs across models
  (≥ 13 on 1.7B) and depends on the path; cosine level below theory;
  momentum breaks the law; big-model time assumes A40 → NPU ratios.
- **Appendix:** A.1 momentum (why not; 6b, run 4, a22); A.2 LoRA rank,
  batch size; A.3 P 40; A.4 stop rule; A.5 client weighting; A.6 C;
  A.7 finite-difference spacing; A.8 rejected alternatives (constant ρ
  = −budget is main text now; 1/t, Adam / Yogi, adaptive buffer, time
  window).

## 10. Runs (status 2026-10-30)

### 10.0 Where the data is

- Shepherd,
  `/home/dgarg39/aish_test/flame/lib/python/examples/fwdllm/`: q1–q3,
  q5a / q5b in `paper_pitch_20261029b/`; q4 in
  `paper_pitch_20261029a/`; r2s r2b r4a r7a r7b r8a r8b in
  `paper_runs_20261029/`; r3a r3b in `paper_runs_20261029y/`
  (relaunched with cos every 150; the r3a / r3b rows in
  `paper_runs_20261029/` are the killed first try); runs 1–2 in
  `p1_distil_20261025/bench/`. Telemetry:
  `POOL/<arm>/runs/*/run_*/telemetry/aggregator_*.jsonl`; watcher
  verdicts in `verdicts.json`.
- Arm settings, launch commands, watcher stop rules and the Q1–Q8
  readout: `_metadata/temp_fxtune_abl.md`.
- Read rules: smoothed = 5-eval trailing mean; gaps < 2.3 pt are ties;
  logged `phi` is the plan, so on WD arms read
  `trainable_weight_norm`; q4's vclock is 10× inflated (divisor bug),
  its per-commit data is fine; cos audit costs ~7.4 min wall per fire
  on the bench.

### 10.1 Done (all DistilBERT, shepherd bench, one seed)

| R | arm | result (ledger row) | text consequence |
|---|---|---|---|
| R1a | none needed | q2 + run 1 share bench and recipe: +3.1 pt, 7.0× to FwdLLM's peak (E16) | fills AG News row, abstract low end |
| R1b | r3b vs. run 28 | +6.4 pt; 0.65 at 3.5 h vs. never (E16) | fills Yahoo row |
| R2 | r2s (−step) | ties SwiFT (E19) | §2 baseline argument reframed; step rule not credited on this cell |
| R2 | r2b (−buffer) | far slower, guard stop at c210 (E20) | buffering credited |
| R3 | r3a, r3b | fall begins on Yahoo (−3.2 pt by Φ 6.4, wall cap); budget holds; settings transfer (E2, E4, E23) | §3.2 scope "two datasets, to chance on one"; title rung 2 open |
| R4 | r4a | gated system at 3× lr goes to chance by Φ 8 (E21) | Fig 2a arm; §3.3 brake; §0 ban narrowed to shipped settings |
| R7 | r7a, r7b | WD holds the peak, costs time (E22) | §5.2 "why not WD"; obvious-fixes table |
| R8 | r8a, r8b | √(PN/p) helps at N 90, moot at N 10 (E14, E24) | §5.1 "N 50 is enough" |
| R5 | — | cannot run (no unavailability wiring in the fwdllm aggregator) | trace clauses deleted |

### 10.2 Not run / open (priority order)

1. **R1c** FwdLLM RoBERTa-L · Yelp-P (BitFit lr 1e-2, var 0.2, divisor
   0.169; 12–30 h, 4 GPUs). Operator decides. Fills the last cell and
   §3.4's RoBERTa-L hold (now ~20 min from lr screens).
2. **−step on a second model** (1.7B or RoBERTa-L, SGD at the
   DistilBERT lr): the only way to show the lr does not transfer for
   the −step arm, which §2 now needs (E19). Or run FedKSeed / FedMeZO.
3. **Second seed** of q2 and r2s: whether −step's tie holds.
4. **R6** offline cosine bench: needs a script and saved q2
   checkpoints (none saved). Until then §3.1 best-of-P stays analytic,
   non-IID sentence dropped.
5. Optional: r3a to c1500 without the 8 h wall cap (collapse to chance
   on Yahoo); R4b (shipped lr, no timeout); R7c FedAdam; R9 −select
   (needs code).

### 10.3 From existing logs (no runs)

Answered 10-30 (now in §1): 1.7B time / bytes to FwdLLM's peak (E15);
forward passes per server update (E10); 83 min hold reconciled and
SwiFT's 1.7B task time (E12); MobiPerf session CDF (E12); q5a / q5b ρ0;
RoBERTa-L Φ_b 4 = run 4b; run 1 shares q2's bench. Open:
1. Bytes to FwdLLM's peak on DistilBERT · AG News (q2 vs. run 1) and
   Yahoo (r3b vs. run 28), plus Yahoo time to 0.616 and whether the
   eval rows match (E16).
2. a11–a13 results (FedBuff-async, Felix(P), FedBuff(P)): §3.3 async
   defence.
3. Runs 7 / 26 (−avg, pick): exist, and on which engine.
4. RoBERTa-L's 0.2 threshold on 1.7B giving zero server updates (E11).

### 10.4 Bookkeeping

- Positioning cells: FedKSeed and FedMeZO filled from the papers;
  verify against their code, and that both run fixed-lr SGD with fixed
  averaging and several local steps (client drift).
- Citations: Gemini Nano (no bib entry); the 1/√p cosine; verify that
  Arora 2019 / van Laarhoven 2017 / Kosson 2024 show growth from
  perpendicular steps; session-length trace; FwdLLM's "NPUs 30×
  faster" (than what?); FwdLLM numbers vs. the ATC'24 version.
- FwdLLM's pick criterion (largest |z| or similarity to the last
  aggregate) for §2.
- PARITY: does simulator-vs-real data exist for the current engine
  (contribution 4)?
- Traffic in the figures: per iteration and per server update, bytes
  from the code.
- CLAUDE.md: system-name line updated to SwiFT 10-29; its spine,
  contributions, settings, title ladder ("FedSiFT", "Predicting, Not
  Measuring") and metric map ("3 × 3") are still stale against this
  outline.
- 05-evaluation.tex still uses retired content (dynamic K/C,
  gradient-aware aggregation, perturbation picking as L3, E-labels,
  LLaMA2-7B in tables/eval/hyperparams.tex), cites labels that no
  longer exist (`sec:arch:*`), and writes N for the client population
  (use n).
- This outline, run_answers and paper_template are not tracked in git.

## 11. Written front matter: open decisions and holes (2026-10-30)

**Open decisions (authors):**
- Mechanism names: "signal-calibrated steps" and "signal-sized
  buffering" were chosen to echo "SiFT"; with SwiFT, keep them or
  rename (they are reused verbatim in ablation legends).
- Front-matter length: draft build has intro ≈1.4 pages, background
  ≈0.85, motivation ≈1.75 (budgets 1.25 / 0.5 + one table / 1.5).
  Fig 3 is folded into Fig 2 (panel e); the notation table moves to
  the appendix.
- The tension sentence (intro ¶3) is plain, not bold: B.1 asks for
  bold, the prose rules ban bold in prose.
- The intro's ideas do not map one-to-one to the costs: both ideas
  address cost 2 (drift and tuning).

**From the R results, for the ledger (2026-10-30; authors approve):**
- **Intro ¶5 last sentence is contradicted.** It reads "We attribute
  FwdLLM's escape from the fall to its shrinking relative steps, which
  also stall it at a lower peak."
  - The escape part holds: q4 vs. r4a (E21).
  - The lower-peak part does not: r2s has the same fixed lr and a
    shrinking ρ, and it peaks at 87.7% with a fixed buffer (E19).
    r2b's gate + pick + cohort learns far slower (E20).
  - Proposed: "...to its shrinking relative steps (§3.2); at 3× the
    learning rate the gate still passes and the run falls to chance."
  - The lower peak is credited in the ablations.
- Ledger claim "growth law within 1.6% on 7 runs, up to 39×": change to
  "within 2.4% on 13 runs, up to 39× and relative steps up to 0.6".
- Ledger / intro "+9.8 pt" on 1.7B: change to +9.4 (smoothed vs.
  smoothed, E15).
- Ledger claim "without the growth budget, accuracy falls": add "falls
  on two datasets (−3.2 pt on Yahoo by Φ 6.4), to chance on AG News".
- New ledger claim, home §3.2 gate: the gated system at 3× lr peaks at
  43.2% and ends at chance; it is a controlled run, not a baseline
  result.
- D4 is now measured (r4a), no longer only argued.
- Ledger home of "cosine ≈ 0.003": from Motivation §noise to the
  Motivation opening.
- Non-IID in §3: the argument plus FwdLLM's own report only. Nothing
  measured (R6 and R9 not run).
- Conflicts to settle, not from the R runs:
  - ledger O1 gives DistilBERT Φ_budget 2, while R1a / R1b used 4 (q2,
    r3b);
  - §0 bans "the noise is predictable", which is the abstract's
    sentence and the ledger's insight.
- E19 (−step ties) leaves the ledger's spine wording "removes the
  fixed learning rate" true only as "no lr to screen". The eval must
  state the tie. Design, not front matter.

**Holes in the .tex (`\ph`, `\tbd`), by source (2026-10-30):**
- Fill now from §1: abstract / intro peak range (+3.1 to +9.4 pt),
  6.2–7.0× sooner, 3.0× fewer GB on 1.7B; Fig 1 numbers (E15, E12);
  §3.2 forward passes `\ph{X}` / `\ph{Y}` = 180 → 1,140 vs. 1,000
  (E10); §3.3 83 min explanation and session numbers (E12); §3.2 collapse scope (E2, two datasets); §3.3 brake
  paragraph (E21, add to Fig 2a); title rung 2 (E23).
- Rewrite now: §2 baseline justification (E19); delete intro ¶7 trace
  clause, §3.4 real-trace and non-IID sentences (R5, R9); §3.1
  best-of-P stays analytic, non-IID sentence dropped (R6 not run).
- R1c: RoBERTa-L row, §3.4 RoBERTa-L hold.
- Logs (§10.3): DistilBERT bytes ratios, Yahoo time to FwdLLM's peak,
  a11–a13, E11 re-verify.
- Bookkeeping: Gemini Nano, 1/√p and norm-growth citations; 30× NPU;
  pick criterion; positioning cells; PARITY; Fig 2a backprop ceiling
  (no run).

## 12. Figures made (2026-10-30)

All in `lib/python/examples/fluxtune_paper/` (next to `paper_felix/`):
`figs/<section>/<name>.{pdf,png}`, regenerated by `code/make_intro_fig.py`
and `code/make_motiv_figs.py` from per-run digests in `data/` (README
there has the run paths and how to refresh). Style follows
`plot_guideline_fluxtune.md` (`code/ft_style.py`): SwiFT wine, gated
system teal, gated at 3× lr olive dashed, relative step with no bound
green, other runs grey, phone sessions sand. Every PDF is drawn at print
size (include unscaled), fonts are Type 42, and the overlap check is
clean. Lines are the 5-eval trailing mean (Fig 1: display-smoothed, numbers
from the trailing mean); legends use paper terms, run ids are in the PDF
metadata only.

| fig | label | file | size (in) | runs | shows |
|---|---|---|---|---|---|
| 1 | `fig:intro-teaser` | `figs/intro/fig1_teaser` | 3.25 × 1.6 | 6c, 1.7B FwdLLM | (a) accuracy vs. simulated h: +9.4 pt peak, 6.2× sooner, 3.0× fewer GB; (b) time online per contribution CDF: hold 83 min vs. task 28.4 min, 2.9× shorter, over sessions (E15, E12) |
| 2a | `fig:motiv-collapse` | `figs/background/motiv_collapse` | 1.35 × 1.75 | q1, q4, r4a | accuracy vs. measured Φ (log); chance and Φ ≈ 3.7 lines (E2, E21) |
| 2b | `fig:motiv-growth` | `figs/background/motiv_growth` | 1.35 × 1.75 | 13 runs (E1) | predicted vs. measured Φ, log–log; "13 runs within 2.4%" |
| 2c | `fig:motiv-gate` | `figs/background/motiv_gate` | 1.35 × 1.75 | q4 | variance, ‖g‖², iterations per update, variance ÷ ‖g‖², each × its start (E7) |
| 2d | `fig:motiv-rho` | `figs/background/motiv_rho` | 1.35 × 1.75 | q4, r4a | ρ per server update (log); accuracy-peak and reaches-chance markers (E21) |
| 2e | `fig:motiv-hold` | `figs/background/motiv_hold` | 1.35 × 1.75 | run 1, 1.7B FwdLLM | hold CDFs over MobiPerf sessions; one-task ticks (E12) |

**Definitions used (state in §2 / captions):** Φ = ‖θ_t‖/‖θ_0‖ with θ_0
the initial trainable weights; ρ_t = ‖Δθ_t‖/‖θ_{t−1}‖; predicted
Φ_t = Π√(1+ρ_s²). The logged `rho` divides by the post-step norm and
the logged `phi` is the plan; neither is used.

**Departures from the specs (authors may revert):** fonts 7 / 6.5 pt in
the 1.35 in panels (8 / 7 does not fit); guideline palette instead of
the old `BASE` orange; 2a y-axis from 10% to leave room for the
"Chance" label; 2e and Fig 1b time on a log / linear axis respectively
(2e spans 2 s to 4.5 h; Fig 1b is cut at 3 h, max hold 273 min);
RoBERTa-L dropped from 2e. Legends keep "lr" and "Relative step, no
bound" (longer names do not fit 1.35 in; captions spell them out).

**Caption read-hints:** 2a "higher is better; x is how far the weights
have grown, log scale"; 2b "points on the diagonal mean the prediction
is exact"; 2d "lower means each step moves the weights less relative
to their size"; 2e "a hold right of most sessions means phones leave
before the server update".

**Not plotted:** backprop ceiling (no run); cosine per server update
(one number in the text); non-IID (R6, R9 not run); trace stall (R5);
any ablation (r2s, r2b, r7a/b, r8) in §3.
