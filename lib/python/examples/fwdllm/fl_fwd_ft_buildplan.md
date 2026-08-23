# Build plan — **the status doc**: the claim, where it stands, and what to launch next

> **One file for status and next steps.** §1 is the claim and its scoreboard · §2 is what is running now ·
> §3 is the queue · §4 is how to run and read a run · §5 is what must not be re-derived · §6 is the rules
> any change inherits.
>
> **The other two docs are evidence, not status.** [fl_fwd_ft_practice.md](fl_fwd_ft_practice.md) owns
> *what is true*: the P3 knob ledger, the P4 run ledger, P6's dead ends, P8's reproduction recipes. Every
> number cited here lives there. [fl_fwd_ft_solution.md](fl_fwd_ft_solution.md) owns *why* — the model,
> cited as "model §x". **Read [P6](fl_fwd_ft_practice.md#p6--dead-ends--do-not-retry) before proposing any
> change.**

**How to update this doc — IN PLACE, never append.** One fact, one home: change *that* line, never add a
second statement of it. Replace, do not accumulate — a measurement carries its value and its date, the old
value is deleted. No changelog, no session log, no dated append sections; chronology lives in git and P4.
**When a queue row lands, delete it** — what survives is one sentence in §5 if the fact still binds, and
its number in P3/P4. **§1–§3 are the status budget: ~180 lines at 24 live queue rows.** If it grows without
the queue growing, something in it has stopped being status and belongs in §5 — that is how §5.7 got there.

**§3 is the single source of next steps.** The writeup names the same work in prose and points here.

---

## §1 — The claim, and what is missing

> **FluxTune reaches and holds a plateau on a new dataset with no learning knob tuned by hand — same
> DistilBERT + adapters, three datasets, against a version of itself whose step size was hand-searched.**

**Three systems; use these names everywhere, figures included.** **FwdLLM** — prior work, variance gate,
raw SGD. **FluxTune-v2** — trust-ratio + `n_target`, but a **static `ρ*`=0.06 hand-searched on agnews**,
RM-decayed (`rm`/`setpoint`; the code already calls it `fluxtune_v2`). **FluxTune** — this work, `ρ*` from
law C on a **sensed** `B_max`. **backprop ceiling** — exact gradients, **centralized**, 10 clients × 3
epochs: a plumbing diagnostic, not a target (§5.1).

**Why sensing is required:** B-1's *offline* sweep measured `B_max` erratic across task (agnews knee
≈3.0–3.5, yahoo and yelp-p ≈2.0–2.3, non-monotone in class count), so it cannot ship as a constant. **The
live probe has not reproduced that in 40 fires** — hole 2.

**What a new dataset costs the operator — the zero-input claim, itemised: §5.7.**

### Scoreboard

*FluxTune column = the 2026-08-21 `anchor` + `log_only` runs (N1–N3); **peak** accuracy, per [P4.4](fl_fwd_ft_practice.md#p44-scoring-rules-for-any-ab).*

**Only the first row is the goal.** The **FL target** is what the task should reach *in the federated
setting* — 100 non-IID clients, forward-only, adapters. The **backprop ceiling** is centralized and
10-client; it is a **plumbing diagnostic**, never a bar to clear, and yahoo's 0.734 was never the right
target for a 100-client non-IID run.

| what the claim needs | agnews | yahoo | yelp-p |
|---|---|---|---|
| **FL target** *(the goal)* | **0.880** | **0.660** | **0.820** |
| backprop ceiling *(plumbing diagnostic only)* | 0.850 | 0.734 | 0.874 |
| its own sim charge profile | `fluxtune.yaml` | built, **node-2 local only** — row **D0** | `fluxtune_yelp-p.yaml`, in git |
| **FluxTune-v2** run valid | **yes** — 938 commits, peak 0.843 | **yes** — 1,138 commits, peak 0.428 | **yes** — 997 commits, peak 0.728 |
| **FluxTune** ends on its own stop | **no — no stop exists** | **no — same** | **no — same** |
| **reaches the FL target** | **−0.007** (0.873) | **YES — +0.003** (0.663) | **−0.008** (0.812) |
| **beats FluxTune-v2** | 0.873 vs 0.843, **5.3×** | 0.663 vs 0.428, **6.2×** | 0.812 vs 0.728, **7.9×** |
| v2 against the FL target | −0.037 | **−0.233** | −0.092 |
| accuracy still has slope in `B`? | **no** | **no — retracted** | **no** |
| `B_max` sensed, not supplied | 1.694, 12 fires | 2.017, 16 fires | 1.847, 12 fires — **all hole 2** |

**FluxTune lands within 0.008 of the FL target on all three and clears it on yahoo; v2 clears none.** The
sized saturation stop (row **E**) would land at 0.867 / 0.652 / 0.807 — 0.013 / 0.008 / 0.018 under target,
in **45% / 55% / 60%** of the wall clock these runs actually spent.

**×** is the vclock at which FluxTune passes v2's *own full-budget peak*. The 08-20 pairs read 5.5 / 6.5 /
8.5, so the headline **replicates under a different combiner** — marginally *worse* under `anchor`, which is
what schedule-neutrality predicts
([P4.11](fl_fwd_ft_practice.md#p411-the-2026-08-20-p-4-pairs--the-law-wins-on-all-three-one-pair-is-valid)).

### The finding that reorders everything — §5.5

**All three datasets are saturated, and every run peaks at the same `Φ`.** N1–N3 ran `anchor` + `log_only`
with nothing halting them, straight past the rail to `Φ` = 4.65 / 6.00 / 5.17. They peak at
**`Φ` = 2.82 / 3.00 / 2.91** and `Λ` = 1.38 / 1.46 / 1.43 — a band of 0.18 in `Φ` across three tasks — then
**give it back**: −0.03 / −0.14 / −0.11 by the end. Curves and the saturation-stop sizing: **§5.5**.
What `Φ` is and what it is a property of: **§5.6**.

> **Two readings are retracted with it** (numbers in §5.5). *(1)* The tail-slope extrapolation over-predicted
> the extra budget's gain by 3.4× on agnews, **17× on yahoo** and got the sign wrong on yelp-p — **no
> dataset has slope left in `B`**, and yahoo's "still climbing" status goes with it. *(2)* The combiner
> throttle **cost hours and cost accuracy nowhere**; un-throttling did not even shorten time-to-v2's-peak.
> `Λ = 2B/s` is schedule-neutral, now measured twice on independent runs.

**Four axes of generality, and only one is exercised:**

| axis | coverage | status | rows |
|---|---|---|---|
| **datasets** | 3 of 3 run, 3 valid v2 runs | **all three land within 0.008 of the FL target**, yahoo clears it; all three saturate, so no gap left is a budget problem. **But none of them terminated** — the peak is demonstrated, the plateau is not | C · D · **Y** · Score |
| **models** | **0** | every run on record is DistilBERT + adapters. No second model ever tried | **N5a** → **N5b** → **N5c**, gated on P1 |
| **PEFT capacity within that model** | `rf` 16 vs 64 | **negative** — hole 3, and it is what N5b re-derives against a measured `Φ*` | N5a · N5b · P1 |
| **heterogeneity** | α = 1 only | ablations go **up** to α = 10/100, never below 1. Not started | — |

### The holes

**Holes 1, 2 and 5 are one defect, and hole 5 is its root.** The probe's grid floor pins `B_rem` at ≈0.22, which makes `B_max` a
*receding* target, and a receding target breaks law C in two places at once — it never anneals and it never
arrives. Everything the three runs did wrong follows from that single number.

```
Φ_knee pinned at 1.25   ⇒  B_rem = ln 1.25 = 0.223, constant       (hole 2, the cause)
   ⇒ B_max = B + 0.223, RECEDING, never a fixed point
   ⇒ ρ* = √(2·0.223/300) = 0.0386  CONSTANT — law C stops annealing   (hole 1a)
   ⇒ B ≥ f·B_max needs B ≥ f·r/(1−f) = 4.75, i.e. Φ≈116 — never fires (hole 1b)
   ⇒ constant ρ ⇒ Σρ² = ∞ ⇒ Φ grows without bound ⇒ past the cliff, every time
```

1. **A FluxTune run does not slow down. It can now stop — untested on a run.** *(a)* Law C degenerated into a **constant step** —
   measured ρ flat at **0.034 ± 0.002** over the last 70% of every run, a sawtooth reset upward at every
   probe fire, drifting only −0.007/1,000 commits. **That is exactly FluxTune-v1's failure mode**, reached
   from the opposite direction. *(b)* **Both stops now can fire, as of 2026-08-22** — the three predicates
   are an `OR` at the settled 3.0 rail and the saturation stop ships behind `--saturation-stop`
   (§5.3, §5.5). **No run has ended on either yet**: rows **C**, **D** and **Y** are what test that. The
   anneal half of this hole is untouched — row **N4a′** is still the cause, row **A** the fallback.
2. **The `B_max` probe reports its own grid floor. Confirmed on 40 fires, and now explained.** `knee()`
   interpolates from an implicit `(Φ=1, normalized 1.0)` anchor to the first grid point. When the model is
   already below half-accuracy at `Φ`=1.5 — which it is on **all 40 fires** — the answer is
   `1 + 0.5·(1−0.5)/(1−n₁)` → **1.25 as `n₁`→0**. That is the 1.25–1.28 every fire returns: not a
   measurement, an extrapolation off the anchor. The four points at 2.5–4.0 never do any work. **Row N4a′.**
3. **The MODEL axis is untested and its one probe came back negative.** Every run is DistilBERT + adapters
   at `rf`=16; the three datasets differ in `p` by 1.4%. At `rf`=64 law C + `annealed` does not compose with
   the gate under **any** `T_res` (§5.3), so `T_res`=300 and `f`=0.95 are **pinned to one `p`**. Rows
   **N5a** → **N5b** → **N5c** walk §5.8's porting order; **P1** tests the same question from the other
   side, whether `Φ*` itself moves with `p` (§5.6). **N5b is the `rf`=64 failure restated as a task**, and
   it is worth redoing because that derivation assumed the `ln 2.7` prior rather than a measured `Φ*`.
4. **`Φ*` ≈ 2.9 is the most transferable number here and we do not know what it is a property of.** Peaks
   land at 2.82 / 3.00 / 2.91 on 4-, 10- and 2-class tasks, and the ledger's `p`-ladder held its peak at
   `Φ` 3.04 (`rf`=64) and 3.28 (`rf`=32) across a **3.8× range in `p`**. If `Φ*` is a property of the
   *model*, the whole controller reduces to "measure `Φ*` once, then spend to it" — §5.6. **Row P1**, and
   it is cheap.
5. **The `B_max` sensor measures a different quantity than the one law C anneals against, and this is the
   root of holes 1 and 2.** *(measured 2026-08-21; the numbers are §5.9)* Injection reads a **frozen**
   model; a run **re-fits** between every increment, and that difference is worth **~1.8× in `Φ`** — not
   the 0.6–1.2 offset assumed. ⇒ **Row N4a′ makes `Φ_knee` honest but still ~1.8× low**, so it is not
   sufficient on its own; `B_max` must come from a probe that allows a re-fit (**row P1′**) or from `Φ*`
   measured once per model (**row P1**).


---

## §2 — Now · what is running

*Read 2026-08-22 18:00. **Nothing is running** — all four nodes are free.* N1–N3 landed; node 4 produced
nothing and both its rows must be re-launched (§3). All nine runs are on this node's disk and their curves
are cached in `expt_scripts/writeup_figs/data/*.json`, so nothing needs to re-scan `experiments/`.

**Rows S · E · E2 · M1 · M2 · D0 landed as code on 2026-08-22, each gated offline; nothing on the queue is
blocked on a node any more.** What a launch looks like now is §4.1's table and §4.2b's commands.

| run | state |
|---|---|
| **N1** agnews `014242` · **N2** yahoo `014328` · **N3** yelp-p `014406` | **all three COMPLETED** on their vclock ceiling — 1,913 / 2,400 / 1,863 commits, 12 / 16 / 12 probe fires, `s`=1.5, `anchor`, `log_only`. **Correct, not void**: `log_only` runs have nothing to halt them (§4.2b). Peaks 0.873 / 0.663 / 0.812 |
| **N4a** re-range the probe grid | **NEVER LAUNCHED** — preflight `BLOCKED` on the shortened ceiling. Corrected overrides in §3 row N4a′; why the floors exist, §4.2b |
| **N4b** `s`=1.0 | **VOID, environment** — a peer job at 43.7 GB, the launcher warned and launched anyway, 85 of 100 trainers died on CUDA OOM, 0 commits. Rows **M1**/**M2**; the launch rule is §4.2b |

**The 08-20 runs keep their status**: the two void watchdog kills (`152215`, `125003`) replay clean under
the fix; the three v2 runs are valid; `125010` is the only run that ever ended on `[BudgetStop]`, and hole 1
now explains why no run since has.

**[fl_fwd_ft_writeup.md](fl_fwd_ft_writeup.md)** is the prose account with twelve rendered figures — the doc to
hand to anyone outside this work. Regenerate with `writeup_figs/make_figures.py` (§4.8).

---

## §3 — Next · the ordered queue

**This table is the single source of next steps.** [fl_fwd_ft_writeup.md](fl_fwd_ft_writeup.md) §8 names
the same work in prose and points here; nothing is queued there. **The backstops landed 2026-08-22** — the
stop is an `OR`, the saturation detector is sized and gated, the two launcher guards are armed — so the
queue is now *runs*, not code. **C, D and Y are the ones the claim needs**; **P1, P1′ and N4a′** are the
one defect behind everything the 08-21 runs did wrong (§1 holes 1, 2 and 5) and are independent of them.

> **The queue tracks two generality axes, and they are in very different states.** **Task** (same model,
> new dataset) is demonstrated on three datasets but **not yet terminated on any of them** — that is
> C · D · Y · Score, and it is the shorter half. **Model** (same dataset, new model) has
> **zero runs and one negative probe**, and is N5a → N5b → N5c walking §5.8's porting order, gated on P1
> for the one constant that might not be a constant. **A row that ships a hand-fitted number is not
> progress on either axis** — that is why E2 exists.

> **Re-ordered 2026-08-21 by §5.9.** N4a′ was "the root cause"; it is now the *shallower* half. The probe
> measures tolerance to **unearned** noise on a frozen model, and `Φ*` is where a **re-fitting** model
> peaks — ~1.8× apart in `Φ`. Re-ranging the grid removes the receding target; it does not make `B_max`
> the number law C should anneal against. **P1** (is `Φ*` a model constant?) and **P1′** (a probe that
> lets the model re-fit) are the two ways to get that number, and they are cheap and independent.

| # | node | task | done when |
|---|---|---|---|
| **P1** | 1 GPU, ~1 h, **no training run needed** | **Is `Φ*` a property of the MODEL or of the TASK?** The deepest open question in the work, and the cheapest to answer — `scripts/probe_inflation_damage.py` does it and now prints a `Φ_knee` per mode with `expts/bmax_probe.knee` — the same arithmetic the live sensor uses, so its number is directly comparable to a `[BmaxProbe] Phi_knee=` line. Train to peak, inject noise, read accuracy back, **on a grid that brackets**: `--phis 1.5,2,2.5,3,3.5,4 --rf 16` then `--rf 32` and `--rf 64`, on all three datasets. **Predicted (§5.6):** `Φ_knee` clusters near **2.9 across datasets** and moves — if at all — with `rf`. **Falsified if** the three datasets disagree by more than the 0.18 the runs show. Ledger support already: the `p`-ladder held its peak at `Φ` 3.04 (`rf`=64) and 3.28 (`rf`=32) across a 3.8× range in `p` | a `Φ_knee` per (dataset × `rf`), and a statement of which factor it tracks |
| **P1′** | 1 GPU, ~2 h, **after P1** | **Noise-then-refit probe — the instrument `B_max` actually needs.** The shipped probe injects at `Φ` and reads a **frozen** model; a run re-fits between every increment, and §5.9 measures that difference at **~1.8× in `Φ`**. **Built 2026-08-22: `scripts/probe_inflation_refit.py`** — inject at `Φ`, run **`m` steps**, *then* read. (`m` is the rig's own AdamW steps at batch 32, not FL commits: the question is whether the damage is re-fittable at all.) Sweep `m` ∈ {0, 10, 50, 150} × `Φ` ∈ {1.5, 2, 2.5, 3, 3.5, 4} on agnews. `m`=0 must reproduce the shipped probe's chance readings — that is the positive control. **Predicted:** the knee moves up with `m` and lands near **2.9** by `m`≈50, and stops moving after. **Falsified if** the knee is still ≤1.6 at `m`=150 — then the trajectory's junk is *not* isotropic and §5.9's mechanism is wrong, which would make `Φ*` unmeasurable without a training run | a knee-vs-`m` curve, and a stated `m` at which it saturates |
| **P2′** | 1 GPU, ~1 h | **Is `Φ*` an FL number or a training-geometry number?** Run the **same forward-gradient estimator centralized** (1 client, IID, same `p`, same `s`, same law C) on agnews and read `Φ` at peak. The model doc's §2.6 says heterogeneity is a step-size multiplier only, so `Φ*` should not move; the 22-run ledger already spans α 0.1–1 at 2.41–3.11 without moving. **Predicted:** peak at `Φ` = 2.8–3.0, i.e. inside the FL band. **Falsified if** centralized peaks below 2.4 or above 3.3 — then `Φ*` carries a federated component and every "property of the model" claim in §5.6 is wrong. **Do not run backprop as the comparator here** — backprop's steps are not ⟂ `θ`, so it reaches target accuracy at `Φ`≈1 and has no `Φ*` to compare (§5.6c) | a centralized forward-gradient `Φ` at peak, against 2.82 |
| **P3′** | any GPU node, rides on any run | **Log `cos(θ_t, θ_0)` and confirm it equals `1/Φ`.** §4.1a of the model doc *derives* retention `= 1/Φ` from the same perpendicularity that makes the norm law exact, and the whole angular reading of `Φ` (peak at ≈70° of drift) rests on it. It has never been measured. **Landed 2026-08-22 as `--retention-probe-every N`** (0 = off, byte-identical); `[Retention]` carries `cos`, `Φ`, `cos*Φ` and the drift angle, and `check_arm_health.py` scores the ratio. **Needs a run to carry it.** **Predicted:** `cos(θ_t,θ_0)·Φ` = 1.00 ± 0.02, matching the cross-term's 1.000 ± 0.005. **Falsified if** it runs materially above 1 — that means the aligned ~7% of each step overlaps `θ_0`, retention is better than `Φ` says, and `Φ` overstates the damage | the ratio logged over one full run |
| **N4a′** | node 4 · agnews, ~35 min | **Re-range the `B_max` probe grid.** *(Demoted 2026-08-21: this is necessary, not sufficient — hole 5 says no grid range makes this probe return `Φ*`.)* `P4_BMAX_PHIS="1.05,1.1,1.2,1.3,1.5,2.0"`. **Budgets, corrected:** `VCLOCK_OVERRIDE=12000` (6,000 gave ~120 commits — under the 150-commit probe cadence, so no probe could fire even had it launched) and `CEIL_OVERRIDE=2.0` (the preflight prices law C's full 898-commit length at 6,322 s regardless of the vclock, so any ceiling under 1.8 h is refused). **Predicted:** the knee lands **below 1.5**, `B_rem` comes out **smaller than 0.22 and shrinking with `B`**, and law C therefore **anneals again on its own**. **Falsified if** `B_rem` is still flat on a bracketing grid — then budget really is re-earned and the anneal must be imposed another way (row **A**) | a knee bracketed by real grid points, and a `B_rem` that falls as `B` rises |
| **A** | any CPU, **after N4a′** | **Restore the anneal — only if N4a′ does not.** Measured ρ is flat at 0.034 ± 0.002 over the last 70% of every run, so law C is running as a constant-step rule and `Σρ²` diverges. If a bracketing grid gives a shrinking `B_rem`, this row closes for free. If not, law C needs a term that cannot be defeated by a receding target — the obvious candidate is to anneal against **`Φ*` from row P1** rather than against a per-fire sense, which also removes the sawtooth | ρ falls monotonically over a run, and `Σρ²` converges |
| **N4b′** | node 4 · agnews | **FluxTune at `s`=1.0** (`P4_GATE_S=1.0`), `anchor` + `log_only`. Untouched by N4b — its config was correct (`gate_safety_s=1.0`, `rho_max` 0.0666, `n_req` ≈2.2× the `s`=1.5 run) and the node killed it. `Λ = 2B/s` says lowering `s` moves *along* the accuracy-vs-`Λ` curve, not up it. **Predicted:** the same peak (≈0.872) at the same `Λ`≈1.4, reached at lower `B`. **Falsified if** the peak is higher | whether `s` moves along the curve or shifts it |
| **C** | any GPU node | **agnews controller**, 48,000 vclock, on the new stack (`anchor`, saturation-primary, rail at 3.0). `condition_fp` will no longer read `c2ef1528` — expected and correct; control `021843` does not run the probe, so it stays the valid partner | ends on `[BudgetStop] reason=saturation` near commit 1,180, at or above 0.850 |
| **D** | any GPU node | **yahoo controller**, 60,000 vclock, same new stack | ends on saturation near commit 1,080, at or above 0.657 |
| **Y** | any GPU node | **yelp-p controller**, 50,000 vclock, same new stack. **This row was missing and the claim needs it:** yelp-p's only self-terminating run is `125010`, which ran the *old* `mean` combiner and the *old* budget stop — and hole 1 now explains that termination as `mean` lagging a rising sequence, i.e. **arithmetic on the combiner, not a run reaching its budget**. It is not evidence for the shipped stack. Control `161751` ran the full 50,000 clean and stays the valid partner | ends on `[BudgetStop] reason=saturation` near commit 1,126, at or above 0.807 |
| **R2** | 1 GPU, ~1 h | **Is the estimator the limit on yahoo AND yelp-p?** Both saturate below their reference (−0.071, −0.062) with flat accuracy-vs-`B`, so this is no longer a yelp-p-only question. cos audit for ~100 commits + `replay_scoring.py --cos`; a `D` materially below agnews' 0.10–0.15 means the forward estimate degrades with class count or seq 256 — an FwdLLM-layer finding, not a controller one. Plus H-S (`probe_fd_chord.py`) | a `D` for each against agnews' band |
| **N5a** | the node holding the `rf` 32/64 runs, replay, **free** | **Audit the FD chord ratio across `p` — step 2 of §5.8's porting order, and the least-audited thing in the stack.** `FWDLLM_FD_SCALE_INVARIANT` rescales `h` to hold the **absolute** chord `h√p` fixed at 6.7107 across the ladder (`h` = 0.01 / 0.014023 / 0.019507, [P9.1](fl_fwd_ft_practice.md#p91-preflight)) — but `‖θ_tr‖` is 13.35 / 9.6 / 6.86, so the **relative** chord `h√p/‖θ_tr‖` computes to **0.50 / 0.70 / 0.98**. If that holds, the flag holds the *united* quantity fixed and lets the dimensionless one nearly double, which is §5.3's ratio principle violated by the one knob that exists to enforce it. **`expt_scripts/audit_fd_chord.py` reads it off any run dir (built 2026-08-22); rf=16 confirms 0.503 here, and the rf=32/64 ladder runs are node-local.** **Predicted:** they read 0.70 and 0.98. **Falsified if** the logs show `h√p/‖θ_tr‖` constant — then the flag is already right and only the docs are wrong. Either way this also feeds **H-S**, which names the chord as prime suspect for the 3.5× shadow deficit | the ratio read off the three ladder runs, and a statement of which quantity the flag should hold fixed |
| **N5b** | any CPU + 1 short GPU run, **after P1 + N5a** | **Re-derive `T_res` at a second `p` — step 4 of §5.8, and hole 3 stated as a task.** At `rf`=64 the `Λ ≥ 0.95` and `trips/commit ≥ 3` floors close against each other: a 9-unit window at `T_res` 82–90 where `Λ` clears by 0.001–0.007 while 28% of commits still floor to `I`=1 ([P5.2](fl_fwd_ft_practice.md#p52-execution-plan--to-a-zero-input-run) phase 4). **That derivation assumed `B_max` = the `ln 2.7` prior**, so it is worth redoing once **P1** supplies a measured `Φ*` at that `p` — `ρ_max` and `n_req` both move with it. **Predicted:** with a measured `Φ*` the window opens to ≥50 units and law C composes. **Falsified if** it stays ≤10 or stays empty — then `T_res` cannot be re-derived from the same closed form at a new `p`, and the gate's reachability floor, not the anneal, is what does not port | a `T_res` at a second `p` holding trips/commit ≥3 in every quintile at `Λ` ≥ 0.95 — **or** a statement of which floor binds and why no `T_res` satisfies both |
| **N5c** | 1 GPU, **after N5a + N5b + P1** | **The second-model run — the largest hole in the claim, and the only row that closes it.** Every run on record is DistilBERT + adapters at `rf`=16 and the three datasets differ in `p` by 1.4%, so **nothing in this work has been tested across `p`**, while `T_res`, the `Φ` rail, the probe cadence, the saturation warm-up, `h`'s chord ratio and `Φ*` itself are all sized at that one `p` (§5.8). Run FluxTune unchanged on a second architecture, supplying only model + PEFT scheme + compute budget. **Predicted:** the peak lands inside the `Φ` = 2.4–3.3 band and no learning knob is set by hand beyond N5b's re-derived `T_res`. **Falsified if** the peak lands outside that band — then `Φ*` is not a property of the model family either, the rail must be sensed per model, and §5.6's hypothesis fails on the axis it was proposed for | a scored run on a second architecture, and a `Φ` at peak against 2.82–3.00 |
| **W′** | node holding `003648` | **The unverified half of the watchdog fix.** The `I`-floor kill needs `ΔB ≤ 0.005` and three healthy runs now replay silent; **that it still fires on a true death is unverified**. `003648`'s run dir is node-local | replay `003648`, confirm it fires |
| **Score** | any CPU | **Score all three pairs once C, D and Y land** — peak, whether it clears the **FL target**, the vclock at which FluxTune passes v2's full-budget peak, **and whether the run ended on its own stop**, which no run has yet done under the shipped stack. **Drop the 0.015-of-peak bar as a headline**: N1–N3 all end 0.03–0.15 below peak, and a still-climbing run passes it trivially | a scored table for all three datasets, every row ending on `[BudgetStop] reason=saturation` |
| **B4** | any CPU | **The n=1 rule for `anchor`.** The first sense replaces the `ln 2` prior at maximum variance and was **46% low** on yelp-p; under `anchor` it sets `ρ*` alone. **Arm the probe's influence only from n ≥ 2**, keeping `ln 2` for the first 150 commits. *(The old `budget_stop_frac` margin framing is moot — the budget stop is not a termination rule.)* | a stated n=1 rule, replayed against `021735` and the three 08-20 controllers |
| **G** | any CPU, **after** the P-4 runs | **`read_instance_from_h5` returns rows in thread-completion order**, so shard row order is not reproducible across tokenizations and `guid` names the wrong row. `X`/`y` stay paired under one lock and nothing reads `guid`, so **no ledger number is wrong**; it waits because it re-orders every future shard against the caches the P-4 runs used | two tokenizations of one client agree byte-for-byte, and `guid` round-trips |

**Nothing above is blocked by an open hypothesis.** H-S, H-H, H-T and H-J are specs + discriminating
numbers in [P5.3](fl_fwd_ft_practice.md#p53-open-hypotheses); all are rung 1–2 and none needs a node. K-C
is closed. Two ideas are **standing-refused** and must not be re-proposed — probe selection by `|d|`, and
block-coordinate probing ([P6](fl_fwd_ft_practice.md#p6--dead-ends--do-not-retry) has both, with the
replacements worth trying if selection is ever revived).

---

## §4 — How to launch a run, and how to read it

### §4.0 Environment — required by everything

```bash
export FLAME_CONDA_ENV=test_fwdllm        # base lacks h5py; every preflight exits 2 without it
export FWDLLM_FD_SCALE_INVARIANT=1        # the FD-rescale preflight refuses without it
PY=/coc/scratch/dgarg/miniconda3/envs/test_fwdllm/bin/python
REPO=/home/dgarg39/flame
FW=$REPO/lib/python/examples/fwdllm
```

**`/home/dgarg39/flame` is LOCAL disk per node; only `/coc/scratch` is shared** — so every node needs its
own `git pull`, and "I did not find it on this node" is never evidence it was not run. The tokenizer cache
is on `/coc/scratch` at 101/101 for all three datasets, and launch directory no longer matters (`cache_dir`
and `sim_charge_profile_path` are both emitted absolute).

### §4.1 One command per node

```bash
N=<1|2|3|4>
cd $REPO && git pull
NODE_DRY_RUN=1 $FW/expt_scripts/nodes/run_node.sh $N     # every preflight, seconds, no GPU
tmux new -s p4 "$FW/expt_scripts/nodes/run_node.sh $N 2>&1 | tee ~/p4_node$N.log"
```

**Always `tmux`** — every slot outlives a login, and that is how the 2026-08-16 backprop ceiling was lost.
Node 1 = agnews pair · 2 = real yahoo → profile → pair · 3 = the same for yelp-p · 4 = the two backprop
ceilings, then free. **A pair normally stays on one node** so both runs are priced by one profile; putting
them on two nodes is safe only if you copy the profile and check the md5 (row **D**).

`run_node_p4.sh` pins everything: `rf`=16, cos audit **off**, `--num-trainers 100 --c 30 --agg-goal 10`,
per-dataset vclock and real-wall ceiling (agnews 48,000 · 10 h; yahoo 60,000 · 14 h; yelp-p 50,000 · 14 h),
`--eval-max-samples 10000` on both seq-256 datasets. `controller` = **FluxTune**: law C at `T_res`=300 with
**no `--rho-star` and no `--b-max`** — that is what makes it zero-input. `control` = **FluxTune-v2**:
`rm`/0.25 at `ρ*`=0.06 with `gate_rho_ref=setpoint` and `--phi-stop log_only`, deliberately, so P4.1's
past-the-stop counterfactual keeps being measured.

**Four env hooks, all defaulting to the shipped behaviour** — nothing below needs a code change:

| var | default | what it does |
|---|---|---|
| `P4_BMAX_POLICY` | **`anchor`** *(was `mean`; flipped 2026-08-22 to the settled combiner)* | the **latest** sense instead of the mean (§5.3) |
| `P4_SAT_STOP` | **`1`** | row **E**'s saturation stop. `0` reverts to the old stop set for an A/B |
| `P4_PHI_STOP` | `halt` | `log_only` makes **all three** stop reasons emit and keep training — the only way to see past the rail |
| `P4_GATE_S` | `1.5` | moves `s`, the only lever `Λ = 2B/s` allows on progress per unit budget. Floor ≈0.9 at `K`=10 |
| `P4_BMAX_PHIS` | *(module default `1.5,2,2.5,3,3.5,4`)* | re-ranges the probe grid. **`condition_fp` does not cover it** — same blind spot the sim profile had |
| `P4_RETENTION_EVERY` | *(unset = off)* | row **P3′**: emit `[Retention] cos(θ_t,θ_0)` every N commits |
| `P4_BMAX_EVERY` | `150` | the probe cadence — **and both saturation horizons ride it** (warm-up 3×, progress 1×), so lowering it is the only way a SHORT run reaches the stop. Pair with `P4_PHI_STOP=log_only` |
| `P4_MIN_INIT_FRAC` | **`0.9`** | the join barrier. `1.0` restores the set-exact first cohort at the cost of zero straggler tolerance |

`P4_ALLOW_AGNEWS_PRICING=1` overrides the missing-profile refusal (§4.5). `VCLOCK_OVERRIDE` /
`CEIL_OVERRIDE` shorten a run (§4.2).

**Why the two runs cost so differently.** The controller stops *itself* at `B ≥ f·B_max` and law C's length
comes from `(B_max, T_res, f)`, not from the budget. The control has no stop, so it runs its budget out.
**Shrinking a controller's budget does not shorten it — it voids it**, on `max_runtime_s`.

### §4.2 Short runs and smokes

```bash
REAL_BUDGET=1200 $FW/expt_scripts/nodes/run_node.sh 2               # real-mode SECONDS
VCLOCK_OVERRIDE=6000 CEIL_OVERRIDE=2.0 \
  $FW/expt_scripts/nodes/run_node_p4.sh agnews controller           # vclock SECONDS, wall HOURS
SMOKE=1 $FW/expt_scripts/nodes/run_node.sh $N                       # ~20-30 min, whole chain
```

Prefer the two overrides over `SMOKE=1` for a sanity run: `SMOKE` also swaps the watch config and routes
profiles to `smoke/`, which price nothing by design. **A short controller run ends on `max_runtime_s`, not
`[BudgetStop]` — expected, and the one gate a short run cannot check.** It also cannot fire a `[BmaxProbe]`
(cadence 150 commits). Everything else reads exactly as it will on the long run.

### §4.2b Launching a run — copy these, and the two ways node 4 lost a slot

```bash
cd $REPO && git pull
export FLAME_CONDA_ENV=test_fwdllm FWDLLM_FD_SCALE_INVARIANT=1
NODES=$FW/expt_scripts/nodes

# --- a full-length run (this is what N1-N3 ran)
tmux new -s p4 "$NODES/run_node_p4.sh <agnews|yahoo|yelp-p> controller 2>&1 | tee ~/p4_N.log"
# ^ anchor + saturation + rail 3.0 are the DEFAULTS now; add P4_PHI_STOP=log_only
#   only when the point is to measure past the stop.

# --- a SHORT controller run (row N4a'). Both overrides are load-bearing; see below.
tmux new -s p4 "P4_PHI_STOP=log_only \
    P4_BMAX_PHIS='1.05,1.1,1.2,1.3,1.5,2.0' VCLOCK_OVERRIDE=12000 CEIL_OVERRIDE=2.0 \
    $NODES/run_node_p4.sh agnews controller 2>&1 | tee ~/p4_N4a.log"
```

**Shortening a controller run has two independent floors, and 2026-08-21 hit both.**

| override | floor | why |
|---|---|---|
| `VCLOCK_OVERRIDE` | **≥ ~12,000** on agnews | the `[BmaxProbe]` cadence is **150 commits**, and agnews runs ~12,500 vclock/h at ~500 commits/h. 6,000 vclock buys ~120 commits — **no probe fires at all**, so a probe-grid run measures nothing |
| `CEIL_OVERRIDE` | **≥ 1.8 h** on agnews | the wall-clock preflight prices **law C's own 898-commit length**, not the shortened vclock: 3.42 trips/commit × (4.41 s + 0.77 s/trip) = **6,322 s**. Anything under that is `BLOCKED (exit 2)` and **nothing launches** |

That is the §4.1 rule — *shrinking a controller's budget does not shorten it* — showing up one step
earlier than expected. The refusal is cheap and correct; budget for the run law C thinks it is running.

**yahoo must run on node 2** — its sim charge profile is node-2-local and the launcher now *refuses*
elsewhere rather than silently pricing it on agnews (§4.5). Fix that permanently with row **D0**.

**Prefix every launch with `NODE_DRY_RUN=1` once** — seconds, no GPU, and it prints the generated config so
`b_max_policy`, `phi_stop`, `gate_safety_s` and `b_max_probe_phis` can be read back before the slot is spent.

**The resident-GPU check now refuses above 2 GB** (`EXPT_GPU_REFUSE_MB`, override `EXPT_GPU_ALLOW_PEER=1`);
between 500 MB and 2 GB it still only warns. On 2026-08-21 a peer job at **43.7 GB of 46 GB** let the run start
and then killed **85 of 100 trainers on CUDA OOM inside 90 s** — in `pin_memory` at
`base_data_manager.py:453`, before a single forward pass. Because `minInitialTrainers` equals
`--num-trainers`, the selector then returned `ends: []` on **12,937 consecutive** distribute cycles and the
run sat at 0 commits for its full 45-min grace. **Both halves now fail fast**: the join barrier is at 90 of
100 and logs a `WARNING` naming the shortfall, and the watchdog kills on 10 uncaught trainer exceptions —
which on N4b's own log is reached **52 s** after the first line, against 45 min.

**Expect `log_only` runs to run their full vclock ceiling** — nothing halts them, by design. That is the
point: they measure what lies *past* the stop. A `log_only` run ending on `max_runtime_s` is **correct**,
not void; the §4.4 gate-4 rule applies only to a run whose stop is armed. **Under `anchor` that is every
controller run**, armed or not, until row **S** lands — hole 1.

### §4.3 The watchdog

`_node_lib.sh` execs `watch_arm.py` as a side-car **per run** and **kills the run** on:

| predicate | default | why this and not accuracy |
|---|---|---|
| no new commit | 20 min steady-state, 45 min pre-first-commit | a genuine hang; the only unambiguous one |
| `I` floored at 1 over the last 200 commits **and `B` not advancing** (`--b-advance-min`, 0.005) | ≥ 90% | G-2's `003648` died at 98% — but the `I` share **alone** is not that death: the n_target gate drives `I` to 1 at the landing point by design, and it voided a healthy yahoo controller at 86% of `B_max`. `B` is the exact progress measure and rides on the same record |
| trips/commit < 3 **and** pool demand unmet > 50% | after 200 commits | same argument: `trips/commit` is `n_req/K`, and law C drives `n_req` down **by design** |
| any `rho_star == 0` | after 200 commits | a requirement of *zero*, not an absent one |
| `CRITICAL .* Uncaught exception` in the trainers log (`--trainer-deaths-max`) | ≥ 10, **from the first poll** | the one failure every predicate above is blind to — they all read the aggregator's record, where a run whose trainers never started looks like one that is merely slow. Reads 85 on N4b and 0 on N1–N3 |

**The dead-trainer predicate does not wait for the grace**, which is the whole point: N4b lost 85 of 100
trainers to CUDA OOM in 90 s and the watcher spent its full 45-min pre-first-commit grace before calling
it. Its tenth death lands 52 s into the trainers log.

**The watcher stops watching once the run reaches its OWN end** — `stopping run.` or `[BudgetStop]
action=halt` in the last 256 KB of the aggregator log. `--pgid` clears only when the whole launcher group
exits, which lags the aggregator by the teardown: yahoo `151619` finished cleanly at 19:02 and its watcher
fired the hang guard at 19:22, writing a false `arm_stall.json` and killing the teardown. That is the same
failure that cost the two 03:04 real-mode runs their profiles. **A stall file on a run that also has
`plots/` is that false positive, not a death.**

**Both rate predicates need their conjunct, and this is the lesson.** Fitting the 2026-08-20 agnews
controller's last 150 commits gives `n_req ≈ 89.4 − 88.4·B_frac`, i.e. `n_req` ≈ 5 at its own 0.95 stop —
a bare floor voids every controller run at any setting above ~0.5. That run was killed at `n_req`=18 with
demand met on **all 577** commits; the yahoo run at 964 commits / 86% of `B_max`; and yelp-p's valid run
ran at `I==1` on 100% while still gaining 0.034 of `B_max` per 200 commits. **Starving means the gate is
not being met. Asking for less is the controller working. Kill on `B` not advancing, never on the shape of
a healthy landing.**

**Commits are counted from `version_bump_census`, not `server_update`** — the latter exists only under
`--server-update-audit`, which the scored runs set and the real profiling run deliberately does not, so
reading it alone saw `commits=0` on two healthy 61- and 63-commit runs and killed both. The scan is
incremental (per-file byte offsets); re-reading each poll is O(run²) and a 14 h run would re-read ~1.6 TB.

**Deliberately NOT `converge_watch.py`** — it runs on held-out accuracy and needs `--target-acc`, which
would end the run on convergence, and a run that does not end on `[BudgetStop]` is void. Its signal is
backwards here: **holding a plateau is what the controller is supposed to do.**

`NODE_WATCH=0` disables it; `NODE_WATCH_ARGS="--max-hours <CEIL+1> --i-floor-frac 1.01"` drops the `I`
predicate alone (`1.01` is unreachable; setting `NODE_WATCH_ARGS` replaces `run_node_p4.sh`'s `--max-hours`
default, so pass both). **A run already running holds the old module in memory** — a patch does not reach
it; relaunch, or `kill $(pgrep -f watch_arm.py)`, which drops the hang guard too. A killed run leaves
`arm_stall.json` in its run dir and the node prints it.

> **Open (row B4's neighbour):** a commit pooled from ~5 of 100 trainers at the landing point is what the
> gate says is correct for a tiny step, but it is also where the server path has little trainer work
> amortising it — the sim-fidelity worry the old floor was reaching for. **Decide it on the runs, not in
> the watchdog.**

### §4.4 Reading a run, and the scoring rules

```bash
RUN=$(ls -dt $FW/experiments/run_* | head -1)
$PY $FW/expt_scripts/check_arm_health.py $RUN --expect-controller   # exit 1 = a gate is breached
$PY $FW/expt_scripts/replay_scoring.py $RUN                         # B, Lambda, Phi, A
```

Four gates, each of which cost a node and none of which is visible at launch — **a clean `--dry-run` is a
prior, not a guarantee.** Run this by hand at ~200 commits on a live run; `_node_lib.sh` runs it after
every run.

```
[DataBins] from the registry, 100% coverage, trainer-confirmed   # 150 hardcoded gave yahoo 8.6% of its data
no server_update with rho_star == 0                              # a step of length zero
trips/commit >= 3 per quintile                                   # FAILs on a healthy landing -- read G-2 signature
controller ends on [BudgetStop] reason=saturation|phi_fixed      # now passable; no run has passed it yet
```

`_node_lib.sh` also echoes the enactment lines after every run — `[ProbeDim]` `[FD] spacing`
`[probe_combine]` `[TrainableScope]` `[ServerStep]` `[CommitGate]` `[CosProbe]` `[DataBins]` `[Landing]`
`[BmaxProbe]`. They are cheap and they are the only way to catch a knob that did not take.

**Gate 3 is not decisive on its own, and FAILs on every healthy controller.** The `G-2 signature` line
beneath it (`I==1` share, pool demand unmet, `n_req`) is what separates a controller annealing on plan
from a starving gate — yelp-p's *valid* run read Q5=1.34 with `I==1` on 100% of its last 200 commits and
the pool demand met on every one. Read both. On a run
without `--server-update-audit`, gate 2 reads `UNREADABLE`, not `ok`. **Gate 4 became passable on 2026-08-22** and
`reason=budget` is now a `WARN` in its own right — the budget rule reads the sensor §5.9 retired. The `[BmaxProbe]` trajectory and
its **first firing commit** are printed too — on yahoo that index is itself a result.

**Scoring rules** (derivation: [P4.4](fl_fwd_ft_practice.md#p44-scoring-rules-for-any-ab)):

- Score **peak** accuracy and the stability columns, **never final** accuracy of a diverging run — ±0.045
  between byte-identical replicates past the turn, against ±0.0009 at peak.
- Compare across datasets on **`A` and per-vclock-hour** — never on `Λ` (different `p`), never per round
  (11.7× different bins/round).
- Read `B` as a fraction of `B_max` and `A` against P4's calibration **while the run is alive**. Both are
  exact at any horizon, so both failure modes are diagnosable ~20 commits in.

### §4.5 Sizing a budget, and the profile

**Size off measured rate, never off `expts/wall_clock_preflight.py`** — it prices every commit at a
dataset-independent 4.41 s, and pre-fix yahoo measured 45.4 s. `check_arm_health.py` prints a
`budget sizing` line converting any short run's rate into the vclock 898 commits will cost.

| run | commits/h | vclock/h | 898 commits need |
|---|---|---|---|
| agnews `130614` | 351 | 17,523 | 44,839 vclock · 2.6 h |
| yahoo `125713` (no `--eval-max-samples`) | 79 | 5,470 | 61,932 vclock · 11.3 h |
| yahoo smoke `162439` (`--eval-max-samples 10000`) | 282 | 10,088 | 32,082 vclock · 3.2 h |
| yelp-p smoke `162510` (same) | 296 | 10,556 | 32,044 vclock · 3.0 h |

**yahoo's old 4.4× was the eval tax, not seq 256.** Eval is still ~30% of run wall — a cadence choice now,
not a defect.

**A per-dataset profile is mandatory before a sim run is scoreable:** yahoo burns 0.658 real-s per
vclock-s against agnews' 0.255, so no cross-dataset per-vclock comparison is valid until each is profiled.
It needs a REAL-mode run — `profile_sim_charges.py` pools `vclock_charge` events with `time_mode == "real"`
and finds nothing in a sim run — and both flags must **match the scored runs**:

```bash
cd $REPO && $FW/expt_scripts/run_sequential.sh --only fluxtune --mode real --dataset yahoo \
  --yes --clean --no-cos-ground-truth-audit --eval-max-samples 10000 \
  --num-trainers 100 --num-gpus 8 --agg-goal 10 --c 30 \
  --adapter-reduction-factor 16 --max-runtime-s 3000
cd $FW/expt_scripts && $PY profile_sim_charges.py \
  --real-run $(ls -1dt $FW/experiments/run_*yahoo*real* | head -1) \
  --out ../sim_charge_profiles/fluxtune_yahoo.yaml --only-observed
```

**A missing profile is now a refusal, not a fallback.** `run_node_p4.sh` used to add `--force` for any
non-agnews dataset lacking its own profile — which also disables `matches dataset`, so the run ran priced
on agnews' **0.255** real-s per vclock-s against yahoo's **0.658**. `condition_fp` does not cover the
profile: a yahoo run launched on node 3 with no profile reads the same `7174b984` as node 2's correctly
priced control, so nothing downstream catches the mismatch. It now exits 2 and names the file;
`P4_ALLOW_AGNEWS_PRICING=1` restores the old behaviour with two warning lines.

**Read its `WARN` lines, never `--force` past them.** A refused entry keeps its prior (agnews) value — a
*known* mis-pricing rather than a plausible wrong one. The guard scores the mass carried by the top 1% of
samples, not the single largest: at n=47 a top-1 test caught the cos probe and at n=489 it did not, because
three stalls of 21% each sat under the 25% threshold.

**`--allow-stale-profile` is not `--force`.** The staleness preflight globs the **local** `experiments/`
for reals newer than the profile's sources, and that directory is node-local — the same profile, config and
code pass on a node with no old reals and block on one that has them (`kaylee` blocked where `jayne`
passed). `run_node_p4.sh` downgrades that one check and leaves the other nine armed. `--force` would also
disable `sim charge profile matches dataset`, which is the config-derived check that actually protects the
vclock. And re-profiling agnews is wrong: `fluxtune.yaml` is what every historical agnews run and P4's own
calibration were priced against.

### §4.6 Adding a dataset

**In order:** a `configs/datasets.yaml` row → `build_niid_partitions.py` → `check_partitions.py` →
**`pretokenize_dataset.py`** → a real-mode run → `profile_sim_charges.py` → the run.

```bash
$PY $FW/expt_scripts/pretokenize_dataset.py --dataset NAME --clients 100 --dry-run   # missing + size
$PY $FW/expt_scripts/pretokenize_dataset.py --dataset NAME --clients 100 --jobs 16   # ~9 s/client @ seq 256
```

A cache file is one client's tokenized shard, keyed by everything that changes its tensors:
`{model_type}_{model_name}_cached_{max_seq_length}_{model_class}_{dataset}_{partition_method}_{client_id}`.

- **`partition_method` carries both `C` and α**, so α=1 and α=100 are different files and a
  `--partition-method` switch is a MISS, never a stale hit. **An α ablation must be pre-tokenized first**
  or it pays ~30 min inside its own wall budget. Tokenized today: agnews `alpha=1` + `uniform` (101 each)
  and `alpha=0.1` (90, partial); yahoo and yelp-p `alpha=1` (101 each).
- **`client_id` is the trainer's `client_idx`, not its trainer id** — `runner.py:389` sets
  `client_idx = (trainer_id − 1) % client_idx_modulo` (100 in every fwdllm yaml), so 200 trainers wrap onto
  the same 100 shards. `client_id = −1` is the aggregator's global test set, which `agg_eval` needs.
- Override the location with `FWDLLM_CACHE_ROOT`. Skipping this costs ~32 of 44 wall minutes inside the
  run's own budget.

**Backprop ceiling first (§5.1).** `probe_backprop_ceiling.py --config <an aggregator_config.json>
--dataset NAME --clients 10 --epochs 3`, and **`tee` it** — the script only prints, and the 2026-08-16 run
was lost to a closed terminal. ≈0.70 clears the data path; ≈0.30 indicts it and that dataset's runs measure
nothing.

### §4.7 Reading the new runs — what to look for first

For every `log_only` run, in this order:

```bash
RUN=$(ls -dt $FW/experiments/run_* | head -1)
grep -ao "\[BmaxProbe\][^|]*" $RUN/*aggregator.log | tail -20   # sensed trajectory + the curve
grep -ao "\[BudgetStop\][^|]*" $RUN/*aggregator.log   # reason=saturation|phi_fixed|budget
grep -ao "\[SatStop\][^|]*"    $RUN/*aggregator.log   # row E's detector, armed at 3x the cadence
grep -ac "Uncaught exception" $RUN/*trainers.log                  # >0 means trainers died; see 4.2b
$PY $FW/expt_scripts/check_arm_health.py $RUN                    # gate 4 is now decisive
```

**`[BudgetStop]` came back empty on every pre-08-22 `landing` run and that was the bug, not the run** — the
fixed-`Φ` test was the `else` of the budget test, so `log_only` never emitted the crossing it was launched
to measure. Recover it offline from `acc_budget` (`Φ = e^B`) for those runs; new runs emit it.

Then rebuild the accuracy-vs-`B` picture, which is what the predictions are stated against:

```bash
cd $FW/expt_scripts/writeup_figs
# add the new run to RUNS in extract.py, then:
python3 extract.py --force && python3 make_figures.py 7
```

**The question each run answers is in its §3 row, with a falsifier.** Read the falsifier first — a run
that fails its prediction is worth more than one that confirms it. N2 is the case in point: it was
predicted to reach 0.734 and read **0.663**, and that single falsification retired the tail-slope model,
yahoo's "still climbing" status, and hole 4's motivation in one stroke (§5.5).

### §4.8 The figure pipeline

`expt_scripts/writeup_figs/` renders every figure in the writeup. It exists so numbers are never copied:

| file | what it does |
|---|---|
| `ledger.py` | **parses P4's run ledger out of `fl_fwd_ft_practice.md`** — the ledger stays the one source of truth, and a figure can never drift from it |
| `extract.py` | one pass over the ~25 GB of aggregator telemetry into `data/*.json` (accuracy-vs-vclock, accuracy-vs-`B`, per-commit `ρ`/`B_frac`, every `[BmaxProbe]` fire + its curve). `--force` re-scans. **Nine runs cached**: `<ds>_control` = v2, `<ds>_controller` = the 08-20 `mean` runs, `<ds>_anchor` = N1–N3 |
| `figstyle.py` | palette + rcParams. Three-slot categorical, validated all-pairs; colour means **dataset** |
| `make_figures.py` | `./make_figures.py` for all twelve, `./make_figures.py 4 7 12` for individual ones |

**To add a run:** put it in `extract.py`'s `RUNS` dict, `./extract.py --force`, re-render. Nothing else.

---

## §5 — Standing facts · do not re-derive any of these

### §5.1 The dataset substrate *(landed 2026-08-12)*

| | agnews | yahoo | yelp-p |
|---|---|---|---|
| classes | 4 | 10 | 2 |
| `p` at `rf`=16 / 64 | 450,340 / 118,348 | 454,954 / 122,962 | 448,802 / 116,810 |
| official split | 120,000 / 7,600 | 1,400,000 / 60,000 | 520,000 / 40,000 |
| shard at `C`=100 | 1,200 / 76 | 14,000 / 600 | 5,200 / 400 |
| **bins/round at `C`=100** | **150** | **1,750** | **650** |
| token length p50/p95 | 41 / 70 | 84 / 367 | 137 / 493 |
| `max_seq_length` | 192 | 256 | 256 |
| **FL target** — the goal for a 100-client non-IID forward-only run | **0.880** | **0.660** | **0.820** |
| **backprop ceiling** — centralized, 10 clients × 3 epochs. **A plumbing diagnostic, not a target** | 0.850 | 0.734 | 0.874 |
| niid groups on disk | α = 0.1…100, `C`=100 | α = 1, 100 at `C` = 100 and 1000 | α = 1, 100 at `C` = 100 and 1000 |

`configs/datasets.yaml` + `expts/dataset_registry.py` hold every one of these. Partitions pass
`check_partitions.py` 6/6 on all three. Data plumbing is exact: `bins × 8 × C == n_train` at `C` = 100 and
1,000, verified in `test_dataset_launcher.py` and logged in-run as `[DataBins] coverage`.

**Three facts that change how a non-agnews run is read:**

- **Bins/round differ 11.7×.** Anything expressed *per round* — `max_data_id_progress`, a `data_id` sweep,
  an epoch — is not comparable across datasets. Score per **commit** or per **vclock-hour**.
- **`max_seq_length` 256 is a cost choice**, covering ~p89 of yahoo and ~p82 of yelp-p; trainer wall is
  ~linear in it, so a yahoo pass costs ~1.33× an agnews one **before** the 11.7× in bins.
- **`Λ` does not transfer across `p`**, and `p` differs by dataset. **Use `A`.**

**No attention mask anywhere in the stack** — `tc_transformer_trainer_distribute.py:713` and `:950` both do
`x = batch[1]; self.model(x)`, dropping `batch[2]`, and the backprop probe does the same, so the probe is
*faithful to production*. The model attends to PAD tokens on every run. **Rung 1 came back 0.734, so the
mask is not the fault** — it depresses both sides of every comparison equally. A candidate for absolute
accuracy, nothing more.

### §5.2 The yahoo gap — closed against the FL target, open against the ceiling

**Settled 2026-08-21: yahoo reaches 0.663 against an FL target of 0.660, so the gap that mattered is
closed.** Controller `125003` read 0.657 at `Λ`=0.994, closing the `Λ`-transfer question; N2 then spent
0.40 more `B` under `anchor` and gained +0.005. **The 08-20 reading — "it was budget, and yahoo is the one
dataset where budget still binds" — is retracted**: yahoo is as saturated as the other two.

What remains is **0.071 against the centralized backprop ceiling**, a comparison that was never
like-for-like (10 clients centralized against 100 non-IID, forward-only). Data plumbing and the data path
are cleared — bins exact, and centralized AdamW on the FL rig's own path returns 0.733 flat from epoch 1
against untrained 0.102, **which is all the ceiling was ever for.**

**What still bounds absolute accuracy on yahoo, in order** — every obvious suspect is cleared (`p`,
`max_seq_length`, `num_labels`, `learning_rate` inert under `trust_ratio`; `G_rule`, `s`, `P`,
`probe_combine` dataset-free), **and budget is now cleared with them**: **(a)** the `ρ*` band was sized on
agnews and a 10-class head may need a larger relative step to leave its init; **(b)** `train_batch_size`=8
means each JVP is estimated on a batch missing most of the 10 classes; **(c)** seq 256 truncates ~11% of
yahoo documents. Row **R2** is the diagnostic and it is no longer urgent.

### §5.3 The controller's settled constants

| | value | why |
|---|---|---|
| anneal law | **C** — `ρ*_t = min(ρ_max, √(2·(B_max_t − B_t)/T_res))`, `T_res` a rate **never decremented** | A makes `ρ*` constant under perfect tracking and smuggles `T` back as an input; B is a receding horizon that never terminates. C approaches `B_max` monotonically **from below** — **but only if `B_max` stands still.** Measured 2026-08-21: with `B_rem` pinned at `r`, law C degenerates to the constant `ρ* = √(2r/T_res)` = 0.0386, i.e. **law B wearing law C's clothes**. Observed ρ is flat at 0.034 ± 0.002 over the last 70% of all three runs, a sawtooth reset upward at every probe fire. **C is only an anneal to the extent the sensor is honest** — rows **N4a′** and **A** |
| `T_res` | **300** | 500 refuted on replay: 2.46 trips/commit on yahoo, 2.26 on the `ln 2` prior, against the ≥3 gate |
| `f` (stop at `B ≥ f·B_max`) | **0.95** — and **unreachable under `anchor`** | with `B_rem` pinned at `r`≈0.25, `B ≥ f(B+r)` needs `B ≥ f·r/(1−f)` = **4.75** (`Φ`≈116). Not a margin to re-tune: the budget stop is simply gone, which is why row **E** must land before any run is armed |
| `ρ_max` | `s·√(max_iter·K·G_rule/p)` ≈ **0.0999** | gate reachability, mechanical. **Not** a `ρ* ≤ ρ*₀` clamp — that would block a re-sense from spending the budget it just found |
| `B_max` prior | `ln 2` | replaced outright by the first sense, never averaged into it |
| `B_max` origin | **`B + ln Φ_knee`** | the probe measures headroom from `θ_t`; `B` accumulates from `θ_0` |
| `B_max` combiner | **`anchor`** — the latest sense *(decision 2026-08-20, run 2026-08-21)* | `mean` assumed the fires estimate **one constant**; 40 fires say otherwise — `B_rem` is flat at 0.21–0.30 on all three, so `mean`-minus-`B` collapsed and annealed `ρ*` **1.7× below** the live measurement. **Confirmed as a fix and as a non-event:** it roughly doubled `B_max` (0.795→1.694 on agnews) and bought **+0.006 / +0.005 / −0.002** accuracy and no speed-up (§5.5). Its known defect — it does not terminate — is now the load-bearing one. `b_max_policy=anchor`, no code change |
| what the stop does | **`halt`** via `_work_done` | one line into a tested path. Three states ship: `off` · `log_only` (emit the crossing, keep training) · `halt`, and as of 2026-08-22 all three stop REASONS honour it. Until then `_check_budget_stop` was `if landing: budget-test else: Φ-test`, so on a controller run the Φ branch was unreachable and N1–N3 emitted no crossing at all |
| `Φ` rail | **3.0** — `PHI_RAIL_DEFAULT`, the code default *(2026-08-21; was 2.7, and 2.7 never actually ran)* | measured under law C: peaks land at **2.82 / 3.00 / 2.91**, so 2.7 costs 0.005 on all three and 3.63 costs 0.008–0.015. 3.0 sits on the peak band and coincides with where the sized saturation stop fires (§5.5). Replaying N1's budget, it crosses at **commit 1,152** |
| stop reasons | **`saturation` primary · `phi_fixed` as the rail · `budget` demoted** *(decided 2026-08-20, sized 2026-08-21, shipped 2026-08-22)* | The run must end because **learning** stopped, not because a cumulative total was reached. `B ≥ f·B_max` is no longer a termination rule — `B_max` stays only to drive law C's `ρ*`. The three are an **`OR`** tested in that order; `saturation` is behind `--saturation-stop` (default off in code, **on** in `run_node_p4.sh`'s controller arm) and the other two ship enabled. The detector is **GL + Prechelt's progress term**; GL alone gave up 0.153 out of sample (§5.5) |
| is `B_max` a fixed total at all? | **no — as the sensor currently reads it** | `B_rem` is flat at 0.21–0.30 over 40 fires, so `B_max` is `B` + a constant and "spend `B_max` then stop" has no fixed point. **The cause is the instrument, not the world (§5.9)** — it measures tolerance to *unearned* noise on a frozen model, ~1.8× below where a re-fitting model peaks. Row **N4a′** makes it honest; rows **P1**/**P1′** make it right |

**`Λ = 2B/s` is an identity wherever the gate holds `s`** (−0.3% out of sample on both `s`-pinned runs,
+21.5–23.3% where `s` drifts). **So the `ρ` schedule is `Λ`-neutral at fixed `B`** — law A and law C bank
the same `Λ` and differ only in commits spent. Any "this schedule learns more" claim is a comparison at
unequal `B`.

**Real-wall cost model** (±1% over 5 runs): `wall = 7.81·commits + 0.77·trips` with the audit on at stride
25. Audit-off is `4.41 s/commit + 0.77 s/trip` — a subtraction, not a measurement.

**Still-binding cautions.** (a) `ρ`=0 is a requirement of *zero*, not an absent one. (b) Under `β > 0` the
`Φ` law changes — refuse to launch. (c) `Φ` from `ρ` is exact; never re-derive it from `‖θ‖` ratios.
(d) `Φ`-stop and budget-exhausted are the same trigger **only for a `B_max` that stands still** — the code
collapsed them into an `if/else` on that basis and `anchor` broke it, since `B_max` tracks `B` upward. Ship
them as an `OR` (row **S**).
(e) The 3.1 probe must reuse the cos probe's fixed-seed reference batch **and its class-skew guard** — with
the audit off on P-4 runs that guard is otherwise not running at all. (f) **Do not wire `n_eff` to
anything** — it is an identity, 1.00 ± 0.01 over 17 runs. (g) `dynamic_kc`'s `k_max`=15 is backwards and
must not be reused as a starting point.

**Landed and closed:** hill-climb `C`, not `K` (commit throughput is flat in `K` at fixed `C`); `P` is
compute-bound (`τ(30)/τ(10)`=2.56), so adaptive `P` is no longer motivated as a throughput lever — a
**mid-run `P` change**, which no code path supports, is the only engineering left there.

### §5.4 What the runs still have to answer

| open | closes on | if it comes out wrong |
|---|---|---|
| **Is `Φ*` a property of the MODEL or of the TASK?** — the biggest one, and the cheapest (§5.6) | row **P1**: `probe_inflation_damage.py` on a bracketing grid at `rf` 16 / 32 / 64 × 3 datasets | if it is a model constant, the per-run sensor, the combiner and the receding target all become unnecessary — **`B_max = ln Φ*`, measured once** |
| **Can any forward-only probe read `Φ*`?** The shipped one cannot — it measures tolerance to *unearned* noise and reads chance from `Φ`=2 up, while the trajectory at `Φ`=2 is at 0.97–0.99 of peak (§5.9) | row **P1′** — inject, run `m` commits, *then* read | if the knee is still ≤1.6 at `m`=150, `Φ*` is only knowable after a run, and row **E**'s saturation stop becomes the primary sensor rather than a backstop |
| **Is `Φ*` the same in centralized forward-gradient training?** §2.6 says heterogeneity is a step-size multiplier only, so it should be | row **P2′** — same estimator, 1 client, IID | a centralized peak outside [2.4, 3.3] makes `Φ*` partly federated and voids §5.6's "property of the model" framing |
| **Is retention really `1/Φ`?** The whole angular reading (peak at ≈70° of drift) is *derived* from perpendicularity and has never been logged | row **P3′** — one dot product per commit against a stashed `θ_0` | if `cos(θ_t,θ_0)·Φ` > 1, the aligned 7% overlaps `θ_0` and `Φ` overstates the damage |
| **Does a bracketing grid make `B_rem` shrink with `B`?** If yes, hole 1 closes with hole 2 | row **N4a′** | a still-flat `B_rem` means budget really is re-earned, and the anneal must be imposed rather than sensed (row **A**) |
| **Does `s` shift the accuracy-vs-`Λ` curve, or only move along it?** `Λ = 2B/s` says *along* | N4b′ at `s`=1.0 against N1 at matched `Λ` | a higher peak than 0.872 makes `s` a real accuracy lever — and the first thing to try on yahoo and yelp-p |
| **Is the residual gap to the *centralized ceiling* the estimator?** Not urgent — all three now sit within 0.008 of their FL targets | row **R2**: cos audit + `D` against agnews' 0.10–0.15 | if `D` matches agnews, the limit is adapter capacity, not the estimator |
| **Does the saturation stop hold on a fourth dataset?** Both horizons are multiples of the probe cadence and neither moves any fire commit (E2, closed 2026-08-22), and the rule now survives six curves it was not sized on — but **thr 0.005 and patience 20 are still read off three curves** | **C**, **D**, **Y** first: no run has ever ended on it. Then any new dataset | a threshold tuned to three tasks is a hand-set constant by another name, and it would be the only one left in the loop |
| **trips/commit ≥ 3** is calibrated on one death and two survivals | every run reports it per quintile; re-size once there are ten | a config passes preflight and still burns wall |

**Closed 2026-08-22:** the saturation warm-up is a multiple of the probe cadence, not a fit — 400, 450 and
600 give the same three fire commits (row E2) · the three stop reasons are an `OR`, so the rail is
reachable on a `landing` arm and crosses N1's budget at commit 1,152 (row S).

**Closed 2026-08-20/21, do not re-open:** `Λ`→accuracy transfers across task (yahoo 0.657 at `Λ`=0.994) ·
the combiner is **`anchor`**, not `mean` · `f`=0.95 is **moot** as a termination rule (§5.3) · **the `Φ`
cliff does not move under law C** — N1–N3 peak at 2.82–3.00 and lose 0.005 within 0.1 of the peak, inside
P4's historical 2.41–3.11 band · **no dataset has slope left in `B`** (§5.5) · **law C does not anneal
under a receding `B_max`** — it degenerates to a constant `ρ`, measured flat at 0.034 on all three (§5.3).

### §5.5 Saturation, and where every run peaks *(measured 2026-08-21 on N1–N3)*

**Method, so it is reproducible.** Accuracy is joined to `B` by walking the aggregator jsonl **in emission
order**, incrementing the commit count on each `server_update` and stamping every `agg_eval` with the
running `B = ½Σln(1+ρ²)` — exact, where a timestamp join would be approximate
(`writeup_figs/extract.py:acc_vs_budget`). Peaks and losses below are on an **11-eval trailing mean**, the
same smoothing the saturation stop uses; raw peaks are within 0.005 of them.

**Every run peaks in the same place, on all three tasks.**

| | peak acc | at `Φ` | at `Λ` | at commit | end `Φ` | end acc | given back |
|---|---|---|---|---|---|---|---|
| **agnews** | 0.872 | **2.82** | 1.38 | 1,052 of 1,913 | 4.65 | 0.843 | **−0.029** |
| **yahoo** | 0.660 | **3.00** | 1.46 | 1,230 of 2,400 | 6.00 | 0.523 | **−0.137** |
| **yelp-p** | 0.807 | **2.91** | 1.43 | 940 of 1,863 | 5.17 | 0.701 | **−0.107** |

A band of **0.18 in `Φ`** and **0.08 in `Λ`** across 4-, 10- and 2-class tasks, and it sits inside P4's
historical 2.41–3.11 band measured under the *old diverging* dynamics. **The peak location is the most
transferable number in this work.**

**Where the peak is lost, smoothed** — this is the cliff at better resolution than P4's ledger gives:

| loss from peak | agnews | yahoo | yelp-p |
|---|---|---|---|
| −0.005 | `Φ` 2.99 | `Φ` 3.11 | `Φ` 2.94 |
| −0.010 | `Φ` 3.75 | `Φ` 3.33 | `Φ` 3.21 |
| −0.020 | `Φ` 3.93 | `Φ` 3.81 | `Φ` 3.55 |

**The decay starts immediately after the peak and is gentle to ≈3.3, then steepens.** P4's "held peak to
3.63" is compatible — it used a 0.014 tolerance — but the honest rail is **3.0**, not 3.63 and not 4.23.

**Three consequences, and they are the reason the queue looks the way it does.**

1. **No dataset has slope left in `B`.** Fitting `dAcc/dB` over the last 20% of the 08-20 runs and
   extrapolating forward predicted +0.019 / +0.092 / +0.004; the runs that spent that budget
   (+0.35 / +0.40 / +0.11 in `B`) measured **+0.006 / +0.005 / −0.002** — over by **3.4× / 17× / wrong
   sign**. **A tail slope fitted inside the rise does not survive past the plateau — do not extrapolate
   one again** (§6, failure mode 6, now with a second instance).
2. **Un-throttling `ρ*` bought commits, not learning, exactly as `Λ = 2B/s` requires** — and not even
   speed: time to v2's own peak went 8,257 → 8,604 vclock on agnews. The combiner throttle cost **hours on
   all three and accuracy on none**. Schedule-neutrality is now measured twice, on independent runs.
3. **Slowing down and stopping are the whole remaining problem, and they are one problem.** The runs
   neither annealed (ρ flat at 0.034 over the last 70%, `Σρ²` after commit 400 = 2.09 / 2.72 / 2.23
   against 0.51 / 0.53 / 0.99 for the `mean` runs) nor stopped (both rules inert), and **both follow from
   the pinned `B_rem`** — §1's holes. Nothing else on the queue is worth more.

**Against the FL targets this reads very differently.** Peaks are 0.873 / 0.663 / 0.812 against
0.880 / 0.660 / 0.820, so the controller is **within 0.008 everywhere and over on yahoo**. What the missing
stop costs is not the target — it is the 0.03–0.14 given back afterwards, and 40–60% of the wall clock.

**The saturation stop, sized by replay on these three curves** (row **E**, shipped 2026-08-22 as
`expts/saturation_stop.py`): 11-eval trailing mean · `GL_t = (Acc_best − Acc_t)/Acc_best` ·
**threshold 0.005** · **patience 20 evals** · **armed after `3 × b_max_probe_every` = 450 commits**.
Reproduce the table below with `expt_scripts/replay_saturation_stop.py`; `test_saturation_stop.py` is
the gate.

| | fires at | vs peak | run saved |
|---|---|---|---|
| **agnews** | commit 1,194 | −0.005 | 38% |
| **yahoo** | commit 1,084 | −0.008 | 55% |
| **yelp-p** | commit 1,179 | −0.007 | 37% |

**It was re-derived on 2026-08-22 against six curves it had never seen, and the
first version failed them.** GL alone fires on `yahoo_control` at commit 597 and
**0.153 below that run's eventual peak** — a run still crawling at 0.27 accuracy
that went on to 0.42. A running-max GL is scale-free but not slope-aware: it
cannot tell a plateau at the top from a slow noisy climb, because both sit below
their own running max for 20 straight evals. Adding **Prechelt's own progress
term** — do not stop while the trailing mean is still above where it was one
probe-cadence ago — removes that fire and every other out-of-sample one, at a cost
of 0.000 / 0.000 / 0.002 on the three curves that sized it. **Neither constant is
fitted now: warm-up = 3 × cadence, progress horizon = 1 × cadence**, and 1× / 1.5×
/ 2× all give the same out-of-sample verdict.

**The warm-up stopped being load-bearing when the progress term landed, and the honest statement is that
it is now inert.** It used to be the only thing preventing a false fire — at 300 the detector fired on
yahoo's early plateau at commit 346 and 0.24 accuracy. The slope test now catches that case, and the
warm-up moves no fire commit on any of the nine cached curves **at any setting from 0 to 600**. It ships
anyway, as `3 × b_max_probe_every`, because it costs nothing and a pathological curve could still need it —
but do not cite it as the reason the rule does not false-fire. **What still is fitted: the 0.005 threshold
and the 20-eval patience**, and the six out-of-sample curves are the only evidence they generalise.

**Note the two stops now agree.** Saturation fires at `Φ` 2.76–3.27 and the re-derived rail is 3.0. That is
the argument for shipping them as an `OR`: on these runs either one alone would have been nearly right, and
together they cover the case where a curve is noisy near its peak.

---

### §5.7 What a new dataset costs the operator *(the zero-input claim, itemised)*

| supplied by hand | mechanically derived | universal constant |
|---|---|---|
| a `configs/datasets.yaml` row (h5 paths, `num_labels`, `max_seq_length`, split sizes) — a *description of the data* | `dataset` / `data_file_path` / `partition_file_path` / `max_seq_length` into both override blocks (`--dataset`) | `probe_combine=mean` · `server_step_rule=trust_ratio` · `commit_gate=n_target` · `gate_rho_ref=annealed` |
| a partition build (`build_niid_partitions.py`) + `check_partitions.py` | `num_labels` from the h5 label vocab | `s`=1.5 · `T_res`=300 · `b_max_policy=anchor` |
| **a compute budget** (`max_runtime_s`, `sim_wall_ceiling_s`) | `p` from `[ProbeDim]`; `total_data_bins` from the registry | `P`=10 · `K`/`C`=10/30 |
| `eval_max_samples` — a **cost** knob, not a learning one | `ρ*_t` from law C · `ρ_max` from gate reachability · `n_req` closed-form · the saturation warm-up as `3 ×` the probe cadence | `B_max` **prior** `ln 2`, replaced by the first sense · the `Φ` rail **3.0** |
| a sim charge profile (**sim-only artifact**, needs a real run) | `B_max` itself — **sensed** by the 3.1 probe | — |


Only the first column is input, and **none of it is a learning knob**. `eval_max_samples` is a cost knob;
`max_runtime_s` is how long you are willing to pay. No learning rate, no variance threshold, no cohort
width, no run length, no target accuracy (§6.7), no probe count, no safety factor.

---

### §5.6 What `Φ` is, and what it is a property of *(opened 2026-08-21)*

**Two different quantities share the letter, and conflating them is the easiest error here.**

| | what it is | how it is obtained | costs |
|---|---|---|---|
| **`Φ_t`** — the **inflation ratio** *(was "the odometer"; retired, §5.6a)* | `‖θ_t‖/‖θ_0‖`, how far the model has rotated off its starting point | **pure arithmetic on the step sizes**: `B = ½Σln(1+ρ_t²)`, `Φ = e^B`. Exact because the step is ⟂ `θ` (cross-term 1.000 ± 0.005). **No model, no data, no accuracy enters it** | zero |
| **`Φ_knee`** — the wall | how much isotropic noise *this* model's weights tolerate before normalized accuracy halves | **measured**, `expts/bmax_probe.py:knee` — inject noise scaled to inflate by each grid `Φ`, read held-out accuracy back, interpolate to the 0.5 level on `(acc − chance)/(base − chance)` | ~6 forward passes |
| **`Φ*`** — where peak accuracy lands | empirical, **2.82 / 3.00 / 2.91** | read off the runs | a run |

**So `Φ_t` is dynamic but not adaptive.** It is a running total the controller keeps for free; it does not
look at the model or the task. That is the whole point — it is the one quantity in the stack that carries
no units of `‖θ‖`, `‖g‖`, `p` or the label set (§6, scoring vocabulary), which is why a fixed number can be
compared against it at all. `Φ_knee` and `Φ*` are the numbers that *might* depend on something.

**The hypothesis, stated so it can be killed: `Φ*` is a property of the model + PEFT scheme, not of the
task.** The mechanism argues for it — `Φ` is exactly `1/retention` = `1/cos(θ_t, θ_0)` (§5.6a), so `Φ*`
is asking *how far the pretrained representation can be rotated before it stops functioning*. The **task**
decides how much `Λ` you bank per unit `B` and what accuracy that buys; it has no obvious reason to move
where the wall is.

**Evidence for, four independent lines:**

- **2.82 / 3.00 / 2.91** across 2-, 4- and 10-class tasks, at three very different accuracy levels
  (0.81 / 0.87 / 0.66) and three different bins/round (11.7× apart). A 0.18 band.
- The ledger's **`p` ladder**: `select rf=64` (`p`=118k) held its peak with `Φ`=3.04 and `rf`=32
  (`p`=229k) with `Φ`=3.28 — **a 3.8× range in `p`, both inside/adjacent to the band**, and peak accuracy
  barely moved (0.859 / 0.857 / 0.852).
- **22 historical runs**, 2 step rules, α 0.1–1, 66–3,353 commits: peaks at `Φ` = 2.41–3.11. **Nothing has
  ever peaked outside [2.4, 3.3].**
- `Λ` at peak is **1.38 / 1.43 / 1.46**, an even tighter band — and `Λ = 2B/s`, so at fixed `s` the two
  statements are the same statement.

**Evidence against, and it must be stated:** B-1's *offline* knee sweep read **agnews ≈3.0–3.5, yahoo and
yelp-p ≈2.0–2.3** — dataset-dependent, non-monotone in class count, and the measurement the whole "`B_max`
must be sensed" argument rests on. **But it is the injection probe, so it is measuring the *other*
quantity** (§5.9): a frozen-model knee, ~1.8× below where a re-fitting model peaks, on a grid that never
brackets it. The offline spread is therefore **not comparable to the runs' `Φ*` at all** — it is a spread
in a badly calibrated instrument, which is exactly what row **P1** re-measures on a bracketing grid.

**Why this matters more than anything else in the queue.** If `Φ*` is a model constant, the controller
collapses: **measure `Φ*` once per model with forward passes, then spend budget to it.** No per-run
sensing, no combiner, no receding target — and the anneal and the stop both come back for free, because
`B_max = ln Φ*` is a genuine fixed point. If `Φ*` moves with the task, the current architecture is right
and the probe just needs a working grid. **Row P1 discriminates these for about an hour of one GPU**, and
`scripts/probe_inflation_damage.py` already registers it as its own "Q1: absolute vs relative".

**Do not confuse this with the `Φ` cliff.** Peak accuracy *occurs* at `Φ*`≈2.9; accuracy is *lost* from
≈3.0 and badly past 3.6. The shipped 2.7 rail sits just below the peak, which is why raising it to 3.0
costs nothing and buys a little.

#### §5.6a `Φ` is a rotation, not a distance

**Derivation, the retention/drift-angle table, and the fixed vocabulary live in
[model §4.1a](fl_fwd_ft_solution.md) (R2).** What it gives this doc in one sentence: `Φ` is exactly
`1/cos(θ_t, θ_0)`, so **every run peaks when the trainable weights have rotated ≈70° off the pretrained
point, and dies past ≈75°.** Say it that way — it names a mechanism where `Φ`=2.9 names a number.
⚠ **Derived, not yet measured**; **row P3′** is one dot product per commit and settles it.

**Naming, fixed 2026-08-21:** "odometer" is retired for **inflation ratio** (`Φ`), `ρ` is the **trust
ratio**, `1/Φ` is **retention**, and the `Φ` rail is a **retention floor**. Full table in model §4.1a —
including why "the fraction that earned its accuracy" is the wrong gloss for `1/Φ`.

#### §5.6b How `Φ*` is measured — three operational definitions, and only one of them works today

| | definition | cost | status |
|---|---|---|---|
| **`Φ*_traj`** | `Φ` at the peak of an 11-eval trailing mean of held-out accuracy | **a full run** | **the only one that has ever produced 2.9.** 2.82 / 3.00 / 2.91 on N1–N3 |
| **`Φ_knee` (frozen)** | shipped injection probe: inject at `Φ`, read a frozen model | ~6 evals | **measures a different quantity** — 1.25–1.28, ~1.8× low (§5.9) |
| **`Φ_knee` (re-fit)** | inject at `Φ`, run `m` commits, *then* read | `m` commits × grid | **not built.** Row **P1′**. This is the one that could replace `Φ*_traj` |

**Why this matters for the zero-input claim.** `Φ*_traj` costs a run, so it cannot size that same run's
budget — which is exactly why the injection probe exists. If **P1** says `Φ*` is a model constant and
**P1′** gives a forward-only way to read it, the operator measures it **once per model, before any run**,
and `B_max = ln Φ*` becomes a genuine fixed point. If either fails, `Φ*` is only ever knowable *after* the
fact, and **row E's saturation stop is not a backstop — it is the primary sensor**, because a peak detector
is then the only instrument that sees `Φ*` at all.

#### §5.6c Is `Φ*` different in FL and in centralized training?

**Two separate questions, and conflating them is the trap.**

**(a) Does the wall move?** No reason it should, and nothing measured says it does. `Φ*` asks how far a
pretrained representation can be rotated before its head can no longer be re-fit — a property of the
weights and the PEFT scheme. Federation enters only through `α`, and the model doc's §2.6 measures heterogeneity as a
**step-size multiplier and nothing more** over 1000× in `α`; the 22-run ledger spans α 0.1–1 and never
peaks outside [2.4, 3.3]. **Row P2′ tests it directly** by running the same estimator centralized.

**(b) Does a run ever *reach* the wall?** **Only if it is forward-gradient.** This is the part that is not
symmetric, and it is the reason `Φ` is a FluxTune control variable rather than a general training one:

| | `cos(step, g)` | steps ⟂ `θ`? | `Φ` at target accuracy |
|---|---|---|---|
| **backprop, centralized or FL** | ≈ 1 | **no** — the gradient has a radial component, so `‖θ‖` is free to *shrink* | ≈ **1.0–1.2**. Never approaches the wall |
| **forward-gradient** (this work) | **≈ 0.07** | **yes**, to 1.000 ± 0.005 | `Λ = 2B/s` ⇒ banking `Λ`≈1.4 *requires* `B`≈1.05, i.e. `Φ`≈2.9 |

> **The wall is a property of the model; the *bill* is a property of the estimator.** Backprop banks the
> same `Λ` for ~14× less rotation, so it retires with 80% retention and the question never arises.
> Forward-gradient pays for its accuracy in drift, and `Φ*`≈2.9 is where the payment runs out. **This is
> also why the centralized backprop number in the writeup is a plumbing check and not a target** — it is
> not operating anywhere near the same point in this geometry.

---

### §5.8 What a new MODEL costs the operator — and what is still pinned to one `p`

**§5.7 itemises a new *dataset*. This is the same table for a new *model*, and it is shorter on the left
and much longer on the right.** Nothing here is measured yet — hole 3 — so read the third column as the
list of things that must be re-checked, not as a list of known failures.

| supplied by hand | mechanically derived at init | **pinned to DistilBERT + adapters at `p`=450k — must be re-derived** |
|---|---|---|
| the model + PEFT scheme (**it is the deployment**) | `p` from `[ProbeDim]` — the *production* count after the trainer drops `pre_classifier`, not `create_model`'s | `T_res` = **300**. Sized on one `p`; at `rf`=64 law C + the annealed gate did not compose under **any** `T_res` |
| the PEFT rank ⇒ `p` (a **device-memory** choice; measurably inert for learning and for time, §4.2 of the model doc) | `‖θ_tr‖` at init — 13.35 here, and `‖θ_tr‖ ∝ √p` to ±1.3% | the `Φ` rail = **3.0**, and `Φ*` ≈ 2.9 behind it. Consistent across a 3.8× range in `p` (3.04 at 118k, 3.28 at 229k) but **never tested on a second architecture** — this is what **P1** is for |
| a compute budget (`max_runtime_s`, `sim_wall_ceiling_s`) | `ρ_max = s·√(max_iter·K·G_rule/p)` — gate reachability, mechanical | `f` = 0.95. Moot as a termination rule (§5.3), but still in the code |
| — | `n_req` closed-form from `s` | probe cadence **150** and saturation warm-up **400**. The warm-up must be tied to the cadence, not to three curves (§5.5) |
| — | chance = `1/num_labels`, for the probe's normalization | `h` = 0.01. **`h‖v‖ = h√p` scales with `p`**, so the FD chord silently changes with the model — `FWDLLM_FD_SCALE_INVARIANT` exists for this and is the least-audited thing here |
| — | `B_max` **prior** `ln 2`, replaced by the first sense | `s` = 1.5. Derived from Cauchy–Schwarz, so it should carry — but `D` = 0.050 is folded into it, and `D` is a constant of *this* setting (§6.3 of the model doc) |

**The honest summary the writeup should carry:** a new **dataset** costs a `datasets.yaml` row, a partition
build and a compute budget — **no learning knob**. A new **model** costs those plus a re-derivation of
`T_res`, the rail, the probe cadence and `h`'s scale-invariance, **because all four were sized at one `p`
and one architecture.** Generality is demonstrated across *task*; across *model* it is a design intent with
one negative data point (`rf`=64) against it.

**Ordering, if someone actually ports this — and every step is now a queue row.** `p` and `‖θ_tr‖` (free,
at init) → `h`'s chord ratio `h√p/‖θ_tr‖` against this model's 0.50 (**N5a**) → `Φ*` (**P1** / **P1′**) →
`T_res` from the gate-reachability check at the new `p` (**N5b**) → only then a run (**N5c**).

> ⚠ **N5a exists because the ratio may already be broken on the ladder we have.**
> `FWDLLM_FD_SCALE_INVARIANT` holds `h√p` fixed at 6.7107 across `rf` = 16/32/64, but `‖θ_tr‖` falls
> 13.35 → 9.6 → 6.86 over the same ladder — so the *dimensionless* chord computes to **0.50 / 0.70 /
> 0.98**, nearly doubling. If that is what the logs say, the one flag that exists to enforce scale
> invariance is holding the united quantity fixed and letting the ratio move.

### §5.9 Injected inflation ≠ earned inflation *(measured 2026-08-21 on N1–N3; §1 hole 5)*

**The measurement.** Both columns chance-corrected and normalized to each run's own 11-eval-smoothed peak.
*traj* is the run's own accuracy at the commit where `Φ = e^B` hits the row's value; *inj* is the mean over
every `[BmaxProbe]` fire from commit 600 on. 40 fires, three datasets.

| `Φ` | agnews traj / inj | yahoo traj / inj | yelp-p traj / inj |
|---|---|---|---|
| 1.5 | **0.94** / 0.10 | **0.45** / 0.10 | **0.73** / 0.07 |
| 2.0 | **0.99** / −0.02 | **0.97** / 0.01 | **0.98** / −0.04 |
| 2.5 | **0.99** / −0.00 | **1.00** / 0.00 | **0.99** / −0.01 |
| 3.0 | **1.00** / 0.01 | **1.00** / −0.01 | **1.00** / 0.01 |
| 4.0 | 0.97 / −0.01 | 0.95 / −0.00 | 0.90 / −0.03 |

**Reproduce:** `writeup_figs/data/*_anchor.json` — `acc_budget` gives `(commit, B, Λ, acc)` and `probes[]`
gives each fire's `curve` and `base_acc`. No re-scan of `experiments/` needed.

**Three readings, in order of how much they change.**

1. **`Φ_knee`'s pinning is now doubly explained.** **35 of 40 fires** are already below the 0.5 level at the
   *first* grid point, so `knee()` interpolates off its synthetic `(Φ=1, 1.0)` anchor and returns
   `1 + 0.5·0.5/(1−n₁)` → 1.25. That is the arithmetic. *Why* the first point is low is this section.
2. **The mechanism is re-fitting, and it is the only difference between the two protocols.** Injection adds
   `‖θ‖√(Φ²−1)` of isotropic noise **in one shot** to a model then evaluated frozen. Training adds the same
   total length in ~1,000 increments **with the head re-fitting between every one**. Same end `‖θ‖`, same
   drift angle from `θ_0`, opposite verdict. **The gap therefore measures how much of the wall is
   re-fittable, and it is most of it:** 60° of injected drift is fatal, 70° of earned drift is optimal.
3. **N4a′ is demoted from "the fix" to "necessary."** A bracketing grid returns an honest 1.2–1.6 instead of
   a pinned 1.25 — worth having, because it is the difference between a target that recedes with `B` and one
   that does not. But it is still ~1.8× below `Φ*`, so `B_max` from this probe still anneals law C against
   the wrong number. **Rows P1 and P1′ are the fix.**

> **What the shipped probe is still good for:** ranking two models' tolerance to unearned perturbation,
> forward-only and cheap. **What it must stop being used for:** setting `B_max`.

**Closed by this, do not re-open:** "the probe is biased conservative by 0.6–1.2 in `Φ`" — it is not an
offset, it is a different quantity, and the 0.6–1.2 figure came from the offline B-1 sweep read against
offline knees rather than against trajectory peaks.

---

## §6 — Rules any change inherits, and the failure modes behind them

**Violating one is a rejected change, however good the result.**

1. **Flag-gated, default = old, byte-identical off.** Parity/correctness fixes are the exception — they
   ship enabled, including the code-level default. Terminal flag state (`PERMANENT`/`FLAGGED`/`REVERTED`)
   is the operator's call, never the implementer's.
2. **A flag is read in exactly one place and echoed once.** A flag both sides read (`perturbation_count`,
   `probe_combine`) goes in **both** override blocks and must agree — `test_model_args_parity.py` enforces
   it, and a new dual-read flag must be added to it.
3. **Emit-only is not free.** Anything added to the commit path gets timed before it ships. The cos audit
   cost eight runs by being "just logging".
4. **Probes do not modify `trainer/`, `aggregator/`, or any yaml on the critical path.** They import
   production code; a validated result transfers as a config flag, not a rewrite.
5. **Every new number needs predicted-vs-observed and a run id** before it reaches a ledger, and a task is
   done when its gate reproduces a number already in P3/P4 — not when the code runs.
6. **Dataset constants come from `expts/dataset_registry.py`.** Never re-hardcode `num_labels`, `p`, a
   class-balance threshold, or an h5 path.
7. **No stopping rule may take a target accuracy.** Stopping *at* a supplied target makes the target a
   knob and voids the zero-input claim outright — where the run lands is a **result**, not a setting. This
   is why `converge_watch.py` is refused as a watchdog (§4.3) and why row **E** stops on the *shape* of
   the accuracy curve, never on its level.

**Scoring vocabulary** (definitions in [P8](fl_fwd_ft_practice.md#p8--reproducing-any-number-from-logs);
all exact at any horizon): `ρ` step/norm ratio · `B = ½Σlog(1+ρ²)` budget spent · `Φ = e^B` norm inflation
· `Λ = Σρ√(G_rule·N/p)` progress · `A = Σρ√(G_rule·N/p)·‖θ_tr‖` absolute progress, the only one comparable
across `p`.

**The failure modes this plan is written against — every one has already happened once**
([P9.3](fl_fwd_ft_practice.md#p93-process-lessons) has the accounting):

1. A superseded constant left in a config — grep the configs in the same edit. `gate_safety_s`=0.4 cost
   three runs and a node.
2. An emit-only flag never re-costed after being made correct.
3. A sinking condition without its precondition or its smoothing rule.
4. An instrument whose arithmetic is right and whose **input** is not.
5. **Scoring a feature without scoring the composition.** Gate and anneal are each correct and multiply
   into a stall (`N_req ∝ ρ_t²`). Two new flags need a composition test before both default on.
6. **Extrapolating any rate as an accuracy rate.** Twice now: `A` accumulates *through* the turn while
   accuracy falls; and a tail `dAcc/dB` fitted over the last 20% of a still-rising run over-predicted the
   next 0.35–0.40 of `B` by **3.4× / 17× / wrong sign** (§5.5). A slope measured before the plateau says nothing
   about the plateau. Extrapolate only alongside `Φ`, and label it a prediction until a run has run it.
7. Two quantities with different origins, subtracted. State the origin of every accumulated quantity next
   to its formula.
8. A flag whose writer is not its only writer.
9. **A dataset constant that is right on agnews by arithmetic coincidence.** `total_data_bins = 150` lived
   in `lib/python/flame/mode/horizontal/syncfl/fwdllm_aggregator.py`, outside the example tree, and is agnews' `1,200/8` exactly — yahoo trained on
   **8.6% of its data, the same 1,200 rows every lap**, and nothing raised. **Grep the *derived* agnews
   numbers (150 / 1,200 / 7,600 / 192), not just the name, and grep `lib/python/flame/` too.**
