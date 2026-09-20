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

> **⚠ THE FOUR DOCS ARE ONE CORPUS — no inconsistency, no staleness, no redundancy.** Any session that
> measures something updates **every** doc the measurement touches, in the same session. One fact has one
> home: [buildplan](fl_fwd_ft_buildplan.md) owns status + the queue, [practice](fl_fwd_ft_practice.md) owns
> numbers (P3 knobs, P4 runs), [solution](fl_fwd_ft_solution.md) owns mechanism, [writeup](fl_fwd_ft_writeup.md)
> owns the prose account. Elsewhere a fact is **cited, never restated**. A measurement that contradicts a
> standing claim **deletes** that claim — it is never left standing beside its refutation, and never
> softened into "some evidence suggests". Retractions replace the retracted text and say what killed it.

**§3 is the single source of next steps.** The writeup names the same work in prose and points here.

---

## §0 — THE GOAL. Everything on this queue is judged against it

> **FluxTune fine-tunes a NEW model on a NEW dataset, forward-only, with no learning knob supplied by the
> operator — deriving its own step size, its own pool size, and its own stopping point from quantities it
> measures on itself. The operator supplies the model, the data, and a compute budget. Nothing else.**

**Two axes, and only one is demonstrated.**

| axis | where it stands | what closes it |
|---|---|---|
| **datasets** | 3 of 3 land within 0.008 of their FL target — but **none terminated on its own stop** | C · D · Y · **E firing live** |
| **models** | **ZERO FL runs on a second architecture.** Every number in this corpus is DistilBERT + adapters at `p`≈450k | **N5c**, and it is the single largest hole |

**Read the second row as the priority.** A fourth dataset on DistilBERT adds a data point to an axis that is
already demonstrated. A first *run* on roberta-large tests whether the self-derivation — `ρ_max` from gate
reachability, `n_req` in closed form, `T_res`, law C, the `Φ` rail, the saturation stop — survives a **9.4×
change in `p`** and a different architecture. **That is the claim.** Everything else is supporting evidence.

**How to plan against this.** Prefer one run that tests many things over many runs that each test one.
Every launched run should carry every free instrument it can (`P4_RETENTION_EVERY`, the probe, the health
gates) so a single slot answers several rows at once. **Sizing rules and the evidence tiers: §4.2a.**
**A row that only debunks something, without advancing model-or-dataset generality, is not worth a node
while the model axis reads zero.**

---

## §1 — The claim, and what is missing

> **FluxTune reaches and holds a plateau on a new dataset with no learning knob tuned by hand — same
> DistilBERT + adapters, three datasets, against a version of itself whose step size was hand-searched.**

**Three systems; use these names everywhere, figures included.** **FwdLLM** — prior work, variance gate,
raw SGD. **FluxTune-v2** — trust-ratio + `n_target`, but a **static `ρ*`=0.06 hand-searched on agnews**,
RM-decayed (`rm`/`setpoint`; the code already calls it `fluxtune_v2`). **FluxTune** — this work, `ρ*` from
law C on a **sensed** `B_max`. **backprop ceiling** — exact gradients, **centralized**, 10 clients × 3
epochs: a plumbing diagnostic, not a target (§5.1).

**Sensing `B_max` is dead, and B-1 was right all along.** B-1's *offline* sweep measured the knee erratic
across task (agnews ≈3.0–3.5, yahoo and yelp-p ≈2.0–2.3). **P1 reproduced it almost exactly on a bracketing
grid** — 3.14–3.36 / 2.07–2.19 / 2.35–2.45 — so the instrument is precise and repeatable; it simply does not
measure `Φ*`, and what it does measure depends on the trajectory (§5.6). The live probe's 1.25 is the grid
floor (hole 2). **Nothing forward-only supplies `B_max` in advance; the run's own accuracy curve is the
only sensor left** — hole 5, row **E**.

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
| its own sim charge profile | `fluxtune.yaml` | `fluxtune_yahoo.yaml`, in git | `fluxtune_yelp-p.yaml`, in git |
| **FluxTune-v2** run valid | **yes** — 938 commits, peak 0.843 | **yes** — 1,138 commits, peak 0.428 | **yes** — 997 commits, peak 0.728 |
| **FluxTune** ends on its own stop | **no — no stop exists** | **no — same** | **no — same** |
| **reaches the FL target** | **−0.007** (0.873) | **YES — +0.003** (0.663) | **−0.008** (0.812) |
| **beats FluxTune-v2** | 0.873 vs 0.843, **5.3×** | 0.663 vs 0.428, **6.2×** | 0.812 vs 0.728, **7.9×** |
| v2 against the FL target | −0.037 | **−0.233** | −0.092 |
| accuracy still has slope in `B`? | **no** | **no — retracted** | **no** |
| `B_max` sensed, not supplied | 1.694, 12 fires | 2.017, 16 fires | 1.847, 12 fires — **every fire is the grid floor, and sensing is now known to be impossible (§5.6). This row cannot be made to pass; the claim must rest on row E instead** |

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
| **models** | **0 runs**, but P1 has now probed roberta-large offline | every *run* on record is DistilBERT + adapters. The offline knee differs by 1.08 between the two architectures on one dataset (§5.6) | **N5b** → **N5c** |
| **PEFT capacity within that model** | `rf` 16 vs 64 | **negative** — hole 3. N5b must re-derive `T_res` against the `ln 2.7` prior, since P1 showed no measured `Φ*` is available in advance | N5a · N5b |
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
   anneal half of this hole is untouched, and **row A is now the only fix** — N4a′ is closed unrun (hole 2),
   so nothing will restore the anneal by making the sensor honest.
2. **The `B_max` probe reports its own grid floor — and re-ranging it would not help, because the quantity
   it measures is not fixed.** *(closed 2026-08-23 by P1; §5.6)* The arithmetic is as stated: `knee()`
   interpolates from an implicit `(Φ=1, normalized 1.0)` anchor to the first grid point, so a model already
   below half-accuracy at `Φ`=1.5 — all 40 fires — returns `1 + 0.5·(1−0.5)/(1−n₁)` → **1.25**. But the
   defect is deeper than the range: **`Φ_knee` is a property of the training trajectory**, moving 1.44→2.07
   on one model and one dataset when only the rig's learning rate changed. **No grid makes a trajectory-
   dependent quantity available before the trajectory.** Row **N4a′ is closed unrun.**
3. **The MODEL axis is untested and its one probe came back negative.** Every run is DistilBERT + adapters
   at `rf`=16; the three datasets differ in `p` by 1.4%. At `rf`=64 law C + `annealed` does not compose with
   the gate under **any** `T_res` (§5.3), so `T_res`=300 and `f`=0.95 are **pinned to one `p`**. Rows
   **N5a** → **N5b** → **N5c** walk §5.8's porting order; **P1** tests the same question from the other
   side (§5.6) and **answered it negatively — `Φ_knee` is not readable in advance, so N5b cannot be handed a
   measured `Φ*`.** **N5b is the `rf`=64 failure restated as a task**, and it must now be re-derived against
   the `ln 2.7` prior or against `Φ*` from a completed run, not from a probe.
4. **`Φ*` ≈ 2.9 is the most transferable number here, and it is only ever visible after a run.** Peaks land
   at 2.82 / 3.00 / 2.91 on 4-, 10- and 2-class tasks, and the ledger's `p`-ladder held its peak at `Φ` 3.04
   (`rf`=64) and 3.28 (`rf`=32) across a **3.8× range in `p`**. **P1 tested whether a forward-only probe can
   read it in advance and the answer is no** (§5.6): `Φ_knee` is not a property of the model, of the task,
   or of the pair. So "measure `Φ*` once, then spend to it" is **dead**, and the saturation stop is the only
   instrument that ever sees `Φ*`. Why `Φ*` itself stays so tight across tasks is now the open question.
5. **No forward-only probe can supply `B_max` before a run. The sensing architecture is dead.**
   *(measured 2026-08-23; §5.6)* The previous statement of this hole — injection reads a **frozen** model, a
   run **re-fits**, the gap is **~1.8× in `Φ`** — is **retracted**: the offline probe at `m`=0 *is* frozen
   and reads **3.175**, not 1.25, so frozen-vs-re-fit is not what separates the instruments. What P1 measured
   instead is that `Φ_knee` moves with **dataset** (spread 1.10), with **architecture** (agnews: 3.15
   DistilBERT vs 2.07 roberta-large), and with the **optimizer path** (1.44→2.07 at fixed model, dataset and
   accuracy) — while `rf` across a 3.8× range in `p` moves it ≤0.22. It also **anti-correlates** with `Φ*`.
   ⇒ rows **P1′** and **N4a′** are closed unrun, and **row E's saturation stop is the primary sensor**, not
   a backstop.


---

## §2 — Now · what is running

*Read 2026-08-24 01:00. **Rows C · D · Y launch tonight on nodes 1–3**; node 4 is free and its arm is a
decision, see §3.1.* All ten runs are on this node's disk and their curves are cached in
`expt_scripts/writeup_figs/data/*.json`, so nothing needs to re-scan `experiments/`.

### What is running overnight, and the hypothesis each arm tests

**One hypothesis, three datasets: `FluxTune ends itself at the right place, without being told where.`**
No run in this corpus has ever ended on `[BudgetStop] reason=saturation` under the shipped stack — the peak
is demonstrated on all three datasets and the *plateau* is not, which is the second half of §0's dataset
axis and the last thing standing between §1's claim and its evidence.

| arm | node | dataset | budget | **predicted** | **falsified if** |
|---|---|---|---|---|---|
| **C** | 1 | agnews | 48,000 vclock / 10 h | `[BudgetStop] reason=saturation` near commit **1,180**, acc **≥ 0.850**, `Φ` in **2.4–3.3** | it runs to the ceiling, or stops below `Φ`=2.4 |
| **D** | 2 | yahoo | 60,000 vclock / 14 h | saturation near commit **1,080**, acc **≥ 0.657** | same |
| **Y** | 3 | yelp-p | 50,000 vclock / 14 h | saturation near commit **1,126**, acc **≥ 0.807** | same |

**Why these three and not the roberta arms.** §3.1's M-DAY plan is **withdrawn as written**: the 08-23
roberta smoke prices N5c at **~50 h**, so a 3.5 h arm reaches `Φ`≈1.08 and a 12 h arm `Φ`≈1.29 — neither
reaches the phenomenon (**§5.11**). C · D · Y are correctly sized at 10–14 h, run on the stack stage 2
smoked clean, and each carries `P4_RETENTION_EVERY=10` so **P3′ is confirmed past `Φ`=2.9** for free.

**Three secondary readings ride these arms at no cost:** whether saturation's `thr`=0.005 / `patience`=20
survive out of sample on the three curves they were sized on; whether the `anchor` combiner's n=1 rule
(row **B4**) matters in a run that actually terminates; and the first `[CommitGate] -> COMMIT` counts under
the 2026-08-24 boundary fix.

**Stages 1 and 2 both ran 2026-08-23, and a third smoke followed.** Stage 1 (~28 min GPU) closed **P1, P1′
and N4a′**: `Φ_knee` is a property of the trajectory, not of the model or the task, and it anti-correlates
with `Φ*` (**§5.6**; consequences in **§1 holes 2/4/5**). Stage 2 (2 h, run `123727`) proved the wiring and
**answered P3′** — `cos·Φ`=0.9999 — but **did not fire the stop** (**§5.10**). The roberta-large smoke
(2 h, run `145932`) then **ported the stack to a second architecture cleanly** — `p`=4,225,540, 0 deaths,
`cos·Φ`=1.0000 at the new `p` — **and priced N5c at ~50 h**, which is what withdrew M-DAY (**§5.11**).

**One correctness fix landed 2026-08-24, enabled by default per §6 rule 1.** `_gate_satisfied` compared
`n_have >= n_req` exactly; `ρ_max` is *defined* as the `ρ` where `n_req` = `max_iter·K`, so a run pinned at
the cap is satisfiable only at equality and float rounding lost it (live `n_req` = 200.00000000000006
against `n_have` = 200.0). Roberta's smoke committed **0 times through the gate and 80 times through the
`max_iter` bypass**. Now a 1e-9 relative tolerance; regression case (g) in `expt_scripts/test_commit_gate.py`
covers both architectures at the cap. **It corrects accounting, not throughput** — the gate fires on the
same 200-upload pool it was already waiting for.

**Rows S · E · E2 · M1 · M2 · D0 landed as code on 2026-08-22 and N5a closed the same day**, each gated
offline; nothing on the queue is blocked on a node. Launch commands are §4.1 and §4.2b; the stage order is
§3. **roberta-large is ported through step 3 and has now RUN** — see §5.11 for what that cost. Note the
offline rig needs `--lr 3e-4`; at the DistilBERT default 1e-3 roberta-large collapses to one class.

**Two probe-script changes landed 2026-08-23, both default-preserving:** `grad_over` is computed only when
`--modes` includes `signal` (it OOM'd roberta-large and was unused otherwise), and `--lr` is exposed on
`probe_inflation_damage.py` (default 1e-3, so the nine DistilBERT cells are byte-identical).

| run | state |
|---|---|
| **N1** agnews `014242` · **N2** yahoo `014328` · **N3** yelp-p `014406` | **all three COMPLETED** on their vclock ceiling — 1,913 / 2,400 / 1,863 commits, 12 / 16 / 12 probe fires, `s`=1.5, `anchor`, `log_only`. **Correct, not void**: `log_only` runs have nothing to halt them (§4.2b). Peaks 0.873 / 0.663 / 0.812 |
| **N4a** re-range the probe grid | **NEVER LAUNCHED** — preflight `BLOCKED` on the shortened ceiling. Corrected overrides in §3 row N4a′; why the floors exist, §4.2b |
| **roberta-large smoke** `145932` | **COMPLETED, not scoreable** — 80 commits in 2 h, `Φ`=1.0435, acc 0.295 at chance 0.250. Ported clean (0 deaths, `p`=4,225,540, `cos·Φ`=1.0000); `n50` read only 50 of 100 shards, so it is a smoke by construction. **Prices N5c at ~50 h — §5.11** |
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
stop is an `OR`, the saturation detector is sized and gated out of sample, the two launcher guards are
armed — so the queue is now *runs*, not code.

**Run them in stages, cheapest falsifier first.** Stage 2 was meant to show the stop firing before stage 3
spent 38 h; it showed the *wiring* clean but never reached saturation (§5.10), so **C · D · Y are both the
termination evidence and its own smoke** — the first of them to fire the stop retires the risk for the
other two, and all three are cheap enough to run in parallel rather than in series.

| stage | what | tier (§4.2a) | cost | why first |
|---|---|---|---|---|
| ~~1~~ | ~~P1 + P1′~~ | T0 | **DONE 08-23, 28 min** | `Φ_knee` is trajectory-dependent (§5.6). Closed P1, P1′, N4a′ |
| ~~2~~ | ~~P3′ + smoke~~ | T1 | **DONE 08-23, 2 h** | wiring clean on all 4 gates, 0 deaths; **P3′ holds at `cos·Φ`=0.9999**. Stop did NOT fire — the 2 h ceiling never reached saturation (§5.10) |
| ~~3~~ | ~~M-DAY: roberta-large × 3 datasets, 4 nodes~~ | T1 | **WITHDRAWN 08-23** | the roberta smoke (`145932`) ported clean and priced N5c at **~50 h**; a 3.5 h arm reaches `Φ`≈1.08 and cannot answer it. **§5.11** re-sizes N5c and lists the three ways to make it affordable |
| **3′** | **C · D · Y** overnight, 3 nodes in parallel | T2 | 10–14 h each | **RUNNING.** The termination evidence — no run has ever ended on its own stop under the shipped stack, and it is the last thing between §1's claim and its evidence. Each arm carries `P4_RETENTION_EVERY=10`, so **P3′ past `Φ`=2.9** rides along free |
| **4** | **N5c re-sized**, most likely at `s`=2.9 (~13 h) | T2 | 13–50 h, 1 node | §0's other axis. Gated on **§5.11**'s `s` lever being scored — the DistilBERT `s`=1.0 arm (**N4b′**) is the cheap read on whether `s` moves along the curve or shifts it |

### §3.1 M-DAY — **WITHDRAWN 2026-08-23**, and what replaced it

> **The three roberta arms below cannot answer N5c in 3.5 h** — the smoke measured 40.1 commits/h against
> the ~2,000 commits `Φ*` needs (**§5.11**). Nodes 1–3 run **C · D · Y** instead (§2). **Node 4 is free**
> and its arm is a live decision: the DistilBERT IID control below (partition and cache exist, costs a
> config flag, closes the α→∞ end of the heterogeneity axis) or **N4b′** at `s`=1.0, which is the cheap
> read on the `s` lever that §5.11 needs before N5c can be re-sized. **Prefer N4b′** — it gates a 50 h run.
>
> *Kept below for the record: the arm table is still the right shape once N5c is affordable.*

#### The original plan (for reference)

**One run should answer many rows.** Each arm below carries the retention probe and the health gates for
free, so a single slot closes several queue rows at once.

| node | arm | rows it advances | why this one |
|---|---|---|---|
| **1** | **roberta-large · agnews** `rf`=16, seq 192 | **N5c** · N5b · F · P3′@new `p` | cleanest first cross-model run — warmest cache, shortest sequence |
| **2** | **roberta-large · yelp-p** seq 256, 2 classes | **N5c** · R2 | model × dataset jointly; 2-class is the easiest transfer |
| **3** | **roberta-large · yahoo** seq 256, 10 classes | **N5c** · R2 | the hardest task on the new model — where `T_res` should break first if it breaks |
| **4** | **DistilBERT · agnews · `uniform` (IID)** | **P2′-lite** · heterogeneity axis | the α→∞ end of the only untouched axis. Partition **and cache already exist** — costs a config flag |

**Gate it on a 15-minute smoke first (§4.2a rule 1).** roberta-large has never run under the FL stack; if
it OOMs or the config path is wrong, an ungated launch wastes three nodes. Smoke one arm, read
`[ProbeDim] p=4225540` and `trips/commit`, then launch the rest.

**What M-DAY cannot answer, and that is fine.** `ρ_max = s√(max_iter·K·G_rule/p)` falls **3.06×** at
roberta's `p` (0.1000 → 0.0326), so reaching `Φ`≈2.9 takes **~2,000 commits** against DistilBERT's ~214 at
the same ceiling — a **10 h+** run at best. **A 3.5 h roberta arm will not reach `Φ*` and will not fire the
stop.** It is not meant to: it answers *does the self-derivation compose at a new `p`* — memory, `p`
threading, gate reachability, `trips/commit` ≥ 3, law C's shape, retention. Those are exactly the things
that would void an overnight run, and they cost 3.5 h to learn instead of 14.

**Memory is the live risk and it sets `--num-trainers`.** roberta-large is 355M params; at the shipped
12.5 trainers/GPU that is **17.8 GB of weights and ~35.5 GB with the forward-mode tangent**, against a
46 GB card *before activations*. **Launch roberta arms at `--num-trainers 50`** (6.25/GPU → ~17.8 GB with
tangent) and confirm on the smoke. `run_node_p4.sh` hardcodes `--num-trainers 100` and the model, so both
need to become overrides — that is the one build task M-DAY depends on.

**The roberta port runs alongside, not after.** Steps 1–2 are already done (§5.8); pre-tokenization is
CPU-bound and competes with nothing.

> **Two generality axes.** **Task** is demonstrated on three datasets but **terminated on none** —
> C · D · Y · Score. **Model** has **zero runs**; §5.8's porting order is through step 2 for roberta-large,
> and F → N5b → N5c is the rest, gated on P1. **A row that ships a hand-fitted number is progress on
> neither.**

> **P1, P1′ and N4a′ are closed as of 2026-08-23 and must not be re-proposed.** `Φ_knee` is a property of
> the *trajectory* — it moves with dataset, with architecture, and with the optimizer path at fixed model
> and task — so no grid range, cadence or re-fit variant makes it available before the run that would use
> it (§5.6). **`B_max` by sensing is not a broken feature, it is an impossible one.** What replaces it is
> row **E** (the peak detector, now the primary sensor) and row **A**, whose anneal target is **also sensed**
> — driven from measured progress, not a rail ([P4.13](fl_fwd_ft_practice.md#p413-row-a--the-sensed-anneal-candidate-replayed-2026-08-23)).

| # | node | task | done when |
|---|---|---|---|
| **P2′** | 1 GPU, ~2 h, **promoted** | **Is `Φ*` an FL number or a training-geometry number?** *(the only surviving cheap question about `Φ*`; needs a short centralized RUN, not a probe — P1 showed probes cannot see `Φ*`)* Run the **same forward-gradient estimator centralized** (1 client, IID, same `p`, same `s`, same law C) on agnews and read `Φ` at peak. The model doc's §2.6 says heterogeneity is a step-size multiplier only, so `Φ*` should not move; the 22-run ledger already spans α 0.1–1 at 2.41–3.11 without moving. **Predicted:** peak at `Φ` = 2.8–3.0, i.e. inside the FL band. **Falsified if** centralized peaks below 2.4 or above 3.3 — then `Φ*` carries a federated component and every "property of the model" claim in §5.6 is wrong. **Do not run backprop as the comparator here** — backprop's steps are not ⟂ `θ`, so it reaches target accuracy at `Φ`≈1 and has no `Φ*` to compare (§5.6c) | a centralized forward-gradient `Φ` at peak, against 2.82 |
| **P3′** | rides any run — **now default-on** | ~~Log `cos(θ_t,θ_0)`~~ **ANSWERED 2026-08-23.** `cos·Φ` = **0.9999** over 42 commits on the stage-2 smoke, 0 outside 1.00 ± 0.02, `cos` 0.9776→0.6296 as `Φ` 1.02→1.59. Retention **is** `1/Φ`; the ≈70° drift reading is measured, not derived. **Still open:** only verified to `Φ`=1.64 — **carry `P4_RETENTION_EVERY` on every M-DAY arm** to confirm it past `Φ`=2.9 and at roberta's `p` | confirmed past `Φ`=2.9 and on a second architecture |
| **A** | 1 CPU to finish sizing, then it rides stage 3 | **Restore the anneal from SENSED progress — no hand-set target.** *(candidate drafted + replayed 2026-08-23; see [P4.13](fl_fwd_ft_practice.md#p413-row-a--the-sensed-anneal-candidate-replayed-2026-08-23))* `ρ*_t = ρ_max·√(clip(g_eff/running_max(g_eff), 0, 1))` with `g_t = (m_t − m_{t−h})/m_t` on row E's own 11-eval trailing mean, `h` = one probe cadence, and `g_eff = max(0, g − 1·σ)` where **σ is sensed from the curve's own step scatter**. `ρ_max` is *derived* (gate reachability, measured 0.0678 on all three runs), so the only constant is the dimensionless **1σ** noise floor. **This replaces the `B_max = ln 3.0` rail** — rows A and E become one rule off one signal, and `B_rem`→0 makes the *original* budget stop reachable for the first time (hole 1b needed `Φ`≈116). **Replayed on 9 cached curves:** `ρ` reaches **exactly 0** on all three saturated runs (Σρ² 0.98/1.91/1.48 against the runs' own 3.08/3.59/3.29), self-stops at commit 1062/1301/807, and correctly keeps stepping on the runs that had not saturated. **Three open problems, all stated in P4.13:** the replay is **open-loop** so its budget is a *lower bound* (simulated final `Φ` 1.63/2.59/2.10 against `Φ*` 2.82/3.00/2.91); `Σρ² < ∞` is **empirical, not proven** — the telescoping bound fails by 57–113× under noise rectification; and the rejected first draft (`B_rem = B_t·frac`) shows how easily this family hides a bootstrap defect. **Predicted:** a live run stops on its own between `Φ` 2.4 and 3.3. **Falsified if** it stops below 2.4 — then the normalisation under-spends and `g_eff/running_max` is the wrong shape | ρ falls monotonically to 0, `Σρ²` stops growing, and the run ends on its own stop inside the `Φ` band |
| **N4b′** | node 4 · agnews | **FluxTune at `s`=1.0** (`P4_GATE_S=1.0`), `anchor` + `log_only`. Untouched by N4b — its config was correct (`gate_safety_s=1.0`, `rho_max` 0.0666, `n_req` ≈2.2× the `s`=1.5 run) and the node killed it. `Λ = 2B/s` says lowering `s` moves *along* the accuracy-vs-`Λ` curve, not up it. **Predicted:** the same peak (≈0.872) at the same `Λ`≈1.4, reached at lower `B`. **Falsified if** the peak is higher | whether `s` moves along the curve or shifts it |
| **C** | any GPU node | **agnews controller**, 48,000 vclock, on the new stack (`anchor`, saturation-primary, rail at 3.0). `condition_fp` will no longer read `c2ef1528` — expected and correct; control `021843` does not run the probe, so it stays the valid partner | ends on `[BudgetStop] reason=saturation` near commit 1,180, at or above 0.850 |
| **D** | any GPU node | **yahoo controller**, 60,000 vclock, same new stack | ends on saturation near commit 1,080, at or above 0.657 |
| **Y** | any GPU node | **yelp-p controller**, 50,000 vclock, same new stack. **This row was missing and the claim needs it:** yelp-p's only self-terminating run is `125010`, which ran the *old* `mean` combiner and the *old* budget stop — and hole 1 now explains that termination as `mean` lagging a rising sequence, i.e. **arithmetic on the combiner, not a run reaching its budget**. It is not evidence for the shipped stack. Control `161751` ran the full 50,000 clean and stays the valid partner | ends on `[BudgetStop] reason=saturation` near commit 1,126, at or above 0.807 |
| **R2** | 1 GPU, ~1 h | **Is the estimator the limit on yahoo AND yelp-p?** Both saturate below their reference (−0.071, −0.062) with flat accuracy-vs-`B`, so this is no longer a yelp-p-only question. cos audit for ~100 commits + `replay_scoring.py --cos`; a `D` materially below agnews' 0.10–0.15 means the forward estimate degrades with class count or seq 256 — an FwdLLM-layer finding, not a controller one. Plus H-S (`probe_fd_chord.py`) | a `D` for each against agnews' band |
| **F** | 1 GPU, ~1 h | **`FWDLLM_FD_SCALE_INVARIANT` holds the wrong quantity fixed — decide what it costs.** N5a measured `‖θ_tr‖/√p` constant at 0.0196–0.0199 across a **35.7× range in `p` and two architectures**, which makes the flag's effect mechanical: holding the *absolute* chord `h√p` fixed (flag **ON**, what every P-4 run sets and the preflight *refuses* to launch without) sends the *dimensionless* chord as `1/√p` — **0.503 / 0.709 / 0.997 / 0.164** on distilbert rf 16/32/64 and roberta-large rf=16, **6.1×**. Flag **OFF** (`h`=0.01) holds it at **0.501–0.511, 1.02×**. So the knob that exists to enforce scale invariance is the one that breaks it (§5.3's ratio principle). **This is measured at INIT and says only that the flag misses its stated intent — not what it costs.** Run H-S (`scripts/probe_fd_chord.py`) at both flag states on one `p`, and read `D` against agnews' 0.10–0.15. **Predicted:** flag OFF gives the better `D` at `rf`=64, where the ON chord is 2× the reference. **Falsified if** `D` is flat in the chord over 0.5–1.0 — then the ratio is not what the estimator is sensitive to and the flag is merely mislabelled | a `D` per flag state, and a decision on which quantity `h` should hold fixed |
| **N5b** | any CPU + 1 short GPU run, **partly answered by `145932`** | **Re-derive `T_res` at a second `p` — step 4 of §5.8, and hole 3 stated as a task.** At `rf`=64 the `Λ ≥ 0.95` and `trips/commit ≥ 3` floors close against each other: a 9-unit window at `T_res` 82–90 where `Λ` clears by 0.001–0.007 while 28% of commits still floor to `I`=1 ([P5.2](fl_fwd_ft_practice.md#p52-execution-plan--to-a-zero-input-run) phase 4). **That derivation assumed `B_max` = the `ln 2.7` prior, and P1 has shown no probe can improve on it** (§5.6), so the re-derivation must run against that prior or against a `Φ*` read off a completed run. **Predicted:** with a measured `Φ*` the window opens to ≥50 units and law C composes. **Falsified if** it stays ≤10 or stays empty — then `T_res` cannot be re-derived from the same closed form at a new `p`, and the gate's reachability floor, not the anneal, is what does not port | a `T_res` at a second `p` holding trips/commit ≥3 in every quintile at `Λ` ≥ 0.95 — **or** a statement of which floor binds and why no `T_res` satisfies both. **The second half is answered: gate reachability binds**, and it binds by pinning the run at `ρ_max` (trips/commit = 20.00, the ceiling, in every quintile) rather than by closing a `T_res` window — §5.11 |
| **N5c** | 1 GPU, **after N5b** | **The second-model run — the largest hole in the claim, and the only row that closes it.** Every run on record is DistilBERT + adapters at `rf`=16 and the three datasets differ in `p` by 1.4%, so **nothing in this work has been tested across `p`**, while `T_res`, the `Φ` rail, the probe cadence, the saturation warm-up, `h`'s chord ratio and `Φ*` itself are all sized at that one `p` (§5.8). Run FluxTune unchanged on a second architecture, supplying only model + PEFT scheme + compute budget. **roberta-large is ported through step 3 and has now run** (§5.11): it builds with adapters, `p`=4,225,540, `‖θ_tr‖`=40.99, knee 2.065 from P1 (not usable as `B_max`, §5.6), and the `145932` smoke cleared the **memory check empirically** — 50 trainers × 358M params on 8×46 GB, 0 OOM, 0 deaths. A scored run needs 100 trainers, which is the one memory question left. Pre-tokenize first — the cache is keyed by model name, so it is a 100% miss (`pretokenize_dataset.py --model-type roberta-large`, ~0.8 GB and ~15 min per dataset). **RE-SIZED 2026-08-23 by the `145932` smoke: this is a ~50 h run at `rf`=16, `s`=1.5, not an overnight one** — `ρ_max ∝ 1/√p` makes `B` accrue `∝ 1/p`, so the run sits pinned at `ρ_max` paying 20 trips/commit for ~2,000 commits (**§5.11**, which also lists the three levers; `s`=2.9 is the only one that is both cheap and answers the same question). Everything mechanical already ported: 0 deaths, `p`=4,225,540, `cos·Φ`=1.0000. **Predicted:** the peak lands inside the `Φ` = 2.4–3.3 band and no learning knob is set by hand beyond N5b's re-derived `T_res`. **Falsified if** the peak lands outside that band — then `Φ*` is not a property of the model family either, the rail must be sensed per model, and §5.6's hypothesis fails on the axis it was proposed for | a scored run on a second architecture, and a `Φ` at peak against 2.82–3.00 |
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

### §4.2a Sizing the evidence to the question — **read this before queueing any run**

**GPU time is the scarce input. Every queued row gets the cheapest instrument that can falsify it, and a
row only escalates a tier when the tier below has already run clean.** Three tiers, and the boundary is
*what kind of evidence the question needs*, not how important the question is:

| tier | cost | the questions it, and only it, can answer | what it cannot answer |
|---|---|---|---|
| **T0 — offline probe**, no FL stack (`scripts/probe_*.py`) | 1 GPU, 10 min – 2 h, **fans out across the node's 8 GPUs** | model/training geometry: `Φ*`, the knee, chord/`D`, `‖θ_tr‖` vs `p`, anything measurable on a frozen or centrally-refit model | anything the aggregator, the gate or the scheduler does |
| **T1 — short FL run**, `VCLOCK_OVERRIDE` + `CEIL_OVERRIDE` | 1 node, 35 min – 2 h | **does the code path execute at all** — a new flag threading through, a log line appearing, a stop *firing*, a probe grid bracketing, trainers joining, gates 1–3 of §4.4 | where a run lands, whether it holds a plateau, gate 4 |
| **T2 — full run** | 1 node, 10–14 h | only these: peak accuracy, plateau retention, termination on its own stop, and any reading that needs the trajectory *past* the knee | — |

**The four rules that fall out.**

1. **No T2 until every code path it depends on has executed once at T1.** A wiring fault costs 35 min to
   find and 14 h to find the other way. This is why §3 is staged and why stage 2 exists at all.
2. **A T2 run cannot be truncated into a short window.** Shrinking a controller's budget does not shorten
   it — law C's length comes from `(B_max, T_res, f)`, so the run dies on `max_runtime_s` and is **void,
   not partial** (§4.1, §4.2b's two floors). **If the free window is shorter than the row's own ceiling,
   spend the whole window on T0 + T1 instead.** A half-run buys nothing.
3. **T0 sweeps are wall-clock-cheap because they parallelise; T1/T2 are not.** A 9-cell `(dataset × rf)`
   probe grid is *one* wall hour on 8 GPUs, not nine. An FL run takes the whole node at 100 trainers. So a
   free node-hour is worth ~8× more to a T0 row than to a T1 row — fill spare nodes with T0 first.
4. **State the tier and the falsifier in the row when you queue it.** A row that cannot name the cheapest
   tier that would falsify it is not specified yet.

**The two failure modes this is written against, both already paid for.** N4a was queued as a short run
against a preflight that prices law C's *full* length — `BLOCKED`, slot lost. N4b was queued as a
full-length run onto a node with a 43.7 GB peer and died at 0 commits — a T1 run would have lost 35 min
instead of a night.

### §4.2b Launching a run — copy these, and the two ways node 4 lost a slot

```bash
cd $REPO && git pull
export FLAME_CONDA_ENV=test_fwdllm FWDLLM_FD_SCALE_INVARIANT=1
NODES=$FW/expt_scripts/nodes

# --- a full-length run (this is what N1-N3 ran)
tmux new -s p4 "$NODES/run_node_p4.sh <agnews|yahoo|yelp-p> controller 2>&1 | tee ~/p4_N.log"
# ^ anchor + saturation + rail 3.0 are the DEFAULTS now; add P4_PHI_STOP=log_only
#   only when the point is to measure past the stop.

# --- STAGE 2: one short run that carries P3' AND smokes the new stack.
#     P4_BMAX_EVERY=50 drops the probe cadence, and BOTH saturation horizons ride
#     it (warm-up 3x, progress 1x), so the detector arms at commit 150 instead of
#     450 and a short run can exercise it. log_only keeps it observational.
#     No P4_BMAX_PHIS: N4a' is closed, the grid range is not worth a slot (5.6).
tmux new -s p4 "P4_PHI_STOP=log_only \
    P4_BMAX_EVERY=50 P4_RETENTION_EVERY=10 \
    VCLOCK_OVERRIDE=12000 CEIL_OVERRIDE=2.0 \
    $NODES/run_node_p4.sh agnews controller 2>&1 | tee ~/p4_stage2.log"

```

**Shortening a controller run has two independent floors, and 2026-08-21 hit both.**

| override | floor | why |
|---|---|---|
| `VCLOCK_OVERRIDE` | **≥ ~12,000** on agnews | the `[BmaxProbe]` cadence is **150 commits**, and agnews runs ~12,500 vclock/h at ~500 commits/h. 6,000 vclock buys ~120 commits — **no probe fires at all**, so a probe-grid run measures nothing |
| `CEIL_OVERRIDE` | **≥ 1.8 h** on agnews | the wall-clock preflight prices **law C's own 898-commit length**, not the shortened vclock: 3.42 trips/commit × (4.41 s + 0.77 s/trip) = **6,322 s**. Anything under that is `BLOCKED (exit 2)` and **nothing launches** |

That is the §4.1 rule — *shrinking a controller's budget does not shorten it* — showing up one step
earlier than expected. The refusal is cheap and correct; budget for the run law C thinks it is running.

**yahoo must run on node 2** — its sim charge profile is node-2-local and the launcher now *refuses*
elsewhere rather than silently pricing it on agnews (§4.5). **`fluxtune_yahoo.yaml` is in git as of
2026-08-22, so yahoo runs anywhere.**

**Carry `P4_RETENTION_EVERY=10` on every arm from now on** — it is byte-identical when off, costs one dot
product per commit, and P3′ still needs confirming past `Φ`=2.9 and at a second `p` (§5.10).

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
controller run**, armed or not — that was hole 1, and row **S** closed it on 2026-08-22.

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
- **A new MODEL is a 100% cache miss**, because the key leads with `model_type` and `model_name`.
  `pretokenize_dataset.py --model-type roberta-large` fills it (~0.8 GB and ~15 min per dataset on agnews);
  the flag reaches `cache_file` too, so the script's cached/to-write accounting is right rather than
  reporting distilbert's shards as present.

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
| `f` (stop at `B ≥ f·B_max`) | **0.95** — and **unreachable under `anchor`** | with `B_rem` pinned at `r`≈0.25, `B ≥ f(B+r)` needs `B ≥ f·r/(1−f)` = **4.75** (`Φ`≈116). Not a margin to re-tune: the budget stop is simply gone, which is why the saturation stop had to land before any run was armed |
| `ρ_max` | `s·√(max_iter·K·G_rule/p)` ≈ **0.0999** | gate reachability, mechanical. **Not** a `ρ* ≤ ρ*₀` clamp — that would block a re-sense from spending the budget it just found |
| `B_max` prior | `ln 2` | replaced outright by the first sense, never averaged into it |
| `B_max` origin | **`B + ln Φ_knee`** | the probe measures headroom from `θ_t`; `B` accumulates from `θ_0` |
| `B_max` combiner | **`anchor`** — the latest sense *(decision 2026-08-20, run 2026-08-21)* | `mean` assumed the fires estimate **one constant**; 40 fires say otherwise — `B_rem` is flat at 0.21–0.30 on all three, so `mean`-minus-`B` collapsed and annealed `ρ*` **1.7× below** the live measurement. **Confirmed as a fix and as a non-event:** it roughly doubled `B_max` (0.795→1.694 on agnews) and bought **+0.006 / +0.005 / −0.002** accuracy and no speed-up (§5.5). Its known defect — it does not terminate — is now the load-bearing one. `b_max_policy=anchor`, no code change |
| what the stop does | **`halt`** via `_work_done` | one line into a tested path. Three states ship: `off` · `log_only` (emit the crossing, keep training) · `halt`, and as of 2026-08-22 all three stop REASONS honour it. Until then `_check_budget_stop` was `if landing: budget-test else: Φ-test`, so on a controller run the Φ branch was unreachable and N1–N3 emitted no crossing at all |
| `Φ` rail | **3.0** — `PHI_RAIL_DEFAULT`, the code default *(2026-08-21; was 2.7, and 2.7 never actually ran)* | measured under law C: peaks land at **2.82 / 3.00 / 2.91**, so 2.7 costs 0.005 on all three and 3.63 costs 0.008–0.015. 3.0 sits on the peak band and coincides with where the sized saturation stop fires (§5.5). Replaying N1's budget, it crosses at **commit 1,152** |
| stop reasons | **`saturation` primary · `phi_fixed` as the rail · `budget` demoted** *(decided 2026-08-20, sized 2026-08-21, shipped 2026-08-22)* | The run must end because **learning** stopped, not because a cumulative total was reached. `B ≥ f·B_max` is no longer a termination rule — `B_max` stays only to drive law C's `ρ*`. The three are an **`OR`** tested in that order; `saturation` is behind `--saturation-stop` (default off in code, **on** in `run_node_p4.sh`'s controller arm) and the other two ship enabled. The detector is **GL + Prechelt's progress term**; GL alone gave up 0.153 out of sample (§5.5) |
| is `B_max` a fixed total at all? | **no, and no probe can supply one** | `B_rem` is flat at 0.21–0.30 over 40 fires, so `B_max` is `B` + a constant and "spend `B_max` then stop" has no fixed point. **P1 (2026-08-23) closed this the hard way** (§5.6): the offline knee is a property of the *trajectory*, so there is no quantity to sense. What ships instead is row **E**'s saturation stop plus, if row **A** needs one, an openly hand-set rail |

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
them as an `OR` — done 2026-08-22.
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
| ~~**Is `Φ*` a property of the MODEL or of the TASK?**~~ **ANSWERED 2026-08-23: neither** (§5.6) | P1 ran the bracketing grid at `rf` 16/32/64 × 3 datasets + roberta-large | `Φ_knee` tracks the **trajectory** and anti-correlates with `Φ*`. The per-run sensor and the combiner are not *unnecessary* — they are **unbuildable**. Open in their place: **why is `Φ*` itself so tight (2.82–3.00)?** |
| ~~**Can any forward-only probe read `Φ*`?**~~ **ANSWERED 2026-08-23: no** | P1 + P1′'s `m`=0 control (§5.6) | `Φ*` is knowable only after a run ⇒ row **E**'s saturation stop **is** the primary sensor. Stage 2 must now prove it fires |
| **Is `Φ*` the same in centralized forward-gradient training?** §2.6 says heterogeneity is a step-size multiplier only, so it should be. **Now the only cheap open question about `Φ*`** | row **P2′** — same estimator, 1 client, IID, and it needs a short *run* (P1 showed probes cannot see `Φ*`) | a centralized peak outside [2.4, 3.3] makes `Φ*` partly federated. Inside it, `Φ*`'s tightness is a training-geometry fact and the next question is why |
| ~~**Is retention really `1/Φ`?**~~ **YES — measured 2026-08-23** | row **P3′**: `cos·Φ` = **0.9999** over 42 commits, 0 outside 1.00 ± 0.02 | the ≈70° drift reading is measured. **Residual:** verified only to `Φ`=1.64 and at one `p` — carry the probe on every M-DAY arm |
| ~~**Does a bracketing grid make `B_rem` shrink with `B`?**~~ **MOOT 2026-08-23** | closed by §5.6, unrun | the sensed quantity is trajectory-dependent, so the anneal cannot be driven by `B_max`. **It can still be sensed — from progress rather than from a wall** (row **A**, [P4.13](fl_fwd_ft_practice.md#p413-row-a--the-sensed-anneal-candidate-replayed-2026-08-23)) |
| **Does `s` shift the accuracy-vs-`Λ` curve, or only move along it?** `Λ = 2B/s` says *along* | N4b′ at `s`=1.0 against N1 at matched `Λ` | a higher peak than 0.872 makes `s` a real accuracy lever — and the first thing to try on yahoo and yelp-p |
| **Is the residual gap to the *centralized ceiling* the estimator?** Not urgent — all three now sit within 0.008 of their FL targets | row **R2**: cos audit + `D` against agnews' 0.10–0.15 | if `D` matches agnews, the limit is adapter capacity, not the estimator |
| **Does the saturation stop hold on a fourth dataset?** Both horizons are multiples of the probe cadence and neither moves any fire commit (E2, closed 2026-08-22), and the rule now survives six curves it was not sized on — but **thr 0.005 and patience 20 are still read off three curves** | **C**, **D**, **Y** first: no run has ever ended on it. Then any new dataset | a threshold tuned to three tasks is a hand-set constant by another name, and it would be the only one left in the loop |
| **What does the FD chord ratio cost?** The flag holds the absolute chord fixed and the dimensionless one moves 6.1× across `p` and two models (§5.8) — measured, but only at init, and only as a mismatch with its own stated intent | row **F**: H-S at both flag states, `D` against agnews' 0.10–0.15 | if `D` is flat in the chord over 0.5–1.0 the flag is merely mislabelled; if not, every `p`-ladder and cross-model comparison ran at a different effective probe step |
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

### §5.6 What `Φ` is, and what it is a property of *(opened 2026-08-21; settled 2026-08-23)*

**Two different quantities share the letter, and conflating them is the easiest error here.**

| | what it is | how it is obtained | costs |
|---|---|---|---|
| **`Φ_t`** — the **inflation ratio** *(was "the odometer"; retired, §5.6a)* | `‖θ_t‖/‖θ_0‖`, how far the model has rotated off its starting point | **pure arithmetic on the step sizes**: `B = ½Σln(1+ρ_t²)`, `Φ = e^B`. Exact because the step is ⟂ `θ` (cross-term 1.000 ± 0.005). **No model, no data, no accuracy enters it** | zero |
| **`Φ_knee`** — what the probe reads | how much isotropic noise *this trajectory's* weights tolerate before normalized accuracy halves. **Not a property of the model or the task (P1)** | **measured**, `expts/bmax_probe.py:knee` — inject noise scaled to inflate by each grid `Φ`, read held-out accuracy back, interpolate to the 0.5 level on `(acc − chance)/(base − chance)` | ~6 forward passes |
| **`Φ*`** — where peak accuracy lands | empirical, **2.82 / 3.00 / 2.91** | read off the runs | a run |

**So `Φ_t` is dynamic but not adaptive.** It is a running total the controller keeps for free; it does not
look at the model or the task. That is the whole point — it is the one quantity in the stack that carries
no units of `‖θ‖`, `‖g‖`, `p` or the label set (§6, scoring vocabulary), which is why a fixed number can be
compared against it at all. `Φ_knee` and `Φ*` are the numbers that *might* depend on something.

**The hypothesis is FALSIFIED. `Φ_knee` is a property of the training trajectory — not of the model, not of
the task, not of the pair.** *(P1, 2026-08-23, `probe_inflation_damage.py --modes noise`, grid 1.5–4.0, all
cells bracketed unless noted. Row P1 named its own falsifier: "the three datasets disagree by more than
0.18". They disagree by 1.10.)*

| `Φ_knee` | rf=16 | rf=32 | rf=64 | within-dataset spread | base_acc |
|---|---|---|---|---|---|
| **agnews** DistilBERT | 3.146 | 3.363 | 3.139 | 0.22 | 0.90 |
| **yelp-p** DistilBERT | 2.447 | 2.348 | 2.368 | 0.10 | 0.86 |
| **yahoo** DistilBERT | 2.186 | 2.065 | 2.094 | 0.12 | 0.73 |
| **agnews roberta-large** `p`=4,225,540 | **2.065** | — | — | — | 0.86 |

**Three nested falsifications, each stronger than the last.**

1. **Not a model constant.** Between-dataset spread **1.10**; `rf` over a **3.8× range in `p`** moves it
   **≤0.22**. The prediction was the exact reverse — cluster near 2.9, move with `rf`.
2. **Not a task constant either.** Same dataset (agnews), different architecture: **3.15** DistilBERT vs
   **2.07** roberta-large.
3. **Not stable at fixed model AND task.** Changing only the rig's learning rate moved the knee
   **1.44 → 1.75 → 2.07** (lr 3e-5 / 1e-4 / 3e-4) at essentially equal accuracy (0.821 / 0.815 / 0.859).
   This also kills the `base_acc` confound — two cells 0.006 apart in accuracy read 0.3 apart in knee.
   *(The 1.435 cell sits below the 1.5 grid floor, so read it as "below 1.5"; the direction does not
   depend on it.)*

**And it anti-correlates with `Φ*`.** `Φ_knee` ranks agnews > yelp-p > yahoo; `Φ*` ranks yahoo > yelp-p >
agnews — perfectly reversed on n=3. Suggestive, not proven, but the instrument plainly carries no positive
signal about the peak.

**B-1 was right and its dismissal is retracted.** B-1 read agnews ≈3.0–3.5, yahoo and yelp-p ≈2.0–2.3; P1
reads **3.14–3.36 / 2.07–2.19 / 2.35–2.45** on a bracketing grid. It reproduces. The instrument is precise
and repeatable — it measures a real quantity that is not `Φ*` and is not available before a run.

**What follows for the design.** A quantity that depends on the trajectory cannot be measured in advance to
size that trajectory's budget, at any grid range or cadence. The rig's optimizer is not even the run's:
these are AdamW steps, the runs take forward-gradient steps. ⇒ **`B_max`-by-sensing is dead; rows P1′ and
N4a′ are closed unrun; row E's saturation stop is the primary sensor.** What survives is the empirical
tightness of `Φ*` itself — 2.82 / 3.00 / 2.91 — and **why that is tight is now the open question**, with no
cheap instrument known to answer it.

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
| **`Φ_knee` (frozen)** | shipped injection probe: inject at `Φ`, read a frozen model | ~6 evals | **live: 1.25–1.28, the grid floor. Offline on a bracketing grid: 2.07–3.36, trajectory-dependent and anti-correlated with `Φ*` (§5.6)** |
| **`Φ_knee` (re-fit)** | inject at `Φ`, run `m` steps, *then* read | `m` steps × grid | **built, and moot.** At `m`=0 it reads **3.175**, not the 1.25 the frozen/re-fit story predicted — so re-fitting is not the variable. Row **P1′ closed unrun** |

**Both cheap definitions failed, so the expensive one is all there is.** `Φ*_traj` costs a run and
therefore cannot size that same run's budget — which is why the injection probe existed at all. P1 killed
the frozen reading (trajectory-dependent, anti-correlated with `Φ*`) and P1′'s `m`=0 control killed the
re-fit story before its sweep was worth running. ⇒ **`Φ*` is knowable only after the fact, and row E's
saturation stop is the PRIMARY SENSOR, not a backstop** — a peak detector is the only instrument that ever
sees `Φ*`. This is the branch this section was written to anticipate; it is now the live case, and it makes
stage 2 (proving the stop fires at all) the highest-value work on the queue.

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

> **The *bill* is a property of the estimator, and that part is not in doubt.** *(The companion phrase
> "the wall is a property of the model" is **withdrawn** — §5.6 measured the probe's wall moving with the
> trajectory, and `Φ*` itself has never been shown to be a model property.)* Backprop banks the
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
| — | chance = `1/num_labels`, for the probe's normalization | `h` = 0.01. **Audited 2026-08-22 and the flag is backwards** — see the table below and row **F**. Nothing else in this column has been measured on a second model |
| — | `B_max` **prior** `ln 2`, replaced by the first sense | `s` = 1.5. Derived from Cauchy–Schwarz, so it should carry — but `D` = 0.050 is folded into it, and `D` is a constant of *this* setting (§6.3 of the model doc) |

**The honest summary the writeup should carry:** a new **dataset** costs a `datasets.yaml` row, a partition
build and a compute budget — **no learning knob**. A new **model** costs those plus a re-derivation of
`T_res`, the rail, the probe cadence and `h`'s scale-invariance, **because all four were sized at one `p`
and one architecture.** Generality is demonstrated across *task*; across *model* it is a design intent with
one negative data point (`rf`=64) against it.

**Ordering, if someone actually ports this. Steps 1–3 are DONE for roberta-large (2026-08-22/23), and
step 5 has now been ATTEMPTED — it ports clean and costs ~50 h, §5.11.**
`p` and `‖θ_tr‖` (free, at init) → `h`'s chord ratio (**N5a, closed**) → ~~`Φ*` by probe~~ **(deleted:
P1 showed no probe reads `Φ*`, §5.6)** → `T_res` at the new `p` against the `ln 2.7` prior (**N5b**) → only
then a run (**N5c**), whose own saturation stop is what reveals that model's `Φ*`. `expt_scripts/probe_port_init.py` does steps 1–2 for
any model in one command, no data and no run.

| at init, `rf`=16 unless noted | `p` | `‖θ_tr‖` | `‖θ_tr‖/√p` | chord, flag **ON** | chord, flag **OFF** |
|---|---|---|---|---|---|
| distilbert | 450,340 | 13.347 | 0.01989 | 0.503 | 0.503 |
| distilbert `rf`=32 | 229,012 | 9.471 | 0.01979 | 0.709 | 0.505 |
| distilbert `rf`=64 | 118,348 | 6.732 | 0.01957 | 0.997 | 0.511 |
| **roberta-large** | **4,225,540** | **40.991** | **0.01994** | **0.164** | **0.501** |

**Two results, and the second one is a live defect.** *(1)* `‖θ_tr‖ ∝ √p` to **±1%** — measured across a
35.7× range in `p` **and a second architecture**, so the ±1.3% claim below is now an architecture-independent
one. *(2)* Because of it, `FWDLLM_FD_SCALE_INVARIANT=1` — which every P-4 run sets and the preflight refuses
to launch without — sends the **dimensionless** chord as `1/√p`, 6.1× across those four rows, while the flag
**off** holds it constant to 1.02%. **The knob that exists to enforce scale invariance is the one that breaks
it.** Row **F** decides what that costs; until it lands, nothing about the flag changes, because this is
measured at *init* (`‖θ_tr‖` grows 13.35 → 61.9 over an agnews run) and says only that the flag misses its
stated intent, not what the estimator loses.

> ⚠ **N5a predicted 0.50 / 0.70 / 0.98 on the ladder and measured 0.503 / 0.709 / 0.997.** The worry was
> right, and roberta-large extends it to 0.164 — a 6.1× spread rather than a doubling. Row **F**.

### §5.9 Injected inflation ≠ earned inflation *(measured 2026-08-21; mechanism RETRACTED 2026-08-23 — see §5.6)*

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

**Two readings survive, and the mechanism this section originally proposed does not.**

1. **`Φ_knee`'s live pinning is arithmetic.** **35 of 40 fires** are already below the 0.5 level at the
   *first* grid point, so `knee()` interpolates off its synthetic `(Φ=1, 1.0)` anchor and returns
   `1 + 0.5·0.5/(1−n₁)` → 1.25. That much stands.
2. **The trajectory really is more robust than the injection at matched `Φ`.** The table above is a real
   measurement and is not retracted: at `Φ`=2.0 the run is within 3% of peak while the live probe reads
   chance. What it does *not* license is a fixed conversion factor between the two.

> **⚠ RETRACTED 2026-08-23 — "the mechanism is re-fitting, ~1.8× in `Φ`".** P1′ run at **`m`=0** is a frozen
> read, the same protocol as the live sensor, and it returns **3.175 on agnews**, not 1.25. So frozen-vs-
> re-fit is **not** what separates the two instruments, and the gap is **not a 1.8× offset**: the offline
> knee ranges 2.07–3.36 and moves with dataset, architecture and optimizer path (§5.6). The live 1.25 is the
> grid floor, not a measurement of anything. **Do not restate the 1.8× figure anywhere.**

> **What the shipped probe is still good for:** ranking one trajectory's tolerance to unearned perturbation,
> forward-only and cheap. **What it must never be used for:** setting `B_max`.

**Closed by this, do not re-open:** "the probe is biased conservative by a fixed offset in `Φ`" — in any
version, 0.6–1.2 or 1.8×. There is no conversion factor, because the offline knee is not a constant of the
model or the task (§5.6).

---

### §5.10 The stage-2 smoke, and the sizing trap it exposed *(2026-08-23, run `123727`)*

426 commits in 2 h, `Φ`=1.636, `B`=0.492/1.045, acc 0.859 **and still rising**. All four health gates hold,
8 probe fires, **0 trainer deaths**, every data bin visited. Numbers: [P4.14](fl_fwd_ft_practice.md).

**Won:** every new code path executed — the saturation detector armed, the `OR` stop threaded,
`P4_RETENTION_EVERY` emitted, and **row P3′ answered**: `cos(θ_t,θ_0)·Φ` = 0.9999 over 42 commits.

**Lost:** the stop did not fire, which is what the run was launched to prove.

> **⚠ THE TRAP, and it is new: §4.2b's overrides are PREFLIGHT floors, not EXPERIMENT floors.**
> `CEIL_OVERRIDE ≥ 1.8 h` is what the launcher needs to *accept* an agnews controller. It is **not** what
> the run needs to reach the thing being measured. agnews peaks near commit 1,050 (`Φ`=2.82) and at the
> measured 363 commits/h that is ~3 h to peak plus patience-20 before the detector can fire. **A 2 h
> ceiling could never have fired it**, and the detector staying silent on a still-rising curve was
> *correct behaviour*, not a defect.
>
> **The rule: size a T1 run from the phenomenon, not from the preflight minimum.** Before launching, state
> the commit at which the thing you are measuring is expected to happen, divide by the measured commits/h,
> and add the detector's own patience. If that exceeds the window, the run cannot answer the question and
> you should either lengthen it or launch something that fits.


### §5.11 What a second model costs in WALL TIME — and why N5c cannot be an overnight run *(2026-08-23, run `145932`)*

**§5.8 itemises what a new model costs the operator in re-derivation. This is the bill in node-hours, and
it is the number that reorders the queue.** Numbers: [P4.15](fl_fwd_ft_practice.md#p415-the-roberta-large-smoke--ports-clean-and-prices-n5c-at-50-h-2026-08-23-run_20260823_145932).

**The port itself is clean.** roberta-large runs under the FL stack at `p`=4,225,540 with 0 trainer deaths,
full bin coverage, a correctly-refusing `B_max` probe, and **P3′ holding at `cos·Φ` = 1.0000 on a second
architecture**. Nothing in the self-derivation is wrong. What is wrong is the *sizing*, and it compounds:

| | DistilBERT `p`=450k | roberta-large `p`=4.23M | ratio |
|---|---|---|---|
| `ρ_max = s√(max_iter·K·G_rule/p)` | 0.1000 | **0.0326** | ÷3.06 (`∝ 1/√p`) |
| `B` banked per commit, `½ln(1+ρ²)` | 5.0e-3 | **5.33e-4** | ÷9.4 (`∝ 1/p`) |
| commits to `Φ`=2.9 | ~214 | **~2,000** | ×9.4 |
| trips/commit while pinned at `ρ_max` | 5.05 (annealed off it) | **20.00** (never left it) | ×4.0 |
| measured commits/h | 341–363 | **40.1** | ÷8.5 |
| **wall to `Φ*`** | **~3 h** | **~50 h** | **×17** |

**The compounding is the finding.** `ρ_max ∝ 1/√p` is not merely a smaller step — it means `B` accrues
`∝ 1/p`, so a bigger model spends proportionally longer *below* its own anneal threshold. At `B/B_max` = 6%
after 2 h, law C never engaged; `ρ` sat at the cap, and at the cap `n_req` = `max_iter·K` = 200 by
construction, so **every commit costs the maximum 20 round trips**. Small step, maximum price per step.

> **⚠ THE RULE, and it is §5.10's trap generalised from ceiling to model.** §5.10 said *size a T1 run from
> the phenomenon, not the preflight minimum*. This adds: **the phenomenon's own commit count is a function
> of `p`, so re-derive it before every cross-model launch.** `commits_to_Φ* ≈ ln(Φ*)/(½ln(1+ρ_max²))` and
> `hours = commits / (measured commits/h)`. Both terms move against you at larger `p`. **A 3.5 h roberta arm
> reaches `Φ`≈1.08 and a 12 h overnight reaches `Φ`≈1.29** — the §3.1 M-DAY plan as written could not have
> answered N5c, and is withdrawn in that form.

**What this does NOT say.** It is not evidence that the self-derivation fails at a new `p` — the run never
got far enough to test that, and everything that *was* testable ported clean. N5c is unanswered, not
refuted. **It says N5c is a ~50 h single-node run at `rf`=16, and must be planned as one or made cheaper.**

**The three ways to make it cheaper, none of them free, none yet measured:**

| lever | effect on the 50 h | what it costs |
|---|---|---|
| **raise `s`** (1.5 → 2.9) | `ρ_max ∝ s`, so `B`/commit `∝ s²` — **÷3.7, to ~13 h** | `Λ = 2B/s` says a higher `s` moves *along* the accuracy-vs-`Λ` curve; G-1b already scored `s` as efficiency, not safety. **The cheapest lever and the one to try first** |
| **raise `max_iter`** (20 → 60) | `ρ_max ∝ √max_iter` — ÷3, but trips/commit rises the same 3× at the cap | **net zero in wall time.** Buys a larger step per commit at exactly proportional cost. Not a lever |
| **drop `rf`** (16 → 4) | `p` ÷~4, so ÷4 in commits | changes the deployment being tested, and hole 3 says PEFT capacity does not compose. **Answers a different question** |

**Row N5c is re-sized on this table; row N5b's question is answered in passing** — at roberta's `p` the
floor that binds is **gate reachability**, exactly as the `rf`=64 failure predicted, and it binds not by
closing a `T_res` window but by pinning the run at `ρ_max` for its whole affordable length.

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
8. **Every queued row names its evidence tier — T0 offline probe · T1 short run · T2 full run — and no
   T2 launches until its code paths have executed once at T1.** GPU time is the scarce input and a
   truncated T2 is void, not partial. The tiers, and the two slots already lost to ignoring them: **§4.2a**.

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
