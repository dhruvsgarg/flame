# Build plan — **the status doc**: the claim, where it stands, and what to launch next

> **One file for status and next steps.** §1 is the claim and its scoreboard · §2 is what is running now ·
> §3 is the queue · §4 is how to run and read an arm · §5 is what must not be re-derived · §6 is the rules
> any change inherits.
>
> **The other two docs are evidence, not status.** [fl_fwd_ft_practice.md](fl_fwd_ft_practice.md) owns
> *what is true*: the P3 knob ledger, the P4 arm ledger, P6's dead ends, P8's reproduction recipes. Every
> number cited here lives there. [fl_fwd_ft_solution.md](fl_fwd_ft_solution.md) owns *why* — the model,
> cited as "model §x". **Read [P6](fl_fwd_ft_practice.md#p6--dead-ends--do-not-retry) before proposing any
> change.**
>
> *Renumbered 2026-08-20. Code comments citing the old scheme map as: §-1/§-0 → §1-§3 · §0 → §6 ·
> §1/§2/§3/§4/§5 → §5 · §6 → §1 · §8 → §6 · §9 → §5 · §10 → §4.6 · §11 → §4.*

**How to update this doc — IN PLACE, never append.** One fact, one home: change *that* line, never add a
second statement of it. Replace, do not accumulate — a measurement carries its value and its date, the old
value is deleted. No changelog, no session log, no dated append sections; chronology lives in git and P4.
**When a queue row lands, delete it** — what survives is one sentence in §5 if the fact still binds, and
its number in P3/P4. **§1–§3 stay under ~120 lines** — two screens; if an edit
pushes past that, something in it has stopped being status.

---

## §1 — The claim, and what is missing

> **FluxTune reaches and holds a plateau on a new dataset with no learning knob tuned by hand — same
> DistilBERT + adapters, three datasets, against a version of itself whose step size was hand-searched.**

**Three systems; use these names everywhere, figures included.** **FwdLLM** — prior work, variance gate,
raw SGD. **FluxTune-v2** — trust-ratio + `n_target`, but a **static `ρ*`=0.06 hand-searched on agnews**,
RM-decayed (`rm`/`setpoint`; the code already calls it `fluxtune_v2`). **FluxTune** — this work, `ρ*` from
law C on a **sensed** `B_max`. **backprop reference** — exact gradients, same rig, 10 clients × 3 epochs.

**Why sensing is required:** B-1's *offline* sweep measured `B_max` erratic across task (agnews knee
≈3.0–3.5, yahoo and yelp-p ≈2.0–2.3, non-monotone in class count), so it cannot ship as a constant. **The
live probe has not reproduced that** — hole 2.

**What a new dataset costs the operator. Only the first column is input, and none of it is a learning knob:**

| supplied by hand | mechanically derived | universal constant |
|---|---|---|
| a `configs/datasets.yaml` row (h5 paths, `num_labels`, `max_seq_length`, split sizes) — a *description of the data* | `dataset` / `data_file_path` / `partition_file_path` / `max_seq_length` into both override blocks (`--dataset`) | `probe_combine=mean` · `server_step_rule=trust_ratio` · `commit_gate=n_target` · `gate_rho_ref=annealed` |
| a partition build (`build_niid_partitions.py`) + `check_partitions.py` | `num_labels` from the h5 label vocab | `s`=1.5 · `T_res`=300 · `b_max_policy=anchor` |
| **a compute budget** (`max_runtime_s`, `sim_wall_ceiling_s`) | `p` from `[ProbeDim]`; `total_data_bins` from the registry | `P`=10 · `K`/`C`=10/30 |
| `eval_max_samples` — a **cost** knob, not a learning one | `ρ*_t` from law C · `ρ_max` from gate reachability · `n_req` closed-form | `B_max` **prior** `ln 2`, replaced by the first sense |
| a sim charge profile (**sim-only artifact**, needs a real run) | `B_max` itself — **sensed** by the 3.1 probe | — |

### Scoreboard

| what the claim needs | agnews | yahoo | yelp-p |
|---|---|---|---|
| backprop reference | **0.850** | **0.734** | **0.874** |
| its own sim charge profile | `fluxtune.yaml` | built, **node-2 local only** — row **D0** | `fluxtune_yelp-p.yaml`, in git |
| **FluxTune-v2** arm valid | **yes** — 938 commits, peak 0.843 | **yes** — 1,138 commits, peak 0.428 | **yes** — 997 commits, peak 0.728 |
| **FluxTune** ends on its own stop | **void** (watchdog) | **void ×2** (watchdog) | **yes** — `[BudgetStop]`, commit 1,348 |
| **reaches the reference** | **YES — 0.868 > 0.850** | no — 0.657 of 0.734 | no — 0.814 of 0.874 |
| **beats FluxTune-v2** | 0.868 vs 0.843, **5.5×** | 0.657 vs 0.428, **6.5×** | 0.814 vs 0.728, **8.5×** |
| accuracy still has slope in `B`? | **no** — 0.06, saturated | **YES — 0.23** | **no** — 0.03, saturated |
| `B_max` sensed, not supplied | 0.795, 5 fires | 0.805, 6 fires | 1.023, 8 fires — **but hole 2** |

**×** is the vclock at which FluxTune passes v2's *own full-budget peak*; **v2 never reaches FluxTune's
accuracy on any dataset**, and both void arms were still gaining when killed, so those rows are floors
([P4.11](fl_fwd_ft_practice.md#p411-the-2026-08-20-p-4-pairs--the-law-wins-on-all-three-one-pair-is-valid)).

### The finding that reorders everything — §5.5

**FluxTune throttles its own step**: the `mean` combiner turns a flat headroom measurement into a
shrinking one and anneals `ρ*` **1.7× below** what the probe currently supports (§5.3). But `Λ = 2B/s`
means un-throttling buys **commits, not learning** — it raises accuracy only where accuracy still had
slope in `B`. It does on **yahoo alone**:

| | tail `dAcc/dB` | what the throttle cost | what more budget buys |
|---|---|---|---|
| **agnews** | 0.06 | time only | nothing — already past its reference |
| **yahoo** | **0.23** | **accuracy** | `ΔB`≈0.33 → `Φ`≈2.8 → **≈0.734, its reference** |
| **yelp-p** | 0.03 | time only | nothing — closing 0.060 would take `Φ`≈17 |

> **yelp-p's 0.060 gap is not a stopping problem, not a budget problem, and not an `s` problem.** Its
> accuracy-vs-`B` curve is flat, and `Λ = 2B/s` only rescales that axis. It is an estimator-quality or
> adapter-capacity limit — row **R2** is the diagnostic. *This retracts the earlier reading that yelp-p
> "stopped early while still climbing"; the +0.0041 tail is real but two orders of magnitude short.*

**Four axes of generality, and only one is exercised:**

| axis | coverage | status |
|---|---|---|
| **datasets** | 3 of 3 run, 3 valid v2 arms | agnews clears its reference; yahoo projected to; yelp-p saturated below |
| **models** | **0** | every arm on record is DistilBERT + adapters. No second model ever tried |
| **PEFT capacity within that model** | `rf` 16 vs 64 | **negative** — hole 3 |
| **heterogeneity** | α = 1 only | ablations go **up** to α = 10/100, never below 1. Not started |

### The holes

1. **Termination is being rebuilt.** `B ≥ f·B_max` is demoted; **saturation** becomes primary and `Φ`
   the rail (§5.3). The detector is **not built** — row **E** — and must be sized on N1–N3's curves.
2. **The `B_max` probe's Φ grid does not bracket the knee, so it mostly senses `B` itself.** Sized off
   B-1's *offline* knees; **all 19 live fires read below the knee level at the very first point**, so
   `knee()` extrapolates from its `(Φ=1, 1.0)` anchor to the lone Φ=1.5 reading and 2.5–4.0 do no work.
   `B_rem` is pinned into ≈0.21–0.42, `B_max = B + B_rem` **recedes as `B` is spent**, and the three
   `B_max` order by **fire count (5/6/8), not by dataset**. **Row N4a.**
3. **The MODEL axis is untested and its one probe came back negative.** Every arm is DistilBERT + adapters
   at `rf`=16; the three datasets differ in `p` by 1.4%. At `rf`=64 law C + `annealed` does not compose with
   the gate under **any** `T_res` (§5.3), so `T_res`=300 and `f`=0.95 are **pinned to one `p`**. Blocker on
   ship item 5b.
4. **The `Φ`=2.7 rail is probably too tight.** P4 says arms hold their peak to `Φ`=3.63 and only lose it
   past 4.23 (§5.5) — and yahoo's projected landing is `Φ`≈2.8, in that gap. 2.7 comes from the *old
   diverging* dynamics and has never been tested under a controlled `ρ`. **Rows N1–N3.**

---

## §2 — Now · what is running

*Read 2026-08-21 00:30. **Nothing is running** — all four nodes are free and N1–N4 are ready to launch
(§4.7).* All six 2026-08-20 arms are on node 3's disk; their extracted curves are cached in
`expt_scripts/writeup_figs/data/*.json`, so nothing needs to re-scan `experiments/`.

| arm | state |
|---|---|
| yelp-p **FluxTune** `125010` | **VALID** — `[BudgetStop] reason=budget action=halt`, commit 1,348, stopped itself at 28,885 of 50,000 vclock, peak 0.8141, ends 0.0006 below it. **Saturated** (§5.5) |
| **v2** arms `161751` / `151619` / `021843` | **all VALID** — full budgets, peaks 0.7280 / 0.4275 / 0.8432. `151619`'s `arm_stall.json` is **false** (§4.3) |
| agnews **FluxTune** `152215` | **VOID** — killed by the **pre-fix** watcher at 87.9% of `B_max`, 899 commits, peak 0.8676 (**past its reference**) |
| yahoo **FluxTune** `125003` | **VOID** — same kill, 964 commits / 86.0% of `B_max`, peak 0.6571, **still climbing hard** |

**Both void kills replay clean under the fix** (ΔB 0.0701 and 0.0628 against a 0.005 floor); a true death
has never been replayed — row **W′**. **The watchdog is fixed and pushed**: the `I`-floor kill now needs
`ΔB ≤ 0.005`, and the watcher exits when the arm prints its own terminal line (§4.3).

**[fl_fwd_ft_writeup.md](fl_fwd_ft_writeup.md)** is the prose account with seven rendered figures — the
doc to hand to anyone outside this work. Regenerate with `writeup_figs/make_figures.py` (§4.8).

---

## §3 — Next · the ordered queue

| # | node | task | done when |
|---|---|---|---|
| **N1** | node 1 · agnews | **FluxTune, `anchor` + `log_only`, 48,000 vclock.** Nothing halts, so it runs straight through the `Φ`=2.7 rail. **Predicted:** holds ≈0.868 past 2.7 and does **not** turn before `Φ`=3.6 (P4's cliff). **Falsified if** it turns below `Φ`=3.0 — the cliff would have moved under law C | an accuracy curve through `Φ`=2.7, and where it turns |
| **N2** | node 2 · yahoo | **FluxTune, `anchor` + `log_only`, 60,000 vclock — the decisive arm.** yahoo is the only dataset with slope left: tail `dAcc/dB`=0.23, so it needs `ΔB`≈0.33 from its 0.690, landing at `Φ`≈2.8. **Predicted: reaches ≈0.734, its backprop reference.** **Falsified if** it plateaus below 0.70. Must run here — the yahoo profile is node-2-local (row **D0**) | reaches the reference, or a measured plateau below it |
| **N3** | node 3 · yelp-p | **FluxTune, `anchor` + `log_only`, 50,000 vclock.** **Predicted: no accuracy gain.** Its curve is flat (tail `dAcc/dB`=0.03), so budget past `Φ`=2.7 buys nothing and only `B` climbs. **Falsified if** accuracy rises above 0.83 — which would overturn the saturation reading and make the gap a budget problem after all | whether a saturated curve gains anything from a raised rail |
| **N4a** | node 4 · agnews, ~40 min | **Re-range the `B_max` probe grid** — `P4_BMAX_PHIS="1.05,1.1,1.2,1.3,1.5,2.0"`, short arm to ~200 commits so one probe fires. **Predicted:** the knee lands **below 1.5** and the honest `B_rem` comes out **smaller** than the ~0.25 today's grid reports — a correct sensor *tightens* the budget, it does not loosen it. Nothing on disk answers this: no arm checkpoints a model | a knee bracketed by real grid points |
| **N4b** | node 4 · agnews, after N4a | **FluxTune at `s`=1.0** (`P4_GATE_S=1.0`), `anchor` + `log_only`. Tests the claim the design rests on: `Λ = 2B/s` says lowering `s` raises progress per unit budget but **does not move the accuracy-vs-`Λ` curve**. **Predicted:** the same plateau (≈0.865) reached at lower `B`. **Falsified if** the plateau is higher — `s` would then be a real accuracy lever and the first thing to try on yelp-p | whether `s` moves along the curve or shifts it |
| **E** | any CPU, **while N1–N4 run** | **Build the saturation stop — it is now the primary termination rule.** Prechelt generalization-loss / patience on *smoothed* held-out accuracy (11-eval trailing window, as `Φ` already is — a raw `dAcc/dΛ` slope false-triggers on a dip and misses a masked plateau). Track `Acc_best` as a running max; `GL_t = (Acc_best − Acc_t)/Acc_best`; stop when `GL_t` holds above a threshold for a patience window. Ships as `stop = saturation OR Φ-cross`. Resample onto `Λ`, never `comm_round`. **Size it on N1–N3's curves** — the first arms that will have run past their own plateau | window, threshold and patience sized **by replay**, reproducing a stop on each of N1–N3 at its own plateau, and **not** firing early on yahoo, which climbs longest. `replay_phi_stop.py` is the model to copy |
| **C** | any GPU node, **after** N1–N4 + E | **agnews controller**, 48,000 vclock, on the new stack (`anchor`, saturation-primary). `condition_fp` will no longer read `c2ef1528` — that is expected and correct; control `021843` does not run the probe, so it stays the valid partner. `152215` reached 87.9% of `B_max` in 1.85 h | ends on `[BudgetStop] reason=saturation`, at or above 0.850 |
| **D0** | node 2 | **Commit `sim_charge_profiles/fluxtune_yahoo.yaml` to git**, the way `fluxtune_yelp-p.yaml` already is. It exists only on node 2's local disk, so row **D** cannot run anywhere else — and `run_node_p4.sh` now *refuses* rather than silently pricing yahoo on agnews (§4.5) | the file is in git and its md5 matches node 2's |
| **D** | any GPU node, after **D0** + N2 + E | **yahoo controller**, 60,000 vclock, same new stack. The dataset furthest from its reference (0.657 of 0.734) and the one climbing fastest when killed, so it has the most to gain from `anchor` | ends on saturation, at or above 0.734 |
| **W′** | node holding `003648` | **The last unverified half of the watchdog fix.** The `I`-floor kill needs `ΔB ≤ --b-advance-min` (0.005) across the window, and three healthy arms now replay silent (`125010` 0.0339, `152215` 0.0701, `125003` 0.0628). **Unverified: that it still fires on a true death.** `003648`'s run dir is node-local and is not on node 3 | replay `003648`, confirm it fires |
| **Score** | any CPU | **The two remaining pairs, once C and D land.** yelp-p is scored ([P4.11](fl_fwd_ft_practice.md#p411-the-2026-08-20-p-4-pairs--the-law-wins-on-all-three-one-pair-is-valid)); repeat it — peak, whether it clears the backprop reference, and the vclock at which FluxTune passes v2's full-budget peak. **Drop the 0.015-of-peak bar as a headline** — §1 shows a still-climbing arm passes it trivially | a scored table for all three datasets |
| **B4** | any CPU | ~~**`budget_stop_frac` needs a margin**~~ **MOOT** once the budget stop is not a termination rule. The underlying fact survives and belongs to `anchor`: the **first** sense replaces the `ln 2` prior at n=1, at maximum variance, and was **46% low** on yelp-p (0.5223 against a later 1.19). Under `anchor` that first sense sets `ρ*` alone with no averaging to cushion it, so **arm the probe's influence only from n ≥ 2**, keeping the `ln 2` prior for the first 150 commits | a stated rule for n=1, replayed against `021735` and the three 2026-08-20 controllers |
| **R2** | 1 GPU, ~1 h | **Is the estimator itself weaker on yahoo?** cos audit for ~100 commits + `replay_scoring.py --cos`; a `D` materially below agnews' 0.10–0.15 means the forward estimate degrades with 10 classes / seq 256 — an FwdLLM-layer finding, not a controller one. Plus H-S on yahoo (`probe_fd_chord.py`) | a `D` for yahoo against agnews' band |
| **G** | any CPU, **after** the P-4 arms | **`read_instance_from_h5` returns rows in thread-completion order**, so a shard's row order — and its bin composition — is not reproducible across tokenizations, and `guid` names the wrong row. `X`/`y` stay paired under one lock and nothing reads `guid`, so **no ledger number is wrong**. It waits because it re-orders every future shard against the caches the P-4 arms run on | two tokenizations of one client agree byte-for-byte, and `guid` round-trips |

**Nothing above is blocked by an open hypothesis.** H-S, H-H, H-T and H-J are specs + discriminating
numbers in [P5.3](fl_fwd_ft_practice.md#p53-open-hypotheses); all are rung 1–2 and none needs a node. K-C
is closed. Two ideas are **standing-refused** and must not be re-proposed — probe selection by `|d|`, and
block-coordinate probing ([P6](fl_fwd_ft_practice.md#p6--dead-ends--do-not-retry) has both, with the
replacements worth trying if selection is ever revived).

---

## §4 — How to run an arm, and how to read it

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
ceilings, then free. **A pair normally stays on one node** so both arms are priced by one profile; putting
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
| `P4_BMAX_POLICY` | `mean` | `anchor` uses the **latest** sense instead of the mean (§5.3). **The new arms want `anchor`** |
| `P4_PHI_STOP` | `halt` | `log_only` makes **both** stops emit and keep training — the only way to see past the `Φ`=2.7 rail |
| `P4_GATE_S` | `1.5` | moves `s`, the only lever `Λ = 2B/s` allows on progress per unit budget. Floor ≈0.9 at `K`=10 |
| `P4_BMAX_PHIS` | *(module default `1.5,2,2.5,3,3.5,4`)* | re-ranges the probe grid. **`condition_fp` does not cover it** — same blind spot the sim profile had |

`P4_ALLOW_AGNEWS_PRICING=1` overrides the missing-profile refusal (§4.5). `VCLOCK_OVERRIDE` /
`CEIL_OVERRIDE` shorten an arm (§4.2).

**Why the two arms cost so differently.** The controller stops *itself* at `B ≥ f·B_max` and law C's length
comes from `(B_max, T_res, f)`, not from the budget. The control has no stop, so it runs its budget out.
**Shrinking a controller's budget does not shorten it — it voids it**, on `max_runtime_s`.

### §4.2 Short arms and smokes

```bash
REAL_BUDGET=1200 $FW/expt_scripts/nodes/run_node.sh 2               # real-mode SECONDS
VCLOCK_OVERRIDE=6000 CEIL_OVERRIDE=2.0 \
  $FW/expt_scripts/nodes/run_node_p4.sh agnews controller           # vclock SECONDS, wall HOURS
SMOKE=1 $FW/expt_scripts/nodes/run_node.sh $N                       # ~20-30 min, whole chain
```

Prefer the two overrides over `SMOKE=1` for a sanity arm: `SMOKE` also swaps the watch config and routes
profiles to `smoke/`, which price nothing by design. **A short controller arm ends on `max_runtime_s`, not
`[BudgetStop]` — expected, and the one gate a short arm cannot check.** It also cannot fire a `[BmaxProbe]`
(cadence 150 commits). Everything else reads exactly as it will on the long run.

### §4.2b The four overnight launches — copy these

All four need only `git pull`; **no code change is pending for any of them.**

```bash
cd $REPO && git pull
export FLAME_CONDA_ENV=test_fwdllm FWDLLM_FD_SCALE_INVARIANT=1
NODES=$FW/expt_scripts/nodes

# --- node 1 (agnews) / node 2 (yahoo) / node 3 (yelp-p): one line each, per node
tmux new -s p4 "P4_BMAX_POLICY=anchor P4_PHI_STOP=log_only \
  $NODES/run_node_p4.sh <agnews|yahoo|yelp-p> controller 2>&1 | tee ~/p4_N.log"

# --- node 4: N4a (~40 min) then N4b, sequentially
tmux new -s p4 "P4_BMAX_POLICY=anchor P4_PHI_STOP=log_only \
    P4_BMAX_PHIS='1.05,1.1,1.2,1.3,1.5,2.0' VCLOCK_OVERRIDE=6000 CEIL_OVERRIDE=1.5 \
    $NODES/run_node_p4.sh agnews controller 2>&1 | tee ~/p4_N4a.log ; \
  P4_BMAX_POLICY=anchor P4_PHI_STOP=log_only P4_GATE_S=1.0 \
    $NODES/run_node_p4.sh agnews controller 2>&1 | tee ~/p4_N4b.log"
```

**yahoo must run on node 2** — its sim charge profile is node-2-local and the launcher now *refuses*
elsewhere rather than silently pricing it on agnews (§4.5). Fix that permanently with row **D0**.

**Prefix every launch with `NODE_DRY_RUN=1` once** — seconds, no GPU, and it prints the generated config so
`b_max_policy`, `phi_stop`, `gate_safety_s` and `b_max_probe_phis` can be read back before the slot is spent.

**Expect `log_only` arms to run their full vclock ceiling** — nothing halts them, by design. That is the
point: they measure what lies *past* the stop. A `log_only` arm ending on `max_runtime_s` is **correct**,
not void; the §4.4 gate-4 rule applies only to an arm whose stop is armed.

### §4.3 The watchdog

`_node_lib.sh` execs `watch_arm.py` as a side-car **per arm** and **kills the run** on:

| predicate | default | why this and not accuracy |
|---|---|---|
| no new commit | 20 min steady-state, 45 min pre-first-commit | a genuine hang; the only unambiguous one |
| `I` floored at 1 over the last 200 commits **and `B` not advancing** (`--b-advance-min`, 0.005) | ≥ 90% | G-2's `003648` died at 98% — but the `I` share **alone** is not that death: the n_target gate drives `I` to 1 at the landing point by design, and it voided a healthy yahoo controller at 86% of `B_max`. `B` is the exact progress measure and rides on the same record |
| trips/commit < 3 **and** pool demand unmet > 50% | after 200 commits | same argument: `trips/commit` is `n_req/K`, and law C drives `n_req` down **by design** |
| any `rho_star == 0` | after 200 commits | a requirement of *zero*, not an absent one |

**The watcher stops watching once the arm reaches its OWN end** — `stopping run.` or `[BudgetStop]
action=halt` in the last 256 KB of the aggregator log. `--pgid` clears only when the whole launcher group
exits, which lags the aggregator by the teardown: yahoo `151619` finished cleanly at 19:02 and its watcher
fired the hang guard at 19:22, writing a false `arm_stall.json` and killing the teardown. That is the same
failure that cost the two 03:04 real-mode arms their profiles. **A stall file on an arm that also has
`plots/` is that false positive, not a death.**

**Both rate predicates need their conjunct, and this is the lesson.** Fitting the 2026-08-20 agnews
controller's last 150 commits gives `n_req ≈ 89.4 − 88.4·B_frac`, i.e. `n_req` ≈ 5 at its own 0.95 stop —
a bare floor voids every controller arm at any setting above ~0.5. That arm was killed at `n_req`=18 with
demand met on **all 577** commits; the yahoo arm at 964 commits / 86% of `B_max`; and yelp-p's valid arm
ran at `I==1` on 100% while still gaining 0.034 of `B_max` per 200 commits. **Starving means the gate is
not being met. Asking for less is the controller working. Kill on `B` not advancing, never on the shape of
a healthy landing.**

**Commits are counted from `version_bump_census`, not `server_update`** — the latter exists only under
`--server-update-audit`, which the scored arms set and the real profiling arm deliberately does not, so
reading it alone saw `commits=0` on two healthy 61- and 63-commit arms and killed both. The scan is
incremental (per-file byte offsets); re-reading each poll is O(run²) and a 14 h arm would re-read ~1.6 TB.

**Deliberately NOT `converge_watch.py`** — it arms on held-out accuracy and needs `--target-acc`, which
would end the arm on convergence, and an arm that does not end on `[BudgetStop]` is void. Its signal is
backwards here: **holding a plateau is what the controller is supposed to do.**

`NODE_WATCH=0` disables it; `NODE_WATCH_ARGS="--max-hours <CEIL+1> --i-floor-frac 1.01"` drops the `I`
predicate alone (`1.01` is unreachable; setting `NODE_WATCH_ARGS` replaces `run_node_p4.sh`'s `--max-hours`
default, so pass both). **An arm already running holds the old module in memory** — a patch does not reach
it; relaunch, or `kill $(pgrep -f watch_arm.py)`, which drops the hang guard too. A killed arm leaves
`arm_stall.json` in its run dir and the node prints it.

> **Open (row B4's neighbour):** a commit pooled from ~5 of 100 trainers at the landing point is what the
> gate says is correct for a tiny step, but it is also where the server path has little trainer work
> amortising it — the sim-fidelity worry the old floor was reaching for. **Decide it on the arms, not in
> the watchdog.**

### §4.4 Reading an arm, and the scoring rules

```bash
RUN=$(ls -dt $FW/experiments/run_* | head -1)
$PY $FW/expt_scripts/check_arm_health.py $RUN --expect-controller   # exit 1 = a gate is breached
$PY $FW/expt_scripts/replay_scoring.py $RUN                         # B, Lambda, Phi, A
```

Four gates, each of which cost a node and none of which is visible at launch — **a clean `--dry-run` is a
prior, not a guarantee.** Run this by hand at ~200 commits on a live arm; `_node_lib.sh` runs it after
every arm.

```
[DataBins] from the registry, 100% coverage, trainer-confirmed   # 150 hardcoded gave yahoo 8.6% of its data
no server_update with rho_star == 0                              # a step of length zero
trips/commit >= 3 per quintile                                   # FAILs on a healthy landing -- read G-2 signature
controller ends on [BudgetStop] reason=budget, not max_runtime_s  # voided all four 2026-08-16 arms
```

`_node_lib.sh` also echoes the enactment lines after every arm — `[ProbeDim]` `[FD] spacing`
`[probe_combine]` `[TrainableScope]` `[ServerStep]` `[CommitGate]` `[CosProbe]` `[DataBins]` `[Landing]`
`[BmaxProbe]`. They are cheap and they are the only way to catch a knob that did not take.

**Gate 3 is not decisive on its own, and FAILs on every healthy controller.** The `G-2 signature` line
beneath it (`I==1` share, pool demand unmet, `n_req`) is what separates a controller annealing on plan
from a starving gate — yelp-p's *valid* arm read Q5=1.34 with `I==1` on 100% of its last 200 commits and
the pool demand met on every one. Read both. On an arm
without `--server-update-audit`, gate 2 reads `UNREADABLE`, not `ok`. The `[BmaxProbe]` trajectory and its
**first firing commit** are printed too — on yahoo that index is itself a result.

**Scoring rules** (derivation: [P4.4](fl_fwd_ft_practice.md#p44-scoring-rules-for-any-ab)):

- Score **peak** accuracy and the stability columns, **never final** accuracy of a diverging arm — ±0.045
  between byte-identical replicates past the turn, against ±0.0009 at peak.
- Compare across datasets on **`A` and per-vclock-hour** — never on `Λ` (different `p`), never per round
  (11.7× different bins/round).
- Read `B` as a fraction of `B_max` and `A` against P4's calibration **while the arm is alive**. Both are
  exact at any horizon, so both failure modes are diagnosable ~20 commits in.

### §4.5 Sizing a budget, and the profile

**Size off measured rate, never off `expts/wall_clock_preflight.py`** — it prices every commit at a
dataset-independent 4.41 s, and pre-fix yahoo measured 45.4 s. `check_arm_health.py` prints a
`budget sizing` line converting any short arm's rate into the vclock 898 commits will cost.

| arm | commits/h | vclock/h | 898 commits need |
|---|---|---|---|
| agnews `130614` | 351 | 17,523 | 44,839 vclock · 2.6 h |
| yahoo `125713` (no `--eval-max-samples`) | 79 | 5,470 | 61,932 vclock · 11.3 h |
| yahoo smoke `162439` (`--eval-max-samples 10000`) | 282 | 10,088 | 32,082 vclock · 3.2 h |
| yelp-p smoke `162510` (same) | 296 | 10,556 | 32,044 vclock · 3.0 h |

**yahoo's old 4.4× was the eval tax, not seq 256.** Eval is still ~30% of arm wall — a cadence choice now,
not a defect.

**A per-dataset profile is mandatory before a sim arm is scoreable:** yahoo burns 0.658 real-s per
vclock-s against agnews' 0.255, so no cross-dataset per-vclock comparison is valid until each is profiled.
It needs a REAL-mode run — `profile_sim_charges.py` pools `vclock_charge` events with `time_mode == "real"`
and finds nothing in a sim run — and both flags must **match the scored arms**:

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
non-agnews dataset lacking its own profile — which also disables `matches dataset`, so the arm ran priced
on agnews' **0.255** real-s per vclock-s against yahoo's **0.658**. `condition_fp` does not cover the
profile: a yahoo arm launched on node 3 with no profile reads the same `7174b984` as node 2's correctly
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
vclock. And re-profiling agnews is wrong: `fluxtune.yaml` is what every historical agnews arm and P4's own
calibration were priced against.

### §4.6 Adding a dataset

**In order:** a `configs/datasets.yaml` row → `build_niid_partitions.py` → `check_partitions.py` →
**`pretokenize_dataset.py`** → a real-mode run → `profile_sim_charges.py` → the arm.

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
  arm's own budget.

**Backprop ceiling first (§5.1).** `probe_backprop_ceiling.py --config <an aggregator_config.json>
--dataset NAME --clients 10 --epochs 3`, and **`tee` it** — the script only prints, and the 2026-08-16 run
was lost to a closed terminal. ≈0.70 clears the data path; ≈0.30 indicts it and that dataset's arms measure
nothing.

### §4.7 Reading the new arms — what to look for first

For every `log_only` arm, in this order:

```bash
RUN=$(ls -dt $FW/experiments/run_* | head -1)
grep -ao "\[BmaxProbe\][^|]*" $RUN/*aggregator.log | tail -20   # sensed trajectory + the curve
grep -ao "\[BudgetStop\][^|]*" $RUN/*aggregator.log             # crossings, logged not halted
$PY $FW/expt_scripts/check_arm_health.py $RUN                    # gate 4 will WARN -- expected here
```

Then rebuild the accuracy-vs-`B` picture, which is what the predictions are stated against:

```bash
cd $FW/expt_scripts/writeup_figs
# add the new run to ARMS in extract.py, then:
python3 extract.py --force && python3 make_figures.py 7
```

**The question each arm answers is in its §3 row, with a falsifier.** Read the falsifier first — an arm
that fails its prediction is worth more than one that confirms it, and the failure modes are all
informative (the cliff moved · saturation was mis-read · `s` is a real lever · the knee is above 1.5).

### §4.8 The figure pipeline

`expt_scripts/writeup_figs/` renders every figure in the writeup. It exists so numbers are never copied:

| file | what it does |
|---|---|
| `ledger.py` | **parses P4's arm ledger out of `fl_fwd_ft_practice.md`** — the ledger stays the one source of truth, and a figure can never drift from it |
| `extract.py` | one pass over the ~12 GB of aggregator telemetry into `data/*.json` (accuracy-vs-vclock, accuracy-vs-`B`, per-commit `ρ`/`B_frac`, every `[BmaxProbe]` fire + its curve). `--force` re-scans |
| `figstyle.py` | palette + rcParams. Three-slot categorical, validated all-pairs; colour means **dataset** |
| `make_figures.py` | `./make_figures.py` for all seven, `./make_figures.py 4 7` for individual ones |

**To add an arm:** put it in `extract.py`'s `ARMS` dict, `./extract.py --force`, re-render. Nothing else.

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
| **backprop ceiling** (exact-gradient reference) | **0.850** | **0.734** | **0.874** |
| niid groups on disk | α = 0.1…100, `C`=100 | α = 1, 100 at `C` = 100 and 1000 | α = 1, 100 at `C` = 100 and 1000 |

`configs/datasets.yaml` + `expts/dataset_registry.py` hold every one of these. Partitions pass
`check_partitions.py` 6/6 on all three. Data plumbing is exact: `bins × 8 × C == n_train` at `C` = 100 and
1,000, verified in `test_dataset_launcher.py` and logged in-run as `[DataBins] coverage`.

**Three facts that change how a non-agnews arm is read:**

- **Bins/round differ 11.7×.** Anything expressed *per round* — `max_data_id_progress`, a `data_id` sweep,
  an epoch — is not comparable across datasets. Score per **commit** or per **vclock-hour**.
- **`max_seq_length` 256 is a cost choice**, covering ~p89 of yahoo and ~p82 of yelp-p; trainer wall is
  ~linear in it, so a yahoo pass costs ~1.33× an agnews one **before** the 11.7× in bins.
- **`Λ` does not transfer across `p`**, and `p` differs by dataset. **Use `A`.**

**No attention mask anywhere in the stack** — `tc_transformer_trainer_distribute.py:713` and `:950` both do
`x = batch[1]; self.model(x)`, dropping `batch[2]`, and the backprop probe does the same, so the probe is
*faithful to production*. The model attends to PAD tokens on every arm. **Rung 1 came back 0.734, so the
mask is not the fault** — it depresses both sides of every comparison equally. A candidate for absolute
accuracy, nothing more.

### §5.2 The yahoo gap — it is budget, not plumbing

**Settled 2026-08-20: it was budget, and yahoo is the one dataset where budget still binds** (§5.5). Controller `125003` reached **0.657 at `Λ`=0.994**, and its
control read 0.428 over a full 60,000 vclock. The paragraphs below are the derivation, kept because
suspects (a)–(c) still bound yahoo's *absolute* accuracy against its 0.734 ceiling.

B-1's backprop reference reaches **0.73** on yahoo; the P-4 arms reached **0.30**. Data plumbing and the
data path both cleared (bins exact; centralized AdamW on the FL rig's own path returns 0.7333 / 0.7263 /
0.7339 over three epochs, flat from epoch 1, against untrained 0.1018). **So the gap is optimization
budget**, and the P-4 arms are the test: accuracy against `Λ`, not commits. Pre-registered both ways —
yahoo reaching ~0.6–0.7 by `Λ` ≈ 1.0 means the agnews `Λ`-curve transfers; yahoo plateauing near 0.35 with
`Λ` > 1.0 means **`Λ` does not transfer across task**, which is a more interesting result than the arm.

**What is left to suspect, in order** — the obvious ones are cleared (`p` confirmed from `[ProbeDim]`;
`max_seq_length` and `num_labels` plumbed both sides; `learning_rate` inert under `trust_ratio`; `G_rule`,
`s`, `P`, `probe_combine` dataset-free): **(a)** the `ρ*` band was sized on agnews and a 10-class head may
need a larger relative step to leave its init — the sensed `B_max` is supposed to discover this;
**(b)** `train_batch_size` = 8 means each JVP is estimated on a batch missing most of the 10 classes;
**(c)** seq 256 truncates ~11% of yahoo documents.

### §5.3 The controller's settled constants

| | value | why |
|---|---|---|
| anneal law | **C** — `ρ*_t = min(ρ_max, √(2·(B_max_t − B_t)/T_res))`, `T_res` a rate **never decremented** | A makes `ρ*` constant under perfect tracking and smuggles `T` back as an input; B is a receding horizon that never terminates. C approaches `B_max` monotonically **from below**, so the stop is a genuine backstop |
| `T_res` | **300** | 500 refuted on replay: 2.46 trips/commit on yahoo, 2.26 on the `ln 2` prior, against the ≥3 gate |
| `f` (stop at `B ≥ f·B_max`) | **0.95** | yahoo needs ≥0.90 to clear `Λ`≥0.95; agnews clears at 0.70. **Re-derive — sized against pre-fix `B_max` semantics** |
| `ρ_max` | `s·√(max_iter·K·G_rule/p)` ≈ **0.0999** | gate reachability, mechanical. **Not** a `ρ* ≤ ρ*₀` clamp — that would block a re-sense from spending the budget it just found |
| `B_max` prior | `ln 2` | replaced outright by the first sense, never averaged into it |
| `B_max` origin | **`B + ln Φ_knee`** | the probe measures headroom from `θ_t`; `B` accumulates from `θ_0` |
| `B_max` combiner | **`anchor`** — the latest sense *(decision 2026-08-20; was `mean`)* | `mean` assumed the fires estimate **one constant**. 19 fires say otherwise: measured `B_rem` shows **no downward trend** on any dataset while `mean`-minus-`B` collapses — yelp-p 0.246 measured against 0.084 used at fire 8, a **2.9×** understatement that anneals `ρ*` **1.7× below** what the current measurement supports. `anchor`'s known defect — it does not terminate — is void once **saturation** terminates instead. `b_max_policy=anchor`, no code change |
| what the stop does | **`halt`** via `_work_done` | one line into a tested path. Three states ship: `off` · `log_only` (emit the crossing, keep training) · `halt`. **`log_only` applies to BOTH stops**, which is what makes the past-the-rail arms possible |
| `Φ` rail | **2.7**, and **probably too tight** | it is the collapse backstop, and it comes from the *old diverging* dynamics. P4 puts the cliff at **3.63 / 4.23** (§5.5) — a gap of ~0.3 in `B` that the current design never spends, and yahoo's projected landing sits inside it. Never tested under a controlled `ρ` |
| stop reasons | **`saturation` primary · `phi_fixed` as the rail · `budget` demoted** *(decision 2026-08-20)* | The run must end because **learning** stopped, not because a cumulative total was reached. `B ≥ f·B_max` is no longer a termination rule — `B_max` stays only to drive law C's `ρ*`. Rationale: yelp-p halted 0.060 below its reference while still climbing, and the cause was the combiner above, not the road running out |
| is `B_max` a fixed total at all? | **open — the sensor says no** | The probe measures headroom *from `θ_t`* on a model that **cannot re-fit**; a run re-fits continuously. A flat `B_rem` across a run means budget behaves like a **rate limit that is re-earned**, not a tank that drains. If that holds, "spend `B_max` then stop" is the wrong shape and only the `Φ` rail is load-bearing. Rows **N1–N3** and **N4a** |

**`Λ = 2B/s` is an identity wherever the gate holds `s`** (−0.3% out of sample on both `s`-pinned arms,
+21.5–23.3% where `s` drifts). **So the `ρ` schedule is `Λ`-neutral at fixed `B`** — law A and law C bank
the same `Λ` and differ only in commits spent. Any "this schedule learns more" claim is a comparison at
unequal `B`.

**Real-wall cost model** (±1% over 5 arms): `wall = 7.81·commits + 0.77·trips` with the audit on at stride
25. Audit-off is `4.41 s/commit + 0.77 s/trip` — a subtraction, not a measurement.

**Still-binding cautions.** (a) `ρ`=0 is a requirement of *zero*, not an absent one. (b) Under `β > 0` the
`Φ` law changes — refuse to launch. (c) `Φ` from `ρ` is exact; never re-derive it from `‖θ‖` ratios.
(d) `Φ`-stop and budget-exhausted are the **same** trigger: `Φ = e^B`, `B` monotone, `B_max = ln Φ_peak`.
(e) The 3.1 probe must reuse the cos probe's fixed-seed reference batch **and its class-skew guard** — with
the audit off on P-4 arms that guard is otherwise not running at all. (f) **Do not wire `n_eff` to
anything** — it is an identity, 1.00 ± 0.01 over 17 arms. (g) `dynamic_kc`'s `k_max`=15 is backwards and
must not be reused as a starting point.

**Landed and closed:** hill-climb `C`, not `K` (commit throughput is flat in `K` at fixed `C`); `P` is
compute-bound (`τ(30)/τ(10)`=2.56), so adaptive `P` is no longer motivated as a throughput lever — a
**mid-run `P` change**, which no code path supports, is the only engineering left there.

### §5.4 What the arms still have to answer

| open | closes on | if it comes out wrong |
|---|---|---|
| **Does the `Φ` rail move under law C?** 2.7 is from the diverging regime; P4's cliff is 3.63/4.23 | N1–N3 run past 2.7 with nothing halting | if an arm turns below `Φ`=3.0 the cliff moved and 2.7 is right after all |
| **Does `s` shift the accuracy-vs-`Λ` curve, or only move along it?** `Λ = 2B/s` says *along* | N4b at `s`=1.0 against the `s`=1.5 arms at matched `Λ` | a higher plateau makes `s` a real accuracy lever — and the first thing to try on yelp-p |
| **Where is the knee really?** | N4a, on a grid that brackets it from below | a knee *above* 1.5 would mean the live probe was right and hole 2 is wrong |
| **trips/commit ≥ 3** is calibrated on one death and two survivals | every arm reports it per quintile; re-size once there are ten | a config passes preflight and still burns wall |
| **Is yelp-p's 0.060 gap the estimator?** | row **R2**: cos audit + `D` for yelp-p against agnews' 0.10–0.15 | if `D` matches agnews, the limit is adapter capacity, not the estimator |

**Closed 2026-08-20/21, do not re-open:** `Λ`→accuracy transfers across task (yahoo 0.657 at `Λ`=0.994) ·
`B_max` **does** drift within a run, upward, on all three · the combiner is **`anchor`**, not `mean` ·
`f`=0.95 is **moot** as a termination rule (§5.3) and survives only as an input to `ρ*`.

### §5.5 Saturation — what more budget actually buys *(landed 2026-08-21)*

**Method, so it is reproducible.** Accuracy is joined to `B` by walking the aggregator jsonl **in emission
order**, incrementing the commit count on each `server_update` and stamping every `agg_eval` with the
running `B = ½Σln(1+ρ²)` — exact, where a timestamp join would be approximate. Slope is OLS over the **last
20% of `B`**, the most recent and most conservative window (`writeup_figs/extract.py:acc_vs_budget`).

| | accuracy at end | reference | tail `dAcc/dB` | `ΔB` to the reference | lands at |
|---|---|---|---|---|---|
| **agnews** | 0.868 | 0.850 | 0.06 | — **already past** | — |
| **yahoo** | 0.657 | 0.734 | **0.23** | **0.33** | **`Φ`≈2.8** |
| **yelp-p** | 0.814 | 0.874 | 0.03 | 1.86 | `Φ`≈17 — out of reach |

**Three consequences, and they are the reason the queue looks the way it does.**

1. **Un-throttling `ρ*` buys commits, not learning** — `Λ = 2B/s` is schedule-neutral. It raises accuracy
   only where accuracy still had slope in `B`.
2. **yahoo is the only dataset where the combiner bug cost a result.** Everywhere else it cost hours.
3. **yelp-p is saturated**, so its gap is not addressable by the controller at all — not by budget, not by
   the rail, and not by `s` (which only rescales the `Λ` axis of a flat curve). Row **R2**.

**The `Φ` cliff, from P4's ledger, 22 arms that learned (peak ≥ 0.80):** every arm at **`Φ` ≤ 3.63 held its
peak** (worst loss 0.014); every arm at **`Φ` ≥ 4.23 lost it** (0.083–0.604, two ending at half their peak).
Arms that never learned are excluded — `Φ` governs *losing* what you learned, not failing to learn.

**Do not confuse the two `Φ` numbers.** Peak accuracy *occurs* at `Φ`=2.41–3.11; accuracy is *lost* past
`Φ`≈3.6–4.2. The shipped 2.7 rail sits below both.

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
   cost eight arms by being "just logging".
4. **Probes do not modify `trainer/`, `aggregator/`, or any yaml on the critical path.** They import
   production code; a validated result transfers as a config flag, not a rewrite.
5. **Every new number needs predicted-vs-observed and a run id** before it reaches a ledger, and a task is
   done when its gate reproduces a number already in P3/P4 — not when the code runs.
6. **Dataset constants come from `expts/dataset_registry.py`.** Never re-hardcode `num_labels`, `p`, a
   class-balance threshold, or an h5 path.

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
6. Extrapolating a progress rate as an accuracy rate — `A` accumulates *through* the turn while accuracy
   falls. Extrapolate `A` only alongside `Φ`.
7. Two quantities with different origins, subtracted. State the origin of every accumulated quantity next
   to its formula.
8. A flag whose writer is not its only writer.
9. **A dataset constant that is right on agnews by arithmetic coincidence.** `total_data_bins = 150` lived
   in `lib/python/flame/mode/horizontal/syncfl/fwdllm_aggregator.py`, outside the example tree, and is agnews' `1,200/8` exactly — yahoo trained on
   **8.6% of its data, the same 1,200 rows every lap**, and nothing raised. **Grep the *derived* agnews
   numbers (150 / 1,200 / 7,600 / 192), not just the name, and grep `lib/python/flame/` too.**
