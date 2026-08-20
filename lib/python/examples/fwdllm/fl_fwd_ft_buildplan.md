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
its number in P3/P4. **§1–§3 stay under ~115 lines**; if an edit
pushes past that, something in it has stopped being status.

---

## §1 — The claim, and what is missing

> **FluxTune's law-C controller reaches and holds a plateau on a new dataset with no learning knob tuned
> by hand — same DistilBERT + adapters, three datasets, controller vs control at equal vclock.**

**Why it is plausible:** every constant on the operating path is universal machinery, mechanically derived
from the data, or **sensed online**. B-1 is why the third category exists — `B_max` measured *erratic
across task* (agnews knee ≈3.0–3.5, yahoo and yelp-p both ≈2.0–2.3, non-monotone in class count), so it
cannot be shipped as a constant.

**What a new dataset costs the operator. Only the first column is input, and none of it is a learning knob:**

| supplied by hand | mechanically derived | universal constant |
|---|---|---|
| a `configs/datasets.yaml` row (h5 paths, `num_labels`, `max_seq_length`, split sizes) — a *description of the data* | `dataset` / `data_file_path` / `partition_file_path` / `max_seq_length` into both override blocks (`--dataset`) | `probe_combine=mean` · `server_step_rule=trust_ratio` · `commit_gate=n_target` · `gate_rho_ref=annealed` |
| a partition build (`build_niid_partitions.py`) + `check_partitions.py` | `num_labels` from the h5 label vocab | `s`=1.5 · `T_res`=300 · `f`=0.95 |
| **a compute budget** (`max_runtime_s`, `sim_wall_ceiling_s`) | `p` from `[ProbeDim]`; `total_data_bins` from the registry | `P`=10 · `K`/`C`=10/30 · `b_max_policy=mean` |
| `eval_max_samples` — a **cost** knob, not a learning one | `ρ*_t` from law C · `ρ_max` from gate reachability · `n_req` closed-form | `B_max` **prior** `ln 2`, replaced outright by the first sense |
| a sim charge profile (**sim-only artifact**, needs a real run) | `B_max` itself — **sensed** by the 3.1 probe | — |

### Scoreboard

A pair counts only if **both** arms are valid at the **same** `condition_fp` and the controller ended on
`[BudgetStop] reason=budget` (§4.4).

| what the claim needs | agnews | yahoo | yelp-p |
|---|---|---|---|
| backprop ceiling clears ≈0.70 | **0.850** | **0.734** | **0.874** |
| its own sim charge profile | `fluxtune.yaml` | built (node 2, local disk) | built (node 3, local disk) |
| **control** arm valid | **yes** — 938 commits | live, ends ~22:15 | starting (row **H**) |
| **controller** ends on `[BudgetStop]` | live (row **C**) | **void ×2** — row **D** | **yes** — commit 1,348, 95.0% of `B_max` |
| ends within 0.015 of peak | — | — | **score it** (§4.4), first arm that can be |
| `B_max`/`ρ*` **diverge across datasets, unsupplied** | sensed **down** 0.693 → 0.510 | — | sensed **up** 0.693 → **1.023** over 8 probes |
| no learning knob supplied | **yes**, by construction | **yes** | **yes** |

**The effect is not in doubt — its acceptance is.** The 2026-08-16 arms are void as an acceptance test,
but the law beat its control on both datasets while **23–48% of its commits took a step of length zero**,
so that margin is a floor: agnews `125619` peaked 0.857 against the control's 0.835 and reached 0.83 at
**62%** of the vclock budget against 92%, and 2026-08-20's controller reached **80.7% of `B_max` on 29%**
of its vclock against the control's 15.5% on 100%
([P4.7](fl_fwd_ft_practice.md#p47-p-4--the-law-beats-the-control-the-implementation-had-four-defects)).
What is missing is an arm that ends on its own stop with every gate clean.

**Four axes of generality, and only one of them is being exercised right now:**

| axis | coverage | status |
|---|---|---|
| **datasets** | 3 of 3 launched | 1 valid controller (yelp-p); agnews live, yahoo re-running |
| **models** | **0** | every arm on record is DistilBERT + adapters. No second model has ever been tried |
| **PEFT capacity within that model** | `rf` 16 vs 64 | **negative** — see hole 3 |
| **heterogeneity** | α = 1 only | ablations go **up** to α = 10/100, never below 1. Not started |

**The divergence row is the one already paying.** agnews and yelp-p sensed `B_max` in **opposite
directions** with nobody supplying either — which §5.4 pre-registered as possibly failing to reproduce.

### The four holes, all live

1. **The dataset axis is one arm short of closed.** yelp-p is the only valid controller; agnews is live and
   yahoo needs re-running. Nothing structural is in the way.
2. **The sensor does not fire below chance.** On yahoo `234931`'s first probe declined at commit 25 (`base_acc`=0.105 against
   chance 0.100), so the controller ran the `ln 2` prior. The machinery is universal; the *sensing* half is
   unexercised on the dataset that most needs it ([P4.8](fl_fwd_ft_practice.md#p48-yahoo-is-under-trained-not-broken)).
3. **The MODEL axis is untested, and its one probe came back negative.** Every arm on record is DistilBERT
   + adapters at `rf`=16, `p`=450,340. The only portability evidence is *within* that model: at `rf`=64
   (`p`=118,348) law C + `annealed` does not compose with the gate under **any** `T_res` — the `Λ`≥0.95 and
   trips/commit≥3 floors leave a 9-unit window (`T_res` 82–90), and the two-phase trajectory reads 1.71
   trips/commit on agnews against `rf`=16's 5.01. So `T_res`=300 and `f`=0.95 are **pinned to one `p`**,
   and "same controller, new model" has no evidence behind it. Standing blocker on ship-checklist item 5b.
4. **`f`=0.95 was sized against the pre-fix `B_max` semantics**, and rests on a `Λ`≥0.95 floor read off the
   agnews curve. `Λ` has never been tested across task. Re-derive both (§5.4's open table).

---

## §2 — Now · what is running

*Read 2026-08-20 16:20.*

| node | arm | state |
|---|---|---|
| 1 | agnews **controller** | live 15:22, `condition_fp c2ef1528`. **Old watcher armed — restart it, see row C** |
| 2 | yahoo **control** | live 15:16, ends ~22:15. `setpoint`, trips/commit ~8, **immune** to the W bug. `condition_fp 7174b984` |
| 2 | yahoo **controller** | **void** — killed 15:15 by the `I` floor at 964 commits / 86.0% of `B_max`. Never budget-starved (~20,600 of 60,000 vclock). Row **D** |
| 3 | yelp-p **controller** | **VALID** — `[BudgetStop] reason=budget action=halt`, commit 1,348, 95.0% of `B_max`, every bin visited, zero null steps, 8 `BmaxProbe` fires |
| 3 | yelp-p **control** | auto-starts behind it on the **fixed** watcher (`node_run` execs a new `watch_arm.py` per arm) |
| 4 | — | **idle** — take row **D** there rather than queue it behind node 2 |

**Gate 3 FAILs on every healthy controller and that is correct.** yelp-p's valid arm read Q5=1.34 with
`I==1` on **100%** of its last 200 commits and pool demand met on every one. Read the `G-2 signature` line
beneath it, never gate 3 alone (§4.4).

**The pairs end up split across nodes** (agnews 1/4, yahoo 4/2, yelp-p 3) and `/home` is node-local —
gather the run dirs before scoring.

---

## §3 — Next · the ordered queue

| # | node | task | done when |
|---|---|---|---|
| **C** | 1 | **agnews controller.** Restart on the fixed watcher: the running arm loaded `watch_arm.py` before the W fix and its endgame is where the bug bites. `condition_fp` must read `c2ef1528`; the 2026-08-20 control is valid on disk, do not re-run it | ends on `[BudgetStop] reason=budget` |
| **D** | 4 | **yahoo controller re-run**, 60,000 vclock, on the idle node. **Copy node 2's `sim_charge_profiles/fluxtune_yahoo.yaml` across first** and check the md5 — one profile on both arms is the only reason a pair normally stays on one node | ends on `[BudgetStop]`; `condition_fp 7174b984` matches node 2's control |
| **H** | 3 | **yelp-p control**, 50,000 vclock — auto-started behind the valid controller | same gates |
| **W′** | node holding `003648` | **Close out the watchdog fix.** The `I`-floor kill now needs `ΔB ≤ --b-advance-min` (0.005) across the window; replayed silent on yelp-p `125010` (`I==1` 100%, `ΔB`=0.0339, which the old rule would have killed) and on the agnews smoke. **Unverified: that it still fires on a true death.** `003648`'s run dir is node-local and was not on jayne | replay `003648`, confirm it fires |
| **Score** | any CPU | **All three pairs, once D and C land.** `replay_scoring.py` per arm; peak accuracy and the 0.015-of-peak bar; accuracy-vs-`Λ` on yahoo, which closes §5.4's first open row and is §1 hole 2's answer | a scored table for all three datasets |
| **B4** | any CPU | **`budget_stop_frac` needs a margin against a moving `B_max`.** The exposure is the **first** sense: it replaces the `ln 2` prior outright at n=1, at maximum variance. yelp-p 2026-08-20 confirms it — n=1 sensed 0.5223, then 0.7440 / 0.8961 / 0.9425 / 0.9598, i.e. the first sense was **46% low**; agnews' only sense went the other way to 0.5102. Had `B` been past 0.95·(first sense) at commit 150 the arm would have stopped on the spot. Candidate: do not arm the stop until n ≥ 2 | a rule with a stated margin, replayed against `021735` and 2026-08-16's arms. `ratchet` is **not** it — it is `min`, which stops sooner |
| **E** | any CPU | **3.5's saturation stop — not built.** Prechelt generalization-loss / patience on *smoothed* held-out accuracy (11-eval trailing window, as `Φ` already is — a raw `dAcc/dΛ` slope false-triggers on a dip and misses a masked plateau). Track `Acc_best` as a running max; `GL_t = (Acc_best − Acc_t)/Acc_best`; stop when `GL_t` holds above a threshold for a patience window. **Not a duplicate of the `Φ` stop** — that one catches noise-driven collapse (0.874→0.296 on `003601`); this catches the model having extracted the signal the task allows. Combine as `stop = Φ-cross OR saturation`. Resample onto `Λ`, never `comm_round` | window, threshold and patience all sized **by replay** against the arms on disk, the way task 0.5 sized `Φ`'s window. `replay_phi_stop.py` is the model to copy |
| **R2** | 1 GPU, ~1 h | **Is the estimator itself weaker on yahoo?** cos audit for ~100 commits + `replay_scoring.py --cos`; a `D` materially below agnews' 0.10–0.15 means the forward estimate degrades with 10 classes / seq 256 — an FwdLLM-layer finding, not a controller one. Plus H-S on yahoo (`probe_fd_chord.py`) | a `D` for yahoo against agnews' band |
| **G** | any CPU, **after** the P-4 arms | **`read_instance_from_h5` returns rows in thread-completion order**, so a shard's row order — and its bin composition — is not reproducible across tokenizations, and `guid` names the wrong row. `X`/`y` stay paired under one lock and nothing reads `guid`, so **no ledger number is wrong**. It waits because it re-orders every future shard against the caches the P-4 arms run on | two tokenizations of one client agree byte-for-byte, and `guid` round-trips |

**Two standing conclusions, so they are not re-proposed.** *If probe selection is ever revived*, select
on something other than `|d|` (P6 closed that): curvature `vᵀHv` (≈free), split-half SNR within the bin
(free, orthogonal to `|d|`), or loss decrease at the step scale. *Block-coordinate probing is predicted
inert on paper* — progress/commit falls as `1/√L` while budget/commit falls as `1/L`, so progress per unit
`B` is unchanged and it needs `L`× the commits; it escapes `√(n/p)` only if the gradient is *unevenly*
spread.

**Open hypotheses do not block any of this.** H-S, H-H, H-T and H-J are specs + discriminating numbers in
[P5.3](fl_fwd_ft_practice.md#p53-open-hypotheses); all are rung 1–2 and none needs a node. K-C is closed.

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
`--eval-max-samples 10000` on both seq-256 datasets. **Controller** = law C at `T_res`=300 with **no
`--rho-star` and no `--b-max`** — that is what makes it zero-input. **Control** = `rm`/0.25 at `ρ*`=0.06
with `gate_rho_ref=setpoint` and `--phi-stop log_only`, deliberately, so P4.1's past-the-stop
counterfactual keeps being measured instead of being destroyed by the controller shipping.

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

### §4.3 The watchdog

`_node_lib.sh` execs `watch_arm.py` as a side-car **per arm** and **kills the run** on:

| predicate | default | why this and not accuracy |
|---|---|---|
| no new commit | 20 min steady-state, 45 min pre-first-commit | a genuine hang; the only unambiguous one |
| `I` floored at 1 over the last 200 commits **and `B` not advancing** (`--b-advance-min`, 0.005) | ≥ 90% | G-2's `003648` died at 98% — but the `I` share **alone** is not that death: the n_target gate drives `I` to 1 at the landing point by design, and it voided a healthy yahoo controller at 86% of `B_max`. `B` is the exact progress measure and rides on the same record |
| trips/commit < 3 **and** pool demand unmet > 50% | after 200 commits | same argument: `trips/commit` is `n_req/K`, and law C drives `n_req` down **by design** |
| any `rho_star == 0` | after 200 commits | a requirement of *zero*, not an absent one |

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

**Gate 3 is not decisive on its own.** The `G-2 signature` line beneath it (`I==1` share, pool demand unmet,
`n_req`) is what separates a controller annealing on plan from a starving gate. Read both. On an arm
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
| `B_max` combiner | **`mean`** (`b_max_policy`) | the fires are noisy estimates of **one constant** — P4.1's fixed-`Φ` stop working across 10 arms with different schedules is that evidence, and the anchored quantity behaves like one (`B+R` CV ≈ 25% agnews, ±6% yahoo, against 2.4× spread in raw `R`). It also terminates, which `anchor` does not |
| what the stop does | **`halt`** via `_work_done` | one line into a tested path. Three states ship: `off` · `log_only` (emit the crossing, keep training) · `halt` |
| stop reasons | `budget` · `phi_fixed` · `saturation` | one predicate, one code path |

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

### §5.4 What the P-4 arms still have to answer

| open | closes on | if it comes out wrong |
|---|---|---|
| **`Λ` → accuracy on yahoo.** Cleared at `Λ`=1.04 against a ≥0.95 floor **read off agnews**; never tested across task | yahoo's own accuracy-vs-`Λ` curve | the floor moves and yahoo needs more than 0.95·`B_max` |
| **Does `B_max` drift within one run?** | `sensed=` per fire — yelp-p already says **yes, upward**, 0.522 → 1.023 over 8 | if it drifts a lot, `mean` lags and an EWMA is the fallback; law C self-corrects, so control is unaffected |
| **Is `mean` the right combiner?** | replay the valid arms under all three policies — free, after the fact | `ratchet` stops sooner, `anchor` may not stop. Terminal-flag call |
| **trips/commit ≥ 3** is calibrated on one death and two survivals | every arm reports it per quintile; re-size once there are ten | a config passes preflight and still burns wall |
| **`f`=0.95** rests on the `Λ` floor with **no accuracy evidence** | replay the valid arms at every `f` | a smaller `f` ends runs sooner at the same peak, which is a win |

**Pre-registered before the re-runs:** with the origin fixed, the two knees at commit 150 were 0.248 /
0.237 ⇒ `ρ*` 0.0407 / 0.0397 — nearly identical, so **the agnews-vs-yahoo divergence may not reproduce**.
The live knee tracks how far the model sits above chance, which is a property of the *sensor*, not the
task. **yelp-p and agnews have since diverged strongly and in opposite directions** (§1), so this row is
now about yahoo specifically.

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
