# FLUXTUNE vs FwdLLM / FedBuff(P) / Felix(P) — experiment design (living doc)

**Status:** N=100 α=1 runs landed (2×2 opt ablation + baseline comparison — charter) on the original
3-baseline `main` run-set; E1 headline holds on **peak** accuracy, runs do not yet *hold* the minimum
(Issue I-1, **root-caused** → `fluxtune_contributions.md` §8, next = S1 server optimizer). Since then, the
9-baseline rebuild landed (`../_metadata/BASELINES.md`) and a new sim-mode `main_v2` run-set (4-anchor +
attribution) exists but is **not yet launched** — see §10a's Layer-0 map for what's ready vs. blocked.
**Owner:** dgarg39 · **Branch:** `dg/fluxtune_expts_sim_init`

Human design doc. Its machine-readable twin [`experiments.yaml`](experiments.yaml) is what the tooling
**consumes** — `run_sequential.sh` launches the run-set from it, `compare_baselines.py` reads which runs feed
which metric. Keep them in sync. On disagreement, `experiments.yaml` is source of truth for **what ran**; this
doc for **what we intend and why**.

**Paper ⇄ code reconciliation:** [`EXPTS_CHARTER.md`](EXPTS_CHARTER.md) reconciles this doc with the paper
draft [`05-evaluation.tex`](05-evaluation.tex); [`BRIDGE_DESIGN.md`](BRIDGE_DESIGN.md) is the (in-progress)
9-baseline rebuild's own design/checklist doc — §10a below folds in its Layer-0 tex↔experiment map and §2b
instrumentation ledger, kept in sync as `BRIDGE_DESIGN.md`'s checklist advances. The **run ledger** (which
log on which node feeds which result) is §10b.

Related: [`simulate_fwdllm.md`](simulate_fwdllm.md) (real↔sim parity — principles + the open sim front, §J),
[`../_metadata/BASELINES.md`](../_metadata/BASELINES.md) (baseline catalog + restructure plan),
[`fluxtune_contributions.md`](fluxtune_contributions.md) (systems/ML contributions, incl.
the memory/inference-only-NPU thesis — a motivation/design claim, not an eval experiment).

---

## 0. Architecture: runs vs. analyses (the anti-redundancy backbone)

The organizing rule that makes cross-baseline numbers reusable AND trustworthy:

- **Run-set** — the expensive GPU work. Produced **once** per `(baseline × condition)`. Each run dir
  self-describes via `snapshot.yaml` + `telemetry/*.jsonl`. Governed by the **convergence stop** (§2) with
  `max_runtime_s` / `max_data_id_progress` as safety caps.
- **Analyses** — cheap pure functions over `telemetry/`. **All five experiments are views over the same
  run-set.** Expt 1 *defines* the runs; Expts 2–5 add **zero** GPU runs — they are reducers.
- **Mix-guard** — `compare_baselines.py` discovers the latest run per baseline (exact-token regex, so
  `fwdllm` never captures `fwdllm_plus`) and **warns if the baselines' shared axes (N/partition/trace)
  disagree** — the post-hoc twin of the launch-time `condition_fp`. The fingerprint proves the runs launched
  identically; the mix-guard proves the compared dirs still agree.

Consequence: "Experiment 6" later = a reducer, not new hardware. Re-deriving a metric after a telemetry fix =
re-run the analysis, not the experiment.

```
experiments.yaml ──> run_sequential.sh ──> experiments/run_*/telemetry/*.jsonl
       │                                              │
       └────────────> compare_baselines.py <──────────┘   (reducers → one table/CSV + overlay plots)
```

---

## 1. Baselines (what distinguishes them)

> **Superseded by the 9-baseline rebuild.** [`../_metadata/BASELINES.md`](../_metadata/BASELINES.md) is now
> the canonical baseline catalog (`fwdllm`, `fwdllm_it_unaware`, `fwdllm_it_oracular`, `fedbuff_round`,
> `fedbuff_it_unaware`, `fedbuff_it_oracular`, `felix_round`, `felix_it`, `fluxtune` — the `(P)`/`+IT`/`+O`
> naming grammar + comparison-fit reasoning per baseline). The table below is the **original 3-baseline
> view** (`fwdllm`/`fwdllm_plus`→`fwdllm_it_oracular`/`fluxtune`) that the landed α=1 numbers (§4, §10b) were
> run against — kept as historical record (decision #3: already-landed numbers stay as-is), not extended
> in place. Don't duplicate the 9-baseline table here; read `BASELINES.md` for anything beyond these three.

Substance lives in `_metadata/baselines.yaml`; run YAMLs only pick `baseline:` + a few overrides. The
canonical cross-cutting catalog of **all** baselines is [`../_metadata/BASELINES.md`](../_metadata/BASELINES.md);
the table below is the experiment-local view for the three baselines `main` actually ran.

> **Don't create unnecessary comparison points.** A baseline earns a slot only if it *innovates on the
> same axis one of our contributions claims*, on a substrate where the comparison isn't confounded.
> Our baselines ARE the related works that tried to innovate on those axes; performance is measured
> against exactly those. Backprop CNN/speech selection schemes (refl, feddance, raw oort) are
> **related work to cite, not eval baselines**, unless ported onto the forward-grad LLM substrate.
> Which candidates fit (feddance / felix / refl / oort scored, with the reasoning and a compare-or-not
> decision) is in [`../_metadata/BASELINES.md`](../_metadata/BASELINES.md) §3. **For now we proceed
> with the set we have.**
>
> **Naming — landed 2026-07-23** ([`BASELINES.md`](../_metadata/BASELINES.md) §2, full 9-baseline grammar).
> `fwdllm_plus` → `fwdllm_it_oracular`; the table below (and every already-landed run, §10b) still uses the
> pre-rename `fwdllm_plus` name since those numbers predate the rename and are kept as-is (decision #3).

| Knob | **fwdllm** | **fwdllm_plus** | **fluxtune** |
|---|---|---|---|
| sync / async | sync | sync | **async** (`fedbuff`) |
| selector | `random` | `random` | `async_oort` |
| availability tracking | unaware | **ORACULAR** | client_notify (self-report) |
| reselect | per-round | per-iteration | continuous (async) |
| staleness | `exact` | `round_data_id` | `none` (down-weight) |
| JVP perturbation scoring | off | off | **on** (`jvp_perf_opt`) |
| native `agg_goal` | 10 | 10 | 3 |

**Signed-off experiment override:** the `main` condition matches **`agg_goal=10` across all three** (fluxtune
uses `agg_goal=10` with `C=30`, NOT its native fedbuff 3) — same aggregation batch for a fair head-to-head. A
pre-flight check enforces the match. See §7.0.

**Design risk (Phase 2, mobiperf):** fwdllm_plus's sync barrier (`agg_goal ≥ available trainers`) **cannot
assemble** under unavailability — the pre-flight gate **blocks** this. A fair Expt-1 comparison under a
`mobiperf_*` trace needs a deliberate answer for what fwdllm_plus does. Phase 1 (the signed-off `main`
condition) uses `syn_0` (100% available) where all three complete; the mobiperf policy is a Phase-2 decision.

---

## 2. Convergence stop (run termination) — runner-side watcher

**A run terminates when the aggregator has completed `W` consecutive data bins all remaining ≥ target
accuracy `τ`.** Defaults `W=20`, `τ` per-condition in `experiments.yaml`.

- **Signal:** `agg_eval` telemetry events. Verified cadence: **exactly one `agg_eval` per data bin**, emitted
  at bin completion (`data_id` k → the eval at its final `iteration_per_data_id`). If a bin emits multiple,
  the watcher takes the **last**.
- **Mechanism:** a poller in the `expt_launch` ticker loop tails the aggregator telemetry, maintains the
  per-`data_id` representative accuracy, and fires when the last `W` distinct completed bins are all ≥ `τ` →
  clean process-group termination.
- **Health verdicts:** `CONVERGED` (window satisfied), `STALLED` (early-terminated, not learning),
  `DID_NOT_CONVERGE` (hit wall ceiling still learning), plus `COMPLETED` / `CRASH` / `WALL_CEILING`.
- **Output:** per-run `converge.json` = time-to-converge in **wall + vclock (sim) + data_id + round** (or
  `stall.json` on a stall). *This is the Expt-1 metric captured at the source*, not reconstructed.
- **Wall ceiling = 48h.** Convergence runs are accuracy-governed, so `max_runtime_s` defaults to **172800s
  (48h)** whenever `--target-acc` is set. `max_data_id_progress` also bounds it.
- **Stall guard (early-out).** Terminate BEFORE the 48h ceiling if clearly not learning: **no PROGRESS within
  `stall_window_s` (default 7200s = 2h)** → verdict `STALLED`. Any progress resets the clock; convergence is
  checked first, so a just-converged run is never called stalled. `stall_window_s=0` disables it. **Progress
  is set by `--stall-on` (default `either`):**
  - `acc` — best accuracy gained ≥ `stall_min_delta` (default 0.01 = **1% absolute**; accuracy ∈ [0,1]).
  - `loss` — best (running-min) test-loss dropped ≥ `loss_min_rel_delta` (default 0.01 = **1% relative**; loss
    is unbounded/scale-dependent, so the bar is *fractional vs the running-best*, not absolute).
  - `either` — reset if **either** fired. Motivating case: a run can plateau in accuracy while test-loss keeps
    falling (still learning) — `either`/`loss` keeps it alive; `acc` would kill it.

  Both signals use the running-best (max acc / min loss), so a single noisy eval can neither reset the clock
  nor fake progress. *Validated: flat acc + flat loss → `STALLED [either]`; flat acc + steadily-falling loss →
  clock resets on the loss signal.*

New flags on `run_sequential.sh`: `--target-acc τ`, `--converge-window W`, `--stall-window-s S` (or the hours
alias `--stall-window-h H`, e.g. `6` ⇒ 21600s), `--stall-min-delta D`, `--stall-on acc|loss|either`,
`--loss-min-rel-delta R` (all also settable from the registry via `--run-set`). **The CLI value overrides the
registry** — a run that stalls too eagerly (flat accuracy but loss still falling) can be re-launched with a
wider idle window, e.g. `--stall-window-h 6`, without editing `experiments.yaml`. The stall window is part of
`condition_fp` but is a *termination-policy* knob, not a scientific-condition one: the cross-baseline
mix-guard compares only N/partition/trace, so widening it for one baseline's re-run does not taint the Expt-1
comparison.

---

## 3. Telemetry provenance

Most metrics are **already emitted** (map below). Two instrumentation additions were made, both "route an
already-known quantity to `telemetry.emit`", off when `FLAME_TELEMETRY_DIR` unset (zero cost elsewhere).
**Both implemented and validated** (2026-07-06):

- **(WS3-a) Network bytes + message counts** — a dedicated **`comm` event**
  (`flame/telemetry/events.py:build_comm`), emitted **both directions** with `direction`
  (`agg_to_trainer`/`trainer_to_agg`), `size_bytes`, `payload_kind` (`weights`/`var_bad`/`gradients`),
  `n_tensors`, tagged with round/data_id. Sizes reuse the debug-log values:
  - trainer upload at [`fwdllm_trainer.py`](../../flame/mode/horizontal/syncfl/fwdllm_trainer.py) `_send_weights`;
  - aggregator dispatch at **both** send sites — sync `_distribute_weights` **and** async
    `_distribute_weights_async` (the async site is fluxtune's; instrumenting only sync silently produced zero
    agg→trainer events until fixed).
- **(WS3-b) Perturbations / forward-passes per client** — module-global counters in
  [`fwdgrad_utils.py`](trainer/forward_training/fwdgrad_utils.py) (`_FWD_PASSES` / `_JVP_EVALS`, incremented
  in `calculate_jvp*`; each trainer is its own process ⇒ per-client), read into `trainer_round` as
  `forward_passes_iter/total` + `perturbations_iter/total`. Gives Expt 3 a **hardware-independent** compute
  denominator immune to the 8-GPU contention confound. (Validated: 10 perturbations = 20 forward passes/iter.)

**Aggregator "active GPU time" = aggregator compute wall-time** — measured from the NON-OVERLAPPING `agg_round`
phase fields (`aggregate_fedavg_s` + `eval_s`), **not** summed `step_timing` (double-counts — `timer_decorator`
calls nest), **not** CUDA-event-isolated GPU time. CUDA-event timing only if a reviewer challenges the number.

---

## 4. E1–E5 (analyses over a run-set)

Metric logic lives in `expt_scripts/plotlib/reducers.py` (`load_run` → `RunResult`), consumed by
`compare_baselines.py` (the `expt1..5_*` table wrappers) and `plot_run.py`. Legend — **provenance**: `EMIT`
already in telemetry · `DERIVE` reducer over existing telemetry · `WS3` the instrumentation add (now emitted)
· `WS2` from the convergence watcher. Each E-number below carries its tex `\label` (§10a); the write-ups and
"Observed" callouts are the **landed `main` numbers** (3-baseline, real-mode, α=1) — `main_v2`'s 4/5-baseline
sim-mode numbers don't exist yet (§10a `Ready?` column).

> ✅ **Reducer-audit findings (2026-07-07, tracked in §5 below) — N3/N5/N6 FIXED (2026-07-23,
> `plotlib/reducers.py`/`compare_baselines.py`).** **N3** — E1 time-to-τ used to be *reconstructed* from
> `agg_eval` (streak-over-window scan over the loss-truncated series) rather than reading `converge.json`;
> `expt1_time_to_target` now prefers the watcher's own verdict (matched via `converge.json`'s
> `agg_telemetry` field) when its target/window match, falling back to the reconstruction otherwise (a
> `source` field on the output records which). **N5** — E2 idle used to be a bare `1−busy_frac`; now
> decomposed into `net_wait_frac` (from `mqtt_fetch_s`, emitted but previously unused) + a true residual
> `idle_frac`. **N6** — E5 sync sessions used `contributor_intervals` (dispatch→commit) for *all* baselines,
> undercounting a multi-iteration round to its last iteration's trip time only; new `round_span_durs`
> groups consecutive same-`data_id` cycles into the actual per-data_id engagement span — confirmed ~4×
> different on the real `fwdllm` run (1490 per-cycle samples p50≈11s → 56 round-level samples p50≈45s).
> All three are `CODE-READY`/fixed but not yet `VALIDATED` at smoke scale on the new baselines — see §5.
> `agg_round.contributing_trainers` is still emitted but not consumed by any reducer (§5 row 13).

### E1 (`sec:eval:tta`) — Time to target accuracy
> **Takeaway:** Fluxtune reaches target accuracy faster than FwdLLM and FwdLLM_Plus.
- **Config:** the `main` run-set (landed, real-mode, 3 baselines — §10b). **Reuse:** none (this *defines*
  the runs). **Going forward:** `main_v2` (sim-mode, 4-anchor + `fwdllm_it_oracular` — `experiments.yaml`,
  §10a row `sec:eval:tta`) supersedes `main` for new numbers; not yet launched.
- **Metrics reported:**
  - Time to reach `τ` (the convergence event) — **wall, #rounds, #data_bins, #iterations**. *(WS2 intent;
    currently DERIVE — see N3.)* **Virtual-clock is deferred** (real-mode runs emit no vclock; sim vclock is
    unvalidated, `sim_rate≈0.50`) — add later if a validated sim lands.
  - Maximum accuracy attained. *(DERIVE: max `agg_eval.test-accuracy`)*

> ✅ **Observed (N=100 α=1, 2026-07-08) — supports the takeaway on PEAK accuracy + speed.** Peak test accuracy:
> **FluxTune (R4 full) 84.1% @ 3.9h** reaches target · FwdLLM++ 80.9% @ 7.4h (−3.1 from target, ~2× slower) ·
> FwdLLM 30.3% (never learned). Within the fluxtune 2×2, peak rises with the opt ladder (R1 FluxTune-base 83.0
> → R2 82.2 → R3 grad-aware 83.9 → R4 full 84.1; R1 = C1 guided-JVP base, **not FeLiX** — forward-mode LLM
> fine-tuning, only borrowing FeLiX's scalar agg rate). Plots (`e1_acc_vs_time.pdf`, both sets) mark each run's
> peak with a ★ + legend value against the 84% target line.
> ⚠ **Caveat (Issue I-1):** peak is **transient** — runs oscillate and collapse after the round-1 peak,
> **root-caused** as an undamped high-variance optimizer (NOT the once-suspected epoch bug; see charter I-1 +
> `fluxtune_contributions.md` §8, fix = server optimizer S1 / M2 below). E1 uses the round-1 peak, plots
> clipped there (`--cutoff-mode peak_acc`); the time-to-τ streak never fires (accuracy only grazes 84% amid
> oscillation).

### E2 (`sec:eval:util`) — Resource utilization (wait-time reduction)
> **Takeaway:** Fluxtune improves utilization by cutting wait times at trainers (primary, thousands) and the
> aggregator (secondary, single).
- **Reuse:** **Expt-1 run-set** (no new runs).
- **Approach:** **derived busy/idle time-fraction** (operator call — no hardware sampler).
  - Trainer busy = Σ`gpu_compute_s` / (last−first ts); idle = `mqtt_fetch_s` + inter-round gaps.
  - Aggregator busy = Σ(`aggregate_fedavg_s`+`eval_s`) / `agg_round.wall_elapsed_s` (non-overlapping phases,
    NOT summed `step_timing`); wait fractions = Σ`barrier_wait_s` / Σ`drain_tail_s` over wall.
- **Metrics reported:** P50 / P90 / P99 busy-fraction **across trainers** over runtime; aggregator busy vs.
  barrier-wait vs. drain fraction. *(DERIVE)*

### E3 (`sec:eval:compute`) — Compute productivity (learning per unit compute)
> **Takeaway:** At resource-constrained clients, Fluxtune yields more learning per unit compute.
- **Reuse:** **Expt-1 run-set**.
- **Metrics reported:** `Δloss / cumulative compute`, against **two** compute denominators:
  - **Forward passes** (perturbations): Σ per-client perturbation count — **PRIMARY / clean**. *(WS3-b —
    hardware-independent: counts actual passes regardless of contention.)*
  - GPU-seconds: Σ(trainer `gpu_compute_s`) + aggregator compute wall-time — **SECONDARY, confounded**.
    *(DERIVE.)* ⚠ `gpu_compute_s` is **wall-time** GPU work; 100 trainers time-share **8 GPUs**, so contention
    inflates it (~8–10 ms/pass clean → ~0.21 s/pass under load, ≈20×), and Fluxtune's ~30 concurrent clients
    contend differently than the sync baselines' K=10 bursts — so this denominator measures scheduling
    contention, not algorithmic compute. Report with the caveat; lean on the forward-pass denominator.
  - `Δloss` = first `agg_eval.test-loss` − final `agg_eval.test-loss`, **in time order** (never keyed by
    `data_id`, which cycles per round). *(EMIT)*

> ⚠ **Observed (N=100 smoke, 2026-07-07) — does NOT yet support the takeaway.** Learning-per-compute ranks
> **FwdLLM++ > Fluxtune > FwdLLM** on *both* denominators (Δloss/GPU-h 0.055 vs 0.016 vs −0.014; Δloss/M-fwd
> **5.62 vs 1.27** vs −1.26). Fluxtune attains the most *total* Δloss (0.82) and the fastest wall-clock
> convergence (E1) but spends **~5–6× the forward-pass compute** to get there, so per-unit it is *less*
> efficient. GPU-h is confounded by 8-GPU contention (§7.1); the clean forward-pass denominator still favors
> FwdLLM++.
> **Why:** Fluxtune's async high-concurrency design does more *unproductive* compute — (i) fedbuff
> **staleness**: concurrent clients train on stale models and their updates are down-weighted, so part of the
> forward-pass compute yields little Δloss; (ii) it keeps ~30 clients busy speculatively. FwdLLM++ is
> synchronous with **oracular** availability, so every forward pass feeds a fresh, fully-weighted update on a
> client that will contribute. Concurrency buys wall-clock speed, not compute efficiency.
> **Optimize:** staleness-aware admission / adaptive concurrency + cutting JVP forward-pass overhead. (The
> per-resource-constrained-client framing still needs a per-client-normalized metric — a metric decision, not
> a plot bug.)

### E4 (`sec:eval:comm`) — Data transmitted over the network
> **Takeaway:** Fluxtune incurs lower total data overhead despite more messages per round/iteration.
- **Reuse:** **Expt-1 run-set** (requires WS3-a telemetry present at run time).
- **Metrics reported:** total messages sent (each side); total bytes transmitted (each side); per-message size
  distribution. *(WS3-a)*

> ⚠ **Observed (N=100 smoke, 2026-07-07) — contradicts the takeaway.** Fluxtune transmits **~1.9× more total
> bytes** than FwdLLM++ (**146 vs 79 GB**) and ~1.6× more messages — it does *not* incur lower overhead here.
> Per-message size is identical (1.8 MB/upload), so the gap is message **count**, not payload.
> **Why:** (i) **Model distribution dominates** — FwdLLM++ (sync) sends the model once per round to the K
> selected clients (weights **9 GB**); Fluxtune (async) has ~30 clients *continuously re-pull* the latest
> global model as they finish and re-enlist → weights balloon to **88 GB**. (ii) **Uploads scale with
> iterations** — one gradient upload (1.8 MB) per trainer-iteration; Fluxtune runs ~1.6× more iterations
> (concurrency) → 58 vs 36 GB up. FwdLLM++'s synchronous rounds amortize both, and oracular selection avoids
> dispatching to non-contributors (it *does* pay a 33 GB method-specific `var_bad` payload, yet still totals
> less). **Same root cause as E3:** async concurrency does more total work.
> **Optimize:** delta/compressed model distribution on re-pull + staleness-aware throttling would cut
> Fluxtune's dominant weight-download term.

> 🔧 **Single-contribution results — efficiency levers not yet on.** Contributions (charter B2): **C1** guided
> perturbations · **C2** dynamic K/C · **C3** gradient-aware aggregation. In `main`: **C1 active**; **C2 OFF**
> (static `agg_goal=10`/`C=30` for the agg_goal-matched head-to-head, §7.0; `dynamic_kc.enabled=false`); **C3**
> runs the fedbuff "new" staleness×utility *scalar* rate + `var≤0.3` gate — a borrowed FeLiX placeholder, not
> the intended gradient-aware rule (charter N2; `grad_aware`=Opt-3). C2/C3 target exactly the E2-E4
> inefficiencies. **So E2/E3/E4 efficiency claims need a separate full-system run (C2 + real C3 on)** —
> enabling dynamic K/C breaks the agg_goal match. **E1 (speed + final accuracy) holds on C1 alone.**

### Experiment 3-adjacent — C3 intelligent (gradient-aware) aggregation: design investigation
> **Status: NOT the intended contribution yet.** The active weighting (above) is a borrowed scalar rate.
> Fluxtune needs a **gradient-aware** rule. **Full design starter:**
> [`docs/aggregation_design.md`](docs/aggregation_design.md) (FedBuff→FeLiX→FluxTune regime, hypothesis,
> 5-axis design space, candidate schemes S0–S5). This section seeds that design (implement later, then move
> the feature doc to [`fluxtune_contributions.md`](fluxtune_contributions.md) and delete from here).
> **Dimensions to evaluate:**
> 1. **Staleness under iteration-based progression.** Staleness = `agg_model_version − trainer_version`;
>    `_model_version` advances **per data-bin completion**, and a data-bin completes only when the **variance
>    threshold is met** ([`fwdllm_aggregator.py:1422`](../../flame/mode/horizontal/syncfl/fwdllm_aggregator.py#L1422),
>    [`fedbuff.py:194`](../../flame/optimizer/fedbuff.py#L194)) → staleness accrues at the *variance-gated
>    data-bin rate, not wall-clock*. Hypothesis: staleness grows **slower** in Fluxtune than round-based FL.
>    Quantify the effective staleness distribution vs a round-based baseline.
> 2. **Scalar rate vs gradient-aware combination.** Scalar-multiplying a *gradient* update (forward-mode JVP
>    estimate) may be the wrong operator vs down-weighting *weights*. Explore direction-/variance-aware
>    combination (weight by JVP magnitude / SNR / agreement with the running aggregate), not just staleness×loss.
> 3. **Interaction with C1 + the variance gate.** Updates already passed `var ≤ var_threshold`; does
>    re-weighting by loss (`stat_utility`) double-count what the gate filtered?
> 4. **Interaction with C2 (dynamic K/C).** Concurrency C sets how many stale/in-flight updates coexist;
>    aggregation rule and concurrency controller co-determine the wasted work E3/E4 measure.

### E5 (`sec:eval:sessions`) — Client training-session durations & participation
> **Takeaway:** Fluxtune's client sessions are much shorter than FwdLLM's.
- **Reuse:** **Expt-1 run-set**.
- **Active-session definition (precise, per operator) — N6-fixed (§5 row 12):**
  - **async (fluxtune, fedbuff_round/it_*, felix_round/it):** selected → next reselection ≈
    `dispatch_ts → commit_ts` (`agg_round.contributor_intervals` per contribution — `session_durs`).
  - **sync (fwdllm / fwdllm_it_unaware / fwdllm_it_oracular):** one-round span — the per-`data_id`
    engagement duration (first `agg_round` cycle's ts → the cycle that commits it), NOT the old
    per-cycle `contributor_intervals` reading (`round_span_durs`, `plotlib/reducers.py`).
- **Metrics reported:**
  - P50 / P90 / P99 of session duration across clients + a histogram. *(DERIVE)*
  - **Per-client participation counts at three granularities across baselines** — #rounds, #data_bins,
    #iterations each client participated in. *(DERIVE: `trainer_round` + `agg_round.contributing_trainers`)*

### Experiments M1–M2 — training-stability track (motivation / ablation)
> Context: the α=1 N=100 runs oscillate and single-class-collapse (charter I-1). The H0 diagnostic **refuted**
> data-class-bias as the cause (`fluxtune_contributions.md` §8, F11-F15) → the driver is the **undamped,
> high-variance forward-gradient optimizer**. These two experiments motivate and validate the fix.

**M1 — Data-bin size × heterogeneity (the forward-mode bias/variance tradeoff).**
- **Hypothesis (bidirectional):** per-commit **variance** falls as bin/cohort size grows (favors *large* bins),
  BUT a large, class-mixed bin **averages conflicting per-sample gradients → small mean-gradient magnitude →
  the scalar JVP signal sinks below its noise floor and learning stalls** (favors *small* bins). ⇒ an
  **α-dependent optimum**: too-small = oscillation (observed), too-large = no learning (recalled).
- **Design:** sweep `train_batch_size` (bin) ∈ {2,4,8,16,32,…} × α ∈ {0.1,1,100}, N=100, fluxtune. Report
  peak/final acc, Δloss, and **per-commit JVP SNR / variance** (WS3-b + var telemetry). Cross-baseline (bin
  size is a shared knob), flag-gated.
- **Reads on:** whether small bins are *required* for a usable forward-gradient signal — i.e. we **cannot**
  just "enlarge bins to kill variance" (the H2 caveat) → motivates fixing variance at the **aggregator** (M2),
  not the data.

**M2 — Aggregator optimizer (descent to a minimum) vs. random walk. [FLUXTUNE CONTRIBUTION]**
- **Claim:** FwdLLM applies each committed JVP estimate as a **raw, undamped SGD step**
  (`FedSgdAggregator.py:322-324`) → under small-bin noise the global model **random-walks** (charter I-1; §8
  F8/F13). Fluxtune's contribution is a **server-side optimizer that integrates the noisy-but-informative
  small-bin updates into a smooth descent and holds the minimum** — momentum / EMA-of-weights / adaptive step
  (§8 S1). **Distinct from and composable with C3**: C3 reweights contributions *within* a commit; the
  optimizer damps *across* commits.
- **Metric:** sustained peak (no post-peak divergence) · monotone test-loss envelope · and the real prize —
  the run **fills the W=20 convergence window** (never fires today, I-1) → turns E1's *transient* peak into a
  genuine time-to-τ. Compare undamped baseline vs optimizer, α=1 N=100.
- **Opportunity experiments:** (a) momentum/EMA coefficient sweep; (b) does the optimizer let the run
  *converge & sustain* 84%?; (c) optimizer × grad-aware (Opt-3) — does damping remove the "R4 diverges worst"
  effect?; (d) optimizer × bin size (M1) — does a real optimizer widen the usable bin range?
- **Validate / fix before finalizing:** (i) optimizer-not-data established (✅ H0); (ii) confirm momentum does
  not **double-damp** with the fedbuff staleness/utility rate and the variance gate; (iii) **real↔sim parity**
  preserved (server optimizer state must be deterministic under the frozen update order); (iv) gain holds
  across α, not just α=1. Flag-gated, default off; A/B before permanent (contributions §8 rule 6).

---

## 5. Instrumentation-readiness ledger (§2b, `BRIDGE_DESIGN.md`)

Supersedes the old bare "Provenance | Feeds" table with the full readiness-level ledger — **row numbers 1–13
are kept aligned with `BRIDGE_DESIGN.md` §2b** (that doc's own copy; edit both together, they must not
drift). **Rule: no run-set launches (not even smoke) while any metric it needs is below `CODE-READY`** — see
`BRIDGE_DESIGN.md` §4 gate 0 for the full readiness-level definitions (`PLANNED` / `NEEDS CODE` /
`CODE-READY` / `VALIDATED` / `REDUCER-GAP`).

| # | Metric | Telemetry event.field | Provenance | Status | Feeds | Note |
|---|---|---|---|---|---|---|
| 1 | Experiment wall-time before exit | `converge.json` (WS2) | WS2 | CODE-READY (N3 fixed) | E1 | `expt1_time_to_target` prefers `converge.json` (matched via `agg_telemetry`) over the `agg_eval` reconstruction when target/window match. No real `--target-acc` run exists yet — `VALIDATED` on first convergence-watched launch. |
| 2 | Aggregator active GPU time | `agg_round.aggregate_fedavg_s`+`eval_s` | DERIVE | VALIDATED (existing runs) | E2,E3 | Re-validate on `fedbuff_round`/`felix_round` once they exist — different aggregator config, same entrypoint. |
| 3 | Each client active GPU time | `trainer_round.gpu_compute_s` | EMIT | CODE-READY (N5 fixed) | E2,E3 | `expt2_utilization` now splits idle into `net_wait_frac` (`mqtt_fetch_s`, previously unused) + residual `idle_frac`. `barrier_wait_s`/`drain_tail_s` stay excluded (agg-side, ≈0 in sim). Confirmed firing on both existing real runs; formal per-baseline SIM smoke validation still pending (checklist step 6). |
| 4 | Loss at first iteration | first `agg_eval.test-loss` | EMIT | VALIDATED | E3 | Time-ordered (data_id-cycling bug already fixed). |
| 5 | Loss after training complete | last `agg_eval.test-loss` | EMIT | VALIDATED | E3 | Same. |
| 6 | Updates used per iteration | `agg_round.agg_goal_count`/`updates_in_queue` | EMIT | VALIDATED, unused | — | Not currently feeding any of E1-E5 directly. |
| 7 | Total updates over training | Σ `agg_round.agg_goal_count` | EMIT | VALIDATED, unused | — | Same. |
| 8 | Forward passes / perturbations per client | `trainer_round.forward_passes_total`/`perturbations_total` | WS3-b | VALIDATED (fwdllm/fwdllm_it_oracular/fluxtune only) | E3 | Client-side counter in `fwdgrad_utils.py`, selector-independent by code inspection (`FedSgdTrainer.py` emit site has no baseline/selector branching) — **must still smoke-validate on `fedbuff_round`/`felix_round`** before calling it `VALIDATED` for them. |
| 9 | Data sent from aggregator | `comm{direction=agg_to_trainer}.size_bytes` | WS3-a | VALIDATED (sync+async paths, existing 3 baselines) | E4 | New async baselines reuse fluxtune's already-instrumented async dispatch site (confirmed by reading `_distribute_weights_async` — the `build_comm` call is selector-agnostic) — still smoke-validate per baseline, don't assume from fluxtune's pass. |
| 10 | Data sent from clients | `comm{direction=trainer_to_agg}.size_bytes` | WS3-a | VALIDATED (existing 3 baselines) | E4 | Same caveat as #9. |
| 11 | Client active: selected→next reselection (async) | `agg_round.contributor_intervals` | EMIT | VALIDATED | E5 | Built in `_process_aggregation_goal_met`, shared by sync+async, not selector-conditional. |
| 12 | Client active per round (sync reselect) | `trainer_round`/`agg_round` ts | EMIT | CODE-READY (N6 fixed) | E5 | New `round_span_durs`: groups consecutive same-`data_id` `agg_round` cycles into one per-data_id span. Confirmed materially different on the real `fwdllm` run (old: 1490 samples p50≈11s; fixed: 56 samples p50≈45s, ~4×) — a real undercounting bug, not cosmetic. |
| 13 | Participation counts (rounds/data_bins/iterations) | `trainer_round` + `contributing_trainers` | EMIT | VALIDATED, `contributing_trainers` unused | E5 | Reducer only reads a subset. |
| 14 | `commit_reason` (natural/cap/plateau) | `agg_round.commit_reason` | EMIT | VALIDATED (R1–R4 opt-ladder) | ablation | Feeds the L-ladder ablation only (`BRIDGE_DESIGN.md` §2). |
| 15 | `grad_aware_gated_total` | aggregator telemetry | EMIT | VALIDATED (R1–R4) | ablation | Feeds L-ladder / L1L3-c3. |
| 16 | Selector identity per iteration (random/cosine/quasi/JVP) | *unknown* | NEEDS CODE CHECK | — | ablation | Needed for the `c1alt` selection-policy ablation; unverified whether non-JVP-guided selectors even exist. |
| 17 | JVP threshold/refresh-frequency config | *config, not telemetry* | NEEDS CODE CHECK | — | ablation | Feeds the `jvp` sensitivity ablation. |
| 18 | Effective K/C value over time (dynamic_kc) | *suspected missing* | NEEDS CODE | — | ablation | Feeds the `kc` ablation's sensitivity analysis — likely needs new instrumentation, not just a config sweep. |
| 19 | `align_floor`/`inverse_var` gate telemetry | aggregator telemetry | EMIT | VALIDATED (R3/R4) | ablation | Feeds the `c3` ablation's grad-aware arm; the `naive_avg` arm still needs the aggregation mode built. |

**Per-baseline validation checklist** (rows 8–10 depend on this — "code exists" ≠ "confirmed firing per
baseline"). Fill in during smoke tests (checklist step 6), one row per newly-wired baseline:

| Baseline | WS3-a `comm` both directions | WS3-b forward-pass/perturbation counters | Notes |
|---|:---:|:---:|---|
| fwdllm | ✅ validated | ✅ validated | existing |
| fwdllm_it_unaware | ✅ validated | ✅ validated | existing, rename only |
| fwdllm_it_oracular | ✅ validated | ✅ validated | existing, rename only |
| fluxtune | ✅ validated | ✅ validated | existing |
| fedbuff_round | ⬜ smoke-pending | ⬜ smoke-pending | async dispatch site shared w/ fluxtune (confirmed by code read) — expected to pass, don't skip the check |
| fedbuff_it_unaware | ⬜ smoke-pending | ⬜ smoke-pending | — |
| fedbuff_it_oracular | ⬜ smoke-pending | ⬜ smoke-pending | — |
| felix_round | ⬜ smoke-pending | ⬜ smoke-pending | async_oort selector — verify WS3-b counters aren't selector-coupled |
| felix_it | ⬜ smoke-pending | ⬜ smoke-pending | — |

---

## 6. Implementation workstreams & status

| WS | Item | State |
|----|------|-------|
| WS1 | `EXPERIMENTS.md` + `experiments.yaml` (backbone) | ✅ |
| WS2 | Convergence-stop watcher (`--target-acc`,`--converge-window`, `CONVERGED`/`DID_NOT_CONVERGE`, `converge.json`) | ✅ validated N=10 (fired at bin 0) |
| WS3-a | Network bytes + message-count telemetry (`comm` event, both directions) | ✅ validated (agg→trainer + trainer→agg) |
| WS3-b | Per-client perturbation / forward-pass counter (`fwdgrad_utils` counters → `trainer_round`) | ✅ validated (20 passes / 10 perturbations per iter) |
| WS4 | `compare_baselines.py` cross-baseline reducer → table/CSV + overlay plots | ✅ (mix-guard + degrades on missing telemetry) |
| — | Enhanced gate: `condition_fp` (incl. delay_factor), tier ② baseline internals, agg_goal-match check, `--run-set` registry launch | ✅ |
| — | Phase 2: fwdllm_plus-under-mobiperf policy; N=100/8-GPU utilization confound | ⬜ deferred |

**Validated end-to-end at N=10** (run `run_20260706_152025_fluxtune_n10_smoke_syn_0_real`): `CONVERGED` fired,
`comm` bytes (both directions) + `forward_passes`/`perturbations` landed in JSONL, comparison table produced.
**Sign-off applied** (2026-07-06): `main` is now N=100, delay_factor=2, agg_goal=10 matched, target 0.82.
**Next: the N=100 real convergence runs** (§7).

---

## 7. Run workflow (multi-node, misconfig-proof)

`experiments.yaml` + the enhanced gate define a comparison run **once**; every node verifies it launched the
same thing.

### 7.0 Signed-off condition (2026-07-06) — `run_set: main`
N=100 · K=10 · C=10 (sync) / 30 (fluxtune) · **agg_goal=10 matched across all three** · partition alpha=1 ·
syn_0 · delays ON at **delay_factor=2** · target 0.84 / window 20 · **wall ceiling 48h · stall-out after 2h
with no ≥1% gain**. A pre-flight check enforces the agg_goal match; the `condition_fp` (includes delay_factor)
must match across nodes. **Manual pre-run check:** verify the partition group `niid_label_clients=100_alpha=1`
exists in `agnews_partition.h5` (the gate only warns — it can't read the H5 from the launch env); a missing
group crashes all 100 trainers.

### 7.1 Baseline defaults — REVIEW before the first real run
Resolved from `_metadata/baselines.yaml` (the gate's tier ② shows these live per run):

| knob | fwdllm | fwdllm_plus | fluxtune |
|---|---|---|---|
| mode | sync | sync | **async** |
| selector | random | random | **async_oort** |
| optimizer | fedavg | fedavg | **fedbuff** (lr 0.075) |
| native agg_goal | 10 | 10 | **3** |
| native k / c | 5 / 15 | 5 / 15 | 5 / 15 |
| availability tracking | unaware | **ORACULAR** | client_notify |

**agg_goal / k coupling (decided 2026-07-06):** `--c N` fans `agg_goal=N` + `minInit` to **every** baseline
and `--k N` sets k on all — so `--run-set main` (C.sync=10, K=10) drives fluxtune to `agg_goal=10, k=10`.
**Intended:** the experiment matches `agg_goal=10` across all three, and fluxtune's async concurrency is set by
`C.async=30` (its `c` becomes 30, `agg_goal` stays 10). The agg_goal-match pre-flight check enforces it. A
future experiment wanting fluxtune's native `agg_goal=3` needs a per-baseline agg_goal override (not currently
wired).

### 7.2 Single-node run (all three baselines)
```bash
cd lib/python/examples/fwdllm/expt_scripts
bash run_sequential.sh --run-set main --mode real --yes
```

### 7.3 Multi-node run (split baselines, ONE source of truth)
All nodes read the SAME condition from `experiments.yaml` via `--run-set main`; only `--only` differs. **The
gate prints `condition_fp` — it MUST be identical on every node.** If fingerprints differ, a knob was
mistyped: stop and fix.

**Three-node run (one baseline per node) — pass `--clean` so each node auto-clears any stray workers from a
prior run before launching:**
```bash
# node A
bash run_sequential.sh --run-set main --only fwdllm              --mode real --clean --yes
# node B
bash run_sequential.sh --run-set main --only fwdllm_it_oracular   --mode real --clean --yes
# node C  (shared filesystem: run dirs land in the same experiments/)
bash run_sequential.sh --run-set main --only fluxtune             --mode real --clean --yes
```
Pre-run checklist (the gate does most of this — eyeball, don't skip):
1. `condition_fp` identical across nodes (expect `04d64814` for the current `main`).
2. tier ② `mode/selector/optim` match the baseline table above (right algorithm per baseline).
3. `target_acc`, `trace`, `part` are the intended values (🟢 = from flag/registry); `part` = `niid_label_clients=100_alpha=1`.
4. no ✗ pre-flight checks (a ⚠ on the niid partition group just says "verify it exists").
5. `[<baseline>] clean slate verified` printed before launch (the clean-slate guard, §7.6).

### 7.4 After the runs — compare
```bash
python compare_baselines.py --variant real --target-acc 0.84 --window 20 --plots
```
Reducers pull from the same run dirs; the **mix-guard** warns if the baselines' shared axes (N/partition/trace)
disagree — the post-hoc twin of `condition_fp`.

### 7.5 Convergence-stop knobs (recap)
`--target-acc 0.84 --converge-window 20` arm the watcher (WS2). A run ends `CONVERGED` (window met,
`converge.json` written) or `DID_NOT_CONVERGE` (hit a safety cap). `--run-set main` sets these from the
registry, so you rarely pass them.

### 7.6 Stopping a run & the clean-slate guard
The run's trainers/aggregator run in their **own process group** (so the watcher can signal the whole tree),
which means a bare terminal **Ctrl+C would not reach them**. `run_sequential.sh`/`expt_runner.sh` install a
**SIGINT/SIGTERM trap**: one Ctrl+C tears down the run's process group + the convergence watcher
(`converge_watch.py`) + the progress ticker, escalates SIGTERM→SIGKILL after a short grace
(`EXPT_INT_GRACE_S`, default 5s), sweeps stragglers, and frees GPU/RAM.

**Manual teardown** (if a run was killed the wrong way and left orphans):
```bash
pkill -TERM -f 'flame.launch.run_experiment'; sleep 3
pkill -9 -f 'trainer/forward_training'; pkill -9 -f 'trainer/pytorch/main.py'
pkill -9 -f 'aggregator/pytorch/main_';  pkill -9 -f converge_watch.py
pkill -9 -f run_sequential.sh
# verify clean (want: nothing, GPU ~0 MiB)
pgrep -af -u "$USER" -f 'run_experiment|forward_training|trainer/pytorch/main.py|aggregator/pytorch/main_|converge_watch.py' || echo clean
nvidia-smi --query-gpu=index,memory.used --format=csv
```

**Clean-slate guard (`expt_assert_clean_slate`).** Before every launch the runner checks for stray FL workers
(own procs only) and residual GPU memory:
- **default:** if the node is dirty it **ABORTS** and prints the kill command (never nukes a process you didn't
  sign off on — safe on shared boxes);
- **`--clean`** (or `EXPT_AUTOCLEAN=1`): kills the stragglers, re-verifies, and only aborts if still dirty;
- `EXPT_GPU_FREE_MB` (default 500) warns on residual GPU memory; `EXPT_GPU_STRICT=1` turns that warning into an
  abort.

---

## 8. Files & entry points

| File | Role |
|---|---|
| [`experiments.yaml`](experiments.yaml) | machine registry — run-sets, conditions, analyses (source of truth for *what runs*) |
| [`expt_scripts/run_sequential.sh`](expt_scripts/run_sequential.sh) | launcher — `--run-set`, condition_fp gate, tier ② internals, agg_goal-match check, convergence flags |
| [`../scripts/expt_runner.sh`](../scripts/expt_runner.sh) | shared harness — `expt_launch` (arms watcher + SIGINT/SIGTERM teardown), `expt_assert_clean_slate` (pre-launch guard), `expt_assert_run` (`CONVERGED`/`DID_NOT_CONVERGE`) |
| [`../scripts/converge_watch.py`](../scripts/converge_watch.py) | WS2 side-car — polls `agg_eval`, writes `converge.json`, kills the run on convergence |
| [`expt_scripts/compare_baselines.py`](expt_scripts/compare_baselines.py) | WS4 reducer — 5-experiment table/CSV + overlay plots + mix-guard (cross-baseline) |
| [`expt_scripts/plot_run.py`](expt_scripts/plot_run.py) | per-run twin — streams ONE run's telemetry (handles the >1 GB agg JSONL) → full 5-experiment plot set + `summary.json` |
| `flame/telemetry/events.py` | `build_comm` (WS3-a) |
| `flame/.../fwdllm_aggregator.py`, `fwdllm_trainer.py` | `comm` emit sites (WS3-a, both dispatch paths + upload) |
| `trainer/forward_training/fwdgrad_utils.py`, `FedSgdTrainer.py` | forward-pass counters (WS3-b) |

Session artifacts: `expt_scripts/smoke_logs/<ts>/` (`converge_<run>.json`, manifest, gate spec); per-run
`experiments/run_*/telemetry/*.jsonl`; comparison output `experiments/_compare/`.

---

## 9. Changelog
- **2026-07-23 — 9-baseline rebuild + Layer-0 bridge (`BRIDGE_DESIGN.md` checklist steps 1–5).** Rebuilt
  `_metadata/baselines.yaml` to 9 keys (5 new async baselines `fedbuff_round/it_*`, `felix_round/it` +
  `fwdllm_it_unaware`; `fwdllm_plus`→`fwdllm_it_oracular` renamed); propagated the rename through
  `run_sequential.sh`, tests, `plotlib/baselines.py` legend colors, `experiments.yaml` (local `baselines:`
  stanza deleted per no-duplication — see §1's new pointer to `BASELINES.md`). Fixed reducer gaps N3/N5/N6
  (§5) + a latent `is_async` hardcode bug in `compare_baselines.py` that would have mislabeled every new
  async baseline as sync. Added `main_v2` (sim-mode, 4-anchor + `fwdllm_it_oracular`) + tex-anchored
  `e1..e5`/`a0_attribution` analyses to `experiments.yaml`; added the Layer-0 tex↔experiment map (§10a) and
  replaced §5's bare metric table with the full instrumentation-readiness ledger. Async round-cache
  extension (`fwdllm_aggregator.py`) fixes a real gap the rename surfaced: `reselect_each_iteration` only
  worked on the sync path before this, so `fedbuff_round`/`felix_round` would have been byte-identical to
  their `+IT` siblings. 403 tests green. `main_v2` not yet launched (blocked on checklist step 6's
  per-baseline smoke validation). See `BRIDGE_DESIGN.md` for full detail — this doc keeps only the
  human-facing summary + the two synced tables (§5, §10a).
- **2026-07-08 — stability track (M1–M2).** Added M1 (data-bin size × α bias/variance tradeoff) + M2
  (aggregator optimizer vs. random walk — fluxtune contribution). Motivated by charter I-1 + the H0 diagnostic
  refuting data-class-bias (`fluxtune_contributions.md` §8, F11-F15). Flag-gated; ledger row added (§10).
- **2026-07-07 (g) — paper ⇄ code reconciliation ([`EXPTS_CHARTER.md`](EXPTS_CHARTER.md)).** Corrections: (i)
  contribution taxonomy → C1/C2/C3 (async is structural); (ii) C3 is a borrowed fedbuff scalar-rate placeholder,
  not gradient-aware (+ C3 design investigation, §4); (iii) E1 units → wall/rounds/data-bins/iterations, vclock
  deferred; (iv) E3 forward-pass denominator primary, GPU-seconds secondary; (v) reducer gaps N3/N5/N6 logged;
  (vi) run ledger added (§10). Open: α (config 1, runs ran 0.1). Paper-side rewrites in `05-evaluation.tex`.
- **2026-07-07 (f) — E3/E4 observed (Fluxtune trades efficiency for speed).** N=100 smoke contradicts E3/E4:
  FwdLLM++ more compute-efficient (Δloss/M-fwd 5.62 vs 1.27) + communication-efficient (79 vs 146 GB). Both:
  async concurrency does more total work (staleness + speculative compute; weight re-pulls 88 vs 9 GB). Fluxtune's
  win is E1 speed. §4 has optimization directions. Caveat: only C1 on; C2/C3 OFF → E2/E3/E4 need a full-system run.
- **2026-07-07 (e) — target 0.82 → 0.84 + paper-figure pipeline.** Raised `main` target to **0.84** in
  `experiments.yaml` + plot defaults. Added `expt_scripts/plotlib/` + `make_paper_figs.py`;
  `plot_run.py`/`compare_baselines.py` on the shared reducer. Fixed the data_id-keyed eval bug (cycles per round,
  corrupted Δloss) → time-ordered series. Per-run cutoff at last significant test-loss improvement; EMA `--smooth`.
- **2026-07-07 (c) — loss-aware stall guard (`--stall-on`).** Idle clock resets on **acc / loss / either**
  (default `either`). Loss = relative drop vs running-best (`--loss-min-rel-delta` 1%); acc = absolute 1%; both
  running-best. Fixes killing a run whose acc plateaus while loss still falls. Wired through `converge_watch.py`,
  `expt_runner.sh`, `run_sequential.sh` (+ `condition_fp`), `experiments.yaml`. Validated end-to-end.
- **2026-07-07 (b) — CLI stall window (hours alias).** `--stall-window-h H` on `run_sequential.sh` overrides the
  registry `stall_window_s`. Motivated by a fwdllm N=100 run the 2h guard killed at the start (loss still
  falling). Re-launch: `run_sequential.sh --run-set main --only fwdllm --mode real --clean --yes --stall-window-h 6`.
- **2026-07-07 — per-run plots + ticker-orphan fix.** Added `plot_run.py` (single-run twin, streams the >1 GB agg
  JSONL → 5-experiment plot set + `summary.json`; first N=100 fluxtune run `run_20260706_185045…` STALLED at
  **84.08%**, 139 bins). Fixed a harness hang: the backgrounded ticker wasn't in its own process group, so the
  process-group `kill` missed it and `wait` blocked forever; the ticker now self-terminates by PID when the run dies.
- **2026-07-06 (c) — teardown, clean-slate guard, alpha=1.** Ctrl+C/SIGTERM tears down the run (own process
  group) + watcher + ticker + frees GPU/RAM. Added `expt_assert_clean_slate` (`--clean`/`EXPT_AUTOCLEAN`), §7.6.
  **`main` partition alpha=0.1 → alpha=1** (group present in `agnews_partition.h5`); `condition_fp` now `04d64814`.
- **2026-07-06 (b) — termination policy.** **48h wall ceiling** (was 1h) when `--target-acc` set + **stall guard**
  (`STALLED` if best-acc gain < `stall_min_delta` 1% within `stall_window_s` 2h). Wired through watcher, registry,
  gate, `--run-set`. Unit + integration tested.
- **2026-07-06 — implementation landed & N=10-validated.** WS2 watcher; WS3-a `comm` telemetry both directions;
  WS3-b forward-pass/perturbation counters; WS4 `compare_baselines.py`; enhanced gate (`condition_fp` incl.
  delay_factor, tier ② internals, agg_goal-match, `--run-set`). Sign-off: N=100, delay_factor=2, agg_goal=10, target 0.82.
- _(init)_ Doc + registry scaffold. Backbone, convergence-stop spec, metric map (incl. #8 forward passes, #13
  participation granularities).

---

## 10a. Layer-0 — tex↔experiment map

The actual "bridge" artifact (`BRIDGE_DESIGN.md` §2, kept in sync here): for every tex `\label` in
`05-evaluation.tex`, which experiment id, which run-set, and — the launch gate — whether every metric it
needs is at least `CODE-READY` (§5). This replaces free-form cross-referencing with a literal checklist:
*yaml exists / run launched / telemetry verified / plot generated / number in tex?*

| tex label | Experiment id | Baselines needed | Run-set | `Ready?` | Status |
|---|---|---|---|---|---|
| `sec:eval:tta` | **E1** time-to-accuracy | fwdllm, fedbuff_round, felix_round, fluxtune | `main_v2` | ⚠ NOT LAUNCHED | fwdllm/fluxtune landed on `main` (real, 3-baseline, α=1, §10b); `main_v2` (sim, 4-anchor) added to `experiments.yaml`, validated via `--dry-run`, **not yet launched**. N3 fixed. |
| `sec:eval:util` | **E2** resource utilization | same as E1 | reuses `main_v2` | ⚠ NOT LAUNCHED | N5 fixed; blocked only on the launch. |
| `sec:eval:compute` | **E3** compute effectiveness | same as E1 | reuses `main_v2` | ⚠ NOT LAUNCHED | reducer exists, `main`'s 3-baseline numbers are stale (superseded); needs `main_v2` launch + fedbuff_round/felix_round WS3-b re-validation (§5 row 8). |
| `sec:eval:comm` | **E4** communication | same as E1 | reuses `main_v2` | ⚠ NOT LAUNCHED | reducer exists, numbers stale; needs `main_v2` launch + per-baseline WS3-a re-validation (§5 row 9-10). |
| `sec:eval:sessions` | **E5** session length | same as E1 | reuses `main_v2` | ⚠ NOT LAUNCHED | N6 fixed; blocked only on the launch. |
| `sec:eval:attribution` | **A0** attribution (reuses E1–E4 metrics) | fluxtune vs `fwdllm_it_oracular` | reuses `main_v2` | ⚠ blocked on E1-E4 | `a0_attribution` analysis added to `experiments.yaml`, scoped to the 2-baseline pair — closest to already-landed (old `fwdllm_plus` vs `fluxtune` comparison, renamed), same launch blocker as E1-E4. |
| `sec:ablation:ladder` | **L-ladder** opt on/off ladder | fluxtune only, 4 configs (R1–R4) | `opt_ladder` (existing, informal) | ✅ landed / ⬜ L1-completion unrun | ✅ **landed** (§10b R1–R4, `commit_reason`+`grad_aware_gated_total` telemetry validated); per-contribution C1/C2/C3 on/off ("L1 completion") is separate, unrun. |
| `sec:ablation:c1alt` | **L3-c1alt** selection-policy sweep | fluxtune, selector ∈ {random, cosine, quasi, JVP-guided} | NEW, unwritten | ❌ NEEDS CODE | selector code-existence unverified (§5 row 16). |
| `sec:ablation:jvp` | **L2-jvp** JVP threshold/refresh sensitivity | fluxtune, param sweep | NEW, unwritten | ❌ NEEDS CODE CHECK | refresh-frequency knob unverified (§5 row 17). |
| `sec:ablation:kc` | **L1L2-kc** static vs dynamic K/C | fluxtune, `dynamic_kc` on/off + sweep | NEW, unwritten | ❌ NEEDS INSTRUMENTATION | flag exists (Opt-4) but no telemetry for K/C value *over time* found (§5 row 18) — likely needs new instrumentation before config work starts. |
| `sec:ablation:c3` | **L1L3-c3** gradient-aware vs naive agg | fluxtune, agg_rate_conf ∈ {naive_avg, felix scalar, grad_aware} | NEW, unwritten | ❌ NEEDS CODE | `naive_avg` mode code-existence unverified (§5 row 16); `grad_aware`/`new` already exist+validated (R1/R3). |
| `sec:ablation:databin` | **L2L3-databin** databin-size sweep | fluxtune, `train_batch_size` sweep | overlaps §4 M1 but reframed (comm/learning tradeoff) | ⚠ partial | reuses validated WS3-b + var telemetry; needs re-scoping to tex's framing, not a telemetry gap. |
| `sec:ablation:noniid` | **L-noniid** α sweep | fwdllm, fwdllm_it_oracular, fluxtune at α∈{10,100} (charter A5, not tex's {0.5,1.0} — paper-side stale, flagged back to operator) | NEW, unwritten | ✅ CODE-READY | pure config sweep (`partition_method`), reuses fully-validated E1 metrics — blocked only on yaml/launch. |
| `sec:eval:setup` (models) | **G1** model generalization | fluxtune (+anchors) on LLaMA2-7B/Mistral-7B | PLACEHOLDER | — deferred | tex `\tbd`, not started. |
| `sec:eval:setup` (datasets) | **G2** dataset generalization | fluxtune (+anchors) on Yahoo!Answers/D3 | PLACEHOLDER | — deferred | tex `\tbd`, not started. |
| _(no tex anchor)_ | **S-databin** bias/variance databin×heterogeneity sweep | fluxtune | PLACEHOLDER (was M1 below) | — deferred | tracked only. |
| _(no tex anchor)_ | **S-optimizer** server-optimizer fix (Issue I-1) | fluxtune | PLACEHOLDER (was M2 below) | — deferred | PAUSED (`fluxtune_contributions.md` §8 S1), tracked only. |

This table is kept in sync with `BRIDGE_DESIGN.md` §2 — every future tex edit gets diffed against both.

---

## 10b. Run ledger (which log feeds which result)

Update as runs land — how we know which log file on which node backs each figure/claim. `Status`: SMOKE
(validation, not for paper) · FINAL (paper number) · STALLED/CONVERGED/DNC (verdict). ⚠ The three N=100 runs
below ran at **α=0.1** (log filenames say `alpha0p1`). **α=0.1 is now excluded** — learning was too slow across
all baselines to complete convergence runs — so the paper's primary condition is **α=1**
(`experiments.yaml main`), and these runs will be **re-run at α=1** for final numbers. Treat the α=0.1 runs as
smoke / evidence-that-0.1-is-too-slow, not paper numbers.

| Run dir | Baseline | Node | Condition | Verdict | Feeds | Notes |
|---|---|---|---|---|---|---|
| `run_20260707_015846_fwdllm_n100_smoke_syn_0_real` | fwdllm | shepherd | N=100, syn_0, α0.1, df=2, agg_goal=10 | SMOKE | E1–E5 (baseline) | log `07_07_26_01_59_random_n100_default_alpha0p1_syn0_*` |
| `run_20260706_185023_fwdllm_plus_n100_smoke_syn_0_real` | fwdllm_plus | kaylee | N=100, syn_0, α0.1, df=2, agg_goal=10 | SMOKE | E1–E5 (baseline) | log `06_07_26_18_50_random_n100_oracular_alpha0p1_syn0_*`; oracular **inert** at syn_0 |
| `run_20260706_185045_fluxtune_n100_smoke_syn_0_real` | fluxtune | shepherd | N=100, syn_0, α0.1, df=2, agg_goal=10, C=30 | STALLED @84.08% (139 bins) | E1–E5 (C1-only, superseded) | log `06_07_26_18_51_…`; C2 off, C3=placeholder. **Superseded by R4 (`…025716…`) for the baseline comparison.** |
| `run_20260708_025543_fluxtune_n100_smoke_syn_0_real` | fluxtune **R1** base | — | N=100, α=1, agg_goal=10, C=30, var-stop=off, agg-rate=new | max 83.00% | ablation (Opt-2/3 off) | 2×2 opt ablation; `figs_ablation.yaml`. **FluxTune-base (C1 guided-JVP + Opt-1), NOT FeLiX.** α=1 (dir mislabeled `alpha0p1`) |
| `run_20260708_025616_fluxtune_n100_smoke_syn_0_real` | fluxtune **R2** +var-stop | — | …var-stop=plateau, agg-rate=new | max 82.25% | ablation (Opt-2 only) | isolates Opt-2 |
| `run_20260708_025636_fluxtune_n100_smoke_syn_0_real` | fluxtune **R3** +grad-aware | — | …var-stop=off, agg-rate=grad_aware | max 83.91% | ablation (Opt-3 only) | isolates Opt-3 |
| `run_20260708_025716_fluxtune_n100_smoke_syn_0_real` | fluxtune **R4** full (=default) | — | …var-stop=plateau, agg-rate=grad_aware | max 84.08% | ablation (full) **+ E1–E5 baseline comparison** | full-stack default; feeds `figs.yaml fluxtune` |
| _pending_ | fluxtune (full-system) | — | C2 dynamic K/C **ON** + real C3 ON | — | E2/E3/E4 efficiency | breaks agg_goal match → separate run-set |
| _pending_ | all three | — | `mobiperf_*` (real-world availability) | — | E1 headline | needs fwdllm_it_oracular-under-scarcity policy |
| _pending_ | fwdllm (or port) | — | fidelity: accuracy vs `xu2024fwdllm` | — | Setup (D3) | locate **old** run data; accuracy parity, not time |
| _pending_ | ablations | — | JVP-sens / K-C-sens / α∈{0.1,0.5} | — | Ablation §§ | tooling TBD (charter §2e) |
| _pending_ | fluxtune | — | stability: bin-size×α sweep (M1) + server-optimizer (M2) | — | I-1 fix / E1 convergence | flag-gated; see `fluxtune_contributions.md` §8 |
