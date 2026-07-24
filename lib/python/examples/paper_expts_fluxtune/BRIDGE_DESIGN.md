# FluxTune paper⇄code bridge — design doc (DRAFT, pre-implementation)

**Purpose of this file.** Not the bridge itself — the plan for building it. We iterate here until
signed off, then this doc's checklist becomes the implementation order. Once landed, its content
folds into `EXPERIMENTS.md` (the living bridge) and this file is deleted.

**Why now.** `evaluation.tex` changed materially since `EXPERIMENTS.md`/`EXPTS_CHARTER.md` were last
touched: new 9-baseline taxonomy, a restructured eval section (`sec:eval:sota` vs
`sec:eval:attribution`), and 7 explicit `plannedexp` ablation blocks with stable tex labels. The docs,
`experiments.yaml`, `_metadata/baselines.yaml`, and `run_sequential.sh` still reflect the old 3-baseline
world. This doc reconciles them into one architecture before we touch code.

**Decisions locked in (asked up front, 2026-07-23):**
1. **Full baseline rebuild now** — wire all 9 tex baselines into `baselines.yaml` / `experiments.yaml` /
   `run_sequential.sh` (not just the 4 EVAL anchors), since ablation experiments need the `+IT`/`+O` rows too.
2. **Registry stays under `fwdllm/`** — `experiments.yaml`, `expt_scripts/`, `figs*.yaml`, `plotlib/` are
   not moved; `paper_expts_fluxtune/` holds the docs/ledger and references them by relative path (as it
   does today).
3. **Sim mode for everything going forward** — all new launches use `--mode sim`. Already-landed
   real-mode numbers (R1–R4 opt-ladder, current fwdllm/fwdllm_plus/fluxtune comparison) are kept as-is,
   not re-run, unless we later decide we want sim-mode re-verification of those specific numbers.
4. **Alpha conflict resolved: charter wins.** `EXPTS_CHARTER.md` A5 (never below α=1, ablate up to
   α∈{10,100}) is the standing decision. `evaluation.tex`'s `sec:ablation:noniid` block (currently
   α∈{0.5,1.0}) is paper-side stale and needs a `\tbd`-style fix flag back to the operator, not a
   code-side change.
5. **No duplication — reference, don't copy.** Anywhere the bridge needs baseline knobs, run scripts, or
   plotting code, it references `fwdllm/`/`_metadata/` by path; it never re-declares the same facts.
   Concretely: `experiments.yaml`'s local `baselines:` stanza is **deleted**, run-sets reference baseline
   *keys* only and resolve their knobs from `_metadata/baselines.yaml` (§3.2).

**Deferred-but-tracked policy** (applies uniformly to items below, not full implementation scope this
pass — placeholder rows only, promoted to real run-sets later):
- **G1/G2 — model & dataset generalization.** `evaluation.tex`'s own `\tbd`s: G1 = two further model
  families beyond DistilBERT (tex calls them "M2, M3" inline; the preempt block names LLaMA2-7B primary /
  Mistral-7B secondary), G2 = two further datasets (Yahoo! Answers, "D3"). Placeholder rows in Layer 0,
  no run-set yet.
- **S-databin / S-optimizer — `EXPERIMENTS.md`'s stability track** (previously labeled M1/M2 there;
  renamed here to avoid colliding with G1/G2's tex-native "M2/M3" model labels). S-databin = the
  bias/variance databin×heterogeneity sweep, S-optimizer = the server-optimizer fix for Issue I-1
  (`fluxtune_contributions.md` §8 S1, currently PAUSED). Neither has a tex anchor today. Kept as
  placeholder rows, not cut — matches your answer (keep for later) applied consistently with G1/G2.
- **`sec:ablation:c1alt` / `sec:ablation:c3` code-existence gaps** (random/cosine/quasi selectors;
  naive-average aggregation) — do **not** verify now; confirmed a placeholder in Layer 0, existence check
  happens at build-time in checklist step 7.
- **Fidelity run (D3 decision) + mobiperf E1 run** — already `_pending` in today's ledger; carried
  forward unchanged as placeholders, same policy.

---

## 1. Architecture — five layers

```
evaluation.tex (labels)                    ← narrative source of truth for WHAT the paper claims
       │  reconciled by
       ▼
EXPERIMENTS.md §10-ledger (NEW: "tex-map") ← Layer 0: the bridge — tex label ⇄ expt id ⇄ run-set ⇄ status
       │  drives
       ▼
experiments.yaml + _metadata/baselines.yaml ← Layer 1: machine registry — what CAN run
       │  launched by
       ▼
run_sequential.sh (--mode sim)              ← Layer 2: execution — what DID run (+ pre-flight gate)
       │  produces
       ▼
experiments/run_*/{snapshot.yaml,telemetry} ← Layer 3: correctness — self-describing run + manual sign-off
       │  reduced by
       ▼
compare_baselines.py / plot_run.py /        ← Layer 4: plotting — figs*.yaml manifest → PDF → Overleaf
make_paper_figs.py
```

This is the existing anti-redundancy backbone (`EXPERIMENTS.md` §0) — we are not replacing it, we're
adding **Layer 0** (which doesn't exist yet as a *tex-anchored* table; today `EXPERIMENTS.md` organizes
around its own "Experiment 1–5" numbering that predates the tex restructure) and closing the Layer 1 gap
(5 of 9 baselines unwired).

---

## 2. Layer 0 — the tex↔experiment map (the actual "bridge" artifact)

New table, keyed by **tex `\label`** (stable, already exists) → our experiment id → run-set → status.
This replaces free-form cross-referencing with a literal checklist: for every row, can we answer
*yaml exists / run launched / telemetry verified / plot generated / number in tex*?

**`Ready?` column is a launch gate, not a status note** — it rolls up §2b's per-metric instrumentation
ledger. A run-set may not be launched, not even at smoke scale, while any metric it needs is below
`CODE-READY` (§2b defines the levels). This is the answer to "how do we know instrumentation exists
*before* we spend GPU time discovering it doesn't" — the WS3-a bug (async dispatch silently emitted zero
`comm` events until someone noticed) is exactly the failure mode this gate exists to catch earlier.

| tex label | Experiment id | Baselines needed | Run-set (`experiments.yaml`) | `Ready?` | Status today |
|---|---|---|---|---|---|
| `sec:eval:tta` | **E1** time-to-accuracy | fwdllm, fedbuff_round, felix_round, fluxtune | `main_v2` (NEW — 4-anchor) | ⚠ NOT READY | fwdllm/fluxtune landed (real); fedbuff_round/felix_round **unrun, baseline unwired**; reducer gap N3 **fixed** (§2b row 1), still blocked on the unrun baselines |
| `sec:eval:util` | **E2** resource utilization | same as E1 | reuses `main_v2` | ⚠ NOT READY | reducer gap N5 **fixed** (§2b row 3); still blocked on the unrun baselines |
| `sec:eval:compute` | **E3** compute effectiveness | same as E1 | reuses `main_v2` | ⚠ NOT READY (needs new-baseline validation) | reducer exists; numbers stale (old 3-baseline run); WS3-b needs re-validation on fedbuff_round/felix_round (§2b row 8) |
| `sec:eval:comm` | **E4** communication | same as E1 | reuses `main_v2`, needs WS3-a | ⚠ NOT READY (needs new-baseline validation) | reducer exists; numbers stale; WS3-a needs re-validation per baseline (§2b row 9-10, per-baseline checklist) |
| `sec:eval:sessions` | **E5** session length | same as E1 | reuses `main_v2` | ⚠ NOT READY | reducer gap N6 **fixed** (§2b row 11-12) — sync now uses one-round-span (`round_span_durs`), not `contributor_intervals`; still blocked on the unrun baselines |
| `sec:eval:attribution` | **A0** attribution (reuses E1–E4 metrics) | fluxtune vs `fwdllm_it_oracular` (rename of `fwdllm_plus`) | reuses `main_v2` once renamed | ⚠ blocked on E1-E4 | **this is exactly the old `fwdllm_plus` comparison, renamed** — closest to already-landed, same reducer gaps apply |
| `sec:ablation:ladder` | **L-ladder** opt on/off ladder | fluxtune only, 4 configs (R1–R4) | `opt_ladder` (existing, informal) | ✅ landed / ⬜ L1-completion unrun | ✅ **landed** (`EXPTS_CHARTER.md` R1–R4, `commit_reason`+`grad_aware_gated_total` telemetry validated); `plannedexp` "L1 completion" (per-contribution C1/C2/C3 on/off) is a **separate, unrun** block, needs its own readiness pass |
| `sec:ablation:c1alt` | **L3-c1alt** selection-policy sweep | fluxtune, selector ∈ {random, cosine, quasi, JVP-guided} | NEW run-set | ❌ NEEDS CODE | `quasi`/`random`/`cosine` selectors: code-existence unverified (deferred to build-time, §2b row 13) |
| `sec:ablation:jvp` | **L2-jvp** JVP threshold/refresh sensitivity | fluxtune, param sweep | NEW run-set | ❌ NEEDS CODE CHECK | threshold likely a config knob already; refresh-frequency knob unverified (§2b row 14) |
| `sec:ablation:kc` | **L1L2-kc** static vs dynamic K/C | fluxtune, `dynamic_kc` on/off + N/factor sweep | NEW run-set, uses `fluxtune_dynkc` flag (parked, wired) | ❌ NEEDS INSTRUMENTATION | flag exists (`Opt-4`) but **no telemetry for K/C value over time found** — the sensitivity analysis needs per-round/iteration K,C logged, not just the static config value (§2b row 15, suspected gap) |
| `sec:ablation:c3` | **L1L3-c3** gradient-aware vs naive agg | fluxtune, agg_rate_conf ∈ {naive_avg (NEW), felix scalar (`new`), grad_aware} | NEW run-set | ❌ NEEDS CODE | `naive_avg` aggregation mode: code-existence unverified (§2b row 16); `new`/`grad_aware` already exist+validated (R1/R3) |
| `sec:ablation:databin` | **L2L3-databin** databin-size sweep | fluxtune, `train_batch_size` sweep | overlaps `EXPERIMENTS.md` M1 but **reframed** (comm/learning tradeoff, not bias-variance) — reconcile or split | ⚠ partial | reuses WS3-b (validated) + var telemetry (validated); needs re-scoping to tex's comm/learning framing, not a telemetry gap |
| `sec:ablation:noniid` | **L-noniid** α sweep | fwdllm, fwdllm_it_oracular, fluxtune at α∈{10,100} (per charter, not tex's {0.5,1.0}) | NEW run-set | ✅ CODE-READY | pure config sweep (`partition_method`), reuses fully-validated E1 metrics; blocked only on yaml/launch, not instrumentation |
| `sec:eval:setup` (models paragraph) | **G1** model generalization | fluxtune (+ anchors later) on LLaMA2-7B / Mistral-7B | PLACEHOLDER, no run-set | — deferred | deferred — tex `\tbd`, not started |
| `sec:eval:setup` (models paragraph) | **G2** dataset generalization | fluxtune (+ anchors later) on Yahoo! Answers / D3 | PLACEHOLDER, no run-set | — deferred | deferred — tex `\tbd`, not started |
| _(no tex anchor)_ | **S-databin** bias/variance databin×heterogeneity sweep | fluxtune | PLACEHOLDER — was `EXPERIMENTS.md` M1 | — deferred | deferred, tracked only |
| _(no tex anchor)_ | **S-optimizer** server-optimizer fix (Issue I-1) | fluxtune | PLACEHOLDER — was `EXPERIMENTS.md` M2 | — deferred | PAUSED (`fluxtune_contributions.md` §8 S1), tracked only |

This table is the thing we keep in sync going forward — every future tex edit gets diffed against it.
Placeholder rows (G1/G2, S-databin/S-optimizer) stay in the table so nothing silently falls off the map,
but carry no run-set until explicitly promoted.

---

## 2b. Instrumentation readiness ledger — the required-fields gate

**Rule: no run-set launches (not even smoke) until every metric its mapped experiments need is at least
`CODE-READY` below.** `VALIDATED` is achieved *by* the smoke run itself (that's what smoke runs are for),
so it can't be a pre-launch requirement — but `CODE-READY` (the emit call exists, on the code path this
experiment will actually exercise) must be confirmed by reading the code first. This is the difference
between "the telemetry probably works" (what caused the WS3-a silent-gap bug) and "we checked."

**Readiness levels** (per metric):
- **PLANNED** — needed, not yet designed.
- **NEEDS CODE** — the underlying feature/config knob this metric depends on may not exist yet (e.g. a
  selector policy, an aggregation mode). Must resolve before `CODE-READY`.
- **CODE-READY** — the telemetry emit call exists on every code path the experiment's baselines will
  exercise, confirmed by reading the emit site(s), not assumed from a similar baseline.
- **VALIDATED** — a smoke run (N=10, sim) has been inspected and the field is confirmed **non-empty,
  correctly shaped, and firing on every baseline in the run-set** (not just the baseline it was first
  built for). Only VALIDATED metrics may feed a `FINAL`-status ledger row.
- **REDUCER-GAP** — telemetry is VALIDATED but the reducer computing the metric has a known bug
  (references the existing N3/N5/N6 findings, `EXPERIMENTS.md` §4 reducer-audit note) — blocks `Ready?`
  independent of telemetry status.

Extends the existing metric map (`EXPERIMENTS.md` §5) with status columns; numbers 1–13 below are that
table's own row numbers, kept aligned so the two documents don't drift.

| # | Metric | Telemetry event.field | Provenance | Status | Note |
|---|---|---|---|---|---|
| 1 | Experiment wall-time before exit | `converge.json` (WS2) | WS2 | CODE-READY (N3 fixed) | `expt1_time_to_target` now prefers `converge.json` (matched to the run via its `agg_telemetry` field — it lives in `expt_scripts/smoke_logs/`, not the run dir) when its target/window match the request, falling back to the `agg_eval` reconstruction otherwise (`source` field on the output records which). No real `--target-acc` run exists yet to validate against, so still `CODE-READY` not `VALIDATED` — first convergence-watched launch flips it. |
| 2 | Aggregator active GPU time | `agg_round.aggregate_fedavg_s`+`eval_s` | DERIVE | VALIDATED (existing runs) | re-validate on fedbuff_round/felix_round once they exist — different aggregator config, same entrypoint |
| 3 | Each client active GPU time | `trainer_round.gpu_compute_s` | EMIT | CODE-READY (N5 fixed) | `expt2_utilization` now decomposes idle into `net_wait_frac` (from `mqtt_fetch_s`, previously emitted-but-unused) + a true residual `idle_frac`, instead of one undifferentiated `1−busy_frac` bucket. `barrier_wait_s`/`drain_tail_s` deliberately still excluded (agg-side, ≈0 in sim, not a trainer idle signal). Confirmed non-empty/firing on the existing `fwdllm`+`fluxtune` REAL runs; not yet the formal per-baseline SIM smoke validation §2b's `VALIDATED` requires (checklist step 6). |
| 4 | Loss at first iteration | first `agg_eval.test-loss` | EMIT | VALIDATED | time-ordered (data_id-cycling bug already fixed per memory) |
| 5 | Loss after training complete | last `agg_eval.test-loss` | EMIT | VALIDATED | same |
| 6 | Updates used per iteration | `agg_round.agg_goal_count`/`updates_in_queue` | EMIT | VALIDATED, unused | not currently feeding any of E1-E5 directly |
| 7 | Total updates over training | Σ `agg_round.agg_goal_count` | EMIT | VALIDATED, unused | same |
| 8 | Forward passes / perturbations per client | `trainer_round.forward_passes_total`/`perturbations_total` | WS3-b | VALIDATED (fwdllm/fwdllm_plus/fluxtune only) | **must re-validate on fedbuff_round/felix_round** — client-side counter in `fwdgrad_utils.py`, selector-independent in principle, but unconfirmed for the new selectors until smoke-tested |
| 9 | Data sent from aggregator | `comm{direction=agg_to_trainer}.size_bytes` | WS3-a | VALIDATED (sync+async paths, existing 3 baselines) | new async baselines (fedbuff_round, felix_round) reuse fluxtune's already-instrumented async dispatch site — **still smoke-validate per baseline**, don't assume from fluxtune's pass |
| 10 | Data sent from clients | `comm{direction=trainer_to_agg}.size_bytes` | WS3-a | VALIDATED (existing 3 baselines) | same caveat as #9 |
| 11 | Client active: selected→next reselection (async) | `agg_round.contributor_intervals` | EMIT | VALIDATED | — |
| 12 | Client active per round (sync reselect) | `trainer_round`/`agg_round` ts | EMIT | CODE-READY (N6 fixed) | New `round_span_durs`: groups consecutive same-`data_id` `agg_round` cycles (bounded on `is_async=False`) into one per-data_id span, first cycle's ts → commit ts — the actual multi-iteration "one round span" the label already claimed. `expt5_sessions` uses it for `is_async=False`, `session_durs` unchanged for async. On the real `fwdllm` run: old (mis-applied) computation gave 1490 per-cycle samples (p50≈11s); the fix gives 56 round-level samples (p50≈45s) — confirms the old number was undercounting by ~4x, not a cosmetic relabel. |
| 13 | Participation counts (rounds/data_bins/iterations) | `trainer_round` + `contributing_trainers` | EMIT | VALIDATED, `contributing_trainers` unused | reducer only reads a subset |
| 14 | `commit_reason` (natural/cap/plateau) | `agg_round.commit_reason` | EMIT | VALIDATED (R1–R4 opt-ladder) | feeds L-ladder only |
| 15 | `grad_aware_gated_total` | aggregator telemetry | EMIT | VALIDATED (R1–R4) | feeds L-ladder / L1L3-c3 |
| 16 | Selector identity per iteration (random/cosine/quasi/JVP) | *unknown* | NEEDS CODE CHECK | — | needed for L3-c1alt to attribute accuracy-vs-compute to a specific selection policy; unverified whether non-JVP-guided selectors are even implemented |
| 17 | JVP threshold/refresh-frequency config | *config, not telemetry* | NEEDS CODE CHECK | — | feeds L2-jvp; refresh-frequency knob existence unverified |
| 18 | Effective K/C value over time (dynamic_kc) | *suspected missing* | NEEDS CODE | — | feeds L1L2-kc's sensitivity analysis (window N, growth/shrink factor) — the *static* config K/C is logged, but a moving-average controller changing K/C over the run needs its own time series; flagged as a likely net-new instrumentation item, not just a config sweep |
| 19 | `align_floor`/`inverse_var` gate telemetry | aggregator telemetry | EMIT | VALIDATED (R3/R4) | feeds L1L3-c3's grad-aware arm; the `naive_avg` arm (#16-adjacent) still needs the aggregation mode itself built |

**Per-baseline validation checklist** (rows 8–10 above depend on this — "code exists" ≠ "confirmed firing
per baseline"). Fill in during checklist step 5 (smoke tests), one row per newly-wired baseline:

| Baseline | WS3-a `comm` both directions | WS3-b forward-pass/perturbation counters | Notes |
|---|:---:|:---:|---|
| fwdllm | ✅ validated | ✅ validated | existing |
| fwdllm_it_unaware (was fwdllm_plus) | ✅ validated | ✅ validated | existing, rename only |
| fwdllm_it_oracular (was fwdllm_plus) | ✅ validated | ✅ validated | existing, rename only |
| fluxtune | ✅ validated | ✅ validated | existing |
| fedbuff_round | ⬜ smoke-pending | ⬜ smoke-pending | async dispatch site shared w/ fluxtune — expected to pass, don't skip the check |
| fedbuff_it_unaware | ⬜ smoke-pending | ⬜ smoke-pending | — |
| fedbuff_it_oracular | ⬜ smoke-pending | ⬜ smoke-pending | — |
| felix_round | ⬜ smoke-pending | ⬜ smoke-pending | async_oort selector — verify WS3-b counters aren't selector-coupled |
| felix_it | ⬜ smoke-pending | ⬜ smoke-pending | — |

---

## 3. Layer 1 — baseline rebuild (the 9-baseline wiring)

Per decision #1, do the full rebuild BASELINES.md already spec'd (its own "Remaining work" #1/#2), now:

1. **`_metadata/baselines.yaml`**: add 5 new keys — `fwdllm_it_unaware`, `fwdllm_it_oracular` (=
   rename target for `fwdllm_plus`), `fedbuff_round`, `fedbuff_it_unaware`, `fedbuff_it_oracular`,
   `felix_round`, `felix_it` (7 new/renamed, not 5 — BASELINES.md undercounts by not listing the two
   `fwdllm_it_*` renames as "new"). Verify `felix_round`'s placeholder `learning_rate=0.075` before
   trusting any run built on it (BASELINES.md item #5).
2. **`experiments.yaml`**: **delete** the local `baselines:` stanza entirely (it duplicated
   sync/selector/tracking facts that `_metadata/baselines.yaml` already owns). Run-sets reference
   baseline keys only (`baselines: [fwdllm, fedbuff_round, felix_round, fluxtune]`); anything that needs
   a baseline's knobs (the pre-flight gate's tier ② print, `plotlib/baselines.py`) reads
   `_metadata/baselines.yaml` directly. One source of truth, no drift possible between the two files.
3. **`run_sequential.sh`**: extend the baseline/smoke-yaml pair list; add smoke yaml for each new key
   (`fedbuff_round_n10_smoke_sim.yaml` etc., following the existing `_sim` suffix convention already used
   for `fluxtune_n10_smoke_sim.yaml`/`fwdllm_n100_smoke_sim.yaml`).
4. **Tests**: `test_baselines.py`, `test_config_generator.py` — add cases per BASELINES.md's own item #1,
   rename the `fwdllm_plus` cases per item #2.
5. **Code-level rename propagation** (BASELINES.md item #2's cosmetic-comment list) — low priority,
   do last, mechanical.
6. **`plotlib/baselines.py`**: register display names/colors for all 9 (the `(P)`/`+IT`/`+O` suffixes) so
   legends render correctly — BASELINES.md item #4 names this file explicitly as feeding figure legends.

**New run-set in `experiments.yaml`**: `main_v2` — same condition block as today's `main` (N=100, K=10,
C={sync:10,async:30}, α=1, syn_0, delay_factor=2, target 0.84, window 20) but `baselines: [fwdllm,
fedbuff_round, felix_round, fluxtune]` for E1–E5, plus `fwdllm_it_oracular` folded in for the `A0`
attribution reducer (reuses the same run dirs, no new launch). **All launches `--mode sim`** per decision #3.

---

## 4. Layer 3 — correctness gates (the "ensure sanity of runs at all costs" part)

Five checkpoints, three automatable, two deliberately manual:

0. **Instrumentation pre-check (NEW, manual — a code read, not a launch check).** Before a run-set's yaml
   is even written: walk §2b for every metric its mapped experiments need, confirm each is at least
   `CODE-READY` (emit call exists on the code path this run-set's baselines actually exercise). This is
   the step the WS3-a bug skipped — the emit call existed on the *sync* path, was assumed to cover async
   too, and didn't. `NEEDS CODE` / `NEEDS CODE CHECK` rows block yaml authoring entirely, not just launch.
   Not automatable because it requires reading the actual emit site, not pattern-matching a similar
   baseline's telemetry.
1. **Pre-flight (automated, exists today)** — `condition_fp` fingerprint gate in `run_sequential.sh`,
   agg_goal-match check, clean-slate guard. Keep as-is; extend the baseline-table pre-flight print (tier
   ②) to cover the 6 new baselines' knobs so a misconfigured `felix_round` selector is caught before launch,
   same as today's sync/selector/optimizer eyeball check.
2. **Post-run ledger auto-append (NEW, automated)** — today §10 of `EXPERIMENTS.md` says "update as runs
   land," which is manual transcription (error-prone: dir names have already been observed to
   *mislabel* α in this project's own history — `…alpha0p1…` dirs that were actually α=1). Add a small
   script (`expt_scripts/append_ledger_row.py`) that reads `snapshot.yaml` + `converge.json`/`stall.json`
   from a finished run dir and emits a ready-to-paste ledger row (run dir, baseline, condition **read from
   the config, not the dir name**, verdict, git sha). Human still pastes it in — this only removes the
   transcription-error class, it doesn't remove the human.
3. **Telemetry sanity check (NEW, semi-automated)** — before a run dir is allowed to feed a plot, run a
   fixed checklist against its `telemetry/*.jsonl`: expected event types present (`agg_eval`, `comm` if
   WS3-a needed, `trainer_round` counters if WS3-b needed), no obviously-truncated file, `condition_fp`
   in `snapshot.yaml` matches the run-set's expected fingerprint. Fold into `compare_baselines.py`'s
   existing mix-guard rather than a new tool — it already discovers+warns on baseline mismatches; extend
   it to also warn on missing-telemetry-for-required-metric (partially exists per `requires_telemetry`
   field in `experiments.yaml`, generalize it). **This is what flips a §2b metric from `CODE-READY` to
   `VALIDATED`** for that specific run dir — the ledger update in checkpoint 2 should record the flip
   (which metrics validated, which baseline) so §2b's per-baseline checklist gets filled in from real
   runs, not from memory.
4. **Manual sign-off (deliberately NOT automated)** — before a number/plot is marked `FINAL` in the
   ledger (vs `SMOKE`), you eyeball the `summary.json` + the actual PDF. This is the step the user
   explicitly called out as non-automatable; the doc should keep a `Status: SMOKE|FINAL` column exactly
   as today and never auto-promote SMOKE→FINAL.

---

## 5. Layer 4 — plotting (mostly reuse, minor extension)

`figs.yaml`/`figs_ablation.yaml` pattern is good and stays. Add one manifest per new ablation run-set
(`figs_c1alt.yaml`, `figs_jvp.yaml`, `figs_kc.yaml`, `figs_c3.yaml`, `figs_databin.yaml`,
`figs_noniid.yaml`) — each a flat `baseline_key: run_dir` map like today's, consumed by
`make_paper_figs.py --manifest <file>`. No new plotting code needed until we know which of E1–E5's plot
types each ablation actually wants (e.g. `sec:ablation:ladder` wants peak-accuracy-vs-config, not the
full 5-plot set) — that's a `plotlib/figures.py` scoping question to answer per-ablation during
implementation, not now.

---

## 6. Implementation checklist (for sign-off)

Ordered so each step is independently testable before the next depends on it.

- [x] **1.** Rebuild `_metadata/baselines.yaml` — add/rename the 7 baseline entries (§3.1). Verify
      `felix_round`'s LR against a smoke run before trusting it downstream. **DONE** — see "Handoff status"
      below for exact state; LR verification still pending (needs a real smoke run).
- [x] **2.** Propagate rename into `experiments.yaml`, `run_sequential.sh`, tests, `plotlib/baselines.py`
      (§3.2–3.6). **DONE** — `run_sequential.sh` heredoc validated standalone (sync + 2 new async
      baselines, exit 0, correct `is_async` resolution); `test_config_generator.py` rename fixed (29/29
      pass); `plotlib/baselines.py` registers all 9 baselines with `(P)`/`+IT`/`+O` display names +
      2 new CVD-safe family hues (fedbuff=green, felix=pink); `experiments.yaml`'s local `baselines:`
      stanza deleted, run-sets reference keys only. Full `-k fwdllm` suite: 381 passed. See "Handoff
      status" below for what's still deferred (BASELINES.md item #2's cosmetic comment sweep — separate,
      later, mechanical).
- [x] **3.** Instrumentation pre-check for `main_v2` (§4 gate 0): confirm metrics #1–13 in §2b are
      `CODE-READY` for fedbuff_round/felix_round specifically (not just "fluxtune already validated it") —
      read the async dispatch + selector code paths these two new baselines actually run through. Fix
      reducer gaps **N3** (E1 should read `converge.json`, not reconstruct), **N5** (E2 idle-fraction
      reducer), **N6** (E5 sync session should use one-round-span) — these block `Ready?` regardless of
      telemetry status, so fixing them now avoids re-deriving numbers later. **DONE** — see "Handoff
      status" below.
- [ ] **4.** Add `main_v2` run-set to `experiments.yaml` (4-anchor E1–E5 + attribution reuse), sim-mode.
- [ ] **5.** Write the new Layer-0 tex-map + §2b instrumentation ledger into `EXPERIMENTS.md`, superseding
      its old "Experiment 1–5" framing — keep the metric/reducer content (§4, §5 of `EXPERIMENTS.md`
      today), re-anchor to tex labels, keep row numbers 1–13 aligned between the two docs.
- [ ] **6.** Smoke-test each new baseline (`fedbuff_round`, `felix_round`, and the `+IT`/`+O` rows) at
      N=10 sim. Two things happen in this step, not one: (a) confirm `run_parity.py` doesn't need new
      rungs for them (new *configs* of an already-parity-tested code path, but verify); (b) walk the §2b
      per-baseline checklist and flip WS3-a/WS3-b from `smoke-pending` to `validated` **per baseline** by
      actually inspecting each one's `telemetry/*.jsonl` — do not bulk-mark all baselines validated because
      one passed.
- [ ] **7.** Launch `main_v2` at N=100 sim only after step 6's checklist is fully green — this is gate 0
      (§4) enforced in practice, not just on paper.
- [ ] **8.** For each of the 6 net-new ablation blocks (`c1alt`, `jvp`, `kc`, `c3`, `databin`, `noniid`):
      run the §2b code-existence checks first (rows 16–19 — random/cosine/quasi selectors, naive-average
      aggregation, and the suspected **K/C-over-time telemetry gap** for `kc` specifically), design the
      run-set only once `CODE-READY`, smoke it, land it. These are separate follow-up passes, not one shot
      — `kc` in particular may need new instrumentation before any config work starts.
- [ ] **9.** Write `append_ledger_row.py` (§4.2) and wire it into `run_sequential.sh`'s post-run hook —
      have it also stamp which §2b metrics validated for that run, feeding the per-baseline checklist.
- [ ] **10.** Extend `compare_baselines.py` mix-guard for telemetry-completeness (§4.3).

Steps 1–5 are the actual "bridge" skeleton; step 6 is where instrumentation claims get tested against
reality for the first time. I'd suggest landing 1–7 (through the `main_v2` launch) before starting step 8's
ablation work, since `kc`'s suspected telemetry gap and `c1alt`/`c3`'s code-existence questions could each
turn into small feature work, not just config authoring.

---

## Status

All open questions from the previous two rounds are resolved (baseline rebuild scope, registry location,
sim mode, alpha conflict, M1/M2 naming + deferred-placeholder policy, no-duplication rule, and now the
instrumentation-readiness gate: §2b ledger + §4 gate 0 + per-baseline validation checklist).
Implementation started (checklist step 1) and **checklist steps 1–3 are now complete** (rename fully
propagated + validated; instrumentation pre-check done, N3/N5/N6 reducer gaps fixed) — see "Handoff
status" below for exact state. Do not re-derive the round/iteration async design from scratch; it's
settled (see below). Next up: checklist step 4 (add `main_v2` run-set to `experiments.yaml`).

---

## Handoff status (checklist step 3 complete, resume at step 4)

**Session context you need before touching anything:** this implementation surfaced a real gap not
anticipated when this doc was signed off — the `reselect_each_iteration` flag (round vs `+IT` baselines)
only ever worked on the *sync* path; the async dispatch path (`_distribute_weights_async`) had zero
caching and always behaved like `+IT`. Without fixing that, `fedbuff_round`/`felix_round` would have been
byte-identical to their `+IT` siblings. This was found, scoped, and confirmed with the operator (two
`AskUserQuestion` rounds — round-cache design confirmed: pin the async_oort/fedbuff-selected cohort for
the span of `self._round`, only replacing departed members via the existing
`_prune_departed_from_round_cache`; default `True` stays a byte-identical passthrough, zero risk to
already-landed `fluxtune`/`felix` numbers) **and implemented and tested** — this is done, not a to-do.

### Done and verified (tests green)

1. **Async round-cache extension** (`lib/python/flame/mode/horizontal/syncfl/fwdllm_aggregator.py`) —
   new `_select_ends_for_async_respecting_reselect_gate` method, wired into `_distribute_weights_async`.
   Extended `__init__`'s comment. Tests: `TestAsyncReselectGate` (4 new tests) added to
   `lib/python/tests/mode/test_fwdllm_reselection.py`; fixed 2 other test doubles
   (`test_fwdllm_dispatch_queue.py`, `test_fwdllm_distribute_vclock_stamp.py`) that needed the new
   method/attribute wired into their hand-rolled aggregator fakes. **Full `-k fwdllm` suite: 340 passed.**
2. **`_metadata/baselines.yaml` rebuilt** — 18 baseline keys now present (was 12; `fwdllm_plus` retired).
   Added `fwdllm_it_unaware` (new), renamed `fwdllm_plus`→`fwdllm_it_oracular`, added
   `fedbuff_round`/`fedbuff_it_unaware`/`fedbuff_it_oracular`/`felix_round`/`felix_it` (all new). Verified
   `python3 -c "import yaml; ..."` parses and lists all 18 keys correctly.
   - **Known unverified placeholders left in the yaml (flagged inline with `TODO(verify)`)**: `felix_round`/
     `felix_it`'s `learning_rate: 0.075` (ported from fluxtune, needs its own smoke-run retune — this is
     literally checklist step 1's own explicit caveat); `fedbuff_round`/`fedbuff_it_*`'s same LR placeholder
     (same reasoning, no dataset_name-table entry for agnews); `fedbuff_it_oracular`'s `trace: mobiperf_2st`
     (2-tier, ported from `fwdllm_it_oracular` — unconfirmed whether the async eval-dispatch path needs a
     3-tier trace instead).
3. **`test_baselines.py`** (`lib/python/tests/launch/test_baselines.py`) updated: renamed/added cases in
   `TestFwdllmBaselines`, new `TestNewAsyncFwdllmBaselines` class (7 tests) covering the 5 net-new async
   baselines' invariants (selector sort, `is_async`, `reselect_each_iteration`, `agg_rate_conf.type`, no
   accidental C1/C3 leakage). **All 31 tests in the file pass.**
4. **Smoke yaml files** — renamed (via `git mv`) `fwdllm_plus_n100_smoke{,_sim}.yaml` →
   `fwdllm_it_oracular_n100_smoke{,_sim}.yaml` (content + internal name/baseline/job-id fields updated).
   Created 6 new sim-only yamls (BRIDGE_DESIGN.md decision #3: sim-only for new launches):
   `fwdllm_it_unaware_n100_smoke_sim.yaml`, `fedbuff_round_n10_smoke_sim.yaml`,
   `fedbuff_it_unaware_n10_smoke_sim.yaml`, `fedbuff_it_oracular_n10_smoke_sim.yaml`,
   `felix_round_n10_smoke_sim.yaml`, `felix_it_n10_smoke_sim.yaml`. All 8 files verified to parse and
   resolve to the correct `baseline:` key via a `yaml.safe_load` check. These yamls deliberately do NOT
   copy fluxtune's own profiled sim-fidelity numbers (`sim_gate_compute_cap_s`, `sim_completion_leg_s`) —
   those were measured for fluxtune's specific JVP-guided compute/payload cost, not these baselines'; left
   at sim defaults (inert/off) with a `TODO(verify)` comment pointing at the still-needed profiling pass.

### Done and verified this session (all of checklist step 2)

5. **`run_sequential.sh`** — `bash -n` syntax-checked AND the embedded Python heredoc (the `python -
   <<'PY' ... PY` block, ~line 320) exercised standalone (extracted, fed a representative `RUN_TSV` of
   one sync + two new async baselines — `fwdllm`, `fedbuff_round`, `felix_it`). Confirmed: `--mode both`
   correctly blocks on the expected missing-real-yaml checks for the 6 sim-only baselines (exit 2, by
   design); `--mode sim` generates all 3 cfgs cleanly (exit 0), `is_async` resolves correctly per baseline
   (fwdllm=sync, fedbuff_round/felix_it=async) in the tier-② table, and `patch()`/`_BL_INTERNALS` don't
   throw. Prior content (header comment, `ALL_RUNS`, `BASELINE_DELAY_DEFAULTS`, the `is_async` hardcode
   fix, the K-D20 comment fix) unchanged from earlier in the session — see git diff for full detail.
6. **`experiments.yaml`** — local `baselines:` stanza deleted (decision #5; confirmed nothing in the repo
   programmatically read it). `main` run-set's `baselines: [fwdllm, fwdllm_plus, fluxtune]` renamed to
   `[fwdllm, fwdllm_it_oracular, fluxtune]` (plain rename of the same config, not a new baseline — kept
   consistent with the rest of the rename rather than left stale). Cosmetic `fwdllm_plus` mentions in the
   file header comment, the `1_time_to_target` takeaway string, and the `scale_n100` `_blocked_pending`
   tag updated too. Parses clean (`yaml.safe_load` verified).
7. **`plotlib/baselines.py`** — all 9 baselines now registered per `BASELINES.md`'s naming grammar
   (`(P)`/`+IT`/`+O`): FwdLLM family (orange, unchanged) gets `fwdllm_it_unaware`/`fwdllm_it_oracular`
   rows; two new CVD-safe family hues added — FedBuff(P) (green, `#009E73`/`#66C2A5`) and Felix(P) (pink,
   `#CC79A7`/`#E8A5C7`) — for the 5 new baselines. Legend order (0–8) follows the paper's sync→async→ours
   staircase narrative; ✅ EVAL rows are bold, ⚠ ABLATION `+IT`/`+IT+O` rows are italic (same convention
   the old `fwdllm_plus` row used). `style_for()`'s unknown-key fallback extended to recognize
   `fedbuff_`/`felix_` prefixes (previously only `fwdllm_`/`fluxtune_`, which would have mis-bucketed any
   future ablation of the new families into the fluxtune-blue fallback). Verified by direct import +
   `style_for()` calls on all 9 keys.
8. **`test_config_generator.py`** — `fwdllm_plus` renamed to `fwdllm_it_oracular` in
   `TestFwdllmEndToEndConfigGeneration`'s two parametrize lists and `TestFwdllmSmokeYamlsResolve`'s
   yaml-filename/baseline-key pair (`fwdllm_plus_n100_smoke.yaml` → `fwdllm_it_oracular_n100_smoke.yaml`),
   plus the class docstring. All 29 tests in the file pass.
9. **Full affected suite re-run**: `pytest -k fwdllm` (excluding an unrelated pre-existing collection
   error in `examples/async_cifar10/launch/test_monitor.py` — a `ModuleNotFoundError` on a `launch`
   package that predates this session, from commit `9503cad4`, orthogonal to fwdllm) — **381 passed**
   (up from the 340 noted earlier in the session, reflecting the tests added since). No regressions.

### Done and verified this session (checklist step 3)

10. **CODE-READY confirmation for fedbuff_round/felix_round** (§4 gate 0, read-only): traced the actual
    code paths these two new async baselines exercise, not assumed from fluxtune's already-validated
    pass. WS3-a comm (`build_comm` in both `_distribute_weights_sync`/`_distribute_weights_async`,
    `flame/mode/horizontal/syncfl/fwdllm_aggregator.py`), WS3-b forward/perturbation counters
    (`FedSgdTrainer.py`), and `contributor_intervals` (`_process_aggregation_goal_met`) are all emitted
    from selector-agnostic code shared by every baseline — none of it branches on which selector
    (`random`/`fedbuff`/`async_oort`) is active. §2b rows 1–13 confirmed `CODE-READY` (not re-derived
    from "a similar baseline" — read directly).
11. **N3 fixed** (`plotlib/reducers.py`, `compare_baselines.py`): `expt1_time_to_target` now prefers
    `converge_watch.py`'s own `converge.json` verdict over the `agg_eval` reconstruction, when their
    target/window match the request. New `RunResult.converge_json` + `_find_converge_json()` — matches
    by `agg_telemetry` path (converge.json lives in `expt_scripts/smoke_logs/`, not the run dir, so
    name-based discovery would be fragile; the payload's own `agg_telemetry` field names the run
    unambiguously). Output now carries a `source` field (`"converge.json"` vs `"reconstructed"`) so a
    future audit can tell which path produced a given number. No real `--target-acc` run exists yet to
    validate end-to-end (no `converge_*.json` anywhere in the repo) — unit-tested only (match, mismatch,
    absent cases); flips to `VALIDATED` on the first convergence-watched launch.
12. **N5 fixed**: `expt2_utilization` decomposes idle into `net_wait_frac` (new, from `mqtt_fetch_s` —
    emitted per `trainer_round` already, previously read by nothing) + a true residual `idle_frac =
    1 - busy - net_wait`, instead of the old single `1-busy_frac` bucket. `barrier_wait_s`/`drain_tail_s`
    deliberately still excluded (aggregator-side, ≈0 in sim, not a trainer idle signal — the gap note's
    own diagnosis). Confirmed non-empty and firing on both existing REAL runs (`fwdllm`, `fluxtune`);
    not yet the formal per-baseline SIM smoke validation §2b's `VALIDATED` requires (checklist step 6).
13. **N6 fixed**: new `RunResult.round_span_durs` — groups consecutive same-`data_id` `agg_round` cycles
    (`is_async=False` only) into one per-data_id span (first cycle's ts → commit ts), the actual
    multi-iteration "one round span" the `session_def` label already claimed but `session_durs`
    (per-cycle `contributor_intervals`, correct for async, wrong for sync) never computed. `is_async=False`
    baselines now report `round_span_durs` in `expt5_sessions`; async is untouched (`session_durs`,
    already validated). **Confirmed materially different on the real `fwdllm` run** — old (mis-applied)
    per-cycle computation: 1490 samples, p50≈11s; fixed per-round computation: 56 samples, p50≈45s (~4x)
    — this was a real undercounting bug, not a cosmetic relabel.
14. **Bonus fix, found while wiring N6**: `compare_baselines.py`'s `main()` had `is_async = (b ==
    "fluxtune")` — the exact same hardcoded-name bug class already fixed in `run_sequential.sh` this
    session, silently mislabeling every new async baseline (`fedbuff_round/it_*`, `felix_round/it`) as
    sync. Replaced with `_is_async(baseline)`, which reads `selector.kwargs.is_async` from
    `_metadata/baselines.yaml` (warns on an unresolvable/unknown key rather than silently defaulting).
15. **Tests**: two new colocated test files (repo convention — see `async_cifar10/scripts/parity/
    test_*.py`): `plotlib/test_reducers.py` (8 tests: round-span grouping incl. edge cases, converge.json
    discovery/matching) and `test_compare_baselines.py` (8 tests: `_is_async` resolution, N3 source
    selection, N5 decomposition, N6 baseline switch). All 16 pass. Also re-ran the full `-k fwdllm` suite
    (381 passed, unchanged) and re-imported `plot_run.py`/`make_paper_figs.py` to confirm the new
    `RunResult` fields don't break their (untouched) call sites.

### Deliberately deferred (separate, later checklist step — NOT step 2 or 3)

BASELINES.md item #2's **cosmetic-comment sweep** (mentions of `fwdllm_plus` in code comments across
`flame/config.py`, `flame/launch/runner.py`, `flame/mode/horizontal/syncfl/fwdllm_aggregator.py`,
`flame/selector/random.py`) and item #4's **doc sweep** (`EXPERIMENTS.md`, `EXPTS_CHARTER.md`,
`simulate_fwdllm.md`, `fluxtune_contributions.md`, `MIGRATION_TO_LAUNCHER_FWDLLM.md`,
`compare_baselines.py`, `logical_parity.py`, `run_parity.py`, `telemetry_manifest.yaml`, `figs.yaml`) are
this doc's own checklist **step 5** ("low priority, do last, mechanical") — a distinct, later step from
checklist step 2 (rename propagation into the *registry/test* files), not part of what this session
resumed. Old `run_*` experiment directories under `experiments/` also still carry the old
`fwdllm_plus_*` name in their dirnames/snapshots — those are historical run artifacts, not renamed
retroactively (matches decision #3's "already-landed numbers kept as-is" policy).

### Git state (uncommitted, nothing pushed)

Checklist steps 1–2 were committed earlier this session (`7062d35e`, `e4d1ff3d`). Checklist step 3's
work is done but **not yet committed**. `git status --short`:
```
 M lib/python/examples/fwdllm/expt_scripts/compare_baselines.py
 M lib/python/examples/fwdllm/expt_scripts/plotlib/reducers.py
 M lib/python/examples/paper_expts_fluxtune/BRIDGE_DESIGN.md
?? lib/python/examples/fwdllm/expt_scripts/plotlib/test_reducers.py
?? lib/python/examples/fwdllm/expt_scripts/test_compare_baselines.py
```
Per CLAUDE.md, ask the user before committing/pushing — checklist step 3 (code-only, no yaml/config
changes, 16 new tests + full `-k fwdllm` suite green) is a natural commit boundary before starting step 4
(a config change: adding the `main_v2` run-set to `experiments.yaml`).

### Next actions, in order

1. Ask the user whether to commit checklist step 3 now.
2. Checklist step 4: add the `main_v2` run-set to `experiments.yaml` (4-anchor E1–E5 + attribution reuse,
   sim-mode only, per §3's spec).
3. Checklist step 5: write the new Layer-0 tex-map + §2b instrumentation ledger into `EXPERIMENTS.md`,
   superseding its old "Experiment 1–5" framing.
