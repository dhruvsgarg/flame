# Real / Sim Parity — Methodical Causal Ladder

Living doc for the async_cifar10 parity checker. Kept in sync with
`scripts/parity/checks.py` (checks), `scripts/parity/report.py` (stage grouping +
verdict), and the pytest suite. **ONE `## Status` section, updated in place.**

**Comparator — give two run dirs, get a report JSON:**
```bash
cd lib/python/examples/async_cifar10
PYTHONIOENCODING=utf-8 python scripts/parity_check.py \
  --real experiments/<real_run_dir> --sim experiments/<sim_run_dir> \
  --agg-goal 10 --budget-s <runtime_s> --json-out experiments/parity_<baseline>_<tag>.json
# batch: --batch --experiments-dir experiments --baselines felix oort refl feddance
```
`--budget-s` = the run's `--runtime-s`; `--lenient` demotes DIST fails to warnings.
Per-run plots: `python ../../../scripts/analysis/analyze_run.py <run>/telemetry`.

**Launch runs** (node-agnostic): `bash scripts/debug_run.sh --baselines 'oort refl'
--runtime-s 3600 --mode both`. Reads `expt_scripts_2026/felix_oort_refl_feddance_alpha0.1_parity.yaml`
(each baseline × sim/real), seeds real+sim identically (`SEED=1234`), applies per-baseline
sim fixes. Split across machines via `--baselines`.

**Tests** (no cluster; under lib/python with `dg_flame`): `pytest tests/mode/
tests/selector/test_oort_selector.py tests/sim/ examples/async_cifar10/scripts/parity/`.
Last green (Jun 22): mode+selector+telemetry **300 pass / 7 skip**, sim **32 pass**.

---

## Workflow policy: minimize time & runs to parity

1. **Run real only when the real path changes.** Sim-only changes validate against the
   stored real dir. A real-path fix → re-run both (or real-only if sim code is unchanged).
2. **Over-instrument telemetry deliberately** — cheap to log, expensive to re-run for.
   (per-round `inflight_residence`, `SIM_CLOCK_DIAG` past-dating counters, per-commit
   `commit_gap` localized roots from stored runs.)
3. **Root-cause per baseline, scope the fix to its blast radius.** Common cause → fix once.
   A fix that could perturb another baseline → serialize (one baseline per run round).
4. **Shortest run that exhibits the issue** (table below). Reserve long runs for C1/C2.
5. **Crisp comments (≤1 sentence); context-free names** (Naming discipline below).

### Run-length budget (state min duration up front, keyed here; never default to 3–4h)
| validating | min run | why |
|---|---|---|
| telemetry field present / instrument sane | **5–10 min** | a few hundred commits populate any per-commit field |
| one MECHANISM rung (gate hold, `commit_visibility`, `residence`, `selection_detail`) | **45 min** | every mechanism fires; per-commit dists stabilize |
| compounding clock/advance residual (`K2`/`K3b`, past-dating) | **90 min – 2 h** | round-count-compounding drift needs the rounds |
| low-frequency eligibility/round-count drift (refl `K2`) | **3 h** | only surfaced at 3h despite a clean 45min pass |
| `C1`/`C2` convergence sign-off | **full (3–4 h+)** | terminal-state + curve parity only |

Smoke (5 min) before any multi-hour run. One mechanism per run round when a fix could
perturb another baseline.

---

## Status (Jun 22 — felix + oort CLOSED 46/46; refl 43/46 + feddance 42/44, both ~1 residual)

**Scoreboard (latest per baseline):**

| baseline | score | state / root | run dirs |
|---|---|---|---|
| **felix** | **46/46** ✅ | mechanism parity closed (§3.drain + §3.resid); only C1/C2 sign-off (≥7200s) left | `…002022…real` / `…154600…sim` |
| **oort** | **46/46** ✅ (90min, VALIDATED Jun 22) | WALL_SEND_TS fix confirmed: A2c `selected_KS=0.024`, K3b residual −0.27s (rel 0.039), Sd 0.581/0.734, S2 KS 0.042. Only C1/C2 LOWC (budget 5400<7200) left. | `…152051…real` / `…152203…sim` |
| **refl** | **43/46** (1.5h Jun 22) ⬆ from 38/44@1h | shared oort `WALL_SEND_TS`/stale-props fix landed CLEAN — K2/K3b/A2c/Sr/Sd/Sx all PASS. Sole FAIL = **A2 num_eligible** KS=0.384: a residence-release-timing gap (sim releases §4.5 pool-hold at `vclock≥sct`, real at commit → sim holds ~3 fewer in-flight ⇒ ~3 more eligible). Means within 1.2%; near-threshold. | `…182200…real` / `…175808…sim` |
| **feddance** | **42/44** (1.5h Jun 22) | old `feddance_U`/`selection_bias` root **CLOSED** (inherited `WALL_SEND_TS` duration fix: A2c, Sx `feddance_U`=0.154, S2 all PASS). Residual: **U6 commit_visibility** (sync-barrier OBSERVABILITY artifact, not a bug) + **K2** marginal (0.052 vs 0.05). | `…180730…real` / `…175847…sim` |

JSONs kept: `parity_felix_20260621_resid`, `parity_oort_20260622_wallsend`,
`parity_refl_20260622_1p5h`, `parity_feddance_20260622_1p5h`.

### Next steps
1. **feddance U6 — TRUE FIX LANDED (real path, §6.u6), needs a confirming real rerun.** The U6
   divergence (sim 15.5s vs real 0.02s) was a REAL telemetry flaw, not a sim bug. Root: real
   `_update_visibility_lag` evaluated `committed = datetime.now()` **per-message inside the recv
   loop**, so it measured arrival→ingestion (~0.02s); the strict barrier actually applies all K
   at ONE post-loop instant, so an early finisher's true lag = barrier − own completion. Updates
   physically arrive SPREAD (`[MSG_ARRIVAL]` 33→48s, `queue_depth=0`) — real had the spread, the
   metric just didn't see it. **Fix:** real now anchors on the single round barrier:
   `lag_i = max_dur − dur_i`, `dur = WALL_SEND_TS − dispatch` (client task-train duration; sim
   stays `vclock−sct`, already the barrier). [syncfl/top_aggregator.py
   `_barrier_anchored_lags`](../../flame/mode/horizontal/syncfl/top_aggregator.py). **Validated
   against STORED real logs (no rerun):** recomputed real lag = mean 15.65 / min 0 / p50 16 /
   max 53 ≈ sim 15.48/0/16/52. oort/refl UNAFFECTED (streaming aggregator, own per-message path
   stays — see "why only feddance" in §6.u6). Guard `test_sync_sim_ordering.py::
   test_barrier_anchored_lags_*`. **Rerun real feddance** (sim dir reusable — sim path
   untouched) to repopulate telemetry and confirm U6 PASS → 43/44.
   **K2** (0.052 vs 0.05, the 44th) is an EMERGENT rollup whose every child rung PASSES (K3
   grid_KS 0.125, K3a, K3b rel 0.056, K4 0.005). It decomposes to real selecting marginally
   SLOWER trainers/round; the largest selector-term gap `feddance_I` (real 51.7 vs sim 59.7) is
   the Oort **stat_utility (training loss)** — an EMERGENT of each trainer's divergent stochastic
   trajectory, NOT a clean computational asymmetry like `WALL_SEND_TS`. With only 176 rounds the
   5.2% gap is ~1.4σ. **No clean code lever** (do NOT add a scalar overhead — Dead ends; do NOT
   chase the stat_utility mix). Resolution = the 3h rerun: more rounds tighten the estimate.
2. **refl A2 num_eligible — DECOMPOSED to `selected_ends` residence SHAPE; instrumentation
   landed, run pending.** The `[DISTRIBUTE]` log already decomposes eligible: `unavail=0` both
   modes (all-UNKNOWN trace), so eligible = `300 − |selected_ends|` and the **entire gap is
   `selected_ends`** (sim 47.2 vs real 50.1 = the 252.7/249.8 gap exactly). NOT the §4.5 buffer
   hold — that feeds the *scorer's* pool (A2b), not this count; the `pending_after`→`pending_ends`
   lever was FALSIFIED (Dead ends). By Little's law `|selected_ends| = num_chosen(13) × residence`,
   so it's purely residence: sim 3.63 vs real 3.86 rounds. Mining stored `inflight_residence`
   telemetry: the residence **tail (≥4) is IDENTICAL (0.449/0.450)** — the gap is the **body
   SHAPE**: real has a sharp mode at residence=3 (19.2%), sim is flatter (peak at 1). Real commits
   with a ~3-round pipeline cadence; sim's modeled commit timing is more spread. **Landed
   instrumentation** (telemetry-only, no dynamics change): `inflight_residence` now emits
   `residence_staleness` + `residence_was_fresh` **paired 1:1** with `residence_rounds`
   ([events.py](../../flame/telemetry/events.py),
   [oort/top_aggregator.py](../../flame/mode/horizontal/oort/top_aggregator.py)), to split the
   shape gap by commit class. Guard `test_parity_checks.py::test_builder_paired_commit_class`.
   **Run:** rerun refl real+sim, then per residence bucket compare sim/real `(staleness,
   fresh-frac)` — does sim under-hold the fresh-committed body (the residence=3 mode)? Near-
   threshold (means 1.2%, rest clean 43/46); confirm a *systematic* class skew before any fix —
   may be real-pipeline cadence (no sim fix).
3. **felix / oort.** One ≥7200s sim-only run each vs stored real to promote C1/C2 LOWC→PASS
   (mechanism already closed for both).

### 3h rerun (feddance + refl, real+sim) — what to EXPECT
`bash scripts/debug_run.sh --baselines 'feddance refl' --runtime-s 10800 --mode both`. 3h=10800s
> 7200 ⇒ C1/C2 become ENFORCED (no longer LOWC) for both.
- **feddance — expect a real FIX + cleanup.** **U6 PASSES** (the §6.u6 barrier-anchor is landed
  code; real telemetry now carries the corrected lag, validated ≈ sim). C1/C2 enforce and should
  PASS (1.5h: acc 0.0163≪0.05, loss 0.0912<0.15). **K2 is the swing:** more rounds tighten the
  ~1.4σ estimate — likely PASS if it was noise, may stay ~5% if the stat_utility mix bias is
  systematic. Best case 44/44 (+ C1/C2). It is NOT guaranteed to "go away" — it's borderline.
- **refl — DIAGNOSTIC run, NOT a fix. Expect A2 to STILL FAIL.** We added only telemetry
  (`residence_staleness`/`residence_was_fresh`), no dynamics change — so A2 num_eligible KS and
  its downstream (S2, C2) persist. The run's PURPOSE is to populate the paired residence telemetry
  (stored runs lack it) so we can bucket residence by commit class and decide the fix (or confirm
  no-fix cadence). Also at 3h watch refl's low-frequency `K2` (the prior 3h run newly failed it,
  rel .075 — round-count-compounding); C2 may fail as A2-downstream. **Do not read a refl A2/C2
  fail at 3h as a regression** — no fix shipped yet. After: analyze the residence-class split,
  THEN decide whether to code a fix.

### Roadmap: lock mechanism parity on ALL baselines BEFORE the perf pass (Jun 21)
Get 46/46 on oort+refl+feddance first, then the sim perf pass — do not interleave. The
checker reads over-instrumented per-commit telemetry; stripping/gating it before A2c roots
close removes the instrumentation needed to root-cause them. A perf change can perturb all
four baselines at once, so it must land on a clean fully-parity base. **Perf pass levers
(deferred):** flag-gate diagnostic logging behind a `simDiag` switch (default OFF), trim
per-commit JSONL volume; guard rail — every perf commit re-runs the 90-min parity (all
baselines) and must hold 46/46.

### Settled roots
| baseline | root |
|---|---|
| **felix** | **ALL MECHANISM ROOTS CLOSED (Jun 21, 46/46).** Eval-stale-`sct` ✅; commit-side ingestion (`recv_fifo` stranding) ✅ `simSctOrderedDrain` (§3.drain); overlapping re-dispatch ✅ `_sim_hold_busy_slots`/`simInflightResidence` (§3.resid). Speed model exonerated. Open: C1/C2 (≥7200s). |
| **oort** | **ALL MECHANISM ROOTS CLOSED (Jun 22, 46/46).** **(1) Stale-property recording** — real dropped a stale-returning trainer's speed/utility (`continue` before `_handle_weights_msg`) → Oort treated slow trainers as unexplored and re-picked forever; fix `_record_returned_trainer_props`. **(2) K3b stale read-wait inflation** — fix (1) recorded stale durations as `recv−dispatch`, bundling **aggregator read-wait** (finished straggler sits unread until a later round drains the buffer; up to 1.65×D) — a server artifact, not client speed, inflating slow trainers so real over-avoided them. Fix: real records **client task-train duration = `WALL_SEND_TS − dispatch`** (`_real_client_task_train_duration`, fresh+stale); sim unchanged (already D). **VALIDATED Jun 22 90min:** A2c `selected_KS=0.024` (pool 12.13/12.13 matched, raw observed 17.1/12.1 still diverges = the excluded read-wait, as diagnosed), K3b residual −0.27s, Sd 0.581/0.734, S2 KS 0.042. Open: C1/C2 (≥7200s). |
| **refl** | Shared `oort/top_aggregator` stale-property + `WALL_SEND_TS` fix landed CLEAN (1.5h Jun 22): K2/K3b/A2c/Sr/Sd/Sx all PASS. **Open: A2 num_eligible** (KS 0.384) DECOMPOSED: `[DISTRIBUTE]` shows `unavail=0`, so eligible = `300−|selected_ends|`; the whole gap is `selected_ends` (sim 47.2 vs real 50.1), which by Little's law = residence (3.63 vs 3.86, num_chosen=13 matches). Residence **tail (≥4) identical**; gap is body SHAPE (real mode at 3, sim flatter = real's ~3-round pipeline cadence). NOT §4.5 buffer (FALSIFIED). Instrumented `residence_staleness`/`residence_was_fresh` paired w/ `residence_rounds` to split by commit class; rerun pending. Near-threshold (means 1.2%); may be no-fix cadence. |
| **feddance** | `selection_bias`/`feddance_U` **CLOSED** (1.5h Jun 22) — inherited the syncfl `WALL_SEND_TS` duration fix ([syncfl/top_aggregator.py:499](../../flame/mode/horizontal/syncfl/top_aggregator.py#L499)): A2c PASS, Sx `feddance_U` KS=0.154, `feddance_I`/`A`/`V` all PASS, S2 KS=0.067. **U6 commit_visibility** root = real per-message metric blind to the barrier wait; **FIXED real-path (§6.u6 barrier-anchor)**, validated vs stored logs (15.65≈15.48), confirming rerun pending. **K2** 0.052-vs-0.05 = emergent rollup, all child rungs pass; residual = sub-threshold `feddance_I`(=stat_utility/loss, emergent) mix-bias + run-length noise (~1.4σ at 176 rd) → 3h rerun, no code lever. |

---

## Durable lessons (update in place, don't append)

- **Real is the reference, but VERIFY real is correct first — a real↔sim mechanism gap has
  TWO fix directions.** Parity ≠ blindly tuning sim to real. Oort case: sim recorded observed
  durations for 298 trainers (31% ≥15s), real only 201 (3%). (a) *Recording-or-not:* real
  `continue`d before `_handle_weights_msg` so a stale-returning trainer's speed/utility were
  never recorded → Oort reads `PROP_STAT_UTILITY is None` as *unexplored* (oort.py:472-478)
  and re-explores forever. Fix is on the **real** path (record props for stale-but-returned
  updates) + complete sim's utility recording — NOT a sim-side discard. (b) *Value:* the real
  path then recorded `recv_ts − dispatch`, which for a stale straggler bundles in the
  **aggregator read-wait** (finished update sits unread until a later round drains the buffer)
  — a server artifact, not client speed, inflating slow trainers up to 1.65×D so real
  over-avoided them. **Invariant: the selector speed signal `PROP_CLIENT_TASK_TRAIN_DURATION`
  is the CLIENT's task-train duration (`WALL_SEND_TS − dispatch` ≈ compute+net), NOT the
  aggregator-observed latency (`recv_ts − dispatch`)** — same concept both modes (sim's
  `SIM_CLIENT_TASK_TRAIN_DURATION_S = max(gpu,D)`). Single-sourced in
  `_real_client_task_train_duration`. Proof it was the scorer not residence:
  `scripts/oort_residence_discriminator.py`. See [[project_oort_a2c_root]].
- **`A2 num_eligible` can FAIL (KS) while `S3/4 in_flight` PASSES — read it as the same gap at
  two tolerances (refl Jun 22).** With an all-available trace (`A1 UNKNOWN≈300`), eligible =
  `candidates − in_flight_hold`, so a small in-flight gap (60.2 vs 63.15, rel 0.047 — under
  S3/4's 0.15 bar) lands directly on eligible (252.7 vs 249.8) where A2's tight KS≤0.2 binds.
  Don't chase A2 as a separate eligibility bug; walk to the in-flight/residence channel
  (`residence_rounds` 3.63 vs 3.86 localized the §4.5 release-timing root).
- **Decompose a net selection-rate gap into channels before fixing it.**
  `sel_rate = eligible_fraction × P(sel|eligible)`. If `eligible_fraction` matches sim/real
  (the menu is identical) the gap is the **scorer** (P(sel|elig)), not residence/eligibility.
  This invalidated the "sim under-holds slow trainers" (residence-leak) hypothesis for oort:
  slow-trainer in-flight residence was dead-equal (0.00444 vs 0.00438). The `in_flight_after`
  1.68-vs-1.33 aggregate that *suggested* residence was a RED HERRING — it decomposes to FAST
  trainers (real carries 4× more, slow ≈equal), a downstream consequence of real selecting
  fast more, read backwards.
- **A check consuming `agg_rounds` must split train vs eval (Jun 20).** Eval commits emit
  `event=agg_round` (`task_to_perform="eval"`) with no `agg_goal_count`, don't advance the
  clock. Letting them into the shared stream broke K1 monotone (eval's higher `vclock_now`
  faked 2656 backward steps), U3 staleness (24,980 eval `staleness=0` deflated 15.17→9.44),
  U1/U5. **Rule:** `load_agg_jsonl` partitions eval into `eval_commits`; `agg_rounds` is
  train-only; only U6/U6e read combined. Validate monotone invariants in true `ts` order.
- **`Sdet` triage.** `eligible_match≈0` + matching aggregates = stochastic-class, PASS;
  `eligible_match≈0` + diverging clock = genuine, fix clock; `eligible_match` high but
  `decision_match≈0` = the **values** diverge, not the set.
- **`participation`/S2 vs per-round Jaccard.** Round-to-round draw mismatch (`Sdet`/`S1`) is
  expected for a stochastic selector. A **systematic per-trainer skew over the full run** (S2
  `matched_count_ks` large, `max_diff` not averaging out) is a real bias (how refl's
  `participation` finding was distinguished from noise).
- **`U6 commit_visibility` KS is a false-positive on a sub-ms point mass.** When both modes
  commit immediately (real_mean 0.004s, sim 0.001s) KS→1.0 is signal-free; read `mean_diff`
  (3ms ≪ 2.0s bar). A real divergence (felix past-dating) shows a large `mean_diff` (14.8s).
- **`U6` LARGE `mean_diff` on a STRICT SYNC BARRIER was a REAL-telemetry flaw, fixed on the real
  path — NOT a sim bug, NOT a checker gate (feddance, Jun 22).** sim 15.5s vs real 0.02s. Sim
  `vclock−sct` is correct (barrier commits all K at `max(sct)`; slowest lag=0, early finishers
  0→52s). The bug: real `_update_visibility_lag` took `committed=datetime.now()` **per-message in
  the recv loop**, measuring arrival→ingestion (~0.02s), NOT the barrier wait — even though
  updates physically arrive SPREAD (`[MSG_ARRIVAL]` 33→48s, `queue_depth=0`). Don't reach for a
  checker WARN-gate or "real is blind" — real HAS the spread; **recompute it from stored raw
  logs first**: `lag_i = max_dur − dur_i`, `dur = WALL_SEND_TS − dispatch` gave 15.65 ≈ sim 15.48,
  proving the quantity exists and the metric (not the dynamics) was wrong. Fix = barrier-anchor
  the real path (§6.u6); only the sync-barrier baseline (feddance/fedavg) needs it — a STREAMING
  aggregator (oort/refl) commits each update at its own `sct`, so per-message is already correct.
  **Tell it's a metric flaw not past-dating:** sim per-round MIN lag ≈0 (past-dating shifts the
  whole dist up) AND `U3 staleness` 0/0 matched.
- **`P3 mean_overhead` is wall-capture, not a speed-model bug**, when sub-second,
  opposite-sign across baselines, and `grid_KS`/`training_delay_s` match — trust `P3` only
  when `grid_KS` also fails.
- **`K3b` ≠ "missing overhead" when `implied_per_commit_overhead_s`≈0.** Its residual
  `real_advance − sim_advance` is also moved by the per-round **max-of-K speed** order
  statistic; if implied overhead is sub-0.05s yet K3b fails, read K3a `max_speed` (a gap
  there = thinner sim speed-tail / selection mix). Do NOT add a `simCommitOverheadSeconds`
  scalar. *(Oort Jun 22: even this read was incomplete — K3b's oort residual was the stale
  read-wait inflation in the SCORER's duration input, not the speed model; see Settled roots.)*
- **Oort `in_flight_after` decay ≠ "carry-over gate broken" once decile-0 matches.** Gate
  (pinned `_round_start_vclock`) is correct; decay is a selection-mix tail. The `system_util`
  recency guard was the WRONG fix (Jun 19): with intrinsic per-task latency the last-observed
  duration is *correct*, so returning `system_util=1` removes a correct penalty (= forbidden
  speed-tail widening, no principled threshold). Instrument with `commit_visibility` to
  classify the decay instead.
- **`phase_gpu_compute` gap is run-length-sensitive.** felix 1h sim 0.18 vs real 0.42 (FAIL);
  2.5h sim 0.422 vs real 0.356 (PASS). Re-check at 2.5h before acting on a 1h FAIL.
- **`pastdated_by_source=[fresh=…]` was an EVAL artifact (Jun 20).** Classifier keys on
  `MODEL_VERSION` (= current round for a fresh-dispatched eval), so eval commits carrying a
  stale-train `sct` are mislabeled "fresh" with huge gaps. Root: `evaluate()` reused the last
  train `_sim_completion_ts` (`syncfl/trainer.py:418`). **Tell:** one trainer's eval commits
  repeat the same `sct` for hundreds of rounds. Fixed by stamping a per-eval `sct`.
- **`mqtt_fetch_s` is re-selection wait, NOT network transfer (Jun 20).** It's `wall(recv)` —
  the trainer blocked in `recv()` between sending round N and being re-selected
  (`syncfl/trainer.py:187-192`); scales with rounds-skipped (10-gap → ~35s). Real
  `gpu_compute_s` is 0.17s; cadence is governed by re-selection timing. Do NOT add the ~18.8s
  mean to `sct` (`mqtt-on-sct` dead end). The spread sim misses is per-trainer availability
  stagger, destroyed by round-boundary batch re-dispatch.
- **Two past-dating populations, two streams.** `commit_gap_s`/U6 is emitted only in the
  train (WEIGHTS) branch; eval exits before it. The train-only U6 mean and the all-commits
  SIM_BARRIER/CLOCK_DIAG stream can diverge wildly (16.8s vs "fresh=95%"). Always
  disambiguate which stream a "past-dating" number came from.
- **`gate_holds=0` over a whole run = the gate is structurally INERT, not satisfied (Jun
  21).** A correct gate never fires when the *state it judges* is wrong (felix: per-end
  `_sim_inflight_expected` overwritten by overlapping re-dispatch → earlier `sct` untracked).
  Read a pinned-0 counter as a tell, suspect upstream accounting.
- **Diagnose past-dating by partitioning the TAIL, not the mean (Jun 21).** After a fix drops
  the mean (26s→3.9s) the median can be 0; the residual is a thin growing tail (14% >5s,
  decile-0 0.46s→decile-9 8.2s). Bin per-commit `commit_gap` by run-fraction and read the
  tail's signature; the growth over the run is the compounding tell.
- **Measure invariants from overlapping intervals, not cumulative warning counters (Jun
  21).** `[SELECTION_CHECK] N unreturned versions` over-counts (one lost update inflates it
  for the rest of the run). The clean metric for one-in-flight: overlapping per-trainer
  dispatch→commit intervals — real 0% vs sim 13.9% settled the felix root.
- **`sim_committed_fresh` = agg_goal (10) confirms the block-for-K fix is closed** (oort 2.5h);
  any future `committed_fresh` gap is a different root.
- **45min exercises every mechanism but not C1/C2 or low-frequency drift** (refl 3h-only `K2`).
  Checker-side fixes validate instantly against stored dirs; only sim *mechanism* changes
  need a cluster rerun.

## Dead ends — do NOT retry
- **refl A2 num_eligible via §4.5 `pending_after`→`pending_ends` (hold ALL buffered until
  commit) — FALSIFIED (Jun 22).** Over-holds: sim `buf_depth`≈57–63 vs `held`≈46–48, so it
  would exclude ~13 more (eligible 252.7→~240) vs the ~3-end target (real 249.8). ~13 ends sit
  ready-but-uncommitted in the reorder buffer, but real does NOT hold all of them out — its
  selected_ends/eligible accounting is subtler than buffer occupancy. Instrument the exact
  per-round eligible decomposition in both modes BEFORE any hold change.
- Overhead > 0 on the virtual clock (masks & drifts; clock must `= max(vclock, sct)`).
- Prediction-only gates with no real blocking (never fire).
- **Tuning the `_sim_recv_min` gate predictor (`exp = sim_send_ts + budget`)** — realized
  compute is deterministic so `exp == sct` exactly; `gate_holds=0` came from the
  `_sim_inflight_expected` overwrite, not a loose predictor. Enforce one-in-flight instead.
- **Expressing "busy" via the UN_AVL unavailable list** (felix `simInflightResidence` v1,
  reverted). UN_AVL = can't participate at all; a busy trainer is AVL_*, just occupied.
  Routing busy → `curr_unavail_trainer_list` makes `_handle_send_state` evict it from
  `selected_ends`, free its slot, refill `c` with NEW trainers → in-flight ramps to N≈300,
  vclock crawls 0.27 s/rd. Hold the slot in `selected_ends` until commit instead (§3.resid).
  Sync oort's §4.5 unavail-list use is unaffected (barrier re-selects the cohort).
- `version_at(sct)` staleness relabel (fedbuff consumes the real number; inert).
- Adding `mqtt_fetch` (~57s) to `sct` (not version-relevant; inflates staleness ~6×).
- `simRedispatchGapSeconds=0` (over-overlaps) and `=0.6` (no measurable effect) for felix —
  don't retune this scalar; the gap is a real mechanism (or supersede via §3.drain).
- **"21s MQTT weight-fetch spreads real completions" (Jun 20)** — disproven; real
  `gpu_compute_s`=0.17s, `mqtt_fetch_s` is re-selection wait. No `mqtt-on-sct` / transfer-leg.
- **Re-dispatch `sim_send_ts = prior_sct` / `= max(now, prior_sct+latency)`** — backdates the
  stamp before the round whose weights arrive → `MODEL_VERSION`/causality break.
- **`simStaggeredRedispatch` / event-driven re-dispatch as the felix throughput fix —
  FALSIFIED (§3.evt).** Made advance worse (1.93→1.38): injected stagger is bounded by the
  very clock advance it's meant to create (≤0.55s injectable — circular). Throughput root was
  commit-side (`recv_fifo` stranding), fixed by `simSctOrderedDrain` (§3.drain). Kept in code
  (gated off), disabled in the yaml; never enable with `simSctOrderedDrain`.
- **Reading high wall-clock commit density as sim "running fast" (Jun 20)** — expected (no
  trainer wait); judge per-round *vclock* advance and `commit_gap`, not wall cadence.
- **`system_util` recency guard for oort carry-over decay (Jun 19)** — value-fudge identical
  to speed-tail widening, no principled threshold; it's the A2c selection-mix class.
- **"Widen the oort slow-speed tail" for carry-over decay** — `trainer_speed` already passes;
  sim tail is if anything wider. Selection-mix tail effect, not a speed-model gap.
- **oort task-type-keyed latency / train-vs-eval duration split (Jun 20)** — premise
  FALSIFIED: sync oort dispatches 0 eval tasks (eval bundled into the train commit,
  `oort/top_aggregator.py:901-903`); felix's `system_util` is inert (`round_threshold=70`).
- **felix clock-jump clamp / dispatch-ts pacing for "fresh" past-dating** — wrong target; the
  "fresh" past-dating is EVAL committing a past stale `sct`. Fix is in `evaluate()` (per-eval
  `sct`), not a forward-advance cap or `_SIM_ORDER_SLACK_S` retune.
- Expecting the felix min-budget seed fix alone to kill past-dating (drops 73%→14%, recovers
  to 59%; other cascade sources remained).
- Expecting oort carry-over decay to be a run-length transient (2.5h: sim 0.47 vs real 3.65,
  zero by decile 2 — structural).
- Re-chasing: GPU contention (overrun 0), SEND_TIMEOUT (0×), MQTT drops (0); per-trainer
  exact-set/identity on a stochastic streaming selector (path-dependent).
- Expecting seeding to align per-round sets (`Sdet eligible_match≈0` is expected; judge
  S2/participation). A scalar fudge for `P3 mean_overhead` ~1s offset (wall-capture).

## Naming discipline (apply when touching baseline code)
Names must be **context-free** (round/version/time confusion caused real bugs — D5):
- `_round` = round index (int), never a timestamp; use `_ts`/`_time_s` for times.
- Qualify *whose* round (agg global `self._round` vs selector `self._last_selection_round` vs
  per-trainer `end_last_selection_round`). Deferred: base aggregator `self._round` →
  `self._agg_round` (all-baseline pass).
- Clients do **tasks** (train/eval), not rounds: the selector speed property is
  `PROP_CLIENT_TASK_TRAIN_DURATION` ("client_task_train_duration_s"); msg enums
  `SIM_CLIENT_TASK_TRAIN_DURATION_S`, `CLIENT_TASK_TRAIN_COMPUTE_S` (Jun 22 rename).
- A local says *what it is*, not its type-shape (`trainer_model_version` kept over
  `trained_round`). Don't paper over an ambiguous name with a comment — rename it. Scope
  renames to the baseline you're in (oort+refl share `OortSelector`; felix is separate).

---

## §1  The parity ladder (methodology)
An FL run is a pipeline; each round flows the same stages in both modes:
`clock/time-base → availability → selection → dispatch+training → return+ordering →
aggregation → utility → emergent outcomes`. Parity must hold at every stage; if it breaks at
stage N, every stage above also diverges — those are **consequences, not bugs**. The checker
finds the **lowest broken rung** (earliest stage with sound inputs but diverging output) = the
root.

**Three roles** (tag each check): **CONTROL** confirms a stage's *input* is identical (failing
= fix the input model); **MECHANISM** confirms one *transformation* is modeled (failing with
passing controls = the localized bug, the prize); **EMERGENT** an aggregate outcome (never
localizes alone — walk *down* the ladder, never fix an emergent directly).

**Two axes:** STAGE (0–9, drives diagnosis) × TIER (drives verdict): `INV` (sim invariant,
hard FAIL), `EXACT` (tight tolerance, hard FAIL), `DIST` (distributional, FAIL unless
`--lenient`), `DIAG` (informational, feeds root-cause).

**Dependency gating:** each check declares upstream prerequisites. The engine walks rungs
bottom-up, labels the lowest enforced FAIL with all-passing upstreams **ROOT-CAUSE**, demotes
higher FAILs whose chain contains a failed check to **DOWNSTREAM**. `deps` names the
*strongest causal link*, not a generic base (TC1 gates only K10 — a missing field makes a
check SKIP, not FAIL).

**Growth rule:** every root-caused bug leaves behind the most fine-grained check that would
have localized it, at its stage with deps. Checks are **append-only** (a redundant check is a
future regression guard). Split a coarse check into one assertion per mechanism.

## §2  The ladder — check catalog
`[NEW]` = to implement; else exists in checks.py. "Isolates" = what a FAIL means when its
upstreams pass. "Dep" = upstream prerequisites.

**Stage 0 — Telemetry coverage** (gate for everything)
| ID | Check | Role/Tier | Isolates | Dep |
|---|---|---|---|---|
| TC1 `[NEW]` | Field coverage matrix | CONTROL/INV | a downstream-read field missing/sparse in one mode (explains every SKIP) | — |
| K10 | vclock_now present (sim) | CONTROL/INV | sim never stamps vclock | TC1 |

**Stage 1 — Clock / time-base** (the foundation; most bugs live here)
| ID | Check | Role/Tier | Isolates | Dep |
|---|---|---|---|---|
| K1 | vclock monotone (sim) | MECHANISM/INV | vclock goes backwards | K10 |
| K7 | sim_rate in [0.01,100] | MECHANISM/INV | vclock/wall absurd | K10 |
| P3 | trainer_speed_s distribution | CONTROL/DIST | speed-model *input* differs | — |
| K3a `[NEW]` | Modeled-compute advance | MECHANISM/EXACT | advance **formula** (K-th fastest async / max-of-K sync), overhead excluded | P3,K1 |
| K3b `[NEW]` | Overhead residual | MECHANISM/EXACT | `real_advance − sim_advance` ≈ 0 (missing per-commit overhead) | K3a |
| K4 | Overlap factor | MECHANISM/DIAG | sim misses inter-round async pipelining | P3,K1 |
| K3 | Per-round advance dist | EMERGENT/EXACT | K3a+K3b+K4 rollup | K3a,K3b,K4 |
| K2 | Rounds-per-virtual-second | EMERGENT/EXACT | throughput rollup | K3 |

> Decomposition is the point: P3✓ K3a✓ **K3b✗** K4✓ → pure missing overhead. K3b
> cross-validates at Stage 4 (mqtt_fetch): trainer-level overhead = K3b residual × agg_goal.

**Stage 2 — Availability** (indexed by the clock; gated on Stage 1)
| ID | Check | Role/Tier | Isolates | Dep |
|---|---|---|---|---|
| A1 | avail_composition parity | MECHANISM/DIST | per-state counts diverge | — |
| A2 | num_eligible / num_candidates | MECHANISM/DIST | eligible-set size diverges | A1 |
| A3 `[NEW]` | Trace time-base consistency | CONTROL/DIST | availability indexed by different clocks (REFL HIGH-1) | K3 |
| A4 `[NEW]` | Per-trainer duty-cycle | MECHANISM/DIST | on/off fraction differs even when set sizes match | A3 |

> A2-fail + A3-pass = fix the clock first; A2-fail + A3-fail = fix the trace lookup.

**Stage 3 — Selection**
| ID | Check | Role/Tier | Isolates | Dep |
|---|---|---|---|---|
| S3/4 | num_chosen / in_flight / effective_c | MECHANISM/DIST | selector picks a different count | A2 |
| A2c | selected-vs-pool speed bias | MECHANISM/DIST | scoring bias diverges with pool matched (oort) vs pool itself (A2b, refl) | A2b |
| Sx | selector score-term localize | DIAG | which utility term drives a mix split (oort believed_I/temporal/system_util) | A2b |
| Sd | preferred-duration penalty bind | MECHANISM/DIST | oort speed-penalty binding freq + reconstructed `pref` (caught D1: real 80% vs sim 46%) | A2b |
| S2 | Participation frequency | EMERGENT/DIST | per-trainer chosen-count diverges | S3/4 |
| S1 | Per-round Jaccard | DIAG | exact set identity (WARN for stochastic) | A2 |

**Stage 4 — Dispatch & training** (per-trainer timing; the overhead source)
| ID | Check | Role/Tier | Isolates | Dep |
|---|---|---|---|---|
| T2 `[NEW]` | training_budget_s dist | CONTROL/DIST | speed-model *input* differs | — |
| T_pre/T_w2g/T_gpu/T_w2r/T_post `[NEW]` | per-phase splits | MECHANISM/DIST | one timing phase each (T_gpu dep T2) | — |
| T_mqtt `[NEW]` | mqtt_fetch_s phase | MECHANISM/DIST | per-commit MQTT overhead (cross-checks K3b) | — |
| T3 | GPU budget respected | MECHANISM/INV | real GPU overruns modeled budget | T2 |
| K6 | sim_send_ts correctness | CONTROL/INV | sim dispatch ts not stamped/advancing | K10 |

> Split the one `trainer_phase` DIAG blob into per-phase DIST sub-checks so the report names
> the diverging phase; keep the combined table for at-a-glance reading.

**Stage 5 — Update return & ordering**
| ID | Check | Role/Tier | Isolates | Dep |
|---|---|---|---|---|
| U5 | Inter-arrival order (Spearman) | MECHANISM/DIST (WARN) | arrival rank within a round diverges | K3,S3/4 |
| U4 | agg_goal_count cycles 1..K | MECHANISM/INV | lost/double-counted update | — |

**Stage 6 — Aggregation**
| ID | Check | Role/Tier | Isolates | Dep |
|---|---|---|---|---|
| U6 | Commit visibility lag | MECHANISM/DIST | aggregator-clock delay READY→COMMITTED diverges (sim past-dating); upstream of staleness. Same metric both modes (sim `vclock−sct`, real `wall commit−arrival`) | K3 |
| U3 | Staleness distribution | MECHANISM/DIST | staleness diverges (async: downstream of clock under-charge) | K3,U5,U6 |
| P1 | Aggregation sequence | EMERGENT/DIST (WARN) | per-round contributing set diverges | S2,U5 |

**Stage 7 — Statistical utility**: F1-3 per-trainer utility dists (EMERGENT/DIST; dep S2,T_gpu,U3).
**Stage 8 — Emergent**: K8 terminal-state @ matched V (dep K2,S2); U2 total commits @ V (dep
K2,U4); C1 accuracy curve, C2 `[NEW]` loss curve (dep F1-3,K8).
**Stage 9 — Budget/stop sanity**: K9 stopped-by-budget-not-cap (WARN); K5 failsafe ceiling
(sim wall overshoot >20%).

---

## §3  Mechanism reference — landed sim fixes
The sim does **real GPU compute** but stamps a *modeled* completion `sct` (no wall sleep).
**Overriding principle:** parity ≠ goal, a *correct* simulator is; real is the reference only
after `validate_real` shows it admissible (concurrency 28.8/c30, double-dispatch 0). Never
tune sim to a wrong real. All mechanisms below are config-gated (flag-off ⇒ byte-identical)
and guarded by tests.

### Clock & ordering (felix async, validated)
- **Overhead → 0** (`simCommitOverheadSeconds=0`): clock TRACKS completions
  (`vclock = max(vclock, sct)`), not an overhead ramp.
- **Drain by physical READINESS** (`_sim_recv_min`): admit any in-flight end whose message
  physically arrived into the reorder buffer → slow trainers buffer as futures, commit in
  `sct` order (staleness 7.2→3.5).
- **`realDistributeSettleSeconds=0`**: removes a real-only 2×`sleep(0.1)`/commit (advance 4.1,
  staleness 2.8).

### §3.drain  sct-ordered DIRECT drain (felix async; LANDED Jun 20)
`simSctOrderedDrain`. The async clock under-advanced because in-flight updates (instant in
sim) ingested via the `recv_fifo` streamer could be **stranded** out of the reorder buffer's
view (background task + shared `_rx_queue` + per-end dedup + grace timeout). The clock advanced
off the incomplete buffer and lapped stranded lower-`sct` updates → past-dating (`queue_wait`
26s/90.6% >5s, staleness 15 vs 2.8; 1.4 s/rd vs 3.85). Fix: drain each live in-flight end's rx
queue **directly** so the buffer is a complete snapshot and the min-`sct` gate + clock-jump
clamp commit in true order (`commit_gap≈0`). `End.get_ready_nowait` (non-blocking, peek-aware)
+ `Channel.drain_ready` (pull raw on backend loop, decode off-loop so `cloudpickle.loads`
can't stall the pump); `_sim_recv_min` drains `recv_ends ∪ _sim_inflight_expected`. Sync
untouched (`recv_fifo(first_k=len(ends))` barrier already waits for the whole cohort). Guard:
`test_async_sct_ordered_drain.py`. **Validated Jun 21:** `commit_gap` median 0, staleness
15→5.1, advance 1.4→3.04. Residual = overlapping re-dispatch → §3.resid.

### §3.resid  One-in-flight-per-trainer (felix async; `simInflightResidence`; LANDED Jun 21)
**Invariant:** a trainer is re-pickable ONLY after its update returns AND is committed. Real
satisfies it by construction (channel holds an in-flight trainer out of `VAL_CH_STATE_SEND`
until aggregated — real 0% overlap). Felix sim freed trainers instantly → a fast trainer
re-selected while its prior update was in flight (sim 13.9%), overwriting `_sim_inflight_expected`
→ earlier update untracked, invisible to the `sct` gate (`gate_holds=0`) → lapped → past-dated
(14% tail, staleness 34). **First attempt (busy → UN_AVL) was WRONG** — see Dead ends.
**Correct fix:** a busy trainer HOLDS its concurrency slot in `selected_ends` (like real) until
commit. `_sim_hold_busy_slots` (`asyncfl/top_aggregator.py`, called from `_aggregate_weights`
at agg-goal) holds `pending_ends() ∪ set(_sim_inflight_expected)` in `selected_ends`/
`all_selected`/`_sim_pending_commit` (a SLOT, not unavail); released on commit in
`_sim_recv_min`. Bounds concurrency (`extra = c − len(selected_ends)`) AND excludes from the
pool. Sim only; sync uses §4.5 unavail path (correct — barrier re-selects). Guard:
`test_async_inflight_residence.py`. **VALIDATED Jun 21** (`…154600…sim` 90min): **46/46**, K3b
0.82→**−0.08**, advance 3.04→**3.93** (real 3.85), staleness 5.11→**2.83** (real 2.79), U6
mean_diff 3.91→**0.016s**. The whole K3b→{K2,U3,U6,K8,U2} cluster cleared together — one root.

### §3.evt  Event-driven re-dispatch (felix; FALSIFIED Jun 20, superseded by §3.drain)
`simStaggeredRedispatch` (gated, kept off). Pushed each commit's advanced vclock to a freed-slot
FIFO to re-stamp re-dispatched `sim_send_ts` and regain stagger. **Falsified:** advance got
worse (1.93→1.38) — the injected stagger is bounded by the clock advance it's meant to create
(≤0.55s injectable, circular). Real root was commit-side (§3.drain). Do not enable with
`simSctOrderedDrain`. Guard kept: `test_async_staggered_redispatch.py`.

### §3.async  Async ≠ sync selector knobs — do NOT inherit Oort *paper* defaults
`third_party/Oort` is sync-only; the paper defaults are SYNC values. Applied to the async
`AsyncOortSelector` (felix) they regress it (overlap 10.9× vs 6.6×) because the overlap model
is calibrated to the selected MIX and sync knobs narrow it. Root theme: many Oort knobs are
**per-round**, but "a round" differs in async (one `agg_goal` batch) vs sync (a barrier), and
async runs ~2–3× more of them.
- **`round_threshold`** (speed penalty) protects a SYNC barrier; async has no barrier →
  largely inert. Felix uses **70** (the pacer only raises toward 100, so the start washes out).
- **`exploration_decay`** applied per-round; sync 0.95 floors exploration in ~29 async rounds.
  Felix uses **0.999** (`0.9999` ≈ never exploits — rejected).
- temporal/UCB `√(0.1·log(round)/last_selected)` auto-inflates with round count; pacer cadence
  fires more often in wall-time; staleness weighting is async-only.
- Principled generalization (not done): re-parameterize per-round terms by wall-time /
  samples-seen. Until then knobs are config-driven, anchored to the real run's spacing.

### §4.5  refl/oort — `sct`-gated pool exclusion (`simInflightResidence`, validated)
A trainer modeled as still computing (`vclock < sct`) must NOT re-enter the eligible pool. In
`oort/top_aggregator._distribute_weights`, `_sim_buffer.pending_after(vclock)` is added to
`trainer_unavail_list` (the *unavailable* path, NOT `selected_ends` — which would re-dispatch
and reset `sct`); released at `vclock ≥ sct`. Fixed refl's pool composition (A2b 12.4→~6.5 =
real). Guard: `test_virtual_clock.py::test_pending_after_*`, `TestSimInflightResidence`.

### §4.9  oort — `sct`-gated carry-over (`simInflightCarryover`)
Sync oort over-selects (×1.3), closes at agg_goal=10, leaving ~3 slowest computing. Real keeps
them in `selected_ends` across rounds (`in_flight_after` 3.3); sim's update arrives at once,
gets stale-rejected, frees its slot → drains to 0.15. Fix: a prior-round straggler
(`_round − MODEL_VERSION > 0`) with `sct > vclock_round_start` is held (not yielded, not
clock-advanced), re-buffered in `selected_ends`, commits a few rounds later. Three follow-on
bugs fixed: (1) **lost straggler** — re-buffer ran after the yield-loop the caller abandons;
wrapped in `try/finally` (Jun 16). (2) **threshold creep** — `vclock_round_start` re-read each
`_oort_sim_recv` call crept forward across block-for-K retries; `_aggregate_weights` now pins
`self._round_start_vclock` once (Jun 17). (3) **block-for-K-fresh starvation** — the second
poll loop ran `while not self.simulated`, so sim got one pass and skipped not-yet-ready fresh
trainers → committed stale; removed the gate (confirmed 2.5h `sim_committed_fresh=10`). Guard:
`TestSimInflightCarryover`. *(The residual carry-over "decay" was the A2c scorer-input root —
see Settled roots; NOT a gate or speed-tail bug.)*

### §6.u6  syncfl real U6 barrier-anchor (feddance/fedavg; LANDED Jun 22)
`update_visibility_lag_s` on the REAL strict-sync path was wrong: `_update_visibility_lag`
evaluated `committed = datetime.now()` **per-message inside the recv loop**, so it measured
arrival→ingestion (~0.02s/update) — NOT the barrier wait. A strict barrier applies all K at ONE
post-loop instant (`optimizer.do`), so an early finisher's true visibility lag = barrier − its
own completion. Real updates do physically arrive spread (`[MSG_ARRIVAL]` 33→48s within a round,
`queue_depth=0`); the per-message metric was just blind to it (sim 15.5s vs real 0.02s = U6
FAIL). **Fix:** real anchors on the single round barrier — `_barrier_anchored_lags(durs)` returns
`max_dur − dur_i` with `dur = WALL_SEND_TS − dispatch` (client task-train duration, the
dispatch-relative completion matching sim's `sct`); sim is unchanged (`vclock−sct`, vclock is
already advanced to the barrier). **Why only feddance, not oort/refl:** oort/refl use the
oort overlay's STREAMING commit — `_oort_sim_recv` pops in `sct` order and `_advance_sim_clock`
tracks each pop, so each update commits at `vclock≈own sct` → lag≈0 in BOTH modes (real commits
first-K-to-arrive near arrival too). The barrier wait only exists for a baseline that waits for
the slowest of its cohort (feddance). So oort's per-message helper stays correct and is left
untouched; only `syncfl._aggregate_weights` (feddance + fedavg base) is barrier-anchored.
**Validated against STORED real logs (no rerun):** recomputed lag mean 15.65/min 0/max 53 ≈ sim
15.48/0/52. Pure telemetry (no dynamics/staleness effect). Guard:
`test_sync_sim_ordering.py::test_barrier_anchored_lags_*`. Pending: confirming real feddance rerun.

## §5  Checker corrections (stochastic / observability classes)
Once dynamics match, some residual FAILs were the checker enforcing exact identity on
quantities a stochastic/in-memory sim can't reproduce (tell: byte-identical across runs despite
large dynamics changes). All principled, guarded, append-only; a future *deterministic*
selector still gets exact enforcement via `DETERMINISTIC_SELECTORS`.
- **P1 aggregation_sequence** → WARN for stochastic (S2 participation is the enforced invariant).
- **F1-3 utility** → pooled KS (per-trainer KS=1.0 was mechanical for n≤2; means identical).
- **phase_mqtt_fetch** → DIAG (in-mem cache wall time, deliberately off the virtual clock).
- **trainer_speed / eligible_speed / selection_bias** → integer-grid / metadata-pool.

## Discrepancy ledger — flame vs reference Oort
flame has ONE `OortSelector` for both `oort` (matches `third_party/Oort`) and `refl` (matches
`third_party/REFL` fork). The references differ on defaults, so each baseline's knobs are
config-driven (`selector.kwargs`), defaulting to the Oort paper (`OORT_PAPER_DEFAULTS`) with
refl overriding.

| # | discrepancy | resolution |
|---|---|---|
| D1 | `pref` not sorted | FIXED (sort added) — port bug; validated on oort |
| D2 | stat-utility not normalized/clipped | FIXED (`scoring.oort_normalize_reward`, config) |
| D3 | `round_threshold` | config: oort/felix=10 (paper), refl=30 (fork) |
| D4 | `cut_off_util` + cutoff-index | FIXED: config (0.7 paper / 0.05 refl); index thresholds the exploit-boundary score (was inert) |
| D5 | temporal time-base | **FIXED (Jun 16) oort+refl** — VALUE always correct (selection round == `MODEL_VERSION`); bug was *write timing* (written at commit → value rode commit ordering). Now stamped at **selection** (`oort.py::_record_last_selected_round`). felix (`AsyncOortSelector`) deferred. |
| D6 | `clip_bound` | config: 0.98 paper / 0.9 fork |
| S | refl exploitation | FIXED: was deterministic top-k; now fork's cut_off_util-weighted `np.random.choice` |
