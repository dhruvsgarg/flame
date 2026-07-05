# High-Fidelity Simulator for FwdLLM — Real↔Sim Parity

**Active build (branch `dg/fwdllm_sim_unavail`).** A simulated-clock runner for the `fwdllm` example
(FedFwd / forward-gradient FL) that reaches **real↔sim parity** across the **fluxtune / fwdllm / fwdllm++**
baselines — at **100% availability (syn_0)** first (Phase 1), then under **unavailability**
(syn_20/50/mobiperf, Phase 2), then **beyond syn_0** (Phase 3). fwdllm has no native sim clock; the build
wires flame-core's virtual clock + availability substrate into fwdllm's variance-gated gradient loop.

> **DESIGN PRINCIPLE — reuse async_cifar10's concepts, deviate where the workload demands.** We reuse its
> virtual clock, sct reorder buffer, in-flight gate, availability substrate, and parity ladder wherever they
> transfer. But fwdllm is a different workload (aggregates **gradients** not weights; endogenous
> **variance-gated dynamic-K** commit cadence; **`data_id`** progress axis; one-message-per-call grad loop;
> rollback across agg-goal cycles), so a verbatim port is sometimes wrong. When a fork appears, **decide
> explicitly** (don't silently copy or silently invent) and **log it in §K**. The at-a-glance delta table is
> **§B.1** (curated, current); §K is the one-line-per-decision rationale log.

> **DOC MAINTENANCE — standing instruction (keep this doc rich but CRISP).** This is a living status doc, not
> a changelog. Every edit obeys:
> - **Fold, don't append.** A decision that has landed is **one line** (anchor + what/why). A mechanism that
>   **worked** is **one sentence** in §G Fixes-landed. Something that **didn't work / was believed-then-refuted**
>   goes to §H Dead-ends as one line. No multi-paragraph appends — update the relevant section **in place**.
> - **Status sections are rewritten in place, not grown.** §A (current state), the scoreboard, the open-issues
>   index, and open-roots reflect the CURRENT truth only; per-run history lives in git + the parity JSONs.
> - **§K is one entry per deviation.** Keep the anchor (referenced from §B.1) + a one-line rationale once folded;
>   expand only an entry that is still open/contested.
> - **Delete stale scaffolding.** Cold-start build maps, superseded plans, and finished batch checklists are
>   removed once landed (the code is the source of truth), leaving a one-line pointer if still referenced.

**Prerequisites:** [async_cifar10/PARITY.md](../async_cifar10/PARITY.md) — the parity methodology (ladder,
roles/tiers/gating, run-length budget, landed sim mechanisms); **fwdllm's rung catalog is PARITY.md §F**.
[async_cifar10/UNAVAILABILITY_DESIGN.md](../async_cifar10/UNAVAILABILITY_DESIGN.md) — the availability substrate
fwdllm inherits via its aggregator class chain (ClientAvailability mixin, trace-read effect path, two-ledger
discipline, starvation self-termination, A6/A7/A8/K11 ground-truth rungs).

---

## §A  Current status

**Ground truth (2026-07-04 PM): the Phase-1 sign-off run RAN and is USABLE — the first matched-`data_id`
real↔sim pair for all three baselines.** Command:
`run_sequential.sh --mode both --delays on --max-data-id 10 --max-runtime-s 3600 --yes`
(runs `run_20260704_172155..181912`). **Config sanity PASS** (flags honored bash→yaml→banked config→runtime;
delays matched per-trainer real-sleep = sim-modeled; caveat: `training_delay_factor=None` → the hardcoded ÷10,
so D≈0.4–1.8s, not the full 4–18s — pass `--delay-factor` for the full delay, issue #12c).

**The S1 blocker is gone:** all 6 processes stopped cleanly on **matched `data_id=10`**, none on
`[SIM_WALL_CEILING]` — the Phase-4a ceiling-decouple + matched-`data_id` stop + Stage-C fwdllm_plus liveness
all worked (fwdllm_plus real completed 10, was stalling at ≈3).

**Headline results:** (1) **#6 clock-rate ROOT-CAUSED + FIXED (K-D25).** The "sim under-models real wall 2×"
gap was a **checker-anchor artifact, not a sim under-charge** — the rungs compared sim-vclock (genuine
algorithmic time) against real's FULL wall, which carries a ~constant ~7.7s/round localhost transport ARTIFACT
(mqtt re-fetch/redistribute/drain-tail/sleeps) the sim correctly omits. Fix: real now anchors on its intrinsic
algorithmic clock; throughput 0.47→0.048, wall_disparity 90→1.6s (landed, 878-green). The **phase rungs (Root B)**
are fixed too (real's slept delay excluded from post_train; B2 straggler out of the emitted training_budget).
(2) **fluxtune R1 residence REGRESSED to 60.4%** — the K-D19 "fixed in code" claim is **refuted** by this
emergent smoke (was 44.7%); `[SIM_R1_DISPATCH]` fires despite `_release_end_on_return`. Now the top blocker.
(3) **#13 sim_rate** for the sync baselines is near-1 (0.82/0.85) and Root A shows the vclock is correct — the
residual is a small physical overhead; the severe case (fluxtune 0.26) is **R1-confounded** → fix #1c first.

**UPDATE (2026-07-04 later PM, `run_20260704_2224..2258`, data_id=20): K-D25/K-D26 clock-rate+phase fixes
VALIDATED on fresh banked pairs — for BOTH sync baselines.** fwdllm rose **44→49 pass**, fwdllm_plus **40→45**:
the 5 clock-rate rungs (throughput/overhead/per_round base/commits/terminal) + phase all **flipped to PASS**
exactly as predicted. fwdllm's 4 survivors are benign (2 marginal-KS, 2 mis-applied gpu_budget). fwdllm_plus's 7
survivors are the #7 selection divergence (eligibility/selection_detail/avail_timebase — real 4.9 vs sim 9.6
eligible at syn_0), gpu_budget, and a 13-vs-14 total_commits/terminal off-by-one (real capped at
`max_runtime_s=1800s`, data_id<20; sim finished 20 — a stop-mismatch + boundary, not a bug). **fluxtune sim did
NOT run** — a startup MQTT join-notify race dropped trainer #379's JOIN → 9/10 ends → async_oort never reached
`minInitialTrainers=10` → hung with **zero selections** (#14, **FIXED** — `join()` waits for connect; full
`tests/` 879-green). The re-run (`_2327` sim) then banked cleanly (10 ends, data_id=20) → **39/14/19**, and
**answers the K-D26 open question: part (a) alone recovered ~0 of R1** — sim R1 residence is still **62.9%**
(215 `SIM_R1_DISPATCH` fires) at sim_rate 0.29, so the sustained NONE-delete path / **#13 slow sim is required**
(part-b), not optional. Every fluxtune clock/commit/staleness fail is downstream of this R1+#13 desync.

### Per-baseline ground state
| baseline | stop | R1 | cadence (V1/V2/g2/U3/S2/conv) | sim_rate | surviving fails | verdict |
|---|---|---|---|---|---|---|
| **fwdllm** (sync) | data_id=20 both | **0% PASS** | **all PASS** | ~0.8 | per_round_advance + phase_gpu_compute (marginal KS 0.211/0.257 vs 0.20/0.25, means match 17.85≈17.48 / 1.03≈1.13), gpu_budget_real/sim (~40% overrun **symmetric** both modes — mis-applied invariant, not a divergence) | **49/4/21.** K-D25/K-D26 clock-rate+phase rungs VALIDATED PASS (44→49). Cadence CLEAN; 4 surviving fails all benign |
| **fwdllm_plus** (sync) | real→wall-cap 1800s (data_id<20); sim→data_id=20 | 0% PASS | **all PASS** | ~0.85 | #7 eligibility/selection_detail/avail_timebase (real 4.9/4.87 vs sim 9.6 eligible/chosen; in_flight 10=10), gpu_budget (symmetric ~40%), total_commits/terminal (13 vs 14 off-by-one, stop-mismatch) | **45/7/21.** Clock-rate rungs VALIDATED PASS (40→45). Fails = #7 + gpu_budget + stop-mismatch boundary; real ~4× slow (#7) |
| **fluxtune** (async) | data_id=20 both | **62.9% FAIL** ⛔ | V2 PASS; V1/g2 marginal (10.9 vs 9.1) | **0.29** | **R1** (part-a alone recovered ~0), throughput/overhead/per_round/total_commits/terminal/staleness (all R1+#13-confounded), selection_detail (1.68 vs 1.34), gpu_budget, convergence (1 eval) | **39/14/19.** #14 fix let the sim RUN (10 ends, data_id=20). R1 still 62.9% (215 tripwire fires) → **#1c part (b)/#13 required**, K-D26 confirmed |

### Parity scoreboard — LATEST vs penultimate (rewrite in place; two columns only)
| baseline | penultimate (sign-off `_1721..1819`) | **LATEST (`_2224..2258`, data_id=20)** | Δpass | key failing rungs |
|---|---|---|---|---|
| **fwdllm** | `0704_173105` · 44/9/21 | **49 / 4 / 21** | +5 (clock-rate+phase→PASS) | per_round_advance (KS 0.211), phase_gpu_compute (KS 0.257), gpu_budget_real/sim (~40% symmetric overrun) |
| **fwdllm_plus** | `0704_180511` · 40/12/21 | **45 / 7 / 21** | +5 (clock-rate→PASS) | #7 eligibility/selection_detail/avail_timebase, gpu_budget_real/sim, total_commits/terminal (13 vs 14, stop-mismatch) |
| **fluxtune** | `0704_181912` · 39/15/19 | **39 / 14 / 19** (`_2327` sim, post-#14) | ≈0 | **r1_inflight_overlap** (sim 62.9% vs real 0.2%), throughput/overhead/per_round/total_commits/terminal/staleness (R1+#13-confounded), selection_detail, gpu_budget, v1/g2, convergence |

*fwdllm's +5 is the K-D25/K-D26 validation: the 5 clock-rate rungs (throughput/overhead/per_round base/commits/
terminal) + phase (training_budget/post_train) all flipped to PASS on this fresh `intrinsic_span_s`-carrying
pair. The 4 survivors are benign — 2 marginal-KS (means match) + the mis-applied gpu_budget invariant (symmetric
real/sim overrun, not a divergence). **fluxtune could not be scored** — see #14.*

### Next run — expected per-baseline deltas (K-D25 baseline-agnostic; K-D26 fluxtune-only)
*A FRESH run is required for ALL THREE — `intrinsic_span_s` is a new agg emit; the checker's #6 anchor only
engages on runs that carry it (old banked runs fall back to raw `ts`). Consider `--max-data-id 20` to clear
fwdllm's 3 marginal rungs (finer KS + boundary washout); fwdllm_plus real ~26 min/10 data_ids fits in 3600s.*
- **fwdllm** — clock-rate (throughput/overhead/per_round/commits/terminal) + phase (training_budget/post_train)
  flip to PASS (~44→~50). The **3 marginal rungs** (per_round KS, total_commits/terminal off-by-one) stay
  borderline at exactly 10 data_id; clear at a longer run. `gpu_budget` may persist (mis-applied to fwdllm).
- **fwdllm_plus** — same clock-rate + phase improvement (real completes 10, slowly). **eligibility/
  selection_detail divergence (#7)** persists — separate root, not touched.
- **fluxtune** — clock-rate + phase improve; R1's **round-1 trigger gone (K-D26a)** but the **sustained
  NONE-delete path (part b) not fixed** → R1 partially recovers, throughput stays R1-confounded. **This run
  measures how much (a) alone recovers R1** → decides whether #13 (or a NONE-race fix) is required.

### Open issues — master index (OPEN top; 1-line issue + next step)
| # | issue | baseline(s) | next step |
|---|---|---|---|
| **#14** ✅ | **fluxtune sim hung at startup — a shared-MQTT join-notify race dropped 1/10 trainers' JOIN → no selection.** `backend/mqtt.py` registers an end ONLY on receipt of a `NotifyType.JOIN` (health-check `ON` is ignored for adds, mqtt.py:243). `join()` fired `notify(JOIN)` fire-and-forget; for trainer #379 the MainThread called `join`→`notify` **2ms before** the async `on_connect` set `_is_connected=True`, so notify returned False ("Cannot send notify: MQTT client not connected") and the JOIN was **dropped with no retry** (subscriptions too). Later STATE_UPDATE notifies don't add an unknown end → #379 invisible → async_oort stuck below `minInitialTrainers=10` → looped `distribute→ends:None→no ends yet`, zero selections, killed by timeout. Real dodged it (all 10 `on_connect` fired before `join`) → flaky startup race, not sim-logic. | all (MQTT) | **FIXED (robust):** `join()` calls `_wait_for_connect()` (spin on `_is_connected`, 10s bound) before subscribe+notify, so both the subscriptions and the JOIN happen post-connect; no-op when already connected (async_cifar10 unchanged). Full `tests/` 879-green. **Re-run the fluxtune sim to bank the pair.** |
| **#1c** ⭐⭐ | **R1 residence 60.4% — ROOT-CAUSED (K-D26): a physical-wall vs vclock desync in the shared `async_oort` re-pick guard, exposed by the slow sim (#13).** Two paths drop a still-outstanding trainer from `all_selected`: (a) the 90s `SEND_TIMEOUT_WAIT_S` abandon-timeout keyed on `time.time()` (round-1 trigger); (b) the aggregator marks carried grads `KEY_END_STATE=NONE` to keep their slot, and async_oort reads `NONE` as "left" → deletes from `all_selected` (sustained). Both misfire only because wall ≫ vclock. K-D19 misdiagnosed (checked the selection filter, not these deletion paths); K-D17b's NONE hypothesis was right. **DEEPER ROOT-CAUSE (K-D27, telemetry `_2327`): the NONE-delete is just ONE symptom of a two-ledger split — async_oort's selection eligibility (`filtered_ends`, async_oort:1607) is gated SOLELY on `all_selected`, a PHYSICAL-event-pruned ledger (recv-fifo 2s re-select, RECVD/NONE cleanup), and NEVER consults the aggregator's VIRTUAL in-flight truth (`_sim_inflight_expected ∪ _sim_buffer`). The bridge that would fix this — `_agg_pending_commit_ref` — is read for telemetry (async_oort:423) but NEVER assigned (always ∅) and never used as a filter; the aggregator's `_sim_pending_commit` is accumulated (fwdllm_aggregator:908) but never cleared/bridged (contrast asyncfl:618/1490 where felix DOES maintain it). So in the slow sim a grad sits returned-but-uncommitted for a long virtual window; the selector frees it on a physical event while the aggregator still holds it → own select() re-dispatches → R1. Real/async_cifar10 dodge it ONLY because wall≈vclock (window≈0), not by design.** Evidence: R1 62.9% vs 0.2%, 215 `SIM_R1_DISPATCH`, `num_eligible=10` while `in_flight=10`, `in_pending_commit=0/879` (dead hook), `[CHANNEL_CLEANUP] freed=0`×218 + 0 fires of the 90s-timeout/removed-ends prunes. | fluxtune | **Fix (a) LANDED (K-D26)** but insufficient. **Robust fix (b) = wire the virtual in-flight set into selection eligibility:** assign `_agg_pending_commit_ref` = aggregator's `_sim_inflight_expected ∪ _sim_buffer` (kept in sync: add on dispatch, discard on COMMIT) and add `end not in _agg_pending_commit_ref` to `filtered_ends` — sim-only (∅ in real → async_cifar10 byte-identical). Drives `SIM_R1_DISPATCH→0`. Underlying window also collapses if #13 makes the sim fast. |
| **#6** ✅ | clock-rate rungs anchored real on FULL wall (genuine + ~7.7s/round transport artifact) vs sim-vclock — **checker-anchor bug, sim vclock is correct** | all | **FIXED (K-D25):** agg emits `intrinsic_span_s` (barrier+eval, mirrors the sim vclock composition, fedavg excluded); the 5 clock-rate rungs + `wall_disparity` anchor real on the cumulative intrinsic clock (async byte-identical). Validated: throughput 0.47→0.048, overhead 21→1.1, wall_disparity 90→1.6. **Residual:** `per_round_advance` KS + `total_commits`/`terminal_state` off-by-one at 10 data_id (small-sample + B2 straggler boundary) → clear at longer run / B2 calibration. |
| **~~phase~~** ✅ | `phase_post_train`/`training_budget` compared a mode-dependent phase (real slept the delay; sim carried B2 straggler in its emitted budget) | all | **FIXED (Root B, K-D25):** post_train stamped after the delay (pure post-proc ~0 both modes); B2 straggler moved from `_delay_s` into the sct only, so `training_budget_s` emits the base delay (identical real/sim). Validated 0/0 and 1.14/1.14. |
| **#13** ⭐ | sim is physically slow (`sim_rate` 0.26 fluxtune / 0.82–0.85 sync). **Now promoted:** the slow sim (wall ≫ vclock) is the ROOT that makes async_oort's wall-keyed guard machinery misfire → #1c (both the 90s timeout and the NONE-delete race). | all (fluxtune worst) | Cure the fluxtune inter-round physical wall (the recv_fifo/MQTT waits, #11-adjacent) so wall≈vclock like async_cifar10 — then the #1c NONE-delete sustained path collapses on its own. Tracked as the deeper follow-up per the #1c "both" decision. |
| **#7** | fwdllm_plus real ~4× slower/round (156 vs 41 s/round); at syn_0 real sees only 4.9 eligible vs sim 9.6 | fwdllm_plus | Real liveness FIXED (Stage C — completes 10). Remaining: profile the per-iteration reselection + oracular read cost from the banked per-phase log; explain the real/sim eligible-count gap at 100% avail. Not a sim bug. |
| **#12c** | `--delays on` uses the hardcoded `training_delay_factor=10` ÷10 shrink unless `--delay-factor` passed | all | Optional: pass `--delay-factor` for the full 4–18s registry delay. Config-flow otherwise correct (K-D24/K-D21). |
| **#11** | real-mode critical-path waste (`sleep(0.1)` MQTT-settle busy-waits, real-only; one-grad-per-poll drain tail) | fwdllm, fwdllm_plus (real) | **Deferred to a validated pass** — ZERO parity impact (sim already skips them); removing them changes the working real reference + needs a real run to validate the MQTT settle timing (principle #8/#11c). Never change grad values/cadence. |

### Open roots — ranked lowest-rung-first
1. **#1c R1 residence (fluxtune)** — ROOT-CAUSED (K-D26): a wall-vs-vclock desync in the shared async_oort
   re-pick guard. Wall-timeout path FIXED; the sustained NONE-delete path is the same slow-sim root → **#13**.
2. **#13 slow sim (fluxtune wall ≫ vclock)** — promoted: it is what makes the async_oort guard machinery misfire
   (#1c) AND the headline slowdown. Cure the fluxtune inter-round physical wall → wall≈vclock (async_cifar10).
3. **#7 fwdllm_plus real speed / selection divergence** — real completes now; explain the 4× slowness and the
   syn_0 eligible-count gap from banked telemetry.
3. *(#6 clock-rate — RESOLVED as a checker anchor, K-D25; sim vclock was correct. Only the 3 marginal rungs +
   the B2 straggler calibration remain, both emergent at the longer sign-off run.)*

### SKIP audit (19–21 skips; ~16 legit)
Legit (mechanism genuinely absent at Phase-1 syn_0 + `random` selector): 7 availability ground-truth rungs +
4 delivery/withheld (Phase-2 effect path, not built) + 3 DynamicKC (disabled by design) + 2 oort-only
(`random` baselines) + `residence` (async telemetry). The 12 former rigor-gap skips (4 advance rungs, 8
phase-timing rungs) are now **un-skipped** (K-D21 A1/A3) — they run and mostly FAIL on #6, as intended.

---

## §B  How fwdllm differs structurally

fwdllm aggregates **gradients** (JVPs) not weights; its commit cadence is **endogenous** (variance-gated
dynamic-K); its progress axis is committed **`data_id`** (variance passes), not update count. Gradient values
are mode-invariant given identical input+perturbation seed, so parity reduces to **clock + selection + ordering
parity plus a variance-cadence layer**. Anchors (`flame/mode/horizontal/syncfl/fwdllm_aggregator.py`):
`aggregate()` var gate + rollback/`cached_v` at each `_agg_goal` boundary; `total_data_bins=150`;
force-commit cap `max_iterations_per_data_id`; `reselect_each_iteration` (fwdllm++ per-iteration reselection);
sync path `_aggregate_grads_sync`. Full detail: PARITY.md §F.1.

### §B.1  Real↔sim design deltas vs async_cifar10 — **CURATED, KEEP CURRENT**
Separates an **intentional fwdllm choice** from an accidental discrepancy. Update a row when a deviation
lands/changes; the §K column points at the one-line rationale.

| # | Axis | async_cifar10 | fwdllm | Why | §K |
|---|---|---|---|---|---|
| 1 | Aggregated object | model **weights** | **gradients** (JVPs) | grad values mode-invariant → parity = clock+order+selection + variance-cadence | §F.1 |
| 2 | Progress axis | update/round count | committed **`data_id`** | cadence (updates-per-data_id) is an **output to match**, not an input | principle #2 |
| 3 | Commit cadence | fixed `agg_goal` | endogenous **variance-gated dynamic-K** | the emergent layer cifar doesn't model; V/DK/G rungs verify it | §F.1 |
| 4 | sct delay model | `send + max(gpu, D)` | `send + gpu + D` (**additive**) | fwdllm's real mode sleeps D *on top of* GPU; `max()` would desync | K-D2 |
| 5 | Per-eval sct | distinct, ~20× faster | **collapses to train sct** | eval lives on the aggregator; forward-grad "train" IS a forward pass (no 20× factor) | K-D3 |
| 6 | Slot residence | per-commit release | **hold slot to COMMIT** (felix-aligned) — every dispatched-but-uncommitted trainer held in `selected_ends`+`all_selected`, released on commit | a returned-but-uncommitted grad is still in flight in VIRTUAL time (commits when vclock reaches sct) | K-D5/K-D17b, principle #4 |
| 7 | Surplus grad on rollback | carried | **carried** for async (fluxtune, c≫agg_goal); **drop** stays correct for sync (c≈agg_goal) | drop was benign only for sync; fluxtune dropped ~7/cycle → 2× passes | K-D6/K-D12 |
| 8 | Async drain primitive | `_sim_recv_min` verbatim | purpose-built `_sim_recv_min_grad` / sync `_sync_sim_recv_first_k` | cifar's per-commit release + withheld paths key on WEIGHTS semantics | K-D4 |
| 9 | `time_mode` default | `"simulated"` | `"real"` (getattr fallback) | fwdllm's whole config corpus is `real`; a `simulated` default risks half-activating an unbuilt path | K-D1 |
| 10 | Cadence telemetry | n/a | **pre-mutation** cycle snapshot (`cycle_data_id`/`cycle_iteration`/`grad_pool_size`/`cached_v_size`) | post-mutation `data_id` advances before emit → off-by-one; snapshot makes V1 exact | K-D9 |
| 11 | Availability tracking (v1) | all `trace_read` | **mixed**: fwdllm unaware, fwdllm_plus `oracular`, fluxtune `client_notify`→approx `trace_read` | baselines carry different models; first-class `client_notify` deferred to Stage H | D1 |
| 12 | Launch tooling | single parity template (`debug_run.sh`) | per-baseline yamls + `_sim` files (`run_sequential.sh`) | different config models; both source the shared harness `examples/scripts/expt_runner.{sh,py}` | this work |

---

## §C  Baseline matrix (resolved from the launcher configs)
| baseline | sync/async | selector | agg | tracking / avail | reselection | K (agg_goal) | dynamic_kc |
|---|---|---|---|---|---|---|---|
| **fluxtune** | async | `async_oort` | fedbuff (+server LR, JVP) | `client_notify`→`trace_read` v1 (aware-at-select, reactive-90s) | — | 3 | disabled |
| **fwdllm** | sync | `random` | fedavg | `default` unaware (reactive-90s) | per-round | 10 (=c) | — |
| **fwdllm_plus** | sync | `random` | fedavg | `oracular` (aware-at-select via trace read) | per-iteration | **10** (yaml is source of truth; §C draft read 2) | — |

Phase order (locked): Phase-1 syn_0 → Phase-2 unavailability → Phase-3 beyond syn_0. One baseline at a time.
Stage 3 oort rungs run **only** for fluxtune; sync-barrier rungs run for **fwdllm/fwdllm_plus**.

---

## §D  Parity ladder for fwdllm (rung defs live in PARITY.md §F)
- **Reused verbatim:** Stage 0 TC1/K10; Stage 1 P3/K1/K7; Stage 2 A1–A3 (+A6/A7/A8 once avail telemetry lands);
  Stage 3 S3/4, A2c + the oort stack (fluxtune only); Stage 4 T2/K6/T_mqtt; Stage 8 C1; Stage 9 budget/stop.
- **Modified** for variance-gated dynamic-K: K3a/K3b/K2/U3/K8/U2 → re-keyed to the variance-pass boundary /
  committed `data_id` (PARITY.md §F.3).
- **New** variance-cadence layer: V1–V5, DK1–DK3, G1–G2 (PARITY.md §F.4). Localize down; never fix an EMERGENT
  rung directly. `var_threshold` / `max_iterations_per_data_id` are baseline knobs, not parity levers.
- **Availability** rungs (A1–A5, A6/A7/A8/K11, withheld/abandon/starvation) inherited; apply once Phase-2
  wires the effect path + telemetry.

---

## §E  Roadmap — remaining phases

**Phase 1 (syn_0) — the sign-off run is banked (§A); close-out work:**
1. **#1c fluxtune R1** — root-cause + fix (top blocker).
2. **#13/#6 clock-rate** — confirm the eval fold fires; drive `sim_rate>1` and `wall_disparity`→~0.
3. **#7 fwdllm_plus** — explain real slowness + selection divergence from telemetry.
Then C1/C2 convergence at matched `data_id` per baseline → gate to Phase 2.

**Phase 2 — unavailability (syn_20/50/mobiperf).** Wire the ClientAvailability effect path into the grad loop:
send-time gate (real) / `delivery_ts = max(sct, next_avail)` buffering (sim); two ledgers; reactive-90s
in-flight; starvation vclock-advance; per-baseline `avail_select_filter`. Emit `EVENT_AVAIL_CHANGE` +
`agg_belief_change`/`send_gate_wait` (builders exist in `flame/telemetry/events.py`) → unlocks A6/A7/A8/K11.
**D3 watch:** does a withheld/late grad roll into `cached_v` on rollback, or a stale-version late grad inflate
`var` (and thus dynamic-K)? Over-instrument before trusting cadence. Pytest: adapt async_cifar10's availability
test patterns; syn_0 gate-ON-vs-OFF byte-identical. Exit: A1–A5 + A6/A7/A8 PASS; self-stops; withheld grads
delivered not dropped.

**Phase 3 — beyond syn_0.** Full ladder under scarcity; bin V1/V2 by run-fraction to separate a constant mix
bias from a compounding variance-feedback loop (§G). Exit: curves within tolerance at matched `data_id`; K8/U2
within bar; V1/V2 binned residual flat.

---

## §F  Locked principles (from async_cifar10, carried over)
1. **Sim does real forward-grad compute, charges modeled time.** GPU runs the real JVP; the agg stamps
   `sct = sim_send_ts + max(real_gpu_s, D) + leg`. **Never** put overhead on the vclock (`vclock = max(vclock, sct)`).
2. **Progress axis is `data_id`.** Updates-per-data_id is the dynamic-K random variable V1 validates — an output
   to match, not an input to assume.
3. **Variance is an emergent gate; localize, never tune it.** `var_threshold` / `max_iterations_per_data_id` are
   baseline-defining config knobs.
4. **Slot residence must survive variance rollbacks.** Release keys on the agg-goal boundary; the sct reorder
   buffer must not strand a grad across a rollback.
5. **DynamicKC: validate the input before the policy** (DK3 CONTROL before DK1/DK2). The controller is shared;
   don't fork it per baseline.
6. **Real is the reference only after admissibility.** A real↔sim gap has two fix directions — check whether the
   **real** input is the divergent side before tuning sim.
7. **syn_0 byte-identical gate OFF; unavailability config-gated.** Reuse the async_cifar10 flag axes; don't delete
   eligibility plumbing.
8. **Fix the concept, not the symptom — never regress a working example.** Classify a mechanism as
   **real-transport artifact** (incremental drain, `recv_fifo` timeouts, queue persistence — no sim analog; gate
   `and not self.simulated`) vs **algorithmic property**. Prefer the byte-identical-for-the-working-side fix.
   Scope-check before editing shared code: `fwdllm_aggregator.py` = fwdllm blast radius; `top_aggregator.py` /
   shared `parity` engine / `_sim_recv_min` can silently break async_cifar10.
9. **Match pytest scope to blast radius.** fwdllm-only edit → `pytest tests/mode -k fwdllm`; shared parity engine
   → add `examples/async_cifar10/scripts/parity` + `tests/mode -k parity`; shared stack (`flame/launch/*`,
   `flame/telemetry/*`, `config.py`, optimizer/selector, `top_aggregator.py`) → full `pytest tests/`. Full suite
   once after a cross-stack change, not before every run.
10. **Comment the WHY, crisply.** A conceptual/architectural choice, a real↔sim divergence + rationale, or a
    failure mode + why the fix takes its shape — one or two tight sentences. Deeper rationale → §K.
11. **Telemetry-FIRST, then instrument, then (rarely) run.** (a) Validate/refute a hypothesis from telemetry
    ALREADY ON DISK before considering a run — name the exact field/line first. (b) Ship telemetry + plot + pytest
    IN THE SAME CHANGE as any new mechanism. (c) A run is justified only to observe an EMERGENT quantity no stored
    telemetry can yield (convergence, a concurrency/wall number, a longer cadence trajectory) — then run the
    SHORTEST length that exhibits it, smoke first, one mechanism per round. (d) Cross-pollinate fluxtune↔sync
    baselines — fluxtune surfaces roots first.
12. **Consult PARITY.md vclock principles BEFORE any sim-clock change.** Settled rules: clock is a monotone `max`
    (`_advance_sim_clock = vclock = max(vclock, sct)`); NEVER put overhead on the vclock; `sct = send + max(gpu,D)
    + leg` (leg NOT on `trainer_speed`/utility); sync charges MAX-of-K sct, async the K-th-fastest; the sim SKIPS
    real waits and reconstructs order from sct (`SimReorderBuffer`).
13. **The vclock is virtual wall-time; the sim MUST produce SPEEDUP (`vclock ≥ physical_wall_elapsed`,
    `sim_rate = vclock/wall ≥ 1`).** fwdllm nuance: the forward-grad "train" is a REAL GPU pass (~2s) that MUST run
    in sim for grad mode-invariance — that GPU wall is irreducible; speedup comes from skipping transport/inter-round
    waits, NOT compute. `sim_rate<1` (as in the sign-off run, 0.26–0.85) means the sim under-charges the vclock or
    fails to skip a real wait. `sim_rate` is the top-line health metric — emit it every run (`[VCLOCK_PROGRESS]`).

**Open design decisions:** D2 (avail telemetry port — Phase 2); D3 (variance-cadence × withheld/late grads —
Phase 2); D4 (eval-delay factor — confirmed ~1× train cost, K-D3). D1/D6 resolved (see §K).

---

## §G  Fixes landed (what worked — one line each; do not redo)
- **#14 MQTT join-notify startup race (2026-07-04)** — `backend/mqtt.py` `join()` fired `notify(JOIN)`
  fire-and-forget; when the MainThread beat the async `on_connect`, the subscriptions AND the JOIN were dropped
  (notify's own `_is_connected` guard) with no retry → the end never registered → async_oort hung below
  `minInitialTrainers`. Fix: `join()` calls `_wait_for_connect()` (spin on `_is_connected`, 10s bound) before
  subscribe+notify; no-op when already connected → async_cifar10/all examples unchanged. Full `tests/` 879-green.
- **#1c abandon-timeout vclock-key (K-D26, part a)** — the shared `async_oort` 90s `SEND_TIMEOUT_WAIT_S`
  in-flight abandon-timeout was keyed on `time.time()` (physical wall); in a slow sim (wall ≫ vclock) it evicted
  still-outstanding trainers from `all_selected` → R1 re-dispatch. Now runs on the vclock in sim
  (`_abandon_clock_now`: stash `channel_props["vclock_now"]` per-select, stamp + check on it); None in real →
  `time.time()` → async_cifar10 byte-identical (its vclock≈wall). `test_sim_vclock_keeps_virtually_recent_end`.
  *(Sustained NONE-delete path is the same slow-sim root → #13.)*
- **#6 clock-rate anchor + Root B phase rungs (K-D25)** — root-caused #6 to a checker anchor (sim vclock is
  correct; real's full wall carried a ~7.7s/round transport artifact). Agg emits `intrinsic_span_s` (barrier =
  max committed intrinsic duration + eval, mirroring the sim vclock composition, **fedavg excluded**); the 5
  clock-rate rungs (K2/K3/K3b/K8/U2) + `wall_disparity` anchor real on the cumulative intrinsic clock, `ts`
  fallback → async_cifar10 byte-identical. Root B: `post_train_s` stamped after the delay (pure post-proc); B2
  straggler moved from `_delay_s` into the sct only (`training_budget_s` = base delay). Validated on banked data
  (throughput 0.47→0.048, wall_disparity 90→1.6, post_train 0/0, budget 1.14/1.14); full `tests/` 878-green.
- **Phase 4 stopping rule + folds (K-D24)** — `sim_wall_ceiling_s` decoupled to `max_runtime_s × 20` (fixed root
  S1, no more wall-truncation of a real-compute sim); B1 `sim_model_eval_time` + B2 `sim_straggler_spread_s:0.9`
  ON in the 3 sim yamls; matched `--max-data-id` is the primary stop. *(Sign-off run confirms S1 fixed; folds
  did NOT yet lift `sim_rate>1` — see #13.)*
- **Phase 2 speedup-leak skips (K-D23)** — gated the trainer per-round `pause_execution` `sleep(1)` throttle +
  `_check_availability` avail-spin `and not self.simulated` (fwdllm-only). The trainer `recv`/`await_join` and
  agg grace/`await_join` are LEFT AS-IS (barrier-wait over irreducible agg eval+GPU, not skippable sleeps).
- **Pre-run instrumentation A–E (K-D21)** — Stage-C fwdllm_plus scarcity-wait liveness; A1 trainer phase timing;
  A2 agg wall-decomp; A3 advance-rung re-key to `data_id` (shared engine, async byte-identical); A4
  `wall_disparity` DIAG rung; B1/B2/B3 sct folds (config-gated); D1 `task_recv.sim_send_ts`; D2 K5 real-compute
  exemption; E `sleep(0.1)`-pad gating. Un-skipped the 12 rigor-gap rungs (#10).
- **Availability params end-to-end (K-D22)** — Phase-1 defaults to syn_0; the pre-flight table reads back the
  RESOLVED config (print == run); feasibility gate blocks a full-participation sync barrier under a non-syn_0 trace.
- **Trainer sim clock (Batch 1, K-D2/K-D3)** — additive `sct = send + gpu + D`, no-sleep on the sim path.
- **Async grad loop (Batch 1, K-D4/K-D5)** — `_sim_recv_min_grad` sct reorder buffer + in-flight gate +
  agg-goal-boundary rollback cleanup.
- **Sync barrier (K-D11)** — `_sync_sim_recv_first_k` first-k-smallest; the `ends_not_selected_yet` "commit-1-per-
  pass" clamp gated real-only so the sim drains the full dynamic-K cohort in one pass.
- **Variance-cadence rung layer (Batch 2, K-D9/K-D10)** — V1–V5 / DK1–DK3 / G1–G2 + the pre-mutation
  `cycle_data_id` snapshot that makes V1 exact; DK3/G1 emit deferred with a logged SKIP.
- **Residence: commit-then-carry + R1/W1 rungs (Batch 2.5, K-D12/K-D14)** — killed the 2× recompute; R1 sourced
  from an echoed per-contribution interval (immutable across re-dispatch). fwdllm/fwdllm_plus R1 exact 0%.
- **Slot residence realigned to felix (K-D17/K-D17b)** — sim drain keys on `_sim_buffer` not channel RECV; the
  re-pick triplet stamped on grad RETURN not dispatch; hold the compute slot to COMMIT so `len(selected_ends)` =
  virtual-time in-flight (fixed the in_flight mis-measurement, 2.7→9.5 vs real 9.75).
- **staleness_policy wired from config (K-D15/K-D13)** — was silently `none` for every run; fluxtune set to
  `fedbuff` staleness-weighted accept (real+sim identical, a definition not a lever).
- **Clock family re-keyed to `data_id` (#2)** — `_progress_axis`/`_per_progress_last_event`; async_cifar10
  auto-detects `round` → byte-identical. field_coverage accepts fwdllm field aliases (#3). W1 made asymmetric
  (flags only sim EXCESS). Launcher fail-fast on early aggregator death (D-d).

---

## §H  Dead-ends & corrections — do NOT retry
- **"K-D19 fixed fluxtune R1 / the NONE reset is a red herring."** REFUTED (K-D26). R1 stayed 60.4%. K-D19
  fixed the wrong release path (the aggregator's RETURN-path `cleanup_provided_ends`) and dismissed K-D17b's
  NONE hypothesis by checking the *selection filter* (async_oort:1586, which excludes `all_selected` members) —
  but the guard fails because `all_selected` MEMBERSHIP is deleted first, by (a) the 90s wall-timeout and (b)
  async_oort reading the aggregator's `KEY_END_STATE=NONE` slot-hold as "trainer left." K-D17b was right.
  *Lessons:* (i) an R1 fix is not done until the emergent smoke banks R1≤2%; (ii) when a guard "should exclude"
  but doesn't, check whether the member is being *deleted* upstream, not just whether the filter reads it.
- **"K-D16 Option-A: free the compute slot on physical RETURN."** WRONG for virtual time — a returned-but-
  uncommitted trainer is still in flight until the vclock reaches its sct; freeing the slot undercounted
  `in_flight` 3×. Reverted to hold-to-COMMIT (felix-aligned, K-D17b).
- **"K-D16 fixed fluxtune; the re-run just banks numbers."** WRONG — it DEADLOCKED (K-D17): the sim drain was
  gated on channel RECV state (not the buffer), and the re-pick triplet was stamped at DISPATCH (freezing the pool
  before the first commit). The guard belongs at RETURN; the sim commit path keys on its OWN reorder buffer.
- **K-D6 "drop stranded grads at the agg-goal boundary."** REVERSED for async (K-D12) — the "|selected|≈agg_goal"
  premise holds only for sync; fluxtune (c=10≫agg_goal=3) dropped ~7 grads/cycle → residence violation → 2× passes.
  Commit-then-carry replaced it. Drop stays correct for sync.
- **"sim wall ≈ real, comparable."** WRONG — always check avg in-flight concurrency, not just total pass counts
  (the concurrency collapse hid behind a similar pass count).
- **"D=0 smoke, so clock-family fails are artifacts."** WRONG — the runs are D>0; the fails traced to the rung
  being keyed on `round` not `data_id` (#2).
- **Tuning `var_threshold` / `max_iterations_per_data_id` to close a cadence gap** — rejected; baseline-defining
  knobs, not parity levers. A cadence gap is an upstream set/order/clock divergence.
- **Adding a scalar overhead to the vclock to close #6** — forbidden (principle #1/#12); the vclock is
  `max(vclock, sct)`. Fold only genuine unmodeled compute terms (eval_s, straggler spread); artifacts stay off.
- **"Drive `wall_disparity`→0 by folding more terms into the vclock" (K-D24's #6 plan).** WRONG target — the
  eval fold was the LAST genuine term; the residual gap was real's ~7.7s/round localhost transport ARTIFACT, so
  folding further would over-charge the vclock (principle #1). #6 was a CHECKER-ANCHOR problem (K-D25): anchor
  real on its intrinsic algorithmic clock, and retarget `wall_disparity` to |sim_vclock − real_intrinsic|.
- **Including the FedAvg merge in `intrinsic_span_s`.** Made real intrinsic overshoot the sim vclock by
  ~fedavg×cycles (wall_disparity 1.6→5.5); the sim vclock charges barrier+eval but NOT fedavg, so intrinsic must
  mirror that composition. (Re-add only if fedavg is ever folded into the sim vclock.)
- **Subtracting the slept delay from `post_train_s` to exclude it (Root B first attempt).** Went NEGATIVE when
  the delay wall wasn't in the post window (fake-time tests / sim never sleeps). Correct fix: stamp
  `_phase_post_start` AFTER the delay so post_train is pure post-proc, never negative.

---

## §K  Deviation log — one line per decision (anchor + rationale; referenced from §B.1)
*Folded/landed entries are one line. Expand only an entry still open/contested.*

- **K-D1** — `time_mode` default `"real"` (not cifar's `"simulated"`): fwdllm's whole corpus is `real`; a
  `simulated` default risks half-activating an unbuilt path. Sim variants set it explicitly.
- **K-D2** — additive `sct = send + gpu + D` (not `max`): fwdllm's real mode sleeps D on top of GPU time.
- **K-D3** — per-eval sct collapses to the train sct (D4): eval lives on the aggregator; forward-grad "train" is a
  forward pass (no 20× factor); the trainer eval message is a utility report, not a clocked commit.
- **K-D4** — purpose-built `_sim_recv_min_grad` (not `_sim_recv_min` verbatim): the latter's per-commit release +
  withheld paths key on WEIGHTS semantics; reuse the primitives, fork the orchestration.
- **K-D5** — slot release + buffer clear on the AGG-GOAL boundary (not per-commit): a `data_id` spans many cycles
  with rollbacks; per-commit release would strand a re-contributing trainer (principle #4).
- **K-D6** — boundary buffer-drop DROPPED stranded grads. *Superseded by K-D12* (benign only for sync).
- **K-D7** — U6 sync-barrier visibility-lag telemetry: compute+stash now, emit later (kept Batch 1 a clock port).
- **K-D8** — sim smoke launchers keep D=0 for mechanics-only comparability; D>0 belongs to convergence/parity runs,
  enabled in BOTH real and sim together (else mismatched enable is a false divergence).
- **K-D9** — cadence telemetry is a **pre-mutation** cycle snapshot (`cycle_data_id`/`cycle_iteration`/
  `grad_pool_size`/`cached_v_size`): the post-mutation `data_id` advances before emit → off-by-one; snapshot makes
  V1 exact for natural-pass and force-commit paths. Existing post-mutation fields unchanged (byte-identical real).
- **K-D10** — DK3 / G1 emit DEFERRED (checker reads-if-present, logged SKIP otherwise): DynamicKC disabled for all
  three baselines; G1 needs a trainer-side per-update emit. Wire when a DynamicKC/G1 baseline is added.
- **K-D11** — `ends_not_selected_yet` "commit-1-per-pass" clamp gated real-only: it's a real-transport draining
  discipline (queue re-collection); the sim barrier is single-pass (principle #8).
- **K-D12** — fluxtune async: commit-then-CARRY the surplus + hold residence (reverses K-D6 for async); drop stays
  correct for sync (c≈agg_goal).
- **K-D13/K-D15** — fluxtune `staleness_policy = fedbuff` staleness-weighted accept (was silently `none` — never
  wired from config); set identically real+sim (a definition, not a lever). Provisional lock.
- **K-D14** — R1/W1 sourced from an ECHOED per-contribution interval (not the agg's per-end dispatch stamp, which
  is overwritten on re-dispatch — exactly when residence is broken).
- **K-D16** — D-e resolution Option A (two-lifetime split). *Slot-on-return half superseded by K-D17b; the
  hold-guard-to-COMMIT half survives.*
- **K-D17** — two bugs that deadlocked the K-D16 re-run: (A) drain gated on channel RECV not `_sim_buffer`;
  (B) triplet stamped at DISPATCH froze the pool pre-commit. Fixed: drain keys on the buffer; triplet on RETURN.
- **K-D17b** — hold the compute slot to COMMIT (felix-aligned): `len(selected_ends)` now = virtual-time in-flight
  (fixed in_flight 2.7→9.5). Reverts K-D16's slot-on-return.
- **K-D18** — overnight 10-`data_id` grounding runs; `--max-data-id` default raised to 9999 (was 10, silently
  capping 1h runs); cleared #4 truncation for fwdllm.
- **K-D19** — R1 regression fix: defer the re-pick guard release from physical RETURN to COMMIT
  (`_release_end_on_return`, fwdllm-only) + a `[SIM_R1_DISPATCH]` tripwire + RETURN-path pytest. **⚠ REFUTED by
  the 2026-07-04 sign-off smoke** — R1 still 60.4% (§H dead-ends, #1c reopened); the guard leaks despite the defer.
  Root-cause from the tripwire lines next.
- **K-D20** — real per-round wall decomposition (telemetry-only): ~8.7 s/round GENUINE (eval_s 3.34 + straggler
  3.06 + compute) vs ~4.7 s/round ARTIFACT (drain tail 2.48 + `sleep(0.1)` pads) → fold ONLY eval_s + straggler
  spread; artifacts stay off the vclock and become optimization targets (#11). Root-caused #7 (real-mode liveness).
- **K-D21** — pre-run instrumentation A–E landed & pytest-green (see §G).
- **K-D22** — availability params respected end-to-end; Phase-1 syn_0 default; print==run (see §G).
- **K-D23** — Phase-2 speedup skips; the "trainer fetch is the dominant leak" hypothesis was REFUTED by telemetry
  (it's barrier-wait over irreducible agg eval+GPU); the real skippable wall was two per-round sleeps (see §G).
- **K-D24** — Phase-4 ceiling decouple (×20) + B1/B2 folds ON for the sign-off run (see §G). Sign-off confirmed S1
  fixed. *(#6-via-folds superseded by K-D25 — the residual gap was a checker anchor, not a missing fold.)*
- **K-D25** — #6 is a CHECKER-ANCHOR bug, not a sim under-charge (root-caused from the sign-off #6 telemetry).
  The clock-rate rungs compared sim-Δvclock (genuine algorithmic time) against real-Δts (FULL wall = genuine +
  a ~constant ~7.7s/round localhost transport artifact real trainers pay via pipelined inter-round fetch; the
  agg `barrier_wait_s` reads ~0 in real for the same pipelining reason). Proof: on a non-eval round real burns
  10.9s wall vs the sim's 3.5s genuine, all artifact; on eval rounds real genuine (barrier+eval 15.8) ≈ sim
  vclock 15.86 exactly. Fix (fwdllm agg emit + shared checker, async byte-identical): emit `intrinsic_span_s` =
  max(committed intrinsic duration) + eval (mirrors the sim vclock; **fedavg excluded**, else overshoot); the 5
  clock-rate rungs + `wall_disparity` anchor real on its cumulative intrinsic clock. **Root B (bundled):**
  `post_train_s` stamped after the delay (real stops bundling the slept delay); B2 straggler moved from
  `_delay_s` into the sct so `training_budget_s` is a mode-invariant input. `TestIntrinsicSpanAnchor` + emit +
  straggler-separation + post_train tests; full `tests/` 878-green. Residual (3 marginal rungs + B2 calibration)
  is emergent at the longer sign-off run. #11 real-side waste deferred (zero parity impact; needs a real run).
- **K-D26** — #1c R1 root-caused to a physical-wall vs vclock desync in the SHARED `async_oort` re-pick guard
  (`all_selected`), exposed by fluxtune's slow sim (sim_rate 0.26, wall ≫ vclock) and masked in async_cifar10
  (wall ≈ vclock). Evidence (banked sign-off log, principle #11a): trainer 370 dispatched at vclock ~0, its grad
  commits at vclock ~7, but the sim burns ~99s wall first → at 91s wall async_oort's 90s `SEND_TIMEOUT_WAIT_S`
  (keyed on `time.time()`) evicts it from `all_selected` → re-dispatch → R1. Two deletion paths: **(a)** the 90s
  wall-timeout (round-1 trigger, 10 evictions); **(b)** the aggregator marks carried grads `KEY_END_STATE=NONE`
  to keep their `selected_ends` slot (`_sim_hold_busy_slots`), and async_oort's recv-handler reads `NONE` as
  "left/rejoined" → deletes from `all_selected` (sustained, 113 violations). Corrects K-D19 (fixed the RETURN
  path; dismissed the NONE hypothesis by checking the selection FILTER not the DELETION) — K-D17b was right.
  **Fix (a) LANDED** (operator: "both — direct fix now + #13 tracked"): `_abandon_clock_now()` runs the timeout
  on the vclock in sim — `select()` stashes `channel_props["vclock_now"]`; the dispatch STAMP and the timeout
  CHECK both use it (90 = 90 *virtual* s); None in real → `time.time()` → async_cifar10 byte-identical.
  `test_sim_vclock_keeps_virtually_recent_end` + reclaim regressions green. **Fix (b)** = the sustained NONE path,
  same slow-sim root → cured by **#13** (make the fluxtune sim fast so physical receipt ≈ virtual commit, as in
  async_cifar10) or a follow-up aggregator-side re-assertion; the emergent re-run measures how much (a) recovers.
- **K-D27** — #1c R1 DEEPER root-cause from the post-#14 `_2327` telemetry (part (a) recovered ~0; R1 still
  62.9%, 215 `SIM_R1_DISPATCH`). The NONE-delete (K-D26 part b) is one symptom of a **two-ledger split**:
  async_oort decides selection eligibility SOLELY from `all_selected` (`filtered_ends`, async_oort:1607), a
  ledger it prunes on PHYSICAL events (the recv-fifo 2s re-select loop, RECVD/NONE cleanup) — and NEVER consults
  the aggregator's VIRTUAL in-flight truth (`_sim_inflight_expected ∪ _sim_buffer`). The intended bridge
  `_agg_pending_commit_ref` is read for telemetry (async_oort:423) but **never assigned** (∅) and never filtered
  on; `_sim_pending_commit` is accumulated (fwdllm_aggregator:908) but never cleared or bridged (felix's asyncfl
  DOES maintain it, asyncfl:618/1490). fwdllm ported felix's `_sim_hold_busy_slots` (mirror the truth INTO
  all_selected) but not the selector-side exclusion, so any physical-schedule prune in the (large, slow-sim)
  returned-but-uncommitted window frees a still-in-flight trainer → own select() re-dispatches → R1. Real/
  async_cifar10 mask it via wall≈vclock (window≈0), not by design. **Telemetry proof:** `num_eligible=10` while
  `in_flight=10`; `in_pending_commit=0/879` (dead hook); `[CHANNEL_CLEANUP] freed=0`×218; 0 fires of the 90s /
  removed-ends prune paths (part (a) confirmed dead). **Fix (proposed):** wire `_agg_pending_commit_ref` to the
  aggregator's virtual in-flight set (add on dispatch, discard on COMMIT) + exclude it in `filtered_ends`; sim-
  only (∅ in real → async_cifar10 byte-identical). Controlling metric: `SIM_R1_DISPATCH → 0`, R1 overlap → ~0.2%.
