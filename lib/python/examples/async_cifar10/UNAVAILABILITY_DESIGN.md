# Sim Unavailability -- Design

> **DEPRECATED — reference only.** The single source of truth is [ROBUST_FL_READINESS.md](../_metadata/ROBUST_FL_READINESS.md) → [FELIX_READINESS.md](../_metadata/FELIX_READINESS.md). This file only gets trimmed from here on as its content moves there; don't add to it.

Design reference for modeling client **unavailability** in the FLAME FL simulator (v1 substrate, all six
async_cifar10 baselines).

---

## Preamble -- what this is

**Goal.** Model client *unavailability* (devices dropping offline mid-training) so a fast **simulated**
run (virtual clock, no real sleeps) reproduces what a **real** run (wall-clock, MQTT, true delays) does --
**sim/real parity** -- for every baseline, **config-gated and default-OFF** (byte-identical to today when off).

**What was built (v1).** A shared availability substrate (`flame/availability/trace.py` +
`ClientAvailability`) mixed into the syncfl base and inherited by asyncfl/oort, so all baselines share one
trace-read effect path:
- **Send-time gate, deliver-late-stale.** A trainer that goes UN_AVL mid-flight *keeps computing*; its
  upload is gated at send-time (real) / buffered to `delivery_ts = max(sct, next_avail)` (sim) and
  committed later as a stale update. Nothing is cancelled or dropped.
- **Two ledgers, never conflated.** Slot ledger (90s vclock *abandon* frees the in-flight slot) + delivery
  ledger (`pending_withheld[end]=delivery_ts`, commits through the existing staleness gate).
- **Proactive in-flight eviction** -- **felix only** (the one fully-aware baseline): frees a slot the trace
  shows UN_AVL at the next selection boundary, no 90s wait.
- **Starvation / vclock-advance under scarcity.** When the eligible pool is too small to start a round, sim
  advances the vclock to the next availability transition instead of spinning (self-terminating, B2.0.2).
- **Absolute (vs. ground-truth-trace) fidelity checks** on top of the relative (real-vs-sim) ones: A6
  (trainer state), A7 (aggregator belief, per selection/commit checkpoint), A8 (send-gate wait), K11
  (commit promptness). Relative checks answer "do the two modes agree?"; these answer "is either one
  *correct*?" -- shared `scripts/parity/ground_truth.py`, one canonical trainer<->aggregator time origin.
- **Parity ladder** (`scripts/parity/`) -- availability rungs A1/A3/A4/A4dur/A5/A6/A7/A8/K11,
  withheld_delivery, abandon_timeout, starvation_advance, eligible_pool_reduction.

**Two orthogonal axes per baseline (keep separate).**
1. **Knowledge at selection** (`avail_select_filter`): does the selector read the trace to avoid
   *selecting* currently-UN_AVL trainers? aware = yes, unaware = select blind.
2. **In-flight slot-free timing** (`proactive_inflight_evict`): when a *dispatched* trainer goes UN_AVL
   mid-round, free its slot at the next boundary (proactive, felix only) or wait the 90s vclock abandon
   (reactive-90s, everyone else). Aware-at-selection != in-flight eviction.

The knowledge *model* is **trace-read** for all v1 baselines; message-transport (`client_notify`) and
predictive models are **Stage H** (future).

---

## Baseline matrix (CANONICAL)

| baseline | sync/async | agg base / entry | knowledge @ selection (`avail_select_filter`) | in-flight slot-free (`proactive_inflight_evict`) | config-gate (as run) |
|---|---|---|---|---|---|
| **felix** | **async** | `asyncfl` (<- syncfl) / `main_asyncfl_agg.py` | aware | **proactive** (felix only) | `simUnavailability` |
| **fedbuff** | **async** | `asyncfl` / `main_asyncfl_agg.py` | unaware | reactive-90s | `simUnavailability` |
| **oort** | **sync** | `oort/top_aggregator` / `main_oort_sync_agg.py` | unaware | reactive-90s | `simUnavailability`+ |
| **oort_star** | **sync** | `oort/top_aggregator` / `main_oort_sync_agg.py` | aware | reactive-90s | `simUnavailability`+ |
| **refl** | **sync** | `syncfl` FedAvg / `main_fedavg_agg.py` | aware | reactive-90s | `simUnavailability`+ |
| **feddance** | **sync** | `syncfl` FedAvg / `main_fedavg_agg.py` | aware | reactive-90s | `simUnavailability` |

+ oort/oort_star/refl's `baselines.yaml` catalog entry still carries the legacy `trackTrainerAvail:
{enabled: True, type: ORACULAR}` block (pre-dates this project). In every `debug_run.sh --trace`-launched
run the substitution sets `simUnavailability=True` for them too, so they run the same modern path. The
legacy block only matters if the parity YAML is loaded *without* that substitution -- untested territory,
and the reason it isn't cleaned up yet (removing it risks silently disabling availability in that path).
See FELIX_READINESS FX-N7.

**Notes.** (1) `ClientAvailability` lives in `flame/availability/client_availability.py`, mixed into
`syncfl/top_aggregator.py` (`class TopAggregator(ClientAvailability, Role)`); asyncfl/oort extend it -- all
six share the substrate. (2) trace-read is v1; the knowledge model becomes message-transport / predictive
in Stage H, but the select-filter / in-flight-evict *behavior* is unchanged. (3) **felix is the only
baseline that de-selects an in-flight trainer** when it goes UN_AVL; the aware-at-selection-only baselines
still hit the 90s abandon for mid-round drop-offs.

### Flag reference
- `avail_select_filter: bool` -- selector excludes currently-UN_AVL trainers from the **selection** pool
  (`get_curr_task_ineligible_trainers`). ON: felix/oort_star/refl/feddance. OFF: oort/fedbuff.
- `proactive_inflight_evict: bool` -- gates `_sim_evict_unavail_inflight` (in-flight boundary eviction).
  ON: **felix only**. OFF: everyone else (reactive-90s).
- `tracking_mode` -- knowledge-model axis: `trace_read` (v1, live) | `client_notify` (Stage H) |
  `predictive` (future). Replaces the `oracular` value at concept/log level (YAML field *value* compat kept).

---

## v1 core decisions (resolved -- durable reference)

- **Knowledge model:** trace-read for all; one shared trace + `state_at(trainer, vclock)` + one effect path.
- **Mid-flight UN_AVL = compute-completes, gate the send, deliver-late (stale).** Real: gate at send-time.
  Sim: buffer at `delivery_ts = max(sct, next_avail_ts)`.
- **Two ledgers, never conflated:** slot ledger (frees `selected_ends`) + delivery ledger
  (`pending_withheld`, commits stale through the staleness gate). Order commits by `(delivery_ts, end_id)`,
  never `sct`.
- **Busy != unavailable != withheld** -- three distinct non-pool states. Never route busy->UN_AVL.
- **All availability time on the vclock in sim.** Never wall, never a frozen per-trainer clock.
- **Config-gated, default OFF** => byte-identical. `simUnavailability` is the gate for all 6 baselines as
  run (see Baseline matrix + note).
- **availability state** (`AVL_TRAIN/AVL_EVAL/UN_AVL`) x **busy?** x **has in-flight update?** are
  orthogonal. `syn_0/20/50` are 2-state (no AVL_EVAL); `_trace_has_avl_eval` guard collapses D.2 for them.

---

## Parity rungs (availability tier -- what exists)

- **A1** `avail_composition` (per-state counts, binned). **A3** `trace_time_base_consistency` -- CONTROL
  hard gate (dep K3). **A4** `per_trainer_duty_cycle`. **A4dur** duration-weighted TVD
  (`mean_err<=0.05`, `frac_within_tol(0.10)>=0.95`). **A5** `state_timeline_agreement` -- per-(trainer,t)
  exact match. All five are **relative** (real vs sim).
- **A6** `trainer_trace_fidelity`, **A7** `agg_belief_fidelity` (tagged `selection`/`commit`), **A8**
  `send_gate_wait_fidelity` (real-mode only) -- **absolute** (vs. ground-truth trace), independently per
  mode. A7 is mechanism-agnostic (trace_read now, client_notify/predictive later).
- **K11** `commit_promptness` (INV) -- per-event: actual commit vs. earliest-legally-committable time,
  generic over gate reason; primary promptness gate. `withheld_delivery` / `abandon_timeout` stay as
  secondary distributional diagnostics.
- **eligible_pool_reduction** (`Aa`, HELD), **observation_lag** (live in v1 trace_read via A7),
  **starvation_advance** (vclock jumps under scarcity). Calibrate HELD rungs at mobiperf.
- **Ramp:** syn_0 -> syn_20 -> syn_50 -> mobiperf_*.

---

## Challenges / land-mines (durable -- consult before the fwdllm port)

Most are resolved; the still-open ones (3, 15) matter for cross-baseline validation and the port.

1. Ordering on `delivery_ts`, not `sct` -- resolved (U6/U3 validated).
2. A3 time-base drift -- resolved; hard CONTROL gate; 90s abandon re-clocked to vclock.
3. **A2 two-tolerance trap (OPEN, watch).** Bimodal sim vs smoother real -> KS shape artifact; means match;
   improving with run length (0.437->0.338). Expect <=0.2 at n=300/3h -- confirm in FX-N9.
4-14. Resolved: busy/unavail/withheld three ledgers (4); real send-gate fidelity (5); determinism via
   `(delivery_ts,end_id)` (6); compound straggler x UN_AVL (7); AVL_EVAL inert for oort + `_trace_has_avl_eval`
   guard (8); staleness-on-sync cohort movement (9); scarcity advance skips no events (10); syn_0
   byte-identity discipline (11); library mixin spans examples, never example-local (12); empty per-task
   pool corrupting `selected_ends` -- fixed by keying cleanup off `connected_ends` in all 3 selectors,
   **still needs live mobiperf_3st exercise** (13); scarcity threshold via F.2 unified pattern (14).
15. **Per-baseline in-flight accounting + scenario sizing (OPEN, per-baseline).** In-flight is NOT constant:
    oort (sync, over-selects) `in_flight ~= overcommitment*agg_goal - completed`; felix/fedbuff (async)
    concurrency-bound, can exceed agg_goal; refl/feddance (sync FedAvg) clear `selected_ends` each round,
    feddance returns *partial* selections so `eligible ~= (1-unavail)*n`. Manage in-flight per baseline; do
    NOT assume "sync has no in-flight term." syn_50 caps ~43% unavail, so feddance's straddle window is
    narrow (n~19) -- size `n ~= threshold / (1-unavail_frac)`.
16-18. Resolved: real syncfl recv-barrier bounded `timeout=min(90s,budget)` (B2.0.1, 16); sim starvation
   self-termination `>`->`>=` budget check (B2.0.2, 17); real-mode trace-clock join-ramp re-anchor
   (B2.0.3, 18) -- correct, but was masking item 20.
19. Resolved (Batch 4): oort/felix `K6 sim_send_ts` -- sim `Trainer._sim_now()` froze at last dispatch;
   fixed via due-ts stamping + EOT final wake-up. K6/A6 PASS on the felix n=300 run.
20. Resolved (Batch 3 T3.1a): `debug_run.sh` never wired the trainer's own `client_notify.trace`, so real
   trainers ran their send-gate against always-available `syn_0` regardless of `--trace` (sim was
   aggregator-driven, so unaffected). Fixed + regression-tested
   (`tests/launch/test_debug_run_trace_substitution.py`). This was the true cause behind the feddance A3/K3b
   symptoms earlier blamed on clock-origin.

---

## Stage H (future -- out of scope)

Two independent knowledge-model upgrades, both replacing `trace_read` on the `tracking_mode` axis; the
effect logic (select-filter / in-flight-evict) is unchanged -- only how the agg learns state changes:
- **H.1 Message-transport** (`client_notify` ON for aware baselines): trainers push avl-state changes over
  MQTT instead of the agg reading the trace + a continuous/event-scheduled vclock clamp. Re-measure
  `observation_lag` (must be ~0) once live. (fwdllm's fluxtune baseline already configures `client_notify`
  -- see simulate_fwdllm.md D1.)
- **H.2 Predictive**: a learned/heuristic availability model (no trace read or message push). Not designed yet.

---

## History (collapsed -- full detail in git)

- **A-G, C.6, D, E, F.2, B2.0.x, T0-T5, Batch 2/3/4** all landed. Substrate + A3 time-base CONTROL,
  send-gate/deliver-late, two-ledger, proactive evict (felix), syncfl path, starvation self-termination,
  the absolute ground-truth fidelity checks (Batch 3 T3.0-T3.5: A6/A7/A8/K11 + `ground_truth.py` + shared
  canonical time origin), and Batch 4's four gating fixes (asyncfl real self-stop, sim trainer-clock freeze
  / K6 / A6, A7-commit checker `max_gap_s`) are all in code + tests, confirmed on the felix n=300 run.
  Per-task mechanism/file/exit detail lives in commit history; the durable decisions are in "v1 core
  decisions", "Challenges", and "Dead-ends" above.
