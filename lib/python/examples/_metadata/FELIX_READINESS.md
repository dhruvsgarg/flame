# Felix readiness — backprop FL (async_cifar10, google_speech)

## Preamble
- **Read [ROBUST_FL_READINESS.md](ROBUST_FL_READINESS.md) first.** Its doc rules, operating rules (R1-R13),
  shared lessons (L) and tripwires (T) apply here and are not repeated. This doc holds only what is specific
  to Felix: weight-aggregating backprop FL, round/`agg_goal` progress axis, Oort-family selectors.
- **CURRENT FOCUS.** Goal: Felix feature-complete (syn_0 + unavailability, both datasets) and paper
  experiments running sim-only.
- **Scope:** `felix`, `oort`, `oort_star`, `refl`, `feddance`, `fedbuff` (+ each one's `*_oracle` arm for the
  streaming experiment, FX-N13). Out of scope: `fedavg`, `oracle`.
- **Traces (both datasets, both papers):** `syn_0`, `syn_20`, `syn_50` (synthetic) and `mobiperf_3st` (the
  real-world 3-state trace).
- **IDs:** `FX-N` next steps · `FX-L` lessons · `FX-T` tripwires · `FX-D` done. Shared work is `S#` in the parent.
- **Reference (read only for detail):** rung catalog and derivations → [PARITY.md](../async_cifar10/PARITY.md)
  §1-§5 · checker internals → [PARITY_CHECKER_README.md](../async_cifar10/scripts/parity/PARITY_CHECKER_README.md)
  · availability design → [UNAVAILABILITY_DESIGN.md](../async_cifar10/UNAVAILABILITY_DESIGN.md) · streaming
  experiment → [EXPERIMENT_felix_streaming.md](../async_cifar10/docs/EXPERIMENT_felix_streaming.md) · identity →
  [BASELINES.md](BASELINES.md) (Felix section).

---

## Status grid

✅ parity at HEAD · 🟡 last good on old code (stale) · ⚠ open failures · ⬚ not started. Score = enforced
passing / total, with run length.

**async_cifar10** (n=300, α=0.1)

| baseline | real | sim syn_0 | sim syn_20 / syn_50 | sim mobiperf_3st |
|---|---|---|---|---|
| felix | ✅ | 🟡 46/46 (3h) — the §S.pacer fix landed after it | ⚠ 62/62 (syn_50) is INVALID: real dropped withheld updates (FX-D5) | ⬚ |
| oort | ✅ | 🟡⚠ 42/46 (1.5h) — Sd: pacer input signal | ⚠ A2 / K3b / P3 / throughput | ⬚ |
| oort_star | ✅ | ⬚ | ⬚ | ⬚ |
| refl | ✅ | 🟡 44/46 (3h) | ⬚ | ⬚ |
| feddance | ✅ | 🟡 43/44 (3h) — C2 loss only | ⚠ A2 / U5 | ⬚ |
| fedbuff | ✅ | ⬚ | ⚠ run spoiled by a concurrent run | ⬚ |

Every real leg under unavailability before FX-D5 dropped its withheld updates, so all syn_20/50 cells
are re-run items. Everything is 🟡 because PRs #72-#85 rewrote code these runs depended on: `async_oort.py` (+ new
`async_base.py`), `fedbuff.py`, both `top_aggregator.py`, `syncfl/trainer.py`, `channel.py`, the checker.
Stored grades: `async_cifar10/experiments/parity_{felix,oort,refl}_20260624_{5400,3h}.json`,
`parity_feddance_20260623_3h.json` (run dirs are named inside each JSON).

**google_speech** — all six ⬚. Only legacy 2024 JSON configs (`async_google_speech/aggregator/*.json`); not
on the launcher, no sim support.

---

## Next steps (persistent queue — top item is next)

- **FX-N1 · Read the harness campaign (`experiments/campaign_<launch ts>/SUMMARY.txt`) · blocked: operator run.** Run 1
  was void (pre-spawn GPU probe; aggregator `harness_mode` read before init; trainer cwd-relative LEO path; P0
  import-time `sys.exit`) — all fixed; the 120s felix smoke now runs clean (real 13/13; sim EV5/EV10/EV11 = FX-N14). P1 syn_0 x6 · P2 syn_50 x6 · P3 mobiperf_3st x6 (first live 3-state run ever) · P4 simColdStartGate
  A/B · P5 syn_20 · P6 tiny_cpu. Triage every EV FAIL as checker-gap vs bug (FX-D5 method); then the injected-
  bug checks (parent S1 exit).
- **FX-N14 · Sim one-in-flight leak under unavailability · todo.** EV10 on the stored Jul-2 syn_50 sims:
  felix re-dispatched 6/8590 and fedbuff 2/8030 train tasks to a trainer whose update had ARRIVED but was not
  yet committed; both updates commit later (0 at syn_0). Find the path that frees the slot between arrival and
  commit when the trainer is withheld/evicted; the harness P2/P3 legs should reproduce it.
- **FX-N15 · fedbuff sim diverges to NaN at syn_50 · todo.** Stored Jul-2 sim: test-loss 8.3 at round 550,
  NaN from 600 (EV14). Training health, not timing: check fedbuff's `agg_rate_conf` (`old`) staleness weight
  against the withheld-delivery staleness (~160) it now sees.
- **FX-N16 · Trainer availability thread crashes at shutdown · todo.** `notify_trainer_avail` outlives the
  channel (EV0 WARN, 6 per syn_50 run). Stop the thread on shutdown; then drop EV0's benign-signature case.
- **FX-N2 · Parent S2 (parity pipeline) for async_cifar10 · todo.** *Exit:* re-grading the stored Jun 23-24
  pairs through the new pipeline reproduces the old scores to within the floor, or each difference is
  explained.
- **FX-N3 · A/B `simColdStartGate` on the harness, then on felix · todo.** The flag closes the first-contact
  blind spot in both the commit gate and the busy-slot hold (FX-D4). Harness: syn_0 felix + fedbuff, off vs
  on; compare `cold_start_holds`, past-dating and R1 overlap. Promote into the felix/fedbuff parity YAML only
  on evidence (R9). Remaining audit: one-instruction-per-version on async dispatch (simulate_fwdllm.md §F 25).
- **FX-N4 · syn_0 cluster block, async first: felix, fedbuff · todo.** Real n=3 + sim n=3 at one length and
  commit → floors → grade. A 90-min block for mechanisms, then 3h for sign-off. *Exit:* INV/EXACT green,
  convergence inside the replicate band, DIST residuals common-mode.
- **FX-N5 · syn_0 cluster block: oort, oort_star, refl, feddance · todo.** Same protocol. oort's open root:
  per-round `relative_change` of the exploited utility, binned by quartile, in both modes (don't touch the
  pacer). refl: confirm at 3h. *Exit:* as FX-N4.
- **FX-N13 · Streaming motivation experiment (heterogeneous, α=0.1) · blocked: FX-N4, FX-N5 for felix, oort,
  refl, feddance.** Show empirically that (a) per-trainer statistical utility changes over time as data
  streams in and is trained on, (b) an unaware aggregator mis-selects, and (c) one that tracks utility but
  mis-estimates it still mis-selects. Code is on HEAD (per-baseline `*_oracle` injection, target-accuracy stop,
  `scripts/run_felix_streaming.sh`, `oracle_misselection.py`, `felix_streaming_figures.py`); it has never been run.
  Steps: local smoke (harness) → calibrate the streaming horizon → sweep → analyze. Exact design comes from the
  operator before the sweep. *Exit:* utility-over-time and mis-selection figures for each baseline.
- **FX-N6 · Unavailability design re-audit · todo.** Keep v1 semantics; check them against the fwdllm
  invariants (FX-N3b), logical-budget grading and the drain primitives. Record decisions in
  UNAVAILABILITY_DESIGN.md; any semantic change needs operator sign-off. *Exit:* one audit table (item ·
  keep/change · evidence).
- **FX-N7 · Remove legacy `trackTrainerAvail` (oort, oort_star, refl) · todo.** First add
  `simUnavailability: true` statically, then zero the legacy block, then verify with a generated config
  and a short real run.
- **FX-N8 · Root-cause the concurrent-run confound on fedbuff · todo.** Candidates: shared MQTT broker
  (~600 connections), network, storage. Harness at n=300 stub with two concurrent runs first.
- **FX-N9 · Unavailability cluster: syn_20 → syn_50 → mobiperf_3st, all six · blocked: FX-N5, FX-N6.**
  *Exit:* A1-A8/K11 + the syn_0 ladder green; clean self-stop; withheld updates delivered, not dropped;
  AVL_EVAL and the empty-pool cleanup exercised live on mobiperf_3st.
- **FX-N10 · Move google_speech to the launcher · blocked: parent S3; reference config awaiting operator OK.**
  The newest config trail is the 2024-07-14 SoCC set (`expt_runs_speech_n100_14Jul24_felix_*.sh` +
  `aggregator/fedbuff_config_final_expt_14jul24_felix_v3.json`): ResNet (`main_resnet.py`), n=100, α=1
  (splits exist for 0.1/1/10/100), 48h mobiperf trace, `aggGoal` 10, `c` 30, lr 0.001, batch 32, stop at 20
  consecutive evals ≥ 60%. The later paper numbers (eurosys26/socc26 `all_plots.ipynb`) are hard-coded and
  leave no config behind. Configs exist only for felix and fedbuff; oort, oort_star, refl and feddance must be
  rebuilt from their cifar `baselines.yaml` entries. Then: dataset entry in `_metadata`, splits, registry,
  sim-mode trainer, one smoke YAML per baseline; legacy JSON removed under parent S4.
- **FX-N11 · google_speech real + syn_0 + unavailability parity · blocked: FX-N9, FX-N10.** Reuse the
  FX-N4/5/9 protocol unchanged.
- **FX-N12 · Felix paper experiments, sim-only · blocked: FX-N11** (async_cifar10 experiments may start after
  FX-N9). Experiment list: open question below.

---

## Felix lessons (dos)

**Selectors (Oort family)**
- **FX-L1** Oort's pacer has two branches: a flat trend (`|Δ| ≤ 0.1·last`) raises `round_threshold`, a
  sharp one (`≥ 5·last`) lowers it. It fires on TRAIN only. Log `round_threshold` every round.
- **FX-L2** The UCB temporal term keys on the round of the last RECEIPT (`PROP_LAST_RETURNED_ROUND`),
  initialised at registration — never None, never the dispatch round.
- **FX-L3** Once a faithful controller still diverges, instrument its INPUT by quartile; stop touching the
  controller.
- **FX-L4** Async and sync selector knobs differ: Oort paper defaults are sync values. Per-round terms
  fire 2-3× more often in async. Felix uses `exploration_decay` 0.999.
- **FX-L5** Real records a stale-but-returned trainer's speed and utility too; skipping it makes Oort treat
  slow trainers as unexplored and re-pick them forever.

**Aggregation, ordering, clock**
- **FX-L6** Async (felix): drain each in-flight end's queue directly (`simSctOrderedDrain`) and hold busy
  slots until commit (`_sim_hold_busy_slots`). This one root cleared K3b/K2/U3/U6/K8/U2 together.
- **FX-L7** Sync oort over-selects ×1.3: a prior-round straggler with `sct` past the pinned round start is
  held in `selected_ends` and commits later (`simInflightCarryover`).
- **FX-L8** refl/oort: a trainer still computing (`vclock < sct`) is kept out of the eligible pool via the
  unavailable path, not `selected_ends`.
- **FX-L9** For a strict-barrier baseline (feddance, fedavg), real visibility lag is anchored on the barrier
  (`max_dur − dur_i`); streaming oort/refl stay per-message.
- **FX-L10** Split eval from train in any check that reads `agg_rounds`. Each eval gets its own `sct`,
  never the last train `sct`.

**Availability**
- **FX-L11** Two separate axes per baseline: knowledge at selection (`avail_select_filter`: felix, oort_star,
  refl, feddance) and in-flight slot release (`proactive_inflight_evict`: felix only; others use the 90s
  vclock abandon).
- **FX-L12** If a trainer drops mid-flight, its compute still completes; the send is gated (real), or
  buffered to `delivery_ts = max(sct, next_avail)` (sim), and it commits late as stale. Nothing is dropped.
- **FX-L13** Under scarcity, sim jumps the vclock to the next availability transition; size `--runtime-s`
  for it rather than subtracting the jumps.
- **FX-L14** In-flight count differs per baseline (oort over-selects, async is bound by concurrency, refl
  and feddance clear each round). Size n ≈ threshold / (1 − unavailable fraction).
- **FX-L15** The ramp is syn_0 (byte-identical to availability off) → syn_20 → syn_50 → mobiperf. 2-state
  traces collapse AVL_EVAL (`_trace_has_avl_eval`); only mobiperf exercises it.

**Reading the checker**
- **FX-L16** A2 failing (KS) while S3/4 passes is one in-flight gap graded at two tolerances; walk to
  residence, not eligibility.
- **FX-L17** U6 KS on a sub-ms point mass is signal-free; read `mean_diff`. P3 sub-second opposite-sign
  offsets are wall-capture; trust P3 only when `grid_KS` also fails.
- **FX-L18** `gate_holds = 0` over a whole run means the gate is structurally inert, so suspect the
  accounting upstream. Measure one-in-flight from overlapping dispatch→commit intervals, not warning
  counters.
- **FX-L19** `phase_gpu_compute` and refl K2 are sensitive to run length; re-check at ≥2.5h before acting
  on a short-run FAIL.
- **FX-L20** Run lengths: telemetry sanity 5-10 min · one mechanism rung 45 min · compounding clock residual
  (K2/K3b) 90 min-2h · refl low-frequency K2 drift 3h · C1/C2 sign-off 3-4h. Smoke 5 min first.
- **FX-L21** `Sdet` triage: eligible set differs + aggregates match = stochastic, PASS; + clock diverges = fix
  the clock; eligible set matches but decisions differ = the score VALUES diverge.
- **FX-L22** Past-dating comes in two streams (train-only U6 vs all-commit SIM_CLOCK_DIAG); always say which
  one a number came from.
- **FX-L23** `sim_committed_fresh == agg_goal` confirms oort's block-for-K fix; a later fresh-count gap is a
  different root.
- **FX-L24** Only re-run real when the real path changed; sim-only changes grade against the stored real dir.

## Felix tripwires (don'ts)

- **FX-T1** Don't enable `simStaggeredRedispatch` (falsified; kept off) or retune `simRedispatchGapSeconds`.
- **FX-T2** Don't add `mqtt_fetch` (~20-57s) to `sct`; it is the wait before re-selection, not transfer.
- **FX-T3** Don't add a `system_util` recency guard or widen the slow-speed tail for oort carry-over decay.
- **FX-T4** Don't tune the `_sim_recv_min` gate predictor; `exp == sct` exactly.
- **FX-T5** Don't hold ALL buffered ends out of refl's pool (`pending_ends`); it over-holds.
- **FX-T6** Don't key oort latency by task type (sync oort sends no eval tasks).
- **FX-T7** Don't clamp felix's clock jump or pace dispatch to fix "fresh" past-dating; that was eval
  reusing a stale `sct`.
- **FX-T8** Don't expect seeding to align per-round selection sets; judge S2 by speed class.
- **FX-T9** Don't express "busy" through the UN_AVL list, including in any unavailability redesign.
- **FX-T10** Don't use per-tick MQTT availability broadcasts (comms storm); v1 reads the trace.
- **FX-T11** Don't exclude AVL_TRAIN from eval on a 2-state trace (it empties the pool and wipes
  `selected_ends`).
- **FX-T12** Don't zero the legacy `trackTrainerAvail` block before `simUnavailability` is set
  statically (FX-N7).
- **FX-T13** Don't grade a fedbuff/felix real run while another n=300 real run shares the broker (FX-N8).
- **FX-T14** Don't add prediction-only gates with no real blocking (they never fire), or a `version_at(sct)`
  staleness relabel (inert).
- **FX-T15** Don't read high wall-clock commit density as sim "running fast"; judge per-round vclock
  advance and `commit_gap`.
- **FX-T16** Don't re-chase GPU contention (overrun 0), SEND_TIMEOUT or MQTT drops at n=300 cifar; all
  measured 0.
- **FX-T17** Don't expect oort carry-over decay to be a run-length transient (structural), or the felix
  min-budget seed alone to fix past-dating.
- **FX-T18** Don't add a scalar fudge for P3's ~1s `mean_overhead` offset (wall-capture).
- **FX-T19** Don't fork withhold/abandon logic per stack (one shared `ClientAvailability`), or grade A4 on
  the bare transition fraction (use A4dur).

---

## Done
- **FX-D5** Single-run event checker (EV0-EV14) run on the stored Jul-2 runs found: (1) real send-gate
  compared the bool `wait_until_next_avl` to `"True"`, so every real withheld update was DROPPED (felix 43,
  fedbuff 137 trainers) — fixed in `syncfl/trainer.py` + bool tests; (2) `channel.one_end` crashed on a
  departed peer at shutdown — fixed; (3) `mobiperf_3st` was unlaunchable (loader/spawner knew only `_50/_75`)
  — aliased; (4) FX-N14/15/16 opened. The syn_0 sim run is clean on all 15 checks.
- **FX-D4** Felix sim-path audit at HEAD: pacer still two-branch + train-gated, now also once-per-round
  (`_last_pacer_round`, new since the 46/46 run). First-contact trainers were invisible to
  `_sim_inflight_expected`, so the commit gate and `_sim_hold_busy_slots` both missed them → ported fwdllm's
  cold-start gate to asyncfl behind `simColdStartGate` (+ `simGateComputeCapSeconds`), 7 new tests.
- **FX-D1** Sim fidelity fixes, async_cifar10: sct-ordered drain (§3.drain), one-in-flight residence
  (§3.resid), oort carry-over (§4.9), refl pool exclusion (§4.5), intrinsic selector duration (§S.dur), UCB
  temporal fix (§S.temporal), faithful pacer (§S.pacer), feddance barrier-anchored U6 (§6.u6) — PARITY.md §3.
- **FX-D2** Unavailability v1 substrate across all six (send-gate / deliver-late, two ledgers, proactive
  evict, starvation advance, absolute A6/A7/A8/K11 checks); felix 62/62 at syn_50 — UNAVAILABILITY_DESIGN.md.
- **FX-D3** Readiness docs created; scope and ordering set with the operator (2026-09-25).

## Open questions (operator)
- google_speech reference: adopt the FX-N10 settings (n=100, α=1, ResNet, 60% target), or move to n=300 / α=0.1 to
  match async_cifar10?
- Felix paper experiment list, and the exact streaming-experiment design (FX-N13), when ready.
