# FluxTune readiness — forward-gradient fine-tuning (fwdllm)

## Preamble
- **Read [ROBUST_FL_READINESS.md](ROBUST_FL_READINESS.md) first.** Its doc rules, operating rules (R1-R13),
  shared lessons (L) and tripwires (T) apply here. This doc holds only what is specific to FluxTune:
  gradients (JVPs) aggregated instead of weights, a variance-gated commit, and the `data_id` progress axis.
- **PARKED until Felix experiments are running (FELIX_READINESS FX-N12).** Nothing here is worked except
  (a) keeping this track green under shared changes (R10) and (b) the parent S-steps that cover fwdllm.
- **Scope:** nine baselines on AG News / DistilBERT, n=100, α=1, `jvp_eval_mode` ON: `fluxtune`, `fwdllm`,
  `fwdllm_it_unaware`, `fwdllm_it_oracular`, `fedbuff_round`, `fedbuff_it_unaware`, `fedbuff_it_oracular`,
  `felix_round`, `felix_it`.
- **IDs:** `FT-N` next steps · `FT-L` lessons · `FT-T` tripwires · `FT-D` done.
- **Reference (read only for detail):** parity board, runbook, full lesson and dead-end list →
  [simulate_fwdllm.md](../fwdllm/simulate_fwdllm.md) (§A, §B.1, §D, §E, §F) · design, charge registry →
  [FWDLLM_DESIGN.md](../fwdllm/FWDLLM_DESIGN.md) · training stability →
  [fluxtune_contributions.md](../fwdllm/fluxtune_contributions.md) §8 · experiments →
  [EXPERIMENTS.md](../paper_expts_fluxtune/EXPERIMENTS.md), [EXPTS_CHARTER.md](../paper_expts_fluxtune/EXPTS_CHARTER.md),
  [BRIDGE_DESIGN.md](../paper_expts_fluxtune/BRIDGE_DESIGN.md) · identity → [BASELINES.md](BASELINES.md).

---

## Status grid

✅ done · ⚠ open · ⬚ not started. Parity = `run_parity.py`, two-sided floors, median over real legs.

| baseline | real (n≥2) | sim syn_0 (pass/fail/skip) | sim + unavailability |
|---|---|---|---|
| fluxtune v1 | ✅ | ✅ 74/0/19 | ⬚ |
| fluxtune v2 (now the default) | ⚠ collapses at the round-2 boundary (I-1) | ⚠ charge profile predates v2 | ⬚ |
| fwdllm | ✅ | ✅ 68/0/24 | ⬚ |
| fwdllm_it_unaware / _oracular | ✅ | ✅ 69/0/23 each | ⬚ |
| fedbuff_round | ✅ | ✅ 63/0/30 | ⬚ |
| fedbuff_it_unaware / _oracular | ✅ | ✅ 63/0/29, 62/0/29 (`utility` SKIP) | ⬚ |
| felix_round | ✅ | ✅ 62/0/30 | ⬚ |
| felix_it | ✅ | ✅ 66/0/27 | ⬚ |

syn_0 parity is closed on v1. Unavailability: `fwdllm_aggregator.py` has no send-gate, deliver-late or
withheld-delivery wiring yet (only the oracular select filter and the sync scarcity wait).

---

## Next steps (persistent queue — resumes after Felix)

- **FT-N1 · Re-profile `sim_charge_profiles/fluxtune.yaml` against v2 real legs · todo.** Decide whether v1
  keeps its own profile. *Exit:* a v2 sim block graded 0-fail against v2 reals.
- **FT-N2 · Settle whether `trackTrainerAvail` is inert at syn_0 · todo.** One `fwdllm_it_unaware` real leg
  on `_oracular`'s node. Prediction: 38 bins / 9.64 iters per bin = the knob matters; 40 / 9.20 = the host. If it isn't inert, un-pool `fedbuff_it`.
- **FT-N3 · Baseline wiring leftovers · todo.** (a) Confirm that the 5 new baseline keys and the
  `fwdllm_plus` → `fwdllm_it_*` rename reached code, tests, smoke YAMLs, `plotlib/baselines.py` legends and
  `run_parity.py` (BRIDGE_DESIGN steps 1-5 say they did). *Exit:* a repo-wide grep for the old keys finds
  only historical hits; `test_baselines.py` + `test_config_generator.py` are green. (b) Retune the
  placeholder `learning_rate: 0.075` for `felix_round`/`felix_it` (`TODO(verify)` in `baselines.yaml`).
  (c) BRIDGE_DESIGN step 6: an N=10 sim smoke per new baseline. Keep `fluxtune_dynkc` parked, not deleted.
- **FT-N4 · Training stability I-1 under v2 · todo** (owned by fluxtune_contributions.md §8). *Exit:* a 4h+
  run holds past the lap boundary. Gates every long experiment run.
- **FT-N5 · Unavailability for the grad loop · blocked: Felix FX-N6 audit + FX-N9.** Reuse the audited
  `ClientAvailability` path; harness first; syn_20 → syn_50 → mobiperf_3st. *Exit:* A1-A8/K11 + the syn_0
  board green; withheld gradients delivered, not dropped.
- **FT-N6 · Paper experiments E1-E5, sim-only · blocked: FT-N1, FT-N4** (and FT-N5 for the unavailability
  figures).
- **FT-N7 · Re-enforce `fedbuff_it_*` `utility` · todo.** Its sim floor (0.314) swallows the 0.2 gate; find
  why two `_oracular` sim legs differ that much. More legs won't help.
- **FT-N8 · Promote `sim_sct_ordered_drain` + `sim_model_dispatch_queue` · todo.** Currently fluxtune-only;
  smoke fwdllm and fwdllm_it with both on, confirm inert-or-better, then promote.
- **FT-N10 · 5 pre-existing red tests at HEAD · todo.** `test_fwdllm_eval_background` (3),
  `test_fwdllm_probe_report::test_trainer_entrypoint_defaults_on`,
  `test_fwdllm_server_update_audit::test_norms_are_inside_the_gate` (red with the Felix changes stashed too).
- **FT-N9 · Estimator/optimizer roadmap (momentum S1-S3, server optimizer) · todo.** Re-measure the replicate
  floor after the damping lands.

Shared items that already touch fwdllm: parent S1 (stub mode for the fwdllm family), S2, S3 (merge
`run_sequential.sh`), S4 (~550 legacy JSON), S5 (knob contract).

---

## FluxTune lessons (dos)

- **FT-L1** Progress is committed `data_id`; version identity is `version_key = (model_version,
  iteration_per_data_id)`. `data_id` wraps every lap, so never key a cache or identity on it.
- **FT-L2** `model_version` bumps once per completed bin; a same-version re-pick gets a RETRY signal, not a
  weight re-send. One instruction per `version_key` per trainer.
- **FT-L3** Slot residence survives variance rollbacks: the slot frees at commit and the re-pick guard
  holds to the agg-goal boundary (capacity vs identity, parent L5).
- **FT-L4** The variance gate sits far below the achieved variance, so iterations per bin is a hitting time.
  Its spread is noise; a cap or barrier pins both the residual and the floor.
- **FT-L5** A cadence gap is always an upstream set, order or clock divergence. Reduce it to the bin's first
  variance ÷ threshold before hunting.
- **FT-L6** Charge profiles are per baseline, from that baseline's paired real legs, and the launch
  preflight checks their provenance. Re-profile after ANY training-knob change, before reading a residual.
- **FT-L7** Pair legs on `jvp_eval_mode` AND `max_runtime_s` AND commit. A pinned baseline (0.0% floor) is
  the only one that can show a small between-config effect.
- **FT-L8** The collapse happens at the lap boundary (~wall 4300-4800s, round 1→2); any accuracy or
  stability claim needs a 4h+ run.
- **FT-L9** Bench repros must build the model in the mode production uses (dropout live, training mode);
  inference mode gives bit-exact nulls.
- **FT-L10** Sim must produce a speedup (`vclock/wall ≥ 1`). The current limit is the serial per-commit
  aggregator (~225 ms), not sleeps.

## FluxTune tripwires (don'ts)

- **FT-T1** Don't touch `var_threshold`, `max_iterations_per_data_id` or DynamicKC to close a parity gap.
- **FT-T2** Don't re-open `calculate_var` or pool assembly on cadence evidence alone; stale charges
  explained it.
- **FT-T3** Don't re-derive redispatch turnaround from timestamp differences (it prices waiting).
- **FT-T4** Don't admit by lowest modeled completion time instead of first-arrived (FIFO-violating;
  deadlocks under unavailability).
- **FT-T5** Don't treat the round-cadence cohort pin as a defect (by design).
- **FT-T6** Don't blame reduced precision or GPU nondeterminism for the replicate floor; the source is live
  dropout inside the finite difference.
- **FT-T7** Don't read cadence or convergence off a single real leg on an uncapped baseline.
- **FT-T8** Don't run a `fluxtune` v2 sim on the v1 charge profile (`--only fluxtune` isn't blocked by the
  preflight).
- **FT-T9** Don't forget `export FWDLLM_FD_SCALE_INVARIANT=1` for v2 (the preflight errors without it).

---

## Done
- **FT-D1** syn_0 parity landed on all nine: two-sided floors, real↔real control, median over real legs,
  zero fails — simulate_fwdllm.md §A-§B.
- **FT-D2** Sim barrier redesign (event-driven, no hard-coded waits), profiled charge registry,
  estimator v2 — FWDLLM_DESIGN.md §M/§P, BASELINES.md.

## Open questions (operator)
- Does C2 (dynamic K/C) land in the headline `fluxtune` config or get descoped? (BASELINES.md ᵖ note.)
- Should the oracular rows stay on the syn_0 board? Depends on FT-N2.
- Checker invariants I1-I6 were drafted once and lost; what semantics were intended?
- `felix_it`'s ON accuracy sits 1.0-3.4 pts below its OFF band (`felix_round` improved). Low priority;
  falsified if a second loss-derived baseline also degrades ON.
