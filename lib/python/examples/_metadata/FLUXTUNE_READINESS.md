# FluxTune readiness — forward-gradient fine-tuning (fwdllm)

## Preamble
- **Read [ROBUST_FL_READINESS.md](ROBUST_FL_READINESS.md) first.** Its doc rules, operating rules (R1-R13),
  shared lessons (L) and tripwires (T) apply here. This doc holds only what is specific to FluxTune:
  gradients (JVPs) aggregated instead of weights, a variance-gated commit, and the `data_id` progress axis.
- **Paper runs unparked (operator 10-17):** [RUN_PLAN.md](../paper_expts_fluxtune/RUN_PLAN.md) gates G0 (FT-N1), P0 (FT-N4), G1 (grad-aware keep/drop)
  first. **Read RUN_PLAN's Status board first** (what is running, done, next). The rest of this queue stays parked behind Felix (FX-N9, FX-N11), except keeping this track green (R10).
- **Run status (operator 10-18):** keep RUN_PLAN's Status board current: update it in the same session as every
  launch, finish or decision (run dir, node, GPUs, tmux, logs, outcome).
- **Scope:** nine baselines on AG News / DistilBERT, n=100, α=1, `jvp_eval_mode` ON: `fluxtune`, `fwdllm`,
  `fwdllm_it_unaware`, `fwdllm_it_oracular`, `fedbuff_round`, `fedbuff_it_unaware`, `fedbuff_it_oracular`,
  `felix_round`, `felix_it`.
- **IDs:** `FT-N` next steps · `FT-L` lessons · `FT-T` tripwires · `FT-D` done.
- **Reference (read only for detail):** parity board, runbook, full lesson and dead-end list →
  [simulate_fwdllm.md](../fwdllm/simulate_fwdllm.md) (§A, §B.1, §D, §E, §F) · design, charge registry →
  [FWDLLM_DESIGN.md](../fwdllm/FWDLLM_DESIGN.md) · training stability →
  [fluxtune_contributions.md](../fwdllm/fluxtune_contributions.md) §8 · experiments →
  [EXPERIMENTS.md](../paper_expts_fluxtune/EXPERIMENTS.md) (live paper plan) · identity → [BASELINES.md](BASELINES.md).

---

## Status grid

✅ done · ⚠ open · ⬚ not started. Parity = `run_parity.py`, two-sided floors, median over real legs.

| baseline | real (n≥2) | sim syn_0 (pass/fail/skip) | sim + unavailability |
|---|---|---|---|
| fluxtune v1 | ✅ | ✅ 74/0/19 | ⬚ |
| fluxtune v2 (now the default) | ⚠ collapses at the round-2 boundary (I-1) | ✅ v2 profiles final 10-25 (AG News, Yahoo) | ⬚ |
| fwdllm | ✅ | ✅ 68/0/24 | ⬚ |
| fwdllm_it_unaware / _oracular | ✅ | ✅ 69/0/23 each | ⬚ |
| fedbuff_round | ✅ | ✅ 63/0/30 | ⬚ |
| fedbuff_it_unaware / _oracular | ✅ | ✅ 63/0/29, 62/0/29 (`utility` SKIP) | ⬚ |
| felix_round | ✅ | ✅ 62/0/30 | ⬚ |
| felix_it | ✅ | ✅ 66/0/27 | ⬚ |

Big models (RoBERTa-L, SmolLM2-1.7B): real↔sim parity not graded; it transfers from DistilBERT under FT-L19, checked by
FT-N21. syn_0 parity is closed on v1. Unavailability: `fwdllm_aggregator.py` has no send-gate, deliver-late or
withheld-delivery wiring yet (only the oracular select filter and the sync scarcity wait).

---

## Next steps (persistent queue — resumes after Felix)

- **FT-N21 · Big-model real↔sim transfer check (FT-L19) · todo.** Per model, one 1–2 h real leg at the paper packing beside
  a same-seed sim leg. *Exit:* 0 `[TIMING_OVERRUN]`, real `vclock_charge` spans near the dataset profile, no eval-held
  commits beyond seconds, accuracy within band at matched commits. *Why:* the 10-min legs (10-24) reached no real commit.
- **FT-N22 · Server momentum with ρ0 0.14 · todo (proposed 10-24).** DistilBERT · AG News, N 50, ρ0 0.14, T_res 71, momentum
  0.5 vs none, 3 seeds each, to 0.86 two in a row; bench is fine (DistilBERT `fluxtune` passed B2). *Exit:* keep or drop a
  momentum arm for RoBERTa. *Why:* only single-seed evidence to 0.60 (FT-T15), and Φ ≠ e^B under momentum (FT-T16).

- **FT-N20 · Data augmentation like every Felix stack (FELIX FX-D233, FX-N143, Decisions "Augmentation") · todo (on resume).**
  *Principle (operator 10-24):* if the baselines' own pipelines augment, every stack and the ceiling do, with fresh draws per pass on
  the device; never adopted or skipped for convenience. Felix found its GPU preload froze one draw per sample (+10-17 pp once fixed).
  *Verify:* (1) what fwdllm's / FwdLLM's and FedScale's AG News loaders do (text usually has no standard augmentation: then record
  "none, as the baselines", not a new trick); (2) whether pre-tokenized / cached samples freeze any random step (masking, truncation
  offset). *Add* only what the baselines' code does, to FluxTune and every fwdllm baseline alike, plus the defense (what, standard,
  on-device cost, why it helps, no privacy claim, targets re-derived). *Exit:* noted in the setup text; one smoke per baseline.
- **FT-N19 · One bench queue for Felix and FluxTune · code done 10-24; exit run waits for a Felix screen + shepherd (operator).**
  *Built:* `bench_pool.py` jobs-file columns `gpu_gb= ram_gb= workers= runtime_s= project=`, else the dataset's
  `scripts/bench_profiles.yaml` entry (measured: cifar10, agnews), else the flags; the queue starts the first leg that fits; a mixed
  queue writes `POOL/<project>/<phase>/` (project from the baseline's stack). Smoke (jayne GPUs 0-3): 2 Felix FedBuff cifar n 300 +
  2 fwdllm AG News legs, 4/4 ok in 3.9 min, cifar and fwdllm legs sharing a GPU; the fwdllm leg bit-identical to its solo run.
  *Left:* RoBERTa-L / speech profiles; per-job `--gpus-per-leg`; the exit run below. *Why:* both projects share jayne /
  shepherd; one `bench_pool.py` queue that takes cifar / speech (Felix) and fwdllm (FluxTune) legs together runs them interleaved
  without hand-split GPU lists or one batch waiting on the other. `bench_pool.py` already places each leg by free GPU memory + host
  RAM (FELIX FX-N141 Built); left: fwdllm legs on the bench (FT-N18), a per-leg resource class (fwdllm DistilBERT legs need more GPU
  memory than cifar's ~3.4 GB: set `--leg-gpu-gb` per job from a profile, not one flag), a jobs file mixing tiers from both
  harnesses, and per-project output pools. *Exit:* one mixed batch (Felix screen + fwdllm screen) on both nodes, every leg ok,
  no OOM, legs / h ≥ the two batches run back to back.

- **FT-N18 · fwdllm on the in-process bench (FELIX V5, FX-D213 / D218) · B2 + B4 green 10-24 (FT-D3); B1 open.** *Why:* Felix's harness legs bound on host
  RAM (84-99 GB, one process per trainer), so a node ran 4-5 legs; the bench (aggregator + W worker processes, in-memory channel) runs
  ~4 GB legs: cifar 121-141 legs / h on jayne vs ~15 on both nodes (8-9×). It pays only when per-trainer process cost binds; speech
  sims are GPU-compute-bound and gained ~1×.
  - *Step 0, decide (≤ 15 min):* profile one fwdllm harness sim leg at production n (`resource_report.py --util`: RAM, GPU memory, SM).
    RAM / GPU memory bound (one DistilBERT copy per trainer) → port; SM-bound → stay on the harness, record why here.
  - *Port:* `fwdllm_aggregator` (barrier / variance-gated cadence, `version_key` with iterations), `FedSgdTrainer` (its own
    `ChannelManager` import in `fwdllm_trainer.py` needs the same patch), every channel method it calls on the in-memory channel, H5
    path-style data and `client_idx_modulo`, the trace keys it reads unconditionally (`avl_events_syn_*`), workers per GPU by LLM memory;
    `bench_pool.py --harness-args` so legs come from the harness plan.
  - *Adoption gate (Felix passed B1, B2, B4 before any screen; B3 still open there):* B1 replay of real commits from a checkpoint within
    1 pp; B2 per baseline × trace, 3 bench seeds vs fwdllm harness sim legs at matched clock: |AUC Δ| ≤ max(1.5 pp, either side's seed
    range), updates / h ±10%, bytes ±5%, mean task time ±5% with 0 task-time formula mismatches; fwdllm-specific: iterations per
    `data_id`, variance pass rate and commit cadence within the same bands; B4 bench seed floors; a red re-runs with 2-3 harness seeds
    before naming a root (both Felix reds were draws); `config_audit` on bench legs (knobs = table, holdout P3 / P4); same results at
    2 and 4 workers; analysis tools read bench legs unchanged; packing per FX-T73 / T74 (one math thread, CUDA contexts per GPU).
  - *Using it:* bench results stand for real only on cells green on both links, real ⇄ harness sim (FT parity) and harness ⇄ bench
    (FX-L96); never grade the bench against real; re-run B2 on touched cells after any shared `flame/` or trainer change.
  *Exit:* step 0 recorded; if ported, B1 / B2 / B4 green on the fwdllm board cells.
  - *Scope (operator 10-24):* B2 tested `fwdllm` and `fluxtune` only; `fedbuff_*`, `felix_*` and `fwdllm_it_*` were not tested
    on the bench specifically (same aggregator / trainer classes; their async-staleness, pacing and per-iteration re-pick paths are
    unchecked there), so run a B2 cell before reading any of them off the bench.
  - *Left:* B1 (checkpoint replay of real commits); a B2 cell plus a `bench_profiles.yaml` entry per big model (Yelp-P,
    Yahoo fall back to 4 GB/leg, so SmolLM legs would OOM); 12 h cells stay on the harness (operator 10-24); a fwdllm knob table for
    `config_audit` (FT-N15); `bench_pool --harness-args` for `run_sequential.sh` plans (legs come from its dry-run YAMLs via `--jobs` now).

- **FT-N16 · Measured packing for FluxTune blocks · todo (on resume).** PARITY C21/C22 and Throughput: screen each block's legs at
  production density, record per-leg CPU/RAM/GPU footprints, pack with `--gpu-share`/`--lease-from-telemetry`. *Exit:* a FluxTune block's
  `--util` record in this doc before its first long run.

- **FT-N15 · Baseline fidelity for the fwdllm family · todo (after Felix).** Adopt ROBUST L33-L37: add fwdllm baselines to
  `baseline_reference.yaml` (local work, lr/batch, server step, K/N >= 5%, cited from their papers/code); check compute < D per setting.
  *Exit:* `test_baseline_reference.py` covers every FluxTune baseline.
- **FT-N12 · Port FX-N77's real-cost fixes to fwdllm · todo (after Felix).** Already shared: channel framing + MQTT transport, event-driven
  `drain_ready`, `materialize_weights` decode in `fwdllm_aggregator`. Left: fwdllm's distribute still pickles `WEIGHTS` and
  `fwdllm_trainer` reads it directly (→ `pack_weights` / `materialize_weights`); fwdllm files are exempt from
  `test_no_eager_payload_logs.py`; profile a fwdllm PROF-shaped leg with `FLAME_PYSPY` first. *Exit:* fwdllm board unchanged, per-round
  serialization ~0 in `profile_report.py`.
- **FT-N13 · Retire legacy `trackTrainerAvail` ORACULAR (FX-N7 Felix half done) · todo.** Felix reads one runner-fanned
  `availability_trace`. Left: `fwdllm_it_oracular`, `fedbuff_it_oracular`, `FedSgdAggregator`, `fwdllm_aggregator.check_trainer_availability`
  → `simUnavailability`; then drop the legacy branch in `ClientAvailability._init_availability`. *Exit:* no `trackTrainerAvail` in code.
- **FT-N16 · U6 barrier anchor on absolute ready times (FX-D151) · todo.** `fwdllm_aggregator` feeds `_barrier_anchored_lags`
  dispatch-relative durations (`SIM_CLIENT_TASK_TRAIN_DURATION_S`); wrong once picks are dispatched at different times (top-ups).
- **FT-N17 · Mid-barrier withheld deliveries (FX-D194) · todo.** `fwdllm_aggregator`'s incremental sync commits K fresh updates before
  re-injecting a send-gated one due earlier; port `_sync_sim_commit_due_withheld`. *Exit:* a test as `test_sim_barrier`'s FX-D194.
- **FT-N14 · Real abandon telemetry (FX-N77) · todo.** asyncfl selector reclaims now emit `abandon_timeout`; check fwdllm's
  own abandon path emits it in real. *Exit:* real fwdllm abandons visible to stall-cause stamping.
- **FT-N11 · 6 red fwdllm tests (R10) · todo.** `tests/mode/test_fwdllm_{probe_report,server_update_audit,eval_background}`
  (source drift: moved strings, a `config`-less fake) and `test_replicate_floor` (`pretokenize_dataset.py` not in
  `LAUNCHER_INVOKES`). *Exit:* full pytest green.
- **FT-N1 · Re-profile `sim_charge_profiles/fluxtune.yaml` against v2 real legs · done 10-25, final (operator: used for all future runs; exit waived).** 30-min real legs
  (RUN_PLAN decision 12, `charge_profile_20261025/`): AG News 0.032 / 0.031 s, Yahoo 0.120 / 0.917 s (`drain_tail` / `fedavg`); Yelp-P left.
  Decide whether v1 keeps its own profile. *Exit:* a v2 sim block graded 0-fail against v2 reals.
- **FT-N2 · Settle whether `trackTrainerAvail` is inert at syn_0 · todo.** One `fwdllm_it_unaware` real leg
  on `_oracular`'s node. Prediction: 38 bins / 9.64 iters per bin = the knob matters; 40 / 9.20 = the host. If it isn't inert, un-pool `fedbuff_it`.
- **FT-N3 · Baseline wiring leftovers · todo.** (a) Confirm that the 5 new baseline keys and the
  `fwdllm_plus` → `fwdllm_it_*` rename reached code, tests, smoke YAMLs, `plotlib/baselines.py` legends and
  `run_parity.py`. *Exit:* a repo-wide grep for the old keys finds
  only historical hits; `test_baselines.py` + `test_config_generator.py` are green. (b) Retune the
  placeholder `learning_rate: 0.075` for `felix_round`/`felix_it` (`TODO(verify)` in `baselines.yaml`).
  (c) An N=10 sim smoke per new baseline. Keep `fluxtune_dynkc` parked, not deleted.
- **FT-N4 · Training stability I-1 under v2 · closed 10-18 for DistilBERT** (P0 `exp` crossed 7 lap wraps, no collapse; check P3/P4 at their first wrap) (owned by fluxtune_contributions.md §8). *Exit:* a 4h+
  run holds past the lap boundary. Gates every long experiment run.
- **FT-N5 · Unavailability for the grad loop · blocked: Felix FX-N6 audit + FX-N9.** Reuse the audited
  `ClientAvailability` path; harness first; syn_20_3st → syn_50_3st → mobiperf_3st. *Exit:* A1-A8/K11 + the syn_0
  board green; withheld gradients delivered, not dropped.
- **FT-N6 · Paper experiments, sim-only · FT-N1 closed 10-25; blocked: FT-N4** (FT-N5 for C4 availability). Queue and figures:
  [EXPERIMENTS.md](../paper_expts_fluxtune/EXPERIMENTS.md) §3.
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
- **FT-L10** Sim must produce a speedup (`vclock/wall ≥ 1`); measured 10-24: DistilBERT 3.4×, RoBERTa-L 33×, SmolLM2-1.7B
  14× (RUN_PLAN §4). The limit is the serial per-commit aggregator (~225 ms), not sleeps.
- **FT-L11** Sim charges each task's modeled delay D = `training_delay_s / divisor` to the vclock; real sleeps it. Recipe D
  dwarfs GPU time (RoBERTa 237–1,124 s vs 9 s), so the gain ≈ D / GPU time.
- **FT-L12** Size a real leg by D, not wall: an iteration takes 4–46 min on RoBERTa-L, 14 min–2.8 h on 1.7B (D_CALIBRATION);
  first commit ≥ 1 h. A 10-min real leg measures speed only (`ft_models_check_20261024`, shepherd).
- **FT-L13** Compare real and sim at matched commits, never matched wall: n 50, real 103 commits at 0.33–0.34 vs sim
  0.32–0.37 at c100 (`ft_n50_check_20261024`, shepherd).
- **FT-L14** Real `max_runtime_s` counts from aggregator start, join barrier included (minInit = N); 100 trainers take
  ~3.5 min to join, so a real smoke needs ≥ 6 min (`smoke_merge_20261024`).
- **FT-L15** A sim leg ends on its wall ceiling: `--max-runtime-s 1000000000 --force` plus `--sim-wall-ceiling-*` (40× ended
  rb60 at 4.7 h); `run_sequential`'s kill budget follows `max_runtime_s` (~+130 s), not the ceiling.
- **FT-L16** Side-by-side legs run as lanes: private mosquitto, `FLAME_SLOT_GPUS`, `taskset` cores and a distinct
  `FLAME_RUN_LABEL` each (`ft_models_check_20261024/run.sh`).
- **FT-L17** SmolLM2-1.7B bs 32: 8 trainers per A40 on a 2-GPU lane holding the aggregator OOMed two; 4 per GPU ran
  clean (`ft_models_check_20261024/first_try`). Check GPU memory in the first 10 min at any denser packing.
- **FT-L18** Measure a model's D from one trainer alone on a GPU, eval off: per-task cost varies ≤ 2%, so ~25 tasks fix
  it; anchor the ratio to DistilBERT at the same bs (`ft_profile_20261024`, shepherd).
- **FT-L19** DistilBERT's real↔sim parity extends to a new model: no code branches on the model between modes; it holds while
  GPU time ≪ D (no overruns) and first commits match a same-seed sim leg (FT-N21).
- **FT-L20** Stop on target with `--target-stop A --target-stop-window W` (in-process, clean exit, `--final-full-eval` runs), not
  `--target-acc` (its watcher kills the process group: no final eval, no checkpoint). A `--max-commits` cap also scales the saturation
  warm-up (0.45 × cap), so "no cap" means leaving it unset, never a huge number.

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
- **FT-T10** Don't leave short real `fluxtune` legs in `experiments/`: the profile-staleness preflight counts them and
  blocks every later `fluxtune` sim. Move smokes to their check dir.
- **FT-T11** Don't read RoBERTa-L or 1.7B vclock hours as final: `fluxtune_yelp-p.yaml` predates the §2 recipe
  (`fluxtune_yahoo.yaml` re-profiled 10-25); 1.7B runs before 10-24 charged D 4.1× short (FT-T13).
- **FT-T12** Don't give lanes one `FLAME_RUN_LABEL`: `run_sequential` then merges both legs' summaries (`AGG_KILLED`,
  `2 agg log(s)`); read stop reasons from each aggregator log.
- **FT-T13** Don't scale D by model size alone: the divisor needs the bs factor too (D ∝ bs). 1.7B's 0.0192 missed bs 8 → 32;
  measured 0.0047 (`paper_expts_fluxtune/D_CALIBRATION.md`).
- **FT-T14** Don't time trainer cost with the aggregator's eval on the same GPU: a full-test eval slowed tasks 2.7–3.5× for
  4–8 min (`ft_profile_20261024`); cap or skip eval when profiling.
- **FT-T15** Don't read the ag60 preview's time-to-0.60 as ablation evidence: 1 seed per arm, stopped at the first eval
  ≥ 0.60 (no consecutive rule); `Smom` dipped to 0.28 just before.
- **FT-T16** Don't add server momentum to an `exp` run without re-deriving T_res: Φ ≠ e^B under momentum
  (`FedSgdAggregator.py:433`), so Φ\* 2 no longer holds; 0.9 diverged.

---

## Done
- **FT-D1** syn_0 parity landed on all nine: two-sided floors, real↔real control, median over real legs,
  zero fails — simulate_fwdllm.md §A-§B.
- **FT-D2** Sim barrier redesign (event-driven, no hard-coded waits), profiled charge registry,
  estimator v2 — FWDLLM_DESIGN.md §M/§P, BASELINES.md.
- **FT-D3** fwdllm on the bench (FT-N18, 10-24). *Step 0:* ported without a ledger leg: ag60 step timings showed 1.5 (`fwdllm`) and
  4.9 (`fluxtune`) of 100 trainers training at once on 4 GPUs, every trainer holding its own model + CUDA context; a bench leg measured
  11.0 GB RAM / 3.6 GB GPU (DistilBERT · AG News n 100, 4 workers; 2.3 GB GPU at 2). *Port:* `trainer/main.py` `build_trainer`,
  `main_fedfwd_agg.py` `build_aggregator` (harness `__main__` unchanged); `fl_bench.py` detects the FedFwd stack, applies
  `client_idx_modulo`, one model per worker with each trainer's trainable slice + global RNG swapped around its task;
  `bench_validate.py` adds iterations / `data_id`, variance pass rate, commits / h (±10%). *Gate:* a bench seed replays the
  same-seed harness leg exactly (every commit and eval, seeds 1234-1236 over 8,427 s; ≤ 2e-7 relative float drift); inline, 2 and 4
  workers bit-identical. B2 (`bench_B2_20261024/`, 3 bench vs 3 harness seeds): `fwdllm` syn_0 AUC 32.1 / 31.8, upd / h, GB,
  iterations, var pass, commits / h equal; `fluxtune` syn_0 (1 harness seed) AUC 44.6 / 43.4, all within band. B4 floors at this
  cell: `fwdllm` seeds span AUC 24.6-44.4, upd / h 3,606-7,699, final acc 0.25-0.76 (FT-T7: never read one seed); `fluxtune` AUC
  43.2-45.6. `fluxtune` does not replay same-seed (vclock 82.67 vs 82.70 s by commit 8): its config charges measured wall time
  (`sim_model_agg_compute_time`, truthful gate), so it grades on B2's bands only. Packing: 1 leg / GPU gave 8-9 legs / h on 3 GPUs
  (~20 min legs); denser packing per FX-T74 untested.

- **FT-D4** `main_fedfwd_agg.py` ends with `exit_after_run` (PARITY PL24, FX-D178): 4 of 7 legs had aborted in native
  teardown after `channel leave done`; `test_aggregator_mains_end_with_exit_after_run` guards every aggregator main.
- **FT-D5** Task cost D per model (10-24): [D_CALIBRATION.md](../paper_expts_fluxtune/D_CALIBRATION.md) and
  `profile_task_cost.sh`; RoBERTa-L 0.0169 confirmed, SmolLM2-1.7B 0.0192 → 0.0047 (per arm in its launcher).

## Open questions (operator)
- Does C2 (dynamic K/C) land in the headline `fluxtune` config or get descoped? (BASELINES.md ᵖ note.)
- Should the oracular rows stay on the syn_0 board? Depends on FT-N2.
- Checker invariants I1-I6 were drafted once and lost; what semantics were intended?
- `felix_it`'s ON accuracy sits 1.0-3.4 pts below its OFF band (`felix_round` improved). Low priority;
  falsified if a second loss-derived baseline also degrades ON.
