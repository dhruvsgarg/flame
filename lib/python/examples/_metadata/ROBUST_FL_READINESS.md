# Robust FL readiness — parent doc (read first, every session)

**Goal:** run Felix and FluxTune, with all their baselines, in the simulator with high fidelity and a real
speedup, with and without client unavailability, so paper experiments run sim-only and fast.

**Three docs are the only working references:**
- **this doc** — the rules for all three docs, the lessons and tripwires both tracks share, and the queue of
  shared infrastructure work;
- [FELIX_READINESS.md](FELIX_READINESS.md) — Felix: backprop FL (async_cifar10, google_speech). **CURRENT FOCUS.**
- [FLUXTUNE_READINESS.md](FLUXTUNE_READINESS.md) — FluxTune: perturbation / forward-gradient fine-tuning
  (fwdllm). **PARKED** until Felix experiments are done; shared changes must still keep it green (R10).

**Every older doc is DEPRECATED** (PARITY.md, UNAVAILABILITY_DESIGN.md, simulate_fwdllm.md, FWDLLM_DESIGN.md,
PARITY_CHECKER_README.md, BASELINES.md, the `docs/`, `fl_fwd_ft_*` and `paper_expts_fluxtune/` files; each
carries a banner saying so). They keep derivations, rung catalogs and history for reference, and they only
ever SHRINK: when you use or change something in one, move the live part here and delete it at the source in
the same edit, leaving a one-line pointer where code or another doc cites the section. Never add to them.

**Self-goal: context clarity and crispness.** A fresh session must be able to act from these three docs
alone, in minutes. Prefer one exact line to a paragraph; delete before adding; never restate what another
section already says.

**Read order at session start:** this doc top to bottom → the active child's preamble → its **Next steps**
→ only the lessons and tripwires tagged for the area you are touching.

---

## Doc rules (apply to all three docs)

1. **Live ledger, not a log.** Every line must be true *now*. No dated "update:" notes, no changelogs, no
   narrative. `git log` is the history.
2. **Fixed sections, fixed purposes.** Each child has: Preamble · Status grid · Next steps · Lessons (dos) ·
   Tripwires (don'ts) · Done. Content goes in exactly one section.
3. **Next steps are the persistent queue across sessions.** An item is `ID · action · exit criterion ·
   state` (`todo` / `wip` / `blocked: <why>`). Ordered; the top item is what the next session picks up.
4. **Close means move.** When an item finishes, delete it from Next steps and add ONE line to Done
   (`ID · what landed · evidence/commit`) in the same edit. When Done passes ~15 lines, fold the oldest
   into one summary line.
5. **Lessons are dos, tripwires are don'ts.** Each is at most 30 words, carries an ID, and is edited in
   place — never a near-duplicate. A falsified hypothesis becomes a tripwire; a confirmed pattern becomes a
   lesson; anything else is deleted.
6. **Put it at the lowest scope that is true.** It goes here only if it holds for BOTH backprop and
   forward-gradient training. Otherwise it goes in the child. Promote an entry here when the second track
   confirms it.
7. **Evidence, not argument.** Every status cell, lesson and Done line cites a run dir, a parity JSON, a
   test, or a commit. "Should work" is `todo`, not done.
8. **IDs are stable.** Never renumber or reuse one; other lines cite them. Retire an ID by deleting its line.
9. **Budget.** Parent ≤ ~250 lines, each child ≤ ~300. Over budget means Done or the lessons need folding.
   Size is a symptom of logging.
10. **Whole-doc pass on every edit.** Re-read the section you touched and push down or delete anything the
    new fact supersedes, in the same edit.
11. **Open questions for the operator** live at the bottom of the child as one line each and are removed
    when answered (the answer goes wherever it belongs: a lesson, a next step or the grid).
12. **Instructions live here, on git.** Every standing instruction from the operator goes into these docs
    (or CLAUDE.md), never only into an agent's private memory. Private memory may only point here.

## Operating rules (non-negotiable, both tracks)

- **R1 Correctness first; passing tests and parity are consequences.** If something looks wrong against
  FL/systems first principles, stop, raise it, and fix the concept, even when every test and rung is green.
  A rung green because both sides are equally wrong is a regression. Pair every relative check with an
  absolute one (ground-truth trace, invariant).
- **R2 Real is the reference only after it is shown admissible.** A divergence names two sides that
  disagree, not which one is wrong. Check each side's own signal before changing either.
- **R3 Evidence ladder: telemetry on disk → local harness/bench → short cluster run → long run.** Never
  spend a GPU run on a question the harness or stored telemetry can answer. A long run confirms; it never
  finds the first bug.
- **R4 The operator launches everything that spawns processes.** Claude runs only short foreground,
  single-process work itself (pytest, static checks, parsing stored runs). Anything that starts brokers,
  aggregators, trainers or a multi-test sweep — local harness or cluster — goes into ONE script the operator
  launches (killing that parent kills every child). The script writes all results under one
  `experiments/<suite>_<ts>/` dir with a summary file; Claude reads that dir afterwards. Claude hands over the
  command, the expected duration and the prediction (what result confirms and what refutes). Assume 3-4
  nodes; a baseline's real and sim legs run one at a time on one node.
- **R5 One mechanism per run** when a fix could perturb another baseline. Shared roots before per-baseline
  roots: a bug failing rungs on 2+ baselines outranks one that fails on 1.
- **R6 Run length follows the residual's shape.** Per-cycle mechanisms show up in 15-45 min. Accumulating or
  compounding ones need 90 min to 3h+. Convergence sign-off needs a full run that crosses every boundary
  that matters.
- **R7 Control before mechanism.** Run the real↔real (and sim↔sim) control and read the replicate floor
  before naming a cause for any DIST residual.
- **R8 Ship each new mechanism with its telemetry, plot and pytest in the same change.** A field nothing
  reads is dark data.
- **R9 Config-gated, default-off, byte-identical when off** for every sim or real behaviour change.
  Promote to default only with A/B evidence and operator sign-off.
- **R10 Shared code keeps both tracks green.** Any change under `flame/` (aggregators, selectors, channel,
  availability, launch) runs the full pytest (below) plus the harness smoke for Felix AND FluxTune
  baselines before it lands.
- **R11 Fix the concept, not the symptom; no hacks.** A number moved without a correct mechanism is a
  regression in disguise. Never tune a baseline-defining knob to close a parity gap. When unsure, stop and
  ask.
- **R12 Reuse by default, rewrite when reuse hurts.** Code, plotting, harness and test infrastructure are
  shared so an improvement in one project lands in the other (propagate in the same workstream, S-queue).
  When shared code is complex or a nuisance to maintain, simplify or reimplement it instead of stacking
  special cases onto it.
- **R14 Harness first.** Every code change is validated on the no-GPU harness before any GPU run:
  `scripts/harness_campaign.sh` (unattended, all baselines x traces, ~4.5h) or a targeted
  `scripts/harness_suite.sh`. It grades each leg against the paradigm's ground-truth events
  (`scripts/parity/event_invariants.py`) and each pair for parity. GPU real/sim runs come only after the
  harness is green for the affected baselines and traces. Every scripted step carries a hard timeout so one
  hang never stalls the rest; long runs are sized so the operator can launch and sleep.
- **R15 Code hygiene.** One vocabulary everywhere (L20). Comments are one crisp line, only where the code
  can't say it. Dead code: delete when tests prove it dead, else mark `# DEAD?(<ID>)` and list it in a queue.
- **R16 IDs, not prose.** Issues and tests may carry alphanumeric IDs; the description lives ONCE (the
  docstring or its doc row); code comments cite the ID only.
- **R17 No redundancy.** Define once, fix once, and refer to it (file:symbol, test, commit, doc ID) everywhere
  else. Never restate a fact across docs and code; the readiness docs point, they don't copy.
- **R18 Fast, never at correctness' cost.** Use every idle core (pytest `-n auto`, spare cores widen each
  trainer's pin block, parallel analysis). Never oversubscribe cores or share a broker/node between graded legs.
- **R19 Smoke before hand-off; fail in minutes.** Before handing over any long script, Claude runs its ≤5-min
  smoke itself (the one R4 exception: foreground, hard timeout, one script): `pytest --collect-only` plus one
  short real/sim pair showing commits on both legs. Every long script opens with that gate (campaign: P00)
  and aborts when it fails. Never hand over a command that has not run end-to-end once at small scale.
  Every run and test uses the `dg_flame` env (`FLAME_CONDA_ENV`), never the active shell's.
- **R13 Commits:** follow CLAUDE.md (crisp comments, minimal diff, short title and body). Confirm before any
  push.

**Parity exit criteria (per baseline).** Parity is done when the simulator cannot change a paper's
conclusion, not when every rung is green:
1. every INV/EXACT rung is green;
2. convergence and terminal state are inside the baseline's own replicate band;
3. every remaining DIST residual is common-mode (the same across baselines);
4. no residual correlates with a knob that distinguishes one baseline from another.

Signs of over-optimizing: chasing a residual that sits inside its own control; the board changing from
measurement edits rather than fixes; instrumentation growing faster than bugs close.

**Test inventory** (generated, never hand-kept): `python lib/python/examples/scripts/test_inventory.py
[--ref <git-ref>]`: test counts per suite and track.

**Full pytest** (three suites; the checker's tests live beside it):
```bash
conda run -n dg_flame python -m pytest lib/python/tests lib/python/examples/fwdllm/expt_scripts \
    lib/python/examples/async_cifar10/scripts/parity -q
```

---

## Shared lessons (dos)

Tags: `[clock]` `[order]` `[slot]` `[select]` `[avail]` `[measure]` `[floor]` `[ops]`.

**Simulator mechanics**
- **L1 `[clock]`** Sim does the real compute but charges modeled time: `sct = send + max(gpu, D)`, and
  `vclock = max(vclock, sct)`. Never put overhead on the vclock unless it is a profiled charge.
- **L2 `[clock]`** All sim time (availability lookups, abandon timeouts, scarcity waits) runs on the vclock.
  Never use wall clock or a frozen per-trainer clock in sim.
- **L3 `[order]`** Commit in `(sct or delivery_ts, end_id)` order from a complete snapshot of in-flight
  updates. A stranded update gets lapped by the clock and commits past-dated.
- **L4 `[slot]`** One-in-flight per trainer: a busy trainer holds its slot until its update commits. Busy,
  unavailable and withheld are three separate states.
- **L5 `[slot]`** Keep CAPACITY (who holds a slot) and IDENTITY (who can't be re-picked yet) as two sets.
  One set serving both roles hides a bug until a rule changes one of them.
- **L6 `[select]`** A duration fed to a selector is the client's intrinsic span (`WALL_SEND − WALL_RECV`).
  Any aggregator-stamped endpoint smuggles in a server wait (single source: `client_duration.py`).
- **L7 `[select]`** A ported controller or score term that diverges and compounds is usually an unfaithful
  port. Diff it line by line against `third_party/` for every baseline and every selector hand that uses
  it.
- **L8 `[select]`** A score term that is byte-zero for a whole run is a dead input, not a quiet term. Check
  the property it reads is populated, and that no subclass dropped the parent's stamp.
- **L9 `[ops]`** A knob must be logged identically in config, snapshot and both roles' telemetry. Before a
  long run, prove the knob reached the trainer with a dry-run plus a grep of a short leg.

**Measurement and grading**
- **L10 `[measure]`** Grade on the matched LOGICAL work budget (rounds / data_ids both sides reached), never a
  matched time window. Each leg reads its own clock.
- **L11 `[measure]`** Walk the ladder: fix the lowest failing rung whose upstream checks pass. Emergent rungs
  are never fixed directly.
- **L12 `[floor]`** A tolerance means something only above the pipeline's own same-code replicate spread.
  Replicate both sides (n≥3 if unpinned), gate on the max of the two floors, and measure the floor by calling
  the rung itself.
- **L13 `[floor]`** Grade a sim leg against all same-code real legs and take the median. Fails on a minority
  of legs = the draw; fails on a majority = the code.
- **L14 `[measure]`** For a stochastic selector, enforce the marginals (participation by speed class, counts);
  demote per-trainer identity to diagnostic. Confirm there is no bias before demoting.
- **L15 `[measure]`** Decompose before fixing: split a rate into channels (eligible fraction × P(select |
  eligible); per-cycle cost × cycles per unit). A shared symptom can have disjoint roots.
- **L16 `[measure]`** When the mean is fixed but the tail is not, partition the tail by run fraction. A
  gap that grows = compounding; a flat one = constant bias.
- **L17 `[ops]`** A derived artifact (floor, charge profile) records what it was derived from (runs, code,
  hosts, duration, tool version), and is re-derived after any change to training knobs or hardware.
- **L18 `[ops]`** Legs are replicates only if they ran the same run-affecting code, flags, length and
  achieved span. Quarantine dead legs (`experiments/_aborted/`); a dead leg grades as a near-total SKIP.
- **L19 `[ops]`** Analysis over 2 min is a tool bug: run in parallel per baseline, cache the parse, stream the
  output. Never `| tail` or `conda run` a long job.
- **L20 `[ops]`** Names are context-free: `_round` is an int index, `_ts`/`_time_s` is a time; say whose round
  (aggregator vs selector vs per-trainer); clients do tasks, not rounds. Rename instead of commenting.
- **L21 `[measure]`** After a fix to the REFERENCE (real) side, re-run both and check the gap didn't just
  change sign; a new lowest rung is a deeper mechanism, not a regression.
- **L23 `[slot]`** Anything keyed on a LEARNED per-trainer value (known delay) is blind to first-contact
  trainers. Key "is it busy / may it complete earlier" on the dispatch itself.
- **L24 `[ops]`** A test fixture must build config values the way production does (through the config
  parser). A string `"True"` fixture hid a bool-coerced flag that dropped every real withheld update.
- **L25 `[measure]`** Grade a single run against the paradigm's own ground truth (registry D, lifecycle
  order, cadence), not just against its twin: two runs can agree on a bug the absolute check catches.
- **L22 `[measure]`** A borderline EXACT rung can become the root once an upstream DIST rung converges with
  run length. That is the next rung surfacing, not a regression.

## Shared tripwires (don'ts)

- **T1** Don't route busy → UN_AVL or free a slot before commit (in-flight ramps to N, clock crawls).
- **T2** Don't order commits by `sct` when a delivery time exists; don't backdate a re-dispatch stamp.
- **T3** Don't add a scalar overhead to the vclock to close a throughput gap (it masks and drifts).
- **T4** Don't use `min(vclock, wall)` as a matched window; it conflates the two clocks under test.
- **T5** Don't tune `pacer_delta`, `round_threshold`, `var_threshold` or iteration caps to close a gap.
  Those knobs define the baseline.
- **T6** Don't read a green rung as correct without its control, or a red one as a bug before its control
  has run.
- **T7** Don't widen or re-window a failing rung if a sibling baseline passes it on the same code path.
- **T8** Don't pool legs across commits, run lengths or configs, or build a floor from 2 legs on an
  unpinned baseline.
- **T9** Don't copy `parity_floors/` or `sim_charge_profiles/` between nodes; re-derive them from the run dirs.
- **T10** Don't chain a grader that exits non-zero on findings with `&&`.
- **T11** Don't blame GPU contention below ~100 trainers; at n≥100 it is a real root for shared-compute
  wall rungs.
- **T13** Don't compare a config flag to a string (`== "True"`); pydantic coerces it to bool. Normalize
  with `str(x).lower() == "true"`.
- **T14** Don't run module-level code that calls `sys.exit` in a `test_*.py`; it aborts xdist collection for
  the whole suite. Wrap it in a test function plus `__main__`.
- **T12** Don't add a fix to one copy of a shared concept (pacer, drain, residence) without checking every
  copy.

---

## Shared infrastructure queue (S-steps)

Cross-cutting work that serves both tracks. Felix drives it now (R12); each item must leave FluxTune
green (R10).

- **S1 · No-GPU local harness · wip (Felix half landed, FX-N1; campaign re-run pending).** Landed:
  stub/tiny_cpu modes, `FLAME_TRACE_TIME_SCALE` (compresses availability traces by the delay factor, all
  consumers via `load_trace`), the single-run event checker (EV0-EV14), `harness_suite.sh`,
  `harness_campaign.sh`. Harness caveat: absolute timeouts (90s abandon, join) are not scaled. A trainer with two switchable modes. *Stub mode:* no training, seeded
  fake weights/grads, seeded synthetic per-trainer loss/utility, honours `D`. *Tiny-CPU mode:* a small model
  actually trained on CPU. Real MQTT and sim paths, n ≤ 50, pair piped through the parity checker; a pytest
  marker runs a short smoke per baseline. Plus a **scenario library** of fast deterministic cases:
  syn_0/20/50 and 3-state traces, scarcity/starvation, stragglers, mid-flight drop-off, eval hand, lap/round
  boundaries. *Exit:* all six Felix baselines + the fwdllm family (stub) run end-to-end real+sim locally, and
  each of three injected bugs (drop the one-in-flight hold; order by `sct` instead of delivery time;
  freeze the trainer clock) is caught by an INV/EXACT rung.
- **S2 · One parity pipeline for every example · todo.** Generalise fwdllm's `run_parity.py` /
  `replicate_floor.py` / `profile_sim_charges.py` / preflight so they drive async_cifar10 and google_speech
  too: auto-pairing on flag + length + commit, two-sided floors, median-over-real-legs, `--control`,
  threshold-provenance ratchet. *Exit:* async_cifar10 stored Jun 23-24 pairs re-grade through it; the
  fwdllm board is unchanged.
- **S3 · One launcher and one run driver · todo.** Everything runs through `flame.launch` + `baselines.yaml`.
  Merge `async_cifar10/scripts/debug_run.sh` and `fwdllm/expt_scripts/run_sequential.sh` into one
  example-agnostic driver (preflight, dry-run, trace substitution, sim/real pairing). Dump the resolved
  trainer config, and each leg's floor/charge-profile inputs, into the run dir. Move the preflight out of the
  `run_sequential.sh` heredoc into an importable, tested module. Reject dead legs (never reached the first
  commit) in the shared leg-discovery helper. Auto-skip a broken GPU ordinal. *Exit:* both examples launch
  real+sim pairs through the one driver.
- **S4 · Legacy config removal · awaiting operator sign-off per row.** Inventory:
  | path | what | proposal |
  |---|---|---|
  | `async_google_speech/trainer/config_*` (3.5k JSON, 33M) + `aggregator/*.json` + `expt_runs_*.sh` | 2024 per-trainer configs | delete after FX-N10 lands the launcher path |
  | `async_cifar10/launch/` | stale, unused copy of `flame/launch` | delete |
  | `async_cifar10/{socc24_mlsys25,euromlsys25}_expts/`, `expt_scripts_2026/{scripts,configs}` | DEPRECATED shell/JSON runs | delete |
  | `async_cifar10/{eurosys26,socc26}_expts/` (10G) | paper logs/plots/notebooks | archive outside the repo, keep plots |
  | `async_cifar10/wandb`, `aggregator/wandb` (17G) | wandb run caches | delete (untrack) |
  | `fwdllm/expt_scripts/probe_*`, `writeup_figs/data` JSON | probe/figure DATA, not configs | keep |
  Plus the `trackTrainerAvail` blocks (FX-N7). *Exit:* no per-trainer JSON or `pytorch/main*.py` shell path left.
- **S5 · Knob contract · todo.** Declare which knobs apply to which baseline in `baselines.yaml` and enforce
  that in `test_baseline_readiness.py`, the launch preflight and `--validate`, so a missing knob is caught
  before a run.
- **S6 · Sim perf pass · blocked: on Felix parity (FX board green).** Gate diagnostic logging, trim per-commit
  JSONL, pipeline the serial per-commit aggregator path. Every perf commit re-runs the harness and a 90-min
  parity pass on all baselines.

- **S7 · Logging hygiene, all examples and baselines · todo.** Replace every `print` with the module logger at
  the right level; the default file log carries INFO+ that matters, DEBUG is opt-in. Fold into S6 (diag
  gating) but land per example, harness-green each time.

## Shared Done
- **S0** `dg_flame` → torch 2.12.1+cu129 / torchvision 0.27.1+cu129 (driver 12.9); cu13 libs removed; jayne 8/8 GPUs
  verified (matmul, cuDNN, ResNet). Other nodes unchecked.
- Parity methodology (causal ladder, roles/tiers, dependency gating), availability substrate v1, and the
  fwdllm floor/control/median grading all landed — derivations in PARITY.md §1-§5, UNAVAILABILITY_DESIGN.md,
  simulate_fwdllm.md §A-§D.
