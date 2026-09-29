# Parity readiness — real↔sim parity, Felix and FluxTune

## Preamble
- **Read [ROBUST_FL_READINESS.md](ROBUST_FL_READINESS.md) first** (doc rules, R-rules, shared L/T). This doc owns
  everything about climbing real↔sim parity: the climbing rules, the method, the two axes, the tools, the ladder
  build and each track's scoreboard. The track docs ([FELIX_READINESS.md](FELIX_READINESS.md),
  [FLUXTUNE_READINESS.md](FLUXTUNE_READINESS.md)) keep their queue (`FX-N`/`FT-N`), lessons and built features.
- **Deprecated parity docs** — [PARITY.md](../async_cifar10/PARITY.md),
  [simulate_fwdllm.md](../fwdllm/simulate_fwdllm.md),
  [PARITY_CHECKER_README.md](../async_cifar10/scripts/parity/PARITY_CHECKER_README.md) — only shrink: move a part
  here when you use it, delete it there, leave a pointer.
- **IDs:** `C#` climbing rules · `Q#` ladder-build tasks. Items themselves live in the track queue.

---

## Climbing rules (operator)

- **C1 Every run closes roots.** Before the next launch, trace every red cell from stored logs to a root; each root
  gets a fix in tree or a queue item with a hypothesis and a prediction. Target: fewest runs to a fully green matrix.
- **C2 Lowest red check per cell first.** A cell's higher fails are presumed downstream (Method). A root failing 2+
  cells outranks one failing 1 (R5).
- **C3 No cell blocks another.** Climb per cell (baseline × dataset × avail/unavail): a cell green at a rung runs the
  next rung in the next run, whatever other cells show. Always `--keep-going`.
- **C4 Logical before timing.** Logical = the same steps in the same order, ignoring timestamps (Axes). Timing closes
  afterwards through profiled charges, never knob tuning (T3, FX-T31). A logical miss on a cell whose timing is also
  red is first checked for clock coupling (sim selects against its vclock; async arrival order follows timing); with
  timing green it is a pure logical root.
- **C5 Loop per run.** (a) Regrade both nodes and rewrite the scoreboard; (b) for each cell's lowest red check read
  both sides' numbers and group cells by root; (c) apply C1 and C6-C8; (d) the next launch states what confirms and
  what refutes each fix (R4); a refuted prediction becomes a tripwire, a confirmed one a lesson.
- **C6 Correct first, equal second.** Real is not the reference by default (R1, R2). For every divergence decide
  which side is wrong from first principles (FL semantics, the reference implementation in `third_party/`, the
  design's stated invariant): fix that side, or both. A check green because both sides share a bug is a bug; hunt
  for these (e.g. both sides picking the same 3 trainers forever, FX-N49).
- **C7 Instrument what you can't see.** When stored logs can't name a root, add the telemetry (a field, an event, a
  checker invariant) in the same change as the next fix, so the next run answers it. A mechanism one side logs
  and the other doesn't is itself a finding (real had no `dispatch` event, so EV10 could not see real re-picks).
- **C8 Launch only when fixes are maximized.** The next run goes out only once every red cell of the latest run has
  a fix in tree or an item whose missing evidence the run will collect (C7), and full pytest is green (R10, R22).
- **C9 Correct fixes default on (operator).** A fix shown correct from first principles (C6) ships default ON with a
  knob to revert; R9's default-off applies to unproven behaviour changes and A/B candidates.

---

## Method (moved from PARITY.md §1/§1.5)

- **Pipeline.** Each round flows `clock → availability → selection → dispatch/train → return/order → aggregation →
  utility → emergent`. A break at stage N makes every stage above diverge as a consequence; the root is the
  **lowest failing check whose inputs match**. Checker stages (`CHECK_META`): 0 telemetry, 1 clock, 2 availability,
  3 selection, 4 dispatch/phases, 5 return/order, 6 aggregation, 7 utility, 8 emergent, 9 budget.
- **Roles.** CONTROL (a stage's input matches), MECHANISM (one transformation — the prize), EMERGENT (an aggregate;
  never fixed directly, walk down). **Tiers.** INV/EXACT hard fail · DIST fails unless lenient; gates only above a
  replicate floor (L12, Q2) · DIAG informational. Dependency gating labels the lowest failing check with passing
  upstreams ROOT and higher fails DOWNSTREAM.
- **Growth rule.** Every root leaves behind the finest check that would have localized it, at its stage.
- **Logical budget (L10).** Grade on the work both sides reached (rounds / `data_id`s, `_matched_logical_budget`),
  never a matched time window: a time window hides throughput gaps and penalizes a legitimate speedup.
- **Stochastic selectors (L14).** Gate marginals (counts, speed class); identity is DIAG. Async pairs diverge in
  per-selection identity early by construction (which slot frees first is timing); read them on marginals.

## Axes

| axis | graded on | checks (checker name) |
|---|---|---|
| **Logical** (gate first) | EV (both legs) + time-stripped round-indexed quantities | `eligibility`, `selection_detail`, `participation`, `selection_bias`, `residence`, `agg_goal_cycles_*`, `aggregation_sequence`, `staleness`, `withheld_delivery`, `utility`, `convergence`, `convergence_loss`; EV17 (real one-in-flight) |
| **Timing** (after) | stage-1 clock + time-to-N | `overhead_residual`, `per_round_advance`, `throughput`, `overlap_factor`, `matched_budget_coverage`, `total_commits`, `terminal_state`; phases (stage 4) |

`inter_arrival_order` is not a gate (stochastic within-round rank). Logical checks are DIST at nominal tolerance until
Q2, so each red one is read before it becomes an item. A timing-metric red on a pair with < ~20 commits is noise.

## Tools

```
L="conda run --no-capture-output -n dg_flame python lib/python/examples/scripts/parity_ladder.py"
$L --rungs L1-L4 --datasets all --keep-going      # CPU node
$L --rungs L5 --datasets all --keep-going         # GPU node
conda run -n dg_flame python lib/python/examples/scripts/parity_ladder.py --grade <pool> --max-stage 1
python lib/python/examples/scripts/logical_diff.py <pool> [--baselines oort]   # first diverging selection/commit
conda run -n dg_flame python lib/python/examples/async_cifar10/scripts/parity/event_invariants.py <run_dir>
```
- `logical_diff.py` prints, per pair, the first selection and commit where the legs stop taking the same steps; at a
  diverging selection it says whether the inputs differ (eligible/decision fingerprints) or the RNG desynced.
- Pulling a node's ladder to jayne (rsync destination without a trailing slash, or it nests):
```
R=/home/dgarg39/flame/lib/python/examples; N=kaylee
LAD=$(ssh $N "ls -dt $R/experiments/ladder_* | head -1 | xargs basename")
rsync -av $N:$R/experiments/$LAD $R/experiments/${LAD}_$N
ssh $N "cat $R/experiments/$LAD/*/*/*/runs/*/legs.txt" | sort -u > /tmp/legs_$LAD.txt
rsync -av --files-from=/tmp/legs_$LAD.txt -r $N:/ /
```

---

## Felix scoreboard

### Status today (2026-09-29, before run 4)

Measured on run 3; every fix below is in tree and unmeasured until run 4 (FX-N52). "Green cells" = cells whose every
leg passes every rung that ran (L1-L5); the four cells are cifar/speech × avail/unavail.

| baseline | logical: green cells | logical: open reds (lowest first) | in-tree fixes → run 4 should show | timing: green cells |
|---|---|---|---|---|
| felix | cifar avail | `staleness` on the other 3 CPU cells | FX-N46 → staleness green | speech avail |
| fedbuff | cifar avail | `eligibility` (both unavail), `staleness` (speech avail) | FX-N50, N46 → both green | none |
| oort | none | L1 EV1 (known, unavail), `eligibility` (L5), `utility`/`selection_bias`/`participation` (L2) | FX-N50, N26, N44, N51, N53 → eligibility green, selections track longer | none |
| oort_star | none (speech unavail green at L2 only) | `eligibility` (L5), `utility`/`selection_bias`/`convergence_loss` (L2) | FX-N50, N26, N44, N51, N53 | none |
| refl | none (cifar and speech avail green at L2 only) | `eligibility` (L5, all cells), `selection_bias`/`participation` (unavail L2) | FX-N50, N49 → eligibility green; ≥ 30 distinct picks in 20 rounds (was 3) | cifar avail |
| feddance | cifar avail, speech avail | `eligibility` (cifar unavail), `utility` (speech unavail) | FX-N50 → eligibility green | cifar avail |

Rung reached by every cell: logical L1 except oort unavail (known EV1); timing none (FX-N43 is next once logical holds).
L4 is green everywhere except known items; L6/L7 have not run at HEAD.

### Run 3 detail (per cell)

One **cell** = baseline × dataset × {avail = syn_0/syn_0b, unavail = syn_20/syn_50/mobiperf_3st}; it passes a rung
only if every leg passes. Each entry: the rung and its **lowest failing check** (✅ = every leg passes); fix the
leftmost red entry per cell (C2). Source: run 3, jayne `experiments/ladder_20260928_214934` (L1-L4) + kaylee
`experiments/ladder_20260928_214901_kaylee/ladder_20260928_214901` (L5). Every red entry below has a fix in tree
awaiting run 4 (root column) except FX-N43 timing.

**Logical** (L1, L4 = EV; L2 CPU pairs, L5 GPU pairs = EV + logical checks)

| baseline | cifar avail | cifar unavail | speech avail | speech unavail |
|---|---|---|---|---|
| felix | ✅ L1-L5 | L2 `staleness` · L5 ✅ | L2 `staleness` · L5 ✅ | L2 `staleness` · L5 ✅ |
| fedbuff | ✅ L1-L5 | L2 `eligibility` · L5 ✅ | L2 `staleness` · L5 ✅ | L2 `eligibility` · L5 ✅ |
| oort | L2 `utility` · L5 ✅ | L1 EV1 (known) · L2 `eligibility` · L5 `eligibility` | L2 `selection_bias` · L5 `eligibility` | L1 EV1 (known) · L2 `participation` · L5 `eligibility` |
| oort_star | L2 `utility` · L5 `eligibility` | L2 `convergence_loss` · L5 `eligibility` | L2 `selection_bias` · L5 `eligibility` | L2 ✅ · L5 `eligibility` |
| refl | L2 ✅ · L5 `eligibility` | L2 `selection_bias` · L5 `eligibility` | L2 ✅ · L5 `eligibility` | L2 `participation` · L5 `eligibility` |
| feddance | ✅ L1-L5 | L2 `eligibility` · L5 `eligibility` | ✅ L1-L5 | L2 `utility` · L5 ✅ |

**Timing** (L2, L5 = stage-1 clock + time-to-N)

| baseline | cifar avail | cifar unavail | speech avail | speech unavail |
|---|---|---|---|---|
| felix | L2 `overhead_residual` · L5 ✅ | L2 `overhead_residual` · L5 `throughput` | ✅ | L2 `overhead_residual` · L5 ✅ |
| fedbuff | L2 `overhead_residual` · L5 ✅ | L2 `overhead_residual` · L5 ✅ | L2 `overhead_residual` · L5 `throughput` | L2 `overhead_residual` · L5 `throughput` |
| oort | L2 `per_round_advance` · L5 ✅ | L2 `matched_budget_coverage` · L5 `overhead_residual` | L2, L5 `overhead_residual` | L2 `overhead_residual` · L5 `matched_budget_coverage` |
| oort_star | L2, L5 time-to-N | L2 `overhead_residual` · L5 time-to-N | L2, L5 `overhead_residual` | L2, L5 `overhead_residual` |
| refl | ✅ | L2 `overhead_residual` · L5 ✅ | L2 `overhead_residual` · L5 `throughput` | L2, L5 `overhead_residual` |
| feddance | ✅ | L2 `overhead_residual` · L5 ✅ | L2 `overhead_residual` · L5 ✅ | L2 `overhead_residual` · L5 ✅ |

L4 is green on every cell except known items (FX-N30, FX-N38); L3 adds no INV/EXACT check over L2; L6/L7 not run at HEAD.

**Roots of run 3's logical reds** (each has a fix in tree; the track queue carries prediction and exit):

| red entries | root | wrong side | item |
|---|---|---|---|
| L5 `eligibility` (all sync), speech GPU oort/oort_star picks | join barrier released at n−8: a fast sim ends before stragglers join | both (input) | FX-N50 |
| fedbuff unavail `eligibility`; real re-picks on 4 baselines (EV17) | real re-picks a trainer whose update is still behind its send-gate | real | FX-N50 |
| felix/fedbuff `staleness` (staleness-0 share real 26% vs sim 6%) | sim frees an ingested trainer's identity before the version bump | sim | FX-N46 |
| oort/oort_star `utility`, `selection_bias`, `participation` | real duration adds the first optimizer ctor (1.3s, FX-N26) and the send-gate wait (FX-N44) | real | FX-N26, FX-N44 |
| refl picks the same 3 of 100 trainers for 20 rounds (both sides) | REFL port never explores unexplored trainers | both | FX-N49 |
| oort/oort_star under-select every round (top-up each round) | exploit pool never augmented below the cut-off | both | FX-N51 |

Timing (all cells): the sim's clock charges are constants fitted on cifar GPU n=300 (FX-N43). Sim runs fast on 23 of
24 speech CPU pairs (10-32% on the red ones); on cifar CPU async runs slow (+17-33%) and sync unavail fast
(−18-27%); GPU cifar is within ±9% except oort. Re-read after run 4: FX-N46/N50 change async overlap.

## FluxTune scoreboard
Parked with the track (FLUXTUNE_READINESS preamble); its live board still sits in simulate_fwdllm.md §A until it moves
here.

---

## Active build — parity ladder (FX-N42)

`examples/scripts/parity_ladder.py` (rungs and gates: `LADDER`; known misses: `KNOWN`, each citing an item; deleting
an item deletes its row). Each rung is one pool (`<out>/<rung>/`, fail-fast on); `LADDER.txt` lists per rung
green/known/red and each red cell's lowest failing rung. Today's runner gate = EV + INV/EXACT up to the rung's stage:
it gates timing and leaves the logical checks (all DIST) ungated, the reverse of C4 (Q4); the scoreboard is built by
hand until then.

| rung | legs | est. wall (cifar / speech) | fixes land here |
|---|---|---|---|
| L0 | static: pytest scoped, collect, data, knob preflight (pool gate) | 4 min | config, imports |
| L1 | T1: sim legs only, 120s, syn_0 + syn_50, EV | 6 / 6 min | sim-only logic |
| L2 | T3: real+sim pairs × 4 shapes | 59 / 82 min | logical + clock |
| L3 | L2's legs re-graded to stage 3 (+ CPU real↔real control → floors: Q2) | 0 (+ ~30 min) | availability, selection |
| L4 | T4 extras: controls P4/P9/P10, streaming P7/P7o, P5/P6/P8, injected P11a-c; EV | 37 / 48 min | streaming, retries |
| L5 | GS: GPU pairs 10 min, G0 cohort, syn_0 + syn_20 | 122 / 238 min (3 GPUs) | GPU-only faults, real overheads |
| L6 | G0C + G0: 30 min + real↔real control; INV/EXACT all, DIST vs floor | ~317 min | noise vs bug |
| L7 | G1/G2 at reference n, 90 min → 3h | ~5h per dataset | sign-off (parent exit criteria) |

**Tasks**
- Q1 · done: runner, GS tier, offline `--grade`, `KNOWN`/`EXPECTED_FAIL`, `tests/harness/test_parity_ladder.py`.
- Q2 · todo: CPU real↔real control per L2 cell + floor derivation into `floor_gated_tol` (parent S2), so DIST gates.
- Q3 · todo: `--grade` re-runs the checker on stored pairs.
- Q4 · todo: two axes per cell in the runner (`LOGICAL` at nominal tolerance until Q2, `TIMING`); `LADDER.txt` prints
  the two scoreboard tables.
- Q5 · todo: per-cell scheduling (C3): each cell starts at its lowest red rung; the pool takes a cell list.
- Q6 · wip: `scripts/logical_diff.py` names the first diverging selection/commit per pair; next: per-round marginals for
  async pairs, and into the runner's report.

## Run length (moved from simulate_fwdllm.md §C)
Duration follows the residual's shape (R6): a per-cycle mechanism is fully present in cycle 1; an accumulating one
reads a false PASS on a short run. Telemetry sanity 5-10 min · clock family (`overlap_factor`, `throughput`,
`per_round_advance`, `overhead_residual`) 15-30 min · one mechanism rung 30 min · stochastic identity /
participation 60 min+ · convergence sign-off 2h+. A check carrying `matched_logical_budget_n` grades only the shared
work, so a short run gives it a thin N; one without it grades at full strength immediately (every INV).
