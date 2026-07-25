# FluxTune paper — plot tracker (living doc)

**Purpose.** Per-figure-set readiness: which baselines a paper figure needs, which
run dirs currently back them, how far each has progressed, and what's still
missing. This is the single place to check "what's left before I can regenerate
the plots" — narrower than [`EXPERIMENTS.md`](EXPERIMENTS.md) §10a (which maps
tex labels → experiment ids → readiness at the *metric* level) and §10b (the
full run ledger); this doc is baseline-readiness at the *plot-manifest* level.
**Owner:** dgarg39. Re-sync this doc's progress numbers whenever new logs land —
they are a snapshot, not live.

**Key architectural fact (EXPERIMENTS.md §0):** all 7 figure basenames in a set
(`e1_acc_vs_time`, `e1_loss_vs_time`, `e2_trainer_busy_cdf`,
`e3_dloss_per_gpu_hour`, `e3_dloss_per_mfwd`, `e4_network_bytes`,
`e5_session_cdf`) are reducers over the **same** manifest/run-set — landing one
baseline's run dir unlocks it for all 7 plots at once. So readiness is tracked
**baseline-wise per manifest**, not per individual plot — one row below feeds
seven figures.

Two manifests = two `evaluation.tex` subsections (§7 of the original design
ask): `figs_main_v2.yaml` is the **top-row published-anchor comparison**
(`sec:eval:sota`); `figs_attribution.yaml` is the **second subsection**, FluxTune
vs. the round→iteration *derived* (`+IT`/`+IT+O`) baselines (`sec:eval:attribution`).

---

## Manifest 1 — `expt_scripts/figs_main_v2.yaml` (`sec:eval:sota`, top row)

Feeds: `e1_acc_vs_time`, `e1_loss_vs_time`, `e2_trainer_busy_cdf`,
`e3_dloss_per_gpu_hour`, `e3_dloss_per_mfwd`, `e4_network_bytes`,
`e5_session_cdf` → `paper_figs_main_v2/`.

| Baseline | Run dir | `agg_eval` count (at render) | peak test-acc | Status |
|---|---|---:|---:|---|
| `fwdllm` | `run_20260724_023627_fwdllm_n100_smoke_syn_0_sim` | 292 | 84.97% @ 2.13h | ✅ **rendered** 2026-07-24 14:53 |
| `fedbuff_round` | `run_20260724_093607_fedbuff_round_n100_smoke_syn_0_sim` | 410 | 85.54% @ 4.18h | ✅ **rendered** — added this sync, supersedes the dead `run_20260724_000757_…` attempt (0 `agg_eval` events, no live process backing it) |
| `felix_round` | `run_20260724_042943_felix_round_n100_smoke_syn_0_sim` | 381 | 84.63% @ 3.95h | ✅ **rendered** — supersedes the dead `run_20260724_015555_…` attempt (only 11 events) |
| `fluxtune` | `run_20260724_023501_fluxtune_n100_smoke_syn_0_sim` | 124 | 84.72% @ 0.80h | ✅ **rendered** |
| `fwdllm_it_oracular` | — | — | — | ➖ **intentionally not run** (confirmed by operator 2026-07-25) — oracular availability-tracking is inert under `syn_0`'s 100% availability (identical to `fwdllm_it_unaware`), so this condition needs no oracular run. Not a gap. |

**This manifest is COMPLETE — 4/4 target baselines rendered** 2026-07-24 →
`expt_scripts/paper_figs_main_v2/` (all 7 basenames, `--cutoff-mode peak_acc`;
flat/overwrite dir, no timestamped subdirs — see "Regenerating" below).

All four landed baselines have already peaked ≥ the 84% target (consistent with
the known Issue I-1 oscillation/collapse pattern, EXPTS_CHARTER.md) — cutoff
plotting (`--cutoff-mode peak_acc`) still applies.

---

## Manifest 2 — `expt_scripts/figs_attribution.yaml` (`sec:eval:attribution`, +IT staircase)

Feeds the same 7 basenames → `paper_figs_attribution/`.
`speedup_baseline: fwdllm_it_unaware` (the *strengthened* floor, not raw `fwdllm`
— this subsection's point per `evaluation.tex`).

| Baseline | Run dir | `agg_eval` count (at render) | peak test-acc | Status |
|---|---|---:|---:|---|
| `fluxtune` | `run_20260724_023501_fluxtune_n100_smoke_syn_0_sim` | 124 | 84.72% @ 0.80h | ✅ **rendered** |
| `fwdllm_it_unaware` | `run_20260724_053835_fwdllm_it_unaware_n100_smoke_syn_0_sim` | 222 | 85.26% @ 1.56h | ✅ **rendered** — this was the **only** comparison point at the 11am generation, which is why that figure set showed just fluxtune-vs-fwdllm_it_unaware |
| `felix_it` | `run_20260724_092656_felix_it_n100_smoke_syn_0_sim` | 291 (was 2 as of 09:33) | 84.49% @ 2.35h | ✅ **rendered** — was flagged "too early to plot" at the 11am generation; now far enough along |
| `fedbuff_it_unaware` | `run_20260724_144822_fedbuff_it_unaware_n100_smoke_syn_0_sim` | 95 (at peak-acc cutoff; 112 raw) | 79.51% @ 2.99h | ✅ **rendered** 2026-07-25 — supersedes the dead `run_20260724_001229` attempt (16 evals, 39.5%); 144822 is the 0230-batch relaunch (launched 14:48, still inside the original long-run window, well before the unrelated post-17:30 real-sim parity batch — see note below) |
| `fwdllm_it_oracular` | — | — | — | ➖ **intentionally not run** (confirmed by operator 2026-07-25) — oracular tracking is inert under `syn_0`'s 100% availability, identical to the landed `fwdllm_it_unaware` twin. Not a gap. |
| `fedbuff_it_oracular` | — | — | — | ➖ **intentionally not run**, same reason. (A short post-17:30 `fedbuff_it_oracular` sim run, `run_20260724_182737`, exists but belongs to the separate real-sim parity batch, `simulate_fwdllm.md` — unrelated to this decision, not reused here.) |

**This manifest is COMPLETE — 4/4 target baselines rendered** (the two
`_oracular` rows were never targets, not missing rows). Rendered 2026-07-24,
refreshed 2026-07-25 (added `fedbuff_it_unaware`) →
`expt_scripts/paper_figs_attribution/` (all 7 basenames, `--cutoff-mode
peak_acc`; flat/overwrite dir, no timestamped subdirs — see "Regenerating"
below).

### Why the attribution plot only had one comparison baseline at first

Not a script/mapping bug — `figs_attribution.yaml`'s other rows were
genuinely commented out because those runs either didn't exist yet or hadn't
progressed far enough to plot at the time of the 11am (2026-07-24) generation.
Re-checking `experiments/` telemetry that day showed `felix_it` had progressed
from 2 to 467 evals, so it was promoted; `fedbuff_it_unaware` was still at 16
evals and stayed parked. As of the 2026-07-25 sync, the original 0230-batch
`fedbuff_it_unaware` relaunch (`run_20260724_144822`) has landed 112 evals /
79.5% peak, so it's promoted too — **4/4 target baselines now render.** The two
`_oracular` legs were never additional rows to fill: confirmed 2026-07-25 that
oracular tracking is intentionally not run under `syn_0` (inert at 100%
availability), so this manifest's target was 4 baselines all along, not 6.

---

## Regenerating

```bash
cd lib/python/examples/fwdllm/expt_scripts
python make_paper_figs.py --manifest figs_main_v2.yaml     --out-root paper_figs_main_v2
python make_paper_figs.py --manifest figs_attribution.yaml --out-root paper_figs_attribution
```

Missing baselines are skipped, not fatal — a manifest with 3 of 4 rows filled
still renders every figure, just with fewer bars/curves.

---

## ⏳ OPEN WORK (temporary tracking section — 2026-07-24 batch, remove/fold in as each item lands)

Three items remain open from the original six (2026-07-24 planning
discussion). Update the status inline as each moves; delete an item's block
once it's fully landed and folded into the relevant doc (tex, EXPERIMENTS.md
§10a, or the manifest tables above) — this section is meant to be temporary,
not a permanent seventh doc. **Purged 2026-07-24**: items #2 (palette-preview
script — landed, command now documented in `EXPTS_CHARTER.md`'s "Latest
figures" section), #3 (auto broken y-axis — landed in `fig_e1_acc_vs_time`,
x-axis explicitly scoped out, not a pending task), #4 (CDF P50/P90 — landed
in `fig_e2_trainer_busy_cdf`/`fig_e5_session_cdf`). All three are covered by
`plotlib/test_figures.py` and committed; see git history for detail instead
of this doc.

### 1. tex ⇄ EXPERIMENTS.md §10a ⇄ PLOT_TRACKER.md are out of alignment
**Status:** wiring fixed 2026-07-24; one follow-up decision left for the operator (not a wiring bug).

Fixed: created the five missing `figs/code/eval/e2e/{e1_tta,e2_util,e3_compute,
e4_comm,e5_sessions}.tex` fragments `evaluation.tex`'s `sec:eval:sota` already
`\input`s — each `\includegraphics`s straight from `expt_scripts/
paper_figs_main_v2/*.pdf` (no copy step, stays in sync on every regen).
Added the missing `sec:eval:attribution` hook: four new `figs/code/eval/attr/
{a_tta,a_util,a_compute,a_comm}.tex` fragments (no sessions fragment — the
attribution prose never discusses session length) with their own
`fig:eval:attr:*` labels pointing at `paper_figs_attribution/*.pdf`,
`\input`-ed at the right point in each attribution paragraph, and the
subsection's `\figref`s repointed off sota's labels onto the new ones.
`EXPERIMENTS.md` §10a's E1–E5 rows now read ✅ RENDERED and its A0 row was
rewritten to cite `figs_attribution.yaml` (this doc's Manifest 2) directly
instead of free-standing prose describing a stale 2-baseline comparison.

**Left open — a claims decision, not wiring:** while rewiring `a_tta.tex`
(flagged inline as a `\tbd`), found that the attribution prose's numbers
("`\fwdllmito{}` peaks at 80.9%, takes 7.4h") are for the **oracular** `+IT+O`
variant, but `fwdllm_it_oracular` has **no run at all** (Manifest 2 row 5,
❌ NOT LAUNCHED) — the baseline actually landed and rendered is
`fwdllm_it_unaware` (85.26% @ 1.56h, `+IT` not `+IT+O`). Did not silently
swap the prose's numbers or baseline name — that's a scientific/editorial
call (do we launch the oracular run, or rewrite the claim to `\fwdllmit{}`?),
not something to guess at. Ask the operator before touching those numbers.

### 2. New bar plot: total perturbations to reach target accuracy, per baseline
**Status:** grounded 2026-07-24 — (a) confirmed sufficient, no new counter
needed. Still not implemented (plot itself is a separate follow-up); this is
the requested grounding pass only.

Findings:
- **(a) Direct instrumentation is not just "mostly there" — the raw counters
  are already fully there.** `perturbations_total`/`forward_passes_total`
  (`fwdgrad_utils.py` `_JVP_EVALS`/`_FWD_PASSES` → `trainer_round`) are
  monotonic per-trainer cumulative counters with a `ts` on every event.
  `plotlib/reducers.py`'s `_read_trainers` already sums each trainer's max
  counter value at events with `ts <= cutoff` into `RunResult.pert_total`/
  `fwd_total` — exactly the "sum up to a timestamp" operation this item needs, just
  parameterized by the wrong cutoff today (`_compute_cutoff`'s run-wide
  peak-acc/plateau cutoff, not "first eval crossing the target accuracy").
  The missing piece is small and reducer-side only: a new cutoff mode (or a
  standalone `pert_total_at_target(target)` helper) that finds the first
  `agg_eval` crossing `target` — the same event `figures.py`'s
  `_target_crossing_hr` already computes for the E1 speedup callouts — and
  reuses `_read_trainers` with that timestamp as `cutoff`. **No new trainer
  log line needed.**
- **Live-telemetry re-validation, not just a code read**: spot-checked
  `perturbations_total`/`forward_passes_total` directly in
  `fedbuff_round`'s and `felix_round`'s `trainer_*.jsonl` files (the two
  baselines `BRIDGE_DESIGN.md`'s §2b per-baseline checklist still marks
  "⬜ smoke-pending" for WS3-b) — both fields are present and non-null on
  real events (`fedbuff_round`: `perturbations_total: 1`,
  `forward_passes_total: 5` on an early event). **Flips WS3-b to validated
  for these two baselines** — update `BRIDGE_DESIGN.md`/`EXPERIMENTS.md` §5's
  checklist next time either doc is touched.
- (b) not needed — (a) is sufficient, per the finding above.

Paper-text grounding (what claim this would support) not yet done — still
needs a pass over `evaluation.tex` to confirm where "perturbations to target"
would land before the plot itself is built.

### 3. Investigate: iterations-per-data-id trend vs. baseline / round-vs-iteration control
**Status:** grounded 2026-07-24 — real, sample-based signal found. Still
exploratory; not promoted to a figure spec (needs operator sign-off per the
"do the grounding work only" scoping).

Pulled `trainer_round.iteration_per_data_id` (not `agg_eval`'s copy — the
per-trainer telemetry, sampled 5/100 trainer files per baseline, first/last
20k lines per file as an early-vs-late proxy — a sample, not the full run)
from the six landed `main_v2`/`figs_attribution.yaml` baselines:

| Baseline | control | early mean | late mean | Δ |
|---|---|---:|---:|---:|
| `fwdllm` | round | 10.55 | 16.17 | +53% |
| `fedbuff_round` | round | 3.73 | 9.35 | +151% |
| `felix_round` | round | 8.22 | 17.93 | +118% |
| `fwdllm_it_unaware` | iteration | 9.86 | 25.00 | +154% |
| `felix_it` | iteration | 11.66 | 23.28 | +100% |
| **`fluxtune`** | **async/continuous** | **8.22** | **8.62** | **+5%** |

**Signal**: every round- and iteration-controlled baseline's per-databin
iteration count roughly doubles (or more) from early- to late-run, while
FluxTune's stays essentially flat. Reads as consistent with FluxTune's
continuous reselection spreading iterations evenly across data bins over
time, instead of the pileup-on-slow-bins pattern round/`+IT` control produces
— but that's a hypothesis from a 5%-file sample, not a validated finding.
Not yet a plot spec: needs (a) the full 100-trainer-file pull (this was
capped to avoid the ~2-4GB-per-baseline read timing out), (b) a decision on
whether this is a standalone figure or a footnote in the attribution
subsection's mechanism discussion, and (c) operator sign-off before either.

---

## Changelog

- **2026-07-24** — created. Added `fedbuff_round` (093607 relaunch) to
  `figs_main_v2.yaml` and `felix_it` (092656) to `figs_attribution.yaml`;
  dropped/documented two dead early-attempt run dirs (`fedbuff_round` 000757,
  `felix_round` 015555 — both 0-to-few `agg_eval` events, no live process).
  `fwdllm_it_oracular` and `fedbuff_it_oracular` remain unlaunched in both
  manifests; `fedbuff_it_unaware` remains too early in the attribution manifest.
- **2026-07-24 (2)** — added the "OPEN WORK" tracking section (6 items: tex⇄doc
  alignment, palette-preview script, auto broken-axes, CDF P50/P90 annotations,
  perturbations-to-target bar plot, iteration-per-data-id trend investigation).
  Tracking only — no implementation yet, by request.
- **2026-07-24 (3)** — closed item 1 (tex⇄doc alignment): created the five
  missing `figs/code/eval/e2e/*.tex` fragments + four new `figs/code/eval/
  attr/*.tex` fragments with their own `fig:eval:attr:*` labels, wired into
  `evaluation.tex`, both pointing live at the render dirs; rebuilt
  `EXPERIMENTS.md` §10a's E1–E5/A0 rows to cite the actual manifests. Surfaced
  (not fixed) a real claims mismatch in the attribution prose — see item 1's
  updated block. Grounded items 5 and 6 (no implementation, per request):
  #5 confirmed the perturbation/forward-pass counters already support
  "sum to a timestamp," just need a target-crossing cutoff mode, and
  re-validated WS3-b live on `fedbuff_round`/`felix_round`; #6 found a real
  sample-based signal (FluxTune's `iteration_per_data_id` stays flat
  early→late while every round/`+IT` baseline roughly doubles) but needs a
  full-data pull and operator sign-off before becoming a figure. Items 2/3/4
  untouched this pass.
- **2026-07-24 (4)** — closed items 2–4. New `expt_scripts/preview_palette.py`
  (dummy-data line/scatter/bar palette preview, no telemetry I/O). New
  `plotlib/figures.py._detect_axis_break` + `_broken_y_axes`, wired into
  `fig_e1_acc_vs_time` (y-axis only; x-axis deferred, see item 3's block —
  caught and fixed a real point-count-vs-value-range bug in the gap-detector
  during testing). New `plotlib/figures.py._annotate_cdf_percentiles`, wired
  into `fig_e2_trainer_busy_cdf`/`fig_e5_session_cdf`. 18 new tests
  (`plotlib/test_figures.py`); full `-k fwdllm` suite green (32 passed). Both
  manifests re-rendered end-to-end on real telemetry with all three changes
  combined, no errors.
- **2026-07-24 (5)** — purged items #2–#4 from OPEN WORK (landed + committed,
  per this doc's own "delete once fully landed" policy); renumbered the
  remaining items (former #5→#2, #6→#3). No code change, doc cleanup only.
- **2026-07-25** — swept the 0230-batch long sim run-set (started 2026-07-24
  ~02:30, now fully stopped/complete) for newly-matured logs. `figs_main_v2.yaml`
  unchanged (its 4 run dirs are still each baseline's best/most-progressed —
  checked every later same-baseline dir in `experiments/`, none beat the
  incumbent). `figs_attribution.yaml` promoted `fedbuff_it_unaware` from the
  dead 16-eval `run_20260724_001229` attempt to the 0230-batch relaunch
  `run_20260724_144822` (95 evals @ peak-acc cutoff, 79.5% peak @ 2.99h) —
  Manifest 2 now 4/6 rendered, only the two `_oracular` legs remain unfilled.
  Re-rendered `paper_figs_attribution/` (all 7 basenames, clean). Deliberately
  did **not** touch the several other-baseline sim/real dirs launched
  post-17:30 that day (`fedbuff_round`/`felix_round`/`felix_it`/
  `fwdllm_it_unaware`/`fedbuff_it_oracular` re-attempts, all short ~1-2h spans)
  — those belong to the separate real-sim parity batch (`simulate_fwdllm.md`),
  a distinct task, not this run-set; see the note under Manifest 2.
