#!/usr/bin/env python3
"""Measures the pipeline's own run-to-run REPRODUCIBILITY floor from same-mode,
same-seed, same-config replicate runs — the number every DIST parity tolerance
has to clear to be meaningful.

Why this exists: a parity rung compares real against sim and calls a gap a bug.
But the fwdllm cadence loop is a feedback system (variance gate -> iterations ->
model updates -> variance), so two runs of the SAME mode with the SAME seed do
not land on the same numbers either. Until that self-variance is measured, a
rung tolerance is a guess, and a rung tighter than the floor manufactures fails
nobody can ever fix (`v2_var_trajectory` shipped at 2% against a ~2% floor).
This is §D-5's "absolute, mode-independent sanity check", instantiated.

Reads only telemetry already on disk — no runs required. Compare its output
against the corresponding rung tolerance in
`async_cifar10/scripts/parity/checks.py`:

    metric              rung                 tolerance field   measured by
    throughput_rel      throughput           tol_rel           the rung
    time_to_n           terminal_state       time_tol          the rung
    trainers_at_n       terminal_state       trainers_tol      the rung
    iters/bin           v1_iter_per_data_id  mean_tol_rel      the rung
    mean var            v2_var_trajectory    mean_tol_rel      the rung
    cycles              cohort_sequence      count_tol         the rung

Where a rung windows on the matched logical budget, the floor is measured by
CALLING THAT RUNG over every pair of legs — the same thing a real↔real CONTROL
(`run_parity.py --control`) reports, and the only way to be sure the floor and
the tolerance it sizes grade the same window (§D-53). Asking the rung costs a
few minutes for all nine baselines; `--run-level` is the old, fast, wrong-window
estimator, kept for comparison.

Usage:
    python replicate_floor.py                          # all baselines, real legs
    python replicate_floor.py --baselines fedbuff_round --mode real
    python replicate_floor.py --min-duration 3000      # only compare like durations

Replicates must be comparable: runs are grouped by (baseline, trace, mode,
max_runtime_s) and only groups with >= 2 members are reported. Runs of different
lengths are NEVER pooled — `iters/bin` rises monotonically with run length
(9.82 @1800s -> 10.85 @3600s -> 12.39 @5400s on fedbuff_round real), so mixing
durations measures the training curve, not reproducibility.
"""
from __future__ import annotations

import argparse
import datetime
import glob
import itertools
from concurrent.futures import ProcessPoolExecutor
import json
import os
import re
import statistics as st
import subprocess
import sys
from pathlib import Path

import yaml

_HERE = Path(__file__).resolve().parent
_DEFAULT_EXPERIMENTS = _HERE.parent / "experiments"

# The floor is now measured by the rungs themselves (`_CALIBRATES`), so the
# checker must be importable — same two entries `run_parity.py` adds.
for _p in (str(_HERE.parents[2]),
           str(_HERE.parents[2] / "examples" / "async_cifar10" / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_PROFILE_HEADER = """\
# Measured same-seed replicate floor: the spread two config-identical real legs
# show with NO code change. The parity checker sizes its DIST tolerances from
# this (simulate_fwdllm.md D-24) -- a tolerance far above the floor passes real
# divergences, one below it grades noise. Regenerate with
# `replicate_floor.py --mode real --profile-out <dir>`; do not hand-edit.
"""


# Bump when a floor's MEANING changes (a new metric, a changed window, a rung
# fix that moves the number). Stamped into every profile so a stale file is
# visible rather than silently mixed with a fresh one.
_FLOOR_TOOL_VERSION = 2


def _today() -> str:
    return datetime.date.today().isoformat()

_RUN_RE = re.compile(
    r"^run_(?P<ts>\d{8}_\d{6})_(?P<baseline>.+)_n(?P<n>\d+)_smoke"
    r"(?:_(?P<trace>.+))?_(?P<variant>real|sim)$"
)

def _rung_gap(call, field: str, same_mode: bool = False):
    """The gap the RUNG ITSELF reports between two legs, or None if it can't read
    them. `call` names the checker function; imported lazily so the module stays
    importable without the checker.

    `same_mode` is the vclock/time family: those rungs BAIL when the B side has no
    `vclock_now`, which two real legs never do. The rung is uncontrollable; its
    quantity is not (§D-72)."""
    def gap(agg_a: dict, agg_b: dict):
        import parity.checks as C
        r = (getattr(C, call)(agg_a, agg_b, same_mode=True) if same_mode
             else getattr(C, call)(agg_a, agg_b))
        if field.startswith("count."):
            r = r.get("count") or {}
        if r.get("status") == "SKIP":
            return None
        name = field.split(".")[-1]
        # Take the value the rung's VERDICT uses. `v2` reports the pooled figure
        # under `mean_rel_diff` but decides on `matched_window_mean_rel_diff`,
        # and reading the pooled one understated its floor by up to 2.3x -- §D-53
        # a second time, on the tool that exists to prevent it. A rung that
        # switches field by baseline says so in `decided_on`; trust that first.
        v = (r.get(r["decided_on"]) if r.get("decided_on")
             else r.get(f"matched_window_{name}", r.get(name)))
        return None if v is None else abs(v)
    return gap


# Metric -> the parity rung it calibrates, that rung's tolerance field and
# nominal value, and how the RUNG measures the gap between two legs.
#
# A floor must be measured on the SAME window and axis the rung grades (§D-53).
# Most of these rungs truncate both sides to the matched logical budget, so a
# run-level mean over each leg's FULL run understates their floor -- most where
# the residual is largest. Asking the rung is the only way to be sure the two
# agree: there is nothing left to reimplement, and no window to get wrong.
#
# `iters_per_bin` has three consumers, because `cohort_sequence.count` and
# `v1b`'s cumulative mean ARE `v1`'s number rolled up (§D-22). Measured here
# once, under `v1`, and shared by the checker.
#
# The three `same_mode=True` entries are the vclock/time family: they bail on two
# real legs for want of `vclock_now`, so they went uncalibrated -- but the bail is
# the RUNG's, not the quantity's (§D-72), and real's own time-to-N reaches 6.7%
# same-code under an 8% gate. `throughput` replaces its `committed_bins` floor:
# that is work VOLUME, and the rung grades a time RATIO.
_CALIBRATES = {
    "throughput_rel": ("throughput", "tol_rel", 0.08,
                       _rung_gap("throughput_parity", "rel_diff",
                                 same_mode=True)),
    "time_to_n": ("terminal_state", "time_tol", 0.08,
                  _rung_gap("terminal_state_parity", "time_rel_diff",
                            same_mode=True)),
    "trainers_at_n": ("terminal_state", "trainers_tol", 0.05,
                      _rung_gap("terminal_state_parity", "trainers_rel_diff",
                                same_mode=True)),
    "iters_per_bin": ("v1_iter_per_data_id", "mean_tol_rel", 0.15,
                      _rung_gap("iters_per_data_id_parity", "mean_rel_diff")),
    "mean_var": ("v2_var_trajectory", "mean_tol_rel", 0.02,
                 _rung_gap("var_trajectory_parity", "mean_rel_diff")),
    "iters_ma_mean_dev": ("v1b_iters_moving_avg", "ma_mean_abs_tol", 0.25,
                          _rung_gap("iters_per_data_id_moving_avg_parity",
                                    "ma_mean_abs_dev")),
    "iters_ma_max_dev": ("v1b_iters_moving_avg", "ma_max_abs_tol", 0.75,
                         _rung_gap("iters_per_data_id_moving_avg_parity",
                                   "ma_max_abs_dev")),
    "iter_drift_lambda": ("v1c_iter_drift_rate", "lambda_floor_per_100", 0.05,
                          _rung_gap("iter_drift_rate_parity",
                                    "lambda_per_100_units")),
    "accuracy_diff": ("convergence", "acc_tol", 0.05,
                      _rung_gap("convergence_parity", "avg_accuracy_diff")),
    # K3's two bounds. `same_mode` is what makes them measurable at all: each
    # side reads its own clock, so a sim↔sim pair stops comparing wall to vclock
    # (§D-73) and a real↔real pair stops bailing (§D-72).
    "round_advance_rel": ("per_round_advance", "mean_tol_rel", 0.15,
                          _rung_gap("per_round_advance_parity", "mean_rel_diff",
                                    same_mode=True)),
    "round_advance_ks": ("per_round_advance", "ks_tol", 0.2,
                         _rung_gap("per_round_advance_parity", "ks_stat",
                                   same_mode=True)),
    "mean_chosen": ("selection_detail", "tol_chosen", 0.05,
                    _rung_gap("selection_detail_parity", "rel_diff_chosen")),
    # A pooled KS over the matched window, in KS units. Its flat 0.2 was never a
    # calibration -- and the max-over-N-trainers variant it used to report reaches
    # 0.30-0.89 between config-identical legs, which is what an uncorrected
    # extreme-value statistic looks like (§B.3).
    "utility_ks": ("utility", "max_ks", 0.2,
                   _rung_gap("utility_parity", "matched_window_pooled_ks_stat")),
    # The last two of the clock family. Measurable only since each side reads its
    # own clock (§D-73) — before that a same-mode pair reported sim's speedup.
    "overhead_rel": ("overhead_residual", "tol_rel", 0.10,
                     _rung_gap("overhead_residual", "matched_window_rel",
                               same_mode=True)),
    "overlap_rel": ("overlap_factor", "tol_rel", 0.10,
                    _rung_gap("overlap_factor", "rel_diff", same_mode=True)),
}


# Floors in the tolerance field's OWN units — iterations, accuracy, slope per
# 100 progress units. Printing them as percentages reads a 3.85-iteration MA
# deviation as "385%".
_ABSOLUTE = {"iters_ma_mean_dev", "iters_ma_max_dev", "iter_drift_lambda",
             "accuracy_diff"}


def _fmt_rel(v) -> str:
    return "—" if v is None else f"{v:.1%}"


def _fmt_abs(v) -> str:
    return "—" if v is None else f"{v:.3f}"


def _agg_events(run_dir: str) -> list:
    paths = glob.glob(os.path.join(run_dir, "telemetry", "aggregator_*.jsonl"))
    if not paths:
        return []
    out = []
    for line in open(paths[0], errors="replace"):
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if e.get("event") == "agg_round":
            out.append(e)
    return out


def _max_runtime_s(run_dir: str):
    cfg = os.path.join(run_dir, "aggregator_config.json")
    if not os.path.exists(cfg):
        return None
    try:
        h = json.load(open(cfg)).get("hyperparameters", {})
        return h.get("max_runtime_s") or h.get("maxRuntimeS")
    except (ValueError, OSError):
        return None


def _seed(run_dir: str):
    cfg = os.path.join(run_dir, "aggregator_config.json")
    if not os.path.exists(cfg):
        return None
    try:
        return json.load(open(cfg)).get("hyperparameters", {}).get("seed")
    except (ValueError, OSError):
        return None


def _span(vals: list):
    return (max(vals) - min(vals)) if len(vals) >= 2 else None


def _wall_span_s(run_dir: str):
    return _span([e["ts"] for e in _agg_events(run_dir)
                  if isinstance(e.get("ts"), (int, float))])


def _vclock_span_s(run_dir: str):
    """Virtual-clock span, or None on a leg that emits no `vclock_now` (real)."""
    return _span([e["vclock_now"] for e in _agg_events(run_dir)
                  if isinstance(e.get("vclock_now"), (int, float))])


def achieved_span_s(run_dir: str):
    """Span the aggregator ACTUALLY covered, on the leg's OWN clock.

    `max_runtime_s` is what the run was ASKED for; a run killed early still
    reports it, and pooling one reads its truncation as irreproducibility.

    A SIM leg's wall span measures the host, not the work — sim skips real waits,
    so identical work differs in wall time by whatever else the node was running.
    Read `vclock_now` where the leg has one, as every clock rung does (§D-73);
    real legs emit none and are unaffected.
    """
    v = _vclock_span_s(run_dir)
    return _wall_span_s(run_dir) if v is None else v


def leg_spans(legs: list) -> tuple:
    """`([(ts, path, span)], axis)` — every leg of a group on ONE axis.

    Mixing axes would compare a vclock span against a wall span and drop the
    wall-measured leg every time, so a group falls back to wall unless EVERY
    leg carries a vclock.
    """
    vspans = [(ts, path, _vclock_span_s(path)) for ts, path in legs]
    if vspans and all(s is not None for _, _, s in vspans):
        return vspans, "vclock"
    return [(ts, path, _wall_span_s(path)) for ts, path in legs], "wall"


def metrics(run_dir: str):
    """The four run-level quantities the DIST rungs grade, from one run."""
    ev = _agg_events(run_dir)
    if not ev:
        return None
    committed, iters, variances = set(), {}, []
    for e in ev:
        cid = e.get("cycle_data_id")
        if cid is None:
            continue
        key = (e.get("round") or 0, cid)
        if e.get("var_good_enough") is True:
            committed.add(key)
        it = e.get("iteration_per_data_id")
        if it is not None:
            iters[key] = max(iters.get(key, 0), it + 1)
        if e.get("var") is not None:
            variances.append(e["var"])
    if not committed or not iters:
        return None
    return {
        "committed_bins": float(len(committed)),
        "cycles": float(len(ev)),
        "iters_per_bin": st.mean(iters.values()),
        "mean_var": st.mean(variances) if variances else float("nan"),
    }


def _jvp_eval_mode(path: str) -> bool:
    """Was this leg trained with dropout off (`jvp_eval_mode`, H13)?

    Read from the trainer log because the knob is written to NO config file in
    the run dir -- it lives in the trainer's `config_overrides`, which the runner
    does not dump. Absent means the run predates the flag, i.e. the code default:
    dropout LIVE. Pooling an ON leg with an OFF one would measure the flag rather
    than the floor, which is exactly what §D-45 forbids.
    """
    for lg in glob.glob(os.path.join(path, "*trainers.log")):
        try:
            with open(lg, errors="ignore") as fh:
                for i, line in enumerate(fh):
                    if "jvp_eval_mode=" in line:
                        return "jvp_eval_mode=True" in line
                    if i > 50000:      # the knob logs at trainer init or never
                        break
        except OSError:
            continue
    return False


# Baselines that are ONE config at syn_0, so their legs pool into one floor
# (§D-63). Each `_oracular`/`_unaware` pair differs in exactly one key,
# `trackTrainerAvail`, measured inert at syn_0 (§B.2: eligible_pool_reduction
# 0.0/0.0 — the oracle removes nobody). Only the FLOOR pools; the parity rows
# stay separate. Pooling needs BOTH halves: the knob proven inert AND the two
# groups actually comparable — same code, same duration, same node (§B.1).
# **Phase 2 deletes this** — once `trackTrainerAvail` bites they are two configs.
#
# ⚠ **`fwdllm_it` is NO LONGER POOLED, and the reason is measured.** At n=3 per
# name per side the two names do not produce the same numbers: real reads bins 38
# / iters-per-bin 9.64-9.67 under `_unaware` against 40 / 9.20-9.22 under
# `_oracular`, and sim 39 / 9.88 against 42 / 9.21 — each name reproducing itself
# to 4 s.f. in BOTH modes. Pooling them therefore reports a SYSTEMATIC offset as
# replicate noise and takes a pinned baseline's floor from **0.0% to 5.1%**
# (§D-86). `fedbuff_it` keeps pooling because its legs INTERLEAVE (183/189/193
# against 186/188/192) — no offset, just its own 9.2% spread.
#
# ⚠ Cause NOT established, and the leading candidate is the KNOB. The nodes are
# identical hardware (operator ruling), which removes the host explanation, and
# the two resolved configs differ in exactly `trackTrainerAvail`
# (enabled False->True, type NONE->ORACULAR). Perfect 3/3 reproducibility on each
# side also rules out background load, which is not repeatable.
# The sim side cannot corroborate: each name carries its OWN charge profile and
# they differ materially (drain_tail 0.147s vs 0.125s), which moves sim cadence
# on its own (§D-50). Settled by ONE leg — §B.4.
_FLOOR_POOL_SYN0 = {"fedbuff_it_oracular": "fedbuff_it",
                    "fedbuff_it_unaware": "fedbuff_it"}


def pool_name(baseline: str, trace: str, enabled: bool = True) -> str:
    """The replicate GROUP `baseline` belongs to — itself, unless it is one of a
    set of names proven to be one config at this trace."""
    if not enabled or trace != "syn_0":
        return baseline
    return _FLOOR_POOL_SYN0.get(baseline, baseline)


def pool_members(name: str) -> list:
    """The baseline names a pooled group's floor is written out to. The checker
    looks its floor up by the baseline it is GRADING, so a pooled floor has to
    land in every member's own file."""
    return sorted(b for b, g in _FLOOR_POOL_SYN0.items() if g == name) or [name]


def _requested(baselines, pool: bool):
    """The raw baseline names `--baselines` accepts, or None for all.

    Naming EITHER member of a pooled group selects the whole group, as does
    naming the group. A floor asked for by one name and answered from half its
    legs is the n=1 problem pooling exists to fix.
    """
    if not baselines:
        return None
    want = set(baselines)
    if pool:
        for b in list(want):
            want.update(pool_members(_FLOOR_POOL_SYN0.get(b, b)))
    return want


def discover(experiments_dir: str, baselines, mode: str, pool: bool = True) -> dict:
    """{(baseline, trace, mode, max_runtime_s, jvp_eval_mode): [(ts, path), ...]}

    `baseline` is the POOLED group name where one applies (`pool_name`). Naming
    either member selects the whole group: a floor asked for by one name and
    answered from half its legs is the n=1 problem this exists to fix.
    """
    groups: dict = {}
    wanted = _requested(baselines, pool)
    for path in sorted(glob.glob(os.path.join(experiments_dir, "run_*"))):
        m = _RUN_RE.match(os.path.basename(path))
        if not m or m["variant"] != mode:
            continue
        if wanted is not None and m["baseline"] not in wanted:
            continue
        trace = m["trace"] or ""
        key = (pool_name(m["baseline"], trace, pool), trace, mode,
               _max_runtime_s(path), _jvp_eval_mode(path))
        groups.setdefault(key, []).append((m["ts"], path))
    return groups


def hostname_of(run_dir: str) -> str | None:
    """The node a leg ran on, from `snapshot.yaml`.

    Recorded in the floor profile because comparability is per-NODE as well as
    per-code: pooling two config-identical groups that ran on different hosts
    reported a systematic offset as replicate noise and inflated a pinned
    baseline's floor from 0.0% to 5.1% (§D-86). A future re-calibration can now
    see, from the file alone, whether its legs are comparable to these."""
    try:
        d = yaml.safe_load(open(os.path.join(run_dir, "snapshot.yaml"),
                                encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return None
    return d.get("hostname")


def code_version(run_dir: str) -> tuple:
    """`(git_sha9, clean)` the run recorded for ITSELF, from `snapshot.yaml`.

    Two legs are replicates only if they ran the SAME code (§D-70). Nothing read
    this before, and the cost was silent: `fwdllm`'s two sim legs sit on either
    side of a charge re-profile and pooled to an 18% "floor" on a baseline whose
    real floor is 0.0%. `(None, None)` when a run predates the snapshot.

    `clean=False` means uncommitted changes at launch, so the SHA is necessary
    but not sufficient — it can only ever prove two legs DIFFER.
    """
    path = os.path.join(run_dir, "snapshot.yaml")
    if not os.path.exists(path):
        return None, None
    try:
        g = (yaml.safe_load(open(path, encoding="utf-8")) or {}).get("git_info") or {}
    except (OSError, yaml.YAMLError):
        return None, None
    sha = g.get("commit")
    return (sha[:9] if sha else None), g.get("clean")


# Paths that CANNOT reach a run: grading artifacts, docs, tests, plotting. A
# commit touching only these does not make two legs different runs. Deny-list,
# not an allow-list -- an unrecognised path counts as run-affecting, so the
# error is toward refusing to pool rather than pooling two different systems.
# `LAUNCHER_INVOKES` is the ONLY reason an `expt_scripts` python file can reach a
# run; everything else there is analysis run after the fact. `test_replicate_floor`
# re-derives this list from `run_sequential.sh` so it cannot drift silently.
LAUNCHER_INVOKES = ("profile_sim_charges", "extract_sanity_checks")
# A charge profile reaches a SIM run and nothing else -- real never reads one
# (§F-1). So it is run-affecting for sim legs and inert for real ones, and
# splitting a real-side group on a re-profile grades a file that leg never
# opened. Mode-dependent, hence separate from the flat list below.
_SIM_ONLY_INPUT = re.compile(r"(/sim_charge_profiles/)")
_RUN_IRRELEVANT = re.compile(
    r"(\.md$)"
    r"|(/parity_floors/)"
    r"|(/_parity_reports/)"
    r"|(/experiments/)"
    r"|((^|/)test_[^/]*\.py$)"
    r"|(\.off-bak$)|(\.bak$)"
    r"|(/plotlib/)"
    r"|((^|/)(plot_|preview_|make_paper_figs))"
    r"|(/expt_scripts/(?!" + "|".join(LAUNCHER_INVOKES) + r")[^/]*\.py$)"
    # `run_block.sh` only SEQUENCES `run_sequential.sh` and the analysis tools;
    # every parameter it can vary (baseline, mode, max_runtime_s) is already in
    # the grouping key, so editing it cannot make two legs different runs.
    # `run_sequential.sh` launches, and stays run-affecting.
    r"|(/expt_scripts/run_block\.sh$)"
)

_DIFF_CACHE: dict = {}
_CHARGE_CACHE: dict = {}


def _run_affecting(sha_a, sha_b, mode: str | None) -> tuple:
    """(paths, inert_summary) — run-affecting files changed between two commits.
    `paths` is None when the diff is undecidable (unknown SHA, no git)."""
    try:
        out = subprocess.run(
            ["git", "diff", "--name-only", sha_a, sha_b],
            cwd=str(_HERE), capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None, "git unavailable"
    if out.returncode != 0:
        return None, "commit not in this repo"
    inert = ["docs/tests/floors"]
    changed = [f for f in out.stdout.splitlines()
               if f.strip() and not _RUN_IRRELEVANT.search(f)]
    if mode == "real":
        keep = [f for f in changed if not _SIM_ONLY_INPUT.search(f)]
        if len(keep) < len(changed):
            inert.append("sim charges (real never reads them)")
        changed = keep
    return changed, " + ".join(inert)


def code_differs(sha_a, sha_b, mode: str | None = None) -> tuple:
    """(differs, why) — did any RUN-AFFECTING file change between two commits?

    Raw SHA inequality over-reports: a docs-only commit between two legs does not
    make them different runs, and this doc gets committed constantly. So diff the
    two trees and drop paths that cannot reach a run (§D-70).

    `mode="real"` additionally drops `sim_charge_profiles/`, which only a sim leg
    reads. Without it, committing a re-profile splits every REAL group across it
    — legs that are byte-identical in every input they actually consumed.

    An unknown or unreachable SHA returns True: refusing to pool is the safe
    error. `git_info.clean=False` is NOT visible here — a same-SHA pair can still
    differ by uncommitted work, so this can only ever prove legs DIFFER, and on
    the charge dimension `largest_same_code` settles it from telemetry instead.
    """
    if sha_a == sha_b:
        return False, "same commit"
    if not sha_a or not sha_b:
        return True, "a leg records no commit"
    key = tuple(sorted((sha_a, sha_b))) + (mode,)
    if key in _DIFF_CACHE:
        return _DIFF_CACHE[key]
    changed, why = _run_affecting(key[0], key[1], mode)
    if changed is None:
        res = (True, why)
    elif changed:
        res = (True, f"{len(changed)} run-affecting file(s), e.g. "
                     f"{os.path.basename(changed[0])}")
    else:
        res = (False, why + " only")
    _DIFF_CACHE[key] = res
    return res


def charge_profile_only(sha_a, sha_b) -> bool:
    """True when the ONLY run-affecting change between two commits is a sim charge
    re-profile — the one difference a leg's own telemetry can settle. Covers every
    baseline's profile, deliberately: `consumed_charges` then decides on what the
    legs actually read, which is baseline-scoped by construction."""
    changed, _ = _run_affecting(sha_a, sha_b, "sim")
    return bool(changed) and all(_SIM_ONLY_INPUT.search(f) for f in changed)


def consumed_charges(run_dir: str) -> dict | None:
    """{(label, payload_kind): (charged_s, source)} the leg ACTUALLY read, from
    its own `vclock_charge` telemetry rather than the profile committed with it.

    A leg launched dirty charges values its SHA does not name: three
    `fedbuff_round` sim legs recorded `bdbde72b7` yet charged the profile
    committed one commit later, so a SHA check split three true replicates
    (§D-91). Keyed on payload kind too, since a profile prices one label per kind
    (`redispatch_turnaround` is ON for `weights`, OFF for `var_bad`) and the
    label alone takes whichever fired last.

    `None` when the run emitted no `vclock_charge` at all — the SHA verdict then
    stands, since refusing to pool is the safe error (§D-70) and an empty dict
    would pool two silent legs on the absence of evidence.
    """
    if run_dir in _CHARGE_CACHE:
        return _CHARGE_CACHE[run_dir]
    out: dict = {}
    for path in sorted(glob.glob(os.path.join(run_dir, "telemetry",
                                              "aggregator_*.jsonl"))):
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if '"vclock_charge"' not in line:
                        continue
                    try:
                        e = json.loads(line)
                    except ValueError:
                        continue
                    if e.get("event") == "vclock_charge":
                        out[(e.get("label"), e.get("payload_kind"))] = (
                            e.get("charged_s"), e.get("charge_source"))
        except OSError:
            out = {}
            break
    res = out or None
    _CHARGE_CACHE[run_dir] = res
    return res


def _baseline_of(path: str):
    m = _RUN_RE.match(os.path.basename(path))
    return m["baseline"] if m else None


def _charges_agree(legs_a: list, legs_b: list) -> bool:
    """Did the two SHA clusters charge the same table, compared PER BASELINE?

    A POOLED group (§D-63) holds two baselines with two different profiles, so a
    single representative leg from each cluster compares `fedbuff_it_oracular`'s
    charges against `fedbuff_it_unaware`'s -- always unequal, always splitting.
    That dropped three `_oracular` legs and left the pooled floor measured on
    3 `_unaware` + 1 `_oracular`, which is the name-mixing §D-86 forbids.

    Compare only baselines present in BOTH clusters, and require at least one:
    with no shared baseline there is no evidence, and refusing to merge is the
    safe error (§D-70).
    """
    def by_base(legs):
        out: dict = {}
        for _ts, path in legs:
            out.setdefault(_baseline_of(path), []).append(path)
        return out

    a, b = by_base(legs_a), by_base(legs_b)
    shared = set(a) & set(b) - {None}
    if not shared:
        return False
    for base in shared:
        ca, cb = consumed_charges(a[base][0]), consumed_charges(b[base][0])
        if ca is None or cb is None or ca != cb:
            return False
    return True


def largest_same_code(legs: list, mode: str | None = None) -> tuple:
    """(kept, dropped, sha) — the biggest subset of `[(ts, path), ...]` that ran
    the same CODE, ties broken toward the NEWEST. Two SHAs count as the same code
    when nothing run-affecting changed between them (`code_differs`), so a
    docs-only commit between two legs does not split them. `mode` narrows what
    counts for that side. Legs with no snapshot group together under `None`, as
    they did before this existed."""
    by: dict = {}
    for ts, path in legs:
        by.setdefault(code_version(path)[0], []).append((ts, path))
    if len(by) <= 1:
        return legs, [], next(iter(by), None)
    # Merge SHAs that differ only by run-irrelevant commits. Diffs compose, so
    # "no run-affecting change between them" is transitive and this is a union.
    shas = sorted(by, key=lambda s: (s is None, s or ""))
    parent = {s: s for s in shas}

    def find(s):
        while parent[s] != s:
            parent[s] = parent[parent[s]]
            s = parent[s]
        return s

    for a, b in itertools.combinations(shas, 2):
        differs = code_differs(a, b, mode)[0]
        # A charge re-profile is the one run-affecting difference a leg can
        # DISPROVE from its own telemetry, so settle it on what was charged
        # rather than on what was committed (§D-70, `consumed_charges`).
        if differs and mode == "sim" and a and b and charge_profile_only(a, b):
            differs = not _charges_agree(by[a], by[b])
        if not differs:
            parent[find(a)] = find(b)
    clusters: dict = {}
    for s in shas:
        clusters.setdefault(find(s), []).extend(by[s])
    if len(clusters) <= 1:
        return legs, [], shas[0]
    root = max(clusters, key=lambda s: (len(clusters[s]),
                                        max(t for t, _ in clusters[s])))
    return (clusters[root],
            [x for s, v in clusters.items() if s != root for x in v], root)


def largest_same_node(legs: list) -> tuple:
    """(kept, dropped, host) — the biggest subset of `[(ts, path), ...]` that ran
    on ONE node, ties broken toward the NEWEST.

    DIAGNOSTIC ONLY — nothing calls this to filter. Operator ruling: shepherd,
    jayne, wash and kaylee are identical hardware, so legs pool across them and a
    cross-node split is NOT an explanation for a systematic offset. Kept because
    the question "is this difference the host?" recurs, and answering it from the
    profile's `nodes:` field beats re-deriving it from run dirs.

    Legs with no recorded hostname group under `None`.
    """
    by: dict = {}
    for ts, path in legs:
        by.setdefault(hostname_of(path), []).append((ts, path))
    if len(by) <= 1:
        return legs, [], next(iter(by), None)
    root = max(by, key=lambda h: (len(by[h]), max(t for t, _ in by[h])))
    return (by[root],
            [x for h, v in by.items() if h != root for x in v], root)


def checker_agg(run_dir: str) -> dict:
    """The run's aggregator telemetry as the CHECKER parses it.

    Not `_agg_events`: that keeps the eval-tagged `agg_round` events the checker
    deliberately routes to `eval_commits`, so a floor built on it would grade a
    different event set than the rung it calibrates. The agg half of
    `load_run_dir`, without the trainer telemetry no floor metric reads.
    """
    from parity.checks import load_agg_jsonl
    out: dict = {"agg_rounds": [], "selection_train": [], "agg_evals": []}
    for path in sorted(glob.glob(os.path.join(run_dir, "telemetry",
                                              "aggregator_*.jsonl"))):
        d = load_agg_jsonl(path)
        for k in out:
            out[k].extend(d[k])
    out["agg_rounds"].sort(key=lambda x: (x["round"], x.get("agg_goal_count", 0),
                                          x["ts"]))
    out["selection_train"].sort(key=lambda x: (x["round"], x["ts"]))
    out["agg_evals"].sort(key=lambda x: x["round"])
    return out


def rung_floors(paths: list, cache: dict | None = None) -> dict:
    """{metric: floor} measured by each rung across EVERY pair of `paths`.

    The floor is a max-pairwise spread (§D-57), so it is the largest gap any two
    config-identical legs show on the rung's own window — which is exactly what a
    real↔real CONTROL pair reports (`run_parity.py --control`). Metrics whose
    rung does not window are absent here and stay run-level.
    """
    cache = {} if cache is None else cache
    for p in paths:
        cache.setdefault(p, checker_agg(p))
    out: dict = {}
    for metric, (_rung, _field, _tol, gap) in _CALIBRATES.items():
        if gap is None:
            continue
        vals = [g for a, b in itertools.combinations(paths, 2)
                if (g := gap(cache[a], cache[b])) is not None]
        if vals:
            out[metric] = max(vals)
    return out


def drop_truncated(legs: list, span_tol: float = 0.05) -> tuple:
    """Split `[(ts, path), ...]` into (kept, dropped, axis) on the ACHIEVED span (§D-44).

    A leg killed early is a shorter run wearing the same `max_runtime_s`, not a
    replicate. Items come back as `(ts, path, span)`. Shared with the real↔real
    control in `run_parity.py`, which must group legs by the same rule.
    """
    legs, axis = leg_spans(legs)
    spans = [s for _, _, s in legs if s]
    if not spans:
        return legs, [], axis
    ref = max(spans)
    kept = [x for x in legs if x[2] is None or x[2] >= ref * (1.0 - span_tol)]
    dropped = [x for x in legs if x not in kept]
    return kept, dropped, axis


def _spread(vals: list) -> float:
    """Max pairwise relative spread — the floor a 2-sided tolerance must clear."""
    lo, hi = min(vals), max(vals)
    return (hi - lo) / hi if hi else 0.0


def _grade_group(job: tuple) -> dict:
    """One group's whole floor computation, as a PROCESS-POOL job.

    Returns printable lines plus the report/profile payloads rather than printing:
    groups are graded concurrently, so output must be re-ordered by the parent or
    a nine-baseline sweep interleaves into nonsense.

    Group-level is the right grain. Each leg's aggregator log is ~1.5 GB and the
    parsed form ~0.7 GB, so a worker must parse ITS OWN legs and hand back only
    the small floor dict — shipping parsed telemetry between processes costs more
    than the parse it saves.
    """
    key, runs, opts = job
    baseline, trace, mode, maxrt, jvp_eval = key
    out: dict = {"key": key, "lines": [], "report": None, "profile": None,
                 "any_group": False}
    lines = out["lines"]
    label = "/".join(x for x in (baseline, trace) if x)
    header = [f"\n=== {label}  mode={mode}  max_runtime_s={maxrt}"
              f"  jvp_eval_mode={jvp_eval}"]

    def hdr():
        lines.extend(header)
        header.clear()

    def emit(line):
        lines.append(line)

    members = pool_members(baseline)
    if members != [baseline]:
        hdr()
        emit(f"    POOLED — one config at {trace} (D-63): " + " + ".join(members))
    code_dropped, sha = [], None
    if not opts["any_code"]:
        runs, code_dropped, sha = largest_same_code(runs, mode)
        if code_dropped:
            hdr()
            emit(f"    keeping the {len(runs)} leg(s) on {sha}; dropped "
                 f"{len(code_dropped)} on other code — pass --any-code to pool")
            for ts, path in code_dropped:
                emit(f"    {ts}  DROPPED — code {code_version(path)[0]}")
    kept, dropped, axis = drop_truncated(runs, opts["span_tol"])
    rows = [(ts, _seed(path), metrics(path), span, path)
            for ts, path, span in kept]
    rows = [r for r in rows if r[2]]
    if dropped:
        ref = max(s for _, _, s in kept if s)
        hdr()
        for ts, _path, span in dropped:
            emit(f"    {ts}  DROPPED — achieved {axis} span {span:.0f}s is "
                 f">{opts['span_tol']:.0%} short of {ref:.0f}s "
                 f"(truncated run, not a replicate)")
    if len(rows) < 2:
        if dropped:
            emit(f"    only {len(rows)} full-length leg(s) left — no floor for this group")
        return out
    seeds = {s for _, s, _, _, _ in rows}
    out["any_group"] = True
    hdr()
    emit(f"    n_replicates={len(rows)}  seeds={sorted(seeds)}"
         + ("   ⚠ MIXED SEEDS — not a reproducibility floor" if len(seeds) > 1 else ""))
    for ts, seed, m, span, _p in rows:
        emit(f"    {ts}  bins={m['committed_bins']:.0f}  cycles={m['cycles']:.0f}  "
             f"iters/bin={m['iters_per_bin']:.2f}  var={m['mean_var']:.4f}"
             + (f"  {axis}_span={span:.0f}s" if span else "  span=?"))
    graded = rung_floors([r[4] for r in rows]) if not opts["run_level"] else {}
    emit(f"    {'metric':18s} {'floor':>8s} {'run-lvl':>8s}   "
         f"{'rung':<22s} {'tol':>7s}  verdict")
    entry = {}
    for metric, (rung, field, tol, _gap) in _CALIBRATES.items():
        vals = [m[metric] for _, _, m, _, _ in rows if metric in m]
        if any(v != v for v in vals):
            continue
        run_level = _spread(vals) if vals else None
        floor = graded.get(metric, run_level)
        if floor is None:
            continue
        verdict = ("OK" if tol > floor * 1.5 else
                   "TIGHT — within 1.5x of the floor" if tol > floor else
                   "BELOW FLOOR — grades noise")
        fmt = _fmt_abs if metric in _ABSOLUTE else _fmt_rel
        emit(f"    {metric:18s} {fmt(floor):>8s} {fmt(run_level):>8s}   "
             f"{rung:<22s} {fmt(tol):>7s}  {verdict}")
        entry[metric] = {"floor_rel": round(floor, 4),
                         "run_level_rel": (None if run_level is None
                                           else round(run_level, 4)),
                         "windowed_by_rung": metric in graded,
                         "rung": rung, "tolerance_field": field,
                         "tolerance": tol, "verdict": verdict}
    out["report"] = (label + f"@{maxrt}", {
        "mode": mode, "jvp_eval_mode": jvp_eval,
        "n_replicates": len(rows), "seeds": sorted(seeds),
        "achieved_span_s": [s for _, _, _, s, _ in rows],
        "achieved_span_axis": axis,
        "dropped_truncated": [t for t, _, _ in dropped], "metrics": entry})
    if opts["profile_out"] and jvp_eval:
        paths = [r[4] for r in rows]
        out["profile"] = {
            "baseline": baseline, "maxrt": maxrt or 0, "mode": mode,
            "members": members, "n_rows": len(rows),
            "source_runs": [t for t, _, _, _, _ in rows], "entry": entry,
            "trace": trace, "span_axis": axis,
            "code": sha or (code_version(paths[0])[0] if paths else None),
            "nodes": sorted({h for p in paths if (h := hostname_of(p))}),
        }
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiments-dir", default=str(_DEFAULT_EXPERIMENTS))
    ap.add_argument("--baselines", nargs="*", default=None)
    ap.add_argument("--mode", choices=("real", "sim"), default="real",
                    help="which side's replicates to pool (default real)")
    ap.add_argument("--min-duration", type=float, default=0.0,
                    help="skip groups whose max_runtime_s is below this")
    ap.add_argument("--duration", type=float, default=None, metavar="S",
                    help="grade ONLY groups at this max_runtime_s. Use it when a "
                         "baseline has ON groups at two durations: the floor must "
                         "come from the run length the rung grades (D-24/D-53)")
    ap.add_argument("--span-tol", type=float, default=0.05,
                    help="drop a leg whose ACHIEVED span is this far below the "
                         "group's longest (default 0.05 = 5%%); a truncated run "
                         "is a shorter run, not a replicate")
    ap.add_argument("--any-code", action="store_true",
                    help="pool replicate legs that ran DIFFERENT code. Off by "
                         "default: a floor measured across code versions grades "
                         "the diff, not the pipeline (D-70)")
    ap.add_argument("--no-pool", action="store_true",
                    help="grade each baseline NAME separately instead of pooling "
                         "the ones proven to be one config at syn_0 (D-63). Use it "
                         "to re-check the pooling claim, not to grade")
    ap.add_argument("--run-level", action="store_true",
                    help="measure every floor as a run-level spread, the way this "
                         "tool did before it asked the rungs. Understates any "
                         "windowed rung's floor (D-53) — for comparison only")
    ap.add_argument("--jobs", type=int, default=None, metavar="N",
                    help="grade this many baseline groups concurrently "
                         "(default: one per CPU). Each leg is a ~1.5 GB parse, so "
                         "a serial nine-baseline sweep is ~10 min and a parallel "
                         "one is bounded by the slowest single group")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--profile-out", default=None, metavar="DIR",
                    help="write per-baseline floor profiles the parity checker "
                         "reads to size its DIST tolerances (D-24). Only the "
                         "longest ON group per baseline is written.")
    args = ap.parse_args(argv)

    groups = discover(args.experiments_dir, args.baselines, args.mode,
                      pool=not args.no_pool)
    report, profiles, any_group = {}, {}, False
    on_durations: dict = {}     # baseline -> ON durations seen, for the ambiguity warning
    jobs = [(key, groups[key],
             {"any_code": args.any_code, "span_tol": args.span_tol,
              "run_level": args.run_level, "profile_out": bool(args.profile_out)})
            for key in sorted(groups, key=lambda k: (k[0], k[3] or 0, k[4]))
            if len(groups[key]) >= 2
            and (key[3] or 0) >= args.min_duration
            and (args.duration is None or (key[3] or 0) == args.duration)]

    # Groups are independent and each is minutes of parsing, so grade them
    # concurrently and re-order the output. One worker per group up to --jobs.
    n_workers = max(1, min(args.jobs or os.cpu_count() or 1, len(jobs) or 1))
    if n_workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            results = list(pool.map(_grade_group, jobs))
    else:
        results = [_grade_group(j) for j in jobs]

    for res in results:
        for line in res["lines"]:
            print(line)
        any_group = any_group or res["any_group"]
        if res["report"]:
            report[res["report"][0]] = res["report"][1]
        if res["profile"]:
            pr = res["profile"]
            baseline, maxrt, mode = pr["baseline"], pr["maxrt"], pr["mode"]
            members, entry = pr["members"], pr["entry"]
            pfx = "" if mode == "real" else "sim_"
            on_durations.setdefault(baseline, set()).add(maxrt)
            prev = profiles.get(baseline)
            if prev is None or maxrt >= prev["max_runtime_s"]:
                profiles[baseline] = {
                    "max_runtime_s": maxrt, "jvp_eval_mode": True,
                    "trace": pr["trace"],
                    f"{pfx}n_replicates": pr["n_rows"],
                    f"{pfx}measured_at": _today(),
                    # PROVENANCE — what makes a future re-calibration comparable
                    # to this one (§D-70, §D-86). Without these a floor file says
                    # what the spread was but not what it was the spread OF.
                    f"{pfx}code_commit": pr["code"],
                    f"{pfx}nodes": pr["nodes"],
                    f"{pfx}span_axis": pr["span_axis"],
                    f"{pfx}floor_tool_version": _FLOOR_TOOL_VERSION,
                    **({"pooled_from": members} if members != [baseline] else {}),
                    f"{pfx}source_runs": pr["source_runs"],
                    f"{pfx}metrics": {k: v["floor_rel"] for k, v in entry.items()},
                    "rungs": {k: {"rung": v["rung"], "field": v["tolerance_field"],
                                  "nominal": v["tolerance"]}
                              for k, v in entry.items()},
                }

    if not any_group:
        print("No replicate groups found (need >= 2 runs sharing "
              "baseline/trace/mode/max_runtime_s).")
        return 1
    print("\nA tolerance at or below its floor cannot be closed by any code change — "
          "raise it to the floor or grade the metric differently.")
    if args.json_out:
        json.dump(report, open(args.json_out, "w"), indent=2)
        print(f"json: {args.json_out}")
    if args.profile_out:
        os.makedirs(args.profile_out, exist_ok=True)
        for baseline, prof in sorted(profiles.items()):
            others = sorted(on_durations.get(baseline, set()) - {prof["max_runtime_s"]})
            if others and args.duration is None:
                print(f"⚠ {baseline}: ON groups at {others + [prof['max_runtime_s']]}s; "
                      f"wrote the {prof['max_runtime_s']:.0f}s one. A floor must come "
                      f"from the run length the rung grades — pass --duration.")
            # A pooled group lands in EVERY member's file: the checker looks a
            # floor up by the baseline it is grading, and the rows stay separate.
            for member in pool_members(baseline):
                path = os.path.join(args.profile_out, f"{member}.yaml")
                # MERGE: one invocation measures one mode, so clobbering would
                # drop the other side's floor and silently re-narrow the gate.
                on_disk = {}
                if os.path.exists(path):
                    on_disk = yaml.safe_load(open(path).read()) or {}
                # Merge, so `--mode real` then `--mode sim` compose into a
                # two-sided floor (§D-78). But keys that describe the CURRENT
                # grouping must be dropped when they no longer apply, or an
                # un-pooled baseline keeps advertising a `pooled_from` it no
                # longer has — the file would then misdescribe its own legs.
                if "pooled_from" not in prof:
                    on_disk.pop("pooled_from", None)
                on_disk.update(prof)
                with open(path, "w") as fh:
                    fh.write(_PROFILE_HEADER)
                    yaml.safe_dump(on_disk, fh, sort_keys=True)
                print(f"floor profile: {path}")
        if not profiles:
            print(f"no ON {args.mode} groups -- no floor profile written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
