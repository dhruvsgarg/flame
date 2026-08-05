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
# (§D-63). `fedbuff_it_oracular` and `_unaware` differ in exactly one key,
# `trackTrainerAvail`, measured inert at syn_0 (§B.2: eligible_pool_reduction
# 0.0/0.0 — the oracle removes nobody). Their 3x floor gap (13.8% vs 4.9%) is
# which extremes landed under which name (§D-67), and unpooled `_unaware` has
# n=1: no floor at all. Only the FLOOR pools; the parity rows stay separate.
# **Phase 2 deletes this** — once `trackTrainerAvail` bites they are two configs.
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
)

_DIFF_CACHE: dict = {}


def code_differs(sha_a, sha_b) -> tuple:
    """(differs, why) — did any RUN-AFFECTING file change between two commits?

    Raw SHA inequality over-reports: a docs-only commit between two legs does not
    make them different runs, and this doc gets committed constantly. So diff the
    two trees and drop paths that cannot reach a run (§D-70).

    An unknown or unreachable SHA returns True: refusing to pool is the safe
    error. `git_info.clean=False` is NOT visible here — a same-SHA pair can still
    differ by uncommitted work, so this can only ever prove legs DIFFER.
    """
    if sha_a == sha_b:
        return False, "same commit"
    if not sha_a or not sha_b:
        return True, "a leg records no commit"
    key = tuple(sorted((sha_a, sha_b)))
    if key in _DIFF_CACHE:
        return _DIFF_CACHE[key]
    try:
        out = subprocess.run(
            ["git", "diff", "--name-only", key[0], key[1]],
            cwd=str(_HERE), capture_output=True, text=True, timeout=30)
        if out.returncode != 0:
            res = (True, "commit not in this repo")
        else:
            changed = [f for f in out.stdout.splitlines()
                       if f.strip() and not _RUN_IRRELEVANT.search(f)]
            res = ((True, f"{len(changed)} run-affecting file(s), e.g. "
                          f"{os.path.basename(changed[0])}") if changed
                   else (False, "docs/tests/floors only"))
    except (OSError, subprocess.SubprocessError):
        res = (True, "git unavailable")
    _DIFF_CACHE[key] = res
    return res


def largest_same_code(legs: list) -> tuple:
    """(kept, dropped, sha) — the biggest subset of `[(ts, path), ...]` that ran
    the same CODE, ties broken toward the NEWEST. Two SHAs count as the same code
    when nothing run-affecting changed between them (`code_differs`), so a
    docs-only commit between two legs does not split them. Legs with no snapshot
    group together under `None`, as they did before this existed."""
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
        if not code_differs(a, b)[0]:
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
    for key in sorted(groups, key=lambda k: (k[0], k[3] or 0, k[4])):
        baseline, trace, mode, maxrt, jvp_eval = key
        # Name the training config in the header: two groups of the same baseline
        # and duration now appear, and reading the wrong one inverts the verdict.
        cfg = f"  jvp_eval_mode={jvp_eval}"
        runs = groups[key]
        if len(runs) < 2 or (maxrt or 0) < args.min_duration:
            continue
        if args.duration is not None and (maxrt or 0) != args.duration:
            continue
        label = "/".join(x for x in (baseline, trace) if x)
        header = [f"\n=== {label}  mode={mode}  max_runtime_s={maxrt}{cfg}"]

        def hdr():
            """Print the group header once, whichever note reaches it first."""
            for line in header:
                print(line)
            header.clear()

        members = pool_members(baseline)
        if members != [baseline]:
            hdr()
            print(f"    POOLED — one config at {trace} (D-63): "
                  + " + ".join(members))
        # A leg that ran DIFFERENT code is not a replicate (§D-70). On by default:
        # a floor pooled across code versions measures the diff, not the pipeline.
        code_dropped, sha = [], None
        if not args.any_code:
            runs, code_dropped, sha = largest_same_code(runs)
            if code_dropped:
                hdr()
                print(f"    keeping the {len(runs)} leg(s) on {sha}; dropped "
                      f"{len(code_dropped)} on other code — pass --any-code to pool")
                for ts, path in code_dropped:
                    print(f"    {ts}  DROPPED — code {code_version(path)[0]}")
        kept, dropped, axis = drop_truncated(runs, args.span_tol)
        rows = [(ts, _seed(path), metrics(path), span, path)
                for ts, path, span in kept]
        rows = [r for r in rows if r[2]]
        # Report drops BEFORE the too-few-replicates bail, else a group that fell
        # below 2 from a truncated leg reads as "no replicates found", unexplained.
        if dropped:
            ref = max(s for _, _, s in kept if s)
            hdr()
            for ts, _path, span in dropped:
                print(f"    {ts}  DROPPED — achieved {axis} span {span:.0f}s is "
                      f">{args.span_tol:.0%} short of {ref:.0f}s "
                      f"(truncated run, not a replicate)")
        if len(rows) < 2:
            if dropped:
                print(f"    only {len(rows)} full-length leg(s) left — no floor for this group")
            continue
        seeds = {s for _, s, _, _, _ in rows}
        any_group = True
        hdr()
        print(f"    n_replicates={len(rows)}  seeds={sorted(seeds)}"
              + ("   ⚠ MIXED SEEDS — not a reproducibility floor" if len(seeds) > 1 else ""))
        for ts, seed, m, span, _p in rows:
            print(f"    {ts}  bins={m['committed_bins']:.0f}  cycles={m['cycles']:.0f}  "
                  f"iters/bin={m['iters_per_bin']:.2f}  var={m['mean_var']:.4f}"
                  + (f"  {axis}_span={span:.0f}s" if span else "  span=?"))
        graded = rung_floors([r[4] for r in rows]) if not args.run_level else {}
        print(f"    {'metric':18s} {'floor':>8s} {'run-lvl':>8s}   "
              f"{'rung':<22s} {'tol':>7s}  verdict")
        entry = {}
        for metric, (rung, field, tol, _gap) in _CALIBRATES.items():
            vals = [m[metric] for _, _, m, _, _ in rows if metric in m]
            if any(v != v for v in vals):     # NaN
                continue
            # The rung's own window where it has one; the run-level spread only
            # where the rung does not window (§D-53). Metrics the rung alone can
            # measure (an MA deviation, a slope, an accuracy gap) have no
            # run-level counterpart at all.
            run_level = _spread(vals) if vals else None
            floor = graded.get(metric, run_level)
            if floor is None:
                continue
            verdict = ("OK" if tol > floor * 1.5 else
                       "TIGHT — within 1.5x of the floor" if tol > floor else
                       "BELOW FLOOR — grades noise")
            fmt = _fmt_abs if metric in _ABSOLUTE else _fmt_rel
            print(f"    {metric:18s} {fmt(floor):>8s} {fmt(run_level):>8s}   "
                  f"{rung:<22s} {fmt(tol):>7s}  {verdict}")
            entry[metric] = {"floor_rel": round(floor, 4),
                             "run_level_rel": (None if run_level is None
                                               else round(run_level, 4)),
                             "windowed_by_rung": metric in graded,
                             "rung": rung, "tolerance_field": field,
                             "tolerance": tol, "verdict": verdict}
        report[label + f"@{maxrt}"] = {
            "mode": mode, "jvp_eval_mode": jvp_eval,
            "n_replicates": len(rows), "seeds": sorted(seeds),
            "achieved_span_s": [s for _, _, _, s, _ in rows],
            "achieved_span_axis": axis,
            "dropped_truncated": [t for t, _, _ in dropped], "metrics": entry}
        # Keep the longest ON group per baseline as that baseline's profile: the
        # floor is what the CURRENT training config reproduces to, and duration
        # changes it (§D-24), so a short or OFF group must never win.
        # Each mode writes its OWN side's keys and leaves the other's alone, so
        # `--mode real` then `--mode sim` builds the two-sided floor (§D-61).
        if args.profile_out and jvp_eval:
            pfx = "" if mode == "real" else "sim_"
            on_durations.setdefault(baseline, set()).add(maxrt or 0)
            prev = profiles.get(baseline)
            if prev is None or (maxrt or 0) >= prev["max_runtime_s"]:
                profiles[baseline] = {
                    "max_runtime_s": maxrt or 0, "jvp_eval_mode": True,
                    f"{pfx}n_replicates": len(rows),
                    f"{pfx}measured_at": _today(),
                    # Say so in the file: a reader finding six source runs under
                    # one baseline's name is owed the reason (D-63).
                    **({"pooled_from": members} if members != [baseline] else {}),
                    f"{pfx}source_runs": [t for t, _, _, _, _ in rows],
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
