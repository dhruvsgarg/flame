#!/usr/bin/env python3
"""Repeatable real<->sim parity checker for the fwdllm baselines.

Discovers the most-recent real/sim run pair per baseline, shows them for
confirmation, runs the shared parity battery, and prints one compact cross-
baseline summary.

Removes two footguns vs calling scripts.parity.cli by hand:
  - baseline is parsed as an exact token from the run-dir name
    (`run_<ts>_<baseline>_n<N>_smoke[_<trace>]_<real|sim>`), so the `*fwdllm*`
    glob never captures `fwdllm_plus`.
  - always takes the latest pair and reads agg_goal from each run's own config,
    so no hand-passed --agg-goal drifts.

Usage:
    python run_parity.py                       # all 3 baselines, latest pairs, confirm
    python run_parity.py --baselines fluxtune  # one baseline
    python run_parity.py --yes                 # skip the confirm prompt
    python run_parity.py --validate            # + live-run checks (staleness/vclock_now)
    python run_parity.py --jobs 1              # serial (default: one worker per pair)

Pairs are graded in PARALLEL: each reads its own two run dirs and writes its own
report, with no shared state. A pair costs ~17s, 77% of it file loading, so the
full 10-baseline sweep goes from ~2.9 min to ~20s.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_LIB_PYTHON = _HERE.parents[2]                       # lib/python
_PARITY_SCRIPTS = _LIB_PYTHON / "examples" / "async_cifar10" / "scripts"
_DEFAULT_EXPERIMENTS = _HERE.parent / "experiments"  # examples/fwdllm/experiments

for _p in (str(_LIB_PYTHON), str(_PARITY_SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# `run_<YYYYMMDD>_<HHMMSS>_<baseline>_n<N>_smoke[_<trace>]_<real|sim>`
_RUN_RE = re.compile(
    r"^run_(?P<ts>\d{8}_\d{6})_(?P<baseline>.+)_n(?P<n>\d+)_smoke"
    r"(?:_(?P<trace>.+))?_(?P<variant>real|sim)$"
)


def _discover(experiments_dir: str) -> dict:
    """{(baseline, trace): {'real': (ts, path), 'sim': (ts, path)}} keeping the
    latest ts per (baseline, trace, variant)."""
    out: dict = {}
    for path in glob.glob(os.path.join(experiments_dir, "run_*")):
        m = _RUN_RE.match(os.path.basename(path))
        if not m:
            continue
        key = (m["baseline"], m["trace"] or "")
        slot = out.setdefault(key, {})
        prev = slot.get(m["variant"])
        if prev is None or m["ts"] > prev[0]:   # ts is fixed-width -> lexical == chronological
            slot[m["variant"]] = (m["ts"], path)
    return out


def _agg_goal(run_dir: str) -> int | None:
    cfg = os.path.join(run_dir, "aggregator_config.json")
    if not os.path.exists(cfg):
        return None
    try:
        h = json.load(open(cfg)).get("hyperparameters", {})
        g = h.get("aggGoal", h.get("agg_goal"))
        return int(g) if g is not None else None
    except (ValueError, OSError, TypeError):
        return None


# rungs shown in the summary (name -> short label), in ladder-ish order
_HEADLINE = [
    ("matched_budget_coverage", "budget"),
    ("cohort_sequence", "cohort"), ("vclock_telemetry", "vclock"),
    ("throughput", "thru"), ("total_commits", "commits"),
    ("terminal_state", "terminal"), ("r1_inflight_overlap", "R1"),
    ("v1_iter_per_data_id", "V1"), ("v2_var_trajectory", "V2"),
    ("staleness", "U3"), ("participation", "S2"),
    ("convergence", "conv"), ("convergence_loss", "conv_loss"),
]


def _status(res: dict) -> str:
    if not isinstance(res, dict):
        return "–"
    if res.get("status") == "SKIP":
        return "–"
    if res.get("ok"):
        return "✓"
    return "~" if res.get("tier") == "DIAG" else "✗"


def _live_checks(baseline: str, real_dir: str, sim_dir: str) -> list[str]:
    """--validate extras: confirm the config/telemetry a fresh run must carry."""
    notes = []
    for tag, d in (("real", real_dir), ("sim", sim_dir)):
        logs = glob.glob(os.path.join(d, "*aggregator*.log"))
        pol = "?"
        if logs:
            for line in open(logs[0], errors="replace"):
                if "staleness_policy =" in line:
                    pol = line.split("staleness_policy =")[1].strip()
        notes.append(f"{tag} staleness_policy={pol}")
    # vclock_now present in sim agg_round telemetry?
    tel = glob.glob(os.path.join(sim_dir, "telemetry", "aggregator_*.jsonl"))
    n_vclock = 0
    if tel:
        for line in open(tel[0], errors="replace"):
            if '"event": "agg_round"' in line or '"event":"agg_round"' in line:
                e = json.loads(line)
                if e.get("vclock_now") is not None:
                    n_vclock += 1
    notes.append(f"sim vclock_now agg_rounds={n_vclock}")
    return notes


# Peak RSS observed grading one n=100/5400s pair (both sides loaded + all 88
# rungs). Used only to pick a safe default worker count -- a pair holds both
# runs' full telemetry in memory, so jobs are RAM-bound before CPU-bound.
_PAIR_RSS_GB = 4.5


def _default_jobs(n_pairs: int) -> int:
    """One worker per pair, capped by cores and by available RAM."""
    try:
        cores = len(os.sched_getaffinity(0))
    except AttributeError:
        cores = os.cpu_count() or 1
    by_ram = n_pairs
    try:                                    # Linux: MemAvailable, in kB
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                by_ram = max(1, int(int(line.split()[1]) / 1e6 / _PAIR_RSS_GB))
                break
    except OSError:
        pass
    return max(1, min(n_pairs, cores, by_ram))


def _grade_pair(job):
    """Grade one real/sim pair. Module-level and self-contained so it can run in
    a worker process: takes paths, returns picklable primitives, writes its own
    JSON. Imports inside so a forked worker resolves them against the sys.path
    this module sets up at import time."""
    label, rdir, sdir, goal, max_bin, json_dir, sts = job
    from parity.checks import load_run_dir, run_all_parity
    try:
        from parity.checks import _WARN_ONLY_CHECKS
    except ImportError:
        _WARN_ONLY_CHECKS = set()
    real_agg, real_tr = load_run_dir(rdir)
    sim_agg, sim_tr = load_run_dir(sdir)
    res = run_all_parity(real_agg, sim_agg, real_tr, sim_tr, agg_goal=goal,
                         max_bin=max_bin)
    jpath = os.path.join(json_dir, f"parity_{label.replace('/', '_')}_{sts}.json")
    json.dump(res, open(jpath, "w"), indent=2, default=str)
    n_pass = sum(1 for v in res.values()
                 if isinstance(v, dict) and v.get("ok") and not v.get("status"))
    # a real FAIL = not-ok, not a DIAG warn, not in the warn-only allowlist
    fails = [k for k, v in res.items()
             if isinstance(v, dict) and v.get("ok") is False
             and v.get("tier") != "DIAG" and k not in _WARN_ONLY_CHECKS]
    n_skip = sum(1 for v in res.values()
                 if isinstance(v, dict) and v.get("status") == "SKIP")
    return label, res, jpath, (n_pass, len(fails), n_skip), fails


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiments-dir", default=str(_DEFAULT_EXPERIMENTS))
    ap.add_argument("--baselines", nargs="+",
                    default=["fwdllm", "fwdllm_plus", "fluxtune"])
    ap.add_argument("--json-dir", default=None,
                    help="where to write per-pair JSON (default <experiments>/_parity_reports)")
    ap.add_argument("--yes", action="store_true", help="skip the confirm prompt")
    ap.add_argument("--jobs", type=int, default=None,
                    help="parallel workers (default: one per pair, capped by "
                         "cores and by RAM at ~4.5 GB/worker). "
                         "--jobs 1 forces serial for debugging.")
    ap.add_argument("--validate", action="store_true",
                    help="also report staleness_policy + vclock_now presence")
    ap.add_argument("--max-bin", type=int, default=None,
                    help="restrict fwdllm cadence rungs (cohort_sequence/V*) to "
                         "cycle_data_id <= MAX_BIN (first-data-bin logical parity)")
    args = ap.parse_args(argv)

    from parity.checks import load_run_dir, run_all_parity  # noqa: E402
    try:
        from parity.checks import _WARN_ONLY_CHECKS  # noqa: E402
    except ImportError:
        _WARN_ONLY_CHECKS = set()

    found = _discover(args.experiments_dir)
    # Build the ordered work list: one pair per requested (baseline, trace).
    pairs = []
    for base in args.baselines:
        matches = sorted(k for k in found if k[0] == base)
        if not matches:
            print(f"  [SKIP] {base}: no run dirs found in {args.experiments_dir}")
            continue
        for key in matches:
            slot = found[key]
            if "real" not in slot or "sim" not in slot:
                have = ",".join(slot) or "none"
                print(f"  [SKIP] {'/'.join(k for k in key if k)}: "
                      f"missing a side (have: {have})")
                continue
            pairs.append((key, slot["real"], slot["sim"]))

    if not pairs:
        print("No complete real/sim pairs to check.")
        return 1

    # ── confirmation: show exactly which dirs will be compared ──
    label_w = max(len("/".join(k for k in key if k)) for key, _, _ in pairs)
    print("\n  Real<->sim pairs to check (latest per baseline):")
    for key, (rts, rdir), (sts, sdir) in pairs:
        goal_r, goal_s = _agg_goal(rdir), _agg_goal(sdir)
        goal = f"agg_goal={goal_r}" + (f"!={goal_s}⚠" if goal_s != goal_r else "")
        label = "/".join(k for k in key if k)
        print(f"    {label.ljust(label_w)}  {goal}")
        print(f"      real {rts}  {os.path.basename(rdir)}")
        print(f"      sim  {sts}  {os.path.basename(sdir)}")
    if not args.yes and sys.stdin.isatty():
        if input("\n  Proceed with these pairs? [y/N] ").strip().lower() not in ("y", "yes"):
            print("  Aborted.")
            return 0

    json_dir = args.json_dir or os.path.join(args.experiments_dir, "_parity_reports")
    os.makedirs(json_dir, exist_ok=True)

    # ── run + collect ──
    jobs = args.jobs if args.jobs else _default_jobs(len(pairs))
    work = [(("/".join(k for k in key if k)), rdir, sdir, _agg_goal(rdir) or 0,
             args.max_bin, json_dir, sts)
            for key, (_rts, rdir), (sts, sdir) in pairs]
    print(f"\n  Grading {len(pairs)} pair(s) with {jobs} worker(s) "
          f"(~{_PAIR_RSS_GB:.0f} GB each)...")

    summary = []  # (label, {name: result}, json_path, tally, fails, live_notes)
    if jobs <= 1:
        done = [_grade_pair(w) for w in work]
    else:
        # Pairs are independent: different input dirs, different output JSON, no
        # shared state -- and a pair is 77% file-load wall (13.3s of 17.3s), so
        # this is close to linear until it saturates memory or disk.
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            done = list(pool.map(_grade_pair, work))    # map preserves input order
    for (label, res, jpath, tally, fails), (_key, (_rts, rdir), (_sts, sdir)) in zip(
            done, pairs):
        live = _live_checks(label, rdir, sdir) if args.validate else []
        summary.append((label, res, jpath, tally, fails, live))

    # ── compact cross-baseline table ──
    print("\n" + "=" * 78)
    lab_w = max(len(s[0]) for s in summary)
    hdr = "  " + "baseline".ljust(lab_w) + "  " + "  ".join(lbl for _, lbl in _HEADLINE)
    print(hdr)
    for label, res, _jp, _tally, _fails, _live in summary:
        cells = []
        for name, lbl in _HEADLINE:
            cells.append(_status(res.get(name, {})).center(max(len(lbl), 1)))
        print("  " + label.ljust(lab_w) + "  " + "  ".join(cells))
    print("  legend: ✓ pass  ✗ fail  ~ warn(diag)  – skip")
    print("=" * 78)
    for label, _res, jpath, tally, fails, live in summary:
        p, f, s = tally
        print(f"  {label}: {p} pass / {f} fail / {s} skip"
              + (f"   FAILS={fails}" if fails else ""))
        # 8 rungs window on the matched budget -- never let their numbers be read
        # without the fraction of the run they were computed over.
        cov = _res.get("matched_budget_coverage") if isinstance(_res, dict) else None
        if isinstance(cov, dict) and cov.get("min_coverage") is not None:
            flag = "  <-- LOW, windowed rungs unreliable" if cov.get("degraded") else ""
            print(f"      · budget: {cov['n']} matched units grade "
                  f"real {cov['real_coverage']:.1%} / sim {cov['sim_coverage']:.1%} "
                  f"({cov['truncation']}){flag}")
            if cov.get("first_divergence"):
                d = cov["first_divergence"]
                print(f"        first divergence at unit {d['position']}: "
                      f"real={d['real_unit']} sim={d['sim_unit']}")
        for note in live:
            print(f"      · {note}")
        print(f"      json: {jpath}")

    any_fail = any(t[1] for _, _, _, t, _, _ in summary)
    return 1 if any_fail else 0


if __name__ == "__main__":
    sys.exit(main())
