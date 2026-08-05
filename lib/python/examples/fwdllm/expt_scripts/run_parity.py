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
    python run_parity.py --control --baselines fedbuff_round --duration 7200
                                               # real<->real CONTROL: no sim leg

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
from itertools import combinations
from pathlib import Path

import yaml

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
    latest ts per (baseline, trace, variant).

    The real leg is the latest whose `jvp_eval_mode` AND `max_runtime_s` match
    the sim's: taking the latest outright grades an OFF control against an ON sim
    leg, i.e. the flag rather than the code, and a 4h real against a 2h sim,
    i.e. the run length rather than the code (simulate_fwdllm.md §A).
    """
    out: dict = {}
    for path in glob.glob(os.path.join(experiments_dir, "run_*")):
        m = _RUN_RE.match(os.path.basename(path))
        if not m:
            continue
        key = (m["baseline"], m["trace"] or "")
        legs = out.setdefault(key, {"real": [], "sim": []})
        legs[m["variant"]].append((m["ts"], path))

    paired: dict = {}
    for key, legs in out.items():
        slot = {}
        sims = sorted(legs["sim"], reverse=True)
        reals = sorted(legs["real"], reverse=True)
        if sims:
            slot["sim"] = sims[0]
        if reals:
            slot["real"] = reals[0]
        if sims and reals:
            want = (_jvp_eval_mode(sims[0][1]), _max_runtime_s(sims[0][1]))
            comparable = [r for r in reals
                          if (_jvp_eval_mode(r[1]), _max_runtime_s(r[1])) == want]
            # Prefer a real that ran the SAME CODE as the sim (§D-70). A charge
            # re-profile or a simulator change between the two legs is graded as
            # a residual otherwise -- it moved `fwdllm` 18.4 points. Fall back to
            # the latest comparable real and SAY the code differs, rather than
            # grading nothing.
            from replicate_floor import code_version
            sim_sha = code_version(sims[0][1])[0]
            same_code = [r for r in comparable if code_version(r[1])[0] == sim_sha]
            match = (same_code or comparable or [None])[0]
            if match is not None:
                slot["real"] = match
            slot["_flag"] = want
            slot["_sim_sha"] = sim_sha
            slot["_same_code"] = bool(same_code)
            slot["_same_code_n"] = len(same_code)
            slot["_flag_skipped"] = [r for r in reals if r[0] > slot["real"][0]]
        if slot:
            paired[key] = slot
    return paired


def _jvp_eval_mode(run_dir: str) -> bool:
    """Was this leg trained with dropout off inside the JVP? Only the trainer log
    records it (§B.6). Absent = predates the flag = old default, dropout live."""
    for lg in glob.glob(os.path.join(run_dir, "*trainers.log")):
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


def _max_runtime_s(run_dir: str):
    """The run length the leg was configured for — half of what makes two legs
    comparable. A 4h real against a 2h sim grades the run length, not the code."""
    cfg = os.path.join(run_dir, "aggregator_config.json")
    if not os.path.exists(cfg):
        return None
    try:
        h = json.load(open(cfg)).get("hyperparameters", {})
        return h.get("max_runtime_s") or h.get("maxRuntimeS")
    except (ValueError, OSError):
        return None


def _code_sha(run_dir: str):
    from replicate_floor import code_version
    return code_version(run_dir)[0]


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


_FLOOR_DIR = _HERE.parent / "parity_floors"


def _floors(label: str) -> dict | None:
    """This baseline's measured replicate floor, if one has been generated.

    Sizes the DIST tolerances (§D-24). Absent file => nominal tolerances, so a
    baseline with no replicate is graded exactly as before rather than silently
    on someone else's floor (§D-36).

    A --control pair is labelled with its POOLED group name (§D-63), which has no
    file of its own — the floor is written to each member. Members carry the same
    pooled profile, so reading either is the same number; grading a control at
    nominal while the board is floor-gated would make the two disagree by
    construction (§D-65).

    TWO-SIDED (§D-61): a real↔sim residual draws one leg from each side, so a
    gate sized on real's spread alone assumes sim is deterministic. `fedbuff_round`
    reproduces to 0.7% real and 21.2% sim, and 3x the real floor failed a residual
    well inside sim's own noise. Take the max — a residual below what EITHER side
    reproduces to carries no information. NOT the spread pooled across both sides:
    that would fold a genuine real↔sim bias into the floor and hide it (§D-5).
    """
    from replicate_floor import pool_members
    name = label.split("/")[0]
    path = _FLOOR_DIR / f"{name}.yaml"
    if not path.exists():
        path = next((p for m in pool_members(name)
                     if (p := _FLOOR_DIR / f"{m}.yaml").exists()), path)
    if not path.exists():
        return None
    prof = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    real, sim = prof.get("metrics") or {}, prof.get("sim_metrics") or {}
    if not real and not sim:
        return None
    return {m: max(v for v in (real.get(m), sim.get(m)) if v is not None)
            for m in set(real) | set(sim)}


def _grade_pair(job):
    """Grade one real/sim pair. Module-level and self-contained so it can run in
    a worker process: takes paths, returns picklable primitives, writes its own
    JSON. Imports inside so a forked worker resolves them against the sys.path
    this module sets up at import time."""
    label, rdir, sdir, goal, max_bin, json_dir, sts, prefix = job
    # A CONTROL pair is two legs of the SAME mode, so the clock rungs must not
    # demand a vclock on the B side (§D-56) -- that bail is what made all three
    # unreadable real↔real. Each leg still reads its own vclock where it has one.
    same_mode = prefix == "control"
    from parity.checks import load_run_dir, run_all_parity
    try:
        from parity.checks import _WARN_ONLY_CHECKS
    except ImportError:
        _WARN_ONLY_CHECKS = set()
    real_agg, real_tr = load_run_dir(rdir)
    sim_agg, sim_tr = load_run_dir(sdir)
    res = run_all_parity(real_agg, sim_agg, real_tr, sim_tr, agg_goal=goal,
                         max_bin=max_bin, floors=_floors(label),
                         same_mode=same_mode)
    jpath = os.path.join(json_dir, f"{prefix}_{label.replace('/', '_')}_{sts}.json")
    json.dump(res, open(jpath, "w"), indent=2, default=str)
    n_pass = sum(1 for v in res.values()
                 if isinstance(v, dict) and v.get("ok") and not v.get("status"))
    # a real FAIL = not-ok, not a DIAG warn, not in the warn-only allowlist
    fails = [k for k, v in res.items()
             if isinstance(v, dict) and v.get("ok") is False
             and v.get("tier") != "DIAG" and k not in _WARN_ONLY_CHECKS]
    n_skip = sum(1 for v in res.values()
                 if isinstance(v, dict) and v.get("status") == "SKIP")
    # A fail on a rung whose gate was never derived from a measured floor is
    # weaker evidence than one on a floor-gated rung -- and its 0-fail siblings
    # may be blind rather than clean (§D-24, §B.5). Name them apart.
    underived = [k for k in fails
                 if res[k].get("threshold_provenance") == "CALIBRATED"
                 and not res[k].get("gate_derived")]
    return label, res, jpath, (n_pass, len(fails), n_skip), fails, underived


# ── the CONTROL: same-mode replicate legs graded against each other ──────────
#
# A red rung is not evidence until the control says it is (§D-55): five rungs
# fail between config-identical REAL legs, so their real<->sim "failure" measures
# the pipeline's own noise. This is the primary reader of every DIST verdict.

def _bailed_for_sim_field(res: dict) -> bool:
    """Did this rung BAIL for want of a sim-only field rather than measure a
    difference? (§D-56) The bail is shaped exactly like a fail.

    Detected from the result, never a static list, because readability is a
    property of the PAIR: `throughput` cannot be read with two real legs and can
    be read with two sim ones. The tell is a narrative field naming the missing
    stamp -- those fields carry a metric's name only when it is absent.
    """
    if res.get("ok") is not False:
        return False
    narrative = json.dumps([res.get(k) for k in
                            ("note", "reason", "issues", "violations", "detail")],
                           default=str)
    return "vclock_now" in narrative or "sim_send_ts" in narrative


def _control_groups(experiments_dir: str, baselines, mode: str, duration=None,
                    span_tol: float = 0.05, jvp: str = "on",
                    any_code: bool = False) -> list:
    """[(key, kept, dropped, code_dropped), ...] per config with >= 2 comparable legs.

    Grouping, the same-CODE filter and the truncated-leg drop all come from
    `replicate_floor`, so a control pair and a floor are measured over exactly the
    same set of legs (§D-44/§D-59/§D-70). They are the same measurement and must
    not disagree (§D-65): without the code filter the control read 13.6% on
    `fedbuff_it`'s time-to-N where the same-code floor read 3.0% -- the gap was
    code drift, and the control was reporting it as pipeline noise.
    `jvp` filters on the training flag: an OFF group is a different training
    config, so pooling its pairs into the roll-up would grade the flag (§D-45).
    """
    from replicate_floor import discover, drop_truncated, largest_same_code
    groups = discover(experiments_dir, baselines, mode)
    out = []
    for key in sorted(groups, key=lambda k: (k[0], k[3] or 0, k[4])):
        if duration is not None and (key[3] or 0) != duration:
            continue
        if jvp != "any" and key[4] != (jvp == "on"):
            continue
        legs, code_dropped = groups[key], []
        if not any_code:
            legs, code_dropped, _sha = largest_same_code(legs)
        kept, dropped, axis = drop_truncated(legs, span_tol)
        if len(kept) >= 2:
            out.append((key, kept, dropped, code_dropped, axis))
    return out


def _control_report(done: list, pair_keys: list, warn_only=frozenset()) -> dict:
    """Per-rung fail counts over every control pair — §A.3's table, computed.

    Returns {rung: {"fail": n, "pairs": n, "unreadable": n, "where": [...]}}.
    A fail is counted exactly as the scoreboard counts one — not DIAG, not
    warn-only — or the two tables would disagree about the same rung.
    """
    tally: dict = {}
    for (_label, res, _jp, _t, _fails), (glabel, _a, _b) in zip(done, pair_keys):
        for rung, r in res.items():
            if not isinstance(r, dict):
                continue
            t = tally.setdefault(rung, {"fail": 0, "pairs": 0, "unreadable": 0,
                                        "skip": 0, "where": []})
            if _bailed_for_sim_field(r):
                t["unreadable"] += 1
                continue
            if r.get("status") == "SKIP":
                t["skip"] += 1
                continue
            t["pairs"] += 1
            if (r.get("ok") is False and r.get("tier") != "DIAG"
                    and rung not in warn_only):
                t["fail"] += 1
                t["where"].append(glabel)
    return tally


def _run_control(args, json_dir: str) -> int:
    """Grade every pair of every config's replicate legs, same mode both sides."""
    modes = ("real", "sim") if args.control_mode == "both" else (args.control_mode,)
    groups = []
    for mode in modes:
        groups += [(mode, *g) for g in _control_groups(
            args.experiments_dir, args.baselines, mode,
            args.duration, args.span_tol, args.jvp_eval_mode, args.any_code)]
    if not groups:
        print("No replicate groups found — a control needs >= 2 legs of one "
              "config in the SAME mode.")
        return 1

    work, pair_keys = [], []
    print("\n  CONTROL pairs (same mode both sides — no sim leg involved):")
    for mode, key, kept, dropped, code_dropped, axis in groups:
        baseline, trace, _m, maxrt, jvp = key
        label = "/".join(x for x in (baseline, trace) if x)
        legs = sorted(kept)
        print(f"    {label}  mode={mode}  max_runtime_s={maxrt}  "
              f"jvp_eval_mode={jvp}  n_legs={len(legs)} "
              f"-> {len(legs) * (len(legs) - 1) // 2} pairs")
        for ts, _p, span in dropped:
            print(f"      [dropped {ts} — achieved {axis} span {span:.0f}s, truncated]")
        for ts, _p in code_dropped:
            print(f"      [dropped {ts} — ran other code (§D-70); --any-code pools]")
        for (ts_a, dir_a, _sa), (ts_b, dir_b, _sb) in combinations(legs, 2):
            work.append((label, dir_a, dir_b, _agg_goal(dir_a) or 0, args.max_bin,
                         json_dir, f"{mode}_{ts_a}_{ts_b}", "control"))
            pair_keys.append((f"{baseline}@{mode} {ts_a[4:]}~{ts_b[4:]}",
                              dir_a, dir_b))

    if not args.yes and sys.stdin.isatty():
        if input(f"\n  Grade {len(work)} control pair(s)? [y/N] "
                 ).strip().lower() not in ("y", "yes"):
            print("  Aborted.")
            return 0

    jobs = args.jobs if args.jobs else _default_jobs(len(work))
    print(f"\n  Grading {len(work)} control pair(s) with {jobs} worker(s)...")
    if jobs <= 1:
        done = [_grade_pair(w) for w in work]
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            done = list(pool.map(_grade_pair, work))

    print("\n" + "=" * 78)
    for (label, _res, jpath, tally, fails, underived), (glabel, _a, _b) in zip(done, pair_keys):
        p, f, s = tally
        print(f"  {glabel}: {p} pass / {f} fail / {s} skip"
              + (f"   FAILS={fails}" if fails else "")
              + (f"   [gate never derived: {underived}]" if underived else ""))
    try:
        from parity.checks import _WARN_ONLY_CHECKS
    except ImportError:
        _WARN_ONLY_CHECKS = set()
    tally = _control_report([d[:5] for d in done], pair_keys, _WARN_ONLY_CHECKS)
    print("\n  Per-rung fail rate on config-identical legs — a rung failing here "
          "\n  measures the pipeline's own noise, not sim (§D-55):")
    print(f"    {'rung':34s} {'fails/pairs':>12s}   where")
    rows = sorted(tally.items(), key=lambda kv: (-kv[1]["fail"], kv[0]))
    for rung, t in rows:
        if not t["fail"]:
            continue
        print(f"    {rung:34s} {t['fail']:>5d}/{t['pairs']:<6d}   "
              + ", ".join(sorted(set(t["where"]))))
    # A rung that bailed on ANY pair is not "clean" — it is partly unmeasured,
    # and listing it as 0-fail is the exact misreading §D-56 warns about.
    clean = [r for r, t in rows if not t["fail"] and t["pairs"] and not t["unreadable"]]
    print(f"\n    0-fail over every pair ({len(clean)}): " + ", ".join(sorted(clean)))
    unread = sorted((r, t) for r, t in tally.items() if t["unreadable"])
    if unread:
        print("\n    UNREADABLE here — the rung reads a sim-only field and BAILS,"
              "\n    which looks like a fail (§D-56): "
              + ", ".join(f"{r} ({t['unreadable']}/{t['unreadable'] + t['pairs']})"
                          for r, t in unread))
    print("\n  The control is a MEASUREMENT, not a gate: exit 0 whatever it finds.")
    return 0


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
    ap.add_argument("--control", action="store_true",
                    help="real<->real (or sim<->sim) CONTROL instead of the "
                         "real/sim pair: grade EVERY pair of a config's replicate "
                         "legs and print the per-rung fail rate. A red rung is not "
                         "evidence until this says it is (D-55); needs no sim leg")
    ap.add_argument("--control-mode", choices=("real", "sim", "both"),
                    default="real", help="which side to replicate (default real)")
    ap.add_argument("--duration", type=float, default=None, metavar="S",
                    help="--control: grade only legs at this max_runtime_s")
    ap.add_argument("--span-tol", type=float, default=0.05,
                    help="--control: drop a leg whose ACHIEVED span is this far "
                         "below the group's longest (D-44)")
    ap.add_argument("--any-code", action="store_true",
                    help="--control: pair legs that ran DIFFERENT code. Off by "
                         "default, matching replicate_floor: a control across code "
                         "versions measures the diff, not the pipeline (D-70)")
    ap.add_argument("--jvp-eval-mode", choices=("on", "off", "any"), default="on",
                    help="--control: which training config to grade (default on; "
                         "an OFF group never pools with an ON one, D-45)")
    args = ap.parse_args(argv)

    json_dir = args.json_dir or os.path.join(args.experiments_dir, "_parity_reports")
    os.makedirs(json_dir, exist_ok=True)
    if args.control:
        return _run_control(args, json_dir)

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
            pairs.append((key, slot["real"], slot["sim"],
                          slot.get("_flag"), slot.get("_flag_skipped") or []))

    if not pairs:
        print("No complete real/sim pairs to check.")
        return 1

    # ── confirmation: show exactly which dirs will be compared ──
    label_w = max(len("/".join(k for k in key if k)) for key, *_ in pairs)
    print("\n  Real<->sim pairs to check (latest per baseline, flag-matched):")
    for key, (rts, rdir), (sts, sdir), flag, skipped in pairs:
        goal_r, goal_s = _agg_goal(rdir), _agg_goal(sdir)
        goal = f"agg_goal={goal_r}" + (f"!={goal_s}⚠" if goal_s != goal_r else "")
        label = "/".join(k for k in key if k)
        print(f"    {label.ljust(label_w)}  {goal}")
        print(f"      real {rts}  {os.path.basename(rdir)}")
        print(f"      sim  {sts}  {os.path.basename(sdir)}")
        if flag is not None:
            print(f"      jvp_eval_mode={flag[0]}, max_runtime_s={flag[1]} on both legs")
            slot = found[key]
            if slot.get("_same_code"):
                print(f"      code {slot['_sim_sha']} on both legs "
                      f"({slot['_same_code_n']} same-code real(s) available)")
            else:
                print(f"      ⚠ CODE DIFFERS — sim {slot.get('_sim_sha')} vs real "
                      f"{_code_sha(rdir)}; residual includes the diff (§D-70)")
        for sts_, sdir_ in skipped:
            got = (_jvp_eval_mode(sdir_), _max_runtime_s(sdir_))
            # Both flags matching means it was passed over for CODE (§D-70) --
            # say so, or the line reads "-- differs" naming nothing.
            why = " and ".join(
                f"{n} differs" for n, a, b in (("jvp_eval_mode", got[0], flag[0]),
                                               ("max_runtime_s", got[1], flag[1]))
                if a != b)
            why = why or (f"code differs ({_code_sha(sdir_)} vs sim's "
                          f"{found[key]['_sim_sha']})")
            print(f"      [skipped newer real {sts_} "
                  f"{os.path.basename(sdir_)} -- {why}]")
    if not args.yes and sys.stdin.isatty():
        if input("\n  Proceed with these pairs? [y/N] ").strip().lower() not in ("y", "yes"):
            print("  Aborted.")
            return 0

    # ── run + collect ──
    jobs = args.jobs if args.jobs else _default_jobs(len(pairs))
    work = [(("/".join(k for k in key if k)), rdir, sdir, _agg_goal(rdir) or 0,
             args.max_bin, json_dir, sts, "parity")
            for key, (_rts, rdir), (sts, sdir), _flag, _skipped in pairs]
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
    for (label, res, jpath, tally, fails, underived), (_key, (_rts, rdir),
                                                      (_sts, sdir), _flag,
                                                      _skipped) in zip(done, pairs):
        live = _live_checks(label, rdir, sdir) if args.validate else []
        if underived:
            live.append(f"⚠ gate never derived from a floor: {', '.join(underived)}")
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
