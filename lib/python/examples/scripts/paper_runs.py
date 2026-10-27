#!/usr/bin/env python3
"""FX-N153: the Felix paper's manifest = felix_paper_outline Table 1 (run sets: harness_pool.FB2_SETS) + Table 2 (figure -> runs).

  paper_runs.py status [--root R] [--seeds 1 2 3] [--smoke] [--json J]   # per figure and run set: legs done / missing / stale
  paper_runs.py launch [--root R] [--prio 1] [--seeds 1] [--smoke] [--gpus 0,1,...] [--datasets cifar10]   # writes R/launch_p<prio>[_<ds>].sh

Layout: R/<cifar|speech>/<phase>/runs/<trace>_<baseline>_s<k>/run_* (bench_pool's, so every analysis tool reads it); the phase
FB2<set>[S]_<cell>_<arm> names the run set. One bench pool per dataset and seed, cifar and speech in parallel on the same GPUs.
A leg is done only if its pool's code key is the manifest's, its digest is valid (leg_digest.check), its config audit passed and it
hit the target or ran its full horizon; else stale (reason listed) or missing. `--smoke` = the FB2<set>S tiers on one cell each.
"""
import argparse
import collections
import datetime
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import harness_pool as hp  # noqa: E402
import leg_digest  # noqa: E402

BASELINES = ("felix", "fedbuff", "refl", "feddance", "fedavg", "oort", "oort_star")
DS = {"cifar10": "cifar", "google_speech": "speech"}
NAMES = {"E2E": "End-to-end", "LAD": "Ladder", "SWC": "Swap: concurrency", "SWA": "Swap: availability", "SWG": "Swap: aggregation",
         "SWS": "Swap: selection", "REJ": "Rejected ideas", "STA": "End-to-end static"}
SMOKE_CELL = {"STA": "s0"}  # smoke: one cell per run set (default smu)
ALL = ("s0u", "s20u", "s50u", "smu")
# Table 2 rows: figure -> (where, [(run set, datasets, cells, arms or None = all)]). Silent departures: FX-N155 (no run set yet).
_ABL = hp.FB2_ABL
_E2E_ABL = [("E2E", DS, _ABL, ("fx", "ctl_srv", "fbstar_srv"))]
FIGURES = {
    "teaser": ("main", [("E2E", ("cifar10",), ("smu",), None)]),
    "headline": ("main", [("E2E", DS, ALL, None)]),
    "curves": ("appendix", [("E2E", DS, ALL, None)]),
    "static": ("appendix", [("STA", DS, ("s0",), None)]),
    "efficiency": ("main", [("E2E", DS, ALL, None)]),
    "slots": ("main", [("E2E", ("cifar10",), ("s50u", "smu"), None), ("E2E", ("google_speech",), ("smu",), None)]),
    "rounds": ("main", [("E2E", DS, ("smu",), None)]),
    "staleness": ("main", [("E2E", ("cifar10",), ("s50u", "smu"), ("fx", "ctl_srv", "fbstar_srv"))]),
    "picks": ("main", [("E2E", DS, _ABL, None)]),
    "concurrency": ("main", [("E2E", ("cifar10",), ("s50u", "smu"), ("fx",)), ("E2E", ("google_speech",), ("smu",), ("fx",))]),
    "wasted": ("main", [("E2E", DS, _ABL, None)]),
    "knowledge": ("main", [("E2E", DS, ("s50u", "smu"), None)]),
    "overhead": ("appendix", [("E2E", DS, ALL, None)]),
    "ladder": ("main", [("LAD", DS, _ABL, None), ("E2E", DS, _ABL, ("fx", "ctl_srv"))]),
    "swap_availability": ("main", [("SWA", DS, _ABL, None), *_E2E_ABL]),
    "swap_selection": ("main", [("SWS", DS, _ABL, None), *_E2E_ABL]),
    "swap_aggregation": ("main", [("SWG", DS, _ABL, None), *_E2E_ABL]),
    "swap_concurrency": ("main", [("SWC", DS, _ABL, None), ("LAD", DS, _ABL, ("fn",)), *_E2E_ABL]),
    "rejected": ("main", [("REJ", DS, _ABL, None), *_E2E_ABL]),
}


def default_root(smoke=False):
    if smoke:
        return os.path.join(HERE, "..", "experiments", f"FB2S_{datetime.date.today():%m%d}")
    roots = sorted(glob.glob("/coc/scratch/dgarg/mlsys27_felix_runs_*"))
    return roots[-1] if roots else f"/coc/scratch/dgarg/mlsys27_felix_runs_{datetime.date.today():%Y%m%d}"


def tier(rs, smoke):
    return f"FB2{rs}{'S' if smoke else ''}"


def legs(rs, ds, smoke=False, cells=None, arms=None):
    """[(phase, cell, arm, baseline, trace)] of run set rs on ds (the harness tier's own plan)."""
    out = []
    for ph in hp.tier_phases(tier(rs, smoke), BASELINES, ds):
        rest = ph.pid[len(hp.DS_TAG[ds] + tier(rs, smoke)) + 1:]
        cell, arm = rest.split("_", 1)
        if smoke and cell != SMOKE_CELL.get(rs, "smu"):
            continue
        if (cells is None or cell in cells) and (arms is None or arm in arms):
            out += [(ph.pid, cell, arm, b, ph.trace) for b in ph.baselines]
    return out


def leg_dir(root, ds, phase, b, trace, seed):
    return os.path.join(root, DS[ds], phase, "runs", f"{trace}_{b}_s{seed}")


def leg_state(root, rs, ds, leg, seed, key, smoke):
    """('done' | 'missing' | 'stale', reason, digest)."""
    phase, cell, arm, b, trace = leg
    runs = sorted(glob.glob(os.path.join(leg_dir(root, ds, phase, b, trace, seed), "run_*")), key=os.path.getmtime)
    if not runs:
        return "missing", "", None
    kf = os.path.join(root, DS[ds], "code_key.txt")
    if key and (not os.path.exists(kf) or open(kf).read().strip() != key):
        return "stale", "code key", None
    d = leg_digest.load(runs[-1])
    if d is None:
        return "stale", "no digest", None
    if leg_digest.check(d):
        return "stale", "digest invalid", d
    if not (d.get("audit") or {}).get("ok"):
        return "stale", "config audit", d
    if d["end"].get("diverged"):
        return "stale", "diverged", d
    horizon_h = hp.FB1_RUNTIME_S["FB1S" if smoke else "FB1"] / 3600
    if not d["end"]["hit"] and d["end"]["h"] < horizon_h - 0.1:
        return "stale", "stopped early", d
    return "done", "", d


def required(fig, smoke=False):
    """{(run set, ds, leg)} a figure reads."""
    out = set()
    for rs, dss, cells, arms in FIGURES[fig][1]:
        for ds in dss:
            out |= {(rs, ds, lg) for lg in legs(rs, ds, smoke, None if smoke else cells, arms)}
    return out


def status(root, seeds, smoke=False, key=None):
    key = hp.code_key() if key is None else key
    states = {}
    for rs in hp.FB2_SETS:
        for ds in DS:
            for lg in legs(rs, ds, smoke):
                for s in seeds:
                    states[(rs, ds, lg, s)] = leg_state(root, rs, ds, lg, s, key, smoke)
    return states


def report(states, seeds, smoke=False):
    lines = ["| figure | where | done | missing | stale (reasons) |", "|---|---|---|---|---|"]
    for fig, (where, _) in FIGURES.items():
        c, why = collections.Counter(), collections.Counter()
        for rs, ds, lg in required(fig, smoke):
            for s in seeds:
                st, r, _ = states[(rs, ds, lg, s)]
                c[st] += 1
                why[r] += bool(r)
        lines.append(f"| {fig} | {where} | {c['done']} | {c['missing']} | {c['stale']} {dict(+why) or ''} |")
    lines += ["", "| run set | prio | dataset | done | missing | stale |", "|---|---|---|---|---|---|"]
    for rs, v in hp.FB2_SETS.items():
        for ds in DS:
            c = collections.Counter(st for (r, d, _, _), (st, _, _) in states.items() if r == rs and d == ds)
            lines.append(f"| {NAMES[rs]} | {v['prio']} | {DS[ds]} | {c['done']} | {c['missing']} | {c['stale']} |")
    return "\n".join(lines)


SPEECH = "--per-gpu 1 --leg-gpu-gb 22 --leg-ram-gb 30 --workers 4"  # speech leg peaks ~20 GB (10-25); FX-N156 measures --per-gpu 2
CIFAR = "--per-gpu 4"
FIRST = "_(sync_srv|rstar_srv|fdstar_srv|favstar_srv|ctl_srv|fbstar_srv|b1|uo_o) "  # stacks that may run the full 120 h


def launch_script(root, states, prio, seeds, smoke, gpus, datasets=tuple(DS)):
    """One tmux-launchable script: every pool of priority <= prio with missing / stale legs, cifar and speech in parallel (R20)."""
    pools, sets = [], [rs for rs, v in sorted(hp.FB2_SETS.items(), key=lambda kv: kv[1]["prio"]) if v["prio"] <= prio]
    for ds in datasets:
        for s in seeds:
            todo = [lg for (r, d, lg, sd), (st, _, _) in states.items() if r in sets and (d, sd) == (ds, s) and st != "done"]
            if not todo:
                continue
            only = "^(" + "|".join(f"{p} {b} {t}" for p, _, _, b, t in todo) + ")$"
            out = os.path.join(root, DS[ds])
            ph = {rs: sorted({lg[0] for (r, d, lg, sd), (st, _, _) in states.items() if (r, d, sd) == (rs, ds, s) and st != "done"})
                  for rs in sets}
            hargs = " ".join(f'--harness-args "--tier {tier(rs, smoke)} --datasets {ds} --baselines {",".join(BASELINES)} '
                             f"--phases '{' '.join(ph[rs])}'\"" for rs in sets if ph[rs])
            pools.append(f'{hargs} --only "{only}" --first "{FIRST}" --out {out} --gpus $GPUS {SPEECH if ds == "google_speech" else CIFAR} '
                         f'--seeds {s} --timeout-s {3600 if smoke else 86400} --digest > {out}_s{s}.log 2>&1  # {len(todo)} legs')
    if not pools:
        return None
    body = "\n".join(f'setsid $PY scripts/bench_pool.py {p.split("  # ")[0]} & pids+=($!)  # {p.split("  # ")[1]}' for p in pools)
    return f"""#!/usr/bin/env bash
# paper_runs.py launch (FX-N153): priority <= {prio}, seeds {seeds}{', smoke' if smoke else ''}; one Ctrl+C stops all (R20).
set -uo pipefail
cd "{os.path.abspath(os.path.join(HERE, '..'))}" || exit 1
[ -z "$(git status --porcelain -uno)" ] || {{ echo "[paper_runs] tree dirty"; exit 1; }}
PY="conda run --no-capture-output -n dg_flame python"
[ "$(hostname -s)" = shepherd ] || [ -n "${{GPUS:-}}" ] || {{ echo "[paper_runs] off shepherd: ask the operator, then set GPUS"; exit 1; }}
GPUS=${{GPUS:-{gpus}}}
mkdir -p {" ".join(os.path.join(root, d) for d in DS.values())}
git log -1 --format='%h %s' > {root}/code_state.txt
pids=()
stop() {{ echo "[paper_runs] stopping"; for p in "${{pids[@]}}"; do kill -TERM -- -"$p" 2>/dev/null; done; sleep 10
         for p in "${{pids[@]}}"; do kill -KILL -- -"$p" 2>/dev/null; done; exit 130; }}
trap stop INT TERM HUP
{body}
wait
$PY scripts/paper_runs.py status --root {root} --seeds {" ".join(map(str, seeds))}{' --smoke' if smoke else ''} > {root}/status.md 2>&1
echo "[paper_runs] done: {root}/status.md"
"""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("status", "launch"))
    ap.add_argument("--root")
    ap.add_argument("--seeds", nargs="+", type=int, default=[1])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--prio", type=int, default=1)
    ap.add_argument("--gpus", default="0,1,2,3")  # shepherd 0-3; elsewhere the launch script demands GPUS (operator 10-26)
    ap.add_argument("--datasets", nargs="+", choices=tuple(DS), default=list(DS), help="launch only these (FX-N159)")
    ap.add_argument("--code-key", help="the manifest's code key (default: this tree's)")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    root = os.path.abspath(a.root or default_root(a.smoke))
    states = status(root, a.seeds, a.smoke, a.code_key)
    if a.cmd == "status":
        print(f"# paper_runs status: {root}, seeds {a.seeds}\n\n" + report(states, a.seeds, a.smoke))
        if a.json:
            json.dump([dict(set=rs, dataset=ds, phase=lg[0], cell=lg[1], arm=lg[2], baseline=lg[3], seed=s, state=st, reason=r)
                       for (rs, ds, lg, s), (st, r, _) in states.items()], open(a.json, "w"), indent=1)
        return 0
    sh = launch_script(root, states, a.prio, a.seeds, a.smoke, a.gpus, a.datasets)
    if sh is None:
        print("[paper_runs] nothing to launch")
        return 0
    os.makedirs(root, exist_ok=True)
    only = "" if set(a.datasets) == set(DS) else "_" + "_".join(DS[d] for d in a.datasets)
    path = os.path.join(root, f"launch_p{a.prio}{'_smoke' if a.smoke else ''}{only}.sh")
    open(path, "w").write(sh)
    print(f"[paper_runs] wrote {path}: tmux -> bash {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
