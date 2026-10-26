#!/usr/bin/env python3
"""Watch bench / harness pools: legs done / failed / running, and ALERTs for what needs a look.

  python scripts/watch_runs.py experiments/EV24_1025_ev2 experiments/B2_cifar10_1025_c2 experiments/FB2S_1026 [--every 600]

ALERTs: a FAIL / FAIL-FAST / DOOMED / ABORT line, a running bench leg whose leg.out has a traceback or CUDA OOM, a running
bench leg whose telemetry has not grown in --stall-min minutes, a GPU at 0% SM while legs run on it.
"""
import argparse
import glob
import os
import re
import subprocess
import time

BAD = re.compile(r"FAIL|DOOMED|ABORT|Traceback|out of memory")
EVENT = re.compile(r"^\[[^\]]+\] (START|DONE|FAIL)\b")  # bench_pool and harness_pool leg lines


def pool_logs(root):
    """bench_pool / harness_pool logs at a pool root (the per-pool *.log files)."""
    return sorted(p for p in glob.glob(os.path.join(root, "*.log")) if os.path.isfile(p))


def counts(log):
    n = {"START": 0, "DONE": 0, "FAIL": 0}
    bad = []
    for ln in open(log, errors="replace"):
        m = EVENT.search(ln)
        if m:
            n[m.group(1)] += 1
        if BAD.search(ln) and "bench_validate" not in ln:
            bad.append(ln.strip()[:220])
    return n, bad


def running_legs(root):
    """Bench leg dirs with a run_* but no bench_summary.json yet (still running or killed)."""
    out = []
    for leg in glob.glob(os.path.join(root, "**", "runs", "*_s[0-9]*"), recursive=True):
        runs = sorted(glob.glob(os.path.join(leg, "run_*")), key=os.path.getmtime)
        if runs and not os.path.exists(os.path.join(runs[-1], "bench_summary.json")):
            out.append((leg, runs[-1]))
    return out


def leg_alerts(leg, run, stall_s, live):
    al = []
    lo = os.path.join(leg, "leg.out")
    if os.path.exists(lo):
        txt = open(lo, errors="replace").read()
        if "Traceback" in txt or "out of memory" in txt:
            al.append(f"traceback/OOM in {lo}")
    tel = glob.glob(os.path.join(run, "telemetry", "aggregator_*.jsonl"))
    if live and tel and time.time() - os.path.getmtime(tel[0]) > stall_s:
        al.append(f"no telemetry for {(time.time() - os.path.getmtime(tel[0])) / 60:.0f} min: {run}")
    return al


def gpus():
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=30).stdout
        return [tuple(int(x) for x in ln.split(",")) for ln in q.strip().splitlines()]
    except Exception:
        return []


def live_pids():
    return subprocess.run(["pgrep", "-f", "fl_bench.py|harness_pool.py|bench_pool.py"], capture_output=True, text=True).stdout.split()


def once(roots, stall_s, seen):
    print(f"\n===== {time.strftime('%F %T')} =====", flush=True)
    live = bool(live_pids())
    alerts = []
    for root in roots:
        for log in pool_logs(root):
            n, bad = counts(log)
            print(f"{os.path.relpath(log)}: started {n['START']} done {n['DONE']} failed {n['FAIL']}")
            for b in bad:
                if (log, b) not in seen:
                    seen.add((log, b))
                    alerts.append(f"{os.path.basename(log)}: {b}")
        for leg, run in running_legs(root):
            for a in leg_alerts(leg, run, stall_s, live):
                if a not in seen:
                    seen.add(a)
                    alerts.append(a)
        for f in glob.glob(os.path.join(root, "**", "ABORT.txt"), recursive=True):
            if f not in seen:
                seen.add(f)
                alerts.append(f"pool aborted: {f}")
    g = gpus()
    if g:
        print("GPUs (idx MiB %SM): " + "  ".join(f"{i}:{m}/{u}%" for i, m, u in g))
        idle = [i for i, m, u in g if m > 1000 and u == 0]
        if idle and live:
            alerts.append(f"GPU(s) {idle} hold memory at 0% SM (stuck leg?)")
    print(f"processes alive: {'yes' if live else 'NO (all pools finished or died)'}")
    for a in alerts:
        print(f"ALERT {a}", flush=True)
    if not alerts:
        print("OK: no new alerts", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--every", type=int, default=600, help="seconds between checks; 0 = once")
    ap.add_argument("--stall-min", type=float, default=15)
    a = ap.parse_args()
    seen = set()
    while True:
        once(a.roots, a.stall_min * 60, seen)
        if not a.every:
            return
        time.sleep(a.every)


if __name__ == "__main__":
    main()
