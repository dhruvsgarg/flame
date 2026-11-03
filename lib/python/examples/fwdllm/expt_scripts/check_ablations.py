#!/usr/bin/env python3
"""Mid-run health check for queue_ablations_bench.sh: python check_ablations.py <OUT> [--stall-min 20]
Per arm: state (queued / running / DONE / FAIL from bench.log), commits, evals, best and last acc, vclock h, minutes since the
last telemetry write (STALL if > --stall-min), error counts in leg.out. Then GPU memory and the bench process count."""
import argparse, glob, json, os, re, subprocess, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--stall-min", type=float, default=20)
ap.add_argument("--jobs"); ap.add_argument("--pool"); ap.add_argument("--bench-log")
a = ap.parse_args()
out = Path(a.out)
jobs = Path(a.jobs or out / "jobs.tsv"); pool = Path(a.pool or out / "POOL"); blog = Path(a.bench_log or out / "bench.log")
arms = [l.split("\t")[0] for l in jobs.read_text().splitlines() if l.strip()]
blines = blog.read_text().splitlines() if blog.exists() else []
state = {}
for l in blines:
    m = re.match(r"\[bench_pool\] (START|DONE|FAIL rc=\S+) (\S+) ", l)
    if m: state[m.group(2)] = m.group(1)

print(f"{time.strftime('%m-%d %H:%M:%S')}  {out}")
print(f"{'arm':5} {'state':12} {'commits':>7} {'evals':>5} {'best':>6} {'last':>6} {'vclk h':>7} {'idle min':>8}  flags")
n_stall = n_fail = 0
for arm in arms:
    st = state.get(arm, "queued")
    runs = sorted(glob.glob(str(pool / arm / "runs" / "*" / "run_*"))) if st != "queued" else []
    commits = evals = rounds = 0; accs = []; vc = 0.0; idle = None; flags = []
    if runs:
        agg = glob.glob(runs[-1] + "/telemetry/aggregator*.jsonl")
        if agg:
            idle = (time.time() - os.path.getmtime(agg[0])) / 60
            for l in open(agg[0], errors="replace"):
                try: r = json.loads(l)
                except ValueError: continue
                vc = r.get("vclock_now", vc) or vc
                e = r.get("event")
                if e == "version_bump_census": commits += 1
                elif e == "agg_round": rounds += 1
                elif e == "agg_eval": evals += 1; accs.append(r.get("test-accuracy"))
    commits = commits or rounds  # sync arms have no version-bump events
    legs = glob.glob(str(pool / arm / "runs" / "*" / "leg.out"))
    if legs:
        txt = Path(legs[0]).read_text(errors="replace")
        for k, pat in (("Traceback", "Traceback"), ("OOM", "out of memory|OOM"), ("nan", r"\bnan\b")):
            c = len(re.findall(pat, txt))
            if c: flags.append(f"{k}x{c}")
    if st == "START" and idle is not None and idle > a.stall_min: flags.append("STALL"); n_stall += 1
    if st.startswith("FAIL"): n_fail += 1
    accs = [x for x in accs if x is not None]
    f = lambda x, p: f"{x:{p}}" if x is not None else "-"
    print(f"{arm:5} {st:12} {commits:7d} {evals:5d} {f(max(accs) if accs else None, '.3f'):>6} {f(accs[-1] if accs else None, '.3f'):>6} "
          f"{vc / 3600:7.2f} {f(idle, '.0f'):>8}  {' '.join(flags)}")
print(f"\nFAIL {n_fail}  STALL {n_stall}  running {sum(v == 'START' for v in state.values())}  "
      f"done {sum(v == 'DONE' for v in state.values())}  queued {sum(a_ not in state for a_ in arms)}")
try:
    g = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader"], capture_output=True, text=True).stdout
    print("GPU (idx, MiB used, util):", "; ".join(" ".join(x.split(", ")) for x in g.strip().splitlines()))
except OSError:
    pass
print("last bench.log lines:"); print("\n".join("  " + l[:160] for l in blines[-4:]))
