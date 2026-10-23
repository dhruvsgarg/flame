#!/usr/bin/env python3
"""FX-N129 (V5): run fl_bench legs packed on GPUs; harness pool layout (POOL/<phase>/summary.tsv) for the analysis tools.

  bench_pool.py --from-pools F42B_POOLS... --out POOL [--seeds 1 2 3] [--only REGEX] [--gpus 0,2,3] [--per-gpu 4]
  bench_pool.py --jobs JOBS.tsv --out POOL ...        # JOBS.tsv: phase baseline trace yaml (one leg per line)
  bench_pool.py --harness-args "--tier F42 --datasets cifar10 --phases '...'" --out POOL ...   # the harness's own legs, now

Every leg runs `fl_bench.py` under a hard timeout; one Ctrl+C stops all (R20). Seed i runs aggregator seed 1233 + i, the
harness's `--seeds` convention. Prints a progress line per finished leg and the pool's legs / h at the end.
"""
import argparse
import glob
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable


def jobs_from_pools(pools):
    """(phase, baseline, trace, yaml) per sim leg of harness pools; `_r<i>` replicates collapse onto their arm."""
    out = {}
    for pool in pools:
        for y in sorted(glob.glob(os.path.join(pool, "*", "*_sim", "runs", "*", "debug_run.yaml"))):
            leg = y.split(os.sep)[-4]
            m = re.match(r"(.+?)_(syn_\w+?|mobiperf\w*?)_([a-z_]+)_sim$", leg)
            if not m:
                continue
            phase, trace, b = m.groups()
            phase = re.sub(r"_r\d+$", "", phase)
            out.setdefault((phase, b, trace), y)
    return [(p, b, t, y) for (p, b, t), y in sorted(out.items())]


def jobs_from_harness(args, out):
    """harness_pool's plan for `args` (same tiers, arms, knobs), each sim leg's YAML from `harness_suite.sh --dry-run` on this tree."""
    import shlex
    sys.path.insert(0, str(HERE))
    import harness_pool as hp
    ap = hp._parser()
    jobs = [j for j in hp.plan(ap.parse_args(shlex.split(args)), ap)[0] if j.mode == "sim"]
    res = []
    for j in jobs:
        d = out.resolve() / "_yaml" / j.jid
        suite = hp.example_dir(j.dataset) / "scripts" / "harness_suite.sh"
        subprocess.run(["bash", str(suite), *j.args, "--dry-run", "--output-dir", str(d)], cwd=str(hp.example_dir(j.dataset)),
                       stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, timeout=600, check=True)
        y = glob.glob(str(d / "runs" / "*" / "debug_run.yaml"))
        if len(y) != 1:
            raise RuntimeError(f"{j.jid}: dry run gave {len(y)} yaml")
        res.append((j.phase, j.baseline, j.trace, y[0]))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-pools", nargs="*", default=[])
    ap.add_argument("--jobs")
    ap.add_argument("--harness-args", default="", help="harness_pool args; legs from its current plan (dry-run YAMLs)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[1])
    ap.add_argument("--only", default="", help="regex on '<phase> <baseline> <trace>'")
    ap.add_argument("--gpus", default="0,2,3,4,5,6,7")
    ap.add_argument("--per-gpu", type=int, default=4)
    ap.add_argument("--gpus-per-leg", type=int, default=1, help="a leg's workers spread over this many GPUs (speech)")
    ap.add_argument("--timeout-s", type=float, default=900)
    ap.add_argument("--runtime-s", type=float, default=None, help="vclock budget override (smokes)")
    ap.add_argument("--workers", type=int, default=2, help="trainer worker processes per leg (fl_bench --workers)")
    a = ap.parse_args(argv)
    jobs = jobs_from_pools(a.from_pools)
    if a.jobs:
        jobs += [tuple(line.split("\t")[:4]) for line in open(a.jobs).read().splitlines() if line and not line.startswith("#")]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.harness_args:
        jobs += jobs_from_harness(a.harness_args, out)
    jobs = [j for j in jobs if re.search(a.only, " ".join(j[:3]))]
    queue = [(j, s) for j in jobs for s in a.seeds]
    g = a.gpus.split(",")
    groups = [",".join(g[i:i + a.gpus_per_leg]) for i in range(0, len(g) - a.gpus_per_leg + 1, a.gpus_per_leg)]
    slots = [x for x in groups for _ in range(a.per_gpu)]
    running, done, t0 = {}, [], time.time()
    print(f"[bench_pool] {len(queue)} legs on {len(slots)} slots ({a.gpus} x {a.per_gpu}), timeout {a.timeout_s:.0f}s", flush=True)

    def kill(p, sig):
        try:
            os.killpg(p.pid, sig)
        except ProcessLookupError:
            pass

    def stop(*_):
        for p, *_ in running.values():
            kill(p, signal.SIGTERM)
        time.sleep(3)
        for p, *_ in running.values():
            kill(p, signal.SIGKILL)
        sys.exit(130)
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    while queue or running:
        for slot in [s for s in range(len(slots)) if s not in running]:
            if not queue:
                break
            (phase, b, trace, y), seed = queue.pop(0)
            d = out / phase / "runs" / f"{trace}_{b}_s{seed}"
            d.mkdir(parents=True, exist_ok=True)
            cmd = [PY, str(HERE / "fl_bench.py"), y, "--out", str(d), "--gpu", slots[slot], "--seed", str(1233 + seed), "--workers", str(a.workers)]
            if a.runtime_s:
                cmd += ["--runtime-s", str(a.runtime_s)]
            log = open(d / "leg.out", "w")
            p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)  # own group: kill takes workers
            running[slot] = (p, phase, b, trace, seed, d, time.time())
        time.sleep(2)
        for slot, (p, phase, b, trace, seed, d, ts) in list(running.items()):
            if p.poll() is None and time.time() - ts > a.timeout_s:
                kill(p, signal.SIGKILL)
                p.wait()
                p.returncode = 124
            if p.poll() is None:
                continue
            kill(p, signal.SIGKILL)  # leftover workers
            del running[slot]
            res = next((json.loads(line) for line in open(d / "leg.out") if line.startswith('{"run_dir"')), None)
            ok = p.returncode == 0 and res is not None
            done.append(ok)
            if ok:
                with open(out / phase / "summary.tsv", "a") as f:
                    if f.tell() == 0:
                        f.write("baseline\ttrace\tseed\treal_dir\tsim_dir\n")
                    f.write(f"{b}\t{trace}\t{seed}\t\t{res['run_dir']}\n")
            print(f"[bench_pool] {'DONE' if ok else 'FAIL rc=' + str(p.returncode)} {phase} {b} {trace} s{seed} "
                  f"{(time.time() - ts) / 60:.1f} min {res and res.get('tasks')} tasks "
                  f"({len(done)}/{len(done) + len(queue) + len(running)})", flush=True)
    h = (time.time() - t0) / 3600
    print(f"[bench_pool] {sum(done)}/{len(done)} legs ok in {h * 60:.1f} min: {sum(done) / max(h, 1e-9):.1f} legs/h", flush=True)
    return 0 if all(done) else 1


if __name__ == "__main__":
    sys.exit(main())
