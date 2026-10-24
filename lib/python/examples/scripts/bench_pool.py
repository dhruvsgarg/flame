#!/usr/bin/env python3
"""FX-N129 (V5): run fl_bench legs packed on GPUs; harness pool layout (POOL/<phase>/summary.tsv) for the analysis tools.

  bench_pool.py --from-pools F42B_POOLS... --out POOL [--seeds 1 2 3] [--only REGEX] [--gpus 0,2,3] [--per-gpu 4]
  bench_pool.py --jobs JOBS.tsv --out POOL ...        # JOBS.tsv: phase baseline trace yaml (one leg per line)
  bench_pool.py --harness-args "--tier F42 --datasets cifar10 --phases '...'" --out POOL ...   # the harness's own legs, now

Every leg runs `fl_bench.py` under a hard timeout; one Ctrl+C stops all (R20). Legs queue and start on the allowed GPU with the
most free memory once it and host RAM fit one more leg (`--leg-gpu-gb`, `--leg-ram-gb`, minus legs still ramping), ≤ `--per-gpu`. Seed i runs aggregator seed 1233 + i, the
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


def free_gb(gpus):
    """({gpu: free GB}, host MemAvailable GB); GPUs None when nvidia-smi fails (count-only placement)."""
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"], capture_output=True,
                           text=True, timeout=30, check=True).stdout
        g = {i.strip(): float(m) / 1024 for i, m in (line.split(",") for line in q.strip().splitlines())}
        g = {x: g[x] for x in gpus if x in g}
    except (OSError, subprocess.SubprocessError, ValueError):
        g = None
    ram = next((int(line.split()[1]) / 2**20 for line in open("/proc/meminfo") if line.startswith("MemAvailable")), float("inf"))
    return g, ram


def pick(groups, running, gfree, ram, a, now):
    """The group (GPUs of one leg) with the most free memory that fits a leg, or None. Legs younger than ramp_s count at full size."""
    young = [v[-1] for v in running.values() if now - v[6] < a.ramp_s]
    if ram - a.leg_ram_gb * len(young) < a.leg_ram_gb:
        return None
    best = None
    for grp in groups:
        gs = grp.split(",")
        if sum(v[-1] == grp for v in running.values()) >= a.per_gpu:
            continue
        free = float("inf") if gfree is None else min(gfree.get(x, 0.0) - a.leg_gpu_gb * sum(grp2 == grp for grp2 in young) for x in gs)
        if free - a.gpu_reserve_gb >= a.leg_gpu_gb and (best is None or free > best[0]):
            best = (free, grp)
    return best and best[1]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-pools", nargs="*", default=[])
    ap.add_argument("--jobs")
    ap.add_argument("--harness-args", default="", help="harness_pool args; legs from its current plan (dry-run YAMLs)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[1])
    ap.add_argument("--only", default="", help="regex on '<phase> <baseline> <trace>'")
    ap.add_argument("--gpus", default="0,2,3,4,5,6,7")
    ap.add_argument("--per-gpu", type=int, default=4, help="cap on legs per GPU group")
    ap.add_argument("--leg-gpu-gb", type=float, default=4.0, help="GPU memory one leg needs (cifar bench ~3.4 GB)")
    ap.add_argument("--leg-ram-gb", type=float, default=8.0, help="host RAM one leg needs")
    ap.add_argument("--gpu-reserve-gb", type=float, default=2.0, help="free GPU memory left for other users' jobs")
    ap.add_argument("--ramp-s", type=float, default=240, help="a new leg reaches full memory within this")
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
    running, done, t0, nid, probe_t, waited = {}, [], time.time(), 0, 0.0, time.time()
    print(f"[bench_pool] {len(queue)} legs on GPUs {a.gpus} (≤ {a.per_gpu} / GPU, {a.leg_gpu_gb:g} GB GPU + {a.leg_ram_gb:g} GB RAM"
          f" per leg), timeout {a.timeout_s:.0f}s", flush=True)

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
        while queue and time.time() - probe_t >= 10:
            probe_t = time.time()
            gfree, ram = free_gb([x for grp in groups for x in grp.split(",")])
            grp = pick(groups, running, gfree, ram, a, probe_t)
            if grp is None:
                if time.time() - waited > 300:
                    print(f"[bench_pool] waiting: {len(queue)} queued, {len(running)} running, no GPU / RAM fits a leg", flush=True)
                    waited = time.time()
                break
            probe_t, waited = 0.0, time.time()  # placed: probe again at once, the new leg counts as ramping
            (phase, b, trace, y), seed = queue.pop(0)
            d = out / phase / "runs" / f"{trace}_{b}_s{seed}"
            d.mkdir(parents=True, exist_ok=True)
            cmd = [PY, str(HERE / "fl_bench.py"), y, "--out", str(d), "--gpu", grp, "--seed", str(1233 + seed), "--workers", str(a.workers)]
            if a.runtime_s:
                cmd += ["--runtime-s", str(a.runtime_s)]
            log = open(d / "leg.out", "w")
            p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)  # own group: kill takes workers
            running[nid], nid = (p, phase, b, trace, seed, d, time.time(), grp), nid + 1
            print(f"[bench_pool] START {phase} {b} {trace} s{seed} on GPU {grp}", flush=True)
        time.sleep(2)
        for slot, (p, phase, b, trace, seed, d, ts, _grp) in list(running.items()):
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
