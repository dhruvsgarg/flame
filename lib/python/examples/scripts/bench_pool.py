#!/usr/bin/env python3
"""FX-N129 (V5): run fl_bench legs packed on GPUs; harness pool layout (POOL/<phase>/summary.tsv) for the analysis tools.

  bench_pool.py --from-pools F42B_POOLS... --out POOL [--seeds 1 2 3] [--only REGEX] [--gpus 0,2,3] [--per-gpu 4]
  bench_pool.py --jobs JOBS.tsv --out POOL ...        # JOBS.tsv: phase baseline trace yaml [key=value ...] (one leg per line)
  bench_pool.py --harness-args "--tier F42 --datasets cifar10 --phases '...'" [--harness-args ...] --out POOL ...   # the harness's own legs, now

Every leg runs `fl_bench.py` under a hard timeout; one Ctrl+C stops all (R20). Legs queue and start on the allowed GPU with the
most free memory once it and host RAM fit one more leg (`--leg-gpu-gb`, `--leg-ram-gb`, minus legs still ramping), ≤ `--per-gpu`. Seed i runs aggregator seed 1233 + i, the
harness's `--seeds` convention. Prints a progress line per finished leg and the pool's legs / h at the end.

FT-N19 (Felix + FluxTune in one queue): a leg's GPU / RAM / workers come from a jobs-file column (gpu_gb= ram_gb= workers=
runtime_s= project=), else its dataset's entry in `--profiles` (bench_profiles.yaml), else the --leg-* flags; the queue starts the
first leg that fits. A queue spanning both projects (FedFwd stack = fluxtune, else felix) writes POOL/<project>/<phase>/.
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
OPT_KEYS = {"gpu_gb": float, "ram_gb": float, "workers": int, "runtime_s": float, "project": str}


def parse_jobs(text):
    """JOBS.tsv lines -> (phase, baseline, trace, yaml, {key: value}) from the optional key=value columns."""
    out = []
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        cols = line.split("\t")
        opts = {}
        for c in cols[4:]:
            k, _, v = c.partition("=")
            if k not in OPT_KEYS:
                raise ValueError(f"jobs file: unknown column {c!r} (known: {sorted(OPT_KEYS)})")
            opts[k] = OPT_KEYS[k](v)
        out.append((*cols[:4], opts))
    return out


def leg_meta(yaml_path, _cache={}):
    """(dataset, project) of a leg: its YAML's dataset, and fluxtune when its baseline runs the FedFwd stack."""
    import yaml
    import fl_bench
    if "baselines" not in _cache:
        _cache["baselines"] = yaml.safe_load(open(HERE.parent / "_metadata" / "baselines.yaml"))["baselines"]
    exp = yaml.safe_load(open(yaml_path))["experiments"][0]
    main = ((_cache["baselines"].get(exp.get("baseline")) or {}).get("example") or {}).get("aggregator_main") or ""
    return exp["trainer"]["dataset"]["name"], "fluxtune" if fl_bench._is_fwdllm(main) else "felix"


def resources(job, profiles, a):
    """A leg's {gpu_gb, ram_gb, workers, runtime_s, project}: jobs-file column > dataset profile > flags."""
    dataset, project = leg_meta(job[3])
    prof = profiles.get(dataset) or {}
    r = dict(gpu_gb=a.leg_gpu_gb, ram_gb=a.leg_ram_gb, workers=a.workers, runtime_s=a.runtime_s, project=project)
    r.update({k: v for k, v in prof.items() if k in r}, **job[4])
    return r


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


def pick(groups, running, gfree, ram, a, now, need=None):
    """The group (GPUs of one leg) with the most free memory that fits a leg of `need` (gpu_gb, ram_gb; default the flags), or
    None. Legs younger than ramp_s count at their full size. A running entry is (p, ..., ts, grp[, gpu_gb, ram_gb])."""
    gpu_gb, ram_gb = need or (a.leg_gpu_gb, a.leg_ram_gb)
    size = lambda v: v[8:10] if len(v) >= 10 else (a.leg_gpu_gb, a.leg_ram_gb)
    young = [v for v in running.values() if now - v[6] < a.ramp_s]
    if ram - sum(size(v)[1] for v in young) < ram_gb:
        return None
    best = None
    for grp in groups:
        gs = grp.split(",")
        if sum(v[7] == grp for v in running.values()) >= a.per_gpu:
            continue
        ramping = sum(size(v)[0] for v in young if v[7] == grp)
        free = float("inf") if gfree is None else min(gfree.get(x, 0.0) - ramping for x in gs)
        if free - a.gpu_reserve_gb >= gpu_gb and (best is None or free > best[0]):
            best = (free, grp)
    return best and best[1]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-pools", nargs="*", default=[])
    ap.add_argument("--jobs")
    ap.add_argument("--harness-args", action="append", default=[], help="harness_pool args; legs from its current plan (dry-run YAMLs); repeatable")
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
    ap.add_argument("--profiles", default=str(HERE / "bench_profiles.yaml"), help="per-dataset leg resources ('' = flags only)")
    ap.add_argument("--first", default="", help="regex: matching legs start first (long legs, so the pool ends sooner)")
    ap.add_argument("--digest", action="store_true", help="FX-N153: leg_digest.py on each finished leg (digest, then gzip raw logs)")
    a = ap.parse_args(argv)
    jobs = [(*j, {}) for j in jobs_from_pools(a.from_pools)]
    if a.jobs:
        jobs += parse_jobs(open(a.jobs).read())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(HERE))
    import harness_pool as hp
    (out / "code_key.txt").write_text(hp.code_key() + "\n")  # FX-D162: paper_runs status keys legs on it
    for h in a.harness_args:
        jobs += [(*j, {}) for j in jobs_from_harness(h, out)]
    jobs = [j for j in jobs if re.search(a.only, " ".join(j[:3]))]
    import yaml
    profiles = (yaml.safe_load(open(a.profiles)) or {}) if a.profiles else {}
    jobs = [(*j[:4], resources(j, profiles, a)) for j in jobs]
    split = len({j[4]["project"] for j in jobs}) > 1  # FT-N19: per-project pools only when the queue mixes them
    queue = [(j, s) for j in jobs for s in a.seeds]
    if a.first:
        queue.sort(key=lambda q: not re.search(a.first, " ".join(q[0][:3])))  # stable: phase order otherwise
    g = a.gpus.split(",")
    groups = [",".join(g[i:i + a.gpus_per_leg]) for i in range(0, len(g) - a.gpus_per_leg + 1, a.gpus_per_leg)]
    running, done, t0, nid, probe_t, waited, digests = {}, [], time.time(), 0, 0.0, time.time(), []
    print(f"[bench_pool] {len(queue)} legs on GPUs {a.gpus} (≤ {a.per_gpu} / GPU, {a.leg_gpu_gb:g} GB GPU + {a.leg_ram_gb:g} GB RAM"
          f" per leg unless profiled), timeout {a.timeout_s:.0f}s", flush=True)
    for *key, y, r in sorted(jobs, key=lambda j: j[4]["project"]):
        print(f"[bench_pool]   {r['project']:8s} {r['gpu_gb']:g} GB GPU {r['ram_gb']:g} GB RAM {r['workers']} workers  {' '.join(key)}",
              flush=True)

    def kill(p, sig):
        try:
            os.killpg(p.pid, sig)
        except ProcessLookupError:
            pass

    def stop(*_):
        procs = [v[0] for v in running.values()] + digests
        for p in procs:
            kill(p, signal.SIGTERM)
        time.sleep(3)
        for p in procs:
            kill(p, signal.SIGKILL)
        sys.exit(130)
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    while queue or running:
        while queue and time.time() - probe_t >= 10:
            probe_t = time.time()
            gfree, ram = free_gb([x for grp in groups for x in grp.split(",")])
            grp, qi = None, 0
            for qi, ((*_, r), _s) in enumerate(queue):  # first queued leg that fits, so a large leg can't block small ones
                grp = pick(groups, running, gfree, ram, a, probe_t, (r["gpu_gb"], r["ram_gb"]))
                if grp is not None:
                    break
            if grp is None:
                if time.time() - waited > 300:
                    print(f"[bench_pool] waiting: {len(queue)} queued, {len(running)} running, no GPU / RAM fits a leg", flush=True)
                    waited = time.time()
                break
            probe_t, waited = 0.0, time.time()  # placed: probe again at once, the new leg counts as ramping
            (phase, b, trace, y, r), seed = queue.pop(qi)
            pool = out / r["project"] if split else out
            d = pool / phase / "runs" / f"{trace}_{b}_s{seed}"
            d.mkdir(parents=True, exist_ok=True)
            cmd = [PY, str(HERE / "fl_bench.py"), y, "--out", str(d), "--gpu", grp, "--seed", str(1233 + seed), "--workers", str(r["workers"])]
            if r["runtime_s"]:
                cmd += ["--runtime-s", str(r["runtime_s"])]
            log = open(d / "leg.out", "w")
            p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)  # own group: kill takes workers
            running[nid], nid = (p, phase, b, trace, seed, d, time.time(), grp, r["gpu_gb"], r["ram_gb"], pool), nid + 1
            print(f"[bench_pool] START {r['project'] + ' ' if split else ''}{phase} {b} {trace} s{seed} on GPU {grp}", flush=True)
        time.sleep(2)
        for slot, (p, phase, b, trace, seed, d, ts, _grp, _g, _r, pool) in list(running.items()):
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
                with open(pool / phase / "summary.tsv", "a") as f:
                    if f.tell() == 0:
                        f.write("baseline\ttrace\tseed\treal_dir\tsim_dir\n")
                    f.write(f"{b}\t{trace}\t{seed}\t\t{res['run_dir']}\n")
            if ok and a.digest:
                digests.append(subprocess.Popen([PY, str(HERE / "leg_digest.py"), res["run_dir"]], stdout=open(d / "digest.out", "w"),
                                                stderr=subprocess.STDOUT, start_new_session=True))
            print(f"[bench_pool] {'DONE' if ok else 'FAIL rc=' + str(p.returncode)} {phase} {b} {trace} s{seed} "
                  f"{(time.time() - ts) / 60:.1f} min {res and res.get('tasks')} tasks "
                  f"({len(done)}/{len(done) + len(queue) + len(running)})", flush=True)
    bad_digest = sum(p.wait() != 0 for p in digests)
    if bad_digest:
        print(f"[bench_pool] {bad_digest}/{len(digests)} digests FAILED (see <leg>/digest.out)", flush=True)
    h = (time.time() - t0) / 3600
    print(f"[bench_pool] {sum(done)}/{len(done)} legs ok in {h * 60:.1f} min: {sum(done) / max(h, 1e-9):.1f} legs/h", flush=True)
    return 0 if all(done) and not bad_digest else 1


if __name__ == "__main__":
    sys.exit(main())
