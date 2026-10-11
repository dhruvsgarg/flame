#!/usr/bin/env python3
"""PR28 overnight batch (jayne, 7 GPUs: GPU 1 has a logged uncorrected ECC error, so every pool gets --gpu-ids 0,2-7).

Lanes run concurrently and share the node through harness_pool leases (L28). Each lane runs its stages in order, so earlier stages win
leases first (priority = lane order, then stage order). The CPU lane is capped at 76 CPUs (--reserve-cores 26) so GPU legs always
find cores. A stage starts only if its longest leg fits before the hard stop (--hours); its pool gets a matching --deadline-h.
Claims, stages and coverage: FELIX_READINESS "PR28 overnight batch".

  run_pr28.py --plan                 # simulate the schedule: per-stage start/finish, CPU/GPU-min, what fits in --hours
  run_pr28.py [--hours 8] [--only C1,G1] [--dry-run]
Output: experiments/pr28_<ts>/{plan.txt, code_state.txt, <stage>.out, probes.txt, OVERNIGHT.txt}; pools experiments/pool_<ts>_PR28_<stage>.
One Ctrl+C stops everything: SIGINT to each pool's process group (pools tear their legs down), then SIGKILL after 120 s.
"""
import argparse
import glob
import json
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
EX = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))
import harness_pool as hp  # noqa: E402

GPU_IDS = "0,2,3,4,5,6,7"
CLIP = "trainerClipGradNorm=1.0"  # FX-D126 A/B arm (R1); off in baseline_reference.yaml until the operator decides
GDB = "/coc/scratch/dgarg/gdb_env/bin/gdb -q -batch -ex run -ex 'thread apply all bt' --args"
NON_OORT = "felix,fedbuff,refl,feddance"

# (id, lane, harness_pool args, extra env, core|stretch, claim). Lane order = priority order within a lane.
BATCH = "pr28"  # run_pr29.py reuses this runner with its own BATCH, STAGES, CLIP_STAGES
CLIP_STAGES = ("B1", "B2", "B3", "C4")
STAGES = [
    # CPU lane: stub/tiny_cpu tiers. P2 (syn_0/20/50) before P5 (mobiperf), per the 1-day plan.
    ("C1", "cpu", f"--tier T3 --datasets cifar10 --baselines oort,oort_star --traces syn_50 --trainer-hp {CLIP}", {}, "core",
     "R1 clip arm on stub syn_50s: 0 update_rejected, sim round ~ real; else clip is not the root"),
    ("C2", "cpu", "--tier T3 --datasets all --traces 'syn_0 syn_20 syn_50'", {}, "core",
     "P2: T3 syn_0/0b/20/50 x six x both on FX-D121-D126 code; INV/EXACT green, DIST within floor (FX-N80 s2, FX-N9)"),
    ("C3", "cpu", "--tier T4 --datasets all --phases 'P11a P11b P11c'", {}, "core",
     "checker sensitivity after FX-D125 widened A2/A3/A4dur: P11a-c CAUGHT both datasets (speech P11b miss known, FX-N32)"),
    ("C4", "cpu", "--tier T3C --datasets google_speech --baselines oort,oort_star,feddance --traces 'syn_20 syn_50'", {}, "core",
     "FX-N85 s2 floors: 3rd real leg per speech Oort syn_20/50 and feddance syn_50 cell (R3, R5); regrade --floors"),
    ("C5", "cpu", "--tier T4 --datasets all --phases P7", {}, "core",
     "FX-N79: P7 feddance KS < 0.2 on HEAD (2nd replicate = C6); P7 streaming tiny_cpu six x both"),
    ("C6", "cpu", "--tier T4 --datasets cifar10 --phases P7", {}, "core", "FX-N79 2nd P7 replicate"),
    ("C7", "cpu", "--tier T3 --datasets all --traces mobiperf_3st", {}, "core",
     "P5: T3 mobiperf x six x both (speech 4/6 gap; Oort family K=10, cifar only)"),
    ("C8", "cpu", "--tier T3C --datasets all --traces 'syn_0 syn_20 syn_50'", {}, "stretch",
     "Q2: a further real leg per T3 cell (floors >= 3 legs)"),
    # N33 lane: aggregator abort under gdb, 4 sim legs
    *[(f"N{i}", "n33", "--tier T1 --datasets cifar10 --baselines oort --traces syn_50 --no-gate",
       {"FLAME_AGG_CMD_PREFIX": GDB}, "core", "FX-N33: gdb backtrace of 'terminate called without an active exception'")
      for i in range(1, 5)],
    # GPU lane A: unavailability screens, floors, accuracy
    ("G1", "gpuA", f"--tier G0U --datasets all --baselines {NON_OORT},oort,oort_star --traces syn_50 --exclude-phases G0U_syn_50s",
     {}, "core", "P2 FX-N9: G0U syn_50, all six (cifar Oort in B1); A1-A8/K11 + INV/EXACT green"),
    ("G2", "gpuA", "--tier G0UC --datasets google_speech --baselines feddance,oort,oort_star --traces syn_50", {}, "core",
     "FX-N85 s2 / R3: n=50 real<->real floors for feddance U6 and Oort stall placement (FX-L53)"),
    ("G3", "gpuA", f"--tier G0U --datasets all --baselines {NON_OORT} --traces mobiperf_3st", {}, "core",
     "P5 FX-N9: G0U mobiperf felix/fedbuff/refl/feddance both datasets"),
    ("G4", "gpuA", "--tier G1AS --datasets google_speech --baselines refl,feddance,oort,oort_star", {}, "core",
     "FX-N74: speech accuracy on the reference clock (target 60%) or named round-bound"),
    ("G5", "gpuA", "--tier G1AS --datasets cifar10 --baselines refl,feddance,oort,oort_star", {}, "stretch",
     "FX-N74: cifar accuracy (target 50%); needs ~400 GB RAM, waits for the node to drain"),
    # GPU lane B: clip A/B, streaming
    ("B1", "gpuB", f"--tier G0U --datasets cifar10 --baselines oort,oort_star --traces syn_50 --trainer-hp {CLIP}", {}, "core",
     "R1/FX-N85 s1: cifar Oort G0U syn_50s with clip: 0 update_rejected, sim rounds ~ real (16 s)"),
    ("B2", "gpuB", f"--tier G0UC --datasets cifar10 --baselines oort,oort_star --traces syn_50 --trainer-hp {CLIP}", {}, "core",
     "R1 clip replicate 2 (0 update_rejected in 3 replicates)"),
    ("B3", "gpuB", f"--tier G0UC --datasets cifar10 --baselines oort,oort_star --traces syn_50 --trainer-hp {CLIP}", {}, "core",
     "R1 clip replicate 3"),
    ("B4", "gpuB", f"--tier G0T --datasets cifar10 --baselines {NON_OORT}", {}, "core",
     "P3 FX-N13: cifar G0T lin/events x syn_0/syn_50: EV19 + INV/EXACT green (Oort family needs a K=10 stream shape)"),
    ("B5", "gpuB", f"--tier G0U --datasets cifar10 --baselines oort,oort_star --traces mobiperf_3st --trainer-hp {CLIP}", {}, "core",
     "P5 + R1: cifar Oort G0U mobiperf_3sts (n=165) with clip"),
    ("B6", "gpuB", f"--tier G0To --datasets cifar10 --baselines {NON_OORT}", {}, "stretch", "P3: cifar oracle arms (G0To)"),
    ("B7", "gpuB", "--tier G0T --datasets google_speech --baselines refl,feddance", {}, "stretch", "FX-N30: speech G0T refl, feddance"),
]
LANE_POOL_ARGS = {"cpu": "--reserve-cores 26", "n33": "", "gpuA": "--gpu-cpus-per-trainer 0.1", "gpuB": "--gpu-cpus-per-trainer 0.1"}
LANE_CPU_CAP = {"cpu": 76}
NODE_CPUS, NODE_GPUS = 120, 7
GATE_S = 240  # a pool's gate (collect + smoke, cached per code state after the first)


def stage_jobs(args: str):
    """Build the stage's jobs exactly as harness_pool would (same filters, history estimates)."""
    ap = argparse.ArgumentParser()
    for f in ("--tier", "--datasets", "--baselines", "--phases", "--exclude-phases", "--traces", "--trainer-hp"):
        ap.add_argument(f, default="")
    ap.add_argument("--no-gate", action="store_true")
    a = ap.parse_args(shlex.split(args))
    bls = tuple(a.baselines.replace(",", " ").split()) or hp.B6
    dss = hp.DATASETS if a.datasets == "all" else tuple(a.datasets.split(","))
    ph = [p for d in dss for t in a.tier.split(",") for p in hp.tier_phases(t, bls, d)]
    if a.phases:
        w = set(a.phases.split())
        ph = [p for p in ph if p.pid in w or p.pid[len(hp.DS_TAG[p.dataset]):] in w]
    if a.exclude_phases:
        ph = [p for p in ph if p.pid not in set(a.exclude_phases.split())]
    if a.traces:
        ph = [p for p in ph if p.trace in a.traces.split()]
    ph = [p for p in ph if p.baselines]
    jobs = hp.build_jobs(ph, 1.0, 8, hp._load_history(), 0.1)
    for j in jobs:
        j.gpus = min(j.gpus, NODE_GPUS)
    return [j for j in jobs if j.mode != "grade"]


def plan(stages, hours: float) -> list:
    """List-schedule every lane on shared CPUs/GPUs/RAM; returns rows (stage, start_min, end_min or None, cpu_min, gpu_min)."""
    mem_cap = hp.mem_available_gb() - 32
    lanes = {}
    for s in stages:
        lanes.setdefault(s[1], []).append(s)
    jobs = {s[0]: sorted(stage_jobs(s[2]), key=lambda j: -j.est_s) for s in stages}
    t, horizon = 0.0, hours * 3600
    lane_i = {ln: 0 for ln in lanes}
    lane_ready = {ln: 0.0 for ln in lanes}  # when the lane's current stage may start launching legs
    started, pend, running, rows = {}, {}, [], {}
    while True:
        for ln, ss in lanes.items():  # advance lanes whose stage finished; skip stages whose longest leg no longer fits
            while lane_i[ln] < len(ss) and ss[lane_i[ln]][0] not in started and t >= lane_ready[ln]:
                sid = ss[lane_i[ln]][0]
                longest = max((j.est_s for j in jobs[sid]), default=0)
                if t + GATE_S + longest > horizon:
                    rows[sid] = (sid, None, None, 0, 0)
                    lane_i[ln] += 1
                    continue
                started[sid] = t
                pend[sid] = list(jobs[sid])
                lane_ready[ln] = t + GATE_S
        order = list(lanes)
        rot = int(t // 60) % len(order)  # pools poll leases independently: rotate which lane goes first
        for ln in order[rot:] + order[:rot]:
            i = lane_i[ln]
            if i >= len(lanes[ln]):
                continue
            sid = lanes[ln][i][0]
            if sid not in pend or t < lane_ready[ln]:
                continue
            for j in list(pend[sid]):
                cpu_used = sum(r[1] for r in running)
                lane_used = sum(r[1] for r in running if r[4] == ln)
                if (cpu_used + j.cpus > NODE_CPUS or lane_used + j.cpus > LANE_CPU_CAP.get(ln, NODE_CPUS)
                        or sum(r[2] for r in running) + j.gpus > NODE_GPUS or sum(r[3] for r in running) + j.mem_gb > mem_cap
                        or t + j.est_s > horizon):
                    continue
                running.append((t + j.est_s, j.cpus, j.gpus, j.mem_gb, ln, sid, j))
                pend[sid].remove(j)
        for ln in lanes:  # close stages with nothing pending or running
            i = lane_i[ln]
            if i < len(lanes[ln]):
                sid = lanes[ln][i][0]
                if sid in pend and not pend[sid] and not any(r[5] == sid for r in running):
                    js = jobs[sid]
                    rows[sid] = (sid, started[sid] / 60, t / 60, sum(j.cpus * j.est_s for j in js) / 60,
                                 sum(j.gpus * j.est_s for j in js) / 60)
                    lane_i[ln] += 1
                    lane_ready[ln] = t
        if all(lane_i[ln] >= len(lanes[ln]) for ln in lanes):
            break
        nxt = [r[0] for r in running] + [v for v in lane_ready.values() if v > t]
        if not nxt or min(nxt) > horizon:  # nothing can progress: unstarted legs past the horizon
            for ln in lanes:
                for s in lanes[ln][lane_i[ln]:]:
                    ran = [r for r in jobs[s[0]] if r not in pend.get(s[0], jobs[s[0]])]
                    rows.setdefault(s[0], (s[0], started[s[0]] / 60 if s[0] in started else None, None,
                                           sum(j.cpus * j.est_s for j in ran) / 60, sum(j.gpus * j.est_s for j in ran) / 60))
            break
        t = min(nxt)
        running = [r for r in running if r[0] > t]
    return [rows[s[0]] for s in stages if s[0] in rows]


def code_state(out: Path) -> None:
    git = lambda *a: subprocess.run(["git", "-C", str(hp.REPO), *a], capture_output=True, text=True).stdout
    diff = git("diff", "HEAD")
    import hashlib
    (out / "code_state.txt").write_text(f"HEAD {git('rev-parse', 'HEAD').strip()}\ndiff sha1 {hashlib.sha1(diff.encode()).hexdigest()}\n"
                                        f"{git('status', '--short')}")


def leg_runs(pool: Path):
    for f in glob.glob(f"{pool}/*/*/runs/*/legs.txt"):
        for ln in Path(f).read_text().split():
            yield ln


def probe(out: Path, pools: dict) -> list:
    """R24 early probes: PASS/FAIL/WAIT per claim from the legs' own telemetry."""
    lines = []
    clip_runs = [r for s in CLIP_STAGES if s in pools for r in leg_runs(pools[s])]
    if clip_runs:
        rej = nan = seen = 0
        for r in clip_runs:
            for f in glob.glob(f"{r}/telemetry/aggregator*.jsonl"):
                for ln in open(f, errors="replace"):
                    if '"update_rejected"' in ln:
                        rej += 1
            for f in glob.glob(f"{r}/*trainers.log"):
                with open(f, errors="replace") as fh:
                    seen += any("clip=1.0" in ln for ln in fh)
                with open(f, errors="replace") as fh:
                    nan += sum("nan" in ln.lower() and "loss" in ln.lower() for ln in fh)
        v = "FAIL" if rej or not seen else "PASS"
        lines.append(f"probe CLIP {v}: legs={len(clip_runs)} clip=1.0 seen in {seen} trainer logs (PL7), update_rejected={rej}, nan-loss lines={nan}")
    for sid, pool in pools.items():
        for bad in ("ABORT.txt", "STALLED.txt", "DOOMED.txt"):
            if (pool / bad).exists():
                lines.append(f"probe {sid} FAIL: {bad}: {(pool / bad).read_text()[:200].strip()}")
        plog = pool / "pool.log"
        if plog.exists():  # FX-D140: measured RAM over estimate or still climbing
            flagged = [ln.split()[3] for ln in plog.read_text(errors="replace").splitlines()
                       if " DONE " in f" {ln} " and ("RAM_OVER" in ln or "LEAK?" in ln) and len(ln.split()) > 3]
            if flagged:
                lines.append(f"probe {sid} RAM: RAM_OVER/LEAK? in {flagged[:8]} (resource_report.py {pool})")
        summ = pool / "SUMMARY.txt"
        if summ.exists():  # cell table = lines after the 'phase trace baseline ev_real ev_sim parity ...' header, to the first blank
            body = summ.read_text().splitlines()
            hdr = next((i for i, ln in enumerate(body) if ln.split()[:3] == ["phase", "trace", "baseline"]), None)
            rows = []
            for ln in body[hdr + 1:] if hdr is not None else []:
                if not ln.strip():
                    break
                rows.append(ln.split())
            fails = [f"{r[0]}:{r[2]}" for r in rows if any(x.startswith("FAIL") for x in r[3:6])]
            lines.append(f"probe {sid} {'PASS' if not fails else 'RED'}: {len(rows)} cells, red (EV or parity) {fails[:12]}")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=8.0)
    ap.add_argument("--only", default="", help="comma list of stage ids")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="pass --dry-run to every pool")
    a = ap.parse_args()
    stages = [s for s in STAGES if not a.only or s[0] in a.only.split(",")]
    rows = plan(stages, a.hours)
    hdr = f"{'stage':6s}{'lane':6s}{'kind':8s}{'start':>7s}{'end':>7s}{'cpu-min':>9s}{'gpu-min':>9s}  claim"
    txt = [hdr] + [f"{r[0]:6s}{s[1]:6s}{s[4]:8s}{('-' if r[1] is None else f'{r[1]:.0f}'):>7s}"
                   f"{(('SKIP' if r[1] is None else 'PART') if r[2] is None else f'{r[2]:.0f}'):>7s}{r[3]:9.0f}{r[4]:9.0f}  {s[5]}"
                   for r, s in zip(rows, stages)]
    fit = [r for r in rows if r[2] is not None]
    txt.append(f"fits in {a.hours:g} h: {len(fit)}/{len(rows)} stages; end ~{max((r[2] for r in fit), default=0) / 60:.1f} h; "
               f"CPU {sum(r[3] for r in fit):.0f}/{NODE_CPUS * a.hours * 60:.0f} CPU-min, GPU {sum(r[4] for r in fit):.0f}/"
               f"{NODE_GPUS * a.hours * 60:.0f} GPU-min (leases, not use)")
    print("\n".join(txt), flush=True)
    if a.plan:
        return 0

    ts = time.strftime("%Y%m%d_%H%M")
    out = EX / "experiments" / f"{BATCH}_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "plan.txt").write_text("\n".join(txt) + "\n")
    code_state(out)
    t0, hard = time.time(), time.time() + a.hours * 3600
    longest = {s[0]: max((j.est_s for j in stage_jobs(s[2])), default=0) for s in stages}
    stop, procs, pools, status, lock = threading.Event(), {}, {}, {}, threading.Lock()
    log = open(out / "OVERNIGHT.log", "a")

    def say(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    def lane(name, ss):
        time.sleep(0 if a.dry_run else {"gpuA": 0, "gpuB": 90, "n33": 180, "cpu": 270}.get(name, 330))  # stagger first pools: leases settle, gate runs once
        for sid, _, args, env, kind, claim in ss:
            if stop.is_set():
                return
            left = hard - time.time() - GATE_S - longest[sid]
            if left <= 0:
                status[sid] = "SKIP (no time)"
                say(f"{sid} SKIP: longest leg {longest[sid] / 60:.0f} min does not fit before the hard stop")
                continue
            pool = EX / "experiments" / f"pool_{ts}_{BATCH.upper()}_{sid}"
            cmd = [sys.executable, str(SCRIPTS / "harness_pool.py"), "--output-dir", str(pool), "--gpu-ids", GPU_IDS,
                   "--deadline-h", f"{left / 3600:.2f}"] + shlex.split(args) + shlex.split(LANE_POOL_ARGS[name]) \
                + (["--dry-run"] if a.dry_run else [])
            with lock:
                pools[sid] = pool
            say(f"{sid} START ({kind}) -> {pool.name}: {claim}")
            with open(out / f"{sid}.out", "w") as fh:
                fh.write(" ".join(cmd) + "\n")
                fh.flush()
                p = subprocess.Popen(cmd, cwd=str(EX), stdout=fh, stderr=subprocess.STDOUT, start_new_session=True,
                                     env={**os.environ, **env})
                with lock:
                    procs[sid] = p
                rc = p.wait()
            status[sid] = f"rc={rc} {(time.time() - t0) / 60:.0f} min"
            say(f"{sid} DONE rc={rc}")
            if rc == 130:
                return

    def on_signal(signum, _frm):
        stop.set()
        say(f"signal {signum}: stopping every pool")
        for p in list(procs.values()):
            if p.poll() is None:
                try:
                    os.killpg(p.pid, signal.SIGINT)
                except OSError:
                    pass

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    by_lane = {}
    for s in stages:
        by_lane.setdefault(s[1], []).append(s)
    threads = [threading.Thread(target=lane, args=(ln, ss), daemon=True) for ln, ss in by_lane.items()]
    for th in threads:
        th.start()
    last_probe = 0.0
    while any(th.is_alive() for th in threads):
        time.sleep(5)
        if stop.is_set():
            end = time.time() + 120
            while time.time() < end and any(p.poll() is None for p in procs.values()):
                time.sleep(2)
            for p in procs.values():
                if p.poll() is None:
                    try:
                        os.killpg(p.pid, signal.SIGKILL)
                    except OSError:
                        pass
            break
        if time.time() - last_probe > 600 and not a.dry_run:
            last_probe = time.time()
            with lock:
                lines = probe(out, dict(pools))
            with open(out / "probes.txt", "a") as fh:
                fh.write(f"--- {time.strftime('%H:%M')}\n" + "\n".join(lines) + "\n")
    interference = []
    for sid, pool in pools.items():
        tsv = pool / "jobs.tsv"
        if tsv.exists():
            for ln in tsv.read_text().splitlines()[1:]:
                f = ln.split("\t")
                if len(f) >= 8 and f[0] != "jid" and (float(f[7] or 0) > 0 or float(f[2] or 0) > 1.25 * float(f[3] or 1)):
                    interference.append(f"INTERFERENCE {sid}: {f[0]} dur={f[2]}s est={f[3]}s sat={f[7]} -> rerun solo")
    final = probe(out, pools) if not a.dry_run else []
    summary = ["stage status:"] + [f"  {s[0]:4s} {s[4]:8s} {status.get(s[0], 'not run'):24s} {pools.get(s[0], '')}" for s in stages] \
        + ["", "final probes:"] + final + ["", "interference:"] + (interference or ["  none"])
    (out / "OVERNIGHT.txt").write_text("\n".join(summary) + "\n")
    say(f"done: {out}/OVERNIGHT.txt")
    return 130 if stop.is_set() else 0


if __name__ == "__main__":
    sys.exit(main())
