#!/usr/bin/env python3
# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""FX-N22 harness pool: run harness legs in parallel, isolated slots on one node (any example, any dataset).

A slot = disjoint physical cores (taskset) + a private mosquitto + a run tag (harness_suite.sh --isolate)
[+ exclusive GPUs]. A real+sim pair runs as two parallel legs plus a grade job. Jobs are packed
longest-first with backfill; the degree of parallelism follows from free cores, memory, GPUs and
--max-parallel. Durations are learned per job key (experiments/_harness_durations.json).

  harness_pool.py --tier T1 --changed origin/main     # sim-only delta smoke (EV only), affected baselines
  harness_pool.py --tier T2 --baselines 'felix fedbuff'  # sim-only matrix, parity vs banked real legs
  harness_pool.py --tier T3 --changed HEAD            # real+sim pairs; refreshes the real bank
  harness_pool.py --tier T4 [--phases 'P1 P2']        # the campaign phases
  ... --shard 1/2    this node's half (by estimated time)     ... --dry-run   print the plan only
Output: experiments/pool_<ts>/{SUMMARY.txt, pool.log, <phase>/summary.tsv, <phase>/<job>/...}.
"""

import argparse
import fnmatch
import json
import math
import os
import shlex
import signal
import socket
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
EXAMPLES = SCRIPT_DIR.parent                 # lib/python/examples
REPO = EXAMPLES.parents[2]                   # the git top level
OUT_DIR = EXAMPLES / "experiments"           # pool roots + per-node state (bank, durations)
DURATIONS = OUT_DIR / "_harness_durations.json"
# dataset -> the launcher example that runs it (one shared backprop-FL example today)
EXAMPLE_OF = {"cifar10": "async_cifar10", "google_speech": "async_cifar10"}


def example_dir(dataset: str) -> Path:
    return EXAMPLES / EXAMPLE_OF[dataset]

B6 = ("felix", "fedbuff", "oort", "oort_star", "refl", "feddance")
DATASETS = ("cifar10", "google_speech")
DS_TAG = {"cifar10": "", "google_speech": "gs_"}  # phase-id prefix; cifar keeps the campaign names
# CPU test shapes (FX-N22): small cohorts with varied c / aggGoal (= sync K via the launcher's agg_goal). A shape is one bank key, so T2 sim
# legs grade against the T3/T4 real legs of the same shape.
SHAPES = {
    "syn_0": dict(trace="syn_0", n=12, agg_goal=3, c=5, runtime_s=180, trace_scale=""),
    "syn_0b": dict(trace="syn_0", n=15, agg_goal=2, c=8, runtime_s=180, trace_scale=""),
    "syn_20": dict(trace="syn_20", n=15, agg_goal=3, c=6, runtime_s=240, trace_scale="4"),
    "syn_50": dict(trace="syn_50", n=15, agg_goal=3, c=6, runtime_s=240, trace_scale="4"),
    # FX-L34: ~10% AVL_TRAIN late in the scaled trace, so the cohort is larger and aggGoal/c smaller
    "mobiperf_3st": dict(trace="mobiperf_3st", n=30, agg_goal=2, c=4, runtime_s=240, trace_scale="4"),
}
GPU_N = {"cifar10": 300, "google_speech": 100}  # the datasets' reference cohorts (datasets.yaml)
STREAM_T = 'data_streaming={"enabled":"True","full_data_available_after_s":240}'
STREAM_A = STREAM_T + ' checkpoint={"enabled":"True","every_n_rounds":10}'


def oracle_a(dataset: str) -> str:
    return ('oracle_utility_injection={"enabled":"True","alpha":0.1,"num_trainers":%d,"sample_size":64}'
            % GPU_N[dataset])


# Changed path -> affected baselines (first match wins); real=True means the real path changed (T3 needed).
DELTA_RULES = (
    ("*.md", (), False),
    ("lib/python/flame/mode/horizontal/syncfl/trainer.py", B6, True),
    ("lib/python/flame/mode/horizontal/asyncfl/trainer.py", ("felix", "fedbuff"), True),
    ("lib/python/flame/mode/horizontal/asyncfl/*", ("felix", "fedbuff"), False),
    ("lib/python/flame/mode/horizontal/syncfl/top_aggregator.py", ("oort", "oort_star", "refl", "feddance"), False),
    ("lib/python/flame/mode/horizontal/syncfl/fwdllm_*", (), False),
    ("lib/python/flame/selector/async_oort.py", ("felix",), False),
    ("lib/python/flame/selector/fedbuff.py", ("fedbuff",), False),
    ("lib/python/flame/selector/async_base.py", ("felix", "fedbuff"), False),
    ("lib/python/flame/selector/scoring.py", ("felix", "oort", "oort_star", "refl"), False),
    ("lib/python/flame/selector/oort.py", ("oort", "oort_star", "refl"), False),
    ("lib/python/flame/selector/refl_oort.py", ("refl",), False),
    ("lib/python/flame/selector/feddance.py", ("feddance",), False),
    ("lib/python/flame/optimizer/fedbuff.py", ("felix", "fedbuff"), False),
    ("lib/python/flame/optimizer/fedavg.py", ("oort", "oort_star", "feddance"), False),
    ("lib/python/flame/optimizer/refl.py", ("refl",), False),
    ("lib/python/flame/availability/feddance_predictor.py", ("feddance",), False),
    ("lib/python/flame/availability/refl_tracker.py", ("refl",), False),
    ("lib/python/flame/channel*.py", B6, True),
    ("lib/python/flame/backend/*", B6, True),
    ("lib/python/flame/*", B6, False),
    ("lib/python/examples/async_cifar10/trainer/*", B6, True),
    ("lib/python/examples/async_cifar10/aggregator/pytorch/main_asyncfl_agg.py", ("felix", "fedbuff"), False),
    ("lib/python/examples/async_cifar10/aggregator/pytorch/main_oort_sync_agg.py", ("oort", "oort_star", "refl"), False),
    ("lib/python/examples/async_cifar10/aggregator/pytorch/main_fedavg_agg.py", ("feddance",), False),
    ("lib/python/examples/async_cifar10/aggregator/*", B6, False),
    ("lib/python/examples/async_cifar10/scripts/*", B6, False),
    ("lib/python/examples/scripts/expt_runner.sh", B6, False),
    ("lib/python/examples/async_cifar10/expt_scripts_2026/felix_oort_refl_feddance_alpha0.1_parity.yaml", B6, False),
    ("lib/python/examples/_metadata/baselines.yaml", B6, False),
    ("lib/python/examples/_metadata/datasets.yaml", B6, False),
    ("lib/python/examples/_metadata/dataset_splits/*", B6, False),
    ("lib/python/examples/async_cifar10/fl_data.py", B6, True),
)


def affected(paths) -> tuple:
    """(baselines in B6 order, real_path_changed) for a list of repo-relative changed paths."""
    hit, real = set(), False
    for p in paths:
        for pat, bls, is_real in DELTA_RULES:
            if fnmatch.fnmatch(p, pat):
                hit |= set(bls)
                real |= is_real and bool(bls)
                break
    return tuple(b for b in B6 if b in hit), real


def changed_paths(ref: str) -> List[str]:
    """Tracked diff vs `ref` (committed + working tree) plus untracked files."""
    run = lambda *a: subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True).stdout.split()
    return sorted(set(run("diff", "--name-only", ref)) | set(run("ls-files", "--others", "--exclude-standard")))


# ---------------------------------------------------------------- phases and jobs
@dataclass
class Phase:
    pid: str
    baselines: tuple
    trace: str
    runtime_s: int
    n: int
    dataset: str = "cifar10"
    harness: str = "stub"
    trace_scale: str = ""
    agg_goal: Optional[int] = None  # None = the config's (GPU legs keep the reference knobs)
    c: Optional[int] = None
    sim_ceiling_x: int = 1  # sim wall ceiling = x * runtime (speech stub sims are aggregator-bound on CPU)
    agg_hp: str = ""
    trainer_hp: str = ""
    inject_bug: str = ""
    kind: str = "pair"  # pair: real+sim legs + grade | sim: sim leg vs banked real | sim_ev: sim leg, EV only
    cpt: Optional[float] = None  # CPUs per trainer for this phase; None = --cpus-per-trainer


def shaped(pid, baselines, shape, kind, dataset, **kw) -> Phase:
    sh = dict(SHAPES[shape])
    if dataset == "google_speech":
        sh["sim_ceiling_x"] = 2
    sh.update(kw)
    return Phase(DS_TAG[dataset] + pid, tuple(baselines), kind=kind, dataset=dataset, **sh)


def campaign_phases(ds: str) -> List[Phase]:
    ph = lambda pid, bls, shape, **kw: shaped(pid, bls, shape, "pair", ds, **kw)
    return [
        ph("P1", B6, "syn_0"), ph("P1b", B6, "syn_0b"), ph("P2", B6, "syn_50"), ph("P3", B6, "mobiperf_3st"),
        ph("P4", ("felix", "fedbuff"), "syn_0", agg_hp="simColdStartGate=false"),
        ph("P5", ("felix", "refl"), "syn_20"),
        ph("P6", ("felix", "oort"), "syn_0", harness="tiny_cpu"),
        ph("P7", B6, "syn_0", harness="tiny_cpu", trainer_hp=STREAM_T, agg_hp=STREAM_A),
        ph("P7o", B6, "syn_0", harness="tiny_cpu", trainer_hp=STREAM_T, agg_hp=STREAM_A + " " + oracle_a(ds)),
        ph("P8", ("fedbuff",), "syn_50", agg_hp="taskRetryPolicy=exponential taskRetryBackoffSeconds=10"),
        ph("P9", ("felix", "fedbuff"), "syn_0", agg_hp="real_drain_ready_ingest=true"),
        # FX-N18 control: fedbuff with the old 0.1s real-only settle sleep (default is now 0)
        ph("P10", ("fedbuff",), "syn_0", agg_hp="realDistributeSettleSeconds=0.1"),
    ] + injected_phases("pair", ds)


def injected_phases(kind, ds) -> List[Phase]:
    """S1: each injected bug must FAIL its sim rung (EV10 / EV16 / EV3)."""
    return [shaped(pid, ("felix",), "syn_50", kind, ds, inject_bug=bug)
            for pid, bug in (("P11a", "no_busy_hold"), ("P11b", "order_by_sct"), ("P11c", "freeze_trainer_clock"))]


def tier_phases(tier: str, baselines: tuple, ds: str = "cifar10") -> List[Phase]:
    matrix = ("syn_0", "syn_0b", "syn_50", "mobiperf_3st")
    if tier == "T1":
        return [shaped("T1", baselines, sh, "sim_ev", ds, runtime_s=120) for sh in ("syn_0", "syn_50")]
    if tier == "T2":
        return [shaped(f"T2_{sh}", baselines, sh, "sim", ds) for sh in matrix] + injected_phases("sim_ev", ds)
    if tier == "T3":
        return [shaped(f"T3_{sh}", baselines, sh, "pair", ds) for sh in matrix]
    if tier == "T4":
        return campaign_phases(ds)
    if tier == "G1":  # GPU block on the production path at the dataset's reference config (FX-N4)
        bls = tuple(b for b in baselines if b in ("felix", "fedbuff")) if baselines != B6 else ("felix", "fedbuff")
        return [Phase(DS_TAG[ds] + "G1", bls, "syn_0", runtime_s=5400, n=GPU_N[ds], dataset=ds, harness="none")]
    if tier == "G2":  # the other four (FX-N5)
        bls = tuple(b for b in baselines if b not in ("felix", "fedbuff")) or B6[2:]
        return [Phase(DS_TAG[ds] + "G2", bls, "syn_0", runtime_s=5400, n=GPU_N[ds], dataset=ds, harness="none")]
    if tier in ("ISO", "ISO_FILL"):  # P6: felix/fedbuff pairs solo (--max-parallel 1), then packed
        iso = [shaped("ISO", ("felix", "fedbuff"), "syn_0", "pair", ds)]
        # packed stage: neighbours load the node (and double as FX-N20 syn_50 checks)
        return iso + ([shaped("FILL", B6[2:], "syn_50", "sim_ev", ds)] if tier == "ISO_FILL" else [])
    raise SystemExit(f"unknown tier {tier}")


def agg_cores(n: int) -> int:
    """Aggregator cores for a slot: a solo n>=60 run gets 8 (runner CPU partition); small cohorts less."""
    return 2 if n <= 30 else 4 if n <= 60 else 8


def slot_cpus(n: int, per_trainer: float = 1.0, agg: Optional[int] = None) -> int:
    """Even logical-CPU count: the aggregator's cores + `per_trainer` CPUs per trainer
    (< 1 packs trainers onto shared CPUs; the spawner wraps them round-robin)."""
    s = (agg_cores(n) if agg is None else agg) + math.ceil(n * per_trainer)
    return s + (s % 2)


@dataclass
class Job:
    jid: str
    phase: str
    trace: str
    baseline: str
    mode: str  # real | sim | grade
    args: List[str]
    n: int
    runtime_s: int
    harness: str
    cpus: int
    mem_gb: float
    gpus: int = 0
    deps: tuple = ()
    est_s: float = 0.0
    unit: str = ""  # jobs that must land on the same shard (a pair and its grade)
    whole: bool = False  # takes the whole node: no taskset, no FLAME_AGG_CORES (= a solo run's layout)
    dataset: str = "cifar10"

    @property
    def key(self) -> str:
        return f"{self.dataset}|{self.harness}|{self.mode}|{self.trace}|{self.baseline}|n{self.n}|rt{self.runtime_s}"


def estimate_s(job: Job, history: dict) -> float:
    past = history.get(job.key, [])
    if past:
        return float(statistics.median(past[-3:]))
    if job.mode == "grade":
        return 90.0
    join = job.n / 4 + 60 + 30  # join + startup/teardown + events
    return (job.runtime_s if job.mode == "real" else job.runtime_s / 1.6) + join


def build_jobs(phases: List[Phase], per_trainer: float, gpus_per_job: int, history: dict,
               gpu_per_trainer: float = 0.4) -> List[Job]:
    """gpu_per_trainer: CPUs per GPU-leg trainer (0.4 = the historical n=300 on 128 CPUs -> whole node)."""
    jobs: List[Job] = []
    for ph in phases:
        t = ph.trace
        for b in ph.baselines:
            base = ["--harness", ph.harness, "--baselines", b, "--traces", t, "--runtime-s", str(ph.runtime_s),
                    "--num-trainers", str(ph.n), "--dataset", ph.dataset]
            for flag, v in (("--trace-scale", ph.trace_scale), ("--agg-goal", ph.agg_goal),
                            ("--concurrency", ph.c), ("--agg-hp", ph.agg_hp),
                            ("--trainer-hp", ph.trainer_hp), ("--inject-bug", ph.inject_bug)):
                if v:
                    base += [flag, str(v)]
            if ph.sim_ceiling_x != 1:
                base += ["--sim-ceiling-x", str(ph.sim_ceiling_x)]
            gpu = ph.harness == "none"
            cpus = slot_cpus(ph.n, gpu_per_trainer, 8) if gpu else slot_cpus(ph.n, ph.cpt or per_trainer)
            mem = (1.0 if gpu else 0.6) * ph.n + 3  # ~0.55 GB per stub trainer (n=120 legs peak ~68 GB)
            g = gpus_per_job if gpu else 0
            stem = f"{ph.pid}_{b}" if ph.pid.endswith(t) else f"{ph.pid}_{t}_{b}"
            mk = lambda mode, args, deps=(), c=cpus, m=mem, gg=g: Job(
                f"{stem}_{mode}", ph.pid, t, b, mode, args, ph.n, ph.runtime_s, ph.harness, c, m, gg, deps,
                unit=stem, dataset=ph.dataset)
            if ph.kind == "pair":
                real, sim = mk("real", base + ["--mode", "real"]), mk("sim", base + ["--mode", "sim"])
                jobs += [real, sim, mk("grade", base, (real.jid, sim.jid), 2, 1.0, 0)]
            elif ph.kind == "sim":
                jobs.append(mk("sim", base + ["--mode", "sim", "--real-from", "auto"]))
            else:
                jobs.append(mk("sim", base + ["--mode", "sim"]))
    for j in jobs:
        j.est_s = estimate_s(j, history)
    return jobs


def shard(jobs: List[Job], i: int, n: int) -> List[Job]:
    """Units (a pair + its grade stay together) dealt longest-first to the least-loaded shard."""
    units: Dict[str, List[Job]] = {}
    for j in jobs:
        units.setdefault(j.unit, []).append(j)
    load, pick = [0.0] * n, []
    for u in sorted(units.values(), key=lambda js: (-sum(j.est_s for j in js), js[0].jid)):
        k = load.index(min(load))
        load[k] += sum(j.est_s for j in u)
        if k == i - 1:
            pick += u
    return [j for j in jobs if j in pick]


# ---------------------------------------------------------------- node resources
def core_groups() -> Dict[int, List[List[int]]]:
    """{numa node: [[cpu, sibling], ...]} over this process's affinity (whole physical cores)."""
    aff = set(os.sched_getaffinity(0))
    sysd = Path("/sys/devices/system")
    node_of = {}
    for nd in sorted(sysd.glob("node/node[0-9]*")):
        for c in _cpulist((nd / "cpulist").read_text()):
            node_of[c] = int(nd.name[4:])
    seen, out = set(), {}
    for c in sorted(aff):
        if c in seen:
            continue
        sib_f = sysd / f"cpu/cpu{c}/topology/thread_siblings_list"
        sib = [s for s in (_cpulist(sib_f.read_text()) if sib_f.exists() else [c]) if s in aff]
        seen |= set(sib)
        out.setdefault(node_of.get(c, 0), []).append(sorted(sib))
    return out


def _cpulist(text: str) -> List[int]:
    out = []
    for part in text.strip().split(","):
        if "-" in part:
            a, b = part.split("-")
            out += range(int(a), int(b) + 1)
        elif part:
            out.append(int(part))
    return out


class CoreMap:
    """Whole-physical-core allocator; a slot stays on one NUMA node when it fits (best fit)."""

    def __init__(self, groups: Dict[int, List[List[int]]], reserve_cores: int):
        self.free = {nd: list(gs) for nd, gs in groups.items()}
        for _ in range(reserve_cores):  # keep the lowest cores for the pool, checkers and the OS
            nd = max(self.free, key=lambda k: len(self.free[k]))
            self.free[nd].pop(0)

    def total(self) -> int:
        return sum(len(g) for gs in self.free.values() for g in gs)

    def alloc(self, cpus: int) -> Optional[List[int]]:
        fits = [nd for nd, gs in self.free.items() if sum(len(g) for g in gs) >= cpus]
        order = sorted(fits, key=lambda nd: sum(len(g) for g in self.free[nd])) or \
            sorted(self.free, key=lambda nd: -sum(len(g) for g in self.free[nd]))
        if sum(len(g) for gs in self.free.values() for g in gs) < cpus:
            return None
        got: List[int] = []
        for nd in order:
            while self.free[nd] and len(got) < cpus:
                got += self.free[nd].pop(0)
            if len(got) >= cpus:
                break
        return sorted(got)

    def release(self, cpus: List[int], groups: Dict[int, List[List[int]]]) -> None:
        cs = set(cpus)
        for nd, gs in groups.items():
            for g in gs:
                if set(g) <= cs:
                    self.free[nd].append(g)
            self.free[nd].sort()


def mem_available_gb() -> float:
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1e6
    return 0.0


def gpu_ids(healthy_only: bool = True) -> List[int]:
    """Node GPU ordinals; by default without any GPU showing a volatile uncorrected ECC error
    (the runner's health check refuses those, e.g. jayne GPU 1 on 2026-09-26)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,ecc.errors.uncorrected.volatile.total",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return []
    ids = []
    for line in out.strip().splitlines():
        idx, ecc = (x.strip() for x in line.split(","))
        if healthy_only and ecc.isdigit() and int(ecc) > 0:
            continue
        ids.append(int(idx))
    return ids


def free_port(start: int, used: set) -> int:
    p = start
    while True:
        if p not in used:
            with socket.socket() as s:
                try:
                    s.bind(("127.0.0.1", p))
                    return p
                except OSError:
                    pass
        p += 1


def tag_pids(tag: str) -> List[int]:
    """Every process of this user carrying FLAME_RUN_TAG=<tag>."""
    out, uid, needle = [], os.getuid(), f"FLAME_RUN_TAG={tag}".encode()
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            if d.stat().st_uid != uid:
                continue
            if needle in (d / "environ").read_bytes().split(b"\0"):
                out.append(int(d.name))
        except OSError:
            continue
    return out


def tag_cpu_s(tag: str) -> Dict[int, float]:
    """{pid: user+sys CPU seconds} for this tag's live processes."""
    tick = os.sysconf("SC_CLK_TCK")
    out = {}
    for pid in tag_pids(tag):
        try:
            f = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()
            out[pid] = (int(f[11]) + int(f[12])) / tick
        except (OSError, IndexError, ValueError):
            continue
    return out


def kill_tag(tag: str, sig=signal.SIGKILL) -> None:
    for pid in tag_pids(tag):
        try:
            os.kill(pid, sig)
        except OSError:
            pass


# ---------------------------------------------------------------- run
@dataclass
class Running:
    job: Job
    proc: subprocess.Popen
    cpus: List[int]
    gpus: List[int]
    port: int
    tag: str
    t0: float
    out: Path
    cpu: Dict[int, float] = field(default_factory=dict)  # pid -> last seen CPU seconds
    busy: List[float] = field(default_factory=list)      # cores busy per sample interval
    last: tuple = (0.0, 0.0)                              # (time, total CPU s) at the last sample

    def sample(self) -> None:
        self.cpu.update(tag_cpu_s(self.tag))
        now, tot = time.time(), sum(self.cpu.values())
        if self.last[0]:
            self.busy.append((tot - self.last[1]) / max(1e-6, now - self.last[0]))
        self.last = (now, tot)


@dataclass
class Pool:
    root: Path
    jobs: List[Job]
    max_parallel: int
    reserve_cores: int
    mem_headroom_gb: float
    deadline_s: float
    dry: bool
    log: object = None
    done: Dict[str, Path] = field(default_factory=dict)
    bad_gpus: List[int] = field(default_factory=list)

    def say(self, msg: str) -> None:
        line = f"[{time.strftime('%F %T')}] {msg}"
        print(line, flush=True)
        if self.log:
            self.log.write(line + "\n")
            self.log.flush()

    def plan(self, groups) -> None:
        cm = CoreMap(groups, self.reserve_cores)
        self.say(f"node: {cm.total()} cpus for slots (reserve {self.reserve_cores} cores), "
                 f"{mem_available_gb():.0f} GB available, gpus {gpu_ids() or '-'}")
        tot = sum(j.est_s for j in self.jobs)
        self.say(f"{len(self.jobs)} jobs, {tot / 60:.0f} slot-min total")
        for j in sorted(self.jobs, key=lambda j: -j.est_s):
            self.say(f"  {j.jid:<44} cpus={j.cpus:<3} mem={j.mem_gb:<5.0f} est={j.est_s / 60:5.1f}m deps={list(j.deps)}")
        self.say(f"estimated makespan ~{simulate_makespan(self.jobs, cm.total(), self.max_parallel) / 60:.0f} min")

    def run(self, groups) -> int:
        cm = CoreMap(groups, self.reserve_cores)
        gpus_free = gpu_ids()
        pending = sorted(self.jobs, key=_order)
        running: Dict[str, Running] = {}
        ports: set = set()
        history = _load_history()
        t0 = self.t0 = time.time()
        mem_budget = mem_available_gb() - self.mem_headroom_gb
        stop = {"flag": False}
        tick = 0
        jobs_tsv = open(self.root / "jobs.tsv", "a")
        jobs_tsv.write("jid\trc\tdur_s\test_s\tcpus\tcores_avg\tcores_p95\n")

        def _on_signal(signum, _frm):
            stop["flag"] = True

        signal.signal(signal.SIGINT, _on_signal)
        signal.signal(signal.SIGTERM, _on_signal)
        try:
            while (pending or running) and not stop["flag"]:
                for j in list(pending):
                    if any(d not in self.done for d in j.deps):
                        continue
                    if time.time() - t0 > self.deadline_s:
                        self.say(f"SKIP {j.jid}: past the deadline")
                        pending.remove(j)
                        self.done[j.jid] = None
                        continue
                    if len(running) >= self.max_parallel:
                        break
                    if j.mem_gb + sum(r.job.mem_gb for r in running.values()) > mem_budget and running:
                        continue
                    if j.gpus > len(gpus_free):
                        continue
                    cpus = cm.alloc(j.cpus)
                    if cpus is None:
                        continue
                    gp = [gpus_free.pop(0) for _ in range(j.gpus)]
                    port = free_port(18830, ports)
                    ports.add(port)
                    running[j.jid] = self._launch(j, cpus, gp, port)
                    pending.remove(j)
                time.sleep(2)
                tick += 1
                for jid, r in list(running.items()):
                    rc = r.proc.poll()
                    if rc is None:
                        if tick % 3 == 0:
                            r.sample()
                        continue
                    kill_tag(r.tag)  # leftovers of a leg that died hard
                    dur = time.time() - r.t0
                    avg = sum(r.cpu.values()) / max(1.0, dur)
                    p95 = sorted(r.busy)[int(0.95 * (len(r.busy) - 1))] if r.busy else 0.0
                    self.say(f"DONE  {jid} rc={rc} {dur / 60:.1f}m (est {r.job.est_s / 60:.1f}m) "
                             f"cores avg {avg:.1f} p95 {p95:.1f} of {len(r.cpus)}")
                    jobs_tsv.write(f"{jid}\t{rc}\t{dur:.0f}\t{r.job.est_s:.0f}\t{len(r.cpus)}\t{avg:.2f}\t{p95:.2f}\n")
                    jobs_tsv.flush()
                    row = _summary_row(r.out)
                    if rc == 0 and (r.job.mode == "grade" or row.get(f"{r.job.mode}_dir")):
                        history.setdefault(r.job.key, []).append(round(dur))
                    cm.release(r.cpus, groups)
                    gpus_free += r.gpus
                    ports.discard(r.port)
                    self.done[jid] = r.out
                    del running[jid]
        finally:
            if running:
                self.say(f"INTERRUPT — tearing down {len(running)} slot(s)")
                for r in running.values():
                    try:
                        os.killpg(r.proc.pid, signal.SIGTERM)
                    except OSError:
                        pass
                deadline = time.time() + 45
                while time.time() < deadline and any(r.proc.poll() is None for r in running.values()):
                    time.sleep(1)
                for r in running.values():
                    try:
                        os.killpg(r.proc.pid, signal.SIGKILL)
                    except OSError:
                        pass
                    kill_tag(r.tag)
            _save_history(history)
        if stop["flag"]:
            return 130
        self.merge()
        return 0

    def _launch(self, j: Job, cpus: List[int], gp: List[int], port: int) -> Running:
        out = self.root / j.phase / j.jid
        out.mkdir(parents=True, exist_ok=True)
        tag = f"pool_{self.root.name}_{j.jid}"
        args = list(j.args)
        if j.mode == "grade":
            legs = [_summary_row(self.done.get(d)) for d in j.deps]
            real = next((r.get("real_dir", "") for r in legs if r.get("real_dir")), "")
            sim = next((r.get("sim_dir", "") for r in legs if r.get("sim_dir")), "")
            args += ["--grade-only", "--real-dir", real, "--sim-dir", sim]
        else:
            args += ["--isolate", "--broker-port", str(port), "--run-tag", tag]
        if gp and (not j.whole or self.bad_gpus):
            args += ["--gpu-ids", ",".join(map(str, gp))]
        pin = [] if j.whole else ["taskset", "-c", ",".join(map(str, cpus))]
        suite = example_dir(j.dataset) / "scripts" / "harness_suite.sh"
        cmd = [*pin, "bash", str(suite), *args, "--output-dir", str(out)]
        (out / "cmd.txt").write_text(shlex.join(cmd) + "\n")
        env = {**os.environ, "EXPT_AUTOCLEAN": "1", "FLAME_RUN_TAG": tag}
        if not j.whole:
            env["FLAME_AGG_CORES"] = str(8 if j.gpus else agg_cores(j.n))
        proc = subprocess.Popen(cmd, stdout=open(out / "suite.log", "w"), stderr=subprocess.STDOUT,
                                env=env, start_new_session=True, cwd=str(example_dir(j.dataset)))
        self.say(f"START {j.jid} cpus={_ranges(cpus)} port={port}{' gpus=' + str(gp) if gp else ''} est={j.est_s / 60:.1f}m")
        return Running(j, proc, cpus, gp, port, tag, time.time(), out)

    def merge(self) -> None:
        """<phase>/summary.tsv: the grade row of a pair, else the sim row (harness_suite columns)."""
        by_phase: Dict[str, List[Job]] = {}
        for j in self.jobs:
            by_phase.setdefault(j.phase, []).append(j)
        for ph, js in by_phase.items():
            rows, header = [], None
            graded = {d for j in js if j.mode == "grade" for d in j.deps}
            for j in sorted(js, key=lambda j: j.jid):
                if j.jid in graded:
                    continue
                f = (self.done.get(j.jid) or Path("/nonexistent")) / "summary.tsv"
                if not f.exists():
                    rows.append("\t".join([j.trace, j.baseline] + ["MISSING"] * 3 + [""] * 8))
                    continue
                lines = f.read_text().splitlines()
                header = header or lines[0]
                rows += lines[1:]
            header = header or "\t".join(["trace", "baseline", "ev_real", "ev_sim", "parity", "score", "n_fail",
                                          "roots", "crash_lines", "timeout", "real_dir", "sim_dir", "real_src"])
            (self.root / ph / "summary.tsv").write_text("\n".join([header] + rows) + "\n")


def _summary_row(d: Optional[Path]) -> dict:
    f = (d or Path("/nonexistent")) / "summary.tsv"
    if not f.exists():
        return {}
    lines = f.read_text().splitlines()
    if len(lines) < 2:
        return {}
    return dict(zip(lines[0].split("\t"), lines[1].split("\t")))


def _ranges(cpus: List[int]) -> str:
    out, start, prev = [], None, None
    for c in sorted(cpus) + [None]:
        if start is None:
            start = prev = c
        elif c is not None and c == prev + 1:
            prev = c
        else:
            out.append(f"{start}-{prev}" if prev != start else f"{start}")
            start = prev = c
    return ",".join(out)


def _order(j: Job) -> tuple:
    """Longest first, but whole-node jobs last: they block every slot while they run."""
    return (j.whole, -j.est_s, j.jid)


def simulate_makespan(jobs: List[Job], cpus: int, max_parallel: int) -> float:
    """List-scheduling estimate of wall time (cores only; deps respected)."""
    pending = sorted(jobs, key=_order)
    done_at: Dict[str, float] = {}
    running: List[tuple] = []  # (end, cpus, jid)
    now, free = 0.0, cpus
    while pending or running:
        for j in list(pending):
            if any(done_at.get(d, math.inf) > now for d in j.deps) or len(running) >= max_parallel:
                continue
            if j.cpus <= free:
                running.append((now + j.est_s, j.cpus, j.jid))
                free -= j.cpus
                pending.remove(j)
        if not running:
            if pending and all(j.cpus > cpus for j in pending):
                return math.inf
            now += 1
            continue
        running.sort()
        end, c, jid = running.pop(0)
        now, free = end, free + c
        done_at[jid] = end
    return now


def _load_history() -> dict:
    try:
        return json.loads(DURATIONS.read_text())
    except Exception:
        return {}


def _save_history(h: dict) -> None:
    DURATIONS.parent.mkdir(parents=True, exist_ok=True)
    DURATIONS.write_text(json.dumps({k: v[-10:] for k, v in h.items()}, indent=1, sort_keys=True))


def run_gate(root: Path, datasets, pool: "Pool") -> int:
    """R19: collect every test, check each dataset is complete, run one felix smoke pair per dataset;
    nonzero = abort (4)."""
    lib = REPO / "lib" / "python"
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "--collect-only", "-p", "no:cacheprovider", "tests",
                        "examples/async_cifar10/scripts/parity", "examples/async_cifar10/trainer/pytorch",
                        "examples/fwdllm/expt_scripts"], cwd=str(lib), capture_output=True, text=True, timeout=300)
    (root / "P00_collect.txt").write_text(r.stdout + r.stderr)
    if r.returncode:
        pool.say(f"ABORT gate: pytest collection failed -- {root}/P00_collect.txt")
        return 4
    r = subprocess.run([sys.executable, "-c", "import sys; sys.path.insert(0, sys.argv[1]); import fl_data; "
                        "[print(fl_data.verify(d)) for d in sys.argv[2:]]", str(example_dir(datasets[0])), *datasets],
                       capture_output=True, text=True, timeout=600)
    (root / "P00_data.txt").write_text(r.stdout + r.stderr)
    if r.returncode:
        pool.say(f"ABORT gate: dataset missing or incomplete -- {root}/P00_data.txt")
        return 4
    pool.say("data ok: " + " | ".join(r.stdout.split("\n")[:-1]))
    g = root / "P00"
    r = subprocess.run([sys.executable, __file__, "--tier", "T3", "--baselines", "felix", "--phases", "T3_syn_0",
                        "--smoke", "--no-gate", "--datasets", ",".join(datasets), "--output-dir", str(g)],
                       capture_output=True, text=True, timeout=900)
    leg_bad = lambda v: v == "MISSING" or "EV1" in v.replace("FAIL:", "").split(",")
    bad = [row for f in g.glob("*/summary.tsv") for row in f.read_text().splitlines()[1:]
           if leg_bad(row.split("\t")[2]) or leg_bad(row.split("\t")[3])
           or row.split("\t")[4] in ("MISSING", "CHECKER_ERROR")]
    if r.returncode or bad or not list(g.glob("*/summary.tsv")):
        pool.say(f"ABORT gate: smoke pair made no progress -- {g}")
        return 4
    pool.say("gate ok (collect + smoke pair per dataset)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tier", required=True,
                    help="comma list of T1 T2 T3 T4 G1 G2 ISO ISO_FILL, e.g. 'T2,G1' = CPU matrix + GPU block")
    ap.add_argument("--datasets", default="cifar10", help="comma list of cifar10, google_speech, or 'all'")
    ap.add_argument("--baselines", default="", help="space-separated; default = --changed set, else all six")
    ap.add_argument("--changed", default="", help="git ref: run only baselines affected by the diff vs it")
    ap.add_argument("--phases", default="", help="subset of the tier's phase ids (e.g. 'P1 P2', 'P11a')")
    ap.add_argument("--shard", default="1/1", help="i/N: this node's share of the job list")
    ap.add_argument("--max-parallel", type=int, default=32)
    ap.add_argument("--reserve-cores", type=int, default=4, help="physical cores kept out of every slot")
    ap.add_argument("--cpus-per-trainer", type=float, default=1.0,
                    help="logical CPUs per trainer; < 1 only after the ISO control proves it (FX-N22 P6)")
    ap.add_argument("--traces", default="", help="restrict the tier to these traces")
    ap.add_argument("--agg-hp", default="", help="'k=v ...' appended to every phase's aggregator hp (A/B)")
    ap.add_argument("--trainer-hp", default="", help="'k=v ...' appended to every phase's trainer hp")
    ap.add_argument("--gpus-per-job", type=int, default=8, help="GPU legs: GPUs each (capped at the healthy count)")
    ap.add_argument("--gpu-cpus-per-trainer", type=float, default=0.4,
                    help="GPU legs: CPUs per trainer (0.4 = n=300 on 128 CPUs -> whole node; n=100 -> 48 CPUs)")
    ap.add_argument("--mem-headroom-gb", type=float, default=32)
    ap.add_argument("--deadline-h", type=float, default=12)
    ap.add_argument("--output-dir", default="")
    ap.add_argument("--smoke", action="store_true", help="every leg 60s, n<=12 (the pool's own gate)")
    ap.add_argument("--gate", action=argparse.BooleanOptionalAction, default=None,
                    help="R19 gate first: pytest --collect-only + a felix smoke pair per dataset; abort on "
                         "failure (default on unless --smoke/--dry-run)")
    ap.add_argument("--pytest", action="store_true", help="run the full pytest (P0) before the jobs")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    real_changed = False
    if a.baselines:
        baselines = tuple(a.baselines.split())
    elif a.changed:
        baselines, real_changed = affected(changed_paths(a.changed))
    else:
        baselines = B6
    datasets = DATASETS if a.datasets == "all" else tuple(a.datasets.replace(",", " ").split())
    tiers = a.tier.replace(",", " ").split()
    phases = [p for ds in datasets for t in tiers for p in tier_phases(t, baselines, ds)]
    if a.phases:
        want = set(a.phases.split())
        phases = [p for p in phases if p.pid in want or p.pid[len(DS_TAG[p.dataset]):] in want]
    if a.traces:
        phases = [p for p in phases if p.trace in a.traces.split()]
    phases = [p for p in phases if p.baselines]
    for p in phases:
        p.agg_hp = " ".join(x for x in (p.agg_hp, a.agg_hp) if x)
        p.trainer_hp = " ".join(x for x in (p.trainer_hp, a.trainer_hp) if x)
    if a.smoke:  # R19 gate: every leg 60s, small cohorts
        for p in phases:
            p.runtime_s, p.n = 60, min(p.n, 12)
    history = _load_history()
    jobs = build_jobs(phases, a.cpus_per_trainer, a.gpus_per_job, history, a.gpu_cpus_per_trainer)
    i, n = map(int, a.shard.split("/"))
    jobs = shard(jobs, i, n)

    root = Path(a.output_dir or OUT_DIR / f"pool_{time.strftime('%Y%m%d_%H%M%S')}_{'_'.join(tiers)}").resolve()
    root.mkdir(parents=True, exist_ok=True)
    pool = Pool(root, jobs, a.max_parallel, a.reserve_cores, a.mem_headroom_gb, a.deadline_h * 3600, a.dry_run,
                log=open(root / "pool.log", "a"))
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True).stdout.strip()
    pool.say(f"pool {root} host={socket.gethostname()} tier={a.tier} datasets={','.join(datasets)} baselines={' '.join(baselines) or '-'} "
             f"shard={a.shard} commit={commit}")
    if a.changed:
        pool.say(f"changed vs {a.changed}: real path {'CHANGED -> run T3 too' if real_changed else 'unchanged'}")
    groups = core_groups()
    cap = CoreMap(groups, a.reserve_cores).total()
    for j in jobs:
        if j.cpus > cap:  # runs alone on every slot CPU, as a solo run does today
            pool.say(f"note: {j.jid} wants {j.cpus} cpus > {cap}: takes the whole node")
            j.cpus, j.whole = cap, True
        if j.gpus:
            j.gpus = min(j.gpus, len(gpu_ids()))
    bad_gpus = sorted(set(gpu_ids(healthy_only=False)) - set(gpu_ids()))
    if bad_gpus:
        pool.say(f"GPU(s) {bad_gpus} show uncorrected ECC errors: excluded (GPU legs use {gpu_ids()})")
    pool.bad_gpus = bad_gpus
    pool.plan(groups)
    if a.dry_run or not jobs:
        return 0
    if (a.gate is None and not a.smoke) or a.gate:
        rc = run_gate(root, datasets, pool)
        if rc:
            return rc
    if a.pytest:
        pool.say("P0 full pytest ...")
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-n", "auto", "-p", "no:cacheprovider", "tests",
                            "examples/fwdllm/expt_scripts", "examples/async_cifar10/scripts/parity",
                            "examples/async_cifar10/trainer/pytorch"], cwd=str(REPO / "lib" / "python"),
                           capture_output=True, text=True, timeout=2400)
        (root / "P0_pytest.txt").write_text(r.stdout + r.stderr)
        pool.say(f"P0 rc={r.returncode} :: {(r.stdout.strip().splitlines() or [''])[-1]}")
    rc = pool.run(groups)
    if rc == 0:
        subprocess.run(["bash", str(SCRIPT_DIR / "harness_report.sh"), str(root), str(int(pool.t0))])
        pool.say(f"done: {root}/SUMMARY.txt")
        print((root / "SUMMARY.txt").read_text() if (root / "SUMMARY.txt").exists() else "")
    return rc


if __name__ == "__main__":
    sys.exit(main())
