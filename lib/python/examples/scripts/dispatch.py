#!/usr/bin/env python3
"""Central dispatcher (FX-D186; operator 10-17): one batch over several nodes, balanced so they finish together.

  dispatch.py BATCH.yaml --plan            # expand, estimate per node, print the balanced plan and predicted finish per node
  dispatch.py BATCH.yaml                   # launch (run it in tmux `dg_flame`, PL4); one Ctrl+C stops every node

BATCH.yaml:
  name: pr37rep
  common: "--lease-from-telemetry"         # harness_pool args every stage gets
  stages:                                   # in priority order (lower = GPU priority, harness_pool --priority)
    - {id: R1, args: "--tier G3C --datasets google_speech --traces syn_0"}
    - {id: R2, args: "--tier G3C --datasets cifar10 --traces syn_0", nodes: [shepherd]}   # optional node allow-list

Each node runs one harness_pool per stage with `--queue`: the pool holds every unit (a pair + its grade, or one leg) until this
dispatcher lists it. Planning: each node expands the stage itself (`--jobs-json`: its own history, cores, GPUs, RAM budget); units go
longest-first to the node whose list-scheduled finish (CPU, GPU, RAM; FX-L64) is earliest. Late binding: a unit is released only when
its planned start is within LOOKAHEAD_S, and the plan is redone every POLL_S from each pool's status.json (elapsed vs estimate,
finished legs), so a node that runs slow (foreign load, overruns, slow legs) simply receives fewer of the remaining units.
Health per poll: pool aborted / STALLED / DOOMED / rc != 0, legs > 1.25 x estimate, foreign GPU processes, host RAM < 10% free.
At the end every remote pool and its run dirs are copied here (rsync -r, FX-T48) and DISPATCH.txt lists planned vs actual per node.
"""
import argparse
import json
import math
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

REPO = Path("/home/dgarg39/flame")
EX = REPO / "lib" / "python" / "examples"
PY = "/coc/scratch/dgarg/miniconda3/envs/dg_flame/bin/python"
NODES = {  # R4: jayne GPUs 0,2-7 (GPU 1 out, S0), shepherd 0-3; never kaylee or wash
    "jayne": {"ssh": None, "gpus": "0,2,3,4,5,6,7"},
    "shepherd": {"ssh": "shepherd", "gpus": "0,1,2,3"},
}
POLL_S, LOOKAHEAD_S, BOARD_S, SLOW_X = 60, 1800, 300, 1.25


@dataclass
class Leg:
    key: str  # stage:jid
    est: Dict[str, float]  # node -> seconds
    cpus: int
    mem_gb: float
    gpus: int
    deps: List[str]


@dataclass
class Unit:
    uid: str  # stage:unit
    stage: str
    prio: int
    legs: List[Leg]
    nodes: List[str]
    node: Optional[str] = None  # where it was released
    released_at: float = 0.0

    def span(self, n: str) -> float:  # critical path on node n: parallel legs, then the legs that wait on them
        return max((l.est[n] for l in self.legs if not l.deps), default=0) + sum(l.est[n] for l in self.legs if l.deps)


@dataclass
class NodeCap:
    cpus: int
    gpus: int
    mem_gb: float


def units_from_jobs(stage: str, prio: int, per_node: Dict[str, dict], allowed: List[str]) -> List[Unit]:
    """One unit per harness_pool unit (a pair + its grade, or one leg); each node's estimate per leg."""
    out: Dict[str, Unit] = {}
    for node, d in per_node.items():
        for j in d["jobs"]:
            uid = f"{stage}:{j['unit']}"
            u = out.setdefault(uid, Unit(uid, stage, prio, [], allowed))
            key = f"{stage}:{j['jid']}"
            leg = next((l for l in u.legs if l.key == key), None)
            if leg is None:
                leg = Leg(key, {}, j["cpus"], j["mem_gb"], j["gpus"], [f"{stage}:{x}" for x in j["deps"]])
                u.legs.append(leg)
            leg.est[node] = float(j["est_s"])
    return sorted(out.values(), key=lambda u: (u.prio, -max(u.span(n) for n in u.legs[0].est), u.uid))


def simulate(items: List[tuple], cap: NodeCap, now: float = 0.0) -> tuple:
    """List-schedule legs [(key, dur, cpus, mem, gpus, fixed_start, deps)] on one node (CPU, RAM, GPU; deps on legs not listed
    count as done). fixed_start = already running. Returns (makespan, {key: start})."""
    keys = {it[0] for it in items}
    running, starts, done_at = [], {}, {}
    free = [cap.cpus, cap.mem_gb, cap.gpus]
    t = now
    for k, dur, c, m, g, fs, _ in items:
        if fs is not None:
            running.append((fs + dur, k, c, m, g))
            free = [free[0] - c, free[1] - m, free[2] - g]
            starts[k] = fs
    queue = [it for it in items if it[5] is None]
    while queue or running:
        for it in list(queue):
            k, dur, c, m, g, _, deps = it
            if any(d in keys and done_at.get(d, math.inf) > t for d in deps):
                continue
            if (c <= free[0] and m <= free[1] and g <= free[2]) or not running:  # bigger than the node: runs alone
                running.append((t + dur, k, c, m, g))
                free = [free[0] - c, free[1] - m, free[2] - g]
                starts[k] = t
                queue.remove(it)
        if not running:
            break
        running.sort()
        end, k, c, m, g = running.pop(0)
        t = max(t, end)
        done_at[k] = end
        free = [free[0] + c, free[1] + m, free[2] + g]
    return max([now] + list(done_at.values())), starts


def unit_items(u: Unit, n: str) -> List[tuple]:
    return [(l.key, l.est[n], l.cpus, l.mem_gb, l.gpus, None, l.deps) for l in u.legs]


def plan(units: List[Unit], caps: Dict[str, NodeCap], committed: Dict[str, List[tuple]], now: float = 0.0) -> tuple:
    """Greedy LPT: each free unit (priority, then longest) goes to the node with the earliest resulting finish.
    committed[node] = leg items already released there. Returns ({uid: node}, {uid: planned start}, {node: makespan})."""
    assign, items = {}, {n: list(committed.get(n, [])) for n in caps}
    for u in [u for u in units if u.node is None]:
        best = None
        for n in u.nodes:
            if n not in caps or any(n not in l.est for l in u.legs):
                continue
            span, _ = simulate(items[n] + unit_items(u, n), caps[n], now)
            if best is None or span < best[0]:
                best = (span, n)
        if best is None:
            continue
        assign[u.uid] = best[1]
        items[best[1]] += unit_items(u, best[1])
    starts, spans = {}, {}
    for n in caps:
        spans[n], st = simulate(items[n], caps[n], now)
        for u in units:
            if assign.get(u.uid) == n:
                starts[u.uid] = min(st.get(l.key, math.inf) for l in u.legs)
    return assign, starts, spans


def sh(node: str, cmd: str, timeout: int = 120, check: bool = False) -> subprocess.CompletedProcess:
    host = NODES[node]["ssh"]
    argv = ["bash", "-c", cmd] if host is None else ["ssh", "-q", "-o", "BatchMode=yes", host, cmd]
    r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    if check and r.returncode:
        raise RuntimeError(f"{node}: {cmd[:80]} rc={r.returncode} {r.stderr[-400:]}")
    return r


@dataclass
class Dispatcher:
    batch: dict
    nodes: List[str]
    root: Path
    units: List[Unit] = field(default_factory=list)
    caps: Dict[str, NodeCap] = field(default_factory=dict)
    pools: Dict[tuple, str] = field(default_factory=dict)  # (node, stage) -> remote pool dir
    status: Dict[tuple, dict] = field(default_factory=dict)
    stop: bool = False

    def log(self, msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(self.root / "DISPATCH.log", "a") as fh:
            fh.write(line + "\n")

    def preflight(self) -> None:
        """PL18 one code state on every node; PL6 GPUs ECC-clean."""
        keys = {}
        for n in self.nodes:
            r = sh(n, f"cd {EX} && git -C {REPO} rev-parse --short HEAD && {PY} -c \"import sys; sys.path.insert(0,'scripts'); "
                      f"import harness_pool as h; print(h.code_key())\"", check=True)
            keys[n] = tuple(r.stdout.split()[-2:])
            ecc = sh(n, "nvidia-smi --query-gpu=index,ecc.errors.uncorrected.volatile.total --format=csv,noheader").stdout
            bad = [l for l in ecc.splitlines() if l.split(",")[-1].strip() not in ("0", "[N/A]")]
            if bad:
                self.log(f"WARN {n}: GPUs with volatile ECC errors: {bad}")
        if len(set(keys.values())) != 1:
            raise SystemExit(f"code state differs across nodes (commit, code_key): {keys}")
        self.log(f"preflight: commit / code_key {next(iter(keys.values()))} on {', '.join(self.nodes)}")

    def expand(self) -> None:
        common = self.batch.get("common", "")
        for prio, st in enumerate(self.batch["stages"]):
            allowed = [n for n in st.get("nodes", self.nodes) if n in self.nodes]
            per = {}
            for n in allowed:
                out = f"/tmp/dispatch_{self.batch['name']}_{st['id']}_{n}"
                sh(n, f"cd {EX} && {PY} scripts/harness_pool.py {st['args']} {common} --gpu-ids {NODES[n]['gpus']} "
                      f"--output-dir {out} --jobs-json {out}.json >/dev/null 2>&1", timeout=600)
                per[n] = json.loads(sh(n, f"cat {out}.json", check=True).stdout)
                self.caps[n] = NodeCap(per[n]["cpus"], len(per[n]["gpus"]), per[n]["mem_budget_gb"])
            self.units += units_from_jobs(st["id"], prio, per, allowed)
        self.log(f"{len(self.units)} units over {len(self.batch['stages'])} stages; caps " +
                 ", ".join(f"{n} {c.cpus} CPUs {c.gpus} GPUs {c.mem_gb:.0f} GB" for n, c in self.caps.items()))

    def show_plan(self, assign, starts, spans) -> None:
        for n in self.caps:
            mine = [u for u in self.units if assign.get(u.uid) == n or u.node == n]
            legh = sum(l.est[n] for u in mine for l in u.legs) / 3600
            self.log(f"PLAN {n}: {len(mine)} units, {legh:.1f} leg-h, finish ~{spans[n] / 3600:.1f} h")
        for u in self.units[:200]:
            if u.uid in assign:
                n = assign[u.uid]
                print(f"  {n:9s} start {starts.get(u.uid, 0) / 3600:5.2f} h  {u.span(n) / 60:6.1f} min  "
                      f"{max(l.gpus for l in u.legs)} GPU {sum(l.mem_gb for l in u.legs if not l.deps):4.0f} GB  {u.uid}")

    def launch_pools(self, ts: str) -> None:
        common = self.batch.get("common", "")
        for prio, st in enumerate(self.batch["stages"]):
            for n in [n for n in st.get("nodes", self.nodes) if n in self.nodes]:
                d = f"{EX}/experiments/pool_{ts}_{self.batch['name'].upper()}_{n.upper()}_{st['id']}"
                cmd = (f"mkdir -p {d} && : > {d}/queue.txt && cd {EX} && {{ FLAME_CONDA_ENV=dg_flame setsid nohup {PY} "
                       f"scripts/harness_pool.py {st['args']} {common} --gpu-ids {NODES[n]['gpus']} --priority {prio} "
                       f"--deadline-h 96 --queue {d}/queue.txt --output-dir {d} > {d}.out 2>&1 < /dev/null & echo $! > {d}/pool.pid; }}")
                sh(n, cmd, check=True)
                self.pools[(n, st["id"])] = d
                self.log(f"pool {n} {st['id']}: {d}")

    def release(self, u: Unit, n: str) -> None:
        d = self.pools[(n, u.stage)]
        sh(n, f"echo {shlex.quote(u.uid.split(':', 1)[1])} >> {d}/queue.txt", check=True)
        u.node, u.released_at = n, time.time()

    def poll(self) -> None:
        for (n, sid), d in self.pools.items():
            r = sh(n, f"cat {d}/status.json 2>/dev/null; echo; tail -n 3 {d}.out 2>/dev/null | tr -d '\\r' | tail -c 400")
            try:
                self.status[(n, sid)] = json.loads(r.stdout.splitlines()[0])
            except (IndexError, json.JSONDecodeError):
                self.status.setdefault((n, sid), {})["tail"] = r.stdout[-400:]

    def committed(self, now: float) -> Dict[str, List[tuple]]:
        """Released, unfinished legs: running ones with remaining time re-estimated from elapsed (slow legs grow), the rest at estimate."""
        out = {n: [] for n in self.caps}
        for u in self.units:
            if u.node is None:
                continue
            st = self.status.get((u.node, u.stage), {})
            for l in u.legs:
                jid = l.key.split(":", 1)[1]
                if jid in st.get("done", {}):
                    continue
                run = st.get("running", {}).get(jid)
                if run:
                    rem = max(l.est[u.node] - run["elapsed_s"], 0.1 * run["elapsed_s"], 60.0)
                    out[u.node].append((l.key, rem, l.cpus, l.mem_gb, l.gpus, now, l.deps))
                else:
                    out[u.node].append((l.key, l.est[u.node], l.cpus, l.mem_gb, l.gpus, None, l.deps))
        return out

    def health(self) -> List[str]:
        warn = []
        for (n, sid), st in self.status.items():
            if st.get("aborted"):
                warn.append(f"!! {n} {sid}: pool ABORTED")
            for j, v in st.get("running", {}).items():
                if v.get("stalled"):
                    warn.append(f"!! {n} {j}: {v['stalled']}")
                if v["est_s"] and v["elapsed_s"] > SLOW_X * v["est_s"]:
                    warn.append(f"!  {n} {j}: slow {v['elapsed_s'] / 60:.0f} vs est {v['est_s'] / 60:.0f} min")
            mt = st.get("mem_total_gb") or 0
            if mt and st.get("mem_available_gb", mt) < 0.1 * mt:
                warn.append(f"!  {n}: host RAM {st['mem_available_gb']:.0f} of {mt:.0f} GB free")
        for n in self.nodes:  # foreign GPU processes (PL16): any compute app not owned by us
            r = sh(n, "for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do ps -o user= -p $p; done | sort | uniq -c")
            foreign = [l for l in r.stdout.splitlines() if l.split()[-1] != "dgarg39"]
            if foreign:
                warn.append(f"!  {n}: foreign GPU processes {foreign}")
        return warn

    def board(self, spans) -> None:
        for n in self.caps:
            stz = [st for (m, _), st in self.status.items() if m == n]
            running = sum(len(st.get("running", {})) for st in stz)
            done = sum(len(st.get("done", {})) for st in stz)
            left = sum(1 for u in self.units if u.node in (None, n))
            self.log(f"STATUS {n}: legs running {running}, done {done}; units unfinished/unreleased {left}; "
                     f"finish ~{(spans.get(n, 0) - time.time()) / 3600:.1f} h")

    def run(self) -> int:
        signal.signal(signal.SIGINT, lambda *_: setattr(self, "stop", True))
        signal.signal(signal.SIGTERM, lambda *_: setattr(self, "stop", True))
        ts = time.strftime("%Y%m%d_%H%M")
        self.launch_pools(ts)
        t_board, ended = 0.0, False
        while not self.stop:
            now = time.time()
            self.poll()
            assign, starts, spans = plan(self.units, self.caps, self.committed(now), now)
            for u in self.units:
                if u.uid in assign and starts.get(u.uid, math.inf) <= now + LOOKAHEAD_S:
                    self.release(u, assign[u.uid])
                    self.log(f"RELEASE {u.uid} -> {u.node} (planned start +{(starts[u.uid] - now) / 60:.0f} min)")
            if not ended and all(u.node for u in self.units):
                for (n, _), d in self.pools.items():
                    sh(n, f"echo END >> {d}/queue.txt", check=True)
                ended = True
                self.log("all units released: END sent")
            if now - t_board > BOARD_S:
                self.board(spans)
                for w in self.health():
                    self.log(w)
                t_board = now
            alive = [k for k, d in self.pools.items() if sh(k[0], f"kill -0 $(cat {d}/pool.pid) 2>/dev/null && echo up").stdout.strip()]
            if ended and not alive:
                break
            time.sleep(POLL_S)
        if self.stop:
            self.log("STOP: ABORT to every pool")
            for (n, _), d in self.pools.items():
                sh(n, f"echo ABORT >> {d}/queue.txt")
            return 130
        self.collect()
        return 0

    def collect(self) -> None:
        """Copy remote pools + their run dirs here (rsync -r: --files-from does not recurse, FX-T48)."""
        for (n, sid), d in self.pools.items():
            host = NODES[n]["ssh"]
            if host is None:
                continue
            lst = f"/tmp/dispatch_runs_{Path(d).name}.txt"
            sh(n, f"cd {d} && cat */*/summary.tsv 2>/dev/null | tr '\\t' '\\n' | grep '/async_cifar10/experiments/run_' "
                  f"| sed 's#.*/async_cifar10/experiments/##' | sort -u > {lst}")
            subprocess.run(["scp", "-q", f"{host}:{lst}", lst])
            subprocess.run(["rsync", "-a", "-r", "-e", "ssh -q", f"--files-from={lst}",
                            f"{host}:{EX}/async_cifar10/experiments/", f"{EX}/async_cifar10/experiments/"])
            subprocess.run(["rsync", "-a", "-e", "ssh -q", f"{host}:{d}", f"{host}:{d}.out", f"{EX}/experiments/"])
        lines = [f"{n} {sid}: {d}" for (n, sid), d in self.pools.items()]
        (self.root / "DISPATCH.txt").write_text("\n".join(lines) + "\n")
        self.log(f"collected; pools listed in {self.root}/DISPATCH.txt")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("batch")
    ap.add_argument("--nodes", default="jayne,shepherd")
    ap.add_argument("--plan", action="store_true")
    a = ap.parse_args(argv)
    batch = yaml.safe_load(open(a.batch))
    root = EX / "experiments" / f"dispatch_{batch['name']}_{time.strftime('%Y%m%d_%H%M')}"
    root.mkdir(parents=True, exist_ok=True)
    d = Dispatcher(batch, a.nodes.split(","), root)
    d.preflight()
    d.expand()
    assign, starts, spans = plan(d.units, d.caps, {})
    d.show_plan(assign, starts, spans)
    if a.plan:
        return 0
    return d.run()


if __name__ == "__main__":
    sys.exit(main())
