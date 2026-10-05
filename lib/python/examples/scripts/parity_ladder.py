#!/usr/bin/env python3
# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""FX-N42 parity ladder: run the real<->sim tests rung by rung, cheapest first, across the whole matrix (baselines x
datasets x avail/unavail x CPU/GPU). A rung is green when every cell passes its gate or its miss is a known item; the
ladder stops at the first red rung, so no GPU time is spent above a broken fundamental. Rungs and gates: LADDER below
and FELIX_READINESS.md (Active build — parity ladder).

  parity_ladder.py --rungs L1-L5 --datasets all [harness_pool args...]   # run (one pool per rung, under <out>/<rung>)
  parity_ladder.py --grade POOL_DIR [--max-stage 1] [--regrade]          # gate an existing pool (--regrade: re-run
                                                                         # the checker on its stored pairs first)

Writes <out>/LADDER.txt (per rung: cells green / known / red, and each red cell's lowest failing rung).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
EXAMPLES = SCRIPT_DIR.parent
sys.path[:0] = [str(EXAMPLES.parent), str(EXAMPLES / "async_cifar10" / "scripts")]

AVAIL = {"syn_0", "syn_0b"}  # every other trace is an unavailability cell
GATED_TIERS = ("INV", "EXACT")  # DIST rungs gate only once a replicate floor sizes them (parent S2; L6+)


@dataclass(frozen=True)
class Rung:
    rid: str
    what: str
    tiers: str = ""           # harness_pool --tier ('' = gate-only, re-grades `reuse`)
    phases: str = ""          # harness_pool --phases
    max_stage: int = -1       # parity stages gated (-1 = EV only; 99 = all)
    reuse: str = ""           # gate-only rung: the rung whose legs it re-grades
    gpu: bool = False


LADDER: Tuple[Rung, ...] = (
    Rung("L1", "sim alone is correct (EV, CPU, 120s)", tiers="T1"),
    Rung("L2", "real+sim CPU pairs: EV + telemetry/clock parity (stages 0-1)", tiers="T3", max_stage=1),
    Rung("L3", "CPU availability + selection parity (stages 2-3) on the same legs", reuse="L2", max_stage=3),
    Rung("L4", "CPU campaign: controls, streaming, injected bugs (EV)", tiers="T4",
         phases="P4 P5 P6 P7 P7o P8 P9 P10 P11a P11b P11c"),
    Rung("L5", "GPU short pairs (10 min, G0 cohort): EV + stages 0-1", tiers="GS", max_stage=1, gpu=True),
    Rung("L6", "GPU screen (30 min) + real<->real control: EV + INV/EXACT, DIST reported", tiers="G0C,G0",
         max_stage=99, gpu=True),
    Rung("L7", "GPU reference n, 90 min: parent exit criteria", tiers="G1,G2", max_stage=99, gpu=True),
)
BY_ID = {r.rid: r for r in LADDER}

# Known misses: (check, baselines, trace/phase regex, item). A known cell never blocks; closing the item deletes its row.
KNOWN: Tuple[Tuple[str, Tuple[str, ...], str, str], ...] = (
    ("EV1", ("oort",), r"syn_50 (gs_)?T1$", "open question: P3 oort"),  # L1's 120s; CPU mobiperf legs run 960s
    ("EV0|EV1|EV12", tuple(), r"gs_P7o?$", "FX-N30"),
    # Unaware oort stalls 90s on most syn_50 rounds: ~6 stall-free rounds per leg can't grade timing.
    ("overhead_residual|per_round_advance|throughput", ("oort",), r"^syn_50 (gs_)?T3_", "FX-N62"),
    # fedbuff mobiperf (240s): 2-3 rounds, all stalls; < 20 commits grade nothing.
    ("overhead_residual|per_round_advance|throughput", ("fedbuff",), r"^mobiperf_3st (gs_)?T3_", "FX-N62"),
    # feddance lock-in draw at n=50 (FX-D37).
    ("overhead_residual|per_round_advance|throughput|terminal_state|total_commits", ("feddance",),
     r"^syn_0 (gs_)?G0T", "FX-D37"),
)
# Phases whose sim leg must FAIL a named check (injected bugs; P4 = the cold-start-gate-off control, FX-D8).
EXPECTED_FAIL = {"P11a": "EV10", "P11b": "EV16", "P11c": "EV3", "P4": "EV10"}
# Real-only phases (harness_pool kind="real"): no sim leg by design, graded on the real leg's EV.
REAL_ONLY = ("G0C_",)


def _known(check: str, baseline: str, where: str) -> Optional[str]:
    for pat, bls, wre, item in KNOWN:
        if re.fullmatch(pat, check) and (not bls or baseline in bls) and re.search(wre, where):
            return item
    return None


def _ev_fails(v: str) -> List[str]:
    return [] if v in ("PASS", "-", "") else (["MISSING"] if v == "MISSING" else v.replace("FAIL:", "").split(","))


def _regraded_ev(phase_dir: Path, tr: str, b: str) -> dict:
    """--regrade's EV result per side ({'ev_real': [...], 'ev_sim': [...]}), else {} (stored summary wins)."""
    js = next(iter(phase_dir.glob(f"*/parity_regrade/events_{tr}_{b}.json")), None)
    if js is None:
        return {}
    return {f"ev_{r['mode']}": [k.split("_")[0] for k, c in r["checks"].items() if c["status"] == "FAIL"]
            for r in json.loads(js.read_text())}


def _parity_fails(path: Optional[Path], max_stage: int) -> List[Tuple[int, str, str]]:
    """(stage, check, tier) of every failing gated rung at stage <= max_stage, lowest stage first."""
    if max_stage < 0 or path is None or not path.exists():
        return []
    from parity.checks import CHECK_META  # needs flame on the path (dg_flame)
    out = []
    for k, v in json.loads(path.read_text()).items():
        if not isinstance(v, dict) or k == "summary" or v.get("ok") is not False:
            continue
        stage = CHECK_META.get(k, {}).get("stage", 99)
        if v.get("status") in ("SKIP", "WARN") or v.get("tier") not in GATED_TIERS or stage > max_stage:
            continue
        out.append((stage, k, v.get("tier")))
    return sorted(out)


@dataclass
class Cell:
    phase: str
    trace: str
    baseline: str
    status: str   # green | known | red
    why: str

    @property
    def dataset(self) -> str:
        return "google_speech" if self.phase.startswith("gs_") else "cifar10"


def _control_floors(root: Path, checker: Path, jobs: int) -> dict:
    """Q2: {(phase, baseline): floors} from each real replicate leg vs its cell's real leg (n=2: a lower bound, T8):
    G0C per G0 syn_0 cell (syn_20 reuses it), G0UC per G0U cell (may sit in a sibling pool of the block).
    Writes <control phase>/control_<b>.json."""
    import concurrent.futures as cf
    sys.path.insert(0, str(EXAMPLES / "async_cifar10" / "scripts"))
    from parity.checks import control_floors

    def one(item):
        key, (ctl_dir, cmd_txt, out) = item
        t = cmd_txt.read_text()
        real, goal, rt = (re.search(rf"--{k} (\S+)", t) for k in ("real-dir", "agg-goal", "runtime-s"))
        subprocess.run([sys.executable, str(checker), "--real", real[1], "--sim", str(ctl_dir), "--control",
                        "--agg-goal", goal[1], "--budget-s", rt[1], "--json-out", str(out)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return key, (control_floors(json.loads(out.read_text())) if out.exists() else {})

    pairs = {}
    for f in {*root.glob("*G0C_syn_0/summary.tsv"), *root.parent.glob("*/*G0UC_*/summary.tsv")}:
        tag = "G0UC_" if "G0UC_" in f.parent.name else "G0C_"
        pre, _, trace = f.parent.name.partition(tag)
        target = f"{pre}{tag[:-2]}_{trace}"
        for r in csv.DictReader(open(f), delimiter="\t"):
            b = r["baseline"]
            cmd = next(iter(root.parent.glob(f"*/{target}/{target}_{b}_grade/cmd.txt")), None)
            if cmd and r.get("real_dir"):
                pairs[(target, b)] = (Path(r["real_dir"]), cmd, f.parent / f"control_{b}.json")
    with cf.ThreadPoolExecutor(jobs) as ex:
        floors = dict(ex.map(one, pairs.items()))
    for (target, b), fl in list(floors.items()):
        if target.endswith("G0_syn_0"):
            floors.setdefault((target[:-1] + "20", b), fl)
        for mode in ("T_lin", "T_eve", "To_lin", "To_eve"):  # streaming screens share the G0U cohort (lower bound)
            floors.setdefault((target.replace("G0U_", f"G0{mode}_"), b), fl)
    return floors


def regrade_pool(root: Path, jobs: int = 20) -> None:
    """Q3: re-run the checker on every stored pair of a pool into <cell>_grade/parity_regrade/ (read by grade_pool);
    Q2: pairs of a pool with G0C legs are gated against their baseline's real<->real floor (`floors.json`)."""
    import concurrent.futures as cf
    checker = EXAMPLES / "async_cifar10" / "scripts" / "parity_check.py"
    events = checker.parent / "parity" / "event_invariants.py"
    floors = _control_floors(root, checker, jobs)

    def one(cmd_txt: Path) -> None:
        t = cmd_txt.read_text()
        g = {k: re.search(rf"--{k} (\S+)", t) for k in ("real-dir", "sim-dir", "agg-goal", "runtime-s", "baselines", "traces")}
        if not all(g[k] for k in ("real-dir", "sim-dir", "runtime-s")):
            return
        out = cmd_txt.parent / "parity_regrade"
        out.mkdir(exist_ok=True)
        name = next(iter((cmd_txt.parent / "parity").glob("*.json")), None)
        if name is None:
            return
        phase = cmd_txt.parent.parent.name
        fl = floors.get((phase, g["baselines"][1] if g["baselines"] else ""))
        extra = []
        if fl:
            (cmd_txt.parent / "floors.json").write_text(json.dumps(fl))
            extra = ["--floors", str(cmd_txt.parent / "floors.json")]
        goal = ["--agg-goal", g["agg-goal"][1]] if g["agg-goal"] else []  # reference-config tiers (G1/G1S) omit it
        subprocess.run([sys.executable, str(checker), "--real", g["real-dir"][1], "--sim", g["sim-dir"][1],
                        *goal, "--budget-s", g["runtime-s"][1], *extra,
                        "--json-out", str(out / name.name)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run([sys.executable, str(events), g["real-dir"][1], g["sim-dir"][1],  # EV re-run too
                        "--json-out", str(out / f"events_{name.name}")],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    with cf.ThreadPoolExecutor(jobs) as ex:
        list(ex.map(one, root.glob("*/*_grade/cmd.txt")))


def grade_pool(root: Path, max_stage: int) -> List[Cell]:
    cells = []
    for f in sorted(root.glob("*/summary.tsv")):
        phase = f.parent.name
        bare = phase[3:] if phase.startswith("gs_") else phase
        for r in csv.DictReader(open(f), delimiter="\t"):
            b, tr = r["baseline"], r["trace"]
            where = f"{tr} {phase}"
            red, known = [], []
            want = EXPECTED_FAIL.get(bare)
            sides = ("ev_real",) if bare.startswith(REAL_ONLY) else ("ev_real", "ev_sim")
            ev = _regraded_ev(f.parent, tr, b)
            for side in sides:
                fails = ev[side] if side in ev else _ev_fails(r.get(side, ""))
                if want and side == "ev_sim":
                    if want not in fails:
                        red.append(f"sim did not FAIL {want}")
                    fails = [x for x in fails if x not in ("EV5", "EV10", "EV11", "EV16", "EV3", "EV12", want)]
                for c in fails:
                    item = _known(c, b, where)
                    (known if item else red).append(f"{side[3:]} {c}" + (f" ({item})" if item else ""))
            js = next(iter(f.parent.glob(f"*/parity_regrade/{tr}_{b}.json")), None) or \
                next(iter(f.parent.glob(f"*/parity/{tr}_{b}.json")), None)
            for stage, k, tier in _parity_fails(js, max_stage):
                item = _known(k, b, where)
                (known if item else red).append(f"S{stage} {k} [{tier}]" + (f" ({item})" if item else ""))
            status = "red" if red else ("known" if known else "green")
            cells.append(Cell(phase, tr, b, status, "; ".join(red or known)))
    return cells


def render(rung: Rung, cells: List[Cell]) -> str:
    n = {s: sum(c.status == s for c in cells) for s in ("green", "known", "red")}
    lines = [f"== {rung.rid} {rung.what}: {'GREEN' if not n['red'] else 'RED'} "
             f"(green {n['green']}, known {n['known']}, red {n['red']} of {len(cells)})"]
    for c in sorted(cells, key=lambda c: (c.status != "red", c.dataset, c.trace not in AVAIL, c.baseline, c.phase)):
        if c.status != "green":
            av = "avail" if c.trace in AVAIL else "unavail"
            lines.append(f"  {c.status:<5} {c.dataset:<13} {av:<7} {c.baseline:<9} {c.phase:<14} {c.why}")
    return "\n".join(lines)


def _rung_ids(spec: str) -> List[str]:
    ids = [r.rid for r in LADDER]
    out = []
    for part in spec.replace(",", " ").split():
        lo, _, hi = part.partition("-")
        out += ids[ids.index(lo): ids.index(hi or lo) + 1]
    return out


def _run_pool(cmd: List[str]) -> int:
    """R20: wait out the pool's own teardown on Ctrl+C (subprocess.run SIGKILLs it, orphaning legs); forward SIGTERM."""
    proc = subprocess.Popen(cmd)
    stop = {"sig": False}

    def _on(signum, _frm):
        stop["sig"] = True
        if signum == signal.SIGTERM:
            proc.send_signal(signal.SIGTERM)

    old = {s: signal.signal(s, _on) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        rc = proc.wait()
    finally:
        for s, h in old.items():
            signal.signal(s, h)
    return 130 if stop["sig"] else rc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rungs", default="L1-L5", help="e.g. L1-L5, L2,L3, L6")
    ap.add_argument("--grade", help="gate an existing pool dir instead of running")
    ap.add_argument("--max-stage", type=int, default=1, help="with --grade: parity stages gated (-1 = EV only)")
    ap.add_argument("--regrade", action="store_true", help="with --grade: re-run the checker on the stored pairs first")
    ap.add_argument("--output-dir", default="")
    ap.add_argument("--keep-going", action="store_true", help="run every rung even after a red one (report all)")
    a, pool_args = ap.parse_known_args(argv)
    if a.grade:
        if a.regrade:
            regrade_pool(Path(a.grade))
        rung = Rung("grade", f"{a.grade} (stages <= {a.max_stage})", max_stage=a.max_stage)
        print(render(rung, grade_pool(Path(a.grade), a.max_stage)))
        return 0

    out = Path(a.output_dir or EXAMPLES / "experiments" / f"ladder_{time.strftime('%Y%m%d_%H%M%S')}").resolve()
    out.mkdir(parents=True, exist_ok=True)
    report = out / "LADDER.txt"
    first = True
    rids = _rung_ids(a.rungs)
    for i, rid in enumerate(rids, 1):
        rung = BY_ID[rid]
        root = out / (rung.reuse or rid)
        if rung.tiers:
            cmd = [sys.executable, str(SCRIPT_DIR / "harness_pool.py"), "--tier", rung.tiers, "--output-dir", str(root),
                   *(["--phases", rung.phases] if rung.phases else []), *([] if first else ["--no-gate"]),
                   "--progress-label", f"{rid} (rung {i}/{len(rids)})", *pool_args]
            print(f"[{time.strftime('%F %T')}] {rid}: {' '.join(cmd[2:])}", flush=True)
            rc = _run_pool(cmd)
            if rc == 130:
                return 130
            first = False
            if rc not in (0, 3):  # 3 = fail-fast abort: gate what finished, it is red
                with open(report, "a") as f:
                    f.write(f"== {rid} pool rc={rc}: stopped\n")
                return rc
        if "G0C" in rung.tiers:
            regrade_pool(root)  # Q2: floor-gate against the real<->real legs
        cells = grade_pool(root, rung.max_stage)
        text = render(rung, cells)
        with open(report, "a") as f:
            f.write(text + "\n\n")
        print(text, flush=True)
        if any(c.status == "red" for c in cells) and not a.keep_going:
            print(f"ladder stopped at {rid} (red): fix the red cells, then --rungs {rid}-... -- {report}")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
