#!/usr/bin/env python3
"""Felix readiness score (PARITY C5): baseline x dataset x scenario cells, each green only if every graded pair of that cell
in the given pools is green (INV/EXACT + EV, `parity_ladder --grade --max-stage 9`; worst pair wins).

  readiness_score.py POOL [POOL ...]          # markdown matrix + score; pass the pools of one code state

Scenarios: syn_0 (syn_0/0b), syn_20, syn_50, mobiperf (pair tiers T3/G0U/P*), stream_lin / stream_eve (G0T), stream_cpu (P7).
Not scored: P11 (injected bugs, graded CAUGHT), oracle arms, real-only (*C) and sim-only (G1AS) phases. Speech mobiperf
Oort cells are n/a (FX-D106).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import parity_ladder as pl  # noqa: E402

BASELINES = ("felix", "fedbuff", "refl", "oort", "oort_star", "feddance")
DATASETS = ("cifar10", "google_speech")
SCENARIOS = ("syn_0", "syn_20", "syn_50", "mobiperf", "stream_lin", "stream_eve", "stream_cpu")
TRACE = {"syn_0": "syn_0", "syn_0b": "syn_0", "syn_20": "syn_20", "syn_50": "syn_50", "mobiperf_3st": "mobiperf"}
MARK = {"green": "✅", "known": "⚠", "red": "❌", None: "·", "n/a": "–"}


def scenario(phase: str, trace: str):
    p = phase[3:] if phase.startswith("gs_") else phase
    if p.startswith(("P11", "P7o", "G0To", "TSo", "G1AS", "G2S")) or p.startswith(pl.REAL_ONLY):
        return None
    if p.startswith("P7"):
        return "stream_cpu"
    if p.startswith("G0T"):
        return "stream_lin" if "_lin_" in p else "stream_eve"
    return TRACE.get(trace)


def score(pools):
    rank = {"green": 0, "known": 1, "red": 2}
    cells = {}
    for root in pools:
        for c in pl.grade_pool(Path(root), 9):
            sc = scenario(c.phase, c.trace)
            if sc is None or c.baseline not in BASELINES:
                continue
            k = (c.baseline, c.dataset, sc)
            if k not in cells or rank[c.status] > rank[cells[k]]:
                cells[k] = c.status
    for b in ("oort", "oort_star"):
        cells[(b, "google_speech", "mobiperf")] = "n/a"
    return cells


def render(cells) -> str:
    out = ["| baseline · dataset | " + " | ".join(SCENARIOS) + " |", "|---" * (len(SCENARIOS) + 1) + "|"]
    for b in BASELINES:
        for d in DATASETS:
            out.append(f"| {b} · {'cifar' if d == 'cifar10' else 'speech'} | "
                       + " | ".join(MARK[cells.get((b, d, s))] for s in SCENARIOS) + " |")
    applicable = [k for k in ((b, d, s) for b in BASELINES for d in DATASETS for s in SCENARIOS) if cells.get(k) != "n/a"]
    green = sum(cells.get(k) == "green" for k in applicable)
    tested = sum(cells.get(k) in ("green", "known", "red") for k in applicable)
    per = " · ".join(f"{s} {sum(cells.get((b, d, s)) == 'green' for b in BASELINES for d in DATASETS)}/"
                     f"{sum(cells.get((b, d, s)) != 'n/a' for b in BASELINES for d in DATASETS)}" for s in SCENARIOS)
    out += ["", f"**Parity score: {green} / {len(applicable)} green** ({tested} tested, {len(applicable) - tested} untested). {per}"]
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pools", nargs="+")
    print(render(score(ap.parse_args(argv).pools)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
