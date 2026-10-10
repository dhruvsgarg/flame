#!/usr/bin/env python3
"""Free experiment storage without losing anything a grade, regrade, the real bank or a doc still needs.

  cleanup_experiments.py                  # dry run: what would go, GB per category
  cleanup_experiments.py --apply          # delete it
  ... --days 7 --ckpt-days 2 --fwdllm     # keep window; checkpoint window; also sweep fwdllm/experiments (FluxTune parked)

Keeps: every dir named in a tracked .md; pools/batches newer than --days; every run those pools reference (legs.txt,
summary.tsv real_dir/sim_dir); the newest real_dir per real-bank key (harness_bank); runs newer than --days; anything
touched in the last 6 h (live legs); git-tracked files (configs/, README.md).
Deletes: other run dirs (async_cifar10/experiments run_*/smoke_*/campaign_*/repro_*/_aborted), old uncited pools/batches
in examples/experiments, and checkpoints/*.pt of kept runs older than --ckpt-days unless the run is cited
(only oracle_misselection.py reads checkpoints; grading never does).
"""
import argparse
import csv
import shutil
import subprocess
import sys
import time
from pathlib import Path

EX = Path(__file__).resolve().parents[1]
REPO = EX.parents[2]
POOLS = EX / "experiments"
RUNS = [EX / "async_cifar10" / "experiments"]
LIVE_S = 6 * 3600
csv.field_size_limit(sys.maxsize)


def size(p: Path) -> int:
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file() and not f.is_symlink())


def touched(p: Path) -> float:
    """Newest mtime of the dir, its children and its telemetry (a live leg grows these)."""
    ts = [p.stat().st_mtime]
    for sub in (p, p / "telemetry"):
        if sub.is_dir():
            ts += [c.stat().st_mtime for c in sub.iterdir()]
    return max(ts)


def tracked() -> set:
    out = subprocess.run(["git", "-C", str(REPO), "ls-files"], capture_output=True, text=True).stdout.split()
    return {REPO / f for f in out}


def doc_text(files: set) -> str:
    return "\n".join(f.read_text(errors="replace") for f in files if f.suffix == ".md" and f.exists())


def pool_refs(pool: Path) -> set:
    refs = set()
    for f in pool.rglob("legs.txt"):
        refs |= {Path(ln.strip()).resolve() for ln in f.read_text().splitlines() if ln.strip()}
    for f in pool.rglob("summary.tsv"):
        for r in csv.DictReader(open(f), delimiter="\t"):
            refs |= {Path(r[k]).resolve() for k in ("real_dir", "sim_dir") if r.get(k) and r[k] != "-"}
    return refs


def bank_refs() -> set:
    bank = POOLS / "_real_bank.tsv"
    if not bank.exists():
        return set()
    newest = {}
    for r in csv.DictReader(open(bank), delimiter="\t"):
        if r["ts"] >= newest.get(r["key"], ("",))[0]:
            newest[r["key"]] = (r["ts"], r["real_dir"])
    return {Path(d).resolve() for _, d in newest.values()}


def plan(days: float, ckpt_days: float, fwdllm: bool):
    now, git = time.time(), tracked()
    docs = doc_text(git)
    old = lambda p, d: now - touched(p) > d * 86400
    cited = lambda p: p.name in docs
    pools = [p for p in POOLS.iterdir() if p.is_dir() and not p.name.startswith(".")]
    keep_pools = [p for p in pools if cited(p) or not old(p, days)]
    drop_pools = [p for p in pools if p not in keep_pools and now - touched(p) > LIVE_S]
    refs = bank_refs()
    for p in keep_pools:
        refs |= pool_refs(p)
    drop_runs, ckpts = [], []
    for root in RUNS + ([EX / "fwdllm" / "experiments"] if fwdllm else []):
        has_tracked = {root / g.relative_to(root).parts[0] for g in git if g.is_relative_to(root)}
        for d in sorted(root.iterdir()):
            if not d.is_dir() or d in has_tracked:
                continue
            if now - touched(d) < LIVE_S:
                continue
            if d.resolve() in refs or cited(d) or not old(d, days):
                if not cited(d) and old(d, ckpt_days):
                    ckpts += list((d / "checkpoints").glob("*.pt"))
                continue
            drop_runs.append(d)
    return drop_runs, drop_pools, ckpts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="delete (default: dry run)")
    ap.add_argument("--days", type=float, default=7, help="keep runs/pools touched within this many days")
    ap.add_argument("--ckpt-days", type=float, default=2, help="drop checkpoints of kept runs older than this")
    ap.add_argument("--fwdllm", action="store_true", help="also sweep fwdllm/experiments")
    ap.add_argument("-v", "--verbose", action="store_true", help="list every path")
    a = ap.parse_args(argv)
    drop_runs, drop_pools, ckpts = plan(a.days, a.ckpt_days, a.fwdllm)
    total = 0
    for name, items in (("run dirs", drop_runs), ("pools/batches", drop_pools), ("checkpoints (.pt)", ckpts)):
        gb = sum(size(p) for p in items) / 1e9
        total += gb
        print(f"{name:18s} {len(items):6d}  {gb:8.1f} GB")
        if a.verbose:
            print("".join(f"  {p}\n" for p in items), end="")
    print(f"{'total':18s} {'':6s}  {total:8.1f} GB  ({'deleting' if a.apply else 'dry run; --apply to delete'})")
    if a.apply:
        for p in drop_runs + drop_pools + ckpts:
            shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
