# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""FX-N22 real-leg bank (FX-L24): a sim-only harness leg grades against a stored real leg.

A real leg registers its config key plus hashes of the real-path files. `lookup` returns the newest
leg with the same key; if a real-path file changed since, it is STALE and a real leg must re-run (T3).
Aggregator-file changes are NOT tracked: whether they touch the real path is the caller's call (FX-L24).

  harness_bank.py register --real-dir D --baseline B --trace T ... (every KEY_FIELDS option)
  harness_bank.py lookup --baseline B --trace T ...  -> "<real_dir>\\t<OK|STALE:f1,f2|NONE>"
  harness_bank.py from-dir DIR --trace T --baseline B  -> real_dir of a stored suite/campaign row
"""

import argparse
import csv
import fcntl
import glob
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parents[1]
LIB_DIR = EXAMPLES.parent
BANK = EXAMPLES / "experiments" / "_real_bank.tsv"
KEY_FIELDS = ("dataset", "baseline", "trace", "n", "runtime_s", "delay_factor", "trace_scale", "harness",
              "agg_goal", "concurrency", "agg_hp", "trainer_hp", "inject_bug")
# Code that only the real leg exercises (trainer, transport, availability); relative to lib/python.
REAL_PATH_GLOBS = (
    "flame/channel.py", "flame/channel_manager.py", "flame/backend/*.py",
    "flame/mode/horizontal/syncfl/trainer.py", "flame/availability/*.py",
    "examples/async_cifar10/trainer/pytorch/*.py", "examples/async_cifar10/fl_data.py",
)


def real_path_hashes(root: Path = LIB_DIR) -> dict:
    out = {}
    for pat in REAL_PATH_GLOBS:
        for f in sorted(glob.glob(str(root / pat))):
            out[os.path.relpath(f, root)] = hashlib.sha1(Path(f).read_bytes()).hexdigest()
    return out


def _rows(bank: Path):
    if not bank.exists():
        return []
    with open(bank, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def register(real_dir: str, key: dict, bank: Path = BANK, hashes: dict = None) -> None:
    bank.parent.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "-C", str(LIB_DIR), "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    with open(bank, "a", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)  # parallel slots register concurrently
        w = csv.writer(f, delimiter="\t")
        if f.tell() == 0:
            w.writerow(["ts", "key", "real_dir", "commit", "real_path_hashes"])
        w.writerow([time.strftime("%Y%m%d_%H%M%S"), json.dumps(key, sort_keys=True), real_dir, commit,
                    json.dumps(hashes if hashes is not None else real_path_hashes(), sort_keys=True)])


def lookup(key: dict, bank: Path = BANK, hashes: dict = None) -> tuple:
    """(real_dir, 'OK' | 'STALE:<changed files>' | 'NONE') for the newest leg with this key."""
    want = json.dumps(key, sort_keys=True)
    rows = [r for r in _rows(bank) if r["key"] == want and Path(r["real_dir"]).is_dir()]
    if not rows:
        return "", "NONE"
    row = rows[-1]
    now = hashes if hashes is not None else real_path_hashes()
    then = json.loads(row["real_path_hashes"])
    changed = sorted(f for f in set(now) | set(then) if now.get(f) != then.get(f))
    return row["real_dir"], ("STALE:" + ",".join(Path(f).name for f in changed)) if changed else "OK"


def from_dir(d: str, trace: str, baseline: str) -> str:
    """Real leg of (trace, baseline) in a stored suite/campaign dir (newest summary row), or ''."""
    hit = ""
    for tsv in sorted(glob.glob(os.path.join(d, "**", "summary.tsv"), recursive=True)) or \
            ([os.path.join(d, "summary.tsv")] if os.path.exists(os.path.join(d, "summary.tsv")) else []):
        with open(tsv, newline="") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                if r.get("trace") == trace and r.get("baseline") == baseline and r.get("real_dir"):
                    hit = r["real_dir"] if Path(r["real_dir"]).is_dir() else hit
    return hit


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("register", "lookup"):
        p = sub.add_parser(name)
        if name == "register":
            p.add_argument("--real-dir", required=True)
        for k in KEY_FIELDS:
            p.add_argument("--" + k.replace("_", "-"), dest=k, default="")
    fd = sub.add_parser("from-dir")
    fd.add_argument("dir")
    fd.add_argument("--trace", required=True)
    fd.add_argument("--baseline", required=True)
    a = ap.parse_args(argv)
    key = {k: getattr(a, k, "") for k in KEY_FIELDS}
    if a.cmd == "register":
        register(a.real_dir, key)
    elif a.cmd == "lookup":
        print("\t".join(lookup(key)))
    else:
        print(from_dir(a.dir, a.trace, a.baseline))
    return 0


if __name__ == "__main__":
    sys.exit(main())
