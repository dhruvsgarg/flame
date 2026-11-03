# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Test-function counts per suite and track (fwdllm vs shared/felix). Generated, never hand-kept.

    python lib/python/examples/scripts/test_inventory.py [--ref HEAD]
"""

import argparse
import re
import subprocess
from collections import Counter
from pathlib import Path

LIB = Path(__file__).resolve().parents[2]
SUITES = ["tests", "examples/async_cifar10/scripts/parity", "examples/async_cifar10/trainer/pytorch",
          "examples/fwdllm/expt_scripts"]
_DEF = re.compile(r"^\s*def test_", re.M)


def _files(ref):
    if ref:
        out = subprocess.run(["git", "ls-files", *SUITES], cwd=LIB, capture_output=True, text=True).stdout
        paths = [p for p in out.split() if re.search(r"(^|/)test_[^/]*\.py$", p)]
        for p in paths:
            src = subprocess.run(["git", "show", f"{ref}:./{p}"], cwd=LIB, capture_output=True, text=True).stdout
            yield p, src
    else:
        for s in SUITES:
            for p in (LIB / s).rglob("test_*.py"):
                yield str(p.relative_to(LIB)), p.read_text(errors="replace")


def _bucket(path):
    parts = path.split("/")
    suite = "/".join(parts[:2]) if parts[0] == "tests" and len(parts) > 2 else "/".join(parts[:-1])
    track = "fluxtune" if re.search(r"fwdllm|fluxtune", path) else "shared/felix"
    return suite, track


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", help="git ref to count instead of the working tree")
    args = ap.parse_args()
    counts = Counter()
    for path, src in _files(args.ref):
        counts[_bucket(path)] += len(_DEF.findall(src))
    print(f"{'suite':45s} {'track':13s} tests")
    for (suite, track), n in sorted(counts.items()):
        print(f"{suite:45s} {track:13s} {n:5d}")
    by_track = Counter()
    for (_, track), n in counts.items():
        by_track[track] += n
    print(f"\nTOTAL {sum(counts.values())}  " + "  ".join(f"{t}={n}" for t, n in sorted(by_track.items())))


if __name__ == "__main__":
    main()
