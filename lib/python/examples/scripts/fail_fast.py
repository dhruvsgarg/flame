#!/usr/bin/env python3
# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""FX-N40 fail fast: a fatal line in any leg's logs stops the whole batch (a GPU or code fault that hit one leg
hits the next). Fatal = the lines EV0 counts; the FX-N33 post-leave exit abort is allowlisted by signature.
harness_pool.py scans live legs every ~30s; harness_suite.sh after each pair.

  fail_fast.py RUN_DIR... [--abort-file ABORT.txt]   # exit 3 when fatal, printing file:line + context
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

FATAL = re.compile(r"Traceback \(most recent call last\)|Fatal Python error|Segmentation fault")
BENIGN_PRECEDED_BY = {"Fatal Python error: Aborted": "terminate called without an active exception"}  # FX-N33
CONTEXT_LINES = 40
EXIT_FATAL = 3


@dataclass(frozen=True)
class Finding:
    path: Path
    lineno: int
    line: str

    def context(self) -> str:
        """file:line, then the lines after it (a traceback's frames and its exception)."""
        out = [f"{self.path}:{self.lineno}"]
        try:
            with open(self.path, errors="replace") as f:
                for i, ln in enumerate(f, 1):
                    if i >= self.lineno:
                        out.append(ln.rstrip("\n"))
                    if i >= self.lineno + CONTEXT_LINES:
                        break
        except OSError:
            out.append(self.line)
        return "\n".join(out)


class Scanner:
    """Incremental: each call reads only what the logs gained since the last one."""

    def __init__(self) -> None:
        self._pos: Dict[Path, tuple] = {}  # log -> (byte offset, lines read, last line)

    def scan(self, run_dirs: Iterable) -> List[Finding]:
        found = []
        for d in run_dirs:
            d = Path(d)
            if not d.is_dir():
                continue
            for log in sorted(d.rglob("*.log")):
                if not log.name.endswith("_resources.log"):  # node-wide monitor, not the run
                    found += self._scan_file(log)
        return found

    def _scan_file(self, log: Path) -> List[Finding]:
        off, n, prev = self._pos.get(log, (0, 0, ""))
        try:
            with open(log, "rb") as f:
                f.seek(off)
                chunk = f.read()
        except OSError:
            return []
        end = chunk.rfind(b"\n") + 1  # a partial last line waits for the next scan
        found = []
        for raw in chunk[:end].splitlines():
            n += 1
            line = raw.decode(errors="replace").rstrip("\r")
            if FATAL.search(line) and BENIGN_PRECEDED_BY.get(line.strip()) != prev.strip():
                found.append(Finding(log, n, line))
            prev = line
        self._pos[log] = (off + end, n, prev)
        return found


def leg_run_dirs(out: Path) -> List[str]:
    """A harness_suite output dir's run dirs, as each leg registered them (FLAME_RUN_DIR_FILE)."""
    dirs = []
    for legs in sorted(Path(out).glob("runs/*/legs.txt")):
        dirs += [ln.strip() for ln in legs.read_text().splitlines() if ln.strip()]
    return dirs


def write_abort(path: Path, where: str, findings: List[Finding]) -> None:
    head = f"FAIL-FAST (FX-N40): {len(findings)} fatal line(s) in {where}; first:\n\n"
    Path(path).write_text(head + findings[0].context() + "\n\nall:\n"
                          + "\n".join(f"{f.path}:{f.lineno}: {f.line}" for f in findings) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="*")
    ap.add_argument("--abort-file")
    a = ap.parse_args(argv)
    found = Scanner().scan(a.run_dirs)
    if not found:
        return 0
    if a.abort_file:
        write_abort(Path(a.abort_file), " ".join(a.run_dirs), found)
    print(found[0].context())
    return EXIT_FATAL


if __name__ == "__main__":
    sys.exit(main())
