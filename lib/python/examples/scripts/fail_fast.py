#!/usr/bin/env python3
# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""FX-N40 fail fast: a fatal line in any leg's logs stops the whole batch (a GPU or code fault that hit one leg
hits the next). FX-D45 StallWatch: a leg that stops progressing (rules S1/S2) is killed alone; the pool goes on. Fatal = the lines EV0 counts; the FX-N33 post-leave exit abort is allowlisted by signature.
harness_pool.py scans live legs every ~30s; harness_suite.sh after each pair.

  fail_fast.py RUN_DIR... [--abort-file ABORT.txt]   # exit 3 when fatal, printing file:line + context
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

FATAL = re.compile(r"Traceback \(most recent call last\)|Fatal Python error|Segmentation fault")
BROKER_FATAL = re.compile(r"already connected, closing old connection")  # a client id live twice: two runs, one broker
BENIGN_PRECEDED_BY = {"Fatal Python error: Aborted": "terminate called without an active exception"}  # FX-N33
TRACEBACK = re.compile(r"Traceback \(most recent call last\)")
EXIT_REQUEST = re.compile(r"^SystemExit(: 0)?$")  # SIGTERM's clean exit landing in a gc callback at teardown
_INTERLEAVED = re.compile(r"^\s|^\d{4}-\d\d-\d\d |^$")  # frame lines and other threads' log lines
TB_WAIT_LINES = 60
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


def fatal_hits(lines: Iterable, pat: re.Pattern = FATAL, state: Optional[dict] = None, final: bool = False):
    """Yield (lineno, line) per fatal line. A Traceback is held until its exception line and dropped when that is a
    clean exit request; `state` carries a held one across incremental reads, `final` (a complete file) flushes it."""
    st = {} if state is None else state
    for n, line in lines:
        held = st.get("held")
        if held is not None:
            if _INTERLEAVED.match(line) and st["waited"] < TB_WAIT_LINES:
                st["waited"] += 1
                st["prev"] = line
                continue
            st["held"] = None
            if not EXIT_REQUEST.match(line.strip()):
                yield held
        if pat.search(line) and TRACEBACK.search(line):
            st["held"], st["waited"] = (n, line), 0
        elif pat.search(line) and BENIGN_PRECEDED_BY.get(line.strip()) != st.get("prev", "").strip():
            yield n, line
        st["prev"] = line
    if final and st.get("held") is not None:
        yield st.pop("held")


class Scanner:
    """Incremental: each call reads only what the logs gained since the last one."""

    def __init__(self) -> None:
        self._pos: Dict[Path, tuple] = {}  # log -> (byte offset, lines read, fatal_hits state)

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

    def scan_broker(self, log: Path) -> List[Finding]:
        """L28: the leg's private broker log."""
        return self._scan_file(log, BROKER_FATAL) if log.exists() else []

    def _scan_file(self, log: Path, pat: re.Pattern = FATAL) -> List[Finding]:
        off, n, st = self._pos.get(log, (0, 0, {}))
        try:
            with open(log, "rb") as f:
                f.seek(off)
                chunk = f.read()
        except OSError:
            return []
        end = chunk.rfind(b"\n") + 1  # a partial last line waits for the next scan
        lines = [(n + i + 1, raw.decode(errors="replace").rstrip("\r")) for i, raw in enumerate(chunk[:end].splitlines())]
        found = [Finding(log, k, line) for k, line in fatal_hits(lines, pat, st)]
        self._pos[log] = (off + end, n + len(lines), st)
        return found


STALL_NO_ROUND_S = 15 * 60  # S1: no new agg_round (no committed version) for this long
STALL_SILENT_S = 10 * 60    # S2: no leg log grew for this long


class StallWatch:
    """Early-termination rules for a live leg; only a leg that stopped progressing trips one, never a slow one.

    S1 no new `agg_round` event for STALL_NO_ROUND_S (from leg start until the first) · S2 no leg log grew for
    STALL_SILENT_S. Disarmed once the aggregator logs `run_end`: teardown and grading are bounded by the leg's own
    timeouts (FX-D62). Reads only what the files gained since the last call."""

    def __init__(self, t0: float, no_round_s: float = STALL_NO_ROUND_S, silent_s: float = STALL_SILENT_S) -> None:
        self.no_round_s, self.silent_s = no_round_s, silent_s
        self.last_round = self.last_growth = t0
        self.rounds = 0
        self.ended = False
        self._off: Dict[Path, int] = {}
        self._size: Dict[Path, int] = {}

    def check(self, run_dirs: Iterable, now: float) -> str:
        """'' while healthy, else the tripped rule with its evidence."""
        for d in map(Path, run_dirs):
            if not d.is_dir():
                continue
            for f in list(d.rglob("*.log")) + list(d.glob("telemetry/aggregator_*.jsonl")):
                if f.name.endswith("_resources.log"):
                    continue
                try:
                    size = f.stat().st_size
                except OSError:
                    continue
                if size != self._size.get(f):
                    self._size[f] = size
                    self.last_growth = now
                if f.suffix == ".jsonl":
                    self._read_rounds(f, now)
        if self.ended:
            return ""
        if self.no_round_s and now - self.last_round > self.no_round_s:
            return (f"S1 no new agg_round for {(now - self.last_round) / 60:.0f} min "
                    f"({self.rounds} rounds so far; limit {self.no_round_s / 60:.0f} min)")
        if self.silent_s and now - self.last_growth > self.silent_s:
            return f"S2 no leg log grew for {(now - self.last_growth) / 60:.0f} min (limit {self.silent_s / 60:.0f} min)"
        return ""

    def _read_rounds(self, f: Path, now: float) -> None:
        off = self._off.get(f, 0)
        try:
            with open(f, "rb") as fh:
                fh.seek(off)
                chunk = fh.read()
        except OSError:
            return
        end = chunk.rfind(b"\n") + 1
        n = chunk[:end].count(b'"event": "agg_round"')
        self.ended = self.ended or b'"event": "run_end"' in chunk[:end]
        if n:
            self.rounds += n
            self.last_round = now
        self._off[f] = off + end


def write_stalled(path: Path, where: str, rule: str, run_dirs: List[str]) -> None:
    Path(path).write_text(f"STALLED: {where}: {rule}\nrun dirs:\n" + "\n".join(run_dirs) + "\n")


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
