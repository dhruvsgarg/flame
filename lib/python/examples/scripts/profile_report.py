#!/usr/bin/env python3
"""FX-N77: rank where each role's time goes, from the py-spy profiles a leg wrote (`FLAME_PYSPY`, flame/telemetry/profiler.py).

Usage: profile_report.py <run_dir> [--top 25] [--rate 50] [--view wall|cpu] [--match REGEX] [--tree FRAME --depth 3]
Per role (aggregator; trainers averaged per process) and view: seconds per thread, then the top functions by inclusive and by
self time, descending. `wall` includes idle samples (where time goes); `cpu` only on-CPU ones (what costs).
"""

from __future__ import annotations

import argparse
import collections
import re
from pathlib import Path

_FRAME = re.compile(r"^(?P<fn>.*?) \((?P<file>[^():]+?)(?::\d+)?\)$")


def thread_class(label: str) -> str:
    """'thread (123): Thread-7 (run_forever)' -> 'Thread (run_forever)': same-target threads pool together."""
    name = label.split(": ", 1)[-1]
    return re.sub(r"Thread-\d+", "Thread", name)


def frame_key(frame: str) -> str:
    """'fn (path/to/file.py:12)' -> 'file.py:fn' (lines merged)."""
    m = _FRAME.match(frame.strip())
    return f"{Path(m['file']).name}:{m['fn']}" if m else frame.strip()


def parse(path: Path):
    """Yield (thread class, [frame keys root->leaf], samples) per collapsed-stack line."""
    for line in path.read_text(errors="ignore").splitlines():
        stack, _, n = line.rpartition(" ")
        if not stack or not n.isdigit():
            continue
        parts = stack.split(";")
        yield thread_class(parts[0]), [frame_key(f) for f in parts[1:]], int(n)


def role_of(path: Path) -> tuple:
    stem, view = path.name[: -len(".txt")].rsplit(".", 1)
    return ("aggregator" if stem.startswith("aggregator") else stem.split("_", 1)[0]), view


def report(files, rate: float, top: int, match) -> str:
    out = []
    by = collections.defaultdict(list)
    for f in files:
        by[role_of(f)].append(f)
    for (role, view), fs in sorted(by.items()):
        threads, incl, self_ = collections.Counter(), collections.Counter(), collections.Counter()
        for f in fs:
            for th, frames, n in parse(f):
                threads[th] += n
                if not frames or (match and not any(match.search(x) for x in frames)):
                    continue
                for k in dict.fromkeys(frames):  # inclusive: once per stack
                    incl[(th, k)] += n
                self_[(th, frames[-1])] += n
        per = len(fs) * rate  # seconds per process
        out.append(f"\n=== {role} · {view} · {len(fs)} process(es) · seconds per process")
        out.append("threads: " + ", ".join(f"{t} {c / per:.1f}" for t, c in threads.most_common(12)))
        for title, cnt in (("inclusive", incl), ("self", self_)):
            out.append(f"-- top {top} {title}")
            for (th, k), c in cnt.most_common(top):
                out.append(f"{c / per:9.2f}s  {100 * c / max(1, threads[th]):5.1f}%  [{th[:28]}] {k}")
    return "\n".join(out)


def tree(files, rate: float, root: str, depth: int, min_s: float) -> str:
    """Callee tree under every frame matching `root` (e.g. 'trainer.py:_send_weights'), inclusive seconds per process."""
    out = []
    by = collections.defaultdict(list)
    for f in files:
        by[role_of(f)].append(f)
    for (role, view), fs in sorted(by.items()):
        node = collections.Counter()
        for f in fs:
            for th, frames, n in parse(f):
                i = next((i for i, k in enumerate(frames) if root in k), None)
                if i is None:
                    continue
                path = tuple(frames[i:i + depth + 1])
                for d in range(1, len(path) + 1):
                    node[(th,) + path[:d]] += n
        if not node:
            continue
        per = len(fs) * rate
        out.append(f"\n=== {role} · {view} · tree under {root} · seconds per process")
        def walk(prefix):
            kids = sorted((k for k in node if len(k) == len(prefix) + 1 and k[:len(prefix)] == prefix),
                          key=lambda k: -node[k])
            for k in kids:
                if node[k] / per >= min_s:
                    out.append(f"{node[k] / per:9.2f}s  {'  ' * (len(k) - 2)}{k[-1]}" + (f"  [{k[0][:24]}]" if len(k) == 2 else ""))
                    walk(k)
        for top in sorted({k[:2] for k in node if len(k) == 2}, key=lambda k: -node[k]):
            out.append(f"{node[top] / per:9.2f}s  {top[1]}  [{top[0][:24]}]")
            walk(top)
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--rate", type=float, default=50.0)
    ap.add_argument("--view", default="")
    ap.add_argument("--match", default="", help="keep only stacks with a frame matching this regex")
    ap.add_argument("--tree", default="", help="print the callee tree under frames containing this text")
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--min-s", type=float, default=0.2, help="tree: hide nodes below this many seconds")
    a = ap.parse_args(argv)
    files = sorted(p for p in a.run_dir.rglob("*.txt")
                   if p.parent.name == "profile" and (not a.view or p.name.endswith(f".{a.view}.txt")))
    if not files:
        raise SystemExit(f"no profiles under {a.run_dir}")
    print(tree(files, a.rate, a.tree, a.depth, a.min_s) if a.tree
          else report(files, a.rate, a.top, re.compile(a.match) if a.match else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
