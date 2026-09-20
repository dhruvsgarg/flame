#!/usr/bin/env python3
"""Row N5a: is `FWDLLM_FD_SCALE_INVARIANT` holding the right quantity fixed?

    ./audit_fd_chord.py RUN_DIR [RUN_DIR ...]

The flag rescales `h` so the ABSOLUTE finite-difference chord `h*sqrt(p)` stays
at 6.7107 across the `rf` ladder (h = 0.01 / 0.014023 / 0.019507 at rf 16/32/64,
[P9.1]). But `||theta_tr||` falls 13.35 -> 9.6 -> 6.86 over the same ladder, so
the DIMENSIONLESS chord

    h*sqrt(p) / ||theta_tr||

is what actually says how far along theta the probe steps -- and if it moves, the
one knob that exists to enforce scale invariance is holding the united quantity
fixed while letting the ratio nearly double (§5.3's ratio principle, violated by
its own enforcement mechanism).

**Predicted:** 0.50 / 0.70 / 0.98 at rf 16 / 32 / 64.
**Falsified if** the logs show the ratio constant -- then the flag is already
right and only the docs are wrong.

Reads two lines per run and nothing else: `[FD] spacing:` from the trainers log
(p, h, h*sqrt(p) as actually enacted, not as configured) and the first
`[ServerStep] ... ||theta_tr||=` from the aggregator log, which is the norm at
theta_0 before any step lands. `--at-commit` reads a later commit instead, for
the ratio as it drifts through a run.
"""
import argparse
import glob
import os
import re
import sys

FD = re.compile(r"\[FD\] spacing: p=(\d+) h=([\d.eE+-]+) h\*sqrt\(p\)=([\d.]+)")
STEP = re.compile(r"\[ServerStep\].*commit=(\d+).*\|\|theta_tr\|\|=([\d.eE+-]+)")
LADDER = {16: 0.50, 32: 0.70, 64: 0.98}     # what §5.8 predicts, by rf


def first_match(paths, pat, at_commit=None):
    for f in sorted(paths):
        try:
            with open(f, "rb") as fh:
                for raw in fh:
                    m = pat.search(raw.decode("utf-8", "replace"))
                    if not m:
                        continue
                    if at_commit is not None and int(m.group(1)) < at_commit:
                        continue
                    return m
        except OSError:
            continue
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--at-commit", type=int, default=None,
                    help="read ||theta_tr|| at the first commit >= this instead "
                         "of at theta_0")
    a = ap.parse_args()

    print(f"{'run':<52}{'p':>9}{'h':>11}{'h*sqrt(p)':>11}"
          f"{'||th_tr||':>11}{'ratio':>8}")
    print("-" * 102)
    rows = []
    for run in a.runs:
        fd = first_match(glob.glob(os.path.join(run, "*trainers.log")), FD)
        st = first_match(glob.glob(os.path.join(run, "*aggregator.log")), STEP,
                         a.at_commit)
        name = os.path.basename(run.rstrip("/"))[:52]
        if not fd or not st:
            print(f"{name:<52}  missing "
                  f"{'[FD] spacing' if not fd else '[ServerStep] ||theta_tr||'}")
            continue
        p, h, chord = int(fd.group(1)), float(fd.group(2)), float(fd.group(3))
        n = float(st.group(2))
        rows.append((name, p, h, chord, n, chord / n))
        print(f"{name:<52}{p:>9}{h:>11.6g}{chord:>11.4f}{n:>11.4f}"
              f"{chord / n:>8.3f}")

    if len(rows) > 1:
        lo = min(r[5] for r in rows)
        hi = max(r[5] for r in rows)
        print(f"\nrelative chord spans {lo:.3f} to {hi:.3f} -- {hi / lo:.2f}x")
        print("N5a gate: " + (
            "the flag holds the ABSOLUTE chord fixed and lets the relative one "
            "move -- §5.8's reading confirmed" if hi / lo > 1.1 else
            "the relative chord is constant -- the flag is already right and "
            "§5.8's reading is wrong"))
    elif rows:
        r = rows[0]
        near = min(LADDER, key=lambda k: abs(LADDER[k] - r[5]))
        print(f"\none run only. ratio {r[5]:.3f} vs the ladder's predicted "
              f"{LADDER}: closest is rf={near}. Point this at the rf=32 and "
              f"rf=64 runs -- they are node-local -- to close N5a.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
