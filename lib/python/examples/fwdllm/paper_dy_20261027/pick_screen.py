#!/usr/bin/env python3
"""Pick the fwdllm screen winner: best eval accuracy up to the vclock both legs reached.

  pick_screen.py <POOL dir> <phase A> <phase B>   -> prints the winning phase; per-leg detail on stderr
"""
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "expt_scripts"))
from plotlib import reducers as R  # noqa: E402

pool, phases = sys.argv[1], sys.argv[2:]
curves = {}
for ph in phases:
    runs = sorted(glob.glob(os.path.join(pool, "**", ph, "runs", "*", "run_*"), recursive=True))
    rr = R.load_run(runs[-1], post_peak_grace_s=None) if runs else None
    hrs, acc, _, _ = rr.learning_curve() if rr else ([], [], [], [])
    pts = [(h, a) for h, a in zip(hrs, acc) if a is not None]
    if not pts:
        sys.exit(f"{ph}: no evals ({runs[-1] if runs else 'no run dir'})")
    curves[ph] = pts
horizon = min(p[-1][0] for p in curves.values())
score = {ph: max(a for h, a in p if h <= horizon) for ph, p in curves.items()}
for ph in phases:
    print(f"{ph}: best {score[ph]:.2f}% by {horizon:.2f} vclock h ({len(curves[ph])} evals)", file=sys.stderr)
print(max(phases, key=score.get))
