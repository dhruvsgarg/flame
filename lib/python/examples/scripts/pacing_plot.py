#!/usr/bin/env python3
"""F42 C8: what Felix's pacer saw and did, per leg: c and K, its validation-probe signal (1 h windows), test loss and accuracy.

  pacing_plot.py RUN_DIR... [--out PNG]
"""
import argparse
import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def leg(d):
    kc, probes, ev = [], [], []
    with open(glob.glob(os.path.join(d, "telemetry", "aggregator_*.jsonl"))[0], "rb") as fh:
        for line in fh:
            if b'"dynamic_kc"' in line:
                e = json.loads(line)
                kc.append((float(e["now"]) / 3600, e["k"], e["c"]))
            elif b'"val_probe"' in line:
                e = json.loads(line)
                probes.append((float(e["vclock_now"]) / 3600, e["loss"] * e["n"], e["acc"] * e["n"], e["n"]))
            elif b'"agg_eval"' in line or b'"agg_round"' in line:
                e = json.loads(line)
                if e["event"] == "agg_round":
                    last_t = float(e.get("vclock_now") or 0) / 3600
                elif e.get("test-loss") is not None and kc:
                    ev.append((kc[-1][0], e["test-loss"], 100 * e["test-accuracy"]))
    hours = {}
    for t, l, a, n in probes:
        h = hours.setdefault(int(t), [0.0, 0.0, 0])
        h[0] += l; h[1] += a; h[2] += n
    sig = [(h + 0.5, v[0] / v[2], v[2]) for h, v in sorted(hours.items()) if v[2]]
    return kc, sig, ev


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default="pacing.png")
    a = ap.parse_args(argv)
    fig, ax = plt.subplots(3, len(a.runs), figsize=(4.6 * len(a.runs), 9), squeeze=False, sharex=True)
    for j, d in enumerate(a.runs):
        kc, sig, ev = leg(d)
        name = os.path.basename(d.rstrip("/")).split("_dbg_")[0].split("_", 3)[-1]
        ax[0, j].step([x[0] for x in kc], [x[2] for x in kc], where="post", color="#2a78d6", lw=2, label="c (in flight)")
        ax[0, j].step([x[0] for x in kc], [x[1] for x in kc], where="post", color="#eb6834", lw=1.4, label="K (buffer)")
        ax[0, j].set_title(name, fontsize=10)
        ax[0, j].set_ylim(0, 17)
        ax[1, j].plot([s[0] for s in sig], [s[1] for s in sig], "o-", color="#6250d6", ms=4, lw=1.6, label="probe loss (1 h pool)")
        for s in sig:
            ax[1, j].annotate(str(s[2]), (s[0], s[1]), fontsize=6, color="#52514e", xytext=(0, 4), textcoords="offset points", ha="center")
        ax[2, j].plot([e[0] for e in ev], [e[1] for e in ev], color="#e34948", lw=1.4, label="test loss")
        tw = ax[2, j].twinx()
        tw.plot([e[0] for e in ev], [e[2] for e in ev], color="#1baf7a", lw=1.4, label="test acc %")
        tw.set_ylim(0, 60)
        for r in range(3):
            ax[r, j].grid(alpha=0.25, lw=0.6)
            ax[r, j].spines[["top"]].set_visible(False)
        ax[2, j].set_xlabel("hours (sim clock)", fontsize=9)
    ax[0, 0].set_ylabel("c, K"); ax[1, 0].set_ylabel("held-out probe loss\n(label: samples / h)"); ax[2, 0].set_ylabel("test loss")
    h0, l0 = ax[0, 0].get_legend_handles_labels()
    fig.legend(h0 + [ax[1, 0].lines[0], ax[2, 0].lines[0], tw.lines[0]], l0 + ["probe loss (1 h pool)", "test loss", "test acc % (right)"],
               loc="lower center", ncol=5, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(a.out, dpi=100)
    print(a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
