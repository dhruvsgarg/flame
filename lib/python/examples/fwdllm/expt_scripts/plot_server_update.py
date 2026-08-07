# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Plot the I-1 audit: applied-update size against the accuracy it produces.

EXPTS_CHARTER Issue I-1 says the round-boundary collapse (84% -> 25%) is an
undamped forward-gradient optimizer applying each noisy commit raw. If that is
right, `update_ratio` = ||delta|| / ||w|| climbs BEFORE accuracy falls, and the
climb is visible at the boundary. If accuracy falls while the ratio stays flat,
the step size is not the mechanism and I-1 needs a different root.

Needs `server_update_audit: true` in the aggregator's config_overrides.

    python plot_server_update.py <run_dir> [--out <png>]
"""
import argparse
import glob
import json
import os
import sys


def load(run_dir):
    """(server_update records, agg_eval records) from a run's aggregator log."""
    paths = glob.glob(os.path.join(run_dir, "telemetry", "aggregator_*.jsonl"))
    if not paths:
        return [], []
    upd, ev = [], []
    with open(paths[0]) as fh:
        for line in fh:
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("event") == "server_update":
                upd.append(e)
            elif e.get("event") == "agg_eval":
                ev.append(e)
    return upd, ev


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    upd, ev = load(args.run_dir)
    if not upd:
        print("no `server_update` records -- this run was launched without "
              "`server_update_audit: true` in the aggregator config_overrides")
        return 1

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t0 = min(e["ts"] for e in upd + ev)
    uh = [(e["ts"] - t0) / 3600.0 for e in upd]
    ratio = [e.get("update_ratio") for e in upd]
    rounds = sorted({e.get("round") for e in upd if e.get("round") is not None})

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(uh, ratio, lw=0.6, alpha=0.75, color="#3b6ea5",
            label=r"$\|\Delta\| / \|w\|$ per commit")
    ax.set_yscale("log")
    ax.set_xlabel("wall time (h)")
    ax.set_ylabel(r"update ratio $\|\Delta\|/\|w\|$ (log)")
    ax.set_title(f"I-1 audit — {os.path.basename(args.run_dir)}")

    # Round boundaries: the instant I-1 says the collapse happens.
    for r in rounds[1:]:
        first = next((e for e in upd if e.get("round") == r), None)
        if first:
            ax.axvline((first["ts"] - t0) / 3600.0, color="#b04040", ls="--",
                       lw=1.0, alpha=0.8)
            ax.text((first["ts"] - t0) / 3600.0, ax.get_ylim()[1],
                    f" round {r}", color="#b04040", va="top", fontsize=8)

    if ev:
        ax2 = ax.twinx()
        ax2.plot([(e["ts"] - t0) / 3600.0 for e in ev],
                 [100 * e["test-accuracy"] for e in ev],
                 color="#2e8b57", lw=1.4, label="test accuracy")
        ax2.set_ylabel("test accuracy (%)", color="#2e8b57")
        ax2.tick_params(axis="y", labelcolor="#2e8b57")

    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    out = args.out or os.path.join(args.run_dir, "plots", "server_update.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"wrote {out}  ({len(upd)} commits, {len(ev)} evals, rounds {rounds})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
