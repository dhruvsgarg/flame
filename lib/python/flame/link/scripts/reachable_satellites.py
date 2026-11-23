# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""How many satellites ever reach the ground station, and how many are in view at once.

Use it to size a direct-routing trainer pool: a satellite that never rises above the
elevation mask has no path, so its trainer can never exchange a model.

    python -m flame.link.scripts.reachable_satellites --masks 10 15 20 25 --num-trainers 200
"""

import argparse

import numpy as np

from flame.link.config import LinkLayerConfig
from flame.link.visibility import visibility_matrix


def main() -> None:
    cfg = LinkLayerConfig.from_dict({})
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ecef", default=cfg.satellite_ecef_path)
    ap.add_argument("--masks", type=float, nargs="+", default=[cfg.min_elevation_deg],
                    help="elevation masks to compare (degrees)")
    ap.add_argument("--num-trainers", type=int, default=200,
                    help="trainer pool = first N satellites (trainer_id - 1)")
    ap.add_argument("--list", action="store_true",
                    help="print the reachable trainer ids for the first mask")
    args = ap.parse_args()

    ecef = np.load(args.ecef)["ecef_km"]
    n_steps, n_sats = ecef.shape[:2]
    n = min(args.num_trainers, n_sats)
    print(f"{n_steps} timesteps, {n_sats} satellites; pool = first {n}")
    print(f"{'mask':>6} {'reachable(all)':>15} {'reachable(pool)':>16} {'in view: mean/min/max (pool)':>30}")
    for mask in args.masks:
        vis = visibility_matrix(ecef, cfg, mask)
        ever = vis.any(0)
        per_step = vis[:, :n].sum(1)
        print(f"{mask:>6.1f} {int(ever.sum()):>15d} {int(ever[:n].sum()):>16d} "
              f"{per_step.mean():>14.1f}/{per_step.min()}/{per_step.max()}")
        if args.list and mask == args.masks[0]:
            print("reachable trainer ids (pool):", [int(i) + 1 for i in np.flatnonzero(ever[:n])])


if __name__ == "__main__":
    main()
