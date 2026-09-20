#!/usr/bin/env python3
"""Phase 1 / B-1 step 2 (fl_fwd_ft_buildplan.md §3): clone a real aggregator
config and repoint it at yahoo / yelp-p via dataset_registry, for
probe_inflation_damage.py --config. Rung-2 offline rig -- reads production
config JSON only, never modifies aggregator/trainer/yaml.

    python prep_b1_configs.py --base <aggregator_config.json> --out-dir DIR

Writes DIR/agnews.json (byte-identical hyperparameters block to --base, dataset
keys added explicitly for clarity), DIR/yahoo.json, DIR/yelp-p.json.
"""
import argparse
import copy
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "..", "..", ".."))
from examples.fwdllm.expts import dataset_registry as dsreg  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="an existing agnews aggregator_config.json")
    ap.add_argument("--out-dir", default="/tmp/b1_configs")
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    base = json.load(open(a.base))

    for name in dsreg.names():
        cfg = copy.deepcopy(base)
        cfg["hyperparameters"].update(dsreg.hyperparameter_overrides(name))
        out = os.path.join(a.out_dir, f"{name}.json")
        json.dump(cfg, open(out, "w"), indent=2)
        spec = dsreg.get(name)
        for rf in (16, 32, 64):
            print(f"{name:8s} rf={rf:<3d} p={spec.probe_dim(rf):>7,d}", end="  ")
        print(f" -> {out}")


if __name__ == "__main__":
    main()
