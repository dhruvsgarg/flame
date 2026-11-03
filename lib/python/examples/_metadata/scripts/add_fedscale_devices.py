#!/usr/bin/env python3
"""FX-D197: give every registry trainer its FedScale device (compute ms/sample, bandwidth kbps).

  add_fedscale_devices.py [--capacity third_party/REFL/dataset/data/device_info/client_device_capacity]

Trainer k gets FedScale device k (REFL/FedScale `mapped_id = clientId % len(profiles)`). Fields are inserted after `task_id`,
so the rest of the registry is untouched; re-running rewrites them in place.
"""
import argparse
import os
import pickle
import re

HERE = os.path.dirname(os.path.abspath(__file__))
REG = os.path.join(HERE, "..", "trainer_registry.yaml")
CAP = os.path.join(HERE, "..", "..", "..", "..", "..", "third_party", "REFL", "dataset", "data", "device_info",
                   "client_device_capacity")
FIELDS = ("fedscale_device_id", "compute_ms_per_sample", "bandwidth_kbps")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capacity", default=CAP)
    a = ap.parse_args(argv)
    with open(a.capacity, "rb") as fh:
        cap = pickle.load(fh)
    out, tid = [], None
    for line in open(REG):
        if any(line.strip().startswith(f + ":") for f in FIELDS):
            continue  # rewritten below
        out.append(line)
        if (m := re.match(r"\s+trainer_id:\s*(\d+)", line)):
            tid = int(m.group(1))
        if re.match(r"\s+task_id:", line) and tid is not None:
            ind = line[: len(line) - len(line.lstrip())]
            dev = tid % len(cap)
            out += [f"{ind}fedscale_device_id: {dev}\n",
                    f"{ind}compute_ms_per_sample: {float(cap[dev]['computation'])}\n",
                    f"{ind}bandwidth_kbps: {float(cap[dev]['communication'])}\n"]
    with open(REG, "w") as fh:
        fh.writelines(out)


if __name__ == "__main__":
    main()
