#!/usr/bin/env python3
"""FedDance sprint F5: export runs into the EuroSys'26 notebook's inputs (`eurosys26_expts/plots/all_plots.ipynb`).

  eurosys_export.py --out fd.json --target 0.7 RUN_DIR [RUN_DIR ...]

Per run: `acc` = [(t_s, acc %)] on the run's own clock (sim vclock, real wall since trace_origin; accuracy_table.curve),
`comm` = the notebook's four counts (parse_communication_overhead semantics: agg_train_sent, agg_eval_sent from selection
picks; agg_weight_recv, agg_eval_recv from [MSG_ARRIVAL] sizes), `tta_h` at --target, and `e2e` = {time, total_comm,
eval_comm} with the notebook's 1 MB per model message / 0.01 MB per eval reply.
"""
import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from accuracy_table import curve, dataset_of  # noqa: E402

MODEL_MSG_MIN_BYTES = 100_000  # an eval reply is a few hundred bytes; any model is > 1 MB


def comm_counts(run_dir):
    c = {"agg_train_sent": 0, "agg_eval_sent": 0, "agg_weight_recv": 0, "agg_eval_recv": 0}
    for f in glob.glob(os.path.join(run_dir, "telemetry", "aggregator_*.jsonl")):
        with open(f, "rb") as fh:
            for line in fh:
                if b'"selection"' in line:  # both modes log picks; sim has no dispatch events
                    e = json.loads(line)
                    if e.get("event") == "selection":
                        c["agg_eval_sent" if e.get("task") == "eval" else "agg_train_sent"] += len(e.get("chosen") or [])
    size = re.compile(r"\[MSG_ARRIVAL\] Putting message .*msg_size=(\d+)")
    for f in glob.glob(os.path.join(run_dir, "*aggregator.log")):
        with open(f, errors="replace") as fh:
            for line in fh:
                m = size.search(line)
                if m:
                    c["agg_weight_recv" if int(m.group(1)) >= MODEL_MSG_MIN_BYTES else "agg_eval_recv"] += 1
    return c


def export(run_dir, target):
    pts, end = curve(run_dir)
    acc = [(t, 100 * a) for t, _, a in pts]
    hit = next((t for t, a in acc if a >= 100 * target), None)
    c = comm_counts(run_dir)
    if hit is not None:  # the paper's e2e counts traffic up to time-to-accuracy; scale by the clock share
        share = hit / end if end else 1.0
    else:
        share = 1.0
    total_mb = (c["agg_train_sent"] + c["agg_eval_sent"] + c["agg_weight_recv"] + 0.01 * c["agg_eval_recv"]) * share
    eval_mb = (c["agg_eval_sent"] + 0.01 * c["agg_eval_recv"]) * share
    return {"run_dir": run_dir, "dataset": dataset_of(run_dir), "mode": "sim" if run_dir.rstrip("/").endswith("_sim") else "real",
            "acc": acc, "end_s": end, "comm": c, "tta_h": hit / 3600 if hit is not None else None,
            "e2e": {"time": hit / 3600 if hit is not None else None, "total_comm": total_mb / 1024, "eval_comm": eval_mb / 1024}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--target", type=float, default=0.7, help="accuracy fraction for time-to-accuracy (paper cifar: 0.7)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rows = [export(r, a.target) for r in a.runs]
    for r in rows:
        print(f"{os.path.basename(r['run_dir'].rstrip('/'))}: max {max((x for _, x in r['acc']), default=0):.1f}% "
              f"tta {r['tta_h'] if r['tta_h'] is None else round(r['tta_h'], 2)} h comm {r['comm']} e2e {r['e2e']}")
    with open(a.out, "w") as fh:
        json.dump(rows, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
