#!/usr/bin/env python3
"""FX-D100: every way our baselines differ from their sources (model, adapted knobs), as the paper's table.

  baseline_deviations.py [--md]   # rows from _metadata/baseline_reference.yaml; Felix is ours and never listed
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from flame.launch import baseline_reference as br  # noqa: E402

HEAD = ("baseline", "dataset", "knob", "source", "ours", "why", "evidence")


def rows(ref: dict) -> list:
    out = []
    for bl, e in ref.items():
        e = br.entry(ref, bl)
        for ds, d in (e or {}).get("datasets", {}).items():
            m = d.get("model", {})
            if m.get("source") not in (None, "ours") and m.get("source") != m.get("ours"):
                out.append((bl, ds, "model", m["source"], m["ours"], "", ""))
            for k, item in d.items():
                if isinstance(item, dict) and "source_v" in item:
                    out.append((bl, ds, k, str(item["source_v"]), str(item["v"]), item["why"], item["evidence"]))
    return [r for r in out if r[0] != "felix"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--md", action="store_true", help="markdown table (default: tab-separated)")
    a = ap.parse_args(argv)
    rs = [HEAD] + rows(br.load())
    for i, r in enumerate(rs):
        print("| " + " | ".join(r) + " |" if a.md else "\t".join(r))
        if a.md and i == 0:
            print("|" + "---|" * len(HEAD))
    return 0


if __name__ == "__main__":
    sys.exit(main())
