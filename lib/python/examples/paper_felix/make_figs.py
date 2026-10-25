#!/usr/bin/env python3
"""Regenerate every Felix paper figure (§1-§5) into figs/paper/<section>/ (PDF + PNG); see README.md for the index.

  make_figs.py [--only s3 ...] [--refresh]     # --refresh re-reads run telemetry (else data/runs/ cache)

FX-N147's trace data comes from `examples/scripts/eligibility_forecast.py` (writes data/fx_n147_*.json; rerun it to refresh).
"""
import argparse
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import extract  # noqa: E402
import sec1_intro  # noqa: E402
import sec3_motivation  # noqa: E402
import sec4_overview  # noqa: E402
import sec5_design  # noqa: E402

SECTIONS = {"s1": sec1_intro, "s3": sec3_motivation, "s4": sec4_overview, "s5": sec5_design}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", choices=list(SECTIONS))
    ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args(argv)
    if a.refresh:
        extract.load([d for c in extract.CELLS for d in extract.f49(c).values()] +
                     [extract.run_dir(extract.F49, "F49_smu_fn", "felix")], refresh=True)
    bad = 0
    for key in a.only or SECTIONS:
        for f in SECTIONS[key].FIGS:
            try:
                print(f"{key} {f.__name__}: {f()}")
            except Exception:  # one broken figure must not stop the rest
                bad += 1
                print(f"{key} {f.__name__}: FAILED\n{traceback.format_exc()}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
