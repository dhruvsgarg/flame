#!/usr/bin/env python3
"""H14: why is sim's variance at each bin's first aggregation ~7% below real's?

Everything summary-level already matches between the two modes — pool size,
staleness, aggregation rate, contributor concentration, and raw grad norm to
0.09%. Variance is DISPERSION, not magnitude, so the remaining question is how
the individual contributions are spread within the pool, which needs the
per-contribution norms `var_calc` carries (`var_calc_audit: true` on BOTH legs).

Reads only telemetry. Pairs cycles at identical (bin ordinal, iteration) so the
shared training curve cancels (§D-38), and reports the reduction's inputs beside
its output: if the inputs match and the output does not, the divergence is in
the reduction; if the inputs are less dispersed in sim, it is upstream in how
the pool is assembled.
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics as st
import sys


def cycles(run_dir: str, max_bins: int) -> dict:
    """{(bin ordinal, iteration): var_calc record} for one leg."""
    paths = glob.glob(f"{run_dir}/telemetry/aggregator_*.jsonl")
    if not paths:
        return {}
    seen, out = {}, {}
    for line in open(paths[0], errors="replace"):
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if e.get("event") != "var_calc":
            continue
        key = (e.get("round") or 0, e.get("data_id"))
        if key not in seen:
            seen[key] = len(seen)
        b = seen[key]
        if b >= max_bins:
            continue
        out[(b, e.get("iteration_per_data_id"))] = e
    return out


def dispersion(norms: list) -> tuple:
    """Spread of the pool, scale-free — the quantity `var` actually reacts to."""
    if not norms or len(norms) < 2:
        return float("nan"), float("nan")
    m = st.mean(norms)
    return st.pstdev(norms), (st.pstdev(norms) / m if m else float("nan"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("real_run")
    ap.add_argument("sim_run")
    ap.add_argument("--max-bins", type=int, default=196)
    a = ap.parse_args(argv)

    r, s = cycles(a.real_run, a.max_bins), cycles(a.sim_run, a.max_bins)
    if not r or not s:
        print("no `var_calc` records — relaunch both legs with "
              "`var_calc_audit: true` in the AGGREGATOR config_overrides")
        return 1

    both = sorted(set(r) & set(s))
    print(f"paired cycles at identical (bin, iteration): {len(both)}\n")
    rows = []
    for k in both:
        rn, sn = r[k].get("input_grad_norms") or [], s[k].get("input_grad_norms") or []
        rows.append((k, rn, sn, r[k].get("output_var"), s[k].get("output_var")))

    def summarize(sel, label):
        sub = [x for x in rows if sel(x[0])]
        if not sub:
            return
        rl = [len(x[1]) for x in sub]
        sl = [len(x[2]) for x in sub]
        rm = [st.mean(x[1]) for x in sub if x[1]]
        sm = [st.mean(x[2]) for x in sub if x[2]]
        rc = [dispersion(x[1])[1] for x in sub if len(x[1]) > 1]
        sc = [dispersion(x[2])[1] for x in sub if len(x[2]) > 1]
        rv = [x[3] for x in sub if x[3] is not None]
        sv = [x[4] for x in sub if x[4] is not None]

        def rel(x, y):
            return (st.mean(y) - st.mean(x)) / st.mean(x) if x and y and st.mean(x) else float("nan")
        print(f"--- {label}  (n={len(sub)})")
        print(f"    pool entries   real={st.mean(rl):.2f}  sim={st.mean(sl):.2f}  ({rel(rl, sl):+.2%})")
        print(f"    mean norm      real={st.mean(rm):.4f}  sim={st.mean(sm):.4f}  ({rel(rm, sm):+.2%})")
        print(f"    CoV of pool    real={st.mean(rc):.4f}  sim={st.mean(sc):.4f}  ({rel(rc, sc):+.2%})"
              "   <- dispersion: the term var reacts to")
        print(f"    output var     real={st.mean(rv):.4f}  sim={st.mean(sv):.4f}  ({rel(rv, sv):+.2%})")

    summarize(lambda k: k[1] == 0, "iteration 0 (the gate-setting cycle, §D-34)")
    summarize(lambda k: True, "all iterations")

    # Same inputs, different output => the reduction. Different inputs => upstream.
    same_in = [x for x in rows if x[1] and x[2] and len(x[1]) == len(x[2])
               and all(abs(p - q) <= 1e-6 * max(abs(p), 1.0) for p, q in zip(sorted(x[1]), sorted(x[2])))]
    print(f"\ncycles whose INPUT norms match bit-close: {len(same_in)}/{len(rows)}")
    if same_in:
        d = [(x[4] - x[3]) / x[3] for x in same_in if x[3]]
        if d:
            print(f"  ...of those, output var differs by median {st.median(d):+.3%} "
                  "(non-zero => the reduction itself diverges)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
