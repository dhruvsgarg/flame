#!/usr/bin/env python3
"""Headline table (felix_paper_outline "Headline table plan") from figs/expts/results_<dataset>.json.

  make_tables.py [--expts DIR]

Writes into DIR (default figs/expts/): table_headline.json (per stack × cell values), table_headline.tex
(booktabs table*, `\\input` as is; needs booktabs, colortbl, xcolor[table]) and numbers.tex (`\\newcommand` macros for prose numbers).
Per cell, reference = the fastest stack to the target; a hit shows stack h / ref h and stack GB / ref GB, a miss shows
a shaded −Δ pp, Δ = target − peak 5-eval accuracy over the run. Re-run after paper_json.py.
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
EXPTS = os.path.join(HERE, "figs", "expts")
DATASETS = (("cifar10", "CIFAR-10", "Cifar"), ("google_speech", "Google Speech Commands", "Speech"))
CELLS = ("s0u", "s20u", "s50u", "smu")
TRACE_TEX = {"s0u": "Syn-0\\%", "s20u": "Syn-20\\%", "s50u": "Syn-50\\%", "smu": "MobiPerf"}  # Syn-X% = X% synthetic unavailability
BASELINES = ("FedBuff", "Oort", "REFL", "FedDance", "FedAvg")  # each followed by its oracle `*` row
SYNC = ("Oort", "REFL", "FedDance", "FedAvg")


def cell_values(cell):
    """{stack: row dict} for one cell, ratios vs the cell's fastest hit."""
    stacks, tgt = cell["stacks"], round(cell["target"] * 100)
    hits = {n: s for n, s in stacks.items() if s["status"] == "hit"}
    ref = min(hits, key=lambda n: hits[n]["time_h"]) if hits else None
    out = {}
    for n, s in stacks.items():
        r = {"status": s["status"], "kind": s["kind"], "time_h": s["time_h"], "gb": s["gb"], "peak_acc5": s["peak_acc5"],
             "final_acc": s["final_acc"], "ref": n == ref, "time_ratio": None, "gb_ratio": None, "acc_gap_pp": None}
        if s["status"] == "hit" and ref:
            r["time_ratio"] = round(s["time_h"] / hits[ref]["time_h"], 3)
            r["gb_ratio"] = round(s["gb"] / hits[ref]["gb"], 3)
            fv = s.get("felix_vs_this") or {}
            if ref == "Felix" and fv.get("speedup") is not None:  # paper_json's speedup = stack h / Felix h
                assert abs(fv["speedup"] - r["time_ratio"]) < 0.01, (n, fv, r)
        elif s["status"] == "miss":
            r["acc_gap_pp"] = round(max(tgt - s["peak_acc5"], 0.0), 2)
        out[n] = r
    return {"label": cell["label"], "target_pct": tgt, "ref": ref, "stacks": out}


SHADE = ((2, 0), (5, 10), (10, 22), (30, 38), (1e9, 56))  # fxbad tint by gap (pp): < 2 none, < 5, < 10, < 30, ≥ 30


def cell_tex(r, best_gb, worst_gap):
    """Two sub-column entries (h, GB) for one stack × cell; a miss is one shaded −Δ cell, the leg's worst in bold."""
    if r["status"] == "pending":
        return "\\multicolumn{2}{c}{--}"
    if r["status"] == "miss":
        gap = r["acc_gap_pp"]
        tint = next(t for hi, t in SHADE if gap < hi)
        v = f"$-${gap:.1f}"
        if gap == worst_gap:
            v = f"\\textbf{{{v}}}"
        if tint:
            v = f"\\cellcolor{{fxbad!{tint}}}{v}"
        return f"\\multicolumn{{2}}{{c}}{{{v}}}"
    if r["ref"]:
        t, g = f"\\textbf{{{r['time_h']:.1f}}}", f"{r['gb']:.0f}"
    else:
        t, g = f"{r['time_ratio']:.2f}$\\times$", f"{r['gb_ratio']:.2f}$\\times$"
    if best_gb:
        g = f"\\textbf{{{g}}}"
    return f"{t} & {g}"


def rows_order():
    return ["Felix"] + [n for b in BASELINES for n in (b, b + "*")]


def build_tex(data):
    nc = 2 * len(CELLS)
    cols = "@{\\extracolsep{\\fill}}l" + "rr" * len(CELLS) + "|" + "rr" * len(CELLS) + "c"
    hdr1 = " & " + " & ".join(f"\\multicolumn{{{nc}}}{{c{'|' if i == 0 else ''}}}{{\\textbf{{{lab}}}}}"
                              for i, (_, lab, _) in enumerate(DATASETS)) + " & \\\\"
    rule1 = " ".join(f"\\cmidrule(lr){{{2 + i * nc}-{1 + (i + 1) * nc}}}" for i in range(len(DATASETS)))
    hdr2, hdrt, hdr3 = [], [], []
    for ds, _, _ in DATASETS:
        for c in CELLS:
            v = data[ds][c]
            hdr2.append(f"\\multicolumn{{2}}{{c}}{{{TRACE_TEX[c]}}}")
            hdrt.append(f"\\multicolumn{{2}}{{c}}{{{v['target_pct']}\\%}}")
            hdr3.append("\\shortstack{Time\\\\(h)} & \\shortstack{Data\\\\(GB)}")
    rule2 = " ".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(DATASETS) * len(CELLS)))
    body = []
    for n in rows_order():
        label = "\\textbf{\\sys{} (ours)}" if n == "Felix" else (f"\\quad {n[:-1]}$^*$" if n.endswith("*") else n)
        ents, hits = [], 0
        for ds, _, _ in DATASETS:
            for c in CELLS:
                st = data[ds][c]["stacks"]
                gbs = [s["gb"] for s in st.values() if s["status"] == "hit"]
                worst = max((s["acc_gap_pp"] for s in st.values() if s["acc_gap_pp"] is not None), default=0)
                r = st[n]
                hits += r["status"] == "hit"
                ents.append(cell_tex(r, r["status"] == "hit" and r["gb"] == min(gbs), worst))
        reached = f"{hits}/{len(DATASETS) * len(CELLS)}"
        if n == "Felix":
            reached = f"\\textbf{{{reached}}}"
        body.append(("\\rowcolor{fxgood!10}" if n == "Felix" else "") + f"{label} & " + " & ".join(ents)
                    + f" & {reached} \\\\")
        if n == "Felix":
            body.append("\\midrule")
        elif n.endswith("*") and n != BASELINES[-1] + "*":
            body.append("\\addlinespace[3pt]")
    return "\n".join([
        "% generated by paper_felix/make_tables.py from figs/expts/results_*.json; do not edit by hand",
        "% needs \\usepackage{booktabs,colortbl} and \\usepackage[table]{xcolor}",
        "\\providecommand{\\sys}{Felix}",
        "\\providecolor{fxgood}{HTML}{2A78D6}",
        "\\providecolor{fxbad}{HTML}{E34948}",
        "\\begin{table*}[t]",
        "\\centering\\small\\setlength{\\tabcolsep}{2pt}",
        f"\\begin{{tabular*}}{{\\textwidth}}{{{cols}}}",
        "\\toprule", hdr1, rule1,
        "Eligibility trace & " + " & ".join(hdr2) + " & \\\\",
        "Target Acc. & " + " & ".join(hdrt) + " & \\\\", rule2,
        "Stack & " + " & ".join(hdr3) + " & Hits \\\\",
        "\\midrule", *body,
        "\\bottomrule",
        "\\end{tabular*}",
        "\\caption{\\sys{} alone reaches the target on every dataset--trace pair. \\sys{}'s row: hours and GB to target. "
        "Other rows: multiples of \\sys{}'s ($>1\\times$ = worse; bold: least data) or, if missed, target minus peak accuracy "
        "in points (shaded by gap; bold: worst). $^*$: oracle eligibility.}",
        "\\label{tab:headline}",
        "\\end{table*}", "", ""]).rstrip("\n") + "\n"


def rng(vals, fmt):
    return fmt.format(min(vals)), fmt.format(max(vals))


def build_numbers(data):
    m = {}
    allsp, allgb = [], []
    n_first = n_cells = 0
    for ds, _, tag in DATASETS:
        sp, gb, sync, gaps = [], [], [], []
        for c in CELLS:
            v = data[ds][c]
            n_cells += 1
            n_first += v["ref"] == "Felix"
            fb, fx = v["stacks"]["FedBuff"], v["stacks"]["Felix"]
            if fb["status"] == "hit" and fx["status"] == "hit":
                sp.append(fb["time_h"] / fx["time_h"])
                gb.append((fx["gb"] / fb["gb"] - 1) * 100)
            for n, r in v["stacks"].items():
                if n.rstrip("*") in SYNC and r["final_acc"] is not None:
                    sync.append(r["final_acc"])
                if r["acc_gap_pp"] is not None:
                    gaps.append(r["acc_gap_pp"])
        allsp += sp
        allgb += gb
        for key, vals, fmt in (("SpeedupFedBuff", sp, "{:.2f}"), ("GbChangeFedBuff", gb, "{:+.0f}"),
                               ("SyncFinal", sync, "{:.0f}"), ("MissGap", gaps, "{:.1f}")):
            if vals:
                m[f"fx{tag}{key}Min"], m[f"fx{tag}{key}Max"] = rng(vals, fmt)
    m["fxSpeedupFedBuffMin"], m["fxSpeedupFedBuffMax"] = rng(allsp, "{:.1f}")
    m["fxGbChangeFedBuffMin"], m["fxGbChangeFedBuffMax"] = rng(allgb, "{:+.0f}")
    m["fxCellsFirst"], m["fxCellsTotal"] = str(n_first), str(n_cells)
    lines = ["% generated by paper_felix/make_tables.py; Speedup = FedBuff h / Felix h over cells both hit; GbChange = Felix GB "
             "vs FedBuff (%); SyncFinal = sync stacks' (incl. *) final accuracy (%); MissGap = target - peak, misses (pp)"]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in m.items()]
    return "\n".join(lines) + "\n", m


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--expts", default=EXPTS)
    a = ap.parse_args(argv)
    data, meta = {}, {}
    for ds, _, _ in DATASETS:
        res = json.load(open(os.path.join(a.expts, f"results_{ds}.json")))
        meta[ds] = {k: res["meta"][k] for k in ("generated", "seeds", "complete", "code_key")}
        data[ds] = {c: cell_values(res["cells"][c]) for c in CELLS}
    tex = build_tex(data)
    nums, macros = build_numbers(data)
    out = {"meta": meta, "rows": rows_order(), "cells": data, "macros": macros}
    with open(os.path.join(a.expts, "table_headline.json"), "w") as f:
        json.dump(out, f, indent=1)
    for name, s in (("table_headline.tex", tex), ("numbers.tex", nums)):
        with open(os.path.join(a.expts, name), "w") as f:
            f.write(s)
    print(f"wrote table_headline.{{json,tex}}, numbers.tex -> {a.expts}")


if __name__ == "__main__":
    main()
