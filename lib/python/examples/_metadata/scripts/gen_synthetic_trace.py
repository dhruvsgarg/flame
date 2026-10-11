#!/usr/bin/env python3
# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Generate a synthetic availability trace into availability_traces/synthetic_traces.yaml (FX-N35).

Same family as syn_10/syn_50: a per-trainer two-state chain on `--slot-s` slots, outages in slot multiples,
a final UN_AVL at `--horizon-s`. Each trainer's first state is drawn from the chain's stationary law, so the
unavailable fraction is `--unavail` at every run length.

  gen_synthetic_trace.py --name syn_20 --unavail 0.2 --insert-before syn_50
  gen_synthetic_trace.py --extend syn_10 syn_20 syn_50 --horizon-s 536400   # to the mobiperf span (149 h)

--extend keeps each trainer's events before its day end (the old end-of-trace UN_AVL marker) and continues
the same chain from the state there, with outage/up means measured from that trace's own first day.
"""

import argparse
import os
import random
import statistics
from pathlib import Path

import yaml

TRACES = Path(__file__).resolve().parent.parent / "availability_traces" / "synthetic_traces.yaml"


def trainer_events(rng: random.Random, f: float, mean_out_s: float, slot_s: int, horizon_s: int) -> list:
    """[[ts, state], ...] for one trainer; state changes only."""
    p_end_out = slot_s / mean_out_s                  # UN_AVL -> AVL_TRAIN per slot
    p_end_up = p_end_out * f / (1.0 - f)             # AVL_TRAIN -> UN_AVL per slot (stationary fraction f)
    down = rng.random() < f
    ev = [[0, "UN_AVL" if down else "AVL_TRAIN"]]
    for t in range(slot_s, horizon_s, slot_s):
        if rng.random() < (p_end_out if down else p_end_up):
            down = not down
            ev.append([t, "UN_AVL" if down else "AVL_TRAIN"])
    if ev[-1][1] != "UN_AVL":
        ev.append([horizon_s, "UN_AVL"])
    return ev


def extend_events(rng: random.Random, ev: list, day_end: float, p_end_out: float, p_end_up: float,
                  slot_s: int, horizon_s: int) -> list:
    """Events before day_end, then the chain continued from the state there to horizon_s (final UN_AVL marker)."""
    keep = [e for e in ev if e[0] < day_end]
    down = keep[-1][1] == "UN_AVL"
    start = int(-(-day_end // slot_s) * slot_s)
    for t in range(start, horizon_s, slot_s):
        if rng.random() < (p_end_out if down else p_end_up):
            down = not down
            keep.append([t, "UN_AVL" if down else "AVL_TRAIN"])
    if keep[-1][1] != "UN_AVL":
        keep.append([horizon_s, "UN_AVL"])
    return keep


def _means(per_trainer: dict, day_end: float) -> tuple:
    """(mean outage, mean up) in s over complete periods before day_end."""
    out, up = [], []
    for ev in per_trainer.values():
        ev = [e for e in ev if e[0] < day_end]
        for (t0, s), (t1, _) in zip(ev, ev[1:]):
            (out if s == "UN_AVL" else up).append(t1 - t0)
    return statistics.mean(out), statistics.mean(up)


def extend(names: list, horizon_s: int, slot_s: int, seed: int) -> None:
    d = yaml.safe_load(TRACES.read_text())
    for name in names:
        e = d["traces"][name]
        per = e["per_trainer"]["n300"]
        day_end = max(ev[-1][0] for ev in per.values())  # the end-of-trace marker
        m_out, m_up = _means(per, day_end)
        rng = random.Random(f"{seed}|{name}")
        ext = lambda ev: extend_events(rng, ev, day_end, slot_s / m_out, slot_s / m_up, slot_s, horizon_s)
        e["per_trainer"]["n300"] = {k: ext(v) for k, v in per.items()}
        e["pattern"] = ext(e["pattern"])
        e["description"] = (f"{e['description']}; extended {day_end:.0f}s -> {horizon_s}s (same chain: mean outage "
                            f"{m_out:.0f}s, up {m_up:.0f}s)")
        print(f"{name}: day end {day_end:.0f}s, mean outage {m_out:.0f}s up {m_up:.0f}s -> {horizon_s}s")
    tmp = TRACES.with_suffix(".yaml.tmp")
    tmp.write_text(yaml.safe_dump(d, default_flow_style=False, sort_keys=False, width=4096))
    os.replace(tmp, TRACES)  # readers never see a partial file


def _block(name: str, desc: str, pattern: list, per_trainer: dict) -> str:
    """The trace as text in the file's own layout (2-space keys, `- - ts` pairs)."""
    def pairs(ev, ind):
        return "".join(f"{ind}- - {t}\n{ind}  - {s}\n" for t, s in ev)
    out = f"  {name}:\n    description: {desc}\n    pattern:\n" + pairs(pattern, "    ")
    out += "    per_trainer:\n      n300:\n"
    for k, ev in per_trainer.items():
        out += f"        {k}:\n" + pairs(ev, "        ")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--extend", nargs="+", default=[], help="existing traces to extend to --horizon-s")
    ap.add_argument("--name")
    ap.add_argument("--unavail", type=float, help="stationary unavailable fraction")
    ap.add_argument("--mean-outage-s", type=float, default=672.0, help="syn_10's mean outage (600 s slots)")
    ap.add_argument("--slot-s", type=int, default=600)
    ap.add_argument("--horizon-s", type=int, default=86400)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--insert-before", help="existing trace key the new block goes above")
    a = ap.parse_args()
    if a.extend:
        extend(a.extend, a.horizon_s, a.slot_s, a.seed)
        return
    rng = random.Random(a.seed)
    per = {f"trainer_{i:03d}": trainer_events(rng, a.unavail, a.mean_outage_s, a.slot_s, a.horizon_s)
           for i in range(1, a.n + 1)}
    desc = (f"{a.unavail:.0%} unavailability (stationary start; gen_synthetic_trace.py seed {a.seed}, "
            f"mean outage {a.mean_outage_s:.0f}s)")
    text = TRACES.read_text()
    if f"\n  {a.name}:\n" in text:
        raise SystemExit(f"{a.name} already in {TRACES.name}")
    anchor = f"\n  {a.insert_before}:\n"
    if text.count(anchor) != 1:
        raise SystemExit(f"anchor {a.insert_before!r} not found once")
    block = _block(a.name, desc, per["trainer_001"], per)
    TRACES.write_text(text.replace(anchor, "\n" + block + anchor[1:], 1))
    print(f"wrote {a.name} ({a.n} trainers) above {a.insert_before}")


if __name__ == "__main__":
    main()
