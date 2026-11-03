#!/usr/bin/env python3
# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Generate a synthetic availability trace into availability_traces/synthetic_traces.yaml (FX-N35).

Same family as syn_10/syn_50: a per-trainer two-state chain on `--slot-s` slots, outages in slot multiples,
a final UN_AVL at `--horizon-s`. Each trainer's first state is drawn from the chain's stationary law, so the
unavailable fraction is `--unavail` at every run length.

  gen_synthetic_trace.py --name syn_20 --unavail 0.2 --insert-before syn_50
  gen_synthetic_trace.py --extend syn_10 syn_20 syn_50 --horizon-s 536400   # to the mobiperf span (149 h)
  gen_synthetic_trace.py --three-state syn_50 --like mobiperf_3st_50       # writes syn_50_3st (FX-N96)

The 2-state syn_10/syn_20/syn_50 were deleted once their _3st twins were verified (operator 10-14); the commands
above that name them need the 2-state file from git history (`git show b71ec9664:<path>`).

--extend keeps each trainer's events before its day end (the old end-of-trace UN_AVL marker) and continues
the same chain from the state there, with outage/up means measured from that trace's own first day.
"""

import argparse
import json
import os
import random
import re
import statistics
from collections import Counter
from pathlib import Path

import yaml

TRACES = Path(__file__).resolve().parent.parent / "availability_traces" / "synthetic_traces.yaml"
MOBIPERF = TRACES.with_name("mobiperf_traces.yaml")
TRAIN, EVAL, UN = "AVL_TRAIN", "AVL_EVAL", "UN_AVL"


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
    out = f"  {name}:\n    description: {json.dumps(desc)}\n    pattern:\n" + pairs(pattern, "    ")
    out += "    per_trainer:\n      n300:\n"
    for k, ev in per_trainer.items():
        out += f"        {k}:\n" + pairs(ev, "        ")
    return out



# ── FX-N96: 3-state traces ──────────────────────────────────────────────────────────────────────────────────

def _spans(ev: list) -> list:
    """[(start, end, state)] for every event that has a successor (the final horizon marker has none)."""
    return [(t0, t1, s) for (t0, s), (t1, _) in zip(ev, ev[1:])]


def like_stats(mobiperf_events: dict) -> dict:
    """What a 3-state trace must match, measured on `{device: [[ts, state], ...]}`: P(TRAIN -> EVAL),
    P(EVAL -> TRAIN) and EVAL's share of non-train time."""
    exits, dwell = Counter(), Counter()
    horizon = max(ev[-1][0] for ev in mobiperf_events.values())  # a device's last state holds to the common end
    for ev in mobiperf_events.values():
        for (_, a), (_, b) in zip(ev, ev[1:]):
            exits[a, b] += 1
        for t0, t1, s in _spans(ev + [[horizon, None]]):
            dwell[s] += t1 - t0
    p_te = exits[TRAIN, EVAL] / (exits[TRAIN, EVAL] + exits[TRAIN, UN])
    p_et = exits[EVAL, TRAIN] / (exits[EVAL, TRAIN] + exits[EVAL, UN])
    return {"p_train_to_eval": p_te, "p_eval_to_train": p_et,
            "p_un_to_eval": exits[UN, EVAL] / (exits[UN, EVAL] + exits[UN, TRAIN]),
            "eval_share_of_nontrain": dwell[EVAL] / (dwell[EVAL] + dwell[UN])}


def three_state_events(per_trainer: dict, like: dict, seed: str) -> dict:
    """Relabel each non-train window of the 2-state `per_trainer` as all-EVAL, EVAL-then-UN or all-UN with
    MobiPerf's exit probabilities; AVL_TRAIN stays exactly. One split fraction per trace puts EVAL at
    `like['eval_share_of_nontrain']` of non-train time. Seeded per (trace, trainer)."""
    p_all_eval = like["p_train_to_eval"] * like["p_eval_to_train"]
    p_split = like["p_train_to_eval"] * (1 - like["p_eval_to_train"])
    kinds = {}
    for key, ev in per_trainer.items():
        rng = random.Random(f"{seed}|{key}")
        kinds[key] = {t0: ("eval" if (u := rng.random()) < p_all_eval else "split" if u < p_all_eval + p_split else "un")
                      for t0, t1, s in _spans(ev) if s != TRAIN}
    tot = {"eval": 0.0, "split": 0.0, "un": 0.0}
    for key, ev in per_trainer.items():
        for t0, t1, s in _spans(ev):
            if s != TRAIN:
                tot[kinds[key][t0]] += t1 - t0
    nontrain = sum(tot.values())
    frac = min(1.0, max(0.0, (like["eval_share_of_nontrain"] * nontrain - tot["eval"]) / tot["split"])) if tot["split"] else 0.0
    out = {}
    for key, ev in per_trainer.items():
        new = []
        for (t0, s), nxt in zip(ev, ev[1:] + [None]):
            if s == TRAIN or nxt is None:
                new.append([t0, s])  # AVL_TRAIN and the final marker are kept
                continue
            kind, cut = kinds[key][t0], t0 + max(1, round(frac * (nxt[0] - t0)))
            if kind == "split" and cut >= nxt[0]:
                kind = "eval"
            new += [[t0, EVAL]] if kind == "eval" else [[t0, EVAL], [cut, UN]] if kind == "split" else [[t0, UN]]
        out[key] = new
    return out


def _shares(per_trainer: dict, t_end: float) -> dict:
    dwell = Counter()
    for ev in per_trainer.values():
        for t0, t1, s in _spans(ev):
            dwell[s] += max(0.0, min(t1, t_end) - min(t0, t_end))
    tot = sum(dwell.values())
    return {s: dwell[s] / tot for s in (TRAIN, EVAL, UN)}


def verify_three_state(src: dict, new: dict, like: dict, share_tol: float = 0.01, exit_tol: float = 0.05) -> dict:
    """FX-N96 self-checks; raises AssertionError listing every failure, else returns the measured numbers."""
    bad = []
    for key, ev in src.items():
        if [e for e in ev if e[1] == TRAIN] != [e for e in new[key] if e[1] == TRAIN]:
            bad.append(f"{key}: AVL_TRAIN events differ from the source")
        if any(a[1] == b[1] for a, b in zip(new[key], new[key][1:])) or new[key][-1] != ev[-1] or new[key][-1][1] != UN:
            bad.append(f"{key}: repeated state or final UN_AVL marker lost")
    horizon = max(ev[-1][0] for ev in src.values())
    res = {}
    for label, t_end in (("full", horizon), ("0-3h", 10800.0)):
        sh, base = _shares(new, t_end), _shares(src, t_end)
        want = {TRAIN: base[TRAIN], EVAL: like["eval_share_of_nontrain"] * (1 - base[TRAIN])}
        want[UN] = 1 - want[TRAIN] - want[EVAL]
        res[label] = sh
        bad += [f"{label} {s} share {sh[s]:.3f} vs target {want[s]:.3f}" for s in want if abs(sh[s] - want[s]) > share_tol]
    ex = Counter((a, b) for ev in new.values() for (_, a), (_, b) in zip(ev, ev[1:]))
    rates = {"train_to_eval": ex[TRAIN, EVAL] / (ex[TRAIN, EVAL] + ex[TRAIN, UN]),
             "eval_to_train": ex[EVAL, TRAIN] / (ex[EVAL, TRAIN] + ex[EVAL, UN])}
    for k, want in (("train_to_eval", like["p_train_to_eval"]), ("eval_to_train", like["p_eval_to_train"])):
        if abs(rates[k] - want) > exit_tol:
            bad.append(f"exit {k} {rates[k]:.3f} vs MobiPerf {want:.3f}")
    res["exits"] = rates
    assert not bad, "; ".join(bad[:12])
    return res


def make_three_state(name: str, like_name: str, seed: int) -> None:
    d = yaml.load(TRACES.read_text(), Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))
    src = d["traces"][name]
    mobi = yaml.load(MOBIPERF.read_text(), Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))["traces"]
    sub = {"mobiperf_3st_50": "states_3st_50", "mobiperf_3st_75": "states_3st_75"}[like_name]
    like = like_stats({k: v[sub] for k, v in mobi.items()})
    per = src["per_trainer"]["n300"]
    tag = f"{seed}|{name}_3st"
    new_per = three_state_events(per, like, tag)
    pat = three_state_events({"pattern": src["pattern"]}, like, tag)["pattern"]
    res = verify_three_state(per, new_per, like)
    print(f"{name}_3st like {like_name}: " + ", ".join(f"{k} {v}" for k, v in res.items()))
    desc = (f"{src['description']}; 3-state: AVL_TRAIN identical, each non-train window relabeled AVL_EVAL / "
            f"AVL_EVAL->UN_AVL / UN_AVL with {like_name}'s exit probabilities (gen_synthetic_trace.py --three-state, seed {seed})")
    text = TRACES.read_text()
    key = f"{name}_3st"
    if f"\n  {key}:\n" in text:
        raise SystemExit(f"{key} already in {TRACES.name}")
    m = re.search(rf"\n  {re.escape(name)}:\n(?:(?!\n  \S).)*", text, re.S)  # the source block, up to the next key
    TRACES.write_text(text[:m.end()] + "\n" + _block(key, desc, pat, new_per).rstrip("\n") + text[m.end():])
    print(f"wrote {key} after {name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--extend", nargs="+", default=[], help="existing traces to extend to --horizon-s")
    ap.add_argument("--three-state", metavar="SYN", help="write SYN_3st from the 2-state SYN (FX-N96)")
    ap.add_argument("--like", default="mobiperf_3st_50", help="3-state MobiPerf trace the exits and shares come from")
    ap.add_argument("--name")
    ap.add_argument("--unavail", type=float, help="stationary unavailable fraction")
    ap.add_argument("--mean-outage-s", type=float, default=672.0, help="syn_10's mean outage (600 s slots)")
    ap.add_argument("--slot-s", type=int, default=600)
    ap.add_argument("--horizon-s", type=int, default=86400)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--insert-before", help="existing trace key the new block goes above")
    a = ap.parse_args()
    if a.three_state:
        make_three_state(a.three_state, a.like, a.seed)
        return
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
