"""FX-N68: per-leg s/round and slowest-pick speed of replicate legs, and their spread (sim<->sim or real<->real).

  replicate_spread.py RUN_DIR [RUN_DIR ...]
"""
import glob
import json
import sys


def leg_stats(run_dir: str) -> dict:
    f = glob.glob(f"{run_dir}/telemetry/aggregator*.jsonl")[0]
    rounds = [e for e in map(json.loads, open(f)) if e.get("event") == "agg_round"]
    clock = [e.get("vclock_now") if e.get("vclock_now") is not None else e["ts"] for e in rounds]
    spd = [max(e.get("trainer_speed_s") or [0.0]) for e in rounds]
    tail = spd[len(spd) * 2 // 3:]
    return {"rounds": len(rounds), "s_per_round": (clock[-1] - clock[0]) / max(1, len(rounds) - 1),
            "slowest_pick_tail_s": sum(tail) / max(1, len(tail))}


def main(dirs: list) -> None:
    stats = [leg_stats(d) for d in dirs]
    for d, s in zip(dirs, stats):
        print(f"{d.rsplit('/', 1)[-1][:70]:70s} rounds={s['rounds']:4d} s/round={s['s_per_round']:.2f} "
              f"slowest_pick(last third)={s['slowest_pick_tail_s']:.1f}s")
    r = [s["s_per_round"] for s in stats]
    print(f"spread s/round: min={min(r):.2f} max={max(r):.2f} rel={(max(r) - min(r)) / max(r):.3f} (n={len(r)})")


if __name__ == "__main__":
    main(sys.argv[1:])
