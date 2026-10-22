#!/usr/bin/env python3
"""Side-car that logs every agg_eval of the newest run and flags accuracy trouble.

Reads telemetry only, never kills (watch_arm.py does that). One line per eval to
--out; ALERT lines for: no eval within --eval-gap-s, accuracy --drop below the
best so far (FT-L8 collapse), non-finite loss. Exits when --pgid is gone.
"""
import argparse, glob, json, math, os, time


def newest_run(exp_dir, since):
    runs = [d for d in glob.glob(os.path.join(exp_dir, "run_*")) if os.path.getmtime(d) >= since]
    return max(runs, key=os.path.getmtime) if runs else None


def alive(pgid):
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", required=True)
    ap.add_argument("--since", type=float, required=True)
    ap.add_argument("--pgid", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--poll", type=float, default=60.0)
    ap.add_argument("--eval-gap-s", type=float, default=1200.0)
    ap.add_argument("--drop", type=float, default=0.05)
    a = ap.parse_args()

    def log(msg):
        with open(a.out, "a") as f:
            f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")

    run, pos, best, n, last_eval = None, 0, 0.0, 0, time.time()
    log(f"start pgid={a.pgid} exp_dir={a.exp_dir}")
    while alive(a.pgid):
        if run is None:
            run = newest_run(a.exp_dir, a.since)
            if run:
                log(f"run {os.path.basename(run)}")
        tel = glob.glob(os.path.join(run, "telemetry", "aggregator_*.jsonl")) if run else []
        if tel:
            with open(tel[0]) as f:
                f.seek(pos)
                for line in f:
                    if '"agg_eval"' not in line:
                        continue
                    try:
                        e = json.loads(line)
                    except ValueError:
                        continue
                    acc, loss = e.get("test-accuracy"), e.get("test-loss")
                    n += 1
                    last_eval = time.time()
                    best = max(best, acc or 0.0)
                    h = (time.time() - a.since) / 3600
                    log(f"eval#{n} {h:.2f}h commit={e.get('data_id')} acc={acc:.4f} best={best:.4f} "
                        f"loss={loss:.4f} eval_s={e.get('eval_wall_s', 0):.0f}")
                    if loss is None or not math.isfinite(loss):
                        log(f"ALERT non-finite loss at eval#{n}")
                    if acc is not None and best - acc > a.drop:
                        log(f"ALERT acc {acc:.4f} is {best - acc:.3f} below best {best:.4f}")
                pos = f.tell()
        if run and time.time() - last_eval > a.eval_gap_s:
            log(f"ALERT no eval for {(time.time() - last_eval) / 60:.0f} min")
            last_eval = time.time()
        time.sleep(a.poll)
    log(f"run exited: {n} evals, best acc {best:.4f}")


if __name__ == "__main__":
    main()
