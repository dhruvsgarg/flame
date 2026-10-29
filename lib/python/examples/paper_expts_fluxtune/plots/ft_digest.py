"""Compact per-run digest of an aggregator telemetry file (stdlib only, so it runs on any node).

Usage: python ft_digest.py <run_dir> <out.json>
Writes {"meta", "end", "evals": [[vclock_s, commit, acc, down_B, up_B]], "final": [vclock_s, acc] | null}.
`commit` counts agg rounds with a commit_reason (server commits so far); bytes are cumulative aggregator wire bytes at that eval.
"""
import glob
import json
import os
import sys

META_KEYS = ("p_trainable", "agg_goal", "concurrency", "train_batch_size", "commit_gate", "server_step_rule", "rho_schedule",
             "rho_star", "t_res", "probe_combine", "perturbation_count", "var_threshold")


def digest(run_dir):
    f = glob.glob(os.path.join(run_dir, "telemetry", "aggregator_*.jsonl"))[0]
    t, down, up, evals, final, meta, end, commits = 0.0, 0, 0, [], None, {}, {}, 0
    with open(f, "rb") as fh:
        for line in fh:
            if b'"wire"' in line:
                e = json.loads(line)
                if e.get("dir") == "send":
                    down += e["bytes"]
                else:
                    up += e["bytes"]
            elif b'"agg_round"' in line:
                e = json.loads(line)
                t = float(e["vclock_now"]) if e.get("vclock_now") is not None else t
                commits += e.get("commit_reason") is not None
            elif b'"agg_eval_final"' in line:
                e = json.loads(line)
                final = [float(e.get("vclock_now") or t), float(e["test-accuracy"])]
            elif b'"agg_eval"' in line:
                e = json.loads(line)
                if e.get("test-accuracy") is not None:
                    evals.append([round(t, 1), commits, float(e["test-accuracy"]), down, up])
            elif b'"run_meta"' in line:
                e = json.loads(line)
                meta = {k: e[k] for k in META_KEYS if k in e}
            elif b'"run_end"' in line:
                e = json.loads(line)
                end = {k: v for k, v in e.items() if isinstance(v, (int, float, str)) and k not in ("ts", "role", "event", "end_id")}
    return dict(run_dir=os.path.abspath(run_dir), meta=meta, end=end, evals=evals, final=final, down=down, up=up, t_end=t, commits=commits)


if __name__ == "__main__":
    with open(sys.argv[2], "w") as fh:
        json.dump(digest(sys.argv[1]), fh)
