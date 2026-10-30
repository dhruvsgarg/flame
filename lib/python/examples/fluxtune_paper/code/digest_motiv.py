"""Per-server-update digest of one run for the Motivation figures (stdlib only: run on the node that holds the run).

Usage: python3 digest_motiv.py <run_dir or aggregator_*.jsonl> <out.json>

Digest = {"th0": initial trainable norm, "su": [[commit, twn, delta, rho_logged, lr, var_at_commit, pooled_norm, iters,
cos | null, vclock_s]], "evals": [[commit, acc]], "final": acc | null, "gate": [[commit, first_iter_var, mean ||g||^2 of the
first iteration's client updates, iterations]], "rounds": [[vclock_s, data_id, iteration, n_clients, task_s_median]]}.
"""
import glob
import json
import os
import statistics
import sys


def digest(path):
    f = path if path.endswith(".jsonl") else glob.glob(os.path.join(path, "telemetry", "aggregator_*.jsonl"))[0]
    th0, su, evals, final, gate, rounds = None, [], [], None, [], []
    vclock, commits, first = 0.0, 0, None
    with open(f, "rb") as fh:
        for line in fh:
            if b'"agg_round"' in line:
                e = json.loads(line)
                vclock = float(e["vclock_now"]) if e.get("vclock_now") is not None else vclock
                it = e.get("iteration_per_data_id")
                sp = e.get("trainer_speed_s") or []
                rounds.append([round(vclock, 1), e.get("data_id"), it, len(e.get("contributing_trainers") or []),
                               statistics.median(sp) if sp else None])
                if it == 1 and e.get("var") is not None:
                    gn = [g for g in (e.get("grad_norm") or []) if g is not None]
                    first = [e["var"], sum(g * g for g in gn) / len(gn) if gn else None]
            elif b'"server_update"' in line:
                e = json.loads(line)
                commits += 1
                su.append([e.get("commit_count", commits), e["trainable_weight_norm"], e.get("trainable_delta_norm"), e.get("rho"),
                           e.get("learning_rate"), e.get("var_at_commit"), e.get("pooled_norm"), e.get("iteration_per_data_id"),
                           e.get("cos_ground_truth"), round(vclock, 1)])
                if first is not None:
                    gate.append([su[-1][0], first[0], first[1], e.get("iteration_per_data_id")])
                    first = None
            elif b'"agg_eval_final"' in line:
                final = json.loads(line).get("test-accuracy")
            elif b'"agg_eval"' in line:
                e = json.loads(line)
                if e.get("test-accuracy") is not None:
                    evals.append([su[-1][0] if su else 0, e["test-accuracy"]])
            elif b'"run_meta"' in line:
                th0 = json.loads(line).get("theta_tr_norm_init")
    return dict(file=os.path.abspath(f), th0=th0, su=su, evals=evals, final=final, gate=gate, rounds=rounds)


if __name__ == "__main__":
    with open(sys.argv[2], "w") as fh:
        json.dump(digest(sys.argv[1]), fh)
