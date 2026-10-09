#!/usr/bin/env python3
"""In-process FL lr check (FX-N10): synchronous FedBuff-style rounds on the real model, split and local loop, no MQTT.

  fl_lr_check.py --dataset google_speech --rounds 30 --pairs 0.000195:0.065 0.04:0.065 0.000195:0.075

Each pair = trainer lr : server lr. Per round, `--k` trainers (seeded) train one local epoch from the global model with a
fresh optimizer (the dataset's: Adam speech, SGD cifar), as the trainer does; the server applies
base += server_lr * sum(rate * delta) / k (fedbuff.py, staleness 0). Pairs run in parallel threads, one GPU each.
Staleness 0 makes this an upper bound on async progress per commit; it answers "does this lr pair learn at all".
`--staleness S --flame-opt felix|fedbuff`: each update trains from the global of a seeded 0..S rounds ago and the round is
applied by flame's FedBuff optimizer with that baseline's rate; a third pair field sets `bn_absolute_mean` (FX-D61; default 1).
`--baseline B`: batch, client optimizer/lr, local steps|epochs, lr decay, server optimizer and K from
_metadata/baseline_reference.yaml (FX-D100) -- the same values the launcher applies; explicit flags still win.
`--flame-opt refl`: rounds go through flame's REFL optimizer (SAA weights, `--gradient-policy yogi`); server lr unused.
`--lr-decay F:E:MIN`: client lr x F every E rounds, floored at MIN (trainer lrDecay*, REFL).
"""
import argparse
import random
import sys
import threading
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.utils.data as data_utils
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "async_cifar10"))
import fl_data  # noqa: E402

SPLITS = HERE.parent / "_metadata" / "dataset_splits"


class Cached(data_utils.Dataset):
    """RAM cache over a dataset (speech decodes each wav per access)."""

    def __init__(self, ds):
        self.ds, self.mem, self.lock = ds, {}, threading.Lock()

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        v = self.mem.get(i)
        if v is None:
            v = self.ds[i]
            with self.lock:
                self.mem[i] = v
        return v


RATE_CONF = {"felix": {"type": "new", "scale": 0.4, "a_exp": 0.25, "b_exp": 0.1}, "fedbuff": {"type": "old"}}


class _Cache(dict):
    def iterkeys(self):
        return iter(list(self.keys()))


SYNC_SORTS = ("refl", "fedavg", "fedavg_yogi")  # one aggregation over the round's k updates


def _flame_opt(a, server_lr, bn_absolute_mean):
    if a.server:  # --baseline: the reference server optimizer, as the aggregator builds it
        from flame.optimizer.fedavg import FedAvg
        from flame.optimizer.fedbuff import FedBuff
        from flame.optimizer.fedscale_yogi import FedAvgYoGi
        from flame.optimizer.refl import REFL
        kw = dict(a.server["kwargs"])
        if a.server["sort"] == "refl":
            kw["deadline"] = 1e9  # no wall-clock deadline in-process
        return {"refl": REFL, "fedavg": lambda **k: FedAvg(), "fedavg_yogi": FedAvgYoGi,
                "fedbuff": lambda **k: FedBuff(bn_absolute_mean=bn_absolute_mean, **k)}[a.server["sort"]](**kw)
    if a.flame_opt == "refl":  # baselines.yaml refl optimizer kwargs (FX-N74)
        from flame.optimizer.refl import REFL
        return REFL(deadline=1e9, stale_update=5, stale_factor=-4, stale_beta=0.35, scale_coff=a.scale_coff,
                    gradient_policy=a.gradient_policy)
    from flame.optimizer.fedbuff import FedBuff
    return FedBuff(use_oort_lr=str(a.flame_opt == "felix"), dataset_name="google-speech", learning_rate=server_lr,
                   agg_rate_conf=RATE_CONF[a.flame_opt], bn_absolute_mean=bn_absolute_mean)


def run_pair(spec, train, test_idx, test, splits, client_lr, server_lr, a, gpu, out, bn_absolute_mean=True):
    dev = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed)
    model = spec.model().to(dev)
    glob_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    rng = random.Random(a.seed)
    names = sorted(splits)
    opt_cls = torch.optim.Adam if (a.optimizer or spec.optimizer) == "adam" else torch.optim.SGD
    test_loader = data_utils.DataLoader(data_utils.Subset(test, test_idx), batch_size=256)
    t0, hist = time.time(), []
    opt_f = _flame_opt(a, server_lr, bn_absolute_mean) if (a.flame_opt or a.server) else None
    sync = a.flame_opt == "refl" or bool(a.server and a.server["sort"] in SYNC_SORTS)
    past = [{k: v.clone() for k, v in glob_state.items()}]  # past[-1-s] = global s rounds ago
    for r in range(1, a.rounds + 1):
        acc_delta, agg_w, refl_in = None, None, _Cache()
        lr_r = client_lr
        if a.lr_decay:  # trainer main.py: decays from round 2 on
            f, e, lo = (float(x) for x in a.lr_decay.split(":"))
            lr_r = max(lo, client_lr * f ** ((r - 1) // int(e)))
        for tid in rng.sample(names, a.k):
            s_i = min(rng.randint(0, a.staleness), len(past) - 1)
            base = past[-1 - s_i]
            model.load_state_dict(base)
            model.train()
            opt = opt_cls(model.parameters(), lr=lr_r)
            loader = data_utils.DataLoader(data_utils.Subset(train, splits[tid]), batch_size=a.batch, shuffle=True)
            steps, done = a.local_steps, 0
            for _ in range(a.epochs if steps is None else 10 ** 9):  # local_steps cycles the data (trainer FX-N74)
                for x, y in loader:
                    if steps is not None and done >= steps:
                        break
                    x, y = x.to(dev), y.to(dev)
                    opt.zero_grad(set_to_none=True)
                    F.nll_loss(model(x), y).backward()
                    opt.step()
                    done += 1
                if steps is None or done >= steps:
                    break
            if opt_f is not None:
                from flame.optimizer.train_result import TrainResult
                with torch.no_grad():
                    delta = {k: v.detach() - base[k] for k, v in model.state_dict().items()}
                if sync:
                    refl_in[tid] = TrainResult(delta, len(splits[tid]), r - s_i, a.utility, staleness=s_i, end_id=tid)
                    continue
                agg_w = opt_f.do(agg_w, _Cache({tid: TrainResult(delta, 1, r - s_i, a.utility)}), total=1, version=r)
                continue
            with torch.no_grad():
                for k, v in model.state_dict().items():
                    d = (v.float() - base[k].float()) * a.rate
                    if acc_delta is None:
                        acc_delta = {}
                    acc_delta[k] = acc_delta[k] + d if k in acc_delta else d
        with torch.no_grad():
            if sync:
                new = opt_f.do({k: v.clone() for k, v in glob_state.items()}, refl_in,
                               total=sum(t.count for t in refl_in.values()), version=r, round_duration=1.0)
                glob_state = {k: v.to(glob_state[k].dtype) for k, v in new.items()}
            elif opt_f is not None:
                new = opt_f.scale_add_agg_weights({k: v.clone() for k, v in glob_state.items()}, agg_w, a.k)
                glob_state = {k: v.to(glob_state[k].dtype) for k, v in new.items()}
            else:
                for k in glob_state:
                    glob_state[k] = (glob_state[k].float() + server_lr * acc_delta[k] / a.k).to(glob_state[k].dtype)
        past = (past + [{k: v.clone() for k, v in glob_state.items()}])[-(a.staleness + 1):]
        if r % a.eval_every == 0 or r == a.rounds:
            model.load_state_dict(glob_state)
            model.eval()
            correct = n = 0
            loss = 0.0
            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.to(dev), y.to(dev)
                    o = model(x)
                    loss += F.nll_loss(o, y, reduction="sum").item()
                    correct += (o.argmax(1) == y).sum().item()
                    n += len(y)
            hist.append((r, correct / n, loss / n))
            print(f"[{client_lr}:{server_lr}] round {r} acc {correct / n:.3f} loss {loss / n:.3f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    out[(client_lr, server_lr, bn_absolute_mean)] = hist


def _apply_reference(a):
    """Fill unset flags from the baseline's reference cell (the launcher's overlay)."""
    from flame.launch import baseline_reference as br
    n = {"google_speech": 100, "cifar10": 300}[a.dataset]
    ov = br.overlay(br.load(), a.baseline, a.dataset, n)
    hp, srv = ov["trainer"]["hyperparameters"], ov["aggregator"]["config_overrides"]["optimizer"]
    a.batch = a.batch or hp["batchSize"]
    a.optimizer = a.optimizer or hp["trainerOptimizer"]
    if a.local_steps is None and "localSteps" in hp:
        a.local_steps = hp["localSteps"]
    if not a.lr_decay and hp.get("lrDecayEnabled"):
        a.lr_decay = f"{hp['lrDecayFactor']}:{hp['lrDecayEpoch']}:{hp['minLearningRate']}"
    a.server = srv
    a.k = a.k or ov["aggregator"].get("agg_goal") or 10
    a.pairs = a.pairs or [f"{hp['learningRate']}:{srv['kwargs'].get('learning_rate', 1)}"]
    print(f"[reference] {a.baseline} {a.dataset}: batch {a.batch} {a.optimizer} steps {a.local_steps} decay {a.lr_decay or '-'} "
          f"server {srv['sort']} {srv['kwargs']} k {a.k} pairs {a.pairs}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="google_speech")
    ap.add_argument("--pairs", nargs="+", help="client_lr:server_lr (default with --baseline: its client lr : 1)")
    ap.add_argument("--baseline", help="take every training/server knob from baseline_reference.yaml")
    ap.add_argument("--local-steps", type=int, help="mini-batch iterations per update (default: one --epochs pass)")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--k", type=int, help="updates per round (aggGoal; default 10, or the reference K)")
    ap.add_argument("--batch", type=int, help="default 32, or the reference batch")
    ap.add_argument("--optimizer", choices=["sgd", "adam"], help="trainer optimizer (default: the dataset's)")
    ap.add_argument("--rate", type=float, default=0.88, help="per-update weight (felix 'new' rate ~0.88; fedbuff 1)")
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--test-n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--gpus", default="0,1,2")
    ap.add_argument("--staleness", type=int, default=0, help="max rounds an update's base lags the global")
    ap.add_argument("--flame-opt", choices=["felix", "fedbuff", "refl"], help="apply rounds with flame's optimizer")
    ap.add_argument("--gradient-policy", choices=["yogi"], help="refl: server step after SAA")
    ap.add_argument("--scale-coff", type=float, default=1.0, help="refl: SAA scale_coff (baselines.yaml)")
    ap.add_argument("--lr-decay", default="", help="F:E:MIN client lr decay (refl 0.98:10:0.00005)")
    ap.add_argument("--utility", type=float, default=600.0, help="stat_utility fed to the felix rate")
    a = ap.parse_args(argv)
    a.server = None
    if a.baseline:
        _apply_reference(a)
    a.k, a.batch = a.k or 10, a.batch or 32
    if not a.pairs:
        sys.exit("--pairs required without --baseline")
    spec = fl_data.SPECS[a.dataset]
    split_file = {"google_speech": "google_speech_alpha0.1_n100.yaml", "cifar10": "cifar10_alpha0.1_n300.yaml"}.get(a.dataset)
    if split_file is None:
        sys.exit(f"no stored split for {a.dataset}")
    raw = yaml.safe_load(open(SPLITS / split_file))["trainer_data_splits"]
    train, test = Cached(spec.train()), Cached(spec.test())
    test_idx = random.Random(0).sample(range(len(test)), min(a.test_n, len(test)))
    gpus = [int(g) for g in a.gpus.split(",")]
    out, threads = {}, []
    for i, p in enumerate(a.pairs):
        c, s, *bn = p.split(":")
        th = threading.Thread(target=run_pair, args=(spec, train, test_idx, test, raw, float(c), float(s), a,
                                                     gpus[i % len(gpus)], out, (bn or ["1"])[0] == "1"))
        th.start()
        threads.append(th)
    for th in threads:
        th.join()
    print("\nclient_lr:server_lr  " + "  ".join(f"r{r}" for r, _, _ in next(iter(out.values()))))
    for (c, s, bn), h in out.items():
        print(f"{c}:{s}:{int(bn)}  " + "  ".join(f"{acc:.3f}/{loss:.3g}" for _, acc, loss in h))
    return 0


if __name__ == "__main__":
    sys.exit(main())
