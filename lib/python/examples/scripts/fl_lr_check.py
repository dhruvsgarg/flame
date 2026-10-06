#!/usr/bin/env python3
"""In-process FL lr check (FX-N10): synchronous FedBuff-style rounds on the real model, split and local loop, no MQTT.

  fl_lr_check.py --dataset google_speech --rounds 30 --pairs 0.000195:0.065 0.04:0.065 0.000195:0.075

Each pair = trainer lr : server lr. Per round, `--k` trainers (seeded) train one local epoch from the global model with a
fresh optimizer (the dataset's: Adam speech, SGD cifar), as the trainer does; the server applies
base += server_lr * sum(rate * delta) / k (fedbuff.py, staleness 0). Pairs run in parallel threads, one GPU each.
Staleness 0 makes this an upper bound on async progress per commit; it answers "does this lr pair learn at all".
`--staleness S --flame-opt felix|fedbuff`: each update trains from the global of a seeded 0..S rounds ago and the round is
applied by flame's FedBuff optimizer with that baseline's rate; a third pair field sets `bn_absolute_mean` (FX-D61; default 1).
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


def _flame_opt(a, server_lr, bn_absolute_mean):
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
    opt_cls = torch.optim.Adam if spec.optimizer == "adam" else torch.optim.SGD
    test_loader = data_utils.DataLoader(data_utils.Subset(test, test_idx), batch_size=256)
    t0, hist = time.time(), []
    opt_f = _flame_opt(a, server_lr, bn_absolute_mean) if a.flame_opt else None
    past = [{k: v.clone() for k, v in glob_state.items()}]  # past[-1-s] = global s rounds ago
    for r in range(1, a.rounds + 1):
        acc_delta, agg_w = None, None
        for tid in rng.sample(names, a.k):
            s_i = min(rng.randint(0, a.staleness), len(past) - 1)
            base = past[-1 - s_i]
            model.load_state_dict(base)
            model.train()
            opt = opt_cls(model.parameters(), lr=client_lr)
            loader = data_utils.DataLoader(data_utils.Subset(train, splits[tid]), batch_size=a.batch, shuffle=True)
            for x, y in loader:
                x, y = x.to(dev), y.to(dev)
                opt.zero_grad(set_to_none=True)
                F.nll_loss(model(x), y).backward()
                opt.step()
            if opt_f is not None:
                from flame.optimizer.train_result import TrainResult
                with torch.no_grad():
                    delta = {k: v.detach() - base[k] for k, v in model.state_dict().items()}
                agg_w = opt_f.do(agg_w, _Cache({tid: TrainResult(delta, 1, r - s_i, a.utility)}), total=1, version=r)
                continue
            with torch.no_grad():
                for k, v in model.state_dict().items():
                    d = (v.float() - base[k].float()) * a.rate
                    if acc_delta is None:
                        acc_delta = {}
                    acc_delta[k] = acc_delta[k] + d if k in acc_delta else d
        with torch.no_grad():
            if opt_f is not None:
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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="google_speech")
    ap.add_argument("--pairs", nargs="+", required=True, help="client_lr:server_lr")
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--k", type=int, default=10, help="updates per round (aggGoal)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--rate", type=float, default=0.88, help="per-update weight (felix 'new' rate ~0.88; fedbuff 1)")
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--test-n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--gpus", default="0,1,2")
    ap.add_argument("--staleness", type=int, default=0, help="max rounds an update's base lags the global")
    ap.add_argument("--flame-opt", choices=["felix", "fedbuff"], help="apply rounds with flame's FedBuff optimizer")
    ap.add_argument("--utility", type=float, default=600.0, help="stat_utility fed to the felix rate")
    a = ap.parse_args(argv)
    spec = fl_data.SPECS[a.dataset]
    split_file = {"google_speech": "google_speech_alpha0.1_n100.yaml"}.get(a.dataset)
    if split_file is None:
        sys.exit("only google_speech has a stored split here")
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
