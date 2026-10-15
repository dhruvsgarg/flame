#!/usr/bin/env python3
"""FX-N98 micro-benchmark: K concurrent local trainings on one GPU. Reports per-step time and total steps/s per K,
to pick `simGpuTrainSlots` (best throughput with per-task time <= real's, which already shares a GPU).

  gpu_slot_bench.py --gpu 0 --ks 1,2,3,4,7 --steps 20 --tasks 3
"""
import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "async_cifar10"))


def worker(gpu, steps, tasks, batch, barrier, q):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import torch
    import torch.nn.functional as F
    import fl_data
    dev = torch.device("cuda")
    model = fl_data.ResNet34_1D().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=0.000195)
    x = torch.randn(batch, 1, 16000, device=dev)
    y = torch.randint(0, 35, (batch,), device=dev)

    def task():
        model.train()
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(model(x), y).backward()
            opt.step()
        torch.cuda.synchronize()

    task()  # warm-up
    barrier.wait()
    out = []
    for _ in range(tasks):
        t = time.time()
        task()
        out.append(time.time() - t)
    q.put(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--ks", default="1,2,3,4,7")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--tasks", type=int, default=3)
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()
    ctx = mp.get_context("spawn")
    print(f"{'K':>2} {'task s (mean)':>14} {'step ms':>8} {'steps/s total':>14} {'vs K=1 throughput':>18}")
    base = None
    for k in map(int, a.ks.split(",")):
        barrier, q = ctx.Barrier(k), ctx.Queue()
        ps = [ctx.Process(target=worker, args=(a.gpu, a.steps, a.tasks, a.batch, barrier, q)) for _ in range(k)]
        [p.start() for p in ps]
        res = [t for _ in ps for t in q.get()]
        [p.join() for p in ps]
        task_s = sum(res) / len(res)
        tput = k * a.steps / task_s
        base = base or tput
        print(f"{k:>2} {task_s:>14.2f} {1000 * task_s / a.steps:>8.1f} {tput:>14.1f} {tput / base:>17.2f}x", flush=True)


if __name__ == "__main__":
    main()
