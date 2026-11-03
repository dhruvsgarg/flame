"""Centralised backprop ceiling for a FluxTune model/dataset: same model build as expts/initializer.py
(adapters LoRA, alpha=2r, train_adapter), exact gradients, AdamW. Reports accuracy vs step.

  python backprop_ceiling.py --lora-r 4 --steps 3000 --out DIR   (one GPU; CUDA_VISIBLE_DEVICES picks it)
"""
import argparse, json, os, random, time

import adapters
import h5py
import numpy as np
import torch
from adapters import LoRAConfig
from transformers import AutoTokenizer, LlamaForSequenceClassification

P = "/coc/scratch/dgarg/fl_datasets/fwdllm/fednlp_data/"
PART = "niid_label_clients=100_alpha=1"

ap = argparse.ArgumentParser()
ap.add_argument("--model-name", default="HuggingFaceTB/SmolLM2-1.7B")
ap.add_argument("--lora-r", type=int, default=4)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--steps", type=int, default=3000)
ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--micro-bs", type=int, default=16)
ap.add_argument("--max-len", type=int, default=256)
ap.add_argument("--eval-every", type=int, default=250)
ap.add_argument("--eval-n", type=int, default=3000)
ap.add_argument("--seed", type=int, default=1233)
ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
dev = "cuda"

tok = AutoTokenizer.from_pretrained(a.model_name)
tok.pad_token = tok.pad_token or tok.eos_token
model = LlamaForSequenceClassification.from_pretrained(a.model_name, num_labels=10, torch_dtype=torch.bfloat16)
model.config.pad_token_id = tok.pad_token_id
adapters.init(model)
model.add_adapter("lora_adapter", config=LoRAConfig(r=a.lora_r, alpha=2 * a.lora_r))
model.train_adapter("lora_adapter")
for n, p in model.named_parameters():
    if "score" in n:
        p.requires_grad = True
for p in model.parameters():
    if p.requires_grad:
        p.data = p.data.float()
model.to(dev)
trainable = [p for p in model.parameters() if p.requires_grad]
print(f"trainable p = {sum(p.numel() for p in trainable)} (harness r=4: 806912)", flush=True)

d = h5py.File(P + "data_files/yahoo_data.h5", "r")
pf = h5py.File(P + "partition_files/yahoo_partition.h5", "r")[PART]["partition_data"]
tr_idx, te_idx = [], []
for c in pf.keys():
    tr_idx.extend(pf[c]["train"][()].tolist()); te_idx.extend(pf[c]["test"][()].tolist())
rng = np.random.RandomState(a.seed)
te_idx = sorted(te_idx)
te_idx = [te_idx[i] for i in rng.permutation(len(te_idx))[:a.eval_n]]
vocab = json.loads(d["attributes"][()])["label_vocab"]


def rows(idx):
    xs = [d["X"][str(i)][()].decode() for i in idx]
    ys = [vocab[d["Y"][str(i)][()].decode()] for i in idx]
    return xs, torch.tensor(ys)


def enc(xs):
    b = tok(xs, truncation=True, max_length=a.max_len, padding=True, return_tensors="pt")
    return {k: v.to(dev) for k, v in b.items()}


te_x, te_y = rows(te_idx)


@torch.no_grad()
def evaluate():
    model.eval(); ok, loss = 0, 0.0
    for i in range(0, len(te_x), 64):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = model(**enc(te_x[i:i + 64])).logits.float()
        y = te_y[i:i + 64].to(dev)
        ok += (o.argmax(-1) == y).sum().item(); loss += torch.nn.functional.cross_entropy(o, y, reduction="sum").item()
    model.train()
    return ok / len(te_x), loss / len(te_x)


opt = torch.optim.AdamW(trainable, lr=a.lr, weight_decay=0.0)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 100))
model.train(); log = open(os.path.join(a.out, "curve.jsonl"), "w"); t0 = time.time()
for step in range(1, a.steps + 1):
    idx = random.sample(tr_idx, a.bs); xs, ys = rows(idx); tl = 0.0
    for i in range(0, a.bs, a.micro_bs):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = model(**enc(xs[i:i + a.micro_bs])).logits.float()
        l = torch.nn.functional.cross_entropy(o, ys[i:i + a.micro_bs].to(dev)) * (len(xs[i:i + a.micro_bs]) / a.bs)
        l.backward(); tl += l.item()
    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
    opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
    if step % 25 == 0:
        print(f"step {step} loss {tl:.3f} t={time.time() - t0:.0f}s", flush=True)
    if step % a.eval_every == 0 or step == a.steps:
        acc, tel = evaluate()
        r = {"step": step, "test_acc": acc, "test_loss": tel, "train_loss": tl, "t_s": time.time() - t0}
        print("EVAL", r, flush=True); log.write(json.dumps(r) + "\n"); log.flush()
