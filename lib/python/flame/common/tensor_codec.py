# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""FX-N77 flat tensor codec: a state dict as one contiguous byte buffer behind a small header.

Replaces cloudpickle-of-tensors for model weights on the wire (per-storage torch.save/load, per-tensor device copies):
encode = one device->host copy; decode = zero-copy views, or one host->device copy. `FLAME_WEIGHT_CODEC=pickle` reverts
the encoder; `util.materialize_weights` decodes both formats.
"""

from __future__ import annotations

import json
import os
import warnings
from typing import Mapping, Optional

import torch

MAGIC = b"FTC1"
ENV_CODEC = "FLAME_WEIGHT_CODEC"
_ALIGN = 8  # every tensor starts 8-byte aligned so its uint8 slice can be viewed as its dtype


def enabled() -> bool:
    return os.environ.get(ENV_CODEC, "flat").lower() != "pickle"


def is_encoded(blob) -> bool:
    return bytes(memoryview(blob)[: len(MAGIC)]) == MAGIC


def encode(weights: Mapping[str, torch.Tensor]) -> bytes:
    """Header (name, dtype, shape per tensor) + the tensors' bytes, padded to _ALIGN, in one buffer."""
    parts, meta = [], []
    for name, t in weights.items():
        t = t.detach()
        if t.device != next(iter(weights.values())).device:
            t = t.cpu()
        b = t.contiguous().reshape(-1).view(torch.uint8)
        parts.append(b)
        pad = -b.numel() % _ALIGN
        if pad:
            parts.append(b.new_zeros(pad))
        meta.append((name, str(t.dtype).removeprefix("torch."), list(t.shape)))
    flat = (torch.cat(parts) if parts else torch.empty(0, dtype=torch.uint8)).cpu()
    hdr = json.dumps(meta).encode()
    return b"".join((MAGIC, len(hdr).to_bytes(4, "little"), hdr, memoryview(flat.numpy())))


def decode(blob, device: Optional[torch.device] = None) -> dict:
    """Tensors on `device` (default CPU). CPU tensors own a copy of the body (writable); others share one device buffer."""
    mv = memoryview(blob)
    if not is_encoded(mv):
        raise ValueError("not a flat tensor codec payload")
    n = int.from_bytes(mv[4:8], "little")
    meta = json.loads(bytes(mv[8 : 8 + n]))
    body = mv[8 + n :]
    with warnings.catch_warnings():  # read-only source: copied below before anyone can write
        warnings.simplefilter("ignore", UserWarning)
        flat = torch.frombuffer(body, dtype=torch.uint8) if len(body) else torch.empty(0, dtype=torch.uint8)
    device = torch.device(device) if device is not None else torch.device("cpu")
    flat = flat.to(device) if device.type != "cpu" else flat.clone()
    out, off = {}, 0
    for name, dt, shape in meta:
        dtype = getattr(torch, dt)
        nbytes = torch.Size(shape).numel() * torch.empty((), dtype=dtype).element_size()
        out[name] = flat[off : off + nbytes].view(dtype).reshape(shape)
        off += nbytes + (-nbytes % _ALIGN)
    return out
