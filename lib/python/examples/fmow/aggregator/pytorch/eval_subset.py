"""Fixed, seeded test subset so every round and every arm is scored on the same images."""

import numpy as np


def select_eval_indices(indices, max_samples, seed):
    """Return `indices` unchanged if max_samples is falsy or >= len; else a sorted seeded sample."""
    indices = np.asarray(indices)
    if not max_samples or max_samples <= 0 or max_samples >= len(indices):
        return indices.tolist()
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(indices, size=int(max_samples), replace=False)).tolist()
