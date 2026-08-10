import os
import torch
import math
from contextlib import nullcontext as _nullcontext

from torch.nn import CrossEntropyLoss
from typing import Callable, Tuple
from torch.cuda.amp import autocast
import logging

logger = logging.getLogger(__name__)

# --- forward-pass accounting (WS3-b) ---------------------------------------
# Each trainer runs in its own process, so these module-level counters are
# per-client cumulative. The three calculate_jvp* helpers below are the only
# places the functional model is evaluated (a "forward pass"):
#   calculate_jvp                      -> 2 passes (loss + terbulence_loss) = 1 scored perturbation
#   calculate_jvp_before_actual_update -> 1 pass
#   calculate_jvp_after_actual_update  -> 1 pass
# FedSgdTrainer reads fwd_pass_counts() into trainer_round telemetry, giving
# Exp 3 a hardware-independent compute denominator immune to GPU-contention.
# Pure counters: no behavior change, zero cost when unread.
_FWD_PASSES = 0     # total forward passes (func evaluations) this trainer
_JVP_EVALS = 0      # total calculate_jvp() calls (= perturbations scored)


def fwd_pass_counts() -> Tuple[int, int]:
    """(forward_passes, jvp_evals) cumulative for THIS trainer process."""
    return _FWD_PASSES, _JVP_EVALS


def _get_loss(x: torch.Tensor, t: torch.Tensor, num_classes: int = 10) -> torch.Tensor:
    """Compute cross-entropy loss.

    Args:
        x (torch.Tensor): Output of the model.
        t (torch.Tensor): Targets.
        num_classes (int, optional): Maximum number of classes. Defaults to 10.

    Returns:
        torch.Tensor: Cross-entropy loss.
    """
    loss_fct = CrossEntropyLoss()
    loss = loss_fct(x.view(-1, num_classes), t.view(-1))
    return loss


def get_loss(
    model: torch.nn.Module, x: torch.Tensor, t: torch.Tensor, num_classes: int = 10
) -> torch.Tensor:
    """Cross-entropy loss. Given a pytorch model, it computes the cross-entropy loss.

    Args:
        model (torch.nn.Module): PyTorch model.
        x (torch.Tensor): Input tensor for the PyTorch model.
        t (torch.Tensor): Targets.
        num_classes (int, optional): Maximum number of classes. Defaults to 10.

    Returns:
        torch.Tensor: Cross-entropy loss.
    """
    y = model(x)[0]
    return _get_loss(y, t, num_classes)


def functional_get_loss(
    params: Tuple[torch.nn.Parameter, ...],
    model: Callable[[Tuple[torch.nn.Parameter, ...], torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    t: torch.Tensor,
    num_classes: int,
    buffers: list,
) -> torch.Tensor:
    """Functional cross-entropy loss. Given a functional version of a pytorch model, which can be obtained with
    `fmodel, params = functorch.make_functional(model)`, it computes the cross-entropy loss.

    Args:
        params (Tuple[torch.nn.Parameter, ...]): Model parameters obtained by `fmodel, params = fc.make_functional(model)`.
        model (Callable[[Tuple[torch.nn.Parameter, ...], torch.Tensor], torch.Tensor]): Functional version of a pytorch model,
            obtained by fmodel, `params = fc.make_functional(model)`
        x (torch.Tensor): Input tensor for the PyTorch model.
        t (torch.Tensor): Targets.
        num_classes (int, optional): Maximum number of classes. Defaults to 10.
        buffers (list): Model buffers.

    Returns:
        torch.Tensor: Cross-entropy loss.
    """
    y = model(params, buffers, x)[0]
    return _get_loss(y, t, num_classes)


def jvp_fp32_enabled() -> bool:
    """`FWDLLM_JVP_FP32=1` runs the two JVP forward passes OUTSIDE autocast.

    The central difference below divides two nearly-equal losses by 2h, so its
    condition number is ~L/(2h*|dL|); under autocast they carry fp16's ~1e-3
    error, measured amplifying ~72x into the gradient (H12, simulate_fwdllm.md).
    Env-gated so a standalone probe can read it with no config in scope.
    Default OFF => byte-identical.
    """
    return os.environ.get("FWDLLM_JVP_FP32", "").strip().lower() in ("1", "true", "yes")


# S-I (handoff §2.4): `v` is a raw Gaussian draw, so the probe DISPLACEMENT is
# h*||v|| = h*sqrt(p) -- nobody chose it, it fell out of the parameter count.
# Changing p therefore moves the finite difference as a side effect, which would
# confound any p sweep. Holding h*sqrt(p) at its reference value makes the FD
# scale-invariant. Default OFF => h stays the historical 0.01, byte-identical.
# PRODUCTION p, not the census's 1,040,932: the trainer drops pre_classifier
# before the probe is drawn (tc_transformer_trainer_distribute.py:217). Anchoring
# on the census value rescaled h by 1.52x in arms meant to hold it fixed.
_FD_REF_P = 450340             # fluxtune trainable p, where h was 0.01
_FD_REF_DISPLACEMENT = 0.01 * math.sqrt(_FD_REF_P)   # 6.711
_fd_p_cache = {}


def fd_scale_invariant_enabled() -> bool:
    return os.environ.get("FWDLLM_FD_SCALE_INVARIANT", "").strip().lower() in (
        "1", "true", "yes")


def _fd_spacing(v, trainable_idx):
    if not fd_scale_invariant_enabled():
        return 0.01
    idx = tuple(trainable_idx) if trainable_idx is not None else None
    key = (len(v), idx)
    p = _fd_p_cache.get(key)
    if p is None:
        rng = range(len(v)) if idx is None else idx
        p = sum(v[i].numel() for i in rng)
        _fd_p_cache[key] = p
        logger.info(
            f"[FD] scale-invariant spacing: p={p} h={_FD_REF_DISPLACEMENT / math.sqrt(p):.6g} "
            f"(h*sqrt(p) held at {_FD_REF_DISPLACEMENT:.4f}; h was 0.01 at p={_FD_REF_P})"
        )
    return _FD_REF_DISPLACEMENT / math.sqrt(p)


def calculate_jvp(func, params, v, trainable_idx=None):
    """
    Calculations Jacobian-vector product using numerical differentiation.

    trainable_idx (fluxtune perf-opt, simulate_fwdllm.md §L): when given, only
    those param indices are perturbed; the rest keep v=0 so `p - h*0 = p`
    exactly -> bit-identical to perturbing every param, but skips copying the
    frozen backbone twice per perturbation. None => legacy all-param path.
    """
    global _FWD_PASSES, _JVP_EVALS
    _FWD_PASSES += 2   # loss + terbulence_loss forward passes below
    _JVP_EVALS += 1
    h = _fd_spacing(v, trainable_idx)
    _cast = _nullcontext() if jvp_fp32_enabled() else autocast()
    with torch.no_grad(), _cast:
        if trainable_idx is None:
            minus = tuple([params[i] - h * v[i] for i in range(len(params))])
            plus = tuple([params[i] + h * v[i] for i in range(len(params))])
        else:
            minus = list(params)
            plus = list(params)
            for i in trainable_idx:
                minus[i] = params[i] - h * v[i]
                plus[i] = params[i] + h * v[i]
            minus, plus = tuple(minus), tuple(plus)
        loss = func(minus)
        terbulence_loss = func(plus)
    avg_loss = (terbulence_loss + loss) / 2
    jvp = (terbulence_loss - loss) / (2 * h)
    return avg_loss, jvp


def calculate_jvp_after_actual_update(func, params, v, jvp_scalar):
    """
    Calculations Jacobian-vector product using numerical differentiation
    """
    global _FWD_PASSES
    _FWD_PASSES += 1
    h = 0.01 # learning rate factor
    with torch.no_grad(), autocast():
        loss = func(tuple([params[i] - h * jvp_scalar * v[i] for i in range(len(params))]))
    return loss

def calculate_jvp_before_actual_update(func, params):
    """
    Calculations Jacobian-vector product using numerical differentiation
    """
    global _FWD_PASSES
    _FWD_PASSES += 1
    with torch.no_grad(), autocast():
        loss = func(tuple([params[i] for i in range(len(params))]))
    return loss

# Might contain useful memory optimizations. Look at this only if you're running into a memory bottleneck & you need ideas
# def calculate_jvp_experiment(func, params, v):
#     """
#     This implementation is mathematically similar to the implementation provided by Vanilla FwdLLM with some memory optimizations
#     Calculations Jacobian-vector product using numerical differentiation
#     """
#     h = 0.01
#     # logger.info(f"[MEM] Before: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
#     with torch.no_grad(), autocast():
#         # logger.info(f"params[0].device = {params[0].device}, v[0].device = {v[0].device}")
#         device = torch.device("cuda:0")
#         params = [p.to(device) for p in params]
#         v = [vi.to(device) for vi in v]
#         loss = func(tuple([params[i] - h * v[i] for i in range(len(params))]))
#         torch.cuda.empty_cache()  # optional, but can help with fragmentation
#         # logger.info(f"[MEM] After loss: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
#         terbulence_loss = func(tuple([params[i] + h * v[i] for i in range(len(params))]))
#         torch.cuda.empty_cache()
#         # logger.info(f"[MEM] After turbulence loss: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

#     avg_loss = (terbulence_loss + loss) / 2
#     jvp = (terbulence_loss - loss) / (2 * h)
#     del loss, terbulence_loss
#     torch.cuda.empty_cache()
#     # logger.info(f"[MEM] After cleanup: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
#     return avg_loss, jvp


def calculate_var(fwdgrad_list):
    n = len(fwdgrad_list)

    # Need at least 2 updates to split into two halves; with fewer (e.g. a
    # stale trainer update straggling in right after the grad-check list
    # was cleared for a new data_id) torch.stack on an empty slice crashes
    # the aggregator. Treat as "not enough signal yet" by returning a
    # sentinel above any var_threshold, so the caller's var <= threshold
    # check fails and retries instead of crashing/force-passing.
    if n < 2:
        logger.warning(
            f"calculate_var called with only {n} gradient(s); not enough "
            "to compute split-half variance, returning inf to force a retry."
        )
        return torch.tensor(float("inf"))

    # 计算前一半tensor的平均值
    first_half_mean = torch.mean(torch.stack(fwdgrad_list[: n // 2]), dim=0)

    # 计算后一半tensor的平均值
    second_half_mean = torch.mean(torch.stack(fwdgrad_list[n // 2 :]), dim=0)

    # 计算两个平均值之间的方差
    var = torch.var(torch.stack([first_half_mean, second_half_mean]), dim=0).mean()

    return var

def calculate_real_var(fwdgrad_list):
    n = len(fwdgrad_list)

    # Same n < 2 guard as calculate_var above; this value is only used for
    # logging (var_jvp), so 0.0 is a safe neutral default.
    if n < 2:
        return torch.tensor(0.0)

    # 计算两个平均值之间的方差
    var = torch.var(torch.stack(fwdgrad_list), dim=0).mean()

    return var

def log_dist(name, tensor):
    """Helper to log distribution statistics of a tensor."""
    if tensor.numel() == 0:
        return
    
    # Flatten to ensure we are looking at the distribution of all scalar values
    flat = tensor.detach().float().view(-1)
    
    # Define percentiles to track
    q = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).to(flat.device)
    percentiles = torch.quantile(flat, q)
    
    logger.info(
        f"{name:15} | Mean: {flat.mean():.6f} | "
        f"P10: {percentiles[0]:.6f} | P50: {percentiles[2]:.6f} | P90: {percentiles[4]:.6f} | P95: {percentiles[5]:.6f} | P99: {percentiles[6]:.6f}"
    )

def calculate_snr(fwdgrad_list):
    """
    Calculates SNR using the variance of all individual updates.
    This factors in magnitude outliers and client-to-client disagreement.
    """
    n = len(fwdgrad_list)
    
    
    # Requirement: Need at least 2 updates to calculate variance
    if n < 2:
        return 0.0

    # 1. Stack all individual updates: shape (N, Parameters)
    all_grads_stacked = torch.stack(fwdgrad_list)
    
    # 2. Calculate the Global Mean (The Signal)
    global_mean = torch.mean(all_grads_stacked, dim=0)
    
    # 4. Actual Variance: Variance across all N updates
    # We calculate variance for each parameter (dim=0), 
    # then take the mean to get a single scalar representing total noise.
    actual_var = torch.var(all_grads_stacked, dim=0)

    logger.info(f"shape of actual_var: {actual_var.shape}")
    logger.info("--- Gradient Distribution Stats ---")
    logger.info(f"JVP of all updates so far {all_grads_stacked}")
    log_dist("Signal (Mean)", all_grads_stacked)
    log_dist("Signal^2", all_grads_stacked**2)
    # log_dist("Variance", actual_var)

    mean_of_mean = torch.mean(global_mean)
    mean_of_mean_2 = torch.mean(global_mean ** 2)
    mean_of_var = torch.mean(actual_var)

    logger.info(f"number of updates: {n} - mean_of_mean : {mean_of_mean} mean_of_mean_squared : {mean_of_mean_2} and  mean_of_var : {mean_of_var}")

    snr = torch.mean((global_mean ** 2) / (actual_var))

    return snr.item()

def calculate_snr_gradients(fwdgrad_list):
    """
    Calculates SNR using the variance of all individual updates.
    This factors in magnitude outliers and client-to-client disagreement.
    """
    n = len(fwdgrad_list)
    
    
    # Requirement: Need at least 2 updates to calculate variance
    if n < 2:
        return 0.0

    # 1. Stack all individual updates: shape (N, Parameters)
    all_grads_stacked = torch.stack(fwdgrad_list)
    
    # 2. Calculate the Global Mean (The Signal)
    global_mean = torch.mean(all_grads_stacked, dim=0)
    
    # 4. Actual Variance: Variance across all N updates
    # We calculate variance for each parameter (dim=0), 
    # then take the mean to get a single scalar representing total noise.
    actual_var = torch.var(all_grads_stacked, dim=0)

    logger.info(f"shape of actual_var: {actual_var.shape}")
    logger.info("--- Gradient Distribution Stats ---")
    logger.info(f"JVP of all updates so far {all_grads_stacked}")
    # log_dist("Signal (Mean)", all_grads_stacked)
    # log_dist("Signal^2", all_grads_stacked**2)
    # log_dist("Variance", actual_var)

    mean_of_mean = torch.mean(global_mean)
    mean_of_mean_2 = torch.mean(global_mean ** 2)
    mean_of_var = torch.mean(actual_var)
    snr = torch.mean((global_mean ** 2) / (actual_var))

    logger.info(f"number of gradient updates: {n} - mean_of_mean : {mean_of_mean} mean_of_mean_squared : {mean_of_mean_2} and  mean_of_var : {mean_of_var} and snr : {snr}")

    return snr.item()

def calculate_cv(fwdgrad_list):
    n = len(fwdgrad_list)
    
    # Does not work for n == 1 (need at least 2 to compute standard deviation)
    if n < 2:
        return 0.0

    # 将所有tensor堆叠在一起
    stacked_grads = torch.stack(fwdgrad_list)

    # 计算所有tensor在各个维度上的标准差 (Standard Deviation: sigma)
    std_dev = torch.std(stacked_grads, dim=0)

    # 计算所有tensor在各个维度上的平均值 (Mean: mu)
    mean_val = torch.mean(stacked_grads, dim=0)

    # 计算变异系数 CV = Std / |Mean|
    # Note: We use torch.abs() because means can be negative, and CV should be positive.
    # Added 1e-8 to prevent division by zero when the mean is exactly 0.
    cv_tensor = std_dev / (torch.abs(mean_val) + 1e-8)

    # 求所有维度的平均值，返回一个标量 (scalar)
    cv = cv_tensor.mean()

    return cv.item()


def calculate_cos_sim(A, target_grad, device):
    batch_size = 1000

    # 计算总批次数
    num_batches = math.ceil(A.size(0) / batch_size)

    # 创建一个空的结果张量
    result = torch.empty(A.size(0))

    # 逐批次计算余弦相似度
    for i in range(num_batches):
        # 获取当前批次的起始索引和结束索引
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, A.size(0))

        # 提取当前批次的向量
        # TODO: See why .to(device) was called here
        batch = A[start_idx:end_idx]  # .to(device)

        # 计算当前批次的余弦相似度
        similarity = torch.cosine_similarity(batch, target_grad, dim=-1)

        # 将结果保存到结果张量的对应位置
        result[start_idx:end_idx] = similarity

    return similarity
