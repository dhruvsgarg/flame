# coding: utf-8

from __future__ import absolute_import, division, print_function

from hashlib import shake_128
import logging
import psutil
import time
import numpy as np
import sklearn
import torch
from torch import nn
from examples.fwdllm.trainer.utils.text_classification_utils import *
from examples.fwdllm.trainer.forward_training.fwdgrad_utils import *
from torch.nn import CrossEntropyLoss
# AdamW was removed from the transformers top-level namespace in v4.x; it now
# lives in torch.optim. (Imported for API compatibility; the forward-mode path
# does not actually step an optimizer.)
from torch.optim import AdamW  # noqa: F401
from transformers import get_linear_schedule_with_warmup
from functools import partial
import functorch as fc
import gc
import os
import contextlib
from flame.monitor.runtime import timer_decorator, FwdLLMStage

logger = logging.getLogger(__name__)

import hashlib
import copy


def _rng_state_hash(gen: torch.Generator, device=None):
    """Return a short hash of RNG state for logging."""
    state = gen.get_state()
    return hashlib.sha256(state.numpy().tobytes()).hexdigest()


def _calculate_hash(tensor):
    if tensor is None:
        return ""

    """Calculate a hash for a tensor for logging."""
    return hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest()


@contextlib.contextmanager
def _stage_timer(owner, name: str):
    """Emit a `step_timing` record for a named wall phase of the training step.

    timer_decorator can't do this: it does `self = args[0]` and only emits when
    that carries `fwd_llm_stage`, but every helper inside `_train_one_batch` is
    a nested function whose first arg is `device` -- so the batch interior was
    invisible in telemetry. Same event shape as timer_decorator, so the
    `step_timing_breakdown` rung and the phase-CDF plots read these for free.
    `tb_` prefix keeps the family greppable and collision-free.
    """
    t0 = time.time()
    try:
        yield
    finally:
        dur = time.time() - t0
        stage = getattr(owner, "fwd_llm_stage", None)
        if stage is not None:
            try:
                from flame import telemetry
                if telemetry.is_enabled():
                    from flame.telemetry.events import build_step_timing
                    ev, fields = build_step_timing(
                        func=name, duration_s=dur,
                        round_num=stage.round_id, data_id=stage.data_id,
                        iteration=stage.iteration, trainer_id=stage.trainer_id,
                    )
                    telemetry.emit(ev, **fields)
            except Exception:  # pragma: no cover - never break training
                logging.debug("stage_timer telemetry emit failed", exc_info=True)


def _pert_audit_enabled() -> bool:
    """Is the perturbation determinism audit ([RNG_FINGERPRINT] + the rolling
    candidate_v hash) wanted this run?

    These sha256 every 10x perturbation tensor and the whole 67M-param model --
    together ~10x the cost of the JVP they audit -- so they can't ride along
    unconditionally. Opt in with FWDLLM_PERT_AUDIT=1, or by enabling DEBUG.
    """
    if os.environ.get("FWDLLM_PERT_AUDIT", "").strip().lower() in ("1", "true", "yes"):
        return True
    return logging.getLogger().isEnabledFor(logging.DEBUG)


def _torch_rng_fingerprint(generator: "torch.Generator") -> str:
    """Short hex digest of a torch.Generator's internal state, for
    determinism audits: two same-seed (same client_idx) runs whose
    fingerprint differs at the same call site proves the perturbation RNG
    consumed a different number of draws before that point."""
    return _calculate_hash(generator.get_state())[:12]


def _calculate_rolling_hash(tensor: torch.Tensor, hash_str: str) -> str:
    """Calculate a rolling hash for a tensor for logging."""
    if hash_str is None:
        return _calculate_hash(tensor)

    if tensor is None:
        return ""

    # Encode the string to bytes before concatenating
    return hashlib.sha256(
        tensor.detach().cpu().numpy().tobytes() + hash_str.encode("utf-8")
    ).hexdigest()


def _randn_wrapper(
    *size,
    device=None,
    generator=None,
    label="randn",
    logging_state=None,
    param_name=None,
    **kwargs,
):
    """Wrapper for torch.randn that logs device + RNG info."""
    if device is None:
        device = torch.device("cpu")
    else:
        device = torch.device(device)

    # Pick generator: provided or default one for this device
    if generator is None:
        if device.type == "cpu":
            gen = torch.default_generator
        else:
            gen = torch.cuda.default_generators[device.index]
    else:
        gen = generator

    # f-string args evaluate BEFORE logging.debug() checks the level, so both
    # hashes ran on every call with DEBUG off: 12.3us of hashing to guard a
    # disabled log line vs 5.7us of actual RNG work, once per model parameter
    # per perturbation pass.
    _dbg = logging.getLogger().isEnabledFor(logging.DEBUG)
    pre_state = _rng_state_hash(gen) if _dbg else None

    res = torch.randn(*size, device=device, generator=gen, **kwargs)

    if _dbg:
        logging.debug(
            f"[{label}] device={device}, generator={gen}, post_state={_rng_state_hash(gen)}, logging_state={logging_state}, size={size}, kwargs={kwargs}, pre_state={pre_state}, param_name={param_name}"
        )
    return res




class ForwardTextClassificationTrainer:
    def __init__(
        self, args, device, model, train_dl=None, test_dl=None, trainer_id=None
    ):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        gpu_id = torch.cuda.current_device()

        # Get free and total memory on the selected CUDA device
        free_mem, total_mem = torch.cuda.mem_get_info(gpu_id)

        # Convert to MB for easier reading
        free_mb = free_mem / (1024 * 1024)
        total_mb = total_mem / (1024 * 1024)

        print(
            f"[GPU Memory Info] Device: {self.device}, logical_device_id: {device}, Free: {free_mb:.2f} MB / Total: {total_mb:.2f} MB, Occupied: {(total_mb-free_mb):.2f} MB"
        )

        device_name = torch.cuda.get_device_name(gpu_id)

        real_index = (
            visible_devices.split(",")[gpu_id] if visible_devices else str(gpu_id)
        )

        logging.info(
            f"[Device Init] CUDA_VISIBLE_DEVICES={visible_devices}, "
            f"torch.device={self.device}, torch.cuda.current_device={gpu_id}, "
            f"Real GPU Index (Global) = {real_index}, Device Name: {device_name}"
        )

        self.loss_fn = None
        self.dataset_size = 0
        self.trainer_id = trainer_id
        # set data
        self.num_labels = args.num_labels
        self.set_data(train_dl, test_dl)

        # model
        self.model = model
        if self.args.model_type == "distilbert":
            self.model.add_module("pre_classifier", nn.Sequential())
        # self.model.to(self.device)

        # training results
        self.results = {}
        self.best_accuracy = 0.0

        # freeze
        self.freeze_layers = args.freeze_layers.split(",") if args.freeze_layers else []

        self.grad = None
        if self.args.perturbation_sampling and self.args.var_control:
            self.old_grad = None

        self.select_perturbation_using_jvp = False
        if self.args.select_perturbation_using_jvp:
            self.select_perturbation_using_jvp = self.args.select_perturbation_using_jvp

        # Number of candidate perturbations sampled per param. Drives the
        # forward-pass count: the select_perturbation_using_jvp path does 2 JVP
        # passes per perturbation, so N perturbations = ~2N passes (dominant
        # fluxtune GPU cost). Default 10 (byte-identical to the historical
        # hardcode); a knob so JVP cost can be tuned to keep GPU << the modeled
        # mobile delay. Real and sim must use the same value (same config).
        try:
            self.perturbation_count = int(getattr(self.args, "perturbation_count", 10) or 10)
        except (TypeError, ValueError):
            self.perturbation_count = 10

        # Fluxtune JVP perf optimizations (simulate_fwdllm.md §L) — all
        # bit-identical to the current grads:
        #   (a) trainable-only finite difference (skip frozen p-h*0=p);
        #   (b) skip the 3 diagnostic-only forward passes (loss logging only);
        #   (c) reuse the selected perturbation's JVP computed in selection.
        # Config-gated, default OFF (byte-identical); enabled only in the fluxtune
        # yamls (`jvp_perf_opt: true`). Real and sim must match (same config).
        self.jvp_perf_opt = bool(getattr(self.args, "jvp_perf_opt", False))
        self._sel_jvp_cache = {}
        logging.info(
            f"[JVP_PERF_OPT] jvp_perf_opt={self.jvp_perf_opt} "
            f"(trainable-only FD + skip diagnostic passes + reuse winner JVP; "
            f"all bit-identical — simulate_fwdllm.md §L)"
        )

        # var control TODO: It is not layer id it is param id. Distilbert for eg
        # has only 6 layers.
        if self.args.model_type == "distilbert":
            self.layer_id_for_check = 20
        elif self.args.model_type == "bert":
            self.layer_id_for_check = 12
        elif self.args.model_type == "roberta-large":
            self.layer_id_for_check = 12
        elif self.args.model_type == "albert":
            self.layer_id_for_check = 22
        self.var = 0
        logger.info(f"Client Trainer learning rate: {self.args.learning_rate}")

        # Initialized RNGs with client_ids and the exact same static seed (42), to avoid all trainers generating the same sequence of perturbations, which was stalling the accuracy increase
        self.torch_rng = torch.Generator(device="cpu")
        self.torch_rng.manual_seed(self.args.client_idx)
        self.torch_cuda_rng = torch.Generator(device="cuda")
        self.torch_cuda_rng.manual_seed(self.args.client_idx)

        self.total_rng_iter = 0

        # Optimization: cache fmodel, params, and buffers to avoid recreation
        self.fmodel = None
        self.params = None
        self.buffers = None
        self.grad_for_var_check = None
        self.jvp_for_snr_check = None
        self.databin_best_jvp_val = 0.0
        self.databin_best_v_params = None
        self.last_model_version_jvp_updated = -1


    # def initialize(self) -> None: """Initialize role.""" self.device =
    #     torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #     # self.model = Net().to(self.device)
    #     logging.info(f"Task_id: {self.trainer_id} initialize completed at
    #                  timestamp: " f"{time.time()}")

    def set_data(self, train_dl=None, test_dl=None):
        # Used for fedtrainer
        self.train_dl = train_dl
        self.test_dl = test_dl

    def log_memory(self, tag, device):
        # GPU memory
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)

        # CPU memory
        process = psutil.Process()
        cpu_memory = process.memory_info().rss  # in bytes

        logging.info(
            f"[MEM:{tag}] "
            f"GPU Allocated: {allocated/1e6:.2f} MB | "
            f"GPU Reserved: {reserved/1e6:.2f} MB | "
            f"CPU Memory: {cpu_memory/1e6:.2f} MB | "
            f"Device: {device}, trainer_id: {self.trainer_id}"
        )

    def compute_metrics_with_logging_train(self, x, labels):

        logging.info(f"Trainer ID |  'Example'  | 'Label")

        for j, example in enumerate(x):
            logging.info(f"trainer: {self.trainer_id} | {_calculate_hash(example)}... | {labels[j]} ")
        return
    
    @timer_decorator
    def _make_model_functional(self, device):
        # Ensure model is on the correct device
        self.model.to(device)               # TODO(Gaurav): does this need to be in CPU?
        
        self.fmodel, self.params, self.buffers = fc.make_functional_with_buffers(
            self.model
        )
        self.params = [p.to(device) for p in self.params]    # In case it was moved to CPU for serialization before being sent over the channel
        self.buffers = [b.to(device) for b in self.buffers]

    @timer_decorator
    # Removed 2026-07-15: the unreachable `_select_optimal_perturbations` and
    # `_setup_training_state` METHODS lived here (103 lines). Nothing ever called
    # them -- `_train_one_batch` defines and calls its own nested copies, which
    # are the live ones. They were stale forks that had drifted from the live
    # code (e.g. they still gated on `self.grad is not None`), so every read of
    # this file had to first work out which of three near-identical copies
    # actually runs. Deleted rather than left to rot further.



    @timer_decorator

    @timer_decorator
    def _train_one_batch(self, device, batch, epoch, batch_idx, logging_state):
        @timer_decorator
        def _setup_training_state( device, logging_state, x, labels):
            @timer_decorator
            def _select_optimal_perturbations(device, logging_state ):
                if logging_state.get('data_id') != self.last_model_version_jvp_updated :
                    self.databin_best_jvp_val = 0.0 # reset after a databin is complete
                    self.databin_best_v_params = None
                    self.last_model_version_jvp_updated = logging_state.get('data_id')
                    logging.info(f"data_id_iteration {logging_state.get('iteration')} - resetting")
                
                if self.args.var_control:
                    # 126 CPU-tensor clones of the full 67M-param grad shape
                    # (old_grad arrives from the aggregator on CPU); ~51ms in
                    # the offline profile, and only on odd data_ids (old_grad is
                    # None on even ones -- fwdllm_trainer.py `data_id % 2`).
                    with _stage_timer(self, "tb_grad_clone"):
                        self.grad = None if self.old_grad is None else [g.clone() for g in self.old_grad]

                # v_all_pert = []
                v_buffer = {}
                all_perturbations_hash = ""
                index = 0
                # [RNG_FINGERPRINT] is a determinism tool, not run telemetry:
                # it costs a full sha256 walk of every 10x candidate_v, so it is
                # opt-in. Off, the fingerprints report "off" rather than lying.
                _pert_audit = _pert_audit_enabled()
                rng_before = _torch_rng_fingerprint(self.torch_rng) if _pert_audit else "off"
                for k, v in self.model.named_parameters():
                    if v.requires_grad:
                        self.total_rng_iter += 1
                        shape = v.shape
                        # perturbation_count(10) x this param, drawn on CPU then
                        # moved to device downstream. ~40ms/batch offline across
                        # the 26 trainable tensors.
                        with _stage_timer(self, "tb_perturb_draw_cpu"):
                            candidate_v = _randn_wrapper((self.perturbation_count, *shape), device="cpu", generator=self.torch_rng, logging_state=logging_state, param_name=k)
                            candidate_v = torch.flatten(candidate_v, start_dim=1)
                        # Dropped a per-trainable-tensor INFO of the constant
                        # perturbation_count (26 x 4902 = 127k lines/run, the bulk
                        # of the 184-219MB trainer logs). Both hashes below walk
                        # the full 10x tensor and are audit-only.
                        if _pert_audit:
                            logging.debug(f"candidate_v for client_idx {self.args.client_idx} is {_calculate_hash(candidate_v)} for param_name {k}")
                            all_perturbations_hash = _calculate_rolling_hash(candidate_v, all_perturbations_hash)

                        if not self.select_perturbation_using_jvp and self.grad is not None:
                            # cosine-sim rank of the 10 candidates against the
                            # carried grad -- runs on CPU (calculate_cos_sim's
                            # `.to(device)` is commented out and old_grad arrives
                            # on CPU), ~54ms/batch offline.
                            with _stage_timer(self, "tb_cos_sim_select"):
                                target_grad = self.grad[index]
                                target_grad = torch.flatten(target_grad)
                                cos_sim = calculate_cos_sim(candidate_v, target_grad, device)
                                sorted_values, sorted_indices = torch.sort(cos_sim, descending=True)
                            logging.debug("cos sim values for trainer %s: %s", self.trainer_id, sorted_values)
                            v_buffer[index] = [
                                candidate_v[i].reshape(v.shape) for i in sorted_indices[:1]
                            ]
                            del candidate_v, target_grad, cos_sim, sorted_indices, shape
                        else:
                            v_buffer[index] = [
                                candidate_v[i].reshape(v.shape) for i in range(0, self.perturbation_count)
                            ]
                            del candidate_v, shape
                    index += 1

                if _pert_audit:
                    logging.info(
                        f"[RNG_FINGERPRINT] client_idx={self.args.client_idx} "
                        f"data_id={logging_state.get('data_id')} iteration={logging_state.get('iteration')} "
                        f"torch_rng before={rng_before} after={_torch_rng_fingerprint(self.torch_rng)} "
                        f"all_perturbations_hash={all_perturbations_hash}"
                    )

                if not self.select_perturbation_using_jvp:
                    return v_buffer, 0 # we add only the best cos sim values here
                

                jvp_all_perturbations = []
                # perf-opt: cache each perturbation's (loss, jvp) so the winner's
                # JVP is reused below instead of recomputed (2 fewer passes, §L).
                self._sel_jvp_cache = {}
                for i in range(0, self.perturbation_count):
                    v_params = _prepare_perturbation_tensors(device, v_buffer, i)
                    loss, jvp = _compute_forward_jvp(device, x, labels, v_params)
                    # logging.info(f"Jvp of option: {jvp}")
                    jvp_all_perturbations.append(jvp)
                    if self.jvp_perf_opt:
                        self._sel_jvp_cache[i] = (loss, jvp)

                logging.info(f"Number of pert and jvps: {len(jvp_all_perturbations)}")
                
                sorted_indices = [i for i, v in sorted(enumerate(jvp_all_perturbations), key=lambda x: abs(x[1]))]
                sorted_jvps = [jvp_all_perturbations[i] for i in sorted_indices]

                # carefully delete this condition, we do not want to use best across iterations - this reduces exploration
                # if 0.8 * self.databin_best_jvp_val > abs(sorted_jvps[-1]):
                if False:
                    best_idx = -1
                    logging.info(f"Databin best jvp so far: {self.databin_best_jvp_val} - best this iteration: {abs(sorted_jvps[-1])}")
                else:
                    # Draw from the client's dedicated, client_idx-seeded
                    # torch_rng -- NOT np.random.choice (process-global,
                    # unseeded on the trainer side; would silently break
                    # real<->real / real<->sim reproducibility the moment
                    # select_perturbation_using_jvp=True is exercised).
                    pair = [sorted_indices[-1], sorted_indices[-2]]
                    pick = int(torch.randint(0, 2, (1,), generator=self.torch_rng).item())
                    best_idx = pair[pick]
                    self.databin_best_jvp_val = abs(sorted_jvps[-1])
                    logging.info(f"All JVPs sorted by magnitude: {sorted_jvps} and chosen jvp: {jvp_all_perturbations[sorted_indices[-1]]} for trainer : {self.trainer_id} for model version: {logging_state.get('round_id')} data-id: {logging_state.get('data_id')}. iteration: {logging_state.get('iteration')}")
  
                return v_buffer, best_idx # v_buffer here contains all perturbations

            self.log_memory("after_fmodel_setup", device)

            v_buffer = {}
            # Perturbation selection logic slightly differs from the vanilla FwdLLM implementation. Their logic has a flaw which cannot be used in a true-FL setting 
            # with distributed clients. As their clients are emulated in a for loop, they generate `num_clients` * 10 candidate perturbations & select the top `num_clients`
            # perturbations based on cosine similarity. This is not the same as generating 10 candidate perturbations per client & selecting the top 1. We did not
            # observe any significant changes in accuracy, after assigning clients distinct RNG seeds, & hence we chose the later approach.
            if self.args.perturbation_sampling:
                v_buffer, best_idx = _select_optimal_perturbations(device, logging_state)

            # Efficient grad allocation / zeroing
            if (
                not hasattr(self, "grad")
                or self.grad is None
                or len(self.grad) != len(self.params)
            ):
                # Optimization: Initialize on device to avoid Host to Device transfer every batch
                self.grad = [torch.zeros_like(p, device=device) for p in self.params]
            else:
                # Ensure gradients are on the correct device (they might have been moved to CPU in a previous round)
                self.grad = [fg.to(device).zero_() for fg in self.grad]
                
            return v_buffer, best_idx
        

        @timer_decorator
        def _compute_batch_stat_utility(device, x, labels):

            with torch.no_grad():
                pred = self.model(x)
                if hasattr(pred, "logits"):
                    logits = pred.logits
                elif isinstance(pred, (tuple, list)):
                    logits = pred[0]
                else:
                    logits = pred
                loss = self.base_trainer.oort_loss(logits, labels.view(-1), epoch=0, batch_idx=0, reduction="mean")
            # Optimization: Lazy logging & removed .item() to avoid Host-device sync
            logging.debug("stat_utility for trainerId: %s is %s, loss: %s", self.trainer_id, self.base_trainer._stat_utility, loss.mean())

        @timer_decorator
        def _prepare_perturbation_tensors(device, v_buffer, idx=0):
            if self.args.perturbation_sampling and v_buffer != {}:
                logging.debug(f"V buffer is populated")
                v_params = [
                    (
                        v_buffer[i][idx].to(device)
                        if p.requires_grad
                        else torch.zeros_like(p).to(device)
                    )
                    for i, p in enumerate(self.params)
                ]
            else:
                logging.debug(f"V buffer empty, creating random perturbations")
                v_params = [
                    (
                        torch.randn_like(p, device=device)
                        if p.requires_grad
                        else torch.zeros_like(p, device=device)
                    )
                    for p in self.params
                ]

            # Second copy of the same audit pair (see _train_one_batch); the
            # unfiltered 67M-param hash ran twice per batch, both eagerly.
            if logging.getLogger().isEnabledFor(logging.DEBUG):
                logging.debug(
                    f"v_params hashes: {[(_calculate_hash(v), v.shape) for v in v_params if v.requires_grad]}"
                )
                logging.debug(
                    f"params hashes: {[(_calculate_hash(p), p.shape) for p in self.params]}"
                )

            return v_params

        @timer_decorator
        def _compute_forward_jvp(device, x, labels, v_params):
            # def wrapped_func(p):
            #     return functional_get_loss(
            #         p,
            #         self.fmodel,
            #         x,
            #         labels,
            #         num_classes=self.num_labels,
            #         buffers=self.buffers,
            #     )
            # loss, jvp = calculate_jvp_experiment(wrapped_func, self.params, v_params)
            
            f = partial(
                functional_get_loss,
                model=self.fmodel,
                buffers = self.buffers,
                num_classes = self.num_labels,
                x=x,
                t=labels,
            )

            # Perf-opt (fluxtune): perturb only trainable params — bit-identical
            # since v=0 on frozen params (p-h*0=p). None => legacy all-param path.
            _tidx = ([i for i, p in enumerate(self.params) if p.requires_grad]
                     if self.jvp_perf_opt else None)
            loss, jvp = calculate_jvp(f, self.params, v_params, trainable_idx=_tidx)
            jvp = jvp.to(device)
            return loss, jvp
        
        @timer_decorator
        def _compute_loss_after_update(device, x, labels, v_params, jvp_scalar):
            
            f = partial(
                functional_get_loss,
                model=self.fmodel,
                buffers = self.buffers,
                num_classes = self.num_labels,
                x=x,
                t=labels,
            )

            loss = calculate_jvp_after_actual_update(f, self.params, v_params, jvp_scalar)
            return loss
        
        @timer_decorator
        def _compute_loss_before_update(device, x, labels, v_params, jvp_scalar):
            
            f = partial(
                functional_get_loss,
                model=self.fmodel,
                buffers = self.buffers,
                num_classes = self.num_labels,
                x=x,
                t=labels,
            )

            loss = calculate_jvp_before_actual_update(f, self.params)
            return loss

        @timer_decorator
        def _accumulate_and_extract_grads(device, jvp, v_params):
            # Optimization: Accumulate on device. self.grad should be on device.
            for j, fg in enumerate(self.grad):
                updated = (jvp * v_params[j]) # Keep on device
                fg.add_(updated)
                if self.args.var_control and j == self.layer_id_for_check:
                    self.grad_for_var_check = updated.detach().cpu() # Move to CPU only for check

        curr_client_idx = self.args.client_idx
        if batch_idx == 0 and epoch == 0 and not batch[2].is_cuda:
            # batch[2] is typically the attention_mask. Summing it gives the count of non-padding tokens.
            max_seq_len_in_batch = (batch[2] != 0).sum(dim=1).max().item()
            logging.debug(f"Max active sequence length in first batch: {max_seq_len_in_batch}")

        self.log_memory(
            f"epoch{epoch}_batch{batch_idx}_client{curr_client_idx}_start",
            device,
        )

        with _stage_timer(self, "tb_batch_to_device"):
            x = batch[1].to(device, non_blocking=True)
            labels = batch[4].to(device, non_blocking=True)

        # Use this logging for debugging
        # self.compute_metrics_with_logging_train(x,labels)

        # Stat-utility calculation
        with _stage_timer(self, "tb_stat_utility"):
            _compute_batch_stat_utility(device, x, labels)
        with _stage_timer(self, "tb_setup_training_state"):
            v_buffer, best_idx = _setup_training_state(device, logging_state, x, labels)

        if best_idx == -1 and self.databin_best_v_params is not None:
            v_params = self.databin_best_v_params
            logging.debug("Using global best, not using a new perturbation.")
        else:
            with _stage_timer(self, "tb_prepare_perturbation"):
                v_params = _prepare_perturbation_tensors(device, v_buffer, best_idx)
            # deepcopy of all 104 v_param tensors (~257MB incl. the frozen zeros)
            with _stage_timer(self, "tb_deepcopy_best_v"):
                self.databin_best_v_params = copy.deepcopy(v_params)
        # Determinism-audit only: _calculate_hash pulls each tensor GPU->CPU and
        # sha256s it, so ungated these hashed the whole 67M-param model twice per
        # batch (~1050ms) against a ~10ms JVP. Reading a tensor can't perturb it,
        # so gating is numerically inert.
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.debug(f"v_params hashes: {[(_calculate_hash(v), v.shape) for v in v_params if v.requires_grad]}")
            logging.debug(f"params hashes: {[(_calculate_hash(p), p.shape) for p in self.params]}")

        # perf-opt: reuse the winner's JVP already computed during selection
        # (same params + v_params for best_idx -> bit-identical), saving 2
        # passes. Falls back to compute when the cache is absent (cos-sim / sync
        # path never ran the selection loop) or best_idx used the carried
        # global-best v_params (the `best_idx == -1` branch).
        if (self.jvp_perf_opt and best_idx != -1
                and best_idx in getattr(self, "_sel_jvp_cache", {})):
            loss, jvp = self._sel_jvp_cache[best_idx]
        else:
            # THE MATH: calculate_jvp = 2 autocast forward passes (finite
            # difference), ~10ms offline. Everything else in this batch is
            # scaffolding around it.
            with _stage_timer(self, "tb_forward_jvp"):
                loss, jvp = _compute_forward_jvp(device, x, labels, v_params)
        # 3 diagnostic-only passes: their losses ONLY feed the log below (never
        # grads/telemetry), so skip them under perf-opt (bit-identical grads, §L).
        if not self.jvp_perf_opt:
            nonscaled_global_loss = _compute_loss_after_update(device, x, labels, v_params, jvp)
            scaled_global_loss = _compute_loss_after_update(device, x, labels, v_params, jvp/15)
            loss_before_update = _compute_loss_before_update(device, x, labels, v_params, jvp)
            logging.info(f"At trainer: {self.trainer_id} - iteration: {logging_state.get('iteration')} - jvp_magnitude: {jvp} - loss before update: { loss_before_update } - loss after update (not downscaled): {nonscaled_global_loss}  - loss after update (down scaled): {scaled_global_loss}")
        self.jvp_for_snr_check = abs(jvp)
        logging.debug("JVP of the perturbation: %s", jvp)

        # Carries a .detach().cpu() of the var-check layer -- a device sync that
        # dominated this phase in the offline profile (55ms of its 62ms).
        with _stage_timer(self, "tb_accumulate_grads"):
            _accumulate_and_extract_grads(device, jvp, v_params)

        # Optimization: Remove GC & buffer flushes from the batch loop
        # self._force_cuda_memory_cleanup(device, f"epoch{epoch}_batch{batch_idx}_end")
        
        self.base_trainer.normalize_stat_utility(epoch)
        logging.debug(
            f"stat_utility - normalized for trainerId: {self.trainer_id} = {self.base_trainer._stat_utility}"
        )
        
        del x, labels, jvp, v_params
        # self._force_cuda_memory_cleanup(device, f"epoch{epoch}_batch{batch_idx}_end")
        return loss

    @timer_decorator
    def _training_loop(self, device, logging_state):
        global_step = 0
        # Optimization: Use autocast for training loop if enabled
        from torch.cuda.amp import autocast
        autocast_cm = autocast() if self.args.fp16 else contextlib.nullcontext()
        if not self.args.fp16: logging.warning(f"Autocast is disabled: {self.args.fp16}")

        with torch.no_grad(), autocast_cm:
            for epoch in range(self.args.epochs):
                logging.info(f"train_dl size: {len(self.train_dl)}")
                for batch_idx, batch in enumerate(self.train_dl):
                    loss = self._train_one_batch(device, batch, epoch, batch_idx, logging_state)

                    global_step += 1
                    logging.info(
                        f"epoch = {epoch}, trainer_id = {self.trainer_id}, loss = {loss}"
                    )

                    if (
                        self.args.evaluate_during_training
                        and global_step % self.args.evaluate_during_training_steps == 0
                    ):
                        self.eval_model(epoch, global_step)

                    if self.args.is_debug_mode == 1 and global_step > 3:
                        break

                    del batch, loss

    @timer_decorator
    def _finalize_training(self, device):
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        gradients = [p.grad for p in trainable_params if p.grad is not None]
        logging.info(
            f"Trainable parameters: {len(trainable_params)} | Size: {human_readable_size(get_size_in_bytes(trainable_params))}"
        )
        logging.info(
            f"Total parameters: {len(list(self.model.parameters()))} | Size: {human_readable_size(get_size_in_bytes(list(self.model.parameters())))}"
        )
        logging.info(
            f"Gradients: {len(gradients)} | Size: {human_readable_size(get_size_in_bytes(gradients))}"
        )

        # Optimization: Set p.grad only ONCE at the end of training
        for p, fg in zip(self.model.parameters(), self.grad):
            if p.requires_grad:
                p.grad = fg.clone() # Already on device

        self.grad = [fg.detach().cpu() for fg in self.grad]
        if self.grad_for_var_check is not None:
            self.grad_for_var_check = self.grad_for_var_check.detach().cpu()

        allocated_after = torch.cuda.memory_allocated(device)
        self.log_memory("end", device)
        logging.info(
            f"[MEM] Allocated Before/After: {self.allocated_before/1e6:.2f}MB \u2192 {allocated_after/1e6:.2f}MB, \u0394: {(allocated_after-self.allocated_before)/1e6:.2f}MB | trainer id: {self.trainer_id}"
        )

    @timer_decorator

    @timer_decorator
    def train_model(self, device=None, logging_state=None):
        if not device:
            device = self.device

        if logging_state:
            self.fwd_llm_stage = FwdLLMStage(
                logging_state.get("round_id"),
                logging_state.get("data_id"),
                logging_state.get("iteration"),
                self.trainer_id
            )

        self.log_memory("train_model_start", device)
        self.allocated_before = torch.cuda.memory_allocated(device)
        
        """
        If you want absolute determinism between runs, run the model in eval mode. Make sure to switch the model back to train model before the method returns: `self.model.train()`. 
        Even though this seems to not affect training, this is commented as we're not sure how the model trains in eval mode. Any relative impact on accuracy without it isn't measured.
        self.model.eval()
        """
        # No _force_cuda_memory_cleanup() here (same reasoning that already
        # retired the two in-loop calls below): empty_cache() issues cudaFree, a
        # device-wide sync, ~13x/GPU per run, and had nothing to reclaim --
        # allocated is flat at 817MB on a 46GB card across the whole run.
        self.log_memory("before_train_model", device)
        self._make_model_functional(device)
        
        
        self._training_loop(device, logging_state)
        
        self._finalize_training(device)


    @timer_decorator
    def eval_model(self, epoch=0, global_step=0, device=None):
        """
        Evaluate the model and compute metrics.
        
        As a future optimization, consider the following improvements mirroring the aggregator:
        1. Autocasting: Wrap the evaluation loop with `torch.cuda.amp.autocast()` to leverage TensorCores.
        2. Keep on GPU: Accumulate `eval_loss`, `preds`, and `out_label_ids` on GPU as `torch.tensor` 
           and only move to CPU at the end to avoid frequent pipeline flushes.
        3. One-time GPU transfer: Cache evaluation data in GPU memory if it doesn't change between calls.
        """
        if not device:
            device = self.device

        # Ensure model is on the correct device
        self.model.to(device)

        # todo: Make sure that the model doesn't need to be put back into train mode using: `self.model.train()` before this method returns
        self.model.eval()
        # Optimization: use cached functional model components
        self.fmodel, self.params, self.buffers = fc.make_functional_with_buffers(
            self.model
        )
        self.params = [p.to(device) for p in self.params]
        self.buffers = [b.to(device) for b in self.buffers]

        eval_loss, nb_eval_steps = 0.0, 0
        n_batches = len(self.test_dl)
        test_sample_len = len(self.test_dl.dataset)

        preds = np.empty((test_sample_len, self.num_labels))
        out_label_ids = np.empty(test_sample_len)

        logging.info(f"len(test_dl) = {n_batches}, total samples = {test_sample_len}")

        with torch.no_grad():
            for i, batch in enumerate(self.test_dl):
                x = batch[1].to(device, non_blocking=True)
                labels = batch[4].to(device, non_blocking=True)

                output = self.model(x)
                logits = output[0] if isinstance(output, (tuple, list)) else output

                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                eval_loss += loss.item()

                start_index = self.args.eval_batch_size * i
                end_index = min(
                    start_index + self.args.eval_batch_size, test_sample_len
                )

                preds[start_index:end_index] = logits.detach().cpu().numpy()
                out_label_ids[start_index:end_index] = labels.detach().cpu().numpy()

                del x, labels, output, logits, loss
                # Optimization: Remove GC & buffer flushes from the batch loop
                # torch.cuda.empty_cache()
                # gc.collect()

                nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        model_outputs = preds
        preds = np.argmax(preds, axis=1)

        result, wrong = self.compute_metrics(
            preds, out_label_ids, self.test_dl.examples
        )
        result["eval_loss"] = eval_loss
        self.results.update(result)
        logging.info(self.results)

        # Optimization: keep functional components
        # del self.fmodel, self.params, self.buffers
        # self.fmodel, self.params, self.buffers = None, None, None
        
        # Optimization: GC and buffer flushes moved to the framework level.

        return result, model_outputs, wrong

    def compute_metrics(self, preds, labels, eval_examples=None):
        assert len(preds) == len(labels)

        extra_metrics = {}
        extra_metrics["acc"] = sklearn.metrics.accuracy_score(labels, preds)
        mismatched = labels != preds

        if eval_examples:
            wrong = [i for (i, v) in zip(eval_examples, mismatched) if v.any()]
        else:
            wrong = ["NA"]

        mcc = matthews_corrcoef(labels, preds)

        tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
        return (
            {**{"mcc": mcc, "tp": tp, "tn": tn, "fp": fp, "fn": fn}, **extra_metrics},
            wrong,
        )


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {"Total": total_num, "Trainable": trainable_num}


# Convert to human-readable format
def human_readable_size(size_in_bytes):
    for unit in ["B", "KB", "MB", "GB"]:
        if size_in_bytes < 1024:
            return f"{size_in_bytes:.2f} {unit}"
        size_in_bytes /= 1024
    return f"{size_in_bytes:.2f} TB"


def get_size_in_bytes(tensors):
    return sum(t.numel() * t.element_size() for t in tensors)
