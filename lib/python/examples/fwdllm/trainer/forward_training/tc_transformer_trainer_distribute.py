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
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
)
from functools import partial
import functorch as fc
from torch.cuda.amp import autocast
import gc
import os

logger = logging.getLogger(__name__)


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
            self.grad_pool = []

        # var control TODO: It is not layer id it is param id. Distilbert for eg
        # has only 6 layers.
        self.grad_for_var_check_list = []
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

    ### Old train_model(), works but bloats the CPU Mem
    # def train_model(self, device=None):
    #     if not device:
    #         device = self.device

    #     self.log_memory("start", device)
    #     allocated_before = torch.cuda.memory_allocated(device)

    #     gc.collect()
    #     torch.cuda.empty_cache()

    #     self.fmodel, self.params, self.buffers = fc.make_functional_with_buffers(
    #         self.model
    #     )
    #     self.buffers = [b.to(device) for b in self.buffers]
    #     self.log_memory("after_fmodel_setup", device)

    #     global_step, tr_loss = 0, 0.0

    #     v_buffer = {}
    #     if self.args.perturbation_sampling:
    #         v_num = len(self.train_dl)
    #         if self.args.var_control:
    #             self.grad = self.old_grad

    #         for idx, (k, v) in enumerate(self.model.named_parameters()):
    #             if self.grad is not None and v.requires_grad:
    #                 candidate_v = torch.randn(
    #                     (v_num * 10, *v.shape), device="cpu"
    #                 ).flatten(start_dim=1)
    #                 target_grad = self.grad[idx].flatten()
    #                 cos_sim = calculate_cos_sim(candidate_v, target_grad, device)
    #                 sorted_indices = torch.topk(cos_sim, v_num).indices
    #                 v_buffer[idx] = [
    #                     candidate_v[i].reshape(v.shape) for i in sorted_indices
    #                 ]
    #                 del candidate_v, target_grad, cos_sim, sorted_indices
    #     self.grad = [torch.zeros_like(p, device="cpu") for p in self.params]
    #     with torch.no_grad():
    #         for epoch in range(self.args.epochs):
    #             for batch_idx, batch in enumerate(self.train_dl):
    #                 self.log_memory(f"epoch{epoch}_batch{batch_idx}_start", device)

    #                 x = batch[1].to(device, non_blocking=True)
    #                 labels = batch[4].to(device, non_blocking=True)

    #                 if self.args.perturbation_sampling and v_buffer:
    #                     v_params = [
    #                         (
    #                             v_buffer[i][batch_idx].to(device)
    #                             if p.requires_grad
    #                             else torch.zeros_like(p)
    #                         )
    #                         for i, p in enumerate(self.params)
    #                     ]
    #                 else:
    #                     v_params = [
    #                         (
    #                             torch.randn_like(p, device=device)
    #                             if p.requires_grad
    #                             else torch.zeros_like(p, device=device)
    #                         )
    #                         for p in self.params
    #                     ]

    #                 def wrapped_func(p):
    #                     return functional_get_loss(
    #                         p,
    #                         self.fmodel,
    #                         x,
    #                         labels,
    #                         num_classes=self.num_labels,
    #                         buffers=self.buffers,
    #                     )

    #                 loss, jvp = calculate_jvp(wrapped_func, self.params, v_params)
    #                 jvp = jvp.to(device)

    #                 for j, g in enumerate(self.grad):
    #                     updated = (jvp * v_params[j]).detach().cpu()
    #                     g.add_(updated)
    #                     if self.args.var_control and j == self.layer_id_for_check:
    #                         self.grad_for_var_check = updated.clone()

    #                 for p, g in zip(self.model.parameters(), self.grad):
    #                     if p.requires_grad:
    #                         p.grad = g.clone().to(device)

    #                 current_loss = loss.item()
    #                 tr_loss += current_loss
    #                 global_step += 1
    #                 logging.info(
    #                     f"epoch = {epoch}, trainer_id = {self.trainer_id}, loss = {current_loss}"
    #                 )

    #                 if (
    #                     self.args.evaluate_during_training
    #                     and global_step % self.args.evaluate_during_training_steps == 0
    #                 ):
    #                     self.eval_model(epoch, global_step)

    #                 if self.args.is_debug_mode == 1 and global_step > 3:
    #                     break

    #                 for p in self.model.parameters():
    #                     p.grad = None
    #                 del x, labels, jvp, v_params, loss
    #                 gc.collect()
    #                 torch.cuda.empty_cache()
    #                 self.log_memory(f"epoch{epoch}_batch{batch_idx}_end", device)

    #     if self.args.var_control and self.args.perturbation_sampling:
    #         self.grad_pool.append(self.grad)

    #     trainable_params = [p for p in self.model.parameters() if p.requires_grad]
    #     gradients = [p.grad for p in trainable_params if p.grad is not None]
    #     logging.info(
    #         f"Trainable parameters: {len(trainable_params)} | Size: {human_readable_size(get_size_in_bytes(trainable_params))}"
    #     )
    #     logging.info(
    #         f"Total parameters: {len(list(self.model.parameters()))} | Size: {human_readable_size(get_size_in_bytes(list(self.model.parameters())))}"
    #     )
    #     logging.info(
    #         f"Gradients: {len(gradients)} | Size: {human_readable_size(get_size_in_bytes(gradients))}"
    #     )

    #     # Final cleanup
    #     del self.fmodel, self.params, self.buffers
    #     self.fmodel, self.params, self.buffers = None, None, None

    #     if self.args.perturbation_sampling:
    #         del v_buffer

    #     self.grad = [g.detach().cpu() for g in self.grad]
    #     if hasattr(self, "grad_for_var_check"):
    #         self.grad_for_var_check = self.grad_for_var_check.detach().cpu()

    #     gc.collect()
    #     torch.cuda.empty_cache()

    #     allocated_after = torch.cuda.memory_allocated(device)
    #     self.log_memory("end", device)
    #     logging.info(
    #         f"[MEM] Allocated Before/After: {allocated_before/1e6:.2f}MB → {allocated_after/1e6:.2f}MB, Δ: {(allocated_after-allocated_before)/1e6:.2f}MB | trainer id: {self.trainer_id}"
    #     )

    #     return global_step, tr_loss / global_step if global_step > 0 else 0.0

    ## TODO: Needs checking. Newer main memory optimized version of train_model().
    def train_model(self, device=None):
        if not device:
            device = self.device

        self.log_memory("start", device)
        allocated_before = torch.cuda.memory_allocated(device)

        gc.collect()
        torch.cuda.empty_cache()

        self.fmodel, self.params, self.buffers = fc.make_functional_with_buffers(
            self.model
        )
        self.buffers = [b.to(device) for b in self.buffers]
        self.log_memory("after_fmodel_setup", device)

        global_step, tr_loss = 0, 0.0
        v_buffer = {}

        if self.args.perturbation_sampling:
            v_num = len(self.train_dl)
            if self.args.var_control:
                self.grad = self.old_grad

            for idx, (k, v) in enumerate(self.model.named_parameters()):
                if self.grad is not None and v.requires_grad:
                    candidate_v = torch.randn(
                        (v_num * 10, *v.shape), device="cpu"
                    ).flatten(start_dim=1)
                    target_grad = self.grad[idx].flatten()
                    cos_sim = calculate_cos_sim(candidate_v, target_grad, device)
                    sorted_indices = torch.topk(cos_sim, v_num).indices
                    v_buffer[idx] = [
                        candidate_v[i].reshape(v.shape) for i in sorted_indices
                    ]
                    del candidate_v, target_grad, cos_sim, sorted_indices
            gc.collect()

        # Initialize grad and reuse it across batches
        if not hasattr(self, "grad") or self.grad is None:
            self.grad = [torch.zeros_like(p, device="cpu") for p in self.params]
        else:
            for g in self.grad:
                g.zero_()

        with torch.no_grad():
            for epoch in range(self.args.epochs):
                for batch_idx, batch in enumerate(self.train_dl):
                    self.log_memory(f"epoch{epoch}_batch{batch_idx}_start", device)

                    x = batch[1].to(device, non_blocking=True)
                    labels = batch[4].to(device, non_blocking=True)

                    if self.args.perturbation_sampling and v_buffer:
                        v_params = [
                            (
                                v_buffer[i][batch_idx].to(device)
                                if p.requires_grad
                                else torch.zeros_like(p, device=device)
                            )
                            for i, p in enumerate(self.params)
                        ]
                    else:
                        v_params = [
                            (
                                torch.randn_like(p, device=device)
                                if p.requires_grad
                                else torch.zeros_like(p, device=device)
                            )
                            for p in self.params
                        ]

                    def wrapped_func(p):
                        return functional_get_loss(
                            p,
                            self.fmodel,
                            x,
                            labels,
                            num_classes=self.num_labels,
                            buffers=self.buffers,
                        )

                    loss, jvp = calculate_jvp(wrapped_func, self.params, v_params)
                    jvp = jvp.to(device)

                    for j, g in enumerate(self.grad):
                        if self.params[j].requires_grad:
                            updated = (jvp * v_params[j]).detach().cpu()
                            g.add_(updated)
                            if self.args.var_control and j == self.layer_id_for_check:
                                self.grad_for_var_check = updated.clone()
                            del updated  # explicitly release

                    for p, g in zip(self.model.parameters(), self.grad):
                        if p.requires_grad:
                            if p.grad is None:
                                p.grad = g.to(device)
                            else:
                                p.grad.copy_(g.to(device))

                    current_loss = loss.item()
                    tr_loss += current_loss
                    global_step += 1
                    logging.info(
                        f"epoch = {epoch}, trainer_id = {self.trainer_id}, loss = {current_loss}"
                    )

                    if (
                        self.args.evaluate_during_training
                        and global_step % self.args.evaluate_during_training_steps == 0
                    ):
                        self.eval_model(epoch, global_step)

                    if self.args.is_debug_mode == 1 and global_step > 3:
                        break

                    # Clear gradients explicitly and release memory
                    for p in self.model.parameters():
                        p.grad = None

                    del x, labels, jvp, v_params, loss
                    gc.collect()
                    torch.cuda.empty_cache()
                    self.log_memory(f"epoch{epoch}_batch{batch_idx}_end", device)

        if self.args.var_control and self.args.perturbation_sampling:
            self.grad_pool.append(
                [g.clone() for g in self.grad]
            )  # Deepcopy to avoid side effects

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

        # Final cleanup
        del self.fmodel, self.params, self.buffers
        self.fmodel, self.params, self.buffers = None, None, None
        if self.args.perturbation_sampling:
            del v_buffer

        # Move gradients to CPU and detach
        self.grad = [g.detach().cpu() for g in self.grad]
        if hasattr(self, "grad_for_var_check"):
            self.grad_for_var_check = self.grad_for_var_check.detach().cpu()

        gc.collect()
        torch.cuda.empty_cache()

        allocated_after = torch.cuda.memory_allocated(device)
        self.log_memory("end", device)
        logging.info(
            f"[MEM] Allocated Before/After: {allocated_before/1e6:.2f}MB → {allocated_after/1e6:.2f}MB, Δ: {(allocated_after - allocated_before)/1e6:.2f}MB | trainer id: {self.trainer_id}"
        )

        return global_step, tr_loss / global_step if global_step > 0 else 0.0

    def eval_model(self, epoch=0, global_step=0, device=None):
        if not device:
            device = self.device

        self.model.eval()
        self.fmodel, self.params, self.buffers = fc.make_functional_with_buffers(
            self.model
        )

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
                torch.cuda.empty_cache()
                gc.collect()

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

        # Free memory
        del self.fmodel, self.params, self.buffers
        self.fmodel, self.params, self.buffers = None, None, None
        gc.collect()
        torch.cuda.empty_cache()

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
