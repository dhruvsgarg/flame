# coding: utf-8

from __future__ import absolute_import, division, print_function

from hashlib import shake_128
import logging
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

        print(f"[GPU Memory Info] Device: {self.device}, logical_device_id: {device}, Free: {free_mb:.2f} MB / Total: {total_mb:.2f} MB, Occupied: {(total_mb-free_mb):.2f} MB")
        
        device_name = torch.cuda.get_device_name(gpu_id)

        real_index = visible_devices.split(',')[gpu_id] if visible_devices else str(gpu_id)

        logging.info(f"[Device Init] CUDA_VISIBLE_DEVICES={visible_devices}, "
                    f"torch.device={self.device}, torch.cuda.current_device={gpu_id}, "
                    f"Real GPU Index (Global) = {real_index}, Device Name: {device_name}")

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

    def log_gpu_memory(self, tag, device):
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        logging.info(f"[MEM:{tag}] Allocated: {allocated/1e6:.2f} MB | Reserved: {reserved/1e6:.2f} MB | Device: {device}, trainer_id: {self.trainer_id}")

    def train_model(self, device=None):
        if not device:
            device = self.device

        self.log_gpu_memory("start", device)
        allocated_before = torch.cuda.memory_allocated(device)

        if not hasattr(self, "fmodel") or self.fmodel is None:
            self.fmodel, self.params, self.buffers = fc.make_functional_with_buffers(self.model)
            self.buffers = [b.to(device) for b in self.buffers]
            self.log_gpu_memory("after_make_functional_with_buffers", device)

        gc.collect()
        torch.cuda.empty_cache()
        self.log_gpu_memory("after_initial_gc", device)

        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0

        if self.args.perturbation_sampling:
            v_num = len(self.train_dl)
            v_buffer = {}
            index = 0
            if self.args.var_control:
                self.grad = self.old_grad
            for k, v in self.model.named_parameters():
                if self.grad != None and v.requires_grad:
                    shape = v.shape
                    candidate_v = torch.randn((v_num * 10, *shape), device="cpu")
                    target_grad = self.grad[index].flatten()
                    candidate_v = candidate_v.flatten(start_dim=1)
                    cos_sim = calculate_cos_sim(candidate_v, target_grad, device)
                    sorted_values, sorted_indices = torch.sort(cos_sim, descending=True)
                    v_buffer[index] = [
                        candidate_v[i].reshape(v.shape) for i in sorted_indices[:v_num]
                    ]
                index += 1

        self.grad = [torch.zeros_like(p) for p in self.params]

        with torch.no_grad():
            for epoch in range(0, self.args.epochs):
                logging.debug(f"train_dl size: {len(self.train_dl)}")
                for batch_idx, batch in enumerate(self.train_dl):
                    self.log_gpu_memory(f"epoch{epoch}_batch{batch_idx}_start", device)

                    batch = tuple(t for t in batch)
                    x = batch[1].to(device)
                    labels = batch[4].to(device)
                    self.log_gpu_memory(f"epoch{epoch}_batch{batch_idx}_after_data_move", device)

                    if self.args.perturbation_sampling and v_buffer != {}:
                        v_params = tuple(
                            [
                                (
                                    v_buffer[i][batch_idx]
                                    if p.requires_grad
                                    else torch.zeros_like(p)
                                )
                                for i, p in enumerate(self.params)
                            ]
                        )
                    else:
                        v_params = tuple(
                            [
                                torch.randn_like(p) if p.requires_grad else torch.zeros_like(p)
                                for p in self.params
                            ]
                        )

                    def wrapped_func(p):
                        return functional_get_loss(
                            p,
                            self.fmodel,
                            x,
                            labels,
                            num_classes=self.num_labels,
                            buffers=self.buffers
                        )

                    loss, jvp = calculate_jvp(wrapped_func, self.params, v_params)
                    self.log_gpu_memory(f"epoch{epoch}_batch{batch_idx}_after_jvp", device)

                    if isinstance(jvp, torch.Tensor):
                        jvp = jvp.to(device)
                    else:
                        jvp = torch.tensor(jvp, device=device)

                    v_params = [v.to(device) for v in v_params]
                    self.grad = [g.to(device) for g in self.grad]

                    for j, fg in enumerate(self.grad):
                        fg.add_(jvp * v_params[j])
                        if self.args.var_control and j == self.layer_id_for_check:
                            self.grad_for_var_check = jvp * v_params[j]

                    for p, g in zip(self.model.parameters(), self.grad):
                        if p.requires_grad:
                            p.grad = g.clone()

                    current_loss = loss.item()
                    logging.info(f"epoch = {epoch}, trainer_id = {self.trainer_id}, loss = {current_loss}")
                    global_step += 1

                    if self.args.evaluate_during_training and (
                        self.args.evaluate_during_training_steps > 0
                        and global_step % self.args.evaluate_during_training_steps == 0
                    ):
                        results, _, _ = self.eval_model(epoch, global_step)

                    if self.args.is_debug_mode == 1 and global_step > 3:
                        break
                    del jvp, v_params, loss, x, labels
                    
                    
                    gc.collect()
                    torch.cuda.empty_cache()
                    self.log_gpu_memory(f"epoch{epoch}_batch{batch_idx}_after_cleanup", device)

        if self.args.var_control and self.args.perturbation_sampling:
            self.grad_pool.append(self.grad)

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        total_params = [p for p in self.model.parameters()]
        gradients = [p.grad for p in trainable_params if p.grad is not None]

        trainable_size = get_size_in_bytes(trainable_params)
        total_params_size = get_size_in_bytes(total_params)
        gradient_size = get_size_in_bytes(gradients)

        logging.info(f"Trainable parameters: {len(trainable_params)} | Size: {human_readable_size(trainable_size)}")
        logging.info(f"Total parameters: {len(total_params)} | Size: {human_readable_size(total_params_size)}")
        logging.info(f"Gradients: {len(gradients)} | Size: {human_readable_size(gradient_size)}")

        del self.fmodel, self.params, self.buffers
        if self.args.perturbation_sampling:
            del v_buffer
        self.fmodel, self.params, self.buffers = None, None, None
        self.grad = [g.detach().cpu() for g in self.grad]
        if hasattr(self, "grad_for_var_check"):
            self.grad_for_var_check = self.grad_for_var_check.detach().cpu()
        
        gc.collect()
        torch.cuda.empty_cache()

        allocated_after = torch.cuda.memory_allocated(device)
        self.log_gpu_memory("end", device)

        logging.info(f"[MEM] Allocated Before/After: {allocated_before/1e6:.2f}MB → {allocated_after/1e6:.2f}MB, Added/Removed: {(allocated_after-allocated_before)/1e6:.2f}MB | trainer id: {self.trainer_id}")

        return global_step, tr_loss / global_step

    def eval_model(self, epoch=0, global_step=0, device=None):
        if not device:
            device = self.device

        results = {}

        eval_loss = 0.0
        nb_eval_steps = 0
        n_batches = len(self.test_dl)
        test_sample_len = len(self.test_dl.dataset)
        preds = np.empty((test_sample_len, self.num_labels))

        out_label_ids = np.empty(test_sample_len)
        # TODO: See why .to(device) was called here
        self.model  # .to(device)
        self.model.eval()
        self.fmodel, self.params, self.buffers = fc.make_functional_with_buffers(
            self.model
        )
        logging.info(
            "len(test_dl) = %d, n_batches = %d" % (len(self.test_dl), n_batches)
        )
        for i, batch in enumerate(self.test_dl):
            with torch.no_grad():
                batch = tuple(t for t in batch)
                x = batch[1]
                labels = batch[4]

                output = self.model(x)
                logits = output[0]

                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                eval_loss += loss.item()

            nb_eval_steps += 1
            start_index = self.args.eval_batch_size * i

            end_index = (
                start_index + self.args.eval_batch_size
                if i != (n_batches - 1)
                else test_sample_len
            )
            preds[start_index:end_index] = logits.detach().cpu().numpy()
            out_label_ids[start_index:end_index] = labels.detach().cpu().numpy()

        eval_loss = eval_loss / nb_eval_steps

        model_outputs = preds
        preds = np.argmax(preds, axis=1)
        result, wrong = self.compute_metrics(
            preds, out_label_ids, self.test_dl.examples
        )
        result["eval_loss"] = eval_loss
        results.update(result)

        self.results.update(result)
        logging.info(self.results)

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