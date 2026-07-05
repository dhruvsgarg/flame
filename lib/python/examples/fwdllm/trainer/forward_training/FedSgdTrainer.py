import logging
from flame.mode.horizontal.syncfl.fwdllm_trainer import Trainer
import torch
import time
import io
import json
import hashlib
import os
import zlib
import numpy as np
from datetime import datetime
import ast
from flame.config import TrainerAvailState

from flame.monitor.runtime import FwdLLMStage, timer_decorator
import flame.monitor.runtime
import math

from flame import telemetry
from flame.telemetry.events import build_trainer_round

logger = logging.getLogger(__name__)


def _parse_avl_events(val):
    # YAML delivers already-parsed lists; legacy JSON delivered string-encoded lists.
    return val if isinstance(val, list) else ast.literal_eval(val)


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy types and PyTorch tensors"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        return super().default(obj)


def _serialize_value(value):
    """Serialize a value to JSON-compatible format"""
    if isinstance(value, torch.Tensor):
        # Convert PyTorch tensor to list
        return value.detach().cpu().numpy().tolist()
    elif isinstance(value, np.ndarray):
        # Convert numpy array to list
        return value.tolist()
    elif isinstance(value, np.integer):
        # Convert numpy integer to Python int
        return int(value)
    elif isinstance(value, np.floating):
        # Convert numpy float to Python float
        return float(value)
    elif isinstance(value, (list, tuple)):
        # Recursively serialize lists/tuples
        return [_serialize_value(item) for item in value]
    elif isinstance(value, dict):
        # Recursively serialize dictionaries
        return {key: _serialize_value(val) for key, val in value.items()}
    else:
        # For other types, try to convert to string
        return str(value)


def _extract_sample_data(example):
    """Extract standardized data from an example object"""
    # Try to get text fields
    text_a = getattr(example, "text_a", None)
    text_b = getattr(example, "text_b", None)
    text = getattr(example, "text", None)

    # If no specific text field, use string representation
    if text_a is None and text_b is None and text is None:
        text = str(example)
    else:
        # Combine text fields if multiple exist
        text_parts = []
        if text_a:
            # Serialize text_a if it's a tensor/array
            text_a_serialized = _serialize_value(text_a)
            text_parts.append(str(text_a_serialized))
        if text_b:
            # Serialize text_b if it's a tensor/array
            text_b_serialized = _serialize_value(text_b)
            text_parts.append(str(text_b_serialized))
        if text:
            # Serialize text if it's a tensor/array
            text_serialized = _serialize_value(text)
            text_parts.append(str(text_serialized))
        text = " [SEP] ".join(text_parts) if len(text_parts) > 1 else text_parts[0]

    # Get label and serialize it properly
    label = getattr(example, "label", None)
    if label is None:
        label = str(example)
    else:
        # Serialize the label to handle numpy/torch types
        label = _serialize_value(label)

    # Create a hash for easy comparison
    sample_str = f"{text}|{label}"
    sample_hash = hashlib.md5(sample_str.encode("utf-8")).hexdigest()

    return {"text": text, "label": label, "hash": sample_hash}


class FedSGDTrainer(Trainer):

    def __init__(
        self,
        trainer_id,
        train_data_local_dict,
        train_data_local_num_dict,
        test_data_local_dict,
        train_data_num,
        device,
        args,
        model_trainer,
        config=None,
        client_index=None,
    ):
        self.trainer = model_trainer
        self.trainer_id = trainer_id
        self.client_index = client_index  # this variable is diff from client_idx because it contains a list of clients. we dont need it. was used by fwdllm

        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict
        # NRL most of this is reduntant since our dict is of size=1. But keeping this code for consistency
        logger.debug(
            f"train_data_local_dict keys: {train_data_local_dict.keys()}, client idx: {args.client_idx}, {type(args.client_idx)}"
        )

        self.train_local = [self.train_data_local_dict[args.client_idx]]

        # this will return -1
        self.test_local = self.test_data_local_dict[args.client_idx]

        self.train_local_list = [
            [data for data in self.train_local[i]] for i in range(len(self.train_local))
        ]
        self.dataset_size = len(self.train_local_list)
        self.train_data_local_num_dict = train_data_local_num_dict
        self.all_train_data_num = train_data_num

        self.local_sample_number = None
        # logger.info(f"self.device: {self.device}, torch.cuda.device_count(): {torch.cuda.device_count()}")

        # self.train_data_local_dict.to(self.device)

        self.args = args
        self.accumulated_error = None
        self.config = config
        self.device = device

        # abstract attributes
        self.loss_fn = torch.nn.CrossEntropyLoss
        self.dataset_size = None
        self.model = model_trainer.model
        # NRL adding new variables
        self.data_id = None
        logger.info("[GJD] self.data_id is reset to None")
        self.total_data_bins = None
        self.grad_for_var_check = None
        self.jvp_for_snr_check = None
        self.data_written_to_file = False  # Flag to prevent writing data multiple times
        setattr(
            self.trainer.model_trainer, "base_trainer", self
        )  # for accessing FedSGDTrainer methods inside model_trainer - stat utility

        # Check if client will emulate delays in training time
        self.training_delay_enabled = self.config.hyperparameters.training_delay_enabled
        self.training_delay_s = float(self.config.hyperparameters.training_delay_s)
        self.training_delay_factor = float(
            self.config.hyperparameters.training_delay_factor
        )
        self.speedup_factor = 1.0

        # --- Simulated-clock support (Batch 1, config-gated) ------------------
        # time_mode is threaded into hyperparameters from the launcher's
        # --time_mode CLI arg (see trainer/main.py). "simulated": _emulate_
        # training_delay() computes the modeled delay but does NOT sleep, and
        # train_with_data_id() stamps a modeled completion timestamp
        # (_sim_completion_ts) the aggregator orders updates by. "real"
        # (default): unchanged wall-clock behavior (flag-off => byte-identical).
        self.time_mode = getattr(self.config.hyperparameters, "time_mode", "real")
        self.simulated = self.time_mode == "simulated"
        _leg = getattr(self.config.hyperparameters, "sim_completion_leg_s", 0.0)
        self.sim_completion_leg_s = float(_leg) if _leg is not None else 0.0
        # SIM_SEND_TS is stamped by the aggregator on each dispatch and read in
        # the base trainer's _fetch_weights; the rest are stamped after training
        # for _send_grads / telemetry / the intrinsic-duration selector signal.
        self._sim_send_ts = None
        self._sim_completion_ts = None
        self._sim_round_duration_s = None
        # Pure modeled delay D for this round (K-D31/P2-7a): stamped in the grad
        # message (MODELED_DELAY_S) so the aggregator can canonically order the
        # cohort's commits by (D, trainer_id) identically in real and sim. None
        # until the first round completes / when delays are disabled.
        self._modeled_delay_s = None
        self._wall_recv_ts = None

        self.trainer_start_ts = time.time()
        # TODO (ARM): Fix this to read traces better!
        # Storing synthetic avail traces
        self.avl_events_syn_0 = _parse_avl_events(
            self.config.hyperparameters.avl_events_syn_0
        )

        self.avl_events_syn_20 = _parse_avl_events(
            self.config.hyperparameters.avl_events_syn_20
        )

        self.avl_events_syn_50 = _parse_avl_events(
            self.config.hyperparameters.avl_events_syn_50
        )

        self.avl_events_syn_train_100_eval_0_unavail_0 = _parse_avl_events(
            self.config.hyperparameters.avl_events_syn_train_100_eval_0_unavail_0
        )

        self.avl_events_syn_train_90_eval_10_unavail_0 = _parse_avl_events(
            self.config.hyperparameters.avl_events_syn_train_90_eval_10_unavail_0
        )

        self.avl_events_syn_train_50_eval_30_unavail_20 = _parse_avl_events(
            self.config.hyperparameters.avl_events_syn_train_50_eval_30_unavail_20
        )

        self.avl_events_mobiperf_2st = _parse_avl_events(
            self.config.hyperparameters.avl_events_mobiperf_2st
        )
        self.avl_events_mobiperf_3st_75 = _parse_avl_events(
            self.config.hyperparameters.avl_events_mobiperf_3st_75
        )
        self.avl_events_mobiperf_3st_50 = _parse_avl_events(
            self.config.hyperparameters.avl_events_mobiperf_3st_50
        )

        self.client_notify = self.config.hyperparameters.client_notify

        self.state_avl_event_ts = []
        if self.client_notify["trace"] == "syn_0":
            self.state_avl_event_ts = self.avl_events_syn_0
            logger.info(f"Set avl_events_syn_0 for trainer id {self.trainer_id}.")
        elif self.client_notify["trace"] == "syn_20":
            self.state_avl_event_ts = self.avl_events_syn_20
            logger.info(f"Set avl_events_syn_20 for trainer id {self.trainer_id}.")
        elif self.client_notify["trace"] == "syn_50":
            self.state_avl_event_ts = self.avl_events_syn_50
            logger.info(f"Set avl_events_syn_50 for trainer id {self.trainer_id}.")
        elif self.client_notify["trace"] == "avl_events_syn_train_100_eval_0_unavail_0":
            self.state_avl_event_ts = self.avl_events_syn_train_100_eval_0_unavail_0
            logger.info(
                f"Set avl_events_syn_train_100_eval_0_unavail_0 for trainer id {self.trainer_id}."
            )
        elif self.client_notify["trace"] == "avl_events_syn_train_90_eval_10_unavail_0":
            self.state_avl_event_ts = self.avl_events_syn_train_90_eval_10_unavail_0
            logger.info(
                f"Set avl_events_syn_train_90_eval_10_unavail_0 for trainer id {self.trainer_id}."
            )
        elif (
            self.client_notify["trace"] == "avl_events_syn_train_50_eval_30_unavail_20"
        ):
            self.state_avl_event_ts = self.avl_events_syn_train_50_eval_30_unavail_20
            logger.info(
                f"Set avl_events_syn_train_50_eval_30_unavail_20 for trainer id {self.trainer_id}."
            )
        # Accept both the long form ("avl_events_mobiperf_2st", this trainer's
        # own historical convention) and the short form ("mobiperf_2st", the
        # flame.launch spawner's availability_mode/client_notify.trace
        # convention mirrored from async_cifar10) -- the launcher's
        # _metadata/ injects the short form.
        elif self.client_notify["trace"] in (
            "avl_events_mobiperf_2st",
            "mobiperf_2st",
        ):
            self.state_avl_event_ts = self.avl_events_mobiperf_2st
            logger.info(f"Set avl_events_mobiperf_2st for trainer id {self.trainer_id}.")
        elif self.client_notify["trace"] in (
            "avl_events_mobiperf_3st_75",
            "mobiperf_3st_75",
        ):
            self.state_avl_event_ts = self.avl_events_mobiperf_3st_75
            logger.info(f"Set avl_events_mobiperf_3st_75 for trainer id {self.trainer_id}.")
        elif self.client_notify["trace"] in (
            "avl_events_mobiperf_3st_50",
            "mobiperf_3st_50",
        ):
            self.state_avl_event_ts = self.avl_events_mobiperf_3st_50
            logger.info(f"Set avl_events_mobiperf_3st_50 for trainer id {self.trainer_id}.")
        else:
            logger.info(
                f"No avl_events set for trainer id {self.trainer_id} since state not specified."
            )

        self.avl_state = TrainerAvailState.AVL_TRAIN
        logger.info(f"Set the available_state for {self.trainer_id} to AVL_TRAIN.")

        # flag to decide whether the trainer upon unavailability will wait or exit
        self.wait_until_next_avl = self.config.hyperparameters.wait_until_next_avl

        logger.info(f"Set the wait_until_next_avl to be {self.wait_until_next_avl}")

    def _write_client_data_to_file(self, client_id, train_data, round_idx=None):
        """Write all training data for a client to a JSON file"""
        try:
            # Create output directory if it doesn't exist
            output_dir = os.path.join(self.args.output_dir, "client_data_files")
            os.makedirs(output_dir, exist_ok=True)

            # Include round information in filename if provided
            if round_idx is not None:
                filename = os.path.join(
                    output_dir,
                    f"flame_client_{client_id}_round_{round_idx}_training_data.json",
                )
            else:
                filename = os.path.join(
                    output_dir, f"flame_client_{client_id}_training_data.json"
                )

            # Prepare data structure
            client_data = {
                "metadata": {
                    "client_id": int(client_id),  # Ensure it's a Python int
                    "round_idx": int(round_idx) if round_idx is not None else None,
                    "timestamp": datetime.now().isoformat(),
                    "total_samples": int(
                        len(train_data.examples)
                    ),  # Ensure it's a Python int
                    "file_format_version": "1.0",
                },
                "samples": [],
            }

            # Extract all samples
            for i, example in enumerate(train_data.examples):
                try:
                    sample_data = _extract_sample_data(example)
                    sample_data["sample_index"] = int(i)  # Ensure it's a Python int

                    # Verify all values are JSON serializable
                    for key, value in sample_data.items():
                        if isinstance(
                            value, (np.integer, np.floating, np.ndarray, torch.Tensor)
                        ):
                            sample_data[key] = _serialize_value(value)

                    client_data["samples"].append(sample_data)
                except Exception as e:
                    logger.error(
                        f"Failed to extract sample {i} for client {client_id}: {e}"
                    )
                    # Log the problematic example for debugging
                    logger.error(f"Problematic example type: {type(example)}")
                    logger.error(f"Example attributes: {dir(example)}")
                    if hasattr(example, "label"):
                        logger.error(f"Label type: {type(example.label)}")
                        logger.error(f"Label value: {example.label}")
                    raise  # Re-raise to see the full error

            # Write to file using custom encoder
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(
                    client_data, f, indent=2, ensure_ascii=False, cls=NumpyEncoder
                )

            logger.info(
                f"Successfully wrote {len(client_data['samples'])} samples to {filename}"
            )

        except Exception as e:
            logger.error(f"Failed to write data for client {client_id}: {e}")
            # Add more detailed error information
            import traceback

            logger.error(f"Full traceback: {traceback.format_exc()}")

    def initialize(self) -> None:
        """Initialize role."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)
        logger.debug(f"self.device: {self.device}")
        self.total_data_bins = len(self.train_local[0])

        # Write training data to files during initialization (data doesn't change across rounds)
        # Use args.client_idx since that's what's used to set up the training data
        if (
            not self.data_written_to_file
            and hasattr(self.args, "client_idx")
            and self.train_local is not None
        ):
            logger.info(
                f"Writing training data to files during initialization for client {self.args.client_idx}"
            )
            client_id = self.args.client_idx
            if len(self.train_local) > 0:
                # Use None for round_idx since this is initialization, not a specific round
                self._write_client_data_to_file(
                    client_id, self.train_local[0], round_idx=None
                )
                self.data_written_to_file = True  # Mark as written
                logger.info(
                    f"Successfully wrote training data for client {client_id} during initialization"
                )
            else:
                logger.warning(
                    f"No training data available for client {client_id} during initialization"
                )
        elif self.data_written_to_file:
            logger.info(
                "Training data already written to files, skipping initialization write"
            )
        else:
            logger.warning(
                "Cannot write training data during initialization: missing client_idx or train_local"
            )

        # loading data to gpu
        # NRL TODO: This didnt work. Error: expected all tensors to be on the same device. Needed to load them on gpu again during train_model
        for each_train_local in self.train_local[0]:
            train_data = tuple(t for t in each_train_local)
            # logger.info(f"train data: {len(train_data)}")
            train_data[1].to(self.device)
            train_data[4].to(self.device)
        logger.info(
            f"Task_id: {self.trainer_id} initialize completed at timestamp: "
            f"{time.time()}"
        )
        self.init_oort_variables()  # initialize oort variables for stat_utility calculation (fwdllm)

    def update_model(self, weights):
        # logger.info(f"NRL: Updated model weights: {weights}")
        self.trainer.set_model_params(weights)

    def train(self, round_idx=None):
        logger.info("entered train where weights = params and not grad")
        self.args.round_idx = round_idx

        self.trainer.train(self.train_local, self.device, self.args)

        weights = self.trainer.get_model_params()

        return weights, self.local_sample_number

    @timer_decorator
    def _check_availability(self):
        if self.avl_state != TrainerAvailState.AVL_TRAIN:
            if self.wait_until_next_avl:
                logger.info(
                    f"Trainer id {self.trainer_id} is not available to train. Waiting for it to be available"
                )
                while self.avl_state != TrainerAvailState.AVL_TRAIN:
                    # Sim availability is enforced AGGREGATOR-side (ClientAvailability
                    # send-gate / vclock-jump to the next avail event), never by a
                    # trainer wall sleep: sim time cannot advance while a trainer is
                    # blocked on time.sleep, so this spin would freeze the virtual
                    # clock (root #13). The sim therefore never spins here -- it
                    # proceeds and lets the agg-side gate withhold. Mirrors
                    # async_cifar10's gated trainer avail wait; real byte-identical.
                    if self.simulated:
                        break
                    time.sleep(1)
                logger.info(
                    f"Trainer id {self.trainer_id} is back to available to train."
                )
            else:
                logger.info(
                    f"Trainer id {self.trainer_id} is not available to train. Exiting training."
                )
                return False
        return True

    @timer_decorator
    def _perform_training(self):
        logger.info(
            f"starting training for trainer id: {self.trainer_id}, data_id = {self.data_id}"
        )
        logger.info(
            f"train_local_list[0][0]: {len(self.train_local_list[0][0])}, {len(self.train_local_list)}"
        )

        self.reset_stat_utility()  # reset stat_utility for this databin (fwdllm)

        # List Index to be used in case of both sync and async version.
        # In sync model version = round hence, Index = model version
        # In async: Index = model version % round
        # list_index = self._model_version % self._round if self._model_version  > self._round else self._model_version
        list_index = self.data_id  # Which data bin to use for training
        logging.info(
            f"self._model_version: {self._model_version } - list-index/data-id = {list_index}"
        )
        self.trainer.train(
            [self.train_local_list[0][list_index]],
            self.device,
            self.args,
            {
                "round_id": self._round,
                "data_id": self.data_id,
                "iteration": self.iteration_per_data_id,
            },
        )
        self.grad_for_var_check = self.trainer.model_trainer.grad_for_var_check
        self.jvp_for_snr_check = self.trainer.model_trainer.jvp_for_snr_check

    @timer_decorator
    def _emulate_training_delay(self, gpu_time_s: float = 0.0):
        """REMAINDER-WAIT delay model (aligned with async_cifar10; supersedes the
        old flat-additive model, K-D2 → K-D29). The modeled mobile device takes
        `_delay_s = training_delay_s / factor / speedup`; on our GPU the forward
        pass takes `gpu_time_s`, which SHOULD be << the device time (a faithful
        emulation of a slow mobile client). So:

          - REAL mode sleeps ONLY the remainder max(0, _delay_s - gpu_time_s) →
            real wall this round ≈ _delay_s (the mobile device wall), with GPU
            compute hidden inside it.
          - SIM mode skips the sleep (charged to the vclock); the sct round
            duration is max(gpu, _delay_s) (see train_with_data_id), NOT
            gpu + _delay_s. The per-trainer registry delays give the completion
            SPREAD, so update ORDER = delay order = deterministic and identical
            real↔sim (this is what makes cohort_sequence parity attainable).

        OVERRUN: if gpu_time_s > _delay_s the GPU is slower than the modeled
        device (contention / too many trainers-per-GPU / delay_factor too big) —
        the emulation is no longer faithful and update ORDER can flip, so we log
        [TIMING_OVERRUN] and flag it in telemetry. Returns
        (modeled_delay_s, remaining_s, overran).
        """
        # config schema types training_delay_enabled as bool (default False)
        # but historical launcher yamls pass the string "True"; accept both so
        # the modeled delay is not silently dropped to 0.
        _enabled = self.training_delay_enabled in (True, "True", "true")
        if not _enabled:
            return 0.0, 0.0, False
        _delay_s = (self.training_delay_s / self.training_delay_factor) / self.speedup_factor
        _remaining_s = max(0.0, _delay_s - gpu_time_s)
        _overran = gpu_time_s > _delay_s
        if _overran:
            logger.warning(
                f"[TIMING_OVERRUN] trainer {self.trainer_id} data_id={self.data_id} "
                f"iter={self.iteration_per_data_id}: gpu={gpu_time_s:.2f}s > "
                f"budget={_delay_s:.2f}s (excess={gpu_time_s - _delay_s:.2f}s) — "
                f"emulation unfaithful, update order may flip. Reduce trainers/GPU "
                f"or raise training_delay_factor."
            )
        if self.simulated:
            logger.info(
                f"time_mode=simulated: modeled delay for trainer {self.trainer_id} "
                f"= {_delay_s:.3f}s (not slept; charged to vclock; gpu={gpu_time_s:.3f}s)."
            )
        elif _remaining_s > 0:
            time.sleep(_remaining_s)
            logger.info(
                f"Trainer {self.trainer_id} slept remainder {_remaining_s:.3f}s "
                f"(budget {_delay_s:.3f}s - gpu {gpu_time_s:.3f}s)."
            )
        return _delay_s, _remaining_s, _overran

    def _sim_straggler_offset_s(self) -> float:
        """B2 (K-D20 #6): the modeled delay D is flat across trainers, so the
        sync barrier's k-th-smallest sct under-spreads vs real's trainer_speed_s
        dispersion (~2.3 s/round). In SIM only (real gets its spread from GPU
        contention) add a STABLE per-trainer offset in [0, simStragglerSpreadS)
        so the cohort completion spread matches real. Deterministic in
        trainer_id (crc32, not salted like hash()) ⇒ reproducible; spread 0 ⇒
        byte-identical."""
        if not self.simulated:
            return 0.0
        _hp = getattr(getattr(self, "config", None), "hyperparameters", None)
        spread = float(getattr(_hp, "sim_straggler_spread_s", 0.0) or 0.0)
        if spread <= 0.0:
            return 0.0
        frac = (zlib.crc32(str(self.trainer_id).encode()) % 1000) / 1000.0
        return spread * frac

    @timer_decorator
    def train_with_data_id(self):
        if self.abort_training == True:
            logger.info(
                f"Aborting training for trainer id: {self.trainer_id} because it has already sent updates for iteration_per_data_id: {self.iteration_per_data_id}"
            )
            return

        # Create FwdLLMStage for timing/metrics logging. Set AFTER the abort
        # check so an aborted round stays a true no-op (no fwd_llm_stage → the
        # @timer_decorator emits no step_timing record for it, and nothing else
        # fires either — see test_aborted_round_emits_nothing).
        self.fwd_llm_stage = FwdLLMStage(
            self._round, self.data_id, self.iteration_per_data_id, self.trainer_id
        )

        # Phase-timing entry (Stage A1): everything up to the compute loop is
        # pre_train (avail check, FwdLLMStage setup, loader state).
        _phase_entry = time.time()
        if not self._check_availability():
            return

        _round_start_ts = time.time()
        _pre_train_s = _round_start_ts - _phase_entry
        self._perform_training()
        _real_gpu_time_s = time.time() - _round_start_ts

        # emulate the mobile-device delay via the REMAINDER-WAIT model: real
        # sleeps max(0, delay - gpu); sim skips it. Returns the modeled budget,
        # the remainder actually waited, and whether the GPU overran the budget.
        _delay_s, _remaining_s, _overran = self._emulate_training_delay(_real_gpu_time_s)
        # Stash the pure modeled delay D (deterministic from the registry) so
        # _send_grads can stamp it (MODELED_DELAY_S) for the aggregator's
        # canonical (D, trainer_id) commit ordering. 0.0 when delays are off ->
        # stays None-equivalent (all-zero => no ordering signal, arrival order).
        self._modeled_delay_s = _delay_s if _delay_s else None

        # post_train phase starts AFTER the modeled delay (Root B / #6): real
        # SLEEPS _delay_s above (the modeled-latency term, compared via
        # training_budget_s), so stamping here EXCLUDES it -> post_train_s is pure
        # post-processing, mode-comparable (~0 both modes; sim never slept).
        # Without this, phase_post_train saw real ~1.14 vs sim 0.0.
        _phase_post_start = time.time()

        # Sim-mode stamps: the modeled round duration is the REMAINDER-WAIT
        # max(real_gpu, D) (K-D29), matching real mode's sleep-the-remainder
        # semantics (device wall = D, GPU hidden inside) -- NOT the old additive
        # gpu + D (K-D2, reversed). The sct (when this update COMMITS on the
        # virtual clock) is the aggregator's dispatch stamp (SIM_SEND_TS, read in
        # _fetch_weights) + that duration + the optional pre-commit holding leg.
        # _send_grads sends these so the aggregator can order updates by
        # _sim_completion_ts. In real mode these stay None and the aggregator
        # falls back to arrival order (unchanged).
        # B3 (K-D20 #6): WAN payload-transfer term (up+down). NOT measurable on
        # localhost (no ground truth), so this is a DOCUMENTED knob left at 0 --
        # do NOT enable without a real WAN measurement. Byte-identical at 0.
        _hp = getattr(getattr(self, "config", None), "hyperparameters", None)
        _wan_s = (
            float(getattr(_hp, "sim_wan_transfer_s", 0.0) or 0.0)
            if (self.simulated and _hp is not None) else 0.0
        )
        # REMAINDER-WAIT sct (K-D29): the modeled round wall is max(gpu, delay),
        # NOT gpu + delay — the mobile device's compute is HIDDEN inside its
        # delay budget (real slept only the remainder above; sim charges the
        # same max() to the vclock). The per-trainer registry delays supply the
        # completion spread, so the legacy crc32 straggler offset is redundant
        # (kept flag-gated at 0 in the sim yamls; supersedes #6/Root B B2).
        # _wan_s stays a documented knob at 0 (no localhost ground truth).
        self._sim_round_duration_s = (
            max(_real_gpu_time_s, _delay_s) + self._sim_straggler_offset_s() + _wan_s
        )
        if self.simulated:
            _leg = self.sim_completion_leg_s
            _base = self._sim_send_ts if self._sim_send_ts is not None else time.time()
            self._sim_completion_ts = _base + self._sim_round_duration_s + _leg

        logger.info(
            f"completed training for trainer id: {self.trainer_id}, data_id = {self.data_id}"
        )

        # self._stat_utility (inherited from the base Trainer class) is
        # accumulated from per-batch loss during _perform_training() --
        # the same value the aggregator's variance/utility checks use, so
        # this is the natural point to report it via telemetry.
        if telemetry.is_enabled():
            try:
                _stat_utility = float(self._stat_utility)
            except (TypeError, ValueError):
                _stat_utility = None
            # Post-compute overhead (telemetry build); the last trainer-side phase
            # term for the #6 wall decomposition (Stage A1). The modeled delay is
            # EXCLUDED via the _phase_post_start stamp position (after the delay) --
            # see Root B note above.
            _post_train_s = time.time() - _phase_post_start
            ev, fields = build_trainer_round(
                round_num=int(self._round),
                real_gpu_time_s=_real_gpu_time_s,
                # the total modeled wall this round took: max(real_gpu, D) under
                # the K-D29 remainder-wait model (device wall = D, GPU hidden),
                # + the sim sct-model folds (B2 straggler spread, B3 WAN, both 0
                # by default). training_budget_s below carries D separately so
                # the overrun (gpu > D) is recoverable from telemetry.
                sim_round_duration_s=self._sim_round_duration_s,
                avail_state=self.avl_state.value,
                dataset_size=self.dataset_size,
                stat_utility=_stat_utility,
                extra={
                    "data_id": self.data_id,
                    "iteration_per_data_id": self.iteration_per_data_id,
                    "model_version": self._model_version,
                    # Per-phase wall breakdown (Stage A1): pre/gpu/post are stamped
                    # here; mqtt_fetch_s + weights_to_{ram,gpu}_s ride in via
                    # _phase_times (populated in fwdllm_trainer._fetch_weights).
                    "pre_train_s": _pre_train_s,
                    "gpu_compute_s": _real_gpu_time_s,
                    "post_train_s": _post_train_s,
                    "training_budget_s": _delay_s,
                    # Remainder-wait model (K-D29): what real actually slept +
                    # whether the GPU overran the modeled device budget (P2-6:
                    # an overrun can flip update order → cohort_sequence break).
                    "remaining_time_s": _remaining_s,
                    "training_overran": _overran,
                    "trainer_phase": (
                        f"{self._round}/{self.data_id}/{self.iteration_per_data_id}"
                    ),
                    **getattr(self, "_phase_times", {}),
                },
            )
            telemetry.emit(ev, **fields)

    def test(self):
        # train data
        train_metrics = self.trainer.test(self.train_local, self.device, self.args)
        train_tot_correct, train_num_sample, train_loss = (
            train_metrics["test_correct"],
            train_metrics["test_total"],
            train_metrics["test_loss"],
        )

        # test data
        test_metrics = self.trainer.test(self.test_local, self.device, self.args)
        test_tot_correct, test_num_sample, test_loss = (
            test_metrics["test_correct"],
            test_metrics["test_total"],
            test_metrics["test_loss"],
        )

        return (
            train_tot_correct,
            train_loss,
            train_num_sample,
            test_tot_correct,
            test_loss,
            test_num_sample,
        )

    def load_data(self) -> None:
        pass

    def train(self) -> None:
        pass

    def evaluate(self) -> None:
        pass

    def check_and_sleep(self) -> None:
        pass

    def check_and_update_state_avl(self):
        if hasattr(self, "cm") and self.cm is not None:
            if len(self.state_avl_event_ts) > 0:
                next_event_ts = self.trainer_start_ts + (self.state_avl_event_ts[0][0])
                if time.time() >= next_event_ts:
                    state_to_set = self.state_avl_event_ts.pop(0)[1]
                    old_status = self.avl_state.value
                    try:
                        self.avl_state = TrainerAvailState(state_to_set)
                    except ValueError:
                        logger.error(
                            f"Invalid status encountered: {state_to_set}. Retaining old status {old_status}."
                        )
                        return
                    new_status = self.avl_state.value
                    logger.info(
                        f"Changed the availability status of trainer {self.trainer_id} from {old_status} to {new_status}"
                    )
                    if self.client_notify["enabled"] == "True":
                        logger.info("Trainer trying to notify aggregator")
                        self._perform_channel_state_update(
                            tag="upload",
                            state=self.avl_state,
                            timestamp=str(time.time()),
                        )
            else:
                logger.debug(
                    f"No availability events pending for trainer {self.trainer_id}"
                )
        else:
            logger.info(
                f"Channel manager not set yet for trainer {self.trainer_id}. "
                f"Skipping avail status update. "
                f"Sleep for 20s before checking again."
            )
            time.sleep(20)

    def notify_trainer_avail(self) -> None:
        logger.info("notify_trainer_avail thread running")
        while True:
            time.sleep(1)  # Will check every 1 second
            self.check_and_update_state_avl()
