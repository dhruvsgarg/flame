"""One place that turns a run config into the `ClassificationArgs` both sides use.

The aggregator builds `ForwardTextClassificationTrainer` for server-side eval, so
it needs the *same* args object the trainer does. Keeping two hand-written dicts
in `trainer/main.py` and `aggregator/main_fedfwd_agg.py` let them drift, and a
knob missing from the aggregator copy is an AttributeError at startup, after 100
trainers are already up (2026-08-08: 12 arms lost, handoff §22.1).

`expt_scripts/test_model_args_parity.py` is the static guard on top of this.
"""
from examples.fwdllm.trainer.model.transformer.model_args import ClassificationArgs

# Read straight off the config; absent => the run is misconfigured, so fail loudly.
_REQUIRED = (
    "fl_algorithm", "freeze_layers", "epochs", "learning_rate",
    "gradient_accumulation_steps", "do_lower_case", "manual_seed",
    "max_seq_length", "train_batch_size", "eval_batch_size",
    "evaluate_during_training_steps", "fp16", "data_file_path",
    "partition_file_path", "partition_method", "dataset", "output_dir",
    "is_debug_mode", "fedprox_mu", "use_adapter", "comm_round", "peft_method",
    "var_control", "perturbation_sampling", "client_idx",
)

# Newer knobs: absent in older configs, so each carries the default that keeps
# behaviour byte-identical to before it existed.
_OPTIONAL = {
    "trainable_scope": "adapters_head",   # S-I (inert on distilbert -- see §11.5)
    "adapter_reduction_factor": 16,       # the real p knob: 768/rf per adapter
    "perturbation_count": 10,             # P
    "probe_combine": "select",            # S-H
    "jvp_perf_opt": False,                # §L perf, bit-identical
    "jvp_eval_mode": True,                # H13 dropout fix
    # Enacted trainer-side only; the aggregator's trainer object never probes, so
    # the default is inert there and the key stays out of aggregator_base.json.
    # `require_trainer_knobs` keeps it from silently defaulting on a real trainer.
    "select_perturbation_using_jvp": False,
}

# Knobs whose default would silently change what a TRAINER computes.
_TRAINER_REQUIRED = ("select_perturbation_using_jvp",)


def require_trainer_knobs(hp):
    """Trainer-side preflight: fail loudly rather than default a probe knob."""
    missing = [k for k in _TRAINER_REQUIRED if not hasattr(hp, k)]
    if missing:
        raise AttributeError(
            f"trainer config is missing probe knob(s) {missing}; refusing to run "
            "with a default that would change the estimator"
        )

# Fixed by how this example runs, not by the config.
_FIXED = {
    "reprocess_input_data": False,        # ignore cached features
    "overwrite_output_dir": True,
    "evaluate_during_training": False,    # disabled for FedAvg
}


def build_model_args(hp, num_labels):
    """`hp` is config.hyperparameters. Identical output on trainer and aggregator."""
    model_args = ClassificationArgs()
    model_args.model_name = hp.model_name
    model_args.model_type = hp.model_type
    model_args.load(model_args.model_name)
    model_args.num_labels = num_labels
    model_args.client_idx = hp.client_idx
    values = {k: getattr(hp, k) for k in _REQUIRED}
    values.update({k: getattr(hp, k, d) for k, d in _OPTIONAL.items()})
    values.update(_FIXED)
    model_args.update_from_dict(values)
    model_args.config["num_labels"] = num_labels
    return model_args
