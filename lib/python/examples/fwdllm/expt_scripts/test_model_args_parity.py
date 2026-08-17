"""Every arg the trainer's __init__ reads unguarded must come out of the builder.

The aggregator builds ForwardTextClassificationTrainer for eval, so an arg the
builder does not set is an AttributeError at aggregator startup, ~30 s in, after
100 trainers are up -- how the 08-08 node 2/4 sweeps lost 12 arms. Since both
mains now call `build_model_args`, checking the builder checks both. Also asserts
neither main hand-rolls its own ClassificationArgs dict again.

Static: no model, no GPU, no data.
"""
import ast
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAINER = f"{ROOT}/trainer/forward_training/tc_transformer_trainer_distribute.py"
BUILDER = f"{ROOT}/trainer/model_args_builder.py"
REGISTRY = f"{ROOT}/expts/dataset_registry.py"
MAINS = [f"{ROOT}/trainer/main.py", f"{ROOT}/aggregator/main_fedfwd_agg.py"]

# Set on the args object outside the dicts, by the builder itself.
PRESET = {"model_name", "model_type", "num_labels", "client_idx", "config"}

# Knobs BOTH roles enact (§0 rule 2), so they must be written to both override
# blocks and must agree. `run_sequential.sh` fans `hyperparameter_overrides()` to
# both blocks verbatim, so registry membership is what makes them agree.
DUAL_READ_FROM_REGISTRY = {"dataset", "data_file_path", "partition_file_path",
                           "max_seq_length", "cache_dir"}


def unguarded_args_reads(path):
    """`self.args.X` / `args.X` in __init__, excluding getattr(args, "X", ...)."""
    tree = ast.parse(open(path).read())
    init = next(
        n for cls in ast.walk(tree) if isinstance(cls, ast.ClassDef)
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "__init__"
    )
    guarded = {
        c.args[1].value for c in ast.walk(init)
        if isinstance(c, ast.Call) and getattr(c.func, "id", "") == "getattr"
        and len(c.args) > 1 and isinstance(c.args[1], ast.Constant)
    }
    names = set()
    for node in ast.walk(init):
        if not isinstance(node, ast.Attribute):
            continue
        base = node.value
        if isinstance(base, ast.Name) and base.id == "args":
            names.add(node.attr)
        elif (isinstance(base, ast.Attribute) and base.attr == "args"
              and isinstance(base.value, ast.Name) and base.value.id == "self"):
            names.add(node.attr)
    return names - guarded


def builder_supplies():
    """Keys of _REQUIRED / _OPTIONAL / _FIXED in the builder module."""
    tree = ast.parse(open(BUILDER).read())
    out = set()
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        name = getattr(node.targets[0], "id", "")
        if name not in ("_REQUIRED", "_OPTIONAL", "_FIXED"):
            continue
        val = node.value
        if isinstance(val, ast.Tuple):
            out |= {e.value for e in val.elts if isinstance(e, ast.Constant)}
        elif isinstance(val, ast.Dict):
            out |= {k.value for k in val.keys if isinstance(k, ast.Constant)}
    return out


def registry_override_keys():
    """Keys of the dict `hyperparameter_overrides` returns."""
    tree = ast.parse(open(REGISTRY).read())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
              and n.name == "hyperparameter_overrides")
    out = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Dict):
            out |= {k.value for k in node.keys if isinstance(k, ast.Constant)}
    return out


needed = unguarded_args_reads(TRAINER) - PRESET
supplied = builder_supplies()
missing = sorted(needed - supplied)
print(f"  trainer __init__ reads unguarded : {len(needed)}   builder supplies: {len(supplied)}")
print(f"  missing from build_model_args    : {missing or 'none'}")

unfanned = sorted(DUAL_READ_FROM_REGISTRY - registry_override_keys())
unbuilt = sorted(DUAL_READ_FROM_REGISTRY - supplied)
print(f"  dual-read, not in hyperparameter_overrides : {unfanned or 'none'}")
print(f"  dual-read, not supplied by the builder     : {unbuilt or 'none'}")

fail = bool(missing or unfanned or unbuilt)
for main in MAINS:
    src = open(main).read()
    if "ClassificationArgs()" in src:
        print(f"  {os.path.relpath(main, ROOT)}: hand-rolls ClassificationArgs again -- use the builder")
        fail = True
sys.exit(1 if fail else 0)
