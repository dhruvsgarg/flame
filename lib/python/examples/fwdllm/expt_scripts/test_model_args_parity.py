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
MAINS = [f"{ROOT}/trainer/main.py", f"{ROOT}/aggregator/main_fedfwd_agg.py"]

# Set on the args object outside the dicts, by the builder itself.
PRESET = {"model_name", "model_type", "num_labels", "client_idx", "config"}


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


needed = unguarded_args_reads(TRAINER) - PRESET
supplied = builder_supplies()
missing = sorted(needed - supplied)
print(f"  trainer __init__ reads unguarded : {len(needed)}   builder supplies: {len(supplied)}")
print(f"  missing from build_model_args    : {missing or 'none'}")

fail = bool(missing)
for main in MAINS:
    src = open(main).read()
    if "ClassificationArgs()" in src:
        print(f"  {os.path.relpath(main, ROOT)}: hand-rolls ClassificationArgs again -- use the builder")
        fail = True
sys.exit(1 if fail else 0)
