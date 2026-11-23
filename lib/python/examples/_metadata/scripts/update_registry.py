"""Add per-device computation_time_ms (ms) to each trainer in trainer_registry.yaml.

mobile_device is derived from training_delay_s, which is kept for examples
that still read it. Idempotent: re-running overwrites computation_time_ms.
"""
import yaml

PATH = "lib/python/examples/_metadata/trainer_registry.yaml"
GPU_A40_MS = 200.0


def main() -> None:
    with open(PATH) as f:
        registry = yaml.safe_load(f)

    for trainer in registry["trainers"].values():
        trainer["computation_time_ms"] = {
            "mobile_device": float(trainer["training_delay_s"]) * 1000,
            "gpu_a40": GPU_A40_MS,
        }

    with open(PATH, "w") as f:
        yaml.dump(registry, f, default_flow_style=False, sort_keys=False)

    print(f"Updated computation_time_ms for {len(registry['trainers'])} trainers.")


if __name__ == "__main__":
    main()
