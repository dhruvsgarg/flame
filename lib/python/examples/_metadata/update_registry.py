import yaml

PATH = "lib/python/examples/_metadata/trainer_registry.yaml"
GPU_A40_DELAY_S = 0.2

def main() -> None:
    with open(PATH) as f:
        registry = yaml.safe_load(f)

    for trainer in registry["trainers"].values():
        # Convert training_delay_s to dict, preserving the original value as mobile_device
        existing_delay = float(trainer["training_delay_s"])
        trainer["training_delay_s"] = {
            "mobile_device": existing_delay,
            "gpu_a40": GPU_A40_DELAY_S,
        }
        # computation_time_ms stays as-is — still used in main.py sleep/eval logic

    with open(PATH, "w") as f:
        yaml.dump(registry, f, default_flow_style=False, sort_keys=False)

    print(f"Done. training_delay_s converted to dict for {len(registry['trainers'])} trainers.")

if __name__ == "__main__":
    main()