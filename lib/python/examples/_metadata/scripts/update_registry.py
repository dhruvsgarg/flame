import yaml

PATH = "lib/python/examples/_metadata/trainer_registry.yaml"
GPU_A40_DELAY_S = 0.2

def main() -> None:
    with open(PATH) as f:
        registry = yaml.safe_load(f)

    for trainer in registry["trainers"].values():
        # Rename training_delay_s -> computation_time_ms as a per-device dict
        # (training_delay_s had mobile_device and gpu_a40 values in seconds)
        existing = trainer["training_delay_s"]
        trainer["computation_time_ms"] = {
            "mobile_device": float(existing["mobile_device"]) * 1000,
            "gpu_a40": float(existing["gpu_a40"]) * 1000,
        }
        # Remove the old field
        del trainer["training_delay_s"]
        # Remove the old flat computation_time_ms (was a single number, now replaced above)
        # Note: the new computation_time_ms above overwrites it, so del not needed

    with open(PATH, "w") as f:
        yaml.dump(registry, f, default_flow_style=False, sort_keys=False)

    print(f"Done. training_delay_s renamed to computation_time_ms (in ms) for {len(registry['trainers'])} trainers.")

if __name__ == "__main__":
    main()