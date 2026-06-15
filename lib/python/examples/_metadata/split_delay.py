import yaml

path = "lib/python/examples/_metadata/trainer_registry.yaml"

with open(path) as f:
    registry = yaml.safe_load(f)

for name, trainer in registry["trainers"].items():
    delay_s = float(trainer["training_delay_s"])
    delay_ms = delay_s * 1000
    trainer["computation_time_ms"] = str(round(delay_ms * 0.7, 1))
    trainer["rtt_communication_time_ms"] = str(round(delay_ms * 0.3, 1))

with open(path, "w") as f:
    yaml.dump(registry, f, default_flow_style=False, sort_keys=False)

print("Done.")

