import yaml

path = "lib/python/examples/_metadata/trainer_registry.yaml"

with open(path) as f:
    registry = yaml.safe_load(f)

for name, trainer in registry["trainers"].items():
    rtt_base = float(trainer["rtt_communication_time_ms"])
    trainer["rtt_base_ms"] = str(round(rtt_base, 1))
    trainer["rtt_amplitude"] = "0.3"   # RTT varies ±30% of base
    trainer["rtt_period_s"] = "120.0"  # Full cycle every 2 minutes

with open(path, "w") as f:
    yaml.dump(registry, f, default_flow_style=False, sort_keys=False)

print("Done.")
