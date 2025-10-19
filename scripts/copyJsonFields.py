import json
import os

# ---- CONFIG ----
SRC_DIR = "/home/dgarg39/gaurav/flame/lib/python/examples/async_cifar10/trainer/config_dir100_num300_traceFail_6d_3state"
DST_DIR = "/home/dgarg39/gaurav/flame/lib/python/examples/fwdllm/expts/run_tc_expts/json_scripts"

FIELDS_TO_COPY = ["training_delay_enabled", "training_delay_s"]

def update_json_fields(src_path, dst_path):
    # Read source JSON
    with open(src_path, "r") as f:
        src_data = json.load(f)

    # Read destination JSON
    with open(dst_path, "r") as f:
        dst_data = json.load(f)

    # Copy desired fields from source → destination
    src_hyp = src_data.get("hyperparameters", {})
    dst_hyp = dst_data.setdefault("hyperparameters", {})

    for field in FIELDS_TO_COPY:
        if field in src_hyp:
            dst_hyp[field] = src_hyp[field]

    # Write back destination JSON (minimal formatting change)
    with open(dst_path, "w") as f:
        json.dump(dst_data, f, indent=4)
        # f.write("\n")  # preserve trailing newline

def main():
    for i in range(1, 101):  # 1 → 100
        src_file = f"trainer_{i}.json"
        dst_file = f"trainer_{i-1}.json"

        src_path = os.path.join(SRC_DIR, src_file)
        dst_path = os.path.join(DST_DIR, dst_file)

        if not os.path.exists(src_path):
            print(f"Missing source: {src_path}")
            continue
        if not os.path.exists(dst_path):
            print(f"Missing destination: {dst_path}")
            continue

        update_json_fields(src_path, dst_path)
        print(f"Updated: {dst_path} using {src_path}")

if __name__ == "__main__":
    main()