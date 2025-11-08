import re
import csv
from collections import defaultdict
from pathlib import Path

# === Paths ===
LOG_FILE = Path("/home/dgarg39/shreya/flame/lib/python/examples/fwdllm/expts/run_tc_expts/log/new/test_trainer_fedFwd_distilbert_agnews_lr0.01_client_num_5_numerical_08_10_09_07.log")
OUTPUT_CSV = LOG_FILE.with_suffix(".parsed_v3.csv")

# === Regex patterns ===
fetch_pattern = re.compile(
    r"FETCH WEIGHTS complete for trainer_id (\w+), round: (\d+), data id: (\d+)"
)
stat_pattern = re.compile(
    r"stat_utility for trainerId: (\w+) is ([\d\.]+), loss: ([\d\.]+)"
)
normalized_stat_pattern = re.compile(
    r"stat_utility - normalized for trainerId: (\w+) = ([\d\.]+)"
)
databin_pattern = re.compile(
    r"sending total databin=(\d+)\s+for trainerId: (\w+)"
)
batch_pattern = re.compile(
    r"batch size: (\d+)"
)
timestamp_pattern = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+)")

# === State tracking ===
trainer_context = defaultdict(lambda: {"round": None, "data_id": None})
iteration_tracker = defaultdict(lambda: defaultdict(int))
trainer_meta = defaultdict(lambda: {"databin": None, "batch_size": None, "normalized_stat_utility": None})

rows = []

# === Helper to assign short trainer IDs ===
trainer_to_num = {}
next_trainer_num = 1

def get_trainer_num(trainer_id):
    global next_trainer_num
    if trainer_id not in trainer_to_num:
        trainer_to_num[trainer_id] = next_trainer_num
        next_trainer_num += 1
    return trainer_to_num[trainer_id]

# === Parse log ===
with open(LOG_FILE, "r") as f:
    for line in f:
        # Extract timestamp if available
        ts_match = timestamp_pattern.search(line)
        timestamp = ts_match.group(1) if ts_match else ""

        # FETCH WEIGHTS line → update context
        fetch_match = fetch_pattern.search(line)
        if fetch_match:
            trainer_id, round_num, data_id = fetch_match.groups()
            trainer_context[trainer_id]["round"] = int(round_num)
            trainer_context[trainer_id]["data_id"] = int(data_id)
            continue

        # DATABIN info
        databin_match = databin_pattern.search(line)
        if databin_match:
            databin_count, trainer_id = databin_match.groups()
            trainer_meta[trainer_id]["databin"] = int(databin_count)
            continue

        # BATCH SIZE info
        batch_match = batch_pattern.search(line)
        if batch_match:
            batch_size = int(batch_match.group(1))
            # Store globally (applied to all trainers until changed)
            for t in trainer_meta:
                trainer_meta[t]["batch_size"] = batch_size
            continue

        # NORMALIZED STAT UTILITY
        norm_match = normalized_stat_pattern.search(line)
        if norm_match:
            trainer_id, normalized_val = norm_match.groups()
            trainer_meta[trainer_id]["normalized_stat_utility"] = float(normalized_val)
            continue

        # STAT UTILITY info
        stat_match = stat_pattern.search(line)
        if stat_match:
            trainer_id, stat_utility, loss = stat_match.groups()
            ctx = trainer_context[trainer_id]
            round_num = ctx["round"]
            data_id = ctx["data_id"]

            # Increment iteration for (trainer, round, data_id)
            iteration_tracker[trainer_id][(round_num, data_id)] += 1
            iteration = iteration_tracker[trainer_id][(round_num, data_id)]

            # Get meta info
            databin = trainer_meta[trainer_id]["databin"]
            batch_size = trainer_meta[trainer_id]["batch_size"]
            normalized_val = trainer_meta[trainer_id]["normalized_stat_utility"]

            # Map trainer to number
            trainer_num = get_trainer_num(trainer_id)

            rows.append({
                "timestamp": timestamp,
                "trainer_num": trainer_num,
                "trainer_id": trainer_id,
                "round": round_num,
                "data_id": data_id,
                "iteration": iteration,
                "stat_utility": float(stat_utility),
                "loss": float(loss),
                "normalized_stat_utility": normalized_val,
                "databins": databin,
                "batch_size": batch_size,
            })

# === Sort by timestamp ===
rows.sort(key=lambda r: r["timestamp"])

# === Write CSV ===
with open(OUTPUT_CSV, "w", newline="") as csvfile:
    fieldnames = [
        "timestamp",
        "trainer_num",
        "trainer_id",
        "round",
        "data_id",
        "iteration",
        "stat_utility",
        "loss",
        "normalized_stat_utility",
        "databins",
        "batch_size"
    ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"✅ Parsed {len(rows)} stat_utility entries.")
print(f"📄 Output saved to: {OUTPUT_CSV}")
print(f"👥 Trainer mapping:")
for t, num in trainer_to_num.items():
    print(f"  Trainer {num}: {t}")



# import re
# import csv
# from collections import defaultdict
# from pathlib import Path

# # === Paths ===
# LOG_FILE = Path("/home/dgarg39/shreya/flame/lib/python/examples/fwdllm/expts/run_tc_expts/log/new/test_trainer_fedFwd_distilbert_agnews_lr0.01_client_num_5_numerical_07_10_18_14.log")
# OUTPUT_CSV = LOG_FILE.with_suffix(".parsed.csv")

# # === Regex patterns ===
# fetch_pattern = re.compile(
#     r"FETCH WEIGHTS complete for trainer_id (\w+), round: (\d+), data id: (\d+)"
# )
# stat_pattern = re.compile(
#     r"stat_utility for trainerId: (\w+) is ([\d\.]+), loss: ([\d\.]+)"
# )
# timestamp_pattern = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+)")

# # === State tracking ===
# # Current (round, data_id) per trainer
# trainer_context = defaultdict(lambda: {"round": None, "adata_id": None})
# # Iteration count per trainer per (round, data_id)
# iteration_tracker = defaultdict(lambda: defaultdict(int))

# rows = []

# # === Parse file ===
# with open(LOG_FILE, "r") as f:
#     for line in f:
#         # Extract timestamp
#         ts_match = timestamp_pattern.search(line)
#         timestamp = ts_match.group(1) if ts_match else ""

#         # If FETCH line found → update round/data_id
#         fetch_match = fetch_pattern.search(line)
#         if fetch_match:
#             trainer_id, round_num, data_id = fetch_match.groups()
#             trainer_context[trainer_id]["round"] = int(round_num)
#             trainer_context[trainer_id]["data_id"] = int(data_id)
#             # no reset here — we track iterations per (round, data_id)
#             continue

#         # If stat_utility line found → log entry
#         stat_match = stat_pattern.search(line)
#         if stat_match:
#             trainer_id, stat_utility, loss = stat_match.groups()

#             ctx = trainer_context[trainer_id]
#             round_num = ctx["round"]
#             data_id = ctx["data_id"]

#             # Increment iteration count for this specific (trainer, round, data_id)
#             iteration_tracker[trainer_id][(round_num, data_id)] += 1
#             iteration = iteration_tracker[trainer_id][(round_num, data_id)]

#             rows.append({
#                 "timestamp": timestamp,
#                 "trainer_id": trainer_id,
#                 "round": round_num,
#                 "data_id": data_id,
#                 "iteration": iteration,
#                 "stat_utility": float(stat_utility),
#                 "loss": float(loss),
#             })

# # === Write CSV ===
# with open(OUTPUT_CSV, "w", newline="") as csvfile:
#     fieldnames = ["timestamp", "trainer_id", "round", "data_id", "iteration", "stat_utility", "loss"]
#     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#     writer.writeheader()
#     writer.writerows(rows)

# print(f"Parsed {len(rows)} stat_utility entries.")
# print(f"Output saved to: {OUTPUT_CSV}")
