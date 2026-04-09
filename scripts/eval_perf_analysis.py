# %% [markdown]
# # Eval Model Performance Analysis
# This notebook extracts the `eval_model` logic from `fwdllm_aggregator.py` to analyze performance bottlenecks across different batch sizes.

# %%
import torch
import time
import numpy as np
import logging
import gc
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoConfig
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# %%
def run_eval_benchmark(batch_size=8, num_samples=7600, model_name="distilbert-base-uncased"):
    num_labels = 4
    config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
    model = AutoModelForSequenceClassification.from_config(config).to(device)
    model.eval()

    # Dummy Data
    max_seq_len = 128
    input_ids = torch.randint(0, 30000, (num_samples, max_seq_len))
    input_mask = torch.ones((num_samples, max_seq_len))
    segment_ids = torch.zeros((num_samples, max_seq_len))
    label_ids = torch.randint(0, num_labels, (num_samples,))
    dataset = TensorDataset(input_ids, input_mask, segment_ids, label_ids)
    total_len = len(dataset)
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    loss_fct = CrossEntropyLoss()
    
    # Timing metrics
    batch_times = []
    data_move_times = []
    forward_times = []
    cpu_move_times = []
    
    torch.cuda.synchronize() if device == "cuda" else None
    start_time = time.time()
    
    for i, batch in enumerate(loader):
        b_start = time.time()
        
        # 1. To Device
        t0 = time.time()
        x = batch[0].to(device)
        labels = batch[3].to(device)
        torch.cuda.synchronize() if device == "cuda" else None
        data_move_times.append(time.time() - t0)
        
        # 2. Forward Pass
        t1 = time.time()
        with torch.no_grad():
            output = model(x)
            logits = output[0]
        torch.cuda.synchronize() if device == "cuda" else None
        forward_times.append(time.time() - t1)
        
        # 3. CPU Move & Metrics (simulating preds storage)
        t2 = time.time()
        _ = logits.detach().cpu().numpy()
        _ = labels.detach().cpu().numpy()
        torch.cuda.synchronize() if device == "cuda" else None
        cpu_move_times.append(time.time() - t2)
        
        batch_times.append(time.time() - b_start)
        
    total_duration = time.time() - start_time
    return {
        "total": total_duration,
        "batch_avg": np.mean(batch_times),
        "data_move_avg": np.mean(data_move_times),
        "forward_avg": np.mean(forward_times),
        "cpu_move_avg": np.mean(cpu_move_times),
        "batch_times": batch_times
    }

# %%
bs_list = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
results = {}

for bs in bs_list:
    print(f"Benchmarking BS={bs}...")
    results[bs] = run_eval_benchmark(batch_size=bs)
    print(f"  Total Time: {results[bs]['total']:.2f}s")

# %%
# Plotting Results
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(bs_list, [results[bs]['total'] for bs in bs_list], marker='o')
plt.title("Total Execution Time vs Batch Size")
plt.xlabel("Batch Size")
plt.ylabel("Time (s)")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(bs_list, [results[bs]['forward_avg'] * 1000 for bs in bs_list], label="Forward", marker='x')
plt.plot(bs_list, [results[bs]['data_move_avg'] * 1000 for bs in bs_list], label="Data Move", marker='s')
plt.plot(bs_list, [results[bs]['cpu_move_avg'] * 1000 for bs in bs_list], label="To CPU", marker='d')
plt.title("Avg Time per Batch (ms)")
plt.xlabel("Batch Size")
plt.ylabel("Time (ms)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## CDF of Batch Latency
# If you want to see the distribution of latencies for a specific batch size:

# %%
def plot_cdf(data, label):
    sorted_data = np.sort(data)
    yvals = np.arange(len(sorted_data))/float(len(sorted_data)-1)
    plt.plot(sorted_data, yvals, label=label)

plt.figure(figsize=(10, 6))
for bs in [8, 256]:
    plot_cdf(results[bs]['batch_times'], f"BS={bs}")

plt.title("CDF of Batch Latencies")
plt.xlabel("Latency (s)")
plt.ylabel("CDF")
plt.legend()
plt.grid(True)
plt.show()


