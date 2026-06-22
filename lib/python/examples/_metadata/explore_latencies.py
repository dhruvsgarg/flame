import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define file path
FILE_PATH = 'lib/python/examples/_metadata/satellite_latencies.npy'

if not os.path.exists(FILE_PATH):
    raise FileNotFoundError(f"Could not find the file at {FILE_PATH}. Check your current working directory!")

# Load data
data = np.load(FILE_PATH)
print("=" * 50)
print(f"Loaded Array Shape: {data.shape}")
print("=" * 50)

# Print first 10 rows of first 10 columns
print("First 10 Rows and First 10 Columns:")
if data.ndim > 1:
    # Slices rows 0-9 and columns 0-9 cleanly
    print(data[:10, :10])
else:
    print("Note: Array is 1-dimensional. Printing first 10 elements instead:")
    print(data[:10])
print("=" * 50)

# Create a figure with two subplots side-by-side or stacked
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# ------------------------------------------------------------------
# PLOT 1: Single Satellite Latency Over Time
# ------------------------------------------------------------------
# track Satellite Index 4 (Column 4) over the first 1000 time steps
satellite_id = 4
time_steps = 1000

ax1.plot(data[:time_steps, satellite_id], color='darkorange', linewidth=2, label=f'Satellite #{satellite_id}')
ax1.set_title(f'Orbital Latency Profile for a Single LEO Satellite (First {time_steps} Steps)', fontsize=14)
ax1.set_xlabel('Simulation Time Step', fontsize=12)
ax1.set_ylabel('Latency (Seconds)', fontsize=12)
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.legend(fontsize=11)