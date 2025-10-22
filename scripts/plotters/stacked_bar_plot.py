import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- User Input ---
# List your 4 CSV file names here
csv_files = [
    # 'output/sync_1hr_train_times.csv',
    # 'output/sync_stagglers_1hr_train_times.csv',
    # 'output/train_times_noDelay_2_5hrs.csv',
    # 'output/train_times_delayBy20_3hrs.csv',
    'output/train_times_noDelay_slow_4hr.csv',
    'output/train_times_delay_slow_6hr.csv',
    'async_with_stragglers.csv',
    'async_no_stragglers.csv',
]

# X-axis labels for the bars
x_labels = [
    'Sync\n(no stragglers)',
    'Sync\n(stragglers)',
    'Async\n(no stragglers)',
    'Async\n(stragglers)',
]

# --- Data Processing ---
training_proportions = []
idle_proportions = []

for file in csv_files:
    # Read only the last line of the CSV file
    # We use tail(1) to get the last row
    try:
        df = pd.read_csv(file).tail(1)

        # Extract the values
        avg_training_time = df['mean:cumulative_recv_weights_time'].iloc[0]
        total_time = df['time_since_start'].iloc[0]

        # Calculate proportions
        train_proportion = avg_training_time / total_time
        idle_proportion = 1 - train_proportion

        training_proportions.append(train_proportion)
        idle_proportions.append(idle_proportion)

    except FileNotFoundError:
        print(f"Error: The file '{file}' was not found. Please check the filename and path.")
        # Add placeholder data to allow the script to continue for demonstration
        training_proportions.append(0)
        idle_proportions.append(0)
    except KeyError as e:
        print(f"Error: Column {e} not found in '{file}'. Please check your CSV file's header.")
        training_proportions.append(0)
        idle_proportions.append(0)


# --- Plotting ---
fig, ax = plt.subplots(figsize=(10, 6))

# Convert proportions to numpy arrays for plotting
training_proportions = np.array(training_proportions)
idle_proportions = np.array(idle_proportions)

# Bar positions
ind = np.arange(len(x_labels))

# Create the stacked bar chart
ax.bar(ind, training_proportions, label='Stall Time')
ax.bar(ind, idle_proportions, bottom=training_proportions, label='Train Time')

# --- Chart Customization ---
ax.set_ylabel('Proportion of Time')
ax.set_title('Proportion of Time Spent in Training vs. Idle')
ax.set_xticks(ind)
ax.set_xticklabels(x_labels)
ax.legend()

# Add a note about the number of clients
plt.figtext(0.5, 0.01, 'Average across 10 clients', ha='center', fontsize=10, style='italic')


# Display the plot
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for the figtext
plt.savefig("stacked_bar_chart.png")
plt.show()