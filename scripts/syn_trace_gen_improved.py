import numpy as np
import json
import os
import glob
import re
import matplotlib.pyplot as plt

def batch_inject_and_plot(folder_path='.', max_trainers=100, train_p=0.90, eval_p=0.05, 
                          unavail_p=0.05, total_minutes=1440, interval=10, stickiness=0.95):
    states = ['UN_AVL', 'AVL_EVAL', 'AVL_TRAIN']
    target_dist = np.array([unavail_p, eval_p, train_p])
    total_rounds = total_minutes // interval
    
    # 1. ROBUST TRANSITION MATRIX (Detailed Balance)
    n = len(target_dist)
    trans_matrix = np.zeros((n, n))
    
    # We define a base transition rate 'alpha'
    # High stickiness means alpha is small
    alpha = 1.0 - stickiness 
    
    for i in range(n):
        for j in range(n):
            if i != j:
                # The probability of moving i -> j depends on the target density of j
                # This ensures the Markov chain is 'pulled' toward the target distribution
                trans_matrix[i, j] = alpha * target_dist[j]
        
        # The diagonal (staying put) is 1 minus the sum of moving elsewhere
        trans_matrix[i, i] = 1.0 - np.sum(trans_matrix[i, :])

    # 2. File and Key Setup
    key_name = f"avl_events_syn_train_{int(train_p*100)}_eval_{int(eval_p*100)}_unavail_{int(unavail_p*100)}"
    search_pattern = os.path.join(folder_path, "trainer_*.json")
    files = glob.glob(search_pattern)
    files.sort(key=lambda f: int(re.sub('\D', '', os.path.basename(f)) or 0))
    files_to_process = files[:max_trainers]

    if not files_to_process:
        print(f"No files found in {folder_path}")
        return

    all_states_history = np.zeros((total_rounds, len(files_to_process)))

    for node_idx, file_path in enumerate(files_to_process):
        with open(file_path, 'r') as f:
            data = json.load(f)

        node_trace = []
        # INITIALIZE BASED ON TARGET
        curr_state_idx = np.random.choice([0, 1, 2], p=target_dist)
        all_states_history[0, node_idx] = curr_state_idx
        last_state_name = states[curr_state_idx]
        node_trace.append((0, last_state_name))
        
        for r in range(1, total_rounds):
            curr_state_idx = np.random.choice([0, 1, 2], p=trans_matrix[curr_state_idx])
            all_states_history[r, node_idx] = curr_state_idx
            curr_state_name = states[curr_state_idx]
            
            if curr_state_name != last_state_name:
                node_trace.append((r * interval * 60, curr_state_name))
                last_state_name = curr_state_name

        if "hyperparameters" in data:
            data["hyperparameters"][key_name] = str(node_trace)
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=4)

    # 3. Validation Plot
    time_axis = np.arange(total_rounds) * interval * 60
    plt.figure(figsize=(14, 5))
    colors = ['red', 'blue', 'green']
    for i, state_name in enumerate(states):
        pct = np.mean(all_states_history == i, axis=1) * 100
        plt.plot(time_axis, pct, label=f'Actual {state_name}', color=colors[i], linewidth=1.5)
        plt.axhline(y=target_dist[i]*100, color=colors[i], linestyle='--', alpha=0.5, label=f'Target {state_name}')

    plt.title(f"Corrected Distribution ({max_trainers} Trainers, Stickiness {stickiness})")
    plt.ylabel("% of Trainers")
    plt.xlabel("Seconds")
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'state_distribution_train_{int(train_p*100)}_eval_{int(eval_p*100)}_unavail_{int(unavail_p*100)}.png')
    plt.show()

# Run the corrected version
batch_inject_and_plot(
    folder_path='/home/dgarg39/aish_test/flame/lib/python/examples/fwdllm/expts/run_tc_expts/json_scripts/',
    max_trainers=100,
    train_p=0.50,
    eval_p=0.30,
    unavail_p=0.20,
    stickiness=0.90 
)