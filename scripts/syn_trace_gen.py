import random
import json

def generate_mobiperf_traces(num_trainers=100, duration_sec=3600, 
                             p_unavl=0.10, p_eval=0.20, p_train=0.70):
    """
    Generates synthetic state traces for mobile trainers.
    """
    states = ['UN_AVL', 'AVL_EVAL', 'AVL_TRAIN']
    target_dist = [p_unavl, p_eval, p_train]
    
    # Configuration: Average time (seconds) spent in a state before switching
    # Adjusting these changes the 'frequency' of churn
    avg_stay_duration = 300 
    
    all_traces = {}

    for i in range(num_trainers):
        current_time = 0
        trace = []
        
        # Initial state based on target distribution
        current_state = random.choices(states, weights=target_dist)[0]
        trace.append((current_time, current_state))
        
        while current_time < duration_sec:
            # 1. Determine how long to stay in current state (Exponential distribution)
            stay_duration = int(random.expovariate(1.0 / avg_stay_duration))
            if stay_duration < 1: stay_duration = 1
            
            current_time += stay_duration
            if current_time >= duration_sec:
                break
                
            # 2. Transition to a NEW state
            # To maintain steady state, we sample from the target distribution excluding current state
            remaining_states = [s for s in states if s != current_state]
            remaining_weights = [target_dist[states.index(s)] for s in remaining_states]
            
            current_state = random.choices(remaining_states, weights=remaining_weights)[0]
            trace.append((current_time, current_state))
        
        all_traces[f"trainer_{i}"] = str(trace)

    return all_traces

# --- Configuration ---
TRAINERS = 100
MINUTES = 60
DURATION = MINUTES * 60 # Convert to seconds

# Targets: 10% UN_AVL, 20% AVL_EVAL, 70% AVL_TRAIN
traces = generate_mobiperf_traces(
    num_trainers=TRAINERS, 
    duration_sec=DURATION,
    p_unavl=0.10, 
    p_eval=0.20, 
    p_train=0.70
)

# Output example
print(json.dumps({"synthetic_trace": traces["trainer_0"]}, indent=4))