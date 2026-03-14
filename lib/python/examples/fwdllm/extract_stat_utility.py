import sys
import re
import csv
import os

def extract_utilities(log_file, output_file):
    if not os.path.exists(log_file):
        print(f"Error: Log file not found at {log_file}")
        sys.exit(1)

    print(f"Extracting stat_utility metrics from {log_file}...")

    # Regex patterns for matching top_aggregator and fwdllm_aggregator logs
    # top_agg_pattern = re.compile(r"Received weights from (\S+)\. It was trained on model version (\d+), with (\d+) samples\. Returned partial stat utility ([\d\.]+) and full stat utility ([\d\.]+)")
    fwdllm_agg_pattern = re.compile(r"Aggregated utilities for (\S+)\. Partial stat utility used for FedBuff: ([\d\.]+)\. Full stat utility stored for Oort: ([\d\.]+)")
    
    # Regex for async_oort selector debug logs (if enabled)
    oort_selector_pattern = re.compile(r"Trainer (\S+) full_dataset_stat_utility: unnormalized = ([\d\.]+), normalized = ([\d\.]+), partial_dataset_stat_utility: ([\d\.]+)")

    results = []

    with open(log_file, 'r') as f:
        for line in f:
            # Match top aggregator logs
            m_top = top_agg_pattern.search(line)
            if m_top:
                tid = m_top.group(1)
                version = m_top.group(2)
                samples = m_top.group(3)
                partial_util = m_top.group(4)
                full_util = m_top.group(5)
                results.append([tid, version, samples, partial_util, full_util, "N/A", "top_aggregator"])
                continue

            # Match fwdllm aggregator logs
            m_fwd = fwdllm_agg_pattern.search(line)
            if m_fwd:
                tid = m_fwd.group(1)
                partial_util = m_fwd.group(2)
                full_util = m_fwd.group(3)
                results.append([tid, "N/A", "N/A", partial_util, full_util, "N/A", "fwdllm_aggregator"])
                continue

            # Match async oort selector logs
            m_oort = oort_selector_pattern.search(line)
            if m_oort:
                tid = m_oort.group(1)
                unnorm_full = m_oort.group(2)
                norm_full = m_oort.group(3)
                partial_util = m_oort.group(4)
                results.append([tid, "N/A", "N/A", partial_util, unnorm_full, norm_full, "async_oort_selector"])

    # Write results to CSV
    with open(output_file, 'w', newline='') as out_csv:
        writer = csv.writer(out_csv)
        writer.writerow(["TrainerID", "ModelVersion", "Samples", "PartialStatUtility", "FullStatUtility_Unnormalized", "FullStatUtility_Normalized", "Source"])
        writer.writerows(results)

    print(f"Done! {len(results)} records extracted. Output saved to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <path_to_log_file> [output_csv_file]")
        sys.exit(1)
        
    log_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else "stat_utility_comparison.csv"
    
    extract_utilities(log_path, out_path)
