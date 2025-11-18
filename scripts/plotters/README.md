### Steps to run the plotting script:

1. Checkout to the `dev/log_parser` branch. I use a cloned repo on my laptop, just for plotting (optional)
2. Update log_parser.py with log file:
    - Jump to this line: `log_file_type = "flame_fwdllm_aggregator"` and start making the following additions, commenting the previous values of these 3 variables:
        - Populate `log_file` with the aggregator log file path
        - Populate `suffix` to describe the run variation. e.g - reject_stale, keep_stale, ...
        - Update `EXPORT_CONFIG...['evaluation_metrics']['default_output_filename']`  with a prefix that describes the setup. e.g - async_k10_c50_n150
3. Update comparative_plotter.py with parsed file:
    1. Make the following additions at the top of the file, commenting the previous values of these 2 variables:
        1. Add `SYSTEM1_FILES` = ["output/{prefix}-{suffix}.csv"]
        2. Add `SYSTEM1_NAME` to describe the experiment in words
        3. Do the same for the SYSTEM2 variables
4. Run commands:

python log_parser.py
python comparative_plotter.py

5. You can see 3 output plots generated in the `plots` directory