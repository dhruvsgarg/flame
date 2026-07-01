# Migrate `fwdllm` onto the `flame.launch` YAML launcher

The fwdllm migration is complete. Current guidance lives in
[`../MIGRATING_TO_LAUNCHER.md`](../MIGRATING_TO_LAUNCHER.md), § 9
"fwdllm-specific migration notes" — dataset path-style handling, `client_idx`
injection, the single-aggregator-entrypoint pattern, custom stopping
criteria, availability traces, NLP dependencies, the baseline taxonomy, and
lessons from smoke-testing (CUDA/logging ordering, variance-check guards,
staleness policy, `run_sequential.sh` conventions).

For the original phase-by-phase investigation notes (bugs found, decisions
made, smoke-test results), see `git log -- lib/python/examples/fwdllm/MIGRATION_TO_LAUNCHER_FWDLLM.md`.
