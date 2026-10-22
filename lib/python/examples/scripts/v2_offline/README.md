# Felix v2 offline analyses (10-22; FELIX_READINESS Active build "Felix v2")

Exploratory tools behind the Felix v2 design; FX-N129 folds the reusable parts into the bench and deletes the rest.
Run from `lib/python/examples` with the `dg_flame` python, `-I` for the ones that take a run dir.

| script | what |
|---|---|
| oracle_probe.py | per-checkpoint marginal update value and oracle buffer weights (test half A fit, half B graded); `Probe` = shared data / model / client rebuild |
| oracle_replay.py | closed-loop 40-version replay from a checkpoint under selection / aggregation policies incl. oracles |
| replay.py | fidelity check: replays a run's real commits from a checkpoint (B1) |
| ceiling.py / ceiling_full.py / ceiling300.py | centralized ceiling on visible data at t / shard shares / n 300 |
| novelty.py | loss on newly arrived vs trained samples (FX-T67 re-check) |
| aggsig.py | which signals predict oracle weights |
| holdout_traj.py / holdout_power.py | population held-out loss vs test; paired per-client detection power |
| pacer_model.py / pacer_rule.py | training-loss-change pacer signal (rejected) |
| legstats.py / fresh.py | per-leg throughput, concentration, fresh-data consumption |
