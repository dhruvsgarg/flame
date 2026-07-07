# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""plotlib — shared plotting core for the FWDLLM/FLUXTUNE experiment suite.

  style      SOCC-2026 rcParams + PDF/timestamp output convention (one "look")
  baselines  display registry: names, colors, emphasis (one place to re-brand)
  reducers   streaming telemetry -> RunResult (one source of the 5 metrics)
  figures    single-panel paper-figure builders over RunResult lists

`make_paper_figs.py`, `plot_run.py` and `compare_baselines.py` all build on these.
"""
