# Felix paper figures

One place for every planned figure in [felix_paper_outline.md](../_metadata/felix_paper_outline.md), title → §5. Regenerate all:

    python make_figs.py [--only s1 s3 s4 s5] [--refresh]      # dg_flame env; PDF + PNG into figs/paper/<section>/

- `figs/paper/<section>/`: paper figures (§1-§5). `figs/expts/`: evaluation plots (later).
- `python make_tables.py`: headline table from `figs/expts/results_*.json` → `figs/expts/table_headline.{tex,json}` + `numbers.tex`.
- `data/fx_n147_<trace>.json`: FX-N147 trace data, from `../scripts/eligibility_forecast.py` (rerun it to refresh).
- `data/runs/`: per-run telemetry cache (`extract.py`); `--refresh` re-reads the runs.
- **Style:** [plot_guideline.md](../_metadata/plot_guideline.md) (rules) + [plot_config.py](plot_config.py) (the one master config:
  `SYS_NAME`, sizes, fonts, palette, rcParams). Every save runs `figlib.overlaps()`; a figure that prints `OVERLAP` is not done.

**Paper figures (FB2 seed 1 unless noted; all one column, 3.25 in wide, except F2, F6 and the appendix curves):**

| section | figure | file |
|---|---|---|
| §1 | F1 teaser: cifar MobiPerf, accuracy vs time and GB | `figs/expts/teaser` |
| §3.1 | F2 (figure*): pool + turnover, window-length CDF vs task time, autocorrelation, forecast AUC | `figs/paper/s3_motivation/fig2_trace` |
| §3.2 | F3 slot time lost by cause, baselines only | `figs/expts/slots_base` |
| §3.3 | F4 filling every slot (`b1` bound) vs FedBuff | `figs/expts/fill_slots` |
| §3.4 | F5a eligible pool vs FedBuff's c (no FLuX) | `figs/expts/concurrency_base` |
| §3.4 | F5b choice vs c / eligible: trace replay (`scripts/choice_replay.py` → `data/choice_replay_*.json`) + run markers | `figs/expts/choice_replay` |
| §4 | F6 architecture (figure*), F7 toy timeline | `figs/paper/s4_overview/fig_architecture`, `fig_toy_timeline` |
| §5.1 | F8 slot time lost per departure (log) | `figs/expts/stall` |
| §5.2 / eval | F9 selection: picks–size correlation, samples per update (cifar; speech for eval) | `figs/expts/picks`, `picks_speech` |
| §5.3 | F10a FLuX c vs eligible pool; F10b time / GB to each accuracy level per c arm | `figs/expts/concurrency`, `pace_levels` |
| §5.3 | FLuX vs fixed c = 60: accuracy vs GB | `figs/expts/pace_bytes` |
| §5.4 | F11a staleness weight per rule (analytic); F11b aggregation swap | `figs/paper/s5_design/fig5_4_agg`, `figs/expts/swap_aggregation` |
| eval | Slot time lost by cause, with FLuX | `figs/expts/slots` |
| ablations | F12 ladder, F13 swaps: peak accuracy, then time and GB to target | `figs/expts/ladder`, `swap_*` |
| appendix | Oracle gain, sync round time, staleness by source | `figs/expts/oracle_gain`, `rounds`, `staleness` |
| appendix | Accuracy vs time, all 8 cells (full width) | `figs/expts/curves` |

Regenerate: `python scripts/paper_plots.py --root $R/paper_root` (all data figures) and `python paper_felix/make_figs.py` (§3.1
traces, §4 schematics, §5.4 analytic).
