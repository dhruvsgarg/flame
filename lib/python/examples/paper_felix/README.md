# Felix paper figures

One place for every planned figure in [felix_paper_outline.md](../_metadata/felix_paper_outline.md), title → §5. Regenerate all:

    python make_figs.py [--only s1 s3 s4 s5] [--refresh]      # dg_flame env; PDF + PNG into figs/paper/<section>/

- `figs/paper/<section>/`: paper figures (§1-§5). `figs/expts/`: evaluation plots (later).
- `python make_tables.py`: headline table from `figs/expts/results_*.json` → `figs/expts/table_headline.{tex,json}` + `numbers.tex`.
- `data/fx_n147_<trace>.json`: FX-N147 trace data, from `../scripts/eligibility_forecast.py` (rerun it to refresh).
- `data/runs/`: per-run telemetry cache (`extract.py`); `--refresh` re-reads the runs.
- **Style:** [plot_guideline.md](../_metadata/plot_guideline.md) (rules) + [plot_config.py](plot_config.py) (the one master config:
  `SYS_NAME`, sizes, fonts, palette, rcParams). Every save runs `figlib.overlaps()`; a figure that prints `OVERLAP` is not done.

**Paper figures (FB2 seed 1 unless noted; all one column, 3.25 in wide, except the appendix curves):**

| section | figure | file |
|---|---|---|
| §1 | Fig 1 teaser: cifar MobiPerf, accuracy vs time and GB | `figs/expts/teaser` |
| §3.1 | Eligible pool + eligibility memory (FX-N147 traces) | `figs/paper/s3_motivation/fig2_eligibility` |
| §3.1 | Forecast AUC vs horizon (FX-N147) | `figs/paper/s3_motivation/fig2_forecast` |
| §3.2 | Oracle gain in peak accuracy per stack and trace | `figs/expts/oracle_gain` |
| §3.2 | Sync round time by component, ± oracle | `figs/expts/rounds` |
| §3.3 | Filling every slot (`b1` bound) vs FedBuff | `figs/expts/fill_slots` |
| §3.3 / eval | Slot time lost by cause (share) | `figs/expts/slots` |
| §3.3 / eval | Slot time lost per departure (log) | `figs/expts/stall` |
| §3.4 | Staleness: concurrency vs held across a departure | `figs/expts/staleness` |
| §3.5 | Choice vs concurrency / eligible pool | `figs/expts/choice` |
| §3.5 / §5.3 | Concurrency vs eligible pool | `figs/expts/concurrency` |
| §5.3 | FLuX vs fixed c = 60: accuracy vs GB | `figs/expts/pace_bytes` |
| §5.4 | Staleness weight per rule (analytic) | `figs/paper/s5_design/fig5_4_agg` |
| §5.2 / eval | Selection: picks–size correlation, samples per update | `figs/expts/picks` |
| ablations | Ladder, swaps (availability, selection, aggregation, concurrency) | `figs/expts/ladder`, `swap_*` |
| appendix | Accuracy vs time, all 8 cells (full width) | `figs/expts/curves` |
| §4 | Architecture, overview-by-example timeline (schematics) | `figs/paper/s4_overview/` |

Regenerate: `python scripts/paper_plots.py --root $R/paper_root` (all data figures) and `python paper_felix/make_figs.py` (§3.1
traces, §4 schematics, §5.4 analytic).
