# Felix paper figures

One place for every planned figure in [felix_paper_outline.md](../_metadata/felix_paper_outline.md), title → §5. Regenerate all:

    python make_figs.py [--only s1 s3 s4 s5] [--refresh]      # dg_flame env; PDF + PNG into figs/paper/<section>/

- `figs/paper/<section>/`: paper figures (§1-§5). `figs/expts/`: evaluation plots (later).
- `python make_tables.py`: headline table from `figs/expts/results_*.json` → `figs/expts/table_headline.{tex,json}` + `numbers.tex`.
- `data/fx_n147_<trace>.json`: FX-N147 trace data, from `../scripts/eligibility_forecast.py` (rerun it to refresh).
- `data/runs/`: per-run telemetry cache (`extract.py`); `--refresh` re-reads the runs.
- **Style:** [plot_config.py](plot_config.py) is the one master config (fonts, line widths, ticks, colors per stack / predictor /
  trace, z-order: grid under bars, lines over bars). Every figure script, incl. later eval plots, imports it via `figlib`.
- **Status labels** (per figure, or per panel in mixed figures): **MOCK** = placeholder numbers, **PRELIM** = real data, not final
  (1 seed, 12 h, placeholder target, or another cell), **DRAFT** = schematic. Real figures and panels carry no label.

| section | figure | file | status | data now → data needed |
|---|---|---|---|---|
| §1 | Fig 1 teaser: accuracy vs time and vs GB | `s1_intro/fig1_teaser` | PRELIM | F49 cifar `smu` → FB1 at the FX-N146 target, 3 seeds |
| §3.1 | Fig 2 eligibility: pool, sessions, memory, forecast AUC / Brier vs the trace | `s3_motivation/fig2_eligibility` | real (FX-N147) | MobiPerf + syn_50 traces, n 300, days 1-5 |
| §3.2 | Fig 3 sync stacks vs oracle + round length | `s3_motivation/fig3_sync_oracle` | PRELIM | F49 cifar → speech (outline cell) |
| §3.3 | Fig 4 slots taken vs computing; filling slots | `s3_motivation/fig4_slots` | (a) PRELIM, (b) MOCK | (b) F44 numbers → bound arm `b1` + FedBuff* on FL1 cells |
| §3.4 | Fig 5 staleness by source | `s3_motivation/fig5_staleness_source` | PRELIM | F49 FedBuff → FB1 seeds |
| §3.5 | Fig 6 eligible pool vs c; choice vs c / pool | `s3_motivation/fig6_choice` | (a) real, (b) MOCK | (b) → c sweep (F48a cells) with ρ(picks, size) |
| §4 | architecture, numbered steps | `s4_overview/fig_architecture` | DRAFT | schematic |
| §4 | overview by example (FedBuff vs Felix) | `s4_overview/fig_toy_timeline` | DRAFT | schematic |
| §5.1 | Felix-Notify: Felix by check-in (K1) | `s5_design/fig5_1_notify` | MOCK | F42b numbers → `fck` on FL1 cells |
| §5.2 | Felix-Select: ρ(picks, size), samples / update; speed term | `s5_design/fig5_2_select` | (a, b) PRELIM, (c) MOCK | (c) F45-F46 numbers → speed-term arm on FL1 cells |
| §5.3 | Felix-Pace: c vs pool, φ, GB vs accuracy | `s5_design/fig5_3_pace` | PRELIM | F49 `fn_chr` vs `fn` → FB1 seeds |
| §5.4 | Felix-Agg: weight vs τ; time to 50% | `s5_design/fig5_4_agg` | (a) real (analytic), (b) MOCK | (b) F46 ranges → rate arms on FL1 cells |

§2 has no figure (Table 1 is text). Table 2 (§3.2, knowledge per stack) takes its forecast rows from `data/fx_n147_*.json`.
