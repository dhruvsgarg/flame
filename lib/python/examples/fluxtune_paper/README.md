# SwiFT (FluxTune) paper figures

Plots for [fluxtune_paper_outline.md](../../../../fluxtune_paper_outline.md) §12 (Background and Motivation). Style:
[plot_guideline_fluxtune.md](../_metadata/plot_guideline_fluxtune.md) + `code/ft_style.py` (copy of
`paper_expts_fluxtune/plots/ft_style.py`, with marker-only lines excluded from the overlap path test).

    cd code && python3 make_motiv_figs.py [--only motiv_collapse ...]   # -> figs/background/<name>.{pdf,png}

Fig 1 teaser (`fig:intro-teaser`, one column, 3.25 × 1.6 in): `python3 make_intro_fig.py` → `figs/intro/fig1_teaser`;
SmolLM2-1.7B · Yahoo, run 6c vs FwdLLM (jayne), digests `data/sm6c_fluxtune_e2e.json`, `data/sm_fwdllm_e2e.json`.

| panel | label | file | runs |
|---|---|---|---|
| 2a | `fig:motiv-collapse` | `figs/background/motiv_collapse` | q1, q4, r4a |
| 2b | `fig:motiv-growth` | `figs/background/motiv_growth` | q1–q5b, 1.7B FwdLLM, r2s, r3a, r3b, r8a, r8b, r4a |
| 2c | `fig:motiv-gate` | `figs/background/motiv_gate` | q4 |
| 2d | `fig:motiv-rho` | `figs/background/motiv_rho` | q4, r4a |
| 2e | `fig:motiv-hold` | `figs/background/motiv_hold` | run 1, 1.7B FwdLLM, MobiPerf `states_2st` sessions |

Each panel is 1.35 × 1.75 in (one fifth of the 6.75 in text width, legend above): include at `width=\linewidth` inside a
1.35 in subfigure, never scaled. Fonts are 7 / 6.5 pt (guideline's 8 / 7 pt does not fit a 1.35 in panel).

**Data.** `data/<run>.json` are per-run digests from `code/digest_motiv.py` (stdlib). q*, r* and run 1 were digested on
shepherd (`/home/dgarg39/aish_test/flame/lib/python/examples/fwdllm/`: `paper_pitch_20261029{a,b}`, `paper_runs_20261029{,y}`,
`p1_distil_20261025/bench`); `sm_fwdllm` on jayne (`experiments/run_20261028_182751_fwdllm_yahoo_n60_smoke_syn_0_sim`). To
refresh: rerun the digest on the node that holds the run and copy the JSON here.

**Definitions.** Measured Φ = ‖θ_t‖ / ‖θ_0‖ (initial trainable norm). ρ_t = ‖Δθ_t‖ / ‖θ_{t−1}‖ (logged update norm over the
pre-step norm). The logged `rho` divides by the post-step norm and under-reads large steps; using it is what made r4a look
13% off.
