# Paper plot guideline — FluxTune

FluxTune copy of [plot_guideline.md](plot_guideline.md) (Felix). Same rules for layout, content, process and checks; only the
look differs (font, palette, markers, line and grid style, fills: §1 font, §1a, §6) so the two papers' figures don't look alike.
Port new shared rules from plot_guideline.md; add FluxTune-only rules here.

## 1. Output format (camera-ready rules)

- **Vector PDF**, fonts embedded as **TrueType (Type 42)**: `pdf.fonttype = 42`, `ps.fonttype = 42`. Never Type 3 (rejected by
  many camera-ready checkers). Check: the PDF has `FontFile2`, no `/Type3`.
- **Raster fallback** (PNG, or rasterized dense scatter inside a PDF) at **≥ 300 dpi**.
- **One column by default.** MLSys (ICML-style template): column width 3.25 in (`\columnwidth`), text width 6.75 in, 0.25 in
  gutter. Almost every figure is one column; a full-width `figure*` only when one column is impossible (rare: many panels that
  must be read together).
- **Draw at print size; never let LaTeX scale.** `figsize` = the slot it fills.
  Use constrained layout with `savefig.bbox = None`, so the PDF is exactly `figsize`, then `\includegraphics[width=\columnwidth]`
  is a 1.0 scale and the font sizes below are the printed sizes. (`bbox="tight"` crops to a random width and LaTeX rescales it.)
- **Font = one sans-serif family** (Helvetica-like: TeX Gyre Heros → Liberation Sans → DejaVu Sans fallback;
  `mathtext.fontset = stixsans` so math matches). Felix uses the serif body font; FluxTune's sans figures set them apart.
- **Sizes in print:** labels 8 pt, ticks / legend / annotations 7 pt, nothing below 6 pt. Lines 0.9-1.2 pt, hero 1.8 pt,
  reference lines 0.8 pt dashed, grid 0.4 pt light.
- **One master config** (rcParams, sizes, colors, line styles, z-order) imported by every figure script; no per-figure style.
- **Aspect ratio:** about 2:1 (width : height) for a two-panel column figure: **3.25 × 1.6 in** is the default (any 2:1 size such
  as 3 × 1.5 is the same shape, but the width must equal the column), about 1.6:1 for a single panel, and 3-4:1 for a full-width row of 3-4
  panels (6.75 × 1.7-2.0 in). Below about 0.85 in of plot area per panel, close curves merge.

## 1a. FluxTune look (differs from Felix on purpose)

| element | Felix | FluxTune |
|---|---|---|
| font | STIX serif | sans (Heros / Liberation Sans / DejaVu Sans), `stixsans` math |
| palette | Okabe-Ito (blue hero) | Paul Tol muted / high-contrast (wine hero), §6 |
| ours marker | `o` filled | `D` filled |
| grid | light, both axes | y-only, dotted 0.4 pt, `#d9d9d9` |
| ticks | out | in, length 2.5 pt |
| line caps / joins | default | round |
| legend | — | frameless, handle length 2.2 |
| non-method fills | grey `#cfcfcf`, `//` `..` | warm sand `#ebe3d0`, `\\` `xx` |

Everything functional stays identical: sizes, line widths, black for text / axes / callouts / references, Type 42, exact
figsize, collision check.

## 2. Content

- **Every plot stands alone (context free):** legend, tick labels and axis labels always on the figure. Never move them to the
  caption to save space.
- **Axis labels are complete and descriptive** ("Test Accuracy (%)", "Data Moved (GB, log scale)", "Time (hours)"), with no
  abbreviations, as long as the label fits along its axis.
- **Standard symbols over abbreviations** the reader would need explained: % (not "pp"), × for ratios, h / GB / s units,
  ≤ ≥ ±, τ only if the paper defines it. Accuracy deltas are absolute and written "+1.8%" (the paper's text and caption use
  the same convention).
- **Capitalize:** the first letter of axis-label words (at least the first word), and the first word of every in-plot text,
  annotation and legend entry ("Target 75%", "2.8× fewer GB than FedBuff"), schematics included; a label starting with a
  symbol ($c$, $K$) stays as is.
- **Schematics of our system name no other system:** name the concept ("timeout", "steep discount"), not the baseline.
- **No titles** inside the plot. The caption says what it is; panel identity goes in the caption's (a) / (b) or a short in-axes tag.
- **System name is a parameter.** Every label passes through one `label()` / `SYS_NAME`; renaming the system is a one-line change.
- **Each plot has one message.** Write it down first ("ours hits the target at 95 h; no baseline reaches it"); every design choice
  below serves it.
- **Annotate the message on the plot** so the reader doesn't decode axes. Two to four annotations per panel; numbers match the
  paper's tables and are computed, never typed in.
- **Call out comparisons, not raw values.** Show ratios and deltas of ours vs the baselines ("1.4× faster than FedBuff",
  "2.8× fewer GB", "+3.7%"), not "95 h" or "2.3k GB", which leave the reader to do the division. Draw the comparison: a horizontal
  arrow from ours to the baseline at the same accuracy level (time / bytes to reach the baseline's peak, or the target). Compare
  against the strongest baseline, plus the best of each other family (e.g. best sync stack).
- **Fair comparisons, in the order of the success metrics** (e.g. accuracy, then time, then bytes):
  - Baseline reaches the target: report time / bytes to the target (speedup, bytes ratio).
  - Baseline misses the target: a time or bytes ratio at the target is undefined, so compare at the *same accuracy level* (time /
    bytes for ours to reach the baseline's peak vs the baseline's own time to it) **and** report the accuracy gained
    (+% over the baseline's peak), so the reader sees both how much sooner and how much better.
  - Compare only like with like: no bytes ratio against stacks that use fewer bytes but never reach the accuracy (that isn't a fair
    win or a fair loss; leave it out of the plot, discuss in text).
  - Highlight the best numbers that are also fair and interesting to a reviewer: the strongest baseline (always), plus the best
    of each other family (e.g. best sync stack). Never a weak baseline just because the ratio is large.
- **Callouts are black, never per-baseline colors** (a plot with 6 method colors is already colorful enough). Emphasize with a
  **bold number** and plain words ("**1.4×** faster, **+1.8%** vs FedBuff"); a box or italics only if a callout must stand
  apart from other text. Comparison arrows black, thin.
- **No text may overlap any plot element**: data lines, markers, arrows, other text, the axes edge. Not "barely", not "mostly
  readable". Enforce it in code: place callouts with an auto-placer that tests candidate spots against every element, and run an
  overlap check on save that prints every collision; a figure with a reported collision is not done.
- **Place callouts in empty regions** (corners the curves leave free). If no region is free, change the plot: move the callout,
  shorten it, or extend the axis to make a strip for annotations (e.g. a band above the target line where no curve goes).
  Place them after the layout is final (legend added, limits set), and freeze the layout so later draws don't move the axes.
- **Give the real estate to the interesting part.** Zoom the axis to where the comparison happens (e.g. y from 60%, not 0%, when the
  race is for 75%: the early climb is uninteresting); use log axes when values span decades; cut long flat tails. Keep zooming
  until the empty area left is only what annotations use.
- **Axis breaks and cropped axes** (decided):
  - *Line / scatter plots* (position encodes value): a y-axis not starting at 0 needs **no break mark**; the tick labels show the
    range, and this is standard practice. Crop freely to the interesting range.
  - *Bar charts* (length encodes value): the value axis **must start at 0**; a cropped bar exaggerates differences. If one bar
    dwarfs the rest, use a log axis, or a broken axis with explicit break marks (`//` on the axis and the bar) and a gap.
  - *Skipping a middle range* of an axis (data on both sides, nothing interesting between): broken axis with `//` marks and a
    visible gap, never a silent jump.
  - *Data cut at the edge* (curves start below the y minimum): no break mark; state the crop in the caption if a reader could
    miss it.
- **No unexplained whitespace.** An empty region is either removed (tighter limits, smaller figure, different chart form) or used
  (legend, key annotation). If a plot is mostly empty and there is nothing to say there, change the plot.
- **Declutter.** No top / right spines; light grid below the data; few ticks at meaningful values (24 h steps for days).
- **Smooth noisy curves for display** with a centered moving average wide enough to remove eval-to-eval jitter but keep the trend
  and plateaus (e.g. about 5 h of 10-min evals). Smoothing is for display only: every number (hit time, peak, ratio) comes from the
  metric's own definition (e.g. the 5-eval trailing mean). Say "smoothed" in the caption.
- **Ours stands out, baselines recede.** Ours: saturated hero color, thickest line, drawn on top. Strongest baseline: its own color,
  normal width. Weak / grouped baselines: thinner. Oracle / idealized / async variants: their own color, dashed.
- **One color per method across the whole paper**, from a color-blind-safe palette; never reuse a method's color for something
  else. **Every line in a plot gets a different color**: never tell two series apart by line style alone.
- **Colors must be clearly distinct from each other**, one hue family each: no two of orange / vermillion / pink / yellow /
  red together, no two blues of similar lightness. Check side by side, also in grayscale and under CVD simulation.
- **Black stays black.** Axis lines, ticks, tick labels, axis labels, legend text, callouts and reference lines (e.g. the target)
  are black. **Grey only for what must not draw the eye** (grid, a de-emphasized context series); never suppress blacks into
  greys, and no method is drawn in grey.
- **Grayscale-print safe:** colors alone must not carry identity. Give each method a marker (sparse, `markevery`; ours filled,
  baselines hollow) and vary line width (ours thickest). Variants: own color, parent's marker, dashed. Check by
  printing (or converting) to grayscale.
- **Legend:** outside the axes in one row above the panels when it fits, else in a genuinely empty region; order = ours first, then
  as the reader should compare. Prefer direct labels on curves when there are ≤ 3 series.
- **Non-method categories** (e.g. causes of lost time, round components) never borrow a method color: use black, white +
  hatch (`\\`, `xx`) and one light fill, so they read in grayscale and don't clash with the method palette. The loss we want
  seen gets black; the uninteresting part gets the light fill.
- **Bar-end labels** carry the comparison (ratio vs ours, or "Missed: peak %"); the x range widens until every label fits.
- **Panel identity** (dataset, trace) goes in a free in-axes spot, else into the axis label ("Time (hours)\nCIFAR-10, MobiPerf"),
  never a title.
- **Shared axes** across panels that compare the same quantity (`sharey`), so only the left panel carries the y label.
- **Units in axis labels**, in parentheses; state "log scale" for log axes. Time ticks in natural units ("≤1 s", "1 min",
  "1 h", "1 day"; 24 h steps for days), not raw seconds.
- **Ratios read one way:** "N× faster / fewer GB" with N ≥ 1 when ours wins; when ours loses, say so plainly (0.7× GB is shown,
  not hidden). Near 1 (within 5%), write "Same …" and the reason ("pool stays above c"), never "1.0× faster".
- **§-specific variants:** a plot used both in motivation (baselines only) and in design/eval gets two files (`<name>_base`
  without ours, `<name>` with ours), never one figure that previews the fix.
- **Honest data beats the outline.** If a figure contradicts a claim, draw it truthfully and fix the claim (or drop the figure);
  never pick the cell, window or metric that rescues it.

## 3. Picking the chart form

| message | form | notes |
|---|---|---|
| ours reaches a level sooner / cheaper | accuracy vs time and vs GB (log), two panels, `sharey` | y zoomed to the race; arrows at the baseline's peak |
| where a resource goes (share by cause) | horizontal 100%-style stacked bars of the *lost* part only, rows = methods | bar-end % label; ours bold |
| order-of-magnitude gap (s vs min) | dot plot on a log axis, rows = methods, one marker shape per cell | clamp 0 to the axis floor ("≤1 s") |
| ablation / swap arms | peak-accuracy dots first (cropped axis, Δ vs ours), then horizontal bars of time and GB; rows = arms, rows of panels = cells | misses hatched to run end; accuracy as dots because bars must start at 0 |
| one quantity across a few cells | grouped vertical bars, ours vs strongest baseline, ratio above each pair | value axis from 0 |
| a trend vs a ratio (e.g. choice vs c / pool) | scatter, one panel per cell, x range per panel | ours filled, other arms hollow |
| a delta per method and cell (oracle gain) | dot plot around a 0 line | x range from the data, never clipped |
| a time series vs a fixed setting | lines + a horizontal reference, callout with the % of time one exceeds the other | crop y; no twin axes |
| analytic rule comparison | single panel of curves | one callout at a meaningful point (e.g. "at τ = 10: 1.7×") |
| architecture | boxes + numbered arrows, step list *inside* the figure | ours' components edged in our color; no DRAFT stamp |

Avoid: twin y-axes (split into panels or drop the series), more than ~4 panels in a column figure, pie charts, 3-D, legends inside
crowded data, a panel per cell when one cell carries the message (put the rest in the appendix).

## 4. Process

- Settle one **demo figure** first (style, sizes, annotation density), iterate with the author until it is signed off, then apply
  it to the rest. Every new rule from that loop goes into this file immediately.
- Before rolling out, map every planned figure to: its claim, the data that backs it now, and whether a newer figure already
  covers it. Say explicitly what is kept, rebuilt, restyled or dropped, and why; never skip a figure silently.
- Placeholders (MOCK / old data) are rebuilt as soon as real runs exist; superseded files and their drawing code are deleted
  (git keeps them), so nothing stale can be regenerated or included by mistake.
- Every figure regenerates from data with one command; annotation numbers are computed, never typed.
- Before calling a figure done: the overlap check is silent; view the PNG at 100% *and* at print size; nothing is clipped; the
  PDF size equals the slot; fonts are Type 42.
- Status labels (MOCK / PRELIM / DRAFT) only on placeholder figures; final figures carry none.
- Report the figures that contradict the paper's claims to the author with the numbers, in the same message as the figures.

## 5. Implementation tricks (matplotlib)

- **Exact size:** `figure.constrained_layout.use = True` with small pads, `savefig.bbox = None`, `savefig.pad_inches` small;
  outside legends via `fig.legend(loc="outside upper center")`; shared axis labels via `fig.supxlabel`. `subplots_adjust` /
  `tight_layout` conflict with it: don't mix.
- **Collision test** (`overlaps`): per in-axes text, test its window bbox (padded ~3% / 8%) against: each line as a path
  (`Path.intersects_bbox(..., filled=False)`, padded by half the line width; sampling points misses segments), markers only
  where `markevery` draws them (padded by marker size), bars / boxes (except a label fully inside its own box, and background
  regions tagged `gid="region"`), arrow paths (vertices, not bboxes), other texts, and the axes edge (unpadded).
- **Auto-placer** (`place`): try the author's candidate spots in order, then a 40 × 40 grid of axes positions nearest the first
  spot; warn if nothing is free. Call it **after** the legend, limits and scales are final, and freeze the layout first
  (`fig.set_layout_engine("none")` after a draw): constrained layout re-runs on every draw and moves axes under placed text.
- **Mathtext bold numbers** (`$\mathbf{1.4\times}$`) give bold-number / plain-word callouts in the body font, but their bboxes
  are taller than plain text: leave a little more line spacing.
- **Bar-end labels:** draw them, measure their extent, widen `xlim` until they fit (a few iterations), instead of guessing a margin.
- **Smoothing:** centered moving average with a shrinking window at the ends; numbers from the metric's own trailing mean.
- **Shared references** (target line, zero line) in black, dashed / thin, `zorder` below data; grid below everything
  (`axes.axisbelow`).
- **Check output in code:** PDF `MediaBox` = figsize, `FontFile2` present, no `/Type3`.

## 6. This project (FluxTune)

- Master config: `examples/paper_expts_fluxtune/plots/ft_style.py` (`SYS_NAME`, `label()`, sizes, palette, rcParams, `place` /
  `overlaps` / `save`). The older `fwdllm/expt_scripts/plotlib/style.py` predates these rules.
- Paper uses **FluxTune** in every plot (`SYS_NAME = "FluxTune"`); internal keys stay `"fluxtune"`.
- Palette (Paul Tol, color-blind safe, no hue shared with Felix's Okabe-Ito blue / vermillion / green), one per stack; each
  variant gets its own color too (no two lines in a plot share a color):
  - FluxTune wine `#882255`, `D` filled (hero, 1.8 pt, on top)
  - FwdLLM teal `#44AA99`, `o` hollow
  - FedBuff indigo `#332288`, `s` hollow
  - Felix (as a baseline) ochre `#DDAA33`, `^` hollow
  - FwdLLM async orange `#EE7733`, `o` hollow, dash-dot
  - `_it_unaware` olive `#999933`, parent's marker, dash-dot; `_oracular` cyan `#88CCEE`, dashed, `*` marker.
  - Ablation / tuning arms (`ARM`): green / blue / olive / cyan; don't plot arms next to a variant of the same color.
- Text placement: `ft_style.place(ax, text, spots)`; `ft_style.overlaps(fig)` runs in every `save()` and prints `OVERLAP …`.
- Data figures: `plots/make_ft_figs.py` (run registry `RUNS`, cells `CELLS`) → `paper_expts_fluxtune/figs/`; inputs are
  per-run digests (`plots/ft_digest.py`, stdlib only: digest on the node that holds the run, copy the JSON back).
- Non-method fills: black (the loss to see), white + `\\` / `xx` hatches, warm sand `#ebe3d0` (the uninteresting part).
