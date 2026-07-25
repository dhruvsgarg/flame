# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Baseline display registry — the single place paper names, colors, markers and
legend emphasis live (telemetry keys stay internal: fwdllm/fwdllm_it_oracular/
fedbuff_round/felix_round/fluxtune/…).

Four orthogonal visual channels (revised 2026-07-25, superseding the 2026-07-24
cadence-color scheme — that scheme made round-level baselines share a hue purely
because they're round, so three *unrelated* published baselines (FwdLLM,
FedBuff(P), Felix(P)) looked like one family while each substrate's own
round/+IT/+IT+O siblings, which ARE one family, were scattered across two
different hues. Substrate lineage is the more important thing for color to
carry — round vs iteration now rides linestyle instead), each carrying exactly
one signal so a reader learns one rule per channel instead of decoding a
combined code:

1. **Hue family = substrate lineage.** Each published-baseline family gets its
   own dedicated hue across ALL its variants: fwdllm-lineage (fwdllm,
   fwdllm_it_unaware, fwdllm_it_oracular) = vermillion, fedbuff-lineage
   (fedbuff_round, fedbuff_it_unaware, fedbuff_it_oracular) = bluish-green,
   felix-lineage (felix_round, felix_it) = orange, FluxTune (+ its own R1-R4 opt
   ladder) = its dedicated blue. A reader immediately sees "these three curves
   are the same underlying selection strategy at different cadences" instead of
   mistaking cross-substrate round-level siblings for a family.
2. **Shade within a hue family = derivation distance from the true baseline —
   intuition: lighter means "further from what that baseline actually is."**
   For fwdllm/fedbuff/felix, the round-level `(P)`/native row IS that published
   baseline; `+IT` and `+IT+O` graft our own cadence/tracking mechanism onto
   it, moving it away from its native published form even when the change
   happens to help accuracy — so darkest = the true/native baseline, lighter =
   `+IT` (one derivation step away), lightest = `+IT+O` (a further step: +
   oracular availability tracking); same lightening fraction applied at each
   step across every substrate, so the *amount* of lightening itself signals
   "one step removed" vs "two steps removed" consistently. FluxTune's own
   R1-R4 opt-ablation ramp runs the SAME intuition in the opposite visual
   direction because it's built the other way round: R1-base is a stripped-down
   starting point and R4-full (= FluxTune's true, native form) is reached by
   *adding* opts, so it starts lightest and ends darkest at the true baseline —
   same rule ("true baseline = darkest"), just walked forward instead of
   backward since these rows aren't a ported-baseline derivation chain, they're
   FluxTune's own build-up. FluxTune's headline color and its R4 "(full)"
   ladder color are the literal same hex (`FLUXTUNE_COLOR == _ABLATION_BLUES[-1]`)
   — they're the same run/system, not two things, even though they appear in
   different figures.
3. **Linestyle = control cadence** (the paper's central axis): solid = round,
   dashed = `+IT`, dash-dot = `+IT+O` — consistent across every substrate, so
   "which lines are dashed" answers "which baselines are iteration-level"
   regardless of color. Paired with legend bold (✅ EVAL, always round-level) vs
   regular-italic (⚠ ABLATION, always the `+IT`/`+IT+O` rows) as a second,
   independent cadence cue.
4. **Marker shape = substrate lineage, redundant with hue** (not a second
   signal — deliberately doubled for CVD/grayscale safety): fwdllm-lineage = o,
   fedbuff-lineage = s, felix-lineage = D, FluxTune = ^.
5. **Marker fill = sync vs async**: hollow (white face, colored edge) = sync,
   solid-filled (colored face, white edge) = async; bars get a `///` hatch for
   sync, plain fill for async. Sync/async happens to be redundant with substrate
   today (fwdllm-lineage is always sync, everything else always async per
   BASELINES.md) but gets its own channel anyway — the fill/hatch makes that
   split visible without requiring the reader to have memorized marker shapes.

**Accessibility rule — hard requirement, must stay true through future edits.**
Color is a convenience for readers with typical color vision; it is never the
*only* carrier of a distinction that matters, because two audiences can't see
it: colorblind readers and anyone reading a black-and-white printout/photocopy.
Concretely:
- Every base hue is one of the 8 Okabe-Ito colors (CVD-safe under
  deuteranopia/protanopia/tritanopia by construction, not just "looks
  distinct" to a typical reader) — vermillion `#D55E00` (fwdllm), bluish-green
  `#009E73` (fedbuff), orange `#E69F00` (felix), blue `#0072B2`-family
  (FluxTune). Lightened tints (channel 2, ported/derived) stay in-family, never
  cross a hue boundary.
- Marker shape (substrate) and linestyle (cadence) together must uniquely
  identify every baseline **within any single figure's `runs:` manifest** even
  with color removed entirely — checked against every manifest in
  `expt_scripts/figs*.yaml`, not just checked in the abstract, since two
  baselines that never share a figure are allowed to share a (marker,
  linestyle) pair. Verify this whenever a new baseline or manifest is added.
- Marker fill (hollow/filled) and bar hatch (`///`/plain) are patterns, not
  colors, so sync/async survives grayscale/B&W by construction — no separate
  check needed there.
- When adding a hue or shade, sanity-check its converted grayscale luminance
  isn't identical to a same-manifest sibling's (a fast eyeball check on a
  grayscale-printed/desaturated render is enough — exact luminance match is
  a soft signal, marker+linestyle are what actually guarantee legibility).

Unknown keys fall back by substrate-name prefix, or `fluxtune`'s opt-ladder
blue. Rename our system via SYSTEM_LABEL. Display-name grammar (`(P)`/`+IT`/
`+O`) is display-only — see BASELINES.md "Naming grammar" for what each suffix
means.
"""

from __future__ import annotations

from dataclasses import dataclass


# ---- the switchable codename for OUR system ------------------------------- #
# logs still say "fluxtune"; only the printed/plotted label changes here.
SYSTEM_LABEL = "FluxTune"


# ---- per-substrate hue ramps (the primary color signal) --------------------- #
# One dedicated Okabe-Ito hue per published-baseline family, darkest->lightest =
# round-level anchor -> +IT -> +IT+O (each step lightens by the same fraction,
# so the amount of tinting itself reads as "how many derivation steps away").
# fwdllm-lineage: vermillion (Okabe-Ito #D55E00).
_FWDLLM_HUES = ["#D55E00", "#E29252", "#EDBB94"]      # fwdllm, +IT, +IT+O
# fedbuff-lineage: bluish-green (Okabe-Ito #009E73).
_FEDBUFF_HUES = ["#009E73", "#52BDA0", "#94D6C4"]     # fedbuff_round, +IT, +IT+O
# felix-lineage: orange (Okabe-Ito #E69F00) — no +IT+O twin, just the 2 steps.
_FELIX_HUES = ["#E69F00", "#EEBE52"]                  # felix_round, +IT
# Sequential CVD-safe blue ramp (ColorBrewer "Blues"), light→dark encodes
# FluxTune's own opt-ablation ladder: more learning-affecting opts enabled ⇒
# darker. Ordered, not categorical — the four fluxtune configs form a 2×2 over
# Opt-2/Opt-3.
_ABLATION_BLUES = ["#9ecae1", "#4292c6", "#2171b5", "#08306b"]
# FluxTune: continuous/async reselect on its own selection+agg — not a ported
# baseline with round/+IT/+IT+O siblings, so it gets its own dedicated blue
# rather than joining a substrate ramp above. Literally the opt-ladder ramp's
# darkest step, not just a matching hue -- FluxTune's headline curve and its R4
# "(full)" opt-ablation curve are the same run/system, so they render as the
# exact same color across figures, not two close-but-different blues.
FLUXTUNE_COLOR = _ABLATION_BLUES[-1]

# emphasis -> (fontweight, fontstyle) for the legend text
_EMPHASIS = {
    "bold":   ("bold", "normal"),
    "regular": ("normal", "normal"),
    "italic": ("normal", "italic"),
}


@dataclass(frozen=True)
class Style:
    key: str
    label: str
    color: str
    linestyle: str
    marker: str
    emphasis: str       # key into _EMPHASIS
    order: int          # legend / z-order (lower plots first, listed first)
    sync: bool          # True = sync (hollow marker) / False = async (filled)

    def legend_font(self):
        return _EMPHASIS.get(self.emphasis, _EMPHASIS["regular"])

    def marker_fill_kwargs(self) -> dict:
        """kwargs for ax.plot(...) marker fill: hollow = sync, filled = async."""
        if self.sync:
            return dict(markerfacecolor="white", markeredgecolor=self.color,
                        markeredgewidth=1.1)
        return dict(markerfacecolor=self.color, markeredgecolor="white",
                    markeredgewidth=0.5)

    def scatter_fill_kwargs(self) -> dict:
        """kwargs for ax.scatter(...) marker fill: hollow = sync, filled = async."""
        if self.sync:
            return dict(facecolor="white", edgecolor=self.color, linewidth=1.1)
        return dict(facecolor=self.color, edgecolor="white", linewidth=0.4)

    def bar_hatch(self):
        """Bar hatch pattern: '///' = sync, None (plain fill) = async."""
        return "///" if self.sync else None


def _mk(key, color, linestyle, emphasis, order, label, marker="o", sync=False) -> Style:
    return Style(key=key, label=label, color=color, linestyle=linestyle,
                 marker=marker, emphasis=emphasis, order=order, sync=sync)


# --------------------------------------------------------------------------- #
# THE registry — one row per baseline/ablation
# --------------------------------------------------------------------------- #
# Marker encodes SUBSTRATE (redundant with hue, deliberately doubled for
# CVD/grayscale safety): fwdllm-lineage = o, fedbuff-lineage = s, felix-lineage
# = D. Linestyle carries cadence (solid = round, dashed = +IT, dash-dot =
# +IT+O), consistent across every substrate.
BASELINES: dict[str, Style] = {
    # ---- fwdllm-lineage (vermillion) -------------------------------------------
    "fwdllm":        _mk("fwdllm",        _FWDLLM_HUES[0], "-", "bold", 0,
                          "FwdLLM",     marker="o", sync=True),
    "fwdllm_it_unaware":   _mk("fwdllm_it_unaware",   _FWDLLM_HUES[1], "--", "italic", 1,
                                "FwdLLM+IT",       marker="o", sync=True),
    "fwdllm_it_oracular":  _mk("fwdllm_it_oracular",  _FWDLLM_HUES[2], "-.", "italic", 2,
                                "FwdLLM+IT+O",     marker="o", sync=True),

    # ---- fedbuff-lineage (bluish-green) ----------------------------------------
    "fedbuff_round": _mk("fedbuff_round", _FEDBUFF_HUES[0], "-", "bold", 3,
                          "FedBuff(P)", marker="s", sync=False),
    "fedbuff_it_unaware":  _mk("fedbuff_it_unaware",  _FEDBUFF_HUES[1], "--", "italic", 4,
                                "FedBuff(P)+IT",   marker="s", sync=False),
    "fedbuff_it_oracular": _mk("fedbuff_it_oracular", _FEDBUFF_HUES[2], "-.", "italic", 5,
                                "FedBuff(P)+IT+O", marker="s", sync=False),

    # ---- felix-lineage (orange) -------------------------------------------------
    "felix_round":   _mk("felix_round",   _FELIX_HUES[0], "-", "bold", 6,
                          "Felix(P)",   marker="D", sync=False),
    "felix_it":            _mk("felix_it",            _FELIX_HUES[1], "--", "italic", 7,
                                "Felix(P)+IT",     marker="D", sync=False),

    # ---- FluxTune: our system (✅ EVAL, headline) ------------------------------
    "fluxtune": _mk("fluxtune", FLUXTUNE_COLOR, "-", "bold", 8, SYSTEM_LABEL,
                    marker="^", sync=False),
    # ablations (uncomment when the runs exist):
    # "fluxtune_nojvp": _mk("fluxtune_nojvp", "#4A97CC", "--", "regular", 9,
    #                       f"{SYSTEM_LABEL}−JVP", marker="v", sync=False),

    # ---- 2×2 opt ablation (N=100, α=1) ---------------------------------------
    # Constant across all four: C1 guided JVP perturbations (trainer-side
    # select_perturbation_using_jvp=true) + Opt-1 weight-suppression. Varies:
    # Opt-2 var-stop, Opt-3 grad-aware. R1 is the FluxTune base (forward-mode LLM
    # perturbation fine-tuning; borrows only FeLiX's scalar aggregation rate
    # `type=new`); R4 == the full FluxTune default ("fluxtune" above). Blue ramp +
    # distinct marker/linestyle so identity survives grayscale/CVD.
    "fluxtune_r1_base":      _mk("fluxtune_r1_base",      _ABLATION_BLUES[0], ":",  "regular", 10,
                                 f"{SYSTEM_LABEL}-base", marker="o", sync=False),
    "fluxtune_r2_varstop":   _mk("fluxtune_r2_varstop",   _ABLATION_BLUES[1], "-.", "regular", 11,
                                 "+ var-stop",      marker="s", sync=False),
    "fluxtune_r3_gradaware": _mk("fluxtune_r3_gradaware", _ABLATION_BLUES[2], "--", "regular", 12,
                                 "+ grad-aware",    marker="^", sync=False),
    "fluxtune_r4_full":      _mk("fluxtune_r4_full",      _ABLATION_BLUES[3], "-",  "bold",    13,
                                 f"{SYSTEM_LABEL} (full)", marker="D", sync=False),
}

# linestyles + markers cycled for un-registered ablations of the same cadence
_FALLBACK_LS = ["--", ":", "-."]
_FALLBACK_MARKERS = ["v", "D", "P", "X"]


def style_for(key: str) -> Style:
    """Resolve a baseline key to a Style, with a safe fallback for unknown keys."""
    if key in BASELINES:
        return BASELINES[key]
    # fallback: resolve to a substrate hue family by name (fluxtune-prefixed ->
    # its blue opt-ladder; felix/fedbuff -> their family; else -> fwdllm family,
    # which also covers legacy keys like the pre-rename `fwdllm_plus`). Lightest
    # ramp step, cycled linestyle/marker so repeated fallbacks stay
    # distinguishable from each other and from the registered rows.
    if key.startswith("fluxtune"):
        ramp, label = _ABLATION_BLUES, SYSTEM_LABEL + key[len("fluxtune"):].replace("_", " ")
    elif "felix" in key:
        ramp, label = _FELIX_HUES, key
    elif "fedbuff" in key:
        ramp, label = _FEDBUFF_HUES, key
    else:
        ramp, label = _FWDLLM_HUES, key
    # sync/async fallback mirrors BASELINES.md: fwdllm-lineage is the only sync
    # family, everything else (fedbuff/felix/fluxtune) is async.
    sync = ramp is _FWDLLM_HUES
    n_sibling = sum(1 for s in BASELINES.values() if s.color in ramp)
    return _mk(key, ramp[-1], _FALLBACK_LS[n_sibling % len(_FALLBACK_LS)],
               "regular", 100 + len(BASELINES), label,
               marker=_FALLBACK_MARKERS[n_sibling % len(_FALLBACK_MARKERS)], sync=sync)


def ordered(keys) -> list[str]:
    """Keys sorted by registry order (stable legend/plot order across figures)."""
    return sorted(keys, key=lambda k: style_for(k).order)


def apply_legend_emphasis(legend):
    """Set per-entry fontweight/style from the registry (bold vs regular-italic)."""
    for txt in legend.get_texts():
        label = txt.get_text()
        st = next((s for s in BASELINES.values() if s.label == label), None)
        if st is not None:
            weight, style = st.legend_font()
            txt.set_fontweight(weight)
            txt.set_fontstyle(style)
