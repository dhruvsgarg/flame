# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Baseline display registry — the single place paper names, colors, markers and
legend emphasis live (telemetry keys stay internal: fwdllm/fwdllm_it_oracular/
fedbuff_round/felix_round/fluxtune/…).

Color encodes CONTROL CADENCE, not substrate (decided 2026-07-24, superseding the
prior per-substrate-hue scheme): round-level baselines (fwdllm, fedbuff_round,
felix_round) share a vermillion ramp, iteration-level baselines (the five `+IT`/
`+IT+O` rows) share a bluish-green ramp, and FluxTune gets its own dedicated blue —
this makes the paper's central round→iteration axis (BASELINES.md "why round→
iteration matters more here") the first thing a reader's eye groups on. Substrate
identity (fwdllm vs fedbuff vs felix) still survives via a per-substrate marker
(o/s/D) that's consistent across both cadence ramps, plus linestyle. All hues
Okabe-Ito-derived / CVD-safe. ✅ EVAL baselines (BASELINES.md status column) are
legend-bold; ⚠ ABLATION rows (the `+IT`/`+IT+O` cadence variants) are
regular-italic. Unknown keys fall back by cadence (name contains `_it`) or family
prefix (fluxtune). Rename our system via SYSTEM_LABEL. Display-name grammar
(`(P)`/`+IT`/`+O`) is display-only — see BASELINES.md "Naming grammar" for what
each suffix means.
"""

from __future__ import annotations

from dataclasses import dataclass


# ---- the switchable codename for OUR system ------------------------------- #
# logs still say "fluxtune"; only the printed/plotted label changes here.
SYSTEM_LABEL = "FluxTune"


# ---- cadence ramps (the primary color signal) ------------------------------ #
# Round-level: whole-databin commit, reselect only at round boundary (fwdllm,
# fedbuff_round, felix_round). Vermillion (Okabe-Ito #D55E00), dark->light in
# BASELINES.md table order.
_ROUND_HUES = ["#D55E00", "#E8853A", "#F4B183"]
# Iteration-level: the `+IT`/`+IT+O` rows — per-iteration reselect/commit cadence.
# Bluish-green (Okabe-Ito #009E73 family), dark->light in BASELINES.md table order
# (fwdllm_it_unaware, fwdllm_it_oracular, fedbuff_it_unaware, fedbuff_it_oracular,
# felix_it).
_ITERATION_HUES = ["#00441B", "#1B7837", "#238B45", "#41AE76", "#66C2A4"]
# FluxTune: continuous/async reselect — neither round nor per-iteration cadence,
# so it gets its own hue rather than joining either ramp. Existing validated blue
# (Okabe-Ito #0072B2, ~dodgerblue) — kept identical to the opt-ladder ramp's
# darkest step (_ABLATION_BLUES[3] below) so FluxTune's headline curve and its R4
# opt-ablation curve read as the same color across figures.
FLUXTUNE_COLOR = "#0072B2"

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

    def legend_font(self):
        return _EMPHASIS.get(self.emphasis, _EMPHASIS["regular"])


def _mk(key, color, linestyle, emphasis, order, label, marker="o") -> Style:
    return Style(key=key, label=label, color=color, linestyle=linestyle,
                 marker=marker, emphasis=emphasis, order=order)


# Sequential CVD-safe blue ramp (ColorBrewer "Blues"), light→dark encodes the
# ablation ladder: more learning-affecting opts enabled ⇒ darker. Ordered, not
# categorical — the four fluxtune configs form a 2×2 over Opt-2/Opt-3.
_ABLATION_BLUES = ["#9ecae1", "#4292c6", "#2171b5", "#08306b"]


# --------------------------------------------------------------------------- #
# THE registry — one row per baseline/ablation
# --------------------------------------------------------------------------- #
# Marker encodes SUBSTRATE (consistent across both cadence ramps): fwdllm-lineage
# = o, fedbuff-lineage = s, felix-lineage = D. Linestyle is a redundant cadence
# cue (solid = round, dashed = +IT, dash-dot = +IT+O) so identity survives
# grayscale/CVD on two independent channels, not hue alone.
BASELINES: dict[str, Style] = {
    # ---- round-level (✅ EVAL anchors — vermillion ramp) -----------------------
    "fwdllm":        _mk("fwdllm",        _ROUND_HUES[0], "-", "bold", 0,
                          "FwdLLM",     marker="o"),
    "fedbuff_round": _mk("fedbuff_round", _ROUND_HUES[1], "-", "bold", 3,
                          "FedBuff(P)", marker="s"),
    "felix_round":   _mk("felix_round",   _ROUND_HUES[2], "-", "bold", 6,
                          "Felix(P)",   marker="D"),

    # ---- iteration-level (⚠ ABLATION — bluish-green ramp) ---------------------
    "fwdllm_it_unaware":   _mk("fwdllm_it_unaware",   _ITERATION_HUES[0], "--", "italic", 1,
                                "FwdLLM+IT",       marker="o"),
    "fwdllm_it_oracular":  _mk("fwdllm_it_oracular",  _ITERATION_HUES[1], "-.", "italic", 2,
                                "FwdLLM+IT+O",     marker="o"),
    "fedbuff_it_unaware":  _mk("fedbuff_it_unaware",  _ITERATION_HUES[2], "--", "italic", 4,
                                "FedBuff(P)+IT",   marker="s"),
    "fedbuff_it_oracular": _mk("fedbuff_it_oracular", _ITERATION_HUES[3], "-.", "italic", 5,
                                "FedBuff(P)+IT+O", marker="s"),
    "felix_it":            _mk("felix_it",            _ITERATION_HUES[4], "--", "italic", 7,
                                "Felix(P)+IT",     marker="D"),

    # ---- FluxTune: our system (✅ EVAL, headline) ------------------------------
    "fluxtune": _mk("fluxtune", FLUXTUNE_COLOR, "-", "bold", 8, SYSTEM_LABEL, marker="^"),
    # ablations (uncomment when the runs exist):
    # "fluxtune_nojvp": _mk("fluxtune_nojvp", "#4A97CC", "--", "regular", 9,
    #                       f"{SYSTEM_LABEL}−JVP", marker="v"),

    # ---- 2×2 opt ablation (N=100, α=1) ---------------------------------------
    # Constant across all four: C1 guided JVP perturbations (trainer-side
    # select_perturbation_using_jvp=true) + Opt-1 weight-suppression. Varies:
    # Opt-2 var-stop, Opt-3 grad-aware. R1 is the FluxTune base (forward-mode LLM
    # perturbation fine-tuning; borrows only FeLiX's scalar aggregation rate
    # `type=new`); R4 == the full FluxTune default ("fluxtune" above). Blue ramp +
    # distinct marker/linestyle so identity survives grayscale/CVD.
    "fluxtune_r1_base":      _mk("fluxtune_r1_base",      _ABLATION_BLUES[0], ":",  "regular", 10,
                                 f"{SYSTEM_LABEL}-base", marker="o"),
    "fluxtune_r2_varstop":   _mk("fluxtune_r2_varstop",   _ABLATION_BLUES[1], "-.", "regular", 11,
                                 "+ var-stop",      marker="s"),
    "fluxtune_r3_gradaware": _mk("fluxtune_r3_gradaware", _ABLATION_BLUES[2], "--", "regular", 12,
                                 "+ grad-aware",    marker="^"),
    "fluxtune_r4_full":      _mk("fluxtune_r4_full",      _ABLATION_BLUES[3], "-",  "bold",    13,
                                 f"{SYSTEM_LABEL} (full)", marker="D"),
}

# linestyles + markers cycled for un-registered ablations of the same cadence
_FALLBACK_LS = ["--", ":", "-."]
_FALLBACK_MARKERS = ["v", "D", "P", "X"]


def style_for(key: str) -> Style:
    """Resolve a baseline key to a Style, with a safe fallback for unknown keys."""
    if key in BASELINES:
        return BASELINES[key]
    # fallback: fluxtune-prefixed -> blue family; "_it" in the name -> iteration
    # (green) ramp; else -> round (vermillion) ramp. Lightest ramp step, cycled
    # linestyle/marker so repeated fallbacks stay distinguishable.
    if key.startswith("fluxtune"):
        ramp, label = _ABLATION_BLUES, SYSTEM_LABEL + key[len("fluxtune"):].replace("_", " ")
    elif "_it" in key:
        ramp, label = _ITERATION_HUES, key
    else:
        ramp, label = _ROUND_HUES, key
    n_sibling = sum(1 for s in BASELINES.values() if s.color in ramp)
    return _mk(key, ramp[-1], _FALLBACK_LS[n_sibling % len(_FALLBACK_LS)],
               "regular", 100 + len(BASELINES), label,
               marker=_FALLBACK_MARKERS[n_sibling % len(_FALLBACK_MARKERS)])


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
