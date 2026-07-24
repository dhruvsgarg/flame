# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Baseline display registry — the single place paper names, colors, markers and
legend emphasis live (telemetry keys stay internal: fwdllm/fwdllm_it_oracular/
fedbuff_round/felix_round/fluxtune/…).

Families carry identity by hue (FwdLLM=orange, FedBuff(P)=green, Felix(P)=pink,
our system=blue — all four colorblind-safe); members within a family differ by a
lighter validated tint + linestyle + marker (grayscale/CVD-safe, not hue alone).
✅ EVAL baselines (BASELINES.md status column) are legend-bold; ⚠ ABLATION rows
(the `+IT`/`+IT+O` cadence variants) are regular-italic. Add an ablation = one
BASELINES row; unknown keys fall back by family prefix. Rename our system via
SYSTEM_LABEL. Display-name grammar (`(P)`/`+IT`/`+O`) is display-only — see
BASELINES.md "Naming grammar" for what each suffix means.
"""

from __future__ import annotations

from dataclasses import dataclass


# ---- the switchable codename for OUR system ------------------------------- #
# logs still say "fluxtune"; only the printed/plotted label changes here.
SYSTEM_LABEL = "FluxTune"


# family root hue + one validated lighter tint (index 0 = root). Roots+tint pass
# dataviz/validate_palette.js (lightness/chroma/CVD>=12/contrast); a 3rd goes gray.
# Okabe-Ito CVD-safe categorical hues, one per baseline family (BASELINES.md's
# 9-baseline "FwdLLM / FedBuff(P) / Felix(P) / FluxTune" grouping).
_FAMILY_TINTS = {
    "fwdllm":   ["#D55E00", "#E8853A"],   # orange
    "fedbuff":  ["#009E73", "#66C2A5"],   # green (FedBuff(P) family)
    "felix":    ["#CC79A7", "#E8A5C7"],   # reddish-purple (Felix(P) family)
    "fluxtune": ["#0072B2", "#4A97CC"],   # blue (our system + ablations)
}

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


def _tint(family: str, level: int) -> str:
    ramp = _FAMILY_TINTS[family]
    return ramp[min(level, len(ramp) - 1)]


def _mk(key, family, tint, linestyle, emphasis, order, label, marker="o",
        color=None) -> Style:
    # `color` overrides the family tint ramp — used by the fluxtune ablation, whose
    # four variants need a dedicated sequential-blue ramp (light→dark = more opts on),
    # not the two categorical family tints.
    return Style(key=key, label=label, color=(color or _tint(family, tint)),
                 linestyle=linestyle, marker=marker, emphasis=emphasis, order=order)


# Sequential CVD-safe blue ramp (ColorBrewer "Blues"), light→dark encodes the
# ablation ladder: more learning-affecting opts enabled ⇒ darker. Ordered, not
# categorical — the four fluxtune configs form a 2×2 over Opt-2/Opt-3.
_ABLATION_BLUES = ["#9ecae1", "#4292c6", "#2171b5", "#08306b"]


# --------------------------------------------------------------------------- #
# THE registry — one row per baseline/ablation
# --------------------------------------------------------------------------- #
# Grayscale/CVD-safe: orange & blue sit at near-identical grayscale luminance, so
# each baseline also gets a distinct linestyle + marker (identity survives B&W).
BASELINES: dict[str, Style] = {
    # ---- FwdLLM family: sync, random select (BASELINES.md ✅ FwdLLM row + its
    # ⚠ +IT/+IT+O ablations) ----------------------------------------------------
    "fwdllm":             _mk("fwdllm",             "fwdllm", 0, "-",  "bold",   0,
                               "FwdLLM",       marker="o"),
    "fwdllm_it_unaware":  _mk("fwdllm_it_unaware",  "fwdllm", 1, "--", "italic", 1,
                               "FwdLLM+IT",    marker="s"),
    "fwdllm_it_oracular": _mk("fwdllm_it_oracular", "fwdllm", 1, "-.", "italic", 2,
                               "FwdLLM+IT+O",  marker="D"),

    # ---- FedBuff(P) family: async, random select, ported (✅ EVAL + its ⚠
    # ablations — fedbuff_it_unaware is the L2 attribution floor) --------------
    "fedbuff_round":       _mk("fedbuff_round",       "fedbuff", 0, "-",  "bold",   3,
                                "FedBuff(P)",      marker="o"),
    "fedbuff_it_unaware":  _mk("fedbuff_it_unaware",  "fedbuff", 1, "--", "italic", 4,
                                "FedBuff(P)+IT",   marker="s"),
    "fedbuff_it_oracular": _mk("fedbuff_it_oracular", "fedbuff", 1, "-.", "italic", 5,
                                "FedBuff(P)+IT+O", marker="D"),

    # ---- Felix(P) family: async, oort-smart select, ported (✅ EVAL + its ⚠
    # +IT ablation) --------------------------------------------------------------
    "felix_round": _mk("felix_round", "felix", 0, "-",  "bold",   6,
                        "Felix(P)",    marker="o"),
    "felix_it":    _mk("felix_it",    "felix", 1, "--", "italic", 7,
                        "Felix(P)+IT", marker="s"),

    # ---- FluxTune: our system (✅ EVAL, headline) ------------------------------
    "fluxtune":    _mk("fluxtune",    "fluxtune", 0, "-",  "bold",   8, SYSTEM_LABEL, marker="^"),
    # ablations (uncomment when the runs exist):
    # "fluxtune_nojvp": _mk("fluxtune_nojvp", "fluxtune", 1, "--", "regular", 9,
    #                       f"{SYSTEM_LABEL}−JVP", marker="v"),

    # ---- 2×2 opt ablation (N=100, α=1) ---------------------------------------
    # Constant across all four: C1 guided JVP perturbations (trainer-side
    # select_perturbation_using_jvp=true) + Opt-1 weight-suppression. Varies:
    # Opt-2 var-stop, Opt-3 grad-aware. R1 is the FluxTune base (forward-mode LLM
    # perturbation fine-tuning; borrows only FeLiX's scalar aggregation rate
    # `type=new`); R4 == the full FluxTune default ("fluxtune" above). Blue ramp +
    # distinct marker/linestyle so identity survives grayscale/CVD.
    "fluxtune_r1_base":      _mk("fluxtune_r1_base",      "fluxtune", 0, ":",  "regular", 10,
                                 f"{SYSTEM_LABEL}-base", marker="o", color=_ABLATION_BLUES[0]),
    "fluxtune_r2_varstop":   _mk("fluxtune_r2_varstop",   "fluxtune", 0, "-.", "regular", 11,
                                 "+ var-stop",      marker="s", color=_ABLATION_BLUES[1]),
    "fluxtune_r3_gradaware": _mk("fluxtune_r3_gradaware", "fluxtune", 0, "--", "regular", 12,
                                 "+ grad-aware",    marker="^", color=_ABLATION_BLUES[2]),
    "fluxtune_r4_full":      _mk("fluxtune_r4_full",      "fluxtune", 0, "-",  "bold",    13,
                                 f"{SYSTEM_LABEL} (full)", marker="D", color=_ABLATION_BLUES[3]),
}

# linestyles + markers cycled for un-registered ablations of the same family
_FALLBACK_LS = ["--", ":", "-."]
_FALLBACK_MARKERS = ["v", "D", "P", "X"]


def style_for(key: str) -> Style:
    """Resolve a baseline key to a Style, with a safe fallback for unknown keys."""
    if key in BASELINES:
        return BASELINES[key]
    # fallback: attach to a family by name prefix, lighter tint + cycled linestyle
    family = "fluxtune" if key.startswith("fluxtune") else \
             "fwdllm" if key.startswith("fwdllm") else \
             "fedbuff" if key.startswith("fedbuff") else \
             "felix" if key.startswith("felix") else "fluxtune"
    n_sibling = sum(1 for s in BASELINES.values()
                    if s.color in _FAMILY_TINTS[family])
    label = SYSTEM_LABEL + key[len("fluxtune"):].replace("_", " ") \
        if key.startswith("fluxtune") else key
    return _mk(key, family, 1, _FALLBACK_LS[n_sibling % len(_FALLBACK_LS)],
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
