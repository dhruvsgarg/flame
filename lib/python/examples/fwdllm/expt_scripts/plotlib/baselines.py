# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Baseline display registry — the single place paper names, colors, markers and
legend emphasis live (telemetry keys stay internal: fwdllm/fwdllm_plus/fluxtune/…).

Families carry identity by hue (FwdLLM=orange, our system=blue — the colorblind-safe
pair); members within a family differ by a lighter validated tint + linestyle +
marker (grayscale/CVD-safe, not hue alone). Head-to-head systems are legend-bold,
FwdLLM++ (a derived config) regular-italic. Add an ablation = one BASELINES row;
unknown keys fall back by family prefix. Rename our system via SYSTEM_LABEL.
"""

from __future__ import annotations

from dataclasses import dataclass


# ---- the switchable codename for OUR system ------------------------------- #
# logs still say "fluxtune"; only the printed/plotted label changes here.
SYSTEM_LABEL = "FluxTune"


# family root hue + one validated lighter tint (index 0 = root). Roots+tint pass
# dataviz/validate_palette.js (lightness/chroma/CVD>=12/contrast); a 3rd goes gray.
_FAMILY_TINTS = {
    "fwdllm":   ["#D55E00", "#E8853A"],   # orange
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


def _mk(key, family, tint, linestyle, emphasis, order, label, marker="o") -> Style:
    return Style(key=key, label=label, color=_tint(family, tint),
                 linestyle=linestyle, marker=marker, emphasis=emphasis, order=order)


# --------------------------------------------------------------------------- #
# THE registry — one row per baseline/ablation
# --------------------------------------------------------------------------- #
# Grayscale/CVD-safe: orange & blue sit at near-identical grayscale luminance, so
# each baseline also gets a distinct linestyle + marker (identity survives B&W).
BASELINES: dict[str, Style] = {
    "fwdllm":      _mk("fwdllm",      "fwdllm",   0, "-",  "bold",   0, "FwdLLM",   marker="o"),
    "fwdllm_plus": _mk("fwdllm_plus", "fwdllm",   1, "--", "italic", 1, "FwdLLM++", marker="s"),
    "fluxtune":    _mk("fluxtune",    "fluxtune", 0, "-",  "bold",   2, SYSTEM_LABEL, marker="^"),
    # ablations (uncomment when the runs exist):
    # "fluxtune_nojvp": _mk("fluxtune_nojvp", "fluxtune", 1, "--", "regular", 3,
    #                       f"{SYSTEM_LABEL}−JVP", marker="v"),
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
             "fwdllm" if key.startswith("fwdllm") else "fluxtune"
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
