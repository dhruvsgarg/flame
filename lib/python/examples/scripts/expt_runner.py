"""Shared experiment-launch display + pre-flight gate for the flame examples.

Generic and example-agnostic: it knows how to *render* a tiered hyperparameter
table and *gate* on a list of feasibility checks, but nothing about any specific
example's knobs. Each example's driver (fwdllm `run_sequential.sh`,
async_cifar10 `debug_run.sh`) builds a `spec` dict describing its own tiers and
checks and calls `render_and_gate(spec)`.

Design goal (why this exists): a live GPU/MQTT run is expensive, so before we
fire one we (a) show the operator the hyperparameters organized by how often
they change -- ① review-every-run, ② per-baseline, ③ config-baked -- and
(b) run correctness checks that block infeasible configs. Colour/callout markers
draw the eye to anything overridden or dangerous.

Spec schema
-----------
    spec = {
        "title":    str,                 # e.g. "FWDLLM RUN"
        "subtitle": str,                 # e.g. "mode=both  baselines=fwdllm ..."
        "dry_run":  bool,                # cosmetic tag in the header
        "tiers": [
            {"name": "① REVIEW EVERY RUN",
             "collapsed": False,          # tier ③ sets True -> hidden unless EXPT_SHOW_ALL=1
             "rows": [
                 {"label": "stop", "value": "max_runtime_s=600  max_data_id=10"},
                 {"label": "delays", "value": "D=0", "level": "warn",
                  "note": "both sides matched"},
             ]},
        ],
        "checks": [
            {"name": "D matched across real/sim pair", "level": "ok",
             "detail": "..."},
            {"name": "agg_goal <= c (sync)", "level": "error",
             "detail": "fwdllm_plus agg_goal=5 > c=2"},
        ],
        "next": "python -m scripts.parity.cli --batch ...",   # optional hand-off
    }

`level` is one of "ok" (default), "warn", "error". `render_and_gate` returns 2 if
any check (or any *row*) is level "error", else 0 -- the driver treats 2 as
"blocked unless --force".
"""

from __future__ import annotations

import json
import os
import sys

# ---- colour / icon helpers -------------------------------------------------

def _use_colour(stream) -> bool:
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("EXPT_FORCE_COLOR") is not None:
        return True
    return hasattr(stream, "isatty") and stream.isatty()


class _Style:
    def __init__(self, on: bool):
        self.on = on

    def _w(self, code: str, s: str) -> str:
        return f"\033[{code}m{s}\033[0m" if self.on else s

    def red(self, s):    return self._w("1;31", s)
    def yellow(self, s): return self._w("1;33", s)
    def green(self, s):  return self._w("32", s)
    def dim(self, s):    return self._w("2", s)
    def bold(self, s):   return self._w("1", s)


# level -> (icon, colouriser name). "ok" rows stay quiet; set/warn/error shout.
#   set   = value explicitly overridden by a command-line flag (overrides yaml) -> green
#   warn  = review / attention                                                  -> yellow
#   error = infeasible                                                          -> red
_ICON = {"ok": " ", "set": "\U0001f7e2", "warn": "\U0001f7e1", "error": "\U0001f534"}  #   🟢 🟡 🔴
_CHECK_ICON = {"ok": "✓", "warn": "⚠", "error": "✗"}  # ✓ ⚠ ✗


def _colour_for(st: _Style, level: str):
    return {"ok": st.dim, "set": st.green, "warn": st.yellow, "error": st.red}.get(level, st.dim)


# ---- renderer --------------------------------------------------------------

_RULE = "─" * 79  # ─────


def render_and_gate(spec: dict, show_all: bool | None = None, stream=None) -> int:
    """Render the tiered table + checks; return 2 if any error, else 0.

    Rows/checks at level "error" mark the config infeasible; the driver should
    refuse to launch (unless the operator passes --force).
    """
    stream = stream or sys.stdout
    st = _Style(_use_colour(stream))
    if show_all is None:
        show_all = os.environ.get("EXPT_SHOW_ALL") not in (None, "", "0")

    def out(s=""):
        print(s, file=stream)

    title = spec.get("title", "EXPERIMENT RUN")
    subtitle = spec.get("subtitle", "")
    tag = st.yellow("[DRY-RUN]") if spec.get("dry_run") else ""
    out()
    out(f" {st.bold(title)}  {subtitle}  {tag}".rstrip())
    out(" " + _RULE)

    n_err_rows = 0
    for tier in spec.get("tiers", []):
        name = tier.get("name", "")
        rows = tier.get("rows", [])
        collapsed = tier.get("collapsed", False)
        out(" " + st.bold(name))
        if collapsed and not show_all:
            out(st.dim(f"      [{len(rows)} row(s) hidden — set EXPT_SHOW_ALL=1]"))
            continue
        for r in rows:
            level = r.get("level", "ok")
            if level == "error":
                n_err_rows += 1
            icon = _ICON.get(level, " ")
            col = _colour_for(st, level)
            # Pad short labels to a column; always keep >=1 space before the value
            # so a label at/over the column width doesn't run into it.
            raw = r.get("label", "")
            label = raw.ljust(13) if len(raw) < 13 else raw + " "
            value = r.get("value", "")
            note = r.get("note", "")
            note_s = f"  {st.dim('· ' + note)}" if note else ""
            # ok rows: label dim, value plain. warn/error: value coloured.
            if level == "ok":
                out(f"  {icon} {st.dim(label)} {value}{note_s}")
            else:
                out(f"  {icon} {st.bold(label)} {col(value)}{note_s}")

    out(" " + _RULE)

    # ---- checks / gate ----
    checks = spec.get("checks", [])
    n_ok = sum(1 for c in checks if c.get("level", "ok") == "ok")
    n_warn = sum(1 for c in checks if c.get("level") == "warn")
    n_err = sum(1 for c in checks if c.get("level") == "error")
    summary = (f" PRE-FLIGHT: {len(checks)} check(s) … "
               f"{st.green(str(n_ok) + ' ✓')}  "
               f"{st.yellow(str(n_warn) + ' ⚠')}  "
               f"{st.red(str(n_err) + ' ✗')}")
    out(summary)
    for c in checks:
        level = c.get("level", "ok")
        if level == "ok" and not show_all:
            continue  # keep the passing checks quiet unless asked
        icon = _CHECK_ICON.get(level, "?")
        col = _colour_for(st, level)
        detail = c.get("detail", "")
        detail_s = f"  {st.dim('— ' + detail)}" if detail else ""
        out(f"   {col(icon)} {c.get('name', '')}{detail_s}")

    blocked = (n_err + n_err_rows) > 0
    if blocked:
        out(" " + st.red("BLOCKED: infeasible config — fix, or re-run with --force to override."))
    if spec.get("next"):
        out(" " + st.bold("NEXT: ") + spec["next"])
    out()
    return 2 if blocked else 0


def main(argv=None) -> int:
    """CLI form: `expt_runner.py <spec.json>` -> render + gate, exit 0/2."""
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print("usage: expt_runner.py <spec.json>", file=sys.stderr)
        return 2
    with open(argv[0], encoding="utf-8") as fh:
        spec = json.load(fh)
    return render_and_gate(spec)


if __name__ == "__main__":
    sys.exit(main())
