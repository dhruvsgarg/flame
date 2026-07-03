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
             "bar": "44",                 # optional header-bar bg colour code (default cycles)
             "rows": [                     # label/value rows …
                 {"label": "stop", "value": "max_runtime_s=600  max_data_id=10"},
                 {"label": "delays", "value": "D=0", "level": "warn",
                  "note": "both sides matched"},
             ]},
            {"name": "② PER-BASELINE",     # … OR an aligned table (mutually exclusive with rows)
             "table": {
                 "columns": ["c", "agg_goal", ("min_init", "minInit")],  # key or (key, header)
                 "rows": [{"name": "fluxtune", "cells": {"c": 10, "agg_goal": 3}}],
                 "overridden": ["c"],      # optional: columns set by a CLI flag (green •)
             }},
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
    def cyan(self, s):   return self._w("1;36", s)
    def dim(self, s):    return self._w("2", s)
    def bold(self, s):   return self._w("1", s)

    def bar(self, s: str, width: int, code: str = "44") -> str:
        """A solid-background full-width header bar (bold bright-white on `code`),
        so the ①②③ section markers read as distinct blocks, not plain text."""
        txt = (" " + s).ljust(width)
        return self._w(f"1;97;{code}", txt)


# level -> (icon, colouriser name). "ok" rows stay quiet; set/warn/error shout.
#   set   = value explicitly overridden by a command-line flag (overrides yaml) -> green
#   warn  = review / attention                                                  -> yellow
#   error = infeasible                                                          -> red
# NB: the emoji markers render 2 cells wide, so "ok" uses TWO spaces to keep the
# label column aligned with the set/warn/error rows.
_ICON = {"ok": "  ", "set": "\U0001f7e2", "warn": "\U0001f7e1", "error": "\U0001f534"}  #   🟢 🟡 🔴
_CHECK_ICON = {"ok": "✓", "warn": "⚠", "error": "✗"}  # ✓ ⚠ ✗


def _colour_for(st: _Style, level: str):
    return {"ok": st.dim, "set": st.green, "warn": st.yellow, "error": st.red}.get(level, st.dim)


# ---- renderer --------------------------------------------------------------

_RULE = "─" * 79  # ─────
_BAR_W = 79       # header-bar width (matches the rule)

# Per-tier header-bar background colours so ①②③ are visually distinct blocks.
_TIER_BAR = ["44", "45", "100"]  # blue, magenta, bright-black(grey)


def _cell(v) -> str:
    """Table cell text: None -> '–', everything else str()."""
    return "–" if v is None else str(v)


def _render_table(out, st, table: dict) -> None:
    """Render an aligned per-entity table (tier ②'s per-baseline knobs).

    table = {
      "columns": [key | (key, header), ...],   # column order
      "rows":    [{"name": str, "cells": {key: value}}, ...],
      "overridden": [key, ...],                 # optional: flag-overridden cols
    }

    Columns whose value is NOT identical across every row are HIGHLIGHTED (bold
    yellow header + cells) -- those are the knobs that differ between baselines
    and must be eyeballed. Columns identical across all rows stay dim (expected).
    Row (baseline) names are cyan. A flag-overridden column gets a green '•'.
    """
    cols = [(c, c) if isinstance(c, str) else (c[0], c[1]) for c in table["columns"]]
    rows = table["rows"]
    overridden = set(table.get("overridden", []))

    # Which columns differ across baselines?
    differs = {}
    for key, _h in cols:
        vals = {_cell(r["cells"].get(key)) for r in rows}
        differs[key] = len(vals) > 1

    name_w = max([len("baseline")] + [len(_cell(r["name"])) for r in rows])
    col_w = {}
    for key, hdr in cols:
        col_w[key] = max(len(hdr), max((len(_cell(r["cells"].get(key))) for r in rows),
                                       default=0))

    # header row
    hcells = [st.dim("baseline".ljust(name_w))]
    for key, hdr in cols:
        mark = st.green("•") if key in overridden else " "
        htxt = hdr.ljust(col_w[key])
        hcells.append((st.yellow(htxt) if differs[key] else st.dim(htxt)) + mark)
    out("   " + "  ".join(hcells))

    # data rows
    for r in rows:
        line = [st.cyan(_cell(r["name"]).ljust(name_w))]
        for key, hdr in cols:
            v = _cell(r["cells"].get(key)).ljust(col_w[key])
            line.append((st.yellow(v) if differs[key] else st.dim(v)) + " ")
        out("   " + "  ".join(line))
    # legend for the highlighting
    diff_names = [h for k, h in cols if differs[k]]
    legend = st.yellow("yellow") + st.dim(" = differs across baselines (review)")
    if overridden:
        legend += st.dim("   ") + st.green("•") + st.dim(" = flag override")
    out("   " + st.dim("· ") + legend
        + (st.dim(f"   [{', '.join(diff_names)}]") if diff_names else ""))


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
    for ti, tier in enumerate(spec.get("tiers", [])):
        name = tier.get("name", "")
        rows = tier.get("rows", [])
        table = tier.get("table")
        collapsed = tier.get("collapsed", False)
        bar_code = tier.get("bar", _TIER_BAR[ti % len(_TIER_BAR)])
        out(" " + st.bar(name, _BAR_W, bar_code))
        n_hidden = len(table["rows"]) if table else len(rows)
        if collapsed and not show_all:
            out(st.dim(f"      [{n_hidden} row(s) hidden — set EXPT_SHOW_ALL=1]"))
            continue
        if table:
            _render_table(out, st, table)
            continue
        # Align every value in this tier to one column: pad labels to the widest
        # label present (so a long `max_data_id_progress` doesn't stagger the
        # shorter rows' values). Cap so a pathological label doesn't push values
        # off-screen.
        label_w = min(max((len(r.get("label", "")) for r in rows), default=13), 24)
        for r in rows:
            level = r.get("level", "ok")
            if level == "error":
                n_err_rows += 1
            icon = _ICON.get(level, " ")
            col = _colour_for(st, level)
            # Pad every label to the same width + one trailing space, so all
            # values in the tier start at the same column (no stagger).
            raw = r.get("label", "")
            label = raw.ljust(label_w) + " "
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
