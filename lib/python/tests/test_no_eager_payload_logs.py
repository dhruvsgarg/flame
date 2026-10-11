# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""FX-N77: no log f-string interpolates a whole message, payload, weights dict or version ledger.

An f-string formats even when its level is off: `logger.debug(f"... {msg}")` repr'd a 29 MB update per commit
(0.27 s/update on speech feddance). fwdllm files are exempt while FluxTune is parked (FLUXTUNE_READINESS).
"""

import ast
from pathlib import Path

FLAME = Path(__file__).resolve().parents[1] / "flame"
BAD = {"msg", "message", "payload", "weights", "delta_weights", "self.weights", "msg.payload",
       "self._track_trainer_version_duration_s", "self._track_trainer_version_duration_s[end]"}
LEVELS = {"debug", "info", "warning", "error", "critical", "exception"}


def _hits(path: Path):
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in LEVELS:
            for arg in node.args:
                for v in ast.walk(arg):
                    if isinstance(v, ast.FormattedValue) and ast.unparse(v.value) in BAD:
                        yield f"{path.relative_to(FLAME.parent)}:{v.lineno}: {{{ast.unparse(v.value)}}}"


def test_no_eager_payload_formatting_in_logs():
    hits = [h for p in FLAME.rglob("*.py") if "fwdllm" not in p.name for h in _hits(p)]
    assert not hits, "\n".join(hits)
