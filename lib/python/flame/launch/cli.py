# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Helpers for example main.py files: dual --config / --config-json support."""

import argparse
import json
import os
import tempfile

from flame.config import Config


def load_config_from_argv(default_config: str = "./config.json") -> Config:
    """Parse argv for --config <path> or --config-json <str> and return a Config.

    Examples accept either form so the runtime config can come from a file on
    disk or be piped in by the launcher as a JSON string.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", default=default_config)
    parser.add_argument("--config-json", default=None)
    args, _ = parser.parse_known_args()

    if args.config_json:
        cfg_dict = json.loads(args.config_json)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(cfg_dict, f)
            tmp = f.name
        try:
            return Config(tmp)
        finally:
            os.unlink(tmp)

    return Config(args.config)


def exit_after_run(code: int = 0) -> None:
    """FX-D178: join threads, run atexit hooks, flush, then os._exit (skips native teardown); FLAME_HARD_EXIT=0 reverts."""
    if os.environ.get("FLAME_HARD_EXIT", "1") == "0":
        return
    import atexit
    import logging
    import sys
    import threading

    for t in threading.enumerate():  # as interpreter shutdown: join non-daemon threads first
        if t is not threading.current_thread() and not t.daemon:
            t.join()
    atexit._run_exitfuncs()
    logging.shutdown()
    for s in (sys.stdout, sys.stderr):
        try:
            s.flush()
        except Exception:
            pass
    os._exit(code)
