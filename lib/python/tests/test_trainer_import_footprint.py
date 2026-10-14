# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""FX-N84: every trainer/aggregator process imports flame; heavy optional deps must stay lazy."""

import subprocess
import sys
from pathlib import Path

LIB = Path(__file__).resolve().parents[1]


def test_flame_roles_do_not_import_mlflow_or_pandas():
    code = ("import sys; import flame.registries, flame.mode.horizontal.asyncfl.trainer; "
            "print(sorted(m for m in ('mlflow', 'pandas') if m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], cwd=LIB, capture_output=True, text=True, check=True)
    assert out.stdout.strip().splitlines()[-1] == "[]"


def test_mlflow_registry_still_resolves():
    from flame.config import RegistryType
    from flame.registries import registry_provider

    assert RegistryType.MLFLOW in registry_provider._objects
    assert type(registry_provider.get(RegistryType.DUMMY)).__name__ == "DummyRegistryClient"
