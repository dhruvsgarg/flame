# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0
"""FX-D44: spawned FL processes keep freed model buffers in the glibc heap."""

from flame.launch.aggregator_spawner import _MALLOC_ENV, apply_malloc_env


def test_sets_heap_knobs_by_default():
    env = apply_malloc_env({})
    assert all(env[k] == v for k, v in _MALLOC_ENV.items())


def test_caller_value_wins_and_knob_disables():
    assert apply_malloc_env({"MALLOC_TRIM_THRESHOLD_": "1"})["MALLOC_TRIM_THRESHOLD_"] == "1"
    assert not set(_MALLOC_ENV) & set(apply_malloc_env({"FLAME_MALLOC_TUNE": "0"}))


def test_trainer_spawner_keeps_glibc_defaults():
    """FX-D59: only aggregators get the heap knobs; trainers opt in with FLAME_MALLOC_TUNE_TRAINERS=1."""
    import inspect
    from flame.launch import spawner
    src = inspect.getsource(spawner)
    assert 'env.get("FLAME_MALLOC_TUNE_TRAINERS") == "1"' in src
    assert "env = apply_malloc_env(os.environ.copy())" not in src
