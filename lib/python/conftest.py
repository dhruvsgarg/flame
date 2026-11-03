"""Repo-wide pytest setup."""
import os

# xdist runs one worker per core: one math thread each, or torch's per-process pools oversubscribe (R18).
if os.environ.get("PYTEST_XDIST_WORKER"):
    for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[_var] = "1"
