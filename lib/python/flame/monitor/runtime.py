# Copyright 2023 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
"""Runtime for Metric Collector."""

import logging
import time

logger = logging.getLogger(__name__)


def timer_decorator(func):
    """Decorator to time TopAggregator function and log round/data info.
    Make sure to populate fwd_llm_stage within the function or in the same class for detailed logging."""

    def wrapper(*args, **kwargs):
        logger.debug("Inside timer_decorator wrapper")
        self = args[0]  # TopAggregator or Trainer -- both expose vclock_now

        # vclock_now is a property on both base classes (simulate_fwdllm.md §N):
        # None in real mode (there is no virtual clock there), a float in sim
        # mode. getattr is defensive only for a `self` that's neither (e.g. a
        # free function wrapped by mistake) -- not a mode branch, there is
        # exactly one such branch and it lives inside the property itself.
        vclock_start = getattr(self, "vclock_now", None)
        start = time.time()
        cpu_start = time.process_time()
        result = func(*args, **kwargs)
        end = time.time()
        cpu_end = time.process_time()
        vclock_end = getattr(self, "vclock_now", None)
        duration = end - start
        # CPU time (this process only) vs wall time (simulate_fwdllm.md §B,
        # 07-19 pm): distinguishes "waiting on contention" (wall inflated, CPU
        # flat) from "genuinely more compute" (CPU itself inflated) for the
        # agg_step_timing_breakdown residual the narrow GPU-pass-window density
        # check already refuted as pure contention.
        cpu_duration = cpu_end - cpu_start
        # Meaningful delta on the aggregator (vclock ticks live there);
        # usually 0 on a trainer (vclock_now is a snapshot between messages,
        # see Trainer.vclock_now) -- both are correct, not a bug.
        vclock_delta = (
            vclock_end - vclock_start
            if vclock_start is not None and vclock_end is not None
            else None
        )

        stage = getattr(self, "fwd_llm_stage", None)
        if stage:
            logger.info(
                f"[decorator] Runtime of {func.__name__}: {duration:.6f}s "
                f"(Round={stage.round_id}, DataId={stage.data_id}, Iter={stage.iteration}, TrainerId={stage.trainer_id})"
                + (f" vclock={vclock_delta:.3f}s" if vclock_delta is not None else "")
            )
            # Structured companion to the log line: attribute step wall time to
            # (func, data_id, iteration) for GPU-cost decomposition. No-op when
            # telemetry is off. The `stage` guard keeps nested helpers whose
            # args[0] is not the trainer self from emitting mis-attributed
            # records. Best-effort: a telemetry hiccup must never break training.
            try:
                from flame import telemetry
                if telemetry.is_enabled():
                    from flame.telemetry.events import build_step_timing
                    ev, fields = build_step_timing(
                        func=func.__name__, duration_s=duration,
                        round_num=stage.round_id, data_id=stage.data_id,
                        iteration=stage.iteration, trainer_id=stage.trainer_id,
                        vclock_s=vclock_delta, vclock_now_s=vclock_end,
                        cpu_duration_s=cpu_duration,
                    )
                    telemetry.emit(ev, **fields)
            except Exception:  # pragma: no cover - telemetry must never fault training
                logger.debug("step_timing telemetry emit failed", exc_info=True)
        else:
            logger.info(
                f"[decorator] Runtime of {func.__name__}: {duration:.6f}s (no stage info)"
            )
        return result

    return wrapper


class FwdLLMStage:
    """Lightweight metadata object for each federated round of FwdLLM."""

    def __init__(self, round_id, data_id, iteration, trainer_id=None):
        self.round_id = round_id
        self.data_id = data_id
        self.iteration = iteration
        self.trainer_id = trainer_id

    def __repr__(self):
        if self.trainer_id:
            return f"FwdLLMStage(round={self.round_id}, data_id={self.data_id}, iter={self.iteration}, trainer_id={self.trainer_id})"
        else:
            return f"FwdLLMStage(round={self.round_id}, data_id={self.data_id}, iter={self.iteration})"


def time_tasklet(func):
    """Decorator to time Tasklet.do() function"""

    def wrapper(*args, **kwargs):
        s = args[0]
        if s.composer.mc:
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()

            s.composer.mc.save("runtime", s.alias, end - start)
            s.composer.mc.save("starttime", s.alias, start)
            logger.info(f"Runtime of {s.alias} is {end-start}")
            return result
        else:
            logger.debug("No MetricCollector; won't record runtime")
            return func(*args, **kwargs)

    return wrapper
