import contextlib
import copy
import logging
import random
import time
import math
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from flame.mode.horizontal.syncfl.fwdllm_aggregator import TopAggregator
from examples.fwdllm.trainer.forward_training.fwdgrad_utils import calculate_var, calculate_snr, calculate_cv, calculate_real_var, calculate_snr_gradients
from examples.fwdllm.expts.dataset_registry import max_dominant_share
from examples.fwdllm.expts.landing_law import (
    B_MAX_PRIOR, BUDGET_STOP_FRAC_DEFAULT, T_RES_DEFAULT,
    rho_gate_cap, rho_star_now,
)
from examples.fwdllm.expts.bmax_probe import (
    PHI_GRID, b_max_from_knee, knee, noise_scale,
)
from flame.monitor.runtime import timer_decorator, FwdLLMStage

logger = logging.getLogger(__name__)
import functorch as fc

import hashlib

# B17: fixed so the cos probe reads the SAME batch every commit and across arms.
_COS_PROBE_SEED = 20260810


def _calculate_hash(tensor):
    if tensor is None:
        return ""

    """Calculate a hash for a tensor for logging."""
    return hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest()


@contextlib.contextmanager
def _agg_sync_timer(owner, name: str):
    """Times a single sync point (`.item()`, `.to("cpu")`) separately from
    its enclosing `@timer_decorator`-wrapped function, as its own named
    `step_timing` event."""
    t0 = time.time()
    try:
        yield
    finally:
        dur = time.time() - t0
        stage = getattr(owner, "fwd_llm_stage", None)
        if stage is not None:
            try:
                from flame import telemetry
                if telemetry.is_enabled():
                    from flame.telemetry.events import build_step_timing
                    ev, fields = build_step_timing(
                        func=name, duration_s=dur,
                        round_num=stage.round_id, data_id=stage.data_id,
                        iteration=stage.iteration, trainer_id=stage.trainer_id,
                    )
                    telemetry.emit(ev, **fields)
            except Exception:  # pragma: no cover - telemetry must never fault training
                logger.debug("agg_sync_timer telemetry emit failed", exc_info=True)


# E[v_par^2] for coin-flip-top-2-of-10: 2.988 over 37k events, 2.987 synthetic
# (handoff §11.2). A property of the rule, so a constant and not a knob.
_E_SELECT_COIN_TOP2 = 2.988


class FedSGDAggregator(TopAggregator):

    def __init__(
        self,
        train_global,
        test_global,
        all_train_data_num,
        train_data_local_dict,
        test_data_local_dict,
        train_data_local_num_dict,
        worker_num,
        device,
        args,
        model_trainer,
        num_labels,
    ):
        self.trainer = model_trainer
        logger.info(f"self.trainer = {self.trainer}")
        self.args = args.hyperparameters
        self.train_global = train_global
        self.test_global = test_global
        self.val_global = self._generate_validation_set()
        self.all_train_data_num = all_train_data_num
        self.config = args
        self.model = model_trainer.model
        self.dataset = None
        self.num_labels = num_labels
        # H14 audit: per-contribution grad norms into the variance gate. Default
        # OFF (costs a GPU->CPU sync per pool entry); flip on BOTH legs of a pair.
        self._var_calc_audit = bool(getattr(self.args, "var_calc_audit", False))
        if self._var_calc_audit:
            logger.info("[VAR_CALC_AUDIT] emitting per-cycle grad-norm records")
        # I-1 audit: per-commit applied-update vs weight norm. Default OFF — its
        # wall cost perturbs the async arrival order that separates two legs (§D-45).
        self._server_update_audit = bool(
            getattr(self.args, "server_update_audit", False)
        )
        if self._server_update_audit:
            logger.info("[SERVER_UPDATE_AUDIT] emitting per-commit update/weight norms")
        # L1 audit: pool split-half cosine. Own flag, not server_update_audit's:
        # it adds a pass over (params x uploads), which would change that flag's
        # cost profile and with it the arrival order (§D-45).
        self._pool_split_half_audit = bool(
            getattr(self.args, "pool_split_half_audit", False)
        )
        if self._pool_split_half_audit:
            logger.info("[POOL_SPLIT_HALF_AUDIT] emitting per-commit pool agreement")
        # B1 (§15.1): ground-truth cos(G,g). The split-half estimator returns zero
        # within noise even at matched N (§22.3e), so manufacture a real `g` with a
        # backward pass -- server-side, one fixed held-out batch, once per commit.
        self._cos_ground_truth_audit = bool(
            getattr(self.args, "cos_ground_truth_audit", False)
        )
        self._cos_probe_batch = None
        # B17: even shuffled, n=64 aligns with the true held-out gradient at
        # only 0.48; 1024 reaches 0.94 (§3.9).
        self._cos_probe_batch_size = int(
            getattr(self.args, "cos_probe_batch_size", 1024) or 1024
        )
        # ~80 ms per reference sample = 83 s/commit at 1024, against 1.6 s for
        # the rest of the commit path. Cost is LINEAR in the reference, so only
        # a stride helps; D is read in 50-commit blocks and loses no resolution.
        self._cos_probe_every = max(
            1, int(getattr(self.args, "cos_probe_every", 1) or 1)
        )
        if self._cos_ground_truth_audit:
            logger.info(
                "[COS_GROUND_TRUTH_AUDIT] emitting cos(G,g) every "
                f"{self._cos_probe_every} commit(s) against a "
                f"fixed held-out batch of {self._cos_probe_batch_size}"
            )

        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict
        self.train_data_local_num_dict = train_data_local_num_dict

        self.worker_num = worker_num
        self.device = device
        self.model_dict = dict()
        self.sample_num_dict = dict()
        self.flag_client_model_uploaded_dict = dict()

        for idx in range(self.worker_num):
            self.flag_client_model_uploaded_dict[idx] = False
        # ratio is 0 and the comm_round is 3000 rn
        self.warmup_rounds = math.ceil(self.args.comm_round * self.args.warmup_ratio)

        # 之前的v不够，暂存在cached_v
        self.cached_v = []

        # Per-model defaults; overridden by hyperparameters.var_threshold in config.
        _DEFAULT_VAR_THRESHOLD_BY_MODEL = {
            "distilbert": 0.1,
            "bert": 0.2,
            "roberta-large": 0.2,
            "albert": 0.1,
        }
        _default_thr = _DEFAULT_VAR_THRESHOLD_BY_MODEL.get(self.args.model_type, 0.1)

        _json_thr = getattr(self.args, "var_threshold", None)
        if _json_thr is not None:
            self.var_threshold = float(_json_thr)
            logger.info(
                f"[VarThreshold] Using JSON override var_threshold={self.var_threshold} "
                f"(model_type={self.args.model_type}; default would be {_default_thr})"
            )
        else:
            self.var_threshold = _default_thr
            logger.info(
                f"[VarThreshold] Using default var_threshold={self.var_threshold} "
                f"for model_type={self.args.model_type}"
            )

        # S-A + S-B (handoff §15.4, §15.6). raw_sgd = theta -= eta*G/N (historical).
        # trust_ratio = theta -= rho*_t * ||theta_tr|| * G/||G||, making rho an
        # operator constant instead of an emergent one. They must ship together:
        # a CONSTANT rho* still grows ||theta|| geometrically, so boundedness comes
        # from the anneal. rm = rho*_0 * t^-rho_exp with rho_exp > 0.5, strictly
        # inside Robbins-Monro rather than on its edge (1/sqrt(t) only defers).
        self._server_step_rule = str(
            getattr(self.args, "server_step_rule", "raw_sgd") or "raw_sgd"
        ).lower()
        if self._server_step_rule not in ("raw_sgd", "trust_ratio"):
            logger.warning(
                f"unknown server_step_rule={self._server_step_rule!r}; using raw_sgd"
            )
            self._server_step_rule = "raw_sgd"
        self._rho_star = float(getattr(self.args, "rho_star", 0.01) or 0.01)
        self._rho_schedule = str(
            getattr(self.args, "rho_schedule", "const") or "const"
        ).lower()
        self._rho_exp = float(getattr(self.args, "rho_exp", 0.55) or 0.55)
        self._commit_count = 0
        # Q2 (§6.1). Decay the trainable slice after the step. `auto` = rho^2/2
        # cancels Leg 1's inflation exactly, pinning Phi = 1 by construction, which
        # is what separates "||theta_tr|| is causal" from "it is a symptom".
        _wd = getattr(self.args, "server_weight_decay", None)
        self._weight_decay = None
        if _wd is not None and str(_wd).strip() != "":
            self._weight_decay = (
                "auto" if str(_wd).lower() == "auto" else float(_wd)
            )
            if self._weight_decay != "auto" and self._weight_decay < 0:
                logger.warning(
                    f"negative server_weight_decay={self._weight_decay}; disabling"
                )
                self._weight_decay = None
        if self._weight_decay is not None:
            logger.info(f"[WeightDecay] server_weight_decay={self._weight_decay} "
                        "(trainable slice only, applied after the step)")
        if self._server_step_rule == "trust_ratio":
            logger.info(
                f"[ServerStep] trust_ratio rho_star={self._rho_star} "
                f"schedule={self._rho_schedule} exp={self._rho_exp}"
            )

        # S-C (handoff §15.7). Scale-free commit gate: `rho <= s*cos` solved for
        # the pool, N >= p*(rho_t/s)^2 / G_rule. No unit-carrying constant -- p is
        # read off the model, G_rule is closed form, s is O(1) -- so the setpoint
        # survives a change of alpha, K, ||theta|| or anneal. Default `var` = old.
        self._commit_gate = str(
            getattr(self.args, "commit_gate", "var") or "var"
        ).lower()
        if self._commit_gate not in ("var", "n_target"):
            logger.warning(f"unknown commit_gate={self._commit_gate!r}; using var")
            self._commit_gate = "var"
        self._gate_safety_s = float(getattr(self.args, "gate_safety_s", 0.4) or 0.4)
        # Which rho sizes the pool. `annealed` (shipped) uses rho_t, which under
        # S-B drives N_req -> 0, floors the gate at I=1 and decays progress as
        # rho^2 (§22.3a). `setpoint` sizes from rho*_0, so the anneal shrinks the
        # step while the gate holds the aim.
        self._gate_rho_ref = str(
            getattr(self.args, "gate_rho_ref", "annealed") or "annealed"
        ).lower()
        if self._gate_rho_ref not in ("annealed", "setpoint"):
            logger.warning(
                f"unknown gate_rho_ref={self._gate_rho_ref!r}; using annealed"
            )
            self._gate_rho_ref = "annealed"
        self._last_rho = None
        self._p_trainable = self._g_rule = None
        if self._commit_gate == "n_target":
            self._p_trainable = sum(
                p.numel() for p in self.trainer.model.parameters() if p.requires_grad
            )
            # G_rule = E[v_par^2] under `select`, P under `mean` (§3.5). Needs the
            # aggregator's copy of two trainer knobs; logged so a mismatch shows.
            _pc = str(getattr(self.args, "probe_combine", "select") or "select").lower()
            _P = int(getattr(self.args, "perturbation_count", 10) or 10)
            self._g_rule = float(_P) if _pc == "mean" else _E_SELECT_COIN_TOP2
            logger.info(
                f"[CommitGate] n_target s={self._gate_safety_s} p={self._p_trainable} "
                f"probe_combine={_pc} P={_P} G_rule={self._g_rule} "
                f"rho_ref={self._gate_rho_ref}"
            )

        self.track_trainer_avail = (
            self.config.hyperparameters.track_trainer_avail or None
        )
        self.reject_stale_updates = (
            self.config.hyperparameters.reject_stale_updates or False
        )
        # See Hyperparameters.staleness_policy (flame/config.py) for the
        # three modes. Falls back to the older reject_stale_updates boolean
        # ("round_data_id" if set, else "none") when unset, so examples that
        # predate this knob keep their existing behavior.
        self.staleness_policy = getattr(
            self.config.hyperparameters, "staleness_policy", None
        ) or ("round_data_id" if self.reject_stale_updates else "none")
        logger.info(f"[StalenessPolicy] using policy={self.staleness_policy}")
        self.trainer_event_dict = None
        if (
            self.track_trainer_avail["enabled"]
            and self.track_trainer_avail["type"] == "ORACULAR"
        ):
            self.trainer_event_dict = self.read_trainer_unavailability(
                self.track_trainer_avail["trace"]
            )
            logger.info(f"self.trainer_event_dict:{ self.trainer_event_dict}")

        self.loss_list = []
        self.grad_for_var_check_list = []
        self.jvp_for_snr_check_list = []
        self.var_good_enough = True
        self.var_prev_iter_list = []
        self._n_eff_scalar = None
        self.snr = None
        self.snr_prev_iter_list = []
        self.fmodel, self.params, self.buffers = fc.make_functional_with_buffers(
            self.model
        )
        self.grad = [torch.zeros_like(p) for p in self.params]

        # Server-side momentum on the raw SGD update; 0.0 (default) is byte-identical.
        self.server_momentum = float(
            getattr(self.config.hyperparameters, "server_momentum", 0.0) or 0.0
        )
        self._server_momentum_buf = {}

        # C-1 (buildplan §5, T5 2026-08-15). B = 1/2*sum log1p(rho^2) depends on
        # rho alone, so accounting is exact, free and always on; everything that
        # READS it -- the anneal and the stop -- is flag-gated off.
        self._B = 0.0
        self._stop_fired = None
        self._rho_max = None
        # `landing` = law C, with T_res a fixed RATE never decremented, so B
        # approaches B_max as B_max*(1-e^{-t/T_res}), always from below.
        # Decrementing it would reinstate T as an operator input (§4.6a) and buy
        # nothing: Lambda = 2B/s is schedule-free. b_max starts at D1's ln 2.
        self._b_max = float(getattr(self.args, "b_max", 0.0) or B_MAX_PRIOR)
        self._t_res = float(getattr(self.args, "t_res", 0.0) or T_RES_DEFAULT)
        self._budget_stop_frac = float(
            getattr(self.args, "budget_stop_frac", 0.0) or BUDGET_STOP_FRAC_DEFAULT
        )
        self._phi_stop = str(getattr(self.args, "phi_stop", "off") or "off").lower()
        if self._phi_stop not in ("off", "log_only", "halt"):
            logger.warning(f"unknown phi_stop={self._phi_stop!r}; using off")
            self._phi_stop = "off"
        self._phi_stop_threshold = float(
            getattr(self.args, "phi_stop_threshold", 0.0) or 2.7
        )
        # 3.1: re-sense B_max on a stride. 0 = off (byte-identical); the probe
        # costs len(phis)+1 forward passes over b_max_probe_n samples per fire,
        # so it is strided and its elapsed time is logged per fire (task 0.7).
        self._b_max_probe_every = max(
            0, int(getattr(self.args, "b_max_probe_every", 0) or 0)
        )
        _phis = getattr(self.args, "b_max_probe_phis", None)
        self._b_max_phis = ([float(x) for x in str(_phis).split(",")]
                            if _phis else list(PHI_GRID))
        self._b_max_probe_n = int(
            getattr(self.args, "b_max_probe_n", 0) or 512
        )
        if self._b_max_probe_every:
            logger.info(
                f"[BmaxProbe] re-sensing every {self._b_max_probe_every} commits "
                f"on {self._b_max_probe_n} held-out samples, "
                f"Phi grid {self._b_max_phis}"
            )
        if self._rho_schedule == "landing":
            if self.server_momentum:
                # Edge case (b): under beta > 0, Phi = exp((1+beta)/(1-beta)*B),
                # so B no longer measures the inflation the budget is denominated
                # in. Refuse rather than silently misprice it.
                raise ValueError(
                    "rho_schedule=landing is incompatible with server_momentum="
                    f"{self.server_momentum} -- Phi != e^B under momentum"
                )
            # Gate reachability, ceil(n_req/K) <= max_iter solved for rho.
            # NOT a rho* <= rho*_0 clamp: rho*_0 is the ln 2 prior, so that
            # would block 3.1 from spending the budget it just measured.
            _K = int(getattr(self.args, "aggregation_goal", 10) or 10)
            self._rho_max = rho_gate_cap(
                self._gate_safety_s, self._p_trainable, self._g_rule, _K,
                getattr(self.args, "max_iterations_per_data_id", None),
            )
            logger.info(
                f"[Landing] law=C B_max={self._b_max:.6g} T_res={self._t_res:g} "
                f"rho*_0={self._rho_star_now():.6g} rho_max={self._rho_max} "
                f"stop_frac={self._budget_stop_frac} phi_stop={self._phi_stop}"
            )
        elif self._phi_stop != "off":
            logger.info(
                f"[Landing] phi_stop={self._phi_stop} "
                f"threshold={self._phi_stop_threshold} (no sensed B_max)"
            )

    def var_within_epsilon(self):
        if self.var < self.var_threshold:
            logger.info("Var under threshold, aggregate now")
            return True
        if len(self.var_prev_iter_list) < 6:
            logger.info("Not enough iterations for stable delta")
            return False
        last_5_var_mean = np.mean(self.var_prev_iter_list[-6:-1])
        logger.info(f"Last 5 iterations var values : {self.var_prev_iter_list[-5:]} - mean {last_5_var_mean} - current: {self.var}")
        if (abs(last_5_var_mean - self.var) / last_5_var_mean ) < 0.03:
            logger.info("Delta Var is less, minimal ROI, aggregate now")
            return True
        logger.info("Var high, delta high")
        return False

    def snr_within_epsilon(self):
        logger.info(f"len(self.snr_prev_iter_list): {len(self.snr_prev_iter_list)}")
        if len(self.snr_prev_iter_list) < 6:
            return False
        last_5_var_mean = np.mean(self.snr_prev_iter_list[-6:-1])
        logger.info(f"Last 5 iterations snr values : {self.snr_prev_iter_list[-5:]} - mean {last_5_var_mean} - current: {self.snr}")
        if (abs(last_5_var_mean - self.snr) / last_5_var_mean ) < 0.005:
            return True
        return False

    def snr_within_epsilon_and_var_under(self, jvp_var):
        snr_within_epsilon = self.snr_within_epsilon()
        if snr_within_epsilon and jvp_var < 20:
            logger.info("SNR within range and variance also under control")
            return True
        elif snr_within_epsilon:
            logger.info(f"SNR within range but variance is high- var: {jvp_var}")
            return False
        logger.info("SNR is not stable yet")
        return False
        

    @timer_decorator
    def get_global_model_params(self):
        """Timed: `.cpu().state_dict()` is a real device transfer, not a free accessor."""
        return self.trainer.get_model_params()

    def get_global_model(self):
        return self.trainer.get_model()

    def set_global_model_params(self, model_parameters):
        self.trainer.set_model_params(model_parameters)

    def add_local_trained_result(self, index, model_params, sample_num):
        logger.info("add_model. index = %d" % index)
        self.model_dict[index] = model_params
        self.sample_num_dict[index] = sample_num
        self.flag_client_model_uploaded_dict[index] = True

    def check_whether_all_receive(self):
        for idx in range(self.worker_num):
            if not self.flag_client_model_uploaded_dict[idx]:
                return False
        for idx in range(self.worker_num):
            self.flag_client_model_uploaded_dict[idx] = False
        return True

    def _server_update_step(self, param_idx: int, raw_update: "torch.Tensor") -> "torch.Tensor":
        """Heavy-ball momentum on one parameter's update; momentum=0.0 is a no-op
        (byte-identical)."""
        if not self.server_momentum:
            return raw_update
        buf = self._server_momentum_buf.get(param_idx)
        if buf is None:
            buf = raw_update.clone()
        else:
            buf = buf.mul(self.server_momentum).add_(raw_update)
        self._server_momentum_buf[param_idx] = buf
        return buf

    def _compute_n_eff(self, var_scalar):
        """S-K (handoff §15.13): how much pooling the commit actually got.

        calculate_var averages over the check layer's m coords, so
        var = ||G_A - G_B||^2 / (2m); each upload is u_k = d_k*v_k with
        ||v_k||^2 ~= m, so m cancels and n_eff = 2*mean_k(||u_k||^2)/(m*var).
        Dimensionless and rule-agnostic: ||g||^2 and the estimator constant b^2
        cancel too. Anything that makes uploads disagree -- heterogeneity,
        staleness, correlated probes -- inflates var without inflating
        mean||u_k||^2, so it registers here as n_eff below the nominal pool.
        """
        lst = self.grad_for_var_check_list
        n = len(lst)
        if n < 2 or not var_scalar or var_scalar <= 0:
            return None
        m = lst[0].numel()
        if m == 0:
            return None
        mean_sq = sum(float(t.pow(2).sum()) for t in lst) / n
        return 2.0 * mean_sq / (m * var_scalar)

    def _gate_rho(self):
        """The rho the gate sizes against — see `_gate_rho_ref`.

        Under trust_ratio it is exact (it IS the setpoint); under raw SGD it is the
        last realised rho, which lags a commit. None until one exists.
        """
        if self._server_step_rule != "trust_ratio":
            return self._last_rho
        if getattr(self, "_gate_rho_ref", "annealed") == "setpoint":
            return self._rho_star
        return self._rho_star_now()

    def _n_required(self):
        """S-C: pool this commit's step needs, N_req = p*(rho/s)^2 / G_rule.

        None when no rho exists yet (raw_sgd, first commit) -> the cap fires.
        """
        rho = self._gate_rho()
        if not rho or not self._p_trainable or not self._g_rule:
            return None
        return self._p_trainable * (rho / self._gate_safety_s) ** 2 / self._g_rule

    def _gate_satisfied(self):
        """Has this pool earned a commit? `var` = fixed threshold; `n_target` = S-C."""
        if self._commit_gate != "n_target":
            return bool(self.var <= self.var_threshold)
        n_req = self._n_required()
        if n_req is None:
            return False
        # n_eff is the MEASURED pool (§15.13): the count while uploads are
        # independent, below it if they ever stop being.
        n_have = float(self._n_eff_scalar or len(self.grad_for_var_check_list))
        ok = n_have >= n_req
        logger.info(
            f"[CommitGate] n_have={n_have:.1f} n_req={n_req:.1f} "
            f"rho_t={self._gate_rho()} -> {'COMMIT' if ok else 'POOL'}"
        )
        return ok

    @timer_decorator
    def _compute_var(self):
        """Timed separately to localize aggregate()'s sim/real cost gap (simulate_fwdllm.md §B)."""
        result = calculate_var(self.grad_for_var_check_list)
        # Diagnostic telemetry (grad norms + var). Off by default: ~n_pool GPU->CPU
        # syncs per cycle. `var_calc_audit` turns it on WITHOUT global DEBUG, which
        # would also flood the log and perturb timing — the point is to diff the
        # pool's dispersion real vs sim (H14), not to debug everything.
        if getattr(self, "_var_calc_audit", False) or logger.isEnabledFor(logging.DEBUG):
            try:
                from flame import telemetry
                if telemetry.is_enabled():
                    from flame.telemetry.events import build_var_calc
                    stage = getattr(self, "fwd_llm_stage", None)
                    norms = [p.detach().norm().item() for p in self.grad_for_var_check_list]
                    ev, fields = build_var_calc(
                        round_num=getattr(stage, "round_id", None),
                        data_id=getattr(stage, "data_id", None),
                        iteration=getattr(stage, "iteration", None),
                        input_grad_norms=norms,
                        output_var=result.item() if hasattr(result, "item") else float(result),
                    )
                    telemetry.emit(ev, **fields)
            except Exception:  # pragma: no cover - telemetry must never fault training
                logger.debug("var_calc telemetry emit failed", exc_info=True)
        return result

    @timer_decorator
    def _snapshot_retry_cache(self):
        """Timed separately to localize aggregate()'s sim/real cost gap (simulate_fwdllm.md §B)."""
        return copy.deepcopy(self.model_dict)

    @timer_decorator
    def _snapshot_last_round_update(self, weighted_gradient_sum):
        """Timed separately; shared by both commit branches (was duplicated verbatim)."""
        return [p.clone().detach() for p in weighted_gradient_sum]

    @timer_decorator
    def _accumulate_retry_cache(self, model_list, training_num):
        """Timed separately to localize aggregate()'s sim/real cost gap (simulate_fwdllm.md §B)."""
        for cached_v in self.cached_v:
            model_list.append(cached_v)
            if logger.isEnabledFor(logging.DEBUG):
                format_hash = lambda d: [_calculate_hash(v)[:8] for v in d]
                logger.debug(
                    f"cached-v[i] - length : {len(cached_v[1])} (should be same as grad pool):  {format_hash(cached_v[1])}"
                )
            training_num += cached_v[0]
        return training_num

    @timer_decorator
    def _cache_grad_for_retry(self, model_dict_cached):
        """Timed separately to localize aggregate()'s sim/real cost gap (simulate_fwdllm.md §B)."""
        for idx in range(self.worker_num):
            self.cached_v.append(
                (self.sample_num_dict[idx], model_dict_cached[idx])
            )

    @timer_decorator
    def _apply_weighted_update(self, model_list, weighted_gradient_sum, old_param,
                                learning_rate, training_num):
        """Timed separately; shared by both commit branches (natural / force-commit),
        was duplicated verbatim."""
        _audit = getattr(self, "_server_update_audit", False)
        # Before the loop: it aliases weighted_gradient_sum[id] to model_list[0]'s
        # tensor at i==0, then accumulates into it in place.
        _split = self._pool_split_half_stats(model_list)
        _delta_sq = _weight_sq = 0.0
        _tr_delta_sq = _tr_weight_sq = 0.0
        # B1: `g` must be taken at theta_t, BEFORE the loops below mutate it --
        # cos(G,g) is the aim of the step about to be taken, not of the next one.
        _cos_due = (
            getattr(self, "_cos_ground_truth_audit", False)
            and self._commit_count % getattr(self, "_cos_probe_every", 1) == 0
        )
        _cos_probe = self._cos_probe_gradient() if _cos_due else None
        _cos_dot = _cos_g_sq = 0.0

        def _cos_accumulate(idx, pooled):
            """Fold one pooled per-param block of G into <G,g> and ||G||^2."""
            nonlocal _cos_dot, _cos_g_sq
            if _cos_probe is None:
                return
            _g = _cos_probe[0][idx]
            if _g is None:
                return
            _blk = pooled.detach().to("cpu", torch.float32)
            _cos_dot += float((_blk * _g).sum())
            _cos_g_sq += float(_blk.pow(2).sum())

        # Pass 1 pools the uploads. Under raw_sgd this stays fused with the write
        # below (byte-identical); trust_ratio needs ||G|| and ||theta_tr|| over the
        # WHOLE trainable slice before any tensor is touched, so it is split out.
        _trust = self._server_step_rule == "trust_ratio"
        # Q2: trust_ratio knows rho before the step; raw_sgd only after, so
        # `auto` falls back to the previous commit's realised rho.
        _wd_lam = self._wd_lambda(self._rho_star_now() if _trust else None)
        if _wd_lam:
            logger.info(f"[WeightDecay] commit={self._commit_count} "
                        f"lambda={_wd_lam:.6g}")
        for id, k in enumerate(weighted_gradient_sum):
            for i in range(0, len(model_list)):
                local_sample_number, local_model_params = model_list[i]
                # w = local_sample_number / training_num
                if i == 0:
                    weighted_gradient_sum[id] = local_model_params[id]
                else:
                    weighted_gradient_sum[id] += local_model_params[id]
            if not _trust:
                _cos_accumulate(id, weighted_gradient_sum[id] / training_num)
                # `.to("cpu")` syncs GPU->CPU; timed separately since it runs once
                # per param, not once per call.
                with _agg_sync_timer(self, "agg_apply_update_cpu_sync"):
                    _param_src = next(old_param)
                    _trainable = bool(_param_src.requires_grad)  # read before detach()
                    _param = _param_src.detach().to("cpu")
                    _update = self._server_update_step(
                        id, learning_rate * weighted_gradient_sum[id] / training_num
                    )
                    _param.sub_(_update)
                    self._apply_weight_decay(_param, _trainable, _wd_lam)
                    if _audit:
                        _d = float(_update.pow(2).sum())
                        _w = float(_param.pow(2).sum())
                        _delta_sq += _d
                        _weight_sq += _w
                        if _trainable:  # L3: the slice that can actually diverge
                            _tr_delta_sq += _d
                            _tr_weight_sq += _w
        if _trust:
            _rho_t = self._rho_star_now()
            _g_sq = _t_sq = 0.0
            _dirs = {}
            for id, _p in zip(range(len(weighted_gradient_sum)),
                              self.trainer.model.parameters()):
                _pooled = weighted_gradient_sum[id] / training_num
                # Momentum BEFORE normalisation, so the trust-ratio scale still
                # pins rho at rho*. Applied after it (the raw_sgd path) heavy-ball
                # multiplies the step by 1/(1-beta) -- which is the arithmetic
                # behind the original S1 NaN, and would confound any momentum A/B.
                _dirs[id] = self._server_update_step(id, _pooled)
                if not _p.requires_grad:
                    continue
                _cos_accumulate(id, _dirs[id])
                _g_sq += float(_dirs[id].pow(2).sum())
                _t_sq += float(_p.detach().pow(2).sum())
            _gn, _tn = _g_sq ** 0.5, _t_sq ** 0.5
            # ||G||=0 means an empty/degenerate pool: skip rather than divide.
            _scale = (_rho_t * _tn / _gn) if _gn > 0 else 0.0
            logger.info(
                f"[ServerStep] trust_ratio commit={self._commit_count} rho*={_rho_t:.6g} "
                f"||G||={_gn:.6g} ||theta_tr||={_tn:.6g} scale={_scale:.6g}"
            )
            for id, _param_src in zip(range(len(weighted_gradient_sum)),
                                      self.trainer.model.parameters()):
                with _agg_sync_timer(self, "agg_apply_update_cpu_sync"):
                    _trainable = bool(_param_src.requires_grad)
                    _param = _param_src.detach().to("cpu")
                    # Frozen tensors probe as zeros, so their pooled sum is already
                    # zero -- no branch needed, same as raw_sgd.
                    _update = _scale * _dirs[id]
                    _param.sub_(_update)
                    self._apply_weight_decay(_param, _trainable, _wd_lam)
                if _audit:
                    _d = float(_update.pow(2).sum())
                    _w = float(_param.pow(2).sum())
                    _delta_sq += _d
                    _weight_sq += _w
                    if _trainable:  # L3: the slice that can actually diverge
                        _tr_delta_sq += _d
                        _tr_weight_sq += _w
        self._commit_count += 1
        # S-C sensor: the step just taken sizes the next commit's pool.
        if _trust:
            self._last_rho = _rho_t
        elif _audit and _tr_weight_sq > 0:
            self._last_rho = (_tr_delta_sq / _tr_weight_sq) ** 0.5
        # C-1: bank this step. Counts EVERY step, unlike replay_scoring.py's
        # t = 0..T-2 (which brackets Phi_obs between two norms) -- one commit in
        # ~900, and the conservative direction for a controller.
        if self._last_rho:
            self._B += 0.5 * math.log1p(self._last_rho ** 2)
        # 3.1 fires BEFORE the stop is tested, so a re-sense that lowers B_max
        # can stop the run on the same commit it lands rather than one later.
        if (self._b_max_probe_every
                and self._commit_count % self._b_max_probe_every == 0):
            self._resense_b_max()
        self._check_budget_stop()
        _cos_gt = None
        if _cos_probe is not None and _cos_g_sq > 0 and _cos_probe[1] > 0:
            _cos_gt = (_cos_dot / (_cos_g_sq ** 0.5 * _cos_probe[1]),
                       _cos_g_sq ** 0.5, _cos_probe[1])
            logger.info(
                f"[CosProbe] commit={self._commit_count} cos={_cos_gt[0]:.6g} "
                f"||G||={_cos_gt[1]:.6g} ||g||={_cos_gt[2]:.6g}"
            )
        if _audit:
            self._emit_server_update(
                _delta_sq**0.5, _weight_sq**0.5, learning_rate,
                trainable_delta_norm=_tr_delta_sq**0.5,
                trainable_weight_norm=_tr_weight_sq**0.5,
                split=_split, cos_gt=_cos_gt,
            )

    def _check_budget_stop(self):
        """C-1's stop (buildplan §5). One predicate, three reasons, one exit.

        `Phi = e^B` and `B` is monotone, so "Phi crossed a threshold" and "the
        budget is spent" are the SAME test up to a log -- with a sensed B_max
        they are literally the same number, since B_max = ln Phi_peak. The fixed
        Phi threshold survives only for an arm with no sensed B_max.

        LATCHED, not level-triggered: B cannot fall, but 3.1 can re-sense B_max
        downward and move the threshold under it, so the predicate can flip.

        `halt` routes through `_work_done` -- the same exit max_runtime_s and
        [SIM_WALL_CEILING] already take. Freezing theta and continuing to
        evaluate is dominated, not a trade-off: a frozen model's accuracy is
        fixed, so further evals cost GPU and return eval noise (D6).
        """
        if self._phi_stop == "off" or self._stop_fired:
            return
        if self._rho_schedule == "landing":
            if self._B < self._budget_stop_frac * self._b_max:
                return
            reason = "budget"
        else:
            if math.exp(self._B) < self._phi_stop_threshold:
                return
            reason = "phi_fixed"
        self._stop_fired = reason
        logger.warning(
            f"[BudgetStop] reason={reason} action={self._phi_stop} "
            f"commit={self._commit_count} B={self._B:.6g} "
            f"B_max={self._b_max:.6g} Phi={math.exp(self._B):.4g}"
        )
        if self._phi_stop == "halt":
            self._work_done = True

    def _wd_lambda(self, rho_hint):
        """Q2: the decay coefficient for this commit, or None if disabled.

        `auto` = rho^2/2, the value that exactly cancels Leg 1's inflation
        (||theta||^2 grows by 1+rho^2 per commit; (1-rho^2/2)^2 ~= 1-rho^2).
        """
        wd = getattr(self, "_weight_decay", None)
        if wd is None:
            return None
        if wd != "auto":
            return wd
        rho = rho_hint if rho_hint else (getattr(self, "_last_rho", 0.0) or 0.0)
        return 0.5 * float(rho) ** 2 if rho else None

    def _apply_weight_decay(self, param, trainable, lam):
        """Shrink the TRAINABLE slice in place, after the step. Frozen params are
        left alone: decaying them would change the backbone, not the budget."""
        if lam and trainable:
            param.mul_(1.0 - lam)

    def _rho_star_now(self):
        """S-B: the relative step this commit is allowed to take.

        `const` holds the setpoint -- which isolates S-A but still grows
        ||theta|| geometrically as (1+rho*^2)^(T/2). `rm` anneals as
        rho*_0 * t^-rho_exp; rho_exp must exceed 0.5, since sum (1/sqrt(t))^2
        diverges logarithmically and merely defers the blow-up (handoff §15.6).
        """
        t = max(1, self._commit_count + 1)
        if self._rho_schedule == "landing":
            # Law C. B_rem clamps at 0, so a B_max re-sensed below the spend
            # gives rho* = 0 -- the correct stop, not a sqrt of a negative
            # (edge case f). T_res being constant kills edge case (a) too.
            return rho_star_now(self._b_max, self._B, self._t_res, self._rho_max)
        if self._rho_schedule == "rm":
            return self._rho_star * (t ** -self._rho_exp)
        return self._rho_star

    def _pool_split_half_stats(self, model_list):
        """L1 audit: raw components of the committed pool's SPLIT-HALF COSINE.

        Sum the uploads into even/odd halves (interleaved -- arrival order tracks
        trainer speed and staleness) and return `(<a,b>, ||a||, ||b||, pool_size)`.
        The halves share one true-gradient component and carry independent probe
        noise, so their agreement measures pooling adequacy without needing the
        true gradient: S-E's gate statistic and S-C's setpoint.

        Raw components, never a per-commit ratio: at p~1e6 one commit's cosine is
        under the 1/sqrt(p) ~ 1e-3 sampling floor, so only sum(dot)/sum(|a||b|)
        pooled over ~100 commits is meaningful.

        Returns None when disabled or the pool is too small to split.
        """
        if not getattr(self, "_pool_split_half_audit", False):
            return None
        try:
            n = len(model_list)
            if n < 2:
                return None
            dot = a_sq = b_sq = 0.0
            for id in range(len(model_list[0][1])):
                sum_a = sum_b = None
                for i in range(n):
                    t = model_list[i][1][id]
                    if i % 2 == 0:
                        sum_a = t.clone() if sum_a is None else sum_a + t
                    else:
                        sum_b = t.clone() if sum_b is None else sum_b + t
                if sum_a is None or sum_b is None:
                    continue
                # float64 accumulation: the per-param dots are tiny and many.
                dot += float((sum_a.double() * sum_b.double()).sum())
                a_sq += float(sum_a.double().pow(2).sum())
                b_sq += float(sum_b.double().pow(2).sum())
            return (dot, a_sq**0.5, b_sq**0.5, n)
        except Exception:  # pragma: no cover - audit must never fault training
            logger.debug("pool split-half audit failed", exc_info=True)
            return None

    @torch.no_grad()
    def _probe_accuracy(self, x, labels, chunk=256):
        """Held-out accuracy at the current weights. Forward only, eval mode."""
        model = self.trainer.model
        ok = 0
        for s in range(0, x.shape[0], chunk):
            out = model(x[s:s + chunk])
            logits = (out.logits if hasattr(out, "logits")
                      else out[0] if isinstance(out, (tuple, list)) else out)
            ok += int((logits.argmax(-1) == labels[s:s + chunk].view(-1)).sum())
        return ok / max(1, x.shape[0])

    def _resense_b_max(self):
        """3.1: re-sense `B_max` by INJECTING inflation instead of waiting for it.

        Isotropic Gaussian noise on the trainable slice, scaled so `||theta_tr||`
        grows by `Phi`; the knee of chance-normalized accuracy is `Phi_peak` and
        `B_max = ln Phi_peak` (model §5.5b, `expts/bmax_probe.py`). Forward passes
        only -- no gradients, no training.

        B-1 (2026-08-13) made this mandatory rather than optional: knees are
        neither invariant nor monotone in `num_labels` (agnews ~3.0-3.5 vs
        yahoo/yelp-p ~2.0-2.3), so a fixed or `num_labels`-derived constant
        misprices the budget on at least two of three datasets tested.

        `T_res` is NOT touched -- it is a rate, not run state, so there is no
        horizon for a re-sense to reset (buildplan §5, D5). Only `_b_max` moves,
        and 3.2 picks it up on the very next commit with no other bookkeeping.

        The weights are perturbed IN PLACE against a cloned baseline and restored
        in a `finally`. That clone is the "copy" the spec asks for; cloning the
        whole model instead would double resident memory for no added safety,
        since anything that could skip the restore also ends the run.
        """
        params = [p for p in self.trainer.model.parameters() if p.requires_grad]
        if not params:
            return
        model = self.trainer.model
        was_training = model.training
        base = [p.detach().clone() for p in params]
        t0 = time.time()
        try:
            x, labels = self._reference_batch(self._b_max_probe_n)
            model.eval()
            base_acc = self._probe_accuracy(x, labels)
            n0 = math.sqrt(sum(float(p.detach().pow(2).sum()) for p in params))
            gen = torch.Generator().manual_seed(_COS_PROBE_SEED + self._commit_count)
            accs = []
            for phi in self._b_max_phis:
                with torch.no_grad():
                    for p, b in zip(params, base):
                        p.copy_(b)
                    eps = [torch.randn(p.shape, generator=gen) for p in params]
                    en = math.sqrt(sum(float(e.pow(2).sum()) for e in eps))
                    target = n0 * noise_scale(phi)
                    for p, e in zip(params, eps):
                        p.add_(e.to(p.device, p.dtype) * (target / en))
                accs.append(self._probe_accuracy(x, labels))
        except Exception:  # pragma: no cover - the probe must never fault training
            logger.warning("[BmaxProbe] failed; keeping current B_max",
                           exc_info=True)
            return
        finally:
            with torch.no_grad():
                for p, b in zip(params, base):
                    p.copy_(b)
            if was_training:
                model.train()

        phi_knee = knee(self._b_max_phis, accs, base_acc, self.num_labels)
        new = b_max_from_knee(phi_knee)
        curve = " ".join(f"{p:g}:{a:.3f}" for p, a in zip(self._b_max_phis, accs))
        if new is None:
            # A head at chance has no readable curve -- sizing a budget off it
            # would be worse than keeping a prior that is merely conservative.
            logger.warning(
                f"[BmaxProbe] commit={self._commit_count} base_acc={base_acc:.3f} "
                f"too close to chance {1.0 / self.num_labels:.3f}; keeping "
                f"B_max={self._b_max:.6g}  curve[{curve}]"
            )
            return
        old = self._b_max
        self._b_max = new
        logger.info(
            f"[BmaxProbe] commit={self._commit_count} B_max {old:.6g} -> {new:.6g} "
            f"(Phi_knee={phi_knee:.3f}) base_acc={base_acc:.3f} B={self._B:.6g} "
            f"rho*={self._rho_star_now():.6g} took={time.time() - t0:.1f}s  "
            f"curve[{curve}]"
        )

    def _reference_batch(self, n_want):
        """The fixed held-out batch both server-side probes read (B17).

        Cached once and shared: `_cos_probe_gradient` needs a gradient on it and
        3.1's `B_max` probe needs accuracy on it, and they MUST be the same
        draw -- otherwise the two instruments disagree for a reason that has
        nothing to do with the model.

        NEVER slice `[:n]`: `test_index_list` is per-client shards concatenated
        in client order, never shuffled (`base_data_manager.py:204-216`), so the
        head is ONE client's skewed shard -- 75% single-class, and its gradient
        came out ANTI-correlated (-0.46) with the true one. A fixed-seed
        permutation keeps the batch identical across commits and across arms.
        """
        if self._cos_probe_batch is not None:
            return self._cos_probe_batch
        device = next(self.trainer.model.parameters()).device
        tensors = self.test_global.dataset.tensors
        total = tensors[0].shape[0]
        n = min(n_want, total)
        idx = torch.randperm(
            total, generator=torch.Generator().manual_seed(_COS_PROBE_SEED)
        )[:n]
        idx, _ = torch.sort(idx)  # locality; order is irrelevant to either probe
        self._cos_probe_batch = (
            tensors[1][idx].to(device),   # input_ids (eval_model's layout)
            tensors[4][idx].to(device),   # labels
        )
        _lab = tensors[4][idx].view(-1).tolist()
        _share = max(_lab.count(c) for c in set(_lab)) / max(len(_lab), 1)
        logger.info(
            f"[RefBatch] shuffled held-out batch of {n} cached "
            f"(seed={_COS_PROBE_SEED}, dominant-class share={_share:.2f})"
        )
        # threshold is 1/K + margin, not a fixed 0.5: balanced IS 0.50 on a
        # 2-class task and 0.10 on a 10-class one.
        _skew_max = max_dominant_share(self.num_labels)
        if _share > _skew_max:
            logger.warning(
                f"[RefBatch] reference batch is still class-skewed at "
                f"{_share:.2f} (>{_skew_max:.2f} for {self.num_labels} classes); "
                f"raise cos_probe_batch_size"
            )
        return self._cos_probe_batch

    def _cos_probe_gradient(self):
        """B1 (§15.1): a REAL gradient at the current theta, for one fixed batch.

        `G` is already server-side, so the only missing half of cos(G,g) is some
        `g`. A backward pass on a fixed held-out batch is one: biased toward that
        batch, but the SAME batch every commit, so trend and scale are comparable
        across commits and arms. fp32, no autocast -- a reference, not a step.

        Returns `([g_i or None per model param], ||g||)` on CPU, or None.
        """
        try:
            model = self.trainer.model
            x, labels = self._reference_batch(self._cos_probe_batch_size)
            was_training = model.training
            model.eval()
            model.zero_grad(set_to_none=True)
            output = model(x)
            if hasattr(output, "logits"):
                logits = output.logits
            elif isinstance(output, (tuple, list)):
                logits = output[0]
            else:
                logits = output
            loss = CrossEntropyLoss()(
                logits.view(-1, self.num_labels), labels.view(-1)
            )
            loss.backward()
            grads, g_sq = [], 0.0
            for p in model.parameters():
                if not p.requires_grad or p.grad is None:
                    grads.append(None)
                    continue
                g = p.grad.detach().to("cpu", torch.float32).clone()
                grads.append(g)
                g_sq += float(g.pow(2).sum())
            model.zero_grad(set_to_none=True)
            if was_training:
                model.train()
            # A dead probe emits nothing, which at scoring time is indistinguishable
            # from "audit off". Say so once.
            if g_sq <= 0 and not getattr(self, "_cos_probe_warned", False):
                self._cos_probe_warned = True
                logger.warning(
                    "[CosProbe] backward produced no trainable gradient -- B1 is "
                    "emitting nothing. Check requires_grad on the aggregator model."
                )
            return grads, g_sq ** 0.5
        except Exception:  # pragma: no cover - audit must never fault training
            logger.debug("cos ground-truth probe failed", exc_info=True)
            return None

    def _emit_server_update(self, delta_norm, weight_norm, learning_rate,
                            trainable_delta_norm=None, trainable_weight_norm=None,
                            split=None, cos_gt=None):
        """One `server_update` record per commit (I-1). Never faults training."""
        try:
            from flame import telemetry
            if telemetry.is_enabled():
                from flame.telemetry.events import build_server_update
                stage = getattr(self, "fwd_llm_stage", None)
                _dot, _na, _nb, _psize = split if split else (None, None, None, None)
                _cos, _gn, _pgn = cos_gt if cos_gt else (None, None, None)
                ev, fields = build_server_update(
                    round_num=getattr(stage, "round_id", None),
                    data_id=getattr(stage, "data_id", None),
                    iteration=getattr(stage, "iteration", None),
                    model_version=getattr(self, "_model_version", None),
                    update_delta_norm=delta_norm,
                    weight_norm=weight_norm,
                    learning_rate=learning_rate,
                    trainable_delta_norm=trainable_delta_norm,
                    trainable_weight_norm=trainable_weight_norm,
                    pool_size=_psize,
                    split_half_dot=_dot,
                    split_half_norm_a=_na,
                    split_half_norm_b=_nb,
                    var_at_commit=getattr(self, "_var_scalar", None),
                    n_eff=getattr(self, "_n_eff_scalar", None),
                    cos_ground_truth=_cos, pooled_norm=_gn, probe_grad_norm=_pgn,
                    budget_b=self._B, budget_b_max=self._b_max,
                    rho_star=self._rho_star_now(), n_req=self._n_required(),
                    stop_reason=self._stop_fired,
                )
                telemetry.emit(ev, **fields)
        except Exception:  # pragma: no cover - telemetry must never fault training
            logger.debug("server_update telemetry emit failed", exc_info=True)

    @timer_decorator
    def _prepare_round_state(self, current_round):
        """Timed separately; shared preamble (var bookkeeping, plateau check, accumulation)
        that runs before the commit/rollback branch split."""
        # self.var drives the live commit gate; snr/real-var/grad-snr/cv are
        # diagnostics with no live consumer (snr gate is commented out) -> DEBUG only.
        self.var = self._compute_var()
        # Cached scalar avoids re-syncing the GPU tensor on every log print;
        # `.item()` is timed separately as the sync point.
        with _agg_sync_timer(self, "agg_var_item_sync"):
            self._var_scalar = self.var.item()
        self.var_prev_iter_list.append(self._var_scalar)
        logger.info(f"self.var = {self._var_scalar}")
        # S-K sensor: rides on server_update_audit since it lands in that record --
        # and is mandatory under commit_gate=n_target, which reads it.
        if getattr(self, "_server_update_audit", False) or self._commit_gate == "n_target":
            self._n_eff_scalar = self._compute_n_eff(self._var_scalar)
            logger.info(
                f"[n_eff] n_eff={self._n_eff_scalar} pool={len(self.grad_for_var_check_list)} "
                f"var={self._var_scalar}"
            )
        if logger.isEnabledFor(logging.DEBUG):
            var_jvp = calculate_real_var(self.jvp_for_snr_check_list)
            self.snr = calculate_snr(self.jvp_for_snr_check_list)
            grads_snr = calculate_snr_gradients(self.grad_for_var_check_list)
            self.snr_prev_iter_list.append(self.snr)
            c_of_variation = calculate_cv(self.grad_for_var_check_list)
            logger.debug(f"snr of jvps = {self.snr}")
            logger.debug(f"coefficient of variation = {c_of_variation}")

        # Opt-2 (charter §5c/§5e): variance-plateau force-commit. Under the
        # 'plateau' policy, additionally force a commit once the per-bin variance
        # curve has flattened (relative drop over the last N cycles < rel_delta)
        # while var is still above threshold -- more denoising buys nothing, so
        # commit the denoised estimate. var_prev_iter_list is the per-bin var
        # history (reset on commit), already including this cycle's var. Policy
        # off/absent => never arms a commit => byte-identical.
        self._force_commit_reason = None
        self._plateau_fired_this_cycle = self._should_force_commit_on_plateau()
        if self._plateau_fired_this_cycle:
            self._force_commit_this_cycle = True
            logger.info(
                f"[VarPlateau] curve flattened over N="
                f"{getattr(self, '_var_plateau_patience', 3)} "
                f"(var={self.var_prev_iter_list[-1]:.4f} > "
                f"thr={self.var_threshold}); force-committing at plateau."
            )
        logger.debug(
            f"self.grad_for_var_check_list size: {len(self.grad_for_var_check_list)}"
        )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"self.grad_for_var_check_list hashes: {[(_calculate_hash(p), p.shape) for p in self.grad_for_var_check_list]}"
            )

        model_list = []
        training_num = 0
        # self.warmup_rounds = 0
        if current_round < self.warmup_rounds:
            ratio = float(current_round + 1) / float(max(1, self.warmup_rounds))
        else:
            ratio = max(
                0.0,
                float(self.args.comm_round - current_round)
                / float(max(1, self.args.comm_round - self.warmup_rounds)),
            )
        learning_rate = self.args.learning_rate * ratio
        # learning_rate =0.01 # Current learning rate is 0.0099..
        logger.info(f"learning rate: {learning_rate}")

        # Will use 0th grads from model_dict since worker_num = 1
        for idx in range(self.worker_num):
            model_list.append((self.sample_num_dict[idx], self.model_dict[idx]))
            training_num += self.sample_num_dict[idx]
            logger.info(
                f"Model dict length (should be same as total layers in the model) : {len(self.model_dict[idx])}"
            )
        return model_list, training_num, learning_rate

    @timer_decorator
    def aggregate(self, current_round):
        start_time = time.time()
        model_list, training_num, learning_rate = self._prepare_round_state(current_round)

        # logger.info(f"len(model_list): {len(model_list)}")

        # Snapshot model_dict (mutated below) for cache_v reuse.
        if self.args.var_control:
            model_dict_cached = self._snapshot_retry_cache()

            # cached_v:  (num, params)
            logger.info(f"len of cached v: {len(self.cached_v)}")
            training_num = self._accumulate_retry_cache(model_list, training_num)
            logger.info(f"training_num : {training_num}")

        logger.info("len of self.model_dict[idx] = " + str(len(self.model_dict)))
        logger.info(
            f"length of model list : {len(model_list)} - (should be same as # of iterations in the mini-batch completed so far)"
        )
        
        if self.args.var_control:
            _force_commit = getattr(self, "_force_commit_this_cycle", False)
            if self._gate_satisfied():
            # Use different stopping conditions if necessary
            # if self.var_within_epsilon(): 
            # if self.snr_within_epsilon_and_var_under(var_jvp):
                self.var_prev_iter_list = []
                self.snr_prev_iter_list = []
                
                # old_param = self.get_global_model_params()
                old_param = self.trainer.model.parameters()
                if training_num == 0:
                    logger.warning("Not updating the model, division by 0 error")
                    return old_param
                # If weighted_aggregation_enabled is False, then the weight of each gradient in this sum is 1. Else, the weight the is determined by calling self.optimizer.weight_factor()
                (_, weighted_gradient_sum) = model_list[0]
                if logger.isEnabledFor(logging.DEBUG):
                    format_hash = lambda d: [_calculate_hash(v)[:8] for v in d]
                    logger.debug(
                        f"model_list[0] - length : {len(weighted_gradient_sum)} (should be same as grad pool):  {format_hash(weighted_gradient_sum)}"
                    )
                logger.info(f"Length of model_list : {len(model_list)}")
                self._apply_weighted_update(model_list, weighted_gradient_sum, old_param,
                                             learning_rate, training_num)
                if logger.isEnabledFor(logging.DEBUG):
                    format_hash = lambda d: [_calculate_hash(v)[:8] for v in d]
                    logger.debug(
                        f"weighted_gradient_sum - length : {len(weighted_gradient_sum)} (should be same as grad pool):  {format_hash(weighted_gradient_sum)}"
                    )
                self.last_round_update = self._snapshot_last_round_update(weighted_gradient_sum)
                logger.info(  # cached float, avoids extra GPU sync
                    f"[Variance=GOOD] var={self._var_scalar} <= thr={self.var_threshold}; "
                    f"keeping weight update, clearing cached_v."
                )
                self.var_good_enough = True
                self._force_commit_reason = "natural"
                # 方差满足要求
                self.cached_v = []
            elif _force_commit:
                # max_iter_per_data_id cap hit; skip rollback to advance model despite failed variance.
                self.var_prev_iter_list = []
                self.snr_prev_iter_list = []
                
                # old_param = self.get_global_model_params()
                old_param = self.trainer.model.parameters()
                if training_num == 0:
                    logger.warning("Not updating the model, division by 0 error")
                    return old_param
                # If weighted_aggregation_enabled is False, then the weight of each gradient in this sum is 1. Else, the weight the is determined by calling self.optimizer.weight_factor()
                (_, weighted_gradient_sum) = model_list[0]
                if logger.isEnabledFor(logging.DEBUG):
                    format_hash = lambda d: [_calculate_hash(v)[:8] for v in d]
                    logger.debug(
                        f"model_list[0] - length : {len(weighted_gradient_sum)} (should be same as grad pool):  {format_hash(weighted_gradient_sum)}"
                    )
                logger.info(f"Length of model_list : {len(model_list)}")
                self._apply_weighted_update(model_list, weighted_gradient_sum, old_param,
                                             learning_rate, training_num)
                if logger.isEnabledFor(logging.DEBUG):
                    format_hash = lambda d: [_calculate_hash(v)[:8] for v in d]
                    logger.debug(
                        f"weighted_gradient_sum - length : {len(weighted_gradient_sum)} (should be same as grad pool):  {format_hash(weighted_gradient_sum)}"
                    )

                self.last_round_update = self._snapshot_last_round_update(weighted_gradient_sum)
                self._force_commit_reason = (
                    "plateau" if getattr(self, "_plateau_fired_this_cycle", False)
                    else "cap"
                )
                logger.info(  # cached float, avoids extra GPU sync
                    f"[MaxIterBypass] Variance FAILED (var={self._var_scalar} > "
                    f"thr={self.var_threshold}) but force-commit is set "
                    f"(reason={self._force_commit_reason}); "
                    f"committing weights anyway, clearing cached_v."
                )
                self.var_good_enough = True
                self.cached_v = []
            else:
                self.var_good_enough = False
                logger.info(  # cached float, avoids extra GPU sync
                    f"[Variance=BAD] var={self._var_scalar} > thr={self.var_threshold}; "
                    f"rolling back weights, caching grads for next iteration."
                )
                # 当前模型不行，v不够，暂存起来，后面再计算更多的v
                self._cache_grad_for_retry(model_dict_cached)

        self._force_commit_this_cycle = False

        old_param = self.get_global_model_params()

        end_time = time.time()
        logger.info("aggregate time cost: %d" % (end_time - start_time))
        return old_param

    def client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [
                client_index for client_index in range(client_num_in_total)
            ]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(
                round_idx
            )  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(
                range(client_num_in_total), num_clients, replace=False
            )

        index_list = [[] for _ in range(self.args.worker_num)]

        for i in range(len(client_indexes)):
            index_list[i % self.args.worker_num].append(client_indexes[i])

        client_indexes = index_list

        logger.info("client_indexes = %s" % str(client_indexes))
        return client_indexes

    def _generate_validation_set(self, num_samples=10000):
        if self.args.dataset.startswith("stackoverflow"):
            test_data_num = len(self.test_global.dataset)
            sample_indices = random.sample(
                range(test_data_num), min(num_samples, test_data_num)
            )
            subset = torch.utils.data.Subset(self.test_global.dataset, sample_indices)
            sample_testset = torch.utils.data.DataLoader(
                subset, batch_size=self.args.batch_size
            )
            return sample_testset
        else:
            return self.test_global

    def test_on_server_for_all_clients(self, round_idx):
        if (
            round_idx % self.args.frequency_of_the_test == 0
            or round_idx == self.args.comm_round - 1
        ):
            if self.trainer.test_on_the_server(
                self.train_data_local_dict,
                self.test_data_local_dict,
                self.device,
                self.args,
            ):
                return

        if (
            round_idx % self.args.frequency_of_the_test == 0
            or round_idx == self.args.comm_round - 1
        ):
            logger.info(
                "################test_on_server_for_all_clients : {}".format(round_idx)
            )
            train_num_samples = []
            train_tot_corrects = []
            train_losses = []
            for client_idx in range(self.args.client_num_in_total):
                # train data
                metrics = self.trainer.test(
                    self.train_data_local_dict[client_idx], self.device, self.args
                )
                train_tot_correct, train_num_sample, train_loss = (
                    metrics["test_correct"],
                    metrics["test_total"],
                    metrics["test_loss"],
                )
                train_tot_corrects.append(copy.deepcopy(train_tot_correct))
                train_num_samples.append(copy.deepcopy(train_num_sample))
                train_losses.append(copy.deepcopy(train_loss))

                """
                Note: CI environment is CPU-based computing. The training speed
                for RNN training is to slow in this setting, so we only test a
                client to make sure there is no programming error.
                """
                if self.args.ci == 1:
                    break

            # test on training dataset
            train_acc = sum(train_tot_corrects) / sum(train_num_samples)
            train_loss = sum(train_losses) / sum(train_num_samples)
            # wandb.log({"Train/Acc": train_acc, "round": round_idx})
            # wandb.log({"Train/Loss": train_loss, "round": round_idx})
            stats = {"training_acc": train_acc, "training_loss": train_loss}
            logger.info(stats)

            # test data
            test_num_samples = []
            test_tot_corrects = []
            test_losses = []

            if round_idx == self.args.comm_round - 1:
                metrics = self.trainer.test(self.test_global, self.device, self.args)
            else:
                metrics = self.trainer.test(self.val_global, self.device, self.args)

            test_tot_correct, test_num_sample, test_loss = (
                metrics["test_correct"],
                metrics["test_total"],
                metrics["test_loss"],
            )
            test_tot_corrects.append(copy.deepcopy(test_tot_correct))
            test_num_samples.append(copy.deepcopy(test_num_sample))
            test_losses.append(copy.deepcopy(test_loss))

            # test on test dataset
            test_acc = sum(test_tot_corrects) / sum(test_num_samples)
            test_loss = sum(test_losses) / sum(test_num_samples)
            stats = {"test_acc": test_acc, "test_loss": test_loss}
            logger.info(stats)

    def load_data(self) -> None:
        pass

    def train(self) -> None:
        pass

    def evaluate(self) -> None:
        pass

    def check_and_sleep(self) -> None:
        pass

    def initialize(self):
        """Initialize role."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # logger.info(f"model for agg is not None: {self.model}")
        self.model.to(self.device)
