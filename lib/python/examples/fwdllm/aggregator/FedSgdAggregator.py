import contextlib
import copy
import logging
import random
import time
import math
import numpy as np
import torch
from flame.mode.horizontal.syncfl.fwdllm_aggregator import TopAggregator
from examples.fwdllm.trainer.forward_training.fwdgrad_utils import calculate_var, calculate_snr, calculate_cv, calculate_real_var, calculate_snr_gradients
from flame.monitor.runtime import timer_decorator, FwdLLMStage

logger = logging.getLogger(__name__)
import functorch as fc

import hashlib


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
                f"probe_combine={_pc} P={_P} G_rule={self._g_rule}"
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

    def _n_required(self):
        """S-C: pool this commit's step needs, N_req = p*(rho_t/s)^2 / G_rule.

        `rho_t` is exact under trust_ratio (it IS the setpoint), else the last
        realised rho, which lags a commit. None until one exists -> the cap fires.
        """
        rho = (self._rho_star_now() if self._server_step_rule == "trust_ratio"
               else self._last_rho)
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
            f"rho_t={self._rho_star_now() if self._server_step_rule == 'trust_ratio' else self._last_rho} "
            f"-> {'COMMIT' if ok else 'POOL'}"
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
        # Pass 1 pools the uploads. Under raw_sgd this stays fused with the write
        # below (byte-identical); trust_ratio needs ||G|| and ||theta_tr|| over the
        # WHOLE trainable slice before any tensor is touched, so it is split out.
        _trust = self._server_step_rule == "trust_ratio"
        for id, k in enumerate(weighted_gradient_sum):
            for i in range(0, len(model_list)):
                local_sample_number, local_model_params = model_list[i]
                # w = local_sample_number / training_num
                if i == 0:
                    weighted_gradient_sum[id] = local_model_params[id]
                else:
                    weighted_gradient_sum[id] += local_model_params[id]
            if not _trust:
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
            for id, _p in zip(range(len(weighted_gradient_sum)),
                              self.trainer.model.parameters()):
                if not _p.requires_grad:
                    continue
                _g_sq += float((weighted_gradient_sum[id] / training_num).pow(2).sum())
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
                    _update = self._server_update_step(
                        id, (_scale / training_num) * weighted_gradient_sum[id]
                    )
                    _param.sub_(_update)
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
        if _audit:
            self._emit_server_update(
                _delta_sq**0.5, _weight_sq**0.5, learning_rate,
                trainable_delta_norm=_tr_delta_sq**0.5,
                trainable_weight_norm=_tr_weight_sq**0.5,
                split=_split,
            )

    def _rho_star_now(self):
        """S-B: the relative step this commit is allowed to take.

        `const` holds the setpoint -- which isolates S-A but still grows
        ||theta|| geometrically as (1+rho*^2)^(T/2). `rm` anneals as
        rho*_0 * t^-rho_exp; rho_exp must exceed 0.5, since sum (1/sqrt(t))^2
        diverges logarithmically and merely defers the blow-up (handoff §15.6).
        """
        t = max(1, self._commit_count + 1)
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

    def _emit_server_update(self, delta_norm, weight_norm, learning_rate,
                            trainable_delta_norm=None, trainable_weight_norm=None,
                            split=None):
        """One `server_update` record per commit (I-1). Never faults training."""
        try:
            from flame import telemetry
            if telemetry.is_enabled():
                from flame.telemetry.events import build_server_update
                stage = getattr(self, "fwd_llm_stage", None)
                _dot, _na, _nb, _psize = split if split else (None, None, None, None)
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
