from typing import Any, Dict, List, Optional
import torch

class AggregatorToTrainerDTO:
    def __init__(
        self,
        weights: Dict[str, torch.Tensor],
        grad_pool: Optional[List[torch.Tensor]],
        round: int,
        model_version: int,
        task_to_perform: str,
        data_id: int,
        iteration_per_data_id: int,
    ):
        self.weights = weights
        self.grad_pool = grad_pool
        self.round = round
        self.model_version = model_version          # usually same as round
        self.task_to_perform = task_to_perform      # train or eval. eval is not sent now
        self.data_id = data_id
        self.iteration_per_data_id = iteration_per_data_id

class TrainerToAggregatorDTO:
    def __init__(
        self,
        gradients: Dict[str, torch.Tensor],
        gradients_for_var_check: List[torch.Tensor],
        dataset_size: int,
        model_version: int,
        datasampler_metadata: Any,
        total_data_bins: int,
        stat_utility: float,
    ):
        self.gradients = gradients
        self.gradients_for_var_check = gradients_for_var_check
        self.dataset_size = dataset_size
        self.model_version = model_version
        self.datasampler_metadata = datasampler_metadata
        self.total_data_bins = total_data_bins
        self.stat_utility = stat_utility            # for OORT, only when task_to_perform == eval


class Variables:
    def __init__(
        self,
        agg_goal: int,                              # K trainers
        cached_v: List[Tuple[int, List[torch.Tensor]]],     # stores the current perturbations
            # Each tuple: (sample_num, model_params)
        GRAD_POOL: List[torch.Tensor],              # set to shared_grad_pool_trainable (deep clone of shared_grad_pool)
        v_buffer: List[List[torch.Tensor]],         
        v_params: ,                                 # Derived from v_buffer
        sample_num_dict: Dict[int, int],            # trainerID: number of samples used by that trainer
        model_dict: Dict[int, Any],                 # iteration of round?: Model params
        self.grad: List[torch.Tensor],                             # Same as model params. Not the same as self.gradients
        shared_grad_pool:  List[torch.Tensor] or List[List[torch.Tensor]]  # shared_grad_pool = self.aggregate_grad_pool(self.grad_pool) 
        # self.trainer.model_trainer.old_grad = full_grad
        weights: Dict[str, torch.Tensor]            # parameter name (e.g. "layer1.weight"): 
        grad_for_var_check: List[List[torch.Tensor]],?    # = calculate_jvp() * v_params
    ):
        return